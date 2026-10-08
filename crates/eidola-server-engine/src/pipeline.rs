//! From a validated request to engine tokens, and from engine tokens to OpenAI output.
//!
//! In: the chat crate renders the prompt with the model's own template (byte-exact),
//! tokenizes it, and the request becomes an engine [`Request`] with checked
//! [`SamplingParams`] and a [`CacheScope`]: the client's key turned into an engine salt
//! by the per-boot HMAC, or [`CacheScope::Private`] without one.
//!
//! Out: engine tokens go through the chat crate's streaming detokenizer, then the stop
//! sequences (applied here, to the decoded text, since the core leaves them to its
//! caller), then the reasoning and tool-call parsers ([`OutputParser::push_text`]).

use std::sync::Arc;

use eidola_engine::engine::{CacheScope, FinishReason as EngineFinish, Request, RequestId};
use eidola_engine::sampling::SamplingParams;
use eidola_engine::secret::SaltDeriver;
use eidola_engine_chat::{
    ChatDelta, ChatInput, Detokenizer, FinishReason, OutputConfig, OutputParser, PrefixedCallIds,
    RenderOptions, StopCause, ToolSchemas,
};

use crate::api::ValidRequest;
use crate::error::ApiError;
use crate::model::LoadedModel;

/// The engine request for `req`, plus what the output side needs.
pub struct Prepared {
    pub request: Request,
    pub prompt_tokens: u32,
    pub output: OutputConfig,
    pub stop: Vec<String>,
}

/// Renders, tokenizes and builds the engine request. CPU work: run it off the async
/// runtime.
pub fn prepare(
    model: &LoadedModel,
    salts: &SaltDeriver,
    max_model_len: u32,
    id: RequestId,
    req: ValidRequest,
) -> Result<Prepared, ApiError> {
    let ValidRequest {
        messages,
        tools,
        parse_tools,
        max_completion_tokens,
        temperature,
        top_p,
        stop,
        cache_key,
        ..
    } = req;
    let defaults = model.defaults();
    let sampling = SamplingParams::new(
        temperature.unwrap_or(defaults.temperature),
        defaults.top_k,
        top_p.unwrap_or(defaults.top_p),
        0.0,
        random_seed(),
    )
    .map_err(|e| ApiError::invalid(e.to_string()))?;
    let cache = match &cache_key {
        Some(key) => CacheScope::Keyed(salts.derive(key)),
        None => CacheScope::Private,
    };
    drop(cache_key);

    let schemas = if parse_tools {
        tools.as_ref().map(ToolSchemas::from_tools)
    } else {
        None
    };
    let mut input = ChatInput { messages, tools };
    input
        .normalize_tool_call_arguments()
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    let options = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: None,
    };
    let prompt = model
        .template()
        .render(&input, options)
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    let tokens = model
        .tokenizer()
        .encode(&prompt)
        .map_err(|e| ApiError::invalid(e.to_string()))?;
    let prompt_tokens = tokens.len() as u32;
    if prompt_tokens >= max_model_len {
        return Err(ApiError::ContextLengthExceeded(format!(
            "the prompt is {prompt_tokens} tokens; this model's limit is {max_model_len} \
             including at least one completion token"
        )));
    }
    let room = max_model_len - prompt_tokens;
    let max_tokens = max_completion_tokens.map_or(room, |m| m.min(room));
    Ok(Prepared {
        request: Request {
            id,
            prompt: tokens,
            sampling,
            max_tokens,
            stop_token_ids: Vec::new(),
            cache,
        },
        prompt_tokens,
        output: OutputConfig::new(options.enable_thinking, schemas),
        stop,
    })
}

fn random_seed() -> u64 {
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).expect("operating-system randomness unavailable");
    u64::from_le_bytes(b)
}

/// A per-response random hex id.
pub fn random_id(prefix: &str) -> String {
    let mut b = [0u8; 12];
    getrandom::fill(&mut b).expect("operating-system randomness unavailable");
    format!("{prefix}{}", hex::encode(b))
}

/// What one batch of engine tokens turned into.
#[derive(Debug, Default)]
pub struct Step {
    pub delta: ChatDelta,
    /// Set when the response is complete (by the engine, or by a stop sequence here).
    pub finish: Option<FinishReason>,
    /// A stop sequence matched: the caller must cancel the engine request.
    pub stopped_by_sequence: bool,
}

/// Turns one request's engine tokens into OpenAI deltas.
pub struct Decoder {
    model: Arc<LoadedModel>,
    detok: Detokenizer,
    stop: StopMatcher,
    parser: OutputParser,
    completion_tokens: u32,
    done: bool,
}

impl Decoder {
    pub fn new(
        model: Arc<LoadedModel>,
        output: OutputConfig,
        stop: Vec<String>,
        call_prefix: String,
    ) -> Self {
        Decoder {
            model,
            detok: Detokenizer::new(output.skip_special_tokens),
            stop: StopMatcher::new(stop),
            parser: OutputParser::new(output, PrefixedCallIds(call_prefix)),
            completion_tokens: 0,
            done: false,
        }
    }

    /// Completion tokens consumed so far (through the token that completed a stop
    /// sequence, if one did).
    pub fn completion_tokens(&self) -> u32 {
        self.completion_tokens
    }

    /// Feeds one engine event's tokens and finish.
    pub fn push(&mut self, tokens: &[u32], finish: Option<EngineFinish>) -> Result<Step, ApiError> {
        let mut step = Step::default();
        if self.done {
            return Ok(step);
        }
        let tokenizer = self.model.tokenizer();
        for (i, &t) in tokens.iter().enumerate() {
            self.completion_tokens += 1;
            // The EOS that ends the response is not text (it is usually special anyway).
            let last = i + 1 == tokens.len();
            if last && finish == Some(EngineFinish::Stop) && tokenizer.is_eos(t) {
                break;
            }
            let text = self
                .detok
                .push(tokenizer, t)
                .map_err(|_| ApiError::Internal("the engine produced an unknown token"))?;
            let (released, matched) = self.stop.push(&text);
            append(&mut step.delta, self.parser.push_text(&released));
            if matched {
                let (tail, reason) = self.parser.finish(StopCause::StopSequence);
                append(&mut step.delta, tail);
                step.finish = Some(reason);
                step.stopped_by_sequence = true;
                self.done = true;
                return Ok(step);
            }
        }
        if let Some(f) = finish {
            let cause = match f {
                EngineFinish::Stop => StopCause::EndOfSequence,
                EngineFinish::Length => StopCause::MaxTokens,
                EngineFinish::Cancelled => {
                    return Err(ApiError::Internal("the request was cancelled"));
                }
            };
            let rest = self.detok.finish();
            let (released, matched) = self.stop.push(&rest);
            append(&mut step.delta, self.parser.push_text(&released));
            let cause = if matched {
                StopCause::StopSequence
            } else {
                append(&mut step.delta, self.parser.push_text(&self.stop.flush()));
                cause
            };
            let (tail, reason) = self.parser.finish(cause);
            append(&mut step.delta, tail);
            step.finish = Some(reason);
            self.done = true;
        }
        Ok(step)
    }
}

fn append(into: &mut ChatDelta, d: ChatDelta) {
    into.reasoning_content.push_str(&d.reasoning_content);
    into.content.push_str(&d.content);
    into.tool_calls.extend(d.tool_calls);
}

/// Stop-sequence matching over streamed text.
///
/// Text that could still be the start of a stop sequence is held back (at most the
/// longest sequence's length minus one byte, extended to a character boundary), so a
/// sequence split across tokens is caught and nothing past a match is ever released.
/// On a match, the text before it is released and the sequence and everything after it
/// are dropped (OpenAI semantics: the stop sequence is not part of the output).
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
    matched: bool,
}

impl StopMatcher {
    pub fn new(stops: Vec<String>) -> Self {
        StopMatcher {
            stops,
            pending: String::new(),
            matched: false,
        }
    }

    /// Adds text; returns the text now safe to release and whether a stop sequence
    /// matched.
    pub fn push(&mut self, text: &str) -> (String, bool) {
        if self.matched {
            return (String::new(), true);
        }
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        self.pending.push_str(text);
        let first = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min();
        if let Some(at) = first {
            let out = self.pending[..at].to_string();
            self.pending.clear();
            self.matched = true;
            return (out, true);
        }
        let hold = self.stops.iter().map(String::len).max().unwrap_or(1) - 1;
        let mut cut = self.pending.len().saturating_sub(hold);
        while !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        let out = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (out, false)
    }

    /// Releases whatever is held back (the response ended without a match).
    pub fn flush(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], chunks: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for c in chunks {
            let (r, hit) = m.push(c);
            out.push_str(&r);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&m.flush());
        (out, false)
    }

    #[test]
    fn matches_across_chunks_and_drops_the_sequence() {
        assert_eq!(
            run(&["END"], &["abc E", "N", "Dxyz"]),
            ("abc ".into(), true)
        );
        assert_eq!(run(&["END"], &["abc EN"]), ("abc EN".into(), false));
        assert_eq!(run(&["xy", "b"], &["abxy"]), ("a".into(), true));
        assert_eq!(run(&[], &["a", "b"]), ("ab".into(), false));
    }

    #[test]
    fn holds_back_on_character_boundaries() {
        let mut m = StopMatcher::new(vec!["ééé".into()]);
        let (r, hit) = m.push("aéé");
        assert!(!hit);
        assert!(r.is_empty() || "aéé".starts_with(&r));
        let (r2, hit) = m.push("é!");
        assert!(hit);
        assert_eq!(format!("{r}{r2}"), "a");
    }
}
