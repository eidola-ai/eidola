//! The node serving Qwen3.5, end to end and in process: real HTTP, the model's pinned
//! Qwen3.5 template and tokenizer, the engine core over the Qwen CPU executor on the
//! Qwen3.5 dev model (`common::fixture`: BF16 weights, the head tied to the embedding).
//! The model's output is gibberish; these tests assert structure, the format's defaults,
//! the prefix relation a client's declared cached prefix rests on, refusals, and
//! token-level equality with the engine driven directly.

mod common;

use std::sync::Arc;

use base64::Engine as _;
use common::*;
use eidola_common::architecture::Architecture;
use eidola_common::engine_protocol::{CachedPrefix, declared_prompt};
use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
use eidola_engine::kv::CachePolicy;
use eidola_engine::sampling::SamplingParams;
use eidola_engine::spec::Bucket;
use eidola_engine_chat::{
    ChatDelta, ChatFormat, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, StopCause,
};
use eidola_engine_cpu::qwen::{QwenCpuExecutor, QwenCpuExecutorConfig};
use eidola_engine_model::qwen::{QwenModel, WeightFormat};
use eidola_server_engine::model::{GenerationDefaults, LoadedModel, Served};
use eidola_server_engine::pipeline::qwen_options;
use serde_json::{Value, json};

const LONG_TEXT: &str = "The quick brown fox jumps over the lazy dog while the five boxing \
    wizards jump quickly, and a wizard's job is to vex chumps quickly in fog. Pack my box \
    with five dozen liquor jugs; how vexingly quick daft zebras jump!";

/// The dev model's window in tokens (`common`'s `MAX_MODEL_LEN`).
const WINDOW: u64 = 1_024;

/// The node's KV block (`common`'s `KV_BLOCK_SIZE`).
const BLOCK: usize = 16;

fn reference(model: &LoadedModel) -> Arc<QwenModel> {
    match model.served() {
        Served::Qwen { model, .. } => model.clone(),
        _ => panic!("the Qwen3.5 dev model"),
    }
}

/// A greedy request for `messages`.
fn body(messages: Value, max_tokens: u32) -> Value {
    json!({
        "model": QWEN35_MODEL_ID,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    })
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

/// The prompt (text and tokens) the template renders for `messages` with the node's
/// options, rendered here (not by the node), request data never a control token.
fn prompt(messages: &Value) -> (String, Vec<u32>) {
    let model = qwen35_loaded();
    let Served::Qwen { template, .. } = model.served() else {
        panic!("the Qwen3.5 dev model")
    };
    let input = ChatInput::from_json(&messages.to_string(), None).unwrap();
    let prompt = template.render_prompt(&input, &qwen_options()).unwrap();
    let ids = model.tokenizer().encode_prompt(&prompt).unwrap();
    (prompt.into_text(), ids)
}

/// Greedy generation through the engine core and the CPU executor, without speculation.
fn direct_greedy(prompt: Vec<u32>, max_tokens: u32) -> (Vec<u32>, FinishReason) {
    let model = qwen35_loaded();
    let exec = QwenCpuExecutor::new(
        reference(&model),
        QwenCpuExecutorConfig {
            block_size: 16,
            num_blocks: 256,
            host_blocks: 0,
            state_buffers: 21,
            host_state_buffers: 0,
            num_state_slots: 4,
            max_model_len: 1024,
            buckets: vec![Bucket {
                max_seqs: 4,
                max_tokens: 256,
            }],
            draft_tokens: 0,
            sampleable_vocab_size: model.sampleable_vocab_size(),
            record: false,
        },
    );
    let mut engine = Engine::new(
        exec,
        SchedulerConfig {
            max_batched_tokens: 256,
            max_seqs: 4,
            max_prefill_chunk: 256,
            eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
            speculative: false,
            cache: CachePolicy::default(),
            sweep_interval_ms: 1000,
        },
    )
    .unwrap();
    engine
        .submit(Request {
            id: 1,
            prompt,
            sampling: SamplingParams::greedy(),
            max_tokens,
            required_cached_tokens: 0,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
        })
        .unwrap();
    let mut out = Vec::new();
    for now in 0.. {
        for ev in engine.step(now).unwrap() {
            out.extend(ev.tokens);
            if let Some(f) = ev.finish {
                return (out, f);
            }
        }
    }
    unreachable!()
}

/// What the format's output parser makes of `tokens`, thinking off (the node's default).
fn expected_output(tokens: &[u32], finish: FinishReason) -> (ChatDelta, &'static str) {
    let model = qwen35_loaded();
    let mut parser = OutputParser::new(
        OutputConfig::qwen3_5(None, None),
        PrefixedCallIds("c".into()),
    );
    let mut all = ChatDelta::default();
    let mut text_tokens = tokens;
    if finish == FinishReason::Stop {
        assert!(model.tokenizer().is_eos(*tokens.last().unwrap()));
        text_tokens = &tokens[..tokens.len() - 1];
    }
    for &t in text_tokens {
        let d = parser.push_token(model.tokenizer(), t).unwrap();
        all.reasoning_content.push_str(&d.reasoning_content);
        all.content.push_str(&d.content);
    }
    let cause = match finish {
        FinishReason::Stop => StopCause::EndOfSequence,
        _ => StopCause::MaxTokens,
    };
    let (d, reason) = parser.finish(cause);
    all.reasoning_content.push_str(&d.reasoning_content);
    all.content.push_str(&d.content);
    (all, reason.as_str())
}

fn text_of(message: &Value, key: &str) -> String {
    message[key].as_str().unwrap_or("").to_string()
}

fn usage(v: &Value, member: &str) -> u64 {
    v["usage"][member].as_u64().unwrap()
}

fn key(byte: u8) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32])
}

fn declaring(mut b: Value, prefix: CachedPrefix) -> Value {
    b["cached_prefix"] = prefix.to_value();
    b
}

fn cached(v: &Value) -> u64 {
    v["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap()
}

// ---------------------------------------------------------------------------------------
// The served model.
// ---------------------------------------------------------------------------------------

/// A BF16 `qwen3_5` configuration is Qwen3.5 (the 2B's architecture, the dev model not
/// being the 0.8B's size), served through the Qwen3.5 template and tokenizer: its
/// end-of-sequence ids `<|im_end|>` and `config.json`'s `<|endoftext|>`, and no sampling
/// defaults (the checkpoints ship no generation config).
#[test]
fn the_dev_model_is_qwen35() {
    let model = qwen35_loaded();
    assert_eq!(model.architecture(), Architecture::Qwen35Dense2B);
    assert_eq!(model.format(), ChatFormat::Qwen3_5);
    let config = model.qwen_config().unwrap();
    assert_eq!(
        (config.weights, config.tie_word_embeddings),
        (WeightFormat::Bf16, true)
    );
    assert_eq!(config.vocab_size, fixture::QWEN_VOCAB);
    assert_eq!(model.tokenizer().vocab_size(), 248_070);
    assert_eq!(model.sampleable_vocab_size(), 248_070);
    assert_eq!(model.tokenizer().eos_token_ids(), [248_046, 248_044]);
    assert_eq!(model.defaults(), GenerationDefaults::NONE);
    assert!(
        !qwen35_model_dir().join("generation_config.json").exists(),
        "the dev model ships no generation config, as the checkpoints do not"
    );
}

/// A plain chat: the answer is the engine core's greedy generation over the same prompt
/// tokens (without speculation, while the node drafts two tokens a step from the model's
/// one MTP layer), parsed with thinking off: all of it is the answer.
#[tokio::test(flavor = "multi_thread")]
async fn a_plain_chat_is_what_the_engine_computes() {
    let node = TestNode::start_qwen35(&[]).await;
    let messages = json!([user(LONG_TEXT)]);
    let (_, ids) = prompt(&messages);
    let (tokens, finish) = direct_greedy(ids.clone(), 12);
    let (expected, reason) = expected_output(&tokens, finish);
    assert_eq!(expected.reasoning_content, "");

    let (status, v) = node.chat_json(&body(messages, 12)).await;
    assert_eq!(status, 200, "{v}");
    let message = &v["choices"][0]["message"];
    assert_eq!(text_of(message, "reasoning_content"), "");
    assert_eq!(text_of(message, "content"), expected.content);
    assert_eq!(v["choices"][0]["finish_reason"], reason);
    assert_eq!(usage(&v, "prompt_tokens"), ids.len() as u64);
    assert_eq!(usage(&v, "completion_tokens"), tokens.len() as u64);
    assert_eq!(v["model"], QWEN35_MODEL_ID);
    assert!(node.stats().drafted > 0, "the node drafts");
}

/// The node's request carries no thinking switch, so Qwen3.5's template default holds:
/// thinking off, the generation prompt closing an empty reasoning; and a length ending is
/// never continuable (the template trims the field a continuation would continue).
#[tokio::test(flavor = "multi_thread")]
async fn every_prompt_answers_without_thinking() {
    let (text, _) = prompt(&json!([user("hi")]));
    assert!(
        text.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "{text:?}"
    );
    let node = TestNode::start_qwen35(&[]).await;
    let events = node.chat_stream(&body(json!([user(LONG_TEXT)]), 6)).await;
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    let finish = &chunks[chunks.len() - 2]["choices"][0];
    assert_eq!(finish["finish_reason"], "length");
    assert_eq!(finish["continuable"], false);
}

// ---------------------------------------------------------------------------------------
// The prefix across requests, and declared cached prefixes.
// ---------------------------------------------------------------------------------------

fn bash(command: &str) -> Value {
    json!({"id": "call_1", "type": "function",
           "function": {"name": "bash", "arguments": {"command": command}}})
}

/// Within a user turn (a tool loop) the next prompt is this prompt, the answer and
/// `<|im_end|>` and more; a new user message's prompt shares the last one only up to its
/// generation header, the last four tokens (`<think>`, `\n\n`, `</think>`, `\n\n`)
/// differing, since the template writes no reasoning block for an assistant turn before
/// the last user message.
#[test]
fn a_tool_loop_extends_the_prompt_and_a_new_user_message_only_its_header() {
    let first = json!([user("List the files.")]);
    let (text, ids) = prompt(&first);
    let answer = "Let me look.";
    let call = "\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>";
    let tool_round = json!([
        user("List the files."),
        {"role": "assistant", "content": answer, "tool_calls": [bash("ls")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
    ]);
    let (next, next_ids) = prompt(&tool_round);
    assert!(
        next.starts_with(&format!("{text}{answer}{call}<|im_end|>")),
        "{next:?}"
    );
    assert_eq!(&next_ids[..ids.len()], &ids[..]);

    let new_turn = json!([
        user("List the files."),
        {"role": "assistant", "content": "a.txt and b.txt."},
        user("And their sizes?"),
    ]);
    let (_, turn_ids) = prompt(&new_turn);
    let shared = ids
        .iter()
        .zip(&turn_ids)
        .take_while(|(a, b)| a == b)
        .count();
    assert_eq!(shared, ids.len() - 4);
}

/// A user message whose prompt has `n` tokens and the property `pick` asks of `n`, from
/// growing prose.
fn first_turn_where(pick: impl Fn(usize) -> bool) -> Value {
    for words in 0..400 {
        let text = format!("{}{}", LONG_TEXT.repeat(6), " and more".repeat(words));
        let messages = json!([user(&text)]);
        if pick(prompt(&messages).1.len()) {
            return messages;
        }
    }
    panic!("no prompt length has the property")
}

/// A tool round declaring the last request's prompt by the shared rule is served with at
/// least the declared hit, the same answer as undeclared: its prompt extends the last
/// one and its output.
#[tokio::test(flavor = "multi_thread")]
async fn a_tool_round_declaring_the_last_prompt_is_served() {
    let node = TestNode::start_qwen35(&[]).await;
    let messages = first_turn_where(|n| n > 256);
    let mut first = body(messages.clone(), 6);
    first["cache_key"] = key(1).into();
    let (status, answer) = node.chat_json(&first).await;
    assert_eq!(status, 200, "{answer}");
    let content = text_of(&answer["choices"][0]["message"], "content");
    let mut round = messages.as_array().unwrap().clone();
    round.push(json!({"role": "assistant", "content": content, "tool_calls": [bash("ls")]}));
    round.push(json!({"role": "tool", "tool_call_id": "call_1", "content": "a.txt"}));
    let mut next = body(Value::Array(round), 6);
    next["cache_key"] = key(1).into();
    let added = (content.len() + 40) as u64;
    let prefix = declared_prompt(usage(&answer, "prompt_tokens"), added, 2, WINDOW).unwrap();
    let (status, served) = node.chat_json(&declaring(next.clone(), prefix)).await;
    assert_eq!(status, 200, "{served}");
    assert!(
        cached(&served) >= u64::from(prefix.cached_tokens),
        "{served}"
    );
    let (_, undeclared) = node.chat_json(&next).await;
    assert_eq!(served["choices"], undeclared["choices"]);
}

/// A new user message declaring the last request's prompt by the shared rule
/// (`declarable_cached_tokens`) is served when the last prompt's regeneration point (its
/// last block boundary before its final token) is within the `n - 4` tokens the new
/// prompt shares with it, and refused before anything is computed when it falls after
/// them, which `(n - 1) % BLOCK < 3` says: the last request then left no snapshot of the
/// recurrent state the new prompt can resume from. The client sends it again undeclared,
/// which is served.
#[tokio::test(flavor = "multi_thread")]
async fn a_new_user_turns_declaration_holds_unless_the_last_block_ends_in_the_header() {
    let node = TestNode::start_qwen35(&[]).await;
    for (byte, served) in [(2u8, true), (3, false)] {
        let messages = first_turn_where(|n| n > 256 && ((n - 1) % BLOCK >= 3) == served);
        let mut first = body(messages.clone(), 6);
        first["cache_key"] = key(byte).into();
        let (status, answer) = node.chat_json(&first).await;
        assert_eq!(status, 200, "{answer}");
        let content = text_of(&answer["choices"][0]["message"], "content");
        let question = "And what would a second, shorter answer say?";
        let mut turn = messages.as_array().unwrap().clone();
        turn.push(json!({"role": "assistant", "content": content}));
        turn.push(user(question));
        let mut next = body(Value::Array(turn), 6);
        next["cache_key"] = key(byte).into();
        let added = (content.len() + question.len()) as u64;
        let prefix = declared_prompt(usage(&answer, "prompt_tokens"), added, 2, WINDOW).unwrap();
        let (status, v) = node.chat_json(&declaring(next.clone(), prefix)).await;
        if served {
            assert_eq!(status, 200, "{v}");
            assert!(cached(&v) >= u64::from(prefix.cached_tokens), "{v}");
        } else {
            assert_eq!(status, 409, "{v}");
            assert_eq!(v["error"]["type"], "cached_prefix_missing", "{v}");
        }
        let (status, undeclared) = node.chat_json(&next).await;
        assert_eq!(status, 200, "{undeclared}");
        if served {
            assert_eq!(v["choices"], undeclared["choices"]);
        }
    }
}

/// After a tool loop, a new user message shares only the loop's first request up to its
/// generation header, while the client declares from the loop's last prompt: a loop that
/// added a quantum (256 tokens) or more past the first request leaves no snapshot within
/// the shared tokens at the declared boundary, and the declaration is refused; sent again
/// undeclared, the request is served.
#[tokio::test(flavor = "multi_thread")]
async fn after_a_long_tool_loop_a_new_user_turns_declaration_is_refused() {
    let node = TestNode::start_qwen35(&[]).await;
    let messages = json!([user("List the files.")]);
    let mut first = body(messages.clone(), 4);
    first["cache_key"] = key(4).into();
    let (status, answer) = node.chat_json(&first).await;
    assert_eq!(status, 200, "{answer}");
    let content = text_of(&answer["choices"][0]["message"], "content");
    let mut turn = messages.as_array().unwrap().clone();
    turn.push(json!({"role": "assistant", "content": content, "tool_calls": [bash("ls")]}));
    turn.push(json!({"role": "tool", "tool_call_id": "call_1", "content": LONG_TEXT.repeat(8)}));
    let mut round = body(Value::Array(turn.clone()), 4);
    round["cache_key"] = key(4).into();
    let (status, last) = node.chat_json(&round).await;
    assert_eq!(status, 200, "{last}");
    let last_prompt = usage(&last, "prompt_tokens");
    assert!(
        last_prompt > usage(&answer, "prompt_tokens") + 256,
        "{last}"
    );
    let reply = text_of(&last["choices"][0]["message"], "content");
    turn.push(json!({"role": "assistant", "content": reply}));
    turn.push(user("Thanks. And their sizes?"));
    let mut next = body(Value::Array(turn), 4);
    next["cache_key"] = key(4).into();
    let prefix = declared_prompt(last_prompt, (reply.len() + 30) as u64, 2, WINDOW).unwrap();
    let (status, v) = node.chat_json(&declaring(next.clone(), prefix)).await;
    assert_eq!(status, 409, "{v}");
    assert_eq!(v["error"]["type"], "cached_prefix_missing", "{v}");
    let (status, v) = node.chat_json(&next).await;
    assert_eq!(status, 200, "{v}");
}

// ---------------------------------------------------------------------------------------
// Refusals.
// ---------------------------------------------------------------------------------------

/// The node's strict subset refuses the template switches as for any model, and a
/// history call whose arguments are the empty string is refused (the template raises on
/// it); images are refused, the model being served text only.
#[tokio::test(flavor = "multi_thread")]
async fn what_the_format_does_not_have_is_refused() {
    let node = TestNode::start_qwen35(&[]).await;
    for (field, value) in [
        ("reasoning_effort", json!("low")),
        ("chat_template_kwargs", json!({"enable_thinking": true})),
    ] {
        let mut b = body(json!([user("hi")]), 2);
        b[field] = value;
        let (status, v) = node.chat_json(&b).await;
        assert_eq!(status, 400, "{field}: {v}");
    }
    let empty_arguments = body(
        json!([
            user("hi"),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": "bash", "arguments": ""}}]},
            {"role": "tool", "tool_call_id": "c", "content": "x"},
        ]),
        2,
    );
    let (status, v) = node.chat_json(&empty_arguments).await;
    assert_eq!(status, 400, "{v}");
    let image = body(
        json!([{"role": "user", "content": [{"type": "image", "image": "x"}]}]),
        2,
    );
    let (status, v) = node.chat_json(&image).await;
    assert_eq!(status, 400, "{v}");
}

/// A full boot from the directory (the weights hash, `config.json` read as Qwen3.5, the
/// pinned template and tokenizer, no generation config), and a Qwen3.5 directory holding
/// Qwen3.8's template refused after the hash.
#[test]
fn boot_verifies_the_qwen35_weights_and_its_template() {
    let config = config_from(&qwen35_env_map()).unwrap();
    let node = boot_without_swap(config).unwrap();
    assert_eq!(node.weights_hash, qwen35_weights_hash());

    let tmp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(qwen35_model_dir()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with('.') {
            std::fs::copy(entry.path(), tmp.path().join(&name)).unwrap();
        }
    }
    std::fs::copy(
        qwen_model_dir().join("chat_template.jinja"),
        tmp.path().join("chat_template.jinja"),
    )
    .unwrap();
    let hash = fixture::weights_hash_of(tmp.path());
    let e = LoadedModel::load(
        tmp.path(),
        &hash,
        eidola_server_engine::config::WeightsStorage::DevWritable,
        eidola_server_engine::config::ExecutorKind::Cpu,
    )
    .unwrap_err();
    assert!(e.to_string().contains("chat template"), "{e}");
}
