//! One whole answer, for a caller that reads one, from any backend.
//!
//! A turn's blocking rounds, the chores ([`crate::utility`]) and the proxy's
//! non-streamed route each read one JSON completion. Every backend but
//! Eidola's answers that as asked. The Eidola gateway serves chat only as a
//! stream — a stream is what hands the client what it paid for whatever ends
//! it — so for that backend the request asks for one and the answer is
//! assembled here from its chunks into the body a blocking backend writes:
//! the completion, plus the terminal metadata event's `privacy`,
//! `verification` and `refunds` members. The readers downstream parse and
//! settle that body exactly as they read a blocking one, so the in-band
//! refunds settle the spend through the same code either way.
//!
//! [`WholeAnswer::for_backend`] is the one choice, keyed on the backend kind:
//! it says whether the request streams ([`WholeAnswer::streams`]) and how the
//! answer is read ([`WholeAnswer::read`]).
//!
//! The stream is read by [`read_events`], the reader a turn's streaming round
//! reads with, under the caller's ceiling on what the answer is made of
//! (padding uncounted). A stream that ends without saying it finished — the
//! gateway cut it short — is no whole answer: it is an [`AnswerFailure`]
//! carrying the refunds the stream delivered and what the Record keeps of it,
//! which is what a streaming twin does with the same ending.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::backends::BackendKind;
use crate::error::AppError;
use crate::peer_read::{self, BoundedBody};
use crate::recorded::{self, RecordedBody};

/// How a backend gives a blocking caller one whole answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WholeAnswer {
    /// The backend answers a blocking request with the completion itself.
    Body,
    /// The backend answers chat only as a stream, which is assembled here.
    Assembled,
}

/// A whole answer: its body as a blocking backend writes it, and what the
/// Record keeps of the exchange.
pub(crate) struct Answer {
    /// The completion (or, for a non-2xx, the error document), read under the
    /// caller's ceiling.
    pub(crate) body: BoundedBody,
    pub(crate) record: Vec<u8>,
}

/// An answer that did not arrive whole: a read that failed, a stream cut
/// short, or a stream that was not one. The exchange happened, so the
/// Record keeps what arrived, and the refunds it carried settle the spend.
pub(crate) struct AnswerFailure {
    pub(crate) error: AppError,
    pub(crate) record: Vec<u8>,
    /// The `refunds` the answer delivered before it ended, if any.
    pub(crate) refund: Option<Value>,
}

impl WholeAnswer {
    pub(crate) fn for_backend(kind: BackendKind) -> Self {
        match kind {
            BackendKind::Eidola => WholeAnswer::Assembled,
            BackendKind::Local | BackendKind::OpenAi | BackendKind::LlamaCpp => WholeAnswer::Body,
        }
    }

    /// The request's `stream` member.
    pub(crate) fn streams(self) -> bool {
        self == WholeAnswer::Assembled
    }

    /// Read `response` into one answer of at most `ceiling` bytes: a body's
    /// own length, or what is assembled out of a stream (its content,
    /// reasoning and tool calls), whose envelope has a ceiling of its own
    /// ([`peer_read::MAX_ASSEMBLED_STREAM_BYTES`]).
    ///
    /// A non-2xx is an error document on either transport and is read whole,
    /// as a blocking body: its refunds, when it carries them, are in it.
    pub(crate) async fn read(
        self,
        response: reqwest::Response,
        ceiling: usize,
        backend_id: &str,
    ) -> Result<Answer, AnswerFailure> {
        let status = response.status();
        if self == WholeAnswer::Body || !status.is_success() {
            return read_body(response, ceiling).await;
        }
        if !crate::is_event_stream(response.headers()) {
            // Asked for a stream, answered something else: refused, after its
            // refunds are read, since the credential is spent either way.
            let answer = read_body(response, ceiling).await?;
            let refund = (!answer.body.over_ceiling)
                .then(|| serde_json::from_slice::<Value>(&answer.body.bytes).ok())
                .flatten()
                .and_then(|body| crate::inline_refund(&body).cloned());
            let error = if answer.body.over_ceiling {
                peer_read::answer_past_ceiling(backend_id)
            } else {
                not_an_event_stream(backend_id, status.as_u16())
            };
            return Err(AnswerFailure {
                error,
                record: answer.record,
                refund,
            });
        }
        let mut assembler = Assembler::under(ceiling);
        let envelope = peer_read::MAX_ASSEMBLED_STREAM_BYTES.max(ceiling);
        let read = read_events(response, backend_id, envelope, |event| {
            assembler.push(event)
        })
        .await;
        if let Some(error) = read.read_error {
            return Err(AnswerFailure {
                error,
                record: read.raw.seal_response(false),
                refund: read.refund,
            });
        }
        // Read to its end all the same: the stream's refunds are at its end.
        if assembler.past_ceiling {
            return Err(AnswerFailure {
                error: peer_read::answer_past_ceiling(backend_id),
                record: read.raw.seal_response(read.finished),
                refund: read.refund,
            });
        }
        if !read.finished && !assembler.finished() {
            return Err(AnswerFailure {
                error: stream_ended_unfinished(),
                record: read.raw.seal_response(true),
                refund: read.refund,
            });
        }
        let body = assembler.finish(read.refund);
        Ok(Answer {
            body: BoundedBody {
                bytes: body.to_string().into_bytes(),
                over_ceiling: false,
            },
            record: read.raw.seal_response(true),
        })
    }
}

/// A body read whole, bounded as it arrives.
async fn read_body(response: reqwest::Response, ceiling: usize) -> Result<Answer, AnswerFailure> {
    match peer_read::read_bounded(response, ceiling).await {
        Ok(body) => {
            let record = recorded::recorded_answer(&body);
            Ok(Answer { body, record })
        }
        Err(failure) => Err(AnswerFailure {
            error: failure.error,
            record: recorded::recorded_cut_answer(&failure.partial),
            refund: None,
        }),
    }
}

/// The refusal of a `2xx` to a streaming request that is not server-sent
/// events.
pub(crate) fn not_an_event_stream(backend_id: &str, status: u16) -> AppError {
    AppError::Network {
        message: format!(
            "`{backend_id}` answered {status} to a streaming request with a body that is not \
             server-sent events"
        ),
    }
}

/// The failure of a stream that ended without saying it was over: no
/// `[DONE]`, and no `finish_reason` on any chunk.
pub(crate) fn stream_ended_unfinished() -> AppError {
    AppError::Network {
        message: "the model's response stream ended without finishing".to_string(),
    }
}

/// What reading a stream came to.
pub(crate) struct EventsRead {
    /// What the Record keeps, padding counted and not kept.
    pub(crate) raw: RecordedBody,
    /// What stopped the read before the upstream's own end, if anything did.
    pub(crate) read_error: Option<AppError>,
    /// Whether `[DONE]` arrived.
    pub(crate) finished: bool,
    /// The last `refunds` the stream carried: the gateway's terminal metadata
    /// event, or an unterminated final event holding one.
    pub(crate) refund: Option<Value>,
}

/// Read a server-sent event stream to its end, handing each event's JSON
/// data to `on_event` (`[DONE]` is [`EventsRead::finished`]).
///
/// **Three bounds, because a peer chooses every byte here.** What the Record
/// keeps ([`RecordedBody`], the retention cap), what one event may accumulate
/// before it is refused ([`peer_read::MAX_SSE_EVENT_BYTES`], the frame
/// buffer's ceiling: a backend that never ends an event would otherwise grow
/// it until the process dies), and what the whole stream may deliver
/// (`ceiling`: the events are assembled into an answer in memory, so a stream
/// without end is an answer without end).
///
/// Bytes accumulate in a frame buffer split at every event boundary the
/// format allows ([`crate::find_event_boundary`]); each event's data is
/// assembled once from all of its `data:` fields ([`crate::sse_event_data`])
/// and parsed once. An event that is not UTF-8 or not JSON is skipped.
pub(crate) async fn read_events(
    response: reqwest::Response,
    backend_id: &str,
    ceiling: usize,
    mut on_event: impl FnMut(&Value),
) -> EventsRead {
    use futures_util::StreamExt;

    let mut byte_stream = response.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut raw = RecordedBody::default();
    let mut read_error: Option<AppError> = None;
    let mut finished = false;
    let mut refund: Option<Value> = None;

    while let Some(chunk) = byte_stream.next().await {
        let bytes = match chunk {
            Ok(bytes) => bytes,
            Err(e) => {
                read_error = Some(AppError::Network {
                    message: format!("stream read failed: {e}"),
                });
                break;
            }
        };
        buf.extend_from_slice(&bytes);

        let mut oversized = false;
        let mut past_ceiling = false;
        while let Some((pos, boundary_len)) = crate::find_event_boundary(&buf) {
            // The frame ceiling is asked of a complete event **before** it is
            // drained (`peer_read::event_past_ceiling`).
            if peer_read::event_past_ceiling(Some(pos), 0) {
                oversized = true;
                break;
            }
            let event_bytes = buf.drain(..pos).collect::<Vec<u8>>();
            // The boundary itself — 2, 3 or 4 bytes, whichever pair of line
            // terminators matched — goes to the Record with its event.
            let boundary = buf.drain(..boundary_len).collect::<Vec<u8>>();
            raw.push_event(&event_bytes, &boundary);
            // **The ceiling bounds what the answer is made of, not padding**,
            // so it is asked of each event once it is classified, never of a
            // transport chunk: the Eidola server streams fixed-size frames on
            // a fixed tick, filled with comment events for as long as the
            // answer runs. The event that crosses the ceiling is not used.
            if raw.payload_received() > ceiling {
                read_error = Some(AppError::Network {
                    message: format!(
                        "the model's response stream passed the {ceiling}-byte ceiling this app \
                         reads for one answer"
                    ),
                });
                past_ceiling = true;
                break;
            }
            let Ok(event) = std::str::from_utf8(&event_bytes) else {
                continue;
            };
            let Some(payload) = crate::sse_event_data(event) else {
                continue;
            };
            let payload = payload.trim_start();
            if payload == "[DONE]" {
                finished = true;
                continue;
            }
            let Ok(json) = serde_json::from_str::<Value>(payload) else {
                continue; // comments and heartbeats
            };
            // Read before anything else of the event: the metadata event
            // carries no `choices` at all.
            if let Some(found) = crate::inline_refund(&json) {
                refund = Some(found.clone());
            }
            on_event(&json);
        }

        // **What is left over is an event still being accumulated**, and a
        // backend that never terminates one grows it without end: the other
        // half of the frame ceiling, beside the complete event refused above.
        if past_ceiling {
            break;
        }
        if oversized || peer_read::event_past_ceiling(None, buf.len()) {
            read_error = Some(peer_read::oversized_event(backend_id));
            raw.push_event(&buf, &[]);
            buf.clear();
            break;
        }
        if finished {
            break;
        }
    }
    // Whatever arrived and was not drained as an event (a partial event, or
    // what followed `[DONE]` in its read) is recorded as it came.
    raw.push_event(&buf, &[]);

    // An unterminated final event is not an event the format delivers, so
    // nothing in it becomes part of the answer; a refund in it is still the
    // credential's successor, and wallet material settles wherever the server
    // put it.
    if read_error.is_none()
        && !finished
        && let Ok(tail) = std::str::from_utf8(&buf)
        && let Some(payload) = crate::sse_event_data(tail)
        && let Ok(json) = serde_json::from_str::<Value>(payload.trim_start())
        && let Some(found) = crate::inline_refund(&json)
    {
        refund = Some(found.clone());
    }

    EventsRead {
        raw,
        read_error,
        finished,
        refund,
    }
}

/// The `object` of the gateway's terminal metadata event.
const METADATA_OBJECT: &str = "eidola.chat.completion.metadata";

/// One choice as its deltas arrive.
#[derive(Default)]
struct ChoiceParts {
    content: String,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    /// Each tool call by its `index`, assembled by the streaming turn's rule
    /// ([`crate::accumulate_tool_call_deltas`]).
    tool_calls: BTreeMap<u64, crate::StreamingToolCall>,
    /// A `tool_calls` delta that is present, not null and not an array: kept
    /// as it came, so the blocking reader refuses it as it refuses such a
    /// body.
    unusable_tool_calls: Option<Value>,
    finish_reason: Option<Value>,
    continuable: Option<bool>,
}

/// A completion being assembled from a stream's chunks.
///
/// Each choice's deltas joined in order (`content`, `reasoning_content`,
/// `reasoning`, and each tool call by its `index`), its finish reason and
/// `continuable` from the chunk that carries them, the last `usage` the
/// stream reported, and the answer named (`id`, `created`, `model`) by its
/// first chunk. The result is the completion a blocking answer is: `content`
/// null when nothing was given out, the reasoning fields only when something
/// was, `tool_calls` only when a call was made, each call without the
/// stream's `index` framing key. Held to the record of the gateway's blocking
/// answers by `tests::the_assembled_answer_is_the_blocking_answer`.
#[derive(Default)]
struct Assembler {
    id: Option<Value>,
    created: Option<Value>,
    model: Option<Value>,
    choices: BTreeMap<u64, ChoiceParts>,
    usage: Option<Value>,
    privacy: Option<Value>,
    verification: Option<Value>,
    /// The most the deltas may add up to, when there is a limit.
    ceiling: Option<usize>,
    /// The bytes of delta taken in so far.
    held: usize,
    /// A delta passed `ceiling`: nothing more is taken in, and what was is
    /// not an answer.
    past_ceiling: bool,
}

impl Assembler {
    /// An assembler that takes in at most `ceiling` bytes of deltas.
    fn under(ceiling: usize) -> Self {
        Assembler {
            ceiling: Some(ceiling),
            ..Assembler::default()
        }
    }

    /// Count `bytes` of delta; false once they pass the ceiling.
    fn take(&mut self, bytes: usize) -> bool {
        self.held = self.held.saturating_add(bytes);
        if self.ceiling.is_some_and(|ceiling| self.held > ceiling) {
            self.past_ceiling = true;
        }
        !self.past_ceiling
    }

    fn push(&mut self, event: &Value) {
        if event.get("object").and_then(Value::as_str) == Some(METADATA_OBJECT) {
            self.privacy = event.get("privacy").cloned();
            self.verification = event.get("verification").cloned();
            return;
        }
        if self.past_ceiling {
            return;
        }
        if self.id.is_none() {
            self.id = Some(event.get("id").cloned().unwrap_or_else(|| "".into()));
            self.created = event.get("created").cloned();
            self.model = event.get("model").cloned();
        }
        if let Some(usage) = event.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choices) = event.get("choices").and_then(Value::as_array) else {
            return;
        };
        for choice in choices {
            let index = choice.get("index").and_then(Value::as_u64).unwrap_or(0);
            if let Some(delta) = choice.get("delta").filter(|d| !d.is_null())
                && !self.take(delta.to_string().len())
            {
                return;
            }
            let parts = self.choices.entry(index).or_default();
            if let Some(delta) = choice.get("delta") {
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    parts.content.push_str(text);
                }
                if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
                    parts
                        .reasoning_content
                        .get_or_insert_default()
                        .push_str(text);
                }
                if let Some(text) = delta.get("reasoning").and_then(Value::as_str) {
                    parts.reasoning.get_or_insert_default().push_str(text);
                }
                match delta.get("tool_calls") {
                    None | Some(Value::Null) => {}
                    Some(Value::Array(calls)) => {
                        crate::accumulate_tool_call_deltas(&mut parts.tool_calls, calls);
                    }
                    Some(other) => {
                        parts
                            .unusable_tool_calls
                            .get_or_insert_with(|| other.clone());
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").filter(|r| !r.is_null()) {
                parts.finish_reason = Some(reason.clone());
            }
            if let Some(continuable) = choice.get("continuable").and_then(Value::as_bool) {
                parts.continuable = Some(continuable);
            }
        }
    }

    /// Whether any choice named a finish reason: a stream that said it was
    /// done in that place, if not with `[DONE]`.
    fn finished(&self) -> bool {
        self.choices.values().any(|c| c.finish_reason.is_some())
    }

    /// The completion, as a blocking body, carrying `refunds`.
    fn finish(self, refunds: Option<Value>) -> Value {
        let choices: Vec<Value> = self
            .choices
            .into_iter()
            .map(|(index, parts)| {
                let mut message = serde_json::Map::new();
                message.insert("role".into(), "assistant".into());
                message.insert(
                    "content".into(),
                    if parts.content.is_empty() {
                        Value::Null
                    } else {
                        Value::String(parts.content)
                    },
                );
                if let Some(text) = parts.reasoning_content.filter(|s| !s.is_empty()) {
                    message.insert("reasoning_content".into(), Value::String(text));
                }
                if let Some(text) = parts.reasoning.filter(|s| !s.is_empty()) {
                    message.insert("reasoning".into(), Value::String(text));
                }
                if let Some(unusable) = parts.unusable_tool_calls {
                    message.insert("tool_calls".into(), unusable);
                } else if !parts.tool_calls.is_empty() {
                    message.insert(
                        "tool_calls".into(),
                        Value::Array(crate::streamed_tool_call_objects(parts.tool_calls)),
                    );
                }
                let mut choice = serde_json::Map::new();
                choice.insert("index".into(), index.into());
                choice.insert("message".into(), Value::Object(message));
                choice.insert(
                    "finish_reason".into(),
                    parts.finish_reason.unwrap_or(Value::Null),
                );
                if let Some(continuable) = parts.continuable {
                    choice.insert("continuable".into(), continuable.into());
                }
                Value::Object(choice)
            })
            .collect();
        let mut body = serde_json::Map::new();
        body.insert("id".into(), self.id.unwrap_or_else(|| "".into()));
        body.insert("object".into(), "eidola.chat.completion".into());
        body.insert("created".into(), self.created.unwrap_or_else(|| 0.into()));
        body.insert("model".into(), self.model.unwrap_or_else(|| "".into()));
        body.insert("choices".into(), Value::Array(choices));
        if let Some(usage) = self.usage {
            body.insert("usage".into(), usage);
        }
        if let Some(privacy) = self.privacy {
            body.insert("privacy".into(), privacy);
        }
        if let Some(verification) = self.verification {
            body.insert("verification".into(), verification);
        }
        if let Some(refunds) = refunds {
            body.insert(
                eidola_common::spend_credentials::REFUNDS_MEMBER.into(),
                refunds,
            );
        }
        Value::Object(body)
    }
}

#[cfg(test)]
mod tests;
