//! The traffic shape of the client-facing chat response.
//!
//! A network observer between the client and this server sees ciphertext, but
//! ciphertext has a length and an arrival time. Relayed chunk by chunk, a
//! streamed completion puts one TLS record on the wire per upstream chunk, so
//! record sizes trace each token's length and their spacing traces the
//! model's cadence (and, with speculative decoding, how many tokens each step
//! accepted). Published attacks recover response topics and text from exactly
//! that. This module re-frames what the client receives so an observer learns
//! only how long the response ran and a coarse bound on its size.
//!
//! **Streaming** ([`padded_sse_response`]): the body is written as frames of
//! exactly [`FRAME_BYTES`], one per [`TICK`], from the moment the response
//! opens until the frame that carries `[DONE]`, cut by
//! `eidola_common::stream_shape`'s [`Framer`], the one an inference node's
//! stream to this server is cut by too. A frame holds whatever event
//! bytes are waiting, up to its size, and the remainder is padding; an event
//! larger than a frame continues in the next one. So every write is the same
//! size and the writes run on a clock, whether a token arrived, a hundred did,
//! or none. Neither depends on the content, so nothing downstream of these
//! writes (hyper's chunked encoding, the attestation shim that terminates TLS,
//! the TLS records it cuts, TCP) can re-introduce a content-dependent shape:
//! any re-chunking it does is a function of the frame size and the schedule.
//!
//! **Padding is server-sent-event syntax every client already ignores.** When
//! the waiting bytes end before the frame does, the rest is a comment event
//! (`:` + spaces + a blank line), which the format defines as dispatching
//! nothing; axum's keep-alive has always put the same shape on this route. A
//! remainder of one or two bytes is too short for a comment, so it goes inside
//! the last JSON event as trailing whitespace after the value (`data: {…}  `),
//! which JSON permits; after `[DONE]`, the end of the stream, it is blank lines.
//! Nothing is added inside a JSON value: a padding member in the chunk would
//! be read by a strict client's typed parse.
//!
//! **Everything else** ([`shape`], a layer around every route): a response
//! body is padded with trailing whitespace to the next power of two, at least
//! [`MIN_BODY_BUCKET`] — a blocking completion, an error, an extractor's
//! refusal, a 404.
//!
//! Doctrine, and what an observer still learns:
//! `crates/eidola-server-gateway/AGENTS.md` → Client-facing traffic shape.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use std::panic::AssertUnwindSafe;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderValue, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures_util::{FutureExt, Stream};
use tokio::sync::Notify;
use tokio::time::{Interval, MissedTickBehavior};

use crate::error::ServerError;

/// The size of every write of a streamed chat response, in bytes.
///
/// With [`TICK`], this caps delivery at 40 KiB/s: about 190 tokens a second
/// at the roughly 210 bytes a relayed one-token chunk takes on the wire. That
/// is above what the models sold here stream to one request, so the cap
/// rarely binds; when it does, output queues ([`MAX_BACKLOG_BYTES`]) and is
/// delivered at the cap. It is also the stream's cost: 40 KiB/s for as long
/// as the response runs.
pub const FRAME_BYTES: usize = 2048;

/// The interval between writes of a streamed chat response.
///
/// Twenty updates a second reads as continuous text, and it is the most a
/// token waits for its frame: a response's first output reaches the client up
/// to this much later than it reached this server.
pub const TICK: Duration = Duration::from_millis(50);

/// The most event bytes a stream holds waiting for frames before it stops
/// taking more from the upstream.
///
/// Past this, the producer waits ([`EventSender::send`]) and so does the
/// upstream read behind it, so a model faster than the frame rate costs this
/// much memory per stream rather than its whole answer. One event larger than
/// this on its own is held alone, and is cut across frames.
pub const MAX_BACKLOG_BYTES: usize = 1 << 20;

pub use eidola_common::stream_shape::{Framer, MIN_BODY_BUCKET, StreamEvent, body_bucket};

/// How a streamed response is framed, and how much it holds waiting.
#[derive(Debug, Clone, Copy)]
pub struct Cadence {
    /// The size of every frame and the interval between them.
    pub frames: eidola_common::stream_shape::Cadence,
    /// See [`MAX_BACKLOG_BYTES`].
    pub max_backlog_bytes: usize,
}

impl Cadence {
    /// What every client of this server receives.
    pub const CLIENT_FACING: Self = Self {
        frames: eidola_common::stream_shape::Cadence::new(FRAME_BYTES, TICK),
        max_backlog_bytes: MAX_BACKLOG_BYTES,
    };
}

/// What the producer and the stream share: the framer, bounded in bytes.
struct Shared {
    state: Mutex<SharedState>,
    /// Signalled when frames take bytes out, or the stream goes.
    space: Notify,
    max_backlog_bytes: usize,
}

struct SharedState {
    framer: Framer,
    sender_alive: bool,
    stream_alive: bool,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, SharedState> {
        // Nothing panics while holding it; a poisoned lock is still consistent.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The client has gone: its stream was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamClosed;

/// The producer's half of a padded stream (see [`channel`]).
pub struct EventSender {
    shared: Arc<Shared>,
}

impl EventSender {
    /// Queue `event` for framing, waiting while it does not fit.
    ///
    /// **The bound is in bytes.** An event is admitted only when the bytes
    /// already waiting plus its own fit in `max_backlog_bytes`, or, for an
    /// event larger than that on its own, when nothing else is waiting. So
    /// what the stream holds never passes the bound, except for one event
    /// larger than the bound, held alone (moved here from the producer, which
    /// already held it) and cut across as many frames as it takes. Fails once
    /// the stream has gone, which is how the producer sees a disconnect.
    pub async fn send(&self, event: StreamEvent) -> Result<(), StreamClosed> {
        let encoded = event.encode();
        drop(event);
        loop {
            let space = self.shared.space.notified();
            {
                let mut state = self.shared.lock();
                if !state.stream_alive {
                    return Err(StreamClosed);
                }
                let waiting = state.framer.backlog();
                if waiting == 0 || waiting + encoded.len() <= self.shared.max_backlog_bytes {
                    state.framer.push_encoded(encoded);
                    return Ok(());
                }
            }
            space.await;
        }
    }

    /// Resolves once the client has gone (its stream was dropped): how the
    /// producer notices a disconnect while it has nothing to send.
    pub async fn closed(&self) {
        loop {
            let space = self.shared.space.notified();
            if !self.shared.lock().stream_alive {
                return;
            }
            space.await;
        }
    }
}

impl Drop for EventSender {
    fn drop(&mut self) {
        self.shared.lock().sender_alive = false;
    }
}

/// A padded stream and the sender that feeds it, framed at `cadence`. Must
/// be called inside a Tokio runtime; the first frame is written at once and
/// the rest every `cadence.tick`.
pub fn channel(cadence: Cadence) -> (EventSender, PaddedStream) {
    let shared = Arc::new(Shared {
        state: Mutex::new(SharedState {
            framer: Framer::new(cadence.frames),
            sender_alive: true,
            stream_alive: true,
        }),
        space: Notify::new(),
        max_backlog_bytes: cadence.max_backlog_bytes,
    });
    let mut interval = tokio::time::interval(cadence.frames.tick());
    // A late tick moves the schedule rather than writing two frames back to
    // back.
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let stream = PaddedStream {
        shared: shared.clone(),
        interval,
        ended: false,
    };
    (EventSender { shared }, stream)
}

/// The body of a streamed chat response: [`Framer`]'s frames, one per tick.
///
/// Writes nothing between ticks. Ends after the frame that carries `[DONE]`,
/// or once the sender has gone and everything it sent is written. Dropping it
/// (the client disconnected) fails the sender's next send.
pub struct PaddedStream {
    shared: Arc<Shared>,
    interval: Interval,
    ended: bool,
}

impl PaddedStream {
    /// Event bytes waiting for a frame.
    pub fn backlog(&self) -> usize {
        self.shared.lock().framer.backlog()
    }
}

impl Drop for PaddedStream {
    fn drop(&mut self) {
        self.shared.lock().stream_alive = false;
        self.shared.space.notify_one();
    }
}

impl Stream for PaddedStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        {
            let state = this.shared.lock();
            if state.framer.backlog() == 0 && (state.framer.done() || !state.sender_alive) {
                this.ended = true;
                return Poll::Ready(None);
            }
        }
        match this.interval.poll_tick(cx) {
            Poll::Ready(_) => {
                let frame = this.shared.lock().framer.frame();
                this.shared.space.notify_one();
                Poll::Ready(Some(Ok(Bytes::from(frame))))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A server-sent-events response whose body is `stream`.
///
/// `no-transform` asks intermediaries not to compress: compressed, padding
/// would shrink to nothing and the frames would take the content's sizes.
pub fn padded_sse_response(stream: PaddedStream) -> Response {
    let mut response = Body::from_stream(stream).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    response
}

/// Every response the server sends, shaped: the layer [`shape`] installs.
///
/// Runs outside every route and the fallback, so it sees what the handlers
/// never do: an extractor's refusal (a bad credential, a malformed or
/// oversized body), a 404 for an unknown path, a 405 for a wrong method.
/// A handler that panics is answered with a fixed 500 rather than a dropped
/// connection. Then [`pad_body`].
pub async fn shape_response(request: Request, next: Next) -> Response {
    let response = match AssertUnwindSafe(next.run(request)).catch_unwind().await {
        Ok(response) => response,
        Err(_) => {
            ServerError::Internal("the request could not be handled".to_string()).into_response()
        }
    };
    pad_body(response).await
}

/// `router` with [`shape_response`] around every route and the fallback.
///
/// **Every route, not only chat completions.** One rule leaves nothing to
/// classify, so nothing can be misclassified (the reasoning the telemetry
/// sampler follows). Inference content reaches only the chat route, but a
/// response anywhere whose length depends on what the client sent (a refusal
/// that names the field it refused, say) is the same leak, and padding a
/// small account answer to 4 KiB costs nothing that matters.
pub fn shape(router: axum::Router) -> axum::Router {
    router.layer(axum::middleware::from_fn(shape_response))
}

/// `response` with its body padded to [`body_bucket`] by trailing spaces, a
/// matching `Content-Length`, and `no-transform` added to its
/// `Cache-Control`.
///
/// Padded: a JSON body (trailing whitespace leaves the value unchanged), a
/// `text/*` body, and a body with no type (an empty 404 or 405). Left alone:
/// an event stream, which its frames shape; a status that carries no body
/// (1xx, 204, 304), whose size is fixed; and any other type, which trailing
/// bytes would corrupt (no route serves one). A body that is already
/// content-encoded is refused with a 500: nothing in this server compresses,
/// and compression would undo the padding, so one appearing is a defect to
/// surface rather than a response to send as it is.
pub async fn pad_body(response: Response) -> Response {
    let status = response.status();
    if status.is_informational()
        || status == axum::http::StatusCode::NO_CONTENT
        || status == axum::http::StatusCode::NOT_MODIFIED
    {
        return response;
    }
    let media = response.headers().get(header::CONTENT_TYPE).map(|v| {
        v.to_str()
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    });
    let paddable = match media.as_deref() {
        None => true,
        Some("application/json") => true,
        Some(m) => m.starts_with("text/") && m != "text/event-stream",
    };
    if !paddable {
        return response;
    }
    if response.headers().contains_key(header::CONTENT_ENCODING) {
        let refused =
            ServerError::Internal("a response was content-encoded".to_string()).into_response();
        return Box::pin(pad_body(refused)).await;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        // The bodies here are in memory; one that cannot be read is a defect,
        // answered the same padded way.
        Err(_) => {
            let failed =
                ServerError::Internal("the response could not be read".to_string()).into_response();
            return Box::pin(pad_body(failed)).await;
        }
    };
    let padded = eidola_common::stream_shape::pad_body(bytes.to_vec());
    let size = padded.len();
    parts
        .headers
        .insert(header::CONTENT_LENGTH, HeaderValue::from(size));
    let cache_control = match parts.headers.get(header::CACHE_CONTROL) {
        Some(existing) => {
            let existing = existing.to_str().unwrap_or("");
            if existing.contains("no-transform") {
                existing.to_string()
            } else {
                format!("{existing}, no-transform")
            }
        }
        None => "no-transform".to_string(),
    };
    if let Ok(value) = HeaderValue::from_str(&cache_control) {
        parts.headers.insert(header::CACHE_CONTROL, value);
    }
    Response::from_parts(parts, Body::from(padded))
}

#[cfg(test)]
mod tests;
