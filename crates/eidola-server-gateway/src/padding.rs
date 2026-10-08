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
//! opens until the frame that carries `[DONE]`. A frame holds whatever event
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
//! **Blocking** ([`pad_json_response`]): a JSON body is padded with trailing
//! whitespace to the next power of two, at least [`MIN_BODY_BUCKET`].
//!
//! Doctrine, and what an observer still learns:
//! `crates/eidola-server-gateway/AGENTS.md` → Client-facing traffic shape.

use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::Body;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use bytes::{Bytes, BytesMut};
use futures_util::Stream;
use tokio::sync::mpsc;
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
/// Past this, the producer's bounded channel fills and the upstream read waits,
/// so a model faster than the frame rate costs this much memory per stream
/// rather than its whole answer.
pub const MAX_BACKLOG_BYTES: usize = 1 << 20;

/// The smallest size a blocking JSON body is padded to.
pub const MIN_BODY_BUCKET: usize = 4096;

/// A frame must hold a whole comment event (`:\n\n`) with room to spare.
const MIN_FRAME_BYTES: usize = 16;

/// One server-sent event of a chat response, before framing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// A JSON payload (a completion chunk, or the metadata event): compact
    /// `serde_json` output, so it holds no line break.
    Json(String),
    /// The `[DONE]` sentinel, which ends the stream.
    Done,
}

/// How a streamed response is framed.
#[derive(Debug, Clone, Copy)]
pub struct Cadence {
    /// The size of every frame, in bytes.
    pub frame_bytes: usize,
    /// The interval between frames.
    pub tick: Duration,
    /// See [`MAX_BACKLOG_BYTES`].
    pub max_backlog_bytes: usize,
}

impl Cadence {
    /// What every client of this server receives.
    pub const CLIENT_FACING: Self = Self {
        frame_bytes: FRAME_BYTES,
        tick: TICK,
        max_backlog_bytes: MAX_BACKLOG_BYTES,
    };
}

/// Encodes events and cuts them into frames of one size.
///
/// Pure: what goes in a frame is decided here, when it goes out is
/// [`PaddedStream`]'s.
#[derive(Debug)]
pub struct Framer {
    frame_bytes: usize,
    /// Encoded event bytes not yet in a frame. Always ends at an event
    /// boundary, since events go in whole.
    pending: BytesMut,
    /// Whether the last event pushed is JSON, which can take trailing
    /// whitespace inside its data value.
    last_is_json: bool,
    done: bool,
}

impl Framer {
    /// # Panics
    /// If `frame_bytes` is too small to hold a padding comment.
    pub fn new(frame_bytes: usize) -> Self {
        assert!(
            frame_bytes >= MIN_FRAME_BYTES,
            "a frame must hold a padding comment"
        );
        Self {
            frame_bytes,
            pending: BytesMut::new(),
            last_is_json: false,
            done: false,
        }
    }

    /// Queue one event. Nothing is queued after [`StreamEvent::Done`].
    pub fn push(&mut self, event: &StreamEvent) {
        if self.done {
            return;
        }
        match event {
            StreamEvent::Json(json) => {
                // One `data:` field per line, as the format joins them back;
                // compact JSON has none, so this is one field.
                for line in json.split('\n') {
                    self.pending.extend_from_slice(b"data: ");
                    self.pending
                        .extend_from_slice(line.trim_end_matches('\r').as_bytes());
                    self.pending.extend_from_slice(b"\n");
                }
                self.pending.extend_from_slice(b"\n");
                self.last_is_json = true;
            }
            StreamEvent::Done => {
                self.pending.extend_from_slice(b"data: [DONE]\n\n");
                self.last_is_json = false;
                self.done = true;
            }
        }
    }

    /// Event bytes waiting for a frame.
    pub fn backlog(&self) -> usize {
        self.pending.len()
    }

    /// Whether `[DONE]` has been queued.
    pub fn done(&self) -> bool {
        self.done
    }

    /// The next frame: exactly `frame_bytes` long, whatever is waiting.
    pub fn frame(&mut self) -> Bytes {
        let size = self.frame_bytes;
        if self.pending.len() >= size {
            // Possibly mid-event; the next frame continues it.
            return self.pending.split_to(size).freeze();
        }
        // Everything waiting fits, and it ends at an event boundary.
        let mut frame = self.pending.split();
        let room = size - frame.len();
        if room >= 3 {
            frame.extend_from_slice(b":");
            frame.resize(size - 2, b' ');
            frame.extend_from_slice(b"\n\n");
        } else if !frame.is_empty() && self.last_is_json {
            // Too little room for a comment: trailing whitespace on the last
            // event's data line, before the blank line that ends it.
            let end = frame.len() - 2;
            let tail = frame.split_off(end);
            frame.resize(end + room, b' ');
            frame.extend_from_slice(&tail);
        } else {
            // Only after `[DONE]`: empty lines, which dispatch nothing.
            frame.resize(size, b'\n');
        }
        debug_assert_eq!(frame.len(), size);
        frame.freeze()
    }
}

/// The body of a streamed chat response: [`Framer`]'s frames, one per tick.
///
/// Takes events from the producer as soon as they are sent (up to the backlog
/// bound) and writes nothing until the next tick. Ends after the frame that
/// carries `[DONE]`, or once the producer has gone and everything it sent is
/// written. Dropping it (the client disconnected) drops the receiver, so the
/// producer's next send fails.
pub struct PaddedStream {
    rx: mpsc::Receiver<StreamEvent>,
    rx_open: bool,
    framer: Framer,
    interval: Interval,
    max_backlog_bytes: usize,
    ended: bool,
}

impl PaddedStream {
    /// The first frame is written at once; the rest follow every
    /// `cadence.tick`. Must be called inside a Tokio runtime.
    pub fn new(rx: mpsc::Receiver<StreamEvent>, cadence: Cadence) -> Self {
        let mut interval = tokio::time::interval(cadence.tick);
        // A late tick moves the schedule rather than writing two frames
        // back to back.
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self {
            rx,
            rx_open: true,
            framer: Framer::new(cadence.frame_bytes),
            interval,
            max_backlog_bytes: cadence.max_backlog_bytes,
            ended: false,
        }
    }
}

impl Stream for PaddedStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        while this.rx_open && !this.framer.done() && this.framer.backlog() < this.max_backlog_bytes
        {
            match this.rx.poll_recv(cx) {
                Poll::Ready(Some(event)) => this.framer.push(&event),
                Poll::Ready(None) => this.rx_open = false,
                Poll::Pending => break,
            }
        }
        if this.framer.backlog() == 0 && (this.framer.done() || !this.rx_open) {
            this.ended = true;
            return Poll::Ready(None);
        }
        match this.interval.poll_tick(cx) {
            Poll::Ready(_) => Poll::Ready(Some(Ok(this.framer.frame()))),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// A server-sent-events response whose body is `rx`'s events, framed at
/// `cadence`.
///
/// `no-transform` asks intermediaries not to compress: compressed, padding
/// would shrink to nothing and the frames would take the content's sizes.
pub fn padded_sse_response(rx: mpsc::Receiver<StreamEvent>, cadence: Cadence) -> Response {
    let mut response = Body::from_stream(PaddedStream::new(rx, cadence)).into_response();
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

/// The size a blocking body of `len` bytes is padded to: the next power of
/// two, at least [`MIN_BODY_BUCKET`].
pub fn body_bucket(len: usize) -> usize {
    len.max(MIN_BODY_BUCKET).next_power_of_two()
}

/// `response` with a JSON body padded to [`body_bucket`] by trailing
/// whitespace, which leaves the JSON value unchanged. Any other response (an
/// event stream, which its frames already shape) is returned as it is.
pub async fn pad_json_response(response: Response) -> Response {
    let is_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.split(';').next().unwrap_or("").trim() == "application/json");
    if !is_json {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(bytes) => bytes,
        // The bodies here are in memory; a failed read has nothing to pad.
        Err(_) => {
            return ServerError::Internal("response body unreadable".to_string()).into_response();
        }
    };
    let size = body_bucket(bytes.len());
    let mut padded = Vec::with_capacity(size);
    padded.extend_from_slice(&bytes);
    padded.resize(size, b' ');
    parts
        .headers
        .insert(header::CONTENT_LENGTH, HeaderValue::from(size));
    Response::from_parts(parts, Body::from(padded))
}

#[cfg(test)]
mod tests;
