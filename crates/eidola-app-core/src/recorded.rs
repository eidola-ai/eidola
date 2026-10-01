//! What the Record **keeps** of an exchange: bounded, and honest about it.
//!
//! Two surfaces write `request` rows — a chat turn (`lib.rs`) and a proxied
//! completion (`proxy::route`) — and both write the request body this app sent
//! and the response body it read. Neither may keep an unbounded copy: a prompt
//! is the reader's own and may be arbitrarily long, a response is whatever a
//! peer chose to send, and every row is durable — in the profile database and
//! its WAL, with nothing pruning `request` rows to take the bytes back. So the
//! retention is capped at [`RECORD_BODY_MAX_BYTES`] per column, enforced as the
//! bytes arrive, and **a partial says it is one**: the seal states both numbers
//! in the payload itself, and names which side of the exchange it truncated.
//!
//! This module is the retention policy and nothing else. What a surface does
//! when a *read* stops — the ceilings in [`crate::peer_read`] — is the
//! surface's ending to state, and each states it beside the seal
//! ([`BodyRead`] here for a blocking answer; the proxy's stream adds its own
//! delivery note).

use serde_json::Value;

use crate::peer_read::{BoundedBody, MAX_RESPONSE_BYTES};

/// The most of one body a `request` row keeps — per column.
///
/// **A bound the Record needs and the exchange does not.** The prompt still
/// travels upstream whole and a delivered answer still reaches its reader;
/// what is bounded is what is *retained*, which is held in memory until the
/// row is written and then kept on disk. A megabyte is far past any answer a
/// person reads and far short of a size worth holding: an SSE transcript of a
/// 4k-token completion is tens of kilobytes.
pub(crate) const RECORD_BODY_MAX_BYTES: usize = 1 << 20;

/// A body on its way to the Record, kept to [`RECORD_BODY_MAX_BYTES`].
///
/// **Truncation is recorded as truncation.** A Record row holding the first
/// megabyte of a larger body and saying nothing would be a partial claiming to
/// be whole, which is the one thing this trail may never be — a reader opens it
/// to see what left this machine. So the seal states both numbers, in the
/// payload itself, in a form no upstream could have sent by accident.
#[derive(Default)]
pub(crate) struct RecordedBody {
    pub(crate) kept: Vec<u8>,
    pub(crate) received: usize,
}

impl RecordedBody {
    /// Take more of the body, keeping only what fits under the cap — enforced
    /// here, as the bytes arrive, because sealing alone would keep the row
    /// honest while the memory was already spent.
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.received += bytes.len();
        let room = RECORD_BODY_MAX_BYTES.saturating_sub(self.kept.len());
        if room > 0 {
            self.kept.extend_from_slice(&bytes[..room.min(bytes.len())]);
        }
    }

    /// The bytes to record for a **response**, with the cap's note when they
    /// are not all of them, and nothing about how the response ended — that is
    /// the caller's ending to state (see the module doc).
    ///
    /// `read_to_end` is whether the read ran to the upstream's own end, because
    /// only then is `received` the response's size rather than how far this app
    /// got.
    pub(crate) fn seal_response(self, read_to_end: bool) -> Vec<u8> {
        seal_recorded_body(
            self.kept,
            self.received,
            RecordedSide::Response { read_to_end },
        )
    }

    /// The bytes to record for a **request** body.
    ///
    /// **There is no ending to state here, and that is why this seal is its
    /// own.** A request body was constructed by this app and handed to the
    /// transport in one piece, so nothing about how it *ended* can be in
    /// question — only the retention cap can make the row a partial, and the
    /// cap's own note says exactly that. Every seal names its class, so the
    /// absence of an ending is a stated fact rather than a forgotten one.
    ///
    /// **And the cap's note is a request's note** ([`RecordedSide`]): a shared
    /// wording had this column saying it kept part of a "response" whose
    /// remainder "was delivered", which is the other side's story told about
    /// these bytes.
    pub(crate) fn seal_request(self) -> Vec<u8> {
        seal_recorded_body(self.kept, self.received, RecordedSide::Request)
    }

    /// The bytes to record for a blocking answer, with the note when the
    /// ceiling — not the upstream — is why the read stopped.
    pub(crate) fn seal_blocking(self, read: BodyRead) -> Vec<u8> {
        let mut out = self.seal_response(read == BodyRead::Complete);
        if let Some(note) = read.note() {
            out.extend_from_slice(note.as_bytes());
        }
        out
    }
}

/// How a blocking answer's read ended.
///
/// A read this app ended at its own ceiling looks byte-for-byte like an
/// upstream that finished, so a row carrying no marker would be a partial
/// claiming to be whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BodyRead {
    /// The upstream closed on its own and everything it sent was read.
    Complete,
    /// [`MAX_RESPONSE_BYTES`] stopped the read. What is recorded stops where
    /// this app stopped reading, not where the upstream stopped sending.
    CeilingReached,
}

impl BodyRead {
    fn note(self) -> Option<String> {
        match self {
            BodyRead::Complete => None,
            // **No claim about what is above**: the retention cap may have
            // kept less than was read, and its own note says so. What this
            // note knows is where reading stopped and what that cost.
            BodyRead::CeilingReached => Some(format!(
                "\n\n[eidola: the read stopped at the {MAX_RESPONSE_BYTES}-byte ceiling this app \
                 holds for one answer. Anything the upstream sent beyond it was never read, and \
                 no answer was passed on.]\n"
            )),
        }
    }
}

/// Which side of an exchange a recorded body is.
///
/// **The cap's note names the side it truncated, because the Record's whole
/// value is that it describes itself.** One wording for both columns had the
/// `request_body` column saying it kept part of a "response" whose remainder
/// "was delivered" — response words, on the bytes this app *sent*, so a reader
/// looking at a truncated exchange could not tell which half was cut. And each
/// side's note says exactly what it knows: a request's size is known, because
/// this app built it whole, while a response's remainder was **received**, and
/// whether it then reached anyone is the ending's to say, not this note's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecordedSide {
    /// A body this app constructed in one piece.
    Request,
    /// A body an upstream sent to this app. `read_to_end` is whether the read
    /// ran to the upstream's own end — only then is the byte count read the
    /// response's size; otherwise it is a lower bound, and the note says so.
    Response { read_to_end: bool },
}

impl RecordedSide {
    fn cap_note(self, kept: usize, received: usize) -> String {
        match self {
            // The size is known — this app built the body whole — but not
            // whether it was sent: a row is also written for a request
            // refused before sending or one whose send failed, and the row's
            // status and error are what say which.
            RecordedSide::Request => format!(
                "\n\n[eidola: this Record entry keeps the first {kept} bytes of a {received}-byte \
                 request. The rest was not retained.]\n"
            ),
            // Received, never "delivered": whether these bytes reached a
            // reader is not this note's to say, and the endings that do say
            // it would be contradicted on the same row by a note that did.
            RecordedSide::Response { read_to_end: true } => format!(
                "\n\n[eidola: this Record entry keeps the first {kept} bytes of a {received}-byte \
                 response. The rest was received and not retained.]\n"
            ),
            // **A read this app stopped knows how far it got, not how large the
            // response was** — `received` is a prefix, so naming it as the
            // size would understate the body and contradict the ending's note
            // on the same row. Stated as the lower bound it is.
            RecordedSide::Response { read_to_end: false } => format!(
                "\n\n[eidola: this Record entry keeps the first {kept} of the {received} bytes \
                 this app read of the response before it stopped reading; the response was at \
                 least that large. The rest of what was read was not retained.]\n"
            ),
        }
    }
}

/// Take a whole body down to what the Record keeps, saying so — and saying
/// which side it is — if it did.
fn seal_recorded_body(mut kept: Vec<u8>, received: usize, side: RecordedSide) -> Vec<u8> {
    if kept.len() > RECORD_BODY_MAX_BYTES {
        kept.truncate(RECORD_BODY_MAX_BYTES);
    }
    if received <= kept.len() {
        return kept;
    }
    let note = side.cap_note(kept.len(), received);
    kept.extend_from_slice(note.as_bytes());
    kept
}

/// What the Record keeps of a request body this app built — bounded, and
/// honest about it. See [`RecordedBody::seal_request`].
///
/// **The request still travels whole**; what is bounded is what is kept.
pub(crate) fn recorded_request(body: &Value) -> Vec<u8> {
    let mut kept = RecordedBody::default();
    kept.push(body.to_string().as_bytes());
    kept.seal_request()
}

/// What the Record keeps of a blocking answer, stating both the retention cap
/// and the read ceiling where either applied.
pub(crate) fn recorded_answer(answer: &BoundedBody) -> Vec<u8> {
    let mut kept = RecordedBody::default();
    kept.push(&answer.bytes);
    // The ceiling can stop the read part-way through a chunk, so more arrived
    // than was kept to parse; the Record states the larger number.
    kept.received = answer.received;
    kept.seal_blocking(if answer.over_ceiling {
        BodyRead::CeilingReached
    } else {
        BodyRead::Complete
    })
}

/// What the Record keeps of a body whose read **failed** part-way: the prefix
/// that arrived, sealed as a read this app did not finish (a lower bound where
/// the cap applied). The failure itself is the row's `error`, which is where
/// every surface states an ending it did not choose.
pub(crate) fn recorded_cut_answer(partial: &BoundedBody) -> Vec<u8> {
    let mut kept = RecordedBody::default();
    kept.push(&partial.bytes);
    kept.received = partial.received;
    kept.seal_response(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request's note is a request's: it states the size this app built and
    /// claims nothing about sending, delivering or receiving.
    #[test]
    fn a_truncated_request_says_it_is_a_request() {
        let body = serde_json::json!({ "prompt": "x".repeat(RECORD_BODY_MAX_BYTES + 4096) });
        let sealed = recorded_request(&body);
        let text = String::from_utf8_lossy(&sealed);
        let tail = &text[text.len().saturating_sub(300)..];
        assert!(tail.contains("-byte request"), "{tail}");
        assert!(
            !tail.contains("response") && !tail.contains("delivered") && !tail.contains("sent"),
            "{tail}"
        );
        assert!(sealed.len() < RECORD_BODY_MAX_BYTES + 1024);
    }

    /// A body under the cap is kept byte-for-byte, with nothing appended.
    #[test]
    fn a_body_under_the_cap_is_kept_whole() {
        let body = serde_json::json!({ "prompt": "hi" });
        assert_eq!(recorded_request(&body), body.to_string().into_bytes());
        let mut response = RecordedBody::default();
        response.push(b"data: hi\n\n");
        assert_eq!(response.seal_response(true), b"data: hi\n\n".to_vec());
    }

    /// A response read to its end names its size; one this app stopped reading
    /// names only a lower bound.
    #[test]
    fn a_stopped_read_states_a_lower_bound_rather_than_a_size() {
        let mut whole = RecordedBody::default();
        whole.push(&vec![b'x'; RECORD_BODY_MAX_BYTES + 10]);
        let whole = String::from_utf8_lossy(&whole.seal_response(true)).to_string();
        assert!(
            whole.contains("-byte response"),
            "{}",
            &whole[whole.len() - 200..]
        );

        let mut stopped = RecordedBody::default();
        stopped.push(&vec![b'x'; RECORD_BODY_MAX_BYTES + 10]);
        let stopped = String::from_utf8_lossy(&stopped.seal_response(false)).to_string();
        assert!(
            stopped.contains("at least that large"),
            "{}",
            &stopped[stopped.len() - 300..]
        );
    }
}
