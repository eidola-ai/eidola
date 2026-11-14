//! The traffic shape of a streamed chat response, both halves: the writer
//! that cuts server-sent events into frames of one size, and the reader that
//! takes them back out, padding and all.
//!
//! A network observer of an encrypted stream sees each write's size and
//! arrival time. Written one event per token, those trace the tokens' lengths
//! and the model's cadence. Written as frames of one size on a fixed tick
//! ([`Cadence`]), whatever is waiting, they trace neither. The gateway writes
//! its client's stream this way, and an inference node writes its stream to
//! the gateway the same way (`engine_protocol::STREAM_CADENCE`); the gateway
//! reads the node's frames with [`StreamReader`]. When the frames go out is
//! the writer's own (each side drives a [`Framer`] from its runtime's clock);
//! what goes in them is decided here, once.
//!
//! **Padding is event-stream syntax every reader already ignores.** When the
//! waiting bytes end before the frame does, the rest is a comment event
//! (`:` + spaces + a blank line), which dispatches nothing. A remainder of one
//! or two bytes is too short for a comment and goes inside the last event as
//! trailing whitespace after its JSON value, which JSON permits; after the
//! stream's last event (`[DONE]`), it is blank lines. Nothing is added inside
//! a JSON value.
//!
//! **Every other response** is padded with trailing spaces to a size bucket
//! ([`body_bucket`], [`pad_body`]); [`unpadded`] is the reader's half for a
//! body that is not JSON.

use std::collections::VecDeque;
use std::time::Duration;

/// The smallest frame: one must hold a whole comment event (`:\n\n`) with
/// room to spare.
pub const MIN_FRAME_BYTES: usize = 16;

/// The smallest size a padded body is padded to.
pub const MIN_BODY_BUCKET: usize = 4096;

/// How a stream is framed: every write exactly [`Cadence::frame_bytes`] long,
/// one every [`Cadence::tick`]. Only [`Cadence::new`] makes one, so a frame
/// too small to pad, or a cadence with no tick, cannot exist; built in a
/// `const`, either is a compile error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cadence {
    frame_bytes: usize,
    tick: Duration,
}

impl Cadence {
    /// # Panics
    /// If `frame_bytes` is below [`MIN_FRAME_BYTES`] or `tick` is zero.
    pub const fn new(frame_bytes: usize, tick: Duration) -> Self {
        assert!(
            frame_bytes >= MIN_FRAME_BYTES,
            "a frame must hold a padding comment"
        );
        assert!(!tick.is_zero(), "a cadence needs a tick");
        Self { frame_bytes, tick }
    }

    /// The size of every frame, in bytes.
    pub const fn frame_bytes(self) -> usize {
        self.frame_bytes
    }

    /// The interval between frames.
    pub const fn tick(self) -> Duration {
        self.tick
    }

    /// The bytes a stream at this cadence carries in one second, padding
    /// included: its constant rate, and the most event bytes it can deliver.
    pub const fn bytes_per_second(self) -> u128 {
        self.frame_bytes as u128 * 1_000_000_000 / self.tick.as_nanos()
    }
}

/// One server-sent event of a chat response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEvent {
    /// A data event carrying a JSON payload (a completion chunk, or the
    /// gateway's metadata event): compact `serde_json` output.
    Json(String),
    /// An `event: error` event carrying a JSON error body: a node's stream
    /// that failed partway ends with one.
    Error(String),
    /// The `[DONE]` sentinel.
    Done,
}

impl StreamEvent {
    /// Whether nothing follows this event in a stream: `[DONE]`, or an error.
    pub fn ends_stream(&self) -> bool {
        !matches!(self, StreamEvent::Json(_))
    }

    /// The event as it goes on the wire.
    pub fn encode(&self) -> Encoded {
        fn data_lines(out: &mut Vec<u8>, json: &str) {
            // One `data:` field per line, as the format joins them back;
            // compact JSON has none, so this is one field.
            for line in json.split('\n') {
                out.extend_from_slice(b"data: ");
                out.extend_from_slice(line.trim_end_matches('\r').as_bytes());
                out.push(b'\n');
            }
            out.push(b'\n');
        }
        let mut bytes = Vec::new();
        let json = match self {
            StreamEvent::Json(json) => {
                bytes.reserve(json.len() + 8);
                data_lines(&mut bytes, json);
                true
            }
            StreamEvent::Error(json) => {
                bytes.reserve(json.len() + 21);
                bytes.extend_from_slice(b"event: error\n");
                data_lines(&mut bytes, json);
                true
            }
            StreamEvent::Done => {
                bytes.extend_from_slice(b"data: [DONE]\n\n");
                false
            }
        };
        Encoded {
            bytes,
            json,
            ends_stream: self.ends_stream(),
        }
    }
}

/// An event as it goes on the wire ([`StreamEvent::encode`]).
#[derive(Debug, Clone)]
pub struct Encoded {
    bytes: Vec<u8>,
    /// Whether its data is JSON, which can take trailing whitespace.
    json: bool,
    ends_stream: bool,
}

impl Encoded {
    /// Its length on the wire.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Never true: every event has a blank line at least.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Encodes events and cuts them into frames of one size.
///
/// Pure: what goes in a frame is decided here; when it goes out is the
/// writer's (one [`Framer::frame`] per tick).
#[derive(Debug)]
pub struct Framer {
    frame_bytes: usize,
    /// Encoded events not yet wholly in a frame, the front one possibly
    /// started: `front_sent` of its bytes are already out. The queue always
    /// ends at an event boundary, since events go in whole.
    events: VecDeque<Vec<u8>>,
    front_sent: usize,
    /// Event bytes not yet in a frame.
    queued: usize,
    /// Whether the last event pushed can take trailing whitespace inside its
    /// data value.
    last_is_json: bool,
    done: bool,
}

impl Framer {
    pub fn new(cadence: Cadence) -> Self {
        Self {
            frame_bytes: cadence.frame_bytes,
            events: VecDeque::new(),
            front_sent: 0,
            queued: 0,
            last_is_json: false,
            done: false,
        }
    }

    /// Queue one event. Nothing is queued after an event that ends the
    /// stream.
    pub fn push(&mut self, event: &StreamEvent) {
        self.push_encoded(event.encode());
    }

    /// Queue an event already encoded.
    pub fn push_encoded(&mut self, encoded: Encoded) {
        if self.done {
            return;
        }
        self.queued += encoded.bytes.len();
        self.events.push_back(encoded.bytes);
        self.last_is_json = encoded.json;
        self.done = encoded.ends_stream;
    }

    /// Event bytes waiting for a frame.
    pub fn backlog(&self) -> usize {
        self.queued
    }

    /// Events not yet wholly in a frame, a started one included.
    pub fn queued_events(&self) -> usize {
        self.events.len()
    }

    /// Whether an event that ends the stream has been queued.
    pub fn done(&self) -> bool {
        self.done
    }

    /// The next frame: exactly the cadence's frame size, whatever is waiting.
    pub fn frame(&mut self) -> Vec<u8> {
        let size = self.frame_bytes;
        let mut frame = Vec::with_capacity(size);
        while frame.len() < size {
            let Some(front) = self.events.front() else {
                break;
            };
            // Possibly mid-event; the next frame continues it.
            let rest = &front[self.front_sent..];
            let take = rest.len().min(size - frame.len());
            frame.extend_from_slice(&rest[..take]);
            self.front_sent += take;
            self.queued -= take;
            if self.front_sent == front.len() {
                self.events.pop_front();
                self.front_sent = 0;
            }
        }
        if frame.len() == size {
            return frame;
        }
        // Everything waiting fitted, and it ended at an event boundary.
        let room = size - frame.len();
        if room >= 3 {
            frame.push(b':');
            frame.resize(size - 2, b' ');
            frame.extend_from_slice(b"\n\n");
        } else if !frame.is_empty() && self.last_is_json {
            // Too little room for a comment: trailing whitespace on the last
            // event's data line, before the blank line that ends it. The
            // frame is nearly full, so those two bytes are that event's.
            let end = frame.len() - 2;
            let tail = frame.split_off(end);
            frame.resize(end + room, b' ');
            frame.extend_from_slice(&tail);
        } else {
            // Only after `[DONE]`: empty lines, which dispatch nothing.
            frame.resize(size, b'\n');
        }
        debug_assert_eq!(frame.len(), size);
        frame
    }
}

/// An event block a reader cannot read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unreadable {
    /// The block is not UTF-8. It is refused, never lossily repaired: a lossy
    /// decode would pass on altered output as if the writer had sent it.
    NotUtf8,
    /// The block names an event type no writer here sends.
    UnknownEvent,
}

/// Takes events back out of a framed stream, however its bytes arrive.
///
/// Padding dispatches nothing: a comment block is skipped, whitespace after
/// a JSON value stays in its data (a JSON parse ignores it), and the blank
/// lines after `[DONE]` are never read, since nothing follows it.
#[derive(Debug, Default)]
pub struct StreamReader {
    buffer: Vec<u8>,
}

impl StreamReader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bytes as they arrived.
    pub fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// The next complete event, if one has arrived.
    pub fn next_event(&mut self) -> Result<Option<StreamEvent>, Unreadable> {
        loop {
            let Some(end) = self.buffer.windows(2).position(|w| w == b"\n\n") else {
                return Ok(None);
            };
            let block: Vec<u8> = self.buffer.drain(..end + 2).collect();
            let text = std::str::from_utf8(&block[..end]).map_err(|_| Unreadable::NotUtf8)?;
            let mut event = None;
            let mut data: Option<String> = None;
            for line in text.lines() {
                let line = line.strip_suffix('\r').unwrap_or(line);
                if let Some(value) = line.strip_prefix("event:") {
                    event = Some(value.trim_start());
                } else if let Some(value) = line.strip_prefix("data:") {
                    let value = value.strip_prefix(' ').unwrap_or(value);
                    match &mut data {
                        Some(d) => {
                            d.push('\n');
                            d.push_str(value);
                        }
                        None => data = Some(value.to_string()),
                    }
                }
            }
            // A block with no data (a comment, padding) dispatches nothing,
            // as the format defines.
            let Some(data) = data else { continue };
            return match event {
                None | Some("message") if data == "[DONE]" => Ok(Some(StreamEvent::Done)),
                None | Some("message") => Ok(Some(StreamEvent::Json(data))),
                Some("error") => Ok(Some(StreamEvent::Error(data))),
                Some(_) => Err(Unreadable::UnknownEvent),
            };
        }
    }
}

/// The size a body of `len` bytes is padded to: the next power of two, at
/// least [`MIN_BODY_BUCKET`].
pub const fn body_bucket(len: usize) -> usize {
    let len = if len < MIN_BODY_BUCKET {
        MIN_BODY_BUCKET
    } else {
        len
    };
    len.next_power_of_two()
}

/// `body` with trailing spaces to [`body_bucket`] of its length. Trailing
/// whitespace leaves a JSON value unchanged.
pub fn pad_body(mut body: Vec<u8>) -> Vec<u8> {
    let size = body_bucket(body.len());
    body.resize(size, b' ');
    body
}

/// A padded body without its padding: what a reader compares a body that is
/// not JSON against.
pub fn unpadded(body: &[u8]) -> &[u8] {
    let end = body.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
    &body[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TICK: Duration = Duration::from_millis(50);

    /// A deterministic generator, so the sweeps below are reproducible.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// A stream's events: chunks of varied JSON (escapes, multi-byte text,
    /// sizes from a few bytes to several frames), then an end.
    fn events(rng: &mut Lcg) -> Vec<StreamEvent> {
        let pieces = [
            "a", "é", "\\n", "\\\"", "日本", " ", "{", "}", "data: x", ":",
        ];
        let mut out = Vec::new();
        for _ in 0..rng.below(12) {
            let mut text = String::new();
            for _ in 0..rng.below(80) {
                text.push_str(pieces[rng.below(pieces.len() as u64) as usize]);
            }
            out.push(StreamEvent::Json(format!(r#"{{"delta":"{text}"}}"#)));
        }
        out.push(match rng.below(3) {
            0 => StreamEvent::Error(r#"{"error":{"type":"engine_unavailable"}}"#.to_string()),
            _ => StreamEvent::Done,
        });
        out
    }

    fn frames(frame_bytes: usize, events: &[StreamEvent]) -> Vec<Vec<u8>> {
        let mut framer = Framer::new(Cadence::new(frame_bytes, TICK));
        for e in events {
            framer.push(e);
        }
        let mut out = vec![framer.frame()];
        while framer.backlog() > 0 {
            out.push(framer.frame());
        }
        out
    }

    /// What the reader gives back from `bytes`, split into reads of the
    /// sizes `rng` picks, up to the event that ends the stream.
    fn read_back(bytes: &[u8], rng: &mut Lcg) -> Result<Vec<StreamEvent>, Unreadable> {
        let mut reader = StreamReader::new();
        let mut out = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let n = (1 + rng.below(700) as usize).min(bytes.len() - at);
            reader.push(&bytes[at..at + n]);
            at += n;
            while let Some(event) = reader.next_event()? {
                let ends = event.ends_stream();
                out.push(event);
                if ends {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }

    /// JSON data read back equals the original once the padding a JSON
    /// parse ignores is set aside.
    fn without_trailing_padding(events: Vec<StreamEvent>) -> Vec<StreamEvent> {
        events
            .into_iter()
            .map(|e| match e {
                StreamEvent::Json(s) => StreamEvent::Json(s.trim_end_matches(' ').to_string()),
                StreamEvent::Error(s) => StreamEvent::Error(s.trim_end_matches(' ').to_string()),
                StreamEvent::Done => StreamEvent::Done,
            })
            .collect()
    }

    #[test]
    fn the_reader_takes_back_exactly_what_the_framer_was_given() {
        let mut rng = Lcg(7);
        for frame_bytes in MIN_FRAME_BYTES..120 {
            for _ in 0..40 {
                let events = events(&mut rng);
                let frames = frames(frame_bytes, &events);
                assert!(frames.iter().all(|f| f.len() == frame_bytes));
                let bytes = frames.concat();
                let read = read_back(&bytes, &mut rng).unwrap();
                assert_eq!(without_trailing_padding(read), events, "{frame_bytes}");
            }
        }
    }

    #[test]
    fn a_frames_count_is_a_function_of_the_encoded_length_alone() {
        let mut rng = Lcg(11);
        for _ in 0..500 {
            let frame_bytes = MIN_FRAME_BYTES + rng.below(200) as usize;
            let events = events(&mut rng);
            let total: usize = events.iter().map(|e| e.encode().len()).sum();
            assert_eq!(
                frames(frame_bytes, &events).len(),
                total.div_ceil(frame_bytes).max(1)
            );
        }
    }

    #[test]
    fn a_one_or_two_byte_remainder_pads_inside_the_last_json_event() {
        for event in [
            StreamEvent::Json(r#"{"a":1}"#.to_string()),
            StreamEvent::Error(r#"{"a":1}"#.to_string()),
        ] {
            let len = event.encode().len();
            for room in [1, 2] {
                let mut framer = Framer::new(Cadence::new(len + room, TICK));
                framer.push(&event);
                let frame = framer.frame();
                assert_eq!(frame.len(), len + room);
                let tail = format!("}}{}\n\n", " ".repeat(room));
                assert!(frame.ends_with(tail.as_bytes()));
                let mut reader = StreamReader::new();
                reader.push(&frame);
                let read = reader.next_event().unwrap().unwrap();
                let (StreamEvent::Json(data) | StreamEvent::Error(data)) = read else {
                    panic!("{read:?}");
                };
                let value: serde_json::Value = serde_json::from_str(&data).unwrap();
                assert_eq!(value["a"], 1);
            }
        }
    }

    #[test]
    fn a_frame_with_nothing_waiting_is_one_comment() {
        let mut framer = Framer::new(Cadence::new(32, TICK));
        let frame = framer.frame();
        assert_eq!(frame.len(), 32);
        assert_eq!(&frame[..1], b":");
        assert!(frame.ends_with(b"\n\n"));
        let mut reader = StreamReader::new();
        reader.push(&frame);
        assert_eq!(reader.next_event(), Ok(None));
    }

    #[test]
    fn nothing_is_queued_after_the_stream_ends() {
        let mut framer = Framer::new(Cadence::new(64, TICK));
        framer.push(&StreamEvent::Error("{}".to_string()));
        let queued = framer.backlog();
        framer.push(&StreamEvent::Json("{}".to_string()));
        framer.push(&StreamEvent::Done);
        assert!(framer.done());
        assert_eq!(framer.backlog(), queued);
    }

    #[test]
    fn the_reader_refuses_what_no_writer_sends() {
        let mut reader = StreamReader::new();
        reader.push(b"event: surprise\ndata: {}\n\n");
        assert_eq!(reader.next_event(), Err(Unreadable::UnknownEvent));
        let mut reader = StreamReader::new();
        reader.push(b"data: \xff\n\n");
        assert_eq!(reader.next_event(), Err(Unreadable::NotUtf8));
        // A block with no data is nothing, whatever it names.
        let mut reader = StreamReader::new();
        reader.push(b"event: surprise\n\n: comment\n\ndata: [DONE]\n\n");
        assert_eq!(reader.next_event(), Ok(Some(StreamEvent::Done)));
    }

    #[test]
    fn bodies_pad_to_a_power_of_two_and_read_back() {
        assert_eq!(body_bucket(0), MIN_BODY_BUCKET);
        assert_eq!(body_bucket(MIN_BODY_BUCKET), MIN_BODY_BUCKET);
        assert_eq!(body_bucket(MIN_BODY_BUCKET + 1), 2 * MIN_BODY_BUCKET);
        let padded = pad_body(b"ok".to_vec());
        assert_eq!(padded.len(), MIN_BODY_BUCKET);
        assert_eq!(unpadded(&padded), b"ok");
        assert_eq!(unpadded(b"   "), b"");
        let json = pad_body(br#"{"a":"b "}"#.to_vec());
        let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(value["a"], "b ");
    }

    #[test]
    fn a_cadence_states_its_rate() {
        assert_eq!(Cadence::new(2048, TICK).bytes_per_second(), 40_960);
    }

    #[test]
    #[should_panic(expected = "a frame must hold a padding comment")]
    fn a_frame_too_small_to_pad_cannot_be_made() {
        let _ = Cadence::new(MIN_FRAME_BYTES - 1, TICK);
    }
}
