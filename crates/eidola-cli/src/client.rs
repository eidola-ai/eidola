//! Talking to the Eidola that holds the profile.
//!
//! The local database has one writer. When the app is running, it is that
//! writer, and this process cannot open the profile at all — so it asks the
//! app instead, over the control socket in the data directory. The wire
//! contract is [`eidola_app_core::ipc`]; this is the dialing half of it.
//!
//! ## One request at a time
//!
//! The protocol correlates frames by id and serves them concurrently, because
//! a long-lived caller will want that. This one does not: a command is one
//! operation, so the client sends a request and reads until that request's
//! terminal frame. The id is still checked rather than assumed — a frame
//! answering something else answers nothing here, and is skipped.
//!
//! ## What is bounded and what is not
//!
//! Only the handshake has a deadline. `hello` states two constants and touches
//! neither database nor network, so a connection that has not been greeted in
//! [`HANDSHAKE_TIMEOUT`] is not busy — it is not being served, which is a
//! socket that exists with nothing behind it and the one case the selection
//! rule must never turn into a silent wait. After that, a verb takes as long
//! as the work takes: a turn is minutes, and a client that gave up on it would
//! walk away from an answer the account has already paid for — the app would
//! finish and persist that turn regardless, so giving up buys nothing and
//! loses the delivery.
//!
//! ## Two ceilings, and this side reads the larger one
//!
//! Requests are bounded by [`eidola_app_core::ipc::MAX_FRAME_BYTES`]; answers
//! are bounded by [`eidola_app_core::ipc::MAX_RESPONSE_BYTES`], which is far
//! larger because a result grows with the profile while a request does not.
//! The reader here is built with [`FrameReader::for_responses`] for exactly
//! that reason: reading answers under the request ceiling would refuse a
//! legitimate listing as though the app had malfunctioned.

use std::io;
use std::path::Path;
use std::time::Duration;

use eidola_app_core::error::AppError;
use eidola_app_core::ipc::{
    Call, FrameReader, HelloResult, NO_REQUEST, PROTOCOL_VERSION, ProtocolError, RemoteError,
    Request, Response, ResponseBody, decode_response, encode_line, socket_path,
};
use eidola_app_core::{ChatResult, ChatStreamEvent};
use serde::de::DeserializeOwned;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};

/// How long the handshake may take before the app counts as not answering.
///
/// Deliberately generous for an exchange that is two constants: what this
/// bounds is a socket nothing is serving, not a slow machine.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Why a command failed — the same three-way answer in either mode, plus the
/// one that only a conversation can give.
///
/// [`Failure::App`] is the common case and is *identical* in both modes: the
/// wire carries the typed [`AppError`], so a command renders it and routes on
/// it without knowing which side ran the work.
#[derive(Debug)]
pub enum Failure {
    /// A typed app failure. What an in-process call would have returned.
    App(AppError),
    /// The running app refused the protocol itself — an older or newer build.
    Protocol(ProtocolError),
    /// A failure this build has no type for, with the app's own rendering of
    /// it. Never guessed at, never flattened into an error it resembles.
    Unrecognized { kind: String, message: String },
    /// The conversation broke: the socket died, or said something that is not
    /// a frame.
    Transport { message: String },
    /// The command needs the profile open in *this* process, and a running
    /// Eidola has it. Named rather than dressed up as an app error, because
    /// nothing was attempted and the remedy is about which process should be
    /// running, not about the command's arguments.
    EmbeddedOnly { what: &'static str },
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Failure::App(e) => write!(f, "{e}"),
            Failure::Protocol(e) => write!(f, "{e}"),
            Failure::Unrecognized { message, .. } => write!(f, "{message}"),
            Failure::Transport { message } => write!(f, "{message}"),
            Failure::EmbeddedOnly { what } => write!(
                f,
                "{what} needs the local profile in this process, and Eidola is running"
            ),
        }
    }
}

impl From<AppError> for Failure {
    fn from(e: AppError) -> Self {
        Failure::App(e)
    }
}

impl From<RemoteError> for Failure {
    fn from(e: RemoteError) -> Self {
        match e {
            RemoteError::App(e) => Failure::App(e),
            RemoteError::Protocol(e) => Failure::Protocol(e),
            RemoteError::Unrecognized { kind, message } => Failure::Unrecognized { kind, message },
        }
    }
}

impl From<ProtocolError> for Failure {
    fn from(e: ProtocolError) -> Self {
        Failure::Protocol(e)
    }
}

/// How dialing the control socket went, for the one caller that has to choose
/// what to do about it.
#[derive(Debug)]
pub enum Dial {
    /// Nothing is listening — no socket file, or a file left behind by a
    /// process that is gone. There is no app to ask, so the caller opens the
    /// profile itself.
    NoListener,
    /// The socket exists and accepted a connection, but the handshake went
    /// unanswered. Something is holding the path without serving it.
    NotAccepting,
    /// The conversation started and failed.
    Failed(Failure),
}

/// Whether a failure to reach the socket means "no app is running here".
///
/// Exactly two conditions qualify: no socket file at all, and a socket file
/// nobody is listening on — what a process that died without tidying up
/// leaves behind. Anything else (a permission failure, a path occupied by
/// something that is not a socket) is a condition in its own right and is
/// reported as one; quietly reading it as "the app must not be running" would
/// send the caller off to take a lock it may not be entitled to and report the
/// wrong problem when that failed.
pub fn no_listener(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
    )
}

/// A conversation with the app that holds the profile.
pub struct Client {
    reader: FrameReader<tokio::io::BufReader<OwnedReadHalf>>,
    writer: OwnedWriteHalf,
    next_id: u64,
    app_version: String,
    unread_events: u32,
}

impl Client {
    /// Dial the control socket for a profile and complete the handshake.
    pub async fn connect(data_dir: &Path) -> Result<Client, Dial> {
        let path = socket_path(data_dir);
        let stream = match UnixStream::connect(&path).await {
            Ok(stream) => stream,
            Err(e) if no_listener(&e) => return Err(Dial::NoListener),
            Err(e) => {
                return Err(Dial::Failed(Failure::Transport {
                    message: format!("could not reach {}: {e}", path.display()),
                }));
            }
        };
        let (reader, writer) = stream.into_split();
        let mut client = Client {
            // Answers, not requests: a listing grows with the profile, and
            // holding one to the request ceiling would refuse a legitimate
            // result as though the app had malfunctioned.
            reader: FrameReader::for_responses(tokio::io::BufReader::new(reader)),
            writer,
            next_id: 1,
            app_version: String::new(),
            unread_events: 0,
        };
        let greeting = client.call::<HelloResult>(&Call::Hello);
        let hello = match tokio::time::timeout(HANDSHAKE_TIMEOUT, greeting).await {
            Ok(Ok(hello)) => hello,
            Ok(Err(e)) => return Err(Dial::Failed(e)),
            Err(_elapsed) => return Err(Dial::NotAccepting),
        };
        // The app states the protocol it speaks; a mismatch it did not refuse
        // for us is refused here, naming both sides so whoever reads it knows
        // which half is behind.
        if hello.protocol != PROTOCOL_VERSION {
            return Err(Dial::Failed(Failure::Protocol(
                ProtocolError::UnsupportedProtocol {
                    supported: hello.protocol,
                    requested: PROTOCOL_VERSION,
                },
            )));
        }
        client.app_version = hello.app_version;
        Ok(client)
    }

    /// The version of the app answering this socket.
    pub fn app_version(&self) -> &str {
        &self.app_version
    }

    /// How many streamed events this build could not read.
    ///
    /// Non-zero only against an app newer than this binary, and worth saying
    /// out loud: those events were part of an answer, and dropping them
    /// silently would leave a turn looking complete when it is not.
    pub fn unread_events(&self) -> u32 {
        self.unread_events
    }

    /// Run one non-streaming verb and decode its `end` payload.
    pub async fn call<T: DeserializeOwned>(&mut self, call: &Call) -> Result<T, Failure> {
        let id = self.send(call).await?;
        loop {
            match self.step(id).await? {
                // A verb this build reads as one-shot that the far side chose
                // to narrate. Its answer is still the `end` frame, so read on
                // rather than refuse a newer app's extra courtesy.
                Step::Chunk(_) => continue,
                Step::End(data) => return decode(data),
            }
        }
    }

    /// Run a turn, feeding its chunks to `events`, and decode its result.
    ///
    /// **Dropping this future — Ctrl-C, or the process exiting — costs the
    /// delivery, not the turn.** The app runs the turn on its own runtime, so
    /// losing the connection detaches it rather than cancelling it: the answer
    /// is written, the credential settles, and the post is in the space when
    /// the caller comes back to it. Stating that backwards would be
    /// billing-relevant, since the tokens are spent the moment the request goes
    /// upstream. `crates/eidola-app-core/src/ipc/serve.rs` → What a lost caller
    /// costs carries the whole rule, including the one racy edge (a caller that
    /// vanishes before the turn is handed to the runtime, where nothing runs
    /// and nothing is spent).
    pub async fn chat_stream(
        &mut self,
        call: &Call,
        events: &tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, Failure> {
        let id = self.send(call).await?;
        loop {
            match self.step(id).await? {
                Step::Chunk(data) => match serde_json::from_value::<ChatStreamEvent>(data) {
                    Ok(event) => {
                        let _ = events.send(event);
                    }
                    // An event shape this build does not have. Counted rather
                    // than dropped in silence: it was part of the answer.
                    Err(_) => self.unread_events = self.unread_events.saturating_add(1),
                },
                Step::End(data) => return decode(data),
            }
        }
    }

    /// Write a request frame and take its id.
    async fn send(&mut self, call: &Call) -> Result<u64, Failure> {
        let id = self.next_id;
        self.next_id += 1;
        let line = encode_line(&Request::new(id, call));
        let sending = |e: io::Error| transport(format!("could not send `{}`: {e}", call.verb()));
        self.writer.write_all(&line).await.map_err(sending)?;
        self.writer.flush().await.map_err(sending)?;
        Ok(id)
    }

    /// The next frame that says something about request `id`.
    ///
    /// Frames for another id are skipped: this connection has one request in
    /// flight, so such a frame answers nothing here. A refusal carrying
    /// [`NO_REQUEST`] is the exception — it is the app saying the line it just
    /// read was not a frame, which can only be about ours.
    async fn step(&mut self, id: u64) -> Result<Step, Failure> {
        loop {
            let frame = self.read_frame().await?;
            match frame.body {
                ResponseBody::Chunk { data } if frame.id == id => return Ok(Step::Chunk(data)),
                ResponseBody::End { data } if frame.id == id => return Ok(Step::End(data)),
                ResponseBody::Err { error } if frame.id == id || frame.id == NO_REQUEST => {
                    return Err(error.to_remote().into());
                }
                _ => continue,
            }
        }
    }

    /// Read one response frame off the wire.
    async fn read_frame(&mut self) -> Result<Response, Failure> {
        match self.reader.next_line().await {
            Ok(Some(line)) => Ok(decode_response(line)?),
            // The app went away mid-request. Nothing was mis-said, so this is
            // not a protocol fault — but it is not an answer either, and the
            // caller must never be told the operation succeeded.
            Ok(None) => Err(transport(
                "the running Eidola closed the connection before answering".into(),
            )),
            Err(e) => Err(Failure::Protocol(e)),
        }
    }
}

/// One frame that said something about the request in flight.
enum Step {
    Chunk(serde_json::Value),
    End(serde_json::Value),
}

/// Decode an `end` payload into the verb's result type.
fn decode<T: DeserializeOwned>(data: serde_json::Value) -> Result<T, Failure> {
    serde_json::from_value(data).map_err(|e| transport(format!("unreadable answer: {e}")))
}

fn transport(message: String) -> Failure {
    Failure::Transport { message }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_app_core::ipc::{
        AccountPricesResult, DefaultModelResult, Request, WireError, decode_request,
    };

    /// A stand-in for the app: one connection, every request answered from
    /// `script`. Written with the protocol's own codec rather than hand-rolled
    /// bytes, so what it proves is the contract and not a spelling.
    fn serve<F>(dir: &std::path::Path, script: F)
    where
        F: Fn(&Request) -> Vec<Response> + Send + 'static,
    {
        let listener = tokio::net::UnixListener::bind(socket_path(dir)).expect("bind");
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.expect("accept");
            let (reader, mut writer) = stream.into_split();
            let mut frames = FrameReader::new(tokio::io::BufReader::new(reader));
            while let Ok(Some(line)) = frames.next_line().await {
                let request = decode_request(line).expect("a frame");
                for response in script(&request) {
                    if writer.write_all(&encode_line(&response)).await.is_err() {
                        return;
                    }
                }
            }
        });
    }

    /// The ordinary app: greets, and answers the two verbs these tests ask for.
    fn ordinary(request: &Request) -> Vec<Response> {
        match request.verb.as_str() {
            "hello" => vec![Response::end(
                request.id,
                &HelloResult {
                    protocol: PROTOCOL_VERSION,
                    app_version: "9.9.9".into(),
                },
            )],
            "chat.default_model" => vec![Response::end(
                request.id,
                &DefaultModelResult {
                    model: "m@local".into(),
                },
            )],
            other => vec![Response::err(
                request.id,
                WireError::from_protocol(&ProtocolError::UnknownVerb {
                    verb: other.to_string(),
                }),
            )],
        }
    }

    #[test]
    fn only_a_missing_or_unanswered_socket_means_no_app() {
        for kind in [io::ErrorKind::NotFound, io::ErrorKind::ConnectionRefused] {
            assert!(no_listener(&io::Error::from(kind)), "{kind:?}");
        }
        for kind in [
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::TimedOut,
            io::ErrorKind::Other,
        ] {
            assert!(
                !no_listener(&io::Error::from(kind)),
                "{kind:?} is a condition of its own, not an absent app"
            );
        }
    }

    #[tokio::test]
    async fn no_socket_file_is_no_listener() {
        let dir = tempfile::tempdir().expect("tempdir");
        match Client::connect(dir.path()).await {
            Err(Dial::NoListener) => {}
            other => panic!("unexpected: {:?}", other.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn a_socket_nobody_listens_on_is_no_listener() {
        let dir = tempfile::tempdir().expect("tempdir");
        let stale = std::os::unix::net::UnixListener::bind(socket_path(dir.path())).expect("bind");
        drop(stale);
        match Client::connect(dir.path()).await {
            Err(Dial::NoListener) => {}
            other => panic!("the file outlived its listener: {:?}", other.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn the_handshake_names_the_app_and_the_verbs_answer() {
        let dir = tempfile::tempdir().expect("tempdir");
        serve(dir.path(), ordinary);
        let mut client = Client::connect(dir.path()).await.expect("connect");
        assert_eq!(client.app_version(), "9.9.9");
        let model: DefaultModelResult = client
            .call(&Call::ChatDefaultModel)
            .await
            .expect("the verb answers");
        assert_eq!(model.model, "m@local");
    }

    #[tokio::test(start_paused = true)]
    async fn a_socket_that_never_greets_is_not_accepting() {
        let dir = tempfile::tempdir().expect("tempdir");
        let listener = tokio::net::UnixListener::bind(socket_path(dir.path())).expect("bind");
        tokio::spawn(async move {
            let _held = listener.accept().await.expect("accept");
            // Holds the connection open and says nothing — a listener that
            // takes the call and never answers it.
            std::future::pending::<()>().await;
        });
        match Client::connect(dir.path()).await {
            Err(Dial::NotAccepting) => {}
            other => panic!(
                "a silent app must never be a silent wait: {:?}",
                other.map(|_| ())
            ),
        }
    }

    #[tokio::test]
    async fn an_app_speaking_another_protocol_names_both_versions() {
        let dir = tempfile::tempdir().expect("tempdir");
        serve(dir.path(), |request| {
            vec![Response::end(
                request.id,
                &HelloResult {
                    protocol: PROTOCOL_VERSION + 1,
                    app_version: "9.9.9".into(),
                },
            )]
        });
        match Client::connect(dir.path()).await {
            Err(Dial::Failed(Failure::Protocol(ProtocolError::UnsupportedProtocol {
                supported,
                requested,
            }))) => {
                assert_eq!(supported, PROTOCOL_VERSION + 1);
                assert_eq!(requested, PROTOCOL_VERSION);
            }
            other => panic!("unexpected: {:?}", other.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn an_app_that_refuses_our_protocol_is_reported_as_it_said_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        serve(dir.path(), |request| {
            vec![Response::err(
                request.id,
                WireError::from_protocol(&ProtocolError::UnsupportedProtocol {
                    supported: 7,
                    requested: PROTOCOL_VERSION,
                }),
            )]
        });
        match Client::connect(dir.path()).await {
            Err(Dial::Failed(Failure::Protocol(ProtocolError::UnsupportedProtocol {
                supported,
                ..
            }))) => assert_eq!(supported, 7),
            other => panic!("unexpected: {:?}", other.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn an_answer_larger_than_the_request_ceiling_still_arrives() {
        use eidola_app_core::ipc::{MAX_FRAME_BYTES, WalletRecoverResult};
        let dir = tempfile::tempdir().expect("tempdir");
        // A listing that grew with the profile: past the *request* ceiling,
        // well inside the answer one. Requests do not grow like this and
        // answers do, which is the whole reason the two bounds differ.
        let recovered: Vec<String> = (0..50_000).map(|i| format!("nonce-{i:026}")).collect();
        let expected = recovered.len();
        assert!(
            encode_line(&Response::end(
                1,
                &WalletRecoverResult {
                    recovered: recovered.clone()
                }
            ))
            .len()
                > MAX_FRAME_BYTES,
            "the fixture has to actually exceed the request bound to prove anything"
        );

        serve(dir.path(), move |request| match request.verb.as_str() {
            "hello" => vec![Response::end(
                request.id,
                &HelloResult {
                    protocol: PROTOCOL_VERSION,
                    app_version: "9.9.9".into(),
                },
            )],
            _ => vec![Response::end(
                request.id,
                &WalletRecoverResult {
                    recovered: recovered.clone(),
                },
            )],
        });
        let mut client = Client::connect(dir.path()).await.expect("connect");
        let listing: WalletRecoverResult = client
            .call(&Call::WalletRecover)
            .await
            .expect("a large listing is an answer, not a malfunction");
        assert_eq!(listing.recovered.len(), expected);
        assert_eq!(
            listing.recovered[expected - 1],
            format!("nonce-{:026}", expected - 1)
        );
    }

    #[tokio::test]
    async fn a_verb_the_app_does_not_have_costs_one_command() {
        let dir = tempfile::tempdir().expect("tempdir");
        serve(dir.path(), ordinary);
        let mut client = Client::connect(dir.path()).await.expect("connect");
        let refused = client
            .call::<AccountPricesResult>(&Call::AccountPrices)
            .await;
        match refused {
            Err(Failure::Protocol(ProtocolError::UnknownVerb { verb })) => {
                assert_eq!(verb, "account.prices");
            }
            other => panic!("unexpected: {:?}", other.map(|_| ())),
        }
        // The connection is still good: one refusal is one request's.
        let model: DefaultModelResult = client.call(&Call::ChatDefaultModel).await.expect("still");
        assert_eq!(model.model, "m@local");
    }

    #[tokio::test]
    async fn a_typed_app_failure_arrives_typed() {
        let dir = tempfile::tempdir().expect("tempdir");
        serve(dir.path(), |request| match request.verb.as_str() {
            "hello" => vec![Response::end(
                request.id,
                &HelloResult {
                    protocol: PROTOCOL_VERSION,
                    app_version: "9.9.9".into(),
                },
            )],
            _ => vec![Response::err(
                request.id,
                WireError::from_app_error(&AppError::InsufficientBalance {
                    available: 3,
                    required: 40,
                }),
            )],
        });
        let mut client = Client::connect(dir.path()).await.expect("connect");
        match client
            .call::<AccountPricesResult>(&Call::AccountPrices)
            .await
        {
            Err(Failure::App(AppError::InsufficientBalance {
                available,
                required,
            })) => {
                assert_eq!((available, required), (3, 40));
            }
            other => panic!("the variant must survive: {:?}", other.map(|_| ())),
        }
    }

    #[tokio::test]
    async fn a_turn_streams_its_events_and_ends_with_its_result() {
        let dir = tempfile::tempdir().expect("tempdir");
        serve(dir.path(), |request| match request.verb.as_str() {
            "hello" => vec![Response::end(
                request.id,
                &HelloResult {
                    protocol: PROTOCOL_VERSION,
                    app_version: "9.9.9".into(),
                },
            )],
            _ => vec![
                // A frame answering another request: this connection has one
                // in flight, so it must be skipped rather than acted on.
                Response::chunk(
                    request.id + 500,
                    serde_json::json!({"type": "content_delta", "text": "no"}),
                ),
                Response::chunk(
                    request.id,
                    serde_json::to_value(ChatStreamEvent::ReasoningDelta("hmm".into())).unwrap(),
                ),
                Response::chunk(
                    request.id,
                    serde_json::to_value(ChatStreamEvent::ContentDelta("hello".into())).unwrap(),
                ),
                // An event shape this build has no variant for.
                Response::chunk(request.id, serde_json::json!({"type": "aura_delta"})),
                Response::end(
                    request.id,
                    &serde_json::json!({
                        "space_id": "sp1",
                        "content": "hello",
                        "model": "m@local",
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "credits_charged": 2,
                        "truncated": false,
                        "declined": null
                    }),
                ),
            ],
        });
        let mut client = Client::connect(dir.path()).await.expect("connect");
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let call = Call::ChatStream {
            prompt: "hi".into(),
            model: None,
            space_id: None,
        };
        let result = client.chat_stream(&call, &tx).await.expect("a turn");
        drop(tx);
        let mut seen = Vec::new();
        while let Some(event) = rx.recv().await {
            seen.push(event);
        }
        assert_eq!(seen.len(), 2, "the readable events, and only those");
        assert!(matches!(seen[0], ChatStreamEvent::ReasoningDelta(_)));
        assert!(matches!(seen[1], ChatStreamEvent::ContentDelta(_)));
        assert_eq!(result.content, "hello");
        assert_eq!(
            client.unread_events(),
            1,
            "an event this build cannot read is counted, never lost in silence"
        );
    }
}
