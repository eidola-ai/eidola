//! The local control protocol, end to end over an in-memory pipe.
//!
//! `serve_connection` takes any reader and writer, so everything below runs
//! with no socket, no privileges and no listener — the transport's own
//! concerns (binding, peer authentication) belong to the process that owns the
//! socket and are tested there. What is tested here is the conversation: the
//! handshake, the verbs, and — at least as important — every way a caller can
//! say something wrong.
//!
//! The adversarial half is the point. A local socket is spoken to by whatever
//! is running as the user, which includes a half-written client, a shell
//! pipeline, and a stray `cat`. Each of those must cost **one request**, typed
//! and correlated, and leave the connection standing:
//!
//! | What arrives | What comes back |
//! |---|---|
//! | a verb before `hello` | `HandshakeRequired`, connection lives |
//! | another protocol version | `UnsupportedProtocol` naming both, connection lives |
//! | a verb this build does not have | `UnknownVerb` on its own id, connection lives |
//! | parameters of the wrong shape | `BadParams` naming the verb, connection lives |
//! | a line that is not JSON | `MalformedFrame` on `NO_REQUEST`, connection lives |
//! | a line past the frame ceiling | `FrameTooLarge`, then the connection closes |
//!
//! The last one is the only fatal case, and for a structural reason: the reader
//! abandoned a line part-way through, so it no longer knows where the next
//! frame begins.

mod chat_harness;

use std::sync::Arc;

use chat_harness::{ChatBehavior, MODEL, MockConfig, core_for, with_account};
use eidola_app_core::AppCore;
use eidola_app_core::ipc::{
    Call, HelloResult, MAX_FRAME_BYTES, NO_REQUEST, PROTOCOL_VERSION, ProtocolError, RemoteError,
    Request, Response, ResponseBody, SpacesListResult, WalletCredentialsResult, decode_response,
    encode_line, serve_connection,
};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, DuplexStream};

/// The version string the served connection reports as the app's. Arbitrary,
/// and deliberately not this crate's — the socket's owner supplies it.
const APP_VERSION: &str = "9.9.9-test";

/// The in-memory pipe's buffer. Smaller than the frame ceiling on purpose, so
/// the oversized-frame test genuinely exercises a reader consuming a line
/// across many buffer-fulls rather than one big slice.
const PIPE_BUFFER: usize = 64 * 1024;

/// `AppCore` owns its runtime and must be dropped off any other one; every
/// test body therefore runs on a plain OS thread, as the rest of the suite
/// does.
fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

// ---------------------------------------------------------------------------
// A client, of the crudest possible kind: it writes bytes and reads lines.
// ---------------------------------------------------------------------------

struct Client {
    writer: DuplexStream,
    reader: BufReader<DuplexStream>,
    next_id: u64,
}

/// The next frame, or `None` at the end of the stream. The bare-pipe twin of
/// [`Client::recv`], for the tests that build their own halves.
async fn client_frame(reader: &mut BufReader<DuplexStream>) -> Option<Response> {
    read_frame(reader).await
}

async fn read_frame(reader: &mut BufReader<DuplexStream>) -> Option<Response> {
    let mut line = String::new();
    let read = reader.read_line(&mut line).await.expect("read a frame");
    if read == 0 {
        return None;
    }
    Some(decode_response(line.trim_end().as_bytes()).expect("the server wrote a frame"))
}

impl Client {
    /// Open a connection to a served core. Must be called inside the core's
    /// runtime.
    fn connect(core: &Arc<AppCore>) -> Client {
        Client::connect_sized(core, PIPE_BUFFER)
    }

    /// The same, with an explicit pipe size — a small one is how a test plays
    /// a caller whose socket buffer fills.
    fn connect_sized(core: &Arc<AppCore>, pipe: usize) -> Client {
        let (client_writes, server_reads) = tokio::io::duplex(pipe);
        let (server_writes, client_reads) = tokio::io::duplex(pipe);
        tokio::spawn(serve_connection(
            Arc::clone(core),
            APP_VERSION.to_string(),
            server_reads,
            server_writes,
        ));
        Client {
            writer: client_writes,
            reader: BufReader::new(client_reads),
            next_id: 1,
        }
    }

    /// Send a typed call; answers with the id it was given.
    async fn send(&mut self, call: &Call) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.send_raw(&encode_line(&Request::new(id, call))).await;
        id
    }

    /// Send a typed call under an id the caller chooses.
    async fn send_with_id(&mut self, id: u64, call: &Call) {
        self.send_raw(&encode_line(&Request::new(id, call))).await;
    }

    /// Send bytes exactly as given — the door every malformed case comes in by.
    async fn send_raw(&mut self, bytes: &[u8]) {
        self.writer.write_all(bytes).await.expect("write");
        self.writer.flush().await.expect("flush");
    }

    async fn recv(&mut self) -> Option<Response> {
        read_frame(&mut self.reader).await
    }

    /// The next frame, which must be there.
    async fn expect_frame(&mut self) -> Response {
        self.recv()
            .await
            .expect("a frame, not the end of the stream")
    }

    /// Complete the handshake and answer with what the server said about
    /// itself.
    async fn hello(&mut self) -> HelloResult {
        let id = self.send(&Call::Hello).await;
        let frame = self.expect_frame().await;
        assert_eq!(frame.id, id);
        serde_json::from_value(end_of(&frame)).expect("a hello result")
    }

    /// Run a call to its terminal frame, collecting any chunks on the way.
    async fn call(&mut self, call: &Call) -> (Vec<serde_json::Value>, Outcome) {
        let id = self.send(call).await;
        let mut chunks = Vec::new();
        loop {
            let frame = self.expect_frame().await;
            assert_eq!(frame.id, id, "a frame answering a request nobody made");
            match frame.body {
                ResponseBody::Chunk { data } => chunks.push(data),
                ResponseBody::End { data } => return (chunks, Outcome::End(data)),
                ResponseBody::Err { error } => {
                    return (chunks, Outcome::Err(error.to_remote()));
                }
            }
        }
    }

    /// Run a call that must succeed, and deserialize its result.
    async fn ok<T: serde::de::DeserializeOwned>(&mut self, call: &Call) -> T {
        match self.call(call).await {
            (_, Outcome::End(data)) => serde_json::from_value(data).expect("the verb's result"),
            (_, Outcome::Err(e)) => panic!("expected `{}` to succeed: {e}", call.verb()),
        }
    }

    /// Run a call that must be refused by the protocol itself.
    async fn refused(&mut self, call: &Call) -> ProtocolError {
        match self.call(call).await {
            (_, Outcome::Err(RemoteError::Protocol(e))) => e,
            other => panic!("expected a protocol refusal, got {other:?}"),
        }
    }
}

#[derive(Debug)]
enum Outcome {
    End(serde_json::Value),
    Err(RemoteError),
}

fn end_of(frame: &Response) -> serde_json::Value {
    match &frame.body {
        ResponseBody::End { data } => data.clone(),
        other => panic!("expected an end frame, got {other:?}"),
    }
}

/// A served core plus the mock upstream and tempdir keeping it alive.
fn served(config: MockConfig) -> (chat_harness::MockServer, Arc<AppCore>, tempfile::TempDir) {
    let (mock, core, dir) = core_for(config);
    (mock, Arc::new(core), dir)
}

// ---------------------------------------------------------------------------
// The handshake
// ---------------------------------------------------------------------------

#[test]
fn a_connection_opens_by_saying_who_it_is_talking_to() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            let hello = client.hello().await;
            assert_eq!(hello.protocol, PROTOCOL_VERSION);
            assert_eq!(
                hello.app_version, APP_VERSION,
                "the app version is the serving process's, not this crate's"
            );
        });
    });
}

#[test]
fn nothing_is_answered_before_the_handshake() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            let refusal = client
                .refused(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert_eq!(refusal, ProtocolError::HandshakeRequired);
            // …and the connection is still good for the handshake it was owed.
            client.hello().await;
            let listing: SpacesListResult = client
                .ok(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert!(listing.spaces.is_empty());
        });
    });
}

#[test]
fn a_frame_from_another_protocol_names_both_versions() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            let line = format!(
                r#"{{"v":{},"id":4,"verb":"hello","params":{{}}}}"#,
                PROTOCOL_VERSION + 1
            );
            client.send_raw(format!("{line}\n").as_bytes()).await;
            let frame = client.expect_frame().await;
            assert_eq!(frame.id, 4, "a version refusal still answers the request");
            match frame.body {
                ResponseBody::Err { error } => match error.to_remote() {
                    RemoteError::Protocol(ProtocolError::UnsupportedProtocol {
                        supported,
                        requested,
                    }) => {
                        assert_eq!(supported, PROTOCOL_VERSION);
                        assert_eq!(requested, PROTOCOL_VERSION + 1);
                    }
                    other => panic!("unexpected: {other:?}"),
                },
                other => panic!("unexpected: {other:?}"),
            }
            // The connection lives: a caller that can drop to our version may.
            client.hello().await;
        });
    });
}

// ---------------------------------------------------------------------------
// Everything a caller can say wrong
// ---------------------------------------------------------------------------

#[test]
fn an_unknown_verb_costs_one_request_and_not_the_connection() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .send_raw(br#"{"v":1,"id":8,"verb":"db.query","params":{"sql":"select 1"}}"#)
                .await;
            client.send_raw(b"\n").await;
            let frame = client.expect_frame().await;
            assert_eq!(frame.id, 8);
            match frame.body {
                ResponseBody::Err { error } => match error.to_remote() {
                    RemoteError::Protocol(ProtocolError::UnknownVerb { verb }) => {
                        assert_eq!(verb, "db.query", "the refusal names what was asked for");
                    }
                    other => panic!("unexpected: {other:?}"),
                },
                other => panic!("unexpected: {other:?}"),
            }
            let listing: SpacesListResult = client
                .ok(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert!(listing.spaces.is_empty());
        });
    });
}

#[test]
fn parameters_of_the_wrong_shape_cost_one_request() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .send_raw(
                    b"{\"v\":1,\"id\":9,\"verb\":\"chat.stream\",\"params\":{\"prompt\":[]}}\n",
                )
                .await;
            let frame = client.expect_frame().await;
            assert_eq!(frame.id, 9);
            match frame.body {
                ResponseBody::Err { error } => match error.to_remote() {
                    RemoteError::Protocol(ProtocolError::BadParams { verb, .. }) => {
                        assert_eq!(verb, "chat.stream");
                    }
                    other => panic!("unexpected: {other:?}"),
                },
                other => panic!("unexpected: {other:?}"),
            }
            client.hello().await;
        });
    });
}

#[test]
fn a_line_that_is_not_a_frame_is_refused_with_no_request_to_answer() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            for junk in [&b"hello?\n"[..], &b"[1,2,3]\n"[..], &b"{}\n"[..]] {
                client.send_raw(junk).await;
                let frame = client.expect_frame().await;
                assert_eq!(
                    frame.id, NO_REQUEST,
                    "there was no readable id to answer on"
                );
                match frame.body {
                    ResponseBody::Err { error } => assert!(
                        matches!(
                            error.to_remote(),
                            RemoteError::Protocol(ProtocolError::MalformedFrame { .. })
                        ),
                        "unexpected: {error:?}"
                    ),
                    other => panic!("unexpected: {other:?}"),
                }
            }
            // Three bad lines later, the connection still works.
            client.hello().await;
        });
    });
}

#[test]
fn a_line_past_the_ceiling_says_why_and_then_the_connection_ends() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            // No newline, ever. Written and read at the same time: the server
            // stops reading the moment it gives up, so a test that wrote it
            // all first would simply hang.
            let flood = vec![b'x'; MAX_FRAME_BYTES + 4096];
            let writer = &mut client.writer;
            let reader = &mut client.reader;
            let (_, frame) = tokio::join!(
                async {
                    let _ = writer.write_all(&flood).await;
                },
                read_frame(reader)
            );
            let frame = frame.expect("the refusal is sent before the door closes");
            assert_eq!(frame.id, NO_REQUEST);
            match frame.body {
                ResponseBody::Err { error } => match error.to_remote() {
                    RemoteError::Protocol(ProtocolError::FrameTooLarge { limit }) => {
                        assert_eq!(limit, MAX_FRAME_BYTES);
                    }
                    other => panic!("unexpected: {other:?}"),
                },
                other => panic!("unexpected: {other:?}"),
            }
            assert!(
                client.recv().await.is_none(),
                "the reader lost its place in the stream, so the connection ends"
            );
        });
    });
}

// ---------------------------------------------------------------------------
// The verbs
// ---------------------------------------------------------------------------

#[test]
fn the_read_verbs_answer_from_the_open_profile() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        let space = core
            .runtime()
            .block_on(core.create_space(Some("Field notes".into())))
            .expect("create a space");

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;

            let listing: SpacesListResult = client
                .ok(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            let ids: Vec<&str> = listing.spaces.iter().map(|s| s.id.as_str()).collect();
            assert_eq!(ids, vec![space.id.as_str()]);
            assert_eq!(listing.spaces[0].title.as_deref(), Some("Field notes"));

            let backends: eidola_app_core::ipc::BackendListResult =
                client.ok(&Call::BackendList).await;
            assert!(
                backends.backends.iter().any(|b| b.id == "eidola"),
                "the seeded registry came back through the socket"
            );

            let wallet: WalletCredentialsResult = client.ok(&Call::WalletCredentials).await;
            assert!(
                wallet.credentials.is_empty(),
                "a fresh profile has spent nothing"
            );
        });
    });
}

#[test]
fn a_verb_whose_operation_fails_comes_back_typed() {
    run(|| {
        // No account configured, so the account read has nothing to read with.
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            match client.call(&Call::AccountShow).await {
                (_, Outcome::Err(RemoteError::App(e))) => {
                    assert!(
                        !e.to_string().is_empty(),
                        "a typed failure still renders a sentence"
                    );
                }
                other => panic!("expected a typed app failure, got {other:?}"),
            }
            // The failed verb took nothing else with it.
            let listing: SpacesListResult = client
                .ok(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert!(listing.spaces.is_empty());
        });
    });
}

#[test]
fn a_turn_streams_its_chunks_and_then_its_result() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let result = core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .call(&Call::ChatStream {
                    prompt: "stream me".into(),
                    model: Some(MODEL.into()),
                    space_id: None,
                })
                .await
        });

        let (chunks, outcome) = result;
        let mut content = String::new();
        let mut reasoning = String::new();
        for chunk in &chunks {
            let event: eidola_app_core::ChatStreamEvent =
                serde_json::from_value(chunk.clone()).expect("a chunk is a stream event");
            match event {
                eidola_app_core::ChatStreamEvent::ContentDelta(t) => content.push_str(&t),
                eidola_app_core::ChatStreamEvent::ReasoningDelta(t) => reasoning.push_str(&t),
            }
        }
        assert_eq!(content, "Hello from the stream.");
        assert_eq!(reasoning, "thinking…");

        let data = match outcome {
            Outcome::End(data) => data,
            Outcome::Err(e) => panic!("the turn failed: {e}"),
        };
        let result: eidola_app_core::ChatResult =
            serde_json::from_value(data).expect("the end frame is a chat result");
        assert_eq!(result.content, content, "the end restates the whole answer");
        assert_eq!(result.model, MODEL);
        assert!(
            !result.space_id.is_empty(),
            "the turn created a space to live in"
        );
        assert!(result.response_action_id.is_some());
    });
}

#[test]
fn a_turn_that_names_no_model_uses_the_profiles_default() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let expected = core
            .runtime()
            .block_on(core.default_model())
            .expect("a default model");

        let outcome = core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .call(&Call::ChatStream {
                    prompt: "no model named".into(),
                    model: None,
                    space_id: None,
                })
                .await
                .1
        });

        let data = match outcome {
            Outcome::End(data) => data,
            Outcome::Err(e) => panic!("the turn failed: {e}"),
        };
        let result: eidola_app_core::ChatResult = serde_json::from_value(data).expect("a result");
        assert_eq!(
            result.model, expected,
            "resolving the default needs the database, so the server does it"
        );
    });
}

#[test]
fn a_slow_turn_does_not_hold_the_connection_shut() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            chat_delay_ms: 400,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            let turn = client
                .send(&Call::ChatStream {
                    prompt: "take your time".into(),
                    model: Some(MODEL.into()),
                    space_id: None,
                })
                .await;
            let listing = client
                .send(&Call::SpacesList {
                    include_archived: false,
                })
                .await;

            // The read verb answers while the turn is still upstream — which
            // is what the id in every frame is for.
            loop {
                let frame = client.expect_frame().await;
                match (frame.id, &frame.body) {
                    (id, ResponseBody::End { .. }) if id == listing => break,
                    (id, ResponseBody::End { .. }) if id == turn => {
                        panic!("the turn finished first: the connection served one at a time");
                    }
                    (id, ResponseBody::Err { error }) => {
                        panic!("unexpected refusal on {id}: {error:?}");
                    }
                    _ => {}
                }
            }
        });
    });
}

#[test]
fn the_reserved_request_id_is_refused_before_anything_is_dispatched() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            // Otherwise perfectly valid — the id is the whole problem. Accepting
            // it would put a legitimate answer and an uncorrelated refusal on
            // the same id, and nothing downstream could tell them apart.
            client
                .send_raw(br#"{"v":1,"id":0,"verb":"hello","params":{}}"#)
                .await;
            client.send_raw(b"\n").await;
            let frame = client.expect_frame().await;
            assert_eq!(frame.id, NO_REQUEST);
            match frame.body {
                ResponseBody::Err { error } => match error.to_remote() {
                    RemoteError::Protocol(ProtocolError::ReservedRequestId { reserved }) => {
                        assert_eq!(reserved, NO_REQUEST);
                    }
                    other => panic!("unexpected: {other:?}"),
                },
                other => panic!("unexpected: {other:?}"),
            }
            // Refused before dispatch, so it is not a handshake either.
            let refusal = client
                .refused(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert_eq!(
                refusal,
                ProtocolError::HandshakeRequired,
                "the reserved id was never allowed to greet"
            );
            client.hello().await;
        });
    });
}

#[test]
fn a_turn_already_upstream_outlives_the_caller_that_asked_for_it() {
    run(|| {
        // The tokens are spent the moment the request reaches the upstream, so
        // a turn already under way finishes and persists whatever happens to
        // the connection: the caller loses the delivery, never the work it paid
        // for. Aborting it part-way would be cancellation landing inside a
        // spend, which is what the atomicity rules refuse — and it would bill
        // for an answer nobody ever gets.
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            // Long enough that the caller can leave while the request is
            // demonstrably upstream and not yet answered.
            chat_delay_ms: 1_500,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .send(&Call::ChatStream {
                    prompt: "asked and abandoned".into(),
                    model: Some(MODEL.into()),
                    space_id: None,
                })
                .await;
            // Wait for the request to actually be upstream — that is the moment
            // the turn stops being free to discard. Dropping earlier would race
            // the dispatch and prove nothing.
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the turn never reached the upstream");
            drop(client);
        });

        // The turn lands anyway: a space with the answer in it.
        let mut persisted = None;
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(50));
            let spaces = core
                .runtime()
                .block_on(core.list_spaces(false))
                .expect("list spaces");
            if let Some(space) = spaces.first() {
                let messages = core
                    .runtime()
                    .block_on(core.get_space_messages(space.id.clone()))
                    .expect("messages");
                if messages.iter().any(|m| m.role == "assistant") {
                    persisted = Some(messages);
                    break;
                }
            }
        }
        let messages = persisted.expect("the abandoned turn still persisted its answer");
        assert!(
            messages
                .iter()
                .any(|m| m.role == "assistant" && m.content.contains("Hello from the stream.")),
            "the answer the caller paid for is in the space: {messages:?}"
        );
    });
}

#[test]
fn an_id_already_in_flight_is_refused_and_freed_again_when_it_ends() {
    run(|| {
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            chat_delay_ms: 1_500,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .send_with_id(
                    5,
                    &Call::ChatStream {
                        prompt: "the slow one".into(),
                        model: Some(MODEL.into()),
                        space_id: None,
                    },
                )
                .await;
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the turn never started");

            // Same id, while the first is demonstrably still running.
            client
                .send_with_id(
                    5,
                    &Call::SpacesList {
                        include_archived: false,
                    },
                )
                .await;

            // The refusal answers on NO_REQUEST, deliberately: id 5 still
            // belongs to the turn, and a refusal wearing it would be the second
            // terminal frame the rule exists to prevent.
            let mut terminals_for_five = 0;
            let mut refused = false;
            loop {
                let frame = client.expect_frame().await;
                match (frame.id, &frame.body) {
                    (NO_REQUEST, ResponseBody::Err { error }) => match error.to_remote() {
                        RemoteError::Protocol(ProtocolError::DuplicateRequestId { duplicate }) => {
                            assert_eq!(duplicate, 5, "the refusal names the id that was reused");
                            refused = true;
                        }
                        other => panic!("unexpected: {other:?}"),
                    },
                    (5, ResponseBody::End { .. }) => {
                        terminals_for_five += 1;
                        break;
                    }
                    (5, ResponseBody::Err { error }) => panic!("the turn failed: {error:?}"),
                    _ => {}
                }
            }
            assert!(refused, "the duplicate was served instead of refused");
            assert_eq!(
                terminals_for_five, 1,
                "exactly one terminal frame answers an id"
            );

            // Ended, so the id is ordinary again.
            client
                .send_with_id(
                    5,
                    &Call::SpacesList {
                        include_archived: false,
                    },
                )
                .await;
            let frame = client.expect_frame().await;
            assert_eq!(frame.id, 5);
            assert!(
                matches!(frame.body, ResponseBody::End { .. }),
                "reuse after a terminal frame is allowed"
            );
        });
    });
}

#[test]
fn a_caller_that_stops_reading_stops_being_served() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            // A small pipe stands in for a socket buffer that fills.
            let mut client = Client::connect_sized(&core, 4096);
            client.hello().await;

            // …and from here the caller never reads another byte, while
            // pipelining perfectly valid requests. With an unbounded outbox the
            // app would take every one of them and hold every answer; the queue
            // is bounded so that a request's permit is still held while its
            // answer waits, and the pressure reaches the read loop.
            let flood: Vec<u8> = (2..800u64)
                .flat_map(|id| {
                    encode_line(&Request::new(
                        id,
                        &Call::SpacesList {
                            include_archived: false,
                        },
                    ))
                })
                .collect();
            let wrote = tokio::time::timeout(
                std::time::Duration::from_secs(3),
                client.writer.write_all(&flood),
            )
            .await;
            assert!(
                wrote.is_err(),
                "the app kept taking work from a caller that had stopped reading"
            );
        });
    });
}

#[test]
fn a_pipelined_hello_is_answered_before_anything_sent_behind_it() {
    run(|| {
        // A caller may write its whole opening in one go, and one of the frames
        // behind `hello` can be a turn — billed work that must not go upstream
        // before the caller has been told what it is talking to.
        //
        // Dispatched, `hello` was only *started* before the next line was read,
        // and the flag rose there: everything behind it ran against a handshake
        // whose answer did not exist yet. The single ordered writer does not
        // cover it, because ordering is decided where a frame is *queued* and
        // nothing sequenced a spawned task against the read loop — which is why
        // the frame chosen here is one the loop answers itself, without ever
        // yielding, and so reliably beat the handshake it followed.
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            chat_delay_ms: 1_500,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            let mut opening = encode_line(&Request::new(1, &Call::Hello));
            // A version this build does not speak: refused by the read loop
            // itself, in the iteration straight after `hello`'s.
            opening.extend(encode_line(&serde_json::json!({
                "v": PROTOCOL_VERSION + 1,
                "id": 2u64,
                "verb": "spaces.list",
                "params": {},
            })));
            // …and a turn, which is what makes the ordering cost money.
            opening.extend(encode_line(&Request::new(
                3,
                &Call::ChatStream {
                    prompt: "pipelined behind the handshake".into(),
                    model: Some(MODEL.into()),
                    space_id: None,
                },
            )));
            client.send_raw(&opening).await;

            let first = client.expect_frame().await;
            assert_eq!(first.id, 1, "the handshake is answered first");
            let hello: HelloResult =
                serde_json::from_value(end_of(&first)).expect("a hello result");
            assert_eq!(hello.protocol, PROTOCOL_VERSION);

            // The refusal the read loop wrote for itself comes after it, not
            // before — an answer to a frame sent behind a handshake cannot
            // reach the caller ahead of the handshake's own.
            let second = client.expect_frame().await;
            assert_eq!(second.id, 2, "the refusal follows the handshake");
            assert!(matches!(second.body, ResponseBody::Err { .. }));

            // And the turn, which the mock holds open, was started only after
            // all of that.
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the pipelined turn never ran at all");
        });
    });
}

#[test]
fn a_reused_id_never_overlaps_the_exchange_it_reuses() {
    run(|| {
        // The id claim is released once the terminal frame is *queued*, not
        // once the writer has put it on the wire — so a pipelined reuse can be
        // accepted while the first answer is still in the outbox.
        //
        // That is not an overlap the caller can see, and this pins why: the
        // outbox is one queue with one writer, the claim is held across the
        // send, and the reuse is not even dispatched until that send returned.
        // So the wire is strictly sequential, and the property to hold is that
        // every frame of the second exchange follows the first's terminal.
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;

            // Both under one id, written together so the second is already
            // buffered when the first releases its claim. Either outcome is
            // legal — the reuse is refused while the claim is live, and served
            // once it is not — and the property under test is what they share.
            let ask = Call::SpacesList {
                include_archived: false,
            };
            let mut pipelined = encode_line(&Request::new(9, &ask));
            pipelined.extend(encode_line(&Request::new(9, &ask)));
            client.send_raw(&pipelined).await;

            let mut answered = 0;
            let mut refused = 0;
            while answered + refused < 2 {
                let frame = client.expect_frame().await;
                match (frame.id, frame.body) {
                    // A refusal for the reuse deliberately wears no id — it is
                    // the rule holding, not a second answer on a live one.
                    (NO_REQUEST, ResponseBody::Err { error }) => {
                        assert!(
                            matches!(
                                error.to_remote(),
                                RemoteError::Protocol(ProtocolError::DuplicateRequestId {
                                    duplicate: 9
                                })
                            ),
                            "unexpected: {error:?}"
                        );
                        refused += 1;
                    }
                    (9, ResponseBody::End { .. }) => answered += 1,
                    // The claim exists so that nothing else can wear this id
                    // while an exchange on it is open: no second terminal, and
                    // no half of one interleaved with another's.
                    (id, body) => panic!("a frame nobody may write: id {id}, {body:?}"),
                }
            }
            assert_eq!(
                answered + refused,
                2,
                "each ask was answered exactly once, on its own id or on none"
            );

            // Reuse after the terminal frame is ordinary, which is the other
            // half of holding the claim only until the answer is queued.
            let reused: SpacesListResult = client.ok(&ask).await;
            let _ = reused;
            client.hello().await;
        });
    });
}

#[test]
fn cancelling_the_connection_ends_its_writer_and_its_forwarder_too() {
    run(|| {
        // Whoever owns the connection cancels it by dropping this future — the
        // socket's owner does exactly that on a full shutdown. **A dropped
        // `JoinHandle` detaches its task rather than ending it**, so the writer
        // and a turn's chunk forwarder each escaped the connection they belong
        // to: the socket's write half stayed open and went on delivering until
        // the turn finished, which is not what "this connection has ended"
        // means to the process that ended it.
        //
        // The observable is prompt end-of-stream: the write half is released
        // with the connection rather than at the turn's own pace.
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            // Far longer than the settle below, so "the writer went with the
            // connection" and "the turn happened to finish" cannot be confused.
            chat_delay_ms: 3_000,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let (client_writes, server_reads) = tokio::io::duplex(PIPE_BUFFER);
            let (server_writes, client_reads) = tokio::io::duplex(PIPE_BUFFER);
            let serving = tokio::spawn(serve_connection(
                Arc::clone(&core),
                APP_VERSION.to_string(),
                server_reads,
                server_writes,
            ));
            let mut writer = client_writes;
            let mut reader = BufReader::new(client_reads);

            writer
                .write_all(&encode_line(&Request::new(1, &Call::Hello)))
                .await
                .expect("write");
            read_frame(&mut reader).await.expect("the handshake");

            writer
                .write_all(&encode_line(&Request::new(
                    2,
                    &Call::ChatStream {
                        prompt: "running when the connection ends".into(),
                        model: Some(MODEL.into()),
                        space_id: None,
                    },
                )))
                .await
                .expect("write");
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the turn never started");

            // The socket's owner ending this connection.
            serving.abort();

            let ended = tokio::time::timeout(
                std::time::Duration::from_millis(500),
                client_frame(&mut reader),
            )
            .await;
            assert!(
                matches!(ended, Ok(None)),
                "the write half outlived the connection that was ended: {ended:?}"
            );
        });
    });
}

#[test]
fn a_caller_that_half_closes_still_gets_the_answers_it_asked_for() {
    run(|| {
        // `shutdown(SHUT_WR)` and then read to completion is the polite client:
        // it has said everything it means to say and is waiting on the answers
        // it already asked for, with its read half wide open.
        //
        // A clean end of the *request* stream was read as the end of the
        // conversation and aborted every in-flight task, so a turn already
        // upstream could never enqueue its terminal frame — the caller waited
        // out a `chat.stream` that finished and persisted, and got nothing.
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            // Long enough that the request is demonstrably still running when
            // the write half closes.
            chat_delay_ms: 800,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let (client_writes, server_reads) = tokio::io::duplex(PIPE_BUFFER);
            let (server_writes, client_reads) = tokio::io::duplex(PIPE_BUFFER);
            tokio::spawn(serve_connection(
                Arc::clone(&core),
                APP_VERSION.to_string(),
                server_reads,
                server_writes,
            ));
            let mut writer = client_writes;
            let mut reader = BufReader::new(client_reads);

            writer
                .write_all(&encode_line(&Request::new(1, &Call::Hello)))
                .await
                .expect("write");
            read_frame(&mut reader).await.expect("the handshake");

            writer
                .write_all(&encode_line(&Request::new(
                    2,
                    &Call::ChatStream {
                        prompt: "asked before saying goodbye".into(),
                        model: Some(MODEL.into()),
                        space_id: None,
                    },
                )))
                .await
                .expect("write");
            // Demonstrably running before the write half goes.
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the turn never started");

            // The half-close: nothing more will be sent, and the read half
            // stays open for exactly the answer above.
            drop(writer);

            let mut chunks = 0;
            let end = loop {
                let frame = tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    client_frame(&mut reader),
                )
                .await
                .expect("the answer arrived rather than the connection being torn down")
                .expect("a frame, not the end of the stream");
                assert_eq!(frame.id, 2);
                match frame.body {
                    ResponseBody::Chunk { .. } => chunks += 1,
                    other => break other,
                }
            };
            assert!(chunks > 0, "the turn's chunks were delivered");
            assert!(
                matches!(end, ResponseBody::End { .. }),
                "the terminal frame is what a half-closed caller is waiting for: {end:?}"
            );

            // …and then the connection ends on its own, the answers being done.
            assert!(
                client_frame(&mut reader).await.is_none(),
                "nothing follows the last answer"
            );
        });
    });
}

#[test]
fn a_caller_that_will_never_read_again_is_asked_for_nothing_more() {
    run(|| {
        // The half-close, which the previous test cannot reach: a peer that
        // shuts its *read* half and goes on writing looks, from the reader's
        // side, exactly like a well-behaved caller. Nothing in the frames says
        // the answers have nowhere to go — so without the writer's death
        // reaching the read loop, every one of those requests is dispatched,
        // and `chat.stream` is a turn that goes upstream and is paid for.
        //
        // The first turn here is legitimate: it was asked for while the answer
        // could still be delivered, and it is what makes the writer try to
        // write and fail. What must not happen is a second one.
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let (client_writes, server_reads) = tokio::io::duplex(PIPE_BUFFER);
            let (server_writes, client_reads) = tokio::io::duplex(PIPE_BUFFER);
            tokio::spawn(serve_connection(
                Arc::clone(&core),
                APP_VERSION.to_string(),
                server_reads,
                server_writes,
            ));
            let mut writer = client_writes;
            let mut reader = BufReader::new(client_reads);

            writer
                .write_all(&encode_line(&Request::new(1, &Call::Hello)))
                .await
                .expect("write");
            read_frame(&mut reader)
                .await
                .expect("the connection is answering");

            // From here the app can be told things and can answer none of them.
            drop(reader);

            let turn = Call::ChatStream {
                prompt: "the one that was asked in good faith".into(),
                model: Some(MODEL.into()),
                space_id: None,
            };
            writer
                .write_all(&encode_line(&Request::new(2, &turn)))
                .await
                .expect("write");
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the first turn never started");
            // Long enough for that turn's frames to be attempted and refused,
            // which is the moment the app learns it cannot answer anybody.
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Now ask again, and keep asking. Each of these is a billed turn if
            // it is dispatched.
            let mut refused_the_write = false;
            for id in 3..20u64 {
                if writer
                    .write_all(&encode_line(&Request::new(id, &turn)))
                    .await
                    .is_err()
                {
                    refused_the_write = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
            }
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;

            assert_eq!(
                mock.chat_hits(),
                1,
                "a peer the app cannot answer went on starting paid turns"
            );
            assert!(
                refused_the_write,
                "the connection went on reading after its answers had nowhere to go"
            );
        });
    });
}

#[test]
fn no_refusal_ever_wears_an_id_a_live_request_holds() {
    run(|| {
        // The one-terminal-frame-per-id rule has to survive frames that are
        // *also* wrong in some other way. A bad version, bad parameters and an
        // unknown verb are each refused on the caller's own id — correlating a
        // refusal is the protocol's norm — but not when that id belongs to a
        // request still going to answer on it, or the caller gets two terminal
        // frames wearing one id and can correlate neither.
        let (mock, core, _dir) = served(MockConfig {
            chat: ChatBehavior::OkStreaming,
            chat_delay_ms: 1_500,
            ..MockConfig::default()
        });
        with_account(&core);

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;
            client
                .send_with_id(
                    5,
                    &Call::ChatStream {
                        prompt: "the slow one".into(),
                        model: Some(MODEL.into()),
                        space_id: None,
                    },
                )
                .await;
            for _ in 0..600 {
                if mock.chat_hits() > 0 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
            assert!(mock.chat_hits() > 0, "the turn never started");

            // Every gate that would otherwise echo the id, while 5 is live.
            let offending = [
                format!(
                    r#"{{"v":{},"id":5,"verb":"hello","params":{{}}}}"#,
                    PROTOCOL_VERSION + 1
                ),
                r#"{"v":1,"id":5,"verb":"chat.stream","params":{"prompt":5}}"#.to_string(),
                r#"{"v":1,"id":5,"verb":"db.query","params":{}}"#.to_string(),
            ];
            for frame in &offending {
                client.send_raw(format!("{frame}\n").as_bytes()).await;
            }

            let mut terminals_for_five = 0;
            let mut refusals = 0;
            loop {
                let frame = client.expect_frame().await;
                match (frame.id, &frame.body) {
                    (NO_REQUEST, ResponseBody::Err { error }) => match error.to_remote() {
                        RemoteError::Protocol(ProtocolError::DuplicateRequestId { duplicate }) => {
                            assert_eq!(duplicate, 5, "the refusal names the id it could not use");
                            refusals += 1;
                        }
                        other => panic!("unexpected uncorrelated refusal: {other:?}"),
                    },
                    (5, ResponseBody::End { .. }) => {
                        terminals_for_five += 1;
                        break;
                    }
                    (5, ResponseBody::Err { error }) => {
                        // The exact defect: a gate refusal answering on an id
                        // whose request is still running.
                        panic!("a refusal wore the live id 5: {error:?}");
                    }
                    _ => {}
                }
            }
            assert_eq!(
                refusals,
                offending.len(),
                "each offending frame is refused, uncorrelated, exactly once"
            );
            assert_eq!(
                terminals_for_five, 1,
                "exactly one terminal frame ever wears id 5"
            );

            // The judgements themselves are unchanged where the id is free: a
            // bad version on an id nobody holds is still correlated to it.
            client
                .send_raw(
                    format!(
                        "{{\"v\":{},\"id\":9,\"verb\":\"hello\",\"params\":{{}}}}\n",
                        PROTOCOL_VERSION + 1
                    )
                    .as_bytes(),
                )
                .await;
            let frame = client.expect_frame().await;
            assert_eq!(frame.id, 9, "an unambiguous refusal stays correlated");
            match &frame.body {
                ResponseBody::Err { error } => assert!(
                    matches!(
                        error.to_remote(),
                        RemoteError::Protocol(ProtocolError::UnsupportedProtocol { .. })
                    ),
                    "unexpected: {error:?}"
                ),
                other => panic!("unexpected: {other:?}"),
            }
        });
    });
}
