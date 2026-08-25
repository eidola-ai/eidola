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
    Call, DefaultModelResult, Done, HelloResult, MAX_FRAME_BYTES, ModelListResult, NO_REQUEST,
    PROTOCOL_VERSION, ProtocolError, RemoteError, Request, Response, ResponseBody,
    SpacesArchiveResult, SpacesListResult, WalletCredentialsResult, decode_response, encode_line,
    serve_connection,
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
        let (client_writes, server_reads) = tokio::io::duplex(PIPE_BUFFER);
        let (server_writes, client_reads) = tokio::io::duplex(PIPE_BUFFER);
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
fn a_write_verb_changes_the_profile_it_speaks_for() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        let space = core
            .runtime()
            .block_on(core.create_space(Some("Field notes".into())))
            .expect("create a space");

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;

            let _: Done = client
                .ok(&Call::SpacesRename {
                    space_id: space.id.clone(),
                    title: "Renamed over the socket".into(),
                })
                .await;
            let listing: SpacesListResult = client
                .ok(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert_eq!(
                listing.spaces[0].title.as_deref(),
                Some("Renamed over the socket"),
                "the write landed in the profile, not in a copy of it"
            );

            let archived: SpacesArchiveResult = client
                .ok(&Call::SpacesArchive {
                    space_id: space.id.clone(),
                })
                .await;
            assert!(archived.archived);
            let listing: SpacesListResult = client
                .ok(&Call::SpacesList {
                    include_archived: false,
                })
                .await;
            assert!(listing.spaces.is_empty(), "and the listing agrees");

            // Archiving what is already archived is an answer, not a failure.
            let again: SpacesArchiveResult = client
                .ok(&Call::SpacesArchive {
                    space_id: space.id.clone(),
                })
                .await;
            assert!(!again.archived);
        });
    });
}

#[test]
fn the_backend_registry_answers_and_takes_a_choice() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;

            let _: Done = client
                .ok(&Call::BackendSetEnabled {
                    id: "eidola".into(),
                    enabled: false,
                })
                .await;
            let backends: eidola_app_core::ipc::BackendListResult =
                client.ok(&Call::BackendList).await;
            let eidola = backends
                .backends
                .iter()
                .find(|b| b.id == "eidola")
                .expect("the seeded row");
            assert!(!eidola.enabled, "the choice took, and the listing shows it");
        });
    });
}

#[test]
fn the_model_verbs_answer_for_the_process_that_owns_the_engines() {
    run(|| {
        let (_mock, core, _dir) = served(MockConfig::default());
        let expected = core
            .runtime()
            .block_on(core.default_model())
            .expect("a default model");

        core.runtime().block_on(async {
            let mut client = Client::connect(&core);
            client.hello().await;

            // Read as a frame first: `running` has no serde default, so a
            // payload that stopped carrying the registry would not decode.
            // **What this cannot reach is a non-empty registry** — putting an
            // engine in it needs a real `llama-server` and a real `.gguf`, so
            // the value's honesty rests on `running_engines` itself and on the
            // CLI's reconciliation tests, not on this one.
            let (_, outcome) = client.call(&Call::ModelList).await;
            let data = match outcome {
                Outcome::End(data) => data,
                other => panic!("expected a result, got {other:?}"),
            };
            assert!(
                data["running"].is_array(),
                "the registry travels beside the scan"
            );
            let models: ModelListResult = serde_json::from_value(data).expect("the verb's result");
            assert!(models.state.models.is_empty(), "a fresh profile has none");

            // Which model a turn that names none would use — the resolution
            // that needs the database, which is why it has a verb at all.
            let default: DefaultModelResult = client.ok(&Call::ChatDefaultModel).await;
            assert_eq!(default.model, expected);
            assert_eq!(default.model, MODEL);
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
