//! The contract client against a shell written in Rust (`fake_shell`),
//! natively: the pipe's framing, pull and backpressure, the file API under
//! app-core's storage, boot and its confirmation, the durability barrier, and
//! attestation over the pipe against an in-process shim.

mod fake_shell;

use std::time::Duration;

use base64::Engine as _;
use eidola_app_core::error::AppError;
use eidola_app_core::storage::HostStorage;
use eidola_mobile::boot::{self, BootError, BootOptions, Booted};
use eidola_mobile::client::ShellClient;
use eidola_mobile::files::ShellFiles;
use eidola_mobile::protocol::{REQUIRED_METHODS, ShellError};
use eidola_mobile::socket::ShellDialer;
use fake_shell::FakeShell;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// A runtime for the test's own futures; the core keeps its own.
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

/// A TCP echo server like `mobile/tools/echo_server.py`: echoes, and on the
/// client's FIN answers `bye <n>\n` and closes.
async fn echo_server() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = vec![0u8; 65536];
                let mut total = 0;
                loop {
                    match stream.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            total += n;
                            if stream.write_all(&buf[..n]).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                let _ = stream.write_all(format!("bye {total}\n").as_bytes()).await;
            });
        }
    });
    port
}

#[test]
fn bytes_cross_the_pipe_intact_in_both_directions() {
    block_on(async {
        let port = echo_server().await;
        let shell = FakeShell::default();
        let client = ShellClient::new(shell.clone());
        // A small receive bound, so the stream takes many receives.
        let socket = ShellDialer::new(client)
            .receive_max(10_000)
            .connect("127.0.0.1", port)
            .await
            .unwrap();
        let (mut reader, mut writer) = tokio::io::split(socket);
        let total = 3 << 20;
        let data: Vec<u8> = (0..total).map(|i| (i * 7 % 251) as u8).collect();
        let send = async {
            for chunk in data.chunks(50_000) {
                writer.write_all(chunk).await.unwrap();
            }
            writer.shutdown().await.unwrap();
        };
        let receive = async {
            let mut got = Vec::new();
            reader.read_to_end(&mut got).await.unwrap();
            got
        };
        let ((), got) = tokio::join!(send, receive);
        assert_eq!(&got[..total], &data[..]);
        assert_eq!(&got[total..], format!("bye {total}\n").as_bytes());
        let state = shell.state.lock().unwrap();
        assert_eq!(state.most_receives_in_flight, 1, "one receive at a time");
    });
}

#[test]
fn a_reader_that_stops_reading_stops_the_pipe() {
    block_on(async {
        // A peer with 4 MiB to send at once.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = stream.write_all(&vec![1u8; 4 << 20]).await;
        });
        let shell = FakeShell::default();
        let mut socket = ShellDialer::new(ShellClient::new(shell.clone()))
            .receive_max(64 << 10)
            .connect("127.0.0.1", port)
            .await
            .unwrap();
        let id = socket.id();
        let mut buf = vec![0u8; 1000];
        socket.read_exact(&mut buf).await.unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        // One read of 1000 bytes took one receive: nothing more was pulled
        // from the network while nobody read.
        let pulled = shell.state.lock().unwrap().received[&id];
        assert!(pulled <= 64 << 10, "{pulled} bytes pulled");
        let mut rest = Vec::new();
        socket.read_to_end(&mut rest).await.unwrap();
        assert_eq!(1000 + rest.len(), 4 << 20);
    });
}

#[test]
fn a_dropped_socket_is_closed_and_a_refused_connect_is_typed() {
    block_on(async {
        let port = echo_server().await;
        let shell = FakeShell::default();
        let dialer = ShellDialer::new(ShellClient::new(shell.clone()));
        let socket = dialer.connect("[127.0.0.1]", port).await.unwrap();
        let id = socket.id();
        drop(socket);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(shell.log().contains(&format!("-> tcp.close socket {id}")));

        // Nothing listens on a port just freed.
        let free = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let free_port = free.local_addr().unwrap().port();
        drop(free);
        let Err(error) = dialer.connect("127.0.0.1", free_port).await else {
            panic!("connected to nothing");
        };
        assert!(
            matches!(
                &error,
                ShellError::Refused {
                    method: "tcp.connect",
                    ..
                }
            ),
            "{error:?}"
        );
        assert_eq!(
            std::io::Error::from(error).kind(),
            std::io::ErrorKind::ConnectionRefused
        );
    });
}

#[test]
fn files_round_trip_and_large_writes_split() {
    block_on(async {
        let shell = FakeShell::default();
        let files = ShellFiles::new(ShellClient::new(shell.clone()));
        assert_eq!(files.read("a").await.unwrap(), None);
        let big: Vec<u8> = (0..9 << 20).map(|i| (i % 249) as u8).collect();
        files.write_at("a", 0, big.clone()).await.unwrap();
        assert_eq!(
            shell
                .log()
                .iter()
                .filter(|l| l.as_str() == "-> files.write a")
                .count(),
            2,
            "9 MiB is two writes of at most 8 MiB"
        );
        assert_eq!(files.read("a").await.unwrap().unwrap(), big);
        files.write_at("a", 2, b"xy".to_vec()).await.unwrap();
        files.truncate("a", 4).await.unwrap();
        assert_eq!(files.read("a").await.unwrap().unwrap(), [0, 1, b'x', b'y']);
        files.sync("a").await.unwrap();
        files.rename("a", "b").await.unwrap();
        assert_eq!(files.list().await.unwrap(), ["b"]);
        files.remove("b").await.unwrap();
        files.remove("b").await.unwrap();
        assert!(files.list().await.unwrap().is_empty());
        let missing = files.rename("a", "b").await.unwrap_err();
        assert_eq!(missing.kind(), std::io::ErrorKind::NotFound);
    });
}

#[test]
fn a_shell_answering_nonsense_fails_typed() {
    for garbage in ["", "nope", "{\"ok\":true}", "{\"ok\":false}", "[1,2]"] {
        let shell = FakeShell::default().answering_garbage(garbage);
        let result = boot_on(&shell);
        match result {
            Err(BootError::Shell(ShellError::Malformed {
                method: "shell.hello",
                ..
            })) => {}
            Err(e) => panic!("{garbage:?}: {e:?}"),
            Ok(_) => panic!("{garbage:?}: booted"),
        }
    }
}

/// The profile's lock is a real file lock natively: every shell gets a
/// directory of its own (a relaunch shares its shell's).
fn lock_dir(shell: &FakeShell) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "eidola-mobile-test-{}-{:p}",
        std::process::id(),
        std::sync::Arc::as_ptr(&shell.files_key())
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn boot_on(shell: &FakeShell) -> Result<Booted<FakeShell>, BootError> {
    boot_with(shell, BootOptions::default())
}

fn boot_with(shell: &FakeShell, mut options: BootOptions) -> Result<Booted<FakeShell>, BootError> {
    options.profile = Some(lock_dir(shell));
    // Outside any runtime: a boot that fails drops the core it opened, and
    // its runtime may not be dropped inside another's.
    park_on(boot::boot(ShellClient::new(shell.clone()), options))
}

/// Run `f` to completion on this thread, parking between polls.
fn park_on<F: std::future::Future>(f: F) -> F::Output {
    struct Unpark(std::thread::Thread);
    impl std::task::Wake for Unpark {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(Unpark(std::thread::current())));
    let mut cx = std::task::Context::from_waker(&waker);
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(output) = f.as_mut().poll(&mut cx) {
            return output;
        }
        std::thread::park();
    }
}

#[test]
fn boot_confirms_only_once_the_profile_is_open_and_durable() {
    let shell = FakeShell::default();
    let booted = boot_on(&shell).unwrap();
    let log = shell.log();
    assert_eq!(log[0], "-> shell.hello");
    assert_eq!(log.last().unwrap(), "<- bundle.confirm");
    let confirm = log.iter().position(|l| l == "-> bundle.confirm").unwrap();
    // The new database's log was written and synced before the confirm.
    let synced = log
        .iter()
        .position(|l| l == "<- files.sync eidola.db-wal")
        .expect("the log was synced");
    assert!(synced < confirm, "{log:#?}");
    assert_eq!(shell.state.lock().unwrap().confirms, 1);
    assert!(booted.bundle.confirmed);
    drop(booted);
}

#[test]
fn an_older_shell_is_refused_before_anything_else_is_asked() {
    let v1: Vec<&str> = REQUIRED_METHODS
        .iter()
        .copied()
        .filter(|m| *m != "shell.openExternal")
        .collect();
    let shell = FakeShell::default().with_contract(1, &v1);
    let Err(error) = boot_on(&shell) else {
        panic!("booted on contract 1");
    };
    assert!(
        matches!(
            error,
            BootError::Shell(ShellError::Contract { contract: 1, .. })
        ),
        "{error:?}"
    );
    assert_eq!(shell.log(), ["-> shell.hello", "<- shell.hello"]);
}

#[test]
fn a_profile_that_cannot_open_never_confirms() {
    let shell = FakeShell::default();
    shell
        .state
        .lock()
        .unwrap()
        .files
        .insert("eidola.db".into(), b"this is not a database".repeat(400));
    let Err(error) = boot_on(&shell) else {
        panic!("booted on a corrupt database");
    };
    assert!(matches!(error, BootError::Profile(_)), "{error:?}");
    assert_eq!(shell.state.lock().unwrap().confirms, 0);
}

#[test]
fn a_commit_survives_a_relaunch() {
    let shell = FakeShell::default();
    let booted = boot_on(&shell).unwrap();
    let core = &booted.core;
    let space = core
        .runtime()
        .block_on(core.create_space(Some("kept".into())))
        .unwrap();
    core.runtime().block_on(core.flush_storage()).unwrap();
    drop(booted);

    let relaunched = shell.relaunch();
    let booted = boot_on(&relaunched).unwrap();
    let core = &booted.core;
    let spaces = core.runtime().block_on(core.list_spaces(false)).unwrap();
    assert!(
        spaces
            .iter()
            .any(|s| s.id == space.id && s.title.as_deref() == Some("kept"))
    );
}

/// A peer that accepts connections and never answers: the request's
/// arrival at the shell is all this test watches.
async fn silent_peer() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((stream, _)) = listener.accept().await {
            held.push(stream);
        }
    });
    port
}

#[test]
fn a_request_leaves_only_after_the_commit_before_it_is_synced() {
    let shell = FakeShell::default();
    let booted = boot_on(&shell).unwrap();
    let core = &booted.core;
    let rt = tokio::runtime::Runtime::new().unwrap();
    let port = rt.block_on(silent_peer());
    core.runtime()
        .block_on(core.set_base_url(format!("https://127.0.0.1:{port}")))
        .unwrap();
    core.runtime().block_on(core.flush_storage()).unwrap();
    // Every file reply now takes a while, so a request that did not wait for
    // the commit's writes would reach the shell before their replies.
    shell.set_file_delay(Duration::from_millis(500));
    let mark = shell.log().len();

    core.runtime()
        .block_on(core.create_space(Some("before the request".into())))
        .unwrap();
    // The request fails (the peer never answers TLS); only its departure
    // matters.
    let _ = core.runtime().block_on(async {
        tokio::time::timeout(Duration::from_secs(3), core.current_terms()).await
    });

    let log = shell.log()[mark..].to_vec();
    let connect = log
        .iter()
        .position(|l| l.starts_with("-> tcp.connect"))
        .unwrap_or_else(|| panic!("no request left: {log:#?}"));
    let before = &log[..connect];
    let wal_synced = before.iter().any(|l| l == "<- files.sync eidola.db-wal");
    let unanswered: Vec<&String> = before
        .iter()
        .filter(|l| l.starts_with("-> files."))
        .filter(|sent| {
            let replied = sent.replacen("->", "<-", 1);
            before.iter().filter(|l| **l == replied).count()
                < before.iter().filter(|l| l == sent).count()
        })
        .collect();
    assert!(wal_synced && unanswered.is_empty(), "{log:#?}");
}

const MEASUREMENT: [u8; 48] = [0x6b; 48];

fn pem(der: &[u8]) -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(der);
    let lines: Vec<&str> = b64
        .as_bytes()
        .chunks(64)
        .map(|c| std::str::from_utf8(c).unwrap())
        .collect();
    format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        lines.join("\n")
    )
}

/// An upstream that answers every request with an empty terms list.
async fn terms_upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let _ = stream.read(&mut buf).await;
                let body = r#"{"documents":[]}"#;
                let _ = stream
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await;
            });
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[test]
fn attestation_runs_over_the_shells_pipe_and_refuses_another_measurement() {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let cert_dir = dir.path().join("certs");
    tinfoil_shim_mock::ensure_persistent_material(&cert_dir).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let shim = |measurement: [u8; 48]| {
        let cert_dir = cert_dir.clone();
        rt.block_on(async move {
            tinfoil_shim_mock::start(tinfoil_shim_mock::ShimConfig {
                upstream_url: terms_upstream().await,
                listen_addr: "127.0.0.1:0".parse().unwrap(),
                measurement: measurement.to_vec(),
                host_data: [0; 32],
                cert_dir,
                gpus: 0,
                extra_tls_names: Vec::new(),
                misbind: None,
            })
            .await
            .unwrap()
        })
    };
    let trusted = shim(MEASUREMENT);
    let untrusted = shim([0x11; 48]);

    let tls_ca = std::fs::read_to_string(cert_dir.join("tls-ca.pem")).unwrap();
    let tls_ca_der = base64::engine::general_purpose::STANDARD
        .decode(
            tls_ca
                .lines()
                .filter(|l| !l.starts_with("-----"))
                .collect::<String>(),
        )
        .unwrap();
    let shell = FakeShell::default();
    let booted = boot_with(
        &shell,
        BootOptions {
            extra_tls_roots: vec![tls_ca_der.into()],
            ..BootOptions::default()
        },
    )
    .unwrap();
    let core = &booted.core;
    let url = |port: u16| format!("https://127.0.0.1:{port}");
    core.runtime().block_on(async {
        core.set_base_url(url(trusted.local_addr.port()))
            .await
            .unwrap();
        core.set_hardware_root_ca(pem(&trusted.ark_der))
            .await
            .unwrap();
        core.set_hardware_intermediate_ca(pem(&trusted.ask_der))
            .await
            .unwrap();
        core.trust_measurement(
            hex::encode(MEASUREMENT),
            hex::encode([0u8; 32]),
            format!("{:#x}", tinfoil_shim_mock::MOCK_GUEST_POLICY),
        )
        .await
        .unwrap();
    });

    let terms = core.runtime().block_on(core.current_terms()).unwrap();
    assert!(terms.is_empty());
    assert_eq!(trusted.attestations(), 1);
    assert!(shell.log().iter().any(|l| l.starts_with("-> tcp.send")));

    core.runtime()
        .block_on(core.set_base_url(url(untrusted.local_addr.port())))
        .unwrap();
    let refused = core.runtime().block_on(core.current_terms()).unwrap_err();
    assert!(
        matches!(&refused, AppError::Attestation { message } if message.contains("trusted_measurements")),
        "{refused:?}"
    );
    drop(booted);
}
