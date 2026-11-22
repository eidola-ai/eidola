//! The headless self-test: the bundle `mobile/build-bundle.sh --self-test`
//! builds (with this feature; a release bundle has none of it), run by
//! `mobile/self-test.sh` in the iOS Simulator and the Android emulator against
//! host-run attested shims.
//!
//! It runs over two launches, telling them apart by a state file it keeps in
//! the shell's file directory:
//!
//! 1. **First launch.** Boot and confirm; point the profile at the shim with
//!    the trusted measurement (base URL, ARK, ASK, measurement — the
//!    profile's own override surface); then, under a trace of every shell
//!    call, commit (a new space) and at once send a request: **the barrier**
//!    holds if every file call posted before the request's `tcp.connect`
//!    had its reply first, and the commit's write-ahead log was synced among
//!    them (the debug shell's `debug.syncLog` names the flushes it made).
//!    That request is **attested** over the shell's pipe; the same request
//!    to a shim attesting another measurement is refused. Then the bundle
//!    store (stage, seal, compare the shell's digest, remove) and the
//!    throughput of the pipe and the file API.
//! 2. **After a cold relaunch.** Boot again: the space committed in the
//!    first launch is there, and the trust settings with it (the request is
//!    attested again without configuring anything).
//!
//! Each launch writes its report (`selftest-<n>.json`, synced) to the file
//! directory, where the harness reads it from outside the app, and logs it
//! to the console.

use std::fmt::Write as _;

use base64::Engine as _;
use eidola_app_core::AppCore;
use eidola_app_core::error::AppError;
use rustls::pki_types::CertificateDer;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::boot::{self, BootOptions};
use crate::client::{ShellClient, TraceEntry, TraceEvent, Transport};
use crate::socket::ShellDialer;

const STATE: &str = "selftest-state.json";

/// `selftest.json`, written into the bundle by the harness.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// The shim attesting `measurement`.
    pub good_url: String,
    /// A shim attesting another measurement.
    pub bad_url: String,
    pub measurement: String,
    pub host_data: String,
    pub guest_policy: String,
    pub ark_pem: String,
    pub ask_pem: String,
    /// The shims' TLS root.
    pub tls_ca_pem: String,
    /// `mobile/tools/echo_server.py`, for the pipe's throughput.
    pub echo_host: String,
    pub echo_port: u16,
    #[serde(default = "default_mib")]
    pub throughput_mib: usize,
}

fn default_mib() -> usize {
    16
}

/// What the page provides.
pub struct Env {
    /// A millisecond clock.
    pub now_ms: fn() -> f64,
    pub log: fn(&str),
    /// A timer, in milliseconds.
    pub sleep_ms: fn(u32) -> crate::client::BoxFuture<'static, ()>,
    /// What the page found in the webview's own storage.
    pub webview: String,
}

#[derive(Serialize, Deserialize)]
struct State {
    title: String,
}

#[derive(Serialize)]
struct Check {
    name: &'static str,
    pass: bool,
    detail: Value,
}

#[derive(Serialize, Default)]
struct Report {
    phase: u32,
    pass: bool,
    checks: Vec<Check>,
    measurements: Vec<Value>,
}

impl Report {
    fn check(&mut self, env: &Env, name: &'static str, pass: bool, detail: Value) {
        (env.log)(&format!(
            "self-test: {name}: {} {detail}",
            if pass { "PASS" } else { "FAIL" }
        ));
        self.checks.push(Check { name, pass, detail });
    }
}

fn pem_der(pem: &str) -> Result<Vec<u8>, String> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .map_err(|e| format!("not a PEM certificate: {e}"))
}

/// Run this launch's phase; resolves to its report's JSON text.
pub async fn run<T: Transport>(
    client: ShellClient<T>,
    config: &str,
    env: Env,
) -> Result<String, String> {
    let config: Config = serde_json::from_str(config).map_err(|e| format!("selftest.json: {e}"))?;
    let state = client
        .files_read(STATE)
        .await
        .map_err(|e| e.to_string())?
        .and_then(|bytes| serde_json::from_slice::<State>(&bytes).ok());
    let mut report = Report::default();
    match state {
        None => first_launch(&client, &config, &env, &mut report).await,
        Some(state) => relaunch(&client, &config, &env, &mut report, state).await,
    }
    report.pass = !report.checks.is_empty() && report.checks.iter().all(|c| c.pass);
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    let name = format!("selftest-{}.json", report.phase);
    client
        .files_write(&name, 0, text.as_bytes(), true)
        .await
        .map_err(|e| e.to_string())?;
    (env.log)(&format!(
        "self-test: phase {} {}",
        report.phase,
        if report.pass { "PASS" } else { "FAIL" }
    ));
    Ok(text)
}

async fn boot_core<T: Transport>(
    client: &ShellClient<T>,
    config: &Config,
    env: &Env,
    report: &mut Report,
) -> Option<AppCore> {
    let roots = match pem_der(&config.tls_ca_pem) {
        Ok(der) => vec![CertificateDer::from(der)],
        Err(e) => {
            report.check(env, "boot", false, json!(e));
            return None;
        }
    };
    let started = (env.now_ms)();
    match boot::boot(
        client.clone(),
        BootOptions {
            extra_tls_roots: roots,
            ..BootOptions::default()
        },
    )
    .await
    {
        Ok(booted) => {
            let ms = (env.now_ms)() - started;
            report.check(
                env,
                "boot",
                booted.bundle.serving.confirmed,
                json!({
                    "contract": booted.hello.contract,
                    "platform": booted.hello.platform,
                    "shell": booted.hello.shell,
                    "serving": booted.bundle.serving.version,
                    "digest": booted.bundle.serving.digest,
                    "confirmed": booted.bundle.serving.confirmed,
                    "ms": ms,
                }),
            );
            Some(booted.core)
        }
        Err(e) => {
            report.check(env, "boot", false, json!(e.to_string()));
            None
        }
    }
}

async fn first_launch<T: Transport>(
    client: &ShellClient<T>,
    config: &Config,
    env: &Env,
    report: &mut Report,
) {
    report.phase = 1;
    let Some(core) = boot_core(client, config, env, report).await else {
        return;
    };

    let trusted = async {
        core.set_base_url(config.good_url.clone()).await?;
        core.set_hardware_root_ca(config.ark_pem.clone()).await?;
        core.set_hardware_intermediate_ca(config.ask_pem.clone())
            .await?;
        core.trust_measurement(
            config.measurement.clone(),
            config.host_data.clone(),
            config.guest_policy.clone(),
        )
        .await
    }
    .await;
    if let Err(e) = trusted {
        report.check(env, "configure", false, json!(e.to_string()));
        return;
    }

    // (c) A commit, then at once a request: the request leaves only after the
    // commit is synced on the shell.
    let shell_log = client.debug_sync_log().await.is_ok();
    let title = format!("self-test {}", (env.now_ms)() as u64);
    client.start_trace();
    let created = core.create_space(Some(title.clone())).await;
    let started = (env.now_ms)();
    let terms = core.current_terms().await;
    let attested_ms = (env.now_ms)() - started;
    let trace = client.take_trace();
    let syncs = if shell_log {
        client.debug_sync_log().await.ok()
    } else {
        None
    };
    if let Err(e) = &created {
        report.check(env, "commit", false, json!(e.to_string()));
        return;
    }
    let (pass, detail) = barrier_held(&trace, syncs.as_deref());
    report.check(env, "barrier", pass, detail);

    // (a) Attested over the shell's pipe, and refused for another
    // measurement.
    report.check(
        env,
        "attested",
        terms.is_ok(),
        json!({
            "request": result_text(&terms.map(|t| format!("{} terms documents", t.len()))),
            "ms": attested_ms,
        }),
    );
    let refused = async {
        core.set_base_url(config.bad_url.clone()).await?;
        let refused = core.current_terms().await;
        core.set_base_url(config.good_url.clone()).await?;
        Ok::<_, AppError>(refused)
    }
    .await;
    report.check(
        env,
        "wrong measurement refused",
        matches!(&refused, Ok(Err(AppError::Attestation { message })) if message.contains("trusted_measurements")),
        json!(match &refused {
            Ok(Ok(_)) => "the request went through".to_owned(),
            Ok(Err(e)) | Err(e) => e.to_string(),
        }),
    );

    bundle_store(client, env, report).await;
    throughput(client, config, env, report).await;

    // (b) Leave the space for the relaunch to find, durably.
    let flushed = core.flush_storage().await;
    let state = serde_json::to_vec(&State {
        title: title.clone(),
    })
    .unwrap_or_default();
    let saved = client.files_write(STATE, 0, &state, true).await;
    let files = client.files_list().await;
    report.check(
        env,
        "persisted",
        flushed.is_ok() && saved.is_ok(),
        json!({
            "space": title,
            "flush": result_text(&flushed.map(|()| "ok".to_owned())),
            "files": files.map_err(|e| e.to_string()),
            "webview_storage": env.webview,
        }),
    );
}

async fn relaunch<T: Transport>(
    client: &ShellClient<T>,
    config: &Config,
    env: &Env,
    report: &mut Report,
    state: State,
) {
    report.phase = 2;
    let Some(core) = boot_core(client, config, env, report).await else {
        return;
    };
    let spaces = core.list_spaces(false).await;
    let found = spaces
        .as_ref()
        .map(|spaces| {
            spaces
                .iter()
                .any(|s| s.title.as_deref() == Some(&state.title))
        })
        .unwrap_or(false);
    report.check(
        env,
        "space survived the relaunch",
        found,
        json!({
            "space": state.title,
            "spaces": spaces.map(|s| s.len()).map_err(|e| e.to_string()),
        }),
    );
    // The trust settings survived too: attested again, configuring nothing.
    let terms = core.current_terms().await;
    report.check(
        env,
        "trust settings survived the relaunch",
        terms.is_ok(),
        json!(result_text(
            &terms.map(|t| format!("{} terms documents", t.len()))
        )),
    );
    // Nothing of the app's lives in the webview's own storage: every store
    // the page could see is absent or empty after the relaunch.
    let webview: Value = serde_json::from_str(&env.webview).unwrap_or(Value::Null);
    let empty = webview
        .as_object()
        .is_some_and(|stores| stores.values().all(|v| v.is_null() || v == &json!(0)));
    report.check(env, "nothing in webview storage", empty, webview);
    let _ = client.files_remove(STATE).await;
}

fn result_text<E: std::fmt::Display>(result: &Result<String, E>) -> String {
    match result {
        Ok(text) => text.clone(),
        Err(e) => format!("error: {e}"),
    }
}

/// Whether every file call posted before the first `tcp.connect` had its
/// reply first, a sync of the write-ahead log among them.
fn barrier_held(trace: &[TraceEntry], syncs: Option<&[String]>) -> (bool, Value) {
    let lines: Vec<String> = trace
        .iter()
        .map(|e| {
            let mut line = format!("#{} {}", e.id, e.method);
            if !e.target.is_empty() {
                let _ = write!(line, " {}", e.target);
            }
            match e.event {
                TraceEvent::Sent => line.insert_str(0, "-> "),
                TraceEvent::Replied { ok } => {
                    line.insert_str(0, "<- ");
                    if !ok {
                        line.push_str(" (error)");
                    }
                }
            }
            line
        })
        .collect();
    let Some(connect) = trace
        .iter()
        .position(|e| e.method == "tcp.connect" && e.event == TraceEvent::Sent)
    else {
        return (false, json!({"why": "no request was sent", "trace": lines}));
    };
    let before = &trace[..connect];
    let unanswered: Vec<u64> = before
        .iter()
        .filter(|e| e.method.starts_with("files.") && e.event == TraceEvent::Sent)
        .filter(|sent| {
            !before
                .iter()
                .any(|r| r.id == sent.id && matches!(r.event, TraceEvent::Replied { ok: true }))
        })
        .map(|e| e.id)
        .collect();
    let wal_written = before
        .iter()
        .any(|e| e.method == "files.write" && e.target.ends_with("-wal"));
    let wal_synced = before.iter().any(|e| {
        e.method == "files.sync"
            && e.target.ends_with("-wal")
            && e.event == (TraceEvent::Replied { ok: true })
    });
    let shell_flushed = syncs.map(|s| s.iter().any(|path| path.ends_with("-wal")));
    let pass = unanswered.is_empty() && wal_written && wal_synced && shell_flushed != Some(false);
    (
        pass,
        json!({
            "unanswered_before_connect": unanswered,
            "wal_written_before_connect": wal_written,
            "wal_synced_before_connect": wal_synced,
            "shell_sync_log": syncs,
            "trace": lines,
        }),
    )
}

/// Stage a two-file bundle, seal it, check the shell's digest against the
/// contract's definition, and remove it.
async fn bundle_store<T: Transport>(client: &ShellClient<T>, env: &Env, report: &mut Report) {
    let version = "selftest";
    let files: [(&str, &[u8]); 2] = [
        ("index.html", b"<!doctype html><title>x</title>"),
        ("a/b.txt", b"self-test"),
    ];
    let mut lines: Vec<(String, String)> = files
        .iter()
        .map(|(path, bytes)| (path.to_string(), hex(&Sha256::digest(bytes))))
        .collect();
    lines.sort();
    let manifest: String = lines
        .iter()
        .map(|(path, sha)| format!("{sha}  {path}\n"))
        .collect();
    let expected = hex(&Sha256::digest(manifest.as_bytes()));
    let result = async {
        client.bundle_begin(version).await?;
        for (path, bytes) in files {
            client.bundle_write(version, path, 0, bytes).await?;
        }
        let sealed = client.bundle_seal(version).await?;
        let state = client.bundle_state().await?;
        client.bundle_remove(version).await?;
        Ok::<_, crate::protocol::ShellError>((sealed, state))
    }
    .await;
    match result {
        Ok((sealed, state)) => report.check(
            env,
            "bundle store",
            sealed.digest == expected && sealed.files == 2,
            json!({
                "sealed": sealed.digest,
                "expected": expected,
                "serving": state.serving.version,
                "active": state.active,
            }),
        ),
        Err(e) => report.check(env, "bundle store", false, json!(e.to_string())),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

async fn throughput<T: Transport>(
    client: &ShellClient<T>,
    config: &Config,
    env: &Env,
    report: &mut Report,
) {
    // The bridge's round trip, with nothing to carry.
    let calls = 200;
    let started = (env.now_ms)();
    let mut answered = 0;
    for _ in 0..calls {
        if client.hello().await.is_ok() {
            answered += 1;
        }
    }
    let ms = (env.now_ms)() - started;
    let line = json!({
        "what": "bridge round trip (shell.hello)",
        "calls": calls,
        "answered": answered,
        "ms_per_call": ms / calls as f64,
    });
    (env.log)(&format!("self-test: throughput {line}"));
    report.measurements.push(line);

    let total = config.throughput_mib << 20;
    for max in [64 << 10, 256 << 10, 1 << 20] {
        let started = (env.now_ms)();
        let result = {
            let echo = std::pin::pin!(echo(client, config, total, max));
            match futures_util::future::select(echo, (env.sleep_ms)(60_000)).await {
                futures_util::future::Either::Left((result, _)) => result,
                futures_util::future::Either::Right(_) => Err("no echo within 60 s".into()),
            }
        };
        let ms = (env.now_ms)() - started;
        let line = match result {
            Ok(()) => json!({
                "what": "tcp echo (bytes each way)",
                "receive_max": max,
                "bytes": total,
                "ms": ms,
                "mib_per_s": (total as f64 / (1 << 20) as f64) / (ms / 1000.0),
            }),
            Err(e) => json!({"what": "tcp echo", "receive_max": max, "error": e}),
        };
        (env.log)(&format!("self-test: throughput {line}"));
        report.measurements.push(line);
    }

    let name = "selftest-throughput.bin";
    let chunk: Vec<u8> = (0..1 << 20).map(|i: u32| (i % 251) as u8).collect();
    let mib = 8;
    let started = (env.now_ms)();
    let mut written = Ok(());
    for i in 0..mib {
        if let Err(e) = client
            .files_write(name, (i << 20) as u64, &chunk, false)
            .await
        {
            written = Err(e);
            break;
        }
    }
    let synced = client.files_sync(name).await;
    let write_ms = (env.now_ms)() - started;
    let started = (env.now_ms)();
    let read = client.files_read(name).await;
    let read_ms = (env.now_ms)() - started;
    let intact = matches!(&read, Ok(Some(bytes)) if bytes.len() == mib << 20 && bytes.chunks(1 << 20).all(|c| c == chunk));
    let _ = client.files_remove(name).await;
    let line = json!({
        "what": "files: 8 MiB as 1 MiB writes + sync, then one read",
        "write_ms": write_ms,
        "write_mib_per_s": mib as f64 / (write_ms / 1000.0),
        "read_ms": read_ms,
        "read_mib_per_s": mib as f64 / (read_ms / 1000.0),
        "ok": written.is_ok() && synced.is_ok() && intact,
    });
    (env.log)(&format!("self-test: throughput {line}"));
    report.measurements.push(line);
}

/// Send `total` bytes to the echo server and read them back, concurrently.
async fn echo<T: Transport>(
    client: &ShellClient<T>,
    config: &Config,
    total: usize,
    receive_max: usize,
) -> Result<(), String> {
    let socket = ShellDialer::new(client.clone())
        .receive_max(receive_max)
        .connect(&config.echo_host, config.echo_port)
        .await
        .map_err(|e| e.to_string())?;
    let (mut reader, mut writer) = tokio::io::split(socket);
    let pattern = |i: usize| (i % 253) as u8;
    let write = async {
        // A whole number of periods, so every chunk starts the pattern afresh.
        let chunk: Vec<u8> = (0..253 * 259).map(pattern).collect();
        let mut sent = 0;
        while sent < total {
            let n = chunk.len().min(total - sent);
            writer
                .write_all(&chunk[..n])
                .await
                .map_err(|e| format!("send: {e}"))?;
            sent += n;
        }
        writer.shutdown().await.map_err(|e| format!("send: {e}"))
    };
    let read = async {
        let mut buf = vec![0u8; 256 << 10];
        let mut got = 0usize;
        let mut tail = Vec::new();
        loop {
            let n = reader.read(&mut buf).await?;
            if n == 0 {
                break;
            }
            for &b in &buf[..n] {
                if got < total {
                    if b != pattern(got) {
                        return Err(std::io::Error::other(format!("byte {got} differs")));
                    }
                    got += 1;
                } else {
                    tail.push(b);
                }
            }
        }
        Ok::<_, std::io::Error>((got, tail))
    };
    let read = async { read.await.map_err(|e| format!("receive: {e}")) };
    // Either side failing ends both: a reader that stopped would otherwise
    // leave the writer waiting on a peer that waits on the reader.
    let ((), (got, tail)) = futures_util::future::try_join(write, read).await?;
    let bye = format!("bye {total}\n");
    if got != total || tail != bye.as_bytes() {
        return Err(format!(
            "echoed {got} of {total} bytes, then {:?}",
            String::from_utf8_lossy(&tail)
        ));
    }
    Ok(())
}
