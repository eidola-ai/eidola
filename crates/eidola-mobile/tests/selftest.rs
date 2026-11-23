//! The self-test's verdict follows what its transfers did: an echo that
//! comes back wrong, or a file that reads back wrong, fails it, however fast.

mod fake_shell;

use eidola_mobile::client::{BoxFuture, ShellClient};
use eidola_mobile::selftest::{self, Config, Env};
use fake_shell::FakeShell;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// Like `mobile/tools/echo_server.py`, flipping one byte when `corrupt`.
async fn echo_server(corrupt: bool) -> u16 {
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
                            if corrupt && total == 0 {
                                buf[0] ^= 1;
                            }
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

fn env() -> Env {
    fn sleep_ms(ms: u32) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep(std::time::Duration::from_millis(
            ms.into(),
        )))
    }
    Env {
        now_ms: || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64()
                * 1000.0
        },
        log: |_| {},
        sleep_ms,
        webview: String::new(),
    }
}

fn config(port: u16) -> Config {
    Config {
        good_url: String::new(),
        bad_url: String::new(),
        measurement: String::new(),
        host_data: String::new(),
        guest_policy: String::new(),
        ark_pem: String::new(),
        ask_pem: String::new(),
        tls_ca_pem: String::new(),
        echo_host: "127.0.0.1".into(),
        echo_port: port,
        throughput_mib: 1,
        update_url: None,
    }
}

async fn verdict(echo_corrupts: bool, reads_corrupt: bool) -> (bool, String) {
    let port = echo_server(echo_corrupts).await;
    let shell = FakeShell::default();
    shell.state.lock().unwrap().corrupt_reads = reads_corrupt;
    selftest::transfers(&ShellClient::new(shell), &config(port), &env()).await
}

#[tokio::test]
async fn correct_transfers_pass() {
    let (pass, report) = verdict(false, false).await;
    assert!(pass, "{report}");
}

#[tokio::test]
async fn an_echo_that_comes_back_wrong_fails_the_verdict() {
    let (pass, report) = verdict(true, false).await;
    assert!(!pass, "{report}");
    assert!(report.contains("differs"), "{report}");
}

#[tokio::test]
async fn a_file_that_reads_back_wrong_fails_the_verdict() {
    let (pass, report) = verdict(false, true).await;
    assert!(!pass, "{report}");
    assert!(report.contains("differ from the"), "{report}");
}

#[tokio::test]
async fn an_echo_that_cannot_connect_fails_the_verdict() {
    let free = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = free.local_addr().unwrap().port();
    drop(free);
    let shell = FakeShell::default();
    let (pass, report) = selftest::transfers(&ShellClient::new(shell), &config(port), &env()).await;
    assert!(!pass, "{report}");
}
