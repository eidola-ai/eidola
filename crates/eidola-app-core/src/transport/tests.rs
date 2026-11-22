//! The host-socket transport's HTTP against loopback servers: what it writes
//! is what the native client writes, and its redirects and timeouts follow
//! the native client's rules.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::StreamExt as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;

/// One request as a server read it.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

type Answer = Box<dyn Fn(&Seen) -> String + Send + Sync>;

/// A loopback server answering its `n`th connection's request with
/// `answers[n]` (the last one repeated), one request per connection, and
/// recording what it read.
async fn server(answers: Vec<Answer>) -> (u16, Arc<Mutex<Vec<Seen>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    let answers = Arc::new(answers);
    tokio::spawn(async move {
        let mut n = 0;
        while let Ok((mut socket, _)) = listener.accept().await {
            let answers = answers.clone();
            let log = log.clone();
            let index = n.min(answers.len() - 1);
            n += 1;
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut byte = [0u8; 1];
                while !buf.ends_with(b"\r\n\r\n") {
                    match socket.read(&mut byte).await {
                        Ok(1) => buf.push(byte[0]),
                        _ => return,
                    }
                }
                let head = String::from_utf8(buf).unwrap();
                let mut lines = head.split("\r\n");
                let mut request_line = lines.next().unwrap().split(' ');
                let method = request_line.next().unwrap().to_string();
                let path = request_line.next().unwrap().to_string();
                let headers: Vec<(String, String)> = lines
                    .filter(|l| !l.is_empty())
                    .map(|l| {
                        let (n, v) = l.split_once(':').unwrap();
                        (n.trim().to_ascii_lowercase(), v.trim().to_string())
                    })
                    .collect();
                let length = headers
                    .iter()
                    .find(|(n, _)| n == "content-length")
                    .map(|(_, v)| v.parse::<usize>().unwrap())
                    .unwrap_or(0);
                let mut body = vec![0u8; length];
                socket.read_exact(&mut body).await.unwrap();
                let request = Seen {
                    method,
                    path,
                    headers,
                    body,
                };
                let answer = (answers[index])(&request);
                log.lock().unwrap().push(request);
                let _ = socket.write_all(answer.as_bytes()).await;
                // Hold the connection for an answer that ends in a stall.
                tokio::time::sleep(Duration::from_secs(5)).await;
            });
        }
    });
    (port, seen)
}

fn ok(body: &'static str) -> Answer {
    Box::new(move |_| {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    })
}

fn redirect(status: &'static str, location: String) -> Answer {
    Box::new(move |_| {
        format!(
            "HTTP/1.1 {status}\r\nLocation: {location}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
    })
}

fn pipe_client(redirects: Redirects) -> Client {
    let host = HostNetwork {
        dialer: Arc::new(TcpDialer),
        extra_tls_roots: Vec::new(),
    };
    Client::pipe(pipe::PipeClient::plain(
        &host,
        Some(USER_AGENT),
        redirects,
        None,
    ))
}

async fn body_text(response: Response) -> Result<String, Error> {
    let mut stream = response.bytes_stream();
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk?);
    }
    Ok(String::from_utf8(out).unwrap())
}

/// The same request, built the same way, reaches a server with the same head
/// over either transport.
#[tokio::test]
async fn both_transports_write_the_same_request() {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let (port, seen) = server(vec![ok("{}")]).await;
    let native = Client::native(native::plain_client(Some(USER_AGENT), Redirects::Refuse).unwrap());
    for client in [native, pipe_client(Redirects::Refuse)] {
        let response = client
            .post(format!("http://127.0.0.1:{port}/v1/account?x=1"))
            .basic_auth("id", Some("secret"))
            .header("X-Extra", "a")
            .header("X-Extra", "b")
            .json(&serde_json::json!({ "credits": 3 }))
            .send()
            .await
            .expect("sent");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_text(response).await.unwrap(), "{}");
    }
    let seen = seen.lock().unwrap();
    let (native, pipe) = (&seen[0], &seen[1]);
    let sorted = |s: &Seen| {
        let mut h = s.headers.clone();
        h.sort();
        h
    };
    assert_eq!(
        (&native.method, &native.path, sorted(native), &native.body),
        (&pipe.method, &pipe.path, sorted(pipe), &pipe.body)
    );
    assert_eq!(pipe.header("accept"), Some("*/*"));
    assert_eq!(pipe.header("content-type"), Some("application/json"));
}

/// A deferred body is produced only once the connection is open, and sent
/// with the length it declared.
#[tokio::test]
async fn a_deferred_body_is_produced_after_the_connection_opens() {
    let (port, seen) = server(vec![ok("done")]).await;
    let asked = Arc::new(Mutex::new(false));
    let flag = asked.clone();
    let body = deferred_body(async move {
        *flag.lock().unwrap() = true;
        Ok(Bytes::from_static(b"{\"k\":1}"))
    });
    let request = pipe_client(Redirects::Refuse)
        .post(format!("http://127.0.0.1:{port}/x"))
        .header(header::CONTENT_LENGTH, 7)
        .body(body);
    assert!(!*asked.lock().unwrap());
    let response = request.send().await.expect("sent");
    assert_eq!(body_text(response).await.unwrap(), "done");
    assert!(*asked.lock().unwrap());
    assert_eq!(seen.lock().unwrap()[0].body, b"{\"k\":1}");

    // A connection that never opens never asks for the body.
    let asked = Arc::new(Mutex::new(false));
    let flag = asked.clone();
    let body = deferred_body(async move {
        *flag.lock().unwrap() = true;
        Ok(Bytes::new())
    });
    let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let err = pipe_client(Redirects::Refuse)
        .post(format!("http://127.0.0.1:{port}/x"))
        .body(body)
        .send()
        .await
        .err()
        .expect("refused");
    assert!(!err.is_builder());
    assert!(!*asked.lock().unwrap());
}

/// `303` (and a `POST` answered `301`/`302`) is followed with a `GET` and
/// no body; `307` resends the body.
#[tokio::test]
async fn redirects_rewrite_the_method_as_reqwest_does() {
    let (port, seen) = server(vec![
        redirect("303 See Other", "/b".into()),
        ok("b"),
        redirect("307 Temporary Redirect", "/c".into()),
        ok("c"),
    ])
    .await;
    let client = pipe_client(Redirects::Follow);
    let response = client
        .post(format!("http://127.0.0.1:{port}/a"))
        .json(&serde_json::json!({ "x": 1 }))
        .send()
        .await
        .unwrap();
    assert_eq!(body_text(response).await.unwrap(), "b");
    let response = client
        .post(format!("http://127.0.0.1:{port}/a"))
        .json(&serde_json::json!({ "y": 2 }))
        .send()
        .await
        .unwrap();
    assert_eq!(body_text(response).await.unwrap(), "c");

    let seen = seen.lock().unwrap();
    assert_eq!(
        (seen[1].method.as_str(), seen[1].path.as_str()),
        ("GET", "/b")
    );
    assert!(seen[1].body.is_empty());
    assert_eq!(seen[1].header("content-type"), None);
    assert_eq!(
        (seen[3].method.as_str(), seen[3].path.as_str()),
        ("POST", "/c")
    );
    assert_eq!(seen[3].body, br#"{"y":2}"#);
}

/// A `307` whose body cannot be produced again is handed back, as is every
/// redirect to a client that refuses them.
#[tokio::test]
async fn a_redirect_that_cannot_be_followed_is_the_answer() {
    let (port, seen) = server(vec![redirect("307 Temporary Redirect", "/b".into())]).await;
    let response = pipe_client(Redirects::Follow)
        .post(format!("http://127.0.0.1:{port}/a"))
        .body(deferred_body(async { Ok(Bytes::from_static(b"once")) }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    let response = pipe_client(Redirects::Refuse)
        .get(format!("http://127.0.0.1:{port}/a"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(seen.lock().unwrap().len(), 2);
}

/// A hop to another origin drops the credentials; one on the same origin
/// keeps them.
#[tokio::test]
async fn a_cross_origin_hop_drops_the_credentials() {
    let (elsewhere, elsewhere_seen) = server(vec![ok("there")]).await;
    let (port, seen) = server(vec![
        redirect("302 Found", "/same".into()),
        redirect("302 Found", format!("http://127.0.0.1:{elsewhere}/x")),
    ])
    .await;
    let response = pipe_client(Redirects::Follow)
        .get(format!("http://127.0.0.1:{port}/a"))
        .bearer_auth("token")
        .send()
        .await
        .unwrap();
    assert_eq!(body_text(response).await.unwrap(), "there");
    let seen = seen.lock().unwrap();
    assert_eq!(seen[1].header("authorization"), Some("Bearer token"));
    assert_eq!(
        elsewhere_seen.lock().unwrap()[0].header("authorization"),
        None
    );
}

#[tokio::test]
async fn redirects_stop_after_ten() {
    let (port, seen) = server(vec![redirect("302 Found", "/again".into())]).await;
    let err = pipe_client(Redirects::Follow)
        .get(format!("http://127.0.0.1:{port}/a"))
        .send()
        .await
        .err()
        .expect("refused");
    assert!(err.to_string().contains("too many redirects"), "{err}");
    assert_eq!(seen.lock().unwrap().len(), 11);
}

/// A request's timeout bounds its body too, not only its head.
#[tokio::test]
async fn a_request_timeout_bounds_the_body() {
    let (port, _) = server(vec![Box::new(|_| {
        "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial".to_string()
    })])
    .await;
    let response = pipe_client(Redirects::Refuse)
        .get(format!("http://127.0.0.1:{port}/a"))
        .timeout(Duration::from_millis(300))
        .send()
        .await
        .unwrap();
    let err = body_text(response).await.unwrap_err();
    assert!(err.to_string().contains("timed out"), "{err}");
}

/// No error from this transport carries the request URL.
#[tokio::test]
async fn errors_carry_no_url() {
    let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let err = pipe_client(Redirects::Refuse)
        .get(format!("http://127.0.0.1:{port}/model?token=secret"))
        .send()
        .await
        .err()
        .expect("refused");
    let text = crate::error::request_error_text(err);
    assert!(
        !text.contains("secret") && !text.contains("/model"),
        "{text}"
    );
}
