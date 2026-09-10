//! Reading an answer from a peer, **bounded as the bytes arrive**.
//!
//! ## The class rule
//!
//! *Every read from a peer is capped while it runs, and every await on a peer
//! has a deadline.* Neither half is optional and neither implies the other: a
//! deadline bounds elapsed time and not bytes (a backend can send gigabytes
//! inside ten seconds), and a cap bounds bytes and not time (a backend can send
//! one byte a minute forever). A "peer" here is anything on the other end of a
//! socket — an external OpenAI-compatible backend, a local engine, the Eidola
//! server — because the property that matters is that this process does not
//! choose what arrives, not who is nominally trusted.
//!
//! `Content-Length` is not the cap. It is a claim by the same party that
//! supplies the body, so it may be used to refuse *early* and never to decide
//! when to stop: the bound is enforced against bytes actually received. That is
//! the rule the updater's `fetch_url_network` already states for signed release
//! documents, and this module is where the rest of the crate says it once.
//!
//! ## Two endings, one reader
//!
//! What a caller does at the ceiling differs by surface and the difference is
//! deliberate:
//!
//! - **An API answer** (a model catalog, an account call) is refused. A
//!   truncated JSON document is not a smaller answer, it is no answer, and the
//!   caller has somewhere honest to put the failure.
//! - **A proxied completion** keeps what it read, records it as the truncation
//!   it is, and answers the caller a gateway failure — because the Record is
//!   evidence a reader goes looking for, and a body that stopped at this app's
//!   ceiling is a fact about the exchange rather than a reason to forget it.
//!
//! Both come out of [`read_bounded`]; only the ending differs.

use std::borrow::Cow;

use crate::error::AppError;

/// The ceiling for a small API answer — a model catalog, an account or
/// credential call.
///
/// These documents are kilobytes: a catalog of a hundred models with full
/// pricing is well under a hundred of them. Two megabytes is a ceiling rather
/// than a size, generous enough that no honest server meets it and small enough
/// that a dishonest one cannot spend this process's memory through it.
pub(crate) const API_ANSWER_MAX_BYTES: usize = 2 << 20;

/// One answer read from a peer, bounded.
#[derive(Default)]
pub(crate) struct BoundedBody {
    /// What was read, to the ceiling.
    pub(crate) bytes: Vec<u8>,
    /// How much arrived — larger than `bytes` only where the ceiling stopped
    /// the read part-way through a chunk.
    pub(crate) received: usize,
    /// Whether the ceiling is why the read stopped.
    pub(crate) over_ceiling: bool,
}

impl BoundedBody {
    pub(crate) fn text(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.bytes)
    }
}

/// Read a response body, stopping at `ceiling` bytes.
///
/// The bound is applied **while reading**: `Response::text()` and
/// `Response::json()` buffer whatever the peer sends before anything can cap
/// it, so a ceiling checked afterwards is a ceiling on the value and not on the
/// cost.
pub(crate) async fn read_bounded(
    response: reqwest::Response,
    ceiling: usize,
) -> Result<BoundedBody, AppError> {
    use futures_util::StreamExt;

    let mut stream = response.bytes_stream();
    let mut body = BoundedBody::default();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AppError::Network {
            message: format!(
                "failed to read the response: {}",
                crate::error::request_error_text(e)
            ),
        })?;
        body.received += chunk.len();
        let room = ceiling.saturating_sub(body.bytes.len());
        if chunk.len() > room {
            body.bytes.extend_from_slice(&chunk[..room]);
            body.over_ceiling = true;
            break;
        }
        body.bytes.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Read a small API answer: its status and its text, refused if it passes
/// [`API_ANSWER_MAX_BYTES`].
///
/// The refusal is the honest ending for this shape — a truncated document does
/// not parse, so continuing would only turn a size failure into a confusing
/// syntax one — and it names the ceiling rather than the size, because the size
/// is the peer's claim.
pub(crate) async fn read_api_answer(
    response: reqwest::Response,
    what: &str,
) -> Result<(reqwest::StatusCode, String), AppError> {
    let status = response.status();
    let body = read_bounded(response, API_ANSWER_MAX_BYTES).await?;
    if body.over_ceiling {
        return Err(AppError::Network {
            message: format!(
                "{what} was larger than the {API_ANSWER_MAX_BYTES}-byte ceiling this app reads"
            ),
        });
    }
    Ok((status, body.text().into_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-shot loopback server answering with `body`. The bound lives on a
    /// real `reqwest::Response`, so the only honest way to hold it is against
    /// one.
    async fn serving_once(body: Vec<u8>) -> std::net::SocketAddr {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            // Read the request first: answering over unread bytes and closing
            // resets the connection, which the client meets as a read failure.
            let mut request = Vec::new();
            let mut byte = [0u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                match socket.read(&mut byte).await {
                    Ok(1) => request.push(byte[0]),
                    _ => break,
                }
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Connection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(&body).await;
            let _ = socket.shutdown().await;
        });
        addr
    }

    async fn get(addr: std::net::SocketAddr) -> reqwest::Response {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        reqwest::Client::builder()
            .build()
            .expect("client")
            .get(format!("http://{addr}/"))
            .send()
            .await
            .expect("send")
    }

    /// **The bound is on the bytes, not on the value.** A ceiling checked after
    /// `text()` has buffered the whole answer bounds what is kept and not what
    /// the read cost, which is the half a peer can spend.
    #[tokio::test]
    async fn a_peers_answer_is_bounded_while_it_arrives() {
        let addr = serving_once(vec![b'x'; API_ANSWER_MAX_BYTES + 4096]).await;
        let body = read_bounded(get(addr).await, API_ANSWER_MAX_BYTES)
            .await
            .expect("read");
        assert!(
            body.bytes.len() <= API_ANSWER_MAX_BYTES,
            "the read stops at the ceiling rather than after it: {} bytes",
            body.bytes.len()
        );
        assert!(body.over_ceiling, "and it knows why it stopped");
    }

    /// An API answer that passes the ceiling is **refused**, because a
    /// truncated document is not a smaller answer.
    #[tokio::test]
    async fn an_oversized_api_answer_is_refused_rather_than_truncated() {
        let addr = serving_once(vec![b'x'; API_ANSWER_MAX_BYTES + 1]).await;
        let refused = read_api_answer(get(addr).await, "the model list")
            .await
            .expect_err("an answer past the ceiling is no answer");
        assert!(
            refused.to_string().contains("ceiling"),
            "and it says so in the peer's own terms: {refused}"
        );

        // An ordinary answer comes back whole, with its status.
        let addr = serving_once(br#"{"data":[]}"#.to_vec()).await;
        let (status, text) = read_api_answer(get(addr).await, "the model list")
            .await
            .expect("read");
        assert_eq!(status, reqwest::StatusCode::OK);
        assert_eq!(text, r#"{"data":[]}"#);
    }
}
