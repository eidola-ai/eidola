//! Serving one connection: read frames, run verbs, write frames back.
//!
//! Deliberately transport-agnostic — it takes any async reader and writer, so
//! the process that owns the socket owns *binding* it and authenticating the
//! peer, while the answers to the verbs stay beside the [`AppCore`] methods
//! they wrap. It is also what lets the whole protocol be exercised over an
//! in-memory pipe, with no socket and no privileges.
//!
//! ## Shape
//!
//! One reader loop, one writer task, and one task per in-flight request:
//!
//! - **Every frame leaves through the writer task**, over a channel, so two
//!   requests answering at once cannot interleave halves of a line.
//! - **Requests run concurrently** (that is what the id in the frame is for),
//!   bounded by [`MAX_IN_FLIGHT_REQUESTS`] — the read loop waits for a permit
//!   rather than refusing, which is backpressure a caller feels as slowness
//!   instead of as an error it has to handle.
//! - **A request's terminal frame is always sent.** Every path through
//!   [`answer`] ends in exactly one `end` or `err`, so a caller waiting on an
//!   id is never left waiting on a request that quietly evaporated.
//! - **The connection ends the requests.** In-flight tasks are aborted when the
//!   reader stops, because their only consumer is gone; a turn cancelled that
//!   way is exactly what pressing Ctrl-C on an embedded caller does.
//!
//! ## What a bad frame costs
//!
//! One request, not the connection — a refusal is typed, carries the id it
//! answers, and the loop reads on. The single exception is
//! [`ProtocolError::FrameTooLarge`], which is fatal because the reader gave up
//! part-way through a line and no longer knows where the next frame begins.
//!
//! [`AppCore`]: crate::AppCore

use std::sync::Arc;

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;

use super::{
    Call, FrameReader, HelloResult, NO_REQUEST, PROTOCOL_VERSION, ProtocolError, Request, Response,
    SpacesListResult, WalletCredentialsResult, WireError, decode_request, encode_line,
};
use crate::AppCore;
use crate::error::AppError;

/// How many requests one connection may have in flight at once.
///
/// A ceiling rather than a refusal: the read loop waits for a slot, so a
/// caller that pipelines hard is slowed, never failed. Ample for the one
/// consumer this protocol has (a command-line client, whose only long verb is
/// a turn) and low enough that a runaway peer cannot spawn without bound.
pub const MAX_IN_FLIGHT_REQUESTS: usize = 16;

/// Serve one connection until the peer goes away.
///
/// `app_version` is the version of the process answering — the socket's owner
/// knows it; this module does not.
pub async fn serve_connection<R, W>(core: Arc<AppCore>, app_version: String, reader: R, writer: W)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (out, outbox) = mpsc::unbounded_channel::<Vec<u8>>();
    let pen = tokio::spawn(write_frames(writer, outbox));
    read_frames(core, app_version, reader, out).await;
    // Dropping the last sender ends the writer once it has drained — which is
    // what flushes the final terminal frame before the socket closes.
    let _ = pen.await;
}

/// Drain the outbox onto the wire, in order, until the connection ends.
async fn write_frames<W: AsyncWrite + Unpin>(
    mut writer: W,
    mut outbox: mpsc::UnboundedReceiver<Vec<u8>>,
) {
    while let Some(line) = outbox.recv().await {
        if writer.write_all(&line).await.is_err() {
            break;
        }
        if writer.flush().await.is_err() {
            break;
        }
    }
}

/// The reader loop: parse, gate, dispatch.
async fn read_frames<R: AsyncRead + Unpin>(
    core: Arc<AppCore>,
    app_version: String,
    reader: R,
    out: mpsc::UnboundedSender<Vec<u8>>,
) {
    let mut frames = FrameReader::new(tokio::io::BufReader::new(reader));
    let mut tasks = tokio::task::JoinSet::new();
    let permits = Arc::new(tokio::sync::Semaphore::new(MAX_IN_FLIGHT_REQUESTS));
    // The handshake gate. Checked here rather than inside a request task so
    // there is no window in which a racing verb slips past a `hello` that has
    // not been answered yet.
    let mut greeted = false;

    loop {
        let line = match frames.next_line().await {
            Ok(Some(line)) => line.to_vec(),
            Ok(None) => break,
            Err(e) => {
                let _ = out.send(encode_line(&Response::err(
                    NO_REQUEST,
                    WireError::from_protocol(&e),
                )));
                if e.is_fatal() {
                    break;
                }
                continue;
            }
        };

        let request = match decode_request(&line) {
            Ok(request) => request,
            Err(e) => {
                // No readable id, so this answers no request in particular.
                let _ = out.send(encode_line(&Response::err(
                    NO_REQUEST,
                    WireError::from_protocol(&e),
                )));
                continue;
            }
        };

        if let Some(refusal) = gate(&request, greeted) {
            let _ = out.send(encode_line(&Response::err(
                request.id,
                WireError::from_protocol(&refusal),
            )));
            continue;
        }

        let call = match Call::parse(&request.verb, &request.params) {
            Ok(call) => call,
            Err(e) => {
                let _ = out.send(encode_line(&Response::err(
                    request.id,
                    WireError::from_protocol(&e),
                )));
                continue;
            }
        };

        if matches!(call, Call::Hello) {
            greeted = true;
        }

        // Waiting here is the backpressure: a caller that has saturated the
        // connection simply stops being read from until a slot frees.
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            break;
        };
        let core = Arc::clone(&core);
        let app_version = app_version.clone();
        let out = out.clone();
        let id = request.id;
        tasks.spawn(async move {
            let _permit = permit;
            let terminal = match answer(&core, &app_version, id, call, &out).await {
                Ok(data) => Response {
                    v: PROTOCOL_VERSION,
                    id,
                    body: super::ResponseBody::End { data },
                },
                Err(error) => Response::err(id, error),
            };
            let _ = out.send(encode_line(&terminal));
        });

        // Reap finished tasks so the set does not grow across a long-lived
        // connection. `try_join_next` never blocks the read loop.
        while tasks.try_join_next().is_some() {}
    }

    // The peer is gone: nothing is left to receive an answer, so anything
    // still running is work with no reader. `JoinSet`'s drop aborts it.
    tasks.abort_all();
}

/// The two refusals that are decided before a verb is even parsed.
fn gate(request: &Request, greeted: bool) -> Option<ProtocolError> {
    if request.v != PROTOCOL_VERSION {
        return Some(ProtocolError::UnsupportedProtocol {
            supported: PROTOCOL_VERSION,
            requested: request.v,
        });
    }
    if !greeted && request.verb != "hello" {
        return Some(ProtocolError::HandshakeRequired);
    }
    None
}

/// Run one verb, streaming any chunks through `out`, and answer with the
/// `end` payload or a typed failure.
async fn answer(
    core: &AppCore,
    app_version: &str,
    id: u64,
    call: Call,
    out: &mpsc::UnboundedSender<Vec<u8>>,
) -> Result<serde_json::Value, WireError> {
    fn ok<T: serde::Serialize>(value: T) -> Result<serde_json::Value, WireError> {
        serde_json::to_value(value).map_err(|e| {
            WireError::from_app_error(&AppError::Internal {
                message: format!("could not render the result: {e}"),
            })
        })
    }
    fn failed(e: AppError) -> WireError {
        WireError::from_app_error(&e)
    }

    match call {
        Call::Hello => ok(HelloResult {
            protocol: PROTOCOL_VERSION,
            app_version: app_version.to_string(),
        }),
        Call::SpacesList { include_archived } => {
            let spaces = core.list_spaces(include_archived).await.map_err(failed)?;
            ok(SpacesListResult { spaces })
        }
        Call::AccountShow => {
            let account = core.account_show().await.map_err(failed)?;
            ok(account)
        }
        Call::WalletCredentials => {
            let credentials = core.wallet_credentials().await.map_err(failed)?;
            ok(WalletCredentialsResult { credentials })
        }
        Call::SpacesArchive { space_id } => {
            let archived = core.archive_space(space_id).await.map_err(failed)?;
            ok(super::SpacesArchiveResult { archived })
        }
        Call::SpacesRename { space_id, title } => {
            core.rename_space(space_id, title).await.map_err(failed)?;
            ok(super::Done {})
        }
        Call::AccountPrices => {
            let prices = core.account_prices().await.map_err(failed)?;
            ok(super::AccountPricesResult { prices })
        }
        Call::AccountBalances => {
            let balances = core.account_balances().await.map_err(failed)?;
            ok(balances)
        }
        Call::AccountCheckout { price_id } => {
            let url = core.account_checkout(price_id).await.map_err(failed)?;
            ok(super::AccountCheckoutResult { url })
        }
        Call::WalletSpending => {
            let credentials = core.wallet_spending_credentials().await.map_err(failed)?;
            ok(super::WalletSpendingResult { credentials })
        }
        Call::WalletRecover => {
            let recovered = core.recover_spending_credentials().await.map_err(failed)?;
            ok(super::WalletRecoverResult { recovered })
        }
        Call::BackendList => {
            let backends = core.list_backends().await.map_err(failed)?;
            ok(super::BackendListResult { backends })
        }
        Call::BackendSetEnabled { id, enabled } => {
            core.set_backend_enabled(id, enabled)
                .await
                .map_err(failed)?;
            ok(super::Done {})
        }
        Call::BackendModels { id } => {
            let models = core.backend_models(id).await.map_err(failed)?;
            ok(super::BackendModelsResult { models })
        }
        Call::ModelList => {
            let state = core.local_models_state().await.map_err(failed)?;
            // Read after the scan, deliberately: the caller reconciles the two,
            // and an engine that started during the scan is better reported as
            // running-but-unlisted than as listed-but-not-running.
            let running = core.running_engines();
            ok(super::ModelListResult { state, running })
        }
        Call::ModelDownload { url } => {
            let id = core.download_local_model(url).await.map_err(failed)?;
            ok(super::ModelDownloadResult { id })
        }
        Call::ModelDelete { id } => {
            core.delete_local_model(id).await.map_err(failed)?;
            ok(super::Done {})
        }
        Call::ModelLoad { id } => {
            core.load_local_model(id).await.map_err(failed)?;
            ok(super::Done {})
        }
        Call::ModelUnload { id } => {
            core.unload_local_model(id).await.map_err(failed)?;
            ok(super::Done {})
        }
        Call::ModelSetPinned { id, pinned } => {
            core.set_local_model_pinned(id, pinned)
                .await
                .map_err(failed)?;
            ok(super::Done {})
        }
        Call::UpdateCheck => ok(core.update_check().await),
        Call::ChatDefaultModel => {
            let model = core.default_model().await.map_err(failed)?;
            ok(super::DefaultModelResult { model })
        }
        Call::ChatStream {
            prompt,
            model,
            space_id,
        } => {
            // Resolving the default model needs the database, which is exactly
            // what the caller does not have — so it is resolved here rather
            // than being something a client has to guess.
            let model = match model {
                Some(model) => model,
                None => core.default_model().await.map_err(failed)?,
            };
            let (tx, mut rx) = mpsc::unbounded_channel::<crate::ChatStreamEvent>();
            let chunks = out.clone();
            // Drains to the end even when the peer has stopped listening: the
            // turn is running and paid for either way, and a receiver dropped
            // under it would only turn a finished turn into a torn one.
            let pump = tokio::spawn(async move {
                while let Some(event) = rx.recv().await {
                    let data = serde_json::to_value(&event).unwrap_or(serde_json::Value::Null);
                    let _ = chunks.send(encode_line(&Response::chunk(id, data)));
                }
            });
            let result = core.chat_stream(prompt, model, space_id, tx).await;
            // `chat_stream` drops the sender as it returns, so this completes
            // once the last chunk has been queued — which is what puts every
            // `chunk` ahead of the `end` on the wire.
            let _ = pump.await;
            ok(result.map_err(failed)?)
        }
    }
}
