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
//! - **`hello` is the one verb answered in the loop**, before the next line is
//!   read. It is what establishes that both sides speak the same protocol, so
//!   nothing a caller pipelines behind it may start first — and a dispatched
//!   verb is only *started* before the next line, never finished. It reads no
//!   database and makes no request, so serialising it costs nothing.
//! - **A request's terminal frame is always sent.** Every path through
//!   [`answer`] ends in exactly one `end` or `err`, so a caller waiting on an
//!   id is never left waiting on a request that quietly evaporated.
//! - **The connection ends the *answers*, not the work.** In-flight tasks are
//!   aborted when the reader stops, because their only consumer is gone. That
//!   stops frames being written; it does **not** stop a turn. See "What a lost
//!   caller costs" below — the distinction is billing-relevant and easy to
//!   state backwards.
//!
//! ## Backpressure, and who waits on whom
//!
//! The outbox is **bounded** ([`WRITER_QUEUE_FRAMES`]) and every send is
//! awaited, so a caller that stops reading stops being served rather than being
//! buffered at. Without that, the concurrency limit bounds nothing that matters:
//! a request releases its permit as soon as its answer is *enqueued*, so a
//! caller pipelining valid requests and never reading would have the app
//! holding every answer it had produced, each of which can be tens of megabytes.
//!
//! The waiting is a chain, not a cycle, which is what makes it safe: the writer
//! waits on the socket, a finished request waits on the writer **while holding
//! its permit**, and the read loop waits on a permit — or, for the one verb it
//! answers itself, on the writer directly, which is the same chain one link
//! shorter. Pressure therefore travels outward to the one place that can
//! relieve it — the caller reading its socket — and nothing in the chain waits
//! on anything behind it. A turn's chunk forwarder joins the same chain: it
//! waits on the writer, and the request awaiting it holds its permit meanwhile.
//! That stalls *this* connection's other requests, which is the intended answer
//! to "you are not reading".
//!
//! **It always terminates.** If the caller never reads, it eventually goes
//! away; the writer's socket write then fails, it drops the receiver, and every
//! blocked send fails immediately rather than waiting — tasks unwind, permits
//! release. The one thing that keeps growing meanwhile is the turn's own event
//! channel (`chat_stream`'s sender is unbounded and belongs to app-core), and
//! that is bounded by one answer's length rather than by the caller's appetite.
//!
//! **And the writer's death ends the reading, rather than being noticed only by
//! whoever next tries to send.** A peer that closes its read half and keeps
//! writing is still a well-behaved-looking caller from the reader's side, so
//! nothing in the frames themselves says the answers have nowhere to go — and
//! dispatching one anyway starts work that is billed (`chat.stream`) for a
//! result that provably cannot be delivered. The read loop therefore races
//! `next_line` against the outbox closing and asks once more before it
//! dispatches. In-flight requests keep the semantics below: their frames stop,
//! their turns do not.
//!
//! ## What a lost caller costs
//!
//! **A turn already under way finishes and persists, whatever happens to the
//! connection that asked for it.** `AppCore::chat_stream` runs the turn on the
//! core's own runtime and awaits it; dropping that await detaches the turn
//! rather than cancelling it, so the answer is written, the credential settles,
//! and the post is in the space when the caller comes back.
//!
//! That is the house rule, not an accident of this module — `crates/eidola-gui/STATE.md`
//! → Atomicity & cancellation: *cancellation may only ever land between durable
//! operations, never inside one*. A turn is a saga across the database, the
//! upstream and the wallet; aborting it part-way is precisely the in-memory
//! rollback that doctrine refuses, and it would land inside a spend. The GUI
//! behaves the same way — closing a window mid-turn abandons the receiver and
//! the turn completes (`bridge::bridge`).
//!
//! It is also the honest outcome on its own terms: the tokens were spent the
//! moment the request went upstream, so discarding the answer would bill for
//! nothing. What the caller loses is the *delivery* — the `end` frame it was
//! waiting for — never the work it paid for.
//!
//! **The one thing that is racy is whether the turn started at all.** A caller
//! that vanishes in the same breath as its request can be gone before
//! `chat_stream` has handed the turn to the runtime, and then nothing runs and
//! nothing is spent. So the two outcomes are *never started* and *finished* —
//! there is no third one where a caller is charged for half a turn. Regression:
//! `a_turn_already_upstream_outlives_the_caller_that_asked_for_it`, which waits
//! for the request to reach the upstream before leaving, because that is the
//! moment the turn stops being free to discard.
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
    Call, FrameReader, HelloResult, MAX_RESPONSE_BYTES, NO_REQUEST, PROTOCOL_VERSION,
    ProtocolError, Request, Response, SpacesListResult, WalletCredentialsResult, WireError,
    decode_request, encode_line, terminal_error_line, terminal_line,
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

/// How many finished frames may sit waiting for the socket at once.
///
/// **The queue has to be bounded or the concurrency limit means nothing.** A
/// request releases its permit as soon as its answer is *enqueued*, so with an
/// unbounded queue a caller that pipelines valid requests and simply never
/// reads makes the app hold every answer it has produced — and a result can be
/// tens of megabytes. Sixteen requests at a time then bounds nothing but the
/// number of threads doing the allocating.
///
/// One slot per permitted request is the natural pairing: no request can queue
/// a second answer without first releasing and re-acquiring a permit, so what a
/// connection can hold is bounded by the concurrency limit rather than by how
/// fast the caller can type.
const WRITER_QUEUE_FRAMES: usize = MAX_IN_FLIGHT_REQUESTS;

/// Serve one connection until the peer goes away.
///
/// `app_version` is the version of the process answering — the socket's owner
/// knows it; this module does not.
pub async fn serve_connection<R, W>(core: Arc<AppCore>, app_version: String, reader: R, writer: W)
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (out, outbox) = mpsc::channel::<Vec<u8>>(WRITER_QUEUE_FRAMES);
    let pen = tokio::spawn(write_frames(writer, outbox));
    read_frames(core, app_version, reader, out).await;
    // Dropping the last sender ends the writer once it has drained — which is
    // what flushes the final terminal frame before the socket closes.
    let _ = pen.await;
}

/// Drain the outbox onto the wire, in order, until the connection ends.
async fn write_frames<W: AsyncWrite + Unpin>(mut writer: W, mut outbox: mpsc::Receiver<Vec<u8>>) {
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
    out: mpsc::Sender<Vec<u8>>,
) {
    let mut frames = FrameReader::new(tokio::io::BufReader::new(reader));
    let mut tasks = tokio::task::JoinSet::new();
    let permits = Arc::new(tokio::sync::Semaphore::new(MAX_IN_FLIGHT_REQUESTS));
    // The handshake gate. Checked here rather than inside a request task, and
    // raised only once `hello`'s own answer is on the outbox — see the inline
    // answer below, which is what makes "not before the handshake" mean the
    // handshake was *answered* rather than merely recognised.
    let mut greeted = false;
    let active = ActiveIds::default();

    // Every refusal below is awaited, and a writer that has gone away ends the
    // loop rather than being ignored: with a bounded outbox a send is where a
    // stalled connection is felt, so swallowing its failure would spin. The
    // frame goes out through `terminal_error_line` like every other terminal
    // frame — a refusal is one, and measuring only the ones that succeeded
    // would leave the `err` branch as the way past the ceiling.
    macro_rules! refuse {
        ($id:expr, $err:expr) => {
            if out
                .send(terminal_error_line(
                    $id,
                    WireError::from_protocol(&$err),
                    MAX_RESPONSE_BYTES,
                ))
                .await
                .is_err()
            {
                break;
            }
        };
    }

    loop {
        // **Nothing is read while there is nowhere to put the answer.** The
        // writer drops the outbox receiver when the socket stops taking bytes,
        // and a peer that closed only its read half goes on sending perfectly
        // good requests afterwards — so without this the loop would keep
        // dispatching work whose answer is already known to be undeliverable,
        // and `chat.stream` is billed work. Raced against the read rather than
        // checked after it, so a caller already blocked on `next_line` is let
        // go the moment the writer dies instead of at its next frame.
        let next: Option<Result<Vec<u8>, ProtocolError>> = tokio::select! {
            biased;
            () = out.closed() => None,
            read = frames.next_line() => match read {
                Ok(Some(line)) => Some(Ok(line.to_vec())),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            },
        };
        let line = match next {
            Some(Ok(line)) => line,
            Some(Err(e)) => {
                refuse!(NO_REQUEST, e);
                if e.is_fatal() {
                    break;
                }
                continue;
            }
            None => break,
        };

        let request = match decode_request(&line) {
            Ok(request) => request,
            Err(e) => {
                // No readable id, so this answers no request in particular.
                refuse!(NO_REQUEST, e);
                continue;
            }
        };

        // The reserved id is judged before the claim, because it is the one id
        // that is never anybody's to hold. Its refusal is already uncorrelated:
        // `request.id` *is* `NO_REQUEST` here.
        if request.id == NO_REQUEST {
            refuse!(
                NO_REQUEST,
                ProtocolError::ReservedRequestId {
                    reserved: request.id,
                }
            );
            continue;
        }

        // **Claimed before any refusal that would echo the id**, which is what
        // keeps "exactly one frame ever wears an id" true for refusals and not
        // just for results. A frame whose id is already live gets none of the
        // judgements below — there is nothing this connection can say *about
        // that id* while it belongs to a request still going to answer on it —
        // so the refusal goes on `NO_REQUEST` and carries the id as data.
        //
        // Everything after this point holds the claim, so the refusals that do
        // echo `request.id` are provably unambiguous; dropping `in_flight` on
        // the way out of the iteration frees it again.
        let Some(in_flight) = active.claim(request.id) else {
            refuse!(
                NO_REQUEST,
                ProtocolError::DuplicateRequestId {
                    duplicate: request.id,
                }
            );
            continue;
        };

        if let Some(refusal) = gate(&request, greeted) {
            refuse!(request.id, refusal);
            continue;
        }

        let call = match Call::parse(&request.verb, &request.params) {
            Ok(call) => call,
            Err(e) => {
                refuse!(request.id, e);
                continue;
            }
        };

        let verb = call.verb();

        // **`hello` is answered here rather than dispatched**, so the gate above
        // rises on an answer that exists.
        //
        // A dispatched verb is only *started* before the next line is read, and
        // `hello`'s result was produced in its own task — so the flag went up
        // while the answer was still being made, and a caller that pipelines
        // `hello` with a turn had that turn go upstream, billed, before it had
        // been told what it was talking to. The single ordered outbox does not
        // cover it either: it preserves the order frames are *queued* in, and
        // nothing ordered a spawned task against the read loop that carried on
        // without it — a refusal the loop writes itself could reach the caller
        // ahead of the handshake it was refused for.
        //
        // Answering in the loop costs nothing worth having: `hello` reads no
        // database and makes no request, so there was never a reason for it to
        // be concurrent, and it takes no concurrency permit because it occupies
        // nothing. Everything above still applies to it — it is decoded,
        // version-gated, parsed, and holds its id claim while it answers — so
        // the refusal ordering is unchanged; only where the answer is produced
        // has moved.
        if matches!(call, Call::Hello) {
            let line = match answer(&core, &app_version, request.id, call, &out).await {
                Ok(data) => terminal_line(request.id, verb, data, MAX_RESPONSE_BYTES),
                Err(error) => terminal_error_line(request.id, error, MAX_RESPONSE_BYTES),
            };
            if out.send(line).await.is_err() {
                break;
            }
            // Queued before the next line is read, and one writer drains the
            // outbox in order — so the handshake's answer is on the wire ahead
            // of any frame belonging to anything that followed it.
            greeted = true;
            continue;
        }

        // Waiting here is the backpressure: a caller that has saturated the
        // connection simply stops being read from until a slot frees.
        let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
            break;
        };
        // Asked again on the last line before the work starts, because the wait
        // above can be long: the answer to "can this still be delivered" has to
        // be as fresh as the decision it gates.
        if out.is_closed() {
            break;
        }
        let core = Arc::clone(&core);
        let app_version = app_version.clone();
        let out = out.clone();
        let id = request.id;
        tasks.spawn(async move {
            // Released when this task ends, however it ends — including an
            // abort — so an id is never stranded by a connection tearing down.
            let _in_flight = in_flight;
            let _permit = permit;
            // Neither arm is `encode_line` directly: both are terminal frames
            // under the same ceiling, and both carry a payload this app does
            // not get to bound — a result grows with the profile, and a
            // failure's message can be an upstream's own error body. The
            // measured seam is what keeps the app from writing a line its own
            // reader would refuse, whichever way the verb went.
            let line = match answer(&core, &app_version, id, call, &out).await {
                Ok(data) => terminal_line(id, verb, data, MAX_RESPONSE_BYTES),
                Err(error) => terminal_error_line(id, error, MAX_RESPONSE_BYTES),
            };
            // Awaited, and the permit is still held: that is how a caller who
            // stops reading stops being served instead of being buffered.
            let _ = out.send(line).await;
        });

        // Reap finished tasks so the set does not grow across a long-lived
        // connection. `try_join_next` never blocks the read loop.
        while tasks.try_join_next().is_some() {}
    }

    // The peer is gone, so nothing is left to receive an answer. Aborting stops
    // the frames; it does not stop a turn already running on the core's runtime
    // (see "What a lost caller costs" above), and it must not — that would be
    // cancellation landing inside a durable operation.
    tasks.abort_all();
}

/// The request ids currently in flight on one connection.
///
/// Exactly one terminal frame answers an id. Two live requests wearing the same
/// one would put two terminal frames on it, and nothing downstream could say
/// which result belonged to which ask — so the second is refused before it is
/// dispatched. An id becomes free again the moment its request ends, because
/// reuse after a terminal frame is ordinary: a long-lived connection counting
/// from one would otherwise have to remember forever.
#[derive(Clone, Default)]
struct ActiveIds(Arc<std::sync::Mutex<std::collections::HashSet<u64>>>);

impl ActiveIds {
    /// Claim `id` for a request about to be dispatched. `None` when it is
    /// already in flight.
    fn claim(&self, id: u64) -> Option<InFlight> {
        let mut held = self.0.lock().unwrap_or_else(|e| e.into_inner());
        held.insert(id).then(|| InFlight {
            ids: self.clone(),
            id,
        })
    }
}

/// Holds one id for the life of its request and frees it on the way out —
/// through `Drop`, so a task that is aborted with the connection releases its
/// id exactly like one that answered.
struct InFlight {
    ids: ActiveIds,
    id: u64,
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.ids
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.id);
    }
}

/// The refusals that are decided before a verb is even parsed.
///
/// Both answer **on the caller's id**, which is only sound because the caller
/// reaches here holding a claim on it: an id another request is still going to
/// answer on never gets this far (see the read loop). The reserved id is judged
/// before the claim rather than here, since it is nobody's to hold.
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
    out: &mpsc::Sender<Vec<u8>>,
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
        Call::BackendList => {
            let backends = core.list_backends().await.map_err(failed)?;
            ok(super::BackendListResult { backends })
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
                    // Awaited, so a reader that has fallen behind slows the
                    // turn's delivery instead of being queued at. A writer that
                    // has gone away ends the pump: there is nobody to deliver
                    // to, and the turn itself is unaffected either way.
                    if chunks
                        .send(encode_line(&Response::chunk(id, data)))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
            // Detached by construction: `chat_stream` spawns the turn on the
            // core's runtime, so a caller that disappears mid-turn stops being
            // written to and the turn still lands. That is deliberate — see
            // the module docs.
            let result = core.chat_stream(prompt, model, space_id, tx).await;
            // `chat_stream` drops the sender as it returns, so this completes
            // once the last chunk has been queued — which is what puts every
            // `chunk` ahead of the `end` on the wire.
            let _ = pump.await;
            ok(result.map_err(failed)?)
        }
    }
}
