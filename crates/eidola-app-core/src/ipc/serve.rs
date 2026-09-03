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
//! - **No frame this module writes is one its own reader would refuse.** Every
//!   terminal frame is measured and every chunk is split — see the ceilings in
//!   [`crate::ipc`], and `chunk_lines` for why a chunk is split where a result
//!   is refused.
//! - **The connection ends the *answers*, not the work.** A teardown stops
//!   frames being written; it does **not** stop a turn. See "What a lost caller
//!   costs" below — the distinction is billing-relevant and easy to state
//!   backwards.
//! - **How the loop ends decides what happens to what is still running**
//!   ([`Ending`]). A clean end of the *request* stream is a caller that stopped
//!   asking, not one that stopped listening, so its in-flight requests are
//!   awaited and the writer drains; a dead writer or a fatal frame error tears
//!   down instead.
//! - **The connection owns its whole task tree** ([`AbortOnDrop`]), because a
//!   dropped `JoinHandle` detaches its task rather than ending it — so the
//!   writer and a turn's chunk forwarder would otherwise outlive the connection
//!   they belong to.
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
    ProtocolError, Request, SpacesListResult, WalletCredentialsResult, WireError, chunk_lines,
    decode_request, path_bytes, terminal_error_line, terminal_line,
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
    // Guarded, not merely held: whoever cancels this future is cancelling the
    // connection, and a bare `JoinHandle` dropped on that path would *detach*
    // the writer rather than end it — leaving the socket's write half open and
    // still delivering (see `AbortOnDrop`).
    let mut pen = AbortOnDrop::new(tokio::spawn(write_frames(writer, outbox)));
    read_frames(core, app_version, reader, out).await;
    // Dropping the last sender ends the writer once it has drained — which is
    // what flushes the final terminal frame before the socket closes. Awaited
    // with the guard still standing, so a cancellation landing mid-drain takes
    // the writer with it.
    pen.join().await;
}

/// A spawned task that ends with whoever holds this.
///
/// **Dropping a `JoinHandle` detaches its task; it does not stop it.** That is
/// the whole reason this type exists: `serve_connection` and its request tasks
/// each spawn a helper, and on a cancellation — the socket closing on a full
/// shutdown, a connection torn down — the future holding the handle is dropped
/// at its next await point. Bare handles left the writer and a turn's chunk
/// forwarder running, still holding the socket and still queueing frames, so a
/// connection that had been "ended" went on delivering until its turn finished.
///
/// A `JoinSet` already behaves this way (its `Drop` aborts), which is why the
/// request tasks need nothing; this is the same guarantee for the two helpers
/// spawned outside one.
///
/// **What it cancels is delivery, never work.** A turn runs on the core's own
/// runtime and is detached by construction, so ending its forwarder stops the
/// frames and leaves the turn to finish and persist — the rule the module docs
/// state at length.
struct AbortOnDrop(Option<tokio::task::JoinHandle<()>>);

impl AbortOnDrop {
    fn new(handle: tokio::task::JoinHandle<()>) -> Self {
        Self(Some(handle))
    }

    /// Wait for the task to end on its own — **with the guard still armed**, so
    /// a cancellation arriving during the wait ends it rather than detaching it.
    async fn join(&mut self) {
        if let Some(handle) = self.0.as_mut() {
            let _ = handle.await;
        }
        self.0 = None;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = &self.0 {
            handle.abort();
        }
    }
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

/// What the read loop got, or why it is stopping.
enum Next {
    /// A whole frame's bytes.
    Line(Vec<u8>),
    /// A line that was not a frame this build can read.
    Bad(ProtocolError),
    /// Nothing more will be read.
    Ends(Ending),
}

/// Why the read loop stopped — **which is what decides what happens to the
/// requests still running.**
///
/// The three are not interchangeable, and collapsing them is what cost a polite
/// caller its answers: "the peer stopped asking" and "the peer stopped
/// listening" both arrived as an end of stream and both aborted everything.
#[derive(Clone, Copy)]
enum Ending {
    /// **A clean end of the request stream, which is not the end of the
    /// conversation.** `shutdown(SHUT_WR)` and then read to completion is the
    /// classic polite client: it has said everything it means to say and is
    /// waiting on the answers it already asked for. Its read half is open and
    /// every one of those answers is still deliverable, so the in-flight
    /// requests are **awaited** and the writer drains behind them.
    PeerDone,
    /// The outbox closed: the writer is gone and nothing can reach the caller.
    /// Frames are what abort stops, and there is nobody left to receive one.
    WriterGone,
    /// A frame error the protocol calls fatal — the reader abandoned a line
    /// part-way and no longer knows where the next one begins. The byte stream
    /// is desynchronized and the caller is not keeping to the protocol, so this
    /// tears down rather than drains.
    Fatal,
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
                break Ending::WriterGone;
            }
        };
    }

    // **How the loop ends decides what happens to what is still running**, so
    // the reason is what it evaluates to rather than something inferred from
    // the fact that it ended.
    let ending = loop {
        // **Nothing is read while there is nowhere to put the answer.** The
        // writer drops the outbox receiver when the socket stops taking bytes,
        // and a peer that closed only its read half goes on sending perfectly
        // good requests afterwards — so without this the loop would keep
        // dispatching work whose answer is already known to be undeliverable,
        // and `chat.stream` is billed work. Raced against the read rather than
        // checked after it, so a caller already blocked on `next_line` is let
        // go the moment the writer dies instead of at its next frame.
        let next = tokio::select! {
            biased;
            () = out.closed() => Next::Ends(Ending::WriterGone),
            read = frames.next_line() => match read {
                Ok(Some(line)) => Next::Line(line.to_vec()),
                // A clean end of the *request* stream, which is not the end of
                // the conversation — see `Ending::PeerDone`.
                Ok(None) => Next::Ends(Ending::PeerDone),
                Err(e) => Next::Bad(e),
            },
        };
        let line = match next {
            Next::Line(line) => line,
            Next::Bad(e) => {
                refuse!(NO_REQUEST, e);
                if e.is_fatal() {
                    break Ending::Fatal;
                }
                continue;
            }
            Next::Ends(reason) => break reason,
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
                break Ending::WriterGone;
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
            break Ending::WriterGone;
        };
        // Asked again on the last line before the work starts, because the wait
        // above can be long: the answer to "can this still be delivered" has to
        // be as fresh as the decision it gates.
        if out.is_closed() {
            break Ending::WriterGone;
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
    };

    match ending {
        // The caller stopped *asking*, not listening. Its read half is open and
        // it is owed the answers it already asked for, so they are awaited —
        // and the writer then drains behind them, since every task's outbox
        // sender drops as it ends. It terminates on its own: a caller that goes
        // away entirely fails the writer's next write, which drops the outbox
        // receiver, which fails every blocked send, which ends the tasks.
        Ending::PeerDone => while tasks.join_next().await.is_some() {},
        // Nothing is left to receive an answer, or the byte stream can no
        // longer be trusted to be frames. Aborting stops the frames; it does
        // not stop a turn already running on the core's runtime (see "What a
        // lost caller costs" above), and it must not — that would be
        // cancellation landing inside a durable operation.
        Ending::WriterGone | Ending::Fatal => tasks.abort_all(),
    }
}

/// The request ids currently in flight on one connection.
///
/// Exactly one terminal frame answers an id. Two live requests wearing the same
/// one would put two terminal frames on it, and nothing downstream could say
/// which result belonged to which ask — so the second is refused before it is
/// dispatched. An id becomes free again the moment its request ends, because
/// reuse after a terminal frame is ordinary: a long-lived connection counting
/// from one would otherwise have to remember forever.
///
/// **The claim is held until the terminal frame is *queued*, and queued is
/// enough** — which is worth stating, because "released before the writer put
/// it on the wire" reads like a hole and is not one. A request task drops its
/// claim at the end of its body, after the send it awaits, so the terminal
/// frame is in the outbox before the id can be reclaimed; the read loop cannot
/// accept a reuse until then, and the reuse's own frames are therefore queued
/// strictly behind it. One writer drains that queue in order, so the caller's
/// read stream is `chunks… terminal(N)` then `chunks… terminal(N)` — a
/// sequential exchange with no position at which a reuse's frame has arrived
/// and the first terminal has not. Waiting for the writer to *acknowledge* the
/// write would buy nothing a caller can observe, and would put the read loop
/// behind the socket rather than behind the queue. Regression:
/// `a_reused_id_never_overlaps_the_exchange_it_reuses`.
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
            // The socket is in the data directory; the config root is the
            // other half of the profile, and only this side knows it. Its
            // own bytes, because the caller compares it against a path of
            // its own and a lossy rendering would not survive that.
            config_dir: Some(path_bytes(core.config_dir())),
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
        Call::WalletLifecycle => {
            let credentials = core.wallet_lifecycle().await.map_err(failed)?;
            ok(super::WalletLifecycleResult { credentials })
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
            let mut pump = AbortOnDrop::new(tokio::spawn(async move {
                'events: while let Some(event) = rx.recv().await {
                    // A delta's size is the backend's decision, not this app's,
                    // so it goes out through `chunk_lines` — which splits an
                    // oversized one into frames that fit rather than refusing
                    // it. Several frames concatenate to exactly what one would
                    // have said, and they leave in order through this one
                    // outbox.
                    for line in chunk_lines(id, &event, MAX_RESPONSE_BYTES) {
                        // Awaited, so a reader that has fallen behind slows the
                        // turn's delivery instead of being queued at. A writer
                        // that has gone away ends the pump: there is nobody to
                        // deliver to, and the turn itself is unaffected either
                        // way.
                        if chunks.send(line).await.is_err() {
                            break 'events;
                        }
                    }
                }
            }));
            // Detached by construction: `chat_stream` spawns the turn on the
            // core's runtime, so a caller that disappears mid-turn stops being
            // written to and the turn still lands. That is deliberate — see
            // the module docs.
            let result = core.chat_stream(prompt, model, space_id, tx).await;
            // `chat_stream` drops the sender as it returns, so this completes
            // once the last chunk has been queued — which is what puts every
            // `chunk` ahead of the `end` on the wire. Guarded, so a request
            // task cancelled mid-turn takes its forwarder rather than leaving
            // it queueing frames at a connection that has ended.
            pump.join().await;
            ok(result.map_err(failed)?)
        }
    }
}
