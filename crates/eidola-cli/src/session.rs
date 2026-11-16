//! Who answers this run's commands — this process, or the Eidola already
//! running.
//!
//! ## The selection rule
//!
//! Dial the control socket first.
//!
//! - **It answers** ⇒ client mode. The app holds the profile and does the
//!   work; this process renders it.
//! - **Nothing is listening** (no socket file, or one nobody answers on) ⇒
//!   embedded mode: open the profile here, exactly as before the socket
//!   existed. This is the ordinary case for a machine with no app running.
//! - **The profile is held but nothing answered for it** ⇒ say so. This is the
//!   case that must never become a silent wait: an older Eidola, or one whose
//!   listener stopped, holds the lock this process cannot take and serves no
//!   socket to ask instead. It arrives two ways — no socket plus
//!   [`AppError::DatabaseInUse`], and a socket that accepted a connection and
//!   never greeted it — and both say the same actionable thing.
//! - **The profile is held by another `eidola` command** ⇒ wait for it to
//!   let go, bounded by `--lock-wait` and announced once — commands with no
//!   app running take turns rather than failing. The lockfile says what holds
//!   the profile ([`LockHolder`]); only a command is waited on this way.
//! - **It answers, for another profile** ⇒ refuse, in the handshake, before a
//!   verb is dispatched at it. A profile is *both* roots and the socket is
//!   found through the data one alone, so an app reached this way has not yet
//!   been shown to speak for the account, default template and update feed
//!   this command was given. It states its config root in the greeting; one
//!   that is not ours is [`Startup::OtherProfile`], naming both.
//!
//! `--embedded` skips the dial entirely, which is the debugging escape hatch;
//! it does not weaken anything, since the lock still decides who may open the
//! profile.
//!
//! ## Why the two modes share one surface
//!
//! Every method here answers with the same type in both modes, and failures
//! arrive as the same typed [`Failure`], because the wire carries
//! [`AppError`]'s variants rather than prose. So the commands themselves are
//! written once: what changes between modes is who ran the work, never how the
//! answer is rendered or which hint a failure earns.
//!
//! ## What client mode cannot do
//!
//! The socket has no verb for the trust bundle, and deliberately never will
//! until there is a per-client consent layer — so `configure`, and the bare
//! invocation that reads the bundle back, need the profile in *this* process.
//! The account's own identity and consent (`account create`, `accept-terms`,
//! `reset`, `configure`, `allocate`) and the backend registry's membership
//! (`backend add`, `backend remove`) are held to the same line: they decide
//! what this installation *is* and where its prompts may go, which is exactly
//! what a consent layer would gate. Each is refused by name, with the remedy,
//! rather than half-working.

use std::path::PathBuf;

use eidola_app_core::backends::BackendInfo;
use eidola_app_core::db::{self, LockHolder};
use eidola_app_core::error::AppError;
use eidola_app_core::ipc::{
    AccountCheckoutResult, AccountPricesResult, BackendModelsResult, Call, DefaultModelResult,
    Done, ModelDownloadResult, ModelListResult, SpacesArchiveResult, SpacesListResult,
    WalletLifecycleResult, WalletRecoverResult, WalletSpendingResult,
};
use eidola_app_core::updates::UpdateCheckSnapshot;
use eidola_app_core::{
    AccountShowResult, AppCore, BalancesResult, ChatResult, ChatStreamEvent,
    CredentialLifecycleInfo, InFlightCredentialInfo, ModelInfo, PriceInfo, SpaceInfo,
};

use crate::client::{Client, Dial, Failure};

/// Who is answering.
enum Mode {
    /// This process holds the profile.
    Embedded(AppCore),
    /// A running Eidola holds it, and we are asking it.
    Client {
        /// The client's I/O is bound to the runtime that created it, so the
        /// two live and die together.
        runtime: tokio::runtime::Runtime,
        /// One request at a time — a command is one operation, and the lock
        /// is what makes that a fact rather than a convention.
        client: tokio::sync::Mutex<Client>,
    },
}

/// How long a command waits for the listener of a process that **already
/// holds the profile**, before reporting that the holder does not serve it.
///
/// Sized against what the wait actually contains, which has no network in it:
/// a database open and schema validation, the sub-space driver starting, the
/// window system launching, and the app's stores and startup refreshes — all
/// of it between the lock and the bind. Two seconds covers that on a cold,
/// slow start while staying short enough that a genuinely unserved profile is
/// still reported promptly. Nothing waits this long on the ordinary paths:
/// a machine with no app running never reaches here.
const STARTUP_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// How long a command waits, by default, for **another `eidola` command**
/// that holds the profile to finish with it (`--lock-wait` overrides it).
///
/// The local database has one writer, so two commands run against a profile
/// no app is serving take turns: the second waits for the first rather than
/// failing, which is what lets a script run commands side by side. Ten
/// minutes is sized against the longest ordinary command — an agentic chat
/// turn with tools — times a handful queued behind it, while still ending a
/// wait on a command that will never finish (`eidola model load` holds the
/// profile until it is stopped) in a stated failure that names it.
pub const DEFAULT_LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(600);

/// The gap before the second look, doubling up to [`MAX_RETRY_GAP`]. Small
/// first, because most of this race is decided in milliseconds and a command
/// that could have run immediately should.
const FIRST_RETRY: std::time::Duration = std::time::Duration::from_millis(25);

/// The longest gap between looks, so the tail of a slow start is still
/// noticed promptly rather than slept through.
const MAX_RETRY_GAP: std::time::Duration = std::time::Duration::from_millis(250);

/// The session a command runs against.
pub struct Session {
    mode: Mode,
    /// This profile's `config.toml`. Held in both modes, because in client
    /// mode it is the *same file* the answering app reads — the handshake
    /// refuses an app composed from another config root, so whoever answers
    /// this session shares this path.
    config_path: PathBuf,
}

/// Why no session could be opened.
#[derive(Debug)]
pub enum Startup {
    /// Opening the profile in this process failed.
    Core(AppError),
    /// The profile is held by a process that does not answer for it.
    NotAccepting { pid: Option<u32> },
    /// The app answering for this data directory composes its profile from a
    /// different config root, so it speaks for another account, another
    /// default template and another update feed than this command was given.
    OtherProfile { ours: PathBuf, theirs: PathBuf },
    /// The conversation with the running app failed before it began.
    Dial(Failure),
    /// Another `eidola` command held the profile for the whole of the wait.
    Busy {
        pid: Option<u32>,
        waited: std::time::Duration,
    },
}

impl std::fmt::Display for Startup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Startup::Core(e) => write!(f, "{e}"),
            Startup::NotAccepting { pid } => {
                write!(f, "Eidola is running")?;
                if let Some(pid) = pid {
                    write!(f, " (pid {pid})")?;
                }
                write!(f, " but is not accepting local connections")
            }
            // Both roots printed rather than compared: the decision was made
            // on their bytes, and this is only the sentence about it, so a
            // root that does not render exactly still names itself as well as
            // it can be named.
            Startup::OtherProfile { ours, theirs } => write!(
                f,
                "the Eidola running for this data directory reads its config from \
                 {}, not {} — it speaks for a different profile",
                theirs.display(),
                ours.display()
            ),
            Startup::Dial(e) => write!(f, "{e}"),
            Startup::Busy { pid, waited } => {
                write!(f, "another `eidola` command")?;
                if let Some(pid) = pid {
                    write!(f, " (pid {pid})")?;
                }
                write!(
                    f,
                    " still holds this profile after waiting {}s",
                    waited.as_secs()
                )
            }
        }
    }
}

/// What a failure to open the profile means, given that nothing answered the
/// control socket.
///
/// [`AppError::DatabaseInUse`] is the one that changes meaning here. On its
/// own it says another Eidola holds the profile — quit it. Reached *through*
/// an unanswered socket it says something more specific and more useful: the
/// holder is not serving the socket that exists to make this very command
/// work, which is what an Eidola older than the socket looks like.
pub fn unanswered(e: AppError) -> Startup {
    match e {
        AppError::DatabaseInUse { pid, .. } => Startup::NotAccepting { pid },
        other => Startup::Core(other),
    }
}

impl Session {
    /// [`Session::open_waiting`] with the default lock wait and no notice —
    /// the shape most tests want.
    #[cfg(test)]
    pub fn open(
        config_dir: PathBuf,
        data_dir: PathBuf,
        embedded: bool,
    ) -> Result<Session, Startup> {
        Self::open_waiting(
            config_dir,
            data_dir,
            embedded,
            DEFAULT_LOCK_WAIT,
            &mut |_| {},
        )
    }

    /// Choose who answers, following the selection rule.
    ///
    /// One loop, because every branch of the rule can be waiting on the same
    /// two things — a socket appearing, or the lock coming free — and each
    /// look is the full question again:
    ///
    /// 1. **Dial** (skipped under `--embedded`). An answer for this profile
    ///    is client mode; a refusal is final.
    /// 2. **Open the profile here.** Success is embedded mode.
    /// 3. **Refused because it is held**, and the lockfile says by whom:
    ///    - an `eidola` **command** ([`LockHolder::Command`]) lets go when its
    ///      command ends, so wait for it — up to `lock_wait`, announced once
    ///      through `on_wait` with the holder's pid, and ending in
    ///      [`Startup::Busy`] rather than an unbounded wait;
    ///    - **the app**, or a holder that has not said, holds the profile for
    ///      its whole life and is expected to *serve* it: wait only
    ///      [`STARTUP_GRACE`] for its listener, then report
    ///      [`Startup::NotAccepting`] naming it.
    ///
    /// A command's wait can end in client mode: an app started meanwhile
    /// takes the lock the moment the command lets go and binds its socket a
    /// moment later, and the next look's dial finds it. Under `--embedded`
    /// a held-by-the-app profile is refused at once, as it always was —
    /// `--embedded` forces the mode, not the lock.
    pub fn open_waiting(
        config_dir: PathBuf,
        data_dir: PathBuf,
        embedded: bool,
        lock_wait: std::time::Duration,
        on_wait: &mut dyn FnMut(Option<u32>),
    ) -> Result<Session, Startup> {
        let config_path = config_dir.join("config.toml");
        let runtime = if embedded {
            None
        } else {
            match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => Some(runtime),
                Err(e) => {
                    return Err(Startup::Core(AppError::Internal {
                        message: format!("could not start a runtime: {e}"),
                    }));
                }
            }
        };

        let started = std::time::Instant::now();
        let mut gap = FIRST_RETRY;
        // When the current run of "held by something that should be serving"
        // began. Reset whenever the holder turns out to be a command, so an
        // app that takes over from one gets its whole startup grace.
        let mut app_held_since: Option<std::time::Instant> = None;
        let mut announced = false;
        loop {
            if let Some(dialer) = &runtime {
                match Self::dial(dialer, &config_dir, &data_dir) {
                    Ok(client) => {
                        return Ok(Session {
                            mode: Mode::Client {
                                runtime: runtime.expect("dialled through it"),
                                client: tokio::sync::Mutex::new(client),
                            },
                            config_path,
                        });
                    }
                    Err(Dial::NoListener) => {}
                    Err(Dial::NotAccepting) => return Err(Startup::NotAccepting { pid: None }),
                    // Refused during the handshake, so no verb was ever
                    // dispatched at it: the app answering is not serving this
                    // command's profile.
                    Err(Dial::OtherProfile { ours, theirs }) => {
                        return Err(Startup::OtherProfile { ours, theirs });
                    }
                    Err(Dial::Failed(e)) => return Err(Startup::Dial(e)),
                }
            }
            match AppCore::new_for_command(config_dir.clone(), data_dir.clone()) {
                Ok(core) => return Ok(Session::from_core(core, config_path)),
                Err(AppError::DatabaseInUse { pid, message }) => {
                    if db::lock_holder(&data_dir) == Some(LockHolder::Command) {
                        app_held_since = None;
                        if started.elapsed() + gap > lock_wait {
                            return Err(Startup::Busy {
                                pid,
                                waited: lock_wait,
                            });
                        }
                        if !announced {
                            announced = true;
                            on_wait(pid);
                        }
                    } else {
                        let e = AppError::DatabaseInUse { pid, message };
                        if embedded {
                            return Err(Startup::Core(e));
                        }
                        let since = *app_held_since.get_or_insert_with(std::time::Instant::now);
                        if since.elapsed() + gap >= STARTUP_GRACE {
                            // The grace ran out with nothing serving it. The
                            // holder is an Eidola older than the socket, or
                            // one whose listener stopped — which is what
                            // `unanswered` says, naming the process holding
                            // the lock.
                            return Err(unanswered(e));
                        }
                    }
                }
                Err(other) => return Err(unanswered(other)),
            }
            std::thread::sleep(gap);
            gap = (gap * 2).min(MAX_RETRY_GAP);
        }
    }

    /// One dial of the control socket, with every handshake gate the
    /// selection rule applies — the profile check included, so a redial can
    /// no more adopt another profile's app than the first attempt could.
    fn dial(
        runtime: &tokio::runtime::Runtime,
        config_dir: &std::path::Path,
        data_dir: &std::path::Path,
    ) -> Result<Client, Dial> {
        #[cfg(test)]
        if tests::dial_finds_nothing() {
            return Err(Dial::NoListener);
        }
        runtime.block_on(Client::connect(config_dir, data_dir))
    }

    /// Embedded mode: this process opens the profile.
    ///
    /// **The sub-space turn driver is deliberately not started here**, and the
    /// GUI deliberately does start it (`stores::open_app_core`). A delegated
    /// conversation is driven by a loop that plans, takes a turn, and plans
    /// again until the room stops — work measured in model round trips, with
    /// no window waiting on it. This process exits when its command does, so
    /// starting that loop would mean beginning walks it cannot finish and
    /// turns whose answers nobody would be told about.
    ///
    /// The consequence is stated rather than hidden: a command here **can**
    /// open a delegated conversation through the API, and that room simply
    /// does not run until the app is next open, at which point its startup
    /// sweep picks it up exactly as it picks up any room a previous run left
    /// mid-delegation. In client mode the room is driven immediately, because
    /// the process answering is the one running the driver.
    fn from_core(core: AppCore, config_path: PathBuf) -> Session {
        Session {
            mode: Mode::Embedded(core),
            config_path,
        }
    }

    /// The credentials this profile holds **right now**, named by
    /// [`eidola_app_core::config::account_fingerprint`].
    ///
    /// Read from `config.toml` on every call, in both modes, because the
    /// question it answers is about *now* rather than about when the session
    /// opened. It needs no lock — the file is the same one `AppCore` itself
    /// re-reads per call, and reading it is what the app does too — and in
    /// client mode it is authoritative precisely because the handshake
    /// refuses an app composed from another config root.
    ///
    /// The **pair**, not the account id: a profile can be reset and
    /// reconfigured with the same id and a different secret, and an id-only
    /// answer would call those two the same thing.
    ///
    /// `None` is a real answer (no account configured), and so is the answer
    /// after a torn write, which decodes as a default config: either way the
    /// credentials on record are not the ones a caller captured, which is the
    /// conservative direction for the one thing this is used for.
    pub fn account_fingerprint(&self) -> Option<String> {
        eidola_app_core::config::account_fingerprint(&eidola_app_core::config::Config::load_from(
            &self.config_path,
        ))
    }

    /// The runtime this session's work runs on.
    pub fn runtime(&self) -> &tokio::runtime::Runtime {
        match &self.mode {
            Mode::Embedded(core) => core.runtime(),
            Mode::Client { runtime, .. } => runtime,
        }
    }

    /// Whether the work is happening in another process.
    pub fn is_client(&self) -> bool {
        matches!(self.mode, Mode::Client { .. })
    }

    /// The version of the app answering, in client mode.
    pub async fn app_version(&self) -> Option<String> {
        match &self.mode {
            Mode::Embedded(_) => None,
            Mode::Client { client, .. } => Some(client.lock().await.app_version().to_string()),
        }
    }

    /// How many streamed events this build could not read (see
    /// [`Client::unread_events`]). Always zero in embedded mode, where there
    /// is no wire to be newer than us.
    pub async fn unread_events(&self) -> u32 {
        match &self.mode {
            Mode::Embedded(_) => 0,
            Mode::Client { client, .. } => client.lock().await.unread_events(),
        }
    }

    /// The core, for work that must run against the profile in this process.
    /// `what` names what was refused, so the reader is told which part of the
    /// command could not happen rather than that something could not.
    pub fn local(&self, what: &'static str) -> Result<&AppCore, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core),
            Mode::Client { .. } => Err(Failure::EmbeddedOnly { what }),
        }
    }

    // -- verbs ------------------------------------------------------------

    pub async fn default_model(&self) -> Result<String, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.default_model().await?),
            Mode::Client { client, .. } => {
                let r: DefaultModelResult =
                    client.lock().await.call(&Call::ChatDefaultModel).await?;
                Ok(r.model)
            }
        }
    }

    pub async fn list_spaces(&self, include_archived: bool) -> Result<Vec<SpaceInfo>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.list_spaces(include_archived).await?),
            Mode::Client { client, .. } => {
                let r: SpacesListResult = client
                    .lock()
                    .await
                    .call(&Call::SpacesList { include_archived })
                    .await?;
                Ok(r.spaces)
            }
        }
    }

    pub async fn archive_space(&self, space_id: String) -> Result<bool, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.archive_space(space_id).await?),
            Mode::Client { client, .. } => {
                let r: SpacesArchiveResult = client
                    .lock()
                    .await
                    .call(&Call::SpacesArchive { space_id })
                    .await?;
                Ok(r.archived)
            }
        }
    }

    pub async fn rename_space(&self, space_id: String, title: String) -> Result<(), Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.rename_space(space_id, title).await?),
            Mode::Client { client, .. } => {
                let _: Done = client
                    .lock()
                    .await
                    .call(&Call::SpacesRename { space_id, title })
                    .await?;
                Ok(())
            }
        }
    }

    pub async fn account_show(&self) -> Result<AccountShowResult, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.account_show().await?),
            Mode::Client { client, .. } => Ok(client.lock().await.call(&Call::AccountShow).await?),
        }
    }

    pub async fn account_prices(&self) -> Result<Vec<PriceInfo>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.account_prices().await?),
            Mode::Client { client, .. } => {
                let r: AccountPricesResult = client.lock().await.call(&Call::AccountPrices).await?;
                Ok(r.prices)
            }
        }
    }

    pub async fn account_balances(&self) -> Result<BalancesResult, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.account_balances().await?),
            Mode::Client { client, .. } => {
                Ok(client.lock().await.call(&Call::AccountBalances).await?)
            }
        }
    }

    /// Mint a checkout link, **with the identity the mint ran under**.
    ///
    /// The identity comes from whichever process signed the request, because
    /// that is the only one that knows which credentials it used. A caller's
    /// own before-and-after look cannot settle it alone: an account replaced
    /// and replaced back inside the round trip leaves those two looks
    /// agreeing about a link minted for something else in between.
    pub async fn account_checkout(
        &self,
        price_id: String,
    ) -> Result<AccountCheckoutResult, Failure> {
        match &self.mode {
            Mode::Embedded(core) => {
                let mint = core.account_checkout(price_id).await?;
                Ok(AccountCheckoutResult {
                    url: mint.url,
                    minted_for: Some(mint.minted_for),
                })
            }
            Mode::Client { client, .. } => Ok(client
                .lock()
                .await
                .call(&Call::AccountCheckout { price_id })
                .await?),
        }
    }

    /// Every credential with its lifecycle state, from one read.
    ///
    /// The listing's two sections are split out of this rather than fetched
    /// one each: settlement removes a `spending` credential and creates its
    /// `active` successor atomically, so two reads either side of it can show
    /// a credential in flight *beside* the successor it already became. That
    /// pair never existed, and one snapshot cannot claim it. **Embedded mode
    /// reads it the same way** — it had the same two-read shape and was
    /// coherent only because nothing else holds the profile to write during a
    /// one-shot command, which is a fact about deployment rather than a
    /// property of the code.
    pub async fn wallet_lifecycle(&self) -> Result<Vec<CredentialLifecycleInfo>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.wallet_lifecycle().await?),
            Mode::Client { client, .. } => {
                let r: WalletLifecycleResult =
                    client.lock().await.call(&Call::WalletLifecycle).await?;
                Ok(r.credentials)
            }
        }
    }

    pub async fn wallet_spending_credentials(
        &self,
    ) -> Result<Vec<InFlightCredentialInfo>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.wallet_spending_credentials().await?),
            Mode::Client { client, .. } => {
                let r: WalletSpendingResult =
                    client.lock().await.call(&Call::WalletSpending).await?;
                Ok(r.credentials)
            }
        }
    }

    /// Ask again for this account's issuances whose answer never arrived.
    /// `None` when another process answers for the profile: its provisioner
    /// asks at its own times, and an account-authenticated request is not
    /// sent on a caller's behalf beside the recovery of a spend.
    pub async fn recover_pending_issuances(
        &self,
    ) -> Result<Option<eidola_app_core::IssuanceRecovery>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(Some(core.recover_pending_issuances().await?)),
            Mode::Client { .. } => Ok(None),
        }
    }

    pub async fn recover_spending_credentials(&self) -> Result<Vec<String>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.recover_spending_credentials().await?),
            Mode::Client { client, .. } => {
                let r: WalletRecoverResult = client.lock().await.call(&Call::WalletRecover).await?;
                Ok(r.recovered)
            }
        }
    }

    pub async fn list_backends(&self) -> Result<Vec<BackendInfo>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.list_backends().await?),
            Mode::Client { client, .. } => {
                let r: eidola_app_core::ipc::BackendListResult =
                    client.lock().await.call(&Call::BackendList).await?;
                Ok(r.backends)
            }
        }
    }

    pub async fn set_backend_enabled(&self, id: String, enabled: bool) -> Result<(), Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.set_backend_enabled(id, enabled).await?),
            Mode::Client { client, .. } => {
                let _: Done = client
                    .lock()
                    .await
                    .call(&Call::BackendSetEnabled { id, enabled })
                    .await?;
                Ok(())
            }
        }
    }

    pub async fn backend_models(&self, id: String) -> Result<Vec<ModelInfo>, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.backend_models(id).await?),
            Mode::Client { client, .. } => {
                let r: BackendModelsResult = client
                    .lock()
                    .await
                    .call(&Call::BackendModels { id })
                    .await?;
                Ok(r.models)
            }
        }
    }

    /// The local-model scan and the engine registry, from one moment.
    ///
    /// They travel together because they only answer the question between
    /// them: the scan is what is on disk, the registry is what is running, and
    /// reconciling the two is the caller's whole job here.
    pub async fn models(&self) -> Result<ModelListResult, Failure> {
        match &self.mode {
            Mode::Embedded(core) => {
                let state = core.local_models_state().await?;
                let running = core.running_engines();
                Ok(ModelListResult { state, running })
            }
            Mode::Client { client, .. } => Ok(client.lock().await.call(&Call::ModelList).await?),
        }
    }

    pub async fn download_local_model(&self, url: String) -> Result<String, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.download_local_model(url).await?),
            Mode::Client { client, .. } => {
                let r: ModelDownloadResult = client
                    .lock()
                    .await
                    .call(&Call::ModelDownload { url })
                    .await?;
                Ok(r.id)
            }
        }
    }

    pub async fn delete_local_model(&self, id: String) -> Result<(), Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.delete_local_model(id).await?),
            Mode::Client { client, .. } => {
                let _: Done = client.lock().await.call(&Call::ModelDelete { id }).await?;
                Ok(())
            }
        }
    }

    pub async fn load_local_model(&self, id: String) -> Result<(), Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.load_local_model(id).await?),
            Mode::Client { client, .. } => {
                let _: Done = client.lock().await.call(&Call::ModelLoad { id }).await?;
                Ok(())
            }
        }
    }

    pub async fn unload_local_model(&self, id: String) -> Result<(), Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.unload_local_model(id).await?),
            Mode::Client { client, .. } => {
                let _: Done = client.lock().await.call(&Call::ModelUnload { id }).await?;
                Ok(())
            }
        }
    }

    pub async fn set_local_model_pinned(&self, id: String, pinned: bool) -> Result<(), Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.set_local_model_pinned(id, pinned).await?),
            Mode::Client { client, .. } => {
                let _: Done = client
                    .lock()
                    .await
                    .call(&Call::ModelSetPinned { id, pinned })
                    .await?;
                Ok(())
            }
        }
    }

    pub async fn update_check(&self) -> Result<UpdateCheckSnapshot, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.update_check().await),
            Mode::Client { client, .. } => Ok(client.lock().await.call(&Call::UpdateCheck).await?),
        }
    }

    /// Take a turn, streaming its events to `tx`.
    ///
    /// **An unnamed model stays unnamed all the way to the turn.** `None` is
    /// not "we have not looked it up yet" — it is the request that the
    /// process taking the turn resolve the default at the moment it starts
    /// one. Resolving it here and sending the answer would serialize a value
    /// the app can change between the lookup and the turn (the default
    /// template, or that template's agent model), and the app would then be
    /// unable to tell a stale default from a deliberate choice, because on
    /// the wire they are the same field.
    pub async fn chat_stream(
        &self,
        prompt: String,
        model: Option<String>,
        space_id: Option<String>,
        tx: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, Failure> {
        match &self.mode {
            Mode::Embedded(core) => {
                // This process owns the profile for the whole command, so
                // resolving here is resolving at turn start.
                let model = match model {
                    Some(m) => m,
                    None => core.default_model().await?,
                };
                Ok(core.chat_stream(prompt, model, space_id, tx).await?)
            }
            Mode::Client { client, .. } => {
                let call = Call::ChatStream {
                    prompt,
                    model,
                    space_id,
                };
                client.lock().await.chat_stream(&call, &tx).await
            }
        }
    }

    /// Continue an answer that stopped at its length limit
    /// (`AppCore::continue_stream`), streamed like [`Self::chat_stream`] — the
    /// text being continued first, then what the model adds. Nothing is
    /// resolved here: the answer names its own participant and model.
    pub async fn continue_stream(
        &self,
        action_id: String,
        tx: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, Failure> {
        match &self.mode {
            Mode::Embedded(core) => Ok(core.continue_stream(action_id, tx).await?),
            Mode::Client { client, .. } => {
                let call = Call::ChatContinue { action_id };
                client.lock().await.chat_stream(&call, &tx).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_app_core::ipc::{
        HelloResult, PROTOCOL_VERSION, Response, decode_request, encode_line, socket_path,
    };

    /// Answer `hello` on one connection, then read until the caller leaves.
    /// Runs on its own runtime in its own thread, because the thing under test
    /// blocks the one it is called on.
    fn greeter(dir: &std::path::Path) -> std::thread::JoinHandle<()> {
        let config_dir = eidola_app_core::ipc::path_bytes(dir);
        let listener =
            std::os::unix::net::UnixListener::bind(socket_path(dir)).expect("bind the socket");
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                listener.set_nonblocking(true).expect("nonblocking");
                let listener = tokio::net::UnixListener::from_std(listener).expect("adopt");
                let (stream, _) = listener.accept().await.expect("accept");
                let (reader, mut writer) = stream.into_split();
                let mut frames =
                    eidola_app_core::ipc::FrameReader::new(tokio::io::BufReader::new(reader));
                while let Ok(Some(line)) = frames.next_line().await {
                    let request = decode_request(line).expect("a frame");
                    let answer = Response::end(
                        request.id,
                        &HelloResult {
                            protocol: PROTOCOL_VERSION,
                            app_version: "9.9.9".into(),
                            config_dir: Some(config_dir.clone()),
                        },
                    );
                    use tokio::io::AsyncWriteExt;
                    if writer.write_all(&encode_line(&answer)).await.is_err() {
                        return;
                    }
                }
            });
        })
    }

    /// An app whose wallet **settled between** the moments a two-read listing
    /// would have sampled it.
    ///
    /// Asked for the sections separately it answers exactly as such an app
    /// honestly would either side of a settlement: the credential still in
    /// flight, and the successor it has already become — the pair that never
    /// coexisted. Asked for the lifecycle it answers from one view read, where
    /// only the successor is there. Every verb it is asked is recorded, so a
    /// listing that takes more than one read is visible rather than merely
    /// wrong.
    fn wallet_app(
        dir: &std::path::Path,
        seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) -> std::thread::JoinHandle<()> {
        use eidola_app_core::ipc::{WalletCredentialsResult, WalletLifecycleResult};
        let config_dir = eidola_app_core::ipc::path_bytes(dir);
        let listener =
            std::os::unix::net::UnixListener::bind(socket_path(dir)).expect("bind the socket");
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                listener.set_nonblocking(true).expect("nonblocking");
                let listener = tokio::net::UnixListener::from_std(listener).expect("adopt");
                let (stream, _) = listener.accept().await.expect("accept");
                let (reader, mut writer) = stream.into_split();
                let mut frames =
                    eidola_app_core::ipc::FrameReader::new(tokio::io::BufReader::new(reader));
                while let Ok(Some(line)) = frames.next_line().await {
                    let request = decode_request(line).expect("a frame");
                    seen.lock().expect("seen").push(request.verb.clone());
                    let answer = match request.verb.as_str() {
                        "hello" => Response::end(
                            request.id,
                            &HelloResult {
                                protocol: PROTOCOL_VERSION,
                                app_version: "9.9.9".into(),
                                config_dir: Some(config_dir.clone()),
                            },
                        ),
                        "wallet.lifecycle" => Response::end(
                            request.id,
                            &WalletLifecycleResult {
                                credentials: vec![lifecycle_row("successor", 2, "active", None)],
                            },
                        ),
                        // The stale halves a second read could still return.
                        "wallet.spending" => Response::end(
                            request.id,
                            &eidola_app_core::ipc::WalletSpendingResult {
                                credentials: vec![InFlightCredentialInfo {
                                    nonce: "predecessor".into(),
                                    credits: 100,
                                    generation: 1,
                                    spend_amount: 40,
                                }],
                            },
                        ),
                        "wallet.credentials" => Response::end(
                            request.id,
                            &WalletCredentialsResult {
                                credentials: vec![eidola_app_core::CredentialInfo {
                                    nonce: "successor".into(),
                                    credits: 60,
                                    generation: 2,
                                }],
                            },
                        ),
                        other => panic!("unexpected verb {other}"),
                    };
                    use tokio::io::AsyncWriteExt;
                    if writer.write_all(&encode_line(&answer)).await.is_err() {
                        return;
                    }
                }
            });
        })
    }

    fn lifecycle_row(
        nonce: &str,
        generation: i64,
        state: &str,
        spend_amount: Option<i64>,
    ) -> CredentialLifecycleInfo {
        CredentialLifecycleInfo {
            nonce: nonce.to_string(),
            credits: 60,
            generation,
            created_at: 1_000 + generation,
            state: state.to_string(),
            spend_amount,
        }
    }

    #[test]
    fn the_wallet_listing_reads_one_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let server = wallet_app(dir.path(), std::sync::Arc::clone(&seen));
        let session =
            Session::open(dir.path().into(), dir.path().into(), false).expect("client mode");

        let wallet = session
            .runtime()
            .block_on(crate::wallet_listing(&session))
            .expect("the listing reads");
        let (spending, active) = crate::wallet_sections(&wallet);
        assert!(
            spending.is_empty(),
            "the settle already happened; nothing is in flight"
        );
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].nonce, "successor");

        drop(session);
        server.join().expect("the server ends with the connection");
        assert_eq!(
            *seen.lock().expect("seen"),
            vec!["hello".to_string(), "wallet.lifecycle".to_string()],
            "the listing takes one read — a second one is what lets the wallet \
             settle underneath it"
        );
    }

    thread_local! {
        /// How many of the next dials report that nothing is listening.
        ///
        /// A test cannot schedule a listener into this race from outside: the
        /// app it stands in for takes the profile lock and binds its socket
        /// with a whole startup in between, and nothing here can be timed
        /// against that. Making the first looks miss puts the process in
        /// exactly the state the race leaves it in — a real app already
        /// holding the lock, its socket not yet there — and lets the looking
        /// be what is under test rather than a timer. Counted rather than a
        /// flag, so a listener can be made to appear *after* a given look
        /// and the waiting itself is what finds it.
        static DIALS_THAT_MISS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }

    /// Spends one forced miss, if any are left.
    pub(super) fn dial_finds_nothing() -> bool {
        DIALS_THAT_MISS.with(|f| {
            let left = f.get();
            f.set(left.saturating_sub(1));
            left > 0
        })
    }

    fn make_dials_miss(n: u32) {
        DIALS_THAT_MISS.with(|f| f.set(n));
    }

    /// An app serving `config_dir` as its config root, recording every verb
    /// it is asked so a refusal that arrived too late is visible.
    fn profile_greeter(
        dir: &std::path::Path,
        config_dir: Option<Vec<u8>>,
        seen: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) -> std::thread::JoinHandle<()> {
        let listener =
            std::os::unix::net::UnixListener::bind(socket_path(dir)).expect("bind the socket");
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            runtime.block_on(async move {
                listener.set_nonblocking(true).expect("nonblocking");
                let listener = tokio::net::UnixListener::from_std(listener).expect("adopt");
                let (stream, _) = listener.accept().await.expect("accept");
                let (reader, mut writer) = stream.into_split();
                let mut frames =
                    eidola_app_core::ipc::FrameReader::new(tokio::io::BufReader::new(reader));
                while let Ok(Some(line)) = frames.next_line().await {
                    let request = decode_request(line).expect("a frame");
                    seen.lock().expect("seen").push(request.verb.clone());
                    let answer = Response::end(
                        request.id,
                        &HelloResult {
                            protocol: PROTOCOL_VERSION,
                            app_version: "9.9.9".into(),
                            config_dir: config_dir.clone(),
                        },
                    );
                    use tokio::io::AsyncWriteExt;
                    if writer.write_all(&encode_line(&answer)).await.is_err() {
                        return;
                    }
                }
            });
        })
    }

    #[test]
    fn an_app_serving_another_profile_is_refused_before_any_verb() {
        let dir = tempfile::tempdir().expect("tempdir");
        // One data root, two config roots — the state the selection rule
        // could not see before, because the socket is found through the data
        // root alone.
        let ours = dir.path().join("ours");
        let theirs = dir.path().join("theirs");
        std::fs::create_dir(&ours).expect("our config root");
        std::fs::create_dir(&theirs).expect("their config root");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let server = profile_greeter(
            dir.path(),
            Some(eidola_app_core::ipc::path_bytes(&theirs)),
            std::sync::Arc::clone(&seen),
        );

        match Session::open(ours.clone(), dir.path().into(), false) {
            Err(Startup::OtherProfile { ours: o, theirs: t }) => {
                assert_eq!(o, ours);
                assert_eq!(t, theirs);
            }
            other => panic!(
                "an app serving another config root must not take this command: {:?}",
                other.map(|_| ())
            ),
        }
        server.join().expect("the server ends with the connection");
        assert_eq!(
            *seen.lock().expect("seen"),
            vec!["hello".to_string()],
            "the refusal lands in the handshake — nothing was dispatched at it"
        );
    }

    #[test]
    fn the_refusal_names_both_roots_and_what_to_do() {
        let rendered = Startup::OtherProfile {
            ours: "/here/eidola".into(),
            theirs: "/there/eidola".into(),
        }
        .to_string();
        assert!(rendered.contains("/here/eidola"), "{rendered}");
        assert!(rendered.contains("/there/eidola"), "{rendered}");
        assert!(rendered.contains("different profile"), "{rendered}");
    }

    #[test]
    fn the_account_fingerprint_is_read_fresh_and_names_the_pair() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _server = greeter(dir.path());
        let session =
            Session::open(dir.path().into(), dir.path().into(), false).expect("client mode");
        let write = |id: &str, secret: &str| {
            std::fs::write(
                dir.path().join("config.toml"),
                format!("account_id = \"{id}\"\naccount_secret = \"{secret}\"\n"),
            )
            .expect("write config");
        };

        assert_eq!(
            session.account_fingerprint(),
            None,
            "nothing configured yet"
        );

        // The app owns the profile in client mode, so the answer has to come
        // off disk each time it is asked rather than from anything captured
        // when the session opened.
        write("acct", "one");
        let first = session.account_fingerprint().expect("configured");
        write("acct", "one");
        assert_eq!(
            session.account_fingerprint().as_deref(),
            Some(first.as_str()),
            "the same pair is the same answer"
        );

        // Reset and reconfigured under the same id: a different credential
        // pair, and an id-only answer would call it the same one.
        write("acct", "two");
        assert_ne!(
            session.account_fingerprint().as_deref(),
            Some(first.as_str()),
            "a new secret under the same id is not the same credentials"
        );

        // And the secret itself never appears in what travels.
        let named = session.account_fingerprint().expect("configured");
        assert!(!named.contains("two"), "{named}");
        assert!(
            named.len() == 64 && named.chars().all(|c| c.is_ascii_hexdigit()),
            "{named}"
        );

        // No secret is no answer: there is nothing to mint under.
        std::fs::write(dir.path().join("config.toml"), "account_id = \"acct\"\n")
            .expect("write config");
        assert_eq!(session.account_fingerprint(), None);
    }

    #[test]
    fn a_profile_taken_between_the_dial_and_the_open_is_asked_again() {
        // Starting the app is a race a command can lose by microseconds: the
        // GUI takes the profile lock and binds its socket a moment apart, and
        // a dial that arrives in between finds nothing, then the open finds
        // the lock gone. Concluding there that the holder does not serve the
        // profile tells the reader to quit an app that is answering — so the
        // question is asked once more, because its answer has changed.
        let dir = tempfile::tempdir().expect("tempdir");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let server = profile_greeter(
            dir.path(),
            Some(eidola_app_core::ipc::path_bytes(dir.path())),
            std::sync::Arc::clone(&seen),
        );
        // The peer that won the race holds the profile and serves it.
        let holder = AppCore::new(dir.path().into(), dir.path().into()).expect("the peer's open");

        make_dials_miss(1);
        let session =
            Session::open(dir.path().into(), dir.path().into(), false).expect("the redial answers");
        assert!(
            session.is_client(),
            "an app that is serving takes the command, whichever order the \
             race resolved in"
        );

        drop(session);
        drop(holder);
        server.join().expect("the server ends with the connection");
        assert_eq!(
            *seen.lock().expect("seen"),
            vec!["hello".to_string()],
            "and the redial is a real handshake, gates and all"
        );
    }

    #[test]
    fn a_listener_that_only_appears_during_startup_is_still_found() {
        // Losing the lock does not mean the socket is up: the app that took
        // it opens and validates a database, starts the sub-space driver,
        // launches its window system and builds its stores before it binds.
        // A single look lands inside that gap far more often than not, so the
        // looking continues for a bounded moment — here the socket is missed
        // three times over and found on the fourth, which no one-shot retry
        // would reach.
        let dir = tempfile::tempdir().expect("tempdir");
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let server = profile_greeter(
            dir.path(),
            Some(eidola_app_core::ipc::path_bytes(dir.path())),
            std::sync::Arc::clone(&seen),
        );
        let holder = AppCore::new(dir.path().into(), dir.path().into()).expect("the peer's open");

        // The look before the open, and the first two of the ones after it.
        make_dials_miss(3);
        let started = std::time::Instant::now();
        let session =
            Session::open(dir.path().into(), dir.path().into(), false).expect("the app is serving");
        assert!(
            session.is_client(),
            "an app that is serving takes the command, however late it bound"
        );
        assert!(
            started.elapsed() < STARTUP_GRACE,
            "and it is found inside the grace, not at the end of it"
        );

        drop(session);
        drop(holder);
        server.join().expect("the server ends with the connection");
        assert_eq!(
            *seen.lock().expect("seen"),
            vec!["hello".to_string()],
            "and every look is a real handshake, gates and all"
        );
    }

    #[test]
    fn a_holder_that_serves_nothing_is_still_named_as_that() {
        // The other side of the same door, and the reason the waiting is
        // bounded: a lock holder that serves no socket — an Eidola older than
        // it, or one whose listener stopped — must still be reported, and
        // reported after a stated ceiling rather than waited on forever.
        let dir = tempfile::tempdir().expect("tempdir");
        let holder = AppCore::new(dir.path().into(), dir.path().into()).expect("the holder");

        make_dials_miss(1);
        let started = std::time::Instant::now();
        match Session::open(dir.path().into(), dir.path().into(), false) {
            Err(Startup::NotAccepting { .. }) => {}
            other => panic!(
                "looking again and finding nothing is still the honest refusal: {:?}",
                other.map(|_| ())
            ),
        }
        // Up to one gap short of the ceiling, never past it: the looking
        // stops as soon as another sleep would cross the deadline, so the
        // grace is a bound on the wait rather than a duration it always
        // spends.
        assert!(
            started.elapsed() >= STARTUP_GRACE - MAX_RETRY_GAP,
            "it gave the holder the grace before saying so"
        );
        assert!(
            started.elapsed() < STARTUP_GRACE + MAX_RETRY_GAP,
            "and did not wait past it"
        );
        drop(holder);
    }

    /// Hold the profile as a one-shot command would, from another thread,
    /// for `hold`, signalling once the lock is taken.
    fn command_holder(
        dir: &std::path::Path,
        hold: std::time::Duration,
    ) -> std::thread::JoinHandle<()> {
        let dir = dir.to_path_buf();
        let (taken, wait) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let lock = db::DbLock::acquire(&dir, LockHolder::Command).expect("the command's open");
            taken.send(()).expect("signal");
            std::thread::sleep(hold);
            drop(lock);
        });
        wait.recv().expect("the holder took the lock");
        holder
    }

    #[test]
    fn a_command_holding_the_profile_is_waited_for() {
        // Two commands against a profile no app is serving take turns: the
        // second opens the profile as soon as the first lets go, rather than
        // failing — and says once, naming the holder, that it is waiting.
        let dir = tempfile::tempdir().expect("tempdir");
        let hold = std::time::Duration::from_millis(400);
        let holder = command_holder(dir.path(), hold);
        let started = std::time::Instant::now();
        let mut notices = Vec::new();
        let session = Session::open_waiting(
            dir.path().into(),
            dir.path().into(),
            false,
            std::time::Duration::from_secs(30),
            &mut |pid| notices.push(pid),
        )
        .expect("the profile comes free inside the wait");
        assert!(
            !session.is_client(),
            "nothing is serving, so it is opened here"
        );
        assert!(
            started.elapsed() >= hold - std::time::Duration::from_millis(50),
            "it waited for the holder rather than racing it"
        );
        assert_eq!(
            notices,
            vec![Some(std::process::id())],
            "the wait is announced exactly once, naming the holder"
        );
        drop(session);
        holder.join().expect("the holder ends");
    }

    #[test]
    fn a_command_that_never_lets_go_ends_the_wait_as_busy() {
        // The wait is bounded and ends in a stated refusal naming the holder,
        // never a silent hang: `eidola model load` holds the profile until it
        // is stopped.
        let dir = tempfile::tempdir().expect("tempdir");
        let holder = command_holder(dir.path(), std::time::Duration::from_secs(3));
        let limit = std::time::Duration::from_millis(400);
        let started = std::time::Instant::now();
        match Session::open_waiting(
            dir.path().into(),
            dir.path().into(),
            false,
            limit,
            &mut |_| {},
        ) {
            Err(Startup::Busy { pid, waited }) => {
                assert_eq!(pid, Some(std::process::id()));
                assert_eq!(waited, limit);
            }
            other => panic!(
                "a holder outlasting the wait is Busy: {:?}",
                other.map(|_| ())
            ),
        }
        assert!(
            started.elapsed() < limit + MAX_RETRY_GAP,
            "and the wait stops at its ceiling, not past it"
        );
        let rendered = Startup::Busy {
            pid: Some(4242),
            waited: limit,
        }
        .to_string();
        assert!(
            rendered.contains("`eidola` command (pid 4242)"),
            "{rendered}"
        );
        holder.join().expect("the holder ends");
    }

    #[test]
    fn a_zero_lock_wait_refuses_at_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let holder = command_holder(dir.path(), std::time::Duration::from_secs(1));
        let started = std::time::Instant::now();
        let mut notices = 0;
        match Session::open_waiting(
            dir.path().into(),
            dir.path().into(),
            false,
            std::time::Duration::ZERO,
            &mut |_| notices += 1,
        ) {
            Err(Startup::Busy { .. }) => {}
            other => panic!("--lock-wait 0 refuses: {:?}", other.map(|_| ())),
        }
        assert!(started.elapsed() < MAX_RETRY_GAP, "without sleeping first");
        assert_eq!(notices, 0, "and announces no wait it is not going to do");
        holder.join().expect("the holder ends");
    }

    #[test]
    fn embedded_mode_waits_for_a_command_too() {
        // `--embedded` forces the mode, not the lock — and taking turns with
        // another command is the lock's business, so it waits the same way.
        let dir = tempfile::tempdir().expect("tempdir");
        let holder = command_holder(dir.path(), std::time::Duration::from_millis(300));
        let session = Session::open_waiting(
            dir.path().into(),
            dir.path().into(),
            true,
            std::time::Duration::from_secs(30),
            &mut |_| {},
        )
        .expect("the profile comes free");
        assert!(session.local("`eidola configure`").is_ok());
        drop(session);
        holder.join().expect("the holder ends");
    }

    #[test]
    fn embedded_mode_still_refuses_an_app_holder_at_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let holder = AppCore::new(dir.path().into(), dir.path().into()).expect("the app's open");
        let started = std::time::Instant::now();
        match Session::open(dir.path().into(), dir.path().into(), true) {
            Err(Startup::Core(AppError::DatabaseInUse { .. })) => {}
            other => panic!("an app holder is not waited on: {:?}", other.map(|_| ())),
        }
        assert!(started.elapsed() < MAX_RETRY_GAP);
        drop(holder);
    }

    #[test]
    fn a_command_opens_the_profile_as_a_command() {
        // What lets the *next* command wait for this one.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = Session::open(dir.path().into(), dir.path().into(), false).expect("embedded");
        assert_eq!(db::lock_holder(dir.path()), Some(LockHolder::Command));
        drop(session);
    }

    #[test]
    fn a_held_profile_nothing_answered_for_is_named_as_that() {
        let held = AppError::DatabaseInUse {
            pid: Some(4321),
            message: "held".into(),
        };
        match unanswered(held) {
            Startup::NotAccepting { pid } => assert_eq!(pid, Some(4321)),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn every_other_way_of_failing_to_open_stays_itself() {
        let broken = AppError::Database {
            message: "corrupt".into(),
        };
        match unanswered(broken) {
            Startup::Core(AppError::Database { .. }) => {}
            other => panic!("only the held profile changes meaning: {other:?}"),
        }
    }

    #[test]
    fn the_refusal_says_what_to_do_about_it() {
        let rendered = Startup::NotAccepting { pid: Some(77) }.to_string();
        assert!(rendered.contains("running"), "{rendered}");
        assert!(rendered.contains("not accepting"), "{rendered}");
        assert!(rendered.contains("77"), "the holder is named: {rendered}");
        let anonymous = Startup::NotAccepting { pid: None }.to_string();
        assert!(!anonymous.contains("pid"), "{anonymous}");
    }

    #[test]
    fn a_socket_that_answers_takes_the_command() {
        let dir = tempfile::tempdir().expect("tempdir");
        let server = greeter(dir.path());
        let session = Session::open(dir.path().into(), dir.path().into(), false)
            .expect("the socket answered");
        assert!(
            session.is_client(),
            "an app that answers is the one that does the work"
        );
        drop(session);
        server.join().expect("the server ends with the connection");
    }

    #[test]
    fn a_profile_held_with_no_socket_to_ask_is_the_honest_refusal() {
        let dir = tempfile::tempdir().expect("tempdir");
        // The profile is held and there is no socket beside it — an Eidola
        // older than the socket, from this side of the glass.
        let holder = AppCore::new(dir.path().into(), dir.path().into()).expect("first open");
        match Session::open(dir.path().into(), dir.path().into(), false) {
            Err(Startup::NotAccepting { .. }) => {}
            other => panic!(
                "a held profile with nothing answering must say so: {:?}",
                other.map(|_| ())
            ),
        }
        drop(holder);
    }

    #[test]
    fn a_command_that_needs_the_profile_here_is_refused_by_name() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _server = greeter(dir.path());
        let session =
            Session::open(dir.path().into(), dir.path().into(), false).expect("client mode");
        match session.local("`eidola configure`") {
            Err(Failure::EmbeddedOnly { what }) => assert_eq!(
                what, "`eidola configure`",
                "the refusal names the command, not just the condition"
            ),
            Ok(_) => panic!("client mode has no core to hand out"),
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn embedded_mode_hands_the_core_straight_over() {
        let dir = tempfile::tempdir().expect("tempdir");
        let session = Session::open(dir.path().into(), dir.path().into(), true).expect("embedded");
        assert!(session.local("`eidola configure`").is_ok());
    }

    #[test]
    fn embedded_skips_the_socket_even_when_it_answers() {
        let dir = tempfile::tempdir().expect("tempdir");
        let _server = greeter(dir.path());
        let session = Session::open(dir.path().into(), dir.path().into(), true)
            .expect("the profile is free to open");
        assert!(
            !session.is_client(),
            "--embedded forces the mode; the socket is not even dialled"
        );
    }
}
