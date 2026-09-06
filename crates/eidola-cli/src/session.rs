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
    /// Choose who answers, following the selection rule.
    pub fn open(
        config_dir: PathBuf,
        data_dir: PathBuf,
        embedded: bool,
    ) -> Result<Session, Startup> {
        let config_path = config_dir.join("config.toml");
        if embedded {
            return AppCore::new(config_dir, data_dir)
                .map(|core| Session::from_core(core, config_path))
                .map_err(Startup::Core);
        }
        let runtime = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(e) => {
                return Err(Startup::Core(AppError::Internal {
                    message: format!("could not start a runtime: {e}"),
                }));
            }
        };
        let client_mode = |runtime, client, config_path| Session {
            mode: Mode::Client {
                runtime,
                client: tokio::sync::Mutex::new(client),
            },
            config_path,
        };
        match Self::dial(&runtime, &config_dir, &data_dir) {
            Ok(client) => Ok(client_mode(runtime, client, config_path)),
            Err(Dial::NoListener) => {
                // Nothing to ask, so open the profile here — which is also how
                // we learn whether anything is holding it.
                match AppCore::new(config_dir.clone(), data_dir.clone()) {
                    Ok(core) => {
                        drop(runtime);
                        Ok(Session::from_core(core, config_path))
                    }
                    // **Someone took the lock between that dial and this
                    // open**, and starting the app is exactly how that
                    // happens: the GUI takes the profile and binds its socket
                    // a moment apart, and a command launched alongside it can
                    // arrive in between. Look once more before concluding that
                    // the holder does not serve the profile — the answer has
                    // changed since the question was last asked. Once, not in
                    // a loop: what this settles is a race that has already
                    // resolved, and waiting for a holder that might start
                    // serving is the silent wait the whole rule forbids.
                    Err(e) if matches!(e, AppError::DatabaseInUse { .. }) => {
                        match Self::dial(&runtime, &config_dir, &data_dir) {
                            Ok(client) => Ok(client_mode(runtime, client, config_path)),
                            // Still nothing serving it. The holder is an
                            // Eidola older than the socket, or one whose
                            // listener stopped — which is what `unanswered`
                            // says, naming the process holding the lock.
                            Err(Dial::NoListener | Dial::NotAccepting) => Err(unanswered(e)),
                            Err(Dial::OtherProfile { ours, theirs }) => {
                                Err(Startup::OtherProfile { ours, theirs })
                            }
                            Err(Dial::Failed(e)) => Err(Startup::Dial(e)),
                        }
                    }
                    Err(other) => Err(unanswered(other)),
                }
            }
            Err(Dial::NotAccepting) => Err(Startup::NotAccepting { pid: None }),
            // Refused during the handshake, so no verb was ever dispatched at
            // it: the app answering is not serving this command's profile.
            Err(Dial::OtherProfile { ours, theirs }) => Err(Startup::OtherProfile { ours, theirs }),
            Err(Dial::Failed(e)) => Err(Startup::Dial(e)),
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
        if tests::first_dial_finds_nothing() {
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
        /// Makes the **next** dial report that nothing is listening.
        ///
        /// The window this exists to test is a few syscalls wide — between
        /// the first dial and the profile open that loses the lock — so a
        /// test cannot schedule a listener into it from outside. Forcing the
        /// first look to miss puts the process in exactly the state the race
        /// leaves it in, with a real app already serving, and lets the redial
        /// be the thing under test rather than the timer.
        static FIRST_DIAL_MISSES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// Takes the flag, so only the first dial of a run is affected.
    pub(super) fn first_dial_finds_nothing() -> bool {
        FIRST_DIAL_MISSES.with(|f| f.replace(false))
    }

    fn make_first_dial_miss() {
        FIRST_DIAL_MISSES.with(|f| f.set(true));
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

        make_first_dial_miss();
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
    fn a_holder_that_serves_nothing_is_still_named_as_that() {
        // The other side of the same door, and the reason the redial is one
        // look rather than a wait: a lock holder that serves no socket — an
        // Eidola older than it, or one whose listener stopped — must still be
        // reported, not waited on.
        let dir = tempfile::tempdir().expect("tempdir");
        let holder = AppCore::new(dir.path().into(), dir.path().into()).expect("the holder");

        make_first_dial_miss();
        match Session::open(dir.path().into(), dir.path().into(), false) {
            Err(Startup::NotAccepting { .. }) => {}
            other => panic!(
                "a second look that finds nothing is still the honest refusal: {:?}",
                other.map(|_| ())
            ),
        }
        drop(holder);
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
