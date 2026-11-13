/// Errors returned by app-core operations.
///
/// Each variant maps to a distinct failure mode so callers (CLI, GUI) can
/// display appropriate feedback without parsing error strings.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AppError {
    /// A required configuration value is missing (base_url, account, etc.).
    #[error("not configured: {message}")]
    NotConfigured { message: String },

    /// A turn was asked for in a space that has been **archived**.
    ///
    /// Archival is what closes a conversation: the Library stops offering it,
    /// and retiring an agent archives every room it owned. That has to mean
    /// *no new work*, or a cascade planned a moment earlier goes on spending
    /// in a room somebody closed — which matters most exactly where nobody is
    /// watching, in a human-less sub-space of agents answering each other.
    ///
    /// **It stops future turns; it does not abort one already in flight.** A
    /// turn that got past this gate finishes and persists — the request was
    /// made, the tokens were spent, and dropping the answer would bill for
    /// nothing — but its completion re-plans, and planning yields no turns in
    /// an archived space. Nothing else changes: every membership stands, every
    /// read still works, the transcript is whole.
    #[error("this conversation is archived, so it takes no new turns")]
    SpaceArchived { space_id: String },

    /// The human tried to write into an agent-spawned sub-space they have not
    /// joined.
    ///
    /// Reading one is unconditional (oversight — see `db::may_read_space`);
    /// *acting* in one is membership, because the roster the models are shown
    /// has to stay truthful about who is in the room, and because a turn driven
    /// there spends the reader's credits. The two are not the same permission
    /// and this is where they part. Raised by every door that acts as the human
    /// — `post` (and `chat`/`chat_stream` through it), `edit_post`,
    /// `regenerate`, `respond_stream` — before any write and before any spend.
    ///
    /// **Data, not prose.** It carries the `space_id` the caller already named,
    /// so the surface that raises it can offer the join that would satisfy it,
    /// and nothing else: the sentence a reader sees is chosen in their own
    /// locale by the presentation layer (`space-error-not-joined`), and a
    /// message carried from here would be a second source for it to drift
    /// from. This `Display` is the log-shaped fallback every other variant has.
    #[error("this conversation was opened between agents; joining it is what allows posting")]
    NotJoined { space_id: String },

    /// Combined post-and-turn (`chat` / `chat_stream`) was asked in a live
    /// sub-space. Those rooms take their turns from the driver; this door
    /// would post and then start a second cascade alongside it, at twice the
    /// spend. Posting (`post` / `submit`) and an explicit ask
    /// (`respond_stream_as`) remain. Asked after the membership gate, so an
    /// unjoined reader still sees [`Self::NotJoined`].
    ///
    /// **Data, not prose.** Same shape as [`Self::NotJoined`]: the space id
    /// the caller named, and a log-shaped `Display`. The sentence a reader
    /// sees is the presentation layer's.
    #[error("this conversation takes its own turns, so a combined ask would double them")]
    DrivenConversation { space_id: String },

    /// A post-level write was aimed at a kind of post it does not apply to:
    /// editing something that is not the human's own input, or regenerating
    /// something that is not an agent's inferred answer.
    ///
    /// Both writes claim a post's *item* — an edit appends a `user_input`
    /// generation, a regeneration appends an `inference` one — so aimed at the
    /// wrong kind they do not amend a post, they replace it with a different
    /// kind of thing. Typed so a surface can withhold the affordance and say
    /// why, rather than discovering it by writing.
    #[error("{message}")]
    WrongPostKind { message: String },

    /// An agent's request to spawn a sub-space was refused — a guard, an
    /// attenuation check, or an ineligible participant (see
    /// [`crate::subspaces::SpawnRefusal`], which carries which and says so in
    /// words a model can act on). Typed rather than a message so the tool that
    /// exposes the door can render a refusal as a tool result the model
    /// corrects, instead of failing a turn over it. Nothing was written.
    #[error("{refusal}")]
    SpawnRefused {
        refusal: crate::subspaces::SpawnRefusal,
    },

    /// A proxy setting or key operation was refused before anything was
    /// written — typed ([`crate::proxy::ProxyRefusal`]) so a surface can choose
    /// its own words for which one. Prints as the configuration error it
    /// always was.
    #[error("config error: {refusal}")]
    ProxyRefused { refusal: crate::proxy::ProxyRefusal },

    /// An HTTP request failed at the transport layer.
    #[error("network error: {message}")]
    Network { message: String },

    /// Enclave attestation verification failed.
    #[error("attestation failed: {message}")]
    Attestation { message: String },

    /// The server returned a non-success HTTP status.
    #[error("server error ({status}): {message}")]
    Server { status: u16, message: String },

    /// An anonymous credential operation failed.
    #[error("credential error: {message}")]
    Credential { message: String },

    /// The server's current issuer key failed a check this client makes for
    /// itself, against the public issuer calendar or against the key it
    /// already saw for the epoch (see [`IssuerKeyRefusal`]). Refused before
    /// anything is requested, so nothing was charged; credentials already
    /// held are untouched. Typed so a surface can say what it means: unlike
    /// a network failure, retrying now will not help.
    #[error("credential error: {refusal}")]
    IssuerKeyRefused { refusal: IssuerKeyRefusal },

    /// A chat was attempted with no usable credential and no account
    /// configured — onboarding has not begun (or the account was reset).
    /// Distinct from [`AppError::NotConfigured`] so UIs can route to the
    /// account-creation step instead of a generic config error.
    #[error("no account configured — create an account to begin")]
    NoAccount,

    /// The account exists but its available balance cannot cover the
    /// credits required for the attempted operation. Carries both sides
    /// of the comparison so UIs can show honest numbers and route to the
    /// purchase step.
    #[error("insufficient balance: {required} credits required, {available} available")]
    InsufficientBalance { available: i64, required: i64 },

    /// A request's worst-case charge is more than this client will pay
    /// without issuing a credential sized to it. A just-in-time issuance is
    /// always the denomination (`denomination`, `DEFAULT_ALLOCATION_CREDITS`)
    /// or the whole remaining balance, never the request's own size, because
    /// the amount issued is visible on the account surface and a request's
    /// share is visible at its spend; so a request is paid only when what the
    /// wallet holds, or one held credential drained beside one fresh
    /// denomination, covers it (`docs/anonymous-credits.md` → Issuance
    /// sizing). Refused before any credential is touched or anything is
    /// issued, so nothing was charged. Distinct from
    /// [`AppError::InsufficientBalance`]: adding credit does not lift it; a
    /// smaller request does.
    ///
    /// `hold` is the request's worst case in credits, saturated at
    /// `i64::MAX` for one too large to represent.
    #[error(
        "this request's worst case ({hold} credits) is more than one held credential and \
         one fresh credential of {denomination} credits can pay together"
    )]
    HoldExceedsDenomination { hold: i64, denomination: i64 },

    /// The wallet-level ACT provisioning queue timed out waiting for an
    /// in-flight credential refund to free spendable balance for a concurrent
    /// turn. Distinct from [`AppError::InsufficientBalance`] (which is a true
    /// shortfall the user must top up): here a concurrent turn holds the only
    /// coverage mid-spend and its refund never landed within the bounded wait.
    /// Routes through the same recoverable-failure UI as a network blip — the
    /// user can simply retry once the other turn settles.
    #[error("provisioning timed out: {message}")]
    ProvisioningTimeout { message: String },

    /// An Eidola engine refused a request's declared cached prefix (it no
    /// longer held what the conversation's earlier requests left in it), and
    /// the credits already held could not pay for sending it again without
    /// the declaration. That resend is never paid by converting account
    /// credit: whoever can empty an engine's cache could otherwise choose
    /// when this client converts. Nothing was converted; the refused request
    /// was charged its request fee. Retrying is an ordinary request, which
    /// converts as any other does.
    #[error(
        "the engine no longer held this conversation's cached prompt, and the credits held cannot pay to send it again in full; try again"
    )]
    UncachedRetryUnfunded,

    /// The server requires acceptance of the current terms-of-service /
    /// privacy-policy versions before purchases or credential issuance can
    /// proceed (HTTP 428). UIs route to a review-and-accept step:
    /// `AppCore::current_terms` lists the documents and
    /// `AppCore::accept_terms` records acceptance of that same snapshot —
    /// the one the user was shown, never a re-read.
    #[error("terms acceptance required: {message}")]
    TermsAcceptanceRequired { message: String },

    /// A participant tried to reach across a space boundary it is not a member
    /// of: creating a reference to a post in a space it does not take part in
    /// (rule 1 — "you may quote what you can read"), or following one into a
    /// space it does not take part in (rule 4 — "follow requires membership").
    /// Membership is the cross-space ACL (task 36); the fix is an ordinary
    /// grant, so the caller can retry with no special machinery.
    ///
    /// **The payload must never leak the referenced space.** It confirms only
    /// what the actor already knows — the action id it named, and its own
    /// identity — and carries no space id, title, participant or content. That
    /// is not decoration: existence is public *within the referencing space*
    /// (rule 3), everything else about the referenced space is not, and this
    /// error is rendered into tool results a model reads and into UI a
    /// non-member sees. Anything added here is added to both.
    #[error("not a participant of the conversation that post {action_id} belongs to")]
    NotAParticipant {
        /// The actor that was refused (a participant id).
        participant_id: String,
        /// The action the actor named — already known to it, so not a leak.
        action_id: String,
    },

    /// A local database operation failed.
    #[error("database error: {message}")]
    Database { message: String },

    /// The local database is already open by another Eidola process.
    ///
    /// Turso is single-writer and gives no honest signal when a second
    /// process opens the same file — writes just start failing or
    /// misbehaving. So `AppCore` takes an exclusive advisory lock on a
    /// sidecar lockfile at open (see `db::DbLock`) and construction fails
    /// with this variant when another process holds it. `pid` is the holder
    /// recorded in the lockfile (`None` when it could not be read — the
    /// holder may have been mid-write, or the file may be unreadable);
    /// `message` already names it, so UIs can render the variant directly
    /// and use `pid` only when they want to say more.
    ///
    /// Distinct from [`AppError::Database`] so the CLI can print an
    /// actionable hint ("quit the other Eidola") and the GUI can route to a
    /// startup dialog instead of a generic database error.
    #[error("{message}")]
    DatabaseInUse { pid: Option<u32>, message: String },

    /// Configuration read/write error.
    #[error("config error: {message}")]
    Config { message: String },

    /// The server created an account, but this profile could not keep it: the
    /// strict reload refused the config (a hand edit or a sync write made it
    /// unparseable while the request was in flight), the save failed, or an
    /// account was linked in the meantime, which is never overwritten.
    ///
    /// **This is the only copy of the secret.** The server keeps only its
    /// hash, so an ordinary config error here would lose the account for
    /// good. The variant carries the credential back so the surface that
    /// asked can show it to be saved and linked once the profile is fixed.
    /// `Display` names only the account id and the cause; the secret is in
    /// [`AccountSecret`], which prints redacted, so a log line cannot copy it.
    ///
    /// The account's details are boxed ([`UnsavedAccount`]) so this rare
    /// variant does not grow every `Result<_, AppError>` in the crate.
    #[error("the account {} was created but this profile could not save it: {reason}", .account.account_id)]
    AccountNotSaved {
        account: Box<UnsavedAccount>,
        reason: String,
    },

    /// A model notice's action (switching to newer weights, dismissing a
    /// "model updated" notice) was asked of a participant that no longer has
    /// that notice: the catalog, its model or its answers moved since the
    /// notice was shown. Nothing was written; re-reading the roster shows
    /// what applies now.
    #[error("the model notice for participant {participant_id} no longer applies")]
    ModelNoticeStale { participant_id: String },

    /// An internal runtime or system error.
    #[error("internal error: {message}")]
    Internal { message: String },

    /// A local-model operation failed: a download, the llama.cpp engine
    /// lifecycle (spawn/load/unload), or routing a chat turn to a local
    /// model that isn't loaded.
    #[error("local model error: {message}")]
    LocalModel { message: String },

    /// A turn or chore named a model of the managed local store whose file is
    /// not on this machine. Distinct from [`AppError::LocalModel`] so a
    /// surface can name the model and say how to get it, rather than show a
    /// sentence with a filesystem path in it. Nothing was sent.
    ///
    /// **Data, not prose.** `model` is the `<slug>@local` selection; a curated
    /// one maps back to its catalog entry through
    /// [`crate::local_models::catalog_entry_for_model_ref`].
    #[error("the local model `{model}` is not downloaded")]
    LocalModelNotDownloaded { model: String },

    /// A self-update verification step failed.
    ///
    /// Used by [`crate::updater`] to surface fetch/parse/schema/continuity
    /// problems before any cryptographic verification stage runs; the
    /// crypto stages produce [`AppError::Attestation`] instead.
    #[error("update error: {message}")]
    Update { message: String },

    /// A turn's bounded tool-calling loop could not finish honestly.
    ///
    /// Two causes, both deliberately *not* silent truncations:
    ///
    /// * the round cap (`MAX_TURN_ROUNDS`) was reached while the model was
    ///   still asking for tools — the turn produced no final answer, and
    ///   pretending the last tool request was one would be a lie;
    /// * the model's `tool_calls` were structurally unusable (no call id, no
    ///   function name), so there is nothing to execute or to persist as a
    ///   `tool_use` block.
    ///
    /// Everything that *did* happen up to that point (each round's `tool_call`
    /// / `tool_result` actions and its raw request row) is committed before
    /// this is returned.
    #[error("tool loop error: {message}")]
    ToolLoop { message: String },

    /// A selection names a model its backend's catalog no longer offers.
    ///
    /// The catalog is what the backend sells, and it is allowed to shrink: a
    /// row retired upstream simply stops being listed. **Nothing un-picks the
    /// thing that chose it** — a participant's `model_ref`, a space's router
    /// model, the model a post recorded and a regeneration replays — so the
    /// selection goes stale in place and is met here, at the next turn.
    ///
    /// **The refusal is the whole behaviour, and the alternative is worse than
    /// the failure.** Answering with some other model would put words a reader
    /// attributes to one model in the mouth of another, and bill for it; there
    /// is no substitution anywhere on this path, and none may be added. The
    /// refusal lands in `prepare_turn` before the credential and before any
    /// request, so nothing is spent and nothing is written — the post a
    /// combined ask already committed stands, and the answer a regeneration
    /// would have replaced is untouched.
    ///
    /// Distinct from [`AppError::NotConfigured`] so a surface can say which
    /// model went away and point at the picker, rather than render a
    /// log-shaped sentence with the id buried in it.
    ///
    /// **Data, not prose.** `model` is the selection as it was persisted (the
    /// canonical qualified id); the sentence a reader sees is chosen in their
    /// own locale by the presentation layer.
    #[error("the backend no longer offers the model `{model}`")]
    ModelUnavailable { model: String },

    /// A second regeneration of the same post was asked for while one is still
    /// running in this process.
    ///
    /// Regeneration spends before it writes: the credential is provisioned and
    /// the request is in flight long before any row lands, so the durable
    /// single-successor index (`idx_one_successor_per_action`) rejects the
    /// loser only after both have paid. The in-process guard refuses the second
    /// ask before either — which is what makes closing and reopening a window
    /// mid-regeneration cost nothing, since the guard outlives any one view.
    ///
    /// **Data, not prose**: the item whose generation is being replaced.
    #[error("this response is already being regenerated")]
    RegenerationInFlight { item_id: String },
}

/// Why the client refused the server's current issuer key.
///
/// Every token issued under one key hides among every other token under it,
/// so the key decides whom a client's spends look like. An honest gateway has
/// exactly one key per epoch of the public issuer calendar
/// (`eidola_common::issuer_calendar`), with that epoch's windows, and never
/// rotates within an epoch. These are the two ways a key can fail that, and
/// the client checks both for itself: they do not depend on trusting the
/// gateway, its operator or its database.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IssuerKeyRefusal {
    /// The key's windows are not exactly one epoch's on the calendar.
    NotOnCalendar,
    /// The key is not the one this client already saw for its epoch. A
    /// second key for an epoch is what a server would show a client it was
    /// singling out, and the client cannot tell an emergency rotation from
    /// that, so there is none: issuance stops until the next epoch.
    SecondKeyForEpoch { epoch: u64 },
    /// The same key offered for two epochs — in one listing, or for an epoch
    /// other than the one this client already recorded it for. An honest
    /// gateway mints a distinct key for every epoch, so a key reused across
    /// epochs is one the per-epoch record could not otherwise hold to
    /// anything: listed as next epoch's too, it would leave next epoch
    /// unrecorded for a different key to fill.
    OneKeyForSeveralEpochs { epoch: u64 },
}

impl std::fmt::Display for IssuerKeyRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssuerKeyRefusal::NotOnCalendar => f.write_str(
                "the server's current issuer key does not sit on the public issuer calendar, \
                 so no credential was requested under it (nothing was charged)",
            ),
            IssuerKeyRefusal::SecondKeyForEpoch { epoch } => write!(
                f,
                "the server offered a different issuer key for epoch {epoch} from the one this \
                 client already saw; an honest server has one key per epoch, so no credential \
                 was requested (nothing was charged, and credentials already held keep \
                 working). Issuance resumes with the next epoch's key"
            ),
            IssuerKeyRefusal::OneKeyForSeveralEpochs { epoch } => write!(
                f,
                "the server offered, for epoch {epoch}, an issuer key it also offers for another \
                 epoch; an honest server has a distinct key for every epoch, so no credential \
                 was requested (nothing was charged, and credentials already held keep working)"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

impl AppError {
    /// Classify a `reqwest::Error`, surfacing attestation failures explicitly.
    pub(crate) fn from_request(e: reqwest::Error) -> Self {
        let chain = request_error_text(e);
        if chain.contains("measurement") && chain.contains("allowed") {
            return AppError::Attestation {
                message: "the server's enclave measurement is not in your \
                          trusted_measurements list. The running server version \
                          is not trusted by this client."
                    .into(),
            };
        }
        if chain.contains("fingerprint") && chain.contains("mismatch") {
            return AppError::Attestation {
                message: "TLS certificate does not match the attested enclave".into(),
            };
        }
        if chain.contains("attestation") {
            return AppError::Attestation {
                message: format!("could not verify enclave attestation: {chain}"),
            };
        }
        AppError::Network {
            message: format!("request failed: {chain}"),
        }
    }

    pub(crate) fn db(e: impl std::fmt::Display) -> Self {
        AppError::Database {
            message: e.to_string(),
        }
    }
}

/// One `reqwest::Error` rendered as text a person can be shown — **with the
/// request URL removed**.
///
/// reqwest attaches the request URL to every request-phase error and prints it
/// in `Display` (`error sending request for url (https://host/x?token=…)`),
/// and `Response::bytes`/`text`/`json` attach it to a body they collected. A
/// model link's query is its *authorization* — a signed S3/CDN URL carries
/// `?token=…` — so formatting one of these errors copies a live credential
/// into whatever then holds the message: a standing failure a settings pane
/// renders, a support paste, a terminal's scrollback. Nothing there needs the
/// URL; the reader knows what they asked for, and the cause is what they are
/// missing.
///
/// The source chain is appended, which is where the useful part lives (DNS,
/// TLS, connect); it never repeats the URL — established by running each of
/// those failures rather than by reading the docs. **Every path in this crate
/// that turns a `reqwest::Error` into a message goes through here**, so no
/// individual call site has to remember.
pub(crate) fn request_error_text(e: reqwest::Error) -> String {
    let e = e.without_url();
    let mut chain = format!("{e}");
    let mut source = std::error::Error::source(&e);
    while let Some(err) = source {
        use std::fmt::Write;
        let _ = write!(chain, ": {err}");
        source = err.source();
    }
    chain
}

/// An account the server created that the profile could not keep — what
/// [`AppError::AccountNotSaved`] carries back.
#[derive(Debug, Clone)]
pub struct UnsavedAccount {
    pub account_id: String,
    pub secret: AccountSecret,
    /// The accepted terms were still submitted as the new account; this is
    /// the standing that left — the same report a saved create returns — so a
    /// surface can say what linking it will still need.
    pub terms: crate::TermsAcceptance,
    /// Where the credential was also written, owner-only, beside the profile
    /// (`account-recovery-<id>.toml`, in the profile's own shape), so it
    /// outlives a caller that drops the error. `None` only when that write
    /// failed too.
    pub recovery_file: Option<std::path::PathBuf>,
}

/// A freshly created account's secret, carried back in
/// [`AppError::AccountNotSaved`] because nothing else holds it. Prints
/// redacted under both `Debug` and `Display`; [`AccountSecret::expose`] is the
/// one way to read it, for the surface that shows it to be saved.
#[derive(Clone, PartialEq, Eq)]
pub struct AccountSecret(String);

impl AccountSecret {
    pub fn new(secret: String) -> Self {
        Self(secret)
    }

    /// The secret itself, for showing to the person who must save it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for AccountSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AccountSecret(<redacted>)")
    }
}

impl std::fmt::Display for AccountSecret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}
