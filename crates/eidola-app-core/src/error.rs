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

    /// The wallet-level ACT provisioning queue timed out waiting for an
    /// in-flight credential refund to free spendable balance for a concurrent
    /// turn. Distinct from [`AppError::InsufficientBalance`] (which is a true
    /// shortfall the user must top up): here a concurrent turn holds the only
    /// coverage mid-spend and its refund never landed within the bounded wait.
    /// Routes through the same recoverable-failure UI as a network blip — the
    /// user can simply retry once the other turn settles.
    #[error("provisioning timed out: {message}")]
    ProvisioningTimeout { message: String },

    /// The server requires acceptance of the current terms-of-service /
    /// privacy-policy versions before purchases or credential issuance can
    /// proceed (HTTP 428). UIs route to a review-and-accept step:
    /// `AppCore::current_terms` lists the documents,
    /// `AppCore::accept_current_terms` records acceptance.
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

    /// An internal runtime or system error.
    #[error("internal error: {message}")]
    Internal { message: String },

    /// A local-model operation failed: a download, the llama.cpp engine
    /// lifecycle (spawn/load/unload), or routing a chat turn to a local
    /// model that isn't loaded.
    #[error("local model error: {message}")]
    LocalModel { message: String },

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
}

// ---------------------------------------------------------------------------
// Internal conversion helpers
// ---------------------------------------------------------------------------

impl AppError {
    /// Classify a `reqwest::Error`, surfacing attestation failures explicitly.
    pub(crate) fn from_request(e: reqwest::Error) -> Self {
        let chain = format_error_chain(&e);
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

fn format_error_chain(e: &reqwest::Error) -> String {
    let mut chain = format!("{e}");
    let mut source = std::error::Error::source(e);
    while let Some(err) = source {
        use std::fmt::Write;
        let _ = write!(chain, ": {err}");
        source = err.source();
    }
    chain
}
