pub mod backends;
pub mod changes;
pub mod config;
pub mod db;
pub mod error;
pub mod local_models;
pub mod trust_root;
pub mod updater;
pub mod updates;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anonymous_credit_tokens::{
    CreditToken, IssuanceResponse, Params, PreIssuance, PreRefund, PublicKey, Refund, SpendProof,
    credit_to_scalar, scalar_to_credit,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand_core::OsRng;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub use backends::{
    BackendInfo, BackendKind, BackendUpdate, EIDOLA_BACKEND_ID, LOCAL_BACKEND_ID, ModelRef,
    NewBackend, parse_model_ref, qualified_model_id,
};
use changes::{BroadcastSource, Change, ChangeSource};
use config::Config;
pub use config::DEFAULT_TEMPLATE_ID;
pub use db::HUMAN_PARTICIPANT_ID;
use error::AppError;
pub use local_models::{
    ExternalEngineBackend, LOCAL_MODEL_CATALOG, LocalCatalogEntry, LocalModelInfo,
    LocalModelStatus, LocalModelsState,
};

// ============================================================================
// Data transfer types — returned from `AppCore` methods to the apps
// (CLI, GUI).
// ============================================================================

/// Snapshot of the current config for display.
///
/// Purely config-backed (resolvable without the database). The eidola
/// connection + trust bundle (base URL, trusted measurements, hardware CAs)
/// moved to the `eidola` backend row — see [`EidolaTrust`] and
/// [`AppCore::eidola_trust`].
#[derive(Clone, Debug)]
pub struct ConfigState {
    /// The UUID of the default space template new spaces are instantiated from
    /// (the `default_template` config key; the seeded "Default" template's id
    /// by default). The transitional resolved default *model* is **not** on
    /// this (config-backed, DB-free) snapshot — it reads the DB, so it is the
    /// async [`AppCore::default_model`] instead (a nested `block_on` in
    /// `config_state` panicked when called from inside the core runtime).
    pub default_template: String,
    pub has_account: bool,
    pub has_account_secret: bool,
    pub domain_separator: String,
    pub attestation_url: Option<String>,
    /// The resolved circadian day/night axis (`appearance` override if set,
    /// otherwise `system`).
    pub appearance: config::AppearanceSetting,
    /// The resolved circadian time-of-day axis (`time_of_day_tint` override
    /// if set, otherwise `on`).
    pub time_of_day_tint: config::TimeOfDayTint,
    /// The resolved fixed light character used while `time_of_day_tint` is
    /// `off` (`light_character` override if set, otherwise `neutral`).
    pub light_character: config::LightCharacter,
    /// The resolved base type-scale factor (`font_scale` override clamped into
    /// range if set, otherwise [`config::FONT_SCALE_DEFAULT`]). The GUI applies
    /// it over the whole type ramp.
    pub font_scale: f32,
}

/// The eidola backend's resolved connection + trust bundle, honest about
/// override-vs-pin. Read from the `eidola` backend row (see
/// [`AppCore::eidola_trust`]); each field falls back to the embedded
/// trust-root pin when the row's column is NULL. Overriding the trust root
/// is security-relevant state, so UIs render override state + revert-to-pin
/// explicitly.
#[derive(Clone, Debug)]
pub struct EidolaTrust {
    /// The resolved server URL (row override, else the pin).
    pub base_url: String,
    /// The trust-root pin baked into this binary — equal to `base_url`
    /// unless overridden, so the UI can be honest about the source.
    pub base_url_pin: String,
    /// Whether `base_url` is a row override (`true`) or the pin (`false`).
    pub base_url_is_override: bool,
    /// The resolved measurements the client will accept on handshake.
    pub trusted_measurements: Vec<MeasurementInfo>,
    /// Whether `trusted_measurements` is a row override list (`true`) or the
    /// single pinned build measurement (`false`).
    pub trusted_measurements_are_override: bool,
    /// The build-time pinned enclave measurement — present regardless of
    /// override state, so a UI can display/copy it for audit and (since an
    /// override *replaces* the pin in the trusted set) re-add it alongside
    /// custom measurements.
    pub pinned_measurement: MeasurementInfo,
    pub has_hardware_root_ca: bool,
    /// The custom root-CA PEM override, if set (`None` = the built-in AMD/Intel
    /// vendor chain). Exposed so a UI can display/copy the certificate that is
    /// actually trusted, not just a "custom certificate set" flag.
    pub hardware_root_ca_pem: Option<String>,
    pub has_hardware_intermediate_ca: bool,
    /// The custom intermediate-CA PEM override, if set (`None` = vendor chain).
    pub hardware_intermediate_ca_pem: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MeasurementInfo {
    pub snp: String,
    pub tdx_rtmr1: String,
    pub tdx_rtmr2: String,
}

// --- Participants & space templates (Participants v1) ----------------------

/// One effective participant of a space (the override-resolved config + how it
/// is a member). `scope` = the participant's own scope: `global` members are
/// **references** (a shared-library identity — editing edits everywhere, or the
/// GUI can override just this space), `space` members are **owned** by this
/// space (editing edits this space only). `source` echoes that as
/// `"referenced"`/`"owned"`.
#[derive(Clone, Debug)]
pub struct ParticipantInfo {
    pub id: String,
    pub scope: String,
    pub source: String,
    pub kind: String,
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
    /// `owner`/`member`/`observer`.
    pub role: String,
    /// For a **referenced global** (`source == "referenced"`), the global's own
    /// config plus the raw per-membership overrides, so a GUI can render and
    /// edit the "edit everywhere" vs "override here" fork honestly. `None` for
    /// owned (`space`-scoped) participants, which have no override layer.
    pub reference: Option<ParticipantReference>,
}

/// The "edit everywhere vs override here" detail for a referenced global
/// participant (see [`ParticipantInfo::reference`]).
///
/// The `base_*` fields are the shared global's own config — what "edit
/// everywhere" ([`AppCore::update_space_participant`]) writes. Each `override_*`
/// field is the raw membership override: `None` = inherit the base; `Some(s)`
/// (including `Some("")`) = overridden (the doctrine's NULL = inherit,
/// `''` = cleared). What "override here"
/// ([`AppCore::set_space_participant_override`]) writes.
#[derive(Clone, Debug)]
pub struct ParticipantReference {
    pub base_label: String,
    pub base_model_ref: Option<String>,
    pub base_system_prompt: Option<String>,
    pub base_notify_policy: String,
    pub override_label: Option<String>,
    pub override_model_ref: Option<String>,
    pub override_system_prompt: Option<String>,
    pub override_notify_policy: Option<String>,
}

impl ParticipantInfo {
    fn from_effective(r: db::EffectiveParticipantRow) -> Self {
        Self {
            id: r.participant_id,
            scope: r.scope,
            source: r.source,
            kind: r.kind,
            label: r.label,
            model_ref: r.model_ref,
            system_prompt: r.system_prompt,
            notify_policy: r.notify_policy,
            role: r.role,
            reference: None,
        }
    }

    fn from_owned(r: db::ParticipantRow) -> Self {
        Self {
            id: r.id,
            scope: r.scope,
            source: "owned".to_string(),
            kind: r.kind,
            label: r.label,
            model_ref: r.model_ref,
            system_prompt: r.system_prompt,
            notify_policy: r.notify_policy,
            role: r.role,
            reference: None,
        }
    }
}

/// A new agent participant to add to a space (agents only — the human is the
/// seeded shared "You"). `notify_policy` empty ⇒ `explicit`.
#[derive(Clone, Debug, Default)]
pub struct NewParticipant {
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
}

/// A partial update to a space participant's mirrored config columns (each
/// `Some` replaces; the inner `Option` clears/sets the nullable columns).
#[derive(Clone, Debug, Default)]
pub struct ParticipantUpdate {
    pub label: Option<String>,
    pub model_ref: Option<Option<String>>,
    pub system_prompt: Option<Option<String>>,
    pub notify_policy: Option<String>,
}

/// A per-membership override edit for a **referenced global** participant
/// ("override here", this space only — vs [`ParticipantUpdate`]'s "edit
/// everywhere"). Each `Some` outer writes that override column; the inner
/// `Option` sets it (`Some(value)`, including `Some("")` to clear a field to
/// empty) or reverts it to inherited (`None` = SQL NULL). Untouched columns
/// (`None` outer) are left as-is.
#[derive(Clone, Debug, Default)]
pub struct ParticipantOverride {
    pub label: Option<Option<String>>,
    pub model_ref: Option<Option<String>>,
    pub system_prompt: Option<Option<String>>,
    pub notify_policy: Option<Option<String>>,
}

/// A space template (its settings + agent participants).
#[derive(Clone, Debug)]
pub struct SpaceTemplateInfo {
    pub id: String,
    pub title: String,
    pub cascade_limit: i64,
    pub participants: Vec<TemplateParticipantInfo>,
}

/// One agent participant a template OWNS (`scope='template'`).
#[derive(Clone, Debug)]
pub struct TemplateParticipantInfo {
    pub id: String,
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
}

impl TemplateParticipantInfo {
    fn from_owned(r: db::ParticipantRow) -> Self {
        Self {
            id: r.id,
            label: r.label,
            model_ref: r.model_ref,
            system_prompt: r.system_prompt,
            notify_policy: r.notify_policy,
        }
    }
}

/// A new agent participant for a template.
#[derive(Clone, Debug, Default)]
pub struct NewTemplateParticipant {
    pub label: String,
    pub model_ref: Option<String>,
    pub system_prompt: Option<String>,
    pub notify_policy: String,
}

/// A validated, normalized template participant input:
/// `(label, model_ref, system_prompt, notify_policy)`.
type ValidatedTemplateParticipant = (String, Option<String>, Option<String>, String);

/// Valid notify-policy values (mirrors the schema CHECK).
fn validate_notify_policy(policy: &str) -> Result<String, AppError> {
    match policy {
        "explicit" | "human" | "all" => Ok(policy.to_string()),
        other => Err(AppError::Config {
            message: format!("invalid notify_policy `{other}` (expected explicit, human, or all)"),
        }),
    }
}

/// Normalize a notify-policy input, defaulting empty to `explicit`.
fn notify_policy_or_default(policy: &str) -> Result<String, AppError> {
    if policy.trim().is_empty() {
        Ok("explicit".to_string())
    } else {
        validate_notify_policy(policy.trim())
    }
}

#[derive(Clone, Debug)]
pub struct AccountCreateResult {
    pub id: String,
    /// The account secret, returned once at creation so the caller can present
    /// it for the user to save. It is also persisted to the local config; there
    /// is no other API to recover it (losing it means creating a new account).
    pub secret: String,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct AccountShowResult {
    pub id: String,
    pub stripe_customer_id: Option<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct PriceInfo {
    pub id: String,
    pub product_name: String,
    pub product_description: Option<String>,
    pub amount_display: String,
    pub recurrence: String,
    pub credits: i64,
}

#[derive(Clone, Debug)]
pub struct BalancesResult {
    pub available: i64,
    pub pools: Vec<BalancePoolInfo>,
}

#[derive(Clone, Debug)]
pub struct BalancePoolInfo {
    pub amount: i64,
    pub source: String,
    pub expires_at: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct CredentialInfo {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
}

#[derive(Clone, Debug)]
pub struct InFlightCredentialInfo {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub spend_amount: i64,
}

#[derive(Clone, Debug)]
pub struct AllocateResult {
    pub nonce: String,
    pub credits: i64,
    pub issuer_key_id: String,
}

/// A document (terms of service / privacy policy) whose current version the
/// server requires accounts to accept, identified by the SHA-256 of its
/// exact published text — the same accept-by-hash mechanism the
/// repository's CLA uses. `version` is the document's monotonically
/// increasing version number; accepting version N satisfies any server
/// requirement ≤ N.
#[derive(Clone, Debug)]
pub struct TermsDocument {
    /// `terms_of_service` or `privacy_policy`.
    pub document: String,
    /// Monotonically increasing document version.
    pub version: i64,
    /// Where the current text is published.
    pub url: String,
    /// Hex-encoded SHA-256 of the exact published document text.
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct ChatResult {
    pub space_id: String,
    pub content: String,
    pub model: String,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub credits_charged: i64,
    /// The persisted inference action id this turn produced. Lets a caller
    /// continue an auto-notify cascade by re-planning on the fresh post
    /// (`plan_notifications(space_id, response_action_id)`). `None` only on
    /// paths that produced no inference row (none today — every success path
    /// persists one).
    pub response_action_id: Option<String>,
}

/// Outcome of [`AppCore::post`] — saving a thought without requesting a
/// response. Carries the (possibly newly created) space id and the persisted
/// action/item ids so a caller can adopt them.
#[derive(Clone, Debug)]
pub struct PostResult {
    pub space_id: String,
    /// The persisted `user_input` action (gen 0 of a fresh item).
    pub action_id: String,
    /// The new item's stable identity (shared by future generations of this post).
    pub item_id: String,
    /// True when this post created the space.
    pub is_new_space: bool,
    /// True when this post auto-titled a previously untitled space.
    pub auto_titled: bool,
}

/// How a [`AppCore::run_turn`] response attaches to the thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseMode {
    /// A new child node replying to the target action (the normal chat case):
    /// fresh item, `reply` edge to the target.
    Reply,
    /// A new *generation* of the target's item (regenerate / agent revise):
    /// shares the target's `item_id`, supersedes it, replicates its reply edge.
    Revise,
}

/// Which participant should respond in a turn — the Participants-v1 input that
/// replaced the bare model string on the internal turn path.
///
/// * `Participant` — an explicit space participant id; its **effective** config
///   (COALESCE(override, participant config)) supplies the model and system
///   prompt. This is the `submit` → `plan_notifications` → drive path.
/// * `Model` — a bare/qualified model string (the model-picker compatibility
///   path the CLI/GUI still use until wave 3): resolved to the space's agent
///   participant whose effective model matches, **minting** a fresh space-owned
///   agent for the model when none matches (documented on
///   `Inner::resolve_or_mint_agent_by_model`).
#[derive(Clone, Debug)]
enum TurnSelector {
    Participant(String),
    Model(String),
}

/// One planned auto-response turn produced by [`AppCore::plan_notifications`]:
/// the participant that should respond and the post it responds to. Callers
/// drive one streaming turn per entry (via [`AppCore::respond_stream_as`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedTurn {
    /// The participant that should respond (an agent member of the space).
    pub participant_id: String,
    /// The post this turn replies to (the triggering post's action id).
    pub target_action_id: String,
    /// The cascade depth the resulting response will occupy (the triggering
    /// post's derived depth + 1) — informational for the GUI's cascade
    /// indicator. The guard itself re-derives depth from the data on the next
    /// [`AppCore::plan_notifications`], so this value is never persisted.
    pub cascade_depth: i64,
}

/// The outcome of planning notifications for a post.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotificationPlan {
    /// The turns to drive (possibly empty — nobody's notify policy fired).
    Turns(Vec<PlannedTurn>),
    /// The space's `cascade_limit` was reached at this post: instead of turns,
    /// a resumable paused marker the wave-3 GUI renders as "cascade limit
    /// reached — ask to continue". Explicit asks ([`AppCore::respond_stream`] /
    /// [`AppCore::respond_stream_as`]) bypass this guard entirely.
    Paused { depth: i64, limit: i64 },
}

/// The result of [`AppCore::submit`]: the saved post plus the notification plan
/// computed over the space's participants.
#[derive(Clone, Debug)]
pub struct SubmitResult {
    pub post: PostResult,
    pub plan: NotificationPlan,
}

#[derive(Clone, Debug)]
pub struct SpaceInfo {
    pub id: String,
    pub title: Option<String>,
    /// First ~120 chars of the first user message in the space — the UI's
    /// fallback line for untitled spaces. `None` for empty spaces.
    pub snippet: Option<String>,
    pub created_at: i64,
    /// Max `action.created_at` in the space; equals `created_at` for spaces
    /// with no actions yet.
    pub last_activity_at: i64,
    /// Count of terminal (complete/cancelled) actions in the space.
    pub message_count: i64,
    /// When the space was archived, if it has been. Always `None` unless
    /// listing was asked to include archived spaces.
    pub archived_at: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct SpaceMessage {
    pub role: String,
    pub content: String,
}

/// The participant rendered in a post's gutter byline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostParticipant {
    /// `human` | `agent` | `tool` | `system`.
    pub kind: String,
    /// Display label (e.g. "user", a model name, a tool name).
    pub label: String,
}

/// One typed content block of a post — the renderable payload of the action's
/// current generation, in `ordinal` order. v1 renders `text` / `thinking`; the
/// tool fields are carried faithfully for the later tool render.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostBlock {
    /// `text` | `thinking` | `code` | `tool_use` | `tool_result` | `image` | …
    pub block_type: String,
    pub text: Option<String>,
    pub tool_name: Option<String>,
    pub tool_call_id: Option<String>,
    /// JSON sidecar (tool args/results), if any.
    pub data: Option<String>,
}

/// A non-structural antecedent edge (relation `reference`) of a post: a plain
/// backlink, an inline quote (carries a `range`), or an embed. Rendered as the
/// `❝ quote ❞ — re: X` chip at the top of a reply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostReference {
    /// The action this post references.
    pub antecedent_action_id: String,
    pub ordinal: i64,
    /// Quoted character range within the antecedent's content, if a quote.
    pub range_start: Option<i64>,
    pub range_end: Option<i64>,
    pub annotation: Option<String>,
}

/// One render-row of the threaded space — an item's current generation,
/// flattened from the reply DAG into a list (so `list()` virtualization
/// survives). The flattener (`build_post_tree`) encodes the spine-vs-branch
/// rule: the spine follows the first (chronological) reply and stays at the
/// same depth; later siblings of a post become indented branches (`depth + 1`,
/// `is_branch = true`). Regenerations are generations, not siblings — items are
/// resolved to their current tip, so they never appear as branches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostNode {
    pub action_id: String,
    pub item_id: String,
    /// The structural (`reply`) parent post, if any. `None` for a thread root.
    /// Threading follows **item identity**: a reply edge whose antecedent was
    /// superseded (its item edited/regenerated) resolves to the item's current
    /// tip, so this is always the id of a rendered post — never a dangling
    /// superseded generation.
    pub parent_action_id: Option<String>,
    pub participant: PostParticipant,
    /// `user_input` | `inference` | … (the post's action type).
    pub action_type: String,
    /// Derived 0-based generation number of the current tip.
    pub generation: i64,
    /// Total number of generations of this item (`>= 1`); drives the 5.4
    /// generation switcher.
    pub generation_count: i64,
    /// Always `true` — rows are resolved to the item tip. Carried for API
    /// symmetry and the future non-tip render.
    pub is_current: bool,
    pub model: Option<String>,
    pub credits_consumed: Option<i64>,
    /// Edge relation to the structural parent (`reply`); `None` for a root.
    pub relation: Option<String>,
    /// Indent level: `0` is the spine; `> 0` is an indented branch.
    pub depth: usize,
    /// `true` when this post is a non-first reply to its parent — the head of a
    /// branch off the spine.
    pub is_branch: bool,
    pub blocks: Vec<PostBlock>,
    pub references: Vec<PostReference>,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub context_length: u64,
    /// Credits charged per prompt token. Credits are micro-USD-denominated,
    /// so this is numerically the same as USD per million prompt tokens.
    /// Zero for per-request-priced models.
    pub prompt_credits_per_token: f64,
    /// Credits charged per completion token (see
    /// [`prompt_credits_per_token`](Self::prompt_credits_per_token)).
    pub completion_credits_per_token: f64,
    /// Flat per-request price for models that charge per request rather
    /// than per token (e.g. transcription); `None` for token-priced models.
    pub request_credits: Option<f64>,
}

/// One credential row with its lifecycle state, as computed by the local
/// `credential_lifecycle` view: `active` (spendable), `spending` (a spend
/// proof is in flight / unsettled), `spent` (settled into a successor), or
/// `expired` (issuer key past its expiry).
#[derive(Clone, Debug)]
pub struct CredentialLifecycleInfo {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub created_at: i64,
    pub state: String,
    /// For `spending`/`spent` credentials, the amount charged by the
    /// in-flight (or settled) spend.
    pub spend_amount: Option<i64>,
}

/// One row of the attestation listing in the Record.
#[derive(Clone, Debug)]
pub struct AttestationInfo {
    pub hash: String,
    pub pcr_digest: Option<String>,
    pub created_at: i64,
    /// Size of the stored raw document, in bytes.
    pub doc_bytes: i64,
    /// Number of recorded connections that presented this attestation.
    pub connection_count: i64,
}

/// The full raw attestation document for the Record's detail view.
#[derive(Clone, Debug)]
pub struct AttestationDetail {
    pub hash: String,
    pub pcr_digest: Option<String>,
    pub created_at: i64,
    pub doc: Vec<u8>,
}

/// One row of the request listing in the Record — summary only; the raw
/// header/body payloads come from [`AppCore::request_detail`].
#[derive(Clone, Debug)]
pub struct RequestInfo {
    pub id: String,
    pub method: String,
    pub path: String,
    pub response_status: Option<i64>,
    pub duration_ms: Option<i64>,
    pub request_at: i64,
    pub error: Option<String>,
    pub attempt_number: i64,
    pub credential_nonce: Option<String>,
    pub transport: Option<String>,
    pub base_url: Option<String>,
    pub attestation_hash: Option<String>,
}

/// The full recorded request/response pair, raw bodies included. This is
/// the user's own traffic on their own machine — nothing is redacted.
#[derive(Clone, Debug)]
pub struct RequestDetail {
    pub id: String,
    pub method: String,
    pub path: String,
    pub request_headers: Option<String>,
    pub request_body: Option<Vec<u8>>,
    pub response_status: Option<i64>,
    pub response_headers: Option<String>,
    pub response_body: Option<Vec<u8>>,
    pub request_at: i64,
    pub response_at: Option<i64>,
    pub duration_ms: Option<i64>,
    pub error: Option<String>,
    pub retry_of_id: Option<String>,
    pub attempt_number: i64,
    pub credential_nonce: Option<String>,
    pub action_id: Option<String>,
    pub transport: Option<String>,
    pub base_url: Option<String>,
    pub attestation_hash: Option<String>,
    /// The space this request belongs to, if any (via action → space join).
    pub space_id: Option<String>,
    /// The space's title, if set (may be `None` for untitled spaces).
    pub space_title: Option<String>,
    /// The configured backend this request was routed through, if recorded.
    pub backend_id: Option<String>,
    /// That backend's display name (soft-removed backends keep theirs).
    pub backend_display_name: Option<String>,
}

/// One row of the `spend_trail` view: credential → request → action → space.
#[derive(Clone, Debug)]
pub struct SpendTrailEntry {
    pub credential_nonce: String,
    pub spend_amount: Option<i64>,
    pub credential_state: String,
    pub request_id: String,
    pub method: String,
    pub path: String,
    pub request_at: i64,
    pub duration_ms: Option<i64>,
    pub attempt_number: i64,
    pub action_id: Option<String>,
    pub action_type: Option<String>,
    pub model: Option<String>,
    pub credits_consumed: Option<i64>,
    pub intent: Option<String>,
    pub space_id: Option<String>,
    pub space_title: Option<String>,
    pub linkability: Option<String>,
}

/// Default number of credits to allocate into a fresh anonymous credential
/// when a chat needs one and the account has balance available
/// (auto-provisioning). The actual amount is
/// `min(available, max(DEFAULT_ALLOCATION_CREDITS, required))` — see
/// [`auto_allocation_amount`].
///
/// Why 1,000,000: credits are micro-USD-denominated (the server's
/// `PRICING_SCALE_FACTOR` is 1e6 — `usd_per_M_tokens × markup` becomes
/// credits-per-token directly, e.g. gemma4-31b output $1.00/M × 1.5 markup
/// = 1.5 credits/token). A single chat turn's worst-case hold is
/// `chargeable_prompt_tokens(bytes, msgs) × prompt_rate + 4096 ×
/// completion_rate` (the shared `eidola-common` contract): ≈6,200 credits
/// for the default gemma4-31b, ≈32,000 for the most expensive catalog
/// models (output 7.875 credits/token) — mostly refunded after the actual
/// usage settles. 1,000,000 credits ($1.00 of balance) therefore covers
/// ~30 worst-case holds or 100+ typical turns: small enough that only a
/// sliver of the account balance is parked in one unlinkable credential
/// at a time, large enough that re-allocation (an account-linked,
/// timing-correlatable operation) stays infrequent.
pub const DEFAULT_ALLOCATION_CREDITS: i64 = 1_000_000;

/// How long the ACT provisioning queue waits for an in-flight credential
/// refund (a concurrent turn's recovery) to free spendable balance before
/// giving up with [`AppError::ProvisioningTimeout`]. Bounds the worst case
/// where a sibling turn holds the only coverage mid-spend and its refund never
/// lands.
const PROVISION_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Poll cadence while the provisioning queue waits on an in-flight refund.
const PROVISION_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Decide how many credits to auto-allocate, given the account's available
/// balance and the credits required for the operation that triggered
/// provisioning. Pure function so the decision logic is unit-testable
/// without HTTP.
///
/// Returns [`AppError::InsufficientBalance`] when the balance cannot cover
/// even the required amount; otherwise the chunk size:
/// `min(available, max(DEFAULT_ALLOCATION_CREDITS, required))`.
fn auto_allocation_amount(available: i64, required: i64) -> Result<i64, AppError> {
    if available < required {
        return Err(AppError::InsufficientBalance {
            available,
            required,
        });
    }
    Ok(available.min(DEFAULT_ALLOCATION_CREDITS.max(required)))
}

/// Incremental events emitted by `AppCore::chat_stream`. The terminal
/// outcome is the function's `Result<ChatResult, AppError>` return value;
/// senders close their channel when the function returns.
#[derive(Clone, Debug)]
pub enum ChatStreamEvent {
    /// A piece of the model's reasoning ("thinking") output. Append to a
    /// running buffer; treat empty events as no-ops.
    ReasoningDelta(String),
    /// A piece of the assistant's answer text. Append to a running buffer.
    ContentDelta(String),
}

// ============================================================================
// Inner — shared state used by AppCore, wrapped in Arc so it can move into
// spawned futures on the owned tokio runtime.
// ============================================================================

struct Inner {
    config_path: PathBuf,
    data_dir: PathBuf,
    db: tokio::sync::OnceCell<turso::Database>,
    /// Cached update-check state (last result + accepted-claims choice),
    /// lazily loaded from `<data_dir>/update-state.json` on first access.
    update_state: Mutex<Option<updates::UpdateState>>,
    /// Latch so `start_update_polling` spawns at most one poll loop.
    update_polling: std::sync::atomic::AtomicBool,
    /// Invalidation bus — emits a [`Change`] after every durable commit.
    bus: BroadcastSource,
    /// Runtime local-inference state (downloads in flight, running
    /// llama.cpp engines). `Arc` so core-owned transfer/supervisor tasks
    /// can hold it beyond the initiating call.
    local: Arc<local_models::LocalRuntime>,
    /// The wallet-level ACT provisioning queue. Serializes the spend
    /// credential-acquisition + spend-proof + pending-refund step across
    /// concurrent turns, so two turns fired at once (multi-participant
    /// fan-out) can never both grab the same active credential — the first
    /// flips it to `spending` inside the lock, so the second either finds
    /// another active credential, auto-allocates fresh from balance (the
    /// pool), or waits bounded for an in-flight refund. Held only around the
    /// provisioning step in `prepare_turn`; the HTTP request runs outside it.
    spend_gate: tokio::sync::Mutex<()>,
    /// Test-only HTTP client override. When `Some`, [`Inner::build_client`]
    /// returns a clone of this client *instead of* constructing the
    /// attesting client, letting integration tests drive `chat`/`chat_stream`
    /// (and every other HTTP path) against a plain-HTTP mock upstream without
    /// satisfying the per-handshake enclave attestation. Always `None` in
    /// production — only [`AppCore::with_test_http_client`] ever sets it — so
    /// the production attestation path is unchanged.
    http_override: Option<reqwest::Client>,
}

// --- Config helpers (sync) ---------------------------------------------------

impl Inner {
    fn load_config(&self) -> Config {
        Config::load_from(&self.config_path)
    }

    fn require_credentials<'a>(&self, cfg: &'a Config) -> Result<(&'a str, &'a str), AppError> {
        match (&cfg.account_id, &cfg.account_secret) {
            (Some(id), Some(secret)) => Ok((id, secret)),
            _ => Err(AppError::NotConfigured {
                message: "account not configured".into(),
            }),
        }
    }

    /// The **live** default-template id: the configured `default_template` when
    /// it names a non-removed template, otherwise the seeded default. New
    /// spaces are instantiated from this, so a config pointing at a
    /// removed/absent template falls back honestly to a template that exists.
    async fn resolve_default_template_id(
        &self,
        conn: &turso::Connection,
    ) -> Result<String, AppError> {
        let cfg = self.load_config();
        let configured = cfg.default_template().to_string();
        let live = db::get_space_template(conn, &configured)
            .await?
            .filter(|t| t.removed_at.is_none())
            .is_some();
        Ok(if live {
            configured
        } else {
            config::DEFAULT_TEMPLATE_ID.to_string()
        })
    }

    /// The transitional resolved "default model" (see `ConfigState::default_model`):
    /// the default template's first agent participant's `model_ref`, falling
    /// back to [`config::DEFAULT_MODEL`] when the template has no agent.
    async fn resolve_default_model(&self) -> Result<String, AppError> {
        let conn = self.db_conn().await?;
        let template_id = self.resolve_default_template_id(&conn).await?;
        let model = db::list_template_owned_participants(&conn, &template_id)
            .await?
            .into_iter()
            .find(|p| p.kind == "agent")
            .and_then(|p| p.model_ref)
            .unwrap_or_else(|| config::DEFAULT_MODEL.to_string());
        Ok(model)
    }
}

// --- Update checking ----------------------------------------------------------

impl Inner {
    /// Cached-or-loaded copy of the persisted update state.
    fn update_state_snapshot(&self) -> updates::UpdateState {
        let mut guard = self.update_state.lock().expect("update_state lock");
        guard
            .get_or_insert_with(|| updates::load_state(&self.data_dir))
            .clone()
    }

    /// Replace the cached state and persist it. A disk write failure is
    /// logged, not fatal — the in-memory state stays authoritative for the
    /// rest of the run.  Emits [`Change::UpdateState`] after the write.
    fn store_update_state(&self, state: updates::UpdateState) {
        if let Err(e) = updates::save_state(&self.data_dir, &state) {
            eprintln!("warning: failed to persist update-check state: {e}");
        }
        *self.update_state.lock().expect("update_state lock") = Some(state);
        self.bus.emit(Change::UpdateState);
    }

    /// Run one check and fold it into the persisted state (a `CheckFailed`
    /// never clears a standing security state — see
    /// [`updates::UpdateState::absorb`]). Returns the *effective* snapshot:
    /// what the UI should now show.
    async fn run_update_check(&self) -> updates::UpdateCheckSnapshot {
        let mut state = self.update_state_snapshot();
        let feed_url = self.load_config().update_feed_url();

        let result = match updater::build_http_client() {
            Ok(client) => {
                let mut ctx = updates::CheckContext::new(feed_url, env!("CARGO_PKG_VERSION"));
                ctx.accepted = state.accepted.clone();
                updates::check_for_update(&client, &ctx).await
            }
            Err(e) => updates::UpdateCheckResult::CheckFailed {
                message: format!("constructing HTTPS client: {e}"),
            },
        };

        state.absorb(updates::UpdateCheckSnapshot {
            checked_at_ms: now_ms(),
            result,
        });
        let effective = state.last.clone().expect("absorb always leaves a snapshot");
        self.store_update_state(state);
        effective
    }

    fn accept_changed_claims(
        &self,
        version: String,
        manifest_sha256: String,
    ) -> Result<(), AppError> {
        let mut state = self.update_state_snapshot();
        state.accepted = Some(updates::AcceptedClaims {
            version: version.clone(),
            manifest_sha256: manifest_sha256.clone(),
            accepted_at_ms: now_ms(),
        });

        // If the standing result is the claims-changed release being
        // accepted, rewrite it as an available update so UIs reflect the
        // choice without waiting for the next poll.
        if let Some(snapshot) = state.last.as_mut()
            && let updates::UpdateCheckResult::ClaimsChanged { release, .. } = &snapshot.result
            && release.version == version
            && release
                .manifest_sha256
                .eq_ignore_ascii_case(&manifest_sha256)
        {
            let mut release = release.clone();
            release.claims_accepted = true;
            snapshot.result = updates::UpdateCheckResult::UpdateAvailable { release };
        }

        self.store_update_state(state);
        Ok(())
    }
}

// --- Async infrastructure ----------------------------------------------------

impl Inner {
    async fn db_conn(&self) -> Result<turso::Connection, AppError> {
        let database = self.db.get_or_try_init(|| db::open(&self.data_dir)).await?;
        // FK enforcement is per-connection (turso defaults it OFF), and the
        // scope-owned schema depends on it — enable it on every connection.
        db::connect(database).await
    }

    /// Load the `eidola` backend row and resolve its connection + trust
    /// bundle against the embedded trust-root pin. This is the single place
    /// URL / measurements / hardware CAs are sourced now that they live on
    /// the row instead of `Config`.
    async fn eidola_resolved(&self) -> Result<EidolaResolved, AppError> {
        let conn = self.db_conn().await?;
        let row = db::get_backend(&conn, backends::EIDOLA_BACKEND_ID).await?;
        EidolaResolved::from_row(row.as_ref())
    }

    /// The public [`EidolaTrust`] DTO: the resolved bundle plus the
    /// override-vs-pin honesty flags the UIs render.
    async fn eidola_trust(&self) -> Result<EidolaTrust, AppError> {
        let resolved = self.eidola_resolved().await?;
        let to_info = |m: &tinfoil_verifier::EnclaveMeasurement| MeasurementInfo {
            snp: m.snp_measurement.clone(),
            tdx_rtmr1: m.tdx_measurement.rtmr1.clone(),
            tdx_rtmr2: m.tdx_measurement.rtmr2.clone(),
        };
        Ok(EidolaTrust {
            base_url: resolved.base_url.clone(),
            base_url_pin: trust_root::SERVER_URL.to_string(),
            base_url_is_override: resolved.base_url_is_override,
            trusted_measurements: resolved.measurements.iter().map(to_info).collect(),
            trusted_measurements_are_override: resolved.measurements_are_override,
            pinned_measurement: to_info(&trust_root::server_measurement()),
            has_hardware_root_ca: resolved.hardware_root_ca.is_some(),
            hardware_root_ca_pem: resolved.hardware_root_ca.clone(),
            has_hardware_intermediate_ca: resolved.hardware_intermediate_ca.is_some(),
            hardware_intermediate_ca_pem: resolved.hardware_intermediate_ca.clone(),
        })
    }

    /// Read the eidola row's raw enclave-measurement override list (the
    /// *stored* overrides, not the pin-resolved set). Empty when the column
    /// is NULL — the read-modify-write base for trust/untrust.
    async fn eidola_measurement_overrides(
        &self,
    ) -> Result<Vec<tinfoil_verifier::EnclaveMeasurement>, AppError> {
        let conn = self.db_conn().await?;
        let row = db::get_backend(&conn, backends::EIDOLA_BACKEND_ID).await?;
        match row.and_then(|r| r.trusted_measurements) {
            Some(json) if !json.trim().is_empty() => {
                serde_json::from_str(&json).map_err(|e| AppError::Database {
                    message: format!("invalid trusted_measurements JSON on eidola row: {e}"),
                })
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Add a measurement to the eidola row's override list (idempotent by
    /// SNP measurement). Returns whether it was newly added.
    async fn trust_measurement(
        &self,
        entry: tinfoil_verifier::EnclaveMeasurement,
    ) -> Result<bool, AppError> {
        let mut list = self.eidola_measurement_overrides().await?;
        if list.iter().any(|m| {
            m.snp_measurement
                .eq_ignore_ascii_case(&entry.snp_measurement)
        }) {
            return Ok(false);
        }
        list.push(entry);
        let json = serde_json::to_string(&list).map_err(|e| AppError::Database {
            message: format!("failed to serialize trusted_measurements: {e}"),
        })?;
        self.update_backend(
            backends::EIDOLA_BACKEND_ID,
            backends::BackendUpdate {
                trusted_measurements: Some(Some(json)),
                ..Default::default()
            },
        )
        .await?;
        Ok(true)
    }

    /// Remove a measurement (by SNP key) from the eidola row's override list.
    /// Clears the column back to NULL (= pin) when the list empties. Returns
    /// whether a measurement was removed.
    async fn untrust_measurement(&self, key: String) -> Result<bool, AppError> {
        let mut list = self.eidola_measurement_overrides().await?;
        let Some(pos) = list
            .iter()
            .position(|m| m.snp_measurement.eq_ignore_ascii_case(&key))
        else {
            return Ok(false);
        };
        list.remove(pos);
        let json = if list.is_empty() {
            None
        } else {
            Some(
                serde_json::to_string(&list).map_err(|e| AppError::Database {
                    message: format!("failed to serialize trusted_measurements: {e}"),
                })?,
            )
        };
        self.update_backend(
            backends::EIDOLA_BACKEND_ID,
            backends::BackendUpdate {
                trusted_measurements: Some(json),
                ..Default::default()
            },
        )
        .await?;
        Ok(true)
    }

    /// Set/clear a hardware CA (ARK or ASK) override on the eidola row.
    /// `pem = None` clears back to the vendor chain.
    async fn set_hardware_ca(
        &self,
        which: HardwareCa,
        pem: Option<String>,
    ) -> Result<(), AppError> {
        let pem = match pem {
            Some(p) if !p.trim().is_empty() => {
                config::parse_cert_config(Some(&p), which.field_name())?;
                Some(p.trim().to_string())
            }
            _ => None,
        };
        let update = match which {
            HardwareCa::Root => backends::BackendUpdate {
                hardware_root_ca: Some(pem),
                ..Default::default()
            },
            HardwareCa::Intermediate => backends::BackendUpdate {
                hardware_intermediate_ca: Some(pem),
                ..Default::default()
            },
        };
        self.update_backend(backends::EIDOLA_BACKEND_ID, update)
            .await
    }

    async fn build_client(
        &self,
        config: &Config,
        eidola: &EidolaResolved,
        attestation_observer: Option<tinfoil_verifier::AttestationObserver>,
    ) -> Result<reqwest::Client, AppError> {
        // Test seam: a plain-HTTP client injected via
        // `AppCore::with_test_http_client` short-circuits attestation so
        // integration tests can drive the HTTP paths against a mock upstream.
        // `None` in every production build, so the attesting path below is the
        // only one that ever runs outside tests. The attestation observer is
        // simply never invoked on this client (no enclave to observe).
        if let Some(client) = &self.http_override {
            return Ok(client.clone());
        }

        let hardware_root_der =
            config::parse_cert_config(eidola.hardware_root_ca.as_deref(), "hardware_root_ca")?;
        let hardware_intermediate_der = config::parse_cert_config(
            eidola.hardware_intermediate_ca.as_deref(),
            "hardware_intermediate_ca",
        )?;

        tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
            allowed_measurements: &eidola.measurements,
            inference_base_url: &eidola.base_url,
            atc_url: config.attestation_url.as_deref(),
            enclave_repo: Some(config.attestation_repo()),
            trusted_ark_der: hardware_root_der.as_deref(),
            trusted_ask_der: hardware_intermediate_der.as_deref(),
            tdx_advisory_allowlist: None,
            tdx_observer: None,
            snp_min_tcb: None,
            snp_observer: None,
            attestation_observer,
            tls_roots: load_native_root_store(),
        })
        .await
        .map_err(|e| AppError::Attestation {
            message: format!("attestation client build failed: {e}"),
        })
    }
}

/// The `eidola` backend's resolved connection + trust bundle. Overrides come
/// from the eidola row; NULL columns fall back to the embedded trust-root
/// pin. Built once per operation and shared by [`Inner::build_client`] and
/// the request-URL construction so both agree on where they're talking.
struct EidolaResolved {
    base_url: String,
    base_url_is_override: bool,
    measurements: Vec<tinfoil_verifier::EnclaveMeasurement>,
    measurements_are_override: bool,
    hardware_root_ca: Option<String>,
    hardware_intermediate_ca: Option<String>,
}

impl EidolaResolved {
    fn from_row(row: Option<&db::BackendRow>) -> Result<Self, AppError> {
        let base_url_override = row.and_then(|r| r.base_url.clone());
        let base_url_is_override = base_url_override.is_some();
        let base_url = base_url_override.unwrap_or_else(|| trust_root::SERVER_URL.to_string());

        let measurements_override: Vec<tinfoil_verifier::EnclaveMeasurement> =
            match row.and_then(|r| r.trusted_measurements.as_deref()) {
                Some(json) if !json.trim().is_empty() => {
                    serde_json::from_str(json).map_err(|e| AppError::Database {
                        message: format!("invalid trusted_measurements JSON on eidola row: {e}"),
                    })?
                }
                _ => Vec::new(),
            };
        let measurements_are_override = !measurements_override.is_empty();
        let measurements = if measurements_are_override {
            measurements_override
        } else {
            vec![trust_root::server_measurement()]
        };

        Ok(EidolaResolved {
            base_url,
            base_url_is_override,
            measurements,
            measurements_are_override,
            hardware_root_ca: row.and_then(|r| r.hardware_root_ca.clone()),
            hardware_intermediate_ca: row.and_then(|r| r.hardware_intermediate_ca.clone()),
        })
    }
}

/// Which hardware attestation-chain certificate an override targets.
#[derive(Clone, Copy)]
enum HardwareCa {
    Root,
    Intermediate,
}

impl HardwareCa {
    fn field_name(self) -> &'static str {
        match self {
            HardwareCa::Root => "hardware_root_ca",
            HardwareCa::Intermediate => "hardware_intermediate_ca",
        }
    }
}

// --- High-level async operations (run on the owned tokio runtime) ------------

impl Inner {
    async fn account_show(&self) -> Result<AccountShowResult, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let (id, secret) = self.require_credentials(&cfg)?;

        let client = self.build_client(&cfg, &eidola, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/account"))
            .basic_auth(id, Some(secret))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let account: GetAccountResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(AccountShowResult {
            id: account.id.to_string(),
            stripe_customer_id: account.stripe_customer_id,
            created_at: iso_to_ms(&account.created_at)?,
        })
    }

    async fn account_create(&self) -> Result<AccountCreateResult, AppError> {
        let cfg = self.load_config();

        if cfg.account_id.is_some() || cfg.account_secret.is_some() {
            return Err(AppError::Config {
                message: "account credentials already configured — reset first".into(),
            });
        }

        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let client = self.build_client(&cfg, &eidola, None).await?;
        let resp = client
            .post(format!("{base_url}/v1/account"))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let created: CreateAccountResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        let mut cfg = self.load_config();
        cfg.account_id = Some(created.account_id.to_string());
        cfg.account_secret = Some(created.secret.clone());
        cfg.save_to(&self.config_path)?;
        self.bus.emit(Change::Config);

        // Record acceptance of the currently required terms/privacy versions
        // against the new account. The caller is responsible for having
        // presented the documents first (the GUI's consent checkbox, the
        // CLI's --accept-terms flag). Best-effort: if this fails, the
        // server's 428 gate re-prompts at the first purchase or conversion,
        // so a network blip here must not fail the already-created account.
        let _ = self.accept_current_terms().await;

        Ok(AccountCreateResult {
            id: created.account_id.to_string(),
            secret: created.secret,
            created_at: iso_to_ms(&created.created_at)?,
        })
    }

    /// The documents (terms of service, privacy policy) whose current
    /// versions the server requires accounts to accept, with published URLs
    /// and content hashes. Empty when the server has no acceptance gate
    /// configured.
    async fn current_terms(&self) -> Result<Vec<TermsDocument>, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();

        let client = self.build_client(&cfg, &eidola, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/terms"))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let terms: TermsResponse = serde_json::from_str(&body).map_err(|e| AppError::Network {
            message: format!("failed to parse response: {e}"),
        })?;

        Ok(terms
            .documents
            .into_iter()
            .map(|d| TermsDocument {
                document: d.document,
                version: d.version,
                url: d.url,
                sha256: d.sha256,
            })
            .collect())
    }

    /// Record acceptance of every currently required document version
    /// against the configured account, returning what was accepted.
    ///
    /// Callers are responsible for having *presented* the documents to the
    /// user first — this method transmits consent, it does not obtain it.
    async fn accept_current_terms(&self) -> Result<Vec<TermsDocument>, AppError> {
        let docs = self.current_terms().await?;
        if docs.is_empty() {
            return Ok(docs);
        }

        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let (id, secret) = self.require_credentials(&cfg)?;
        let client = self.build_client(&cfg, &eidola, None).await?;

        for d in &docs {
            let resp = client
                .post(format!("{base_url}/v1/account/terms"))
                .basic_auth(id, Some(secret))
                .json(&serde_json::json!({
                    "document": d.document,
                    "sha256": d.sha256,
                }))
                .send()
                .await
                .map_err(AppError::from_request)?;

            let (status, body) = read_response(resp).await?;
            check_status(status, &body)?;
        }

        Ok(docs)
    }

    async fn account_prices(&self) -> Result<Vec<PriceInfo>, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();

        let client = self.build_client(&cfg, &eidola, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/prices"))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let prices: ListPricesResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(prices
            .data
            .into_iter()
            .map(|p| {
                let amount_display = p
                    .unit_amount
                    .map(|a| format!("{}.{:02} {}", a / 100, a % 100, p.currency.to_uppercase()))
                    .unwrap_or_else(|| "free".to_string());

                let recurrence = p
                    .recurring
                    .as_ref()
                    .map(|r| {
                        if r.interval_count == 1 {
                            format!("/{}", r.interval)
                        } else {
                            format!("/{}x{}", r.interval_count, r.interval)
                        }
                    })
                    .unwrap_or_default();

                PriceInfo {
                    id: p.id,
                    product_name: p.product_name,
                    product_description: p.product_description,
                    amount_display,
                    recurrence,
                    credits: p.credits,
                }
            })
            .collect())
    }

    async fn account_checkout(&self, price_id: &str) -> Result<String, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let (id, secret) = self.require_credentials(&cfg)?;

        let client = self.build_client(&cfg, &eidola, None).await?;
        let resp = client
            .post(format!("{base_url}/v1/account/checkout"))
            .basic_auth(id, Some(secret))
            .json(&serde_json::json!({ "price_id": price_id }))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let checkout: CheckoutUrlResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(checkout.checkout_url)
    }

    async fn account_balances(&self) -> Result<BalancesResult, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let (id, secret) = self.require_credentials(&cfg)?;

        let client = self.build_client(&cfg, &eidola, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/account/balances"))
            .basic_auth(id, Some(secret))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let balances: BalancesResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(BalancesResult {
            available: balances.available,
            pools: balances
                .pools
                .into_iter()
                .map(|p| {
                    let expires_at = p.expires_at.as_deref().map(iso_to_ms).transpose()?;
                    Ok::<_, AppError>(BalancePoolInfo {
                        amount: p.amount,
                        source: p.source,
                        expires_at,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    async fn account_allocate(&self, credits: i64) -> Result<AllocateResult, AppError> {
        if credits <= 0 {
            return Err(AppError::Credential {
                message: "credits must be greater than 0".into(),
            });
        }

        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let (account_id, secret) = self.require_credentials(&cfg)?;
        let client = self.build_client(&cfg, &eidola, None).await?;

        // 1. Fetch issuer keys
        let resp = client
            .get(format!("{base_url}/v1/keys"))
            .send()
            .await
            .map_err(AppError::from_request)?;
        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let keys: ListKeysResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse keys response: {e}"),
            })?;

        let expected_ds = cfg.domain_separator();
        let now = now_ms();
        // Every issuer key shares the same domain separator — rotation mints
        // new keys, never new separators — so the separator alone does not
        // identify a single key. `/v1/keys` lists all still-accepted keys
        // oldest-first, which during a rotation grace window includes the
        // just-rotated-out key whose accept window has not yet closed.
        // Issuance always signs with the *current* key (the server's
        // `get_current_issuer_key`, newest by `issue_from`), so we must verify
        // against that same key: select the key whose issuing window covers
        // now (`issue_from <= now < issue_until`), preferring the newest such
        // key. Selecting the first separator match (the old behavior) grabbed
        // the oldest accepted key and failed proof verification whenever an
        // out-of-issuance-but-still-accepted key preceded the current one.
        let key = keys
            .data
            .iter()
            .filter(|k| k.domain_separator == expected_ds)
            .filter_map(|k| {
                let issue_from = iso_to_ms(&k.issue_from).ok()?;
                let issue_until = iso_to_ms(&k.issue_until).ok()?;
                (issue_from <= now && now < issue_until).then_some((issue_from, k))
            })
            .max_by_key(|(issue_from, _)| *issue_from)
            .map(|(_, k)| k)
            .ok_or_else(|| {
                let server_ds: Vec<&str> = keys
                    .data
                    .iter()
                    .map(|k| k.domain_separator.as_str())
                    .collect();
                AppError::Credential {
                    message: format!(
                        "no issuer key currently valid for issuance matches expected \
                         domain separator \"{expected_ds}\"\n\
                         server advertised: {server_ds:?}"
                    ),
                }
            })?;

        let public_key_cbor =
            URL_SAFE_NO_PAD
                .decode(&key.public_key)
                .map_err(|e| AppError::Credential {
                    message: format!("invalid base64 public key: {e}"),
                })?;

        let public_key =
            PublicKey::from_cbor(&public_key_cbor).map_err(|e| AppError::Credential {
                message: format!("invalid public key CBOR: {e}"),
            })?;

        let params = params_from_domain_separator(expected_ds)?;

        // 2. Open DB and store issuer key
        let db_conn = self.db_conn().await?;
        let params_hash = blake3::hash(key.domain_separator.as_bytes())
            .to_hex()
            .to_string();
        let expires_at = iso_to_ms(&key.issue_until)?;

        db::upsert_issuer_key(
            &db_conn,
            &key.id,
            &params_hash,
            &public_key_cbor,
            key.domain_separator.as_bytes(),
            expires_at,
            now,
        )
        .await?;

        // 3. Create PreIssuance checkpoint
        let pre_issuance = PreIssuance::random(OsRng);
        let pre_issuance_cbor = pre_issuance.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode pre_issuance: {e}"),
        })?;
        let pre_credential_id = Uuid::now_v7().to_string();
        db::insert_pre_credential_issuance(
            &db_conn,
            &pre_credential_id,
            &key.id,
            &pre_issuance_cbor,
            credits,
            now,
        )
        .await?;

        // 4. Send issuance request
        let issuance_request = pre_issuance.request(&params, OsRng);
        let request_cbor = issuance_request
            .to_cbor()
            .map_err(|e| AppError::Credential {
                message: format!("failed to encode issuance request: {e}"),
            })?;

        let resp = client
            .post(format!("{base_url}/v1/account/credentials"))
            .basic_auth(account_id, Some(secret))
            .json(&serde_json::json!({
                "issuance_request": URL_SAFE_NO_PAD.encode(&request_cbor),
                "credits": credits,
            }))
            .send()
            .await
            .map_err(AppError::from_request)?;
        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let issued: IssueCredentialsResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse issuance response: {e}"),
            })?;

        // 5. Construct CreditToken
        let response_cbor = URL_SAFE_NO_PAD
            .decode(&issued.issuance_response)
            .map_err(|e| AppError::Credential {
                message: format!("invalid issuance response base64: {e}"),
            })?;
        let issuance_response =
            IssuanceResponse::from_cbor(&response_cbor).map_err(|e| AppError::Credential {
                message: format!("invalid issuance response CBOR: {e}"),
            })?;
        let credit_token = pre_issuance
            .to_credit_token::<128>(&params, &public_key, &issuance_request, &issuance_response)
            .map_err(|e| AppError::Credential {
                message: format!("failed to construct credit token: {e}"),
            })?;

        // 6. Store credential
        let token_cbor = credit_token.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode credit token: {e}"),
        })?;
        let nonce_hex = hex_encode(&credit_token.nullifier().to_bytes());
        let token_credits =
            scalar_to_credit::<128>(&credit_token.credits()).map_err(|e| AppError::Credential {
                message: format!("invalid credit amount in token: {e}"),
            })?;

        db::insert_credential(
            &db_conn,
            &nonce_hex,
            &pre_credential_id,
            &issued.issuer_key_id,
            &token_cbor,
            token_credits as i64,
            0,
            now,
        )
        .await?;

        // Allocation moves credits from the account balance into a credential:
        // both the wallet and account domains changed.
        self.bus.emit(Change::Wallet);
        self.bus.emit(Change::Account);

        Ok(AllocateResult {
            nonce: nonce_hex,
            credits: issued.credits,
            issuer_key_id: issued.issuer_key_id,
        })
    }

    async fn wallet_credentials(&self) -> Result<Vec<CredentialInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_active_credentials(&db_conn).await?;
        Ok(rows
            .into_iter()
            .map(|c| CredentialInfo {
                nonce: c.nonce,
                credits: c.credits,
                generation: c.generation,
            })
            .collect())
    }

    async fn wallet_spending_credentials(&self) -> Result<Vec<InFlightCredentialInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_spending_credentials(&db_conn).await?;
        Ok(rows
            .into_iter()
            .map(|r| InFlightCredentialInfo {
                nonce: r.nonce,
                credits: r.credits,
                generation: r.generation,
                spend_amount: r.spend_amount,
            })
            .collect())
    }

    async fn wallet_lifecycle(&self) -> Result<Vec<CredentialLifecycleInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_credential_lifecycle(&db_conn).await?;
        Ok(rows
            .into_iter()
            .map(|r| CredentialLifecycleInfo {
                nonce: r.nonce,
                credits: r.credits,
                generation: r.generation,
                created_at: r.created_at,
                state: r.state,
                spend_amount: r.spend_amount,
            })
            .collect())
    }

    // --- The Record: read-only queries over the local trail -----------------

    async fn list_attestations(
        &self,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<AttestationInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_attestations(&db_conn, limit, offset).await?;
        Ok(rows
            .into_iter()
            .map(|r| AttestationInfo {
                hash: r.hash,
                pcr_digest: r.pcr_digest,
                created_at: r.created_at,
                doc_bytes: r.doc_bytes,
                connection_count: r.connection_count,
            })
            .collect())
    }

    async fn attestation_detail(&self, hash: &str) -> Result<Option<AttestationDetail>, AppError> {
        let db_conn = self.db_conn().await?;
        Ok(db::get_attestation(&db_conn, hash)
            .await?
            .map(|r| AttestationDetail {
                hash: r.hash,
                pcr_digest: r.pcr_digest,
                created_at: r.created_at,
                doc: r.doc,
            }))
    }

    async fn list_requests(&self, limit: i64, offset: i64) -> Result<Vec<RequestInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_requests(&db_conn, limit, offset).await?;
        Ok(rows
            .into_iter()
            .map(|r| RequestInfo {
                id: r.id,
                method: r.method,
                path: r.path,
                response_status: r.response_status,
                duration_ms: r.duration_ms,
                request_at: r.request_at,
                error: r.error,
                attempt_number: r.attempt_number,
                credential_nonce: r.credential_nonce,
                transport: r.transport,
                base_url: r.base_url,
                attestation_hash: r.attestation_hash,
            })
            .collect())
    }

    async fn request_detail(&self, id: &str) -> Result<Option<RequestDetail>, AppError> {
        let db_conn = self.db_conn().await?;
        Ok(db::get_request(&db_conn, id).await?.map(|r| RequestDetail {
            id: r.id,
            method: r.method,
            path: r.path,
            request_headers: r.request_headers,
            request_body: r.request_body,
            response_status: r.response_status,
            response_headers: r.response_headers,
            response_body: r.response_body,
            request_at: r.request_at,
            response_at: r.response_at,
            duration_ms: r.duration_ms,
            error: r.error,
            retry_of_id: r.retry_of_id,
            attempt_number: r.attempt_number,
            credential_nonce: r.credential_nonce,
            action_id: r.action_id,
            transport: r.transport,
            base_url: r.base_url,
            attestation_hash: r.attestation_hash,
            space_id: r.space_id,
            space_title: r.space_title,
            backend_id: r.backend_id,
            backend_display_name: r.backend_display_name,
        }))
    }

    async fn spend_trail(&self, limit: i64, offset: i64) -> Result<Vec<SpendTrailEntry>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_spend_trail(&db_conn, limit, offset).await?;
        Ok(rows
            .into_iter()
            .map(|r| SpendTrailEntry {
                credential_nonce: r.credential_nonce,
                spend_amount: r.spend_amount,
                credential_state: r.credential_state,
                request_id: r.request_id,
                method: r.method,
                path: r.path,
                request_at: r.request_at,
                duration_ms: r.duration_ms,
                attempt_number: r.attempt_number,
                action_id: r.action_id,
                action_type: r.action_type,
                model: r.model,
                credits_consumed: r.credits_consumed,
                intent: r.intent,
                space_id: r.space_id,
                space_title: r.space_title,
                linkability: r.linkability,
            })
            .collect())
    }

    async fn recover_spending_credentials(&self) -> Result<Vec<String>, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let base_url = eidola.base_url.as_str();
        let client = self.build_client(&cfg, &eidola, None).await?;
        let db_conn = self.db_conn().await?;
        let params = params_from_domain_separator(cfg.domain_separator())?;
        let now = now_ms();

        let rows = db::list_spending_credentials(&db_conn).await?;
        let mut recovered = Vec::new();

        for row in rows {
            let spend_proof_cbor = row.spend_proof_data;

            let spend_proof = match SpendProof::<128>::from_cbor(&spend_proof_cbor) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let pre_refund = match PreRefund::from_cbor(&row.pre_refund_data) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let public_key = match PublicKey::from_cbor(&row.public_key_data) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let issuer_key_hash = match hex_decode(&row.issuer_key_id) {
                Ok(h) => h,
                Err(_) => continue,
            };

            // Reconstruct the PrivateToken auth header
            let challenge_digest = compute_challenge_digest();
            let mut token_bytes = Vec::new();
            token_bytes.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
            token_bytes.extend_from_slice(&challenge_digest);
            token_bytes.extend_from_slice(&issuer_key_hash);
            token_bytes.extend_from_slice(&spend_proof_cbor);
            let token_b64 = URL_SAFE_NO_PAD.encode(&token_bytes);
            let auth_value = format!("PrivateToken token=\"{token_b64}\"");

            if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await
                && process_refund(
                    &refund_obj,
                    &params,
                    &spend_proof,
                    &pre_refund,
                    &public_key,
                    &db_conn,
                    &row.pre_credential_id,
                    row.generation + 1,
                    now,
                )
                .await
                .is_ok()
            {
                recovered.push(row.nonce);
            }
        }

        if !recovered.is_empty() {
            self.bus.emit(Change::Wallet);
        }

        Ok(recovered)
    }

    async fn available_models(&self) -> Result<Vec<ModelInfo>, AppError> {
        let cfg = self.load_config();
        let eidola = self.eidola_resolved().await?;
        let client = self.build_client(&cfg, &eidola, None).await?;

        let models = fetch_models(&client, &eidola.base_url).await?;
        Ok(models
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                context_length: m.context_length,
                prompt_credits_per_token: m.pricing.per_prompt_token.credits_per_unit(),
                completion_credits_per_token: m.pricing.per_completion_token.credits_per_unit(),
                request_credits: m
                    .pricing
                    .per_request
                    .as_ref()
                    .map(ScaledPriceInfo::credits_per_unit),
            })
            .collect())
    }

    async fn list_spaces(&self, include_archived: bool) -> Result<Vec<SpaceInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_spaces(&db_conn, include_archived).await?;
        let mut spaces = Vec::with_capacity(rows.len());
        for r in rows {
            let snippet = db::first_user_text(&db_conn, &r.id)
                .await?
                .as_deref()
                .and_then(snippet_of);
            spaces.push(SpaceInfo {
                id: r.id,
                title: r.title,
                snippet,
                created_at: r.created_at,
                last_activity_at: r.last_activity_at,
                message_count: r.message_count,
                archived_at: r.archived_at,
            });
        }
        Ok(spaces)
    }

    async fn get_space_messages(&self, space_id: &str) -> Result<Vec<SpaceMessage>, AppError> {
        let db_conn = self.db_conn().await?;
        db::get_space(&db_conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        let action_rows = db::get_space_actions_for_context(&db_conn, space_id).await?;
        Ok(actions_to_messages(&action_rows))
    }

    /// Build the threaded-post render tree for a space: each item resolved to
    /// its current generation, the reply DAG flattened (spine flat, genuine
    /// branches indented) into a list of [`PostNode`] render-rows. This is the
    /// render DTO; `get_space_messages` remains the upstream-context view.
    async fn get_space_tree(&self, space_id: &str) -> Result<Vec<PostNode>, AppError> {
        let db_conn = self.db_conn().await?;
        db::get_space(&db_conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        let data = db::get_space_tree_data(&db_conn, space_id).await?;
        Ok(build_post_tree(data))
    }

    /// Create a new space by instantiating the (live) default space template:
    /// the space copies the template's `cascade_limit`, the shared human "You"
    /// joins as owner, and each template agent participant is copied into a
    /// fresh per-space instance. This is the single new-space path so every
    /// space has participants from birth. Returns the new space id.
    async fn instantiate_default_space(
        &self,
        conn: &turso::Connection,
        title: Option<&str>,
        now: i64,
    ) -> Result<String, AppError> {
        let template_id = self.resolve_default_template_id(conn).await?;
        let space_id = Uuid::now_v7().to_string();
        db::instantiate_template(conn, &template_id, &space_id, title, "unlinked", now).await?;
        Ok(space_id)
    }

    /// Resolve the agent participant an inference should be recorded against
    /// **from a model string** — the model-picker **compatibility path** for
    /// `TurnSelector::Model` (the CLI/GUI flow until the wave-3 GUI selects
    /// participants directly). Reuses the space's existing agent (a space-owned
    /// instance from the template, or a referenced global) whose **effective**
    /// `model_ref` matches the (canonicalized) selection, returning its
    /// `(id, scope, effective system_prompt)`; if no member matches the picked
    /// model, mints a fresh **space-owned** agent for it (scope `'space'`,
    /// `owner_space_id` = this space; ownership implies membership, so no
    /// reference row) with no system prompt and returns it.
    ///
    /// This is what keeps a plain model pick working while turns are
    /// participant-aware: an explicit participant never routes here (its config
    /// is read directly in `prepare_turn`); only `TurnSelector::Model` does.
    async fn resolve_or_mint_agent_by_model(
        &self,
        conn: &turso::Connection,
        space_id: &str,
        canonical_model: &str,
        provider_id: &str,
        now: i64,
    ) -> Result<(String, String, Option<String>), AppError> {
        for m in db::space_participants(conn, space_id).await? {
            if m.kind != "agent" {
                continue;
            }
            let matches = m
                .model_ref
                .as_deref()
                .map(canonicalize_model_ref)
                .as_deref()
                == Some(canonical_model);
            if matches {
                return Ok((m.participant_id, m.scope, m.system_prompt));
            }
        }
        // No matching member — mint a fresh SPACE-OWNED agent for the model.
        let pid = Uuid::now_v7().to_string();
        db::insert_participant(
            conn,
            &pid,
            "space",
            Some(space_id),
            None,
            "agent",
            &db::default_agent_label(canonical_model),
            Some(canonical_model),
            None,
            "explicit",
            "member",
            Some(provider_id),
            now,
        )
        .await?;
        Ok((pid, "space".to_string(), None))
    }

    /// Resolve the effective config of an **explicit** space participant
    /// (`TurnSelector::Participant`) into the model + system prompt a turn
    /// needs. Errors if the participant isn't a member of the space, isn't an
    /// agent, or has no model configured (a participant that can't answer).
    async fn resolve_explicit_participant(
        &self,
        conn: &turso::Connection,
        space_id: &str,
        participant_id: &str,
    ) -> Result<db::EffectiveParticipantRow, AppError> {
        let row = db::space_participants(conn, space_id)
            .await?
            .into_iter()
            .find(|m| m.participant_id == participant_id)
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("participant `{participant_id}` is not a member of this space"),
            })?;
        if row.kind != "agent" {
            return Err(AppError::NotConfigured {
                message: format!(
                    "participant `{}` is a {} — only agents can respond",
                    row.label, row.kind
                ),
            });
        }
        if row
            .model_ref
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return Err(AppError::NotConfigured {
                message: format!("participant `{}` has no model configured", row.label),
            });
        }
        Ok(row)
    }

    /// Verify `action_id` exists **and belongs to `space_id`**, returning its
    /// item id. This is the app-layer guard against splicing one space's thread
    /// into another: nothing in the schema constrains an `action_antecedent`
    /// edge (or a turn's `target_action_id`) to a single space — the edge is
    /// keyed on action ids only — so a caller supplying a mismatched
    /// (space, action) pair could otherwise send space A's ancestry to space
    /// B's agent and persist a cross-space reply edge. Typed
    /// [`AppError::NotConfigured`] on a missing or foreign action.
    async fn require_action_in_space(
        &self,
        conn: &turso::Connection,
        action_id: &str,
        space_id: &str,
    ) -> Result<String, AppError> {
        match db::action_item_and_space(conn, action_id).await? {
            Some((item_id, sp)) if sp == space_id => Ok(item_id),
            Some((_, sp)) => Err(AppError::NotConfigured {
                message: format!("action {action_id} belongs to space {sp}, not {space_id}"),
            }),
            None => Err(AppError::NotConfigured {
                message: format!("action not found: {action_id}"),
            }),
        }
    }

    // --- Participant CRUD (per-space) -----------------------------------

    async fn list_space_participants(
        &self,
        space_id: &str,
    ) -> Result<Vec<ParticipantInfo>, AppError> {
        let conn = self.db_conn().await?;
        let rows = db::space_participants(&conn, space_id).await?;
        // Reference detail (base config + raw overrides) for referenced globals,
        // so the GUI can render the edit-everywhere-vs-override-here fork.
        let refs = db::list_space_participant_refs(&conn, space_id).await?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            let mut info = ParticipantInfo::from_effective(r);
            if info.source == "referenced"
                && let Some(base) = db::get_participant(&conn, &info.id).await?
            {
                let ov = refs.iter().find(|x| x.participant_id == info.id);
                info.reference = Some(ParticipantReference {
                    base_label: base.label,
                    base_model_ref: base.model_ref,
                    base_system_prompt: base.system_prompt,
                    base_notify_policy: base.notify_policy,
                    override_label: ov.and_then(|o| o.override_label.clone()),
                    override_model_ref: ov.and_then(|o| o.override_model_ref.clone()),
                    override_system_prompt: ov.and_then(|o| o.override_system_prompt.clone()),
                    override_notify_policy: ov.and_then(|o| o.override_notify_policy.clone()),
                });
            }
            out.push(info);
        }
        Ok(out)
    }

    /// "Override here": write per-membership overrides for a **referenced
    /// global** participant (this space only; the global's own config is
    /// untouched). `None` inner = revert to inherited. Emits
    /// [`Change::Participants`] when anything changed.
    async fn set_space_participant_override(
        &self,
        space_id: &str,
        participant_id: &str,
        ov: ParticipantOverride,
    ) -> Result<(), AppError> {
        // A notify-policy override, when set to a value, must be a valid enum
        // member (the effective config = COALESCE(override, base) must satisfy
        // the schema CHECK on the base column).
        if let Some(Some(p)) = &ov.notify_policy {
            validate_notify_policy(p.trim())?;
        }
        let conn = self.db_conn().await?;
        fn to_ref(o: &Option<Option<String>>) -> Option<Option<&str>> {
            o.as_ref().map(|inner| inner.as_deref())
        }
        let changed = db::update_space_participant_override(
            &conn,
            space_id,
            participant_id,
            to_ref(&ov.label),
            to_ref(&ov.model_ref),
            to_ref(&ov.system_prompt),
            to_ref(&ov.notify_policy),
        )
        .await?;
        if changed {
            self.bus.emit(Change::Participants);
        }
        Ok(())
    }

    async fn add_space_participant(
        &self,
        space_id: &str,
        new: NewParticipant,
    ) -> Result<ParticipantInfo, AppError> {
        let label = new.label.trim().to_string();
        if label.is_empty() {
            return Err(AppError::Config {
                message: "participant label must not be empty".into(),
            });
        }
        let policy = notify_policy_or_default(&new.notify_policy)?;
        let model_ref = new
            .model_ref
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let system_prompt = new.system_prompt.as_deref().filter(|s| !s.is_empty());

        let conn = self.db_conn().await?;
        db::get_space(&conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        let now = now_ms();
        // Added agents are SPACE-OWNED (a fresh per-space instance). Ownership
        // implies membership, so no reference row.
        let pid = Uuid::now_v7().to_string();
        db::insert_participant(
            &conn,
            &pid,
            "space",
            Some(space_id),
            None,
            "agent",
            &label,
            model_ref,
            system_prompt,
            &policy,
            "member",
            None,
            now,
        )
        .await?;
        self.bus.emit(Change::Participants);
        let row = db::get_participant(&conn, &pid)
            .await?
            .ok_or_else(|| AppError::Internal {
                message: "participant vanished after insert".into(),
            })?;
        Ok(ParticipantInfo::from_owned(row))
    }

    /// Edit a participant's **own** config — "edit everywhere" for a global,
    /// "edit this space" for a space-owned row. The complementary "override
    /// here" path (per-membership override, this space only) is the DB
    /// primitive `db::update_space_participant_override` + the COALESCE in
    /// `db::space_participants`; the GUI affordance for it lands in wave 3.
    async fn update_space_participant(
        &self,
        participant_id: &str,
        update: ParticipantUpdate,
    ) -> Result<(), AppError> {
        let policy = match &update.notify_policy {
            Some(p) => Some(validate_notify_policy(p.trim())?),
            None => None,
        };
        let label = match &update.label {
            Some(l) if l.trim().is_empty() => {
                return Err(AppError::Config {
                    message: "participant label must not be empty".into(),
                });
            }
            Some(l) => Some(l.trim().to_string()),
            None => None,
        };
        let conn = self.db_conn().await?;
        let model_ref = update
            .model_ref
            .as_ref()
            .map(|inner| inner.as_deref().filter(|s| !s.is_empty()));
        let system_prompt = update
            .system_prompt
            .as_ref()
            .map(|inner| inner.as_deref().filter(|s| !s.is_empty()));
        let changed = db::update_participant_config(
            &conn,
            participant_id,
            label.as_deref(),
            model_ref,
            system_prompt,
            policy.as_deref(),
            None,
        )
        .await?;
        if changed {
            self.bus.emit(Change::Participants);
        }
        Ok(())
    }

    async fn remove_space_participant(
        &self,
        space_id: &str,
        participant_id: &str,
    ) -> Result<bool, AppError> {
        if participant_id == db::HUMAN_PARTICIPANT_ID {
            return Err(AppError::Config {
                message: "the human participant is shared and cannot be removed from a space"
                    .into(),
            });
        }
        let conn = self.db_conn().await?;
        // A space-owned participant is soft-removed (its row deactivated); a
        // referenced global leaves the space (its reference row's left_at set).
        let removed = match db::get_participant(&conn, participant_id).await? {
            Some(p) if p.scope == "space" && p.owner_space_id.as_deref() == Some(space_id) => {
                db::soft_remove_participant(&conn, participant_id, now_ms()).await?
            }
            _ => db::leave_space_participant(&conn, space_id, participant_id, now_ms()).await?,
        };
        if removed {
            self.bus.emit(Change::Participants);
        }
        Ok(removed)
    }

    // --- Space template CRUD --------------------------------------------

    async fn build_template_info(
        &self,
        conn: &turso::Connection,
        id: &str,
    ) -> Result<SpaceTemplateInfo, AppError> {
        let t = db::get_space_template(conn, id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space template not found: {id}"),
            })?;
        let participants = db::list_template_owned_participants(conn, id)
            .await?
            .into_iter()
            .filter(|p| p.kind == "agent")
            .map(TemplateParticipantInfo::from_owned)
            .collect();
        Ok(SpaceTemplateInfo {
            id: t.id,
            title: t.title,
            cascade_limit: t.cascade_limit,
            participants,
        })
    }

    async fn list_space_templates(&self) -> Result<Vec<SpaceTemplateInfo>, AppError> {
        let conn = self.db_conn().await?;
        let templates = db::list_space_templates(&conn).await?;
        let mut out = Vec::with_capacity(templates.len());
        for t in templates {
            out.push(self.build_template_info(&conn, &t.id).await?);
        }
        Ok(out)
    }

    /// Validate a template's participant inputs before any write (so a
    /// partial template can't be created). Returns the normalized tuples
    /// `(label, model_ref, system_prompt, notify_policy)`.
    fn validate_template_participants(
        participants: &[NewTemplateParticipant],
    ) -> Result<Vec<ValidatedTemplateParticipant>, AppError> {
        participants
            .iter()
            .map(|p| {
                let label = p.label.trim().to_string();
                if label.is_empty() {
                    return Err(AppError::Config {
                        message: "template participant label must not be empty".into(),
                    });
                }
                let policy = notify_policy_or_default(&p.notify_policy)?;
                let model_ref = p
                    .model_ref
                    .as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let system_prompt = p
                    .system_prompt
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                Ok((label, model_ref, system_prompt, policy))
            })
            .collect()
    }

    async fn create_template(
        &self,
        title: &str,
        cascade_limit: i64,
        participants: Vec<NewTemplateParticipant>,
    ) -> Result<SpaceTemplateInfo, AppError> {
        let title = title.trim();
        if title.is_empty() {
            return Err(AppError::Config {
                message: "template title must not be empty".into(),
            });
        }
        if cascade_limit < 1 {
            return Err(AppError::Config {
                message: "cascade limit must be at least 1".into(),
            });
        }
        let validated = Self::validate_template_participants(&participants)?;

        let conn = self.db_conn().await?;
        let now = now_ms();
        let id = Uuid::now_v7().to_string();
        db::insert_space_template(&conn, &id, title, cascade_limit, now).await?;
        for (label, model_ref, system_prompt, policy) in &validated {
            // Template agents are TEMPLATE-OWNED participant rows.
            db::insert_participant(
                &conn,
                &Uuid::now_v7().to_string(),
                "template",
                None,
                Some(&id),
                "agent",
                label,
                model_ref.as_deref(),
                system_prompt.as_deref(),
                policy,
                "member",
                None,
                now,
            )
            .await?;
        }
        self.bus.emit(Change::Templates);
        self.build_template_info(&conn, &id).await
    }

    async fn template_from_space(
        &self,
        space_id: &str,
        title: &str,
    ) -> Result<SpaceTemplateInfo, AppError> {
        let title = title.trim();
        if title.is_empty() {
            return Err(AppError::Config {
                message: "template title must not be empty".into(),
            });
        }
        let conn = self.db_conn().await?;
        let now = now_ms();
        let id = Uuid::now_v7().to_string();
        db::template_from_space(&conn, space_id, title, &id, now).await?;
        self.bus.emit(Change::Templates);
        self.build_template_info(&conn, &id).await
    }

    async fn update_template(
        &self,
        id: &str,
        title: Option<&str>,
        cascade_limit: Option<i64>,
        participants: Option<Vec<NewTemplateParticipant>>,
    ) -> Result<(), AppError> {
        let title = match title {
            Some(t) if t.trim().is_empty() => {
                return Err(AppError::Config {
                    message: "template title must not be empty".into(),
                });
            }
            Some(t) => Some(t.trim().to_string()),
            None => None,
        };
        if let Some(c) = cascade_limit
            && c < 1
        {
            return Err(AppError::Config {
                message: "cascade limit must be at least 1".into(),
            });
        }
        let validated = match &participants {
            Some(ps) => Some(Self::validate_template_participants(ps)?),
            None => None,
        };

        let conn = self.db_conn().await?;
        let live = db::get_space_template(&conn, id)
            .await?
            .filter(|t| t.removed_at.is_none());
        if live.is_none() {
            return Err(AppError::NotConfigured {
                message: format!("space template not found or removed: {id}"),
            });
        }
        let now = now_ms();
        // Settings-update + owned-participant replacement run in ONE transaction
        // (`db::update_template_tx`): a concurrent `instantiate_template` can
        // never observe a half-rebuilt agent set, and an insert error rolls the
        // whole thing back leaving the prior state intact — so the emit below
        // fires only after a committed change (changes.rs emit-after-commit).
        db::update_template_tx(
            &conn,
            id,
            title.as_deref(),
            cascade_limit,
            validated.as_deref(),
            now,
        )
        .await?;
        self.bus.emit(Change::Templates);
        Ok(())
    }

    async fn remove_template(&self, id: &str) -> Result<bool, AppError> {
        if id == config::DEFAULT_TEMPLATE_ID {
            return Err(AppError::Config {
                message: "the built-in Default template cannot be removed".into(),
            });
        }
        let conn = self.db_conn().await?;
        let removed = db::soft_remove_space_template(&conn, id, now_ms()).await?;
        if removed {
            self.bus.emit(Change::Templates);
        }
        Ok(removed)
    }

    async fn create_space(&self, title: Option<&str>) -> Result<SpaceInfo, AppError> {
        let db_conn = self.db_conn().await?;
        let now = now_ms();
        let space_id = self.instantiate_default_space(&db_conn, title, now).await?;

        self.bus.emit(Change::SpaceIndex);

        Ok(SpaceInfo {
            id: space_id,
            title: title.map(String::from),
            snippet: None,
            created_at: now,
            last_activity_at: now,
            message_count: 0,
            archived_at: None,
        })
    }

    /// Instantiate a **specific** template into a new space (the Space menu's
    /// "New Space from Template" path, and the Templates pane's "New space").
    /// Unlike [`Self::create_space`] (which resolves the *default* template),
    /// this takes an explicit template id; a missing/removed template is a
    /// typed error rather than a silent fallback.
    async fn create_space_from_template(
        &self,
        template_id: &str,
        title: Option<&str>,
    ) -> Result<SpaceInfo, AppError> {
        let conn = self.db_conn().await?;
        db::get_space_template(&conn, template_id)
            .await?
            .filter(|t| t.removed_at.is_none())
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space template not found or removed: {template_id}"),
            })?;
        let now = now_ms();
        let space_id = Uuid::now_v7().to_string();
        db::instantiate_template(&conn, template_id, &space_id, title, "unlinked", now).await?;
        self.bus.emit(Change::SpaceIndex);
        Ok(SpaceInfo {
            id: space_id,
            title: title.map(String::from),
            snippet: None,
            created_at: now,
            last_activity_at: now,
            message_count: 0,
            archived_at: None,
        })
    }

    async fn archive_space(&self, space_id: &str) -> Result<bool, AppError> {
        let db_conn = self.db_conn().await?;
        let archived = db::archive_space(&db_conn, space_id, now_ms()).await?;
        if archived {
            self.bus.emit(Change::SpaceIndex);
        }
        Ok(archived)
    }

    /// Save a thought as a `user_input` action **without requesting a
    /// response** — the save-vs-request split (wave 5). Posting needs no
    /// credential and no account; only `chat` / `request_response` spend.
    ///
    /// Creates the space when `space_id` is `None`, appends the user turn as a
    /// fresh gen-0 item with a `reply` edge to the prior tail, auto-titles a
    /// still-untitled space from its first post, and emits `Change::Space(id)`
    /// (content changed) plus `Change::SpaceIndex` when the library listing
    /// changed (new space or auto-title). Unlike `chat`, every write here is
    /// unconditional — there is no credential gate to fail behind, so the post
    /// always persists.
    /// Save a `user_input` post. `reply_to`, when set, is the action this post
    /// replies to (its structural thread parent) — a reply to a non-tail post
    /// **branches** the thread there; when `None` the post links to the space's
    /// current tail (the linear-continuation default).
    async fn post(
        &self,
        space_id: Option<&str>,
        prompt: &str,
        reply_to: Option<&str>,
    ) -> Result<PostResult, AppError> {
        let prompt = prompt.trim();
        if prompt.is_empty() {
            return Err(AppError::Internal {
                message: "cannot post an empty thought".into(),
            });
        }

        let db_conn = self.db_conn().await?;
        let now = now_ms();

        // The one shared human participant records every human post (seeded at
        // DB open, joined into every space by the template instantiation).
        let user_participant_id = db::HUMAN_PARTICIPANT_ID.to_string();

        // A `reply_to` branch antecedent must name an action in the SAME,
        // existing space — replying into a brand-new space is meaningless, and a
        // cross-space reply edge is not something the schema prevents. Validate
        // before any write so a bad `reply_to` leaves no trace.
        if let Some(rt) = reply_to {
            let sid = space_id.ok_or_else(|| AppError::NotConfigured {
                message: "reply_to requires an existing space".into(),
            })?;
            self.require_action_in_space(&db_conn, rt, sid).await?;
        }

        let (space_id, space_title, is_new_space) = if let Some(sid) = space_id {
            let row =
                db::get_space(&db_conn, sid)
                    .await?
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("space not found: {sid}"),
                    })?;
            (sid.to_string(), row.title, false)
        } else {
            // A new space is instantiated from the default template, so it has
            // its participants (You + the template agents) from birth.
            let sid = self.instantiate_default_space(&db_conn, None, now).await?;
            (sid, None, true)
        };

        // No prior terminal action ⇒ this is the space's first post: eligible
        // for auto-title, and there is no antecedent to link to.
        let last_action_id = db::last_action_in_space(&db_conn, &space_id).await?;

        let action_id = Uuid::now_v7().to_string();
        let item_id = Uuid::now_v7().to_string();
        db::insert_action(
            &db_conn,
            &db::ActionEntry {
                id: action_id.clone(),
                space_id: space_id.clone(),
                participant_id: user_participant_id,
                // The human "You" is a global participant.
                participant_scope: "global".to_string(),
                item_id: item_id.clone(),
                supersedes_action_id: None,
                action_type: "user_input".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at: now,
            },
        )
        .await?;
        db::insert_text_content_block(
            &db_conn,
            &Uuid::now_v7().to_string(),
            &action_id,
            0,
            "text",
            prompt,
        )
        .await?;

        let mut auto_titled = false;
        if space_title.is_none()
            && last_action_id.is_none()
            && let Some(title) = derive_space_title(prompt)
        {
            db::update_space_title(&db_conn, &space_id, &title).await?;
            auto_titled = true;
        }

        // Link the structural `reply` edge: to the explicit `reply_to` target
        // (branching there) when given, otherwise to the space's tail (linear
        // continuation). A first post in a space has neither.
        let reply_ante = reply_to
            .map(str::to_string)
            .or_else(|| last_action_id.clone());
        if let Some(ref ante_id) = reply_ante {
            db::insert_action_antecedent(&db_conn, &action_id, ante_id, 0, "reply").await?;
        }

        self.bus.emit(Change::Space(space_id.clone()));
        if is_new_space || auto_titled {
            self.bus.emit(Change::SpaceIndex);
        }

        Ok(PostResult {
            space_id,
            action_id,
            item_id,
            is_new_space,
            auto_titled,
        })
    }

    /// Edit an item by appending a new `user_input` generation — the human side
    /// of collaborative editing. Append-only: the prior generation is preserved
    /// and `item_current` now resolves to this one; the new generation
    /// replicates the item's structural `reply` edge so it keeps its place in
    /// the thread. No credential, no HTTP. `action_id` may be any generation of
    /// the target item; the new generation supersedes the item's current tip.
    async fn edit_post(&self, action_id: &str, new_prompt: &str) -> Result<PostResult, AppError> {
        let new_prompt = new_prompt.trim();
        if new_prompt.is_empty() {
            return Err(AppError::Internal {
                message: "cannot edit a post to empty".into(),
            });
        }

        let db_conn = self.db_conn().await?;
        let now = now_ms();

        let (item_id, space_id) = db::action_item_and_space(&db_conn, action_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("action not found: {action_id}"),
            })?;
        let tip = db::current_tip_of_item(&db_conn, &space_id, &item_id)
            .await?
            .ok_or_else(|| AppError::Internal {
                message: format!("item has no current generation: {item_id}"),
            })?;
        let reply_parent = db::reply_antecedent(&db_conn, &tip).await?;

        // Edits are the human's — recorded against the shared "You" participant.
        let user_participant_id = db::HUMAN_PARTICIPANT_ID.to_string();

        let new_action_id = Uuid::now_v7().to_string();
        db::insert_action(
            &db_conn,
            &db::ActionEntry {
                id: new_action_id.clone(),
                space_id: space_id.clone(),
                participant_id: user_participant_id,
                // The human "You" is a global participant.
                participant_scope: "global".to_string(),
                item_id: item_id.clone(),
                supersedes_action_id: Some(tip),
                action_type: "user_input".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at: now,
            },
        )
        .await?;
        db::insert_text_content_block(
            &db_conn,
            &Uuid::now_v7().to_string(),
            &new_action_id,
            0,
            "text",
            new_prompt,
        )
        .await?;
        if let Some(ref ante) = reply_parent {
            db::insert_action_antecedent(&db_conn, &new_action_id, ante, 0, "reply").await?;
        }

        // Content changed (Space), and the listing's snippet / last-activity may
        // change (SpaceIndex). message_count does NOT change — list_spaces counts
        // current generations, and an edit replaces one, it doesn't add an item.
        self.bus.emit(Change::Space(space_id.clone()));
        self.bus.emit(Change::SpaceIndex);

        Ok(PostResult {
            space_id,
            action_id: new_action_id,
            item_id,
            is_new_space: false,
            auto_titled: false,
        })
    }

    /// Find a credential that can cover `charge_credits`, auto-provisioning
    /// one from the account balance when none exists — the pooled body of the
    /// **ACT provisioning queue**.
    ///
    /// **The caller must hold [`Inner::spend_gate`]** across this call *and* the
    /// spend-proof + `insert_pre_credential_refund` that follow, so the whole
    /// acquire→spend→flip-to-`spending` step is atomic per wallet. Two
    /// concurrent turns therefore never both grab the same active credential:
    /// the first flips it to `spending` inside the lock; the second, entering
    /// next, no longer sees it as spendable.
    ///
    /// Resolution order (retried in a bounded loop):
    /// 1. An active local credential with enough credits → use it.
    /// 2. No usable credential and no account configured →
    ///    [`AppError::NoAccount`] (the UI routes to account creation).
    /// 3. Account exists and the available balance covers a fresh allocation →
    ///    allocate `min(available, max(DEFAULT_ALLOCATION_CREDITS, charge))`
    ///    (the pool) and retry.
    /// 4. Balance can't cover another allocation but a mid-spend credential
    ///    whose own `credits` cover the charge exists (a concurrent turn holds
    ///    usable coverage) → wait bounded for its refund recovery to write a
    ///    successor, then retry. A mid-spend credential *smaller* than the
    ///    charge is ignored — its refund can never yield a covering successor
    ///    (a successor is worth at most the original face value) and refunds
    ///    don't top up balance, so waiting on it is futile.
    /// 5. Bounded wait elapses with a covering credential still in flight →
    ///    [`AppError::ProvisioningTimeout`]; no covering credential in flight
    ///    (true shortfall) → [`AppError::InsufficientBalance`] *immediately*.
    async fn ensure_spendable_credential(
        &self,
        cfg: &Config,
        db_conn: &turso::Connection,
        charge_credits: i64,
    ) -> Result<db::SpendableCredential, AppError> {
        let deadline = tokio::time::Instant::now() + PROVISION_WAIT_TIMEOUT;
        loop {
            if let Some(cred) = db::find_spendable_credential(db_conn, charge_credits).await? {
                return Ok(cred);
            }

            if cfg.account_id.is_none() || cfg.account_secret.is_none() {
                return Err(AppError::NoAccount);
            }

            let balances = self.account_balances().await?;
            if balances.available >= charge_credits {
                // Enough to allocate at least the charge — grow the pool.
                let amount = auto_allocation_amount(balances.available, charge_credits)?;
                self.account_allocate(amount).await?;
                continue;
            }

            // Balance can't cover a fresh allocation. Waiting only helps if a
            // concurrent turn holds coverage this turn could actually use once
            // its refund lands. A refund recovers a *wallet successor* worth at
            // most the original credential's face value (a full refund on a
            // failed turn) — it never tops up the account balance, and a spend
            // consumes one credential (no combining balance + a credential, or
            // two credentials, to fund one turn). So an in-flight credential can
            // only become spendable for THIS turn if its own `credits` already
            // cover the charge; a smaller one — even fully refunded — never
            // will, and no allocation is possible either. Wait only when such a
            // credential exists; otherwise this is a true shortfall.
            let recoverable = db::list_spending_credentials(db_conn)
                .await?
                .into_iter()
                .any(|c| c.credits >= charge_credits);
            if recoverable && tokio::time::Instant::now() < deadline {
                tokio::time::sleep(PROVISION_POLL_INTERVAL).await;
                continue;
            }

            // Before any terminal verdict, re-check for a spendable credential.
            // A concurrent turn's in-flight credential can *settle*
            // (spending → spent, atomically writing an active successor) in the
            // window between the `find_spendable_credential` check at the top of
            // this iteration and the `list_spending_credentials` check above.
            // In that window both snapshots are stale — the successor is not yet
            // visible as active, and the original is no longer visible as
            // spending — so `recoverable` reads false even though funding just
            // became available. Re-reading here (the settle is a single atomic
            // insert, so a spent original guarantees a queryable active
            // successor) picks up that successor instead of falsely reporting a
            // shortfall.
            if let Some(cred) = db::find_spendable_credential(db_conn, charge_credits).await? {
                return Ok(cred);
            }
            if recoverable {
                return Err(AppError::ProvisioningTimeout {
                    message: format!(
                        "timed out waiting for an in-flight credential refund to free \
                         {charge_credits} credits for this turn"
                    ),
                });
            }
            return Err(AppError::InsufficientBalance {
                available: balances.available,
                required: charge_credits,
            });
        }
    }

    async fn rename_space(&self, space_id: &str, title: &str) -> Result<(), AppError> {
        let db_conn = self.db_conn().await?;
        db::get_space(&db_conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        db::update_space_title(&db_conn, space_id, title).await?;
        self.bus.emit(Change::SpaceIndex);
        Ok(())
    }

    /// Drive an agent response *turn* against an already-persisted space — the
    /// request side of the save-vs-request split. The user turn must already
    /// exist (via `post`); `run_turn` provisions a credential, assembles context
    /// from the space's current generations, calls the model, persists the
    /// inference + request rows, and reconciles the refund.
    ///
    /// `target_action_id` + `mode` decide how the response attaches:
    /// The shared preparation of a turn — everything before the HTTP request
    /// fires, identical for the blocking and streaming transports: attested
    /// client + model catalog, space/participant validation, the thread
    /// attach plan (`Reply` → a new child item replying to the target;
    /// `Revise` → a new generation of the target's item), upstream-context
    /// assembly, charge estimate + `budget` gate, credential provisioning +
    /// spend proof (the `Wallet` "spending" emission happens here), and the
    /// ACT auth header.
    async fn prepare_turn(
        &self,
        space_id: &str,
        selector: TurnSelector,
        target_action_id: &str,
        mode: ResponseMode,
        budget: Option<i64>,
    ) -> Result<TurnPrep, AppError> {
        let cfg = self.load_config();
        let now = now_ms();

        let db_conn = self.db_conn().await?;

        let attestation_log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>> =
            Arc::new(Mutex::new(Vec::new()));

        // Resolve the turn's participant. An explicit participant supplies its
        // effective model + system prompt directly; a bare model string is the
        // compatibility path resolved (or minted) against the space's agents
        // *after* the backend is known (below). The model string drives the
        // backend routing either way.
        let explicit_participant = match &selector {
            TurnSelector::Participant(pid) => Some(
                self.resolve_explicit_participant(&db_conn, space_id, pid)
                    .await?,
            ),
            TurnSelector::Model(_) => None,
        };
        let model: String = match (&selector, &explicit_participant) {
            (_, Some(row)) => row
                .model_ref
                .clone()
                .expect("resolve_explicit_participant guarantees a model"),
            (TurnSelector::Model(m), None) => m.clone(),
            (TurnSelector::Participant(_), None) => unreachable!("explicit resolves above"),
        };
        let model = model.as_str();

        // Resolve the selection to its backend, then route on the backend's
        // *kind*: `eidola` keeps the attested + credential-spend path;
        // engine-backed kinds (`local`, `llamacpp`) run against their
        // loopback llama.cpp engine; `openai` runs plain HTTPS with an
        // optional bearer key. Only the eidola kind carries pricing —
        // `remote_pricing` is `Some((prompt_rate, completion_rate,
        // scale_factor))` there, and its absence is what disables the whole
        // spend path below (no charge estimate, no credential, no refunds —
        // which also means non-eidola inference needs no account or
        // onboarding). `external_auth` is the non-spend Authorization header
        // (an openai backend's key).
        let mref = backends::parse_model_ref(model);
        let backend = self.require_backend(&db_conn, &mref.backend_id).await?;
        let backend_kind =
            backends::BackendKind::parse(&backend.kind).ok_or_else(|| AppError::Database {
                message: format!("unknown backend kind `{}`", backend.kind),
            })?;

        // The canonical selection string (recorded on actions, shown in
        // UIs) and the wire model (what the backend's HTTP API expects).
        // Engine-backed aliases are spawned equal to the canonical id, so
        // the two coincide there (as they do for eidola, where canonical is
        // the bare model); openai backends get the bare model.
        let canonical_model = backends::qualified_model_id(&mref.model, &backend.id);
        let wire_model = match backend_kind {
            BackendKind::OpenAi => mref.model.clone(),
            _ => canonical_model.clone(),
        };
        let model = canonical_model.as_str();

        let mut engine_lease: Option<local_models::EngineLease> = None;
        let (
            provider_id,
            client,
            base_url,
            connection_id,
            context_length,
            remote_pricing,
            external_auth,
        ) = match backend_kind {
            BackendKind::Local | BackendKind::LlamaCpp => {
                // A request *is* the load trigger: an unloaded engine is
                // loaded on demand (the eviction planner makes room by
                // unloading LRU idle, unpinned engines first) and a warming
                // one is awaited. The turn then holds a lease on the engine
                // — bumping the LRU clock and shielding it from auto-unload
                // until the turn ends (`TurnPrep` carries the lease; its
                // drop releases).
                let (engine_url, context_tokens, lease) =
                    match self.local.lease_engine(&backend.id, &mref.model) {
                        Some(leased) => leased,
                        None => {
                            // Auto-start gate: a `llamacpp` backend with
                            // auto-start disabled refuses request-triggered
                            // loads before any spawn — the engine must be
                            // pre-loaded explicitly. `local` always
                            // auto-starts (it's ours).
                            if backend_kind == BackendKind::LlamaCpp && !backend.auto_start {
                                return Err(AppError::NotConfigured {
                                    message: format!(
                                        "`{model}` is not loaded and backend `{}` has auto-start \
                                         disabled — load it explicitly (`eidola model load \
                                         {model}`) or enable auto-start",
                                        backend.id
                                    ),
                                });
                            }
                            self.load_local_model(model).await?;
                            self.local
                                .lease_engine(&backend.id, &mref.model)
                                .ok_or_else(|| AppError::LocalModel {
                                    message: format!(
                                        "`{model}` was unloaded while the request was starting"
                                    ),
                                })?
                        }
                    };
                engine_lease = Some(lease);
                let provider_id =
                    db::ensure_provider(&db_conn, &backend.id, "inference", now).await?;
                let client = match &self.http_override {
                    Some(c) => c.clone(),
                    None => local_models::plain_http_client()?,
                };
                (
                    provider_id,
                    client,
                    engine_url,
                    None,
                    context_tokens as u64,
                    None,
                    None,
                )
            }
            BackendKind::OpenAi => {
                let base_url = backend
                    .base_url
                    .clone()
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("backend `{}` has no base URL", backend.id),
                    })?;
                let provider_id =
                    db::ensure_provider(&db_conn, &backend.id, "inference", now).await?;
                let client = match &self.http_override {
                    Some(c) => c.clone(),
                    None => local_models::plain_http_client()?,
                };
                let auth = backend.api_key.as_ref().map(|k| format!("Bearer {k}"));
                // Context length is unknown for a generic server — 0
                // resolves to the 4096 completion default below.
                (provider_id, client, base_url, None, 0u64, None, auth)
            }
            BackendKind::Eidola => {
                // The eidola row was already fetched by `require_backend`
                // above — resolve its connection + trust bundle straight from
                // it (base URL / measurements / hardware CAs, falling back to
                // the embedded pin per column).
                let eidola = EidolaResolved::from_row(Some(&backend))?;
                let base_url = eidola.base_url.clone();
                let provider_id =
                    db::ensure_provider(&db_conn, &backend.id, "inference", now).await?;
                let log_clone = attestation_log.clone();
                let observer: Option<tinfoil_verifier::AttestationObserver> = Some(Arc::new(
                    move |att: tinfoil_verifier::VerifiedAttestation| {
                        log_clone.lock().unwrap().push(att);
                    },
                ));

                let client = self.build_client(&cfg, &eidola, observer).await?;

                let models = fetch_models(&client, &base_url).await?;
                let connection_id =
                    flush_attestations(&attestation_log, &db_conn, &provider_id, &base_url, now)
                        .await?;

                let model_entry =
                    models
                        .data
                        .iter()
                        .find(|m| m.id == wire_model)
                        .ok_or_else(|| AppError::NotConfigured {
                            message: format!("model not found: {model}"),
                        })?;
                let pricing = (
                    model_entry.pricing.per_prompt_token.value as u128,
                    model_entry.pricing.per_completion_token.value as u128,
                    model_entry.pricing.per_prompt_token.scale_factor as u128,
                );
                (
                    provider_id,
                    client,
                    base_url,
                    connection_id,
                    model_entry.context_length,
                    Some(pricing),
                    None,
                )
            }
        };

        let max_completion_tokens = if context_length == 0 {
            4096
        } else {
            context_length.min(4096) as u32
        };

        // The space already exists (post created it). Validate it first.
        db::get_space(&db_conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        // The target (the post being replied to / the generation being revised)
        // must belong to this space — otherwise a caller could splice another
        // space's thread into this turn (cross-space context + reply edge). This
        // covers both modes and every entry point (`respond_stream_as` /
        // `respond_stream`, and the same-space `chat` / `regenerate`). Wrapped
        // by the caller's `into_chat_failed` like the space-existence check.
        self.require_action_in_space(&db_conn, target_action_id, space_id)
            .await?;
        // Record the inference against the responding agent participant, and
        // capture its effective system prompt. An explicit participant supplies
        // all three directly; a bare model resolves (or mints) the space's
        // matching agent (`resolve_or_mint_agent_by_model`). The scope feeds the
        // action's pinned composite echo.
        let (model_participant_id, model_participant_scope, system_prompt) =
            match &explicit_participant {
                Some(row) => (
                    row.participant_id.clone(),
                    row.scope.clone(),
                    row.system_prompt.clone(),
                ),
                None => {
                    self.resolve_or_mint_agent_by_model(
                        &db_conn,
                        space_id,
                        model,
                        &provider_id,
                        now,
                    )
                    .await?
                }
            };
        let space_id = space_id.to_string();

        // The space is always persisted here, so every error exit carries its id
        // for blank-space adoption (a request failure leaves the saved post).
        let wrap = |source: AppError| AppError::ChatFailed {
            space_id: space_id.clone(),
            source: Box::new(source),
        };

        // Resolve how the inference attaches to the thread.
        let (inf_item_id, inf_supersedes, inf_reply_to) = match mode {
            ResponseMode::Reply => (
                Uuid::now_v7().to_string(),
                None,
                Some(target_action_id.to_string()),
            ),
            ResponseMode::Revise => {
                let (item_id, _sp) = db::action_item_and_space(&db_conn, target_action_id)
                    .await?
                    .ok_or_else(|| {
                        wrap(AppError::NotConfigured {
                            message: format!("target action not found: {target_action_id}"),
                        })
                    })?;
                let reply_to = db::reply_antecedent(&db_conn, target_action_id).await?;
                (item_id, Some(target_action_id.to_string()), reply_to)
            }
        };

        // Assemble the **upstream thread** of this turn, each item at its most
        // recent version (`get_upstream_context`): a Reply walks from the post
        // being answered *inclusive* — the whole conversation on a linear
        // thread, exactly the target's branch on a branched space (sibling
        // branches never leak into the context); a Revise (regenerate) walks
        // from the generation being replaced *exclusive*, so the model never
        // sees its own prior output or anything downstream of it.
        let context_rows: Vec<db::SpaceActionRow> = match mode {
            ResponseMode::Reply => {
                db::get_upstream_context(&db_conn, target_action_id, true).await?
            }
            ResponseMode::Revise => {
                db::get_upstream_context(&db_conn, target_action_id, false).await?
            }
        };
        let mut prior_messages = actions_to_messages(&context_rows);

        // Prepend the responding participant's effective system prompt as a
        // leading `system` message (Participants v1). It rides in the same
        // `messages` array the charge estimate and the wire request both use,
        // so the server recomputes the identical `chargeable_prompt_tokens`
        // over the identical array and the hold still covers the charge by
        // construction. It is deliberately NOT persisted as an action — the
        // forensics doctrine keeps mutable participant config out of the trail
        // (a later wave may snapshot its hash per turn).
        if let Some(sp) = system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            prior_messages.insert(
                0,
                SpaceMessage {
                    role: "system".to_string(),
                    content: sp.to_string(),
                },
            );
        }

        // The spend side runs only for eidola turns. Local and external
        // turns carry no charge estimate, no credential, and no ACT header
        // (an openai backend's bearer key rides in `external_auth` instead)
        // — which also means non-eidola inference needs no account or
        // onboarding.
        let (charge_credits, spend, auth_value) = match remote_pricing {
            None => (0u128, None, external_auth),
            Some((prompt_rate, completion_rate, sf)) => {
                // Estimate the charge from the assembled context. The prompt
                // side is the shared client/server pricing contract
                // (`eidola_common::chargeable_prompt_tokens`): a content-byte
                // term at the safe cost factor plus per-message and
                // per-request constants. The server computes the identical
                // function of the identical `messages` array as its
                // pre-flight minimum and clamps its charged prompt tokens to
                // it, so this hold covers the server's charge by
                // construction.
                let total_content_bytes: u64 =
                    prior_messages.iter().map(|m| m.content.len() as u64).sum();
                let chargeable_prompt = eidola_common::chargeable_prompt_tokens(
                    total_content_bytes,
                    prior_messages.len() as u64,
                );

                let prompt_credits = (chargeable_prompt as u128 * prompt_rate).div_ceil(sf);
                let completion_credits =
                    (max_completion_tokens as u128 * completion_rate).div_ceil(sf);
                let charge_credits = prompt_credits + completion_credits;

                if charge_credits == 0 {
                    return Err(wrap(AppError::Credential {
                        message: "computed charge is zero — model pricing may be missing".into(),
                    }));
                }

                // Spend budget ceiling for this turn.
                if let Some(b) = budget
                    && charge_credits as i64 > b
                {
                    return Err(wrap(AppError::Credential {
                        message: format!(
                            "estimated charge {charge_credits} exceeds the turn budget {b}"
                        ),
                    }));
                }

                // ACT provisioning queue: serialize acquire → spend-proof →
                // flip-to-`spending` across concurrent turns so two turns fired
                // at once can never both spend the same credential. The gate is
                // held only through `insert_pre_credential_refund` below (the
                // point the credential becomes `spending`); the HTTP request
                // runs after `prepare_turn` returns, outside the gate.
                let _spend_guard = self.spend_gate.lock().await;
                let cred = self
                    .ensure_spendable_credential(&cfg, &db_conn, charge_credits as i64)
                    .await
                    .map_err(wrap)?;

                let credit_token =
                    CreditToken::from_cbor(&cred.data).map_err(|e| AppError::Credential {
                        message: format!("failed to decode credential: {e}"),
                    })?;
                let public_key = PublicKey::from_cbor(&cred.public_key_data).map_err(|e| {
                    AppError::Credential {
                        message: format!("failed to decode public key: {e}"),
                    }
                })?;

                let params = params_from_domain_separator(cfg.domain_separator())?;

                let charge_scalar =
                    credit_to_scalar::<128>(charge_credits).map_err(|e| AppError::Credential {
                        message: format!("invalid charge amount: {e:?}"),
                    })?;
                let (spend_proof, pre_refund) = credit_token
                    .prove_spend::<128>(&params, charge_scalar, OsRng)
                    .map_err(|e| AppError::Credential {
                        message: format!("failed to create spend proof: {e:?}"),
                    })?;

                let pre_refund_cbor = pre_refund.to_cbor().map_err(|e| AppError::Credential {
                    message: format!("failed to encode pre_refund: {e}"),
                })?;
                let spend_proof_cbor = spend_proof.to_cbor().map_err(|e| AppError::Credential {
                    message: format!("failed to encode spend proof: {e}"),
                })?;
                let pre_cred_id = Uuid::now_v7().to_string();
                db::insert_pre_credential_refund(
                    &db_conn,
                    &pre_cred_id,
                    &cred.nonce,
                    &cred.issuer_key_id,
                    &pre_refund_cbor,
                    charge_credits as i64,
                    &spend_proof_cbor,
                    now,
                )
                .await?;
                // Credential flipped to "spending" — wallet state changed
                // regardless of whether the rest of the operation succeeds.
                self.bus.emit(Change::Wallet);

                let issuer_key_hash = hex_decode(&cred.issuer_key_id)?;
                let challenge_digest = compute_challenge_digest();

                let mut token_bytes = Vec::new();
                token_bytes.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
                token_bytes.extend_from_slice(&challenge_digest);
                token_bytes.extend_from_slice(&issuer_key_hash);
                token_bytes.extend_from_slice(&spend_proof_cbor);

                let token_b64 = URL_SAFE_NO_PAD.encode(&token_bytes);
                let auth_value = format!("PrivateToken token=\"{token_b64}\"");

                (
                    charge_credits,
                    Some(SpendPrep {
                        cred,
                        public_key,
                        params,
                        spend_proof,
                        pre_refund,
                        pre_cred_id,
                    }),
                    Some(auth_value),
                )
            }
        };

        // Build the messages array from the assembled context. The posted user
        // turn is already part of it (post persisted it); the agent's response
        // is appended as a new action at persist time.
        let messages: Vec<serde_json::Value> = prior_messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        Ok(TurnPrep {
            db_conn,
            provider_id,
            backend_id: backend.id,
            engine_lease,
            attestation_log,
            client,
            connection_id,
            base_url,
            now,
            space_id,
            model: model.to_string(),
            wire_model,
            model_participant_id,
            model_participant_scope,
            max_completion_tokens,
            inf_item_id,
            inf_supersedes,
            inf_reply_to,
            context_rows,
            messages,
            charge_credits,
            spend,
            auth_value,
        })
    }

    /// `Reply` → a new child item replying to the target; `Revise` → a new
    /// generation of the target's item (regenerate / agent edit). `budget`, if
    /// set, caps the estimated charge for this turn — the spend ceiling a future
    /// multi-inference agent loop will check per iteration.
    ///
    /// v1 drives a single inference, shaped as one turn so the tool loop slots
    /// in later as additional actions in the same chain. Preparation and
    /// persistence are shared with the streaming twin (`prepare_turn` /
    /// [`TurnPrep::persist_turn`]); this transport reads one JSON body and
    /// takes the inline-refund fast path.
    async fn run_turn(
        &self,
        space_id: &str,
        selector: TurnSelector,
        target_action_id: &str,
        mode: ResponseMode,
        budget: Option<i64>,
    ) -> Result<ChatResult, AppError> {
        // The space is already persisted (post created it) before prepare_turn
        // runs, so every setup failure inside it — client build, `/v1/models`
        // fetch, attestation flush, all *before* the turn's own `wrap` closure
        // — must still carry the space id for blank-space adoption / Retry.
        let mut prep = self
            .prepare_turn(space_id, selector, target_action_id, mode, budget)
            .await
            .map_err(|e| e.into_chat_failed(space_id))?;

        let request_body_json = prep.request_body(false);
        let request_at = now_ms();

        // Send the chat request. On failure, attempt refund recovery before
        // propagating the error so the credential isn't abandoned. Local
        // turns carry no Authorization header — there is nothing to spend.
        let mut request = prep
            .client
            .post(format!("{}/v1/chat/completions", prep.base_url))
            .json(&request_body_json);
        if let Some(auth_value) = &prep.auth_value {
            request = request.header("Authorization", auth_value);
        }
        let chat_result = request.send().await;
        let response_at = now_ms();

        // `post` already emitted the user turn's `Space(id)` + `SpaceIndex`
        // before this turn began. On a run_turn error exit we re-signal the
        // space so subscribers refresh (idempotent); `SpaceIndex` is not
        // re-emitted here — the listing changes (new space / auto-title) were
        // post's, and a failed request doesn't add an item. Call before any
        // error exit between here and the request-row insert.
        let space_for_emit = prep.space_id.clone();
        let emit_user_turn = || {
            self.bus.emit(Change::Space(space_for_emit.clone()));
        };

        let (status, response_text, body) = match chat_result {
            Ok(resp) => {
                prep.flush_new_attestations()
                    .await
                    .inspect_err(|_| emit_user_turn())
                    .map_err(|e| prep.wrap(e))?;

                let status = resp.status();
                let text = resp
                    .text()
                    .await
                    .map_err(|e| AppError::Network {
                        message: format!("failed to read response: {e}"),
                    })
                    .inspect_err(|_| emit_user_turn())
                    .map_err(|e| prep.wrap(e))?;
                let parsed: serde_json::Value = serde_json::from_str(&text)
                    .map_err(|e| AppError::Network {
                        message: format!("failed to parse response JSON: {e}"),
                    })
                    .inspect_err(|_| emit_user_turn())
                    .map_err(|e| prep.wrap(e))?;
                (status, text, parsed)
            }
            Err(e) => {
                // Network error — the server may or may not have received the
                // request. Try to recover the refund token; a written successor
                // credential is a wallet change.
                let original_err = AppError::from_request(e);
                if prep.try_refund_recovery().await {
                    self.bus.emit(Change::Wallet);
                }
                // The user turn (space row + user-message, auto-title) is
                // already committed — emit it so other windows see the persisted
                // turn, then wrap with the space id for blank-space adoption.
                emit_user_turn();
                return Err(prep.wrap(original_err));
            }
        };

        // Process the refund token from the response. If none is present,
        // attempt recovery from the server (best-effort — the final Wallet
        // emission below covers a written successor). Local turns have no
        // spend, hence nothing to refund.
        if prep.spend.is_some() {
            match body.get("refund") {
                Some(refund_obj) => {
                    prep.process_refund_obj(refund_obj)
                        .await
                        .inspect_err(|_| emit_user_turn())
                        .map_err(|e| prep.wrap(e))?;
                }
                None => {
                    let _ = prep.try_refund_recovery().await;
                }
            }
        }

        let usage = body.get("usage");
        let input_tokens = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_i64());
        let output_tokens = usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_i64());

        let response_content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let response_action_id = prep
            .persist_turn(
                if status.is_success() {
                    "complete"
                } else {
                    "error"
                },
                input_tokens,
                output_tokens,
                &response_content,
                &request_body_json,
                request_at,
                response_at,
                status.as_u16(),
                response_text.as_bytes().to_vec(),
            )
            .await?;

        if !status.is_success() {
            // Inference (error status) + request rows committed; Wallet was
            // emitted at spend start. post owns the user-turn SpaceIndex.
            self.bus.emit(Change::Space(prep.space_id.clone()));
            self.bus.emit(Change::Record);
            return Err(prep.wrap(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&response_text),
            }));
        }

        // All durable writes succeeded — emit per affected domain. post owns the
        // SpaceIndex (new space / auto-title); a response doesn't change the
        // listing's identity. Local turns touched no credential, so no
        // Wallet emission.
        self.bus.emit(Change::Space(prep.space_id.clone()));
        if prep.spend.is_some() {
            self.bus.emit(Change::Wallet);
        }
        self.bus.emit(Change::Record);

        Ok(ChatResult {
            space_id: prep.space_id,
            content: response_content,
            model: prep.model,
            input_tokens,
            output_tokens,
            credits_charged: prep.charge_credits as i64,
            response_action_id: Some(response_action_id),
        })
    }

    /// Save a turn and request a (blocking) response in one gesture — the
    /// combined convenience that the CLI and one-shot callers use. Equivalent
    /// to `post` followed by `run_turn(Reply)`: the post persists first (so a
    /// funding failure leaves the saved thought), then the agent replies.
    async fn chat(
        &self,
        prompt: &str,
        model: &str,
        space_id: Option<&str>,
    ) -> Result<ChatResult, AppError> {
        let posted = self.post(space_id, prompt, None).await?;
        self.run_turn(
            &posted.space_id,
            TurnSelector::Model(model.to_string()),
            &posted.action_id,
            ResponseMode::Reply,
            None,
        )
        .await
    }

    /// Regenerate an inference: append a new generation of its item (agent
    /// revise). `action_id` is any generation of the target item. Uses the
    /// model-picker compat path (`TurnSelector::Model`), so the same agent
    /// participant (matched by model) authors the new generation.
    async fn regenerate(&self, action_id: &str, model: &str) -> Result<ChatResult, AppError> {
        let db_conn = self.db_conn().await?;
        let (item_id, space_id) = db::action_item_and_space(&db_conn, action_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("action not found: {action_id}"),
            })?;
        let tip = db::current_tip_of_item(&db_conn, &space_id, &item_id)
            .await?
            .ok_or_else(|| AppError::Internal {
                message: format!("item has no current generation: {item_id}"),
            })?;
        drop(db_conn);
        self.run_turn(
            &space_id,
            TurnSelector::Model(model.to_string()),
            &tip,
            ResponseMode::Revise,
            None,
        )
        .await
    }

    /// Streaming counterpart to `run_turn` — same shared preparation and
    /// persistence (`prepare_turn` / [`TurnPrep::persist_turn`]), but sends
    /// `stream: true` upstream and forwards each SSE chunk to `sender` as it
    /// arrives.
    ///
    /// Reasoning shape: we accept both `delta.reasoning_content` (OpenAI-style
    /// extension used by some providers) and `delta.reasoning` (vLLM's
    /// extension). Either form is forwarded as `ReasoningDelta`. Unknown
    /// fields are ignored — if Tinfoil's upstream uses a third spelling, the
    /// thinking section will simply stay empty until we adapt.
    ///
    /// Refund handling differs from `run_turn` only in *where* the refund
    /// token comes from: SSE responses have no inline body to carry it, so we
    /// always go through the `/v1/credentials/refund` recovery endpoint
    /// after the stream ends. The credential is left in `pre_credential`
    /// state until that recovery completes, same as the network-error path.
    #[allow(clippy::too_many_arguments)]
    async fn run_turn_stream(
        &self,
        space_id: &str,
        selector: TurnSelector,
        target_action_id: &str,
        mode: ResponseMode,
        budget: Option<i64>,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        use futures_util::StreamExt;

        // Setup failures (client build / `/v1/models` fetch / attestation
        // flush) happen before the turn's inline `wrap` closure — carry the
        // already-persisted space id so they wrap like every later exit.
        let mut prep = self
            .prepare_turn(space_id, selector, target_action_id, mode, budget)
            .await
            .map_err(|e| e.into_chat_failed(space_id))?;

        // No `stream_options` here — the server unconditionally sets
        // `include_usage: true` when forwarding the streaming request
        // upstream, since accurate per-token refunds depend on it.
        // Sending it from the client is harmless (the server ignores
        // and overrides the value), but it's also unnecessary, so we
        // keep our outgoing request minimal.
        let request_body_json = prep.request_body(true);
        let request_at = now_ms();

        let mut request = prep
            .client
            .post(format!("{}/v1/chat/completions", prep.base_url))
            .header("Accept", "text/event-stream")
            .json(&request_body_json);
        if let Some(auth_value) = &prep.auth_value {
            request = request.header("Authorization", auth_value);
        }
        let chat_result = request.send().await;

        // post already emitted the user turn's Space(id) + SpaceIndex; on a
        // run_turn_stream error exit we re-signal the space (idempotent
        // refresh). SpaceIndex is post's concern, not re-emitted here.
        let space_for_emit = prep.space_id.clone();
        let emit_user_turn = || {
            self.bus.emit(Change::Space(space_for_emit.clone()));
        };

        let resp = match chat_result {
            Ok(resp) => {
                prep.flush_new_attestations()
                    .await
                    .inspect_err(|_| emit_user_turn())
                    .map_err(|e| prep.wrap(e))?;
                resp
            }
            Err(e) => {
                let original_err = AppError::from_request(e);
                if prep.try_refund_recovery().await {
                    // Successor credential written — wallet state updated.
                    self.bus.emit(Change::Wallet);
                }
                // User turn is committed — emit it, then wrap with the space id.
                emit_user_turn();
                return Err(prep.wrap(original_err));
            }
        };

        let status = resp.status();

        // Non-2xx: server returned an error body (typically JSON, not SSE).
        // Read it normally so we can surface a useful message. (Unlike the
        // blocking twin there is no inference action to attach — the stream
        // never produced one — so the request row stands alone.)
        if !status.is_success() {
            let response_text = resp.text().await.unwrap_or_default();
            if prep.try_refund_recovery().await {
                // Successor credential written — wallet state updated.
                self.bus.emit(Change::Wallet);
            }
            db::insert_request(
                &prep.db_conn,
                &db::Request {
                    id: Uuid::now_v7().to_string(),
                    connection_id: prep.connection_id.clone(),
                    action_id: None,
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    request_headers: None,
                    request_body: Some(request_body_json.to_string().into_bytes()),
                    response_status: Some(status.as_u16() as i64),
                    response_headers: None,
                    response_body: Some(response_text.as_bytes().to_vec()),
                    request_at,
                    response_at: Some(now_ms()),
                    duration_ms: Some(now_ms() - request_at),
                    error: None,
                    credential_nonce: prep.credential_nonce(),
                    created_at: now_ms(),
                    backend_id: Some(prep.backend_id.clone()),
                },
            )
            .await?;
            // Request row committed; Wallet was emitted at spend start (and
            // again if refund recovery succeeded). post owns the user-turn
            // SpaceIndex.
            self.bus.emit(Change::Space(prep.space_id.clone()));
            self.bus.emit(Change::Record);
            return Err(prep.wrap(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&response_text),
            }));
        }

        // Consume the SSE body. We accumulate bytes in a small buffer and
        // split on the SSE event boundary `\n\n`. Each event is a sequence
        // of `field: value\n` lines; we only care about `data:` lines (the
        // chunk JSON) and the sentinel `[DONE]`.
        let mut byte_stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut full_content = String::new();
        let mut full_reasoning = String::new();
        let mut input_tokens: Option<i64> = None;
        let mut output_tokens: Option<i64> = None;
        let mut response_buf: Vec<u8> = Vec::new();
        let mut finished = false;

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk
                .map_err(|e| AppError::Network {
                    message: format!("stream read failed: {e}"),
                })
                // Mid-stream read failure: the user turn is committed (the
                // request row is not yet) — emit the user turn so other windows
                // see it, then wrap with the space id for blank-space adoption.
                .inspect_err(|_| emit_user_turn())
                .map_err(|e| prep.wrap(e))?;
            // Keep the raw bytes for the request log so we can debug
            // upstream behaviour the same way as the non-streaming path.
            response_buf.extend_from_slice(&bytes);
            buf.extend_from_slice(&bytes);

            while let Some(pos) = find_event_boundary(&buf) {
                let event_bytes = buf.drain(..pos).collect::<Vec<u8>>();
                // Drop the boundary itself (\n\n or \r\n\r\n).
                let boundary_len = if buf.starts_with(b"\r\n\r\n") { 4 } else { 2 };
                if buf.len() >= boundary_len {
                    buf.drain(..boundary_len);
                }
                let event_str = match std::str::from_utf8(&event_bytes) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                for line in event_str.lines() {
                    let line = line.trim_end_matches('\r');
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" {
                        finished = true;
                        continue;
                    }
                    let json: serde_json::Value = match serde_json::from_str(payload) {
                        Ok(v) => v,
                        Err(_) => continue, // ignore comments/heartbeats
                    };

                    if let Some(usage) = json.get("usage") {
                        if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_i64()) {
                            input_tokens = Some(v);
                        }
                        if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_i64()) {
                            output_tokens = Some(v);
                        }
                    }

                    let Some(delta) = json
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|c| c.get("delta"))
                    else {
                        continue;
                    };

                    if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        full_content.push_str(text);
                        let _ = sender.send(ChatStreamEvent::ContentDelta(text.to_string()));
                    }

                    // OpenAI o1-style ("reasoning_content") and vLLM-style
                    // ("reasoning"). Either form is forwarded as a
                    // ReasoningDelta; we tolerate providers that emit one,
                    // both, or neither.
                    for key in ["reasoning_content", "reasoning"] {
                        if let Some(text) = delta.get(key).and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            full_reasoning.push_str(text);
                            let _ = sender.send(ChatStreamEvent::ReasoningDelta(text.to_string()));
                        }
                    }
                }
            }

            if finished {
                break;
            }
        }
        let response_at = now_ms();

        // SSE carries no inline refund — always consult the recovery endpoint
        // (best-effort; the final Wallet emission below covers a successor).
        let _ = prep.try_refund_recovery().await;

        let response_action_id = prep
            .persist_turn(
                "complete",
                input_tokens,
                output_tokens,
                &full_content,
                &request_body_json,
                request_at,
                response_at,
                status.as_u16(),
                response_buf,
            )
            .await?;

        // All durable writes succeeded — emit per affected domain. post owns the
        // SpaceIndex (new space / auto-title). Local turns touched no
        // credential, so no Wallet emission.
        self.bus.emit(Change::Space(prep.space_id.clone()));
        if prep.spend.is_some() {
            self.bus.emit(Change::Wallet);
        }
        self.bus.emit(Change::Record);

        Ok(ChatResult {
            space_id: prep.space_id,
            content: full_content,
            model: prep.model,
            input_tokens,
            output_tokens,
            credits_charged: prep.charge_credits as i64,
            response_action_id: Some(response_action_id),
        })
    }

    /// Save a turn and request a streaming response in one gesture (the GUI's
    /// path). Equivalent to `post` followed by `run_turn_stream(Reply)`.
    async fn chat_stream(
        &self,
        prompt: &str,
        model: &str,
        space_id: Option<&str>,
        reply_to: Option<&str>,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        let posted = self.post(space_id, prompt, reply_to).await?;
        self.run_turn_stream(
            &posted.space_id,
            TurnSelector::Model(model.to_string()),
            &posted.action_id,
            ResponseMode::Reply,
            None,
            sender,
        )
        .await
    }

    // --- submit + notification planning (Participants v1) ----------------

    /// Compute the auto-response notification plan for a post over the space's
    /// participants (owned ∪ referenced globals, effective config). Applies the
    /// data-derived cascade guard first: if the post's derived cascade depth has
    /// reached the space's `cascade_limit`, returns [`NotificationPlan::Paused`]
    /// instead of turns. Otherwise the notify set is every agent member (except
    /// the post's author, and skipping model-less agents) whose `notify_policy`
    /// fires: `all` → always; `human` → only when the post's author is human;
    /// `explicit` → never (only an explicit ask reaches them).
    async fn plan_notifications(
        &self,
        space_id: &str,
        post_action_id: &str,
    ) -> Result<NotificationPlan, AppError> {
        let conn = self.db_conn().await?;

        // The post must belong to this space: the notify set + cascade limit come
        // from `space_id`, but depth + authorship come from `post_action_id`, so
        // a mismatched pair would plan this space's participants against another
        // space's post (and driving one would send that post's ancestry to this
        // space's agent, persisting a cross-space reply edge). Reject up front.
        self.require_action_in_space(&conn, post_action_id, space_id)
            .await?;

        let limit = db::space_cascade_limit(&conn, space_id).await?.unwrap_or(4);
        let depth = db::agent_cascade_depth(&conn, post_action_id).await?;
        if depth >= limit {
            return Ok(NotificationPlan::Paused { depth, limit });
        }

        // The post's author (excluded from the notify set; its kind resolves the
        // `human` predicate).
        let (author_id, author_kind) = match db::action_author(&conn, post_action_id).await? {
            Some((id, _scope, kind)) => (id, kind),
            None => return Ok(NotificationPlan::Turns(Vec::new())),
        };

        let mut turns = Vec::new();
        for m in db::space_participants(&conn, space_id).await? {
            if m.kind != "agent" || m.participant_id == author_id {
                continue;
            }
            // An agent with no model can't respond — never plan a turn for it.
            if m.model_ref
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .is_none()
            {
                continue;
            }
            let fires = match m.notify_policy.as_str() {
                "all" => true,
                "human" => author_kind == "human",
                _ => false, // explicit
            };
            if fires {
                turns.push(PlannedTurn {
                    participant_id: m.participant_id,
                    target_action_id: post_action_id.to_string(),
                    cascade_depth: depth + 1,
                });
            }
        }
        Ok(NotificationPlan::Turns(turns))
    }

    /// The composer CTA path: save a post (`post`) then plan notifications over
    /// it. The caller drives one turn per [`PlannedTurn`] (via
    /// [`AppCore::respond_stream_as`]) and may re-plan on each resulting post to
    /// continue an auto-notify cascade until the guard pauses it.
    async fn submit(
        &self,
        space_id: Option<&str>,
        text: &str,
        reply_to: Option<&str>,
    ) -> Result<SubmitResult, AppError> {
        let post = self.post(space_id, text, reply_to).await?;
        let plan = self
            .plan_notifications(&post.space_id, &post.action_id)
            .await?;
        Ok(SubmitResult { post, plan })
    }
}

/// Find the byte offset of the next SSE event boundary (`\n\n` or
/// `\r\n\r\n`) in `buf`, if any. Returns the position *before* the boundary
/// — i.e. the length of the next event's body.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len() {
        if buf[i..].starts_with(b"\r\n\r\n") {
            return Some(i);
        }
        if buf[i..].starts_with(b"\n\n") {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// AppCore — owns the tokio runtime that drives all async work (turso,
// reqwest, tokio primitives). Consumers (CLI, GUI) hold an `Arc<AppCore>`
// and call methods directly.
// ============================================================================

pub struct AppCore {
    runtime: tokio::runtime::Runtime,
    inner: Arc<Inner>,
    /// The invalidation bus, shared with `Inner` so callers can subscribe
    /// while `Inner` emits on the tokio runtime.
    bus: BroadcastSource,
}

impl AppCore {
    pub fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// Streaming chat. Pushes incremental `ChatStreamEvent`s through
    /// `sender` and returns the finalized `ChatResult` when the upstream
    /// stream closes. Drops `sender` on return so receivers see channel
    /// closure as the natural "done" signal.
    pub async fn chat_stream(
        &self,
        prompt: String,
        model: String,
        space_id: Option<String>,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .chat_stream(&prompt, &model, space_id.as_deref(), None, sender)
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Streaming chat that **replies to a specific post** — `reply_to` is the
    /// action the new turn replies to, branching the thread there (vs the
    /// linear tail-continuation of [`chat_stream`]).
    pub async fn chat_stream_reply(
        &self,
        prompt: String,
        model: String,
        space_id: Option<String>,
        reply_to: Option<String>,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .chat_stream(
                        &prompt,
                        &model,
                        space_id.as_deref(),
                        reply_to.as_deref(),
                        sender,
                    )
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Request a **streaming response to an already-persisted post**, without
    /// posting a new user turn. This is the "re-request" / retry entry point:
    /// after a failed ask the user's post is saved but has no reply, so there
    /// is nothing to re-post — this runs a fresh turn *replying to that exact
    /// post*. `target_action_id` is the post being answered (its current tip);
    /// the turn attaches as its reply in `ResponseMode::Reply`.
    ///
    /// This is exactly the `run_turn_stream(Reply)` half of [`Self::chat_stream`]
    /// without the leading `post`, so every exit point and emission is identical
    /// to a `chat_stream` that reused an existing post (see `tests/bus.rs`). A
    /// failure is wrapped as `AppError::ChatFailed { space_id }` so a GUI space
    /// can route it the same way it routes a failed ask.
    pub async fn respond_stream(
        &self,
        space_id: String,
        model: String,
        target_action_id: String,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .run_turn_stream(
                        &space_id,
                        TurnSelector::Model(model),
                        &target_action_id,
                        ResponseMode::Reply,
                        None,
                        sender,
                    )
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Drive a **planned auto-response turn** (participant-aware) — the wave-2
    /// composer path. Streams a response to `target_action_id` **as** the given
    /// `participant_id` (its effective model + system prompt), without posting a
    /// new user turn. This is what a caller runs for each [`PlannedTurn`] a
    /// [`Self::submit`] returned. Like [`Self::respond_stream`] it is an
    /// explicit ask that bypasses the cascade guard (the guard lives in
    /// [`Self::plan_notifications`]); failures wrap the space id the same way.
    pub async fn respond_stream_as(
        &self,
        space_id: String,
        participant_id: String,
        target_action_id: String,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .run_turn_stream(
                        &space_id,
                        TurnSelector::Participant(participant_id),
                        &target_action_id,
                        ResponseMode::Reply,
                        None,
                        sender,
                    )
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Save a post and plan notifications over the space's participants (the
    /// composer CTA path). Returns the saved post plus a [`NotificationPlan`];
    /// the caller drives one [`Self::respond_stream_as`] per planned turn (and
    /// may re-plan on each resulting post via [`Self::plan_notifications`] to
    /// continue a cascade until the guard pauses it).
    pub async fn submit(
        &self,
        text: String,
        space_id: Option<String>,
        reply_to: Option<String>,
    ) -> Result<SubmitResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .submit(space_id.as_deref(), &text, reply_to.as_deref())
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Compute the auto-response notification plan for an already-persisted post
    /// (owned ∪ referenced participants, effective notify policy, data-derived
    /// cascade guard). Used to continue a cascade after each driven turn.
    pub async fn plan_notifications(
        &self,
        space_id: String,
        post_action_id: String,
    ) -> Result<NotificationPlan, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.plan_notifications(&space_id, &post_action_id).await })
            .await
            .map_err(join_err)?
    }
}

/// Convert a `tokio::task::JoinError` (panic / cancellation) into `AppError`.
fn join_err(e: tokio::task::JoinError) -> AppError {
    AppError::Internal {
        message: format!("async task failed: {e}"),
    }
}

impl AppCore {
    /// Create a new core instance.
    ///
    /// `config_dir` — directory containing `config.toml`.
    /// `data_dir` — directory for the local database.
    pub fn new(config_dir: PathBuf, data_dir: PathBuf) -> Self {
        Self::build(config_dir, data_dir, None)
    }

    /// Construct an `AppCore` whose HTTP client is the supplied plain
    /// `reqwest::Client` instead of the attesting client.
    ///
    /// **Test-only seam.** This bypasses per-handshake enclave attestation so
    /// integration tests can exercise `chat` / `chat_stream` (and the account /
    /// credential HTTP paths) against an in-process mock upstream over plain
    /// HTTP. It has no production use — `#[doc(hidden)]` keeps it out of the
    /// rendered API — and `AppCore::new` always passes `None`, so the
    /// production attestation path is untouched. Tests must still point
    /// `base_url` at the mock (via [`AppCore::set_base_url`]); the injected
    /// client only governs *how* requests are made, not *where*.
    #[doc(hidden)]
    pub fn with_test_http_client(
        config_dir: PathBuf,
        data_dir: PathBuf,
        client: reqwest::Client,
    ) -> Self {
        Self::build(config_dir, data_dir, Some(client))
    }

    fn build(
        config_dir: PathBuf,
        data_dir: PathBuf,
        http_override: Option<reqwest::Client>,
    ) -> Self {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(8 * 1024 * 1024) // 8 MB — matches default main-thread size
            .build()
            .expect("failed to create tokio runtime");
        let bus = BroadcastSource::new();
        Self {
            runtime,
            bus: bus.clone(),
            inner: Arc::new(Inner {
                config_path: config_dir.join("config.toml"),
                data_dir,
                db: tokio::sync::OnceCell::new(),
                update_state: Mutex::new(None),
                update_polling: std::sync::atomic::AtomicBool::new(false),
                bus,
                local: Arc::new(local_models::LocalRuntime::default()),
                spend_gate: tokio::sync::Mutex::new(()),
                http_override,
            }),
        }
    }

    /// Subscribe to the invalidation bus.
    ///
    /// Returns a [`tokio::sync::broadcast::Receiver`] that receives a
    /// [`changes::Change`] after every successful durable write in this
    /// `AppCore` instance.  The receiver is independent — dropping it does
    /// not affect the bus or other subscribers.
    ///
    /// ## Lagged receivers
    ///
    /// If a receiver falls more than [`changes::BUS_CAPACITY`] messages behind
    /// it will receive [`tokio::sync::broadcast::error::RecvError::Lagged`].
    /// Treat that as "refresh everything you care about" — at least that many
    /// changes were missed.
    pub fn subscribe_changes(&self) -> tokio::sync::broadcast::Receiver<changes::Change> {
        self.bus.subscribe()
    }

    // -----------------------------------------------------------------------
    // Config — sync methods (no runtime needed, delegate directly)
    // -----------------------------------------------------------------------

    /// A synchronous, purely config-backed snapshot (no DB, no runtime
    /// re-entry) — safe to call from any thread, **including from inside the
    /// core runtime** (the CLI calls it within `runtime().block_on(run(..))`).
    /// The transitional resolved default model is **not** here (it reads the
    /// DB); callers that need it use the async [`AppCore::default_model`].
    pub fn config_state(&self) -> ConfigState {
        let cfg = self.inner.load_config();
        ConfigState {
            default_template: cfg.default_template().to_string(),
            has_account: cfg.account_id.is_some(),
            has_account_secret: cfg.account_secret.is_some(),
            domain_separator: cfg.domain_separator().to_string(),
            attestation_url: cfg.attestation_url.clone(),
            appearance: cfg.appearance(),
            time_of_day_tint: cfg.time_of_day_tint(),
            light_character: cfg.light_character(),
            font_scale: cfg.font_scale(),
        }
    }

    /// The transitional resolved default inference model (see the
    /// `default_model` note in the docs): the default template's first agent
    /// participant's `model_ref`, falling back to [`config::DEFAULT_MODEL`].
    /// Async because it reads the DB — this is the panic-free replacement for
    /// the old DB-in-`config_state` resolution, consistent with the rest of
    /// AppCore's async surface. UIs cache the value and refresh it on
    /// `Change::Config` / `Change::Templates`.
    pub async fn default_model(&self) -> Result<String, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.resolve_default_model().await })
            .await
            .map_err(join_err)?
    }

    /// The eidola backend's resolved connection + trust bundle (base URL,
    /// measurements, hardware CAs) with the override-vs-pin honesty flags —
    /// read from the `eidola` backend row. Async because it reads the DB;
    /// UIs cache the snapshot and refresh it on [`Change::Backends`].
    pub async fn eidola_trust(&self) -> Result<EidolaTrust, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.eidola_trust().await })
            .await
            .map_err(join_err)?
    }

    /// Set the eidola server URL override (persisted on the `eidola` backend
    /// row). Validated before write; emits [`Change::Backends`].
    pub async fn set_base_url(&self, url: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .update_backend(
                        backends::EIDOLA_BACKEND_ID,
                        backends::BackendUpdate {
                            base_url: Some(Some(url)),
                            ..Default::default()
                        },
                    )
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Persist the default space template new spaces are instantiated from
    /// (the `default_template` config key). Replaces the removed
    /// `set_default_model`. Emits [`Change::Config`] **and**
    /// [`Change::Templates`] (the template registry's default marker moved).
    /// The id should name a live template; a stale id is tolerated (space
    /// creation falls back to the seeded default), so this only rejects blanks.
    pub fn set_default_template(&self, template_id: String) -> Result<(), AppError> {
        let template_id = template_id.trim().to_string();
        if template_id.is_empty() {
            return Err(AppError::Config {
                message: "default template id must not be empty".into(),
            });
        }
        let mut cfg = self.inner.load_config();
        cfg.default_template_override = Some(template_id);
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        self.bus.emit(Change::Templates);
        Ok(())
    }

    /// Persist the circadian day/night axis (the `appearance` config
    /// override): `system` tracks the OS appearance, `day`/`night` pin one
    /// palette family, `auto` switches on the system clock.
    pub fn set_appearance(&self, appearance: config::AppearanceSetting) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.appearance_override = Some(appearance);
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    /// Persist the circadian time-of-day axis (the `time_of_day_tint`
    /// config override): whether the palette takes on the character of the
    /// light at the current hour, or stays on the neutral palettes.
    pub fn set_time_of_day_tint(&self, tint: config::TimeOfDayTint) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.time_of_day_tint_override = Some(tint);
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    /// Persist the fixed light character (the `light_character` config
    /// override) the palettes render under while the time-of-day axis is
    /// `off`.
    pub fn set_light_character(&self, character: config::LightCharacter) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.light_character_override = Some(character);
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    /// Persist the base type-scale factor (the `font_scale` config override):
    /// the single multiplier the GUI applies over the whole type ramp. The
    /// value is clamped into `[FONT_SCALE_MIN, FONT_SCALE_MAX]` before writing,
    /// so callers can hand in a raw ladder step (or a stepped value) without
    /// pre-validating. Emits [`Change::Config`] so every window's theme
    /// re-applies.
    pub fn set_font_scale(&self, scale: f32) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.font_scale_override = Some(config::clamp_font_scale(scale));
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    /// Set (or clear, with `None`/blank) the explicit `llama-server` path
    /// used for local inference. When unset, the binary is discovered on
    /// `PATH` and in the usual install locations.
    pub fn set_llama_server_path(&self, path: Option<String>) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.llama_server_path_override = path.filter(|p| !p.trim().is_empty());
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    /// Remove the eidola base-URL override (clears the row column back to
    /// NULL), reverting to the trust-root pin baked into this binary. Emits
    /// [`Change::Backends`].
    pub async fn clear_base_url_override(&self) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .update_backend(
                        backends::EIDOLA_BACKEND_ID,
                        backends::BackendUpdate {
                            base_url: Some(None),
                            ..Default::default()
                        },
                    )
                    .await
            })
            .await
            .map_err(join_err)?
    }

    pub fn set_attestation_url(&self, url: String) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.attestation_url = Some(url);
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    /// Set the eidola hardware root CA (ARK) PEM override on the backend row.
    /// Validated before write; emits [`Change::Backends`].
    pub async fn set_hardware_root_ca(&self, pem: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.set_hardware_ca(HardwareCa::Root, Some(pem)).await })
            .await
            .map_err(join_err)?
    }

    /// Set the eidola hardware intermediate CA (ASK) PEM override on the
    /// backend row. Validated before write; emits [`Change::Backends`].
    pub async fn set_hardware_intermediate_ca(&self, pem: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .set_hardware_ca(HardwareCa::Intermediate, Some(pem))
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Remove the eidola hardware root CA (ARK) override (row column back to
    /// NULL), reverting to the production AMD chain. Emits
    /// [`Change::Backends`].
    pub async fn clear_hardware_root_ca(&self) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.set_hardware_ca(HardwareCa::Root, None).await })
            .await
            .map_err(join_err)?
    }

    /// Remove the eidola hardware intermediate CA (ASK) override (row column
    /// back to NULL), reverting to the production AMD chain. Emits
    /// [`Change::Backends`].
    pub async fn clear_hardware_intermediate_ca(&self) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.set_hardware_ca(HardwareCa::Intermediate, None).await })
            .await
            .map_err(join_err)?
    }

    /// Add a trusted enclave measurement to the eidola backend row's override
    /// list. Returns whether it was newly added (idempotent by SNP
    /// measurement). Emits [`Change::Backends`] when it writes.
    pub async fn trust_measurement(
        &self,
        snp: String,
        tdx_rtmr1: String,
        tdx_rtmr2: String,
    ) -> Result<bool, AppError> {
        let spec = format!("{snp}:{tdx_rtmr1}:{tdx_rtmr2}");
        let entry = config::parse_trust_measurement(&spec)?;
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.trust_measurement(entry).await })
            .await
            .map_err(join_err)?
    }

    /// Remove a trusted measurement (by SNP key) from the eidola backend
    /// row's override list. Returns whether one was removed; clearing the
    /// last reverts to the pin. Emits [`Change::Backends`] when it writes.
    pub async fn untrust_measurement(&self, snp: String) -> Result<bool, AppError> {
        let key = config::parse_untrust_key(&snp)?;
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.untrust_measurement(key).await })
            .await
            .map_err(join_err)?
    }

    pub fn set_account_credentials(&self, id: String, secret: String) -> Result<(), AppError> {
        let cfg = self.inner.load_config();
        if cfg.account_id.is_some() || cfg.account_secret.is_some() {
            return Err(AppError::Config {
                message: "account credentials already configured — reset first".into(),
            });
        }
        let mut cfg = cfg;
        cfg.account_id = Some(id);
        cfg.account_secret = Some(secret);
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    pub fn reset_account(&self) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.account_id = None;
        cfg.account_secret = None;
        cfg.save_to(&self.inner.config_path)?;
        self.bus.emit(Change::Config);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Updates — verified update-notification flow (see `updates` module)
    // -----------------------------------------------------------------------

    /// Run one update check now: fetch the GitHub release marked `latest`
    /// from the resolved feed, verify its `artifact-manifest.json` Sigstore
    /// bundle against the embedded trust root, compare attested claims, and
    /// persist the outcome. Infallible — every failure mode is a typed
    /// [`updates::UpdateCheckResult`] variant.
    pub async fn update_check(&self) -> updates::UpdateCheckSnapshot {
        let inner = self.inner.clone();
        match self
            .runtime
            .spawn(async move { inner.run_update_check().await })
            .await
        {
            Ok(snapshot) => snapshot,
            Err(e) => updates::UpdateCheckSnapshot {
                checked_at_ms: now_ms(),
                result: updates::UpdateCheckResult::CheckFailed {
                    message: format!("update check task failed: {e}"),
                },
            },
        }
    }

    /// The persisted outcome of the most recent completed check, if any.
    /// Reflects background polls as well as manual checks.
    pub fn last_update_check(&self) -> Option<updates::UpdateCheckSnapshot> {
        self.inner.update_state_snapshot().last
    }

    /// Record the user's explicit "treat as update" decision for a
    /// claims-changed release (matched by version + manifest hash). The
    /// persisted last result is rewritten so UIs immediately see the
    /// release as an accepted update.
    pub fn accept_changed_claims(
        &self,
        version: String,
        manifest_sha256: String,
    ) -> Result<(), AppError> {
        self.inner.accept_changed_claims(version, manifest_sha256)
    }

    /// Start the background poll loop: one check immediately, then one
    /// every [`updates::POLL_INTERVAL`] (~6h) while the process runs.
    /// Idempotent — at most one loop is ever spawned.
    pub fn start_update_polling(&self) {
        if self
            .inner
            .update_polling
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        let inner = self.inner.clone();
        self.runtime.spawn(async move {
            loop {
                let _ = inner.run_update_check().await;
                tokio::time::sleep(updates::POLL_INTERVAL).await;
            }
        });
    }

    // -----------------------------------------------------------------------
    // Async methods — spawn onto the owned tokio runtime
    // -----------------------------------------------------------------------

    pub async fn account_show(&self) -> Result<AccountShowResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_show().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_create(&self) -> Result<AccountCreateResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_create().await })
            .await
            .map_err(join_err)?
    }

    /// The documents (terms of service, privacy policy) whose current
    /// versions the server requires accounts to accept. Empty when the
    /// server has no acceptance gate configured.
    pub async fn current_terms(&self) -> Result<Vec<TermsDocument>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.current_terms().await })
            .await
            .map_err(join_err)?
    }

    /// Record the user's acceptance of every currently required document
    /// version, returning what was accepted. Callers must have presented
    /// the documents to the user first — this transmits consent, it does
    /// not obtain it. Routes here after [`AppError::TermsAcceptanceRequired`].
    pub async fn accept_current_terms(&self) -> Result<Vec<TermsDocument>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.accept_current_terms().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_prices(&self) -> Result<Vec<PriceInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_prices().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_checkout(&self, price_id: String) -> Result<String, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_checkout(&price_id).await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_balances(&self) -> Result<BalancesResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_balances().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_allocate(&self, credits: i64) -> Result<AllocateResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_allocate(credits).await })
            .await
            .map_err(join_err)?
    }

    pub async fn wallet_credentials(&self) -> Result<Vec<CredentialInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.wallet_credentials().await })
            .await
            .map_err(join_err)?
    }

    pub async fn wallet_spending_credentials(
        &self,
    ) -> Result<Vec<InFlightCredentialInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.wallet_spending_credentials().await })
            .await
            .map_err(join_err)?
    }

    /// Every credential in the local wallet with its lifecycle state
    /// (`active` / `spending` / `spent` / `expired`), newest first.
    pub async fn wallet_lifecycle(&self) -> Result<Vec<CredentialLifecycleInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.wallet_lifecycle().await })
            .await
            .map_err(join_err)?
    }

    pub async fn recover_spending_credentials(&self) -> Result<Vec<String>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.recover_spending_credentials().await })
            .await
            .map_err(join_err)?
    }

    // -----------------------------------------------------------------------
    // The Record — read-only queries over the local trail. Pure SELECTs;
    // newest first; `limit`/`offset` for windowed fetches.
    // -----------------------------------------------------------------------

    pub async fn list_attestations(
        &self,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<AttestationInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_attestations(limit, offset).await })
            .await
            .map_err(join_err)?
    }

    /// The full raw attestation document, or `None` if the hash is unknown.
    pub async fn attestation_detail(
        &self,
        hash: String,
    ) -> Result<Option<AttestationDetail>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.attestation_detail(&hash).await })
            .await
            .map_err(join_err)?
    }

    pub async fn list_requests(
        &self,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<RequestInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_requests(limit, offset).await })
            .await
            .map_err(join_err)?
    }

    /// The full recorded request/response pair (raw bodies included), or
    /// `None` if the id is unknown.
    pub async fn request_detail(&self, id: String) -> Result<Option<RequestDetail>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.request_detail(&id).await })
            .await
            .map_err(join_err)?
    }

    pub async fn spend_trail(
        &self,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<SpendTrailEntry>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.spend_trail(limit, offset).await })
            .await
            .map_err(join_err)?
    }

    pub async fn available_models(&self) -> Result<Vec<ModelInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.available_models().await })
            .await
            .map_err(join_err)?
    }

    // -----------------------------------------------------------------------
    // Backends — the configured inference destinations
    // -----------------------------------------------------------------------

    /// List the configured backends (soft-removed ones excluded): the
    /// eidola and local singletons first, then external backends in
    /// creation order. Refresh on [`Change::Backends`].
    pub async fn list_backends(&self) -> Result<Vec<BackendInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_backends().await })
            .await
            .map_err(join_err)?
    }

    /// Add an external backend (kind `openai` or `llamacpp`), or revive a
    /// previously removed one with the same id.
    pub async fn add_backend(&self, new: NewBackend) -> Result<BackendInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.add_backend(new).await })
            .await
            .map_err(join_err)?
    }

    /// Enable or disable a backend. Works for the singletons too —
    /// disabling `eidola` is the "no account, on-device only" mode.
    pub async fn set_backend_enabled(&self, id: String, enabled: bool) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.set_backend_enabled(&id, enabled).await })
            .await
            .map_err(join_err)?
    }

    /// Update an external backend's configuration (display name, base URL,
    /// api key, models directory, pinned model list).
    pub async fn update_backend(&self, id: String, update: BackendUpdate) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.update_backend(&id, update).await })
            .await
            .map_err(join_err)?
    }

    /// Soft-remove an external backend. Its forensic trail (request rows)
    /// stays resolvable; re-adding the same id revives the row.
    pub async fn remove_backend(&self, id: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.remove_backend(&id).await })
            .await
            .map_err(join_err)?
    }

    /// The models one backend offers, as selectable entries (ids are the
    /// qualified selection strings). See `backends::backend_models` for the
    /// per-kind sources — and note that a generic OpenAI-compatible server
    /// is not guaranteed to implement `GET /v1/models`; a failed listing
    /// suggests pinning the backend's models manually.
    pub async fn backend_models(&self, id: String) -> Result<Vec<ModelInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.backend_models(&id).await })
            .await
            .map_err(join_err)?
    }

    // -----------------------------------------------------------------------
    // Local models — download, delete, and llama.cpp engine lifecycle
    // -----------------------------------------------------------------------

    /// The curated downloadable-model catalog (static data; installed
    /// state comes from [`AppCore::local_models_state`]).
    pub fn local_model_catalog(&self) -> &'static [LocalCatalogEntry] {
        LOCAL_MODEL_CATALOG
    }

    /// Snapshot of the local-inference domain: resolved engine binary and
    /// every model with its lifecycle state. Refresh on
    /// [`Change::LocalModels`].
    pub async fn local_models_state(&self) -> Result<LocalModelsState, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.local_models_state().await })
            .await
            .map_err(join_err)?
    }

    /// Start a background model download from a direct `.gguf` URL (or a
    /// Hugging Face file-page URL). Returns the future `<slug>@local` id
    /// immediately; progress arrives via [`Change::LocalModels`].
    pub async fn download_local_model(&self, url: String) -> Result<String, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.download_local_model(&url).await })
            .await
            .map_err(join_err)?
    }

    /// Cancel an in-flight model download; the partial file is removed.
    pub async fn cancel_local_model_download(&self, id: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.cancel_local_model_download(&id).await })
            .await
            .map_err(join_err)?
    }

    /// Delete a downloaded model from disk. Fails while the model is
    /// loaded (unload first) or downloading (cancel instead).
    pub async fn delete_local_model(&self, id: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.delete_local_model(&id).await })
            .await
            .map_err(join_err)?
    }

    /// Load a local model: spawn `llama-server` for it and wait until it
    /// is ready to serve. Resolves when the model is chat-selectable (or
    /// the load failed); intermediate states arrive via
    /// [`Change::LocalModels`].
    pub async fn load_local_model(&self, id: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.load_local_model(&id).await })
            .await
            .map_err(join_err)?
    }

    /// Unload a local model, terminating its engine subprocess.
    pub async fn unload_local_model(&self, id: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.unload_local_model(&id).await })
            .await
            .map_err(join_err)?
    }

    /// Pin or unpin a loaded model's engine. Pinned engines are protected
    /// from automatic (LRU) unloading when another model needs the memory;
    /// manual unload still applies.
    pub async fn set_local_model_pinned(&self, id: String, pinned: bool) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.set_local_model_pinned(&id, pinned).await })
            .await
            .map_err(join_err)?
    }

    /// Test-only seam: mark `slug` as a loaded engine model for
    /// `backend_id` served at `127.0.0.1:<port>`, so integration tests can
    /// drive engine-backed chat turns against a mock upstream without
    /// spawning a real `llama-server`.
    #[doc(hidden)]
    pub fn test_register_loaded_local_model(&self, backend_id: &str, slug: &str, port: u16) {
        self.inner.local.register_for_test(backend_id, slug, port);
    }

    /// Test-only seam: register a fake ready engine with explicit
    /// footprint / pin / LRU timestamp — the eviction tests' fixture.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn test_register_engine(
        &self,
        backend_id: &str,
        slug: &str,
        port: u16,
        footprint: u64,
        pinned: bool,
        last_used_ms: i64,
    ) {
        self.inner.local.register_engine_for_test(
            backend_id,
            slug,
            port,
            footprint,
            pinned,
            last_used_ms,
        );
    }

    /// Test-only seam: pin the engine-pool memory budget so eviction tests
    /// are deterministic on any machine.
    #[doc(hidden)]
    pub fn test_set_memory_budget(&self, budget: u64) {
        self.inner.local.set_memory_budget_for_test(budget);
    }

    /// List spaces, most recently active first. Archived spaces are
    /// excluded unless `include_archived` is set.
    pub async fn list_spaces(&self, include_archived: bool) -> Result<Vec<SpaceInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_spaces(include_archived).await })
            .await
            .map_err(join_err)?
    }

    pub async fn get_space_messages(
        &self,
        space_id: String,
    ) -> Result<Vec<SpaceMessage>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.get_space_messages(&space_id).await })
            .await
            .map_err(join_err)?
    }

    /// The threaded-post render tree for a space (current generations, reply
    /// DAG flattened to render-rows). The GUI's transcript renders from this;
    /// `get_space_messages` remains the upstream-context view.
    pub async fn get_space_tree(&self, space_id: String) -> Result<Vec<PostNode>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.get_space_tree(&space_id).await })
            .await
            .map_err(join_err)?
    }

    pub async fn create_space(&self, title: Option<String>) -> Result<SpaceInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.create_space(title.as_deref()).await })
            .await
            .map_err(join_err)?
    }

    pub async fn archive_space(&self, space_id: String) -> Result<bool, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.archive_space(&space_id).await })
            .await
            .map_err(join_err)?
    }

    /// Create a new space from a **specific** template (vs [`Self::create_space`],
    /// which uses the default). A missing/removed template is a typed error.
    pub async fn create_space_from_template(
        &self,
        template_id: String,
        title: Option<String>,
    ) -> Result<SpaceInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .create_space_from_template(&template_id, title.as_deref())
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Save a thought without requesting a response (the save-vs-request
    /// split). Needs no account or credential; creates the space when
    /// `space_id` is `None`.
    pub async fn post(
        &self,
        prompt: String,
        space_id: Option<String>,
    ) -> Result<PostResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.post(space_id.as_deref(), &prompt, None).await })
            .await
            .map_err(join_err)?
    }

    /// Save a post that **replies to a specific post** (`reply_to`), branching
    /// the thread there rather than continuing the tail. The save side of an
    /// inline reply.
    pub async fn post_reply(
        &self,
        prompt: String,
        space_id: Option<String>,
        reply_to: Option<String>,
    ) -> Result<PostResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .post(space_id.as_deref(), &prompt, reply_to.as_deref())
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Edit a post by appending a new generation (append-only; the prior
    /// version is preserved and resolvable). `action_id` is any generation of
    /// the target item.
    pub async fn edit_post(
        &self,
        action_id: String,
        new_prompt: String,
    ) -> Result<PostResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.edit_post(&action_id, &new_prompt).await })
            .await
            .map_err(join_err)?
    }

    pub async fn rename_space(&self, space_id: String, title: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.rename_space(&space_id, &title).await })
            .await
            .map_err(join_err)?
    }

    pub async fn chat(
        &self,
        prompt: String,
        model: String,
        space_id: Option<String>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.chat(&prompt, &model, space_id.as_deref()).await })
            .await
            .map_err(join_err)?
    }

    /// Regenerate an inference: append a new generation of its item (agent
    /// revise). `action_id` is any generation of the target item.
    pub async fn regenerate(
        &self,
        action_id: String,
        model: String,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.regenerate(&action_id, &model).await })
            .await
            .map_err(join_err)?
    }

    // -----------------------------------------------------------------------
    // Participants & space templates (Participants v1)
    // -----------------------------------------------------------------------

    /// The current participants of a space (the shared human "You" plus the
    /// space's per-space agent instances).
    pub async fn list_space_participants(
        &self,
        space_id: String,
    ) -> Result<Vec<ParticipantInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_space_participants(&space_id).await })
            .await
            .map_err(join_err)?
    }

    /// Add a new agent participant to a space. Emits [`Change::Participants`].
    pub async fn add_space_participant(
        &self,
        space_id: String,
        participant: NewParticipant,
    ) -> Result<ParticipantInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.add_space_participant(&space_id, participant).await })
            .await
            .map_err(join_err)?
    }

    /// Update a space participant's config (label, model, system prompt, notify
    /// policy). Editing a participant edits **that space only**. Emits
    /// [`Change::Participants`].
    pub async fn update_space_participant(
        &self,
        participant_id: String,
        update: ParticipantUpdate,
    ) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .update_space_participant(&participant_id, update)
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// "Override here": set per-membership overrides for a **referenced global**
    /// participant — this space only, leaving the shared global's own config
    /// untouched (vs [`Self::update_space_participant`]'s "edit everywhere").
    /// Each field's inner `None` reverts it to inherited. Emits
    /// [`Change::Participants`].
    pub async fn set_space_participant_override(
        &self,
        space_id: String,
        participant_id: String,
        override_: ParticipantOverride,
    ) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .set_space_participant_override(&space_id, &participant_id, override_)
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Remove an agent participant from a space (soft — the participant row
    /// survives so forensic `action.participant_id` references resolve). The
    /// shared human cannot be removed. Emits [`Change::Participants`].
    pub async fn remove_space_participant(
        &self,
        space_id: String,
        participant_id: String,
    ) -> Result<bool, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .remove_space_participant(&space_id, &participant_id)
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// The live (non-removed) space templates, each with its agent participants.
    pub async fn list_space_templates(&self) -> Result<Vec<SpaceTemplateInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_space_templates().await })
            .await
            .map_err(join_err)?
    }

    /// Create a new space template. Emits [`Change::Templates`].
    pub async fn create_template(
        &self,
        title: String,
        cascade_limit: i64,
        participants: Vec<NewTemplateParticipant>,
    ) -> Result<SpaceTemplateInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .create_template(&title, cascade_limit, participants)
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Project a space's current participants + settings into a **new** template
    /// ("Make template from this space"). Emits [`Change::Templates`].
    pub async fn template_from_space(
        &self,
        space_id: String,
        title: String,
    ) -> Result<SpaceTemplateInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.template_from_space(&space_id, &title).await })
            .await
            .map_err(join_err)?
    }

    /// Update a template's title / cascade limit and (when `participants` is
    /// `Some`) replace its participant set. Emits [`Change::Templates`].
    pub async fn update_template(
        &self,
        id: String,
        title: Option<String>,
        cascade_limit: Option<i64>,
        participants: Option<Vec<NewTemplateParticipant>>,
    ) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .update_template(&id, title.as_deref(), cascade_limit, participants)
                    .await
            })
            .await
            .map_err(join_err)?
    }

    /// Soft-remove a template (the built-in Default cannot be removed). Emits
    /// [`Change::Templates`].
    pub async fn remove_template(&self, id: String) -> Result<bool, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.remove_template(&id).await })
            .await
            .map_err(join_err)?
    }
}

// ============================================================================
// Internal API response types
// ============================================================================

#[derive(Deserialize)]
struct CreateAccountResponse {
    account_id: Uuid,
    secret: String,
    created_at: String,
}

#[derive(Deserialize)]
struct GetAccountResponse {
    id: Uuid,
    stripe_customer_id: Option<String>,
    created_at: String,
}

#[derive(Deserialize)]
struct TermsResponse {
    documents: Vec<TermsDocumentResponse>,
}

#[derive(Deserialize)]
struct TermsDocumentResponse {
    document: String,
    version: i64,
    url: String,
    sha256: String,
}

#[derive(Deserialize)]
struct ListPricesResponse {
    data: Vec<PriceResponse>,
}

#[derive(Deserialize)]
struct PriceResponse {
    id: String,
    product_name: String,
    product_description: Option<String>,
    unit_amount: Option<i64>,
    currency: String,
    #[allow(dead_code)]
    #[serde(rename = "type")]
    price_type: String,
    recurring: Option<RecurringResponse>,
    credits: i64,
}

#[derive(Deserialize)]
struct RecurringResponse {
    interval: String,
    interval_count: i64,
}

#[derive(Deserialize)]
struct CheckoutUrlResponse {
    checkout_url: String,
}

#[derive(Deserialize)]
struct BalancesResponse {
    available: i64,
    pools: Vec<BalancePoolResponse>,
}

#[derive(Deserialize)]
struct BalancePoolResponse {
    amount: i64,
    source: String,
    expires_at: Option<String>,
}

#[derive(Deserialize)]
struct ListKeysResponse {
    data: Vec<IssuerKeyResponse>,
}

#[derive(Deserialize)]
struct IssuerKeyResponse {
    id: String,
    public_key: String,
    domain_separator: String,
    issue_from: String,
    issue_until: String,
    #[allow(dead_code)]
    accept_until: String,
}

#[derive(Deserialize)]
struct IssueCredentialsResponse {
    issuance_response: String,
    issuer_key_id: String,
    credits: i64,
    #[allow(dead_code)]
    ledger_entry_id: String,
}

#[derive(Deserialize)]
struct ModelPricingInfo {
    per_prompt_token: ScaledPriceInfo,
    per_completion_token: ScaledPriceInfo,
    /// Present only for models priced per request (e.g. transcription).
    #[serde(default)]
    per_request: Option<ScaledPriceInfo>,
}

#[derive(Deserialize)]
struct ScaledPriceInfo {
    value: u64,
    scale_factor: u64,
}

impl ScaledPriceInfo {
    /// Credits per priced unit (token or request) as a float for display.
    /// Charging math elsewhere stays in integer `value`/`scale_factor`
    /// space; this is only the honest human-readable rate.
    fn credits_per_unit(&self) -> f64 {
        if self.scale_factor == 0 {
            0.0
        } else {
            self.value as f64 / self.scale_factor as f64
        }
    }
}

#[derive(Deserialize)]
struct ModelsResponseInfo {
    data: Vec<ModelListEntry>,
}

#[derive(Deserialize)]
struct ModelListEntry {
    id: String,
    context_length: u64,
    pricing: ModelPricingInfo,
}

// ============================================================================
// Refund processing helpers
// ============================================================================

/// Extract a refund token from a JSON object and store the resulting credential.
///
/// `refund_obj` is the `"refund"` value from a server response (either a chat
/// completion or the recovery endpoint). Returns `true` if the credential was
/// successfully stored.
#[allow(clippy::too_many_arguments)]
/// The shared preparation of a turn, built by [`Inner::prepare_turn`]:
/// the attested client, the thread attach plan, the assembled upstream
/// context, and the in-flight spend (proof + pending refund). The blocking
/// and streaming transports differ only in how they carry the request and
/// read the response; everything durable before and after the wire lives
/// here.
struct TurnPrep {
    db_conn: turso::Connection,
    provider_id: String,
    /// The configured backend this turn routes through — recorded on the
    /// request row (the forensic reference).
    backend_id: String,
    /// For engine-backed turns: the in-flight hold on the serving engine.
    /// Held for the turn's whole life (this struct's) so the engine is
    /// never auto-unloaded mid-request; dropping releases it.
    #[allow(dead_code)]
    engine_lease: Option<local_models::EngineLease>,
    attestation_log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>>,
    client: reqwest::Client,
    /// Connection row adopted from the most recent attestation flush; the
    /// request row records it.
    connection_id: Option<String>,
    base_url: String,
    /// The preparation timestamp — spend/refund rows key off it.
    now: i64,
    space_id: String,
    /// The canonical selection string (`qualified_model_id`) — recorded on
    /// actions, shown in UIs, returned in `ChatResult`.
    model: String,
    /// The model id the backend's HTTP API expects in the request body.
    /// Equals `model` except for openai backends (bare wire model).
    wire_model: String,
    model_participant_id: String,
    /// The acting agent participant's scope (`'space'` or `'global'`) — the
    /// action's pinned composite echo.
    model_participant_scope: String,
    max_completion_tokens: u32,
    /// The inference's attach plan (Reply → fresh item; Revise → a new
    /// generation superseding the target).
    inf_item_id: String,
    inf_supersedes: Option<String>,
    inf_reply_to: Option<String>,
    /// The context rows fed upstream, recorded as the context assembly.
    context_rows: Vec<db::SpaceActionRow>,
    /// The OpenAI messages array built from `context_rows`.
    messages: Vec<serde_json::Value>,
    /// Estimated hold for this turn. Always `0` when `spend` is `None`.
    charge_credits: u128,
    /// The in-flight credential spend — `None` for local turns, which have
    /// no billing. Its absence disables the ACT header, the refund
    /// machinery, and the `Wallet` emissions throughout the transports.
    spend: Option<SpendPrep>,
    /// The `Authorization` header value; present iff `spend` is.
    auth_value: Option<String>,
}

/// The credential-spend half of a prepared (remote) turn: the spendable
/// credential, the proof materials, and the pending-refund row id.
struct SpendPrep {
    cred: db::SpendableCredential,
    public_key: PublicKey,
    params: Params,
    spend_proof: SpendProof<128>,
    pre_refund: PreRefund,
    pre_cred_id: String,
}

impl TurnPrep {
    /// Wrap an error with the (always-persisted) space id so a blank GUI
    /// window can adopt it on failure.
    fn wrap(&self, source: AppError) -> AppError {
        AppError::ChatFailed {
            space_id: self.space_id.clone(),
            source: Box::new(source),
        }
    }

    /// The chat request body for this turn.
    fn request_body(&self, stream: bool) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.wire_model,
            "messages": self.messages,
            "max_completion_tokens": self.max_completion_tokens,
        });
        if stream {
            body["stream"] = serde_json::Value::Bool(true);
            // The Eidola server forces `include_usage` upstream regardless
            // (accurate refunds depend on it), so the remote request stays
            // minimal — but a local llama-server only reports usage when the
            // client asks.
            if self.spend.is_none() {
                body["stream_options"] = serde_json::json!({ "include_usage": true });
            }
        }
        body
    }

    /// Flush attestations captured since the last flush (a fresh handshake
    /// during the request), adopting the new connection id if one was
    /// recorded.
    async fn flush_new_attestations(&mut self) -> Result<(), AppError> {
        if let Some(new_cid) = flush_attestations(
            &self.attestation_log,
            &self.db_conn,
            &self.provider_id,
            &self.base_url,
            self.now,
        )
        .await?
        {
            self.connection_id = Some(new_cid);
        }
        Ok(())
    }

    /// Apply a refund object (inline from a response body, or recovered) to
    /// the pending spend — writes the successor credential. A no-op for
    /// local turns (nothing was spent).
    async fn process_refund_obj(&self, refund_obj: &serde_json::Value) -> Result<(), AppError> {
        let Some(spend) = &self.spend else {
            return Ok(());
        };
        process_refund(
            refund_obj,
            &spend.params,
            &spend.spend_proof,
            &spend.pre_refund,
            &spend.public_key,
            &self.db_conn,
            &spend.pre_cred_id,
            spend.cred.generation + 1,
            self.now,
        )
        .await
    }

    /// Best-effort refund recovery via `/v1/credentials/refund`. Returns
    /// whether a successor credential was written (the caller decides whether
    /// that warrants an immediate `Wallet` emission). Always `false` for
    /// local turns — there is no spend to recover.
    async fn try_refund_recovery(&self) -> bool {
        let (Some(_), Some(auth_value)) = (&self.spend, &self.auth_value) else {
            return false;
        };
        match recover_refund(&self.client, &self.base_url, auth_value).await {
            Ok(refund_obj) => self.process_refund_obj(&refund_obj).await.is_ok(),
            Err(_) => false,
        }
    }

    /// The nonce of the spending credential, if this turn carries a spend.
    /// Recorded on request rows; `None` keeps local turns honest in the
    /// Record.
    fn credential_nonce(&self) -> Option<String> {
        self.spend.as_ref().map(|s| s.cred.nonce.clone())
    }

    /// Persist the turn's durable rows — the inference action (per the attach
    /// plan, with its reply edge), the context-assembly record (exactly the
    /// actions fed upstream), the response content block, and the request row
    /// — and return the inference action id. Emissions stay with the caller
    /// (they differ per exit point; see the table in `tests/bus.rs`).
    #[allow(clippy::too_many_arguments)]
    async fn persist_turn(
        &self,
        action_status: &str,
        input_tokens: Option<i64>,
        output_tokens: Option<i64>,
        content: &str,
        request_body_json: &serde_json::Value,
        request_at: i64,
        response_at: i64,
        http_status: u16,
        response_body: Vec<u8>,
    ) -> Result<String, AppError> {
        let inference_action_id = Uuid::now_v7().to_string();
        db::insert_action(
            &self.db_conn,
            &db::ActionEntry {
                id: inference_action_id.clone(),
                space_id: self.space_id.clone(),
                participant_id: self.model_participant_id.clone(),
                participant_scope: self.model_participant_scope.clone(),
                item_id: self.inf_item_id.clone(),
                supersedes_action_id: self.inf_supersedes.clone(),
                action_type: "inference".to_string(),
                status: action_status.to_string(),
                intent: None,
                model: Some(self.model.clone()),
                input_tokens,
                output_tokens,
                // Local turns record no charge (`None`), not a fake zero.
                credits_consumed: self.spend.as_ref().map(|_| self.charge_credits as i64),
                created_at: now_ms(),
            },
        )
        .await?;
        if let Some(ref ante) = self.inf_reply_to {
            db::insert_action_antecedent(&self.db_conn, &inference_action_id, ante, 0, "reply")
                .await?;
        }

        // Record context assembly: exactly the actions fed into this inference.
        let context_assembly_id = Uuid::now_v7().to_string();
        db::insert_context_assembly(
            &self.db_conn,
            &context_assembly_id,
            &inference_action_id,
            None,
            input_tokens,
            false,
            now_ms(),
        )
        .await?;

        let mut fed_ids: Vec<String> = Vec::new();
        for r in &self.context_rows {
            if !fed_ids.contains(&r.action_id) {
                fed_ids.push(r.action_id.clone());
            }
        }
        for (pos, aid) in fed_ids.iter().enumerate() {
            db::insert_context_assembly_action(
                &self.db_conn,
                &context_assembly_id,
                aid,
                pos as i64,
            )
            .await?;
        }

        if !content.is_empty() {
            db::insert_text_content_block(
                &self.db_conn,
                &Uuid::now_v7().to_string(),
                &inference_action_id,
                0,
                "text",
                content,
            )
            .await?;
        }

        db::insert_request(
            &self.db_conn,
            &db::Request {
                id: Uuid::now_v7().to_string(),
                connection_id: self.connection_id.clone(),
                action_id: Some(inference_action_id.clone()),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                request_headers: None,
                request_body: Some(request_body_json.to_string().into_bytes()),
                response_status: Some(http_status as i64),
                response_headers: None,
                response_body: Some(response_body),
                request_at,
                response_at: Some(response_at),
                duration_ms: Some(response_at - request_at),
                error: None,
                credential_nonce: self.credential_nonce(),
                created_at: now_ms(),
                backend_id: Some(self.backend_id.clone()),
            },
        )
        .await?;

        Ok(inference_action_id)
    }
}

/// Apply a refund object to a pending spend: decode + verify the refund
/// against the spend proof and write the successor credential. Called from
/// `TurnPrep::process_refund_obj` (live turns) and from the startup wallet
/// recovery (which reconstructs the spend materials from persisted rows —
/// which is why this stays a free function taking them piecewise).
#[allow(clippy::too_many_arguments)]
async fn process_refund(
    refund_obj: &serde_json::Value,
    params: &Params,
    spend_proof: &SpendProof<128>,
    pre_refund: &PreRefund,
    public_key: &PublicKey,
    db_conn: &turso::Connection,
    pre_cred_id: &str,
    generation: i64,
    now: i64,
) -> Result<(), AppError> {
    let refund_b64 = refund_obj
        .get("refund")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::Credential {
            message: "missing refund data in response".into(),
        })?;
    let refund_key_id = refund_obj
        .get("issuer_key_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::Credential {
            message: "missing issuer_key_id in refund".into(),
        })?;

    let refund_cbor = URL_SAFE_NO_PAD
        .decode(refund_b64)
        .map_err(|e| AppError::Credential {
            message: format!("invalid refund base64: {e}"),
        })?;
    let refund = Refund::from_cbor(&refund_cbor).map_err(|e| AppError::Credential {
        message: format!("invalid refund CBOR: {e}"),
    })?;

    let new_token = pre_refund
        .to_credit_token::<128>(params, spend_proof, &refund, public_key)
        .map_err(|e| AppError::Credential {
            message: format!("failed to construct refund credit token: {e:?}"),
        })?;

    let new_token_cbor = new_token.to_cbor().map_err(|e| AppError::Credential {
        message: format!("failed to encode new credit token: {e}"),
    })?;
    let new_nonce = hex_encode(&new_token.nullifier().to_bytes());
    let new_credits =
        scalar_to_credit::<128>(&new_token.credits()).map_err(|e| AppError::Credential {
            message: format!("invalid credit amount in refund token: {e}"),
        })?;

    db::insert_credential(
        db_conn,
        &new_nonce,
        pre_cred_id,
        refund_key_id,
        &new_token_cbor,
        new_credits as i64,
        generation,
        now,
    )
    .await?;

    Ok(())
}

/// Attempt to recover a refund token from the server via
/// `POST /v1/credentials/refund`. Returns the refund JSON object on success.
async fn recover_refund(
    client: &reqwest::Client,
    base_url: &str,
    auth_value: &str,
) -> Result<serde_json::Value, AppError> {
    let resp = client
        .post(format!("{base_url}/v1/credentials/refund"))
        .header("Authorization", auth_value)
        .send()
        .await
        .map_err(AppError::from_request)?;

    let status = resp.status();
    let body_text = resp.text().await.map_err(|e| AppError::Network {
        message: format!("failed to read recovery response: {e}"),
    })?;
    let body: serde_json::Value =
        serde_json::from_str(&body_text).map_err(|e| AppError::Network {
            message: format!("failed to parse recovery response: {e}"),
        })?;

    if !status.is_success() {
        return Err(AppError::Server {
            status: status.as_u16(),
            message: format!(
                "refund recovery failed: {}",
                parse_server_error_message(&body_text)
            ),
        });
    }

    body.get("refund")
        .cloned()
        .ok_or_else(|| AppError::Credential {
            message: "recovery response missing refund field".into(),
        })
}

// ============================================================================
// Free-standing helpers
// ============================================================================

/// Canonicalize a model reference to its `<model>@<backend-id>` form (the bare
/// `eidola` default stays bare), so a participant's stored `model_ref` and a
/// picked selection compare equal regardless of which sugar spelling either
/// used. Mirrors `prepare_turn`'s own canonicalization.
fn canonicalize_model_ref(model_ref: &str) -> String {
    let mr = backends::parse_model_ref(model_ref);
    backends::qualified_model_id(&mr.model, &mr.backend_id)
}

/// Convert space action rows into a sequence of role/content messages suitable
/// for the OpenAI messages array and for UI display. Groups content blocks by
/// action and concatenates text.
fn actions_to_messages(action_rows: &[db::SpaceActionRow]) -> Vec<SpaceMessage> {
    let mut messages: Vec<SpaceMessage> = Vec::new();
    let mut current_action_id: Option<&str> = None;

    for row in action_rows {
        let role = match (row.action_type.as_str(), row.participant_kind.as_str()) {
            ("user_input", _) => "user",
            ("inference", _) => "assistant",
            _ => continue, // skip tool_call, tool_result, etc. for now
        };

        if current_action_id == Some(row.action_id.as_str()) {
            // Additional content block for the same action — append text
            if let Some(text) = &row.text_content
                && let Some(last) = messages.last_mut()
            {
                last.content.push_str(text);
            }
        } else {
            // New action
            current_action_id = Some(&row.action_id);
            let content = row.text_content.clone().unwrap_or_default();
            messages.push(SpaceMessage {
                role: role.to_string(),
                content,
            });
        }
    }

    messages
}

/// Assemble the flattened threaded-post render list from the raw space-tree
/// materials. Pure (no I/O) so the spine-vs-branch flattening is unit-testable
/// without a database.
///
/// Threading uses the structural `reply` edges. The spine follows the first
/// (chronological) reply of each post and stays at the same `depth`; later
/// replies to the same post become branches (`depth + 1`, `is_branch = true`),
/// and their own sub-spines recurse the rule. Multiple thread roots are treated
/// as siblings of a virtual root (first root on the spine, the rest branches).
/// Output order is a pre-order walk (each post, then its first reply's whole
/// subtree, then later replies) — i.e. the spine reads top-to-bottom with
/// branches hanging beneath the post they reply to.
///
/// v1 threads by the **raw** reply antecedent: a reply edge whose target isn't
/// in the resolved post set (a non-tip generation once edits exist) is treated
/// as absent, making that post a root. Re-rooting such a child onto its parent
/// item's current tip is a 5.4 addition — a no-op while no edits exist.
fn build_post_tree(data: db::SpaceTreeData) -> Vec<PostNode> {
    use std::collections::HashMap;

    let db::SpaceTreeData {
        actions,
        blocks,
        edges,
    } = data;

    // Blocks grouped by action, preserving the query's (action, ordinal) order.
    let mut blocks_by_action: HashMap<String, Vec<PostBlock>> = HashMap::new();
    for b in blocks {
        blocks_by_action
            .entry(b.action_id)
            .or_default()
            .push(PostBlock {
                block_type: b.block_type,
                text: b.text_content,
                tool_name: b.tool_name,
                tool_call_id: b.tool_call_id,
                data: b.data,
            });
    }

    // The set of resolved post action ids (used to test whether a reply edge
    // points at a renderable post).
    let in_set: std::collections::HashSet<&str> =
        actions.iter().map(|a| a.action_id.as_str()).collect();

    // Split edges into the structural reply parent (at most one per action) and
    // the non-structural references.
    let mut reply_parent: HashMap<String, String> = HashMap::new();
    let mut references_by_action: HashMap<String, Vec<PostReference>> = HashMap::new();
    for e in edges {
        if e.relation == "reply" {
            // Reply threading follows **item identity**, not the raw causal
            // edge: the antecedent action may be a superseded generation (its
            // item was edited/regenerated after the reply), in which case the
            // reply threads under the item's *current tip* — the edited post
            // stays in place with its replies attached. The raw action id
            // stays on reference edges (causality is action ids; intended
            // logical flow is item ids). A target whose tip isn't a
            // renderable post (trace type, non-terminal status) still falls
            // out of `in_set` and the reply renders as a root.
            let target = e
                .antecedent_current_action_id
                .clone()
                .unwrap_or_else(|| e.antecedent_action_id.clone());
            // Only the first reply edge is structural (schema enforces one).
            if in_set.contains(target.as_str()) {
                reply_parent.entry(e.action_id).or_insert(target);
            }
        } else {
            references_by_action
                .entry(e.action_id)
                .or_default()
                .push(PostReference {
                    antecedent_action_id: e.antecedent_action_id,
                    ordinal: e.ordinal,
                    range_start: e.range_start,
                    range_end: e.range_end,
                    annotation: e.annotation,
                });
        }
    }

    // Children map (reverse of reply_parent). `actions` is already chronological
    // (created_at, action_id), so children land in chronological order and roots
    // keep their chronological order.
    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    let mut roots: Vec<String> = Vec::new();
    for a in &actions {
        match reply_parent.get(&a.action_id) {
            Some(parent) => children
                .entry(parent.clone())
                .or_default()
                .push(a.action_id.clone()),
            None => roots.push(a.action_id.clone()),
        }
    }

    // Index the action rows for O(1) lookup during the walk.
    let mut rows: HashMap<String, db::PostActionRow> = HashMap::with_capacity(actions.len());
    for a in actions {
        rows.insert(a.action_id.clone(), a);
    }

    // Pre-order walk with an explicit stack (avoids deep recursion on long
    // linear threads). Each frame carries (action_id, depth, is_branch); we push
    // children reversed so the first child pops next.
    let mut out: Vec<PostNode> = Vec::with_capacity(rows.len());
    let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut stack: Vec<(String, usize, bool)> = Vec::new();
    for (i, root) in roots.iter().enumerate().rev() {
        let (depth, is_branch) = if i == 0 { (0, false) } else { (1, true) };
        stack.push((root.clone(), depth, is_branch));
    }

    while let Some((action_id, depth, is_branch)) = stack.pop() {
        // Guard against malformed data forming a cycle (impossible by
        // construction — one reply parent, backward-pointing edges — but cheap
        // insurance against an infinite loop on bad input).
        if !emitted.insert(action_id.clone()) {
            continue;
        }
        let Some(row) = rows.get(&action_id) else {
            continue;
        };

        out.push(PostNode {
            action_id: row.action_id.clone(),
            item_id: row.item_id.clone(),
            parent_action_id: reply_parent.get(&action_id).cloned(),
            participant: PostParticipant {
                kind: row.participant_kind.clone(),
                label: row.participant_label.clone(),
            },
            action_type: row.action_type.clone(),
            generation: row.generation,
            generation_count: row.generation + 1,
            is_current: true,
            model: row.model.clone(),
            credits_consumed: row.credits_consumed,
            relation: reply_parent.get(&action_id).map(|_| "reply".to_string()),
            depth,
            is_branch,
            blocks: blocks_by_action
                .get(&action_id)
                .cloned()
                .unwrap_or_default(),
            references: references_by_action
                .get(&action_id)
                .cloned()
                .unwrap_or_default(),
            created_at: row.created_at,
        });

        if let Some(kids) = children.get(&action_id) {
            for (i, kid) in kids.iter().enumerate().rev() {
                let (kid_depth, kid_branch) = if i == 0 {
                    (depth, false)
                } else {
                    (depth + 1, true)
                };
                stack.push((kid.clone(), kid_depth, kid_branch));
            }
        }
    }

    out
}

/// Maximum length of an auto-derived space title, in characters.
const TITLE_MAX_CHARS: usize = 64;

/// Maximum length of a listing snippet, in characters.
const SNIPPET_MAX_CHARS: usize = 120;

/// Derive a space title from the user's first prompt: take the first
/// non-empty line, strip leading markdown block markers (headings, list
/// bullets, blockquotes, numbered lists) and surrounding emphasis
/// characters, then truncate to ~64 chars on a word boundary (appending an
/// ellipsis when truncated). Returns `None` if nothing presentable is left.
fn derive_space_title(prompt: &str) -> Option<String> {
    let line = prompt.lines().map(str::trim).find(|l| !l.is_empty())?;

    // Strip leading block markers, repeatedly — "> # Heading" etc.
    let mut s = line;
    loop {
        let mut t = s.trim_start_matches(['#', '>']).trim_start();
        // Unordered list bullets.
        for marker in ["- ", "* ", "+ "] {
            if let Some(rest) = t.strip_prefix(marker) {
                t = rest.trim_start();
            }
        }
        // Ordered list markers like "1. " / "12) ".
        let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 {
            let after = &t[digits..];
            if let Some(rest) = after
                .strip_prefix(". ")
                .or_else(|| after.strip_prefix(") "))
            {
                t = rest.trim_start();
            }
        }
        if t == s {
            break;
        }
        s = t;
    }

    // Strip emphasis/code markers from the edges ("**Bold ask**", "`code`").
    let s = s.trim_matches(['*', '_', '`']).trim();
    if s.is_empty() {
        return None;
    }

    Some(truncate_on_word_boundary(s, TITLE_MAX_CHARS))
}

/// Snippet for the space listing: first line-collapsed ~120 chars of the
/// given text, truncated on a word boundary. Returns `None` for
/// whitespace-only input.
fn snippet_of(text: &str) -> Option<String> {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    Some(truncate_on_word_boundary(&collapsed, SNIPPET_MAX_CHARS))
}

/// Truncate `s` to at most `max_chars` characters, breaking on a word
/// boundary where possible and appending `…` when anything was cut. The
/// ellipsis is not counted against the budget; `max_chars` must be ≥ 1.
fn truncate_on_word_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::new();
    let mut count = 0usize;
    for word in s.split_whitespace() {
        let word_chars = word.chars().count();
        let sep = usize::from(!out.is_empty());
        if count + sep + word_chars > max_chars {
            break;
        }
        if sep == 1 {
            out.push(' ');
        }
        out.push_str(word);
        count += sep + word_chars;
    }
    if out.is_empty() {
        // Single word longer than the budget — hard-cut it.
        out = s.chars().take(max_chars).collect();
    }
    out.push('…');
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, AppError> {
    if !s.len().is_multiple_of(2) {
        return Err(AppError::Credential {
            message: "odd-length hex string".into(),
        });
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| AppError::Credential {
                message: format!("invalid hex: {e}"),
            })
        })
        .collect()
}

const ACT_TOKEN_TYPE: u16 = 0xE5AD;
const ISSUER_NAME: &str = "eidola";
const ORIGIN_INFO: &str = "inference";

fn serialize_token_challenge() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
    buf.extend_from_slice(&(ISSUER_NAME.len() as u16).to_be_bytes());
    buf.extend_from_slice(ISSUER_NAME.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&(ORIGIN_INFO.len() as u16).to_be_bytes());
    buf.extend_from_slice(ORIGIN_INFO.as_bytes());
    buf.push(0);
    buf
}

fn compute_challenge_digest() -> [u8; 32] {
    Sha256::digest(serialize_token_challenge()).into()
}

/// Current time as milliseconds since Unix epoch.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_millis() as i64
}

/// Parse an ISO 8601 timestamp to epoch ms.
pub fn iso_to_ms(s: &str) -> Result<i64, AppError> {
    let s = s.trim().trim_end_matches('Z');
    let (date, time) = s.split_once('T').ok_or_else(|| AppError::Network {
        message: format!("invalid ISO 8601: {s}"),
    })?;
    let dp: Vec<&str> = date.split('-').collect();
    let tp: Vec<&str> = time.split(':').collect();
    if dp.len() != 3 || tp.len() != 3 {
        return Err(AppError::Network {
            message: format!("invalid ISO 8601: {s}"),
        });
    }
    let y: i64 = dp[0].parse().map_err(|_| AppError::Network {
        message: "bad year".into(),
    })?;
    let m: u32 = dp[1].parse().map_err(|_| AppError::Network {
        message: "bad month".into(),
    })?;
    let d: u32 = dp[2].parse().map_err(|_| AppError::Network {
        message: "bad day".into(),
    })?;
    let hour: i64 = tp[0].parse().map_err(|_| AppError::Network {
        message: "bad hour".into(),
    })?;
    let min: i64 = tp[1].parse().map_err(|_| AppError::Network {
        message: "bad minute".into(),
    })?;
    let sec: i64 = tp[2]
        .split('.')
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| AppError::Network {
            message: "bad second".into(),
        })?;
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj.rem_euclid(400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i64 - 719468;
    let secs = days * 86400 + hour * 3600 + min * 60 + sec;
    Ok(secs * 1000)
}

pub(crate) fn load_native_root_store() -> rustls::RootCertStore {
    let mut store = rustls::RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    for cert in native.certs {
        let _ = store.add(cert);
    }
    store
}

fn params_from_domain_separator(ds: &str) -> Result<Params, AppError> {
    let parts: Vec<&str> = ds.split(':').collect();
    if parts.len() != 5 {
        return Err(AppError::Config {
            message: format!("domain separator has wrong format: {ds}"),
        });
    }
    Ok(Params::new(parts[1], parts[2], parts[3], parts[4]))
}

async fn read_response(resp: reqwest::Response) -> Result<(reqwest::StatusCode, String), AppError> {
    let status = resp.status();
    let body = resp.text().await.map_err(|e| AppError::Network {
        message: format!("failed to read response body: {e}"),
    })?;
    Ok((status, body))
}

fn check_status(status: reqwest::StatusCode, body: &str) -> Result<(), AppError> {
    if status.is_success() {
        return Ok(());
    }
    // 428 Precondition Required is the server's terms-acceptance gate —
    // typed so UIs route to a review-and-accept step instead of showing a
    // generic server error.
    if status == reqwest::StatusCode::PRECONDITION_REQUIRED {
        return Err(AppError::TermsAcceptanceRequired {
            message: parse_server_error_message(body),
        });
    }
    Err(AppError::Server {
        status: status.as_u16(),
        message: parse_server_error_message(body),
    })
}

/// Best-effort extraction of a human-readable error message from a
/// non-2xx response body. Tries the OpenAI-shaped `{"error":{"message":"..."}}`
/// envelope first; falls back to the raw body text (trimmed and
/// length-capped — axum's body-extractor rejection bodies are plain
/// text, not JSON, and were previously bucketed to "unknown error" by
/// the old JSON-only path); finally falls back to a literal "unknown
/// error" only when the body is empty.
pub(crate) fn parse_server_error_message(body: &str) -> String {
    if let Some(msg) = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(String::from)
        })
    {
        return msg;
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "unknown error".to_string();
    }
    // Cap the message so a chatty rejection body doesn't blow up the UI.
    const MAX_LEN: usize = 500;
    if trimmed.len() > MAX_LEN {
        let mut capped: String = trimmed.chars().take(MAX_LEN).collect();
        capped.push('…');
        capped
    } else {
        trimmed.to_string()
    }
}

async fn fetch_models(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<ModelsResponseInfo, AppError> {
    let resp = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .map_err(AppError::from_request)?;
    let (status, body) = read_response(resp).await?;
    check_status(status, &body)?;
    serde_json::from_str(&body).map_err(|e| AppError::Network {
        message: format!("failed to parse models response: {e}"),
    })
}

async fn flush_attestations(
    attestation_log: &Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>,
    db_conn: &turso::Connection,
    provider_id: &str,
    base_url: &str,
    now: i64,
) -> Result<Option<String>, AppError> {
    let attestations: Vec<_> = attestation_log.lock().unwrap().drain(..).collect();
    let mut connection_id = None;
    for att in &attestations {
        db::upsert_attestation(
            db_conn,
            &att.attestation_hash,
            &att.attestation_doc,
            Some(&att.pcr_digest),
            now,
        )
        .await?;
        let cid = Uuid::now_v7().to_string();
        db::insert_connection(
            db_conn,
            &cid,
            provider_id,
            base_url,
            "clearnet",
            Some(&att.attestation_hash),
            now,
            now,
        )
        .await?;
        connection_id = Some(cid);
    }
    Ok(connection_id)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        let data = vec![0xde, 0xad, 0xbe, 0xef];
        let encoded = hex_encode(&data);
        assert_eq!(encoded, "deadbeef");
        assert_eq!(hex_decode(&encoded).unwrap(), data);
    }

    #[test]
    fn hex_decode_rejects_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn challenge_digest_is_deterministic() {
        let d1 = compute_challenge_digest();
        let d2 = compute_challenge_digest();
        assert_eq!(d1, d2);
    }

    #[test]
    fn iso_to_ms_basic() {
        // 2026-01-01T00:00:00Z → 1767225600000
        let ms = iso_to_ms("2026-01-01T00:00:00Z").unwrap();
        assert_eq!(ms, 1767225600000);
    }

    #[test]
    fn token_challenge_serialization() {
        let buf = serialize_token_challenge();
        assert_eq!(u16::from_be_bytes([buf[0], buf[1]]), ACT_TOKEN_TYPE);
    }

    #[test]
    fn params_from_domain_separator_valid() {
        let p = params_from_domain_separator("ACT-v1:org:service:deploy:ver");
        assert!(p.is_ok());
    }

    #[test]
    fn params_from_domain_separator_rejects_wrong_format() {
        assert!(params_from_domain_separator("bad").is_err());
    }

    // --- Default template config -------------------------------------------

    #[test]
    fn default_model_resolves_from_default_template_agent() {
        let dir = tempfile::tempdir().unwrap();
        let config_dir = dir.path().to_path_buf();
        let data_dir = dir.path().join("data");

        let core = AppCore::new(config_dir.clone(), data_dir.clone());
        // Fresh install: default_template is the seeded id, and the async
        // default_model resolves from the seeded template's agent (= DEFAULT_MODEL).
        assert_eq!(
            core.config_state().default_template,
            config::DEFAULT_TEMPLATE_ID
        );
        let model = core.runtime().block_on(core.default_model()).unwrap();
        assert_eq!(model, config::DEFAULT_MODEL);
    }

    /// Regression for the nested-runtime panic (PR #221 review): the CLI runs
    /// `runtime().block_on(run(..))` and inside `run` calls `config_state()`
    /// (no-subcommand + `chat` without `--model`). config_state must be
    /// runtime-safe, and default_model resolution must work via `.await`
    /// inside that context — not a nested `block_on` ("Cannot start a runtime
    /// from within a runtime").
    #[test]
    fn config_state_and_default_model_are_safe_inside_the_runtime() {
        let dir = tempfile::tempdir().unwrap();
        let core = AppCore::new(dir.path().to_path_buf(), dir.path().join("data"));
        // Exactly the CLI's shape: sync config_state() + awaited default_model()
        // driven from within the core runtime.
        let (tmpl, model) = core.runtime().block_on(async {
            let state = core.config_state(); // must not panic here
            let model = core.default_model().await.unwrap();
            (state.default_template, model)
        });
        assert_eq!(tmpl, config::DEFAULT_TEMPLATE_ID);
        assert_eq!(model, config::DEFAULT_MODEL);
    }

    #[test]
    fn set_default_template_round_trips_and_rejects_blank() {
        let dir = tempfile::tempdir().unwrap();
        let config_dir = dir.path().to_path_buf();
        let data_dir = dir.path().join("data");

        let core = AppCore::new(config_dir.clone(), data_dir.clone());
        core.set_default_template("00000000-0000-7000-8000-0000000000ab".into())
            .unwrap();
        assert_eq!(
            core.config_state().default_template,
            "00000000-0000-7000-8000-0000000000ab"
        );

        // A fresh core over the same config dir sees the persisted value.
        let core2 = AppCore::new(config_dir, data_dir);
        assert_eq!(
            core2.config_state().default_template,
            "00000000-0000-7000-8000-0000000000ab"
        );

        // Whitespace-only is rejected and leaves the config untouched.
        assert!(core2.set_default_template("   ".into()).is_err());
        assert_eq!(
            core2.config_state().default_template,
            "00000000-0000-7000-8000-0000000000ab"
        );
    }

    #[test]
    fn set_circadian_settings_round_trip_through_config_state() {
        let dir = tempfile::tempdir().unwrap();
        let config_dir = dir.path().to_path_buf();
        let data_dir = dir.path().join("data");

        let core = AppCore::new(config_dir.clone(), data_dir.clone());
        let state = core.config_state();
        // `auto` (follow the sun) is the shipped default since the
        // local-inference wave flipped it from `system`.
        assert_eq!(state.appearance, config::AppearanceSetting::Auto);
        assert_eq!(state.time_of_day_tint, config::TimeOfDayTint::On);
        assert_eq!(state.light_character, config::LightCharacter::Neutral);

        // Font scale defaults to Actual Size and round-trips its override.
        assert_eq!(state.font_scale, config::FONT_SCALE_DEFAULT);

        core.set_appearance(config::AppearanceSetting::Day).unwrap();
        core.set_time_of_day_tint(config::TimeOfDayTint::Off)
            .unwrap();
        core.set_light_character(config::LightCharacter::Cool)
            .unwrap();
        core.set_font_scale(1.25).unwrap();
        let state = core.config_state();
        assert_eq!(state.appearance, config::AppearanceSetting::Day);
        assert_eq!(state.time_of_day_tint, config::TimeOfDayTint::Off);
        assert_eq!(state.light_character, config::LightCharacter::Cool);
        assert_eq!(state.font_scale, 1.25);

        // An out-of-range scale is clamped on write, never stored verbatim.
        core.set_font_scale(50.0).unwrap();
        assert_eq!(core.config_state().font_scale, config::FONT_SCALE_MAX);
        core.set_font_scale(1.0).unwrap();

        // A fresh core over the same config dir sees the persisted values.
        let core2 = AppCore::new(config_dir, data_dir);
        let state = core2.config_state();
        assert_eq!(state.appearance, config::AppearanceSetting::Day);
        assert_eq!(state.time_of_day_tint, config::TimeOfDayTint::Off);
        assert_eq!(state.font_scale, config::FONT_SCALE_DEFAULT);
    }

    // --- Auto-provisioning decision logic ---------------------------------

    #[test]
    fn auto_allocation_uses_default_chunk_when_balance_is_plentiful() {
        // Plenty of balance, small charge → allocate the default chunk,
        // leaving the rest of the balance in the account.
        let amount = auto_allocation_amount(10 * DEFAULT_ALLOCATION_CREDITS, 6_200).unwrap();
        assert_eq!(amount, DEFAULT_ALLOCATION_CREDITS);
    }

    #[test]
    fn auto_allocation_is_capped_by_available_balance() {
        // Balance covers the charge but is below the default chunk →
        // allocate everything that's available.
        let amount = auto_allocation_amount(50_000, 6_200).unwrap();
        assert_eq!(amount, 50_000);
    }

    #[test]
    fn auto_allocation_grows_to_cover_a_large_charge() {
        // A single charge larger than the default chunk must still fit in
        // one credential (spends draw from exactly one credential).
        let required = DEFAULT_ALLOCATION_CREDITS + 500_000;
        let amount = auto_allocation_amount(10 * DEFAULT_ALLOCATION_CREDITS, required).unwrap();
        assert_eq!(amount, required);
    }

    #[test]
    fn auto_allocation_exact_balance_allocates_it_all() {
        let amount = auto_allocation_amount(6_200, 6_200).unwrap();
        assert_eq!(amount, 6_200);
    }

    #[test]
    fn check_status_maps_428_to_terms_acceptance_required() {
        let body = r#"{"error":{"message":"acceptance of the current terms_of_service is required","type":"terms_acceptance_required"}}"#;
        let err = check_status(reqwest::StatusCode::PRECONDITION_REQUIRED, body).unwrap_err();
        match err {
            AppError::TermsAcceptanceRequired { message } => {
                assert!(message.contains("terms_of_service"));
            }
            other => panic!("expected TermsAcceptanceRequired, got {other:?}"),
        }
    }

    #[test]
    fn auto_allocation_fails_typed_when_balance_cannot_cover_charge() {
        let err = auto_allocation_amount(1_000, 6_200).unwrap_err();
        match err {
            AppError::InsufficientBalance {
                available,
                required,
            } => {
                assert_eq!(available, 1_000);
                assert_eq!(required, 6_200);
            }
            other => panic!("expected InsufficientBalance, got {other:?}"),
        }
    }

    #[test]
    fn auto_allocation_fails_typed_on_zero_balance() {
        assert!(matches!(
            auto_allocation_amount(0, 1),
            Err(AppError::InsufficientBalance {
                available: 0,
                required: 1
            })
        ));
    }

    #[test]
    fn derive_title_takes_first_line() {
        assert_eq!(
            derive_space_title("How do tides work?\n\nAnd a second question."),
            Some("How do tides work?".to_string())
        );
    }

    #[test]
    fn derive_title_skips_leading_blank_lines() {
        assert_eq!(
            derive_space_title("\n\n  \nActual question"),
            Some("Actual question".to_string())
        );
    }

    #[test]
    fn derive_title_strips_markdown_markers() {
        assert_eq!(
            derive_space_title("## A heading prompt"),
            Some("A heading prompt".to_string())
        );
        assert_eq!(
            derive_space_title("- a list item"),
            Some("a list item".to_string())
        );
        assert_eq!(
            derive_space_title("> # quoted heading"),
            Some("quoted heading".to_string())
        );
        assert_eq!(
            derive_space_title("1. first thing"),
            Some("first thing".to_string())
        );
        assert_eq!(
            derive_space_title("**Bold ask**"),
            Some("Bold ask".to_string())
        );
    }

    #[test]
    fn derive_title_truncates_on_word_boundary() {
        let long = "Please explain in detail how the borrow checker reasons about \
                    lifetimes when closures capture references";
        let title = derive_space_title(long).unwrap();
        assert!(title.ends_with('…'));
        assert!(title.trim_end_matches('…').chars().count() <= 64);
        // Word-boundary: must not end mid-word.
        assert!(long.starts_with(title.trim_end_matches('…')));
        assert!(
            title
                .trim_end_matches('…')
                .ends_with(|c: char| !c.is_whitespace())
        );
        let kept = title.trim_end_matches('…');
        assert!(
            long[kept.len()..].starts_with(' '),
            "cut mid-word: {title:?}"
        );
    }

    #[test]
    fn derive_title_rejects_empty_and_marker_only() {
        assert_eq!(derive_space_title(""), None);
        assert_eq!(derive_space_title("   \n  "), None);
        assert_eq!(derive_space_title("###"), None);
    }

    #[test]
    fn derive_title_hard_cuts_single_long_word() {
        let word = "a".repeat(100);
        let title = derive_space_title(&word).unwrap();
        assert_eq!(title.chars().count(), 65); // 64 + ellipsis
        assert!(title.ends_with('…'));
    }

    #[test]
    fn snippet_collapses_whitespace_and_truncates() {
        assert_eq!(
            snippet_of("first line\nsecond   line"),
            Some("first line second line".to_string())
        );
        assert_eq!(snippet_of("  \n \t "), None);
        let long = "word ".repeat(60);
        let snippet = snippet_of(&long).unwrap();
        assert!(snippet.ends_with('…'));
        assert!(snippet.trim_end_matches('…').chars().count() <= 120);
    }

    // --- Post-tree flattening (build_post_tree) ----------------------------

    /// A tip post action with sensible defaults; override the fields a test
    /// cares about.
    fn post_action(action_id: &str, item_id: &str, created_at: i64) -> db::PostActionRow {
        db::PostActionRow {
            action_id: action_id.into(),
            item_id: item_id.into(),
            participant_kind: "human".into(),
            participant_label: "user".into(),
            action_type: "user_input".into(),
            model: None,
            credits_consumed: None,
            generation: 0,
            created_at,
        }
    }

    fn reply_edge(action_id: &str, antecedent: &str) -> db::AntecedentEdgeRow {
        db::AntecedentEdgeRow {
            action_id: action_id.into(),
            antecedent_action_id: antecedent.into(),
            // Tests default to an un-edited antecedent: its item's tip is the
            // antecedent itself (what the SQL resolves for gen-0 targets).
            antecedent_current_action_id: Some(antecedent.into()),
            ordinal: 0,
            relation: "reply".into(),
            range_start: None,
            range_end: None,
            annotation: None,
        }
    }

    fn text_block(action_id: &str, text: &str) -> db::PostBlockRow {
        db::PostBlockRow {
            action_id: action_id.into(),
            ordinal: 0,
            block_type: "text".into(),
            text_content: Some(text.into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }
    }

    /// A linear reply chain stays flat: every post is on the spine (depth 0).
    #[test]
    fn build_tree_linear_chain_is_flat() {
        // u1 <- i1 <- u2 <- i2 (each replies to the prior tail).
        let mut a1 = post_action("u1", "iu1", 1);
        a1.action_type = "user_input".into();
        let mut a2 = post_action("i1", "ii1", 2);
        a2.action_type = "inference".into();
        a2.participant_kind = "agent".into();
        a2.participant_label = "kimi".into();
        a2.model = Some("kimi".into());
        let mut a3 = post_action("u2", "iu2", 3);
        a3.action_type = "user_input".into();
        let mut a4 = post_action("i2", "ii2", 4);
        a4.action_type = "inference".into();

        let data = db::SpaceTreeData {
            actions: vec![a1, a2, a3, a4],
            blocks: vec![
                text_block("u1", "hello"),
                text_block("i1", "hi there"),
                text_block("u2", "more"),
                text_block("i2", "ok"),
            ],
            edges: vec![
                reply_edge("i1", "u1"),
                reply_edge("u2", "i1"),
                reply_edge("i2", "u2"),
            ],
        };

        let tree = build_post_tree(data);
        let ids: Vec<&str> = tree.iter().map(|n| n.action_id.as_str()).collect();
        assert_eq!(ids, vec!["u1", "i1", "u2", "i2"]);
        assert!(tree.iter().all(|n| n.depth == 0 && !n.is_branch));

        let root = &tree[0];
        assert_eq!(root.parent_action_id, None);
        assert_eq!(root.relation, None);
        assert_eq!(root.blocks, vec![text_block_dto("hello")]);

        let inf = &tree[1];
        assert_eq!(inf.parent_action_id.as_deref(), Some("u1"));
        assert_eq!(inf.relation.as_deref(), Some("reply"));
        assert_eq!(inf.participant.kind, "agent");
        assert_eq!(inf.model.as_deref(), Some("kimi"));
    }

    fn text_block_dto(text: &str) -> PostBlock {
        PostBlock {
            block_type: "text".into(),
            text: Some(text.into()),
            tool_name: None,
            tool_call_id: None,
            data: None,
        }
    }

    /// When a post has more than one reply, the first (chronological) reply
    /// continues the spine and later replies become indented branches; the walk
    /// emits the first reply's whole subtree before the branch.
    #[test]
    fn build_tree_first_reply_spine_later_replies_branch() {
        // root r; replies a (t=2) then b (t=3). a has its own reply a1 (t=4).
        // b has its own reply b1 (t=5).
        let data = db::SpaceTreeData {
            actions: vec![
                post_action("r", "ir", 1),
                post_action("a", "ia", 2),
                post_action("b", "ib", 3),
                post_action("a1", "ia1", 4),
                post_action("b1", "ib1", 5),
            ],
            blocks: vec![],
            edges: vec![
                reply_edge("a", "r"),
                reply_edge("b", "r"),
                reply_edge("a1", "a"),
                reply_edge("b1", "b"),
            ],
        };

        let tree = build_post_tree(data);
        let shape: Vec<(&str, usize, bool)> = tree
            .iter()
            .map(|n| (n.action_id.as_str(), n.depth, n.is_branch))
            .collect();
        // r spine; a is first reply (spine); a1 continues a's spine; THEN b is
        // the later reply (branch, depth 1) with b1 continuing its sub-spine.
        assert_eq!(
            shape,
            vec![
                ("r", 0, false),
                ("a", 0, false),
                ("a1", 0, false),
                ("b", 1, true),
                ("b1", 1, false),
            ]
        );
    }

    /// A branch off a branch indents again (only genuine branch points narrow;
    /// the linear spine never does).
    #[test]
    fn build_tree_nested_branches_indent_cumulatively() {
        // r -> a (spine). a has replies a1 (spine) and a2 (branch @1).
        // a2 has replies a2a (spine @1) and a2b (branch @2).
        let data = db::SpaceTreeData {
            actions: vec![
                post_action("r", "ir", 1),
                post_action("a", "ia", 2),
                post_action("a1", "ia1", 3),
                post_action("a2", "ia2", 4),
                post_action("a2a", "ia2a", 5),
                post_action("a2b", "ia2b", 6),
            ],
            blocks: vec![],
            edges: vec![
                reply_edge("a", "r"),
                reply_edge("a1", "a"),
                reply_edge("a2", "a"),
                reply_edge("a2a", "a2"),
                reply_edge("a2b", "a2"),
            ],
        };

        let tree = build_post_tree(data);
        let shape: Vec<(&str, usize)> = tree
            .iter()
            .map(|n| (n.action_id.as_str(), n.depth))
            .collect();
        assert_eq!(
            shape,
            vec![
                ("r", 0),
                ("a", 0),
                ("a1", 0),
                ("a2", 1),
                ("a2a", 1),
                ("a2b", 2),
            ]
        );
    }

    /// generation_count reflects the item's total generations (an edit appends a
    /// generation; only the tip is fetched, carrying the derived generation).
    #[test]
    fn build_tree_carries_generation_count() {
        let mut tip = post_action("u1-gen2", "item-u1", 5);
        tip.generation = 2; // gen-0 + gen-1 superseded; this tip is generation 2
        let data = db::SpaceTreeData {
            actions: vec![tip],
            blocks: vec![text_block("u1-gen2", "edited text")],
            edges: vec![],
        };

        let tree = build_post_tree(data);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].generation, 2);
        assert_eq!(tree[0].generation_count, 3);
        assert!(tree[0].is_current);
    }

    /// reference edges become PostReference entries (the quote chip), distinct
    /// from the structural reply parent.
    #[test]
    fn build_tree_collects_reference_edges() {
        let data = db::SpaceTreeData {
            actions: vec![post_action("r", "ir", 1), post_action("a", "ia", 2)],
            blocks: vec![],
            edges: vec![
                reply_edge("a", "r"),
                db::AntecedentEdgeRow {
                    action_id: "a".into(),
                    antecedent_action_id: "r".into(),
                    antecedent_current_action_id: Some("r".into()),
                    ordinal: 1,
                    relation: "reference".into(),
                    range_start: Some(0),
                    range_end: Some(5),
                    annotation: Some("see here".into()),
                },
            ],
        };

        let tree = build_post_tree(data);
        let a = tree.iter().find(|n| n.action_id == "a").unwrap();
        assert_eq!(a.parent_action_id.as_deref(), Some("r"));
        assert_eq!(a.references.len(), 1);
        assert_eq!(a.references[0].antecedent_action_id, "r");
        assert_eq!(a.references[0].range_start, Some(0));
        assert_eq!(a.references[0].annotation.as_deref(), Some("see here"));
    }

    /// A reply edge whose target isn't a resolved post (e.g. a non-tip
    /// generation) is treated as absent, leaving the post a root — the v1
    /// raw-antecedent behavior (re-rooting is a 5.4 addition).
    #[test]
    fn build_tree_dangling_reply_parent_becomes_root() {
        let data = db::SpaceTreeData {
            actions: vec![post_action("a", "ia", 2)],
            blocks: vec![],
            edges: vec![reply_edge("a", "missing-non-tip")],
        };

        let tree = build_post_tree(data);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].parent_action_id, None);
        assert_eq!(tree[0].relation, None);
        assert_eq!(tree[0].depth, 0);
    }

    /// Multiple thread roots are siblings of a virtual root: the first root is
    /// on the spine, later roots are branches.
    #[test]
    fn build_tree_multiple_roots_first_spine_rest_branch() {
        let data = db::SpaceTreeData {
            actions: vec![post_action("r1", "ir1", 1), post_action("r2", "ir2", 2)],
            blocks: vec![],
            edges: vec![],
        };

        let tree = build_post_tree(data);
        let shape: Vec<(&str, usize, bool)> = tree
            .iter()
            .map(|n| (n.action_id.as_str(), n.depth, n.is_branch))
            .collect();
        assert_eq!(shape, vec![("r1", 0, false), ("r2", 1, true)]);
    }
}
