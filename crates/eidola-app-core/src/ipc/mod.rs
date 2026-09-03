//! The local control protocol — one long-lived Eidola process, spoken to by
//! another process running as the same user.
//!
//! The local database is single-writer and says so loudly ([`AppError::DatabaseInUse`]):
//! a second process cannot simply open the profile. So the process that *does*
//! hold the lock answers for it, over a Unix domain socket in the data
//! directory. This module is the wire contract for that conversation; the
//! listener that binds the socket belongs to the process that owns the lock
//! (the app), and a client library lives with whoever dials it.
//!
//! ## Frames
//!
//! Newline-delimited JSON, one frame per line, JSON-RPC-*shaped* but bespoke —
//! streaming is first-class here rather than a contortion:
//!
//! ```text
//! → {"v":1,"id":7,"verb":"chat.stream","params":{…}}
//! ← {"v":1,"id":7,"kind":"chunk","data":{…}}   (0..n)
//! ← {"v":1,"id":7,"kind":"end","data":{…}}     (terminal)
//! ← {"v":1,"id":7,"kind":"err","error":{"type":"InsufficientBalance",…}}  (terminal)
//! ```
//!
//! Exactly one terminal frame per request id. `id` is the caller's, echoed
//! back; [`NO_REQUEST`] is the reserved id for a refusal that could not be
//! correlated with one (a line that did not parse at all).
//!
//! ## Why a `verb` string and not an enum on [`Request`]
//!
//! The frame keeps `verb` as text and `params` as an unread [`serde_json::Value`],
//! and [`Call::parse`] is the typed layer above it. That ordering is
//! load-bearing: a request naming a verb this build does not have must still
//! be answered *on its own id*, and an enum would have failed the whole frame's
//! deserialization with the id inside it. Same for `params` — a bad shape is a
//! typed refusal of one request, never a dead connection.
//!
//! ## Errors are typed, never prose-only
//!
//! [`WireError`] carries an [`AppError`]'s variant name and its fields, so the
//! far side routes on the variant exactly as an in-process caller would, plus a
//! rendered `message` that is always present. A variant the reader's build does
//! not know degrades to [`RemoteError::Unrecognized`] — the message still
//! renders — instead of failing to parse. That is the whole skew story for
//! errors; the version handshake below covers the rest.
//!
//! ## Versioning
//!
//! [`PROTOCOL_VERSION`] is independent of the app version and bumps only on a
//! breaking change to frames or existing verbs — *adding* a verb is free, since
//! an unknown one is already a typed refusal. Every connection opens with
//! [`Call::Hello`], whose result states the protocol the server speaks and the
//! app version it is; a request carrying another `v` is refused by
//! [`ProtocolError::UnsupportedProtocol`], which names both sides.
//!
//! ## What is not here, and why
//!
//! There is **no raw-database verb, and there never will be**. The surface is a
//! capability surface: each verb is a typed wrapper over the same [`AppCore`]
//! method an in-process caller would use, so what has no verb cannot be
//! reached. Trust-bundle mutation (base URL, hardware roots, trusted
//! measurements) is deliberately absent — it is the one surface that rewrites
//! what the client will believe about an enclave, and it stays a same-process
//! operation until there is a per-client consent layer to gate it.
//!
//! [`AppCore`]: crate::AppCore
//! [`AppError::DatabaseInUse`]: crate::error::AppError::DatabaseInUse

use serde::{Deserialize, Serialize};

use crate::error::AppError;

pub mod serve;

pub use serve::serve_connection;

/// The protocol this build speaks. Bumped only for a breaking change to the
/// frames or to an existing verb's shape; new verbs are additive.
pub const PROTOCOL_VERSION: u32 = 1;

/// The largest **request** line the app will read, in bytes.
///
/// A line is buffered until its newline arrives, so without a ceiling a peer
/// that never sends one is an unbounded allocation. 1 MiB is far above any
/// legitimate request (a chat prompt is the largest) and far below anything
/// that hurts.
pub const MAX_FRAME_BYTES: usize = 1024 * 1024;

/// The largest **response** line a caller should be willing to read, in bytes.
///
/// **The two ceilings are deliberately different, and the asymmetry is the
/// point.** The request limit bounds what an arbitrary process on the machine
/// can make the app allocate — that is where the exposure is. A response comes
/// from the app the caller chose to dial, which is already holding that
/// caller's entire profile open; a tight limit there buys nothing and costs
/// something real, because a *legitimate* answer grows with the profile. A
/// listing of every conversation is the case that grows without bound, and at
/// roughly 250 bytes a row the old shared 1 MiB ceiling turned a few thousand
/// conversations into an answer a conforming reader would reject as malformed
/// and hang up on — a working profile, refused by its own client.
///
/// 64 MiB is chosen so that reaching it means something is wrong rather than
/// merely large (it is hundreds of thousands of rows), and
/// [`ProtocolError::ResultTooLarge`] is the honest backstop for that case:
/// **the app never emits a line its own reader would refuse.**
pub const MAX_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// The control socket's name inside a profile's data directory.
///
/// Here rather than with the listener because **both** halves need it and only
/// one of them binds: whoever dials has to find the same file the app created,
/// and a name the client re-spelled for itself would be a second answer to
/// "where is it" waiting to disagree.
pub const SOCKET_NAME: &str = "eidola.sock";

/// Where the control socket for a profile lives.
pub fn socket_path(data_dir: &std::path::Path) -> std::path::PathBuf {
    data_dir.join(SOCKET_NAME)
}

/// A result grows with the profile where a request does not, so holding answers
/// to the request ceiling is what would refuse a working profile. Asserted at
/// compile time rather than in a test: the ordering is the whole reason there
/// are two constants, and a build that inverted it should not exist.
const _: () = assert!(MAX_RESPONSE_BYTES > MAX_FRAME_BYTES);

/// The request id used when a refusal cannot be correlated with a request —
/// a line that did not parse as a frame at all, so no id was recoverable.
pub const NO_REQUEST: u64 = 0;

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

/// One request line.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Request {
    /// Protocol version the caller is speaking.
    pub v: u32,
    /// Caller-assigned id, echoed on every frame answering this request.
    pub id: u64,
    /// The verb, as text — see the module docs for why this is not an enum.
    pub verb: String,
    /// The verb's parameters. Absent is the same as `{}`.
    #[serde(default)]
    pub params: serde_json::Value,
}

impl Request {
    /// Build a request frame for `call` at this build's protocol version.
    pub fn new(id: u64, call: &Call) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            verb: call.verb().to_string(),
            params: call.params(),
        }
    }
}

/// One response line. Every frame carries the id of the request it answers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    pub v: u32,
    pub id: u64,
    #[serde(flatten)]
    pub body: ResponseBody,
}

/// What a response frame says. `chunk` may repeat; `end` and `err` are
/// terminal, and exactly one of them ends a request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum ResponseBody {
    /// An incremental piece of a streaming verb's answer.
    Chunk { data: serde_json::Value },
    /// The verb's result. Terminal.
    End { data: serde_json::Value },
    /// The verb failed. Terminal.
    Err { error: WireError },
}

impl Response {
    /// A `chunk` frame carrying `data`.
    pub fn chunk(id: u64, data: serde_json::Value) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            body: ResponseBody::Chunk { data },
        }
    }

    /// An `end` frame carrying a serializable result.
    pub fn end<T: Serialize>(id: u64, value: &T) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            body: ResponseBody::End {
                // A result type that cannot serialize is a bug in this crate,
                // not a runtime condition — but a panic on a serving thread
                // would take the whole app's socket down over one verb, so it
                // degrades to null and the caller sees a shape it refuses.
                data: serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
            },
        }
    }

    /// An `err` frame carrying a typed failure.
    pub fn err(id: u64, error: WireError) -> Self {
        Self {
            v: PROTOCOL_VERSION,
            id,
            body: ResponseBody::Err { error },
        }
    }
}

// ---------------------------------------------------------------------------
// Verbs
// ---------------------------------------------------------------------------

/// A typed call — the verb plus its parameters, parsed.
///
/// Each variant is a thin wrapper over one [`AppCore`] method. There is no
/// general-purpose verb by design: the surface can only reach what it names.
///
/// [`AppCore`]: crate::AppCore
#[derive(Clone, Debug, PartialEq)]
pub enum Call {
    /// Open the connection: state the protocol and the app version.
    Hello,
    /// `AppCore::list_spaces`.
    SpacesList { include_archived: bool },
    /// `AppCore::account_show`.
    AccountShow,
    /// `AppCore::wallet_credentials`.
    WalletCredentials,
    /// `AppCore::list_backends`.
    BackendList,
    /// `AppCore::chat_stream` — a turn, streamed as `chunk` frames with the
    /// finished [`crate::ChatResult`] as its `end`.
    ChatStream {
        prompt: String,
        /// `None` asks the server for the default template's agent model — the
        /// resolution needs the database, so it belongs on this side.
        model: Option<String>,
        space_id: Option<String>,
    },
}

/// Every verb this build serves, in the order they appear in [`Call`].
///
/// Exposed so a client can tell "this build has no such verb" apart from "the
/// server refused it", and so the tests can assert the two lists agree.
pub const VERBS: &[&str] = &[
    "hello",
    "spaces.list",
    "account.show",
    "wallet.credentials",
    "backend.list",
    "chat.stream",
];

/// Parameters of `spaces.list`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SpacesListParams {
    /// Include archived conversations in the listing.
    #[serde(default)]
    pub include_archived: bool,
}

/// Parameters of `chat.stream`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatStreamParams {
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub space_id: Option<String>,
}

impl Call {
    /// The verb this call names.
    pub fn verb(&self) -> &'static str {
        match self {
            Call::Hello => "hello",
            Call::SpacesList { .. } => "spaces.list",
            Call::AccountShow => "account.show",
            Call::WalletCredentials => "wallet.credentials",
            Call::BackendList => "backend.list",
            Call::ChatStream { .. } => "chat.stream",
        }
    }

    /// The call's parameters, as the frame carries them.
    pub fn params(&self) -> serde_json::Value {
        match self {
            Call::Hello | Call::AccountShow | Call::WalletCredentials | Call::BackendList => {
                serde_json::json!({})
            }
            Call::SpacesList { include_archived } => {
                serde_json::json!({ "include_archived": include_archived })
            }
            Call::ChatStream {
                prompt,
                model,
                space_id,
            } => serde_json::json!({
                "prompt": prompt,
                "model": model,
                "space_id": space_id,
            }),
        }
    }

    /// Parse a verb and its parameters.
    ///
    /// Both failures are typed refusals of *one request*: an unknown verb
    /// (which is what a newer caller's additive verb looks like here) and
    /// parameters that do not fit the verb's shape.
    pub fn parse(verb: &str, params: &serde_json::Value) -> Result<Call, ProtocolError> {
        fn of<T: serde::de::DeserializeOwned>(
            verb: &str,
            params: &serde_json::Value,
        ) -> Result<T, ProtocolError> {
            // `null` is how an omitted `params` arrives; treat it as `{}` so a
            // verb whose fields all have defaults can be called bare.
            let params = if params.is_null() {
                serde_json::Value::Object(Default::default())
            } else {
                params.clone()
            };
            serde_json::from_value(params).map_err(|e| ProtocolError::BadParams {
                verb: verb.to_string(),
                message: e.to_string(),
            })
        }
        /// The check a verb that takes no parameters still owes.
        ///
        /// **A verb with no fields to read is not a verb with nothing to
        /// check.** Constructing the call directly let anything at all through
        /// as `params` — `"hello"` with a bare string succeeded and *completed
        /// the handshake*, which is the one exchange whose whole job is
        /// establishing that both sides agree on the protocol.
        ///
        /// Accepted: **absent (`null`) or any object**. That is the same shape
        /// the parameterized verbs accept, since `of` folds `null` into `{}`
        /// and serde ignores fields a struct does not name — so a newer caller
        /// adding a field to a verb this build reads as parameterless keeps
        /// working, which is how this protocol grows. What is refused is a
        /// `params` that is not an object at all: a string, a number, a
        /// boolean, an array. None of those is a parameter set under any
        /// version, so refusing them forecloses nothing.
        fn none(verb: &str, params: &serde_json::Value) -> Result<(), ProtocolError> {
            if params.is_null() || params.is_object() {
                return Ok(());
            }
            let found = match params {
                serde_json::Value::Bool(_) => "a boolean",
                serde_json::Value::Number(_) => "a number",
                serde_json::Value::String(_) => "a string",
                serde_json::Value::Array(_) => "an array",
                // Both answered `Ok` above.
                serde_json::Value::Null | serde_json::Value::Object(_) => unreachable!(),
            };
            Err(ProtocolError::BadParams {
                verb: verb.to_string(),
                message: format!("expected an object or no parameters, found {found}"),
            })
        }
        match verb {
            "hello" => {
                none(verb, params)?;
                Ok(Call::Hello)
            }
            "spaces.list" => {
                let p: SpacesListParams = of(verb, params)?;
                Ok(Call::SpacesList {
                    include_archived: p.include_archived,
                })
            }
            "account.show" => {
                none(verb, params)?;
                Ok(Call::AccountShow)
            }
            "wallet.credentials" => {
                none(verb, params)?;
                Ok(Call::WalletCredentials)
            }
            "backend.list" => {
                none(verb, params)?;
                Ok(Call::BackendList)
            }
            "chat.stream" => {
                let p: ChatStreamParams = of(verb, params)?;
                Ok(Call::ChatStream {
                    prompt: p.prompt,
                    model: p.model,
                    space_id: p.space_id,
                })
            }
            other => Err(ProtocolError::UnknownVerb {
                verb: other.to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Verb results
// ---------------------------------------------------------------------------

/// The `end` payload of `hello`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HelloResult {
    /// The protocol the answering process speaks.
    pub protocol: u32,
    /// The version of the app answering. Informational — the compatibility
    /// decision is `protocol`, which moves independently.
    pub app_version: String,
}

/// The `end` payload of `spaces.list`.
///
/// An object rather than a bare array so a later field (a cursor, a count) is
/// an additive change instead of a breaking one. Same for every list below.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpacesListResult {
    pub spaces: Vec<crate::SpaceInfo>,
}

/// The `end` payload of `wallet.credentials`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WalletCredentialsResult {
    pub credentials: Vec<crate::CredentialInfo>,
}

/// The `end` payload of `backend.list`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackendListResult {
    pub backends: Vec<crate::backends::BackendInfo>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failure of the protocol itself, as opposed to of the operation asked for.
///
/// These share the `type` namespace with [`AppError`]'s variant names, kept
/// disjoint by `protocol_and_app_error_names_do_not_collide`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(tag = "type")]
pub enum ProtocolError {
    /// The line was not a JSON frame this build can read. Not correlated with
    /// a request — there was no readable id.
    #[error("malformed frame: {message}")]
    MalformedFrame { message: String },

    /// A line exceeded [`MAX_FRAME_BYTES`] before its newline arrived. The
    /// connection is closed after this: the reader cannot tell where the
    /// oversized line ends, so nothing after it can be trusted to be a frame.
    #[error("frame exceeds the {limit}-byte limit")]
    FrameTooLarge { limit: usize },

    /// The caller speaks a protocol version this build does not. Names both
    /// sides so whoever reads it can say which half to update.
    #[error("this app speaks protocol {supported}, the caller asked for {requested}")]
    UnsupportedProtocol { supported: u32, requested: u32 },

    /// The connection asked for something before `hello`. The handshake is
    /// what establishes that both sides speak the same protocol, so it comes
    /// first or the exchange is a guess.
    #[error("say hello first: this connection has not completed the handshake")]
    HandshakeRequired,

    /// The caller reused a request id that is still in flight. Exactly one
    /// terminal frame answers an id, so a second request wearing a live one
    /// would put two terminal frames on it and leave the caller unable to say
    /// which result was which.
    ///
    /// **Answered on [`NO_REQUEST`], not on the duplicate**, and that is the
    /// point rather than an inconvenience: the id named here already belongs to
    /// a request that is still going to answer on it, so a refusal wearing it
    /// would be the second frame this exists to prevent. The id travels as data
    /// instead, so the caller still learns which one it reused. Reusing an id
    /// *after* its request has terminated is ordinary and allowed.
    #[error("request id {duplicate} is already in flight on this connection")]
    DuplicateRequestId { duplicate: u64 },

    /// The caller used [`NO_REQUEST`] as its request id. That id is reserved
    /// for a refusal that answers no request, so a call carrying it would be
    /// indistinguishable from one — and a connection that also produced a
    /// malformed line would have two different answers wearing the same id,
    /// with no way for the caller to tell them apart. Refused before anything
    /// is dispatched, so the ambiguity never reaches the wire.
    #[error("request id {reserved} is reserved for refusals that answer no request")]
    ReservedRequestId { reserved: u64 },

    /// The verb succeeded, and its result does not fit in a response line
    /// ([`MAX_RESPONSE_BYTES`]). Sent **instead of** the oversized frame: a
    /// reader applying the protocol's own limit would refuse that line and
    /// close the connection, reporting a malformed app rather than a large
    /// answer. Nothing was written and nothing is wrong with the profile — the
    /// request has to ask for less.
    #[error("the result of `{verb}` is {bytes} bytes, past the {limit}-byte response limit")]
    ResultTooLarge {
        verb: String,
        bytes: usize,
        limit: usize,
    },

    /// The verb *failed*, and the failure itself does not fit in a response
    /// line ([`MAX_RESPONSE_BYTES`]) — so this stands in for it, naming the
    /// variant that could not be delivered.
    ///
    /// The twin of [`ProtocolError::ResultTooLarge`], and needed for the same
    /// reason: an `err` frame is a terminal frame too, and its text is not this
    /// app's to bound. A failure's `message` is whatever rendered it, and an
    /// upstream's own error body travels inside several [`AppError`] variants
    /// uncapped — so the one frame that says "something went wrong" could
    /// itself be the line the caller's reader refuses, turning a typed failure
    /// into a closed connection.
    ///
    /// **Fixed shape, so the error about the error always fits**: the variant
    /// name is a compile-time constant in every case, and the two unbounded
    /// halves — the rendered sentence and the variant's own fields — are the
    /// ones dropped. What the caller loses is the wording; what it keeps is
    /// that the verb failed, which variant it failed with, and why the rest is
    /// missing.
    #[error("the `{kind}` failure is {bytes} bytes, past the {limit}-byte response limit")]
    ErrorTooLarge {
        kind: String,
        bytes: usize,
        limit: usize,
    },

    /// No such verb in this build. What a newer caller's additive verb looks
    /// like from here, which is why it is a refusal of one request rather than
    /// of the connection.
    #[error("unknown verb `{verb}`")]
    UnknownVerb { verb: String },

    /// The parameters did not fit the verb.
    #[error("bad parameters for `{verb}`: {message}")]
    BadParams { verb: String, message: String },

    /// The app is running but cannot serve this verb — there is no open core
    /// behind the socket. Distinct from a failing operation: nothing was
    /// attempted.
    #[error("the app is running but has no open profile to answer with")]
    Unavailable,
}

impl ProtocolError {
    /// The variant name, as it appears in a [`WireError`]'s `type`.
    pub fn kind(&self) -> &'static str {
        match self {
            ProtocolError::MalformedFrame { .. } => "MalformedFrame",
            ProtocolError::FrameTooLarge { .. } => "FrameTooLarge",
            ProtocolError::UnsupportedProtocol { .. } => "UnsupportedProtocol",
            ProtocolError::HandshakeRequired => "HandshakeRequired",
            ProtocolError::DuplicateRequestId { .. } => "DuplicateRequestId",
            ProtocolError::ReservedRequestId { .. } => "ReservedRequestId",
            ProtocolError::ResultTooLarge { .. } => "ResultTooLarge",
            ProtocolError::ErrorTooLarge { .. } => "ErrorTooLarge",
            ProtocolError::UnknownVerb { .. } => "UnknownVerb",
            ProtocolError::BadParams { .. } => "BadParams",
            ProtocolError::Unavailable => "Unavailable",
        }
    }

    /// Whether the connection can carry on after this refusal.
    ///
    /// Only [`ProtocolError::FrameTooLarge`] is fatal, and for a structural
    /// reason: the reader gave up mid-line, so it no longer knows where the
    /// next frame begins.
    pub fn is_fatal(&self) -> bool {
        matches!(self, ProtocolError::FrameTooLarge { .. })
    }
}

/// One error as it travels: the variant name, a rendered message, and the
/// variant's own fields.
///
/// **`message` is always present**, so a reader that has never heard of the
/// variant still has something honest to show. **`fields` carries the variant's
/// own data**, so a reader that *has* heard of it recovers the typed value and
/// routes on it exactly as an in-process caller would.
///
/// The two are kept in separate objects rather than flattened together for a
/// reason that is not stylistic: a dozen [`AppError`] variants carry a field
/// literally called `message`, whose value is a *part* of the rendered
/// sentence, not the sentence. Flattened, the reader could not tell which one
/// it was holding, and every `Display` that prefixes its field ("network error:
/// …") would double on the next round trip.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WireError {
    /// The variant name — an [`AppError`] variant, or a [`ProtocolError`] one.
    #[serde(rename = "type")]
    pub kind: String,
    /// The error rendered as text. Never omitted.
    pub message: String,
    /// The variant's own fields. Empty for a variant that carries none.
    #[serde(default)]
    pub fields: serde_json::Map<String, serde_json::Value>,
}

impl WireError {
    fn build(kind: &str, message: String, fields: serde_json::Value) -> Self {
        let fields = match fields {
            serde_json::Value::Object(map) => map,
            _ => Default::default(),
        };
        Self {
            kind: kind.to_string(),
            message,
            fields,
        }
    }

    /// Render a protocol failure.
    pub fn from_protocol(e: &ProtocolError) -> Self {
        let mut fields = serde_json::to_value(e).unwrap_or(serde_json::Value::Null);
        // The tag is already carried by `kind`; leaving a second copy in the
        // flattened fields would make the object's `type` ambiguous.
        if let serde_json::Value::Object(map) = &mut fields {
            map.remove("type");
        }
        Self::build(e.kind(), e.to_string(), fields)
    }

    /// Render an app failure with its fields.
    ///
    /// The match is exhaustive on purpose: a new [`AppError`] variant does not
    /// compile until it has decided what it puts on the wire, which is the only
    /// way "typed, never prose-only" survives the next variant.
    pub fn from_app_error(e: &AppError) -> Self {
        use serde_json::json;
        let (kind, fields) = match e {
            AppError::NotConfigured { message } => ("NotConfigured", json!({ "message": message })),
            AppError::SpaceArchived { space_id } => {
                ("SpaceArchived", json!({ "space_id": space_id }))
            }
            AppError::NotJoined { space_id } => ("NotJoined", json!({ "space_id": space_id })),
            AppError::DrivenConversation { space_id } => {
                ("DrivenConversation", json!({ "space_id": space_id }))
            }
            AppError::WrongPostKind { message } => ("WrongPostKind", json!({ "message": message })),
            // The refusal is a typed enum of its own, and this is deliberately
            // the one variant that does not carry its fields: reconstructing it
            // would put a second copy of that enum's shape on the wire for a
            // value whose whole purpose is the sentence it renders to. The
            // reader gets `message` — the sentence — and the variant name.
            AppError::SpawnRefused { .. } => ("SpawnRefused", json!({})),
            AppError::Network { message } => ("Network", json!({ "message": message })),
            AppError::Attestation { message } => ("Attestation", json!({ "message": message })),
            AppError::Server { status, message } => {
                ("Server", json!({ "status": status, "message": message }))
            }
            AppError::Credential { message } => ("Credential", json!({ "message": message })),
            AppError::NoAccount => ("NoAccount", json!({})),
            AppError::InsufficientBalance {
                available,
                required,
            } => (
                "InsufficientBalance",
                json!({ "available": available, "required": required }),
            ),
            AppError::ProvisioningTimeout { message } => {
                ("ProvisioningTimeout", json!({ "message": message }))
            }
            AppError::TermsAcceptanceRequired { message } => {
                ("TermsAcceptanceRequired", json!({ "message": message }))
            }
            AppError::NotAParticipant {
                participant_id,
                action_id,
            } => (
                "NotAParticipant",
                json!({ "participant_id": participant_id, "action_id": action_id }),
            ),
            AppError::Database { message } => ("Database", json!({ "message": message })),
            AppError::DatabaseInUse { pid, message } => {
                ("DatabaseInUse", json!({ "pid": pid, "message": message }))
            }
            AppError::Config { message } => ("Config", json!({ "message": message })),
            AppError::Internal { message } => ("Internal", json!({ "message": message })),
            AppError::LocalModel { message } => ("LocalModel", json!({ "message": message })),
            AppError::Update { message } => ("Update", json!({ "message": message })),
            AppError::ToolLoop { message } => ("ToolLoop", json!({ "message": message })),
            AppError::ResponseTruncated { output_tokens } => (
                "ResponseTruncated",
                json!({ "output_tokens": output_tokens }),
            ),
            AppError::RegenerationInFlight { item_id } => {
                ("RegenerationInFlight", json!({ "item_id": item_id }))
            }
        };
        Self::build(kind, e.to_string(), fields)
    }

    /// Read the error back on the far side.
    ///
    /// Three honest outcomes, and the third is the point: a variant this build
    /// has never heard of is [`RemoteError::Unrecognized`] with its message
    /// intact, never a parse failure and never a guess at which typed variant
    /// it "probably" is.
    pub fn to_remote(&self) -> RemoteError {
        if let Some(e) = self.to_protocol_error() {
            return RemoteError::Protocol(e);
        }
        if let Some(e) = self.to_app_error() {
            return RemoteError::App(e);
        }
        RemoteError::Unrecognized {
            kind: self.kind.clone(),
            message: self.message.clone(),
        }
    }

    fn to_protocol_error(&self) -> Option<ProtocolError> {
        let mut map = self.fields.clone();
        map.insert("type".into(), serde_json::Value::String(self.kind.clone()));
        serde_json::from_value(serde_json::Value::Object(map)).ok()
    }

    /// Reconstruct the typed [`AppError`], where this build knows the variant.
    ///
    /// `None` for a variant it does not know — and for `SpawnRefused`, whose
    /// payload deliberately does not travel (see [`Self::from_app_error`]); the
    /// caller renders `message` for both.
    fn to_app_error(&self) -> Option<AppError> {
        let f = &self.fields;
        let s =
            |k: &str| -> Option<String> { f.get(k).and_then(|v| v.as_str()).map(str::to_string) };
        let i = |k: &str| -> Option<i64> { f.get(k).and_then(serde_json::Value::as_i64) };
        // A field the sender did carry but this build reads as the wrong type
        // is a broken sender, not a skew case; falling back to the rendered
        // message keeps the shape of the answer honest either way.
        let msg = || s("message").unwrap_or_else(|| self.message.clone());
        Some(match self.kind.as_str() {
            "NotConfigured" => AppError::NotConfigured { message: msg() },
            "SpaceArchived" => AppError::SpaceArchived {
                space_id: s("space_id")?,
            },
            "NotJoined" => AppError::NotJoined {
                space_id: s("space_id")?,
            },
            "DrivenConversation" => AppError::DrivenConversation {
                space_id: s("space_id")?,
            },
            "WrongPostKind" => AppError::WrongPostKind { message: msg() },
            "Network" => AppError::Network { message: msg() },
            "Attestation" => AppError::Attestation { message: msg() },
            "Server" => AppError::Server {
                status: u16::try_from(i("status")?).ok()?,
                message: msg(),
            },
            "Credential" => AppError::Credential { message: msg() },
            "NoAccount" => AppError::NoAccount,
            "InsufficientBalance" => AppError::InsufficientBalance {
                available: i("available")?,
                required: i("required")?,
            },
            "ProvisioningTimeout" => AppError::ProvisioningTimeout { message: msg() },
            "TermsAcceptanceRequired" => AppError::TermsAcceptanceRequired { message: msg() },
            "NotAParticipant" => AppError::NotAParticipant {
                participant_id: s("participant_id")?,
                action_id: s("action_id")?,
            },
            "Database" => AppError::Database { message: msg() },
            "DatabaseInUse" => AppError::DatabaseInUse {
                pid: f
                    .get("pid")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|p| u32::try_from(p).ok()),
                message: msg(),
            },
            "Config" => AppError::Config { message: msg() },
            "Internal" => AppError::Internal { message: msg() },
            "LocalModel" => AppError::LocalModel { message: msg() },
            "Update" => AppError::Update { message: msg() },
            "ToolLoop" => AppError::ToolLoop { message: msg() },
            "ResponseTruncated" => AppError::ResponseTruncated {
                output_tokens: i("output_tokens"),
            },
            "RegenerationInFlight" => AppError::RegenerationInFlight {
                item_id: s("item_id")?,
            },
            _ => return None,
        })
    }
}

impl From<&AppError> for WireError {
    fn from(e: &AppError) -> Self {
        WireError::from_app_error(e)
    }
}

impl From<&ProtocolError> for WireError {
    fn from(e: &ProtocolError) -> Self {
        WireError::from_protocol(e)
    }
}

/// A failure as the far side of the socket understands it.
///
/// Not `PartialEq`: [`AppError`] is not, deliberately — comparing two failures
/// for equality is a thing tests want and callers should not do.
#[derive(Clone, Debug, thiserror::Error)]
pub enum RemoteError {
    /// A typed app failure, reconstructed — route on it exactly as in-process.
    #[error("{0}")]
    App(AppError),
    /// The protocol itself refused.
    #[error("{0}")]
    Protocol(ProtocolError),
    /// A failure this build does not have a type for — a newer app's variant.
    /// The message is the app's own rendering of it.
    #[error("{message}")]
    Unrecognized { kind: String, message: String },
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------

/// Encode one frame as its NDJSON line, newline included.
pub fn encode_line<T: Serialize>(frame: &T) -> Vec<u8> {
    let mut buf = serde_json::to_vec(frame).unwrap_or_else(|_| b"{}".to_vec());
    buf.push(b'\n');
    buf
}

/// Decode one request line.
///
/// Trailing `\r` is tolerated so a line written by a CRLF-minded tool still
/// reads; everything else that is not a frame is [`ProtocolError::MalformedFrame`].
pub fn decode_request(line: &[u8]) -> Result<Request, ProtocolError> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    serde_json::from_slice(line).map_err(|e| ProtocolError::MalformedFrame {
        message: e.to_string(),
    })
}

/// Decode one response line.
pub fn decode_response(line: &[u8]) -> Result<Response, ProtocolError> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    serde_json::from_slice(line).map_err(|e| ProtocolError::MalformedFrame {
        message: e.to_string(),
    })
}

/// The encoded terminal line for a verb that succeeded.
///
/// Encodes the `end` frame and, when that line is past `limit`, answers with
/// [`ProtocolError::ResultTooLarge`] instead. **Together with
/// [`terminal_error_line`] this is the whole of where a terminal frame becomes
/// bytes**, which is what makes the guarantee hold for every verb at once
/// rather than for whichever list somebody remembered: the app never writes a
/// line its own reader would reject.
///
/// Chunk frames are held to the same ceiling by [`chunk_lines`], which **splits**
/// rather than refuses: a chunk is a piece of an answer, so replacing one with a
/// refusal would truncate that answer and put a second terminal frame behind it.
pub fn terminal_line(id: u64, verb: &str, data: serde_json::Value, limit: usize) -> Vec<u8> {
    let line = encode_line(&Response {
        v: PROTOCOL_VERSION,
        id,
        body: ResponseBody::End { data },
    });
    if line.len() <= limit {
        return line;
    }
    // Encoded rather than measured, and that is the base case rather than an
    // omission: the substitute is of fixed shape over a verb name this build
    // spells as a constant, so it fits any ceiling worth having. Measuring it
    // would only be asking what to substitute for the substitute.
    encode_line(&Response::err(
        id,
        WireError::from_protocol(&ProtocolError::ResultTooLarge {
            verb: verb.to_string(),
            bytes: line.len(),
            limit,
        }),
    ))
}

/// The encoded terminal line for a verb that failed — and for every refusal,
/// which is the same frame arriving by a shorter road.
///
/// The `err` half of [`terminal_line`], and it exists because an `err` frame is
/// a terminal frame with the same ceiling and *less* control over its own size:
/// a result is this app's own rendering of its own data, while a failure's
/// `message` can be an upstream's error body travelling uncapped inside an
/// [`AppError`]. Encoded directly, one of those is a line the caller's reader
/// refuses — so the typed failure the protocol exists to deliver arrives as a
/// closed connection instead.
///
/// Past `limit` it answers with [`ProtocolError::ErrorTooLarge`], which is of
/// fixed shape and therefore always fits: an error about an error that could
/// itself be too large would be no backstop at all.
pub fn terminal_error_line(id: u64, error: WireError, limit: usize) -> Vec<u8> {
    // Only the variant name is kept back, because only the variant name is
    // needed if the frame turns out not to fit — copying the whole error to
    // measure it would double the very allocation being judged.
    let kind = error.kind.clone();
    let line = encode_line(&Response::err(id, error));
    if line.len() <= limit {
        return line;
    }
    // The base case, for the same reason `terminal_line`'s substitute is: the
    // variant name is a constant in every build of this enum, so the frame that
    // says the failure did not fit is one that always does.
    encode_line(&Response::err(
        id,
        WireError::from_protocol(&ProtocolError::ErrorTooLarge {
            kind,
            bytes: line.len(),
            limit,
        }),
    ))
}

/// The most bytes JSON escaping can make of one input byte: a control
/// character becomes ` `, six for one.
///
/// Used as an upper bound rather than measured per piece, which is what lets
/// [`chunk_lines`] size a split without encoding a candidate and retrying. It
/// over-splits pathological text and never under-splits, and the case it costs
/// anything in is already the case nothing sensible produced.
const JSON_ESCAPE_WORST_CASE: usize = 6;

/// The encoded `chunk` lines for one streamed event — one where it fits, and
/// **several where it does not**.
///
/// A chunk is under the same ceiling as a terminal frame, for the same reason:
/// the app must never write a line its own reader would refuse. But a chunk is
/// not a terminal frame, and the answer that works for those is wrong here —
/// replacing an oversized chunk with a refusal would truncate an answer and
/// then put a second terminal frame behind it. **Splitting costs nothing at
/// all**, because a chunk is a piece of text a reader is already appending to a
/// buffer: several frames concatenate to exactly what one would have said, in
/// order, through the one writer that drains the outbox.
///
/// This is not a hypothetical ceiling. A chunk's size is decided by whatever
/// backend the profile is pointed at, not by this app — the SSE reader buffers
/// an event until its boundary with no cap of its own — so a single delta past
/// the limit is representable on the wire, and unsplit it makes a conforming
/// reader report [`ProtocolError::FrameTooLarge`] and hang up on a turn that
/// then finishes and persists perfectly well.
///
/// **The match over the event is exhaustive on purpose.** Every variant that
/// reaches a chunk today carries text and nothing else, which is what makes
/// splitting honest — bytes of a string concatenate; a structural payload
/// (a tool-call delta, say) would not, and must decide what it does here rather
/// than inherit an answer written for prose. Tool rounds are invisible to this
/// protocol in v1, so no such variant exists yet.
///
/// Pieces end on character boundaries, so every frame is a valid JSON string
/// and a reader never sees half a character.
pub fn chunk_lines(id: u64, event: &crate::ChatStreamEvent, limit: usize) -> Vec<Vec<u8>> {
    let whole = chunk_line(id, event);
    if whole.len() <= limit {
        return vec![whole];
    }
    // Measured rather than counted: the scaffolding is the frame plus the
    // variant's own tag, and asking for it with an empty text is the one way
    // that cannot drift from what the encoder actually writes.
    let overhead = chunk_line(id, &with_text(event, "")).len();
    let budget = limit.saturating_sub(overhead) / JSON_ESCAPE_WORST_CASE;

    let mut rest = text_of(event);
    let mut lines = Vec::new();
    while !rest.is_empty() {
        let (head, tail) = rest.split_at(prefix_within(rest, budget));
        lines.push(chunk_line(id, &with_text(event, head)));
        rest = tail;
    }
    lines
}

/// One `chunk` frame's bytes, whatever its size.
fn chunk_line(id: u64, event: &crate::ChatStreamEvent) -> Vec<u8> {
    let data = serde_json::to_value(event).unwrap_or(serde_json::Value::Null);
    encode_line(&Response::chunk(id, data))
}

/// The text a streamed event carries.
fn text_of(event: &crate::ChatStreamEvent) -> &str {
    match event {
        crate::ChatStreamEvent::ReasoningDelta(text)
        | crate::ChatStreamEvent::ContentDelta(text) => text,
    }
}

/// The same kind of event, saying something shorter.
fn with_text(event: &crate::ChatStreamEvent, text: &str) -> crate::ChatStreamEvent {
    match event {
        crate::ChatStreamEvent::ReasoningDelta(_) => {
            crate::ChatStreamEvent::ReasoningDelta(text.to_string())
        }
        crate::ChatStreamEvent::ContentDelta(_) => {
            crate::ChatStreamEvent::ContentDelta(text.to_string())
        }
    }
}

/// How much of `s` fits in `budget` bytes, ending on a character boundary.
///
/// **Never zero for a non-empty string.** A budget too small to hold even one
/// character still has to make progress or the split cannot terminate; the
/// frame that produces may exceed the limit, which is the honest outcome when
/// nothing at all can fit under it. Unreachable at the real ceiling, where the
/// budget is megabytes.
fn prefix_within(s: &str, budget: usize) -> usize {
    let mut end = 0;
    for (i, c) in s.char_indices() {
        let next = i + c.len_utf8();
        if next > budget {
            break;
        }
        end = next;
    }
    if end == 0 {
        end = s.chars().next().map_or(0, char::len_utf8);
    }
    end
}

/// A bounded NDJSON line reader.
///
/// Reads whole lines from any async source, refusing one that grows past
/// [`MAX_FRAME_BYTES`] before its newline instead of buffering it. Blank lines
/// are skipped rather than reported — a stray newline is not a frame, and
/// answering one with an error would turn a harmless keepalive into a fault.
pub struct FrameReader<R> {
    inner: R,
    buf: Vec<u8>,
    limit: usize,
}

impl<R: tokio::io::AsyncBufRead + Unpin> FrameReader<R> {
    /// A reader over `inner`, bounded at [`MAX_FRAME_BYTES`].
    pub fn new(inner: R) -> Self {
        Self::with_limit(inner, MAX_FRAME_BYTES)
    }

    /// A reader for the **answers** side of the conversation, bounded at
    /// [`MAX_RESPONSE_BYTES`].
    ///
    /// Whoever dials uses this rather than [`Self::new`]: a result grows with
    /// the profile, and holding an answer to the request ceiling would refuse
    /// a legitimate listing as though the app had malfunctioned.
    pub fn for_responses(inner: R) -> Self {
        Self::with_limit(inner, MAX_RESPONSE_BYTES)
    }

    /// A reader with an explicit ceiling (the tests use a small one; nothing
    /// in production should).
    pub fn with_limit(inner: R, limit: usize) -> Self {
        Self {
            inner,
            buf: Vec::new(),
            limit,
        }
    }

    /// The next frame's bytes, or `None` at a clean end of stream.
    ///
    /// A partial final line — bytes with no newline before EOF — is *not* a
    /// frame and is dropped rather than parsed: half a frame that happens to
    /// be valid JSON is the one case where guessing would act on something
    /// nobody finished saying.
    pub async fn next_line(&mut self) -> Result<Option<&[u8]>, ProtocolError> {
        use tokio::io::AsyncBufReadExt;
        self.buf.clear();
        loop {
            let available = match self.inner.fill_buf().await {
                Ok(b) => b,
                // A peer that vanished mid-read is an end of stream, not a
                // protocol fault: nothing was mis-said, the other end is gone.
                Err(_) => return Ok(None),
            };
            if available.is_empty() {
                return Ok(None);
            }
            match available.iter().position(|b| *b == b'\n') {
                Some(idx) => {
                    let take = idx + 1;
                    if self.buf.len() + idx > self.limit {
                        return Err(ProtocolError::FrameTooLarge { limit: self.limit });
                    }
                    self.buf.extend_from_slice(&available[..idx]);
                    self.inner.consume(take);
                    if self.buf.is_empty() {
                        continue;
                    }
                    return Ok(Some(&self.buf));
                }
                None => {
                    let len = available.len();
                    if self.buf.len() + len > self.limit {
                        return Err(ProtocolError::FrameTooLarge { limit: self.limit });
                    }
                    self.buf.extend_from_slice(available);
                    self.inner.consume(len);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_socket_sits_beside_the_profile_it_speaks_for() {
        assert_eq!(
            socket_path(std::path::Path::new("/data/eidola")),
            std::path::Path::new("/data/eidola/eidola.sock")
        );
    }

    #[test]
    fn a_request_frame_looks_like_the_protocol_says() {
        let req = Request::new(
            7,
            &Call::ChatStream {
                prompt: "hi".into(),
                model: None,
                space_id: None,
            },
        );
        let line = String::from_utf8(encode_line(&req)).unwrap();
        let value: serde_json::Value = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(value["v"], 1);
        assert_eq!(value["id"], 7);
        assert_eq!(value["verb"], "chat.stream");
        assert_eq!(value["params"]["prompt"], "hi");
        assert!(line.ends_with('\n'));
    }

    #[test]
    fn response_frames_carry_kind_beside_the_id() {
        let chunk = Response::chunk(3, serde_json::json!({"type": "content_delta"}));
        let value: serde_json::Value = serde_json::from_slice(&encode_line(&chunk)).unwrap();
        assert_eq!(value["kind"], "chunk");
        assert_eq!(value["id"], 3);
        assert_eq!(value["data"]["type"], "content_delta");

        let end = Response::end(
            3,
            &HelloResult {
                protocol: 1,
                app_version: "0.0.0".into(),
            },
        );
        let value: serde_json::Value = serde_json::from_slice(&encode_line(&end)).unwrap();
        assert_eq!(value["kind"], "end");
        assert_eq!(value["data"]["protocol"], 1);

        let err = Response::err(3, WireError::from_app_error(&AppError::NoAccount));
        let value: serde_json::Value = serde_json::from_slice(&encode_line(&err)).unwrap();
        assert_eq!(value["kind"], "err");
        assert_eq!(value["error"]["type"], "NoAccount");
    }

    #[test]
    fn every_frame_round_trips() {
        for body in [
            ResponseBody::Chunk {
                data: serde_json::json!({"a": 1}),
            },
            ResponseBody::End {
                data: serde_json::json!({"b": 2}),
            },
            ResponseBody::Err {
                error: WireError::from_protocol(&ProtocolError::HandshakeRequired),
            },
        ] {
            let frame = Response {
                v: PROTOCOL_VERSION,
                id: 11,
                body,
            };
            let line = encode_line(&frame);
            let back = decode_response(&line[..line.len() - 1]).expect("decode");
            assert_eq!(back.id, 11);
            assert_eq!(
                serde_json::to_value(&back.body).unwrap(),
                serde_json::to_value(&frame.body).unwrap()
            );
        }
    }

    #[test]
    fn every_call_round_trips_through_its_verb_and_params() {
        let calls = [
            Call::Hello,
            Call::SpacesList {
                include_archived: true,
            },
            Call::AccountShow,
            Call::WalletCredentials,
            Call::BackendList,
            Call::ChatStream {
                prompt: "hello".into(),
                model: Some("m@b".into()),
                space_id: Some("sp".into()),
            },
        ];
        for call in &calls {
            let req = Request::new(1, call);
            let back = Call::parse(&req.verb, &req.params).expect("parse");
            assert_eq!(&back, call);
        }
        // The exported list and the parser agree — a verb added to one and not
        // the other is the drift this catches.
        let named: Vec<&str> = calls.iter().map(|c| c.verb()).collect();
        assert_eq!(named, VERBS);
    }

    #[test]
    fn an_omitted_params_object_is_an_empty_one() {
        let line = br#"{"v":1,"id":1,"verb":"spaces.list"}"#;
        let req = decode_request(line).expect("decode");
        assert_eq!(
            Call::parse(&req.verb, &req.params).unwrap(),
            Call::SpacesList {
                include_archived: false
            }
        );
    }

    #[test]
    fn an_unknown_verb_is_a_refusal_that_names_it() {
        let err = Call::parse("spaces.destroy", &serde_json::json!({})).unwrap_err();
        assert_eq!(
            err,
            ProtocolError::UnknownVerb {
                verb: "spaces.destroy".into()
            }
        );
        assert!(!err.is_fatal());
    }

    #[test]
    fn parameters_of_the_wrong_shape_name_the_verb() {
        let err = Call::parse("chat.stream", &serde_json::json!({"prompt": 5})).unwrap_err();
        match err {
            ProtocolError::BadParams { verb, .. } => assert_eq!(verb, "chat.stream"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn a_line_that_is_not_a_frame_is_malformed_not_a_panic() {
        for line in [
            &b"not json"[..],
            &b"[]"[..],
            &b"{}"[..],
            &b"{\"v\":1}"[..],
            &b"\xff\xfe"[..],
        ] {
            let err = decode_request(line).unwrap_err();
            assert!(matches!(err, ProtocolError::MalformedFrame { .. }));
            assert!(!err.is_fatal());
        }
    }

    #[test]
    fn a_carriage_return_before_the_newline_still_reads() {
        let req = decode_request(b"{\"v\":1,\"id\":2,\"verb\":\"hello\"}\r").expect("decode");
        assert_eq!(req.id, 2);
    }

    #[test]
    fn protocol_and_app_error_names_do_not_collide() {
        let protocol = [
            ProtocolError::MalformedFrame {
                message: String::new(),
            },
            ProtocolError::FrameTooLarge { limit: 1 },
            ProtocolError::UnsupportedProtocol {
                supported: 1,
                requested: 2,
            },
            ProtocolError::HandshakeRequired,
            ProtocolError::DuplicateRequestId { duplicate: 7 },
            ProtocolError::ReservedRequestId {
                reserved: NO_REQUEST,
            },
            ProtocolError::ResultTooLarge {
                verb: "spaces.list".into(),
                bytes: 2,
                limit: 1,
            },
            ProtocolError::UnknownVerb {
                verb: String::new(),
            },
            ProtocolError::BadParams {
                verb: String::new(),
                message: String::new(),
            },
            ProtocolError::Unavailable,
        ];
        for e in &protocol {
            let wire = WireError::from_protocol(e);
            match wire.to_remote() {
                RemoteError::Protocol(got) => assert_eq!(&got, e),
                other => panic!("`{}` did not come back typed: {other:?}", wire.kind),
            }
            // A protocol name must not also read as an app error, or the
            // reader would route a refusal as an operation failure.
            assert!(
                wire.to_app_error().is_none(),
                "`{}` reads as both",
                wire.kind
            );
        }
    }

    #[test]
    fn a_typed_app_error_survives_the_round_trip_with_its_fields() {
        let cases = [
            AppError::NoAccount,
            AppError::InsufficientBalance {
                available: 12,
                required: 34,
            },
            AppError::DatabaseInUse {
                pid: Some(4242),
                message: "another Eidola".into(),
            },
            AppError::DatabaseInUse {
                pid: None,
                message: "another Eidola".into(),
            },
            AppError::Server {
                status: 503,
                message: "upstream".into(),
            },
            AppError::NotAParticipant {
                participant_id: "p".into(),
                action_id: "a".into(),
            },
            AppError::ResponseTruncated {
                output_tokens: Some(99),
            },
            AppError::ResponseTruncated {
                output_tokens: None,
            },
            AppError::SpaceArchived {
                space_id: "sp".into(),
            },
            AppError::NotJoined {
                space_id: "sp".into(),
            },
            AppError::DrivenConversation {
                space_id: "sp".into(),
            },
            AppError::RegenerationInFlight {
                item_id: "it".into(),
            },
            AppError::Attestation {
                message: "bad measurement".into(),
            },
        ];
        for e in &cases {
            let wire = WireError::from_app_error(e);
            assert_eq!(wire.message, e.to_string(), "message is always rendered");
            let line = encode_line(&wire);
            let back: WireError = serde_json::from_slice(&line[..line.len() - 1]).unwrap();
            match back.to_remote() {
                RemoteError::App(got) => {
                    assert_eq!(format!("{got:?}"), format!("{e:?}"));
                }
                other => panic!("`{}` did not come back typed: {other:?}", wire.kind),
            }
        }
    }

    #[test]
    fn a_variant_this_build_does_not_know_keeps_its_message() {
        let line =
            br#"{"type":"SomethingNewer","message":"the newer app said this","fields":{"extra":7}}"#;
        let wire: WireError = serde_json::from_slice(line).expect("decode");
        let remote = wire.to_remote();
        match &remote {
            RemoteError::Unrecognized { kind, message } => {
                assert_eq!(kind, "SomethingNewer");
                assert_eq!(message, "the newer app said this");
            }
            other => panic!("unexpected: {other:?}"),
        }
        assert_eq!(remote.to_string(), "the newer app said this");
    }

    #[test]
    fn a_refusal_whose_payload_does_not_travel_still_renders() {
        let e = AppError::SpawnRefused {
            refusal: crate::subspaces::SpawnRefusal::EmptyBrief,
        };
        let wire = WireError::from_app_error(&e);
        assert_eq!(wire.kind, "SpawnRefused");
        assert_eq!(wire.message, e.to_string());
        match wire.to_remote() {
            RemoteError::Unrecognized { kind, message } => {
                assert_eq!(kind, "SpawnRefused");
                assert_eq!(message, e.to_string());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn an_added_field_on_a_known_variant_does_not_break_the_reader() {
        // Forward compatibility in the small: a newer app adding a field to a
        // variant this build knows must still read as that variant.
        let line = br#"{"type":"InsufficientBalance","message":"m",
            "fields":{"available":1,"required":2,"currency":"credits"}}"#;
        let wire: WireError = serde_json::from_slice(line).expect("decode");
        match wire.to_remote() {
            RemoteError::App(AppError::InsufficientBalance {
                available,
                required,
            }) => {
                assert_eq!((available, required), (1, 2));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn a_result_that_fits_is_sent_as_it_is() {
        let line = terminal_line(3, "spaces.list", serde_json::json!({"spaces": []}), 1024);
        let frame = decode_response(&line[..line.len() - 1]).expect("decode");
        assert_eq!(frame.id, 3);
        assert!(matches!(frame.body, ResponseBody::End { .. }));
    }

    #[test]
    fn a_result_too_big_to_read_is_refused_instead_of_written() {
        // The failure this prevents: a legitimate answer emitted as a line the
        // protocol's own reader rejects, which a caller reports as a broken app
        // and answers by closing the connection.
        let big = serde_json::json!({ "spaces": vec!["x"; 100] });
        let limit = 64;
        let line = terminal_line(5, "spaces.list", big, limit);
        assert!(
            line.len() <= MAX_FRAME_BYTES,
            "the refusal itself has to be readable"
        );
        let frame = decode_response(&line[..line.len() - 1]).expect("decode");
        assert_eq!(frame.id, 5, "the refusal answers the request that asked");
        match frame.body {
            ResponseBody::Err { error } => match error.to_remote() {
                RemoteError::Protocol(ProtocolError::ResultTooLarge {
                    verb,
                    bytes,
                    limit: reported,
                }) => {
                    assert_eq!(verb, "spaces.list", "the refusal names what was asked");
                    assert!(bytes > reported);
                    assert_eq!(reported, limit);
                }
                other => panic!("unexpected: {other:?}"),
            },
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn a_failure_that_fits_travels_whole() {
        let error = WireError::from_app_error(&AppError::Server {
            status: 503,
            message: "upstream said no".into(),
        });
        let line = terminal_error_line(9, error, 4096);
        let frame = decode_response(&line[..line.len() - 1]).expect("decode");
        assert_eq!(frame.id, 9);
        match frame.body {
            ResponseBody::Err { error } => match error.to_remote() {
                RemoteError::App(AppError::Server { status, message }) => {
                    assert_eq!(status, 503);
                    assert_eq!(message, "upstream said no", "the typed failure is intact");
                }
                other => panic!("unexpected: {other:?}"),
            },
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn a_failure_too_big_to_read_is_named_rather_than_written() {
        // The `err` twin of the result ceiling, and reachable where the result
        // one is not: a failure's `message` is whoever rendered it, and an
        // upstream's own error body travels uncapped inside `AppError::Server`.
        // Written whole, that frame is the line the caller's reader refuses —
        // so a typed failure would arrive as a closed connection instead.
        let error = WireError::from_app_error(&AppError::Server {
            status: 500,
            message: "x".repeat(4096),
        });
        let limit = 256;
        let line = terminal_error_line(11, error, limit);
        assert!(
            line.len() <= limit,
            "the error about the error has to fit where the error did not"
        );
        let frame = decode_response(&line[..line.len() - 1]).expect("decode");
        assert_eq!(frame.id, 11, "it still answers the request that asked");
        match frame.body {
            ResponseBody::Err { error } => match error.to_remote() {
                RemoteError::Protocol(ProtocolError::ErrorTooLarge {
                    kind,
                    bytes,
                    limit: reported,
                }) => {
                    assert_eq!(kind, "Server", "the caller still learns what failed");
                    assert!(bytes > reported);
                    assert_eq!(reported, limit);
                }
                other => panic!("unexpected: {other:?}"),
            },
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn what_one_connection_can_hold_is_a_stated_number() {
        // The queue is bounded by frame *count* while what costs is a frame's
        // *size*, so the two constants together are the memory bound — and the
        // dominant term is the concurrency limit, not the queue: a permitted
        // task holds its completed frame from the moment it is built until its
        // awaited send is taken. Asserted as a number so raising either
        // constant has to be a decision rather than a discovery.
        assert_eq!(
            serve::MAX_RETAINED_RESPONSE_BYTES,
            7 * MAX_RESPONSE_BYTES,
            "four requests in hand, two queued, one in the writer's"
        );
        assert!(
            serve::MAX_RETAINED_RESPONSE_BYTES <= 512 * 1024 * 1024,
            "one connection may hold {} bytes",
            serve::MAX_RETAINED_RESPONSE_BYTES
        );
    }

    #[test]
    fn a_parameterless_verb_still_checks_the_shape_of_its_params() {
        // Every verb that reads no fields — the four of them — used to build
        // its call without looking at `params` at all, so anything whatsoever
        // rode through. `hello` is the sharp one: a malformed frame completed
        // the handshake, which is the exchange whose entire job is agreeing on
        // the protocol.
        for verb in [
            "hello",
            "account.show",
            "wallet.credentials",
            "backend.list",
        ] {
            for accepted in [
                serde_json::Value::Null,
                serde_json::json!({}),
                // A field this build does not name is how the protocol grows,
                // and is what the parameterized verbs already tolerate.
                serde_json::json!({ "added_later": 1 }),
            ] {
                assert!(
                    Call::parse(verb, &accepted).is_ok(),
                    "`{verb}` refused {accepted}, which is a parameter set"
                );
            }
            for refused in [
                serde_json::json!("not-an-object"),
                serde_json::json!(7),
                serde_json::json!(true),
                serde_json::json!([]),
            ] {
                match Call::parse(verb, &refused) {
                    Err(ProtocolError::BadParams { verb: named, .. }) => {
                        assert_eq!(named, verb, "the refusal names the verb");
                    }
                    other => panic!("`{verb}` accepted {refused}: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn an_ordinary_delta_is_one_frame() {
        let event = crate::ChatStreamEvent::ContentDelta("a token".into());
        let lines = chunk_lines(2, &event, MAX_RESPONSE_BYTES);
        assert_eq!(lines.len(), 1, "nothing is split that already fits");
        let line = &lines[0];
        let frame = decode_response(&line[..line.len() - 1]).expect("decode");
        assert_eq!(frame.id, 2);
        assert!(matches!(frame.body, ResponseBody::Chunk { .. }));
    }

    /// The split, read back the way whoever dials reads.
    ///
    /// `for_responses` is `with_limit(MAX_RESPONSE_BYTES)`, and the serving
    /// side passes that same constant — so scaling both ends together is the
    /// same reader under the same rule, without allocating 64 MiB to say so.
    async fn split_round_trip(event: crate::ChatStreamEvent, limit: usize) -> String {
        let lines = chunk_lines(7, &event, limit);
        assert!(lines.len() > 1, "an oversized delta has to be split at all");
        for line in &lines {
            assert!(
                line.len() <= limit,
                "a split piece is still a line the reader would refuse: {} > {limit}",
                line.len()
            );
        }

        let joined: &'static [u8] = Box::leak(lines.concat().into_boxed_slice());
        let mut reader = FrameReader::with_limit(tokio::io::BufReader::new(joined), limit);
        let mut rebuilt = String::new();
        let mut frames = 0;
        while let Some(line) = reader
            .next_line()
            .await
            .expect("every frame is one the reader accepts")
        {
            let frame = decode_response(line).expect("decode");
            assert_eq!(frame.id, 7, "every piece answers the request that asked");
            match frame.body {
                ResponseBody::Chunk { data } => {
                    let piece: crate::ChatStreamEvent =
                        serde_json::from_value(data).expect("a chunk event");
                    assert_eq!(
                        std::mem::discriminant(&piece),
                        std::mem::discriminant(&event),
                        "a piece never changes what kind of delta it is"
                    );
                    rebuilt.push_str(text_of(&piece));
                }
                other => panic!("unexpected: {other:?}"),
            }
            frames += 1;
        }
        assert_eq!(frames, lines.len(), "every piece was readable");
        rebuilt
    }

    #[tokio::test]
    async fn a_delta_past_the_ceiling_arrives_whole_across_several_frames() {
        // The defect this prevents: a backend the profile is pointed at emits
        // one enormous delta, the app writes it as one line, and a conforming
        // reader reports `FrameTooLarge` and hangs up — on a turn that then
        // finishes and persists perfectly well.
        //
        // The text is adversarial on both axes the split has to respect:
        // multi-byte characters it must not cut through, and control
        // characters whose JSON escape is six bytes for one.
        let text = "héllo\u{0}wörld — ✂\u{1f}".repeat(400);
        let rebuilt =
            split_round_trip(crate::ChatStreamEvent::ContentDelta(text.clone()), 512).await;
        assert_eq!(
            rebuilt, text,
            "the pieces concatenate to exactly what the model said"
        );
    }

    #[tokio::test]
    async fn a_reasoning_delta_splits_the_same_way() {
        // Both variants carry text and nothing else, which is what makes
        // splitting honest for the whole vocabulary rather than for one arm.
        let text = "thinking ".repeat(400);
        let rebuilt =
            split_round_trip(crate::ChatStreamEvent::ReasoningDelta(text.clone()), 512).await;
        assert_eq!(rebuilt, text);
    }

    #[test]
    fn a_split_never_cuts_a_character_in_half() {
        // Every piece is decodable UTF-8 by construction — a `String` that
        // came back through serde proves it — so the assertion is that the
        // pieces are *characters*, not bytes: joined they equal the original
        // and each one alone is well formed.
        let text = "✂".repeat(300);
        let lines = chunk_lines(1, &crate::ChatStreamEvent::ContentDelta(text.clone()), 256);
        assert!(lines.len() > 1);
        let mut rebuilt = String::new();
        for line in &lines {
            let frame = decode_response(&line[..line.len() - 1]).expect("decode");
            let ResponseBody::Chunk { data } = frame.body else {
                panic!("not a chunk");
            };
            let piece: crate::ChatStreamEvent = serde_json::from_value(data).expect("event");
            let piece = text_of(&piece);
            assert!(
                !piece.is_empty(),
                "a piece that says nothing makes no progress"
            );
            assert_eq!(piece.len() % "✂".len(), 0, "a character was cut in half");
            rebuilt.push_str(piece);
        }
        assert_eq!(rebuilt, text);
    }

    #[tokio::test]
    async fn the_response_reader_carries_a_line_the_request_reader_would_refuse() {
        let mut oversized = vec![b'x'; MAX_FRAME_BYTES + 1];
        oversized.push(b'\n');
        let leaked: &'static [u8] = Box::leak(oversized.into_boxed_slice());
        let mut reader = FrameReader::for_responses(tokio::io::BufReader::new(leaked));
        let line = reader.next_line().await.expect("read").expect("a line");
        assert_eq!(line.len(), MAX_FRAME_BYTES + 1);
    }

    async fn lines(input: &'static [u8], limit: usize) -> Vec<Result<Vec<u8>, ProtocolError>> {
        let mut reader = FrameReader::with_limit(tokio::io::BufReader::new(input), limit);
        let mut out = Vec::new();
        loop {
            match reader.next_line().await {
                Ok(Some(line)) => out.push(Ok(line.to_vec())),
                Ok(None) => break,
                Err(e) => {
                    out.push(Err(e));
                    break;
                }
            }
        }
        out
    }

    #[tokio::test]
    async fn the_reader_splits_on_newlines_and_skips_blank_ones() {
        let got = lines(b"{\"a\":1}\n\n{\"b\":2}\n", MAX_FRAME_BYTES).await;
        let got: Vec<_> = got.into_iter().map(|r| r.expect("line")).collect();
        assert_eq!(got, vec![b"{\"a\":1}".to_vec(), b"{\"b\":2}".to_vec()]);
    }

    #[tokio::test]
    async fn a_final_line_with_no_newline_is_not_a_frame() {
        let got = lines(b"{\"a\":1}\n{\"unfinished\":", MAX_FRAME_BYTES).await;
        let got: Vec<_> = got.into_iter().map(|r| r.expect("line")).collect();
        assert_eq!(got, vec![b"{\"a\":1}".to_vec()]);
    }

    #[tokio::test]
    async fn an_oversized_line_is_refused_rather_than_buffered() {
        let got = lines(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", 8).await;
        assert_eq!(got.len(), 1);
        let err = got.into_iter().next().unwrap().unwrap_err();
        assert_eq!(err, ProtocolError::FrameTooLarge { limit: 8 });
        assert!(err.is_fatal(), "the reader lost its place in the stream");
    }

    #[tokio::test]
    async fn a_line_exactly_at_the_limit_is_read() {
        let got = lines(b"12345678\n123456789\n", 8).await;
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].as_ref().unwrap(), b"12345678");
        assert!(got[1].is_err(), "the next one is one byte over");
    }
}
