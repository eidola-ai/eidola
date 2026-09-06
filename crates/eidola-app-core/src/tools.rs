//! The tool registry — the seam a turn's bounded tool-calling loop runs.
//!
//! # What a tool is (v1)
//!
//! **Harness-internal only.** A tool is a Rust implementation living inside
//! app-core (or registered by a consumer that links it) which the turn loop
//! executes in-process. There is deliberately no external/MCP surface here:
//! that is the deferred roles/tools/capabilities item, and admitting it now
//! would drag process spawning, sandboxing and a permission model into the
//! chat path before the loop itself is proven.
//!
//! # The seam
//!
//! A turn resolves the registry once, in `prepare_turn`, and carries it for
//! the whole turn (`TurnPrep.tools`). Two properties are load-bearing:
//!
//! * **Empty is invisible.** When the registry holds no tools the request body
//!   carries no `tools` field at all — today's requests stay byte-identical, so
//!   upstream prefix caches and every existing pinned-bytes test are
//!   undisturbed. Tools are opt-in per install, not a global wire change.
//! * **Names are the key.** The model addresses a tool by name; [`ToolRegistry`]
//!   resolves a name back to its implementation. A name the registry doesn't
//!   know is not a turn failure — it is reported back to the model as a tool
//!   *result* (see `ToolOutcome` in `lib.rs`), the same way an argument-schema
//!   mistake is, because both are model errors the model can correct on the
//!   next round.
//!
//! Tasks 21 (thread map + navigation tools) and 22 (the may-decline router's
//! agent-side `decline` tool) plug in here: they register [`Tool`]
//! implementations and change nothing else about the turn machinery.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Weak};

/// The boxed future a [`Tool::call`] returns.
pub type ToolFuture<'a> = Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + 'a>>;

/// A tool execution failure.
///
/// This is deliberately *not* an [`crate::error::AppError`]: a tool that fails
/// does not fail the turn. The message is handed back to the model as the tool
/// result so it can adapt (retry with different arguments, take another route,
/// or explain the failure to the reader) — the same contract a file-reading
/// agent harness gives its model.
#[derive(Debug, Clone)]
pub struct ToolError {
    pub message: String,
}

impl ToolError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// A tool the turn loop can execute.
///
/// Implementations must be cheap to clone-by-`Arc` and safe to call
/// concurrently: a single turn executes a round's calls sequentially, but
/// several turns (a participant fan-out) can be in flight at once.
pub trait Tool: Send + Sync {
    /// The name the model calls. Must be stable — it is what the model emits
    /// and what the persisted `tool_use` content block records.
    fn name(&self) -> &str;

    /// One-line description sent upstream in the function schema.
    fn description(&self) -> &str;

    /// JSON Schema (an object schema) for the arguments. Sent verbatim as the
    /// function's `parameters`.
    fn parameters(&self) -> serde_json::Value;

    /// Execute the call. `arguments` is the model's argument object, already
    /// parsed — a model that emitted invalid JSON never reaches here (the loop
    /// reports that as a tool error itself, without calling).
    fn call<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a>;
}

/// The set of tools available to a turn.
///
/// Cloning is cheap (the tools are behind `Arc`), which is what lets a turn
/// snapshot the registry at `prepare_turn` and keep a stable tool set for the
/// whole loop even if the process registry changes mid-turn.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: Vec<Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("tools", &self.names())
            .finish()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, replacing any earlier registration of the same name
    /// (last write wins — a duplicate name would otherwise be an ambiguous
    /// wire schema).
    ///
    /// This is the raw mechanism, and it is also what `prepare_turn` uses to
    /// layer a turn's navigation tools onto its registry snapshot. The
    /// *reservation* of those names lives at the public seam
    /// ([`crate::AppCore::register_tool`]), which is the only path by which a
    /// consumer can reach the process registry — so a collision is refused
    /// loudly at registration time rather than resolved silently, on branched
    /// turns only, by whichever write happened to be last.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.retain(|t| t.name() != name);
        self.tools.push(tool);
    }

    /// Remove a tool from this registry, if it holds one by that name.
    ///
    /// The counterpart to [`Self::register`], and used for the same reason a
    /// turn layers tools on: a turn's snapshot is *this turn's* tool set. A
    /// turn nobody invited — the driver's own mechanical notification — takes
    /// the decline checkpoint back out, because declining one would be a turn
    /// bowing out of a message it was not asked to have an opinion about.
    pub fn withdraw(&mut self, name: &str) {
        self.tools.retain(|t| t.name() != name);
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Resolve a tool by the name the model used.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.iter().find(|t| t.name() == name)
    }

    /// The OpenAI function-calling `tools` array for this registry. Never
    /// called when the registry is empty — the request omits the field
    /// entirely in that case.
    pub fn schemas(&self) -> Vec<serde_json::Value> {
        self.tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters(),
                    }
                })
            })
            .collect()
    }
}

/// Tool names [`crate::AppCore::register_tool`] refuses.
///
/// These are the **turn-scoped** tools `prepare_turn` attaches on top of the
/// process-registry snapshot: the navigation tools of a branched turn (see
/// below), the agent-memory `remember`, `list_my_spaces`, and `delegate`. They are
/// protocol surface, not ordinary built-ins — a system note promises the model
/// these names with these semantics, and each is bound to something only the
/// turn has (its own `ThreadSnapshot`; for `remember`, the responding
/// participant's identity and residence space; for `list_my_spaces`, the
/// responding participant, whose membership *is* the boundary the tool
/// enforces; for `delegate`, the responding participant, the space it is
/// delegating from, the roster it was shown, and the post this turn answers)
/// that a process-scoped registration structurally cannot
/// supply. Reserving them keeps "what the model was promised" and "what
/// executes" the same object on every turn, instead of silently diverging the
/// moment a space branches or memory is switched on.
pub const RESERVED_TOOL_NAMES: [&str; 6] = [
    "list_branches",
    "read_thread",
    "read_post",
    crate::memory::REMEMBER_TOOL_NAME,
    crate::discovery::LIST_MY_SPACES_TOOL_NAME,
    crate::subspaces::DELEGATE_TOOL_NAME,
];

/// Whether `name` is reserved for a turn-scoped tool.
pub fn is_reserved_tool_name(name: &str) -> bool {
    RESERVED_TOOL_NAMES.contains(&name)
}

/// A trivial tool that returns its `text` argument verbatim.
///
/// Ships with the registry so the loop is exercisable end-to-end before task
/// 21's navigation tools land: it has no dependencies, no side effects, and a
/// deterministic result, which makes it the right fixture for the chat-path
/// tests. Nothing registers it in production — the registry starts empty and a
/// consumer must call [`crate::AppCore::register_tool`].
pub struct EchoTool;

impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Return the supplied text verbatim. Useful only for exercising the tool loop."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": { "type": "string", "description": "The text to echo back." }
            },
            "required": ["text"],
        })
    }

    fn call<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            match arguments.get("text").and_then(|v| v.as_str()) {
                Some(text) => Ok(text.to_string()),
                None => Err(ToolError::new("`text` is required and must be a string")),
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Navigation tools (task 21)
//
// The three tools a branched space's turn gets, so a model that reads a thread
// map entry and wants more can descend. They are deliberately tiny and
// deliberately **read-only over a snapshot**: each holds an `Arc` of the
// `ThreadSnapshot` `prepare_turn` built for this turn, so a result is a
// point-in-time view (stale-ok — the same contract a file-reading agent harness
// gives its model) and the tools need no database handle, no lifetime, and no
// synchronization.
//
// Handle → id resolution is harness-side: the snapshot derives `post_handle`
// over the space's items and matches. A handle it does not know is answered
// honestly *in the result* — the map is a snapshot and a model reading a stale
// one must get something it can act on, not a failed turn.
//
// Attachment is `prepare_turn`'s job, and it is gated: only when the space
// actually has branches (so the field appears with the affordance, once)
// and only when the endpoint has not been observed to reject a `tools` field.
// See the `backend_accepts_tools` comment there for why capability is learned
// per (backend, model) rather than assumed from the backend's kind.
// ---------------------------------------------------------------------------

/// Normalize a model-supplied handle: trim, drop a leading `#`, lowercase.
/// Handles are lowercase base32 by construction, so this only ever recovers
/// from the model's own formatting.
fn normalize_handle(raw: &str) -> String {
    raw.trim().trim_start_matches('#').trim().to_lowercase()
}

/// Read the required `handle` argument.
fn handle_arg(arguments: &serde_json::Value) -> Result<String, ToolError> {
    match arguments.get("handle").and_then(|v| v.as_str()) {
        Some(h) if !normalize_handle(h).is_empty() => Ok(normalize_handle(h)),
        _ => Err(ToolError::new(
            "`handle` is required and must be a post handle string, e.g. \"#a2c4e6g\"",
        )),
    }
}

/// Read an optional non-negative integer argument.
fn usize_arg(arguments: &serde_json::Value, name: &str, default: usize) -> usize {
    arguments
        .get(name)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

/// `list_branches` — the whole space's fork structure.
pub(crate) struct ListBranchesTool {
    snapshot: Arc<crate::ThreadSnapshot>,
}

impl ListBranchesTool {
    pub(crate) fn new(snapshot: Arc<crate::ThreadSnapshot>) -> Self {
        Self { snapshot }
    }
}

impl Tool for ListBranchesTool {
    fn name(&self) -> &str {
        "list_branches"
    }

    fn description(&self) -> &str {
        "List every fork point in this space and the branches at it. The thread map you were \
         given covers only the forks on the conversation you can see; this covers the whole \
         space. A snapshot taken when this turn started."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    fn call<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok(self.snapshot.render_all_forks()) })
    }
}

/// `read_thread` — a bounded window of one branch.
pub(crate) struct ReadThreadTool {
    snapshot: Arc<crate::ThreadSnapshot>,
}

impl ReadThreadTool {
    pub(crate) fn new(snapshot: Arc<crate::ThreadSnapshot>) -> Self {
        Self { snapshot }
    }
}

impl Tool for ReadThreadTool {
    fn name(&self) -> &str {
        "read_thread"
    }

    fn description(&self) -> &str {
        "Read a branch: the post with the given handle and everything below it, depth-first, in \
         the same format as the conversation above. Bounded — the result says how many posts \
         exist and how to page through them. A snapshot taken when this turn started."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Handle of the post to read from, e.g. \"#a2c4e6g\".",
                },
                "limit": {
                    "type": "integer",
                    "description": format!(
                        "How many posts to return (default {}, maximum {}).",
                        crate::READ_THREAD_DEFAULT_LIMIT, crate::READ_THREAD_MAX_LIMIT,
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "How many posts to skip (default 0).",
                },
            },
            "required": ["handle"],
        })
    }

    fn call<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let handle = handle_arg(&arguments)?;
            let limit = usize_arg(&arguments, "limit", crate::READ_THREAD_DEFAULT_LIMIT);
            let offset = usize_arg(&arguments, "offset", 0);
            Ok(self.snapshot.render_thread(&handle, offset, limit))
        })
    }
}

/// The refusal a model reads when it follows a quote into a conversation it is
/// not part of (task 37 rule 4).
///
/// **Non-leaking by construction, and it must stay that way.** It confirms only
/// what the model already sees — that this post quotes something from elsewhere,
/// which rule 3 makes public inside *this* space — and names no title, no
/// participant and no content of the other conversation. It is also an ordinary
/// tool *result*, not a turn failure: "I can see this was quoted from a
/// conversation I'm not in" is exactly what the model should be able to say, and
/// the fix is a human granting membership, after which the very next call
/// resolves (tools re-check membership per call, so retry needs no machinery).
pub const FOLLOW_DENIED: &str = "You do not take part in the conversation that passage was \
     quoted from, so you cannot read it. What was quoted into this conversation is above — that \
     excerpt is what was shared here. If you need more, say so: a human can add you to that \
     conversation, and then this will work.";

/// Render a post reached by following a quote (task 37). Pure over its inputs —
/// these are wire bytes a model reads, so they are unit-tested.
///
/// The body goes through [`crate::with_header`] over
/// [`crate::render_post_for_model`], the same two rendering paths every other
/// post takes: a followed post quotes things too, and a model that followed one
/// quote must not be handed the next one as a literal `{{ embed N }}` marker.
/// The one added line is provenance: which conversation this came from, or that
/// it is a superseded version of a post in this one.
pub(crate) fn render_followed_post(
    row: &crate::db::ReferencedPostRow,
    current_space_id: &str,
    references: &std::collections::BTreeMap<u64, crate::ReferenceEntry>,
) -> String {
    let body = crate::with_header(
        &row.item_id,
        &row.participant_label,
        row.created_at,
        &crate::render_post_for_model(&row.text, references),
    );
    if row.space_id == current_space_id {
        // Same space, but not in the snapshot: the quote named a generation
        // that has since been edited or regenerated. References name concrete
        // generations and are never remapped, so this is the honest answer.
        return format!("An earlier version of a post in this conversation.\n\n{body}");
    }
    let title = row
        .space_title
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(crate::one_line)
        .or_else(|| crate::derive_space_title(&row.text))
        .unwrap_or_else(|| "(untitled)".to_string());
    format!("From another conversation you take part in — {title}.\n\n{body}")
}

/// `read_post` — one post in full, plus the passages it quotes; and, with
/// `quote`, the post a quoted passage came from.
///
/// **Cross-space addressing goes through the reference edge, never a handle.**
/// Handles are derived (`post_handle(item_id)`) and therefore global, but this
/// snapshot only knows *this* space's, and that is the point: a model cannot
/// name a post in another conversation, only *this* post's quote number — the
/// ordinal that is already visible to every participant here (rule 3) as the
/// footnote index and the `{{ embed N }}` marker. So the reachable set is
/// exactly "what someone here already quoted", and the membership check decides
/// whether it opens. Guessing is not a failure mode, it is unrepresentable.
pub(crate) struct ReadPostTool {
    snapshot: Arc<crate::ThreadSnapshot>,
    /// The core, for the cross-space follow's reads. `Weak`, like every other
    /// turn-scoped tool, so a tool that outlived its turn cannot keep the
    /// database open.
    inner: Weak<crate::Inner>,
    /// The responding participant — whose membership is re-read on every call,
    /// which is what makes grant-then-retry work with no extra machinery.
    participant_id: String,
    current_space_id: String,
}

impl ReadPostTool {
    pub(crate) fn new(
        snapshot: Arc<crate::ThreadSnapshot>,
        inner: Weak<crate::Inner>,
        participant_id: String,
        current_space_id: String,
    ) -> Self {
        Self {
            snapshot,
            inner,
            participant_id,
            current_space_id,
        }
    }

    /// Follow reference `ordinal` of the post at `handle`.
    async fn follow(&self, handle: &str, ordinal: i64) -> String {
        let Some(&idx) = self.snapshot.by_handle.get(handle) else {
            return self.snapshot.unknown_handle(handle);
        };
        let node = &self.snapshot.nodes[idx];
        let Some(reference) = node.references.iter().find(|r| r.ordinal == ordinal) else {
            let mut ordinals: Vec<String> = node
                .references
                .iter()
                .map(|r| r.ordinal.to_string())
                .collect();
            ordinals.sort();
            return if ordinals.is_empty() {
                format!("Post #{handle} quotes nothing, so there is no quote {ordinal} to follow.")
            } else {
                format!(
                    "Post #{handle} has no quote {ordinal}. It quotes: {}.",
                    ordinals.join(", ")
                )
            };
        };
        // Already in this turn's snapshot: it is a current post of this space,
        // which the model can read anyway — render it the ordinary way rather
        // than taking a database round trip to say the same thing.
        if let Some(&target) = self.snapshot.by_action.get(&reference.antecedent_action_id) {
            return self
                .snapshot
                .render_post(&self.snapshot.handles[target].clone());
        }
        let Some(inner) = self.inner.upgrade() else {
            return "That quoted post is unavailable in this turn.".to_string();
        };
        let Ok(conn) = inner.db_conn().await else {
            return "That quoted post could not be read.".to_string();
        };
        let row = match crate::db::referenced_post(&conn, &reference.antecedent_action_id).await {
            Ok(Some(row)) => row,
            // Gone, or never a post to begin with — `referenced_post` reads
            // only post types, so a quote that somehow names a tool trace, a
            // decision or a memory block lands here rather than rendering one.
            // One answer for both: neither is followable and neither reveals
            // anything.
            Ok(None) => return "That quote does not point at a readable post.".to_string(),
            Err(_) => return "That quoted post could not be read.".to_string(),
        };
        // Rule 4: follow requires membership, re-read per call.
        match crate::db::is_space_member(&conn, &row.space_id, &self.participant_id).await {
            Ok(true) => {
                // The followed post's own references, addressed from *this*
                // turn's space: its handles are the only ones the model can
                // resolve, so a quote living anywhere else renders as an
                // earlier version rather than an address that opens nothing.
                let Ok(references) = crate::reference_entries(
                    &conn,
                    &self.snapshot,
                    &reference.antecedent_action_id,
                )
                .await
                else {
                    return "That quoted post could not be read.".to_string();
                };
                render_followed_post(&row, &self.current_space_id, &references)
            }
            Ok(false) => FOLLOW_DENIED.to_string(),
            Err(_) => "That quoted post could not be read.".to_string(),
        }
    }
}

impl Tool for ReadPostTool {
    fn name(&self) -> &str {
        "read_post"
    }

    fn description(&self) -> &str {
        "Read one post in full by handle, together with any passages it quotes from other posts. \
         Pass `quote` to read the post a quoted passage came from — which may be in another \
         conversation, and only opens if you take part in it. A snapshot taken when this turn \
         started."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Handle of the post to read, e.g. \"#a2c4e6g\".",
                },
                "quote": {
                    "type": "integer",
                    "description": "Instead of the post itself, read the post its quote with this \
                                    number came from (the number shown beside the quote).",
                },
            },
            "required": ["handle"],
        })
    }

    fn call<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let handle = handle_arg(&arguments)?;
            match arguments.get("quote").and_then(|v| v.as_i64()) {
                Some(ordinal) => Ok(self.follow(&handle, ordinal).await),
                None => Ok(self.snapshot.render_post(&handle)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_navigation_tool_names_are_reserved() {
        for t in [
            Arc::new(ListBranchesTool::new(Arc::new(crate::ThreadSnapshot::new(
                Vec::new(),
                0,
            )))) as Arc<dyn Tool>,
            Arc::new(ReadThreadTool::new(Arc::new(crate::ThreadSnapshot::new(
                Vec::new(),
                0,
            )))),
            Arc::new(ReadPostTool::new(
                Arc::new(crate::ThreadSnapshot::new(Vec::new(), 0)),
                Weak::new(),
                String::new(),
                String::new(),
            )),
        ] {
            assert!(
                is_reserved_tool_name(t.name()),
                "{} must be reserved",
                t.name()
            );
        }
        assert!(!is_reserved_tool_name("echo"));
        // Task 35's `remember` is reserved for the same reason: it is bound to
        // the turn's responding participant, which the process registry has no
        // way to express.
        assert!(is_reserved_tool_name(crate::memory::REMEMBER_TOOL_NAME));
        // Task 36's `list_my_spaces` likewise: it is bound to the responding
        // participant, and membership is the boundary it enforces.
        assert!(is_reserved_tool_name(
            crate::discovery::LIST_MY_SPACES_TOOL_NAME
        ));
        // `delegate` likewise: it owns a room on behalf of the responding
        // participant, from the space this turn is in, anchored on the post
        // this turn answers.
        assert!(is_reserved_tool_name(crate::subspaces::DELEGATE_TOOL_NAME));
        assert_eq!(RESERVED_TOOL_NAMES.len(), 6);
    }

    #[test]
    fn handle_arguments_are_normalized_and_validated() {
        assert_eq!(
            handle_arg(&serde_json::json!({"handle": " #A2C4E6G "})).unwrap(),
            "a2c4e6g"
        );
        assert_eq!(
            handle_arg(&serde_json::json!({"handle": "a2c4e6g"})).unwrap(),
            "a2c4e6g"
        );
        for bad in [
            serde_json::json!({}),
            serde_json::json!({"handle": ""}),
            serde_json::json!({"handle": "#"}),
            serde_json::json!({"handle": 7}),
        ] {
            assert!(handle_arg(&bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn optional_integer_arguments_fall_back_to_defaults() {
        assert_eq!(usize_arg(&serde_json::json!({}), "limit", 10), 10);
        assert_eq!(usize_arg(&serde_json::json!({"limit": 3}), "limit", 10), 3);
        // A negative or non-numeric value is the model's mistake; falling back
        // to the default keeps the call useful instead of failing it.
        assert_eq!(
            usize_arg(&serde_json::json!({"limit": -3}), "limit", 10),
            10
        );
        assert_eq!(
            usize_arg(&serde_json::json!({"limit": "many"}), "limit", 10),
            10
        );
    }

    #[test]
    fn empty_registry_reports_empty() {
        let reg = ToolRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.schemas().is_empty());
        assert!(reg.get("echo").is_none());
    }

    #[test]
    fn register_resolves_by_name_and_emits_a_function_schema() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        assert!(!reg.is_empty());
        assert!(reg.get("echo").is_some());
        assert!(reg.get("nope").is_none());

        let schemas = reg.schemas();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["type"], "function");
        assert_eq!(schemas[0]["function"]["name"], "echo");
        assert_eq!(schemas[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn re_registering_a_name_replaces_rather_than_duplicates() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(EchoTool));
        reg.register(Arc::new(EchoTool));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.schemas().len(), 1);
    }

    /// The creation time every synthetic followed post carries:
    /// `2026-08-11T14:02:33Z`.
    const FOLLOWED_AT: i64 = 1_786_456_953_000;

    fn referenced(space: &str, title: Option<&str>, text: &str) -> crate::db::ReferencedPostRow {
        crate::db::ReferencedPostRow {
            space_id: space.to_string(),
            space_title: title.map(str::to_string),
            item_id: "item-1".to_string(),
            participant_label: "Ada".to_string(),
            action_type: "inference".to_string(),
            text: text.to_string(),
            created_at: FOLLOWED_AT,
        }
    }

    #[test]
    fn a_followed_post_says_where_it_came_from_and_renders_as_a_post() {
        let row = referenced("other", Some("Tides"), "Spring tides at syzygy.");
        let rendered = render_followed_post(&row, "here", &Default::default());
        assert_eq!(
            rendered,
            format!(
                "From another conversation you take part in — Tides.\n\n{}",
                crate::with_header("item-1", "Ada", FOLLOWED_AT, "Spring tides at syzygy.")
            )
        );
        // Same space ⇒ not another conversation, just a generation the current
        // view no longer shows.
        let rendered = render_followed_post(
            &referenced("here", None, "Older wording."),
            "here",
            &Default::default(),
        );
        assert!(
            rendered.starts_with("An earlier version of a post in this conversation.\n\n#"),
            "{rendered}"
        );
    }

    /// A followed post is a post: its own quotes render exactly as they do
    /// everywhere else — expanded and attributed at the marker, footnoted when
    /// the body never embedded them. Following one quote must not hand the
    /// model a literal marker for the next.
    #[test]
    fn a_followed_post_renders_its_own_quotes() {
        let row = referenced("other", Some("Tides"), "As noted:\n\n{{ embed 1 }}");
        let addressed = crate::ReferenceEntry {
            ordinal: 1,
            target: crate::ReferenceTarget::Addressable {
                item_id: "item-bo".to_string(),
                label: "Bo".to_string(),
            },
            annotation: None,
            body: crate::ReferenceBody::Passage("syzygy".to_string()),
        };
        let elsewhere = crate::ReferenceEntry {
            ordinal: 2,
            target: crate::ReferenceTarget::Elsewhere {
                label: Some("Cy".to_string()),
            },
            annotation: None,
            body: crate::ReferenceBody::Passage("neap".to_string()),
        };
        let references = std::collections::BTreeMap::from([(1, addressed), (2, elsewhere)]);
        let rendered = render_followed_post(&row, "here", &references);
        assert!(
            rendered.contains(&format!(
                "As noted:\n\n[1] {}\n> syzygy",
                crate::post_byline("item-bo", "Bo")
            )),
            "{rendered}"
        );
        assert!(
            rendered.ends_with(
                "Passages this post quotes:\n[2] Cy (a post outside this space, or an earlier \
                 version)\n> neap"
            ),
            "{rendered}"
        );
        assert!(!rendered.contains("{{ embed"), "{rendered}");
    }

    #[test]
    fn an_untitled_source_conversation_still_gets_a_name() {
        let row = referenced(
            "other",
            Some("   "),
            "# Why do tides lag the moon?\n\nBecause…",
        );
        let rendered = render_followed_post(&row, "here", &Default::default());
        assert!(
            rendered.starts_with(
                "From another conversation you take part in — Why do tides lag \
                                  the moon?."
            ),
            "{rendered}"
        );
    }

    #[test]
    fn the_follow_denial_names_nothing_about_the_conversation_it_refuses() {
        // A guard on the constant itself: it may say *that* a passage was
        // quoted from elsewhere (public here per rule 3) and must never grow a
        // placeholder for anything else.
        for leak in ['{', '}', '<', '>', '#'] {
            assert!(
                !FOLLOW_DENIED.contains(leak),
                "the denial is a fixed sentence with no interpolation slot"
            );
        }
    }
}
