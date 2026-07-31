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
use std::sync::Arc;

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
/// below), task 35's `remember`, and task 36's `list_my_spaces`. They are
/// protocol surface, not ordinary built-ins — a system note promises the model
/// these names with these semantics, and each is bound to something only the
/// turn has (its own `ThreadSnapshot`; for `remember`, the responding
/// participant's identity and residence space; for `list_my_spaces`, the
/// responding participant, whose membership *is* the boundary the tool
/// enforces) that a process-scoped registration structurally cannot
/// supply. Reserving them keeps "what the model was promised" and "what
/// executes" the same object on every turn, instead of silently diverging the
/// moment a space branches or memory is switched on.
pub const RESERVED_TOOL_NAMES: [&str; 5] = [
    "list_branches",
    "read_thread",
    "read_post",
    crate::memory::REMEMBER_TOOL_NAME,
    crate::discovery::LIST_MY_SPACES_TOOL_NAME,
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
// actually has branches (so a linear space stays byte-identical to pre-task-21)
// and only when the backend can carry a `tools` field at all. See the
// `backend_accepts_tools` comment there for the removal trigger.
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

/// `read_post` — one post in full, plus the passages it quotes.
pub(crate) struct ReadPostTool {
    snapshot: Arc<crate::ThreadSnapshot>,
}

impl ReadPostTool {
    pub(crate) fn new(snapshot: Arc<crate::ThreadSnapshot>) -> Self {
        Self { snapshot }
    }
}

impl Tool for ReadPostTool {
    fn name(&self) -> &str {
        "read_post"
    }

    fn description(&self) -> &str {
        "Read one post in full by handle, together with any passages it quotes from other posts. \
         A snapshot taken when this turn started."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Handle of the post to read, e.g. \"#a2c4e6g\".",
                },
            },
            "required": ["handle"],
        })
    }

    fn call<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let handle = handle_arg(&arguments)?;
            Ok(self.snapshot.render_post(&handle))
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
            Arc::new(ReadPostTool::new(Arc::new(crate::ThreadSnapshot::new(
                Vec::new(),
                0,
            )))),
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
        assert_eq!(RESERVED_TOOL_NAMES.len(), 5);
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
}
