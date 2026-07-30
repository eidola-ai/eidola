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
    /// (last write wins — a consumer overriding a built-in is a feature, and a
    /// duplicate name would otherwise be an ambiguous wire schema).
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

#[cfg(test)]
mod tests {
    use super::*;

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
