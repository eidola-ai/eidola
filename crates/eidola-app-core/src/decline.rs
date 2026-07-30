//! The **agent-side decline checkpoint** — a tool a driven agent can call to
//! bow out of a turn after seeing the full context.
//!
//! # What it is (and is not)
//!
//! This is an orthogonal *quality* valve, not a gating mechanism. By the time a
//! turn can call [`DeclineTool`] the prefill is already paid for, so declining
//! saves no meaningful cost — what it saves is the *reader*, from a reply that
//! had nothing to add. The cheap filter is the may-decline router (see
//! [`crate::router`]), which decides before a turn is driven at all; this is
//! the second, better-informed opinion.
//!
//! # Mechanics
//!
//! [`DeclineTool`] is an ordinary [`Tool`]: it is advertised in the request's
//! function schema like any other, and the model calls it by name. What makes
//! it a *checkpoint* is one seam in the turn loop: when a round's tool calls
//! include `decline` **and the turn's registry actually holds it**, the turn
//! ends there:
//!
//! * the round's `tool_call` action and `tool_result` action are persisted like
//!   any other round (the trace stays honest — the model's reasoning, its call,
//!   and the acknowledgement are all in the Record);
//! * a **`decision` action** is written with the post the turn answers as its
//!   antecedent, carrying the stated reason as its text — auditable, and
//!   labeled training data for a future fine-tune;
//! * the would-be post is **suppressed**: no `inference` action is written, and
//!   [`crate::ChatResult::declined`] carries the reason so the caller knows the
//!   turn ended in a decline rather than an empty answer;
//! * `Change::Space` is emitted (a UI wants to render "saw this, declined").
//!
//! **No spend is refunded beyond the normal turn machinery.** The turn ran; its
//! rounds settled their holds exactly as they always do.
//!
//! # Registration
//!
//! Nothing registers this by default, and that is load-bearing: the tool
//! registry starts **empty** so a registry-less install sends byte-identical
//! requests (see [`crate::tools`]). A consumer that wants the checkpoint calls
//! `AppCore::register_tool(Arc::new(DeclineTool))`. A model that invents the
//! name `decline` against a registry that has not registered it gets the
//! ordinary "unknown tool" result, and the turn continues — the checkpoint
//! keys on the registry, never on the name alone.

use std::sync::Arc;

use crate::tools::{Tool, ToolError, ToolFuture, ToolRegistry};

/// The name the model calls to bow out. Stable — it is what the model emits
/// and what the persisted `tool_use` block records.
pub const DECLINE_TOOL_NAME: &str = "decline";

/// The tool result handed back to the model when it declines. The model never
/// gets another round (the turn ends here), so this exists for the Record.
const DECLINE_ACK: &str = "Declined. This turn ends without a reply.";

/// A driven agent's opt-out: "I saw this and have nothing to add."
///
/// Deliberately argument-light — one optional `reason`, which is persisted on
/// the `decision` action. Asking for structure here would invite the model to
/// perform an explanation instead of making a judgment.
pub struct DeclineTool;

impl Tool for DeclineTool {
    fn name(&self) -> &str {
        DECLINE_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Bow out of this turn without replying. Call this when you have nothing \
         worth adding, when another participant is the right one to answer, or \
         when the conversation does not need you. Declining is a good outcome, \
         not a failure."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "One short line on why you are bowing out. Recorded, not shown as a reply.",
                }
            },
        })
    }

    fn call<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move { Ok::<String, ToolError>(DECLINE_ACK.to_string()) })
    }
}

/// Convenience: an `Arc`'d [`DeclineTool`] ready for `AppCore::register_tool`.
pub fn decline_tool() -> Arc<dyn Tool> {
    Arc::new(DeclineTool)
}

/// The reason a round's tool calls declined the turn, if they did.
///
/// Returns `None` unless the registry actually holds the decline tool: a model
/// that guesses the name against a registry without it must get the ordinary
/// unknown-tool result, not a silent turn-ending side effect. An empty or
/// absent `reason` yields an empty string — the decision is recorded either
/// way, since the *act* of declining is the datum.
pub(crate) fn declined_reason(
    registry: &ToolRegistry,
    names_and_args: &[(&str, &str)],
) -> Option<String> {
    registry.get(DECLINE_TOOL_NAME)?;
    let (_, arguments) = names_and_args
        .iter()
        .find(|(name, _)| *name == DECLINE_TOOL_NAME)?;
    Some(reason_from_arguments(arguments))
}

/// Read the `reason` out of a raw tool-argument string. Tolerant by design:
/// unparseable arguments still decline (the call itself is the decision), they
/// just record no reason.
pub(crate) fn reason_from_arguments(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .as_ref()
        .and_then(|v| v.get("reason"))
        .and_then(|r| r.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry_with_decline() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(Arc::new(DeclineTool));
        r
    }

    #[test]
    fn the_schema_is_an_object_with_an_optional_reason() {
        let schemas = registry_with_decline().schemas();
        assert_eq!(schemas[0]["function"]["name"], DECLINE_TOOL_NAME);
        assert_eq!(schemas[0]["function"]["parameters"]["type"], "object");
        // `reason` is optional: no `required` key at all.
        assert!(
            schemas[0]["function"]["parameters"]
                .get("required")
                .is_none()
        );
    }

    #[test]
    fn an_unregistered_decline_name_is_not_a_decline() {
        let empty = ToolRegistry::new();
        assert_eq!(
            declined_reason(&empty, &[(DECLINE_TOOL_NAME, r#"{"reason":"nope"}"#)]),
            None,
            "the checkpoint keys on the registry, never on the name alone"
        );
    }

    #[test]
    fn a_registered_decline_call_yields_its_reason() {
        let r = registry_with_decline();
        assert_eq!(
            declined_reason(
                &r,
                &[
                    ("echo", "{}"),
                    (DECLINE_TOOL_NAME, r#"{"reason":" nothing to add "}"#)
                ]
            ),
            Some("nothing to add".to_string())
        );
    }

    #[test]
    fn a_reasonless_or_malformed_call_still_declines() {
        let r = registry_with_decline();
        assert_eq!(
            declined_reason(&r, &[(DECLINE_TOOL_NAME, "{}")]),
            Some(String::new())
        );
        assert_eq!(
            declined_reason(&r, &[(DECLINE_TOOL_NAME, "not json")]),
            Some(String::new())
        );
    }

    #[test]
    fn other_tools_alone_do_not_decline() {
        let r = registry_with_decline();
        assert_eq!(declined_reason(&r, &[("echo", "{}")]), None);
    }
}
