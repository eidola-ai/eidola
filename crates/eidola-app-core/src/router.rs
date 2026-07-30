//! The **may-decline router** — one cheap model call that filters the
//! mechanical notify set down to who should actually respond.
//!
//! # Why a router and not per-agent polling
//!
//! Asking every candidate "do you want to answer this?" costs N full prefills
//! per post (prefill dominates both remote cost and local engine churn), and a
//! trimmed decline-context makes each decision worse-informed than the turn it
//! is deciding about. The deployed prior art (AutoGen speaker selection,
//! SillyTavern group-chat activation) converged on a single cheap router call,
//! and so do we.
//!
//! # Where it sits
//!
//! ```text
//!   post ──► plan_notifications ──► refine_notifications ──► drive turns
//!            (pure read, CI-       (this module: one small-  (unchanged)
//!             deterministic)        model call, degrading)
//! ```
//!
//! [`crate::AppCore::plan_notifications`] stays a **pure read** producing the
//! mechanical candidate set from the notify policies (`all` / `human` /
//! `explicit`) and the data-derived cascade guard. This module adds a step
//! *between* planning and turn-driving:
//! [`crate::AppCore::refine_notifications`], which takes that plan and returns
//! a (possibly empty) subset.
//!
//! Two things it deliberately does **not** touch:
//!
//! * **Explicit asks bypass entirely.** `respond_stream` / `respond_stream_as`
//!   consult no guards and no router — an explicit ask always runs. The router
//!   only ever filters *automatic* notification.
//! * **The cascade guard is unchanged.** It runs first, inside
//!   `plan_notifications`; a `Paused` plan is returned untouched (the router is
//!   not even called). The guard then applies to whatever the router lets
//!   through, because depth is re-derived from the data on the next plan.
//!
//! # Failure doctrine: degrade, never block
//!
//! A post is **never** blocked on router availability. If the router model is
//! unset (the default — the feature is off), unreachable, refuses to load, or
//! answers with something we can't parse, `refine_notifications` returns the
//! **mechanical candidate set unchanged** and logs honestly on stderr. The
//! failure mode is *extra* notifications, never lost ones.
//!
//! Router **non-selections are not persisted**. Not notifying someone is
//! exactly what an `explicit`-policy participant experiences today, and that
//! writes nothing; a row per participant per post would be pure noise.
//!
//! # Cost
//!
//! The router model is an ordinary qualified `<model>@<backend>` reference and
//! routes through the ordinary backend registry. An engine-backed reference
//! (`local` / `llamacpp`) takes the zero-spend path — no charge, no credential,
//! no account — which is what makes a local router genuinely free. A remote
//! (`eidola`) reference is allowed and **bills a normal inference on every
//! triggering post**; any settings surface must say so plainly.
//!
//! The call is deliberately **not a turn**: no actions, no context assembly, no
//! request rows, no attestation records. (For an `eidola` router the
//! per-handshake attestation still *happens* — it is simply not recorded, since
//! there is no action to hang the forensic trail from. That is the one honest
//! gap in "everything is in the Record".)
//!
//! # Testing
//!
//! The plumbing is CI-deterministic: the router is just another HTTP call, so
//! the chat harness scripts it exactly like a turn (a `RouterBehavior` arm on
//! the mock upstream, selected per test). Whether the router decides *well* is
//! a judgment question — an offline eval over a golden set of
//! (post, cards, slice) → expected notify set, scored against candidate
//! models and prompts. That never runs in CI: real-model output is not
//! deterministic across machines (Metal vs CPU kernels). It would live beside
//! the other offline harnesses, not in `tests/`.

use crate::Change;
use crate::db;
use crate::error::AppError;
use crate::{
    EidolaResolved, Inner, NotificationPlan, backends, estimate_charge_credits, fetch_models,
    local_models, now_ms, process_refund, recover_refund,
};

/// How many posts of the triggering post's upstream thread the router sees.
/// Small on purpose — the router is a routing decision, not a reader.
const THREAD_SLICE_POSTS: usize = 6;

/// Per-post byte budget inside the thread slice.
const POST_MAX_BYTES: usize = 480;

/// Per-candidate byte budget for the persona digest (the head of the
/// participant's system prompt).
const PERSONA_MAX_BYTES: usize = 240;

/// Completion cap for the router call. The answer is a tiny JSON object; this
/// is generous enough for a chatty small model and small enough that a remote
/// router's hold stays cheap.
const MAX_COMPLETION_TOKENS: u32 = 256;

/// One candidate the router chooses among — a participant the mechanical plan
/// already selected. Addressed by its **1-based index**, not its id: the ids
/// are UUIDv7s, and asking a small model to echo one back verbatim is a
/// needless failure mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RouterCandidate {
    pub participant_id: String,
    pub label: String,
    /// The head of the participant's effective system prompt, one line — what
    /// this agent is *for*, as far as the router needs to know.
    pub persona_digest: Option<String>,
    pub notify_policy: String,
}

/// One rendered line of the thread slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RouterThreadLine {
    pub author: String,
    pub text: String,
}

/// The router's instruction. Deliberately terse and output-shape-first: a
/// small model that gets nothing else right still has a chance at emitting the
/// object, and an unparseable answer degrades to "notify everyone the policies
/// picked" anyway.
pub(crate) const ROUTER_SYSTEM_PROMPT: &str = "\
You are a routing filter for a group conversation. You do not answer the \
conversation; you decide which of the listed participants should respond to \
the most recent message.

Choose only participants whose stated purpose is genuinely relevant to the \
latest message. Choosing nobody is a valid and often correct answer — silence \
is better than an off-topic reply. Never explain your reasoning.

Reply with exactly one JSON object and nothing else:

{\"notify\": [<candidate numbers>]}

Use the numbers shown in the CANDIDATES list. An empty list means nobody \
responds.";

/// Truncate to at most `max_bytes`, on a char boundary, marking the cut.
fn clip(text: &str, max_bytes: usize) -> String {
    let text = text.trim();
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &text[..end])
}

/// A participant's system prompt reduced to one clipped line — enough for the
/// router to tell a code reviewer from a copy editor, cheap enough to send for
/// every candidate on every post.
pub(crate) fn persona_digest(system_prompt: Option<&str>) -> Option<String> {
    let raw = system_prompt?.trim();
    if raw.is_empty() {
        return None;
    }
    let single_line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(clip(&single_line, PERSONA_MAX_BYTES))
}

/// Render the router's user message: the thread slice (oldest first, the
/// triggering post last and marked), then the candidate cards.
///
/// Pure and unit-tested — the prompt is the part most likely to be tuned by an
/// offline eval, so it is kept free of I/O.
pub(crate) fn build_router_prompt(
    thread: &[RouterThreadLine],
    candidates: &[RouterCandidate],
) -> String {
    let mut out = String::new();
    out.push_str("CONVERSATION (oldest first):\n");
    if thread.is_empty() {
        out.push_str("(nothing yet)\n");
    }
    let last = thread.len().saturating_sub(1);
    for (i, line) in thread.iter().enumerate() {
        let marker = if i == last { "LATEST — " } else { "" };
        out.push_str(&format!(
            "{marker}{}: {}\n",
            line.author,
            clip(&line.text, POST_MAX_BYTES)
        ));
    }
    out.push_str("\nCANDIDATES:\n");
    for (i, c) in candidates.iter().enumerate() {
        out.push_str(&format!("{}. {}", i + 1, c.label));
        if let Some(digest) = &c.persona_digest {
            out.push_str(&format!(" — {digest}"));
        }
        out.push_str(&format!(" [notify policy: {}]\n", c.notify_policy));
    }
    out.push_str(
        "\nWhich candidates should respond to the LATEST message? \
         Reply with the JSON object only.",
    );
    out
}

/// Read the router's reply into a set of 1-based candidate indices.
///
/// Tolerant of the ways a small model wraps an object (prose before it, a
/// ```` ```json ```` fence, a trailing sentence) by scanning for the first
/// balanced-looking `{ … }` span; strict about the object itself. Accepts
/// numbers or numeric strings in `notify`, ignores out-of-range entries, and
/// de-duplicates. An **empty** list is a valid answer (notify nobody).
///
/// `Err` means "we could not read a decision" — the caller degrades to the
/// mechanical set.
pub(crate) fn parse_router_selection(
    text: &str,
    candidate_count: usize,
) -> Result<Vec<usize>, String> {
    let start = text.find('{').ok_or("no JSON object in the router reply")?;
    let end = text
        .rfind('}')
        .ok_or("no JSON object in the router reply")?;
    if end <= start {
        return Err("no JSON object in the router reply".into());
    }
    let value: serde_json::Value =
        serde_json::from_str(&text[start..=end]).map_err(|e| format!("invalid JSON: {e}"))?;
    let entries = value
        .get("notify")
        .ok_or("the router reply has no `notify` key")?
        .as_array()
        .ok_or("`notify` is not an array")?;

    let mut out: Vec<usize> = Vec::new();
    for entry in entries {
        let n = match entry {
            serde_json::Value::Number(n) => n.as_u64(),
            serde_json::Value::String(s) => s.trim().parse::<u64>().ok(),
            _ => None,
        };
        let Some(n) = n else { continue };
        let n = n as usize;
        if n >= 1 && n <= candidate_count && !out.contains(&n) {
            out.push(n);
        }
    }
    Ok(out)
}

impl Inner {
    /// Filter a mechanical notification plan through the space's router model.
    ///
    /// Never fails: every error path returns `plan` unchanged (see the module
    /// docs' failure doctrine). Returns immediately — without touching the
    /// network — when the plan is `Paused`, when it holds fewer than two turns'
    /// worth of choice to make, or when the space has no router model.
    pub(crate) async fn refine_notifications(
        &self,
        space_id: &str,
        post_action_id: &str,
        plan: NotificationPlan,
    ) -> NotificationPlan {
        let turns = match &plan {
            // The cascade guard already spoke; the router is not consulted.
            NotificationPlan::Paused { .. } => return plan,
            NotificationPlan::Turns(t) if t.is_empty() => return plan,
            NotificationPlan::Turns(t) => t.clone(),
        };

        let Ok(conn) = self.db_conn().await else {
            return plan;
        };
        // Unset (the default) = the feature is off: no router call is made.
        let router_model = match db::space_router_model(&conn, space_id).await {
            Ok(Some(m)) => m,
            Ok(None) => return plan,
            Err(e) => {
                eprintln!("warning: could not read the space's router model: {e}");
                return plan;
            }
        };

        let (thread, candidates) = match self
            .router_inputs(&conn, space_id, post_action_id, &turns)
            .await
        {
            Ok(inputs) => inputs,
            Err(e) => {
                eprintln!("warning: may-decline router inputs unavailable: {e}");
                return plan;
            }
        };
        if candidates.is_empty() {
            return plan;
        }

        let prompt = build_router_prompt(&thread, &candidates);
        let reply = match self
            .router_completion(&router_model, ROUTER_SYSTEM_PROMPT, &prompt)
            .await
        {
            Ok(reply) => reply,
            Err(e) => {
                eprintln!(
                    "warning: may-decline router `{router_model}` unavailable ({e}); \
                     notifying the mechanical set"
                );
                return plan;
            }
        };

        let selected = match parse_router_selection(&reply, candidates.len()) {
            Ok(selected) => selected,
            Err(e) => {
                eprintln!(
                    "warning: may-decline router `{router_model}` returned unusable output \
                     ({e}); notifying the mechanical set"
                );
                return plan;
            }
        };

        let keep: Vec<&str> = selected
            .iter()
            .map(|i| candidates[i - 1].participant_id.as_str())
            .collect();
        NotificationPlan::Turns(
            turns
                .into_iter()
                .filter(|t| keep.contains(&t.participant_id.as_str()))
                .collect(),
        )
    }

    /// Gather the router's inputs: the triggering post's upstream thread slice
    /// and one card per planned participant.
    async fn router_inputs(
        &self,
        conn: &turso::Connection,
        space_id: &str,
        post_action_id: &str,
        turns: &[crate::PlannedTurn],
    ) -> Result<(Vec<RouterThreadLine>, Vec<RouterCandidate>), AppError> {
        // The same branch-scoped, item-tip-resolved ancestry a turn would send
        // upstream — inclusive of the triggering post — trimmed to its tail.
        let rows = db::get_upstream_context(conn, post_action_id, true).await?;
        let skip = rows.len().saturating_sub(THREAD_SLICE_POSTS);
        let thread: Vec<RouterThreadLine> = rows
            .into_iter()
            .skip(skip)
            .filter_map(|r| {
                r.text_content.map(|text| RouterThreadLine {
                    author: r.participant_label,
                    text,
                })
            })
            .collect();

        let members = db::space_participants(conn, space_id).await?;
        let candidates = turns
            .iter()
            .filter_map(|t| {
                let m = members
                    .iter()
                    .find(|m| m.participant_id == t.participant_id)?;
                Some(RouterCandidate {
                    participant_id: m.participant_id.clone(),
                    label: m.label.clone(),
                    persona_digest: persona_digest(m.system_prompt.as_deref()),
                    notify_policy: m.notify_policy.clone(),
                })
            })
            .collect();
        Ok((thread, candidates))
    }

    /// One non-streaming chat completion against the router model, routed
    /// through the ordinary backend registry.
    ///
    /// This is **not a turn**: nothing is persisted, no action is written, no
    /// engine lease outlives the call. Engine-backed and `openai` backends take
    /// the zero-spend path; an `eidola` backend spends a credential and settles
    /// its refund exactly like a turn's round would.
    async fn router_completion(
        &self,
        model_ref: &str,
        system: &str,
        user: &str,
    ) -> Result<String, AppError> {
        let cfg = self.load_config();
        let now = now_ms();
        let db_conn = self.db_conn().await?;

        let mref = backends::parse_model_ref(model_ref);
        let backend = self.require_backend(&db_conn, &mref.backend_id).await?;
        let kind =
            backends::BackendKind::parse(&backend.kind).ok_or_else(|| AppError::Database {
                message: format!("unknown backend kind `{}`", backend.kind),
            })?;
        let canonical = backends::qualified_model_id(&mref.model, &backend.id);

        let messages = vec![
            serde_json::json!({ "role": "system", "content": system }),
            serde_json::json!({ "role": "user", "content": user }),
        ];

        // Held for the life of the call so the engine is not evicted underneath
        // it; dropped on return.
        let mut _engine_lease: Option<local_models::EngineLease> = None;

        let (client, base_url, wire_model, pricing, external_auth) = match kind {
            backends::BackendKind::Local | backends::BackendKind::LlamaCpp => {
                let (engine_url, _ctx, lease) =
                    match self.local.lease_engine(&backend.id, &mref.model) {
                        Some(leased) => leased,
                        None => {
                            if kind == backends::BackendKind::LlamaCpp && !backend.auto_start {
                                return Err(AppError::NotConfigured {
                                    message: format!(
                                        "router model `{canonical}` is not loaded and backend \
                                         `{}` has auto-start disabled",
                                        backend.id
                                    ),
                                });
                            }
                            self.load_local_model(&canonical).await?;
                            self.local
                                .lease_engine(&backend.id, &mref.model)
                                .ok_or_else(|| AppError::LocalModel {
                                    message: format!(
                                        "router model `{canonical}` was unloaded while starting"
                                    ),
                                })?
                        }
                    };
                _engine_lease = Some(lease);
                let client = match &self.http_override {
                    Some(c) => c.clone(),
                    None => local_models::plain_http_client()?,
                };
                (client, engine_url, canonical.clone(), None, None)
            }
            backends::BackendKind::OpenAi => {
                let base_url = backend
                    .base_url
                    .clone()
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("backend `{}` has no base URL", backend.id),
                    })?;
                let client = match &self.http_override {
                    Some(c) => c.clone(),
                    None => local_models::plain_http_client()?,
                };
                let auth = backend.api_key.as_ref().map(|k| format!("Bearer {k}"));
                (client, base_url, mref.model.clone(), None, auth)
            }
            backends::BackendKind::Eidola => {
                let eidola = EidolaResolved::from_row(Some(&backend))?;
                // No attestation observer: the handshake is still verified,
                // but a router call writes no rows to hang a record from.
                let client = self.build_client(&cfg, &eidola, None).await?;
                let models = fetch_models(&client, &eidola.base_url).await?;
                let entry = models
                    .data
                    .iter()
                    .find(|m| m.id == mref.model)
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("router model not found: {canonical}"),
                    })?;
                let pricing = (
                    entry.pricing.per_prompt_token.value as u128,
                    entry.pricing.per_completion_token.value as u128,
                    entry.pricing.per_prompt_token.scale_factor as u128,
                );
                (
                    client,
                    eidola.base_url.clone(),
                    mref.model.clone(),
                    Some(pricing),
                    None,
                )
            }
        };

        // The spend side runs only for a remote (eidola) router.
        let mut spend = None;
        let auth_value = match pricing {
            None => external_auth,
            Some(pricing) => {
                let charge = estimate_charge_credits(&messages, MAX_COMPLETION_TOKENS, pricing);
                if charge == 0 {
                    return Err(AppError::Credential {
                        message: "computed router charge is zero — model pricing may be missing"
                            .into(),
                    });
                }
                let (prep, auth) = self.acquire_spend(&cfg, &db_conn, charge, now).await?;
                spend = Some(prep);
                Some(auth)
            }
        };

        let body = serde_json::json!({
            "model": wire_model,
            "messages": messages,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        });

        let mut request = client
            .post(format!("{base_url}/v1/chat/completions"))
            .json(&body);
        if let Some(auth) = &auth_value {
            request = request.header("Authorization", auth);
        }

        let response = match request.send().await {
            Ok(resp) => resp,
            Err(e) => {
                // The server may have received the request: recover the refund
                // rather than abandoning the credential.
                self.settle_router_refund(
                    &db_conn,
                    &spend,
                    &auth_value,
                    &client,
                    &base_url,
                    None,
                    now,
                )
                .await;
                return Err(AppError::from_request(e));
            }
        };
        let status = response.status();
        let text = response.text().await.map_err(|e| AppError::Network {
            message: format!("failed to read the router response: {e}"),
        })?;
        let parsed: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);

        self.settle_router_refund(
            &db_conn,
            &spend,
            &auth_value,
            &client,
            &base_url,
            parsed.get("refund"),
            now,
        )
        .await;

        if !status.is_success() {
            return Err(AppError::Server {
                status: status.as_u16(),
                message: crate::parse_server_error_message(&text),
            });
        }

        Ok(parsed
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or_default()
            .to_string())
    }

    /// Settle a remote router call's credential: apply the inline refund when
    /// the response carried one, otherwise try the recovery endpoint. A no-op
    /// for the zero-spend backends. Best-effort throughout — a router call must
    /// never turn a wallet hiccup into a failed post.
    #[allow(clippy::too_many_arguments)]
    async fn settle_router_refund(
        &self,
        db_conn: &turso::Connection,
        spend: &Option<crate::SpendPrep>,
        auth_value: &Option<String>,
        client: &reqwest::Client,
        base_url: &str,
        inline: Option<&serde_json::Value>,
        now: i64,
    ) {
        let Some(spend) = spend else { return };
        let refund_obj = match inline {
            Some(obj) => Some(obj.clone()),
            None => match auth_value {
                Some(auth) => recover_refund(client, base_url, auth).await.ok(),
                None => None,
            },
        };
        let Some(refund_obj) = refund_obj else {
            eprintln!("warning: the may-decline router's credential refund could not be recovered");
            return;
        };
        let applied = process_refund(
            &refund_obj,
            &spend.params,
            &spend.spend_proof,
            &spend.pre_refund,
            &spend.public_key,
            db_conn,
            &spend.pre_cred_id,
            spend.cred.generation + 1,
            now,
        )
        .await;
        match applied {
            Ok(()) => self.bus.emit(Change::Wallet),
            Err(e) => eprintln!("warning: the may-decline router's refund failed to apply: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(n: usize) -> Vec<RouterCandidate> {
        (1..=n)
            .map(|i| RouterCandidate {
                participant_id: format!("p{i}"),
                label: format!("Agent {i}"),
                persona_digest: None,
                notify_policy: "all".into(),
            })
            .collect()
    }

    #[test]
    fn parses_a_bare_object() {
        assert_eq!(
            parse_router_selection(r#"{"notify": [1, 3]}"#, 3).unwrap(),
            vec![1, 3]
        );
    }

    #[test]
    fn parses_a_fenced_object_with_prose_around_it() {
        let reply = "Sure!\n```json\n{\"notify\": [2]}\n```\nHope that helps.";
        assert_eq!(parse_router_selection(reply, 3).unwrap(), vec![2]);
    }

    #[test]
    fn an_empty_selection_is_valid_and_means_nobody() {
        assert!(
            parse_router_selection(r#"{"notify": []}"#, 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn numeric_strings_are_accepted_and_duplicates_collapse() {
        assert_eq!(
            parse_router_selection(r#"{"notify": ["2", 2, 1]}"#, 3).unwrap(),
            vec![2, 1]
        );
    }

    #[test]
    fn out_of_range_entries_are_ignored_not_fatal() {
        assert_eq!(
            parse_router_selection(r#"{"notify": [0, 1, 9]}"#, 3).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn unusable_output_is_an_error_so_the_caller_degrades() {
        for reply in [
            "I think Agent 2 should answer.",
            "{oops",
            r#"{"notified": [1]}"#,
            r#"{"notify": "everyone"}"#,
        ] {
            assert!(
                parse_router_selection(reply, 3).is_err(),
                "expected an error for {reply:?}"
            );
        }
    }

    #[test]
    fn the_prompt_numbers_candidates_and_marks_the_latest_post() {
        let thread = vec![
            RouterThreadLine {
                author: "You".into(),
                text: "first".into(),
            },
            RouterThreadLine {
                author: "You".into(),
                text: "second".into(),
            },
        ];
        let prompt = build_router_prompt(&thread, &candidates(2));
        assert!(prompt.contains("You: first"));
        assert!(prompt.contains("LATEST — You: second"));
        assert!(prompt.contains("1. Agent 1"));
        assert!(prompt.contains("2. Agent 2"));
        assert!(prompt.contains("[notify policy: all]"));
    }

    #[test]
    fn persona_digests_are_one_line_and_clipped() {
        assert_eq!(persona_digest(None), None);
        assert_eq!(persona_digest(Some("   ")), None);
        assert_eq!(
            persona_digest(Some("You are\na careful\treviewer.")).unwrap(),
            "You are a careful reviewer."
        );
        let long = "x".repeat(PERSONA_MAX_BYTES + 50);
        let digest = persona_digest(Some(&long)).unwrap();
        assert!(digest.ends_with('…'));
        assert!(digest.len() <= PERSONA_MAX_BYTES + 4);
    }

    #[test]
    fn clipping_respects_char_boundaries() {
        // A 3-byte char straddling the budget must not panic or split.
        let text = "★".repeat(10);
        let clipped = clip(&text, 8);
        assert!(clipped.ends_with('…'));
        assert!(clipped.is_char_boundary(clipped.len() - '…'.len_utf8()));
    }
}
