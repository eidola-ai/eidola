//! The **harness chore runner** — one cheap model call that is deliberately
//! *not* a turn.
//!
//! A chore is a call the harness makes on its own behalf: routing a
//! notification set ([`crate::router`]), summarizing a branch
//! ([`crate::summaries`]). It routes through the ordinary backend registry, so
//! an engine-backed reference (`local` / `llamacpp`) takes the zero-spend path
//! — no charge, no credential, no account — which is what makes a local chore
//! runner genuinely free, while an `eidola` reference spends and settles a
//! credential exactly like a turn's round would.
//!
//! What a chore is **not**: no actions, no context assembly, no request rows,
//! no attestation records. (For an `eidola` chore the per-handshake attestation
//! still *happens* — it is simply not recorded, since there is no action to
//! hang the forensic trail from.)
//!
//! # The two-phase seam
//!
//! [`Inner::resolve_utility_target`] answers *where would this go* — cheap, no
//! engine start, no network, so an unresolvable model reference degrades before
//! anything is started. [`Inner::utility_completion`] then opens the route
//! (leasing or starting an engine, building the client, placing a hold for a
//! remote call), makes the call, and settles. The summarizer uses that split to
//! decide what is stale before it starts an engine at all.

use crate::error::AppError;
use crate::{
    Change, EidolaResolved, Inner, backends, db, estimate_charge_credits, fetch_models,
    local_models, now_ms, process_refund, recover_refund,
};

/// Where a chore call would go, and what it would cost — resolved without
/// starting an engine or opening a connection.
pub(crate) struct UtilityTarget {
    pub(crate) backend: db::BackendRow,
    pub(crate) kind: backends::BackendKind,
    /// The model id the backend's own API expects.
    pub(crate) model: String,
    /// The canonical `<model>@<backend>` selection string.
    pub(crate) canonical: String,
    /// What to call this chore in a diagnostic ("router", "branch summary").
    pub(crate) chore: &'static str,
}

/// An opened route: everything one HTTP call needs, plus the engine lease that
/// must outlive it.
struct UtilityRoute {
    client: reqwest::Client,
    base_url: String,
    wire_model: String,
    /// `(prompt_rate, completion_rate, scale_factor)` — `Some` only for a
    /// remote (billing) call.
    pricing: Option<(u128, u128, u128)>,
    /// An external backend's bearer key, when it has one.
    external_auth: Option<String>,
    /// Held for the life of the call so the engine is not evicted underneath
    /// it; dropped on return.
    #[allow(dead_code)]
    engine_lease: Option<local_models::EngineLease>,
}

/// Truncate to at most `max_bytes`, on a char boundary, marking the cut.
/// Shared by every chore that clips conversation text into a prompt.
pub(crate) fn clip(text: &str, max_bytes: usize) -> String {
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

/// Share of an over-budget post kept from its **head**; the rest is kept from
/// its tail. A post opens with its subject and closes with its ask, and the
/// opening is the half that places it, so the head gets the larger share.
const CLIP_HEAD_SHARE: (usize, usize) = (2, 3);

/// Clip a **post body** to at most `max_bytes` by eliding its *middle* rather
/// than its tail, on char boundaries, marking the cut. Shared by every chore
/// that puts a whole post inside a fixed per-post budget.
///
/// A post body is not a digest — it is a *rendered* post ([`crate::
/// render_post_for_model`]), and its `{{ embed N }}` markers have already
/// expanded into the passages they name. A quoted passage is a range the
/// author chose and nothing bounds its length, so a marker standing before the
/// post's own words can push everything the author actually wrote past a
/// head-only budget: `{{ embed 1 }}\n\nHave the legal reviewer assess this`
/// spends the whole budget on the quotation and reaches the model without the
/// cue it was written to carry. That is a *routing* decision made from someone
/// else's words, and a branch summary — written down and read for as long as
/// the branch lives — that describes the passage while dropping the author's
/// disagreement with it.
///
/// Keeping both ends is the cure that needs no arithmetic about how a
/// rendering will lay out: the expansion sits between the author's words in
/// the shapes that matter (before them, after them, or between two of them),
/// so eliding the middle spends the budget on the quotation only after both
/// ends are paid for. It is also the rule the summarizer already applies one
/// level up, where an over-cap *branch* is sliced head + tail rather than
/// oldest-N. The case it cannot save is prose sandwiched between two long
/// quotations, which loses the middle like any other over-budget text.
pub(crate) fn clip_middle(text: &str, max_bytes: usize) -> String {
    let text = text.trim();
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let (num, denom) = CLIP_HEAD_SHARE;
    let mut head = max_bytes * num / denom;
    while head > 0 && !text.is_char_boundary(head) {
        head -= 1;
    }
    // `text.len() > max_bytes >= max_bytes - head`, so the tail starts at or
    // after the head's end and the two never overlap.
    let mut tail = text.len() - (max_bytes - head);
    while tail < text.len() && !text.is_char_boundary(tail) {
        tail += 1;
    }
    format!("{}…{}", &text[..head], &text[tail..])
}

impl Inner {
    /// Resolve a chore's model reference through the backend registry.
    ///
    /// Pure resolution: no engine is started, no client is built, nothing is
    /// spent. Fails when the backend is unknown, removed, disabled, or of an
    /// unrecognized kind.
    pub(crate) async fn resolve_utility_target(
        &self,
        db_conn: &turso::Connection,
        model_ref: &str,
        chore: &'static str,
    ) -> Result<UtilityTarget, AppError> {
        let mref = backends::parse_model_ref(model_ref);
        let backend = self.require_backend(db_conn, &mref.backend_id).await?;
        let kind =
            backends::BackendKind::parse(&backend.kind).ok_or_else(|| AppError::Database {
                message: format!("unknown backend kind `{}`", backend.kind),
            })?;
        let canonical = backends::qualified_model_id(&mref.model, &backend.id);
        Ok(UtilityTarget {
            backend,
            kind,
            model: mref.model,
            canonical,
            chore,
        })
    }

    /// Open the route: lease or start the engine, build the client, and read
    /// the remote catalog's pricing when the call bills.
    async fn open_utility_route(&self, target: &UtilityTarget) -> Result<UtilityRoute, AppError> {
        let backend = &target.backend;
        match target.kind {
            backends::BackendKind::Local | backends::BackendKind::LlamaCpp => {
                let (engine_url, _ctx, lease) = match self
                    .local
                    .lease_engine(&backend.id, &target.model)
                {
                    Some(leased) => leased,
                    None => {
                        if target.kind == backends::BackendKind::LlamaCpp && !backend.auto_start {
                            return Err(AppError::NotConfigured {
                                message: format!(
                                    "{} model `{}` is not loaded and backend `{}` has auto-start \
                                     disabled",
                                    target.chore, target.canonical, backend.id
                                ),
                            });
                        }
                        self.load_local_model(&target.canonical).await?;
                        self.local
                            .lease_engine(&backend.id, &target.model)
                            .ok_or_else(|| AppError::LocalModel {
                                message: format!(
                                    "{} model `{}` was unloaded while starting",
                                    target.chore, target.canonical
                                ),
                            })?
                    }
                };
                Ok(UtilityRoute {
                    client: self.plain_client()?,
                    base_url: engine_url,
                    wire_model: target.canonical.clone(),
                    pricing: None,
                    external_auth: None,
                    engine_lease: Some(lease),
                })
            }
            backends::BackendKind::OpenAi => {
                let base_url = backend
                    .base_url
                    .clone()
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("backend `{}` has no base URL", backend.id),
                    })?;
                Ok(UtilityRoute {
                    client: self.plain_client()?,
                    base_url,
                    wire_model: target.model.clone(),
                    pricing: None,
                    external_auth: backend.api_key.as_ref().map(|k| format!("Bearer {k}")),
                    engine_lease: None,
                })
            }
            backends::BackendKind::Eidola => {
                let eidola = EidolaResolved::from_row(Some(backend))?;
                // No attestation observer: the handshake is still verified,
                // but a chore call writes no rows to hang a record from.
                let client = self.build_client(&eidola, None).await?;
                let models = fetch_models(&client, &eidola.base_url).await?;
                let entry = models
                    .data
                    .iter()
                    .find(|m| m.id == target.model)
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("{} model not found: {}", target.chore, target.canonical),
                    })?;
                Ok(UtilityRoute {
                    client,
                    base_url: eidola.base_url.clone(),
                    wire_model: target.model.clone(),
                    pricing: Some((
                        entry.pricing.per_prompt_token.value as u128,
                        entry.pricing.per_completion_token.value as u128,
                        entry.pricing.per_prompt_token.scale_factor as u128,
                    )),
                    external_auth: None,
                    engine_lease: None,
                })
            }
        }
    }

    /// One non-streaming chat completion against a resolved chore target.
    ///
    /// Opens the route, places and settles a hold when the target bills, and
    /// returns the assistant's content. Persists nothing.
    pub(crate) async fn utility_completion(
        &self,
        db_conn: &turso::Connection,
        target: &UtilityTarget,
        system: &str,
        user: &str,
        max_completion_tokens: u32,
    ) -> Result<String, AppError> {
        let cfg = self.load_config();
        let now = now_ms();
        let route = self.open_utility_route(target).await?;

        let messages = vec![
            serde_json::json!({ "role": "system", "content": system }),
            serde_json::json!({ "role": "user", "content": user }),
        ];

        // The spend side runs only for a remote (eidola) target.
        let mut spend = None;
        let auth_value = match route.pricing {
            None => route.external_auth.clone(),
            Some(pricing) => {
                // A chore call advertises no tools, so the tool-schema term of
                // the pricing contract is the empty slice.
                let charge =
                    estimate_charge_credits(&messages, &[], max_completion_tokens, pricing);
                if charge == 0 {
                    return Err(AppError::Credential {
                        message: format!(
                            "computed {} charge is zero — model pricing may be missing",
                            target.chore
                        ),
                    });
                }
                let (prep, auth) = self.acquire_spend(&cfg, db_conn, charge, now).await?;
                spend = Some(prep);
                Some(auth)
            }
        };

        let body = serde_json::json!({
            "model": route.wire_model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        });

        let mut request = route
            .client
            .post(format!("{}/v1/chat/completions", route.base_url))
            .json(&body);
        if let Some(auth) = &auth_value {
            request = request.header("Authorization", auth);
        }

        let response = match request.send().await {
            Ok(resp) => resp,
            Err(e) => {
                // The server may have received the request: recover the refund
                // rather than abandoning the credential.
                self.settle_utility_refund(db_conn, &spend, &auth_value, &route, None, now)
                    .await;
                return Err(AppError::from_request(e));
            }
        };
        let status = response.status();
        // The body read can fail *after* the request was accepted (a connection
        // dropped mid-body). The hold is already placed at that point, so this
        // must settle before it returns — exactly the discipline the chat
        // transports keep on their own body-read failures. Leaking out of here
        // would strand the credential in `spending`, and the very next turn
        // would burn its bounded provisioning wait on a refund that is never
        // coming.
        let text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                self.settle_utility_refund(db_conn, &spend, &auth_value, &route, None, now)
                    .await;
                return Err(AppError::Network {
                    message: format!(
                        "failed to read the {} response: {}",
                        target.chore,
                        crate::error::request_error_text(e)
                    ),
                });
            }
        };
        let parsed: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);

        self.settle_utility_refund(
            db_conn,
            &spend,
            &auth_value,
            &route,
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

    /// Settle a remote chore call's credential: apply the inline refund when
    /// the response carried one, otherwise try the recovery endpoint. A no-op
    /// for the zero-spend backends. Best-effort throughout — a chore must never
    /// turn a wallet hiccup into a failed post.
    async fn settle_utility_refund(
        &self,
        db_conn: &turso::Connection,
        spend: &Option<crate::SpendPrep>,
        auth_value: &Option<String>,
        route: &UtilityRoute,
        inline: Option<&serde_json::Value>,
        now: i64,
    ) {
        let Some(spend) = spend else { return };
        let refund_obj = match inline {
            Some(obj) => Some(obj.clone()),
            None => match auth_value {
                Some(auth) => recover_refund(&route.client, &route.base_url, auth)
                    .await
                    .ok(),
                None => None,
            },
        };
        let Some(refund_obj) = refund_obj else {
            eprintln!("warning: a utility call's credential refund could not be recovered");
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
            Err(e) => eprintln!("warning: a utility call's refund failed to apply: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_respects_char_boundaries() {
        // A 3-byte char straddling the budget must not panic or split.
        let text = "★".repeat(10);
        let clipped = clip(&text, 8);
        assert!(clipped.ends_with('…'));
        assert!(clipped.is_char_boundary(clipped.len() - '…'.len_utf8()));
    }

    #[test]
    fn clipping_leaves_short_text_alone() {
        assert_eq!(clip("  hello  ", 32), "hello");
    }

    #[test]
    fn a_middle_clip_keeps_both_ends_within_budget() {
        let text = format!("OPEN{}CLOSE", "x".repeat(500));
        let clipped = clip_middle(&text, 60);
        assert!(clipped.starts_with("OPEN"), "got {clipped:?}");
        assert!(clipped.ends_with("CLOSE"), "got {clipped:?}");
        assert!(clipped.contains('…'), "the cut is marked; got {clipped:?}");
        // The ellipsis is the only thing over budget, as with `clip`.
        assert!(clipped.len() <= 60 + '…'.len_utf8(), "got {clipped:?}");
    }

    #[test]
    fn a_middle_clip_respects_char_boundaries_at_both_cuts() {
        // Multi-byte chars straddling both the head cut and the tail cut.
        let text = "★".repeat(40);
        let clipped = clip_middle(&text, 8);
        assert!(clipped.contains('…'));
        assert!(clipped.chars().all(|c| c == '★' || c == '…'));
    }

    #[test]
    fn a_middle_clip_leaves_text_that_fits_alone() {
        assert_eq!(clip_middle("  hello  ", 32), "hello");
        // Exactly at budget is not a cut.
        assert_eq!(clip_middle("hello", 5), "hello");
    }
}
