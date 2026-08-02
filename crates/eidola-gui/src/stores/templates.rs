//! `TemplatesStore` — the space-template registry (the Settings → Space
//! Templates pane's data source).
//!
//! Per `crates/eidola-gui/STATE.md`: one gpui entity owning the
//! `Loadable<Vec<SpaceTemplateInfo>>` snapshot, one supersede task slot (shared
//! by the list refresh and the write-through CRUD ops, which compose
//! `write; re-list` so the store reconciles even without the bus), its
//! subscription to `Change::Templates` (routed in `stores::dispatch_change`),
//! and *all* mutations of the template domain. Views call these methods; they
//! never touch `AppCore` directly.

use std::sync::Arc;

use eidola_app_core::{AppCore, NewTemplateParticipant, SpaceTemplateInfo};
use gpui::{Context, Task};

use crate::bridge::bridge;
use crate::loadable::Loadable;

pub struct TemplatesStore {
    app_core: Option<Arc<AppCore>>,
    templates: Loadable<Vec<SpaceTemplateInfo>>,
    /// Supersede slot for the list refresh and every write-through CRUD op
    /// (each composes `write; re-list`, replace-cancels).
    task: Option<Task<()>>,
    /// The last write error, surfaced by the pane's op-error banner.
    op_error: Option<String>,
}

impl TemplatesStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        Self {
            app_core,
            templates: Loadable::NotLoaded,
            task: None,
            op_error: None,
        }
    }

    /// A stub store with a fixture listing (tests).
    pub fn stub(templates: Vec<SpaceTemplateInfo>) -> Self {
        Self {
            app_core: None,
            templates: if templates.is_empty() {
                Loadable::NotLoaded
            } else {
                Loadable::loaded(templates)
            },
            task: None,
            op_error: None,
        }
    }

    pub fn templates(&self) -> &Loadable<Vec<SpaceTemplateInfo>> {
        &self.templates
    }

    /// Test-only: replace the registry snapshot, standing in for the refresh a
    /// `Change::Templates` / `Change::Participants` dispatch would drive on a
    /// backed store. Lets a stub-backed test move the registry *under* an open
    /// editor.
    #[doc(hidden)]
    pub fn set_templates_for_test(
        &mut self,
        templates: Vec<SpaceTemplateInfo>,
        cx: &mut Context<Self>,
    ) {
        self.templates = Loadable::loaded(templates);
        cx.notify();
    }

    /// Test-only: force the registry into `Failed` (no prior) to exercise the
    /// failed-initial-load rendering.
    #[doc(hidden)]
    pub fn set_failed_for_test(&mut self, error: &str) {
        self.templates = Loadable::Failed {
            error: eidola_app_core::error::AppError::Config {
                message: error.to_string(),
            },
            prior: None,
        };
    }

    pub fn list(&self) -> &[SpaceTemplateInfo] {
        self.templates.value().map(|v| v.as_slice()).unwrap_or(&[])
    }

    pub fn op_error(&self) -> Option<&str> {
        self.op_error.as_deref()
    }

    pub fn clear_op_error(&mut self, cx: &mut Context<Self>) {
        if self.op_error.take().is_some() {
            cx.notify();
        }
    }

    /// Refresh the registry. Fire-and-notify supersede slot.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.templates = std::mem::take(&mut self.templates).to_loading();
        self.task = Some(cx.spawn(async move |this, cx| {
            let result = bridge(core, |c| async move { c.list_space_templates().await }).await;
            let _ = this.update(cx, |this, cx| {
                this.templates = std::mem::take(&mut this.templates).resolve(result);
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Run a write-through op (its result discarded) then re-list, updating the
    /// snapshot and op-error on completion. The shared shape behind every CRUD
    /// method: the bus's `Change::Templates` would also refresh, but composing
    /// the re-list here keeps the store correct on a bus-less test run.
    ///
    /// **The re-list runs whether the op succeeded or failed**, because a
    /// router-bearing create/update is two core calls and can therefore fail
    /// *partially* — a template created whose router the setter then refused.
    /// Leaving the snapshot untouched on failure would show the error beside a
    /// listing that doesn't contain the template that was in fact created. The
    /// error still wins the `op_error` slot; a re-list that itself fails leaves
    /// the prior snapshot in place rather than replacing the write's error with
    /// a read's.
    fn write_then_relist<F>(&mut self, cx: &mut Context<Self>, op: F)
    where
        F: FnOnce(
                Arc<AppCore>,
            ) -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<(), String>> + Send>,
            > + Send
            + 'static,
    {
        let Some(core) = self.app_core.clone() else {
            return;
        };
        self.op_error = None;
        let relist_core = core.clone();
        self.task = Some(cx.spawn(async move |this, cx| {
            let op_result = op(core).await;
            let list = bridge(
                relist_core,
                |c| async move { c.list_space_templates().await },
            )
            .await;
            let _ = this.update(cx, |this, cx| {
                if let Ok(templates) = list {
                    this.templates = Loadable::loaded(templates);
                }
                if let Err(e) = op_result {
                    this.op_error = Some(e);
                }
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    /// Create a template. `router_model` is the may-decline router reference
    /// (`None` = off, the default); it rides through the dedicated
    /// [`AppCore::set_template_router_model`] setter, which is why this is a
    /// two-call write.
    ///
    /// **Both calls live in ONE [`bridge`] closure — one tokio future — and
    /// that is load-bearing, not tidiness.** `refresh` and every CRUD op share
    /// this store's single task slot, and `create_template` emits
    /// `Change::Templates` core-side *before it returns*; the bus dispatch that
    /// event drives calls `refresh`, which replaces the slot and drops the gpui
    /// half of this op. Split across two `bridge` calls, the second one is
    /// constructed only after the first await returns — so a refunded slot
    /// mid-op meant the template was created with its router silently left
    /// NULL. `bridge` spawns onto the core's tokio runtime and drops the
    /// `JoinHandle`, so a dropped gpui receiver cancels nothing core-side
    /// (see `bridge.rs`): as one future, both writes complete regardless.
    pub fn create(
        &mut self,
        title: String,
        cascade_limit: i64,
        participants: Vec<NewTemplateParticipant>,
        router_model: Option<String>,
        cx: &mut Context<Self>,
    ) {
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    // Check the reference BEFORE the create. Setting the router
                    // is a separate call that validates its backend, so a stale
                    // pick — the backend disabled or removed while the editor was
                    // open — would otherwise land a template *without* the router
                    // it was created with. Validating first makes that a
                    // zero-trace refusal; what's left is a TOCTOU of microseconds
                    // inside one future.
                    if router_model.is_some() {
                        c.validate_router_model(router_model.clone()).await?;
                    }
                    let created = c
                        .create_template(title, cascade_limit, participants)
                        .await?;
                    // Off is the default, so an unset router needs no second write.
                    if router_model.is_some() {
                        c.set_template_router_model(created.id, router_model)
                            .await?;
                    }
                    Ok(())
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    /// Update a template. `router_model` follows the "untouched field" idiom of
    /// the other arguments: the outer `None` leaves the setting alone, `Some(r)`
    /// writes it (`Some(None)` = off).
    ///
    /// Both writes share one `bridge` closure for the reason spelled out on
    /// [`Self::create`] — a bus-driven `refresh` landing between them would
    /// otherwise drop the router write — and the router reference is validated
    /// before the update for the same reason it is before a create: the two
    /// calls cannot be one transaction, so a refused router must not follow an
    /// applied edit.
    pub fn update(
        &mut self,
        id: String,
        title: Option<String>,
        cascade_limit: Option<i64>,
        participants: Option<Vec<NewTemplateParticipant>>,
        router_model: Option<Option<String>>,
        cx: &mut Context<Self>,
    ) {
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    if let Some(router_model) = router_model.clone().flatten() {
                        c.validate_router_model(Some(router_model)).await?;
                    }
                    c.update_template(id.clone(), title, cascade_limit, participants)
                        .await?;
                    if let Some(router_model) = router_model {
                        c.set_template_router_model(id, router_model).await?;
                    }
                    Ok(())
                })
                .await
                .map_err(|e| e.to_string())
            })
        });
    }

    pub fn remove(&mut self, id: String, cx: &mut Context<Self>) {
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move { c.remove_template(id).await })
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            })
        });
    }

    /// "Make template from this space" — project a space's participants +
    /// settings into a new template.
    pub fn create_from_space(&mut self, space_id: String, title: String, cx: &mut Context<Self>) {
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.template_from_space(space_id, title).await
                })
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
            })
        });
    }
}
