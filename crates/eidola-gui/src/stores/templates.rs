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
            // Re-list only on success; a failed op leaves the current snapshot
            // in place and surfaces the error.
            let list = match &op_result {
                Ok(()) => Some(
                    bridge(
                        relist_core,
                        |c| async move { c.list_space_templates().await },
                    )
                    .await,
                ),
                Err(_) => None,
            };
            let _ = this.update(cx, |this, cx| {
                match op_result {
                    Ok(()) => {
                        if let Some(Ok(templates)) = list {
                            this.templates = Loadable::loaded(templates);
                        }
                    }
                    Err(e) => this.op_error = Some(e),
                }
                this.task = None;
                cx.notify();
            });
        }));
        cx.notify();
    }

    pub fn create(
        &mut self,
        title: String,
        cascade_limit: i64,
        participants: Vec<NewTemplateParticipant>,
        cx: &mut Context<Self>,
    ) {
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.create_template(title, cascade_limit, participants).await
                })
                .await
                .map(|_| ())
                .map_err(|e| e.to_string())
            })
        });
    }

    pub fn update(
        &mut self,
        id: String,
        title: Option<String>,
        cascade_limit: Option<i64>,
        participants: Option<Vec<NewTemplateParticipant>>,
        cx: &mut Context<Self>,
    ) {
        self.write_then_relist(cx, move |core| {
            Box::pin(async move {
                bridge(core, move |c| async move {
                    c.update_template(id, title, cascade_limit, participants)
                        .await
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
