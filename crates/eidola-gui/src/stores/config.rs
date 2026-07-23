//! `ConfigStore` — the synchronous, write-through projection of app-core's
//! config. Owns the `ConfigState` snapshot and all config mutations
//! (base URL, default model, attestation URL, …). Reads are synchronous;
//! writes go straight through to `AppCore` and re-read the snapshot. On a
//! real backend each write also emits `Change::Config`, so other windows'
//! `ConfigStore`s refresh via the bus.
//!
//! Per `crates/eidola-gui/STATE.md`: `ConfigState` is not a `Loadable` — it is
//! always present (read synchronously at construction) on a backed store, and
//! `None` only on a stub. The store keeps a `last_error` for write failures so
//! the settings panes can surface them.

use std::sync::Arc;

use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ConfigState, EidolaTrust};
use gpui::Context;

pub struct ConfigStore {
    app_core: Option<Arc<AppCore>>,
    /// The current config snapshot. `Some` on a backed store (seeded at
    /// construction, refreshed on `Change::Config`); `None` on a stub until a
    /// test installs one.
    state: Option<ConfigState>,
    /// The eidola connection + trust bundle (base URL, measurements, hardware
    /// CAs) — read from the `eidola` backend row via `AppCore::eidola_trust`.
    /// Since it moved off `ConfigState` it is fetched separately (blocking on
    /// the core runtime, off the gpui main thread) and re-read on each write.
    trust: Option<EidolaTrust>,
    /// The last config-write error, surfaced by the settings panes.
    error: Option<AppError>,
}

impl ConfigStore {
    pub fn new(app_core: Option<Arc<AppCore>>) -> Self {
        let state = app_core.as_ref().map(|c| c.config_state());
        let trust = app_core
            .as_ref()
            .and_then(|c| c.runtime().block_on(c.eidola_trust()).ok());
        Self {
            app_core,
            state,
            trust,
            error: None,
        }
    }

    /// A stub store for tests, with no backend and the given fixture state.
    pub fn stub(state: Option<ConfigState>) -> Self {
        Self {
            app_core: None,
            state,
            trust: None,
            error: None,
        }
    }

    /// The current config snapshot, if known.
    pub fn state(&self) -> Option<&ConfigState> {
        self.state.as_ref()
    }

    /// The resolved eidola connection + trust bundle, if known.
    pub fn eidola_trust(&self) -> Option<&EidolaTrust> {
        self.trust.as_ref()
    }

    /// Test-only: install a fixture config snapshot.
    #[doc(hidden)]
    pub fn set_state_for_test(&mut self, state: Option<ConfigState>) {
        self.state = state;
    }

    /// Test-only: install a fixture eidola-trust snapshot.
    #[doc(hidden)]
    pub fn set_eidola_trust_for_test(&mut self, trust: Option<EidolaTrust>) {
        self.trust = trust;
    }

    /// The last config-write error, if any.
    pub fn error(&self) -> Option<&AppError> {
        self.error.as_ref()
    }

    pub fn clear_error(&mut self, cx: &mut Context<Self>) {
        if self.error.take().is_some() {
            cx.notify();
        }
    }

    /// Re-read the snapshot from the backend (bus-driven on `Change::Config`,
    /// and called by views after a config-mutating op completes elsewhere).
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        if let Some(core) = self.app_core.as_ref() {
            self.state = Some(core.config_state());
            self.trust = core.runtime().block_on(core.eidola_trust()).ok();
            cx.notify();
        }
    }

    fn write(&mut self, cx: &mut Context<Self>, f: impl FnOnce(&AppCore) -> Result<(), AppError>) {
        let Some(core) = self.app_core.as_ref() else {
            return;
        };
        match f(core) {
            Ok(()) => {
                self.state = Some(core.config_state());
                self.trust = core.runtime().block_on(core.eidola_trust()).ok();
                self.error = None;
            }
            Err(e) => self.error = Some(e),
        }
        cx.notify();
    }

    pub fn set_base_url(&mut self, url: String, cx: &mut Context<Self>) {
        // The setter moved to the eidola backend row (async); block on the
        // core runtime off the gpui main thread.
        self.write(cx, |c| c.runtime().block_on(c.set_base_url(url)));
    }

    pub fn clear_base_url_override(&mut self, cx: &mut Context<Self>) {
        self.write(cx, |c| c.runtime().block_on(c.clear_base_url_override()));
    }

    /// Revert the trusted-measurements override back to the built-in pin.
    /// There is no single "clear" core call — untrusting every entry in the
    /// override list reverts to the pin (clearing the last one does it), so
    /// this reads the current override set off the trust snapshot and
    /// untrusts each. Writes through the eidola backend row; emits
    /// `Change::Backends`.
    pub fn revert_trusted_measurements(&mut self, cx: &mut Context<Self>) {
        self.write(cx, |c| {
            let snps: Vec<String> = c
                .runtime()
                .block_on(c.eidola_trust())
                .map(|t| {
                    t.trusted_measurements
                        .iter()
                        .map(|m| m.snp.clone())
                        .collect()
                })
                .unwrap_or_default();
            for snp in snps {
                c.runtime().block_on(c.untrust_measurement(snp))?;
            }
            Ok(())
        });
    }

    /// Add a trusted enclave measurement to the eidola row's override list.
    /// `spec` is the CLI's `<snp>:<rtmr1>:<rtmr2>` triple; a parse failure is
    /// surfaced in the store's `error` (the settings op-error banner) rather
    /// than silently dropped. Writes through the backend row; emits
    /// `Change::Backends`.
    pub fn trust_measurement(&mut self, spec: String, cx: &mut Context<Self>) {
        self.write(cx, |c| {
            let m = eidola_app_core::config::parse_trust_measurement(spec.trim())?;
            c.runtime()
                .block_on(c.trust_measurement(
                    m.snp_measurement,
                    m.tdx_measurement.rtmr1,
                    m.tdx_measurement.rtmr2,
                ))
                .map(|_| ())
        });
    }

    /// Remove a trusted measurement (by SNP key) from the eidola row's
    /// override list. Clearing the last one reverts to the pin. Writes through
    /// the backend row; emits `Change::Backends`.
    pub fn untrust_measurement(&mut self, snp: String, cx: &mut Context<Self>) {
        self.write(cx, |c| {
            c.runtime().block_on(c.untrust_measurement(snp)).map(|_| ())
        });
    }

    /// Set the eidola hardware root CA (ARK) PEM override. PEM validation
    /// lives in app-core; a failure lands in the op-error banner.
    pub fn set_hardware_root_ca(&mut self, pem: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.runtime().block_on(c.set_hardware_root_ca(pem)));
    }

    /// Set the eidola hardware intermediate CA (ASK) PEM override.
    pub fn set_hardware_intermediate_ca(&mut self, pem: String, cx: &mut Context<Self>) {
        self.write(cx, |c| {
            c.runtime().block_on(c.set_hardware_intermediate_ca(pem))
        });
    }

    /// Remove the eidola hardware root CA override (row column back to NULL).
    pub fn clear_hardware_root_ca(&mut self, cx: &mut Context<Self>) {
        self.write(cx, |c| c.runtime().block_on(c.clear_hardware_root_ca()));
    }

    /// Remove the eidola hardware intermediate CA override.
    pub fn clear_hardware_intermediate_ca(&mut self, cx: &mut Context<Self>) {
        self.write(cx, |c| {
            c.runtime().block_on(c.clear_hardware_intermediate_ca())
        });
    }

    /// Circadian day/night axis (`appearance`). The theme reacts via its
    /// config observation (`theme::wire_config`), not from here.
    pub fn set_appearance(
        &mut self,
        appearance: eidola_app_core::config::AppearanceSetting,
        cx: &mut Context<Self>,
    ) {
        self.write(cx, |c| c.set_appearance(appearance));
    }

    /// Circadian time-of-day axis (`time_of_day_tint`).
    pub fn set_time_of_day_tint(
        &mut self,
        tint: eidola_app_core::config::TimeOfDayTint,
        cx: &mut Context<Self>,
    ) {
        self.write(cx, |c| c.set_time_of_day_tint(tint));
    }

    /// Circadian fixed light character (`light_character`, used while the
    /// time-of-day axis is off).
    pub fn set_light_character(
        &mut self,
        character: eidola_app_core::config::LightCharacter,
        cx: &mut Context<Self>,
    ) {
        self.write(cx, |c| c.set_light_character(character));
    }

    /// The current resolved type-scale factor (Actual Size = `1.0`); `1.0` on a
    /// stub with no config snapshot.
    pub fn font_scale(&self) -> f32 {
        self.state
            .as_ref()
            .map(|s| s.font_scale)
            .unwrap_or(eidola_app_core::config::FONT_SCALE_DEFAULT)
    }

    /// Persist an explicit type-scale factor (clamped in app-core). The theme
    /// reacts via its config observation (`theme::wire_config`), not from here.
    pub fn set_font_scale(&mut self, scale: f32, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_font_scale(scale));
    }

    /// View → Zoom In: step the type scale up one ladder rung (saturating at the
    /// max). Reads the current scale off the snapshot so it works on any window.
    pub fn zoom_in(&mut self, cx: &mut Context<Self>) {
        let next = eidola_app_core::config::font_scale_step_up(self.font_scale());
        self.set_font_scale(next, cx);
    }

    /// View → Zoom Out: step the type scale down one ladder rung (saturating at
    /// the min).
    pub fn zoom_out(&mut self, cx: &mut Context<Self>) {
        let next = eidola_app_core::config::font_scale_step_down(self.font_scale());
        self.set_font_scale(next, cx);
    }

    /// View → Actual Size: reset the type scale to the designed `1.0`.
    pub fn reset_zoom(&mut self, cx: &mut Context<Self>) {
        self.set_font_scale(eidola_app_core::config::FONT_SCALE_DEFAULT, cx);
    }

    #[allow(dead_code)]
    pub fn set_attestation_url(&mut self, url: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_attestation_url(url));
    }

    #[allow(dead_code)]
    pub fn set_account_credentials(&mut self, id: String, secret: String, cx: &mut Context<Self>) {
        self.write(cx, |c| c.set_account_credentials(id, secret));
    }
}
