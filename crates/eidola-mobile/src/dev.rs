//! Development bundles only (feature `dev`): seat the default agent on an
//! OpenAI-compatible server the developer runs on this machine, so a
//! conversation answers end to end — through the real pipe, the real turn,
//! the real stream — without an account or an Eidola engine. The bundle build
//! writes the server's address into `dev-backend.json` (`mobile/build-bundle.sh
//! --dev-backend URL`); `mobile/tools/openai_stub.py` is such a server.

use eidola_app_core::backends::{BackendKind, NewBackend, qualified_model_id};
use eidola_app_core::config::{AppearanceSetting, TimeOfDayTint};
use eidola_app_core::error::AppError;
use eidola_app_core::updater::ReleaseCheckError;
use eidola_app_core::updater::mobile::PendingBundleUpdate;
use eidola_app_core::{AppCore, DEFAULT_TEMPLATE_ID, NewTemplateParticipant};
use serde::Deserialize;
use std::rc::Rc;

use crate::fixture::{StagedRelease, staged_evidence};
use crate::ui::host::CheckUpdate;

/// The backend id the development server is registered under.
pub const BACKEND_ID: &str = "dev";

#[derive(Debug, Deserialize)]
pub struct DevConfig {
    /// The server's base URL, as the shell's pipe reaches it (the Android
    /// emulator reaches this machine at `10.0.2.2`).
    pub base_url: String,
    /// The model id the server answers as.
    pub model: String,
    /// The appearance setting to store (`system`, `day`, `night`, `auto`):
    /// the UI tests store `system`, so the simulator's own light/dark switch
    /// shows each palette.
    #[serde(default)]
    pub appearance: Option<AppearanceSetting>,
    /// The time-of-day tint to store (`on`, `off`): the UI tests turn it off,
    /// so a screenshot's palette does not depend on the hour it was taken.
    #[serde(default)]
    pub tint: Option<TimeOfDayTint>,
    /// Where the update check reads its release from instead of the release
    /// feed: a [`StagedRelease`] (`mobile/tools/`'s UI-test harness serves
    /// one), taken as verified, so the update screens and the approval run
    /// end to end against a bundle the harness built. Absent: the real feed.
    #[serde(default)]
    pub update_fixtures: Option<String>,
    /// An Eidola gateway to trust in place of the built-in pins: the local
    /// dev stack behind its attesting shim mock (`mobile/tools/account_stack.sh`),
    /// so the first run's account steps run against a real gateway. The same
    /// per-handshake attestation runs, against the mock's chain.
    #[serde(default)]
    pub eidola: Option<DevEidola>,
}

/// What a development bundle trusts for the Eidola backend.
#[derive(Debug, Clone, Deserialize)]
pub struct DevEidola {
    /// The shim's URL, as the shell's pipe reaches it.
    pub base_url: String,
    pub ark_pem: String,
    pub ask_pem: String,
    /// The shim's TLS root, trusted beside the built-in roots.
    pub tls_ca_pem: String,
    pub measurement: String,
    pub host_data: String,
    pub guest_policy: String,
}

impl DevConfig {
    /// The TLS roots this configuration adds (the dev shim's).
    pub fn extra_tls_roots(&self) -> Vec<rustls::pki_types::CertificateDer<'static>> {
        self.eidola
            .iter()
            .filter_map(|e| pem_der(&e.tls_ca_pem).ok())
            .map(rustls::pki_types::CertificateDer::from)
            .collect()
    }
}

fn pem_der(pem: &str) -> Result<Vec<u8>, base64::DecodeError> {
    use base64::Engine as _;
    let body: String = pem.lines().filter(|l| !l.starts_with("-----")).collect();
    base64::engine::general_purpose::STANDARD.decode(body.trim())
}

/// The update check of a development bundle with `update_fixtures`: the
/// release at `url`, read as a [`StagedRelease`] and taken as verified; then
/// the real bundle reader decides what it offers, and the approval the real
/// download, checks and staging.
pub fn fixture_update_check(url: String) -> CheckUpdate {
    Rc::new(move |core: Rc<AppCore>| {
        let url = url.clone();
        Box::pin(async move {
            let fetcher = core
                .update_fetcher()
                .map_err(|e| ReleaseCheckError::Unreachable(e.to_string()))?;
            let bytes = fetcher
                .fetch_url_bounded(&url, "the development update fixture", 1 << 20)
                .await?;
            let release: StagedRelease = serde_json::from_slice(&bytes)
                .map_err(|e| ReleaseCheckError::Rejected(e.to_string()))?;
            let evidence = staged_evidence(&release).map_err(ReleaseCheckError::Rejected)?;
            PendingBundleUpdate::from_evidence(&release.index, evidence)
        })
    })
}

/// Register the server (once) and seat the default template's agent on its
/// model, then flush, so the next launch finds both.
pub async fn seed(core: &AppCore, config: &DevConfig) -> Result<(), AppError> {
    let known = core.list_backends().await?;
    if !known.iter().any(|b| b.id == BACKEND_ID) {
        core.add_backend(NewBackend {
            id: BACKEND_ID.into(),
            kind: BackendKind::OpenAi,
            display_name: "Development server".into(),
            base_url: Some(config.base_url.clone()),
            api_key: None,
            models_dir: None,
            model_overrides: Some(vec![config.model.clone()]),
            engine_path: None,
            auto_start: true,
        })
        .await?;
    }
    core.update_template(
        DEFAULT_TEMPLATE_ID.into(),
        None,
        None,
        Some(vec![NewTemplateParticipant {
            label: "Assistant".into(),
            model_ref: Some(qualified_model_id(&config.model, BACKEND_ID)),
            system_prompt: None,
            notify_policy: "human".into(),
        }]),
    )
    .await?;
    if let Some(eidola) = &config.eidola {
        core.set_base_url(eidola.base_url.clone()).await?;
        core.set_hardware_root_ca(eidola.ark_pem.clone()).await?;
        core.set_hardware_intermediate_ca(eidola.ask_pem.clone())
            .await?;
        core.trust_measurement(
            eidola.measurement.clone(),
            eidola.host_data.clone(),
            eidola.guest_policy.clone(),
        )
        .await?;
    }
    if let Some(appearance) = config.appearance {
        core.set_appearance(appearance)?;
    }
    if let Some(tint) = config.tint {
        core.set_time_of_day_tint(tint)?;
    }
    core.flush_storage().await
}
