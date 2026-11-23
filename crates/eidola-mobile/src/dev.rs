//! Development bundles only (feature `dev`): seat the default agent on an
//! OpenAI-compatible server the developer runs on this machine, so a
//! conversation answers end to end — through the real pipe, the real turn,
//! the real stream — without an account or an Eidola engine. The bundle build
//! writes the server's address into `dev-backend.json` (`mobile/build-bundle.sh
//! --dev-backend URL`); `mobile/tools/openai_stub.py` is such a server.

use eidola_app_core::backends::{BackendKind, NewBackend, qualified_model_id};
use eidola_app_core::config::{AppearanceSetting, TimeOfDayTint};
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, DEFAULT_TEMPLATE_ID, NewTemplateParticipant};
use serde::Deserialize;

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
    if let Some(appearance) = config.appearance {
        core.set_appearance(appearance)?;
    }
    if let Some(tint) = config.tint {
        core.set_time_of_day_tint(tint)?;
    }
    core.flush_storage().await
}
