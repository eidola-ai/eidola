//! Local models in a build without the `desktop` feature: there are no
//! engines. Nothing is ever registered (outside `test-support`), so a lease finds nothing, a load is
//! refused with [`AppError::LocalModel`] before anything is touched, the
//! snapshot is empty, and retiring a backend has nothing to stop.

use std::path::Path;

use super::{LocalModelInfo, LocalModelsState, Retirement};
use crate::Inner;
use crate::error::AppError;

/// What a turn holds while it uses an engine: nothing, since none is leased.
pub(crate) type EngineLease = ();

/// The engine registry, empty but for what a test registers.
#[derive(Default)]
pub(crate) struct LocalRuntime {
    /// Engines a test marked loaded (`AppCore::test_register_loaded_local_model`),
    /// by `(backend_id, slug)`: the shared engine route's own tests run in
    /// this build too, against a mock upstream.
    #[cfg(feature = "test-support")]
    test_engines: std::sync::Mutex<std::collections::HashMap<(String, String), u16>>,
}

impl LocalRuntime {
    pub(crate) fn lease_engine(
        &self,
        _backend_id: &str,
        _slug: &str,
    ) -> Option<(String, u32, EngineLease)> {
        #[cfg(feature = "test-support")]
        if let Some(port) = self
            .test_engines
            .lock()
            .expect("test engines lock")
            .get(&(_backend_id.to_string(), _slug.to_string()))
        {
            return Some((
                format!("http://127.0.0.1:{port}"),
                super::LOCAL_CONTEXT_TOKENS,
                (),
            ));
        }
        None
    }

    #[cfg(feature = "test-support")]
    pub(crate) fn register_for_test(&self, backend_id: &str, slug: &str, port: u16) {
        self.test_engines
            .lock()
            .expect("test engines lock")
            .insert((backend_id.to_string(), slug.to_string()), port);
    }
}

impl Inner {
    pub(crate) async fn load_local_model(&self, _id: &str) -> Result<(), AppError> {
        Err(AppError::LocalModel {
            message: "local models are not available in this build".to_string(),
        })
    }

    pub(crate) async fn local_models_state(&self) -> Result<LocalModelsState, AppError> {
        Ok(LocalModelsState {
            engine_path: None,
            models: Vec::new(),
            external: Vec::new(),
        })
    }

    pub(crate) async fn scan_engine_dir(
        &self,
        _backend_id: &str,
        _dir: &Path,
    ) -> Vec<LocalModelInfo> {
        Vec::new()
    }

    pub(crate) fn retire_backend_engines(&self, _backend_id: &str) -> Retirement {
        Retirement::default()
    }
}
