pub mod app;

use std::sync::LazyLock;

pub use app::{Effect, EffectFfi, EidolaApp, Event, Model, ViewModel};
pub use crux_core::{
    Core,
    bridge::{Bridge, EffectId},
};

uniffi::setup_scaffolding!();

/// Static bridge instance that holds the core application
static CORE: LazyLock<Bridge<EidolaApp>> = LazyLock::new(|| Bridge::new(Core::new()));

/// Process an event from the shell
///
/// Takes a bincode-serialized Event and returns bincode-serialized effects (requests).
/// The shell should deserialize the response to get the list of effects to handle.
#[uniffi::export]
pub fn process_event(data: &[u8]) -> Vec<u8> {
    let mut requests = Vec::new();
    CORE.update(data, &mut requests)
        .unwrap_or_else(|e| panic!("process_event failed: {e}"));
    requests
}

/// Handle a response from the shell
///
/// Takes a request ID and bincode-serialized response data.
/// Returns bincode-serialized effects (requests) for any follow-up operations.
#[uniffi::export]
pub fn handle_response(id: u32, data: &[u8]) -> Vec<u8> {
    let mut requests = Vec::new();
    CORE.resolve(EffectId(id), data, &mut requests)
        .unwrap_or_else(|e| panic!("handle_response failed: {e}"));
    requests
}

/// Get the current view model
///
/// Returns the bincode-serialized ViewModel representing the current UI state.
#[uniffi::export]
pub fn view() -> Vec<u8> {
    let mut view = Vec::new();
    CORE.view(&mut view)
        .unwrap_or_else(|e| panic!("view failed: {e}"));
    view
}
