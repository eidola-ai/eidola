use bincode::Options;
use crux_core::bridge::Request;
use eidolons_shared::{EffectFfi, Event, ViewModel};

pub fn bincode_options() -> impl bincode::Options + Copy {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
}

pub fn send_event(event: &Event) -> Vec<Request<EffectFfi>> {
    let event_bytes = bincode_options()
        .serialize(event)
        .expect("serialize event");
    let response_bytes = eidolons_shared::process_event(&event_bytes);
    bincode_options()
        .deserialize(&response_bytes)
        .expect("deserialize requests")
}

pub fn send_response(id: u32, data: &[u8]) -> Vec<Request<EffectFfi>> {
    let response_bytes = eidolons_shared::handle_response(id, data);
    bincode_options()
        .deserialize(&response_bytes)
        .expect("deserialize requests")
}

pub fn get_view() -> ViewModel {
    let view_bytes = eidolons_shared::view();
    bincode_options()
        .deserialize(&view_bytes)
        .expect("deserialize view model")
}
