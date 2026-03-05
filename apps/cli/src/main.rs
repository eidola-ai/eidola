use bincode::Options;
use crux_core::bridge::Request;
use eidolons_shared::{EffectFfi, Event, ViewModel};

fn bincode_options() -> impl bincode::Options + Copy {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
}

fn send_event(event: &Event) -> Vec<Request<EffectFfi>> {
    let event_bytes = bincode_options().serialize(event).expect("serialize event");
    let response_bytes = eidolons_shared::process_event(&event_bytes);
    bincode_options()
        .deserialize(&response_bytes)
        .expect("deserialize requests")
}

fn get_view() -> ViewModel {
    let view_bytes = eidolons_shared::view();
    bincode_options()
        .deserialize(&view_bytes)
        .expect("deserialize view model")
}

fn handle_effects(requests: Vec<Request<EffectFfi>>) {
    for request in requests {
        match request.effect {
            EffectFfi::Render(_) => {
                // Render: read and display the current view model
            }
        }
    }
}

fn main() {
    let requests = send_event(&Event::Greet);
    handle_effects(requests);

    let vm = get_view();
    println!("{}", vm.greeting);
}
