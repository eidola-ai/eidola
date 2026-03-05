use bincode::Options;
use crux_core::bridge::Request;
use eidolons_shared::{
    EffectFfi, Event, PerceptionResponse, ViewModel,
    capabilities::perception::{PerceptionStreamingResponse, Role},
};
use iocraft::prelude::*;

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

fn send_response(id: u32, data: &[u8]) -> Vec<Request<EffectFfi>> {
    let response_bytes = eidolons_shared::handle_response(id, data);
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
        let id = request.id.0;
        match request.effect {
            EffectFfi::Render(_) => {
                // Render is handled by the component reading the view model
            }
            EffectFfi::Perception(_) => {
                let response = PerceptionResponse {
                    response: "On-device inference is not yet available.".to_string(),
                };
                let response_bytes = bincode_options()
                    .serialize(&response)
                    .expect("serialize perception response");
                let follow_up = send_response(id, &response_bytes);
                handle_effects(follow_up);
            }
            EffectFfi::PerceptionStreaming(_) => {
                let chunk = PerceptionStreamingResponse::Chunk(
                    "On-device inference is not yet available.".to_string(),
                );
                let chunk_bytes = bincode_options()
                    .serialize(&chunk)
                    .expect("serialize streaming chunk");
                let follow_up = send_response(id, &chunk_bytes);
                handle_effects(follow_up);

                let done = PerceptionStreamingResponse::Done;
                let done_bytes = bincode_options()
                    .serialize(&done)
                    .expect("serialize streaming done");
                let follow_up = send_response(id, &done_bytes);
                handle_effects(follow_up);
            }
        }
    }
}

#[component]
fn App(mut hooks: Hooks) -> impl Into<AnyElement<'static>> {
    let mut view_model = hooks.use_state(get_view);
    let mut input = hooks.use_state(String::new);
    let mut should_exit = hooks.use_state(|| false);

    if should_exit.get() {
        hooks.use_context_mut::<SystemContext>().exit();
    }

    hooks.use_terminal_events(move |event| match event {
        TerminalEvent::Key(KeyEvent { code, kind, .. }) if kind != KeyEventKind::Release => {
            match code {
                KeyCode::Enter => {
                    let message = input.to_string();
                    if !message.is_empty() {
                        input.set(String::new());
                        let requests = send_event(&Event::SubmitMessage(message));
                        handle_effects(requests);
                        view_model.set(get_view());
                    }
                }
                KeyCode::Esc => {
                    should_exit.set(true);
                }
                _ => {}
            }
        }
        _ => {}
    });

    let vm = view_model.read();

    element! {
        View(
            flex_direction: FlexDirection::Column,
            width: 80pct,
            max_width: 100,
        ) {
            View(
                border_style: BorderStyle::Round,
                border_color: Color::Blue,
                flex_direction: FlexDirection::Column,
                padding: 1,
                min_height: 3,
            ) {
                #(vm.conversation.iter().enumerate().map(|(i, msg)| {
                    let (label, color) = match msg.role {
                        Role::User => ("You", Color::Green),
                        Role::Assistant => ("AI", Color::Cyan),
                    };
                    let content = format!("{}: {}", label, msg.content);
                    element! {
                        View(key: i) {
                            Text(content: content, color: color)
                        }
                    }
                }))
                #(if vm.is_processing && !vm.streaming_text.is_empty() {
                    Some(element! {
                        Text(
                            content: format!("AI: {}", vm.streaming_text),
                            color: Color::DarkGrey,
                        )
                    })
                } else if vm.is_processing {
                    Some(element! {
                        Text(content: "Thinking...", color: Color::DarkGrey)
                    })
                } else if vm.conversation.is_empty() {
                    Some(element! {
                        Text(content: "Send a message to begin.", color: Color::DarkGrey)
                    })
                } else {
                    None
                })
            }
            View(
                border_style: BorderStyle::Round,
                border_color: Color::White,
                padding_left: 1,
                padding_right: 1,
            ) {
                View(width: 4) {
                    Text(content: "> ", color: Color::Green)
                }
                TextInput(
                    has_focus: !vm.is_processing,
                    value: input.to_string(),
                    on_change: move |new_value| input.set(new_value),
                )
            }
            Text(content: "Press Enter to send, Esc to quit", color: Color::DarkGrey)
        }
    }
}

fn main() {
    smol::block_on(element!(App).render_loop()).unwrap();
}
