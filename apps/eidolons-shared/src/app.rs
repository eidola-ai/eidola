use crux_core::{
    App, Command,
    macros::effect,
    render::{RenderOperation, render},
};
use serde::{Deserialize, Serialize};

use crate::capabilities::hello::{HelloRequest, HelloResponse, hello};

/// Events that can be sent from the shell to the core
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// Request a greeting for the given name
    Greet(String),
    /// Response from the hello capability with the greeting
    #[serde(skip)]
    GreetingReceived(HelloResponse),
}

/// The internal application model (private state)
#[derive(Default)]
pub struct Model {
    greeting: Option<String>,
}

/// The view model exposed to the shell (public view state)
#[derive(Serialize, Deserialize, Clone, Default, Debug, PartialEq, Eq)]
pub struct ViewModel {
    pub greeting: String,
}

/// Side effects the core can request from the shell
#[effect(typegen)]
pub enum Effect {
    /// Request a render of the current view
    Render(RenderOperation),
    /// Request the hello capability
    Hello(HelloRequest),
}

/// The main Crux application
#[derive(Default)]
pub struct EidolonsApp;

impl App for EidolonsApp {
    type Event = Event;
    type Model = Model;
    type ViewModel = ViewModel;
    type Capabilities = ();
    type Effect = Effect;

    fn update(
        &self,
        event: Self::Event,
        model: &mut Self::Model,
        _caps: &Self::Capabilities,
    ) -> Command<Self::Effect, Self::Event> {
        match event {
            Event::Greet(name) => {
                // Request the greeting from the eidolons capability
                hello(name).then_send(Event::GreetingReceived)
            }
            Event::GreetingReceived(response) => {
                // Store the greeting and trigger a render
                model.greeting = Some(response.greeting);
                render()
            }
        }
    }

    fn view(&self, model: &Self::Model) -> Self::ViewModel {
        ViewModel {
            greeting: model.greeting.clone().unwrap_or_default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crux_core::testing::AppTester;

    #[test]
    fn test_greet_emits_hello_effect() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();

        let cmd = app.update(Event::Greet("World".to_string()), &mut model);

        let effect = cmd.expect_one_effect();
        match effect {
            Effect::Hello(req) => {
                assert_eq!(req.operation.name, "World");
            }
            _ => panic!("Expected Hello effect"),
        }
    }

    #[test]
    fn test_greeting_received_updates_model() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();

        let response = HelloResponse {
            greeting: "Hello, World!".to_string(),
        };
        let cmd = app.update(Event::GreetingReceived(response), &mut model);

        assert_eq!(model.greeting, Some("Hello, World!".to_string()));

        let effect = cmd.expect_one_effect();
        assert!(matches!(effect, Effect::Render(_)));
    }

    #[test]
    fn test_view_returns_greeting() {
        let app = EidolonsApp;
        let model = Model {
            greeting: Some("Hello, Test!".to_string()),
        };

        let view = app.view(&model);
        assert_eq!(view.greeting, "Hello, Test!");
    }

    #[test]
    fn test_view_returns_empty_when_no_greeting() {
        let app = EidolonsApp;
        let model = Model::default();

        let view = app.view(&model);
        assert_eq!(view.greeting, "");
    }
}
