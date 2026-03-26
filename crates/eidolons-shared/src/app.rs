use crux_core::{
    App, Command,
    macros::effect,
    render::{RenderOperation, render},
};
use serde::{Deserialize, Serialize};

/// Events that can be sent from the shell to the core
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// Trigger a greeting
    Greet,
}

/// The internal application model (private state)
#[derive(Default)]
pub struct Model {
    pub greeting: String,
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
}

/// The main Crux application
#[derive(Default)]
pub struct EidolonsApp;

impl App for EidolonsApp {
    type Event = Event;
    type Model = Model;
    type ViewModel = ViewModel;
    type Effect = Effect;

    fn update(
        &self,
        event: Self::Event,
        model: &mut Self::Model,
    ) -> Command<Self::Effect, Self::Event> {
        match event {
            Event::Greet => {
                model.greeting = "Hello, World!".to_string();
                render()
            }
        }
    }

    fn view(&self, model: &Self::Model) -> Self::ViewModel {
        ViewModel {
            greeting: model.greeting.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet_sets_greeting() {
        let app = EidolonsApp;
        let mut model = Model::default();

        let mut cmd = app.update(Event::Greet, &mut model);

        assert_eq!(model.greeting, "Hello, World!");

        let effect = cmd.effects().next().expect("expected one effect");
        assert!(matches!(effect, Effect::Render(_)));
    }

    #[test]
    fn test_view_reflects_model() {
        let app = EidolonsApp;
        let model = Model {
            greeting: "Hello, World!".to_string(),
        };

        let view = app.view(&model);
        assert_eq!(view.greeting, "Hello, World!");
    }

    #[test]
    fn test_initial_view_is_empty() {
        let app = EidolonsApp;
        let model = Model::default();

        let view = app.view(&model);
        assert_eq!(view.greeting, "");
    }
}
