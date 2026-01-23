use crux_core::{
    App, Command,
    macros::effect,
    render::{RenderOperation, render},
};
use serde::{Deserialize, Serialize};

use crate::capabilities::hello::{HelloRequest, HelloResponse, hello};
use crate::capabilities::perception::{
    ChatMessage, PerceptionRequest, PerceptionResponse, PerceptionStreamingRequest,
    PerceptionStreamingResponse, Role, ask_with_history, ask_with_history_streaming,
};

/// Events that can be sent from the shell to the core
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// Request a greeting for the given name
    Greet(String),
    /// Response from the hello capability with the greeting
    #[serde(skip)]
    GreetingReceived(HelloResponse),
    /// Submit a chat message to the AI (non-streaming)
    SubmitMessage(String),
    /// Submit a chat message to the AI with streaming response
    SubmitMessageStreaming(String),
    /// Response from the perception capability (non-streaming)
    #[serde(skip)]
    PerceptionResponse(PerceptionResponse),
    /// A chunk of text received during streaming
    PerceptionChunk(String),
    /// Streaming generation completed successfully
    PerceptionStreamComplete,
    /// An error occurred during streaming generation
    PerceptionStreamError(String),
}

/// The internal application model (private state)
#[derive(Default)]
pub struct Model {
    /// The greeting from the hello capability
    greeting: Option<String>,
    /// The conversation history
    pub conversation: Vec<ChatMessage>,
    /// Whether we're waiting for an AI response
    pub is_processing: bool,
    /// The current streaming response being built up
    pub streaming_response: String,
}

/// The view model exposed to the shell (public view state)
#[derive(Serialize, Deserialize, Clone, Default, Debug, PartialEq, Eq)]
pub struct ViewModel {
    /// The greeting message
    pub greeting: String,
    /// The conversation history
    pub conversation: Vec<ChatMessage>,
    /// Whether we're waiting for an AI response
    pub is_processing: bool,
    /// The current streaming text being generated
    pub streaming_text: String,
}

/// Side effects the core can request from the shell
#[effect(typegen)]
pub enum Effect {
    /// Request a render of the current view
    Render(RenderOperation),
    /// Request the hello capability
    Hello(HelloRequest),
    /// Request the perception capability (non-streaming)
    Perception(PerceptionRequest),
    /// Request the perception capability with streaming
    PerceptionStreaming(PerceptionStreamingRequest),
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
            Event::SubmitMessage(message) => {
                // Add user message to conversation
                model.conversation.push(ChatMessage {
                    role: Role::User,
                    content: message,
                });
                model.is_processing = true;

                // Pass full conversation history to perception for context-aware responses
                let messages = model.conversation.clone();
                Command::all([
                    render(),
                    ask_with_history(messages).then_send(Event::PerceptionResponse),
                ])
            }
            Event::PerceptionResponse(response) => {
                // Add assistant message to conversation
                model.conversation.push(ChatMessage {
                    role: Role::Assistant,
                    content: response.response,
                });
                model.is_processing = false;

                render()
            }
            Event::SubmitMessageStreaming(message) => {
                // Add user message to conversation
                model.conversation.push(ChatMessage {
                    role: Role::User,
                    content: message,
                });
                model.is_processing = true;
                model.streaming_response.clear();

                // Pass full conversation history to perception for streaming response
                let messages = model.conversation.clone();

                // Create a command that streams responses and sends events for each
                let stream_cmd = Command::new(|ctx| async move {
                    use futures::StreamExt;
                    let mut stream = ask_with_history_streaming(messages).into_stream(ctx.clone());
                    while let Some(response) = stream.next().await {
                        match response {
                            PerceptionStreamingResponse::Chunk(text) => {
                                ctx.send_event(Event::PerceptionChunk(text));
                            }
                            PerceptionStreamingResponse::Done => {
                                ctx.send_event(Event::PerceptionStreamComplete);
                            }
                            PerceptionStreamingResponse::Error(e) => {
                                ctx.send_event(Event::PerceptionStreamError(e));
                            }
                        }
                    }
                });

                Command::all([render(), stream_cmd])
            }
            Event::PerceptionChunk(text) => {
                // Append the chunk to the streaming response
                model.streaming_response.push_str(&text);
                render()
            }
            Event::PerceptionStreamComplete => {
                // Move the completed streaming response to conversation
                let response = std::mem::take(&mut model.streaming_response);
                model.conversation.push(ChatMessage {
                    role: Role::Assistant,
                    content: response,
                });
                model.is_processing = false;
                render()
            }
            Event::PerceptionStreamError(error) => {
                // Add error as assistant message
                model.conversation.push(ChatMessage {
                    role: Role::Assistant,
                    content: format!("Error: {}", error),
                });
                model.streaming_response.clear();
                model.is_processing = false;
                render()
            }
        }
    }

    fn view(&self, model: &Self::Model) -> Self::ViewModel {
        ViewModel {
            greeting: model.greeting.clone().unwrap_or_default(),
            conversation: model.conversation.clone(),
            is_processing: model.is_processing,
            streaming_text: model.streaming_response.clone(),
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
            ..Default::default()
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

    #[test]
    fn test_submit_message_adds_user_message_and_requests_perception() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();

        let cmd = app.update(Event::SubmitMessage("Hello AI".to_string()), &mut model);

        // User message should be added
        assert_eq!(model.conversation.len(), 1);
        assert_eq!(model.conversation[0].role, Role::User);
        assert_eq!(model.conversation[0].content, "Hello AI");

        // Should be processing
        assert!(model.is_processing);

        // Should have Render and Perception effects
        let effects: Vec<_> = cmd.into_effects().collect();
        assert_eq!(effects.len(), 2);

        let has_render = effects.iter().any(|e| matches!(e, Effect::Render(_)));
        let has_perception = effects.iter().any(|e| {
            matches!(e, Effect::Perception(req) if
                req.operation.messages.len() == 1 &&
                req.operation.messages[0].content == "Hello AI" &&
                req.operation.messages[0].role == Role::User
            )
        });

        assert!(has_render, "Should have Render effect");
        assert!(
            has_perception,
            "Should have Perception effect with full conversation"
        );
    }

    #[test]
    fn test_perception_response_adds_assistant_message() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model {
            conversation: vec![ChatMessage {
                role: Role::User,
                content: "Hello AI".to_string(),
            }],
            is_processing: true,
            ..Default::default()
        };

        let response = PerceptionResponse {
            response: "Hello human!".to_string(),
        };
        let cmd = app.update(Event::PerceptionResponse(response), &mut model);

        // Assistant message should be added
        assert_eq!(model.conversation.len(), 2);
        assert_eq!(model.conversation[1].role, Role::Assistant);
        assert_eq!(model.conversation[1].content, "Hello human!");

        // Should no longer be processing
        assert!(!model.is_processing);

        // Should trigger render
        let effect = cmd.expect_one_effect();
        assert!(matches!(effect, Effect::Render(_)));
    }

    #[test]
    fn test_view_includes_conversation() {
        let app = EidolonsApp;
        let model = Model {
            conversation: vec![
                ChatMessage {
                    role: Role::User,
                    content: "Hi".to_string(),
                },
                ChatMessage {
                    role: Role::Assistant,
                    content: "Hello!".to_string(),
                },
            ],
            is_processing: false,
            ..Default::default()
        };

        let view = app.view(&model);
        assert_eq!(view.conversation.len(), 2);
        assert_eq!(view.conversation[0].role, Role::User);
        assert_eq!(view.conversation[1].role, Role::Assistant);
        assert!(!view.is_processing);
    }
}
