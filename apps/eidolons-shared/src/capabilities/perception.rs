use crux_core::{Command, Request, capability::Operation, command::RequestBuilder};
use serde::{Deserialize, Serialize};

/// Role of a chat message sender
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub enum Role {
    /// Message from the user
    User,
    /// Message from the AI assistant
    Assistant,
}

/// A single message in the conversation
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    /// Who sent this message
    pub role: Role,
    /// The message content
    pub content: String,
}

/// Operation for the Perception capability
///
/// This represents a request to the perception service that the shell must fulfill.
/// The shell receives this operation, calls the PerceptionService,
/// and sends the response back via handle_response.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PerceptionRequest {
    /// Full conversation history for multi-turn chat
    pub messages: Vec<ChatMessage>,
}

/// Response from the Perception capability
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PerceptionResponse {
    /// The AI-generated response
    pub response: String,
}

impl Operation for PerceptionRequest {
    type Output = PerceptionResponse;
}

/// Request an AI response for the given conversation history
///
/// The shell will receive a PerceptionRequest effect, call the
/// PerceptionService's chat() method with the full conversation,
/// and send the result back as PerceptionResponse.
pub fn ask_with_history<Effect, Event>(
    messages: Vec<ChatMessage>,
) -> RequestBuilder<Effect, Event, impl core::future::Future<Output = PerceptionResponse>>
where
    Effect: Send + From<Request<PerceptionRequest>> + 'static,
    Event: Send + 'static,
{
    Command::request_from_shell(PerceptionRequest { messages })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perception_request_serialization() {
        let req = PerceptionRequest {
            messages: vec![
                ChatMessage {
                    role: Role::User,
                    content: "Hello, AI!".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: PerceptionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.messages.len(), 1);
        assert_eq!(deserialized.messages[0].content, "Hello, AI!");
        assert_eq!(deserialized.messages[0].role, Role::User);
    }

    #[test]
    fn test_perception_response_serialization() {
        let resp = PerceptionResponse {
            response: "Hello, human!".to_string(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: PerceptionResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.response, "Hello, human!");
    }

    #[test]
    fn test_chat_message_serialization() {
        let msg = ChatMessage {
            role: Role::Assistant,
            content: "I can help you.".to_string(),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: ChatMessage = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.role, Role::Assistant);
        assert_eq!(deserialized.content, "I can help you.");
    }
}
