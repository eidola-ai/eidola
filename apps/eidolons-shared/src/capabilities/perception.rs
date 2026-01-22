use crux_core::{Command, Request, capability::Operation, command::RequestBuilder};
use serde::{Deserialize, Serialize};

/// Operation for the Perception capability
///
/// This represents a request to the perception service that the shell must fulfill.
/// The shell receives this operation, calls the PerceptionService,
/// and sends the response back via handle_response.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct PerceptionRequest {
    /// The prompt/message to send to the AI
    pub prompt: String,
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

/// Request an AI response for the given prompt
///
/// The shell will receive a PerceptionRequest effect, call the
/// PerceptionService's chat() method, and send the result back as PerceptionResponse.
pub fn ask<Effect, Event>(
    prompt: impl Into<String>,
) -> RequestBuilder<Effect, Event, impl core::future::Future<Output = PerceptionResponse>>
where
    Effect: Send + From<Request<PerceptionRequest>> + 'static,
    Event: Send + 'static,
{
    Command::request_from_shell(PerceptionRequest {
        prompt: prompt.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perception_request_serialization() {
        let req = PerceptionRequest {
            prompt: "Hello, AI!".to_string(),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: PerceptionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.prompt, "Hello, AI!");
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
}
