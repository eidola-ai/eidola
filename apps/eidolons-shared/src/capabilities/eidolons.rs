use crux_core::{Command, Request, capability::Operation, command::RequestBuilder};
use serde::{Deserialize, Serialize};

/// Operation for the Eidolons capability
///
/// This represents a request to the eidolons library that the shell must fulfill.
/// The shell receives this operation, calls the actual eidolons library,
/// and sends the response back via handle_response.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct EidolonsRequest {
    /// The name to greet
    pub name: String,
}

/// Response from the Eidolons capability
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct EidolonsResponse {
    /// The greeting string
    pub greeting: String,
}

impl Operation for EidolonsRequest {
    type Output = EidolonsResponse;
}

/// Request a greeting for the given name
///
/// The shell will receive an EidolonsRequest effect, call the eidolons
/// library's hello() function, and send the result back as EidolonsResponse.
pub fn hello<Effect, Event>(
    name: impl Into<String>,
) -> RequestBuilder<Effect, Event, impl core::future::Future<Output = EidolonsResponse>>
where
    Effect: Send + From<Request<EidolonsRequest>> + 'static,
    Event: Send + 'static,
{
    Command::request_from_shell(EidolonsRequest { name: name.into() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eidolons_request_serialization() {
        let req = EidolonsRequest {
            name: "World".to_string(),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: EidolonsRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "World");
    }

    #[test]
    fn test_eidolons_response_serialization() {
        let resp = EidolonsResponse {
            greeting: "Hello, World!".to_string(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: EidolonsResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.greeting, "Hello, World!");
    }
}
