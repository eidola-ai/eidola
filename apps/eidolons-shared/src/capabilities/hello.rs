use crux_core::{Command, Request, capability::Operation, command::RequestBuilder};
use serde::{Deserialize, Serialize};

/// Operation for the Hello capability
///
/// This represents a request to the hello library that the shell must fulfill.
/// The shell receives this operation, calls the actual hello function,
/// and sends the response back via handle_response.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct HelloRequest {
    /// The name to greet
    pub name: String,
}

/// Response from the Hello capability
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct HelloResponse {
    /// The greeting string
    pub greeting: String,
}

impl Operation for HelloRequest {
    type Output = HelloResponse;
}

/// Request a greeting for the given name
///
/// The shell will receive a HelloRequest effect, call the hello
/// library's hello() function, and send the result back as HelloResponse.
pub fn hello<Effect, Event>(
    name: impl Into<String>,
) -> RequestBuilder<Effect, Event, impl core::future::Future<Output = HelloResponse>>
where
    Effect: Send + From<Request<HelloRequest>> + 'static,
    Event: Send + 'static,
{
    Command::request_from_shell(HelloRequest { name: name.into() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hello_request_serialization() {
        let req = HelloRequest {
            name: "World".to_string(),
        };

        let json = serde_json::to_string(&req).unwrap();
        let deserialized: HelloRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, "World");
    }

    #[test]
    fn test_hello_response_serialization() {
        let resp = HelloResponse {
            greeting: "Hello, World!".to_string(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: HelloResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.greeting, "Hello, World!");
    }
}
