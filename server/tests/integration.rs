//! Integration tests for the Eidolons server.
//!
//! These tests require a real Anthropic API key and are marked with `#[ignore]`.
//!
//! Run them with:
//! ```bash
//! ANTHROPIC_API_KEY=sk-... cargo test -p eidolons-server -- --ignored
//! ```

use eidolons_server::anthropic::{self, MessageContent, MessagesRequest, Role, StreamEvent};
use eidolons_server::proxy::AnthropicClient;

fn get_api_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY").ok()
}

// ============================================================================
// Smoke tests - require real API key
// ============================================================================

/// Smoke test: send a simple non-streaming request to Anthropic.
#[tokio::test]
#[ignore]
async fn smoke_test_anthropic_non_streaming() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    };

    let client = AnthropicClient::new(api_key);

    let request = MessagesRequest {
        model: "claude-3-5-haiku-20241022".to_string(),
        max_tokens: 50,
        messages: vec![anthropic::Message {
            role: Role::User,
            content: MessageContent::Text("Say 'hello' and nothing else.".to_string()),
        }],
        system: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: None,
        metadata: None,
    };

    let response = client.send(&request).await.expect("request should succeed");

    // Verify response structure
    assert!(!response.id.is_empty(), "response should have an ID");
    assert_eq!(response.role, Role::Assistant);
    assert!(!response.content.is_empty(), "response should have content");

    // Check that we got text content
    match &response.content[0] {
        anthropic::ResponseContentBlock::Text { text } => {
            assert!(!text.is_empty(), "response text should not be empty");
            println!("Response: {}", text);
        }
    }

    // Verify usage stats
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
}

/// Smoke test: send a streaming request to Anthropic.
#[tokio::test]
#[ignore]
async fn smoke_test_anthropic_streaming() {
    let Some(api_key) = get_api_key() else {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    };

    let client = AnthropicClient::new(api_key);

    let request = MessagesRequest {
        model: "claude-3-5-haiku-20241022".to_string(),
        max_tokens: 50,
        messages: vec![anthropic::Message {
            role: Role::User,
            content: MessageContent::Text("Count from 1 to 5.".to_string()),
        }],
        system: None,
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: Some(true),
        metadata: None,
    };

    let mut rx = client
        .send_stream(&request)
        .await
        .expect("stream request should succeed");

    let mut saw_message_start = false;
    let mut saw_content_delta = false;
    let mut saw_message_delta = false;
    let mut collected_text = String::new();

    while let Some(result) = rx.recv().await {
        let event = result.expect("stream event should parse");

        match event {
            StreamEvent::MessageStart { message } => {
                saw_message_start = true;
                assert!(!message.id.is_empty());
                println!("Message started: {}", message.id);
            }
            StreamEvent::ContentBlockDelta { delta, .. } => {
                saw_content_delta = true;
                match delta {
                    anthropic::ContentDelta::TextDelta { text } => {
                        collected_text.push_str(&text);
                        print!("{}", text);
                    }
                }
            }
            StreamEvent::MessageDelta { delta, usage } => {
                saw_message_delta = true;
                println!("\nStream complete. Stop reason: {:?}", delta.stop_reason);
                println!("Usage: {} input, {} output", usage.input_tokens, usage.output_tokens);
            }
            StreamEvent::Ping => {
                // Expected keepalive
            }
            StreamEvent::ContentBlockStart { .. }
            | StreamEvent::ContentBlockStop { .. }
            | StreamEvent::MessageStop => {
                // Expected lifecycle events
            }
            StreamEvent::Error { error } => {
                panic!("Unexpected error event: {:?}", error);
            }
        }
    }

    assert!(saw_message_start, "should have received message_start");
    assert!(saw_content_delta, "should have received content_block_delta");
    assert!(saw_message_delta, "should have received message_delta");
    assert!(!collected_text.is_empty(), "should have collected text");

    println!("\nFull response: {}", collected_text);
}

/// Smoke test: verify error handling for invalid API key.
#[tokio::test]
#[ignore]
async fn smoke_test_invalid_api_key() {
    let client = AnthropicClient::new("invalid-key".to_string());

    let request = MessagesRequest {
        model: "claude-3-5-haiku-20241022".to_string(),
        max_tokens: 10,
        messages: vec![anthropic::Message {
            role: Role::User,
            content: MessageContent::Text("Hi".to_string()),
        }],
        system: None,
        temperature: None,
        top_p: None,
        top_k: None,
        stop_sequences: None,
        stream: None,
        metadata: None,
    };

    let result = client.send(&request).await;

    assert!(result.is_err(), "should fail with invalid API key");

    let err = result.unwrap_err();
    let err_string = err.to_string();

    // Should be an upstream error (401)
    assert!(
        err_string.contains("401") || err_string.contains("authentication"),
        "error should indicate authentication failure: {}",
        err_string
    );
}
