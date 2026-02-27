//! RedPill attestation signature fetching.
//!
//! For `phala/*` models running in TEEs, RedPill provides per-response ECDSA
//! signatures that bind the request and response together. This module fetches
//! those signatures and converts them to our `BackendAttestation` type.

use serde::Deserialize;

use crate::error::ServerError;
use crate::response::BackendAttestation;

/// Client for fetching attestation signatures from RedPill.
pub struct AttestationClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

/// RedPill signature response format.
#[derive(Debug, Deserialize)]
struct RedPillSignatureResponse {
    /// The signed content (typically "sha256(request):sha256(response)").
    text: String,

    /// The ECDSA signature.
    signature: String,

    /// The Ethereum signing address.
    signing_address: String,
}

impl AttestationClient {
    pub fn new(api_key: String, base_url: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://api.redpill.ai/v1".to_string()),
        }
    }

    /// Fetch the per-response attestation signature for a chat completion.
    ///
    /// Returns `None` if the fetch fails (attestation is best-effort).
    pub async fn fetch_signature(&self, chat_id: &str, model: &str) -> Option<BackendAttestation> {
        match self.fetch_signature_inner(chat_id, model).await {
            Ok(attestation) => Some(attestation),
            Err(e) => {
                tracing::warn!("Failed to fetch attestation for chat_id={}: {}", chat_id, e);
                None
            }
        }
    }

    async fn fetch_signature_inner(
        &self,
        chat_id: &str,
        model: &str,
    ) -> Result<BackendAttestation, ServerError> {
        let url = format!("{}/signature/{}?model={}", self.base_url, chat_id, model);

        let response = self
            .client
            .get(&url)
            .header("authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .map_err(|e| ServerError::Network(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            return Err(ServerError::Backend {
                status: status.as_u16(),
                error_type: "attestation_error".to_string(),
                message: body,
            });
        }

        let sig_response: RedPillSignatureResponse = response
            .json()
            .await
            .map_err(|e| ServerError::Parse(e.to_string()))?;

        Ok(BackendAttestation {
            provider: "redpill".to_string(),
            signing_address: sig_response.signing_address,
            signature: sig_response.signature,
            signed_content: sig_response.text,
            signing_algorithm: "ecdsa".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_redpill_signature_response() {
        let json = r#"{
            "text": "sha256(req):sha256(resp)",
            "signature": "0xabc123",
            "signing_address": "0xdef456"
        }"#;

        let response: RedPillSignatureResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.text, "sha256(req):sha256(resp)");
        assert_eq!(response.signature, "0xabc123");
        assert_eq!(response.signing_address, "0xdef456");
    }
}
