//! Eidolons-specific response types with inline privacy and verification metadata.
//!
//! These types define the public API contract between the eidolons server and its clients.
//! They wrap standard chat completion data with privacy transparency and cryptographic
//! verification information.

use serde::Serialize;
use utoipa::ToSchema;

use crate::auth::AuthMethod;
use crate::types::{Choice, Usage};

/// An Eidolons chat completion response.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EidolonsResponse {
    /// Unique identifier for the completion.
    pub id: String,

    /// The object type.
    pub object: String,

    /// Unix timestamp of when the completion was created.
    pub created: u64,

    /// The model that produced the completion.
    pub model: String,

    /// Completion choices.
    pub choices: Vec<Choice>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// Privacy metadata describing data exposure and authorization.
    pub privacy: PrivacyMetadata,

    /// Verification metadata with cryptographic attestations.
    pub verification: VerificationMetadata,
}

impl EidolonsResponse {
    /// Build from an OpenAI-format completion response plus privacy/verification data.
    pub fn from_completion(
        response: crate::types::ChatCompletionResponse,
        privacy: PrivacyMetadata,
        verification: VerificationMetadata,
    ) -> Self {
        Self {
            id: response.id,
            object: "eidolons.chat.completion".to_string(),
            created: response.created,
            model: response.model,
            choices: response.choices,
            usage: response.usage,
            privacy,
            verification,
        }
    }
}

// ---------------------------------------------------------------------------
// Privacy metadata
// ---------------------------------------------------------------------------

/// Privacy metadata describing how the request was authorized, what data is
/// visible to which parties, and transport security properties.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct PrivacyMetadata {
    /// Authorization details.
    pub authorization: AuthorizationInfo,

    /// Which parties can see which data, and under what guarantees.
    pub data_exposure: Vec<DataExposure>,

    /// Transport security between hops.
    pub transport: TransportInfo,
}

/// Authorization information.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AuthorizationInfo {
    /// The authorization method used.
    pub method: AuthMethod,

    /// Whether this authorization method allows linking across requests.
    pub linkable: bool,
}

/// Describes a party's visibility into request/response data.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DataExposure {
    /// The party (e.g., "eidolons-proxy", "phala-gpu-tee", "anthropic").
    pub party: String,

    /// What data this party can see (e.g., ["prompt", "response", "token_count"]).
    pub sees: Vec<String>,

    /// How confidentiality is enforced: "tee" or "provider-policy".
    pub confidentiality: String,

    /// Whether this party logs requests: "none", "unknown", "yes".
    pub logged: String,
}

/// Transport security information.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct TransportInfo {
    /// Transport from client to this proxy.
    pub client_to_proxy: String,

    /// Transport from this proxy to the backend.
    pub proxy_to_backend: String,
}

// ---------------------------------------------------------------------------
// Verification metadata
// ---------------------------------------------------------------------------

/// Verification metadata with cryptographic attestations for the proxy and backend.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct VerificationMetadata {
    /// Proxy attestation (None until the proxy runs in a TEE).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<ProxyAttestation>,

    /// Backend attestation (only for TEE-backed models like phala/*).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendAttestation>,
}

/// Proxy attestation (placeholder for future TEE deployment).
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ProxyAttestation {
    /// Current attestation status.
    pub status: AttestationStatus,
}

/// Attestation status.
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AttestationStatus {
    /// Attestation is not yet available (proxy not in TEE).
    Unavailable,
    /// Attestation is present and verified.
    Verified,
}

/// Backend attestation from the inference provider.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct BackendAttestation {
    /// The provider that issued the attestation (e.g., "redpill").
    pub provider: String,

    /// The signing address (e.g., Ethereum address for Phala/RedPill).
    pub signing_address: String,

    /// The ECDSA signature over the signed content.
    pub signature: String,

    /// What was signed: "sha256(request_body):sha256(response_body)".
    pub signed_content: String,

    /// The signing algorithm (e.g., "ecdsa").
    pub signing_algorithm: String,
}

// ---------------------------------------------------------------------------
// Streaming metadata event
// ---------------------------------------------------------------------------

/// Metadata event sent as the final SSE event before [DONE] in streaming responses.
///
/// Clients identify this by checking `object == "eidolons.chat.completion.metadata"`.
/// Standard OpenAI clients will ignore it.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EidolonsStreamMetadata {
    /// The object type.
    pub object: String,

    /// The completion ID this metadata belongs to.
    pub id: String,

    /// Privacy metadata.
    pub privacy: PrivacyMetadata,

    /// Verification metadata.
    pub verification: VerificationMetadata,
}

impl EidolonsStreamMetadata {
    pub fn new(id: String, privacy: PrivacyMetadata, verification: VerificationMetadata) -> Self {
        Self {
            object: "eidolons.chat.completion.metadata".to_string(),
            id,
            privacy,
            verification,
        }
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

/// Build privacy metadata for a request.
pub fn build_privacy_metadata(
    auth: &crate::auth::AuthContext,
    tee_model: bool,
    backend_provider: &str,
) -> PrivacyMetadata {
    let mut exposure = vec![DataExposure {
        party: "eidolons-proxy".to_string(),
        sees: vec![
            "prompt".to_string(),
            "response".to_string(),
            "token_count".to_string(),
        ],
        confidentiality: "process".to_string(),
        logged: "none".to_string(),
    }];

    if tee_model {
        exposure.push(DataExposure {
            party: format!("{}-tee", backend_provider),
            sees: vec!["prompt".to_string(), "response".to_string()],
            confidentiality: "tee".to_string(),
            logged: "none".to_string(),
        });
    } else {
        exposure.push(DataExposure {
            party: backend_provider.to_string(),
            sees: vec![
                "prompt".to_string(),
                "response".to_string(),
                "metadata".to_string(),
            ],
            confidentiality: "provider-policy".to_string(),
            logged: "unknown".to_string(),
        });
    }

    PrivacyMetadata {
        authorization: AuthorizationInfo {
            method: auth.method.clone(),
            linkable: auth.method.linkable(),
        },
        data_exposure: exposure,
        transport: TransportInfo {
            client_to_proxy: "tls-1.3".to_string(),
            proxy_to_backend: "tls-1.3".to_string(),
        },
    }
}

/// Build verification metadata.
pub fn build_verification_metadata(
    backend_attestation: Option<BackendAttestation>,
) -> VerificationMetadata {
    VerificationMetadata {
        proxy: None, // Placeholder until proxy runs in TEE
        backend: backend_attestation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eidolons_response_serialization() {
        let response = EidolonsResponse {
            id: "eid-123".to_string(),
            object: "eidolons.chat.completion".to_string(),
            created: 1234567890,
            model: "phala/deepseek-v3".to_string(),
            choices: vec![],
            usage: None,
            privacy: PrivacyMetadata {
                authorization: AuthorizationInfo {
                    method: AuthMethod::AnonymousCredential,
                    linkable: false,
                },
                data_exposure: vec![],
                transport: TransportInfo {
                    client_to_proxy: "tls-1.3".to_string(),
                    proxy_to_backend: "tls-1.3".to_string(),
                },
            },
            verification: VerificationMetadata {
                proxy: None,
                backend: None,
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"object\":\"eidolons.chat.completion\""));
        assert!(json.contains("\"method\":\"anonymous_credential\""));
        assert!(json.contains("\"linkable\":false"));
    }

    #[test]
    fn test_stream_metadata_serialization() {
        let meta = EidolonsStreamMetadata::new(
            "eid-123".to_string(),
            PrivacyMetadata {
                authorization: AuthorizationInfo {
                    method: AuthMethod::None,
                    linkable: false,
                },
                data_exposure: vec![],
                transport: TransportInfo {
                    client_to_proxy: "tls-1.3".to_string(),
                    proxy_to_backend: "tls-1.3".to_string(),
                },
            },
            VerificationMetadata {
                proxy: None,
                backend: None,
            },
        );

        let json = serde_json::to_string(&meta).unwrap();
        assert!(json.contains("\"object\":\"eidolons.chat.completion.metadata\""));
    }

    #[test]
    fn test_build_privacy_metadata_tee() {
        let auth = crate::auth::AuthContext {
            method: AuthMethod::AnonymousCredential,
        };
        let privacy = build_privacy_metadata(&auth, true, "redpill");
        assert_eq!(privacy.data_exposure.len(), 2);
        assert_eq!(privacy.data_exposure[1].confidentiality, "tee");
    }

    #[test]
    fn test_build_privacy_metadata_non_tee() {
        let auth = crate::auth::AuthContext {
            method: AuthMethod::None,
        };
        let privacy = build_privacy_metadata(&auth, false, "anthropic");
        assert_eq!(privacy.data_exposure.len(), 2);
        assert_eq!(privacy.data_exposure[1].confidentiality, "provider-policy");
        assert_eq!(privacy.data_exposure[1].logged, "unknown");
    }
}
