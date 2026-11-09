//! Authorization for the Eidola server.
//!
//! A metered request pays with ACT (Anonymous Credit Token) spend proofs,
//! following draft-schlesinger-privacypass-act-01, presented in the request
//! body's `credentials` member (`eidola_common::spend_credentials`); account
//! endpoints take HTTP Basic auth.

use anonymous_credit_tokens::SpendProof;
use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use base64::Engine;
use eidola_common::refund_key::REFUND_KEY_BYTES;
use eidola_common::spend_credentials::{
    ACT_L, CredentialWire, CredentialsError, DecodedCredential, decode_credentials,
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::AppState;
use crate::credentials::ACT_TOKEN_TYPE;
use crate::error::ServerError;

/// Result of successful token validation.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// The authorization method that was used.
    pub method: AuthMethod,
}

/// Supported authorization methods.
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// Anonymous credential for privacy-preserving usage.
    AnonymousCredential,
    /// No authentication (development only).
    None,
}

impl AuthMethod {
    /// Whether requests using this method are linkable across requests.
    pub fn linkable(&self) -> bool {
        match self {
            AuthMethod::AnonymousCredential => false,
            AuthMethod::None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// ACT spends presented in the request body
// ---------------------------------------------------------------------------

/// One presented ACT Token, from an element of the request body's
/// `credentials` member (`eidola_common::spend_credentials`).
///
/// Token structure (draft-schlesinger-privacypass-act-01, Section 9.1):
/// ```text
/// struct {
///     uint16_t token_type = 0xE5AD;
///     uint8_t challenge_digest[32];
///     uint8_t issuer_key_id[32];   // SHA-256(pkI_serialized)
///     uint8_t encoded_spend_proof[...];
/// } Token;
/// ```
#[derive(Clone)]
pub struct ActSpend {
    /// SHA-256(TokenChallenge) — binds the token to the challenge context.
    pub challenge_digest: [u8; 32],
    /// SHA-256(pkI_serialized) — identifies which issuer key signed the credential.
    pub issuer_key_hash: [u8; 32],
    /// The zero-knowledge spend proof.
    pub spend_proof: SpendProof<ACT_L>,
    /// The client's refund recovery key for this spend, from the same
    /// `credentials` element: what the stored refund is sealed under.
    pub refund_key: RefundKey,
    /// When this gateway recorded the spend's nullifier, once it has: what
    /// decides whether its final refund may still be stored
    /// (`credentials::final_refund_still_storable`).
    pub recorded_at: Option<std::time::SystemTime>,
}

/// A client's refund recovery key (`eidola_common::refund_key`), held only in
/// enclave memory for the request that carried it.
///
/// The stored copy of a spend's refund is sealed under a key that needs this
/// value, so neither the database nor anyone holding the gateway's own
/// secrets can open it without the client. It is never stored, logged or
/// traced: `Debug` is redacted, and the bytes are zeroed on drop. (The
/// request body's bytes, which the HTTP stack owns, are not scrubbed.)
#[derive(Clone)]
pub struct RefundKey(zeroize::Zeroizing<[u8; REFUND_KEY_BYTES]>);

impl RefundKey {
    pub fn from_bytes(bytes: [u8; REFUND_KEY_BYTES]) -> Self {
        Self(zeroize::Zeroizing::new(bytes))
    }

    pub(crate) fn as_bytes(&self) -> &[u8; REFUND_KEY_BYTES] {
        &self.0
    }
}

impl std::fmt::Debug for RefundKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RefundKey(<redacted>)")
    }
}

/// The request headers a credential could travel in outside the body: the
/// RFC 9577 `Authorization` header, and a refund-key header. A spend endpoint
/// takes no other authorization, so any `Authorization` header is refused.
const CREDENTIAL_HEADERS: [&str; 2] = ["authorization", "x-eidola-refund-key"];

/// Refuse a spend request that carries a credential in a header. A
/// credential travels in the body's `credentials` member and nowhere else,
/// so there is one transport and nothing to reconcile between two. Checked
/// before the body is read.
pub fn refuse_header_credentials(headers: &axum::http::HeaderMap) -> Result<(), ServerError> {
    if CREDENTIAL_HEADERS
        .iter()
        .any(|name| headers.contains_key(*name))
    {
        return Err(ServerError::BadRequest {
            message: "a credential travels in the request body's `credentials` member, \
                      never in a header"
                .to_string(),
        });
    }
    Ok(())
}

/// The spends a request body's `credentials` member presents, in input order,
/// each parsed as a Token: the count first, then each value's spelling, then
/// each Token's layout. Nothing here verifies a proof or touches the
/// database; a refusal is answered before anything is recorded.
pub fn presented_spends(wires: &[CredentialWire]) -> Result<Vec<ActSpend>, ServerError> {
    let decoded = decode_credentials(wires).map_err(|e| match e {
        CredentialsError::RefundKeyEncoding | CredentialsError::TooMany => {
            ServerError::BadRequest {
                message: e.to_string(),
            }
        }
        CredentialsError::Missing | CredentialsError::TokenEncoding => ServerError::Unauthorized {
            message: e.to_string(),
        },
    })?;
    decoded.iter().map(parse_token).collect()
}

/// Exactly one spend: the shape every endpoint takes today
/// (`MAX_CREDENTIALS_PER_REQUEST`).
pub fn presented_spend(wires: &[CredentialWire]) -> Result<ActSpend, ServerError> {
    let mut spends = presented_spends(wires)?;
    match (spends.pop(), spends.is_empty()) {
        (Some(spend), true) => Ok(spend),
        _ => Err(ServerError::BadRequest {
            message: CredentialsError::TooMany.to_string(),
        }),
    }
}

/// One Token's layout: `token_type`, `challenge_digest`, `issuer_key_id`, and
/// a CBOR spend proof with no trailing bytes.
fn parse_token(credential: &DecodedCredential) -> Result<ActSpend, ServerError> {
    let token_bytes = &credential.token;
    // Minimum size: 2 (token_type) + 32 (challenge_digest) + 32 (issuer_key_id) + 1 (spend proof)
    if token_bytes.len() < 67 {
        return Err(ServerError::Unauthorized {
            message: "token too short".to_string(),
        });
    }

    let token_type = u16::from_be_bytes([token_bytes[0], token_bytes[1]]);
    if token_type != ACT_TOKEN_TYPE {
        // The presented token type is caller-derived and this message
        // reaches the logs via `Unauthorized`'s `Display`, so only the
        // expected constant is interpolated.
        return Err(ServerError::Unauthorized {
            message: format!("unsupported token type, expected 0x{ACT_TOKEN_TYPE:04X}"),
        });
    }

    let mut challenge_digest = [0u8; 32];
    challenge_digest.copy_from_slice(&token_bytes[2..34]);
    let mut issuer_key_hash = [0u8; 32];
    issuer_key_hash.copy_from_slice(&token_bytes[34..66]);

    // The spend proof is the rest of the Token, as one CBOR item and nothing
    // after it: the proof re-encodes to exactly the bytes presented.
    let proof_bytes = &token_bytes[66..];
    let spend_proof = SpendProof::<ACT_L>::from_cbor(proof_bytes)
        .ok()
        .filter(|proof| proof.to_cbor().is_ok_and(|cbor| cbor == proof_bytes))
        .ok_or_else(|| ServerError::Unauthorized {
            message: "invalid CBOR spend proof in token".to_string(),
        })?;

    Ok(ActSpend {
        challenge_digest,
        issuer_key_hash,
        spend_proof,
        refund_key: RefundKey::from_bytes(credential.refund_key),
        recorded_at: None,
    })
}

// ---------------------------------------------------------------------------
// BasicAuth extractor (for account endpoints)
// ---------------------------------------------------------------------------

/// Axum extractor that validates HTTP Basic auth against the account table.
///
/// The username is the account UUID, and the password is the credential secret.
pub struct BasicAuth(pub uuid::Uuid);

impl FromRequestParts<AppState> for BasicAuth {
    type Rejection = ServerError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        use argon2::{Argon2, PasswordHash, PasswordVerifier};

        let header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| ServerError::Unauthorized {
                message: "missing authorization header".to_string(),
            })?;

        let encoded = header
            .strip_prefix("Basic ")
            .ok_or_else(|| ServerError::Unauthorized {
                message: "expected Basic auth".to_string(),
            })?;

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| ServerError::Unauthorized {
                message: "invalid base64 in authorization header".to_string(),
            })?;

        let decoded_str = String::from_utf8(decoded).map_err(|_| ServerError::Unauthorized {
            message: "invalid utf-8 in authorization header".to_string(),
        })?;

        let (account_id_str, secret) =
            decoded_str
                .split_once(':')
                .ok_or_else(|| ServerError::Unauthorized {
                    message: "invalid Basic auth format".to_string(),
                })?;

        let account_id =
            uuid::Uuid::parse_str(account_id_str).map_err(|_| ServerError::Unauthorized {
                message: "invalid account_id".to_string(),
            })?;

        let account = crate::db::get_account_by_id(&state.db_pool, account_id)
            .await
            .map_err(|e| match e {
                ServerError::NotFound { .. } => ServerError::Unauthorized {
                    message: "invalid credentials".to_string(),
                },
                other => other,
            })?;

        let parsed_hash = PasswordHash::new(&account.secret_hash)
            .map_err(|_| ServerError::Internal("corrupt credential hash".to_string()))?;

        Argon2::default()
            .verify_password(secret.as_bytes(), &parsed_hash)
            .map_err(|_| ServerError::Unauthorized {
                message: "invalid credentials".to_string(),
            })?;

        Ok(BasicAuth(account_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_method_linkable() {
        assert!(!AuthMethod::AnonymousCredential.linkable());
        assert!(!AuthMethod::None.linkable());
    }
}
