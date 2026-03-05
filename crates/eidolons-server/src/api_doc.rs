//! OpenAPI documentation for the Eidolons API.

use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::response::EidolonsStreamMetadata;
use crate::types::{ChatCompletionChunk, ChatCompletionResponse, ChunkChoice, ChunkDelta};

/// OpenAPI documentation for the Eidolons API.
///
/// Paths and schemas are collected automatically by `OpenApiRouter` from
/// `#[utoipa::path]` annotations and their referenced `ToSchema` types.
/// Only streaming SSE types (not referenced from path annotations) and
/// metadata/security modifiers are declared here.
#[derive(OpenApi)]
#[openapi(
    info(
        title = "Eidolons API",
        description = "Privacy-transparent AI proxy API with inline verification metadata",
        version = "0.1.0",
        license(identifier = "UNLICENSED"),
    ),
    servers(
        (url = "http://localhost:8080", description = "Local"),
    ),
    tags(
        (name = "Public", description = "Unauthenticated endpoints available to anyone."),
        (name = "Linked", description = "Authenticated endpoints tied to a known account (HTTP Basic auth)."),
        (name = "Unlinked", description = "Endpoints authenticated with anonymous credit tokens — usage cannot be linked back to an account."),
    ),
    components(schemas(
        // SSE streaming types — not referenced from #[utoipa::path] response
        // bodies so they must be registered explicitly.
        ChatCompletionResponse,
        ChatCompletionChunk,
        ChunkChoice,
        ChunkDelta,
        EidolonsStreamMetadata,
    )),
    modifiers(&BasicAuthAddon)
)]
pub struct ApiDoc;

/// Adds the HTTP Basic auth security scheme to the OpenAPI spec.
struct BasicAuthAddon;

impl Modify for BasicAuthAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.get_or_insert_default();
        components.security_schemes.insert(
            "basic".to_string(),
            SecurityScheme::Http(
                HttpBuilder::new()
                    .scheme(HttpAuthScheme::Basic)
                    .description(Some(
                        "Account credentials. Username is the account UUID, \
                         password is the secret returned at account creation.",
                    ))
                    .build(),
            ),
        );
    }
}
