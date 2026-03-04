//! OpenAPI documentation for the Eidolons API.

use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};

use crate::account::{
    BalancePool, BalancesResponse, CheckoutRequest, CheckoutUrlResponse, CreateAccountResponse,
    GetAccountResponse, LedgerEntry, LedgerResponse, ListPricesResponse, PriceResponse,
    RecurringResponse, SubscriptionResponse,
};
use crate::auth::AuthMethod;
use crate::backend::TeeType;
use crate::response::{
    AttestationStatus, AuthorizationInfo, BackendAttestation, DataExposure, EidolonsResponse,
    EidolonsStreamMetadata, PrivacyMetadata, ProxyAttestation, TransportInfo, VerificationMetadata,
};
use crate::tokens::{IssueTokensRequest, IssueTokensResponse, IssuerKeyResponse, ListKeysResponse};
use crate::types::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, ChunkDelta, ContentPart, ErrorDetail, ErrorResponse, FinishReason, ImageUrl,
    Message, MessageContent, Model, ModelsResponse, Role, StopSequence, Usage,
};

/// OpenAPI documentation for the Eidolons API.
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
    paths(
        openapi_paths::health,
        openapi_paths::list_models,
        openapi_paths::chat_completions,
        openapi_paths::list_prices,
        openapi_paths::create_account,
        openapi_paths::get_account,
        openapi_paths::get_subscription,
        openapi_paths::create_checkout,
        openapi_paths::get_balances,
        openapi_paths::get_ledger,
        openapi_paths::stripe_webhook,
        openapi_paths::list_keys,
        openapi_paths::issue_tokens,
    ),
    components(schemas(
        // Request types
        ChatCompletionRequest,
        Message,
        Role,
        MessageContent,
        ContentPart,
        ImageUrl,
        StopSequence,
        // OpenAI response types (internal, used by EidolonsResponse)
        ChatCompletionResponse,
        ChatCompletionChunk,
        Choice,
        ChunkChoice,
        ChunkDelta,
        AssistantMessage,
        FinishReason,
        Usage,
        // Eidolons response types
        EidolonsResponse,
        EidolonsStreamMetadata,
        // Privacy metadata
        PrivacyMetadata,
        AuthorizationInfo,
        AuthMethod,
        DataExposure,
        TransportInfo,
        // Verification metadata
        VerificationMetadata,
        ProxyAttestation,
        AttestationStatus,
        BackendAttestation,
        TeeType,
        // Model listing types
        ModelsResponse,
        Model,
        // Account types
        CreateAccountResponse,
        GetAccountResponse,
        SubscriptionResponse,
        CheckoutUrlResponse,
        CheckoutRequest,
        // Pricing types
        ListPricesResponse,
        PriceResponse,
        RecurringResponse,
        // Balance & ledger types
        BalancesResponse,
        BalancePool,
        LedgerResponse,
        LedgerEntry,
        // Token issuance types
        ListKeysResponse,
        IssuerKeyResponse,
        IssueTokensRequest,
        IssueTokensResponse,
        // Error types
        ErrorResponse,
        ErrorDetail,
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

// Dummy functions for utoipa path documentation.
// These are never called - they exist only to provide OpenAPI endpoint metadata.
#[allow(dead_code)]
mod openapi_paths {
    use crate::account::{
        BalancesResponse, CheckoutRequest, CheckoutUrlResponse, CreateAccountResponse,
        GetAccountResponse, LedgerResponse, ListPricesResponse, SubscriptionResponse,
    };
    use crate::response::EidolonsResponse;
    use crate::tokens::{IssueTokensRequest, IssueTokensResponse, ListKeysResponse};
    use crate::types::{ChatCompletionRequest, ErrorResponse, ModelsResponse};

    /// Health check endpoint.
    #[utoipa::path(
        get,
        path = "/health",
        responses(
            (status = 200, description = "Server is healthy", body = String, example = json!({"status": "ok"}))
        )
    )]
    pub fn health() {}

    /// Create a chat completion.
    ///
    /// Proxies the request to the configured backend and returns a response
    /// enriched with privacy and verification metadata.
    #[utoipa::path(
        post,
        path = "/v1/chat/completions",
        request_body = ChatCompletionRequest,
        responses(
            (status = 200, description = "Chat completion response with privacy and verification metadata", body = EidolonsResponse),
            (status = 400, description = "Invalid request", body = ErrorResponse),
            (status = 401, description = "Authentication failed", body = ErrorResponse),
            (status = 502, description = "Upstream provider error", body = ErrorResponse)
        )
    )]
    pub fn chat_completions() {}

    /// List available models.
    #[utoipa::path(
        get,
        path = "/v1/models",
        responses(
            (status = 200, description = "List of available models", body = ModelsResponse),
            (status = 502, description = "Upstream provider error", body = ErrorResponse)
        )
    )]
    pub fn list_models() {}

    /// Create a new account.
    ///
    /// Returns the account ID and credential secret. The credential secret is
    /// returned exactly once and never stored in plaintext.
    #[utoipa::path(
        post,
        path = "/v1/account",
        responses(
            (status = 201, description = "Account created", body = CreateAccountResponse),
            (status = 500, description = "Internal error", body = ErrorResponse)
        )
    )]
    pub fn create_account() {}

    /// Get account information.
    ///
    /// Requires HTTP Basic auth (account_id:secret).
    #[utoipa::path(
        get,
        path = "/v1/account",
        security(("basic" = [])),
        responses(
            (status = 200, description = "Account info", body = GetAccountResponse),
            (status = 401, description = "Invalid credentials", body = ErrorResponse)
        )
    )]
    pub fn get_account() {}

    /// Get subscription details.
    ///
    /// Returns the active subscription and a Stripe portal URL for management.
    /// Requires HTTP Basic auth.
    #[utoipa::path(
        get,
        path = "/v1/account/subscription",
        security(("basic" = [])),
        responses(
            (status = 200, description = "Subscription details", body = SubscriptionResponse),
            (status = 401, description = "Invalid credentials", body = ErrorResponse),
            (status = 404, description = "No Stripe customer or no subscription", body = ErrorResponse),
            (status = 503, description = "Stripe not configured", body = ErrorResponse)
        )
    )]
    pub fn get_subscription() {}

    /// List available prices.
    ///
    /// Returns all active prices from Stripe with product details.
    #[utoipa::path(
        get,
        path = "/v1/prices",
        responses(
            (status = 200, description = "List of available prices", body = ListPricesResponse),
            (status = 503, description = "Stripe not configured", body = ErrorResponse)
        )
    )]
    pub fn list_prices() {}

    /// Create a checkout session.
    ///
    /// Accepts any active Stripe price ID. Automatically determines whether to
    /// create a subscription or one-time payment checkout based on the price type.
    /// For subscription prices, fails with 409 if the account already has an
    /// active subscription. Requires HTTP Basic auth.
    #[utoipa::path(
        post,
        path = "/v1/account/checkout",
        request_body = CheckoutRequest,
        security(("basic" = [])),
        responses(
            (status = 200, description = "Checkout session created", body = CheckoutUrlResponse),
            (status = 400, description = "Invalid request", body = ErrorResponse),
            (status = 401, description = "Invalid credentials", body = ErrorResponse),
            (status = 409, description = "Already subscribed", body = ErrorResponse),
            (status = 503, description = "Stripe not configured", body = ErrorResponse)
        )
    )]
    pub fn create_checkout() {}

    /// Get account balance breakdown.
    ///
    /// Returns the total available balance and per-pool breakdown.
    /// Requires HTTP Basic auth.
    #[utoipa::path(
        get,
        path = "/v1/account/balances",
        security(("basic" = [])),
        responses(
            (status = 200, description = "Balance breakdown", body = BalancesResponse),
            (status = 401, description = "Invalid credentials", body = ErrorResponse)
        )
    )]
    pub fn get_balances() {}

    /// Get credit ledger entries.
    ///
    /// Returns paginated ledger entries with optional filtering by reason type.
    /// Requires HTTP Basic auth.
    #[utoipa::path(
        get,
        path = "/v1/account/ledger",
        security(("basic" = [])),
        params(
            ("reason" = Option<String>, Query, description = "Comma-separated reason filter"),
            ("cursor" = Option<String>, Query, description = "Opaque pagination cursor from a previous response"),
            ("limit" = Option<i64>, Query, description = "Max entries to return (default 50, max 200)"),
        ),
        responses(
            (status = 200, description = "Ledger entries", body = LedgerResponse),
            (status = 401, description = "Invalid credentials", body = ErrorResponse)
        )
    )]
    pub fn get_ledger() {}

    /// Stripe webhook endpoint.
    ///
    /// Receives and processes Stripe webhook events. Verifies the webhook
    /// signature before processing. Always returns 200 except for signature
    /// verification failures.
    #[utoipa::path(
        post,
        path = "/v1/webhooks/stripe",
        responses(
            (status = 200, description = "Event received"),
            (status = 400, description = "Signature verification failed", body = ErrorResponse),
            (status = 503, description = "Webhook secret not configured", body = ErrorResponse)
        )
    )]
    pub fn stripe_webhook() {}

    /// List valid issuer public keys.
    ///
    /// Returns public keys for all epochs whose tokens are still accepted.
    /// Clients use these to determine which key to request issuance under.
    #[utoipa::path(
        get,
        path = "/v1/keys",
        responses(
            (status = 200, description = "Valid issuer public keys", body = ListKeysResponse),
        )
    )]
    pub fn list_keys() {}

    /// Issue anonymous credit tokens.
    ///
    /// Atomically debits the account's credit balance and issues a signed
    /// anonymous credit token. The issuance request and response use CBOR
    /// encoding (base64url-no-pad). Requires HTTP Basic auth.
    #[utoipa::path(
        post,
        path = "/v1/account/tokens",
        request_body = IssueTokensRequest,
        security(("basic" = [])),
        responses(
            (status = 200, description = "Token issued", body = IssueTokensResponse),
            (status = 400, description = "Invalid request", body = ErrorResponse),
            (status = 401, description = "Invalid credentials", body = ErrorResponse),
            (status = 402, description = "Insufficient credit balance", body = ErrorResponse),
            (status = 503, description = "Token issuance not configured", body = ErrorResponse),
        )
    )]
    pub fn issue_tokens() {}
}
