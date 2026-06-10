pub mod config;
pub mod db;
pub mod error;
pub mod trust_root;
pub mod updater;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anonymous_credit_tokens::{
    CreditToken, IssuanceResponse, Params, PreIssuance, PreRefund, PublicKey, Refund, SpendProof,
    credit_to_scalar, scalar_to_credit,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand_core::OsRng;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use config::Config;
use error::AppError;

// ============================================================================
// Data transfer types — returned from `AppCore` methods to the apps
// (CLI, GUI).
// ============================================================================

/// Snapshot of the current config for display.
///
/// `base_url` and `trusted_measurements` are the *resolved* values: the
/// user's override if set, the trust-root pin otherwise. UI displays these
/// directly without needing to know which source they came from.
#[derive(Clone, Debug)]
pub struct ConfigState {
    pub base_url: String,
    /// The resolved default inference model: the user's `default_model`
    /// config override if set, otherwise the embedded fallback
    /// ([`config::DEFAULT_MODEL`]).
    pub default_model: String,
    pub has_account: bool,
    pub has_account_secret: bool,
    pub domain_separator: String,
    pub trusted_measurements: Vec<MeasurementInfo>,
    pub has_hardware_root_ca: bool,
    pub has_hardware_intermediate_ca: bool,
    pub attestation_url: Option<String>,
}

#[derive(Clone, Debug)]
pub struct MeasurementInfo {
    pub snp: String,
    pub tdx_rtmr1: String,
    pub tdx_rtmr2: String,
}

#[derive(Clone, Debug)]
pub struct AccountCreateResult {
    pub id: String,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct AccountShowResult {
    pub id: String,
    pub stripe_customer_id: Option<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct PriceInfo {
    pub id: String,
    pub product_name: String,
    pub product_description: Option<String>,
    pub amount_display: String,
    pub recurrence: String,
    pub credits: i64,
}

#[derive(Clone, Debug)]
pub struct BalancesResult {
    pub available: i64,
    pub pools: Vec<BalancePoolInfo>,
}

#[derive(Clone, Debug)]
pub struct BalancePoolInfo {
    pub amount: i64,
    pub source: String,
    pub expires_at: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct CredentialInfo {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
}

#[derive(Clone, Debug)]
pub struct InFlightCredentialInfo {
    pub nonce: String,
    pub credits: i64,
    pub generation: i64,
    pub spend_amount: i64,
}

#[derive(Clone, Debug)]
pub struct AllocateResult {
    pub nonce: String,
    pub credits: i64,
    pub issuer_key_id: String,
}

#[derive(Clone, Debug)]
pub struct ChatResult {
    pub space_id: String,
    pub content: String,
    pub model: String,
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub credits_charged: i64,
}

#[derive(Clone, Debug)]
pub struct SpaceInfo {
    pub id: String,
    pub title: Option<String>,
    /// First ~120 chars of the first user message in the space — the UI's
    /// fallback line for untitled spaces. `None` for empty spaces.
    pub snippet: Option<String>,
    pub created_at: i64,
    /// Max `action.created_at` in the space; equals `created_at` for spaces
    /// with no actions yet.
    pub last_activity_at: i64,
    /// Count of terminal (complete/cancelled) actions in the space.
    pub message_count: i64,
    /// When the space was archived, if it has been. Always `None` unless
    /// listing was asked to include archived spaces.
    pub archived_at: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct SpaceMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug)]
pub struct ModelInfo {
    pub id: String,
    pub context_length: u64,
    /// Credits charged per prompt token. Credits are micro-USD-denominated,
    /// so this is numerically the same as USD per million prompt tokens.
    /// Zero for per-request-priced models.
    pub prompt_credits_per_token: f64,
    /// Credits charged per completion token (see
    /// [`prompt_credits_per_token`](Self::prompt_credits_per_token)).
    pub completion_credits_per_token: f64,
    /// Flat per-request price for models that charge per request rather
    /// than per token (e.g. transcription); `None` for token-priced models.
    pub request_credits: Option<f64>,
}

/// Default number of credits to allocate into a fresh anonymous credential
/// when a chat needs one and the account has balance available
/// (auto-provisioning). The actual amount is
/// `min(available, max(DEFAULT_ALLOCATION_CREDITS, required))` — see
/// [`auto_allocation_amount`].
///
/// Why 1,000,000: credits are micro-USD-denominated (the server's
/// `PRICING_SCALE_FACTOR` is 1e6 — `usd_per_M_tokens × markup` becomes
/// credits-per-token directly, e.g. gemma4-31b output $1.00/M × 1.5 markup
/// = 1.5 credits/token). A single chat turn's worst-case hold is
/// `prompt_bytes × prompt_rate + 4096 × completion_rate`: ≈6,200 credits
/// for the default gemma4-31b, ≈32,000 for the most expensive catalog
/// models (output 7.875 credits/token) — mostly refunded after the actual
/// usage settles. 1,000,000 credits ($1.00 of balance) therefore covers
/// ~30 worst-case holds or 100+ typical turns: small enough that only a
/// sliver of the account balance is parked in one unlinkable credential
/// at a time, large enough that re-allocation (an account-linked,
/// timing-correlatable operation) stays infrequent.
pub const DEFAULT_ALLOCATION_CREDITS: i64 = 1_000_000;

/// Decide how many credits to auto-allocate, given the account's available
/// balance and the credits required for the operation that triggered
/// provisioning. Pure function so the decision logic is unit-testable
/// without HTTP.
///
/// Returns [`AppError::InsufficientBalance`] when the balance cannot cover
/// even the required amount; otherwise the chunk size:
/// `min(available, max(DEFAULT_ALLOCATION_CREDITS, required))`.
fn auto_allocation_amount(available: i64, required: i64) -> Result<i64, AppError> {
    if available < required {
        return Err(AppError::InsufficientBalance {
            available,
            required,
        });
    }
    Ok(available.min(DEFAULT_ALLOCATION_CREDITS.max(required)))
}

/// Incremental events emitted by `AppCore::chat_stream`. The terminal
/// outcome is the function's `Result<ChatResult, AppError>` return value;
/// senders close their channel when the function returns.
#[derive(Clone, Debug)]
pub enum ChatStreamEvent {
    /// A piece of the model's reasoning ("thinking") output. Append to a
    /// running buffer; treat empty events as no-ops.
    ReasoningDelta(String),
    /// A piece of the assistant's answer text. Append to a running buffer.
    ContentDelta(String),
}

// ============================================================================
// Inner — shared state used by AppCore, wrapped in Arc so it can move into
// spawned futures on the owned tokio runtime.
// ============================================================================

struct Inner {
    config_path: PathBuf,
    data_dir: PathBuf,
    db: tokio::sync::OnceCell<turso::Database>,
}

// --- Config helpers (sync) ---------------------------------------------------

impl Inner {
    fn load_config(&self) -> Config {
        Config::load_from(&self.config_path)
    }

    fn require_credentials<'a>(&self, cfg: &'a Config) -> Result<(&'a str, &'a str), AppError> {
        match (&cfg.account_id, &cfg.account_secret) {
            (Some(id), Some(secret)) => Ok((id, secret)),
            _ => Err(AppError::NotConfigured {
                message: "account not configured".into(),
            }),
        }
    }
}

// --- Async infrastructure ----------------------------------------------------

impl Inner {
    async fn db_conn(&self) -> Result<turso::Connection, AppError> {
        let database = self.db.get_or_try_init(|| db::open(&self.data_dir)).await?;
        database.connect().map_err(AppError::db)
    }

    async fn build_client(
        &self,
        config: &Config,
        attestation_observer: Option<tinfoil_verifier::AttestationObserver>,
    ) -> Result<reqwest::Client, AppError> {
        let origin = config.base_url();
        let allowed_measurements = config.trusted_measurements();

        let hardware_root_der =
            config::parse_cert_config(config.hardware_root_ca.as_deref(), "hardware_root_ca")?;
        let hardware_intermediate_der = config::parse_cert_config(
            config.hardware_intermediate_ca.as_deref(),
            "hardware_intermediate_ca",
        )?;

        tinfoil_verifier::attesting_client(tinfoil_verifier::AttestingClientConfig {
            allowed_measurements: &allowed_measurements,
            inference_base_url: origin,
            atc_url: config.attestation_url.as_deref(),
            enclave_repo: Some(config.attestation_repo()),
            trusted_ark_der: hardware_root_der.as_deref(),
            trusted_ask_der: hardware_intermediate_der.as_deref(),
            tdx_advisory_allowlist: None,
            tdx_observer: None,
            snp_min_tcb: None,
            snp_observer: None,
            attestation_observer,
            tls_roots: load_native_root_store(),
        })
        .await
        .map_err(|e| AppError::Attestation {
            message: format!("attestation client build failed: {e}"),
        })
    }
}

// --- High-level async operations (run on the owned tokio runtime) ------------

impl Inner {
    async fn account_show(&self) -> Result<AccountShowResult, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let (id, secret) = self.require_credentials(&cfg)?;

        let client = self.build_client(&cfg, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/account"))
            .basic_auth(id, Some(secret))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let account: GetAccountResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(AccountShowResult {
            id: account.id.to_string(),
            stripe_customer_id: account.stripe_customer_id,
            created_at: iso_to_ms(&account.created_at)?,
        })
    }

    async fn account_create(&self) -> Result<AccountCreateResult, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();

        if cfg.account_id.is_some() || cfg.account_secret.is_some() {
            return Err(AppError::Config {
                message: "account credentials already configured — reset first".into(),
            });
        }

        let client = self.build_client(&cfg, None).await?;
        let resp = client
            .post(format!("{base_url}/v1/account"))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let created: CreateAccountResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        let mut cfg = self.load_config();
        cfg.account_id = Some(created.account_id.to_string());
        cfg.account_secret = Some(created.secret);
        cfg.save_to(&self.config_path)?;

        Ok(AccountCreateResult {
            id: created.account_id.to_string(),
            created_at: iso_to_ms(&created.created_at)?,
        })
    }

    async fn account_prices(&self) -> Result<Vec<PriceInfo>, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();

        let client = self.build_client(&cfg, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/prices"))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let prices: ListPricesResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(prices
            .data
            .into_iter()
            .map(|p| {
                let amount_display = p
                    .unit_amount
                    .map(|a| format!("{}.{:02} {}", a / 100, a % 100, p.currency.to_uppercase()))
                    .unwrap_or_else(|| "free".to_string());

                let recurrence = p
                    .recurring
                    .as_ref()
                    .map(|r| {
                        if r.interval_count == 1 {
                            format!("/{}", r.interval)
                        } else {
                            format!("/{}x{}", r.interval_count, r.interval)
                        }
                    })
                    .unwrap_or_default();

                PriceInfo {
                    id: p.id,
                    product_name: p.product_name,
                    product_description: p.product_description,
                    amount_display,
                    recurrence,
                    credits: p.credits,
                }
            })
            .collect())
    }

    async fn account_checkout(&self, price_id: &str) -> Result<String, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let (id, secret) = self.require_credentials(&cfg)?;

        let client = self.build_client(&cfg, None).await?;
        let resp = client
            .post(format!("{base_url}/v1/account/checkout"))
            .basic_auth(id, Some(secret))
            .json(&serde_json::json!({ "price_id": price_id }))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let checkout: CheckoutUrlResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(checkout.checkout_url)
    }

    async fn account_balances(&self) -> Result<BalancesResult, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let (id, secret) = self.require_credentials(&cfg)?;

        let client = self.build_client(&cfg, None).await?;
        let resp = client
            .get(format!("{base_url}/v1/account/balances"))
            .basic_auth(id, Some(secret))
            .send()
            .await
            .map_err(AppError::from_request)?;

        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let balances: BalancesResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse response: {e}"),
            })?;

        Ok(BalancesResult {
            available: balances.available,
            pools: balances
                .pools
                .into_iter()
                .map(|p| {
                    let expires_at = p.expires_at.as_deref().map(iso_to_ms).transpose()?;
                    Ok::<_, AppError>(BalancePoolInfo {
                        amount: p.amount,
                        source: p.source,
                        expires_at,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    async fn account_allocate(&self, credits: i64) -> Result<AllocateResult, AppError> {
        if credits <= 0 {
            return Err(AppError::Credential {
                message: "credits must be greater than 0".into(),
            });
        }

        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let (account_id, secret) = self.require_credentials(&cfg)?;
        let client = self.build_client(&cfg, None).await?;

        // 1. Fetch issuer keys
        let resp = client
            .get(format!("{base_url}/v1/keys"))
            .send()
            .await
            .map_err(AppError::from_request)?;
        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let keys: ListKeysResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse keys response: {e}"),
            })?;

        let expected_ds = cfg.domain_separator();
        let key = keys
            .data
            .iter()
            .find(|k| k.domain_separator == expected_ds)
            .ok_or_else(|| {
                let server_ds: Vec<&str> = keys
                    .data
                    .iter()
                    .map(|k| k.domain_separator.as_str())
                    .collect();
                AppError::Credential {
                    message: format!(
                        "no issuer key matches expected domain separator \"{expected_ds}\"\n\
                         server advertised: {server_ds:?}"
                    ),
                }
            })?;

        let public_key_cbor =
            URL_SAFE_NO_PAD
                .decode(&key.public_key)
                .map_err(|e| AppError::Credential {
                    message: format!("invalid base64 public key: {e}"),
                })?;

        let public_key =
            PublicKey::from_cbor(&public_key_cbor).map_err(|e| AppError::Credential {
                message: format!("invalid public key CBOR: {e}"),
            })?;

        let params = params_from_domain_separator(expected_ds)?;

        // 2. Open DB and store issuer key
        let db_conn = self.db_conn().await?;
        let params_hash = blake3::hash(key.domain_separator.as_bytes())
            .to_hex()
            .to_string();
        let now = now_ms();
        let expires_at = iso_to_ms(&key.issue_until)?;

        db::upsert_issuer_key(
            &db_conn,
            &key.id,
            &params_hash,
            &public_key_cbor,
            key.domain_separator.as_bytes(),
            expires_at,
            now,
        )
        .await?;

        // 3. Create PreIssuance checkpoint
        let pre_issuance = PreIssuance::random(OsRng);
        let pre_issuance_cbor = pre_issuance.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode pre_issuance: {e}"),
        })?;
        let pre_credential_id = Uuid::now_v7().to_string();
        db::insert_pre_credential_issuance(
            &db_conn,
            &pre_credential_id,
            &key.id,
            &pre_issuance_cbor,
            credits,
            now,
        )
        .await?;

        // 4. Send issuance request
        let issuance_request = pre_issuance.request(&params, OsRng);
        let request_cbor = issuance_request
            .to_cbor()
            .map_err(|e| AppError::Credential {
                message: format!("failed to encode issuance request: {e}"),
            })?;

        let resp = client
            .post(format!("{base_url}/v1/account/credentials"))
            .basic_auth(account_id, Some(secret))
            .json(&serde_json::json!({
                "issuance_request": URL_SAFE_NO_PAD.encode(&request_cbor),
                "credits": credits,
            }))
            .send()
            .await
            .map_err(AppError::from_request)?;
        let (status, body) = read_response(resp).await?;
        check_status(status, &body)?;

        let issued: IssueCredentialsResponse =
            serde_json::from_str(&body).map_err(|e| AppError::Network {
                message: format!("failed to parse issuance response: {e}"),
            })?;

        // 5. Construct CreditToken
        let response_cbor = URL_SAFE_NO_PAD
            .decode(&issued.issuance_response)
            .map_err(|e| AppError::Credential {
                message: format!("invalid issuance response base64: {e}"),
            })?;
        let issuance_response =
            IssuanceResponse::from_cbor(&response_cbor).map_err(|e| AppError::Credential {
                message: format!("invalid issuance response CBOR: {e}"),
            })?;
        let credit_token = pre_issuance
            .to_credit_token::<128>(&params, &public_key, &issuance_request, &issuance_response)
            .map_err(|e| AppError::Credential {
                message: format!("failed to construct credit token: {e}"),
            })?;

        // 6. Store credential
        let token_cbor = credit_token.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode credit token: {e}"),
        })?;
        let nonce_hex = hex_encode(&credit_token.nullifier().to_bytes());
        let token_credits =
            scalar_to_credit::<128>(&credit_token.credits()).map_err(|e| AppError::Credential {
                message: format!("invalid credit amount in token: {e}"),
            })?;

        db::insert_credential(
            &db_conn,
            &nonce_hex,
            &pre_credential_id,
            &issued.issuer_key_id,
            &token_cbor,
            token_credits as i64,
            0,
            now,
        )
        .await?;

        Ok(AllocateResult {
            nonce: nonce_hex,
            credits: issued.credits,
            issuer_key_id: issued.issuer_key_id,
        })
    }

    async fn wallet_credentials(&self) -> Result<Vec<CredentialInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_active_credentials(&db_conn).await?;
        Ok(rows
            .into_iter()
            .map(|c| CredentialInfo {
                nonce: c.nonce,
                credits: c.credits,
                generation: c.generation,
            })
            .collect())
    }

    async fn wallet_spending_credentials(&self) -> Result<Vec<InFlightCredentialInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_spending_credentials(&db_conn).await?;
        Ok(rows
            .into_iter()
            .map(|r| InFlightCredentialInfo {
                nonce: r.nonce,
                credits: r.credits,
                generation: r.generation,
                spend_amount: r.spend_amount,
            })
            .collect())
    }

    async fn recover_spending_credentials(&self) -> Result<Vec<String>, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let client = self.build_client(&cfg, None).await?;
        let db_conn = self.db_conn().await?;
        let params = params_from_domain_separator(cfg.domain_separator())?;
        let now = now_ms();

        let rows = db::list_spending_credentials(&db_conn).await?;
        let mut recovered = Vec::new();

        for row in rows {
            let spend_proof_cbor = row.spend_proof_data;

            let spend_proof = match SpendProof::<128>::from_cbor(&spend_proof_cbor) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let pre_refund = match PreRefund::from_cbor(&row.pre_refund_data) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let public_key = match PublicKey::from_cbor(&row.public_key_data) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let issuer_key_hash = match hex_decode(&row.issuer_key_id) {
                Ok(h) => h,
                Err(_) => continue,
            };

            // Reconstruct the PrivateToken auth header
            let challenge_digest = compute_challenge_digest();
            let mut token_bytes = Vec::new();
            token_bytes.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
            token_bytes.extend_from_slice(&challenge_digest);
            token_bytes.extend_from_slice(&issuer_key_hash);
            token_bytes.extend_from_slice(&spend_proof_cbor);
            let token_b64 = URL_SAFE_NO_PAD.encode(&token_bytes);
            let auth_value = format!("PrivateToken token=\"{token_b64}\"");

            if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await
                && process_refund(
                    &refund_obj,
                    &params,
                    &spend_proof,
                    &pre_refund,
                    &public_key,
                    &db_conn,
                    &row.pre_credential_id,
                    row.generation + 1,
                    now,
                )
                .await
                .is_ok()
            {
                recovered.push(row.nonce);
            }
        }

        Ok(recovered)
    }

    async fn available_models(&self) -> Result<Vec<ModelInfo>, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let client = self.build_client(&cfg, None).await?;

        let models = fetch_models(&client, base_url).await?;
        Ok(models
            .data
            .into_iter()
            .map(|m| ModelInfo {
                id: m.id,
                context_length: m.context_length,
                prompt_credits_per_token: m.pricing.per_prompt_token.credits_per_unit(),
                completion_credits_per_token: m.pricing.per_completion_token.credits_per_unit(),
                request_credits: m
                    .pricing
                    .per_request
                    .as_ref()
                    .map(ScaledPriceInfo::credits_per_unit),
            })
            .collect())
    }

    async fn list_spaces(&self, include_archived: bool) -> Result<Vec<SpaceInfo>, AppError> {
        let db_conn = self.db_conn().await?;
        let rows = db::list_spaces(&db_conn, include_archived).await?;
        let mut spaces = Vec::with_capacity(rows.len());
        for r in rows {
            let snippet = db::first_user_text(&db_conn, &r.id)
                .await?
                .as_deref()
                .and_then(snippet_of);
            spaces.push(SpaceInfo {
                id: r.id,
                title: r.title,
                snippet,
                created_at: r.created_at,
                last_activity_at: r.last_activity_at,
                message_count: r.message_count,
                archived_at: r.archived_at,
            });
        }
        Ok(spaces)
    }

    async fn get_space_messages(&self, space_id: &str) -> Result<Vec<SpaceMessage>, AppError> {
        let db_conn = self.db_conn().await?;
        db::get_space(&db_conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        let action_rows = db::get_space_actions_for_context(&db_conn, space_id).await?;
        Ok(actions_to_messages(&action_rows))
    }

    async fn create_space(&self, title: Option<&str>) -> Result<SpaceInfo, AppError> {
        let db_conn = self.db_conn().await?;
        let now = now_ms();
        let space_id = Uuid::now_v7().to_string();
        db::insert_space(&db_conn, &space_id, title, "unlinked", now).await?;

        let user_participant_id =
            db::ensure_participant(&db_conn, "human", "user", None, now).await?;
        db::insert_space_participant(&db_conn, &space_id, &user_participant_id, "owner", now)
            .await?;

        Ok(SpaceInfo {
            id: space_id,
            title: title.map(String::from),
            snippet: None,
            created_at: now,
            last_activity_at: now,
            message_count: 0,
            archived_at: None,
        })
    }

    async fn archive_space(&self, space_id: &str) -> Result<bool, AppError> {
        let db_conn = self.db_conn().await?;
        db::archive_space(&db_conn, space_id, now_ms()).await
    }

    /// Find a credential that can cover `charge_credits`, auto-provisioning
    /// one from the account balance when none exists.
    ///
    /// Resolution order:
    /// 1. An active local credential with enough credits → use it.
    /// 2. No usable credential and no account configured →
    ///    [`AppError::NoAccount`] (the UI routes to account creation).
    /// 3. Account exists: fetch balances; if the available balance cannot
    ///    cover the charge → [`AppError::InsufficientBalance`] (the UI
    ///    routes to purchase). Otherwise allocate
    ///    `min(available, max(DEFAULT_ALLOCATION_CREDITS, charge))` and
    ///    use the freshly issued credential.
    ///
    /// This keeps credential provisioning out of the happy path's UI: the
    /// explicit `account_allocate` flow still works, but a chat never fails
    /// just because the wallet is empty while the account is funded.
    async fn ensure_spendable_credential(
        &self,
        cfg: &Config,
        db_conn: &turso::Connection,
        charge_credits: i64,
    ) -> Result<db::SpendableCredential, AppError> {
        if let Some(cred) = db::find_spendable_credential(db_conn, charge_credits).await? {
            return Ok(cred);
        }

        if cfg.account_id.is_none() || cfg.account_secret.is_none() {
            return Err(AppError::NoAccount);
        }

        let balances = self.account_balances().await?;
        let amount = auto_allocation_amount(balances.available, charge_credits)?;
        self.account_allocate(amount).await?;

        db::find_spendable_credential(db_conn, charge_credits)
            .await?
            .ok_or_else(|| AppError::Credential {
                message: "credential was allocated but is not spendable".into(),
            })
    }

    async fn rename_space(&self, space_id: &str, title: &str) -> Result<(), AppError> {
        let db_conn = self.db_conn().await?;
        db::get_space(&db_conn, space_id)
            .await?
            .ok_or_else(|| AppError::NotConfigured {
                message: format!("space not found: {space_id}"),
            })?;
        db::update_space_title(&db_conn, space_id, title).await
    }

    async fn chat(
        &self,
        prompt: &str,
        model: &str,
        space_id: Option<&str>,
    ) -> Result<ChatResult, AppError> {
        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let now = now_ms();

        let db_conn = self.db_conn().await?;
        let provider_id = db::ensure_provider(&db_conn, "eidola", "inference", now).await?;

        let attestation_log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>> =
            Arc::new(Mutex::new(Vec::new()));
        let log_clone = attestation_log.clone();
        let observer: Option<tinfoil_verifier::AttestationObserver> = Some(Arc::new(
            move |att: tinfoil_verifier::VerifiedAttestation| {
                log_clone.lock().unwrap().push(att);
            },
        ));

        let client = self.build_client(&cfg, observer).await?;

        let models = fetch_models(&client, base_url).await?;
        let mut connection_id =
            flush_attestations(&attestation_log, &db_conn, &provider_id, base_url, now).await?;

        let model_entry =
            models
                .data
                .iter()
                .find(|m| m.id == model)
                .ok_or_else(|| AppError::NotConfigured {
                    message: format!("model not found: {model}"),
                })?;

        let max_completion_tokens = (model_entry.context_length).min(4096) as u32;

        let user_participant_id =
            db::ensure_participant(&db_conn, "human", "user", None, now).await?;
        let model_participant_id =
            db::ensure_participant(&db_conn, "agent", model, Some(&provider_id), now).await?;

        // Reuse existing space or create a new one
        let (space_id, space_title) = if let Some(sid) = space_id {
            let row =
                db::get_space(&db_conn, sid)
                    .await?
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("space not found: {sid}"),
                    })?;
            // Ensure model participant is in the space (may be new model for this space)
            let _ =
                db::insert_space_participant(&db_conn, sid, &model_participant_id, "member", now)
                    .await; // ignore duplicate
            (sid.to_string(), row.title)
        } else {
            let sid = Uuid::now_v7().to_string();
            db::insert_space(&db_conn, &sid, None, "unlinked", now).await?;
            db::insert_space_participant(&db_conn, &sid, &user_participant_id, "owner", now)
                .await?;
            db::insert_space_participant(&db_conn, &sid, &model_participant_id, "member", now)
                .await?;
            (sid, None)
        };

        // Load prior actions to build multi-turn context — needed both for
        // credit estimation (total prompt size) and message assembly.
        let prior_action_rows = db::get_space_actions_for_context(&db_conn, &space_id).await?;
        let prior_messages = actions_to_messages(&prior_action_rows);

        // Estimate prompt size from ALL messages (prior history + current prompt)
        let total_prompt_bytes: u128 = prior_messages
            .iter()
            .map(|m| m.content.len() as u128)
            .sum::<u128>()
            + prompt.len() as u128;

        let sf = model_entry.pricing.per_prompt_token.scale_factor as u128;
        let prompt_rate = model_entry.pricing.per_prompt_token.value as u128;
        let prompt_credits = (total_prompt_bytes * prompt_rate).div_ceil(sf);
        let completion_rate = model_entry.pricing.per_completion_token.value as u128;
        let completion_credits = (max_completion_tokens as u128 * completion_rate).div_ceil(sf);
        let charge_credits = prompt_credits + completion_credits;

        if charge_credits == 0 {
            return Err(AppError::Credential {
                message: "computed charge is zero — model pricing may be missing".into(),
            });
        }

        let cred = self
            .ensure_spendable_credential(&cfg, &db_conn, charge_credits as i64)
            .await?;

        let credit_token =
            CreditToken::from_cbor(&cred.data).map_err(|e| AppError::Credential {
                message: format!("failed to decode credential: {e}"),
            })?;
        let public_key =
            PublicKey::from_cbor(&cred.public_key_data).map_err(|e| AppError::Credential {
                message: format!("failed to decode public key: {e}"),
            })?;

        let params = params_from_domain_separator(cfg.domain_separator())?;

        let charge_scalar =
            credit_to_scalar::<128>(charge_credits).map_err(|e| AppError::Credential {
                message: format!("invalid charge amount: {e:?}"),
            })?;
        let (spend_proof, pre_refund) = credit_token
            .prove_spend::<128>(&params, charge_scalar, OsRng)
            .map_err(|e| AppError::Credential {
                message: format!("failed to create spend proof: {e:?}"),
            })?;

        let pre_refund_cbor = pre_refund.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode pre_refund: {e}"),
        })?;
        let spend_proof_cbor = spend_proof.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode spend proof: {e}"),
        })?;
        let pre_cred_id = Uuid::now_v7().to_string();
        db::insert_pre_credential_refund(
            &db_conn,
            &pre_cred_id,
            &cred.nonce,
            &cred.issuer_key_id,
            &pre_refund_cbor,
            charge_credits as i64,
            &spend_proof_cbor,
            now,
        )
        .await?;

        let issuer_key_hash = hex_decode(&cred.issuer_key_id)?;
        let challenge_digest = compute_challenge_digest();

        let mut token_bytes = Vec::new();
        token_bytes.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
        token_bytes.extend_from_slice(&challenge_digest);
        token_bytes.extend_from_slice(&issuer_key_hash);
        token_bytes.extend_from_slice(&spend_proof_cbor);

        let token_b64 = URL_SAFE_NO_PAD.encode(&token_bytes);
        let auth_value = format!("PrivateToken token=\"{token_b64}\"");

        // Find the last action in the space for antecedent linking
        let last_action_id = db::last_action_in_space(&db_conn, &space_id).await?;

        // Insert the new user_input action
        let user_action_id = Uuid::now_v7().to_string();
        db::insert_action(
            &db_conn,
            &db::ActionEntry {
                id: user_action_id.clone(),
                space_id: space_id.clone(),
                participant_id: user_participant_id,
                action_type: "user_input".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at: now,
            },
        )
        .await?;
        db::insert_text_content_block(
            &db_conn,
            &Uuid::now_v7().to_string(),
            &user_action_id,
            0,
            "text",
            prompt,
        )
        .await?;

        // Auto-title: the first exchange in an untitled space names the
        // space after the user's prompt. Purely local — no model call.
        if space_title.is_none()
            && prior_messages.is_empty()
            && let Some(title) = derive_space_title(prompt)
        {
            db::update_space_title(&db_conn, &space_id, &title).await?;
        }

        // Link to previous action as antecedent
        if let Some(ref ante_id) = last_action_id {
            db::insert_action_antecedent(&db_conn, &user_action_id, ante_id, 0).await?;
        }

        // Build the messages array: prior history + current prompt
        let mut messages: Vec<serde_json::Value> = prior_messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        messages.push(serde_json::json!({"role": "user", "content": prompt}));

        let request_body_json = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
        });
        let request_at = now_ms();

        // Send the chat request. On failure, attempt refund recovery before
        // propagating the error so the credential isn't abandoned.
        let chat_result = client
            .post(format!("{base_url}/v1/chat/completions"))
            .header("Authorization", &auth_value)
            .json(&request_body_json)
            .send()
            .await;
        let response_at = now_ms();

        let (status, response_text, body) = match chat_result {
            Ok(resp) => {
                if let Some(new_cid) =
                    flush_attestations(&attestation_log, &db_conn, &provider_id, base_url, now)
                        .await?
                {
                    connection_id = Some(new_cid);
                }

                let status = resp.status();
                let text = resp.text().await.map_err(|e| AppError::Network {
                    message: format!("failed to read response: {e}"),
                })?;
                let parsed: serde_json::Value =
                    serde_json::from_str(&text).map_err(|e| AppError::Network {
                        message: format!("failed to parse response JSON: {e}"),
                    })?;
                (status, text, parsed)
            }
            Err(e) => {
                // Network error — the server may or may not have received the
                // request. Try to recover the refund token.
                let original_err = AppError::from_request(e);
                if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await {
                    let _ = process_refund(
                        &refund_obj,
                        &params,
                        &spend_proof,
                        &pre_refund,
                        &public_key,
                        &db_conn,
                        &pre_cred_id,
                        cred.generation + 1,
                        now,
                    )
                    .await;
                }
                return Err(original_err);
            }
        };

        // Process the refund token from the response. If none is present,
        // attempt recovery from the server.
        let mut refund_stored = false;
        if let Some(refund_obj) = body.get("refund") {
            process_refund(
                refund_obj,
                &params,
                &spend_proof,
                &pre_refund,
                &public_key,
                &db_conn,
                &pre_cred_id,
                cred.generation + 1,
                now,
            )
            .await?;
            refund_stored = true;
        }

        if !refund_stored {
            // No refund in the response — try the recovery endpoint.
            if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await {
                let _ = process_refund(
                    &refund_obj,
                    &params,
                    &spend_proof,
                    &pre_refund,
                    &public_key,
                    &db_conn,
                    &pre_cred_id,
                    cred.generation + 1,
                    now,
                )
                .await;
            }
        }

        let usage = body.get("usage");
        let input_tokens = usage
            .and_then(|u| u.get("prompt_tokens"))
            .and_then(|v| v.as_i64());
        let output_tokens = usage
            .and_then(|u| u.get("completion_tokens"))
            .and_then(|v| v.as_i64());

        let inference_action_id = Uuid::now_v7().to_string();
        db::insert_action(
            &db_conn,
            &db::ActionEntry {
                id: inference_action_id.clone(),
                space_id: space_id.clone(),
                participant_id: model_participant_id,
                action_type: "inference".to_string(),
                status: if status.is_success() {
                    "complete"
                } else {
                    "error"
                }
                .to_string(),
                intent: None,
                model: Some(model.to_string()),
                input_tokens,
                output_tokens,
                credits_consumed: Some(charge_credits as i64),
                created_at: now_ms(),
            },
        )
        .await?;
        db::insert_action_antecedent(&db_conn, &inference_action_id, &user_action_id, 0).await?;

        // Record context assembly: all prior actions + the new user action
        let context_assembly_id = Uuid::now_v7().to_string();
        db::insert_context_assembly(
            &db_conn,
            &context_assembly_id,
            &inference_action_id,
            None,
            input_tokens,
            false,
            now_ms(),
        )
        .await?;

        let prior_action_ids = db::space_action_ids(&db_conn, &space_id).await?;
        for (pos, aid) in prior_action_ids.iter().enumerate() {
            // Skip the inference action we just inserted (it's not context, it's the output)
            if aid != &inference_action_id {
                db::insert_context_assembly_action(&db_conn, &context_assembly_id, aid, pos as i64)
                    .await?;
            }
        }

        let response_content = body
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        if !response_content.is_empty() {
            db::insert_text_content_block(
                &db_conn,
                &Uuid::now_v7().to_string(),
                &inference_action_id,
                0,
                "text",
                &response_content,
            )
            .await?;
        }

        db::insert_request(
            &db_conn,
            &db::Request {
                id: Uuid::now_v7().to_string(),
                connection_id,
                action_id: Some(inference_action_id),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                request_headers: None,
                request_body: Some(request_body_json.to_string().into_bytes()),
                response_status: Some(status.as_u16() as i64),
                response_headers: None,
                response_body: Some(response_text.as_bytes().to_vec()),
                request_at,
                response_at: Some(response_at),
                duration_ms: Some(response_at - request_at),
                error: None,
                credential_nonce: Some(cred.nonce.clone()),
                created_at: now_ms(),
            },
        )
        .await?;

        if !status.is_success() {
            return Err(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&response_text),
            });
        }

        Ok(ChatResult {
            space_id,
            content: response_content,
            model: model.to_string(),
            input_tokens,
            output_tokens,
            credits_charged: charge_credits as i64,
        })
    }

    /// Streaming counterpart to `chat`. Mirrors the same setup (ACT token,
    /// DB action insertion, prior-context assembly) and post-stream cleanup
    /// (refund recovery, DB persistence of the inference action and content
    /// blocks), but sends `stream: true` upstream and forwards each SSE chunk
    /// to `sender` as it arrives.
    ///
    /// Reasoning shape: we accept both `delta.reasoning_content` (OpenAI-style
    /// extension used by some providers) and `delta.reasoning` (vLLM's
    /// extension). Either form is forwarded as `ReasoningDelta`. Unknown
    /// fields are ignored — if Tinfoil's upstream uses a third spelling, the
    /// thinking section will simply stay empty until we adapt.
    ///
    /// Refund handling differs from `chat` only in *where* the refund token
    /// comes from: SSE responses have no inline body to carry it, so we
    /// always go through the `/v1/credentials/refund` recovery endpoint
    /// after the stream ends. The credential is left in `pre_credential`
    /// state until that recovery completes, same as today's network-error
    /// path.
    #[allow(clippy::too_many_arguments)]
    async fn chat_stream(
        &self,
        prompt: &str,
        model: &str,
        space_id: Option<&str>,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        use futures_util::StreamExt;

        let cfg = self.load_config();
        let base_url = cfg.base_url();
        let now = now_ms();

        let db_conn = self.db_conn().await?;
        let provider_id = db::ensure_provider(&db_conn, "eidola", "inference", now).await?;

        let attestation_log: Arc<Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>> =
            Arc::new(Mutex::new(Vec::new()));
        let log_clone = attestation_log.clone();
        let observer: Option<tinfoil_verifier::AttestationObserver> = Some(Arc::new(
            move |att: tinfoil_verifier::VerifiedAttestation| {
                log_clone.lock().unwrap().push(att);
            },
        ));

        let client = self.build_client(&cfg, observer).await?;

        let models = fetch_models(&client, base_url).await?;
        let mut connection_id =
            flush_attestations(&attestation_log, &db_conn, &provider_id, base_url, now).await?;

        let model_entry =
            models
                .data
                .iter()
                .find(|m| m.id == model)
                .ok_or_else(|| AppError::NotConfigured {
                    message: format!("model not found: {model}"),
                })?;

        let max_completion_tokens = (model_entry.context_length).min(4096) as u32;

        let user_participant_id =
            db::ensure_participant(&db_conn, "human", "user", None, now).await?;
        let model_participant_id =
            db::ensure_participant(&db_conn, "agent", model, Some(&provider_id), now).await?;

        let (space_id, space_title) = if let Some(sid) = space_id {
            let row =
                db::get_space(&db_conn, sid)
                    .await?
                    .ok_or_else(|| AppError::NotConfigured {
                        message: format!("space not found: {sid}"),
                    })?;
            let _ =
                db::insert_space_participant(&db_conn, sid, &model_participant_id, "member", now)
                    .await;
            (sid.to_string(), row.title)
        } else {
            let sid = Uuid::now_v7().to_string();
            db::insert_space(&db_conn, &sid, None, "unlinked", now).await?;
            db::insert_space_participant(&db_conn, &sid, &user_participant_id, "owner", now)
                .await?;
            db::insert_space_participant(&db_conn, &sid, &model_participant_id, "member", now)
                .await?;
            (sid, None)
        };

        let prior_action_rows = db::get_space_actions_for_context(&db_conn, &space_id).await?;
        let prior_messages = actions_to_messages(&prior_action_rows);

        let total_prompt_bytes: u128 = prior_messages
            .iter()
            .map(|m| m.content.len() as u128)
            .sum::<u128>()
            + prompt.len() as u128;

        let sf = model_entry.pricing.per_prompt_token.scale_factor as u128;
        let prompt_rate = model_entry.pricing.per_prompt_token.value as u128;
        let prompt_credits = (total_prompt_bytes * prompt_rate).div_ceil(sf);
        let completion_rate = model_entry.pricing.per_completion_token.value as u128;
        let completion_credits = (max_completion_tokens as u128 * completion_rate).div_ceil(sf);
        let charge_credits = prompt_credits + completion_credits;

        if charge_credits == 0 {
            return Err(AppError::Credential {
                message: "computed charge is zero — model pricing may be missing".into(),
            });
        }

        let cred = self
            .ensure_spendable_credential(&cfg, &db_conn, charge_credits as i64)
            .await?;

        let credit_token =
            CreditToken::from_cbor(&cred.data).map_err(|e| AppError::Credential {
                message: format!("failed to decode credential: {e}"),
            })?;
        let public_key =
            PublicKey::from_cbor(&cred.public_key_data).map_err(|e| AppError::Credential {
                message: format!("failed to decode public key: {e}"),
            })?;

        let params = params_from_domain_separator(cfg.domain_separator())?;

        let charge_scalar =
            credit_to_scalar::<128>(charge_credits).map_err(|e| AppError::Credential {
                message: format!("invalid charge amount: {e:?}"),
            })?;
        let (spend_proof, pre_refund) = credit_token
            .prove_spend::<128>(&params, charge_scalar, OsRng)
            .map_err(|e| AppError::Credential {
                message: format!("failed to create spend proof: {e:?}"),
            })?;

        let pre_refund_cbor = pre_refund.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode pre_refund: {e}"),
        })?;
        let spend_proof_cbor = spend_proof.to_cbor().map_err(|e| AppError::Credential {
            message: format!("failed to encode spend proof: {e}"),
        })?;
        let pre_cred_id = Uuid::now_v7().to_string();
        db::insert_pre_credential_refund(
            &db_conn,
            &pre_cred_id,
            &cred.nonce,
            &cred.issuer_key_id,
            &pre_refund_cbor,
            charge_credits as i64,
            &spend_proof_cbor,
            now,
        )
        .await?;

        let issuer_key_hash = hex_decode(&cred.issuer_key_id)?;
        let challenge_digest = compute_challenge_digest();

        let mut token_bytes = Vec::new();
        token_bytes.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
        token_bytes.extend_from_slice(&challenge_digest);
        token_bytes.extend_from_slice(&issuer_key_hash);
        token_bytes.extend_from_slice(&spend_proof_cbor);

        let token_b64 = URL_SAFE_NO_PAD.encode(&token_bytes);
        let auth_value = format!("PrivateToken token=\"{token_b64}\"");

        let last_action_id = db::last_action_in_space(&db_conn, &space_id).await?;

        let user_action_id = Uuid::now_v7().to_string();
        db::insert_action(
            &db_conn,
            &db::ActionEntry {
                id: user_action_id.clone(),
                space_id: space_id.clone(),
                participant_id: user_participant_id,
                action_type: "user_input".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: None,
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at: now,
            },
        )
        .await?;
        db::insert_text_content_block(
            &db_conn,
            &Uuid::now_v7().to_string(),
            &user_action_id,
            0,
            "text",
            prompt,
        )
        .await?;

        // Auto-title: the first exchange in an untitled space names the
        // space after the user's prompt. Purely local — no model call.
        if space_title.is_none()
            && prior_messages.is_empty()
            && let Some(title) = derive_space_title(prompt)
        {
            db::update_space_title(&db_conn, &space_id, &title).await?;
        }

        if let Some(ref ante_id) = last_action_id {
            db::insert_action_antecedent(&db_conn, &user_action_id, ante_id, 0).await?;
        }

        let mut messages: Vec<serde_json::Value> = prior_messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();
        messages.push(serde_json::json!({"role": "user", "content": prompt}));

        // No `stream_options` here — the server unconditionally sets
        // `include_usage: true` when forwarding the streaming request
        // upstream, since accurate per-token refunds depend on it.
        // Sending it from the client is harmless (the server ignores
        // and overrides the value), but it's also unnecessary, so we
        // keep our outgoing request minimal.
        let request_body_json = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "stream": true,
        });
        let request_at = now_ms();

        let chat_result = client
            .post(format!("{base_url}/v1/chat/completions"))
            .header("Authorization", &auth_value)
            .header("Accept", "text/event-stream")
            .json(&request_body_json)
            .send()
            .await;

        let resp = match chat_result {
            Ok(resp) => {
                if let Some(new_cid) =
                    flush_attestations(&attestation_log, &db_conn, &provider_id, base_url, now)
                        .await?
                {
                    connection_id = Some(new_cid);
                }
                resp
            }
            Err(e) => {
                let original_err = AppError::from_request(e);
                if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await {
                    let _ = process_refund(
                        &refund_obj,
                        &params,
                        &spend_proof,
                        &pre_refund,
                        &public_key,
                        &db_conn,
                        &pre_cred_id,
                        cred.generation + 1,
                        now,
                    )
                    .await;
                }
                return Err(original_err);
            }
        };

        let status = resp.status();

        // Non-2xx: server returned an error body (typically JSON, not SSE).
        // Read it normally so we can surface a useful message.
        if !status.is_success() {
            let response_text = resp.text().await.unwrap_or_default();
            if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await {
                let _ = process_refund(
                    &refund_obj,
                    &params,
                    &spend_proof,
                    &pre_refund,
                    &public_key,
                    &db_conn,
                    &pre_cred_id,
                    cred.generation + 1,
                    now,
                )
                .await;
            }
            db::insert_request(
                &db_conn,
                &db::Request {
                    id: Uuid::now_v7().to_string(),
                    connection_id,
                    action_id: None,
                    method: "POST".to_string(),
                    path: "/v1/chat/completions".to_string(),
                    request_headers: None,
                    request_body: Some(request_body_json.to_string().into_bytes()),
                    response_status: Some(status.as_u16() as i64),
                    response_headers: None,
                    response_body: Some(response_text.as_bytes().to_vec()),
                    request_at,
                    response_at: Some(now_ms()),
                    duration_ms: Some(now_ms() - request_at),
                    error: None,
                    credential_nonce: Some(cred.nonce.clone()),
                    created_at: now_ms(),
                },
            )
            .await?;
            return Err(AppError::Server {
                status: status.as_u16(),
                message: parse_server_error_message(&response_text),
            });
        }

        // Consume the SSE body. We accumulate bytes in a small buffer and
        // split on the SSE event boundary `\n\n`. Each event is a sequence
        // of `field: value\n` lines; we only care about `data:` lines (the
        // chunk JSON) and the sentinel `[DONE]`.
        let mut byte_stream = resp.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut full_content = String::new();
        let mut full_reasoning = String::new();
        let mut input_tokens: Option<i64> = None;
        let mut output_tokens: Option<i64> = None;
        let mut response_buf: Vec<u8> = Vec::new();
        let mut finished = false;

        while let Some(chunk) = byte_stream.next().await {
            let bytes = chunk.map_err(|e| AppError::Network {
                message: format!("stream read failed: {e}"),
            })?;
            // Keep the raw bytes for the request log so we can debug
            // upstream behaviour the same way as the non-streaming path.
            response_buf.extend_from_slice(&bytes);
            buf.extend_from_slice(&bytes);

            while let Some(pos) = find_event_boundary(&buf) {
                let event_bytes = buf.drain(..pos).collect::<Vec<u8>>();
                // Drop the boundary itself (\n\n or \r\n\r\n).
                let boundary_len = if buf.starts_with(b"\r\n\r\n") { 4 } else { 2 };
                if buf.len() >= boundary_len {
                    buf.drain(..boundary_len);
                }
                let event_str = match std::str::from_utf8(&event_bytes) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                for line in event_str.lines() {
                    let line = line.trim_end_matches('\r');
                    let Some(payload) = line.strip_prefix("data:") else {
                        continue;
                    };
                    let payload = payload.trim_start();
                    if payload == "[DONE]" {
                        finished = true;
                        continue;
                    }
                    let json: serde_json::Value = match serde_json::from_str(payload) {
                        Ok(v) => v,
                        Err(_) => continue, // ignore comments/heartbeats
                    };

                    if let Some(usage) = json.get("usage") {
                        if let Some(v) = usage.get("prompt_tokens").and_then(|v| v.as_i64()) {
                            input_tokens = Some(v);
                        }
                        if let Some(v) = usage.get("completion_tokens").and_then(|v| v.as_i64()) {
                            output_tokens = Some(v);
                        }
                    }

                    let Some(delta) = json
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|c| c.get("delta"))
                    else {
                        continue;
                    };

                    if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                        && !text.is_empty()
                    {
                        full_content.push_str(text);
                        let _ = sender.send(ChatStreamEvent::ContentDelta(text.to_string()));
                    }

                    // OpenAI o1-style ("reasoning_content") and vLLM-style
                    // ("reasoning"). Either form is forwarded as a
                    // ReasoningDelta; we tolerate providers that emit one,
                    // both, or neither.
                    for key in ["reasoning_content", "reasoning"] {
                        if let Some(text) = delta.get(key).and_then(|v| v.as_str())
                            && !text.is_empty()
                        {
                            full_reasoning.push_str(text);
                            let _ = sender.send(ChatStreamEvent::ReasoningDelta(text.to_string()));
                        }
                    }
                }
            }

            if finished {
                break;
            }
        }
        let response_at = now_ms();

        if let Ok(refund_obj) = recover_refund(&client, base_url, &auth_value).await {
            let _ = process_refund(
                &refund_obj,
                &params,
                &spend_proof,
                &pre_refund,
                &public_key,
                &db_conn,
                &pre_cred_id,
                cred.generation + 1,
                now,
            )
            .await;
        }

        let inference_action_id = Uuid::now_v7().to_string();
        db::insert_action(
            &db_conn,
            &db::ActionEntry {
                id: inference_action_id.clone(),
                space_id: space_id.clone(),
                participant_id: model_participant_id,
                action_type: "inference".to_string(),
                status: "complete".to_string(),
                intent: None,
                model: Some(model.to_string()),
                input_tokens,
                output_tokens,
                credits_consumed: Some(charge_credits as i64),
                created_at: now_ms(),
            },
        )
        .await?;
        db::insert_action_antecedent(&db_conn, &inference_action_id, &user_action_id, 0).await?;

        let context_assembly_id = Uuid::now_v7().to_string();
        db::insert_context_assembly(
            &db_conn,
            &context_assembly_id,
            &inference_action_id,
            None,
            input_tokens,
            false,
            now_ms(),
        )
        .await?;

        let prior_action_ids = db::space_action_ids(&db_conn, &space_id).await?;
        for (pos, aid) in prior_action_ids.iter().enumerate() {
            if aid != &inference_action_id {
                db::insert_context_assembly_action(&db_conn, &context_assembly_id, aid, pos as i64)
                    .await?;
            }
        }

        if !full_content.is_empty() {
            db::insert_text_content_block(
                &db_conn,
                &Uuid::now_v7().to_string(),
                &inference_action_id,
                0,
                "text",
                &full_content,
            )
            .await?;
        }

        db::insert_request(
            &db_conn,
            &db::Request {
                id: Uuid::now_v7().to_string(),
                connection_id,
                action_id: Some(inference_action_id),
                method: "POST".to_string(),
                path: "/v1/chat/completions".to_string(),
                request_headers: None,
                request_body: Some(request_body_json.to_string().into_bytes()),
                response_status: Some(status.as_u16() as i64),
                response_headers: None,
                response_body: Some(response_buf),
                request_at,
                response_at: Some(response_at),
                duration_ms: Some(response_at - request_at),
                error: None,
                credential_nonce: Some(cred.nonce.clone()),
                created_at: now_ms(),
            },
        )
        .await?;

        Ok(ChatResult {
            space_id,
            content: full_content,
            model: model.to_string(),
            input_tokens,
            output_tokens,
            credits_charged: charge_credits as i64,
        })
    }
}

/// Find the byte offset of the next SSE event boundary (`\n\n` or
/// `\r\n\r\n`) in `buf`, if any. Returns the position *before* the boundary
/// — i.e. the length of the next event's body.
fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    for i in 0..buf.len() {
        if buf[i..].starts_with(b"\r\n\r\n") {
            return Some(i);
        }
        if buf[i..].starts_with(b"\n\n") {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// AppCore — owns the tokio runtime that drives all async work (turso,
// reqwest, tokio primitives). Consumers (CLI, GUI) hold an `Arc<AppCore>`
// and call methods directly.
// ============================================================================

pub struct AppCore {
    runtime: tokio::runtime::Runtime,
    inner: Arc<Inner>,
}

impl AppCore {
    pub fn runtime(&self) -> &tokio::runtime::Runtime {
        &self.runtime
    }

    /// Streaming chat. Pushes incremental `ChatStreamEvent`s through
    /// `sender` and returns the finalized `ChatResult` when the upstream
    /// stream closes. Drops `sender` on return so receivers see channel
    /// closure as the natural "done" signal.
    pub async fn chat_stream(
        &self,
        prompt: String,
        model: String,
        space_id: Option<String>,
        sender: tokio::sync::mpsc::UnboundedSender<ChatStreamEvent>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move {
                inner
                    .chat_stream(&prompt, &model, space_id.as_deref(), sender)
                    .await
            })
            .await
            .map_err(join_err)?
    }
}

/// Convert a `tokio::task::JoinError` (panic / cancellation) into `AppError`.
fn join_err(e: tokio::task::JoinError) -> AppError {
    AppError::Internal {
        message: format!("async task failed: {e}"),
    }
}

impl AppCore {
    /// Create a new core instance.
    ///
    /// `config_dir` — directory containing `config.toml`.
    /// `data_dir` — directory for the local database.
    pub fn new(config_dir: PathBuf, data_dir: PathBuf) -> Self {
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(8 * 1024 * 1024) // 8 MB — matches default main-thread size
            .build()
            .expect("failed to create tokio runtime");
        Self {
            runtime,
            inner: Arc::new(Inner {
                config_path: config_dir.join("config.toml"),
                data_dir,
                db: tokio::sync::OnceCell::new(),
            }),
        }
    }

    // -----------------------------------------------------------------------
    // Config — sync methods (no runtime needed, delegate directly)
    // -----------------------------------------------------------------------

    pub fn config_state(&self) -> ConfigState {
        let cfg = self.inner.load_config();
        let to_info = |m: &tinfoil_verifier::EnclaveMeasurement| MeasurementInfo {
            snp: m.snp_measurement.clone(),
            tdx_rtmr1: m.tdx_measurement.rtmr1.clone(),
            tdx_rtmr2: m.tdx_measurement.rtmr2.clone(),
        };
        ConfigState {
            base_url: cfg.base_url().to_string(),
            default_model: cfg.default_model().to_string(),
            has_account: cfg.account_id.is_some(),
            has_account_secret: cfg.account_secret.is_some(),
            domain_separator: cfg.domain_separator().to_string(),
            trusted_measurements: cfg.trusted_measurements().iter().map(&to_info).collect(),
            has_hardware_root_ca: cfg.hardware_root_ca.is_some(),
            has_hardware_intermediate_ca: cfg.hardware_intermediate_ca.is_some(),
            attestation_url: cfg.attestation_url.clone(),
        }
    }

    pub fn set_base_url(&self, url: String) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.base_url_override = Some(url);
        cfg.save_to(&self.inner.config_path)
    }

    /// Persist the user's default inference model (the `default_model`
    /// config override). New chat surfaces resolve their starting model
    /// from this; an existing window's explicit selection is unaffected.
    pub fn set_default_model(&self, model: String) -> Result<(), AppError> {
        let model = model.trim().to_string();
        if model.is_empty() {
            return Err(AppError::Config {
                message: "default model must not be empty".into(),
            });
        }
        let mut cfg = self.inner.load_config();
        cfg.default_model_override = Some(model);
        cfg.save_to(&self.inner.config_path)
    }

    pub fn set_attestation_url(&self, url: String) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.attestation_url = Some(url);
        cfg.save_to(&self.inner.config_path)
    }

    pub fn set_hardware_root_ca(&self, pem: String) -> Result<(), AppError> {
        config::parse_cert_config(Some(&pem), "hardware_root_ca")?;
        let mut cfg = self.inner.load_config();
        cfg.hardware_root_ca = Some(pem.trim().to_string());
        cfg.save_to(&self.inner.config_path)
    }

    pub fn set_hardware_intermediate_ca(&self, pem: String) -> Result<(), AppError> {
        config::parse_cert_config(Some(&pem), "hardware_intermediate_ca")?;
        let mut cfg = self.inner.load_config();
        cfg.hardware_intermediate_ca = Some(pem.trim().to_string());
        cfg.save_to(&self.inner.config_path)
    }

    pub fn trust_measurement(
        &self,
        snp: String,
        tdx_rtmr1: String,
        tdx_rtmr2: String,
    ) -> Result<bool, AppError> {
        let spec = format!("{snp}:{tdx_rtmr1}:{tdx_rtmr2}");
        let entry = config::parse_trust_measurement(&spec)?;
        let mut cfg = self.inner.load_config();
        if cfg.trusted_measurements_override.iter().any(|m| {
            m.snp_measurement
                .eq_ignore_ascii_case(&entry.snp_measurement)
        }) {
            return Ok(false);
        }
        cfg.trusted_measurements_override.push(entry);
        cfg.save_to(&self.inner.config_path)?;
        Ok(true)
    }

    pub fn untrust_measurement(&self, snp: String) -> Result<bool, AppError> {
        let key = config::parse_untrust_key(&snp)?;
        let mut cfg = self.inner.load_config();
        if let Some(pos) = cfg
            .trusted_measurements_override
            .iter()
            .position(|m| m.snp_measurement.eq_ignore_ascii_case(&key))
        {
            cfg.trusted_measurements_override.remove(pos);
            cfg.save_to(&self.inner.config_path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn set_account_credentials(&self, id: String, secret: String) -> Result<(), AppError> {
        let cfg = self.inner.load_config();
        if cfg.account_id.is_some() || cfg.account_secret.is_some() {
            return Err(AppError::Config {
                message: "account credentials already configured — reset first".into(),
            });
        }
        let mut cfg = cfg;
        cfg.account_id = Some(id);
        cfg.account_secret = Some(secret);
        cfg.save_to(&self.inner.config_path)
    }

    pub fn reset_account(&self) -> Result<(), AppError> {
        let mut cfg = self.inner.load_config();
        cfg.account_id = None;
        cfg.account_secret = None;
        cfg.save_to(&self.inner.config_path)
    }

    // -----------------------------------------------------------------------
    // Async methods — spawn onto the owned tokio runtime
    // -----------------------------------------------------------------------

    pub async fn account_show(&self) -> Result<AccountShowResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_show().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_create(&self) -> Result<AccountCreateResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_create().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_prices(&self) -> Result<Vec<PriceInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_prices().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_checkout(&self, price_id: String) -> Result<String, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_checkout(&price_id).await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_balances(&self) -> Result<BalancesResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_balances().await })
            .await
            .map_err(join_err)?
    }

    pub async fn account_allocate(&self, credits: i64) -> Result<AllocateResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.account_allocate(credits).await })
            .await
            .map_err(join_err)?
    }

    pub async fn wallet_credentials(&self) -> Result<Vec<CredentialInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.wallet_credentials().await })
            .await
            .map_err(join_err)?
    }

    pub async fn wallet_spending_credentials(
        &self,
    ) -> Result<Vec<InFlightCredentialInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.wallet_spending_credentials().await })
            .await
            .map_err(join_err)?
    }

    pub async fn recover_spending_credentials(&self) -> Result<Vec<String>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.recover_spending_credentials().await })
            .await
            .map_err(join_err)?
    }

    pub async fn available_models(&self) -> Result<Vec<ModelInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.available_models().await })
            .await
            .map_err(join_err)?
    }

    /// List spaces, most recently active first. Archived spaces are
    /// excluded unless `include_archived` is set.
    pub async fn list_spaces(&self, include_archived: bool) -> Result<Vec<SpaceInfo>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.list_spaces(include_archived).await })
            .await
            .map_err(join_err)?
    }

    pub async fn get_space_messages(
        &self,
        space_id: String,
    ) -> Result<Vec<SpaceMessage>, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.get_space_messages(&space_id).await })
            .await
            .map_err(join_err)?
    }

    pub async fn create_space(&self, title: Option<String>) -> Result<SpaceInfo, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.create_space(title.as_deref()).await })
            .await
            .map_err(join_err)?
    }

    pub async fn archive_space(&self, space_id: String) -> Result<bool, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.archive_space(&space_id).await })
            .await
            .map_err(join_err)?
    }

    pub async fn rename_space(&self, space_id: String, title: String) -> Result<(), AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.rename_space(&space_id, &title).await })
            .await
            .map_err(join_err)?
    }

    pub async fn chat(
        &self,
        prompt: String,
        model: String,
        space_id: Option<String>,
    ) -> Result<ChatResult, AppError> {
        let inner = self.inner.clone();
        self.runtime
            .spawn(async move { inner.chat(&prompt, &model, space_id.as_deref()).await })
            .await
            .map_err(join_err)?
    }
}

// ============================================================================
// Internal API response types
// ============================================================================

#[derive(Deserialize)]
struct CreateAccountResponse {
    account_id: Uuid,
    secret: String,
    created_at: String,
}

#[derive(Deserialize)]
struct GetAccountResponse {
    id: Uuid,
    stripe_customer_id: Option<String>,
    created_at: String,
}

#[derive(Deserialize)]
struct ListPricesResponse {
    data: Vec<PriceResponse>,
}

#[derive(Deserialize)]
struct PriceResponse {
    id: String,
    product_name: String,
    product_description: Option<String>,
    unit_amount: Option<i64>,
    currency: String,
    #[allow(dead_code)]
    #[serde(rename = "type")]
    price_type: String,
    recurring: Option<RecurringResponse>,
    credits: i64,
}

#[derive(Deserialize)]
struct RecurringResponse {
    interval: String,
    interval_count: i64,
}

#[derive(Deserialize)]
struct CheckoutUrlResponse {
    checkout_url: String,
}

#[derive(Deserialize)]
struct BalancesResponse {
    available: i64,
    pools: Vec<BalancePoolResponse>,
}

#[derive(Deserialize)]
struct BalancePoolResponse {
    amount: i64,
    source: String,
    expires_at: Option<String>,
}

#[derive(Deserialize)]
struct ListKeysResponse {
    data: Vec<IssuerKeyResponse>,
}

#[derive(Deserialize)]
struct IssuerKeyResponse {
    id: String,
    public_key: String,
    domain_separator: String,
    #[allow(dead_code)]
    issue_from: String,
    issue_until: String,
    #[allow(dead_code)]
    accept_until: String,
}

#[derive(Deserialize)]
struct IssueCredentialsResponse {
    issuance_response: String,
    issuer_key_id: String,
    credits: i64,
    #[allow(dead_code)]
    ledger_entry_id: String,
}

#[derive(Deserialize)]
struct ModelPricingInfo {
    per_prompt_token: ScaledPriceInfo,
    per_completion_token: ScaledPriceInfo,
    /// Present only for models priced per request (e.g. transcription).
    #[serde(default)]
    per_request: Option<ScaledPriceInfo>,
}

#[derive(Deserialize)]
struct ScaledPriceInfo {
    value: u64,
    scale_factor: u64,
}

impl ScaledPriceInfo {
    /// Credits per priced unit (token or request) as a float for display.
    /// Charging math elsewhere stays in integer `value`/`scale_factor`
    /// space; this is only the honest human-readable rate.
    fn credits_per_unit(&self) -> f64 {
        if self.scale_factor == 0 {
            0.0
        } else {
            self.value as f64 / self.scale_factor as f64
        }
    }
}

#[derive(Deserialize)]
struct ModelsResponseInfo {
    data: Vec<ModelListEntry>,
}

#[derive(Deserialize)]
struct ModelListEntry {
    id: String,
    context_length: u64,
    pricing: ModelPricingInfo,
}

// ============================================================================
// Refund processing helpers
// ============================================================================

/// Extract a refund token from a JSON object and store the resulting credential.
///
/// `refund_obj` is the `"refund"` value from a server response (either a chat
/// completion or the recovery endpoint). Returns `true` if the credential was
/// successfully stored.
#[allow(clippy::too_many_arguments)]
async fn process_refund(
    refund_obj: &serde_json::Value,
    params: &Params,
    spend_proof: &SpendProof<128>,
    pre_refund: &PreRefund,
    public_key: &PublicKey,
    db_conn: &turso::Connection,
    pre_cred_id: &str,
    generation: i64,
    now: i64,
) -> Result<(), AppError> {
    let refund_b64 = refund_obj
        .get("refund")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::Credential {
            message: "missing refund data in response".into(),
        })?;
    let refund_key_id = refund_obj
        .get("issuer_key_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| AppError::Credential {
            message: "missing issuer_key_id in refund".into(),
        })?;

    let refund_cbor = URL_SAFE_NO_PAD
        .decode(refund_b64)
        .map_err(|e| AppError::Credential {
            message: format!("invalid refund base64: {e}"),
        })?;
    let refund = Refund::from_cbor(&refund_cbor).map_err(|e| AppError::Credential {
        message: format!("invalid refund CBOR: {e}"),
    })?;

    let new_token = pre_refund
        .to_credit_token::<128>(params, spend_proof, &refund, public_key)
        .map_err(|e| AppError::Credential {
            message: format!("failed to construct refund credit token: {e:?}"),
        })?;

    let new_token_cbor = new_token.to_cbor().map_err(|e| AppError::Credential {
        message: format!("failed to encode new credit token: {e}"),
    })?;
    let new_nonce = hex_encode(&new_token.nullifier().to_bytes());
    let new_credits =
        scalar_to_credit::<128>(&new_token.credits()).map_err(|e| AppError::Credential {
            message: format!("invalid credit amount in refund token: {e}"),
        })?;

    db::insert_credential(
        db_conn,
        &new_nonce,
        pre_cred_id,
        refund_key_id,
        &new_token_cbor,
        new_credits as i64,
        generation,
        now,
    )
    .await?;

    Ok(())
}

/// Attempt to recover a refund token from the server via
/// `POST /v1/credentials/refund`. Returns the refund JSON object on success.
async fn recover_refund(
    client: &reqwest::Client,
    base_url: &str,
    auth_value: &str,
) -> Result<serde_json::Value, AppError> {
    let resp = client
        .post(format!("{base_url}/v1/credentials/refund"))
        .header("Authorization", auth_value)
        .send()
        .await
        .map_err(AppError::from_request)?;

    let status = resp.status();
    let body_text = resp.text().await.map_err(|e| AppError::Network {
        message: format!("failed to read recovery response: {e}"),
    })?;
    let body: serde_json::Value =
        serde_json::from_str(&body_text).map_err(|e| AppError::Network {
            message: format!("failed to parse recovery response: {e}"),
        })?;

    if !status.is_success() {
        return Err(AppError::Server {
            status: status.as_u16(),
            message: format!(
                "refund recovery failed: {}",
                parse_server_error_message(&body_text)
            ),
        });
    }

    body.get("refund")
        .cloned()
        .ok_or_else(|| AppError::Credential {
            message: "recovery response missing refund field".into(),
        })
}

// ============================================================================
// Free-standing helpers
// ============================================================================

/// Convert space action rows into a sequence of role/content messages suitable
/// for the OpenAI messages array and for UI display. Groups content blocks by
/// action and concatenates text.
fn actions_to_messages(action_rows: &[db::SpaceActionRow]) -> Vec<SpaceMessage> {
    let mut messages: Vec<SpaceMessage> = Vec::new();
    let mut current_action_id: Option<&str> = None;

    for row in action_rows {
        let role = match (row.action_type.as_str(), row.participant_kind.as_str()) {
            ("user_input", _) => "user",
            ("inference", _) => "assistant",
            _ => continue, // skip tool_call, tool_result, etc. for now
        };

        if current_action_id == Some(row.action_id.as_str()) {
            // Additional content block for the same action — append text
            if let Some(text) = &row.text_content
                && let Some(last) = messages.last_mut()
            {
                last.content.push_str(text);
            }
        } else {
            // New action
            current_action_id = Some(&row.action_id);
            let content = row.text_content.clone().unwrap_or_default();
            messages.push(SpaceMessage {
                role: role.to_string(),
                content,
            });
        }
    }

    messages
}

/// Maximum length of an auto-derived space title, in characters.
const TITLE_MAX_CHARS: usize = 64;

/// Maximum length of a listing snippet, in characters.
const SNIPPET_MAX_CHARS: usize = 120;

/// Derive a space title from the user's first prompt: take the first
/// non-empty line, strip leading markdown block markers (headings, list
/// bullets, blockquotes, numbered lists) and surrounding emphasis
/// characters, then truncate to ~64 chars on a word boundary (appending an
/// ellipsis when truncated). Returns `None` if nothing presentable is left.
fn derive_space_title(prompt: &str) -> Option<String> {
    let line = prompt.lines().map(str::trim).find(|l| !l.is_empty())?;

    // Strip leading block markers, repeatedly — "> # Heading" etc.
    let mut s = line;
    loop {
        let mut t = s.trim_start_matches(['#', '>']).trim_start();
        // Unordered list bullets.
        for marker in ["- ", "* ", "+ "] {
            if let Some(rest) = t.strip_prefix(marker) {
                t = rest.trim_start();
            }
        }
        // Ordered list markers like "1. " / "12) ".
        let digits = t.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 {
            let after = &t[digits..];
            if let Some(rest) = after
                .strip_prefix(". ")
                .or_else(|| after.strip_prefix(") "))
            {
                t = rest.trim_start();
            }
        }
        if t == s {
            break;
        }
        s = t;
    }

    // Strip emphasis/code markers from the edges ("**Bold ask**", "`code`").
    let s = s.trim_matches(['*', '_', '`']).trim();
    if s.is_empty() {
        return None;
    }

    Some(truncate_on_word_boundary(s, TITLE_MAX_CHARS))
}

/// Snippet for the space listing: first line-collapsed ~120 chars of the
/// given text, truncated on a word boundary. Returns `None` for
/// whitespace-only input.
fn snippet_of(text: &str) -> Option<String> {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    Some(truncate_on_word_boundary(&collapsed, SNIPPET_MAX_CHARS))
}

/// Truncate `s` to at most `max_chars` characters, breaking on a word
/// boundary where possible and appending `…` when anything was cut. The
/// ellipsis is not counted against the budget; `max_chars` must be ≥ 1.
fn truncate_on_word_boundary(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out = String::new();
    let mut count = 0usize;
    for word in s.split_whitespace() {
        let word_chars = word.chars().count();
        let sep = usize::from(!out.is_empty());
        if count + sep + word_chars > max_chars {
            break;
        }
        if sep == 1 {
            out.push(' ');
        }
        out.push_str(word);
        count += sep + word_chars;
    }
    if out.is_empty() {
        // Single word longer than the budget — hard-cut it.
        out = s.chars().take(max_chars).collect();
    }
    out.push('…');
    out
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, AppError> {
    if !s.len().is_multiple_of(2) {
        return Err(AppError::Credential {
            message: "odd-length hex string".into(),
        });
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| AppError::Credential {
                message: format!("invalid hex: {e}"),
            })
        })
        .collect()
}

const ACT_TOKEN_TYPE: u16 = 0xE5AD;
const ISSUER_NAME: &str = "eidola";
const ORIGIN_INFO: &str = "inference";

fn serialize_token_challenge() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
    buf.extend_from_slice(&(ISSUER_NAME.len() as u16).to_be_bytes());
    buf.extend_from_slice(ISSUER_NAME.as_bytes());
    buf.push(0);
    buf.extend_from_slice(&(ORIGIN_INFO.len() as u16).to_be_bytes());
    buf.extend_from_slice(ORIGIN_INFO.as_bytes());
    buf.push(0);
    buf
}

fn compute_challenge_digest() -> [u8; 32] {
    Sha256::digest(serialize_token_challenge()).into()
}

/// Current time as milliseconds since Unix epoch.
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_millis() as i64
}

/// Parse an ISO 8601 timestamp to epoch ms.
pub fn iso_to_ms(s: &str) -> Result<i64, AppError> {
    let s = s.trim().trim_end_matches('Z');
    let (date, time) = s.split_once('T').ok_or_else(|| AppError::Network {
        message: format!("invalid ISO 8601: {s}"),
    })?;
    let dp: Vec<&str> = date.split('-').collect();
    let tp: Vec<&str> = time.split(':').collect();
    if dp.len() != 3 || tp.len() != 3 {
        return Err(AppError::Network {
            message: format!("invalid ISO 8601: {s}"),
        });
    }
    let y: i64 = dp[0].parse().map_err(|_| AppError::Network {
        message: "bad year".into(),
    })?;
    let m: u32 = dp[1].parse().map_err(|_| AppError::Network {
        message: "bad month".into(),
    })?;
    let d: u32 = dp[2].parse().map_err(|_| AppError::Network {
        message: "bad day".into(),
    })?;
    let hour: i64 = tp[0].parse().map_err(|_| AppError::Network {
        message: "bad hour".into(),
    })?;
    let min: i64 = tp[1].parse().map_err(|_| AppError::Network {
        message: "bad minute".into(),
    })?;
    let sec: i64 = tp[2]
        .split('.')
        .next()
        .unwrap_or("0")
        .parse()
        .map_err(|_| AppError::Network {
            message: "bad second".into(),
        })?;
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj.rem_euclid(400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i64 - 719468;
    let secs = days * 86400 + hour * 3600 + min * 60 + sec;
    Ok(secs * 1000)
}

pub(crate) fn load_native_root_store() -> rustls::RootCertStore {
    let mut store = rustls::RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    for cert in native.certs {
        let _ = store.add(cert);
    }
    store
}

fn params_from_domain_separator(ds: &str) -> Result<Params, AppError> {
    let parts: Vec<&str> = ds.split(':').collect();
    if parts.len() != 5 {
        return Err(AppError::Config {
            message: format!("domain separator has wrong format: {ds}"),
        });
    }
    Ok(Params::new(parts[1], parts[2], parts[3], parts[4]))
}

async fn read_response(resp: reqwest::Response) -> Result<(reqwest::StatusCode, String), AppError> {
    let status = resp.status();
    let body = resp.text().await.map_err(|e| AppError::Network {
        message: format!("failed to read response body: {e}"),
    })?;
    Ok((status, body))
}

fn check_status(status: reqwest::StatusCode, body: &str) -> Result<(), AppError> {
    if status.is_success() {
        return Ok(());
    }
    Err(AppError::Server {
        status: status.as_u16(),
        message: parse_server_error_message(body),
    })
}

/// Best-effort extraction of a human-readable error message from a
/// non-2xx response body. Tries the OpenAI-shaped `{"error":{"message":"..."}}`
/// envelope first; falls back to the raw body text (trimmed and
/// length-capped — axum's body-extractor rejection bodies are plain
/// text, not JSON, and were previously bucketed to "unknown error" by
/// the old JSON-only path); finally falls back to a literal "unknown
/// error" only when the body is empty.
pub(crate) fn parse_server_error_message(body: &str) -> String {
    if let Some(msg) = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(String::from)
        })
    {
        return msg;
    }
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "unknown error".to_string();
    }
    // Cap the message so a chatty rejection body doesn't blow up the UI.
    const MAX_LEN: usize = 500;
    if trimmed.len() > MAX_LEN {
        let mut capped: String = trimmed.chars().take(MAX_LEN).collect();
        capped.push('…');
        capped
    } else {
        trimmed.to_string()
    }
}

async fn fetch_models(
    client: &reqwest::Client,
    base_url: &str,
) -> Result<ModelsResponseInfo, AppError> {
    let resp = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .map_err(AppError::from_request)?;
    let (status, body) = read_response(resp).await?;
    check_status(status, &body)?;
    serde_json::from_str(&body).map_err(|e| AppError::Network {
        message: format!("failed to parse models response: {e}"),
    })
}

async fn flush_attestations(
    attestation_log: &Mutex<Vec<tinfoil_verifier::VerifiedAttestation>>,
    db_conn: &turso::Connection,
    provider_id: &str,
    base_url: &str,
    now: i64,
) -> Result<Option<String>, AppError> {
    let attestations: Vec<_> = attestation_log.lock().unwrap().drain(..).collect();
    let mut connection_id = None;
    for att in &attestations {
        db::upsert_attestation(
            db_conn,
            &att.attestation_hash,
            &att.attestation_doc,
            Some(&att.pcr_digest),
            now,
        )
        .await?;
        let cid = Uuid::now_v7().to_string();
        db::insert_connection(
            db_conn,
            &cid,
            provider_id,
            base_url,
            "clearnet",
            Some(&att.attestation_hash),
            now,
            now,
        )
        .await?;
        connection_id = Some(cid);
    }
    Ok(connection_id)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trip() {
        let data = vec![0xde, 0xad, 0xbe, 0xef];
        let encoded = hex_encode(&data);
        assert_eq!(encoded, "deadbeef");
        assert_eq!(hex_decode(&encoded).unwrap(), data);
    }

    #[test]
    fn hex_decode_rejects_odd_length() {
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn challenge_digest_is_deterministic() {
        let d1 = compute_challenge_digest();
        let d2 = compute_challenge_digest();
        assert_eq!(d1, d2);
    }

    #[test]
    fn iso_to_ms_basic() {
        // 2026-01-01T00:00:00Z → 1767225600000
        let ms = iso_to_ms("2026-01-01T00:00:00Z").unwrap();
        assert_eq!(ms, 1767225600000);
    }

    #[test]
    fn token_challenge_serialization() {
        let buf = serialize_token_challenge();
        assert_eq!(u16::from_be_bytes([buf[0], buf[1]]), ACT_TOKEN_TYPE);
    }

    #[test]
    fn params_from_domain_separator_valid() {
        let p = params_from_domain_separator("ACT-v1:org:service:deploy:ver");
        assert!(p.is_ok());
    }

    #[test]
    fn params_from_domain_separator_rejects_wrong_format() {
        assert!(params_from_domain_separator("bad").is_err());
    }

    // --- Default model config ----------------------------------------------

    #[test]
    fn set_default_model_round_trips_through_config_state() {
        let dir = tempfile::tempdir().unwrap();
        let config_dir = dir.path().to_path_buf();
        let data_dir = dir.path().join("data");

        let core = AppCore::new(config_dir.clone(), data_dir.clone());
        assert_eq!(core.config_state().default_model, config::DEFAULT_MODEL);

        core.set_default_model("kimi-k2-6".into()).unwrap();
        assert_eq!(core.config_state().default_model, "kimi-k2-6");

        // A fresh core over the same config dir sees the persisted value.
        let core2 = AppCore::new(config_dir, data_dir);
        assert_eq!(core2.config_state().default_model, "kimi-k2-6");

        // Whitespace-only is rejected and leaves the config untouched.
        assert!(core2.set_default_model("   ".into()).is_err());
        assert_eq!(core2.config_state().default_model, "kimi-k2-6");
    }

    // --- Auto-provisioning decision logic ---------------------------------

    #[test]
    fn auto_allocation_uses_default_chunk_when_balance_is_plentiful() {
        // Plenty of balance, small charge → allocate the default chunk,
        // leaving the rest of the balance in the account.
        let amount = auto_allocation_amount(10 * DEFAULT_ALLOCATION_CREDITS, 6_200).unwrap();
        assert_eq!(amount, DEFAULT_ALLOCATION_CREDITS);
    }

    #[test]
    fn auto_allocation_is_capped_by_available_balance() {
        // Balance covers the charge but is below the default chunk →
        // allocate everything that's available.
        let amount = auto_allocation_amount(50_000, 6_200).unwrap();
        assert_eq!(amount, 50_000);
    }

    #[test]
    fn auto_allocation_grows_to_cover_a_large_charge() {
        // A single charge larger than the default chunk must still fit in
        // one credential (spends draw from exactly one credential).
        let required = DEFAULT_ALLOCATION_CREDITS + 500_000;
        let amount = auto_allocation_amount(10 * DEFAULT_ALLOCATION_CREDITS, required).unwrap();
        assert_eq!(amount, required);
    }

    #[test]
    fn auto_allocation_exact_balance_allocates_it_all() {
        let amount = auto_allocation_amount(6_200, 6_200).unwrap();
        assert_eq!(amount, 6_200);
    }

    #[test]
    fn auto_allocation_fails_typed_when_balance_cannot_cover_charge() {
        let err = auto_allocation_amount(1_000, 6_200).unwrap_err();
        match err {
            AppError::InsufficientBalance {
                available,
                required,
            } => {
                assert_eq!(available, 1_000);
                assert_eq!(required, 6_200);
            }
            other => panic!("expected InsufficientBalance, got {other:?}"),
        }
    }

    #[test]
    fn auto_allocation_fails_typed_on_zero_balance() {
        assert!(matches!(
            auto_allocation_amount(0, 1),
            Err(AppError::InsufficientBalance {
                available: 0,
                required: 1
            })
        ));
    }

    #[test]
    fn derive_title_takes_first_line() {
        assert_eq!(
            derive_space_title("How do tides work?\n\nAnd a second question."),
            Some("How do tides work?".to_string())
        );
    }

    #[test]
    fn derive_title_skips_leading_blank_lines() {
        assert_eq!(
            derive_space_title("\n\n  \nActual question"),
            Some("Actual question".to_string())
        );
    }

    #[test]
    fn derive_title_strips_markdown_markers() {
        assert_eq!(
            derive_space_title("## A heading prompt"),
            Some("A heading prompt".to_string())
        );
        assert_eq!(
            derive_space_title("- a list item"),
            Some("a list item".to_string())
        );
        assert_eq!(
            derive_space_title("> # quoted heading"),
            Some("quoted heading".to_string())
        );
        assert_eq!(
            derive_space_title("1. first thing"),
            Some("first thing".to_string())
        );
        assert_eq!(
            derive_space_title("**Bold ask**"),
            Some("Bold ask".to_string())
        );
    }

    #[test]
    fn derive_title_truncates_on_word_boundary() {
        let long = "Please explain in detail how the borrow checker reasons about \
                    lifetimes when closures capture references";
        let title = derive_space_title(long).unwrap();
        assert!(title.ends_with('…'));
        assert!(title.trim_end_matches('…').chars().count() <= 64);
        // Word-boundary: must not end mid-word.
        assert!(long.starts_with(title.trim_end_matches('…')));
        assert!(
            title
                .trim_end_matches('…')
                .ends_with(|c: char| !c.is_whitespace())
        );
        let kept = title.trim_end_matches('…');
        assert!(
            long[kept.len()..].starts_with(' '),
            "cut mid-word: {title:?}"
        );
    }

    #[test]
    fn derive_title_rejects_empty_and_marker_only() {
        assert_eq!(derive_space_title(""), None);
        assert_eq!(derive_space_title("   \n  "), None);
        assert_eq!(derive_space_title("###"), None);
    }

    #[test]
    fn derive_title_hard_cuts_single_long_word() {
        let word = "a".repeat(100);
        let title = derive_space_title(&word).unwrap();
        assert_eq!(title.chars().count(), 65); // 64 + ellipsis
        assert!(title.ends_with('…'));
    }

    #[test]
    fn snippet_collapses_whitespace_and_truncates() {
        assert_eq!(
            snippet_of("first line\nsecond   line"),
            Some("first line second line".to_string())
        );
        assert_eq!(snippet_of("  \n \t "), None);
        let long = "word ".repeat(60);
        let snippet = snippet_of(&long).unwrap();
        assert!(snippet.ends_with('…'));
        assert!(snippet.trim_end_matches('…').chars().count() <= 120);
    }
}
