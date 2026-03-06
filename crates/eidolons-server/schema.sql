-- Eidólons Billing Schema
-- Targeting PostgreSQL 16+
--
-- This schema covers the "identified" side of the system: accounts, payments,
-- credit balances, and credential provisioning records. It also includes issuer
-- key management and nullifier storage, which serve both the identified and
-- anonymous contexts but require durable persistence.

BEGIN;

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()

-- ---------------------------------------------------------------------------
-- Account
-- ---------------------------------------------------------------------------

CREATE TABLE account (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    secret_hash         TEXT NOT NULL,
    stripe_customer_id  TEXT UNIQUE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE account IS
    'A lightweight account record. No PII is stored here; identity is a '
    'bearer credential (id + secret). The Stripe Customer ID is the only '
    'link to payment identity. Subscription state is denormalized from Stripe '
    'webhooks for fast reads; Stripe remains the source of truth for billing.';

COMMENT ON COLUMN account.id IS
    'Public account identifier, exposed in the API as account_id. '
    'Safe to include in logs.';

COMMENT ON COLUMN account.secret_hash IS
    'Argon2id hash of the account secret. The plaintext is returned '
    'exactly once at account creation and never stored.';

COMMENT ON COLUMN account.stripe_customer_id IS
    'Set when the account first interacts with Stripe (subscription or top-up). '
    'NULL for accounts that have never made a payment. UNIQUE because one Stripe '
    'customer should map to exactly one account.';

CREATE INDEX idx_account_stripe_customer ON account (stripe_customer_id)
    WHERE stripe_customer_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Issuer Key
-- ---------------------------------------------------------------------------

CREATE TABLE issuer_key (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    private_key_enc BYTEA NOT NULL,
    public_key      BYTEA NOT NULL,
    domain_separator TEXT NOT NULL,
    issue_from      TIMESTAMPTZ NOT NULL,
    issue_until     TIMESTAMPTZ NOT NULL,
    accept_until    TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT issue_window CHECK (issue_from < issue_until),
    CONSTRAINT grace_window CHECK (issue_until <= accept_until)
);

-- Only one key per issuance period (used for race-safe upsert).
CREATE UNIQUE INDEX idx_issuer_key_issue_from ON issuer_key (issue_from);

COMMENT ON TABLE issuer_key IS
    'Credential issuer key pairs, rotated monthly. The private key is encrypted '
    'at rest (application-layer encryption using a TEE-held master key). The '
    'public key and domain separator are served publicly via GET /v1/keys. '
    'Key periods align with calendar months to simplify billing period alignment.';

COMMENT ON COLUMN issuer_key.id IS
    'Unique key identifier (UUID). Primary key and foreign key '
    'target for the nullifier table.';

COMMENT ON COLUMN issuer_key.private_key_enc IS
    'AES-256-GCM encrypted credential private key (Ristretto255 scalar). '
    'Decrypted only inside the TEE at runtime. The encryption key is derived '
    'from the TEE''s sealing key.';

COMMENT ON COLUMN issuer_key.public_key IS
    'Credential public key (compressed Ristretto255 point, 32 bytes). Served '
    'to clients for credential verification.';

COMMENT ON COLUMN issuer_key.domain_separator IS
    'Full domain separator string, e.g., '
    '''ACT-v1:eidolons:inference:production:2026-03''. Included in all '
    'cryptographic operations for domain separation.';

COMMENT ON COLUMN issuer_key.issue_from IS
    'Start of the period during which new credentials may be issued with this key.';

COMMENT ON COLUMN issuer_key.issue_until IS
    'End of the issuance window. After this, no new credentials are issued with '
    'this key, but existing credentials remain spendable until accept_until.';

COMMENT ON COLUMN issuer_key.accept_until IS
    'Grace period end. Credentials signed by this key are accepted for spending '
    'until this timestamp. Typically 2-3 days after issue_until to give '
    'clients time to spend down. Nullifiers for this key can be pruned after '
    'this date.';

-- ---------------------------------------------------------------------------
-- Credit Ledger
-- ---------------------------------------------------------------------------

CREATE TABLE credit_ledger (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id      UUID NOT NULL REFERENCES account(id),
    delta           BIGINT NOT NULL,
    reason          TEXT NOT NULL
        CHECK (reason IN (
            'subscription_renewal',
            'purchase',
            'refund',
            'credential_issuance',
            'dispute_clawback',
            'dispute_reversal',
            'manual_adjustment'
        )),
    stripe_event_id TEXT UNIQUE,
    memo            TEXT,
    expires_at      TIMESTAMPTZ,
    credential_key_id    UUID REFERENCES issuer_key(id),
    credential_credits   BIGINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Stripe-originated entries must carry their event ID for idempotency.
    CONSTRAINT require_stripe_event_id CHECK (
        reason NOT IN (
            'subscription_renewal', 'purchase', 'refund',
            'dispute_clawback', 'dispute_reversal'
        )
        OR stripe_event_id IS NOT NULL
    ),

    -- Credits from payments must be positive; debits must be negative;
    -- manual adjustments can go either way. delta = 0 is never valid.
    CONSTRAINT delta_nonzero CHECK (delta != 0),
    CONSTRAINT delta_sign CHECK (
        reason = 'manual_adjustment'
        OR (reason IN ('subscription_renewal', 'purchase', 'dispute_reversal') AND delta > 0)
        OR (reason IN ('credential_issuance', 'refund', 'dispute_clawback') AND delta < 0)
    ),

    -- credential_key_id and credential_credits are only set for credential issuance entries.
    CONSTRAINT credential_issuance_metadata CHECK (
        (reason = 'credential_issuance' AND credential_key_id IS NOT NULL AND credential_credits IS NOT NULL)
        OR
        (reason != 'credential_issuance' AND credential_key_id IS NULL AND credential_credits IS NULL)
    ),

    -- credential_credits must equal the absolute value of delta on issuance rows.
    -- Redundant by construction, but guards against application bugs.
    CONSTRAINT credential_issuance_credits_match CHECK (
        reason != 'credential_issuance'
        OR credential_credits = -delta
    )
);

COMMENT ON TABLE credit_ledger IS
    'Append-only, single-entry ledger of credit mutations. Every change to an '
    'account''s balance — whether from payment, provisioning, dispute, or admin '
    'action — is a row in this table. The current balance is always derived: '
    'SUM(delta) WHERE expires_at IS NULL OR expires_at > now(). '
    'Rows are never updated or deleted.';

COMMENT ON COLUMN credit_ledger.delta IS
    'Credit amount in micro-dollars (1 credit = $0.000001). Positive for '
    'credits added (payments, adjustments), negative for credits consumed '
    '(credential issuance, refunds, clawbacks). A $20 subscription renewal is +20000000.';

COMMENT ON COLUMN credit_ledger.reason IS
    'Informational tag for filtering and auditing. Does not drive business '
    'logic — only delta and expires_at have operational meaning. Reasons: '
    'subscription_renewal = recurring Stripe subscription payment; '
    'purchase = one-time Stripe purchase (premium pricing, no expiry); '
    'refund = Stripe refund (full or partial), cooperative; '
    'credential_issuance = credits converted into anonymous credentials (the privacy boundary); '
    'dispute_clawback = Stripe dispute/chargeback, adversarial; '
    'dispute_reversal = dispute resolved in our favor; '
    'manual_adjustment = admin correction (positive or negative).';

COMMENT ON COLUMN credit_ledger.stripe_event_id IS
    'Stripe event ID (e.g., evt_xxx) for entries originating from webhooks. '
    'UNIQUE constraint provides idempotent webhook handling — duplicate '
    'delivery simply fails the insert. NULL for non-Stripe entries.';

COMMENT ON COLUMN credit_ledger.memo IS
    'Optional free-text note. Primarily useful for manual_adjustment entries '
    '("reversed accidental double-credit per support ticket #123"). '
    'Not exposed to end users.';

COMMENT ON COLUMN credit_ledger.expires_at IS
    'NULL means credits never expire (top-ups, manual adjustments). '
    'For subscription renewals, set to the billing period end date. '
    'Expired credits are excluded from balance calculations by the query, '
    'not by a cron job. '
    'For debit entries (credential_issuance, refund, dispute_clawback), set to '
    'match the balance pool being consumed. E.g., if consuming subscription '
    'credits expiring Mar 1, the debit also carries expires_at = Mar 1. '
    'This keeps the expiring vs permanent breakdown accurate.';

COMMENT ON COLUMN credit_ledger.created_at IS
    'When this ledger entry was created in our system. NOT the upstream event '
    'timestamp — for that, look up the stripe_event_id via Stripe''s API.';

COMMENT ON COLUMN credit_ledger.credential_key_id IS
    'Only set for credential_issuance entries. The issuer key ID '
    'used to sign the credential. Allows querying "how many credits were provisioned '
    'per key" without joining to issuer_key.';

COMMENT ON COLUMN credit_ledger.credential_credits IS
    'Only set for credential_issuance entries. The credit amount loaded into the '
    'issued credential. Always equals -delta by constraint. Stored explicitly for '
    'query convenience ("show me the distribution of credential sizes").';

-- Primary query path: "what is this account's available balance?"
CREATE INDEX idx_ledger_account_balance ON credit_ledger (account_id, expires_at)
    INCLUDE (delta);

-- Audit/admin: "show me all entries for a given reason this month"
CREATE INDEX idx_ledger_reason_created ON credit_ledger (reason, created_at);

-- ---------------------------------------------------------------------------
-- Nullifier
-- ---------------------------------------------------------------------------

CREATE TABLE nullifier (
    issuer_key_id UUID NOT NULL REFERENCES issuer_key(id),
    value         BYTEA NOT NULL,
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (issuer_key_id, value)
);

COMMENT ON TABLE nullifier IS
    'Spent credential nullifiers. A nullifier must be durably recorded BEFORE '
    'a refund credential is issued to prevent the forking attack (re-spending a '
    'credential with a different amount to create divergent credential chains). '
    'The compound primary key (issuer_key_id, value) partitions nullifiers by key, '
    'enabling efficient bulk pruning once a key''s accept_until has '
    'passed. This table lives in the same database as the billing schema for '
    'durability guarantees. In a production split-environment deployment, '
    'it would move to the service environment''s own durable store.';

COMMENT ON COLUMN nullifier.issuer_key_id IS
    'The issuer key under which this credential was issued and spent. '
    'Partitions the nullifier space for lifecycle management.';

COMMENT ON COLUMN nullifier.value IS
    'The raw nullifier scalar (32 bytes for Ristretto255). Revealed in the '
    'clear during the spend proof. Uniqueness within a key is enforced by '
    'the primary key — a duplicate insert fails, indicating a double-spend.';

COMMENT ON COLUMN nullifier.recorded_at IS
    'When the nullifier was recorded. Useful for debugging and for verifying '
    'that nullifiers were recorded before refunds were issued.';

-- ---------------------------------------------------------------------------
-- Helper Functions
-- ---------------------------------------------------------------------------

CREATE FUNCTION available_balance(p_account_id UUID)
RETURNS BIGINT
LANGUAGE SQL STABLE
AS $$
    SELECT COALESCE(SUM(delta), 0)
    FROM credit_ledger
    WHERE account_id = p_account_id
      AND (expires_at IS NULL OR expires_at > now())
$$;

COMMENT ON FUNCTION available_balance IS
    'Returns the total available credit balance for an account. Used as a '
    'fast check before credential provisioning. For the full breakdown '
    '(expiring vs permanent), query the account_balance view instead.';

CREATE FUNCTION record_nullifier(p_issuer_key_id UUID, p_value BYTEA)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO nullifier (issuer_key_id, value) VALUES (p_issuer_key_id, p_value);
    RETURN TRUE;
EXCEPTION
    WHEN unique_violation THEN
        RETURN FALSE;  -- double-spend attempt
END;
$$;

COMMENT ON FUNCTION record_nullifier IS
    'Attempts to record a nullifier. Returns TRUE on success, FALSE if the '
    'nullifier was already recorded (double-spend). Uses the primary key '
    'unique constraint for atomicity — no TOCTOU race. The caller MUST NOT '
    'issue a refund credential unless this function returns TRUE.';

CREATE FUNCTION prune_expired_nullifiers()
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    pruned BIGINT;
BEGIN
    DELETE FROM nullifier
    WHERE issuer_key_id IN (
        SELECT id FROM issuer_key WHERE accept_until < now()
    );
    GET DIAGNOSTICS pruned = ROW_COUNT;
    RETURN pruned;
END;
$$;

COMMENT ON FUNCTION prune_expired_nullifiers IS
    'Removes nullifiers for keys whose accept_until has passed. '
    'Credentials from these keys can no longer be spent, so their nullifiers '
    'are no longer needed. Returns the number of nullifiers pruned. '
    'Safe to run periodically (e.g., daily) or manually after key rotation.';

COMMIT;
