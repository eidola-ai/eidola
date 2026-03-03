-- Eidólons Billing Schema
-- Targeting PostgreSQL 16+
--
-- This schema covers the "identified" side of the system: accounts, payments,
-- credit balances, and ACT provisioning records. It also includes issuer key
-- management and nullifier storage, which serve both the identified and
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
            'act_issuance',
            'dispute_clawback',
            'dispute_reversal',
            'manual_adjustment'
        )),
    stripe_event_id TEXT UNIQUE,
    memo            TEXT,
    expires_at      TIMESTAMPTZ,
    token_epoch     TEXT,
    token_credits   BIGINT,
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
        OR (reason IN ('act_issuance', 'refund', 'dispute_clawback') AND delta < 0)
    ),

    -- token_epoch and token_credits are only set for ACT issuance entries.
    CONSTRAINT act_issuance_metadata CHECK (
        (reason = 'act_issuance' AND token_epoch IS NOT NULL AND token_credits IS NOT NULL)
        OR
        (reason != 'act_issuance' AND token_epoch IS NULL AND token_credits IS NULL)
    ),

    -- token_credits must equal the absolute value of delta on issuance rows.
    -- Redundant by construction, but guards against application bugs.
    CONSTRAINT act_issuance_credits_match CHECK (
        reason != 'act_issuance'
        OR token_credits = -delta
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
    '(ACT issuance, refunds, clawbacks). A $20 subscription renewal is +20000000.';

COMMENT ON COLUMN credit_ledger.reason IS
    'Informational tag for filtering and auditing. Does not drive business '
    'logic — only delta and expires_at have operational meaning. Reasons: '
    'subscription_renewal = recurring Stripe subscription payment; '
    'purchase = one-time Stripe purchase (premium pricing, no expiry); '
    'refund = Stripe refund (full or partial), cooperative; '
    'act_issuance = credits converted into anonymous tokens (the privacy boundary); '
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
    'For debit entries (act_issuance, refund, dispute_clawback), set to '
    'match the balance pool being consumed. E.g., if consuming subscription '
    'credits expiring Mar 1, the debit also carries expires_at = Mar 1. '
    'This keeps the expiring vs permanent breakdown accurate.';

COMMENT ON COLUMN credit_ledger.created_at IS
    'When this ledger entry was created in our system. NOT the upstream event '
    'timestamp — for that, look up the stripe_event_id via Stripe''s API.';

COMMENT ON COLUMN credit_ledger.token_epoch IS
    'Only set for act_issuance entries. The issuer key epoch (e.g., "2026-03") '
    'used to sign the token. Allows querying "how many credits were provisioned '
    'in each epoch" without joining to issuer_key.';

COMMENT ON COLUMN credit_ledger.token_credits IS
    'Only set for act_issuance entries. The credit amount loaded into the '
    'issued token. Always equals -delta by constraint. Stored explicitly for '
    'query convenience ("show me the distribution of token sizes").';

-- Primary query path: "what is this account's available balance?"
CREATE INDEX idx_ledger_account_balance ON credit_ledger (account_id, expires_at)
    INCLUDE (delta);

-- Audit/admin: "show me all entries for a given reason this month"
CREATE INDEX idx_ledger_reason_created ON credit_ledger (reason, created_at);

-- ---------------------------------------------------------------------------
-- Issuer Key
-- ---------------------------------------------------------------------------

CREATE TABLE issuer_key (
    epoch           TEXT PRIMARY KEY,
    private_key_enc BYTEA NOT NULL,
    public_key      BYTEA NOT NULL,
    domain_separator TEXT NOT NULL,
    valid_from      TIMESTAMPTZ NOT NULL,
    valid_until     TIMESTAMPTZ NOT NULL,
    accept_until    TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT epoch_format CHECK (epoch ~ '^\d{4}-\d{2}$'),
    CONSTRAINT valid_window CHECK (valid_from < valid_until),
    CONSTRAINT grace_window CHECK (valid_until <= accept_until)
);

COMMENT ON TABLE issuer_key IS
    'ACT issuer key pairs, rotated monthly. The private key is encrypted at '
    'rest (application-layer encryption using a TEE-held master key). The '
    'public key and domain separator are served publicly via GET /v1/keys. '
    'Key epochs align with calendar months to simplify billing period alignment.';

COMMENT ON COLUMN issuer_key.epoch IS
    'Key epoch identifier in YYYY-MM format. Primary key and foreign key '
    'target for the nullifier table. Aligns with calendar months.';

COMMENT ON COLUMN issuer_key.private_key_enc IS
    'AES-256-GCM encrypted ACT private key (Ristretto255 scalar). Decrypted '
    'only inside the TEE at runtime. The encryption key is derived from the '
    'TEE''s sealing key.';

COMMENT ON COLUMN issuer_key.public_key IS
    'ACT public key (compressed Ristretto255 point, 32 bytes). Served to '
    'clients for token verification.';

COMMENT ON COLUMN issuer_key.domain_separator IS
    'Full ACT domain separator string, e.g., '
    '''ACT-v1:eidolons:inference:production:2026-03''. Included in all '
    'cryptographic operations for domain separation.';

COMMENT ON COLUMN issuer_key.valid_from IS
    'Start of the period during which new tokens may be issued with this key.';

COMMENT ON COLUMN issuer_key.valid_until IS
    'End of the issuance window. After this, no new tokens are issued with '
    'this key, but existing tokens remain spendable until accept_until.';

COMMENT ON COLUMN issuer_key.accept_until IS
    'Grace period end. Tokens signed by this key are accepted for spending '
    'until this timestamp. Typically 2-3 days after valid_until to give '
    'clients time to spend down. Nullifiers for this key can be pruned after '
    'this date.';

-- ---------------------------------------------------------------------------
-- Nullifier
-- ---------------------------------------------------------------------------

CREATE TABLE nullifier (
    epoch       TEXT NOT NULL REFERENCES issuer_key(epoch),
    value       BYTEA NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (epoch, value)
);

COMMENT ON TABLE nullifier IS
    'Spent ACT token nullifiers. A nullifier must be durably recorded BEFORE '
    'a refund token is issued to prevent the forking attack (re-spending a '
    'token with a different amount to create divergent token chains). '
    'The compound primary key (epoch, value) partitions nullifiers by key '
    'epoch, enabling efficient bulk pruning once an epoch''s accept_until has '
    'passed. This table lives in the same database as the billing schema for '
    'durability guarantees. In a production split-environment deployment, '
    'it would move to the service environment''s own durable store.';

COMMENT ON COLUMN nullifier.epoch IS
    'The issuer key epoch under which this token was issued and spent. '
    'Partitions the nullifier space for lifecycle management.';

COMMENT ON COLUMN nullifier.value IS
    'The raw nullifier scalar (32 bytes for Ristretto255). Revealed in the '
    'clear during the spend proof. Uniqueness within an epoch is enforced by '
    'the primary key — a duplicate insert fails, indicating a double-spend.';

COMMENT ON COLUMN nullifier.recorded_at IS
    'When the nullifier was recorded. Useful for debugging and for verifying '
    'that nullifiers were recorded before refunds were issued.';

-- ---------------------------------------------------------------------------
-- Convenience Views
-- ---------------------------------------------------------------------------

CREATE VIEW account_balance AS
    SELECT
        account_id,
        COALESCE(SUM(delta) FILTER (
            WHERE expires_at IS NOT NULL AND expires_at > now()
        ), 0) AS expiring_credits,
        MIN(expires_at) FILTER (
            WHERE expires_at IS NOT NULL AND expires_at > now() AND delta > 0
        ) AS earliest_expiry,
        COALESCE(SUM(delta) FILTER (
            WHERE expires_at IS NULL
        ), 0) AS permanent_credits,
        COALESCE(SUM(delta) FILTER (
            WHERE expires_at IS NULL OR expires_at > now()
        ), 0) AS total_available
    FROM credit_ledger
    GROUP BY account_id;

COMMENT ON VIEW account_balance IS
    'Derived view of per-account credit balances, broken down by expiring '
    '(subscription) and permanent (top-up) pools. total_available is the '
    'number used for provisioning eligibility checks. Debit entries should '
    'carry the same expires_at as the balance pool they consume from so the '
    'breakdown stays accurate. The provisioning endpoint should consume '
    'expiring credits first, selecting those with the earliest expires_at.';

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
    'fast check before ACT provisioning. For the full breakdown (expiring '
    'vs permanent), query the account_balance view instead.';

CREATE FUNCTION record_nullifier(p_epoch TEXT, p_value BYTEA)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO nullifier (epoch, value) VALUES (p_epoch, p_value);
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
    'issue a refund token unless this function returns TRUE.';

CREATE FUNCTION prune_expired_nullifiers()
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    pruned BIGINT;
BEGIN
    DELETE FROM nullifier
    WHERE epoch IN (
        SELECT epoch FROM issuer_key WHERE accept_until < now()
    );
    GET DIAGNOSTICS pruned = ROW_COUNT;
    RETURN pruned;
END;
$$;

COMMENT ON FUNCTION prune_expired_nullifiers IS
    'Removes nullifiers for key epochs whose accept_until has passed. '
    'Tokens from these epochs can no longer be spent, so their nullifiers '
    'are no longer needed. Returns the number of nullifiers pruned. '
    'Safe to run periodically (e.g., daily) or manually after key rotation.';

COMMIT;
