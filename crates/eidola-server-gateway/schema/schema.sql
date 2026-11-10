-- Eidola Billing Schema
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
    'link to payment identity. No subscription state is stored here: Stripe '
    'is the source of truth and is read on demand. What webhooks durably '
    'record is their effect on credits (see credit_ledger), not the '
    'subscription itself.';

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
-- Account Acceptance
-- ---------------------------------------------------------------------------

CREATE TABLE account_acceptance (
    account_id  UUID NOT NULL REFERENCES account(id),
    document    TEXT NOT NULL
        CHECK (document IN ('terms_of_service', 'privacy_policy')),
    version     BIGINT NOT NULL CHECK (version >= 1),
    sha256      TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    accepted_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (account_id, document, sha256)
);

COMMENT ON TABLE account_acceptance IS
    'Append-only record of terms-of-service / privacy-policy acceptance. '
    'Each row binds an account to the SHA-256 of the exact document text it '
    'accepted (the same accept-by-hash mechanism the repository''s CLA uses), '
    'so a dispute can be resolved against the precise version in git history. '
    'Rows are never updated or deleted; re-accepting an already-accepted '
    'version is a no-op that preserves the original timestamp. The currently '
    'REQUIRED versions live in required_document (advanced by the terms-feed '
    'poller and/or seeded from server configuration).';

COMMENT ON COLUMN account_acceptance.version IS
    'The document''s front-matter version number at acceptance time, '
    'stamped from the requiring server''s view. The acceptance gate is '
    'MAX(version) >= required version, so accepting a newer version '
    'satisfies an instance whose required view briefly lags.';

COMMENT ON COLUMN account_acceptance.sha256 IS
    'Hex-encoded SHA-256 of the exact published document text the account '
    'accepted.';

-- ---------------------------------------------------------------------------
-- Required Document
-- ---------------------------------------------------------------------------

CREATE TABLE required_document (
    document    TEXT NOT NULL
        CHECK (document IN ('terms_of_service', 'privacy_policy')),
    version     BIGINT NOT NULL CHECK (version >= 1),
    sha256      TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    url         TEXT NOT NULL,
    first_required_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (document, version)
);

COMMENT ON TABLE required_document IS
    'Append-only record of every legal-document version this cluster has '
    'ever required. One row per (document, version); rows are never '
    'updated or deleted. The CURRENT requirement is the highest version '
    'per document, so monotonicity is structural — a stale poll or a '
    'website briefly serving old content inserts nothing new and can '
    'never regress the requirement — and the full history answers "which '
    'version was being enforced on date X" forever (the acceptance-record '
    'counterpart in account_acceptance answers "what did this account '
    'agree to, and when"). Fed by the terms-feed poller and/or the '
    'startup env seed, both through the same insert-if-absent. Empty '
    'table = acceptance gate disabled. CI enforces that a published '
    'document''s bytes never change without a version increment, which is '
    'what makes the integer ordering trustworthy; a same-version row with '
    'a different hash is a contract violation the poller detects and '
    'refuses to record.';

COMMENT ON COLUMN required_document.first_required_at IS
    'When this version was first observed and became enforced '
    'cluster-wide. Never overwritten — later observations of the same '
    'version are no-ops.';

-- ---------------------------------------------------------------------------
-- Issuer Key
-- ---------------------------------------------------------------------------

CREATE TABLE issuer_key (
    id              TEXT PRIMARY KEY,
    private_key_enc BYTEA NOT NULL,
    public_key      BYTEA NOT NULL,
    domain_separator TEXT NOT NULL,
    issue_from      TIMESTAMPTZ NOT NULL,
    issue_until     TIMESTAMPTZ NOT NULL,
    accept_until    TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT id_is_key_hash CHECK (id = encode(sha256(public_key), 'hex')),
    CONSTRAINT issue_window CHECK (issue_from < issue_until),
    CONSTRAINT grace_window CHECK (issue_until <= accept_until),
    -- One key per epoch: an honest gateway never mints a second.
    CONSTRAINT one_key_per_epoch UNIQUE (issue_from)
);

COMMENT ON TABLE issuer_key IS
    'Credential issuer key pairs, one per epoch of the public issuer calendar '
    '(eidola_common::issuer_calendar: 7-day epochs from the Unix epoch; a key '
    'issues for its epoch and is accepted for one more). The private key is '
    'encrypted at rest under a TEE-held master key, with the id, public key, '
    'domain separator and all three windows bound into the encryption as '
    'associated data: a row whose plain columns were edited does not decrypt, '
    'and the gateway refuses it. The public key and domain separator are '
    'served publicly via GET /v1/keys. Race-safe provisioning is handled by '
    'serializable transactions in the application layer.';

COMMENT ON COLUMN issuer_key.id IS
    'Hex-encoded SHA-256 hash of the serialized public key (64 hex chars). '
    'This is the issuer_key_id per the ACT Privacy Pass spec '
    '(draft-schlesinger-privacypass-act-01, Section 6). Clients use this '
    'to identify keys in Token structures and credential requests.';

COMMENT ON COLUMN issuer_key.private_key_enc IS
    'AES-256-GCM encrypted credential private key (Ristretto255 scalar), as '
    'nonce || ciphertext+tag. Decrypted only inside the TEE at runtime. The '
    'encryption key is derived from CREDENTIAL_MASTER_KEY and the key id; the '
    'associated data binds the id, public key, domain separator, issue_from, '
    'issue_until and accept_until.';

COMMENT ON COLUMN issuer_key.public_key IS
    'Credential public key (compressed Ristretto255 point, 32 bytes). Served '
    'to clients for credential verification.';

COMMENT ON COLUMN issuer_key.domain_separator IS
    'Full domain separator string, e.g., '
    '''ACT-v1:eidola:inference:production:2026-03''. Included in all '
    'cryptographic operations for domain separation.';

COMMENT ON COLUMN issuer_key.issue_from IS
    'Start of the period during which new credentials may be issued with this key.';

COMMENT ON COLUMN issuer_key.issue_until IS
    'End of the issuance window. After this, no new credentials are issued with '
    'this key, but existing credentials remain spendable until accept_until.';

COMMENT ON COLUMN issuer_key.accept_until IS
    'Grace period end. Credentials signed by this key are accepted for spending '
    'until this timestamp: one epoch after issue_until. '
    'Nullifiers for this key are pruned only once the gateway''s retention '
    'margin past this timestamp has also passed (prune_expired_nullifiers).';

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
    stripe_payment_intent TEXT,
    memo            TEXT,
    expires_at      TIMESTAMPTZ,
    credential_key_id    TEXT REFERENCES issuer_key(id),
    credential_credits   BIGINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

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
    'Rows are never updated or deleted. Purchase-level provenance preserves '
    'prices and expiry when different purchases share an expiration date. '
    'Cash debts live separately in billing_dispute and available_balance '
    'shows a negative equivalent only when no positive credits remain.';

COMMENT ON COLUMN credit_ledger.delta IS
    'Credit amount in micro-dollars (1 credit = $0.000001). Positive for '
    'credits added (payments, adjustments), negative for credits consumed '
    '(credential issuance, refund reservations, clawbacks). Cash value is '
    'separately recorded at each purchase''s actual price in billing_purchase.';

COMMENT ON COLUMN credit_ledger.reason IS
    'Operation tag for filtering and auditing; payment reasons require '
    'a billing operation and purchase provenance. Reasons: '
    'subscription_renewal = recurring Stripe subscription payment; '
    'purchase = one-time Stripe purchase (premium pricing, 1-year expiry); '
    'refund = Stripe refund (full or partial), cooperative; '
    'credential_issuance = credits converted into anonymous credentials (the privacy boundary); '
    'dispute_clawback = Stripe dispute/chargeback, adversarial; '
    'dispute_reversal = dispute resolved in our favor; '
    'manual_adjustment = admin correction (positive or negative).';

COMMENT ON COLUMN credit_ledger.stripe_event_id IS
    'Legacy optional diagnostic reference. Billing idempotency is owned by '
    'billing_purchase source ids and billing_operation ids, not webhook event ids.';

COMMENT ON COLUMN credit_ledger.stripe_payment_intent IS
    'Optional payment reference for presentation. Financial identity and '
    'price allocation live in billing_purchase; refund and dispute effects '
    'are keyed by their durable operation, never by a webhook event.';

COMMENT ON COLUMN credit_ledger.memo IS
    'Optional free-text note. Primarily useful for manual_adjustment entries '
    '("reversed accidental double-credit per support ticket #123"). '
    'Not exposed to end users.';

COMMENT ON COLUMN credit_ledger.expires_at IS
    'The original credit lot''s expiry, retained by issuance debits, refund '
    'reservations and all dispute adjustments. NULL is reserved for '
    'non-purchase manual credits. Uncovered chargeback debt is stored in '
    'currency minor units in billing_dispute, not in an expiring credit row. '
    'Cash refund eligibility frozen by a hold or closure outlives this date.';

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

-- Refund matching: locate the original credit entry for a payment intent.
CREATE INDEX idx_ledger_payment_intent ON credit_ledger (stripe_payment_intent)
    WHERE stripe_payment_intent IS NOT NULL;

-- ---------------------------------------------------------------------------
-- Credential Issuance
-- ---------------------------------------------------------------------------
--
-- One row per idempotent issuance: the signed response, kept so a client whose
-- response was lost can ask again under the same idempotency key and receive
-- the same response instead of being debited a second time. Account-surface
-- data only, beside the ledger row it was debited by (which already records
-- the account, amount, key and time). No timestamp of its own: the ledger
-- row has one, and nothing here needs a second.

CREATE TABLE credential_issuance (
    account_id        UUID   NOT NULL REFERENCES account(id),
    idempotency_key   BYTEA  NOT NULL CHECK (octet_length(idempotency_key) = 32),
    request_digest    BYTEA  NOT NULL CHECK (octet_length(request_digest) = 32),
    issuer_key_id     TEXT   NOT NULL REFERENCES issuer_key(id),
    credits           BIGINT NOT NULL CHECK (credits > 0),
    issuance_response BYTEA  NOT NULL,
    ledger_entry_id   UUID   NOT NULL UNIQUE REFERENCES credit_ledger(id),
    PRIMARY KEY (account_id, idempotency_key)
);

-- Pruning walks a key's rows at once.
CREATE INDEX idx_credential_issuance_key ON credential_issuance (issuer_key_id);

COMMENT ON TABLE credential_issuance IS
    'Stored responses of idempotent credential issuances, keyed by the account '
    'and the client-chosen idempotency key. A retry with the same key and the '
    'same request is answered from here without a second debit; one with the '
    'same key and a different request is refused. Written in the same '
    'transaction as the debit (record_credential_issuance), so a row exists '
    'exactly when the account was debited for it. Pruned with the key''s '
    'nullifiers (prune_expired_issuances), once the key can be used for nothing.';

COMMENT ON COLUMN credential_issuance.idempotency_key IS
    '32 random bytes the client generated and persisted before it first sent '
    'the request. Used on this surface only; it is never sent with a spend.';

COMMENT ON COLUMN credential_issuance.request_digest IS
    'SHA-256 over the request the response answers (the CBOR issuance request '
    'and the credit amount), so a retry naming this idempotency key with any '
    'other request is refused rather than answered with a response to a '
    'different commitment.';

COMMENT ON COLUMN credential_issuance.issuance_response IS
    'The CBOR IssuanceResponse as returned. The issuer computed it and could '
    'always have kept it: issuance is blind, so it carries nothing that '
    'recognises the credential when it is later spent.';

-- ---------------------------------------------------------------------------
-- Nullifier
-- ---------------------------------------------------------------------------

-- Deliberately timestamp-free: a per-spend timestamp on the unlinked
-- surface would sit in the same database as per-account issuance
-- timestamps in credit_ledger, giving the operator statistical
-- issuance-to-redemption correlation material under low traffic
-- (privacy-guarantees.md §2.5). Rows carry only what double-spend
-- prevention and refund recovery require.
CREATE TABLE nullifier (
    issuer_key_id TEXT NOT NULL REFERENCES issuer_key(id),
    value         BYTEA NOT NULL,
    refund_token  BYTEA,
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

COMMENT ON COLUMN nullifier.refund_token IS
    'The refund token for this spend, sealed: a version byte (0x03), a '
    '12-byte random nonce, then a stage byte (0x00 provisional, 0x01 final) '
    'and the CBOR RefundMsg, under AES-256-GCM with its tag. Written with the '
    'nullifier in one statement, as the provisional refund (t = 0), and '
    'overwritten with the final refund when the request ends, or by recovery '
    'declaring the provisional one final, each a compare-and-swap; the stage '
    'is inside the seal, so the database sees only the second write. The key is HKDF-SHA256 (salt = issuer_key_id, info = '
    'eidola-refund-token-v1) over the refund recovery key the client sent with '
    'the spend, and nothing else, derived in enclave memory and kept nowhere: '
    'the client key is never stored or logged, so no one without the client '
    '(the operator included) can open the value; the client presents its key '
    'again to recover the token. The associated '
    'data is the version byte, the length-prefixed issuer_key_id and the '
    'nullifier value, so a token opens only on its own row. Sealed because the '
    'RefundMsg carries the refund amount t as a plain scalar (only its '
    'signature parts are useful solely to the client holding the matching '
    'PreRefund), and t is a function of the request''s content lengths. The '
    'ciphertext is as long as the stage prefix and the fixed-size CBOR, so its '
    'length says nothing, at either stage. '
    'Stored so the client can recover the token via POST '
    '/v1/credentials/refund if the original response was lost; a value that '
    'does not open for the presented key is answered as no token stored.';

-- A database holding refund tokens in an earlier format (bare CBOR, which
-- begins 0xa5, or an earlier seal version) is cleared by hand, idempotently:
--
--   UPDATE nullifier SET refund_token = NULL
--    WHERE substring(refund_token FROM 1 FOR 1) <> '\x03'::bytea;
--
-- That clears current values only: WAL, base backups and replicas keep the
-- old ones until they age out.

-- ---------------------------------------------------------------------------
-- Engine Placement
-- ---------------------------------------------------------------------------
--
-- Self-contained and idempotent (IF NOT EXISTS throughout), so it can be
-- applied on its own to a database created before it existed. Shared by every
-- gateway version running against the database, so it changes additively
-- only: gateways read named columns, and a column one adds must not change
-- what an older gateway's read means.

CREATE TABLE IF NOT EXISTS engine_placement (
    model_id    TEXT NOT NULL,
    deployment  TEXT NOT NULL CHECK (deployment ~ '^[0-9a-f]{64}$'),
    base_url    TEXT NOT NULL CHECK (base_url ~ '^https://'),
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (model_id, base_url)
);

COMMENT ON TABLE engine_placement IS
    'Where Eidola-hosted inference engines can be reached, edited by an '
    'operator. Availability only, never trust: a gateway uses a row only '
    'when it is enabled and names a model and a deployment that gateway''s '
    'own build pins, and attests every connection to the engine against '
    'those compiled-in pins. Health is each gateway''s own and is never '
    'written here.';

COMMENT ON COLUMN engine_placement.model_id IS
    'The model''s weights hash, lowercase hex: the key it is listed under in '
    'releases/trust/engine-enclaves.json and the models/ catalog. Never a '
    'tag: a gateway resolves a client''s tag to this hash before routing.';

COMMENT ON COLUMN engine_placement.deployment IS
    'The engine deployment''s config hash: config_sha256 in '
    'releases/trust/engine-enclaves.json (on Intel TDX, the launch''s '
    'MRCONFIGID). A gateway built from a tree that does not pin this '
    'deployment ignores the row.';

COMMENT ON COLUMN engine_placement.base_url IS
    'The engine''s https origin (optionally with a path prefix); the gateway '
    'calls /v1/chat/completions, /v1/engine/info and /healthz under it.';

-- ---------------------------------------------------------------------------
-- Helper Functions
-- ---------------------------------------------------------------------------

-- Self-contained and re-runnable, so an operator can apply it to an existing
-- database by hand. The zero-argument form pruned at accept_until itself,
-- with no margin for spends accepted just before it; it is dropped.
DROP FUNCTION IF EXISTS prune_expired_nullifiers();

CREATE OR REPLACE FUNCTION prune_expired_nullifiers(p_retain_past_accept INTERVAL)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    pruned BIGINT;
BEGIN
    IF p_retain_past_accept IS NULL OR p_retain_past_accept <= interval '0' THEN
        RAISE EXCEPTION 'prune_expired_nullifiers: retention past accept_until must be positive';
    END IF;
    DELETE FROM nullifier
    WHERE issuer_key_id IN (
        SELECT id FROM issuer_key
        WHERE accept_until + p_retain_past_accept < now()
    );
    GET DIAGNOSTICS pruned = ROW_COUNT;
    RETURN pruned;
END;
$$;

COMMENT ON FUNCTION prune_expired_nullifiers(INTERVAL) IS
    'Removes nullifiers for keys retired at least p_retain_past_accept ago '
    '(accept_until + p_retain_past_accept < now()). The gateway calls it '
    'from its key-rotation task with RETENTION_PAST_ACCEPT, the same margin '
    'during which it still signs refunds under a key, so no nullifier is '
    'pruned while its key could accept a spend. Do not call it by hand with '
    'a smaller interval. Returns the number of nullifiers pruned.';

-- Record an issuance idempotently. Locks the account row (debit_account would
-- take the same lock) before looking for an earlier issuance under the key, so
-- a retry racing its own original waits for it and then finds its row rather
-- than debiting a second time. Under that lock, and on the database's own
-- clock read after it is taken (clock_timestamp(), not the statement's now()),
-- the named key must still be issuing: an attempt that passed the gateway's
-- current-key check and then stalled past the key's issue_until debits
-- nothing, so a retry refused for that key (resolve_issuance_off_current_key)
-- is never followed by a debit under it. Returns the stored row's columns:
--   outcome 'issued'  — debited now, row written, response is p_response;
--   outcome 'stored'  — an earlier attempt under this key was debited; its
--                       stored response, digest, key and amount are returned
--                       and nothing is debited (the caller compares digests);
--   outcome 'retired' — the named key has stopped issuing (or is unknown);
--                       nothing debited or written;
--   outcome 'insufficient' — balance too low, nothing debited or written.
CREATE OR REPLACE FUNCTION record_credential_issuance(
    p_account_id      UUID,
    p_idempotency_key BYTEA,
    p_request_digest  BYTEA,
    p_issuer_key_id   TEXT,
    p_credits         BIGINT,
    p_response        BYTEA
)
RETURNS TABLE (
    outcome           TEXT,
    request_digest    BYTEA,
    issuer_key_id     TEXT,
    credits           BIGINT,
    issuance_response BYTEA,
    ledger_entry_id   UUID
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
DECLARE
    v_ids UUID[];
BEGIN
    PERFORM 1 FROM account WHERE id = p_account_id FOR UPDATE;

    RETURN QUERY
        SELECT 'stored'::TEXT, ci.request_digest, ci.issuer_key_id, ci.credits,
               ci.issuance_response, ci.ledger_entry_id
        FROM credential_issuance ci
        WHERE ci.account_id = p_account_id
          AND ci.idempotency_key = p_idempotency_key;
    IF FOUND THEN
        RETURN;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM issuer_key ik
                   WHERE ik.id = p_issuer_key_id
                     AND ik.issue_until > clock_timestamp()) THEN
        RETURN QUERY SELECT 'retired'::TEXT, NULL::BYTEA, NULL::TEXT,
                            NULL::BIGINT, NULL::BYTEA, NULL::UUID;
        RETURN;
    END IF;

    IF EXISTS (SELECT 1 FROM account a WHERE a.id = p_account_id
               AND a.billing_state <> 'open') THEN
        RETURN QUERY SELECT 'closed'::TEXT, NULL::BYTEA, NULL::TEXT,
                            NULL::BIGINT, NULL::BYTEA, NULL::UUID;
        RETURN;
    END IF;

    v_ids := debit_account(p_account_id, p_credits, 'credential_issuance',
                           NULL, p_issuer_key_id, TRUE);
    IF v_ids IS NULL OR cardinality(v_ids) = 0 THEN
        RETURN QUERY SELECT 'insufficient'::TEXT, NULL::BYTEA, NULL::TEXT,
                            NULL::BIGINT, NULL::BYTEA, NULL::UUID;
        RETURN;
    END IF;

    INSERT INTO credential_issuance (account_id, idempotency_key, request_digest,
                                     issuer_key_id, credits, issuance_response,
                                     ledger_entry_id)
    VALUES (p_account_id, p_idempotency_key, p_request_digest, p_issuer_key_id,
            p_credits, p_response, v_ids[1]);

    RETURN QUERY SELECT 'issued'::TEXT, p_request_digest, p_issuer_key_id,
                        p_credits, p_response, v_ids[1];
END;
$$;

COMMENT ON FUNCTION record_credential_issuance IS
    'Idempotent issuance debit. Serialized per account by the account row lock; '
    'answers an idempotency key already used with the row stored for it and '
    'debits nothing; refuses a key whose issue window has closed by the '
    'database clock read under the lock; returns closed for a new issuance '
    'on a closing or closed account, after the stored-response lookup; '
    'otherwise debits and stores in one transaction.';

-- Decide an issuance naming a key that is not the gateway's current issuing
-- key, under the same account lock record_credential_issuance takes, so the
-- verdict is ordered against every attempt under the idempotency key that may
-- still be in flight. Returns the same columns:
--   outcome 'stored'  — an earlier attempt under this key was debited; its
--                       stored row is returned (the caller compares digests);
--   outcome 'retired' — nothing is stored and the named key's issue window
--                       has closed by the database clock read under the lock
--                       (or the key is unknown). record_credential_issuance
--                       refuses such a key on the same clock, so no attempt
--                       under this idempotency key can ever debit: final;
--   outcome 'issuing' — nothing is stored but the named key may still issue
--                       (the gateway's clock or key cache disagrees with the
--                       database). An attempt may yet debit under it: not
--                       final, and the caller answers so.
CREATE OR REPLACE FUNCTION resolve_issuance_off_current_key(
    p_account_id      UUID,
    p_idempotency_key BYTEA,
    p_issuer_key_id   TEXT
)
RETURNS TABLE (
    outcome           TEXT,
    request_digest    BYTEA,
    issuer_key_id     TEXT,
    credits           BIGINT,
    issuance_response BYTEA,
    ledger_entry_id   UUID
)
LANGUAGE plpgsql
AS $$
#variable_conflict use_column
BEGIN
    PERFORM 1 FROM account WHERE id = p_account_id FOR UPDATE;

    RETURN QUERY
        SELECT 'stored'::TEXT, ci.request_digest, ci.issuer_key_id, ci.credits,
               ci.issuance_response, ci.ledger_entry_id
        FROM credential_issuance ci
        WHERE ci.account_id = p_account_id
          AND ci.idempotency_key = p_idempotency_key;
    IF FOUND THEN
        RETURN;
    END IF;

    IF EXISTS (SELECT 1 FROM issuer_key ik
               WHERE ik.id = p_issuer_key_id
                 AND ik.issue_until > clock_timestamp()) THEN
        RETURN QUERY SELECT 'issuing'::TEXT, NULL::BYTEA, NULL::TEXT,
                            NULL::BIGINT, NULL::BYTEA, NULL::UUID;
    ELSE
        RETURN QUERY SELECT 'retired'::TEXT, NULL::BYTEA, NULL::TEXT,
                            NULL::BIGINT, NULL::BYTEA, NULL::UUID;
    END IF;
END;
$$;

COMMENT ON FUNCTION resolve_issuance_off_current_key IS
    'Decides an issuance naming a key that is not the current issuing key, '
    'serialized with record_credential_issuance by the account row lock: the '
    'stored row if an earlier attempt was debited, otherwise whether the key '
    'has stopped issuing by the database clock (final: nothing can debit under '
    'it again) or may still issue (not final).';

CREATE OR REPLACE FUNCTION prune_expired_issuances(p_retain_past_accept INTERVAL)
RETURNS BIGINT
LANGUAGE plpgsql
AS $$
DECLARE
    pruned BIGINT;
BEGIN
    IF p_retain_past_accept IS NULL OR p_retain_past_accept <= interval '0' THEN
        RAISE EXCEPTION 'prune_expired_issuances: retention past accept_until must be positive';
    END IF;
    DELETE FROM credential_issuance
    WHERE issuer_key_id IN (
        SELECT id FROM issuer_key
        WHERE accept_until + p_retain_past_accept < now()
    );
    GET DIAGNOSTICS pruned = ROW_COUNT;
    RETURN pruned;
END;
$$;

COMMENT ON FUNCTION prune_expired_issuances(INTERVAL) IS
    'Removes stored issuance responses for keys retired at least '
    'p_retain_past_accept ago, on the same deadline as their nullifiers. A '
    'credential built from a pruned response could no longer be spent, and the '
    'issuance endpoint refuses any key that is not the current one before it '
    'debits, so a retry arriving after pruning is refused rather than debited '
    'again. Returns the number of rows pruned.';

-- BEGIN SHARED BILLING OPERATIONS
-- Billing operations. This block is embedded verbatim in the gateway baseline.
-- All state here belongs to the linked account surface, never to ACT spends.
-- Producers can remain on their stable ingress contract while billing executors
-- change. An incompatible executor rollout must quiesce the previous executors
-- before changing execution_version; these are admission checks, not migrations.
CREATE TABLE billing_protocol (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
    ingress_version INTEGER NOT NULL CHECK (ingress_version > 0),
    execution_version INTEGER NOT NULL CHECK (execution_version > 0)
);
INSERT INTO billing_protocol(singleton,ingress_version,execution_version) VALUES(TRUE,1,1);

ALTER TABLE account ADD COLUMN billing_state TEXT NOT NULL DEFAULT 'open'
    CHECK (billing_state IN ('open', 'closing', 'closed'));
ALTER TABLE account ADD COLUMN billing_closed_at TIMESTAMPTZ;

CREATE TABLE billing_operation (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID NOT NULL REFERENCES account(id),
    kind TEXT NOT NULL CHECK (kind IN ('purchase', 'refund', 'dispute', 'closure')),
    actor TEXT NOT NULL DEFAULT current_user,
    reason TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(id,account_id)
);

CREATE TABLE billing_purchase (
    id TEXT PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES account(id),
    operation_id UUID NOT NULL UNIQUE REFERENCES billing_operation(id),
    charge_id TEXT UNIQUE,
    payment_intent TEXT UNIQUE,
    invoice_id TEXT UNIQUE,
    currency TEXT NOT NULL CHECK (currency ~ '^[a-z]{3}$'),
    paid_minor BIGINT NOT NULL CHECK (paid_minor >= 0),
    credits BIGINT NOT NULL CHECK (credits > 0),
    expires_at TIMESTAMPTZ NOT NULL,
    purchased_at TIMESTAMPTZ NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('purchase', 'subscription_renewal')),
    CHECK (paid_minor = 0 OR charge_id IS NOT NULL),
    CHECK (kind = 'subscription_renewal' OR expires_at > purchased_at),
    UNIQUE(id,account_id),
    UNIQUE(id,expires_at)
);

ALTER TABLE credit_ledger ADD COLUMN purchase_id TEXT REFERENCES billing_purchase(id);
ALTER TABLE credit_ledger ADD COLUMN billing_operation_id UUID REFERENCES billing_operation(id);
ALTER TABLE credit_ledger ADD CONSTRAINT payment_requires_operation CHECK (
    reason NOT IN ('purchase', 'subscription_renewal', 'refund', 'dispute_clawback', 'dispute_reversal')
    OR (billing_operation_id IS NOT NULL AND purchase_id IS NOT NULL)
);
ALTER TABLE credit_ledger ADD FOREIGN KEY(purchase_id,account_id) REFERENCES billing_purchase(id,account_id);
ALTER TABLE credit_ledger ADD FOREIGN KEY(billing_operation_id,account_id) REFERENCES billing_operation(id,account_id);
ALTER TABLE credit_ledger ADD FOREIGN KEY(purchase_id,expires_at) REFERENCES billing_purchase(id,expires_at);
ALTER TABLE credit_ledger ADD CONSTRAINT purchase_expiry_required CHECK (purchase_id IS NULL OR expires_at IS NOT NULL);
CREATE INDEX idx_ledger_purchase ON credit_ledger(purchase_id);
CREATE INDEX idx_ledger_operation ON credit_ledger(billing_operation_id);

CREATE TABLE billing_refund_operation (
    id UUID PRIMARY KEY REFERENCES billing_operation(id),
    purchase_id TEXT NOT NULL REFERENCES billing_purchase(id),
    credits BIGINT NOT NULL CHECK (credits >= 0),
    amount_minor BIGINT NOT NULL CHECK (amount_minor >= 0),
    state TEXT NOT NULL DEFAULT 'prepared' CHECK (state IN (
        'prepared', 'submitting', 'pending', 'requires_action', 'succeeded',
        'failed', 'canceled', 'released', 'superseded')),
    refund_id TEXT UNIQUE,
    submitted_at TIMESTAMPTZ,
    replaces UUID UNIQUE REFERENCES billing_refund_operation(id)
);

CREATE TABLE billing_external_refund (
    operation_id UUID PRIMARY KEY REFERENCES billing_refund_operation(id),
    reference TEXT NOT NULL UNIQUE,
    actor TEXT NOT NULL DEFAULT current_user,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

CREATE TABLE billing_refund_observation (
    id TEXT PRIMARY KEY,
    account_id UUID NOT NULL REFERENCES account(id),
    charge_id TEXT NOT NULL,
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    currency TEXT NOT NULL CHECK (currency ~ '^[a-z]{3}$'),
    status TEXT NOT NULL CHECK (status IN ('pending', 'requires_action', 'succeeded', 'failed', 'canceled')),
    operation_id UUID UNIQUE REFERENCES billing_refund_operation(id),
    observed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

-- Financial facts whose purchase or operation provenance needs human review.
-- These have no credit effect. Keeping them must not block unrelated purchases.
CREATE TABLE billing_unmatched_observation (
    account_id UUID NOT NULL REFERENCES account(id),
    kind TEXT NOT NULL CHECK (kind IN ('refund', 'dispute')),
    id TEXT NOT NULL,
    charge_id TEXT NOT NULL,
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    currency TEXT NOT NULL,
    status TEXT NOT NULL,
    operation_id UUID,
    reason TEXT NOT NULL,
    PRIMARY KEY(kind,id)
);

CREATE TABLE billing_dispute (
    id TEXT PRIMARY KEY,
    purchase_id TEXT NOT NULL REFERENCES billing_purchase(id),
    operation_id UUID NOT NULL UNIQUE REFERENCES billing_operation(id),
    amount_minor BIGINT NOT NULL CHECK (amount_minor > 0),
    status TEXT NOT NULL CHECK (status IN ('needs_response', 'under_review', 'lost', 'won',
        'warning_needs_response', 'warning_under_review', 'warning_closed')),
    target_minor NUMERIC NOT NULL DEFAULT 0 CHECK (target_minor >= 0),
    debt_minor NUMERIC NOT NULL DEFAULT 0 CHECK (debt_minor >= 0)
);

-- The exact allocations a reversal must undo, including their original expiry.
CREATE TABLE billing_dispute_effect (
    ledger_id UUID PRIMARY KEY REFERENCES credit_ledger(id),
    dispute_id TEXT NOT NULL REFERENCES billing_dispute(id),
    purchase_id TEXT NOT NULL REFERENCES billing_purchase(id),
    credits BIGINT NOT NULL CHECK (credits > 0),
    restored_credits BIGINT NOT NULL DEFAULT 0 CHECK (restored_credits >= 0 AND restored_credits <= credits),
    cash_minor NUMERIC NOT NULL CHECK (cash_minor > 0)
);

CREATE TABLE billing_event (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    protocol_version INTEGER NOT NULL DEFAULT 1 CHECK (protocol_version > 0),
    generation BIGINT NOT NULL DEFAULT 1 CHECK (generation > 0),
    completed_generation BIGINT NOT NULL DEFAULT 0 CHECK (
        completed_generation >= 0 AND completed_generation <= generation),
    report JSONB,
    attempts INTEGER NOT NULL DEFAULT 0,
    completed_at TIMESTAMPTZ,
    last_error TEXT,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX idx_billing_event_pending ON billing_event(next_attempt_at,received_at,id) WHERE completed_at IS NULL;

-- Purchase-level allocation replaces pools which accidentally merged prices.
-- This signature is also called by record_credential_issuance.
CREATE OR REPLACE FUNCTION debit_account(
    p_account_id UUID, p_amount BIGINT, p_reason TEXT,
    p_stripe_event_id TEXT DEFAULT NULL, p_credential_key_id TEXT DEFAULT NULL,
    p_require_balance BOOLEAN DEFAULT TRUE, p_payment_intent TEXT DEFAULT NULL,
    p_memo TEXT DEFAULT NULL
) RETURNS UUID[] LANGUAGE plpgsql AS $$
DECLARE
    v_remaining BIGINT := p_amount;
    v_pool RECORD;
    v_take BIGINT;
    v_id UUID;
    v_ids UUID[] := '{}';
    v_state TEXT;
BEGIN
    SELECT billing_state INTO STRICT v_state FROM account WHERE id = p_account_id FOR UPDATE;
    IF p_amount <= 0 OR p_reason NOT IN ('credential_issuance', 'manual_adjustment') THEN
        RAISE EXCEPTION USING ERRCODE = '22023', MESSAGE = 'use a billing operation for payment adjustments';
    END IF;
    IF p_reason = 'credential_issuance' AND v_state <> 'open' THEN RETURN NULL; END IF;
    IF p_require_balance AND available_balance(p_account_id) < p_amount THEN RETURN NULL; END IF;
    IF p_stripe_event_id IS NOT NULL AND EXISTS (
        SELECT 1 FROM credit_ledger WHERE stripe_event_id = p_stripe_event_id
    ) THEN RETURN '{}'; END IF;
    FOR v_pool IN
        SELECT purchase_id, expires_at, SUM(delta)::BIGINT AS amount
        FROM credit_ledger WHERE account_id = p_account_id
          AND (expires_at IS NULL OR expires_at > clock_timestamp())
        GROUP BY purchase_id, expires_at HAVING SUM(delta) > 0
        ORDER BY expires_at NULLS LAST, purchase_id NULLS LAST
    LOOP
        EXIT WHEN v_remaining = 0;
        v_take := LEAST(v_remaining, v_pool.amount);
        INSERT INTO credit_ledger(account_id, delta, reason, expires_at, purchase_id,
                                  credential_key_id, credential_credits, memo)
        VALUES (p_account_id, -v_take, p_reason, v_pool.expires_at, v_pool.purchase_id,
                p_credential_key_id, CASE WHEN p_reason = 'credential_issuance' THEN v_take END, p_memo)
        RETURNING id INTO v_id;
        v_ids := v_ids || v_id;
        v_remaining := v_remaining - v_take;
    END LOOP;
    IF v_remaining > 0 THEN
        IF p_require_balance THEN RAISE EXCEPTION 'balance allocation invariant'; END IF;
        INSERT INTO credit_ledger(account_id, delta, reason, memo)
        VALUES (p_account_id, -v_remaining, 'manual_adjustment', p_memo) RETURNING id INTO v_id;
        v_ids := v_ids || v_id;
    END IF;
    RETURN v_ids;
END;
$$;

-- Debt is money, in its original currency; never an at-par credit overdraft.
-- The negative equivalent is only a display when no positive credits remain.
CREATE OR REPLACE FUNCTION available_balance(p_account_id UUID)
RETURNS BIGINT LANGUAGE SQL STABLE AS $$
    WITH balance AS (
        SELECT COALESCE(SUM(delta), 0)::BIGINT AS credits FROM credit_ledger
        WHERE account_id = p_account_id AND (expires_at IS NULL OR expires_at > now())
    ), debt AS (
        SELECT COALESCE(SUM(CEIL(d.debt_minor * p.credits / NULLIF(p.paid_minor, 0))), 0)::BIGINT AS credits
        FROM billing_dispute d JOIN billing_purchase p ON p.id = d.purchase_id
        WHERE p.account_id = p_account_id
    ) SELECT CASE WHEN balance.credits > 0 THEN balance.credits
                  ELSE balance.credits - debt.credits END FROM balance, debt
$$;

-- Consume an outstanding cash debt using credits at each lot's own price.
-- Integer credits are rounded down: we never take extra credits to round cash.
CREATE FUNCTION billing_collect_debt(p_dispute TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    d RECORD;
    lot RECORD;
    take BIGINT;
    value NUMERIC;
    entry UUID;
BEGIN
    SELECT bd.*, p.account_id, p.currency INTO STRICT d
    FROM billing_dispute bd JOIN billing_purchase p ON p.id = bd.purchase_id WHERE bd.id = p_dispute;
    PERFORM 1 FROM account WHERE id = d.account_id FOR UPDATE;
    FOR lot IN
        SELECT p.*, COALESCE(SUM(cl.delta), 0)::BIGINT AS remaining
        FROM billing_purchase p JOIN credit_ledger cl ON cl.purchase_id = p.id
        WHERE p.account_id = d.account_id AND p.currency = d.currency AND p.paid_minor > 0
          AND (p.expires_at > COALESCE((SELECT billing_closed_at FROM account WHERE id = d.account_id), clock_timestamp())
               OR p.purchased_at >= (SELECT billing_closed_at FROM account WHERE id = d.account_id))
        GROUP BY p.id HAVING SUM(cl.delta) > 0
        ORDER BY (p.id = d.purchase_id) DESC, p.expires_at, p.id
    LOOP
        EXIT WHEN d.debt_minor <= 0;
        take := LEAST(lot.remaining, FLOOR(d.debt_minor * lot.credits / lot.paid_minor)::BIGINT);
        CONTINUE WHEN take <= 0;
        value := take::NUMERIC * lot.paid_minor / lot.credits;
        INSERT INTO credit_ledger(account_id, delta, reason, expires_at, purchase_id, billing_operation_id)
        VALUES (d.account_id, -take, 'dispute_clawback', lot.expires_at, lot.id, d.operation_id)
        RETURNING id INTO entry;
        INSERT INTO billing_dispute_effect(ledger_id, dispute_id, purchase_id, credits, cash_minor)
        VALUES (entry, d.id, lot.id, take, value);
        d.debt_minor := GREATEST(0, d.debt_minor - value);
    END LOOP;
    UPDATE billing_dispute SET debt_minor = d.debt_minor WHERE id = d.id;
END;
$$;

-- Move toward the authoritative cash target, reversing recorded allocations
-- (not today's price or expiry) when a refund overlaps or a dispute is won.
CREATE FUNCTION billing_set_dispute_target(p_dispute TEXT, p_target NUMERIC)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE
    d RECORD;
    effect RECORD;
    release NUMERIC;
    amount NUMERIC;
    restore BIGINT;
BEGIN
    SELECT bd.*, p.account_id INTO STRICT d FROM billing_dispute bd
    JOIN billing_purchase p ON p.id = bd.purchase_id WHERE bd.id = p_dispute;
    PERFORM 1 FROM account WHERE id = d.account_id FOR UPDATE;
    IF p_target > d.target_minor THEN
        UPDATE billing_dispute SET debt_minor = debt_minor + p_target - target_minor,
            target_minor = p_target WHERE id = d.id;
        PERFORM billing_collect_debt(d.id);
        RETURN;
    END IF;
    release := d.target_minor - p_target;
    amount := LEAST(release, d.debt_minor);
    d.debt_minor := d.debt_minor - amount;
    release := release - amount;
    FOR effect IN
        SELECT e.*, cl.expires_at FROM billing_dispute_effect e
        JOIN credit_ledger cl ON cl.id = e.ledger_id
        WHERE e.dispute_id = d.id AND e.restored_credits < e.credits
        ORDER BY cl.created_at DESC, e.ledger_id
    LOOP
        EXIT WHEN release <= 0 AND p_target <> 0;
        restore := CASE WHEN p_target = 0 THEN effect.credits - effect.restored_credits
                    ELSE LEAST(effect.credits - effect.restored_credits,
                               CEIL(release * effect.credits / effect.cash_minor)::BIGINT) END;
        CONTINUE WHEN restore <= 0;
        amount := restore::NUMERIC * effect.cash_minor / effect.credits;
        INSERT INTO credit_ledger(account_id, delta, reason, expires_at, purchase_id, billing_operation_id)
        VALUES (d.account_id, restore, 'dispute_reversal', effect.expires_at, effect.purchase_id, d.operation_id);
        UPDATE billing_dispute_effect SET restored_credits = restored_credits + restore WHERE ledger_id = effect.ledger_id;
        release := release - amount;
    END LOOP;
    UPDATE billing_dispute SET target_minor = p_target,
        debt_minor = CASE WHEN p_target = 0 THEN 0 ELSE d.debt_minor + GREATEST(0, -release) END
    WHERE id = d.id;
END;
$$;

CREATE FUNCTION billing_refresh_charge(p_purchase TEXT)
RETURNS VOID LANGUAGE plpgsql AS $$
DECLARE p RECORD; d RECORD; remaining NUMERIC; target NUMERIC;
BEGIN
    SELECT * INTO STRICT p FROM billing_purchase WHERE id = p_purchase;
    PERFORM 1 FROM account WHERE id = p.account_id FOR UPDATE;
    SELECT GREATEST(0, p.paid_minor
        - COALESCE((SELECT SUM(amount_minor) FROM billing_refund_observation WHERE charge_id = p.charge_id
                    AND status = 'succeeded'), 0)
        - COALESCE((SELECT SUM(r.amount_minor) FROM billing_external_refund e
                    JOIN billing_refund_operation r ON r.id = e.operation_id WHERE r.purchase_id = p.id), 0))
    INTO remaining;
    FOR d IN SELECT * FROM billing_dispute WHERE purchase_id = p.id ORDER BY id LOOP
        target := CASE WHEN d.status IN ('needs_response', 'under_review', 'lost')
                       THEN LEAST(d.amount_minor, remaining) ELSE 0 END;
        PERFORM billing_set_dispute_target(d.id, target);
        remaining := remaining - target;
    END LOOP;
    -- A win can restore paid credits another outstanding dispute is owed.
    -- Settle those debts before any restored credit becomes convertible.
    FOR d IN SELECT bd.id FROM billing_dispute bd JOIN billing_purchase bp ON bp.id = bd.purchase_id
             WHERE bp.account_id = p.account_id AND bd.debt_minor > 0 ORDER BY bd.id LOOP
        PERFORM billing_collect_debt(d.id);
    END LOOP;
END;
$$;
-- END SHARED BILLING OPERATIONS

COMMIT;
