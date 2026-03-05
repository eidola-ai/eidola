PRAGMA foreign_keys = ON;

-- ============================================================
-- Issuer key registry
-- ============================================================
CREATE TABLE issuer_key (
    key_id          TEXT PRIMARY KEY,
    epoch           TEXT NOT NULL,
    params_hash     TEXT NOT NULL,
    public_key_data BLOB NOT NULL,
    params_data     BLOB NOT NULL,
    expires_at      TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- ============================================================
-- Pre-credential: append-only log of in-flight protocol states
-- ============================================================
CREATE TABLE pre_credential (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL CHECK (type IN ('issuance', 'refund')),
    credential_id   TEXT REFERENCES credential(id),
    issuer_key_id   TEXT NOT NULL REFERENCES issuer_key(key_id),
    data            BLOB NOT NULL,
    credits         INTEGER,
    spend_amount    INTEGER,
    created_at      TEXT NOT NULL,

    CHECK (
        (type = 'issuance' AND credential_id IS NULL AND spend_amount IS NULL)
        OR
        (type = 'refund'   AND credential_id IS NOT NULL AND spend_amount IS NOT NULL)
    )
);

CREATE UNIQUE INDEX idx_one_spend_per_credential
    ON pre_credential (credential_id)
    WHERE type = 'refund';

-- ============================================================
-- Credential: immutable materialized CreditToken
-- ============================================================
CREATE TABLE credential (
    id                  TEXT PRIMARY KEY,
    pre_credential_id   TEXT NOT NULL UNIQUE
                        REFERENCES pre_credential(id),
    issuer_key_id       TEXT NOT NULL
                        REFERENCES issuer_key(key_id),
    data                BLOB NOT NULL,
    nonce               BLOB NOT NULL,
    credits             INTEGER NOT NULL,
    generation          INTEGER NOT NULL DEFAULT 0,
    expires_at          TEXT,
    created_at          TEXT NOT NULL
);

CREATE INDEX idx_credential_fefo
    ON credential (expires_at, created_at)
    WHERE expires_at IS NOT NULL;

-- ============================================================
-- Lifecycle view
-- ============================================================
CREATE VIEW credential_lifecycle AS
SELECT
    c.id,
    c.credits,
    c.generation,
    c.expires_at,
    c.created_at,
    c.issuer_key_id,
    CASE
        WHEN c.expires_at IS NOT NULL
             AND c.expires_at < datetime('now')     THEN 'expired'
        WHEN pc_spend.id IS NULL                    THEN 'active'
        WHEN c_next.id IS NULL                      THEN 'spending'
        ELSE                                             'spent'
    END AS state,
    pc_spend.id             AS pending_spend_id,
    pc_spend.spend_amount   AS spend_amount,
    c_next.id               AS successor_id
FROM credential c
LEFT JOIN pre_credential pc_spend
    ON  pc_spend.credential_id = c.id
    AND pc_spend.type = 'refund'
LEFT JOIN credential c_next
    ON  c_next.pre_credential_id = pc_spend.id;
