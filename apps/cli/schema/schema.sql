PRAGMA foreign_keys = ON;

-- ============================================================
-- Issuer key registry
-- ============================================================
CREATE TABLE issuer_key (
    id              TEXT PRIMARY KEY,
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
    credential_nonce TEXT REFERENCES credential(nonce),
    issuer_key_id   TEXT NOT NULL REFERENCES issuer_key(id),
    data            BLOB NOT NULL,
    credits         INTEGER,
    spend_amount    INTEGER,
    created_at      TEXT NOT NULL,

    CHECK (
        (type = 'issuance'
            AND credential_nonce IS NULL
            AND spend_amount IS NULL
            AND credits IS NOT NULL)
        OR
        (type = 'refund'
            AND credential_nonce IS NOT NULL
            AND spend_amount IS NOT NULL
            AND credits IS NULL)
    )
);

CREATE UNIQUE INDEX idx_one_spend_per_credential
    ON pre_credential (credential_nonce)
    WHERE type = 'refund';

-- ============================================================
-- Credential: immutable materialized CreditToken
-- ============================================================
CREATE TABLE credential (
    nonce               TEXT PRIMARY KEY,
    pre_credential_id   TEXT NOT NULL UNIQUE
                        REFERENCES pre_credential(id),
    issuer_key_id       TEXT NOT NULL
                        REFERENCES issuer_key(id),
    data                BLOB NOT NULL,
    credits             INTEGER NOT NULL,
    generation          INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL
);

-- ============================================================
-- Lifecycle view
-- ============================================================
CREATE VIEW credential_lifecycle AS
SELECT
    c.nonce,
    c.credits,
    c.generation,
    c.created_at,
    c.issuer_key_id,
    CASE
        WHEN ik.expires_at IS NOT NULL
             AND ik.expires_at < datetime('now')    THEN 'expired'
        WHEN pc_spend.id IS NULL                    THEN 'active'
        WHEN c_next.nonce IS NULL                   THEN 'spending'
        ELSE                                             'spent'
    END AS state,
    pc_spend.id             AS pending_spend_id,
    pc_spend.spend_amount   AS spend_amount,
    c_next.nonce            AS successor_nonce
FROM credential c
JOIN issuer_key ik
    ON  ik.id = c.issuer_key_id
LEFT JOIN pre_credential pc_spend
    ON  pc_spend.credential_nonce = c.nonce
    AND pc_spend.type = 'refund'
LEFT JOIN credential c_next
    ON  c_next.pre_credential_id = pc_spend.id;
