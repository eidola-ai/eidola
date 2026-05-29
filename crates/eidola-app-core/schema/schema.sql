PRAGMA foreign_keys = ON;

-- ############################################################
-- #  LAYER 0 — WALLET                                        #
-- ############################################################

CREATE TABLE issuer_key (
    id              TEXT PRIMARY KEY,
    params_hash     TEXT NOT NULL,
    public_key_data BLOB NOT NULL,
    params_data     BLOB NOT NULL,
    expires_at      INTEGER NOT NULL,         -- ms since epoch
    created_at      INTEGER NOT NULL
);

CREATE TABLE pre_credential (
    id               TEXT PRIMARY KEY,         -- UUIDv7
    type             TEXT NOT NULL CHECK (type IN ('issuance', 'refund')),
    credential_nonce TEXT REFERENCES credential(nonce),
    issuer_key_id    TEXT NOT NULL REFERENCES issuer_key(id),
    data             BLOB NOT NULL,
    credits          INTEGER,
    spend_amount     INTEGER,
    spend_proof_data BLOB,
    created_at       INTEGER NOT NULL,

    CHECK (
        (type = 'issuance'
            AND credential_nonce IS NULL
            AND spend_amount IS NULL
            AND spend_proof_data IS NULL
            AND credits IS NOT NULL)
        OR
        (type = 'refund'
            AND credential_nonce IS NOT NULL
            AND spend_amount IS NOT NULL
            AND spend_proof_data IS NOT NULL
            AND credits IS NULL)
    )
);

CREATE UNIQUE INDEX idx_one_spend_per_credential
    ON pre_credential (credential_nonce)
    WHERE type = 'refund';

CREATE TABLE credential (
    nonce             TEXT PRIMARY KEY,
    pre_credential_id TEXT NOT NULL UNIQUE
                      REFERENCES pre_credential(id),
    issuer_key_id     TEXT NOT NULL
                      REFERENCES issuer_key(id),
    data              BLOB NOT NULL,
    credits           INTEGER NOT NULL,
    generation        INTEGER NOT NULL DEFAULT 0,
    created_at        INTEGER NOT NULL
);


-- ############################################################
-- #  LAYER 1 — TRANSPORT & ATTESTATION                       #
-- ############################################################

CREATE TABLE provider (
    id          TEXT PRIMARY KEY,              -- UUIDv7
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN (
                    'inference', 'tool', 'retrieval', 'issuance', 'other'
                )),
    created_at  INTEGER NOT NULL
);

CREATE TABLE attestation (
    hash        TEXT PRIMARY KEY,
    doc         BLOB NOT NULL,
    pcr_digest  TEXT,
    created_at  INTEGER NOT NULL
);

CREATE TABLE connection (
    id                TEXT PRIMARY KEY,        -- UUIDv7
    provider_id       TEXT NOT NULL REFERENCES provider(id),
    base_url          TEXT NOT NULL,
    transport         TEXT NOT NULL CHECK (transport IN (
                          'tor', 'clearnet', 'ohttp'
                      )),
    attestation_hash  TEXT REFERENCES attestation(hash),
    opened_at         INTEGER NOT NULL,
    closed_at         INTEGER,
    created_at        INTEGER NOT NULL
);

CREATE INDEX idx_connection_attestation
    ON connection (attestation_hash)
    WHERE attestation_hash IS NOT NULL;


-- ############################################################
-- #  LAYER 2 — SEMANTIC / LOGICAL                            #
-- ############################################################

-- ============================================================
-- Participant: an actor that can emit actions into a space
-- ============================================================
CREATE TABLE participant (
    id          TEXT PRIMARY KEY,              -- UUIDv7
    kind        TEXT NOT NULL CHECK (kind IN (
                    'human', 'agent', 'tool', 'system'
                )),
    label       TEXT NOT NULL,
    provider_id TEXT REFERENCES provider(id),
    created_at  INTEGER NOT NULL
);

-- ============================================================
-- Space: a context namespace
--
-- parent_space_id is navigational: "this space was derived
-- from that space." It does NOT define content boundaries.
-- Content boundaries are handled by context_assembly.
--
-- No fork_point_action_id: the old "everything up to action X"
-- shorthand doesn't translate cleanly to a antecedent DAG where
-- multiple independent causal chains may coexist in one space.
-- ============================================================
CREATE TABLE space (
    id                TEXT PRIMARY KEY,        -- UUIDv7
    parent_space_id   TEXT REFERENCES space(id),
    title             TEXT,
    linkability       TEXT NOT NULL CHECK (linkability IN (
                          'linked', 'unlinked', 'public'
                      )),
    created_at        INTEGER NOT NULL,
    archived_at       INTEGER
);

-- ============================================================
-- Space membership
-- ============================================================
CREATE TABLE space_participant (
    space_id       TEXT NOT NULL REFERENCES space(id),
    participant_id TEXT NOT NULL REFERENCES participant(id),
    role           TEXT NOT NULL CHECK (role IN (
                       'owner', 'member', 'observer'
                   )) DEFAULT 'member',
    joined_at      INTEGER NOT NULL,
    left_at        INTEGER,

    PRIMARY KEY (space_id, participant_id)
);

-- ============================================================
-- Action: the fundamental unit
--
-- origin_action_id: nullable FK to an action in another space.
-- When set, this action is a *reference* — a lightweight
-- pointer created during an edit-and-fork operation. It carries
-- no content_blocks of its own (follow the FK) and no cost
-- attribution. It exists so the forked space has a self-
-- contained history and so that action_antecedent edges within
-- the new space can reference local IDs.
--
-- Origin references are created mechanically. Original work
-- has origin_action_id = NULL. Cost queries filter on this.
-- ============================================================
CREATE TABLE action (
    id              TEXT PRIMARY KEY,          -- UUIDv7
    space_id        TEXT NOT NULL REFERENCES space(id),
    participant_id  TEXT NOT NULL REFERENCES participant(id),

    action_type     TEXT NOT NULL CHECK (action_type IN (
                        'user_input',
                        'inference',
                        'tool_call',
                        'tool_result',
                        'retrieval',
                        'request',
                        'checkpoint',
                        'decision',
                        'publish',
                        'system',
                        'error'
                    )),

    status          TEXT NOT NULL CHECK (status IN (
                        'draft',
                        'streaming',
                        'complete',
                        'cancelled',
                        'error'
                    )) DEFAULT 'complete',

    intent          TEXT,
    model           TEXT,

    -- usage / cost (NULL for origin references)
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    credits_consumed INTEGER,

    -- edit-and-fork: points to the original action this
    -- references. NULL for original work.
    origin_action_id TEXT REFERENCES action(id),

    created_at      INTEGER NOT NULL,

    -- origin references must not carry their own costs
    CHECK (
        origin_action_id IS NULL
        OR (input_tokens IS NULL
            AND output_tokens IS NULL
            AND credits_consumed IS NULL)
    )
);

CREATE INDEX idx_action_space ON action (space_id, created_at);
CREATE INDEX idx_action_participant ON action (participant_id);
CREATE INDEX idx_action_type ON action (action_type);
CREATE INDEX idx_action_origin ON action (origin_action_id)
    WHERE origin_action_id IS NOT NULL;
CREATE INDEX idx_action_status ON action (status)
    WHERE status != 'complete';

-- ============================================================
-- Action antecedent: the causal graph
-- ============================================================
CREATE TABLE action_antecedent (
    action_id               TEXT NOT NULL REFERENCES action(id),
    antecedent_action_id     TEXT NOT NULL REFERENCES action(id),
    ordinal                 INTEGER NOT NULL,

    content_block_id        TEXT REFERENCES content_block(id),
    range_start             INTEGER,
    range_end               INTEGER,
    annotation              TEXT,

    PRIMARY KEY (action_id, antecedent_action_id),
    UNIQUE (action_id, ordinal),

    CHECK (action_id != antecedent_action_id),
    CHECK (
        (range_start IS NULL AND range_end IS NULL)
        OR
        (range_start IS NOT NULL AND range_end IS NOT NULL
         AND range_start >= 0 AND range_end > range_start)
    )
);

CREATE INDEX idx_action_antecedent_reverse
    ON action_antecedent (antecedent_action_id);

-- ============================================================
-- Content block: the typed payload of an action
--
-- Origin-reference actions have no content_blocks; their
-- content is accessed via action.origin_action_id.
-- ============================================================
CREATE TABLE content_block (
    id              TEXT PRIMARY KEY,          -- UUIDv7
    action_id       TEXT NOT NULL REFERENCES action(id),
    ordinal         INTEGER NOT NULL,

    block_type      TEXT NOT NULL CHECK (block_type IN (
                        'text',
                        'thinking',
                        'tool_use',
                        'tool_result',
                        'image',
                        'document',
                        'code',
                        'error'
                    )),

    text_content    TEXT,
    data            TEXT,                      -- JSON
    media_type      TEXT,
    media_data      BLOB,

    tool_name       TEXT,
    tool_call_id    TEXT,

    UNIQUE (action_id, ordinal),

    -- Per-type invariants
    CHECK (
        (block_type IN ('text', 'thinking', 'code', 'error')
            AND text_content IS NOT NULL
            AND media_data IS NULL
            AND tool_name IS NULL
            AND tool_call_id IS NULL)
        OR
        (block_type = 'tool_use'
            AND tool_name IS NOT NULL
            AND tool_call_id IS NOT NULL
            AND media_data IS NULL)
        OR
        (block_type = 'tool_result'
            AND tool_call_id IS NOT NULL
            AND tool_name IS NULL
            AND media_data IS NULL)
        OR
        (block_type IN ('image', 'document')
            AND media_type IS NOT NULL
            AND media_data IS NOT NULL
            AND tool_name IS NULL
            AND tool_call_id IS NULL)
    )
);

CREATE INDEX idx_content_block_tool
    ON content_block (tool_name)
    WHERE tool_name IS NOT NULL;

CREATE INDEX idx_content_block_type
    ON content_block (block_type);

-- ============================================================
-- System prompt: deduplicated by hash
-- ============================================================
CREATE TABLE system_prompt (
    hash    TEXT PRIMARY KEY,
    text    TEXT NOT NULL
);

-- ============================================================
-- Context assembly: what was composed into an inference prompt
-- ============================================================
CREATE TABLE context_assembly (
    id                 TEXT PRIMARY KEY,       -- UUIDv7
    action_id          TEXT NOT NULL UNIQUE REFERENCES action(id),

    system_prompt_hash TEXT REFERENCES system_prompt(hash),

    retrieval_refs     TEXT,                   -- JSON

    total_tokens       INTEGER,
    truncation_applied INTEGER NOT NULL DEFAULT 0,

    created_at         INTEGER NOT NULL
);

-- ============================================================
-- Context assembly <-> action junction
--
-- May reference actions from ANY space. This is the mechanism
-- for cross-space context (dreaming, sub-agent results, etc.).
-- ============================================================
CREATE TABLE context_assembly_action (
    context_assembly_id TEXT NOT NULL
                        REFERENCES context_assembly(id),
    action_id           TEXT NOT NULL
                        REFERENCES action(id),
    position            INTEGER NOT NULL,

    PRIMARY KEY (context_assembly_id, action_id),
    UNIQUE (context_assembly_id, position)
);

-- ============================================================
-- Request: raw HTTP request/response pairs
-- ============================================================
CREATE TABLE request (
    id                TEXT PRIMARY KEY,        -- UUIDv7
    connection_id     TEXT REFERENCES connection(id),
    action_id         TEXT REFERENCES action(id),

    method            TEXT NOT NULL,
    path              TEXT NOT NULL,
    request_headers   TEXT,
    request_body      BLOB,

    response_status   INTEGER,
    response_headers  TEXT,
    response_body     BLOB,

    request_at        INTEGER NOT NULL,
    response_at       INTEGER,
    duration_ms       INTEGER,

    error             TEXT,

    retry_of_id       TEXT REFERENCES request(id),
    attempt_number    INTEGER NOT NULL DEFAULT 1,

    credential_nonce  TEXT REFERENCES credential(nonce),

    created_at        INTEGER NOT NULL
);

CREATE INDEX idx_request_action
    ON request (action_id)
    WHERE action_id IS NOT NULL;

CREATE INDEX idx_request_connection
    ON request (connection_id);

CREATE INDEX idx_request_credential
    ON request (credential_nonce)
    WHERE credential_nonce IS NOT NULL;


-- ############################################################
-- #  CONVENIENCE VIEWS                                       #
-- ############################################################

-- ============================================================
-- Credential lifecycle
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
             AND ik.expires_at < (strftime('%s', 'now') * 1000)
                                                    THEN 'expired'
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

-- ============================================================
-- Action resolved: dereferences origin pointers, so queries
-- always see content regardless of whether an action is
-- original or a reference.
-- ============================================================
CREATE VIEW action_resolved AS
SELECT
    a.id                AS action_id,
    a.space_id,
    a.participant_id,
    a.action_type,
    a.status,
    a.intent,
    a.origin_action_id,
    -- costs come from original, not reference
    COALESCE(orig.model, a.model)                AS model,
    COALESCE(orig.input_tokens, a.input_tokens)  AS input_tokens,
    COALESCE(orig.output_tokens, a.output_tokens) AS output_tokens,
    COALESCE(orig.credits_consumed, a.credits_consumed) AS credits_consumed,
    -- content source: which action_id to query content_blocks against
    COALESCE(a.origin_action_id, a.id)           AS content_source_id,
    a.created_at,
    CASE WHEN a.origin_action_id IS NOT NULL
         THEN 1 ELSE 0
    END AS is_reference
FROM action a
LEFT JOIN action orig ON orig.id = a.origin_action_id;

-- ============================================================
-- Action detail: resolved action + content blocks, flattened
-- ============================================================
CREATE VIEW action_detail AS
SELECT
    ar.action_id,
    ar.space_id,
    ar.participant_id,
    p.kind              AS participant_kind,
    p.label             AS participant_label,
    ar.action_type,
    ar.status,
    ar.intent,
    ar.model,
    ar.credits_consumed,
    ar.is_reference,
    ar.created_at,
    cb.ordinal          AS block_ordinal,
    cb.block_type,
    cb.text_content,
    cb.data             AS block_data,
    cb.tool_name,
    cb.tool_call_id
FROM action_resolved ar
JOIN participant p ON p.id = ar.participant_id
LEFT JOIN content_block cb ON cb.action_id = ar.content_source_id
ORDER BY ar.created_at, cb.ordinal;

-- ============================================================
-- Space history: actions in a space, with drafts filtered out.
-- No lineage walking — spaces are self-contained.
-- For forked spaces, origin-reference actions provide the
-- inherited history.
-- ============================================================
CREATE VIEW space_history AS
SELECT
    a.id                AS action_id,
    a.space_id,
    a.participant_id,
    a.action_type,
    a.status,
    a.intent,
    a.origin_action_id,
    a.created_at
FROM action a
WHERE a.status IN ('complete', 'cancelled')
ORDER BY a.created_at ASC;

-- ============================================================
-- Consequent tree: transitive closure of the antecedent graph
-- ============================================================
CREATE VIEW consequent_tree AS
WITH RECURSIVE descendants (root_action_id, action_id, depth) AS (
    SELECT id, id, 0 FROM action
    UNION ALL
    SELECT d.root_action_id, ap.action_id, d.depth + 1
    FROM descendants d
    JOIN action_antecedent ap ON ap.antecedent_action_id = d.action_id
    WHERE d.depth < 50
)
SELECT
    d.root_action_id,
    d.action_id,
    d.depth,
    a.space_id,
    a.participant_id,
    a.action_type,
    a.status,
    a.intent,
    a.credits_consumed,
    a.created_at
FROM descendants d
JOIN action a ON a.id = d.action_id
WHERE d.depth > 0;

-- ============================================================
-- Spend trail: credential -> request -> action -> space
-- Only counts original work (not origin references).
-- ============================================================
CREATE VIEW spend_trail AS
SELECT
    cl.nonce            AS credential_nonce,
    cl.spend_amount,
    cl.state            AS credential_state,
    r.id                AS request_id,
    r.method,
    r.path,
    r.request_at,
    r.duration_ms,
    r.attempt_number,
    a.id                AS action_id,
    a.action_type,
    a.model,
    a.credits_consumed,
    a.intent,
    s.id                AS space_id,
    s.title             AS space_title,
    s.linkability
FROM credential_lifecycle cl
JOIN request r        ON r.credential_nonce = cl.nonce
LEFT JOIN action a    ON a.id = r.action_id
LEFT JOIN space s     ON s.id = a.space_id
WHERE cl.state IN ('spending', 'spent')
  AND (a.origin_action_id IS NULL);  -- exclude references
