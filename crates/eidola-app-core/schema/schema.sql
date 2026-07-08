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
-- Action: the fundamental unit (immutable, append-only)
--
-- item_id: the stable identity shared by every generation of a
-- logical item (a post / code project / image / tool result —
-- anything an action is a version of). An edit or regeneration
-- appends a NEW action with the same (space_id, item_id);
-- nothing is ever mutated in place. item_id is an independent
-- UUIDv7 (NOT the gen-0 action id) so a future item(space_id,
-- id, …) table is a pure additive FK. An item is space-scoped:
-- all of its generations live in one space.
--
-- supersedes_action_id: the prior generation (NULL for gen 0).
-- The chain is linear (idx_one_successor_per_action), so an
-- item's "current" generation is the unique tip — the action no
-- other action supersedes (see the item_current view). The
-- generation *number* is derived, not stored (see the generation
-- expression in action_resolved); the supersedes chain is the
-- source of truth.
--
-- supersedes_item_id: denormalized item of the superseded
-- generation — by invariant this row's own item (CHECKed), NULL
-- exactly when supersedes_action_id is NULL. It exists so the
-- compound FK (supersedes_action_id, supersedes_item_id) →
-- action(id, item_id) can enforce declaratively that a
-- generation chain never hops items. Causality is preserved
-- through action ids (antecedent edges record which concrete
-- generation was replied to / quoted); the *intended* logical
-- flow is described by item ids (rendering and context assembly
-- resolve through the item to its current tip).
--
-- The DAG stays acyclic by construction: actions are immutable,
-- edges only ever point at already-existing (earlier) actions,
-- and UUIDv7 ids are time-ordered — so no edge can point forward
-- and no cycle can form. (Generation re-rooting in the resolved
-- view is the one logical-cycle case; consumers guard it there.)
-- ============================================================
CREATE TABLE action (
    id              TEXT PRIMARY KEY,          -- UUIDv7
    space_id        TEXT NOT NULL REFERENCES space(id),
    participant_id  TEXT NOT NULL REFERENCES participant(id),

    -- generation identity (generation number is derived, not stored)
    item_id              TEXT NOT NULL,
    supersedes_action_id TEXT,
    supersedes_item_id   TEXT,

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

    -- usage / cost
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    credits_consumed INTEGER,

    created_at      INTEGER NOT NULL,

    -- supersedes is item-scoped: both halves present together, the item
    -- is this row's own, and the referenced (action, item) pair must
    -- really exist — so a generation chain cannot hop items.
    CHECK ((supersedes_action_id IS NULL) = (supersedes_item_id IS NULL)),
    CHECK (supersedes_item_id IS NULL OR supersedes_item_id = item_id),
    FOREIGN KEY (supersedes_action_id, supersedes_item_id)
        REFERENCES action (id, item_id)
);

CREATE INDEX idx_action_space ON action (space_id, created_at);
CREATE INDEX idx_action_participant ON action (participant_id);
CREATE INDEX idx_action_type ON action (action_type);
CREATE INDEX idx_action_item ON action (space_id, item_id);
CREATE INDEX idx_action_status ON action (status)
    WHERE status != 'complete';

-- Linear generation chain: each generation has at most one
-- successor, so an item's tip (current generation) is unique.
CREATE UNIQUE INDEX idx_one_successor_per_action
    ON action (supersedes_action_id)
    WHERE supersedes_action_id IS NOT NULL;

-- One gen-0 per item: together with the one-successor index above,
-- an item's tip is provably unique — item_current can never yield
-- two rows for one item (which would duplicate posts in every
-- resolved view and double edges in the item-tip threading joins).
CREATE UNIQUE INDEX idx_one_root_per_item
    ON action (space_id, item_id)
    WHERE supersedes_action_id IS NULL;

-- Parent key for the compound supersedes FK.
CREATE UNIQUE INDEX idx_action_id_item ON action (id, item_id);

-- ============================================================
-- Action antecedent: the causal graph (a DAG). Every edge points
-- at a temporally prior, already-existing action (its
-- antecedent) — directed and backward, the dual of consequent_tree.
--
-- relation classifies the edge — deliberately minimal, just the
-- structural distinction:
--   reply       structural thread parent (indentation). At most
--               ONE per action (idx_one_reply_parent); roots have
--               none. This is the tree-render key.
--   reference   any non-structural link: a plain backlink, an
--               inline quote (carries range_start/range_end), or
--               an embed (carries content_block_id). The
--               specializer columns disambiguate the kind, so the
--               schema needs no separate quote/transclude values.
--               (Re-expand the enum if a kind ever gains a
--               ramification outside the referring action's content.)
--
-- PK is (action_id, ordinal) — NOT (action_id, antecedent) — so
-- one action may reference the same antecedent multiple times
-- with different ranges/annotations (multi-passage quote).
-- ordinal is the stable order of an action's outgoing edges.
-- ============================================================
CREATE TABLE action_antecedent (
    action_id               TEXT NOT NULL REFERENCES action(id),
    antecedent_action_id    TEXT NOT NULL REFERENCES action(id),
    ordinal                 INTEGER NOT NULL,

    relation                TEXT NOT NULL CHECK (relation IN (
                                'reply', 'reference'
                            )) DEFAULT 'reply',

    content_block_id        TEXT REFERENCES content_block(id),
    range_start             INTEGER,
    range_end               INTEGER,
    annotation              TEXT,

    PRIMARY KEY (action_id, ordinal),

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

-- At most one structural thread parent per action.
CREATE UNIQUE INDEX idx_one_reply_parent
    ON action_antecedent (action_id)
    WHERE relation = 'reply';

-- ============================================================
-- Content block: the typed payload of an action.
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
-- Item current: the tip (current generation) of each item —
-- the action no other action supersedes.
-- ============================================================
CREATE VIEW item_current AS
SELECT
    a.space_id,
    a.item_id,
    a.id          AS current_action_id
FROM action a
WHERE NOT EXISTS (
    SELECT 1 FROM action s WHERE s.supersedes_action_id = a.id
);

-- ============================================================
-- Action resolved: annotates each action with its generation
-- state. (Origin-dereferencing is gone — content and cost now
-- always live on the action itself.) is_current = 1 iff this
-- action is its item's tip generation. generation is *derived*
-- (count of earlier generations in the same item), not stored;
-- the supersedes chain is the source of truth.
-- ============================================================
CREATE VIEW action_resolved AS
SELECT
    a.id                AS action_id,
    a.space_id,
    a.participant_id,
    a.action_type,
    a.status,
    a.intent,
    a.item_id,
    a.supersedes_action_id,
    a.model,
    a.input_tokens,
    a.output_tokens,
    a.credits_consumed,
    a.created_at,
    (SELECT COUNT(*) FROM action b
     WHERE b.item_id = a.item_id
       AND (b.created_at < a.created_at
            OR (b.created_at = a.created_at AND b.id < a.id))
    ) AS generation,
    CASE WHEN NOT EXISTS (
        SELECT 1 FROM action s WHERE s.supersedes_action_id = a.id
    ) THEN 1 ELSE 0 END AS is_current
FROM action a;

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
    ar.item_id,
    ar.generation,
    ar.is_current,
    ar.created_at,
    cb.ordinal          AS block_ordinal,
    cb.block_type,
    cb.text_content,
    cb.data             AS block_data,
    cb.tool_name,
    cb.tool_call_id
FROM action_resolved ar
JOIN participant p ON p.id = ar.participant_id
LEFT JOIN content_block cb ON cb.action_id = ar.action_id
ORDER BY ar.created_at, cb.ordinal;

-- ============================================================
-- Space history: actions in a space, with drafts filtered out.
-- No lineage walking — spaces are self-contained. Includes all
-- generations; consumers resolve to the current tip per item
-- (item_current) for the default view, and join action_resolved
-- if they need the derived generation number.
-- ============================================================
CREATE VIEW space_history AS
SELECT
    a.id                AS action_id,
    a.space_id,
    a.participant_id,
    a.action_type,
    a.status,
    a.intent,
    a.item_id,
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
WHERE cl.state IN ('spending', 'spent');
