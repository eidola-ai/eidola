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

-- ============================================================
-- Backend: a *configured* inference destination — where an ask
-- can be routed. Distinct from `provider` (below), which is the
-- append-only forensic identity of whoever serviced a
-- connection: a backend is user configuration (mutable,
-- enable/disable-able, removable); a provider row is a record.
--
-- A backend row describes how to reach and trust that backend.
--
-- kind:
--   eidola    the confidential Eidola service (singleton). Its
--             base_url + trust columns (trusted_measurements,
--             hardware_root_ca, hardware_intermediate_ca) are the
--             connection + trust bundle; each is NULL by default,
--             which means "use the embedded trust-root pin baked
--             into this build."
--   local     Eidola-managed on-device models (singleton; the
--             models live in <data_dir>/models)
--   openai    any OpenAI-compatible HTTP server the user
--             configures (base_url + optional api_key)
--   llamacpp  a user-owned llama.cpp install: Eidola scans
--             models_dir and starts/stops llama-server engines,
--             but does NOT manage (download/delete) the models
--
-- model_overrides: JSON array of model ids. "OpenAI-compatible"
-- does not guarantee GET /v1/models (Azure's deployment model,
-- scoped gateway keys, partial proxy listings), so a backend's
-- model list can be pinned manually; NULL = trust the listing.
--
-- engine_path: for 'llamacpp' backends only, an explicit path to
-- the user's llama-server binary; NULL = discover it ($PATH, then
-- the usual install prefixes). The managed 'local' engine is the
-- bundled sidecar and never reads this column.
--
-- auto_start: for 'llamacpp' backends only, whether a request may
-- start an engine on demand (1) or must be pre-loaded explicitly
-- (0). The 'local' backend always auto-starts (it's ours).
--
-- trusted_measurements: for the 'eidola' backend only, a JSON
-- array of enclave-measurement overrides ({snp_measurement,
-- tdx_measurement:{rtmr1,rtmr2}}). NULL/absent = the single build
-- measurement pinned in the trust root.
--
-- hardware_root_ca / hardware_intermediate_ca: for the 'eidola'
-- backend only, PEM certificate overrides for the AMD/Intel
-- attestation chain (the dev-shim ARK/ASK). NULL = the vendor
-- chain baked into the verifier.
--
-- removed_at: soft delete. Forensic rows (request.backend_id)
-- keep a valid FK target forever; re-adding the same id revives
-- the row.
-- ============================================================
CREATE TABLE backend (
    id              TEXT PRIMARY KEY,          -- user-visible slug
    kind            TEXT NOT NULL CHECK (kind IN (
                        'eidola', 'local', 'openai', 'llamacpp'
                    )),
    display_name    TEXT NOT NULL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    base_url        TEXT,
    api_key         TEXT,
    models_dir      TEXT,
    model_overrides TEXT,                      -- JSON array
    engine_path     TEXT,                      -- llamacpp: explicit binary path
    auto_start      INTEGER NOT NULL DEFAULT 1, -- llamacpp: start engines on request
    trusted_measurements     TEXT,             -- eidola: JSON array of measurement overrides
    hardware_root_ca         TEXT,             -- eidola: PEM ARK override
    hardware_intermediate_ca TEXT,             -- eidola: PEM ASK override
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    removed_at      INTEGER
);

-- The eidola and local backends are singletons; externally
-- configured kinds may have any number of rows.
CREATE UNIQUE INDEX idx_backend_singleton
    ON backend (kind)
    WHERE kind IN ('eidola', 'local');

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

-- Participants are SCOPE-OWNED (Participants v1, spec §0). Every
-- participant row has exactly one scope; the config columns live
-- ONLY on `participant`, so cross-table drift is impossible rather
-- than managed. `space` and `space_template` are defined first
-- because `participant` references both as owner FKs.

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
    -- Navigational in exactly the sense parent_space_id is, one level
    -- finer: "this space was derived from that *post*". Written only by
    -- the sub-space spawn door, which is reached from inside a turn and
    -- so knows the post being answered when the delegation was opened.
    -- It is what a delegation's report attaches to, so the answer stays
    -- on the branch the work was asked for on rather than wherever the
    -- owning agent happened to speak last. Fixed at birth and never
    -- updated: it records a fact about the space's origin, not its state.
    parent_action_id  TEXT REFERENCES action(id),
    -- Which *turn* opened this room, recorded as the item that turn will
    -- write its answer under. The anchor above says which post the work was
    -- asked on; it cannot say which of that post's answers the report belongs
    -- beneath, because nothing serializes two turns of one agent against one
    -- post — a second explicit ask, or a regeneration running beside a reply,
    -- answers the same anchor from the same owner. An item does say it:
    -- `prepare_turn` mints it before the turn's first request (a turn that is
    -- capped or budget-stopped writes no inference at all, so there is no
    -- answer id yet to record), and a regeneration reuses the item it revises.
    -- Written here, inside the spawn's own transaction, so the fact commits
    -- with the room and outlives the process that opened it: a delegation runs
    -- for as long as its work takes, and the app being quit in the middle of
    -- one is ordinary rather than exceptional.
    -- No foreign key: at the spawn that answer does not exist, so no row
    -- carries the item yet. Fixed at birth, like the anchor beside it.
    -- NULL for a spawn with no turn behind it (a direct API caller), whose
    -- report falls back to the owner's newest answer on the anchor.
    parent_answer_item_id TEXT,
    title             TEXT,
    linkability       TEXT NOT NULL CHECK (linkability IN (
                          'linked', 'unlinked', 'public'
                      )),
    -- The first real space setting: the cascade guard (wave 2). At this
    -- many auto-notified turns from one triggering post, planning pauses.
    -- Seeded from the template a space is instantiated from (default 4);
    -- the PoC for the future copy-from-template space-settings surface.
    cascade_limit     INTEGER NOT NULL DEFAULT 4,
    -- The may-decline router's model (task 22): a qualified
    -- `<model>@<backend>` reference to the small model that filters the
    -- *mechanical* notify set down to who should actually respond.
    -- NULL (the default) means the feature is OFF — the mechanical
    -- notify policies decide alone and no router call is ever made.
    -- Copied from the template a space is instantiated from, exactly like
    -- cascade_limit. A local (engine-backed) reference is free; a remote
    -- one bills a normal inference per triggering post.
    router_model      TEXT,
    -- Notebook space (task 36): when set, this space IS that global
    -- agent's private notebook — the residence of its core memory
    -- blocks and the stage for self-dialogue. A real space, so
    -- item space-scoping stays load-bearing and versioning,
    -- references, rendering and the Record all work unchanged. One
    -- column rather than a `kind` because both consumers need to
    -- know *whose* it is, not merely that it is one: the default
    -- Library listing hides it, and the agent-management surface
    -- opens it. Created inside the promotion transaction; NULL for
    -- every ordinary space. (Forward reference to `participant`,
    -- which is defined below — FK targets resolve at DML time.)
    notebook_participant_id TEXT REFERENCES participant(id),
    created_at        INTEGER NOT NULL,
    archived_at       INTEGER,
    -- The pristineness stamp: NULL means nothing has ever changed this space's
    -- own configuration since it was instantiated. Set (first-write-wins, so
    -- the value is the moment it stopped being untouched) by every write that
    -- alters the space's CONFIGURATION FOOTPRINT — the columns above, its
    -- `space_participant` rows, and the `participant` rows it owns. That is
    -- exactly the footprint a pristine space's disposal deletes.
    --
    -- **Actions deliberately do not stamp it.** A post, an inference, a trace,
    -- a memory revision or a branch summary is its own witness: the disposal
    -- predicate refuses on any `action` row in the space, so stamping the hot
    -- write path would buy a second statement per action for a fact the first
    -- leg already knows.
    --
    -- **Instantiation is not a change.** `instantiate_template` writes this
    -- column last, inside its own transaction, resetting whatever its copies
    -- stamped: a fresh instantiation is pristine by definition. The one thing
    -- it can carry that is not is a caller-supplied `title`, which is a user
    -- saying what this conversation is for, so a titled creation is born
    -- stamped.
    touched_at        INTEGER,
    -- The action named above must live in the parent named beside it.
    -- A single-column FK only proves the row exists; this tuple is the
    -- navigational fact the column records. MATCH SIMPLE: an ordinary
    -- space (both NULL) and an anchorless spawn (action NULL) skip it.
    FOREIGN KEY (parent_action_id, parent_space_id)
        REFERENCES action (id, space_id)
);

-- One notebook per agent.
CREATE UNIQUE INDEX idx_space_notebook_participant
    ON space (notebook_participant_id)
    WHERE notebook_participant_id IS NOT NULL;

-- ============================================================
-- Space capabilities: the attenuation snapshot.
--
-- An agent-spawned sub-space (space.parent_space_id) holds the
-- capabilities its spawner held in the parent at the moment of
-- the spawn, and never more. The rows are written inside the
-- spawning transaction and are immutable afterwards: a capability
-- gained in the parent later does not reach a sub-space that
-- already exists, and the remedy for a grant someone regrets is
-- archiving the sub-space rather than editing this table.
--
-- Absence of a row IS absence of the capability, which is what
-- makes "a sub-space holding a grant its spawner lacked" an
-- invalid state checkable once, at mint, against data — rather
-- than a rule every future consumer has to remember. A grant is
-- COPIED from the parent's row, never composed by the requester,
-- so `config` cannot be widened on the way down either.
--
-- The table is empty in practice today: nothing in the harness is
-- gated on it yet (tool availability is fully derived from the
-- registry, the backend probe and the participant's scope). The
-- shape is the point — a future capability arrives as a `name` +
-- `config` row flowing through the spawn-time check instead of as
-- a new security surface.
-- ============================================================
CREATE TABLE space_capability (
    space_id  TEXT NOT NULL REFERENCES space(id),
    name      TEXT NOT NULL,
    -- JSON, capability-specific; `{}` when the capability is a bare
    -- grant with nothing to configure.
    config    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (space_id, name)
);

-- ============================================================
-- Space template: a reusable blueprint for new spaces. A DB-backed
-- registry (soft-remove, backend-registry style) so a config
-- pointing at a removed template fails honestly and revives
-- cleanly. Distinct from spaces — templates have no actions/posts,
-- no linkability, no archival; they are edited rather than
-- appended to. "Template from space" is a projection between the
-- two types (template_from_space), never a type mutation.
--
-- A template OWNS participant rows exactly like a space does
-- (participant.scope = 'template'); cascade_limit and router_model are
-- copied into a space at instantiation. removed_at is the soft delete.
-- ============================================================
CREATE TABLE space_template (
    id            TEXT PRIMARY KEY,            -- UUIDv7 (well-known for the seeded default)
    title         TEXT NOT NULL,
    cascade_limit INTEGER NOT NULL DEFAULT 4,
    -- The may-decline router model (see space.router_model). NULL = off.
    router_model  TEXT,
    created_at    INTEGER NOT NULL,
    removed_at    INTEGER                      -- soft delete
);

-- ============================================================
-- Participant: an actor that can emit actions into a space.
--
-- SCOPE-OWNED (spec §0). Every row has exactly one `scope`:
--   global    the shared library — today just "User"; later the
--             agent library and other humans. Owns no space/template.
--   space     a per-space instance, born by copying a template's
--             owned rows (owner_space_id = the space).
--   template  a per-template instance — templates own participant
--             rows just like spaces (owner_template_id = the template).
-- The three-way CHECK ties scope to exactly the matching owner
-- column, so an invalid ownership shape is unrepresentable.
--
-- The CONFIG COLUMNS LIVE ONLY HERE (kind, label, model_ref,
-- system_prompt, notify_policy) — drift across tables becomes
-- impossible rather than managed. Reference tables
-- (space_participant / space_template_participant) may point only
-- at globals and carry per-membership overrides.
--
--   model_ref     qualified `<model>@<backend-id>` (bare = eidola
--                 sugar, per backends.rs) the agent answers with
--   system_prompt the agent's system prompt
--   notify_policy auto-response policy (wave 2 plan_notifications):
--                 'explicit' (only when asked) / 'human' (when a
--                 human posted) / 'all' (always)
--   role          membership role of an OWNED participant
--                 (referenced globals carry their role on the
--                 reference row instead)
--   removed_at    global: library soft-remove; owned: left/deactivated
--
-- UNIQUE(id, scope) is the target the pinned composite-FK echoes on
-- space_participant / space_template_participant / action reference.
-- ============================================================
CREATE TABLE participant (
    id                TEXT PRIMARY KEY,        -- UUIDv7
    scope             TEXT NOT NULL CHECK (scope IN ('global', 'space', 'template')),
    owner_space_id    TEXT REFERENCES space(id),
    owner_template_id TEXT REFERENCES space_template(id),
    kind              TEXT NOT NULL CHECK (kind IN (
                          'human', 'agent', 'tool', 'system'
                      )),
    label             TEXT NOT NULL,
    model_ref         TEXT,
    system_prompt     TEXT,
    notify_policy     TEXT NOT NULL DEFAULT 'explicit'
                      CHECK (notify_policy IN ('explicit', 'human', 'all')),
    role              TEXT NOT NULL DEFAULT 'member'
                      CHECK (role IN ('owner', 'member', 'observer')),
    provider_id       TEXT REFERENCES provider(id),
    created_at        INTEGER NOT NULL,
    removed_at        INTEGER,

    CHECK ((scope = 'global'   AND owner_space_id IS NULL     AND owner_template_id IS NULL)
        OR (scope = 'space'    AND owner_space_id IS NOT NULL AND owner_template_id IS NULL)
        OR (scope = 'template' AND owner_space_id IS NULL     AND owner_template_id IS NOT NULL))
);

-- The composite-FK target for the pinned `(participant_id, participant_scope)`
-- echoes on the reference tables and `action`.
CREATE UNIQUE INDEX idx_participant_id_scope ON participant (id, scope);
CREATE INDEX idx_participant_owner_space ON participant (owner_space_id)
    WHERE owner_space_id IS NOT NULL;
CREATE INDEX idx_participant_owner_template ON participant (owner_template_id)
    WHERE owner_template_id IS NOT NULL;

-- ============================================================
-- Space membership: a space's participants = its OWNED rows
-- (participant.owner_space_id = space) ∪ the GLOBALS it references
-- here. This table holds references to globals ONLY, made
-- declarative by the pinned echo + composite FK: `participant_scope`
-- is CHECK-pinned to 'global' and the tuple FK points at
-- participant(id, scope), so an owned participant can never be
-- smuggled into a reference row.
--
-- Overrides mirror the config columns, per-membership: NULL =
-- inherit the global's config; '' = override to empty. Effective
-- config = COALESCE(override, participant config). `role` is the
-- membership role for the referenced global (owned rows carry role
-- on the participant row).
-- ============================================================
CREATE TABLE space_participant (
    space_id               TEXT NOT NULL REFERENCES space(id),
    participant_id         TEXT NOT NULL,
    participant_scope      TEXT NOT NULL CHECK (participant_scope = 'global'),
    override_label         TEXT,               -- NULL = inherit; '' = override to empty
    override_model_ref     TEXT,               -- NULL = inherit; '' = override to empty
    override_system_prompt TEXT,               -- NULL = inherit; '' = override to empty
    override_notify_policy TEXT CHECK (override_notify_policy IS NULL
                               OR override_notify_policy IN ('explicit', 'human', 'all')),
    role                   TEXT NOT NULL DEFAULT 'member'
                           CHECK (role IN ('owner', 'member', 'observer')),
    joined_at              INTEGER NOT NULL,
    left_at                INTEGER,

    PRIMARY KEY (space_id, participant_id),
    FOREIGN KEY (participant_id, participant_scope)
        REFERENCES participant (id, scope)
);

-- ============================================================
-- Space template membership: identical shape keyed on template_id.
-- Templates reference globals here (with overrides) and own their
-- agents via participant.scope = 'template'.
-- ============================================================
CREATE TABLE space_template_participant (
    template_id            TEXT NOT NULL REFERENCES space_template(id),
    participant_id         TEXT NOT NULL,
    participant_scope      TEXT NOT NULL CHECK (participant_scope = 'global'),
    override_label         TEXT,               -- NULL = inherit; '' = override to empty
    override_model_ref     TEXT,               -- NULL = inherit; '' = override to empty
    override_system_prompt TEXT,               -- NULL = inherit; '' = override to empty
    override_notify_policy TEXT CHECK (override_notify_policy IS NULL
                               OR override_notify_policy IN ('explicit', 'human', 'all')),
    role                   TEXT NOT NULL DEFAULT 'member'
                           CHECK (role IN ('owner', 'member', 'observer')),
    joined_at              INTEGER NOT NULL,
    left_at                INTEGER,

    PRIMARY KEY (template_id, participant_id),
    FOREIGN KEY (participant_id, participant_scope)
        REFERENCES participant (id, scope)
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
    -- The acting participant, referenced by the pinned composite echo. An
    -- action can only be authored by a GLOBAL (e.g. the shared "User") or a
    -- SPACE-owned participant — never a template-owned one — enforced by the
    -- CHECK + tuple FK, not convention.
    participant_id     TEXT NOT NULL,
    participant_scope  TEXT NOT NULL CHECK (participant_scope IN ('global', 'space')),

    -- generation identity (generation number is derived, not stored)
    item_id              TEXT NOT NULL,
    supersedes_action_id TEXT,
    supersedes_item_id   TEXT,

    action_type     TEXT NOT NULL CHECK (action_type IN (
                        'user_input',
                        'inference',
                        -- A post an agent wrote directly rather than by
                        -- inferring it: today the brief that opens an
                        -- agent-spawned sub-space, written by the owning
                        -- agent inside the spawning transaction. It is a
                        -- POST (it renders, it is replied to, it notifies,
                        -- it may be quoted) and it is neither of the two
                        -- above: `user_input` would attribute it to a human
                        -- on every surface that maps that type to the human
                        -- role, and `inference` would claim a model call and
                        -- a spend that never happened.
                        'brief',
                        'tool_call',
                        'tool_result',
                        'retrieval',
                        'request',
                        'checkpoint',
                        'decision',
                        -- One generation of an agent's memory block
                        -- (task 35). Not a post type, so it collapses
                        -- out of every render, tree and context query
                        -- exactly like a tool trace.
                        'memory',
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

    -- The upstream stopped this generation at the completion ceiling
    -- (`finish_reason: "length"`) rather than because the model was
    -- done — the durable half of "this answer reached its length
    -- limit", so a reader who reopens the space still sees the mark
    -- under an answer that stops mid-thought.
    --
    -- A COLUMN, not a `status` value, and the distinction is load
    -- bearing. `status` is the lifecycle slot: one value at a time,
    -- and the reads that mean "durable and renderable" spell it
    -- `status IN ('complete', 'cancelled')` in a dozen places (the
    -- tree, the transcript, context assembly, search, the record). A
    -- truncated answer IS complete in exactly that sense — it
    -- committed, it renders, it is context for the next turn — so a
    -- 'truncated' status would drop it out of every one of those
    -- reads until each was widened, and the cost of missing one is a
    -- real answer vanishing from a conversation. How generation
    -- stopped is an orthogonal fact about a completed action, so it
    -- gets its own field.
    --
    -- Only an inference has a `finish_reason` to read, so only an
    -- inference may carry the flag.
    truncated       INTEGER NOT NULL DEFAULT 0 CHECK (truncated IN (0, 1)),

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
    CHECK (truncated = 0 OR action_type = 'inference'),
    CHECK ((supersedes_action_id IS NULL) = (supersedes_item_id IS NULL)),
    CHECK (supersedes_item_id IS NULL OR supersedes_item_id = item_id),
    FOREIGN KEY (supersedes_action_id, supersedes_item_id)
        REFERENCES action (id, item_id),
    -- ON UPDATE CASCADE carries task 36's in-place promotion: an agent
    -- is promoted to a global identity by flipping scope 'space' →
    -- 'global' on the SAME participant row, and the pinned echo on
    -- every past action follows declaratively rather than by an
    -- app-layer rewrite. This is legitimate, not a forensics edit: the
    -- echo is a constraint device (its job — "authored by a
    -- template-owned participant is unrepresentable" — holds just as
    -- well afterwards), while participant_id, the actual identity in
    -- the trail, never mutates. Proven under this turso build by
    -- `turso_enforcement_smoke` case (e).
    FOREIGN KEY (participant_id, participant_scope)
        REFERENCES participant (id, scope) ON UPDATE CASCADE
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

-- Parent key for space (parent_action_id, parent_space_id) → action.
CREATE UNIQUE INDEX idx_action_id_space ON action (id, space_id);

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
-- Memory block: an agent's own notes (task 35).
--
-- A block IS an item: its text and its authorship live on the
-- item's action generations (action_type = 'memory'), so a
-- revision supersedes exactly like an edited post, the whole
-- history stays readable, and every revision records WHO wrote
-- it — which is what distinguishes a self-revision from a human
-- correction structurally rather than by convention.
--
-- This row carries what the generations cannot: the block's
-- identity. Ownership and scope are deliberately separate:
--
--   owner_participant_id  the agent whose memory this is — ONE
--                         owner, always, pinned by the same
--                         (id, scope) composite echo `action`
--                         uses. Blocks are never co-owned by a
--                         space, which is what makes promoting
--                         an agent to global (task 36) a no-op
--                         for its memory.
--   scope                 'core'  = load wherever the agent goes
--                         'space' = load only in `space_id`
--                         Scope is ADDRESSING, not ownership.
--   space_id              residence: the space the block is
--                         about, and where its actions live. For
--                         a space-owned agent that is always its
--                         owner space, so v1 needs no separate
--                         residence concept (task 36 adds the
--                         notebook space for a global's core
--                         blocks).
--
-- Names are unique per OWNER, not per (owner, space): the
-- `remember` tool addresses a block by name, so one namespace
-- per agent is what makes "write the name again" unambiguously
-- a revision of the same block, wherever the agent is standing.
-- ============================================================
CREATE TABLE memory_block (
    item_id              TEXT PRIMARY KEY,
    -- Generation 0 of that item. The pair FK proves the root
    -- really is an action of this item.
    root_action_id       TEXT NOT NULL,

    owner_participant_id TEXT NOT NULL,
    owner_scope          TEXT NOT NULL CHECK (owner_scope IN ('global', 'space')),

    name                 TEXT NOT NULL,
    scope                TEXT NOT NULL CHECK (scope IN ('core', 'space')) DEFAULT 'space',
    space_id             TEXT NOT NULL REFERENCES space(id),

    created_at           INTEGER NOT NULL,
    updated_at           INTEGER NOT NULL,

    UNIQUE (owner_participant_id, name),
    FOREIGN KEY (root_action_id, item_id) REFERENCES action (id, item_id),
    -- Same cascade as `action`, and for the same reason: promoting an
    -- agent that already holds memory must not fail the FK or strand a
    -- stale echo. Memory is agent-owned, so promotion is a no-op for
    -- it — the blocks keep their owner, their names, their scope
    -- labels and their residence; only the echo moves.
    FOREIGN KEY (owner_participant_id, owner_scope)
        REFERENCES participant (id, scope) ON UPDATE CASCADE
);

CREATE INDEX idx_memory_block_owner
    ON memory_block (owner_participant_id);

CREATE INDEX idx_memory_block_space
    ON memory_block (space_id);

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

    created_at        INTEGER NOT NULL,

    -- The configured backend this request was routed through, when the
    -- request belongs to one (chat turns; NULL for traffic that isn't
    -- addressed to a configured backend).
    backend_id        TEXT REFERENCES backend(id)
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
    a.truncated,
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
