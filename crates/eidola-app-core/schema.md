# Eidola client schema

## Orientation

The client database is organized in three layers, each with a distinct concern.

**Layer 0 (Wallet)** manages anonymous credit tokens — the credentials that fund inference and tool calls without linking payment identity to usage. This layer is mature and largely independent of the others; its only touchpoint is the `credential_nonce` FK on `request`.

**Layer 1 (Transport)** records the raw facts of network communication: which provider was contacted, over what channel, with what attestation evidence, and the exact bytes sent and received. This is the audit trail.

**Layer 2 (Semantic)** captures the logical structure of interaction — who did what, why, in what context, and how context was assembled for each inference call. This is where the interesting design lives, and the focus of this document.

## Core concepts

### Spaces

A space is a namespace for context accumulation. Think of it as a room: participants join, actions occur, and shared history builds up. A space is *not* a conversation, a thread, or a session — those are interaction patterns that emerge from how actions are arranged within a space, not properties of the space itself.

Every space carries a `linkability` tier (`linked`, `unlinked`, `public`) that governs the privacy posture of interactions within it. Spaces form a navigational DAG via `parent_space_id` — you can trace that a space was derived from another — but this is metadata for UI breadcrumbs and organizational queries, not a content boundary. A child space does not automatically inherit its parent's actions. (How inherited context works is covered under "Origin references" below.)

Spaces can be `archived_at` to mark them as inactive. Archived spaces and their actions remain queryable.

### Participants

A participant is an actor that can emit actions: a human, an agent instance, a tool, or the system itself. Two agent instances backed by the same model are distinct participants — the primary agent and a sub-agent have separate identities, separate `space_participant` memberships, and separately attributable actions.

The `space_participant` junction table tracks who is present in a space and with what role (`owner`, `member`, `observer`). The `joined_at`/`left_at` timestamps support participants entering and leaving mid-interaction.

### Actions

An action is the fundamental unit of the schema. It records: "participant P did thing X in space S at time T." Actions are immutable once they reach a terminal status (`complete`, `cancelled`, `error`).

Not all actions introduce content. Some are structural signals. A `request` action asks another participant to act. A `checkpoint` action marks a milestone in a long-running workflow. A `publish` action makes prior draft actions visible. A `decision` action records an agent's routing or planning step. These structural action types are the mechanism for orchestration — where a traditional system might use a "task" entity with lifecycle state, we use actions whose types and antecedent relationships encode the same information without a separate abstraction.

An action's `status` field tracks its lifecycle: `draft` (created but not yet visible to other participants), `streaming` (being generated, content is partial), `complete`, `cancelled` (interrupted), or `error`. The `space_history` view filters to terminal statuses, so draft and streaming actions are invisible to queries against the space's accumulated context.

The optional `intent` field carries a natural-language description of purpose. Most actions won't have one. When present, it serves the same role a "task name" would in a task-oriented schema — it's metadata that aids human and LLM comprehension of what a cluster of actions was trying to accomplish.

### Content blocks

An action's payload is an ordered sequence of typed content blocks. This normalizes the content-array concept shared by the Anthropic, OpenAI, and Gemini APIs (each uses different shapes — `content`, `tool_calls` + `content`, and `parts` respectively). Content blocks are the provider-agnostic queryable layer; the raw provider-specific format lives in `request.response_body`.

Block types include `text`, `thinking`, `tool_use`, `tool_result`, `image`, `document`, `code`, `error`, and `other`. The `tool_name` and `tool_call_id` fields are denormalized onto tool-related blocks for query convenience — "find all invocations of function X" should be a simple indexed lookup, not a JSON parse.

### Antecedents

The `action_antecedent` table encodes causal structure: "this action happened because of (was predicated on) that action." An action may have zero antecedents (a spontaneous user message, a scheduled event), one (the common case in turn-based interaction), or several (a synthesis action drawing on multiple prior actions).

This is explicitly *not* a sequential ordering mechanism. Two actions in the same space may have the same antecedent (they're alternative responses to the same prompt), or no shared antecedents at all (independent parallel work). Wall-clock ordering for display comes from `created_at` (UUIDv7 provides monotonic timestamps); the antecedent graph captures *why* an action exists, not *when*.

Antecedent edges carry optional range fields: `content_block_id`, `range_start`, and `range_end` can pinpoint a specific substring within a specific content block of the antecedent action. This supports inline annotation, quoting, and code-review-style commenting — "this action is predicated on characters 10–25 of block cb4 in action A4." The optional `annotation` field provides human-readable rationale for the link. These fields are informational; they don't affect context assembly.

The `consequent_tree` view walks the antecedent graph forward via recursive CTE, producing the transitive closure of all actions consequent upon a given root. This is how you answer "show me everything that flowed from this action" — useful for impact analysis, cost attribution across a workflow, and understanding how a single user message rippled through the system.

## Context assembly

Context assembly records exactly what was composed into the prompt for each inference action. It answers a different question than the antecedent graph: antecedents record "why did this action happen?" while context assembly records "what information was available when it happened?"

Each inference action gets one `context_assembly` row, which captures the system prompt (with a hash for deduplication and the full text for audit), retrieval augmentation references, total token count, and whether truncation was applied.

The `context_assembly_action` junction table lists which prior actions were included in the prompt, in order. This junction has no space constraint — it can reference actions from any space. This is a designed capability: a scheduled "dreaming" agent might scan actions across many spaces to synthesize insights; a sub-agent working in a scratch space might have its context assembled from actions in the parent space. Cross-space reads happen here, at the operational layer, while the antecedent graph remains local to the space where work is being done.

## Transport and audit

`request` records raw HTTP request/response pairs. Its FK points *from* `request` *to* `action`, supporting one-to-many: a single action may have multiple request rows (retries), but each request belongs to at most one action. The `retry_of_id` and `attempt_number` fields make the retry chain explicit.

`connection` captures the transport channel and, when applicable, the attestation evidence anchoring trust in that channel. A clearnet connection has no attestation fields; a Tor connection through a TEE has an attestation document, its hash, and the platform measurements digest.

The chain from logical to physical is: `action` ← `request` → `connection` → `provider`. For billing: `request.credential_nonce` links to the credential that funded the request, and the `spend_trail` view joins this all the way through to the space and action type.

## Origin references and edit-and-fork

Actions are immutable once terminal. If a user wants to "edit" a completed action, the system forks: it creates a new space, populates it with origin-reference actions for the history that should carry over, and then inserts the edited action as new original work.

An origin-reference action has `origin_action_id` set to the ID of the action it mirrors. It carries no content blocks of its own — queries follow the FK to get content from the original. It carries no cost attribution — a CHECK constraint structurally prevents origin references from having non-null token or credit fields. The `action_resolved` view transparently dereferences origin pointers, so downstream queries see content and cost information regardless of whether an action is original or a reference.

This means cost accounting requires no special logic: `WHERE origin_action_id IS NULL` excludes references, and the `spend_trail` view applies this filter automatically.

For v1, we expect spaces to fork only at leaf positions (creating a fresh scratch space for a sub-agent, with no inherited history to carry over). The `origin_action_id` machinery exists in the schema but is dormant until we implement the edit-and-fork UI. No migration will be needed — all current actions have `origin_action_id = NULL`.

## Patterns

These are not prescriptive; they illustrate how the schema's primitives compose.

**Turn-based chat.** A single space, two participants (human + agent). Each user message is a `user_input` action; each agent response is an `inference` action with one antecedent pointing to the preceding user message. If the agent calls tools, the sequence is `inference` (containing a `tool_use` content block) → `tool_result` → `inference` (final response), each predicated on its predecessor.

**Multi-turn follow-ups.** A second user question in the same space is a new action predicated on the prior agent response. Context assembly for the new inference includes all prior actions in the space. The antecedent chain captures the causal thread; the space captures the accumulated context.

**Batch review.** A user drafts several `user_input` actions with `status = 'draft'`, each predicated on a specific prior action (inline comments on specific sections). A `publish` action predicated on all the drafts makes them visible and serves as the summary or cover letter. The agent responds to each comment individually (each response predicated on one draft) and then produces a synthesis (predicated on all its own responses plus the publish action).

**Sub-agent in a scratch space.** The primary agent creates a child space with `parent_space_id` pointing to the main space. A sub-agent participant is added. The sub-agent's inference actions live in the child space, but their `context_assembly` references actions from the parent space. When the sub-agent finishes, the primary agent creates a new action in the main space whose antecedent points to its own earlier action (the one that triggered the sub-agent), and whose `context_assembly` includes the sub-agent's results from the child space.

**Scheduled dreaming.** A `system` action with no antecedents triggers an agent in a dedicated space. The agent's `context_assembly` draws from actions across multiple other spaces (cross-space read). Its outputs (tool calls to update an insights store, or new actions summarizing findings) exist only in the dreaming space. No other space is modified.

**Regeneration.** User is unhappy with a response. The system creates a new space with `parent_space_id` pointing to the current one, inserts origin-reference actions for the history up to the point of regeneration, and archives the original space. The new inference in the new space has the same antecedent as the rejected response — both are alternative continuations from the same point. The original response and everything that followed it remain in the archived space for audit.

**Streaming cancellation.** An inference action in `streaming` status is interrupted. Its status becomes `cancelled`. Its content blocks contain whatever was generated before interruption. A subsequent user action predicated on the cancelled action can reference it (the agent may benefit from seeing what was already started), or can share the same antecedent as the cancelled action (ignoring it entirely). This is a context assembly decision, not a schema constraint.

## Conventions

All primary keys are UUIDv7, providing monotonic wall-clock ordering within a single client process. Timestamps are unix timestamps. JSON fields (request/response headers, retrieval refs, structured data in content blocks) are stored as TEXT. Binary payloads (request/response bodies, media, attestation documents) are stored as BLOB.

Credit amounts are denominated in credits, nominally equivalent to micro-dollars but often purchased at a discount. The term "credits" describes value consumed from credentials (anonymous credit tokens) and avoids conflation with inference tokens or monetary amounts.

Action types and statuses use inline TEXT CHECK constraints rather than reference tables. This is deliberate: an LLM querying the schema sees the vocabulary directly in the table definition, which substantially improves query generation accuracy without requiring joins through lookup tables.
