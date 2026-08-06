//! **Agent memory** (task 35) — the second of the three participant layers.
//!
//! 1. **Charter** — who an agent is asked to be: the human's `system_prompt`,
//!    model and policies. Agents never edit it.
//! 2. **Memory** — who they have become: agent-owned, self-revised,
//!    human-inspectable. This module.
//! 3. **Experience** — what they have lived: the transcript, the traces, the
//!    Record (task 33's first-person rules).
//!
//! Keeping the charter and the memory separate is deliberate. The human's
//! instructions stay authoritative, and the two read as what they are: one is
//! operator configuration, the other is "notes you wrote yourself" — both more
//! credible to a model than a single undifferentiated blob of standing
//! instructions.
//!
//! # A block is an item
//!
//! Storage mirrors the task-27 branch-summary precedent exactly. A memory block
//! is an **item**; each revision is a generation ([`db::MEMORY_ACTION_TYPE`],
//! superseding the item's tip), so the existing generation machinery supplies
//! append-only revision history, a diffable trail for the inspector, and —
//! because every generation records its author — a **structural** distinction
//! between a self-revision and a human correction. `memory` is not a post type,
//! so `get_space_tree`, both context queries and the notification planner's
//! post allowlist collapse it out for free.
//!
//! What a string-blob store cannot do is **provenance**: a revision may carry
//! `reference` antecedent edges (ordinals `1..=N`, ordinal 0 stays empty — a
//! block has no `reply` edge because it is not part of the thread) naming the
//! posts it learned from, so the inspector can ask *why* an agent believes each
//! thing.
//!
//! The `memory_block` row carries what the generations cannot: the block's
//! identity. **Ownership and scope are separate concepts.** Every block has one
//! owner — the agent — and a scope *label*: `core` (loads wherever the owner
//! goes) or `space` (loads in its residence only). Scope is addressing, never
//! co-ownership, which is what makes promoting an agent to a global identity
//! (task 36) a no-op for its memory. Residence is the space the block is about;
//! for a space-owned agent that is always the space it is standing in. A
//! **global** agent's `core` blocks are about no space at all, so they reside
//! in that agent's private notebook space (task 36) — which is what the
//! notebook is for. Residence is decided once, when a block is created, and
//! never moves: promotion therefore leaves every existing block exactly where
//! it was, with the label it had.
//!
//! # The loading rule
//!
//! Per turn, the **responding** participant's memory — core blocks plus the
//! blocks about the current space, and nobody else's, ever — renders at the
//! **head**, inside the system message, after the charter and the protocol
//! notes. Identity governs everything downstream, so it belongs before the
//! conversation; a memory edit therefore invalidates that participant's prefix
//! cache, which is correct and accepted (rare and deliberate — CLAUDE.md
//! semantics). Slow memory at the head, volatile thread map at the tail.
//! Putting it *after* the static notes keeps the charter + notes prefix
//! byte-stable, so a revision moves as few bytes as the rule allows.
//!
//! The bytes ride the `messages` array like everything else, so
//! `eidola_common::prompt_charge` covers them on both sides by construction.
//!
//! # The tool seam
//!
//! [`RememberTool`] is **turn-scoped**, attached in `prepare_turn` on top of
//! the process-registry snapshot — the task-21 navigation-tool precedent rather
//! than the task-20/22 consumer-registration one. The reason is the same one
//! that reserved those names: a `remember` that is not bound to *this* turn's
//! responding participant, residence space and thread snapshot is not the tool
//! the system note promises the model. The process registry's [`Tool`] trait
//! carries no identity (a shared `Arc<dyn Tool>` is called concurrently by
//! every turn in the process), so a process-scoped `remember` could only ever
//! guess whose memory it was writing. The name is therefore reserved
//! ([`crate::tools::RESERVED_TOOL_NAMES`]) and the tool is constructed per turn.
//!
//! **Off by default**, like every other agentic capability here (the router,
//! `decline`): the whole feature — the loading read *and* the tool — is gated
//! on [`crate::AppCore::set_memory_enabled`]. That is what preserves the
//! `tools.rs` invariant that an install which has not opted in sends
//! byte-identical requests. Note the `eidola` backend cannot carry a `tools`
//! field at all (its request type is `deny_unknown_fields`), so a remote turn
//! still *reads* its memory but cannot revise it until task 25 — the same
//! limitation the navigation tools have. A turn that discovers its backend
//! rejects `tools` withdraws `remember` with the rest (`withdraw_auto_tools`)
//! and keeps [`MEMORY_NOTE`] in the already-sent system message, exactly as it
//! does for the thread-map note: the notes describe the affordance, the
//! absence of a schema is what makes it uncallable.
//!
//! # Hygiene
//!
//! The tool is revision-shaped, not append-shaped: `remember(block, text)`
//! replaces a named block's contents by superseding its tip. Copy-on-write
//! makes revising the cheap default, which is the nudge that keeps memory from
//! becoming a log. Two guard rails bound the rest — [`MAX_MEMORY_BLOCKS`] per
//! owner and [`MAX_MEMORY_BLOCK_BYTES`] per block — and a refusal is a *tool
//! result*, honest about the limit and what is already stored, with **zero
//! durable trace**: every check runs before any write.

use std::sync::{Arc, Weak};

use uuid::Uuid;

use crate::db;
use crate::error::AppError;
use crate::tools::{Tool, ToolError, ToolFuture};
use crate::{Change, Inner, ThreadSnapshot, now_ms};

/// The tool name the system note promises the model. Reserved — see the module
/// docs for why it cannot be a process-registry registration.
pub const REMEMBER_TOOL_NAME: &str = "remember";

/// How many blocks one participant may hold. Small on purpose: Letta's
/// production study found agents accrete near-duplicate blocks until retrieval
/// degrades, and a hard ceiling is what turns "write another one" into "revise
/// the right one".
pub const MAX_MEMORY_BLOCKS: usize = 8;

/// Byte budget for one block's contents. Roughly a page — enough for durable
/// facts and preferences, far too small for a transcript.
pub const MAX_MEMORY_BLOCK_BYTES: usize = 2_000;

/// Byte budget for a block name. It is an address, not a sentence.
const MAX_BLOCK_NAME_BYTES: usize = 48;

/// How many provenance edges one revision may carry.
const MAX_SOURCES: usize = 8;

/// The scope label of a block: `core` travels with the owner, `space` stays
/// with its residence.
const SCOPE_CORE: &str = "core";
const SCOPE_SPACE: &str = "space";

/// The note that joins the turn's system message when the tool attaches —
/// telling the model the affordance exists and what belongs in it. (The
/// `<memory>` block's own preamble describes what memory *is*, and renders
/// whether or not the tool is available.)
pub const MEMORY_NOTE: &str = "\
You keep your own memory: a few short, named blocks that are shown to you at \
the start of every turn. Revise one with the `remember` tool — give the \
block's name and its full new contents, which replace what was there. Record \
durable things (preferences, decisions, standing facts, what you have learned \
about the people here), not a log of what happened; the conversation is \
already the record of that. Prefer revising an existing block to adding \
another.";

/// The `<memory>` block's opening line. Static, so it costs the prefix cache
/// nothing.
const MEMORY_PREAMBLE: &str = "Notes you wrote for yourself in earlier turns. They are not part of \
                               the conversation and no one else is shown them. Core notes travel \
                               with you; the rest are about this space.";

/// Render the loaded blocks as the `<memory>` block that goes into the system
/// message. Empty input renders nothing at all (the caller omits the section),
/// which is what keeps a memory-less turn byte-identical.
///
/// Pure and unit-tested: these are wire bytes.
pub(crate) fn render_memory(entries: &[db::MemoryEntryRow]) -> String {
    let mut out = String::from("<memory>\n");
    out.push_str(MEMORY_PREAMBLE);
    out.push('\n');
    for e in entries {
        let scope = if e.scope == SCOPE_CORE {
            "core"
        } else {
            "this space"
        };
        out.push_str(&format!("\n--- {} ({scope}) ---\n{}\n", e.name, e.text));
    }
    out.push_str("</memory>");
    out
}

/// Why a `remember` call was refused. Every variant is a *model* mistake it can
/// correct on the next round, so it is reported as a tool result rather than
/// failing the turn — and every one is decided before any write, so a refusal
/// leaves no trace at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MemoryRefusal {
    EmptyName,
    NameTooLong,
    NameNotOneLine,
    EmptyText,
    TextTooLarge {
        bytes: usize,
    },
    TooManyBlocks {
        existing: Vec<String>,
    },
    /// The owner was retired while its turn was still running. Memory is
    /// durable and its notebook is archived, so the write is refused rather
    /// than landing behind the retirement.
    OwnerRetired,
}

impl std::fmt::Display for MemoryRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyName => write!(
                f,
                "`block` is required and must be a short name, e.g. \"preferences\""
            ),
            Self::NameTooLong => write!(
                f,
                "that block name is too long (limit {MAX_BLOCK_NAME_BYTES} bytes) — a block name \
                 is an address, not a sentence"
            ),
            Self::NameNotOneLine => write!(
                f,
                "a block name must be a single line without control characters"
            ),
            Self::OwnerRetired => write!(
                f,
                "you have been retired, so your memory is closed — nothing was written"
            ),
            Self::EmptyText => write!(
                f,
                "`text` is required and must be the block's full new contents"
            ),
            Self::TextTooLarge { bytes } => write!(
                f,
                "that is {bytes} bytes; one block holds at most {MAX_MEMORY_BLOCK_BYTES}. Keep \
                 only what stays true and write it shorter"
            ),
            Self::TooManyBlocks { existing } => write!(
                f,
                "you already hold the maximum of {MAX_MEMORY_BLOCKS} memory blocks ({}). Revise \
                 one of them by writing its name instead of adding another",
                existing.join(", ")
            ),
        }
    }
}

/// What a committed revision did — the material for the model's tool result.
#[derive(Clone, Debug)]
pub(crate) struct MemoryWrite {
    pub name: String,
    pub scope: String,
    /// 1-based generation number of the revision just written.
    pub revision: usize,
    /// How many blocks the owner holds now.
    pub blocks: usize,
    /// Handles the model named that the turn's snapshot does not know. Stated
    /// in the result rather than silently dropped.
    pub unknown_handles: Vec<String>,
    /// An annotation was supplied with no sources to attach it to.
    pub annotation_dropped: bool,
}

/// The verdict of a `remember` call: the model-facing outcome. A machine
/// failure (the database itself) surfaces as an `AppError` around this.
pub(crate) enum MemoryOutcome {
    Written(Box<MemoryWrite>),
    Refused(MemoryRefusal),
}

/// One resolved `remember` call, as the tool hands it to the write path.
pub(crate) struct RememberRequest {
    pub name: String,
    pub text: String,
    pub scope: String,
    /// Provenance: concrete action ids, already resolved from handles.
    pub sources: Vec<String>,
    pub annotation: Option<String>,
    pub unknown_handles: Vec<String>,
}

/// Normalize a block name, or say why it cannot be one. Names are addresses:
/// one line, trimmed, lowercase-insensitive is *not* imposed (the model's own
/// casing is part of how it reads its notes back).
pub(crate) fn validate_block_name(raw: &str) -> Result<String, MemoryRefusal> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MemoryRefusal::EmptyName);
    }
    // The name is rendered into a one-line delimiter inside the system
    // message, exactly like a participant label in a message header — so the
    // same one-line invariant applies, and for the same reason.
    if trimmed
        .chars()
        .any(|c| c.is_control() || c == '\u{2028}' || c == '\u{2029}')
    {
        return Err(MemoryRefusal::NameNotOneLine);
    }
    if trimmed.len() > MAX_BLOCK_NAME_BYTES {
        return Err(MemoryRefusal::NameTooLong);
    }
    Ok(trimmed.to_string())
}

/// Normalize a block's contents, or say why they cannot be stored.
pub(crate) fn validate_block_text(raw: &str) -> Result<String, MemoryRefusal> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(MemoryRefusal::EmptyText);
    }
    if trimmed.len() > MAX_MEMORY_BLOCK_BYTES {
        return Err(MemoryRefusal::TextTooLarge {
            bytes: trimmed.len(),
        });
    }
    Ok(trimmed.to_string())
}

/// Normalize the scope argument. Anything unrecognized means the safe default:
/// a note about this space only.
pub(crate) fn normalize_scope(raw: Option<&str>) -> String {
    match raw.map(str::trim) {
        Some(s) if s.eq_ignore_ascii_case(SCOPE_CORE) => SCOPE_CORE.to_string(),
        _ => SCOPE_SPACE.to_string(),
    }
}

/// One block as the inspector reads it: its identity, its current contents,
/// and the whole revision trail that produced them.
#[derive(Clone, Debug)]
pub struct MemoryBlockInfo {
    pub item_id: String,
    pub name: String,
    /// `core` or `space`.
    pub scope: String,
    /// Residence — the space this block is about.
    pub space_id: String,
    pub owner_participant_id: String,
    /// The owner's scope at the time of reading (`global` / `space`) — the
    /// pinned composite echo, which task 36's promotion carries across every
    /// block via `ON UPDATE CASCADE`. Owner and residence never move; only
    /// this follows.
    pub owner_scope: String,
    /// The current generation's contents.
    pub text: String,
    pub updated_at: i64,
    /// Oldest first. `len()` is the number of revisions; each entry's author
    /// is what distinguishes a self-revision from a human correction.
    pub revisions: Vec<MemoryRevisionInfo>,
}

/// One generation of a block.
#[derive(Clone, Debug)]
pub struct MemoryRevisionInfo {
    pub action_id: String,
    pub author_participant_id: String,
    pub created_at: i64,
    pub text: String,
    /// Provenance: the concrete post generations this revision learned from,
    /// in ordinal order, each with the annotation it was recorded with.
    pub sources: Vec<MemorySource>,
}

/// One provenance edge of a revision.
#[derive(Clone, Debug)]
pub struct MemorySource {
    pub ordinal: i64,
    pub action_id: String,
    pub annotation: Option<String>,
}

impl Inner {
    /// The participant's memory for a turn in `space_id` — core blocks plus
    /// the blocks about that space, in the order they render.
    pub(crate) async fn load_memory(
        &self,
        conn: &turso::Connection,
        participant_id: &str,
        space_id: &str,
    ) -> Result<Vec<db::MemoryEntryRow>, AppError> {
        db::participant_memory(conn, participant_id, space_id).await
    }

    /// Everything one participant holds, with each block's revision trail —
    /// the read behind [`crate::AppCore::memory_blocks`]. A pure read: no
    /// writes, no emissions.
    pub(crate) async fn memory_blocks(
        &self,
        participant_id: &str,
    ) -> Result<Vec<MemoryBlockInfo>, AppError> {
        let conn = self.db_conn().await?;
        let mut out = Vec::new();
        for block in db::memory_blocks_owned(&conn, participant_id).await? {
            let mut revisions = Vec::new();
            for r in db::memory_revisions(&conn, &block.item_id).await? {
                let sources = db::reference_antecedents(&conn, &r.action_id)
                    .await?
                    .into_iter()
                    .map(|e| MemorySource {
                        ordinal: e.ordinal,
                        action_id: e.antecedent_action_id,
                        annotation: e.annotation,
                    })
                    .collect();
                revisions.push(MemoryRevisionInfo {
                    action_id: r.action_id,
                    author_participant_id: r.author_participant_id,
                    created_at: r.created_at,
                    text: r.text,
                    sources,
                });
            }
            out.push(MemoryBlockInfo {
                item_id: block.item_id,
                name: block.name,
                scope: block.scope,
                space_id: block.space_id,
                owner_participant_id: block.owner_participant_id,
                owner_scope: block.owner_scope,
                text: revisions.last().map(|r| r.text.clone()).unwrap_or_default(),
                updated_at: block.updated_at,
                revisions,
            });
        }
        Ok(out)
    }

    /// Whether agent memory is switched on for this process.
    pub(crate) fn memory_enabled(&self) -> bool {
        self.memory_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Commit one revision of a named block, creating it if the owner has none
    /// by that name.
    ///
    /// Serialized per process on `memory_gate`: two concurrent turns of the
    /// same participant revising one block would otherwise both read the same
    /// tip and race the unique-successor index, and a model does not need to
    /// see that.
    ///
    /// Every refusal is decided before the first write, so a refused call
    /// leaves no action, no block row and no emission.
    pub(crate) async fn remember(
        &self,
        owner_participant_id: &str,
        space_id: &str,
        req: RememberRequest,
    ) -> Result<MemoryOutcome, AppError> {
        let name = match validate_block_name(&req.name) {
            Ok(n) => n,
            Err(r) => return Ok(MemoryOutcome::Refused(r)),
        };
        let text = match validate_block_text(&req.text) {
            Ok(t) => t,
            Err(r) => return Ok(MemoryOutcome::Refused(r)),
        };

        let conn = self.db_conn().await?;
        let _gate = self.memory_gate.lock().await;

        let owned = db::memory_blocks_owned(&conn, owner_participant_id).await?;
        let existing = owned.iter().find(|b| b.name == name).cloned();
        if existing.is_none() && owned.len() >= MAX_MEMORY_BLOCKS {
            return Ok(MemoryOutcome::Refused(MemoryRefusal::TooManyBlocks {
                existing: owned.into_iter().map(|b| b.name).collect(),
            }));
        }

        // Where a *new* block lives. A block resides in the space it is about,
        // which for a space-owned agent is always the space it is standing in.
        // For a **global** agent (task 36) a `core` block is not about any
        // space — it travels with the identity — so it resides in that agent's
        // private notebook, which is exactly what the notebook exists for.
        // Everything else, global or not, stays where it was written.
        // The owner's scope is read **here**, not carried from the turn: a
        // promotion can land mid-turn, and residence should follow what the
        // agent is now rather than what it was when the turn was prepared.
        let owner_is_global = db::get_participant(&conn, owner_participant_id)
            .await?
            .is_some_and(|p| p.scope == "global");
        let new_residence = if req.scope == SCOPE_CORE && owner_is_global {
            db::notebook_space_for(&conn, owner_participant_id)
                .await?
                .unwrap_or_else(|| space_id.to_string())
        } else {
            space_id.to_string()
        };

        let now = now_ms();
        let action_id = Uuid::now_v7().to_string();
        // Residence never moves: a block stays in the space it is about, so a
        // revision written from anywhere lands where the block already lives.
        // (Which is also why promotion is a no-op for memory — an existing
        // space block keeps its label *and* its home.)
        let (item_id, supersedes, residence, revision) = match &existing {
            Some(b) => {
                let tip = db::current_tip_of_item(&conn, &b.space_id, &b.item_id)
                    .await?
                    .ok_or_else(|| AppError::Database {
                        message: format!("memory block `{name}` has no current generation"),
                    })?;
                let revision = db::memory_revisions(&conn, &b.item_id).await?.len() + 1;
                (b.item_id.clone(), Some(tip), b.space_id.clone(), revision)
            }
            None => (Uuid::now_v7().to_string(), None, new_residence, 1usize),
        };

        // **The owner's liveness is asked inside the transaction that carries
        // the writes.** A turn binds its tools to the responding participant
        // when it starts, so a retirement can land between two rounds — and the
        // scope read above is not liveness. Asked out here it would be another
        // read-then-write window; asked in there, no retirement can commit
        // between the question and the answer, and a refusal leaves nothing at
        // all (Codex review, PR #279).
        conn.execute("BEGIN", ()).await.map_err(AppError::db)?;
        let written: Result<Option<usize>, AppError> = async {
            if !db::participant_is_live(&conn, owner_participant_id).await? {
                return Ok(None);
            }
            db::insert_action(
                &conn,
                &db::ActionEntry {
                    id: action_id.clone(),
                    space_id: residence.clone(),
                    // The **author** of this generation — which in v1 is always
                    // the owner, and is the field a later human correction would
                    // differ in.
                    participant_id: owner_participant_id.to_string(),
                    item_id: item_id.clone(),
                    supersedes_action_id: supersedes,
                    action_type: db::MEMORY_ACTION_TYPE.to_string(),
                    status: "complete".to_string(),
                    // The block's name as of this revision — readable in the
                    // Record without a join. The `memory_block` row is the
                    // authority.
                    intent: Some(name.clone()),
                    model: None,
                    input_tokens: None,
                    output_tokens: None,
                    credits_consumed: None,
                    created_at: now,
                },
            )
            .await?;
            db::insert_text_content_block(
                &conn,
                &Uuid::now_v7().to_string(),
                &action_id,
                0,
                "text",
                &text,
            )
            .await?;
            // Provenance: ordinals 1..=N, ordinal 0 reserved for the structural
            // `reply` edge a memory block does not have.
            for (i, source) in req.sources.iter().take(MAX_SOURCES).enumerate() {
                db::insert_reference_antecedent(
                    &conn,
                    &action_id,
                    source,
                    (i + 1) as i64,
                    None,
                    None,
                    None,
                    req.annotation.as_deref(),
                )
                .await?;
            }

            let blocks = match &existing {
                Some(b) => {
                    db::touch_memory_block(&conn, &b.item_id, &req.scope, now).await?;
                    owned.len()
                }
                None => {
                    db::insert_memory_block(
                        &conn,
                        &db::NewMemoryBlock {
                            item_id: item_id.clone(),
                            root_action_id: action_id.clone(),
                            owner_participant_id: owner_participant_id.to_string(),
                            name: name.clone(),
                            scope: req.scope.clone(),
                            space_id: residence.clone(),
                            created_at: now,
                            updated_at: now,
                        },
                    )
                    .await?;
                    owned.len() + 1
                }
            };
            Ok(Some(blocks))
        }
        .await;
        let blocks = match written {
            Ok(Some(blocks)) => {
                conn.execute("COMMIT", ()).await.map_err(AppError::db)?;
                blocks
            }
            Ok(None) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Ok(MemoryOutcome::Refused(MemoryRefusal::OwnerRetired));
            }
            Err(e) => {
                let _ = conn.execute("ROLLBACK", ()).await;
                return Err(e);
            }
        };

        // A durable commit — the residence space's trail gained an action.
        // Memory is not a post type, so no subscriber's rendered thread
        // changes; the Record and the (future) inspector do.
        self.bus.emit(Change::Space(residence));

        Ok(MemoryOutcome::Written(Box::new(MemoryWrite {
            name,
            scope: req.scope,
            revision,
            blocks,
            unknown_handles: req.unknown_handles,
            annotation_dropped: req.annotation.is_some() && req.sources.is_empty(),
        })))
    }
}

/// The turn-scoped `remember` tool: bound to one responding participant, one
/// residence space and one thread snapshot (for handle → post resolution).
///
/// `Weak` back to the core so a tool that outlives its turn — it cannot, but
/// the registry is `Arc`-shaped — can never keep the database lock alive.
pub(crate) struct RememberTool {
    inner: Weak<Inner>,
    owner_participant_id: String,
    space_id: String,
    snapshot: Arc<ThreadSnapshot>,
}

impl RememberTool {
    pub(crate) fn new(
        inner: Weak<Inner>,
        owner_participant_id: String,
        space_id: String,
        snapshot: Arc<ThreadSnapshot>,
    ) -> Self {
        Self {
            inner,
            owner_participant_id,
            space_id,
            snapshot,
        }
    }
}

/// Read the optional `sources` argument as post handles, resolving each
/// against the turn's snapshot. Unknown handles come back separately: they are
/// reported in the result, never silently dropped and never a turn failure
/// (the snapshot is a point-in-time view, exactly like the navigation tools').
fn resolve_sources(
    snapshot: &ThreadSnapshot,
    arguments: &serde_json::Value,
) -> (Vec<String>, Vec<String>) {
    let mut resolved = Vec::new();
    let mut unknown = Vec::new();
    let Some(list) = arguments.get("sources").and_then(|v| v.as_array()) else {
        return (resolved, unknown);
    };
    for raw in list.iter().take(MAX_SOURCES) {
        let Some(h) = raw.as_str() else { continue };
        let handle = h.trim().trim_start_matches('#').trim().to_lowercase();
        match snapshot.action_for_handle(&handle) {
            Some(action_id) => {
                let action_id = action_id.to_string();
                if !resolved.contains(&action_id) {
                    resolved.push(action_id);
                }
            }
            None => unknown.push(format!("#{handle}")),
        }
    }
    (resolved, unknown)
}

impl Tool for RememberTool {
    fn name(&self) -> &str {
        REMEMBER_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Write one of your own memory blocks. A block is addressed by name: writing a name you \
         already use replaces that block's contents (memory is revised, never appended to), and a \
         new name creates a block. Your blocks are shown to you at the start of every turn."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "block": {
                    "type": "string",
                    "description": format!(
                        "Short name of the block, e.g. \"preferences\". Reuse a name to revise \
                         that block. You may hold at most {MAX_MEMORY_BLOCKS} blocks.",
                    ),
                },
                "text": {
                    "type": "string",
                    "description": format!(
                        "The block's full new contents — this replaces what the block held, so \
                         include whatever should be kept. At most {MAX_MEMORY_BLOCK_BYTES} bytes.",
                    ),
                },
                "scope": {
                    "type": "string",
                    "enum": ["core", "space"],
                    "description": "\"core\" for something true of you wherever you are; \
                                    \"space\" (the default) for something about this \
                                    conversation.",
                },
                "sources": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional handles of the posts this came from, e.g. \
                                    [\"#a2c4e6g\"] — recorded so a reader can see where the note \
                                    came from.",
                },
                "annotation": {
                    "type": "string",
                    "description": "Optional one-line note about why, recorded alongside \
                                    `sources`.",
                },
            },
            "required": ["block", "text"],
        })
    }

    fn call<'a>(&'a self, arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let Some(inner) = self.inner.upgrade() else {
                return Err(ToolError::new("memory is unavailable in this turn"));
            };
            let name = arguments
                .get("block")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let text = arguments
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let scope = normalize_scope(arguments.get("scope").and_then(|v| v.as_str()));
            let annotation = arguments
                .get("annotation")
                .and_then(|v| v.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            let (sources, unknown_handles) = resolve_sources(&self.snapshot, &arguments);

            let outcome = inner
                .remember(
                    &self.owner_participant_id,
                    &self.space_id,
                    RememberRequest {
                        name,
                        text,
                        scope,
                        sources,
                        annotation,
                        unknown_handles,
                    },
                )
                .await
                .map_err(|e| ToolError::new(format!("memory could not be written: {e}")))?;

            match outcome {
                MemoryOutcome::Refused(r) => Err(ToolError::new(r.to_string())),
                MemoryOutcome::Written(w) => Ok(write_receipt(&w)),
            }
        })
    }
}

/// The result the model reads back. States what happened plainly, including
/// the parts that were *not* recorded.
pub(crate) fn write_receipt(w: &MemoryWrite) -> String {
    let where_ = if w.scope == SCOPE_CORE {
        "core"
    } else {
        "this space"
    };
    let mut out = if w.revision == 1 {
        format!(
            "Wrote `{}` ({where_}). You now hold {} of {MAX_MEMORY_BLOCKS} memory blocks.",
            w.name, w.blocks
        )
    } else {
        format!(
            "Revised `{}` ({where_}, revision {}). You hold {} of {MAX_MEMORY_BLOCKS} memory \
             blocks.",
            w.name, w.revision, w.blocks
        )
    };
    if !w.unknown_handles.is_empty() {
        out.push_str(&format!(
            " These handles are not in this space, so they were not recorded as sources: {}.",
            w.unknown_handles.join(", ")
        ));
    }
    if w.annotation_dropped {
        out.push_str(" The annotation was not recorded — it attaches to `sources`.");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, scope: &str, text: &str) -> db::MemoryEntryRow {
        db::MemoryEntryRow {
            item_id: format!("item-{name}"),
            name: name.to_string(),
            scope: scope.to_string(),
            space_id: "space".to_string(),
            action_id: format!("action-{name}"),
            text: text.to_string(),
            updated_at: 0,
        }
    }

    #[test]
    fn the_memory_block_renders_core_and_space_notes() {
        let rendered = render_memory(&[
            entry("about you", SCOPE_CORE, "You prefer terse answers."),
            entry("plan", SCOPE_SPACE, "Ship the parser first."),
        ]);
        assert_eq!(
            rendered,
            format!(
                "<memory>\n{MEMORY_PREAMBLE}\n\n--- about you (core) ---\nYou prefer terse \
                 answers.\n\n--- plan (this space) ---\nShip the parser first.\n</memory>"
            )
        );
    }

    #[test]
    fn a_hostile_block_name_cannot_break_the_one_line_delimiter() {
        for hostile in [
            "notes\n--- other (core) ---",
            "notes\rx",
            "notes\u{2028}x",
            "notes\u{0007}x",
        ] {
            assert_eq!(
                validate_block_name(hostile),
                Err(MemoryRefusal::NameNotOneLine),
                "for {hostile:?}"
            );
        }
        assert_eq!(validate_block_name("  plan  ").unwrap(), "plan");
        assert_eq!(validate_block_name("   "), Err(MemoryRefusal::EmptyName));
        assert_eq!(
            validate_block_name(&"x".repeat(MAX_BLOCK_NAME_BYTES + 1)),
            Err(MemoryRefusal::NameTooLong)
        );
    }

    #[test]
    fn block_text_is_trimmed_bounded_and_never_empty() {
        assert_eq!(validate_block_text("  hi  ").unwrap(), "hi");
        assert_eq!(validate_block_text(" \n "), Err(MemoryRefusal::EmptyText));
        let long = "x".repeat(MAX_MEMORY_BLOCK_BYTES + 1);
        assert_eq!(
            validate_block_text(&long),
            Err(MemoryRefusal::TextTooLarge {
                bytes: MAX_MEMORY_BLOCK_BYTES + 1
            })
        );
    }

    #[test]
    fn scope_defaults_to_this_space() {
        assert_eq!(normalize_scope(Some("core")), SCOPE_CORE);
        assert_eq!(normalize_scope(Some(" CORE ")), SCOPE_CORE);
        assert_eq!(normalize_scope(Some("space")), SCOPE_SPACE);
        assert_eq!(normalize_scope(Some("everywhere")), SCOPE_SPACE);
        assert_eq!(normalize_scope(None), SCOPE_SPACE);
    }

    #[test]
    fn a_receipt_states_what_was_not_recorded() {
        let receipt = write_receipt(&MemoryWrite {
            name: "plan".into(),
            scope: SCOPE_SPACE.into(),
            revision: 2,
            blocks: 3,
            unknown_handles: vec!["#zzzzzzz".into()],
            annotation_dropped: true,
        });
        assert!(receipt.starts_with("Revised `plan` (this space, revision 2)."));
        assert!(receipt.contains("#zzzzzzz"));
        assert!(receipt.contains("annotation was not recorded"));
    }
}
