# State & async — the Eidola doctrine

How application state is owned, shared, mutated, and synchronized across the GUI and app-core. This is the contract for all state-touching code; `crates/eidola-gui/AGENTS.md` and the top-level `AGENTS.md` link here rather than restating it.

## Principles

1. **The entity graph is the application; views are disposable lenses.** Domain state lives in long-lived gpui entities. A window closing must never destroy domain state; a window opening must never need to re-derive it. Two views of the same thing observe the same entity — sync between them is structural, not a feature.
2. **The database is durable truth; stores are its in-memory projection.** Views never own domain data and never copy it into view fields. If a view field duplicates something queryable, it's a bug waiting for a second window.
3. **Tasks are values; cancellation is drop.** In-flight async work is a `Task` stored in a field on the entity that owns the operation. Replacing the field cancels the predecessor; dropping the entity (or window) cancels everything it owned. `.detach()` is forbidden for domain work — it is permitted only for app-lifetime work spawned once at startup, with a comment justifying why nothing ever needs to cancel it.
4. **No shared busy flags, ever.** "In flight" is the presence of a task field or a `Loading` state on the specific operation. A request must never be silently dropped because an unrelated request is running — that class of bug (wave-2's missing model list) is the reason this doctrine exists.
5. **Every write emits.** All durable writes go through app-core, and app-core announces each commit on the invalidation bus. Reads subscribe. Nothing polls its own database, and nothing assumes it is the only writer.
6. **Honest state machines.** The product rule "no fake states" applies internally: every spinner maps to a live task, every "stale" marker to a received invalidation, every error to a typed `AppError`. State enums over boolean soup.

## The pieces

### Loadable — the universal async cell

```rust
enum Loadable<T> {
    /// Never requested. Render nothing or a quiet placeholder.
    NotLoaded,
    /// A real request is in flight (initial load). `Task` lives in a
    /// sibling field on the store, not here — Loadable stays cloneable.
    Loading,
    /// Data present. `stale` means an invalidation arrived after this
    /// snapshot; a refresh task may already be replacing it.
    Loaded { value: T, stale: bool },
    /// The last attempt failed; `value` may retain the previous snapshot
    /// so the UI can show old-data-plus-error instead of a blank page.
    Failed { error: AppError, prior: Option<T> },
}
```

Stores expose `Loadable<T>` snapshots; views match on it exhaustively. A refresh over existing data keeps `Loaded { stale: true }` visible rather than flashing through `Loading` — re-fetches must never blank a page.

### Domain stores (app-global)

One gpui entity per domain, created at startup, held by `AppGlobal`, observed by any view that renders the domain. Each store owns: its `Loadable` snapshots, its in-flight `Task` fields, its subscription to the invalidation bus, and *all* mutations of its domain. Views call store methods; views never call `AppCore` directly.

| Store | Owns | Notes |
|---|---|---|
| `ConfigStore` | `ConfigState` snapshot; set/clear overrides, default model | Synchronous reads, write-through; emits `Config` |
| `ModelsStore` | model list + pricing | Refreshed at launch + on demand; first-window list comes from here (fixes wave-2 bug 1 structurally) |
| `AccountStore` | balances, prices, account lifecycle, checkout | The checkout *poll* is view-owned (see scoping); its result lands here via invalidation |
| `WalletStore` | credential lifecycle list, recovery | |
| `SpacesStore` | the Library index **and** the space-entity registry (below) | |
| `UpdateStore` | `UpdateCheckSnapshot`, polling, accept-claims | Mostly exists already; the pattern to which everything else converges |

Store method shape (the only sanctioned async idiom):

```rust
pub fn refresh_balances(&mut self, cx: &mut Context<Self>) {
    let core = self.app_core.clone();
    self.balances = self.balances.to_loading();          // honest state
    self.balances_task = Some(cx.spawn(async move |this, cx| {
        let result = bridge(core, |c| async move { c.account_balances().await }).await;
        this.update(cx, |this, cx| {
            this.balances = Loadable::from(result);
            this.balances_task = None;
            cx.notify();
        }).ok();
    }));                                                  // replace = cancel
}
```

No generation counters: because the previous task is cancelled at the moment of replacement, the completing task is always the latest one. Where ordering spans *operations* (a submit racing an initial load), both operations are owned by the same entity and serialized there — which retires the `transcript_generation` workaround.

Stores expose two method shapes, distinguished by who owns the task:

- **`refresh_*` — fire-and-notify.** The store owns the slot; callers get the result by observing. For state whose lifetime outlives any one window (balances, models, listings).
- **`request_*` — awaitable.** Returns a receiver; the *caller* awaits it inside the caller's own slot, so the work dies with the caller (a checkout poll dies with its window). The store still performs the write and the bus still emits; only task ownership moves. Ownership = cancellation authority, decided per operation.

### Concurrency patterns

"Task as a field" gives one slot of mutually-exclusive work per *field*, replace-cancels. The field's **type** sets the concurrency width — entity granularity is driven by identity and observation, never by concurrency needs. The banned thing is ownerless work (`detach`), not parallel work.

| Need | Pattern | Slot shape |
|---|---|---|
| Latest request wins (most reads) | supersede | `Option<Task<()>>` |
| Independent per-key work (detail panes, image loads) | keyed | `HashMap<K, Task<()>>` |
| All must run, in order (submits to a space) | queue + runner | `VecDeque<Op>` + `Option<Task<()>>` runner |
| Concurrent callers want one result (two windows opening one space) | join-existing | keyed `Shared` future (Zed's `open_buffer` pattern) |
| Wait-for-quiet (search-as-you-type) | debounce | `timer(DEBOUNCE).await` at the head of a supersede slot — replacement cancels the pending timer |
| Rate limit | throttle | supersede slot + `last_ran: Instant` in entity state |
| One operation with parallel parts | join inside | one slot, `futures::join!` in the body |

Notes:

- **Supersede is keep-newest.** The deleted `busy` flag was keep-oldest (drop the new request) — that inversion caused the wave-2 bug class. UI-shaped work almost always wants keep-newest.
- **Queues are state, not executor side-effects.** A queue lives in entity fields so it is renderable ("1 queued" — honest states) and dies with its owner. The runner re-enters via `this.update(cx, …)` between awaits; it never holds `&mut self` across an await.
- **Composition rules.** (1) Within one logical operation: compose inside the task (sequential awaits, `join!`). (2) Across stores: compose by event — write through app-core, let the bus drive each store's own refresh; store A never orchestrates store B's tasks (retires wave-2's nested allocate→refresh spawns). (3) View-scoped flows: the view awaits a `request_*` receiver inside its own slot.

### Atomicity & cancellation

Cancellation is the *easy* way for a task to die — crash and power loss are the hard ways, and in-memory rollback protects against neither. So atomicity is never the task's job:

1. **Atomicity lives below the cancellation boundary.** A multi-part-but- atomic operation is one app-core method wrapping one DB transaction on the core runtime. The gpui `Task` is a *subscription to the outcome*, not the operation: dropping it abandons the oneshot receiver while the core-side work runs to completion, and the bus (emit-after-commit) delivers the state change to every store even if the initiator died. Queue runners perform each op as a single core-side call, so the invariant holds: **cancellation may only ever land between durable operations, never inside one.**
2. **Multi-part and interruptible (cross-system) → saga with durable intent.** Where one transaction can't span the parts (DB write → HTTP call → settlement): persist intent before the side effect, make every step idempotent, recover resumably at startup/on demand. The wallet's spend flow is the house example — `pre_credential` refund intent written before the spend, the `spending` lifecycle state as the visible intermediate, `recover_spending_credentials` as compensation. If an operation survives `kill -9`, task cancellation is a strictly weaker adversary and needs no special handling. No in-memory rollback machinery: it covers only polite cancellation, hides the intermediate state, and then lies about it having never happened.
3. **Owner = blast radius.** A view may own a task only if killing it at any await point is harmless (a poll, a fetch). If a half-done operation would be bad when a window closes, promote ownership to a store (app lifetime) or push the effects core-side.

Hygiene corollaries: Rust has no async-drop — `Drop`-guard cleanup in a cancelled task is synchronous-only, fit for in-memory tidying and nothing else (one more reason effects live below the boundary). And `Loading` must imply a live task: never cancel-without-replace unless the `Loadable` is reset in the same update, or the entity shows a spinner forever — a fake state.

### Space entities — shared, registried

`Space` is a gpui entity owning everything about one conversation: the transcript (`Loadable<Vec<...>>`), streaming buffers, the pending submit task, the per-space model selection, and auto-title state. `SpacesStore` holds `HashMap<SpaceId, WeakEntity<Space>>`; opening a space goes through `SpacesStore::open(space_id)` which gets-or-creates. **Two windows on one space hold the same entity** — submits, streams, and edits appear in both, structurally (fixes wave-2 bug 4).

- A blank ⌘N window holds a `Space` with `id: None`; the registry learns about it when the first exchange persists and assigns an id. The blank page stays instant.
- `Space` is an `EventEmitter<SpaceEvent>` (`MessagesChanged`, `StreamDelta`, `StreamEnded`, `Failed`) so views can react semantically (e.g. tail-scroll only on `StreamDelta`), while plain `cx.observe` covers re-render.
- The composer draft is **not** in `Space` — it is window-local by design: two windows on one space are two cursors, and two drafts is the intuitive behavior. (Revisit only if real usage disagrees.)

### Window-scoped readers (the Record pattern)

Paginated/exploratory reads (Record listings, detail panes) are window-scoped reader entities: they hold an immutable snapshot + cursor + their own fetch tasks, and die with the window. They subscribe to the bus only to mark themselves `stale` and surface a quiet "new entries — refresh" affordance — they never mutate rows under the user's scroll position (offset-shifting rug-pulls are a fake state of position).

### The invalidation bus (app-core)

```rust
pub enum Change {
    Config,
    Account,            // balances, account lifecycle
    Wallet,             // credentials, lifecycle states
    SpaceIndex,         // create/archive/rename/title
    Space(SpaceId),     // actions/messages within one space
    Record,             // attestations / requests / spend trail appended
    UpdateState,
}
```

- A `tokio::sync::broadcast::Sender<Change>` lives in `AppCore`. Every app-core write path emits **after** its durable commit. Because the CLI uses the same AppCore methods, its writes emit too (within its own process).
- The GUI installs **one** bridge at startup: a task on the core runtime forwards bus events through a channel into a single gpui main-thread loop, which dispatches to stores. No store talks tokio directly.
- The bus is behind a narrow trait (`ChangeSource`) with the in-process broadcast as the v1 implementation. The documented v2 seam is **Turso CDC tailing** (`PRAGMA capture_data_changes_conn` → `turso_cdc`), which extends the same `Change` stream across *processes* — the CLI-writes- while-GUI-open gap. Not in scope now; the seam is.

### Input-state sharing (the ⌥ pattern)

gpui dispatches `ModifiersChangedEvent` along the **focused element's ancestor path only** (verified at pin 969a67fc) — a listener on a sibling branch never fires (wave-2 bug 2). Therefore: exactly one listener per window, on the window-root view (the one whose tracked focus handle is always an ancestor of focus), mirroring into a per-window `WindowInput` entity (`alt_held`, future modifier/chord state). Descendant views observe `WindowInput`; **no view below the root may register `on_modifiers_changed`**. Chat's existing reveal migrates onto this; the Settings ⌥ reveal starts working structurally.

### Lists

Any list that can exceed roughly one screen renders through gpui's virtualized primitives: `uniform_list` for fixed-height rows (Record, Library), `list` with measured items for variable heights (transcript, eventually). Appending pages must not grow per-frame work linearly (fixes wave-2 bug 3). Large raw payloads (Record details) render through a cached parse — parse-on-state-change, not parse-per-frame.

### Scoping decision table

| State | Scope | Owner |
|---|---|---|
| Config / overrides / default model | global | `ConfigStore` |
| Models + pricing | global | `ModelsStore` |
| Balances, prices, account lifecycle | global | `AccountStore` |
| Credential lifecycle | global | `WalletStore` |
| Library index | global | `SpacesStore` |
| Transcript, streaming, submit, per-space model choice | per-space, shared | `Space` entity (registry) |
| Update check state | global | `UpdateStore` |
| Record listings, cursors, detail | per-window | reader entity |
| Checkout balance poll | per-window task | initiating view (dies with window; outcome lands in `AccountStore` via the bus) |
| Composer draft, scroll, hover, disclosure, picker-open, ⌥ | per-window | view / `WindowInput` |

### Testing doctrine

- Stores get stub constructors (fixture `Loadable` values) — replacing `Core::stub`'s field-poking. Views are tested against stores, never against private copies.
- Every ordering bug gets a deterministic replay test: `TestAppContext` + `run_until_parked` between the interleaved steps that produced it. The four wave-2 bugs each get one as part of their fix.
- The bus gets unit tests in app-core (emit-on-commit, not emit-on-error) and one GUI test proving a write in store A invalidates a subscriber B.

## Out of scope (documented seams)

- Cross-process invalidation (CDC tailing) and any sync/multi-device story — the `ChangeSource` seam is the contract.
- Turso live materialized views — revisit when our pinned version stabilizes them; they would slot in *below* the bus as query acceleration, not replace it.
- Offline mutation queueing.
