//! The Record window — the trust door's engine room.
//!
//! Everything the app ever sent or received is recorded in the local Turso
//! database; this window is the door to that raw record. Three sections:
//!
//! - **Attestations** — hardware attestation documents captured per
//!   connection (`attestation` table). Detail: the full raw document,
//!   pretty-printed when it parses as JSON, hex otherwise.
//! - **Requests** — every HTTP request/response pair (`request` joined with
//!   `connection`). Detail: raw headers and bodies, selectable, capped at
//!   64 KiB inline (Copy always yields the full payload). Nothing is
//!   redacted — this is the user's own traffic on their own machine.
//! - **Spending** — the `spend_trail` view: credential → request → action →
//!   space, grouped by credential.
//!
//! Newest first, windowed fetch (`LIMIT/OFFSET` via the app-core Record
//! APIs — the window never loads a whole table). Design language: mono
//! (Menlo) for raw data, the UI font for chrome, hairline rules, no
//! boxes-in-boxes.

use std::ops::Range;

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AttestationDetail, AttestationInfo, RequestDetail, RequestInfo, SpendTrailEntry,
};
use gpui::{
    AsyncApp, ClipboardItem, Context, Div, FocusHandle, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, Subscription, Task,
    UniformListScrollHandle, WeakEntity, Window, div, px, uniform_list,
};
use gpui_component::{
    ActiveTheme, StyledExt, h_flex,
    highlighter::HighlightTheme,
    text::{TextView, TextViewStyle},
    v_flex,
};

use crate::actions::CloseWindow;
use crate::bridge;
use crate::probe::Probe as _;
use crate::stores::Stores;

/// Horizontal clearance for the macOS traffic lights — the section strip
/// doubles as the title bar, same pattern as the old settings tab strip
/// (matches gpui-component's `TITLE_BAR_LEFT_PADDING`).
#[cfg(target_os = "macos")]
const STRIP_LEFT_PAD: gpui::Pixels = gpui::px(80.);
#[cfg(not(target_os = "macos"))]
const STRIP_LEFT_PAD: gpui::Pixels = gpui::px(12.);

/// Rows fetched per page. The fetch asks for `PAGE + 1` to learn whether a
/// further page exists without a COUNT query.
const PAGE: i64 = 50;

/// Cap on inline-rendered raw payloads. Larger payloads render their first
/// 64 KiB with an honest note; Copy always yields the full data.
const INLINE_CAP: usize = 64 * 1024;

/// Fixed row height for the virtualized listings. `uniform_list` lays every
/// row out at the height of the measured row, so all listing rows — data
/// rows, spending group headers, and the trailing load-more row — share one
/// height. Two text lines (title + subline) plus the original `py_2p5`
/// breathing room land here; the slight tightening of the formerly taller
/// spending group headers is intentional (see AGENTS.md → The Record).
const ROW_H: gpui::Pixels = gpui::px(56.);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RecordSection {
    Attestations,
    Requests,
    Spending,
}

impl RecordSection {
    fn label(self) -> &'static str {
        match self {
            RecordSection::Attestations => "Attestations",
            RecordSection::Requests => "Requests",
            RecordSection::Spending => "Spending",
        }
    }
}

/// An open detail view: one attestation document or one request/response
/// pair.
pub enum RecordDetail {
    Attestation(AttestationDetail),
    Request(Box<RequestDetail>),
}

/// A raw payload projected for inline display **once**, when the detail is
/// opened — not on every frame. `fenced` is the markdown-code-fenced string
/// handed to `TextView::markdown` (stable across frames, so TextView's keyed
/// `TextViewState` parses it exactly once — see the parse-caching note in
/// `AGENTS.md`). `full` is the complete, uncapped payload for Copy. `note`
/// is the honest "showing the first 64 KiB" line when the inline view was
/// capped.
struct CachedPayload {
    /// Section heading ("Document", "Request headers", …).
    title: &'static str,
    /// Stable `TextView` element id; keeping it constant lets the keyed
    /// `TextViewState` survive and skip re-parsing.
    id: &'static str,
    /// The fenced, capped markdown string rendered inline (computed once).
    fenced: String,
    /// The full uncapped payload — what Copy writes to the clipboard.
    full: String,
    /// Set when the inline view was truncated to [`INLINE_CAP`].
    note: Option<String>,
}

/// Everything about an open detail that is expensive to recompute: the
/// projected/fenced raw payloads (JSON pretty-print, UTF-8, or hex dump,
/// then fenced + capped). Built once via [`DetailCache::build`] when a
/// detail opens; the render path only reads it.
struct DetailCache {
    detail: RecordDetail,
    payloads: Vec<CachedPayload>,
}

impl DetailCache {
    fn build(detail: RecordDetail) -> Self {
        let payloads = match &detail {
            RecordDetail::Attestation(d) => {
                vec![cached_payload(
                    "Document",
                    "attestation-doc",
                    render_payload_text(&d.doc),
                )]
            }
            RecordDetail::Request(d) => vec![
                cached_payload(
                    "Request headers",
                    "req-headers",
                    header_text(d.request_headers.as_deref()),
                ),
                cached_payload(
                    "Request body",
                    "req-body",
                    body_text(d.request_body.as_deref()),
                ),
                cached_payload(
                    "Response headers",
                    "resp-headers",
                    header_text(d.response_headers.as_deref()),
                ),
                cached_payload(
                    "Response body",
                    "resp-body",
                    body_text(d.response_body.as_deref()),
                ),
            ],
        };
        Self { detail, payloads }
    }
}

/// Project a raw payload string into a [`CachedPayload`] — done once when a
/// detail opens, never per frame.
fn cached_payload(title: &'static str, id: &'static str, full: String) -> CachedPayload {
    let (shown, note) = cap_inline(&full);
    CachedPayload {
        title,
        id,
        fenced: fenced(&shown),
        full,
        note,
    }
}

/// One row of a virtualized listing, after the rows have been flattened into
/// a display model. The `uniform_list` closure is a dumb indexer over a
/// `Vec<DisplayRow>` — group headers and the trailing load-more affordance
/// survive virtualization because they are precomputed rows, not render-time
/// branches the closure would have to reconstruct from a windowed range.
#[derive(Clone, Copy)]
enum DisplayRow {
    /// A spending group header; the payload is the index (into `rows`) of the
    /// first data row in the group, which carries the credential identity.
    Header(usize),
    /// A data row; the payload indexes into `rows`.
    Data(usize),
    /// The trailing "Load more…" affordance (or its in-flight variant).
    LoadMore,
}

/// Per-section listing state: the rows fetched so far, windowing flags, and
/// the precomputed flat display model the `uniform_list` closure indexes.
struct Listing<T> {
    rows: Vec<T>,
    has_more: bool,
    loaded: bool,
    loading: bool,
    /// Flattened display rows (data rows + spending group headers + an
    /// optional trailing load-more row). Rebuilt on any state change so the
    /// list closure stays O(visible).
    display: Vec<DisplayRow>,
    scroll: UniformListScrollHandle,
    /// Supersede slot for this section's page fetch. Replacing the `Listing`
    /// (refresh) drops the slot and cancels the in-flight task, so a
    /// superseded fetch can never land late and append stale or duplicate
    /// rows — replace-cancels, no generation counters
    /// (`docs/architecture/state.md` → "Concurrency patterns").
    task: Option<Task<()>>,
}

impl<T> Default for Listing<T> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            has_more: false,
            loaded: false,
            loading: false,
            display: Vec::new(),
            scroll: UniformListScrollHandle::new(),
            task: None,
        }
    }
}

impl<T> Listing<T> {
    /// Whether a trailing load-more row should be appended: when there are
    /// more pages, or a page fetch is currently in flight over existing rows
    /// (the in-flight row is honest — it maps to a real fetch task).
    fn wants_load_more(&self) -> bool {
        !self.rows.is_empty() && (self.has_more || self.loading)
    }

    /// Rebuild the flat display model for the simple sections (attestations,
    /// requests): one `Data` row per fetched row, plus an optional trailing
    /// load-more row.
    fn rebuild_flat_display(&mut self) {
        let mut display: Vec<DisplayRow> = (0..self.rows.len()).map(DisplayRow::Data).collect();
        if self.wants_load_more() {
            display.push(DisplayRow::LoadMore);
        }
        self.display = display;
    }
}

pub struct RecordView {
    stores: Stores,
    section: RecordSection,
    attestations: Listing<AttestationInfo>,
    requests: Listing<RequestInfo>,
    spending: Listing<SpendTrailEntry>,
    /// The open detail, with its raw payloads projected once (not per frame).
    detail: Option<DetailCache>,
    /// Identifier (hash or request id) of a detail fetch in flight.
    detail_pending: Option<String>,
    /// Supersede slot for the detail fetch — one slot for both detail kinds,
    /// so only the *latest* click's result can ever land (replace-cancels),
    /// and closing the detail cancels the fetch outright.
    detail_task: Option<Task<()>>,
    /// The bus reported new Record rows since this window's last (re)fetch.
    /// Surfaces the quiet "new entries — refresh" affordance in the strip;
    /// rows are never mutated under the user's scroll position.
    stale: bool,
    /// `RecordStore` epoch this window last refreshed at.
    seen_epoch: u64,
    /// Keeps the `RecordStore` observation alive for the window's lifetime.
    _record_observer: Subscription,
    error: Option<String>,
    focus_handle: FocusHandle,
}

impl RecordView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        // The Record is a window-scoped reader (the doctrine's "Record
        // pattern"): it owns its rows and fetch tasks, and subscribes to the
        // bus — via the RecordStore relay — only to mark itself stale.
        let seen_epoch = stores.record.read(cx).epoch();
        let record_observer = cx.observe(&stores.record, |this, store, cx| {
            if store.read(cx).epoch() > this.seen_epoch && !this.stale {
                this.stale = true;
                cx.notify();
            }
        });

        let mut this = Self {
            stores,
            section: RecordSection::Attestations,
            attestations: Listing::default(),
            requests: Listing::default(),
            spending: Listing::default(),
            detail: None,
            detail_pending: None,
            detail_task: None,
            stale: false,
            seen_epoch,
            _record_observer: record_observer,
            error: None,
            focus_handle,
        };
        this.fetch_page(RecordSection::Attestations, cx);
        this
    }

    /// The focus handle the root tracks — for behavior tests' action
    /// dispatch.
    pub fn focus_handle(&self) -> FocusHandle {
        self.focus_handle.clone()
    }

    pub fn section(&self) -> RecordSection {
        self.section
    }

    pub fn detail(&self) -> Option<&RecordDetail> {
        self.detail.as_ref().map(|c| &c.detail)
    }

    /// Rebuild the spending display model: a group header whenever the
    /// credential nonce changes (rows are time-ordered, so consecutive runs
    /// of one nonce share a header — matching reality), interleaved with the
    /// data rows, plus an optional trailing load-more row.
    fn rebuild_spending_display(&mut self) {
        let mut display: Vec<DisplayRow> = Vec::with_capacity(self.spending.rows.len() + 4);
        let mut prev_nonce: Option<&str> = None;
        for (idx, e) in self.spending.rows.iter().enumerate() {
            if prev_nonce != Some(e.credential_nonce.as_str()) {
                display.push(DisplayRow::Header(idx));
                prev_nonce = Some(e.credential_nonce.as_str());
            }
            display.push(DisplayRow::Data(idx));
        }
        if self.spending.wants_load_more() {
            display.push(DisplayRow::LoadMore);
        }
        self.spending.display = display;
    }

    /// Rebuild the flat display model for whichever section changed.
    fn rebuild_display(&mut self, section: RecordSection) {
        match section {
            RecordSection::Attestations => self.attestations.rebuild_flat_display(),
            RecordSection::Requests => self.requests.rebuild_flat_display(),
            RecordSection::Spending => self.rebuild_spending_display(),
        }
    }

    pub fn detail_pending(&self) -> Option<&str> {
        self.detail_pending.as_deref()
    }

    /// Whether the bus has reported new Record rows since this window's last
    /// (re)fetch.
    pub fn stale(&self) -> bool {
        self.stale
    }

    /// Switch sections. Closes any open detail (cancelling an in-flight
    /// detail fetch); fetches the section's first page on first visit.
    pub fn select_section(&mut self, section: RecordSection, cx: &mut Context<Self>) {
        self.detail = None;
        self.detail_pending = None;
        self.detail_task = None;
        if self.section != section {
            self.section = section;
        }
        let loaded = match section {
            RecordSection::Attestations => self.attestations.loaded || self.attestations.loading,
            RecordSection::Requests => self.requests.loaded || self.requests.loading,
            RecordSection::Spending => self.spending.loaded || self.spending.loading,
        };
        if !loaded {
            self.fetch_page(section, cx);
        }
        cx.notify();
    }

    /// Re-fetch from the top. Resets every section — a Record change can
    /// touch any of them, so the stale marker is only honest to clear if all
    /// three re-read (the current one immediately; the others on next
    /// visit). Replacing each `Listing` drops its in-flight fetch task, so a
    /// superseded fetch can never land late and append duplicate rows.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        self.attestations = Listing::default();
        self.requests = Listing::default();
        self.spending = Listing::default();
        self.stale = false;
        self.seen_epoch = self.stores.record.read(cx).epoch();
        self.fetch_page(self.section, cx);
        cx.notify();
    }

    /// Fetch the next page of the current section.
    pub fn load_more(&mut self, cx: &mut Context<Self>) {
        self.fetch_page(self.section, cx);
    }

    fn fetch_page(&mut self, section: RecordSection, cx: &mut Context<Self>) {
        let Some(app_core) = self.stores.app_core() else {
            // Stub stores (tests): rows are installed via the test setters.
            return;
        };
        macro_rules! fetch {
            ($listing:ident, $helper:ident, $section:expr) => {{
                if self.$listing.loading {
                    return;
                }
                self.$listing.loading = true;
                // Rebuild now so a fetch over existing rows shows its honest
                // in-flight load-more row (a real task is in flight).
                self.rebuild_display($section);
                let offset = self.$listing.rows.len() as i64;
                let rx = bridge::$helper(app_core, PAGE + 1, offset);
                // The listing owns its fetch (task-as-field): dropping the
                // `Listing` (refresh) cancels the task, so only a fetch
                // against the *current* listing can ever append rows.
                self.$listing.task = Some(cx.spawn(
                    async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                        let res = rx.await.unwrap_or_else(|_| {
                            Err(AppError::Internal {
                                message: "record query cancelled".into(),
                            })
                        });
                        let _ = this.update(cx, |this, cx| {
                            this.$listing.task = None;
                            this.$listing.loading = false;
                            this.$listing.loaded = true;
                            match res {
                                Ok(mut rows) => {
                                    this.$listing.has_more = rows.len() as i64 > PAGE;
                                    rows.truncate(PAGE as usize);
                                    this.$listing.rows.extend(rows);
                                }
                                Err(e) => this.error = Some(e.to_string()),
                            }
                            this.rebuild_display($section);
                            cx.notify();
                        });
                    },
                ));
            }};
        }
        match section {
            RecordSection::Attestations => {
                fetch!(attestations, list_attestations, RecordSection::Attestations)
            }
            RecordSection::Requests => fetch!(requests, list_requests, RecordSection::Requests),
            RecordSection::Spending => fetch!(spending, spend_trail, RecordSection::Spending),
        }
        cx.notify();
    }

    /// Open an attestation's raw document.
    pub fn open_attestation(&mut self, hash: String, cx: &mut Context<Self>) {
        self.detail_pending = Some(hash.clone());
        self.error = None;
        cx.notify();
        let Some(app_core) = self.stores.app_core() else {
            return;
        };
        let rx = bridge::attestation_detail(app_core, hash.clone());
        // Replace-cancels: assigning the slot drops any in-flight detail
        // fetch, so only the latest click's result can land — no staleness
        // check needed.
        self.detail_task = Some(cx.spawn(
            async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                let res = rx.await.unwrap_or_else(|_| {
                    Err(AppError::Internal {
                        message: "record query cancelled".into(),
                    })
                });
                let _ = this.update(cx, |this, cx| {
                    this.detail_task = None;
                    this.detail_pending = None;
                    match res {
                        Ok(Some(d)) => {
                            this.detail = Some(DetailCache::build(RecordDetail::Attestation(d)))
                        }
                        Ok(None) => this.error = Some(format!("attestation not found: {hash}")),
                        Err(e) => this.error = Some(e.to_string()),
                    }
                    cx.notify();
                });
            },
        ));
    }

    /// Open a request's raw request/response pair. Reachable from both the
    /// Requests listing and the Spending trail.
    pub fn open_request(&mut self, id: String, cx: &mut Context<Self>) {
        self.detail_pending = Some(id.clone());
        self.error = None;
        cx.notify();
        let Some(app_core) = self.stores.app_core() else {
            return;
        };
        let rx = bridge::request_detail(app_core, id.clone());
        // Replace-cancels, same as `open_attestation` — one slot covers both
        // detail kinds, so the latest click always wins.
        self.detail_task = Some(cx.spawn(
            async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                let res = rx.await.unwrap_or_else(|_| {
                    Err(AppError::Internal {
                        message: "record query cancelled".into(),
                    })
                });
                let _ = this.update(cx, |this, cx| {
                    this.detail_task = None;
                    this.detail_pending = None;
                    match res {
                        Ok(Some(d)) => {
                            this.detail =
                                Some(DetailCache::build(RecordDetail::Request(Box::new(d))))
                        }
                        Ok(None) => this.error = Some(format!("request not found: {id}")),
                        Err(e) => this.error = Some(e.to_string()),
                    }
                    cx.notify();
                });
            },
        ));
    }

    /// Back from a detail to the section listing. Dropping the detail task
    /// cancels an in-flight fetch, so a slow detail can't reopen after the
    /// user backed out.
    pub fn close_detail(&mut self, cx: &mut Context<Self>) {
        self.detail = None;
        self.detail_pending = None;
        self.detail_task = None;
        cx.notify();
    }

    // --- Test setters (stub cores can't drive the async fetches) ---------

    #[doc(hidden)]
    pub fn set_attestations_for_test(&mut self, rows: Vec<AttestationInfo>, has_more: bool) {
        self.attestations.rows = rows;
        self.attestations.has_more = has_more;
        self.attestations.loaded = true;
        self.attestations.rebuild_flat_display();
    }

    #[doc(hidden)]
    pub fn set_requests_for_test(&mut self, rows: Vec<RequestInfo>, has_more: bool) {
        self.requests.rows = rows;
        self.requests.has_more = has_more;
        self.requests.loaded = true;
        self.requests.rebuild_flat_display();
    }

    #[doc(hidden)]
    pub fn set_spending_for_test(&mut self, rows: Vec<SpendTrailEntry>, has_more: bool) {
        self.spending.rows = rows;
        self.spending.has_more = has_more;
        self.spending.loaded = true;
        self.rebuild_spending_display();
    }

    #[doc(hidden)]
    pub fn set_detail_for_test(&mut self, detail: Option<RecordDetail>) {
        self.detail = detail.map(DetailCache::build);
        self.detail_pending = None;
    }

    /// Test/perf hook: render exactly the visible window of display rows for
    /// the current section and return how many elements were produced. This
    /// is the per-frame work `uniform_list` performs (it calls the same
    /// `render_rows` indexer with the visible range), so it lets a test assert
    /// frame cost is O(visible), independent of the total loaded-row count.
    #[doc(hidden)]
    pub fn render_visible_window_for_test(
        &self,
        range: std::ops::Range<usize>,
        cx: &Context<Self>,
    ) -> usize {
        self.render_rows(self.section, range, cx).len()
    }

    /// Test hook: the current section's (row count, loading flag) — lets the
    /// stale-fetch replay assert that a refresh during an in-flight fetch
    /// resets the rows and starts a fresh (sole) fetch.
    #[doc(hidden)]
    pub fn listing_state_for_test(&self) -> (usize, bool) {
        match self.section {
            RecordSection::Attestations => {
                (self.attestations.rows.len(), self.attestations.loading)
            }
            RecordSection::Requests => (self.requests.rows.len(), self.requests.loading),
            RecordSection::Spending => (self.spending.rows.len(), self.spending.loading),
        }
    }

    /// Test hook: number of display rows in the current section (data rows +
    /// spending group headers + any trailing load-more row).
    #[doc(hidden)]
    pub fn display_len_for_test(&self) -> usize {
        match self.section {
            RecordSection::Attestations => self.attestations.display.len(),
            RecordSection::Requests => self.requests.display.len(),
            RecordSection::Spending => self.spending.display.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

impl Render for RecordView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Copy the colors out (Hsla is Copy) so no shared borrow of `cx`
        // outlives the &mut reborrows the render helpers need.
        let (bg, fg, muted_fg) = {
            let t = cx.theme();
            (t.background, t.foreground, t.muted_foreground)
        };

        // The body falls into two shapes:
        // - Detail / pending / empty: ordinary flow content inside an
        //   `overflow_y_scroll` container (these never grow with loaded-row
        //   count, so they need no virtualization).
        // - A populated listing: a self-scrolling `uniform_list` placed
        //   directly as the `flex_1` child — its render cost is O(visible),
        //   not O(loaded).
        let body: gpui::AnyElement = if self.detail.is_some() {
            scroll_wrap(self.render_detail(cx)).into_any_element()
        } else if self.detail_pending.is_some() {
            scroll_wrap(
                div()
                    .px_6()
                    .py_4()
                    .italic()
                    .text_color(muted_fg)
                    .child("Loading…"),
            )
            .into_any_element()
        } else if self.current_listing_is_empty() {
            scroll_wrap(self.render_empty(cx)).into_any_element()
        } else {
            self.render_listing(cx)
        };

        v_flex()
            .track_focus(&self.focus_handle)
            .on_action(cx.listener(|_, _: &CloseWindow, window, _| {
                window.remove_window();
            }))
            .size_full()
            .bg(bg)
            .text_color(fg)
            .child(self.render_strip(cx))
            .child(body)
    }
}

/// Wrap ordinary (non-virtualized) body content in the scroll container.
fn scroll_wrap(content: impl IntoElement) -> gpui::Stateful<Div> {
    div()
        .id("record-scroll")
        .flex_1()
        .w_full()
        .overflow_y_scroll()
        .child(content)
}

impl RecordView {
    /// The section strip doubles as the title bar — traffic lights to its
    /// left, quiet text sections, an italic wordmark on the right.
    fn render_strip(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let mut strip = h_flex()
            .w_full()
            .flex_none()
            .h(px(36.))
            .items_center()
            .gap_5()
            .pl(STRIP_LEFT_PAD)
            .pr_4()
            .border_b_1()
            .border_color(theme.border);

        for section in [
            RecordSection::Attestations,
            RecordSection::Requests,
            RecordSection::Spending,
        ] {
            let active = self.section == section && self.detail.is_none();
            let mut label = div()
                .id(section.label())
                .probe(
                    format!("record/section/{}", section.label().to_lowercase()),
                    gpui::Role::Tab,
                    section.label(),
                )
                .aria_selected(active)
                .text_sm()
                .cursor_pointer()
                .on_click(cx.listener(move |this, _, _, cx| this.select_section(section, cx)))
                .child(section.label());
            if active {
                label = label.text_color(theme.foreground);
            } else {
                label = label
                    .text_color(theme.muted_foreground)
                    .hover(|s| s.text_color(theme.foreground));
            }
            strip = strip.child(label);
        }

        // The bus-driven stale marker surfaces here: the refresh affordance
        // quietly announces that new rows exist, instead of mutating the
        // listing under the user's scroll position (the doctrine's "new
        // entries — refresh" affordance).
        let (refresh_label, refresh_color, refresh_hover) = if self.stale {
            ("New entries — refresh", theme.link, theme.link_hover)
        } else {
            ("Refresh", theme.muted_foreground, theme.foreground)
        };
        strip
            .child(div().flex_1())
            .child(
                div()
                    .id("record-refresh")
                    .probe("record/refresh", gpui::Role::Button, refresh_label)
                    .text_xs()
                    .cursor_pointer()
                    .text_color(refresh_color)
                    .hover(move |s| s.text_color(refresh_hover))
                    .child(refresh_label)
                    .on_click(cx.listener(|this, _, _, cx| this.refresh(cx))),
            )
            .child(
                div()
                    .text_sm()
                    .italic()
                    .text_color(theme.muted_foreground)
                    .child("The Record"),
            )
    }

    fn list_frame(&self) -> Div {
        v_flex().w_full().px_6().pt_2().pb_8()
    }

    fn empty_line(&self, text: &'static str, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        div()
            .py_8()
            .w_full()
            .text_color(theme.muted_foreground)
            .child(text)
    }

    fn error_line(&self, cx: &Context<Self>) -> Option<Div> {
        let theme = cx.theme();
        self.error.as_deref().map(|err| {
            div()
                .pt_3()
                .text_sm()
                .text_color(theme.danger)
                .child(SharedString::from(err.to_string()))
        })
    }

    // --- Listing dispatch + virtualization --------------------------------

    /// Whether the current section has no data rows yet (so the body should
    /// render the empty/loading state rather than the virtualized list).
    fn current_listing_is_empty(&self) -> bool {
        match self.section {
            RecordSection::Attestations => self.attestations.rows.is_empty(),
            RecordSection::Requests => self.requests.rows.is_empty(),
            RecordSection::Spending => self.spending.rows.is_empty(),
        }
    }

    /// The empty/loading column for the current section (rendered when there
    /// are no rows yet — the populated path is the virtualized list).
    fn render_empty(&self, cx: &Context<Self>) -> Div {
        let (loading, empty_text) = match self.section {
            RecordSection::Attestations => {
                (self.attestations.loading, "No attestations recorded yet.")
            }
            RecordSection::Requests => (self.requests.loading, "No requests recorded yet."),
            RecordSection::Spending => (self.spending.loading, "Nothing spent yet."),
        };
        let mut col = self.list_frame();
        if loading {
            return col.child(self.empty_line("Loading…", cx));
        }
        col = col.child(self.empty_line(empty_text, cx));
        if let Some(err) = self.error_line(cx) {
            col = col.child(err);
        }
        col
    }

    /// Build the virtualized listing for the current section. The
    /// `uniform_list` closure is a dumb indexer over the precomputed
    /// `display` model: it renders only the visible window of rows, so frame
    /// work is O(visible) rather than O(loaded) (the wave-2 bug-3 fix).
    fn render_listing(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let count = match self.section {
            RecordSection::Attestations => self.attestations.display.len(),
            RecordSection::Requests => self.requests.display.len(),
            RecordSection::Spending => self.spending.display.len(),
        };
        let scroll = match self.section {
            RecordSection::Attestations => self.attestations.scroll.clone(),
            RecordSection::Requests => self.requests.scroll.clone(),
            RecordSection::Spending => self.spending.scroll.clone(),
        };
        let section = self.section;

        uniform_list(
            ("record-list", section as usize),
            count,
            cx.processor(move |this, range: Range<usize>, _window, cx| {
                this.render_rows(section, range, cx)
            }),
        )
        .flex_1()
        .w_full()
        .px_6()
        .pt_2()
        .track_scroll(&scroll)
        .into_any_element()
    }

    /// Render the visible window of display rows for `section`. Each returned
    /// element is exactly [`ROW_H`] tall so `uniform_list`'s single-measure
    /// layout holds.
    fn render_rows(
        &self,
        section: RecordSection,
        range: Range<usize>,
        cx: &Context<Self>,
    ) -> Vec<gpui::AnyElement> {
        let display = match section {
            RecordSection::Attestations => &self.attestations.display,
            RecordSection::Requests => &self.requests.display,
            RecordSection::Spending => &self.spending.display,
        };
        range
            .filter_map(|dix| display.get(dix).copied().map(|row| (dix, row)))
            .map(|(dix, row)| match (section, row) {
                (RecordSection::Attestations, DisplayRow::Data(i)) => {
                    self.render_attestation_row(dix, i, cx)
                }
                (RecordSection::Requests, DisplayRow::Data(i)) => {
                    self.render_request_row(dix, i, cx)
                }
                (RecordSection::Spending, DisplayRow::Header(i)) => self.render_spend_header(i, cx),
                (RecordSection::Spending, DisplayRow::Data(i)) => self.render_spend_row(i, cx),
                (_, DisplayRow::LoadMore) => self.render_load_more_row(cx),
                // No other (section, row) combinations are produced by the
                // display builders.
                _ => div().h(ROW_H).into_any_element(),
            })
            .collect()
    }

    /// One fixed-height listing row shell: the hover background, click target,
    /// hairline top rule (between rows), and `ROW_H` height shared by every
    /// row kind. `dix` is the display index (so the first *visible* row in a
    /// virtualized window still gets no top rule when it's display row 0).
    fn row_shell(
        &self,
        id: (&'static str, usize),
        dix: usize,
        cx: &Context<Self>,
    ) -> gpui::Stateful<Div> {
        let theme = cx.theme();
        let mut row = v_flex()
            .id(id)
            .w_full()
            .h(ROW_H)
            .justify_center()
            .gap_0p5()
            .cursor_pointer()
            .hover(|s| s.bg(theme.muted.opacity(0.35)));
        if dix > 0 {
            row = row.border_t_1().border_color(theme.border);
        }
        row
    }

    /// The trailing load-more row: a quiet "Load more…" affordance, or — when
    /// a page fetch is in flight — an honest "Loading more…" in-flight row
    /// (it maps to a real task).
    fn render_load_more_row(&self, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let loading = match self.section {
            RecordSection::Attestations => self.attestations.loading,
            RecordSection::Requests => self.requests.loading,
            RecordSection::Spending => self.spending.loading,
        };
        if loading {
            return div()
                .w_full()
                .h(ROW_H)
                .flex()
                .items_center()
                .text_sm()
                .italic()
                .text_color(theme.muted_foreground)
                .child("Loading more…")
                .into_any_element();
        }
        div()
            .id("load-more")
            .w_full()
            .h(ROW_H)
            .flex()
            .items_center()
            .text_sm()
            .cursor_pointer()
            .text_color(theme.muted_foreground)
            .hover(|s| s.text_color(theme.foreground))
            .child("Load more…")
            .on_click(cx.listener(|this, _, _, cx| this.load_more(cx)))
            .into_any_element()
    }

    // --- Attestations -----------------------------------------------------

    fn render_attestation_row(&self, dix: usize, i: usize, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let a = &self.attestations.rows[i];
        let hash = a.hash.clone();
        let sub = format!(
            "{} · {} · {}",
            match a.pcr_digest.as_deref() {
                Some(d) => format!("pcr {}", truncate_middle(d, 18)),
                None => "no pcr digest".to_string(),
            },
            plural(a.connection_count, "connection"),
            format_bytes(a.doc_bytes),
        );
        self.row_shell(("attestation", dix), dix, cx)
            .on_click(cx.listener(move |this, _, _, cx| this.open_attestation(hash.clone(), cx)))
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_baseline()
                    .gap_4()
                    .child(mono(13.).child(SharedString::from(truncate_middle(&a.hash, 44))))
                    .child(
                        div()
                            .text_xs()
                            .whitespace_nowrap()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(fmt_utc(a.created_at, false))),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(sub)),
            )
            .into_any_element()
    }

    fn render_attestation_detail(&self, d: &AttestationDetail, cx: &Context<Self>) -> Div {
        let col = self
            .list_frame()
            .child(self.back_row("Attestations", cx))
            .child(kv_row(
                "Hash",
                mono(12.).child(SharedString::from(d.hash.clone())),
                cx,
            ))
            .child(kv_row(
                "PCR digest",
                mono(12.).child(SharedString::from(
                    d.pcr_digest.clone().unwrap_or_else(|| "—".into()),
                )),
                cx,
            ))
            .child(kv_row(
                "Recorded",
                div()
                    .text_sm()
                    .child(SharedString::from(fmt_utc(d.created_at, true))),
                cx,
            ))
            .child(kv_row(
                "Size",
                div()
                    .text_sm()
                    .child(SharedString::from(format_bytes(d.doc.len() as i64))),
                cx,
            ));
        // The payloads are projected + fenced once in `DetailCache::build`;
        // the render path only reads them.
        col.children(self.detail_payload_sections(cx))
    }

    // --- Requests -----------------------------------------------------------

    fn render_request_row(&self, dix: usize, i: usize, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let r = &self.requests.rows[i];
        let id = r.id.clone();

        // Status: the honest outcome — HTTP status, or the recorded error,
        // or "no response".
        let (status_text, status_color) = match (r.response_status, r.error.as_deref()) {
            (Some(s), _) if (200..400).contains(&s) => (s.to_string(), theme.muted_foreground),
            (Some(s), _) => (s.to_string(), theme.danger),
            (None, Some(_)) => ("failed".to_string(), theme.danger),
            (None, None) => ("no response".to_string(), theme.muted_foreground),
        };

        let mut sub_parts: Vec<String> = Vec::new();
        if let Some(t) = r.transport.as_deref() {
            sub_parts.push(t.to_string());
        }
        if let Some(d) = r.duration_ms {
            sub_parts.push(format!("{d} ms"));
        }
        if r.attempt_number > 1 {
            sub_parts.push(format!("attempt {}", r.attempt_number));
        }
        if let Some(n) = r.credential_nonce.as_deref() {
            sub_parts.push(format!("credential {}", truncate_middle(n, 14)));
        }
        if r.attestation_hash.is_some() {
            sub_parts.push("attested".to_string());
        }
        if let Some(e) = r.error.as_deref() {
            sub_parts.push(e.to_string());
        }
        sub_parts.push(fmt_utc(r.request_at, false));

        self.row_shell(("request", dix), dix, cx)
            .on_click(cx.listener(move |this, _, _, cx| this.open_request(id.clone(), cx)))
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_baseline()
                    .gap_4()
                    .child(mono(13.).child(SharedString::from(format!("{} {}", r.method, r.path))))
                    .child(
                        div()
                            .text_xs()
                            .whitespace_nowrap()
                            .text_color(status_color)
                            .child(SharedString::from(status_text)),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(sub_parts.join(" · "))),
            )
            .into_any_element()
    }

    fn render_request_detail(&self, d: &RequestDetail, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let mut col = self
            .list_frame()
            .child(self.back_row(self.section.label(), cx))
            .child(kv_row(
                "Request",
                mono(12.).child(SharedString::from(format!("{} {}", d.method, d.path))),
                cx,
            ));

        if let Some(base) = d.base_url.as_deref() {
            col = col.child(kv_row(
                "Endpoint",
                mono(12.).child(SharedString::from(base.to_string())),
                cx,
            ));
        }
        if let Some(t) = d.transport.as_deref() {
            col = col.child(kv_row(
                "Transport",
                div().text_sm().child(SharedString::from(t.to_string())),
                cx,
            ));
        }

        let status_line = match (d.response_status, d.error.as_deref()) {
            (Some(s), None) => s.to_string(),
            (Some(s), Some(e)) => format!("{s} — {e}"),
            (None, Some(e)) => format!("no response — {e}"),
            (None, None) => "no response recorded".to_string(),
        };
        let status_color = match (d.response_status, d.error.as_deref()) {
            (Some(s), None) if (200..400).contains(&s) => theme.foreground,
            (None, None) => theme.foreground,
            _ => theme.danger,
        };
        col = col.child(kv_row(
            "Status",
            div()
                .text_sm()
                .text_color(status_color)
                .child(SharedString::from(status_line)),
            cx,
        ));

        let mut timing = fmt_utc(d.request_at, true);
        if let Some(ms) = d.duration_ms {
            timing = format!("{timing} · {ms} ms");
        }
        col = col.child(kv_row(
            "Sent",
            div().text_sm().child(SharedString::from(timing)),
            cx,
        ));

        if let Some(hash) = d.attestation_hash.clone() {
            let hash_for_click = hash.clone();
            col = col.child(kv_row(
                "Attestation",
                mono(12.)
                    .id("open-attestation")
                    .cursor_pointer()
                    .text_color(theme.link)
                    .hover(|s| s.text_color(theme.link_hover))
                    .child(SharedString::from(truncate_middle(&hash, 44)))
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.open_attestation(hash_for_click.clone(), cx)
                    })),
                cx,
            ));
        }
        if let Some(n) = d.credential_nonce.as_deref() {
            col = col.child(kv_row(
                "Credential",
                mono(12.).child(SharedString::from(n.to_string())),
                cx,
            ));
        }
        if d.attempt_number > 1 || d.retry_of_id.is_some() {
            let mut s = format!("attempt {}", d.attempt_number);
            if let Some(prev) = d.retry_of_id.as_deref() {
                s = format!("{s} — retry of {prev}");
            }
            col = col.child(kv_row(
                "Retry",
                div().text_sm().child(SharedString::from(s)),
                cx,
            ));
        }

        // Space cross-link: when this request belongs to a conversation, show
        // a quiet link that opens the space window.  Label is the space title
        // when one exists, the bare id otherwise — either way the user can
        // jump from the raw trail back to the conversation.
        if let (Some(space_id), space_label) = (
            d.space_id.clone(),
            d.space_title
                .clone()
                .unwrap_or_else(|| d.space_id.clone().unwrap_or_default()),
        ) {
            let stores = self.stores.clone();
            col = col.child(kv_row(
                "Space",
                div()
                    .id("open-space")
                    .text_sm()
                    .cursor_pointer()
                    .text_color(theme.link)
                    .hover(|s| s.text_color(theme.link_hover))
                    .child(SharedString::from(space_label))
                    .on_click(cx.listener(move |_, _, _, cx| {
                        let stores = stores.clone();
                        let id = space_id.clone();
                        cx.defer(move |cx| {
                            crate::open_space_window(cx, stores, id);
                        });
                    })),
                cx,
            ));
        }

        // The payloads are projected + fenced once in `DetailCache::build`;
        // the render path only reads them.
        col.children(self.detail_payload_sections(cx))
    }

    // --- Spending --------------------------------------------------------

    /// A spending group header row. `i` is the index of the first data row of
    /// the group, which carries the credential identity and held/charged
    /// total. Fixed [`ROW_H`] height like every other listing row — the group
    /// rhythm is the even row spacing, not extra top padding (a minor
    /// intentional change from the unvirtualized layout; see AGENTS.md).
    fn render_spend_header(&self, i: usize, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let e = &self.spending.rows[i];
        let mut head_line = format!("credential {}", truncate_middle(&e.credential_nonce, 24));
        head_line = format!("{head_line} · {}", e.credential_state);
        let mut header = h_flex()
            .w_full()
            .h(ROW_H)
            .justify_between()
            .items_end()
            .pb_1()
            .border_b_1()
            .border_color(theme.border)
            .child(
                mono(12.)
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(head_line)),
            );
        if let Some(amount) = e.spend_amount {
            header = header.child(
                div()
                    .text_xs()
                    .whitespace_nowrap()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(format!(
                        "{} credits {}",
                        crate::plans::format_credits(amount),
                        if e.credential_state == "spending" {
                            "held"
                        } else {
                            "charged"
                        }
                    ))),
            );
        }
        header.into_any_element()
    }

    /// A spending data row (clicks through to the request detail).
    fn render_spend_row(&self, i: usize, cx: &Context<Self>) -> gpui::AnyElement {
        let theme = cx.theme();
        let e = &self.spending.rows[i];
        let id = e.request_id.clone();
        let mut sub_parts: Vec<String> = Vec::new();
        if let Some(m) = e.model.as_deref() {
            sub_parts.push(m.to_string());
        }
        if let Some(t) = e.action_type.as_deref() {
            sub_parts.push(t.to_string());
        }
        match (e.space_title.as_deref(), e.space_id.as_deref()) {
            (Some(t), _) => sub_parts.push(format!("in “{t}”")),
            (None, Some(_)) => sub_parts.push("in untitled space".to_string()),
            (None, None) => {}
        }
        sub_parts.push(fmt_utc(e.request_at, false));

        v_flex()
            .id(("spend", i))
            .w_full()
            .h(ROW_H)
            .justify_center()
            .gap_0p5()
            .cursor_pointer()
            .hover(|s| s.bg(theme.muted.opacity(0.35)))
            .on_click(cx.listener(move |this, _, _, cx| this.open_request(id.clone(), cx)))
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_baseline()
                    .gap_4()
                    .child(mono(13.).child(SharedString::from(format!("{} {}", e.method, e.path))))
                    .child(
                        div()
                            .text_xs()
                            .whitespace_nowrap()
                            .text_color(theme.muted_foreground)
                            .child(SharedString::from(match e.credits_consumed {
                                Some(c) => {
                                    format!("{} credits", crate::plans::format_credits(c))
                                }
                                None => "—".to_string(),
                            })),
                    ),
            )
            .child(
                div()
                    .text_xs()
                    .text_color(theme.muted_foreground)
                    .child(SharedString::from(sub_parts.join(" · "))),
            )
            .into_any_element()
    }

    // --- Detail dispatch + cached payloads --------------------------------

    /// Render the open detail (attestation document or request/response
    /// pair). The expensive raw-payload projection is read from the cache,
    /// not recomputed.
    fn render_detail(&self, cx: &Context<Self>) -> Div {
        match self.detail.as_ref().map(|c| &c.detail) {
            Some(RecordDetail::Attestation(d)) => {
                let d = d.clone();
                self.render_attestation_detail(&d, cx)
            }
            Some(RecordDetail::Request(d)) => {
                let d = (**d).clone();
                self.render_request_detail(&d, cx)
            }
            None => self.list_frame(),
        }
    }

    /// The cached raw-payload sections for the open detail (headers/bodies or
    /// the attestation document). Each renders the fenced string computed
    /// once when the detail opened.
    fn detail_payload_sections(&self, cx: &Context<Self>) -> Vec<Div> {
        let Some(cache) = self.detail.as_ref() else {
            return Vec::new();
        };
        cache.payloads.iter().map(|p| raw_section(p, cx)).collect()
    }

    // --- Shared chrome ------------------------------------------------------

    fn back_row(&self, label: &'static str, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        h_flex().pb_2().child(
            div()
                .id("back")
                .text_sm()
                .cursor_pointer()
                .text_color(theme.muted_foreground)
                .hover(|s| s.text_color(theme.foreground))
                .child(SharedString::from(format!("‹ {label}")))
                .on_click(cx.listener(|this, _, _, cx| this.close_detail(cx))),
        )
    }
}

/// Mono (Menlo) text container for raw data — never used for chrome.
fn mono(size: f32) -> Div {
    div().font_family("Menlo").text_size(px(size))
}

fn kv_row<C: IntoElement>(label: &'static str, value: C, cx: &Context<RecordView>) -> Div {
    let theme = cx.theme();
    h_flex()
        .w_full()
        .gap_4()
        .py_1()
        .items_start()
        .child(
            div()
                .w(px(112.))
                .flex_none()
                .text_sm()
                .text_color(theme.muted_foreground)
                .child(label),
        )
        .child(div().flex_1().min_w_0().child(value))
}

/// A raw payload section: small header with a Copy affordance, then the
/// content in a selectable mono block. The fenced (capped) string and the
/// full payload are precomputed in [`DetailCache::build`] — this only reads
/// `CachedPayload`, never re-projects. The `TextView` element id is the
/// payload's stable `id`, so its keyed `TextViewState` survives across frames
/// and the markdown is parsed exactly once (see the parse-caching note in
/// `AGENTS.md`).
fn raw_section(payload: &CachedPayload, cx: &Context<RecordView>) -> Div {
    let theme = cx.theme();
    let full = payload.full.clone();

    let mut section = v_flex().w_full().pt_4().gap_1().child(
        h_flex()
            .w_full()
            .justify_between()
            .items_baseline()
            .child(
                div()
                    .text_sm()
                    .font_medium()
                    .text_color(theme.muted_foreground)
                    .child(payload.title),
            )
            .child(
                div()
                    .id(SharedString::from(format!("copy-{}", payload.id)))
                    .text_xs()
                    .cursor_pointer()
                    .text_color(theme.muted_foreground)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("Copy")
                    .on_click(move |_, _, cx| {
                        cx.write_to_clipboard(ClipboardItem::new_string(full.clone()));
                    }),
            ),
    );

    // Default `TextViewStyle` assumes light mode; track the active
    // Circadian mode so the code-block ground stays dark at night.
    let is_dark = theme.mode.is_dark();
    let style = TextViewStyle {
        is_dark,
        highlight_theme: if is_dark {
            HighlightTheme::default_dark().clone()
        } else {
            HighlightTheme::default_light().clone()
        },
        ..TextViewStyle::default()
    };
    section = section.child(
        div().w_full().text_size(px(12.)).child(
            TextView::markdown(payload.id, payload.fenced.clone())
                .style(style)
                .selectable(true),
        ),
    );

    if let Some(note) = &payload.note {
        section = section.child(
            div()
                .text_xs()
                .italic()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(note.clone())),
        );
    }
    section
}

// ---------------------------------------------------------------------------
// Raw-data formatting
// ---------------------------------------------------------------------------

/// Wrap raw text in a markdown code fence long enough that the content can
/// never close it (verbatim rendering, mono, selectable via TextView).
fn fenced(text: &str) -> String {
    let mut longest = 0usize;
    let mut run = 0usize;
    for ch in text.chars() {
        if ch == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    let fence = "`".repeat((longest + 1).max(3));
    format!("{fence}text\n{text}\n{fence}")
}

/// Cap text for inline rendering. Returns the (possibly truncated) text and
/// an explanatory note when truncation happened.
fn cap_inline(text: &str) -> (String, Option<String>) {
    if text.len() <= INLINE_CAP {
        return (text.to_string(), None);
    }
    let mut end = INLINE_CAP;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (
        text[..end].to_string(),
        Some(format!(
            "Showing the first 64 KiB of {} — Copy carries the full data.",
            format_bytes(text.len() as i64)
        )),
    )
}

fn header_text(headers: Option<&str>) -> String {
    match headers {
        Some(h) if !h.trim().is_empty() => h.to_string(),
        _ => "(none recorded)".to_string(),
    }
}

/// Render a raw body for display: pretty-printed JSON when it parses,
/// the UTF-8 text when it decodes, a hex dump otherwise.
fn body_text(body: Option<&[u8]>) -> String {
    let Some(bytes) = body else {
        return "(none recorded)".to_string();
    };
    if bytes.is_empty() {
        return "(empty)".to_string();
    }
    render_payload_text(bytes)
}

fn render_payload_text(bytes: &[u8]) -> String {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes)
        && let Ok(pretty) = serde_json::to_string_pretty(&v)
    {
        return pretty;
    }
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => hex_dump(bytes),
    }
}

/// Plain hex, 32 bytes (64 hex chars) per line.
fn hex_dump(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2 + bytes.len() / 32 + 1);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && i % 32 == 0 {
            out.push('\n');
        }
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Middle-ellipsis truncation for hashes and nonces: keeps both ends, which
/// is what you compare when eyeballing identifiers.
fn truncate_middle(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(1);
    let head = keep / 2 + keep % 2;
    let tail = keep / 2;
    let mut out: String = chars[..head].iter().collect();
    out.push('…');
    out.extend(&chars[chars.len() - tail..]);
    out
}

fn plural(n: i64, noun: &str) -> String {
    if n == 1 {
        format!("1 {noun}")
    } else {
        format!("{n} {noun}s")
    }
}

fn format_bytes(n: i64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.1} KiB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MiB", n as f64 / (1024.0 * 1024.0))
    }
}

/// UTC timestamp for the engine room: precise, honest about its timezone.
/// `seconds` includes `:ss` (detail views); the listing keeps minutes.
fn fmt_utc(ms: i64, seconds: bool) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    if seconds {
        format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02} UTC")
    } else {
        format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02} UTC")
    }
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's `civil_from_days`
/// algorithm — exact for the proleptic Gregorian calendar.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_utc_known_instants() {
        assert_eq!(fmt_utc(0, true), "1970-01-01 00:00:00 UTC");
        // 2026-06-09 14:02:33 UTC
        assert_eq!(fmt_utc(1_781_013_753_000, true), "2026-06-09 14:02:33 UTC");
        assert_eq!(fmt_utc(1_781_013_753_000, false), "2026-06-09 14:02 UTC");
    }

    #[test]
    fn fenced_grows_past_backtick_runs() {
        assert!(fenced("plain").starts_with("```text\n"));
        let tricky = "a\n````\nb";
        let f = fenced(tricky);
        assert!(f.starts_with("`````text\n"), "{f}");
    }

    #[test]
    fn payload_text_modes() {
        assert_eq!(
            render_payload_text(b"{\"a\":1}"),
            "{\n  \"a\": 1\n}".to_string()
        );
        assert_eq!(render_payload_text(b"hello"), "hello");
        // 0xff can never appear in UTF-8 -> hex dump path.
        assert_eq!(render_payload_text(&[0xff, 0x00, 0xab]), "ff00ab");
    }

    #[test]
    fn truncate_middle_keeps_ends() {
        assert_eq!(truncate_middle("abcdef", 10), "abcdef");
        let t = truncate_middle("abcdefghijklmnop", 9);
        assert_eq!(t.chars().count(), 9);
        assert!(t.starts_with("abcd") && t.ends_with("mnop"));
    }

    #[test]
    fn cap_inline_notes_truncation() {
        let (s, note) = cap_inline("short");
        assert_eq!(s, "short");
        assert!(note.is_none());
        let big = "x".repeat(INLINE_CAP + 10);
        let (s, note) = cap_inline(&big);
        assert_eq!(s.len(), INLINE_CAP);
        assert!(note.unwrap().contains("64 KiB"));
    }
}
