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

use eidola_app_core::error::AppError;
use eidola_app_core::{
    AttestationDetail, AttestationInfo, RequestDetail, RequestInfo, SpendTrailEntry,
};
use gpui::{
    AsyncApp, ClipboardItem, Context, Div, FocusHandle, InteractiveElement, IntoElement,
    ParentElement, Render, SharedString, StatefulInteractiveElement, Styled, WeakEntity, Window,
    div, px,
};
use gpui_component::{
    ActiveTheme, StyledExt, h_flex,
    highlighter::HighlightTheme,
    text::{TextView, TextViewStyle},
    v_flex,
};

use crate::actions::CloseWindow;
use crate::bridge;
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

/// Per-section listing state: the rows fetched so far plus windowing flags.
struct Listing<T> {
    rows: Vec<T>,
    has_more: bool,
    loaded: bool,
    loading: bool,
}

impl<T> Default for Listing<T> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            has_more: false,
            loaded: false,
            loading: false,
        }
    }
}

pub struct RecordView {
    stores: Stores,
    section: RecordSection,
    attestations: Listing<AttestationInfo>,
    requests: Listing<RequestInfo>,
    spending: Listing<SpendTrailEntry>,
    detail: Option<RecordDetail>,
    /// Identifier (hash or request id) of a detail fetch in flight.
    detail_pending: Option<String>,
    error: Option<String>,
    focus_handle: FocusHandle,
}

impl RecordView {
    pub fn new(stores: Stores, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let focus_handle = cx.focus_handle();
        focus_handle.focus(window, cx);

        let mut this = Self {
            stores,
            section: RecordSection::Attestations,
            attestations: Listing::default(),
            requests: Listing::default(),
            spending: Listing::default(),
            detail: None,
            detail_pending: None,
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
        self.detail.as_ref()
    }

    pub fn detail_pending(&self) -> Option<&str> {
        self.detail_pending.as_deref()
    }

    /// Switch sections. Closes any open detail; fetches the section's first
    /// page on first visit.
    pub fn select_section(&mut self, section: RecordSection, cx: &mut Context<Self>) {
        self.detail = None;
        self.detail_pending = None;
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

    /// Re-fetch the current section from the top.
    pub fn refresh(&mut self, cx: &mut Context<Self>) {
        match self.section {
            RecordSection::Attestations => self.attestations = Listing::default(),
            RecordSection::Requests => self.requests = Listing::default(),
            RecordSection::Spending => self.spending = Listing::default(),
        }
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
            ($listing:ident, $helper:ident) => {{
                if self.$listing.loading {
                    return;
                }
                self.$listing.loading = true;
                let offset = self.$listing.rows.len() as i64;
                let rx = bridge::$helper(app_core, PAGE + 1, offset);
                cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
                    let res = rx.await.unwrap_or_else(|_| {
                        Err(AppError::Internal {
                            message: "record query cancelled".into(),
                        })
                    });
                    let _ = this.update(cx, |this, cx| {
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
                        cx.notify();
                    });
                })
                .detach();
            }};
        }
        match section {
            RecordSection::Attestations => fetch!(attestations, list_attestations),
            RecordSection::Requests => fetch!(requests, list_requests),
            RecordSection::Spending => fetch!(spending, spend_trail),
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
        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "record query cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                if this.detail_pending.as_deref() != Some(hash.as_str()) {
                    return; // superseded by another click
                }
                this.detail_pending = None;
                match res {
                    Ok(Some(d)) => this.detail = Some(RecordDetail::Attestation(d)),
                    Ok(None) => this.error = Some(format!("attestation not found: {hash}")),
                    Err(e) => this.error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
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
        cx.spawn(async move |this: WeakEntity<Self>, cx: &mut AsyncApp| {
            let res = rx.await.unwrap_or_else(|_| {
                Err(AppError::Internal {
                    message: "record query cancelled".into(),
                })
            });
            let _ = this.update(cx, |this, cx| {
                if this.detail_pending.as_deref() != Some(id.as_str()) {
                    return;
                }
                this.detail_pending = None;
                match res {
                    Ok(Some(d)) => this.detail = Some(RecordDetail::Request(Box::new(d))),
                    Ok(None) => this.error = Some(format!("request not found: {id}")),
                    Err(e) => this.error = Some(e.to_string()),
                }
                cx.notify();
            });
        })
        .detach();
    }

    /// Back from a detail to the section listing.
    pub fn close_detail(&mut self, cx: &mut Context<Self>) {
        self.detail = None;
        self.detail_pending = None;
        cx.notify();
    }

    // --- Test setters (stub cores can't drive the async fetches) ---------

    #[doc(hidden)]
    pub fn set_attestations_for_test(&mut self, rows: Vec<AttestationInfo>, has_more: bool) {
        self.attestations.rows = rows;
        self.attestations.has_more = has_more;
        self.attestations.loaded = true;
    }

    #[doc(hidden)]
    pub fn set_requests_for_test(&mut self, rows: Vec<RequestInfo>, has_more: bool) {
        self.requests.rows = rows;
        self.requests.has_more = has_more;
        self.requests.loaded = true;
    }

    #[doc(hidden)]
    pub fn set_spending_for_test(&mut self, rows: Vec<SpendTrailEntry>, has_more: bool) {
        self.spending.rows = rows;
        self.spending.has_more = has_more;
        self.spending.loaded = true;
    }

    #[doc(hidden)]
    pub fn set_detail_for_test(&mut self, detail: Option<RecordDetail>) {
        self.detail = detail;
        self.detail_pending = None;
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

        let body: gpui::AnyElement = if let Some(detail) = self.detail.as_ref() {
            match detail {
                RecordDetail::Attestation(d) => {
                    let d = d.clone();
                    self.render_attestation_detail(&d, cx).into_any_element()
                }
                RecordDetail::Request(d) => {
                    let d = d.clone();
                    self.render_request_detail(&d, cx).into_any_element()
                }
            }
        } else if self.detail_pending.is_some() {
            div()
                .px_6()
                .py_4()
                .italic()
                .text_color(muted_fg)
                .child("Loading…")
                .into_any_element()
        } else {
            match self.section {
                RecordSection::Attestations => self.render_attestations(cx).into_any_element(),
                RecordSection::Requests => self.render_requests(cx).into_any_element(),
                RecordSection::Spending => self.render_spending(cx).into_any_element(),
            }
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
            .child(
                div()
                    .id("record-scroll")
                    .flex_1()
                    .w_full()
                    .overflow_y_scroll()
                    .child(body),
            )
    }
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

        strip
            .child(div().flex_1())
            .child(
                div()
                    .id("record-refresh")
                    .text_xs()
                    .cursor_pointer()
                    .text_color(theme.muted_foreground)
                    .hover(|s| s.text_color(theme.foreground))
                    .child("Refresh")
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

    fn load_more_row(&self, cx: &Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        div()
            .id("load-more")
            .pt_3()
            .text_sm()
            .cursor_pointer()
            .text_color(theme.muted_foreground)
            .hover(|s| s.text_color(theme.foreground))
            .child("Load more…")
            .on_click(cx.listener(|this, _, _, cx| this.load_more(cx)))
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

    // --- Attestations -----------------------------------------------------

    fn render_attestations(&self, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let mut col = self.list_frame();

        if self.attestations.rows.is_empty() {
            if self.attestations.loading {
                return col.child(self.empty_line("Loading…", cx));
            }
            col = col.child(self.empty_line("No attestations recorded yet.", cx));
            if let Some(err) = self.error_line(cx) {
                col = col.child(err);
            }
            return col;
        }

        for (idx, a) in self.attestations.rows.iter().enumerate() {
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
            let mut row = v_flex()
                .id(("attestation", idx))
                .w_full()
                .py_2p5()
                .gap_0p5()
                .cursor_pointer()
                .hover(|s| s.bg(theme.muted.opacity(0.35)))
                .on_click(
                    cx.listener(move |this, _, _, cx| this.open_attestation(hash.clone(), cx)),
                )
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
                );
            if idx > 0 {
                row = row.border_t_1().border_color(theme.border);
            }
            col = col.child(row);
        }
        if self.attestations.has_more {
            col = col.child(self.load_more_row(cx));
        }
        if let Some(err) = self.error_line(cx) {
            col = col.child(err);
        }
        col
    }

    fn render_attestation_detail(&self, d: &AttestationDetail, cx: &Context<Self>) -> Div {
        let mut col = self
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

        let text = render_payload_text(&d.doc);
        col = col.child(raw_section("Document", "attestation-doc", &text, cx));
        col
    }

    // --- Requests -----------------------------------------------------------

    fn render_requests(&self, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let mut col = self.list_frame();

        if self.requests.rows.is_empty() {
            if self.requests.loading {
                return col.child(self.empty_line("Loading…", cx));
            }
            col = col.child(self.empty_line("No requests recorded yet.", cx));
            if let Some(err) = self.error_line(cx) {
                col = col.child(err);
            }
            return col;
        }

        for (idx, r) in self.requests.rows.iter().enumerate() {
            let id = r.id.clone();

            // Status: the honest outcome — HTTP status, or the recorded
            // error, or "no response".
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

            let mut row = v_flex()
                .id(("request", idx))
                .w_full()
                .py_2p5()
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
                        .child(
                            mono(13.).child(SharedString::from(format!("{} {}", r.method, r.path))),
                        )
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
                );
            if idx > 0 {
                row = row.border_t_1().border_color(theme.border);
            }
            col = col.child(row);
        }
        if self.requests.has_more {
            col = col.child(self.load_more_row(cx));
        }
        if let Some(err) = self.error_line(cx) {
            col = col.child(err);
        }
        col
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

        col = col.child(raw_section(
            "Request headers",
            "req-headers",
            &header_text(d.request_headers.as_deref()),
            cx,
        ));
        col = col.child(raw_section(
            "Request body",
            "req-body",
            &body_text(d.request_body.as_deref()),
            cx,
        ));
        col = col.child(raw_section(
            "Response headers",
            "resp-headers",
            &header_text(d.response_headers.as_deref()),
            cx,
        ));
        col = col.child(raw_section(
            "Response body",
            "resp-body",
            &body_text(d.response_body.as_deref()),
            cx,
        ));
        col
    }

    // --- Spending --------------------------------------------------------

    fn render_spending(&self, cx: &Context<Self>) -> Div {
        let theme = cx.theme();
        let mut col = self.list_frame();

        if self.spending.rows.is_empty() {
            if self.spending.loading {
                return col.child(self.empty_line("Loading…", cx));
            }
            col = col.child(self.empty_line("Nothing spent yet.", cx));
            if let Some(err) = self.error_line(cx) {
                col = col.child(err);
            }
            return col;
        }

        let mut prev_nonce: Option<&str> = None;
        for (idx, e) in self.spending.rows.iter().enumerate() {
            // Group header whenever the credential changes (rows are
            // time-ordered; one credential serves a run of consecutive
            // requests, so consecutive grouping matches reality).
            if prev_nonce != Some(e.credential_nonce.as_str()) {
                let mut head_line =
                    format!("credential {}", truncate_middle(&e.credential_nonce, 24));
                head_line = format!("{head_line} · {}", e.credential_state);
                let mut header = h_flex()
                    .w_full()
                    .justify_between()
                    .items_baseline()
                    .pt(if idx == 0 { px(4.) } else { px(20.) })
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
                col = col.child(header);
                prev_nonce = Some(e.credential_nonce.as_str());
            }

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

            let row = v_flex()
                .id(("spend", idx))
                .w_full()
                .py_2()
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
                        .child(
                            mono(13.).child(SharedString::from(format!("{} {}", e.method, e.path))),
                        )
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
                );
            col = col.child(row);
        }
        if self.spending.has_more {
            col = col.child(self.load_more_row(cx));
        }
        if let Some(err) = self.error_line(cx) {
            col = col.child(err);
        }
        col
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
/// content in a selectable mono block. Content larger than [`INLINE_CAP`]
/// renders its head with an honest note; Copy always carries everything.
fn raw_section(title: &'static str, id: &'static str, text: &str, cx: &Context<RecordView>) -> Div {
    let theme = cx.theme();
    let full = text.to_string();
    let (shown, capped) = cap_inline(text);

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
                    .child(title),
            )
            .child(
                div()
                    .id(SharedString::from(format!("copy-{id}")))
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
            TextView::markdown(id, fenced(&shown))
                .style(style)
                .selectable(true),
        ),
    );

    if let Some(note) = capped {
        section = section.child(
            div()
                .text_xs()
                .italic()
                .text_color(theme.muted_foreground)
                .child(SharedString::from(note)),
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
