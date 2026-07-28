//! Frame cost of the Record detail view — a measuring instrument, not a gate.
//!
//! Runs like the visual tier (`harness = false`, macOS only, opt in with
//! `EIDOLA_RUN_RECORD_PERF=1`) so it never costs anything in CI, and for the
//! same reason: `VisualTestAppContext` needs the AppKit main thread. It opens a
//! **real** offscreen window (Metal + CoreText — mocked rendering would measure
//! nothing) on a Record request detail, and reports two things.
//!
//! **Frame cost.** `Window::draw` — the per-frame CPU work a scroll triggers —
//! timed in four states: `idle`, `scrolled`, after a plain `clicked`, and while
//! a drag `selecting` is live. The last one is the one that matters: an active
//! text selection used to make every subsequent frame quadratic in payload
//! size (see AGENTS.md → The Record), so a payload of a few tens of KB scrolled
//! at ~2 fps in a dev build. Frame cost must stay flat across all four states
//! and grow no worse than linearly with payload size.
//!
//! **Selection fingerprints.** What a set of fixed drag geometries actually
//! selects, hashed. This is the differential check for any change to the
//! selection machinery (ours or upstream's): capture the fingerprints before
//! and after and diff them — they must be byte-identical, because the Record is
//! a forensic surface and "faster" must never mean "selects something else".
//!
//! Payloads come from real captured bytes when `EIDOLA_RECORD_PERF_REQ` /
//! `EIDOLA_RECORD_PERF_RESP` point at files (e.g. extracted from a `request`
//! row), plus a synthetic SSE-shaped ladder that brackets them by size.
//! `EIDOLA_RECORD_PERF_SHOTS=<dir>` also writes a PNG per case so the surface
//! can be eyeballed.
//!
//! Run:
//! ```text
//! EIDOLA_RUN_RECORD_PERF=1 cargo test -p eidola-gui --test record_perf --release
//! ```
//!
//! Use `--release` for absolute numbers and the plain dev profile for the
//! ratios a user actually feels — `just build gui` is unoptimized, so the dev
//! profile is the one the app ships to a developer's own machine.

#[cfg(target_os = "macos")]
fn main() {
    if !matches!(
        std::env::var("EIDOLA_RUN_RECORD_PERF").as_deref(),
        Ok("1") | Ok("true")
    ) {
        println!("record perf skipped; set EIDOLA_RUN_RECORD_PERF=1");
        return;
    }
    perf::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {}

#[cfg(target_os = "macos")]
mod perf {
    use std::sync::Arc;
    use std::time::Instant;

    use eidola_app_core::RequestDetail;
    use eidola_gui::record::{RecordDetail, RecordSection, RecordView};
    use eidola_gui::stores::{Stores, StoresStub};
    use gpui::{
        AppContext, Modifiers, MouseButton, ScrollDelta, ScrollWheelEvent, TouchPhase,
        VisualTestAppContext, point, px, size,
    };
    use gpui_component::{Root, Theme, ThemeMode};
    use gpui_component_assets::Assets;

    /// One synthetic SSE `data:` chunk, ~135 bytes — the shape a streamed chat
    /// completion actually records.
    fn sse_chunk(i: usize) -> String {
        format!(
            "data: {{\"choices\":[{{\"finish_reason\":null,\"index\":0,\"delta\":{{\"content\":\"tok{i:05}\"}}}}],\"created\":1785121066,\"id\":\"chatcmpl-eYUVknDzN1IVXGlgI6w8PiIDKRsCHW60\",\"model\":\"gemma-4-E2B_q4_0-it@local\",\"object\":\"chat.completion.chunk\"}}\n\n"
        )
    }

    fn synthetic_sse(target_bytes: usize) -> Vec<u8> {
        let mut s = String::with_capacity(target_bytes + 256);
        let mut i = 0;
        while s.len() < target_bytes {
            s.push_str(&sse_chunk(i));
            i += 1;
        }
        s.into_bytes()
    }

    fn detail(req: Vec<u8>, resp: Vec<u8>) -> RequestDetail {
        RequestDetail {
            id: "req-perf".into(),
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            request_headers: None,
            request_body: Some(req),
            response_status: Some(200),
            response_headers: None,
            response_body: Some(resp),
            request_at: 1_785_121_066_000,
            response_at: Some(1_785_121_067_310),
            duration_ms: Some(1_310),
            error: None,
            retry_of_id: None,
            attempt_number: 1,
            credential_nonce: Some("a1b2c3d4e5f60718293a4b5c6d7e8f90".into()),
            action_id: Some("act-1".into()),
            transport: Some("clearnet".into()),
            base_url: Some("https://eidola.example".into()),
            attestation_hash: None,
            space_id: None,
            space_title: None,
            backend_id: Some("local".into()),
            backend_display_name: Some("On-device".into()),
        }
    }

    struct Sample {
        label: String,
        bytes: usize,
        first_ms: f64,
        median_ms: f64,
        max_ms: f64,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Mode {
        Idle,
        Scrolled,
        Clicked,
        Selecting,
    }

    impl Mode {
        fn tag(self) -> &'static str {
            match self {
                Mode::Idle => "idle",
                Mode::Scrolled => "scrolled",
                Mode::Clicked => "clicked",
                Mode::Selecting => "selecting",
            }
        }
    }

    const MODES: &[Mode] = &[Mode::Idle, Mode::Scrolled, Mode::Clicked, Mode::Selecting];

    pub fn run() {
        let platform = gpui_platform::current_platform(false);
        let mut cx = VisualTestAppContext::with_asset_source(platform, Arc::new(Assets));
        cx.update(|cx| {
            gpui_component::init(cx);
            eidola_gui::theme::install(cx);
            Theme::change(ThemeMode::Light, None, cx);
        });

        let mut cases: Vec<(String, Vec<u8>, Vec<u8>)> = Vec::new();

        if let (Ok(rq), Ok(rs)) = (
            std::env::var("EIDOLA_RECORD_PERF_REQ"),
            std::env::var("EIDOLA_RECORD_PERF_RESP"),
        ) {
            let req = std::fs::read(&rq).expect("read request body fixture");
            let resp = std::fs::read(&rs).expect("read response body fixture");
            cases.push(("captured".to_string(), req, resp));
        }

        for kb in [4usize, 16, 32, 64] {
            cases.push((
                format!("synthetic-{kb}KiB"),
                b"{\"model\":\"gemma-4\",\"messages\":[]}".to_vec(),
                synthetic_sse(kb * 1024),
            ));
        }

        // Differential correctness pass: what a drag actually selects. Run
        // this build against the reference build and diff the fingerprints —
        // any change to the selection geometry shows up here.
        println!("--- selection fingerprints ---");
        for (label, req, resp) in &cases {
            for (name, from, to, scroll) in DRAGS {
                let text = selected_text_after_drag(
                    &mut cx,
                    req.clone(),
                    resp.clone(),
                    *from,
                    *to,
                    *scroll,
                );
                println!("{label:<18} {name:<16} {}", fingerprint(&text));
            }
        }
        println!("--- frame cost ---");

        let mut samples = Vec::new();
        for (label, req, resp) in cases {
            let bytes = req.len() + resp.len();
            for mode in MODES.iter().copied() {
                let s = measure(&mut cx, &label, mode, bytes, req.clone(), resp.clone());
                println!(
                    "{:<18} {:<10} payload {:>8} B   first {:>9.1} ms   median {:>9.1} ms   max {:>9.1} ms",
                    s.label,
                    mode.tag(),
                    s.bytes,
                    s.first_ms,
                    s.median_ms,
                    s.max_ms
                );
                samples.push(s);
            }
        }
        let _ = samples;
    }

    /// Drag geometries the fingerprint pass replays: short in-line, wide
    /// multi-line, bottom-clipped (exercises the content-mask edge), and one
    /// taken after scrolling (so the drag lands mid-payload rather than at the
    /// document top). Tuple: (name, from, to, scroll-steps-before-drag).
    #[allow(clippy::type_complexity)]
    const DRAGS: &[(&str, (f32, f32), (f32, f32), usize)] = &[
        ("short", (120., 300.), (300., 300.), 0),
        ("multiline", (120., 280.), (600., 430.), 0),
        ("clipped", (120., 500.), (700., 900.), 0),
        ("scrolled", (120., 300.), (700., 560.), 6),
        ("scrolled-wide", (60., 120.), (820., 630.), 12),
    ];

    fn fingerprint(text: &str) -> String {
        // FNV-1a — enough to detect any byte-level change, cheap and dep-free.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in text.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        let head: String = text.chars().take(24).collect();
        let tail: String = {
            let all: Vec<char> = text.chars().collect();
            all[all.len().saturating_sub(24)..].iter().collect()
        };
        format!(
            "len={:<7} fnv={h:016x}  {:?} … {:?}",
            text.len(),
            head,
            tail
        )
    }

    /// Open a detail, optionally scroll, drag from → to, and return what the
    /// window reports as selected.
    fn selected_text_after_drag(
        cx: &mut VisualTestAppContext,
        req: Vec<u8>,
        resp: Vec<u8>,
        from: (f32, f32),
        to: (f32, f32),
        scroll_steps: usize,
    ) -> String {
        let stores = cx.update(|cx| Stores::stub_with(StoresStub::default(), cx));
        let window = cx
            .open_offscreen_window(size(px(860.), px(640.)), |window, cx| {
                let view = cx.new(|cx| {
                    let mut v = RecordView::new(stores.clone(), window, cx);
                    v.select_section(RecordSection::Requests, cx);
                    v.set_detail_for_test(Some(RecordDetail::Request(Box::new(detail(req, resp)))));
                    v
                });
                cx.new(|cx| Root::new(view, window, cx))
            })
            .expect("open offscreen window");
        cx.run_until_parked();
        let handle = window.into();

        for _ in 0..scroll_steps {
            cx.simulate_event(
                handle,
                ScrollWheelEvent {
                    position: point(px(430.), px(400.)),
                    delta: ScrollDelta::Pixels(point(px(0.), px(-400.))),
                    modifiers: Modifiers::default(),
                    touch_phase: TouchPhase::Moved,
                },
            );
        }
        cx.run_until_parked();

        cx.simulate_mouse_down(
            handle,
            point(px(from.0), px(from.1)),
            MouseButton::Left,
            Modifiers::default(),
        );
        cx.simulate_mouse_move(
            handle,
            point(px(to.0), px(to.1)),
            Some(MouseButton::Left),
            Modifiers::default(),
        );
        cx.run_until_parked();
        // The selection is computed during paint, so it needs a frame. gpui
        // requires the element arena to be cleared before the next draw, and
        // `draw` hands back the token that owes it — so clear it here rather
        // than dropping it (see `measure` for the same contract).
        cx.update_window(handle, |_, window, cx| {
            window.draw(cx).clear();
        })
        .ok();

        let text = cx
            .update_window(handle, |_, window, cx| {
                use gpui_component::WindowExt as _;
                window.selected_text(cx)
            })
            .unwrap_or_default();

        cx.simulate_mouse_up(
            handle,
            point(px(to.0), px(to.1)),
            MouseButton::Left,
            Modifiers::default(),
        );
        cx.update_window(handle, |_, window, _| window.remove_window())
            .ok();
        cx.run_until_parked();
        text
    }

    fn measure(
        cx: &mut VisualTestAppContext,
        label: &str,
        mode: Mode,
        bytes: usize,
        req: Vec<u8>,
        resp: Vec<u8>,
    ) -> Sample {
        let stores = cx.update(|cx| Stores::stub_with(StoresStub::default(), cx));
        let window = cx
            .open_offscreen_window(size(px(860.), px(640.)), |window, cx| {
                let view = cx.new(|cx| {
                    let mut v = RecordView::new(stores.clone(), window, cx);
                    v.select_section(RecordSection::Requests, cx);
                    v.set_detail_for_test(Some(RecordDetail::Request(Box::new(detail(req, resp)))));
                    v
                });
                cx.new(|cx| Root::new(view, window, cx))
            })
            .expect("open offscreen window");
        cx.run_until_parked();

        let handle = window.into();

        // Put the surface into the state under test.
        let mid = point(px(430.), px(400.));
        match mode {
            Mode::Idle => {}
            Mode::Scrolled => {
                for _ in 0..10 {
                    cx.simulate_event(
                        handle,
                        ScrollWheelEvent {
                            position: mid,
                            delta: ScrollDelta::Pixels(point(px(0.), px(-400.))),
                            modifiers: Modifiers::default(),
                            touch_phase: TouchPhase::Moved,
                        },
                    );
                }
                cx.run_until_parked();
            }
            Mode::Clicked => {
                // A plain click in the body — no drag. This is the cheapest
                // possible interaction with the payload text.
                cx.simulate_click(handle, point(px(200.), px(300.)), Modifiers::default());
                cx.run_until_parked();
            }
            Mode::Selecting => {
                // Press inside the body and drag to the bottom, leaving the
                // drag active — the state a user is in after selecting text.
                cx.simulate_mouse_down(
                    handle,
                    point(px(200.), px(300.)),
                    MouseButton::Left,
                    Modifiers::default(),
                );
                cx.simulate_mouse_move(
                    handle,
                    point(px(700.), px(620.)),
                    Some(MouseButton::Left),
                    Modifiers::default(),
                );
                cx.run_until_parked();
            }
        }

        let mut times = Vec::new();
        for _ in 0..12 {
            cx.update_window(handle, |_, window, _| window.refresh())
                .ok();
            cx.run_until_parked();
            let t = cx
                .update_window(handle, |_, window, cx| {
                    let start = Instant::now();
                    let clear = window.draw(cx);
                    // Stop the clock *before* clearing: the arena teardown is
                    // gpui's per-frame bookkeeping, not the frame cost we are
                    // reporting. But clear it — the token is `#[must_use]`
                    // because gpui requires the arena empty before the next
                    // draw, so each measured frame starts from the same state.
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    clear.clear();
                    elapsed
                })
                .expect("draw");
            times.push(t);
        }
        if let Ok(dir) = std::env::var("EIDOLA_RECORD_PERF_SHOTS") {
            let _ = std::fs::create_dir_all(&dir);
            if let Ok(img) = cx.capture_screenshot(handle) {
                let _ = img.save(format!("{dir}/{label}-{}.png", mode.tag()));
            }
        }
        cx.update_window(handle, |_, window, _| window.remove_window())
            .ok();
        cx.run_until_parked();

        let first = times[0];
        let mut sorted = times.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        let max = *sorted.last().unwrap();
        Sample {
            label: label.to_string(),
            bytes,
            first_ms: first,
            median_ms: median,
            max_ms: max,
        }
    }
}
