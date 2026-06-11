//! Shared plans-list presentation: hairline-rule rows (name · price, credits
//! underneath), no cards. Used by both the chat window's onboarding plans
//! page (`chat.rs`) and the Settings Account pane (`account.rs`) so the two
//! surfaces stay pixel-identical instead of drifting apart.

use std::rc::Rc;

use gpui::{
    App, ClickEvent, Div, InteractiveElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, Window, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use eidola_app_core::PriceInfo;

/// Handler invoked with the clicked plan's price id.
pub type PlanSelectHandler = Rc<dyn Fn(String, &mut Window, &mut App)>;

/// Render the plan rows themselves (no surrounding empty/error states —
/// those belong to the caller, which knows why the list might be empty).
/// `pending` marks the plan whose checkout request is currently in flight;
/// its price line is replaced by "Opening checkout…" (a real request — the
/// no-fake-states rule).
pub fn plan_rows(
    prices: &[PriceInfo],
    pending: Option<&str>,
    on_select: PlanSelectHandler,
    cx: &App,
) -> Div {
    let theme = cx.theme();
    let mut list = v_flex().w_full();

    for (idx, price) in prices.iter().enumerate() {
        let price_line = if pending == Some(price.id.as_str()) {
            "Opening checkout…".to_string()
        } else {
            format!("{}{}", price.amount_display, price.recurrence)
        };
        let mut subline = format!("{} credits", format_credits(price.credits));
        if let Some(desc) = price.product_description.as_deref() {
            subline = format!("{subline} — {desc}");
        }
        let price_id = price.id.clone();
        let on_select = on_select.clone();

        list = list.child(
            v_flex()
                .id(("plan", idx))
                .w_full()
                .py_3()
                .gap_1()
                .border_t_1()
                .border_color(theme.border)
                .cursor_pointer()
                .hover(|s| s.bg(theme.muted.opacity(0.35)))
                .on_click(move |_: &ClickEvent, window, cx| {
                    on_select(price_id.clone(), window, cx);
                })
                .child(
                    h_flex()
                        .w_full()
                        .justify_between()
                        .items_baseline()
                        .child(div().child(SharedString::from(price.product_name.clone())))
                        .child(
                            div()
                                .text_color(theme.muted_foreground)
                                .child(SharedString::from(price_line)),
                        ),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(SharedString::from(subline)),
                ),
        );
    }
    // Closing hairline under the last row.
    list.child(div().w_full().h(px(1.)).bg(theme.border))
}

/// Format a credit amount with thousands separators (credits are micro-USD
/// denominated, so the magnitudes are large).
pub fn format_credits(credits: i64) -> String {
    let raw = credits.abs().to_string();
    let mut out = String::with_capacity(raw.len() + raw.len() / 3 + 1);
    if credits < 0 {
        out.push('-');
    }
    let offset = raw.len() % 3;
    for (i, ch) in raw.chars().enumerate() {
        if i > 0 && (i + 3 - offset).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::format_credits;

    #[test]
    fn format_credits_separators() {
        assert_eq!(format_credits(0), "0");
        assert_eq!(format_credits(999), "999");
        assert_eq!(format_credits(1_000), "1,000");
        assert_eq!(format_credits(5_000_000), "5,000,000");
        assert_eq!(format_credits(-12_345), "-12,345");
    }
}
