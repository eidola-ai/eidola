//! Shared plans-list presentation: hairline-rule rows (name · price, credits
//! underneath), no cards. Used by both the onboarding window's plans slide
//! (`onboarding/`) and the Settings Account pane (`account.rs`) so the two
//! surfaces stay pixel-identical instead of drifting apart.
//!
//! **The component holds no strings of its own; its caller names the locale.**
//! A shared component that localized unconditionally would put translated plan
//! rows inside the Account pane's otherwise-English page — the failure the
//! wholesale-menu rule and `load_error_panel`'s caller-supplied labels both
//! name. So the copy lives in `locales/*/plans.ftl` and [`PlanLabels`] carries
//! the *tag* to read it in: onboarding passes the reader's locale, Settings
//! pins the source locale until its own extraction, and there is still exactly
//! one definition of each sentence.

use std::rc::Rc;

use gpui::{
    App, ClickEvent, Div, InteractiveElement, ParentElement, Role, SharedString, Stateful,
    StatefulInteractiveElement, Styled, Window, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use eidola_app_core::PriceInfo;

use crate::probe::Probe;

/// Handler invoked with the clicked plan's price id.
pub type PlanSelectHandler = Rc<dyn Fn(String, &mut Window, &mut App)>;

/// Which locale a host wants its plan rows worded in.
///
/// **A tag, never a formatted string.** The credits line takes the plan's own
/// amount, so it cannot be pre-rendered — and a sentence held across frames
/// would be the cached-render decision the localization doctrine forbids
/// (`AGENTS.md` → "A localized string must not be cached in state"). Both
/// constructors are called *at render*, and everything is formatted inside
/// [`plan_rows`] from what they name.
#[derive(Clone, Copy, Debug)]
pub struct PlanLabels {
    locale: &'static str,
}

impl PlanLabels {
    /// The reader's active locale — for a host whose whole surface is
    /// localized.
    pub fn localized(cx: &App) -> Self {
        Self {
            locale: crate::i18n::active_locale(cx),
        }
    }

    /// The source locale, explicitly. For a host that is still English around
    /// these rows: translated rows inside an English page read worse than
    /// consistently English ones, and pinning it here keeps one definition of
    /// the copy rather than a second literal that could drift from the FTL.
    /// Its extraction swaps this for [`PlanLabels::localized`] and nothing else.
    pub fn english() -> Self {
        Self {
            locale: crate::i18n::SOURCE_LOCALE,
        }
    }

    fn list(&self) -> SharedString {
        crate::i18n::msg_in::plans_list(self.locale)
    }

    fn opening_checkout(&self) -> SharedString {
        crate::i18n::msg_in::plans_opening_checkout(self.locale)
    }

    /// The credits line for one plan: how many, and when they expire.
    ///
    /// `credits` is grouped for reading and the raw count rides beside it so
    /// the noun and its verb can agree — the balance readout's shape. The
    /// **grouping itself stays a comma** (see [`format_credits`]): locale-aware
    /// number formatting is a decision above this batch, and guessing at it
    /// here would be a second answer to it.
    fn credits(&self, credits: i64, recurring: bool) -> SharedString {
        let amount = format_credits(credits);
        if recurring {
            crate::i18n::msg_in::plans_credits_recurring(self.locale, credits, amount)
        } else {
            crate::i18n::msg_in::plans_credits_one_time(self.locale, credits, amount)
        }
    }

    fn with_description(&self, line: SharedString, description: &str) -> SharedString {
        crate::i18n::msg_in::plans_credits_described(
            self.locale,
            line.to_string(),
            description.to_string(),
        )
    }
}

/// The plans a surface may offer given whether a subscription is already in
/// force.
///
/// With one in force the server refuses to start a second, so listing
/// recurring plans would only sell a refusal — the subscription itself is
/// managed through the billing portal instead. One-time top-ups are
/// unaffected and stay offered, which is why this filters rather than
/// hiding the list.
///
/// `recurrence` is empty exactly for one-time prices (`account_prices`
/// derives it from the price's recurring interval), so it is the test.
pub fn offered_plans(prices: &[PriceInfo], subscribed: bool) -> Vec<PriceInfo> {
    prices
        .iter()
        .filter(|p| !subscribed || p.recurrence.is_empty())
        .cloned()
        .collect()
}

/// Render the plan rows themselves (no surrounding empty/error states —
/// those belong to the caller, which knows why the list might be empty).
/// `pending` marks the plan whose checkout request is currently in flight;
/// its price line is replaced by "Opening checkout…" (a real request — the
/// no-fake-states rule).
pub fn plan_rows(
    prices: &[PriceInfo],
    pending: Option<&str>,
    on_select: PlanSelectHandler,
    name_prefix: &str,
    labels: PlanLabels,
    cx: &App,
) -> Stateful<Div> {
    let theme = cx.theme();
    // The plan list is a single-select listbox; each caller scopes its row
    // probe names (`{prefix}/plan/{idx}`) so the same shared component is
    // addressable in both the onboarding and Settings hosts.
    let mut list = v_flex()
        .id(SharedString::from(format!("{name_prefix}/plans")))
        .probe(format!("{name_prefix}/plans"), Role::ListBox, labels.list())
        .w_full();

    for (idx, price) in prices.iter().enumerate() {
        let price_line = if pending == Some(price.id.as_str()) {
            labels.opening_checkout()
        } else {
            SharedString::from(format!("{}{}", price.amount_display, price.recurrence))
        };
        // Conspicuous expiry disclosure at the point of purchase — must stay
        // consistent with the published terms (www/pages/terms.md) and the
        // server's webhook expiry logic (period end vs. one year). `recurrence`
        // is empty exactly for one-time prices, so it is the test.
        let mut subline = labels.credits(price.credits, !price.recurrence.is_empty());
        if let Some(desc) = price.product_description.as_deref() {
            subline = labels.with_description(subline, desc);
        }
        let price_id = price.id.clone();
        let on_select = on_select.clone();
        let plan_aria = format!("{} — {}", price.product_name, price_line);

        list = list.child(
            v_flex()
                .id(("plan", idx))
                // Name and price make the option's name; the subline — how many
                // credits, and the expiry disclosure that must stay visible at
                // the point of purchase — is the value, so it is not lost to a
                // reader who only hears the row's name.
                .probe_value(
                    format!("{name_prefix}/plan/{idx}"),
                    Role::ListBoxOption,
                    plan_aria,
                    subline.clone(),
                )
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
                        .child(div().text_color(theme.muted_foreground).child(price_line)),
                )
                .child(
                    div()
                        .text_sm()
                        .text_color(theme.muted_foreground)
                        .child(subline),
                ),
        );
    }
    // Closing hairline under the last row.
    list.child(div().w_full().h(px(1.)).bg(theme.border))
}

/// Format a credit amount with thousands separators (credits are micro-USD
/// denominated, so the magnitudes are large).
///
/// **The separator is a comma in every locale, deliberately.** Which grouping
/// and decimal marks a locale wants is one question with one answer for the
/// whole app — it reaches `format_credits`, the Record's byte sizes, the
/// relative-time helpers and the clock — and it is a decision reserved to the
/// maintainer (nothing registers a Fluent `NUMBER`, and ICU4X's decimal crate
/// is not in the graph). Guessing at it here would be a second answer.
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
    use super::{format_credits, offered_plans};
    use eidola_app_core::PriceInfo;

    fn price(id: &str, recurrence: &str) -> PriceInfo {
        PriceInfo {
            id: id.to_string(),
            product_name: id.to_string(),
            product_description: None,
            amount_display: "10.00 USD".to_string(),
            recurrence: recurrence.to_string(),
            credits: 10_000_000,
        }
    }

    #[test]
    fn without_a_subscription_every_plan_is_offered() {
        let prices = [price("topup", ""), price("monthly", "/month")];
        let offered = offered_plans(&prices, false);
        assert_eq!(offered.len(), 2);
    }

    #[test]
    fn with_a_subscription_only_one_time_top_ups_are_offered() {
        let prices = [price("topup", ""), price("monthly", "/month")];
        let offered = offered_plans(&prices, true);
        assert_eq!(offered.len(), 1);
        assert_eq!(offered[0].id, "topup");
    }

    #[test]
    fn format_credits_separators() {
        assert_eq!(format_credits(0), "0");
        assert_eq!(format_credits(999), "999");
        assert_eq!(format_credits(1_000), "1,000");
        assert_eq!(format_credits(5_000_000), "5,000,000");
        assert_eq!(format_credits(-12_345), "-12,345");
    }
}
