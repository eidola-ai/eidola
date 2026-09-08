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

use eidola_app_core::{PriceCadence, PriceInfo};

use crate::probe::Probe;

/// Handler invoked with the clicked plan's price id.
pub type PlanSelectHandler = Rc<dyn Fn(String, &mut Window, &mut App)>;

/// Which voice a host wants its plan rows in.
///
/// **Never a formatted string, and never even a tag for the common case.** The
/// credits line takes the plan's own amount, so it cannot be pre-rendered — and
/// a sentence held across frames would be the cached-render decision the
/// localization doctrine forbids (`AGENTS.md` → "A localized string must not be
/// cached in state"). [`PlanLabels::localized`] therefore records only *that*
/// the reader's language is wanted and asks the installed global at render, the
/// path every other localized surface takes; a captured tag would be one more
/// thing that could disagree with the global it was copied from.
#[derive(Clone, Copy, Debug)]
pub struct PlanLabels {
    voice: Voice,
}

#[derive(Clone, Copy, Debug)]
enum Voice {
    /// Whatever the reader's active locale is, asked for at render.
    Active,
    /// One locale, named — for a host still English around these rows.
    Fixed(&'static str),
}

impl PlanLabels {
    /// The reader's active locale — for a host whose whole surface is
    /// localized.
    pub fn localized() -> Self {
        Self {
            voice: Voice::Active,
        }
    }

    /// The source locale, explicitly. For a host that is still English around
    /// these rows: translated rows inside an English page read worse than
    /// consistently English ones, and pinning it here keeps one definition of
    /// the copy rather than a second literal that could drift from the FTL.
    /// Its extraction swaps this for [`PlanLabels::localized`] and nothing else.
    pub fn english() -> Self {
        Self {
            voice: Voice::Fixed(crate::i18n::SOURCE_LOCALE),
        }
    }

    fn list(&self, cx: &App) -> SharedString {
        match self.voice {
            Voice::Active => crate::i18n::msg::plans_list(cx),
            Voice::Fixed(tag) => crate::i18n::msg_in::plans_list(tag),
        }
    }

    fn opening_checkout(&self, cx: &App) -> SharedString {
        match self.voice {
            Voice::Active => crate::i18n::msg::plans_opening_checkout(cx),
            Voice::Fixed(tag) => crate::i18n::msg_in::plans_opening_checkout(tag),
        }
    }

    /// The credits line for one plan: how many, and when they expire.
    ///
    /// `credits` is grouped for reading and the raw count rides beside it so
    /// the noun and its verb can agree — the balance readout's shape. The
    /// **grouping itself stays a comma** (see [`format_credits`]): locale-aware
    /// number formatting is a decision above this batch, and guessing at it
    /// here would be a second answer to it.
    fn credits(&self, credits: i64, recurring: bool, cx: &App) -> SharedString {
        let amount = format_credits(credits);
        match (self.voice, recurring) {
            (Voice::Active, true) => crate::i18n::msg::plans_credits_recurring(cx, credits, amount),
            (Voice::Active, false) => crate::i18n::msg::plans_credits_one_time(cx, credits, amount),
            (Voice::Fixed(tag), true) => {
                crate::i18n::msg_in::plans_credits_recurring(tag, credits, amount)
            }
            (Voice::Fixed(tag), false) => {
                crate::i18n::msg_in::plans_credits_one_time(tag, credits, amount)
            }
        }
    }

    /// A price's own line: what it costs, and how often.
    ///
    /// **The amount is not re-formatted here** — `amount_display` is the
    /// upstream's figure already written out, and how a locale groups digits
    /// and places its currency symbol is the same deferred decision
    /// [`format_credits`] names. What *is* localized is the words around it: the
    /// cadence, and the one case that is a word rather than a number.
    fn price(&self, price: &PriceInfo, cx: &App) -> SharedString {
        if price.amount.is_none() {
            return match self.voice {
                Voice::Active => crate::i18n::msg::plans_free(cx),
                Voice::Fixed(tag) => crate::i18n::msg_in::plans_free(tag),
            };
        }
        let amount = price.amount_display.clone();
        match &price.cadence {
            PriceCadence::OneTime => SharedString::from(amount),
            PriceCadence::Every { interval, count } => {
                let (interval, count) = (interval.clone(), *count);
                match self.voice {
                    Voice::Active => {
                        crate::i18n::msg::plans_price_cadence(cx, count, interval, amount)
                    }
                    Voice::Fixed(tag) => {
                        crate::i18n::msg_in::plans_price_cadence(tag, count, interval, amount)
                    }
                }
            }
        }
    }

    fn with_description(&self, line: SharedString, description: &str, cx: &App) -> SharedString {
        let (line, description) = (line.to_string(), description.to_string());
        match self.voice {
            Voice::Active => crate::i18n::msg::plans_credits_described(cx, line, description),
            Voice::Fixed(tag) => {
                crate::i18n::msg_in::plans_credits_described(tag, line, description)
            }
        }
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
/// The test is [`PriceCadence::OneTime`] — the typed half, not the rendered
/// `recurrence` string it is derived from: a presentation string is the wrong
/// thing to branch on, and this one is about to be worded per locale.
pub fn offered_plans(prices: &[PriceInfo], subscribed: bool) -> Vec<PriceInfo> {
    prices
        .iter()
        .filter(|p| !subscribed || p.cadence == PriceCadence::OneTime)
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
        .probe(
            format!("{name_prefix}/plans"),
            Role::ListBox,
            labels.list(cx),
        )
        .w_full();

    for (idx, price) in prices.iter().enumerate() {
        let price_line = if pending == Some(price.id.as_str()) {
            labels.opening_checkout(cx)
        } else {
            labels.price(price, cx)
        };
        // Conspicuous expiry disclosure at the point of purchase — must stay
        // consistent with the published terms (www/pages/terms.md) and the
        // server's webhook expiry logic (period end vs. one year).
        let recurring = price.cadence != PriceCadence::OneTime;
        let mut subline = labels.credits(price.credits, recurring, cx);
        if let Some(desc) = price.product_description.as_deref() {
            subline = labels.with_description(subline, desc, cx);
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
    use eidola_app_core::{PriceAmount, PriceCadence, PriceInfo};

    fn price(id: &str, cadence: PriceCadence) -> PriceInfo {
        PriceInfo {
            id: id.to_string(),
            product_name: id.to_string(),
            product_description: None,
            amount_display: "10.00 USD".to_string(),
            recurrence: match &cadence {
                PriceCadence::OneTime => String::new(),
                PriceCadence::Every { interval, .. } => format!("/{interval}"),
            },
            credits: 10_000_000,
            amount: Some(PriceAmount {
                minor_units: 1_000,
                currency: "USD".to_string(),
            }),
            cadence,
        }
    }

    fn monthly() -> PriceCadence {
        PriceCadence::Every {
            interval: "month".to_string(),
            count: 1,
        }
    }

    #[test]
    fn without_a_subscription_every_plan_is_offered() {
        let prices = [
            price("topup", PriceCadence::OneTime),
            price("monthly", monthly()),
        ];
        let offered = offered_plans(&prices, false);
        assert_eq!(offered.len(), 2);
    }

    #[test]
    fn with_a_subscription_only_one_time_top_ups_are_offered() {
        let prices = [
            price("topup", PriceCadence::OneTime),
            price("monthly", monthly()),
        ];
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
