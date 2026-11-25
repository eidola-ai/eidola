//! Localization — the app's strings and the locale they render in.
//!
//! The desktop app's pipeline (`crates/eidola-gui/AGENTS.md` → Localization):
//! `locales/<tag>/*.ftl` are checked at build time by `eidola-i18n`'s codegen
//! and embedded as string literals in the generated module below, with one
//! typed accessor per English message — `msg::library_title(cx)`. **Nothing is
//! read at runtime**: the bundle digest is what a reader verifies, and a
//! translation loaded from anywhere else would be an unmeasured input able to
//! rewrite what the app says. The fallback (an untranslated message is present
//! as English, inside the one bundle) and the negotiation (Chinese by script)
//! are the shared runtime's, so the two apps cannot disagree about either.
//!
//! **State holds the value, render chooses the words**: a view keeps the typed
//! value (an `AppError`, a count) and formats it here at render, so a locale
//! change repaints in the new language. Probe names never localize; the labels
//! do.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use fluent_bundle::{FluentBundle, FluentResource};
use gpui::{App, Global, SharedString};

include!(concat!(env!("OUT_DIR"), "/i18n_generated.rs"));

/// One argument of a localized message.
pub type Arg<'a> = fluent_bundle::FluentValue<'a>;
/// The argument bag a generated accessor builds for its message.
pub type Args<'a> = fluent_bundle::FluentArgs<'a>;

pub use eidola_i18n::runtime::SOURCE_LOCALE;

/// The active locale and the one bundle it formats through.
pub struct Localization {
    tag: &'static str,
    bundle: FluentBundle<Arc<FluentResource>>,
}

impl Global for Localization {}

impl Localization {
    fn for_locale(tag: &'static str) -> Self {
        Self {
            tag,
            bundle: eidola_i18n::runtime::bundle_for(LOCALES, tag),
        }
    }

    fn format(&self, id: &str, args: Option<&Args>) -> SharedString {
        SharedString::from(eidola_i18n::runtime::format(
            &self.bundle,
            self.tag,
            id,
            args,
        ))
    }
}

thread_local! {
    /// One bundle per locale named explicitly, built once: every resource is
    /// a constant embedded in this bundle, so a built one never goes stale.
    static BUNDLES: RefCell<HashMap<&'static str, Rc<Localization>>> =
        RefCell::new(HashMap::new());
}

fn cached(tag: &'static str) -> Rc<Localization> {
    BUNDLES.with(|bundles| {
        Rc::clone(
            bundles
                .borrow_mut()
                .entry(tag)
                .or_insert_with(|| Rc::new(Localization::for_locale(tag))),
        )
    })
}

/// Format in a locale named explicitly — the generated `msg_in` accessors.
pub fn format_in(tag: &str, id: &str, args: Option<&Args>) -> SharedString {
    let tag = eidola_i18n::runtime::shipped(LOCALES, tag).unwrap_or(SOURCE_LOCALE);
    cached(tag).format(id, args)
}

/// Format in the active locale — the generated `msg` accessors. With no
/// locale installed (a test `App`), the source locale.
pub fn format(cx: &App, id: &str, args: Option<&Args>) -> SharedString {
    match cx.try_global::<Localization>() {
        Some(l10n) => l10n.format(id, args),
        None => cached(SOURCE_LOCALE).format(id, args),
    }
}

/// The active locale's tag.
pub fn active_locale(cx: &App) -> &'static str {
    cx.try_global::<Localization>()
        .map(|l| l.tag)
        .unwrap_or(SOURCE_LOCALE)
}

/// The locale to use: the stored `language` preference when it names a
/// shipped locale, else the webview's preferred languages (which follow the
/// device's), else the source locale.
pub fn resolve(stored: Option<&str>, preferred: &[String]) -> &'static str {
    if let Some(stored) = stored.map(str::trim).filter(|s| {
        !s.is_empty() && !s.eq_ignore_ascii_case("auto") && !s.eq_ignore_ascii_case("system")
    }) && let Some(tag) = eidola_i18n::runtime::match_locale(LOCALES, stored)
    {
        return tag;
    }
    eidola_i18n::runtime::negotiate(LOCALES, preferred)
}

/// Switch the active locale and repaint. The one path a locale changes through.
pub fn apply(tag: &str, cx: &mut App) {
    let Some(tag) = eidola_i18n::runtime::shipped(LOCALES, tag) else {
        return;
    };
    if cx.has_global::<Localization>() && active_locale(cx) == tag {
        return;
    }
    cx.set_global(Localization::for_locale(tag));
    for handle in cx.windows() {
        handle.update(cx, |_, window, _| window.refresh()).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_locale_carries_the_whole_source_locale() {
        for (tag, _) in LOCALES {
            let bundle = eidola_i18n::runtime::bundle_for(LOCALES, tag);
            for id in MESSAGE_IDS {
                assert!(bundle.get_message(id).is_some(), "`{tag}` lacks `{id}`");
            }
        }
    }

    #[test]
    fn the_five_locales_ship_and_are_negotiated() {
        let tags: Vec<&str> = LOCALES.iter().map(|(t, _)| *t).collect();
        assert_eq!(tags, ["en", "es", "fr", "zh-Hans", "zh-Hant"]);
        assert_eq!(resolve(None, &["zh-TW".into()]), "zh-Hant");
        assert_eq!(resolve(None, &["zh-CN".into()]), "zh-Hans");
        assert_eq!(resolve(None, &["fr-CA".into(), "en".into()]), "fr");
        assert_eq!(resolve(Some("es"), &["fr".into()]), "es");
        assert_eq!(resolve(Some("auto"), &["de".into()]), "en");
    }

    /// Every message formats in every locale with every variable it takes —
    /// a translation that broke a placeable would show raw ids on screen.
    #[test]
    fn every_message_formats_in_every_locale() {
        let mut args = Args::new();
        for var in [
            "amount",
            "author",
            "choice",
            "contract",
            "count",
            "credits",
            "depth",
            "description",
            "documents",
            "failed",
            "generation",
            "id",
            "index",
            "interval",
            "label",
            "line",
            "logs",
            "matching",
            "model",
            "name",
            "names",
            "path",
            "percent",
            "platform",
            "reason",
            "shell",
            "status",
            "title",
            "total",
            "value",
            "version",
            "when",
        ] {
            args.set(var, "x");
        }
        for (tag, _) in LOCALES {
            let bundle = eidola_i18n::runtime::bundle_for(LOCALES, tag);
            for id in MESSAGE_IDS {
                let pattern = bundle.get_message(id).and_then(|m| m.value()).unwrap();
                let mut errors = Vec::new();
                let text = bundle.format_pattern(pattern, Some(&args), &mut errors);
                assert!(errors.is_empty(), "`{id}` in `{tag}`: {errors:?}");
                assert!(
                    !text.contains('{'),
                    "`{id}` in `{tag}` left a placeable: {text}"
                );
            }
        }
    }
}
