//! Localization — the app's user-facing strings and the locale they render in.
//!
//! **Every string ships inside the binary.** The FTL under `locales/` is turned
//! into string literals by `build.rs` and embedded in the generated module
//! below; nothing is ever read from disk at runtime. That is a security
//! property, not a packaging convenience: Eidola's artifact hash is its trust
//! root, and a translation file loaded at runtime would be an unmeasured input
//! able to rewrite security-critical UI text — "this connection could not be
//! verified" becomes "verified" with a text edit and no signature to break. The
//! cost is that adding a language is a release, which is the right trade.
//!
//! **Call sites use the generated accessors, never raw ids** — `msg::about_title(cx)`,
//! one function per message with one parameter per placeable variable. A
//! mistyped id is a missing function and a changed argument set is a changed
//! signature, so both are compile errors. The build-time rules (malformed FTL in
//! *any* locale, a translation reading a variable English never passes, a
//! translation inventing an id) live in `build_support/codegen.rs`.
//!
//! **A missing translation is not an error** — the lookup chain is
//! `[active locale, en]`, so an untranslated message renders in English rather
//! than failing. Hard-failing on an untranslated string is the wrong production
//! behavior.
//!
//! **The active locale is a gpui [`Global`]** (per `STATE.md`), installed by
//! [`install`] and pointed at the user's real preference by [`wire_config`].
//! Without that global — a bare test `App`, a visual-snapshot context, the
//! driver before a `locale` command — every lookup answers from the English
//! source, so wording assertions are locale-independent by construction and no
//! test has to pin anything.

use std::sync::Arc;

use eidola_app_core::ConfigState;
use fluent_bundle::{FluentBundle, FluentResource};
use gpui::{App, Entity, Global, SharedString};
use unic_langid::LanguageIdentifier;

use crate::stores::ConfigStore;

include!(concat!(env!("OUT_DIR"), "/i18n_generated.rs"));

/// One argument of a localized message. Aliased so call sites and the generated
/// accessors never name the Fluent types directly.
pub type Arg<'a> = fluent_bundle::FluentValue<'a>;
/// The argument bag a generated accessor builds for its message.
pub type Args<'a> = fluent_bundle::FluentArgs<'a>;

/// The source locale — the one every accessor is generated from and every other
/// locale falls back to.
pub const SOURCE_LOCALE: &str = "en";
/// Simplified Chinese: the target for `zh-CN`, `zh-SG` and a bare `zh`.
pub const ZH_HANS: &str = "zh-Hans";
/// Traditional Chinese: the target for `zh-TW`, `zh-HK` and `zh-MO`.
pub const ZH_HANT: &str = "zh-Hant";

/// Every locale this binary ships, in tag order.
pub fn available_locales() -> impl Iterator<Item = &'static str> {
    LOCALES.iter().map(|(tag, _)| *tag)
}

/// The shipped locale matching `tag` exactly (case-insensitively), if any.
fn shipped(tag: &str) -> Option<&'static str> {
    LOCALES
        .iter()
        .find(|(t, _)| t.eq_ignore_ascii_case(tag))
        .map(|(t, _)| *t)
}

// ---------------------------------------------------------------------------
// The active locale
// ---------------------------------------------------------------------------

/// The active locale and its lookup chain. Held in a [`Global`] so any render
/// pass can format a message with only an `&App` in hand.
pub struct Localization {
    tag: &'static str,
    /// `[active locale, en]` — or just `[en]` when the active locale *is* the
    /// source. Ordered: the first bundle carrying the message wins, which is
    /// what makes a missing translation fall back rather than fail.
    chain: Vec<FluentBundle<Arc<FluentResource>>>,
}

impl Global for Localization {}

impl Localization {
    /// Build the chain for `tag`, which must be a shipped locale.
    fn for_locale(tag: &'static str) -> Self {
        let mut chain = Vec::new();
        if tag != SOURCE_LOCALE {
            chain.extend(bundle_for(tag));
        }
        chain.extend(bundle_for(SOURCE_LOCALE));
        Self { tag, chain }
    }

    /// The active locale's tag.
    pub fn tag(&self) -> &'static str {
        self.tag
    }

    fn format(&self, id: &str, args: Option<&Args>) -> SharedString {
        for bundle in &self.chain {
            let Some(message) = bundle.get_message(id) else {
                continue;
            };
            let Some(pattern) = message.value() else {
                continue;
            };
            let mut errors = Vec::new();
            let formatted = bundle.format_pattern(pattern, args, &mut errors);
            debug_assert!(
                errors.is_empty(),
                "localization: `{id}` in `{}` formatted with errors: {errors:?}",
                self.tag
            );
            return SharedString::from(formatted.into_owned());
        }
        // Unreachable through a generated accessor: every id it names exists in
        // the source locale, which is always the last link of the chain. Answer
        // with the id rather than panicking — a chrome string is never worth a
        // crash, and the id is a legible thing to see in a screenshot.
        debug_assert!(false, "localization: no bundle carries `{id}`");
        SharedString::from(id.to_string())
    }
}

fn bundle_for(tag: &'static str) -> Option<FluentBundle<Arc<FluentResource>>> {
    let (_, source) = LOCALES.iter().find(|(t, _)| *t == tag)?;
    // Both of these were validated at build time: the tag is a directory name
    // we ship, and the FTL parsed. A failure here means the generated file and
    // this module disagree, which is a bug rather than a user-facing state.
    let langid: LanguageIdentifier = tag.parse().ok()?;
    let resource = FluentResource::try_new((*source).to_string()).ok()?;
    let mut bundle = FluentBundle::new(vec![langid]);
    // No Unicode isolation marks around placeables: they are invisible
    // directional controls for bidirectional text, we ship no RTL locale, and
    // gpui would have to shape them.
    bundle.set_use_isolating(false);
    bundle.add_resource(Arc::new(resource)).ok()?;
    Some(bundle)
}

thread_local! {
    /// The answer when no [`Localization`] global is installed: the source
    /// locale alone. Built once per thread, lazily.
    static SOURCE_ONLY: Localization = Localization::for_locale(SOURCE_LOCALE);
}

/// Format a message in the active locale. The generated accessors are the only
/// intended callers — reaching for this directly gives up the compile-time id
/// and argument checking that is the point of the codegen.
pub fn format(cx: &App, id: &str, args: Option<&Args>) -> SharedString {
    match cx.try_global::<Localization>() {
        Some(l10n) => l10n.format(id, args),
        None => SOURCE_ONLY.with(|l10n| l10n.format(id, args)),
    }
}

/// The active locale's tag — the source locale when nothing was installed.
pub fn active_locale(cx: &App) -> &'static str {
    cx.try_global::<Localization>()
        .map(Localization::tag)
        .unwrap_or(SOURCE_LOCALE)
}

// ---------------------------------------------------------------------------
// Negotiation
// ---------------------------------------------------------------------------

/// What the persisted `language` config key asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguagePreference {
    /// Follow the operating system's preferred languages.
    System,
    /// Always this locale, whatever the system says.
    Fixed(&'static str),
}

/// Read the stored `language` config value. Unset, `auto`/`system`, and a tag
/// naming no shipped locale all mean "follow the system" — a preference we
/// cannot honor is not a reason to render nothing, and app-core stores the key
/// verbatim precisely so this decision lives here.
pub fn preference(stored: Option<&str>) -> LanguagePreference {
    let Some(stored) = stored.map(str::trim).filter(|s| !s.is_empty()) else {
        return LanguagePreference::System;
    };
    if stored.eq_ignore_ascii_case("auto") || stored.eq_ignore_ascii_case("system") {
        return LanguagePreference::System;
    }
    match match_locale(stored) {
        Some(tag) => LanguagePreference::Fixed(tag),
        None => LanguagePreference::System,
    }
}

/// Pick a shipped locale for one OS-reported language tag.
///
/// Chinese is mapped by script rather than by exact tag, because the platforms
/// report a region and we ship the two writing systems: an explicit `Hans`/`Hant`
/// subtag wins; otherwise `TW`/`HK`/`MO` are Traditional and everything else
/// (including a bare `zh`) is Simplified. Every other language matches on its
/// language subtag alone — a French speaker in Canada gets French.
fn match_locale(tag: &str) -> Option<&'static str> {
    let lower = tag.trim().to_ascii_lowercase();
    let parts: Vec<&str> = lower
        .split(['-', '_', '.'])
        .filter(|p| !p.is_empty())
        .collect();
    let language = *parts.first()?;

    if language == "zh" {
        let script = parts[1..]
            .iter()
            .find(|p| **p == "hans" || **p == "hant")
            .copied();
        let region = parts[1..]
            .iter()
            .find(|p| p.len() == 2 && p.chars().all(|c| c.is_ascii_alphabetic()))
            .copied();
        let wanted = match (script, region) {
            (Some("hant"), _) => ZH_HANT,
            (Some(_), _) => ZH_HANS,
            (None, Some("tw" | "hk" | "mo")) => ZH_HANT,
            (None, _) => ZH_HANS,
        };
        return shipped(wanted).or_else(|| shipped(ZH_HANS));
    }

    available_locales().find(|available| {
        available
            .split('-')
            .next()
            .is_some_and(|l| l.eq_ignore_ascii_case(language))
    })
}

/// Choose a locale from the operating system's preferred-language list, in the
/// order the OS reports it. Nothing matching falls back to the source locale.
pub fn negotiate(preferred: &[String]) -> &'static str {
    preferred
        .iter()
        .find_map(|tag| match_locale(tag))
        .unwrap_or(SOURCE_LOCALE)
}

/// The whole resolution, as one pure function: the stored override decides, and
/// only when it defers does the system list get a say.
pub fn resolve(stored: Option<&str>, system_preferred: &[String]) -> &'static str {
    match preference(stored) {
        LanguagePreference::Fixed(tag) => tag,
        LanguagePreference::System => negotiate(system_preferred),
    }
}

/// The OS-reported preferred languages, most-preferred first.
pub fn system_preferred() -> Vec<String> {
    sys_locale::get_locales().collect()
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

/// Install the source locale. Call once after `gpui_component::init`, beside
/// `theme::install`. Tests, the visual harness and the driver stop here — and
/// so does any context that never calls it at all, since a lookup with no
/// global installed answers from the source locale too.
pub fn install(cx: &mut App) {
    if !cx.has_global::<Localization>() {
        cx.set_global(Localization::for_locale(SOURCE_LOCALE));
    }
}

/// Point the app at the user's real locale: resolve the persisted `language`
/// preference against the OS's preferred languages, and re-resolve whenever the
/// config changes. Called once from `run()`; anything that skips it stays on the
/// source locale.
pub fn wire_config(config: &Entity<ConfigStore>, cx: &mut App) {
    let stored = |store: &ConfigStore| store.state().and_then(|s| s.language.clone());

    let initial = resolve(stored(config.read(cx)).as_deref(), &system_preferred());
    apply(initial, cx);

    // App-lifetime observation — nothing to cancel it against, so `.detach()`
    // is sanctioned here (the same rationale as `theme::wire_config`).
    cx.observe(config, move |config, cx| {
        let tag = resolve(stored(config.read(cx)).as_deref(), &system_preferred());
        apply(tag, cx);
    })
    .detach();
}

/// The language preference carried by a config snapshot, resolved against this
/// machine's OS languages — what a Settings picker will render as its current
/// value once one exists.
pub fn preference_of(state: &ConfigState) -> LanguagePreference {
    preference(state.language.as_deref())
}

/// Switch the active locale and repaint. The one path a locale ever changes
/// through; also the QA seam the driver's `locale` command uses, mirroring
/// `theme::apply_fixed`.
pub fn apply(tag: &str, cx: &mut App) {
    let Some(tag) = shipped(tag) else {
        return;
    };
    if active_locale(cx) == tag && cx.has_global::<Localization>() {
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
    fn the_source_locale_and_both_chinese_scripts_are_shipped() {
        assert!(shipped(SOURCE_LOCALE).is_some());
        assert!(shipped(ZH_HANS).is_some());
        assert!(shipped(ZH_HANT).is_some());
    }

    #[test]
    fn every_shipped_locale_builds_a_bundle() {
        // A tag that does not parse as a language identifier, or FTL the
        // runtime cannot load, would silently drop that locale from its own
        // chain and render English — the one failure the build-time checks
        // cannot see.
        for tag in available_locales() {
            assert!(
                bundle_for(tag).is_some(),
                "`{tag}` did not build a Fluent bundle"
            );
        }
    }
}
