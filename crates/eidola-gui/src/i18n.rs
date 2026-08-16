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
//! **A missing translation is not an error, and the fallback is structural.**
//! Each locale's bundle is built as the English resource *overridden by* that
//! locale's own, so an untranslated message is present as English rather than
//! absent — one bundle, no lookup chain, and no runtime error branch for the
//! promise to depend on. It has to be that way: Fluent resolves a message
//! reference against one bundle only, so a translated message reaching through
//! `{ another-message }` for something its locale never translated would render
//! the reference's error text and never consult a fallback bundle. Merging makes
//! the guarantee true by construction; see [`bundle_for`].
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

/// The active locale and the single bundle it formats through. Held in a
/// [`Global`] so any render pass can format a message with only an `&App` in
/// hand.
pub struct Localization {
    tag: &'static str,
    /// **One** bundle, built as the English resource *overridden by* the active
    /// locale's. Fallback is a property of the bundle's contents rather than of
    /// a lookup order — see [`bundle_for`].
    bundle: FluentBundle<Arc<FluentResource>>,
}

impl Global for Localization {}

impl Localization {
    /// Build the bundle for `tag`, which must be a shipped locale.
    fn for_locale(tag: &'static str) -> Self {
        Self {
            tag,
            bundle: bundle_for(tag),
        }
    }

    /// The active locale's tag.
    pub fn tag(&self) -> &'static str {
        self.tag
    }

    fn format(&self, id: &str, args: Option<&Args>) -> SharedString {
        if let Some(pattern) = self.bundle.get_message(id).and_then(|m| m.value()) {
            let mut errors = Vec::new();
            let formatted = self.bundle.format_pattern(pattern, args, &mut errors);
            // Not a case to recover from — a formatting error means the
            // build-time contract was violated, and `build_support/codegen.rs`
            // refuses every shape that can produce one (an unresolvable
            // reference, a cycle, a variable the accessor does not pass). The
            // assertion is how that claim stays true as the rules change.
            debug_assert!(
                errors.is_empty(),
                "localization: `{id}` in `{}` formatted with errors: {errors:?}",
                self.tag
            );
            return SharedString::from(formatted.into_owned());
        }
        // Unreachable through a generated accessor: every id it names exists in
        // the source locale, which every bundle contains. Answer with the id
        // rather than panicking — a chrome string is never worth a crash, and
        // the id is a legible thing to see in a screenshot.
        debug_assert!(
            false,
            "localization: the `{}` bundle has no `{id}`",
            self.tag
        );
        SharedString::from(id.to_string())
    }
}

/// Build one locale's bundle: **the English resource, then the locale's own
/// overriding it.**
///
/// Fallback has to live *inside* the bundle rather than in a chain of them,
/// because Fluent resolves a message reference against one bundle only: a
/// translated message that says `{ about-title }` reaches nothing if its own
/// bundle lacks `about-title`, and `format_pattern` then answers with the
/// reference's error text (`关于 {about-title}`) rather than deferring to a
/// fallback bundle the chain would have tried next. Merging makes the promised
/// fallback true by construction — an untranslated message *is* in this bundle,
/// as English — and leaves no runtime error branch for it to depend on.
///
/// The accepted cost: the bundle's locale is the active one, and fluent-bundle
/// takes plural rules from `locales[0]`, so an *untranslated* message carrying a
/// plural selector selects with the active locale's categories rather than
/// English's. That only reaches a string already showing in the wrong language,
/// and the cure for it is to translate the string.
fn bundle_for(tag: &'static str) -> FluentBundle<Arc<FluentResource>> {
    // The tag is a directory name we ship and every locale's FTL parsed at build
    // time, so neither of these can fail; a failure would mean the generated
    // file and this module disagree, which is a bug and not a user-facing state.
    let langid: LanguageIdentifier = tag.parse().unwrap_or_default();
    let mut bundle = FluentBundle::new(vec![langid]);
    // No Unicode isolation marks around placeables: they are invisible
    // directional controls for bidirectional text, we ship no RTL locale, and
    // gpui would have to shape them.
    bundle.set_use_isolating(false);
    if let Some(source) = resource_for(SOURCE_LOCALE) {
        bundle.add_resource(source).ok();
    }
    if tag != SOURCE_LOCALE
        && let Some(translation) = resource_for(tag)
    {
        // Whole-message override: this locale's `about-version-label` replaces
        // English's, while every message it did not translate stays.
        bundle.add_resource_overriding(translation);
    }
    bundle
}

fn resource_for(tag: &str) -> Option<Arc<FluentResource>> {
    let (_, source) = LOCALES.iter().find(|(t, _)| *t == tag)?;
    FluentResource::try_new((*source).to_string())
        .ok()
        .map(Arc::new)
}

thread_local! {
    /// The answer when no [`Localization`] global is installed: the source
    /// locale alone. Built once per thread, lazily.
    static SOURCE_ONLY: Localization = Localization::for_locale(SOURCE_LOCALE);
}

/// Format a message in a locale named explicitly, with **no `App`** — the entry
/// the generated `msg_in` accessors call.
///
/// It exists for the surfaces that run before gpui does (the startup-failure
/// alert, which is raised before `Application::run` and so has no `cx` to ask).
/// Resolving the tag is pure — [`resolve`] over [`system_preferred`] and the
/// stored preference — so the whole path is available that early.
///
/// A tag this build does not ship answers in the source locale, exactly as
/// [`apply`] refuses one. The bundle is built per call rather than cached: the
/// callers are a handful of strings on a path that is about to end the process,
/// and a cache keyed by tag would outlive the reason for it.
pub fn format_in(tag: &str, id: &str, args: Option<&Args>) -> SharedString {
    Localization::for_locale(shipped(tag).unwrap_or(SOURCE_LOCALE)).format(id, args)
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

    /// A locale nothing can negotiate to is a translation shipped inside the
    /// binary that no reader could ever see. The build refuses a directory whose
    /// name is not a canonical tag (`codegen::validate_locale_tag`); this is the
    /// same claim stated where it is actually observable — through negotiation.
    #[test]
    fn every_shipped_locale_is_reachable_by_negotiation() {
        for tag in available_locales() {
            assert_eq!(
                negotiate(&[tag.to_string()]),
                tag,
                "nothing negotiates to `{tag}`"
            );
            // Compared as *strings*, deliberately: `LanguageIdentifier`'s own
            // `PartialEq<&str>` re-parses the right-hand side, so it would call
            // `zh-hans` equal to `zh-Hans` and prove nothing about spelling.
            let canonical = tag
                .parse::<LanguageIdentifier>()
                .map(|id| id.to_string())
                .unwrap_or_default();
            assert_eq!(
                canonical.as_str(),
                tag,
                "`{tag}` is not a canonical language tag"
            );
        }
    }

    #[test]
    fn every_shipped_locale_carries_the_whole_source_locale() {
        // The merge is what makes fallback true, so each bundle must actually
        // hold every source message — a resource the runtime failed to load
        // would silently answer in English (or, for a reference, not at all),
        // which the build-time checks cannot see.
        for tag in available_locales() {
            let bundle = bundle_for(tag);
            for id in MESSAGE_IDS {
                assert!(
                    bundle.get_message(id).is_some(),
                    "the `{tag}` bundle is missing `{id}`"
                );
            }
        }
    }
}
