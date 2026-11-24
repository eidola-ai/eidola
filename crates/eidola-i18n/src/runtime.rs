//! The runtime half: the rules every app formats its embedded locales by, so
//! the desktop and mobile apps cannot disagree about fallback or negotiation.
//! Each app keeps its own glue (the active-locale global, where the preference
//! comes from, which windows to refresh); the table of shipped locales is the
//! `LOCALES` constant its generated module carries.

use std::sync::Arc;

use fluent_bundle::{FluentArgs, FluentBundle, FluentResource};
use unic_langid::LanguageIdentifier;

pub use crate::SOURCE_LOCALE;

/// Simplified Chinese: the target for `zh-CN`, `zh-SG` and a bare `zh`.
pub const ZH_HANS: &str = "zh-Hans";
/// Traditional Chinese: the target for `zh-TW`, `zh-HK` and `zh-MO`.
pub const ZH_HANT: &str = "zh-Hant";

/// A generated `LOCALES` table: every shipped locale as `(tag, FTL source)`.
pub type Locales = &'static [(&'static str, &'static str)];

/// The shipped locale matching `tag` exactly (case-insensitively), if any.
pub fn shipped(locales: Locales, tag: &str) -> Option<&'static str> {
    locales
        .iter()
        .find(|(t, _)| t.eq_ignore_ascii_case(tag))
        .map(|(t, _)| *t)
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
pub fn bundle_for(locales: Locales, tag: &str) -> FluentBundle<Arc<FluentResource>> {
    // The tag is a directory name we ship and every locale's FTL parsed at build
    // time, so neither of these can fail; a failure would mean the generated
    // file and this module disagree, which is a bug and not a user-facing state.
    let langid: LanguageIdentifier = tag.parse().unwrap_or_default();
    let mut bundle = FluentBundle::new(vec![langid]);
    // No Unicode isolation marks around placeables: they are invisible
    // directional controls for bidirectional text, we ship no RTL locale, and
    // gpui would have to shape them.
    bundle.set_use_isolating(false);
    if let Some(source) = resource_for(locales, SOURCE_LOCALE) {
        bundle.add_resource(source).ok();
    }
    if tag != SOURCE_LOCALE
        && let Some(translation) = resource_for(locales, tag)
    {
        // Whole-message override: this locale's `about-version-label` replaces
        // English's, while every message it did not translate stays.
        bundle.add_resource_overriding(translation);
    }
    bundle
}

fn resource_for(locales: Locales, tag: &str) -> Option<Arc<FluentResource>> {
    let (_, source) = locales.iter().find(|(t, _)| *t == tag)?;
    FluentResource::try_new((*source).to_string())
        .ok()
        .map(Arc::new)
}

/// Format `id` from `bundle` (whose locale is `tag`).
///
/// A formatting error is not a case to recover from — it means the build-time
/// contract was violated, and [`crate::codegen`] refuses every shape that can
/// produce one (an unresolvable reference, a cycle, a variable the accessor
/// does not pass). The assertion is how that claim stays true as the rules
/// change. A missing id is unreachable through a generated accessor (every id
/// it names exists in the source locale, which every bundle contains); it
/// answers with the id rather than panicking — a chrome string is never worth
/// a crash, and the id is a legible thing to see in a screenshot.
pub fn format(
    bundle: &FluentBundle<Arc<FluentResource>>,
    tag: &str,
    id: &str,
    args: Option<&FluentArgs>,
) -> String {
    if let Some(pattern) = bundle.get_message(id).and_then(|m| m.value()) {
        let mut errors = Vec::new();
        let formatted = bundle.format_pattern(pattern, args, &mut errors);
        debug_assert!(
            errors.is_empty(),
            "localization: `{id}` in `{tag}` formatted with errors: {errors:?}"
        );
        return formatted.into_owned();
    }
    debug_assert!(false, "localization: the `{tag}` bundle has no `{id}`");
    id.to_string()
}

/// Pick a shipped locale for one OS-reported language tag.
///
/// Chinese is mapped by script rather than by exact tag, because the platforms
/// report a region and we ship the two writing systems: an explicit `Hans`/`Hant`
/// subtag wins; otherwise `TW`/`HK`/`MO` are Traditional and everything else
/// (including a bare `zh`) is Simplified. Every other language matches on its
/// language subtag alone — a French speaker in Canada gets French.
pub fn match_locale(locales: Locales, tag: &str) -> Option<&'static str> {
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
        return shipped(locales, wanted).or_else(|| shipped(locales, ZH_HANS));
    }

    locales.iter().map(|(t, _)| *t).find(|available| {
        available
            .split('-')
            .next()
            .is_some_and(|l| l.eq_ignore_ascii_case(language))
    })
}

/// Choose a locale from a preferred-language list, in the order given. Nothing
/// matching falls back to the source locale.
pub fn negotiate(locales: Locales, preferred: &[String]) -> &'static str {
    preferred
        .iter()
        .find_map(|tag| match_locale(locales, tag))
        .unwrap_or(SOURCE_LOCALE)
}
