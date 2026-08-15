//! Localization: locale negotiation, the fallback chain, and the build-time
//! contract.
//!
//! The build-error rules are asserted against the codegen module directly
//! (`build_support/codegen.rs`, the same file `build.rs` includes) rather than
//! by committing malformed FTL — a locales tree that fails to build would fail
//! every other test in the crate too.

use eidola_gui::i18n::{self, LanguagePreference, msg};
use gpui::TestAppContext;

// Only the rule-checking half is exercised here; `build.rs` is the other
// caller and uses the rest.
#[allow(dead_code)]
#[path = "../build_support/codegen.rs"]
mod codegen;

fn tags(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Negotiation
// ---------------------------------------------------------------------------

#[test]
fn the_first_system_language_we_ship_wins() {
    assert_eq!(i18n::negotiate(&tags(&["fr-CA", "en-US"])), "fr");
    // A language we do not ship is skipped, not treated as a failure — the
    // reader's *second* choice is still a preference they expressed.
    assert_eq!(i18n::negotiate(&tags(&["de-DE", "es-MX", "en"])), "es");
    assert_eq!(i18n::negotiate(&tags(&["en-GB"])), "en");
}

#[test]
fn an_unsupported_language_falls_back_to_the_source_locale() {
    assert_eq!(i18n::negotiate(&tags(&["de-DE", "ja-JP"])), "en");
    assert_eq!(i18n::negotiate(&[]), "en");
    assert_eq!(i18n::negotiate(&tags(&["", "   "])), "en");
}

/// The platforms report Chinese by region, and we ship the two writing systems,
/// so the mapping is by script: an explicit subtag wins, then the Traditional
/// regions, and everything else — including a bare `zh` — is Simplified.
#[test]
fn chinese_is_negotiated_by_script_not_by_exact_tag() {
    for tag in ["zh-TW", "zh-HK", "zh-MO", "zh-Hant", "zh-Hant-TW", "zh_TW"] {
        assert_eq!(
            i18n::negotiate(&tags(&[tag])),
            "zh-Hant",
            "`{tag}` should read as Traditional"
        );
    }
    for tag in [
        "zh",
        "zh-CN",
        "zh-SG",
        "zh-Hans",
        "zh-Hans-HK",
        "zh_CN.UTF-8",
    ] {
        assert_eq!(
            i18n::negotiate(&tags(&[tag])),
            "zh-Hans",
            "`{tag}` should read as Simplified"
        );
    }
}

/// An explicit script always beats the region it is paired with — `zh-Hans-HK`
/// is a Hong Kong reader who writes Simplified, and the reverse for `zh-Hant-CN`.
#[test]
fn an_explicit_chinese_script_outranks_the_region() {
    assert_eq!(i18n::negotiate(&tags(&["zh-Hant-CN"])), "zh-Hant");
    assert_eq!(i18n::negotiate(&tags(&["zh-Hans-TW"])), "zh-Hans");
}

// ---------------------------------------------------------------------------
// The stored override
// ---------------------------------------------------------------------------

#[test]
fn the_stored_preference_decides_and_only_defers_when_it_says_so() {
    assert_eq!(i18n::preference(None), LanguagePreference::System);
    assert_eq!(i18n::preference(Some("auto")), LanguagePreference::System);
    assert_eq!(i18n::preference(Some("  ")), LanguagePreference::System);
    assert_eq!(
        i18n::preference(Some("zh-Hant")),
        LanguagePreference::Fixed("zh-Hant")
    );
    // Hand-edited config: a regional tag resolves through the same rule the
    // system list does, so `language = "zh-TW"` means something sensible.
    assert_eq!(
        i18n::preference(Some("zh-TW")),
        LanguagePreference::Fixed("zh-Hant")
    );
    // A preference we cannot honor is not a reason to render nothing.
    assert_eq!(i18n::preference(Some("qya")), LanguagePreference::System);

    // The whole resolution: the override wins outright, and the system list is
    // consulted only when it defers.
    assert_eq!(i18n::resolve(Some("es"), &tags(&["fr", "en"])), "es");
    assert_eq!(i18n::resolve(None, &tags(&["fr", "en"])), "fr");
    assert_eq!(i18n::resolve(Some("auto"), &tags(&["zh-TW"])), "zh-Hant");
    assert_eq!(i18n::resolve(Some("qya"), &tags(&["de"])), "en");
}

// ---------------------------------------------------------------------------
// The fallback chain
// ---------------------------------------------------------------------------

/// A translated message renders in the active locale; a message that locale
/// never translated falls back to the English source rather than failing.
/// `about-title` is the untranslated one by design — the wordmark is the same
/// in every language.
#[gpui::test]
fn a_missing_translation_falls_back_to_the_source_locale(cx: &mut TestAppContext) {
    cx.update(|cx| {
        i18n::install(cx);
        assert_eq!(i18n::active_locale(cx), "en");

        i18n::apply("zh-Hans", cx);
        assert_eq!(i18n::active_locale(cx), "zh-Hans");
        assert_eq!(msg::about_version_label(cx).as_ref(), "版本");
        // The window's OS-level name localizes with the rest of the surface,
        // even though it paints nothing (it names the window for the Window
        // menu, the window switcher, and VoiceOver).
        assert_eq!(msg::about_window_title(cx).as_ref(), "关于 Eidola");
        // Untranslated: the source locale answers.
        assert_eq!(msg::about_title(cx).as_ref(), "Eidola");

        i18n::apply("fr", cx);
        assert_eq!(msg::about_github(cx).as_ref(), "Voir sur GitHub");
        assert_eq!(msg::about_title(cx).as_ref(), "Eidola");

        i18n::apply("en", cx);
        assert_eq!(msg::about_version_label(cx).as_ref(), "Version");
    });
}

/// An `App` that never installed the global is on the source locale — which is
/// what keeps every wording assertion in the crate locale-independent without a
/// single test having to pin anything.
#[gpui::test]
fn without_the_global_every_lookup_answers_in_english(cx: &mut TestAppContext) {
    cx.update(|cx| {
        assert_eq!(i18n::active_locale(cx), "en");
        assert_eq!(msg::about_version_label(cx).as_ref(), "Version");
        assert_eq!(
            msg::about_version_value(cx, "9.9.9").as_ref(),
            "v9.9.9",
            "an argument reaches the pattern through the fallback path too"
        );
    });
}

/// A locale change has to reach the strings that live *outside* a view's
/// render — the OS window titles, set once at open. `run()` hangs that re-set
/// off `observe_global::<Localization>`, so `apply` must notify, and must not
/// notify when the active locale did not actually move (which would re-title
/// windows for nothing).
///
/// This observes the trigger rather than the effect: gpui's test platform does
/// not record a window title (`PlatformWindow::get_title` keeps its empty
/// default there) and `Window::a11y` is crate-private, so the title itself is
/// not readable from a test.
#[gpui::test]
fn a_locale_change_notifies_global_observers(cx: &mut TestAppContext) {
    use std::cell::Cell;
    use std::rc::Rc;

    let changes = Rc::new(Cell::new(0usize));
    cx.update(|cx| {
        i18n::install(cx);
        let changes = changes.clone();
        cx.observe_global::<i18n::Localization>(move |_| changes.set(changes.get() + 1))
            .detach();
    });
    cx.run_until_parked();
    let baseline = changes.get();

    cx.update(|cx| i18n::apply("fr", cx));
    cx.run_until_parked();
    assert_eq!(
        changes.get(),
        baseline + 1,
        "a real locale change must reach the window titles"
    );

    // Re-applying the locale already active changes nothing, so nothing is
    // re-titled for it.
    cx.update(|cx| i18n::apply("fr", cx));
    cx.run_until_parked();
    assert_eq!(changes.get(), baseline + 1, "a no-op apply must not notify");
}

#[gpui::test]
fn a_locale_we_do_not_ship_leaves_the_active_one_alone(cx: &mut TestAppContext) {
    cx.update(|cx| {
        i18n::install(cx);
        i18n::apply("es", cx);
        i18n::apply("de", cx);
        assert_eq!(i18n::active_locale(cx), "es");
    });
}

/// Every message the codegen generated an accessor for formats in every shipped
/// locale — the runtime half of the build-time contract, and what catches a
/// translation whose pattern is well-formed but unresolvable.
#[gpui::test]
fn every_message_formats_in_every_shipped_locale(cx: &mut TestAppContext) {
    cx.update(|cx| {
        i18n::install(cx);
        for tag in i18n::available_locales() {
            i18n::apply(tag, cx);
            for id in i18n::MESSAGE_IDS {
                let mut args = i18n::Args::new();
                // The pilot's only variable. Passing an argument a message does
                // not read is harmless in Fluent; a *missing* one is what shows
                // up as an unresolved placeable in the output.
                args.set("version", "0.0.0");
                let formatted = i18n::format(cx, id, Some(&args));
                assert!(
                    !formatted.is_empty() && !formatted.contains('{'),
                    "`{id}` did not resolve in `{tag}`: {formatted:?}"
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// The build-time contract
// ---------------------------------------------------------------------------

fn parse(tag: &str, source: &str) -> codegen::LocaleDef {
    codegen::parse_locale(tag, source).expect("fixture should parse")
}

#[test]
fn malformed_ftl_is_a_build_error() {
    let err = codegen::parse_locale("es", "about-title Eidola\n").expect_err("should refuse");
    assert!(err.contains("es"), "the error names the locale: {err}");
}

/// The whole point of the codegen: the accessor's parameters are the source
/// message's variables, in first-appearance order.
#[test]
fn accessor_parameters_come_from_the_source_messages_variables() {
    let en = parse(
        "en",
        "greeting = Hello { $name }, you have { $count } messages\nplain = Hello\n",
    );
    assert_eq!(en.message("greeting").unwrap().vars, ["name", "count"]);
    assert!(en.message("plain").unwrap().vars.is_empty());
}

/// A selector's variable counts — a plural form still needs its count passed.
#[test]
fn a_selectors_variable_is_a_parameter_too() {
    let en = parse(
        "en",
        "posts = { $count ->\n    [one] one post\n   *[other] { $count } posts\n  }\n",
    );
    assert_eq!(en.message("posts").unwrap().vars, ["count"]);
}

/// A message that references another needs whatever the referenced message
/// needs, or the reference would resolve to an unfilled placeable.
#[test]
fn a_message_reference_carries_the_referenced_messages_variables() {
    let en = parse("en", "inner = v{ $version }\nouter = { inner } —\n");
    assert_eq!(en.message("outer").unwrap().vars, ["version"]);
}

#[test]
fn a_translation_reading_a_variable_the_source_lacks_is_a_build_error() {
    let en = parse("en", "hello = Hello\n");
    let es = parse("es", "hello = Hola, { $name }\n");
    let err = codegen::check_translation(&en, &es).expect_err("should refuse");
    assert!(err.contains("$name"), "{err}");
    assert!(err.contains("hello"), "{err}");
}

#[test]
fn a_translation_inventing_a_message_id_is_a_build_error() {
    let en = parse("en", "hello = Hello\n");
    let fr = parse("fr", "hello = Bonjour\nhelo = Bonjour\n");
    let err = codegen::check_translation(&en, &fr).expect_err("should refuse");
    assert!(err.contains("helo"), "{err}");
}

/// A *missing* translation is not an error — hard-failing on an untranslated
/// string is the wrong production behavior, and the runtime falls back.
#[test]
fn a_missing_translation_is_not_a_build_error() {
    let en = parse("en", "hello = Hello\ngoodbye = Goodbye\n");
    let fr = parse("fr", "hello = Bonjour\n");
    codegen::check_translation(&en, &fr).expect("a partial translation is allowed");
}

#[test]
fn attributes_and_duplicates_are_refused_rather_than_quietly_ignored() {
    let err = codegen::parse_locale("en", "hello = Hello\n    .tooltip = Greeting\n")
        .expect_err("should refuse");
    assert!(err.contains("attributes"), "{err}");

    let err =
        codegen::parse_locale("en", "hello = Hello\nhello = Hi\n").expect_err("should refuse");
    assert!(err.contains("duplicate"), "{err}");
}

/// A term has its own scope, so a variable in a term body can never be filled.
#[test]
fn a_term_reading_a_message_variable_is_refused() {
    let err = codegen::parse_locale("en", "-brand = { $name }\n").expect_err("should refuse");
    assert!(err.contains("term"), "{err}");
}

#[test]
fn ids_become_identifiers_and_keywords_are_escaped() {
    assert_eq!(
        codegen::rust_ident("about-version-value").unwrap(),
        "about_version_value"
    );
    assert_eq!(codegen::rust_ident("type").unwrap(), "type_");
    assert!(codegen::rust_ident("2fast").is_err());
}

/// The emitted accessor takes one parameter per variable and none otherwise —
/// which is what makes a wrong argument set a compile error at the call site.
#[test]
fn the_generated_accessor_signature_is_the_messages_variables() {
    let en = parse(
        "en",
        "plain = Hello\ngreeting = Hi { $name } ({ $count })\n",
    );
    let generated = codegen::generate(&en, std::slice::from_ref(&en)).expect("should generate");
    assert!(
        generated.contains("pub fn plain(cx: &gpui::App) -> gpui::SharedString"),
        "{generated}"
    );
    assert!(
        generated.contains(
            "pub fn greeting<'a>(cx: &gpui::App, name: impl Into<super::Arg<'a>>, \
             count: impl Into<super::Arg<'a>>)"
        ),
        "{generated}"
    );
    assert!(
        generated.contains("args.set(\"name\", name);"),
        "{generated}"
    );
}

/// The shipped tree must satisfy the contract the codegen enforces — the same
/// check `build.rs` runs, asserted here so a failure reads as a test rather
/// than as a mysterious compile error.
#[test]
fn the_shipped_locales_satisfy_the_contract() {
    let parsed: Vec<codegen::LocaleDef> = i18n::LOCALES
        .iter()
        .map(|(tag, source)| parse(tag, source))
        .collect();
    let en = parsed
        .iter()
        .find(|l| l.tag == "en")
        .expect("the source locale ships");
    for locale in &parsed {
        if locale.tag == "en" {
            continue;
        }
        codegen::check_translation(en, locale)
            .unwrap_or_else(|e| panic!("shipped locale failed the contract: {e}"));
    }
    assert_eq!(
        i18n::MESSAGE_IDS.len(),
        en.messages.len(),
        "every source message gets an accessor"
    );
}
