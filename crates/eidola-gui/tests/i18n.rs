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

/// A translated message may reference an **untranslated** one, and the
/// reference has to resolve across that boundary.
///
/// `about-window-title` is the live case: every locale translates it, and every
/// locale reaches through `{ about-title }` for the wordmark, which no locale
/// translates. Fluent resolves a message reference against one bundle only, so
/// the whole fallback story has to hold *inside* the bundle the message is
/// formatted in — a per-message chain would answer with the reference's error
/// text (`关于 {about-title}`) and never reach the source locale.
#[gpui::test]
fn a_translated_message_can_reference_an_untranslated_one(cx: &mut TestAppContext) {
    cx.update(|cx| {
        i18n::install(cx);
        assert_eq!(msg::about_window_title(cx).as_ref(), "About Eidola");

        i18n::apply("zh-Hans", cx);
        assert_eq!(msg::about_window_title(cx).as_ref(), "关于 Eidola");

        i18n::apply("zh-Hant", cx);
        assert_eq!(msg::about_window_title(cx).as_ref(), "關於 Eidola");

        i18n::apply("es", cx);
        assert_eq!(msg::about_window_title(cx).as_ref(), "Acerca de Eidola");

        i18n::apply("fr", cx);
        assert_eq!(msg::about_window_title(cx).as_ref(), "À propos d'Eidola");
    });
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
                // **Every variable any message reads, in one bag.** Passing an
                // argument a message does not read is harmless in Fluent; a
                // *missing* one is what shows up as an unresolved placeable in
                // the output — so a new variable joins this list or the message
                // fails here rather than on a reader's screen.
                args.set("version", "0.0.0");
                args.set("model", "Gemma 4 27B");
                args.set("depth", 4);
                args.set("limit", 4);
                args.set("reason", "upstream");
                args.set("byline", "Sofia");
                args.set("snippet", "the opening of what it says");
                args.set("space", "Tides and the moon");
                args.set("label", "Sofia, in another space");
                args.set("n", 2);
                let formatted = i18n::format(cx, id, Some(&args));
                assert!(
                    !formatted.is_empty() && !formatted.contains('{'),
                    "`{id}` did not resolve in `{tag}`: {formatted:?}"
                );
                // The pre-gpui twin resolves the same message in the same
                // locale — the surfaces that have no `cx` are not a second,
                // weaker path (`msg_in`, `i18n::format_in`).
                let explicit = i18n::format_in(tag, id, Some(&args));
                assert_eq!(
                    explicit, formatted,
                    "`{id}` differs between the `App` and explicit-locale paths in `{tag}`"
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
    assert_eq!(
        codegen::resolve_vars("outer", &en, None).unwrap(),
        ["version"]
    );
}

/// The resolution is done through the same merged view the runtime formats
/// through, so a translation's reference into an *untranslated* message carries
/// that message's variables — English's, not this locale's.
#[test]
fn a_translations_reference_resolves_through_the_source_locale() {
    let en = parse("en", "inner = v{ $version }\nouter = { inner } —\n");
    let es = parse("es", "outer = versión { inner }\n");
    assert_eq!(
        codegen::resolve_vars("outer", &es, Some(&en)).unwrap(),
        ["version"],
        "the reference reaches English's `inner`, which reads $version"
    );
    // And the pair is consistent: English passes $version, so this is allowed.
    codegen::check_translation(&en, &es).expect("a reference into the source locale is fine");
}

/// The merge is what makes that reference resolve, and it is also what lets a
/// translation reach a variable the *call site* never passes — so the check
/// follows the reference rather than looking only inside the translation.
#[test]
fn a_translation_reaching_a_variable_through_the_source_locale_is_a_build_error() {
    let en = parse("en", "inner = v{ $version }\nouter = plain\n");
    let es = parse("es", "outer = { inner }\n");
    let err = codegen::check_translation(&en, &es).expect_err("should refuse");
    assert!(err.contains("$version"), "{err}");
    assert!(err.contains("outer"), "{err}");
}

/// A reference that reaches nothing in either locale would render its own error
/// text at runtime, so it never reaches runtime.
#[test]
fn a_reference_to_a_message_that_exists_nowhere_is_a_build_error() {
    let en = parse("en", "outer = { nope }\n");
    let err = codegen::resolve_vars("outer", &en, None).expect_err("should refuse");
    assert!(err.contains("nope"), "{err}");

    let en = parse("en", "outer = plain\n");
    let fr = parse("fr", "outer = { nope }\n");
    let err = codegen::check_translation(&en, &fr).expect_err("should refuse");
    assert!(err.contains("nope"), "{err}");
}

/// Fluent gives up on a reference cycle at runtime; the build gives up first.
#[test]
fn a_reference_cycle_is_a_build_error() {
    let en = parse("en", "a = { b }\nb = { a }\n");
    let err = codegen::resolve_vars("a", &en, None).expect_err("should refuse");
    assert!(err.contains("cycle"), "{err}");
}

/// No message may have attributes, so a reference to one could only fail.
#[test]
fn an_attribute_reference_is_a_build_error() {
    let err =
        codegen::parse_locale("en", "outer = { inner.tooltip }\n").expect_err("should refuse");
    assert!(err.contains("attribute"), "{err}");
}

/// Terms stay closed: a term that reached a message could drag a variable
/// across the merge with nothing to fill it.
#[test]
fn a_term_referencing_a_message_is_refused() {
    let err =
        codegen::parse_locale("en", "-brand = { other }\nother = Eidola\n").expect_err("refuse");
    assert!(err.contains("term"), "{err}");
}

/// A term a locale never defines is unresolvable in that locale's bundle.
#[test]
fn a_reference_to_an_undefined_term_is_a_build_error() {
    let en = parse("en", "hello = { -brand }\n");
    let err = codegen::resolve_vars("hello", &en, None).expect_err("should refuse");
    assert!(err.contains("-brand"), "{err}");

    // Defined in the source locale, it resolves for every locale that merges it.
    let en = parse("en", "-brand = Eidola\nhello = { -brand }\n");
    let es = parse("es", "hello = Hola, { -brand }\n");
    codegen::resolve_vars("hello", &es, Some(&en)).expect("the merge supplies the term");
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

    // The pre-gpui twin: same ids, same parameters, only the locale is named
    // differently — so a surface with no `cx` gives up no checking.
    assert!(
        generated.contains("pub fn plain(locale: &str) -> gpui::SharedString"),
        "{generated}"
    );
    assert!(
        generated.contains(
            "pub fn greeting<'a>(locale: &str, name: impl Into<super::Arg<'a>>, \
             count: impl Into<super::Arg<'a>>)"
        ),
        "{generated}"
    );
    assert_eq!(
        generated.matches("pub mod msg").count(),
        2,
        "both accessor modules are emitted from the one pass: {generated}"
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

// ---------------------------------------------------------------------------
// Terms, functions, and locale tags
// ---------------------------------------------------------------------------

/// A term body is validated like any other pattern: a term reaching a term that
/// exists nowhere is unresolvable at runtime, however many hops away it is.
#[test]
fn a_term_reaching_an_undefined_term_is_a_build_error() {
    let en = parse("en", "-brand = { -missing }\nhello = { -brand }\n");
    let err = codegen::resolve_vars("hello", &en, None).expect_err("should refuse");
    assert!(err.contains("-missing"), "{err}");
}

/// Terms cycle the same way messages do, and Fluent gives up on it the same way.
#[test]
fn a_term_reference_cycle_is_a_build_error() {
    let en = parse("en", "-a = { -b }\n-b = { -a }\nhello = { -a }\n");
    let err = codegen::resolve_vars("hello", &en, None).expect_err("should refuse");
    assert!(err.contains("cycle"), "{err}");
}

/// A term defined only in the source locale still resolves for a translation
/// that reaches it, because the bundle is merged.
#[test]
fn a_term_chain_resolves_through_the_merged_view() {
    let en = parse(
        "en",
        "-inner = Eidola\n-brand = { -inner }\nhello = { -brand }\n",
    );
    let es = parse("es", "hello = Hola de { -brand }\n");
    codegen::resolve_vars("hello", &es, Some(&en)).expect("the merge supplies the term chain");
}

/// No function is callable: the bundle registers none, and fluent-bundle
/// registers none of its own.
#[test]
fn a_function_reference_is_a_build_error() {
    let err = codegen::parse_locale("en", "count = { NUMBER($n) }\n").expect_err("should refuse");
    assert!(err.contains("NUMBER"), "{err}");

    // Including a misspelling that would otherwise look plausible.
    let err = codegen::parse_locale("en", "count = { NUMBR($n) }\n").expect_err("should refuse");
    assert!(err.contains("NUMBR"), "{err}");
}
/// A directory name becomes the locale's tag verbatim, and negotiation matches
/// on canonical tags — so a non-canonical directory would ship a translation
/// nothing could ever select.
#[test]
fn a_locale_directory_must_be_a_canonical_language_tag() {
    for good in ["en", "es", "fr", "zh-Hans", "zh-Hant", "pt-BR"] {
        codegen::validate_locale_tag(good).unwrap_or_else(|e| panic!("`{good}` refused: {e}"));
    }
    for bad in ["zh_Hans", "zh-hans", "ZH-HANS", "en_US", "not a tag", ""] {
        let err =
            codegen::validate_locale_tag(bad).expect_err(&format!("`{bad}` should be refused"));
        assert!(err.contains(bad) || bad.is_empty(), "{err}");
    }
}

/// A term attribute is text nothing can reference — it would ship inside the
/// binary and never reach a screen, exactly like a message attribute.
#[test]
fn a_term_attribute_is_a_build_error() {
    let err = codegen::parse_locale("es", "-brand = Eidola\n    .short = E\n")
        .expect_err("should refuse");
    assert!(err.contains("attributes"), "{err}");
    assert!(err.contains("-brand"), "{err}");
    assert!(err.contains("es"), "the error names the locale: {err}");
}

/// Validation covers everything a locale ships, not only what something else
/// happens to reach: a term nothing references today is referenced tomorrow,
/// and the reference site would not be the broken one.
#[test]
fn an_unreferenced_term_is_validated_too() {
    // Reaches a term that exists nowhere; no message mentions `-brand`.
    let en = parse("en", "hello = Hello\n-brand = { -missing }\n");
    let err = codegen::check_locale(&en, None).expect_err("should refuse");
    assert!(err.contains("-missing"), "{err}");

    // A cycle no message enters.
    let en = parse("en", "hello = Hello\n-a = { -b }\n-b = { -a }\n");
    let err = codegen::check_locale(&en, None).expect_err("should refuse");
    assert!(err.contains("cycle"), "{err}");

    // Same rule inside a translation, resolved through the merged view.
    let en = parse("en", "hello = Hello\n");
    let es = parse("es", "hello = Hola\n-marca = { -inexistente }\n");
    let err = codegen::check_locale(&es, Some(&en)).expect_err("should refuse");
    assert!(err.contains("-inexistente"), "{err}");

    // And a translation's term reaching one the *source* defines is fine.
    let en = parse("en", "-brand = Eidola\nhello = Hello\n");
    let es = parse("es", "hello = Hola\n-marca = { -brand }\n");
    codegen::check_locale(&es, Some(&en)).expect("the merge supplies the term");
}

/// The shipped tree satisfies the whole-locale invariant, not just the
/// cross-locale one.
#[test]
fn every_shipped_locale_validates_on_its_own_terms() {
    let parsed: Vec<codegen::LocaleDef> = i18n::LOCALES
        .iter()
        .map(|(tag, source)| parse(tag, source))
        .collect();
    let en = parsed.iter().find(|l| l.tag == "en").expect("source ships");
    codegen::check_locale(en, None).expect("the source locale validates");
    for locale in &parsed {
        if locale.tag == "en" {
            continue;
        }
        codegen::check_locale(locale, Some(en))
            .unwrap_or_else(|e| panic!("shipped locale failed the contract: {e}"));
    }
}

// ---------------------------------------------------------------------------
// The resolver's placeable limit
// ---------------------------------------------------------------------------

/// Build `n` literal placeables — the cheapest way to hit a known count.
fn placeables(n: usize) -> String {
    "{\"x\"}".repeat(n)
}

/// The limit is fluent-bundle's, so the boundary has to be its boundary: it
/// trips on `> MAX_PLACEABLES`, meaning exactly 100 resolve and 101 do not.
#[test]
fn the_placeable_limit_boundary_is_the_resolvers() {
    let en = parse("en", &format!("m = {}\n", placeables(100)));
    codegen::check_locale(&en, None).expect("100 placeables is legal");

    let en = parse("en", &format!("m = {}\n", placeables(101)));
    let err = codegen::check_locale(&en, None).expect_err("101 is over the limit");
    assert!(err.contains("101"), "the count is named: {err}");
    assert!(err.contains("100"), "the limit is named: {err}");
    assert!(err.contains("`m`"), "the message is named: {err}");
}

/// The count follows reference expansion, which is the whole point — a chain
/// that doubles is the Billion Laughs shape the limit exists to stop, and every
/// other rule is satisfied by it.
#[test]
fn a_doubling_reference_chain_is_refused() {
    let mut ftl = String::from("c0 = {\"x\"}\n");
    for i in 1..8 {
        ftl.push_str(&format!("c{i} = {{ c{} }}{{ c{} }}\n", i - 1, i - 1));
    }
    let en = parse("en", &ftl);
    let err = codegen::check_locale(&en, None).expect_err("the chain blows the limit");
    // c5 costs 94 and is legal; c6 is the first over, at 190.
    assert!(err.contains("`c6`"), "names the first message over: {err}");
    assert!(err.contains("190"), "{err}");
}

/// Deep chains must be evaluated, not walked exponentially, and must not
/// overflow — 60 levels of doubling is far past any integer width.
#[test]
fn a_deep_chain_terminates_instead_of_exploding() {
    let mut ftl = String::from("c0 = {\"x\"}\n");
    for i in 1..60 {
        ftl.push_str(&format!("c{i} = {{ c{} }}{{ c{} }}\n", i - 1, i - 1));
    }
    let en = parse("en", &ftl);
    assert!(codegen::check_locale(&en, None).is_err());
}

/// A term reference costs what the term's own pattern costs, plus the placeable
/// that reaches it.
#[test]
fn a_terms_placeables_count_where_it_is_referenced() {
    let en = parse(
        "en",
        &format!("-heavy = {}\nm = {{ -heavy }}\n", placeables(99)),
    );
    codegen::check_locale(&en, None).expect("1 + 99 == 100 is legal");

    let en = parse(
        "en",
        &format!("-heavy = {}\nm = {{ -heavy }}\n", placeables(100)),
    );
    let err = codegen::check_locale(&en, None).expect_err("1 + 100 == 101 is over");
    assert!(err.contains("101"), "{err}");
}

/// Exactly one variant of a select is ever written, so the bound is the worst
/// variant — not the sum of them, which would refuse legal FTL.
#[test]
fn a_select_costs_its_worst_variant_not_all_of_them() {
    // Two 60-placeable variants: 1 + max(60, 60) == 61 legal; summing would be 121.
    let ftl = format!(
        "m = {{ $n ->\n    [one] {}\n   *[other] {}\n  }}\n",
        placeables(60),
        placeables(60)
    );
    let en = parse("en", &ftl);
    codegen::check_locale(&en, None).expect("the worst variant is 60, well under the limit");

    // But a single variant over the line still fails, because it is selectable.
    let ftl = format!(
        "m = {{ $n ->\n    [one] {}\n   *[other] {}\n  }}\n",
        placeables(100),
        placeables(1)
    );
    let en = parse("en", &ftl);
    let err =
        codegen::check_locale(&en, None).expect_err("that variant breaks for whoever picks it");
    assert!(err.contains("101"), "{err}");
}

/// A placeable nested inside an expression is written through rather than
/// counted as a pattern element — counting it would refuse legal FTL.
#[test]
fn a_nested_inline_placeable_adds_no_count_of_its_own() {
    let en = parse(
        "en",
        &format!("-t = {}\nm = {{ {{ -t }} }}\n", placeables(99)),
    );
    codegen::check_locale(&en, None).expect("1 + 99 == 100, the nesting adds nothing");
}

/// Cost is resolved through the merged view, like every other reference rule:
/// a translation reaching an untranslated message pays *English's* cost.
#[test]
fn placeable_cost_resolves_through_the_merged_view() {
    let en = parse("en", &format!("heavy = {}\nm = plain\n", placeables(100)));
    let es = parse("es", "m = { heavy }\n");
    let err = codegen::check_locale(&es, Some(&en)).expect_err("1 + 100 == 101 through the merge");
    assert!(err.contains("101"), "{err}");
    assert!(err.contains("es"), "the error names the locale: {err}");
}

/// The shipped tree is nowhere near the limit, and stays checked.
#[test]
fn every_shipped_locale_is_within_the_placeable_limit() {
    let parsed: Vec<codegen::LocaleDef> = i18n::LOCALES
        .iter()
        .map(|(tag, source)| parse(tag, source))
        .collect();
    let en = parsed.iter().find(|l| l.tag == "en").expect("source ships");
    for locale in &parsed {
        let base = (locale.tag != "en").then_some(en);
        codegen::check_locale(locale, base).unwrap_or_else(|e| panic!("{e}"));
    }
}

// ---------------------------------------------------------------------------
// Composition: what a locale's overrides do to the source's own messages
// ---------------------------------------------------------------------------

/// The reported hole, and the reason rule 8 is scoped to the *view* rather than
/// the locale: a source message the locale never translates still formats in
/// that locale, through that locale's overrides. Each side is legal alone.
#[test]
fn a_source_message_is_costed_under_each_locales_overrides() {
    let en = parse("en", "-heavy = {\"x\"}\nm = { -heavy }\n");
    let es = parse("es", &format!("-heavy = {}\n", placeables(100)));

    // Both sides are unimpeachable on their own.
    codegen::check_locale(&en, None).expect("the source costs 2");
    codegen::check_translation(&en, &es).expect("the override translates nothing wrongly");

    // Composed, the source's own message resolves to 101 here.
    let err = codegen::check_locale(&es, Some(&en)).expect_err("101 under this locale");
    assert!(err.contains("101"), "{err}");
    assert!(err.contains("`m`"), "names the source message: {err}");
    assert!(err.contains("es"), "names the locale it breaks in: {err}");
}

/// Reachability is closed under composition — an override replaces a definition
/// but never removes an id, so a source reference still resolves. Asserted
/// rather than assumed, and it is now structural: the walk starts from every id
/// in the view.
#[test]
fn an_override_cannot_strand_a_source_reference() {
    let en = parse("en", "inner = one\nm = { inner }\n");
    let es = parse("es", "inner = uno\n");
    codegen::check_locale(&es, Some(&en)).expect("`m` still reaches `inner`, now the Spanish one");
}

/// A cycle that exists in neither locale alone: the source reaches the locale's
/// override, which reaches back. Caught before and after this change (the walk
/// from the overriding node traverses the merged view), and now caught from
/// either end.
#[test]
fn a_cycle_created_only_by_composition_is_refused() {
    let en = parse("en", "a = { b }\nb = plain\n");
    let es = parse("es", "b = { a }\n");
    codegen::check_locale(&en, None).expect("the source alone is acyclic");
    codegen::check_locale(&es, None).expect_err("es alone cannot resolve `a`");

    let err = codegen::check_locale(&es, Some(&en)).expect_err("composed, it cycles");
    assert!(err.contains("cycle"), "{err}");
}

/// The variable rule is closed under composition by `check_translation`'s
/// pointwise bound: an override may only need what the *source's* accessor for
/// that same id already passes, and any message reaching it passes at least
/// that. Terms cannot widen it either, because a term body may not read a
/// variable at all (rule 12).
#[test]
fn an_override_cannot_demand_a_variable_the_accessor_never_passes() {
    let en = parse("en", "inner = plain\nm = { inner }\n");
    let es = parse("es", "inner = { $x }\n");
    let err = codegen::check_translation(&en, &es).expect_err("nothing passes $x");
    assert!(err.contains("$x"), "{err}");

    // A term override cannot smuggle one in: variables in term bodies are
    // refused outright, so a term contributes none however it is overridden.
    let err = codegen::parse_locale("es", "-t = { $x }\n").expect_err("refused at parse");
    assert!(err.contains("term"), "{err}");
}

/// Every shipped locale still validates with the whole view in scope.
#[test]
fn every_shipped_locale_validates_the_whole_view() {
    let parsed: Vec<codegen::LocaleDef> = i18n::LOCALES
        .iter()
        .map(|(tag, source)| parse(tag, source))
        .collect();
    let en = parsed.iter().find(|l| l.tag == "en").expect("source ships");
    for locale in &parsed {
        let base = (locale.tag != "en").then_some(en);
        codegen::check_locale(locale, base).unwrap_or_else(|e| panic!("{e}"));
    }
}
