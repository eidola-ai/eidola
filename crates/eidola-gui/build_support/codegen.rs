//! FTL → typed-accessor code generation.
//!
//! Included verbatim by `build.rs` (which supplies the file I/O) and by
//! `tests/i18n.rs` (which exercises the contract directly), so the rules below
//! are testable without committing deliberately-broken FTL to the locales tree.
//!
//! **The contract this module enforces — the complete list.** Anything not
//! refused here has to be safe at runtime, because `Localization::format`
//! treats a formatting error as impossible rather than as a case to recover
//! from. Add a rule here when you add a way to fail.
//!
//! *Shape of the tree*
//!
//! 1. **A locale directory must be a canonical language tag**
//!    ([`validate_locale_tag`]). The directory name *is* the tag at runtime, and
//!    negotiation resolves to canonical spellings — so `zh_Hans` would ship a
//!    translation nothing could ever select.
//! 2. **Malformed FTL in any locale is a build error.** Every shipped locale is
//!    parsed, not just English.
//!
//! *What English defines*
//!
//! 3. **English is the source.** One accessor per English message, taking one
//!    parameter per placeable variable — so a mistyped id or a changed argument
//!    set is a compile error at the call site.
//! 4. **A translation may not invent a message id.** Nothing would ever ask for
//!    it, so it is either dead weight or a typo silently falling back.
//! 5. **A translation may not need a variable English does not pass** — not in
//!    its own body and not through anything it references. The call site has no
//!    such argument, so it could only fail at runtime.
//! 6. **A missing translation is fine** — it renders as English (see the merge
//!    in `src/i18n.rs`).
//!
//! *What a pattern may contain*
//!
//! 7. **Every reference must resolve through the merged view** — locale first,
//!    English behind it, which is exactly what the runtime bundle contains. A
//!    message or term that exists in neither is refused, and **term chains are
//!    followed to their end**, not checked one hop deep.
//! 8. **Everything a locale ships is validated, reached or not**
//!    ([`check_locale`]). There is no dead-text exception: a term nothing
//!    references today is referenced tomorrow, and the reference site would not
//!    be the broken one.
//! 9. **No reference cycles**, message or term. Fluent gives up on one at
//!    runtime; the build gives up first.
//! 10. **No function references.** Nothing registers a Fluent function — see the
//!     note at the `FunctionReference` arm — so `{ NUMBER($n) }` would render
//!     `{NUMBER()}`. The number/date formatting work is what will register some
//!     and loosen this to the registered names.
//! 11. **No attributes and no attribute references** — on messages *or* terms.
//!     The model is one message, one accessor. An attribute is refused loudly
//!     rather than ignored, because ignoring it would let a translator write text
//!     that ships inside the binary and never reaches a screen — and a reference
//!     to one could then only fail.
//! 12. **Terms stay closed.** A term body may reference other terms, never
//!     messages or variables, so nothing can reach through a term for a variable
//!     to fill.
//! 13. **No duplicate message or term ids.** A later definition silently shadows
//!     the earlier one, so it is refused instead.

use std::collections::{BTreeMap, HashSet};

use fluent_syntax::ast;
use fluent_syntax::parser;

/// One message, reduced to what codegen needs. Everything here is **direct** —
/// what this message's own body says — because what a reference resolves to
/// depends on which locale is being built (see [`resolve_vars`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDef {
    /// The FTL id, e.g. `about-version-value`.
    pub id: String,
    /// Placeable variables read directly by this message, in first-appearance
    /// order.
    pub vars: Vec<String>,
    /// Message ids this message references directly (`{ other-message }`).
    pub refs: Vec<String>,
    /// Term ids this message references directly (`{ -brand }`).
    pub terms: Vec<String>,
    /// A one-line rendering of the value, for the accessor's doc comment.
    pub preview: String,
}

/// One term of a locale. A term body may not read variables or reference
/// messages (both refused at parse time), so the only thing left to follow is
/// the terms it reaches — which still has to be followed, since a term chain
/// ending nowhere fails at runtime exactly like a message chain does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermDef {
    /// The id without its leading `-`.
    pub id: String,
    /// Term ids this term references directly.
    pub terms: Vec<String>,
}

/// One shipped locale: its tag, its concatenated FTL source, its messages, and
/// the terms it defines.
#[derive(Debug, Clone)]
pub struct LocaleDef {
    pub tag: String,
    pub source: String,
    pub messages: Vec<MessageDef>,
    pub terms: Vec<TermDef>,
}

impl LocaleDef {
    pub fn message(&self, id: &str) -> Option<&MessageDef> {
        self.messages.iter().find(|m| m.id == id)
    }

    fn term(&self, id: &str) -> Option<&TermDef> {
        self.terms.iter().find(|t| t.id == id)
    }
}

/// The view a bundle actually resolves against: `primary` overriding `base`.
/// For the source locale `base` is `None` (it *is* the base).
struct MergedView<'a> {
    primary: &'a LocaleDef,
    base: Option<&'a LocaleDef>,
}

impl<'a> MergedView<'a> {
    fn message(&self, id: &str) -> Option<&'a MessageDef> {
        self.primary
            .message(id)
            .or_else(|| self.base.and_then(|b| b.message(id)))
    }

    fn term(&self, id: &str) -> Option<&'a TermDef> {
        self.primary
            .term(id)
            .or_else(|| self.base.and_then(|b| b.term(id)))
    }

    /// How to describe this view in an error — the locale being built.
    fn tag(&self) -> &str {
        &self.primary.tag
    }

    /// Where a reference could have been satisfied from, for an error message.
    fn scope(&self) -> &'static str {
        if self.base.is_some() {
            "this locale or the `en` source"
        } else {
            "the `en` source"
        }
    }
}

/// The variables `id` needs when formatted in `primary` (with `base` behind it):
/// its own, plus those of every message it reaches through, transitively.
///
/// Errors on anything the runtime could not resolve — an unknown message or
/// term, an attribute reference (no message may have attributes), or a
/// reference cycle. Those are exactly the formatting errors the runtime would
/// otherwise have to survive.
pub fn resolve_vars(
    id: &str,
    primary: &LocaleDef,
    base: Option<&LocaleDef>,
) -> Result<Vec<String>, String> {
    let view = MergedView { primary, base };
    let mut out = Vec::new();
    let mut stack = Vec::new();
    walk_message(id, &view, &mut stack, &mut out)?;
    Ok(out)
}

fn walk_message(
    id: &str,
    view: &MergedView,
    stack: &mut Vec<String>,
    out: &mut Vec<String>,
) -> Result<(), String> {
    if stack.iter().any(|s| s == id) {
        stack.push(id.to_string());
        return Err(format!(
            "{}: message reference cycle {} — Fluent cannot resolve it at runtime",
            view.tag(),
            stack.join(" → ")
        ));
    }
    let Some(message) = view.message(id) else {
        return Err(format!(
            "{}: message `{}` references `{id}`, which is not defined in {}",
            view.tag(),
            stack.last().map(String::as_str).unwrap_or(id),
            view.scope()
        ));
    };
    stack.push(id.to_string());
    for var in &message.vars {
        if !out.contains(var) {
            out.push(var.clone());
        }
    }
    for term in &message.terms {
        // A term chain is followed, not merely checked one hop deep: `-brand`
        // being defined says nothing about the `-missing` it reaches for.
        walk_term(term, view, &mut Vec::new())?;
    }
    for reference in &message.refs {
        walk_message(reference, view, stack, out)?;
    }
    stack.pop();
    Ok(())
}

/// Parse one locale's concatenated FTL source.
///
/// `tag` is only used in error messages. Every rule that can be decided from a
/// single file is decided here; cross-locale rules live in
/// [`check_translation`].
pub fn parse_locale(tag: &str, source: &str) -> Result<LocaleDef, String> {
    let resource = parser::parse(source).map_err(|(_, errors)| {
        let mut out = format!("{tag}: malformed FTL");
        for e in errors.iter().take(8) {
            out.push_str(&format!("\n  {e:?}"));
        }
        out
    })?;

    let mut messages: Vec<MessageDef> = Vec::new();
    let mut terms: Vec<TermDef> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for entry in &resource.body {
        match entry {
            ast::Entry::Message(message) => {
                let id = message.id.name.to_string();
                if !seen.insert(id.clone()) {
                    return Err(format!(
                        "{tag}: duplicate message `{id}` — a later definition silently \
                         shadows the earlier one, so it is refused here instead"
                    ));
                }
                if !message.attributes.is_empty() {
                    return Err(format!(
                        "{tag}: message `{id}` has attributes, which the codegen does not \
                         support — the model is one message, one accessor. Give the \
                         attribute its own message id."
                    ));
                }
                let Some(pattern) = message.value.as_ref() else {
                    return Err(format!("{tag}: message `{id}` has no value"));
                };
                let mut refs = Refs::default();
                walk_pattern(pattern, &mut refs)
                    .map_err(|e| format!("{tag}: message `{id}`: {e}"))?;
                messages.push(MessageDef {
                    id,
                    vars: refs.vars,
                    refs: refs.messages,
                    terms: refs.terms,
                    preview: preview(pattern),
                });
            }
            ast::Entry::Term(term) => {
                // Terms carry no accessor and are kept *closed*: a term body may
                // reference other terms, but a variable it reads could never be
                // filled (Fluent gives terms their own scope) and a message it
                // reached could drag in a variable nothing passes. Both are
                // refused so a term contributes only its own call arguments.
                if !term.attributes.is_empty() {
                    return Err(format!(
                        "{tag}: term `-{}` has attributes, which the codegen does not \
                         support — nothing may reference one, so the text would ship \
                         inside the binary and never reach a screen. Give it its own term.",
                        term.id.name
                    ));
                }
                let mut refs = Refs::default();
                walk_pattern(&term.value, &mut refs)
                    .map_err(|e| format!("{tag}: term `-{}`: {e}", term.id.name))?;
                if let Some(var) = refs.vars.first() {
                    return Err(format!(
                        "{tag}: term `-{}` reads variable `${var}` — a term has its own scope \
                         and never sees a message's arguments; pass it as a term argument \
                         instead",
                        term.id.name
                    ));
                }
                if let Some(message) = refs.messages.first() {
                    return Err(format!(
                        "{tag}: term `-{}` references message `{message}` — terms stay closed \
                         so nothing can reach through one for a variable to fill; inline the \
                         text or use a term",
                        term.id.name
                    ));
                }
                if terms.iter().any(|t| t.id == term.id.name) {
                    return Err(format!(
                        "{tag}: duplicate term `-{}` — a later definition silently shadows \
                         the earlier one, so it is refused here instead",
                        term.id.name
                    ));
                }
                terms.push(TermDef {
                    id: term.id.name.to_string(),
                    terms: refs.terms,
                });
            }
            ast::Entry::Junk { content } => {
                let head: String = content.chars().take(60).collect();
                return Err(format!("{tag}: unparsed FTL near `{}`", head.trim()));
            }
            ast::Entry::Comment(_)
            | ast::Entry::GroupComment(_)
            | ast::Entry::ResourceComment(_) => {}
        }
    }

    Ok(LocaleDef {
        tag: tag.to_string(),
        source: source.to_string(),
        messages,
        terms,
    })
}

/// Follow a term through the merged view: it must exist, and so must everything
/// it reaches, without cycling. Terms carry no variables of their own (a term
/// body reading `$var` is refused at parse time), so this validates reachability
/// rather than collecting anything.
fn walk_term(id: &str, view: &MergedView, stack: &mut Vec<String>) -> Result<(), String> {
    if stack.iter().any(|s| s == id) {
        stack.push(id.to_string());
        return Err(format!(
            "{}: term reference cycle {} — Fluent cannot resolve it at runtime",
            view.tag(),
            stack
                .iter()
                .map(|t| format!("-{t}"))
                .collect::<Vec<_>>()
                .join(" → ")
        ));
    }
    let Some(term) = view.term(id) else {
        return Err(format!(
            "{}: term `-{id}` is not defined in {}",
            view.tag(),
            view.scope()
        ));
    };
    stack.push(id.to_string());
    for nested in &term.terms {
        walk_term(nested, view, stack)?;
    }
    stack.pop();
    Ok(())
}

/// Everything this locale ships resolves through the view it will be formatted
/// in — `primary` overriding `base`, or `base: None` for the source locale
/// itself.
///
/// This walks **every** message and **every** term the locale defines, not only
/// what something else happens to reach. A term nothing references today is
/// referenced tomorrow, and the reference site would not be the broken one — so
/// the invariant is that all shipped FTL validates, with no dead-text exception.
pub fn check_locale(primary: &LocaleDef, base: Option<&LocaleDef>) -> Result<(), String> {
    let view = MergedView { primary, base };
    for message in &primary.messages {
        walk_message(&message.id, &view, &mut Vec::new(), &mut Vec::new())?;
    }
    for term in &primary.terms {
        walk_term(&term.id, &view, &mut Vec::new())?;
    }
    Ok(())
}

/// The cross-locale rules, resolved through the same merged view the runtime
/// formats through: a translation may only restate messages the source locale
/// has, and — following every reference into whatever the merge resolves it to
/// — may only end up needing variables the source locale's own message passes.
///
/// The second half is what the merge makes necessary. A translation's
/// `{ other-message }` reaches English's `other-message` when the locale did not
/// translate it, so the variables it needs are English's, not this locale's; a
/// check that looked only inside the translation would pass and the runtime
/// would render an unfilled placeable.
pub fn check_translation(en: &LocaleDef, other: &LocaleDef) -> Result<(), String> {
    for message in &other.messages {
        let Some(source) = en.message(&message.id) else {
            return Err(format!(
                "{}: message `{}` does not exist in the `en` source — nothing would ever \
                 ask for it. Add it to `locales/en/` first, or fix the id.",
                other.tag, message.id
            ));
        };
        // What the call site actually passes: the source message's own resolved
        // variables, which is what the generated accessor's parameters are.
        let passed = resolve_vars(&source.id, en, None)?;
        let needed = resolve_vars(&message.id, other, Some(en))?;
        for var in &needed {
            if !passed.contains(var) {
                return Err(format!(
                    "{}: message `{}` needs `${}`, which the `en` source does not pass — \
                     the call site has no such argument, so this could only fail at \
                     runtime. Source variables: {}",
                    other.tag,
                    message.id,
                    var,
                    if passed.is_empty() {
                        "(none)".to_string()
                    } else {
                        passed.join(", ")
                    }
                ));
            }
        }
    }
    Ok(())
}

/// Emit the generated module: the embedded FTL for every locale, the source
/// locale's message ids, and one typed accessor per message.
///
/// `locales` must include the source locale and be ordered — the caller sorts
/// by tag so the output is byte-stable across machines.
pub fn generate(en: &LocaleDef, locales: &[LocaleDef]) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(
        "// @generated by build.rs from crates/eidola-gui/locales — do not edit by hand.\n\
         //\n\
         // The FTL is embedded as string literals rather than read at runtime: a\n\
         // translation loaded from disk would be an unmeasured input able to rewrite\n\
         // security-critical UI text, so every string ships inside the measured artifact.\n\n",
    );

    out.push_str(
        "/// Every shipped locale, as `(tag, concatenated FTL source)`, sorted by tag.\n\
         #[allow(dead_code)]\n\
         pub const LOCALES: &[(&str, &str)] = &[\n",
    );
    for locale in locales {
        out.push_str(&format!(
            "    ({:?}, {:?}),\n",
            locale.tag,
            locale.source.as_str()
        ));
    }
    out.push_str("];\n\n");

    out.push_str(
        "/// Every message id in the source locale, in file order.\n\
         #[allow(dead_code)]\n\
         pub const MESSAGE_IDS: &[&str] = &[\n",
    );
    for message in &en.messages {
        out.push_str(&format!("    {:?},\n", message.id));
    }
    out.push_str("];\n\n");

    out.push_str(
        "/// Typed accessors — one per message of the source locale.\n\
         ///\n\
         /// A mistyped id is a missing function; a changed argument set is a changed\n\
         /// signature. Both are compile errors at the call site, which is the whole\n\
         /// point of generating this file.\n\
         #[allow(dead_code)]\n\
         pub mod msg {\n",
    );

    let mut fn_names: BTreeMap<String, String> = BTreeMap::new();
    for message in &en.messages {
        let fn_name = rust_ident(&message.id)?;
        if let Some(previous) = fn_names.insert(fn_name.clone(), message.id.clone()) {
            return Err(format!(
                "messages `{previous}` and `{}` both generate the accessor `{fn_name}`",
                message.id
            ));
        }

        // The accessor's parameters are the message's *resolved* variables —
        // its own plus everything it reaches through. This call is also what
        // validates the source locale's own references (an unknown id, an
        // attribute reference, a cycle), since nothing else walks them.
        let mut params: Vec<(String, String)> = Vec::new();
        for var in &resolve_vars(&message.id, en, None)? {
            let ident = rust_ident(var)?;
            if params.iter().any(|(i, _)| *i == ident) {
                return Err(format!(
                    "message `{}` has two variables that generate the parameter `{ident}`",
                    message.id
                ));
            }
            params.push((ident, var.clone()));
        }

        out.push_str(&format!(
            "    /// `{}` — en: {:?}\n",
            message.id, message.preview
        ));
        if params.is_empty() {
            out.push_str(&format!(
                "    pub fn {fn_name}(cx: &gpui::App) -> gpui::SharedString {{\n\
                 \x20       super::format(cx, {:?}, None)\n\
                 \x20   }}\n\n",
                message.id
            ));
        } else {
            let signature: Vec<String> = params
                .iter()
                .map(|(ident, _)| format!("{ident}: impl Into<super::Arg<'a>>"))
                .collect();
            out.push_str(&format!(
                "    pub fn {fn_name}<'a>(cx: &gpui::App, {}) -> gpui::SharedString {{\n\
                 \x20       let mut args = super::Args::new();\n",
                signature.join(", ")
            ));
            for (ident, var) in &params {
                out.push_str(&format!("        args.set({var:?}, {ident});\n"));
            }
            out.push_str(&format!(
                "        super::format(cx, {:?}, Some(&args))\n    }}\n\n",
                message.id
            ));
        }
    }
    out.push_str("}\n");
    Ok(out)
}

/// A locale directory name must be a **canonical** language identifier.
///
/// The directory name becomes the locale's tag verbatim — it is what the
/// runtime parses to build the bundle and what negotiation matches against. A
/// tag that is merely *parseable* is not enough: negotiation resolves to the
/// canonical spelling (`zh-Hans`), so a `zh_Hans` or `zh-hans` directory ships a
/// translation inside the binary that nothing can ever select, silently. The
/// round-trip through `LanguageIdentifier` is what makes "canonical" precise.
pub fn validate_locale_tag(tag: &str) -> Result<(), String> {
    let parsed: unic_langid::LanguageIdentifier = tag.parse().map_err(|e| {
        format!(
            "locale directory `{tag}` is not a language tag ({e}) — name it for the locale \
             it holds, e.g. `en`, `fr`, `zh-Hans`"
        )
    })?;
    let canonical = parsed.to_string();
    if canonical != tag {
        return Err(format!(
            "locale directory `{tag}` is not the canonical spelling of the tag it parses to \
             (`{canonical}`) — negotiation resolves to the canonical form, so the directory \
             must be `{canonical}` or its translations could never be selected"
        ));
    }
    Ok(())
}

/// A message or variable name turned into a Rust identifier: `-` becomes `_`,
/// and a name that collides with a keyword takes a trailing `_`.
pub fn rust_ident(name: &str) -> Result<String, String> {
    let mut ident: String = name
        .chars()
        .map(|c| if c == '-' { '_' } else { c })
        .collect();
    if ident.is_empty() {
        return Err("empty identifier".to_string());
    }
    if !ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err(format!(
            "`{name}` cannot become a Rust identifier — use ASCII letters, digits, \
             `-` and `_` in FTL ids"
        ));
    }
    if ident.starts_with(|c: char| c.is_ascii_digit()) {
        return Err(format!("`{name}` starts with a digit"));
    }
    if RUST_KEYWORDS.contains(&ident.as_str()) {
        ident.push('_');
    }
    Ok(ident)
}

const RUST_KEYWORDS: &[&str] = &[
    "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub",
    "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true", "type",
    "unsafe", "use", "where", "while", "abstract", "become", "box", "do", "final", "gen", "macro",
    "override", "priv", "try", "typeof", "unsized", "virtual", "yield",
];

// ---------------------------------------------------------------------------
// AST walking
// ---------------------------------------------------------------------------

/// What one pattern refers to, in first-appearance order.
#[derive(Default)]
struct Refs {
    vars: Vec<String>,
    messages: Vec<String>,
    terms: Vec<String>,
}

fn walk_pattern(pattern: &ast::Pattern<&str>, refs: &mut Refs) -> Result<(), String> {
    for element in &pattern.elements {
        if let ast::PatternElement::Placeable { expression } = element {
            walk_expression(expression, refs)?;
        }
    }
    Ok(())
}

fn walk_expression(expr: &ast::Expression<&str>, refs: &mut Refs) -> Result<(), String> {
    match expr {
        ast::Expression::Select { selector, variants } => {
            walk_inline(selector, refs)?;
            for variant in variants {
                walk_pattern(&variant.value, refs)?;
            }
        }
        ast::Expression::Inline(inline) => walk_inline(inline, refs)?,
    }
    Ok(())
}

fn walk_inline(inline: &ast::InlineExpression<&str>, refs: &mut Refs) -> Result<(), String> {
    match inline {
        ast::InlineExpression::VariableReference { id } => push_unique(&mut refs.vars, id.name),
        // No message may have attributes, so an attribute reference can only
        // ever fail to resolve. Refusing it here is what lets the runtime treat
        // a formatting error as impossible.
        ast::InlineExpression::MessageReference {
            id,
            attribute: Some(attribute),
        } => {
            return Err(format!(
                "`{}.{}` references an attribute, which no message may have",
                id.name, attribute.name
            ));
        }
        ast::InlineExpression::MessageReference { id, .. } => {
            push_unique(&mut refs.messages, id.name)
        }
        ast::InlineExpression::TermReference {
            id,
            attribute: Some(attribute),
            ..
        } => {
            return Err(format!(
                "`-{}.{}` references a term attribute, which the codegen does not support",
                id.name, attribute.name
            ));
        }
        // A term's own body has its own scope; only what is *passed* to it
        // comes from the message's arguments.
        ast::InlineExpression::TermReference { id, arguments, .. } => {
            push_unique(&mut refs.terms, id.name);
            walk_call_arguments(arguments.as_ref(), refs)?;
        }
        // No function is callable. `FluentBundle::new` starts with an empty
        // entry table and fluent-bundle registers nothing of its own — its
        // `add_builtins()` is opt-in and, at 0.16, supplies only `NUMBER`
        // (`DATETIME` is still a TODO upstream). `bundle_for` calls neither, so
        // every function reference resolves to nothing and renders its own error
        // text. Refusing them here keeps that impossible; the number/date
        // formatting work is what will register functions and loosen this to a
        // list of the registered names.
        ast::InlineExpression::FunctionReference { id, .. } => {
            return Err(format!(
                "`{}()` is a function reference, but no Fluent function is registered — \
                 the runtime would render `{{{}()}}` instead of a value",
                id.name, id.name
            ));
        }
        ast::InlineExpression::Placeable { expression } => walk_expression(expression, refs)?,
        ast::InlineExpression::StringLiteral { .. }
        | ast::InlineExpression::NumberLiteral { .. } => {}
    }
    Ok(())
}

fn walk_call_arguments(
    arguments: Option<&ast::CallArguments<&str>>,
    refs: &mut Refs,
) -> Result<(), String> {
    let Some(arguments) = arguments else {
        return Ok(());
    };
    for positional in &arguments.positional {
        walk_inline(positional, refs)?;
    }
    for named in &arguments.named {
        walk_inline(&named.value, refs)?;
    }
    Ok(())
}

fn push_unique(list: &mut Vec<String>, name: &str) {
    if !list.iter().any(|n| n == name) {
        list.push(name.to_string());
    }
}

/// A one-line rendering of a pattern for the accessor's doc comment: text as
/// written, placeables collapsed to `{$var}` / `{…}`.
fn preview(pattern: &ast::Pattern<&str>) -> String {
    let mut out = String::new();
    for element in &pattern.elements {
        match element {
            ast::PatternElement::TextElement { value } => out.push_str(value),
            ast::PatternElement::Placeable { expression } => match expression {
                ast::Expression::Inline(ast::InlineExpression::VariableReference { id }) => {
                    out.push_str(&format!("{{${}}}", id.name));
                }
                _ => out.push_str("{…}"),
            },
        }
    }
    let out = out.split_whitespace().collect::<Vec<_>>().join(" ");
    if out.chars().count() > 96 {
        let head: String = out.chars().take(93).collect();
        format!("{head}…")
    } else {
        out
    }
}
