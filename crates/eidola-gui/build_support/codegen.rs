//! FTL → typed-accessor code generation.
//!
//! Included verbatim by `build.rs` (which supplies the file I/O) and by
//! `tests/i18n.rs` (which exercises the contract directly), so the rules below
//! are testable without committing deliberately-broken FTL to the locales tree.
//!
//! The contract this module enforces, in one place:
//!
//! 1. **English is the source.** One accessor per English message, taking one
//!    parameter per placeable variable — so a mistyped id or a changed argument
//!    set is a compile error at the call site.
//! 2. **Malformed FTL in any locale is a build error.** Every shipped locale is
//!    parsed, not just English.
//! 3. **A translation may not reference a variable English lacks** — nothing
//!    would ever pass it, so it could only fail at runtime.
//! 4. **A translation may not invent a message id.** Nothing would ever ask for
//!    it, so it is either dead weight or a typo silently falling back.
//! 5. **A missing translation is fine** — the runtime falls back to English.
//!
//! Attributes and message-level `Term` accessors are deliberately unsupported:
//! the model is one message, one accessor. An attribute is rejected loudly
//! rather than ignored, because ignoring it would let a translator write text
//! that never reaches a screen.

use std::collections::{BTreeMap, HashMap, HashSet};

use fluent_syntax::ast;
use fluent_syntax::parser;

/// One message of the source locale, reduced to what codegen needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageDef {
    /// The FTL id, e.g. `about-version-value`.
    pub id: String,
    /// Placeable variables, in first-appearance order — the accessor's
    /// parameter list.
    pub vars: Vec<String>,
    /// A one-line rendering of the value, for the accessor's doc comment.
    pub preview: String,
}

/// One shipped locale: its tag, its concatenated FTL source, and its messages.
#[derive(Debug, Clone)]
pub struct LocaleDef {
    pub tag: String,
    pub source: String,
    pub messages: Vec<MessageDef>,
}

impl LocaleDef {
    pub fn message(&self, id: &str) -> Option<&MessageDef> {
        self.messages.iter().find(|m| m.id == id)
    }
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

    // Pass 1: direct variables and message references, per message.
    let mut direct: Vec<(String, Vec<String>, Vec<String>, String)> = Vec::new();
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
                let mut vars = Vec::new();
                let mut refs = Vec::new();
                walk_pattern(pattern, &mut vars, &mut refs);
                direct.push((id, vars, refs, preview(pattern)));
            }
            ast::Entry::Term(term) => {
                // Terms carry no accessor, but a term whose body reads a
                // variable is a scoping mistake (Fluent gives terms their own
                // scope), so it is worth refusing early.
                let mut vars = Vec::new();
                let mut refs = Vec::new();
                walk_pattern(&term.value, &mut vars, &mut refs);
                if !vars.is_empty() {
                    return Err(format!(
                        "{tag}: term `-{}` reads variable `${}` — a term has its own scope \
                         and never sees a message's arguments; pass it as a term argument \
                         instead",
                        term.id.name, vars[0]
                    ));
                }
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

    // Pass 2: fold each message's referenced messages into its variable set —
    // `{ other-message }` needs whatever `other-message` needs.
    let index: HashMap<&str, usize> = direct
        .iter()
        .enumerate()
        .map(|(i, (id, _, _, _))| (id.as_str(), i))
        .collect();
    let mut messages = Vec::with_capacity(direct.len());
    for (i, (id, _, _, preview)) in direct.iter().enumerate() {
        let mut vars: Vec<String> = Vec::new();
        let mut visiting: HashSet<usize> = HashSet::new();
        collect_transitive(i, &direct, &index, &mut visiting, &mut vars);
        messages.push(MessageDef {
            id: id.clone(),
            vars,
            preview: preview.clone(),
        });
    }

    Ok(LocaleDef {
        tag: tag.to_string(),
        source: source.to_string(),
        messages,
    })
}

/// Fold `i`'s own variables plus every referenced message's, depth-first, in
/// first-appearance order. A reference cycle is walked once and left alone —
/// Fluent detects it at runtime, and it is not this pass's error to report.
fn collect_transitive(
    i: usize,
    direct: &[(String, Vec<String>, Vec<String>, String)],
    index: &HashMap<&str, usize>,
    visiting: &mut HashSet<usize>,
    out: &mut Vec<String>,
) {
    if !visiting.insert(i) {
        return;
    }
    for v in &direct[i].1 {
        if !out.contains(v) {
            out.push(v.clone());
        }
    }
    for r in &direct[i].2 {
        if let Some(&j) = index.get(r.as_str()) {
            collect_transitive(j, direct, index, visiting, out);
        }
    }
}

/// The cross-locale rules: a translation may only restate messages the source
/// locale has, using only the variables the source locale passes.
pub fn check_translation(en: &LocaleDef, other: &LocaleDef) -> Result<(), String> {
    for message in &other.messages {
        let Some(source) = en.message(&message.id) else {
            return Err(format!(
                "{}: message `{}` does not exist in the `en` source — nothing would ever \
                 ask for it. Add it to `locales/en/` first, or fix the id.",
                other.tag, message.id
            ));
        };
        for var in &message.vars {
            if !source.vars.contains(var) {
                return Err(format!(
                    "{}: message `{}` reads `${}`, which the `en` source does not pass — \
                     the call site has no such argument, so this could only fail at \
                     runtime. Source variables: {}",
                    other.tag,
                    message.id,
                    var,
                    if source.vars.is_empty() {
                        "(none)".to_string()
                    } else {
                        source.vars.join(", ")
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

        let mut params: Vec<(String, String)> = Vec::new();
        for var in &message.vars {
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

fn walk_pattern(pattern: &ast::Pattern<&str>, vars: &mut Vec<String>, refs: &mut Vec<String>) {
    for element in &pattern.elements {
        if let ast::PatternElement::Placeable { expression } = element {
            walk_expression(expression, vars, refs);
        }
    }
}

fn walk_expression(expr: &ast::Expression<&str>, vars: &mut Vec<String>, refs: &mut Vec<String>) {
    match expr {
        ast::Expression::Select { selector, variants } => {
            walk_inline(selector, vars, refs);
            for variant in variants {
                walk_pattern(&variant.value, vars, refs);
            }
        }
        ast::Expression::Inline(inline) => walk_inline(inline, vars, refs),
    }
}

fn walk_inline(
    inline: &ast::InlineExpression<&str>,
    vars: &mut Vec<String>,
    refs: &mut Vec<String>,
) {
    match inline {
        ast::InlineExpression::VariableReference { id } => push_unique(vars, id.name),
        ast::InlineExpression::MessageReference { id, .. } => push_unique(refs, id.name),
        // A term's own body has its own scope; only what is *passed* to it
        // comes from the message's arguments.
        ast::InlineExpression::TermReference { arguments, .. } => {
            walk_call_arguments(arguments.as_ref(), vars, refs);
        }
        ast::InlineExpression::FunctionReference { arguments, .. } => {
            walk_call_arguments(Some(arguments), vars, refs);
        }
        ast::InlineExpression::Placeable { expression } => walk_expression(expression, vars, refs),
        ast::InlineExpression::StringLiteral { .. }
        | ast::InlineExpression::NumberLiteral { .. } => {}
    }
}

fn walk_call_arguments(
    arguments: Option<&ast::CallArguments<&str>>,
    vars: &mut Vec<String>,
    refs: &mut Vec<String>,
) {
    let Some(arguments) = arguments else { return };
    for positional in &arguments.positional {
        walk_inline(positional, vars, refs);
    }
    for named in &arguments.named {
        walk_inline(&named.value, vars, refs);
    }
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
