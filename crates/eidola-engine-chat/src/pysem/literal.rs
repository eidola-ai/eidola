//! `ast.literal_eval` for the JSON-representable subset of Python literals.
//!
//! Accepts what CPython's `ast.literal_eval(s)` accepts and produces the same
//! value, for every input whose result `json.dumps` can serialise: strings
//! (every prefix, quote style and escape except `\N{…}`), integers in every
//! base, floats, `True`/`False`/`None`, lists, tuples, dicts, a sign on a
//! number, implicit string concatenation, comments, and line joining.
//!
//! Inputs whose result `json.dumps` would reject (bytes, complex numbers,
//! sets, `...`, tuple keys) are reported as [`PyError::Uncaught`], matching
//! the failure the reference parser hits one step later. `\N{NAME}` escapes
//! need the Unicode name database and are reported as [`PyError::Invalid`]
//! even when the name is valid; that is the one known divergence.

use super::{PyError, is_py_space};
use crate::json::{BigInt, PY_MAX_STR_DIGITS};

/// CPython's tokenizer refuses more than 200 nested brackets.
const MAX_NESTING: usize = 200;

/// A literal value.
#[derive(Clone, Debug, PartialEq)]
pub enum PyLit {
    None,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Str(String),
    List(Vec<PyLit>),
    Tuple(Vec<PyLit>),
    /// Keys deduplicated by Python equality (first key kept, last value).
    Dict(Vec<(PyLit, PyLit)>),
}

/// `ast.literal_eval(s)`.
pub fn literal_eval(s: &str) -> Result<PyLit, PyError> {
    let source = s.trim_start_matches([' ', '\t']);
    if source.contains('\0') {
        return Err(PyError::Invalid);
    }
    // CPython translates every newline convention to `\n` before tokenizing,
    // including inside string literals.
    let source = source.replace("\r\n", "\n").replace('\r', "\n");
    let tokens = tokenize(&source)?;
    let mut parser = Parser { tokens, pos: 0 };
    let value = parser.expression_list(0)?;
    if parser.pos != parser.tokens.len() {
        return Err(PyError::Invalid);
    }
    Ok(value)
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(BigInt),
    Float(f64),
    /// Imaginary literal: valid syntax, but `json.dumps` rejects the result.
    Imaginary,
    Str(String),
    /// A bytes literal: likewise unserialisable.
    Bytes,
    /// An f-string or t-string: not a literal.
    Formatted,
    Name(String),
    Op(&'static str),
}

fn tokenize(src: &str) -> Result<Vec<Tok>, PyError> {
    let chars: Vec<char> = src.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut depth = 0usize;
    // Start of a logical line outside brackets: indentation is checked there.
    let mut line_start = true;
    let mut first_line = true;
    let mut saw_expression_line = false;
    while i < chars.len() {
        let c = chars[i];
        if line_start && depth == 0 {
            // Measure the indentation of this physical line.
            let mut j = i;
            let mut indent = 0;
            while j < chars.len() && matches!(chars[j], ' ' | '\t' | '\x0c') {
                indent = if chars[j] == '\x0c' { 0 } else { indent + 1 };
                j += 1;
            }
            let blank = j == chars.len() || matches!(chars[j], '\n' | '#');
            if !blank {
                if saw_expression_line {
                    // A second logical line in eval mode.
                    return Err(PyError::Invalid);
                }
                if indent > 0 && !first_line {
                    return Err(PyError::Invalid);
                }
                saw_expression_line = true;
            }
            line_start = false;
            i = j;
            continue;
        }
        match c {
            ' ' | '\t' | '\x0c' => i += 1,
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            }
            '\n' => {
                i += 1;
                if depth == 0 {
                    line_start = true;
                    first_line = false;
                }
            }
            '\\' => {
                if chars.get(i + 1) == Some(&'\n') {
                    i += 2;
                    if i == chars.len() {
                        // EOF in a continued line.
                        return Err(PyError::Invalid);
                    }
                } else {
                    return Err(PyError::Invalid);
                }
            }
            '(' | '[' | '{' => {
                depth += 1;
                if depth >= MAX_NESTING {
                    return Err(PyError::Invalid);
                }
                tokens.push(Tok::Op(match c {
                    '(' => "(",
                    '[' => "[",
                    _ => "{",
                }));
                i += 1;
            }
            ')' | ']' | '}' => {
                if depth == 0 {
                    return Err(PyError::Invalid);
                }
                depth -= 1;
                tokens.push(Tok::Op(match c {
                    ')' => ")",
                    ']' => "]",
                    _ => "}",
                }));
                i += 1;
            }
            ',' | ':' | '+' | '-' => {
                tokens.push(Tok::Op(match c {
                    ',' => ",",
                    ':' => ":",
                    '+' => "+",
                    _ => "-",
                }));
                i += 1;
            }
            '.' if chars.get(i + 1).is_some_and(char::is_ascii_digit) => {
                i = number(&chars, i, &mut tokens)?;
            }
            '.' if chars.get(i + 1) == Some(&'.') && chars.get(i + 2) == Some(&'.') => {
                // Ellipsis: a constant json.dumps rejects.
                tokens.push(Tok::Name("...".to_string()));
                i += 3;
            }
            '0'..='9' => i = number(&chars, i, &mut tokens)?,
            '\'' | '"' => i = string(&chars, i, "", &mut tokens)?,
            c if is_identifier_start(c) => {
                let start = i;
                while i < chars.len() && is_identifier_continue(chars[i]) {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                if matches!(chars.get(i), Some('\'' | '"')) && is_string_prefix(&word) {
                    i = string(&chars, i, &word.to_ascii_lowercase(), &mut tokens)?;
                } else {
                    tokens.push(Tok::Name(word));
                }
            }
            // Any other character is either a syntax error or an operator
            // that cannot appear in a literal.
            _ => return Err(PyError::Invalid),
        }
    }
    if depth != 0 {
        return Err(PyError::Invalid);
    }
    Ok(tokens)
}

fn is_identifier_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_identifier_continue(c: char) -> bool {
    c == '_' || c.is_alphanumeric() || (!c.is_ascii() && !is_py_space(c) && !c.is_control())
}

fn is_string_prefix(word: &str) -> bool {
    matches!(
        word.to_ascii_lowercase().as_str(),
        "r" | "u" | "b" | "br" | "rb" | "f" | "fr" | "rf" | "t" | "tr" | "rt"
    )
}

/// Scans a numeric literal starting at `i`.
fn number(chars: &[char], mut i: usize, tokens: &mut Vec<Tok>) -> Result<usize, PyError> {
    let start = i;
    let digits_with_underscores =
        |i: &mut usize, valid: &dyn Fn(char) -> bool| -> Result<usize, PyError> {
            // `digit (_? digit)*`; returns the digit count.
            let mut count = 0;
            while *i < chars.len() {
                let c = chars[*i];
                if valid(c) {
                    count += 1;
                    *i += 1;
                } else if c == '_' {
                    if count == 0 || !chars.get(*i + 1).is_some_and(|n| valid(*n)) {
                        return Err(PyError::Invalid);
                    }
                    *i += 1;
                } else {
                    break;
                }
            }
            Ok(count)
        };
    if chars[i] == '0' && matches!(chars.get(i + 1), Some('x' | 'X' | 'o' | 'O' | 'b' | 'B')) {
        let radix = match chars[i + 1] {
            'x' | 'X' => 16,
            'o' | 'O' => 8,
            _ => 2,
        };
        i += 2;
        // An underscore may follow the base prefix.
        if chars.get(i) == Some(&'_') {
            i += 1;
        }
        let digit_start = i;
        let count = digits_with_underscores(&mut i, &|c: char| c.is_digit(radix))?;
        if count == 0 {
            return Err(PyError::Invalid);
        }
        end_of_number(chars, i)?;
        let digits: Vec<u32> = chars[digit_start..i]
            .iter()
            .filter(|c| **c != '_')
            .map(|c| c.to_digit(radix).expect("validated digit"))
            .collect();
        let value = BigInt::from_radix_digits(false, radix, &digits);
        tokens.push(Tok::Int(value));
        return Ok(i);
    }
    let decimal = |c: char| c.is_ascii_digit();
    let int_count = digits_with_underscores(&mut i, &decimal)?;
    let mut is_float = false;
    if chars.get(i) == Some(&'.') {
        is_float = true;
        i += 1;
        if chars.get(i).is_some_and(char::is_ascii_digit) {
            digits_with_underscores(&mut i, &decimal)?;
        } else if int_count == 0 {
            return Err(PyError::Invalid);
        }
    }
    if matches!(chars.get(i), Some('e' | 'E')) {
        let mut j = i + 1;
        if matches!(chars.get(j), Some('+' | '-')) {
            j += 1;
        }
        if chars.get(j).is_some_and(char::is_ascii_digit) {
            is_float = true;
            i = j;
            digits_with_underscores(&mut i, &decimal)?;
        } else {
            return Err(PyError::Invalid);
        }
    }
    let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
    if matches!(chars.get(i), Some('j' | 'J')) {
        i += 1;
        end_of_number(chars, i)?;
        tokens.push(Tok::Imaginary);
        return Ok(i);
    }
    end_of_number(chars, i)?;
    if is_float {
        tokens.push(Tok::Float(text.parse().map_err(|_| PyError::Invalid)?));
    } else {
        // Leading zeros are only allowed when every digit is zero.
        if text.len() > 1 && text.starts_with('0') && text.bytes().any(|b| b != b'0') {
            return Err(PyError::Invalid);
        }
        // The digit limit applies to the conversion, which CPython skips
        // for an all-zero literal.
        if text.len() > PY_MAX_STR_DIGITS && text.bytes().any(|b| b != b'0') {
            return Err(PyError::Invalid);
        }
        tokens.push(Tok::Int(BigInt::from_digits(false, &text)));
    }
    Ok(i)
}

/// A number may not run straight into an identifier character.
fn end_of_number(chars: &[char], i: usize) -> Result<(), PyError> {
    match chars.get(i) {
        Some(&c)
            if is_identifier_continue(c)
                || c == '.' && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit()) =>
        {
            Err(PyError::Invalid)
        }
        _ => Ok(()),
    }
}

/// Scans a string literal whose opening quote is at `i`.
fn string(
    chars: &[char],
    mut i: usize,
    prefix: &str,
    tokens: &mut Vec<Tok>,
) -> Result<usize, PyError> {
    let raw = prefix.contains('r');
    let bytes = prefix.contains('b');
    let formatted = prefix.contains('f') || prefix.contains('t');
    let quote = chars[i];
    let triple = chars.get(i + 1) == Some(&quote) && chars.get(i + 2) == Some(&quote);
    i += if triple { 3 } else { 1 };
    let body_start = i;
    loop {
        let Some(&c) = chars.get(i) else {
            return Err(PyError::Invalid);
        };
        if c == '\\' {
            if i + 1 >= chars.len() {
                return Err(PyError::Invalid);
            }
            i += 2;
            continue;
        }
        if c == '\n' && !triple {
            return Err(PyError::Invalid);
        }
        if c == quote
            && (!triple || (chars.get(i + 1) == Some(&quote) && chars.get(i + 2) == Some(&quote)))
        {
            break;
        }
        i += 1;
    }
    let body = &chars[body_start..i];
    i += if triple { 3 } else { 1 };
    if formatted {
        tokens.push(Tok::Formatted);
    } else if bytes {
        if body.iter().any(|c| !c.is_ascii()) {
            return Err(PyError::Invalid);
        }
        tokens.push(Tok::Bytes);
    } else if raw {
        tokens.push(Tok::Str(body.iter().collect()));
    } else {
        tokens.push(Tok::Str(decode_escapes(body)?));
    }
    Ok(i)
}

fn decode_escapes(body: &[char]) -> Result<String, PyError> {
    let mut out = String::with_capacity(body.len());
    let mut i = 0;
    let hex = |digits: &[char]| -> Option<u32> {
        if digits.iter().all(char::is_ascii_hexdigit) {
            u32::from_str_radix(&digits.iter().collect::<String>(), 16).ok()
        } else {
            None
        }
    };
    while i < body.len() {
        let c = body[i];
        if c != '\\' {
            out.push(c);
            i += 1;
            continue;
        }
        let e = body[i + 1];
        i += 2;
        let push_code = |out: &mut String, code: u32| -> Result<(), PyError> {
            // Lone surrogates are valid Python strings but cannot be encoded
            // in a response, where the reference request fails.
            match char::from_u32(code) {
                Some(ch) => {
                    out.push(ch);
                    Ok(())
                }
                None if (0xd800..0xe000).contains(&code) => Err(PyError::Uncaught),
                None => Err(PyError::Invalid),
            }
        };
        match e {
            '\n' => {}
            '\\' => out.push('\\'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'a' => out.push('\x07'),
            'b' => out.push('\x08'),
            'f' => out.push('\x0c'),
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            'v' => out.push('\x0b'),
            '0'..='7' => {
                let mut code = e.to_digit(8).expect("octal digit");
                let mut taken = 1;
                while taken < 3 && body.get(i).is_some_and(|d| d.is_digit(8)) {
                    code = code * 8 + body[i].to_digit(8).expect("octal digit");
                    i += 1;
                    taken += 1;
                }
                push_code(&mut out, code)?;
            }
            'x' | 'u' | 'U' => {
                let len = match e {
                    'x' => 2,
                    'u' => 4,
                    _ => 8,
                };
                let Some(code) = body.get(i..i + len).and_then(hex) else {
                    return Err(PyError::Invalid);
                };
                i += len;
                push_code(&mut out, code)?;
            }
            'N' => return Err(PyError::Invalid),
            other => {
                out.push('\\');
                out.push(other);
            }
        }
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn eat_op(&mut self, op: &str) -> bool {
        if matches!(self.peek(), Some(Tok::Op(o)) if *o == op) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn at_op(&self, op: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(o)) if *o == op)
    }

    /// `expr (, expr)* [,]` — a bare tuple when any comma is present.
    fn expression_list(&mut self, depth: usize) -> Result<PyLit, PyError> {
        let first = self.expression(depth)?;
        if !self.at_op(",") {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.eat_op(",") {
            if self.peek().is_none() || self.at_op(")") {
                break;
            }
            items.push(self.expression(depth)?);
        }
        Ok(PyLit::Tuple(items))
    }

    fn expression(&mut self, depth: usize) -> Result<PyLit, PyError> {
        let value = if self.at_op("+") || self.at_op("-") {
            let negative = self.at_op("-");
            self.pos += 1;
            // The operand of a sign must be a bare numeric constant, possibly
            // parenthesised.
            let mut parens = 0;
            while self.eat_op("(") {
                parens += 1;
            }
            let value = match self.peek() {
                Some(Tok::Int(i)) => PyLit::Int(BigInt {
                    negative: negative != i.negative && i.digits != "0",
                    digits: i.digits.clone(),
                }),
                Some(Tok::Float(f)) => PyLit::Float(if negative { -f } else { *f }),
                Some(Tok::Imaginary) => return Err(PyError::Uncaught),
                _ => return Err(PyError::Invalid),
            };
            self.pos += 1;
            for _ in 0..parens {
                if !self.eat_op(")") {
                    return Err(PyError::Invalid);
                }
            }
            value
        } else {
            self.atom(depth)?
        };
        // A binary `+`/`-` only yields a literal when the right side is
        // complex, which json.dumps rejects; anything else is not a literal.
        if self.at_op("+") || self.at_op("-") {
            return Err(PyError::Invalid);
        }
        Ok(value)
    }

    fn atom(&mut self, depth: usize) -> Result<PyLit, PyError> {
        if depth >= MAX_NESTING {
            return Err(PyError::Invalid);
        }
        let Some(token) = self.peek().cloned() else {
            return Err(PyError::Invalid);
        };
        self.pos += 1;
        let value = match token {
            Tok::Int(i) => PyLit::Int(i),
            Tok::Float(f) => PyLit::Float(f),
            Tok::Imaginary | Tok::Bytes => {
                self.skip_string_run();
                self.check_no_trailer()?;
                return Err(PyError::Uncaught);
            }
            Tok::Formatted => return Err(PyError::Invalid),
            Tok::Str(first) => {
                let mut s = first;
                loop {
                    match self.peek() {
                        Some(Tok::Str(next)) => {
                            s.push_str(next);
                            self.pos += 1;
                        }
                        Some(Tok::Bytes) => return Err(PyError::Invalid),
                        Some(Tok::Formatted) => return Err(PyError::Invalid),
                        _ => break,
                    }
                }
                PyLit::Str(s)
            }
            Tok::Name(name) => match name.as_str() {
                "True" => PyLit::Bool(true),
                "False" => PyLit::Bool(false),
                "None" => PyLit::None,
                "..." => {
                    self.check_no_trailer()?;
                    return Err(PyError::Uncaught);
                }
                "set" if self.at_op("(") => {
                    self.pos += 1;
                    if !self.eat_op(")") {
                        return Err(PyError::Invalid);
                    }
                    self.check_no_trailer()?;
                    return Err(PyError::Uncaught);
                }
                _ => return Err(PyError::Invalid),
            },
            Tok::Op("(") => {
                if self.eat_op(")") {
                    PyLit::Tuple(Vec::new())
                } else {
                    let inner = self.expression_list(depth + 1)?;
                    if !self.eat_op(")") {
                        return Err(PyError::Invalid);
                    }
                    inner
                }
            }
            Tok::Op("[") => {
                let mut items = Vec::new();
                while !self.at_op("]") {
                    items.push(self.expression(depth + 1)?);
                    if !self.eat_op(",") {
                        break;
                    }
                }
                if !self.eat_op("]") {
                    return Err(PyError::Invalid);
                }
                PyLit::List(items)
            }
            Tok::Op("{") => self.dict_or_set(depth)?,
            Tok::Op(_) => return Err(PyError::Invalid),
        };
        self.check_no_trailer()?;
        Ok(value)
    }

    fn skip_string_run(&mut self) {
        while matches!(self.peek(), Some(Tok::Str(_) | Tok::Bytes)) {
            self.pos += 1;
        }
    }

    /// Calls, subscripts and attribute access make an expression non-literal.
    fn check_no_trailer(&self) -> Result<(), PyError> {
        if self.at_op("(") || self.at_op("[") {
            Err(PyError::Invalid)
        } else {
            Ok(())
        }
    }

    fn dict_or_set(&mut self, depth: usize) -> Result<PyLit, PyError> {
        if self.eat_op("}") {
            return Ok(PyLit::Dict(Vec::new()));
        }
        let first = self.expression(depth + 1)?;
        if !self.eat_op(":") {
            // A set display: syntactically a literal, but json.dumps rejects
            // sets. Validate the rest so syntax errors stay `Invalid`.
            let mut hashable = is_hashable(&first);
            while self.eat_op(",") {
                if self.at_op("}") {
                    break;
                }
                hashable &= is_hashable(&self.expression(depth + 1)?);
            }
            if !self.eat_op("}") {
                return Err(PyError::Invalid);
            }
            self.check_no_trailer()?;
            return Err(if hashable {
                PyError::Uncaught
            } else {
                PyError::Invalid
            });
        }
        let mut entries: Vec<(PyLit, PyLit)> = Vec::new();
        let mut key = first;
        loop {
            let value = self.expression(depth + 1)?;
            insert_python_key(&mut entries, key, value)?;
            if !self.eat_op(",") || self.at_op("}") {
                break;
            }
            key = self.expression(depth + 1)?;
            if !self.eat_op(":") {
                return Err(PyError::Invalid);
            }
        }
        if !self.eat_op("}") {
            return Err(PyError::Invalid);
        }
        Ok(PyLit::Dict(entries))
    }
}

fn is_hashable(v: &PyLit) -> bool {
    match v {
        PyLit::List(_) | PyLit::Dict(_) => false,
        PyLit::Tuple(items) => items.iter().all(is_hashable),
        _ => true,
    }
}

/// Python's `dict` insertion: equal keys (by `==`, so `1 == 1.0 == True`)
/// keep the first key and take the last value. Unhashable keys raise
/// `TypeError` (caught by the reference parser).
fn insert_python_key(
    entries: &mut Vec<(PyLit, PyLit)>,
    key: PyLit,
    value: PyLit,
) -> Result<(), PyError> {
    if !is_hashable(&key) {
        return Err(PyError::Invalid);
    }
    for (existing, slot) in entries.iter_mut() {
        if py_eq(existing, &key) {
            *slot = value;
            return Ok(());
        }
    }
    entries.push((key, value));
    Ok(())
}

/// Numeric value used for cross-type equality.
enum Num {
    Int(BigInt),
    Float(f64),
}

fn as_num(v: &PyLit) -> Option<Num> {
    match v {
        PyLit::Bool(b) => Some(Num::Int(BigInt::from_digits(
            false,
            if *b { "1" } else { "0" },
        ))),
        PyLit::Int(i) => Some(Num::Int(i.clone())),
        PyLit::Float(f) => Some(Num::Float(*f)),
        _ => None,
    }
}

fn py_eq(a: &PyLit, b: &PyLit) -> bool {
    if let (Some(x), Some(y)) = (as_num(a), as_num(b)) {
        return match (x, y) {
            (Num::Int(x), Num::Int(y)) => x == y,
            (Num::Float(x), Num::Float(y)) => x == y,
            (Num::Int(i), Num::Float(f)) | (Num::Float(f), Num::Int(i)) => {
                f.is_finite() && f.trunc() == f && BigInt::from_integral_f64(f) == i
            }
        };
    }
    match (a, b) {
        (PyLit::None, PyLit::None) => true,
        (PyLit::Str(x), PyLit::Str(y)) => x == y,
        (PyLit::Tuple(x), PyLit::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> PyLit {
        literal_eval(s).unwrap_or_else(|e| panic!("{s:?}: {e:?}"))
    }

    #[test]
    fn accepts_python_literals() {
        assert_eq!(ok("True"), PyLit::Bool(true));
        assert_eq!(ok("  -5"), PyLit::Int(BigInt::from_digits(true, "5")));
        assert_eq!(ok("-(5)"), PyLit::Int(BigInt::from_digits(true, "5")));
        assert_eq!(ok("0x_ff"), PyLit::Int(BigInt::from_digits(false, "255")));
        assert_eq!(ok("1_000.5"), PyLit::Float(1000.5));
        assert_eq!(ok("'a' \"b\" r'\\c'"), PyLit::Str("ab\\c".into()));
        assert_eq!(
            ok("'\\x41\\u00e9\\101\\q'"),
            PyLit::Str("Aé A\\q".replace(' ', ""))
        );
        assert_eq!(ok("1, 2"), PyLit::Tuple(vec![ok("1"), ok("2")]));
        assert_eq!(ok("[1,\n 2,  # c\n]"), PyLit::List(vec![ok("1"), ok("2")]));
        assert_eq!(
            ok("{1: 'a', True: 'b', 1.0: 'c', '1': 'd'}"),
            PyLit::Dict(vec![(ok("1"), ok("'c'")), (ok("'1'"), ok("'d'"))])
        );
        assert_eq!(ok("\n\n00"), PyLit::Int(BigInt::from_digits(false, "0")));
    }

    #[test]
    fn rejects_non_literals() {
        for s in [
            "",
            "x",
            "1 +",
            "1+2",
            "--1",
            "-True",
            "[1][0]",
            "f(1)",
            "01",
            "1_",
            "1__0",
            "0x",
            "1a",
            "'unterminated",
            "'a\nb'",
            "\n 1",
            "1\n2",
            "{**a}",
            "[*a]",
            "lambda: 1",
            "'\\N{BULLET}'",
            "f'x'",
            "'\\x4'",
            "(1",
            "1)",
            "1\\",
        ] {
            assert_eq!(literal_eval(s), Err(PyError::Invalid), "{s:?}");
        }
        for s in ["b'x'", "1j", "-1j", "{1, 2}", "set()", "...", "'\\ud800'"] {
            assert_eq!(literal_eval(s), Err(PyError::Uncaught), "{s:?}");
        }
    }
}
