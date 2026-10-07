//! Reproductions of the CPython built-ins the reference tool-call parser
//! applies to model output: `str.strip()`, `int(str)`, `float(str)`,
//! `html.unescape`, and `ast.literal_eval`.
//!
//! Each function returns `Err` exactly where CPython raises. Callers decide
//! what an error means (usually "degenerate to the string").

mod literal;
mod tables;

pub use literal::{PyLit, literal_eval};

use crate::json::{BigInt, PY_MAX_STR_DIGITS};

/// A CPython exception, reduced to whether the reference parser catches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PyError {
    /// `ValueError`, `SyntaxError` or `TypeError` from the conversion itself:
    /// the reference parser catches these and falls back.
    Invalid,
    /// An exception the reference parser does not catch (`OverflowError`,
    /// `RecursionError`, a `ValueError` raised inside `html.unescape`, a
    /// `TypeError` from `json.dumps`): the reference request fails.
    Uncaught,
}

/// `str.isspace()` for one character.
pub fn is_py_space(c: char) -> bool {
    let cp = c as u32;
    tables::PY_WHITESPACE
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// `str.strip()` with no arguments.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// The value of a Unicode decimal digit (`unicodedata.decimal`).
pub fn py_decimal(c: char) -> Option<u8> {
    let cp = c as u32;
    let i = tables::PY_DECIMAL_DIGITS
        .binary_search_by(|&(lo, hi, _)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .ok()?;
    let (lo, _, first) = tables::PY_DECIMAL_DIGITS[i];
    Some(first + (cp - lo) as u8)
}

/// CPython's `_PyUnicode_TransformDecimalAndSpaceToASCII`: whitespace becomes
/// a space, decimal digits become ASCII digits, other non-ASCII characters
/// become `?` (which no numeric grammar accepts).
fn transform_decimal_and_space(s: &str) -> String {
    s.chars()
        .map(|c| {
            if is_py_space(c) {
                ' '
            } else if let Some(d) = py_decimal(c) {
                (b'0' + d) as char
            } else if c.is_ascii() {
                c
            } else {
                '?'
            }
        })
        .collect()
}

fn ascii_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c')
}

/// `int(s)` for a string `s` (base 10).
pub fn py_int(s: &str) -> Result<BigInt, PyError> {
    let t = transform_decimal_and_space(s);
    let b = t.as_bytes();
    let mut i = 0;
    while i < b.len() && ascii_space(b[i]) {
        i += 1;
    }
    let mut negative = false;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        negative = b[i] == b'-';
        i += 1;
    }
    let mut digits = String::new();
    let mut prev_digit = false;
    while i < b.len() {
        match b[i] {
            d @ b'0'..=b'9' => {
                digits.push(d as char);
                prev_digit = true;
            }
            b'_' if prev_digit && b.get(i + 1).is_some_and(u8::is_ascii_digit) => {
                prev_digit = false;
            }
            _ => break,
        }
        i += 1;
    }
    while i < b.len() && ascii_space(b[i]) {
        i += 1;
    }
    if digits.is_empty() || i != b.len() {
        return Err(PyError::Invalid);
    }
    if digits.len() > PY_MAX_STR_DIGITS {
        return Err(PyError::Invalid);
    }
    Ok(BigInt::from_digits(negative, &digits))
}

/// `float(s)` for a string `s`.
pub fn py_float(s: &str) -> Result<f64, PyError> {
    let t = transform_decimal_and_space(s);
    let trimmed = t.trim_matches(|c: char| c.is_ascii() && ascii_space(c as u8));
    // Underscores are allowed only between two ASCII digits.
    let mut cleaned = String::with_capacity(trimmed.len());
    let bytes = trimmed.as_bytes();
    for (i, &c) in bytes.iter().enumerate() {
        if c == b'_' {
            let before = i > 0 && bytes[i - 1].is_ascii_digit();
            let after = bytes.get(i + 1).is_some_and(u8::is_ascii_digit);
            if !(before && after) {
                return Err(PyError::Invalid);
            }
        } else {
            cleaned.push(c as char);
        }
    }
    let (negative, body) = match cleaned.as_bytes().first() {
        Some(b'-') => (true, &cleaned[1..]),
        Some(b'+') => (false, &cleaned[1..]),
        _ => (false, cleaned.as_str()),
    };
    let lower = body.to_ascii_lowercase();
    let value = if lower == "inf" || lower == "infinity" {
        f64::INFINITY
    } else if lower == "nan" {
        f64::NAN
    } else {
        if !is_decimal_float(body.as_bytes()) {
            return Err(PyError::Invalid);
        }
        body.parse::<f64>().map_err(|_| PyError::Invalid)?
    };
    Ok(if negative { -value } else { value })
}

/// `digits [. [digits]] [e [sign] digits]` or `. digits [exponent]`.
fn is_decimal_float(b: &[u8]) -> bool {
    let mut i = 0;
    let int_digits = b.iter().take_while(|c| c.is_ascii_digit()).count();
    i += int_digits;
    let mut frac_digits = 0;
    if b.get(i) == Some(&b'.') {
        i += 1;
        frac_digits = b[i..].iter().take_while(|c| c.is_ascii_digit()).count();
        i += frac_digits;
    }
    if int_digits + frac_digits == 0 {
        return false;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let exp_digits = b[i..].iter().take_while(|c| c.is_ascii_digit()).count();
        if exp_digits == 0 {
            return false;
        }
        i += exp_digits;
    }
    i == b.len()
}

fn html5_entity(name: &str) -> Option<&'static str> {
    tables::HTML5_ENTITIES
        .binary_search_by(|(n, _)| (*n).cmp(name))
        .ok()
        .map(|i| tables::HTML5_ENTITIES[i].1)
}

/// `html.unescape(s)`.
///
/// Fails (as an uncaught `ValueError` in CPython) only for a decimal
/// character reference longer than `int()`'s digit limit.
pub fn html_unescape(s: &str) -> Result<String, PyError> {
    if !s.contains('&') {
        return Ok(s.to_string());
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        match charref(after)? {
            Some((replacement, consumed)) => {
                out.push_str(&replacement);
                rest = &after[consumed..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    Ok(out)
}

/// Matches `#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?` at the
/// start of `s` and returns the replacement and the bytes consumed.
fn charref(s: &str) -> Result<Option<(String, usize)>, PyError> {
    let b = s.as_bytes();
    if b.first() == Some(&b'#') {
        let (hex, start) = match b.get(1) {
            Some(b'x' | b'X') if b.get(2).is_some_and(u8::is_ascii_hexdigit) => (true, 2),
            Some(c) if c.is_ascii_digit() => (false, 1),
            _ => return Ok(None),
        };
        let digits = b[start..]
            .iter()
            .take_while(|c| {
                if hex {
                    c.is_ascii_hexdigit()
                } else {
                    c.is_ascii_digit()
                }
            })
            .count();
        let mut end = start + digits;
        if b.get(end) == Some(&b';') {
            end += 1;
        }
        let text = &s[start..start + digits];
        if !hex && text.len() > PY_MAX_STR_DIGITS {
            return Err(PyError::Uncaught);
        }
        // Anything above U+10FFFF maps to U+FFFD, so saturate.
        let trimmed = text.trim_start_matches('0');
        let num = if trimmed.len() > 8 {
            u32::MAX
        } else if trimmed.is_empty() {
            0
        } else {
            u32::from_str_radix(trimmed, if hex { 16 } else { 10 }).unwrap_or(u32::MAX)
        };
        let replacement =
            if let Ok(i) = tables::INVALID_CHARREFS.binary_search_by_key(&num, |e| e.0) {
                tables::INVALID_CHARREFS[i].1.to_string()
            } else if (0xd800..=0xdfff).contains(&num) || num > 0x10ffff {
                "\u{fffd}".to_string()
            } else if tables::INVALID_CODEPOINTS.binary_search(&num).is_ok() {
                String::new()
            } else {
                char::from_u32(num).expect("valid scalar").to_string()
            };
        return Ok(Some((replacement, end)));
    }
    let mut end = 0;
    for (count, c) in s.chars().enumerate() {
        if count == 32 || matches!(c, '\t' | '\n' | '\x0c' | ' ' | '<' | '&' | '#' | ';') {
            break;
        }
        end += c.len_utf8();
    }
    if end == 0 {
        return Ok(None);
    }
    if b.get(end) == Some(&b';') {
        end += 1;
    }
    let name = &s[..end];
    if let Some(value) = html5_entity(name) {
        return Ok(Some((value.to_string(), end)));
    }
    // Longest prefix of at least two characters that is an entity
    // (the legacy entities without a semicolon).
    let boundaries: Vec<usize> = name.char_indices().map(|(i, _)| i).skip(2).collect();
    for &cut in boundaries.iter().rev() {
        if let Some(value) = html5_entity(&name[..cut]) {
            return Ok(Some((format!("{value}{}", &name[cut..]), end)));
        }
    }
    Ok(Some((format!("&{name}"), end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_matches_python() {
        let ok = [
            ("42", "42"),
            ("  -7 ", "-7"),
            ("+0", "0"),
            ("007", "7"),
            ("1_000", "1000"),
            ("٣٤", "34"),
            ("\u{3000}５\u{3000}", "5"),
        ];
        for (s, v) in ok {
            assert_eq!(py_int(s).unwrap().to_string(), v, "{s:?}");
        }
        for s in [
            "", "1.0", "1_", "_1", "1__0", "0x10", "- 5", "1 2", "١a", "+",
        ] {
            assert_eq!(py_int(s), Err(PyError::Invalid), "{s:?}");
        }
    }

    #[test]
    fn float_matches_python() {
        let ok = [
            ("1.5", 1.5),
            (" .5", 0.5),
            ("5.", 5.0),
            ("1e3", 1000.0),
            ("1_0.2_5", 10.25),
            ("-InFiNiTy", f64::NEG_INFINITY),
            ("1e400", f64::INFINITY),
            ("٣.٥", 3.5),
        ];
        for (s, v) in ok {
            assert_eq!(py_float(s).unwrap(), v, "{s:?}");
        }
        assert!(py_float("nan").unwrap().is_nan());
        for s in ["", ".", "1e", "1._5", "e5", "0x1p3", "1,5", "infinite"] {
            assert_eq!(py_float(s), Err(PyError::Invalid), "{s:?}");
        }
    }

    #[test]
    fn html_unescape_matches_python() {
        let cases = [
            ("a &amp; b", "a & b"),
            ("&lt;div&gt;", "<div>"),
            ("&copy=2", "©=2"),
            ("&notit;", "¬it;"),
            ("&#65;&#x42;&#X43", "ABC"),
            ("&#0;", "\u{fffd}"),
            ("&#x80;", "\u{20ac}"),
            ("&#xd800;", "\u{fffd}"),
            ("&#1;", ""),
            ("&#99999999999;", "\u{fffd}"),
            ("&unknown;", "&unknown;"),
            ("& amp", "& amp"),
            ("&#;", "&#;"),
            ("&#xZ", "&#xZ"),
        ];
        for (s, v) in cases {
            assert_eq!(html_unescape(s).unwrap(), v, "{s:?}");
        }
    }
}
