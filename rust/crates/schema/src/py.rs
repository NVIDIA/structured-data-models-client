// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What Python's `str` methods, `int()` and `float()` answer about text, and
//! how Python writes values in a message.

pub use sdm_pql::pyrepr::{float_repr, str_repr};

/// `str.isspace()` of one character.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.isdigit()` of one character.
pub fn is_digit(c: char) -> bool {
    c.is_ascii_digit() || (!c.is_ascii() && c.is_numeric())
}

/// `str.isalpha()` of one character.
pub fn is_alpha(c: char) -> bool {
    c.is_alphabetic()
}

/// `text.isdigit()`: non-empty and every character a digit.
pub fn all_digits(text: &str) -> bool {
    !text.is_empty() && text.chars().all(is_digit)
}

/// `text.strip()`.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// The value of an ASCII digit run with Python's `int()` rules: optional
/// whitespace around, an optional sign, digits with single underscores
/// between them. `None` where `int()` raises, or the value overflows.
pub fn int(text: &str) -> Option<i64> {
    let body = strip(text);
    let (negative, digits) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    let mut value: i64 = 0;
    let mut seen = false;
    let mut underscore = false;
    for c in digits.chars() {
        if c == '_' {
            if !seen || underscore {
                return None;
            }
            underscore = true;
            continue;
        }
        let d = c.to_digit(10)?;
        if !c.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(i64::from(d))?;
        seen = true;
        underscore = false;
    }
    if !seen || underscore {
        return None;
    }
    Some(if negative { -value } else { value })
}

/// Whether Python's `float(text)` accepts `text`.
pub fn is_float(text: &str) -> bool {
    float(text).is_some()
}

/// Python's `float(text)`, or `None` where it raises.
pub fn float(text: &str) -> Option<f64> {
    let body = strip(text);
    let (sign, rest) = match body.as_bytes().first() {
        Some(b'-') => ("-", &body[1..]),
        Some(b'+') => ("", &body[1..]),
        _ => ("", body),
    };
    let lower = rest.to_ascii_lowercase();
    if lower == "inf" || lower == "infinity" {
        return Some(if sign == "-" {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if lower == "nan" {
        return Some(f64::NAN);
    }
    // digitpart ['.' [digitpart]] | '.' digitpart, then [e [sign] digitpart]
    let chars: Vec<char> = rest.chars().collect();
    let mut i = 0;
    let mut cleaned = String::from(sign);
    let digit_part = |i: &mut usize, out: &mut String| -> bool {
        let start = *i;
        let mut last_underscore = false;
        while *i < chars.len() {
            let c = chars[*i];
            if let Some(d) = c.to_digit(10).filter(|_| is_digit(c)) {
                out.push(char::from_digit(d, 10).unwrap_or('0'));
                last_underscore = false;
            } else if c == '_' && *i > start && !last_underscore {
                last_underscore = true;
            } else {
                break;
            }
            *i += 1;
        }
        if last_underscore {
            return false;
        }
        *i > start
    };
    let whole = digit_part(&mut i, &mut cleaned);
    if whole && i < chars.len() && chars[i] == '_' {
        return None;
    }
    let mut fraction = false;
    if i < chars.len() && chars[i] == '.' {
        cleaned.push('.');
        i += 1;
        let before = i;
        let ok = digit_part(&mut i, &mut cleaned);
        if !ok && i != before {
            return None;
        }
        fraction = ok;
        if !ok && i < chars.len() && chars[i] == '_' {
            return None;
        }
    }
    if !whole && !fraction {
        return None;
    }
    if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
        cleaned.push('e');
        i += 1;
        if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
            cleaned.push(chars[i]);
            i += 1;
        }
        if !digit_part(&mut i, &mut cleaned) {
            return None;
        }
    }
    if i != chars.len() {
        return None;
    }
    if cleaned.ends_with('.') {
        cleaned.push('0');
    }
    if cleaned.starts_with('.') || cleaned.starts_with("-.") {
        cleaned = cleaned.replacen('.', "0.", 1);
    }
    cleaned.parse::<f64>().ok()
}

/// `text.zfill(width)`.
pub fn zfill(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len >= width {
        return text.to_string();
    }
    let pad = "0".repeat(width - len);
    match text.chars().next() {
        Some(sign @ ('+' | '-')) => format!("{sign}{pad}{}", &text[1..]),
        _ => format!("{pad}{text}"),
    }
}

/// An integer with thousands separators, as `f'{n:,}'`.
pub fn thousands(n: i64) -> String {
    let digits = n.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    if n < 0 { format!("-{out}") } else { out }
}

/// `repr` of a list of strings, as Python writes `['a', 'b']`.
pub fn str_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| str_repr(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_float_grammar() {
        for good in [
            "1", " 1.5 ", "-2", ".5", "5.", "1e5", "1_000", "inf", "-NaN", "1E-3",
        ] {
            assert!(is_float(good), "{good}");
        }
        for bad in ["", "-", ".", "e5", "1_", "_1", "1__0", "1e", "abc", "1.2.3"] {
            assert!(!is_float(bad), "{bad}");
        }
        assert_eq!(int(" 12 "), Some(12));
        assert_eq!(int("+1_000"), Some(1000));
        assert_eq!(int("1e3"), None);
        assert_eq!(zfill("-5", 3), "-05");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }
}
