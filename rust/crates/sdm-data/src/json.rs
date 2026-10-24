// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON as the reference writes it (its `json.dumps`, compact): a string
//! quoted, ASCII only or with its characters as they are, and a double's
//! number. Postgres's value rules write a JSON value back with it
//! ([`crate::pg_values`]), and a key set bound as text is written with it
//! ([`crate::key_text`]).

/// A string as the reference's writer quotes one, ASCII only: `"` and `\`
/// escaped, the named control characters by name, every other code unit
/// outside space to `~` as `\u` and four lowercase hexadecimal digits (one
/// past the first plane as its two surrogates, a lone surrogate as itself).
pub(crate) fn quoted(units: &[u16], out: &mut String) {
    out.push('"');
    for &unit in units {
        match unit {
            0x22 => out.push_str("\\\""),
            0x5C => out.push_str("\\\\"),
            0x0A => out.push_str("\\n"),
            0x0D => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            0x08 => out.push_str("\\b"),
            0x0C => out.push_str("\\f"),
            0x20..=0x7E => out.push(char::from(u8::try_from(unit).unwrap_or(b'?'))),
            _ => out.push_str(&format!("\\u{unit:04x}")),
        }
    }
    out.push('"');
}

/// `text` as a JSON string, as the old client's `json.dumps` writes one:
/// ASCII only ([`quoted`]), its default; else with a quote, a backslash and
/// the five control characters JSON names escaped by name, every other
/// control character as `\u` and four lowercase hexadecimal digits, and
/// every other character as it is (`ensure_ascii=False`).
pub(crate) fn string(text: &str, ascii: bool) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    if ascii {
        quoted(&text.encode_utf16().collect::<Vec<_>>(), &mut out);
        return out;
    }
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A double as a JSON number that reads back as it, its shortest digits;
/// a NaN and an infinity as the old client's JSON writes them (`NaN`,
/// `Infinity`, `-Infinity`), which the warehouse's JSON reader takes.
pub(crate) fn number(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_owned()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        format!("{x:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::{number, string};

    #[test]
    fn a_string_is_escaped_as_the_old_client_s_json_escapes_it() {
        assert_eq!(string("a\"b\\c\n\u{1}", true), r#""a\"b\\c\n\u0001""#);
        assert_eq!(string("é😀\u{7f}", true), r#""\u00e9\ud83d\ude00\u007f""#);
        assert_eq!(string("é😀\u{7f}", false), "\"é😀\u{7f}\"");
    }

    #[test]
    fn a_double_reads_back_as_itself() {
        for x in [1.5, 0.1, -2.0, 1e16, 1e-7, 5e-324, f64::MAX, -0.0] {
            assert_eq!(number(x).parse::<f64>().unwrap().to_bits(), x.to_bits());
        }
        assert_eq!(number(f64::NAN), "NaN");
        assert_eq!(number(f64::NEG_INFINITY), "-Infinity");
    }
}
