// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The value a constant stands for (`Constant.typed_value`), and how Python
//! writes one into a message.

use crate::dates::Stamp;
use crate::pyint::PyInt;
use crate::pyrepr::{float_repr, str_repr};

/// What `Constant.typed_value` returns.
#[derive(Clone, Debug, PartialEq)]
pub enum Typed {
    None,
    Bool(bool),
    Int(PyInt),
    Float(f64),
    Str(String),
    Time(Stamp),
    List(Vec<Typed>),
}

impl Typed {
    /// `str(value)`.
    pub fn py_str(&self) -> String {
        match self {
            Typed::Str(s) => s.clone(),
            Typed::Time(t) => t.display(),
            other => other.py_repr(),
        }
    }

    /// `repr(value)`.
    pub fn py_repr(&self) -> String {
        match self {
            Typed::None => "None".to_string(),
            Typed::Bool(true) => "True".to_string(),
            Typed::Bool(false) => "False".to_string(),
            Typed::Int(i) => i.to_string(),
            Typed::Float(f) => float_repr(*f),
            Typed::Str(s) => str_repr(s),
            Typed::Time(t) => timestamp_repr(t),
            Typed::List(items) => format!(
                "[{}]",
                items
                    .iter()
                    .map(Typed::py_repr)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// `repr(pd.Timestamp)`: the year unpadded, the zone as `+HHMM` and named.
fn timestamp_repr(t: &Stamp) -> String {
    let mut stamp = format!(
        "{}-{:02}-{:02} {:02}:{:02}:{:02}",
        t.year, t.month, t.day, t.hour, t.minute, t.second
    );
    if t.microsecond != 0 {
        stamp.push_str(&format!(".{:06}", t.microsecond));
    }
    match t.offset {
        None => format!("Timestamp('{stamp}')"),
        Some(offset) => {
            let sign = if offset < 0 { '-' } else { '+' };
            let total = offset.unsigned_abs();
            let (h, m) = (total / 3600, total % 3600 / 60);
            let name = if offset == 0 {
                "UTC".to_string()
            } else {
                format!("UTC{sign}{h:02}:{m:02}")
            };
            format!("Timestamp('{stamp}{sign}{h:02}{m:02}', tz='{name}')")
        }
    }
}

/// `text.encode('latin-1', 'backslashreplace').decode('unicode-escape')`,
/// which reads a string constant's escapes.
///
/// A `\u` escape of a surrogate gives Python a lone surrogate, which Rust
/// text cannot hold: a pair is joined into the character it encodes, and a
/// lone one becomes U+FFFD.
pub fn decode_escapes(text: &str) -> Result<String, String> {
    let chars: Vec<char> = text.chars().collect();
    let mut units: Vec<u32> = Vec::with_capacity(chars.len());
    let mut i = 0;
    let hex = |slice: &[char], n: usize, what: &str| -> Result<u32, String> {
        if slice.len() < n || !slice[..n].iter().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!("truncated {what} escape"));
        }
        let digits: String = slice[..n].iter().collect();
        u32::from_str_radix(&digits, 16).map_err(|e| e.to_string())
    };
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' {
            units.push(c as u32);
            i += 1;
            continue;
        }
        let Some(&next) = chars.get(i + 1) else {
            return Err("\\ at end of string".to_string());
        };
        i += 2;
        match next {
            '\n' => {}
            '\\' | '\'' | '"' => units.push(next as u32),
            'b' => units.push(0x08),
            'f' => units.push(0x0c),
            't' => units.push(0x09),
            'n' => units.push(0x0a),
            'r' => units.push(0x0d),
            'v' => units.push(0x0b),
            'a' => units.push(0x07),
            '0'..='7' => {
                let mut value = next as u32 - '0' as u32;
                for _ in 0..2 {
                    match chars.get(i) {
                        Some(&d @ '0'..='7') => {
                            value = value * 8 + (d as u32 - '0' as u32);
                            i += 1;
                        }
                        _ => break,
                    }
                }
                units.push(value);
            }
            'x' => {
                units.push(hex(&chars[i..], 2, "\\xXX")?);
                i += 2;
            }
            'u' => {
                units.push(hex(&chars[i..], 4, "\\uXXXX")?);
                i += 4;
            }
            'U' => {
                let value = hex(&chars[i..], 8, "\\UXXXXXXXX")?;
                if value > 0x10FFFF {
                    return Err("illegal Unicode character".to_string());
                }
                units.push(value);
                i += 8;
            }
            'N' => return Err("malformed \\N character escape".to_string()),
            other if (other as u32) > 0xFF => {
                // The character after the backslash was itself written as an
                // escape by `backslashreplace`, whose backslash this one
                // escapes.
                units.push('\\' as u32);
                let code = other as u32;
                let written = if code > 0xFFFF {
                    format!("U{code:08x}")
                } else {
                    format!("u{code:04x}")
                };
                units.extend(written.chars().map(|c| c as u32));
            }
            other => {
                units.push('\\' as u32);
                units.push(other as u32);
            }
        }
    }
    let mut out = String::with_capacity(units.len());
    let mut k = 0;
    while k < units.len() {
        let u = units[k];
        if (0xD800..0xDC00).contains(&u)
            && let Some(&low) = units.get(k + 1)
            && (0xDC00..0xE000).contains(&low)
        {
            let joined = 0x10000 + ((u - 0xD800) << 10) + (low - 0xDC00);
            out.push(char::from_u32(joined).unwrap_or('\u{fffd}'));
            k += 2;
            continue;
        }
        out.push(char::from_u32(u).unwrap_or('\u{fffd}'));
        k += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_decode_as_python_does() {
        assert_eq!(decode_escapes("a\\\\b").unwrap(), "a\\b");
        assert_eq!(decode_escapes("x\\u0041").unwrap(), "xA");
        assert_eq!(decode_escapes("\\/").unwrap(), "\\/");
        assert_eq!(decode_escapes("é\\u00e9").unwrap(), "éé");
        assert_eq!(decode_escapes("\\€").unwrap(), "\\u20ac");
        assert!(decode_escapes("a\\").is_err());
        assert!(decode_escapes("\\u00").is_err());
    }

    #[test]
    fn values_print_as_python_does() {
        let list = Typed::List(vec![
            Typed::Float(1.5),
            Typed::Str("a".into()),
            Typed::None,
            Typed::Bool(true),
        ]);
        assert_eq!(list.py_str(), "[1.5, 'a', None, True]");
    }
}
