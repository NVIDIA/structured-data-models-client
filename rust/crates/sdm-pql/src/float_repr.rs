// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The text Python gives a float, the shortest that reads back to it, in
//! Python's layout: one of the emulation modules this crate may hold
//! (`scripts/rust_core/emulation-allowlist.txt`, Phase 1 decisions A4).
//! The reference writes a float constant built from a value with it
//! (`Constant.from_value`); here it writes a float literal's value into
//! messages, and decides no validity (Phase 1 decisions, E9).
//!
//! From T3.5 it also writes the other values the old client's messages
//! quote (a view definition's YAML values and the pieces of an expression):
//! [`Shown`], a value as Python held it, with its `repr` and `str`, and
//! [`text_repr`], a text's `repr`. They decide nothing either: `sdm-core`
//! words a view's conversion with them.

/// The text Python's `repr` gives `x`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // As many digits as the shortest that round-trip, as Python's repr
    // uses. Of the decimals of that many digits that round-trip, Python
    // writes the nearest, a tie to the even one, which exact formatting
    // gives where it round-trips; the shortest formatting may break a tie
    // the other way. Next to a power of two the nearest may not round-trip,
    // and the shortest is the one that does.
    let shortest = format!("{:e}", x.abs());
    let precision = shortest
        .split_once('e')
        .map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count())
        .saturating_sub(1);
    let nearest = format!("{:.*e}", precision, x.abs());
    let sci = if nearest.parse::<f64>() == Ok(x.abs()) {
        nearest
    } else {
        shortest
    };
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = match digits.trim_end_matches('0') {
        "" => "0".to_string(),
        kept => kept.to_string(),
    };
    // Python's decpt: the value is 0.DIGITS * 10**decpt.
    let decpt = exponent + 1;
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let mut text = digits[..1].to_string();
        if n > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        let esign = if e < 0 { '-' } else { '+' };
        return format!("{sign}{text}e{esign}{:02}", e.abs());
    }
    let text = if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt >= n {
        format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    format!("{sign}{text}")
}

/// A value as Python held it, to be written as Python writes it: the
/// values PyYAML's `safe_load` makes, which the old client's messages quote.
#[derive(Clone, Debug, PartialEq)]
pub enum Shown {
    None,
    Bool(bool),
    /// An integer, as its decimal digits, a `-` before them if negative.
    Int(String),
    Float(f64),
    Text(String),
    Bytes(Vec<u8>),
    Date {
        year: i32,
        month: u8,
        day: u8,
    },
    /// A `datetime`, with its offset from UTC in seconds if it has one.
    DateTime {
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        microsecond: u32,
        offset: Option<i32>,
    },
    List(Vec<Shown>),
    /// A `dict`'s entries in order.
    Dict(Vec<(Shown, Shown)>),
    /// A `set`'s members, in the order they are to be written.
    Set(Vec<Shown>),
    Tuple(Vec<Shown>),
    /// A list (`true`) or a `dict` (`false`) inside itself, which Python
    /// writes as `[...]` or `{...}`.
    Recursive(bool),
}

impl Shown {
    /// Python's `repr`: `'text'`, `True`, `None`, `[1, 'a']`,
    /// `{'name': 'x'}`, `datetime.date(2024, 1, 1)`.
    pub fn repr(&self) -> String {
        match self {
            Shown::None => "None".into(),
            Shown::Bool(true) => "True".into(),
            Shown::Bool(false) => "False".into(),
            Shown::Int(n) => n.clone(),
            Shown::Float(x) => float_repr(*x),
            Shown::Text(text) => text_repr(text),
            Shown::Bytes(bytes) => bytes_repr(bytes),
            Shown::Date { year, month, day } => format!("datetime.date({year}, {month}, {day})"),
            Shown::DateTime {
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                offset,
            } => {
                let mut fields = vec![
                    year.to_string(),
                    month.to_string(),
                    day.to_string(),
                    hour.to_string(),
                    minute.to_string(),
                ];
                if *second > 0 || *microsecond > 0 {
                    fields.push(second.to_string());
                }
                if *microsecond > 0 {
                    fields.push(microsecond.to_string());
                }
                let zone = match offset {
                    None => String::new(),
                    Some(0) => ", tzinfo=datetime.timezone.utc".into(),
                    Some(seconds) => {
                        let (days, rest) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
                        let delta = match (days, rest) {
                            (0, s) => format!("seconds={s}"),
                            (d, 0) => format!("days={d}"),
                            (d, s) => format!("days={d}, seconds={s}"),
                        };
                        format!(", tzinfo=datetime.timezone(datetime.timedelta({delta}))")
                    }
                };
                format!("datetime.datetime({}{zone})", fields.join(", "))
            }
            Shown::List(items) => format!(
                "[{}]",
                items.iter().map(Shown::repr).collect::<Vec<_>>().join(", ")
            ),
            Shown::Dict(pairs) => format!(
                "{{{}}}",
                pairs
                    .iter()
                    .map(|(key, value)| format!("{}: {}", key.repr(), value.repr()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Shown::Set(members) if members.is_empty() => "set()".into(),
            Shown::Set(members) => format!(
                "{{{}}}",
                members
                    .iter()
                    .map(Shown::repr)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Shown::Tuple(items) if items.len() == 1 => format!("({},)", items[0].repr()),
            Shown::Tuple(items) => format!(
                "({})",
                items.iter().map(Shown::repr).collect::<Vec<_>>().join(", ")
            ),
            Shown::Recursive(true) => "[...]".into(),
            Shown::Recursive(false) => "{...}".into(),
        }
    }

    /// Python's `str`: a text as itself, a date or time in ISO form with a
    /// space between them (`2024-01-01 10:00:00`), anything else its
    /// [`Shown::repr`].
    pub fn text(&self) -> String {
        match self {
            Shown::Text(text) => text.clone(),
            Shown::Date { year, month, day } => format!("{year:04}-{month:02}-{day:02}"),
            Shown::DateTime {
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                offset,
            } => {
                let mut text =
                    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}");
                if *microsecond > 0 {
                    text.push_str(&format!(".{microsecond:06}"));
                }
                if let Some(offset) = offset {
                    let sign = if *offset < 0 { '-' } else { '+' };
                    let seconds = offset.unsigned_abs();
                    text.push_str(&format!(
                        "{sign}{:02}:{:02}",
                        seconds / 3600,
                        seconds / 60 % 60
                    ));
                    if seconds % 60 > 0 {
                        text.push_str(&format!(":{:02}", seconds % 60));
                    }
                }
                text
            }
            _ => self.repr(),
        }
    }
}

fn bytes_repr(bytes: &[u8]) -> String {
    let quote = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        '"'
    } else {
        '\''
    };
    let mut text = format!("b{quote}");
    for &b in bytes {
        match b {
            b'\\' => text.push_str("\\\\"),
            b'\t' => text.push_str("\\t"),
            b'\n' => text.push_str("\\n"),
            b'\r' => text.push_str("\\r"),
            _ if char::from(b) == quote => {
                text.push('\\');
                text.push(quote);
            }
            0x20..=0x7E => text.push(char::from(b)),
            _ => text.push_str(&format!("\\x{b:02x}")),
        }
    }
    text.push(quote);
    text
}

/// Python's `repr` of a text: in single quotes unless it holds one and no
/// double quote, a backslash and that quote escaped, `\t`, `\n` and `\r` as
/// such, and each character Python does not print as `\xhh`, `\uhhhh` or
/// `\Uhhhhhhhh`.
pub fn text_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            _ if c == quote => {
                out.push('\\');
                out.push(c);
            }
            _ if printable(c) => out.push(c),
            _ => {
                let code = u32::from(c);
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
        }
    }
    out.push(quote);
    out
}

/// Whether Python prints `c` as itself in a `repr` (`str.isprintable`):
/// not a control or format character, a separator other than the space,
/// a private-use character, a surrogate or a noncharacter. Unassigned code
/// points, which Python also escapes, are not told apart: the core has no
/// table of Unicode's assignments.
fn printable(c: char) -> bool {
    let code = u32::from(c);
    if code < 0x80 {
        return (0x20..0x7F).contains(&code);
    }
    !(c.is_control()
        || matches!(code,
            0xA0 | 0xAD | 0x1680 | 0x2000..=0x200F | 0x2028..=0x202F | 0x205F..=0x2064
            | 0x2066..=0x206F | 0x3000 | 0xFEFF | 0xFFF9..=0xFFFB | 0x600..=0x605 | 0x61C
            | 0x6DD | 0x70F | 0x890..=0x891 | 0x8E2 | 0x180E | 0x110BD | 0x110CD
            | 0x13430..=0x1343F | 0x1BCA0..=0x1BCA3 | 0x1D173..=0x1D17A | 0xE0001
            | 0xE0020..=0xE007F | 0xE000..=0xF8FF | 0xF0000..=0xFFFFD
            | 0x100000..=0x10FFFD | 0xFDD0..=0xFDEF)
        || code & 0xFFFE == 0xFFFE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        for (x, want) in [
            (1.5, "1.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5e-7, "-2.5e-07"),
            (123.0, "123.0"),
            (0.1, "0.1"),
            (1.2345678901234567e20, "1.2345678901234567e+20"),
            (-0.0, "-0.0"),
            (f64::INFINITY, "inf"),
            // Exactly halfway between two shortest decimals: the even one.
            (f64::from(943.7_f32), "943.7000122070312"),
            (-690_600_950_624_650.0 - 0.25, "-690600950624650.2"),
            (5e-324, "5e-324"),
            // Below a power of two, where the nearest does not round-trip.
            (-5.960_464_477_539_063e-8, "-5.960464477539063e-08"),
            (1e22, "1e+22"),
            // A float32's double, as a float target's classes spell it.
            (f64::from(3.0_f32), "3.0"),
            (f64::from(0.1_f32), "0.10000000149011612"),
            (f64::from(1e20_f32), "1.0000000200408773e+20"),
            (f64::from(1e-7_f32), "1.0000000116860974e-07"),
            (f64::from(1e-4_f32), "9.999999747378752e-05"),
            (f64::from(123_456_789_f32), "123456792.0"),
            (
                f64::from(f32::from_bits(0x0001_16c2)),
                "9.99994610111476e-41",
            ),
            (f64::from(-0.0_f32), "-0.0"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }
}
