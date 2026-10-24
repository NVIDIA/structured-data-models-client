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

use crate::conversions::{Clock, date_iso, datetime_iso};
use crate::printable::NOT_PRINTABLE;

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
    /// A list (`true`) or a `dict` (`false`) inside itself, which Python
    /// writes as `[...]` or `{...}`.
    Recursive(bool),
}

/// A write past a [`Sink`]'s bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

/// Text written up to a bound in characters: a write that would pass it
/// fails, and nothing past it is ever kept.
pub struct Sink {
    out: String,
    room: usize,
}

impl Sink {
    /// A sink for at most `room` characters.
    pub fn new(room: usize) -> Sink {
        Sink {
            out: String::new(),
            room,
        }
    }

    pub fn push(&mut self, text: &str) -> Result<(), Full> {
        let count = text.chars().count();
        self.room = self.room.checked_sub(count).ok_or(Full)?;
        self.out.push_str(text);
        Ok(())
    }

    pub fn push_char(&mut self, c: char) -> Result<(), Full> {
        self.room = self.room.checked_sub(1).ok_or(Full)?;
        self.out.push(c);
        Ok(())
    }

    pub fn into_string(self) -> String {
        self.out
    }
}

impl Shown {
    /// Python's `repr`: `'text'`, `True`, `None`, `[1, 'a']`,
    /// `{'name': 'x'}`, `datetime.date(2024, 1, 1)`.
    pub fn repr(&self) -> String {
        self.repr_within(usize::MAX).unwrap_or_default()
    }

    /// Python's `str`: a text as itself, a date or time in ISO form with a
    /// space between them (`2024-01-01 10:00:00`), anything else its
    /// [`Shown::repr`].
    pub fn text(&self) -> String {
        self.text_within(usize::MAX).unwrap_or_default()
    }

    /// [`Shown::repr`], unless it is longer than `limit` characters, in
    /// which case no more than `limit` is written.
    pub fn repr_within(&self, limit: usize) -> Option<String> {
        let mut sink = Sink::new(limit);
        self.write_repr(&mut sink).ok()?;
        Some(sink.into_string())
    }

    /// [`Shown::text`], unless it is longer than `limit` characters, in
    /// which case no more than `limit` is written.
    pub fn text_within(&self, limit: usize) -> Option<String> {
        let mut sink = Sink::new(limit);
        self.write_text(&mut sink).ok()?;
        Some(sink.into_string())
    }

    fn write_repr(&self, sink: &mut Sink) -> Result<(), Full> {
        match self {
            Shown::None => sink.push("None"),
            Shown::Bool(true) => sink.push("True"),
            Shown::Bool(false) => sink.push("False"),
            Shown::Int(n) => sink.push(n),
            Shown::Float(x) => sink.push(&float_repr(*x)),
            Shown::Text(text) => write_text_repr(text, sink),
            Shown::Date { year, month, day } => {
                sink.push(&format!("datetime.date({year}, {month}, {day})"))
            }
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
                sink.push(&format!("datetime.datetime({}{zone})", fields.join(", ")))
            }
            Shown::List(items) => {
                sink.push_char('[')?;
                for (at, item) in items.iter().enumerate() {
                    if at > 0 {
                        sink.push(", ")?;
                    }
                    item.write_repr(sink)?;
                }
                sink.push_char(']')
            }
            Shown::Dict(pairs) => {
                sink.push_char('{')?;
                for (at, (key, value)) in pairs.iter().enumerate() {
                    if at > 0 {
                        sink.push(", ")?;
                    }
                    key.write_repr(sink)?;
                    sink.push(": ")?;
                    value.write_repr(sink)?;
                }
                sink.push_char('}')
            }
            Shown::Recursive(true) => sink.push("[...]"),
            Shown::Recursive(false) => sink.push("{...}"),
        }
    }

    fn write_text(&self, sink: &mut Sink) -> Result<(), Full> {
        match self {
            Shown::Text(text) => sink.push(text),
            Shown::Date { year, month, day } => sink.push(&date_iso(*year, *month, *day)),
            Shown::DateTime {
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                offset,
            } => sink.push(&datetime_iso(
                (*year, *month, *day),
                Clock {
                    hour: *hour,
                    minute: *minute,
                    second: *second,
                    microsecond: *microsecond,
                },
                offset.map(|seconds| i64::from(seconds) * 1_000_000),
                ' ',
            )),
            _ => self.write_repr(sink),
        }
    }
}

/// Python's `repr` of a text: in single quotes unless it holds one and no
/// double quote, a backslash and that quote escaped, `\t`, `\n` and `\r` as
/// such, and each character Python does not print as `\xhh`, `\uhhhh` or
/// `\Uhhhhhhhh`.
pub fn text_repr(text: &str) -> String {
    let mut sink = Sink::new(usize::MAX);
    let _ = write_text_repr(text, &mut sink);
    sink.into_string()
}

fn write_text_repr(text: &str, sink: &mut Sink) -> Result<(), Full> {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    sink.push_char(quote)?;
    for c in text.chars() {
        match c {
            '\\' => sink.push("\\\\")?,
            '\t' => sink.push("\\t")?,
            '\n' => sink.push("\\n")?,
            '\r' => sink.push("\\r")?,
            _ if c == quote => {
                sink.push_char('\\')?;
                sink.push_char(c)?;
            }
            _ if printable(c) => sink.push_char(c)?,
            _ => {
                let code = u32::from(c);
                if code < 0x100 {
                    sink.push(&format!("\\x{code:02x}"))?;
                } else if code < 0x10000 {
                    sink.push(&format!("\\u{code:04x}"))?;
                } else {
                    sink.push(&format!("\\U{code:08x}"))?;
                }
            }
        }
    }
    sink.push_char(quote)
}

/// Whether Python prints `c` as itself in a `repr` (`str.isprintable`),
/// by Python 3.12's own table ([`crate::printable`]).
fn printable(c: char) -> bool {
    let code = u32::from(c);
    let at = NOT_PRINTABLE.partition_point(|&(first, _)| first <= code);
    at == 0 || NOT_PRINTABLE[at - 1].1 < code
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
