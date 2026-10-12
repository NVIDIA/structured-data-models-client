// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The shapes a date is written in as text, as `date-rules.md` ("The
//! client rules", "Accepted shapes") records them from the reference's
//! `client/core/dates.py`, a port of those two patterns and nothing else
//! (design, T1.25 row; `dependencies.md`):
//!
//! - year first: `YYYY-MM-DD` or `YYYY/MM/DD` (either separator in either
//!   place), two-digit month and day, optionally followed by a space or
//!   `T`, `HH:MM`, optional `:SS` with an optional fraction of any length,
//!   and an optional zone with or without one space before it;
//! - day or month first: `D/M/YY` to `DD/MM/YYYY` with `/` or `-`, a year
//!   of two to four digits, optionally followed by a space or `T`, `H:MM` or
//!   `HH:MM`, optional seconds and fraction, and an optional zone;
//! - a zone is `Z`, `+HH`, `+HHMM`, `+HH:MM` (or with `-`), or a name
//!   starting with a letter (`[A-Za-z][A-Za-z0-9_+/-]*`).
//!
//! A value matches whole, as given; whether surrounding white space is
//! taken off first is the caller's rule. Digits are ASCII digits: the
//! reference's pattern also takes other scripts' digits, which the core's
//! list does not (register DT-15). Which of these values the core reads,
//! and as what, is [`super::text`]'s, by `date-rules.md`'s "The core's text
//! formats": the shapes are the text it recognises as written as a date,
//! the list the part of them it reads.

/// Which field a shape leads with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lead {
    /// `YYYY-MM-DD`: the year.
    Year,
    /// `DD/MM/YY` or `MM/DD/YY`: a day or a month, which the shape does not
    /// say.
    DayOrMonth,
}

/// A zone as a shape writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Zone<'a> {
    /// `Z`.
    Z,
    /// `+HH` or `-HH`: an offset of whole hours.
    Hours { seconds: i32 },
    /// `+HHMM` or `-HHMM`; `minute` the minutes field as written, which may
    /// pass 59.
    Compact { seconds: i32, minute: i32 },
    /// `+HH:MM` or `-HH:MM`.
    Colon { seconds: i32, minute: i32 },
    /// A name, such as `UTC` or `CST`.
    Name(&'a str),
}

impl Zone<'_> {
    /// The offset east of UTC in seconds a numeric zone writes, or `None`
    /// for a name.
    pub fn offset_seconds(&self) -> Option<i32> {
        match self {
            Zone::Z => Some(0),
            Zone::Hours { seconds }
            | Zone::Compact { seconds, .. }
            | Zone::Colon { seconds, .. } => Some(*seconds),
            Zone::Name(_) => None,
        }
    }
}

/// A time of day after a date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeOfDay<'a> {
    /// ` ` or `T`.
    pub separator: char,
    pub hour: &'a str,
    pub minute: &'a str,
    pub second: Option<&'a str>,
    /// The digits after the second's point.
    pub fraction: Option<&'a str>,
    /// The zone, and whether one space comes before it.
    pub zone: Option<(bool, Zone<'a>)>,
}

/// A value in one of the shapes, its fields as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shape<'a> {
    pub lead: Lead,
    /// Year first: the year, month and day; day or month first: the first
    /// field, the second and the year.
    pub fields: [&'a str; 3],
    /// The separators after the first and second fields.
    pub separators: [char; 2],
    pub time: Option<TimeOfDay<'a>>,
}

/// `text` in one of the shapes, or `None`. No text is in both: a year
/// first is four digits, a day or month one or two.
pub fn shape(text: &str) -> Option<Shape<'_>> {
    shaped(text, Lead::Year).or_else(|| shaped(text, Lead::DayOrMonth))
}

fn shaped(text: &str, lead: Lead) -> Option<Shape<'_>> {
    let mut at = Cursor { text, at: 0 };
    let (fields, separators) = match lead {
        Lead::Year => {
            let year = at.digits(4, 4)?;
            let first = at.one_of(&['-', '/'])?;
            let month = at.digits(2, 2)?;
            let second = at.one_of(&['-', '/'])?;
            let day = at.digits(2, 2)?;
            ([year, month, day], [first, second])
        }
        Lead::DayOrMonth => {
            let a = at.digits(1, 2)?;
            let first = at.one_of(&['-', '/'])?;
            let b = at.digits(1, 2)?;
            let second = at.one_of(&['-', '/'])?;
            let year = at.digits(2, 4)?;
            ([a, b, year], [first, second])
        }
    };
    if at.done() {
        return Some(Shape {
            lead,
            fields,
            separators,
            time: None,
        });
    }
    let separator = at.one_of(&[' ', 'T'])?;
    let hour = match lead {
        Lead::Year => at.digits(2, 2)?,
        Lead::DayOrMonth => at.digits(1, 2)?,
    };
    at.one_of(&[':'])?;
    let minute = at.digits(2, 2)?;
    let mut second = None;
    let mut fraction = None;
    if at.peek() == Some(':') {
        at.one_of(&[':'])?;
        second = Some(at.digits(2, 2)?);
        if at.peek() == Some('.') {
            at.one_of(&['.'])?;
            fraction = Some(at.digits(1, usize::MAX)?);
        }
    }
    let zone = if at.done() {
        None
    } else {
        let spaced = at.peek() == Some(' ');
        if spaced {
            at.one_of(&[' '])?;
        }
        Some((spaced, zone(at.rest())?))
    };
    Some(Shape {
        lead,
        fields,
        separators,
        time: Some(TimeOfDay {
            separator,
            hour,
            minute,
            second,
            fraction,
            zone,
        }),
    })
}

/// The whole of `text` as a zone, or `None`.
fn zone(text: &str) -> Option<Zone<'_>> {
    if text == "Z" {
        return Some(Zone::Z);
    }
    let bytes = text.as_bytes();
    if let Some(&sign) = bytes.first()
        && (sign == b'+' || sign == b'-')
    {
        let digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
        let number = |s: &[u8]| s.iter().fold(0_i32, |n, b| n * 10 + i32::from(b - b'0'));
        let sign = if sign == b'+' { 1 } else { -1 };
        let rest = &bytes[1..];
        return match rest.len() {
            2 if digits(rest) => Some(Zone::Hours {
                seconds: sign * number(rest) * 3_600,
            }),
            4 if digits(rest) => Some(Zone::Compact {
                seconds: sign * (number(&rest[..2]) * 3_600 + number(&rest[2..]) * 60),
                minute: number(&rest[2..]),
            }),
            5 if digits(&rest[..2]) && rest[2] == b':' && digits(&rest[3..]) => Some(Zone::Colon {
                seconds: sign * (number(&rest[..2]) * 3_600 + number(&rest[3..]) * 60),
                minute: number(&rest[3..]),
            }),
            _ => None,
        };
    }
    let first = text.chars().next()?;
    let named = first.is_ascii_alphabetic()
        && text
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '+' | '/' | '-'));
    named.then_some(Zone::Name(text))
}

struct Cursor<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Cursor<'a> {
    fn done(&self) -> bool {
        self.at == self.text.len()
    }

    fn peek(&self) -> Option<char> {
        self.text[self.at..].chars().next()
    }

    fn rest(&self) -> &'a str {
        &self.text[self.at..]
    }

    /// A run of `min` to `max` ASCII digits, as long as it goes: a longer
    /// run than `max` does not match, as the patterns' fixed widths do not.
    fn digits(&mut self, min: usize, max: usize) -> Option<&'a str> {
        let run = self.text[self.at..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        if run < min || run > max {
            return None;
        }
        let digits = &self.text[self.at..self.at + run];
        self.at += run;
        Some(digits)
    }

    fn one_of(&mut self, chars: &[char]) -> Option<char> {
        let c = self.peek().filter(|c| chars.contains(c))?;
        self.at += c.len_utf8();
        Some(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shapes_of_date_rules_match_and_nothing_else_does() {
        for text in [
            "2024-01-05",
            "2024/01-05",
            "2024-01-05T10:30",
            "2024-01-05 10:30:15",
            "2024-01-05 10:30:15.123456789",
            "2024-01-05 10:30Z",
            "2024-01-05 10:30 +01:00",
            "2024-01-05 10:30:15+0100",
            "2024-01-05 10:30-08",
            "2024-01-05 10:30 UTC",
            "2024-01-05 10:30Europe/Berlin",
            "1/5/24",
            "15/3/24",
            "15-03-2024",
            "01/05/124",
            "1/5/2024 9:30",
            "01/05/2024T09:30:15.5 Z",
        ] {
            assert!(shape(text).is_some(), "{text}");
        }
        for text in [
            "2024-1-05",
            "2024-01-05 9:30",
            "2024-01-05 10:30 ",
            " 2024-01-05",
            "2024-01-05T",
            "2024.01.05",
            "20240105",
            "1/5/2",
            "1/5/20245",
            "123/5/2024",
            "2024-01-05 10:30+1",
            "2024-01-05 10:30+010",
            "2024-01-05 10:30  UTC",
            "2024-01-05 10:30 9UTC",
            "2024-01-05 10:30.5",
            "2024-01-05 10:30:15.",
            "not a date",
            "",
            "٢٠٢٤-٠١-٠٥",
        ] {
            assert!(shape(text).is_none(), "{text}");
        }
    }

    #[test]
    fn a_shape_holds_its_fields_as_written() {
        let s = shape("15-3-24 9:05:07.25 -05:30").unwrap();
        assert_eq!(s.lead, Lead::DayOrMonth);
        assert_eq!(s.fields, ["15", "3", "24"]);
        assert_eq!(s.separators, ['-', '-']);
        let t = s.time.unwrap();
        assert_eq!(
            (t.separator, t.hour, t.minute, t.second, t.fraction),
            (' ', "9", "05", Some("07"), Some("25"))
        );
        assert_eq!(
            t.zone,
            Some((
                true,
                Zone::Colon {
                    seconds: -19_800,
                    minute: 30
                }
            ))
        );
        assert_eq!(Zone::Hours { seconds: 3_600 }.offset_seconds(), Some(3_600));
        assert_eq!(Zone::Name("UTC").offset_seconds(), None);
    }
}
