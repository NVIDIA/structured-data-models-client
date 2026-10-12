// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Date literals: what `orders.ordered_at > 2024-01-31` compares with.
//!
//! The rules are D1 to D9 of the first attempt's `DATES.md`
//! (`archive/rust-core-attempt-1`, `rust/crates/pql/DATES.md`), which
//! `docs/design/rust-core/phase-0/date-rules.md` confirms against the
//! reference. This is the interim reader of T1.6a: it reads D1 to D3, D5 to
//! D7 and D9 over `jiff`'s calendar. A literal with a field of five digits or
//! more, which D8 reads as a packed date or a year by its value, is not read
//! yet ([`Reading::Deferred`]), and the day-first warning of D4 is not given;
//! T1.6b completes both.

use std::fmt;

use jiff::civil::Date;

/// What reading a literal needs from the host: the current year, which
/// places a two-digit year (D5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reader {
    pub year: i16,
}

/// A date and time of day, as a literal gives them. The year runs from 0
/// to 9999 (D7, D9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DateTime {
    pub year: i16,
    pub month: i8,
    pub day: i8,
    pub hour: i8,
    pub minute: i8,
    pub second: i8,
}

impl fmt::Display for DateTime {
    /// `YYYY-MM-DDTHH:MM:SS`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

/// How a literal reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    Read(DateTime),
    /// Not a date: the reference refuses it too (D9).
    Refused,
    /// A shape this reader does not read yet (D8).
    Deferred,
}

/// A field of this many digits or more is read by D8.
const DEFERRED_DIGITS: usize = 5;

/// The three runs of digits of a `DATE` or `TIME` token, if it is one.
fn fields<'a>(text: &'a str, separators: &[char]) -> Option<[&'a str; 3]> {
    let sep = text.chars().find(|c| separators.contains(c))?;
    let parts: Vec<&str> = text.split(sep).collect();
    let [a, b, c] = parts[..] else { return None };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (digits(a) && digits(b) && digits(c)).then_some([a, b, c])
}

/// A run of digits by its value, saturating: any run this reader reads has
/// fewer than [`DEFERRED_DIGITS`] digits.
fn value(digits: &str) -> u32 {
    digits.bytes().fold(0u32, |n, b| {
        n.saturating_mul(10).saturating_add(u32::from(b - b'0'))
    })
}

/// Whether `text`, a `DATE` token optionally followed by one space and a
/// `TIME` token, needs a rule this reader does not have yet.
pub fn deferred(text: &str) -> bool {
    text.split([' ', '/', '-', '.', ':'])
        .any(|run| run.len() >= DEFERRED_DIGITS)
}

impl Reader {
    /// The reading of `text`: a `DATE` token, optionally followed by one
    /// space and a `TIME` token, as the grammar gives them (D1).
    pub fn read(&self, text: &str) -> Reading {
        if deferred(text) {
            return Reading::Deferred;
        }
        let (date, time) = match text.split_once(' ') {
            Some((date, time)) => (date, Some(time)),
            None => (text, None),
        };
        let Some(date) = fields(date, &['/', '-', '.']) else {
            return Reading::Refused;
        };
        let (hour, minute, second) = match time {
            None => (0, 0, 0),
            Some(time) => {
                let Some([h, m, s]) = fields(time, &[':']) else {
                    return Reading::Refused;
                };
                // D6: by value; an hour of 24 or a minute or second of 60
                // is refused.
                let (h, m, s) = (value(h), value(m), value(s));
                if h > 23 || m > 59 || s > 59 {
                    return Reading::Refused;
                }
                (h, m, s)
            }
        };
        let Some((year, month, day)) = self.order(date) else {
            return Reading::Refused;
        };
        let (Ok(year), Ok(month), Ok(day)) =
            (i16::try_from(year), i8::try_from(month), i8::try_from(day))
        else {
            return Reading::Refused;
        };
        // D9: a date that exists, from the year 0 to 9999.
        if !(0..=9999).contains(&year) || Date::new(year, month, day).is_err() {
            return Reading::Refused;
        }
        let narrow = |v: u32| i8::try_from(v).unwrap_or(i8::MAX);
        Reading::Read(DateTime {
            year,
            month,
            day,
            hour: narrow(hour),
            minute: narrow(minute),
            second: narrow(second),
        })
    }

    /// The year, month and day of three date fields (D2, D5, D7, D9).
    fn order(&self, [first, second, third]: [&str; 3]) -> Option<(i32, u32, u32)> {
        let long = |f: &str| f.len() >= 3;
        // D9: two date fields of three digits or more.
        if [first, second, third].iter().filter(|f| long(f)).count() > 1 {
            return None;
        }
        // D2: the order comes from the first field alone.
        let lead = value(first);
        let (year, month, day) = if lead > 31 || long(first) {
            (first, second, third)
        } else if lead > 12 {
            (third, second, first)
        } else {
            (third, first, second)
        };
        let mut y = i32::try_from(value(year)).ok()?;
        // D7: the year 0 only as `0000` first, then a month and a day of one
        // or two digits.
        if y == 0 && long(year) && !(first == "0000" && !long(second) && !long(third)) {
            return None;
        }
        // D5: with no field of three digits or more, a year below 100 is
        // placed in the hundred years around the current year.
        if ![first, second, third].iter().any(|f| long(f)) && y < 100 {
            let now = i32::from(self.year);
            y += now / 100 * 100;
            if y >= now + 50 {
                y -= 100;
            } else if y < now - 50 {
                y += 100;
            }
        }
        Some((y, value(month), value(day)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> String {
        match (Reader { year: 2026 }).read(text) {
            Reading::Read(t) => t.to_string(),
            Reading::Refused => "refused".into(),
            Reading::Deferred => "deferred".into(),
        }
    }

    /// The examples of `DATES.md` this reader reads, with the reference's
    /// answers.
    #[test]
    fn the_examples_read_as_the_reference_reads_them() {
        for (literal, want) in [
            ("2024-1-5", "2024-01-05T00:00:00"),
            ("2024.1.5", "2024-01-05T00:00:00"),
            ("1/2/2024 12:00:00", "2024-01-02T12:00:00"),
            ("2024-01-31", "2024-01-31T00:00:00"),
            ("001-2-3", "0001-02-03T00:00:00"),
            ("1.2.3456", "3456-01-02T00:00:00"),
            ("1-2024-5", "refused"),
            ("01/02/2024", "2024-01-02T00:00:00"),
            ("01.02.2024", "2024-01-02T00:00:00"),
            ("12/11/2024", "2024-12-11T00:00:00"),
            ("29.02.2024", "2024-02-29T00:00:00"),
            ("13/01/2024", "2024-01-13T00:00:00"),
            ("31/12/2024 23:59:59", "2024-12-31T23:59:59"),
            ("13/1/24", "2024-01-13T00:00:00"),
            ("24-01-05", "2005-01-24T00:00:00"),
            ("99-1-1", "1999-01-01T00:00:00"),
            ("1/2/3", "2003-01-02T00:00:00"),
            ("100-1-1", "0100-01-01T00:00:00"),
            ("02/015/75", "0075-02-15T00:00:00"),
            ("2024-01-02 12:30:00", "2024-01-02T12:30:00"),
            ("2024-01-02 001:002:003", "2024-01-02T01:02:03"),
            ("2024-01-02 24:00:00", "refused"),
            ("2024-01-02 23:59:60", "refused"),
            ("0000-01-01", "0000-01-01T00:00:00"),
            ("0-1-1", "refused"),
            ("2023-02-29", "refused"),
            ("2024-04-31", "refused"),
            ("2024-13-01", "refused"),
            ("2024-01-00", "refused"),
            ("123-4-567", "refused"),
        ] {
            assert_eq!(read(literal), want, "{literal}");
        }
    }

    #[test]
    fn packed_dates_and_long_fields_wait_for_t1_6b() {
        for literal in [
            "230203-06-09",
            "20240101/00/123456",
            "00001/1/2",
            "10000-01-01",
            "2024-01-02 000001:00:00",
        ] {
            assert_eq!(read(literal), "deferred", "{literal}");
        }
    }

    #[test]
    fn two_digit_years_follow_the_host_year() {
        let in_2026 = Reader { year: 2026 };
        let in_2080 = Reader { year: 2080 };
        let year = |reader: Reader, text: &str| match reader.read(text) {
            Reading::Read(t) => t.year,
            other => panic!("{text}: {other:?}"),
        };
        assert_eq!(year(in_2026, "1/1/75"), 2075);
        assert_eq!(year(in_2026, "1/1/76"), 1976);
        assert_eq!(year(in_2080, "1/1/29"), 2129);
        assert_eq!(year(in_2080, "1/1/31"), 2031);
    }
}
