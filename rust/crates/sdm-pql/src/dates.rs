// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Date literals: what `orders.ordered_at > 2024-01-31` compares with.
//!
//! The rules are D1 to D9 of `docs/design/rust-core/phase-0/date-rules.md`
//! ("The PQL literal rules"), first written down as the first attempt's
//! `DATES.md` (`archive/rust-core-attempt-1`, `rust/crates/pql/DATES.md`),
//! here over `jiff`'s calendar. A literal is the grammar's `DATE` token,
//! three runs of digits joined by one separator (`/`, `-` or `.`),
//! optionally followed by one space and a `TIME` token, three runs of
//! digits joined by `:` (D1).
//!
//! - A literal whose every run has four digits or fewer is read by D2 to
//!   D7 and D9 ([`Reader::plain`]); a day-first reading warns by D4.
//! - A literal with a run of five digits or more (D8) is not read: it is
//!   [`Reading::Packed`], refused by name before any field is read, by the
//!   owner's decision of 2026-10-12 (decision E79, register DT-5). The
//!   reference reads such a literal through `dateutil` (a packed date, a
//!   long zero-padded field) or refuses it; the core reads neither.
//!
//! What the reader takes from the host: the current year, which places a
//! two-digit year (D5), and whether the platform's `strftime` pads a year
//! below 1000, which decides D4's warning for such a year.

use std::fmt;

use jiff::civil::Date;

/// What reading a literal needs from the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reader {
    /// The current year, in the host's local zone, which places a two-digit
    /// year (D5); `None` where the host's facts give none, so a literal that
    /// needs it reads as [`Reading::Unplaced`] and every other reads as ever.
    pub year: Option<i16>,
    /// Whether the platform's `strftime` writes a year below 1000 with four
    /// digits (`HostFacts.platform.strftime_pads_year`); without it D4 gives
    /// no warning for such a year.
    pub pads_year: bool,
}

/// A date and time of day, as a literal gives them, on the wall clock of
/// its offset if it has one. The year runs from 0 to 9999 (D7, D9).
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

/// What a literal stands for: a wall date and time, and the UTC offset in
/// seconds when the literal gives one. Only D8's packed dates gave one, and
/// since T1.6c they are refused (decision E79), so the reader gives none;
/// the field stays because the validated query's format carries it
/// (`PqlLiteral::Time`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub at: DateTime,
    pub offset: Option<i32>,
}

impl fmt::Display for Stamp {
    /// `YYYY-MM-DDTHH:MM:SS`, then `+HH:MM` when there is an offset.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.at)?;
        if let Some(offset) = self.offset {
            let sign = if offset < 0 { '-' } else { '+' };
            let minutes = offset.unsigned_abs() / 60;
            write!(f, "{sign}{:02}:{:02}", minutes / 60, minutes % 60)?;
        }
        Ok(())
    }
}

/// The instants a host can hold a date literal as, inclusive at both ends,
/// as nanoseconds from the Unix epoch on the literal's own wall clock
/// (`HostFacts.date_literal_range`): pandas 2 holds a literal as a
/// nanosecond `Timestamp`, so the reference refuses one outside them as a
/// literal it cannot read (D9's refusal), where pandas 3 reads every year.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Held {
    pub min_ns: i64,
    pub max_ns: i64,
}

impl Held {
    /// Whether the host can hold `stamp`, read on its own wall clock.
    pub fn holds(&self, stamp: &Stamp) -> bool {
        // pandas bounds the UTC instant; the wall clock is that instant
        // only for a literal without an offset, and the reader gives none
        // since D8's packed dates are refused (decision E79).
        debug_assert!(stamp.offset.is_none(), "a date literal read with an offset");
        let at = stamp.at;
        let days = days_from_civil(i64::from(at.year), i64::from(at.month), i64::from(at.day));
        let seconds = days * 86_400
            + i64::from(at.hour) * 3_600
            + i64::from(at.minute) * 60
            + i64::from(at.second);
        let ns = i128::from(seconds) * 1_000_000_000;
        i128::from(self.min_ns) <= ns && ns <= i128::from(self.max_ns)
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date (Hinnant's
/// `days_from_civil`), for any year the reader gives (0 to 9999).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// D4's warning: a day-first reading of a literal whose format can be
/// named, in its own separator and with its time of day if it has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DayFirst {
    pub separator: char,
    pub time: bool,
}

impl DayFirst {
    /// The format as `strftime` writes it: `%d/%m/%Y`, then ` %H:%M:%S`
    /// when the literal has a time of day.
    pub fn format(&self) -> String {
        let s = self.separator;
        let time = if self.time { " %H:%M:%S" } else { "" };
        format!("%d{s}%m{s}%Y{time}")
    }
}

/// How a literal reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    Read {
        stamp: Stamp,
        /// D4's warning, for a day-first reading it names.
        day_first: Option<DayFirst>,
    },
    /// Not a date (D9): the reference refuses it too.
    Refused,
    /// A run of five digits or more (D8), refused by name before it is
    /// read, needing no current year (decision E79): the reference reads it
    /// through `dateutil`, or refuses it.
    Packed,
    /// A two-digit year the reader cannot place: the reader was given no
    /// current year (D5).
    Unplaced,
}

/// A run of this many digits or more is D8's, refused as
/// [`Reading::Packed`].
const LONG_DIGITS: usize = 5;
/// A field of this many digits or more is a year as written (D2, D5).
const YEAR_DIGITS: usize = 3;

/// The three runs of digits of a `DATE` or `TIME` token, if it is one, and
/// the separator between them.
fn fields<'a>(text: &'a str, separators: &[char]) -> Option<([&'a str; 3], char)> {
    let sep = text.chars().find(|c| separators.contains(c))?;
    let parts: Vec<&str> = text.split(sep).collect();
    let [a, b, c] = parts[..] else { return None };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (digits(a) && digits(b) && digits(c)).then_some(([a, b, c], sep))
}

/// A run of digits by its value, saturating: a run too long for a `u64`
/// is past every field's range anyway.
fn value(digits: &str) -> u64 {
    digits.bytes().fold(0u64, |n, b| {
        n.saturating_mul(10).saturating_add(u64::from(b - b'0'))
    })
}

/// A year, month and day in D2's order, chosen from the first field's
/// value alone: year first above 31 (or when `year_first`), day first above
/// 12, else month first.
fn in_order(values: [u64; 3], year_first: bool) -> (u64, u64, u64) {
    let [first, second, third] = values;
    if first > 31 || year_first {
        (first, second, third)
    } else if first > 12 {
        (third, second, first)
    } else {
        (third, first, second)
    }
}

/// A parsed date and time, before it is checked.
#[derive(Clone, Copy, Debug, Default)]
struct Fields {
    year: u64,
    month: u64,
    day: u64,
    hour: Option<u64>,
    minute: Option<u64>,
    second: Option<u64>,
    /// A two-digit year that needed the current year, which was not given.
    unplaced: bool,
}

impl Reader {
    /// The reading of `text`: a `DATE` token, optionally followed by one
    /// space and a `TIME` token, as the grammar gives them (D1).
    pub fn read(&self, text: &str) -> Reading {
        let (date, time) = match text.split_once(' ') {
            Some((date, time)) => (date, Some(time)),
            None => (text, None),
        };
        let Some((date, sep)) = fields(date, &['/', '-', '.']) else {
            return Reading::Refused;
        };
        let time = match time {
            None => None,
            Some(time) => match fields(time, &[':']) {
                Some((time, _)) => Some(time),
                None => return Reading::Refused,
            },
        };
        // D8: refused by name, before any field is read (E79).
        if date
            .iter()
            .chain(time.iter().flatten())
            .any(|f| f.len() >= LONG_DIGITS)
        {
            return Reading::Packed;
        }
        let read = self.plain(date, sep, time);
        if read.is_some_and(|(fields, _)| fields.unplaced) {
            return Reading::Unplaced;
        }
        match read.and_then(|(fields, day_first)| checked(fields).map(|s| (s, day_first))) {
            Some((stamp, day_first)) => Reading::Read { stamp, day_first },
            None => Reading::Refused,
        }
    }

    /// A two-digit year placed in the hundred years around the current
    /// year, from 50 years before it to 49 after (D5).
    fn century(&self, year: u64, unplaced: &mut bool) -> u64 {
        let Some(now) = self.year else {
            *unplaced = true;
            return year;
        };
        let now = i64::from(now);
        let mut y = i64::try_from(year).unwrap_or(i64::MAX) + now.div_euclid(100) * 100;
        if y >= now + 50 {
            y -= 100;
        } else if y < now - 50 {
            y += 100;
        }
        u64::try_from(y).unwrap_or(u64::MAX)
    }

    /// D2 to D7 and D9, for a literal whose every run has four digits or
    /// fewer, with D4's warning.
    fn plain(
        &self,
        date: [&str; 3],
        sep: char,
        time: Option<[&str; 3]>,
    ) -> Option<(Fields, Option<DayFirst>)> {
        let [first, second, third] = date;
        let long = |f: &str| f.len() >= YEAR_DIGITS;
        // D9: two date fields of three digits or more.
        if date.iter().filter(|f| long(f)).count() > 1 {
            return None;
        }
        let (mut year, month, day) = in_order(date.map(value), long(first));
        let day_first = !long(first) && (13..=31).contains(&value(first));
        let year_written = if long(first) { first } else { third };
        // D5: with no field of three digits or more, a year below 100 is
        // placed around the current year.
        if !date.iter().any(|f| long(f)) && year < 100 {
            let mut unplaced = false;
            year = self.century(year, &mut unplaced);
            if unplaced {
                let fields = Fields {
                    unplaced,
                    ..Fields::default()
                };
                return Some((fields, None));
            }
        }
        // D7: the year 0 only as `0000` first, then a month and a day of one
        // or two digits, and a time of day, if any, of fields of one or two.
        let short_time = time.is_none_or(|t| t.iter().all(|f| f.len() <= 2));
        if year == 0 && !(first == "0000" && !long(second) && !long(third) && short_time) {
            return None;
        }
        let (hour, minute, second_of) = match time {
            None => (None, None, None),
            Some([h, m, s]) => (Some(value(h)), Some(value(m)), Some(value(s))),
        };
        // D4: a day-first reading warns when its format can be named: a
        // year of four digits (below 1000 only where `strftime` pads it), a
        // day and month of one or two, a time of fields of one or two.
        let warns = day_first
            && second.len() <= 2
            && year_written.len() == 4
            && (self.pads_year || value(year_written) >= 1000)
            && short_time;
        let fields = Fields {
            year,
            month,
            day,
            hour,
            minute,
            second: second_of,
            unplaced: false,
        };
        Some((
            fields,
            warns.then_some(DayFirst {
                separator: sep,
                time: time.is_some(),
            }),
        ))
    }
}

/// The stamp `fields` names, if it names one (D6, D9): a month of 1 to 12,
/// a day the month has, an hour of 0 to 23, a minute and second of 0 to
/// 59, and a year of 0 to 9999 (the year 0 only as D7 lets it through).
fn checked(fields: Fields) -> Option<Stamp> {
    let small = |v: Option<u64>, most: u64| -> Option<i8> {
        let v = v.unwrap_or(0);
        if v > most { None } else { i8::try_from(v).ok() }
    };
    let year = i16::try_from(fields.year).ok().filter(|y| *y <= 9999)?;
    let month = i8::try_from(fields.month).ok()?;
    let day = i8::try_from(fields.day).ok()?;
    Date::new(year, month, day).ok()?;
    let at = DateTime {
        year,
        month,
        day,
        hour: small(fields.hour, 23)?,
        minute: small(fields.minute, 59)?,
        second: small(fields.second, 59)?,
    };
    Some(Stamp { at, offset: None })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IN_2026: Reader = Reader {
        year: Some(2026),
        pads_year: false,
    };

    fn read(reader: Reader, text: &str) -> String {
        match reader.read(text) {
            Reading::Read { stamp, day_first } => match day_first {
                Some(d) => format!("{stamp} warned {}", d.format()),
                None => stamp.to_string(),
            },
            Reading::Refused => "refused".into(),
            Reading::Packed => "packed".into(),
            Reading::Unplaced => "unplaced".into(),
        }
    }

    /// The examples of `DATES.md`, with the reference's answers.
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
            ("29.02.2024", "2024-02-29T00:00:00 warned %d.%m.%Y"),
            ("13/01/2024", "2024-01-13T00:00:00 warned %d/%m/%Y"),
            (
                "31/12/2024 23:59:59",
                "2024-12-31T23:59:59 warned %d/%m/%Y %H:%M:%S",
            ),
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
            ("230203-06-09", "packed"),
            ("20240101/00/123456", "packed"),
            ("00001/1/2", "packed"),
            ("2023-02-29", "refused"),
            ("2024-04-31", "refused"),
            ("2024-13-01", "refused"),
            ("2024-01-00", "refused"),
            ("10000-01-01", "packed"),
            ("123-4-567", "refused"),
        ] {
            assert_eq!(read(IN_2026, literal), want, "{literal}");
        }
    }

    /// D8 (E79): every literal with a run of five digits or more is
    /// refused by name, those the reference read through `dateutil` and
    /// those it refused alike, with or without a current year; a run of
    /// four digits still reads.
    #[test]
    fn a_run_of_five_digits_or_more_is_refused_by_name() {
        let none = Reader {
            year: None,
            pads_year: true,
        };
        for literal in [
            "2024-01-01 00000:00:00",
            "20240101-01-01",
            "202401011230-1-1",
            "20240101-01-00",
            "20240101-01-0199",
            "99991231-23-09",
            "20240101.0102.030405",
            "1.00000025.4",
            "2024-01-01 1:000001:00",
            "20240101-01-24",
            "20240101-1234-56",
            "240101-1-1",
            "2024-01-01 000001:00:00",
            "00000101-01-01",
            "1-1-12345",
            "00001-1-123",
            "991231.23.23",
            "01.5.96 05:000059:000004",
            "760101-01-01",
            "1/2/3 00001:0:0",
        ] {
            assert_eq!(read(IN_2026, literal), "packed", "{literal}");
            assert_eq!(read(none, literal), "packed", "{literal}");
        }
        for (literal, want) in [
            ("2024-01-01 0000:00:00", "2024-01-01T00:00:00"),
            ("9999-12-31", "9999-12-31T00:00:00"),
            ("1-1-1234", "1234-01-01T00:00:00"),
        ] {
            assert_eq!(read(IN_2026, literal), want, "{literal}");
        }
    }

    #[test]
    fn two_digit_years_follow_the_host_year() {
        let at = |year: i16| Reader {
            year: Some(year),
            pads_year: false,
        };
        let year = |reader: Reader, text: &str| match reader.read(text) {
            Reading::Read { stamp, .. } => stamp.at.year,
            other => panic!("{text}: {other:?}"),
        };
        assert_eq!(year(at(2026), "1/1/75"), 2075);
        assert_eq!(year(at(2026), "1/1/76"), 1976);
        assert_eq!(year(at(2027), "1/1/76"), 2076);
        assert_eq!(year(at(2080), "1/1/29"), 2129);
        assert_eq!(year(at(2080), "1/1/31"), 2031);
    }

    #[test]
    fn a_year_below_1000_warns_only_where_strftime_pads_it() {
        let pads = Reader {
            year: Some(2026),
            pads_year: true,
        };
        assert_eq!(read(IN_2026, "13/01/0999"), "0999-01-13T00:00:00");
        assert_eq!(
            read(pads, "13/01/0999"),
            "0999-01-13T00:00:00 warned %d/%m/%Y"
        );
        assert_eq!(read(pads, "13/01/999"), "0999-01-13T00:00:00");
        assert_eq!(
            read(IN_2026, "13/01/1000"),
            "1000-01-13T00:00:00 warned %d/%m/%Y"
        );
        // No warning where the format cannot be named.
        assert_eq!(read(pads, "13/001/2024"), "refused");
        assert_eq!(read(pads, "13/01/2024 001:02:03"), "2024-01-13T01:02:03");
    }

    #[test]
    fn a_two_digit_year_without_a_current_year_is_unplaced_and_nothing_else() {
        let none = Reader {
            year: None,
            pads_year: false,
        };
        // A year to place: unplaced.
        for literal in ["1/2/76", "13/1/24", "24-01-05"] {
            assert_eq!(read(none, literal), "unplaced", "{literal}");
        }
        // Every literal that places no year reads as with a year.
        for literal in [
            "2024-01-31",
            "13/01/2024",
            "20240101/00/123456",
            "00001/1/2",
            "760101-01-01",
            "02/015/75",
            "2024-13-01",
            "0000-01-01",
        ] {
            assert_eq!(read(none, literal), read(IN_2026, literal), "{literal}");
        }
    }
}
