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
//! - A literal with a run of five digits or more is read by D8
//!   ([`Reader::long`]): a first field of six, eight, twelve or fourteen
//!   digits is a packed date, the two fields after it times of day or, after
//!   a `-` once an hour is known, a UTC offset; any other long field is a
//!   value in D2's order, its year as written.
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
/// seconds when the literal gives one (D8).
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
    /// A two-digit year the reader cannot place: the reader was given no
    /// current year (D5).
    Unplaced,
}

/// A run of this many digits or more is read by D8.
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
    offset: Option<i64>,
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
        let long = date
            .iter()
            .chain(time.iter().flatten())
            .any(|f| f.len() >= LONG_DIGITS);
        let read = if long {
            self.long(date, sep, time).map(|f| (f, None))
        } else {
            self.plain(date, sep, time)
        };
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
            offset: None,
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

    /// D8, for a literal with a run of five digits or more.
    fn long(&self, date: [&str; 3], sep: char, time: Option<[&str; 3]>) -> Option<Fields> {
        let [first, second, third] = date;
        let mut read = Fields::default();
        if matches!(first.len(), 6 | 8 | 12 | 14) {
            let at = |from: usize, to: usize| value(&first[from..to]);
            if first.len() == 6 {
                // `YYMMDD` in D2's order, its year placed by D5.
                let (year, month, day) = in_order([at(0, 2), at(2, 4), at(4, 6)], false);
                (read.year, read.month, read.day) =
                    (self.century(year, &mut read.unplaced), month, day);
            } else {
                // `YYYYMMDD`, then `HHMM`, then `SS`.
                (read.year, read.month, read.day) = (at(0, 4), at(4, 6), at(6, 8));
                if first.len() >= 12 {
                    (read.hour, read.minute) = (Some(at(8, 10)), Some(at(10, 12)));
                }
                if first.len() == 14 {
                    read.second = Some(at(12, 14));
                }
            }
            for field in [second, third] {
                let at = |from: usize, to: usize| value(&field[from..to]);
                if sep == '-' && read.hour.is_some() {
                    // A UTC offset: `HHMM` or `H` or `HH` hours, west.
                    let (hours, minutes) = match field.len() {
                        4 => (at(0, 2), at(2, 4)),
                        1 | 2 => (value(field), 0),
                        _ => return None,
                    };
                    let seconds = i64::try_from(hours * 3600 + minutes * 60).ok()?;
                    read.offset = Some(-seconds);
                } else if matches!(field.len(), 2 | 4) && read.hour.is_none() {
                    // An hour, or an hour and minutes.
                    read.hour = Some(at(0, 2));
                    if field.len() == 4 {
                        read.minute = Some(at(2, 4));
                    }
                } else if field.len() == 6 {
                    // `HHMMSS`.
                    (read.hour, read.minute, read.second) =
                        (Some(at(0, 2)), Some(at(2, 4)), Some(at(4, 6)));
                } else {
                    return None;
                }
            }
        } else {
            let long = |f: &str| f.len() >= YEAR_DIGITS;
            if date.iter().filter(|f| long(f)).count() > 1 {
                return None;
            }
            // In D2's order, the year placed by D5 where no date field has
            // three digits or more (the long run is then in the time).
            (read.year, read.month, read.day) = in_order(date.map(value), long(first));
            if !date.iter().any(|f| long(f)) && read.year < 100 {
                read.year = self.century(read.year, &mut read.unplaced);
            }
        }
        if let Some([h, m, s]) = time {
            // An hour of six, eight, twelve or fourteen digits is a packed
            // field where a time of day must be: refused.
            if matches!(h.len(), 6 | 8 | 12 | 14) {
                return None;
            }
            (read.hour, read.minute, read.second) =
                (Some(value(h)), Some(value(m)), Some(value(s)));
        }
        // D7: the year 0 is read only from a plain `0000`.
        (read.year != 0 || read.unplaced).then_some(read)
    }
}

/// The stamp `fields` names, if it names one (D6, D9): a month of 1 to 12,
/// a day the month has, an hour of 0 to 23, a minute and second of 0 to
/// 59, a year of 0 to 9999 (the year 0 only as D7 lets it through), and an
/// offset of less than a day.
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
    let offset = match fields.offset {
        None => None,
        Some(o) if o.abs() < 86_400 => Some(i32::try_from(o).ok()?),
        Some(_) => return None,
    };
    Some(Stamp { at, offset })
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
            ("230203-06-09", "2003-02-23T06:00:00-09:00"),
            ("20240101/00/123456", "2024-01-01T12:34:56"),
            ("00001/1/2", "0001-01-02T00:00:00"),
            ("2023-02-29", "refused"),
            ("2024-04-31", "refused"),
            ("2024-13-01", "refused"),
            ("2024-01-00", "refused"),
            ("10000-01-01", "refused"),
            ("123-4-567", "refused"),
        ] {
            assert_eq!(read(IN_2026, literal), want, "{literal}");
        }
    }

    #[test]
    fn packed_dates_and_long_fields_read_by_d8() {
        for (literal, want) in [
            ("2024-01-01 00000:00:00", "2024-01-01T00:00:00"),
            ("20240101-01-01", "2024-01-01T01:00:00-01:00"),
            ("202401011230-1-1", "2024-01-01T12:30:00-01:00"),
            ("20240101-01-00", "2024-01-01T01:00:00+00:00"),
            ("20240101-01-0199", "2024-01-01T01:00:00-02:39"),
            ("99991231-23-09", "9999-12-31T23:00:00-09:00"),
            ("20240101.0102.030405", "2024-01-01T03:04:05"),
            ("1.00000025.4", "0004-01-25T00:00:00"),
            ("2024-01-01 1:000001:00", "2024-01-01T01:01:00"),
            // Refused: an offset of a day or more, an offset of three
            // digits, a one-digit field after a packed date, a packed hour,
            // the year 0, a year past 9999, two long fields.
            ("20240101-01-24", "refused"),
            ("20240101-1234-56", "refused"),
            ("240101-1-1", "refused"),
            ("2024-01-01 000001:00:00", "refused"),
            ("00000101-01-01", "refused"),
            ("1-1-12345", "refused"),
            ("00001-1-123", "refused"),
            ("991231.23.23", "refused"),
            // A long run in the time alone: the year is still placed.
            ("01.5.96 05:000059:000004", "1996-01-05T05:59:04"),
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
        assert_eq!(year(at(2026), "760101-01-01"), 1976);
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
        // A year to place, in a plain literal and a packed one: unplaced.
        for literal in [
            "1/2/76",
            "13/1/24",
            "24-01-05",
            "760101-01-01",
            "1/2/3 00001:0:0",
        ] {
            assert_eq!(read(none, literal), "unplaced", "{literal}");
        }
        // Every literal that places no year reads as with a year.
        for literal in [
            "2024-01-31",
            "13/01/2024",
            "20240101/00/123456",
            "00001/1/2",
            "02/015/75",
            "2024-13-01",
            "0000-01-01",
        ] {
            assert_eq!(read(none, literal), read(IN_2026, literal), "{literal}");
        }
    }
}
