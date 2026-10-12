// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Times a source stores as text, read by the core's fixed list of formats
//! and nothing else: `date-rules.md`, "The core's text formats", is the
//! spec this module follows (decision E66(h); E68). In short:
//!
//! - a value is read with its surrounding white space taken off, and an
//!   empty value holds no time;
//! - its format is the one of the list it is written in, found from the
//!   value alone ([`super::text_shapes`], the client's two shapes), and it
//!   is parsed by that format with `jiff`'s `strtime`;
//! - a day or month first value is read in the order the values read
//!   together decide by the client's rule (`day_comes_first`), and a
//!   two-digit year in the hundred years around the clock's;
//! - a value either reads as an instant or is missing (empty); a value in
//!   a format whose fields name no date or time of day, and any other
//!   value, is outside the list, and the reading that meets one refuses it
//!   by name rather than guess ([`Unread::Outside`]).
//!
//! No reading here follows pandas or `dateutil`: where the reference reads
//! text otherwise, the difference is registered (E68).

use jiff::fmt::strtime;

use super::text_shapes::{Lead, Shape, Zone, shape};
use super::{Civil, Instant};

/// Why values of text do not read as times here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unread {
    /// A value is outside the list (`text_time`).
    Outside,
    /// Day or month first values the values read together do not put in an
    /// order, read differently by the two (`text_time_order`).
    Order,
    /// Values read with different offsets, or with and without one, where
    /// the reading is not to UTC (`text_time_zones`).
    MixedZones,
}

/// Which field a day or month first value leads with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    MonthFirst,
    DayFirst,
}

/// One value as the list reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Read {
    /// Empty, or no value.
    Missing,
    /// An instant, and the offset east of UTC in seconds the value wrote,
    /// if any.
    Time(Instant, Option<i32>),
    /// Outside the list; `shaped` when it is in one of the client's shapes
    /// all the same.
    Outside { shaped: bool },
}

/// The fewest and the most of a column's values that read as times: the
/// two orders' counts where the values do not decide one, and the most
/// counting every value in the client's shapes but outside the list; and
/// how many are outside the shapes too, which the core never reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Share {
    pub fewest: usize,
    pub most: usize,
    pub unshaped: usize,
}

fn number(digits: &str) -> u32 {
    digits.bytes().fold(0_u32, |n, b| {
        n.saturating_mul(10).saturating_add(u32::from(b - b'0'))
    })
}

/// The order the client's rule (`day_comes_first`) gives `values`: day
/// first when some day or month first value's first field is past 12 and
/// no second field is, month first the other way round, otherwise none.
pub fn decided_order(values: &[Option<&str>]) -> Option<Order> {
    let (mut first, mut second) = (false, false);
    for value in values.iter().flatten() {
        if let Some(shape) = shape(value.trim())
            && shape.lead == Lead::DayOrMonth
        {
            first |= number(shape.fields[0]) > 12;
            second |= number(shape.fields[1]) > 12;
        }
    }
    match (first, second) {
        (true, false) => Some(Order::DayFirst),
        (false, true) => Some(Order::MonthFirst),
        _ => None,
    }
}

/// The `strtime` format of the list `shape` is written in, read in
/// `order`, or `None` where the shape is outside the list: separators that
/// differ, a second past 59, a fraction past nine digits, an offset of a
/// day or more or minutes past 59, a zone named other than `UTC` or `GMT`.
fn format(shape: &Shape<'_>, order: Order) -> Option<String> {
    let [a, b] = shape.separators;
    if a != b {
        return None;
    }
    let mut format = match (shape.lead, order) {
        (Lead::Year, _) => format!("%Y{a}%m{a}%d"),
        (Lead::DayOrMonth, Order::MonthFirst) => format!("%m{a}%d{a}%Y"),
        (Lead::DayOrMonth, Order::DayFirst) => format!("%d{a}%m{a}%Y"),
    };
    let Some(time) = shape.time else {
        return Some(format);
    };
    format.push(time.separator);
    format.push_str("%H:%M");
    if let Some(second) = time.second {
        if number(second) > 59 {
            return None;
        }
        format.push_str(":%S");
    }
    if let Some(fraction) = time.fraction {
        if fraction.len() > 9 {
            return None;
        }
        format.push_str(".%f");
    }
    if let Some((spaced, zone)) = time.zone {
        if spaced {
            format.push(' ');
        }
        let within = |seconds: i32, minute: i32| seconds.abs() < 86_400 && minute < 60;
        match zone {
            Zone::Z => format.push('Z'),
            Zone::Hours { seconds } if within(seconds, 0) => format.push_str("%:::z"),
            Zone::Compact { seconds, minute } if within(seconds, minute) => format.push_str("%z"),
            Zone::Colon { seconds, minute } if within(seconds, minute) => format.push_str("%:z"),
            Zone::Name(name @ ("UTC" | "GMT")) => format.push_str(name),
            _ => return None,
        }
    }
    Some(format)
}

/// Whether the date and time of day `shape` writes, its day and month in
/// `order`, name none: a month past 12, a day past 31 or 0, an hour past
/// 23 or a minute past 59. A day the month lacks is `jiff`'s to find.
/// Such a value is outside the list: what it names is not a date, and
/// another reader may still read one in it (`32-12-24`).
fn names_no_date(shape: &Shape<'_>, order: Order) -> bool {
    let [a, b, c] = shape.fields;
    let (month, day) = match (shape.lead, order) {
        (Lead::Year, _) => (number(b), number(c)),
        (Lead::DayOrMonth, Order::MonthFirst) => (number(a), number(b)),
        (Lead::DayOrMonth, Order::DayFirst) => (number(b), number(a)),
    };
    let clock = shape
        .time
        .is_some_and(|t| number(t.hour) > 23 || number(t.minute) > 59);
    !(1..=12).contains(&month) || !(1..=31).contains(&day) || clock
}

/// The first and last instants the list reads: the years 1 to 9999, in
/// UTC.
fn in_years(at: Instant) -> bool {
    let low = Instant::from_civil(&Civil::new(1, 1, 1, 0, 0, 0, 0).expect("a date"), 0);
    let high = Instant::from_civil(&Civil::new(10_000, 1, 1, 0, 0, 0, 0).expect("a date"), 0);
    (low..high).contains(&at)
}

/// Reads text times for one run: the clock's year places a two-digit
/// year.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reader {
    year: i64,
}

impl Reader {
    /// A reader for a run whose clock is in `year`.
    pub fn new(year: i64) -> Reader {
        Reader { year }
    }

    /// A two-digit year in the hundred years around the clock's: from 50
    /// years before it to 49 after.
    fn century(&self, yy: i64) -> i64 {
        let mut year = yy + self.year.div_euclid(100) * 100;
        if year >= self.year + 50 {
            year -= 100;
        } else if year < self.year - 50 {
            year += 100;
        }
        year
    }

    /// `value` read by the list, a day or month first value in `order`.
    fn one(&self, value: &str, order: Order) -> Read {
        let text = value.trim();
        if text.is_empty() {
            return Read::Missing;
        }
        let Some(shape) = shape(text) else {
            return Read::Outside { shaped: false };
        };
        let outside = Read::Outside { shaped: true };
        let Some(format) = format(&shape, order) else {
            return outside;
        };
        if names_no_date(&shape, order) {
            return outside;
        }
        // The shape fixed every field's width, and the checks above every
        // field's range but the day's: a value the format does not parse
        // is not one the list reads.
        let Ok(mut parsed) = strtime::parse(&format, text) else {
            return outside;
        };
        if shape.lead == Lead::DayOrMonth && shape.fields[2].len() == 2 {
            let year = self.century(i64::from(number(shape.fields[2])));
            let Ok(year) = i16::try_from(year) else {
                return outside;
            };
            if parsed.set_year(Some(year)).is_err() {
                return outside;
            }
        }
        // A day the month lacks names no date.
        let Ok(at) = parsed.to_datetime() else {
            return outside;
        };
        let offset = match shape.time.and_then(|t| t.zone) {
            None => None,
            Some((_, Zone::Z | Zone::Name(_))) => Some(0),
            Some(_) => match parsed.offset() {
                Some(offset) => Some(offset.seconds()),
                None => return outside,
            },
        };
        let Some(civil) = Civil::new(
            i64::from(at.year()),
            at.month() as u8,
            at.day() as u8,
            at.hour() as u8,
            at.minute() as u8,
            at.second() as u8,
            at.subsec_nanosecond() as u32,
        ) else {
            return outside;
        };
        let instant = Instant::from_civil(&civil, offset.unwrap_or(0));
        if !in_years(instant) {
            return outside;
        }
        Read::Time(instant, offset)
    }

    /// `values` read in the order they decide, twice where they decide
    /// none: month first, then day first.
    fn readings(&self, values: &[Option<&str>]) -> (Vec<Read>, Vec<Read>) {
        let read = |order: Order| -> Vec<Read> {
            values
                .iter()
                .map(|v| v.map_or(Read::Missing, |v| self.one(v, order)))
                .collect()
        };
        match decided_order(values) {
            Some(order) => {
                let once = read(order);
                (once.clone(), once)
            }
            None => (read(Order::MonthFirst), read(Order::DayFirst)),
        }
    }

    /// The values' instants, a missing value `None`, as a source's text
    /// time column is read (register DT-7): refused for
    /// a value outside the list, an order the values leave open, and,
    /// unless read to UTC, offsets mixed.
    pub fn column(
        &self,
        values: &[Option<&str>],
        utc: bool,
    ) -> Result<Vec<Option<Instant>>, Unread> {
        let (month_first, day_first) = self.readings(values);
        // Outside the list whichever way round; outside one way only, the
        // order decides it.
        if month_first
            .iter()
            .zip(&day_first)
            .any(|pair| matches!(pair, (Read::Outside { .. }, Read::Outside { .. })))
        {
            return Err(Unread::Outside);
        }
        if month_first != day_first {
            return Err(Unread::Order);
        }
        if !utc && mixed(&month_first) {
            return Err(Unread::MixedZones);
        }
        Ok(month_first
            .into_iter()
            .map(|r| match r {
                Read::Time(at, _) => Some(at),
                _ => None,
            })
            .collect())
    }

    /// [`Reader::column`] not to UTC: a SQL source's text time column.
    pub fn each(&self, values: &[Option<&str>]) -> Result<Vec<Option<Instant>>, Unread> {
        self.column(values, false)
    }

    /// How many of `values` read as times, at fewest and at most, for
    /// inference's score: the fewer of the two orders' counts where the
    /// values decide none, and the more counting every value in the
    /// client's shapes the list does not read as one it might; and how many
    /// are outside the shapes. Refused only for offsets mixed among the
    /// values read.
    pub fn share(&self, values: &[Option<&str>]) -> Result<Share, Unread> {
        let (month_first, day_first) = self.readings(values);
        if mixed(&month_first) || mixed(&day_first) {
            return Err(Unread::MixedZones);
        }
        let count = |read: &[Read], shaped: bool| {
            read.iter()
                .filter(|r| match r {
                    Read::Time(..) => true,
                    Read::Outside { shaped: true } => shaped,
                    _ => false,
                })
                .count()
        };
        Ok(Share {
            fewest: count(&month_first, false).min(count(&day_first, false)),
            most: count(&month_first, true).max(count(&day_first, true)),
            unshaped: month_first
                .iter()
                .filter(|r| matches!(r, Read::Outside { shaped: false }))
                .count(),
        })
    }

    /// The format, written as `strftime` writes one, of the first day or
    /// month first value of `values` when the client's rule decides that
    /// the day comes first, which inference warns of
    /// (`DATES_READ_DAY_FIRST`).
    pub fn day_first_format(&self, values: &[Option<&str>]) -> Option<String> {
        if decided_order(values) != Some(Order::DayFirst) {
            return None;
        }
        let shape = values
            .iter()
            .flatten()
            .filter_map(|v| shape(v.trim()))
            .find(|s| s.lead == Lead::DayOrMonth)?;
        let [a, b] = shape.separators;
        let year = if shape.fields[2].len() == 2 {
            "%y"
        } else {
            "%Y"
        };
        let mut text = format!("%d{a}%m{b}{year}");
        if let Some(time) = shape.time {
            text.push(time.separator);
            text.push_str("%H:%M");
            if time.second.is_some() {
                text.push_str(":%S");
            }
            if time.fraction.is_some() {
                text.push_str(".%f");
            }
            if let Some((spaced, zone)) = time.zone {
                if spaced {
                    text.push(' ');
                }
                text.push_str(match zone {
                    Zone::Name(_) => "%Z",
                    _ => "%z",
                });
            }
        }
        Some(text)
    }
}

/// Whether the values read hold different offsets, or some one and some
/// none.
fn mixed(read: &[Read]) -> bool {
    let mut offsets = read.iter().filter_map(|r| match r {
        Read::Time(_, offset) => Some(*offset),
        _ => None,
    });
    let first = offsets.next();
    first.is_some_and(|first| offsets.any(|o| o != first))
}

/// The reading inference asks for (`sdm_schema::infer::TextTimes`), in
/// nanoseconds.
impl sdm_schema::infer::TextTimes for Reader {
    fn read(
        &self,
        values: &[Option<&str>],
        utc: bool,
    ) -> Result<Vec<Option<i128>>, sdm_schema::infer::TextUnread> {
        self.column(values, utc)
            .map(|read| read.into_iter().map(|at| at.map(Instant::nanos)).collect())
            .map_err(unread)
    }

    fn share(
        &self,
        values: &[Option<&str>],
    ) -> Result<sdm_schema::infer::TextShare, sdm_schema::infer::TextUnread> {
        Reader::share(self, values)
            .map(|s| sdm_schema::infer::TextShare {
                fewest: s.fewest,
                most: s.most,
                unshaped: s.unshaped,
            })
            .map_err(unread)
    }

    fn day_first_format(&self, values: &[Option<&str>]) -> Option<String> {
        Reader::day_first_format(self, values)
    }
}

fn unread(unread: Unread) -> sdm_schema::infer::TextUnread {
    match unread {
        Unread::Outside | Unread::Order => sdm_schema::infer::TextUnread::Undecided,
        Unread::MixedZones => sdm_schema::infer::TextUnread::MixedZones,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[&str], utc: bool) -> Result<Vec<Option<String>>, Unread> {
        let values: Vec<Option<&str>> = values.iter().map(|v| Some(*v)).collect();
        Reader::new(2026).column(&values, utc).map(|r| {
            r.into_iter()
                .map(|at| at.map(|at| at.civil(0).to_string()))
                .collect()
        })
    }

    fn at(text: &str) -> Option<String> {
        Some(format!("{text}.000000000"))
    }

    #[test]
    fn iso_dates_and_timestamps_read_with_their_offsets() {
        assert_eq!(
            column(
                &[
                    "2024-01-05",
                    " 2024/01/06 ",
                    "2024-01-05T10:30",
                    "2024-01-05 10:30:15.25",
                    "2024-01-05 10:30:15.123456789",
                ],
                false
            ),
            Ok(vec![
                at("2024-01-05T00:00:00"),
                at("2024-01-06T00:00:00"),
                at("2024-01-05T10:30:00"),
                Some("2024-01-05T10:30:15.250000000".into()),
                Some("2024-01-05T10:30:15.123456789".into()),
            ])
        );
        for (zone, utc) in [
            ("Z", "2024-01-05T10:30:00"),
            (" UTC", "2024-01-05T10:30:00"),
            ("GMT", "2024-01-05T10:30:00"),
            ("+01", "2024-01-05T09:30:00"),
            ("+0130", "2024-01-05T09:00:00"),
            (" -08:00", "2024-01-05T18:30:00"),
            ("+23:59", "2024-01-04T10:31:00"),
        ] {
            let value = format!("2024-01-05 10:30{zone}");
            assert_eq!(column(&[&value], false), Ok(vec![at(utc)]), "{zone}");
        }
    }

    #[test]
    fn values_outside_the_list_are_refused() {
        for text in [
            "March 3 2020",
            "20240101",
            "2024-1-5",
            "15.01.2024",
            "2024/01-05",
            "1/5-2024",
            "2024-01-05 10:30:60",
            "2024-01-05 10:30:15.1234567891",
            "2024-01-05 10:30+24:00",
            "2024-01-05 10:30+01:99",
            "2024-01-05 10:30 EST",
            "2024-01-05 10:30 utc",
            "0000-01-05",
            "9999-12-31 23:00-02:00",
            "now",
            "NaT",
            "AW00000001",
        ] {
            assert_eq!(column(&[text], false), Err(Unread::Outside), "{text}");
            assert_eq!(column(&[text], true), Err(Unread::Outside), "{text}");
        }
    }

    #[test]
    fn a_value_naming_no_date_is_outside_and_an_empty_one_missing() {
        for text in [
            "2024-13-05",
            "2023-02-29",
            "2024-01-05 24:00",
            "2024-01-05 10:60",
            "32-12-24",
        ] {
            assert_eq!(column(&[text], false), Err(Unread::Outside), "{text}");
        }
        assert_eq!(column(&["", "  ", "\u{3000}"], false), Ok(vec![None; 3]));
    }

    #[test]
    fn the_values_decide_the_order_or_are_refused_where_it_matters() {
        assert_eq!(
            column(&["01/02/2024", "13/02/2024"], false),
            Ok(vec![at("2024-02-01T00:00:00"), at("2024-02-13T00:00:00")])
        );
        assert_eq!(
            column(&["01/02/2024", "02/13/2024"], false),
            Ok(vec![at("2024-01-02T00:00:00"), at("2024-02-13T00:00:00")])
        );
        assert_eq!(column(&["01/02/2024"], false), Err(Unread::Order));
        assert_eq!(
            column(&["13/02/2024", "02/13/2024"], false),
            Err(Unread::Order)
        );
        // The same day either way round.
        assert_eq!(
            column(&["05/05/2024", "2024-01-01"], false),
            Ok(vec![at("2024-05-05T00:00:00"), at("2024-01-01T00:00:00")])
        );
        let reader = Reader::new(2026);
        assert_eq!(
            reader.day_first_format(&[Some("15-03-24 10:30 +01:00")]),
            Some("%d-%m-%y %H:%M %z".into())
        );
        assert_eq!(reader.day_first_format(&[Some("03/15/2024")]), None);
        assert_eq!(reader.day_first_format(&[Some("01/02/2024")]), None);
    }

    #[test]
    fn two_digit_years_fall_in_the_hundred_years_around_the_clock() {
        assert_eq!(
            column(
                &["1/15/24", "1/15/75", "1/15/76", "2/29/00", "1/15/124"],
                false
            ),
            Ok(vec![
                at("2024-01-15T00:00:00"),
                at("2075-01-15T00:00:00"),
                at("1976-01-15T00:00:00"),
                at("2000-02-29T00:00:00"),
                at("0124-01-15T00:00:00"),
            ])
        );
    }

    #[test]
    fn offsets_mixed_are_refused_unless_read_to_utc() {
        let values = ["2024-01-05 10:00+01:00", "2024-01-05 10:00"];
        assert_eq!(column(&values, false), Err(Unread::MixedZones));
        assert_eq!(
            column(&values, true),
            Ok(vec![at("2024-01-05T09:00:00"), at("2024-01-05T10:00:00")])
        );
        assert_eq!(
            column(&["2024-01-05 10:00Z", "2024-01-05 11:00+00:00"], false),
            Ok(vec![at("2024-01-05T10:00:00"), at("2024-01-05T11:00:00")])
        );
    }

    #[test]
    fn the_share_counts_what_reads_at_fewest_and_most() {
        let reader = Reader::new(2026);
        let share = |values: &[&str]| {
            let values: Vec<Option<&str>> = values.iter().map(|v| Some(*v)).collect();
            reader.share(&values)
        };
        assert_eq!(
            share(&[
                "2024-01-05",
                "AW00000001",
                "2024-13-01",
                "2024-01-05 10:30:60"
            ]),
            Ok(Share {
                fewest: 1,
                most: 3,
                unshaped: 1
            })
        );
        assert_eq!(
            share(&["13/02/2024", "02/13/2024", "13/13/2024"]),
            Ok(Share {
                fewest: 1,
                most: 3,
                unshaped: 0
            })
        );
        assert_eq!(
            share(&["2024-01-05 10:00+01", "2024-01-05 10:00"]),
            Err(Unread::MixedZones)
        );
    }
}
