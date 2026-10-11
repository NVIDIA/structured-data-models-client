// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Times a source stores as text, read as the reference reads them, over
//! the shapes of [`super::text_shapes`] only (`date-rules.md`, "The
//! engine's own reading"; register DT-7):
//!
//! - [`Reader::column`] is the engine's inference over a preview,
//!   `pd.to_datetime(values, errors='coerce')`, with `utc=True` where the
//!   time column's choice reads it: pandas guesses one format from the
//!   first value that is not missing and reads every value by it, a value
//!   another way written reading as no time; when no format is guessed it
//!   reads each value on its own.
//! - [`Reader::each`] is the engine's reading of a SQL source's text time
//!   column, `pd.to_datetime(values, unit='ns', errors='coerce')`: each
//!   value on its own, a slash date month first unless its first field
//!   cannot be a month, a two-digit year placed in the hundred years
//!   around the clock's.
//!
//! A value either reads as an instant, or reads as no time, or is
//! undecided: its reading takes a rule outside the shapes, which is not
//! ported, and the whole reading is refused ([`Unread::Undecided`]) rather
//! than guessed. A value reads as no time here only where the reference's
//! every reading of it fails: under a guessed format, a value of another
//! layout of the shapes, or one with no digit, or with an ASCII letter the
//! format cannot hold; read on its own, a run of ASCII letters that is not
//! a word the date reader knows, or three runs of digits two of which are
//! long (D9). pandas' range ends (the year 0, more than six fractional
//! digits, an instant past the years 1 to 9999) are undecided. Every rule
//! here was held to pandas 3.0.6 over generated columns before it was
//! written (the T1.25 commit text records the runs).

use super::text_shapes::{Lead, Shape, TimeOfDay, Zone, shape};
use super::{Civil, Instant};

/// Why a column of text does not read as times here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unread {
    /// A value's reading takes a rule outside `date-rules.md`'s shapes.
    Undecided,
    /// Values with different offsets, or with and without one, where the
    /// reading is not to UTC: pandas raises rather than read them.
    MixedZones,
}

/// Seconds in a day: an offset must be less, either way.
const SECONDS_PER_DAY: i32 = 86_400;

/// The texts pandas reads as a missing time.
const MISSING: [&str; 7] = ["", "NaT", "nat", "NAT", "nan", "NaN", "NAN"];

/// The words the reference's date reader (dateutil's `parserinfo`, with the
/// relative days pandas and the client name) knows: a value holding no
/// other word may read as a date.
const WORDS: [&str; 68] = [
    "at",
    "on",
    "and",
    "ad",
    "m",
    "t",
    "of",
    "st",
    "nd",
    "rd",
    "th",
    "mon",
    "monday",
    "tue",
    "tuesday",
    "wed",
    "wednesday",
    "thu",
    "thursday",
    "fri",
    "friday",
    "sat",
    "saturday",
    "sun",
    "sunday",
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
    "h",
    "hour",
    "hours",
    "minute",
    "minutes",
    "s",
    "second",
    "seconds",
    "am",
    "pm",
    "a",
    "p",
    "utc",
    "gmt",
    "z",
    "now",
    "today",
    "tomorrow",
    "yesterday",
];

fn known(word: &str) -> bool {
    WORDS.contains(&word.to_lowercase().as_str())
}

/// Whether `value` holds no digit and a word the date reader does not
/// know, so cannot read as a date (T1.23's rule, words split at any
/// character that is not a letter).
pub fn cannot_be_a_date(value: &str) -> bool {
    if value.chars().any(char::is_numeric) {
        return false;
    }
    value
        .split(|c: char| !c.is_alphabetic())
        .filter(|word| !word.is_empty())
        .any(|word| !known(word))
}

/// Whether `value` holds a run of ASCII letters the date reader does not
/// know: it reads as no time, read on its own (the reader stops at an
/// unknown word, and pandas reads no zone named but UTC and GMT) or by a
/// format guessed from a value of the shapes (which holds no such word).
fn unknown_word(value: &str) -> bool {
    value
        .split(|c: char| !c.is_ascii_alphabetic())
        .filter(|word| !word.is_empty())
        .any(|word| !known(word))
}

/// Whether `value` is three runs of ASCII digits joined by one separator
/// (`/`, `-` or `.`), the first of at most four digits and two of three
/// digits or more: the date reader takes both long runs as years and
/// refuses it (`date-rules.md`, D9).
fn two_long_fields(value: &str) -> bool {
    let Some(sep) = value.chars().find(|c| matches!(c, '/' | '-' | '.')) else {
        return false;
    };
    let parts: Vec<&str> = value.split(sep).collect();
    let [a, b, c] = parts[..] else {
        return false;
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !(digits(a) && digits(b) && digits(c)) || a.len() > 4 {
        return false;
    }
    [a, b, c].iter().filter(|s| s.len() > 2).count() >= 2
}

/// Whether `value`, read on its own, reads as no time whatever the date
/// reader does with it.
fn certainly_unreadable(value: &str) -> bool {
    cannot_be_a_date(value) || unknown_word(value) || two_long_fields(value)
}

/// Whether `value` reads as no time whichever way pandas reads a column
/// holding it, alone or by any format a first value gives: it holds no
/// digit (every format reads a year, month and day from digits) and reads
/// as no time alone.
fn never_reads(value: &str) -> bool {
    !value.chars().any(char::is_numeric) && certainly_unreadable(value)
}

/// Whether a value might read with an offset, read some way: the column
/// it is in might then be refused for offsets mixed.
fn may_hold_offset(value: &str) -> bool {
    let lowered = value.to_ascii_lowercase();
    lowered.contains(['+', '-', 'z']) || lowered.contains("utc") || lowered.contains("gmt")
}

/// What a guessed format fixes of the values it reads: which field leads,
/// the separators, and the time's parts, as a shape writes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Layout {
    lead: Lead,
    separators: [char; 2],
    /// A day or month first: the year's digits.
    year_digits: Option<usize>,
    time: Option<TimeLayout>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TimeLayout {
    separator: char,
    second: bool,
    fraction: bool,
    /// A numeric zone, and whether one space comes before it.
    zone: Option<bool>,
}

/// A shape's layout, or `None` for a zone named, which no reader here
/// decides.
fn layout(shape: &Shape<'_>) -> Option<Layout> {
    let time = match &shape.time {
        None => None,
        Some(t) => Some(TimeLayout {
            separator: t.separator,
            second: t.second.is_some(),
            fraction: t.fraction.is_some(),
            zone: match t.zone {
                None => None,
                Some((_, Zone::Name(_))) => return None,
                Some((spaced, _)) => Some(spaced),
            },
        }),
    };
    Some(Layout {
        lead: shape.lead,
        separators: shape.separators,
        year_digits: (shape.lead == Lead::DayOrMonth).then(|| shape.fields[2].len()),
        time,
    })
}

/// A day or month first: which the guessed format takes first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    MonthFirst,
    DayFirst,
}

/// The format pandas guesses from a column's first value.
#[derive(Debug, Clone, Copy)]
struct Format {
    layout: Layout,
    order: Order,
}

fn number(digits: &str) -> u32 {
    digits.bytes().fold(0_u32, |n, b| {
        n.saturating_mul(10).saturating_add(u32::from(b - b'0'))
    })
}

/// The first and last instants pandas reads a text time between: the
/// years 1 to 9999.
fn in_years(instant: Instant) -> bool {
    let low = Instant::from_civil(&Civil::new(1, 1, 1, 0, 0, 0, 0).expect("a date"), 0);
    let high = Instant::from_civil(&Civil::new(10_000, 1, 1, 0, 0, 0, 0).expect("a date"), 0);
    (low..high).contains(&instant)
}

/// One value read: its instant and the offset it was written with, if
/// any.
type Read = Option<(Instant, Option<i32>)>;

/// The instant a value of the shapes names, read with `order` for a day or
/// month first and `year` for its year, or `None` when it names no date
/// (a month past 12, a day the month lacks, an hour past 23, a minute or
/// second past 59, a second past 61 where `leap` takes 60 and 61 as
/// pandas' formats do, rolling over, an offset of a day or more). Undecided at pandas' range ends.
fn read_shape(shape: &Shape<'_>, order: Order, year: i64, leap: bool) -> Result<Read, Unread> {
    let [a, b, c] = shape.fields;
    let (month, day) = match (shape.lead, order) {
        (Lead::Year, _) => (number(b), number(c)),
        (Lead::DayOrMonth, Order::MonthFirst) => (number(a), number(b)),
        (Lead::DayOrMonth, Order::DayFirst) => (number(b), number(a)),
    };
    let (hour, minute, second, fraction, zone) = match &shape.time {
        None => (0, 0, 0, None, None),
        Some(TimeOfDay {
            hour,
            minute,
            second,
            fraction,
            zone,
            ..
        }) => (
            number(hour),
            number(minute),
            second.map_or(0, number),
            *fraction,
            zone.map(|(_, z)| z),
        ),
    };
    if year == 0 {
        return Err(Unread::Undecided);
    }
    if fraction.is_some_and(|f| f.len() > 6) {
        return Err(Unread::Undecided);
    }
    if month == 0 || month > 12 || day > 31 || hour > 23 || minute > 59 {
        return Ok(None);
    }
    if second > if leap { 61 } else { 59 } {
        return Ok(None);
    }
    let Some(date) = Civil::new(year, month as u8, day as u8, 0, 0, 0, 0) else {
        return Ok(None);
    };
    let micros = fraction.map_or(0, |f| number(f) * 10_u32.pow(6 - f.len() as u32));
    let seconds = i128::from(hour) * 3_600 + i128::from(minute) * 60 + i128::from(second);
    let midnight = Instant::from_civil(&date, 0);
    let mut at = Instant::from_nanos(
        midnight.nanos() + seconds * 1_000_000_000 + i128::from(micros) * 1_000,
    );
    let offset = zone.and_then(|z| z.offset_seconds());
    if let Some(offset) = offset {
        // An offset of a whole day or more is no offset to Python's
        // `timezone`, which every reading of pandas builds: the value names
        // no time. Its minutes may pass 59 (`+01:99` is 2:39) below that.
        if offset.abs() >= SECONDS_PER_DAY {
            return Ok(None);
        }
        at = Instant::from_nanos(at.nanos() - i128::from(offset) * 1_000_000_000);
    }
    if !in_years(at) {
        return Err(Unread::Undecided);
    }
    Ok(Some((at, offset)))
}

/// Whether the zone writes minutes past 59 (`+01:99`): a first value so
/// written gives no format (pandas' guess writes the offset back as
/// `+0239` and finds it another text), `strptime`'s `%z` reads none, and
/// whether pandas' ISO parser reads one under a format is not ported.
fn minutes_past_59(shape: &Shape<'_>) -> bool {
    matches!(
        shape.time,
        Some(TimeOfDay {
            zone: Some((_, Zone::Compact { minute, .. } | Zone::Colon { minute, .. })),
            ..
        }) if minute > 59
    )
}

/// Whether the zone is a bare hour (`+HH`).
fn bare_hour(shape: &Shape<'_>) -> bool {
    matches!(
        shape.time,
        Some(TimeOfDay {
            zone: Some((_, Zone::Hours { .. })),
            ..
        })
    )
}

/// Whether pandas reads a layout's format with its ISO 8601 parser
/// (`format_is_iso`): a year first, with a zone only right after the
/// seconds; any other format with its `strptime`, whose `%z` takes no bare
/// hour.
fn iso(layout: &Layout) -> bool {
    layout.lead == Lead::Year
        && layout
            .time
            .is_none_or(|t| t.zone.is_none() || (t.second && t.zone == Some(false)))
}

fn mixed_separators(shape: &Shape<'_>) -> bool {
    shape.separators[0] != shape.separators[1]
}

fn long_fraction(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.iter().enumerate().any(|(i, &b)| {
        b == b'.'
            && bytes[i + 1..]
                .iter()
                .take_while(|b| b.is_ascii_digit())
                .count()
                > 6
    })
}

/// Reads text times for one run: the clock's year places a two-digit year.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reader {
    year: i64,
}

impl Reader {
    /// A reader for a run whose clock is in `year`.
    pub fn new(year: i64) -> Reader {
        Reader { year }
    }

    /// `pd.to_datetime(values, errors='coerce')`, or with `utc` its
    /// `utc=True` form: each value's instant, or `None` for a missing value
    /// or one that reads as no time.
    pub fn column(
        &self,
        values: &[Option<&str>],
        utc: bool,
    ) -> Result<Vec<Option<Instant>>, Unread> {
        if values
            .iter()
            .flatten()
            .any(|v| matches!(*v, "now" | "today"))
        {
            return Err(Unread::Undecided);
        }
        let Some(first) = values.iter().flatten().find(|v| !MISSING.contains(v)) else {
            return Ok(vec![None; values.len()]);
        };
        let format = self.guess(first)?;
        let mut read = Vec::with_capacity(values.len());
        for value in values {
            read.push(match value {
                None => None,
                Some(v) if MISSING.contains(v) => None,
                Some(v) => match &format {
                    Some(format) => self.under(v, format)?,
                    None => self.one(v)?,
                },
            });
        }
        finish(read, utc)
    }

    /// The format pandas guesses from `values`' first value when it puts the
    /// day before the month, which pandas warns of as it guesses it
    /// (`guess_datetime_format`'s "Parsing dates in ... format when
    /// dayfirst=False (the default) was specified"): a day or month first
    /// whose first field cannot be a month. Written as pandas writes it.
    pub fn day_first_format(&self, values: &[Option<&str>]) -> Option<String> {
        let first = values.iter().flatten().find(|v| !MISSING.contains(v))?;
        let format = self.guess(first).ok().flatten()?;
        if format.order != Order::DayFirst {
            return None;
        }
        let [a, b] = format.layout.separators;
        let mut text = format!("%d{a}%m{b}%Y");
        if let Some(time) = format.layout.time {
            text.push(time.separator);
            text.push_str("%H:%M");
            if time.second {
                text.push_str(":%S");
            }
            if time.fraction {
                text.push_str(".%f");
            }
            if let Some(spaced) = time.zone {
                if spaced {
                    text.push(' ');
                }
                text.push_str("%z");
            }
        }
        Some(text)
    }

    /// How many of `values` read as no time whichever way pandas reads
    /// them ([`never_reads`]), which bounds the share of them a reading
    /// that does not decide can read; `None` when a value that may read
    /// might hold an offset, so the reading might be refused instead.
    pub fn never_read(&self, values: &[Option<&str>]) -> Option<usize> {
        let mut never = 0;
        for value in values.iter().flatten() {
            if never_reads(value) {
                never += 1;
            } else if may_hold_offset(value) {
                return None;
            }
        }
        Some(never)
    }

    /// `pd.to_datetime(values, unit='ns', errors='coerce')`: each value on
    /// its own, as the engine reads a SQL source's text time column
    /// (register DT-7).
    pub fn each(&self, values: &[Option<&str>]) -> Result<Vec<Option<Instant>>, Unread> {
        if values
            .iter()
            .flatten()
            .any(|v| matches!(*v, "now" | "today"))
        {
            return Err(Unread::Undecided);
        }
        let read = values
            .iter()
            .map(|value| match value {
                None => Ok(None),
                Some(v) if MISSING.contains(v) => Ok(None),
                Some(v) => self.one(v),
            })
            .collect::<Result<Vec<_>, _>>()?;
        finish(read, false)
    }

    /// The format pandas guesses from `first`, `None` when it guesses none
    /// and reads each value on its own.
    fn guess(&self, first: &str) -> Result<Option<Format>, Unread> {
        let Some(shape) = shape(first) else {
            // A value the date reader cannot read gives no format.
            return if certainly_unreadable(first) {
                Ok(None)
            } else {
                Err(Unread::Undecided)
            };
        };
        let layout = layout(&shape).ok_or(Unread::Undecided)?;
        if shape
            .time
            .and_then(|t| t.fraction)
            .is_some_and(|f| f.len() > 6)
            || mixed_separators(&shape)
        {
            return Err(Unread::Undecided);
        }
        match shape.lead {
            Lead::Year => {
                let year = i64::from(number(shape.fields[0]));
                if read_shape(&shape, Order::MonthFirst, year, false)?.is_none() {
                    // The date reader does not read it: no format.
                    return Ok(None);
                }
                // A bare hour's zone only an ISO format reads; minutes past
                // 59 give none.
                if (bare_hour(&shape) && !iso(&layout)) || minutes_past_59(&shape) {
                    return Ok(None);
                }
                Ok(Some(Format {
                    layout,
                    order: Order::MonthFirst,
                }))
            }
            Lead::DayOrMonth => {
                // No format takes a year of two or three digits, nor a
                // bare hour's zone or minutes past 59.
                if shape.fields[2].len() != 4 || bare_hour(&shape) || minutes_past_59(&shape) {
                    return Ok(None);
                }
                let order = if number(shape.fields[0]) > 12 {
                    Order::DayFirst
                } else {
                    Order::MonthFirst
                };
                let year = i64::from(number(shape.fields[2]));
                if read_shape(&shape, order, year, false)?.is_none() {
                    return Ok(None);
                }
                Ok(Some(Format { layout, order }))
            }
        }
    }

    /// `value` read by the guessed `format`.
    fn under(&self, value: &str, format: &Format) -> Result<Read, Unread> {
        if let Some(shape) = shape(value) {
            match layout(&shape) {
                Some(layout) if layout == format.layout => {
                    if mixed_separators(&shape) {
                        return Err(Unread::Undecided);
                    }
                    // A bare hour's zone only an ISO format reads; minutes
                    // past 59 `strptime` does not read, and an ISO format's
                    // reading of them is not ported.
                    if bare_hour(&shape) && !iso(&layout) {
                        return Ok(None);
                    }
                    if minutes_past_59(&shape) {
                        return if iso(&layout) {
                            Err(Unread::Undecided)
                        } else {
                            Ok(None)
                        };
                    }
                    let year = match shape.lead {
                        Lead::Year => number(shape.fields[0]),
                        Lead::DayOrMonth => number(shape.fields[2]),
                    };
                    return read_shape(&shape, format.order, i64::from(year), true);
                }
                // Another layout of the shapes: the format does not read it.
                Some(_) => return Ok(None),
                None => {}
            }
        }
        if long_fraction(value) {
            return Err(Unread::Undecided);
        }
        if !value.bytes().any(|b| b.is_ascii_digit()) {
            return Ok(None);
        }
        // A letter the format cannot hold: only `T` and `Z` (in any case)
        // are in one.
        if value
            .bytes()
            .any(|b| b.is_ascii_alphabetic() && !matches!(b, b'T' | b't' | b'Z' | b'z'))
        {
            return Ok(None);
        }
        Err(Unread::Undecided)
    }

    /// `value` read on its own.
    fn one(&self, value: &str) -> Result<Read, Unread> {
        let Some(shape) = shape(value) else {
            return if certainly_unreadable(value) {
                Ok(None)
            } else {
                Err(Unread::Undecided)
            };
        };
        if matches!(
            shape.time,
            Some(TimeOfDay {
                zone: Some((_, Zone::Name(_))),
                ..
            })
        ) || mixed_separators(&shape)
        {
            return Err(Unread::Undecided);
        }
        match shape.lead {
            Lead::Year => {
                let year = i64::from(number(shape.fields[0]));
                read_shape(&shape, Order::MonthFirst, year, false)
            }
            Lead::DayOrMonth => {
                let year_text = shape.fields[2];
                if year_text.len() == 3 {
                    return Err(Unread::Undecided);
                }
                let (first, second) = (number(shape.fields[0]), number(shape.fields[1]));
                if first > 31 {
                    return Err(Unread::Undecided);
                }
                if first > 12 && second > 12 {
                    return Ok(None);
                }
                let order = if first > 12 {
                    Order::DayFirst
                } else {
                    Order::MonthFirst
                };
                let mut year = i64::from(number(year_text));
                if year_text.len() == 2 {
                    // The hundred years around the clock's year: 50 before
                    // it to 49 after.
                    year += self.year.div_euclid(100) * 100;
                    if year >= self.year + 50 {
                        year -= 100;
                    } else if year < self.year - 50 {
                        year += 100;
                    }
                }
                read_shape(&shape, order, year, false)
            }
        }
    }
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
            .map_err(|unread| match unread {
                Unread::Undecided => sdm_schema::infer::TextUnread::Undecided,
                Unread::MixedZones => sdm_schema::infer::TextUnread::MixedZones,
            })
    }

    fn never_read(&self, values: &[Option<&str>]) -> Option<usize> {
        Reader::never_read(self, values)
    }

    fn day_first_format(&self, values: &[Option<&str>]) -> Option<String> {
        Reader::day_first_format(self, values)
    }
}

/// The instants read, refusing offsets mixed where they are not read to
/// UTC, as pandas does.
fn finish(read: Vec<Read>, utc: bool) -> Result<Vec<Option<Instant>>, Unread> {
    if !utc {
        let mut offsets = read.iter().flatten().map(|(_, offset)| *offset);
        if let Some(first) = offsets.next()
            && offsets.any(|o| o != first)
        {
            return Err(Unread::MixedZones);
        }
    }
    Ok(read.into_iter().map(|r| r.map(|(at, _)| at)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(values: &[&str], utc: bool) -> Result<Vec<Option<String>>, Unread> {
        let values: Vec<Option<&str>> = values.iter().map(|v| Some(*v)).collect();
        Reader::new(2026).column(&values, utc).map(|r| {
            r.into_iter()
                .map(|at| at.map(|at| at.civil(0).to_string()))
                .collect()
        })
    }

    fn each(values: &[&str]) -> Result<Vec<Option<String>>, Unread> {
        let values: Vec<Option<&str>> = values.iter().map(|v| Some(*v)).collect();
        Reader::new(2026).each(&values).map(|r| {
            r.into_iter()
                .map(|at| at.map(|at| at.civil(0).to_string()))
                .collect()
        })
    }

    fn at(text: &str) -> Option<String> {
        Some(format!("{text}.000000000"))
    }

    #[test]
    fn a_column_reads_by_the_format_its_first_value_gives() {
        assert_eq!(
            read(
                &[
                    "2024-01-05",
                    "2024-1-5x",
                    "not a date",
                    "2024/01/05",
                    "15/03/2024"
                ],
                false
            ),
            Ok(vec![at("2024-01-05T00:00:00"), None, None, None, None])
        );
        assert_eq!(
            read(&["01/02/2024", "13/02/2024", "1/3/2024"], false),
            Ok(vec![
                at("2024-01-02T00:00:00"),
                None,
                at("2024-01-03T00:00:00")
            ])
        );
        assert_eq!(
            read(&["15/03/2024", "01/02/2024"], false),
            Ok(vec![at("2024-03-15T00:00:00"), at("2024-02-01T00:00:00")])
        );
        // A first value that is no date gives no format: each on its own.
        assert_eq!(
            read(&["13/13/2020", "11/12/2020"], false),
            Ok(vec![None, at("2020-11-12T00:00:00")])
        );
        // A second of 60 rolls over under a format, and is no time alone.
        assert_eq!(
            read(&["2024-01-05 10:00:00", "2024-12-31 23:59:60"], false),
            Ok(vec![at("2024-01-05T10:00:00"), at("2025-01-01T00:00:00")])
        );
        assert_eq!(each(&["2024-12-31 23:59:60"]), Ok(vec![None]));
    }

    #[test]
    fn offsets_read_to_utc_and_mixed_ones_are_refused_unless_to_utc() {
        let values = ["2024-01-05 10:00+01:00", "2024-01-05 10:00+02:00"];
        assert_eq!(read(&values, false), Err(Unread::MixedZones));
        assert_eq!(
            read(&values, true),
            Ok(vec![at("2024-01-05T09:00:00"), at("2024-01-05T08:00:00")])
        );
        assert_eq!(
            read(&["2024-01-05 10:00+01:00", "2024-01-05 10:00"], false),
            Ok(vec![at("2024-01-05T09:00:00"), None])
        );
        assert_eq!(
            each(&["13/02/2020 00:00 +0100", "15/02/2020"]),
            Err(Unread::MixedZones)
        );
    }

    #[test]
    fn an_offset_of_a_day_or_more_names_no_time() {
        for zone in ["+24:00", "+2360", "-25:00", "+99:00", "+24"] {
            let value = format!("2024-01-01T01:02:03{zone}");
            assert_eq!(read(&[&value], false), Ok(vec![None]), "{zone}");
            assert_eq!(read(&[&value], true), Ok(vec![None]), "{zone}");
            assert_eq!(each(&[&value]), Ok(vec![None]), "{zone}");
        }
        // Below a day, minutes past 59 count on: +01:99 is 2:39 east.
        assert_eq!(
            read(&["2024-01-01T01:02:03+01:99"], true),
            Ok(vec![at("2023-12-31T22:23:03")])
        );
        assert_eq!(
            read(
                &["2024-01-01T01:02:03-23:59", "2024-01-01T01:02:03-00"],
                true
            ),
            Ok(vec![at("2024-01-02T01:01:03"), at("2024-01-01T01:02:03")])
        );
    }

    #[test]
    fn codes_read_as_no_time_and_other_text_is_undecided() {
        for code in [
            "AW00000001",
            "10-4020-000676",
            "SO43659",
            "PO522145787",
            "105041Vi84182",
            "4911-403C-98",
            "BK-R93R-62",
            "not a date",
        ] {
            assert_eq!(read(&[code], false), Ok(vec![None]), "{code}");
            assert_eq!(each(&[code]), Ok(vec![None]), "{code}");
        }
        for text in [
            "March 3 2020",
            "20240101",
            "2024-1-5",
            " 2024-01-05",
            "2024-01-05 10:00 UTC",
            "now",
            "2024-01-05 10:00:00.1234567",
            "0000-01-05",
            "1/5/124",
        ] {
            assert_eq!(read(&[text], false), Err(Unread::Undecided), "{text}");
            assert_eq!(each(&[text]), Err(Unread::Undecided), "{text}");
        }
    }

    #[test]
    fn values_alone_read_month_first_and_two_digit_years_around_the_clock() {
        assert_eq!(
            each(&[
                "01/02/2024",
                "15/03/2024",
                "1/5/24",
                "1/5/76",
                "1/5/75",
                "13/13/2020"
            ]),
            Ok(vec![
                at("2024-01-02T00:00:00"),
                at("2024-03-15T00:00:00"),
                at("2024-01-05T00:00:00"),
                at("1976-01-05T00:00:00"),
                at("2075-01-05T00:00:00"),
                None
            ])
        );
        assert_eq!(
            read(&["1/5/24", "15/3/24"], false),
            Ok(vec![at("2024-01-05T00:00:00"), at("2024-03-15T00:00:00")])
        );
        assert_eq!(
            read(&["nan", "", "2024-01-05"], false),
            Ok(vec![None, None, at("2024-01-05T00:00:00")])
        );
    }

    #[test]
    fn the_digit_free_rule_is_t123s() {
        assert!(cannot_be_a_date("US"));
        assert!(cannot_be_a_date("in"));
        assert!(!cannot_be_a_date("March"));
        assert!(!cannot_be_a_date("now"));
        assert!(!cannot_be_a_date("2024-01-01"));
        assert!(!cannot_be_a_date("a"));
        assert!(!cannot_be_a_date(" "));
    }
}
