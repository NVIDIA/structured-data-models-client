// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `pandas.to_datetime` over a column of text, as pandas 3.0.6 runs it: the
//! format guessed from the first value, `array_strptime` with that format,
//! or, without one, each value read as ISO 8601 and then by `dateutil`; the
//! resolution inferred from what was read; and the zones the values carry,
//! which must agree unless everything is read to UTC (DV20).

use super::civil::{self, Wall};
use super::dateutil;
use super::iso::{self, Unit};
use super::strftime::{Aware, strftime};
use super::strptime::{self, Flavor, Tz};
use crate::py;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimestampSeriesUnits([arrow_schema::TimeUnit; 4]);

impl Default for TimestampSeriesUnits {
    fn default() -> Self {
        use arrow_schema::TimeUnit;
        Self([
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ])
    }
}

impl TimestampSeriesUnits {
    pub fn from_json(value: &serde_json::Value) -> Result<Self, String> {
        let names: [String; 4] = serde_json::from_value(value.clone())
            .map_err(|_| "timestamp Series facts must be four unit names".to_string())?;
        Self::from_names(names.each_ref().map(String::as_str))
    }

    pub fn from_names(names: [&str; 4]) -> Result<Self, String> {
        use arrow_schema::TimeUnit;
        let mut units = Self::default();
        for (position, name) in names.into_iter().enumerate() {
            let (rank, unit) = match name {
                "s" => (0, TimeUnit::Second),
                "ms" => (1, TimeUnit::Millisecond),
                "us" => (2, TimeUnit::Microsecond),
                "ns" => (3, TimeUnit::Nanosecond),
                _ => return Err("timestamp Series facts must name s, ms, us or ns".into()),
            };
            if rank < position {
                return Err("timestamp Series facts must not lose scalar precision".into());
            }
            units.0[position] = unit;
        }
        Ok(units)
    }

    pub fn names(&self) -> [&'static str; 4] {
        self.0.map(|unit| match unit {
            arrow_schema::TimeUnit::Second => "s",
            arrow_schema::TimeUnit::Millisecond => "ms",
            arrow_schema::TimeUnit::Microsecond => "us",
            arrow_schema::TimeUnit::Nanosecond => "ns",
        })
    }

    pub fn for_scalar(&self, unit: &str) -> Option<&'static str> {
        let position = match unit {
            "s" => 0,
            "ms" => 1,
            "us" => 2,
            "ns" => 3,
            _ => return None,
        };
        Some(self.names()[position])
    }
}

/// What a conversion takes from the Python process pandas runs in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Host {
    /// The local year, which `dateutil` reads two-digit years against.
    pub year: i64,
    /// Whether `strftime` writes `%Y` below 1000 with four digits.
    pub pads_year: bool,
    /// Today's local date, the date a value of only a time is read on.
    pub today: (i64, u32, u32),
    /// `time.tzname`: names `dateutil` reads as the local zone.
    pub tznames: Vec<String>,
    pub timestamp_series_units: TimestampSeriesUnits,
    pub numpy_text_prefix: bool,
}

impl Host {
    /// What the core assumes without a Python process to ask: the year and
    /// date in UTC, `%Y` unpadded, and UTC as the local zone.
    pub fn assumed() -> Host {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs() as i64);
        let (y, m, d) = civil::civil_from_days(now.div_euclid(86_400));
        Host {
            year: y,
            pads_year: false,
            today: (y, m, d),
            tznames: vec!["UTC".to_string(), "UTC".to_string()],
            timestamp_series_units: TimestampSeriesUnits::default(),
            numpy_text_prefix: false,
        }
    }
}

/// An exception a conversion raises, by its Python class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyErr {
    pub class: &'static str,
    pub message: String,
}

impl PyErr {
    pub fn value(message: impl Into<String>) -> Self {
        PyErr {
            class: "ValueError",
            message: message.into(),
        }
    }

    fn date_parse(message: impl Into<String>) -> Self {
        PyErr {
            class: "DateParseError",
            message: message.into(),
        }
    }

    /// Whether `except (ValueError, TypeError, OverflowError)` catches it.
    pub fn is_value_type_or_overflow(&self) -> bool {
        matches!(
            self.class,
            "ValueError"
                | "DateParseError"
                | "OutOfBoundsDatetime"
                | "TypeError"
                | "OverflowError"
                | "ParserError"
        )
    }
}

/// The zone of a converted column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnTz {
    Naive,
    /// `datetime.timezone` of this many seconds; zero is UTC.
    Fixed(i64),
}

/// A converted column: `datetime64[unit]` or `datetime64[unit, tz]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Converted {
    pub unit: Unit,
    pub tz: ColumnTz,
    /// Nanoseconds since the epoch: UTC where zoned, wall time otherwise.
    pub values: Vec<Option<i128>>,
}

/// `str(datetime.timezone(timedelta(seconds=offset)))`.
pub fn zone_text(offset: i64) -> String {
    if offset == 0 {
        return "UTC".to_string();
    }
    let sign = if offset < 0 { '-' } else { '+' };
    let total = offset.unsigned_abs();
    let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
    if s != 0 {
        format!("UTC{sign}{h:02}:{m:02}:{s:02}")
    } else {
        format!("UTC{sign}{h:02}:{m:02}")
    }
}

fn unit_text(unit: Unit) -> &'static str {
    match unit {
        Unit::S => "s",
        Unit::Ms => "ms",
        Unit::Us => "us",
        _ => "ns",
    }
}

impl Converted {
    /// `str(series.dtype)`.
    pub fn dtype(&self) -> String {
        match self.tz {
            ColumnTz::Naive => format!("datetime64[{}]", unit_text(self.unit)),
            ColumnTz::Fixed(offset) => {
                format!(
                    "datetime64[{}, {}]",
                    unit_text(self.unit),
                    zone_text(offset)
                )
            }
        }
    }

    fn offset(&self) -> i64 {
        match self.tz {
            ColumnTz::Naive => 0,
            ColumnTz::Fixed(o) => o,
        }
    }

    /// The wall time of value `i` in the column's zone.
    pub fn wall(&self, i: usize) -> Option<Wall> {
        let v = self.values[i]?;
        Some(Wall::from_epoch_nanos(
            v + i128::from(self.offset()) * 1_000_000_000,
        ))
    }

    /// `Timestamp.isoformat()` of value `i`, or `None` for NaT.
    pub fn isoformat(&self, i: usize) -> Option<String> {
        let w = self.wall(i)?;
        let mut out = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            w.year, w.month, w.day, w.hour, w.minute, w.second
        );
        let micro = w.nano / 1000;
        let nano = w.nano % 1000;
        if micro != 0 {
            out.push_str(&format!(".{micro:06}"));
            if nano != 0 {
                out.push_str(&format!("{nano:03}"));
            }
        } else if nano != 0 {
            out.push_str(&format!(".{:09}", w.nano));
        }
        if let ColumnTz::Fixed(offset) = self.tz {
            let sign = if offset < 0 { '-' } else { '+' };
            let total = offset.unsigned_abs();
            let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
            out.push_str(&format!("{sign}{h:02}:{m:02}"));
            if s != 0 {
                out.push_str(&format!(":{s:02}"));
            }
        }
        Some(out)
    }
}

/// `nat_strings`.
fn is_nat_string(text: &str) -> bool {
    matches!(text, "NaT" | "nat" | "NAT" | "nan" | "NaN" | "NAN")
}

/// `first_non_null`: the first value not null, empty, a NaT spelling, `now`
/// or `today`.
pub fn first_non_null(values: &[Option<String>]) -> Option<usize> {
    values.iter().position(|v| match v {
        None => false,
        Some(text) => !(text.is_empty() || is_nat_string(text) || text == "now" || text == "today"),
    })
}

/// One value read: its wall time, the offset it was written with, and the
/// resolution it was read at.
#[derive(Clone, Debug)]
struct Item {
    wall: Wall,
    /// The UTC offset in seconds, where the value named one.
    offset: Option<i64>,
    unit: Unit,
}

fn at_least_us(unit: Unit) -> Unit {
    let unit = unit.supported();
    if unit < Unit::Us { Unit::Us } else { unit }
}

fn iso_wall(dts: &iso::Dts) -> Wall {
    Wall {
        year: dts.year,
        month: dts.month,
        day: dts.day,
        hour: dts.hour,
        minute: dts.min,
        second: dts.sec,
        nano: dts.us * 1000 + dts.ps / 1000,
    }
}

/// `pandas._libs.tslibs.parsing.dateutil_parse`.
fn dateutil_parse(
    text: &str,
    default: Wall,
    dayfirst: bool,
    yearfirst: bool,
    host: &Host,
) -> Result<(Aware, Unit, u32), PyErr> {
    let Some(res) = dateutil::parse(text, dayfirst, yearfirst, host.year) else {
        return Err(PyErr::date_parse(format!(
            "Unknown datetime string format, unable to parse: {text}"
        )));
    };
    let mut wall = default;
    let mut reso: Option<&str> = None;
    let mut micro_value = 0;
    if let Some(y) = res.year {
        wall.year = y;
        reso = Some("year");
    }
    let mut month = i64::from(default.month);
    if let Some(m) = res.month {
        month = m;
        reso = Some("month");
    }
    let mut day = i64::from(default.day);
    if let Some(d) = res.day {
        day = d;
        reso = Some("day");
    }
    let mut hour = i64::from(default.hour);
    if let Some(h) = res.hour {
        hour = h;
        reso = Some("hour");
    }
    let mut minute = i64::from(default.minute);
    if let Some(m) = res.minute {
        minute = m;
        reso = Some("minute");
    }
    let mut second = i64::from(default.second);
    if let Some(s) = res.second {
        second = s;
        reso = Some("second");
    }
    let mut micro = i64::from(default.microsecond());
    if let Some(us) = res.microsecond {
        micro = us;
        micro_value = us;
        reso = Some("microsecond");
    }
    let Some(mut reso) = reso else {
        return Err(PyErr::date_parse(format!(
            "Unable to parse datetime string: {text}"
        )));
    };
    let mut nanos = 0u32;
    if reso == "microsecond" && micro_value % 1000 == 0 {
        let (found, extra) = find_subsecond_reso(text);
        reso = found;
        nanos = extra;
    }
    let valid = (1..=9999).contains(&wall.year)
        && (1..=12).contains(&month)
        && day >= 1
        && day <= i64::from(civil::days_in_month(wall.year, month as u32))
        && (0..24).contains(&hour)
        && (0..60).contains(&minute)
        && (0..60).contains(&second)
        && (0..1_000_000).contains(&micro);
    if !valid {
        return Err(PyErr::date_parse(format!("invalid date: {text}")));
    }
    wall.month = month as u32;
    wall.day = day as u32;
    wall.hour = hour as u32;
    wall.minute = minute as u32;
    wall.second = second as u32;
    wall.nano = micro as u32 * 1000;
    if res.weekday.is_some() && res.day.is_none_or(|d| d == 0) {
        return Err(PyErr::value(
            "Parsing datetimes with weekday but no day information is not supported",
        ));
    }
    let tz = if let Some(name) = res
        .tzname
        .as_deref()
        .filter(|n| !n.is_empty() && host.tznames.iter().any(|t| t == n))
    {
        if name != "UTC" {
            return Err(PyErr::value(format!(
                "Parsing '{name}' as tzlocal (dependent on system timezone) is no longer supported."
            )));
        }
        Some((0, Some("UTC".to_string())))
    } else if res.tzoffset == Some(0) {
        Some((0, Some("UTC".to_string())))
    } else if let Some(offset) = res.tzoffset {
        if offset.abs() >= 86_400 {
            return Err(PyErr::value(format!(
                "Parsed string \"{text}\" gives an invalid tzoffset"
            )));
        }
        Some((offset, res.tzname.clone()))
    } else if res.tzname.is_some() {
        return Err(PyErr::value(format!(
            "Parsed string \"{text}\" included an un-recognized timezone"
        )));
    } else {
        None
    };
    let unit = match reso {
        "year" => Unit::Y,
        "month" => Unit::M,
        "day" => Unit::D,
        "hour" => Unit::H,
        "minute" => Unit::Min,
        "second" => Unit::S,
        "millisecond" => Unit::Ms,
        "microsecond" => Unit::Us,
        _ => Unit::Ns,
    };
    Ok((Aware { wall, tz }, unit, nanos))
}

/// `_find_subsecond_reso`.
fn find_subsecond_reso(text: &str) -> (&'static str, u32) {
    let chars: Vec<char> = text.chars().collect();
    // \d:\d{2}:\d{2}\.(?P<frac>\d+)
    for i in 0..chars.len() {
        let d = |k: usize| chars.get(k).is_some_and(|&c| py::is_digit(c));
        if d(i)
            && chars.get(i + 1) == Some(&':')
            && d(i + 2)
            && d(i + 3)
            && chars.get(i + 4) == Some(&':')
            && d(i + 5)
            && d(i + 6)
            && chars.get(i + 7) == Some(&'.')
            && d(i + 8)
        {
            let mut end = i + 8;
            while d(end) {
                end += 1;
            }
            let frac: String = chars[i + 8..end].iter().collect();
            let n = frac.chars().count();
            if n <= 3 {
                return ("millisecond", 0);
            }
            if n > 6 {
                let tail: String = frac.chars().skip(6).collect();
                if tail.chars().all(|c| c == '0') {
                    return ("nanosecond", 0);
                }
                let digits: String = if n <= 9 {
                    format!("{tail}{}", "0".repeat(9 - n))
                } else {
                    frac.chars().skip(6).take(3).collect()
                };
                return ("nanosecond", digits.parse().unwrap_or(0));
            }
            return ("microsecond", 0);
        }
    }
    ("second", 0)
}

/// `_does_string_look_like_datetime`.
fn looks_like_datetime(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return true;
    }
    if bytes[0] == b'0' {
        return true;
    }
    if matches!(text, "a" | "A" | "m" | "M" | "p" | "P" | "t" | "T") {
        return false;
    }
    // xstrtod(text, '.', 'e', no separator, skip trailing spaces)
    let space = |b: u8| matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let mut p = 0;
    while p < bytes.len() && space(bytes[p]) {
        p += 1;
    }
    let start = p;
    if p < bytes.len() && (bytes[p] == b'-' || bytes[p] == b'+') {
        p += 1;
    }
    let mut digits = 0;
    while p < bytes.len() && bytes[p].is_ascii_digit() {
        p += 1;
        digits += 1;
    }
    if p < bytes.len() && bytes[p] == b'.' {
        p += 1;
        while p < bytes.len() && bytes[p].is_ascii_digit() {
            p += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return true;
    }
    let mantissa_end = p;
    let mut exp_text = String::new();
    if p < bytes.len() && (bytes[p] == b'e' || bytes[p] == b'E') {
        let mut q = p + 1;
        if q < bytes.len() && (bytes[q] == b'-' || bytes[q] == b'+') {
            q += 1;
        }
        let exp_start = q;
        while q < bytes.len() && bytes[q].is_ascii_digit() {
            q += 1;
        }
        if q > exp_start {
            exp_text = text[p + 1..q].to_string();
            p = q;
        }
    }
    while p < bytes.len() && space(bytes[p]) {
        p += 1;
    }
    if p != bytes.len() {
        return true;
    }
    let number = format!(
        "{}{}",
        &text[start..mantissa_end],
        if exp_text.is_empty() {
            String::new()
        } else {
            format!("e{exp_text}")
        }
    );
    let value: f64 = number.trim_start_matches('+').parse().unwrap_or(0.0);
    value >= 1000.0
}

/// `_does_string_look_like_time`.
fn looks_like_time(text: &str) -> bool {
    let b = text.as_bytes();
    if b.len() < 4 {
        return false;
    }
    let digit = |c: u8| -> i32 {
        if c.is_ascii_digit() {
            i32::from(c - b'0')
        } else {
            -1
        }
    };
    let two = |i: usize| -> i32 {
        let a = digit(b[i]);
        let c = digit(*b.get(i + 1).unwrap_or(&0));
        if a < 0 || c < 0 { -1000 } else { a * 10 + c }
    };
    let (hour, minute) = if b[1] == b':' {
        (digit(b[0]), two(2))
    } else if b[2] == b':' {
        (two(0), two(3))
    } else {
        (-1, -1)
    };
    (0..=23).contains(&hour) && (0..=59).contains(&minute)
}

/// `_parse_delimited_date`.
fn parse_delimited_date(text: &str, dayfirst: bool) -> Result<Option<Wall>, PyErr> {
    let b = text.as_bytes();
    let delim = |i: usize| b.get(i).is_some_and(|c| b" /-.".contains(c));
    let digit = |i: usize| -> i32 {
        match b.get(i) {
            Some(c) if c.is_ascii_digit() => i32::from(c - b'0'),
            _ => -10,
        }
    };
    let two = |i: usize| -> i32 {
        digit(i) * 10
            + if b.get(i + 1).is_some_and(u8::is_ascii_digit) {
                digit(i + 1)
            } else {
                -100
            }
    };
    let four = |i: usize| -> i32 {
        let d = |k: usize, penalty: i32| {
            if b.get(k).is_some_and(u8::is_ascii_digit) {
                digit(k)
            } else {
                penalty
            }
        };
        d(i, -10) * 1000 + d(i + 1, -100) * 100 + d(i + 2, -1000) * 10 + d(i + 3, -10000)
    };
    let (month, day, year, can_swap);
    let len = b.len();
    if len == 10 && delim(2) && delim(5) {
        month = two(0);
        day = two(3);
        year = four(6);
        can_swap = true;
    } else if len == 9 && delim(1) && delim(4) {
        month = digit(0);
        day = two(2);
        year = four(5);
        can_swap = true;
    } else if len == 9 && delim(2) && delim(4) {
        month = two(0);
        day = digit(3);
        year = four(5);
        can_swap = true;
    } else if len == 8 && delim(1) && delim(3) {
        month = digit(0);
        day = digit(2);
        year = four(4);
        can_swap = true;
    } else if len == 7 && delim(2) {
        if b[2] == b'.' {
            return Ok(None);
        }
        month = two(0);
        day = 1;
        year = four(3);
        can_swap = false;
    } else {
        return Ok(None);
    }
    if month < 0 || day < 0 || year < 1000 {
        return Ok(None);
    }
    let (mut month, mut day) = (month, day);
    if (1..=31).contains(&month) && (1..=31).contains(&day) && (month <= 12 || day <= 12) {
        if (month > 12 || (day <= 12 && dayfirst)) && can_swap {
            std::mem::swap(&mut day, &mut month);
        }
        let (y, m, d) = (i64::from(year), month as u32, day as u32);
        if !civil::valid_date(y, m, d) {
            return Err(PyErr::value("day is out of range for month"));
        }
        return Ok(Some(Wall::date(y, m, d)));
    }
    Err(PyErr::date_parse(format!(
        "Invalid date specified ({month}/{day})"
    )))
}

/// `_parse_dateabbr_string` with no frequency; `Ok(None)` where it raises a
/// `ValueError` the caller passes over.
fn parse_dateabbr(text: &str) -> Result<Option<(Wall, Unit)>, PyErr> {
    if is_nat_string(text) {
        return Ok(None);
    }
    let upper = text.to_uppercase();
    let len = upper.chars().count();
    if len == 4
        && let Some(year) = py::int(&upper)
        && (1..=9999).contains(&year)
    {
        return Ok(Some((Wall::date(year, 1, 1), Unit::Y)));
    }
    if (4..=7).contains(&len) {
        let chars: Vec<char> = upper.chars().collect();
        if let Some(i) = (1..6.min(len)).find(|&k| chars[k] == 'Q') {
            let s = |a: usize, b: usize| -> String { chars[a..b].iter().collect() };
            let mut quarter: Option<i64> = None;
            let mut year: Option<i64> = None;
            let ok = if i == 1 {
                let q = chars[0].to_digit(10).map(i64::from);
                quarter = Some(q.unwrap_or(-10));
                if len == 4 || (len == 5 && chars[i + 1] == '-') {
                    year = py::int(&s(len - 2, len)).map(|y| 2000 + y);
                    year.is_some()
                } else if len == 6 || (len == 7 && chars[i + 1] == '-') {
                    year = py::int(&s(len - 4, len));
                    year.is_some()
                } else {
                    false
                }
            } else if i == 2 || i == 3 {
                if len == 4 || (len == 5 && chars[i - 1] == '-') {
                    quarter = Some(chars[len - 1].to_digit(10).map_or(-10, i64::from));
                    year = py::int(&s(0, 2)).map(|y| 2000 + y);
                    year.is_some()
                } else {
                    false
                }
            } else if len == 6 || (len == 7 && chars[i - 1] == '-') {
                quarter = Some(chars[len - 1].to_digit(10).map_or(-10, i64::from));
                year = py::int(&s(0, 4));
                year.is_some()
            } else {
                false
            };
            if ok {
                let q = quarter.unwrap_or(-10);
                if !(1..=4).contains(&q) {
                    return Err(PyErr::date_parse(format!(
                        "Incorrect quarterly string is given, quarter must be between 1 and 4: {upper}"
                    )));
                }
                let y = year.unwrap_or(0);
                let m = ((q - 1) * 3 + 1) as u32;
                if (1..=9999).contains(&y) {
                    return Ok(Some((Wall::date(y, m, 1), Unit::M)));
                }
                return Ok(None);
            }
        }
    }
    for pat in ["%Y-%m", "%b %Y", "%b-%Y"] {
        if let Ok(compiled) = strptime::compile(pat, Flavor::Stdlib)
            && let Ok(read) = strptime::parse_with_format(&compiled, &upper, pat, true)
        {
            return Ok(Some((
                Wall::date(read.wall.year, read.wall.month, 1),
                Unit::M,
            )));
        }
    }
    Ok(None)
}

/// `parse_datetime_string`: the datetime, its resolution and any
/// nanoseconds beyond it.
fn parse_datetime_string(
    text: &str,
    dayfirst: bool,
    yearfirst: bool,
    host: &Host,
) -> Result<(Aware, Unit, u32), PyErr> {
    if !looks_like_datetime(text) {
        return Err(PyErr::value(format!(
            "Given date string \"{text}\" not likely a datetime"
        )));
    }
    if looks_like_time(text) {
        let (y, m, d) = host.today;
        return dateutil_parse(text, Wall::date(y, m, d), dayfirst, yearfirst, host);
    }
    if let Some(wall) = parse_delimited_date(text, dayfirst)? {
        return Ok((Aware { wall, tz: None }, Unit::D, 0));
    }
    if let Some((wall, unit)) = parse_dateabbr(text)? {
        return Ok((Aware { wall, tz: None }, unit, 0));
    }
    dateutil_parse(text, Wall::date(1, 1, 1), dayfirst, yearfirst, host)
}

/// `convert_str_to_tsobject`, for a value that is not empty or a NaT
/// spelling.
fn convert_str(text: &str, dayfirst: bool, host: &Host) -> Result<Item, PyErr> {
    if !dayfirst && let Some(found) = iso::parse(text, None, true) {
        let unit = at_least_us(found.unit);
        return Ok(Item {
            wall: iso_wall(&found.dts),
            offset: found.offset.map(|m| i64::from(m) * 60),
            unit,
        });
    }
    let (aware, unit, nanos) = parse_datetime_string(text, dayfirst, false, host)?;
    let mut wall = aware.wall;
    wall.nano += nanos;
    Ok(Item {
        wall,
        offset: aware.tz.map(|(o, _)| o),
        unit: at_least_us(unit),
    })
}

/// What a conversion found across its values: `DatetimeParseState`.
#[derive(Default)]
struct State {
    unit: Option<Unit>,
    found_naive_str: bool,
    found_aware_str: bool,
    offsets: Vec<Option<i64>>,
}

impl State {
    fn note(&mut self, item: &Item) {
        self.unit = Some(self.unit.map_or(item.unit, |u| u.max(item.unit)));
        match item.offset {
            Some(o) => {
                self.found_aware_str = true;
                if !self.offsets.contains(&Some(o)) {
                    self.offsets.push(Some(o));
                }
            }
            None => {
                self.found_naive_str = true;
                if !self.offsets.contains(&None) {
                    self.offsets.push(None);
                }
            }
        }
    }

    /// `check_for_mixed_inputs`: the column's zone.
    fn zone(&self, utc: bool) -> Result<Option<i64>, PyErr> {
        if self.found_aware_str && !utc {
            if self.offsets.len() != 1 {
                return Err(PyErr::value(
                    "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in DatetimeIndex to convert to a common timezone.",
                ));
            }
            return Ok(self.offsets[0]);
        }
        Ok(None)
    }
}

/// The range of a `datetime64[ns]`.
fn fits(unit: Unit, nanos: i128) -> bool {
    if unit != Unit::Ns {
        return true;
    }
    nanos >= -9_223_372_036_854_775_807 && nanos <= i128::from(i64::MAX)
}

fn finish(
    items: Vec<Option<Item>>,
    state: &State,
    utc: bool,
    coerce: bool,
) -> Result<Converted, PyErr> {
    let zone = state.zone(utc)?;
    let unit = state.unit.unwrap_or(Unit::S);
    let mut values = Vec::with_capacity(items.len());
    for item in items {
        let Some(item) = item else {
            values.push(None);
            continue;
        };
        let local = item.wall.epoch_nanos();
        let instant = local - i128::from(item.offset.unwrap_or(0)) * 1_000_000_000;
        let truncated = truncate(instant, unit);
        if !fits(unit, truncated) {
            if coerce {
                values.push(None);
                continue;
            }
            return Err(PyErr {
                class: "OutOfBoundsDatetime",
                message: "Out of bounds nanosecond timestamp".to_string(),
            });
        }
        values.push(Some(truncated));
    }
    let tz = match (zone, utc) {
        (_, true) => ColumnTz::Fixed(0),
        (Some(offset), false) => ColumnTz::Fixed(offset),
        (None, false) => ColumnTz::Naive,
    };
    Ok(Converted { unit, tz, values })
}

fn truncate(nanos: i128, unit: Unit) -> i128 {
    let per = match unit {
        Unit::S => 1_000_000_000,
        Unit::Ms => 1_000_000,
        Unit::Us => 1000,
        _ => 1,
    };
    nanos.div_euclid(per) * per
}

/// `array_to_datetime` over text: each value as ISO 8601, then `dateutil`.
fn objects_to_datetime(
    values: &[Option<String>],
    dayfirst: bool,
    utc: bool,
    coerce: bool,
    host: &Host,
) -> Result<Converted, PyErr> {
    let mut state = State::default();
    let mut items = Vec::with_capacity(values.len());
    for value in values {
        let Some(text) = value else {
            items.push(None);
            continue;
        };
        if text.is_empty() || is_nat_string(text) {
            items.push(None);
            continue;
        }
        if text == "now" || text == "today" {
            return Err(PyErr {
                class: "NotImplemented",
                message: format!("reading {text:?} depends on the clock"),
            });
        }
        match convert_str(text, dayfirst, host) {
            Ok(item) => {
                state.note(&item);
                items.push(Some(item));
            }
            Err(error) => {
                if !error.is_value_type_or_overflow() {
                    return Err(error);
                }
                if coerce {
                    items.push(None);
                } else {
                    return Err(error);
                }
            }
        }
    }
    finish(items, &state, utc, coerce)
}

/// `array_to_datetime` over text, without a format guess: the engine's
/// `to_datetime(ser, unit='ns')` of text.
pub fn objects_to_datetime_public(
    values: &[Option<String>],
    call: Call,
    host: &Host,
) -> Result<Converted, PyErr> {
    objects_to_datetime(values, call.dayfirst, call.utc, call.coerce, host)
}

/// `format_is_iso`.
pub fn format_is_iso(format: &str) -> bool {
    if format == "%Y%m" {
        return false;
    }
    let rest = match format.strip_prefix("%Y") {
        Some(rest) => rest,
        None => return false,
    };
    if rest.is_empty() {
        return true;
    }
    // ([-/ \\.]?)%m
    let (sep, rest) = match rest.chars().next() {
        Some(c @ ('-' | '/' | ' ' | '\\' | '.')) => (c.to_string(), &rest[1..]),
        _ => (String::new(), rest),
    };
    let Some(rest) = rest.strip_prefix("%m") else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let Some(rest) = rest.strip_prefix(&format!("{sep}%d")) else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let rest = match rest
        .strip_prefix(" %H")
        .or_else(|| rest.strip_prefix("T%H"))
    {
        Some(rest) => rest,
        None => return false,
    };
    if rest.is_empty() {
        return true;
    }
    let Some(rest) = rest.strip_prefix(":%M") else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let Some(rest) = rest.strip_prefix(":%S") else {
        return false;
    };
    matches!(rest, "" | "%z" | ".%f" | ".%f%z")
}

/// `array_strptime(values, fmt, exact=True)`.
fn array_strptime(
    values: &[Option<String>],
    fmt: &str,
    utc: bool,
    coerce: bool,
) -> Result<Converted, PyErr> {
    strptime::validate(fmt).map_err(PyErr::value)?;
    let compiled = strptime::compile(fmt, Flavor::Pandas).map_err(PyErr::value)?;
    let iso_format = format_is_iso(fmt);
    let mut state = State::default();
    let mut items = Vec::with_capacity(values.len());
    for value in values {
        let Some(text) = value else {
            items.push(None);
            continue;
        };
        if text.is_empty() || is_nat_string(text) {
            items.push(None);
            continue;
        }
        let read = (|| -> Result<Item, PyErr> {
            if iso_format && let Some(found) = iso::parse(text, Some(fmt), true) {
                return Ok(Item {
                    wall: iso_wall(&found.dts),
                    offset: found.offset.map(|m| i64::from(m) * 60),
                    unit: at_least_us(found.unit),
                });
            }
            if text == "now" || text == "today" {
                return Err(PyErr {
                    class: "NotImplemented",
                    message: format!("reading {text:?} depends on the clock"),
                });
            }
            let read =
                strptime::parse_with_format(&compiled, text, fmt, true).map_err(PyErr::value)?;
            let unit = if read.nanos { Unit::Ns } else { Unit::Us };
            // A second of 60 or 61 runs on into the next minute.
            let wall = Wall::from_epoch_nanos(read.wall.epoch_nanos());
            let offset = match read.tz {
                None => None,
                Some(Tz::Fixed(seconds)) => Some(seconds),
                Some(Tz::Named(name)) => {
                    let Some(zone) = super::zone::zone(&name) else {
                        return Err(PyErr {
                            class: "ZoneInfoNotFoundError",
                            message: format!("'No time zone found with key {name}'"),
                        });
                    };
                    let local = wall.epoch_nanos().div_euclid(1_000_000_000) as i64;
                    let Some(utc) = zone.localize(local) else {
                        return Err(PyErr::value(format!("Cannot localize {text} in {name}")));
                    };
                    Some(local - utc)
                }
            };
            Ok(Item { wall, offset, unit })
        })();
        match read {
            Ok(item) => {
                state.note(&item);
                items.push(Some(item));
            }
            Err(error) => {
                if error.class == "ZoneInfoNotFoundError" || error.class == "NotImplemented" {
                    return Err(error);
                }
                if coerce {
                    items.push(None);
                } else {
                    return Err(error);
                }
            }
        }
    }
    finish(items, &state, utc, coerce)
}

/// pandas' own split of a date into tokens (`_DATEUTIL_LEXER_SPLIT`).
fn pandas_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().filter(|&c| c != '\0').collect();
    let digit = |c: char| py::is_digit(c);
    let alpha = |c: char| c.is_ascii_alphabetic();
    let punct = |c: char| c == '.' || c == '/' || c == ':';
    let mut tokens: Vec<Option<String>> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if py::is_space(c) {
            tokens.push(Some(c.to_string()));
            i += 1;
            continue;
        }
        if digit(c) {
            // (?<![\.\d])\d+\.\d+(?![\.\d])
            let prev_ok = i == 0 || !(chars[i - 1] == '.' || digit(chars[i - 1]));
            let mut j = i;
            while j < chars.len() && digit(chars[j]) {
                j += 1;
            }
            if prev_ok
                && j < chars.len()
                && chars[j] == '.'
                && j + 1 < chars.len()
                && digit(chars[j + 1])
            {
                let mut k = j + 1;
                while k < chars.len() && digit(chars[k]) {
                    k += 1;
                }
                // The fraction backs off digit by digit until nothing follows
                // that is a dot or a digit.
                let mut end = k;
                while end > j + 1 {
                    let next = chars.get(end).copied();
                    if next.is_none_or(|n| !(n == '.' || digit(n))) {
                        break;
                    }
                    end -= 1;
                }
                let next = chars.get(end).copied();
                if next.is_none_or(|n| !(n == '.' || digit(n))) {
                    tokens.push(Some(chars[i..end].iter().collect()));
                    i = end;
                    continue;
                }
            }
            tokens.push(Some(chars[i..j].iter().collect()));
            i = j;
            continue;
        }
        if alpha(c) {
            let mut j = i;
            while j < chars.len() && alpha(chars[j]) {
                j += 1;
            }
            tokens.push(Some(chars[i..j].iter().collect()));
            i = j;
            continue;
        }
        if punct(c) {
            let mut j = i;
            while j < chars.len() && punct(chars[j]) {
                j += 1;
            }
            tokens.push(Some(chars[i..j].iter().collect()));
            i = j;
            continue;
        }
        let mut j = i;
        while j < chars.len()
            && !(digit(chars[j]) || alpha(chars[j]) || punct(chars[j]) || py::is_space(chars[j]))
        {
            j += 1;
        }
        tokens.push(Some(chars[i..j].iter().collect()));
        i = j;
    }
    let n = tokens.len();
    for k in 0..n.saturating_sub(2) {
        let token = tokens[k].clone();
        if let Some(token) = token {
            let is_digits = py::all_digits(&token);
            let comma = tokens[k + 1].as_deref() == Some(",");
            let next_digits = tokens[k + 2].as_deref().is_some_and(py::all_digits);
            if is_digits && comma && next_digits {
                let joined = format!("{token}.{}", tokens[k + 2].clone().unwrap_or_default());
                tokens[k] = Some(joined);
                tokens[k + 1] = None;
                tokens[k + 2] = None;
            }
        }
    }
    tokens.into_iter().flatten().collect()
}

/// `_fill_token`.
fn fill_token(token: &str, padding: usize) -> String {
    let has_decimal = {
        let chars: Vec<char> = token.chars().collect();
        (0..chars.len()).any(|i| {
            py::is_digit(chars[i]) && {
                let mut j = i;
                while j < chars.len() && py::is_digit(chars[j]) {
                    j += 1;
                }
                j < chars.len()
                    && chars[j] == '.'
                    && chars.get(j + 1).is_some_and(|&c| py::is_digit(c))
            }
        })
    };
    if !has_decimal {
        return py::zfill(token, padding);
    }
    let (seconds, nanoseconds) = token.split_once('.').unwrap_or((token, ""));
    let seconds = py::int(seconds).map_or(seconds.to_string(), |s| format!("{s:02}"));
    let mut nanos: String = nanoseconds.to_string();
    while nanos.chars().count() < 9 {
        nanos.push('0');
    }
    let nanos: String = nanos.chars().take(6).collect();
    format!("{seconds}.{nanos}")
}

/// The warning pandas gives when a guessed format reads the day and month
/// the other way round from `dayfirst`.
fn dayfirst_warning(format: &str, dayfirst: bool) -> Option<String> {
    let day = format.find("%d")?;
    let month = format.find("%m")?;
    if day > month && dayfirst {
        return Some(format!(
            "Parsing dates in {format} format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning."
        ));
    }
    if day < month && !dayfirst {
        return Some(format!(
            "Parsing dates in {format} format when dayfirst=False (the default) was specified. Pass `dayfirst=True` or specify a format to silence this warning."
        ));
    }
    None
}

/// `guess_datetime_format(text, dayfirst)`, and the warning it gives.
pub fn guess_format(text: &str, dayfirst: bool, host: &Host) -> (Option<String>, Option<String>) {
    let Ok((parsed, _, _)) = dateutil_parse(text, Wall::date(1970, 1, 1), dayfirst, false, host)
    else {
        return (None, None);
    };
    let mut tokens = pandas_tokens(text);
    if parsed.tz.is_some() {
        let n = tokens.len();
        let offset_index: Option<usize> = if n > 0 && tokens[n - 1] == "Z" {
            Some(n - 1)
        } else if n > 1 && (tokens[n - 2] == "+" || tokens[n - 2] == "-") {
            Some(n - 2)
        } else if n > 3 && (tokens[n - 4] == "+" || tokens[n - 4] == "-") {
            Some(n - 4)
        } else {
            None
        };
        if let Some(index) = offset_index {
            tokens[index] = strftime(&parsed, "%z", host.pads_year);
            tokens.truncate(index + 1);
        }
    }
    let attrs: Vec<(&[&str], &str, usize)> = {
        let mut list: Vec<(&[&str], &str, usize)> = vec![
            (
                &["year", "month", "day", "hour", "minute", "second"],
                "%Y%m%d%H%M%S",
                0,
            ),
            (&["year", "month", "day", "hour", "minute"], "%Y%m%d%H%M", 0),
            (&["year", "month", "day", "hour"], "%Y%m%d%H", 0),
            (&["year", "month", "day"], "%Y%m%d", 0),
            (&["hour", "minute", "second"], "%H%M%S", 0),
            (&["year"], "%Y", 0),
            (&["hour", "minute"], "%H%M", 0),
            (&["month"], "%B", 0),
            (&["month"], "%b", 0),
            (&["month"], "%m", 2),
            (&["day"], "%d", 2),
            (&["hour"], "%H", 2),
            (&["minute"], "%M", 2),
            (&["second"], "%S", 2),
            (&["second", "microsecond"], "%S.%f", 0),
            (&["tzinfo"], "%z", 0),
            (&["tzinfo"], "%Z", 0),
            (&["day_of_week"], "%a", 0),
            (&["day_of_week"], "%A", 0),
            (&["meridiem"], "%p", 0),
        ];
        if dayfirst {
            let day = list.remove(10);
            list.insert(0, day);
        }
        list
    };
    let mut guess: Vec<Option<&str>> = vec![None; tokens.len()];
    let mut found: Vec<&str> = Vec::new();
    for (names, format, padding) in attrs {
        if names.iter().any(|n| found.contains(n)) {
            continue;
        }
        if parsed.tz.is_none() && (format == "%Z" || format == "%z") {
            continue;
        }
        let formatted = strftime(&parsed, format, host.pads_year);
        for i in 0..tokens.len() {
            let filled = fill_token(&tokens[i], padding);
            if guess[i].is_none() && filled == formatted {
                guess[i] = Some(format);
                tokens[i] = filled;
                found.extend(names.iter());
                break;
            }
        }
    }
    let has_ymd = ["year", "month", "day"].iter().all(|n| found.contains(n));
    let only_year = guess == [Some("%Y")];
    let year_month = guess.len() == 3
        && guess[0] == Some("%Y")
        && guess[1].is_none()
        && guess[2] == Some("%m")
        && tokens[1] == "-";
    if !has_ymd && !only_year && !year_month {
        return (None, None);
    }
    let mut output: Vec<String> = Vec::new();
    for (i, g) in guess.iter().enumerate() {
        match g {
            Some(format) => output.push((*format).to_string()),
            None => {
                if py::is_float(&tokens[i]) {
                    return (None, None);
                }
                output.push(tokens[i].clone());
            }
        }
    }
    if output.iter().any(|f| f == "%p")
        && let Some(i) = output.iter().position(|f| f == "%H")
    {
        output[i] = "%I".to_string();
    }
    let format = output.concat();
    let single = [Some(text.to_string())];
    if array_strptime(&single, &format, false, false).is_err() {
        return (None, None);
    }
    let rebuilt = tokens.concat();
    if strftime(&parsed, &format, host.pads_year) == rebuilt {
        let warning = dayfirst_warning(&format, dayfirst);
        return (Some(format), warning);
    }
    (None, None)
}

/// How `pd.to_datetime` is called.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Call {
    pub dayfirst: bool,
    pub utc: bool,
    pub mixed: bool,
    /// `errors='coerce'`, rather than `'raise'`.
    pub coerce: bool,
}

/// The warning pandas gives when it guesses no format for a column of more
/// than one value.
pub const COULD_NOT_INFER: &str = "Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.";

/// `pd.to_datetime(pd.Series(values, dtype=object), ...)`: the column, and
/// the warnings it gave.
pub fn to_datetime(
    values: &[Option<String>],
    call: Call,
    host: &Host,
) -> (Result<Converted, PyErr>, Vec<String>) {
    let mut warnings = Vec::new();
    if !call.mixed
        && let Some(first) = first_non_null(values)
    {
        let text = values[first].as_deref().unwrap_or("");
        let (format, warning) = guess_format(text, call.dayfirst, host);
        if let Some(warning) = warning {
            warnings.push(warning);
        }
        if let Some(format) = format {
            return (
                array_strptime(values, &format, call.utc, call.coerce),
                warnings,
            );
        }
        if first_non_null(&values[first + 1..]).is_some() {
            warnings.push(COULD_NOT_INFER.to_string());
        }
    }
    (
        objects_to_datetime(values, call.dayfirst, call.utc, call.coerce, host),
        warnings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_series_facts_roundtrip_without_losing_precision() {
        for names in [["s", "ms", "us", "ns"], ["ns", "ns", "ns", "ns"]] {
            let facts = TimestampSeriesUnits::from_names(names).unwrap();
            let encoded = serde_json::to_string(&facts.names()).unwrap();
            let decoded: [String; 4] = serde_json::from_str(&encoded).unwrap();
            assert_eq!(
                TimestampSeriesUnits::from_names(decoded.each_ref().map(String::as_str)).unwrap(),
                facts
            );
            for (scalar, output) in ["s", "ms", "us", "ns"].into_iter().zip(names) {
                assert_eq!(facts.for_scalar(scalar), Some(output));
            }
        }
        assert!(TimestampSeriesUnits::from_names(["s", "ms", "us", "us"]).is_err());
        assert!(TimestampSeriesUnits::from_names(["s", "ms", "object", "ns"]).is_err());
        assert_eq!(TimestampSeriesUnits::default().for_scalar("days"), None);
    }

    fn host() -> Host {
        Host {
            year: 2026,
            pads_year: true,
            today: (2026, 10, 3),
            tznames: vec!["PST".into(), "PDT".into()],
            timestamp_series_units: TimestampSeriesUnits::default(),
            numpy_text_prefix: false,
        }
    }

    #[test]
    fn guesses_as_pandas() {
        let h = host();
        assert_eq!(
            guess_format("2021-01-01", false, &h).0.as_deref(),
            Some("%Y-%m-%d")
        );
        assert_eq!(
            guess_format("13/01/2021", false, &h).0.as_deref(),
            Some("%d/%m/%Y")
        );
        assert_eq!(
            guess_format("Jan 5, 2021", false, &h).0.as_deref(),
            Some("%b %d, %Y")
        );
        assert_eq!(
            guess_format("2021-01-01 10:00:00+05:30", false, &h)
                .0
                .as_deref(),
            Some("%Y-%m-%d %H:%M:%S%z")
        );
        assert_eq!(guess_format("abc", false, &h).0, None);
    }

    #[test]
    fn converts_as_pandas() {
        let h = host();
        let values = vec![
            Some("01/02/2021".to_string()),
            Some("13/02/2021".to_string()),
        ];
        let call = Call {
            coerce: true,
            ..Call::default()
        };
        let (got, _) = to_datetime(&values, call, &h);
        let got = got.unwrap();
        assert_eq!(got.dtype(), "datetime64[us]");
        assert_eq!(got.isoformat(0).as_deref(), Some("2021-01-02T00:00:00"));
        assert_eq!(got.isoformat(1), None);
    }
}
