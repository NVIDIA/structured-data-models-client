// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Date literals, read as today's PQL reads them with `pd.to_datetime`.
//!
//! A literal is the grammar's `DATE` token, three runs of digits with one
//! separator (`/`, `-` or `.`) between them, optionally followed by one space
//! and a `TIME` token, three runs of digits with `:` between them. Only those
//! shapes are read; `DATES.md` beside this crate states the rules this gives.
//!
//! The reading follows the steps pandas 3.0 takes for one string:
//!
//! 1. It guesses a `strptime` format from a `dateutil` reading of the
//!    literal, warning when the guess puts the day before the month.
//! 2. With a format, it parses the literal with that format.
//! 3. Without one, it tries ISO 8601, then a fixed `MM/DD/YYYY` reading, then
//!    `dateutil` again.
//!
//! Each step is ported only as far as these shapes reach into it.

use std::fmt;

/// The warning pandas gives when its guessed format reads the day first.
fn dayfirst_warning(format: &str) -> String {
    format!(
        "Parsing dates in {format} format when dayfirst=False (the default) was specified. \
         Pass `dayfirst=True` or specify a format to silence this warning."
    )
}

/// A moment as pandas holds it, with the UTC offset it was written with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub microsecond: u32,
    /// The UTC offset in seconds, when the literal gave one.
    pub offset: Option<i32>,
}

impl Stamp {
    /// `YYYY-MM-DDTHH:MM:SS[.fffffffff][+HH:MM[:SS]]`, the form the Phase 5
    /// checks compare (`tests/rust_core/pql/CONTRACT.md`).
    pub fn iso(&self) -> String {
        let mut out = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if self.microsecond != 0 {
            out.push_str(&format!(".{:06}000", self.microsecond));
        }
        if let Some(offset) = self.offset {
            out.push_str(&offset_text(offset, true));
        }
        out
    }

    /// `str(pd.Timestamp)`: `YYYY-MM-DD HH:MM:SS[+HH:MM]`.
    pub fn display(&self) -> String {
        let mut out = format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if self.microsecond != 0 {
            out.push_str(&format!(".{:06}", self.microsecond));
        }
        if let Some(offset) = self.offset {
            out.push_str(&offset_text(offset, true));
        }
        out
    }
}

impl fmt::Display for Stamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display())
    }
}

/// `+HH:MM`, with `:SS` when not whole minutes; `+HHMM` without colons.
fn offset_text(offset: i32, colons: bool) -> String {
    let sign = if offset < 0 { '-' } else { '+' };
    let total = offset.unsigned_abs();
    let (h, m, s) = (total / 3600, total % 3600 / 60, total % 60);
    let sep = if colons { ":" } else { "" };
    let mut out = format!("{sign}{h:02}{sep}{m:02}");
    if s != 0 {
        out.push_str(&format!("{sep}{s:02}"));
    }
    out
}

/// What reading a literal gives: a moment, or nothing, and the warnings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reading {
    pub value: Option<Stamp>,
    pub warnings: Vec<String>,
}

/// What a reading takes from the Python process pandas runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Host {
    /// The current year, which `dateutil` reads two-digit years against:
    /// within 50 years of it.
    pub year: i64,
    /// Whether Python's `strftime` writes `%Y` with four digits for a year
    /// below 1000. It does on macOS and, from Python 3.14, on Linux; on Linux
    /// up to Python 3.13 it writes `999`, so pandas guesses no format for
    /// `13/01/0999` and reads it without the day-first warning.
    pub pads_year: bool,
}

impl Host {
    /// The year in UTC, and `%Y` unpadded as on Linux with Python 3.12 or
    /// 3.13: what the core assumes when no Python process tells it.
    pub fn assumed() -> Host {
        Host {
            year: this_year(),
            pads_year: false,
        }
    }
}

/// Whether `literal` has one of the shapes a date constant can have.
pub fn is_date_shape(literal: &str) -> bool {
    let (date, time) = match literal.split_once(' ') {
        Some((date, time)) => (date, Some(time)),
        None => (literal, None),
    };
    let runs = |text: &str, seps: &[char]| -> bool {
        let Some(sep) = text.chars().find(|c| !c.is_ascii_digit()) else {
            return false;
        };
        if !seps.contains(&sep) {
            return false;
        }
        let parts: Vec<&str> = text.split(sep).collect();
        parts.len() == 3
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))
    };
    runs(date, &['/', '-', '.']) && time.is_none_or(|t| runs(t, &[':']))
}

/// `pd.to_datetime(literal)`, for a literal of a date constant's shape, as
/// pandas reads it in the Python process `host` describes.
pub fn to_datetime(literal: &str, host: Host) -> Reading {
    let mut warnings = Vec::new();
    if !is_date_shape(literal) {
        return Reading {
            value: None,
            warnings,
        };
    }
    let value = match guess_format(literal, host, &mut warnings) {
        Some(format) => strptime(literal, &format).ok(),
        None => parse_without_format(literal, host.year).ok(),
    };
    Reading { value, warnings }
}

/// The current year in UTC, which two-digit years are read against when
/// the caller gives none.
pub fn this_year() -> i64 {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    civil_year(seconds.div_euclid(86_400))
}

/// The proleptic Gregorian year of a day counted from 1970-01-01.
fn civil_year(days: i64) -> i64 {
    // Howard Hinnant's `civil_from_days`.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    yoe + era * 400 + i64::from(month <= 2)
}

// --- Python's datetime -------------------------------------------------------

/// Python's `datetime`, fields as `replace` checks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DateTime {
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    microsecond: u32,
    offset: Option<i32>,
}

impl DateTime {
    fn stamp(self) -> Stamp {
        Stamp {
            year: self.year,
            month: self.month,
            day: self.day,
            hour: self.hour,
            minute: self.minute,
            second: self.second,
            microsecond: self.microsecond,
            offset: self.offset,
        }
    }
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(year) => 29,
        _ => 28,
    }
}

/// A value read from digits: Python's int, saturated far beyond any field.
type Num = u128;

fn num(digits: &str) -> Option<Num> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(digits.parse().unwrap_or(Num::MAX))
}

#[derive(Clone, Copy, Default)]
struct Replace {
    year: Option<Num>,
    month: Option<Num>,
    day: Option<Num>,
    hour: Option<Num>,
    minute: Option<Num>,
    second: Option<Num>,
    microsecond: Option<Num>,
}

/// `default.replace(**repl)`: refused when a field is out of range.
fn replace(default: DateTime, repl: Replace) -> Option<DateTime> {
    let pick =
        |v: Option<Num>, d: u64| -> u64 { v.map_or(d, |v| u64::try_from(v).unwrap_or(u64::MAX)) };
    let year = pick(repl.year, default.year as u64);
    let month = pick(repl.month, u64::from(default.month));
    let day = pick(repl.day, u64::from(default.day));
    let hour = pick(repl.hour, u64::from(default.hour));
    let minute = pick(repl.minute, u64::from(default.minute));
    let second = pick(repl.second, u64::from(default.second));
    let microsecond = pick(repl.microsecond, u64::from(default.microsecond));
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let year = year as i64;
    let month = month as u32;
    if day < 1 || day > u64::from(days_in_month(year, month)) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 || microsecond > 999_999 {
        return None;
    }
    Some(DateTime {
        year,
        month,
        day: day as u32,
        hour: hour as u32,
        minute: minute as u32,
        second: second as u32,
        microsecond: microsecond as u32,
        offset: default.offset,
    })
}

/// `datetime.strftime` for the directives the format guess uses.
///
/// `%Y` is padded to four digits only when `pads_year` (see [`Host`]); the
/// text directives never equal a token of digits, so they give text no token
/// can be.
fn strftime(dt: &DateTime, format: &str, pads_year: bool) -> String {
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') if pads_year => out.push_str(&format!("{:04}", dt.year)),
            Some('Y') => out.push_str(&dt.year.to_string()),
            Some('m') => out.push_str(&format!("{:02}", dt.month)),
            Some('d') => out.push_str(&format!("{:02}", dt.day)),
            Some('H') => out.push_str(&format!("{:02}", dt.hour)),
            Some('M') => out.push_str(&format!("{:02}", dt.minute)),
            Some('S') => out.push_str(&format!("{:02}", dt.second)),
            Some('f') => out.push_str(&format!("{:06}", dt.microsecond)),
            Some('z') => {
                if let Some(offset) = dt.offset {
                    out.push_str(&offset_text(offset, false));
                }
            }
            Some(other) => {
                out.push('%');
                out.push(other);
                out.push('\u{0}');
            }
            None => out.push('%'),
        }
    }
    out
}

// --- dateutil ------------------------------------------------------------------

/// `dateutil.parser._timelex.split`, for text of digits, punctuation and
/// spaces.
fn dateutil_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().filter(|&c| c != '\0').collect();
    let mut tokens = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut pos = 0;
    loop {
        if !stack.is_empty() {
            tokens.push(stack.remove(0));
            continue;
        }
        if pos >= chars.len() {
            break;
        }
        let first = chars[pos];
        pos += 1;
        let mut token = first.to_string();
        let mut state = if first.is_alphabetic() {
            'a'
        } else if first.is_ascii_digit() {
            '0'
        } else {
            if crate::syntax::is_py_space(first) {
                token = " ".to_string();
            }
            tokens.push(token);
            continue;
        };
        let mut seen_letters = false;
        let mut dotted = false;
        while pos < chars.len() {
            let c = chars[pos];
            let keep = match (state, dotted) {
                ('a', false) => {
                    seen_letters = true;
                    if c.is_alphabetic() {
                        true
                    } else if c == '.' {
                        dotted = true;
                        true
                    } else {
                        false
                    }
                }
                ('0', false) => {
                    if c.is_ascii_digit() {
                        true
                    } else if c == '.' || (c == ',' && token.chars().count() >= 2) {
                        dotted = true;
                        true
                    } else {
                        false
                    }
                }
                ('a', true) => {
                    seen_letters = true;
                    if c == '.' || c.is_alphabetic() {
                        true
                    } else if c.is_ascii_digit() && token.ends_with('.') {
                        state = '0';
                        true
                    } else {
                        false
                    }
                }
                (_, _) => {
                    if c == '.' || c.is_ascii_digit() {
                        true
                    } else if c.is_alphabetic() && token.ends_with('.') {
                        state = 'a';
                        true
                    } else {
                        false
                    }
                }
            };
            if !keep {
                break;
            }
            token.push(c);
            pos += 1;
        }
        let dots = token.matches('.').count();
        if dotted && (seen_letters || dots > 1 || token.ends_with('.') || token.ends_with(',')) {
            // Split on '.' and ',', keeping them, dropping empty pieces.
            let mut pieces = Vec::new();
            let mut current = String::new();
            for c in token.chars() {
                if c == '.' || c == ',' {
                    pieces.push(std::mem::take(&mut current));
                    pieces.push(c.to_string());
                } else {
                    current.push(c);
                }
            }
            pieces.push(current);
            token = pieces[0].clone();
            stack.extend(pieces.into_iter().skip(1).filter(|p| !p.is_empty()));
        }
        if dotted && state == '0' && dots == 0 {
            token = token.replace(',', ".");
        }
        tokens.push(token);
    }
    tokens
}

/// `dateutil`'s `_ymd`, as numeric tokens fill it.
#[derive(Default)]
struct Ymd {
    values: Vec<Num>,
    century_specified: bool,
    year_index: Option<usize>,
}

impl Ymd {
    /// `append` of a token's digits.
    fn append_text(&mut self, text: &str, year_label: bool) -> Result<(), ()> {
        let value = num(text).ok_or(())?;
        let mut year = year_label;
        if text.chars().count() > 2 {
            self.century_specified = true;
            year = true;
        }
        self.push(value, year)
    }

    /// `append` of a number already read.
    fn append_value(&mut self, value: Num) -> Result<(), ()> {
        let mut year = false;
        if value > 100 {
            self.century_specified = true;
            year = true;
        }
        self.push(value, year)
    }

    fn push(&mut self, value: Num, year: bool) -> Result<(), ()> {
        self.values.push(value);
        if year {
            if self.year_index.is_some() {
                return Err(());
            }
            self.year_index = Some(self.values.len() - 1);
        }
        Ok(())
    }

    /// `resolve_ymd(yearfirst=False, dayfirst=False)`: (year, month, day).
    #[allow(clippy::type_complexity)]
    fn resolve(&self) -> Result<(Option<Num>, Option<Num>, Option<Num>), ()> {
        let v = &self.values;
        match v.len() {
            0 => Ok((None, None, None)),
            1 => {
                if self.year_index.is_some() {
                    return Ok((Some(v[0]), None, None));
                }
                if v[0] > 31 {
                    Ok((Some(v[0]), None, None))
                } else {
                    Ok((None, None, Some(v[0])))
                }
            }
            2 => {
                if v[0] > 31 {
                    Ok((Some(v[0]), Some(v[1]), None))
                } else if v[1] > 31 {
                    Ok((Some(v[1]), Some(v[0]), None))
                } else {
                    Ok((None, Some(v[0]), Some(v[1])))
                }
            }
            3 => {
                if v[0] > 31 || self.year_index == Some(0) {
                    Ok((Some(v[0]), Some(v[1]), Some(v[2])))
                } else if v[0] > 12 {
                    Ok((Some(v[2]), Some(v[1]), Some(v[0])))
                } else {
                    Ok((Some(v[2]), Some(v[0]), Some(v[1])))
                }
            }
            _ => Err(()),
        }
    }
}

/// What `dateutil`'s `_parse` finds.
#[derive(Default)]
struct Found {
    year: Option<Num>,
    month: Option<Num>,
    day: Option<Num>,
    hour: Option<Num>,
    minute: Option<Num>,
    second: Option<Num>,
    microsecond: Option<Num>,
    offset: Option<i64>,
}

/// The jump words a numeric-only string can hold.
fn is_jump(token: &str) -> bool {
    matches!(token, " " | "." | "," | ";" | "-" | "/" | "'")
}

fn is_digits(token: &str) -> bool {
    !token.is_empty() && token.bytes().all(|b| b.is_ascii_digit())
}

/// Python's `float(token)` succeeds: for the tokens here, digits.
fn is_number(token: &str) -> bool {
    let t = token.trim_matches(crate::syntax::is_py_space);
    !t.is_empty() && t.parse::<f64>().is_ok()
}

fn two_digits(text: &str) -> Option<Num> {
    num(&text.chars().take(2).collect::<String>())
}

/// `dateutil.parser.parser._parse(timestr)`, with `dayfirst` and
/// `yearfirst` off, for text of digits and punctuation.
fn dateutil_parse_raw(text: &str, year: i64) -> Option<Found> {
    let l = dateutil_tokens(text);
    let mut res = Found::default();
    let mut ymd = Ymd::default();
    let mut i = 0;
    let len = l.len();
    while i < len {
        let token = l[i].clone();
        if is_number(&token) {
            i = parse_numeric_token(&l, i, &mut ymd, &mut res)?;
        } else if res.hour.is_some() && (token == "+" || token == "-") {
            let sign: i64 = if token == "+" { 1 } else { -1 };
            let next = l.get(i + 1)?.clone();
            let next_len = next.chars().count();
            let (h, m);
            if next_len == 4 {
                h = num(&next[..2])?;
                m = num(&next[2..])?;
            } else if i + 2 < len && l[i + 2] == ":" {
                h = num(&next)?;
                m = num(l.get(i + 3)?)?;
                i += 2;
            } else if next_len <= 2 {
                h = two_digits(&next)?;
                m = 0;
            } else {
                return None;
            }
            let h = i64::try_from(h).ok()?;
            let m = i64::try_from(m).ok()?;
            res.offset = Some(sign * (h.checked_mul(3600)? + m.checked_mul(60)?));
            i += 1;
        } else if !is_jump(&token) {
            return None;
        }
        i += 1;
    }
    let (y, m, d) = ymd.resolve().ok()?;
    res.year = y.map(|y| convert_year(y, ymd.century_specified, year));
    res.month = m;
    res.day = d;
    Some(res)
}

/// `parserinfo.convertyear`: a two-digit year within 50 years of `now`.
fn convert_year(year: Num, century_specified: bool, now: i64) -> Num {
    if year < 100 && !century_specified {
        let century = (now / 100 * 100) as i128;
        let mut y = year as i128 + century;
        if y >= now as i128 + 50 {
            y -= 100;
        } else if y < now as i128 - 50 {
            y += 100;
        }
        return y.max(0) as Num;
    }
    year
}

/// `parser._parse_numeric_token`, for a token of digits.
fn parse_numeric_token(
    tokens: &[String],
    idx: usize,
    ymd: &mut Ymd,
    res: &mut Found,
) -> Option<usize> {
    let s = &tokens[idx];
    let value = num(s)?;
    let len_li = s.chars().count();
    let len_l = tokens.len();
    let mut idx = idx;
    if ymd.values.len() == 3
        && (len_li == 2 || len_li == 4)
        && res.hour.is_none()
        && (idx + 1 >= len_l || tokens[idx + 1] != ":")
    {
        res.hour = num(&s[..2]);
        if len_li == 4 {
            res.minute = num(&s[2..]);
        }
    } else if len_li == 6 {
        if ymd.values.is_empty() {
            ymd.append_text(&s[..2], false).ok()?;
            ymd.append_text(&s[2..4], false).ok()?;
            ymd.append_text(&s[4..], false).ok()?;
        } else {
            res.hour = num(&s[..2]);
            res.minute = num(&s[2..4]);
            res.second = num(&s[4..]);
            res.microsecond = Some(0);
        }
    } else if len_li == 8 || len_li == 12 || len_li == 14 {
        ymd.append_text(&s[..4], true).ok()?;
        ymd.append_text(&s[4..6], false).ok()?;
        ymd.append_text(&s[6..8], false).ok()?;
        if len_li > 8 {
            res.hour = num(&s[8..10]);
            res.minute = num(&s[10..12]);
            if len_li > 12 {
                res.second = num(&s[12..]);
            }
        }
    } else if idx + 2 < len_l && tokens[idx + 1] == ":" {
        res.hour = Some(value);
        if !is_number(&tokens[idx + 2]) {
            return None;
        }
        res.minute = Some(num(&tokens[idx + 2])?);
        res.second = None;
        if idx + 4 < len_l && tokens[idx + 3] == ":" {
            res.second = Some(num(&tokens[idx + 4])?);
            res.microsecond = Some(0);
            idx += 2;
        }
        idx += 2;
    } else if idx + 1 < len_l && matches!(tokens[idx + 1].as_str(), "-" | "/" | ".") {
        let sep = tokens[idx + 1].clone();
        ymd.append_text(s, false).ok()?;
        if idx + 2 < len_l && !is_jump(&tokens[idx + 2]) {
            if is_digits(&tokens[idx + 2]) {
                ymd.append_text(&tokens[idx + 2], false).ok()?;
            } else {
                return None;
            }
            if idx + 3 < len_l && tokens[idx + 3] == sep {
                ymd.append_text(tokens.get(idx + 4)?, false).ok()?;
                idx += 2;
            }
            idx += 1;
        }
        idx += 1;
    } else if idx + 1 >= len_l || is_jump(&tokens[idx + 1]) {
        ymd.append_value(value).ok()?;
        idx += 1;
    } else if (1..=31).contains(&value) {
        // `could_be_day`: no month name or day label is ever set here.
        ymd.append_value(value).ok()?;
    } else {
        return None;
    }
    Some(idx)
}

/// pandas' `dateutil_parse`: the `dateutil` reading laid over `default`.
fn dateutil_parse(text: &str, default: DateTime, year: i64) -> Option<DateTime> {
    let found = dateutil_parse_raw(text, year)?;
    let repl = Replace {
        year: found.year,
        month: found.month,
        day: found.day,
        hour: found.hour,
        minute: found.minute,
        second: found.second,
        microsecond: found.microsecond,
    };
    if [
        repl.year,
        repl.month,
        repl.day,
        repl.hour,
        repl.minute,
        repl.second,
        repl.microsecond,
    ]
    .iter()
    .all(Option::is_none)
    {
        return None;
    }
    let mut dt = replace(default, repl)?;
    if let Some(offset) = found.offset {
        // A UTC offset of a day or more is refused.
        if offset.abs() >= 86_400 {
            return None;
        }
        dt.offset = Some(offset as i32);
    }
    Some(dt)
}

// --- pandas: guessing the format ---------------------------------------------

/// pandas' `_timelex.split`: digit runs, runs of `./:`, single spaces, and
/// runs of anything else.
fn pandas_tokens(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().filter(|&c| c != '\0').collect();
    let mut tokens = Vec::new();
    let mut pos = 0;
    while pos < chars.len() {
        let c = chars[pos];
        let class = |c: char| -> u8 {
            if crate::syntax::is_py_space(c) {
                0
            } else if c.is_ascii_digit() {
                1
            } else if c.is_ascii_alphabetic() {
                2
            } else if matches!(c, '.' | '/' | ':') {
                3
            } else {
                4
            }
        };
        let kind = class(c);
        let mut end = pos + 1;
        if kind != 0 {
            while end < chars.len() && class(chars[end]) == kind {
                end += 1;
            }
        }
        tokens.push(chars[pos..end].iter().collect());
        pos = end;
    }
    tokens
}

/// Python's `str.zfill`.
fn zfill(text: &str, width: usize) -> String {
    let len = text.chars().count();
    if len >= width {
        return text.to_string();
    }
    let pad = "0".repeat(width - len);
    match text.chars().next() {
        Some(sign @ ('+' | '-')) => format!("{sign}{pad}{}", &text[1..]),
        _ => format!("{pad}{text}"),
    }
}

/// pandas' `guess_datetime_format(literal)`, which warns on a day-first
/// guess.
fn guess_format(text: &str, host: Host, warnings: &mut Vec<String>) -> Option<String> {
    let default = DateTime {
        year: 1970,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        microsecond: 0,
        offset: None,
    };
    let parsed = dateutil_parse(text, default, host.year)?;
    let mut tokens = pandas_tokens(text);
    if parsed.offset.is_some() {
        let n = tokens.len();
        let index = if n > 0 && tokens[n - 1] == "Z" {
            Some(n - 1)
        } else if n > 1 && matches!(tokens[n - 2].as_str(), "+" | "-") {
            Some(n - 2)
        } else if n > 3 && matches!(tokens[n - 4].as_str(), "+" | "-") {
            Some(n - 4)
        } else {
            None
        };
        if let Some(index) = index {
            tokens[index] = strftime(&parsed, "%z", host.pads_year);
            tokens.truncate(index + 1);
        }
    }
    const FORMATS: &[(&[&str], &str, usize)] = &[
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
    let mut guess: Vec<Option<&str>> = vec![None; tokens.len()];
    let mut found: Vec<&str> = Vec::new();
    for (attrs, format, padding) in FORMATS {
        if attrs.iter().any(|a| found.contains(a)) {
            continue;
        }
        if parsed.offset.is_none() && (*format == "%z" || *format == "%Z") {
            continue;
        }
        let formatted = strftime(&parsed, format, host.pads_year);
        for i in 0..tokens.len() {
            let filled = zfill(&tokens[i], *padding);
            if guess[i].is_none() && filled == formatted {
                guess[i] = Some(format);
                tokens[i] = filled;
                found.extend(attrs.iter());
                break;
            }
        }
    }
    let ymd = ["year", "month", "day"]
        .iter()
        .filter(|a| found.contains(a))
        .count();
    if ymd != 3 {
        let only_year = guess.len() == 1 && guess[0] == Some("%Y");
        let year_month = guess.len() == 3
            && guess[0] == Some("%Y")
            && guess[1].is_none()
            && guess[2] == Some("%m")
            && tokens[1] == "-";
        if !only_year && !year_month {
            return None;
        }
    }
    let mut format = String::new();
    for (i, g) in guess.iter().enumerate() {
        match g {
            Some(f) => format.push_str(f),
            None => {
                if is_number(&tokens[i]) {
                    return None;
                }
                format.push_str(&tokens[i]);
            }
        }
    }
    strptime(text, &format).ok()?;
    if strftime(&parsed, &format, host.pads_year) != tokens.concat() {
        return None;
    }
    if let (Some(d), Some(m)) = (format.find("%d"), format.find("%m"))
        && d < m
    {
        warnings.push(dayfirst_warning(&format));
    }
    Some(format)
}

// --- pandas: parsing with a format --------------------------------------------

/// pandas' `format_is_iso`.
fn format_is_iso(format: &str) -> bool {
    if format == "%Y%m" {
        return false;
    }
    let Some(mut rest) = format.strip_prefix("%Y") else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let sep = match rest.chars().next() {
        Some(c @ ('-' | '/' | ' ' | '\\' | '.')) => {
            rest = &rest[1..];
            Some(c)
        }
        _ => None,
    };
    let Some(r) = rest.strip_prefix("%m") else {
        return false;
    };
    rest = r;
    if rest.is_empty() {
        return true;
    }
    let day = format!("{}%d", sep.map(String::from).unwrap_or_default());
    let Some(r) = rest.strip_prefix(day.as_str()) else {
        return false;
    };
    rest = r;
    if rest.is_empty() {
        return true;
    }
    let Some(r) = rest
        .strip_prefix(" %H")
        .or_else(|| rest.strip_prefix("T%H"))
    else {
        return false;
    };
    rest = r;
    for part in [":%M", ":%S"] {
        if rest.is_empty() {
            return true;
        }
        let Some(r) = rest.strip_prefix(part) else {
            return false;
        };
        rest = r;
    }
    matches!(rest, "" | "%z" | ".%f" | ".%f%z")
}

/// pandas' `array_strptime` for one string and `exact=True`.
fn strptime(text: &str, format: &str) -> Result<Stamp, ()> {
    if format_is_iso(format)
        && let Ok(stamp) = iso8601(text, Some(format))
    {
        return Ok(stamp);
    }
    parse_with_format(text, format)
}

/// One element of a `strptime` pattern.
enum Piece {
    Literal(char),
    Space,
    Directive(char),
}

fn pattern(format: &str) -> Result<Vec<Piece>, ()> {
    let mut pieces = Vec::new();
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '%' {
            match chars.next() {
                Some(d @ ('Y' | 'm' | 'd' | 'H' | 'M' | 'S' | 'z')) => {
                    pieces.push(Piece::Directive(d))
                }
                _ => return Err(()),
            }
        } else if crate::syntax::is_py_space(c) {
            while chars.peek().is_some_and(|&c| crate::syntax::is_py_space(c)) {
                chars.next();
            }
            pieces.push(Piece::Space);
        } else {
            pieces.push(Piece::Literal(c));
        }
    }
    Ok(pieces)
}

/// The alternatives of CPython's `_strptime` regex for a directive, each a
/// list of character classes, tried in order.
fn alternatives(directive: char) -> &'static [&'static [&'static str]] {
    const D: &str = "0123456789";
    match directive {
        'd' => &[
            &["3", "01"],
            &["12", D],
            &["0", "123456789"],
            &["123456789"],
            &[" ", "123456789"],
        ],
        'm' => &[&["1", "012"], &["0", "123456789"], &["123456789"]],
        'Y' => &[&[D, D, D, D]],
        'H' => &[&["2", "0123"], &["01", D], &[D]],
        'M' => &[&["012345", D], &[D]],
        'S' => &[&["6", "01"], &["012345", D], &[D]],
        _ => &[],
    }
}

/// The first match of the pattern at `pos`, in the regex engine's order of
/// preference, as the captured fields and the end.
fn match_pieces(
    text: &[char],
    pos: usize,
    pieces: &[Piece],
    fields: &mut Vec<(char, String)>,
) -> Option<usize> {
    let Some((first, rest)) = pieces.split_first() else {
        return Some(pos);
    };
    match first {
        Piece::Literal(c) => {
            if text.get(pos) == Some(c) {
                match_pieces(text, pos + 1, rest, fields)
            } else {
                None
            }
        }
        Piece::Space => {
            let mut end = pos;
            while end < text.len() && crate::syntax::is_py_space(text[end]) {
                end += 1;
            }
            // `\s+`, greedy, giving back one at a time.
            while end > pos {
                if let Some(found) = match_pieces(text, end, rest, fields) {
                    return Some(found);
                }
                end -= 1;
            }
            None
        }
        Piece::Directive('z') => {
            for len in tz_candidates(text, pos) {
                fields.push(('z', text[pos..pos + len].iter().collect()));
                if let Some(found) = match_pieces(text, pos + len, rest, fields) {
                    return Some(found);
                }
                fields.pop();
            }
            None
        }
        Piece::Directive(d) => {
            for alternative in alternatives(*d) {
                let ok = alternative
                    .iter()
                    .enumerate()
                    .all(|(k, class)| text.get(pos + k).is_some_and(|c| class.contains(*c)));
                if !ok {
                    continue;
                }
                let end = pos + alternative.len();
                fields.push((*d, text[pos..end].iter().collect()));
                if let Some(found) = match_pieces(text, end, rest, fields) {
                    return Some(found);
                }
                fields.pop();
            }
            None
        }
    }
}

/// The lengths `%z`'s regex can match at `pos`, in the order it tries them:
/// `[+-]\d\d:?[0-5]\d(:?[0-5]\d(\.\d{1,6})?)?|Z`.
fn tz_candidates(text: &[char], pos: usize) -> Vec<usize> {
    let at = |k: usize| text.get(pos + k).copied();
    let digit = |k: usize| at(k).is_some_and(|c| c.is_ascii_digit());
    let low = |k: usize| at(k).is_some_and(|c| ('0'..='5').contains(&c));
    let mut out = Vec::new();
    if matches!(at(0), Some('+' | '-')) && digit(1) && digit(2) {
        for colon in [1, 0] {
            if colon == 1 && at(3) != Some(':') {
                continue;
            }
            let m = 3 + colon;
            if !(low(m) && digit(m + 1)) {
                continue;
            }
            let base = m + 2;
            for colon2 in [1, 0] {
                if colon2 == 1 && at(base) != Some(':') {
                    continue;
                }
                let s = base + colon2;
                if low(s) && digit(s + 1) {
                    let after = s + 2;
                    if at(after) == Some('.') {
                        let mut n = 0;
                        while n < 6 && digit(after + 1 + n) {
                            n += 1;
                        }
                        for k in (1..=n).rev() {
                            out.push(after + 1 + k);
                        }
                    }
                    out.push(after);
                }
            }
            out.push(base);
        }
    }
    if matches!(at(0), Some('Z')) {
        out.push(1);
    }
    out
}

/// `parse_timezone_directive`: the offset in seconds.
fn tz_offset(text: &str) -> Option<i32> {
    if text == "Z" {
        return Some(0);
    }
    let sign = if text.starts_with('-') { -1 } else { 1 };
    let body: String = text[1..].chars().filter(|&c| c != ':').collect();
    let (whole, _) = body.split_once('.').unwrap_or((&body, ""));
    let h: i32 = whole.get(0..2)?.parse().ok()?;
    let m: i32 = whole.get(2..4)?.parse().ok()?;
    let s: i32 = whole.get(4..6).map_or(Some(0), |s| s.parse().ok())?;
    Some(sign * (h * 3600 + m * 60 + s))
}

/// pandas' `_parse_with_format` with `exact=True`.
fn parse_with_format(text: &str, format: &str) -> Result<Stamp, ()> {
    let pieces = pattern(format)?;
    let chars: Vec<char> = text.chars().collect();
    let mut fields = Vec::new();
    let end = match_pieces(&chars, 0, &pieces, &mut fields).ok_or(())?;
    if end != chars.len() {
        return Err(());
    }
    let (mut year, mut month, mut day) = (1900i64, 1u32, 1u32);
    let (mut hour, mut minute, mut second) = (0u32, 0u32, 0u32);
    let mut offset = None;
    for (directive, value) in &fields {
        let n = || value.trim().parse::<i64>().map_err(|_| ());
        match directive {
            'Y' => year = n()?,
            'm' => month = n()? as u32,
            'd' => day = n()? as u32,
            'H' => hour = n()? as u32,
            'M' => minute = n()? as u32,
            'S' => second = n()? as u32,
            'z' => offset = Some(tz_offset(value).ok_or(())?),
            _ => return Err(()),
        }
    }
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return Err(());
    }
    if day < 1 || day > days_in_month(year, month) {
        return Err(());
    }
    let mut stamp = Stamp {
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond: 0,
        offset,
    };
    if second >= 60 {
        // `%S` takes 60 and 61, which run on into the next minute.
        stamp.second = second - 60;
        stamp = next_minute(stamp);
    }
    Ok(stamp)
}

/// `stamp` one minute later.
fn next_minute(mut stamp: Stamp) -> Stamp {
    {
        stamp.minute += 1;
        if stamp.minute == 60 {
            stamp.minute = 0;
            stamp.hour += 1;
            if stamp.hour == 24 {
                stamp.hour = 0;
                stamp.day += 1;
                if stamp.day > days_in_month(stamp.year, stamp.month) {
                    stamp.day = 1;
                    stamp.month += 1;
                    if stamp.month > 12 {
                        stamp.month = 1;
                        stamp.year += 1;
                    }
                }
            }
        }
    }
    stamp
}

// --- pandas: ISO 8601 -----------------------------------------------------------

/// The format `parse_iso_8601_datetime` compares against as it goes.
struct Format<'a> {
    rest: &'a [u8],
    exact: bool,
}

impl Format<'_> {
    /// `compare_format`: the format must go on with `want`.
    fn take(&mut self, want: &[u8]) -> Result<(), ()> {
        if !self.exact {
            return Ok(());
        }
        if self.rest.len() < want.len() || &self.rest[..want.len()] != want {
            return Err(());
        }
        self.rest = &self.rest[want.len()..];
        Ok(())
    }

    /// The format must be used up.
    fn done(&self, stamp: Stamp) -> Result<Stamp, ()> {
        if self.exact && !self.rest.is_empty() {
            Err(())
        } else {
            Ok(stamp)
        }
    }
}

/// pandas' `parse_iso_8601_datetime`, with a format to match exactly or none
/// to infer.
fn iso8601(text: &str, format: Option<&str>) -> Result<Stamp, ()> {
    let s = text.as_bytes();
    let mut f = Format {
        rest: format.map_or(b"", str::as_bytes),
        exact: format.is_some(),
    };
    let digit = |k: usize| s.get(k).is_some_and(u8::is_ascii_digit);
    let val = |k: usize| u32::from(s[k] - b'0');
    let mut p = 0;
    while p < s.len() && is_c_space(s[p]) {
        p += 1;
        f.take(b" ")?;
    }
    let negative = s.get(p) == Some(&b'-');
    if negative {
        p += 1;
    }
    if p >= s.len() {
        return Err(());
    }
    f.take(b"%Y")?;
    let mut year: i64 = 0;
    if s.len() - p >= 4 && (0..4).all(|k| digit(p + k)) {
        year = i64::from(1000 * val(p) + 100 * val(p + 1) + 10 * val(p + 2) + val(p + 3));
        p += 4;
    }
    if negative {
        year = -year;
    }
    let mut stamp = Stamp {
        year,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        microsecond: 0,
        offset: None,
    };
    if p == s.len() {
        return f.done(stamp);
    }
    let mut sep = None;
    if !digit(p) {
        let c = s[p];
        if !matches!(c, b'-' | b'.' | b'/' | b'\\' | b' ') {
            return Err(());
        }
        sep = Some(c);
        p += 1;
        f.take(&[c])?;
        if p == s.len() || !digit(p) {
            return Err(());
        }
    }
    f.take(b"%m")?;
    let mut month = val(p);
    p += 1;
    if digit(p) {
        month = 10 * month + val(p);
        p += 1;
    } else if sep.is_none() {
        return Err(());
    }
    if !(1..=12).contains(&month) {
        return Err(());
    }
    stamp.month = month;
    if p == s.len() {
        if sep.is_none() {
            return Err(());
        }
        return f.done(stamp);
    }
    if let Some(sep) = sep {
        if s[p] != sep || p + 1 == s.len() {
            return Err(());
        }
        p += 1;
        f.take(&[sep])?;
    }
    f.take(b"%d")?;
    if !digit(p) {
        return Err(());
    }
    let mut day = val(p);
    p += 1;
    if digit(p) {
        day = 10 * day + val(p);
        p += 1;
    } else if sep.is_none() {
        return Err(());
    }
    if day < 1 || day > days_in_month(year, month) {
        return Err(());
    }
    stamp.day = day;
    if p == s.len() {
        return f.done(stamp);
    }
    if (s[p] != b'T' && s[p] != b' ') || p + 1 == s.len() {
        return Err(());
    }
    f.take(&s[p..p + 1])?;
    p += 1;
    f.take(b"%H")?;
    if !digit(p) {
        return Err(());
    }
    let mut hour = val(p);
    p += 1;
    let mut hour_two = false;
    if digit(p) {
        hour_two = true;
        hour = 10 * hour + val(p);
        p += 1;
        if hour >= 24 {
            return Err(());
        }
    }
    stamp.hour = hour;
    if p == s.len() {
        if !hour_two {
            return Err(());
        }
        return f.done(stamp);
    }
    let mut hms_sep = false;
    if s[p] == b':' {
        hms_sep = true;
        p += 1;
        if p == s.len() || !digit(p) {
            return Err(());
        }
        f.take(b":")?;
    } else if !digit(p) {
        if !hour_two {
            return Err(());
        }
        return iso_timezone(s, p, stamp, &mut f);
    }
    f.take(b"%M")?;
    let mut minute = val(p);
    p += 1;
    if digit(p) {
        minute = 10 * minute + val(p);
        p += 1;
        if minute >= 60 {
            return Err(());
        }
    } else if !hms_sep {
        return Err(());
    }
    stamp.minute = minute;
    if p == s.len() {
        return f.done(stamp);
    }
    if hms_sep && s[p] == b':' {
        f.take(b":")?;
        p += 1;
        if p == s.len() || !digit(p) {
            return Err(());
        }
    } else if !(!hms_sep && digit(p)) {
        return iso_timezone(s, p, stamp, &mut f);
    }
    f.take(b"%S")?;
    let mut second = val(p);
    p += 1;
    if digit(p) {
        second = 10 * second + val(p);
        p += 1;
        if second >= 60 {
            return Err(());
        }
    } else if !hms_sep {
        return Err(());
    }
    stamp.second = second;
    if p < s.len() && s[p] == b'.' {
        // Fractions never occur in a date constant's shape.
        return Err(());
    }
    iso_timezone(s, p, stamp, &mut f)
}

/// C's `isspace` in the "C" locale.
fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// The end of `parse_iso_8601_datetime`: an optional zone, then nothing.
fn iso_timezone(s: &[u8], mut p: usize, mut stamp: Stamp, f: &mut Format<'_>) -> Result<Stamp, ()> {
    let digit = |k: usize| s.get(k).is_some_and(u8::is_ascii_digit);
    while p < s.len() && is_c_space(s[p]) {
        p += 1;
        f.take(b" ")?;
    }
    if p == s.len() {
        return f.done(stamp);
    }
    if s[p] == b'Z' {
        f.take(b"%z")?;
        stamp.offset = Some(0);
        if p + 1 == s.len() {
            return f.done(stamp);
        }
        p += 1;
    } else if s[p] == b'-' || s[p] == b'+' {
        f.take(b"%z")?;
        let negative = s[p] == b'-';
        p += 1;
        let (h, mut m);
        if digit(p) && digit(p + 1) {
            h = i32::from(s[p] - b'0') * 10 + i32::from(s[p + 1] - b'0');
            p += 2;
            if h >= 24 {
                return Err(());
            }
        } else if digit(p) {
            h = i32::from(s[p] - b'0');
            p += 1;
        } else {
            return Err(());
        }
        m = 0;
        if p < s.len() {
            if s[p] == b':' {
                p += 1;
            }
            if digit(p) && digit(p + 1) {
                m = i32::from(s[p] - b'0') * 10 + i32::from(s[p + 1] - b'0');
                p += 2;
                if m >= 60 {
                    return Err(());
                }
            } else if digit(p) {
                m = i32::from(s[p] - b'0');
                p += 1;
            } else {
                return Err(());
            }
        }
        let offset = 60 * (h * 60 + m);
        stamp.offset = Some(if negative { -offset } else { offset });
    }
    while p < s.len() && is_c_space(s[p]) {
        p += 1;
        f.take(b" ")?;
    }
    if p != s.len() {
        return Err(());
    }
    f.done(stamp)
}

// --- pandas: without a format ---------------------------------------------------

/// pandas' `_parse_delimited_date` with `dayfirst=False`.
fn delimited_date(text: &str) -> Result<Option<DateTime>, ()> {
    let b = text.as_bytes();
    let is_delimiter = |k: usize| b.get(k).is_some_and(|c| b" /-.".contains(c));
    let d = |k: usize, default: i32| -> i32 {
        b.get(k)
            .filter(|c| c.is_ascii_digit())
            .map_or(default, |c| i32::from(c - b'0'))
    };
    let two = |k: usize| d(k, -10) * 10 + d(k + 1, -100);
    let four = |k: usize| {
        d(k, -10) * 1000 + d(k + 1, -100) * 100 + d(k + 2, -1000) * 10 + d(k + 3, -10000)
    };
    let (mut month, mut day, year, can_swap);
    match b.len() {
        10 if is_delimiter(2) && is_delimiter(5) => {
            month = two(0);
            day = two(3);
            year = four(6);
            can_swap = true;
        }
        9 if is_delimiter(1) && is_delimiter(4) => {
            month = d(0, -10);
            day = two(2);
            year = four(5);
            can_swap = true;
        }
        9 if is_delimiter(2) && is_delimiter(4) => {
            month = two(0);
            day = d(3, -10);
            year = four(5);
            can_swap = true;
        }
        8 if is_delimiter(1) && is_delimiter(3) => {
            month = d(0, -10);
            day = d(2, -10);
            year = four(4);
            can_swap = true;
        }
        7 if is_delimiter(2) => {
            if b[2] == b'.' {
                return Ok(None);
            }
            month = two(0);
            day = 1;
            year = four(3);
            can_swap = false;
        }
        _ => return Ok(None),
    }
    if month < 0 || day < 0 || year < 1000 {
        return Ok(None);
    }
    if (1..=31).contains(&month) && (1..=31).contains(&day) && (month <= 12 || day <= 12) {
        if month > 12 && can_swap {
            std::mem::swap(&mut day, &mut month);
        }
        let dt = replace(
            DateTime {
                year: 1,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
                offset: None,
            },
            Replace {
                year: Some(year as Num),
                month: Some(month as Num),
                day: Some(day as Num),
                ..Replace::default()
            },
        )
        .ok_or(())?;
        return Ok(Some(dt));
    }
    Err(())
}

/// pandas' `_does_string_look_like_datetime`.
fn looks_like_datetime(text: &str) -> bool {
    if text.starts_with('0') {
        return true;
    }
    if matches!(text, "a" | "A" | "m" | "M" | "p" | "P" | "t" | "T") {
        return false;
    }
    match text.trim().parse::<f64>() {
        Ok(value) if !text.trim().is_empty() => value >= 1000.0,
        _ => true,
    }
}

/// pandas' `_does_string_look_like_time`.
fn looks_like_time(text: &str) -> bool {
    let b = text.as_bytes();
    if b.len() < 4 {
        return false;
    }
    let d = |k: usize, default: i32| {
        b.get(k)
            .filter(|c| c.is_ascii_digit())
            .map_or(default, |c| i32::from(c - b'0'))
    };
    let (hour, minute) = if b[1] == b':' {
        (d(0, -1), d(2, -10) * 10 + d(3, -100))
    } else if b[2] == b':' {
        (d(0, -10) * 10 + d(1, -100), d(3, -10) * 10 + d(4, -100))
    } else {
        (-1, -1)
    };
    (0..=23).contains(&hour) && (0..=59).contains(&minute)
}

/// `pd.to_datetime` when no format was guessed: ISO 8601, then pandas'
/// fixed readings, then `dateutil` over 0001-01-01.
fn parse_without_format(text: &str, year: i64) -> Result<Stamp, ()> {
    if let Ok(stamp) = iso8601(text, None) {
        return Ok(stamp);
    }
    if !looks_like_datetime(text) {
        return Err(());
    }
    let default = DateTime {
        year: 1,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0,
        microsecond: 0,
        offset: None,
    };
    if looks_like_time(text) {
        // A bare time of day is never a date constant's shape.
        return Err(());
    }
    if let Some(dt) = delimited_date(text)? {
        return Ok(dt.stamp());
    }
    dateutil_parse(text, default, year)
        .map(DateTime::stamp)
        .ok_or(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_on(literal: &str, pads_year: bool) -> (Option<String>, usize) {
        let host = Host {
            year: 2026,
            pads_year,
        };
        let reading = to_datetime(literal, host);
        (reading.value.map(|s| s.iso()), reading.warnings.len())
    }

    fn read(literal: &str) -> (Option<String>, usize) {
        let padded = read_on(literal, true);
        assert_eq!(padded, read_on(literal, false), "{literal}");
        padded
    }

    #[test]
    fn a_year_below_1000_warns_only_where_strftime_pads_it() {
        for literal in ["13/01/0999", "31-12-0001 12:30:00", "29.02.0004"] {
            let (value, _) = read_on(literal, true);
            assert!(value.is_some(), "{literal}");
            assert_eq!(read_on(literal, true), (value.clone(), 1), "{literal}");
            assert_eq!(read_on(literal, false), (value, 0), "{literal}");
        }
        // Three digits are never a `%Y`, padded or not.
        assert_eq!(read_on("13/01/999", true), read_on("13/01/999", false));
    }

    #[test]
    fn readings_pandas_gives() {
        for (literal, value, warnings) in [
            ("01.02.2024", Some("2024-01-02T00:00:00"), 0),
            ("29.02.2024", Some("2024-02-29T00:00:00"), 1),
            ("24-01-05", Some("2005-01-24T00:00:00"), 0),
            ("1-1-1", Some("2001-01-01T00:00:00"), 0),
            ("0000-01-01", Some("0000-01-01T00:00:00"), 0),
            ("230203-06-09", Some("2003-02-23T06:00:00-09:00"), 0),
            ("2024-01-02 001:002:003", Some("2024-01-02T01:02:03"), 0),
            ("2023-02-29", None, 0),
            ("13/13/2024", None, 0),
        ] {
            assert_eq!(
                read(literal),
                (value.map(String::from), warnings),
                "{literal}"
            );
        }
    }
}
