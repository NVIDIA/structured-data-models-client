// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading text with a `strptime` format, as pandas' `_parse_with_format`
//! (and Python's `_strptime`) does: the format becomes the regular
//! expression `_strptime.TimeRE` builds, matched case-insensitively from the
//! start with Python's backtracking, and its groups become the fields.
//!
//! The expression is not built as text: each directive lists the lengths
//! its alternatives match, in the order Python's `re` tries them, so the
//! first match found is the one Python finds.

use super::civil::{self, Wall};
use crate::py;

/// Which `TimeRE`: pandas' (`%f` up to nine digits, `%Z` any zone
/// `zoneinfo` lists) or the standard library's.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Pandas,
    Stdlib,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Elem {
    /// A literal character, matched ignoring case.
    Lit(char),
    /// `'`, which matches `'` or U+02BC.
    Quote,
    /// `\s+`.
    Spaces,
    /// A directive, by its letter.
    Dir(char),
}

/// A format's regular expression.
pub struct Compiled {
    elems: Vec<Elem>,
    flavor: Flavor,
}

/// `_strptime`'s errors before any value is read.
pub fn compile(fmt: &str, flavor: Flavor) -> Result<Compiled, String> {
    let chars: Vec<char> = fmt.chars().collect();
    let mut elems = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '%' {
            let Some(&k) = chars.get(i + 1) else {
                return Err(format!("stray % in format '{fmt}'"));
            };
            if py::is_space(k) || k == '\n' {
                return Err(format!("'%' is a bad directive in format '{fmt}'"));
            }
            let (key, used) = if k == 'O' {
                match chars.get(i + 2) {
                    Some(&d) if "dmyHIMSw".contains(d) => (d, 3),
                    Some(&d) => return Err(format!("'O{d}' is a bad directive in format '{fmt}'")),
                    None => return Err(format!("'O' is a bad directive in format '{fmt}'")),
                }
            } else {
                (k, 2)
            };
            if !"dfHIGjmMSUwuVyYzAaBbpZW%".contains(key) {
                return Err(format!("'{key}' is a bad directive in format '{fmt}'"));
            }
            if key == '%' {
                elems.push(Elem::Lit('%'));
            } else {
                elems.push(Elem::Dir(key));
            }
            i += used;
        } else if py::is_space(c) {
            while i < chars.len() && py::is_space(chars[i]) {
                i += 1;
            }
            elems.push(Elem::Spaces);
        } else if c == '\'' {
            elems.push(Elem::Quote);
            i += 1;
        } else {
            elems.push(Elem::Lit(c));
            i += 1;
        }
    }
    Ok(Compiled { elems, flavor })
}

/// pandas' `_validate_fmt`.
pub fn validate(fmt: &str) -> Result<(), String> {
    let has = |d: &str| fmt.contains(d);
    if has("%W") || has("%U") {
        if !has("%Y") && !has("%y") {
            return Err("Cannot use '%W' or '%U' without day and year".to_string());
        }
        if !has("%A") && !has("%a") && !has("%w") {
            return Err("Cannot use '%W' or '%U' without day and year".to_string());
        }
    } else if has("%Z") && has("%z") {
        return Err("Cannot parse both %Z and %z".to_string());
    } else if has("%j") && has("%G") {
        return Err("Day of the year directive '%j' is not compatible with ISO year directive '%G'. Use '%Y' instead.".to_string());
    } else if has("%G") && (!has("%V") || !(has("%A") || has("%a") || has("%w") || has("%u"))) {
        return Err("ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.".to_string());
    } else if has("%V") && has("%Y") {
        return Err("ISO week directive '%V' is incompatible with the year directive '%Y'. Use the ISO year '%G' instead.".to_string());
    } else if has("%V") && (!has("%G") || !(has("%A") || has("%a") || has("%w") || has("%u"))) {
        return Err("ISO week directive '%V' must be used with the ISO year directive '%G' and a weekday directive '%A', '%a', '%w', or '%u'.".to_string());
    }
    Ok(())
}

/// A character class of a fixed-length alternative.
#[derive(Clone, Copy)]
enum Class {
    C(char),
    R(char, char),
    D,
}

fn class_matches(class: Class, c: char) -> bool {
    match class {
        Class::C(x) => c == x,
        Class::R(a, b) => (a..=b).contains(&c),
        Class::D => py::is_digit(c) && c.is_ascii_digit() || (!c.is_ascii() && py::is_digit(c)),
    }
}

fn alternatives(key: char) -> &'static [&'static [Class]] {
    use Class::*;
    match key {
        'd' => &[
            &[C('3'), R('0', '1')],
            &[R('1', '2'), D],
            &[C('0'), R('1', '9')],
            &[R('1', '9')],
            &[C(' '), R('1', '9')],
        ],
        'H' => &[&[C('2'), R('0', '3')], &[R('0', '1'), D], &[D]],
        'I' => &[
            &[C('1'), R('0', '2')],
            &[C('0'), R('1', '9')],
            &[R('1', '9')],
            &[C(' '), R('1', '9')],
        ],
        'G' | 'Y' => &[&[D, D, D, D]],
        'y' => &[&[D, D]],
        'j' => &[
            &[C('3'), C('6'), R('0', '6')],
            &[C('3'), R('0', '5'), D],
            &[R('1', '2'), D, D],
            &[C('0'), R('1', '9'), D],
            &[C('0'), C('0'), R('1', '9')],
            &[R('1', '9'), D],
            &[C('0'), R('1', '9')],
            &[R('1', '9')],
        ],
        'm' => &[
            &[C('1'), R('0', '2')],
            &[C('0'), R('1', '9')],
            &[R('1', '9')],
        ],
        'M' => &[&[R('0', '5'), D], &[D]],
        'S' => &[&[C('6'), R('0', '1')], &[R('0', '5'), D], &[D]],
        'U' | 'W' => &[&[C('5'), R('0', '3')], &[R('0', '4'), D], &[D]],
        'w' => &[&[R('0', '6')]],
        'u' => &[&[R('1', '7')]],
        'V' => &[
            &[C('5'), R('0', '3')],
            &[C('0'), R('1', '9')],
            &[R('1', '4'), D],
            &[D],
        ],
        _ => &[],
    }
}

fn lower_eq(a: char, b: char) -> bool {
    a == b || a.to_lowercase().eq(b.to_lowercase())
}

/// Whether `name` (lower-case) is at `pos`, ignoring case.
fn name_at(text: &[char], pos: usize, name: &str) -> Option<usize> {
    let mut p = pos;
    for n in name.chars() {
        let c = *text.get(p)?;
        if !lower_eq(c, n) {
            return None;
        }
        p += 1;
    }
    Some(p)
}

fn names(key: char) -> Vec<String> {
    let list: Vec<String> = match key {
        'A' => civil::DAYS_LONG.iter().map(|s| s.to_lowercase()).collect(),
        'a' => civil::DAYS_SHORT.iter().map(|s| s.to_lowercase()).collect(),
        'B' => civil::MONTHS_LONG
            .iter()
            .map(|s| s.to_lowercase())
            .collect(),
        'b' => civil::MONTHS_SHORT
            .iter()
            .map(|s| s.to_lowercase())
            .collect(),
        'p' => vec!["am".to_string(), "pm".to_string()],
        _ => Vec::new(),
    };
    let mut sorted = list;
    sorted.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
    sorted
}

/// The ends a directive can match at `pos`, in the order Python tries them.
fn candidates(key: char, flavor: Flavor, text: &[char], pos: usize) -> Vec<usize> {
    let mut out = Vec::new();
    match key {
        'f' => {
            let most = if flavor == Flavor::Pandas { 9 } else { 6 };
            let mut n = 0;
            while n < most && text.get(pos + n).is_some_and(|c| c.is_ascii_digit()) {
                n += 1;
            }
            for k in (1..=n).rev() {
                out.push(pos + k);
            }
        }
        'z' => {
            let digit = |p: usize| text.get(p).is_some_and(|&c| class_matches(Class::D, c));
            let in_range =
                |p: usize, a: char, b: char| text.get(p).is_some_and(|c| (a..=b).contains(c));
            if text.get(pos).is_some_and(|&c| c == '+' || c == '-')
                && digit(pos + 1)
                && digit(pos + 2)
            {
                let mut colon_options = Vec::new();
                if text.get(pos + 3) == Some(&':') {
                    colon_options.push(pos + 4);
                }
                colon_options.push(pos + 3);
                for p in colon_options {
                    if !(in_range(p, '0', '5') && digit(p + 1)) {
                        continue;
                    }
                    let base = p + 2;
                    // (:?[0-5]\d(\.\d{1,6})?)?, greedily.
                    let mut inner = Vec::new();
                    if text.get(base) == Some(&':') {
                        inner.push(base + 1);
                    }
                    inner.push(base);
                    for q in inner {
                        if in_range(q, '0', '5') && digit(q + 1) {
                            let after = q + 2;
                            if text.get(after) == Some(&'.') {
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
            if text.get(pos) == Some(&'Z') {
                out.push(pos + 1);
            }
        }
        'Z' => {
            if flavor == Flavor::Pandas {
                let mut zones: Vec<&String> = super::zone::available().iter().collect();
                zones.sort_by_key(|s| std::cmp::Reverse(s.chars().count()));
                for zone in zones {
                    let lowered = zone.to_lowercase();
                    if let Some(end) = name_at(text, pos, &lowered) {
                        out.push(end);
                    }
                }
            } else {
                for name in ["utc", "gmt"] {
                    if let Some(end) = name_at(text, pos, name) {
                        out.push(end);
                    }
                }
            }
        }
        'A' | 'a' | 'B' | 'b' | 'p' => {
            for name in names(key) {
                if let Some(end) = name_at(text, pos, &name) {
                    out.push(end);
                }
            }
        }
        _ => {
            for alt in alternatives(key) {
                let ok = alt
                    .iter()
                    .enumerate()
                    .all(|(i, &class)| text.get(pos + i).is_some_and(|&c| class_matches(class, c)));
                if ok {
                    out.push(pos + alt.len());
                }
            }
        }
    }
    out
}

/// The first match from the start, with each directive's text: Python's
/// `regex.match(text)`.
fn first_match(compiled: &Compiled, text: &[char]) -> Option<(Vec<(char, String)>, usize)> {
    let mut groups: Vec<(char, String)> = Vec::new();
    let end = walk(compiled, text, 0, 0, &mut groups)?;
    Some((groups, end))
}

fn walk(
    compiled: &Compiled,
    text: &[char],
    index: usize,
    pos: usize,
    groups: &mut Vec<(char, String)>,
) -> Option<usize> {
    let Some(elem) = compiled.elems.get(index) else {
        return Some(pos);
    };
    match elem {
        Elem::Lit(c) => {
            let t = *text.get(pos)?;
            if lower_eq(t, *c) {
                walk(compiled, text, index + 1, pos + 1, groups)
            } else {
                None
            }
        }
        Elem::Quote => {
            let t = *text.get(pos)?;
            if t == '\'' || t == '\u{2bc}' {
                walk(compiled, text, index + 1, pos + 1, groups)
            } else {
                None
            }
        }
        Elem::Spaces => {
            let mut n = 0;
            while text.get(pos + n).is_some_and(|&c| py::is_space(c)) {
                n += 1;
            }
            for k in (1..=n).rev() {
                if let Some(end) = walk(compiled, text, index + 1, pos + k, groups) {
                    return Some(end);
                }
            }
            None
        }
        Elem::Dir(key) => {
            for end in candidates(*key, compiled.flavor, text, pos) {
                let mark = groups.len();
                groups.push((*key, text[pos..end].iter().collect()));
                if let Some(done) = walk(compiled, text, index + 1, end, groups) {
                    return Some(done);
                }
                groups.truncate(mark);
            }
            None
        }
    }
}

/// The zone a value's `%z` or `%Z` names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tz {
    /// A fixed offset, in seconds.
    Fixed(i64),
    /// A `zoneinfo` zone, by the name the text gave.
    Named(String),
}

/// What reading one value gives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Read {
    pub wall: Wall,
    /// Whether the fraction had more than six digits (nanoseconds).
    pub nanos: bool,
    pub tz: Option<Tz>,
}

fn int_of(text: &str) -> Result<i64, String> {
    let digits: String = text
        .chars()
        .map(|c| {
            c.to_digit(10)
                .and_then(|d| char::from_digit(d, 10))
                .unwrap_or(c)
        })
        .collect();
    py::int(&digits).ok_or_else(|| {
        format!(
            "invalid literal for int() with base 10: {}",
            py::str_repr(text)
        )
    })
}

/// `parse_timezone_directive`: the seconds a `%z` group names.
fn parse_z(z: &str) -> Result<i64, String> {
    if z == "Z" {
        return Ok(0);
    }
    let mut z: Vec<char> = z.chars().collect();
    if z.get(3) == Some(&':') {
        z.remove(3);
        if z.len() > 5 {
            if z.get(5) != Some(&':') {
                return Err(format!(
                    "Inconsistent use of : in {}",
                    z.iter().collect::<String>()
                ));
            }
            z.remove(5);
        }
    }
    let s = |a: usize, b: usize| -> String {
        z.get(a..b.min(z.len()))
            .map(|x| x.iter().collect())
            .unwrap_or_default()
    };
    let hours = int_of(&s(1, 3))?;
    let minutes = int_of(&s(3, 5))?;
    let seconds_text = s(5, 7);
    let seconds = if seconds_text.is_empty() {
        0
    } else {
        int_of(&seconds_text)?
    };
    let rest = s(8, z.len());
    let micro_text = format!(
        "{rest}{}",
        "0".repeat(6usize.saturating_sub(rest.chars().count()))
    );
    let micro = int_of(&micro_text)?;
    let mut total_minutes = hours * 60 + minutes + seconds / 60 + micro / 60_000_000;
    if z.first() == Some(&'-') {
        total_minutes = -total_minutes;
    }
    Ok(total_minutes * 60)
}

/// `_parse_with_format(text, fmt, exact)`; the error is its `ValueError`.
pub fn parse_with_format(
    compiled: &Compiled,
    text: &str,
    fmt: &str,
    exact: bool,
) -> Result<Read, String> {
    let chars: Vec<char> = text.chars().collect();
    let found = if exact {
        let (groups, end) = first_match(compiled, &chars)
            .ok_or_else(|| format!("time data \"{text}\" doesn't match format \"{fmt}\""))?;
        if end != chars.len() {
            let rest: String = chars[end..].iter().collect();
            return Err(format!(
                "unconverted data remains when parsing with format \"{fmt}\": \"{rest}\""
            ));
        }
        groups
    } else {
        let mut hit = None;
        for start in 0..=chars.len() {
            if let Some((groups, _)) = first_match(compiled, &chars[start..]) {
                hit = Some(groups);
                break;
            }
        }
        hit.ok_or_else(|| format!("time data \"{text}\" doesn't match format \"{fmt}\""))?
    };
    let mut year: i64 = 1900;
    let mut month: i64 = 1;
    let mut day: i64 = 1;
    let (mut hour, mut minute, mut second): (i64, i64, i64) = (0, 0, 0);
    let mut micro: i64 = 0;
    let mut nano: i64 = 0;
    let mut nanos = false;
    let mut tz: Option<Tz> = None;
    let mut weekday: i64 = -1;
    let mut julian: i64 = -1;
    let mut week_of_year: i64 = -1;
    let mut week_start: i64 = -1;
    let mut iso_year: i64 = -1;
    let mut iso_week: i64 = -1;
    let get = |k: char| -> Option<&str> {
        found
            .iter()
            .find(|(key, _)| *key == k)
            .map(|(_, v)| v.as_str())
    };
    for (key, value) in &found {
        match key {
            'y' => {
                let y = int_of(value)?;
                year = if y <= 68 { y + 2000 } else { y + 1900 };
            }
            'Y' => year = int_of(value)?,
            'm' => month = int_of(value)?,
            'B' => {
                let v = value.to_lowercase();
                month = civil::MONTHS_LONG
                    .iter()
                    .position(|m| m.to_lowercase() == v)
                    .map_or(0, |i| i as i64 + 1);
            }
            'b' => {
                let v = value.to_lowercase();
                month = civil::MONTHS_SHORT
                    .iter()
                    .position(|m| m.to_lowercase() == v)
                    .map_or(0, |i| i as i64 + 1);
            }
            'd' => day = int_of(value)?,
            'H' => hour = int_of(value)?,
            'I' => {
                hour = int_of(value)?;
                let ampm = get('p').unwrap_or("").to_lowercase();
                if ampm.is_empty() || ampm == "am" {
                    if hour == 12 {
                        hour = 0;
                    }
                } else if ampm == "pm" && hour != 12 {
                    hour += 12;
                }
            }
            'M' => minute = int_of(value)?,
            'S' => second = int_of(value)?,
            'f' => {
                let n = value.chars().count();
                nanos = n > 6;
                let padded = format!("{value}{}", "0".repeat(9usize.saturating_sub(n)));
                let all = int_of(&padded)?;
                nano = all % 1000;
                micro = all / 1000;
            }
            'A' => {
                let v = value.to_lowercase();
                weekday = civil::DAYS_LONG
                    .iter()
                    .position(|d| d.to_lowercase() == v)
                    .map_or(-1, |i| i as i64);
            }
            'a' => {
                let v = value.to_lowercase();
                weekday = civil::DAYS_SHORT
                    .iter()
                    .position(|d| d.to_lowercase() == v)
                    .map_or(-1, |i| i as i64);
            }
            'w' => {
                let w = int_of(value)?;
                weekday = if w == 0 { 6 } else { w - 1 };
            }
            'j' => julian = int_of(value)?,
            'U' | 'W' => {
                week_of_year = int_of(value)?;
                week_start = if *key == 'U' { 6 } else { 0 };
            }
            'Z' => tz = Some(Tz::Named(value.clone())),
            'z' => tz = Some(Tz::Fixed(parse_z(value)?)),
            'G' => iso_year = int_of(value)?,
            'V' => iso_week = int_of(value)?,
            'u' => weekday = int_of(value)? - 1,
            _ => {}
        }
    }
    let date_ok = |y: i64, m: i64, d: i64| -> Result<(), String> {
        if !(1..=9999).contains(&y) {
            return Err(format!("year {y} is out of range"));
        }
        if !(1..=12).contains(&m) {
            return Err("month must be in 1..12".to_string());
        }
        if d < 1 || d > i64::from(civil::days_in_month(y, m as u32)) {
            return Err("day is out of range for month".to_string());
        }
        Ok(())
    };
    if julian == -1 && weekday != -1 {
        if week_of_year != -1 {
            date_ok(year, 1, 1)?;
            let mut first_weekday = i64::from(civil::weekday(year, 1, 1));
            let mut day_of_week = weekday;
            if week_start != 0 {
                first_weekday = (first_weekday + 1) % 7;
                day_of_week = (day_of_week + 1) % 7;
            }
            let week_0_length = (7 - first_weekday) % 7;
            julian = if week_of_year == 0 {
                1 + day_of_week - first_weekday
            } else {
                1 + week_0_length + 7 * (week_of_year - 1) + day_of_week
            };
        } else if iso_year != -1 && iso_week != -1 {
            date_ok(iso_year, 1, 4)?;
            let correction = i64::from(civil::weekday(iso_year, 1, 4)) + 1 + 3;
            let mut ordinal = iso_week * 7 + weekday + 1 - correction;
            let mut y = iso_year;
            if ordinal < 1 {
                y -= 1;
                let days_prev = if civil::is_leap(y) { 366 } else { 365 };
                ordinal += days_prev;
            }
            year = y;
            julian = ordinal;
        }
    }
    if julian == -1 {
        date_ok(year, month, day)?;
    } else {
        date_ok(year, 1, 1)?;
        let days = civil::days_from_civil(year, 1, 1) + julian - 1;
        let (y, m, d) = civil::civil_from_days(days);
        date_ok(y, i64::from(m), i64::from(d))?;
        year = y;
        month = i64::from(m);
        day = i64::from(d);
    }
    if !(0..24).contains(&hour) || !(0..60).contains(&minute) || !(0..62).contains(&second) {
        return Err("time out of range".to_string());
    }
    Ok(Read {
        wall: Wall {
            year,
            month: month as u32,
            day: day as u32,
            hour: hour as u32,
            minute: minute as u32,
            second: second as u32,
            nano: (micro * 1000 + nano) as u32,
        },
        nanos,
        tz,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backtracks_as_python() {
        let c = compile("%Y%m%d", Flavor::Pandas).unwrap();
        let got = parse_with_format(&c, "2021110", "%Y%m%d", true).unwrap();
        assert_eq!((got.wall.month, got.wall.day), (1, 10));
        let c = compile("%d %b %Y", Flavor::Pandas).unwrap();
        let got = parse_with_format(&c, "5 jan  2021", "%d %b %Y", true).unwrap();
        assert_eq!((got.wall.year, got.wall.month, got.wall.day), (2021, 1, 5));
        let c = compile("%Y-%m-%d %H:%M:%S%z", Flavor::Pandas).unwrap();
        let got = parse_with_format(&c, "2021-01-01 10:00:00+05:30", "", true).unwrap();
        assert_eq!(got.tz, Some(Tz::Fixed(19_800)));
    }
}
