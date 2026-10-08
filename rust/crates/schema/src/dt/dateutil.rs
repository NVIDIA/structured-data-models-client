// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dateutil.parser.parser._parse` (dateutil 2.9), which pandas calls to read
//! a date it cannot read as ISO 8601: the lexer, the token loop and the
//! year, month and day resolution, without fuzzy parsing.

use crate::py;

/// A number token as dateutil reads it, `decimal.Decimal(token)`: digits
/// with at most one decimal point, held exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dec {
    mantissa: i128,
    scale: u32,
}

impl Dec {
    /// Parse a token Python's `float()` accepts; `None` for the rest (and for
    /// digits too long to hold, which dateutil reads but no date can use).
    pub fn parse(text: &str) -> Option<Dec> {
        let body = py::strip(text);
        let (negative, rest) = match body.as_bytes().first() {
            Some(b'-') => (true, &body[1..]),
            Some(b'+') => (false, &body[1..]),
            _ => (false, body),
        };
        let mut mantissa: i128 = 0;
        let mut scale = 0u32;
        let mut seen_point = false;
        let mut digits = 0;
        for c in rest.chars() {
            if c == '.' {
                if seen_point {
                    return None;
                }
                seen_point = true;
                continue;
            }
            if c == '_' {
                continue;
            }
            let d = c.to_digit(10)?;
            mantissa = mantissa.checked_mul(10)?.checked_add(i128::from(d))?;
            digits += 1;
            if seen_point {
                scale += 1;
            }
        }
        if digits == 0 || scale > 30 {
            return None;
        }
        Some(Dec {
            mantissa: if negative { -mantissa } else { mantissa },
            scale,
        })
    }

    fn pow(&self) -> i128 {
        10_i128.pow(self.scale)
    }

    /// `int(value)`: truncated toward zero.
    pub fn trunc(&self) -> i128 {
        self.mantissa / self.pow()
    }

    /// Whether `value % 1` is non-zero.
    fn has_fraction(&self) -> bool {
        self.mantissa % self.pow() != 0
    }

    /// `int(n * (value % 1))`, the fractional part scaled and truncated.
    fn scaled_fraction(&self, n: i128) -> i128 {
        (self.mantissa % self.pow()) * n / self.pow()
    }

    /// `value` compared with an integer.
    fn cmp_int(&self, other: i128) -> std::cmp::Ordering {
        self.mantissa.cmp(&(other * self.pow()))
    }

    fn gt(&self, other: i128) -> bool {
        self.cmp_int(other).is_gt()
    }

    fn in_range(&self, low: i128, high: i128) -> bool {
        !self.cmp_int(low).is_lt() && !self.cmp_int(high).is_gt()
    }
}

/// `dateutil.parser._timelex.split`.
pub fn timelex(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().filter(|&c| c != '\0').collect();
    let mut pos = 0;
    let mut charstack: Vec<char> = Vec::new();
    let mut tokenstack: Vec<String> = Vec::new();
    let mut out = Vec::new();
    let mut eof = false;
    loop {
        if !tokenstack.is_empty() {
            out.push(tokenstack.remove(0));
            continue;
        }
        if eof {
            break;
        }
        let mut seenletters = false;
        let mut token: Option<String> = None;
        let mut state: Option<&str> = None;
        loop {
            let next = if !charstack.is_empty() {
                Some(charstack.remove(0))
            } else if pos < chars.len() {
                pos += 1;
                Some(chars[pos - 1])
            } else {
                None
            };
            let Some(c) = next else {
                eof = true;
                break;
            };
            match state {
                None => {
                    token = Some(c.to_string());
                    if py::is_alpha(c) {
                        state = Some("a");
                    } else if py::is_digit(c) {
                        state = Some("0");
                    } else if py::is_space(c) {
                        token = Some(" ".to_string());
                        break;
                    } else {
                        break;
                    }
                }
                Some("a") => {
                    seenletters = true;
                    let t = token.as_mut().expect("a token is open");
                    if py::is_alpha(c) {
                        t.push(c);
                    } else if c == '.' {
                        t.push(c);
                        state = Some("a.");
                    } else {
                        charstack.push(c);
                        break;
                    }
                }
                Some("0") => {
                    let t = token.as_mut().expect("a token is open");
                    if py::is_digit(c) {
                        t.push(c);
                    } else if c == '.' || (c == ',' && t.chars().count() >= 2) {
                        t.push(c);
                        state = Some("0.");
                    } else {
                        charstack.push(c);
                        break;
                    }
                }
                Some("a.") => {
                    seenletters = true;
                    let t = token.as_mut().expect("a token is open");
                    if c == '.' || py::is_alpha(c) {
                        t.push(c);
                    } else if py::is_digit(c) && t.ends_with('.') {
                        t.push(c);
                        state = Some("0.");
                    } else {
                        charstack.push(c);
                        break;
                    }
                }
                _ => {
                    let t = token.as_mut().expect("a token is open");
                    if c == '.' || py::is_digit(c) {
                        t.push(c);
                    } else if py::is_alpha(c) && t.ends_with('.') {
                        t.push(c);
                        state = Some("a.");
                    } else {
                        charstack.push(c);
                        break;
                    }
                }
            }
        }
        let Some(mut token) = token else {
            break;
        };
        if matches!(state, Some("a.") | Some("0."))
            && (seenletters
                || token.matches('.').count() > 1
                || token.ends_with('.')
                || token.ends_with(','))
        {
            // re.split('([.,])', token): pieces and the separators.
            let mut pieces: Vec<String> = Vec::new();
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
            for piece in pieces.into_iter().skip(1) {
                if !piece.is_empty() {
                    tokenstack.push(piece);
                }
            }
        }
        if state == Some("0.") && !token.contains('.') {
            token = token.replace(',', ".");
        }
        out.push(token);
    }
    out
}

const JUMP: [&str; 18] = [
    " ", ".", ",", ";", "-", "/", "'", "at", "on", "and", "ad", "m", "t", "of", "st", "nd", "rd",
    "th",
];
const UTCZONE: [&str; 4] = ["UTC", "GMT", "Z", "z"];

fn lower(text: &str) -> String {
    text.to_lowercase()
}

fn jump(name: &str) -> bool {
    JUMP.contains(&lower(name).as_str())
}

/// Monday is 0.
fn weekday(name: &str) -> Option<u32> {
    let name = lower(name);
    super::civil::DAYS_SHORT
        .iter()
        .zip(super::civil::DAYS_LONG)
        .position(|(short, long)| name == short.to_lowercase() || name == long.to_lowercase())
        .map(|i| i as u32)
}

/// January is 1.
pub fn month(name: &str) -> Option<u32> {
    let name = lower(name);
    if name == "sept" {
        return Some(9);
    }
    super::civil::MONTHS_SHORT
        .iter()
        .zip(super::civil::MONTHS_LONG)
        .position(|(short, long)| name == short.to_lowercase() || name == long.to_lowercase())
        .map(|i| i as u32 + 1)
}

/// 0 for hours, 1 minutes, 2 seconds.
fn hms(name: &str) -> Option<u32> {
    match lower(name).as_str() {
        "h" | "hour" | "hours" => Some(0),
        "m" | "minute" | "minutes" => Some(1),
        "s" | "second" | "seconds" => Some(2),
        _ => None,
    }
}

/// 0 for am, 1 for pm.
fn ampm(name: &str) -> Option<u32> {
    match lower(name).as_str() {
        "am" | "a" => Some(0),
        "pm" | "p" => Some(1),
        _ => None,
    }
}

fn pertain(name: &str) -> bool {
    lower(name) == "of"
}

fn utczone(name: &str) -> bool {
    matches!(lower(name).as_str(), "utc" | "gmt" | "z")
}

/// `parserinfo.tzoffset`: only the lower-case spellings are found, as the
/// lookup is against lower-cased keys without lower-casing the name.
fn tzoffset_of(name: &str) -> Option<i64> {
    if matches!(name, "utc" | "gmt" | "z") {
        Some(0)
    } else {
        None
    }
}

/// `parserinfo.convertyear`, against the local year `now`.
pub fn convert_year(year: i64, century_specified: bool, now: i64) -> i64 {
    let mut year = year;
    if year < 100 && !century_specified {
        let century = now / 100 * 100;
        year += century;
        if year >= now + 50 {
            year -= 100;
        } else if year < now - 50 {
            year += 100;
        }
    }
    year
}

/// What `_parse` found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Res {
    pub year: Option<i64>,
    pub month: Option<i64>,
    pub day: Option<i64>,
    pub weekday: Option<u32>,
    pub hour: Option<i64>,
    pub minute: Option<i64>,
    pub second: Option<i64>,
    pub microsecond: Option<i64>,
    pub tzname: Option<String>,
    pub tzoffset: Option<i64>,
    pub ampm: Option<u32>,
    pub century_specified: bool,
}

/// The parse failed: dateutil's `_parse` returns `(None, None)`.
#[derive(Debug)]
struct Fail;

type R<T> = Result<T, Fail>;

#[derive(Default)]
struct Ymd {
    values: Vec<i64>,
    century_specified: bool,
    dstridx: Option<usize>,
    mstridx: Option<usize>,
    ystridx: Option<usize>,
}

enum Val<'a> {
    Text(&'a str),
    Num(Dec),
    Int(i64),
}

impl Ymd {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn could_be_day(&self, value: &Dec) -> R<bool> {
        if self.dstridx.is_some() {
            return Ok(false);
        }
        let Some(m) = self.mstridx else {
            return Ok(value.in_range(1, 31));
        };
        let month = self.values[m];
        if !(1..=12).contains(&month) {
            return Err(Fail);
        }
        let year = match self.ystridx {
            None => 2000,
            Some(y) => self.values[y],
        };
        let last = super::civil::days_in_month(year, month as u32);
        Ok(value.in_range(1, i128::from(last)))
    }

    fn append(&mut self, value: Val<'_>, label: Option<char>) -> R<()> {
        let mut label = label;
        let int = match value {
            Val::Text(text) => {
                if py::all_digits(text) && text.chars().count() > 2 {
                    self.century_specified = true;
                    if !matches!(label, None | Some('Y')) {
                        return Err(Fail);
                    }
                    label = Some('Y');
                }
                py::int(text).ok_or(Fail)?
            }
            Val::Num(dec) => {
                if dec.gt(100) {
                    self.century_specified = true;
                    if !matches!(label, None | Some('Y')) {
                        return Err(Fail);
                    }
                    label = Some('Y');
                }
                i64::try_from(dec.trunc()).map_err(|_| Fail)?
            }
            Val::Int(n) => {
                if n > 100 {
                    self.century_specified = true;
                    if !matches!(label, None | Some('Y')) {
                        return Err(Fail);
                    }
                    label = Some('Y');
                }
                n
            }
        };
        self.values.push(int);
        let index = self.values.len() - 1;
        match label {
            Some('M') => {
                if self.mstridx.is_some() {
                    return Err(Fail);
                }
                self.mstridx = Some(index);
            }
            Some('D') => {
                if self.dstridx.is_some() {
                    return Err(Fail);
                }
                self.dstridx = Some(index);
            }
            Some('Y') => {
                if self.ystridx.is_some() {
                    return Err(Fail);
                }
                self.ystridx = Some(index);
            }
            _ => {}
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    fn resolve(
        &self,
        yearfirst: bool,
        dayfirst: bool,
    ) -> R<(Option<i64>, Option<i64>, Option<i64>)> {
        let v = &self.values;
        let len = v.len();
        let mut strids: Vec<(char, usize)> = Vec::new();
        for (key, idx) in [
            ('y', self.ystridx),
            ('m', self.mstridx),
            ('d', self.dstridx),
        ] {
            if let Some(idx) = idx {
                strids.push((key, idx));
            }
        }
        if (len == strids.len() && len > 0) || (len == 3 && strids.len() == 2) {
            if len == 3 && strids.len() == 2 {
                let missing: Vec<usize> = (0..3)
                    .filter(|x| !strids.iter().any(|&(_, idx)| idx == *x))
                    .collect();
                let key: Vec<char> = ['y', 'm', 'd']
                    .into_iter()
                    .filter(|k| !strids.iter().any(|&(key, _)| key == *k))
                    .collect();
                if missing.len() != 1 || key.len() != 1 {
                    return Err(Fail);
                }
                strids.push((key[0], missing[0]));
            }
            if len != strids.len() {
                return Err(Fail);
            }
            let get = |k: char| {
                strids
                    .iter()
                    .find(|&&(key, _)| key == k)
                    .map(|&(_, i)| v[i])
            };
            return Ok((get('y'), get('m'), get('d')));
        }
        let (mut year, mut month, mut day) = (None, None, None);
        let mstridx = self.mstridx;
        if len > 3 {
            return Err(Fail);
        } else if len == 1 || (mstridx.is_some() && len == 2) {
            let other;
            if let Some(m) = mstridx {
                month = Some(v[m]);
                // self[mstridx - 1] wraps around for index 0.
                other = if m == 0 { v[len - 1] } else { v[m - 1] };
            } else {
                other = v[0];
            }
            if len > 1 || mstridx.is_none() {
                if other > 31 {
                    year = Some(other);
                } else {
                    day = Some(other);
                }
            }
        } else if len == 2 {
            if v[0] > 31 {
                year = Some(v[0]);
                month = Some(v[1]);
            } else if v[1] > 31 {
                month = Some(v[0]);
                year = Some(v[1]);
            } else if dayfirst && v[1] <= 12 {
                day = Some(v[0]);
                month = Some(v[1]);
            } else {
                month = Some(v[0]);
                day = Some(v[1]);
            }
        } else if len == 3 {
            let (a, b, c) = (v[0], v[1], v[2]);
            match mstridx {
                Some(0) => {
                    if b > 31 {
                        (month, year, day) = (Some(a), Some(b), Some(c));
                    } else {
                        (month, day, year) = (Some(a), Some(b), Some(c));
                    }
                }
                Some(1) => {
                    if a > 31 || (yearfirst && c <= 31) {
                        (year, month, day) = (Some(a), Some(b), Some(c));
                    } else {
                        (day, month, year) = (Some(a), Some(b), Some(c));
                    }
                }
                Some(2) => {
                    if b > 31 {
                        (day, year, month) = (Some(a), Some(b), Some(c));
                    } else {
                        (year, day, month) = (Some(a), Some(b), Some(c));
                    }
                }
                _ => {
                    if a > 31 || self.ystridx == Some(0) || (yearfirst && b <= 12 && c <= 31) {
                        if dayfirst && c <= 12 {
                            (year, day, month) = (Some(a), Some(b), Some(c));
                        } else {
                            (year, month, day) = (Some(a), Some(b), Some(c));
                        }
                    } else if a > 12 || (dayfirst && b <= 12) {
                        (day, month, year) = (Some(a), Some(b), Some(c));
                    } else {
                        (month, day, year) = (Some(a), Some(b), Some(c));
                    }
                }
            }
        }
        Ok((year, month, day))
    }
}

fn adjust_ampm(hour: i64, ampm: u32) -> i64 {
    if hour < 12 && ampm == 1 {
        hour + 12
    } else if hour == 12 && ampm == 0 {
        0
    } else {
        hour
    }
}

/// `int(text)`, or the parse fails.
fn int_of(text: &str) -> R<i64> {
    py::int(text).ok_or(Fail)
}

/// `text[a:b]` by characters, as Python slices.
fn slice(text: &str, a: usize, b: Option<usize>) -> String {
    let chars: Vec<char> = text.chars().collect();
    let end = b.unwrap_or(chars.len()).min(chars.len());
    if a >= end {
        return String::new();
    }
    chars[a..end].iter().collect()
}

/// `_parsems`: seconds and microseconds of `I[.F]`.
fn parse_ms(value: &str) -> R<(i64, i64)> {
    match value.split_once('.') {
        None => Ok((int_of(value)?, 0)),
        Some((i, f)) => {
            if f.contains('.') {
                return Err(Fail);
            }
            let mut f: String = f.to_string();
            while f.chars().count() < 6 {
                f.push('0');
            }
            let f = slice(&f, 0, Some(6));
            Ok((int_of(i)?, int_of(&f)?))
        }
    }
}

/// `_parse_min_sec`.
fn parse_min_sec(value: &Dec) -> R<(i64, Option<i64>)> {
    let minute = i64::try_from(value.trunc()).map_err(|_| Fail)?;
    let second = if value.has_fraction() {
        Some(i64::try_from(value.scaled_fraction(60)).map_err(|_| Fail)?)
    } else {
        None
    };
    Ok((minute, second))
}

fn find_hms_idx(idx: usize, tokens: &[String]) -> Option<usize> {
    let len = tokens.len();
    if idx + 1 < len && hms(&tokens[idx + 1]).is_some() {
        Some(idx + 1)
    } else if idx + 2 < len && tokens[idx + 1] == " " && hms(&tokens[idx + 2]).is_some() {
        Some(idx + 2)
    } else if idx > 0 && hms(&tokens[idx - 1]).is_some() {
        Some(idx - 1)
    } else if 1 < idx && idx == len - 1 && tokens[idx - 1] == " " && hms(&tokens[idx - 2]).is_some()
    {
        Some(idx - 2)
    } else {
        None
    }
}

/// `dateutil.parser.parser()._parse(text, dayfirst, yearfirst)`, with the
/// local year `now` for two-digit years. `None` where it fails.
pub fn parse(text: &str, dayfirst: bool, yearfirst: bool, now: i64) -> Option<Res> {
    let mut tokens = timelex(text);
    let mut res = Res::default();
    let mut ymd = Ymd::default();
    if parse_tokens(&mut tokens, &mut res, &mut ymd, now).is_err() {
        return None;
    }
    let (year, month, day) = ymd.resolve(yearfirst, dayfirst).ok()?;
    res.century_specified = ymd.century_specified;
    res.year = year;
    res.month = month;
    res.day = day;
    // parserinfo.validate
    if let Some(year) = res.year {
        if year < 0 {
            return None;
        }
        res.year = Some(convert_year(year, res.century_specified, now));
    }
    let tzname = res.tzname.clone();
    if (res.tzoffset == Some(0) && tzname.as_deref().is_none_or(str::is_empty))
        || matches!(tzname.as_deref(), Some("Z" | "z"))
    {
        res.tzname = Some("UTC".to_string());
        res.tzoffset = Some(0);
    } else if res.tzoffset != Some(0)
        && tzname
            .as_deref()
            .is_some_and(|n| !n.is_empty() && utczone(n))
    {
        res.tzoffset = Some(0);
    }
    Some(res)
}

fn could_be_tzname(res: &Res, tzoffset: Option<i64>, token: &str) -> bool {
    res.hour.is_some()
        && res.tzname.is_none()
        && tzoffset.is_none()
        && token.chars().count() <= 5
        && (token.chars().all(|c| c.is_ascii_uppercase()) || UTCZONE.contains(&token))
}

fn parse_tokens(l: &mut Vec<String>, res: &mut Res, ymd: &mut Ymd, now: i64) -> R<()> {
    let len_l = l.len();
    let mut i = 0;
    let get = |l: &Vec<String>, i: usize| -> R<String> { l.get(i).cloned().ok_or(Fail) };
    while i < len_l {
        let value_repr = l[i].clone();
        if py::is_float(&value_repr) {
            i = parse_numeric_token(l, i, ymd, res)?;
        } else if let Some(day) = weekday(&l[i]) {
            res.weekday = Some(day);
        } else if let Some(m) = month(&l[i]) {
            ymd.append(Val::Int(i64::from(m)), Some('M'))?;
            if i + 1 < len_l {
                if l[i + 1] == "-" || l[i + 1] == "/" {
                    let sep = l[i + 1].clone();
                    let next = get(l, i + 2)?;
                    ymd.append(Val::Text(&next), None)?;
                    if i + 3 < len_l && l[i + 3] == sep {
                        let next = get(l, i + 4)?;
                        ymd.append(Val::Text(&next), None)?;
                        i += 2;
                    }
                    i += 2;
                } else if i + 4 < len_l && l[i + 1] == " " && l[i + 3] == " " && pertain(&l[i + 2])
                {
                    if py::all_digits(&l[i + 4]) {
                        // Converted here, as dateutil does: str(convertyear(value)).
                        let value = int_of(&l[i + 4])?;
                        let year = convert_year(value, false, now);
                        ymd.append(Val::Text(&year.to_string()), Some('Y'))?;
                    }
                    i += 4;
                }
            }
        } else if let Some(flag) = ampm(&l[i]) {
            // _ampm_valid without fuzzy: raises without a valid hour.
            let Some(hour) = res.hour else {
                return Err(Fail);
            };
            if !(0..=12).contains(&hour) {
                return Err(Fail);
            }
            res.hour = Some(adjust_ampm(hour, flag));
            res.ampm = Some(flag);
        } else if could_be_tzname(res, res.tzoffset, &l[i]) {
            res.tzname = Some(l[i].clone());
            res.tzoffset = tzoffset_of(&l[i]);
            if i + 1 < len_l && (l[i + 1] == "+" || l[i + 1] == "-") {
                l[i + 1] = if l[i + 1] == "+" { "-" } else { "+" }.to_string();
                res.tzoffset = None;
                if utczone(&l[i]) {
                    res.tzname = None;
                }
            }
        } else if res.hour.is_some() && (l[i] == "+" || l[i] == "-") {
            let signal = if l[i] == "+" { 1 } else { -1 };
            let next = get(l, i + 1)?;
            let len_li = next.chars().count();
            let (hour_offset, min_offset);
            if len_li == 4 {
                hour_offset = int_of(&slice(&next, 0, Some(2)))?;
                min_offset = int_of(&slice(&next, 2, None))?;
            } else if i + 2 < len_l && l[i + 2] == ":" {
                hour_offset = int_of(&next)?;
                min_offset = int_of(&get(l, i + 3)?)?;
                i += 2;
            } else if len_li <= 2 {
                hour_offset = int_of(&slice(&next, 0, Some(2)))?;
                min_offset = 0;
            } else {
                return Err(Fail);
            }
            res.tzoffset = Some(signal * (hour_offset * 3600 + min_offset * 60));
            if i + 5 < len_l
                && jump(&l[i + 2])
                && l[i + 3] == "("
                && l[i + 5] == ")"
                && l[i + 4].chars().count() >= 3
                && could_be_tzname(res, None, &l[i + 4])
            {
                res.tzname = Some(l[i + 4].clone());
                i += 4;
            }
            i += 1;
        } else if !jump(&l[i]) {
            return Err(Fail);
        }
        i += 1;
    }
    Ok(())
}

fn parse_numeric_token(tokens: &[String], idx: usize, ymd: &mut Ymd, res: &mut Res) -> R<usize> {
    let mut idx = idx;
    let value_repr = tokens[idx].clone();
    let value = Dec::parse(&value_repr).ok_or(Fail)?;
    let len_li = value_repr.chars().count();
    let len_l = tokens.len();
    let s = &tokens[idx];
    if ymd.len() == 3
        && (len_li == 2 || len_li == 4)
        && res.hour.is_none()
        && (idx + 1 >= len_l || (tokens[idx + 1] != ":" && hms(&tokens[idx + 1]).is_none()))
    {
        res.hour = Some(int_of(&slice(s, 0, Some(2)))?);
        if len_li == 4 {
            res.minute = Some(int_of(&slice(s, 2, None))?);
        }
    } else if len_li == 6 || (len_li > 6 && s.chars().position(|c| c == '.') == Some(6)) {
        if ymd.len() == 0 && !s.contains('.') {
            ymd.append(Val::Text(&slice(s, 0, Some(2))), None)?;
            ymd.append(Val::Text(&slice(s, 2, Some(4))), None)?;
            ymd.append(Val::Text(&slice(s, 4, None)), None)?;
        } else {
            res.hour = Some(int_of(&slice(s, 0, Some(2)))?);
            res.minute = Some(int_of(&slice(s, 2, Some(4)))?);
            let (second, micro) = parse_ms(&slice(s, 4, None))?;
            res.second = Some(second);
            res.microsecond = Some(micro);
        }
    } else if len_li == 8 || len_li == 12 || len_li == 14 {
        ymd.append(Val::Text(&slice(s, 0, Some(4))), Some('Y'))?;
        ymd.append(Val::Text(&slice(s, 4, Some(6))), None)?;
        ymd.append(Val::Text(&slice(s, 6, Some(8))), None)?;
        if len_li > 8 {
            res.hour = Some(int_of(&slice(s, 8, Some(10)))?);
            res.minute = Some(int_of(&slice(s, 10, Some(12)))?);
            if len_li > 12 {
                res.second = Some(int_of(&slice(s, 12, None))?);
            }
        }
    } else if let Some(hms_idx) = find_hms_idx(idx, tokens) {
        let (new_idx, which) = if hms_idx > idx {
            (hms_idx, hms(&tokens[hms_idx]).ok_or(Fail)?)
        } else {
            (idx, hms(&tokens[hms_idx]).ok_or(Fail)? + 1)
        };
        idx = new_idx;
        match which {
            0 => {
                res.hour = Some(i64::try_from(value.trunc()).map_err(|_| Fail)?);
                if value.has_fraction() {
                    res.minute = Some(i64::try_from(value.scaled_fraction(60)).map_err(|_| Fail)?);
                }
            }
            1 => {
                let (minute, second) = parse_min_sec(&value)?;
                res.minute = Some(minute);
                res.second = second;
            }
            2 => {
                let (second, micro) = parse_ms(&value_repr)?;
                res.second = Some(second);
                res.microsecond = Some(micro);
            }
            // hms of a following "s" label read backwards: no fourth unit.
            _ => {}
        }
    } else if idx + 2 < len_l && tokens[idx + 1] == ":" {
        res.hour = Some(i64::try_from(value.trunc()).map_err(|_| Fail)?);
        let minute_value = Dec::parse(&tokens[idx + 2]).ok_or(Fail)?;
        if !py::is_float(&tokens[idx + 2]) {
            return Err(Fail);
        }
        let (minute, second) = parse_min_sec(&minute_value)?;
        res.minute = Some(minute);
        res.second = second;
        if idx + 4 < len_l && tokens[idx + 3] == ":" {
            let (second, micro) = parse_ms(&tokens[idx + 4])?;
            res.second = Some(second);
            res.microsecond = Some(micro);
            idx += 2;
        }
        idx += 2;
    } else if idx + 1 < len_l && matches!(tokens[idx + 1].as_str(), "-" | "/" | ".") {
        let sep = tokens[idx + 1].clone();
        ymd.append(Val::Text(&value_repr), None)?;
        if idx + 2 < len_l && !jump(&tokens[idx + 2]) {
            if py::all_digits(&tokens[idx + 2]) {
                ymd.append(Val::Text(&tokens[idx + 2]), None)?;
            } else {
                let Some(m) = month(&tokens[idx + 2]) else {
                    return Err(Fail);
                };
                ymd.append(Val::Int(i64::from(m)), Some('M'))?;
            }
            if idx + 3 < len_l && tokens[idx + 3] == sep {
                let next = tokens.get(idx + 4).ok_or(Fail)?;
                if let Some(m) = month(next) {
                    ymd.append(Val::Int(i64::from(m)), Some('M'))?;
                } else {
                    ymd.append(Val::Text(next), None)?;
                }
                idx += 2;
            }
            idx += 1;
        }
        idx += 1;
    } else if idx + 1 >= len_l || jump(&tokens[idx + 1]) {
        if idx + 2 < len_l && ampm(&tokens[idx + 2]).is_some() {
            let hour = i64::try_from(value.trunc()).map_err(|_| Fail)?;
            res.hour = Some(adjust_ampm(hour, ampm(&tokens[idx + 2]).ok_or(Fail)?));
            idx += 1;
        } else {
            ymd.append(Val::Num(value), None)?;
        }
        idx += 1;
    } else if ampm(&tokens[idx + 1]).is_some()
        && !value.cmp_int(0).is_lt()
        && value.cmp_int(24).is_lt()
    {
        let hour = i64::try_from(value.trunc()).map_err(|_| Fail)?;
        res.hour = Some(adjust_ampm(hour, ampm(&tokens[idx + 1]).ok_or(Fail)?));
        idx += 1;
    } else if ymd.could_be_day(&value)? {
        ymd.append(Val::Num(value), None)?;
    } else {
        return Err(Fail);
    }
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lexes_as_dateutil() {
        assert_eq!(
            timelex("2021-01-01T10:00:00.5Z"),
            [
                "2021", "-", "01", "-", "01", "T", "10", ":", "00", ":", "00.5", "Z"
            ]
        );
        assert_eq!(timelex("Sep.20.2009"), ["Sep", ".", "20", ".", "2009"]);
        assert_eq!(timelex("Jan 5, 2021"), ["Jan", " ", "5", ",", " ", "2021"]);
    }

    #[test]
    fn parses_as_dateutil() {
        let res = parse("Jan 5, 2021", false, false, 2026).unwrap();
        assert_eq!(
            (res.year, res.month, res.day),
            (Some(2021), Some(1), Some(5))
        );
        let res = parse("01/02/03", true, false, 2026).unwrap();
        assert_eq!(
            (res.year, res.month, res.day),
            (Some(2003), Some(2), Some(1))
        );
        assert!(parse("abc", false, false, 2026).is_none());
    }
}
