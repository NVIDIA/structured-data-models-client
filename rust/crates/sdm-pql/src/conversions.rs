// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The conversions a view's parameter default goes through in the old
//! client (T3.5, `_default_text`), as Python 3.12 makes them: `int(text)`,
//! `format(Decimal(text), 'f')`, `date.fromisoformat(text)` and
//! `datetime.fromisoformat(text)`, with the text `isoformat` writes back.
//! Which texts they accept decides whether a default is a value of its
//! type, so they are emulation, on the allowlist
//! (`scripts/rust_core/emulation-allowlist.txt`). The ISO readers are a
//! port of CPython 3.12's C reader (`Modules/_datetimemodule.c`), not of
//! the pure-Python one, whose answers differ.
//!
//! Each answer is bounded: a text longer than [`LONGEST_TEXT`] is refused
//! by name ([`Unconverted::TooLong`]) before it is built, where Python
//! built it or ran out of memory.

use crate::float_repr::text_repr;
pub use crate::pyint::MAX_STR_DIGITS;
use crate::syntax::is_space;

/// The longest text a conversion writes, in characters; a longer one is
/// refused by name (register P3J-10).
pub const LONGEST_TEXT: usize = 1 << 22;

/// Why a text gives no value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unconverted {
    /// Python refuses it (`ValueError`, `InvalidOperation`).
    Invalid,
    /// `int` refuses more digits than Python reads, so many.
    TooManyDigits(usize),
    /// `int` refuses a text the old client's pattern took: one with
    /// `\x1c` to `\x1f` around its digits, which `str.strip` drops and
    /// `int` does not.
    NotAnInteger,
    /// The text written would be longer than [`LONGEST_TEXT`].
    TooLong,
}

/// Python's message for an integer of more digits than it reads.
pub fn too_many_digits_text(digits: usize) -> String {
    format!(
        "Exceeds the limit ({MAX_STR_DIGITS} digits) for integer string conversion: value has \
         {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
    )
}

/// Python's message for a text `int` cannot read.
pub fn not_an_integer_text(text: &str) -> String {
    format!(
        "invalid literal for int() with base 10: {}",
        text_repr(text)
    )
}

/// `text.strip()`: Python's white space (`str.isspace`) off both ends.
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

/// The first code point of each run of ten decimal digits, `0` to `9`, in
/// Unicode 15.0, Python 3.12's (`unicodedata.decimal`).
const ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

/// The value of a decimal digit of any script, as Python reads one (`\d`,
/// `int`, `Decimal`): `'٣'` is 3.
pub fn decimal_digit(c: char) -> Option<u8> {
    let code = u32::from(c);
    let at = ZEROS.partition_point(|&zero| zero <= code).checked_sub(1)?;
    let offset = code - ZEROS[at];
    // Ten digits each: the offset is below 10, so the cast is exact.
    (offset < 10).then_some(offset as u8)
}

/// `str(int(text))` where the old client hands `int` a text: one whose
/// `str.strip()` is an optional sign and decimal digits of any script
/// (`[+-]?\d+`, `DB/metric_view.py:529-532`); anything else is invalid.
/// `int` itself strips less ([`Unconverted::NotAnInteger`]), and refuses
/// more digits than Python reads, leading zeros counted.
pub fn int_text(text: &str) -> Result<String, Unconverted> {
    let digits = shaped_digits(strip(text)).ok_or(Unconverted::Invalid)?;
    // `int` strips ASCII's six white space characters and Unicode's, but
    // not U+001C to U+001F.
    let trimmed = text.trim_matches(|c: char| is_space(c) && !('\u{1c}'..='\u{1f}').contains(&c));
    if shaped_digits(trimmed).is_none() {
        return Err(Unconverted::NotAnInteger);
    }
    let (negative, digits) = digits;
    if digits.len() > MAX_STR_DIGITS {
        return Err(Unconverted::TooManyDigits(digits.len()));
    }
    let kept = digits.trim_start_matches('0');
    Ok(match (negative, kept) {
        (_, "") => "0".into(),
        (true, kept) => format!("-{kept}"),
        (false, kept) => kept.to_owned(),
    })
}

/// A sign and decimal digits of any script, as the sign and ASCII digits.
fn shaped_digits(text: &str) -> Option<(bool, String)> {
    let (negative, body) = match text.chars().next() {
        Some('-') => (true, &text[1..]),
        Some('+') => (false, &text[1..]),
        _ => (false, text),
    };
    let mut digits = String::with_capacity(body.len());
    for c in body.chars() {
        digits.push(char::from(b'0' + decimal_digit(c)?));
    }
    (!digits.is_empty()).then_some((negative, digits))
}

/// libmpdec's bounds on a decimal read exactly (`MPD_MAX_EMAX`,
/// `MPD_MIN_ETINY`, `MPD_MAX_PREC`, `MPD_EXP_INF`, 64-bit).
const MAX_EMAX: i128 = 999_999_999_999_999_999;
const MIN_ETINY: i128 = -1_999_999_999_999_999_997;
const MAX_PREC: usize = 999_999_999_999_999_999;
const EXP_INF: i128 = 2_000_000_000_000_000_001;

/// `format(Decimal(text), 'f')` for a finite decimal, its digits placed by
/// its exponent with no exponent written; infinities, NaNs and anything
/// `Decimal` refuses are invalid. As `Decimal` reads it: each digit of any
/// script read as its value, `_` dropped wherever it is, white space around
/// it allowed, and the exponent bounded as libmpdec bounds it.
pub fn decimal_text(text: &str) -> Result<String, Unconverted> {
    let mut ascii = String::with_capacity(text.len());
    for c in strip(text).chars() {
        match c {
            '_' => {}
            c if is_space(c) => return Err(Unconverted::Invalid),
            c => match decimal_digit(c) {
                Some(d) => ascii.push(char::from(b'0' + d)),
                None if c.is_ascii() => ascii.push(c),
                None => return Err(Unconverted::Invalid),
            },
        }
    }
    let bytes = ascii.as_bytes();
    let (negative, body) = match bytes.first() {
        Some(b'-') => (true, &bytes[1..]),
        Some(b'+') => (false, &bytes[1..]),
        _ => (false, bytes),
    };
    // `scan_dpoint_exp`: digits with at most one point, then `e` and an
    // exponent of digits with an optional sign; the coefficient starts at
    // its first digit that is no leading zero.
    let (mut point, mut exponent, mut coefficient) = (None, None, None);
    let mut at = 0;
    while at < body.len() {
        match body[at] {
            b'.' => {
                if point.is_some() || exponent.is_some() {
                    return Err(Unconverted::Invalid);
                }
                point = Some(at);
            }
            b'e' | b'E' => {
                if exponent.is_some() || coefficient.is_none() {
                    return Err(Unconverted::Invalid);
                }
                exponent = Some(at);
                at += 1;
                if matches!(body.get(at), Some(b'+' | b'-')) {
                    at += 1;
                }
                if !body.get(at).is_some_and(u8::is_ascii_digit) {
                    return Err(Unconverted::Invalid);
                }
            }
            digit if digit.is_ascii_digit() => {
                if coefficient.is_none() && exponent.is_none() {
                    let next = body.get(at + 1).copied();
                    let after = body.get(at + 2).copied();
                    let leading_zero = digit == b'0'
                        && (next.is_some_and(|c| c.is_ascii_digit())
                            || (next == Some(b'.') && after.is_some_and(|c| c.is_ascii_digit())));
                    if !leading_zero {
                        coefficient = Some(at);
                    }
                }
            }
            _ => return Err(Unconverted::Invalid),
        }
        at += 1;
    }
    let Some(start) = coefficient else {
        return Err(Unconverted::Invalid);
    };
    let end = exponent.unwrap_or(body.len());
    // The exponent as `strtoll` reads it, held at its bound past it.
    let mut exp: i128 = match exponent {
        None => 0,
        Some(at) => {
            let written = std::str::from_utf8(&body[at + 1..]).map_err(|_| Unconverted::Invalid)?;
            let (sign, digits) = match written.as_bytes().first() {
                Some(b'-') => (-1, &written[1..]),
                Some(b'+') => (1, &written[1..]),
                _ => (1, written),
            };
            let mut value: i128 = 0;
            for d in digits.bytes() {
                value = (value * 10 + i128::from(d - b'0')).min(i128::from(i64::MAX) + 1);
            }
            (sign * value).clamp(i128::from(i64::MIN), i128::from(i64::MAX))
        }
    };
    let mut digits = end - start;
    if let Some(point) = point {
        let fraction = end - point - 1;
        if point > start {
            digits -= 1;
        }
        if fraction > MAX_PREC {
            return Err(Unconverted::Invalid);
        }
        // `fraction` is below `MAX_PREC`, which fits.
        exp = (exp - fraction as i128).max(i128::from(i64::MIN));
    }
    if digits > MAX_PREC {
        return Err(Unconverted::Invalid);
    }
    exp = exp.min(EXP_INF);
    if exp == i128::from(i64::MIN) {
        exp += 1;
    }
    // Read under the widest context, exactly: an exponent past either
    // bound would round, clamp or overflow, which `Decimal` refuses.
    let adjusted = exp + digits as i128 - 1;
    if adjusted > MAX_EMAX || exp < MIN_ETINY {
        return Err(Unconverted::Invalid);
    }
    let coefficient: String = body[start..end]
        .iter()
        .filter(|&&c| c != b'.')
        .map(|&c| char::from(c))
        .collect();
    let sign = if negative { "-" } else { "" };
    let places = digits as i128 + exp;
    let length = sign.len() as i128
        + if exp >= 0 {
            digits as i128 + exp
        } else if places <= 0 {
            2 - places + digits as i128
        } else {
            digits as i128 + 1
        };
    if length > LONGEST_TEXT as i128 {
        return Err(Unconverted::TooLong);
    }
    // Every length below is bounded by `LONGEST_TEXT`.
    Ok(if exp >= 0 {
        format!("{sign}{coefficient}{}", "0".repeat(exp as usize))
    } else if places <= 0 {
        format!("{sign}0.{}{coefficient}", "0".repeat((-places) as usize))
    } else {
        let (whole, fraction) = coefficient.split_at(places as usize);
        format!("{sign}{whole}.{fraction}")
    })
}

/// A date and time as `datetime` holds one: the offset from UTC, if any,
/// in microseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Moment {
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    microsecond: i32,
    offset: Option<i64>,
}

/// A text as CPython's C reader walks it: its UTF-8 bytes, with the zero
/// byte that ends a C string past them.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn at(&self, i: usize) -> u8 {
        self.0.get(i).copied().unwrap_or(0)
    }

    /// `parse_digits`: `count` ASCII digits from `at` into `value`, and
    /// where they end.
    fn digits(&self, at: usize, value: &mut i32, count: usize) -> Option<usize> {
        for i in 0..count {
            let digit = self.at(at + i).wrapping_sub(b'0');
            if digit > 9 {
                return None;
            }
            *value = *value * 10 + i32::from(digit);
        }
        Some(at + count)
    }
}

fn leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in(year: i32, month: i32) -> i32 {
    match month {
        2 if leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 0001-01-01 counted from 1 (`ymd_to_ord`).
fn ordinal(year: i32, month: i32, day: i32) -> i64 {
    let y = i64::from(year) - 1;
    let before: i64 = (1..month).map(|m| i64::from(days_in(year, m))).sum();
    y * 365 + y / 4 - y / 100 + y / 400 + before + i64::from(day)
}

/// `ord_to_ymd`.
fn from_ordinal(ordinal: i64) -> (i32, i32, i32) {
    let mut n = ordinal - 1;
    let n400 = n / 146_097;
    n %= 146_097;
    let n100 = n / 36_524;
    n %= 36_524;
    let n4 = n / 1461;
    n %= 1461;
    let n1 = n / 365;
    n %= 365;
    let mut year = n400 * 400 + n100 * 100 + n4 * 4 + n1 + 1;
    if n1 == 4 || n100 == 4 {
        year -= 1;
        // The last day of a leap year.
        return (year as i32, 12, 31);
    }
    let year = year as i32;
    let mut month = 1;
    let mut left = n;
    while left >= i64::from(days_in(year, month)) {
        left -= i64::from(days_in(year, month));
        month += 1;
    }
    (year, month, left as i32 + 1)
}

/// `iso_to_ymd`: an ISO year, week and weekday as a date, unchecked.
fn iso_to_ymd(year: i32, week: i32, day: i32) -> Option<(i32, i32, i32)> {
    if !(1..=9999).contains(&year) {
        return None;
    }
    // Monday is 0.
    let weekday = |y: i32| (ordinal(y, 1, 1) + 6) % 7;
    if week <= 0 || week >= 53 {
        let first = weekday(year);
        if !(week == 53 && (first == 3 || (first == 2 && leap(year)))) {
            return None;
        }
    }
    if !(1..=7).contains(&day) {
        return None;
    }
    let first_day = ordinal(year, 1, 1);
    let first_weekday = (first_day + 6) % 7;
    let mut monday = first_day - first_weekday;
    if first_weekday > 3 {
        monday += 7;
    }
    Some(from_ordinal(monday + i64::from((week - 1) * 7 + day - 1)))
}

/// `parse_isoformat_date` over the first `len` bytes.
fn read_date(text: &Bytes, len: usize) -> Option<(i32, i32, i32)> {
    let mut year = 0;
    let mut at = text.digits(0, &mut year, 4)?;
    let separated = text.at(at) == b'-';
    if separated {
        at += 1;
    }
    if text.at(at) == b'W' {
        at += 1;
        let (mut week, mut day) = (0, 0);
        at = text.digits(at, &mut week, 2)?;
        if at < len {
            if separated {
                if text.at(at) != b'-' {
                    return None;
                }
                at += 1;
            }
            text.digits(at, &mut day, 1)?;
        } else {
            day = 1;
        }
        return iso_to_ymd(year, week, day);
    }
    let (mut month, mut day) = (0, 0);
    at = text.digits(at, &mut month, 2)?;
    if separated {
        if text.at(at) != b'-' {
            return None;
        }
        at += 1;
    }
    text.digits(at, &mut day, 2)?;
    Some((year, month, day))
}

/// `parse_hh_mm_ss_ff` from `start` to `end`: the hour, minute, second and
/// microsecond, and whether the text goes on (`1`), or why it is refused.
fn read_clock(text: &Bytes, start: usize, end: usize) -> Result<([i32; 4], bool), ()> {
    let mut values = [0; 4];
    let mut at = start;
    let mut separated = true;
    for i in 0..3 {
        at = text.digits(at, &mut values[i], 2).ok_or(())?;
        let c = text.at(at);
        at += 1;
        if i == 0 {
            separated = c == b':';
        }
        if at >= end {
            return Ok((values, c != 0));
        } else if separated && c == b':' {
            continue;
        } else if c == b'.' || c == b',' {
            break;
        } else if !separated {
            at -= 1;
        } else {
            return Err(());
        }
    }
    let remains = end.saturating_sub(at);
    let count = remains.min(6);
    at = text.digits(at, &mut values[3], count).ok_or(())?;
    if (1..6).contains(&count) {
        values[3] *= [100_000, 10_000, 1000, 100, 10][count - 1];
    }
    while text.at(at).is_ascii_digit() {
        at += 1;
    }
    Ok((values, text.at(at) != 0))
}

/// A time of day read: the hour, minute, second and microsecond, and the
/// offset's seconds and microseconds if one is written.
type Clock = ([i32; 4], Option<(i64, i64)>);

/// `parse_isoformat_time` from `start` over `len` bytes.
fn read_time(text: &Bytes, start: usize, len: usize) -> Option<Clock> {
    let end = start + len;
    let mut zone = start;
    loop {
        if matches!(text.at(zone), b'Z' | b'+' | b'-') {
            break;
        }
        zone += 1;
        if zone >= end {
            break;
        }
    }
    let (clock, goes_on) = read_clock(text, start, zone).ok()?;
    if zone == end {
        return (!goes_on).then_some((clock, None));
    }
    if text.at(zone) == b'Z' {
        return (text.at(zone + 1) == 0).then_some((clock, Some((0, 0))));
    }
    let sign: i64 = if text.at(zone) == b'-' { -1 } else { 1 };
    let (offset, goes_on) = read_clock(text, zone + 1, end).ok()?;
    if goes_on {
        return None;
    }
    let seconds = i64::from(offset[0]) * 3600 + i64::from(offset[1]) * 60 + i64::from(offset[2]);
    // An offset of no whole seconds is UTC, its microseconds dropped
    // (`tzinfo_from_isoformat_results`).
    let micros = if seconds == 0 {
        0
    } else {
        sign * i64::from(offset[3])
    };
    Some((clock, Some((sign * seconds, micros))))
}

/// `_find_isoformat_datetime_separator`.
fn separator(text: &Bytes, len: usize) -> usize {
    if len == 7 {
        return 7;
    }
    if text.at(4) == b'-' {
        if text.at(5) != b'W' {
            return 10;
        }
        if len < 8 {
            return usize::MAX;
        }
        if len > 8 && text.at(8) == b'-' {
            if len == 9 {
                return usize::MAX;
            }
            if len > 10 && text.at(10).is_ascii_digit() {
                return 8;
            }
            return 10;
        }
        return 8;
    }
    if text.at(4) == b'W' {
        let mut at = 7;
        while at < len && text.at(at).is_ascii_digit() {
            at += 1;
        }
        if at < 9 {
            return at;
        }
        return if at % 2 == 0 { 7 } else { 8 };
    }
    8
}

/// The date checks `datetime.date` makes.
fn valid_date(year: i32, month: i32, day: i32) -> bool {
    (1..=9999).contains(&year)
        && (1..=12).contains(&month)
        && day >= 1
        && day <= days_in(year, month)
}

fn date_iso(year: i32, month: i32, day: i32) -> String {
    format!("{year:04}-{month:02}-{day:02}")
}

/// `date.fromisoformat(text).isoformat()`.
pub fn date_text(text: &str) -> Result<String, Unconverted> {
    let bytes = Bytes(text.as_bytes());
    let len = text.len();
    if !matches!(len, 7 | 8 | 10) {
        return Err(Unconverted::Invalid);
    }
    let (year, month, day) = read_date(&bytes, len).ok_or(Unconverted::Invalid)?;
    if !valid_date(year, month, day) {
        return Err(Unconverted::Invalid);
    }
    Ok(date_iso(year, month, day))
}

/// `datetime.fromisoformat(text).isoformat(' ')`.
pub fn datetime_text(text: &str) -> Result<String, Unconverted> {
    if text.chars().count() < 7 {
        return Err(Unconverted::Invalid);
    }
    let bytes = Bytes(text.as_bytes());
    let len = text.len();
    let split = separator(&bytes, len);
    let (year, month, day) = read_date(&bytes, split).ok_or(Unconverted::Invalid)?;
    let mut moment = Moment {
        year,
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
        microsecond: 0,
        offset: None,
    };
    if len > split {
        // The separator is any one character, however long in UTF-8.
        let lead = bytes.at(split);
        let width = match lead {
            _ if lead & 0x80 == 0 => 1,
            _ if lead & 0xf0 == 0xe0 => 3,
            _ if lead & 0xf0 == 0xf0 => 4,
            _ => 2,
        };
        let start = split + width;
        let rest = len.checked_sub(start).ok_or(Unconverted::Invalid)?;
        let (clock, zone) = read_time(&bytes, start, rest).ok_or(Unconverted::Invalid)?;
        [
            moment.hour,
            moment.minute,
            moment.second,
            moment.microsecond,
        ] = clock;
        moment.offset = zone.map(|(seconds, micros)| seconds * 1_000_000 + micros);
    }
    let Moment {
        hour,
        minute,
        second,
        offset,
        ..
    } = moment;
    if !valid_date(year, month, day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0..=59).contains(&second)
        || offset.is_some_and(|micros| micros.abs() >= 86_400_000_000)
    {
        return Err(Unconverted::Invalid);
    }
    Ok(moment_iso(&moment))
}

/// `isoformat(' ')`: the date, a space, the clock, `.ffffff` if it has
/// microseconds, and the offset as `+HH:MM`, with `:SS` and `.ffffff` if
/// it has them.
fn moment_iso(moment: &Moment) -> String {
    let mut text = format!(
        "{} {:02}:{:02}:{:02}",
        date_iso(moment.year, moment.month, moment.day),
        moment.hour,
        moment.minute,
        moment.second
    );
    if moment.microsecond > 0 {
        text.push_str(&format!(".{:06}", moment.microsecond));
    }
    if let Some(offset) = moment.offset {
        let sign = if offset < 0 { '-' } else { '+' };
        let micros = offset.unsigned_abs();
        let (seconds, fraction) = (micros / 1_000_000, micros % 1_000_000);
        text.push_str(&format!(
            "{sign}{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60
        ));
        if fraction > 0 {
            text.push_str(&format!(":{:02}.{fraction:06}", seconds % 60));
        } else if seconds % 60 > 0 {
            text.push_str(&format!(":{:02}", seconds % 60));
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_as_python_3_12_reads() {
        assert_eq!(int_text(" \u{661}\u{662} ").as_deref(), Ok("12"));
        assert_eq!(int_text("-007").as_deref(), Ok("-7"));
        assert_eq!(
            int_text(&"0".repeat(4301)),
            Err(Unconverted::TooManyDigits(4301))
        );
        assert_eq!(
            decimal_text("\u{661}\u{662}.\u{663}").as_deref(),
            Ok("12.3")
        );
        assert_eq!(decimal_text("1e_5").as_deref(), Ok("100000"));
        assert_eq!(decimal_text("-0.0").as_deref(), Ok("-0.0"));
        assert_eq!(
            decimal_text("1e999999999999999999"),
            Err(Unconverted::TooLong)
        );
        assert_eq!(
            decimal_text("1e1000000000000000000"),
            Err(Unconverted::Invalid)
        );
        assert_eq!(
            decimal_text("1.0e-9223372036854775808"),
            Err(Unconverted::Invalid)
        );
        assert_eq!(decimal_text("inf"), Err(Unconverted::Invalid));
        assert_eq!(date_text("2024010199").as_deref(), Ok("2024-01-01"));
        assert_eq!(date_text("9999-W52-7"), Err(Unconverted::Invalid));
        assert_eq!(
            datetime_text("2024-01-01T12.5").as_deref(),
            Ok("2024-01-01 12:00:00.500000")
        );
        assert_eq!(
            datetime_text("2024-01-01T12:34:56+01:60").as_deref(),
            Ok("2024-01-01 12:34:56+02:00")
        );
        assert_eq!(
            datetime_text("2024-01-01 10:00:00 Z").as_deref(),
            Ok("2024-01-01 10:00:00+00:00")
        );
        assert_eq!(
            datetime_text("2024-01-01T10:00:00+00:00:00.5").as_deref(),
            Ok("2024-01-01 10:00:00+00:00")
        );
        assert_eq!(
            datetime_text("2024-01-01T10:00:00\0").as_deref(),
            Ok("2024-01-01 10:00:00")
        );
    }
}
