// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! pandas' ISO 8601 reader, `parse_iso_8601_datetime` (vendored from
//! NumPy), byte for byte: with no format it infers one, and with a format it
//! requires the text to follow it.

use super::civil::days_in_month;

/// NumPy's datetime units, in their order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Unit {
    Y = 0,
    M = 1,
    D = 4,
    H = 5,
    Min = 6,
    S = 7,
    Ms = 8,
    Us = 9,
    Ns = 10,
    Ps = 11,
    Fs = 12,
    As = 13,
    Generic = 14,
}

impl Unit {
    /// pandas' `get_supported_reso`: seconds at the coarsest, nanoseconds
    /// at the finest.
    pub fn supported(self) -> Unit {
        match self {
            Unit::Generic => Unit::Ns,
            u if u < Unit::S => Unit::S,
            u if u > Unit::Ns => Unit::Ns,
            u => u,
        }
    }
}

/// What the reader fills in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dts {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub min: u32,
    pub sec: u32,
    pub us: u32,
    pub ps: u32,
    pub as_: u32,
}

/// A successful read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iso {
    pub dts: Dts,
    pub unit: Unit,
    /// The UTC offset in minutes, where the text gave one.
    pub offset: Option<i32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Requirement {
    Partial,
    Exact,
    Infer,
}

enum Cmp {
    Ok,
    Done,
    Error,
}

struct Format<'a> {
    text: &'a [u8],
    pos: usize,
    req: Requirement,
}

impl Format<'_> {
    fn remaining(&self) -> usize {
        self.text.len() - self.pos
    }

    fn compare(&mut self, what: &[u8]) -> Cmp {
        if self.req == Requirement::Infer {
            return Cmp::Ok;
        }
        if self.req == Requirement::Partial && self.remaining() == 0 {
            return Cmp::Done;
        }
        if self.remaining() < what.len() {
            return Cmp::Error;
        }
        if &self.text[self.pos..self.pos + what.len()] != what {
            return Cmp::Error;
        }
        self.pos += what.len();
        Cmp::Ok
    }
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// `string_to_dts(text, format, exact)`. With no format the reader infers
/// one; with one, `exact` requires all of it.
pub fn parse(text: &str, format: Option<&str>, exact: bool) -> Option<Iso> {
    let req = match format {
        None => Requirement::Infer,
        Some(_) if exact => Requirement::Exact,
        Some(_) => Requirement::Partial,
    };
    let mut fmt = Format {
        text: format.unwrap_or("").as_bytes(),
        pos: 0,
        req,
    };
    let s = text.as_bytes();
    let at = |i: usize| -> u8 { s.get(i).copied().unwrap_or(0) };
    let digit = |i: usize| -> bool { at(i).is_ascii_digit() };
    let mut out = Dts {
        month: 1,
        day: 1,
        ..Dts::default()
    };
    let mut unit = Unit::Generic;
    let mut offset: Option<i32> = None;
    let mut p = 0usize;
    let len = s.len();
    let mut sublen = len as i64;

    macro_rules! cmp {
        ($what:expr) => {
            match fmt.compare($what) {
                Cmp::Ok => {}
                Cmp::Done => {
                    return Some(Iso {
                        dts: out,
                        unit,
                        offset,
                    })
                }
                Cmp::Error => return None,
            }
        };
    }
    macro_rules! finish {
        () => {
            return Some(Iso {
                dts: out,
                unit,
                offset,
            })
        };
    }

    while sublen > 0 && is_space(at(p)) {
        p += 1;
        sublen -= 1;
        cmp!(b" ");
    }
    if at(p) == b'-' {
        p += 1;
        sublen -= 1;
    }
    if sublen == 0 {
        return None;
    }
    cmp!(b"%Y");
    out.year = 0;
    if sublen >= 4 && digit(p) && digit(p + 1) && digit(p + 2) && digit(p + 3) {
        out.year = i64::from(at(p) - b'0') * 1000
            + i64::from(at(p + 1) - b'0') * 100
            + i64::from(at(p + 2) - b'0') * 10
            + i64::from(at(p + 3) - b'0');
        p += 4;
        sublen -= 4;
    }
    if at(0) == b'-' {
        out.year = -out.year;
    }
    if sublen == 0 {
        if fmt.remaining() > 0 && req != Requirement::Infer {
            return None;
        }
        unit = Unit::Y;
        finish!();
    }
    let mut has_ymd_sep = false;
    let mut ymd_sep = 0u8;
    if !digit(p) {
        let c = at(p);
        if !matches!(c, b'-' | b'.' | b'/' | b'\\' | b' ') {
            return None;
        }
        has_ymd_sep = true;
        ymd_sep = c;
        p += 1;
        sublen -= 1;
        cmp!(&[ymd_sep]);
        if sublen == 0 || !digit(p) {
            return None;
        }
    }
    cmp!(b"%m");
    out.month = u32::from(at(p).wrapping_sub(b'0'));
    p += 1;
    sublen -= 1;
    if digit(p) {
        out.month = 10 * out.month + u32::from(at(p) - b'0');
        p += 1;
        sublen -= 1;
    } else if !has_ymd_sep {
        return None;
    }
    if !(1..=12).contains(&out.month) {
        return None;
    }
    if sublen == 0 {
        unit = Unit::M;
        if !has_ymd_sep {
            return None;
        }
        if fmt.remaining() > 0 && req != Requirement::Infer {
            return None;
        }
        finish!();
    }
    if has_ymd_sep {
        if at(p) != ymd_sep || sublen == 1 {
            return None;
        }
        p += 1;
        sublen -= 1;
        cmp!(&[ymd_sep]);
    }
    cmp!(b"%d");
    if !digit(p) {
        return None;
    }
    out.day = u32::from(at(p) - b'0');
    p += 1;
    sublen -= 1;
    if digit(p) {
        out.day = 10 * out.day + u32::from(at(p) - b'0');
        p += 1;
        sublen -= 1;
    } else if !has_ymd_sep {
        return None;
    }
    if out.day < 1 || out.day > days_in_month(out.year, out.month) {
        return None;
    }
    if sublen == 0 {
        if fmt.remaining() > 0 && req != Requirement::Infer {
            return None;
        }
        unit = Unit::D;
        finish!();
    }
    if (at(p) != b'T' && at(p) != b' ') || sublen == 1 {
        return None;
    }
    cmp!(&[at(p)]);
    p += 1;
    sublen -= 1;

    cmp!(b"%H");
    if !digit(p) {
        return None;
    }
    out.hour = u32::from(at(p) - b'0');
    unit = Unit::H;
    p += 1;
    sublen -= 1;
    let mut hour_was_2_digits = false;
    if digit(p) {
        hour_was_2_digits = true;
        out.hour = 10 * out.hour + u32::from(at(p) - b'0');
        p += 1;
        sublen -= 1;
        if out.hour >= 24 {
            return None;
        }
    }
    let mut has_hms_sep = false;
    let mut goto_timezone = false;
    if sublen == 0 {
        if !hour_was_2_digits {
            return None;
        }
        if fmt.remaining() > 0 && req != Requirement::Infer {
            return None;
        }
        unit = Unit::H;
        finish!();
    }
    if at(p) == b':' {
        has_hms_sep = true;
        p += 1;
        sublen -= 1;
        if sublen == 0 || !digit(p) {
            return None;
        }
        cmp!(b":");
    } else if !digit(p) {
        if !hour_was_2_digits {
            return None;
        }
        goto_timezone = true;
    }

    if !goto_timezone {
        cmp!(b"%M");
        out.min = u32::from(at(p).wrapping_sub(b'0'));
        unit = Unit::Min;
        p += 1;
        sublen -= 1;
        if digit(p) {
            out.min = 10 * out.min + u32::from(at(p) - b'0');
            p += 1;
            sublen -= 1;
            if out.min >= 60 {
                return None;
            }
        } else if !has_hms_sep {
            return None;
        }
        if sublen == 0 {
            unit = Unit::Min;
            if fmt.remaining() > 0 && req != Requirement::Infer {
                return None;
            }
            finish!();
        }
        let mut seconds = true;
        if has_hms_sep && at(p) == b':' {
            cmp!(b":");
            p += 1;
            sublen -= 1;
            if sublen == 0 || !digit(p) {
                return None;
            }
        } else if !has_hms_sep && digit(p) {
        } else {
            seconds = false;
        }
        if seconds {
            cmp!(b"%S");
            out.sec = u32::from(at(p).wrapping_sub(b'0'));
            p += 1;
            sublen -= 1;
            if digit(p) {
                out.sec = 10 * out.sec + u32::from(at(p) - b'0');
                p += 1;
                sublen -= 1;
                if out.sec >= 60 {
                    return None;
                }
            } else if !has_hms_sep {
                return None;
            }
            if sublen > 0 && at(p) == b'.' {
                p += 1;
                sublen -= 1;
                cmp!(b".");
                cmp!(b"%f");
                let mut digits = 0;
                for _ in 0..6 {
                    out.us *= 10;
                    if sublen > 0 && digit(p) {
                        out.us += u32::from(at(p) - b'0');
                        p += 1;
                        sublen -= 1;
                        digits += 1;
                    }
                }
                if sublen == 0 || !digit(p) {
                    unit = if digits > 3 { Unit::Us } else { Unit::Ms };
                } else {
                    let mut digits = 0;
                    for _ in 0..6 {
                        out.ps *= 10;
                        if sublen > 0 && digit(p) {
                            out.ps += u32::from(at(p) - b'0');
                            p += 1;
                            sublen -= 1;
                            digits += 1;
                        }
                    }
                    if sublen == 0 || !digit(p) {
                        unit = if digits > 3 { Unit::Ps } else { Unit::Ns };
                    } else {
                        let mut digits = 0;
                        for _ in 0..6 {
                            out.as_ *= 10;
                            if sublen > 0 && digit(p) {
                                out.as_ += u32::from(at(p) - b'0');
                                p += 1;
                                sublen -= 1;
                                digits += 1;
                            }
                        }
                        unit = if digits > 3 { Unit::As } else { Unit::Fs };
                    }
                }
            } else {
                unit = Unit::S;
            }
        }
    }

    // parse_timezone:
    while sublen > 0 && is_space(at(p)) {
        p += 1;
        sublen -= 1;
        cmp!(b" ");
    }
    if sublen == 0 {
        if fmt.remaining() > 0 && req != Requirement::Infer {
            return None;
        }
        finish!();
    }
    if at(p) == b'Z' {
        cmp!(b"%z");
        offset = Some(0);
        if sublen == 1 {
            if fmt.remaining() > 0 && req != Requirement::Infer {
                return None;
            }
            finish!();
        }
        p += 1;
        sublen -= 1;
    } else if at(p) == b'-' || at(p) == b'+' {
        cmp!(b"%z");
        let negative = at(p) == b'-';
        p += 1;
        sublen -= 1;
        let mut hours: i32;
        let mut minutes: i32 = 0;
        if sublen >= 2 && digit(p) && digit(p + 1) {
            hours = i32::from(at(p) - b'0') * 10 + i32::from(at(p + 1) - b'0');
            p += 2;
            sublen -= 2;
            if hours >= 24 {
                return None;
            }
        } else if sublen >= 1 && digit(p) {
            hours = i32::from(at(p) - b'0');
            p += 1;
            sublen -= 1;
        } else {
            return None;
        }
        if sublen > 0 {
            if at(p) == b':' {
                p += 1;
                sublen -= 1;
            }
            if sublen >= 2 && digit(p) && digit(p + 1) {
                minutes = i32::from(at(p) - b'0') * 10 + i32::from(at(p + 1) - b'0');
                p += 2;
                sublen -= 2;
                if minutes >= 60 {
                    return None;
                }
            } else if sublen >= 1 && digit(p) {
                minutes = i32::from(at(p) - b'0');
                p += 1;
                sublen -= 1;
            } else {
                return None;
            }
        }
        if negative {
            hours = -hours;
            minutes = -minutes;
        }
        offset = Some(60 * hours + minutes);
    }
    while sublen > 0 && is_space(at(p)) {
        p += 1;
        sublen -= 1;
        cmp!(b" ");
    }
    if sublen != 0 || (fmt.remaining() != 0 && req != Requirement::Infer) {
        return None;
    }
    finish!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_iso() {
        let got = parse("2021-01-02 03:04:05.123456+05:30", None, true).unwrap();
        assert_eq!(got.dts.us, 123_456);
        assert_eq!(got.offset, Some(330));
        assert_eq!(got.unit, Unit::Us);
        assert!(parse("2021-13-01", None, true).is_none());
        assert!(parse("2021-01-02", Some("%Y-%m-%d"), true).is_some());
        assert!(parse("2021-01-02", Some("%Y/%m/%d"), true).is_none());
        assert!(parse("20210102", None, true).is_some());
        assert!(parse("202101", None, true).is_none());
    }
}
