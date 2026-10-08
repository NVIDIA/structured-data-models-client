// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Times as Python and Arrow write them: pandas' `Timestamp.isoformat`,
//! `datetime`'s `str`, and `pyarrow.compute.strftime` of a timestamp column.

use crate::Error;

const NANOS_PER_SECOND: i128 = 1_000_000_000;
const SECONDS_PER_DAY: i128 = 86_400;

/// The days from 1970-01-01 to a proleptic Gregorian date.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The proleptic Gregorian date `days` after 1970-01-01.
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// A wall-clock date and time, to the nanosecond.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Civil {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    pub nanos: u32,
}

impl Civil {
    /// The wall-clock time `nanos` after the epoch.
    pub fn from_epoch_nanos(nanos: i128) -> Self {
        let seconds = nanos.div_euclid(NANOS_PER_SECOND);
        let sub = nanos.rem_euclid(NANOS_PER_SECOND) as u32;
        let days = seconds.div_euclid(SECONDS_PER_DAY) as i64;
        let rest = seconds.rem_euclid(SECONDS_PER_DAY) as u32;
        let (year, month, day) = civil_from_days(days);
        Self {
            year,
            month,
            day,
            hour: rest / 3600,
            minute: rest / 60 % 60,
            second: rest % 60,
            nanos: sub,
        }
    }

    /// Nanoseconds since the epoch, reading the wall clock as UTC.
    pub fn epoch_nanos(&self) -> i128 {
        let days = i128::from(days_from_civil(self.year, self.month, self.day));
        let seconds =
            days * SECONDS_PER_DAY + i128::from(self.hour * 3600 + self.minute * 60 + self.second);
        seconds * NANOS_PER_SECOND + i128::from(self.nanos)
    }

    /// `%Y-%m-%d`, the year in four digits at least.
    pub fn date_text(&self) -> String {
        format!("{}-{:02}-{:02}", year_text(self.year), self.month, self.day)
    }

    /// `%H:%M:%S`.
    pub fn time_text(&self) -> String {
        format!("{:02}:{:02}:{:02}", self.hour, self.minute, self.second)
    }
}

fn year_text(year: i64) -> String {
    if year < 0 {
        format!("-{:04}", -year)
    } else {
        format!("{year:04}")
    }
}

/// `+HH:MM`, with `:SS` where the offset has seconds, as `datetime` writes a
/// UTC offset.
pub fn offset_text(seconds: i32, colon: bool) -> String {
    let sign = if seconds < 0 { '-' } else { '+' };
    let total = seconds.unsigned_abs();
    let (hours, minutes, rest) = (total / 3600, total / 60 % 60, total % 60);
    let sep = if colon { ":" } else { "" };
    let mut out = format!("{sign}{hours:02}{sep}{minutes:02}");
    if rest != 0 {
        out.push_str(&format!("{sep}{rest:02}"));
    }
    out
}

/// A pandas `Timestamp` or a `datetime`: a wall-clock time, and its UTC
/// offset if it has a zone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub civil: Civil,
    /// Seconds east of UTC.
    pub offset: Option<i32>,
}

impl Stamp {
    /// Reads `YYYY-MM-DD[T ]HH:MM:SS[.f{1,9}][Z|±HH:MM[:SS]]`, as
    /// `isoformat` writes a timestamp.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let bad = || Error::new(format!("not an ISO 8601 time: {text:?}"));
        let bytes = text.as_bytes();
        if bytes.len() < 19 || !matches!(bytes[10], b'T' | b' ') {
            return Err(bad());
        }
        let number = |range: std::ops::Range<usize>| -> Result<u32, Error> {
            let part = text.get(range).ok_or_else(bad)?;
            if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
                return Err(bad());
            }
            part.parse().map_err(|_| bad())
        };
        if &text[4..5] != "-" || &text[7..8] != "-" || &text[13..14] != ":" || &text[16..17] != ":"
        {
            return Err(bad());
        }
        let civil_date = (i64::from(number(0..4)?), number(5..7)?, number(8..10)?);
        let (hour, minute, second) = (number(11..13)?, number(14..16)?, number(17..19)?);
        let mut rest = &text[19..];
        let mut nanos = 0u32;
        if let Some(fraction) = rest.strip_prefix('.') {
            let length = fraction.bytes().take_while(u8::is_ascii_digit).count();
            if length == 0 || length > 9 {
                return Err(bad());
            }
            let digits = &fraction[..length];
            nanos = format!("{digits:0<9}").parse().map_err(|_| bad())?;
            rest = &fraction[length..];
        }
        let offset = match rest {
            "" => None,
            "Z" => Some(0),
            _ => Some(parse_offset(rest).ok_or_else(bad)?),
        };
        let (year, month, day) = civil_date;
        if !(1..=12).contains(&month) || day == 0 || day > 31 || hour > 23 || minute > 59 {
            return Err(bad());
        }
        Ok(Self {
            civil: Civil {
                year,
                month,
                day,
                hour,
                minute,
                second,
                nanos,
            },
            offset,
        })
    }

    /// The instant, in nanoseconds since the epoch; a naive time is read
    /// as UTC.
    pub fn epoch_nanos(&self) -> i128 {
        self.civil.epoch_nanos() - i128::from(self.offset.unwrap_or(0)) * NANOS_PER_SECOND
    }

    /// The same instant in UTC: `tz_convert('UTC')`, or `tz_localize('UTC')`
    /// of a naive time.
    pub fn to_utc(&self) -> Self {
        Self {
            civil: Civil::from_epoch_nanos(self.epoch_nanos()),
            offset: Some(0),
        }
    }

    /// `Timestamp.isoformat(sep)`: microseconds where there are any, three
    /// more digits where there are nanoseconds.
    pub fn isoformat(&self, sep: char) -> String {
        let c = &self.civil;
        let mut out = format!("{}{sep}{}", c.date_text(), c.time_text());
        let micros = c.nanos / 1000;
        let nanos = c.nanos % 1000;
        if micros != 0 {
            out.push_str(&format!(".{micros:06}"));
            if nanos != 0 {
                out.push_str(&format!("{nanos:03}"));
            }
        } else if nanos != 0 {
            out.push_str(&format!(".{nanos:09}"));
        }
        if let Some(offset) = self.offset {
            out.push_str(&offset_text(offset, true));
        }
        out
    }
}

/// `±HH:MM[:SS]` or `±HHMM` as seconds east of UTC.
pub fn parse_offset(text: &str) -> Option<i32> {
    let sign = match text.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let body = &text[1..];
    let parts: Vec<&str> = if body.contains(':') {
        body.split(':').collect()
    } else if body.len() == 4 {
        vec![&body[..2], &body[2..]]
    } else {
        return None;
    };
    if parts.len() < 2 || parts.len() > 3 {
        return None;
    }
    let mut seconds = 0i32;
    for (part, scale) in parts.iter().zip([3600, 60, 1]) {
        if part.len() != 2 || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        seconds += part.parse::<i32>().ok()? * scale;
    }
    Some(sign * seconds)
}

/// `str(datetime.time(...))`: microseconds only where there are any.
pub fn time_of_day_text(micros_of_day: i64) -> String {
    let seconds = micros_of_day / 1_000_000;
    let micros = micros_of_day % 1_000_000;
    let mut out = format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    );
    if micros != 0 {
        out.push_str(&format!(".{micros:06}"));
    }
    out
}

/// `str(datetime.timedelta(microseconds=micros))`.
pub fn timedelta_text(micros: i64) -> String {
    let total = i128::from(micros);
    let days = total.div_euclid(86_400_000_000);
    let rest = total.rem_euclid(86_400_000_000);
    let mut out = String::new();
    if days != 0 {
        let plural = if days.abs() != 1 { "s" } else { "" };
        out.push_str(&format!("{days} day{plural}, "));
    }
    let seconds = rest / 1_000_000;
    out.push_str(&format!(
        "{}:{:02}:{:02}",
        seconds / 3600,
        seconds / 60 % 60,
        seconds % 60
    ));
    let sub = rest % 1_000_000;
    if sub != 0 {
        out.push_str(&format!(".{sub:06}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip() {
        for days in [-719_162, -1, 0, 59, 2_932_896, 19_723] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(civil_from_days(-719_162), (1, 1, 1));
        assert_eq!(civil_from_days(2_932_896), (9999, 12, 31));
    }

    #[test]
    fn isoformat_as_pandas() {
        for (text, want) in [
            ("2024-01-01T00:00:00", "2024-01-01T00:00:00"),
            ("2024-01-01T00:00:00.500000", "2024-01-01T00:00:00.500000"),
            (
                "2024-01-01T00:00:00.000000005",
                "2024-01-01T00:00:00.000000005",
            ),
            (
                "2024-01-01T00:00:00.123456789+05:30",
                "2024-01-01T00:00:00.123456789+05:30",
            ),
            ("0005-01-01T00:00:00", "0005-01-01T00:00:00"),
        ] {
            assert_eq!(Stamp::parse(text).unwrap().isoformat('T'), want);
        }
        let zoned = Stamp::parse("2024-01-01T01:00:00.25-04:00").unwrap();
        assert_eq!(
            zoned.to_utc().isoformat('T'),
            "2024-01-01T05:00:00.250000+00:00"
        );
    }

    #[test]
    fn python_texts() {
        assert_eq!(
            timedelta_text(-86_400_000_000 + 5_000_000),
            "-1 day, 0:00:05"
        );
        assert_eq!(
            timedelta_text(2 * 86_400_000_000 + 7),
            "2 days, 0:00:00.000007"
        );
        assert_eq!(time_of_day_text(3_723_000_004), "01:02:03.000004");
        assert_eq!(offset_text(-18_000, false), "-0500");
    }
}
