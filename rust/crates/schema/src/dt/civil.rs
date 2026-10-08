// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Calendar arithmetic, as Python's `datetime` does it.

/// Whether `year` is a leap year of the proleptic Gregorian calendar.
pub fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// The days in `month` of `year`, as `calendar.monthrange(year, month)[1]`.
pub fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

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

/// Monday is 0, as `date.weekday()`.
pub fn weekday(year: i64, month: u32, day: u32) -> u32 {
    // 1970-01-01 was a Thursday.
    (days_from_civil(year, month, day) + 3).rem_euclid(7) as u32
}

/// Whether `datetime(year, month, day)` exists: Python's year range 1..9999.
pub fn valid_date(year: i64, month: u32, day: u32) -> bool {
    (1..=9999).contains(&year)
        && (1..=12).contains(&month)
        && day >= 1
        && day <= days_in_month(year, month)
}

/// A wall-clock date and time, to the nanosecond.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Wall {
    pub year: i64,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// Nanoseconds within the second.
    pub nano: u32,
}

impl Wall {
    pub fn date(year: i64, month: u32, day: u32) -> Self {
        Wall {
            year,
            month,
            day,
            hour: 0,
            minute: 0,
            second: 0,
            nano: 0,
        }
    }

    /// Nanoseconds since the epoch, reading this wall time as UTC.
    pub fn epoch_nanos(&self) -> i128 {
        let days = i128::from(days_from_civil(self.year, self.month, self.day));
        let seconds = days * 86_400
            + i128::from(self.hour) * 3600
            + i128::from(self.minute) * 60
            + i128::from(self.second);
        seconds * 1_000_000_000 + i128::from(self.nano)
    }

    pub fn from_epoch_nanos(nanos: i128) -> Self {
        let seconds = nanos.div_euclid(1_000_000_000);
        let nano = nanos.rem_euclid(1_000_000_000) as u32;
        let days = seconds.div_euclid(86_400);
        let rest = seconds.rem_euclid(86_400) as u32;
        let (year, month, day) = civil_from_days(days as i64);
        Wall {
            year,
            month,
            day,
            hour: rest / 3600,
            minute: rest % 3600 / 60,
            second: rest % 60,
            nano,
        }
    }

    pub fn microsecond(&self) -> u32 {
        self.nano / 1000
    }
}

pub const MONTHS_SHORT: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
pub const MONTHS_LONG: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
pub const DAYS_SHORT: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
pub const DAYS_LONG: [&str; 7] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_round_trip() {
        for days in [-800_000_i64, -1, 0, 1, 19_000, 2_932_896] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(weekday(2024, 1, 1), 0);
        assert_eq!(weekday(1970, 1, 1), 3);
    }
}
