// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A query window's offsets as the reference's `pd.DateOffset` adds them to
//! a time with no zone (`DateOffsetRange._to_date_offset`, register PK-3):
//! a count of seconds, minutes, hours, days or weeks is a fixed span; a
//! count of months is calendar arithmetic on `jiff`, the year and month
//! moved and the day clamped to the last the month has, the time of day
//! kept to the nanosecond, as `relativedelta` moves a date. Subtracting an
//! offset is adding its negation, as pandas subtracts one.
//!
//! The sampler's step back in time is not an offset but the fixed span
//! between the window's ends counted from the epoch
//! (`DateOffsetRange.timeframe`): [`timeframe`].

use jiff::civil::DateTime;
use sdm_contract::CoreError;
use sdm_contract::pql::TimeUnit;

use super::{Civil, Instant};

const NANOS_PER_SECOND: i128 = 1_000_000_000;

/// `count` of `unit`, as `pd.DateOffset(**{unit: count})`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Offset {
    pub count: i64,
    pub unit: TimeUnit,
}

impl Offset {
    pub fn of(count: i32, unit: TimeUnit) -> Offset {
        Offset {
            count: i64::from(count),
            unit,
        }
    }

    /// The offset `-self`.
    pub fn negated(self) -> Offset {
        Offset {
            count: -self.count,
            unit: self.unit,
        }
    }

    /// The fixed span this offset is, or `None` for months.
    pub fn fixed_nanos(self) -> Option<i128> {
        let seconds: i128 = match self.unit {
            TimeUnit::Seconds => 1,
            TimeUnit::Minutes => 60,
            TimeUnit::Hours => 3_600,
            TimeUnit::Days => 86_400,
            TimeUnit::Weeks => 7 * 86_400,
            TimeUnit::Months => return None,
        };
        Some(i128::from(self.count) * seconds * NANOS_PER_SECOND)
    }

    /// `at + self`, `at` and the answer nanoseconds since the epoch, as
    /// pandas adds the offset to a `Timestamp`; a time past what an int64
    /// of nanoseconds holds, which pandas cannot hold either, is refused
    /// with `VALUE_OUT_OF_RANGE` naming `field`.
    pub fn add_to(self, at: i64, field: &str) -> Result<i64, CoreError> {
        let out = || crate::sample::out_of_range(field);
        let nanos = match self.fixed_nanos() {
            Some(span) => i128::from(at) + span,
            None => {
                let civil = Instant::from_nanos(i128::from(at)).civil(0);
                let start = DateTime::new(
                    i16::try_from(civil.year).map_err(|_| out())?,
                    civil.month as i8,
                    civil.day as i8,
                    civil.hour as i8,
                    civil.minute as i8,
                    civil.second as i8,
                    civil.nanosecond as i32,
                )
                .map_err(|_| out())?;
                let months = jiff::Span::new()
                    .try_months(self.count)
                    .map_err(|_| out())?;
                // jiff clamps a day the month does not have to its last,
                // as `relativedelta` does.
                let moved = start.checked_add(months).map_err(|_| out())?;
                let civil = Civil::new(
                    i64::from(moved.year()),
                    moved.month() as u8,
                    moved.day() as u8,
                    moved.hour() as u8,
                    moved.minute() as u8,
                    moved.second() as u8,
                    moved.subsec_nanosecond() as u32,
                )
                .ok_or_else(out)?;
                Instant::from_civil(&civil, 0).nanos()
            }
        };
        i64::try_from(nanos).map_err(|_| out())
    }
}

/// The fixed span between a window's ends, each added to 1970-01-01, as
/// the reference's sampler steps back by it (`DateOffsetRange.timeframe`):
/// a window of months spans the days those months have from the epoch.
pub fn timeframe(start: Offset, end: Offset, field: &str) -> Result<i128, CoreError> {
    let end = end.add_to(0, field)?;
    let start = start.add_to(0, field)?;
    Ok(i128::from(end) - i128::from(start))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> i64 {
        let dt: DateTime = text.parse().unwrap();
        let civil = Civil::new(
            i64::from(dt.year()),
            dt.month() as u8,
            dt.day() as u8,
            dt.hour() as u8,
            dt.minute() as u8,
            dt.second() as u8,
            dt.subsec_nanosecond() as u32,
        )
        .unwrap();
        i64::try_from(Instant::from_civil(&civil, 0).nanos()).unwrap()
    }

    #[test]
    fn a_month_is_clamped_to_the_month_s_end() {
        let t = at("2024-01-31T12:34:56.123456789");
        let months = |n| Offset::of(n, TimeUnit::Months);
        assert_eq!(
            months(1).add_to(t, "t").unwrap(),
            at("2024-02-29T12:34:56.123456789")
        );
        assert_eq!(
            months(-1).add_to(t, "t").unwrap(),
            at("2023-12-31T12:34:56.123456789")
        );
        assert_eq!(
            months(13).add_to(t, "t").unwrap(),
            at("2025-02-28T12:34:56.123456789")
        );
        assert_eq!(
            months(1).negated().add_to(t, "t").unwrap(),
            at("2023-12-31T12:34:56.123456789")
        );
        assert_eq!(months(0).add_to(t, "t").unwrap(), t);
        assert_eq!(
            Offset::of(30, TimeUnit::Days).add_to(t, "t").unwrap(),
            at("2024-03-01T12:34:56.123456789")
        );
    }

    #[test]
    fn a_timeframe_of_months_counts_from_the_epoch() {
        let months = |n| Offset::of(n, TimeUnit::Months);
        assert_eq!(
            timeframe(months(0), months(1), "t").unwrap(),
            31 * 86_400 * NANOS_PER_SECOND
        );
        assert_eq!(
            timeframe(months(-1), months(2), "t").unwrap(),
            90 * 86_400 * NANOS_PER_SECOND
        );
    }

    #[test]
    fn a_time_past_the_nanosecond_range_is_refused() {
        let t = at("2262-04-01T00:00:00");
        assert!(Offset::of(1, TimeUnit::Months).add_to(t, "t").is_err());
        assert!(
            Offset::of(i32::MAX, TimeUnit::Months)
                .add_to(0, "t")
                .is_err()
        );
    }
}
