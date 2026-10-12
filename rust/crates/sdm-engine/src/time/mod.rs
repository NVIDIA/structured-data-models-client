// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Time (design, "Time"; decision F3): the core's one instant and one span,
//! civil arithmetic, the zones a run resolves names with, and the reading
//! of times a source stores as text.
//!
//! - [`Instant`] is nanoseconds since 1970-01-01T00:00Z and [`Span`] a
//!   fixed duration in nanoseconds, both `i128`, so every count an int64
//!   of any unit holds converts in and out exactly; a conversion to a
//!   named unit is checked and refuses with `VALUE_OUT_OF_RANGE`.
//! - [`Civil`] is a proleptic Gregorian date and time of day for any year
//!   such a count reaches, by civil arithmetic, not `jiff`'s calendar,
//!   whose timestamps stop at 9999-12-30T22:00Z (decision E6).
//! - A zone's offset comes from `jiff` inside its range; outside it the
//!   instant is moved by whole 400-year Gregorian cycles into a window
//!   `jiff` holds on the same side of the zone's history and the offset
//!   read there ([`Instant::to_local`], [`localize`]): before a zone's
//!   first transition its offset is the one it starts with, after its last
//!   the rule its database ends with repeats every cycle.
//! - [`text_shapes`] is the client's two shapes `date-rules.md` names a
//!   date in text by, and [`text`] reads text by the core's fixed list of
//!   formats within them (`date-rules.md`, "The core's text formats";
//!   T1.25b), refusing by name any other text it is asked to read.

pub mod text;
pub mod text_shapes;

use arrow_schema::TimeUnit;
use jiff::Timestamp;
use jiff::civil::DateTime;
use jiff::tz::{AmbiguousOffset, TimeZone};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// A zone name the run's zones do not hold. It carries no name: a zone's
/// name can come from the data or the caller, and no input value reaches a
/// `CoreError` (decision H17).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownZone;

/// The zones a run resolves names with: the host's zone lookup roots in
/// order, then a bundled database, with the source of each zone recorded
/// by the host (decision A7). The host passes them to `sdm_core::start`
/// (decision H14), and the core resolves through nothing else.
pub trait Zones: Send + Sync {
    /// The zone `name`, in any case, or [`UnknownZone`].
    fn zone(&self, name: &str) -> Result<TimeZone, UnknownZone>;
}

const NANOS_PER_SECOND: i128 = 1_000_000_000;
const SECONDS_PER_DAY: i128 = 86_400;
/// Days in 400 Gregorian years, a whole number of weeks: the calendar, and
/// every rule a zone database writes by month, weekday or day of the year,
/// repeats with it.
const DAYS_PER_CYCLE: i128 = 146_097;
const CYCLE_NANOS: i128 = DAYS_PER_CYCLE * SECONDS_PER_DAY * NANOS_PER_SECOND;

/// A time outside what the named unit's count holds.
pub fn out_of_range(field: &str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::ValueOutOfRange,
        "a time is outside what the unit asked for holds",
    )
    .with_field("field", Detail::text(field))
}

fn per(unit: TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => NANOS_PER_SECOND,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// An instant: nanoseconds since 1970-01-01T00:00Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instant(i128);

/// A fixed duration in nanoseconds; a calendar offset is not one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Span(i128);

impl Span {
    pub const ZERO: Span = Span(0);

    pub const fn from_nanos(nanos: i128) -> Span {
        Span(nanos)
    }

    pub const fn nanos(self) -> i128 {
        self.0
    }

    /// `count` of `unit`.
    pub fn of(count: i64, unit: TimeUnit) -> Span {
        Span(i128::from(count) * per(unit))
    }
}

impl Instant {
    pub const EPOCH: Instant = Instant(0);

    pub const fn from_nanos(nanos: i128) -> Instant {
        Instant(nanos)
    }

    /// Nanoseconds since the epoch.
    pub const fn nanos(self) -> i128 {
        self.0
    }

    /// A count of `unit` since the epoch, as an Arrow timestamp holds it.
    pub fn from_count(count: i64, unit: TimeUnit) -> Instant {
        Instant(i128::from(count) * per(unit))
    }

    /// Midnight of `days` since the epoch, as an Arrow date holds it.
    pub fn from_days(days: i32) -> Instant {
        Instant(i128::from(days) * SECONDS_PER_DAY * NANOS_PER_SECOND)
    }

    /// The count of `unit` since the epoch, exact: an instant between two
    /// counts, or past what an int64 count of it holds, is refused with
    /// `VALUE_OUT_OF_RANGE` naming `field`.
    pub fn to_count(self, unit: TimeUnit, field: &str) -> Result<i64, CoreError> {
        let per = per(unit);
        if self.0 % per != 0 {
            return Err(out_of_range(field));
        }
        i64::try_from(self.0 / per).map_err(|_| out_of_range(field))
    }

    /// `self + span`, or `None` past what an `i128` holds.
    pub fn checked_add(self, span: Span) -> Option<Instant> {
        self.0.checked_add(span.0).map(Instant)
    }

    /// The span from `earlier` to `self`.
    pub fn since(self, earlier: Instant) -> Option<Span> {
        self.0.checked_sub(earlier.0).map(Span)
    }

    /// The instant a wall clock `civil` names at a fixed offset east of
    /// UTC, in seconds.
    pub fn from_civil(civil: &Civil, offset_seconds: i32) -> Instant {
        let days = days_from_civil(civil.year, civil.month, civil.day);
        let seconds = days * SECONDS_PER_DAY
            + i128::from(civil.hour) * 3_600
            + i128::from(civil.minute) * 60
            + i128::from(civil.second)
            - i128::from(offset_seconds);
        Instant(seconds * NANOS_PER_SECOND + i128::from(civil.nanosecond))
    }

    /// The wall clock at a fixed offset east of UTC, in seconds.
    pub fn civil(self, offset_seconds: i32) -> Civil {
        let local = self.0 + i128::from(offset_seconds) * NANOS_PER_SECOND;
        let seconds = local.div_euclid(NANOS_PER_SECOND);
        let nanosecond = local.rem_euclid(NANOS_PER_SECOND) as u32;
        let days = seconds.div_euclid(SECONDS_PER_DAY);
        let of_day = seconds.rem_euclid(SECONDS_PER_DAY) as u32;
        let (year, month, day) = civil_from_days(days);
        Civil {
            year,
            month,
            day,
            hour: (of_day / 3_600) as u8,
            minute: (of_day / 60 % 60) as u8,
            second: (of_day % 60) as u8,
            nanosecond,
        }
    }

    /// The wall clock and offset of this instant in `zone`.
    pub fn to_local(self, zone: &TimeZone) -> Local {
        let (inside, _) = into_range(self.0);
        let timestamp = Timestamp::from_nanosecond(inside)
            .expect("an instant moved into jiff's range is one jiff holds");
        let offset = zone.to_offset(timestamp).seconds();
        Local {
            civil: self.civil(offset),
            offset_seconds: offset,
        }
    }
}

/// An instant's wall clock in a zone, and the zone's offset there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Local {
    pub civil: Civil,
    pub offset_seconds: i32,
}

/// Why a wall clock names no one instant in a zone, as pandas'
/// `tz_localize` refuses it by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ambiguity {
    /// A wall clock the zone skips.
    Gap,
    /// A wall clock the zone shows twice.
    Fold,
}

/// The instant the wall clock `civil` names in `zone`, or why it names
/// none (pandas' `tz_localize` defaults: a gap or a fold is refused).
pub fn localize(civil: &Civil, zone: &TimeZone) -> Result<Instant, Ambiguity> {
    let utc = Instant::from_civil(civil, 0);
    let (_, cycles) = into_range(utc.0);
    let shifted = Civil {
        year: civil.year - 400 * cycles,
        ..*civil
    };
    let datetime = DateTime::new(
        i16::try_from(shifted.year).expect("a year moved into jiff's range is an i16"),
        shifted.month as i8,
        shifted.day as i8,
        shifted.hour as i8,
        shifted.minute as i8,
        shifted.second as i8,
        shifted.nanosecond as i32,
    )
    .expect("a civil time moved into jiff's range is one jiff holds");
    match zone.to_ambiguous_timestamp(datetime).offset() {
        AmbiguousOffset::Unambiguous { offset } => Ok(Instant::from_civil(civil, offset.seconds())),
        AmbiguousOffset::Gap { .. } => Err(Ambiguity::Gap),
        AmbiguousOffset::Fold { .. } => Err(Ambiguity::Fold),
    }
}

/// The first and last instants of the windows instants outside `jiff`'s
/// range are moved into: 400 years from -9000-01-01, before every zone's
/// first transition, and from 9200-01-01, after every zone's last.
fn early_window() -> i128 {
    Instant::from_civil(&Civil::midnight(-9000, 1, 1), 0).0
}

fn late_window() -> i128 {
    Instant::from_civil(&Civil::midnight(9200, 1, 1), 0).0
}

/// `nanos` moved by whole cycles into a window `jiff` holds when it is
/// outside `jiff`'s range (two days in from each end, so a zone's offset
/// keeps the wall clock inside it too), with the number of cycles moved;
/// unmoved otherwise.
fn into_range(nanos: i128) -> (i128, i64) {
    let margin = 2 * SECONDS_PER_DAY * NANOS_PER_SECOND;
    let low = Timestamp::MIN.as_nanosecond() + margin;
    let high = Timestamp::MAX.as_nanosecond() - margin;
    if (low..=high).contains(&nanos) {
        return (nanos, 0);
    }
    let window = if nanos < low {
        early_window()
    } else {
        late_window()
    };
    let cycles = (nanos - window).div_euclid(CYCLE_NANOS);
    let moved = nanos - cycles * CYCLE_NANOS;
    (
        moved,
        i64::try_from(cycles).expect("an i128 of nanoseconds spans few cycles"),
    )
}

/// A proleptic Gregorian date and time of day, for any year an `i128` of
/// nanoseconds reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Civil {
    pub year: i64,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub nanosecond: u32,
}

impl Civil {
    /// The date and time given, if it is one: a month of 1 to 12, a day the
    /// month has, an hour to 23, a minute and a second to 59.
    pub fn new(
        year: i64,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        nanosecond: u32,
    ) -> Option<Civil> {
        let ok = (1..=12).contains(&month)
            && day >= 1
            && day <= days_in_month(year, month)
            && hour < 24
            && minute < 60
            && second < 60
            && nanosecond < 1_000_000_000;
        ok.then_some(Civil {
            year,
            month,
            day,
            hour,
            minute,
            second,
            nanosecond,
        })
    }

    fn midnight(year: i64, month: u8, day: u8) -> Civil {
        Civil {
            year,
            month,
            day,
            hour: 0,
            minute: 0,
            second: 0,
            nanosecond: 0,
        }
    }
}

impl std::fmt::Display for Civil {
    /// `YYYY-MM-DDTHH:MM:SS.fffffffff`, a year past four digits or before
    /// the year 0 signed, as ISO 8601's expanded years write it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if (0..=9999).contains(&self.year) {
            write!(f, "{:04}", self.year)?;
        } else {
            write!(f, "{:+07}", self.year)?;
        }
        write!(
            f,
            "-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}",
            self.month, self.day, self.hour, self.minute, self.second, self.nanosecond
        )
    }
}

/// Whether `year` is a Gregorian leap year.
pub fn is_leap(year: i64) -> bool {
    year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

/// The days of `month` in `year`.
pub fn days_in_month(year: i64, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days from 1970-01-01 to a date (H. Hinnant's `days_from_civil`).
fn days_from_civil(year: i64, month: u8, day: u8) -> i128 {
    let y = i128::from(year) - i128::from(month <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i128::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i128::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * DAYS_PER_CYCLE + doe - 719_468
}

/// The date `days` after 1970-01-01 (H. Hinnant's `civil_from_days`).
fn civil_from_days(days: i128) -> (i64, u8, u8) {
    let z = days + 719_468;
    let era = z.div_euclid(DAYS_PER_CYCLE);
    let doe = z - era * DAYS_PER_CYCLE;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u8;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u8;
    let year = yoe + era * 400 + i128::from(month <= 2);
    (
        i64::try_from(year).expect("an i128 of nanoseconds spans an i64 of years"),
        month,
        day,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_arithmetic_round_trips_across_the_range() {
        for nanos in [
            0_i128,
            -1,
            1,
            i128::from(i64::MIN) * 1_000,
            i128::from(i64::MAX) * 1_000,
            i128::from(i64::MIN),
            i128::from(i64::MAX),
            253_402_300_799_999_999_999,
            -62_135_596_800_000_000_000,
        ] {
            let instant = Instant::from_nanos(nanos);
            for offset in [0, 3_600, -18_000, 45_900] {
                let civil = instant.civil(offset);
                assert_eq!(
                    Instant::from_civil(&civil, offset),
                    instant,
                    "{nanos} at {offset}"
                );
                assert!(
                    Civil::new(
                        civil.year,
                        civil.month,
                        civil.day,
                        civil.hour,
                        civil.minute,
                        civil.second,
                        civil.nanosecond
                    )
                    .is_some()
                );
            }
        }
        assert_eq!(
            Instant::from_nanos(0).civil(0).to_string(),
            "1970-01-01T00:00:00.000000000"
        );
        assert_eq!(
            Instant::from_count(i64::MAX, TimeUnit::Microsecond)
                .civil(0)
                .to_string(),
            "+294247-01-10T04:00:54.775807000"
        );
        assert_eq!(
            Instant::from_count(i64::MIN, TimeUnit::Microsecond)
                .civil(0)
                .to_string(),
            "-290308-12-21T19:59:05.224192000"
        );
    }

    #[test]
    fn conversions_to_a_unit_are_checked() {
        let us = Instant::from_count(i64::MAX, TimeUnit::Microsecond);
        assert!(us.to_count(TimeUnit::Nanosecond, "t").is_err());
        assert_eq!(us.to_count(TimeUnit::Microsecond, "t").unwrap(), i64::MAX);
        assert!(
            Instant::from_nanos(1_500)
                .to_count(TimeUnit::Microsecond, "t")
                .is_err()
        );
        let err = us.to_count(TimeUnit::Nanosecond, "t").unwrap_err();
        assert_eq!(err.code, ErrorCode::ValueOutOfRange);
        assert_eq!(Span::of(2, TimeUnit::Second).nanos(), 2_000_000_000);
    }

    #[test]
    fn a_civil_date_is_checked() {
        assert!(Civil::new(2024, 2, 29, 0, 0, 0, 0).is_some());
        assert!(Civil::new(2023, 2, 29, 0, 0, 0, 0).is_none());
        assert!(Civil::new(2100, 2, 29, 0, 0, 0, 0).is_none());
        assert!(Civil::new(2000, 2, 29, 0, 0, 0, 0).is_some());
        assert!(Civil::new(2024, 13, 1, 0, 0, 0, 0).is_none());
        assert!(Civil::new(2024, 1, 1, 24, 0, 0, 0).is_none());
        assert!(Civil::new(2024, 1, 1, 0, 0, 60, 0).is_none());
    }

    #[test]
    fn utc_is_offset_zero_everywhere() {
        for nanos in [
            i128::from(i64::MIN) * 1_000,
            0,
            i128::from(i64::MAX) * 1_000,
        ] {
            let local = Instant::from_nanos(nanos).to_local(&TimeZone::UTC);
            assert_eq!(local.offset_seconds, 0);
            assert_eq!(
                localize(&local.civil, &TimeZone::UTC),
                Ok(Instant::from_nanos(nanos))
            );
        }
    }
}
