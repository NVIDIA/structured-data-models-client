// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `DateOffsetRange`: an aggregation's window, and the date offsets it
//! stands for.
//!
//! Ranges are shared and changed in place exactly where Python shares and
//! changes them: merging two ranges first moves both to the same unit, and
//! that move rewrites the ranges it was given. A window's unit is therefore
//! not always the unit the query wrote, and the JSON shows the rewritten one.

use std::cell::RefCell;
use std::rc::Rc;

use crate::pyint::PyInt;
use crate::types::TimeUnit;

pub type DorRef = Rc<RefCell<DateOffsetRange>>;

/// The bounds `np.iinfo(np.int32)` gives a window's ends.
const INT32_MIN: i64 = i32::MIN as i64;
const INT32_MAX: i64 = i32::MAX as i64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DateOffsetRange {
    /// `None` for an open start, `-INF`.
    pub start: Option<i64>,
    pub end: i64,
    pub unit: TimeUnit,
}

/// `validate_int(value, int32 bounds)` as the `ValueError` it raises.
fn check_int32(value: &PyInt) -> Result<i64, String> {
    match value.to_i64() {
        Some(v) if (INT32_MIN..=INT32_MAX).contains(&v) => Ok(v),
        _ => Err(format!(
            "Unsupported constant. Constant {value} is outside of the range of supported \
             integers ({INT32_MIN}, {INT32_MAX})."
        )),
    }
}

impl DateOffsetRange {
    /// `DateOffsetRange(start, end, unit)`, with its checks: the start
    /// before the end, and both within int32.
    pub fn new(start: Option<&PyInt>, end: &PyInt, unit: TimeUnit) -> Result<DorRef, String> {
        if let Some(start) = start
            && start >= end
        {
            return Err(format!(
                "The start date offset '{start}' needs to be smaller than the end date offset \
                 '{end}' to specify a valid time range."
            ));
        }
        let start = start.map(check_int32).transpose()?;
        let end = check_int32(end)?;
        Ok(Rc::new(RefCell::new(DateOffsetRange { start, end, unit })))
    }

    pub fn from_i64(start: Option<i64>, end: i64, unit: TimeUnit) -> Result<DorRef, String> {
        Self::new(
            start.map(PyInt::from_i64).as_ref(),
            &PyInt::from_i64(end),
            unit,
        )
    }

    pub fn is_open(&self) -> bool {
        self.start.is_none()
    }

    /// The start as a date offset, `None` when open.
    pub fn start_date_offset(&self) -> Option<DateOffset> {
        self.start.map(|v| DateOffset::of(self.unit, v))
    }

    pub fn end_date_offset(&self) -> DateOffset {
        DateOffset::of(self.unit, self.end)
    }

    /// `merge_ranges`: the earlier start and the later end, in one unit.
    /// Both ranges are first moved to the same unit, in place.
    pub fn merge_ranges(a: &DorRef, b: &DorRef) -> Result<DorRef, String> {
        Self::move_to_same_unit(a, b);
        let (a, b) = (a.borrow().clone(), b.borrow().clone());
        let start = match (a.start, b.start) {
            (Some(x), Some(y)) => Some(x.min(y)),
            _ => None,
        };
        Self::from_i64(start, a.end.max(b.end), a.unit)
    }

    /// `move_to_same_unit`: weeks become days, then months days, days
    /// hours, hours minutes and minutes seconds, a step at a time, until the
    /// two units agree. Each step rewrites both ranges.
    pub fn move_to_same_unit(a: &DorRef, b: &DorRef) {
        if Rc::ptr_eq(a, b) || a.borrow().unit == b.borrow().unit {
            return;
        }
        let steps: [(TimeUnit, TimeUnit, i64); 5] = [
            (TimeUnit::Weeks, TimeUnit::Days, 7),
            (TimeUnit::Months, TimeUnit::Days, 30),
            (TimeUnit::Days, TimeUnit::Hours, 24),
            (TimeUnit::Hours, TimeUnit::Minutes, 60),
            (TimeUnit::Minutes, TimeUnit::Seconds, 60),
        ];
        for (from, to, factor) in steps {
            for range in [a, b] {
                let mut range = range.borrow_mut();
                if range.unit == from {
                    range.start = range.start.map(|v| v * factor);
                    range.end *= factor;
                    range.unit = to;
                }
            }
            if a.borrow().unit == b.borrow().unit {
                return;
            }
        }
    }

    /// `timeframe`: how long the window is, `None` when it is open.
    pub fn timeframe(&self) -> Option<Timedelta> {
        let start = self.start_date_offset()?;
        Some(Timedelta(
            self.end_date_offset().seconds_after_epoch() - start.seconds_after_epoch(),
        ))
    }

    /// `date_offset_repr`: `8 months` for `<DateOffset: months=8>`.
    pub fn date_offset_repr(offset: &DateOffset) -> String {
        offset.0.first().map_or_else(String::new, |(unit, value)| {
            format!("{value} {}", unit.value())
        })
    }

    /// `timedelta_repr`: a duration in words, at the granularity of `unit`.
    pub fn timedelta_repr(td: Timedelta, unit: TimeUnit) -> String {
        let seconds = td.0;
        let minutes = seconds.div_euclid(60);
        let (hours, r_minutes) = (minutes.div_euclid(60), minutes.rem_euclid(60));
        let (days, r_hours) = (hours.div_euclid(24), hours.rem_euclid(24));
        let (months, r_days) = (days.div_euclid(30), days.rem_euclid(30));
        let plural = |n: i64, word: &str| {
            if n > 1 {
                format!("{n} {word}s")
            } else {
                format!("{n} {word}")
            }
        };
        let minutes_repr = plural(minutes, "minute");
        let r_minutes_repr = plural(r_minutes, "minute");
        let hours_repr = plural(hours, "hour");
        let r_hours_repr = plural(r_hours, "hour");
        let days_repr = plural(days, "day");
        let r_days_repr = plural(r_days, "day");
        let months_repr = plural(months, "month");
        match unit {
            TimeUnit::Minutes => minutes_repr,
            TimeUnit::Hours if r_minutes == 0 => hours_repr,
            TimeUnit::Hours => format!("{hours_repr} and {r_minutes_repr}"),
            TimeUnit::Days if r_hours == 0 => days_repr,
            TimeUnit::Days => format!("{days_repr} and {r_hours_repr}"),
            _ if r_days == 0 && r_hours == 0 => months_repr,
            _ if r_days == 0 => format!("{months_repr} and {r_hours_repr}"),
            _ if r_hours == 0 => format!("{months_repr} and {r_days_repr}"),
            _ => format!("{months_repr}, {r_days_repr} and {r_hours_repr}"),
        }
    }

    /// `date_offset_multiply`.
    pub fn date_offset_multiply(offset: &DateOffset, multiplier: i64) -> DateOffset {
        DateOffset(offset.0.iter().map(|(u, v)| (*u, v * multiplier)).collect())
    }

    /// `date_offset_add`.
    pub fn date_offset_add(lhs: &DateOffset, rhs: &DateOffset) -> DateOffset {
        let mut out = lhs.clone();
        for (unit, value) in &rhs.0 {
            match out.0.iter_mut().find(|(u, _)| u == unit) {
                Some((_, v)) => *v += value,
                None => out.0.push((*unit, *value)),
            }
        }
        out
    }

    /// `date_offset_subtract`.
    pub fn date_offset_subtract(minuend: &DateOffset, subtrahend: &DateOffset) -> DateOffset {
        let negated = DateOffset(subtrahend.0.iter().map(|(u, v)| (*u, -v)).collect());
        Self::date_offset_add(minuend, &negated)
    }

    /// `date_offset_gt`: compared as two moments after 1970-01-01.
    pub fn date_offset_gt(left: &DateOffset, right: &DateOffset) -> bool {
        left.seconds_after_epoch() > right.seconds_after_epoch()
    }
}

/// A `pd.DateOffset` of whole units, in the order its keywords were given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DateOffset(pub Vec<(TimeUnit, i64)>);

/// A `pd.Timedelta`, in whole seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Timedelta(pub i64);

impl DateOffset {
    pub fn of(unit: TimeUnit, value: i64) -> Self {
        DateOffset(vec![(unit, value)])
    }

    /// Seconds after 1970-01-01 of `1970-01-01 + self`: months first, then
    /// the rest, as `relativedelta` adds them.
    fn seconds_after_epoch(&self) -> i64 {
        let months: i64 = self
            .0
            .iter()
            .filter(|(u, _)| *u == TimeUnit::Months)
            .map(|(_, v)| v)
            .sum();
        let total = 1970 * 12 + months;
        let (year, month) = (total.div_euclid(12), total.rem_euclid(12) + 1);
        let mut seconds = days_from_civil(year, month as u32, 1) * 86_400;
        for (unit, value) in &self.0 {
            seconds += value
                * match unit {
                    TimeUnit::Seconds => 1,
                    TimeUnit::Minutes => 60,
                    TimeUnit::Hours => 3600,
                    TimeUnit::Days => 86_400,
                    TimeUnit::Weeks => 604_800,
                    TimeUnit::Months => 0,
                };
        }
        seconds
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date.
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    // Howard Hinnant's `days_from_civil`.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let m = i64::from(month);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn range(start: Option<i64>, end: i64, unit: TimeUnit) -> DorRef {
        DateOffsetRange::from_i64(start, end, unit).unwrap()
    }

    #[test]
    fn checks_as_python_does() {
        let err = DateOffsetRange::from_i64(Some(30), 0, TimeUnit::Days).unwrap_err();
        assert!(err.contains("'30' needs to be smaller than the end date offset '0'"));
        let big = PyInt::parse("2147483648").unwrap();
        assert!(DateOffsetRange::new(None, &big, TimeUnit::Days).is_err());
    }

    #[test]
    fn merging_moves_both_ranges_in_place() {
        let a = range(Some(0), 2, TimeUnit::Weeks);
        let b = range(Some(0), 5, TimeUnit::Hours);
        let merged = DateOffsetRange::merge_ranges(&a, &b).unwrap();
        assert_eq!(a.borrow().unit, TimeUnit::Hours);
        assert_eq!(a.borrow().end, 336);
        assert_eq!(merged.borrow().end, 336);
        let c = range(None, 1, TimeUnit::Months);
        let merged =
            DateOffsetRange::merge_ranges(&c, &range(Some(-3), 2, TimeUnit::Days)).unwrap();
        assert_eq!(
            *merged.borrow(),
            DateOffsetRange {
                start: None,
                end: 30,
                unit: TimeUnit::Days
            }
        );
    }

    #[test]
    fn offsets_compare_and_print() {
        let month = DateOffset::of(TimeUnit::Months, 1);
        let days = DateOffset::of(TimeUnit::Days, 30);
        assert!(DateOffsetRange::date_offset_gt(&month, &days));
        assert_eq!(DateOffsetRange::date_offset_repr(&month), "1 months");
        let sum = DateOffsetRange::date_offset_add(&month, &days);
        assert_eq!(
            sum,
            DateOffset(vec![(TimeUnit::Months, 1), (TimeUnit::Days, 30)])
        );
        let td = Timedelta(30 * 86_400 + 18 * 3600 + 22 * 60 + 19);
        assert_eq!(
            DateOffsetRange::timedelta_repr(td, TimeUnit::Hours),
            "738 hours and 22 minutes"
        );
        assert_eq!(
            DateOffsetRange::timedelta_repr(td, TimeUnit::Days),
            "30 days and 18 hours"
        );
        assert_eq!(
            DateOffsetRange::timedelta_repr(td, TimeUnit::Months),
            "1 month and 18 hours"
        );
        let r = range(Some(0), 1, TimeUnit::Months);
        assert_eq!(r.borrow().timeframe(), Some(Timedelta(31 * 86_400)));
    }
}
