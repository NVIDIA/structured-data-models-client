// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! pandas' `DateOffset`, as the sampler and the label windows use it: a
//! multiple `n` of calendar units, applied to timestamps exactly as pandas
//! 3 applies it, errors and all.
//!
//! A timestamp plus an offset ([`DateOffset::apply`]) takes one of pandas'
//! two paths. An offset of hours, minutes and seconds only is a fixed span
//! (`datetime.timedelta`), added in nanoseconds. Any other goes through
//! `dateutil`'s `relativedelta` on a Python `datetime`: years and months
//! first, the day clamped to the month's last, then days and the time of
//! day, at microsecond resolution with the nanoseconds put back after. So
//! `ts - 2*o` is not `(ts - o) - o` from a month end. Each step raises where
//! Python does: a year outside 1 to 9999, a date outside Python's range, a
//! result outside pandas' nanosecond range.
//!
//! A column plus an offset ([`DateOffset::apply_array`]) is pandas'
//! vectorized path instead: months shifted with the day clamped, then a
//! fixed span added, `NaT` kept, each step refusing an overflow.

use std::fmt;

use serde_json::{Map, Value};

/// One keyword of `pd.DateOffset(**kwds)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unit {
    Years,
    Months,
    Weeks,
    Days,
    Hours,
    Minutes,
    Seconds,
}

impl Unit {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "years" => Unit::Years,
            "months" => Unit::Months,
            "weeks" => Unit::Weeks,
            "days" => Unit::Days,
            "hours" => Unit::Hours,
            "minutes" => Unit::Minutes,
            "seconds" => Unit::Seconds,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Unit::Years => "years",
            Unit::Months => "months",
            Unit::Weeks => "weeks",
            Unit::Days => "days",
            Unit::Hours => "hours",
            Unit::Minutes => "minutes",
            Unit::Seconds => "seconds",
        }
    }

    /// Whether `timedelta` takes it, which makes an offset of only such
    /// units a fixed span rather than a `relativedelta`.
    fn is_sub_daily(self) -> bool {
        matches!(self, Unit::Hours | Unit::Minutes | Unit::Seconds)
    }
}

/// `pd.DateOffset(n=n, **kwds)`, its keywords in the order given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DateOffset {
    pub n: i64,
    pub kwds: Vec<(Unit, i64)>,
}

/// What applying an offset raised: the exception's class and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OffsetError {
    pub code: &'static str,
    pub message: String,
}

impl OffsetError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl fmt::Display for OffsetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for OffsetError {}

/// Which side of a calendar offset `date_offset_to_seconds` bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Earliest,
    Latest,
}

impl Bound {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "earliest" => Some(Bound::Earliest),
            "latest" => Some(Bound::Latest),
            _ => None,
        }
    }

    fn opposite(self) -> Self {
        match self {
            Bound::Earliest => Bound::Latest,
            Bound::Latest => Bound::Earliest,
        }
    }
}

const US_PER_SECOND: i128 = 1_000_000;
const US_PER_DAY: i128 = 86_400 * US_PER_SECOND;
const NS_PER_US: i128 = 1_000;
/// `timedelta`'s limit on its days.
const MAX_DELTA_DAYS: i128 = 999_999_999;
/// The microseconds from 1970 to 0001-01-01 and to 10000-01-01, the ends of
/// Python's `datetime`.
const MIN_DATETIME_US: i128 = -62_135_596_800 * US_PER_SECOND;
const END_DATETIME_US: i128 = 253_402_300_800 * US_PER_SECOND;

impl DateOffset {
    /// `{"n": N, "kwds": {UNIT: VALUE}}`.
    pub fn from_json(value: &Value) -> Result<Self, String> {
        let Value::Object(fields) = value else {
            return Err(format!("an offset is an object, not {value}"));
        };
        let n = match fields.get("n") {
            None => 1,
            Some(n) => n
                .as_i64()
                .ok_or_else(|| format!("an offset's n is an integer, not {n}"))?,
        };
        let mut kwds = Vec::new();
        if let Some(given) = fields.get("kwds") {
            let Value::Object(given) = given else {
                return Err(format!("an offset's kwds are an object, not {given}"));
            };
            for (name, amount) in given {
                let unit =
                    Unit::parse(name).ok_or_else(|| format!("no offset unit named {name:?}"))?;
                let amount = amount
                    .as_i64()
                    .ok_or_else(|| format!("an offset's {name} is an integer, not {amount}"))?;
                kwds.push((unit, amount));
            }
        }
        if kwds.is_empty() {
            return Err("an offset names at least one unit".to_string());
        }
        Ok(Self { n, kwds })
    }

    pub fn to_json(&self) -> Value {
        let mut kwds = Map::new();
        for (unit, amount) in &self.kwds {
            kwds.insert(unit.name().to_string(), Value::from(*amount));
        }
        let mut out = Map::new();
        out.insert("n".to_string(), Value::from(self.n));
        out.insert("kwds".to_string(), Value::Object(kwds));
        Value::Object(out)
    }

    /// `pd.DateOffset(**{unit: amount})`.
    pub fn single(unit: Unit, amount: i64) -> Self {
        Self {
            n: 1,
            kwds: vec![(unit, amount)],
        }
    }

    fn amount(&self, unit: Unit) -> Option<i64> {
        self.kwds.iter().find(|(u, _)| *u == unit).map(|(_, a)| *a)
    }

    fn is_fixed_span(&self) -> bool {
        self.kwds.iter().all(|(unit, _)| unit.is_sub_daily())
    }

    /// What `pd.DateOffset(n=n, **kwds)` raises when it can't be made: an
    /// offset of hours, minutes and seconds only is a `datetime.timedelta`,
    /// whose constructor refuses more than 999,999,999 days. Any other is a
    /// `relativedelta`, which takes any integers.
    pub fn constructed(&self) -> Result<(), OffsetError> {
        if self.is_fixed_span() {
            timedelta_us(self.fixed_seconds().checked_mul(US_PER_SECOND))?;
        }
        Ok(())
    }

    /// `pd.Timestamp(ts, unit='ns') + self` with `sign` 1, or `- self` with
    /// `sign` -1 (which is adding the offset with `n` negated), in
    /// nanoseconds; `i64::MIN` is `NaT`.
    pub fn apply(&self, ts: i64, sign: i64) -> Result<i64, OffsetError> {
        if sign < 0 && self.n == i64::MIN {
            // `ts - offset` is `ts + (-offset)`, and `-offset` has `n` of
            // 2**63, past the `int64` pandas keeps it in.
            return Err(c_long_overflow());
        }
        let n = i128::from(self.n) * i128::from(sign);
        if self.is_fixed_span() {
            let span_us = self.fixed_span_us(n)?;
            // Timestamp + Timedelta casts the microsecond span to the
            // timestamp's nanoseconds, then adds, each step checked.
            let span_ns = span_us * NS_PER_US;
            if i64::try_from(span_ns).is_err() {
                return Err(OffsetError::new(
                    "OutOfBoundsTimedelta",
                    format!(
                        "Cannot cast {} to unit='ns' without overflow.",
                        timedelta_text(span_us)
                    ),
                ));
            }
            let sum = i128::from(ts) + span_ns;
            return i64::try_from(sum).map_err(|_| {
                OffsetError::new(
                    "OutOfBoundsDatetime",
                    format!("Out of bounds nanosecond timestamp: {sum}"),
                )
            });
        }
        let us = i128::from(ts).div_euclid(NS_PER_US);
        let nanos = i128::from(ts).rem_euclid(NS_PER_US);
        let moved = self.relativedelta_add(us, n)?;
        // Timestamp(datetime) is in microseconds; pandas casts it back to the
        // timestamp's nanoseconds and adds the nanoseconds it set aside.
        let ns = moved * NS_PER_US;
        if i64::try_from(ns).is_err() {
            return Err(OffsetError::new(
                "OutOfBoundsDatetime",
                format!(
                    "Cannot cast {} to unit='ns' without overflow.",
                    timestamp_text(moved)
                ),
            ));
        }
        let sum = ns + nanos;
        i64::try_from(sum).map_err(|_| {
            OffsetError::new(
                "OutOfBoundsDatetime",
                format!("Out of bounds nanosecond timestamp: {sum}"),
            )
        })
    }

    /// `Timedelta(self._offset * n)` of a fixed-span offset, in
    /// microseconds: the `timedelta`'s errors, then pandas' past `int64`
    /// microseconds.
    fn fixed_span_us(&self, n: i128) -> Result<i128, OffsetError> {
        let span_us = timedelta_us(self.fixed_seconds().checked_mul(US_PER_SECOND))?;
        let span_us = timedelta_us(span_us.checked_mul(n))?;
        if i64::try_from(span_us).is_err() {
            return Err(OffsetError::new(
                "OutOfBoundsTimedelta",
                "Python int too large to convert to C long",
            ));
        }
        Ok(span_us)
    }

    /// The seconds a fixed-span offset's keywords add up to, before `n`.
    fn fixed_seconds(&self) -> i128 {
        self.kwds
            .iter()
            .map(|(unit, amount)| {
                let amount = i128::from(*amount);
                match unit {
                    Unit::Hours => amount * 3_600,
                    Unit::Minutes => amount * 60,
                    _ => amount,
                }
            })
            .sum()
    }

    /// `datetime + relativedelta(**kwds) * n` for a `datetime` at `us`
    /// microseconds from 1970, in microseconds.
    fn relativedelta_add(&self, us: i128, n: i128) -> Result<i128, OffsetError> {
        let delta = RelativeDelta::of(self).times(n);
        let day = us.div_euclid(US_PER_DAY);
        let time_of_day = us.rem_euclid(US_PER_DAY);
        let (year, month, mday) = civil_from_days(day);
        // Python's integers don't overflow; one past i128 is past every
        // bound below too, and refused as the largest are.
        let mut year = i128::from(year).saturating_add(delta.years);
        let mut month = i128::from(month) + delta.months;
        if month > 12 {
            year = year.saturating_add(1);
            month -= 12;
        } else if month < 1 {
            year = year.saturating_sub(1);
            month += 12;
        }
        let year = python_int_arg(year)?;
        // `datetime.replace` checks the year; `calendar.monthrange` takes any.
        if !(1..=9999).contains(&year) {
            return Err(OffsetError::new(
                "ValueError",
                format!("year {year} is out of range"),
            ));
        }
        let month = month as u32;
        let mday = mday.min(days_in_month(i64::from(year), month));
        // `timedelta(days=, hours=, minutes=, seconds=)`.
        let span = timedelta_us(
            delta
                .days
                .checked_mul(24)
                .and_then(|v| v.checked_add(delta.hours))
                .and_then(|v| v.checked_mul(60))
                .and_then(|v| v.checked_add(delta.minutes))
                .and_then(|v| v.checked_mul(60))
                .and_then(|v| v.checked_add(delta.seconds))
                .and_then(|v| v.checked_mul(US_PER_SECOND)),
        )?;
        let moved = i128::from(days_from_civil(i64::from(year), month, mday)) * US_PER_DAY
            + time_of_day
            + span;
        if !(MIN_DATETIME_US..END_DATETIME_US).contains(&moved) {
            return Err(OffsetError::new("OverflowError", "date value out of range"));
        }
        Ok(moved)
    }

    /// `base/utils.date_offset_to_seconds(offset, bound)`: a calendar unit at
    /// its longest or shortest, whichever lands on `bound`'s side. Python's
    /// integers don't overflow, and an offset of `2**63` years `2**63` times
    /// spans more seconds than an `i128` holds, so the sum is [`Seconds`].
    ///
    /// It multiplies every attribute of the offset by `n`, its `timedelta`
    /// too, so a fixed-span offset raises what that product raises.
    pub fn to_seconds(&self, bound: Bound) -> Result<Seconds, OffsetError> {
        let n = i128::from(self.n);
        if self.is_fixed_span() {
            timedelta_us(
                self.fixed_seconds()
                    .checked_mul(US_PER_SECOND)
                    .and_then(|us| us.checked_mul(n)),
            )?;
        }
        let mut total = Seconds::default();
        for (unit, amount) in &self.kwds {
            if *amount == 0 {
                continue;
            }
            let scaled = i128::from(*amount) * n;
            let per_unit = match unit {
                Unit::Years => elastic_days(scaled, bound, 365, 366) * 86_400,
                Unit::Months => elastic_days(scaled, bound, 28, 31) * 86_400,
                Unit::Weeks => 7 * 86_400,
                Unit::Days => 86_400,
                Unit::Hours => 3_600,
                Unit::Minutes => 60,
                Unit::Seconds => 1,
            };
            total = total + Seconds::product(scaled, per_unit);
        }
        Ok(total)
    }

    /// `rfm._date_offset_to_ns(offset)`: the nanoseconds the offset spans
    /// from 2020-01-01, in whole seconds.
    pub fn to_ns_from_2020(&self) -> Result<i128, OffsetError> {
        // pd.Timestamp('2020-01-01') is in microseconds.
        let reference = 1_577_836_800 * US_PER_SECOND;
        let moved = if self.is_fixed_span() {
            let moved = reference + self.fixed_span_us(i128::from(self.n))?;
            if i64::try_from(moved).is_err() {
                return Err(OffsetError::new(
                    "OutOfBoundsDatetime",
                    format!("Out of bounds microsecond timestamp: {moved}"),
                ));
            }
            moved
        } else {
            self.relativedelta_add(reference, i128::from(self.n))?
        };
        // int(delta.total_seconds()) of a whole number of seconds.
        Ok((moved - reference) / US_PER_SECOND * 1_000_000_000)
    }

    /// `Series + offset` for a `datetime64` column of `per_second` ticks
    /// a second (`10**9` for nanoseconds), as pandas' vectorized path
    /// (`RelativeDeltaOffset._apply_array`) does it: `None` is `NaT`.
    ///
    /// Its span (`_pd_timedelta`) is a `Timedelta`: for an offset of hours,
    /// minutes and seconds only, of the `timedelta` times `n`, in
    /// microseconds; for any other, of its days, weeks, hours, minutes and
    /// seconds in nanoseconds, then times `n` in seconds. Each is refused past
    /// `int64`. Its years and months, times `n`, must fit a C `int`, and each
    /// date they move to must fit the column's ticks. numpy then adds the
    /// span, refusing an overflow.
    pub fn apply_array(
        &self,
        values: &[Option<i64>],
        per_second: i64,
    ) -> Result<Vec<Option<i64>>, OffsetError> {
        self.constructed()?;
        let n = i128::from(self.n);
        let amount = |unit| i128::from(self.amount(unit).unwrap_or(0));
        let out_of_bounds = |seconds: i128| {
            OffsetError::new(
                "OutOfBoundsTimedelta",
                format!("seconds={seconds}: past the int64 ticks of a Timedelta"),
            )
        };
        let span = if self.is_fixed_span() {
            let us = timedelta_us(
                self.fixed_seconds()
                    .checked_mul(US_PER_SECOND)
                    .and_then(|us| us.checked_mul(n)),
            )?;
            i64::try_from(us).map_err(|_| out_of_bounds(us / US_PER_SECOND))?;
            (us / US_PER_SECOND) as i64
        } else {
            // At most 8 * 2**63 * 86_400, well inside an i128.
            let seconds =
                ((amount(Unit::Days) + 7 * amount(Unit::Weeks)) * 24 + amount(Unit::Hours)) * 3_600
                    + amount(Unit::Minutes) * 60
                    + amount(Unit::Seconds);
            if i64::try_from(seconds * 1_000_000_000).is_err() {
                return Err(out_of_bounds(seconds));
            }
            i64::try_from(seconds * n).map_err(|_| c_long_overflow())?
        };
        let months = (amount(Unit::Years) * 12 + amount(Unit::Months))
            .checked_mul(n)
            .ok_or_else(c_long_overflow)?;
        let months = i64::try_from(months).map_err(|_| c_long_overflow())?;
        let months = i32::try_from(months)
            .map_err(|_| OffsetError::new("OverflowError", "value too large to convert to int"))?;
        let shifted = values
            .iter()
            .map(|value| match value {
                Some(value) if months != 0 => shift_months(*value, months, per_second).map(Some),
                _ => Ok(*value),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // numpy casts the span to the column's ticks and adds it, each step
        // checked; NaT stays NaT, and a sum that would be NaT overflows.
        let span = span.checked_mul(per_second).ok_or_else(|| {
            OffsetError::new(
                "OverflowError",
                "Overflow when converting between datetime64 units",
            )
        })?;
        shifted
            .into_iter()
            .map(|value| {
                let Some(value) = value else {
                    return Ok(None);
                };
                match value.checked_add(span) {
                    Some(sum) if sum != i64::MIN => Ok(Some(sum)),
                    _ => Err(OffsetError::new(
                        "OverflowError",
                        "Overflow in timedelta64 + datetime64 addition",
                    )),
                }
            })
            .collect()
    }
}

fn c_long_overflow() -> OffsetError {
    OffsetError::new("OverflowError", "Python int too large to convert to C long")
}

/// `shift_months`: the date moved by `months`, the day clamped to the new
/// month's last, the time of day kept, or pandas' `OverflowError` when it
/// leaves the column's ticks.
fn shift_months(value: i64, months: i32, per_second: i64) -> Result<i64, OffsetError> {
    let per_day = i128::from(per_second) * 86_400;
    let value = i128::from(value);
    let day = value.div_euclid(per_day);
    let time_of_day = value.rem_euclid(per_day);
    let (year, month, mday) = civil_from_days(day);
    // `year_add_months` and `month_add_months` add in C `int`, wrapping,
    // floor-divide as Python does, and return the year as a C `int`.
    let month = month as i32;
    let carry = month.wrapping_add(months).wrapping_sub(1).div_euclid(12);
    let year = i64::from(year.wrapping_add(i64::from(carry)) as i32);
    let month = match month.wrapping_add(months).rem_euclid(12) {
        0 => 12,
        m => m as u32,
    };
    let mday = mday.min(days_in_month(year, month));
    let out = i128::from(days_from_civil(year, month, mday)) * per_day + time_of_day;
    // `npy_datetimestruct_to_datetime` builds nanoseconds from whole
    // microseconds, each step checked, so the last 808 nanoseconds before
    // int64's minimum can't be reached.
    let whole = if per_second == 1_000_000_000 {
        out.div_euclid(1_000) * 1_000
    } else {
        out
    };
    match (i64::try_from(whole), i64::try_from(out)) {
        (Ok(_), Ok(out)) => Ok(out),
        _ => Err(OffsetError::new(
            "OverflowError",
            "Overflow occurred in npy_datetimestruct_to_datetime",
        )),
    }
}

/// `_elastic_days`: the length of a calendar unit that pushes the offset
/// towards `bound`, given the sign of its scaled amount.
fn elastic_days(scaled: i128, bound: Bound, shortest: i128, longest: i128) -> i128 {
    if (scaled > 0) == (bound == Bound::Latest) {
        longest
    } else {
        shortest
    }
}

/// A `datetime.timedelta` of `us` microseconds, or the `OverflowError` its
/// constructor raises: past a C `int` of days, then past 999,999,999 days.
/// `None` is a count past `i128`, which Python holds and refuses as the
/// former.
fn timedelta_us(us: Option<i128>) -> Result<i128, OffsetError> {
    let days = us.map(|us| us.div_euclid(US_PER_DAY));
    let Some(days) = days.filter(|days| i32::try_from(*days).is_ok()) else {
        return Err(OffsetError::new(
            "OverflowError",
            "Python int too large to convert to C int",
        ));
    };
    if days.abs() > MAX_DELTA_DAYS {
        return Err(OffsetError::new(
            "OverflowError",
            format!("days={days}; must have magnitude <= 999999999"),
        ));
    }
    Ok(us.unwrap_or_default())
}

/// A year as a C `int` argument, as `datetime.replace` parses it: a C
/// `long` first, then the `int`'s range.
fn python_int_arg(value: i128) -> Result<i32, OffsetError> {
    let message = if i64::try_from(value).is_err() {
        "Python int too large to convert to C long"
    } else if value > 0 {
        "signed integer is greater than maximum"
    } else {
        "signed integer is less than minimum"
    };
    i32::try_from(value).map_err(|_| OffsetError::new("OverflowError", message))
}

/// A whole number of seconds as Python's unbounded integers hold it, as far
/// as `date_offset_to_seconds` needs: `high * 2**64 + low`, ordered by
/// value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Seconds {
    high: i128,
    low: u64,
}

impl Seconds {
    /// `a * b` for any `a` and a `b` below `2**32`, exactly.
    fn product(a: i128, b: i128) -> Self {
        debug_assert!((0..1 << 32).contains(&b));
        let (a_high, a_low) = (a >> 64, a as u64);
        let low = u128::from(a_low) * b as u128;
        Self {
            high: a_high * b + (low >> 64) as i128,
            low: low as u64,
        }
    }

    pub fn from_i128(value: i128) -> Self {
        Self::product(value, 1)
    }

    /// The value, when an `i128` holds it.
    pub fn to_i128(self) -> Option<i128> {
        self.high
            .checked_mul(1 << 64)?
            .checked_add(i128::from(self.low))
    }

    /// `(high, low)`, for a caller with unbounded integers to rebuild the
    /// value as `high * 2**64 + low`.
    pub fn parts(self) -> (i128, u64) {
        (self.high, self.low)
    }

    /// The value in decimal, exactly, as Python's `str(int)` writes it.
    pub fn to_decimal(self) -> String {
        if let Some(value) = self.to_i128() {
            return value.to_string();
        }
        let negative = self.high < 0;
        // The magnitude as three 64-bit limbs, most significant first.
        let (mut high, mut low) = (self.high as u128, self.low);
        if negative {
            // Two's complement over the 192 bits: invert, then add one.
            high = !high;
            low = !low;
            let (sum, carried) = low.overflowing_add(1);
            low = sum;
            if carried {
                high = high.wrapping_add(1);
            }
        }
        let mut limbs = [(high >> 64) as u64, high as u64, low];
        const CHUNK: u128 = 10_000_000_000_000_000_000;
        let mut chunks = Vec::new();
        while limbs.iter().any(|&limb| limb != 0) {
            let mut remainder = 0u128;
            for limb in &mut limbs {
                let current = (remainder << 64) | u128::from(*limb);
                *limb = (current / CHUNK) as u64;
                remainder = current % CHUNK;
            }
            chunks.push(remainder as u64);
        }
        let mut text = String::from(if negative { "-" } else { "" });
        for (i, chunk) in chunks.iter().rev().enumerate() {
            if i == 0 {
                text.push_str(&chunk.to_string());
            } else {
                text.push_str(&format!("{chunk:019}"));
            }
        }
        text
    }
}

impl std::ops::Add for Seconds {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let low = u128::from(self.low) + u128::from(other.low);
        Self {
            high: self.high + other.high + (low >> 64) as i128,
            low: low as u64,
        }
    }
}

/// `relativedelta(**kwds)`, normalized as its `_fix` normalizes it.
#[derive(Clone, Copy, Debug, Default)]
struct RelativeDelta {
    years: i128,
    months: i128,
    days: i128,
    hours: i128,
    minutes: i128,
    seconds: i128,
}

impl RelativeDelta {
    fn of(offset: &DateOffset) -> Self {
        let mut delta = Self::default();
        for (unit, amount) in &offset.kwds {
            let amount = i128::from(*amount);
            match unit {
                Unit::Years => delta.years = amount,
                Unit::Months => delta.months = amount,
                Unit::Weeks => delta.days += amount * 7,
                Unit::Days => delta.days += amount,
                Unit::Hours => delta.hours = amount,
                Unit::Minutes => delta.minutes = amount,
                Unit::Seconds => delta.seconds = amount,
            }
        }
        delta.fix()
    }

    /// `_fix`: each field carried into the next larger one, keeping its sign.
    ///
    /// Python's integers are unbounded, so a field can pass `i128` here (days
    /// of `-2**63` weeks, `2**62` times). Such a field saturates: any value
    /// that far out is past every bound `relativedelta_add` checks, and is
    /// refused there as Python refuses the exact one, while the carries out of
    /// the smaller fields (at most `i128::MAX / 24`) can't bring it back
    /// within them.
    fn fix(mut self) -> Self {
        fn carry(value: &mut i128, into: &mut i128, limit: u128, base: u128) {
            let magnitude = value.unsigned_abs();
            if magnitude > limit {
                let negative = *value < 0;
                // `divmod(value * sign, base)` of the magnitude: both fit.
                let (div, rem) = ((magnitude / base) as i128, (magnitude % base) as i128);
                *value = if negative { -rem } else { rem };
                *into = into.saturating_add(if negative { -div } else { div });
            }
        }
        carry(&mut self.seconds, &mut self.minutes, 59, 60);
        carry(&mut self.minutes, &mut self.hours, 59, 60);
        carry(&mut self.hours, &mut self.days, 23, 24);
        carry(&mut self.months, &mut self.years, 11, 12);
        self
    }

    /// `relativedelta * n`: each field `int(field * float(n))`.
    fn times(self, n: i128) -> Self {
        let f = n as f64;
        let scale = |value: i128| (value as f64 * f).trunc() as i128;
        Self {
            years: scale(self.years),
            months: scale(self.months),
            days: scale(self.days),
            hours: scale(self.hours),
            minutes: scale(self.minutes),
            seconds: scale(self.seconds),
        }
        .fix()
    }
}

pub fn is_leap(year: i64) -> bool {
    year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

pub fn days_in_month(year: i64, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(year) => 29,
        _ => 28,
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let yoe = year - era * 400;
    let month = i64::from(month);
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The proleptic Gregorian date `days` after 1970-01-01.
pub fn civil_from_days(days: i128) -> (i64, u32, u32) {
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// `str(pd.Timestamp)` of a timestamp in microseconds.
fn timestamp_text(us: i128) -> String {
    let (year, month, day) = civil_from_days(us.div_euclid(US_PER_DAY));
    let time = us.rem_euclid(US_PER_DAY);
    let seconds = time / US_PER_SECOND;
    let fraction = time % US_PER_SECOND;
    let mut text = format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02}",
        seconds / 3_600,
        seconds / 60 % 60,
        seconds % 60
    );
    if fraction != 0 {
        text.push_str(&format!(".{fraction:06}"));
    }
    text
}

/// `str(pd.Timedelta)` of a span in microseconds: days, then the rest of
/// the span as a positive time of day.
fn timedelta_text(us: i128) -> String {
    let days = us.div_euclid(US_PER_DAY);
    let rest = us.rem_euclid(US_PER_DAY);
    let seconds = rest / US_PER_SECOND;
    let fraction = rest % US_PER_SECOND;
    let sign = if days < 0 { " +" } else { " " };
    let mut text = format!(
        "{days} days{sign}{:02}:{:02}:{:02}",
        seconds / 3_600,
        seconds / 60 % 60,
        seconds % 60
    );
    if fraction != 0 {
        text.push_str(&format!(".{:03}{:03}", fraction / 1_000, fraction % 1_000));
    }
    text
}

/// `base/sampler._pooled_offset(offsets, earliest=earliest)`: one of
/// `offsets` that bounds the rest on `earliest`'s side at every anchor, or
/// else a span of seconds that does.
pub fn pooled_offset(offsets: &[DateOffset], earliest: bool) -> Result<DateOffset, OffsetError> {
    if offsets.is_empty() {
        return Err(OffsetError::new("AssertionError", ""));
    }
    let bound = if earliest {
        Bound::Earliest
    } else {
        Bound::Latest
    };
    // In Python's order, `all` stopping at the first other it doesn't
    // bound, so an offset whose span raises raises where it would.
    for candidate in offsets {
        let mine = candidate.to_seconds(bound.opposite())?;
        let mut bounds_all = true;
        for other in offsets
            .iter()
            .filter(|other| !same_offset(other, candidate))
        {
            let theirs = other.to_seconds(bound)?;
            if (earliest && mine > theirs) || (!earliest && mine < theirs) {
                bounds_all = false;
                break;
            }
        }
        if bounds_all {
            return Ok(candidate.clone());
        }
    }
    let spans = offsets
        .iter()
        .map(|offset| offset.to_seconds(bound))
        .collect::<Result<Vec<_>, _>>()?;
    let span = if earliest {
        spans.into_iter().min()
    } else {
        spans.into_iter().max()
    };
    // `pd.DateOffset(seconds=span)` is a `timedelta`, refused past its range
    // as [`DateOffset::constructed`] refuses it; within it, `span` fits.
    let span = span
        .and_then(Seconds::to_i128)
        .and_then(|span| i64::try_from(span).ok());
    let Some(span) = span else {
        return Err(OffsetError::new(
            "OverflowError",
            "Python int too large to convert to C int",
        ));
    };
    let pooled = DateOffset {
        n: 1,
        kwds: vec![(Unit::Seconds, span)],
    };
    pooled.constructed().map(|()| pooled)
}

/// `base/sampler.min_date_offset(*offsets)`: the offset that reaches
/// furthest back for any anchor; `None` when any of them is `None`.
pub fn min_date_offset(offsets: &[Option<DateOffset>]) -> Result<Option<DateOffset>, OffsetError> {
    let Some(given) = offsets.iter().cloned().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    pooled_offset(&given, true).map(Some)
}

/// `base/sampler.max_date_offset(*offsets)`: the offset that reaches
/// furthest forward for any anchor.
pub fn max_date_offset(offsets: &[DateOffset]) -> Result<DateOffset, OffsetError> {
    pooled_offset(offsets, false)
}

/// `==` of two `DateOffset`s: the same `n` and the same keywords, in any
/// order.
fn same_offset(a: &DateOffset, b: &DateOffset) -> bool {
    a.n == b.n && a.kwds.len() == b.kwds.len() && a.kwds.iter().all(|kwd| b.kwds.contains(kwd))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ns(text: &str) -> i64 {
        // YYYY-MM-DD only.
        let year: i64 = text[0..4].parse().unwrap();
        let month: u32 = text[5..7].parse().unwrap();
        let day: u32 = text[8..10].parse().unwrap();
        days_from_civil(year, month, day) * 86_400 * 1_000_000_000
    }

    #[test]
    fn month_ends_clamp_and_twice_is_not_two_steps() {
        let one = DateOffset::single(Unit::Months, 1);
        let two = DateOffset {
            n: 2,
            kwds: vec![(Unit::Months, 1)],
        };
        let start = ns("2024-03-31");
        assert_eq!(two.apply(start, -1).unwrap(), ns("2024-01-31"));
        let once = one.apply(start, -1).unwrap();
        assert_eq!(one.apply(once, -1).unwrap(), ns("2024-01-29"));
    }

    #[test]
    fn a_year_out_of_range_is_a_value_error() {
        let offset = DateOffset::single(Unit::Years, 9000);
        let error = offset.apply(ns("2024-01-01"), 1).unwrap_err();
        assert_eq!(error.code, "ValueError");
        assert_eq!(error.message, "year 11024 is out of range");
    }

    #[test]
    fn pooled_offsets_bound_every_anchor() {
        let month = DateOffset::single(Unit::Months, 1);
        let thirty = DateOffset::single(Unit::Days, 30);
        let seconds = |s| DateOffset::single(Unit::Seconds, s);
        let both = [Some(month.clone()), Some(thirty.clone())];
        assert_eq!(min_date_offset(&both), Ok(Some(seconds(28 * 86_400))));
        assert_eq!(max_date_offset(&[month, thirty]), Ok(seconds(31 * 86_400)));
        let three = DateOffset::single(Unit::Days, 3);
        assert_eq!(min_date_offset(&[Some(three.clone()), None]), Ok(None));
        let five = DateOffset::single(Unit::Days, 5);
        assert_eq!(
            min_date_offset(&[Some(three.clone()), Some(five)]),
            Ok(Some(three))
        );
    }

    #[test]
    fn spans_past_i128_are_exact_and_errors_are_pythons() {
        // date_offset_to_seconds(pd.DateOffset(n=2**63-1, years=2**63-1)),
        // which Python computes exactly.
        let huge = DateOffset {
            n: i64::MAX,
            kwds: vec![(Unit::Years, i64::MAX)],
        };
        let (high, low) = huge.to_seconds(Bound::Latest).unwrap().parts();
        let want = i128::from(i64::MAX).pow(2);
        // want * 366 * 86400, split at 2**64.
        let per = 366 * 86_400u128;
        let low_part = (want as u128 & u128::from(u64::MAX)) * per;
        assert_eq!(low, low_part as u64);
        assert_eq!(high, (want >> 64) * per as i128 + (low_part >> 64) as i128);
        let latest = huge.to_seconds(Bound::Latest).unwrap();
        assert!(latest.to_i128().is_none());
        assert!(latest > huge.to_seconds(Bound::Earliest).unwrap());
        assert_eq!(
            DateOffset::single(Unit::Seconds, -5).to_seconds(Bound::Latest),
            Ok(Seconds::from_i128(-5))
        );
        // pd.DateOffset(hours=2**40) is a timedelta of more days than a C int.
        let error = DateOffset::single(Unit::Hours, 1 << 40)
            .constructed()
            .unwrap_err();
        assert_eq!(error.message, "Python int too large to convert to C int");
        let error = DateOffset::single(Unit::Hours, 24_000_000_000)
            .constructed()
            .unwrap_err();
        assert_eq!(
            error.message,
            "days=1000000000; must have magnitude <= 999999999"
        );
        // A year past a C long, and the days of n * 2**62 weeks.
        let years = DateOffset {
            n: i64::MAX,
            kwds: vec![(Unit::Years, i64::MAX)],
        };
        assert_eq!(
            years.apply(0, 1).unwrap_err().message,
            "Python int too large to convert to C long"
        );
        let weeks = DateOffset {
            n: 1 << 62,
            kwds: vec![(Unit::Weeks, 1 << 62)],
        };
        assert_eq!(
            weeks.apply(0, 1).unwrap_err().message,
            "Python int too large to convert to C int"
        );
    }

    #[test]
    fn seconds_write_as_pythons_exact_integers() {
        let huge = DateOffset {
            n: i64::MAX,
            kwds: vec![(Unit::Years, i64::MAX)],
        };
        // What `date_offset_to_seconds(pd.DateOffset(n=2**63-1,
        // years=2**63-1), "latest")` gives in Python.
        assert_eq!(
            huge.to_seconds(Bound::Latest).unwrap().to_decimal(),
            "2690136279930171116172723976716113847496377600"
        );
        let negative = DateOffset {
            n: 1,
            kwds: vec![(Unit::Months, -(1 << 62))],
        };
        assert_eq!(
            negative.to_seconds(Bound::Latest).unwrap().to_decimal(),
            "-11156590815779536817356800"
        );
        for value in [0, -1, 1, i128::from(i64::MIN), i128::MAX, i128::MIN] {
            assert_eq!(Seconds::from_i128(value).to_decimal(), value.to_string());
        }
        // 2**64 and -(2**64), at a limb's edge.
        let edge = Seconds::from_i128(1 << 64) + Seconds::from_i128(0);
        assert_eq!(edge.to_decimal(), "18446744073709551616");
        assert_eq!(
            Seconds::from_i128(-(1 << 64)).to_decimal(),
            "-18446744073709551616"
        );
    }

    #[test]
    fn carries_past_i128_saturate_into_pythons_overflow() {
        // n of 2**62 times weeks of -2**63 is days past i128, which Python
        // holds exactly and refuses as a C int; so does the port, in a debug
        // build too.
        let offset = DateOffset {
            n: 1 << 62,
            kwds: vec![
                (Unit::Days, 86_400),
                (Unit::Weeks, i64::MIN),
                (Unit::Seconds, -12),
            ],
        };
        assert_eq!(
            offset
                .apply(1_667_260_800_000_000_000, 1)
                .unwrap_err()
                .message,
            "Python int too large to convert to C int"
        );
    }

    #[test]
    fn civil_round_trip() {
        for days in [-719_468i128, -1, 0, 1, 11_016, 2_932_896] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(i128::from(days_from_civil(y, m, d)), days);
        }
    }

    #[test]
    fn timedelta_text_is_pandas() {
        assert_eq!(timedelta_text(-3_600 * 1_000_000), "-1 days +23:00:00");
        assert_eq!(timedelta_text(-1_500_000), "-1 days +23:59:58.500000");
        assert_eq!(timedelta_text(US_PER_DAY * 3), "3 days 00:00:00");
    }
}
