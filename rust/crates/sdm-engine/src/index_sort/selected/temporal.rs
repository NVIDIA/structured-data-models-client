// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed instant/civil comparisons, including timezone identity and fold equality.
use super::{atoms, invalid, object_sort, sorted};
use sdm_contract::CoreError;
use sdm_contract::tabular::{
    OrderedSampledTabularRows as Outcome, TabularIndexScalar as Atom, TabularOrderError as Error,
    TabularSortProfile as Profile, TemporalUnit,
};
use std::cmp::Ordering;

const DAY: i128 = 86_400_000_000;
const EPOCH: i128 = 719_163;

fn scale(unit: TemporalUnit) -> i128 {
    match unit {
        TemporalUnit::Second => 1_000_000_000,
        TemporalUnit::Millisecond => 1_000_000,
        TemporalUnit::Microsecond => 1_000,
        TemporalUnit::Nanosecond => 1,
    }
}

fn ordinal(year: i64, month: u8, day: u8) -> Option<i128> {
    if !(1..=12).contains(&month) {
        return None;
    }
    let year = i128::from(year);
    let leap = year.rem_euclid(4) == 0 && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0);
    let lengths = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day == 0 || i128::from(day) > lengths[usize::from(month - 1)] {
        return None;
    }
    let before = year - 1;
    Some(
        365 * before + before.div_euclid(4) - before.div_euclid(100)
            + before.div_euclid(400)
            + lengths[..usize::from(month - 1)].iter().sum::<i128>()
            + i128::from(day),
    )
}

#[derive(Clone, Copy)]
struct Civil {
    local: i128,
    offset: Option<i128>,
    alternate: Option<i128>,
    timezone: Option<u64>,
}
impl Civil {
    fn instant(self) -> i128 {
        self.local - self.offset.unwrap_or(0)
    }
}

fn civil(value: &Atom) -> Option<Civil> {
    match value {
        Atom::DateTime {
            ordinal,
            micros,
            offset_micros,
        } => Some(Civil {
            local: i128::from(*ordinal) * DAY + i128::from(*micros),
            offset: offset_micros.map(i128::from),
            alternate: offset_micros.map(i128::from),
            timezone: None,
        }),
        Atom::ZonedDateTime {
            ordinal,
            micros,
            timezone,
            offset_micros,
            alternate_offset_micros,
        } => Some(Civil {
            local: i128::from(*ordinal) * DAY + i128::from(*micros),
            offset: Some(i128::from(*offset_micros)),
            alternate: Some(i128::from(*alternate_offset_micros)),
            timezone: Some(*timezone),
        }),
        Atom::Timestamp {
            ticks,
            unit,
            year,
            month,
            day,
            micros,
            timezone,
            alternate_offset_micros,
            ..
        } => {
            let local = ordinal(*year, *month, *day)? * DAY + i128::from(*micros);
            // Both floors discard the same sub-microsecond remainder.
            let utc = (i128::from(*ticks) * scale(*unit)).div_euclid(1_000);
            Some(Civil {
                local,
                offset: timezone.map(|_| local - (utc + EPOCH * DAY)),
                alternate: alternate_offset_micros.map(i128::from),
                timezone: *timezone,
            })
        }
        _ => None,
    }
}

fn validate(value: &Atom) -> Result<(), CoreError> {
    match value {
        Atom::Timestamp {
            ticks,
            unit,
            year,
            month,
            day,
            micros,
            nanosecond,
            timezone,
            alternate_offset_micros,
        } => {
            if *ticks == i64::MIN
                || ordinal(*year, *month, *day).is_none()
                || i128::from(*micros) >= DAY
                || *nanosecond >= 1_000
                || timezone == &Some(0)
            {
                return Err(invalid());
            }
            let ns = i128::from(*ticks) * scale(*unit);
            if ns.rem_euclid(1_000) != i128::from(*nanosecond) {
                return Err(invalid());
            }
            let c = civil(value).ok_or_else(invalid)?;
            if c.offset.is_some_and(|o| o.abs() >= DAY)
                || alternate_offset_micros.is_some_and(|o| i128::from(o).abs() >= DAY)
                || timezone.is_none()
                    && (alternate_offset_micros.is_some()
                        || c.local != ns.div_euclid(1_000) + EPOCH * DAY)
                || timezone.is_some()
                    && (1..=9999).contains(year)
                    && alternate_offset_micros.is_none()
            {
                return Err(invalid());
            }
            Ok(())
        }
        Atom::ZonedDateTime {
            ordinal,
            micros,
            timezone,
            offset_micros,
            alternate_offset_micros,
        } => {
            if !(1..=3_652_059).contains(ordinal)
                || i128::from(*micros) >= DAY
                || *timezone == 0
                || i128::from(*offset_micros).abs() >= DAY
                || i128::from(*alternate_offset_micros).abs() >= DAY
            {
                return Err(invalid());
            }
            Ok(())
        }
        Atom::Date { .. } | Atom::DateTime { .. } => atoms::validate_atom(value),
        _ => Err(invalid()),
    }
}

#[derive(Clone, Copy)]
struct Compared {
    ordering: Ordering,
    equal: bool,
}
impl From<Ordering> for Compared {
    fn from(ordering: Ordering) -> Self {
        Self {
            ordering,
            equal: ordering == Ordering::Equal,
        }
    }
}
fn civil_compare(a: Civil, b: Civil) -> Result<Compared, Error> {
    if a.offset.is_some() != b.offset.is_some() {
        return Err(Error::NaiveAwareDatetime);
    }
    if a.timezone.is_some() && a.timezone == b.timezone {
        return Ok(a.local.cmp(&b.local).into());
    }
    let ordering = a.instant().cmp(&b.instant());
    Ok(Compared {
        ordering,
        equal: ordering == Ordering::Equal && a.offset == a.alternate && b.offset == b.alternate,
    })
}
fn compare(a: &Atom, b: &Atom) -> Result<Compared, Error> {
    match (a, b) {
        (
            Atom::Timestamp {
                ticks: a,
                unit: au,
                timezone: at,
                ..
            },
            Atom::Timestamp {
                ticks: b,
                unit: bu,
                timezone: bt,
                ..
            },
        ) => {
            if at.is_some() != bt.is_some() {
                return Err(Error::TimestampAwarenessMismatch);
            }
            Ok((i128::from(*a) * scale(*au))
                .cmp(&(i128::from(*b) * scale(*bu)))
                .into())
        }
        (Atom::Timestamp { .. }, Atom::Date { .. })
        | (Atom::Date { .. }, Atom::Timestamp { .. }) => Err(Error::TimestampDateMismatch),
        (
            ts @ Atom::Timestamp {
                year, nanosecond, ..
            },
            dt @ (Atom::DateTime { .. } | Atom::ZonedDateTime { .. }),
        )
        | (
            dt @ (Atom::DateTime { .. } | Atom::ZonedDateTime { .. }),
            ts @ Atom::Timestamp {
                year, nanosecond, ..
            },
        ) => {
            let reverse = !matches!(a, Atom::Timestamp { .. });
            if !(1..=9999).contains(year) {
                return Err(Error::TemporalYearOutOfRange);
            }
            let tc = civil(ts).expect("validated temporal atom");
            let dc = civil(dt).expect("validated civil atom");
            let mut result = if *nanosecond == 0 {
                civil_compare(tc, dc)?
            } else {
                if tc.offset.is_some() != dc.offset.is_some() {
                    return Err(Error::TimestampAwarenessMismatch);
                }
                let left = tc.instant() * 1_000 + i128::from(*nanosecond);
                (left.cmp(&(dc.instant() * 1_000))).into()
            };
            if reverse {
                result.ordering = result.ordering.reverse();
            }
            Ok(result)
        }
        (Atom::ZonedDateTime { .. }, Atom::Date { .. })
        | (Atom::Date { .. }, Atom::ZonedDateTime { .. }) => Err(Error::DateTimeMismatch),
        (Atom::ZonedDateTime { .. }, Atom::DateTime { .. } | Atom::ZonedDateTime { .. })
        | (Atom::DateTime { .. }, Atom::ZonedDateTime { .. }) => {
            civil_compare(civil(a).unwrap(), civil(b).unwrap())
        }
        _ => atoms::compare_atom(a, b)
            .map(Into::into)
            .ok_or_else(|| atoms::failure_reason(a, b)),
    }
}

/// Equality-aware observation for existing incomparable MultiIndex level handling.
pub(super) fn rank_comparison(a: &Atom, b: &Atom) -> Option<Ordering> {
    compare(a, b).ok().and_then(|result| {
        (result.ordering != Ordering::Equal || result.equal).then_some(result.ordering)
    })
}

pub(super) fn order(
    values: &[Atom],
    missing: &[bool],
    profile: Profile,
) -> Result<Outcome, CoreError> {
    if values.len() != missing.len() {
        return Err(invalid());
    }
    for (value, absent) in values.iter().zip(missing) {
        if !absent {
            validate(value)?;
        }
    }
    // Mixed civil/instant equality is not a total order; producer binding remains reached.
    if profile == Profile::Unbound {
        return Ok(Outcome::NeedsProfile);
    }
    let mut increasing = !missing.iter().any(|v| *v);
    let mut decreasing = increasing;
    if increasing {
        for current in 1..values.len() {
            let result = compare(&values[current], &values[current - 1]);
            match result {
                Ok(result) if result.ordering == Ordering::Less => increasing = false,
                Ok(result) if result.ordering == Ordering::Greater => decreasing = false,
                Ok(result) if result.equal => {}
                // All closed temporal failures here are probe-suppressed TypeError/ValueError.
                _ => {
                    increasing = false;
                    decreasing = false;
                }
            }
            if !increasing && !decreasing {
                break;
            }
        }
    }
    if increasing {
        return Ok(sorted((0..values.len()).collect(), true));
    }
    let present: Vec<_> = missing
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (!m).then_some(i))
        .collect();
    let mut first_error = None;
    let ordered = if profile == Profile::Stable {
        let mut order: Vec<_> = (0..present.len()).collect();
        order.sort_by(|a, b| {
            if first_error.is_some() {
                return Ordering::Equal;
            }
            let (left, right) = (present[*a], present[*b]);
            match compare(&values[left], &values[right]) {
                Ok(result) => result.ordering,
                Err(reason) => {
                    first_error = Some((reason, left, right));
                    Ordering::Equal
                }
            }
        });
        first_error.map_or(Ok(order), Err)
    } else {
        object_sort::argsort(present.len(), |a, b| {
            let (left, right) = (present[a], present[b]);
            compare(&values[left], &values[right])
                .map(|r| r.ordering == Ordering::Less)
                .map_err(|reason| (reason, left, right))
        })
    };
    match ordered {
        Ok(order) => {
            let mut positions: Vec<_> = order.into_iter().map(|i| present[i]).collect();
            positions.extend(
                missing
                    .iter()
                    .enumerate()
                    .filter_map(|(i, m)| m.then_some(i)),
            );
            Ok(sorted(positions, false))
        }
        Err((reason, left, right)) => Ok(Outcome::Refused {
            reason,
            left,
            right,
        }),
    }
}
