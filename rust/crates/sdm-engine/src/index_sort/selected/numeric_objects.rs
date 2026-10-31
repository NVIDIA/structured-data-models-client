// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed scalar-owner comparisons; producer promotion differs from sort policy.
use super::atoms::{compare_integer, integer, integer_float};
use super::{invalid, sorted};
use sdm_contract::CoreError;
use sdm_contract::tabular::{
    IntegerDestination, NativeMissingOwner, NumericComparisonProfile as Semantics,
    NumericOverflowPolicy as Overflow, NumericOverflowWarnings, NumericWarningStage as Stage,
    OrderedSampledTabularRows as Outcome, TabularIndexScalar as Atom, TabularOrderError as Error,
    TabularSortProfile as Profile,
};
use std::cmp::Ordering;

use super::object_sort;

/// One operation's bounded overflow diagnostics, shared by nested index levels.
#[derive(Default)]
pub(super) struct Diagnostics {
    pub warnings: Vec<NumericOverflowWarnings>,
    pub prints: u64,
    pub destination: Option<IntegerDestination>,
    pub reverse_failure: bool,
    pub count_overflow: bool,
}
impl Diagnostics {
    fn overflow(&mut self, policy: Option<Overflow>, stage: Stage) -> Result<(), Error> {
        match policy {
            None => Err(Error::NativeEffectsPending),
            Some(Overflow::Ignore) => Ok(()),
            Some(Overflow::Raise) => Err(Error::FloatingOverflow),
            Some(Overflow::Warn) => {
                if let Some(last) = self.warnings.last_mut()
                    && last.stage == stage
                {
                    match last.count.checked_add(1) {
                        Some(count) => last.count = count,
                        None => {
                            self.count_overflow = true;
                            return Err(Error::NativeEffectsPending);
                        }
                    }
                } else {
                    self.warnings
                        .push(NumericOverflowWarnings { stage, count: 1 });
                }
                Ok(())
            }
            Some(Overflow::Print) => {
                match self.prints.checked_add(1) {
                    Some(count) => self.prints = count,
                    None => {
                        self.count_overflow = true;
                        return Err(Error::NativeEffectsPending);
                    }
                }
                Ok(())
            }
        }
    }
}
fn weak(semantics: Semantics) -> bool {
    matches!(
        semantics,
        Semantics::WeakScalars | Semantics::WeakFixedIntegerPairs
    )
}

#[derive(Clone, Copy)]
enum Owner {
    Missing(NativeMissingOwner),
    UnboundedInteger,
    Binary64,
    Boolean,
    Integer { signed: bool, width: u8 },
    Floating { width: u8 },
}
#[derive(Clone, Copy)]
enum Number<'a> {
    Missing,
    Integer(&'a str),
    Floating(f64),
}
#[derive(Clone, Copy)]
struct Key<'a> {
    owner: Owner,
    number: Number<'a>,
}

fn observe(value: &Atom, missing: bool) -> Result<Key<'_>, CoreError> {
    let key = match value {
        Atom::NativeMissing { owner } if missing => Key {
            owner: Owner::Missing(*owner),
            number: Number::Missing,
        },
        Atom::Integer { value } => {
            integer(value)?;
            Key {
                owner: Owner::UnboundedInteger,
                number: Number::Integer(value),
            }
        }
        Atom::Boolean { value } => Key {
            owner: Owner::Boolean,
            number: Number::Integer(if *value { "1" } else { "0" }),
        },
        Atom::Floating { bits } => Key {
            owner: Owner::Binary64,
            number: Number::Floating(f64::from_bits(*bits)),
        },
        Atom::FixedInteger {
            signed,
            width,
            value,
        } => {
            integer(value)?;
            if !matches!(width, 1 | 2 | 4 | 8) {
                return Err(invalid());
            }
            let parsed = value.parse::<i128>().map_err(|_| invalid())?;
            let bits = u32::from(*width) * 8;
            let (min, max) = if *signed {
                (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
            } else {
                (0, (1_i128 << bits) - 1)
            };
            if !(min..=max).contains(&parsed) {
                return Err(invalid());
            }
            Key {
                owner: Owner::Integer {
                    signed: *signed,
                    width: *width,
                },
                number: Number::Integer(value),
            }
        }
        Atom::FixedFloating { width, bits } => {
            let value = match width {
                2 => half::f16::from_bits(u16::try_from(*bits).map_err(|_| invalid())?).to_f64(),
                4 => f64::from(f32::from_bits(u32::try_from(*bits).map_err(|_| invalid())?)),
                8 => f64::from_bits(*bits),
                _ => return Err(invalid()),
            };
            Key {
                owner: Owner::Floating { width: *width },
                number: Number::Floating(value),
            }
        }
        _ => return Err(invalid()),
    };
    if !missing && matches!(key.number,Number::Floating(value) if value.is_nan()) {
        return Err(invalid());
    }
    Ok(key)
}
fn integer_dtype(key: Key<'_>) -> Option<(bool, u8)> {
    match key.owner {
        Owner::Integer { signed, width } => Some((signed, width)),
        Owner::Boolean => Some((false, 0)),
        Owner::UnboundedInteger => {
            let Number::Integer(value) = key.number else {
                unreachable!()
            };
            if value.parse::<i64>().is_ok() {
                Some((true, 8))
            } else if value.parse::<u64>().is_ok() {
                Some((false, 8))
            } else {
                None
            }
        }
        _ => None,
    }
}
fn as_double(key: Key<'_>) -> Result<f64, Error> {
    match key.number {
        Number::Missing => unreachable!("missing owners stop before numeric conversion"),
        Number::Floating(value) => Ok(value),
        Number::Integer(value) => {
            let value = value
                .parse::<f64>()
                .map_err(|_| Error::IntegerToFloatOverflow)?;
            if value.is_infinite() {
                Err(Error::IntegerToFloatOverflow)
            } else {
                Ok(value)
            }
        }
    }
}
fn at_width(
    key: Key<'_>,
    width: u8,
    overflow: Option<Overflow>,
    stage: Stage,
    diagnostics: &mut Diagnostics,
) -> Result<f64, Error> {
    let value = as_double(key)?;
    let narrowed = match width {
        2 => half::f16::from_f64(value).to_f64(),
        4 => f64::from(value as f32),
        8 => value,
        _ => unreachable!(),
    };
    if value.is_finite() && narrowed.is_infinite() {
        diagnostics.overflow(overflow, stage)?;
    }
    Ok(narrowed)
}
fn float_width(owner: Owner) -> u8 {
    match owner {
        Owner::Floating { width } => width,
        _ => 8,
    }
}
fn fixed_width_owner(owner: Owner) -> bool {
    matches!(owner, Owner::Integer { .. } | Owner::Floating { .. })
}
fn mixed_width(integer: Owner, float: Owner, semantics: Semantics) -> u8 {
    let floating = float_width(float);
    match integer {
        Owner::Boolean => floating,
        Owner::UnboundedInteger if weak(semantics) => floating,
        Owner::UnboundedInteger => 8,
        Owner::Integer { width, .. } => match width {
            1 => floating,
            2 => floating.max(4),
            4 | 8 => 8,
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}
fn compare(
    a: Key<'_>,
    b: Key<'_>,
    semantics: Semantics,
    overflow: Option<Overflow>,
    stage: Stage,
    diagnostics: &mut Diagnostics,
) -> Result<Option<Ordering>, Error> {
    if semantics == Semantics::WeakFixedIntegerPairs {
        for (fixed, unbounded, reversed) in [(a, b, true), (b, a, false)] {
            if let (
                Owner::Integer { signed, width },
                Owner::UnboundedInteger,
                Number::Integer(value),
            ) = (fixed.owner, unbounded.owner, unbounded.number)
            {
                let intermediate_max = if !signed && width >= 4 {
                    i128::from(u64::MAX)
                } else {
                    i128::from(i64::MAX)
                };
                if value
                    .parse::<i128>()
                    .map_or(true, |v| v < i128::from(i64::MIN) || v > intermediate_max)
                {
                    diagnostics.destination = Some(IntegerDestination { signed, width });
                    diagnostics.reverse_failure = reversed;
                    return Err(Error::IntegerIntermediateOverflow);
                }
                let bits = u32::from(width) * 8;
                let (min, max) = if signed {
                    (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
                } else {
                    (0, (1_i128 << bits) - 1)
                };
                if value.parse::<i128>().map_or(true, |v| v < min || v > max) {
                    diagnostics.destination = Some(IntegerDestination { signed, width });
                    diagnostics.reverse_failure = reversed;
                    return Err(Error::IntegerOutOfBounds);
                }
            }
        }
    }

    match (a.number, b.number) {
        (Number::Missing, _) | (_, Number::Missing) => Ok(None),
        (Number::Integer(av), Number::Integer(bv)) => {
            if semantics == Semantics::CommonType
                && (fixed_width_owner(a.owner) || fixed_width_owner(b.owner))
                && let (Some((signed_a, width_a)), Some((signed_b, width_b))) =
                    (integer_dtype(a), integer_dtype(b))
                && signed_a != signed_b
                && ((!signed_a && width_a == 8 && width_b > 0)
                    || (!signed_b && width_b == 8 && width_a > 0))
            {
                return Ok(as_double(a)?.partial_cmp(&as_double(b)?));
            }
            Ok(Some(compare_integer(av, bv)))
        }
        (Number::Floating(_), Number::Integer(_)) => {
            compare(b, a, semantics, overflow, stage, diagnostics)
                .map(|order| order.map(Ordering::reverse))
        }
        (Number::Integer(integer), Number::Floating(float)) => {
            // Legacy object fallback converts the floating owner to its Python
            // value when a Python integer cannot be represented by an integer dtype.
            if !fixed_width_owner(a.owner) && !fixed_width_owner(b.owner)
                || matches!(a.owner, Owner::UnboundedInteger)
                    && !weak(semantics)
                    && integer_dtype(a).is_none()
            {
                return Ok(if float.is_nan() {
                    None
                } else {
                    Some(integer_float(integer, float))
                });
            }
            let width = mixed_width(a.owner, b.owner, semantics);
            Ok(
                at_width(a, width, overflow, stage, diagnostics)?.partial_cmp(&at_width(
                    b,
                    width,
                    overflow,
                    stage,
                    diagnostics,
                )?),
            )
        }
        (Number::Floating(_), Number::Floating(_)) => {
            let width = match (a.owner, b.owner) {
                (Owner::Floating { width: a }, Owner::Floating { width: b }) => a.max(b),
                (Owner::Floating { width }, Owner::Binary64)
                | (Owner::Binary64, Owner::Floating { width })
                    if weak(semantics) =>
                {
                    width
                }
                _ => 8,
            };
            Ok(
                at_width(a, width, overflow, stage, diagnostics)?.partial_cmp(&at_width(
                    b,
                    width,
                    overflow,
                    stage,
                    diagnostics,
                )?),
            )
        }
    }
}

/// One reached rich comparison during the native monotonic tuple probe.
pub(super) fn probe_comparison(
    a: &Atom,
    b: &Atom,
    semantics: Semantics,
    overflow: Option<Overflow>,
    diagnostics: &mut Diagnostics,
) -> Result<Option<Ordering>, Error> {
    compare(
        observe(a, true).expect("validated scalar owner"),
        observe(b, true).expect("validated scalar owner"),
        semantics,
        overflow,
        Stage::Probe,
        diagnostics,
    )
}

pub(super) fn sort_comparison(
    a: &Atom,
    b: &Atom,
    a_missing: bool,
    b_missing: bool,
    semantics: Semantics,
    overflow: Option<Overflow>,
    diagnostics: &mut Diagnostics,
) -> Result<Ordering, Error> {
    // The caller has already validated the closed facts through order().
    let a = observe(a, a_missing).expect("validated scalar owner");
    let b = observe(b, b_missing).expect("validated scalar owner");
    if compare(a, b, semantics, overflow, Stage::MultiProbe, diagnostics)? == Some(Ordering::Less) {
        return Ok(Ordering::Less);
    }
    Ok(
        if compare(a, b, semantics, overflow, Stage::MultiProbe, diagnostics)?
            == Some(Ordering::Greater)
        {
            Ordering::Greater
        } else {
            Ordering::Equal
        },
    )
}

fn probe(
    keys: &[Key<'_>],
    semantics: Semantics,
    overflow: Option<Overflow>,
    diagnostics: &mut Diagnostics,
) -> Result<bool, (Error, usize, usize)> {
    if keys.first().is_some_and(|key| {
        matches!(
            key.owner,
            Owner::Missing(NativeMissingOwner::Ambiguous | NativeMissingOwner::Temporal)
        ) || matches!(key.number,Number::Floating(value) if value.is_nan())
    }) {
        return Ok(false);
    }
    let (mut increasing, mut decreasing) = (true, true);
    for current in 1..keys.len() {
        let previous = current - 1;
        let mut relation = || {
            compare(
                keys[current],
                keys[previous],
                semantics,
                overflow,
                Stage::Probe,
                diagnostics,
            )
            .map_err(|reason| (reason, current, previous))
        };
        if relation()? == Some(Ordering::Less) {
            increasing = false;
        } else if relation()? == Some(Ordering::Greater) {
            decreasing = false;
        } else if relation()? != Some(Ordering::Equal) {
            increasing = false;
            decreasing = false;
        }
        if !increasing && !decreasing {
            break;
        }
    }
    Ok(increasing)
}
pub(super) fn monotonic(
    values: &[Atom],
    missing: &[bool],
    positions: &[usize],
    semantics: Semantics,
    overflow: Option<Overflow>,
    diagnostics: &mut Diagnostics,
) -> Result<Result<bool, (Error, usize, usize)>, CoreError> {
    if values.len() != missing.len() {
        return Err(invalid());
    }
    let keys = positions
        .iter()
        .map(|&i| {
            observe(
                values.get(i).ok_or_else(invalid)?,
                *missing.get(i).ok_or_else(invalid)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(probe(&keys, semantics, overflow, diagnostics)
        .map_err(|(reason, left, right)| (reason, positions[left], positions[right])))
}

pub(super) enum OrderStage {
    Rows,
    Level { probe_first: bool },
}

pub(super) fn order(
    values: &[Atom],
    missing: &[bool],
    profile: Profile,
    semantics: Semantics,
    overflow: Option<Overflow>,
    diagnostics: &mut Diagnostics,
    stage: OrderStage,
) -> Result<Outcome, CoreError> {
    let (probe_first, sort_stage) = match stage {
        OrderStage::Rows => (true, Stage::Sort),
        OrderStage::Level { probe_first } => (probe_first, Stage::LevelSort),
    };
    if values.len() != missing.len() {
        return Err(invalid());
    }
    let keys = values
        .iter()
        .zip(missing)
        .map(|(value, absent)| observe(value, *absent))
        .collect::<Result<Vec<_>, _>>()?;
    if profile == Profile::Unbound {
        return Ok(Outcome::NeedsProfile);
    }
    if probe_first {
        match probe(&keys, semantics, overflow, diagnostics) {
            Ok(true) => return Ok(sorted((0..values.len()).collect(), true)),
            Ok(false) => {}
            Err((reason, left, right)) => {
                return Ok(Outcome::Refused {
                    reason,
                    left,
                    right,
                    levels: Vec::new(),
                    overflow_warnings: Vec::new(),
                    overflow_prints: 0,
                    integer_destination: None,
                });
            }
        }
    }
    let present: Vec<_> = missing
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (!m).then_some(i))
        .collect();
    let ordered = object_sort::argsort(present.len(), |left, right| {
        let (left, right) = (present[left], present[right]);
        // The native object comparator performs LT, then GT if LT is false.
        let first = compare(
            keys[left],
            keys[right],
            semantics,
            overflow,
            sort_stage,
            diagnostics,
        )
        .map_err(|reason| (reason, left, right))?;
        if first == Some(Ordering::Less) {
            return Ok(true);
        }
        let _greater = compare(
            keys[left],
            keys[right],
            semantics,
            overflow,
            sort_stage,
            diagnostics,
        )
        .map_err(|reason| (reason, left, right))?
            == Some(Ordering::Greater);
        Ok(false)
    });
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
            levels: Vec::new(),
            reason,
            left,
            right,

            overflow_warnings: Vec::new(),
            overflow_prints: 0,
            integer_destination: None,
        }),
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn warning_and_print_counts_do_not_silently_saturate() {
        let mut warnings = Diagnostics {
            warnings: vec![NumericOverflowWarnings {
                stage: Stage::Probe,
                count: u64::MAX,
            }],
            ..Default::default()
        };
        assert!(
            warnings
                .overflow(Some(Overflow::Warn), Stage::Probe)
                .is_err()
        );
        assert!(warnings.count_overflow);
        let mut prints = Diagnostics {
            prints: u64::MAX,
            ..Default::default()
        };
        assert!(prints.overflow(Some(Overflow::Print), Stage::Sort).is_err());
        assert!(prints.count_overflow);
    }
}
