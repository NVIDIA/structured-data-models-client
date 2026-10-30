// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed scalar-owner comparisons; producer promotion differs from sort policy.
use super::atoms::{compare_integer, integer, integer_float};
use super::{invalid, sorted};
use sdm_contract::CoreError;
use sdm_contract::tabular::{
    NativeMissingOwner, NumpyScalarSemantics as Semantics, OrderedSampledTabularRows as Outcome,
    TabularIndexScalar as Atom, TabularOrderError as Error, TabularSortProfile as Profile,
};
use std::cmp::Ordering;

#[path = "object_sort.rs"]
mod object_sort;

#[derive(Clone, Copy)]
enum Owner {
    Missing(NativeMissingOwner),
    PythonInteger,
    PythonFloat,
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
                owner: Owner::PythonInteger,
                number: Number::Integer(value),
            }
        }
        Atom::Boolean { value } => Key {
            owner: Owner::Boolean,
            number: Number::Integer(if *value { "1" } else { "0" }),
        },
        Atom::Floating { bits } => Key {
            owner: Owner::PythonFloat,
            number: Number::Floating(f64::from_bits(*bits)),
        },
        Atom::NumpyInteger {
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
        Atom::NumpyFloating { width, bits } => {
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
        Owner::PythonInteger => {
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
fn at_width(key: Key<'_>, width: u8) -> Result<f64, Error> {
    let value = as_double(key)?;
    let narrowed = match width {
        2 => half::f16::from_f64(value).to_f64(),
        4 => f64::from(value as f32),
        8 => value,
        _ => unreachable!(),
    };
    if value.is_finite() && narrowed.is_infinite() {
        Err(Error::NativeEffectsPending)
    } else {
        Ok(narrowed)
    }
}
fn float_width(owner: Owner) -> u8 {
    match owner {
        Owner::Floating { width } => width,
        _ => 8,
    }
}
fn numpy_owner(owner: Owner) -> bool {
    matches!(owner, Owner::Integer { .. } | Owner::Floating { .. })
}
fn mixed_width(integer: Owner, float: Owner, semantics: Semantics) -> u8 {
    let floating = float_width(float);
    match integer {
        Owner::Boolean => floating,
        Owner::PythonInteger if semantics == Semantics::Numpy25Weak => floating,
        Owner::PythonInteger => 8,
        Owner::Integer { width, .. } => match width {
            1 => floating,
            2 => floating.max(4),
            4 | 8 => 8,
            _ => unreachable!(),
        },
        _ => unreachable!(),
    }
}
fn compare(a: Key<'_>, b: Key<'_>, semantics: Semantics) -> Result<Option<Ordering>, Error> {
    match (a.number, b.number) {
        (Number::Missing, _) | (_, Number::Missing) => Ok(None),
        (Number::Integer(av), Number::Integer(bv)) => {
            if semantics == Semantics::Numpy123Legacy
                && (numpy_owner(a.owner) || numpy_owner(b.owner))
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
            compare(b, a, semantics).map(|order| order.map(Ordering::reverse))
        }
        (Number::Integer(integer), Number::Floating(float)) => {
            // Legacy object fallback converts the floating owner to its Python
            // value when a Python integer cannot be represented by an integer dtype.
            if !numpy_owner(a.owner) && !numpy_owner(b.owner)
                || matches!(a.owner, Owner::PythonInteger)
                    && semantics != Semantics::Numpy25Weak
                    && integer_dtype(a).is_none()
            {
                return Ok(if float.is_nan() {
                    None
                } else {
                    Some(integer_float(integer, float))
                });
            }
            let width = mixed_width(a.owner, b.owner, semantics);
            Ok(at_width(a, width)?.partial_cmp(&at_width(b, width)?))
        }
        (Number::Floating(_), Number::Floating(_)) => {
            let width = match (a.owner, b.owner) {
                (Owner::Floating { width: a }, Owner::Floating { width: b }) => a.max(b),
                (Owner::Floating { width }, Owner::PythonFloat)
                | (Owner::PythonFloat, Owner::Floating { width })
                    if semantics == Semantics::Numpy25Weak =>
                {
                    width
                }
                _ => 8,
            };
            Ok(at_width(a, width)?.partial_cmp(&at_width(b, width)?))
        }
    }
}

pub(super) fn order(
    values: &[Atom],
    missing: &[bool],
    profile: Profile,
    semantics: Semantics,
) -> Result<Outcome, CoreError> {
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
    let mut increasing = true;
    let mut decreasing = true;
    for current in 1..keys.len() {
        let previous = current - 1;
        let mut probe = || -> Result<(), Error> {
            if compare(keys[current], keys[previous], semantics)? == Some(Ordering::Less) {
                increasing = false;
            } else if compare(keys[current], keys[previous], semantics)? == Some(Ordering::Greater)
            {
                decreasing = false;
            } else if compare(keys[current], keys[previous], semantics)? != Some(Ordering::Equal) {
                increasing = false;
                decreasing = false;
            }
            Ok(())
        };
        if let Err(reason) = probe() {
            return Ok(Outcome::Refused {
                reason,
                left: current,
                right: previous,
            });
        }
        if !increasing && !decreasing {
            break;
        }
    }
    if keys.len() == 1
        && (matches!(
            keys[0].owner,
            Owner::Missing(NativeMissingOwner::Ambiguous | NativeMissingOwner::Temporal)
        ) || matches!(keys[0].number,Number::Floating(value) if value.is_nan()))
    {
        increasing = false;
    }
    if increasing {
        return Ok(sorted((0..values.len()).collect(), true));
    }
    let present: Vec<_> = missing
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (!m).then_some(i))
        .collect();
    let ordered = object_sort::argsort(present.len(), |left, right| {
        let (left, right) = (present[left], present[right]);
        // The native object comparator performs LT, then GT if LT is false.
        let first =
            compare(keys[left], keys[right], semantics).map_err(|reason| (reason, left, right))?;
        if first == Some(Ordering::Less) {
            return Ok(true);
        }
        let _greater = compare(keys[left], keys[right], semantics)
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
            reason,
            left,
            right,
        }),
    }
}
