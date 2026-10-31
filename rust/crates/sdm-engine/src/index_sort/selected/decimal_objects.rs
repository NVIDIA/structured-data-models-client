// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact decimal/binary64 comparisons with closed sticky signal observations.
use super::{atoms, invalid, numeric_objects::Diagnostics, object_sort, sorted};
use sdm_contract::{
    CoreError,
    tabular::{
        NativeMissingOwner, OrderedSampledTabularRows as Outcome, TabularIndexScalar as Atom,
        TabularOrderError as Error, TabularSortProfile as Profile,
    },
};
use std::cmp::Ordering;

#[derive(Clone, Copy)]
pub(super) struct Policy {
    pub float_trap: bool,
    pub invalid_trap: bool,
    pub catches_invalid: bool,
}
/// IEEE binary64 bounds this expansion by 1074 fractional binary places.
fn exact_float(value: f64) -> Atom {
    let bits = value.to_bits();
    let field = ((bits >> 52) & 0x7ff) as i32;
    let mantissa = if field == 0 {
        bits & ((1_u64 << 52) - 1)
    } else {
        (bits & ((1_u64 << 52) - 1)) | (1_u64 << 52)
    };
    let binary_exponent = if field == 0 { -1074 } else { field - 1023 - 52 };
    let mut digits: Vec<u8> = mantissa
        .to_string()
        .bytes()
        .rev()
        .map(|b| b - b'0')
        .collect();
    let multiplier = if binary_exponent < 0 { 5 } else { 2 };
    for _ in 0..binary_exponent.unsigned_abs() {
        let mut carry = 0_u8;
        for digit in &mut digits {
            let next = *digit * multiplier + carry;
            *digit = next % 10;
            carry = next / 10;
        }
        if carry != 0 {
            digits.push(carry);
        }
    }
    Atom::Decimal {
        negative: value.is_sign_negative(),
        digits: digits
            .into_iter()
            .rev()
            .map(|d| char::from(d + b'0'))
            .collect(),
        exponent: i64::from(binary_exponent.min(0)),
    }
}
fn keys<'a>(values: &'a [Atom], missing: &[bool]) -> Result<Vec<&'a Atom>, CoreError> {
    if values.len() != missing.len() {
        return Err(invalid());
    }
    values
        .iter()
        .zip(missing)
        .map(|(value, absent)| {
            match value {
                Atom::Floating { bits } => {
                    if f64::from_bits(*bits).is_nan() && !absent {
                        return Err(invalid());
                    }
                }
                Atom::NativeMissing { .. } if *absent => {}
                Atom::Decimal { .. } | Atom::Integer { .. } | Atom::Boolean { .. } => {
                    atoms::validate_atom(value)?
                }
                _ => return Err(invalid()),
            }
            Ok(value)
        })
        .collect()
}
fn relation(
    a: &Atom,
    b: &Atom,
    operator: Ordering,
    policy: Policy,
    diagnostics: &mut Diagnostics,
) -> Result<bool, Error> {
    let mixed = matches!(
        (a, b),
        (Atom::Decimal { .. }, Atom::Floating { .. })
            | (Atom::Floating { .. }, Atom::Decimal { .. })
    );
    if mixed {
        diagnostics.decimal_float();
        if operator != Ordering::Equal && policy.float_trap {
            return Err(Error::DecimalFloatOperation);
        }
        let (decimal, float, reverse) = if matches!(a, Atom::Decimal { .. }) {
            (a, b, false)
        } else {
            (b, a, true)
        };
        let Atom::Floating { bits } = float else {
            unreachable!()
        };
        let value = f64::from_bits(*bits);
        if value.is_nan() {
            if operator != Ordering::Equal {
                diagnostics.decimal_invalid();
                if policy.invalid_trap {
                    return Err(Error::DecimalInvalidOperation);
                }
            }
            return Ok(false);
        }
        let order = if value.is_finite() {
            let exact = diagnostics
                .decimal_values
                .entry(*bits)
                .or_insert_with(|| exact_float(value));
            atoms::compare_atom(decimal, exact).expect("exact decimal comparison")
        } else if value.is_sign_positive() {
            Ordering::Less
        } else {
            Ordering::Greater
        };
        return Ok(if reverse { order.reverse() } else { order } == operator);
    }
    if let (
        Atom::NativeMissing {
            owner: NativeMissingOwner::None,
        },
        Atom::NativeMissing {
            owner: NativeMissingOwner::None,
        },
    ) = (a, b)
    {
        return Ok(operator == Ordering::Equal);
    }
    if matches!(a, Atom::NativeMissing { .. }) || matches!(b, Atom::NativeMissing { .. }) {
        return Ok(false);
    }
    // Non-decimal scalar NaNs are unordered without Decimal context signals.
    // Keep this after the mixed-owner branch, whose traps have precedence.
    if [a, b]
        .iter()
        .any(|value| matches!(value,Atom::Floating {bits} if f64::from_bits(*bits).is_nan()))
    {
        return Ok(false);
    }
    Ok(atoms::compare_atom(a, b) == Some(operator))
}
fn caught(error: Error, policy: Policy) -> bool {
    error == Error::DecimalFloatOperation
        || error == Error::DecimalInvalidOperation && policy.catches_invalid
}
fn probe(
    keys: &[&Atom],
    policy: Policy,
    diagnostics: &mut Diagnostics,
) -> Result<bool, (Error, usize, usize)> {
    if let Some(first) = keys.first() {
        match relation(first, first, Ordering::Equal, policy, diagnostics) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(error) if caught(error, policy) => return Ok(false),
            Err(error) => return Err((error, 0, 0)),
        }
    }
    let (mut increasing, mut decreasing) = (true, true);
    for current in 1..keys.len() {
        let previous = current - 1;
        let result = (|| {
            if relation(
                &keys[current],
                &keys[previous],
                Ordering::Less,
                policy,
                diagnostics,
            )? {
                increasing = false;
            } else if relation(
                &keys[current],
                &keys[previous],
                Ordering::Greater,
                policy,
                diagnostics,
            )? {
                decreasing = false;
            } else if !relation(
                &keys[current],
                &keys[previous],
                Ordering::Equal,
                policy,
                diagnostics,
            )? {
                increasing = false;
                decreasing = false;
            }
            Ok::<_, Error>(())
        })();
        if let Err(error) = result {
            if caught(error, policy) {
                return Ok(false);
            }
            return Err((error, current, previous));
        }
        if !increasing && !decreasing {
            break;
        }
    }
    Ok(increasing)
}
fn refused((reason, left, right): (Error, usize, usize)) -> Outcome {
    Outcome::Refused {
        reason,
        left,
        right,
        levels: Vec::new(),
        numeric_diagnostics: Vec::new(),
        integer_destination: None,
    }
}
pub(super) fn order(
    values: &[Atom],
    missing: &[bool],
    profile: Profile,
    policy: Policy,
    diagnostics: &mut Diagnostics,
    probe_first: bool,
) -> Result<Outcome, CoreError> {
    let keys = keys(values, missing)?;
    if profile == Profile::Unbound {
        return Ok(Outcome::NeedsProfile);
    }
    if probe_first {
        match probe(&keys, policy, diagnostics) {
            Ok(true) => return Ok(sorted((0..values.len()).collect(), true)),
            Ok(false) => {}
            Err(error) => return Ok(refused(error)),
        }
    }
    let present: Vec<_> = missing
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (!m).then_some(i))
        .collect();
    let result = object_sort::argsort(present.len(), |left, right| {
        let (left, right) = (present[left], present[right]);
        if relation(
            &keys[left],
            &keys[right],
            Ordering::Less,
            policy,
            diagnostics,
        )
        .map_err(|reason| (reason, left, right))?
        {
            return Ok(true);
        }
        let _ = relation(
            &keys[left],
            &keys[right],
            Ordering::Greater,
            policy,
            diagnostics,
        )
        .map_err(|reason| (reason, left, right))?;
        Ok(false)
    });
    Ok(match result {
        Ok(order) => {
            let mut positions: Vec<_> = order.into_iter().map(|i| present[i]).collect();
            positions.extend(
                missing
                    .iter()
                    .enumerate()
                    .filter_map(|(i, m)| m.then_some(i)),
            );
            sorted(positions, false)
        }
        Err(error) => refused(error),
    })
}
pub(super) fn monotonic(
    values: &[Atom],
    missing: &[bool],
    positions: &[usize],
    policy: Policy,
    diagnostics: &mut Diagnostics,
) -> Result<Result<bool, (Error, usize, usize)>, CoreError> {
    let original = keys(values, missing)?;
    let ordered = positions
        .iter()
        .map(|&i| original.get(i).copied().ok_or_else(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(probe(&ordered, policy, diagnostics)
        .map_err(|(reason, left, right)| (reason, positions[left], positions[right])))
}
pub(super) fn comparison(
    a: &Atom,
    b: &Atom,
    operator: Ordering,
    policy: Policy,
    diagnostics: &mut Diagnostics,
) -> Result<bool, Error> {
    relation(a, b, operator, policy, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::tabular::NumericDiagnostic as Event;
    #[test]
    fn equality_signals_float_operation_without_raising_its_trap() {
        let decimal = Atom::Decimal {
            negative: false,
            digits: "11".into(),
            exponent: -1,
        };
        let float = Atom::Floating {
            bits: 1.1_f64.to_bits(),
        };
        let mut diagnostics = Diagnostics::default();
        let policy = Policy {
            float_trap: true,
            invalid_trap: true,
            catches_invalid: false,
        };
        assert!(!comparison(&decimal, &float, Ordering::Equal, policy, &mut diagnostics).unwrap());
        assert_eq!(diagnostics.events, vec![Event::DecimalFloatOperation]);
        assert_eq!(
            comparison(&decimal, &float, Ordering::Less, policy, &mut diagnostics),
            Err(Error::DecimalFloatOperation)
        );
        assert_eq!(diagnostics.events, vec![Event::DecimalFloatOperation]);
    }
    #[test]
    fn float_trap_precedes_invalid_signal_for_nan_ordering() {
        let decimal = Atom::Decimal {
            negative: false,
            digits: "1".into(),
            exponent: 0,
        };
        let nan = Atom::Floating {
            bits: f64::NAN.to_bits(),
        };
        let mut diagnostics = Diagnostics::default();
        assert_eq!(
            comparison(
                &decimal,
                &nan,
                Ordering::Less,
                Policy {
                    float_trap: true,
                    invalid_trap: true,
                    catches_invalid: false
                },
                &mut diagnostics
            ),
            Err(Error::DecimalFloatOperation)
        );
        assert_eq!(diagnostics.events, vec![Event::DecimalFloatOperation]);
    }
}
