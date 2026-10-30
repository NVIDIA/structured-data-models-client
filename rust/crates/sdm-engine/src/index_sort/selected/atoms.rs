// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure comparisons for admitted scalar label owners.
use super::invalid;
use sdm_contract::CoreError;
use sdm_contract::tabular::{TabularIndexScalar as Atom, TabularOrderError};
use std::cmp::Ordering;

fn integer(value: &str) -> Result<(&str, bool), CoreError> {
    let (digits, negative) = value
        .strip_prefix('-')
        .map_or((value, false), |v| (v, true));
    if digits.is_empty()
        || !digits.bytes().all(|c| c.is_ascii_digit())
        || digits.len() > 1 && digits.starts_with('0')
        || negative && digits == "0"
    {
        return Err(invalid());
    }
    Ok((digits, negative))
}
fn compare_integer(a: &str, b: &str) -> Ordering {
    let (a, na) = integer(a).expect("validated integer");
    let (b, nb) = integer(b).expect("validated integer");
    if na != nb {
        return if na {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let result = a.len().cmp(&b.len()).then_with(|| a.cmp(b));
    if na { result.reverse() } else { result }
}
fn integer_float(a: &str, b: f64) -> Ordering {
    if b == f64::INFINITY {
        return Ordering::Less;
    }
    if b == f64::NEG_INFINITY {
        return Ordering::Greater;
    }
    let whole = if b.trunc() == 0.0 {
        "0".into()
    } else {
        format!("{:.0}", b.trunc())
    };
    compare_integer(a, &whole)
        .then_with(|| 0.0_f64.partial_cmp(&b.fract()).expect("nonmissing number"))
}
pub(super) fn compare_atom(a: &Atom, b: &Atom) -> Option<Ordering> {
    match (a, b) {
        (Atom::Text { points: a }, Atom::Text { points: b }) => Some(a.cmp(b)),
        (Atom::Date { ordinal: a }, Atom::Date { ordinal: b }) => Some(a.cmp(b)),
        (
            Atom::DateTime {
                ordinal: a,
                micros: am,
                offset_micros: ao,
            },
            Atom::DateTime {
                ordinal: b,
                micros: bm,
                offset_micros: bo,
            },
        ) if ao.is_some() == bo.is_some() => {
            let left = i128::from(*a) * DAY_MICROS + i128::from(*am) - i128::from(ao.unwrap_or(0));
            let right = i128::from(*b) * DAY_MICROS + i128::from(*bm) - i128::from(bo.unwrap_or(0));
            Some(left.cmp(&right))
        }
        (
            Atom::Decimal { .. },
            Atom::Decimal { .. } | Atom::Integer { .. } | Atom::Boolean { .. },
        )
        | (Atom::Integer { .. } | Atom::Boolean { .. }, Atom::Decimal { .. }) => {
            Some(compare_decimal(decimal(a), decimal(b)))
        }
        (Atom::Floating { bits: a }, Atom::Floating { bits: b }) => {
            f64::from_bits(*a).partial_cmp(&f64::from_bits(*b))
        }
        (Atom::Floating { bits }, other @ (Atom::Integer { .. } | Atom::Boolean { .. })) => {
            Some(integer_float(&integer_atom(other), f64::from_bits(*bits)).reverse())
        }
        (other @ (Atom::Integer { .. } | Atom::Boolean { .. }), Atom::Floating { bits }) => {
            Some(integer_float(&integer_atom(other), f64::from_bits(*bits)))
        }
        (
            Atom::Integer { .. } | Atom::Boolean { .. },
            Atom::Integer { .. } | Atom::Boolean { .. },
        ) => Some(compare_integer(&integer_atom(a), &integer_atom(b))),
        _ => None,
    }
}

fn integer_atom(a: &Atom) -> String {
    match a {
        Atom::Integer { value } => value.clone(),
        Atom::Boolean { value } => if *value { "1" } else { "0" }.into(),
        _ => unreachable!("numeric integer atom"),
    }
}

const DAY_MICROS: i128 = 86_400_000_000;

struct Decimal<'a> {
    negative: bool,
    digits: &'a str,
    exponent: i64,
}
fn decimal(value: &Atom) -> Decimal<'_> {
    match value {
        Atom::Decimal {
            negative,
            digits,
            exponent,
        } => Decimal {
            negative: *negative,
            digits,
            exponent: *exponent,
        },
        Atom::Integer { value } => {
            let (digits, negative) = integer(value).expect("validated integer");
            Decimal {
                negative,
                digits,
                exponent: 0,
            }
        }
        Atom::Boolean { value } => Decimal {
            negative: false,
            digits: if *value { "1" } else { "0" },
            exponent: 0,
        },
        _ => unreachable!("admitted decimal comparison"),
    }
}
fn compare_decimal(a: Decimal<'_>, b: Decimal<'_>) -> Ordering {
    let az = a.digits == "0";
    let bz = b.digits == "0";
    if az && bz {
        return Ordering::Equal;
    }
    let an = a.negative && !az;
    let bn = b.negative && !bz;
    if an != bn {
        return if an {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let magnitude = if az || bz {
        bz.cmp(&az)
    } else {
        let am = i128::from(a.exponent) + a.digits.len() as i128;
        let bm = i128::from(b.exponent) + b.digits.len() as i128;
        am.cmp(&bm).then_with(|| {
            // Equal adjusted exponents bound padding by coefficient length,
            // never by the potentially enormous decimal exponent.
            a.digits
                .bytes()
                .chain(std::iter::repeat(b'0'))
                .zip(b.digits.bytes().chain(std::iter::repeat(b'0')))
                .take(a.digits.len().max(b.digits.len()))
                .map(|(a, b)| a.cmp(&b))
                .find(|v| *v != Ordering::Equal)
                .unwrap_or(Ordering::Equal)
        })
    };
    if an { magnitude.reverse() } else { magnitude }
}

pub(super) fn validate_atom(value: &Atom) -> Result<(), CoreError> {
    let valid = match value {
        Atom::Integer { value } => {
            integer(value)?;
            true
        }
        Atom::Floating { bits } => !f64::from_bits(*bits).is_nan(),
        Atom::Text { points } => points.iter().all(|p| *p <= 0x10ffff),
        Atom::Boolean { .. } | Atom::Complex { .. } => true,
        Atom::Decimal { digits, .. } => {
            !digits.is_empty()
                && digits.bytes().all(|b| b.is_ascii_digit())
                && (digits.len() == 1 || !digits.starts_with('0'))
        }
        Atom::Date { ordinal } => (1..=3_652_059).contains(ordinal),
        Atom::DateTime {
            ordinal,
            micros,
            offset_micros,
        } => {
            (1..=3_652_059).contains(ordinal)
                && i128::from(*micros) < DAY_MICROS
                && offset_micros.is_none_or(|offset| i128::from(offset).abs() < DAY_MICROS)
        }
    };
    if valid { Ok(()) } else { Err(invalid()) }
}

pub(super) fn failure_reason(a: &Atom, b: &Atom) -> TabularOrderError {
    match (a, b) {
        (Atom::Date { .. }, Atom::DateTime { .. }) | (Atom::DateTime { .. }, Atom::Date { .. }) => {
            TabularOrderError::DateTimeMismatch
        }
        (
            Atom::DateTime {
                offset_micros: a, ..
            },
            Atom::DateTime {
                offset_micros: b, ..
            },
        ) if a.is_some() != b.is_some() => TabularOrderError::NaiveAwareDatetime,
        _ => TabularOrderError::Incomparable,
    }
}
