// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selected-subset ordering. Values stay separate from original row identities.

use super::scalar::scalar_argsort;
use super::{SimdArgsortProfile, SimdKey, simd_argsort};
use sdm_contract::tabular::{
    OrderedSampledTabularRows as ResultOrder, TabularIndexFacts as Facts,
    TabularIndexScalar as Atom, TabularIndexValues as Values, TabularSortProfile as Profile,
};
use sdm_contract::{CoreError, Sentence};
use std::cell::Cell;
use std::cmp::Ordering;

mod atoms;
use atoms::{compare_atom, failure_reason, pure_total_family, validate_atom};

fn invalid() -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "sampled index facts are inconsistent",
    )
}

fn sorted(positions: Vec<usize>, already_ordered: bool) -> ResultOrder {
    ResultOrder::Sorted {
        positions,
        already_ordered,
    }
}

fn scalar<T: PartialOrd>(values: &[T], profile: Profile) -> Vec<usize> {
    if profile == Profile::Stable {
        let mut positions: Vec<_> = (0..values.len()).collect();
        positions.sort_by(|a, b| {
            values[*a]
                .partial_cmp(&values[*b])
                .unwrap_or(Ordering::Equal)
        });
        positions
    } else {
        scalar_argsort(values)
    }
}

#[derive(Clone, Copy)]
struct MissingLast<T>(T);
impl<T: PartialOrd> PartialEq for MissingLast<T> {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}
impl<T: PartialOrd> PartialOrd for MissingLast<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0).or_else(|| {
            Some(
                self.0
                    .partial_cmp(&self.0)
                    .is_none()
                    .cmp(&other.0.partial_cmp(&other.0).is_none()),
            )
        })
    }
}

fn numeric<T: SimdKey>(values: &[T], profile: Profile) -> Vec<usize> {
    if values.iter().any(|v| v.partial_cmp(v).is_none()) {
        let ordered: Vec<_> = values.iter().copied().map(MissingLast).collect();
        if matches!(profile, Profile::Avx2Gnu14 | Profile::Avx512Gnu14) {
            let mut positions: Vec<_> = (0..values.len()).collect();
            super::gnu::sort(&ordered, &mut positions);
            return positions;
        }
        return scalar(&ordered, profile);
    }
    match profile {
        Profile::Unbound => unreachable!("unbound ordering resolves before kernel dispatch"),
        Profile::Scalar => scalar(values, profile),
        Profile::Stable => {
            let mut positions: Vec<_> = (0..values.len()).collect();
            positions.sort_by(|a, b| match values[*a].partial_cmp(&values[*b]) {
                Some(order) => order,
                None => values[*a]
                    .partial_cmp(&values[*a])
                    .is_none()
                    .cmp(&values[*b].partial_cmp(&values[*b]).is_none()),
            });
            positions
        }
        Profile::Avx2Gnu14 => simd_argsort(values, SimdArgsortProfile::Avx2Gnu14),
        Profile::Avx512Gnu14 => simd_argsort(values, SimdArgsortProfile::Avx512Gnu14),
    }
}

fn order<T: PartialOrd + Clone>(
    values: &[T],
    missing: &[bool],
    profile: Profile,
    sorter: impl FnOnce(&[T]) -> Vec<usize>,
) -> Result<ResultOrder, CoreError> {
    if values.len() != missing.len() {
        return Err(invalid());
    }
    if !missing.iter().any(|v| *v)
        && values.iter().all(|v| v.partial_cmp(v).is_some())
        && values.windows(2).all(|w| w[0] <= w[1])
    {
        return Ok(sorted((0..values.len()).collect(), true));
    }
    let present: Vec<_> = missing
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (!m).then_some(i))
        .collect();
    let keys: Vec<_> = present.iter().map(|i| values[*i].clone()).collect();
    let ordered = if profile == Profile::Unbound {
        // A provisional stable ordering is observable only when every retained
        // key is distinct under this admitted pure, total comparator.
        let keys: Vec<_> = keys.iter().map(MissingLast).collect();
        let mut positions: Vec<_> = (0..keys.len()).collect();
        positions.sort_by(|a, b| {
            keys[*a]
                .partial_cmp(&keys[*b])
                .expect("pure total comparator")
        });
        if positions
            .windows(2)
            .any(|pair| keys[pair[0]] == keys[pair[1]])
        {
            return Ok(ResultOrder::NeedsProfile);
        }
        positions
    } else {
        sorter(&keys)
    };
    let mut positions: Vec<_> = ordered.into_iter().map(|i| present[i]).collect();
    positions.extend(
        missing
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.then_some(i)),
    );
    Ok(sorted(positions, false))
}

#[derive(Clone)]
struct Object<'a> {
    value: &'a Atom,
    position: usize,
    failure: &'a Cell<Option<(usize, usize)>>,
}
impl PartialEq for Object<'_> {
    fn eq(&self, other: &Self) -> bool {
        compare_atom(self.value, other.value) == Some(Ordering::Equal)
    }
}
impl PartialOrd for Object<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let result = compare_atom(self.value, other.value);
        if result.is_none() && self.failure.get().is_none() {
            self.failure.set(Some((self.position, other.position)));
        }
        result
    }
}

fn multi(levels: &[Facts], codes: &[Vec<i64>], missing: &[bool]) -> Result<ResultOrder, CoreError> {
    if levels.len() != codes.len() || codes.iter().any(|c| c.len() != missing.len()) {
        return Err(invalid());
    }
    let mut ranks = Vec::new();
    let mut incomparable_levels = Vec::new();
    let mut needs_profile = false;
    for (level, code) in levels.iter().zip(codes) {
        let n = level.missing.len();
        let permutation = match order_sampled(level)? {
            ResultOrder::Sorted { positions, .. } => {
                incomparable_levels.push(false);
                positions
            }
            ResultOrder::NeedsProfile => {
                needs_profile = true;
                incomparable_levels.push(false);
                (0..n).collect()
            }
            ResultOrder::Refused { .. } => {
                incomparable_levels.push(true);
                (0..n).collect()
            }
        };
        let mut rank = vec![0; n];
        for (at, original) in permutation.into_iter().enumerate() {
            rank[original] = at;
        }
        let mut row_ranks = Vec::with_capacity(code.len());
        for c in code {
            row_ranks.push(if *c == -1 {
                usize::MAX
            } else {
                *rank
                    .get(usize::try_from(*c).map_err(|_| invalid())?)
                    .ok_or_else(invalid)?
            });
        }
        ranks.push(row_ranks);
    }
    if needs_profile {
        return Ok(ResultOrder::NeedsProfile);
    }
    let mut positions: Vec<_> = (0..missing.len()).collect();
    positions.sort_by(|a, b| {
        ranks
            .iter()
            .map(|level| level[*a].cmp(&level[*b]))
            .find(|v| *v != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    });
    let already_ordered = !codes.iter().flatten().any(|c| *c == -1)
        && (1..missing.len()).all(|row| {
            for (level_no, level) in levels.iter().enumerate() {
                let left = codes[level_no][row - 1] as usize;
                let right = codes[level_no][row] as usize;
                if left == right {
                    continue;
                }
                let compared = if incomparable_levels[level_no] {
                    match &level.values {
                        Values::Object { values } => compare_atom(&values[left], &values[right]),
                        _ => None,
                    }
                } else {
                    Some(ranks[level_no][row - 1].cmp(&ranks[level_no][row]))
                };
                if compared != Some(Ordering::Equal) {
                    return compared == Some(Ordering::Less);
                }
            }
            true
        });
    Ok(sorted(positions, already_ordered))
}

#[derive(Clone, Copy, PartialEq)]
struct Complex {
    real: f64,
    imag: f64,
}
impl PartialOrd for Complex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.real.partial_cmp(&other.real).and_then(|real| {
            if real == Ordering::Equal {
                self.imag.partial_cmp(&other.imag)
            } else {
                Some(real)
            }
        })
    }
}

pub fn order_sampled(facts: &Facts) -> Result<ResultOrder, CoreError> {
    let mask = &facts.missing;
    let profile = facts.profile;
    let result = match &facts.values {
        Values::Signed { values, width }
        | Values::Categorical {
            codes: values,
            width,
        } => match width {
            1 | 2 => order(values, mask, profile, |v| scalar(v, profile))?,
            4 => {
                let v: Vec<_> = values
                    .iter()
                    .map(|v| i32::try_from(*v).map_err(|_| invalid()))
                    .collect::<Result<_, _>>()?;
                order(&v, mask, profile, |v| numeric(v, profile))?
            }
            8 => order(values, mask, profile, |v| numeric(v, profile))?,
            _ => return Err(invalid()),
        },
        Values::Unsigned { values, width } => match width {
            1 | 2 => order(values, mask, profile, |v| scalar(v, profile))?,
            4 => {
                let v: Vec<_> = values
                    .iter()
                    .map(|v| u32::try_from(*v).map_err(|_| invalid()))
                    .collect::<Result<_, _>>()?;
                order(&v, mask, profile, |v| numeric(v, profile))?
            }
            8 => order(values, mask, profile, |v| numeric(v, profile))?,
            _ => return Err(invalid()),
        },
        Values::Floating { bits, width } => {
            // Half storage is carried as its exact float32 value by the host.
            if *width == 2 || *width == 4 {
                let v: Vec<_> = bits
                    .iter()
                    .map(|b| u32::try_from(*b).map(f32::from_bits).map_err(|_| invalid()))
                    .collect::<Result<_, _>>()?;
                order(&v, mask, profile, |v| {
                    if *width == 2 {
                        scalar(
                            &v.iter().copied().map(MissingLast).collect::<Vec<_>>(),
                            profile,
                        )
                    } else {
                        numeric(v, profile)
                    }
                })?
            } else if *width == 8 {
                let v: Vec<_> = bits.iter().map(|b| f64::from_bits(*b)).collect();
                order(&v, mask, profile, |v| numeric(v, profile))?
            } else {
                return Err(invalid());
            }
        }
        Values::Complex {
            real_bits,
            imag_bits,
            width,
        } => {
            if !matches!(width, 4 | 8)
                || real_bits.len() != mask.len()
                || imag_bits.len() != mask.len()
            {
                return Err(invalid());
            }
            let component = |bits: u64| -> Result<f64, CoreError> {
                match width {
                    4 => Ok(f64::from(f32::from_bits(
                        u32::try_from(bits).map_err(|_| invalid())?,
                    ))),
                    8 => Ok(f64::from_bits(bits)),
                    _ => Err(invalid()),
                }
            };
            let values = real_bits
                .iter()
                .zip(imag_bits)
                .zip(mask)
                .map(|((real, imag), missing)| {
                    let value = Complex {
                        real: component(*real)?,
                        imag: component(*imag)?,
                    };
                    if !missing && (value.real.is_nan() || value.imag.is_nan()) {
                        return Err(invalid());
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, CoreError>>()?;
            order(&values, mask, profile, |v| scalar(v, profile))?
        }
        Values::Temporal { values } => order(values, mask, profile, |v| scalar(v, profile))?,
        Values::Unicode { values } => {
            if values.iter().flatten().any(|point| *point > 0x10ffff) {
                return Err(invalid());
            }
            order(values, mask, profile, |v| scalar(v, profile))?
        }
        Values::Multi { levels, codes } => return multi(levels, codes, mask),
        Values::Object { values } => {
            if values.len() != mask.len() {
                return Err(invalid());
            }
            for (v, m) in values.iter().zip(mask) {
                if !m {
                    validate_atom(v)?;
                }
            }
            // Mixed decimal/floating comparisons require native context signals,
            // which this pure-fact operation does not yet represent.
            let present = || {
                values
                    .iter()
                    .zip(mask)
                    .filter_map(|(v, m)| (!m).then_some(v))
            };
            if present().any(|v| matches!(v, Atom::Decimal { .. }))
                && present().any(|v| matches!(v, Atom::Floating { .. }))
            {
                return Err(invalid());
            }
            if profile == Profile::Unbound
                && !pure_total_family(
                    values
                        .iter()
                        .zip(mask)
                        .filter_map(|(v, m)| (!m).then_some(v)),
                )
            {
                return Ok(ResultOrder::NeedsProfile);
            }
            let failed = Cell::new(None);
            let v: Vec<_> = values
                .iter()
                .enumerate()
                .map(|(position, value)| Object {
                    value,
                    position,
                    failure: &failed,
                })
                .collect();
            if profile == Profile::Unbound {
                return order(&v, mask, profile, |_| {
                    unreachable!("unbound order never invokes a kernel")
                });
            }
            // Monotonic probing suppresses incomparable errors in the native index.
            let monotonic = !mask.iter().any(|v| *v) && v.windows(2).all(|w| w[0] <= w[1]);
            failed.set(None);
            let positions = if monotonic {
                (0..v.len()).collect()
            } else {
                let present: Vec<_> = mask
                    .iter()
                    .enumerate()
                    .filter_map(|(i, m)| (!m).then_some(i))
                    .collect();
                let keys: Vec<_> = present.iter().map(|i| v[*i].clone()).collect();
                let mut positions: Vec<_> = scalar(&keys, profile)
                    .into_iter()
                    .map(|i| present[i])
                    .collect();
                positions.extend(mask.iter().enumerate().filter_map(|(i, m)| m.then_some(i)));
                positions
            };
            if let Some((left, right)) = failed.get() {
                return Ok(ResultOrder::Refused {
                    reason: failure_reason(&values[left], &values[right]),
                    left,
                    right,
                });
            }
            return Ok(sorted(positions, monotonic));
        }
    };
    Ok(result)
}
