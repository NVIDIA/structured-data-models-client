// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporal levels preserve the native early guard and expanded-value stable passes.
use super::{
    Complex, Facts, MissingLast, ResultOrder, Values, atoms, invalid, numeric_objects,
    object_stable, order_sampled, sorted, temporal,
};
use sdm_contract::CoreError;
use sdm_contract::tabular::TabularOrderError as Error;
use std::cmp::Ordering;

type Failure = (Error, usize, usize, usize);
fn refused((reason, level, left, right): Failure) -> ResultOrder {
    ResultOrder::Refused {
        reason,
        levels: vec![level],
        left,
        right,
    }
}
fn is_type_error(reason: Error) -> bool {
    !matches!(
        reason,
        Error::IntegerToFloatOverflow | Error::NativeEffectsPending | Error::TemporalYearOutOfRange
    )
}
fn code_order(codes: &[Vec<i64>], rows: usize) -> bool {
    (1..rows).all(|row| {
        codes
            .iter()
            .map(|c| c[row - 1].cmp(&c[row]))
            .find(|c| *c != Ordering::Equal)
            != Some(Ordering::Greater)
    })
}
fn real(bits: u64, width: u8) -> f64 {
    if width == 8 {
        f64::from_bits(bits)
    } else {
        f64::from(f32::from_bits(bits as u32))
    }
}

fn compare(level: &Facts, left: usize, right: usize) -> Result<Ordering, Error> {
    if level.missing[left] || level.missing[right] {
        return Ok(level.missing[left].cmp(&level.missing[right]));
    }
    Ok(match &level.values {
        Values::Signed { values, .. }
        | Values::Temporal { values }
        | Values::Categorical { codes: values, .. } => values[left].cmp(&values[right]),
        Values::Unsigned { values, .. } => values[left].cmp(&values[right]),
        Values::Floating { bits, width } => MissingLast(real(bits[left], *width))
            .partial_cmp(&MissingLast(real(bits[right], *width)))
            .unwrap(),
        Values::Complex {
            real_bits,
            imag_bits,
            width,
        } => {
            let key = |i: usize| {
                MissingLast(Complex {
                    real: real(real_bits[i], *width),
                    imag: real(imag_bits[i], *width),
                })
            };
            key(left).partial_cmp(&key(right)).unwrap()
        }
        Values::Unicode { values } => values[left].cmp(&values[right]),
        Values::Object { values } => atoms::compare_atom(&values[left], &values[right])
            .ok_or_else(|| atoms::failure_reason(&values[left], &values[right]))?,
        Values::ScalarObject { values, semantics } => numeric_objects::sort_comparison(
            &values[left],
            &values[right],
            false,
            false,
            *semantics,
        )?,
        Values::TemporalObject { values } => {
            temporal::sort_comparison(&values[left], &values[right])?
        }
        Values::Multi { .. } => unreachable!("native index levels are single indexes"),
    })
}

fn tuple_monotonic(levels: &[Facts], codes: &[Vec<i64>], rows: usize) -> bool {
    (1..rows).all(|row| {
        for (level, code) in levels.iter().zip(codes) {
            let (left, right) = (code[row - 1] as usize, code[row] as usize);
            // Tuple equality first recognizes the very same retained owner.
            if left == right {
                continue;
            }
            let order = match &level.values {
                Values::TemporalObject { values } => {
                    temporal::rank_comparison(&values[left], &values[right])
                }
                _ => compare(level, left, right).ok(),
            };
            if order != Some(Ordering::Equal) {
                return order == Some(Ordering::Less);
            }
        }
        true
    })
}

fn outer_monotonic(
    levels: &[Facts],
    codes: &[Vec<i64>],
    rows: usize,
    monotonic: &[bool],
    ranks: Option<&[Vec<usize>]>,
) -> Result<bool, Failure> {
    if codes.iter().flatten().any(|c| *c == -1) {
        return Ok(false);
    }
    if monotonic.iter().all(|v| *v) {
        return Ok(if let Some(ranks) = ranks {
            (1..rows).all(|row| {
                ranks
                    .iter()
                    .map(|rank| rank[row - 1].cmp(&rank[row]))
                    .find(|c| *c != Ordering::Equal)
                    != Some(Ordering::Greater)
            })
        } else {
            code_order(codes, rows)
        });
    }
    let mut positions: Vec<_> = (0..rows).collect();
    for level_no in (0..levels.len()).rev() {
        let answer = object_stable::sort(&mut positions, |a, b| {
            let (left, right) = (codes[level_no][a] as usize, codes[level_no][b] as usize);
            compare(&levels[level_no], left, right)
                .map_err(|reason| (reason, level_no, left, right))
        });
        if let Err(failure) = answer {
            if is_type_error(failure.0) {
                return Ok(tuple_monotonic(levels, codes, rows));
            }
            return Err(failure);
        }
    }
    Ok(positions.iter().enumerate().all(|(i, p)| i == *p))
}

pub(super) fn order(
    levels: &[Facts],
    codes: &[Vec<i64>],
    missing: &[bool],
) -> Result<ResultOrder, CoreError> {
    for (level, code) in levels.iter().zip(codes) {
        if matches!(level.values, Values::Multi { .. })
            || code
                .iter()
                .any(|c| *c < -1 || *c >= level.missing.len() as i64)
        {
            return Err(invalid());
        }
    }
    let mut original_monotonic = Vec::new();
    let mut cached = Vec::new();
    for level in levels {
        if let Values::TemporalObject { values } = &level.values {
            original_monotonic.push(temporal::monotonic(
                values,
                &level.missing,
                &(0..values.len()).collect::<Vec<_>>(),
            )?);
            cached.push(None);
        } else {
            let result = order_sampled(level)?;
            if matches!(result, ResultOrder::NeedsProfile) {
                return Ok(result);
            }
            original_monotonic.push(matches!(
                result,
                ResultOrder::Sorted {
                    already_ordered: true,
                    ..
                }
            ));
            cached.push(Some(result));
        }
    }
    // Native level normalization first permits an already-lexsorted index to
    // retain even unused, mutually incomparable level owners without sorting.
    if code_order(codes, missing.len()) {
        match outer_monotonic(levels, codes, missing.len(), &original_monotonic, None) {
            Ok(true) => return Ok(sorted((0..missing.len()).collect(), true)),
            Err(failure) => return Ok(refused(failure)),
            Ok(false) => {}
        }
    }
    let mut ranks = Vec::new();
    let mut normalized_monotonic = Vec::new();
    for (level_no, (level, code)) in levels.iter().zip(codes).enumerate() {
        let result = match cached[level_no].take() {
            Some(result) => result,
            None => order_sampled(level)?,
        };
        let permutation = match result {
            ResultOrder::NeedsProfile => return Ok(ResultOrder::NeedsProfile),
            ResultOrder::Sorted { positions, .. } => {
                normalized_monotonic.push(
                    if let Values::TemporalObject { values } = &level.values {
                        temporal::monotonic(values, &level.missing, &positions)?
                    } else {
                        true
                    },
                );
                positions
            }
            ResultOrder::Refused {
                reason,
                left,
                right,
                levels: mut path,
            } => {
                if !is_type_error(reason) {
                    path.insert(0, level_no);
                    return Ok(ResultOrder::Refused {
                        reason,
                        left,
                        right,
                        levels: path,
                    });
                }
                normalized_monotonic.push(false);
                (0..level.missing.len()).collect()
            }
        };
        let mut rank = vec![0; level.missing.len()];
        for (at, position) in permutation.into_iter().enumerate() {
            rank[position] = at;
        }
        ranks.push(
            code.iter()
                .map(|c| {
                    if *c == -1 {
                        usize::MAX
                    } else {
                        rank[*c as usize]
                    }
                })
                .collect::<Vec<_>>(),
        );
    }
    match outer_monotonic(
        levels,
        codes,
        missing.len(),
        &normalized_monotonic,
        Some(&ranks),
    ) {
        Ok(true) => return Ok(sorted((0..missing.len()).collect(), true)),
        Err(failure) => return Ok(refused(failure)),
        Ok(false) => {}
    }
    let mut positions: Vec<_> = (0..missing.len()).collect();
    positions.sort_by(|a, b| {
        ranks
            .iter()
            .map(|rank| rank[*a].cmp(&rank[*b]))
            .find(|c| *c != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    });
    Ok(sorted(positions, false))
}
