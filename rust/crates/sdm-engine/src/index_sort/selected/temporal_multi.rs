// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporal levels preserve the native early guard and expanded-value stable passes.
use super::{
    Complex, Facts, MissingLast, ResultOrder, Values, atoms, invalid, numeric_objects,
    object_stable, order_inner, sorted, temporal,
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

        overflow_warnings: Vec::new(),
        overflow_prints: 0,
        integer_destination: None,
    }
}
fn is_type_error(reason: Error) -> bool {
    !matches!(
        reason,
        Error::IntegerToFloatOverflow
            | Error::IntegerIntermediateOverflow
            | Error::IntegerOutOfBounds
            | Error::FloatingOverflow
            | Error::NativeEffectsPending
            | Error::TemporalYearOutOfRange
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

fn compare(
    level: &Facts,
    left: usize,
    right: usize,
    diagnostics: &mut numeric_objects::Diagnostics,
) -> Result<Ordering, Error> {
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
        Values::ScalarObject {
            values,
            semantics,
            overflow,
        } => numeric_objects::sort_comparison(
            &values[left],
            &values[right],
            false,
            false,
            *semantics,
            *overflow,
            diagnostics,
        )?,
        Values::TemporalObject { values } => {
            temporal::sort_comparison(&values[left], &values[right])?
        }
        Values::Multi { .. } => unreachable!("native index levels are single indexes"),
    })
}

fn tuple_monotonic(
    levels: &[Facts],
    codes: &[Vec<i64>],
    rows: usize,
    diagnostics: &mut numeric_objects::Diagnostics,
) -> Result<bool, Failure> {
    for row in 1..rows {
        for (level_no, (level, code)) in levels.iter().zip(codes).enumerate() {
            let (left, right) = (code[row - 1] as usize, code[row] as usize);
            // Tuple equality first recognizes the very same retained owner.
            if left == right {
                continue;
            }
            let order = match &level.values {
                Values::TemporalObject { values } => {
                    temporal::rank_comparison(&values[left], &values[right])
                }
                _ => match compare(level, left, right, diagnostics) {
                    Ok(order) => Some(order),
                    Err(reason) if is_type_error(reason) => None,
                    Err(reason) => return Err((reason, level_no, left, right)),
                },
            };
            if order != Some(Ordering::Equal) {
                if order != Some(Ordering::Less) {
                    return Ok(false);
                }
                break;
            }
        }
    }
    Ok(true)
}

fn outer_monotonic(
    levels: &[Facts],
    codes: &[Vec<i64>],
    rows: usize,
    monotonic: &[bool],
    ranks: Option<&[Vec<usize>]>,
    diagnostics: &mut numeric_objects::Diagnostics,
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
            compare(&levels[level_no], left, right, diagnostics)
                .map_err(|reason| (reason, level_no, left, right))
        });
        if let Err(failure) = answer {
            if is_type_error(failure.0) {
                return tuple_monotonic(levels, codes, rows, diagnostics);
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
    diagnostics: &mut numeric_objects::Diagnostics,
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
    let early_guard = code_order(codes, missing.len());
    let mut numeric_probes = vec![None; levels.len()];
    let mut previous_monotonic = true;
    for (level_no, level) in levels.iter().enumerate() {
        if let Values::ScalarObject {
            values,
            semantics,
            overflow,
        } = &level.values
        {
            let monotonic = if early_guard && previous_monotonic {
                match numeric_objects::monotonic(
                    values,
                    &level.missing,
                    &(0..values.len()).collect::<Vec<_>>(),
                    *semantics,
                    *overflow,
                    diagnostics,
                )? {
                    Ok(value) => {
                        numeric_probes[level_no] = Some(value);
                        value
                    }
                    Err((reason, left, right)) => {
                        return Ok(refused((reason, level_no, left, right)));
                    }
                }
            } else {
                false
            };
            original_monotonic.push(monotonic);
            cached.push(None);
        } else if let Values::TemporalObject { values } = &level.values {
            original_monotonic.push(temporal::monotonic(
                values,
                &level.missing,
                &(0..values.len()).collect::<Vec<_>>(),
            )?);
            cached.push(None);
        } else {
            let result = order_inner(level, diagnostics)?;
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
        previous_monotonic &= *original_monotonic.last().unwrap();
    }
    // Native level normalization first permits an already-lexsorted index to
    // retain even unused, mutually incomparable level owners without sorting.
    if early_guard {
        match outer_monotonic(
            levels,
            codes,
            missing.len(),
            &original_monotonic,
            None,
            diagnostics,
        ) {
            Ok(true) => return Ok(sorted((0..missing.len()).collect(), true)),
            Err(failure) => return Ok(refused(failure)),
            Ok(false) => {}
        }
    }
    let mut ranks = Vec::new();
    let mut normalized_monotonic = Vec::new();
    let mut numeric_permutations = vec![None; levels.len()];
    for (level_no, (level, code)) in levels.iter().zip(codes).enumerate() {
        let result = match cached[level_no].take() {
            Some(result) => result,
            None => {
                if let Values::ScalarObject {
                    values,
                    semantics,
                    overflow,
                } = &level.values
                {
                    if numeric_probes[level_no] == Some(true) {
                        sorted((0..values.len()).collect(), true)
                    } else {
                        numeric_objects::order(
                            values,
                            &level.missing,
                            level.profile,
                            *semantics,
                            *overflow,
                            diagnostics,
                            numeric_objects::OrderStage::Level {
                                probe_first: numeric_probes[level_no].is_none(),
                            },
                        )?
                    }
                } else {
                    order_inner(level, diagnostics)?
                }
            }
        };
        let permutation = match result {
            ResultOrder::NeedsProfile => return Ok(ResultOrder::NeedsProfile),
            ResultOrder::Sorted {
                positions,
                already_ordered,
                ..
            } => {
                if !already_ordered && matches!(level.values, Values::ScalarObject { .. }) {
                    numeric_permutations[level_no] = Some(positions.clone());
                }
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
                ..
            } => {
                if !is_type_error(reason) {
                    path.insert(0, level_no);
                    return Ok(ResultOrder::Refused {
                        reason,
                        left,
                        right,
                        levels: path,

                        overflow_warnings: Vec::new(),
                        overflow_prints: 0,
                        integer_destination: None,
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
    for (level_no, level) in levels.iter().enumerate() {
        if let (
            Values::ScalarObject {
                values,
                semantics,
                overflow,
            },
            Some(positions),
        ) = (&level.values, &numeric_permutations[level_no])
        {
            normalized_monotonic[level_no] = match numeric_objects::monotonic(
                values,
                &level.missing,
                positions,
                *semantics,
                *overflow,
                diagnostics,
            )? {
                Ok(value) => value,
                Err((reason, left, right)) => return Ok(refused((reason, level_no, left, right))),
            };
        }
        if !normalized_monotonic[level_no] {
            break;
        }
    }
    match outer_monotonic(
        levels,
        codes,
        missing.len(),
        &normalized_monotonic,
        Some(&ranks),
        diagnostics,
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
