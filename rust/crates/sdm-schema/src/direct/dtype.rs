// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct-column meaning from reached native storage and range facts.

use sdm_contract::direct::dtype::{
    DtypeFinding, DtypeObservation, DtypePopulation, DtypeStorage, ObjectContent,
};
use sdm_contract::ops::Dtype;
use sdm_contract::{CoreError, Sentence};

fn finding(dtype: Dtype, population: DtypePopulation, untyped: bool) -> DtypeFinding {
    DtypeFinding {
        dtype,
        population,
        untyped,
        decimal_safe_precision: 18,
        integer_min: i64::MIN.to_string(),
        integer_max: i64::MAX.to_string(),
    }
}

pub fn declared(dtype: Dtype, untyped: bool) -> DtypeFinding {
    finding(dtype, DtypePopulation::Declared, untyped)
}

pub(super) fn check_finding(value: &DtypeFinding) -> Result<(), CoreError> {
    let valid = match value.population {
        DtypePopulation::WholeNumbers => value.dtype == Dtype::Int64 && !value.untyped,
        DtypePopulation::Decimals => {
            matches!(value.dtype, Dtype::Int64 | Dtype::String) && !value.untyped
        }
        DtypePopulation::Declared => {
            matches!(
                value.dtype,
                Dtype::Bool
                    | Dtype::Time
                    | Dtype::Int64
                    | Dtype::Float32
                    | Dtype::Float64
                    | Dtype::String
            ) && (!value.untyped || value.dtype == Dtype::String)
        }
    };
    if !valid || *value != finding(value.dtype, value.population, value.untyped) {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "native dtype facts carry inconsistent bounds",
        ));
    }
    Ok(())
}

pub fn observe(input: DtypeObservation) -> Result<DtypeFinding, CoreError> {
    use DtypePopulation::{Decimals, Declared, WholeNumbers};
    let invalid = || {
        CoreError::protocol(
            Sentence::InvalidMessage,
            "native dtype observations disagree with their column population",
        )
    };
    Ok(match input {
        DtypeObservation::Storage { storage } => match storage {
            DtypeStorage::Boolean => finding(Dtype::Bool, Declared, false),
            DtypeStorage::Timestamp => finding(Dtype::Time, Declared, false),
            DtypeStorage::Integer { may_exceed_i64 } => finding(
                Dtype::Int64,
                if may_exceed_i64 {
                    WholeNumbers
                } else {
                    Declared
                },
                false,
            ),
            DtypeStorage::Floating { bytes } => finding(
                if bytes > 4 {
                    Dtype::Float64
                } else {
                    Dtype::Float32
                },
                Declared,
                false,
            ),
            DtypeStorage::Decimal { precision, scale } => {
                if scale > 0 {
                    finding(Dtype::Float64, Declared, false)
                } else if precision.ok_or_else(invalid)? <= 18 {
                    finding(Dtype::Int64, Declared, false)
                } else {
                    finding(Dtype::Int64, Decimals, false)
                }
            }
            DtypeStorage::Object { content } => match content {
                ObjectContent::Empty => finding(Dtype::String, Declared, true),
                ObjectContent::Decimal => finding(Dtype::Int64, Decimals, false),
                ObjectContent::Integer => finding(Dtype::Int64, WholeNumbers, false),
                ObjectContent::Floating | ObjectContent::MixedIntegerFloat => {
                    finding(Dtype::Float64, Declared, false)
                }
                ObjectContent::Boolean => finding(Dtype::Bool, Declared, false),
                ObjectContent::Date | ObjectContent::DateTime | ObjectContent::Timestamp => {
                    finding(Dtype::Time, Declared, false)
                }
                ObjectContent::Other => finding(Dtype::String, Declared, false),
            },
            DtypeStorage::Other => finding(Dtype::String, Declared, false),
        },
        DtypeObservation::WholeNumber {
            previous,
            within_range,
        } => {
            check_finding(&previous)?;
            if previous.population != WholeNumbers
                || previous.dtype != Dtype::Int64
                || previous.untyped
            {
                return Err(invalid());
            }
            if within_range == Some(true) {
                previous
            } else {
                finding(Dtype::String, Declared, false)
            }
        }
        DtypeObservation::Decimal {
            previous,
            is_decimal,
            finite,
            fractional,
            outside_integer_range,
        } => {
            check_finding(&previous)?;
            if previous.population != Decimals
                || !matches!(previous.dtype, Dtype::Int64 | Dtype::String)
                || previous.untyped
            {
                return Err(invalid());
            }
            if !is_decimal || !finite || fractional {
                finding(Dtype::Float64, Declared, false)
            } else if outside_integer_range {
                finding(Dtype::String, Decimals, false)
            } else {
                previous
            }
        }
        DtypeObservation::Exhausted { finding: previous } => {
            check_finding(&previous)?;
            finding(previous.dtype, Declared, previous.untyped)
        }
    })
}
