// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scalar source meanings at their native read/construction position.
//! Provider identity and native package names select no conversion policy.

use super::*;
use sdm_contract::frames::NativeDecimal;
use sdm_contract::frames::materialize::FramePhysicalDeclaration;
use sdm_contract::materialize::{MaterializeScalarMeaning, MaterializeSourceFacts};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod failure_tests;

/// Transform a reported actual failure, without claiming that relation
/// membership alone proves that its request was issued by a particular Run.
/// Success data continues through that Run's actual complete read partition.
pub(super) fn reported_failure(
    tables: &[sdm_contract::materialize::MaterializeSourceTable],
    read: &sdm_contract::materialize::MaterializeSourceRead,
    error: sdm_contract::data::ProviderError,
) -> Result<(), CoreError> {
    let table = tables
        .iter()
        .find(|table| table.read == read.source)
        .ok_or_else(|| crate::prepare::invalid("source failure names a foreign relation"))?;
    let mut seen = BTreeSet::new();
    if read
        .columns
        .iter()
        .any(|column| !seen.insert(column) || !table.columns.contains(column))
    {
        return Err(crate::prepare::invalid(
            "source failure does not retain its complete declared column membership",
        ));
    }
    Err(crate::describe::provider_error(error, read.kind.name()))
}

pub(super) fn scalar_meaning(
    declared: Option<Dtype>,
    input: &MaterializeSourceFacts,
) -> Result<MaterializeScalarMeaning, CoreError> {
    use MaterializeScalarMeaning as M;
    let integer = matches!(
        declared,
        Some(Dtype::Int | Dtype::Byte | Dtype::Int16 | Dtype::Int32 | Dtype::Int64)
    );
    let floating = matches!(
        declared,
        Some(Dtype::Float | Dtype::Float32 | Dtype::Float64)
    );
    match input {
        MaterializeSourceFacts::CellRows {} => Ok(if integer {
            M::Nullable {
                physical: ColumnType::Int64,
            }
        } else if floating {
            M::Nullable {
                physical: ColumnType::Float64,
            }
        } else if declared == Some(Dtype::Bool) {
            M::Nullable {
                physical: ColumnType::Bool,
            }
        } else if declared == Some(Dtype::String) {
            M::Nullable {
                physical: ColumnType::Utf8,
            }
        } else {
            M::Original {}
        }),
        MaterializeSourceFacts::ColumnValues { .. }
        | MaterializeSourceFacts::TimeConversion { .. } => Ok(M::Original {}),
        MaterializeSourceFacts::TypedColumns { declaration } => {
            if let Some(declaration) = declaration {
                crate::frame_materialize::values::check_declaration(declaration)?;
            }
            let Some(FramePhysicalDeclaration::Decimal { native }) = declaration else {
                return Ok(M::Original {});
            };
            let scale = match native {
                NativeDecimal::Decimal32 { scale, .. }
                | NativeDecimal::Decimal64 { scale, .. }
                | NativeDecimal::Decimal128 { scale, .. }
                | NativeDecimal::Decimal256 { scale, .. } => *scale,
            };
            Ok(if integer || declared.is_none() && scale == 0 {
                M::ExactDecimal {
                    physical: ColumnType::Int64,
                }
            } else if floating || declared.is_none() {
                M::ExactDecimal {
                    physical: ColumnType::Float64,
                }
            } else if declared == Some(Dtype::String) {
                M::Decimal {
                    physical: ColumnType::Utf8,
                }
            } else {
                M::Original {}
            })
        }
    }
}

/// Fixed native timestamp construction requires both resolved meaning and
/// observed typed storage; unknown scalar/text sources retain their Reader.
pub(super) fn timestamp_meaning(
    declared: Option<Dtype>,
    stype: Stype,
    input: &MaterializeSourceFacts,
) -> Option<MaterializeScalarMeaning> {
    if declared != Some(Dtype::Time) && stype != Stype::Timestamp {
        return None;
    }
    match input {
        MaterializeSourceFacts::ColumnValues {
            inferred,
            all_missing,
        } if *all_missing
            || *inferred == sdm_contract::frames::materialize::FrameObservedContent::Temporal =>
        {
            Some(MaterializeScalarMeaning::Nullable {
                physical: ColumnType::Timestamp {
                    unit: sdm_contract::types::TimestampUnit::Ns,
                    zone: None,
                },
            })
        }
        MaterializeSourceFacts::TypedColumns {
            declaration:
                Some(FramePhysicalDeclaration::Physical {
                    physical: ColumnType::Timestamp { .. },
                }),
        } => Some(MaterializeScalarMeaning::Nullable {
            physical: ColumnType::Timestamp {
                unit: sdm_contract::types::TimestampUnit::Ns,
                zone: None,
            },
        }),
        _ => None,
    }
}

pub(super) fn time_notice(
    declared: Option<Dtype>,
    stype: Stype,
    column: &str,
    input: &MaterializeSourceFacts,
) -> Result<Option<CoreWarning>, CoreError> {
    let MaterializeSourceFacts::TimeConversion {
        rows,
        unread,
        examples,
    } = input
    else {
        return Ok(None);
    };
    if (declared != Some(Dtype::Time) && stype != Stype::Timestamp)
        || *unread == 0
        || unread > rows
        || examples.is_empty()
    {
        return Err(crate::prepare::invalid(
            "native time outcome does not retain a reached temporal coercion",
        ));
    }
    Ok(Some(CoreWarning {
        fields: [
            ("column".to_owned(), Detail::text(column)),
            ("unread".to_owned(), Detail::Scalar(Scalar::UInt64(*unread))),
            ("rows".to_owned(), Detail::Scalar(Scalar::UInt64(*rows))),
            ("examples".to_owned(), Detail::text(examples)),
        ]
        .into(),
        ..CoreWarning::new(
            Sentence::NativeTimeValuesUnread,
            "native datetime values were read as no time and dropped",
        )
    }))
}
