// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema policy over separately reached native observations.

use crate::assemble::{self, Given};
use crate::infer::{ColumnRange, InferredColumn, InferredTable, Taken, TimeChoice, WholeTable};
use crate::keys::KEY_UNIQUENESS;
use sdm_contract::data::Relation;
use sdm_contract::direct::ColumnFlags;
use sdm_contract::direct::dtype::DtypePopulation;
use sdm_contract::direct::schema::{ObservedColumn, ObservedSchema, SchemaObservation};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Detail, Scalar, Sentence};
use std::collections::{BTreeMap, BTreeSet};

fn key_candidate(column: &ObservedColumn) -> bool {
    let name = column.name.to_lowercase();
    ["_id", "_key", "id", "key"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
        && assemble::can_identify(column.dtype)
}

fn time_candidate(column: &ObservedColumn, key: Option<&str>) -> bool {
    column.stype == Stype::Timestamp
        && Some(column.name.as_str()) != key
        && !crate::names::is_load_stamp(&column.name)
        && !crate::names::is_birth_date(&column.name)
}

fn inferred_column(c: ObservedColumn) -> InferredColumn {
    let received = match c.dtype {
        Dtype::Bool => ColumnType::Bool,
        Dtype::Int64 => ColumnType::Int64,
        Dtype::Float32 => ColumnType::Float32,
        Dtype::Float64 => ColumnType::Float64,
        Dtype::Date | Dtype::Time => ColumnType::Timestamp {
            unit: sdm_contract::Unit::Us,
            zone: None,
        },
        _ => ColumnType::Utf8,
    };
    InferredColumn {
        name: c.name,
        dtype: c.dtype,
        stype: c.stype,
        stype_before_key: None,
        received,
        nullable: true,
        unique: false,
        exact_number: None,
    }
}

pub fn observe(input: SchemaObservation) -> Result<(ObservedSchema, Vec<CoreWarning>), CoreError> {
    let result = match input {
        SchemaObservation::TableNames { table, columns } => {
            super::check_name(&table)?;
            for column in &columns {
                super::check_name(column)?;
            }
            ObservedSchema::Names { names: columns }
        }
        SchemaObservation::Column {
            name,
            finding,
            distinct,
        } => {
            super::dtype::check_finding(&finding)?;
            if finding.population != DtypePopulation::Declared {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "a column's native dtype population is unfinished",
                ));
            }
            let stype = match finding.dtype {
                Dtype::Int64 | Dtype::Float32 | Dtype::Float64 => Stype::Numerical,
                Dtype::Bool => Stype::Categorical,
                Dtype::Date | Dtype::Time => Stype::Timestamp,
                _ => {
                    if distinct.ok_or_else(|| {
                        CoreError::protocol(
                            Sentence::InvalidMessage,
                            "a string column lacks its native cardinality",
                        )
                    })? <= 64
                    {
                        Stype::Categorical
                    } else {
                        Stype::Text
                    }
                }
            };
            ObservedSchema::Column {
                column: ObservedColumn {
                    name,
                    dtype: finding.dtype,
                    stype,
                    untyped: finding.untyped,
                },
            }
        }
        SchemaObservation::KeyCandidates { columns, rows } => ObservedSchema::Names {
            names: columns
                .into_iter()
                .filter(|c| rows > 0 && key_candidate(c))
                .map(|c| c.name)
                .collect(),
        },
        SchemaObservation::KeyCount { rows, distinct } => ObservedSchema::Key {
            accepted: rows > 0 && distinct as f64 >= rows as f64 * KEY_UNIQUENESS,
        },
        SchemaObservation::KeyCompanions {
            columns,
            distinguishing,
        } => ObservedSchema::Names {
            names: distinguishing
                .into_iter()
                .filter(|name| {
                    columns
                        .iter()
                        .any(|column| column.name == *name && assemble::can_identify(column.dtype))
                })
                .collect(),
        },
        SchemaObservation::KeyValidity {
            table,
            column,
            nulls,
            duplicates,
            rows,
            empty,
        } => {
            let refused = if nulls > 0 {
                Some((
                    if empty.is_empty() {
                        Sentence::DirectKeyMissingPart
                    } else {
                        Sentence::DirectKeyMissing
                    },
                    nulls,
                ))
            } else if duplicates as f64 > rows as f64 * (1. - KEY_UNIQUENESS) {
                Some((Sentence::DirectKeyNotUnique, duplicates))
            } else {
                None
            };
            if let Some((sentence, count)) = refused {
                return Err(
                    CoreError::new(sentence, "the inferred key changed before adoption")
                        .with_field("table", Detail::text(table))
                        .with_field("column", Detail::text(column))
                        .with_field("count", Detail::Scalar(Scalar::UInt64(count)))
                        .with_field("rows", Detail::Scalar(Scalar::UInt64(rows as u64))),
                );
            }
            ObservedSchema::Names { names: vec![] }
        }
        SchemaObservation::KeyRepeated {
            table,
            column,
            duplicates,
            distinguishing,
            usable,
        } => {
            let sentence = if !usable.is_empty() {
                Sentence::DirectKeyComposite
            } else if !distinguishing.is_empty() {
                Sentence::DirectKeyNoComposite
            } else {
                Sentence::DirectKeyRepeated
            };
            let warning = CoreWarning::new(sentence, "the inferred key repeats")
                .with_field("table", Detail::text(table))
                .with_field("column", Detail::text(column))
                .with_field("count", Detail::Scalar(Scalar::UInt64(duplicates)))
                .with_field(
                    "distinguishing",
                    Detail::List(distinguishing.into_iter().map(Scalar::Utf8).collect()),
                );
            return Ok((
                ObservedSchema::Names { names: vec![] },
                if duplicates > 0 {
                    vec![warning]
                } else {
                    vec![]
                },
            ));
        }
        SchemaObservation::TimeCandidates {
            columns,
            primary_key,
        } => ObservedSchema::Names {
            names: columns
                .into_iter()
                .filter(|c| time_candidate(c, primary_key.as_deref()))
                .map(|c| c.name)
                .collect(),
        },
        SchemaObservation::TimeChoice {
            columns,
            primary_key,
            rows,
            populations,
            starts,
        } => {
            let expected: Vec<_> = columns
                .iter()
                .filter(|c| time_candidate(c, primary_key.as_deref()))
                .map(|c| c.name.as_str())
                .collect();
            if expected
                != populations
                    .iter()
                    .map(|p| p.name.as_str())
                    .collect::<Vec<_>>()
            {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "native time populations do not match the eligible columns",
                ));
            }
            let row_counts: Vec<_> = populations
                .iter()
                .map(|p| (p.name.clone(), p.rows))
                .collect();
            let ranges = populations
                .into_iter()
                .map(|p| {
                    (
                        p.name,
                        ColumnRange {
                            held: p.held,
                            same: p.same,
                            low: None,
                            start: None,
                        },
                    )
                })
                .collect();
            let columns: Vec<_> = columns.into_iter().map(inferred_column).collect();
            let key: Vec<_> = primary_key.into_iter().collect();
            let mut whole = WholeTable {
                rows,
                columns: ranges,
            };
            let mut choice =
                crate::infer::choose_direct_populations(&columns, &key, &whole, &row_counts)?;
            if let Some(starts) = starts {
                if choice.how != Some(Taken::Earliest)
                    || starts.iter().map(|(name, _)| name).ne(choice.among.iter())
                {
                    return Err(CoreError::protocol(
                        Sentence::InvalidMessage,
                        "native starts do not match the time finalists",
                    ));
                }
                for (name, start) in starts {
                    let value = start
                        .nanos
                        .map(|v| {
                            v.parse::<i128>().map_err(|_| {
                                CoreError::protocol(
                                    Sentence::InvalidMessage,
                                    "native time start is not exact nanoseconds",
                                )
                            })
                        })
                        .transpose()?;
                    if let Some((_, range)) =
                        whole.columns.iter_mut().find(|(column, _)| *column == name)
                    {
                        range.start = value;
                    }
                }
                choice =
                    crate::infer::choose_direct_populations(&columns, &key, &whole, &row_counts)?;
                ObservedSchema::Time {
                    column: choice.column,
                    earliest: vec![],
                }
            } else {
                let earliest = if choice.how == Some(Taken::Earliest) {
                    choice.among
                } else {
                    vec![]
                };
                ObservedSchema::Time {
                    column: choice.column,
                    earliest,
                }
            }
        }
        SchemaObservation::CategoricalAdvice {
            table,
            column,
            distinct,
            present,
        } => {
            // These are separate native reads. Mutation between them may
            // legitimately make the earlier cardinality exceed the count.
            let warnings = if distinct as f64 >= present as f64 * 0.5 && distinct >= 32 {
                vec![
                    CoreWarning::new(
                        Sentence::DirectCategoricalPopulation,
                        "categorical meaning depends on the observed population",
                    )
                    .with_field("table", Detail::text(table))
                    .with_field("column", Detail::text(column))
                    .with_field("distinct", Detail::Scalar(Scalar::UInt64(distinct as u64)))
                    .with_field("present", Detail::Scalar(Scalar::UInt64(present as u64))),
                ]
            } else {
                vec![]
            };
            return Ok((ObservedSchema::Names { names: vec![] }, warnings));
        }
        SchemaObservation::DateAdviceCandidates {
            columns,
            primary_key,
            time_column,
            empty,
        } => ObservedSchema::Dates {
            names: if empty || time_column.is_some() {
                vec![]
            } else {
                columns
                    .into_iter()
                    .filter(|c| {
                        c.dtype == Dtype::String
                            && c.stype != Stype::Id
                            && primary_key.as_ref() != Some(&c.name)
                    })
                    .map(|c| c.name)
                    .collect()
            },
            utc_readings: vec![false, true],
        },
        SchemaObservation::DateMinimumYear { year } => ObservedSchema::Key {
            accepted: year >= 1900,
        },
        SchemaObservation::DateMaximumYear {
            table,
            column,
            minimum,
            year,
        } => {
            if minimum < 1900 {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "date advice maximum requires an accepted minimum year",
                ));
            }
            let accepted = year <= 2100;
            let warnings = if accepted {
                vec![
                    CoreWarning::new(
                        Sentence::DirectDatesAsText,
                        "native text conversion found plausible dates",
                    )
                    .with_field("table", Detail::text(table))
                    .with_field("column", Detail::text(column)),
                ]
            } else {
                vec![]
            };
            return Ok((ObservedSchema::Key { accepted }, warnings));
        }
        SchemaObservation::Assemble {
            table,
            columns,
            primary_key,
            duplicates,
            time_column,
        } => {
            super::check_name(&table)?;
            if primary_key
                .as_ref()
                .is_some_and(|key| !columns.iter().any(|c| &c.name == key && key_candidate(c)))
            {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "the native schema key is not an eligible column",
                ));
            }
            let flags = columns
                .iter()
                .map(|c| ColumnFlags {
                    name: c.name.clone(),
                    untyped: c.untyped,
                    stype_declared: false,
                })
                .collect();
            if time_column.as_ref().is_some_and(|name| {
                !columns
                    .iter()
                    .any(|c| &c.name == name && time_candidate(c, primary_key.as_deref()))
            }) {
                return Err(CoreError::protocol(
                    Sentence::InvalidMessage,
                    "native time selection is not an eligible column",
                ));
            }
            let columns = columns.into_iter().map(inferred_column).collect();
            let inferred = InferredTable {
                name: table.clone(),
                source: Relation::named(&table),
                columns,
                primary_key: primary_key.into_iter().collect(),
                time_column: time_column.clone(),
                time: TimeChoice {
                    column: time_column,
                    ..TimeChoice::default()
                },
                left_out: vec![],
                unread_times: vec![],
                day_first: vec![],
                tied_keys: vec![],
                unique: vec![],
                declared: BTreeSet::new(),
            };
            let built = assemble::assemble(
                vec![inferred],
                vec![],
                Given {
                    relationships: Some(&[]),
                    time_columns: None,
                },
                Some(Scalar::UInt64(duplicates)),
            )?;
            let schema = crate::held::held(&built.description, &BTreeSet::new(), &BTreeMap::new());
            return Ok((ObservedSchema::Schema { schema, flags }, built.warnings));
        }
    };
    Ok((result, vec![]))
}
