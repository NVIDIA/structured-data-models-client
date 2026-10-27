// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Primitive native-frame inference, distinct from SQL catalog/preview
//! inference. Shared value/equality and schema assembly rules stay shared.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use sdm_contract::data::Relation;
use sdm_contract::ops::{Dtype, SourceDescription, Stype};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Detail, Scalar, Sentence};

use crate::infer::{InferredColumn, InferredTable};
use crate::keys::{Counted, KEY_UNIQUENESS};
use crate::values::{self, Floats, Value};

/// One table's primitive inference and call-local row statistics.
pub struct Read {
    pub table: InferredTable,
    values: Vec<Vec<Option<Value>>>,
    batches: Vec<RecordBatch>,
    pub rows: usize,
}

impl Read {
    /// Count a key against the current admitted snapshot, not old metadata.
    pub fn count(&self, key: &[String]) -> Counted {
        let columns: Vec<usize> = key
            .iter()
            .filter_map(|name| self.table.columns.iter().position(|c| &c.name == name))
            .collect();
        let mut seen = BTreeSet::new();
        let mut missing = 0;
        let mut repeated = None;
        for row in 0..self.rows {
            let parts: Option<Vec<Value>> = columns
                .iter()
                .map(|&at| self.values[at][row].clone())
                .collect();
            if let Some(parts) = parts {
                if !seen.insert(parts) && repeated.is_none() {
                    repeated = Some(row);
                }
            } else {
                missing += 1;
            }
        }
        let example = repeated.and_then(|mut row| {
            for batch in &self.batches {
                if row < batch.num_rows() {
                    return columns
                        .iter()
                        .map(|&at| values::scalar(batch.column(at).as_ref(), row))
                        .collect();
                }
                row -= batch.num_rows();
            }
            None
        });
        Counted {
            rows: self.rows as u64,
            distinct: seen.len() as u64,
            missing,
            example,
        }
    }

    /// The immediate inferred-key warnings, before a later table is read.
    pub fn hold_inferred(&mut self) -> Result<Vec<CoreWarning>, CoreError> {
        if self.table.primary_key.is_empty() {
            return Ok(vec![]);
        }
        let count = self.count(&self.table.primary_key);
        let mut notices = Vec::new();
        if count.missing > 0 {
            notices.push(
                CoreWarning::new(
                    Sentence::KeyNoValueColumn,
                    "an inferred key has missing values",
                )
                .with_fields(vec![
                    ("table", Detail::text(&self.table.name)),
                    ("column", Detail::text(&self.table.primary_key[0])),
                    ("missing", number(count.missing)),
                ]),
            );
        }
        if count.repeats() > 0 {
            notices.push(self.repeated(&self.table.primary_key, &count));
        }
        Ok(notices)
    }

    fn fields(&self, key: &[String], count: &Counted) -> Vec<(&'static str, Detail)> {
        let examples = count.example.clone().unwrap_or_default();
        vec![
            ("table", Detail::text(&self.table.name)),
            ("key", Detail::text(key.join("."))),
            ("columns", texts(key)),
            ("repeats", number(count.repeats())),
            ("rows", number(count.rows)),
            ("example", Detail::List(examples)),
        ]
    }

    fn repeated(&self, key: &[String], count: &Counted) -> CoreWarning {
        let distinguishing: Vec<_> = self
            .table
            .columns
            .iter()
            .filter(|c| !key.contains(&c.name))
            .filter(|c| {
                let mut combined = key.to_vec();
                combined.push(c.name.clone());
                self.count(&combined).repeats() == 0
            })
            .map(|c| c.name.clone())
            .collect();
        let usable: Vec<_> = distinguishing
            .iter()
            .filter(|name| {
                self.table
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, c)| &c.name == *name)
                    .is_some_and(|(at, c)| {
                        crate::assemble::can_identify(c.dtype)
                            && self.values[at].iter().all(Option::is_some)
                    })
            })
            .cloned()
            .collect();
        let (sentence, others) = if !usable.is_empty() {
            (Sentence::FrameKeyRepeatsComposite, usable)
        } else if !distinguishing.is_empty() {
            (Sentence::FrameKeyRepeatsUnusable, distinguishing)
        } else {
            (Sentence::FrameKeyRepeats, vec![])
        };
        let mut fields = self.fields(key, count);
        fields.push(("others", texts(&others)));
        CoreWarning::new(sentence, "a frame key repeats").with_fields(fields)
    }

    /// Explicit constructor assertions use frame wording and current rows;
    /// SQL's check_keys=False advice is not a frame-constructor option.
    pub fn hold_stated(&self, key: &[String]) -> Result<Vec<CoreWarning>, CoreError> {
        let count = self.count(key);
        if count.missing > 0 {
            let empty: Vec<_> = key
                .iter()
                .filter(|name| {
                    self.table
                        .columns
                        .iter()
                        .position(|c| &c.name == *name)
                        .is_some_and(|at| self.values[at].iter().any(Option::is_none))
                })
                .cloned()
                .collect();
            let mut fields = self.fields(key, &count);
            fields.push(("missing", number(count.missing)));
            fields.push(("missing_columns", texts(&empty)));
            fields.push((
                "pronoun",
                Detail::text(if empty.len() > 1 { "them" } else { "it" }),
            ));
            return Err(failure(Sentence::FrameKeyMissing, fields));
        }
        if count.repeats() as f64 > count.rows as f64 * (1.0 - KEY_UNIQUENESS) {
            return Err(failure(
                Sentence::FrameKeyNotUnique,
                self.fields(key, &count),
            ));
        }
        Ok(if count.repeats() > 0 {
            vec![self.repeated(key, &count)]
        } else {
            vec![]
        })
    }
}

fn number(value: u64) -> Detail {
    Detail::Scalar(Scalar::UInt64(value))
}
fn texts<S: AsRef<str>>(values: &[S]) -> Detail {
    Detail::List(
        values
            .iter()
            .map(|v| Scalar::Utf8(v.as_ref().into()))
            .collect(),
    )
}
fn failure(sentence: Sentence, fields: Vec<(&str, Detail)>) -> CoreError {
    let mut error = CoreError::new(sentence, "a frame primary key assertion failed");
    for (name, value) in fields {
        error = error.with_field(name, value);
    }
    error
}

/// Notices specific to full frame construction, after overrides and links.
pub fn final_notices(
    reads: &[Read],
    description: &SourceDescription,
    infer_links: bool,
) -> Result<Vec<CoreWarning>, CoreError> {
    let mut notices = Vec::new();
    let mut named = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut keyed = BTreeMap::<&str, Vec<&str>>::new();
    for table in &description.tables {
        for c in &table.columns {
            named.entry(&c.name).or_default().insert(&table.name);
        }
        if table.frame_key_form != Some(sdm_contract::frames::FrameKeyForm::Sequence)
            && let [key] = table.primary_key.as_slice()
        {
            keyed.entry(key).or_default().push(&table.name);
        }
    }
    if infer_links {
        let mut spoken = BTreeSet::new();
        for (read, table) in reads.iter().zip(&description.tables) {
            if !table.primary_key.is_empty() || read.rows == 0 {
                continue;
            }
            for c in &table.columns {
                let shaped = ["_id", "_key", "id", "key"]
                    .iter()
                    .any(|suffix| c.name.to_lowercase().ends_with(suffix));
                let count = read.count(std::slice::from_ref(&c.name));
                if shaped
                    && crate::assemble::can_identify(c.dtype)
                    && !keyed.contains_key(c.name.as_str())
                    && named[c.name.as_str()].len() > 1
                    && count.distinct as f64 >= read.rows as f64 * 0.9
                    && count.repeats() > 0
                {
                    let mut fields = read.fields(std::slice::from_ref(&c.name), &count);
                    fields.push(("column", Detail::text(&c.name)));
                    notices.push(
                        CoreWarning::new(
                            Sentence::FrameKeyDeclined,
                            "a frame key-shaped column repeats",
                        )
                        .with_fields(fields),
                    );
                    spoken.insert(table.name.as_str());
                    break;
                }
            }
        }
        for (read, table) in reads.iter().zip(&description.tables) {
            if !table.primary_key.is_empty()
                || read.rows == 0
                || spoken.contains(table.name.as_str())
            {
                continue;
            }
            let candidates: Vec<_> = table
                .columns
                .iter()
                .filter(|c| matches!(c.dtype, Dtype::String | Dtype::Int64))
                .filter(|c| {
                    !keyed.contains_key(c.name.as_str()) && named[c.name.as_str()].len() > 1
                })
                .filter(|c| {
                    let count = read.count(std::slice::from_ref(&c.name));
                    count.missing == 0 && count.repeats() == 0
                })
                .map(|c| c.name.clone())
                .collect();
            if candidates.is_empty() {
                continue;
            }
            let elsewhere: BTreeSet<_> = candidates
                .iter()
                .flat_map(|c| named[c.as_str()].iter().copied())
                .filter(|n| *n != table.name)
                .collect();
            let many = candidates.len() > 1;
            let shared = elsewhere.len() > 1;
            let fields = vec![
                ("table", Detail::text(&table.name)),
                ("candidates", texts(&candidates)),
                (
                    "identify",
                    Detail::text(if many { "identify" } else { "identifies" }),
                ),
                ("verb", Detail::text(if many { "are" } else { "is" })),
                (
                    "elsewhere",
                    Detail::text(if shared {
                        "other tables"
                    } else {
                        "another table"
                    }),
                ),
                (
                    "each",
                    Detail::text(if many { "each of them" } else { "it" }),
                ),
                (
                    "identifies",
                    Detail::text(if many {
                        "one of them identifies"
                    } else {
                        "it identifies"
                    }),
                ),
                ("subject", Detail::text(if many { "they" } else { "it" })),
                (
                    "name_verb",
                    Detail::text(if many { "name" } else { "names" }),
                ),
                (
                    "tables_word",
                    Detail::text(if shared { "tables" } else { "table" }),
                ),
                (
                    "need_key",
                    Detail::text(if shared {
                        "those tables are the ones that need"
                    } else {
                        "that table is the one that needs"
                    }),
                ),
            ];
            notices.push(
                CoreWarning::new(
                    Sentence::FrameKeylessShared,
                    "a frame cannot be a relationship target without a key",
                )
                .with_fields(fields),
            );
        }
        for (key, names) in keyed.iter().filter(|(_, names)| names.len() > 1) {
            let mut names = names.clone();
            names.sort();
            notices.push(
                CoreWarning::new(
                    Sentence::FrameKeyAmbiguous,
                    "a key name is ambiguous among frames",
                )
                .with_fields(vec![
                    ("key", Detail::text(*key)),
                    ("tables", texts(&names)),
                    ("first", Detail::text(names[0])),
                ]),
            );
        }
    }
    for (read, table) in reads.iter().zip(&description.tables) {
        for (at, c) in table.columns.iter().enumerate() {
            if c.dtype != Dtype::String || c.stype != Stype::Categorical {
                continue;
            }
            let distinct = values::distinct(&read.values[at]);
            let present = read.values[at].iter().flatten().count();
            if distinct >= 32 && distinct as f64 >= present as f64 * 0.5 {
                notices.push(
                    CoreWarning::new(
                        Sentence::FrameCategoricalSample,
                        "frame size determined a column's meaning",
                    )
                    .with_fields(vec![
                        ("table", Detail::text(&table.name)),
                        ("column", Detail::text(&c.name)),
                        ("distinct", number(distinct as u64)),
                        ("present", number(present as u64)),
                    ]),
                );
            }
        }
    }
    Ok(notices)
}

/// Whole-table frame inference, not preview-sized SQL inference. This first
/// consumer accepts NumPy/pandas masked numeric/Boolean arrays, native
/// timestamp arrays and built-in pandas text, without object or Decimal conversion. Native
/// conversion has already encoded missingness in Arrow validity.
pub fn read(
    name: &str,
    source: &Relation,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<Read, CoreError> {
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let mut columns = Vec::new();
    let mut all_values = Vec::new();
    for (at, field) in schema.fields().iter().enumerate() {
        let values = values::column(batches, at, Floats::Frame)?;
        let dtype = column_dtype(field.data_type(), &values)?;
        let stype = match dtype {
            Dtype::Int64 | Dtype::Float32 | Dtype::Float64 => Stype::Numerical,
            Dtype::Bool => Stype::Categorical,
            Dtype::Date => Stype::Timestamp,
            _ if values::distinct(&values) <= 64 => Stype::Categorical,
            _ => Stype::Text,
        };
        columns.push(InferredColumn {
            name: field.name().clone(),
            dtype,
            stype,
            stype_before_key: None,
            received: ColumnType::from_arrow(field.name(), field.data_type())?,
            nullable: field.is_nullable(),
            unique: false,
            exact_number: None,
        });
        all_values.push(values);
    }
    check_name(name)?;
    for column in &columns {
        check_name(&column.name)?;
    }
    let primary_key = if rows == 0 {
        None
    } else {
        columns.iter().enumerate().find_map(|(at, c)| {
            let name = c.name.to_lowercase();
            let shaped = ["_id", "_key", "id", "key"]
                .iter()
                .any(|suffix| name.ends_with(suffix));
            (shaped
                && crate::assemble::can_identify(c.dtype)
                && values::distinct(&all_values[at]) as f64 >= rows as f64 * KEY_UNIQUENESS)
                .then(|| c.name.clone())
        })
    };
    let primary_key: Vec<_> = primary_key.into_iter().collect();
    let time = crate::infer::frame_time(&columns, &all_values, batches, rows, &primary_key)?;
    Ok(Read {
        rows,
        values: all_values,
        batches: batches.to_vec(),
        table: InferredTable {
            name: name.into(),
            source: source.clone(),
            columns,
            primary_key,
            left_out: vec![],
            unread_times: vec![],
            day_first: vec![],
            tied_keys: vec![],
            time_column: time.column.clone(),
            time,
            unique: vec![],
            declared: BTreeSet::new(),
        },
    })
}

fn check_name(name: &str) -> Result<(), CoreError> {
    if name.is_empty() {
        return Err(
            CoreError::new(Sentence::FrameNameEmpty, "a frame name is empty")
                .with_field("name", Detail::text(name)),
        );
    }
    let characters: Vec<String> = crate::assemble::NOT_IN_A_NAME
        .iter()
        .filter(|c| name.contains(**c))
        .map(char::to_string)
        .collect();
    if !characters.is_empty() {
        return Err(CoreError::new(
            Sentence::FrameNameCharacters,
            "a frame name contains an invalid character",
        )
        .with_field("name", Detail::text(name))
        .with_field("characters", texts(&characters)));
    }
    Ok(())
}

/// Native constructor distinctness over present numeric values, shared by
/// direct constant-target advice and frame schema/key inference.
pub fn present_distinct(batches: &[RecordBatch], column: usize) -> Result<usize, CoreError> {
    Ok(values::distinct(&values::column(
        batches,
        column,
        Floats::Frame,
    )?))
}

fn column_dtype(physical: &DataType, values: &[Option<Value>]) -> Result<Dtype, CoreError> {
    Ok(match physical {
        DataType::Boolean => Dtype::Bool,
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => Dtype::Int64,
        DataType::UInt64 => {
            if values
                .iter()
                .flatten()
                .any(|v| matches!(v, Value::Int(n) if *n > i128::from(i64::MAX)))
            {
                Dtype::String
            } else {
                Dtype::Int64
            }
        }
        DataType::Float32 => Dtype::Float32,
        DataType::Float64 => Dtype::Float64,
        DataType::Timestamp(_, _) => Dtype::Date,
        DataType::Utf8 => Dtype::String,
        _ => return Err(crate::not_yet("frame_native_family")),
    })
}

/// Storage inference without table-name, key or time-choice side effects.
pub fn storage_dtypes(schema: &Schema, batches: &[RecordBatch]) -> Result<Vec<Dtype>, CoreError> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(at, field)| {
            column_dtype(
                field.data_type(),
                &values::column(batches, at, Floats::Frame)?,
            )
        })
        .collect()
}
