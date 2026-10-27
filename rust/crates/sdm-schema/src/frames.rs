// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Primitive native-frame inference, distinct from SQL catalog/preview
//! inference. Shared value/equality and schema assembly rules stay shared.

pub(crate) mod decimal;
mod decimal_storage;

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use sdm_contract::data::Relation;
use sdm_contract::frames::{
    FrameAdoption, FrameInferred, FrameNomination, FrameSample, FrameSampleColumn, NativeDecimal,
    NativeDecimalColumn, TimeChoice,
};
use sdm_contract::ops::DescribedTable;
use sdm_contract::ops::{Dtype, SourceDescription, Stype};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Detail, Scalar, Sentence};

use crate::infer::{InferredColumn, InferredTable};
use crate::keys::{Counted, KEY_UNIQUENESS};
use crate::values::{self, Floats, Value};

/// One stage's borrowed physical columns and original logical Decimal facts.
/// Shared host admission has already checked and detached the Arrow buffers.
pub struct Columns<'a> {
    schema: &'a Schema,
    batches: &'a [RecordBatch],
    decimals: BTreeMap<usize, &'a NativeDecimal>,
}

impl<'a> Columns<'a> {
    pub fn new(
        schema: &'a Schema,
        batches: &'a [RecordBatch],
        decimals: &'a [NativeDecimalColumn],
    ) -> Result<Self, CoreError> {
        Ok(Self {
            schema,
            batches,
            decimals: decimal_storage::bindings(schema, decimals)?,
        })
    }

    fn read(&self, at: usize) -> Result<(Vec<Option<Value>>, Dtype), CoreError> {
        let native = if let Some(native) = self.decimals.get(&at) {
            Some(decimal_storage::read(self.batches, at, native)?)
        } else {
            decimal::read(self.batches, at, self.schema.field(at).data_type())
        };
        if let Some(native) = native {
            Ok((
                native
                    .values
                    .into_iter()
                    .map(|v| v.map(Value::Decimal))
                    .collect(),
                native.dtype,
            ))
        } else {
            let values = values::column(self.batches, at, Floats::Frame)?;
            let dtype = column_dtype(self.schema.field(at).data_type(), &values)?;
            Ok((values, dtype))
        }
    }

    fn values(&self, at: usize) -> Result<Vec<Option<Value>>, CoreError> {
        self.read(at).map(|(values, _)| values)
    }

    fn scales(&self) -> BTreeMap<usize, i32> {
        self.decimals
            .iter()
            .map(|(&at, native)| {
                let scale = match native {
                    NativeDecimal::Decimal128 { scale, .. }
                    | NativeDecimal::Decimal256 { scale, .. } => *scale,
                };
                (at, scale)
            })
            .collect()
    }
}

/// One table's primitive inference and call-local row statistics.
pub struct Read {
    pub table: InferredTable,
    values: Vec<Vec<Option<Value>>>,
    batches: Vec<RecordBatch>,
    bindings: Vec<Option<usize>>,
    decimal_scales: BTreeMap<usize, i32>,
    pub rows: usize,
}

impl Read {
    /// Count a key against the current admitted snapshot, not old metadata.
    pub fn count(&self, key: &[String]) -> Counted {
        if key.iter().any(|name| {
            self.table
                .columns
                .iter()
                .position(|c| &c.name == name)
                .is_none_or(|at| self.bindings[at].is_none())
        }) {
            return Counted {
                rows: 0,
                distinct: 0,
                missing: 0,
                example: None,
            };
        }
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
                        .map(|&at| values::scalar(batch.column(self.bindings[at]?).as_ref(), row))
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

    fn fields(&self, key: &[String], count: &Counted) -> Vec<(&'static str, Detail)> {
        let examples = count.example.clone().unwrap_or_default();
        let mut fields = vec![
            ("table", Detail::text(&self.table.name)),
            ("key", Detail::text(key.join("."))),
            ("columns", texts(key)),
            ("repeats", number(count.repeats())),
            ("rows", number(count.rows)),
            ("example", Detail::List(examples)),
        ];
        let scales: Vec<_> = key
            .iter()
            .map(|name| {
                self.table
                    .columns
                    .iter()
                    .position(|c| &c.name == name)
                    .and_then(|at| self.bindings[at])
                    .and_then(|at| self.decimal_scales.get(&at).copied())
                    .map(Scalar::Int32)
                    .unwrap_or(Scalar::Null)
            })
            .collect();
        if scales.iter().any(|scale| !matches!(scale, Scalar::Null)) {
            fields.push(("example_decimal_scales", Detail::List(scales)));
        }
        fields
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
                            && c.stype != Stype::Timestamp
                            && (self.bindings[at].is_none()
                                || self.values[at].iter().all(Option::is_some))
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
    final_notices_with_samples(reads, description, infer_links, true)
}

/// Final key advice only; native hosts count sample columns later, individually.
pub fn key_notices(
    reads: &[Read],
    description: &SourceDescription,
    infer_links: bool,
) -> Result<Vec<CoreWarning>, CoreError> {
    final_notices_with_samples(reads, description, infer_links, false)
}

fn final_notices_with_samples(
    reads: &[Read],
    description: &SourceDescription,
    infer_links: bool,
    samples: bool,
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
    if !samples {
        return Ok(notices);
    }
    for (read, table) in reads.iter().zip(&description.tables) {
        for (at, c) in table.columns.iter().enumerate() {
            if c.dtype != Dtype::String || c.stype != Stype::Categorical {
                continue;
            }
            let distinct = values::distinct(&read.values[at]);
            let present = read.values[at].iter().flatten().count();
            if let Some(notice) =
                sample_notice(&table.name, &c.name, distinct as u64, present as u64)
            {
                notices.push(notice);
            }
        }
    }
    Ok(notices)
}

fn sample_notice(table: &str, column: &str, distinct: u64, present: u64) -> Option<CoreWarning> {
    (distinct >= 32 && distinct as f64 >= present as f64 * 0.5).then(|| {
        CoreWarning::new(
            Sentence::FrameCategoricalSample,
            "frame size determined a column's meaning",
        )
        .with_fields(vec![
            ("table", Detail::text(table)),
            ("column", Detail::text(column)),
            ("distinct", number(distinct)),
            ("present", number(present)),
        ])
    })
}

/// A fixed core-selected sample column, observed only at its native loop position.
pub fn sample(
    column: FrameSampleColumn,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(FrameSample, Vec<CoreWarning>), CoreError> {
    sample_from(column, &Columns::new(schema, batches, &[])?)
}

pub fn sample_from(
    column: FrameSampleColumn,
    input: &Columns<'_>,
) -> Result<(FrameSample, Vec<CoreWarning>), CoreError> {
    let schema = input.schema;
    if column.column.dtype != Dtype::String || column.column.stype != Stype::Categorical {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "a sample column must have the original automatically categorical string meaning",
        ));
    }
    if schema.fields().len() > 1
        || schema
            .fields()
            .first()
            .is_some_and(|field| field.name() != &column.column.name)
    {
        return Err(CoreError::protocol(
            Sentence::InvalidMessage,
            "sample Arrow does not identify only its original column",
        ));
    }
    let at = schema
        .fields()
        .iter()
        .position(|field| field.name() == &column.column.name);
    let values = at
        .map(|at| input.values(at))
        .transpose()?
        .unwrap_or_default();
    let distinct = values::distinct(&values) as u64;
    let present = values.iter().flatten().count() as u64;
    let notices = sample_notice(&column.table, &column.column.name, distinct, present)
        .into_iter()
        .collect();
    Ok((
        FrameSample {
            column,
            exists: at.is_some(),
            distinct,
            present,
        },
        notices,
    ))
}

/// A fresh stated-key check/adoption after the metadata edit, without
/// renominating columns or reselecting the earlier inferred time choice.
pub fn adopt_stated(
    value: FrameInferred,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(FrameInferred, Vec<CoreWarning>), CoreError> {
    adopt_stated_from(value, &Columns::new(schema, batches, &[])?)
}

pub fn adopt_stated_from(
    mut value: FrameInferred,
    input: &Columns<'_>,
) -> Result<(FrameInferred, Vec<CoreWarning>), CoreError> {
    let read = observe_from(&value.table, value.time.clone(), input)?;
    let notices = read.hold_stated(&value.table.primary_key)?;
    value.table.duplicate_keys = Some(Scalar::UInt64(
        read.count(&value.table.primary_key).repeats(),
    ));
    let mut description = SourceDescription {
        tables: vec![value.table],
        links: vec![],
    };
    crate::assemble::mark_keys(&mut description);
    value.table = description.tables.remove(0);
    Ok((value, notices))
}

/// Whole-table frame inference, not preview-sized SQL inference. This first
/// consumer accepts NumPy/pandas masked numeric/Boolean arrays, native
/// timestamp arrays, built-in pandas text and exact native Arrow Decimals.
/// Native conversion has already encoded missingness in Arrow validity.
fn read_columns(name: &str, source: &Relation, input: &Columns<'_>) -> Result<Read, CoreError> {
    let schema = input.schema;
    let batches = input.batches;
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let mut columns = Vec::new();
    let mut all_values = Vec::new();
    for (at, field) in schema.fields().iter().enumerate() {
        let (values, dtype) = input.read(at)?;
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
    let time = TimeChoice::default();
    Ok(Read {
        rows,
        values: all_values,
        batches: batches.to_vec(),
        bindings: (0..schema.fields().len()).map(Some).collect(),
        decimal_scales: input.scales(),
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

/// The complete initial column meanings and fixed nominee, before adoption.
pub fn nominate(
    name: &str,
    source: &Relation,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(FrameNomination, Vec<CoreWarning>), CoreError> {
    nominate_from(name, source, &Columns::new(schema, batches, &[])?)
}

pub fn nominate_from(
    name: &str,
    source: &Relation,
    input: &Columns<'_>,
) -> Result<(FrameNomination, Vec<CoreWarning>), CoreError> {
    let mut read = read_columns(name, source, input)?;
    let nominated_key = std::mem::take(&mut read.table.primary_key);
    let count = read.count(&nominated_key);
    let notices = if count.missing > 0 {
        vec![
            CoreWarning::new(
                Sentence::KeyNoValueColumn,
                "an inferred key has missing values",
            )
            .with_fields(vec![
                ("table", Detail::text(name)),
                ("column", Detail::text(&nominated_key[0])),
                ("missing", number(count.missing)),
            ]),
        ]
    } else {
        vec![]
    };
    Ok((
        FrameNomination {
            table: crate::assemble::described_table(read.table, Some(Scalar::UInt64(0))),
            nominated_key,
        },
        notices,
    ))
}

/// Convert settled metadata back to the shared inference assembly value.
/// This performs no nomination, value inference, key check, or time choice.
pub fn inferred(table: &DescribedTable, time: TimeChoice) -> InferredTable {
    InferredTable {
        name: table.name.clone(),
        source: table.source.clone(),
        columns: table
            .columns
            .iter()
            .map(|c| InferredColumn {
                name: c.name.clone(),
                dtype: c.dtype,
                stype: c.stype,
                stype_before_key: c.stype_before_key,
                received: c.received.clone(),
                nullable: c.nullable,
                unique: c.unique,
                exact_number: c.exact_number,
            })
            .collect(),
        primary_key: table.primary_key.clone(),
        time_column: table.time_column.clone(),
        time,
        left_out: vec![],
        unread_times: vec![],
        day_first: vec![],
        tied_keys: vec![],
        unique: vec![],
        declared: BTreeSet::new(),
    }
}

/// Fresh values bound by original column name; absent differs from all-null.
pub fn observe(
    table: &DescribedTable,
    time: TimeChoice,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<Read, CoreError> {
    observe_from(table, time, &Columns::new(schema, batches, &[])?)
}

pub fn observe_from(
    table: &DescribedTable,
    time: TimeChoice,
    input: &Columns<'_>,
) -> Result<Read, CoreError> {
    let schema = input.schema;
    let batches = input.batches;
    let bindings: Vec<_> = table
        .columns
        .iter()
        .map(|c| schema.fields().iter().position(|f| f.name() == &c.name))
        .collect();
    let values = bindings
        .iter()
        .map(|at| match at {
            Some(at) => input.values(*at),
            None => Ok(vec![]),
        })
        .collect::<Result<_, _>>()?;
    Ok(Read {
        table: inferred(table, time),
        rows: batches.iter().map(RecordBatch::num_rows).sum(),
        values,
        bindings,
        batches: batches.to_vec(),
        decimal_scales: input.scales(),
    })
}

/// Adopt the fixed nominee using current rows, then fix repeat count/advice.
pub fn adopt(
    nomination: FrameNomination,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<(FrameAdoption, Vec<CoreWarning>), CoreError> {
    adopt_from(nomination, &Columns::new(schema, batches, &[])?)
}

pub fn adopt_from(
    mut nomination: FrameNomination,
    input: &Columns<'_>,
) -> Result<(FrameAdoption, Vec<CoreWarning>), CoreError> {
    nomination.table.primary_key = nomination.nominated_key;
    nomination.table.frame_key_form = (!nomination.table.primary_key.is_empty())
        .then_some(sdm_contract::frames::FrameKeyForm::Scalar);
    let read = observe_from(&nomination.table, TimeChoice::default(), input)?;
    let count = read.count(&nomination.table.primary_key);
    nomination.table.duplicate_keys =
        Some(Scalar::UInt64(if nomination.table.primary_key.is_empty() {
            0
        } else {
            count.repeats()
        }));
    let notices = if !nomination.table.primary_key.is_empty() && count.repeats() > 0 {
        vec![read.repeated(&nomination.table.primary_key, &count)]
    } else {
        vec![]
    };
    let mut description = SourceDescription {
        tables: vec![nomination.table],
        links: vec![],
    };
    crate::assemble::mark_keys(&mut description);
    Ok((
        FrameAdoption {
            table: description.tables.remove(0),
        },
        notices,
    ))
}

/// Read only original timestamp candidates after any repeat advice effects.
pub fn choose_time(
    adopted: FrameAdoption,
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<FrameInferred, CoreError> {
    choose_time_from(adopted, &Columns::new(schema, batches, &[])?)
}

pub fn choose_time_from(
    adopted: FrameAdoption,
    input: &Columns<'_>,
) -> Result<FrameInferred, CoreError> {
    let batches = input.batches;
    let read = observe_from(&adopted.table, TimeChoice::default(), input)?;
    for (at, c) in read.table.columns.iter().enumerate() {
        if c.stype == Stype::Timestamp
            && !read.table.primary_key.contains(&c.name)
            && !crate::names::is_load_stamp(&c.name)
            && !crate::names::is_birth_date(&c.name)
            && read.bindings[at].is_none()
        {
            return Err(CoreError::new(
                Sentence::DirectSchemaMissing,
                "a timestamp candidate is absent from the current frame",
            )
            .with_field("name", Detail::text(&c.name)));
        }
    }
    // Preserve original ordinals/meanings while reading current physical arrays.
    let present: Vec<_> = read
        .bindings
        .iter()
        .enumerate()
        .filter_map(|(at, binding)| binding.map(|binding| (at, binding)))
        .collect();
    let columns: Vec<_> = present
        .iter()
        .map(|(at, _)| read.table.columns[*at].clone())
        .collect();
    let presence: Vec<_> = present
        .iter()
        .map(|(at, _)| read.values[*at].clone())
        .collect();
    let projection: Vec<_> = present.iter().map(|(_, binding)| *binding).collect();
    let projected: Vec<_> = batches
        .iter()
        .map(|b| {
            b.project(&projection).map_err(|_| {
                CoreError::protocol(
                    Sentence::InvalidMessage,
                    "frame time observation projection failed",
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let time = crate::infer::frame_time(
        &columns,
        &presence,
        &projected,
        read.rows,
        &read.table.primary_key,
    )?;
    let mut table = adopted.table;
    table.time_column = time.column.clone();
    Ok(FrameInferred { table, time })
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
