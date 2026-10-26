// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The prediction request's payload, laid out as the reference's engine
//! lays it out (`predict_request_to_json`): the task, the schema, the
//! context and predict halves, each an instance table and the related
//! tables sampled for it, the output fields, the inference options and the
//! metadata. The payload is `sdm-wire`'s typed one
//! ([`sdm_wire::payload::StatePayload`]), whose derived schema is the
//! contract's locked definition of it (T1.28w); each cell is a JSON value
//! whose number is the token written ([`sdm_wire::json::Value`]).
//!
//! Each column's wire dtype comes from the Arrow type sampled, by the
//! reference's rule for that type (`_dtype_name`): integers as `int64`,
//! floats at their width, text as `string`, booleans, times with no zone
//! as `timestamp[us]` (a time of day too, each cell null, as the
//! reference's engine reads it as no time), and a `DECIMAL(38, s)` by what it holds
//! (`_decimal_dtype_name`): a fraction's scale as `float64`, whole numbers
//! as `int64` where every value fits, else as `string`, its digits kept;
//! every cell is written under its column's dtype (`_cell_json_value`): an
//! integer past the JSON-safe range as its digits, a missing value or a
//! float NaN as null, a time as its UTC wall clock to the microsecond, a
//! decimal float64 does not hold exactly refused as the caller's error, as
//! the reference refuses it. Any other type is refused by name.
//!
//! From T1.28a the layout is the task's (`_task_spec`, `_target_dtype`): a
//! regression's target at the labels' type, a binary task's as booleans
//! with the classes `false` and `true`, a multiclass task's as text where
//! the labels are booleans (`'False'`, `'True'`) and its classes in the
//! order the context first holds them; the outputs the task answers, the
//! regression's statistic where one was chosen, the run mode and the
//! inference options ([`crate::options`]). And the context that answers
//! the question is taken out, as the reference's client takes it out
//! (`_drop_context_that_answers_the_question`): a context row whose entity
//! is one predicted for, at the same anchor for a windowed target and at
//! any for a static one, with the related rows that hang off it; how many
//! is [`Built::answering`], which the core gives as its notice.

use std::collections::BTreeSet;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use sdm_contract::ops::Stype;
use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_engine::labels::Labels;
use sdm_wire::json::Value;
use sdm_wire::payload::{
    ClassTarget, ColumnSpec, Dtype, Format, Half, Inference, Metadata, Named, Output,
    PayloadSchema, PrimaryKey, Relationship, Rows, StatePayload, TableSchema, Target, Task,
};

use crate::options::{Kind, Options, RunMode};

/// The column of every table that names the example a row belongs to.
pub const INSTANCE_ID: &str = "instance_id";
/// The key a related table without one of its own is given.
pub const NODE_ID: &str = "__node_id";
/// The instance table's column naming each example's entity, numbered.
pub const ENTITY_REFERENCE: &str = "__kumo_entity_ref";
/// The instance table's column holding each example's anchor time.
pub const ANCHOR_TIME: &str = "__kumo_anchor_time";
/// The instance table's label column.
pub const TARGET: &str = "TARGET";
/// The name the request's table of examples goes by, as a refusal of one
/// of its cells names it (the reference's `_serialize_frame`).
pub const INSTANCE_TABLE: &str = "instance_table";
/// The names the wire gives its own columns, which a column of the data
/// may not take (the client's `_RESERVED`).
pub const RESERVED: [&str; 2] = [INSTANCE_ID, ANCHOR_TIME];
/// `pd.Timestamp.min` in nanoseconds: an anchor every example has at this
/// value is no anchor at all.
const NO_ANCHOR: i64 = -9_223_372_036_854_775_807;
/// The largest integer a JSON number carries exactly in every reader.
const JSON_SAFE: i128 = 9_007_199_254_740_991;
/// The largest payload the endpoint takes, in bytes (`_MAX_SIZE`), counted
/// as the bytes the payload is written as and sent, UTF-8 (register PD-2),
/// not the reference's `json.dumps` count, which escapes every character
/// past ASCII.
pub const PAYLOAD_BYTES: usize = 30 * 1024 * 1024;

/// One column of a sampled table: as sampled, or a text feature as the
/// reference's sampler sends it, each value its words
/// (`sdm_engine::words`), written as a `stringlist` column.
#[derive(Debug, Clone)]
pub enum SampledColumn {
    Arrow(ArrayRef),
    Words(Vec<Vec<String>>),
}

impl SampledColumn {
    fn len(&self) -> usize {
        match self {
            SampledColumn::Arrow(array) => array.len(),
            SampledColumn::Words(words) => words.len(),
        }
    }
}

/// One table of the sampled subgraph: its rows in the order sampled, each
/// with the example it was sampled for.
#[derive(Debug, Clone)]
pub struct SampledTable {
    pub name: String,
    /// The table's key, for an entity table.
    pub primary_key: Option<String>,
    /// The example each row was sampled for.
    pub batch: Vec<usize>,
    /// The columns sent, in order, each as sampled, its rows aligned with
    /// `batch`.
    pub columns: Vec<(String, SampledColumn)>,
    /// The semantic type of each feature column the table sends.
    pub stypes: Vec<(String, Stype)>,
}

/// A foreign key between two sampled tables: for each row of `src`, the
/// row of `dst` it refers to, if any. The links of one foreign key may come
/// more than once, as the reference's sampled edge types of both
/// directions. Their partial maps are consolidated by sampled source-row
/// position: the first valid mapping supplies that occurrence's key.
#[derive(Debug, Clone)]
pub struct SampledLink {
    pub src: String,
    pub fkey: String,
    pub dst: String,
    pub pairs: Vec<(usize, usize)>,
}

/// The values of one lagged target column (`__kumo_arl<k>__`, T2.9), one
/// per example of the request, a missing one `None`: an aggregation's at
/// float32, or a condition's.
#[derive(Debug, Clone, PartialEq)]
pub enum LagValues {
    Float(Vec<Option<f32>>),
    Bool(Vec<Option<bool>>),
    /// A condition's column that no example of the whole request has a
    /// value in, so many rows: `float64` nulls, as pandas types a column of
    /// booleans with nothing in it (T2.9 fix 3: decided over the request,
    /// not over one call's rows).
    Empty(usize),
}

/// One lagged target column of the instance table (T2.9), as the reference's
/// engine adds it to the task (`add_lagged_target`) and lays it out
/// (`_instance_dataframe`): after the label, with no semantic type; an
/// aggregation's as `float64`, its float32s as doubles; a condition's as
/// `bool`, or `float64` where no example of the whole request has a value
/// ([`LagValues::Empty`]), whatever rows one call of it holds; a missing
/// value null.
#[derive(Debug, Clone, PartialEq)]
pub struct LagColumn {
    pub name: String,
    pub values: LagValues,
}

impl LagColumn {
    /// The column at `rows`, its type kept: a call's slice of the
    /// request's column (T2.9 fix 3), a condition's `bool` though the
    /// call's rows hold no value.
    pub fn rows(&self, rows: &[usize]) -> LagColumn {
        fn picked<T: Copy>(values: &[Option<T>], rows: &[usize]) -> Vec<Option<T>> {
            rows.iter()
                .map(|&at| values.get(at).copied().flatten())
                .collect()
        }
        LagColumn {
            name: self.name.clone(),
            values: match &self.values {
                LagValues::Float(v) => LagValues::Float(picked(v, rows)),
                LagValues::Bool(v) => LagValues::Bool(picked(v, rows)),
                LagValues::Empty(_) => LagValues::Empty(rows.len()),
            },
        }
    }
}

/// What a request is built from.
#[derive(Debug, Clone)]
pub struct Sampled {
    /// The examples with a label, then those to predict.
    pub num_train: usize,
    pub num_test: usize,
    /// Each example's anchor time, nanoseconds since the epoch.
    pub anchors: Vec<i64>,
    /// The task's kind.
    pub kind: Kind,
    /// The labels of the examples with one, in order, as the reference's
    /// executor gives them.
    pub labels: Labels,
    /// Whether the target is a window over time: then only a context row at
    /// a predicted row's own anchor answers its question; a static
    /// target's answers it at any anchor (`windowed`).
    pub windowed: bool,
    /// The entity table first, then each table sampled, in the order
    /// sampled.
    pub tables: Vec<SampledTable>,
    pub links: Vec<SampledLink>,
    pub entity_table: String,
    /// The query, as the reference writes it back.
    pub source_query: String,
    pub run_mode: RunMode,
    /// Whether the model is told each example's anchor time
    /// (`use_prediction_time`, T2.11).
    pub use_prediction_time: bool,
    /// Whether the outputs name `embeddings` (`return_embeddings`,
    /// `_output_fields`, T2.11).
    pub return_embeddings: bool,
    /// The inference options, read for this task ([`crate::options`]).
    pub options: Options,
    /// The regression's statistic the caller chose.
    pub statistic: Option<String>,
    /// How many candidates the call asked be kept (`top_k`), written into
    /// the task as the engine writes it (T2.8).
    pub top_k: Option<u32>,
    /// The most rows any call of the request this payload is one call of
    /// scores (`_the_largest_batch`, `main`'s multi-call `predict`): the
    /// size a refusal of a payload past the endpoint's limit names as the
    /// one to ask below.
    pub rows_per_call: usize,
    /// The lagged target columns (`lag_timesteps`, T2.9), in order; none
    /// where the call asks for none.
    pub lags: Vec<LagColumn>,
    /// The text a float's class is spelled in: the shortest digits that
    /// read back to the double, laid out as the reference's `str` lays them
    /// out (`_class_label`). The parser crate holds it; the caller passes
    /// it in.
    pub float_text: fn(f64) -> String,
}

/// A payload built: its JSON text, the bytes the request sends, and how
/// many context examples answered the question and were taken out.
#[derive(Debug, Clone)]
pub struct Built {
    pub bytes: Vec<u8>,
    pub answering: usize,
}

fn invalid(sentence: Sentence, message: &str, column: &str) -> CoreError {
    CoreError::new(sentence, message).with_field("column", Detail::text(column))
}

/// A value no request holds, with the value as Python writes a float
/// (`inf`, `-inf`) for the host's words (the reference's `_NON_FINITE_MSG`).
fn not_finite(column: &str, value: f64) -> CoreError {
    let shown = if value.is_nan() {
        "nan"
    } else if value > 0.0 {
        "inf"
    } else {
        "-inf"
    };
    invalid(
        Sentence::NotFinite,
        "a value is not finite, and a request holds finite numbers or nulls",
        column,
    )
    .with_field("field", Detail::text("request"))
    .with_field("reason", Detail::text("not_finite"))
    .with_field("value", Detail::text(shown))
}

fn count(n: usize) -> Detail {
    Detail::Scalar(Scalar::UInt64(u64::try_from(n).unwrap_or(u64::MAX)))
}

fn internal(message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message)
}

/// The wire dtype of a column of `data_type`, or a refusal naming the rule:
/// the types whose dtype does not depend on the values.
fn type_dtype(data_type: &DataType) -> Result<Dtype, CoreError> {
    Ok(match data_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Dtype::Int64,
        DataType::Float32 => Dtype::Float32,
        DataType::Float64 => Dtype::Float64,
        DataType::Utf8 => Dtype::String,
        DataType::Boolean => Dtype::Bool,
        // A time of day is a timestamp column the reference's engine reads
        // as holding no time: every cell null ([`cell`]).
        DataType::Timestamp(_, None) | DataType::Time32(_) | DataType::Time64(_) => {
            Dtype::TimestampMicros
        }
        _ => return Err(sdm_engine::not_yet("wire_dtype")),
    })
}

/// The wire dtype of `array`, as the reference names a column by what it
/// holds (`_dtype_name`, for a decimal `_decimal_dtype_name`'s Arrow
/// branch), at any precision: a decimal with a scale as `float64`; a whole
/// one of up to 18 digits as `int64`; a wider whole one as `int64` where
/// every value fits, and as `string` where one does not. A decimal of
/// negative scale, which no source this core reads hands over, is refused
/// by name (`wire_dtype`, register P1B-3).
pub fn wire_dtype(array: &ArrayRef) -> Result<Dtype, CoreError> {
    match array.data_type() {
        DataType::Decimal128(_, scale) if *scale > 0 => Ok(Dtype::Float64),
        DataType::Decimal128(precision, 0) if *precision <= 18 => Ok(Dtype::Int64),
        DataType::Decimal128(_, 0) => {
            let values = array.as_primitive::<Decimal128Type>();
            let fits = values.iter().flatten().all(|v| i64::try_from(v).is_ok());
            Ok(if fits { Dtype::Int64 } else { Dtype::String })
        }
        other => type_dtype(other),
    }
}

/// The wire dtype of a key's column (the entity table's key, and the
/// instance table's reference to it), as the reference names a whole
/// decimal by what it holds (`_decimal_dtype_name`), at any precision: up
/// to 18 digits as `int64`; more as `int64` where every value fits and
/// `string` where one does not. A decimal key with a scale, which the
/// reference cannot sample, is `string`, each value its exact decimal text
/// with no trailing zeros, never a double, which could make two keys one
/// (T1.32a fix round 1, P1B-33). No question sums a key, so the declared
/// precision a decimal feature's sum is held to (`wire_dtype`) never
/// applies to one.
pub fn key_dtype(array: &ArrayRef) -> Result<Dtype, CoreError> {
    match array.data_type() {
        DataType::Decimal128(precision, 0) if *precision <= 18 => Ok(Dtype::Int64),
        DataType::Decimal128(_, 0) => {
            let values = array.as_primitive::<Decimal128Type>();
            let fits = values.iter().flatten().all(|v| i64::try_from(v).is_ok());
            Ok(if fits { Dtype::Int64 } else { Dtype::String })
        }
        DataType::Decimal128(_, scale) if *scale > 0 => Ok(Dtype::String),
        _ => wire_dtype(array),
    }
}

/// A decimal's exact text: `value` ten-to-the-`scale`ths, with no trailing
/// zeros after the point (and no point for a whole one).
fn decimal_text(value: i128, scale: i8) -> String {
    let digits = value.unsigned_abs().to_string();
    let sign = if value < 0 { "-" } else { "" };
    if scale <= 0 {
        let zeros = "0".repeat(usize::from(scale.unsigned_abs()));
        return if value == 0 {
            "0".to_owned()
        } else {
            format!("{sign}{digits}{zeros}")
        };
    }
    let scale = usize::from(scale.unsigned_abs());
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        if whole.trim_start_matches('0').is_empty() {
            return "0".to_owned();
        }
        format!("{sign}{whole}")
    } else {
        format!("{sign}{whole}.{fraction}")
    }
}

/// An integer cell: past the JSON-safe range, its digits as text; past
/// int64, refused, as the reference refuses it.
fn int_cell(value: i128, column: &str) -> Result<Value, CoreError> {
    if value < i128::from(i64::MIN) || value > i128::from(i64::MAX) {
        // The range, for the host's words (the reference's
        // `_cell_json_value`), never the value (decision H17); the table and
        // row are its caller's.
        return Err(invalid(
            Sentence::PastInt64,
            "a value is outside the range int64 holds",
            column,
        )
        .with_field("field", Detail::text("request"))
        .with_field("reason", Detail::text("past_int64"))
        .with_field("dtype", Detail::text("int64")));
    }
    if value.abs() > JSON_SAFE {
        return Ok(Value::Text(value.to_string()));
    }
    Ok(Value::int(value as i64))
}

fn float_cell(value: f64, column: &str) -> Result<Value, CoreError> {
    if value.is_nan() {
        return Ok(Value::Null);
    }
    Value::float(value).ok_or_else(|| not_finite(column, value))
}

/// A decimal's unscaled `value` at `scale` as a float64 cell: the nearest
/// double to its exact value, rounded once, whatever its digits (T2.0c:
/// the reference since `main`'s eb6c68eb sends every decimal feature as a
/// float64, where it refused one a double does not hold exactly; its cast
/// can be one unit in the last place off, where this is the nearest).
fn decimal_float(value: i128, scale: i8, column: &str) -> Result<Value, CoreError> {
    let float: f64 = format!("{value}e{}", -i32::from(scale))
        .parse()
        .map_err(|_| internal("a decimal's text is not a number"))?;
    Value::float(float).ok_or_else(|| not_finite(column, float))
}

/// `ns` nanoseconds since the epoch as the reference writes a time:
/// `YYYY-MM-DDTHH:MM:SS.ffffffZ`, the nanoseconds past the microsecond
/// left out.
pub fn iso_microseconds(ns: i64) -> String {
    let micros = ns.div_euclid(1_000);
    let seconds = micros.div_euclid(1_000_000);
    let fraction = micros.rem_euclid(1_000_000);
    let days = seconds.div_euclid(86_400);
    let of_day = seconds.rem_euclid(86_400);
    // Howard Hinnant's `civil_from_days`.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{fraction:06}Z",
        of_day / 3_600,
        of_day / 60 % 60,
        of_day % 60
    )
}

/// The time `value` in `unit` as nanoseconds, or a refusal: the reference
/// holds every time at nanoseconds.
fn to_ns(value: i64, unit: TimeUnit, column: &str) -> Result<i64, CoreError> {
    let per = match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    value.checked_mul(per).ok_or_else(|| {
        CoreError::new(
            Sentence::ColumnTimePastNanoseconds,
            "a time is outside what nanoseconds since the epoch hold",
        )
        .with_field("column", Detail::text(column))
    })
}

/// Row `row` of `array`, written under its column's wire dtype `dtype`.
fn cell(array: &ArrayRef, row: usize, column: &str, dtype: Dtype) -> Result<Value, CoreError> {
    if array.is_null(row) {
        return Ok(Value::Null);
    }
    macro_rules! int {
        ($t:ty) => {
            int_cell(i128::from(array.as_primitive::<$t>().value(row)), column)
        };
    }
    match array.data_type() {
        DataType::Int8 => int!(Int8Type),
        DataType::Int16 => int!(Int16Type),
        DataType::Int32 => int!(Int32Type),
        DataType::Int64 => int!(Int64Type),
        DataType::UInt8 => int!(UInt8Type),
        DataType::UInt16 => int!(UInt16Type),
        DataType::UInt32 => int!(UInt32Type),
        DataType::UInt64 => int!(UInt64Type),
        DataType::Float32 => float_cell(
            f64::from(array.as_primitive::<Float32Type>().value(row)),
            column,
        ),
        DataType::Float64 => float_cell(array.as_primitive::<Float64Type>().value(row), column),
        DataType::Utf8 => Ok(Value::text(array.as_string::<i32>().value(row))),
        DataType::Boolean => Ok(Value::Bool(array.as_boolean().value(row))),
        DataType::Timestamp(unit, None) => {
            let value = match unit {
                TimeUnit::Second => array.as_primitive::<TimestampSecondType>().value(row),
                TimeUnit::Millisecond => {
                    array.as_primitive::<TimestampMillisecondType>().value(row)
                }
                TimeUnit::Microsecond => {
                    array.as_primitive::<TimestampMicrosecondType>().value(row)
                }
                TimeUnit::Nanosecond => array.as_primitive::<TimestampNanosecondType>().value(row),
            };
            Ok(Value::Text(iso_microseconds(to_ns(value, *unit, column)?)))
        }
        // A time of day, which the reference's engine reads as no time.
        DataType::Time32(_) | DataType::Time64(_) => Ok(Value::Null),
        DataType::Decimal128(_, scale) => {
            let value = array.as_primitive::<Decimal128Type>().value(row);
            match dtype {
                // A whole decimal's own digits (`_decimal_text`).
                Dtype::String => Ok(Value::Text(decimal_text(value, *scale))),
                Dtype::Int64 => int_cell(value, column),
                Dtype::Float64 => decimal_float(value, *scale, column),
                _ => Err(internal("a decimal's dtype is not one a decimal takes")),
            }
        }
        _ => Err(sdm_engine::not_yet("wire_dtype")),
    }
}

/// An integer key cell as the float a column of keys and nulls holds it
/// at: its value rounded to the nearest double, as pandas reads such a
/// column.
fn float_key(cell: &Value) -> Result<Value, CoreError> {
    let value = match cell {
        Value::Number(token) => token
            .parse::<i64>()
            .map_err(|_| internal("an integer key is not an integer"))?
            as f64,
        Value::Text(digits) => digits
            .parse::<i128>()
            .map_err(|_| internal("an integer key's digits do not read"))?
            as f64,
        _ => return Err(internal("an integer key is not an integer")),
    };
    Value::float(value).ok_or_else(|| internal("an integer key is not finite"))
}

/// `prefix`, or `prefix_1`, `prefix_2` and on, whichever `occupied` does not
/// hold first.
fn unique_name(occupied: &[String], prefix: &str) -> String {
    if !occupied.iter().any(|o| o == prefix) {
        return prefix.to_owned();
    }
    (1..)
        .map(|n| format!("{prefix}_{n}"))
        .find(|name| !occupied.contains(name))
        .unwrap_or_default()
}

/// A related table as sent: its name, its frame, its key column and the
/// semantic types its schema gives.
type Related = (String, Frame, String, Vec<(String, Stype)>);

/// One column of a frame: its name, wire dtype and cells.
struct FrameColumn {
    name: String,
    dtype: Dtype,
    cells: Vec<Value>,
}

/// A table as the reference's frame of it: columns in order, one cell per
/// row each.
struct Frame {
    columns: Vec<FrameColumn>,
    rows: usize,
}

impl Frame {
    fn names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    fn column(&self, name: &str) -> Option<&FrameColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// The rows `keep` selects, as `arrays` JSON, under each column's
    /// dtype; `drop` names a column left out.
    fn table(&self, keep: impl Fn(usize) -> bool, drop: Option<&str>) -> Rows {
        let columns: Vec<&FrameColumn> = self
            .columns
            .iter()
            .filter(|c| Some(c.name.as_str()) != drop)
            .collect();
        let rows = (0..self.rows)
            .filter(|row| keep(*row))
            .map(|row| columns.iter().map(|c| c.cells[row].clone()).collect())
            .collect();
        Rows {
            format: Format::Arrays,
            columns: columns.iter().map(|c| c.name.clone()).collect(),
            rows,
        }
    }
}

fn int_column(name: &str, values: impl Iterator<Item = usize>) -> FrameColumn {
    FrameColumn {
        name: name.to_owned(),
        dtype: Dtype::Int64,
        cells: values.map(|v| Value::int(v as i64)).collect(),
    }
}

fn arrow_column(
    name: &str,
    array: &ArrayRef,
    rows: &[usize],
    key: bool,
) -> Result<FrameColumn, CoreError> {
    let dtype = if key {
        key_dtype(array)?
    } else {
        wire_dtype(array)?
    };
    Ok(FrameColumn {
        name: name.to_owned(),
        dtype,
        cells: rows
            .iter()
            .map(|&row| cell(array, row, name, dtype).map_err(|e| e.with_field("row", count(row))))
            .collect::<Result<_, _>>()?,
    })
}

/// A column of `rows`: as sampled, or, for words, a `stringlist` of them,
/// as the reference types a column holding lists (`_list_dtype_name`;
/// a column of no rows holds no list, and is `string`).
fn sampled_column(
    name: &str,
    column: &SampledColumn,
    rows: &[usize],
    key: bool,
) -> Result<FrameColumn, CoreError> {
    match column {
        SampledColumn::Arrow(array) => arrow_column(name, array, rows, key),
        SampledColumn::Words(words) => Ok(FrameColumn {
            name: name.to_owned(),
            dtype: if rows.is_empty() {
                Dtype::String
            } else {
                Dtype::StringList
            },
            cells: rows
                .iter()
                .map(|&row| Value::texts(words[row].iter().map(String::as_str)))
                .collect(),
        }),
    }
}

/// A table's schema: each column's dtype, its semantic type where one is
/// given, and not nullable where it is part of the key.
fn table_schema(frame: &Frame, stypes: &[(String, Stype)], key: &[String]) -> TableSchema {
    let columns = frame
        .columns
        .iter()
        .map(|c| {
            let spec = ColumnSpec {
                dtype: c.dtype,
                stype: stypes
                    .iter()
                    .rev()
                    .find(|(n, _)| *n == c.name)
                    .map(|(_, stype)| *stype),
                nullable: key.contains(&c.name).then_some(false),
            };
            (c.name.clone(), spec)
        })
        .collect();
    let primary_key = match key {
        [one] => PrimaryKey::One(one.clone()),
        several => PrimaryKey::Several(several.to_vec()),
    };
    TableSchema {
        columns: Named(columns),
        primary_key,
    }
}

/// The label column of the context, as the reference writes it for `kind`
/// (`_target_dtype`, `_target_json_value`): its dtype and one cell per
/// label; and a classification's classes, in the order the context first
/// holds them (`pd.unique`), as their text (`_class_label`). A multiclass
/// float target's labels are its float32s as doubles, and its classes the
/// distinct values, a negative and a positive zero one class spelled as
/// first met, each as `float_text` spells its double.
fn target(
    kind: Kind,
    labels: &Labels,
    float_text: fn(f64) -> String,
) -> Result<(Dtype, Vec<Value>, Vec<String>), CoreError> {
    let classes = |texts: Vec<String>| {
        let mut seen = BTreeSet::new();
        texts
            .into_iter()
            .filter(|t| seen.insert(t.clone()))
            .collect::<Vec<_>>()
    };
    Ok(match (kind, labels) {
        (Kind::Binary, Labels::Bool(values)) => (
            Dtype::Bool,
            values.iter().map(|v| Value::Bool(*v)).collect(),
            vec!["false".to_owned(), "true".to_owned()],
        ),
        (Kind::Binary, _) => return Err(internal("a binary task's labels are not booleans")),
        (Kind::Multiclass, Labels::Bool(values)) => {
            let texts: Vec<String> = values
                .iter()
                .map(|v| if *v { "True" } else { "False" }.to_owned())
                .collect();
            (
                Dtype::String,
                texts.iter().map(|t| Value::text(t)).collect(),
                classes(texts),
            )
        }
        (Kind::Multiclass, Labels::Text(values)) => (
            Dtype::String,
            values.iter().map(|t| Value::text(t)).collect(),
            classes(values.clone()),
        ),
        (Kind::Multiclass, Labels::Int(values)) => (
            Dtype::Int64,
            values
                .iter()
                .map(|v| int_cell(i128::from(*v), TARGET))
                .collect::<Result<_, _>>()?,
            classes(values.iter().map(i64::to_string).collect()),
        ),
        (Kind::Multiclass, Labels::Float(values)) => {
            let mut seen = BTreeSet::new();
            (
                Dtype::Float32,
                values
                    .iter()
                    .map(|v| float_cell(f64::from(*v), TARGET))
                    .collect::<Result<_, _>>()?,
                values
                    .iter()
                    // Adding zero makes a negative zero positive.
                    .filter(|v| seen.insert((**v + 0.0).to_bits()))
                    .map(|v| float_text(f64::from(*v)))
                    .collect(),
            )
        }
        (Kind::Regression, Labels::Float(values)) => (
            Dtype::Float32,
            values
                .iter()
                .map(|v| float_cell(f64::from(*v), TARGET))
                .collect::<Result<_, _>>()?,
            vec![],
        ),
        (Kind::Regression, Labels::Int(values)) => (
            Dtype::Int64,
            values
                .iter()
                .map(|v| int_cell(i128::from(*v), TARGET))
                .collect::<Result<_, _>>()?,
            vec![],
        ),
        (Kind::Regression, Labels::Bool(values)) => (
            Dtype::Bool,
            values.iter().map(|v| Value::Bool(*v)).collect(),
            vec![],
        ),
        (Kind::Regression, Labels::Text(_)) => {
            return Err(internal("a regression's labels are text"));
        }
    })
}

/// A lagged target column of the instance table ([`LagColumn`]).
fn lag_column(lag: &LagColumn, examples: usize) -> Result<FrameColumn, CoreError> {
    let (dtype, cells): (Dtype, Vec<Value>) = match &lag.values {
        LagValues::Float(values) => (
            Dtype::Float64,
            values
                .iter()
                .map(|v| match v {
                    Some(v) => float_cell(f64::from(*v), &lag.name),
                    None => Ok(Value::Null),
                })
                .collect::<Result<_, _>>()?,
        ),
        LagValues::Bool(values) => (
            Dtype::Bool,
            values
                .iter()
                .map(|v| v.map_or(Value::Null, Value::Bool))
                .collect(),
        ),
        LagValues::Empty(rows) => (Dtype::Float64, vec![Value::Null; *rows]),
    };
    if cells.len() != examples {
        return Err(internal("a lagged column does not align with the examples"));
    }
    Ok(FrameColumn {
        name: lag.name.clone(),
        dtype,
        cells,
    })
}

/// The instance ids of the context examples that answer the question: a
/// context row whose entity is a predicted row's, at that row's anchor
/// for a windowed target and at any anchor for a static one
/// (`_entities_at_anchors`, `_instances_matching`). A windowed target with
/// no anchor column takes none out.
fn answering(
    instance: &Frame,
    num_train: usize,
    windowed: bool,
    anchor: Option<&str>,
    reference: &str,
) -> Result<BTreeSet<usize>, CoreError> {
    let entities = &instance
        .column(reference)
        .ok_or_else(|| internal("the instance table has no entity column"))?
        .cells;
    let anchors = match (windowed, anchor) {
        (true, None) => return Ok(BTreeSet::new()),
        (true, Some(name)) => Some(
            &instance
                .column(name)
                .ok_or_else(|| internal("the instance table has no anchor column"))?
                .cells,
        ),
        (false, _) => None,
    };
    let key = |row: usize| {
        let mut key = entities[row].to_bytes();
        if let Some(anchors) = anchors {
            key.push(0);
            key.extend(anchors[row].to_bytes());
        }
        key
    };
    let asked: BTreeSet<Vec<u8>> = (num_train..instance.rows).map(key).collect();
    Ok((0..num_train)
        .filter(|&row| asked.contains(&key(row)))
        .collect())
}

/// The payload of one batch's request, as the reference's engine builds it
/// (`predict_request_to_json`) and its client then sends it, the context
/// that answers the question taken out.
pub fn build(sampled: &Sampled) -> Result<Built, CoreError> {
    let examples = sampled.num_train + sampled.num_test;
    if sampled.anchors.len() != examples || sampled.labels.len() != sampled.num_train {
        return Err(internal("the examples, anchors and labels do not align"));
    }
    let kind = sampled.kind;

    // The instance table: one row per example.
    let entity = sampled
        .tables
        .iter()
        .find(|t| t.name == sampled.entity_table)
        .ok_or_else(|| internal("the entity table was not sampled"))?;
    let key = entity
        .primary_key
        .as_deref()
        .ok_or_else(|| internal("the entity table has no key"))?;
    let key_array = entity
        .columns
        .iter()
        .find_map(|(n, column)| match column {
            SampledColumn::Arrow(array) if n == key => Some(array),
            _ => None,
        })
        .ok_or_else(|| internal("the entity table's key was not sampled"))?;
    let mut first = vec![None; examples];
    for (row, &instance) in entity.batch.iter().enumerate() {
        if instance < examples && first[instance].is_none() {
            first[instance] = Some(row);
        }
    }
    // The lagged target columns are the task's own, which the wire's names
    // keep clear of (`occupied`).
    let mut occupied = vec![INSTANCE_ID.to_owned(), TARGET.to_owned()];
    occupied.extend(sampled.lags.iter().map(|lag| lag.name.clone()));
    let reference = unique_name(&occupied, &format!("{ENTITY_REFERENCE}_0"));
    occupied.push(reference.clone());
    let mut instance = Frame {
        columns: vec![int_column(INSTANCE_ID, 0..examples)],
        rows: examples,
    };
    // A float32 key's reference is a float64: the reference builds the
    // column from the keys as Python floats, which pandas types float64
    // (T2.12; each value is written widened either way).
    let reference_dtype = match key_dtype(key_array)? {
        Dtype::Float32 => Dtype::Float64,
        dtype => dtype,
    };
    let reference_cells = first
        .iter()
        .enumerate()
        .map(|(at, row)| match row {
            Some(row) => cell(key_array, *row, &reference, reference_dtype).map_err(|e| {
                e.with_field("table", Detail::text(INSTANCE_TABLE))
                    .with_field("row", count(at))
            }),
            None => Ok(Value::Null),
        })
        .collect::<Result<Vec<_>, _>>()?;
    instance.columns.push(FrameColumn {
        name: reference.clone(),
        dtype: reference_dtype,
        cells: reference_cells,
    });
    let anchor_column =
        if !sampled.anchors.is_empty() && sampled.anchors.iter().any(|a| *a != NO_ANCHOR) {
            let name = unique_name(&occupied, ANCHOR_TIME);
            occupied.push(name.clone());
            instance.columns.push(FrameColumn {
                name: name.clone(),
                dtype: Dtype::TimestampMicros,
                cells: sampled
                    .anchors
                    .iter()
                    .map(|a| {
                        // A missing anchor (an entity's own time null,
                        // `NaT`) is sent as null, as the reference sends it.
                        if *a == sdm_engine::sample::MISSING_ANCHOR {
                            Value::Null
                        } else {
                            Value::Text(iso_microseconds(*a))
                        }
                    })
                    .collect(),
            });
            Some(name)
        } else {
            None
        };
    let (target_dtype, mut target_cells, classes) =
        target(kind, &sampled.labels, sampled.float_text)?;
    target_cells.resize(examples, Value::Null);
    instance.columns.push(FrameColumn {
        name: TARGET.to_owned(),
        dtype: target_dtype,
        cells: target_cells,
    });
    for lag in &sampled.lags {
        instance.columns.push(lag_column(lag, examples)?);
    }

    // The related tables: each row with the example it belongs to, and a
    // key of its own where the table sends none.
    let mut frames: Vec<Related> = Vec::new();
    for table in &sampled.tables {
        let rows: Vec<usize> = (0..table.batch.len()).collect();
        let mut frame = Frame {
            columns: vec![int_column(INSTANCE_ID, table.batch.iter().copied())],
            rows: rows.len(),
        };
        for (name, column) in &table.columns {
            if column.len() != rows.len() {
                return Err(internal("a sampled column does not align with its rows"));
            }
            let key = table.primary_key.as_deref() == Some(name.as_str());
            frame.columns.push(
                sampled_column(name, column, &rows, key)
                    .map_err(|e| e.with_field("table", Detail::text(&table.name)))?,
            );
        }
        let mut stypes = vec![(INSTANCE_ID.to_owned(), Stype::Id)];
        stypes.extend(table.stypes.iter().cloned());
        let key_column = match &table.primary_key {
            Some(key) if frame.column(key).is_some() => key.clone(),
            _ => {
                let name = unique_name(&frame.names(), NODE_ID);
                frame.columns.insert(1, int_column(&name, 0..frame.rows));
                stypes.push((name.clone(), Stype::Id));
                name
            }
        };
        frames.push((table.name.clone(), frame, key_column, stypes));
    }

    let mut relationships = Vec::new();
    {
        let (_, _, target_column, _) = frames
            .iter()
            .find(|(n, ..)| *n == sampled.entity_table)
            .ok_or_else(|| internal("the entity table was not sampled"))?;
        if Some(target_column.as_str()) != entity.primary_key.as_deref() {
            return Err(internal("the entity table sends no key of its own"));
        }
        relationships.push(Relationship {
            source_table: None,
            source_columns: vec![reference.clone()],
            target_table: sampled.entity_table.clone(),
            target_columns: vec![target_column.clone()],
        });
    }
    // Each foreign key once, merging every partial link row by row. The
    // first valid pair for each sampled source row supplies its key; only
    // a row no link supplies is null. An integer key column holding a
    // null, or a float32 key's column, is written as float64, as pandas
    // types such a column.
    let mut written: Vec<(&str, &str, &str)> = Vec::new();
    for link in &sampled.links {
        let edge = (link.src.as_str(), link.fkey.as_str(), link.dst.as_str());
        if written.contains(&edge) {
            continue;
        }
        let (Some(src), Some(dst)) = (
            frames.iter().position(|(n, ..)| *n == link.src),
            frames.iter().position(|(n, ..)| *n == link.dst),
        ) else {
            continue;
        };
        if link.pairs.is_empty() {
            continue;
        }
        let dst_key = frames[dst].2.clone();
        let (dst_rows, key_cells, key_dtype) = {
            let column = frames[dst]
                .1
                .column(&dst_key)
                .ok_or_else(|| internal("a table's key column is missing"))?;
            (frames[dst].1.rows, column.cells.clone(), column.dtype)
        };
        let src_rows = frames[src].1.rows;
        let mut values = vec![None; src_rows];
        for partial in sampled.links.iter().filter(|partial| {
            partial.src == link.src && partial.fkey == link.fkey && partial.dst == link.dst
        }) {
            for &(row, at) in &partial.pairs {
                if row < src_rows && at < dst_rows && values[row].is_none() {
                    values[row] = Some(key_cells[at].clone());
                }
            }
        }
        if values.iter().all(Option::is_none) {
            continue;
        }
        let as_float = key_dtype == Dtype::Int64 && values.iter().any(Option::is_none);
        let cells = values
            .into_iter()
            .map(|v| match v {
                Some(cell) if as_float => float_key(&cell),
                Some(cell) => Ok(cell),
                None => Ok(Value::Null),
            })
            .collect::<Result<_, _>>()?;
        // A float32 key's foreign key is a float64, as pandas types the
        // column the reference fills with the keys as Python floats (T2.12;
        // each value is written widened either way).
        let column = FrameColumn {
            name: link.fkey.clone(),
            dtype: if as_float || key_dtype == Dtype::Float32 {
                Dtype::Float64
            } else {
                key_dtype
            },
            cells,
        };
        let frame = &mut frames[src].1;
        match frame.columns.iter_mut().find(|c| c.name == link.fkey) {
            Some(slot) => *slot = column,
            None => frame.columns.push(column),
        }
        frames[src].3.push((link.fkey.clone(), Stype::Id));
        relationships.push(Relationship {
            source_table: Some(link.src.clone()),
            source_columns: vec![link.fkey.clone()],
            target_table: link.dst.clone(),
            target_columns: vec![dst_key.clone()],
        });
        written.push(edge);
    }

    let num_train = sampled.num_train;
    let mut instance_stypes = vec![
        (INSTANCE_ID.to_owned(), Stype::Id),
        (reference.clone(), Stype::Id),
    ];
    if let Some(anchor) = &anchor_column {
        instance_stypes.push((anchor.clone(), Stype::Timestamp));
    }
    instance_stypes.push((
        TARGET.to_owned(),
        if kind.is_classification() {
            Stype::Categorical
        } else {
            Stype::Numerical
        },
    ));
    let mut instance_schema = table_schema(&instance, &instance_stypes, &[INSTANCE_ID.to_owned()]);
    if let Some(anchor) = &anchor_column
        && let Some((_, spec)) = instance_schema
            .columns
            .0
            .iter_mut()
            .find(|(k, _)| k == anchor)
    {
        spec.nullable = Some(false);
    }
    let related_schema = frames
        .iter()
        .map(|(name, frame, key, stypes)| {
            (
                name.clone(),
                table_schema(frame, stypes, &[INSTANCE_ID.to_owned(), key.clone()]),
            )
        })
        .collect();

    // The context that answers the question, taken out of both the
    // instance table and the related rows that hang off it.
    let answered = answering(
        &instance,
        num_train,
        sampled.windowed,
        anchor_column.as_deref(),
        &reference,
    )?;
    let half = |context: bool| -> Half {
        let in_half = |instance: usize| {
            (instance < num_train) == context && !(context && answered.contains(&instance))
        };
        let instance_table = if context {
            instance.table(in_half, None)
        } else {
            instance.table(in_half, Some(TARGET))
        };
        let related = frames
            .iter()
            .zip(&sampled.tables)
            .map(|((name, frame, ..), table)| {
                (
                    name.clone(),
                    frame.table(|row| in_half(table.batch[row]), None),
                )
            })
            .collect();
        Half {
            instance_table,
            related_tables: Named(related),
        }
    };

    let mut fields: Vec<String> = match kind {
        Kind::Regression if sampled.options.output_type == "quantiles" => vec!["quantiles".into()],
        Kind::Regression => vec!["prediction".into()],
        Kind::Binary | Kind::Multiclass => vec!["prediction".into(), "probabilities".into()],
    };
    if sampled.return_embeddings {
        fields.push("embeddings".into());
    }
    // The task's and the inference's sections, typed by the task's kind.
    macro_rules! task {
        ($target:expr) => {
            Task {
                kind: kind.wire_name().to_owned(),
                target: $target,
                entity_table_names: vec![sampled.entity_table.clone()],
                anchor_time_column: anchor_column.clone(),
                top_k: sampled.top_k,
            }
        };
    }
    macro_rules! inference {
        ($config:expr) => {
            Inference {
                run_mode: sampled.run_mode.wire_name().to_owned(),
                use_prediction_time: sampled.use_prediction_time,
                inference_config: $config,
            }
        };
    }
    let schema = PayloadSchema {
        instance_table: instance_schema,
        related_tables: Named(related_schema),
        relationships,
    };
    let output = Output {
        fields,
        prediction_statistic: sampled.statistic.clone(),
    };
    let metadata = Metadata {
        source_query: (!sampled.source_query.is_empty()).then(|| sampled.source_query.clone()),
        num_context_examples: num_train as u64,
        num_prediction_examples: sampled.num_test as u64,
    };
    let model = "kumo-relational".to_owned();
    let bytes = if kind.is_classification() {
        StatePayload {
            model,
            task: task!(ClassTarget {
                column_name: TARGET.to_owned(),
                dtype: target_dtype,
                classes,
                positive_class: (kind == Kind::Binary).then(|| "true".to_owned()),
            }),
            schema,
            context: half(true),
            predict: half(false),
            output,
            inference: inference!(sampled.options.classification()),
            metadata,
        }
        .to_bytes()
    } else {
        StatePayload {
            model,
            task: task!(Target {
                column_name: TARGET.to_owned(),
                dtype: target_dtype,
            }),
            schema,
            context: half(true),
            predict: half(false),
            output,
            inference: inference!(sampled.options.regression()),
            metadata,
        }
        .to_bytes()
    };
    // The limit counts the bytes the request sends, as written (PD-2).
    // The refusal names what to change, as the reference's client words
    // its engine's (`_said_in_this_client_s_terms`, `main`'s 1fc5a686): a
    // call scoring several rows can be asked as smaller calls, below the
    // most rows a call of the request scores (`smaller_calls`); a call of
    // one row cannot, and only a narrower context helps
    // (`no_smaller_call`).
    if bytes.len() > PAYLOAD_BYTES {
        let count = |n: usize| Detail::Scalar(sdm_contract::Scalar::UInt64(n as u64));
        return Err(CoreError::new(
            if sampled.num_test > 1 {
                Sentence::PayloadSmallerCalls
            } else {
                Sentence::PayloadNoSmallerCall
            },
            "the request is larger than the endpoint takes",
        )
        .with_field("field", Detail::text("payload"))
        .with_field("bytes", count(bytes.len()))
        .with_field("limit", count(PAYLOAD_BYTES))
        .with_field(
            "batch_too_large_to_send",
            Detail::Scalar(sdm_contract::Scalar::Bool(true)),
        )
        .with_field(
            "remedy",
            Detail::text(if sampled.num_test > 1 {
                "smaller_calls"
            } else {
                "no_smaller_call"
            }),
        )
        .with_field("rows", count(sampled.num_test))
        .with_field("rows_per_call", count(sampled.rows_per_call)));
    }
    Ok(Built {
        bytes,
        answering: answered.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::{decimal_float, iso_microseconds};
    use sdm_wire::json::Value;

    #[test]
    fn times_are_written_as_the_reference_writes_them() {
        assert_eq!(iso_microseconds(0), "1970-01-01T00:00:00.000000Z");
        assert_eq!(
            iso_microseconds(1_700_000_000_123_456_789),
            "2023-11-14T22:13:20.123456Z"
        );
        assert_eq!(iso_microseconds(-1), "1969-12-31T23:59:59.999999Z");
        assert_eq!(
            iso_microseconds(1_706_659_200_000_000_000),
            "2024-01-31T00:00:00.000000Z"
        );
    }

    #[test]
    fn a_decimal_is_sent_as_its_nearest_double() {
        let number = |t: &str| Value::Number(t.to_owned());
        // 1.5 at scale 2, a whole 5, and the nearest double to 0.1.
        assert_eq!(decimal_float(150, 2, "c").unwrap(), number("1.5"));
        assert_eq!(decimal_float(500, 2, "c").unwrap(), number("5.0"));
        assert_eq!(decimal_float(1, 1, "c").unwrap(), number("0.1"));
        // 0.70, where pyarrow's cast gives 0.7000000000000001, and an
        // 18-digit fraction, where it gives 0.12345678901234569.
        assert_eq!(decimal_float(70, 2, "c").unwrap(), number("0.7"));
        assert_eq!(
            decimal_float(123_456_789_012_345_678, 18, "c").unwrap(),
            number("0.12345678901234568")
        );
        // What a double does not hold exactly is sent rounded, not refused
        // (T2.0c): 2**53 + 1 whole, and 2**53 - 1 + 0.5.
        let unit = 10_i128.pow(18);
        assert_eq!(
            decimal_float(((1_i128 << 53) + 1) * unit, 18, "c").unwrap(),
            Value::float(9_007_199_254_740_992.0).unwrap()
        );
        assert_eq!(
            decimal_float(9_007_199_254_740_991 * 10 + 5, 1, "c").unwrap(),
            Value::float(9_007_199_254_740_992.0).unwrap()
        );
        // The widest DECIMAL(38, 0) and DECIMAL(38, 38).
        let largest = 10_i128.pow(38) - 1;
        assert_eq!(
            decimal_float(largest, 0, "c").unwrap(),
            Value::float(1e38).unwrap()
        );
        assert_eq!(
            decimal_float(largest, 38, "c").unwrap(),
            Value::float(1.0).unwrap()
        );
    }
}
