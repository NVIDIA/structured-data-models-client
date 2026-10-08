// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The parts of `KumoRelational` (`rfm/rfm.py`) that sampling reaches:
//! `_get_task_table`, which draws the labelled examples a task is made of,
//! and `_get_context`, which samples their subgraphs, with the checks and
//! defaults each makes (`_get_task_type`, `_get_default_anchor_time`,
//! `_validate_time`, `add_lagged_target`, `_validate_task_references`).
//!
//! A task's examples come back as frames; the `TaskTable` and
//! `PredictionContext` objects made of them are the request's (Phase 8.5).

use std::sync::Arc;

use arrow_array::Int64Array;
use sdm_labels::date_offset::DateOffset;
use sdm_schema::decimal;
use sdm_schema::dt::Host;
use sdm_schema::engine::infer;
use serde_json::Value;

use super::inputs::{Inputs, int, optional_text};
use super::output::{SubgraphDone, Warning};
use super::query::{EntityId, SamplerQuery, TargetKind};
use super::sampler::Sampler;
use super::target::{Anchor, TargetArgs, sample_target_with, seed, timestamp_text};
use super::{Db, Raised};
use crate::frame::{Dtype, Frame, Series};
use crate::keys::Key;

/// `rfm.py`'s `_RANDOM_SEED`, the seed a call without one samples with.
pub const RANDOM_SEED: i64 = 42;
/// `_MAX_HOPS`.
pub const MAX_HOPS: usize = 6;
/// `_MAX_SUBGRAPH_TABLES`.
pub const MAX_SUBGRAPH_TABLES: usize = 15;
/// `TaskTable.ENTITY_TIME`.
pub const ENTITY_TIME: &str = "__entity_time__";

/// `_MAX_CONTEXT_SIZE[run_mode]`.
pub fn max_context_size(run_mode: &str) -> Result<i64, Raised> {
    Ok(match run_mode {
        "debug" => 100,
        "fast" => 1_000,
        "normal" => 5_000,
        "best" => 10_000,
        other => {
            return Err(Raised::value_error(format!(
                "{} is not a valid RunMode",
                sdm_schema::py::str_repr(other)
            )));
        }
    })
}

/// `_DEFAULT_NUM_NEIGHBORS[run_mode]`.
pub fn default_num_neighbors(run_mode: &str) -> Result<Vec<i64>, Raised> {
    Ok(match run_mode {
        "debug" => vec![16, 16, 4, 4, 1, 1],
        "fast" => vec![32, 32, 8, 8, 4, 4],
        "normal" | "best" => vec![64, 64, 8, 8, 4, 4],
        other => {
            return Err(Raised::value_error(format!(
                "{} is not a valid RunMode",
                sdm_schema::py::str_repr(other)
            )));
        }
    })
}

/// `TaskType`, by its value.
pub fn is_link_pred(task_type: &str) -> bool {
    matches!(
        task_type,
        "temporal_link_prediction" | "static_link_prediction" | "link_prediction"
    )
}

/// `_MAX_TEST_SIZE[task_type]`.
fn max_test_size(task_type: &str) -> i64 {
    if task_type == "temporal_link_prediction" {
        400
    } else {
        2_000
    }
}

/// `_get_task_type(query, edge_types)`.
pub fn task_type(
    query: &SamplerQuery,
    edge_types: &[(String, String, String)],
) -> Result<&'static str, Raised> {
    if query.is_forecasting {
        return Ok("forecasting");
    }
    match &query.target_kind {
        TargetKind::Condition => Ok("binary_classification"),
        TargetKind::Aggregation { aggr, column } if aggr == "LIST_DISTINCT" => {
            let (table, col) = column.split_once('.').unwrap_or((column, ""));
            let found = edge_types
                .iter()
                .filter(|e| e.0 == table && e.1 == col)
                .count();
            if found != 1 {
                return Err(Raised::new(
                    "NotImplementedError",
                    format!(
                        "Multilabel-classification queries based on 'LIST_DISTINCT' are not \
                         supported yet. If you planned to write a link prediction query \
                         instead, make sure to register '{col}' as a foreign key."
                    ),
                ));
            }
            Ok("temporal_link_prediction")
        }
        TargetKind::Aggregation { .. } => Ok("regression"),
        TargetKind::Column { stype } => match stype.as_deref() {
            Some("ID" | "categorical") => Ok("multiclass_classification"),
            Some("numerical") => Ok("regression"),
            _ => Err(Raised::new(
                "NotImplementedError",
                "Task type not yet supported",
            )),
        },
        TargetKind::Other => Err(Raised::new("AssertionError", "")),
    }
}

/// A `Timestamp`: nanoseconds, and its unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stamp {
    pub ns: i64,
    pub unit: String,
}

fn stamp_of(value: &Value, what: &str) -> Result<Option<Stamp>, Raised> {
    match value {
        Value::Null => Ok(None),
        Value::Object(o) => Ok(Some(Stamp {
            ns: int(o.get("ns").unwrap_or(&Value::Null), what)?,
            unit: optional_text(o.get("unit"), what)?.unwrap_or_else(|| "ns".into()),
        })),
        _ => Err(Raised::value_error(format!("{what} is not a timestamp"))),
    }
}

/// `ts + offset` (`sign` 1) or `ts - offset` (`sign` -1), keeping its unit.
fn shifted(stamp: &Stamp, offset: &DateOffset, sign: i64) -> Result<Stamp, Raised> {
    Ok(Stamp {
        ns: offset.apply(stamp.ns, sign)?,
        unit: stamp.unit.clone(),
    })
}

/// `offset * factor`.
fn times(offset: &DateOffset, factor: i64) -> Result<DateOffset, Raised> {
    Ok(DateOffset {
        n: offset.n.checked_mul(factor).ok_or_else(|| {
            Raised::new("OverflowError", "Python int too large to convert to C long")
        })?,
        kwds: offset.kwds.clone(),
    })
}

/// `pd.DateOffset(0)`.
fn zero_offset() -> DateOffset {
    DateOffset {
        n: 0,
        kwds: Vec::new(),
    }
}

fn aggregation_tables(query: &SamplerQuery) -> Vec<String> {
    query
        .target_aggregations
        .iter()
        .map(|a| a.table.clone())
        .collect()
}

/// `_get_default_anchor_time(query)`.
pub async fn default_anchor_time(
    sampler: &mut Sampler,
    db: &Db,
    query: Option<&SamplerQuery>,
) -> Result<Stamp, Raised> {
    let tables = match query {
        Some(query) if query.is_temporal() => Some(aggregation_tables(query)),
        _ => None,
    };
    let (ns, unit) = sampler.max_time_stamp(db, tables.as_deref()).await?;
    Ok(Stamp { ns, unit })
}

/// `_validate_time(query, anchor_time, context_anchor_time, evaluate)`: its
/// errors, and the warnings it gives.
pub async fn validate_time(
    sampler: &mut Sampler,
    db: &Db,
    query: &SamplerQuery,
    anchor_time: &Stamp,
    context_anchor_time: Option<&Stamp>,
    evaluate: bool,
) -> Result<Vec<Warning>, Raised> {
    let mut warnings = Vec::new();
    if sampler.time_column_dict.is_empty() {
        return Ok(warnings);
    }
    let (min_time, max_time) = if query.is_temporal() {
        let tables = aggregation_tables(query);
        let min = sampler.min_time_stamp(db, Some(&tables)).await?;
        let max = sampler.max_time_stamp(db, Some(&tables)).await?;
        (min, max)
    } else {
        let min = sampler.min_time_stamp(db, None).await?;
        let max = sampler.max_time_stamp(db, None).await?;
        (min, max)
    };
    let (min_time, max_time) = (min_time.0, max_time.0);
    if min_time > max_time {
        let tables = query.is_temporal().then(|| aggregation_tables(query));
        let columns = sampler
            .time_column_dict
            .iter()
            .filter(|(table, _)| tables.as_ref().is_none_or(|tables| tables.contains(table)))
            .map(|(table, column)| format!("{table}.{column}"))
            .collect::<Vec<_>>()
            .join(", ");
        let columns = if columns.is_empty() {
            "required by this query".to_string()
        } else {
            columns
        };
        return Err(Raised::value_error(format!(
            "No readable timestamps remain in the source columns {columns}. Provide at least one valid timestamp."
        )));
    }
    if anchor_time.ns < min_time {
        return Err(Raised::value_error(format!(
            "Anchor timestamp '{}' is before the earliest timestamp '{}' in the data.",
            timestamp_text(anchor_time.ns),
            timestamp_text(min_time)
        )));
    }
    if let Some(context) = context_anchor_time
        && context.ns < min_time
    {
        return Err(Raised::value_error(format!(
            "Context anchor timestamp is too early or aggregation time range is too large. To \
             make this prediction, we would need data back to '{}', however, your data only \
             contains data back to '{}'.",
            timestamp_text(context.ns),
            timestamp_text(min_time)
        )));
    }
    let end_offset = query.target_end_offset.clone().unwrap_or_else(zero_offset);
    let num_forecasts = query.num_forecasts;
    if let Some(context) = context_anchor_time {
        if context.ns > anchor_time.ns {
            warnings.push(Warning::user(format!(
                "Context anchor timestamp (got '{}') is set to a later date than the prediction \
                 anchor timestamp (got '{}'). Please make sure this is intended.",
                timestamp_text(context.ns),
                timestamp_text(anchor_time.ns)
            )));
        } else if query.is_temporal() && end_offset.apply(context.ns, 1)? > anchor_time.ns {
            warnings.push(Warning::user(format!(
                "Aggregation for context examples at timestamp '{}' will leak information from \
                 the prediction anchor timestamp '{}'. Please make sure this is intended.",
                timestamp_text(context.ns),
                timestamp_text(anchor_time.ns)
            )));
        } else {
            let back = times(&end_offset, num_forecasts)?.apply(context.ns, -1)?;
            if back < min_time {
                warnings.push(Warning::user(format!(
                    "Context anchor timestamp is too early or aggregation time range is too \
                     large. To form proper input data, we would need data back to '{}', however, \
                     your data only contains data back to '{}'.",
                    timestamp_text(back),
                    timestamp_text(min_time)
                )));
            }
        }
    }
    let day = DateOffset::single(sdm_labels::date_offset::Unit::Days, 1);
    if !evaluate && anchor_time.ns > day.apply(max_time, 1)? {
        warnings.push(Warning::user(format!(
            "Anchor timestamp '{}' is after the latest timestamp '{}' in the data. Please make \
             sure this is intended.",
            timestamp_text(anchor_time.ns),
            timestamp_text(max_time)
        )));
    }
    if evaluate && anchor_time.ns > times(&end_offset, num_forecasts)?.apply(max_time, -1)? {
        return Err(Raised::value_error(format!(
            "Anchor timestamp for evaluation is after the latest supported timestamp '{}'.",
            timestamp_text(end_offset.apply(max_time, -1)?)
        )));
    }
    Ok(warnings)
}

/// One side of a task: its entities, labels and anchor times.
#[derive(Clone, Debug)]
pub struct TaskSide {
    pub entity: Series,
    pub target: Option<Series>,
    /// `ANCHOR_TIMESTAMP`, in nanoseconds, with its unit; none where the
    /// entities' own times are the anchors.
    pub anchor_time: Option<(Vec<Option<i64>>, String)>,
}

/// What `_get_task_table` makes a `TaskTable` of.
#[derive(Clone, Debug)]
pub struct TaskDone {
    pub task_type: String,
    pub entity_table_names: Vec<String>,
    pub context: TaskSide,
    pub prediction: TaskSide,
    /// `ANCHOR_TIMESTAMP`, or `TaskTable.ENTITY_TIME`.
    pub time_column: String,
    pub num_forecasts: i64,
    pub step_size: Option<i64>,
    /// `add_lagged_target`'s frame, `__kumo_arl0__` first, where lagged
    /// targets are added: its first rows are the context's.
    pub lagged: Option<Frame>,
    pub warnings: Vec<Warning>,
}

/// An `indices` value as the host gives it: JSON's numbers and text.
fn index_key(value: &Value) -> Result<Key, Raised> {
    Ok(match value {
        Value::Null => Key::Null,
        Value::Bool(b) => Key::Int(i64::from(*b)),
        Value::Number(n) => match n.as_i64() {
            Some(i) => Key::Int(i),
            None => Key::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => Key::Text(s.clone()),
        Value::Object(map) => match (
            map.get("$decimal").and_then(Value::as_str),
            map.get("$int").and_then(Value::as_str),
        ) {
            (Some(text), _) => match crate::keys::parse_decimal(text) {
                Some((digits, scale)) => Key::Decimal(digits, scale),
                None => Key::Float(text.parse().unwrap_or(f64::NAN)),
            },
            // An int no key of a number type holds: its digits, which a
            // key of text may still name, and a number key refuses.
            (None, Some(digits)) => Key::Text(digits.to_string()),
            (None, None) => return Err(not_an_index()),
        },
        _ => return Err(not_an_index()),
    })
}

fn not_an_index() -> Raised {
    Raised::new(
        "TypeError",
        "an index that is not a number or text is not supported by the core",
    )
}

fn key_repr(value: &Value) -> String {
    if let Some(text) = value.get("$decimal").and_then(Value::as_str) {
        return format!("Decimal('{text}')");
    }
    if let Some(digits) = value.get("$int").and_then(Value::as_str) {
        return digits.to_string();
    }
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => match n.as_i64() {
            Some(i) => i.to_string(),
            None => sdm_schema::py::float_repr(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => sdm_schema::py::str_repr(s),
        other => other.to_string(),
    }
}

/// `str(series.dtype)` of a column.
fn dtype_text(series: &Series) -> String {
    match &series.dtype {
        Dtype::Arrow => format!(
            "{}[pyarrow]",
            infer::arrow_type_name(series.array.data_type())
        ),
        other => other.tag(),
    }
}

/// Whether the engine's `infer_dtype` reads a column's data type, rather
/// than raising, as `TaskTable.add_columns` asks it to.
fn engine_reads_dtype(series: &Series) -> bool {
    use arrow_schema::DataType;
    match &series.dtype {
        Dtype::Arrow => infer::infer_dtype(&infer::SampleColumn {
            name: String::new(),
            kind: infer::SampleKind::Arrow(series.array.data_type().clone()),
            cells: Vec::new(),
        })
        .is_ok(),
        Dtype::Numpy(name) => {
            name.starts_with("datetime64")
                || name.starts_with("timedelta64")
                || infer::pandas_dtype(name).is_some()
        }
        Dtype::Extension(name) => infer::pandas_dtype(name).is_some(),
        // The engine reads every `category` column as text.
        Dtype::Category => true,
        // An `object` column is read from its contents: lists only when
        // `pa.array` reads their items as numbers or text. `pa.array` makes
        // a `list` of `string` of Python lists of texts, however the core
        // holds them, so a large list or text is read as those.
        Dtype::Object => match series.array.data_type() {
            DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _) => {
                let item = item.data_type();
                item.is_integer()
                    || item.is_floating()
                    || matches!(
                        item,
                        DataType::Decimal128(..)
                            | DataType::Decimal256(..)
                            | DataType::Utf8
                            | DataType::LargeUtf8
                            | DataType::Utf8View
                            | DataType::Null
                    )
            }
            _ => true,
        },
    }
}

/// A key of whole decimals, such as a warehouse's `DECIMAL(10, 0)`, as the
/// integers it holds, which the schema already calls it: today's engine
/// had no reading of a decimal `ENTITY` and failed the task table on it.
fn whole_decimals_as_ints(series: &Series) -> Series {
    use arrow_schema::DataType;
    if !matches!(series.dtype, Dtype::Arrow)
        || !matches!(series.array.data_type(), DataType::Decimal128(_, 0))
    {
        return series.clone();
    }
    use arrow_array::cast::AsArray;
    let decimals = series
        .array
        .as_primitive::<arrow_array::types::Decimal128Type>();
    let ints: Option<Vec<Option<i64>>> = decimals
        .iter()
        .map(|v| v.map(i64::try_from).transpose().ok())
        .collect();
    match ints {
        Some(ints) => Series::new(Dtype::Arrow, Arc::new(Int64Array::from(ints))),
        None => series.clone(),
    }
}

/// `TaskTable(...)`'s checks of the frames `_get_task_table` gives it: each
/// needs a row, and the engine must read a data type for each column of
/// the context (`TaskTable.add_columns`).
fn check_task_table(done: &TaskDone) -> Result<(), Raised> {
    let (context, prediction) = done.frames()?;
    if context.is_empty() {
        return Err(Raised::value_error("No context examples given"));
    }
    if prediction.is_empty() {
        return Err(Raised::value_error(
            "Provide at least one entity to predict for",
        ));
    }
    for (name, column) in context.names.iter().zip(&context.columns) {
        if !engine_reads_dtype(column) {
            return Err(Raised::new(
                "RuntimeError",
                format!(
                    "Encountered unsupported data type '{}' for column '{name}' in task table. \
                     Please either manually override the columns's data type or remove the \
                     column from this table.",
                    dtype_text(column)
                ),
            ));
        }
    }
    Ok(())
}

/// Why `pd.Series(indices, dtype=like.dtype)` failed for a decimal key:
/// pyarrow refused the indices with the `TypeError` or `ValueError`
/// `_get_task_table` turns into its own, or raised what it lets through.
enum Uncast {
    Refused,
    Inferred {
        class: &'static str,
        context: &'static str,
    },
    Raised(Raised),
}

/// One index as pyarrow reads it into an array. JSON holds no `NaN`, no
/// infinity and no `decimal.Decimal` (the inputs are written with
/// `allow_nan=False`, and a `Decimal` index is refused before the core is
/// asked), so neither does a `Scalar`.
#[derive(Clone, Copy)]
enum Scalar<'a> {
    Null,
    Bool(bool),
    /// A JSON or tagged integer representable as `i128`.
    Int(i128),
    WideInt,
    Float(f64),
    Text(&'a str),
}

impl<'a> Scalar<'a> {
    fn of(value: &'a Value) -> Result<Self, Raised> {
        Ok(match value {
            Value::Null => Scalar::Null,
            Value::Bool(b) => Scalar::Bool(*b),
            Value::Number(n) => match (n.as_i64(), n.as_u64()) {
                (Some(i), _) => Scalar::Int(i.into()),
                (None, Some(u)) => Scalar::Int(u.into()),
                _ => Scalar::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            Value::String(s) => Scalar::Text(s),
            Value::Object(object) => match object.get("$int").and_then(Value::as_str) {
                Some(text)
                    if !text.trim_start_matches(['-', '+']).is_empty()
                        && text
                            .strip_prefix(['-', '+'])
                            .unwrap_or(text)
                            .bytes()
                            .all(|digit| digit.is_ascii_digit()) =>
                {
                    text.parse().map_or(Scalar::WideInt, Scalar::Int)
                }
                _ => return Err(not_an_index()),
            },
            _ => {
                return Err(Raised::new(
                    "TypeError",
                    "an index that is not a number or text is not supported by the core",
                ));
            }
        })
    }
}

/// The Arrow type pyarrow infers for a list of values.
#[derive(Clone, Copy, PartialEq)]
enum Inferred {
    Null,
    Bool,
    Int64,
    Double,
    Utf8,
}

/// `pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(precision,
/// scale)))`, as pandas' `ArrowExtensionArray._box_pa_array` builds it:
///
/// 1. `pa.array(values, type=decimal128, mask=...)`, where the mask is the
///    `None`s. pyarrow takes ints (any size) and `None` there, and refuses
///    a bool, float or text, or an int that does not fit, with
///    `ArrowInvalid` or `ArrowTypeError`.
/// 2. Then `pa.array(values, mask=...)` infers a type, looking at values up
///    to the first bool, float or text and no further: `double` if that is
///    a float, else `int64` if an int came before it, else its own type.
///    It converts the values in order, and the first that does not convert
///    fails: a bool to `int64` or anything but a bool to `bool` or anything
///    but text to `utf8` (`ArrowInvalid` or `ArrowTypeError`); an int past
///    2**53 to `double` (`ArrowInvalid`); an int past `int64` to `int64`
///    (`OverflowError`, which pandas lets through).
/// 3. Then it casts the array to the decimal type, which refuses what does
///    not fit (`ArrowInvalid`), refuses any `int64` for a negative scale,
///    and raises `ArrowNotImplementedError` for `bool`.
///
/// `tests/rust_core/rfm/test_decimal_indices.py` holds this to
/// `_get_task_table` over generated lists, in every order, for four key
/// types: every pair of 19 values (`None`, bools, ints, floats and texts at
/// the edges above), every triple of 7 and every quadruple of 5.
///
/// Tagged integers retain exact values and overflow classification past `uint64`.
fn cast_decimals(values: &[Scalar], precision: u8, scale: i8) -> Result<Vec<Option<i128>>, Uncast> {
    // 1. Straight to the decimal type.
    let direct = values
        .iter()
        .map(|value| match *value {
            Scalar::Null => Some(None),
            Scalar::Int(i) => decimal::cast_int(i, precision, scale).map(Some),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    if let Some(digits) = direct {
        return Ok(digits);
    }
    let direct_class = values
        .iter()
        .find_map(|value| match *value {
            Scalar::Null => None,
            Scalar::Int(integer) if decimal::cast_int(integer, precision, scale).is_some() => None,
            Scalar::Int(_) | Scalar::WideInt | Scalar::Bool(_) => Some("pyarrow.lib.ArrowInvalid"),
            Scalar::Float(_) | Scalar::Text(_) => Some("pyarrow.lib.ArrowTypeError"),
        })
        .unwrap();
    // 2. Infer, then convert.
    let mut ints = false;
    let mut stop = None;
    for value in values {
        match value {
            Scalar::Null => {}
            Scalar::Int(_) | Scalar::WideInt => ints = true,
            other => {
                stop = Some(*other);
                break;
            }
        }
    }
    let inferred = match (stop, ints) {
        (Some(Scalar::Float(_)), _) => Inferred::Double,
        (_, true) => Inferred::Int64,
        (Some(Scalar::Bool(_)), _) => Inferred::Bool,
        (Some(Scalar::Text(_)), _) => Inferred::Utf8,
        _ => Inferred::Null,
    };
    for value in values {
        let converts = match (inferred, *value) {
            (_, Scalar::Null)
            | (Inferred::Double, Scalar::Bool(_) | Scalar::Float(_))
            | (Inferred::Bool, Scalar::Bool(_))
            | (Inferred::Utf8, Scalar::Text(_)) => true,
            (Inferred::Double, Scalar::Int(i)) => i.unsigned_abs() <= 1 << 53,
            (Inferred::Int64, Scalar::WideInt) => {
                return Err(Uncast::Raised(
                    Raised::new("OverflowError", "Python int too large to convert to C long")
                        .caused_by(direct_class, "Could not convert the indices to decimal"),
                ));
            }
            (Inferred::Int64, Scalar::Int(i)) => {
                if i64::try_from(i).is_err() {
                    return Err(Uncast::Raised(
                        Raised::new("OverflowError", "Python int too large to convert to C long")
                            .caused_by(direct_class, "Could not convert the indices to decimal"),
                    ));
                }
                true
            }
            _ => false,
        };
        if !converts {
            let class = match (inferred, value) {
                (Inferred::Utf8, _) | (Inferred::Int64, Scalar::Bool(_)) => {
                    "pyarrow.lib.ArrowTypeError"
                }
                _ => "pyarrow.lib.ArrowInvalid",
            };
            return Err(Uncast::Inferred {
                class,
                context: direct_class,
            });
        }
    }
    // 3. Cast.
    if inferred == Inferred::Bool {
        return Err(Uncast::Raised(Raised::new(
            "ArrowNotImplementedError",
            "Unsupported cast from bool to decimal128 using function cast_decimal",
        )));
    }
    values
        .iter()
        .map(|value| match (inferred, *value) {
            (_, Scalar::Null) => Ok(None),
            (Inferred::Double, Scalar::Bool(b)) => {
                decimal::cast_float(f64::from(u8::from(b)), precision, scale)
                    .map(Some)
                    .ok_or(Uncast::Refused)
            }
            (Inferred::Double, Scalar::Int(i)) => decimal::cast_float(i as f64, precision, scale)
                .map(Some)
                .ok_or(Uncast::Refused),
            (Inferred::Double, Scalar::Float(f)) => decimal::cast_float(f, precision, scale)
                .map(Some)
                .ok_or(Uncast::Refused),
            // Arrow casts no int to a negative scale ("Scale must be
            // non-negative").
            (Inferred::Int64, Scalar::Int(_)) if scale < 0 => Err(Uncast::Refused),
            (Inferred::Int64, Scalar::Int(i)) => decimal::cast_int(i, precision, scale)
                .map(Some)
                .ok_or(Uncast::Refused),
            (Inferred::Utf8, Scalar::Text(s)) => decimal::cast_text(s, precision, scale)
                .map(Some)
                .ok_or(Uncast::Refused),
            _ => Err(Uncast::Refused),
        })
        .collect()
}

/// `pd.Series(indices, dtype=like.dtype)` for a `decimal128` key, or what
/// `_get_task_table` raises: the `ValueError` naming the first index that
/// does not cast alone (`_first_uncastable`), if one does not.
fn decimal_indices(
    indices: &[Value],
    precision: u8,
    scale: i8,
    refuse: impl Fn(Option<usize>) -> Raised,
) -> Result<arrow_array::ArrayRef, Raised> {
    let values = indices
        .iter()
        .map(Scalar::of)
        .collect::<Result<Vec<_>, _>>()?;
    let digits = match cast_decimals(&values, precision, scale) {
        Ok(digits) => digits,
        Err(Uncast::Raised(raised)) => return Err(raised),
        Err(failure) => {
            let cause = match failure {
                Uncast::Refused => Raised::new(
                    "pyarrow.lib.ArrowInvalid",
                    "Could not convert the indices with the column's Arrow type",
                ),
                Uncast::Inferred { class, context } => Raised::new(
                    class,
                    "Could not convert the indices with the inferred Arrow type",
                )
                .caused_by(context, "Could not convert the indices to decimal"),
                Uncast::Raised(_) => unreachable!(),
            };
            let mut position = None;
            for (p, value) in values.iter().enumerate() {
                match cast_decimals(std::slice::from_ref(value), precision, scale) {
                    Ok(_) => {}
                    Err(Uncast::Refused | Uncast::Inferred { .. }) => {
                        position = Some(p);
                        break;
                    }
                    Err(Uncast::Raised(raised)) => {
                        let mut raised = raised.caused_by(&cause.class, cause.message);
                        raised.chain.extend(cause.chain);
                        return Err(raised);
                    }
                }
            }
            let mut raised = refuse(position);
            raised.chain.clear();
            raised = raised.caused_by(&cause.class, cause.message);
            raised.chain.extend(cause.chain);
            return Err(raised);
        }
    };
    let array = arrow_array::Decimal128Array::from(digits)
        .with_precision_and_scale(precision, scale)
        .map_err(|e| Raised::new("ValueError", e.to_string()))?;
    Ok(Arc::new(array))
}

/// `pd.Series(indices, dtype=like.dtype)`, or the `ValueError` naming the
/// first index that does not cast.
fn indices_like(indices: &[Value], like: &Series, entity_table: &str) -> Result<Series, Raised> {
    use arrow_array::{Float32Array, Float64Array, StringArray};
    use arrow_schema::DataType;
    let refuse = |position: Option<usize>| {
        let place = position.map_or_else(String::new, |p| {
            format!("; indices[{p}] is {}", key_repr(&indices[p]))
        });
        // Raised from the error `pd.Series(indices, dtype=...)` raised:
        // pyarrow's for an Arrow dtype, numpy's otherwise.
        let (class, why) = if matches!(like.dtype, Dtype::Arrow) {
            (
                "pyarrow.lib.ArrowInvalid",
                "Could not convert the indices with the column's Arrow type",
            )
        } else {
            (
                "builtins.ValueError",
                "could not convert the indices to the column's dtype",
            )
        };
        Raised::value_error(format!(
            "'indices' must hold values matching the primary key of '{entity_table}', which is \
             {}{place}",
            dtype_text(like)
        ))
        .caused_by(class, why)
    };
    let keys = indices
        .iter()
        .map(index_key)
        .collect::<Result<Vec<_>, _>>()?;
    let array: arrow_array::ArrayRef = match like.array.data_type() {
        t if t.is_integer() => {
            let mut values = Vec::with_capacity(keys.len());
            for (p, key) in keys.iter().enumerate() {
                values.push(match key {
                    Key::Int(i) => *i,
                    Key::Float(f) if f.fract() == 0.0 && f.is_finite() => *f as i64,
                    Key::Text(s) => s.trim().parse().map_err(|_| refuse(Some(p)))?,
                    // A whole `Decimal`; pandas refuses any other.
                    Key::Decimal(digits, scale) => crate::keys::whole_decimal(*digits, *scale)
                        .ok_or_else(|| refuse(Some(p)))?,
                    _ => return Err(refuse(Some(p))),
                });
            }
            let ints: arrow_array::ArrayRef = Arc::new(Int64Array::from(values));
            arrow_cast(&ints, t).map_err(|_| refuse(None))?
        }
        DataType::Float32 | DataType::Float64 => {
            let mut values = Vec::with_capacity(keys.len());
            for (p, key) in keys.iter().enumerate() {
                values.push(match key {
                    Key::Int(i) => *i as f64,
                    Key::Float(f) => *f,
                    Key::Text(s) => s.trim().parse().map_err(|_| refuse(Some(p)))?,
                    Key::Null => f64::NAN,
                    Key::Decimal(digits, scale) => sdm_schema::frame::decimal_text(*digits, *scale)
                        .parse()
                        .map_err(|_| refuse(Some(p)))?,
                });
            }
            if like.array.data_type() == &DataType::Float32 {
                Arc::new(Float32Array::from_iter_values(
                    values.into_iter().map(|value| value as f32),
                ))
            } else {
                Arc::new(Float64Array::from(values))
            }
        }
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            let values: Vec<Option<String>> = keys
                .iter()
                .map(|k| match k {
                    Key::Null => None,
                    Key::Int(i) => Some(i.to_string()),
                    Key::Float(f) => Some(sdm_schema::py::float_repr(*f)),
                    Key::Decimal(digits, scale) => {
                        Some(sdm_schema::frame::decimal_text(*digits, *scale))
                    }
                    Key::Text(s) => Some(s.clone()),
                })
                .collect();
            let strings: arrow_array::ArrayRef = Arc::new(StringArray::from(values));
            arrow_cast(&strings, like.array.data_type()).map_err(|_| refuse(None))?
        }
        DataType::Decimal128(precision, scale) => {
            decimal_indices(indices, *precision, *scale, refuse)?
        }
        DataType::Dictionary(..) => categorical_indices(indices, &keys, &like.array)?,
        _ => return Err(refuse(None)),
    };
    Ok(Series::new(like.dtype.clone(), array))
}

/// `pd.Series(indices, dtype=like.dtype)` for a `category` key: each index
/// the category Python's `==` finds it equal to, or missing where none is.
/// pandas refuses none of them; it compares a list of nothing but bools
/// with no category that is not a bool, so each of those is missing.
fn categorical_indices(
    indices: &[Value],
    keys: &[Key],
    like: &arrow_array::ArrayRef,
) -> Result<arrow_array::ArrayRef, Raised> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int8Type, Int16Type, Int32Type, Int64Type};
    use arrow_array::{DictionaryArray, PrimitiveArray};
    use arrow_schema::DataType;
    let dictionary = like
        .as_any_dictionary_opt()
        .ok_or_else(|| Raised::internal("a category column that is no dictionary"))?;
    let values = dictionary.values().clone();
    let categories = crate::keys::keys_of(&Series::new(Dtype::Object, values.clone()))?;
    let index = crate::keys::Index::new(&categories);
    let only_bools = indices.iter().any(|v| matches!(v, Value::Bool(_)))
        && indices
            .iter()
            .all(|v| matches!(v, Value::Bool(_) | Value::Null));
    let codes: Vec<Option<i64>> = keys
        .iter()
        .map(|key| if only_bools { None } else { index.get(key) })
        .collect();
    macro_rules! dictionary_of {
        ($t:ty) => {{
            let codes = codes
                .iter()
                .map(|c| c.map(|c| c.try_into()).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| Raised::internal("a category code out of range"))?;
            let array = DictionaryArray::<$t>::try_new(PrimitiveArray::<$t>::from(codes), values)
                .map_err(|e| Raised::internal(format!("cannot build categories: {e}")))?;
            Arc::new(array) as arrow_array::ArrayRef
        }};
    }
    Ok(match like.data_type() {
        DataType::Dictionary(key, _) => match key.as_ref() {
            DataType::Int8 => dictionary_of!(Int8Type),
            DataType::Int16 => dictionary_of!(Int16Type),
            DataType::Int32 => dictionary_of!(Int32Type),
            DataType::Int64 => dictionary_of!(Int64Type),
            other => {
                return Err(Raised::internal(format!(
                    "category codes of Arrow type {other} are not supported"
                )));
            }
        },
        _ => return Err(Raised::internal("a category column that is no dictionary")),
    })
}

/// An integer or text array cast to `to`, of the same kind.
fn arrow_cast(
    array: &arrow_array::ArrayRef,
    to: &arrow_schema::DataType,
) -> Result<arrow_array::ArrayRef, ()> {
    use arrow_array::{
        Int8Array, Int16Array, Int32Array, LargeStringArray, StringArray, UInt8Array, UInt16Array,
        UInt32Array, UInt64Array,
    };
    use arrow_schema::DataType;
    if array.data_type() == to {
        return Ok(array.clone());
    }
    let ints = crate::frame::ints(array.as_ref());
    let texts = crate::frame::texts(array.as_ref());
    Ok(match (to, ints, texts) {
        (DataType::Int32, Some(v), _) => Arc::new(Int32Array::from(
            v.into_iter()
                .map(|x| x.map(i32::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::Int16, Some(v), _) => Arc::new(Int16Array::from(
            v.into_iter()
                .map(|x| x.map(i16::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::Int8, Some(v), _) => Arc::new(Int8Array::from(
            v.into_iter()
                .map(|x| x.map(i8::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::UInt64, Some(v), _) => Arc::new(UInt64Array::from(
            v.into_iter()
                .map(|x| x.map(u64::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::UInt32, Some(v), _) => Arc::new(UInt32Array::from(
            v.into_iter()
                .map(|x| x.map(u32::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::UInt16, Some(v), _) => Arc::new(UInt16Array::from(
            v.into_iter()
                .map(|x| x.map(u16::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::UInt8, Some(v), _) => Arc::new(UInt8Array::from(
            v.into_iter()
                .map(|x| x.map(u8::try_from).transpose())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| ())?,
        )),
        (DataType::LargeUtf8, _, Some(v)) => Arc::new(LargeStringArray::from(v)),
        (DataType::Utf8, _, Some(v)) => Arc::new(StringArray::from(v)),
        _ => return Err(()),
    })
}

fn concat(parts: &[&Series]) -> Result<Series, Raised> {
    crate::frame::concat_series(parts)
}

/// `KumoRelational._get_task_table(query, indices, anchor_time,
/// context_anchor_time, run_mode, lag_timesteps, max_pq_iterations,
/// random_seed)`.
///
/// What it raises carries every warning it gave before it raised.
pub async fn get_task_table(inputs: Inputs, db: Db, mut host: Host) -> Result<TaskDone, Raised> {
    if let Some(facts) = inputs.arguments.get("timestamp_series_units") {
        host.timestamp_series_units =
            sdm_schema::dt::pandas::TimestampSeriesUnits::from_json(facts)
                .map_err(Raised::value_error)?;
    }
    let mut sampler = Sampler::new(&inputs, host)?;
    let mut given = Vec::new();
    let made = make_task_table(&mut sampler, &inputs, &db, &mut given).await;
    match made {
        Ok(done) => Ok(done),
        Err(raised) => {
            given.extend(sampler.take_warnings());
            Err(raised.after(given))
        }
    }
}

/// [`get_task_table`] through a sampler already built, which keeps what it
/// learns (its minimum and maximum times) for the calls after it, as one
/// `KumoRelational` keeps its sampler.
pub async fn get_task_table_with(
    sampler: &mut Sampler,
    inputs: &Inputs,
    db: &Db,
) -> Result<TaskDone, Raised> {
    let mut given = Vec::new();
    let made = make_task_table(sampler, inputs, db, &mut given).await;
    match made {
        Ok(done) => Ok(done),
        Err(raised) => {
            given.extend(sampler.take_warnings());
            Err(raised.after(given))
        }
    }
}

/// `_get_task_table`'s body, each warning it gives kept in `given` as it
/// gives it, and handed to the task it makes.
async fn make_task_table(
    sampler: &mut Sampler,
    inputs: &Inputs,
    db: &Db,
    given: &mut Vec<Warning>,
) -> Result<TaskDone, Raised> {
    let get = |name: &str| inputs.arguments.get(name).unwrap_or(&Value::Null);
    let mut query = SamplerQuery::from_tree(get("query"))?;
    let indices: Option<Vec<Value>> = match get("indices") {
        Value::Null => None,
        Value::Array(items) => Some(items.clone()),
        _ => return Err(Raised::value_error("indices is not a list")),
    };
    let mut anchor_time: Option<Stamp> = None;
    let mut anchor_is_entity = false;
    match get("anchor_time") {
        Value::String(s) if s == "entity" => anchor_is_entity = true,
        other => anchor_time = stamp_of(other, "anchor_time")?,
    }
    let (mut context_anchor, mut context_is_entity) = (None, false);
    match get("context_anchor_time") {
        Value::String(s) if s == "entity" => context_is_entity = true,
        other => context_anchor = stamp_of(other, "context_anchor_time")?,
    }
    let run_mode = optional_text(inputs.arguments.get("run_mode"), "run_mode")?
        .unwrap_or_else(|| "fast".into());
    let lag_timesteps = match get("lag_timesteps") {
        Value::Null => 0,
        v => int(v, "lag_timesteps")?,
    };
    let max_pq_iterations = match get("max_pq_iterations") {
        Value::Null => 10,
        v => int(v, "max_pq_iterations")?,
    };
    let random_seed = seed(inputs.arguments.get("random_seed"))?;

    if max_pq_iterations < 1 {
        return Err(Raised::value_error(format!(
            "'max_pq_iterations' must be greater than zero (got {max_pq_iterations})"
        )));
    }
    let task_type = task_type(&query, &sampler.edge_types)?;
    let num_train_examples = max_context_size(&run_mode)?;
    let num_test_examples = if indices.is_none() {
        max_test_size(task_type)
    } else {
        0
    };
    if task_type == "forecasting"
        && let Some(indices) = &indices
    {
        let keys = indices
            .iter()
            .map(index_key)
            .collect::<Result<Vec<_>, _>>()?;
        let distinct = crate::keys::unique(&keys).len();
        if distinct > 1 {
            return Err(Raised::value_error(format!(
                "Forecasting requires a single entity ID, but got {distinct} unique IDs in \
                 'indices'."
            )));
        }
    }
    let step_offset = query.target_end_offset.clone().unwrap_or_else(zero_offset);

    if anchor_time.is_none() && !anchor_is_entity {
        let mut anchor = default_anchor_time(sampler, db, Some(&query)).await?;
        if num_test_examples > 0 {
            anchor = shifted(&anchor, &times(&step_offset, query.num_forecasts)?, -1)?;
        }
        anchor_time = Some(anchor);
    }
    if let Some(anchor) = &anchor_time {
        if context_is_entity {
            return Err(Raised::value_error(
                "Anchor time 'entity' needs to be shared for context and prediction examples",
            ));
        }
        if context_anchor.is_none() {
            context_anchor = Some(shifted(anchor, &step_offset, -1)?);
        }
        given.extend(
            validate_time(
                sampler,
                db,
                &query,
                anchor,
                context_anchor.as_ref(),
                num_test_examples > 0,
            )
            .await?,
        );
    } else {
        if query.is_temporal() {
            return Err(Raised::value_error(
                "Anchor time 'entity' is only valid for static predictive queries",
            ));
        }
        if sampler.time_column(&query.entity_table).is_none() {
            return Err(Raised::value_error(format!(
                "Anchor time 'entity' requires the entity table '{}' to have a time column",
                query.entity_table
            )));
        }
        if context_anchor.is_some() {
            return Err(Raised::value_error(
                "Anchor time 'entity' needs to be shared for context and prediction examples",
            ));
        }
        context_is_entity = true;
    }

    // A forecast for given entities samples its context from the first.
    if task_type == "forecasting"
        && let Some(indices) = &indices
        && let Some(first) = indices.first()
    {
        query.rfm_entity_ids = Some(vec![match index_key(first)? {
            Key::Null => EntityId::None,
            Key::Int(i) => EntityId::Int(i),
            Key::Float(f) => EntityId::Float(f),
            Key::Text(s) => EntityId::Text(s),
            // JSON gives no decimal (`index_key`).
            Key::Decimal(..) => {
                return Err(Raised::new(
                    "TypeError",
                    "a decimal index is not supported by the core",
                ));
            }
        }]);
    }

    let as_anchor = |stamp: &Option<Stamp>, entity: bool| {
        if entity {
            Anchor::Entity
        } else {
            Anchor::At(stamp.as_ref().map_or(i64::MIN, |s| s.ns))
        }
    };
    let target = sample_target_with(
        sampler,
        db,
        &query,
        TargetArgs {
            num_train_examples,
            train_anchor: as_anchor(&context_anchor, context_is_entity),
            num_train_trials: max_pq_iterations * num_train_examples,
            num_test_examples,
            test_anchor: as_anchor(&anchor_time, anchor_is_entity),
            num_test_trials: max_pq_iterations * num_test_examples,
            random_seed,
        },
    )
    .await?;
    given.extend(target.warnings.clone());

    let train_pkey = target.train.entity_pkey.clone();
    let mut test_pkey = target.test.entity_pkey.clone();
    let mut test_time = Some((target.test.anchor_time.clone(), "ns".to_string()));
    let mut train_time = Some((target.train.anchor_time.clone(), "ns".to_string()));
    if num_test_examples == 0 {
        let indices = indices
            .as_ref()
            .ok_or_else(|| Raised::new("AssertionError", ""))?;
        test_pkey = indices_like(indices, &train_pkey, &query.entity_table)?;
        match &anchor_time {
            Some(anchor) => {
                test_time = Some(repeated_anchor_time(anchor, indices.len(), &sampler.host)?);
            }
            None => {
                train_time = None;
                test_time = None;
            }
        }
    }

    let entity_table_names = if is_link_pred(task_type) {
        let fkey = query
            .final_aggregation_column
            .clone()
            .ok_or_else(|| Raised::new("AssertionError", ""))?;
        let mut names = vec![query.entity_table.clone()];
        for edge in &sampler.edge_types {
            if fkey == format!("{}.{}", edge.0, edge.1) {
                names = vec![query.entity_table.clone(), edge.2.clone()];
            }
        }
        names
    } else {
        vec![query.entity_table.clone()]
    };

    let mut done = TaskDone {
        task_type: task_type.to_string(),
        entity_table_names,
        context: TaskSide {
            entity: train_pkey,
            target: Some(target.train.target.clone()),
            anchor_time: train_time.clone(),
        },
        prediction: TaskSide {
            entity: test_pkey,
            target: (num_test_examples > 0).then(|| target.test.target.clone()),
            anchor_time: test_time.clone(),
        },
        time_column: if train_time.is_some() {
            "ANCHOR_TIMESTAMP".to_string()
        } else {
            ENTITY_TIME.to_string()
        },
        num_forecasts: query.num_forecasts,
        step_size: if task_type == "forecasting" {
            Some(i64::try_from(step_offset.to_ns_from_2020()?).map_err(|_| {
                Raised::new("OverflowError", "Python int too large to convert to C long")
            })?)
        } else {
            None
        },
        lagged: None,
        warnings: Vec::new(),
    };

    check_task_table(&done)?;

    if lag_timesteps < 0 {
        return Err(Raised::value_error(format!(
            "'lag_timesteps' cannot be negative (got {lag_timesteps})"
        )));
    }
    if query.is_temporal()
        && matches!(
            task_type,
            "binary_classification" | "forecasting" | "multiclass_classification" | "regression"
        )
        && lag_timesteps > 0
    {
        done.lagged = Some(add_lagged_target(sampler, db, inputs, &done, lag_timesteps).await?);
        given.extend(sampler.take_warnings());
    }
    done.warnings = std::mem::take(given);
    Ok(done)
}

impl TaskDone {
    /// The task's two frames as `TaskTable` holds them: `ENTITY`, `TARGET`
    /// where there is one, `ANCHOR_TIMESTAMP` where the anchors are not the
    /// entities' own, then the lagged targets.
    pub fn frames(&self) -> Result<(Frame, Frame), Raised> {
        let context_rows = self.context.entity.len();
        let side = |side: &TaskSide, lagged: Option<Frame>| -> Result<Frame, Raised> {
            let mut frame = Frame::new(Vec::new(), Vec::new(), side.entity.len());
            frame.set("ENTITY", whole_decimals_as_ints(&side.entity));
            if let Some(target) = &side.target {
                frame.set("TARGET", target.clone());
            }
            if let Some((times, unit)) = &side.anchor_time {
                frame.set("ANCHOR_TIMESTAMP", stamps(times, unit)?);
            }
            if let Some(lagged) = lagged {
                for (name, column) in lagged.names.iter().zip(&lagged.columns) {
                    frame.set(name, column.clone());
                }
            }
            Ok(frame)
        };
        let (context_lag, prediction_lag) = match &self.lagged {
            None => (None, None),
            Some(lagged) => {
                let first: Vec<usize> = (0..context_rows.min(lagged.len())).collect();
                let rest: Vec<usize> = (context_rows.min(lagged.len())..lagged.len()).collect();
                (Some(lagged.take(&first)?), Some(lagged.take(&rest)?))
            }
        };
        Ok((
            side(&self.context, context_lag)?,
            side(&self.prediction, prediction_lag)?,
        ))
    }
}

/// A `datetime64[unit]` column of nanosecond times.
fn stamps(times: &[Option<i64>], unit: &str) -> Result<Series, Raised> {
    use arrow_array::{
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray,
    };
    let scaled = |per: i64| -> Vec<Option<i64>> {
        times.iter().map(|t| t.map(|v| v.div_euclid(per))).collect()
    };
    let array: arrow_array::ArrayRef = match unit {
        "ns" => Arc::new(TimestampNanosecondArray::from(times.to_vec())),
        "us" => Arc::new(TimestampMicrosecondArray::from(scaled(1_000))),
        "ms" => Arc::new(TimestampMillisecondArray::from(scaled(1_000_000))),
        "s" => Arc::new(TimestampSecondArray::from(scaled(1_000_000_000))),
        other => return Err(Raised::internal(format!("a time unit {other:?}"))),
    };
    Ok(Series::new(
        Dtype::Numpy(format!("datetime64[{unit}]")),
        array,
    ))
}

fn repeated_anchor_time(
    anchor: &Stamp,
    rows: usize,
    host: &Host,
) -> Result<(Vec<Option<i64>>, String), Raised> {
    let divisor = match anchor.unit.as_str() {
        "s" => 1_000_000_000,
        "ms" => 1_000_000,
        "us" => 1_000,
        "ns" => 1,
        _ => {
            return Err(Raised::value_error(
                "a timestamp scalar must name s, ms, us or ns",
            ));
        }
    };
    if anchor.ns % divisor != 0 {
        return Err(Raised::value_error(
            "a timestamp scalar must be exact in its stated unit",
        ));
    }
    let unit = host
        .timestamp_series_units
        .for_scalar(&anchor.unit)
        .ok_or_else(|| Raised::internal("a timestamp Series unit is missing"))?;
    Ok((vec![Some(anchor.ns); rows], unit.to_string()))
}

/// `add_lagged_target(task, query, lag_timesteps)`: the lagged targets of
/// every example, context first.
async fn add_lagged_target(
    sampler: &mut Sampler,
    db: &Db,
    inputs: &Inputs,
    task: &TaskDone,
    lag_timesteps: i64,
) -> Result<Frame, Raised> {
    let (Some((context_time, _)), Some((prediction_time, _))) =
        (&task.context.anchor_time, &task.prediction.anchor_time)
    else {
        return Err(Raised::value_error(
            "Task requires to have a time columns in order to add lagged target features",
        ));
    };
    if lag_timesteps <= 0 {
        return Err(Raised::value_error(format!(
            "'lag_timesteps' needs to be positive (got '{lag_timesteps}')"
        )));
    }
    let tree = inputs.arguments.get("query").unwrap_or(&Value::Null);
    let query = SamplerQuery::from_tree(tree)?;
    if !query.is_temporal() {
        return Err(Raised::value_error(
            "Lagged target features can only be added for temporal predictive queries",
        ));
    }
    let entity = concat(&[&task.context.entity, &task.prediction.entity])?;
    let anchor: Vec<Option<i64>> = context_time
        .iter()
        .chain(prediction_time.iter())
        .copied()
        .collect();
    super::lagged::lagged_targets(sampler, db, tree, &entity, &anchor, lag_timesteps).await
}

/// What `_get_context` gives besides the request's own pieces.
#[derive(Clone, Debug)]
pub struct ContextDone {
    pub subgraph: SubgraphDone,
    /// The task's feature columns, context rows first, where it has any.
    pub task_table: Option<Frame>,
}

/// `KumoRelational._get_context(task, run_mode, num_neighbors,
/// exclude_cols_dict, top_k, random_seed, _validate_references)`.
///
/// What it raises carries every warning it gave before it raised.
pub async fn get_context(inputs: Inputs, db: Db, host: Host) -> Result<ContextDone, Raised> {
    let mut sampler = Sampler::new(&inputs, host)?;
    let made = make_context(&mut sampler, &inputs, &db).await;
    match made {
        Ok(done) => Ok(done),
        Err(raised) => Err(raised.after(sampler.take_warnings())),
    }
}

/// [`get_context`] through a sampler already built (see
/// [`get_task_table_with`]); `inputs` holds the call's arguments.
pub async fn get_context_with(
    sampler: &mut Sampler,
    inputs: &Inputs,
    db: &Db,
) -> Result<ContextDone, Raised> {
    match make_context(sampler, inputs, db).await {
        Ok(done) => Ok(done),
        Err(raised) => Err(raised.after(sampler.take_warnings())),
    }
}

async fn make_context(
    sampler: &mut Sampler,
    inputs: &Inputs,
    db: &Db,
) -> Result<ContextDone, Raised> {
    let get = |name: &str| inputs.arguments.get(name).unwrap_or(&Value::Null);
    let task_type = optional_text(inputs.arguments.get("task_type"), "task_type")?
        .ok_or_else(|| Raised::value_error("no task_type"))?;
    let run_mode = optional_text(inputs.arguments.get("run_mode"), "run_mode")?
        .unwrap_or_else(|| "fast".into());
    let num_neighbors: Vec<i64> = match get("num_neighbors") {
        Value::Null => {
            let key = if is_link_pred(&task_type) {
                "fast"
            } else {
                run_mode.as_str()
            };
            default_num_neighbors(key)?.into_iter().take(2).collect()
        }
        Value::Array(items) => items
            .iter()
            .map(|v| int(v, "num_neighbors"))
            .collect::<Result<_, _>>()?,
        _ => return Err(Raised::value_error("num_neighbors is not a list")),
    };
    if num_neighbors.len() > MAX_HOPS {
        return Err(Raised::value_error(format!(
            "Cannot predict on subgraphs with more than {MAX_HOPS} hops (got {}). Reduce the \
             number of hops and try again. Please create a feature request at \
             'https://github.com/NVIDIA/structured-data-models-client' if you must go beyond \
             this for your use-case.",
            num_neighbors.len()
        )));
    }
    let entity_table_names: Vec<String> = match get("entity_table_names") {
        Value::Array(items) => items
            .iter()
            .map(|v| super::inputs::text(v, "a table"))
            .collect::<Result<_, _>>()?,
        _ => return Err(Raised::value_error("entity_table_names is not a list")),
    };
    let entity_table = entity_table_names
        .first()
        .cloned()
        .ok_or_else(|| Raised::new("IndexError", "tuple index out of range"))?;
    let entity_pkey = inputs.series_argument("entity_pkey", "entity_pkey_dtype")?;
    let num_context = int(get("num_context"), "num_context")?;
    let validate_references = get("validate_references").as_bool().unwrap_or(true);
    if validate_references {
        sampler.validate_entity_references(&entity_table, &entity_pkey)?;
    }

    let time_column = optional_text(inputs.arguments.get("time_column"), "time_column")?;
    let anchor: Option<Vec<Option<i64>>> = match time_column.as_deref() {
        Some(ENTITY_TIME) => {
            if sampler.time_column(&entity_table).is_none() {
                return Err(Raised::value_error(format!(
                    "The given anchor time requires the entity table '{entity_table}' to have a \
                     time column"
                )));
            }
            None
        }
        Some(_) => Some(super::subgraph::anchor_times(
            &inputs.series_argument("anchor_time", "anchor_time_dtype")?,
        )?),
        None => {
            let anchor = default_anchor_time(sampler, db, None).await?;
            Some(vec![Some(anchor.ns); entity_pkey.len()])
        }
    };
    let random_seed = match inputs.arguments.get("random_seed") {
        None => Some(sdm_dialects::int::Int::from(RANDOM_SEED)),
        given => seed(given)?,
    };
    let scored: Vec<i64> = (num_context..entity_pkey.len() as i64).collect();
    let exclude = get("exclude_cols_dict").clone();
    let subgraph = super::subgraph::sample_subgraph_with(
        sampler,
        db,
        &entity_table_names,
        &entity_pkey,
        anchor.as_deref(),
        &num_neighbors,
        &exclude,
        random_seed.as_ref(),
        Some(&scored),
    )
    .await?;
    // Today's method gave the subgraph's warnings as it sampled it, so they
    // come before anything it raises about it.
    match task_features(inputs, &subgraph, &task_type, &entity_table_names) {
        Ok(task_table) => Ok(ContextDone {
            subgraph,
            task_table,
        }),
        Err(raised) => Err(raised.after(subgraph.warnings)),
    }
}

/// What `_get_context` checks of the subgraph it sampled, and the task's
/// feature columns.
fn task_features(
    inputs: &Inputs,
    subgraph: &SubgraphDone,
    task_type: &str,
    entity_table_names: &[String],
) -> Result<Option<Frame>, Raised> {
    if subgraph.tables.len() > MAX_SUBGRAPH_TABLES {
        return Err(Raised::value_error(format!(
            "Cannot query from a graph with more than {MAX_SUBGRAPH_TABLES} tables (got {}). \
             Please create a feature request at \
             'https://github.com/NVIDIA/structured-data-models-client' if you must go beyond \
             this for your use-case.",
            subgraph.tables.len()
        )));
    }
    if is_link_pred(task_type)
        && !subgraph
            .tables
            .iter()
            .any(|t| Some(&t.name) == entity_table_names.last())
    {
        return Err(Raised::value_error(
            "Cannot perform link prediction on subgraphs without any historical target \
             entities. Please increase the number of hops and try again.",
        ));
    }
    Ok(match inputs.argument_tables.get("features") {
        None => None,
        Some(name) => Some(super::inputs::frame_of(inputs.data(name)?)),
    })
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Decimal128Array};
    use serde_json::json;

    use super::*;

    #[test]
    fn floating_indices_preserve_the_source_array_width() {
        use arrow_array::{Float32Array, Float64Array};

        let indices = vec![json!(1.25), json!("2.5"), json!(-3.75)];
        for (dtype, array) in [
            (
                Dtype::Numpy("float32".into()),
                Arc::new(Float32Array::from(vec![1.25, 2.5, -3.75])) as arrow_array::ArrayRef,
            ),
            (
                Dtype::Arrow,
                Arc::new(Float32Array::from(vec![1.25, 2.5, -3.75])),
            ),
            (
                Dtype::Extension("Float32".into()),
                Arc::new(Float32Array::from(vec![1.25, 2.5, -3.75])),
            ),
            (
                Dtype::Numpy("float64".into()),
                Arc::new(Float64Array::from(vec![1.25, 2.5, -3.75])),
            ),
        ] {
            let source = Series::new(dtype.clone(), array);
            let output = indices_like(&indices, &source, "products").unwrap();
            assert_eq!(output.dtype, dtype);
            assert_eq!(output.array.to_data(), source.array.to_data());
            assert!(indices_like(&[json!("not a number")], &source, "products").is_err());
        }
    }

    #[test]
    fn repeated_timestamps_use_only_bounded_constructor_facts() {
        use sdm_schema::dt::pandas::TimestampSeriesUnits;

        for names in [["s", "ms", "us", "ns"], ["ns", "ns", "ns", "ns"]] {
            let mut host = Host::assumed();
            host.timestamp_series_units = TimestampSeriesUnits::from_names(names).unwrap();
            for ((unit, nanos), expected) in [
                ("s", 1_700_000_000_000_000_000),
                ("ms", 1_700_000_000_123_000_000),
                ("us", 1_700_000_000_123_456_000),
                ("ns", 1_700_000_000_123_456_789),
            ]
            .into_iter()
            .zip(names)
            {
                let anchor = Stamp {
                    ns: nanos,
                    unit: unit.into(),
                };
                for rows in [0, 1, 3] {
                    let (times, inferred) = repeated_anchor_time(&anchor, rows, &host).unwrap();
                    assert_eq!(inferred, expected);
                    assert_eq!(times, vec![Some(nanos); rows]);
                    let column = stamps(&times, &inferred).unwrap();
                    assert_eq!(super::super::target::times_of(&column).unwrap(), times);
                }
                assert_eq!(anchor.unit, unit);
            }
            for nanos in [i64::MIN + 1, i64::MAX] {
                let anchor = Stamp {
                    ns: nanos,
                    unit: "ns".into(),
                };
                let (times, unit) = repeated_anchor_time(&anchor, 1, &host).unwrap();
                assert_eq!(unit, "ns");
                assert_eq!(times, [Some(nanos)]);
            }
            for unit in ["s", "ms", "us", "days"] {
                let anchor = Stamp {
                    ns: 1_700_000_000_123_456_789,
                    unit: unit.into(),
                };
                assert!(repeated_anchor_time(&anchor, 1, &host).is_err());
            }
        }
    }

    fn decimal_key(precision: u8, scale: i8) -> Series {
        let array = Decimal128Array::from(vec![Some(1)])
            .with_precision_and_scale(precision, scale)
            .unwrap();
        Series::new(Dtype::Arrow, Arc::new(array))
    }

    type Cast = Result<Vec<Option<i128>>, (String, String)>;

    /// What `pd.Series(indices, dtype=decimal128(precision, scale)[pyarrow])`
    /// gives, as unscaled digits, or what `_get_task_table` raises for it.
    fn cast_as(indices: Value, precision: u8, scale: i8) -> Cast {
        let Value::Array(indices) = indices else {
            unreachable!()
        };
        let key = decimal_key(precision, scale);
        match indices_like(&indices, &key, "products") {
            Ok(series) => {
                let array = series
                    .array
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .unwrap();
                assert_eq!(array.precision(), precision);
                assert_eq!(array.scale(), scale);
                Ok((0..array.len())
                    .map(|i| array.is_valid(i).then(|| array.value(i)))
                    .collect())
            }
            Err(raised) => Err((raised.class, raised.message)),
        }
    }

    fn cast(indices: Value) -> Cast {
        cast_as(indices, 10, 0)
    }

    fn refused_as(precision: u8, scale: i8, place: &str) -> Cast {
        Err((
            "ValueError".to_string(),
            format!(
                "'indices' must hold values matching the primary key of 'products', which is \
                 decimal128({precision}, {scale})[pyarrow]{place}"
            ),
        ))
    }

    fn refused(place: &str) -> Cast {
        refused_as(10, 0, place)
    }

    fn raised(class: &str, message: &str) -> Cast {
        Err((class.to_string(), message.to_string()))
    }

    fn digits(v: &[Option<i128>]) -> Cast {
        Ok(v.to_vec())
    }

    /// The codes `pd.Series(indices, dtype=CategoricalDtype(categories))`
    /// gives, `-1` where an index is no category.
    fn category_codes(indices: Value, categories: arrow_array::ArrayRef) -> Vec<i64> {
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int8Type;
        use arrow_array::{DictionaryArray, Int8Array};
        let Value::Array(indices) = indices else {
            unreachable!()
        };
        let key =
            DictionaryArray::<Int8Type>::try_new(Int8Array::from(vec![0_i8]), categories).unwrap();
        let series = indices_like(
            &indices,
            &Series::new(Dtype::Category, Arc::new(key)),
            "users",
        )
        .unwrap();
        assert_eq!(series.dtype, Dtype::Category);
        let codes = series.array.as_any_dictionary().normalized_keys();
        (0..series.len())
            .map(|i| {
                if series.array.is_valid(i) {
                    codes[i] as i64
                } else {
                    -1
                }
            })
            .collect()
    }

    #[test]
    fn indices_cast_to_categories_as_pandas_casts_them() {
        use arrow_array::{Float64Array, StringArray};
        // From Python: `pd.Series(indices, dtype=pd.CategoricalDtype(...))`
        // and its `cat.codes`.
        let ints = || -> arrow_array::ArrayRef { Arc::new(Int64Array::from(vec![1, 2, 3])) };
        let texts =
            || -> arrow_array::ArrayRef { Arc::new(StringArray::from(vec!["1", "2", "a"])) };
        let floats =
            || -> arrow_array::ArrayRef { Arc::new(Float64Array::from(vec![1.0, 2.5, 3.0])) };
        for (indices, int, text, float) in [
            (json!([1, 2]), vec![0, 1], vec![-1, -1], vec![0, -1]),
            (json!([1, "2"]), vec![0, -1], vec![-1, 1], vec![0, -1]),
            (json!(["1", "2"]), vec![-1, -1], vec![0, 1], vec![-1, -1]),
            (json!([1.0]), vec![0], vec![-1], vec![0]),
            (json!([2.5]), vec![-1], vec![-1], vec![1]),
            (json!([true]), vec![-1], vec![-1], vec![-1]),
            (json!([true, 2]), vec![0, 1], vec![-1, -1], vec![0, -1]),
            (json!([true, "a"]), vec![0, -1], vec![-1, 2], vec![0, -1]),
            (json!([null]), vec![-1], vec![-1], vec![-1]),
            (json!([1, null]), vec![0, -1], vec![-1, -1], vec![0, -1]),
            (json!(["a"]), vec![-1], vec![2], vec![-1]),
            (json!([1.5]), vec![-1], vec![-1], vec![-1]),
        ] {
            assert_eq!(category_codes(indices.clone(), ints()), int, "{indices}");
            assert_eq!(category_codes(indices.clone(), texts()), text, "{indices}");
            assert_eq!(
                category_codes(indices.clone(), floats()),
                float,
                "{indices}"
            );
        }
    }

    const BOOLS: &str = "Unsupported cast from bool to decimal128 using function cast_decimal";
    const OVERFLOW: &str = "Python int too large to convert to C long";

    #[test]
    fn decimal_refusals_keep_inference_context_but_not_successful_fallback_context() {
        for (indices, expected) in [
            (
                json!([1, "2"]),
                vec!["pyarrow.lib.ArrowInvalid", "pyarrow.lib.ArrowTypeError"],
            ),
            (
                json!(["1", 1]),
                vec!["pyarrow.lib.ArrowTypeError", "pyarrow.lib.ArrowTypeError"],
            ),
            (
                json!([1.5, 9007199254740993_u64]),
                vec!["pyarrow.lib.ArrowInvalid", "pyarrow.lib.ArrowTypeError"],
            ),
            (json!([10000000000_u64]), vec!["pyarrow.lib.ArrowInvalid"]),
            (json!(["1.5"]), vec!["pyarrow.lib.ArrowInvalid"]),
            (
                json!([{"$int": "9223372036854775808"}, "1"]),
                vec!["pyarrow.lib.ArrowInvalid"],
            ),
            (
                json!([{"$int": "340282366920938463463374607431768211456"}, "1"]),
                vec!["pyarrow.lib.ArrowInvalid"],
            ),
        ] {
            let failure =
                indices_like(indices.as_array().unwrap(), &decimal_key(10, 0), "products")
                    .err()
                    .unwrap();
            let classes: Vec<_> = failure
                .chain
                .iter()
                .map(|link| link["class"].as_str().unwrap())
                .collect();
            assert_eq!(classes, expected, "{indices}");
        }
    }

    #[test]
    fn decimal_indices_cast_as_pandas_casts_them() {
        // From Python, `_get_task_table`'s cast and `_first_uncastable`.
        assert_eq!(cast(json!([1, 2, 3])), digits(&[Some(1), Some(2), Some(3)]));
        assert_eq!(cast(json!([1.5, null])), digits(&[Some(2), None]));
        assert_eq!(cast(json!([null])), digits(&[None]));
        assert_eq!(cast(json!([])), digits(&[]));
        assert_eq!(cast(json!(["1.5"])), refused("; indices[0] is '1.5'"));
        assert_eq!(cast(json!([1, "2"])), refused(""));
        assert_eq!(cast(json!(["1", null])), digits(&[Some(1), None]));
        assert_eq!(cast(json!(["1", 1.5])), refused(""));
        assert_eq!(
            cast(json!([10_000_000_000_i64])),
            refused("; indices[0] is 10000000000")
        );
        assert_eq!(cast(json!([1, "x", true])), refused("; indices[1] is 'x'"));
        assert_eq!(
            cast(json!([true])),
            raised("ArrowNotImplementedError", BOOLS)
        );
        assert_eq!(
            cast(json!([1, true])),
            raised("ArrowNotImplementedError", BOOLS)
        );
        assert_eq!(
            cast(json!([9_223_372_036_854_775_808_u64])),
            raised("OverflowError", OVERFLOW)
        );
        // Refused as a whole ("PyLong is too large to fit int64"), then
        // the int alone raises.
        assert_eq!(
            cast(json!([1.5, 9_223_372_036_854_775_808_u64])),
            raised("OverflowError", OVERFLOW)
        );
        assert_eq!(
            cast(json!(["x", 9_223_372_036_854_775_808_u64])),
            refused("; indices[0] is 'x'")
        );
        assert_eq!(
            cast(json!([1, "x", 9_223_372_036_854_775_808_u64])),
            refused("; indices[1] is 'x'")
        );
        assert_eq!(
            cast(json!([1, 9_223_372_036_854_775_808_u64, "x"])),
            raised("OverflowError", OVERFLOW)
        );
    }

    #[test]
    fn decimal_indices_read_a_bool_after_a_float_as_a_float() {
        // pyarrow infers a type from the values up to the first bool, float
        // or text: a float first makes the array `double`, which a bool
        // converts to; a bool or an int first does not (REVIEW-r3 minor 1).
        assert_eq!(cast(json!([1.5, true])), digits(&[Some(2), Some(1)]));
        assert_eq!(
            cast(json!([1, 1.5, true])),
            digits(&[Some(1), Some(2), Some(1)])
        );
        assert_eq!(
            cast(json!([null, 1.5, true])),
            digits(&[None, Some(2), Some(1)])
        );
        assert_eq!(cast(json!([1.5, false])), digits(&[Some(2), Some(0)]));
        assert_eq!(
            cast(json!([1, true, 1.5])),
            raised("ArrowNotImplementedError", BOOLS)
        );
        assert_eq!(
            cast(json!([true, 1.5])),
            raised("ArrowNotImplementedError", BOOLS)
        );
    }

    #[test]
    fn decimal_indices_hold_ints_a_double_cannot() {
        // An int past 2**53 does not convert to `double`.
        assert_eq!(
            cast(json!([1.5, 9_007_199_254_740_993_i64])),
            refused("; indices[1] is 9007199254740993")
        );
        assert_eq!(
            cast_as(json!([1.5, -9_007_199_254_740_992_i64]), 20, 0),
            digits(&[Some(2), Some(-9_007_199_254_740_992)])
        );
        // Ints alone go straight to the decimal type, past int64 too, where
        // it is wide enough.
        let big = 9_223_372_036_854_775_808_u64;
        assert_eq!(cast_as(json!([big]), 38, 0), digits(&[Some(big.into())]));
        assert_eq!(
            cast_as(json!([null, big, 1]), 38, 0),
            digits(&[None, Some(big.into()), Some(1)])
        );
        assert_eq!(
            cast_as(json!([u64::MAX]), 20, 0),
            digits(&[Some(u64::MAX.into())])
        );
        assert_eq!(
            cast_as(json!([u64::MAX]), 19, 0),
            raised("OverflowError", OVERFLOW)
        );
        assert_eq!(
            cast_as(json!([big, true]), 38, 0),
            raised("OverflowError", OVERFLOW)
        );
        assert_eq!(cast_as(json!([big, 1.5]), 38, 0), refused_as(38, 0, ""));
    }

    #[test]
    fn decimal_indices_cast_to_other_scales() {
        assert_eq!(
            cast_as(json!([1, 2]), 12, 2),
            digits(&[Some(100), Some(200)])
        );
        assert_eq!(
            cast_as(json!([1.5, 1]), 12, 2),
            digits(&[Some(150), Some(100)])
        );
        assert_eq!(
            cast_as(json!(["1.5", null]), 12, 2),
            digits(&[Some(150), None])
        );
        // A negative scale: ints go straight there, or not at all.
        assert_eq!(
            cast_as(json!([100, null]), 10, -2),
            digits(&[Some(1), None])
        );
        assert_eq!(
            cast_as(json!([100, 1.5]), 10, -2),
            digits(&[Some(1), Some(0)])
        );
        assert_eq!(
            cast_as(json!([150]), 10, -2),
            refused_as(10, -2, "; indices[0] is 150")
        );
        assert_eq!(
            cast_as(json!([100, "x"]), 10, -2),
            refused_as(10, -2, "; indices[1] is 'x'")
        );
        assert_eq!(cast_as(json!(["100"]), 10, -2), digits(&[Some(1)]));
    }
}
