// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The prediction request's payload, laid out as the reference's engine
//! lays it out (`predict_request_to_json`): the task, the schema, the
//! context and predict halves, each an instance table and the related
//! tables sampled for it, the output fields, the inference options and the
//! metadata.
//!
//! Each column's wire dtype comes from the Arrow type sampled, by the
//! reference's rule for that type (`_dtype_name`): integers as `int64`,
//! floats at their width, text as `string`, booleans, and times with no
//! zone as `timestamp[us]`; every cell is written under its column's dtype
//! (`_cell_json_value`): an integer past the JSON-safe range as its
//! digits, a missing value or a float NaN as null, a time as its UTC wall
//! clock to the microsecond. Any other type is refused by name, as is a
//! value the dtype cannot hold, as the reference refuses it.

use arrow_array::cast::AsArray;
use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{Array, ArrayRef};
use arrow_schema::{DataType, TimeUnit};
use sdm_contract::ops::Stype;
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

use crate::json::Json;

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
/// `pd.Timestamp.min` in nanoseconds: an anchor every example has at this
/// value is no anchor at all.
const NO_ANCHOR: i64 = -9_223_372_036_854_775_807;
/// The largest integer a JSON number carries exactly in every reader.
const JSON_SAFE: i128 = 9_007_199_254_740_991;

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
    pub columns: Vec<(String, ArrayRef)>,
    /// The semantic type of each feature column the table sends.
    pub stypes: Vec<(String, Stype)>,
}

/// A foreign key between two sampled tables: for each row of `src`, the
/// row of `dst` it refers to, if any.
#[derive(Debug, Clone)]
pub struct SampledLink {
    pub src: String,
    pub fkey: String,
    pub dst: String,
    pub pairs: Vec<(usize, usize)>,
}

/// What a regression's request is built from.
#[derive(Debug, Clone)]
pub struct Sampled {
    /// The examples with a label, then those to predict.
    pub num_train: usize,
    pub num_test: usize,
    /// Each example's anchor time, nanoseconds since the epoch.
    pub anchors: Vec<i64>,
    /// The labels of the examples with one, at float32.
    pub targets: Vec<f32>,
    /// The entity table first, then each table sampled, in the order
    /// sampled.
    pub tables: Vec<SampledTable>,
    pub links: Vec<SampledLink>,
    pub entity_table: String,
    /// The query, as the reference writes it back.
    pub source_query: String,
}

fn invalid(message: &str, column: &str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        message,
    )
    .with_field("column", Detail::text(column))
}

fn internal(message: &str) -> CoreError {
    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message)
}

/// The wire dtype of a column of `data_type`, or a refusal naming the rule.
pub fn wire_dtype(data_type: &DataType) -> Result<&'static str, CoreError> {
    Ok(match data_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => "int64",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::Utf8 => "string",
        DataType::Boolean => "bool",
        DataType::Timestamp(_, None) => "timestamp[us]",
        _ => return Err(sdm_engine::not_yet("wire_dtype")),
    })
}

/// An integer cell: past the JSON-safe range, its digits as text; past
/// int64, refused, as the reference refuses it.
fn int_cell(value: i128, column: &str) -> Result<Json, CoreError> {
    if value < i128::from(i64::MIN) || value > i128::from(i64::MAX) {
        return Err(invalid("a value is outside the range int64 holds", column));
    }
    if value.abs() > JSON_SAFE {
        return Ok(Json::Text(value.to_string()));
    }
    Ok(Json::Int(value as i64))
}

fn float_cell(value: f64, column: &str) -> Result<Json, CoreError> {
    if value.is_nan() {
        return Ok(Json::Null);
    }
    if value.is_infinite() {
        return Err(invalid(
            "a value is not finite, and a request holds finite numbers or nulls",
            column,
        ));
    }
    Ok(Json::Float(value))
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
            ErrorKind::Unsupported,
            ErrorCode::ValueOutOfRange,
            "a time is outside what nanoseconds since the epoch hold",
        )
        .with_field("column", Detail::text(column))
    })
}

/// Row `row` of `array`, written under its wire dtype.
pub fn cell(array: &ArrayRef, row: usize, column: &str) -> Result<Json, CoreError> {
    if array.is_null(row) {
        return Ok(Json::Null);
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
        DataType::Utf8 => Ok(Json::text(array.as_string::<i32>().value(row))),
        DataType::Boolean => Ok(Json::Bool(array.as_boolean().value(row))),
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
            Ok(Json::Text(iso_microseconds(to_ns(value, *unit, column)?)))
        }
        _ => Err(sdm_engine::not_yet("wire_dtype")),
    }
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
    dtype: &'static str,
    cells: Vec<Json>,
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
    fn table(&self, keep: impl Fn(usize) -> bool, drop: Option<&str>) -> Json {
        let columns: Vec<&FrameColumn> = self
            .columns
            .iter()
            .filter(|c| Some(c.name.as_str()) != drop)
            .collect();
        let rows = (0..self.rows)
            .filter(|row| keep(*row))
            .map(|row| Json::Array(columns.iter().map(|c| c.cells[row].clone()).collect()))
            .collect();
        Json::object([
            ("format", Json::text("arrays")),
            (
                "columns",
                Json::texts(columns.iter().map(|c| c.name.as_str())),
            ),
            ("rows", Json::Array(rows)),
        ])
    }
}

fn int_column(name: &str, values: impl Iterator<Item = usize>) -> FrameColumn {
    FrameColumn {
        name: name.to_owned(),
        dtype: "int64",
        cells: values.map(|v| Json::Int(v as i64)).collect(),
    }
}

fn arrow_column(name: &str, array: &ArrayRef, rows: &[usize]) -> Result<FrameColumn, CoreError> {
    Ok(FrameColumn {
        name: name.to_owned(),
        dtype: wire_dtype(array.data_type())?,
        cells: rows
            .iter()
            .map(|&row| cell(array, row, name))
            .collect::<Result<_, _>>()?,
    })
}

fn stype_name(stype: Stype) -> &'static str {
    match stype {
        Stype::Numerical => "numerical",
        Stype::Categorical => "categorical",
        Stype::Multicategorical => "multicategorical",
        Stype::Id => "ID",
        Stype::Text => "text",
        Stype::Timestamp => "timestamp",
        Stype::Sequence => "sequence",
        Stype::Image => "image",
        Stype::Unsupported => "unsupported",
    }
}

/// A table's schema: each column's dtype, its semantic type where one is
/// given, and not nullable where it is part of the key.
fn table_schema(frame: &Frame, stypes: &[(String, Stype)], key: &[String]) -> Json {
    let columns = frame
        .columns
        .iter()
        .map(|c| {
            let mut spec = vec![("dtype".to_owned(), Json::text(c.dtype))];
            if let Some((_, stype)) = stypes.iter().rev().find(|(n, _)| *n == c.name) {
                spec.push(("stype".to_owned(), Json::text(stype_name(*stype))));
            }
            if key.contains(&c.name) {
                spec.push(("nullable".to_owned(), Json::Bool(false)));
            }
            (c.name.clone(), Json::Object(spec))
        })
        .collect();
    let primary_key = match key {
        [one] => Json::text(one),
        several => Json::texts(several.iter().map(String::as_str)),
    };
    Json::object([
        ("columns", Json::Object(columns)),
        ("primary_key", primary_key),
    ])
}

/// The payload of a regression's request, as the reference's engine
/// builds it for one batch (`predict_request_to_json`), as its JSON text.
pub fn regression(sampled: &Sampled) -> Result<Vec<u8>, CoreError> {
    let examples = sampled.num_train + sampled.num_test;
    if sampled.anchors.len() != examples || sampled.targets.len() != sampled.num_train {
        return Err(internal("the examples, anchors and labels do not align"));
    }

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
        .find(|(n, _)| n == key)
        .map(|(_, a)| a)
        .ok_or_else(|| internal("the entity table's key was not sampled"))?;
    let mut first = vec![None; examples];
    for (row, &instance) in entity.batch.iter().enumerate() {
        if instance < examples && first[instance].is_none() {
            first[instance] = Some(row);
        }
    }
    let mut occupied = vec![INSTANCE_ID.to_owned(), TARGET.to_owned()];
    let reference = unique_name(&occupied, &format!("{ENTITY_REFERENCE}_0"));
    occupied.push(reference.clone());
    let mut instance = Frame {
        columns: vec![int_column(INSTANCE_ID, 0..examples)],
        rows: examples,
    };
    let reference_cells = first
        .iter()
        .map(|row| match row {
            Some(row) => cell(key_array, *row, &reference),
            None => Ok(Json::Null),
        })
        .collect::<Result<Vec<_>, _>>()?;
    instance.columns.push(FrameColumn {
        name: reference.clone(),
        dtype: wire_dtype(key_array.data_type())?,
        cells: reference_cells,
    });
    let anchor_column =
        if !sampled.anchors.is_empty() && sampled.anchors.iter().any(|a| *a != NO_ANCHOR) {
            let name = unique_name(&occupied, ANCHOR_TIME);
            occupied.push(name.clone());
            instance.columns.push(FrameColumn {
                name: name.clone(),
                dtype: "timestamp[us]",
                cells: sampled
                    .anchors
                    .iter()
                    .map(|a| Json::Text(iso_microseconds(*a)))
                    .collect(),
            });
            Some(name)
        } else {
            None
        };
    let mut target_cells = Vec::with_capacity(examples);
    for value in &sampled.targets {
        target_cells.push(float_cell(f64::from(*value), TARGET)?);
    }
    target_cells.resize(examples, Json::Null);
    instance.columns.push(FrameColumn {
        name: TARGET.to_owned(),
        dtype: "float32",
        cells: target_cells,
    });

    // The related tables: each row with the example it belongs to, and a
    // key of its own where the table sends none.
    let mut frames: Vec<Related> = Vec::new();
    for table in &sampled.tables {
        let rows: Vec<usize> = (0..table.batch.len()).collect();
        let mut frame = Frame {
            columns: vec![int_column(INSTANCE_ID, table.batch.iter().copied())],
            rows: rows.len(),
        };
        for (name, array) in &table.columns {
            if array.len() != rows.len() {
                return Err(internal("a sampled column does not align with its rows"));
            }
            frame.columns.push(arrow_column(name, array, &rows)?);
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
        relationships.push(Json::object([
            ("source_columns", Json::texts([reference.as_str()])),
            ("target_table", Json::text(&sampled.entity_table)),
            ("target_columns", Json::texts([target_column.as_str()])),
        ]));
    }
    for link in &sampled.links {
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
        for &(row, at) in &link.pairs {
            if row < src_rows && at < dst_rows {
                values[row] = Some(key_cells[at].clone());
            }
        }
        if values.iter().all(Option::is_none) {
            continue;
        }
        let cells = values
            .into_iter()
            .map(|v| v.unwrap_or(Json::Null))
            .collect();
        let column = FrameColumn {
            name: link.fkey.clone(),
            dtype: key_dtype,
            cells,
        };
        let frame = &mut frames[src].1;
        match frame.columns.iter_mut().find(|c| c.name == link.fkey) {
            Some(slot) => *slot = column,
            None => frame.columns.push(column),
        }
        frames[src].3.push((link.fkey.clone(), Stype::Id));
        relationships.push(Json::object([
            ("source_table", Json::text(&link.src)),
            ("source_columns", Json::texts([link.fkey.as_str()])),
            ("target_table", Json::text(&link.dst)),
            ("target_columns", Json::texts([dst_key.as_str()])),
        ]));
    }

    let num_train = sampled.num_train;
    let mut instance_stypes = vec![
        (INSTANCE_ID.to_owned(), Stype::Id),
        (reference.clone(), Stype::Id),
    ];
    if let Some(anchor) = &anchor_column {
        instance_stypes.push((anchor.clone(), Stype::Timestamp));
    }
    instance_stypes.push((TARGET.to_owned(), Stype::Numerical));
    let mut instance_schema = table_schema(&instance, &instance_stypes, &[INSTANCE_ID.to_owned()]);
    if let Some(anchor) = &anchor_column
        && let Json::Object(top) = &mut instance_schema
        && let Some((_, Json::Object(columns))) = top.iter_mut().find(|(k, _)| k == "columns")
        && let Some((_, spec)) = columns.iter_mut().find(|(k, _)| k == anchor)
    {
        spec.set("nullable", Json::Bool(false));
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
    let half = |context: bool| -> Json {
        let in_half = |instance: usize| (instance < num_train) == context;
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
        Json::object([
            ("instance_table", instance_table),
            ("related_tables", Json::Object(related)),
        ])
    };

    let mut task = vec![
        ("kind".to_owned(), Json::text("regression")),
        (
            "target".to_owned(),
            Json::object([
                ("column_name", Json::text(TARGET)),
                ("dtype", Json::text("float32")),
            ]),
        ),
        (
            "entity_table_names".to_owned(),
            Json::texts([sampled.entity_table.as_str()]),
        ),
    ];
    if let Some(anchor) = &anchor_column {
        task.push(("anchor_time_column".to_owned(), Json::text(anchor)));
    }
    let mut metadata = Vec::new();
    if !sampled.source_query.is_empty() {
        metadata.push(("source_query".to_owned(), Json::text(&sampled.source_query)));
    }
    metadata.push((
        "num_context_examples".to_owned(),
        Json::Int(num_train as i64),
    ));
    metadata.push((
        "num_prediction_examples".to_owned(),
        Json::Int(sampled.num_test as i64),
    ));
    let payload = Json::object([
        ("model", Json::text("kumo-relational")),
        ("task", Json::Object(task)),
        (
            "schema",
            Json::object([
                ("instance_table", instance_schema),
                ("related_tables", Json::Object(related_schema)),
                ("relationships", Json::Array(relationships)),
            ]),
        ),
        ("context", half(true)),
        ("predict", half(false)),
        (
            "output",
            Json::object([("fields", Json::texts(["prediction"]))]),
        ),
        (
            "inference",
            Json::object([
                ("run_mode", Json::text("fast")),
                ("use_prediction_time", Json::Bool(false)),
                (
                    "inference_config",
                    Json::object([
                        ("kind", Json::text("regression")),
                        ("num_estimators", Json::Int(1)),
                        ("column_shuffle", Json::Bool(false)),
                        ("category_shuffle", Json::Bool(false)),
                        ("hop_shuffle", Json::Bool(false)),
                        ("target_transforms", Json::texts(["quantile"])),
                        ("output_type", Json::text("median")),
                    ]),
                ),
            ]),
        ),
        ("metadata", Json::Object(metadata)),
    ]);
    payload.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::iso_microseconds;

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
}
