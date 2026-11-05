// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inverse of the seven wire column types, for inspection only.
use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch,
    RecordBatchOptions, StringArray, TimestampMicrosecondArray,
};
use arrow_schema::{Field, Schema, TimeUnit};
use indexmap::IndexMap;
use sdm_contract::split::{Half, Selection};
use sdm_contract::{CoreError, Detail, Sentence};
use sdm_engine::time::{Civil, Instant};
use sdm_wire::{
    json::{self, Value},
    payload::Dtype,
};
use std::sync::Arc;

fn invalid(reason: &str) -> CoreError {
    CoreError::new(Sentence::RequestSplitInvalid, reason).with_field("reason", Detail::text(reason))
}
fn object(v: &Value) -> Result<&[(String, Value)], CoreError> {
    if let Value::Object(v) = v {
        Ok(v)
    } else {
        Err(invalid("expected an object"))
    }
}
fn array(v: &Value) -> Result<&[Value], CoreError> {
    if let Value::Array(v) = v {
        Ok(v)
    } else {
        Err(invalid("expected an array"))
    }
}
fn text(v: &Value) -> Result<&str, CoreError> {
    if let Value::Text(v) = v {
        Ok(v)
    } else {
        Err(invalid("expected text"))
    }
}
fn field<'a>(v: &'a Value, name: &str) -> Result<&'a Value, CoreError> {
    v.get(name)
        .ok_or_else(|| invalid("missing table or schema field"))
}
fn dtype(v: &Value) -> Result<Dtype, CoreError> {
    let name = text(v)?;
    [
        Dtype::Int64,
        Dtype::Float32,
        Dtype::Float64,
        Dtype::String,
        Dtype::StringList,
        Dtype::Bool,
        Dtype::TimestampMicros,
    ]
    .into_iter()
    .find(|v| v.wire_name() == name)
    .ok_or_else(|| invalid("unknown wire dtype"))
}
#[derive(Clone, Debug)]
enum Cell {
    Null,
    Int(i64),
    Float(f64),
    Text(String),
    Bool(bool),
    Words(Vec<Option<String>>),
    Time(i64),
}
fn timestamp(v: &str) -> Result<i64, CoreError> {
    let v = v
        .strip_suffix('Z')
        .ok_or_else(|| invalid("timestamp lacks writer UTC suffix"))?;
    let (date, time) = v
        .split_once('T')
        .ok_or_else(|| invalid("invalid timestamp"))?;
    let (rest, day) = date
        .rsplit_once('-')
        .ok_or_else(|| invalid("invalid timestamp date"))?;
    let (year, month) = rest
        .rsplit_once('-')
        .ok_or_else(|| invalid("invalid timestamp date"))?;
    let nums = time.split(':').collect::<Vec<_>>();
    if nums.len() != 3 {
        return Err(invalid("invalid timestamp time"));
    }
    let (seconds, fraction) = nums[2]
        .split_once('.')
        .ok_or_else(|| invalid("timestamp lacks microseconds"))?;
    if fraction.len() != 6 || !fraction.bytes().all(|v| v.is_ascii_digit()) {
        return Err(invalid("invalid timestamp precision"));
    }
    let civil = Civil::new(
        i64::from(year.parse::<i32>().map_err(|_| invalid("invalid year"))?),
        month.parse().map_err(|_| invalid("invalid month"))?,
        day.parse().map_err(|_| invalid("invalid day"))?,
        nums[0].parse().map_err(|_| invalid("invalid hour"))?,
        nums[1].parse().map_err(|_| invalid("invalid minute"))?,
        seconds.parse().map_err(|_| invalid("invalid second"))?,
        fraction
            .parse::<u32>()
            .map_err(|_| invalid("invalid fraction"))?
            * 1000,
    )
    .ok_or_else(|| invalid("invalid calendar time"))?;
    Instant::from_civil(&civil, 0)
        .to_count(TimeUnit::Microsecond, "timestamp")
        .map_err(|_| invalid("timestamp exceeds microsecond range"))
}
fn decode(v: &Value, kind: Dtype) -> Result<Cell, CoreError> {
    if matches!(v, Value::Null) {
        return Ok(Cell::Null);
    }
    Ok(match (kind, v) {
        (Dtype::Int64, Value::Number(n) | Value::Text(n)) => {
            Cell::Int(n.parse().map_err(|_| invalid("invalid int64 cell"))?)
        }
        (Dtype::Float32 | Dtype::Float64, Value::Number(n)) => {
            let f = n
                .parse::<f64>()
                .map_err(|_| invalid("invalid float cell"))?;
            if !f.is_finite() || (kind == Dtype::Float32 && !(f as f32).is_finite()) {
                return Err(invalid("non-finite float cell"));
            }
            Cell::Float(if kind == Dtype::Float32 {
                f64::from(f as f32)
            } else {
                f
            })
        }
        (Dtype::String, Value::Text(s)) => Cell::Text(s.clone()),
        (Dtype::Bool, Value::Bool(b)) => Cell::Bool(*b),
        (Dtype::TimestampMicros, Value::Text(s)) => Cell::Time(timestamp(s)?),
        (Dtype::StringList, Value::Array(a)) => Cell::Words(
            a.iter()
                .map(|v| match v {
                    Value::Null => Ok(None),
                    Value::Text(s) => Ok(Some(s.clone())),
                    _ => Err(invalid("non-text string-list element")),
                })
                .collect::<Result<_, _>>()?,
        ),
        _ => return Err(invalid("cell contradicts its declared wire dtype")),
    })
}
#[derive(Default)]
struct Table {
    columns: IndexMap<String, Dtype>,
    rows: Vec<IndexMap<String, Cell>>,
}
fn table(schema: &Value, data: Option<&Value>) -> Result<Table, CoreError> {
    let declarations = object(field(schema, "columns")?)?;
    let mut seen = std::collections::BTreeSet::new();
    for (name, _) in declarations {
        if !seen.insert(name) {
            return Err(invalid("duplicate schema column"));
        }
    }
    let mut columns = IndexMap::new();
    // Preserve encoded headers; add absent declarations for typed empty columns.
    if let Some(data) = data {
        if text(field(data, "format")?)? != "arrays" {
            return Err(invalid("unknown row format"));
        }
        for name in array(field(data, "columns")?)? {
            let name = text(name)?;
            let spec = declarations
                .iter()
                .find(|(n, _)| n == name)
                .ok_or_else(|| invalid("row column lacks declaration"))?;
            if columns
                .insert(name.to_owned(), dtype(field(&spec.1, "dtype")?)?)
                .is_some()
            {
                return Err(invalid("duplicate row column"));
            }
        }
    }
    for (name, spec) in declarations {
        let d = dtype(field(spec, "dtype")?)?;
        columns.entry(name.clone()).or_insert(d);
    }
    let mut rows = Vec::new();
    if let Some(data) = data {
        let names = array(field(data, "columns")?)?;
        for row in array(field(data, "rows")?)? {
            let row = array(row)?;
            if row.len() != names.len() {
                return Err(invalid("row width differs from headers"));
            }
            rows.push(
                names
                    .iter()
                    .zip(row)
                    .map(|(name, value)| {
                        let name = text(name)?;
                        Ok((name.to_owned(), decode(value, columns[name])?))
                    })
                    .collect::<Result<_, CoreError>>()?,
            );
        }
    }
    Ok(Table { columns, rows })
}
fn bounds(tables: &IndexMap<String, Table>) -> Result<Option<(i128, i128)>, CoreError> {
    let mut out: Option<(i128, i128)> = None;
    for table in tables.values() {
        for row in &table.rows {
            let pair = match row.get("instance_id") {
                Some(Cell::Int(n)) => Some((i128::from(*n), i128::from(*n))),
                Some(Cell::Float(n)) => {
                    if n.abs() >= i128::MAX as f64 {
                        return Err(invalid("numeric instance ID exceeds rebasing range"));
                    }
                    Some((n.floor() as i128, n.ceil() as i128))
                }
                _ => None,
            };
            if let Some((lo, hi)) = pair {
                out = Some(out.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            }
        }
    }
    Ok(out)
}
fn shift(tables: &mut IndexMap<String, Table>, offset: i128) -> Result<(), CoreError> {
    for table in tables.values_mut() {
        for row in &mut table.rows {
            match row.get_mut("instance_id") {
                Some(Cell::Int(n)) => {
                    *n = i64::try_from(
                        i128::from(*n)
                            .checked_add(offset)
                            .ok_or_else(|| invalid("instance ID overflow"))?,
                    )
                    .map_err(|_| invalid("adjusted instance ID exceeds int64"))?
                }
                Some(Cell::Float(n)) if offset != 0 => {
                    // Only exactly representable offsets/results are accepted. No rounded links.
                    let d = offset as f64;
                    let adjusted = *n + d;
                    let b = adjusted - *n;
                    let residual = (*n - (adjusted - b)) + (d - b);
                    if d.abs() >= i128::MAX as f64
                        || d as i128 != offset
                        || !adjusted.is_finite()
                        || residual != 0.0
                        || (table.columns["instance_id"] == Dtype::Float32
                            && f64::from(adjusted as f32) != adjusted)
                    {
                        return Err(invalid("adjusted floating instance ID is not exact"));
                    }
                    *n = adjusted;
                }
                _ => {}
            }
        }
    }
    Ok(())
}
fn append(to: &mut Table, from: Table) -> Result<(), CoreError> {
    for (name, kind) in from.columns {
        if to.columns.get(&name).is_some_and(|v| *v != kind) {
            return Err(invalid("conflicting column dtype across payloads"));
        }
        to.columns.entry(name).or_insert(kind);
    }
    to.rows.extend(from.rows);
    Ok(())
}
fn arrow(table: Table) -> Result<RecordBatch, CoreError> {
    let mut fields = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    for (name, kind) in &table.columns {
        let values = table
            .rows
            .iter()
            .map(|r| r.get(name).unwrap_or(&Cell::Null));
        macro_rules! primitive {
            ($array:ident,$variant:ident) => {
                Arc::new($array::from(
                    values
                        .map(|v| {
                            if let Cell::$variant(v) = v {
                                Some(*v)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef
            };
        }
        let array = match kind {
            Dtype::Int64 => primitive!(Int64Array, Int),
            Dtype::Float64 => primitive!(Float64Array, Float),
            Dtype::Float32 => Arc::new(Float32Array::from(
                values
                    .map(|v| {
                        if let Cell::Float(v) = v {
                            Some(*v as f32)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Dtype::Bool => primitive!(BooleanArray, Bool),
            Dtype::TimestampMicros => primitive!(TimestampMicrosecondArray, Time),
            Dtype::String => Arc::new(StringArray::from(
                values
                    .map(|v| {
                        if let Cell::Text(v) = v {
                            Some(v.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Dtype::StringList => {
                let mut b = ListBuilder::new(StringBuilder::new());
                for v in values {
                    if let Cell::Words(words) = v {
                        for word in words {
                            b.values().append_option(word.as_deref());
                        }
                        b.append(true);
                    } else {
                        b.append(false);
                    }
                }
                Arc::new(b.finish()) as ArrayRef
            }
        };
        fields.push(Field::new(name, array.data_type().clone(), true));
        arrays.push(array);
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(table.rows.len())),
    )
    .map_err(|_| invalid("cannot construct inspected Arrow table"))
}

/// Names are ordered; only requested output tables allocate Arrow arrays.
pub struct Output {
    pub instance: Option<RecordBatch>,
    pub related: Vec<(String, Option<RecordBatch>)>,
}
pub fn read(payloads: &[&[u8]], half: Half, selection: &Selection) -> Result<Output, CoreError> {
    let mut combined: IndexMap<String, Table> = IndexMap::new();
    combined.insert(String::new(), Table::default());
    let mut next = 0i128;
    let mut numbered = false;
    for bytes in payloads {
        let value = json::read(bytes).map_err(|_| invalid("payload is not JSON"))?;
        let schema = field(&value, "schema")?;
        let half = value.get(match half {
            Half::Context => "context",
            Half::Predict => "predict",
        });
        if let Some(half) = half {
            object(half)?;
        }
        let mut tables = IndexMap::new();
        tables.insert(
            String::new(),
            table(
                field(schema, "instance_table")?,
                half.and_then(|h| h.get("instance_table")),
            )?,
        );
        let related = object(field(schema, "related_tables")?)?;
        let data = half.and_then(|h| h.get("related_tables"));
        if let Some(data) = data {
            for (name, _) in object(data)? {
                if !related.iter().any(|(n, _)| n == name) {
                    return Err(invalid("related table lacks schema"));
                }
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for (name, schema) in related {
            if name.is_empty() || !seen.insert(name) {
                return Err(invalid("empty or duplicate related table name"));
            }
            tables.insert(name.clone(), table(schema, data.and_then(|d| d.get(name)))?);
        }
        if let Some((low, high)) = bounds(&tables)? {
            let offset = if numbered {
                next.checked_sub(low)
                    .ok_or_else(|| invalid("instance offset overflow"))?
            } else {
                0
            };
            let by_rows = next
                .checked_add(tables[""].rows.len() as i128)
                .ok_or_else(|| invalid("instance row count overflow"))?;
            let by_high = high
                .checked_add(offset)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| invalid("instance high bound overflow"))?;
            shift(&mut tables, offset)?;
            next = by_rows.max(by_high);
            numbered = true;
        }
        for (name, table) in tables {
            append(combined.entry(name).or_default(), table)?;
        }
    }
    if let Selection::Related { name } = selection
        && (name.is_empty() || !combined.contains_key(name))
    {
        return Err(invalid("unknown related table"));
    }
    let mut instance = None;
    let mut related = Vec::new();
    for (name, table) in combined {
        let selected = matches!(selection, Selection::All)
            || name.is_empty() && matches!(selection, Selection::Instance)
            || matches!(selection,Selection::Related{name:selected} if selected==&name);
        let data = if selected { Some(arrow(table)?) } else { None };
        if name.is_empty() {
            instance = data;
        } else {
            related.push((name, data));
        }
    }
    Ok(Output { instance, related })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, ListArray};
    use serde_json::json;

    fn payload(ids: serde_json::Value, related: serde_json::Value) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema":{"instance_table":{"columns":{"instance_id":{"dtype":"int64"},"value":{"dtype":"string"}}},
                "related_tables":{"events":{"columns":{"instance_id":{"dtype":"int64"},"words":{"dtype":"stringlist"}}}}},
            "context":{"instance_table":{"format":"arrays","columns":["instance_id","value"],"rows":ids},
                "related_tables":{"events":{"format":"arrays","columns":["instance_id","words"],"rows":related}}}
        })).unwrap()
    }
    fn ints(batch: &RecordBatch) -> Vec<Option<i64>> {
        batch
            .column_by_name("instance_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .iter()
            .collect()
    }
    #[test]
    fn joins_numeric_string_ids_and_preserves_nullable_words() {
        let first = payload(
            json!([["9007199254740993", "a"]]),
            json!([["9007199254740993", ["word", null]]]),
        );
        let second = payload(
            json!([[-2, "b"], [null, "c"]]),
            json!([[-3, []], [-2, null]]),
        );
        let out = read(&[&first, &second], Half::Context, &Selection::All).unwrap();
        assert_eq!(
            ints(out.instance.as_ref().unwrap()),
            vec![Some(9007199254740993), Some(9007199254740995), None]
        );
        let events = out.related[0].1.as_ref().unwrap();
        assert_eq!(
            ints(events),
            vec![
                Some(9007199254740993),
                Some(9007199254740994),
                Some(9007199254740995)
            ]
        );
        let words = events
            .column_by_name("words")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        assert!(words.value(0).is_null(1));
        assert!(words.is_null(2));
        let selected = read(
            &[&first, &second],
            Half::Context,
            &Selection::Related {
                name: "events".into(),
            },
        )
        .unwrap();
        assert!(selected.instance.is_none());
        assert_eq!(selected.related[0].1.as_ref().unwrap(), events);
    }
    #[test]
    fn absent_half_preserves_types_and_names_have_no_arrow() {
        let p = payload(json!([[1, "a"]]), json!([]));
        let out = read(&[&p], Half::Predict, &Selection::All).unwrap();
        assert_eq!(out.instance.unwrap().num_rows(), 0);
        assert_eq!(out.related[0].1.as_ref().unwrap().num_columns(), 2);
        let names = read(&[&p], Half::Context, &Selection::Names).unwrap();
        assert!(names.instance.is_none());
        assert_eq!(names.related[0].0, "events");
        assert!(names.related[0].1.is_none());
    }
    #[test]
    fn refuses_dtype_mismatch_and_checked_id_overflow() {
        let malformed = payload(json!([[true, "a"]]), json!([]));
        assert_eq!(
            read(&[&malformed], Half::Context, &Selection::All)
                .err()
                .unwrap()
                .sentence,
            Sentence::RequestSplitInvalid
        );
        let first = payload(json!([[i64::MAX, "a"]]), json!([]));
        let second = payload(json!([[0, "b"]]), json!([]));
        assert!(read(&[&first, &second], Half::Context, &Selection::All).is_err());
    }
    #[test]
    fn text_ids_stay_text_and_fractional_ids_rebase_exactly() {
        let original = payload(serde_json::json!([["42", "a"]]), serde_json::json!([]));
        let mut value: serde_json::Value = serde_json::from_slice(&original).unwrap();
        value["schema"]["instance_table"]["columns"]["instance_id"]["dtype"] =
            serde_json::json!("string");
        let written = serde_json::to_vec(&value).unwrap();
        let out = read(&[&written, &written], Half::Context, &Selection::All).unwrap();
        let batch = out.instance.unwrap();
        let ids = batch
            .column_by_name("instance_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(ids.iter().collect::<Vec<_>>(), vec![Some("42"), Some("42")]);
        value["schema"]["instance_table"]["columns"]["instance_id"]["dtype"] =
            serde_json::json!("float64");
        value["context"]["instance_table"]["rows"] = serde_json::json!([[-2.5, "a"]]);
        let written = serde_json::to_vec(&value).unwrap();
        let out = read(&[&written, &written], Half::Context, &Selection::All).unwrap();
        let batch = out.instance.unwrap();
        let ids = batch
            .column_by_name("instance_id")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        // Negative first IDs still reserve the old algorithm's instance row count.
        assert_eq!(ids.iter().collect::<Vec<_>>(), vec![Some(-2.5), Some(1.5)]);
    }

    #[test]
    fn unions_columns_but_refuses_conflicting_declarations() {
        let first = payload(serde_json::json!([[1, "a"]]), serde_json::json!([]));
        let mut next: serde_json::Value = serde_json::from_slice(&first).unwrap();
        next["schema"]["instance_table"]["columns"]["extra"] = serde_json::json!({"dtype":"bool"});
        let second = serde_json::to_vec(&next).unwrap();
        let out = read(&[&first, &second], Half::Context, &Selection::All).unwrap();
        let batch = out.instance.unwrap();
        assert_eq!(batch.column_by_name("extra").unwrap().null_count(), 2);
        assert_eq!(
            batch.column_by_name("extra").unwrap().data_type(),
            &arrow_schema::DataType::Boolean
        );
        next["schema"]["instance_table"]["columns"]["value"]["dtype"] = serde_json::json!("int64");
        next["context"]["instance_table"]["rows"] = serde_json::json!([[1, 7]]);
        assert!(
            read(
                &[&first, &serde_json::to_vec(&next).unwrap()],
                Half::Context,
                &Selection::All
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_lost_float_offset_and_ambiguous_names() {
        let p = payload(serde_json::json!([[1, "a"]]), serde_json::json!([]));
        let mut value: serde_json::Value = serde_json::from_slice(&p).unwrap();
        value["schema"]["instance_table"]["columns"]["instance_id"]["dtype"] =
            serde_json::json!("float64");
        value["context"]["instance_table"]["rows"] = serde_json::json!([[1e20, "a"]]);
        let p = serde_json::to_vec(&value).unwrap();
        assert!(read(&[&p, &p], Half::Context, &Selection::All).is_err());
        value["schema"]["related_tables"][""] = value["schema"]["instance_table"].clone();
        assert!(
            read(
                &[&serde_json::to_vec(&value).unwrap()],
                Half::Context,
                &Selection::Names
            )
            .is_err()
        );
    }

    #[test]
    fn timestamp_is_exact_inverse_including_expanded_years() {
        for micros in [0, 1, -1, i64::MIN, i64::MAX, 1_234_567_890_123_456] {
            let written = crate::payload::iso_microsecond_count(i128::from(micros));
            assert_eq!(timestamp(&written).unwrap(), micros);
        }
        assert!(timestamp("2026-02-30T01:02:03.123456Z").is_err());
    }
}
