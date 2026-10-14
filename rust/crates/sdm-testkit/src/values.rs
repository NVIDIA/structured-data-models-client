// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A result's rows in canonical form (decision H9; design, "Observations
//! and comparator"), as `contract/runners/run_python.py`'s
//! `result_values` writes them, so the comparator can hold this host's
//! statement results to the Python runners' and the reference's: each value
//! as itself, never as a language's spelling of it. Both are held to
//! `contract/result-vectors.json` (`tests/result_vectors.rs` here,
//! `contract/tests/test_result_vectors.py` there).
//!
//! A null is `null` and a boolean itself; anything else is a list naming
//! its kind: `["i", digits]`, `["f", the IEEE-754 bits of the value as a
//! double, 16 hex digits]` (a `float32` widened exactly), `["s", text]`,
//! `["b", hex]`, `["d", unscaled digits, scale]`, `["date", days since
//! 1970-01-01]`, `["ts", nanoseconds since 1970-01-01 on the wall clock]`
//! (a timestamp without a zone), `["tstz", nanoseconds since the UTC
//! epoch]`, `["time", nanoseconds since midnight]`, `["dur",
//! nanoseconds]`, each temporal count read from the array's integer storage
//! at its unit, `["list", cells]`, `["struct", [[name, cell], ...]]`,
//! `["map", [[key, value], ...]]` (Arrow's entry order, duplicate keys
//! kept) and `["union", type code, member name, cell]` (never null itself:
//! a null member, or a union DuckDB writes as null, keeps its tag); a
//! dictionary is read as its values. A value of any other type (an
//! interval) is `["?", its type's name]` and is listed as unencodable,
//! inside a map or a chosen union member too, which the comparator never
//! takes as equal; a member no row chooses names nothing. A row is the compact JSON of its cells, ASCII only
//! (Python's `json.dumps` with `ensure_ascii`), so its order as text is its
//! order as bytes; `ordered` is the SHA-256 of the rows in the order read,
//! `unordered` of the same rows sorted by their text.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use duckdb::arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, BinaryViewArray, BooleanArray, Date32Array,
    Decimal128Array, Decimal256Array, FixedSizeBinaryArray, FixedSizeListArray, Float16Array,
    Float32Array, Float64Array, GenericListArray, LargeBinaryArray, LargeStringArray, MapArray,
    StringArray, StringViewArray, StructArray, UnionArray,
};
use duckdb::arrow::compute::cast;
use duckdb::arrow::datatypes::{
    DataType, DurationMicrosecondType, DurationMillisecondType, DurationNanosecondType,
    DurationSecondType, Field, Int8Type, Int16Type, Int32Type, Int64Type, IntervalUnit,
    Time32MillisecondType, Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimeUnit,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type, UnionMode,
};
use duckdb::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

/// A result's columns and its rows in canonical form, as the observation
/// format's `values` holds them; `types` is null, since this host has no
/// driver description of DuckDB's own type names.
pub fn values(fields: &[&Field], batches: &[RecordBatch]) -> Value {
    let mut odd = BTreeSet::new();
    let mut texts = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<Value> = batch
                .columns()
                .iter()
                .map(|column| cell(column, row, &mut odd))
                .collect();
            texts.push(ascii_json(&Value::Array(cells)));
        }
    }
    let ordered = digest_of(&texts);
    texts.sort();
    json!({
        "columns": fields.iter().map(|f| f.name().clone()).collect::<Vec<_>>(),
        "types": null,
        "ordered": ordered,
        "unordered": digest_of(&texts),
        "unencodable": odd.into_iter().collect::<Vec<_>>(),
    })
}

fn digest_of(texts: &[String]) -> String {
    crate::case::sha256(format!("[{}]", texts.join(",")).as_bytes())
}

fn tagged(tag: &str, value: impl Into<Value>) -> Value {
    Value::Array(vec![Value::from(tag), value.into()])
}

fn nanos(value: i64, unit: &TimeUnit) -> String {
    let factor: i128 = match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    };
    (i128::from(value) * factor).to_string()
}

/// A decimal's unscaled digits and scale; a negative scale is folded into
/// the digits, as the Python side reads such a value.
fn decimal(unscaled: String, scale: i8) -> Value {
    if scale >= 0 {
        return json!(["d", unscaled, scale]);
    }
    let zero = unscaled == "0";
    let mut digits = unscaled;
    if !zero {
        digits.push_str(&"0".repeat(usize::from(scale.unsigned_abs())));
    }
    json!(["d", digits, 0])
}

fn hex(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(text, "{byte:02x}");
    }
    text
}

fn float(value: f64) -> Value {
    tagged("f", format!("{:016x}", value.to_bits()))
}

/// One value, `row` of `array`, in canonical form.
fn cell(array: &ArrayRef, row: usize, odd: &mut BTreeSet<String>) -> Value {
    let any = array.as_any();
    // An Arrow union has no validity of its own: each row is its member's
    // tag, the member's cell null where the member is.
    if let DataType::Union(fields, _) = array.data_type() {
        return match any.downcast_ref::<UnionArray>() {
            Some(a) => {
                let code = a.type_id(row);
                match fields.iter().find(|(id, _)| *id == code) {
                    Some((_, field)) => json!([
                        "union",
                        code,
                        field.name(),
                        cell(a.child(code), a.value_offset(row), odd)
                    ]),
                    None => unencodable(array.data_type(), odd),
                }
            }
            None => unencodable(array.data_type(), odd),
        };
    }
    if array.is_null(row) {
        return Value::Null;
    }
    match array.data_type() {
        DataType::Null => Value::Null,
        // Fails closed: a Boolean array that does not read as one is
        // unencodable, never false.
        DataType::Boolean => match any.downcast_ref::<BooleanArray>() {
            Some(a) => Value::Bool(a.value(row)),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Int8 => tagged("i", array.as_primitive::<Int8Type>().value(row).to_string()),
        DataType::Int16 => tagged(
            "i",
            array.as_primitive::<Int16Type>().value(row).to_string(),
        ),
        DataType::Int32 => tagged(
            "i",
            array.as_primitive::<Int32Type>().value(row).to_string(),
        ),
        DataType::Int64 => tagged(
            "i",
            array.as_primitive::<Int64Type>().value(row).to_string(),
        ),
        DataType::UInt8 => tagged(
            "i",
            array.as_primitive::<UInt8Type>().value(row).to_string(),
        ),
        DataType::UInt16 => tagged(
            "i",
            array.as_primitive::<UInt16Type>().value(row).to_string(),
        ),
        DataType::UInt32 => tagged(
            "i",
            array.as_primitive::<UInt32Type>().value(row).to_string(),
        ),
        DataType::UInt64 => tagged(
            "i",
            array.as_primitive::<UInt64Type>().value(row).to_string(),
        ),
        DataType::Float16 => match any.downcast_ref::<Float16Array>() {
            Some(a) => float(f64::from(a.value(row).to_f32())),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Float32 => match any.downcast_ref::<Float32Array>() {
            Some(a) => float(f64::from(a.value(row))),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Float64 => match any.downcast_ref::<Float64Array>() {
            Some(a) => float(a.value(row)),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Utf8 => text(
            any.downcast_ref::<StringArray>().map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::LargeUtf8 => text(
            any.downcast_ref::<LargeStringArray>().map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::Utf8View => text(
            any.downcast_ref::<StringViewArray>().map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::Binary => bytes(
            any.downcast_ref::<BinaryArray>().map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::LargeBinary => bytes(
            any.downcast_ref::<LargeBinaryArray>().map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::BinaryView => bytes(
            any.downcast_ref::<BinaryViewArray>().map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::FixedSizeBinary(_) => bytes(
            any.downcast_ref::<FixedSizeBinaryArray>()
                .map(|a| a.value(row)),
            array,
            odd,
        ),
        DataType::Decimal128(_, scale) => match any.downcast_ref::<Decimal128Array>() {
            Some(a) => decimal(a.value(row).to_string(), *scale),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Decimal256(_, scale) => match any.downcast_ref::<Decimal256Array>() {
            Some(a) => decimal(a.value(row).to_string(), *scale),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Date32 => match any.downcast_ref::<Date32Array>() {
            Some(a) => tagged("date", a.value(row)),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Timestamp(unit, zone) => {
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
            tagged(
                if zone.is_some() { "tstz" } else { "ts" },
                nanos(value, unit),
            )
        }
        DataType::Time32(unit) => {
            let value = match unit {
                TimeUnit::Second => array.as_primitive::<Time32SecondType>().value(row),
                TimeUnit::Millisecond => array.as_primitive::<Time32MillisecondType>().value(row),
                _ => return unencodable(array.data_type(), odd),
            };
            tagged("time", nanos(i64::from(value), unit))
        }
        DataType::Time64(unit) => {
            let value = match unit {
                TimeUnit::Microsecond => array.as_primitive::<Time64MicrosecondType>().value(row),
                TimeUnit::Nanosecond => array.as_primitive::<Time64NanosecondType>().value(row),
                _ => return unencodable(array.data_type(), odd),
            };
            tagged("time", nanos(value, unit))
        }
        DataType::Duration(unit) => {
            let value = match unit {
                TimeUnit::Second => array.as_primitive::<DurationSecondType>().value(row),
                TimeUnit::Millisecond => array.as_primitive::<DurationMillisecondType>().value(row),
                TimeUnit::Microsecond => array.as_primitive::<DurationMicrosecondType>().value(row),
                TimeUnit::Nanosecond => array.as_primitive::<DurationNanosecondType>().value(row),
            };
            tagged("dur", nanos(value, unit))
        }
        DataType::List(_) => match any.downcast_ref::<GenericListArray<i32>>() {
            Some(a) => list(&a.value(row), odd),
            None => unencodable(array.data_type(), odd),
        },
        DataType::LargeList(_) => match any.downcast_ref::<GenericListArray<i64>>() {
            Some(a) => list(&a.value(row), odd),
            None => unencodable(array.data_type(), odd),
        },
        DataType::FixedSizeList(_, _) => match any.downcast_ref::<FixedSizeListArray>() {
            Some(a) => list(&a.value(row), odd),
            None => unencodable(array.data_type(), odd),
        },
        DataType::Struct(fields) => match any.downcast_ref::<StructArray>() {
            Some(a) => {
                let members: Vec<Value> = fields
                    .iter()
                    .zip(a.columns())
                    .map(|(field, column)| json!([field.name(), cell(column, row, odd)]))
                    .collect();
                json!(["struct", members])
            }
            None => unencodable(array.data_type(), odd),
        },
        DataType::Map(_, _) => match any.downcast_ref::<MapArray>() {
            Some(a) => {
                let entries = a.value(row);
                let (keys, items) = (entries.column(0), entries.column(1));
                let pairs: Vec<Value> = (0..entries.len())
                    .map(|i| json!([cell(keys, i, odd), cell(items, i, odd)]))
                    .collect();
                json!(["map", pairs])
            }
            None => unencodable(array.data_type(), odd),
        },
        DataType::Dictionary(_, values) => match cast(&array.slice(row, 1), values) {
            Ok(decoded) => cell(&decoded, 0, odd),
            Err(_) => unencodable(array.data_type(), odd),
        },
        other => unencodable(other, odd),
    }
}

fn text(value: Option<&str>, array: &ArrayRef, odd: &mut BTreeSet<String>) -> Value {
    match value {
        Some(value) => tagged("s", value),
        None => unencodable(array.data_type(), odd),
    }
}

fn bytes(value: Option<&[u8]>, array: &ArrayRef, odd: &mut BTreeSet<String>) -> Value {
    match value {
        Some(value) => tagged("b", hex(value)),
        None => unencodable(array.data_type(), odd),
    }
}

fn list(items: &ArrayRef, odd: &mut BTreeSet<String>) -> Value {
    let cells: Vec<Value> = (0..items.len()).map(|i| cell(items, i, odd)).collect();
    json!(["list", cells])
}

fn unencodable(data_type: &DataType, odd: &mut BTreeSet<String>) -> Value {
    let name = format!("arrow:{}", arrow_type(data_type));
    odd.insert(name.clone());
    json!(["?", name])
}

/// An Arrow type as pyarrow spells it (`str(pyarrow.DataType)`), so the
/// two core runners' result schemas compare as text; a type pyarrow would
/// spell otherwise gets its arrow-rs name, which then differs, never
/// matches by accident.
pub fn arrow_type(data_type: &DataType) -> String {
    let unit = |unit: &TimeUnit| match unit {
        TimeUnit::Second => "s",
        TimeUnit::Millisecond => "ms",
        TimeUnit::Microsecond => "us",
        TimeUnit::Nanosecond => "ns",
    };
    let child = |field: &Field| {
        let mut text = format!("{}: {}", field.name(), arrow_type(field.data_type()));
        if !field.is_nullable() {
            text.push_str(" not null");
        }
        text
    };
    match data_type {
        DataType::Null => "null".into(),
        DataType::Boolean => "bool".into(),
        DataType::Int8 => "int8".into(),
        DataType::Int16 => "int16".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::UInt8 => "uint8".into(),
        DataType::UInt16 => "uint16".into(),
        DataType::UInt32 => "uint32".into(),
        DataType::UInt64 => "uint64".into(),
        DataType::Float16 => "halffloat".into(),
        DataType::Float32 => "float".into(),
        DataType::Float64 => "double".into(),
        DataType::Utf8 => "string".into(),
        DataType::LargeUtf8 => "large_string".into(),
        DataType::Utf8View => "string_view".into(),
        DataType::Binary => "binary".into(),
        DataType::LargeBinary => "large_binary".into(),
        DataType::BinaryView => "binary_view".into(),
        DataType::FixedSizeBinary(size) => format!("fixed_size_binary[{size}]"),
        DataType::Date32 => "date32[day]".into(),
        DataType::Date64 => "date64[ms]".into(),
        DataType::Timestamp(u, None) => format!("timestamp[{}]", unit(u)),
        DataType::Timestamp(u, Some(zone)) => format!("timestamp[{}, tz={zone}]", unit(u)),
        DataType::Time32(u) => format!("time32[{}]", unit(u)),
        DataType::Time64(u) => format!("time64[{}]", unit(u)),
        DataType::Duration(u) => format!("duration[{}]", unit(u)),
        DataType::Decimal128(p, s) => format!("decimal128({p}, {s})"),
        DataType::Decimal256(p, s) => format!("decimal256({p}, {s})"),
        DataType::Interval(IntervalUnit::MonthDayNano) => "month_day_nano_interval".into(),
        DataType::Interval(IntervalUnit::YearMonth) => "month_interval".into(),
        DataType::Interval(IntervalUnit::DayTime) => "day_time_interval".into(),
        DataType::List(field) => format!("list<{}>", child(field)),
        DataType::LargeList(field) => format!("large_list<{}>", child(field)),
        DataType::FixedSizeList(field, size) => {
            format!("fixed_size_list<{}>[{size}]", child(field))
        }
        DataType::Struct(fields) => format!(
            "struct<{}>",
            fields
                .iter()
                .map(|f| child(f))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        // As pyarrow spells them; arrow-rs keeps a dictionary's order on
        // its field, not its type, and DuckDB writes an unordered one.
        DataType::Dictionary(indices, values) => format!(
            "dictionary<values={}, indices={}, ordered=0>",
            arrow_type(values),
            arrow_type(indices)
        ),
        DataType::Map(entries, sorted) => match entries.data_type() {
            DataType::Struct(fields) if fields.len() == 2 => format!(
                "map<{}, {}{}>",
                arrow_type(fields[0].data_type()),
                arrow_type(fields[1].data_type()),
                if *sorted { ", keys_sorted" } else { "" }
            ),
            other => format!("{other:?}"),
        },
        DataType::Union(fields, mode) => format!(
            "{}_union<{}>",
            match mode {
                UnionMode::Sparse => "sparse",
                UnionMode::Dense => "dense",
            },
            fields
                .iter()
                .map(|(id, field)| format!("{}={id}", child(field)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => format!("{other:?}"),
    }
}

/// Compact JSON, ASCII only, as Python's `json.dumps(value,
/// ensure_ascii=True, separators=(",", ":"))` writes it: a character
/// outside printable ASCII (space to `~`) is a `\uXXXX` escape in
/// lowercase hex (a surrogate pair past the basic plane, DEL included),
/// but for the short escapes JSON has for some control characters.
pub fn ascii_json(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => write_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(members) => {
            out.push('{');
            for (i, (key, item)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_json(item, out);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            c => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}
