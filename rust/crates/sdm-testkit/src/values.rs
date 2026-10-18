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
//! kept), `["union", type code, member name, cell]` (never null itself:
//! a null member, or a union DuckDB writes as null, keeps its tag) and
//! `["interval", months, days, nanoseconds as digits]` (a month-day-nano
//! interval); a dictionary is read as its values; text a field marks as
//! JSON (Arrow's `arrow.json`) as the value it holds, as a driver reads
//! JSON ([`json_cell`]); a month-day-nano interval of a column tagged as
//! built from `timedelta`s ([`INTERVAL_TAG`]) as that `timedelta`,
//! `["dur", nanoseconds]`, a month 30 days ([`folded_interval_cell`]). A
//! value of any other
//! type (a `date64`, a year-month or day-time interval) is `["?", its
//! type's name]` and is listed as unencodable, inside a map or a chosen
//! union member too, which the comparator never takes as equal; a member
//! no row chooses names nothing. A row is the compact JSON of its cells,
//! ASCII only (Python's `json.dumps` with `ensure_ascii`), so its order as
//! text is its order as bytes; `ordered` is the SHA-256 of the rows in
//! the order read, `unordered` of the same rows sorted by their text.

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
    DurationSecondType, Field, Int8Type, Int16Type, Int32Type, Int64Type, IntervalMonthDayNanoType,
    IntervalUnit, Time32MillisecondType, Time32SecondType, Time64MicrosecondType,
    Time64NanosecondType, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
    TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    UnionMode,
};
use duckdb::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

/// A result's columns and its rows in canonical form, as the observation
/// format's `values` holds them; `types` is null, since this host has no
/// driver description of DuckDB's own type names.
pub fn values(fields: &[&Field], batches: &[RecordBatch]) -> Value {
    let mut odd = BTreeSet::new();
    let mut texts = Vec::new();
    let json: Vec<bool> = fields.iter().map(|f| is_json(f)).collect();
    let folded: Vec<bool> = fields.iter().map(|f| is_folded_interval(f)).collect();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<Value> = batch
                .columns()
                .iter()
                .enumerate()
                .map(|(at, column)| {
                    if json.get(at).copied().unwrap_or(false) {
                        json_cell(column, row, &mut odd)
                    } else if folded.get(at).copied().unwrap_or(false) {
                        folded_interval_cell(column, row)
                    } else {
                        cell(column, row, &mut odd)
                    }
                })
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

/// The field metadata key, and its value, marking a month-day-nano
/// interval column built from a driver's `timedelta`s (the Postgres
/// provider's, psycopg's reading of an `interval`).
pub const INTERVAL_TAG: (&str, &str) = ("sdm:interval", "timedelta");

/// Whether `field` is a month-day-nano interval built from `timedelta`s
/// ([`INTERVAL_TAG`]).
fn is_folded_interval(field: &Field) -> bool {
    field.metadata().get(INTERVAL_TAG.0).map(String::as_str) == Some(INTERVAL_TAG.1)
        && field.data_type() == &DataType::Interval(IntervalUnit::MonthDayNano)
}

/// One interval of a column built from `timedelta`s, `row` of `array`, in
/// the canonical cell of the `timedelta` itself, `["dur", nanoseconds]`,
/// a month counted as 30 days as psycopg counts it; null where it is null.
fn folded_interval_cell(array: &ArrayRef, row: usize) -> Value {
    if array.is_null(row) {
        return Value::Null;
    }
    let value = array.as_primitive::<IntervalMonthDayNanoType>().value(row);
    let days = i128::from(value.months) * 30 + i128::from(value.days);
    let nanos = days * 86_400_000_000_000 + i128::from(value.nanoseconds);
    tagged("dur", nanos.to_string())
}

/// Whether `field` is text holding JSON: Arrow's canonical `arrow.json`
/// extension over a string.
fn is_json(field: &Field) -> bool {
    field
        .metadata()
        .get("ARROW:extension:name")
        .is_some_and(|name| name == "arrow.json")
        && field.data_type() == &DataType::Utf8
}

/// One JSON text, `row` of `array`, as the value it holds, in the canonical
/// form a driver's reading of the same JSON has (`json.loads`, then
/// `run_python.py`'s `canonical_cell`): an object a `struct` of its members
/// in their order (a repeated name its last value, at its first place), an
/// array a `list`, a whole number `["i", its digits]`, any other number a
/// double, a string `["s", text]`; null and the booleans themselves. A
/// text that is not JSON, or holds a string no text holds (a lone
/// surrogate), is unencodable.
fn json_cell(array: &ArrayRef, row: usize, odd: &mut BTreeSet<String>) -> Value {
    if array.is_null(row) {
        return Value::Null;
    }
    let Some(text) = array.as_string_opt::<i32>().map(|a| a.value(row)) else {
        return unencodable(array.data_type(), odd);
    };
    let mut parser = JsonText {
        chars: text.chars().collect(),
        at: 0,
    };
    match parser.value() {
        Some(value) if parser.done() => value,
        _ => {
            odd.insert("arrow:json".to_owned());
            json!(["?", "arrow:json"])
        }
    }
}

/// A JSON text read into its canonical value ([`json_cell`]).
struct JsonText {
    chars: Vec<char>,
    at: usize,
}

impl JsonText {
    fn space(&mut self) {
        while self
            .chars
            .get(self.at)
            .is_some_and(|c| matches!(c, ' ' | '\t' | '\n' | '\r'))
        {
            self.at += 1;
        }
    }

    fn done(&mut self) -> bool {
        self.space();
        self.at == self.chars.len()
    }

    fn eat(&mut self, word: &str) -> bool {
        let n = word.chars().count();
        let found = self
            .chars
            .get(self.at..self.at + n)
            .is_some_and(|w| w.iter().copied().eq(word.chars()));
        if found {
            self.at += n;
        }
        found
    }

    fn value(&mut self) -> Option<Value> {
        self.space();
        let first = *self.chars.get(self.at)?;
        match first {
            '{' => {
                self.at += 1;
                let mut members: Vec<(String, Value)> = Vec::new();
                self.space();
                if !self.eat("}") {
                    loop {
                        self.space();
                        let name = self.string()?;
                        self.space();
                        if !self.eat(":") {
                            return None;
                        }
                        let value = self.value()?;
                        match members.iter_mut().find(|(n, _)| *n == name) {
                            Some(member) => member.1 = value,
                            None => members.push((name, value)),
                        }
                        self.space();
                        if self.eat("}") {
                            break;
                        }
                        if !self.eat(",") {
                            return None;
                        }
                    }
                }
                let members: Vec<Value> = members
                    .into_iter()
                    .map(|(name, value)| json!([name, value]))
                    .collect();
                Some(json!(["struct", members]))
            }
            '[' => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if !self.eat("]") {
                    loop {
                        items.push(self.value()?);
                        self.space();
                        if self.eat("]") {
                            break;
                        }
                        if !self.eat(",") {
                            return None;
                        }
                    }
                }
                Some(json!(["list", items]))
            }
            '"' => self.string().map(|s| tagged("s", s)),
            _ if self.eat("true") => Some(Value::Bool(true)),
            _ if self.eat("false") => Some(Value::Bool(false)),
            _ if self.eat("null") => Some(Value::Null),
            // Python's reading takes these three too.
            _ if self.eat("NaN") => Some(float(f64::NAN)),
            _ if self.eat("Infinity") => Some(float(f64::INFINITY)),
            _ if self.eat("-Infinity") => Some(float(f64::NEG_INFINITY)),
            _ => self.number(),
        }
    }

    fn digits(&mut self) -> bool {
        let from = self.at;
        while self.chars.get(self.at).is_some_and(char::is_ascii_digit) {
            self.at += 1;
        }
        self.at > from
    }

    fn number(&mut self) -> Option<Value> {
        let start = self.at;
        self.eat("-");
        let whole_from = self.at;
        if !self.digits() {
            return None;
        }
        // JSON writes no leading zero.
        if self.chars[whole_from] == '0' && self.at - whole_from > 1 {
            return None;
        }
        let mut whole = true;
        if self.eat(".") {
            whole = false;
            if !self.digits() {
                return None;
            }
        }
        if self
            .chars
            .get(self.at)
            .is_some_and(|c| *c == 'e' || *c == 'E')
        {
            whole = false;
            self.at += 1;
            if !self.eat("+") {
                self.eat("-");
            }
            if !self.digits() {
                return None;
            }
        }
        let token: String = self.chars[start..self.at].iter().collect();
        if !whole {
            return token.parse::<f64>().ok().map(float);
        }
        let digits = token.trim_start_matches('-');
        if token.starts_with('-') && !digits.chars().all(|c| c == '0') {
            Some(tagged("i", format!("-{digits}")))
        } else {
            Some(tagged("i", digits.to_owned()))
        }
    }

    fn string(&mut self) -> Option<String> {
        if !self.eat("\"") {
            return None;
        }
        let mut out = String::new();
        loop {
            let c = *self.chars.get(self.at)?;
            self.at += 1;
            match c {
                '"' => return Some(out),
                '\\' => {
                    let escaped = *self.chars.get(self.at)?;
                    self.at += 1;
                    match escaped {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let high = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&high) {
                                if !self.eat("\\u") {
                                    return None;
                                }
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return None;
                                }
                                0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                high
                            };
                            out.push(char::from_u32(code)?);
                        }
                        _ => return None,
                    }
                }
                c if u32::from(c) < 0x20 => return None,
                c => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let text: String = self.chars.get(self.at..self.at + 4)?.iter().collect();
        self.at += 4;
        u32::from_str_radix(&text, 16).ok()
    }
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
        DataType::Interval(IntervalUnit::MonthDayNano) => {
            let value = array.as_primitive::<IntervalMonthDayNanoType>().value(row);
            json!([
                "interval",
                value.months,
                value.days,
                value.nanoseconds.to_string()
            ])
        }
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use duckdb::arrow::array::IntervalMonthDayNanoArray;
    use duckdb::arrow::datatypes::{IntervalMonthDayNano, Schema};

    use super::*;

    /// The JSON text and its canonical row that
    /// `tests/test_t21e_json_cell.py` holds the Python runner's reading to.
    const TEXT: &str = r#"{"b": [1], "a": {"n": -0, "big": 123456789012345678901234567890}, "c": [2.5, 1e2, null, true, "\u00e9\ud83d\ude00"], "b": "last"}"#;
    const ROW: &str = r#"[["struct",[["b",["s","last"]],["a",["struct",[["n",["i","0"]],["big",["i","123456789012345678901234567890"]]]]],["c",["list",[["f","4004000000000000"],["f","4059000000000000"],null,true,["s","\u00e9\ud83d\ude00"]]]]]]]"#;

    fn rows(json: bool, texts: &[&str]) -> Value {
        let mut field = Field::new("f", DataType::Utf8, true);
        if json {
            field = field.with_metadata(HashMap::from([(
                "ARROW:extension:name".to_owned(),
                "arrow.json".to_owned(),
            )]));
        }
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let array: ArrayRef = Arc::new(StringArray::from(texts.to_vec()));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        values(&[&field], &[batch])
    }

    fn intervals(tagged: bool, cells: &[IntervalMonthDayNano]) -> Value {
        let mut field = Field::new("f", DataType::Interval(IntervalUnit::MonthDayNano), true);
        if tagged {
            field = field.with_metadata(HashMap::from([(
                INTERVAL_TAG.0.to_owned(),
                INTERVAL_TAG.1.to_owned(),
            )]));
        }
        let schema = Arc::new(Schema::new(vec![field.clone()]));
        let array: ArrayRef = Arc::new(IntervalMonthDayNanoArray::from(cells.to_vec()));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        values(&[&field], &[batch])
    }

    /// The canonical row of `timedelta(days=31, microseconds=5)`, which
    /// `tests/test_t21e_interval_timedelta.py` holds the Python runner's
    /// reading to.
    const DELTA_ROW: &str = r#"[["dur","2678400000005000"]]"#;

    #[test]
    fn a_tagged_interval_reads_as_the_timedelta_it_was_built_from() {
        let digest = crate::case::sha256(format!("[{DELTA_ROW}]").as_bytes());
        let one = IntervalMonthDayNano::new(0, 31, 5_000);
        assert_eq!(intervals(true, &[one])["ordered"], digest);
        // A month counts as 30 days, as psycopg counts it.
        let month = IntervalMonthDayNano::new(1, 1, 5_000);
        assert_eq!(intervals(true, &[month])["ordered"], digest);
        // Untagged, an interval keeps its own cell; another one differs.
        assert_ne!(intervals(false, &[one])["ordered"], digest);
        let other = IntervalMonthDayNano::new(0, 31, 6_000);
        assert_ne!(intervals(true, &[other])["ordered"], digest);
    }

    #[test]
    fn json_text_reads_as_the_value_a_driver_reads() {
        let array: ArrayRef = Arc::new(StringArray::from(vec![TEXT]));
        let mut odd = BTreeSet::new();
        let row = Value::Array(vec![json_cell(&array, 0, &mut odd)]);
        assert_eq!(ascii_json(&row), ROW);
        assert!(odd.is_empty());
        let digest = crate::case::sha256(format!("[{ROW}]").as_bytes());
        assert_eq!(rows(true, &[TEXT])["ordered"], digest);
        // Another JSON value differs; text not marked as JSON stays text;
        // a text that is no JSON is unencodable.
        assert_ne!(rows(true, &[r#"{"a": 2}"#])["ordered"], digest);
        assert_ne!(rows(false, &[TEXT])["ordered"], digest);
        assert_eq!(rows(true, &["{"])["unencodable"], json!(["arrow:json"]));
    }
}
