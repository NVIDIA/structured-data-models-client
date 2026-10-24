// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Postgres's value rules (T2.14d, MID2 P2-3): how the reference reads the
//! values its Postgres driver hands over, and how it binds a key set, here
//! in the dialect, so the host's provider only moves the driver's rows into
//! Arrow and binds the text it is given.
//!
//! - A `json` or `jsonb` value, which the provider hands over as JSON text
//!   (Arrow's `arrow.json`), is read as the reference's
//!   `_sanitize_postgres_frame` writes it: a JSON null a null, an object or
//!   an array as compact JSON, ASCII only (its strings' other characters
//!   escaped, as `\u` and four hexadecimal digits), and any other value as
//!   the reference's language writes it (`str`): `true` as `True`, a
//!   number as its digits or the text of its double (`float_text`), a
//!   string as itself. A JSON column's elements in an array alike.
//! - A `numeric` declared past the 76 digits an Arrow decimal holds, which
//!   the provider hands over as each value's exact text, its declared scale
//!   marked ([`DECLARED_SCALE_MARKER`]), is read by its declaration, as the
//!   reference's `_to_dtype` reads it: of a scale above zero as doubles,
//!   of a scale of zero as integers, a column holding a whole number past
//!   `int64` as its digits (where the reference overflows), a NaN missing.
//! - A key set is written as the JSON text its statement reads
//!   ([`key_rows`], T3.28): each cell as its text, a time without a zone as
//!   pandas' `Timestamp.isoformat` writes it (the reference compares a
//!   text time column with it as text), binary as `\x` and its
//!   hexadecimal digits, so the provider binds the text it is handed.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType,
};
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, ListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef, TimeUnit};
use sdm_contract::data::DECLARED_SCALE_MARKER;
use sdm_contract::{CoreError, Detail, Sentence};

use crate::json;

/// The field metadata naming Arrow's canonical JSON extension type.
const EXTENSION_NAME: &str = "ARROW:extension:name";
const JSON: &str = "arrow.json";
/// How deep a JSON value is read; the reference's driver reads none
/// deeper (its reader's recursion limit is about 1,000).
const MOST_DEPTH: usize = 2_000;

type Conformed = Option<(SchemaRef, Vec<RecordBatch>)>;

/// A value of a column the rule cannot read, refused by name: the
/// provider's answer is not what it says it is.
fn unreadable(sentence: Sentence, column: &str, what: &str) -> CoreError {
    CoreError::new(
        sentence,
        format!("column {column:?} holds a value that is not {what}"),
    )
    .with_field("column", Detail::text(column))
}

fn internal(message: &str) -> CoreError {
    CoreError::new(Sentence::Internal, message)
}

fn is_json(field: &Field) -> bool {
    field.metadata().get(EXTENSION_NAME).map(String::as_str) == Some(JSON)
}

/// How a column is read: as it is, as JSON text (or a list of it), or by
/// a `numeric`'s declared scale.
enum Rule {
    Kept,
    Json,
    JsonList(Arc<Field>),
    Declared(u32),
}

fn rule(field: &Field) -> Result<Rule, CoreError> {
    match field.data_type() {
        DataType::Utf8 if is_json(field) => Ok(Rule::Json),
        DataType::List(item) if item.data_type() == &DataType::Utf8 && is_json(item) => {
            Ok(Rule::JsonList(Arc::new(Field::new(
                item.name(),
                DataType::Utf8,
                item.is_nullable(),
            ))))
        }
        DataType::Utf8 => match field.metadata().get(DECLARED_SCALE_MARKER) {
            Some(scale) => scale.parse::<u32>().map(Rule::Declared).map_err(|_| {
                unreadable(Sentence::ValueNotAtScale, field.name(), "a declared scale")
            }),
            None => Ok(Rule::Kept),
        },
        _ => Ok(Rule::Kept),
    }
}

/// Postgres's answer to a fetch, admitted, with its values read as the
/// reference reads them; `None` where no column holds a value a rule
/// reads.
pub fn conform(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    float_text: fn(f64) -> String,
) -> Result<Conformed, CoreError> {
    let rules: Vec<Rule> = schema
        .fields()
        .iter()
        .map(|f| rule(f))
        .collect::<Result<_, _>>()?;
    if rules.iter().all(|r| matches!(r, Rule::Kept)) {
        return Ok(None);
    }
    // A declared integer column's type is its values' over every batch:
    // integers where each fits, else the digits.
    let mut fields = Vec::with_capacity(rules.len());
    for (field, rule) in schema.fields().iter().zip(&rules) {
        let kind = match rule {
            Rule::Kept | Rule::Json => field.data_type().clone(),
            Rule::JsonList(item) => DataType::List(Arc::clone(item)),
            Rule::Declared(0) => {
                let index = schema
                    .index_of(field.name())
                    .map_err(|_| internal("a field"))?;
                let fits = batches.iter().all(|b| {
                    b.column(index)
                        .as_string::<i32>()
                        .iter()
                        .flatten()
                        .all(|text| {
                            is_nan(text) || whole(text).is_some_and(|d| d.parse::<i64>().is_ok())
                        })
                });
                if fits {
                    DataType::Int64
                } else {
                    DataType::Utf8
                }
            }
            Rule::Declared(_) => DataType::Float64,
        };
        let metadata = match rule {
            // The JSON mark stays on a JSON column, as the reference's text
            // of a JSON value is still that value's.
            Rule::Kept | Rule::Json => field.metadata().clone(),
            Rule::JsonList(_) | Rule::Declared(_) => field
                .metadata()
                .clone()
                .into_iter()
                .filter(|(k, _)| k != DECLARED_SCALE_MARKER)
                .collect(),
        };
        fields.push(Field::new(field.name(), kind, field.is_nullable()).with_metadata(metadata));
    }
    let conformed: SchemaRef =
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let mut columns = Vec::with_capacity(rules.len());
        for ((array, rule), field) in batch.columns().iter().zip(&rules).zip(conformed.fields()) {
            let name = field.name();
            columns.push(match rule {
                Rule::Kept => Arc::clone(array),
                Rule::Json => json_column(name, array.as_string::<i32>(), float_text, false)?,
                Rule::JsonList(item) => {
                    let list = array.as_list::<i32>();
                    let values =
                        json_column(name, list.values().as_string::<i32>(), float_text, true)?;
                    Arc::new(
                        ListArray::try_new(
                            Arc::clone(item),
                            list.offsets().clone(),
                            values,
                            list.nulls().cloned(),
                        )
                        .map_err(|_| internal("a JSON list does not rebuild"))?,
                    ) as ArrayRef
                }
                Rule::Declared(scale) => {
                    declared(name, array.as_string::<i32>(), *scale, field.data_type())?
                }
            });
        }
        out.push(
            RecordBatch::try_new(Arc::clone(&conformed), columns)
                .map_err(|_| internal("a conformed batch does not build"))?,
        );
    }
    Ok(Some((conformed, out)))
}

/// A JSON column's cells, each the reference's text of it. A cell that is
/// not JSON is the provider's fault; one whose whole value is a string
/// holding a lone surrogate, which no UTF-8 text holds, is refused by name
/// (`VALUE_OUT_OF_RANGE`, rule `text_not_utf8`), where the reference fails
/// unnamed (`UnicodeEncodeError`); in a list, a column the core leaves
/// out (register P2D-5), such an element is missing (`in_list`).
fn json_column(
    column: &str,
    texts: &StringArray,
    float_text: fn(f64) -> String,
    in_list: bool,
) -> Result<ArrayRef, CoreError> {
    let read: Result<StringArray, CoreError> = texts
        .iter()
        .map(|text| match text {
            None => Ok(None),
            Some(text) => match read_reference_text(text, float_text) {
                Ok(read) => Ok(read),
                Err(Unread::NotUtf8) if in_list => Ok(None),
                Err(Unread::NotUtf8) => Err(CoreError::new(
                    Sentence::TextNotUtf8,
                    "a JSON string holds a lone surrogate, which no UTF-8 text holds",
                )
                .with_field("field", Detail::text("column"))
                .with_field("column", Detail::text(column))),
                Err(Unread::NotJson) => Err(unreadable(Sentence::ValueNotJson, column, "JSON")),
            },
        })
        .map(|r| r.map(|v: Option<String>| v))
        .collect::<Result<Vec<_>, _>>()
        .map(StringArray::from);
    Ok(Arc::new(read?))
}

fn is_nan(text: &str) -> bool {
    text.trim_start_matches(['-', '+'])
        .eq_ignore_ascii_case("nan")
}

/// The digits of a whole number written as a decimal's text (`12`,
/// `-12.000`, `1.2E+3`), or `None` where it is not one.
fn whole(text: &str) -> Option<String> {
    let (negative, unsigned) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (mantissa, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i64>().ok()?),
        None => (unsigned, 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if int.is_empty() && frac.is_empty()
        || !int.chars().chain(frac.chars()).all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let mut digits = format!("{int}{frac}");
    let point = i64::try_from(int.len()).ok()?.checked_add(exponent)?;
    let len = i64::try_from(digits.len()).ok()?;
    if point >= len {
        digits.push_str(&"0".repeat(usize::try_from(point - len).ok()?));
    } else {
        let at = usize::try_from(point.max(0)).ok()?;
        if digits[at..].chars().any(|c| c != '0') {
            return None;
        }
        digits.truncate(at);
    }
    let trimmed = digits.trim_start_matches('0');
    let trimmed = if trimmed.is_empty() { "0" } else { trimmed };
    Some(if negative && trimmed != "0" {
        format!("-{trimmed}")
    } else {
        trimmed.to_owned()
    })
}

/// A declared `numeric`'s exact texts read by its declaration: `to`, the
/// type the column was found to take.
fn declared(
    column: &str,
    texts: &StringArray,
    scale: u32,
    to: &DataType,
) -> Result<ArrayRef, CoreError> {
    let number = |_: &str| unreadable(Sentence::ValueNotANumber, column, "a number");
    Ok(match (scale, to) {
        (0, DataType::Int64) => Arc::new(
            texts
                .iter()
                .map(|t| match t {
                    None => Ok(None),
                    Some(t) if is_nan(t) => Ok(None),
                    Some(t) => whole(t)
                        .and_then(|d| d.parse::<i64>().ok())
                        .map(Some)
                        .ok_or_else(|| number(t)),
                })
                .collect::<Result<Int64Array, _>>()?,
        ),
        (0, _) => Arc::new(
            texts
                .iter()
                .map(|t| match t {
                    None => Ok(None),
                    Some(t) if is_nan(t) => Ok(None),
                    Some(t) => whole(t).map(Some).ok_or_else(|| number(t)),
                })
                .collect::<Result<StringArray, _>>()?,
        ),
        _ => Arc::new(
            texts
                .iter()
                .map(|t| {
                    t.map(|t| t.parse::<f64>().map_err(|_| number(t)))
                        .transpose()
                })
                .collect::<Result<Float64Array, _>>()?,
        ),
    })
}

// --- JSON ----------------------------------------------------------------------

/// A JSON value as the reference's driver reads it: an object's members in
/// the order first named, a name given twice holding its last value, as a
/// mapping of the reference's language keeps them; an integer as its
/// digits, exactly; any other number as a double.
enum Json {
    Null,
    Bool(bool),
    Int(String),
    Float(f64),
    /// Text as UTF-16 code units, as Python holds a JSON string, a lone
    /// surrogate an escape names among them (T2.14b fix 1).
    Text(Vec<u16>),
    List(Vec<Json>),
    Object(Vec<(Vec<u16>, Json)>),
}

/// The reference's text of the JSON value `text` holds, a JSON null none;
/// `None` (outer) where `text` is not JSON.
#[cfg(test)]
fn reference_text(text: &str, float_text: fn(f64) -> String) -> Option<Option<String>> {
    read_reference_text(text, float_text).ok()
}

/// Why a JSON cell is not read.
#[derive(Debug, PartialEq, Eq)]
enum Unread {
    /// It is not JSON.
    NotJson,
    /// It is a JSON string holding a lone surrogate, which Python's `str`
    /// holds and no UTF-8 text does.
    NotUtf8,
}

/// [`reference_text`], saying why a cell is not read.
fn read_reference_text(
    text: &str,
    float_text: fn(f64) -> String,
) -> Result<Option<String>, Unread> {
    let mut reader = Reader {
        text,
        bytes: text.as_bytes(),
        at: 0,
    };
    let value = reader.value(0).ok_or(Unread::NotJson)?;
    reader.space();
    if reader.at != reader.bytes.len() {
        return Err(Unread::NotJson);
    }
    Ok(match value {
        Json::Null => None,
        Json::Bool(true) => Some("True".to_owned()),
        Json::Bool(false) => Some("False".to_owned()),
        Json::Int(digits) => Some(digits),
        Json::Float(f) => Some(float_text(f)),
        // Text as Python's `str` of it; one holding a lone surrogate has no
        // UTF-8 form, and is not read.
        Json::Text(units) => Some(String::from_utf16(&units).map_err(|_| Unread::NotUtf8)?),
        nested => {
            let mut out = String::new();
            write(&nested, float_text, &mut out);
            Some(out)
        }
    })
}

/// A JSON value as the reference's Postgres provider hands it over: the
/// JSON text of the value its driver read (`json.dumps` of it: compact,
/// ASCII only, a float as `float_text` writes it), a JSON null none;
/// `None` (outer) where `text` is not JSON. For a host whose driver gives
/// the server's text of a `json` or `jsonb` value (T3.12 fix 1), so how
/// the value is read stays here.
pub fn json_text(text: &str, float_text: fn(f64) -> String) -> Option<Option<String>> {
    let mut reader = Reader {
        text,
        bytes: text.as_bytes(),
        at: 0,
    };
    let value = reader.value(0)?;
    reader.space();
    if reader.at != reader.bytes.len() {
        return None;
    }
    Some(match value {
        Json::Null => None,
        value => {
            let mut out = String::new();
            write(&value, float_text, &mut out);
            Some(out)
        }
    })
}

/// Compact JSON, ASCII only, as the reference writes a nested value.
fn write(value: &Json, float_text: fn(f64) -> String, out: &mut String) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(digits) => out.push_str(digits),
        Json::Float(f) if f.is_nan() => out.push_str("NaN"),
        Json::Float(f) if f.is_infinite() => {
            out.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" });
        }
        Json::Float(f) => out.push_str(&float_text(*f)),
        Json::Text(s) => json::quoted(s, out),
        Json::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write(item, float_text, out);
            }
            out.push(']');
        }
        Json::Object(members) => {
            out.push('{');
            for (i, (name, item)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json::quoted(name, out);
                out.push(':');
                write(item, float_text, out);
            }
            out.push('}');
        }
    }
}

struct Reader<'a> {
    text: &'a str,
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn space(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
        {
            self.at += 1;
        }
    }

    fn eat(&mut self, word: &str) -> bool {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > MOST_DEPTH {
            return None;
        }
        self.space();
        let value = match *self.bytes.get(self.at)? {
            b'n' if self.eat("null") => Json::Null,
            b't' if self.eat("true") => Json::Bool(true),
            b'f' if self.eat("false") => Json::Bool(false),
            b'N' if self.eat("NaN") => Json::Float(f64::NAN),
            b'I' if self.eat("Infinity") => Json::Float(f64::INFINITY),
            b'-' if self.eat("-Infinity") => Json::Float(f64::NEG_INFINITY),
            b'"' => Json::Text(self.text()?),
            b'[' => {
                self.at += 1;
                let mut items = Vec::new();
                self.space();
                if self.eat("]") {
                    return Some(Json::List(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.space();
                    if self.eat("]") {
                        break;
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
                Json::List(items)
            }
            b'{' => {
                self.at += 1;
                let mut members: Vec<(Vec<u16>, Json)> = Vec::new();
                let mut named: BTreeMap<Vec<u16>, usize> = BTreeMap::new();
                self.space();
                if self.eat("}") {
                    return Some(Json::Object(members));
                }
                loop {
                    self.space();
                    if self.bytes.get(self.at) != Some(&b'"') {
                        return None;
                    }
                    let name = self.text()?;
                    self.space();
                    if !self.eat(":") {
                        return None;
                    }
                    let item = self.value(depth + 1)?;
                    match named.get(&name) {
                        Some(&at) => members[at].1 = item,
                        None => {
                            named.insert(name.clone(), members.len());
                            members.push((name, item));
                        }
                    }
                    self.space();
                    if self.eat("}") {
                        break;
                    }
                    if !self.eat(",") {
                        return None;
                    }
                }
                Json::Object(members)
            }
            _ => self.number()?,
        };
        Some(value)
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.at;
        if self.bytes.get(self.at) == Some(&b'-') {
            self.at += 1;
        }
        let digits = |r: &mut Self| {
            let from = r.at;
            while r.bytes.get(r.at).is_some_and(u8::is_ascii_digit) {
                r.at += 1;
            }
            r.at - from
        };
        let int = digits(self);
        if int == 0 || (int > 1 && self.bytes[self.at - int] == b'0') {
            return None;
        }
        let mut float = false;
        if self.bytes.get(self.at) == Some(&b'.') {
            self.at += 1;
            float = true;
            if digits(self) == 0 {
                return None;
            }
        }
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            self.at += 1;
            float = true;
            if matches!(self.bytes.get(self.at), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if digits(self) == 0 {
                return None;
            }
        }
        let token = std::str::from_utf8(&self.bytes[start..self.at]).ok()?;
        Some(if float {
            Json::Float(token.parse().ok()?)
        } else if token == "-0" {
            Json::Int("0".to_owned())
        } else {
            Json::Int(token.to_owned())
        })
    }

    /// A JSON string as Python's reader holds it: its UTF-16 code units,
    /// each `\u` escape one unit, so a lone surrogate is kept as Python
    /// keeps it.
    fn text(&mut self) -> Option<Vec<u16>> {
        self.at += 1;
        let mut out: Vec<u16> = Vec::new();
        let push = |out: &mut Vec<u16>, c: char| {
            let mut units = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut units));
        };
        loop {
            let c = self.text.get(self.at..)?.chars().next()?;
            self.at += c.len_utf8();
            match c {
                '"' => return Some(out),
                '\\' => {
                    let escape = *self.bytes.get(self.at)?;
                    self.at += 1;
                    match escape {
                        b'"' => push(&mut out, '"'),
                        b'\\' => push(&mut out, '\\'),
                        b'/' => push(&mut out, '/'),
                        b'b' => push(&mut out, '\u{8}'),
                        b'f' => push(&mut out, '\u{c}'),
                        b'n' => push(&mut out, '\n'),
                        b'r' => push(&mut out, '\r'),
                        b't' => push(&mut out, '\t'),
                        b'u' => out.push(u16::try_from(self.hex4()?).ok()?),
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None,
                c => push(&mut out, c),
            }
        }
    }

    fn hex4(&mut self) -> Option<u32> {
        let digits = self.bytes.get(self.at..self.at + 4)?;
        let value = u32::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()?;
        self.at += 4;
        Some(value)
    }
}

// --- key sets --------------------------------------------------------------------

/// Each seed of a key set as Postgres's statement reads it (T3.28): a JSON
/// array of its cells' texts in the key set's column order, the batch
/// among them, a null `null`, each text a JSON string with its characters
/// as they are, as the core route has bound one since T2.3.
pub(crate) fn key_rows(
    request: &sdm_contract::data::DataRequest,
    schema: &SchemaRef,
    batches: &[RecordBatch],
    float_text: fn(f64) -> String,
) -> Result<Vec<String>, CoreError> {
    let _ = (request, float_text);
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(array, field)| {
                    Ok(match key_cell(array.as_ref(), row, field.name())? {
                        Some(text) => json::string(&text, false),
                        None => "null".to_owned(),
                    })
                })
                .collect::<Result<Vec<_>, CoreError>>()?;
            rows.push(format!("[{}]", cells.join(",")));
        }
    }
    Ok(rows)
}

/// One cell of a key set as the text Postgres's input function reads: a
/// time without a zone as pandas' `Timestamp.isoformat` writes it (the
/// reference compares a text time column with it as text) and binary as
/// `\x` and its hexadecimal digits, as the reference spells them; any
/// other value as its own text (an integer's digits, a double's shortest
/// digits, a decimal's with its scale's, a date as `YYYY-MM-DD`, a time of
/// day and a zoned time to its unit's digits, the zoned one in UTC with a
/// `Z`), which the type the statement reads it as reads as that value. A
/// cell of another type is refused by name.
fn key_cell(array: &dyn Array, row: usize, column: &str) -> Result<Option<String>, CoreError> {
    use arrow_array::types as t;
    if array.is_null(row) {
        return Ok(None);
    }
    let text = match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().value(row).to_owned(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(row).to_owned(),
        DataType::Utf8View => array.as_string_view().value(row).to_owned(),
        DataType::Boolean => array.as_boolean().value(row).to_string(),
        DataType::Int8 => array.as_primitive::<t::Int8Type>().value(row).to_string(),
        DataType::Int16 => array.as_primitive::<t::Int16Type>().value(row).to_string(),
        DataType::Int32 => array.as_primitive::<t::Int32Type>().value(row).to_string(),
        DataType::Int64 => array.as_primitive::<t::Int64Type>().value(row).to_string(),
        DataType::UInt8 => array.as_primitive::<t::UInt8Type>().value(row).to_string(),
        DataType::UInt16 => array.as_primitive::<t::UInt16Type>().value(row).to_string(),
        DataType::UInt32 => array.as_primitive::<t::UInt32Type>().value(row).to_string(),
        DataType::UInt64 => array.as_primitive::<t::UInt64Type>().value(row).to_string(),
        DataType::Float32 => {
            let x = array.as_primitive::<t::Float32Type>().value(row);
            if x.is_finite() {
                format!("{x:?}")
            } else {
                json::number(f64::from(x))
            }
        }
        DataType::Float64 => json::number(array.as_primitive::<t::Float64Type>().value(row)),
        DataType::Decimal128(_, scale) => scaled(
            &array
                .as_primitive::<t::Decimal128Type>()
                .value(row)
                .to_string(),
            *scale,
        ),
        DataType::Decimal256(_, scale) => scaled(
            &array
                .as_primitive::<t::Decimal256Type>()
                .value(row)
                .to_string(),
            *scale,
        ),
        DataType::Date32 => date(i64::from(array.as_primitive::<t::Date32Type>().value(row))),
        DataType::Date64 => date(
            array
                .as_primitive::<t::Date64Type>()
                .value(row)
                .div_euclid(86_400_000),
        ),
        DataType::Binary => hex(array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => hex(array.as_binary::<i64>().value(row)),
        DataType::Timestamp(unit, zone) => {
            let (ticks, digits) = match unit {
                TimeUnit::Second => (
                    i128::from(array.as_primitive::<TimestampSecondType>().value(row)),
                    0,
                ),
                TimeUnit::Millisecond => (
                    i128::from(array.as_primitive::<TimestampMillisecondType>().value(row)),
                    3,
                ),
                TimeUnit::Microsecond => (
                    i128::from(array.as_primitive::<TimestampMicrosecondType>().value(row)),
                    6,
                ),
                TimeUnit::Nanosecond => (
                    i128::from(array.as_primitive::<TimestampNanosecondType>().value(row)),
                    9,
                ),
            };
            let nanos = ticks * 10_i128.pow(9 - digits);
            if zone.is_none() {
                isoformat(nanos)
            } else {
                zoned(nanos, digits)
            }
        }
        DataType::Time32(_) | DataType::Time64(_) => {
            let (ticks, digits) = match array.data_type() {
                DataType::Time32(TimeUnit::Second) => (
                    i64::from(array.as_primitive::<t::Time32SecondType>().value(row)),
                    0,
                ),
                DataType::Time32(_) => (
                    i64::from(array.as_primitive::<t::Time32MillisecondType>().value(row)),
                    3,
                ),
                DataType::Time64(TimeUnit::Microsecond) => (
                    array.as_primitive::<t::Time64MicrosecondType>().value(row),
                    6,
                ),
                _ => (
                    array.as_primitive::<t::Time64NanosecondType>().value(row),
                    9,
                ),
            };
            let per = 10_i64.pow(digits);
            let seconds = ticks.div_euclid(per);
            let mut text = format!(
                "{:02}:{:02}:{:02}",
                seconds / 3_600,
                seconds / 60 % 60,
                seconds % 60
            );
            if digits > 0 {
                text.push_str(&format!(
                    ".{:0width$}",
                    ticks.rem_euclid(per),
                    width = digits as usize
                ));
            }
            text
        }
        _ => {
            return Err(CoreError::new(
                Sentence::KeySetTypeUnbound,
                format!(
                    "key set column {column:?} is of a type this dialect does not bind as text"
                ),
            )
            .with_field("column", Detail::text(column)));
        }
    };
    Ok(Some(text))
}

/// A decimal's text, `digits` (signed) ten-to-the-`scale`ths, with its
/// scale's digits after the point.
fn scaled(digits: &str, scale: i8) -> String {
    let (sign, digits) = match digits.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", digits),
    };
    if scale <= 0 {
        let zeros = if digits == "0" {
            String::new()
        } else {
            "0".repeat(usize::from(scale.unsigned_abs()))
        };
        return format!("{sign}{digits}{zeros}");
    }
    let scale = usize::from(scale.unsigned_abs());
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{fraction}")
}

/// Days since 1970-01-01 as `YYYY-MM-DD`.
fn date(days: i64) -> String {
    let (y, m, d) = civil(days);
    format!("{}-{m:02}-{d:02}", year(y))
}

/// An instant, `nanos` since the epoch, in UTC to `digits` fractional
/// digits, a `Z` after it.
fn zoned(nanos: i128, digits: u32) -> String {
    let seconds = nanos.div_euclid(1_000_000_000);
    #[allow(clippy::cast_possible_truncation, reason = "below a second")]
    let fraction = nanos.rem_euclid(1_000_000_000) as i64;
    #[allow(clippy::cast_possible_truncation, reason = "days of an int64 count")]
    let (days, of_day) = (
        seconds.div_euclid(86_400) as i64,
        seconds.rem_euclid(86_400) as i64,
    );
    let mut text = format!(
        "{} {:02}:{:02}:{:02}",
        date(days),
        of_day / 3_600,
        of_day / 60 % 60,
        of_day % 60
    );
    if digits > 0 {
        let kept = fraction / 10_i64.pow(9 - digits);
        text.push_str(&format!(".{kept:0width$}", width = digits as usize));
    }
    text.push('Z');
    text
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(2 + bytes.len() * 2);
    out.push_str("\\x");
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// A year as the reference writes one, at least four digits.
fn year(y: i64) -> String {
    if y < 0 {
        format!("-{:04}", -y)
    } else {
        format!("{y:04}")
    }
}

/// Days since 1970-01-01 as a proleptic Gregorian date.
fn civil(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let of_era = shifted.rem_euclid(146_097);
    let year_of_era = (of_era - of_era / 1_460 + of_era / 36_524 - of_era / 146_096) / 365;
    let of_year = of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * of_year + 2) / 153;
    let day = of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a month and a day"
    )]
    (year, month as u32, day as u32)
}

fn with_fraction(text: String, fraction: i64) -> String {
    if fraction % 1_000 != 0 {
        format!("{text}.{fraction:09}")
    } else if fraction != 0 {
        format!("{text}.{:06}", fraction / 1_000)
    } else {
        text
    }
}

/// A zone-less time, `nanos` since the epoch, as pandas'
/// `Timestamp.isoformat` writes it: `YYYY-MM-DDTHH:MM:SS`, then six
/// fractional digits where it has microseconds and nine where it has
/// nanoseconds, none where it has neither.
fn isoformat(nanos: i128) -> String {
    let seconds = nanos.div_euclid(1_000_000_000);
    #[allow(clippy::cast_possible_truncation, reason = "below a second")]
    let fraction = nanos.rem_euclid(1_000_000_000) as i64;
    #[allow(clippy::cast_possible_truncation, reason = "days of an int64 count")]
    let (days, of_day) = (
        seconds.div_euclid(86_400) as i64,
        seconds.rem_euclid(86_400) as i64,
    );
    let (y, m, d) = civil(days);
    let text = format!(
        "{}-{m:02}-{d:02}T{:02}:{:02}:{:02}",
        year(y),
        of_day / 3_600,
        of_day / 60 % 60,
        of_day % 60
    );
    with_fraction(text, fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(json: &str) -> Option<Option<String>> {
        reference_text(json, |f| format!("{f:?}"))
    }

    #[test]
    fn a_json_scalar_is_the_reference_s_text_of_it() {
        assert_eq!(text("true"), Some(Some("True".into())));
        assert_eq!(text("false"), Some(Some("False".into())));
        assert_eq!(text("null"), Some(None));
        assert_eq!(text("3"), Some(Some("3".into())));
        assert_eq!(text("-0"), Some(Some("0".into())));
        assert_eq!(
            text("123456789012345678901234567890"),
            Some(Some("123456789012345678901234567890".into()))
        );
        assert_eq!(text("2.5"), Some(Some("2.5".into())));
        assert_eq!(text("\"x\""), Some(Some("x".into())));
        assert_eq!(text("{\"a\": 1}"), Some(Some("{\"a\":1}".into())));
        assert_eq!(
            text("[1, \"b\", true, null]"),
            Some(Some("[1,\"b\",true,null]".into()))
        );
        assert_eq!(
            text("{\"a\":1,\"b\":2,\"a\":3}"),
            Some(Some("{\"a\":3,\"b\":2}".into()))
        );
        assert_eq!(
            text("[\"\u{e9}\u{1f600}\\n\u{7f}\"]"),
            Some(Some("[\"\\u00e9\\ud83d\\ude00\\n\\u007f\"]".into()))
        );
        assert_eq!(text("[1e400]"), Some(Some("[Infinity]".into())));
        assert_eq!(text("{"), None);
        assert_eq!(text("01"), None);
        assert_eq!(text("1 2"), None);
    }

    /// T3.12 fix 1 (Codex P1-4): a JSON value handed over as the text the
    /// reference's provider writes of it (`json.dumps` of the value its
    /// driver read: compact, ASCII only), a JSON null none, at any depth
    /// the reference's reader reads; for a host whose driver gives the
    /// server's text, so the rule stays here.
    #[test]
    fn a_json_value_is_written_back_as_the_reference_s_provider_writes_it() {
        let written = |json: &str| json_text(json, |f| format!("{f:?}"));
        assert_eq!(
            written("{\"a\": 1, \"b\": [1.0, 1E2, -0, \"\\u00e9\\ud800\"], \"a\": 2}"),
            Some(Some(
                "{\"a\":2,\"b\":[1.0,100.0,0,\"\\u00e9\\ud800\"]}".into()
            ))
        );
        assert_eq!(written("null"), Some(None));
        assert_eq!(written(" \"text\" "), Some(Some("\"text\"".into())));
        assert_eq!(written("true"), Some(Some("true".into())));
        assert_eq!(
            written("12345678901234567890"),
            Some(Some("12345678901234567890".into()))
        );
        assert_eq!(written("[null, {}]"), Some(Some("[null,{}]".into())));
        let deep = format!("{}{}", "[".repeat(920), "]".repeat(920));
        assert_eq!(written(&deep), Some(Some(deep.clone())));
        assert_eq!(written("{"), None);
    }

    #[test]
    fn a_whole_number_s_digits() {
        assert_eq!(whole("12"), Some("12".into()));
        assert_eq!(whole("-12.000"), Some("-12".into()));
        assert_eq!(whole("1.2E+3"), Some("1200".into()));
        assert_eq!(whole("1.5"), None);
        assert_eq!(whole("-0"), Some("0".into()));
        assert_eq!(whole("x"), None);
    }

    #[test]
    fn a_seed_time_as_pandas_writes_it() {
        assert_eq!(isoformat(1_704_164_645_000_000_000), "2024-01-02T03:04:05");
        assert_eq!(
            isoformat(1_704_164_645_500_000_000),
            "2024-01-02T03:04:05.500000"
        );
        assert_eq!(
            isoformat(1_704_240_000_000_000_001),
            "2024-01-03T00:00:00.000000001"
        );
        assert_eq!(isoformat(-86_399_999_999_000), "1969-12-31T00:00:00.000001");
        assert_eq!(
            isoformat(-62_135_596_800_000_000_000),
            "0001-01-01T00:00:00"
        );
        assert_eq!(hex(&[0, 255]), "\\x00ff");
    }
}
