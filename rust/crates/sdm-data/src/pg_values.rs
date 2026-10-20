// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Postgres's value rules (T2.14d, MID2 P2-3): how the reference reads the
//! values its Postgres driver hands over, and how it binds a key set, here
//! in the dialect, so the host's provider only moves the driver's rows into
//! Arrow and binds the cells it is given.
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
//! - A key set's cells the reference spells otherwise than Arrow's text of
//!   them are handed to the provider as that spelling ([`bound_keys`]): a
//!   time without a zone as pandas' `Timestamp.isoformat` writes it (the
//!   reference compares a text time column with it as text), binary as
//!   `\x` and its hexadecimal digits; the provider binds every other
//!   value as Arrow's text of it.

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
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The field metadata naming Arrow's canonical JSON extension type.
const EXTENSION_NAME: &str = "ARROW:extension:name";
const JSON: &str = "arrow.json";
/// How deep a JSON value is read; the reference's driver reads none
/// deeper (its reader's recursion limit is about 1,000).
const MOST_DEPTH: usize = 2_000;

type Conformed = Option<(SchemaRef, Vec<RecordBatch>)>;

/// A value of a column the rule cannot read, refused by name: the
/// provider's answer is not what it says it is.
fn unreadable(column: &str, what: &str) -> CoreError {
    CoreError::new(
        ErrorKind::DataSource,
        ErrorCode::ProviderSchemaMismatch,
        format!("column {column:?} holds a value that is not {what}"),
    )
    .with_field("column", Detail::text(column))
}

fn internal(message: &str) -> CoreError {
    CoreError::new(ErrorKind::Internal, ErrorCode::InternalError, message)
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
            Some(scale) => scale
                .parse::<u32>()
                .map(Rule::Declared)
                .map_err(|_| unreadable(field.name(), "a declared scale")),
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
                Rule::Json => json_column(name, array.as_string::<i32>(), float_text)?,
                Rule::JsonList(item) => {
                    let list = array.as_list::<i32>();
                    let values = json_column(name, list.values().as_string::<i32>(), float_text)?;
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

fn json_column(
    column: &str,
    texts: &StringArray,
    float_text: fn(f64) -> String,
) -> Result<ArrayRef, CoreError> {
    let read: Result<StringArray, CoreError> = texts
        .iter()
        .map(|text| match text {
            None => Ok(None),
            Some(text) => {
                reference_text(text, float_text).ok_or_else(|| unreadable(column, "JSON"))
            }
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
    let number = |text: &str| unreadable(column, &format!("a number ({text:.40})"));
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
    Text(String),
    List(Vec<Json>),
    Object(Vec<(String, Json)>),
}

/// The reference's text of the JSON value `text` holds, a JSON null none;
/// `None` (outer) where `text` is not JSON.
fn reference_text(text: &str, float_text: fn(f64) -> String) -> Option<Option<String>> {
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
        Json::Bool(true) => Some("True".to_owned()),
        Json::Bool(false) => Some("False".to_owned()),
        Json::Int(digits) => Some(digits),
        Json::Float(f) => Some(float_text(f)),
        Json::Text(s) => Some(s),
        nested => {
            let mut out = String::new();
            write(&nested, float_text, &mut out);
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
        Json::Text(s) => quoted(s, out),
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
                quoted(name, out);
                out.push(':');
                write(item, float_text, out);
            }
            out.push('}');
        }
    }
}

/// A string as the reference's writer quotes one, ASCII only: `"` and `\`
/// escaped, the named control characters by name, every other character
/// outside space to `~` as `\u` and four lowercase hexadecimal digits, one
/// past the first plane as its two surrogates.
fn quoted(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
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
                let mut members: Vec<(String, Json)> = Vec::new();
                let mut named: BTreeMap<String, usize> = BTreeMap::new();
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

    fn text(&mut self) -> Option<String> {
        self.at += 1;
        let mut out = String::new();
        loop {
            let c = self.text.get(self.at..)?.chars().next()?;
            self.at += c.len_utf8();
            match c {
                '"' => return Some(out),
                '\\' => {
                    let escape = *self.bytes.get(self.at)?;
                    self.at += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let high = self.hex4()?;
                            let c = if (0xD800..0xDC00).contains(&high) {
                                if !self.eat("\\u") {
                                    return None;
                                }
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return None;
                                }
                                char::from_u32(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))?
                            } else {
                                char::from_u32(high)?
                            };
                            out.push(c);
                        }
                        _ => return None,
                    }
                }
                c if (c as u32) < 0x20 => return None,
                c => out.push(c),
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

/// A key set as Postgres's provider binds it: each column the reference
/// spells otherwise than Arrow's text of its values, by name, as that text
/// (a null a null): a time without a zone as pandas' `Timestamp.isoformat`
/// writes it, binary as `\x` and its hexadecimal digits. Every other
/// column stays as the core made it, which the provider binds as Arrow's
/// text of each value. `None` where no column is spelled otherwise.
pub fn bound_keys(schema: &SchemaRef, batches: &[RecordBatch]) -> Result<Conformed, CoreError> {
    let spelled = |kind: &DataType| {
        matches!(
            kind,
            DataType::Timestamp(_, None) | DataType::Binary | DataType::LargeBinary
        )
    };
    if !schema.fields().iter().any(|f| spelled(f.data_type())) {
        return Ok(None);
    }
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            if spelled(f.data_type()) {
                Field::new(f.name(), DataType::Utf8, f.is_nullable())
            } else {
                f.as_ref().clone()
            }
        })
        .collect();
    let bound: SchemaRef = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let columns = batch
            .columns()
            .iter()
            .map(|array| {
                if !spelled(array.data_type()) {
                    return Ok(Arc::clone(array));
                }
                let texts: StringArray = (0..array.len())
                    .map(|row| cell_text(array.as_ref(), row))
                    .collect();
                Ok(Arc::new(texts) as ArrayRef)
            })
            .collect::<Result<Vec<_>, CoreError>>()?;
        out.push(
            RecordBatch::try_new(Arc::clone(&bound), columns)
                .map_err(|_| internal("a bound key set does not build"))?,
        );
    }
    Ok(Some((bound, out)))
}

/// One cell of a column [`bound_keys`] spells, as the text the reference
/// binds it as.
fn cell_text(array: &dyn Array, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    Some(match array.data_type() {
        DataType::Binary => hex(array.as_binary::<i32>().value(row)),
        DataType::LargeBinary => hex(array.as_binary::<i64>().value(row)),
        DataType::Timestamp(unit, _) => {
            let (ticks, per) = match unit {
                TimeUnit::Second => (
                    array.as_primitive::<TimestampSecondType>().value(row),
                    1_000_000_000,
                ),
                TimeUnit::Millisecond => (
                    array.as_primitive::<TimestampMillisecondType>().value(row),
                    1_000_000,
                ),
                TimeUnit::Microsecond => (
                    array.as_primitive::<TimestampMicrosecondType>().value(row),
                    1_000,
                ),
                TimeUnit::Nanosecond => (
                    array.as_primitive::<TimestampNanosecondType>().value(row),
                    1,
                ),
            };
            isoformat(i128::from(ticks) * per)
        }
        _ => return None,
    })
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
