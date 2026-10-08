// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The engine's metadata inference over an inference sample:
//! `rfm/metadata_inference/*` (data type, semantic type, primary key, time
//! column), on a sample frame whose columns are what pandas holds after
//! `to_pandas(types_mapper=pd.ArrowDtype)` and `Table._sanitize`.

use std::collections::HashSet;
use std::sync::OnceLock;

use arrow_schema::DataType;
use regex::Regex;
use sdm_pql::types::{Dtype, Stype};

use crate::dt::{self, Call, Host};
use crate::frame;
use crate::pyval::PyVal;

/// How pandas holds one column of a sample.
#[derive(Clone, Debug, PartialEq)]
pub enum SampleKind {
    /// `pd.ArrowDtype(type)`.
    Arrow(DataType),
    /// `datetime64[ns]`, after `_sanitize` read it as times.
    Datetime64,
    /// A nullable extension dtype: `Int64`, `Float64`, `boolean`, `string`.
    Nullable(&'static str),
    /// `object`.
    Object,
}

/// One column of a sample.
#[derive(Clone, Debug)]
pub struct SampleColumn {
    pub name: String,
    pub kind: SampleKind,
    pub cells: Vec<PyVal>,
}

/// A sample frame: columns of equal length, in order.
#[derive(Clone, Debug, Default)]
pub struct SampleFrame {
    pub columns: Vec<SampleColumn>,
    /// The rows of a frame of no columns (`pd.DataFrame(index=range(n))`).
    pub index: usize,
}

impl SampleFrame {
    pub fn empty_index(rows: usize) -> Self {
        SampleFrame {
            columns: Vec::new(),
            index: rows,
        }
    }

    pub fn len(&self) -> usize {
        self.columns.first().map_or(self.index, |c| c.cells.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn column(&self, name: &str) -> Option<&SampleColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.column(name).is_some()
    }

    /// `pd.concat([a.iloc[:n], b.iloc[:n]], axis=1)` without the duplicate
    /// names, the last kept, where `n` is the shorter length.
    pub fn concat(a: &SampleFrame, b: &SampleFrame) -> SampleFrame {
        let n = a.len().min(b.len());
        let mut columns: Vec<SampleColumn> = Vec::new();
        for c in a.columns.iter().chain(&b.columns) {
            let cut = SampleColumn {
                name: c.name.clone(),
                kind: c.kind.clone(),
                cells: c.cells.iter().take(n).cloned().collect(),
            };
            if let Some(i) = columns.iter().position(|x| x.name == c.name) {
                columns.remove(i);
                columns.insert(i, cut);
            } else {
                columns.push(cut);
            }
        }
        // `~df.columns.duplicated(keep='last')` keeps the last position.
        let mut out: Vec<SampleColumn> = Vec::new();
        let all: Vec<&SampleColumn> = a.columns.iter().chain(&b.columns).collect();
        for (i, c) in all.iter().enumerate() {
            if all[i + 1..].iter().any(|later| later.name == c.name) {
                continue;
            }
            out.push(
                columns
                    .iter()
                    .find(|x| x.name == c.name)
                    .cloned()
                    .expect("kept above"),
            );
        }
        SampleFrame {
            columns: out,
            index: n,
        }
    }

    /// `df.drop(columns=names, errors='ignore')`.
    pub fn drop(&mut self, names: &[String]) {
        self.columns.retain(|c| !names.contains(&c.name));
    }
}

fn present(cells: &[PyVal]) -> Vec<&PyVal> {
    cells.iter().filter(|c| !frame::missing(c)).collect()
}

/// `series.nunique()` (dropping missing values).
pub fn nunique(cells: &[PyVal]) -> usize {
    let mut seen = HashSet::new();
    for c in present(cells) {
        seen.insert(c.eq_key());
    }
    seen.len()
}

/// `is_datetime(ser)`.
pub fn is_datetime(column: &SampleColumn) -> bool {
    match &column.kind {
        SampleKind::Arrow(dt) => matches!(
            dt,
            DataType::Timestamp(..)
                | DataType::Date32
                | DataType::Date64
                | DataType::Time32(_)
                | DataType::Time64(_)
        ),
        SampleKind::Datetime64 => true,
        _ => false,
    }
}

/// pandas' `infer_dtype(values, skipna=True)` over cells, as far as the
/// engine asks it.
pub fn content_label(cells: &[PyVal]) -> &'static str {
    let values: Vec<&PyVal> = present(cells);
    if values.is_empty() {
        return "empty";
    }
    let unwrap = |v: &PyVal| -> PyVal {
        match v {
            PyVal::Numpy(_, inner) => inner.as_ref().clone(),
            other => other.clone(),
        }
    };
    let kinds: Vec<&str> = values
        .iter()
        .map(|v| match unwrap(v) {
            PyVal::Str(_) => "string",
            PyVal::Bytes(_) => "bytes",
            PyVal::Bool(_) => "boolean",
            PyVal::Int(_) | PyVal::BigInt(_) => "integer",
            PyVal::Float(_) => "floating",
            PyVal::Decimal(_) => "decimal",
            PyVal::Date(..) => "date",
            PyVal::DateTime(..) | PyVal::Timestamp(_) => "datetime",
            PyVal::Time { .. } => "time",
            PyVal::Timedelta(_) => "timedelta",
            _ => "other",
        })
        .collect();
    let all = |k: &str| kinds.iter().all(|x| *x == k);
    for k in [
        "string",
        "bytes",
        "boolean",
        "integer",
        "floating",
        "decimal",
        "date",
        "datetime",
        "time",
        "timedelta",
    ] {
        if all(k) {
            return match k {
                "string" => "string",
                "bytes" => "bytes",
                "boolean" => "boolean",
                "integer" => "integer",
                "floating" => "floating",
                "decimal" => "decimal",
                "date" => "date",
                "time" => "time",
                "timedelta" => "timedelta",
                _ => "datetime",
            };
        }
    }
    if kinds.iter().all(|k| *k == "integer" || *k == "floating") {
        return "mixed-integer-float";
    }
    if kinds.contains(&"integer") {
        return "mixed-integer";
    }
    "mixed"
}

/// An unsupported column, as `infer_dtype` raises it.
#[derive(Debug)]
pub struct Unsupported(pub String);

/// pandas' name of a sample column's dtype.
pub fn dtype_name(column: &SampleColumn) -> String {
    match &column.kind {
        SampleKind::Arrow(dt) => format!("{}[pyarrow]", arrow_type_name(dt)),
        SampleKind::Datetime64 => "datetime64[ns]".to_string(),
        SampleKind::Nullable(name) => (*name).to_string(),
        SampleKind::Object => "object".to_string(),
    }
}

/// `str(pyarrow_type)`.
pub fn arrow_type_name(dt: &DataType) -> String {
    match dt {
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
        DataType::Binary => "binary".into(),
        DataType::LargeBinary => "large_binary".into(),
        DataType::Date32 => "date32[day]".into(),
        DataType::Decimal128(p, s) => format!("decimal128({p}, {s})"),
        DataType::List(item) => format!("list<{}>", item_name(item)),
        DataType::LargeList(item) => format!("large_list<{}>", item_name(item)),
        DataType::FixedSizeList(item, n) => format!("fixed_size_list<{}>[{n}]", item_name(item)),
        other => other.to_string().to_lowercase(),
    }
}

/// A list's item as `str(pyarrow_type)` writes it: `item: int64`.
fn item_name(item: &arrow_schema::Field) -> String {
    let required = if item.is_nullable() { "" } else { " not null" };
    format!(
        "{}: {}{required}",
        item.name(),
        arrow_type_name(item.data_type())
    )
}

/// `infer_dtype(ser)` of the engine.
pub fn infer_dtype(column: &SampleColumn) -> Result<Dtype, Unsupported> {
    // A time of day, which no date goes with, is text: today's took it for
    // a moment the rows could be ordered by, and could read none of it.
    if let SampleKind::Arrow(DataType::Time32(_) | DataType::Time64(_)) = column.kind {
        return Ok(Dtype::String);
    }
    if is_datetime(column) {
        return Ok(Dtype::Date);
    }
    if let SampleKind::Arrow(DataType::Duration(_)) = column.kind {
        return Ok(Dtype::Timedelta);
    }
    if column.kind == SampleKind::Object {
        let first = column
            .cells
            .iter()
            .take(1000)
            .position(|c| !frame::missing(c));
        if let Some(i) =
            first.filter(|&i| matches!(column.cells[i], PyVal::List(_) | PyVal::Tuple(_)))
        {
            let elems: Vec<PyVal> = column.cells[i..]
                .iter()
                .take(1000)
                .filter(|c| !frame::missing(c))
                .flat_map(|c| match c {
                    PyVal::List(v) | PyVal::Tuple(v) => v.clone(),
                    _ => Vec::new(),
                })
                .collect();
            return Ok(match content_label(&elems) {
                "integer" => Dtype::Intlist,
                "floating" | "decimal" | "empty" | "mixed-integer-float" => Dtype::Floatlist,
                "string" => Dtype::Stringlist,
                _ => return Err(Unsupported("object".to_string())),
            });
        }
        let head: Vec<PyVal> = column.cells.iter().take(1000).cloned().collect();
        return Ok(match content_label(&head) {
            "integer" => Dtype::Int,
            "floating" | "mixed-integer-float" | "decimal" => Dtype::Float,
            "boolean" => Dtype::Bool,
            "timedelta" => Dtype::Timedelta,
            _ => Dtype::String,
        });
    }
    // `pa.types.is_list` or `is_fixed_size_list`: a `large_list` is
    // neither, and its name is in no table.
    if let SampleKind::Arrow(DataType::List(inner) | DataType::FixedSizeList(inner, _)) =
        &column.kind
    {
        let elem = inner.data_type();
        if elem.is_integer() {
            return Ok(Dtype::Intlist);
        }
        if elem.is_floating()
            || matches!(
                elem,
                DataType::Decimal128(..) | DataType::Decimal256(..) | DataType::Null
            )
        {
            return Ok(Dtype::Floatlist);
        }
        if matches!(elem, DataType::Utf8) {
            return Ok(Dtype::Stringlist);
        }
    }
    let name = match &column.kind {
        SampleKind::Arrow(dt) => arrow_type_name(dt).to_lowercase(),
        SampleKind::Nullable(n) => n.to_lowercase(),
        SampleKind::Datetime64 => "datetime64[ns]".into(),
        SampleKind::Object => "object".into(),
    };
    pandas_dtype(&name).ok_or_else(|| Unsupported(dtype_name(column)))
}

/// The data type the engine gives a column of the pandas dtype `name`,
/// with its backend in brackets or not (`PANDAS_TO_DTYPE`), or `None` where
/// `infer_dtype` raises.
pub fn pandas_dtype(name: &str) -> Option<Dtype> {
    let name = name.to_lowercase();
    Some(match name.split('[').next().unwrap_or("") {
        "bool" | "boolean" => Dtype::Bool,
        "uint8" | "uint16" | "uint32" | "uint64" | "int8" | "int16" | "int32" | "int64" => {
            Dtype::Int
        }
        "float" | "double" | "float16" | "float32" | "float64" => Dtype::Float,
        "object" | "str" | "string" => Dtype::String,
        "binary" => Dtype::Binary,
        _ => return None,
    })
}

fn ignore_case(pattern: &str) -> Regex {
    Regex::new(&format!("(?i){pattern}")).expect("pattern compiles")
}

const IGNORED_ID_SUFFIXES: [&str; 16] = [
    "bid", "acid", "grid", "maid", "paid", "raid", "void", "avoid", "braid", "covid", "fluid",
    "rabid", "solid", "hybrid", "inlaid", "liquid",
];

/// `contains_id`.
pub fn contains_id(column_name: &str, dtype: Dtype) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    if !Stype::Id.supports_dtype(dtype) {
        return false;
    }
    let name = column_name.to_lowercase();
    if RE
        .get_or_init(|| ignore_case(r"(^|_)(id|hash|key|code|uuid)(_|$)"))
        .is_match(&name)
    {
        return true;
    }
    if !name.ends_with("id") {
        return false;
    }
    !IGNORED_ID_SUFFIXES.iter().any(|s| name.ends_with(s))
}

/// What a cell is as `astype(str)` writes it.
pub fn cell_str(cell: &PyVal) -> String {
    match cell {
        PyVal::Numpy(_, inner) => inner.str_(),
        other => other.str_(),
    }
}

/// `contains_timestamp`, and the warnings it lets through.
pub fn contains_timestamp(
    column: &SampleColumn,
    dtype: Dtype,
    host: &Host,
) -> (Result<bool, dt::PyErr>, Vec<String>) {
    if !Stype::Timestamp.supports_dtype(dtype) {
        return (Ok(false), Vec::new());
    }
    if dtype.is_timestamp() {
        return (Ok(true), Vec::new());
    }
    // Times of day are no moments, though their text reads as today's.
    let mut present = column
        .cells
        .iter()
        .filter(|c| !frame::missing(c))
        .peekable();
    if present.peek().is_some() && present.all(|c| matches!(c, PyVal::Time { .. })) {
        return (Ok(false), Vec::new());
    }
    let values: Vec<Option<String>> = column
        .cells
        .iter()
        .take(100)
        .filter(|c| !frame::missing(c))
        .filter(|c| !matches!(c, PyVal::Str(s) if s.is_empty()))
        .map(|c| Some(cell_str(c)))
        .collect();
    crate::dates::contains_timestamp(&values, &column.name, host)
}

const MAX_CAT: usize = 100;

/// `contains_multicategorical`.
pub fn contains_multicategorical(column: &SampleColumn, dtype: Dtype) -> bool {
    if !Stype::Multicategorical.supports_dtype(dtype) {
        return false;
    }
    if dtype == Dtype::Stringlist {
        return true;
    }
    let ser: Vec<&PyVal> = column
        .cells
        .iter()
        .take(500)
        .filter(|c| !frame::missing(c))
        .collect();
    let mut num_unique = 0;
    let parts: Vec<PyVal>;
    if dtype == Dtype::String {
        let texts: Vec<String> = ser.iter().map(|c| cell_str(c)).collect();
        let text = texts.join("\n");
        let mut candidates: Vec<(char, usize)> = Vec::new();
        for ch in text.chars() {
            if matches!(ch, ';' | ':' | '|' | '\t') {
                match candidates.iter_mut().find(|(c, _)| *c == ch) {
                    Some(slot) => slot.1 += 1,
                    None => candidates.push((ch, 1)),
                }
            }
        }
        if candidates.is_empty() {
            return false;
        }
        let strs: Vec<PyVal> = texts.iter().map(|t| PyVal::Str(t.clone())).collect();
        num_unique = nunique(&strs);
        let best = candidates.iter().map(|(_, n)| *n).max().unwrap_or(0);
        let sep = candidates
            .iter()
            .find(|(_, n)| *n == best)
            .map(|(c, _)| *c)
            .unwrap_or(';');
        parts = texts
            .iter()
            .flat_map(|t| {
                t.split(sep)
                    .map(|p| PyVal::Str(p.to_string()))
                    .collect::<Vec<_>>()
            })
            .collect();
    } else {
        parts = ser
            .iter()
            .flat_map(|c| match c {
                PyVal::List(v) | PyVal::Tuple(v) => {
                    if v.is_empty() {
                        vec![PyVal::Float(f64::NAN)]
                    } else {
                        v.clone()
                    }
                }
                other => vec![(*other).clone()],
            })
            .collect();
    }
    let num_unique_multi = nunique(&parts);
    if dtype.is_list() {
        return num_unique_multi <= MAX_CAT;
    }
    num_unique as f64 > 1.5 * num_unique_multi as f64 && num_unique_multi <= MAX_CAT
}

/// `contains_categorical`.
pub fn contains_categorical(column: &SampleColumn, dtype: Dtype) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    if !Stype::Categorical.supports_dtype(dtype) {
        return false;
    }
    if dtype == Dtype::Bool {
        return true;
    }
    if dtype.is_numerical()
        && RE
            .get_or_init(|| {
                ignore_case(r"(^|_)(price|sales|amount|quantity|total|cost|score|rating|avg|average|recency|age|num|pos|number|position)(_|$)")
            })
            .is_match(&column.name)
    {
        return false;
    }
    let ser: Vec<PyVal> = column
        .cells
        .iter()
        .take(1000)
        .filter(|c| !frame::missing(c))
        .cloned()
        .collect();
    let num_unique = nunique(&ser);
    if num_unique < 20 {
        return true;
    }
    let ratio = num_unique as f64 / ser.len() as f64;
    if dtype.is_string() {
        return ratio <= 0.5;
    }
    ratio <= 0.05
}

/// `infer_stype(ser, name, dtype)`, and the warnings it lets through.
pub fn infer_stype(
    column: &SampleColumn,
    dtype: Dtype,
    host: &Host,
) -> (Result<Stype, dt::PyErr>, Vec<String>) {
    if contains_id(&column.name, dtype) {
        return (Ok(Stype::Id), Vec::new());
    }
    let (timestamp, warnings) = contains_timestamp(column, dtype, host);
    match timestamp {
        Err(e) => return (Err(e), warnings),
        Ok(true) => return (Ok(Stype::Timestamp), warnings),
        Ok(false) => {}
    }
    if contains_multicategorical(column, dtype) {
        return (Ok(Stype::Multicategorical), warnings);
    }
    if contains_categorical(column, dtype) {
        return (Ok(Stype::Categorical), warnings);
    }
    (Ok(dtype.default_stype()), warnings)
}

fn snake_case(name: &str) -> String {
    static A: OnceLock<Regex> = OnceLock::new();
    static B: OnceLock<Regex> = OnceLock::new();
    let a = A.get_or_init(|| Regex::new(r"(.)([A-Z][a-z]+)").expect("pattern compiles"));
    let b = B.get_or_init(|| Regex::new(r"([a-z0-9])([A-Z])").expect("pattern compiles"));
    let first = a.replace_all(name, "${1}_${2}").to_string();
    b.replace_all(&first, "${1}_${2}").to_string()
}

fn chop(text: &str, n: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    chars[..chars.len().saturating_sub(n)].iter().collect()
}

/// `infer_primary_key(table_name, df, candidates)`, and the warning it
/// gives when candidates tie.
pub fn infer_primary_key(
    table_name: &str,
    df: &SampleFrame,
    candidates: &[String],
) -> (Option<String>, Vec<String>) {
    if candidates.is_empty() {
        return (None, Vec::new());
    }
    let mut bases = vec![table_name.to_string()];
    let lower = table_name.to_lowercase();
    if lower.ends_with("_info") {
        bases.push(chop(table_name, 5));
    } else if lower.ends_with("info") {
        bases.push(chop(table_name, 4));
    }
    let mut names: Vec<String> = Vec::new();
    let add = |n: String, names: &mut Vec<String>| {
        if !names.contains(&n) {
            names.push(n);
        }
    };
    for base in &bases {
        let low = base.to_lowercase();
        let snake = snake_case(base).to_lowercase();
        let camel = base.replace('_', "").to_lowercase();
        add(low.clone(), &mut names);
        add(snake.clone(), &mut names);
        add(camel.clone(), &mut names);
        if low.ends_with('s') {
            for v in [&low, &snake, &camel] {
                add(chop(v, 1), &mut names);
            }
        } else {
            for v in [&low, &snake, &camel] {
                add(format!("{v}s"), &mut names);
            }
        }
        if low.ends_with("ies") {
            for v in [&low, &snake, &camel] {
                add(format!("{}y", chop(v, 3)), &mut names);
            }
        } else if low.ends_with('y') {
            for v in [&low, &snake, &camel] {
                add(format!("{}ies", chop(v, 1)), &mut names);
            }
        }
        if low.ends_with("ing") {
            for v in [&low, &snake, &camel] {
                add(chop(v, 3), &mut names);
            }
        }
    }
    let mut scores: Vec<(String, f64)> = Vec::new();
    for col in candidates {
        let col_lower = col.to_lowercase();
        let mut score = 0.0;
        if col_lower == "id" {
            score += 4.0;
        }
        // A name the column equals scores once, as the first table name
        // it meets does in Python; the set's order decides only which
        // suffixes count before it.
        if names.contains(&col_lower) {
            score += 4.0;
        } else {
            for tn in &names {
                for suffix in ["id", "hash", "key", "code", "uuid"] {
                    if !col_lower.ends_with(suffix) {
                        continue;
                    }
                    if col_lower == format!("{tn}_{suffix}") || col_lower == format!("{tn}{suffix}")
                    {
                        score += 5.0;
                        break;
                    }
                    if col_lower.ends_with(&format!("{tn}_{suffix}")) {
                        score += 2.0;
                    }
                    if col_lower.ends_with(&format!("{tn}{suffix}")) {
                        score += 2.0;
                    }
                }
                if table_name == "studies" && col == "nct_id" {
                    score += 1.0;
                }
            }
        }
        let ser = df.column(col).map(|c| c.cells.clone()).unwrap_or_default();
        let ratio = nunique(&ser) as f64 / ser.len() as f64;
        score += 3.0 * ratio;
        scores.push((col.clone(), score));
    }
    let mut scores: Vec<(String, f64)> = scores.into_iter().filter(|(_, s)| *s >= 4.0).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    match scores.len() {
        0 => (None, Vec::new()),
        1 => (Some(scores[0].0.clone()), Vec::new()),
        _ if scores[0].1 != scores[1].1 => (Some(scores[0].0.clone()), Vec::new()),
        _ => {
            let max = scores[0].1;
            let tied: Vec<String> = scores
                .iter()
                .filter(|(_, s)| *s == max)
                .map(|(c, _)| c.clone())
                .collect();
            (
                None,
                vec![format!(
                    "Found multiple potential primary keys in table '{table_name}': {}. Please specify the primary key for this table manually.",
                    crate::py::str_list_repr(&tied)
                )],
            )
        }
    }
}

/// The engine's `to_datetime(ser)`: each value as a time, missing where it
/// does not read, as UTC wall nanoseconds.
pub fn to_times(column: &SampleColumn, host: &Host) -> Vec<Option<i128>> {
    if is_datetime(column) {
        return column
            .cells
            .iter()
            .map(|c| match c {
                PyVal::DateTime(w, offset) => {
                    Some(w.epoch_nanos() - i128::from(offset.unwrap_or(0)) * 1_000_000_000)
                }
                PyVal::Timestamp(w) => Some(w.epoch_nanos()),
                PyVal::Date(y, m, d) => {
                    Some(crate::dt::civil::Wall::date(*y, *m, *d).epoch_nanos())
                }
                _ => None,
            })
            .collect();
    }
    let texts: Vec<Option<String>> = column
        .cells
        .iter()
        .map(|c| match c {
            PyVal::Str(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let call = Call {
        coerce: true,
        ..Call::default()
    };
    match dt::pandas::objects_to_datetime_public(&texts, call, host) {
        Ok(converted) => converted.values,
        Err(_) => vec![None; texts.len()],
    }
}

/// `infer_time_column(df, candidates)`.
pub fn infer_time_column(df: &SampleFrame, candidates: &[String], host: &Host) -> Option<String> {
    crate::time_columns::choose(df, candidates, None, false, host).column
}

/// `_is_key_like(ser)`.
pub fn is_key_like(column: &SampleColumn) -> bool {
    !column.cells.is_empty()
        && !column.cells.iter().any(frame::missing)
        && nunique(&column.cells) == column.cells.len()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::Field;

    use super::*;

    fn infer(dt: DataType) -> Result<Dtype, String> {
        infer_dtype(&SampleColumn {
            name: "c".to_string(),
            kind: SampleKind::Arrow(dt),
            cells: Vec::new(),
        })
        .map_err(|Unsupported(name)| name)
    }

    #[test]
    fn arrow_lists_infer_as_the_engine_infers_them() {
        // infer_dtype(pd.Series(..., dtype=pd.ArrowDtype(t))), from Python.
        let item = |dt: DataType| Arc::new(Field::new("item", dt, true));
        assert_eq!(
            infer(DataType::List(item(DataType::Int64))),
            Ok(Dtype::Intlist)
        );
        assert_eq!(
            infer(DataType::FixedSizeList(item(DataType::Int64), 2)),
            Ok(Dtype::Intlist)
        );
        assert_eq!(
            infer(DataType::List(item(DataType::Utf8))),
            Ok(Dtype::Stringlist)
        );
        assert_eq!(
            infer(DataType::LargeList(item(DataType::Int64))),
            Err("large_list<item: int64>[pyarrow]".to_string())
        );
        assert_eq!(
            infer(DataType::LargeList(item(DataType::Utf8))),
            Err("large_list<item: string>[pyarrow]".to_string())
        );
        assert_eq!(
            infer(DataType::List(item(DataType::LargeUtf8))),
            Err("list<item: large_string>[pyarrow]".to_string())
        );
    }

    #[test]
    fn list_types_are_named_as_pyarrow_names_them() {
        // str(t), from Python.
        let required = Arc::new(Field::new("x", DataType::Int64, false));
        assert_eq!(
            arrow_type_name(&DataType::List(required)),
            "list<x: int64 not null>"
        );
        let item = |dt: DataType| Arc::new(Field::new("item", dt, true));
        assert_eq!(
            arrow_type_name(&DataType::FixedSizeList(item(DataType::Int64), 3)),
            "fixed_size_list<item: int64>[3]"
        );
        assert_eq!(
            arrow_type_name(&DataType::List(item(DataType::List(item(
                DataType::Float64
            ))))),
            "list<item: list<item: double>>"
        );
        assert_eq!(
            arrow_type_name(&DataType::List(item(DataType::Decimal128(10, 2)))),
            "list<item: decimal128(10, 2)>"
        );
    }
}
