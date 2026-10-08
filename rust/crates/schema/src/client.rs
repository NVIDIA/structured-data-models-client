// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/schema.py`: a schema's tables, columns, keys and links, how
//! they are inferred from frames, checked, edited and said, and
//! `to_schema_definition`.
//!
//! The Python objects keep each other in step as they are assigned; the
//! assignments that do are methods here, run in the same order, so a table
//! and its columns pass through the same states.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use sdm_pql::types::{Dtype, Stype};
use serde_json::{Value, json};

use crate::dates;
use crate::dt::Host;
use crate::frame::{self, Frame, PdType};
use crate::py::{self, str_repr, thousands};
use crate::pyval::PyVal;

/// The warning class `sdm_client` warns with.
pub const SDM_WARNING: &str = "sdm_client.core.warnings.SDMWarning";

pub(crate) const SEMANTIC_TYPES: [&str; 7] = [
    "ID",
    "categorical",
    "multicategorical",
    "numerical",
    "sequence",
    "text",
    "timestamp",
];
pub(crate) const MAX_CATEGORICAL_CARDINALITY: usize = 64;
pub(crate) const KEY_UNIQUENESS: f64 = 0.99;
pub(crate) const NEARLY_A_KEY: f64 = 0.9;
pub(crate) const KEY_SUFFIXES: [&str; 4] = ["_id", "_key", "id", "key"];
pub(crate) const UNSATURATED: f64 = 0.5;
pub(crate) const WITHIN_REACH: f64 = 0.5;
pub(crate) const NAME_ANOTHER_KEY: &str =
    " Name a different primary key if this column is a foreign key rather than an identifier.";

/// A warning, by class and text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Warning {
    pub category: String,
    pub message: String,
}

/// A refusal: the error's code (or class) and message, and the errors it
/// was raised from (`sdm_protocol::Diagnostic::chain`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub code: String,
    pub message: String,
    pub chain: Vec<Value>,
    pub details: Box<serde_json::Map<String, Value>>,
}

impl Refusal {
    pub fn invalid(message: impl Into<String>) -> Self {
        Refusal {
            code: "INVALID_REQUEST".to_string(),
            message: message.into(),
            chain: Vec::new(),
            details: Box::default(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Refusal {
            code: "UNSUPPORTED_REQUEST".to_string(),
            message: message.into(),
            chain: Vec::new(),
            details: Box::default(),
        }
    }

    pub fn class(class: &str, message: impl Into<String>) -> Self {
        Refusal {
            code: class.to_string(),
            message: message.into(),
            chain: Vec::new(),
            details: Box::default(),
        }
    }

    /// The same refusal, raised from `links`, outermost first.
    #[must_use]
    pub fn caused_by(mut self, links: impl IntoIterator<Item = Value>) -> Self {
        self.chain.extend(links);
        self
    }

    /// The module-qualified class today's code raises this refusal as.
    pub fn raised_as(&self) -> String {
        let is_code = self
            .code
            .chars()
            .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit());
        if is_code {
            return "sdm_client.errors.StructuredDataError".to_string();
        }
        crate::diagnostics::public_class(&self.code)
    }

    /// The refusal as a host reads one: `{"code", "message", "class",
    /// "chain"}`.
    pub fn to_json(&self) -> Value {
        let mut value = json!({
            "code": self.code,
            "message": self.message,
            "class": self.raised_as(),
            "chain": self.chain,
        });
        if !self.details.is_empty() {
            value["details"] = Value::Object((*self.details).clone());
        }
        value
    }
}

pub type R<T> = Result<T, Refusal>;

/// Where warnings go, in order.
#[derive(Default)]
pub struct Said {
    pub warnings: Vec<Warning>,
    /// Classes being ignored, as `warnings.simplefilter('ignore', ...)`.
    pub ignoring: Vec<String>,
}

impl Said {
    pub fn warn(&mut self, category: &str, message: impl Into<String>) {
        if self.ignoring.iter().any(|c| c == category) {
            return;
        }
        self.warnings.push(Warning {
            category: category.to_string(),
            message: message.into(),
        });
    }

    pub(crate) fn sdm(&mut self, message: impl Into<String>) {
        self.warn(SDM_WARNING, message);
    }
}

/// A primary key: one column, or several (a tuple in Python).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Key {
    One(String),
    Many(Vec<String>),
}

impl Key {
    pub fn names(&self) -> Vec<String> {
        match self {
            Key::One(n) => vec![n.clone()],
            Key::Many(v) => v.clone(),
        }
    }

    pub(crate) fn repr(&self) -> String {
        match self {
            Key::One(n) => str_repr(n),
            Key::Many(v) => tuple_repr(v),
        }
    }
}

pub(crate) fn tuple_repr(names: &[String]) -> String {
    let parts: Vec<String> = names.iter().map(|n| str_repr(n)).collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// The names of a primary key's columns, none where there is no key.
pub fn key_names_of(key: &Option<Key>) -> Vec<String> {
    key_names(key)
}

pub(crate) fn key_names(key: &Option<Key>) -> Vec<String> {
    key.as_ref().map(Key::names).unwrap_or_default()
}

/// One column's description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Col {
    pub name: String,
    pub dtype: String,
    pub stype: String,
    pub untyped: bool,
    pub stype_declared: bool,
    pub before_key: Option<String>,
    /// The `repr` of the key a caller put the column under in
    /// `table.columns`, where that key is not text (`name` is its `str`).
    /// `Schema.from_data` refuses such labels, so only a caller's edit of
    /// `table.columns` makes one.
    pub name_repr: Option<String>,
}

impl Col {
    pub(crate) fn new(name: &str, dtype: &str, stype: &str) -> R<Col> {
        check_stype(stype)?;
        Ok(Col {
            name: name.to_string(),
            dtype: dtype.to_string(),
            stype: stype.to_string(),
            untyped: false,
            stype_declared: false,
            before_key: None,
            name_repr: None,
        })
    }

    pub(crate) fn mark_as_key(&mut self) {
        if self.stype == "ID" {
            return;
        }
        self.before_key = Some(self.stype.clone());
        self.stype = "ID".to_string();
    }
}

/// One table: a frame's rows, or a table in a source.
#[derive(Clone, Debug)]
pub struct Table {
    pub name: String,
    pub data: Option<Arc<Frame>>,
    pub source_name: Option<String>,
    pub columns: Vec<Col>,
    pub primary_key: Option<Key>,
    pub time_column: Option<String>,
    pub duplicate_keys: Option<i64>,
}

/// A foreign key, `source_table.source_column -> target_table.target_column`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rel {
    pub source_table: String,
    pub source_column: String,
    pub target_table: String,
    pub target_column: String,
}

impl Rel {
    pub fn new(st: &str, sc: &str, tt: &str, tc: &str) -> Rel {
        Rel {
            source_table: st.to_string(),
            source_column: sc.to_string(),
            target_table: tt.to_string(),
            target_column: tc.to_string(),
        }
    }

    /// `repr(link)`.
    pub fn text(&self) -> String {
        format!(
            "{}.{} -> {}.{}",
            self.source_table, self.source_column, self.target_table, self.target_column
        )
    }
}

/// A schema.
#[derive(Clone, Debug)]
pub struct Schema {
    pub tables: Vec<Table>,
    pub relationships: Vec<Rel>,
}

pub(crate) fn check_stype(value: &str) -> R<()> {
    if !SEMANTIC_TYPES.contains(&value) {
        let mut sorted: Vec<&str> = SEMANTIC_TYPES.to_vec();
        sorted.sort_unstable();
        return Err(Refusal::invalid(format!(
            "{} is not a semantic type; use one of {}",
            str_repr(value),
            py::str_list_repr(&sorted)
        )));
    }
    Ok(())
}

/// The language's spelling of a wire dtype.
pub(crate) fn pql_dtype(dtype: &str) -> &str {
    match dtype {
        "int64" | "int32" => "int",
        "float64" | "float32" => "float",
        "timestamp[us]" | "timestamp[ns]" | "timestamp[ms]" | "timestamp[s]" => "time",
        other => other,
    }
}

/// `supports_dtype(stype, dtype)`: an unknown pair is allowed through.
pub fn supports_dtype(stype: &str, dtype: &str) -> bool {
    match (Stype::parse(stype), Dtype::parse(pql_dtype(dtype))) {
        (Some(s), Some(d)) => s.supports_dtype(d),
        _ => true,
    }
}

pub(crate) fn join_family(dtype: &str) -> &str {
    if dtype.starts_with("int") || dtype.starts_with("float") {
        "number"
    } else if dtype.starts_with("timestamp") {
        "timestamp"
    } else {
        dtype
    }
}

/// `_and_list`.
pub fn and_list(names: &[String]) -> String {
    let quoted: Vec<String> = names.iter().map(|n| str_repr(n)).collect();
    if quoted.len() < 3 {
        return quoted.join(" and ");
    }
    format!(
        "{} and {}",
        quoted[..quoted.len() - 1].join(", "),
        quoted[quoted.len() - 1]
    )
}

pub(crate) fn sorted_names<'a>(names: impl Iterator<Item = &'a String>) -> Vec<String> {
    let mut v: Vec<String> = names.cloned().collect();
    v.sort();
    v
}

pub(crate) fn the_key_of(table: &str, names: &[String]) -> String {
    if names.len() == 1 {
        return format!("{table}.{}", names[0]);
    }
    format!(
        "the primary key of {}, {},",
        str_repr(table),
        and_list(names)
    )
}

pub(crate) fn too_many_repeats(duplicates: i64, rows: usize) -> bool {
    duplicates as f64 > rows as f64 * (1.0 - KEY_UNIQUENESS)
}

pub(crate) fn not_unique_enough(
    table: &str,
    names: &[String],
    duplicates: i64,
    rows: usize,
    example: &str,
) -> String {
    format!(
        "{} is not unique enough to identify a row: {} of {} rows repeat a value already used, e.g. {example}",
        the_key_of(table, names),
        thousands(duplicates),
        thousands(rows as i64)
    )
}

pub(crate) fn a_key_missing_values(
    table: &str,
    names: &[String],
    nulls: i64,
    empty: Option<&[String]>,
) -> String {
    let place = match empty {
        Some(e) if !e.is_empty() => format!(", in {},", and_list(e)),
        _ => " in part of it,".to_string(),
    };
    let several = empty.is_none_or(|e| e.len() > 1);
    format!(
        "{} has {} row(s) with no value{place} so those rows cannot be identified; fill or drop them before using {} as a primary key",
        the_key_of(table, names),
        thousands(nulls),
        if several { "them" } else { "it" }
    )
}

pub(crate) fn warn_a_key_repeats(
    said: &mut Said,
    table: &str,
    names: &[String],
    duplicates: i64,
    example: &str,
    remedy: &str,
) {
    said.sdm(format!(
        "{table}.{} identifies a row {} time(s) it has already been used, e.g. {example}. Those rows are indistinguishable to the model.{remedy}",
        names.join("."),
        thousands(duplicates)
    ));
}

pub(crate) fn warn_a_key_has_no_value(said: &mut Said, table: &str, column: &str, missing: i64) {
    said.sdm(format!(
        "{table}.{column} identifies a row, but {} row(s) have no value for it. Those rows cannot be scored and the model will not see them; fill or drop them, or name a different primary key.",
        thousands(missing)
    ));
}

pub(crate) fn warn_a_key_was_declined(
    said: &mut Said,
    table: &str,
    column: &str,
    duplicates: i64,
    rows: i64,
    example: &str,
    remedy: &str,
) {
    let t = str_repr(table);
    said.sdm(format!(
        "{table}.{column} is named like a key and another table names it too, but {} of {} rows repeat a value already used, e.g. {example}, so it does not identify a row and {t} gets no primary key. Nothing can link to a table without one: every relationship pointing at {t} is dropped, so nothing is gathered through it, and a question about another table reaches it only where {t} names that table itself. {remedy}",
        thousands(duplicates),
        thousands(rows)
    ));
}

/// What one pass over a key's columns found: `_KeyQuality`.
pub struct KeyQuality {
    pub duplicates: i64,
    pub nulls: i64,
    /// `repr` of the first repeated value, or `None`.
    pub example: String,
}

/// The value `series.iloc[0]` of a row of `columns` gives, as `tuple(row)`
/// writes each: Python numbers where the columns share a numeric dtype,
/// otherwise the cells themselves.
pub(crate) fn row_tuple(columns: &[&frame::Column], row: usize) -> String {
    let all_int = columns
        .iter()
        .all(|c| matches!(c.dtype, PdType::Int { .. }));
    let all_number = columns
        .iter()
        .all(|c| matches!(c.dtype, PdType::Int { .. } | PdType::Float { .. }));
    let parts: Vec<String> = columns
        .iter()
        .map(|c| {
            let cell = &c.cells[row];
            let inner = match cell {
                PyVal::Numpy(_, inner) => inner.as_ref().clone(),
                other => other.clone(),
            };
            if all_int {
                inner.repr()
            } else if all_number {
                match inner {
                    PyVal::Int(n) => py::float_repr(n as f64),
                    other => other.repr(),
                }
            } else {
                cell.repr()
            }
        })
        .collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// `_key_quality(frame, names)`.
pub fn key_quality(data: &Frame, names: &[String]) -> KeyQuality {
    let columns: Vec<&frame::Column> = names.iter().filter_map(|n| data.column(n)).collect();
    if columns.len() != names.len() || columns.is_empty() {
        return KeyQuality {
            duplicates: 0,
            nulls: 0,
            example: "None".to_string(),
        };
    }
    let mut nulls = 0;
    let mut seen = std::collections::HashSet::new();
    let mut duplicates = 0;
    let mut example: Option<String> = None;
    for row in 0..data.rows {
        if columns.iter().any(|c| frame::missing(&c.cells[row])) {
            nulls += 1;
            continue;
        }
        let key: Vec<String> = columns.iter().map(|c| c.cells[row].eq_key()).collect();
        if !seen.insert(key) {
            duplicates += 1;
            if example.is_none() {
                example = Some(if columns.len() == 1 {
                    columns[0].cells[row].repr()
                } else {
                    row_tuple(&columns, row)
                });
            }
        }
    }
    KeyQuality {
        duplicates,
        nulls,
        example: example.unwrap_or_else(|| "None".to_string()),
    }
}

pub(crate) fn check_name(name: &str) -> R<()> {
    if name.is_empty() {
        return Err(Refusal::invalid(format!(
            "The name {} must be a non-empty string.",
            str_repr(name)
        )));
    }
    let bad: Vec<String> = ['`', '\n', '\r', '.']
        .iter()
        .filter(|c| name.contains(**c))
        .map(|c| c.to_string())
        .collect();
    if !bad.is_empty() {
        return Err(Refusal::invalid(format!(
            "The name {} contains one or more invalid characters: {}.",
            str_repr(name),
            py::str_list_repr(&bad)
        )));
    }
    Ok(())
}

impl Table {
    pub fn col(&self, name: &str) -> Option<&Col> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub(crate) fn col_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    pub(crate) fn sorted_columns(&self) -> String {
        py::str_list_repr(&sorted_names(self.columns.iter().map(|c| &c.name)))
    }

    pub(crate) fn check_column(&self, name: &str) -> R<()> {
        if self.col(name).is_none() {
            return Err(Refusal::invalid(format!(
                "table {} has no column {}; it has {}",
                str_repr(&self.name),
                str_repr(name),
                self.sorted_columns()
            )));
        }
        Ok(())
    }

    fn rows(&self) -> Option<&Frame> {
        self.data.as_deref()
    }

    /// `_could_hold_a_key`.
    fn could_hold_a_key(&self, name: &str) -> bool {
        let Some(column) = self.col(name) else {
            return false;
        };
        if Some(name) == self.time_column.as_deref() {
            return false;
        }
        if !supports_dtype("ID", &column.dtype) {
            return false;
        }
        match self.rows().and_then(|f| f.column(name)) {
            None => true,
            Some(c) => !c.any_missing(),
        }
    }

    /// `_check_key`.
    pub(crate) fn check_key(&self, names: &[String]) -> R<()> {
        if names.is_empty() {
            return Err(Refusal::invalid(format!(
                "table {} was given an empty primary key",
                str_repr(&self.name)
            )));
        }
        for name in names {
            self.check_column(name)?;
            let column = self.col(name).expect("checked");
            if !supports_dtype("ID", &column.dtype) {
                return Err(Refusal::invalid(format!(
                    "column {} holds {}, which cannot identify a row, so it cannot be a primary key",
                    str_repr(name),
                    str_repr(&column.dtype)
                )));
            }
            if Some(name.as_str()) == self.time_column.as_deref() {
                return Err(Refusal::invalid(format!(
                    "column {} is already the time column of table {}, so it cannot also be its primary key",
                    str_repr(name),
                    str_repr(&self.name)
                )));
            }
        }
        let Some(data) = self.rows() else {
            return Ok(());
        };
        if data.rows == 0 || data.columns.is_empty() {
            return Ok(());
        }
        for name in names {
            if let Some(column) = data.column(name)
                && let PdType::Other { name: physical, .. } = &column.dtype
                && ["duration[", "timedelta64[", "time32[", "time64["]
                    .iter()
                    .any(|prefix| physical.starts_with(prefix))
                && column
                    .cells
                    .iter()
                    .any(|cell| matches!(cell, PyVal::Opaque { repr, .. } if repr.is_empty()))
            {
                return Err(Refusal::unsupported(format!(
                    "Primary key {}.{} has unsupported physical type {}. Cast it to numbers or text before building the schema.",
                    self.name,
                    name,
                    str_repr(physical)
                )));
            }
        }
        let quality = key_quality(data, names);
        if quality.nulls > 0 {
            let empty: Vec<String> = names
                .iter()
                .filter(|n| data.column(n).is_some_and(frame::Column::any_missing))
                .cloned()
                .collect();
            return Err(Refusal::invalid(a_key_missing_values(
                &self.name,
                names,
                quality.nulls,
                Some(&empty),
            )));
        }
        if too_many_repeats(quality.duplicates, data.rows) {
            return Err(Refusal::invalid(not_unique_enough(
                &self.name,
                names,
                quality.duplicates,
                data.rows,
                &quality.example,
            )));
        }
        Ok(())
    }

    /// `_adopt_key`.
    pub(crate) fn adopt_key(&mut self, names: &[String], said: &mut Said) {
        for name in names {
            if let Some(i) = self.col_index(name) {
                self.columns[i].mark_as_key();
            }
        }
        let Some(data) = self.data.clone() else {
            self.duplicate_keys = None;
            return;
        };
        if data.rows == 0 || data.columns.is_empty() {
            return;
        }
        let quality = key_quality(&data, names);
        self.duplicate_keys = Some(quality.duplicates);
        if quality.duplicates > 0 {
            let distinguishing: Vec<String> = self
                .columns
                .iter()
                .map(|c| c.name.clone())
                .filter(|other| !names.contains(other))
                .filter(|other| {
                    let mut with: Vec<String> = names.to_vec();
                    with.push(other.clone());
                    key_quality(&data, &with).duplicates == 0
                })
                .collect();
            let usable: Vec<String> = distinguishing
                .iter()
                .filter(|o| self.could_hold_a_key(o))
                .cloned()
                .collect();
            let joined = names.join(".");
            let remedy = if !usable.is_empty() {
                let listed: Vec<String> = usable.iter().map(|o| str_repr(o)).collect();
                format!(
                    " {} is unique together with {}, which is what a composite key looks like; pass that as the primary key if so.",
                    str_repr(&joined),
                    listed.join(", ")
                )
            } else if !distinguishing.is_empty() {
                format!(
                    " They differ only by {}, which a primary key cannot be made of, so there is no composite key to name here: keep one row per {} if the model should see one, or name a different primary key.",
                    and_list(&distinguishing),
                    str_repr(&joined)
                )
            } else {
                NAME_ANOTHER_KEY.to_string()
            };
            warn_a_key_repeats(
                said,
                &self.name,
                names,
                quality.duplicates,
                &quality.example,
                &remedy,
            );
        }
    }

    /// `_release_as_key` of column `i`, outside a schema.
    pub(crate) fn release(&mut self, i: usize, said: &mut Said) -> R<()> {
        let Some(restored) = self.columns[i].before_key.take() else {
            return Ok(());
        };
        self.columns[i].stype = restored.clone();
        if restored == "timestamp" {
            self.adopt_as_time_column(i);
        }
        let _ = said;
        Ok(())
    }

    /// `_adopt_as_time_column` of column `i`.
    pub(crate) fn adopt_as_time_column(&mut self, i: usize) {
        if self.time_column.is_some() {
            return;
        }
        let name = self.columns[i].name.clone();
        if key_names(&self.primary_key).contains(&name) {
            return;
        }
        self.time_column = Some(name);
    }

    /// `column.stype = value`, for a table held on its own.
    pub(crate) fn set_stype_alone(&mut self, i: usize, value: &str) -> R<()> {
        self.set_stype_fields(i, value)?;
        if value == "ID" {
            self.columns[i].before_key = None;
            return Ok(());
        }
        let name = self.columns[i].name.clone();
        if key_names(&self.primary_key).contains(&name) {
            self.columns[i].mark_as_key();
        }
        Ok(())
    }

    /// What `Column.__setattr__('stype', value)` does before it asks the
    /// schema which columns are keys.
    pub(crate) fn set_stype_fields(&mut self, i: usize, value: &str) -> R<()> {
        check_stype(value)?;
        let column = &self.columns[i];
        if !supports_dtype(value, &column.dtype) {
            return Err(Refusal::invalid(format!(
                "column {} holds {}, which cannot mean {}",
                str_repr(&column.name),
                str_repr(&column.dtype),
                str_repr(value)
            )));
        }
        let column = &mut self.columns[i];
        column.stype_declared = true;
        column.stype = value.to_string();
        if value == "timestamp" {
            self.adopt_as_time_column(i);
        }
        Ok(())
    }

    /// `table.primary_key = value`, for a table held on its own.
    pub fn set_primary_key_alone(&mut self, value: Option<Key>, said: &mut Said) -> R<()> {
        match value {
            None => {
                self.duplicate_keys = Some(0);
                self.primary_key = None;
                self.release_unkeyed(said)
            }
            Some(key) => {
                let names = key.names();
                self.check_key(&names)?;
                self.primary_key = Some(key);
                self.adopt_key(&names, said);
                self.release_unkeyed(said)
            }
        }
    }

    pub(crate) fn release_unkeyed(&mut self, said: &mut Said) -> R<()> {
        let named = key_names(&self.primary_key);
        for i in 0..self.columns.len() {
            if !named.contains(&self.columns[i].name) {
                self.release(i, said)?;
            }
        }
        Ok(())
    }

    /// `_check_time_column`, then the assignment, for a table on its own.
    pub fn set_time_column_alone(&mut self, value: Option<&str>) -> R<()> {
        let Some(name) = value else {
            self.time_column = None;
            return Ok(());
        };
        self.check_column(name)?;
        if key_names(&self.primary_key).iter().any(|k| k == name) {
            return Err(Refusal::invalid(format!(
                "column {} is already the primary key of table {}, so it cannot also be its time column",
                str_repr(name),
                str_repr(&self.name)
            )));
        }
        let i = self.col_index(name).expect("checked");
        self.set_stype_alone(i, "timestamp")?;
        self.time_column = Some(name.to_string());
        Ok(())
    }
}

/// `_as_key_tuple` of a key as JSON gives it: a name, or a list of them.
pub fn key_of_json(value: &Value) -> R<Option<Key>> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(Key::One(s.clone()))),
        Value::Array(items) => {
            let mut names = Vec::new();
            for item in items {
                match item {
                    Value::String(s) => names.push(s.clone()),
                    _ => {
                        return Err(Refusal::invalid(format!(
                            "a primary key names a column or a sequence of them, got {}",
                            json_repr(value)
                        )));
                    }
                }
            }
            Ok(Some(Key::Many(names)))
        }
        other => Err(Refusal::invalid(format!(
            "a primary key names a column or a sequence of them, got {}",
            json_repr(other)
        ))),
    }
}

/// `repr` of a value JSON carries.
pub fn json_repr(value: &Value) -> String {
    PyVal::decode(value).map_or_else(|_| value.to_string(), |v| v.repr())
}

// --- Inference ------------------------------------------------------------------

/// `infer_tfm_dtype(series)`: the wire dtype, or `""` for no type at all.
pub fn infer_tfm_dtype(column: &frame::Column) -> String {
    match &column.dtype {
        PdType::Bool => "bool".to_string(),
        PdType::Datetime => "timestamp[us]".to_string(),
        PdType::Int { name, .. } => {
            if !name.to_lowercase().contains("uint64") {
                return "int64".to_string();
            }
            whole_numbers_dtype(&column.cells)
        }
        PdType::Float { bits, .. } => if *bits > 32 { "float64" } else { "float32" }.to_string(),
        PdType::Decimal {
            precision, scale, ..
        } => {
            if *scale > 0 {
                "float64".to_string()
            } else if *precision <= 18 {
                "int64".to_string()
            } else {
                decimal_dtype(&column.cells)
            }
        }
        PdType::Object { content } => match content.as_str() {
            "empty" => String::new(),
            "decimal" => decimal_dtype(&column.cells),
            "integer" => whole_numbers_dtype(&column.cells),
            "floating" | "mixed-integer-float" => "float64".to_string(),
            "boolean" => "bool".to_string(),
            "date" | "datetime" | "datetime64" => "timestamp[us]".to_string(),
            _ => "string".to_string(),
        },
        PdType::Text { .. } | PdType::Other { .. } => "string".to_string(),
    }
}

pub(crate) fn int_value(cell: &PyVal) -> Option<Option<i128>> {
    match cell {
        PyVal::Int(n) => Some(Some(*n)),
        PyVal::BigInt(_) => Some(None),
        PyVal::Bool(b) => Some(Some(i128::from(*b))),
        PyVal::Numpy(_, inner) => int_value(inner),
        PyVal::Float(x) if x.is_finite() => Some(Some(*x as i128)),
        PyVal::Decimal(text) => {
            let key = crate::pyval::PyVal::Decimal(text.clone()).eq_key();
            key.strip_prefix("n:").map(|d| d.parse::<i128>().ok())
        }
        _ => None,
    }
}

/// `_whole_numbers_dtype`, reading the values.
pub(crate) fn whole_numbers_dtype(cells: &[PyVal]) -> String {
    for cell in cells {
        if frame::missing(cell) {
            continue;
        }
        match int_value(cell) {
            Some(Some(n)) if i128::from(i64::MIN) <= n && n <= i128::from(i64::MAX) => {}
            _ => return "string".to_string(),
        }
    }
    "int64".to_string()
}

/// A `Decimal`'s digits and exponent, as `as_tuple()` holds them.
pub(crate) fn decimal_parts(text: &str) -> Option<(bool, String, i64)> {
    let t = text.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.trim_start_matches('+')),
    };
    let lower = body.to_ascii_lowercase();
    let (mantissa, exp) = match lower.split_once('e') {
        Some((m, e)) => (m.to_string(), e.parse::<i64>().ok()?),
        None => (lower.clone(), 0),
    };
    let (int, frac) = mantissa.split_once('.').unwrap_or((mantissa.as_str(), ""));
    if !int.chars().chain(frac.chars()).all(|c| c.is_ascii_digit())
        || (int.is_empty() && frac.is_empty())
    {
        return None;
    }
    let digits = format!("{int}{frac}").trim_start_matches('0').to_string();
    Some((
        neg,
        if digits.is_empty() {
            "0".to_string()
        } else {
            digits
        },
        exp - frac.len() as i64,
    ))
}

/// `_decimal_dtype` over the values.
pub(crate) fn decimal_dtype(cells: &[PyVal]) -> String {
    let mut integral = true;
    for cell in cells {
        if frame::missing(cell) {
            continue;
        }
        let PyVal::Decimal(text) = cell else {
            return "float64".to_string();
        };
        let Some((_, digits, exponent)) = decimal_parts(text) else {
            return "float64".to_string();
        };
        if exponent < 0 {
            // Whole only if the digits past the point are zeros.
            let frac_len = (-exponent) as usize;
            let tail = if digits.len() >= frac_len {
                &digits[digits.len() - frac_len..]
            } else {
                digits.as_str()
            };
            if tail.chars().any(|c| c != '0') {
                return "float64".to_string();
            }
        }
        let adjusted = digits.len() as i64 - 1 + exponent;
        if adjusted >= 18 {
            let in_range = match int_value(cell) {
                Some(Some(n)) => i128::from(i64::MIN) <= n && n <= i128::from(i64::MAX),
                _ => false,
            };
            if !in_range {
                integral = false;
            }
        }
    }
    if integral {
        "int64".to_string()
    } else {
        "string".to_string()
    }
}

/// `require_scalar_cells(series, described)`.
pub fn require_scalar_cells(column: &frame::Column, described: &str) -> R<()> {
    let scalar_only = match &column.dtype {
        PdType::Object { .. } => false,
        PdType::Other { scalar_only, .. } => *scalar_only,
        _ => true,
    };
    if scalar_only {
        return Ok(());
    }
    for cell in &column.cells {
        let (kind, type_name) = match cell {
            PyVal::Bytes(_) => ("bytes", "bytes"),
            PyVal::List(_) => ("container", "list"),
            PyVal::Tuple(_) => ("container", "tuple"),
            PyVal::Dict(_) => ("mapping", "dict"),
            PyVal::Opaque {
                kind, type_name, ..
            } if kind == "mapping" || kind == "container" => (kind.as_str(), type_name.as_str()),
            _ => continue,
        };
        if kind == "bytes" {
            return Err(Refusal::unsupported(format!(
                "{described} holds raw bytes ({type_name}), and the contract has no cell that carries them. It is already one value per row, so what it needs is a reading rather than a flattening: decode the column to text, or leave it out of the schema"
            )));
        }
        let shape = if kind == "mapping" {
            "a mapping"
        } else {
            "a container"
        };
        return Err(Refusal::unsupported(format!(
            "{described} holds {shape} ({type_name}), and a nested column is not something this client can send. Multi-label targets are not supported by these models; for a feature, give one value per row"
        )));
    }
    Ok(())
}

/// `_column(series, name)`.
pub(crate) fn infer_column(column: &frame::Column) -> R<Col> {
    if let PdType::Other {
        infer_dtype_error: Some(message),
        ..
    } = &column.dtype
    {
        return Err(Refusal::class("NotImplementedError", message));
    }
    let inferred = infer_tfm_dtype(column);
    let dtype = if inferred.is_empty() {
        "string".to_string()
    } else {
        inferred.clone()
    };
    let stype = if dtype.starts_with("timestamp") {
        "timestamp".to_string()
    } else if matches!(dtype.as_str(), "int32" | "int64" | "float32" | "float64") {
        "numerical".to_string()
    } else if dtype == "bool" || column.nunique()? <= MAX_CATEGORICAL_CARDINALITY {
        "categorical".to_string()
    } else {
        "text".to_string()
    };
    let mut col = Col::new(&column.name, &dtype, &stype)?;
    if inferred.is_empty() {
        col.untyped = true;
    }
    Ok(col)
}

pub(crate) fn infer_primary_key(data: &Frame, columns: &[Col]) -> R<Option<String>> {
    if data.rows == 0 {
        return Ok(None);
    }
    for column in columns {
        let lowered = column.name.to_lowercase();
        if !KEY_SUFFIXES.iter().any(|s| lowered.ends_with(s)) {
            continue;
        }
        if !supports_dtype("ID", &column.dtype) {
            continue;
        }
        let distinct = data
            .column(&column.name)
            .map_or(Ok(0), frame::Column::nunique)?;
        if distinct as f64 >= data.rows as f64 * KEY_UNIQUENESS {
            return Ok(Some(column.name.clone()));
        }
    }
    Ok(None)
}

/// `_infer_table(name, frame)`.
pub fn infer_table(name: &str, data: Arc<Frame>, said: &mut Said) -> R<Table> {
    let mut seen = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for c in &data.columns {
        if !seen.insert(c.name.clone()) {
            duplicates.insert(c.name.clone());
        }
    }
    if !duplicates.is_empty() {
        let d: Vec<String> = duplicates.into_iter().collect();
        return Err(Refusal::invalid(format!(
            "table {} has duplicate column name(s) {}",
            str_repr(name),
            py::str_list_repr(&d)
        )));
    }
    let non_strings: Vec<&str> = data
        .columns
        .iter()
        .filter_map(|c| c.name_repr.as_deref())
        .collect();
    if !non_strings.is_empty() {
        return Err(Refusal::invalid(format!(
            "table {} column names must be strings; got [{}]",
            str_repr(name),
            non_strings.join(", ")
        )));
    }
    for c in &data.columns {
        require_scalar_cells(c, &format!("{name}.{}", c.name))?;
    }
    let mut columns = Vec::new();
    for c in &data.columns {
        columns.push(infer_column(c)?);
    }
    check_name(name)?;
    for c in &data.columns {
        check_name(&c.name)?;
    }
    let mut table = Table {
        name: name.to_string(),
        data: Some(data.clone()),
        source_name: None,
        columns,
        primary_key: None,
        time_column: None,
        duplicate_keys: Some(0),
    };
    let inferred = infer_primary_key(&data, &table.columns)?;
    if let Some(key) = &inferred {
        let column = data.column(key).expect("a column of the frame");
        if column.any_missing() {
            warn_a_key_has_no_value(said, name, key, column.count_missing() as i64);
            table.primary_key = Some(Key::One(key.clone()));
            table.adopt_key(std::slice::from_ref(key), said);
            return Ok(table);
        }
    }
    table.set_primary_key_alone(inferred.map(Key::One), said)?;
    Ok(table)
}

// --- The schema ---------------------------------------------------------------------

pub(crate) fn require_table<'a>(tables: &'a [Table], name: &str) -> R<&'a Table> {
    tables.iter().find(|t| t.name == name).ok_or_else(|| {
        Refusal::invalid(format!(
            "no table {} in this schema; it has {}",
            str_repr(name),
            py::str_list_repr(&sorted_names(tables.iter().map(|t| &t.name)))
        ))
    })
}

pub(crate) fn require_column(tables: &[Table], table: &str, column: &str) -> R<()> {
    let t = require_table(tables, table)?;
    if t.col(column).is_none() {
        return Err(Refusal::invalid(format!(
            "table {} has no column {}; it has {}",
            str_repr(table),
            str_repr(column),
            t.sorted_columns()
        )));
    }
    Ok(())
}

pub(crate) fn table_index(tables: &[Table], name: &str) -> Option<usize> {
    tables.iter().position(|t| t.name == name)
}

/// `_require_joinable`.
pub(crate) fn require_joinable(tables: &[Table], link: &Rel) -> R<()> {
    let target = require_table(tables, &link.target_table)?;
    let keys = key_names(&target.primary_key);
    if keys.len() > 1 {
        return Err(Refusal::invalid(format!(
            "{} points at {}, whose primary key is {}. An edge joins one column to one key, so a table with a composite key cannot be the target of a relationship",
            link.text(),
            str_repr(&link.target_table),
            py::str_list_repr(&keys)
        )));
    }
    if !keys.contains(&link.target_column) {
        let named = match &target.primary_key {
            Some(k) => k.repr(),
            None => "no primary key".to_string(),
        };
        return Err(Refusal::invalid(format!(
            "{} does not join to a primary key: table {} has {named}. A relationship points at the key that identifies a row, so set that key first if it is not the one inference found",
            link.text(),
            str_repr(&link.target_table)
        )));
    }
    let source = require_table(tables, &link.source_table)?;
    let source_dtype = source
        .col(&link.source_column)
        .map(|c| c.dtype.clone())
        .unwrap_or_default();
    let target_dtype = target
        .col(&link.target_column)
        .map(|c| c.dtype.clone())
        .unwrap_or_default();
    if join_family(&source_dtype) != join_family(&target_dtype) {
        return Err(Refusal::invalid(format!(
            "{} joins {} to {}, which cannot match. Give both columns the same kind of value",
            link.text(),
            str_repr(&source_dtype),
            str_repr(&target_dtype)
        )));
    }
    Ok(())
}

pub(crate) fn is_joinable(tables: &[Table], link: &Rel) -> bool {
    require_joinable(tables, link).is_ok()
}

/// `_check_relationships`.
pub(crate) fn check_relationships(tables: &[Table], links: &[Rel]) -> R<()> {
    for link in links {
        require_column(tables, &link.source_table, &link.source_column)?;
        require_column(tables, &link.target_table, &link.target_column)?;
        require_joinable(tables, link)?;
    }
    Ok(())
}

pub(crate) fn each_link_once(links: &[Rel]) -> Vec<Rel> {
    let mut seen = std::collections::HashSet::new();
    links
        .iter()
        .filter(|l| seen.insert((*l).clone()))
        .cloned()
        .collect()
}

pub(crate) fn warn_of_a_pair_joined_both_ways(links: &[Rel], said: &mut Said) {
    let pairs: BTreeSet<(String, String)> = links
        .iter()
        .map(|l| (l.source_table.clone(), l.target_table.clone()))
        .collect();
    let mut done: BTreeSet<(String, String)> = BTreeSet::new();
    for (source, target) in &pairs {
        if source == target {
            continue;
        }
        let back = (target.clone(), source.clone());
        if !pairs.contains(&back) || done.contains(&back) {
            continue;
        }
        done.insert((source.clone(), target.clone()));
        said.sdm(format!(
            "{} and {} name each other, and a relationship is crossed in both directions, so this is two ways over one pair and an aggregation between them is refused for not having a single one. Declare whichever of the two a question is about and leave the other out.",
            str_repr(source),
            str_repr(target)
        ));
    }
}

/// `_warn_two_ways_to_reach_one_table`.
pub fn warn_two_ways_to_reach_one_table(links: &[Rel], said: &mut Said) {
    let mut ways: BTreeMap<(String, String), Vec<String>> = BTreeMap::new();
    for link in links {
        let columns = ways
            .entry((link.source_table.clone(), link.target_table.clone()))
            .or_default();
        if !columns.contains(&link.source_column) {
            columns.push(link.source_column.clone());
        }
    }
    warn_of_a_pair_joined_both_ways(links, said);
    for ((source, target), columns) in &ways {
        if columns.len() < 2 {
            continue;
        }
        let mut sorted = columns.clone();
        sorted.sort();
        let kept = sorted[0].clone();
        let rest = sorted[1..].to_vec();
        let (s, t) = (str_repr(source), str_repr(target));
        said.sdm(format!(
            "{s} reaches {t} through {}, and an aggregation across the two tables has to be resolved to one of them, so a query that crosses this pair is refused until it is. They are all kept because they are all real, and which one a question means is not something this can decide. To keep {}, drop the others: schema.relationships = [link for link in schema.relationships if not (link.source_table == {s} and link.target_table == {t} and link.source_column in {})].",
            and_list(&sorted),
            str_repr(&kept),
            py::str_list_repr(&rest)
        ));
    }
}

impl Schema {
    /// `Schema(tables, relationships)`, warnings of the relationships'
    /// settling going to `said`.
    pub fn new(tables: &[Table], links: &[Rel], said: &mut Said) -> R<Schema> {
        let mut schema = Schema {
            tables: tables.to_vec(),
            relationships: Vec::new(),
        };
        schema.set_relationships(links, said)?;
        Ok(schema)
    }

    /// `schema.relationships = links`.
    pub fn set_relationships(&mut self, links: &[Rel], said: &mut Said) -> R<()> {
        check_relationships(&self.tables, links)?;
        let settled = each_link_once(links);
        warn_two_ways_to_reach_one_table(&settled, said);
        self.relationships = settled;
        self.mark_key_columns(said)
    }

    /// `_mark_key_columns`.
    pub fn mark_key_columns(&mut self, said: &mut Said) -> R<()> {
        let mut keys: Vec<Vec<String>> = self
            .tables
            .iter()
            .map(|t| {
                key_names(&t.primary_key)
                    .into_iter()
                    .filter(|c| t.col(c).is_some())
                    .collect()
            })
            .collect();
        for link in &self.relationships {
            if let Some(i) = table_index(&self.tables, &link.source_table)
                && self.tables[i].col(&link.source_column).is_some()
                && !keys[i].contains(&link.source_column)
            {
                keys[i].push(link.source_column.clone());
            }
        }
        for (ti, table) in self.tables.iter_mut().enumerate() {
            for ci in 0..table.columns.len() {
                if keys[ti].contains(&table.columns[ci].name) {
                    table.columns[ci].mark_as_key();
                } else {
                    table.release(ci, said)?;
                }
            }
        }
        Ok(())
    }

    /// `check_consistent()`: the keys re-derived, then every column's
    /// storage against its meaning, and every relationship.
    pub fn check_consistent(&mut self, said: &mut Said) -> R<()> {
        self.mark_key_columns(said)?;
        for table in &self.tables {
            for column in &table.columns {
                if !supports_dtype(&column.stype, &column.dtype) {
                    return Err(Refusal::invalid(format!(
                        "column {} holds {}, which cannot mean {}",
                        str_repr(&column.name),
                        str_repr(&column.dtype),
                        str_repr(&column.stype)
                    )));
                }
            }
        }
        check_relationships(&self.tables, &self.relationships)
    }

    pub(crate) fn table_mut(&mut self, name: &str) -> R<usize> {
        table_index(&self.tables, name).ok_or_else(|| {
            Refusal::invalid(format!(
                "no table {} in this schema; it has {}",
                str_repr(name),
                py::str_list_repr(&sorted_names(self.tables.iter().map(|t| &t.name)))
            ))
        })
    }

    /// `setattr(schema[table], attribute, value)`: a table's `primary_key`
    /// or `time_column`, set within this schema.
    pub fn edit_table(
        &mut self,
        table: &str,
        attribute: &str,
        value: &Value,
        said: &mut Said,
    ) -> R<()> {
        let ti = self.table_mut(table)?;
        match attribute {
            "primary_key" => {
                let key = key_of_json(value)?;
                let t = &mut self.tables[ti];
                match key {
                    None => {
                        t.duplicate_keys = Some(0);
                        t.primary_key = None;
                    }
                    Some(key) => {
                        let names = key.names();
                        t.check_key(&names)?;
                        t.primary_key = Some(key);
                        t.adopt_key(&names, said);
                    }
                }
                self.mark_key_columns(said)
            }
            "time_column" => self.tables[ti].set_time_column_alone(value.as_str()),
            other => Err(Refusal::invalid(format!(
                "a table's {} is not one this core sets",
                str_repr(other)
            ))),
        }
    }

    /// `schema.relationships.append(link)`.
    pub fn append_link(&mut self, link: Rel, said: &mut Said) -> R<()> {
        let mut links = self.relationships.clone();
        links.push(link);
        self.set_relationships(&links, said)
    }

    /// `setattr(schema[table][column], attribute, value)`.
    pub fn edit(
        &mut self,
        table: &str,
        column: &str,
        attribute: &str,
        value: &Value,
        said: &mut Said,
    ) -> R<()> {
        let ti = self.table_mut(table)?;
        let t = &self.tables[ti];
        let Some(ci) = t.col_index(column) else {
            return Err(Refusal::invalid(format!(
                "table {} has no column {}; it has {}",
                str_repr(&t.name),
                str_repr(column),
                t.sorted_columns()
            )));
        };
        match attribute {
            "untyped" => Err(Refusal::invalid(
                "'untyped' is inferred from the values and cannot be set; assign the dtype you want instead",
            )),
            "stype" => {
                let Some(text) = value.as_str() else {
                    return Err(Refusal::invalid(format!(
                        "{} is not a semantic type; use one of {}",
                        json_repr(value),
                        py::str_list_repr(&{
                            let mut s: Vec<&str> = SEMANTIC_TYPES.to_vec();
                            s.sort_unstable();
                            s
                        })
                    )));
                };
                self.tables[ti].set_stype_fields(ci, text)?;
                if text == "ID" {
                    self.tables[ti].columns[ci].before_key = None;
                    return Ok(());
                }
                self.mark_key_columns(said)
            }
            "dtype" => {
                let column = &mut self.tables[ti].columns[ci];
                if column.untyped {
                    column.untyped = false;
                }
                column.dtype = value
                    .as_str()
                    .map_or_else(|| json_repr(value), str::to_string);
                Ok(())
            }
            "name" => {
                self.tables[ti].columns[ci].name = value
                    .as_str()
                    .map_or_else(|| json_repr(value), str::to_string);
                Ok(())
            }
            "stype_declared" => {
                self.tables[ti].columns[ci].stype_declared = value.as_bool().unwrap_or(false);
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

pub(crate) fn apply_keys_and_times(
    built: &mut [Table],
    primary_keys: &[(String, KeyArg)],
    time_columns: &[(String, Value)],
    said: &mut Said,
) -> R<()> {
    for (name, key) in primary_keys {
        require_table(built, name)?;
        // `_as_key_tuple(key)`: a value that names no columns is refused.
        let key = match key {
            Ok(key) => key,
            Err(shown) => {
                return Err(Refusal::invalid(format!(
                    "a primary key names a column or a sequence of them, got {shown}"
                )));
            }
        };
        let names = key_names(key);
        for column in &names {
            require_column(built, name, column)?;
        }
        let i = table_index(built, name).expect("checked");
        built[i].set_primary_key_alone(key.clone(), said)?;
    }
    for (name, column) in time_columns {
        require_table(built, name)?;
        if column.is_null() {
            let index = table_index(built, name).expect("checked");
            built[index].set_time_column_alone(None)?;
            continue;
        }
        let column = column
            .as_str()
            .map_or_else(|| json_repr(column), str::to_string);
        require_column(built, name, &column)?;
        let i = table_index(built, name).expect("checked");
        built[i].set_time_column_alone(Some(&column))?;
    }
    Ok(())
}

/// `_infer_relationships`.
pub fn infer_relationships(tables: &[Table]) -> Vec<Rel> {
    let mut ambiguous: BTreeSet<String> = BTreeSet::new();
    let mut keys: BTreeMap<String, String> = BTreeMap::new();
    for t in tables {
        if let Some(Key::One(key)) = &t.primary_key {
            if keys.contains_key(key) {
                ambiguous.insert(key.clone());
            }
            keys.insert(key.clone(), t.name.clone());
        }
    }
    let mut links = Vec::new();
    for t in tables {
        for c in &t.columns {
            let Some(target) = keys.get(&c.name) else {
                continue;
            };
            if *target == t.name || ambiguous.contains(&c.name) {
                continue;
            }
            links.push(Rel::new(&t.name, &c.name, target, &c.name));
        }
    }
    links
}

pub(crate) fn warn_of_a_link_that_cannot_join(tables: &[Table], dropped: &[Rel], said: &mut Said) {
    for link in dropped {
        let source = require_table(tables, &link.source_table)
            .ok()
            .and_then(|t| t.col(&link.source_column));
        let Ok(target_table) = require_table(tables, &link.target_table) else {
            continue;
        };
        let target = target_table.col(&link.target_column);
        let key = key_names(&target_table.primary_key);
        if key != vec![link.target_column.clone()] {
            let has = if key.is_empty() {
                format!("{} has no primary key", str_repr(&link.target_table))
            } else {
                format!(
                    "the primary key of {} is {}",
                    str_repr(&link.target_table),
                    and_list(&key)
                )
            };
            said.sdm(format!(
                "{}.{} refers to {}.{}, but {has}. A relationship joins to a primary key, so none was made from it. Pass primary_keys={{{}: {}}} if {} identifies a row of {}.",
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
                str_repr(&link.target_table),
                str_repr(&link.target_column),
                str_repr(&link.target_column),
                str_repr(&link.target_table)
            ));
            continue;
        }
        said.sdm(format!(
            "{}.{} names the key of {} and holds {} where that key holds {}, which cannot match, so no relationship was made from it. Store the two as one type if they are the same identifier.",
            link.source_table,
            link.source_column,
            str_repr(&link.target_table),
            str_repr(&source.map(|c| c.dtype.clone()).unwrap_or_default()),
            str_repr(&target.map(|c| c.dtype.clone()).unwrap_or_default())
        ));
    }
}

/// `_joinable_inferred_links`.
pub fn joinable_inferred_links(tables: &[Table], inferred: &[Rel], said: &mut Said) -> Vec<Rel> {
    let unjoinable: Vec<Rel> = inferred
        .iter()
        .filter(|l| !is_joinable(tables, l))
        .cloned()
        .collect();
    warn_of_a_link_that_cannot_join(tables, &unjoinable, said);
    inferred
        .iter()
        .filter(|l| is_joinable(tables, l))
        .cloned()
        .collect()
}

/// `_checked_links`.
pub fn checked_links(tables: &[Table], links: &[Rel]) -> R<Vec<Rel>> {
    for link in links {
        require_column(tables, &link.source_table, &link.source_column)?;
        require_column(tables, &link.target_table, &link.target_column)?;
        require_joinable(tables, link)?;
    }
    Ok(links.to_vec())
}

pub(crate) fn tables_naming_each_column(tables: &[Table]) -> BTreeMap<String, BTreeSet<String>> {
    let mut named: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for t in tables {
        for c in &t.columns {
            named
                .entry(c.name.clone())
                .or_default()
                .insert(t.name.clone());
        }
    }
    named
}

pub(crate) fn keyed_by(tables: &[Table]) -> BTreeSet<String> {
    tables
        .iter()
        .filter_map(|t| match &t.primary_key {
            Some(Key::One(k)) => Some(k.clone()),
            _ => None,
        })
        .collect()
}

/// `_warn_dates_stored_as_text(table)`: the column it spoke of.
pub(crate) fn warn_dates_stored_as_text(
    table: &Table,
    host: &Host,
    said: &mut Said,
) -> Option<String> {
    let data = table.data.as_ref()?;
    if table.time_column.is_some() || data.rows == 0 || data.columns.is_empty() {
        return None;
    }
    for column in &table.columns {
        if column.dtype != "string" || column.stype == "ID" {
            continue;
        }
        let Some(cells) = data.column(&column.name) else {
            continue;
        };
        let present: Vec<Option<String>> = cells
            .cells
            .iter()
            .filter(|c| !frame::missing(c))
            .map(|c| match c {
                PyVal::Str(s) => Some(s.clone()),
                // Not text: no date reading takes a leading NUL.
                other => Some(format!("\u{0}{}", other.str_())),
            })
            .collect();
        if !dates::looks_unambiguous(&present) {
            continue;
        }
        let Some(how) = dates::conversion_that_works(&present, true, false, host) else {
            continue;
        };
        let (n, name) = (str_repr(&column.name), str_repr(&table.name));
        said.sdm(format!(
            "{}.{} holds dates but is stored as text, so it is not the time column and no temporal query can use it. Convert it before inferring: frame[{n}] = pd.to_datetime(frame[{n}]{}), or set schema[{name}][{n}].stype = 'timestamp' afterwards.",
            table.name,
            column.name,
            how.text()
        ));
        return Some(column.name.clone());
    }
    None
}

pub(crate) fn could_identify_a_row(table: &Table, data: &Frame, column: &str, meta: &Col) -> bool {
    if Some(column) == table.time_column.as_deref() {
        return false;
    }
    if meta.dtype != "string" && !meta.dtype.starts_with("int") {
        return false;
    }
    if data.column(column).is_some_and(frame::Column::any_missing) {
        return false;
    }
    key_quality(data, &[column.to_string()]).duplicates == 0
}

pub(crate) fn unlinkable_table_reads(
    name: &str,
    candidates: &[String],
    tables_sharing: usize,
) -> String {
    let many = candidates.len() > 1;
    let shared_by_many = tables_sharing > 1;
    let (subject, verb) = if many { ("they", "are") } else { ("it", "is") };
    let tables = if shared_by_many { "tables" } else { "table" };
    let n = str_repr(name);
    format!(
        "{n} has no primary key, and nothing can link to a table without one: every relationship pointing at {n} is dropped, so nothing is gathered through it, and a question about another table reaches it only where {n} names that table itself. {} {} every row of {n} and {verb} named by {} too, which makes {} either a key or a foreign key that happens not to repeat in the rows read. If {} {n}, say so with primary_keys={{{n}: {}}}; if {subject} instead {} the rows of the {tables} {subject} {verb} shared with, {} the key.",
        and_list(candidates),
        if many { "identify" } else { "identifies" },
        if shared_by_many {
            "other tables"
        } else {
            "another table"
        },
        if many { "each of them" } else { "it" },
        if many {
            "one of them identifies"
        } else {
            "it identifies"
        },
        if many {
            "...".to_string()
        } else {
            str_repr(&candidates[0])
        },
        if many { "name" } else { "names" },
        if shared_by_many {
            "those tables are the ones that need"
        } else {
            "that table is the one that needs"
        }
    )
}

pub(crate) fn warn_table_cannot_be_linked_to(
    tables: &[Table],
    dates: &BTreeMap<String, String>,
    already: &BTreeSet<String>,
    said: &mut Said,
) {
    let named = tables_naming_each_column(tables);
    let keyed = keyed_by(tables);
    for table in tables {
        if table.primary_key.is_some() || already.contains(&table.name) {
            continue;
        }
        let Some(data) = table.data.as_ref() else {
            continue;
        };
        if data.rows == 0 || data.columns.is_empty() {
            continue;
        }
        let candidates: Vec<String> = table
            .columns
            .iter()
            .filter(|meta| {
                could_identify_a_row(table, data, &meta.name, meta)
                    && named.get(&meta.name).map_or(0, BTreeSet::len) > 1
                    && !keyed.contains(&meta.name)
                    && dates.get(&table.name) != Some(&meta.name)
            })
            .map(|m| m.name.clone())
            .collect();
        if candidates.is_empty() {
            continue;
        }
        let elsewhere: BTreeSet<&String> = candidates
            .iter()
            .flat_map(|c| named.get(c).into_iter().flatten())
            .filter(|other| **other != table.name)
            .collect();
        said.sdm(unlinkable_table_reads(
            &table.name,
            &candidates,
            elsewhere.len(),
        ));
    }
}

pub(crate) fn warn_repeats_cost_the_table_its_key(
    tables: &[Table],
    said: &mut Said,
) -> R<BTreeSet<String>> {
    let named = tables_naming_each_column(tables);
    let keyed = keyed_by(tables);
    let mut spoken = BTreeSet::new();
    for table in tables {
        if table.primary_key.is_some() {
            continue;
        }
        let Some(data) = table.data.as_ref() else {
            continue;
        };
        if data.rows == 0 || data.columns.is_empty() {
            continue;
        }
        for meta in &table.columns {
            let lowered = meta.name.to_lowercase();
            if !KEY_SUFFIXES.iter().any(|s| lowered.ends_with(s)) {
                continue;
            }
            if !supports_dtype("ID", &meta.dtype) {
                continue;
            }
            if keyed.contains(&meta.name) {
                continue;
            }
            if !named
                .get(&meta.name)
                .is_some_and(|s| s.iter().any(|t| *t != table.name))
            {
                continue;
            }
            let distinct = data
                .column(&meta.name)
                .map_or(Ok(0), frame::Column::nunique)?;
            if (distinct as f64) < data.rows as f64 * NEARLY_A_KEY {
                continue;
            }
            let quality = key_quality(data, std::slice::from_ref(&meta.name));
            if quality.duplicates == 0 {
                continue;
            }
            let c = str_repr(&meta.name);
            warn_a_key_was_declined(
                said,
                &table.name,
                &meta.name,
                quality.duplicates,
                data.rows as i64,
                &quality.example,
                &format!(
                    "Keep one row per {c}: frame = frame.drop_duplicates({c}), and infer again, or name a column that does identify a row with primary_keys={{{}: ...}}.",
                    str_repr(&table.name)
                ),
            );
            spoken.insert(table.name.clone());
            break;
        }
    }
    Ok(spoken)
}

pub(crate) fn warn_one_key_name_keys_several_tables(tables: &[Table], said: &mut Said) {
    let mut keyed: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for t in tables {
        if let Some(Key::One(k)) = &t.primary_key {
            keyed.entry(k.clone()).or_default().push(t.name.clone());
        }
    }
    for (key, names) in &keyed {
        if names.len() < 2 {
            continue;
        }
        let mut shared = names.clone();
        shared.sort();
        let k = str_repr(key);
        said.sdm(format!(
            "{k} is the primary key of {}, so a column named {k} identifies a row in each of them and names a parent in none. No relationship is inferred through an ambiguous name, so nothing can point at any of those tables: none of them is gathered as another table's related rows, and a question about one of them is answered from that table alone. Say what the graph is: relationships=[Relationship(child, column, {}, {k}), ...], or give each table a key of its own name, so that the column naming it can be matched to it.",
            and_list(&shared),
            str_repr(&shared[0])
        ));
    }
}

pub(crate) fn warn_stype_was_decided_by_the_rows_read(tables: &[Table], said: &mut Said) -> R<()> {
    for table in tables {
        let Some(data) = table.data.as_ref() else {
            continue;
        };
        if data.rows == 0 {
            continue;
        }
        for meta in &table.columns {
            if meta.stype_declared || meta.stype != "categorical" {
                continue;
            }
            let Some(column) = data.column(&meta.name) else {
                continue;
            };
            if meta.dtype != "string" {
                continue;
            }
            let distinct = column.nunique()?;
            let present = column.count();
            if (distinct as f64) < present as f64 * UNSATURATED {
                continue;
            }
            if (distinct as f64) < MAX_CATEGORICAL_CARDINALITY as f64 * WITHIN_REACH {
                continue;
            }
            let (t, c) = (str_repr(&table.name), str_repr(&meta.name));
            said.sdm(format!(
                "{}.{} holds {distinct} distinct values in the {present} rows that have one, so the rows ran out before the values did and the count is a measure of the frame rather than of the column. It is read as categorical here because {distinct} is at most {MAX_CATEGORICAL_CARDINALITY}; on more rows the same column could pass that and be read as text, which the models treat differently. If this frame is a sample, say what the column is with schema[{t}][{c}].stype = 'categorical' (or 'text'), so that it does not change under a larger one.",
                table.name, meta.name
            ));
        }
    }
    Ok(())
}

pub(crate) fn settle_time_choices(
    built: &mut [Table],
    choices: &mut BTreeMap<String, crate::time_columns::Choice>,
    given: &[(String, Value)],
    links: &[Rel],
) -> R<()> {
    for table in built {
        if given.iter().any(|(name, _)| name == &table.name) {
            continue;
        }
        if let Some(choice) = choices.get_mut(&table.name) {
            let settled =
                choice.for_links(links.iter().any(|link| link.target_table == table.name));
            if settled != *choice {
                table.set_time_column_alone(settled.column.as_deref())?;
                *choice = settled;
            }
        }
    }
    Ok(())
}

pub(crate) fn warn_time_choices(
    schema: &Schema,
    choices: &BTreeMap<String, crate::time_columns::Choice>,
    given: &[(String, Value)],
    said: &mut Said,
) {
    if schema.tables.len() < 2 && schema.relationships.is_empty() {
        return;
    }
    let mut lines = Vec::new();
    for (name, choice) in choices {
        if given.iter().any(|(named, _)| named == name) {
            continue;
        }
        let Some(table) = schema.tables.iter().find(|table| &table.name == name) else {
            continue;
        };
        let named = str_repr(name);
        if let Some(column) = &table.time_column {
            if choice.column.as_ref() != Some(column) {
                continue;
            }
            if choice.how.as_deref() == Some("earliest") {
                let other = choice
                    .among
                    .iter()
                    .find(|other| *other != column)
                    .expect("several candidates");
                lines.push(format!("{name}.{column} was taken as the time column because its values start earliest of {}. If another says when a row became true, name it: time_columns={{{named}: {}}}. If none does, clear it: time_columns={{{named}: None}}.", and_list(&choice.among), str_repr(other)));
            } else if choice.how.as_deref() == Some("edit_stamp") {
                lines.push(format!("{name}.{column} was taken as the time column although it looks like when a row was last changed, because the table has no other timestamp and no table links to it, so it reads as a log of changes. If it does not say when a row became true, clear it: time_columns={{{named}: None}}."));
            }
            for doubt in &choice.unconfirmed {
                if &doubt.column == column {
                    lines.push(
                        doubt.taken_unchecked(name, &format!("time_columns={{{named}: None}}")),
                    );
                }
            }
        } else {
            for rejected in &choice.refused {
                if table.col(&rejected.column).is_some()
                    && choice.column.as_ref() != Some(&rejected.column)
                {
                    lines.push(format!("{name}.{} was not taken as the time column because {}. If it does say when a row became true, name it: time_columns={{{named}: {}}}.", rejected.column, rejected.why(), str_repr(&rejected.column)));
                }
            }
        }
    }
    if !lines.is_empty() {
        said.sdm(format!("Some time columns were inferred, or refused, on a guess about what a column holds:\n- {}", lines.join("\n- ")));
    }
}

/// `_warn_several_columns_could_be_the_time_column(schema)`.
pub fn warn_several_columns_could_be_the_time_column(schema: &Schema, said: &mut Said) {
    if schema.tables.len() < 2 && schema.relationships.is_empty() {
        return;
    }
    let mut tables: Vec<&Table> = schema.tables.iter().collect();
    tables.sort_by(|a, b| a.name.cmp(&b.name));
    for table in tables {
        if table.time_column.is_some() {
            continue;
        }
        let mut stamps: Vec<String> = table
            .columns
            .iter()
            .filter(|c| c.stype == "timestamp")
            .map(|c| c.name.clone())
            .collect();
        stamps.sort();
        if stamps.len() < 2 {
            continue;
        }
        let n = str_repr(&table.name);
        let first = str_repr(&stamps[0]);
        said.sdm(format!(
            "{n} has more than one column that could be its time column, {}, so it was given none, and a query over a window of it is refused. Name the one that says when a row became true: time_columns={{{n}: {first}}} names {first}, which is only the first of them alphabetically.",
            and_list(&stamps)
        ));
    }
}

/// Keys named per table, in the order given.
pub type NamedKeys = Vec<(String, KeyArg)>;

/// A primary key a caller named: no key, a key, or the `repr` of a value
/// that names no columns, which is refused where it is applied.
pub type KeyArg = Result<Option<Key>, String>;

/// The arguments of `Schema.from_data`, as the core reads them.
#[derive(Default)]
pub struct DataArgs {
    pub primary_keys: Option<NamedKeys>,
    pub time_columns: Option<Vec<(String, Value)>>,
    pub relationships: Option<Vec<Rel>>,
    /// The refusal of the first link that is not a `Relationship`;
    /// `relationships` holds the ones before it.
    pub not_a_link: Option<String>,
}

/// `Schema.from_data(tables, ...)`.
pub fn from_data(
    tables: Vec<(String, Arc<Frame>)>,
    args: &DataArgs,
    host: &Host,
    said: &mut Said,
) -> R<Schema> {
    if tables.is_empty() {
        return Err(Refusal::invalid("a schema needs at least one table"));
    }
    let mut built: Vec<Table> = Vec::new();
    let mut choices = BTreeMap::new();
    for (name, data) in tables {
        if name.is_empty() {
            return Err(Refusal::invalid(format!(
                "table names must be non-empty strings; got {}",
                str_repr(&name)
            )));
        }
        let mut table = infer_table(&name, data.clone(), said)?;
        let candidates = table
            .columns
            .iter()
            .filter(|column| column.stype == "timestamp")
            .map(|column| column.name.clone())
            .collect::<Vec<_>>();
        let choice = crate::time_columns::from_frame(&data, &candidates, host);
        table.set_time_column_alone(choice.column.as_deref())?;
        choices.insert(name.clone(), choice);
        built.push(table);
    }
    apply_keys_and_times(
        &mut built,
        args.primary_keys.as_deref().unwrap_or(&[]),
        args.time_columns.as_deref().unwrap_or(&[]),
        said,
    )?;
    let inferred = match &args.relationships {
        None => {
            let found = infer_relationships(&built);
            joinable_inferred_links(&built, &found, said)
        }
        Some(given) => {
            let checked = checked_links(&built, given)?;
            if let Some(refused) = &args.not_a_link {
                return Err(Refusal::invalid(refused.clone()));
            }
            checked
        }
    };
    settle_time_choices(
        &mut built,
        &mut choices,
        args.time_columns.as_deref().unwrap_or(&[]),
        &inferred,
    )?;
    said.ignoring.push(SDM_WARNING.to_string());
    let made = Schema::new(&built, &inferred, said);
    said.ignoring.pop();
    let mut schema = made?;
    schema.mark_key_columns(said)?;
    let mut dates: BTreeMap<String, String> = BTreeMap::new();
    for table in &built {
        if let Some(named) = warn_dates_stored_as_text(table, host, said) {
            dates.insert(table.name.clone(), named);
        }
    }
    if args.relationships.is_none() {
        let spoken = warn_repeats_cost_the_table_its_key(&built, said)?;
        warn_table_cannot_be_linked_to(&built, &dates, &spoken, said);
        warn_one_key_name_keys_several_tables(&built, said);
    }
    warn_stype_was_decided_by_the_rows_read(&built, said)?;
    warn_time_choices(
        &schema,
        &choices,
        args.time_columns.as_deref().unwrap_or(&[]),
        said,
    );
    let rels = schema.relationships.clone();
    warn_two_ways_to_reach_one_table(&rels, said);
    Ok(schema)
}

// --- What a schema says ---------------------------------------------------------------

pub(crate) fn key_json(key: &Option<Key>) -> Value {
    match key {
        None => Value::Null,
        Some(Key::One(k)) => json!(k),
        Some(Key::Many(v)) => json!(v),
    }
}

pub fn warnings_json(warnings: &[Warning]) -> Value {
    Value::Array(
        warnings
            .iter()
            .map(|w| json!({"category": w.category, "message": w.message}))
            .collect(),
    )
}

/// `to_schema_definition(schema)`, as JSON, and the warnings it gave.
pub fn definition(schema: &Schema) -> Value {
    let mut said = Said::default();
    let mut tables = serde_json::Map::new();
    for table in &schema.tables {
        let mut cols = Vec::new();
        for c in &table.columns {
            let (Some(dtype), Some(stype)) =
                (Dtype::parse(pql_dtype(&c.dtype)), Stype::parse(&c.stype))
            else {
                continue;
            };
            cols.push(json!([c.name, stype.value(), dtype.value(), Value::Null]));
        }
        if let Some(Key::Many(names)) = &table.primary_key {
            let n = str_repr(&table.name);
            said.sdm(format!(
                "table {n} has the composite primary key {}, and a predictive query can only name a single key column, so this query is resolved against {n} as though it had no key. The request still carries the composite key; only the query does not see it. Name a single column as the primary key if the query needs to identify a row of {n}.",
                py::str_list_repr(names)
            ));
        }
        let pkey = match &table.primary_key {
            Some(Key::One(k)) => json!(k),
            _ => Value::Null,
        };
        tables.insert(
            table.name.clone(),
            json!({"cols": cols, "pkey": pkey, "time_col": table.time_column, "end_time_col": Value::Null}),
        );
    }
    let groups: Vec<Value> = schema
        .relationships
        .iter()
        .map(|l| {
            let mut pair = [
                (l.source_table.clone(), l.source_column.clone()),
                (l.target_table.clone(), l.target_column.clone()),
            ];
            pair.sort();
            json!([[pair[0].0, pair[0].1], [pair[1].0, pair[1].1]])
        })
        .collect();
    json!({"tables": tables, "col_groups": groups, "warnings": warnings_json(&said.warnings)})
}

/// The description `CONTRACT.md` names: `_describe_schema` with key counts,
/// how each column was typed, source names, and the definition.
pub fn describe(schema: &Schema) -> Value {
    let mut tables = serde_json::Map::new();
    for table in &schema.tables {
        let mut columns = serde_json::Map::new();
        for c in &table.columns {
            columns.insert(
                c.name.clone(),
                json!({"dtype": c.dtype, "stype": c.stype, "untyped": c.untyped, "stype_declared": c.stype_declared}),
            );
        }
        let mut described = serde_json::Map::new();
        described.insert("primary_key".into(), key_json(&table.primary_key));
        described.insert("time_column".into(), json!(table.time_column));
        described.insert("duplicate_keys".into(), json!(table.duplicate_keys));
        described.insert("columns".into(), Value::Object(columns));
        if let Some(source_name) = &table.source_name {
            described.insert("source_name".into(), json!(source_name));
        }
        tables.insert(table.name.clone(), Value::Object(described));
    }
    let relationships: Vec<Value> = schema
        .relationships
        .iter()
        .map(|r| {
            json!([
                r.source_table,
                r.source_column,
                r.target_table,
                r.target_column
            ])
        })
        .collect();
    json!({"tables": tables, "relationships": relationships, "definition": definition(schema)})
}

/// An outcome's two JSON texts: the description or the refusal, and the
/// warnings.
pub fn outcome_json(result: &R<Schema>, said: &Said) -> (String, String) {
    let first = match result {
        Ok(schema) => describe(schema),
        Err(refusal) => json!({"error": {"code": refusal.code, "message": refusal.message}}),
    };
    (first.to_string(), warnings_json(&said.warnings).to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_refusal_is_delayed_until_inference_needs_it() {
        for (name, precision) in [
            ("decimal32(8, 2)[pyarrow]", 8),
            ("decimal64(16, 2)[pyarrow]", 16),
        ] {
            let mut column = frame::Column {
                name: "value".into(),
                dtype: PdType::Decimal {
                    name: name.into(),
                    precision,
                    scale: 2,
                },
                cells: vec![PyVal::Decimal("1.00".into())],
                name_repr: None,
            };
            let inferred = infer_column(&column).unwrap();
            assert_eq!(inferred.dtype, "float64");
            assert_eq!(inferred.stype, "numerical");
            let frame = Frame {
                columns: vec![column.clone()],
                rows: 1,
            };
            assert_eq!(infer_primary_key(&frame, &[inferred]).unwrap(), None);
            column.name = "item_id".into();
            let inferred = infer_column(&column).unwrap();
            let mut frame = Frame {
                columns: vec![column],
                rows: 1,
            };
            let refusal = infer_primary_key(&frame, std::slice::from_ref(&inferred)).unwrap_err();
            assert_eq!(refusal.raised_as(), "pyarrow.lib.ArrowNotImplementedError");
            frame.rows = 0;
            frame.columns[0].cells.clear();
            assert_eq!(infer_primary_key(&frame, &[inferred]).unwrap(), None);
        }
    }

    #[test]
    fn extension_cardinality_refuses_before_key_inference() {
        for name in ["bool8", "json", "uuid"] {
            for cells in [
                Vec::new(),
                vec![PyVal::None],
                vec![PyVal::Str("one".into())],
            ] {
                let column = frame::Column {
                    name: "value".into(),
                    dtype: PdType::Other {
                        name: format!("extension<arrow.{name}>[pyarrow]"),
                        scalar_only: false,
                        infer_dtype_error: None,
                    },
                    cells,
                    name_repr: None,
                };
                let refusal = infer_column(&column).unwrap_err();
                assert_eq!(
                    refusal.message,
                    format!(
                        "Function 'unique' has no kernel matching input types (extension<arrow.{name}>)"
                    )
                );
            }
        }
    }

    fn table(name: &str, columns: &[(&str, &str, &str)]) -> Table {
        Table {
            name: name.to_string(),
            data: None,
            source_name: Some(name.to_string()),
            columns: columns
                .iter()
                .map(|(n, d, s)| Col::new(n, d, s).expect("a semantic type"))
                .collect(),
            primary_key: None,
            time_column: None,
            duplicate_keys: None,
        }
    }

    #[test]
    fn a_consistent_schema_re_derives_its_keys_and_refuses_a_contradiction() {
        let mut users = table("users", &[("id", "int64", "numerical")]);
        let mut said = Said::default();
        users
            .set_primary_key_alone(Some(Key::One("id".into())), &mut said)
            .unwrap();
        let orders = table(
            "orders",
            &[
                ("user_id", "int64", "numerical"),
                ("at", "timestamp[us]", "timestamp"),
            ],
        );
        let link = Rel::new("orders", "user_id", "users", "id");
        let mut schema = Schema::new(&[orders, users], &[link], &mut said).unwrap();
        schema.check_consistent(&mut said).unwrap();
        assert_eq!(schema.tables[0].columns[0].stype, "ID");
        schema.tables[0].columns[1].stype = "numerical".to_string();
        assert_eq!(
            schema.check_consistent(&mut said).unwrap_err().message,
            "column 'at' holds 'timestamp[us]', which cannot mean 'numerical'"
        );
    }
}
