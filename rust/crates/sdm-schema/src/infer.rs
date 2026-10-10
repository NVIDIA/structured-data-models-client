// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One table's inference, by the reference's rules: each column's data type
//! from the Arrow type its preview returned, its semantic type from its name
//! and the preview's values, the primary key from the catalog or, where it
//! declares none, from names and values, and the time column among the
//! timestamps. Every path these rules take that this build does not port is
//! refused by name ([`crate::not_yet`]).

use std::collections::BTreeSet;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::{ColumnType, CoreError};

use crate::catalog::{CatalogColumn, Constraint};
use crate::names;
use crate::not_yet;
use crate::values::{self, Value};

/// The rows a preview reads: a preview that returns this many may have left
/// rows unread, which the reference then reads the whole table for.
pub const PREVIEW_ROWS: u64 = 1_000;

/// One column, inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
    /// The column's type as admitted from the preview.
    pub received: ColumnType,
    /// Whether the catalog lets the column hold no value.
    pub nullable: bool,
}

/// One table, inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredTable {
    pub name: String,
    pub columns: Vec<InferredColumn>,
    pub primary_key: Option<String>,
    pub time_column: Option<String>,
    /// The column sets the catalog declares a primary or unique key over,
    /// which the source refuses to repeat.
    pub unique: Vec<BTreeSet<String>>,
}

impl InferredTable {
    /// Whether the source refuses both a repeated value of `key` and a row
    /// missing part of it, which leaves nothing to count.
    pub fn enforces(&self, key: &[String]) -> bool {
        let names: BTreeSet<&String> = key.iter().collect();
        let unique = self
            .unique
            .iter()
            .any(|set| set.iter().all(|column| names.contains(column)));
        let never_null = key
            .iter()
            .all(|name| self.columns.iter().any(|c| &c.name == name && !c.nullable));
        unique && never_null
    }

    /// The column `name`, if the table has it.
    pub fn column(&self, name: &str) -> Option<&InferredColumn> {
        self.columns.iter().find(|c| c.name == name)
    }
}

/// Infers table `name` from its catalog columns and constraints and its
/// preview (`schema` and `batches`, as admitted), which holds the catalog's
/// columns in order.
pub fn table(
    name: &str,
    catalog: &[CatalogColumn],
    constraints: &[Constraint],
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<InferredTable, CoreError> {
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if rows == 0 {
        // The reference refuses an empty table as a RuntimeError.
        return Err(not_yet("empty_table"));
    }
    let single = |kinds: &[&str]| -> BTreeSet<String> {
        constraints
            .iter()
            .filter(|c| kinds.contains(&c.kind.as_str()) && c.columns.len() == 1)
            .map(|c| c.columns[0].clone())
            .collect()
    };
    let declared_primary = single(&["PRIMARY KEY"]);
    let declared_unique = single(&["PRIMARY KEY", "UNIQUE"]);
    let unique = constraints
        .iter()
        .filter(|c| c.kind == "PRIMARY KEY" || c.kind == "UNIQUE")
        .map(|c| c.columns.iter().cloned().collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();

    let mut columns = Vec::with_capacity(catalog.len());
    let mut samples = Vec::with_capacity(catalog.len());
    let mut primary = Vec::new();
    for (index, declared) in catalog.iter().enumerate() {
        let field = schema.field(index);
        let dtype = dtype_of(field.data_type())?;
        let sample = values::column(batches, index)?;
        let stype = stype_of(&declared.name, dtype, &sample)?;
        let is_primary = declared.primary_key || declared_primary.contains(&declared.name);
        if is_primary {
            primary.push(declared.name.clone());
        }
        columns.push(InferredColumn {
            name: declared.name.clone(),
            dtype,
            stype,
            received: ColumnType::from_arrow(&declared.name, field.data_type())?,
            nullable: !is_primary && !declared.not_null,
        });
        samples.push(sample);
    }

    // The catalog's key, when it declares exactly one single-column one,
    // then its one unique column, then a guess from names and values; each
    // makes its column an identifier.
    let unique_columns: Vec<&String> = catalog
        .iter()
        .map(|c| &c.name)
        .filter(|n| declared_unique.contains(*n))
        .collect();
    let key = match (primary.as_slice(), unique_columns.as_slice()) {
        ([one], _) => Some(one.clone()),
        (_, [one]) => Some((*one).clone()),
        _ => guess_key(name, &columns, &samples)?,
    };
    if let Some(key) = &key {
        let column = columns
            .iter_mut()
            .find(|c| &c.name == key)
            .expect("the key is a column of the table");
        if !matches!(column.dtype, Dtype::Int | Dtype::Float | Dtype::String) {
            return Err(not_yet("key_type"));
        }
        column.stype = Stype::Id;
    }

    let time_column = choose_time_column(&columns, &samples, rows)?;
    Ok(InferredTable {
        name: name.to_owned(),
        columns,
        primary_key: key,
        time_column,
        unique,
    })
}

/// A column's data type from the Arrow type its preview returned, as the
/// reference's reading of the catalog's types gives it for the plain types;
/// any other is refused.
fn dtype_of(data_type: &DataType) -> Result<Dtype, CoreError> {
    Ok(match data_type {
        DataType::Boolean => Dtype::Bool,
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Decimal128(_, 0) => Dtype::Int,
        DataType::Float32 | DataType::Float64 | DataType::Decimal128(_, _) => Dtype::Float,
        DataType::Utf8 => Dtype::String,
        DataType::Date32 | DataType::Timestamp(_, _) => Dtype::Date,
        DataType::Time64(_) => Dtype::Time,
        _ => return Err(not_yet("column_type")),
    })
}

fn is_numeric(dtype: Dtype) -> bool {
    matches!(dtype, Dtype::Int | Dtype::Float)
}

/// The semantic type: an identifier by its name, a timestamp by its type,
/// then a category by its name and how many distinct values the first
/// 1,000 rows hold, else the type's default.
fn stype_of(name: &str, dtype: Dtype, sample: &[Option<Value>]) -> Result<Stype, CoreError> {
    if matches!(dtype, Dtype::Int | Dtype::Float | Dtype::String) && names::says_id(name) {
        return Ok(Stype::Id);
    }
    match dtype {
        Dtype::Date | Dtype::Time => return Ok(Stype::Timestamp),
        Dtype::String => {
            text_holds_no_dates(name, sample)?;
            text_holds_single_values(sample)?;
        }
        _ => {}
    }
    if dtype == Dtype::Bool {
        return Ok(Stype::Categorical);
    }
    let measured = is_numeric(dtype) && names::says_measure(name);
    if !measured {
        let first: Vec<Option<Value>> = sample.iter().take(1_000).cloned().collect();
        let held = first.iter().flatten().count();
        let distinct = values::distinct(&first);
        let share = |limit: f64| held > 0 && (distinct as f64 / held as f64) <= limit;
        let categorical = distinct < 20
            || if dtype == Dtype::String {
                share(0.5)
            } else {
                share(0.05)
            };
        if categorical {
            return Ok(Stype::Categorical);
        }
    }
    Ok(match dtype {
        Dtype::String => Stype::Text,
        _ => Stype::Numerical,
    })
}

/// The words the reference's date reader knows, which a value without a
/// digit must hold only of to read as a date: its jump words, weekdays,
/// months, units, meridians, UTC names, and the relative days.
const DATE_WORDS: [&str; 61] = [
    "at",
    "on",
    "and",
    "ad",
    "m",
    "t",
    "of",
    "st",
    "nd",
    "rd",
    "th",
    "mon",
    "monday",
    "tue",
    "tuesday",
    "wed",
    "wednesday",
    "thu",
    "thursday",
    "fri",
    "friday",
    "sat",
    "saturday",
    "sun",
    "sunday",
    "jan",
    "january",
    "feb",
    "february",
    "mar",
    "march",
    "apr",
    "april",
    "may",
    "jun",
    "june",
    "jul",
    "july",
    "aug",
    "august",
    "sep",
    "sept",
    "september",
    "oct",
    "october",
    "nov",
    "november",
    "dec",
    "december",
    "h",
    "hour",
    "hours",
    "minute",
    "minutes",
    "s",
    "second",
    "seconds",
    "am",
    "pm",
    "utc",
    "gmt",
];

/// Whether `value` cannot read as a date: it holds no digit, and a word
/// the date reader does not know.
fn cannot_be_a_date(value: &str) -> bool {
    if value.chars().any(char::is_numeric) {
        return false;
    }
    let lowered = value.to_lowercase();
    lowered
        .split(|c: char| !c.is_alphabetic())
        .filter(|word| !word.is_empty())
        .any(|word| {
            !DATE_WORDS.contains(&word)
                && !["a", "p", "z", "now", "today", "tomorrow", "yesterday"].contains(&word)
        })
}

/// Text is a timestamp to the reference when its first 100 values hold
/// one and, if its name says time, most of them read as dates, or, if it
/// does not, every one does. Reading dates is not ported (T1.25, DT-7):
/// text is taken as no timestamp when those values hold none, or when its
/// name does not say time and one of them cannot be a date.
fn text_holds_no_dates(name: &str, sample: &[Option<Value>]) -> Result<(), CoreError> {
    let first: Vec<&str> = sample
        .iter()
        .take(100)
        .flatten()
        .filter_map(|v| match v {
            Value::Text(text) if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect();
    if first.is_empty() {
        return Ok(());
    }
    if !names::says_time(name) && first.iter().any(|v| cannot_be_a_date(v)) {
        return Ok(());
    }
    Err(not_yet("text_time"))
}

/// Text holding a separator (`;`, `:`, `|` or a tab) in its first 500
/// values may hold several categories per value, which is not ported.
fn text_holds_single_values(sample: &[Option<Value>]) -> Result<(), CoreError> {
    let separated = sample
        .iter()
        .take(500)
        .flatten()
        .any(|v| matches!(v, Value::Text(text) if text.contains([';', ':', '|', '\t'])));
    if separated {
        return Err(not_yet("multicategorical"));
    }
    Ok(())
}

/// The table's key among its identifiers, or, with none, among the columns
/// named as the table is, by the reference's score: a name that is the
/// table's (in its singular, plural, snake or camel forms) followed by
/// `id`, `hash`, `key`, `code` or `uuid`, or the table's name itself, and
/// how distinct its values are. A score of 4 or more counts; a tie for the
/// best is refused.
fn guess_key(
    table: &str,
    columns: &[InferredColumn],
    samples: &[Vec<Option<Value>>],
) -> Result<Option<String>, CoreError> {
    let mut candidates: Vec<usize> = (0..columns.len())
        .filter(|&i| columns[i].stype == Stype::Id)
        .collect();
    if candidates.is_empty() {
        let lowered = table.to_lowercase();
        let singular = lowered.strip_suffix('s');
        candidates = (0..columns.len())
            .filter(|&i| {
                let column = columns[i].name.to_lowercase();
                column == lowered || singular == Some(column.as_str())
            })
            .collect();
    }
    if candidates.is_empty() {
        return Ok(None);
    }
    let variants = table_name_variants(table);
    let mut scores: Vec<(usize, f64)> = Vec::new();
    for &i in &candidates {
        let name = &columns[i].name;
        let lowered = name.to_lowercase();
        let mut score: u32 = if lowered == "id" { 4 } else { 0 };
        let named = |variant: &str| -> u32 {
            let mut add = 0;
            for suffix in ["id", "hash", "key", "code", "uuid"] {
                if !lowered.ends_with(suffix) {
                    continue;
                }
                if lowered == format!("{variant}_{suffix}")
                    || lowered == format!("{variant}{suffix}")
                {
                    add += 5;
                    break;
                }
                if lowered.ends_with(&format!("{variant}_{suffix}")) {
                    add += 2;
                }
                if lowered.ends_with(&format!("{variant}{suffix}")) {
                    add += 2;
                }
            }
            if table == "studies" && name == "nct_id" {
                add += 1;
            }
            add
        };
        if variants.contains(&lowered) {
            // The reference stops at the form the name equals, having
            // added what the forms it met first gave, in an order that
            // varies by run; only a name no other form scores is ported.
            if variants.iter().any(|v| v != &lowered && named(v) > 0) {
                return Err(not_yet("key_name_order"));
            }
            score += 4;
        } else {
            score += variants.iter().map(|v| named(v)).sum::<u32>();
        }
        let sample = &samples[i];
        let share = values::distinct(sample) as f64 / sample.len() as f64;
        scores.push((i, f64::from(score) + 3.0 * share));
    }
    scores.retain(|(_, score)| *score >= 4.0);
    // Stable, so a tie keeps the columns' order.
    scores.sort_by(|a, b| b.1.total_cmp(&a.1));
    match scores.as_slice() {
        [] => Ok(None),
        [(i, _)] => Ok(Some(columns[*i].name.clone())),
        [(i, best), (_, next), ..] if best != next => Ok(Some(columns[*i].name.clone())),
        _ => Err(not_yet("key_tie")),
    }
}

/// The table's name in the forms a key's name may take it in: as given and
/// without an `info` ending, each lowercased, in snake_case and joined, in
/// the singular and plural, and without an `ing`.
fn table_name_variants(table: &str) -> BTreeSet<String> {
    let chars: Vec<char> = table.chars().collect();
    let lowered = table.to_lowercase();
    let mut bases = vec![table.to_owned()];
    if lowered.ends_with("_info") && chars.len() >= 5 {
        bases.push(chars[..chars.len() - 5].iter().collect());
    } else if lowered.ends_with("info") && chars.len() >= 4 {
        bases.push(chars[..chars.len() - 4].iter().collect());
    }
    let drop = |text: &str, n: usize| -> String {
        let chars: Vec<char> = text.chars().collect();
        chars[..chars.len().saturating_sub(n)].iter().collect()
    };
    let mut variants = BTreeSet::new();
    for base in bases {
        let lower = base.to_lowercase();
        let forms = [
            lower.clone(),
            snake_case(&base).to_lowercase(),
            base.replace('_', "").to_lowercase(),
        ];
        for form in &forms {
            variants.insert(form.clone());
            if lower.ends_with('s') {
                variants.insert(drop(form, 1));
            } else {
                variants.insert(format!("{form}s"));
            }
            if lower.ends_with("ies") {
                variants.insert(format!("{}y", drop(form, 3)));
            } else if lower.ends_with('y') {
                variants.insert(format!("{}ies", drop(form, 1)));
            }
            if lower.ends_with("ing") {
                variants.insert(drop(form, 3));
            }
        }
    }
    variants
}

/// The reference's snake_case: `(.)([A-Z][a-z]+)` to `\1_\2`, then
/// `([a-z0-9])([A-Z])` to `\1_\2`.
fn snake_case(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut first = Vec::with_capacity(chars.len() + 4);
    let mut i = 0;
    while i < chars.len() {
        let word = i + 2 < chars.len()
            && chars[i] != '\n'
            && chars[i + 1].is_ascii_uppercase()
            && chars[i + 2].is_ascii_lowercase();
        if word {
            let mut end = i + 2;
            while end < chars.len() && chars[end].is_ascii_lowercase() {
                end += 1;
            }
            first.push(chars[i]);
            first.push('_');
            first.extend_from_slice(&chars[i + 1..end]);
            i = end;
        } else {
            first.push(chars[i]);
            i += 1;
        }
    }
    let mut second = String::with_capacity(first.len() + 4);
    let mut i = 0;
    while i < first.len() {
        let c = first[i];
        if (c.is_ascii_lowercase() || c.is_ascii_digit())
            && first.get(i + 1).is_some_and(char::is_ascii_uppercase)
        {
            second.push(c);
            second.push('_');
            second.push(first[i + 1]);
            i += 2;
        } else {
            second.push(c);
            i += 1;
        }
    }
    second
}

/// The time column among the timestamps, as the reference chooses it when
/// one is plain: no name of a load, an edit or a birth, held in at least
/// half the rows and not one value in every row, read in a preview that
/// holds every row. Any other choice is refused.
fn choose_time_column(
    columns: &[InferredColumn],
    samples: &[Vec<Option<Value>>],
    rows: usize,
) -> Result<Option<String>, CoreError> {
    let candidates: Vec<usize> = (0..columns.len())
        .filter(|&i| columns[i].stype == Stype::Timestamp)
        .collect();
    if candidates.is_empty() {
        return Ok(None);
    }
    // A time of day: the reference reads none of its values as a time, so
    // it refuses the column for its values and warns; not ported.
    if candidates.iter().any(|&i| columns[i].dtype == Dtype::Time) {
        return Err(not_yet("time_of_day"));
    }
    if rows as u64 >= PREVIEW_ROWS {
        return Err(not_yet("time_whole_table"));
    }
    for &i in &candidates {
        let name = &columns[i].name;
        if names::is_load_stamp(name) || names::is_birth_date(name) || names::is_edit_stamp(name) {
            return Err(not_yet("time_name"));
        }
        let held = samples[i].iter().flatten().count();
        let missing = rows - held;
        let constant = held == 0 || (rows > 1 && values::distinct(&samples[i]) == 1);
        if constant || 2 * missing > rows {
            return Err(not_yet("time_values"));
        }
    }
    match candidates.as_slice() {
        [one] => Ok(Some(columns[*one].name.clone())),
        _ => Err(not_yet("time_several")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names_take_the_reference_forms() {
        let orders = table_name_variants("orders");
        assert!(orders.contains("orders") && orders.contains("order"));
        let users = table_name_variants("UserInfo");
        for form in ["userinfo", "user_info", "user", "users", "userinfos"] {
            assert!(users.contains(form), "{form}: {users:?}");
        }
        let cities = table_name_variants("cities");
        assert!(cities.contains("city"));
        assert_eq!(snake_case("UserInfo"), "User_Info");
        assert_eq!(snake_case("HTTPRequest2Go"), "HTTP_Request2_Go");
        assert_eq!(snake_case("already_snake"), "already_snake");
    }

    #[test]
    fn a_value_without_digits_or_known_words_cannot_be_a_date() {
        assert!(cannot_be_a_date("US"));
        assert!(cannot_be_a_date("in"));
        assert!(!cannot_be_a_date("March"));
        assert!(!cannot_be_a_date("now"));
        assert!(!cannot_be_a_date("2024-01-01"));
        assert!(!cannot_be_a_date("a"));
        assert!(!cannot_be_a_date(" "));
    }
}
