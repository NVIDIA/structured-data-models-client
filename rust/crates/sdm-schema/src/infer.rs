// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One table's inference, by the reference's rules: each column's data type
//! from the Arrow type its preview returned, its semantic type from its name
//! and the preview's values, the primary key from the catalog or, where it
//! declares none, from names and values, and the time column among the
//! timestamps, those named as a load, an edit or a birth passed over
//! (T1.21d). Every path these rules take that this build does not port is
//! refused by name ([`crate::not_yet`]).

use std::collections::BTreeSet;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema};
use sdm_contract::ops::{Dtype, Stype};
use sdm_contract::{ColumnType, CoreError, Scalar};

use crate::names;
use crate::not_yet;
use crate::values::{self, Value};
use sdm_contract::catalog::{CatalogColumn, CatalogKey, KeyKind};

/// The rows a preview reads: a preview that returns this many may have left
/// rows unread, which the reference then reads the whole table for.
pub const PREVIEW_ROWS: u64 = 1_000;

/// One column, inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredColumn {
    pub name: String,
    pub dtype: Dtype,
    pub stype: Stype,
    /// What the column meant before a key named after inference made it an
    /// identifier (a keyless table keyed by the column its declared foreign
    /// keys refer to), as the reference's key setter records it.
    pub stype_before_key: Option<Stype>,
    /// The column's type as admitted from the preview.
    pub received: ColumnType,
    /// Whether the catalog lets the column hold no value.
    pub nullable: bool,
    /// Whether the catalog declares a primary or unique key of this one
    /// column.
    pub unique: bool,
}

/// Why inference passed a timestamp over for the time column, as the
/// reference's `RejectedTimeColumn` holds it: its values (no row holds
/// one, or every row that holds one holds the same; more than half the
/// rows hold none), or its name (a stamp of a row's load, of its last edit,
/// or a date of birth).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PassedOver {
    /// No row holds a value (`value` `None`), or the table has more than
    /// one row and every row that holds one holds `value`; `missing` of
    /// its `rows` rows hold none.
    Constant {
        value: Option<Scalar>,
        missing: usize,
        rows: usize,
    },
    /// `missing` of its `rows` rows, more than half, hold no value.
    Sparse {
        missing: usize,
        rows: usize,
    },
    LoadStamp,
    EditStamp,
    BirthDate,
}

impl PassedOver {
    /// The reason's name, as the reference's `RejectedTimeColumn.reason`
    /// gives it.
    pub fn reason(&self) -> &'static str {
        match self {
            PassedOver::Constant { .. } => "constant",
            PassedOver::Sparse { .. } => "sparse",
            PassedOver::LoadStamp => "load_stamp",
            PassedOver::EditStamp => "edit_stamp",
            PassedOver::BirthDate => "birth_date",
        }
    }
}

/// How the time column was taken (`TimeColumnChoice.how`): the one
/// timestamp left, the one creation stamp among several, the one whose
/// values start earliest, or a lone stamp of the last edit in a table no
/// other links to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Taken {
    Only,
    Created,
    Earliest,
    EditStamp,
}

/// The time column inference took and how, and the timestamps it passed
/// over, as the reference's `TimeColumnChoice` holds them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimeChoice {
    pub column: Option<String>,
    pub how: Option<Taken>,
    /// The columns it was chosen from, itself included.
    pub among: Vec<String>,
    /// Every timestamp passed over, in table order, the lone edit stamp
    /// taken as the last resort included.
    pub refused: Vec<(String, PassedOver)>,
    /// The lone edit stamp left, when no other timestamp was.
    pub fallback: Option<String>,
}

impl TimeChoice {
    /// This choice for a table another table links to, or none does
    /// (`TimeColumnChoice.for_links`): a lone edit stamp is refused in a
    /// table something links to, an entity edited in place, and taken in
    /// one nothing links to, a log of changes.
    pub fn for_links(&self, linked: bool) -> TimeChoice {
        let Some(fallback) = &self.fallback else {
            return self.clone();
        };
        TimeChoice {
            column: (!linked).then(|| fallback.clone()),
            how: (!linked).then_some(Taken::EditStamp),
            among: if linked {
                vec![]
            } else {
                vec![fallback.clone()]
            },
            refused: self.refused.clone(),
            fallback: self.fallback.clone(),
        }
    }

    /// The timestamps passed over, each with its reason, in table order:
    /// the refused but the one taken.
    pub fn rejected(&self) -> impl Iterator<Item = &(String, PassedOver)> {
        self.refused
            .iter()
            .filter(move |(name, _)| Some(name) != self.column.as_ref())
    }
}

/// A column whose type has no scalar form on the wire, which the schema
/// leaves out and names (`_describe_source_table`): a list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeftOut {
    pub name: String,
    pub dtype: Dtype,
}

/// A time of day whose values the reference's engine reads as no time at
/// all (its `to_datetime` over the preview), which it says it dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnreadTimes {
    pub column: String,
    /// The values not read: every one the preview holds.
    pub unread: usize,
    /// The rows the preview holds.
    pub rows: usize,
    /// The first three distinct values, in the order they appear.
    pub examples: Vec<Scalar>,
}

/// One table, inferred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferredTable {
    pub name: String,
    /// The source's name of the table read.
    pub source: String,
    pub columns: Vec<InferredColumn>,
    /// The columns left out, in table order.
    pub left_out: Vec<LeftOut>,
    /// The times of day the engine's reading drops, in table order.
    pub unread_times: Vec<UnreadTimes>,
    pub primary_key: Option<String>,
    /// How inference chose the time column, before the links settle a
    /// lone edit stamp (`crate::assemble`).
    pub time: TimeChoice,
    /// The table's time column: inference's choice until the caller names
    /// one (`crate::assemble::apply_keys_and_times`).
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

/// Infers table `name`, read as `source` in the source, from its catalog
/// columns and constraints and its preview (`schema` and `batches`, as
/// admitted), which holds the catalog's columns in order. Inference reads
/// the table by the name the schema gives it, as the reference's engine
/// does (its key's guess is by that name).
pub fn table(
    name: &str,
    source: &str,
    catalog: &[CatalogColumn],
    constraints: &[CatalogKey],
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<InferredTable, CoreError> {
    let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    if rows == 0 {
        // The reference refuses an empty table as a RuntimeError.
        return Err(not_yet("empty_table"));
    }
    let single = |kinds: &[KeyKind]| -> BTreeSet<String> {
        constraints
            .iter()
            .filter(|c| kinds.contains(&c.kind) && c.columns.len() == 1)
            .map(|c| c.columns[0].clone())
            .collect()
    };
    let declared_primary = single(&[KeyKind::Primary]);
    let declared_unique = single(&[KeyKind::Primary, KeyKind::Unique]);
    let unique = constraints
        .iter()
        .filter(|c| matches!(c.kind, KeyKind::Primary | KeyKind::Unique))
        .map(|c| c.columns.iter().cloned().collect::<BTreeSet<_>>())
        .collect::<Vec<_>>();

    let mut columns = Vec::with_capacity(catalog.len());
    let mut samples = Vec::with_capacity(catalog.len());
    let mut instants = Vec::with_capacity(catalog.len());
    let mut firsts = Vec::with_capacity(catalog.len());
    let mut left_out = Vec::new();
    let mut unread_times = Vec::new();
    let mut primary = Vec::new();
    for (index, declared) in catalog.iter().enumerate() {
        let field = schema.field(index);
        let dtype = dtype_of(field.data_type())?;
        if matches!(dtype, Dtype::Intlist | Dtype::Floatlist | Dtype::Stringlist) {
            // No scalar form on the wire: left out and named, as the
            // reference's schema leaves it out.
            left_out.push(LeftOut {
                name: declared.name.clone(),
                dtype,
            });
            continue;
        }
        let sample = values::column(batches, index)?;
        if dtype == Dtype::Time {
            // The engine reads a time of day's values as times, which none
            // is, and says how many it dropped.
            let unread = sample.iter().flatten().count();
            if unread > 0 {
                unread_times.push(UnreadTimes {
                    column: declared.name.clone(),
                    unread,
                    rows,
                    examples: values::times_of_day(batches, index, 3),
                });
            }
        }
        let stype = stype_of(&declared.name, dtype, &sample)?;
        let is_primary = declared.primary_key || declared_primary.contains(&declared.name);
        if is_primary {
            primary.push(declared.name.clone());
        }
        columns.push(InferredColumn {
            name: declared.name.clone(),
            dtype,
            stype,
            stype_before_key: None,
            received: ColumnType::from_arrow(&declared.name, field.data_type())?,
            nullable: !is_primary && !declared.not_null,
            unique: declared_unique.contains(&declared.name),
        });
        samples.push(sample);
        instants.push(if dtype == Dtype::Time {
            // Dropped by the engine's reading: none is held.
            vec![None; rows]
        } else {
            values::instants(batches, index)
        });
        firsts.push(values::first_temporal(batches, index));
    }

    // The catalog's key, when it declares exactly one single-column one,
    // then its one unique column, then a guess from names and values; each
    // makes its column an identifier. A column left out is never the key.
    let held = |n: &&String| columns.iter().any(|c| &c.name == *n);
    let primary: Vec<String> = primary.iter().filter(held).cloned().collect();
    let unique_columns: Vec<&String> = catalog
        .iter()
        .map(|c| &c.name)
        .filter(|n| declared_unique.contains(*n))
        .filter(held)
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

    let time = choose_time_column(
        &columns,
        &Read {
            instants: &instants,
            firsts: &firsts,
        },
        rows,
        key.as_deref(),
    )?;
    Ok(InferredTable {
        name: name.to_owned(),
        source: source.to_owned(),
        columns,
        left_out,
        unread_times,
        primary_key: key,
        time_column: time.column.clone(),
        time,
        unique,
    })
}

/// A column's data type from the Arrow type its preview returned, as the
/// reference's reading of the catalog's types gives it for the plain types
/// and the lists of them; any other is refused.
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
        // A list of integers, floats or text, as the reference reads a
        // catalog type ending `[]`; a list of anything else, a list of a
        // fixed size, a struct and a map stay refused, as the reference
        // fails them unnamed (register P1B-9).
        DataType::List(element) | DataType::LargeList(element) => {
            match dtype_of(element.data_type()) {
                Ok(Dtype::Int) => Dtype::Intlist,
                Ok(Dtype::Float) => Dtype::Floatlist,
                Ok(Dtype::String) => Dtype::Stringlist,
                _ => return Err(not_yet("column_type")),
            }
        }
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

/// What the preview holds of each column inferred, for the time column's
/// choice: its values as instants (`values::instants`, a time of day's as
/// none, since the engine's reading drops them) and its first value held.
struct Read<'a> {
    instants: &'a [Vec<Option<i128>>],
    firsts: &'a [Option<Scalar>],
}

/// The first real day: a value before it, such as 0001-01-01 or
/// 1900-01-01, stands for no date (`_FIRST_REAL_DAY`, 1901-01-01 UTC), in
/// nanoseconds since the epoch.
const FIRST_REAL_DAY: i128 = -2_177_452_800 * 1_000_000_000;

/// The time column among the timestamps, as the reference chooses it
/// (`predictive_query.time_columns.choose_time_column`), over a preview
/// that holds every row: a stamp of a load or a date of birth is passed
/// over by its name; any other timestamp held in no row, or in fewer than
/// half the rows, or holding one value in every row of a table of more
/// than one, is passed over for its values; a stamp of the last edit is
/// passed over too, and taken only when it is the one timestamp left
/// (settled by the links, [`TimeChoice::for_links`]); of the rest, the one
/// left is taken, else the one creation stamp, else the one whose values
/// start earliest, placeholders left out, the first in table order on a
/// tie. A time of day holds no value the engine reads, so it is passed
/// over as holding none. The key is never the time column. A preview that
/// may not hold every row is refused (`time_whole_table`): the reference
/// then asks the whole table, which this build does not port.
fn choose_time_column(
    columns: &[InferredColumn],
    read: &Read<'_>,
    rows: usize,
    key: Option<&str>,
) -> Result<TimeChoice, CoreError> {
    let candidates: Vec<usize> = (0..columns.len())
        .filter(|&i| columns[i].stype == Stype::Timestamp && Some(columns[i].name.as_str()) != key)
        .collect();
    if candidates.is_empty() {
        return Ok(TimeChoice::default());
    }
    if rows as u64 >= PREVIEW_ROWS {
        return Err(not_yet("time_whole_table"));
    }
    let mut refused = Vec::new();
    let mut plain = Vec::new();
    let mut edited = Vec::new();
    for &i in &candidates {
        let name = &columns[i].name;
        if names::is_load_stamp(name) {
            refused.push((name.clone(), PassedOver::LoadStamp));
            continue;
        }
        if names::is_birth_date(name) {
            refused.push((name.clone(), PassedOver::BirthDate));
            continue;
        }
        if let Some(why) = refused_for_values(&read.instants[i], read.firsts[i].as_ref(), rows) {
            refused.push((name.clone(), why));
            continue;
        }
        if names::is_edit_stamp(name) {
            refused.push((name.clone(), PassedOver::EditStamp));
            edited.push(name.clone());
        } else {
            plain.push(i);
        }
    }
    let chosen = |column: Option<String>, how: Option<Taken>, among: Vec<String>| TimeChoice {
        column,
        how,
        among,
        refused: refused.clone(),
        fallback: None,
    };
    let names_of = |indices: &[usize]| -> Vec<String> {
        indices.iter().map(|&i| columns[i].name.clone()).collect()
    };
    if plain.is_empty() {
        if let [one] = edited.as_slice() {
            return Ok(TimeChoice {
                fallback: Some(one.clone()),
                ..chosen(None, None, vec![])
            }
            .for_links(false));
        }
        return Ok(chosen(None, None, vec![]));
    }
    let created: Vec<usize> = plain
        .iter()
        .copied()
        .filter(|&i| names::is_creation_stamp(&columns[i].name))
        .collect();
    if let [one] = created.as_slice() {
        let how = if plain.len() > 1 {
            Taken::Created
        } else {
            Taken::Only
        };
        return Ok(chosen(
            Some(columns[*one].name.clone()),
            Some(how),
            names_of(&plain),
        ));
    }
    let among = if created.is_empty() { plain } else { created };
    if let [one] = among.as_slice() {
        return Ok(chosen(
            Some(columns[*one].name.clone()),
            Some(Taken::Only),
            names_of(&among),
        ));
    }
    let earliest = earliest(&among, read);
    Ok(chosen(
        Some(columns[earliest].name.clone()),
        Some(Taken::Earliest),
        names_of(&among),
    ))
}

/// Why a column is passed over for its values (`_refused_for_values`),
/// read over every row: none held; more than half missing; or, in a table
/// of more than one row, one value in every row that holds one, which is
/// named.
fn refused_for_values(
    instants: &[Option<i128>],
    first: Option<&Scalar>,
    rows: usize,
) -> Option<PassedOver> {
    let held: Vec<i128> = instants.iter().flatten().copied().collect();
    let missing = rows - held.len();
    if held.is_empty() {
        return Some(PassedOver::Constant {
            value: None,
            missing,
            rows,
        });
    }
    if 2 * missing > rows {
        return Some(PassedOver::Sparse { missing, rows });
    }
    if rows > 1 && held.iter().all(|v| *v == held[0]) {
        return Some(PassedOver::Constant {
            value: first.cloned(),
            missing,
            rows,
        });
    }
    None
}

/// The column of `among` whose values start earliest (`_earliest`), the
/// first in table order on a tie or when none holds a real value: a value
/// before 1901 or the epoch's midnight is a placeholder, no start.
fn earliest(among: &[usize], read: &Read<'_>) -> usize {
    let mut best: Option<(i128, usize)> = None;
    for &i in among {
        let start = read.instants[i]
            .iter()
            .flatten()
            .copied()
            .filter(|&v| v >= FIRST_REAL_DAY && v != 0)
            .min();
        if let Some(start) = start
            && best.is_none_or(|(low, _)| start < low)
        {
            best = Some((start, i));
        }
    }
    best.map_or(among[0], |(_, i)| i)
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
