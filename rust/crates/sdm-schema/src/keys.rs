// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The keys counted over every row held to the reference's rules (its
//! client's `_check_source_keys`, T1.32e; the owner moved them out of
//! Phase 2 on 2026-10-13): a key the caller named that holds a null is
//! refused; a key the caller named or the catalog declares whose rows
//! repeat a value in more than 1% of them is refused; an inferred key whose
//! rows are less than 99% distinct is declined, the table left without a
//! key; and a key kept with rows missing a value or repeating one is said.
//! Each count is the answer to the one `key_counts` request the core asks,
//! with a repeated value read for each key that repeats (`repeated_key`),
//! as the reference reads one to show it.

use std::collections::{BTreeMap, BTreeSet};

use arrow_array::RecordBatch;
use sdm_contract::{CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, Scalar};

use crate::infer::InferredTable;
use crate::values;

/// The share of a key's rows that must hold a distinct value for an
/// inferred key to be kept, and above whose complement a named or declared
/// key's repeats are refused (`_KEY_UNIQUENESS`).
pub const KEY_UNIQUENESS: f64 = 0.99;
/// Under this share of distinct values a declined key's repeats are the
/// ordinary many-to-one shape, not a key that missed (`_NEARLY_A_KEY`).
pub const NEARLY_A_KEY: f64 = 0.9;

/// The warning that a key kept has rows with no value for it
/// (`_warn_a_key_has_no_value`).
pub const KEY_NO_VALUE: &str = "KEY_NO_VALUE";
/// The warning that a key kept identifies a row more than once
/// (`_warn_a_key_repeats`).
pub const KEY_REPEATS: &str = "KEY_REPEATS";
/// The warning that a column named like a key, which another table names
/// too, was declined for its repeats (`_warn_a_key_was_declined`).
pub const KEY_DECLINED: &str = "KEY_DECLINED";

/// The key rules' warnings, each with its variants (none: one sentence),
/// which a host words (decision E59(d)).
pub const WARNINGS: &[(&str, &[&str])] =
    &[(KEY_NO_VALUE, &[]), (KEY_REPEATS, &[]), (KEY_DECLINED, &[])];

/// One key's count over every row (the reference's `KeyCount`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Counted {
    pub rows: u64,
    /// The distinct values among the rows holding every part of the key.
    pub distinct: u64,
    /// The rows missing any part of it.
    pub missing: u64,
    /// A value more than one row holds, when the source was asked for one.
    pub example: Option<Scalar>,
}

impl Counted {
    /// The rows holding a key value another row holds already.
    pub fn repeats(&self) -> u64 {
        self.rows
            .saturating_sub(self.missing)
            .saturating_sub(self.distinct)
    }
}

/// What holding the keys to the rules gave: the warnings, in the
/// reference's order, and each table's count of repeated keys, in table
/// order.
#[derive(Debug, Clone, PartialEq)]
pub struct Held {
    pub warnings: Vec<CoreWarning>,
    pub duplicates: Vec<u64>,
}

/// The value of the one-row answer to a `repeated_key` request: none when
/// it holds no row, the key's value when it holds one column (a key of
/// several columns is refused before it is counted).
pub fn example(batches: &[RecordBatch]) -> Option<Scalar> {
    let batch = batches.iter().find(|b| b.num_rows() > 0)?;
    if batch.num_columns() != 1 {
        return None;
    }
    values::scalar(batch.column(0).as_ref(), 0)
}

fn number(n: u64) -> Detail {
    Detail::Scalar(Scalar::UInt64(n))
}

fn shown(example: &Option<Scalar>) -> Detail {
    Detail::Scalar(example.clone().unwrap_or(Scalar::Null))
}

fn warning(code: &str, message: &str, fields: Vec<(&str, Detail)>) -> CoreWarning {
    CoreWarning {
        code: code.to_owned(),
        message: message.to_owned(),
        fields: fields
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect::<BTreeMap<_, _>>(),
    }
}

/// Holds each table's key to the reference's rules, in table order,
/// `counted` each counted table's position and count (a key the source
/// enforces, unique and never null, is not counted, as the reference
/// counts none), `stated` the tables whose key `primary_keys` named, and
/// `infer_links` whether the links are inferred, which is when a declined
/// key is worth a word. A key is declared when the catalog declares it a
/// primary or unique key of its one column and the caller did not name it.
/// Refusals are `INVALID_REQUEST`, naming the field, table and column and
/// the counts, never a value but the repeated example the reference shows.
pub fn hold(
    tables: &mut [InferredTable],
    counted: &[(usize, Counted)],
    stated: &BTreeSet<String>,
    infer_links: bool,
) -> Result<Held, CoreError> {
    let mut duplicates = vec![0; tables.len()];
    let mut warnings = Vec::new();
    let mut declined: Vec<(usize, String, &Counted)> = Vec::new();
    for (at, count) in counted {
        let table = &mut tables[*at];
        let Some(column) = table.primary_key.clone() else {
            continue;
        };
        let repeats = count.repeats();
        let is_stated = stated.contains(&table.name);
        let declared = !is_stated && table.column(&column).is_some_and(|c| c.unique);
        let name = table.name.clone();
        let refused = |message: &'static str| {
            CoreError::new(
                ErrorKind::InvalidRequest,
                ErrorCode::InvalidRequest,
                message,
            )
            .with_field(
                "field",
                Detail::text(if is_stated { "primary_keys" } else { "keys" }),
            )
            .with_field("table", Detail::text(&name))
            .with_field("column", Detail::text(&column))
            .with_field("rows", number(count.rows))
        };
        if !is_stated && !declared {
            // Inferred: declined as inference over a frame declines it.
            if (count.distinct as f64) < count.rows as f64 * KEY_UNIQUENESS {
                table.primary_key = None;
                declined.push((*at, column, count));
                continue;
            }
        } else if count.missing > 0 && is_stated {
            return Err(refused(
                "a primary key named has rows with no value, so those rows cannot be identified",
            )
            .with_field("missing", number(count.missing)));
        } else if repeats as f64 > count.rows as f64 * (1.0 - KEY_UNIQUENESS) {
            let enforced = table
                .unique
                .iter()
                .any(|set| set.len() == 1 && set.contains(&column));
            return Err(
                refused("a primary key is not unique enough to identify a row")
                    .with_field("repeats", number(repeats))
                    .with_field("example", shown(&count.example))
                    .with_field(
                        "declared_unenforced",
                        Detail::Scalar(Scalar::Bool(declared && !enforced)),
                    ),
            );
        }
        if count.missing > 0 {
            warnings.push(warning(
                KEY_NO_VALUE,
                "a key has rows with no value for it",
                vec![
                    ("table", Detail::text(&table.name)),
                    ("column", Detail::text(&column)),
                    ("missing", number(count.missing)),
                ],
            ));
        }
        if repeats > 0 {
            warnings.push(warning(
                KEY_REPEATS,
                "a key identifies a row more than once",
                vec![
                    ("table", Detail::text(&table.name)),
                    ("column", Detail::text(&column)),
                    ("repeats", number(repeats)),
                    ("example", shown(&count.example)),
                ],
            ));
        }
        duplicates[*at] = repeats;
    }
    if infer_links && !declined.is_empty() {
        let keyed_by: BTreeSet<&String> = tables
            .iter()
            .filter_map(|t| t.primary_key.as_ref())
            .collect();
        for (at, column, count) in &declined {
            let elsewhere = tables
                .iter()
                .enumerate()
                .any(|(i, t)| i != *at && t.column(column).is_some());
            if keyed_by.contains(column)
                || !elsewhere
                || (count.distinct as f64) < count.rows as f64 * NEARLY_A_KEY
                || count.repeats() == 0
            {
                continue;
            }
            warnings.push(warning(
                KEY_DECLINED,
                "a column named like a key repeats, so its table gets no key",
                vec![
                    ("table", Detail::text(&tables[*at].name)),
                    ("column", Detail::text(column)),
                    ("repeats", number(count.repeats())),
                    ("rows", number(count.rows)),
                    ("example", shown(&count.example)),
                ],
            ));
        }
    }
    Ok(Held {
        warnings,
        duplicates,
    })
}
