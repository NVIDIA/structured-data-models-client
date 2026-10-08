// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::dt::{Host, civil::Wall};
use crate::engine::infer::{SampleColumn, SampleFrame, SampleKind, to_times};
use crate::frame;
use crate::pyval::PyVal;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub column: Option<String>,
    pub how: Option<String>,
    pub among: Vec<String>,
    pub refused: Vec<Rejected>,
    pub fallback: Option<String>,
    pub unconfirmed: Vec<Rejected>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rejected {
    pub column: String,
    pub reason: String,
    pub value: Option<String>,
    pub missing: usize,
    pub rows: usize,
    pub whole: bool,
}

#[derive(Clone, Debug)]
pub struct Range {
    pub low: PyVal,
    pub high: PyVal,
    pub held: usize,
}

pub struct TableRange {
    pub rows: usize,
    pub columns: BTreeMap<String, Range>,
}

fn tokens(name: &str) -> Vec<String> {
    static TOKEN: OnceLock<Regex> = OnceLock::new();
    let pattern = TOKEN.get_or_init(|| Regex::new(r"[A-Z]?[a-z]+|[A-Z]+|[0-9]+").unwrap());
    let mut separated = String::new();
    let chars: Vec<char> = name.chars().collect();
    for (index, character) in chars.iter().enumerate() {
        if index > 0
            && character.is_ascii_uppercase()
            && (chars[index - 1].is_ascii_lowercase()
                || (chars[index - 1].is_ascii_uppercase()
                    && chars.get(index + 1).is_some_and(char::is_ascii_lowercase)))
        {
            separated.push('_');
        }
        separated.push(*character);
    }
    pattern
        .find_iter(&separated)
        .map(|found| found.as_str().to_lowercase())
        .collect()
}

fn joined(name: &str) -> bool {
    name.chars().all(char::is_alphanumeric)
        && (name == name.to_uppercase() || name == name.to_lowercase())
}

fn reads_as(name: &str, words: &[&str], prefixes: &[(&str, &[&str])]) -> bool {
    let lowered = name.to_lowercase();
    (joined(name)
        && prefixes.iter().any(|(prefix, exceptions)| {
            lowered.strip_prefix(prefix).is_some_and(|rest| {
                !exceptions
                    .iter()
                    .any(|exception| rest.starts_with(exception))
            })
        }))
        || tokens(name)
            .iter()
            .any(|word| words.contains(&word.as_str()))
}

pub fn load_stamp(name: &str) -> bool {
    reads_as(
        name,
        &[
            "synced",
            "sync",
            "loaded",
            "load",
            "reload",
            "reloaded",
            "ingest",
            "ingested",
            "ingestion",
            "batched",
            "extract",
            "extracted",
            "extraction",
            "etl",
            "refresh",
            "refreshdate",
            "refreshed",
        ],
        &[
            ("load", &["ing"]),
            ("reload", &["ing"]),
            ("sync", &["hron"]),
            ("etl", &[]),
            ("refresh", &["ing", "ment"]),
            ("ingest", &[]),
            ("extract", &[]),
            ("batched", &[]),
        ],
    )
}

pub fn birth_date(name: &str) -> bool {
    (joined(name) && name.to_lowercase().contains("birth"))
        || reads_as(
            name,
            &["birth", "birthday", "birthdate", "dob", "born"],
            &[("dob", &[]), ("born", &[])],
        )
}

fn edit_stamp(name: &str) -> bool {
    let lowered = name.to_lowercase();
    lowered.contains("modif")
        || lowered.contains("updat")
        || reads_as(
            name,
            &["last", "edit", "edited", "changed", "deleted", "deletion"],
            &[
                ("last", &["ing", "ly"]),
                ("edit", &["ion", "or"]),
                ("changed", &["ate"]),
                ("deleted", &["ate"]),
                ("deletion", &[]),
            ],
        )
}

fn creation_stamp(name: &str) -> bool {
    name.to_lowercase().starts_with("create")
        || tokens(name)
            .iter()
            .any(|word| ["create", "created", "creation"].contains(&word.as_str()))
}

fn instant(value: &PyVal, host: &Host) -> Option<i128> {
    match value {
        PyVal::Timestamp(wall) => Some(wall.epoch_nanos()),
        PyVal::TimestampTz(wall, offset) => {
            Some(wall.epoch_nanos() - i128::from(*offset) * 1_000_000_000)
        }
        PyVal::DateTime(wall, offset) => {
            Some(wall.epoch_nanos() - i128::from(offset.unwrap_or(0)) * 1_000_000_000)
        }
        PyVal::Date(year, month, day) => Some(Wall::date(*year, *month, *day).epoch_nanos()),
        PyVal::Numpy(_, inner) | PyVal::Held { value: inner, .. } => instant(inner, host),
        PyVal::Str(_) => to_times(
            &SampleColumn {
                name: String::new(),
                kind: SampleKind::Object,
                cells: vec![value.clone()],
            },
            host,
        )
        .into_iter()
        .next()
        .flatten(),
        _ => None,
    }
}

fn real(value: i128) -> bool {
    value >= Wall::date(1901, 1, 1).epoch_nanos() && value != 0
}

fn value_text(value: &PyVal, host: &Host) -> String {
    match value {
        PyVal::Str(_) | PyVal::Date(..) => instant(value, host)
            .map(|stamp| PyVal::Timestamp(Wall::from_epoch_nanos(stamp)).str_())
            .unwrap_or_else(|| value.str_()),
        _ => value.str_(),
    }
}

fn grouped(value: usize) -> String {
    let digits = value.to_string();
    digits
        .chars()
        .enumerate()
        .fold(String::new(), |mut text, (index, digit)| {
            if index > 0 && (digits.len() - index).is_multiple_of(3) {
                text.push(',');
            }
            text.push(digit);
            text
        })
}

impl Rejected {
    pub fn why(&self) -> String {
        match self.reason.as_str() {
            "edit_stamp" => {
                return "it looks like when a row was last changed, not when it began".into();
            }
            "load_stamp" => return "it looks like when a row was loaded, not when it began".into(),
            "birth_date" => return "it looks like a date of birth, not when a row began".into(),
            _ => {}
        }
        let rows = grouped(self.rows);
        let missing = grouped(self.missing);
        if self.reason == "sparse" {
            return if self.whole {
                format!("{missing} of its {rows} rows hold no value")
            } else {
                format!("{missing} of the {rows} rows read hold no value")
            };
        }
        let Some(value) = &self.value else {
            return if self.whole {
                "no row holds a value".into()
            } else {
                format!("none of the {rows} rows read holds a value")
            };
        };
        let beginning = if self.whole {
            "every row".into()
        } else {
            format!("every one of the {rows} rows read")
        };
        if self.missing > 0 {
            format!("{beginning} that holds a value holds {value}")
        } else {
            format!("{beginning} holds {value}")
        }
    }

    pub fn taken_unchecked(&self, table: &str, clear: &str) -> String {
        format!(
            "{table}.{} was taken as the time column although {}, because the query over the whole table failed and the rows past them were not checked. If the whole table is like that, clear it: {clear}.",
            self.column,
            self.why()
        )
    }
}

impl Choice {
    pub fn for_links(&self, linked: bool) -> Self {
        let mut settled = self.clone();
        if let Some(fallback) = &self.fallback {
            settled.column = (!linked).then(|| fallback.clone());
            settled.how = (!linked).then(|| "edit_stamp".to_string());
            settled.among = settled.column.iter().cloned().collect();
        }
        settled
    }
}

pub fn choose(
    frame: &SampleFrame,
    candidates: &[String],
    whole: Option<&TableRange>,
    attempted: bool,
    host: &Host,
) -> Choice {
    let mut choice = Choice::default();
    let mut plain = Vec::new();
    let mut doubted = Vec::new();
    let mut edited = Vec::new();
    for name in candidates {
        let named_reason = if load_stamp(name) {
            Some("load_stamp")
        } else if birth_date(name) {
            Some("birth_date")
        } else {
            None
        };
        let mut refusal = Rejected {
            column: name.clone(),
            reason: String::new(),
            value: None,
            missing: 0,
            rows: 0,
            whole: true,
        };
        if let Some(reason) = named_reason {
            refusal.reason = reason.into();
            choice.refused.push(refusal);
            continue;
        }
        let cells = frame.column(name).map_or(&[][..], |column| &column.cells);
        let present: Vec<&PyVal> = cells
            .iter()
            .filter(|value| !frame::missing(value))
            .collect();
        let mut first = present.first().copied();
        let (rows, held, same) =
            match whole.and_then(|table| table.columns.get(name).map(|range| (table, range))) {
                Some((table, range)) => {
                    if first.is_none() {
                        first = Some(&range.low);
                    }
                    (
                        table.rows,
                        range.held,
                        range.held > 0 && range.low.eq_key() == range.high.eq_key(),
                    )
                }
                None => (
                    frame.len(),
                    present.len(),
                    first.is_some_and(|first| {
                        present.iter().all(|value| value.eq_key() == first.eq_key())
                    }),
                ),
            };
        refusal.rows = rows;
        refusal.missing = rows.saturating_sub(held);
        refusal.whole = !attempted || whole.is_some_and(|table| table.columns.contains_key(name));
        if held == 0 {
            refusal.reason = "constant".into();
        } else if refusal.missing * 2 > rows {
            refusal.reason = "sparse".into();
        } else if rows > 1 && same {
            refusal.reason = "constant".into();
            refusal.value = first.map(|value| value_text(value, host));
        }
        let refused = !refusal.reason.is_empty();
        if refused && refusal.whole {
            choice.refused.push(refusal);
            continue;
        }
        if refused {
            choice.unconfirmed.push(refusal);
        }
        if edit_stamp(name) {
            choice.refused.push(Rejected {
                column: name.clone(),
                reason: "edit_stamp".into(),
                value: None,
                missing: 0,
                rows: 0,
                whole: true,
            });
            edited.push(name.clone());
        } else if refused {
            doubted.push(name.clone());
        } else {
            plain.push(name.clone());
        }
    }
    if plain.is_empty() {
        plain = doubted;
    }
    if plain.is_empty() {
        if edited.len() == 1 {
            choice.fallback = edited.pop();
        }
        return choice.for_links(false);
    }
    let created: Vec<String> = plain
        .iter()
        .filter(|name| creation_stamp(name))
        .cloned()
        .collect();
    if created.len() == 1 {
        choice.column = created.first().cloned();
        choice.how = Some(if plain.len() > 1 { "created" } else { "only" }.into());
        choice.among = plain;
        return choice;
    }
    choice.among = if created.is_empty() { plain } else { created };
    choice.how = Some(
        if choice.among.len() == 1 {
            "only"
        } else {
            "earliest"
        }
        .into(),
    );
    choice.column = choice.among.first().cloned();
    let mut earliest = None;
    for name in &choice.among {
        let start = whole
            .and_then(|table| table.columns.get(name))
            .and_then(|range| instant(&range.low, host))
            .filter(|stamp| real(*stamp))
            .or_else(|| {
                frame.column(name).and_then(|column| {
                    column
                        .cells
                        .iter()
                        .filter_map(|value| instant(value, host))
                        .filter(|stamp| real(*stamp))
                        .min()
                })
            });
        if let Some(start) = start
            && earliest.is_none_or(|previous| start < previous)
        {
            earliest = Some(start);
            choice.column = Some(name.clone());
        }
    }
    choice
}

pub fn from_frame(data: &frame::Frame, candidates: &[String], host: &Host) -> Choice {
    let sample = SampleFrame {
        columns: data
            .columns
            .iter()
            .map(|column| SampleColumn {
                name: column.name.clone(),
                kind: SampleKind::Object,
                cells: column.cells.clone(),
            })
            .collect(),
        index: data.rows,
    };
    choose(&sample, candidates, None, false, host)
}
