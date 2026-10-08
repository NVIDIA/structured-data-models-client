// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading a date out of text only where the text says one thing:
//! `sdm_client/core/dates.py`, and the time-column reader built on it
//! (`materializers/relational.py::_to_time`).
//!
//! A column is a list of cells, each text or missing, as an `object`
//! series of strings and `None` is in pandas.

use std::sync::OnceLock;

use regex::Regex;

use crate::dt::{self, Call, Converted, Host, PyErr};
use crate::py;

const ZONE: &str = r"(?:Z|[+-]\d{2}(?::?\d{2})?|[A-Za-z][A-Za-z0-9_+/-]*)";

/// `YEAR_FIRST`, anchored for `fullmatch` and `match`.
fn year_first(anchor_end: bool) -> &'static Regex {
    static WHOLE: OnceLock<Regex> = OnceLock::new();
    static START: OnceLock<Regex> = OnceLock::new();
    let body = format!(
        r"\d{{4}}[-/]\d{{2}}[-/]\d{{2}}([ T]\d{{2}}:\d{{2}}(:\d{{2}}(\.\d+)?)?( ?{ZONE})?)?"
    );
    if anchor_end {
        WHOLE.get_or_init(|| Regex::new(&format!(r"\A(?:{body})\z")).expect("YEAR_FIRST compiles"))
    } else {
        START.get_or_init(|| Regex::new(&format!(r"\A(?:{body})")).expect("YEAR_FIRST compiles"))
    }
}

/// `_TWO_PART_PATTERN`, for `match` (whose `$` also matches before a final
/// newline) or `fullmatch`.
fn two_part(anchor_end: bool) -> &'static Regex {
    static WHOLE: OnceLock<Regex> = OnceLock::new();
    static START: OnceLock<Regex> = OnceLock::new();
    let core = format!(
        r"\A(\d{{1,2}})[-/](\d{{1,2}})[-/]\d{{2,4}}(?:[ T]\d{{1,2}}:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?( ?{ZONE})?)?"
    );
    if anchor_end {
        WHOLE.get_or_init(|| Regex::new(&format!(r"{core}\z")).expect("pattern compiles"))
    } else {
        START.get_or_init(|| Regex::new(&format!(r"{core}\n?\z")).expect("pattern compiles"))
    }
}

fn strip(text: &str) -> String {
    py::strip(text).to_string()
}

/// `str(value)` of a cell, as `astype(str)` writes it.
fn as_str(cell: &Option<String>) -> String {
    match cell {
        Some(text) => text.clone(),
        None => "None".to_string(),
    }
}

/// `reads_as_a_date`: whether each value reads as a date in some reading.
pub fn reads_as_a_date(values: &[Option<String>], host: &Host) -> Vec<bool> {
    let mut read = vec![false; values.len()];
    for dayfirst in [true, false] {
        for mixed in [false, true] {
            let call = Call {
                dayfirst,
                utc: true,
                mixed,
                coerce: true,
            };
            let (converted, _) = dt::to_datetime(values, call, host);
            match converted {
                Ok(converted) => {
                    for (slot, v) in read.iter_mut().zip(&converted.values) {
                        *slot |= v.is_some();
                    }
                }
                Err(error) if error.is_value_type_or_overflow() => continue,
                Err(_) => continue,
            }
        }
    }
    read
}

/// `day_comes_first`.
pub fn day_comes_first(values: &[Option<String>]) -> Option<bool> {
    if values.is_empty() {
        return None;
    }
    let mut first_over = false;
    let mut second_over = false;
    for cell in values {
        let text = strip(&as_str(cell));
        if let Some(found) = two_part(false).captures(&text) {
            let one: i64 = found.get(1).map_or("", |m| m.as_str()).parse().unwrap_or(0);
            let two: i64 = found.get(2).map_or("", |m| m.as_str()).parse().unwrap_or(0);
            first_over |= one > 12;
            second_over |= two > 12;
        }
    }
    if first_over == second_over {
        return None;
    }
    Some(first_over)
}

/// `looks_unambiguous`, with its sample of 1,000.
pub fn looks_unambiguous(values: &[Option<String>]) -> bool {
    let read: Vec<&Option<String>> = values.iter().take(1000).collect();
    if read.is_empty() {
        return false;
    }
    if read.iter().all(|cell| {
        cell.as_deref()
            .is_some_and(|text| year_first(true).is_match(py::strip(text)))
    }) {
        return true;
    }
    day_comes_first(values).is_some()
}

/// `shapes_written`: an example of each shape the column is written in.
pub fn shapes_written(values: &[Option<String>], host: &Host) -> Vec<(&'static str, String)> {
    let text: Vec<String> = values.iter().flatten().map(|t| strip(t)).collect();
    if text.is_empty() {
        return Vec::new();
    }
    let mut year: Vec<Option<String>> = Vec::new();
    let mut other: Vec<Option<String>> = Vec::new();
    for t in &text {
        if year_first(true).is_match(t) {
            year.push(Some(t.clone()));
        } else if two_part(false).is_match(t) {
            other.push(Some(t.clone()));
        }
    }
    let mut found = Vec::new();
    for (name, shaped) in [("year-first", year), ("day-or-month-first", other)] {
        if shaped.is_empty() {
            continue;
        }
        let read = reads_as_a_date(&shaped, host);
        if let Some(i) = read.iter().position(|&r| r) {
            found.push((name, shaped[i].clone().unwrap_or_default()));
        }
    }
    found
}

/// `Conversion`: the arguments that read a column.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Conversion {
    pub utc: bool,
    pub dayfirst: Option<bool>,
    pub mixed: bool,
}

impl Conversion {
    /// `Conversion.arguments`, in order, as JSON.
    pub fn arguments(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut out = serde_json::Map::new();
        if self.utc {
            out.insert("utc".into(), true.into());
        }
        if let Some(d) = self.dayfirst {
            out.insert("dayfirst".into(), d.into());
        }
        if self.mixed {
            out.insert("format".into(), "mixed".into());
        }
        out
    }

    /// `str(conversion)`.
    pub fn text(&self) -> String {
        let mut out = String::new();
        if self.utc {
            out.push_str(", utc=True");
        }
        if let Some(d) = self.dayfirst {
            out.push_str(if d {
                ", dayfirst=True"
            } else {
                ", dayfirst=False"
            });
        }
        if self.mixed {
            out.push_str(", format='mixed'");
        }
        out
    }

    fn call(&self, coerce: bool) -> Call {
        Call {
            dayfirst: self.dayfirst.unwrap_or(false),
            utc: self.utc,
            mixed: self.mixed,
            coerce,
        }
    }
}

/// The year of each value as the column's zone reads it.
fn years(converted: &Converted) -> Vec<i64> {
    (0..converted.values.len())
        .filter_map(|i| converted.wall(i).map(|w| w.year))
        .collect()
}

/// `conversion_that_works`.
pub fn conversion_that_works(
    values: &[Option<String>],
    only_plausible_years: bool,
    per_row: bool,
    host: &Host,
) -> Option<Conversion> {
    let order = day_comes_first(values);
    let modes: &[bool] = if per_row { &[false, true] } else { &[false] };
    for &mixed in modes {
        for utc in [false, true] {
            let candidate = Conversion {
                utc,
                dayfirst: order,
                mixed,
            };
            let (converted, _) = dt::to_datetime(values, candidate.call(true), host);
            let Ok(converted) = converted else {
                continue;
            };
            if converted.values.iter().any(Option::is_none) {
                continue;
            }
            if !only_plausible_years {
                return Some(candidate);
            }
            let ys = years(&converted);
            if let (Some(&low), Some(&high)) = (ys.iter().min(), ys.iter().max())
                && 1900 <= low
                && high <= 2100
            {
                return Some(candidate);
            }
        }
    }
    None
}

/// `read(values, conversion)`.
pub fn read(
    values: &[Option<String>],
    conversion: &Conversion,
    host: &Host,
) -> Result<Converted, PyErr> {
    dt::to_datetime(values, conversion.call(false), host).0
}

/// A refusal the time-column reader gives: `StructuredDataError`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub code: String,
    pub message: String,
}

impl Refusal {
    fn invalid(message: String) -> Self {
        Refusal {
            code: "INVALID_REQUEST".to_string(),
            message,
        }
    }

    fn from_py(error: PyErr) -> Self {
        Refusal {
            code: error.class.to_string(),
            message: error.message,
        }
    }
}

/// What the time-column reader gives: the column as read, or as given.
#[derive(Clone, Debug)]
pub enum TimeColumn {
    /// Every value missing: the column as it was, `object`.
    Untouched(usize),
    Read(Converted),
}

/// `_only_junk_is_unread`.
fn only_junk_is_unread(present: &[Option<String>], host: &Host) -> bool {
    let read = reads_as_a_date(present, host);
    if read.iter().all(|&r| r) {
        return false;
    }
    if !read.iter().any(|&r| r) {
        return looks_unambiguous(present);
    }
    let kept: Vec<Option<String>> = present
        .iter()
        .zip(&read)
        .filter(|(_, r)| **r)
        .map(|(v, _)| v.clone())
        .collect();
    looks_unambiguous(&kept)
}

fn cell_repr(cell: &Option<String>) -> String {
    match cell {
        Some(text) => py::str_repr(text),
        None => "None".to_string(),
    }
}

/// `_to_time(values, where)`.
pub fn time_column(
    values: &[Option<String>],
    place: &str,
    host: &Host,
) -> Result<TimeColumn, Refusal> {
    let present: Vec<Option<String>> = values.iter().filter(|v| v.is_some()).cloned().collect();
    if present.is_empty() {
        return Ok(TimeColumn::Untouched(values.len()));
    }
    let reads = looks_unambiguous(&present);
    if reads && let Some(how) = conversion_that_works(&present, false, false, host) {
        return read(values, &how, host)
            .map(TimeColumn::Read)
            .map_err(Refusal::from_py);
    }
    let shapes = shapes_written(&present, host);
    let column = place.rsplit('.').next().unwrap_or(place);
    if shapes.len() > 1 {
        let unread = reads_as_a_date(&present, host)
            .iter()
            .filter(|r| !**r)
            .count();
        let year_first = &shapes
            .iter()
            .find(|(n, _)| *n == "year-first")
            .map(|(_, e)| e.clone())
            .unwrap_or_default();
        let other = &shapes
            .iter()
            .find(|(n, _)| *n == "day-or-month-first")
            .map(|(_, e)| e.clone())
            .unwrap_or_default();
        let also = if unread > 0 {
            format!(
                " {unread} of its values do not read as a date either, and are a separate thing to fix."
            )
        } else {
            String::new()
        };
        let (y, o) = (py::str_repr(year_first), py::str_repr(other));
        return Err(Refusal::invalid(format!(
            "{place} is the time column and is written two ways: {y} leads with its year and {o} leads with its day or its month. No one reading covers both. The call that reads {o} reads {y} as a different date, so read the halves apart and put them back as one column, or store frame[{}] in one shape before naming it as the time column.{also}",
            py::str_repr(column)
        )));
    }
    if reads && let Some(how) = conversion_that_works(&present, false, true, host) {
        return read(values, &how, host)
            .map(TimeColumn::Read)
            .map_err(Refusal::from_py);
    }
    if shapes.is_empty() || only_junk_is_unread(&present, host) {
        let read = reads_as_a_date(&present, host);
        let unread: Vec<&Option<String>> = present
            .iter()
            .zip(&read)
            .filter(|(_, r)| !**r)
            .map(|(v, _)| v)
            .collect();
        let example = unread.first().copied().unwrap_or(&present[0]);
        if unread.len() == present.len() {
            return Err(Refusal::invalid(format!(
                "{place} is the time column and none of its {} values read as a date, such as {}. A table is windowed by its time column, so this one has no time axis: give it a column that holds dates, or leave it without one and ask a question that does not window it",
                present.len(),
                cell_repr(example)
            )));
        }
        return Err(Refusal::invalid(format!(
            "{place} is the time column and {} of {} of its values do not read as a date, such as {}. Fix or drop those values, or read the column yourself and let them become NaT: frame[{col}] = pd.to_datetime(frame[{col}], errors='coerce'). This client accepts it and says how many rows the model will not see",
            unread.len(),
            present.len(),
            cell_repr(example),
            col = py::str_repr(column)
        )));
    }
    let example = present.iter().find(|v| v.is_some()).unwrap_or(&present[0]);
    let col = py::str_repr(column);
    Err(Refusal::invalid(format!(
        "{place} is the time column and holds text such as {}, which does not say unambiguously what date it is: read day-first it is one date and month-first another, and nothing in the column says which. Convert it yourself so the reading is yours: frame[{col}] = pd.to_datetime(frame[{col}], dayfirst=True), or store it year-first",
        cell_repr(example)
    )))
}

/// `contains_timestamp(pd.Series(values, dtype=object), column, Dtype.string)`
/// of the engine's metadata inference, and the warnings it lets through.
pub fn contains_timestamp(
    values: &[Option<String>],
    column: &str,
    host: &Host,
) -> (Result<bool, PyErr>, Vec<String>) {
    static NAME: OnceLock<Regex> = OnceLock::new();
    let name = NAME.get_or_init(|| {
        Regex::new(r"(?i)(^|_)(date|datetime|dt|time|timedate|timestamp|ts|created|updated)(_|$)")
            .expect("pattern compiles")
    });
    let lowered = column.to_lowercase();
    let mut score: f64 = if name.is_match(&lowered) { 0.3 } else { 0.0 };
    let kept: Vec<Option<String>> = values
        .iter()
        .take(100)
        .filter_map(|v| v.as_ref())
        .filter(|v| !v.is_empty())
        .map(|v| Some(v.clone()))
        .collect();
    if kept.is_empty() {
        return (Ok(false), Vec::new());
    }
    let call = Call {
        coerce: true,
        ..Call::default()
    };
    let (converted, warnings) = dt::to_datetime(&kept, call, host);
    let warnings: Vec<String> = warnings
        .into_iter()
        .filter(|w| !w.starts_with("Could not infer format"))
        .collect();
    match converted {
        Ok(converted) => {
            let hits = converted.values.iter().filter(|v| v.is_some()).count();
            score += hits as f64 / kept.len() as f64;
            (Ok(score >= 1.0), warnings)
        }
        Err(error) => (Err(error), warnings),
    }
}

/// `pattern.fullmatch(value)` or `pattern.match(value)` of `YEAR_FIRST` or
/// `DAY_OR_MONTH_FIRST`.
pub fn pattern_matches(pattern: &str, value: &str, whole: bool) -> Option<bool> {
    match pattern {
        "YEAR_FIRST" => Some(year_first(whole).is_match(value)),
        "DAY_OR_MONTH_FIRST" => Some(two_part(whole).is_match(value)),
        _ => None,
    }
}
