// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The text the old client's messages quote a value by (T3.5): its `repr`
//! and `str`, held to each value PyYAML read in
//! `contract/view-yaml-vectors.json` with the texts Python gave it.

use sdm_pql::float_repr::{Shown, text_repr};
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

/// A recorded value, its typed form, as the text helpers take it; `open`
/// the kinds of the collections it is inside (a list `true`). `None` for a
/// value outside the YAML view definitions use (a byte string, a set, a
/// list of pairs), which the core refuses (register P3J-9).
fn shown(value: &Json, open: &mut Vec<bool>) -> Option<Shown> {
    let inner = &value["v"];
    let nested =
        |kind: bool, open: &mut Vec<bool>, items: &dyn Fn(&mut Vec<bool>) -> Option<Shown>| {
            open.push(kind);
            let shown = items(open);
            open.pop();
            shown
        };
    Some(match value["t"].as_str().unwrap() {
        "bytes" | "tuple" | "set" => return None,
        "null" => Shown::None,
        "bool" => Shown::Bool(inner.as_bool().unwrap()),
        "int" => Shown::Int(inner.as_str().unwrap().to_owned()),
        "float" => Shown::Float(inner.as_str().unwrap().parse().unwrap()),
        "str" => Shown::Text(inner.as_str().unwrap().to_owned()),
        "date" => {
            let (year, month, day) = date(inner.as_str().unwrap());
            Shown::Date { year, month, day }
        }
        "datetime" => datetime(inner.as_str().unwrap()),
        "cycle" => {
            let up = usize::try_from(inner.as_u64().unwrap()).unwrap();
            Shown::Recursive(open[open.len() - up])
        }
        "list" => nested(true, open, &|open| {
            Some(Shown::List(
                inner
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| shown(v, open))
                    .collect::<Option<_>>()?,
            ))
        })?,
        "map" => nested(false, open, &|open| {
            Some(Shown::Dict(
                inner
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| Some((shown(&pair[0], open)?, shown(&pair[1], open)?)))
                    .collect::<Option<_>>()?,
            ))
        })?,
        other => panic!("{other}"),
    })
}

fn date(text: &str) -> (i32, u8, u8) {
    let mut parts = text.split('-');
    (
        parts.next().unwrap().parse().unwrap(),
        parts.next().unwrap().parse().unwrap(),
        parts.next().unwrap().parse().unwrap(),
    )
}

/// `datetime.isoformat()`'s text, read back.
fn datetime(text: &str) -> Shown {
    let (day, time) = text.split_once('T').unwrap();
    let (year, month, day) = date(day);
    let (clock, offset) = match time.find(['+', '-']) {
        Some(at) => {
            let (clock, zone) = time.split_at(at);
            let sign = if zone.starts_with('-') { -1 } else { 1 };
            let mut parts = zone[1..].split(':').map(|p| p.parse::<i32>().unwrap());
            let seconds = parts.next().unwrap() * 3600
                + parts.next().unwrap_or(0) * 60
                + parts.next().unwrap_or(0);
            (clock, Some(sign * seconds))
        }
        None => (time, None),
    };
    let (clock, microsecond) = match clock.split_once('.') {
        Some((clock, fraction)) => (clock, fraction.parse().unwrap()),
        None => (clock, 0),
    };
    let mut parts = clock.split(':').map(|p| p.parse::<u8>().unwrap());
    Shown::DateTime {
        year,
        month,
        day,
        hour: parts.next().unwrap(),
        minute: parts.next().unwrap(),
        second: parts.next().unwrap(),
        microsecond,
        offset,
    }
}

#[test]
fn each_value_is_written_as_python_wrote_it() {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    let vectors = document["vectors"].as_array().unwrap();
    let mut wrong = Vec::new();
    let mut held = 0;
    // A value holding a set has no recorded text: a set's order varies by
    // run in Python.
    for vector in vectors.iter().filter(|v| v.get("repr").is_some()) {
        let Some(value) = shown(&vector["value"], &mut Vec::new()) else {
            continue;
        };
        held += 1;
        if value.repr() != vector["repr"].as_str().unwrap() {
            wrong.push(format!("repr {} where {}", value.repr(), vector["repr"]));
        }
        if let Some(text) = vector["str"].as_str()
            && value.text() != text
        {
            wrong.push(format!("str {} where {text}", value.text()));
        }
    }
    assert!(held > 300, "{held} values");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Python's `repr` of a text: its quotes, escapes, and characters it
/// does not print.
#[test]
fn a_text_is_quoted_as_python_quotes_it() {
    assert_eq!(text_repr("a"), "'a'");
    assert_eq!(text_repr("it's"), "\"it's\"");
    assert_eq!(text_repr("it's \"x\""), "'it\\'s \"x\"'");
    assert_eq!(text_repr("a\\b\tc\n"), "'a\\\\b\\tc\\n'");
    assert_eq!(text_repr("é"), "'é'");
    assert_eq!(text_repr("\u{85}"), "'\\x85'");
    assert_eq!(text_repr("\u{2028}"), "'\\u2028'");
    assert_eq!(text_repr("\u{E000}"), "'\\ue000'");
    assert_eq!(text_repr("$$"), "'$$'");
    assert_eq!(text_repr("'"), "\"'\"");
}

/// A collection inside itself, as Python writes it.
#[test]
fn cycles_are_written_as_python_writes_them() {
    let text = |t: &str| Shown::Text(t.into());
    assert_eq!(Shown::List(vec![Shown::Recursive(true)]).repr(), "[[...]]");
    assert_eq!(
        Shown::Dict(vec![(text("k"), Shown::Recursive(false))]).repr(),
        "{'k': {...}}"
    );
    let big = "1".repeat(50);
    assert_eq!(Shown::Int(big.clone()).text(), big);
}

/// A text is written up to its bound exactly: one character over, it is
/// refused, and nothing past the bound is built (Codex review 3, P1-4).
#[test]
fn a_text_is_written_up_to_its_bound_exactly() {
    let value = Shown::List(vec![
        Shown::Text("é".repeat(3)),
        Shown::Text("\0".into()),
        Shown::Date {
            year: 2024,
            month: 1,
            day: 1,
        },
    ]);
    let repr = value.repr();
    assert_eq!(repr, "['ééé', '\\x00', datetime.date(2024, 1, 1)]");
    let length = repr.chars().count();
    assert_eq!(value.repr_within(length).as_deref(), Some(repr.as_str()));
    assert_eq!(value.repr_within(length - 1), None);
    let text = Shown::Text("é".repeat(5));
    assert_eq!(text.text_within(5).as_deref(), Some("ééééé"));
    assert_eq!(text.text_within(4), None);
}

/// Python 3.12 escapes an unassigned code point, a format character and a
/// soft hyphen in a `repr` (Codex review 3, P0-2).
#[test]
fn python_s_printability_is_its_own_table() {
    assert_eq!(text_repr("\u{378}"), "'\\u0378'");
    assert_eq!(text_repr("\u{e0001}"), "'\\U000e0001'");
    assert_eq!(text_repr("\u{ad}"), "'\\xad'");
    assert_eq!(text_repr("\u{2066}"), "'\\u2066'");
    assert_eq!(text_repr("\u{1F600}"), "'\u{1F600}'");
}
