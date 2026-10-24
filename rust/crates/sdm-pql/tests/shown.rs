// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The text the old client's messages quote a value by (T3.5): its `repr`
//! and `str`, held to each value PyYAML read in
//! `contract/view-yaml-vectors.json` with the texts Python gave it.

use sdm_pql::float_repr::{Shown, text_repr};
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

/// A recorded value, its typed form, as the text helpers take it.
fn shown(value: &Json) -> Shown {
    let inner = &value["v"];
    match value["t"].as_str().unwrap() {
        "null" => Shown::None,
        "bool" => Shown::Bool(inner.as_bool().unwrap()),
        "int" => Shown::Int(inner.as_str().unwrap().parse().unwrap()),
        "float" => Shown::Float(inner.as_str().unwrap().parse().unwrap()),
        "str" => Shown::Text(inner.as_str().unwrap().to_owned()),
        "bytes" => Shown::Bytes(debase64(inner.as_str().unwrap())),
        "date" => {
            let (year, month, day) = date(inner.as_str().unwrap());
            Shown::Date { year, month, day }
        }
        "datetime" => datetime(inner.as_str().unwrap()),
        "list" => Shown::List(inner.as_array().unwrap().iter().map(shown).collect()),
        "map" => Shown::Dict(
            inner
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| (shown(&pair[0]), shown(&pair[1])))
                .collect(),
        ),
        other => panic!("{other}"),
    }
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

fn debase64(text: &str) -> Vec<u8> {
    let mut bits = 0u32;
    let (mut count, mut out) = (0, Vec::new());
    for c in text.bytes().filter(|&c| c != b'=') {
        let sextet = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            _ => 63,
        };
        bits = (bits << 6) | u32::from(sextet);
        count += 6;
        if count >= 8 {
            count -= 8;
            out.push(((bits >> count) & 0xFF) as u8);
        }
    }
    out
}

#[test]
fn each_value_is_written_as_python_wrote_it() {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    let vectors = document["vectors"].as_array().unwrap();
    let mut wrong = Vec::new();
    let mut held = 0;
    for vector in vectors.iter().filter(|v| v.get("value").is_some()) {
        held += 1;
        let value = shown(&vector["value"]);
        if value.repr() != vector["repr"].as_str().unwrap() {
            wrong.push(format!("repr {} where {}", value.repr(), vector["repr"]));
        }
        if let Some(text) = vector["str"].as_str()
            && value.text() != text
        {
            wrong.push(format!("str {} where {text}", value.text()));
        }
    }
    assert!(held > 250, "{held} values");
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
