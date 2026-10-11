// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_engine::time::text` against pandas, from `text-times.json`,
//! recorded from pandas 3.0.6 by `text_times_record.py` (a protected path:
//! never recorded again to pass a dependency bump, F4): columns of text
//! over `date-rules.md`'s shapes, junk, codes and the T0.6 corpus, each read
//! as the reference's engine reads it for inference (`coerce`), for the
//! time column's choice (`utc`) and value by value (`each`, DT-7).
//!
//! Wherever the reader decides, its answer is pandas': each instant, each
//! missing time, and the refusal of mixed offsets. Where it does not
//! (`Unread::Undecided`), the reading takes a rule outside the shapes and
//! the core refuses it by name. [`check`] fails closed: a perturbed
//! answer, and evidence missing, empty or retyped, each fail (seeded
//! below), and so does a run deciding fewer columns than recorded here.

use sdm_engine::time::text::{Reader, Unread};
use serde_json::{Value, json};

const VECTORS: &str = include_str!("text-times.json");

/// What a run compared: readings decided and agreeing, and undecided;
/// the bounds `never_read` gave, each held to pandas' reading; and the
/// day-first formats pandas warned of, where the reading decided.
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    agreed: usize,
    undecided: usize,
    bounded: usize,
    day_first: usize,
}

fn values(column: &Value, at: &str) -> Result<Vec<Option<String>>, String> {
    let items = column
        .as_array()
        .filter(|items| !items.is_empty())
        .ok_or_else(|| format!("{at}: no values"))?;
    items
        .iter()
        .map(|v| match v {
            Value::Null => Ok(None),
            Value::String(s) => Ok(Some(s.clone())),
            _ => Err(format!("{at}: a value is neither text nor null")),
        })
        .collect()
}

/// pandas' reading as the reader writes one: each instant in nanoseconds.
fn expected(reading: &Value, at: &str) -> Result<Result<Vec<Option<i128>>, &'static str>, String> {
    match reading {
        Value::String(s) if s == "MIXED" => Ok(Err("MIXED")),
        Value::String(s) if s == "ERR" => Ok(Err("ERR")),
        Value::String(s) if s == "CLOCK" => Ok(Err("CLOCK")),
        Value::Array(items) => items
            .iter()
            .map(|v| match v {
                Value::Null => Ok(None),
                Value::Number(n) => n
                    .as_i64()
                    .map(|us| Some(i128::from(us) * 1_000))
                    .ok_or_else(|| format!("{at}: not an integer")),
                _ => Err(format!("{at}: neither an instant nor null")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Ok),
        _ => Err(format!("{at}: not a reading")),
    }
}

fn check(document: &Value) -> Result<Counts, String> {
    if document["format"] != "sdm-text-times/1" {
        return Err("not an sdm-text-times/1 document".into());
    }
    let year = document["year"].as_i64().ok_or("no year")?;
    let reader = Reader::new(year);
    let columns = document["columns"]
        .as_array()
        .filter(|c| !c.is_empty())
        .ok_or("no columns")?;
    let mut counts = Counts::default();
    let mut differences = Vec::new();
    for (i, column) in columns.iter().enumerate() {
        let owned = values(&column["values"], &format!("column {i}"))?;
        let given: Vec<Option<&str>> = owned.iter().map(|v| v.as_deref()).collect();
        // The values that never read: pandas reads no more than the rest,
        // and does not refuse the column for its offsets.
        if let Some(never) = reader.never_read(&given) {
            let want = expected(&column["coerce"], &format!("column {i} coerce"))?;
            let present = given.iter().flatten().count();
            match want {
                Ok(read) if read.iter().flatten().count() <= present - never => {
                    counts.bounded += 1;
                }
                // The clock's time, which reads, and holds no offset.
                Err("CLOCK") => {}
                _ => differences.push(format!(
                    "column {i} never_read: {given:?}: {never} never read, pandas {want:?}"
                )),
            }
        }
        // The day-first format pandas warned of, where the inference's
        // reading decides.
        if !matches!(reader.column(&given, false), Err(Unread::Undecided)) {
            let warned = match &column["day_first"] {
                Value::Null => None,
                Value::String(format) => Some(format.clone()),
                _ => {
                    return Err(format!(
                        "column {i}: day_first is neither a format nor null"
                    ));
                }
            };
            let guessed = reader.day_first_format(&given);
            if guessed != warned {
                differences.push(format!(
                    "column {i} day_first: {given:?}: pandas {warned:?}, reader {guessed:?}"
                ));
            } else if warned.is_some() {
                counts.day_first += 1;
            }
        }
        for kind in ["coerce", "utc", "each"] {
            let at = format!("column {i} {kind}");
            let want = expected(&column[kind], &at)?;
            if let Ok(want) = &want
                && want.len() != given.len()
            {
                return Err(format!("{at}: a reading of another length"));
            }
            let got = match kind {
                "coerce" => reader.column(&given, false),
                "utc" => reader.column(&given, true),
                _ => reader.each(&given),
            };
            let agrees = match (&got, &want) {
                (Err(Unread::Undecided), _) => {
                    counts.undecided += 1;
                    continue;
                }
                (Err(Unread::MixedZones), Err("MIXED")) => true,
                (Ok(read), Ok(want)) => {
                    read.iter()
                        .map(|at| at.map(|at| at.nanos()))
                        .collect::<Vec<_>>()
                        == *want
                }
                _ => false,
            };
            if agrees {
                counts.agreed += 1;
            } else {
                differences.push(format!("{at}: {given:?}: pandas {want:?}, reader {got:?}"));
            }
        }
    }
    if differences.is_empty() {
        Ok(counts)
    } else {
        Err(format!(
            "{} differ: {:#?}",
            differences.len(),
            &differences[..differences.len().min(20)]
        ))
    }
}

fn document() -> Value {
    serde_json::from_str(VECTORS).unwrap()
}

/// The readings the reader decided when the vectors were recorded: a
/// change that decides fewer is seen, and one that decides more is a
/// change to record here.
const DECIDED: usize = 18_080;
/// The columns `never_read` bounded when the vectors were recorded.
const BOUNDED: usize = 404;
/// The day-first formats pandas warned of where the reading decided.
const DAY_FIRST: usize = 219;

#[test]
fn the_reader_reads_text_as_pandas_does_wherever_it_decides() {
    let counts = check(&document()).unwrap();
    assert_eq!(counts.agreed, DECIDED, "{counts:?}");
    assert!(counts.undecided > 0);
    assert_eq!(counts.bounded, BOUNDED, "{counts:?}");
    assert_eq!(counts.day_first, DAY_FIRST, "{counts:?}");
}

#[test]
fn the_check_fails_closed() {
    let doc = document();
    // A perturbed instant.
    let mut perturbed = doc.clone();
    let column = perturbed["columns"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| c["values"] == json!(["2024-01-05"]))
        .unwrap();
    column["coerce"][0] = json!(column["coerce"][0].as_i64().unwrap() + 1);
    assert!(check(&perturbed).unwrap_err().contains("differ"));
    // A reading pandas refused for its mixed offsets, said read.
    let mut refused = doc.clone();
    let reader = Reader::new(doc["year"].as_i64().unwrap());
    let column = refused["columns"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|c| {
            let owned = values(&c["values"], "").unwrap();
            let given: Vec<Option<&str>> = owned.iter().map(|v| v.as_deref()).collect();
            reader.column(&given, false) == Err(Unread::MixedZones)
        })
        .unwrap();
    column["coerce"] = json!(vec![
        Value::Null;
        column["values"].as_array().unwrap().len()
    ]);
    assert!(check(&refused).unwrap_err().contains("differ"));
    // Evidence missing, empty or retyped.
    for broken in [
        json!({"format": "sdm-text-times/1", "year": 2026}),
        json!({"format": "sdm-text-times/1", "year": 2026, "columns": []}),
        json!({"format": "sdm-text-times/1", "year": 2026, "columns": [{"values": []}]}),
        json!({"format": "sdm-text-times/1", "year": 2026, "columns": [{"values": ["2024-01-05"]}]}),
        json!({"format": "sdm-text-times/1", "year": 2026, "columns": [{"values": ["2024-01-05"], "coerce": "read", "utc": [1], "each": [1]}]}),
        json!({"format": "sdm-text-times/1", "year": 2026, "columns": [{"values": [5], "coerce": [1], "utc": [1], "each": [1]}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": []}),
        json!({"format": "sdm-text-times/1", "columns": []}),
    ] {
        assert!(check(&broken).is_err(), "{broken}");
    }
}
