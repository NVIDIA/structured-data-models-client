// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_engine::time::text` against its own spec, from `text-times.json`,
//! recorded by `text_times_record.py` (a protected path: never recorded
//! again to pass a dependency bump, F4), which reads `date-rules.md`'s "The
//! core's text formats" apart from the Rust reader, with Python's `re` and
//! `datetime` and no pandas (T1.25b): columns over every format of the
//! list, values outside it, the orders of day and month, two-digit years at
//! the pivot and offsets mixed, each read as the core reads text: the
//! values (`column`), to UTC (`utc`), the share inference scores
//! (`share`) and the day-first format it warns of (`day_first`).
//!
//! Every reading is compared, none skipped. [`check`] fails closed: a
//! perturbed answer, and evidence missing, empty or retyped, each fail
//! (seeded below), and so does a run comparing fewer columns than
//! recorded.

use sdm_engine::time::text::{Reader, Share, Unread};
use serde_json::{Value, json};

const VECTORS: &str = include_str!("text-times.json");

/// The columns recorded, each read four ways.
const COLUMNS: usize = 2_360;

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

/// A recorded reading of the values: each instant in nanoseconds, or the
/// refusal named.
fn expected(reading: &Value, at: &str) -> Result<Result<Vec<Option<i128>>, Unread>, String> {
    match reading {
        Value::String(s) => match s.as_str() {
            "outside" => Ok(Err(Unread::Outside)),
            "order" => Ok(Err(Unread::Order)),
            "mixed" => Ok(Err(Unread::MixedZones)),
            _ => Err(format!("{at}: not a reading")),
        },
        Value::Array(items) => items
            .iter()
            .map(|v| match v {
                Value::Null => Ok(None),
                Value::String(n) => n
                    .parse::<i128>()
                    .map(Some)
                    .map_err(|_| format!("{at}: not an instant")),
                _ => Err(format!("{at}: neither an instant nor null")),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Ok),
        _ => Err(format!("{at}: not a reading")),
    }
}

fn expected_share(share: &Value, at: &str) -> Result<Result<Share, Unread>, String> {
    match share {
        Value::String(s) if s == "mixed" => Ok(Err(Unread::MixedZones)),
        Value::Array(counts) => match counts[..] {
            [
                Value::Number(ref a),
                Value::Number(ref l),
                Value::Number(ref b),
                Value::Number(ref c),
            ] => {
                let count = |n: &serde_json::Number| {
                    n.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| format!("{at}: not a count"))
                };
                Ok(Ok(Share {
                    fewest: count(a)?,
                    listed: count(l)?,
                    most: count(b)?,
                    unshaped: count(c)?,
                }))
            }
            _ => Err(format!("{at}: not a share")),
        },
        _ => Err(format!("{at}: not a share")),
    }
}

/// The columns compared, or every difference found.
fn check(document: &Value) -> Result<usize, String> {
    if document["format"] != "sdm-text-times/2" {
        return Err("not an sdm-text-times/2 document".into());
    }
    let year = document["year"].as_i64().ok_or("no year")?;
    let reader = Reader::new(year);
    let columns = document["columns"]
        .as_array()
        .filter(|c| !c.is_empty())
        .ok_or("no columns")?;
    let mut differences = Vec::new();
    for (i, column) in columns.iter().enumerate() {
        let owned = values(&column["values"], &format!("column {i}"))?;
        let given: Vec<Option<&str>> = owned.iter().map(|v| v.as_deref()).collect();
        for (kind, utc) in [("column", false), ("utc", true)] {
            let at = format!("column {i} {kind}");
            let want = expected(&column[kind], &at)?;
            if let Ok(want) = &want
                && want.len() != given.len()
            {
                return Err(format!("{at}: a reading of another length"));
            }
            let got = reader
                .column(&given, utc)
                .map(|read| read.into_iter().map(|at| at.map(|at| at.nanos())).collect());
            if got != want {
                differences.push(format!("{at}: {given:?}: want {want:?}, got {got:?}"));
            }
        }
        let at = format!("column {i} share");
        let want = expected_share(&column["share"], &at)?;
        let got = reader.share(&given);
        if got != want {
            differences.push(format!("{at}: {given:?}: want {want:?}, got {got:?}"));
        }
        let want = match &column["day_first"] {
            Value::Null => None,
            Value::String(format) => Some(format.clone()),
            _ => {
                return Err(format!(
                    "column {i}: day_first is neither a format nor null"
                ));
            }
        };
        let got = reader.day_first_format(&given);
        if got != want {
            differences.push(format!(
                "column {i} day_first: {given:?}: want {want:?}, got {got:?}"
            ));
        }
        // `each` is the column's reading not to UTC.
        if reader.each(&given) != reader.column(&given, false) {
            differences.push(format!("column {i} each: {given:?}"));
        }
    }
    if differences.is_empty() {
        Ok(columns.len())
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

#[test]
fn the_reader_reads_text_as_its_spec_says() {
    assert_eq!(check(&document()), Ok(COLUMNS));
}

/// Every kind of answer is held somewhere: no vector set that only reads,
/// or only refuses, passes for the whole.
#[test]
fn the_vectors_hold_every_answer() {
    let doc = document();
    let columns = doc["columns"].as_array().unwrap();
    for kind in ["outside", "order", "mixed"] {
        assert!(columns.iter().any(|c| c["column"] == kind), "{kind}");
    }
    assert!(
        columns
            .iter()
            .any(|c| c["utc"].is_array() && c["column"] == "mixed")
    );
    assert!(columns.iter().any(|c| c["day_first"].is_string()));
    assert!(columns.iter().any(|c| {
        c["column"]
            .as_array()
            .is_some_and(|r| r.iter().any(Value::is_null))
    }));
    assert!(columns.iter().any(|c| c["share"][0] != c["share"][1]));
    assert!(columns.iter().any(|c| c["share"][2].as_u64() > Some(0)));
}

#[test]
fn the_check_fails_closed() {
    let doc = document();
    let find = |doc: &mut Value, values: Value| -> usize {
        doc["columns"]
            .as_array()
            .unwrap()
            .iter()
            .position(|c| c["values"] == values)
            .unwrap_or_else(|| panic!("{values} is recorded"))
    };
    // A perturbed instant.
    let mut perturbed = doc.clone();
    let i = find(&mut perturbed, json!(["2024-01-05"]));
    let n: i128 = perturbed["columns"][i]["column"][0]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    perturbed["columns"][i]["column"][0] = json!((n + 1).to_string());
    assert!(check(&perturbed).unwrap_err().contains("differ"));
    // A refusal said read, a read said refused, another refusal named.
    for (values, field, said) in [
        (json!(["March 3 2020"]), "column", json!([null])),
        (json!(["2024-01-05"]), "utc", json!("outside")),
        (json!(["01/02/2024"]), "column", json!("outside")),
        (
            json!(["2024-01-05", "AW00011205"]),
            "share",
            json!([1, 1, 2, 0]),
        ),
        (json!(["15/03/2024"]), "day_first", Value::Null),
    ] {
        let mut changed = doc.clone();
        let i = find(&mut changed, values.clone());
        changed["columns"][i][field] = said;
        assert!(
            check(&changed).unwrap_err().contains("differ"),
            "{values} {field}"
        );
    }
    // Evidence missing, empty or retyped.
    for broken in [
        json!({"format": "sdm-text-times/2", "year": 2026}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": []}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": []}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": ["2024-01-05"]}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": ["2024-01-05"], "column": "read", "utc": ["1"], "share": [1, 1, 1, 0], "day_first": null}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": ["2024-01-05"], "column": [1], "utc": ["1"], "share": [1, 1, 1, 0], "day_first": null}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": [5], "column": ["1"], "utc": ["1"], "share": [1, 1, 1, 0], "day_first": null}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": ["2024-01-05"], "column": ["1704412800000000000"], "utc": ["1704412800000000000"], "share": [1, 1, 1], "day_first": null}]}),
        json!({"format": "sdm-text-times/2", "year": 2026, "columns": [{"values": ["2024-01-05"], "column": ["1704412800000000000"], "utc": ["1704412800000000000"], "share": [1, 1, 1, 0], "day_first": 5}]}),
        json!({"format": "sdm-text-times/1", "year": 2026, "columns": []}),
        json!({"format": "sdm-text-times/2", "columns": []}),
    ] {
        assert!(check(&broken).is_err(), "{broken}");
    }
    // A run comparing fewer columns than recorded.
    let mut fewer = doc.clone();
    fewer["columns"].as_array_mut().unwrap().pop();
    assert_ne!(check(&fewer), Ok(COLUMNS));
}

const DIFFERENTIAL: &str = include_str!("text-times-differential.json");

/// The core's answer for one of the report's examples, as the report
/// writes it.
fn core_answer(reader: &Reader, outcome: &str, example: &Value) -> Result<Value, String> {
    let owned = values(&example["values"], outcome)?;
    let given: Vec<Option<&str>> = owned.iter().map(|v| v.as_deref()).collect();
    let reading = |read: Result<Vec<Option<sdm_engine::time::Instant>>, Unread>| match read {
        Ok(read) => json!(
            read.into_iter()
                .map(|at| at.map(|at| at.nanos().to_string()))
                .collect::<Vec<_>>()
        ),
        Err(Unread::Outside) => json!("outside"),
        Err(Unread::Order) => json!("order"),
        Err(Unread::MixedZones) => json!("mixed"),
    };
    Ok(match outcome {
        "materialize" => reading(reader.each(&given)),
        "time_choice" => reading(reader.column(&given, true)),
        "inference" => {
            let named = example["named"].as_bool().ok_or("no name's weight")?;
            let first: Vec<Option<&str>> = given
                .iter()
                .flatten()
                .filter(|v| !v.is_empty())
                .take(100)
                .map(|v| Some(*v))
                .collect();
            let score =
                |n: usize| f64::from(u8::from(named)) * 0.3 + n as f64 / first.len() as f64 >= 1.0;
            match reader.share(&first) {
                Ok(share) if score(share.fewest) != score(share.listed) => {
                    if share.most > share.listed || share.unshaped > 0 {
                        json!("text_time")
                    } else {
                        json!("text_time_order")
                    }
                }
                Ok(share)
                    if score(share.fewest)
                        == score(share.most + if named { share.unshaped } else { 0 }) =>
                {
                    json!(score(share.fewest))
                }
                Ok(_) => json!("text_time"),
                Err(_) => json!("text_time_zones"),
            }
        }
        "day_first" => json!(reader.day_first_format(&given)),
        _ => return Err(format!("{outcome}: not an outcome")),
    })
}

/// Whether a class `a` answer is made by a value outside the list in both
/// orders of day and month (DT-15): the reading refuses it so, or, for
/// inference, the refusal is `text_time` (an order the values in the list
/// leave open is `text_time_order`, DT-16) and its values hold such a value.
fn made_outside(reader: &Reader, outcome: &str, example: &Value, core: &Value) -> bool {
    if core == "outside" {
        return true;
    }
    if outcome != "inference" || core != "text_time" {
        return false;
    }
    let Ok(owned) = values(&example["values"], outcome) else {
        return false;
    };
    let first: Vec<Option<&str>> = owned
        .iter()
        .flatten()
        .filter(|v| !v.is_empty())
        .take(100)
        .map(|v| Some(v.as_str()))
        .collect();
    reader.column(&first, false) == Err(Unread::Outside)
}

/// An order of day and month two values in the list leave open is never
/// counted as made outside the list (T1.25b fix 1, the review's
/// reproduction): its refusal is DT-16's, not DT-15's.
#[test]
fn an_open_order_in_the_list_is_not_a_refusal_outside_it() {
    let reader = Reader::new(2026);
    for named in [true, false] {
        let example = json!({"values": ["03/15/99", "15/11/00"], "named": named});
        let core = core_answer(&reader, "inference", &example).unwrap();
        assert_eq!(core, "text_time_order");
        assert!(
            !made_outside(&reader, "inference", &example, &core),
            "{named}"
        );
        // Even said text_time, values in the list are no evidence for a.
        let said = json!("text_time");
        assert!(
            !made_outside(&reader, "inference", &example, &said),
            "{named}"
        );
    }
    // A value outside the list beside values that decide alike: made by it.
    let example = json!({"values": ["2024-01-05", "2024-01-06", "2021-13-45"], "named": false});
    let core = core_answer(&reader, "inference", &example).unwrap();
    assert_eq!(core, "text_time");
    assert!(made_outside(&reader, "inference", &example, &core));
}

/// The differential report (`text-times-differential.json`, recorded by
/// `text_times_differential.py` from pandas 3.0.6 over the inputs T1.25
/// recorded) holds the core: each example's answer is the core's, each
/// reference answer differs from it, and the counts add up.
#[test]
fn the_differential_reports_the_core_as_it_reads() {
    let report: Value = serde_json::from_str(DIFFERENTIAL).unwrap();
    assert_eq!(report["format"], "sdm-text-times-differential/1");
    let reader = Reader::new(report["year"].as_i64().unwrap());
    let outcomes = report["outcomes"].as_object().unwrap();
    assert_eq!(
        outcomes.keys().collect::<Vec<_>>(),
        ["materialize", "time_choice", "inference", "day_first"]
    );
    let mut examples = 0;
    for (outcome, held) in outcomes {
        let classes = held["classes"].as_object().unwrap();
        let differing: u64 = classes.values().map(|c| c["count"].as_u64().unwrap()).sum();
        assert_eq!(
            held["agree"].as_u64().unwrap() + differing,
            held["columns"].as_u64().unwrap(),
            "{outcome}"
        );
        for (class, entry) in classes {
            assert!(
                class == "a" || class == "c" || class.starts_with("b."),
                "{class}"
            );
            let listed = entry["examples"].as_array().unwrap();
            assert!(!listed.is_empty(), "{outcome} {class}");
            for example in listed {
                let core = core_answer(&reader, outcome, example).unwrap();
                assert_eq!(core, example["core"], "{outcome} {class}: {example}");
                assert_ne!(core, example["reference"], "{outcome} {class}: {example}");
                if class == "a" {
                    assert!(
                        made_outside(&reader, outcome, example, &core),
                        "{outcome}: {example}"
                    );
                }
                examples += 1;
            }
        }
    }
    assert!(examples > 0);
}
