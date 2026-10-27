// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.1d: off DuckDB rust-host makes every call a case makes through its
//! `until`, as python-core does (`run_python.routes_on`), each missing core
//! input derived from the case's input as the recorders write one
//! (`host::through_until`).

use std::path::PathBuf;

use sdm_testkit::case::{self, Case, Until};
use sdm_testkit::host::through_until;
use serde_json::{Value, json};

fn corpus() -> Vec<Case> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/cases");
    case::load_for(&[root], case::Database::Duckdb)
        .unwrap()
        .into_iter()
        .map(|l| l.case)
        .collect()
}

/// The anchor time the recorders write for a case's `anchor_time` (a
/// date): its midnight, UTC, in nanoseconds, read here by SQLite's
/// `strftime`, not by the host's own date arithmetic; null where the case
/// gives none.
fn recorded_anchor(date: Option<&str>) -> Value {
    let Some(date) = date else {
        return Value::Null;
    };
    let seconds: i64 = rusqlite::Connection::open_in_memory()
        .unwrap()
        .query_row(
            "SELECT CAST(strftime('%s', ?1) AS INTEGER)",
            [date],
            |row| row.get(0),
        )
        .unwrap_or_else(|e| panic!("{date} is not a date: {e}"));
    let nanos = i128::from(seconds) * 1_000_000_000;
    json!({"t": "ts", "v": nanos.to_string(), "unit": "ns", "zone": null})
}

fn named(case: &Case) -> Vec<&'static str> {
    let core = case.core.as_ref().unwrap();
    let mut ops = Vec::new();
    if core.describe_source.is_some() {
        ops.push("describe_source");
    }
    if core.validate_query.is_some() {
        ops.push("validate_query");
    }
    if core.materialize.is_some() {
        ops.push("materialize");
    }
    if core.predict.is_some() {
        ops.push("predict");
    }
    ops
}

#[test]
fn every_call_through_until_is_named_with_the_recorders_inputs() {
    let cases = corpus();
    let base = cases.iter().find(|c| c.id == "base.regression").unwrap();
    let recorded = base.core.as_ref().unwrap();
    let sqlite = json!({"provider": "sql", "dialect": "sqlite"});
    // The expected anchor is the recorders': every case whose materialize
    // input was recorded from a date has the anchor `recorded_anchor`
    // gives for that date (2024-03-01's base.regression's, 2023-05-01's
    // the shop's), so it stands for the recorder where none was recorded.
    let mut controls = 0;
    for case in &cases {
        let given = case.input.materialize.anchor_time.as_deref();
        if let Some(recorded) = case.core.as_ref().and_then(|c| c.materialize.as_ref())
            && given.is_some_and(|d| d.len() == "2024-03-01".len())
        {
            assert_eq!(recorded.anchor_time, recorded_anchor(given), "{}", case.id);
            controls += 1;
        }
    }
    assert!(controls >= 2, "{controls}");
    let mut derived = 0;
    for case in cases
        .iter()
        .filter(|c| c.slice == "S1" || c.id.starts_with("s0."))
    {
        let made = through_until(case, "sqlite").unwrap();
        let core = made.core.as_ref().unwrap();
        // Every call through the case's until, and no other.
        let mut wanted = vec!["describe_source"];
        if core.validate_query.is_some() {
            wanted.push("validate_query");
        }
        if case.until != Until::FromSource {
            wanted.push("materialize");
        }
        if case.until == Until::Predict {
            wanted.push("predict");
        }
        assert_eq!(named(&made), wanted, "{}", case.id);
        // Each provider SQLite's, every other recorded input kept.
        for provider in core
            .describe_source
            .iter()
            .map(|s| &s.provider)
            .chain(core.materialize.iter().map(|m| &m.provider))
        {
            assert_eq!(provider, &sqlite, "{}", case.id);
        }
        let given = case.core.as_ref();
        if given.and_then(|c| c.materialize.as_ref()).is_none()
            && let Some(made) = core.materialize.as_ref()
        {
            derived += 1;
            // The case's entities 0 and 1 and the default seed as the
            // recorder wrote them for base.regression, at the case's own
            // anchor (base.regression's 2024-03-01, or another a case's
            // base or question anchors at), as the recorders write a date.
            let base = recorded.materialize.as_ref().unwrap();
            assert_eq!(made.indices, base.indices, "{}", case.id);
            assert_eq!(
                made.anchor_time,
                recorded_anchor(case.input.materialize.anchor_time.as_deref()),
                "{}",
                case.id
            );
            assert_eq!(made.seed, base.seed, "{}", case.id);
        }
        if given.and_then(|c| c.predict.as_ref()).is_none() && case.until == Until::Predict {
            assert_eq!(
                core.predict.as_ref().unwrap().model,
                recorded.predict.as_ref().unwrap().model,
                "{}",
                case.id
            );
        }
    }
    // The S0 cases and the four DuckDB stops at from_source, not vacuous.
    assert!(derived >= 28, "{derived}");
}

#[test]
fn an_input_no_derivation_writes_is_refused() {
    let cases = corpus();
    let mut case = cases
        .iter()
        .find(|c| c.id == "s0.accept.binary-classification")
        .unwrap()
        .clone();
    case.input.materialize.indices_as = Some(case::IndicesAs::Decimal);
    let failed = through_until(&case, "sqlite").unwrap_err().to_string();
    assert!(failed.contains("no derivation writes"), "{failed}");
    let mut case = cases
        .iter()
        .find(|c| c.id == "s0.accept.binary-classification")
        .unwrap()
        .clone();
    case.input.materialize.anchor_time = Some("2024-02-30".to_owned());
    assert!(through_until(&case, "sqlite").is_err());
    // A text entity is text, an integer past i64 its digits.
    let mut case = case.clone();
    case.input.materialize.anchor_time = Some("2024-03-01".to_owned());
    case.input.materialize.indices = vec![
        case::Index::Integer("9223372036854775808".to_owned()),
        case::Index::Text("u0".to_owned()),
    ];
    let made = through_until(&case, "sqlite").unwrap();
    let indices: &Vec<Value> = &made
        .core
        .as_ref()
        .unwrap()
        .materialize
        .as_ref()
        .unwrap()
        .indices;
    assert_eq!(
        indices,
        &vec![
            json!({"t": "utf8", "v": "9223372036854775808"}),
            json!({"t": "utf8", "v": "u0"})
        ]
    );
}
