// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32e part 2's SQL and answer exits for the time and feature family on
//! `rust-host`, held from outside the implementation (the package's
//! independent test author). Each `time.*` and `feature.*` case of
//! `contract/cases/s1` runs through this host, its statements executed on
//! DuckDB, through `predict` against the scripted endpoint:
//!
//! - Exit "the DuckDB SQL goldens for the family's own statements, executed
//!   on DuckDB in CI, as extensions of `sdm-data`'s `tests/duckdb_catalog.rs`
//!   text pins" (E53(c)), and "a seeded SQL change without a golden update
//!   and an entry fails": every statement the family runs (`from_source`
//!   and `materialize`) must be pinned, verbatim, as a string literal in
//!   `rust/crates/sdm-data/tests/*.rs`; a statement whose text changed
//!   without its pin changing is unpinned and fails. The check is shown to
//!   fail on a seeded change (in the statement over the seeds of
//!   `time.date`, and in the zoned case's first describe statement), on no
//!   pins and on no statements.
//! - Exit "statement counts equal the reference's or are registered,
//!   cross-checked against DuckDB's query log": per public call, the
//!   number of statements this host's provider ran equals the `python-ref`
//!   observation's (`contract/reference/<commit>.jsonl`), unless an
//!   approved register entry other than P1B-1 lists the case in its
//!   `cases`; and the provider's record is complete, its count equal to
//!   DuckDB's query log with no logged statement differing.
//! - Exit "`python-core` and `rust-host` agree on `core` fields", this
//!   host's half: each case's answer is the `python-ref` observation's: its
//!   outcome (`success`, or the refusal's code), and for a success the
//!   description (each table's columns by name and stype, its key, time
//!   column and duplicate keys, and the links), the model call's body (its
//!   digest as the `payload` rule compares it, and the rows and null cells
//!   of each table it carries) and the result's row count; or an approved
//!   entry other than P1B-1 lists the case.
//!   `contract/tests/test_t132e_time_feature_family.py` holds `python-core`
//!   to the same answers; admitted, the gate's `core` rule compares the two.
//!
//! Gating: the tests named in `contract/tests/t132e-pending.json`'s
//! `pending["p2-exits"]` (as `rust:<test name>`) failed when written, and
//! must still fail (strict); the implementer removes each as it passes.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const STAGES: [&str; 2] = ["from_source", "materialize"];
const FAMILY: [&str; 2] = ["time.", "feature."];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn pending(test: &str) -> bool {
    let text = std::fs::read_to_string(root().join("contract/tests/t132e-pending.json")).unwrap();
    let gating: Value = serde_json::from_str(&text).unwrap();
    gating["pending"]["p2-exits"]
        .as_array()
        .unwrap()
        .iter()
        .any(|t| t == &Value::from(format!("rust:{test}")))
}

/// Runs `check`; a pending test must fail, any other must pass.
fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending(test)) {
        (Ok(()), true) => panic!(
            "{test} passes now: take rust:{test} out of \
             contract/tests/t132e-pending.json's pending[\"p2-exits\"]"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

fn family_cases() -> (Vec<Case>, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let mut cases: Vec<Case> = loaded
        .into_iter()
        .map(|l| l.case)
        .filter(|c| FAMILY.iter().any(|p| c.id.starts_with(p)))
        .collect();
    cases.sort_by(|a, b| a.id.cmp(&b.id));
    assert_eq!(
        cases.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        [
            "feature.bool",
            "time.date",
            "time.feature-date-and-timestamp-with-null",
            "time.feature-interval",
            "time.feature-time-of-day",
            "time.text-declared",
            "time.text-iso-date",
            "time.text-one-bad-value",
            "time.text-us-date",
            "time.timestamp-ns",
            "time.timestamp-s",
            "time.timestamptz",
        ],
        "the family's twelve cases"
    );
    (cases, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Each case through this host, through `predict` against the scripted
/// endpoint, in process.
fn observed() -> Vec<(Case, Observed)> {
    let (cases, bases) = family_cases();
    cases
        .into_iter()
        .map(|case| {
            let seen = host::run(&case, &bases).unwrap_or_else(|e| panic!("{}: {e}", case.id));
            (case, seen)
        })
        .collect()
}

/// The statements the provider ran, by public call, in order.
fn statements(seen: &Observed) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let driver = seen.driver.as_ref().expect("a driver record");
    for statement in driver["capture"]["statements"].as_array().unwrap() {
        out.entry(statement["stage"].as_str().unwrap().to_owned())
            .or_default()
            .push(statement["sql"].as_str().unwrap().to_owned());
    }
    out
}

/// The python-ref observations, by case.
fn reference() -> BTreeMap<String, Value> {
    let env = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = env
        .lines()
        .find_map(|l| l.strip_prefix("reference: "))
        .unwrap()
        .trim()
        .to_owned();
    let text =
        std::fs::read_to_string(root().join(format!("contract/reference/{commit}.jsonl"))).unwrap();
    text.lines()
        .map(|line| {
            let o: Value = serde_json::from_str(line).unwrap();
            (o["case"].as_str().unwrap().to_owned(), o)
        })
        .collect()
}

/// A list field of each approved register entry (a proposed one excuses
/// nothing), read from the YAML's text entry by entry, in its flow or
/// block form.
fn register_lists(field: &str) -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let head = format!("  {field}:");
    let mut out = Vec::new();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let status = block
            .lines()
            .find_map(|l| l.strip_prefix("  status: "))
            .unwrap_or_default()
            .trim();
        if status != "approved" {
            continue;
        }
        let lines: Vec<&str> = block.lines().collect();
        let Some(at) = lines.iter().position(|l| l.starts_with(&head)) else {
            continue;
        };
        let mut items = Vec::new();
        let rest = lines[at][head.len()..].trim();
        if let Some(flow) = rest.strip_prefix('[') {
            let mut flow = flow.to_owned();
            let mut next = at + 1;
            while !flow.contains(']') && next < lines.len() {
                flow.push(' ');
                flow.push_str(lines[next].trim());
                next += 1;
            }
            let inner = flow.split(']').next().unwrap_or_default();
            items.extend(inner.split(',').map(|r| r.trim().to_owned()));
        } else {
            for line in &lines[at + 1..] {
                match line.strip_prefix("  - ") {
                    Some(item) => items.push(item.trim().to_owned()),
                    None => break,
                }
            }
        }
        let items = items
            .into_iter()
            .map(|i| i.trim_matches(|c| c == '\'' || c == '"').to_owned())
            .filter(|i| !i.is_empty())
            .collect();
        out.push((id, items));
    }
    out
}

/// The umbrellas, the placeholder P1B-1 among them: approved by the owner as
/// E92's Phase 2 class, they excuse no case, even one they list (T1.33).
const UMBRELLAS: [&str; 4] = ["P1B-1", "P1B-2", "P1B-3", "P1B-4"];

/// The approved entry that lists the case in its `cases`.
/// P1B-1 does not count: it is the inventory's placeholder, whose own notes
/// say each case whose outcome changes gets an entry of its own.
fn registered_case(id: &str) -> Option<String> {
    register_lists("cases")
        .into_iter()
        .find(|(entry, cases)| {
            !UMBRELLAS.contains(&entry.as_str()) && cases.iter().any(|c| c == id)
        })
        .map(|(entry, _)| entry)
}

// --- the goldens --------------------------------------------------------------

/// Every string literal of a Rust source, its escapes read: `"..."` (with
/// `\n`, `\t`, `\\`, `\"`, `\'`, `\0` and a line continuation) and raw
/// `r"..."`, `r#"..."#`; comments and character literals skipped.
fn literals(source: &str) -> Vec<String> {
    let s: Vec<char> = source.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        match s[i] {
            '/' if s.get(i + 1) == Some(&'/') => {
                while i < s.len() && s[i] != '\n' {
                    i += 1;
                }
            }
            '\'' => {
                if s.get(i + 2) == Some(&'\'') {
                    i += 3;
                } else if s.get(i + 1) == Some(&'\\') {
                    let close = (i + 2..(i + 12).min(s.len())).find(|j| s[*j] == '\'');
                    i = close.map_or(i + 1, |j| j + 1);
                } else {
                    i += 1;
                }
            }
            'r' if (s.get(i + 1) == Some(&'"') || s.get(i + 1) == Some(&'#'))
                && (i == 0 || !(s[i - 1].is_alphanumeric() || s[i - 1] == '_')) =>
            {
                let mut j = i + 1;
                let mut hashes = 0;
                while s.get(j) == Some(&'#') {
                    hashes += 1;
                    j += 1;
                }
                if s.get(j) != Some(&'"') {
                    i += 1;
                    continue;
                }
                let close: String = std::iter::once('"')
                    .chain(std::iter::repeat_n('#', hashes))
                    .collect();
                let body_start = j + 1;
                let rest: String = s[body_start..].iter().collect();
                let end = rest.find(&close).expect("a closed raw string");
                out.push(rest[..end].to_owned());
                i = body_start + rest[..end].chars().count() + close.chars().count();
            }
            '"' => {
                let mut text = String::new();
                i += 1;
                while i < s.len() && s[i] != '"' {
                    if s[i] == '\\' {
                        i += 1;
                        match s[i] {
                            'n' => text.push('\n'),
                            't' => text.push('\t'),
                            'r' => text.push('\r'),
                            '0' => text.push('\0'),
                            '\n' => {
                                while i + 1 < s.len() && s[i + 1].is_whitespace() {
                                    i += 1;
                                }
                            }
                            other => text.push(other),
                        }
                    } else {
                        text.push(s[i]);
                    }
                    i += 1;
                }
                out.push(text);
                i += 1;
            }
            _ => i += 1,
        }
    }
    out
}

/// The text pins: every string literal of `sdm-data`'s tests.
fn pins(dir: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.extend(literals(&std::fs::read_to_string(&path).unwrap()));
        }
    }
    out
}

/// The statements not pinned; every one is when the evidence is empty.
fn unpinned(ran: &[(String, String)], pinned: &BTreeSet<String>) -> Vec<String> {
    if ran.is_empty() {
        return vec!["no statement ran: no evidence".to_owned()];
    }
    ran.iter()
        .filter(|(_, sql)| !pinned.contains(sql))
        .map(|(id, sql)| format!("{id}: {sql}"))
        .collect()
}

fn family_statements() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (case, seen) in observed() {
        for (stage, texts) in statements(&seen) {
            if STAGES.contains(&stage.as_str()) {
                out.extend(
                    texts
                        .into_iter()
                        .map(|t| (format!("{} {stage}", case.id), t)),
                );
            }
        }
    }
    out
}

#[test]
fn the_literal_reader_reads_rust_s_strings() {
    let source = "let a = \"x\\n\\\"y\\\"\"; // \"not a pin\"\n\
                  let b = r#\"raw \"q\"\"#; let c = '\"'; let d = \"long \\\n    line\";";
    assert_eq!(
        literals(source),
        vec!["x\n\"y\"", "raw \"q\"", "long line"],
        "{source}"
    );
    // duckdb_catalog.rs's own pins read back as the statements they pin.
    let pinned = pins(&root().join("rust/crates/sdm-data/tests"));
    assert!(pinned.contains(
        "SELECT \"order_id\", \"user_id\", \"price\", \"ts\" FROM \"orders\" LIMIT 1000"
    ));
    assert!(pinned.iter().any(|p| p.starts_with(
        "SELECT constraint_type, constraint_column_names\nFROM duckdb_constraints()"
    )));
}

/// A seeded change: the statement of a case and stage holding a text, and
/// how its text is changed.
type Seed = (&'static str, &'static str, fn(&str) -> String);

#[test]
fn a_seeded_sql_change_without_a_golden_update_fails() {
    // The check itself, on seeded evidence, whatever the goldens hold.
    let ran = family_statements();
    assert!(ran.len() > 12 * 4, "{} statements", ran.len());
    let pinned: BTreeSet<String> = ran.iter().map(|(_, s)| s.clone()).collect();
    assert!(unpinned(&ran, &pinned).is_empty());
    // One statement's text changed, its pin not: it fails; the statement
    // over the seeds (a word added), and the first the zoned case's
    // description runs (a trailing space: pins are verbatim).
    let changes: [Seed; 2] = [
        ("time.date materialize", "__sdm_seeds__", |sql| {
            sql.replacen("SELECT ", "SELECT DISTINCT ", 1)
        }),
        ("time.timestamptz from_source", "", |sql| format!("{sql} ")),
    ];
    for (case, needle, change) in changes {
        let at = ran
            .iter()
            .position(|(id, sql)| id == case && sql.contains(needle))
            .unwrap_or_else(|| panic!("{case}: a statement holding {needle:?}"));
        let mut seeded = ran.clone();
        seeded[at].1 = change(&seeded[at].1);
        assert_ne!(seeded[at].1, ran[at].1, "{case}");
        assert_eq!(unpinned(&seeded, &pinned).len(), 1, "{case}");
    }
    // No pins, or no statements, fail closed.
    assert_eq!(unpinned(&ran, &BTreeSet::new()).len(), ran.len());
    assert_eq!(unpinned(&[], &pinned).len(), 1);
}

#[test]
fn every_statement_the_family_runs_is_pinned_in_sdm_data_s_tests() {
    gated(
        "every_statement_the_family_runs_is_pinned_in_sdm_data_s_tests",
        || {
            let pinned = pins(&root().join("rust/crates/sdm-data/tests"));
            let ran = family_statements();
            let missing: BTreeSet<String> = unpinned(&ran, &pinned)
                .into_iter()
                .map(|m| m.split_once(": ").map_or(m.clone(), |(_, s)| s.to_owned()))
                .collect();
            assert!(
                missing.is_empty(),
                "{} statements of the time and feature family have no golden pin in \
                 rust/crates/sdm-data/tests:\n{}",
                missing.len(),
                missing.into_iter().collect::<Vec<_>>().join("\n---\n")
            );
        },
    );
}

#[test]
fn statement_counts_equal_the_reference_s_or_are_registered() {
    gated(
        "statement_counts_equal_the_reference_s_or_are_registered",
        || {
            let reference = reference();
            let mut problems = Vec::new();
            for (case, seen) in observed() {
                // Complete: the provider's record against DuckDB's log.
                let driver = seen.driver.as_ref().unwrap();
                let recorded = driver["capture"]["statements"].as_array().unwrap().len();
                let logged = &driver["query_log"];
                if logged["statements"].as_u64() != Some(recorded as u64)
                    || logged["differing"].as_array().is_none_or(|d| !d.is_empty())
                {
                    problems.push(format!("{}: the record is not DuckDB's log", case.id));
                }
                let ours = statements(&seen);
                let theirs = &reference[&case.id]["driver"]["capture"]["statements"];
                for stage in STAGES {
                    let mine = ours.get(stage).map_or(0, Vec::len);
                    let want = theirs
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|s| s["stage"] == stage)
                        .count();
                    if mine != want && registered_case(&case.id).is_none() {
                        problems.push(format!(
                            "{} {stage}: {mine} statements, the reference {want}",
                            case.id
                        ));
                    }
                }
            }
            assert!(problems.is_empty(), "{}", problems.join("\n"));
        },
    );
}

// --- the answers --------------------------------------------------------------

/// A number token by kind, sign, significant digits and power of ten, as
/// the recorder's `number_form` reads it.
fn number_form(text: &str) -> Value {
    let integer = {
        let digits = text.strip_prefix('-').unwrap_or(text);
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
    };
    let kind = if integer { "integer" } else { "float" };
    let (negative, body) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let (mantissa, power) = match body.split_once(['e', 'E']) {
        Some((m, p)) => (m, p.parse::<i64>().expect("an exponent")),
        None => (body, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let mut exponent = power - i64::try_from(fraction.len()).unwrap();
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return json!([kind, negative, "0", 0]);
    }
    let stripped = significant.trim_end_matches('0');
    exponent += i64::try_from(significant.len() - stripped.len()).unwrap();
    json!([kind, negative, stripped, exponent])
}

/// Python's name of a parsed JSON value's type, as the recorder masks it.
fn type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn masked(value: &Value, path: &[String]) -> Value {
    let Some((head, rest)) = path.split_first() else {
        return value.clone();
    };
    let Some(object) = value.as_object() else {
        return value.clone();
    };
    let Some(inner) = object.get(head) else {
        return value.clone();
    };
    let mut out = object.clone();
    let replaced = if rest.is_empty() {
        Value::from(format!("<{}>", type_name(inner)))
    } else {
        masked(inner, rest)
    };
    out.insert(head.clone(), replaced);
    Value::Object(out)
}

/// The body with each number as `number_form` reads it and every object's
/// keys sorted.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            if object.len() == 1
                && let Some(Value::String(text)) = object.get("number")
            {
                return json!({"number": number_form(text)});
            }
            let sorted: BTreeMap<&String, &Value> = object.iter().collect();
            let mut out = Map::new();
            for (key, item) in sorted {
                out.insert(key.clone(), canonical(item));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// The recorder's `digest`: the SHA-256 of the canonical body.
fn digest(body: &Value, exclusions: &[Vec<String>]) -> String {
    let mut body = body.clone();
    for path in exclusions {
        body = masked(&body, path);
    }
    let text = serde_json::to_string(&canonical(&body)).unwrap();
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Each object of a body holding `columns` and `rows`, by its path: its
/// row count and null cells (the recorder's `tables_with_rows`).
fn tables_with_rows(value: &Value, at: &str, out: &mut Map<String, Value>) {
    if let Value::Object(object) = value {
        if let (Some(Value::Array(rows)), Some(Value::Array(_))) =
            (object.get("rows"), object.get("columns"))
        {
            let nulls = rows
                .iter()
                .filter_map(Value::as_array)
                .flatten()
                .filter(|c| c.is_null())
                .count();
            let key = if at.is_empty() { "." } else { at };
            out.insert(key.to_owned(), json!({"rows": rows.len(), "nulls": nulls}));
        }
        for (key, item) in object {
            let path = if at.is_empty() {
                key.clone()
            } else {
                format!("{at}.{key}")
            };
            tables_with_rows(item, &path, out);
        }
    }
}

fn payload(body: &Value, exclusions: &[Vec<String>]) -> Value {
    let mut tables = Map::new();
    tables_with_rows(body, "", &mut tables);
    json!({"digest": digest(body, exclusions), "tables": tables})
}

/// A description in the vectors' terms: tables by name, each its columns
/// (name and stype, in order), key, time column and duplicate keys; and
/// the links, sorted.
fn schema_of_core(described: &Value) -> Value {
    let mut tables = BTreeMap::new();
    for table in described["tables"].as_array().into_iter().flatten() {
        let columns: Vec<Value> = table["columns"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| json!([c["name"], c["stype"]]))
            .collect();
        let duplicates = table["duplicate_keys"]["v"]
            .as_str()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(Value::Null, Value::from);
        tables.insert(
            table["name"].as_str().unwrap_or_default().to_owned(),
            json!({
                "columns": columns,
                "primary_key": table["primary_key"],
                "time_column": table["time_column"],
                "duplicate_keys": duplicates,
            }),
        );
    }
    let mut links: Vec<Value> = described["links"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|l| json!([l[0]["table"], l[0]["column"], l[1]["table"], l[1]["column"]]))
        .collect();
    links.sort_by_key(Value::to_string);
    json!({"tables": tables, "links": links})
}

/// The reference's public schema in the same terms.
fn schema_of_reference(schema: &Value) -> Value {
    let mut tables = BTreeMap::new();
    for table in schema["tables"].as_array().into_iter().flatten() {
        let columns: Vec<Value> = table["columns"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| json!([c["name"], c["stype"]]))
            .collect();
        let key = match &table["primary_key"] {
            Value::Null => json!([]),
            Value::String(k) => json!([k]),
            other => other.clone(),
        };
        tables.insert(
            table["name"].as_str().unwrap_or_default().to_owned(),
            json!({
                "columns": columns,
                "primary_key": key,
                "time_column": table["time_column"],
                "duplicate_keys": table["duplicate_keys"],
            }),
        );
    }
    let mut links: Vec<Value> = schema["links"].as_array().cloned().unwrap_or_default();
    links.sort_by_key(Value::to_string);
    json!({"tables": tables, "links": links})
}

/// This host's answer to a case, in the reference's terms.
fn answer_of(seen: &Observed, exclusions: &[Vec<String>]) -> Value {
    match &seen.answer {
        Answer::Error { code, .. } => json!({"outcome": "refusal", "code": code}),
        Answer::Predicted(predicted) => json!({
            "outcome": "success",
            "code": null,
            "schema": seen.described.as_ref().map_or(Value::Null, schema_of_core),
            "payload": [payload(&predicted["body"], exclusions)],
            "rows": predicted["result"]["rows"].as_array().map(Vec::len),
        }),
        other => json!({"outcome": format!("no answer of predict: {other:?}")}),
    }
}

/// The `python-ref` observation's answer in the same terms: its request's
/// digest computed here, as the recorders compute it.
fn reference_answer(observation: &Value, exclusions: &[Vec<String>]) -> Value {
    let outcome = &observation["outcome"];
    if outcome["result"] == "refusal" {
        return json!({"outcome": "refusal", "code": outcome["code"]});
    }
    let payload: Vec<Value> = observation["public"]["payloads"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| payload(&p["parsed"], exclusions))
        .collect();
    json!({
        "outcome": "success",
        "code": null,
        "schema": schema_of_reference(&observation["public"]["schema"]),
        "payload": payload,
        "rows": observation["public"]["result"]["rows"],
    })
}

/// Whether this host's answer is the reference's: an unnamed refusal of
/// the reference is matched by no code.
fn agrees(got: &Value, want: &Value) -> bool {
    if want["outcome"] == "refusal" {
        return !want["code"].is_null()
            && got["outcome"] == "refusal"
            && got["code"] == want["code"];
    }
    got == want
}

/// The payload rule's exclusions, as the vectors record them.
fn exclusions() -> Vec<Vec<String>> {
    let text =
        std::fs::read_to_string(root().join("contract/multicategorical-vectors.json")).unwrap();
    let all: Value = serde_json::from_str(&text).unwrap();
    all["payload_exclusions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            p.as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_owned())
                .collect()
        })
        .collect()
}

#[test]
fn the_answer_comparison_fails_closed() {
    // On this family's own evidence: the reference's answer to time.date,
    // a success, agrees with itself and fails once any part changes; a
    // refusal agrees only by its code, and an unnamed one never.
    let reference = reference();
    let exclusions = exclusions();
    assert_eq!(
        exclusions,
        vec![vec!["metadata".to_owned(), "request_id".to_owned()]]
    );
    let want = reference_answer(&reference["time.date"], &exclusions);
    assert_eq!(want["outcome"], "success");
    assert!(agrees(&want, &want));
    for (part, value) in [
        ("outcome", json!("refusal")),
        ("schema", json!(null)),
        ("payload", json!([])),
        ("rows", json!(0)),
    ] {
        let mut got = want.clone();
        got[part] = value;
        assert!(!agrees(&got, &want), "{part}");
    }
    let mut got = want.clone();
    got["schema"]["tables"]["orders"]["columns"][3][1] = json!("text");
    assert!(!agrees(&got, &want), "another stype");
    let named = reference_answer(&reference["time.text-us-date"], &exclusions);
    assert_eq!(
        named,
        json!({"outcome": "refusal", "code": "INTERNAL_ERROR"})
    );
    assert!(agrees(&named, &named));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": "PROVIDER_ERROR"}),
        &named
    ));
    let unnamed = reference_answer(&reference["time.feature-interval"], &exclusions);
    assert_eq!(unnamed, json!({"outcome": "refusal", "code": null}));
    assert!(!agrees(&unnamed, &unnamed));
    // The placeholder lists cases but excuses none; DT-12, approved, lists
    // the text dates the source cannot cast.
    assert_eq!(registered_case("time.date"), None);
    assert_eq!(
        registered_case("time.text-us-date").as_deref(),
        Some("DT-12")
    );
}

#[test]
fn rust_host_answers_each_case_as_python_ref() {
    gated("rust_host_answers_each_case_as_python_ref", || {
        let reference = reference();
        let exclusions = exclusions();
        let mut problems = Vec::new();
        for (case, seen) in observed() {
            let want = reference_answer(&reference[&case.id], &exclusions);
            let got = answer_of(&seen, &exclusions);
            if !agrees(&got, &want) && registered_case(&case.id).is_none() {
                let parts: Vec<&str> = ["outcome", "code", "schema", "payload", "rows"]
                    .into_iter()
                    .filter(|p| got.get(*p) != want.get(*p))
                    .collect();
                problems.push(format!(
                    "{}: {parts:?} differ: {} {}, python-ref {} {}",
                    case.id, got["outcome"], got["code"], want["outcome"], want["code"]
                ));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    });
}
