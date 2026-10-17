// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T2.2's exits on `rust-host` (the package's test author's), beside
//! `contract/tests/test_t22_sqlite.py`, which holds `python-core` to the
//! same ones. Each test's one line says the exit it proves:
//!
//! - the staging rule's marker: the test host and the core report a SQLite
//!   provider (`sdm_testkit::PROVIDERS`, `sdm_host::providers()`, and
//!   `sdm-contract-run --providers`), so rust-host's SQLite pass runs;
//! - "every S1 case on SQLite": `sdm-contract-run --database sqlite`, the
//!   interface assumed (as `run_python.py --database sqlite`: each case's
//!   tables loaded by the contract loader, `contract/runners/databases.py`,
//!   into a new SQLite file, read through the dev-only `rusqlite`
//!   provider, and its observation keyed `<case>@sqlite`), observes every
//!   S1 case on SQLite with its core operations, each outcome (result and
//!   code) python-ref's committed `<case>@sqlite` line's or an approved
//!   register entry naming that key; `compare.py --database sqlite` holds
//!   the other fields, and `cross-host (sqlite)` the core fields against
//!   python-core;
//! - "L4 goldens executed": every statement rust-host's provider ran on
//!   SQLite is pinned verbatim as a string literal in
//!   `rust/crates/sdm-data/tests/sqlite*.rs`; and each relation form SQLite
//!   takes (one part, or two, the first `main` or an attached database;
//!   design choice 12) renders through the one `FROM` function, each part
//!   quoted alone, and executed on SQLite reads the table its parts name,
//!   while a relation of no part or of three is refused by name (rule
//!   `relation`);
//! - "the key-type pair matrix on SQLite" and "the decimal-key pair matrix,
//!   equal keys across scales included": each vector of
//!   `contract/sqlite-key-pair-vectors.json` (recorded from the reference)
//!   run through rust-host on SQLite answers as the reference does, its
//!   model call the reference's request (its digest as the `payload` rule
//!   compares it); where the reference fails unnamed or with
//!   `INTERNAL_ERROR`, the core refuses by name or joins every row to its
//!   own entity, under an approved register entry naming the vector.
//!
//! Gating: `contract/tests/t22-pending.json`'s `pending` names, as
//! `rust:<test>`, the tests that failed when written. Each must still fail
//! (strict); once one passes it fails, naming itself, until it is taken
//! out.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

use sdm_contract::data::{DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice, Relation};
use sdm_contract::{Detail, ErrorCode};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const DATABASE: &str = "sqlite";
const REGISTER_REF: &str = "contract/sqlite-key-pair-vectors.json#";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn pending(test: &str) -> bool {
    let gating = read_json(&root().join("contract/tests/t22-pending.json"));
    gating["pending"]
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
             contract/tests/t22-pending.json's pending"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    let gating = read_json(&root().join("contract/tests/t22-pending.json"));
    let names = |key: &str| -> BTreeSet<String> {
        gating[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap().to_owned())
            .collect()
    };
    assert!(names("pending").is_subset(&names("authored")));
}

/// `sdm-contract-run` with `arguments`, from the repository's root.
fn contract_run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .args(arguments)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

/// A directory of this test process's own under the system's temporary
/// directory, made empty.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("t22-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// rust-host's observations on SQLite of the cases at `cases`, by key, or
/// what the run said when it failed.
fn observe(cases: &Path, name: &str) -> Result<BTreeMap<String, Value>, String> {
    let out = scratch(name).join("rust-host.jsonl");
    let done = contract_run(&[
        "--database",
        DATABASE,
        "--out",
        out.to_str().unwrap(),
        cases.to_str().unwrap(),
    ]);
    if !done.status.success() {
        return Err(format!(
            "sdm-contract-run --database {DATABASE} failed: {}{}",
            String::from_utf8_lossy(&done.stdout),
            String::from_utf8_lossy(&done.stderr)
        ));
    }
    let text = std::fs::read_to_string(&out).map_err(|e| e.to_string())?;
    std::fs::remove_file(&out).ok();
    Ok(text
        .lines()
        .map(|line| {
            let observation: Value = serde_json::from_str(line).unwrap();
            (
                observation["case"].as_str().unwrap().to_owned(),
                observation,
            )
        })
        .collect())
}

/// python-ref's committed lines, by key.
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

/// The register's umbrellas: approved as E92's Phase 2 class, they excuse
/// no case and no vector (T1.33).
const UMBRELLAS: [&str; 4] = ["P1B-1", "P1B-2", "P1B-3", "P1B-4"];

/// The approved entry, no umbrella, whose text names `wanted` (a
/// `<case>@sqlite` key in its `cases` or `case_outcomes`, or a vector's
/// ref): each entry read from the register's text, block by block.
fn approved_naming(wanted: &str) -> Option<String> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let approved = block.lines().any(|l| l.trim_end() == "  status: approved");
        let names = block.split(|c: char| {
            c.is_whitespace() || matches!(c, ',' | '[' | ']' | '{' | '}' | '\'' | '"')
        });
        if approved && !UMBRELLAS.contains(&id.as_str()) && names.into_iter().any(|n| n == wanted) {
            return Some(id);
        }
    }
    None
}

// --- the staging marker ------------------------------------------------------------

#[test]
fn the_test_host_and_the_core_report_a_sqlite_provider() {
    gated(
        "the_test_host_and_the_core_report_a_sqlite_provider",
        || {
            assert!(sdm_testkit::PROVIDERS.contains(&DATABASE));
            assert!(sdm_host::providers().contains(&DATABASE));
            let done = contract_run(&["--providers"]);
            assert!(done.status.success());
            let listed: Value = serde_json::from_slice(&done.stdout).unwrap();
            assert!(listed.as_array().unwrap().contains(&Value::from(DATABASE)));
        },
    );
}

// --- every S1 case on SQLite --------------------------------------------------------

/// Every S1 case that runs on SQLite: its id and its core operations.
fn s1() -> Vec<(String, BTreeSet<String>)> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root().join("contract/cases/s1")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let case = read_json(&path);
        let on = case
            .get("databases")
            .and_then(Value::as_array)
            .is_none_or(|named| named.contains(&Value::from(DATABASE)));
        if on {
            let operations = case["core"]
                .as_object()
                .map(|core| core.keys().cloned().collect())
                .unwrap_or_default();
            out.push((case["id"].as_str().unwrap().to_owned(), operations));
        }
    }
    out.sort();
    out
}

fn cases_on_sqlite() -> &'static Result<BTreeMap<String, Value>, String> {
    static RUN: OnceLock<Result<BTreeMap<String, Value>, String>> = OnceLock::new();
    RUN.get_or_init(|| observe(&root().join("contract/cases"), "cases"))
}

fn outcome(observation: &Value) -> (Value, Value) {
    (
        observation["outcome"]["result"].clone(),
        observation["outcome"]["code"].clone(),
    )
}

#[test]
fn rust_host_runs_every_s1_case_on_sqlite() {
    gated("rust_host_runs_every_s1_case_on_sqlite", || {
        let observed = cases_on_sqlite().as_ref().unwrap();
        let reference = reference();
        let cases = s1();
        assert!(cases.len() >= 52);
        let mut problems = Vec::new();
        for (id, operations) in cases {
            let key = format!("{id}@{DATABASE}");
            let Some(line) = observed.get(&key) else {
                problems.push(format!("{key}: not observed"));
                continue;
            };
            let ran: BTreeSet<String> = line["identity"]["operations"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| o.as_str().unwrap().to_owned())
                .collect();
            if !operations.is_subset(&ran) || line["core"].is_null() {
                problems.push(format!("{key}: its core operations did not run"));
            }
            if line["identity"]["runner"] != "rust-host" {
                problems.push(format!("{key}: not rust-host's"));
            }
            let wanted = outcome(&reference[&key]);
            if outcome(line) != wanted && approved_naming(&key).is_none() {
                problems.push(format!(
                    "{key}: {:?} where python-ref has {wanted:?}, no approved entry names it",
                    outcome(line)
                ));
            }
        }
        assert!(problems.is_empty(), "{}", problems.join("\n"));
    });
}

// --- the L4 goldens -------------------------------------------------------------------

/// Every string literal of a Rust source, its escapes read (as
/// `t132a_keys_sql.rs` reads them): `"..."` (with `\n`, `\t`, `\r`, `\0`,
/// `\\`, `\"`, `\'` and a line continuation) and raw `r"..."`, `r#"..."#`;
/// comments and character literals skipped.
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

/// The SQLite goldens: every string literal of `sdm-data`'s test files
/// whose name holds `sqlite`.
fn goldens() -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for entry in std::fs::read_dir(root().join("rust/crates/sdm-data/tests")).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.contains("sqlite") && name.ends_with(".rs") {
            out.extend(literals(&std::fs::read_to_string(&path).unwrap()));
        }
    }
    out
}

/// Each statement a run made that no golden pins; every one when none ran.
fn unpinned(observed: &BTreeMap<String, Value>, pinned: &BTreeSet<String>) -> Vec<String> {
    let ran: Vec<(String, String)> = observed
        .iter()
        .flat_map(|(key, line)| {
            line["driver"]["capture"]["statements"]
                .as_array()
                .into_iter()
                .flatten()
                .map(move |s| {
                    let sql = s["sql"].as_str().unwrap_or_default().to_owned();
                    (format!("{key} {}", s["stage"]), sql)
                })
        })
        .collect();
    if ran.is_empty() {
        return vec!["no statement ran: no evidence".to_owned()];
    }
    ran.into_iter()
        .filter(|(_, sql)| !pinned.contains(sql))
        .map(|(at, sql)| format!("{at}: {sql}"))
        .collect()
}

#[test]
fn the_golden_check_reads_rust_and_fails_closed() {
    let source = "let a = \"x\\n\\\"y\\\"\"; // \"not a pin\"\n\
                  let b = r#\"raw \"q\"\"#; let c = '\"'; let d = \"long \\\n    line\";";
    assert_eq!(literals(source), vec!["x\n\"y\"", "raw \"q\"", "long line"]);
    let seen: BTreeMap<String, Value> = [(
        "k@sqlite".to_owned(),
        json!({"driver": {"capture": {"statements": [
            {"stage": "from_source", "sql": "SELECT 1"},
            {"stage": "materialize", "sql": "SELECT 2"},
        ]}}}),
    )]
    .into();
    let both: BTreeSet<String> = ["SELECT 1".to_owned(), "SELECT 2".to_owned()].into();
    assert!(unpinned(&seen, &both).is_empty());
    let one: BTreeSet<String> = ["SELECT 1".to_owned()].into();
    assert_eq!(unpinned(&seen, &one).len(), 1);
    assert_eq!(
        unpinned(&BTreeMap::new(), &one),
        ["no statement ran: no evidence"]
    );
}

#[test]
fn every_statement_rust_host_ran_on_sqlite_is_a_golden() {
    gated(
        "every_statement_rust_host_ran_on_sqlite_is_a_golden",
        || {
            let observed = cases_on_sqlite().as_ref().unwrap();
            let problems = unpinned(observed, &goldens());
            assert!(problems.is_empty(), "{}", problems.join("\n"));
        },
    );
}

fn sqlite() -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new(DATABASE),
    }
}

fn relation(parts: &[&str]) -> Relation {
    Relation {
        source: parts.iter().map(|p| (*p).to_owned()).collect(),
    }
}

fn scan(parts: &[&str]) -> DataRequest {
    DataRequest::Scan {
        table: relation(parts),
        columns: vec!["v".into()],
        filters: vec![],
        limit: None,
    }
}

fn plan(request: &DataRequest) -> Result<FetchPlan, sdm_contract::CoreError> {
    sdm_host::default_root()
        .data_access(&sqlite())?
        .plan(request)
}

/// A SQLite database holding a table of each relation form, each with its
/// own value of `v`, and an attached database `other`.
fn source() -> rusqlite::Connection {
    let connection = rusqlite::Connection::open_in_memory().unwrap();
    connection
        .execute_batch(
            r#"
            CREATE TABLE t (v INTEGER PRIMARY KEY);
            INSERT INTO t VALUES (1);
            CREATE TABLE "a.b" (v INTEGER);
            INSERT INTO "a.b" VALUES (2);
            CREATE TABLE "say ""hi""" (v INTEGER);
            INSERT INTO "say ""hi""" VALUES (5);
            ATTACH ':memory:' AS other;
            CREATE TABLE other.u (v INTEGER PRIMARY KEY);
            INSERT INTO other.u VALUES (6);
            "#,
        )
        .unwrap();
    connection
}

#[test]
fn each_relation_form_sqlite_takes_reads_the_table_its_parts_name() {
    gated(
        "each_relation_form_sqlite_takes_reads_the_table_its_parts_name",
        || {
            let connection = source();
            for (parts, from, value) in [
                (&["t"][..], r#""t""#, 1),
                // A dotted name is one table, never split on its dot.
                (&["a.b"], r#""a.b""#, 2),
                (&["main", "t"], r#""main"."t""#, 1),
                // Two parts naming an attached database.
                (&["other", "u"], r#""other"."u""#, 6),
                // A part holding a double quote, the quote doubled.
                (&["say \"hi\""], r#""say ""hi""""#, 5),
                (&["main", "say \"hi\""], r#""main"."say ""hi""""#, 5),
            ] {
                let FetchPlan::Sql { statements, .. } = plan(&scan(parts)).unwrap() else {
                    panic!("{parts:?}: not SQL");
                };
                let [statement] = &statements[..] else {
                    panic!("{parts:?}: not one statement");
                };
                let text = &statement.text;
                assert!(text.contains(&format!(" FROM {from}")), "{parts:?}: {text}");
                let mut prepared = connection.prepare(text).unwrap();
                let read: Vec<i64> = prepared
                    .query_map([], |row| row.get(0))
                    .unwrap()
                    .map(Result::unwrap)
                    .collect();
                assert_eq!(read, [value], "{parts:?}: {text}");
            }
        },
    );
}

#[test]
fn a_relation_of_no_part_or_of_three_is_refused_by_name_on_sqlite() {
    gated(
        "a_relation_of_no_part_or_of_three_is_refused_by_name_on_sqlite",
        || {
            for (parts, code) in [
                (&[][..], ErrorCode::InvalidRequest),
                (&["main", "s", "t"], ErrorCode::UnsupportedRequest),
            ] {
                for request in [
                    scan(parts),
                    DataRequest::Describe {
                        tables: vec![relation(parts)],
                        part: DescribePart::Columns,
                    },
                ] {
                    let err = plan(&request).unwrap_err();
                    assert_eq!(err.code, code, "{parts:?}");
                    assert_eq!(err.fields["rule"], Detail::text("relation"), "{parts:?}");
                    let given = u64::try_from(parts.len()).unwrap();
                    assert_eq!(
                        err.fields["parts"],
                        Detail::Scalar(sdm_contract::Scalar::UInt64(given))
                    );
                }
            }
        },
    );
}

// --- the key-type and decimal-key pair matrices ---------------------------------------

fn vectors() -> Value {
    let all = read_json(&root().join("contract/sqlite-key-pair-vectors.json"));
    assert_eq!(all["format"], "sdm-sqlite-key-pair-vectors/1");
    assert_eq!(
        all["recorded_by"],
        "contract/tools/sqlite_key_pairs_record.py"
    );
    assert_eq!(all["database"], DATABASE);
    assert_eq!(all["register_ref"], REGISTER_REF);
    all
}

/// A number token by kind, sign, significant digits and power of ten, as
/// the recorder's `number_form` reads it (`t132d_empty_null_vectors.rs`).
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

/// A key cell as an integer: a number's digits or a text's.
fn integral(cell: &Value) -> Option<i128> {
    let text = cell
        .get("number")
        .and_then(Value::as_str)
        .or_else(|| cell.as_str())?;
    text.trim_end_matches(".0").parse().ok()
}

/// Where a call joins a row to another entity than its own, by the shop's
/// facts (`users.age` is 20 + id; `orders.user_id` is `order_id % 8`, at
/// a price of 10 + 1.5 * `order_id`), integral keys read exactly less
/// `base` (the recorder's `joins`).
fn joins(body: &Value, base: i128) -> Vec<String> {
    let mut wrong = Vec::new();
    for part in ["context", "predict"] {
        let section = &body[part];
        let instances = &section["instance_table"];
        let at = |table: &Value, name: &str| {
            table["columns"]
                .as_array()
                .unwrap()
                .iter()
                .position(|c| c == name)
        };
        let (id_at, entity_at) = (
            at(instances, "instance_id").unwrap(),
            at(instances, "__kumo_entity_ref_0").unwrap(),
        );
        let entity: BTreeMap<String, Option<i128>> = instances["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row[id_at].to_string(),
                    integral(&row[entity_at]).map(|e| e - base),
                )
            })
            .collect();
        for (name, table) in section["related_tables"].as_object().unwrap() {
            for row in table["rows"].as_array().unwrap() {
                let cell = |column: &str| at(table, column).map(|i| &row[i]);
                let own = entity[&cell("instance_id").unwrap().to_string()];
                let key = cell("user_id").and_then(integral).map(|k| k - base);
                if cell("user_id").is_some() && (key.is_none() || key != own) {
                    wrong.push(format!("{part}.{name}: user_id of entity {own:?}"));
                }
                if name == "users" && cell("age").and_then(integral).map(|a| a - 20) != own {
                    wrong.push(format!("{part}.users: age of entity {own:?}"));
                }
                if name == "orders" {
                    let price: f64 = cell("price").unwrap()["number"]
                        .as_str()
                        .unwrap()
                        .parse()
                        .unwrap();
                    let order = ((price - 10.0) / 1.5).round() as i128;
                    if Some(order % 8) != own {
                        wrong.push(format!("{part}.orders: order of entity {own:?}"));
                    }
                }
            }
        }
    }
    wrong
}

fn key_pairs_on_sqlite() -> &'static Result<BTreeMap<String, Value>, String> {
    static RUN: OnceLock<Result<BTreeMap<String, Value>, String>> = OnceLock::new();
    RUN.get_or_init(|| {
        let cases = scratch("pairs").join("cases");
        std::fs::create_dir_all(&cases).unwrap();
        for vector in vectors()["vectors"].as_array().unwrap() {
            let case = &vector["case"];
            let path = cases.join(format!("{}.json", case["id"].as_str().unwrap()));
            std::fs::write(path, serde_json::to_string_pretty(case).unwrap()).unwrap();
        }
        observe(&cases, "pairs-out")
    })
}

#[test]
fn the_join_check_fails_on_a_wrong_join() {
    // python-ref's base.regression@sqlite: the shop, its question and
    // entities, as pair.bigint.bigint asks.
    let line = &reference()["base.regression@sqlite"];
    let body = &line["public"]["payloads"][0]["parsed"];
    assert!(joins(body, 0).is_empty());
    let mut moved = body.clone();
    let orders = &mut moved["predict"]["related_tables"]["orders"];
    let at = orders["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|c| c == "user_id")
        .unwrap();
    orders["rows"][0][at] = json!({"number": "5"});
    assert!(!joins(&moved, 0).is_empty());
}

#[test]
fn each_key_pair_on_sqlite_is_the_reference_s_on_rust_host() {
    gated(
        "each_key_pair_on_sqlite_is_the_reference_s_on_rust_host",
        || {
            let all = vectors();
            let exclusions: Vec<Vec<String>> =
                serde_json::from_value(all["payload_exclusions"].clone()).unwrap();
            let observed = key_pairs_on_sqlite().as_ref().unwrap();
            let mut problems = Vec::new();
            for vector in all["vectors"].as_array().unwrap() {
                let id = vector["id"].as_str().unwrap();
                let key = format!("{}@{DATABASE}", vector["case"]["id"].as_str().unwrap());
                let Some(line) = observed.get(&key) else {
                    problems.push(format!("{id}: not observed"));
                    continue;
                };
                let reference = &vector["reference"];
                let wanted = outcome(reference);
                let got = outcome(line);
                let body = &line["core"]["predicted"]["body"];
                let same_call = match reference["payload"].as_array() {
                    Some(sent) => {
                        sent.len() == 1
                            && !body.is_null()
                            && digest(body, &exclusions) == sent[0]["digest"]
                    }
                    None => body.is_null(),
                };
                let registered = approved_naming(&format!("{REGISTER_REF}{id}")).is_some();
                if vector["register"]["required"] == true {
                    let named = got.0 == "refusal" && !got.1.is_null() && got.1 != "INTERNAL_ERROR";
                    let base: i128 = vector["indices"][0]
                        .to_string()
                        .trim_matches('"')
                        .parse()
                        .unwrap_or_default();
                    let right =
                        got.0 == "success" && !body.is_null() && joins(body, base).is_empty();
                    if !(named || right) {
                        problems.push(format!("{id}: neither named nor a right join"));
                    }
                    if !registered {
                        problems.push(format!("{id}: differs from the reference, not registered"));
                    }
                } else if (got != wanted || !same_call) && !registered {
                    problems.push(format!(
                        "{id}: {got:?} where the reference has {wanted:?}, the call {}",
                        if same_call { "the same" } else { "differs" }
                    ));
                }
            }
            assert!(problems.is_empty(), "{}", problems.join("\n"));
        },
    );
}
