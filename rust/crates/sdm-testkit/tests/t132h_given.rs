// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32h (register entry P1B-12): the keys and time columns
//! `describe_source` is given, as the contract's `Given`, held on
//! `rust-host` from outside the implementation (the package's independent
//! test author) to `contract/given-vectors.json`, which
//! `contract/tests/test_t132h_given_vectors.py` holds `python-core` to, so
//! the two hosts agree through it.
//!
//! - Each vector with a `spec` is started here as written, the bytes of
//!   its `SourceSpec` the attachment, over the `shop` base in DuckDB, and
//!   answered as its `core` says: the outcome; where it ends (`start`,
//!   refused before the run exists and nothing fetched, or `run`, after at
//!   least one fetch: the source read before any key or time column is
//!   judged); the code; the error's fields, exactly (`"*"` any value); and
//!   for a success each table's key and time column. A `wire` vector's
//!   spec is outside the contract's shape, `INVALID_MESSAGE` naming the
//!   attachment; a bare list, string or null, contract 22's shape,
//!   included.
//! - Each `case` vector runs as a case through this host's runner, which
//!   reads `core.describe_source.source` as the contract's `SourceSpec`
//!   (its preflight): the same code and text fields. A case left in the
//!   old shape stops at that preflight (the design's seed 12).
//! - Each `edit` vector is `edit_schema`'s, over the schema the core
//!   described: its answer, and beside its `pair` the same code, message
//!   and `rule`, the field the operation's own (`primary_key` against
//!   `primary_keys`); a `null` keeps each operation's meaning.
//! - A start speaking contract 22 is refused before a run exists,
//!   `INVALID_MESSAGE` naming the version.
//! - The comparison fails on each seeded answer of the design's seeds 7 to
//!   11 and 13.
//!
//! Gating: `contract/tests/t132h-pending.json`'s `pending["rust-host"]`
//! lists the vectors that failed here when the test was written (a case
//! run as a case is `case:<id>`), and `pending["exits"]` the other checks
//! (`rs:` and `rs-edit:` ids). A vector of a decision still the owner's
//! (`owner`, `D3` or `D4`) is gated in the group of that name, its id
//! prefixed with the group it would otherwise be in (`rust-host:<id>`,
//! `exits:rs-edit:<id>`). Each must still fail until taken
//! out; the lists only shrink. The implementer changes nothing else of
//! this file or the vectors.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use sdm_contract::data::{AttachmentId, FetchPlan, ProviderChoice, SqlParam};
use sdm_contract::ops::{DescribeSource, EditSchema, Operation, OperationResult};
use sdm_contract::run::{FetchOutcome, FetchReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, CoreError, Detail, Step};
use sdm_host::{ArrowData, Attachment, HostRun, Outgoing, default_root, encode, read};
use sdm_testkit::bases::{self, Bases};
use sdm_testkit::case::Case;
use sdm_testkit::host::{self, Answer};
use sdm_testkit::{duck, edits};
use serde_json::{Value, json};

const VECTORS: &str = "contract/given-vectors.json";
const PENDING: &str = "contract/tests/t132h-pending.json";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn all() -> Value {
    let all = read_json(VECTORS);
    assert_eq!(all["format"], "sdm-given-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/given_record.py");
    assert_eq!(all["register_ref"], "contract/given-vectors.json#");
    assert_eq!(all["base"], "shop");
    all
}

fn vectors() -> Vec<Value> {
    all()["vectors"].as_array().unwrap().clone()
}

fn ids(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{PENDING}: a group that is not a list"))
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn pending(group: &str) -> BTreeSet<String> {
    let gating = read_json(PENDING);
    let Some(listed) = gating["pending"].get(group) else {
        panic!("{PENDING}: no pending group {group}");
    };
    ids(listed)
}

/// The group and name a vector's check is gated under: `group` and
/// `<prefix><id>`, or for a decision still the owner's (`owner`) that
/// decision's group and `<group>:<prefix><id>`.
fn gate(vector: &Value, group: &str, prefix: &str) -> (String, String) {
    let id = vector["id"].as_str().unwrap();
    match vector["owner"].as_str() {
        Some(owner) => (owner.to_owned(), format!("{group}:{prefix}{id}")),
        None => (group.to_owned(), format!("{prefix}{id}")),
    }
}

/// Holds each check to the pending list: a check not pending must pass,
/// a pending one must still fail.
fn hold(results: Vec<((String, String), Vec<String>)>) {
    let mut failing = Vec::new();
    let mut passing_but_pending = Vec::new();
    for ((group, name), problems) in results {
        let listed = pending(&group).contains(&name);
        match (problems.is_empty(), listed) {
            (true, true) => passing_but_pending.push(format!("{group}: {name}")),
            (false, false) => failing.push(format!("{group}: {name}: {problems:?}")),
            _ => {}
        }
    }
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of {PENDING}'s pending lists: \
         {passing_but_pending:?}"
    );
    assert!(failing.is_empty(), "{}", failing.join("\n"));
}

fn bases() -> Bases {
    bases::load(&root().join("contract/bases.yaml")).unwrap()
}

fn shop(bases: &Bases) -> duckdb::Connection {
    let connection = duck::open().unwrap();
    for statement in &bases["shop"] {
        connection.execute_batch(statement).unwrap();
    }
    connection
}

/// A field of an error as the vectors write it: text as a string, an
/// integer as a number.
fn detail(value: &Detail) -> Value {
    let scalar = |s: &sdm_contract::Scalar| -> Value {
        let written = serde_json::to_value(s).unwrap();
        let text = written["v"].as_str().map(str::to_owned);
        match (written["t"].as_str(), text) {
            (Some("utf8"), Some(text)) => Value::from(text),
            (_, Some(text)) => text
                .parse::<i64>()
                .map_or(Value::from(text.clone()), Value::from),
            _ => written,
        }
    };
    match value {
        Detail::Scalar(s) => scalar(s),
        Detail::List(items) => Value::Array(items.iter().map(scalar).collect()),
    }
}

fn spelled(value: impl serde::Serialize) -> Value {
    serde_json::to_value(value).unwrap()
}

/// What a run answered, in the vectors' terms.
#[derive(Debug, Clone)]
struct Got {
    outcome: &'static str,
    at: &'static str,
    code: Value,
    message: Option<String>,
    fields: Value,
    tables: Value,
    fetches: usize,
}

fn refusal(error: &CoreError, at: &'static str, fetches: usize) -> Got {
    Got {
        outcome: "refusal",
        at,
        code: spelled(error.code),
        message: Some(error.message.clone()),
        fields: Value::Object(
            error
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), detail(value)))
                .collect(),
        ),
        tables: Value::Null,
        fetches,
    }
}

fn error_in(out: &Outgoing) -> CoreError {
    match read(out).expect("an error message the host can read") {
        ControlMessage::Error { error } => error,
        other => panic!("not an error: {other:?}"),
    }
}

/// Each table's key and time column, as a success of the vectors writes
/// them.
fn tables_of(description: &Value) -> Value {
    let mut tables = serde_json::Map::new();
    for table in description["tables"].as_array().into_iter().flatten() {
        let key = match &table["primary_key"] {
            Value::Array(columns) if columns.is_empty() => Value::Null,
            other => other.clone(),
        };
        tables.insert(
            table["name"].as_str().unwrap().to_owned(),
            json!({"primary_key": key, "time_column": table["time_column"]}),
        );
    }
    Value::Object(tables)
}

fn start(
    operation: Operation,
    contract: u32,
    attachments: BTreeMap<AttachmentId, Attachment>,
) -> Result<HostRun, Outgoing> {
    let message = encode(&ControlMessage::Start(Box::new(StartRun {
        contract,
        facts: host::facts().unwrap(),
        operation,
    })))
    .unwrap();
    HostRun::start(&default_root(), &message, attachments)
}

/// The run driven to its result, each fetch answered from `connection`:
/// the result, or the error and where, and the fetches answered.
fn drive(
    handle: &HostRun,
    connection: &duckdb::Connection,
) -> (Result<OperationResult, CoreError>, usize) {
    let mut fetches = 0;
    loop {
        let out = match handle.next() {
            Ok(out) => out,
            Err(failed) => return (Err(error_in(&failed)), fetches),
        };
        match read(&out).unwrap() {
            ControlMessage::Step(Step::Done { result }) => return (Ok(result), fetches),
            ControlMessage::Step(Step::Fetch { requests }) => {
                for step in &requests {
                    if let FetchPlan::Sql { statements, .. } = &step.plan {
                        assert!(
                            !statements
                                .iter()
                                .flat_map(|s| &s.params)
                                .any(|p| matches!(p, SqlParam::KeySet { .. })),
                            "describe_source bound a key set"
                        );
                    }
                    let (outcome, attachments) = match duck::answer(connection, step) {
                        Ok(fetched) => (
                            FetchOutcome::Data {
                                data: AttachmentId(1),
                                continuation: None,
                            },
                            BTreeMap::from([(
                                AttachmentId(1),
                                Attachment::Arrow(ArrowData::new(fetched.schema, fetched.batches)),
                            )]),
                        ),
                        Err(error) => (FetchOutcome::Error { error }, BTreeMap::new()),
                    };
                    fetches += 1;
                    let reply = encode(&ControlMessage::FetchReply(FetchReply {
                        id: step.id,
                        outcome,
                    }))
                    .unwrap();
                    if let Err(failed) = handle.reply(&reply, attachments) {
                        return (Err(error_in(&failed)), fetches);
                    }
                }
            }
            other => panic!("describe_source asked for {other:?}"),
        }
    }
}

fn describe_operation() -> Operation {
    Operation::DescribeSource(DescribeSource {
        provider: serde_json::from_value::<ProviderChoice>(all()["provider"].clone()).unwrap(),
        source: AttachmentId(1),
        check_keys: true,
    })
}

/// `describe_source` started with `spec`'s bytes as they stand, over a
/// fresh `shop`: what it answered, and the description on a success.
fn describe_raw(spec: &[u8], bases: &Bases) -> (Got, Option<Value>) {
    let connection = shop(bases);
    let attachments = BTreeMap::from([(AttachmentId(1), Attachment::Bytes(spec.to_vec()))]);
    let handle = match start(describe_operation(), CONTRACT_VERSION, attachments) {
        Ok(handle) => handle,
        Err(refused) => return (refusal(&error_in(&refused), "start", 0), None),
    };
    match drive(&handle, &connection) {
        (Err(error), fetches) => (refusal(&error, "run", fetches), None),
        (Ok(OperationResult::DescribeSource(done)), fetches) => {
            let Ok(Some(Attachment::Bytes(bytes))) = handle.take_attachment(done.description)
            else {
                panic!("the description is not attached");
            };
            let description: Value = serde_json::from_slice(&bytes).unwrap();
            (
                Got {
                    outcome: "success",
                    at: "run",
                    code: Value::Null,
                    message: None,
                    fields: Value::Null,
                    tables: tables_of(&description),
                    fetches,
                },
                Some(description),
            )
        }
        (Ok(other), _) => panic!("describe_source answered {other:?}"),
    }
}

/// `edit_schema` started over `description` with `edit`'s bytes as they
/// stand.
fn edit_raw(description: &Value, edit: &Value) -> Got {
    let schema: sdm_contract::ops::SourceDescription =
        serde_json::from_value(description.clone()).unwrap();
    let held = serde_json::to_vec(&edits::held(&schema, &edits::kept(&schema))).unwrap();
    let attachments = BTreeMap::from([
        (AttachmentId(1), Attachment::Bytes(held)),
        (
            AttachmentId(2),
            Attachment::Bytes(serde_json::to_vec(edit).unwrap()),
        ),
    ]);
    let operation = Operation::EditSchema(EditSchema {
        schema: AttachmentId(1),
        edit: AttachmentId(2),
    });
    let handle = match start(operation, CONTRACT_VERSION, attachments) {
        Ok(handle) => handle,
        Err(refused) => return refusal(&error_in(&refused), "start", 0),
    };
    let unused = duck::open().unwrap();
    match drive(&handle, &unused) {
        (Err(error), fetches) => refusal(&error, "run", fetches),
        (Ok(OperationResult::EditSchema(done)), fetches) => {
            let Ok(Some(Attachment::Bytes(bytes))) = handle.take_attachment(done.schema) else {
                panic!("the schema edited is not attached");
            };
            let edited: Value = serde_json::from_slice(&bytes).unwrap();
            Got {
                outcome: "success",
                at: "run",
                code: Value::Null,
                message: None,
                fields: Value::Null,
                tables: tables_of(&edited),
                fetches,
            }
        }
        (Ok(other), _) => panic!("edit_schema answered {other:?}"),
    }
}

/// Where `got` is not `core` (a vector's): empty when it is. `edit` says
/// an `edit_schema`, which reads nothing: its refusals are at its run, with
/// no fetch.
fn differs(core: &Value, got: &Got, edit: bool) -> Vec<String> {
    let mut problems = Vec::new();
    if core["outcome"] != got.outcome {
        problems.push(format!(
            "outcome {} (wanted {})",
            got.outcome, core["outcome"]
        ));
        return problems;
    }
    if core["at"] != got.at {
        problems.push(format!("at {} (wanted {})", got.at, core["at"]));
    }
    match (got.at, edit) {
        ("start", _) if got.fetches != 0 => {
            problems.push(format!(
                "{} fetches before a refusal at the start",
                got.fetches
            ));
        }
        ("run", false) if got.fetches == 0 => {
            problems.push("judged before the source was read (no fetch)".into());
        }
        _ => {}
    }
    if got.outcome == "success" {
        for (table, wanted) in core["tables"].as_object().unwrap() {
            if got.tables[table] != *wanted {
                problems.push(format!("{table}: {} (wanted {wanted})", got.tables[table]));
            }
        }
        return problems;
    }
    if core["code"] != got.code {
        problems.push(format!("code {} (wanted {})", got.code, core["code"]));
    }
    let wanted = core["fields"].as_object().unwrap();
    let fields = got.fields.as_object().cloned().unwrap_or_default();
    let names: BTreeSet<&String> = fields.keys().collect();
    let wanted_names: BTreeSet<&String> = wanted.keys().collect();
    if names != wanted_names {
        problems.push(format!("fields {names:?} (wanted {wanted_names:?})"));
    }
    for (name, value) in wanted {
        if value != "*" && fields.get(name) != Some(value) {
            problems.push(format!("{name}: {:?} (wanted {value})", fields.get(name)));
        }
    }
    problems
}

#[test]
fn the_vectors_are_the_reference_s_where_a_caller_can_give_them() {
    let all = all();
    let reference = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = all["reference"].as_str().unwrap();
    assert!(reference.contains(&format!("reference: {commit}")));
    let mut seen = BTreeSet::new();
    let mut kinds = BTreeMap::new();
    let mut rules = BTreeSet::new();
    for vector in all["vectors"].as_array().unwrap() {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        let kind = vector["kind"].as_str().unwrap();
        *kinds.entry(kind.to_owned()).or_insert(0) += 1;
        let core = &vector["core"];
        let keys: BTreeSet<&str> = core
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        match core["outcome"].as_str() {
            Some("success") => {
                assert_eq!(keys, BTreeSet::from(["outcome", "at", "tables"]), "{id}")
            }
            Some("refusal") => {
                assert_eq!(
                    keys,
                    BTreeSet::from(["outcome", "at", "code", "fields"]),
                    "{id}"
                );
                if let Some(rule) = core["fields"]["rule"].as_str() {
                    rules.insert(rule.to_owned());
                }
            }
            other => panic!("{id}: outcome {other:?}"),
        }
        match kind {
            "case" => {
                assert!(vector["from_source"].is_object(), "{id}");
                let recorded = &vector["reference"];
                assert!(
                    matches!(recorded["result"].as_str(), Some("success" | "refusal")),
                    "{id}: no reference recorded"
                );
                assert_eq!(recorded["stage"], "from_source", "{id}");
            }
            "wire" | "edit" => {
                assert!(vector["reference"].is_null(), "{id}");
                assert!(
                    vector["reference_why"]
                        .as_str()
                        .is_some_and(|w| !w.is_empty())
                );
            }
            other => panic!("{id}: kind {other}"),
        }
    }
    assert!(
        kinds["case"] >= 20 && kinds["wire"] >= 15 && kinds["edit"] >= 4,
        "{kinds:?}"
    );
    for rule in [
        "key_not_names",
        "time_column_not_a_name",
        "key_column_twice",
    ] {
        assert!(rules.contains(rule), "no vector of {rule}");
    }
}

#[test]
fn the_pending_lists_only_shrink() {
    let gating = read_json(PENDING);
    let groups: BTreeSet<&String> = gating["authored"].as_object().unwrap().keys().collect();
    let pending_groups: BTreeSet<&String> = gating["pending"].as_object().unwrap().keys().collect();
    assert_eq!(groups, pending_groups);
    for group in groups {
        let authored = ids(&gating["authored"][group]);
        let now = ids(&gating["pending"][group]);
        assert!(
            now.is_subset(&authored),
            "{group}: pending ids added: {:?}",
            now.difference(&authored).collect::<Vec<_>>()
        );
    }
}

#[test]
fn rust_host_answers_each_vector_as_its_spec_is_written() {
    let bases = bases();
    let mut results = Vec::new();
    for vector in vectors() {
        if vector["spec"].is_null() {
            continue;
        }
        let bytes = serde_json::to_vec(&vector["spec"]).unwrap();
        let (got, _) = describe_raw(&bytes, &bases);
        results.push((
            gate(&vector, "rust-host", ""),
            differs(&vector["core"], &got, false),
        ));
    }
    hold(results);
}

fn case_of(vector: &Value, source: &Value) -> Case {
    let case = json!({
        "format": "sdm-case/1",
        "id": format!("t132h.{}", vector["id"].as_str().unwrap()),
        "status": "diagnostic",
        "slice": "S1",
        "paths": [],
        "routes": [],
        "cells": [],
        "input": {
            "base": "shop",
            "from_source": vector["from_source"],
            "pql": all()["question"],
            "materialize": {"indices": [0, 1], "anchor_time": "2024-03-01"},
        },
        "until": "from_source",
        "core": {"describe_source": {
            "provider": all()["provider"],
            "source": source,
            "check_keys": true,
        }},
        "expect": {"outcome": "success", "stage": "from_source"},
        "why": format!("T1.32h vector {}", vector["id"].as_str().unwrap()),
        "source": "contract/tools/given_record.py",
    });
    serde_json::from_str(&case.to_string()).expect("the vector's case is a case")
}

/// A case's answer in the vectors' terms: the code and text fields of a
/// refusal (this host's runner keeps a field's text only), or the tables.
fn case_differs(core: &Value, answer: &Answer) -> Vec<String> {
    match answer {
        Answer::Error { code, fields, .. } => {
            if core["outcome"] != "refusal" {
                return vec![format!("refused {code}")];
            }
            let mut problems = Vec::new();
            if core["code"] != code.as_str() {
                problems.push(format!("code {code} (wanted {})", core["code"]));
            }
            let wanted: BTreeMap<String, String> = core["fields"]
                .as_object()
                .unwrap()
                .iter()
                .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned())))
                .collect();
            if *fields != wanted {
                problems.push(format!("fields {fields:?} (wanted {wanted:?})"));
            }
            problems
        }
        Answer::Described(described) => {
            if core["outcome"] != "success" {
                return vec!["described".into()];
            }
            let tables = tables_of(described);
            core["tables"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(table, wanted)| tables[table.as_str()] != **wanted)
                .map(|(table, wanted)| {
                    format!("{table}: {} (wanted {wanted})", tables[table.as_str()])
                })
                .collect()
        }
        other => vec![format!("answered {other:?}")],
    }
}

#[test]
fn rust_host_runs_each_case_vector_as_a_case() {
    let bases = bases();
    let mut results = Vec::new();
    for vector in vectors() {
        if vector["kind"] != "case" {
            continue;
        }
        let problems = match host::run(&case_of(&vector, &vector["spec"]), &bases) {
            Ok(observed) => case_differs(&vector["core"], &observed.answer),
            Err(failure) => vec![format!("host failure: {failure}")],
        };
        results.push((gate(&vector, "rust-host", "case:"), problems));
    }
    hold(results);
}

/// Contract 22's shape of a spec: each key a bare list of names, each time
/// column a bare name or null.
fn old_shape(spec: &Value) -> Value {
    let unwrap = |given: &Value| given.get("value").cloned().unwrap_or(Value::Null);
    let mut old = spec.clone();
    for field in ["primary_keys", "time_columns"] {
        let map = old[field].as_object_mut().unwrap();
        for value in map.values_mut() {
            *value = unwrap(value);
        }
    }
    old
}

#[test]
fn a_case_left_in_the_old_shape_stops_at_the_runner_s_preflight() {
    // The design's seed 12: the runner reads the case's spec as the
    // contract's SourceSpec, so one left in contract 22's shape is this
    // host's failure, never an answer.
    let bases = bases();
    let mut problems = Vec::new();
    for vector in vectors() {
        let written_old = vector["kind"] == "case"
            && vector["spec"]["primary_keys"]
                .as_object()
                .into_iter()
                .chain(vector["spec"]["time_columns"].as_object())
                .flat_map(|m| m.values())
                .all(|v| v.get("value").is_some());
        if !written_old || vector["owner"].is_string() {
            continue;
        }
        let case = case_of(&vector, &old_shape(&vector["spec"]));
        match host::run(&case, &bases) {
            Err(failure) if failure.to_string().contains("is not a SourceSpec") => {}
            Err(failure) => problems.push(format!("{}: {failure}", vector["id"])),
            Ok(observed) => {
                problems.push(format!("{}: answered {:?}", vector["id"], observed.answer))
            }
        }
    }
    let results = vec![(
        ("exits".to_owned(), "rs:old-shape-preflight".to_owned()),
        problems,
    )];
    hold(results);
}

#[test]
fn a_start_speaking_contract_22_is_refused_before_a_run_exists() {
    let spec =
        serde_json::to_vec(&json!({"tables": [], "primary_keys": {}, "time_columns": {}})).unwrap();
    let attachments = BTreeMap::from([(AttachmentId(1), Attachment::Bytes(spec))]);
    let problems = match start(describe_operation(), 22, attachments) {
        Ok(_) => vec!["a start speaking contract 22 began a run".to_owned()],
        Err(refused) => {
            let got = refusal(&error_in(&refused), "start", 0);
            let core = json!({
                "outcome": "refusal", "at": "start", "code": "INVALID_MESSAGE",
                "fields": {"contract": 22},
            });
            differs(&core, &got, false)
        }
    };
    hold(vec![(
        ("exits".to_owned(), "rs:contract-22".to_owned()),
        problems,
    )]);
}

fn by_id(id: &str) -> Value {
    vectors()
        .into_iter()
        .find(|v| v["id"] == id)
        .unwrap_or_else(|| panic!("no vector {id}"))
}

/// The edit's answer against its vector, and, beside the `describe_source`
/// refusal it pairs with, the same code, message and rule, the field each
/// operation's own.
fn edit_problems(vector: &Value, got: &Got, paired: &Got) -> Vec<String> {
    let mut problems = differs(&vector["core"], got, true);
    let pair = by_id(vector["pair"].as_str().unwrap());
    problems.extend(
        differs(&pair["core"], paired, false)
            .into_iter()
            .map(|p| format!("its pair {}: {p}", pair["id"])),
    );
    if got.outcome == "refusal" && paired.outcome == "refusal" {
        if got.code != paired.code {
            problems.push(format!("code {} beside {}", got.code, paired.code));
        }
        if got.message != paired.message {
            problems.push(format!(
                "message {:?} beside {:?}",
                got.message, paired.message
            ));
        }
        if got.fields["rule"] != paired.fields["rule"] || got.fields["rule"].is_null() {
            problems.push(format!(
                "rule {} beside {}",
                got.fields["rule"], paired.fields["rule"]
            ));
        }
        let singular = got.fields["field"].as_str().unwrap_or_default();
        let plural = paired.fields["field"].as_str().unwrap_or_default();
        if format!("{singular}s") != plural {
            problems.push(format!("field {singular} beside {plural}"));
        }
    }
    problems
}

#[test]
fn edit_schema_shares_describe_source_s_refusals_and_keeps_its_null() {
    let bases = bases();
    let empty =
        serde_json::to_vec(&json!({"tables": [], "primary_keys": {}, "time_columns": {}})).unwrap();
    let (described, description) = describe_raw(&empty, &bases);
    assert_eq!(described.outcome, "success", "{described:?}");
    let description = description.unwrap();
    let mut results = Vec::new();
    for vector in vectors() {
        if vector["kind"] != "edit" {
            continue;
        }
        let got = edit_raw(&description, &vector["edit"]);
        let pair = by_id(vector["pair"].as_str().unwrap());
        let (paired, _) = describe_raw(&serde_json::to_vec(&pair["spec"]).unwrap(), &bases);
        results.push((
            gate(&vector, "exits", "rs-edit:"),
            edit_problems(&vector, &got, &paired),
        ));
    }
    hold(results);
}

/// A refusal in the vectors' terms, as a seed writes it.
fn refused(code: &str, at: &'static str, fields: Value, message: &str) -> Got {
    Got {
        outcome: "refusal",
        at,
        code: Value::from(code),
        message: Some(message.to_owned()),
        fields,
        tables: Value::Null,
        fetches: usize::from(at == "run"),
    }
}

#[test]
fn the_comparison_fails_on_each_seeded_answer() {
    // The design's seeds that land on the core, each as the answer it
    // would give: every one must differ from its vector.
    let not_names = "a primary key names a column or a sequence of them";
    let seeds: Vec<(&str, &str, Got)> = vec![
        (
            "7: the core reads unreadable as no key",
            "key.int",
            Got {
                outcome: "success",
                at: "run",
                code: Value::Null,
                message: None,
                fields: Value::Null,
                tables: json!({"orders": {"primary_key": ["order_id"], "time_column": "ts"}}),
                fetches: 3,
            },
        ),
        (
            "7: the core reads unreadable as an empty key",
            "key.int",
            refused(
                "INVALID_REQUEST",
                "run",
                json!({"field": "primary_keys", "table": "orders"}),
                "a table was given an empty primary key",
            ),
        ),
        (
            "8: the decoder takes a bare list",
            "wire.key-bare-list",
            Got {
                outcome: "success",
                at: "run",
                code: Value::Null,
                message: None,
                fields: Value::Null,
                tables: json!({}),
                fetches: 3,
            },
        ),
        (
            "9: minItems makes an empty key INVALID_MESSAGE",
            "key.empty",
            refused(
                "INVALID_MESSAGE",
                "start",
                json!({"attachment": 1, "category": "data", "line": 1, "column": 9}),
                "an attachment is not the JSON its message names",
            ),
        ),
        (
            "10: the rule dropped",
            "key.int",
            refused(
                "INVALID_REQUEST",
                "run",
                json!({"field": "primary_keys", "table": "orders"}),
                not_names,
            ),
        ),
        (
            "10: the time column's rule dropped",
            "time.int",
            refused(
                "INVALID_REQUEST",
                "run",
                json!({"field": "time_columns", "table": "orders"}),
                "a column named is not its table's",
            ),
        ),
        (
            "11: the edit's field made plural",
            "edit.key-not-names",
            refused(
                "INVALID_REQUEST",
                "run",
                json!({"field": "primary_keys", "table": "orders", "rule": "key_not_names"}),
                not_names,
            ),
        ),
        (
            "KO2-86: another value named than the one given",
            "key.int",
            refused(
                "INVALID_REQUEST",
                "run",
                json!({"field": "primary_keys", "table": "orders", "rule": "key_not_names",
                       "given": "4", "variant": "given"}),
                not_names,
            ),
        ),
        (
            "2: the refusal judged before the source is read",
            "key.int",
            Got {
                fetches: 0,
                ..refused(
                    "INVALID_REQUEST",
                    "run",
                    json!({"field": "primary_keys", "table": "orders", "rule": "key_not_names"}),
                    not_names,
                )
            },
        ),
    ];
    for (seed, id, got) in &seeds {
        let vector = by_id(id);
        let edit = vector["kind"] == "edit";
        let mut got = got.clone();
        if edit {
            got.fetches = 0;
        }
        assert!(
            !differs(&vector["core"], &got, edit).is_empty(),
            "seed {seed} passes"
        );
    }
    // Seed 11 beside its pair, and the rule dropped from the edit alone.
    let edit = by_id("edit.key-not-names");
    // From KO2-86 each names the value given, as the host spelled it.
    let paired = refused(
        "INVALID_REQUEST",
        "run",
        json!({"field": "primary_keys", "table": "orders", "rule": "key_not_names",
               "given": "3", "variant": "given"}),
        not_names,
    );
    let good = Got {
        fetches: 0,
        ..refused(
            "INVALID_REQUEST",
            "run",
            json!({"field": "primary_key", "table": "orders", "rule": "key_not_names",
                   "given": "3", "variant": "given"}),
            not_names,
        )
    };
    assert!(
        edit_problems(&edit, &good, &paired).is_empty(),
        "the control fails"
    );
    let ruleless = Got {
        fields: json!({"field": "primary_key", "table": "orders"}),
        ..good.clone()
    };
    assert!(!edit_problems(&edit, &ruleless, &paired).is_empty());
    let worded = Got {
        message: Some("a key is not names".into()),
        ..good.clone()
    };
    assert!(!edit_problems(&edit, &worded, &paired).is_empty());
    // Seed 13: the version check relaxed is a run begun, which the exit
    // reports; a refusal naming another version differs.
    let core = json!({
        "outcome": "refusal", "at": "start", "code": "INVALID_MESSAGE",
        "fields": {"contract": 22},
    });
    let other = refused("INVALID_MESSAGE", "start", json!({"contract": 23}), "");
    assert!(!differs(&core, &other, false).is_empty());
    // The controls: each vector's own answer passes.
    for id in ["key.int", "time.int", "wire.key-bare-list"] {
        let vector = by_id(id);
        let core = &vector["core"];
        let fields: serde_json::Map<String, Value> = core["fields"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), if v == "*" { Value::from(1) } else { v.clone() }))
            .collect();
        let at = if core["at"] == "start" {
            "start"
        } else {
            "run"
        };
        let got = refused(
            core["code"].as_str().unwrap(),
            at,
            Value::Object(fields),
            "",
        );
        assert!(
            differs(core, &got, false).is_empty(),
            "{id}: the control fails"
        );
    }
}
