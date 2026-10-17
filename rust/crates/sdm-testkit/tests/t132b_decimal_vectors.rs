// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32b's decimals on `rust-host`, held from outside the implementation
//! (the package's independent test author) to what the reference gave:
//! `contract/decimal-vectors.json`, recorded from the reference by
//! `contract/tools/decimal_record.py`, which
//! `contract/tests/test_t132b_decimal_vectors.py` holds `python-core` to as
//! well, so the two hosts agree through it.
//!
//! - Exits "`python-core` and `rust-host` agree on `core` fields" and
//!   "every difference is registered", over the family's inputs: each
//!   vector (`shop` changed by the vector's `setup`: DECIMAL(p, s) for p in
//!   4, 9, 18, 19, 38 and s in 0, 2, 10, p as features of both tables,
//!   summed and classified targets, keys and columns beside the time
//!   column; small values, values at the declared precision, zero,
//!   negatives, nulls; sums past the declared precision; HUGEINT and
//!   UHUGEINT past 38 digits) runs through this host as
//!   `decimal.feature-38-0-wide` does, through `describe_source`,
//!   `materialize` and `predict` against the scripted endpoint, the core
//!   handed the entities and anchor the recorder derived (`core_indices`,
//!   `core_anchor`: the client's bridge's). The answer must be the
//!   reference's, as far as this host observes it: a refusal with the
//!   reference's code (an unnamed exception is matched by no code); or a
//!   success whose description is the reference's public schema (each
//!   table's columns by name and stype, key, time column, duplicate keys,
//!   links), whose model call's body is the reference's request (its digest
//!   as the `payload` rule compares it, and the rows and null cells of each
//!   table), whose result has the reference's row count, and where each of
//!   the vector's decimal columns has the reference's stype and reached the
//!   model as the reference sent it (each part's wire dtype and stype, and
//!   its cells counted, a number by its exact value as the payload rule
//!   reads it, `1e-9` and `1e-09` alike, `1` and `1.0` not; text exact).
//! - Exit "statement counts equal the reference's, cross-checked against
//!   DuckDB's query log", over the vectors: where the outcome is the
//!   reference's, the statements the provider ran per public call equal the
//!   vector's `statements`, and the provider's record is DuckDB's log.
//! - A vector may differ from the reference only under an approved entry
//!   of `docs/design/rust-core/register.yaml` whose `refs` name it with the
//!   transition it makes (T1.32a's binding-transition form):
//!   `contract/decimal-vectors.json#<id or glob> <from> -> <to>`, each side
//!   `success` or `refusal <CODE>`, from the reference's answer to this
//!   host's (the reference's unnamed exception is `refusal UNNAMED`). A ref
//!   covers exactly the transition it declares; a proposed entry excuses
//!   nothing. A declared `success -> success` (or same-refusal) difference
//!   still holds the statement counts. The recorder's `register.required`
//!   marks where a difference is expected and excuses nothing by itself.
//! - Every vector ref of a proposed or approved entry names a transition
//!   in that form; a malformed one fails.
//!
//! Gating: `contract/tests/t132b-pending.json`'s `pending["rust-host"]`
//! lists the vectors that failed here when the test was written. Each must
//! still fail (strict: one that passes must be taken out of the list), and
//! the list may only shrink: it stays within `authored["rust-host"]`. The
//! implementer removes ids as they pass and changes nothing else of this
//! file, the vectors or the recorder.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case, Index};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const REF: &str = "contract/decimal-vectors.json#";
const COUNTED: [&str; 3] = ["from_source", "materialize", "predict"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = read_json("contract/decimal-vectors.json");
    assert_eq!(all["format"], "sdm-decimal-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/decimal_record.py");
    assert_eq!(
        all["template"],
        "contract/cases/s1/decimal.feature-38-0-wide.json"
    );
    assert_eq!(all["register_ref"], REF);
    all
}

fn ids(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

/// `*` matches any run of characters, every other character itself.
fn glob(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && p[pi] != '*' && p[pi] == t[ti] {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

/// One vector ref of a register entry, read: the glob, the transition's
/// start and end, or `None` when it names no transition in the form
/// `<glob> <from> -> <to>`, each side `success` or `refusal <CODE>`.
fn transition(item: &str) -> Option<(String, String, String)> {
    let rest = item.strip_prefix(REF)?;
    let (glob, rest) = rest.split_once(' ')?;
    let (old, new) = rest.split_once(" -> ")?;
    let class = |c: &str| {
        c == "success"
            || c.strip_prefix("refusal ").is_some_and(|code| {
                code.starts_with(|ch: char| ch.is_ascii_uppercase())
                    && code.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_')
            })
    };
    (!glob.is_empty() && class(old) && class(new))
        .then(|| (glob.to_owned(), old.to_owned(), new.to_owned()))
}

/// A vector ref read: entry, status, glob, from, to.
type Transition = (String, String, String, String, String);

/// Each vector ref of the register's proposed or approved entries, and the
/// refs naming vectors with no transition in the form (malformed). Read
/// from the YAML's text, entry by entry (`- id:` at the margin), the `refs`
/// field alone, in its flow (`refs: [a, b]`) or block (`refs:` then
/// `  - a`) form.
fn transitions(text: &str) -> (Vec<Transition>, Vec<String>) {
    let mut out = Vec::new();
    let mut malformed = Vec::new();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let status = block
            .lines()
            .find_map(|l| l.strip_prefix("  status: "))
            .unwrap_or_default()
            .trim()
            .to_owned();
        if status != "proposed" && status != "approved" {
            continue;
        }
        let lines: Vec<&str> = block.lines().collect();
        let Some(at) = lines.iter().position(|l| l.starts_with("  refs:")) else {
            continue;
        };
        let mut refs: Vec<String> = Vec::new();
        let rest = lines[at].trim_start_matches("  refs:").trim();
        if let Some(flow) = rest.strip_prefix('[') {
            let mut flow = flow.to_owned();
            let mut next = at + 1;
            while !flow.contains(']') && next < lines.len() {
                flow.push(' ');
                flow.push_str(lines[next].trim());
                next += 1;
            }
            let inner = flow.split(']').next().unwrap_or_default();
            refs.extend(inner.split(',').map(|r| r.trim().to_owned()));
        } else {
            for line in &lines[at + 1..] {
                match line.strip_prefix("  - ") {
                    Some(item) => refs.push(item.trim().to_owned()),
                    None => break,
                }
            }
        }
        for item in refs
            .iter()
            .map(|r| r.trim_matches(|c| c == '\'' || c == '"'))
            .filter(|r| r.starts_with(REF))
        {
            match transition(item) {
                Some((glob, old, new)) => {
                    out.push((id.clone(), status.clone(), glob, old, new));
                }
                None => malformed.push(format!("{id}: {item}")),
            }
        }
    }
    (out, malformed)
}

/// The approved entry naming `id` with exactly the transition `old -> new`.
fn covered(id: &str, old: &str, new: &str, entries: &[Transition]) -> Option<String> {
    entries
        .iter()
        .find(|(_, status, pattern, from, to)| {
            status == "approved" && from == old && to == new && glob(pattern, id)
        })
        .map(|(entry, ..)| entry.clone())
}

/// `success` or `refusal <CODE>` of an answer in this host's terms (the
/// reference's unnamed exception, with no code, is `refusal UNNAMED`).
fn outcome_class(answer: &Value) -> String {
    if answer["outcome"] == "success" {
        "success".to_owned()
    } else {
        format!("refusal {}", answer["code"].as_str().unwrap_or("UNNAMED"))
    }
}

/// Whether this host's answer passes the vector: the reference's, or a
/// difference an approved entry declares exactly (a declared difference
/// within the same outcome still holds the statement counts).
fn passes(id: &str, got: &Value, want: &Value, entries: &[Transition]) -> bool {
    if agrees(got, want) {
        return true;
    }
    let answered = got["outcome"] == "success" || got["outcome"] == "refusal";
    let (old, new) = (outcome_class(want), outcome_class(got));
    if !answered || covered(id, &old, &new, entries).is_none() {
        return false;
    }
    old != new || (got["statements"] == want["statements"] && got["complete"] == true)
}

// --- the case -----------------------------------------------------------------

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect()
}

/// `Schema.from_source`'s keywords as the core's source spec, as the
/// client's bridge builds it (`core_source` of the recorder).
fn core_source(from_source: &Value) -> Value {
    let tables: Vec<Value> = from_source["tables"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(name, source)| json!({"name": name, "source": source}))
        .collect();
    let keys: Map<String, Value> = from_source["primary_keys"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(table, key)| {
            let key = match key {
                Value::String(_) => json!([key]),
                other => other.clone(),
            };
            (table.clone(), json!({"value": key}))
        })
        .collect();
    let times: Map<String, Value> = from_source["time_columns"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(table, column)| (table.clone(), json!({"value": column})))
        .collect();
    json!({"tables": tables, "primary_keys": keys, "time_columns": times})
}

/// The case this host runs for `vector`: `decimal.feature-38-0-wide`'s,
/// its setup, question and tables the vector's, and the core handed the
/// entities and anchor time the recorder derived as the client's bridge
/// builds them (`core_indices`, `core_anchor`).
fn the_case(template: &Case, vector: &Value, question: &str) -> Case {
    let mut case = template.clone();
    case.id = format!("t132b.{}", vector["id"].as_str().unwrap());
    let setup = strings(&vector["setup"]);
    case.input.setup = (!setup.is_empty()).then_some(setup);
    case.input.pql = vector["pql"].as_str().unwrap_or(question).to_owned();
    let entities = vector["core_indices"].as_array().unwrap().clone();
    // The public call's entities as their digits (no double between).
    case.input.materialize.indices = entities
        .iter()
        .map(|e| Index::Integer(e["v"].as_str().unwrap().to_owned()))
        .collect();
    case.input.materialize.anchor_time = match vector.get("anchor_time") {
        Some(Value::Null) => None,
        Some(Value::String(at)) => Some(at.clone()),
        _ => Some("2024-03-01".to_owned()),
    };
    let core = case.core.as_mut().expect("the template's core inputs");
    core.describe_source
        .as_mut()
        .expect("the template describes its source")
        .source = core_source(vector.get("from_source").unwrap_or(&Value::Null));
    let materialize = core
        .materialize
        .as_mut()
        .expect("the template materializes");
    materialize.indices = entities;
    materialize.anchor_time = vector["core_anchor"].clone();
    case
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

/// A cell as its JSON text, keys sorted (the recorder's `cell_text`).
fn cell_text(cell: &Value) -> String {
    fn sorted(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let keys: BTreeMap<&String, &Value> = object.iter().collect();
                let mut out = Map::new();
                for (key, item) in keys {
                    out.insert(key.clone(), sorted(item));
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sorted(cell)).unwrap()
}

/// How one column reached the model in one part (`context` or `predict`)
/// of a request body: the wire dtype and stype its schema declares and its
/// cells counted, or null (the recorder's `column_in`).
fn column_in(body: &Value, part: &str, table: &str, column: &str) -> Value {
    let section = &body[part];
    let (holder, declared) = if table == "instance_table" {
        (
            &section["instance_table"],
            &body["schema"]["instance_table"]["columns"],
        )
    } else {
        (
            &section["related_tables"][table],
            &body["schema"]["related_tables"][table]["columns"],
        )
    };
    let Some(names) = holder["columns"].as_array() else {
        return Value::Null;
    };
    let Some(at) = names.iter().position(|n| n == column) else {
        return Value::Null;
    };
    let mut cells: BTreeMap<String, u64> = BTreeMap::new();
    for row in holder["rows"].as_array().into_iter().flatten() {
        if let Some(cell) = row.as_array().and_then(|r| r.get(at)) {
            *cells.entry(cell_text(cell)).or_default() += 1;
        }
    }
    json!({
        "dtype": declared[column]["dtype"],
        "stype": declared[column]["stype"],
        "cells": cells,
    })
}

/// A request's cells keyed by value: each number cell (`{"number":
/// text}`) by its kind, sign, significant digits and power of ten
/// (`number_form`, the payload rule's reading, exact, no double), so the
/// core's `1e-9` and the reference's `1e-09` are one cell, while `1` and
/// `1.0` (their kinds) are not; text, nulls and structure stay exact.
/// Cells that are one value are counted together.
fn by_value(request: &Value) -> Value {
    let Some(requests) = request.as_array() else {
        return request.clone();
    };
    let mut out = Vec::new();
    for parts in requests {
        let mut each = Map::new();
        for (part, column) in parts.as_object().into_iter().flatten() {
            let Some(cells) = column["cells"].as_object() else {
                each.insert(part.clone(), column.clone());
                continue;
            };
            let mut counted: BTreeMap<String, u64> = BTreeMap::new();
            for (key, count) in cells {
                let cell: Value = serde_json::from_str(key).expect("a cell's JSON text");
                *counted.entry(cell_text(&canonical(&cell))).or_default() +=
                    count.as_u64().expect("a count");
            }
            let mut column = column.clone();
            column["cells"] = json!(counted);
            each.insert(part.clone(), column);
        }
        out.push(Value::Object(each));
    }
    Value::Array(out)
}

/// The vector's columns in this host's terms: each its stype as described
/// and how it reached the model, keyed `table.column`.
fn columns_of_core(observed: &Observed, body: Option<&Value>, columns: &Value) -> Value {
    let mut out = Map::new();
    for pair in columns.as_array().unwrap() {
        let (table, column) = (pair[0].as_str().unwrap(), pair[1].as_str().unwrap());
        let stype = observed
            .described
            .as_ref()
            .and_then(|d| d["tables"].as_array().cloned())
            .into_iter()
            .flatten()
            .filter(|t| t["name"] == table)
            .flat_map(|t| t["columns"].as_array().cloned().unwrap_or_default())
            .find(|c| c["name"] == column)
            .map_or(Value::Null, |c| c["stype"].clone());
        let request = body.map_or(Value::Null, |body| {
            json!([{
                "context": column_in(body, "context", table, column),
                "predict": column_in(body, "predict", table, column),
            }])
        });
        out.insert(
            format!("{table}.{column}"),
            json!({"stype": stype, "request": by_value(&request)}),
        );
    }
    Value::Object(out)
}

/// The statements this host's provider ran per public call, and whether
/// its record is DuckDB's query log.
fn statements(observed: &Observed) -> (Value, bool) {
    let Some(driver) = observed.driver.as_ref() else {
        return (Value::Null, false);
    };
    let recorded = driver["capture"]["statements"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let mut counted = Map::new();
    for stage in COUNTED {
        let n = recorded.iter().filter(|s| s["stage"] == stage).count();
        counted.insert(stage.to_owned(), json!(n));
    }
    let logged = &driver["query_log"];
    let complete = logged["statements"].as_u64() == Some(recorded.len() as u64)
        && logged["differing"].as_array().is_some_and(Vec::is_empty);
    (Value::Object(counted), complete)
}

/// What this host made of the vector, in the vectors' terms.
fn project(observed: &Observed, exclusions: &[Vec<String>], vector: &Value) -> Value {
    let (counted, complete) = statements(observed);
    let (result, code) = match &observed.answer {
        Answer::Error { code, .. } => ("refusal", Value::from(code.clone())),
        Answer::Predicted(_) => ("success", Value::Null),
        other => {
            return json!({"outcome": format!("no answer of predict: {other:?}")});
        }
    };
    if result == "refusal" {
        return json!({
            "outcome": result,
            "code": code,
            "statements": counted,
            "complete": complete,
        });
    }
    let Answer::Predicted(predicted) = &observed.answer else {
        unreachable!()
    };
    json!({
        "outcome": "success",
        "code": null,
        "schema": observed.described.as_ref().map_or(Value::Null, schema_of_core),
        "payload": [payload(&predicted["body"], exclusions)],
        "rows": predicted["result"]["rows"].as_array().map(Vec::len),
        "columns": columns_of_core(observed, Some(&predicted["body"]), &vector["columns"]),
        "statements": counted,
        "complete": complete,
    })
}

/// The reference's answer in this host's terms.
fn wanted(vector: &Value) -> Value {
    let reference = &vector["reference"];
    let outcome = &reference["outcome"];
    if outcome["result"] == "refusal" {
        return json!({
            "outcome": "refusal",
            "code": outcome["code"],
            "statements": vector["statements"],
            "complete": true,
        });
    }
    let payload: Vec<Value> = reference["payload"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| json!({"digest": p["digest"], "tables": p["tables"]}))
        .collect();
    let mut columns = Map::new();
    for (name, column) in reference["columns"].as_object().unwrap() {
        columns.insert(
            name.clone(),
            json!({
                "stype": column["described"]["stype"],
                "request": by_value(&column["request"]),
            }),
        );
    }
    json!({
        "outcome": "success",
        "code": null,
        "schema": schema_of_reference(&reference["schema"]),
        "payload": payload,
        "rows": reference["result"]["rows"],
        "columns": columns,
        "statements": vector["statements"],
        "complete": true,
    })
}

/// Whether this host's answer is the reference's, as far as it observes,
/// its statement counts included.
fn agrees(got: &Value, want: &Value) -> bool {
    if want["outcome"] == "refusal" {
        // An unnamed exception has no code: no core answer is it.
        return !want["code"].is_null()
            && got["outcome"] == "refusal"
            && got["code"] == want["code"]
            && got["statements"] == want["statements"]
            && got["complete"] == true;
    }
    got == want
}

/// Where two answers differ, by part.
fn differing(got: &Value, want: &Value) -> Vec<String> {
    [
        "outcome",
        "code",
        "schema",
        "payload",
        "rows",
        "columns",
        "statements",
        "complete",
    ]
    .iter()
    .filter(|part| got.get(**part) != want.get(**part))
    .map(|part| (*part).to_owned())
    .collect()
}

// --- the tests ----------------------------------------------------------------

#[test]
fn the_vectors_are_the_reference_s_and_not_vacuous() {
    let all = vectors();
    let vectors = all["vectors"].as_array().unwrap();
    assert!(vectors.len() > 150, "{} vectors", vectors.len());
    let reference = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = all["reference"].as_str().unwrap();
    assert!(
        reference.contains(&format!("reference: {commit}")),
        "recorded at the reference of contract/reference-env.txt"
    );
    assert_eq!(
        all["payload_exclusions"],
        json!([["metadata", "request_id"]])
    );
    let mut seen = BTreeSet::new();
    let (mut success, mut refusal, mut marked) = (0, 0, 0);
    for vector in vectors {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        match vector["reference"]["outcome"]["result"].as_str() {
            Some("success") => {
                success += 1;
                assert_eq!(
                    vector["reference"]["payload"].as_array().map(Vec::len),
                    Some(1),
                    "{id}"
                );
            }
            Some("refusal") => refusal += 1,
            other => panic!("{id}: an answer the vectors do not hold: {other:?}"),
        }
        let required = vector["register"]["required"].as_bool().unwrap();
        let why = vector["register"]["why"].as_array().unwrap();
        assert_eq!(required, !why.is_empty(), "{id}: a mark says why");
        marked += usize::from(required);
        assert!(
            !vector["core_indices"].as_array().unwrap().is_empty(),
            "{id}"
        );
        assert!(
            vector["statements"]["from_source"].as_u64().unwrap() > 0,
            "{id}"
        );
        assert!(!vector["columns"].as_array().unwrap().is_empty(), "{id}");
    }
    assert!(success > 120, "{success} {refusal}");
    // At 6298913c (main's eb6c68eb reads a DECIMAL column as int64 or
    // float64 where its values fit) the reference refuses six, each by
    // name: a sum past the declared precision of a decimal whose values do
    // not fit int64 (kept as a decimal), and the four decimal time columns.
    let refused: Vec<&str> = vectors
        .iter()
        .filter(|v| v["reference"]["outcome"]["result"] == "refusal")
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(refusal, refused.len());
    let mut refused = refused;
    refused.sort_unstable();
    assert_eq!(
        refused,
        [
            // The reference reads the DECIMAL(38, 2) primary key's
            // 10**17 + i as float64, merging the users (the pair matrix).
            "key.pair-merging.d38-2.d38-0",
            "overflow.d19-0.sum-past",
            "overflow.d38-0.sum-past",
            "time.epoch-declared",
            "time.epoch-in-place",
            "time.epoch-in-place-declared",
            "time.yyyymmdd-declared",
        ]
    );
    assert!(
        vectors
            .iter()
            .filter(|v| v["reference"]["outcome"]["result"] == "refusal")
            .all(|v| v["reference"]["outcome"]["code"] == "INVALID_REQUEST")
    );
    // None fails unnamed or with INTERNAL_ERROR, so none is marked; the
    // comparison's test holds a marked vector, seeded.
    assert_eq!(marked, 0);
}

#[test]
fn the_pending_list_only_shrinks() {
    let pending = read_json("contract/tests/t132b-pending.json");
    let all: BTreeSet<String> = vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap().to_owned())
        .collect();
    let now = ids(&pending["pending"]["rust-host"]);
    let authored = ids(&pending["authored"]["rust-host"]);
    assert!(
        now.is_subset(&authored),
        "pending ids added: {:?}",
        now.difference(&authored)
    );
    assert!(now.is_subset(&all), "pending ids that are no vector");
}

/// Runs every vector whose id starts with `prefix` on this host, holding
/// each to the reference and to the pending list.
fn kind(prefix: &str) {
    let all = vectors();
    let exclusions: Vec<Vec<String>> = all["payload_exclusions"]
        .as_array()
        .unwrap()
        .iter()
        .map(strings)
        .collect();
    let question = all["question"].as_str().unwrap();
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "decimal.feature-38-0-wide")
        .expect("decimal.feature-38-0-wide is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (entries, _) = transitions(&register);
    let pending = ids(&read_json("contract/tests/t132b-pending.json")["pending"]["rust-host"]);
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        let id = vector["id"].as_str().unwrap();
        if !id.starts_with(prefix) {
            continue;
        }
        ran += 1;
        let want = wanted(vector);
        let case = the_case(&template, vector, question);
        let got = match host::run(&case, &bases) {
            Ok(observed) => project(&observed, &exclusions, vector),
            Err(failure) => json!({"outcome": "host failure", "why": failure.to_string()}),
        };
        let ok = passes(id, &got, &want, &entries);
        match (ok, pending.contains(id)) {
            (true, true) => passing_but_pending.push(id.to_owned()),
            (false, false) => failures.push(format!(
                "{id}: {:?} differ: got {} {}, the reference {} {}",
                differing(&got, &want),
                got["outcome"],
                got["code"],
                want["outcome"],
                want["code"]
            )),
            _ => {}
        }
    }
    assert!(ran > 0, "{prefix}: no vectors");
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132b-pending.json's \
         pending[\"rust-host\"]: {passing_but_pending:?}"
    );
    assert!(
        failures.is_empty(),
        "{} vectors answered otherwise than the reference, with no register \
         entry naming them:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// One test per kind of input, so the kinds run in parallel.
macro_rules! kinds {
    ($($name:ident => $prefix:literal,)*) => {
        $(
            #[test]
            fn $name() {
                kind($prefix);
            }
        )*

        #[test]
        fn every_vector_is_of_a_tested_kind() {
            let prefixes = [$($prefix),*];
            for vector in vectors()["vectors"].as_array().unwrap() {
                let id = vector["id"].as_str().unwrap();
                assert!(
                    prefixes.iter().filter(|p| id.starts_with(*p)).count() == 1,
                    "{id} is run by exactly one test"
                );
            }
        }
    };
}

kinds! {
    rust_host_handles_entity_table_decimal_features_as_the_reference => "feature.users.",
    rust_host_handles_fact_table_decimal_features_as_the_reference => "feature.orders.",
    rust_host_handles_integers_past_38_digits_as_the_reference => "wide.",
    rust_host_handles_decimal_targets_as_the_reference => "target.",
    rust_host_handles_sums_past_the_declared_precision_as_the_reference => "overflow.",
    rust_host_handles_decimal_classification_targets_as_the_reference => "class.",
    rust_host_handles_decimal_keys_as_the_reference => "key.",
    rust_host_handles_decimals_beside_the_time_column_as_the_reference => "time.",
}

#[test]
fn the_payload_digest_and_columns_are_the_recorder_s() {
    // The same bytes as the recorder's `digest` and the same columns as its
    // `columns_projection`, on recorded evidence: the reference's own
    // request for each case of the family (contract/reference/<commit>.jsonl),
    // read here, is what the recorder wrote for it in the vectors' `cases`
    // (test_t132b_decimal_family.py holds that section to the recorder).
    let env = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = env
        .lines()
        .find_map(|l| l.strip_prefix("reference: "))
        .unwrap()
        .trim()
        .to_owned();
    let text =
        std::fs::read_to_string(root().join(format!("contract/reference/{commit}.jsonl"))).unwrap();
    let all = vectors();
    let exclusions: Vec<Vec<String>> = all["payload_exclusions"]
        .as_array()
        .unwrap()
        .iter()
        .map(strings)
        .collect();
    let mut read = 0;
    for (case, recorded) in all["cases"].as_object().unwrap() {
        let observation: Value = text
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .find(|o| o["case"] == case.as_str())
            .unwrap();
        let Some(parsed) = observation["public"]["payloads"][0].get("parsed") else {
            assert!(recorded["payload"].is_null(), "{case}");
            continue;
        };
        read += 1;
        assert_eq!(
            payload(parsed, &exclusions),
            json!({
                "digest": recorded["payload"][0]["digest"],
                "tables": recorded["payload"][0]["tables"],
            }),
            "{case}"
        );
        for (name, column) in recorded["columns"].as_object().unwrap() {
            let (table, col) = name.split_once('.').unwrap();
            for part in ["context", "predict"] {
                assert_eq!(
                    column_in(parsed, part, table, col),
                    column["request"][0][part],
                    "{case} {name} {part}"
                );
            }
        }
    }
    // At 6298913c ten of the family's fifteen cases reach a request: the
    // decimal class target (main's eb6c68eb) beside the wide feature and the
    // decimal target, and seven of T2.0c's exact-decimal cases (the others
    // are refused at predict for one class, or at materialize).
    assert_eq!(read, 10, "the ten cases that reach a request");
    for case in [
        "decimal.class-target-38-18",
        "decimal.compare-finer-gt",
        "decimal.compare-finer-negative",
        "decimal.compare-same-scale",
        "decimal.feature-38-0-wide",
        "decimal.feature-inexact-38-18",
        "decimal.key-wide",
        "decimal.sum-10-2",
        "decimal.target-38-18",
        "decimal.target-regression-collide",
    ] {
        assert!(!all["cases"][case]["payload"].is_null(), "{case}");
    }
    // The wide decimal reached the model as text, every digit kept.
    let wide = &all["cases"]["decimal.feature-38-0-wide"]["columns"]["users.f"];
    let context = &wide["request"][0]["context"];
    assert_eq!(context["dtype"], "string");
    assert_eq!(
        context["cells"]["\"12345678901234567890123456789\""],
        json!(14)
    );
    // A recorded vector's own cells read back through cell_text: the
    // recorder's keys are this function's output.
    let vector = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "feature.users.d4-2.spread")
        .unwrap();
    let cells = &vector["reference"]["columns"]["users.f"]["request"][0]["context"]["cells"];
    for key in cells.as_object().unwrap().keys() {
        let cell: Value = serde_json::from_str(key).unwrap();
        assert_eq!(&cell_text(&cell), key);
    }
    assert_eq!(cell_text(&json!({"number": "1.5"})), "{\"number\":\"1.5\"}");
    assert_eq!(cell_text(&Value::Null), "null");
    // The number forms: 1e-05, 1e-5 and 0.00001 agree; 1 and 1.0 do not.
    assert_eq!(number_form("1e-05"), number_form("0.00001"));
    assert_ne!(number_form("1"), number_form("1.0"));
}

#[test]
fn the_comparison_fails_closed() {
    // A success: the parts this host observes each count.
    let all = vectors();
    let find = |id: &str| {
        all["vectors"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["id"] == id)
            .unwrap()
            .clone()
    };
    let vector = find("feature.users.d4-2.spread");
    let want = wanted(&vector);
    assert!(agrees(&want, &want));
    for (part, value) in [
        ("outcome", json!("refusal")),
        ("schema", json!(null)),
        ("payload", json!([])),
        ("rows", json!(0)),
        ("columns", json!({})),
        ("statements", json!({"from_source": 0})),
        ("complete", json!(false)),
    ] {
        let mut got = want.clone();
        got[part] = value;
        assert!(!agrees(&got, &want), "{part}");
    }
    let mut got = want.clone();
    got["columns"]["users.f"]["request"][0]["context"]["dtype"] = json!("string");
    assert!(!agrees(&got, &want), "another wire dtype");
    let mut got = want.clone();
    got["columns"]["users.f"]["stype"] = json!("timestamp");
    assert!(!agrees(&got, &want), "another stype");
    // Cells by value: one number written two ways is one cell; a number
    // of another kind, another value, or text is not.
    let cells = |keys: &[(&str, u64)]| {
        let map: Map<String, Value> = keys
            .iter()
            .map(|(k, n)| ((*k).to_owned(), json!(n)))
            .collect();
        by_value(
            &json!([{"context": {"dtype": "float64", "stype": "numerical", "cells": map}, "predict": null}]),
        )
    };
    let core = cells(&[("{\"number\":\"1e-9\"}", 2), ("null", 1)]);
    assert_eq!(core, cells(&[("{\"number\":\"1e-09\"}", 2), ("null", 1)]));
    assert_eq!(
        core,
        cells(&[
            ("{\"number\":\"0.000000001\"}", 1),
            ("{\"number\":\"1E-9\"}", 1),
            ("null", 1)
        ])
    );
    assert_ne!(core, cells(&[("{\"number\":\"1e-8\"}", 2), ("null", 1)]));
    assert_ne!(core, cells(&[("{\"number\":\"-1e-9\"}", 2), ("null", 1)]));
    assert_ne!(core, cells(&[("\"1e-9\"", 2), ("null", 1)]));
    assert_ne!(core, cells(&[("{\"number\":\"1e-9\"}", 3)]));
    assert_ne!(
        cells(&[("{\"number\":\"1\"}", 1)]),
        cells(&[("{\"number\":\"1.0\"}", 1)])
    );
    assert_ne!(cells(&[("\"1e-9\"", 1)]), cells(&[("\"1e-09\"", 1)]));
    // A refusal: the same code and counts, and an unnamed one is never
    // matched.
    let past = find("overflow.d38-0.sum-past");
    let named = wanted(&past);
    let mut refused = named.clone();
    assert!(agrees(&refused, &named));
    refused["statements"]["materialize"] = json!(0);
    assert!(!agrees(&refused, &named), "a refusal's counts");
    let mut other = named.clone();
    other["code"] = json!("UNSUPPORTED_REQUEST");
    assert!(!agrees(&other, &named));
    let unnamed = json!({"outcome": "refusal", "code": null, "statements": {}, "complete": true});
    assert!(!agrees(&unnamed, &unnamed));
    // The transitions: an approved entry covers exactly the transition it
    // declares, for the vectors its glob names; a proposed one, nothing.
    // class.static-three.d4-0 as the reference answered it at b413e9ad
    // (INTERNAL_ERROR at materialize), seeded: no recorded vector fails so
    // at 6298913c.
    let mut seeded = find("class.static-three.d4-0");
    seeded["reference"]["outcome"] =
        json!({"result": "refusal", "stage": "materialize", "code": "INTERNAL_ERROR"});
    let marked = wanted(&seeded);
    let mut core = marked.clone();
    core["code"] = json!("UNSUPPORTED_REQUEST");
    let entry = |status: &str, glob: &str, old: &str, new: &str| {
        (
            "E-1".to_owned(),
            status.to_owned(),
            glob.to_owned(),
            old.to_owned(),
            new.to_owned(),
        )
    };
    assert_eq!(outcome_class(&marked), "refusal INTERNAL_ERROR");
    let declared = "refusal UNSUPPORTED_REQUEST";
    let id = "class.static-three.d4-0";
    assert!(!passes(id, &core, &marked, &[]));
    let proposed = [entry(
        "proposed",
        "class.static-*",
        "refusal INTERNAL_ERROR",
        declared,
    )];
    assert!(!passes(id, &core, &marked, &proposed));
    let approved = [entry(
        "approved",
        "class.static-*",
        "refusal INTERNAL_ERROR",
        declared,
    )];
    assert!(passes(id, &core, &marked, &approved));
    assert!(!passes("key.both.d4-0", &core, &marked, &approved));
    let success = json!({"outcome": "success", "code": null});
    assert!(!passes(id, &success, &marked, &approved));
    let wrong = [entry(
        "approved",
        "class.static-*",
        "refusal INVALID_REQUEST",
        declared,
    )];
    assert!(!passes(id, &core, &marked, &wrong));
    // An unmarked vector, a sum past the declared precision refused by
    // name, is excused the same way only.
    let taken = wanted(&find("overflow.d38-0.sum-at"));
    assert!(!passes("overflow.d38-0.sum-past", &taken, &named, &[]));
    let lifted = [entry(
        "approved",
        "overflow.*-past",
        "refusal INVALID_REQUEST",
        "success",
    )];
    assert!(passes("overflow.d38-0.sum-past", &taken, &named, &lifted));
    let failed = json!({"outcome": "host failure"});
    assert!(!passes("overflow.d38-0.sum-past", &failed, &named, &lifted));
    // A declared success -> success still holds the counts.
    let same = [entry("approved", "feature.*", "success", "success")];
    let mut changed = want.clone();
    changed["columns"]["users.f"]["stype"] = json!("timestamp");
    assert!(passes("feature.users.d4-2.spread", &changed, &want, &same));
    changed["statements"]["materialize"] = json!(99);
    assert!(!passes("feature.users.d4-2.spread", &changed, &want, &same));
    // The register's text: refs of proposed and approved entries, in both
    // forms; a rejected entry's and another file's ignored; a ref naming
    // vectors with no transition malformed.
    let text = "entries:\n- id: A-1\n  status: approved\n  refs: ['contract/decimal-vectors.json#class.static-* refusal INTERNAL_ERROR -> refusal UNSUPPORTED_REQUEST']\n\
                - id: A-2\n  status: proposed\n  refs:\n  - contract/decimal-vectors.json#key.* refusal INTERNAL_ERROR -> success\n\
                - id: A-3\n  status: rejected\n  refs:\n  - contract/decimal-vectors.json#key.* refusal INTERNAL_ERROR -> success\n\
                - id: A-4\n  status: approved\n  refs:\n  - contract/empty-null-vectors.json#null.* success -> success\n  - contract/decimal-vectors.json#key.*\n  - contract/decimal-vectors.json#key.* refusal internal_error -> success\n";
    let (read, malformed) = transitions(text);
    assert_eq!(
        read.iter()
            .map(|(e, s, g, o, n)| format!("{e} {s} {g} {o} -> {n}"))
            .collect::<Vec<_>>(),
        vec![
            "A-1 approved class.static-* refusal INTERNAL_ERROR -> refusal UNSUPPORTED_REQUEST",
            "A-2 proposed key.* refusal INTERNAL_ERROR -> success",
        ]
    );
    assert_eq!(
        malformed,
        vec![
            "A-4: contract/decimal-vectors.json#key.*",
            "A-4: contract/decimal-vectors.json#key.* refusal internal_error -> success",
        ]
    );
    assert!(glob("class.static-*", "class.static-three.d4-0"));
    assert!(!glob("class.static-*", "class.binary-static.d4-2"));
}

#[test]
fn every_vector_ref_names_its_transition() {
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (_, malformed) = transitions(&register);
    assert!(
        malformed.is_empty(),
        "vector refs naming no transition `<glob> <from> -> <to>`: {malformed:?}"
    );
}
