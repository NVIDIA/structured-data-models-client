// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32e part 2's multicategorical port on `rust-host`, held from outside
//! the implementation (the package's independent test author) to what the
//! reference gave: `contract/multicategorical-vectors.json`, recorded from
//! the reference by `contract/tools/multicategorical_record.py`, which
//! `contract/tests/test_t132e_multicategorical.py` holds `python-core` to
//! as well, so the two hosts agree through it.
//!
//! - "Port the reference's multi-category detection and feature handling
//!   faithfully" (its stype for multi-category text, its separator
//!   detection, how it encodes the values on the wire) and "stop misfiring
//!   on ordinary text": each vector (`shop` changed by the vector's `setup`:
//!   a text column `f` with each separator the reference counts and some it
//!   does not, the ratio's and the part count's boundaries, times of day,
//!   URLs, ratios, free text, two separators, nulls, empty strings, one
//!   category, 1,200-row tables whose values past row 500 or the preview
//!   differ, the column on the fact table, questions naming it) runs
//!   through this host as `feature.bool` does, through `describe_source`,
//!   `materialize` and `predict` against the scripted endpoint. The answer
//!   must be the reference's, as far as this host observes it: a refusal
//!   with the reference's code (an unnamed exception is matched by no
//!   code); or a success whose description is the reference's public
//!   schema (each table's columns by name and stype in order, its key, its
//!   time column, its duplicate keys, and the links), whose model call's
//!   body is the reference's request (its digest as the `payload` rule
//!   compares it, and the rows and null cells of each table it carries),
//!   whose `f` travels as the reference's did (per table carrying it: the
//!   request schema's dtype and stype for it, its row, null and distinct
//!   counts, the kinds of its cells, the first distinct cells in sorted
//!   order, the digest of every cell in order, and the cells themselves up
//!   to 20 rows), and whose result has the reference's row count.
//! - A vector may differ from the reference only under an approved entry
//!   of `docs/design/rust-core/register.yaml` whose `refs` name it with the
//!   transition it makes (T1.32a's binding-transition form):
//!   `contract/multicategorical-vectors.json#<id or glob> <from> -> <to>`,
//!   each side `success` or `refusal <CODE>`. A proposed entry excuses
//!   nothing; the recorder's `register.required` excuses nothing by itself.
//! - Reviewers: a ref attached to an already approved entry takes effect
//!   here at once, though only the owner approves entries; each newly
//!   attached ref needs a reviewer's check that the entry's approved
//!   behaviour is that transition for exactly the vectors its glob names.
//!
//! Gating: `contract/tests/t132e-pending.json`'s `pending["mc-rust-host"]`
//! lists the vectors that failed here when the test was written. Each must
//! still fail (strict: one that passes must be taken out of the list), and
//! the list may only shrink: it stays within `authored["mc-rust-host"]`.
//! The implementer removes ids as they pass and changes nothing else of
//! this file, the vectors or the recorder.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case, Index};
use sdm_testkit::host::{self, Answer, Observed};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

const REF: &str = "contract/multicategorical-vectors.json#";
/// A table's cells of the column kept whole up to this many rows, and the
/// distinct cells a summary lists (the recorder's `WHOLE` and `LISTED`).
const WHOLE: usize = 20;
const LISTED: usize = 20;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = read_json("contract/multicategorical-vectors.json");
    assert_eq!(all["format"], "sdm-multicategorical-vectors/1");
    assert_eq!(
        all["recorded_by"],
        "contract/tools/multicategorical_record.py"
    );
    assert_eq!(all["template"], "contract/cases/s1/feature.bool.json");
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

fn pending(group: &str) -> BTreeSet<String> {
    ids(&read_json("contract/tests/t132e-pending.json")["pending"][group])
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
/// difference an approved entry declares exactly.
fn passes(id: &str, got: &Value, want: &Value, entries: &[Transition]) -> bool {
    if agrees(got, want) {
        return true;
    }
    let answered = got["outcome"] == "success" || got["outcome"] == "refusal";
    answered && covered(id, &outcome_class(want), &outcome_class(got), entries).is_some()
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

/// The case this host runs for `vector`: `feature.bool`'s, its setup,
/// question and entities the vector's, every table described.
fn the_case(template: &Case, vector: &Value, question: &str, indices: &Value) -> Case {
    let mut case = template.clone();
    case.id = format!("t132e2.{}", vector["id"].as_str().unwrap());
    case.input.setup = Some(strings(&vector["setup"]));
    case.input.pql = vector["pql"].as_str().unwrap_or(question).to_owned();
    let entities: Vec<i64> = vector
        .get("indices")
        .unwrap_or(indices)
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_i64().unwrap())
        .collect();
    case.input.materialize.indices = entities
        .iter()
        .map(|i| Index::Integer(i.to_string()))
        .collect();
    let core = case.core.as_mut().expect("the template's core inputs");
    core.describe_source
        .as_mut()
        .expect("the template describes its source")
        .source = json!({"tables": [], "primary_keys": {}, "time_columns": {}});
    core.materialize
        .as_mut()
        .expect("the template materializes")
        .indices = entities
        .iter()
        .map(|i| json!({"t": "i64", "v": i.to_string()}))
        .collect();
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

// --- the column on the wire ---------------------------------------------------

/// Each table a request body carries rows of, by its path in the body
/// (the recorder's `_sections`).
fn sections(body: &Value) -> Vec<(String, &Value)> {
    let mut out = Vec::new();
    for section in ["context", "predict"] {
        let Some(part) = body.get(section).and_then(Value::as_object) else {
            continue;
        };
        if let Some(instance) = part.get("instance_table").filter(|t| t.is_object()) {
            out.push((format!("{section}.instance_table"), instance));
        }
        if let Some(related) = part.get("related_tables").and_then(Value::as_object) {
            for (name, table) in related {
                if table.is_object() {
                    out.push((format!("{section}.related_tables.{name}"), table));
                }
            }
        }
    }
    out
}

/// The request schema's declaration of `column` in the table at `path`.
fn declared(body: &Value, path: &str, column: &str) -> Value {
    let schema = &body["schema"];
    let table = if path.ends_with(".instance_table") {
        &schema["instance_table"]
    } else {
        &schema["related_tables"][path.rsplit('.').next().unwrap()]
    };
    let found = &table["columns"][column];
    if !found.is_object() {
        return Value::Null;
    }
    json!({"dtype": found.get("dtype"), "stype": found.get("stype")})
}

fn kind(cell: &Value) -> &'static str {
    match cell {
        Value::Null => "null",
        Value::String(_) => "string",
        Value::Array(_) => "list",
        Value::Object(o) if o.len() == 1 && o.contains_key("number") => "number",
        Value::Bool(_) => "bool",
        Value::Number(_) => "int",
        Value::Object(_) => "dict",
    }
}

/// One table's column as sent (the recorder's `column_on_the_wire`).
fn column_on_the_wire(declared: Value, cells: Vec<Value>) -> Value {
    let distinct: BTreeMap<String, Value> = cells
        .iter()
        .map(|c| (serde_json::to_string(c).unwrap(), c.clone()))
        .collect();
    let kinds: BTreeSet<&str> = cells.iter().map(kind).collect();
    let digest: String = Sha256::digest(
        serde_json::to_string(&canonical(&Value::Array(cells.clone())))
            .unwrap()
            .as_bytes(),
    )
    .iter()
    .map(|b| format!("{b:02x}"))
    .collect();
    let mut out = json!({
        "declared": declared,
        "rows": cells.len(),
        "nulls": cells.iter().filter(|c| c.is_null()).count(),
        "kinds": kinds.into_iter().collect::<Vec<_>>(),
        "distinct": distinct.len(),
        "listed": distinct.values().take(LISTED).cloned().collect::<Vec<_>>(),
        "digest": digest,
    });
    if cells.len() <= WHOLE {
        out["cells"] = Value::Array(cells);
    }
    out
}

/// How `f` travels in a body: per table carrying it.
fn wire(body: &Value) -> Value {
    let mut tables = Map::new();
    for (path, table) in sections(body) {
        let Some(columns) = table["columns"].as_array() else {
            continue;
        };
        let Some(at) = columns.iter().position(|c| c == "f") else {
            continue;
        };
        let cells: Vec<Value> = table["rows"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|row| row.get(at).cloned().unwrap_or(Value::Null))
            .collect();
        tables.insert(
            path.clone(),
            column_on_the_wire(declared(body, &path, "f"), cells),
        );
    }
    Value::Object(tables)
}

/// What this host made of the vector, in the vectors' terms.
/// What stands for the run's day in a vector marked `run_day` (the
/// recorder's `RUN_DAY`).
const RUN_DAY: &str = "<run-day>";

/// The day of a timestamp cell, `YYYY-MM-DD` before a `T`, and the rest.
fn timestamp_day(cell: &str) -> Option<(&str, &str)> {
    let (day, rest) = (cell.get(..10)?, cell.get(10..)?);
    let b = day.as_bytes();
    let digits = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    (digits(0..4)
        && b[4] == b'-'
        && digits(5..7)
        && b[7] == b'-'
        && digits(8..10)
        && rest.starts_with('T'))
    .then_some((day, rest))
}

/// A body with the day of each timestamp cell of `f` written as
/// `RUN_DAY`, where every such cell names the same day (the recorder's
/// `at_run_day`): text times of day, which the reference reads as times on
/// the day it runs. Cells of two days, or of another column, are left.
fn at_run_day(body: &Value) -> Value {
    let mut days = BTreeSet::new();
    for (_, table) in sections(body) {
        let Some(at) = table["columns"]
            .as_array()
            .and_then(|c| c.iter().position(|n| n == "f"))
        else {
            continue;
        };
        for row in table["rows"].as_array().into_iter().flatten() {
            if let Some((day, _)) = row.get(at).and_then(Value::as_str).and_then(timestamp_day) {
                days.insert(day.to_owned());
            }
        }
    }
    if days.len() != 1 {
        return body.clone();
    }
    let mut out = body.clone();
    for section in ["context", "predict"] {
        let Some(part) = out.get_mut(section).and_then(Value::as_object_mut) else {
            continue;
        };
        let mut tables: Vec<&mut Value> = Vec::new();
        for (key, value) in part.iter_mut() {
            match key.as_str() {
                "instance_table" => tables.push(value),
                "related_tables" => {
                    if let Some(related) = value.as_object_mut() {
                        tables.extend(related.values_mut());
                    }
                }
                _ => {}
            }
        }
        for table in tables {
            let Some(at) = table["columns"]
                .as_array()
                .and_then(|c| c.iter().position(|n| n == "f"))
            else {
                continue;
            };
            for row in table["rows"].as_array_mut().into_iter().flatten() {
                let Some(cell) = row.get_mut(at) else {
                    continue;
                };
                if let Some((_, rest)) = cell.as_str().and_then(timestamp_day) {
                    *cell = Value::from(format!("{RUN_DAY}{rest}"));
                }
            }
        }
    }
    out
}

fn project(observed: &Observed, exclusions: &[Vec<String>], run_day: bool) -> Value {
    let (result, code) = match &observed.answer {
        Answer::Error { code, .. } => ("refusal", Value::from(code.clone())),
        Answer::Predicted(_) => ("success", Value::Null),
        other => {
            return json!({"outcome": format!("no answer of predict: {other:?}")});
        }
    };
    if result == "refusal" {
        return json!({"outcome": result, "code": code});
    }
    let Answer::Predicted(predicted) = &observed.answer else {
        unreachable!()
    };
    let body = if run_day {
        at_run_day(&predicted["body"])
    } else {
        predicted["body"].clone()
    };
    let body = &body;
    json!({
        "outcome": "success",
        "code": null,
        "schema": observed.described.as_ref().map_or(Value::Null, schema_of_core),
        "payload": [payload(body, exclusions)],
        "wire": [wire(body)],
        "rows": predicted["result"]["rows"].as_array().map(Vec::len),
    })
}

/// The reference's answer in this host's terms.
fn wanted(reference: &Value) -> Value {
    let outcome = &reference["outcome"];
    if outcome["result"] == "refusal" {
        return json!({"outcome": "refusal", "code": outcome["code"]});
    }
    let payload: Vec<Value> = reference["payload"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|p| json!({"digest": p["digest"], "tables": p["tables"]}))
        .collect();
    json!({
        "outcome": "success",
        "code": null,
        "schema": schema_of_reference(&reference["schema"]),
        "payload": payload,
        "wire": reference["wire"],
        "rows": reference["result"]["rows"],
    })
}

/// Whether this host's answer is the reference's, as far as it observes.
fn agrees(got: &Value, want: &Value) -> bool {
    if want["outcome"] == "refusal" {
        return !want["code"].is_null()
            && got["outcome"] == "refusal"
            && got["code"] == want["code"];
    }
    got == want
}

fn differing(got: &Value, want: &Value) -> Vec<String> {
    ["outcome", "code", "schema", "payload", "wire", "rows"]
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
    assert!(vectors.len() > 70, "{} vectors", vectors.len());
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
    let (mut success, mut refusal, mut marked, mut multi) = (0, 0, 0, 0);
    for vector in vectors {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        match vector["reference"]["outcome"]["result"].as_str() {
            Some("success") => {
                success += 1;
                let wire = &vector["reference"]["wire"];
                assert_eq!(wire.as_array().map(Vec::len), Some(1), "{id}");
                for column in wire[0].as_object().unwrap().values() {
                    if column["declared"]["stype"] == "multicategorical" {
                        multi += 1;
                        assert_eq!(column["declared"]["dtype"], "string", "{id}");
                    }
                }
            }
            Some("refusal") => refusal += 1,
            other => panic!("{id}: an answer the vectors do not hold: {other:?}"),
        }
        let required = vector["register"]["required"].as_bool().unwrap();
        let why = vector["register"]["why"].as_array().unwrap();
        assert_eq!(required, !why.is_empty(), "{id}: a mark says why");
        marked += usize::from(required);
    }
    assert!(success > 50 && refusal > 5, "{success} {refusal}");
    assert!(multi > 40, "{multi} multi-category columns on the wire");
    assert!(marked > 0 && marked < vectors.len());
}

#[test]
fn the_pending_list_only_shrinks() {
    let gating = read_json("contract/tests/t132e-pending.json");
    let all: BTreeSet<String> = vectors()["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["id"].as_str().unwrap().to_owned())
        .collect();
    for group in ["mc-python-core", "mc-rust-host", "p2-exits"] {
        let now = ids(&gating["pending"][group]);
        let authored = ids(&gating["authored"][group]);
        assert!(
            now.is_subset(&authored),
            "{group}: pending ids added: {:?}",
            now.difference(&authored)
        );
    }
    assert!(
        pending("mc-rust-host").is_subset(&all),
        "pending ids that are no vector"
    );
}

/// Runs every vector whose id starts with `prefix` on this host, holding
/// each to the reference and to the pending list.
fn kind_of(prefix: &str) {
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
        .find(|l| l.case.id == "feature.bool")
        .expect("feature.bool is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (entries, _) = transitions(&register);
    let pending = pending("mc-rust-host");
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        let id = vector["id"].as_str().unwrap();
        if !id.starts_with(prefix) {
            continue;
        }
        ran += 1;
        let want = wanted(&vector["reference"]);
        let case = the_case(&template, vector, question, &all["indices"]);
        let got = match host::run(&case, &bases) {
            Ok(observed) => project(
                &observed,
                &exclusions,
                vector["run_day"].as_bool().unwrap_or(false),
            ),
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
        "these pass now: take them out of contract/tests/t132e-pending.json's \
         pending[\"mc-rust-host\"]: {passing_but_pending:?}"
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
                kind_of($prefix);
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
    rust_host_reads_each_separator_as_the_reference => "sep.",
    rust_host_reads_ordinary_text_as_the_reference => "text.",
    rust_host_reads_two_separators_as_the_reference => "mixed.",
    rust_host_reads_nulls_among_the_values_as_the_reference => "null.",
    rust_host_reads_empty_strings_as_the_reference => "empty.",
    rust_host_reads_a_single_category_as_the_reference => "single.",
    rust_host_reads_values_past_the_preview_as_the_reference => "large.",
    rust_host_reads_the_fact_table_s_column_as_the_reference => "fact.",
    rust_host_answers_questions_naming_it_as_the_reference => "question.",
}

#[test]
fn the_wire_reading_is_the_recorder_s() {
    // The same bytes and summary as the recorder's `column_on_the_wire`,
    // on recorded evidence: every success's `wire`, read back from its own
    // listed cells where they are whole, gives the recorded summary.
    let all = vectors();
    let mut checked = 0;
    for vector in all["vectors"].as_array().unwrap() {
        let Some(tables) = vector["reference"]["wire"][0].as_object() else {
            continue;
        };
        for (path, column) in tables {
            let Some(cells) = column["cells"].as_array() else {
                continue;
            };
            let again = column_on_the_wire(column["declared"].clone(), cells.clone());
            assert_eq!(&again, column, "{} {path}", vector["id"]);
            checked += 1;
        }
    }
    assert!(checked > 50, "{checked} whole columns read back");
    // The digest reads numbers by kind and exact value, and keeps order and
    // nulls.
    let digest = |cells: Value| {
        column_on_the_wire(Value::Null, cells.as_array().unwrap().clone())["digest"].clone()
    };
    assert_ne!(digest(json!(["a;b", null])), digest(json!(["a;b"])));
    assert_ne!(digest(json!(["a;b", "c"])), digest(json!(["c", "a;b"])));
    assert_eq!(
        digest(json!([{"number": "1e-5"}])),
        digest(json!([{"number": "0.00001"}]))
    );
    assert_ne!(
        digest(json!([{"number": "1"}])),
        digest(json!([{"number": "1.0"}]))
    );
    let small = column_on_the_wire(Value::Null, vec![json!("b;a"), Value::Null, json!("a")]);
    assert_eq!(small["listed"], json!(["a", "b;a", null]));
    assert_eq!(small["kinds"], json!(["null", "string"]));
    let big = column_on_the_wire(Value::Null, vec![json!("x"); WHOLE + 1]);
    assert!(big.get("cells").is_none());
    // A body: the declaration and the cells of the table carrying `f`.
    let body = json!({
        "schema": {"related_tables": {"users": {"columns": {
            "f": {"dtype": "string", "stype": "multicategorical"}}}}},
        "context": {"related_tables": {"users": {
            "columns": ["user_id", "f"], "rows": [[1, "a;b"]]}}},
    });
    let read = wire(&body);
    assert_eq!(
        read["context.related_tables.users"]["declared"],
        json!({"dtype": "string", "stype": "multicategorical"})
    );
    assert_eq!(
        read["context.related_tables.users"]["cells"],
        json!(["a;b"])
    );
}

#[test]
fn the_comparison_fails_closed() {
    let all = vectors();
    let vector = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "sep.semicolon")
        .unwrap();
    let want = wanted(&vector["reference"]);
    assert!(agrees(&want, &want));
    for (part, value) in [
        ("outcome", json!("refusal")),
        ("schema", json!(null)),
        ("payload", json!([])),
        ("wire", json!([])),
        ("rows", json!(0)),
    ] {
        let mut got = want.clone();
        got[part] = value;
        assert!(!agrees(&got, &want), "{part}");
    }
    // The column sent as categorical, or as lists of parts, or other cells.
    for (field, value) in [
        (
            "declared",
            json!({"dtype": "string", "stype": "categorical"}),
        ),
        ("kinds", json!(["list"])),
        ("digest", json!("0".repeat(64))),
    ] {
        let mut got = want.clone();
        got["wire"][0]["context.related_tables.users"][field] = value;
        assert!(!agrees(&got, &want), "{field}");
    }
    // Another stype in the description.
    let mut got = want.clone();
    got["schema"]["tables"]["users"]["columns"][3][1] = json!("categorical");
    assert!(!agrees(&got, &want), "another stype");
    // The core's refusal of the column in place of the reference's
    // success: excused only by an approved entry declaring exactly it.
    let refused = json!({"outcome": "refusal", "code": "UNSUPPORTED_REQUEST"});
    let entry = |status: &str, glob: &str, old: &str, new: &str| {
        (
            "E-1".to_owned(),
            status.to_owned(),
            glob.to_owned(),
            old.to_owned(),
            new.to_owned(),
        )
    };
    let declared = "refusal UNSUPPORTED_REQUEST";
    assert!(!passes("sep.semicolon", &refused, &want, &[]));
    let proposed = [entry("proposed", "sep.*", "success", declared)];
    assert!(!passes("sep.semicolon", &refused, &want, &proposed));
    let approved = [entry("approved", "sep.*", "success", declared)];
    assert!(passes("sep.semicolon", &refused, &want, &approved));
    assert!(!passes("text.url-few", &refused, &want, &approved));
    // A refusal: the same code; an unnamed one is never matched.
    let internal = wanted(&json!({"outcome": {"result": "refusal", "code": "INTERNAL_ERROR"}}));
    assert!(agrees(
        &json!({"outcome": "refusal", "code": "INTERNAL_ERROR"}),
        &internal
    ));
    assert!(!agrees(&refused, &internal));
    let unnamed = wanted(&json!({"outcome": {"result": "refusal", "code": null}}));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": null}),
        &unnamed
    ));
    let failed = json!({"outcome": "host failure"});
    assert!(!passes("sep.semicolon", &failed, &want, &approved));
    // The register's text: refs of proposed and approved entries, in both
    // forms; a rejected entry's and another file's ignored; a ref naming
    // vectors with no transition malformed.
    let text = "entries:\n- id: A-1\n  status: approved\n  refs: ['contract/multicategorical-vectors.json#question.* refusal INTERNAL_ERROR -> refusal UNSUPPORTED_REQUEST']\n\
                - id: A-2\n  status: proposed\n  refs:\n  - contract/multicategorical-vectors.json#text.time-* refusal INVALID_REQUEST -> success\n\
                - id: A-3\n  status: rejected\n  refs:\n  - contract/multicategorical-vectors.json#sep.* success -> refusal UNSUPPORTED_REQUEST\n\
                - id: A-4\n  status: approved\n  refs:\n  - contract/empty-null-vectors.json#empty.* success -> success\n  - contract/multicategorical-vectors.json#sep.*\n";
    let (read, malformed) = transitions(text);
    assert_eq!(
        read.iter()
            .map(|(e, s, g, o, n)| format!("{e} {s} {g} {o} -> {n}"))
            .collect::<Vec<_>>(),
        vec![
            "A-1 approved question.* refusal INTERNAL_ERROR -> refusal UNSUPPORTED_REQUEST",
            "A-2 proposed text.time-* refusal INVALID_REQUEST -> success",
        ]
    );
    assert_eq!(
        malformed,
        vec!["A-4: contract/multicategorical-vectors.json#sep.*"]
    );
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

#[test]
fn the_run_day_placeholder_holds_the_vector_on_any_day() {
    // The recorder's reading, on its own evidence and on seeds: the marked
    // vector holds no calendar day in `f`; one day's cells of `f` are
    // written as the placeholder, another column's and two days' are not.
    let all = vectors();
    let marked: Vec<&str> = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|v| v["run_day"] == true)
        .map(|v| v["id"].as_str().unwrap())
        .collect();
    assert_eq!(marked, ["fact.time-of-day"]);
    let body = |f: Vec<Value>| {
        let rows: Vec<Value> = f
            .into_iter()
            .map(|cell| json!(["2023-01-01T00:00:00.000000Z", cell]))
            .collect();
        json!({"context": {"related_tables": {"orders": {"columns": ["ts", "f"], "rows": rows}}}})
    };
    let one = body(vec![
        json!("2031-02-03T09:45:00.000000Z"),
        Value::Null,
        json!("x"),
    ]);
    let read = at_run_day(&one);
    let rows = &read["context"]["related_tables"]["orders"]["rows"];
    assert_eq!(rows[0][1], json!("<run-day>T09:45:00.000000Z"));
    assert_eq!(rows[0][0], json!("2023-01-01T00:00:00.000000Z"));
    assert_eq!(rows[1][1], Value::Null);
    assert_eq!(rows[2][1], json!("x"));
    let two = body(vec![
        json!("2031-02-03T09:45:00.000000Z"),
        json!("2031-02-04T09:45:00.000000Z"),
    ]);
    assert_eq!(at_run_day(&two), two);
    let none = body(vec![json!("a;b")]);
    assert_eq!(at_run_day(&none), none);
    // Read before the digest and the wire: two days agree.
    let other = body(vec![
        json!("2031-02-09T09:45:00.000000Z"),
        Value::Null,
        json!("x"),
    ]);
    assert_eq!(wire(&at_run_day(&one)), wire(&at_run_day(&other)));
    assert_ne!(wire(&one), wire(&other));
    assert_eq!(
        timestamp_day("2031-02-03T09:45"),
        Some(("2031-02-03", "T09:45"))
    );
    assert_eq!(timestamp_day("09:45"), None);
    assert_eq!(timestamp_day("2031-02-03 09:45"), None);
}
