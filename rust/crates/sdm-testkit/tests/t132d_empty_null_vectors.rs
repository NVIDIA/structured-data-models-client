// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32d's empty tables, nulls and refused questions on `rust-host`, held
//! from outside the implementation (the package's independent test author)
//! to what the reference gave: `contract/empty-null-vectors.json`, recorded
//! from the reference by `contract/tools/empty_null_record.py`, which
//! `contract/tests/test_t132d_empty_null_vectors.py` holds `python-core` to
//! as well, so the two hosts agree through it.
//!
//! - Exits "`python-core` and `rust-host` agree on `core` fields" and
//!   "every difference is registered", over the family's inputs: each
//!   vector (`shop` changed by the vector's `setup`, asked its question for
//!   its entities over the tables it names) runs through this host as
//!   `null.feature-all-null` does, through `describe_source`, `materialize`
//!   and `predict` against the scripted endpoint; an empty entity list is
//!   handed to the core as it is, so the core's refusal of it is reached
//!   here too (the gap `contract/refusal-maps/empty-null.json` gives
//!   T1.32d). The answer must be the reference's, as far as this host
//!   observes it: a refusal with the reference's code (this host observes
//!   no stage or exception class; an unnamed exception is matched by no
//!   code); or a success whose description is the reference's public
//!   schema (each table's columns by name and stype in order, its key, its
//!   time column and its duplicate keys, and the links), whose model call's
//!   body is the reference's request (its digest as the `payload` rule
//!   compares it: the vectors' `payload_exclusions` masked to their type,
//!   each number by kind, sign and exact value, sorted keys; and the rows
//!   and null cells of each table it carries), and whose result has the
//!   reference's row count.
//! - A vector may differ from the reference only under an approved entry
//!   of `docs/design/rust-core/register.yaml` whose `refs` name it with the
//!   transition it makes (T1.32a's binding-transition form):
//!   `contract/empty-null-vectors.json#<id or glob> <from> -> <to>`, each
//!   side `success` or `refusal <CODE>`, from the reference's answer to
//!   this host's (the reference's unnamed exception is `refusal UNNAMED`).
//!   A ref covers exactly the transition it declares; a proposed entry
//!   excuses nothing, so its vectors stay pending until the owner approves
//!   it. The core's `UNSUPPORTED_REQUEST` for an empty table (rule
//!   `empty_table`) passes only where an approved entry declares it
//!   (P1B-11, once a ref names the vectors). The recorder's
//!   `register.required` marks where a difference is expected and excuses
//!   nothing by itself; an unmarked vector (a null key, refused by the
//!   reference with INVALID_REQUEST) is held exactly unless an approved
//!   entry declares its transition.
//! - Reviewers: a ref attached to an already approved entry (P1B-11) takes
//!   effect here at once, though only the owner approves entries; each
//!   newly attached ref needs a reviewer's check that the entry's approved
//!   behaviour is that transition for exactly those vectors.
//! - Every vector ref of a proposed or approved entry names a transition
//!   in that form; a malformed one fails.
//!
//! Gating: `contract/tests/t132d-pending.json`'s `pending["rust-host"]`
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

const REF: &str = "contract/empty-null-vectors.json#";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn read_json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = read_json("contract/empty-null-vectors.json");
    assert_eq!(all["format"], "sdm-empty-null-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/empty_null_record.py");
    assert_eq!(
        all["template"],
        "contract/cases/s1/null.feature-all-null.json"
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
            (table.clone(), key)
        })
        .collect();
    let times = from_source["time_columns"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    json!({"tables": tables, "primary_keys": keys, "time_columns": times})
}

/// The case this host runs for `vector`: `null.feature-all-null`'s, its
/// setup, question, entities and tables the vector's. The entities reach
/// the core as given, none included.
fn the_case(template: &Case, vector: &Value, question: &str, indices: &Value) -> Case {
    let mut case = template.clone();
    case.id = format!("t132d.{}", vector["id"].as_str().unwrap());
    let setup = strings(&vector["setup"]);
    case.input.setup = (!setup.is_empty()).then_some(setup);
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
        .source = core_source(vector.get("from_source").unwrap_or(&Value::Null));
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

/// What this host made of the vector, in the vectors' terms.
fn project(observed: &Observed, exclusions: &[Vec<String>]) -> Value {
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
    json!({
        "outcome": "success",
        "code": null,
        "schema": observed.described.as_ref().map_or(Value::Null, schema_of_core),
        "payload": [payload(&predicted["body"], exclusions)],
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
        "rows": reference["result"]["rows"],
    })
}

/// Whether this host's answer is the reference's, as far as it observes.
fn agrees(got: &Value, want: &Value) -> bool {
    if want["outcome"] == "refusal" {
        // An unnamed exception has no code: no core answer is it.
        return !want["code"].is_null()
            && got["outcome"] == "refusal"
            && got["code"] == want["code"];
    }
    got == want
}

/// Where two answers differ, by part.
fn differing(got: &Value, want: &Value) -> Vec<String> {
    ["outcome", "code", "schema", "payload", "rows"]
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
    assert!(vectors.len() > 80, "{} vectors", vectors.len());
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
    }
    assert!(success > 40 && refusal > 25, "{success} {refusal}");
    assert!(marked > 0 && marked < vectors.len());
}

#[test]
fn the_pending_list_only_shrinks() {
    let pending = read_json("contract/tests/t132d-pending.json");
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
        .find(|l| l.case.id == "null.feature-all-null")
        .expect("null.feature-all-null is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let register =
        std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let (entries, _) = transitions(&register);
    let pending = ids(&read_json("contract/tests/t132d-pending.json")["pending"]["rust-host"]);
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
            Ok(observed) => project(&observed, &exclusions),
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
        "these pass now: take them out of contract/tests/t132d-pending.json's \
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
    rust_host_handles_empty_tables_as_the_reference => "empty.",
    rust_host_handles_one_row_and_the_entity_lists_as_the_reference => "edge.",
    rust_host_handles_all_null_columns_as_the_reference => "null.",
    rust_host_refuses_the_questions_as_the_reference => "refuse.",
}

#[test]
fn the_payload_digest_is_the_recorder_s() {
    // The same bytes as the recorder's `digest`, on recorded evidence: the
    // reference's own request for the case null.some-times
    // (contract/reference/<commit>.jsonl), hashed here, is what the
    // recorder wrote for the vector null.time-some, the same data, question
    // and entities.
    let env = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = env
        .lines()
        .find_map(|l| l.strip_prefix("reference: "))
        .unwrap()
        .trim()
        .to_owned();
    let text =
        std::fs::read_to_string(root().join(format!("contract/reference/{commit}.jsonl"))).unwrap();
    let observation: Value = text
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|o| o["case"] == "null.some-times")
        .unwrap();
    let all = vectors();
    let exclusions: Vec<Vec<String>> = all["payload_exclusions"]
        .as_array()
        .unwrap()
        .iter()
        .map(strings)
        .collect();
    let vector = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "null.time-some")
        .unwrap();
    let parsed = &observation["public"]["payloads"][0]["parsed"];
    assert_eq!(
        payload(parsed, &exclusions),
        json!({
            "digest": vector["reference"]["payload"][0]["digest"],
            "tables": vector["reference"]["payload"][0]["tables"],
        })
    );
    // The number forms: 1e-05, 1e-5 and 0.00001 agree; 1 and 1.0 do not,
    // nor -0.0 and 0.0.
    assert_eq!(number_form("1e-05"), number_form("1e-5"));
    assert_eq!(number_form("1e-5"), number_form("0.00001"));
    assert_ne!(number_form("1"), number_form("1.0"));
    assert_ne!(number_form("-0.0"), number_form("0.0"));
    assert_eq!(number_form("97.0"), json!(["float", false, "97", 0]));
    assert_eq!(number_form("100"), json!(["integer", false, "1", 2]));
    assert_eq!(number_form("-0.25"), json!(["float", true, "25", -2]));
    // The request id is masked to its type; a change of type is not.
    let one = json!({"metadata": {"request_id": "a"}, "x": {"number": "1"}});
    let other = json!({"metadata": {"request_id": "b"}, "x": {"number": "1"}});
    let retyped = json!({"metadata": {"request_id": 1}, "x": {"number": "1"}});
    assert_eq!(digest(&one, &exclusions), digest(&other, &exclusions));
    assert_ne!(digest(&one, &exclusions), digest(&retyped, &exclusions));
}

#[test]
fn the_comparison_fails_closed() {
    // A success: the parts this host observes each count.
    let all = vectors();
    let vector = all["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == "null.integer")
        .unwrap();
    let want = wanted(&vector["reference"]);
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
    got["payload"][0]["tables"]["context.related_tables.users"]["nulls"] = json!(0);
    assert!(!agrees(&got, &want), "a null cell fewer");
    let mut got = want.clone();
    got["schema"]["tables"]["users"]["columns"][3][1] = json!("numerical");
    assert!(!agrees(&got, &want), "another stype");
    // A refusal: the same code, and an unnamed one is never matched.
    let named = wanted(&json!({"outcome": {"result": "refusal", "code": "INVALID_REQUEST"}}));
    assert!(agrees(
        &json!({"outcome": "refusal", "code": "INVALID_REQUEST"}),
        &named
    ));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": "UNSUPPORTED_REQUEST"}),
        &named
    ));
    let unnamed = wanted(&json!({"outcome": {"result": "refusal", "code": null}}));
    assert!(!agrees(
        &json!({"outcome": "refusal", "code": null}),
        &unnamed
    ));
    // The transitions: an approved entry covers exactly the transition it
    // declares, for the vectors its glob names; a proposed one, nothing.
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
    assert_eq!(outcome_class(&unnamed), "refusal UNNAMED");
    assert_eq!(outcome_class(&refused), "refusal UNSUPPORTED_REQUEST");
    assert!(!passes("empty.orders", &refused, &unnamed, &[]));
    let declared = "refusal UNSUPPORTED_REQUEST";
    let proposed = [entry("proposed", "empty.*", "refusal UNNAMED", declared)];
    assert!(!passes("empty.orders", &refused, &unnamed, &proposed));
    let approved = [entry("approved", "empty.*", "refusal UNNAMED", declared)];
    assert!(passes("empty.orders", &refused, &unnamed, &approved));
    assert!(!passes("null.interval", &refused, &unnamed, &approved));
    let success = json!({"outcome": "success", "code": null});
    assert!(!passes("empty.orders", &success, &unnamed, &approved));
    let other = [entry(
        "approved",
        "empty.*",
        "refusal INTERNAL_ERROR",
        declared,
    )];
    assert!(!passes("empty.orders", &refused, &unnamed, &other));
    // An unmarked vector, a null key refused by name, is excused the same
    // way only.
    let key = wanted(&json!({"outcome": {"result": "refusal", "code": "INVALID_REQUEST"}}));
    assert!(!passes("null.key-one", &refused, &key, &[]));
    let keys = [entry(
        "approved",
        "null.key-*",
        "refusal INVALID_REQUEST",
        declared,
    )];
    assert!(passes("null.key-one", &refused, &key, &keys));
    let failed = json!({"outcome": "host failure"});
    assert!(!passes("null.key-one", &failed, &key, &keys));
    // The register's text: refs of proposed and approved entries, in both
    // forms; a rejected entry's and another file's ignored; a ref naming
    // vectors with no transition malformed.
    let text = "entries:\n- id: A-1\n  status: approved\n  refs: ['contract/empty-null-vectors.json#empty.* refusal UNNAMED -> refusal UNSUPPORTED_REQUEST']\n\
                - id: A-2\n  status: proposed\n  refs:\n  - contract/empty-null-vectors.json#null.key-* refusal INVALID_REQUEST -> success\n\
                - id: A-3\n  status: rejected\n  refs:\n  - contract/empty-null-vectors.json#empty.* refusal UNNAMED -> success\n\
                - id: A-4\n  status: approved\n  refs:\n  - contract/nested-type-vectors.json#map.* success -> success\n  - contract/empty-null-vectors.json#empty.*\n  - contract/empty-null-vectors.json#empty.* refusal unnamed -> success\n";
    let (read, malformed) = transitions(text);
    assert_eq!(
        read.iter()
            .map(|(e, s, g, o, n)| format!("{e} {s} {g} {o} -> {n}"))
            .collect::<Vec<_>>(),
        vec![
            "A-1 approved empty.* refusal UNNAMED -> refusal UNSUPPORTED_REQUEST",
            "A-2 proposed null.key-* refusal INVALID_REQUEST -> success",
        ]
    );
    assert_eq!(
        malformed,
        vec![
            "A-4: contract/empty-null-vectors.json#empty.*",
            "A-4: contract/empty-null-vectors.json#empty.* refusal unnamed -> success",
        ]
    );
    assert!(glob("empty.*", "empty.orders"));
    assert!(!glob("empty.*", "edge.no-entities-empty-orders"));
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
