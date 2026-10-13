// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.32a's cast of the entities to the key's type on `rust-host`, held
//! from outside the implementation (the package's independent test
//! author) to what the reference gave: `contract/key-cast-vectors.json`,
//! recorded from the reference by `contract/tools/key_cast_record.py`,
//! which `contract/tests/test_t132a_key_cast.py` holds `python-core` to as
//! well, so the two hosts agree through it.
//!
//! - Exit "the entities' cast runs in the core, both hosts giving the same
//!   answers and error codes over the rule table's vectors": each vector's
//!   key type (`shop` with its `user_id` changed, the vector file's
//!   `keys`) is described and materialized by this host with the vector's
//!   entities as the host hands them over (`host_arrow`: one Arrow column
//!   by the host's natural inference, or text when that fails, built from
//!   the contract's scalars, an all-null column as Arrow's null type). The
//!   core's answer must be the reference's public answer: a request whose
//!   predicted rows carry the reference's entities in order, or a refusal
//!   with its code (this host observes no stage or exception class).
//! - A vector the recorder marked `register.required` (the reference fails
//!   with INTERNAL_ERROR, or its pandas cast changes an entity's kind or
//!   value, or the core cannot tell it from another vector the reference
//!   answers otherwise) may differ only where an entry of
//!   `docs/design/rust-core/register.yaml`, proposed or approved, names it
//!   in its `refs` (`contract/key-cast-vectors.json#<id or glob>`); every
//!   other vector is held to the reference exactly.
//! - A refusal by a rule the core does not take yet (`UNSUPPORTED_REQUEST`)
//!   never passes: the reference gives none on these vectors.
//!
//! Gating: `contract/tests/t132a-pending.json`'s `pending["rust-host"]`
//! lists the vectors that failed here when the test was written. Each must
//! still fail (strict: one that passes must be taken out of the list), and
//! the list may only shrink: it stays within `authored["rust-host"]`. The
//! implementer removes ids as they pass and changes nothing else of this
//! file, the vectors or the recorder.

use std::collections::BTreeSet;
use std::path::PathBuf;

use sdm_testkit::bases;
use sdm_testkit::case::{self, Case, Until};
use sdm_testkit::host::{self, Answer};
use serde_json::Value;

const REF: &str = "contract/key-cast-vectors.json#";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn json(path: &str) -> Value {
    let text = std::fs::read_to_string(root().join(path)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn vectors() -> Value {
    let all = json("contract/key-cast-vectors.json");
    assert_eq!(all["format"], "sdm-key-cast-vectors/1");
    assert_eq!(all["recorded_by"], "contract/tools/key_cast_record.py");
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

/// The register's entries that name a vector in their `refs`, proposed or
/// approved: (id, the patterns). Read from the YAML's text, entry by entry
/// (`- id:` at the margin), the `refs` field alone, in its flow
/// (`refs: [a, b]`) or block (`refs:` then `  - a`) form.
fn register_patterns() -> Vec<(String, Vec<String>)> {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let mut out = Vec::new();
    for block in text.split("\n- id: ").skip(1) {
        let id = block.lines().next().unwrap_or_default().trim().to_owned();
        let status = block
            .lines()
            .find_map(|l| l.strip_prefix("  status: "))
            .unwrap_or_default()
            .trim();
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
        let patterns: Vec<String> = refs
            .iter()
            .map(|r| r.trim_matches(|c| c == '\'' || c == '"'))
            .filter_map(|r| r.strip_prefix(REF))
            .map(str::to_owned)
            .collect();
        if !patterns.is_empty() {
            out.push((id, patterns));
        }
    }
    out
}

fn registered(id: &str, entries: &[(String, Vec<String>)]) -> Option<String> {
    entries
        .iter()
        .find(|(_, patterns)| patterns.iter().any(|p| glob(p, id)))
        .map(|(entry, _)| entry.clone())
}

/// A number the payload carries (`{"number": text}`) as the recorder's
/// canonical cell.
fn canonical(cell: &Value) -> Value {
    match cell {
        Value::Null => Value::Null,
        Value::Bool(b) => serde_json::json!(["b", b]),
        Value::String(s) => serde_json::json!(["s", s]),
        Value::Object(members) => match members.get("number") {
            Some(Value::String(text)) if members.len() == 1 => {
                let digits = text.strip_prefix('-').unwrap_or(text);
                if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) {
                    serde_json::json!(["i", text])
                } else {
                    let value: f64 = text.parse().expect("a number's text");
                    serde_json::json!(["f", format!("{:016x}", value.to_bits())])
                }
            }
            _ => serde_json::json!(["?", "object"]),
        },
        other => serde_json::json!(["?", other.to_string()]),
    }
}

/// The case this host runs for `vector`: `key.uint8`'s, its setup the key
/// type's, its entities the vector's column, stopping after materialize.
fn the_case(template: &Case, vector: &Value, keys: &Value) -> Case {
    let key = keys
        .as_array()
        .unwrap()
        .iter()
        .find(|k| k["id"] == vector["key"])
        .expect("the vector's key type is recorded");
    let mut case = template.clone();
    case.id = format!("t132a.{}", vector["id"].as_str().unwrap());
    case.input.setup = Some(
        key["setup"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_owned())
            .collect(),
    );
    case.until = Until::Materialize;
    let core = case.core.as_mut().expect("key.uint8 names its core inputs");
    core.predict = None;
    core.materialize
        .as_mut()
        .expect("key.uint8 materializes")
        .indices = vector["host_arrow"]["values"].as_array().unwrap().clone();
    case
}

/// This host's answer to `vector`, in the recorder's public form (no stage,
/// no class), or why it gave none.
fn answer(template: &Case, bases: &bases::Bases, vector: &Value, keys: &Value) -> Value {
    let case = the_case(template, vector, keys);
    let observed = match host::run(&case, bases) {
        Ok(observed) => observed,
        Err(failure) => {
            return serde_json::json!({"outcome": "host failure", "why": failure.to_string()});
        }
    };
    match observed.answer {
        Answer::Materialized(state) => {
            let table = &state["payload"]["predict"]["instance_table"];
            let columns = table["columns"].as_array().cloned().unwrap_or_default();
            let Some(at) = columns.iter().position(|c| {
                c.as_str()
                    .unwrap_or_default()
                    .starts_with("__kumo_entity_ref")
            }) else {
                return serde_json::json!({"outcome": "success", "entities": null});
            };
            let entities: Vec<Value> = table["rows"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|row| canonical(&row[at]))
                .collect();
            serde_json::json!({"outcome": "success", "entities": entities})
        }
        Answer::Error { code, fields, .. } => serde_json::json!({
            "outcome": "refusal",
            "code": code,
            "rule": fields.get("rule"),
        }),
        other => serde_json::json!({"outcome": format!("{other:?}")}),
    }
}

/// Whether this host's answer is the reference's, as far as it observes.
fn agrees(got: &Value, want: &Value) -> bool {
    match want["outcome"].as_str() {
        Some("success") => got["outcome"] == "success" && got["entities"] == want["entities"],
        Some("refusal") => got["outcome"] == "refusal" && got["code"] == want["code"],
        _ => false,
    }
}

#[test]
fn the_vectors_are_the_reference_s_and_not_vacuous() {
    let all = vectors();
    let vectors = all["vectors"].as_array().unwrap();
    assert!(vectors.len() > 800, "{} vectors", vectors.len());
    let reference = std::fs::read_to_string(root().join("contract/reference-env.txt")).unwrap();
    let commit = all["reference"].as_str().unwrap();
    assert!(
        reference.contains(&format!("reference: {commit}")),
        "recorded at the reference of contract/reference-env.txt"
    );
    let mut seen = BTreeSet::new();
    let (mut success, mut invalid, mut internal, mut marked) = (0, 0, 0, 0);
    for vector in vectors {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        let public = &vector["reference"]["public"];
        match (public["outcome"].as_str(), public["code"].as_str()) {
            (Some("success"), _) => success += 1,
            (Some("refusal"), Some("INVALID_REQUEST")) => invalid += 1,
            (Some("refusal"), Some("INTERNAL_ERROR")) => internal += 1,
            other => panic!("{id}: an answer the vectors do not hold: {other:?}"),
        }
        let required = vector["register"]["required"].as_bool().unwrap();
        let why = vector["register"]["why"].as_array().unwrap();
        assert_eq!(required, !why.is_empty(), "{id}: a mark says why");
        marked += usize::from(required);
        assert!(
            !vector["host_arrow"]["values"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
    assert!(success > 100 && invalid > 100 && internal > 100);
    assert!(marked > 0 && marked < vectors.len());
}

#[test]
fn the_pending_list_only_shrinks() {
    let pending = json("contract/tests/t132a-pending.json");
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

/// Runs every vector of key type `key` on this host, holding each to the
/// reference and to the pending list.
fn key_type(key: &str) {
    let all = vectors();
    let keys = &all["keys"];
    assert!(
        keys.as_array().unwrap().iter().any(|k| k["id"] == key),
        "{key} is a recorded key type"
    );
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "key.uint8")
        .expect("key.uint8 is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let entries = register_patterns();
    let pending = ids(&json("contract/tests/t132a-pending.json")["pending"]["rust-host"]);
    let mut failures = Vec::new();
    let mut passing_but_pending = Vec::new();
    let mut ran = 0;
    for vector in all["vectors"].as_array().unwrap() {
        if vector["key"] != key {
            continue;
        }
        ran += 1;
        let id = vector["id"].as_str().unwrap();
        let want = &vector["reference"]["public"];
        let got = answer(&template, &bases, vector, keys);
        let unported = got["code"] == "UNSUPPORTED_REQUEST";
        let ok = !unported
            && (agrees(&got, want)
                || (vector["register"]["required"] == true
                    && got["outcome"] != "host failure"
                    && registered(id, &entries).is_some()));
        match (ok, pending.contains(id)) {
            (true, true) => passing_but_pending.push(id.to_owned()),
            (false, false) => failures.push(format!("{id}: got {got}, the reference {want}")),
            _ => {}
        }
    }
    assert!(ran > 40, "{key}: {ran} vectors");
    assert!(
        passing_but_pending.is_empty(),
        "these pass now: take them out of contract/tests/t132a-pending.json's \
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

/// One test per key type, so the key types run in parallel.
macro_rules! key_types {
    ($($name:ident => $key:literal,)*) => {
        $(
            #[test]
            fn $name() {
                key_type($key);
            }
        )*

        #[test]
        fn every_recorded_key_type_has_its_test() {
            let tested: BTreeSet<&str> = [$($key),*].into_iter().collect();
            let recorded: BTreeSet<String> = vectors()["keys"]
                .as_array()
                .unwrap()
                .iter()
                .map(|k| k["id"].as_str().unwrap().to_owned())
                .collect();
            assert_eq!(
                recorded,
                tested.into_iter().map(str::to_owned).collect::<BTreeSet<_>>()
            );
        }
    };
}

key_types! {
    rust_host_casts_to_an_int8_key_as_the_reference => "int8",
    rust_host_casts_to_an_int16_key_as_the_reference => "int16",
    rust_host_casts_to_an_int32_key_as_the_reference => "int32",
    rust_host_casts_to_an_int64_key_as_the_reference => "int64",
    rust_host_casts_to_a_uint8_key_as_the_reference => "uint8",
    rust_host_casts_to_a_uint16_key_as_the_reference => "uint16",
    rust_host_casts_to_a_uint32_key_as_the_reference => "uint32",
    rust_host_casts_to_a_uint64_key_as_the_reference => "uint64",
    rust_host_casts_to_a_uint64_key_past_int64_as_the_reference => "uint64-past-int64",
    rust_host_casts_to_a_float32_key_as_the_reference => "float32",
    rust_host_casts_to_a_float64_key_as_the_reference => "float64",
    rust_host_casts_to_a_utf8_key_as_the_reference => "utf8",
    rust_host_casts_to_a_large_utf8_key_as_the_reference => "large-utf8",
    rust_host_casts_to_a_decimal_18_0_key_as_the_reference => "decimal128-18-0",
    rust_host_casts_to_a_decimal_38_10_key_as_the_reference => "decimal128-38-10",
    rust_host_casts_to_a_hugeint_key_as_the_reference => "decimal128-38-0",
}

#[test]
fn the_glob_and_the_register_reading_hold() {
    assert!(glob("decimal128-18-0.*", "decimal128-18-0.int.present"));
    assert!(glob("*.text.hex", "int64.text.hex"));
    assert!(!glob("int64.*", "int8.int.present"));
    assert!(glob("int64.int.present", "int64.int.present"));
    assert!(!glob("int64.int.present", "int64.int.present-2"));
    // Fail closed: a vector no entry names is not registered.
    assert_eq!(registered("int64.int.present", &[]), None);
    let entries = vec![("X-1".to_owned(), vec!["int64.*".to_owned()])];
    assert_eq!(
        registered("int64.text.hex", &entries),
        Some("X-1".to_owned())
    );
    assert_eq!(registered("int8.text.hex", &entries), None);
}
