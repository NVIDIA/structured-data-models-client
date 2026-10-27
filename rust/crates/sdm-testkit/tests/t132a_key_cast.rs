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
//! - A vector carrying `expected` (derived by the recorder from the exact
//!   value, never through a double) is held to it first: a float key's
//!   nearest float at its width, ties to even (`f32`/`f64` bits), a
//!   decimal's exact value (`dec`), or a refusal by name (INVALID_REQUEST
//!   naming `indices` and why).
//! - A vector marked `register.required` may differ from the reference
//!   only under an entry of `docs/design/rust-core/register.yaml`,
//!   proposed or approved, whose `refs` name it with the transition it
//!   makes (`contract/key-cast-vectors.json#<id or glob> <from> -> <to>`,
//!   each `success`, `refusal <CODE>` or `unfinished`, the reference not
//!   answering within the recorder's limit under the owner's approval of
//!   2026-10-16 for a reference timeout outcome); an entry covers only the
//!   transition it declares. Every other vector is held to the reference
//!   exactly.
//! - A mix of kinds the host hands over as text (`disposition: as_text`)
//!   answers as the reference answers that text given as text, where the
//!   reference does not fail with INTERNAL_ERROR on it, unless the vector
//!   is marked and an entry names it with the transition from the
//!   reference's answer to that text to this host's (since main's eb6c68eb
//!   the reference reads a decimal key's text as a double).
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

/// A ref naming vectors: (entry, glob, from, to).
type Transition = (String, String, String, String);

/// One `refs` item of the vector form: the glob, the transition's start
/// and its end; None for any other text.
fn transition(item: &str) -> Option<(String, String, String)> {
    let rest = item.strip_prefix(REF)?;
    let (glob, classes) = rest.split_once(' ')?;
    let (old, new) = classes.split_once(" -> ")?;
    let class = |c: &str| {
        c == "success"
            || c == "unfinished"
            || c.strip_prefix("refusal ").is_some_and(|code| {
                !code.is_empty()
                    && code.starts_with(|ch: char| ch.is_ascii_uppercase())
                    && code.chars().all(|ch| ch.is_ascii_uppercase() || ch == '_')
            })
    };
    (!glob.is_empty() && class(old) && class(new))
        .then(|| (glob.to_owned(), old.to_owned(), new.to_owned()))
}

/// Each vector ref of the register's proposed or approved entries, as
/// (entry, glob, from, to), read from the YAML's text entry by entry
/// (`- id:` at the margin), the `refs` field alone, in its flow
/// (`refs: [a, b]`) or block (`refs:` then `  - a`) form; and the refs
/// naming vectors that declare no transition.
fn transitions() -> (Vec<Transition>, Vec<String>) {
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    let mut out = Vec::new();
    let mut malformed = Vec::new();
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
        for item in refs {
            let item = item.trim_matches(|c| c == '\'' || c == '"');
            if !item.starts_with(REF) {
                continue;
            }
            match transition(item) {
                Some((glob, old, new)) => out.push((id.clone(), glob, old, new)),
                None => malformed.push(format!("{id}: {item}")),
            }
        }
    }
    (out, malformed)
}

/// The entry naming `id` with exactly the transition `old` to `new`.
fn covered(id: &str, old: &str, new: &str, entries: &[Transition]) -> Option<String> {
    entries
        .iter()
        .find(|(_, glob_, from, to)| from == old && to == new && glob(glob_, id))
        .map(|(entry, ..)| entry.clone())
}

/// `success`, `unfinished` or `refusal <CODE>`.
fn outcome_class(answer: &Value) -> String {
    if answer["outcome"] == "success" {
        "success".to_owned()
    } else if answer["outcome"] == "unfinished" {
        "unfinished".to_owned()
    } else {
        format!("refusal {}", answer["code"].as_str().unwrap_or("-"))
    }
}

/// A plain decimal's text normalized as the recorder's `decimal_text`.
fn decimal_text(text: &str) -> Option<String> {
    let text = text.trim();
    let (sign, body) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text.strip_prefix('+').unwrap_or(text)),
    };
    let (whole, fraction) = body.split_once('.').unwrap_or((body, ""));
    if whole.is_empty() && fraction.is_empty()
        || !whole
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let whole = whole.trim_start_matches('0');
    let whole = if whole.is_empty() { "0" } else { whole };
    let fraction = fraction.trim_end_matches('0');
    if whole == "0" && fraction.is_empty() {
        return Some("0".to_owned());
    }
    Some(if fraction.is_empty() {
        format!("{sign}{whole}")
    } else {
        format!("{sign}{whole}.{fraction}")
    })
}

/// Whether a canonical entity cell is the derived one.
fn cell_matches(wanted: &Value, cell: &Value) -> bool {
    let (Some(kind), Some(value)) = (wanted[0].as_str(), wanted[1].as_str()) else {
        return false;
    };
    let number = || -> Option<f64> {
        match cell[0].as_str()? {
            "f" => Some(f64::from_bits(
                u64::from_str_radix(cell[1].as_str()?, 16).ok()?,
            )),
            "i" => cell[1].as_str()?.parse::<i128>().ok().map(|i| i as f64),
            _ => None,
        }
    };
    match kind {
        "f32" => number().is_some_and(|x| format!("{:08x}", (x as f32).to_bits()) == value),
        "f64" => number().is_some_and(|x| format!("{:016x}", x.to_bits()) == value),
        "dec" => match cell[0].as_str() {
            Some("i") | Some("s") => {
                cell[1].as_str().and_then(decimal_text).as_deref() == Some(value)
            }
            // A payload that writes the key as a double is held to the
            // double nearest the value.
            Some("f") => value
                .parse::<f64>()
                .is_ok_and(|near| cell[1] == format!("{:016x}", near.to_bits()).as_str()),
            _ => false,
        },
        _ => wanted == cell,
    }
}

/// Whether this host's answer is the derived expectation.
fn matches(expected: &Value, got: &Value) -> bool {
    if got["outcome"] != expected["outcome"] {
        return false;
    }
    if expected["outcome"] == "refusal" {
        if got["code"] != expected["code"] {
            return false;
        }
        if expected["named"] == true {
            return got["fields"]["field"] == "indices"
                && got["fields"]["reason"]
                    .as_str()
                    .is_some_and(|r| !r.is_empty());
        }
        return true;
    }
    let (Some(want), Some(have)) = (expected["entities"].as_array(), got["entities"].as_array())
    else {
        return false;
    };
    want.len() == have.len() && want.iter().zip(have).all(|(w, h)| cell_matches(w, h))
}

/// Why this host's answer fails `vector`, or None.
fn problem(vector: &Value, got: &Value, entries: &[Transition]) -> Option<String> {
    let id = vector["id"].as_str().unwrap();
    let reference = &vector["reference"];
    let public = &reference["public"];
    if got["code"] == "UNSUPPORTED_REQUEST" || got["outcome"] == "host failure" {
        return Some(format!("{id}: got {got}"));
    }
    if let Some(expected) = vector.get("expected")
        && !matches(expected, got)
    {
        return Some(format!(
            "{id}: got {got}, the derived expectation {expected}"
        ));
    }
    if vector["register"]["disposition"] == "as_text" {
        let text = &reference["as_text"];
        if outcome_class(text) != "refusal INTERNAL_ERROR" && !agrees(got, text) {
            let (old, new) = (outcome_class(text), outcome_class(got));
            if vector["register"]["required"] != true || covered(id, &old, &new, entries).is_none()
            {
                return Some(format!(
                    "{id}: got {got}, the reference's answer to its text {text}: \
                     no register entry names it with the transition {old} -> {new}"
                ));
            }
        }
    }
    if agrees(got, public) {
        return None;
    }
    if vector["register"]["required"] != true {
        return Some(format!(
            "{id}: got {got}, the reference {public} (held exactly)"
        ));
    }
    let (old, new) = (outcome_class(public), outcome_class(got));
    if covered(id, &old, &new, entries).is_none() {
        return Some(format!(
            "{id}: got {got}, the reference {public}: no register entry names it \
             with the transition {old} -> {new}"
        ));
    }
    None
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
            "fields": {"field": fields.get("field"), "reason": fields.get("reason")},
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
    let mut unfinished = Vec::new();
    for vector in vectors {
        let id = vector["id"].as_str().unwrap();
        assert!(seen.insert(id.to_owned()), "{id} twice");
        let public = &vector["reference"]["public"];
        match (public["outcome"].as_str(), public["code"].as_str()) {
            (Some("success"), _) => success += 1,
            (Some("refusal"), Some("INVALID_REQUEST")) => invalid += 1,
            (Some("refusal"), Some("INTERNAL_ERROR")) => internal += 1,
            // The owner's approval of 2026-10-16 for a reference timeout
            // outcome: the reference did not answer within the recorder's
            // limit, and the vector is marked.
            (Some("unfinished"), None) => {
                assert!(
                    public["within_seconds"].as_u64().is_some_and(|s| s > 0),
                    "{id}"
                );
                assert_eq!(vector["reference"]["cast"], Value::Null, "{id}");
                assert_eq!(vector["register"]["required"], true, "{id}");
                unfinished.push(id.to_owned());
            }
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
    assert_eq!(unfinished, ["decimal128-18-0.exponent.huge.decimal"]);
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
    let loaded = case::load_for(&[contract.join("cases")], case::Database::Duckdb).unwrap();
    let template = loaded
        .iter()
        .find(|l| l.case.id == "key.uint8")
        .expect("key.uint8 is a case")
        .case
        .clone();
    let bases = bases::load(&contract.join("bases.yaml")).unwrap();
    let (entries, _) = transitions();
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
        let got = answer(&template, &bases, vector, keys);
        let failed = problem(vector, &got, &entries);
        match (failed, pending.contains(id)) {
            (None, true) => passing_but_pending.push(id.to_owned()),
            (Some(why), false) => failures.push(why),
            _ => {}
        }
    }
    assert!(ran > 0, "{key}: {ran} vectors");
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
    rust_host_rounds_to_a_float32_key_near_one => "float32-near-one",
    rust_host_rounds_to_a_float32_key_near_its_largest => "float32-large",
    rust_host_casts_to_a_float32_key_past_2pow24 => "float32-past-2pow24",
    rust_host_casts_to_a_float64_key_past_2pow53 => "float64-past-2pow53",
    rust_host_casts_to_a_float64_key_past_minus_2pow53 => "float64-past-minus-2pow53",
    rust_host_holds_a_38_digit_hugeint_key_exactly => "decimal128-38-0-wide",
    rust_host_holds_a_38_digit_decimal_38_10_key_exactly => "decimal128-38-10-wide",
    rust_host_holds_a_decimal_38_37_key_exactly => "decimal128-38-37",
    rust_host_reaches_a_text_key_by_the_core_s_float_text => "utf8-float-text",
    rust_host_writes_a_float_entity_of_a_text_key_as_the_reference => "utf8-float-pyarrow-text",
    rust_host_misses_numeric_ids_by_the_reference_s_float_text => "utf8-numeric-ids",
    rust_host_reaches_numeric_ids_by_the_reference_s_float_text => "utf8-numeric-ids-and-exponent",
    rust_host_misses_rust_s_float_text => "utf8-float-rust-text",
    rust_host_reads_a_float32_key_s_text_with_an_exponent => "float32-exponent",
    rust_host_reads_a_decimal_key_s_text_with_an_exponent => "decimal128-38-10-exponent",
}

#[test]
fn the_glob_the_transitions_and_the_derived_matches_hold() {
    assert!(glob("decimal128-18-0.*", "decimal128-18-0.int.present"));
    assert!(glob("*.text.hex", "int64.text.hex"));
    assert!(!glob("int64.*", "int8.int.present"));
    assert!(glob("int64.int.present", "int64.int.present"));
    assert!(!glob("int64.int.present", "int64.int.present-2"));
    let read = |t: &str| transition(&format!("{REF}{t}"));
    assert_eq!(
        read("*.time.* refusal INTERNAL_ERROR -> refusal INVALID_REQUEST"),
        Some((
            "*.time.*".to_owned(),
            "refusal INTERNAL_ERROR".to_owned(),
            "refusal INVALID_REQUEST".to_owned()
        ))
    );
    assert_eq!(read("*.time.*"), None);
    assert_eq!(read("*.time.* refusal -> success"), None);
    assert_eq!(
        read("*.exponent.huge.* unfinished -> refusal INVALID_REQUEST"),
        Some((
            "*.exponent.huge.*".to_owned(),
            "unfinished".to_owned(),
            "refusal INVALID_REQUEST".to_owned()
        ))
    );
    assert_eq!(read("*.time.* unfinished 300 -> success"), None);
    // An entry covers only the transition it declares; none, nothing.
    let entries = vec![(
        "X-1".to_owned(),
        "int64.*".to_owned(),
        "success".to_owned(),
        "refusal INVALID_REQUEST".to_owned(),
    )];
    assert_eq!(
        covered(
            "int64.text.hex",
            "success",
            "refusal INVALID_REQUEST",
            &entries
        ),
        Some("X-1".to_owned())
    );
    assert_eq!(
        covered(
            "int64.text.hex",
            "refusal INTERNAL_ERROR",
            "refusal INVALID_REQUEST",
            &entries
        ),
        None
    );
    assert_eq!(
        covered(
            "int8.text.hex",
            "success",
            "refusal INVALID_REQUEST",
            &entries
        ),
        None
    );
    assert_eq!(
        covered("int64.text.hex", "success", "refusal INVALID_REQUEST", &[]),
        None
    );
    // Derived matches: a float32 at its width, a decimal by its value, a
    // refusal by name.
    let f32_one_up = serde_json::json!({"outcome": "success", "entities": [["f32", "3f800001"]]});
    let wide = serde_json::json!({"outcome": "success", "entities": [["f", format!("{:016x}", 1.000_000_119_209_289_6_f64.to_bits())]]});
    assert!(matches(&f32_one_up, &wide));
    let one = serde_json::json!({"outcome": "success", "entities": [["f", "3ff0000000000000"]]});
    assert!(!matches(&f32_one_up, &one));
    let dec = serde_json::json!({"outcome": "success", "entities": [["dec", "12.5"]]});
    assert!(matches(
        &dec,
        &serde_json::json!({"outcome": "success", "entities": [["s", "012.500"]]})
    ));
    assert!(!matches(
        &dec,
        &serde_json::json!({"outcome": "success", "entities": [["s", "12.51"]]})
    ));
    let named = serde_json::json!({"outcome": "refusal", "code": "INVALID_REQUEST", "named": true});
    let bare = serde_json::json!({"outcome": "refusal", "code": "INVALID_REQUEST", "fields": {"field": null, "reason": null}});
    assert!(!matches(&named, &bare));
    let by_name = serde_json::json!({"outcome": "refusal", "code": "INVALID_REQUEST", "fields": {"field": "indices", "reason": "out_of_range"}});
    assert!(matches(&named, &by_name));
}

#[test]
fn an_unfinished_or_text_answer_differs_only_under_its_transition() {
    let all = vectors();
    let vectors = all["vectors"].as_array().unwrap();
    let entry = |id: &str, old: &str, new: &str| {
        (
            "SEEDED".to_owned(),
            id.to_owned(),
            old.to_owned(),
            new.to_owned(),
        )
    };
    // The vector the reference did not finish: held with no entry, and
    // excused only by one declaring unfinished -> this host's answer.
    let late = vectors
        .iter()
        .find(|v| v["reference"]["public"]["outcome"] == "unfinished")
        .unwrap();
    let id = late["id"].as_str().unwrap();
    // Refused by name, as the vector's derived expectation asks.
    let refused = serde_json::json!({
        "outcome": "refusal",
        "code": "INVALID_REQUEST",
        "fields": {"field": "indices", "reason": "out_of_range"}
    });
    assert!(problem(late, &refused, &[]).is_some());
    assert!(problem(late, &refused, &[entry(id, "unfinished", "success")]).is_some());
    assert!(
        problem(
            late,
            &refused,
            &[entry(id, "unfinished", "refusal INVALID_REQUEST")]
        )
        .is_none()
    );
    // A marked as_text vector the reference refuses, whose text it reads
    // as doubles: an exact answer passes only with both transitions named.
    let text = vectors
        .iter()
        .find(|v| {
            v["register"]["disposition"] == "as_text"
                && v["register"]["required"] == true
                && v["reference"]["as_text"]["outcome"] == "success"
                && v["reference"]["public"]["outcome"] == "refusal"
                && v["key"].as_str().unwrap().starts_with("decimal")
        })
        .unwrap();
    let id = text["id"].as_str().unwrap();
    let public = outcome_class(&text["reference"]["public"]);
    let exact = serde_json::json!({"outcome": "success", "entities": [["s", "1"]]});
    assert!(problem(text, &exact, &[]).is_some());
    let as_text = entry(id, "success", "success");
    assert!(problem(text, &exact, std::slice::from_ref(&as_text)).is_some());
    assert!(problem(text, &exact, &[as_text, entry(id, &public, "success")]).is_none());
}

#[test]
fn every_vector_ref_names_its_transition() {
    gated("every_vector_ref_names_its_transition", || {
        let (_, malformed) = transitions();
        assert!(malformed.is_empty(), "{malformed:?}");
    });
}

fn gated(test: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let pending = ids(&json("contract/tests/t132a-pending.json")["pending"]["exits"]);
    let outcome = std::panic::catch_unwind(check);
    match (outcome, pending.contains(&format!("rust:{test}"))) {
        (Ok(()), true) => panic!(
            "{test} passes now: take rust:{test} out of \
             contract/tests/t132a-pending.json's pending[\"exits\"]"
        ),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}
