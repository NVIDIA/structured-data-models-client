// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.28b's exits on `rust-host`, held from outside the implementation (the
//! package's independent test author), against what the reference gave.
//!
//! The vectors are the reference's: `t128b-predict-result.json`, recorded by
//! `t128b_predict_result_record.py` beside it, and the boundary-vector
//! matrix in `contract/response-vectors.json`'s `boundaries`, written by the
//! same recorder. Each case this host can run is one of the contract's
//! cases (`base.regression`, `base.binary`, `base.multiclass`), materialized
//! here as the case says, then predicted with the model and the outputs
//! the reference's call asked for (the outputs the caller named, else the
//! reference's call's own `output.fields`), its model call answered with the
//! exact status and bytes the reference's endpoint answered, which
//! `python-core` is held to as well, so the two hosts agree through them:
//!
//! - exit "L1 layout vectors pass, with distinct per-row answers and
//!   reordered responses" (`LAYOUT` cases): the result's `row_index`, `id`
//!   and `prediction`, row by row, are the reference's `to_pandas()`'s;
//! - exit "on both core hosts, over base.binary and base.multiclass, the
//!   boolean and text results match the reference" (`CLASSIFICATION`): the
//!   prediction a bool or text as the reference's (the probabilities'
//!   columns are compared on python-core, where `to_pandas()` names them);
//! - the review of part 2: a regression's integer predictions exact past
//!   2**53 (int64) and mixed with floats (float64), a missing or null
//!   prediction null (the reference's `.items` say `None`) and a NaN a NaN,
//!   and probabilities written as text, accepted (their values compared,
//!   each class as `str()` writes it) or refused as the reference does;
//! - the ledger rows "the call's model and output checks" and "the parts
//!   merged into one result with the notice of an absent output" (`PREPARE`
//!   and `CONTEXT` cases over a contract case): the reference's refusal code,
//!   or its result;
//! - exit "a bounded boundary-vector matrix ... run on both hosts": each
//!   answer replayed; a result where the reference gives one, else the
//!   reference's code, an endpoint's refusal carrying the endpoint's status
//!   (its provenance) and a local error carrying none. An answer the
//!   reference's transport could not decode (`TRANSPORT_ERROR` there) may be
//!   `INVALID_RESPONSE` here, this host having no transport of its own
//!   (decision E46); a vector whose `divergence` names a register entry the
//!   owner approved (P1B-22, P1B-23) may be refused `INVALID_RESPONSE` as a
//!   local error, as that entry says; one naming a proposed entry (P1B-4)
//!   is held to the reference.
//!
//! Before T1.28b the core refuses a classification's outputs (`outputs`)
//! or answer (`problem_type`), a prediction not a float (`prediction_type`)
//! and every non-finite number, and these tests say so by name.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the recorded vectors"
)]

use std::collections::BTreeMap;
use std::path::PathBuf;

use duckdb::arrow::array::{
    Array, BooleanArray, Float64Array, Int64Array, LargeStringArray, MapArray, StringArray,
};
use sdm_contract::data::AttachmentId;
use sdm_contract::ops::{Materialize, Operation, OperationResult, Predict, RequestState};
use sdm_contract::run::{ModelOutcome, ModelReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, Step};
use sdm_host::{Attachment, HostRun, Outgoing, default_root, encode, read};
use sdm_testkit::case::Case;
use sdm_testkit::{bases, case, host};
use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn load(path: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join(path)).unwrap()).unwrap()
}

fn recorded() -> Value {
    load("packages/structured-data-models-client/tests/t128b-predict-result.json")
}

fn boundaries() -> Vec<Value> {
    load("contract/response-vectors.json")["boundaries"]
        .as_array()
        .expect("the boundary vectors are recorded")
        .clone()
}

fn the_case(id: &str) -> (Case, bases::Bases) {
    let contract = root().join("contract");
    let loaded = case::load(&[contract.join("cases")]).unwrap();
    let found = loaded
        .iter()
        .find(|l| l.case.id == id)
        .unwrap_or_else(|| panic!("{id} is a case"))
        .case
        .clone();
    (found, bases::load(&contract.join("bases.yaml")).unwrap())
}

/// Whether register entry `entry` is approved by the owner, read from its
/// own block, anchored: its `- id:` line is exactly `entry`, its `status`
/// exactly `approved` and its `approved_by` exactly `owner` (`register.py`'s
/// owner approval). A proposed entry, a reviewer's approval, or an entry not
/// there approves nothing.
fn owner_approved(entry: Option<&str>) -> bool {
    let Some(entry) = entry.filter(|e| !e.is_empty()) else {
        return false;
    };
    let text = std::fs::read_to_string(root().join("docs/design/rust-core/register.yaml")).unwrap();
    for block in text.split("\n- id: ").skip(1) {
        let mut lines = block.split('\n');
        if lines.next().map(str::trim) != Some(entry) {
            continue;
        }
        let field = |name: &str| -> Option<String> {
            block
                .split('\n')
                .skip(1)
                .filter(|l| l.starts_with("  ") && !l.starts_with("   "))
                .find_map(|l| {
                    let (key, value) = l[2..].split_once(':')?;
                    (key == name).then(|| value.trim().to_owned())
                })
        };
        return field("status").as_deref() == Some("approved")
            && field("approved_by").as_deref() == Some("owner");
    }
    false
}

/// What `rust-host`'s predict ended with.
#[derive(Debug)]
enum Ended {
    /// The result's columns, by name, each its cells.
    Result(BTreeMap<String, Vec<Value>>),
    /// The core's error: its code as the contract spells it, its fields.
    Refused { code: String, fields: Value },
}

fn refused(out: &Outgoing) -> Ended {
    match read(out).expect("a readable message") {
        ControlMessage::Error { error } => Ended::Refused {
            code: serde_json::to_value(error.code)
                .unwrap()
                .as_str()
                .unwrap()
                .to_owned(),
            fields: serde_json::to_value(&error.fields).unwrap(),
        },
        other => panic!("not an error: {other:?}"),
    }
}

/// A result cell as the recorder writes one.
fn cells(column: &dyn Array) -> Vec<Value> {
    let any = column.as_any();
    (0..column.len())
        .map(|row| {
            if column.is_null(row) {
                return Value::Null;
            }
            if let Some(a) = any.downcast_ref::<Int64Array>() {
                json!({"int": a.value(row).to_string()})
            } else if let Some(a) = any.downcast_ref::<Float64Array>() {
                let v = a.value(row);
                if v.is_nan() {
                    json!({"f64": "nan"})
                } else {
                    json!({"f64_bits": format!("{:016x}", v.to_bits())})
                }
            } else if let Some(a) = any.downcast_ref::<BooleanArray>() {
                json!({"bool": a.value(row)})
            } else if let Some(a) = any.downcast_ref::<StringArray>() {
                json!({"str": a.value(row)})
            } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
                json!({"str": a.value(row)})
            } else if let Some(a) = any.downcast_ref::<MapArray>() {
                // Each entry as [its key's text, its value's cell].
                let entries = a.value(row);
                let keys = cells(entries.column(0).as_ref());
                let values = cells(entries.column(1).as_ref());
                Value::Array(
                    keys.into_iter()
                        .zip(values)
                        .map(|(k, v)| json!([k.get("str").cloned().unwrap_or(k), v]))
                        .collect(),
                )
            } else {
                json!({"other": format!("{:?}", column.data_type())})
            }
        })
        .collect()
}

/// The `predict` operation this host starts: the request state attached as
/// 1 (payload) and 2 (query), `model`, this host's request id, `outputs`
/// and no inference option asked. Built from its fields by name through the
/// contract's own reading of the operation, not a struct literal, so a field
/// the contract adds with a default (`#[serde(default)]`) leaves this test
/// compiling and running unchanged.
fn predict_operation(version: u32, model: &str, outputs: &[String]) -> Predict {
    let fields = json!({
        "state": RequestState {
            version,
            payload: AttachmentId(1),
            query: AttachmentId(2),
        },
        "model": model,
        "request_id": host::REQUEST_ID,
        "outputs": outputs,
        "inference_asked": [],
    });
    serde_json::from_value(fields).unwrap_or_else(|e| {
        panic!(
            "the contract's Predict is not built from its named fields ({e}); \
             a field it adds needs a default for this test"
        )
    })
}

/// `id`'s case materialized on this host, then predicted with `model` and
/// `outputs`, its one model call answered with `status` and `body`.
fn predict(id: &str, model: &str, outputs: &[String], status: u16, body: &str) -> Ended {
    let (case, bases) = the_case(id);
    let (observed, driven) =
        host::materialize_as(&case, &bases, &|m: Materialize| Operation::Materialize(m)).unwrap();
    let Some((handle, Ok(OperationResult::Materialize(materialized)))) = driven else {
        panic!("{id}: not materialized: {:?}", observed.answer);
    };
    let take = |at| match handle.take_attachment(at) {
        Ok(Some(Attachment::Bytes(bytes))) => bytes,
        _ => panic!("{id}: the request state is not attached"),
    };
    let state = materialized.state;
    let (payload, query) = (take(state.payload), take(state.query));
    let start = ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: host::facts().unwrap(),
        operation: Operation::Predict(predict_operation(state.version, model, outputs)),
    }));
    let attachments = BTreeMap::from([
        (AttachmentId(1), Attachment::Bytes(payload)),
        (AttachmentId(2), Attachment::Bytes(query)),
    ]);
    let run = match HostRun::start(&default_root(), &encode(&start).unwrap(), attachments) {
        Ok(run) => run,
        Err(out) => return refused(&out),
    };
    let call = match run.next() {
        Err(out) => return refused(&out),
        Ok(out) => match read(&out).unwrap() {
            ControlMessage::Step(Step::CallModel(call)) => call,
            other => panic!("{id}: predict asked for {other:?}"),
        },
    };
    let reply = encode(&ControlMessage::ModelReply(ModelReply {
        id: call.id,
        outcome: ModelOutcome::Response {
            status,
            content_type: "application/json".into(),
            body: AttachmentId(1),
        },
    }))
    .unwrap();
    let answer = BTreeMap::from([(AttachmentId(1), Attachment::Bytes(body.as_bytes().to_vec()))]);
    if let Err(out) = run.reply(&reply, answer) {
        return refused(&out);
    }
    match run.next() {
        Err(out) => refused(&out),
        Ok(out) => match read(&out).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Predict(predicted),
            }) => {
                let Ok(Some(Attachment::Arrow(table))) = run.take_attachment(predicted.result)
                else {
                    panic!("{id}: the result is not attached");
                };
                let mut columns: BTreeMap<String, Vec<Value>> = BTreeMap::new();
                for batch in &table.batches {
                    for (field, column) in table.schema.fields().iter().zip(batch.columns()) {
                        columns
                            .entry(field.name().clone())
                            .or_default()
                            .extend(cells(column.as_ref()));
                    }
                }
                Ended::Result(columns)
            }
            other => panic!("{id}: predict answered {other:?}"),
        },
    }
}

/// An id cell compared by its digits, whether the core lays it out as an
/// integer or as text.
fn digits(cell: &Value) -> Value {
    match cell.get("int").or_else(|| cell.get("str")) {
        Some(text) => text.clone(),
        None => cell.clone(),
    }
}

/// The reference's `row_index`, `id` and `prediction` columns, from the
/// recorded `to_pandas()`.
fn reference_columns(expected: &Value) -> BTreeMap<&'static str, Option<Vec<Value>>> {
    let names: Vec<&str> = expected["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    let mut out = BTreeMap::new();
    for name in ["row_index", "id", "prediction"] {
        // `to_pandas()` leaves out a prediction column no answer filled.
        let column = names.iter().position(|n| *n == name).map(|at| {
            expected["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row[at].clone())
                .collect()
        });
        out.insert(name, column);
    }
    out
}

fn assert_result(name: &str, got: Ended, expected: &Value) {
    let Ended::Result(columns) = got else {
        panic!("{name}: rust-host refused it ({got:?}) where the reference gave a result");
    };
    let want = reference_columns(expected);
    for column in ["row_index", "id"] {
        let cells = want[column]
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: the reference has no {column}"));
        let have = columns
            .get(column)
            .unwrap_or_else(|| panic!("{name}: no {column} column in {:?}", columns.keys()));
        if column == "id" {
            let have: Vec<Value> = have.iter().map(digits).collect();
            let cells: Vec<Value> = cells.iter().map(digits).collect();
            assert_eq!(have, cells, "{name}: {column}");
        } else {
            assert_eq!(have, cells, "{name}: {column}");
        }
    }
    let items = expected.get("items").and_then(Value::as_array);
    let have = columns.get("prediction");
    match (&want["prediction"], have) {
        (Some(cells), Some(have)) => {
            // The frame's cells (an int64 column exact, a mixed one as
            // float64), except that an item the reference left None is
            // null here, never NaN.
            let wanted: Vec<Value> = cells
                .iter()
                .enumerate()
                .map(|(i, cell)| match items {
                    Some(items) if items[i]["prediction"].is_null() => Value::Null,
                    _ => cell.clone(),
                })
                .collect();
            assert_eq!(have, &wanted, "{name}: prediction");
        }
        (Some(_), None) => panic!("{name}: no prediction column in {:?}", columns.keys()),
        // No answer filled one: left out, or every cell null.
        (None, Some(have)) => assert!(
            have.iter().all(Value::is_null),
            "{name}: prediction {have:?} where the reference has none"
        ),
        (None, None) => {}
    }
    let Some(items) = items else {
        return;
    };
    if items.iter().all(|i| i["probabilities"].is_null()) {
        return;
    }
    // Each item's probabilities: each class as `str()` writes it, and its
    // probability, in the order written.
    let have = columns
        .get("probabilities")
        .unwrap_or_else(|| panic!("{name}: no probabilities column in {:?}", columns.keys()));
    for (i, item) in items.iter().enumerate() {
        let wanted: Value = match item["probabilities"].as_array() {
            None => Value::Null,
            Some(pairs) => Value::Array(
                pairs
                    .iter()
                    .map(|p| json!([p[0].clone(), p[2].clone()]))
                    .collect(),
            ),
        };
        assert_eq!(have[i], wanted, "{name}: row {i}'s probabilities");
    }
}

/// The outputs a case's call asks for: the caller's, else the ones the
/// reference's own call asked for (its defaults for the task's kind).
fn outputs_of(spec: &Value) -> Option<Vec<String>> {
    if let Some(named) = spec["predict"].get("outputs") {
        return named
            .as_array()?
            .iter()
            .map(|o| o.as_str().map(str::to_owned))
            .collect();
    }
    let fields = spec["expected"]["calls"].get(0)?["output"]["fields"].as_array()?;
    Some(
        fields
            .iter()
            .map(|f| f.as_str().unwrap().to_owned())
            .collect(),
    )
}

/// The status and body the reference's endpoint answered the case's call
/// with, or a scripted answer that is never read when no call was made.
fn answer_of(spec: &Value) -> (u16, String) {
    match spec["expected"]["calls"].get(0) {
        Some(call) => (
            u16::try_from(call["status"].as_u64().unwrap()).unwrap(),
            call["answer"].as_str().unwrap().to_owned(),
        ),
        None => (500, "{}".to_owned()),
    }
}

fn run_recorded(covers: &[&str]) -> usize {
    let all = recorded();
    let mut ran = 0;
    let mut failures = Vec::new();
    for spec in all["cases"].as_array().unwrap() {
        let Some(id) = spec["rust_case"].as_str() else {
            continue;
        };
        if !covers
            .iter()
            .any(|c| spec["covers"].as_str().unwrap().starts_with(c))
        {
            continue;
        }
        let name = spec["name"].as_str().unwrap();
        let expected = &spec["expected"];
        let Some(outputs) = outputs_of(spec) else {
            // Outputs that are not a list of names: a Python type check.
            continue;
        };
        if spec["predict"].get("request").is_some() {
            continue;
        }
        let model = spec["predict"]["model"]
            .as_str()
            .unwrap_or("kumo-relational");
        let (status, body) = answer_of(spec);
        ran += 1;
        let got = predict(id, model, &outputs, status, &body);
        let outcome = std::panic::catch_unwind(|| {
            if expected["outcome"] == "result" {
                assert_result(name, got, expected);
            } else {
                match got {
                    Ended::Refused { code, fields } => assert_eq!(
                        Some(code.as_str()),
                        expected["code"].as_str(),
                        "{name}: {fields}"
                    ),
                    Ended::Result(_) => panic!("{name}: a result where the reference refused"),
                }
            }
        });
        if let Err(panic) = outcome {
            let text = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            failures.push(text);
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {ran}:\n{}",
        failures.len(),
        failures.join("\n")
    );
    ran
}

#[test]
fn the_recording_names_this_host_s_cases() {
    let all = recorded();
    assert_eq!(all["recorded_by"], "t128b_predict_result_record.py");
    let cases = all["cases"].as_array().unwrap();
    for id in ["base.regression", "base.binary", "base.multiclass"] {
        assert!(cases.iter().any(|c| c["rust_case"] == id), "{id}");
    }
    let vectors = boundaries();
    assert!(vectors.len() >= 40, "{}", vectors.len());
    assert!(vectors.iter().all(|v| v["answers"] == "base.regression"));
}

#[test]
fn distinct_and_reordered_answers_are_laid_out_as_the_reference_s() {
    // Exit: L1 layout vectors, distinct per-row answers, reordered.
    assert!(run_recorded(&["L1 layout"]) >= 4);
}

#[test]
fn base_binary_and_base_multiclass_results_are_the_reference_s() {
    // Exit: the boolean and text results on both core hosts.
    assert!(run_recorded(&["classification results"]) >= 6);
}

#[test]
fn the_model_and_output_checks_and_the_absent_output_are_the_reference_s() {
    // Ledger rows: SDMClient._prepare's model and output checks, the
    // adapter's _preflight, and the parts' result with an absent output.
    assert!(run_recorded(&["the adapter's _preflight", "empty and constant"]) >= 8);
}

#[test]
fn integer_missing_null_and_nan_predictions_and_numeric_text_are_the_reference_s() {
    // The review of part 2: integers exact past 2**53 and mixed with
    // floats; a missing or null prediction null, never NaN, and a NaN a
    // NaN; probabilities written as text accepted and refused as the
    // reference does, their values as it read them.
    assert!(
        run_recorded(&[
            "integer predictions",
            "a missing, a null",
            "probabilities written"
        ]) >= 30
    );
}

#[test]
fn each_boundary_answer_ends_as_on_the_reference() {
    // Exit: the boundary-vector matrix, run on rust-host.
    let outputs = outputs_of(
        recorded()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "layout.regression.given")
            .unwrap(),
    )
    .unwrap();
    let mut failures = Vec::new();
    for vector in boundaries() {
        let name = vector["name"].as_str().unwrap().to_owned();
        let reference = vector["reference"].clone();
        let status = u16::try_from(vector["status"].as_u64().unwrap()).unwrap();
        let got = predict(
            "base.regression",
            "kumo-relational",
            &outputs,
            status,
            vector["body"].as_str().unwrap(),
        );
        let outcome = std::panic::catch_unwind(|| match got {
            Ended::Refused { code, fields }
                if owner_approved(vector["divergence"].as_str()) && code == "INVALID_RESPONSE" =>
            {
                // A divergence the owner approved (the vector names the
                // register entry): refused as a local error, by name.
                assert!(
                    fields.get("status").is_none(),
                    "{name}: given a status: {fields}"
                );
            }
            Ended::Result(_) if reference["outcome"] == "result" => {
                assert_result(&name, got, &reference);
            }
            Ended::Refused { code, fields } if reference["outcome"] == "refused" => {
                let carries_status = fields.get("status").is_some();
                if reference.get("endpoint").is_some() {
                    // The endpoint's own refusal: its provenance carried.
                    assert!(
                        carries_status,
                        "{name}: the endpoint's status is not carried: {fields}"
                    );
                    assert!(
                        fields["status"].to_string().contains(&status.to_string()),
                        "{name}: status {} where the endpoint answered {status}",
                        fields["status"]
                    );
                } else {
                    assert!(
                        !carries_status,
                        "{name}: a local error given a status: {fields}"
                    );
                    let want = reference["code"].as_str().unwrap();
                    let taken =
                        code == want || (want == "TRANSPORT_ERROR" && code == "INVALID_RESPONSE");
                    assert!(
                        taken,
                        "{name}: {code} where the reference gave {want}: {fields}"
                    );
                }
            }
            other => panic!(
                "{name}: {other:?}, where the reference ended {} {}",
                reference["outcome"], reference["code"]
            ),
        });
        if let Err(panic) = outcome {
            let text = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            failures.push(text);
        }
    }
    assert!(
        failures.is_empty(),
        "{}:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
