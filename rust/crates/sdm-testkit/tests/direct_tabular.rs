// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The connected ordinary direct journey through the same public HostRun API
//! used by the Python binding, without Python or a source provider.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::ops::OperationResult;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, Step};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, encode, read};
use serde_json::{Value, json};

fn bytes(value: &Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}
fn start(operation: Value, attachments: Attachments) -> HostRun {
    let message = json!({
        "message": "start", "contract": CONTRACT_VERSION,
        "facts": {
            "now": {"t": "ts", "v": "0", "unit": "us", "zone": "UTC"},
            "local_zone": null, "zone_roots": [],
            "platform": {"os": "linux", "strftime_pads_year": false},
            "date_literal_range": null,
        },
        "operation": operation,
    });
    HostRun::start(
        &sdm_host::default_root(),
        &serde_json::to_vec(&message).unwrap(),
        attachments,
    )
    .unwrap()
}
fn done(run: &HostRun) -> OperationResult {
    let outgoing = run.next().unwrap();
    match read(&outgoing).unwrap() {
        ControlMessage::Step(Step::Done { result }) => result,
        message => panic!("expected completed direct data, got {message:?}"),
    }
}
fn take(run: &HostRun, id: AttachmentId) -> Vec<u8> {
    match run.take_attachment(id).unwrap().unwrap() {
        Attachment::Bytes(bytes) => bytes,
        _ => panic!("expected typed JSON bytes"),
    }
}
fn value(op: &str, input: Value) -> Value {
    let run = start(
        json!({"op": op, "input": 1}),
        BTreeMap::from([(AttachmentId(1), bytes(&input))]),
    );
    let OperationResult::DirectValue(result) = done(&run) else {
        panic!("not a direct value")
    };
    serde_json::from_slice(&take(&run, result.value)).unwrap()
}
fn table(columns: Vec<(&str, Vec<i64>)>) -> Attachment {
    let batch = RecordBatch::try_from_iter(
        columns
            .into_iter()
            .map(|(name, values)| (name, Arc::new(Int64Array::from(values)) as ArrayRef)),
    )
    .unwrap();
    Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
}
fn shape() -> Value {
    json!({
        "context_kind": {"kind": "frame"}, "predict_kind": {"kind": "frame"},
        "related": [], "task": "classification", "target": "target", "entity_table": "instance",
        "context": {"rows": 4, "columns": ["row_id", "feature", "target"]},
        "predict": {"rows": 2, "columns": ["row_id", "feature"]},
    })
}
fn context() -> Attachment {
    table(vec![
        ("row_id", vec![0, 1, 2, 3]),
        ("feature", vec![10, 11, 12, 13]),
        ("target", vec![0, 1, 0, 1]),
    ])
}
fn scoring() -> Attachment {
    table(vec![("row_id", vec![90, 91]), ("feature", vec![20, 21])])
}

#[test]
fn constructor_model_bytes_response_and_restored_keys_use_the_real_host() {
    let shape = value("check_direct_shape", shape());
    let prepared = start(
        json!({"op": "prepare_direct", "input": 1, "context": 2, "predict": 3}),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({"shape": shape, "schema": null, "flags": []})),
            ),
            (AttachmentId(2), context()),
            (AttachmentId(3), scoring()),
        ]),
    );
    let OperationResult::DirectValue(result) = done(&prepared) else {
        panic!("no constructor data")
    };
    let description: Value = serde_json::from_slice(&take(&prepared, result.value)).unwrap();
    assert_eq!(description["task"], "classification");
    assert_eq!(
        description["schema"]["tables"][0]["primary_key"],
        json!(["row_id"])
    );
    let bound = value(
        "bind_direct",
        json!({
            "model": "kumo-tabular", "outputs": null, "kind": "classification",
            "anchored": false, "relationships": false, "tables": 1, "context_rows": 4,
        }),
    );
    let resolving = start(
        json!({"op": "resolve_tabular", "input": 1, "context": 2, "predict": 3}),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({
                    "bound": bound, "schema": description["schema"], "flags": [],
                    "entity_table": "instance", "target": "target", "kind": "classification",
                    "context": shape["context"], "predict": shape["predict"],
                })),
            ),
            (AttachmentId(2), context()),
            (AttachmentId(3), scoring()),
        ]),
    );
    let OperationResult::DirectValue(result) = done(&resolving) else {
        panic!("no resolved columns")
    };
    let resolved: Value = serde_json::from_slice(&take(&resolving, result.value)).unwrap();
    let encoded = start(
        json!({"op": "encode_tabular", "input": 1, "context": 2, "predict": 3}),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({
                    "resolved": resolved, "primary_key": ["row_id"], "kind": "classification",
                    "context": shape["context"], "predict": shape["predict"],
                })),
            ),
            (AttachmentId(2), context()),
            (AttachmentId(3), scoring()),
        ]),
    );
    let OperationResult::DirectValue(result) = done(&encoded) else {
        panic!("no wire data")
    };
    let mut body: Value = serde_json::from_slice(&take(&encoded, result.value)).unwrap();
    let payload = take(
        &encoded,
        AttachmentId(body["payload"].as_u64().unwrap() as u32),
    );
    body["payload"] = json!(2);
    let predicting = start(
        json!({
            "op": "predict", "direct": {"input": 1, "payload": 2},
            "model": "kumo-tabular", "outputs": bound["outputs"], "default_outputs": true,
            "request_id": "call-1",
        }),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({"kind": "encoded", "body": body, "inference": {"entries": []}})),
            ),
            (AttachmentId(2), Attachment::Bytes(payload)),
        ]),
    );
    let outgoing = predicting.next().unwrap();
    let ControlMessage::Step(Step::CallModel(call)) = read(&outgoing).unwrap() else {
        panic!("no model call")
    };
    let payload: Value = serde_json::from_slice(&take(&predicting, call.body)).unwrap();
    assert_eq!(
        payload["task"]["target"]["dtype"],
        payload["schema"]["instance_table"]["columns"]["target"]["dtype"]
    );
    assert_eq!(
        payload["predict"]["instance_table"]["rows"],
        json!([[90, 20], [91, 21]])
    );
    let response = json!({"model": "kumo-tabular", "id": "answer", "metadata": {}, "predictions": [
        {"row_index": 1, "prediction": 1}, {"row_index": 0, "prediction": 0},
    ]});
    let reply: ControlMessage = serde_json::from_value(json!({
        "message": "model_reply", "id": call.id.0,
        "outcome": {"outcome": "response", "status": 200, "content_type": "application/json", "body": 1},
    })).unwrap();
    predicting
        .reply(
            &encode(&reply).unwrap(),
            BTreeMap::from([(AttachmentId(1), bytes(&response))]),
        )
        .unwrap();
    let OperationResult::DecodedDirectResponse(result) = done(&predicting) else {
        panic!("no decoded data")
    };
    let mut data: Value = serde_json::from_slice(&take(&predicting, result.value)).unwrap();
    let buffers: Vec<_> = ["payload", "response", "rows"]
        .iter()
        .map(|name| {
            take(
                &predicting,
                AttachmentId(data[*name].as_u64().unwrap() as u32),
            )
        })
        .collect();
    let decoded_rows: Value = serde_json::from_slice(&buffers[2]).unwrap();
    let checked = value(
        "check_direct_rows",
        json!({"scope": data["scope"], "rows": decoded_rows}),
    );
    assert!(checked["row"].is_null());
    assert_eq!(checked["rows"][0]["id"], json!({"t": "i64", "v": "91"}));
    data["payload"] = json!(2);
    data["response"] = json!(3);
    let laid = start(
        json!({"op": "layout_direct", "input": 1, "payload": 2, "response": 3}),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({"data": data, "attributed": checked["rows"]})),
            ),
            (AttachmentId(2), Attachment::Bytes(buffers[0].clone())),
            (AttachmentId(3), Attachment::Bytes(buffers[1].clone())),
        ]),
    );
    let OperationResult::Predict(result) = done(&laid) else {
        panic!("no final result")
    };
    assert_eq!(result.model, "kumo-tabular");
    assert_eq!(result.request_id, "answer");
    let Some(Attachment::Arrow(rows)) = laid.take_attachment(result.result).unwrap() else {
        panic!("no result rows")
    };
    let ids = rows.batches[0]
        .column_by_name("id")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values().as_ref(), &[90, 91]);
}

#[test]
fn direct_headers_cannot_lie_about_the_admitted_rows() {
    let run = start(
        json!({"op": "prepare_direct", "input": 1, "context": 2, "predict": 3}),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({"shape": shape(), "schema": null, "flags": []})),
            ),
            (AttachmentId(2), context()),
            (
                AttachmentId(3),
                table(vec![("row_id", vec![90]), ("feature", vec![20])]),
            ),
        ]),
    );
    let outgoing = run.next().unwrap_err();
    let ControlMessage::Error { error } = read(&outgoing).unwrap() else {
        panic!("no protocol error")
    };
    assert_eq!(error.code, sdm_contract::ErrorCode::InvalidMessage);
}

#[test]
fn malformed_bound_model_policy_is_recomputed_before_encoding() {
    let mut bound = value(
        "bind_direct",
        json!({
            "model": "kumo-tabular", "outputs": null, "kind": "classification",
            "anchored": false, "relationships": false, "tables": 1, "context_rows": 4,
        }),
    );
    bound["outputs"] = json!(["embeddings"]);
    let run = start(
        json!({"op": "resolve_tabular", "input": 1, "context": 2, "predict": 3}),
        BTreeMap::from([
            (
                AttachmentId(1),
                bytes(&json!({
                    "bound": bound, "schema": {"tables": [], "links": []}, "flags": [],
                    "entity_table": "instance", "target": "target", "kind": "classification",
                    "context": shape()["context"], "predict": shape()["predict"],
                })),
            ),
            (AttachmentId(2), context()),
            (AttachmentId(3), scoring()),
        ]),
    );
    let outgoing = run.next().unwrap_err();
    let ControlMessage::Error { error } = read(&outgoing).unwrap() else {
        panic!("no protocol error")
    };
    assert_eq!(error.code, sdm_contract::ErrorCode::InvalidMessage);
}

#[test]
fn native_literal_data_is_interpreted_and_attributed_in_rust() {
    let scope = json!({"version": 2, "model": "kumo-tabular", "request_id": "answer", "call_request_id": "call-1", "response_rows": 1, "scoring_rows": 1});
    let decoded = json!({"ordinal": 0, "row_index": "0", "id": "[+90]", "key": {"t": "i64", "v": "90"}, "raw_in_batch": false, "prior_ordinal": null});
    let checked = value(
        "check_direct_rows",
        json!({"scope": scope, "rows": [decoded]}),
    );
    let json_fact = json!({"scope": scope, "ordinal": 0, "text": "[+90]", "value": {"outcome": "failed", "failure": {"ancestry": ["value_error"]}}});
    let interpreted = value(
        "interpret_direct_json",
        json!({"checked": checked["row"], "json": json_fact}),
    );
    assert_eq!(
        interpreted["checked"]["id"]["representation"],
        "native_literal_text"
    );
    let literal = json!({"scope": scope, "ordinal": 0, "text": "[+90]", "value": {"outcome": "value", "value": {"kind": "list"}, "members": [{"kind": "integer", "digits": "0x5a", "radix": "hexadecimal"}]}});
    let selected = value(
        "select_direct_member",
        json!({"interpreted": interpreted, "literal": literal}),
    );
    let attributed = value(
        "attribute_direct_row",
        json!({"selected": selected, "string": {"scope": scope, "ordinal": 0, "value": {"outcome": "value", "points": [57, 48]}}}),
    );
    assert_eq!(attributed["id"], json!({"t": "i64", "v": "90"}));
}
