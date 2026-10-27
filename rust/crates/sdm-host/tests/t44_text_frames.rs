// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain Arrow twins of the frozen public text-feature journeys.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Materialized, Operation, OperationResult};
use sdm_contract::run::{ModelOutcome, ModelReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing};
use serde_json::{Value, json};

fn oracle(target: &str, variant: &str) -> Value {
    let all: Value = serde_json::from_str(include_str!(
        "../../../../packages/structured-data-models-client/tests/t44_text_reference.json"
    ))
    .unwrap();
    all["records"][format!("python-{target}-{variant}")].clone()
}

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn arrow(columns: Vec<(&str, ArrayRef)>, chunks: bool) -> Attachment {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let batches = if chunks {
        vec![batch.slice(0, 3), batch.slice(3, batch.num_rows() - 3)]
    } else {
        vec![batch.clone()]
    };
    Attachment::Arrow(ArrowData::new(batch.schema(), batches))
}

fn ints(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))
}

fn text(values: Vec<Option<&str>>, large: bool) -> ArrayRef {
    if large {
        Arc::new(LargeStringArray::from(values))
    } else {
        Arc::new(StringArray::from(values))
    }
}

fn column(name: &str, dtype: &str, stype: &str) -> Value {
    json!({"name":name,"dtype":dtype,"stype":stype,"stype_before_key":null})
}

fn message(operation: Value) -> Vec<u8> {
    let operation: Operation = serde_json::from_value(operation).unwrap();
    sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                value: 0,
                unit: TimestampUnit::Ns,
                zone: Some("UTC".into()),
            },
            local_zone: None,
            zone_roots: vec![],
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
            date_literal_range: None,
        },
        operation,
    })))
    .unwrap()
}

fn materialize_message() -> Vec<u8> {
    message(
        json!({"op":"materialize","provider":{"provider":"frames","manifest":7},"schema":1,"query":2,"indices":3,"anchor_time":null,"seed":"0x7","values":5,"options":[]}),
    )
}

fn inputs(target: &str, variant: &str, large: bool, key: i64) -> Attachments {
    let words = [
        Some(""),
        None,
        Some("é"),
        Some("😺"),
        Some("a\0b"),
        Some("東京"),
        Some("red"),
        Some("blue"),
    ];
    let categories = vec![
        Some("red"),
        Some("blue"),
        Some("red"),
        None,
        Some("blue"),
        Some("red"),
        Some("blue"),
        Some("red"),
    ];
    let note_stype = if variant == "prose" {
        "text"
    } else {
        "categorical"
    };
    let note = |rows: usize| {
        text(
            (0..rows)
                .map(|i| {
                    if variant == "allnull" {
                        None
                    } else {
                        words[i % 8]
                    }
                })
                .collect(),
            large,
        )
    };
    let schema = json!({"tables":[
        {"name":"users","source":{"source":["users"]},"columns":[column("user_id","int64","ID"),column("age","int32","numerical"),column("category","string","categorical"),column("note","string",note_stype)],"primary_key":["user_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"orders","source":{"source":["orders"]},"columns":[column("order_id","int64","ID"),column("user_id","int64","ID"),column("marker","int64","numerical"),column("note","string",note_stype)],"primary_key":["order_id"],"frame_key_form":"scalar","time_column":null}],
        "links":[[{"table":"orders","column":"user_id"},{"table":"users","column":"user_id"}]]});
    let mut options = json!({"num_context_samples":if target == "category" {6} else {3},"num_neighbors":[3],"num_hops":1,"random_seed":7});
    if variant == "batched" {
        options["rows_per_call"] = json!(1);
    }
    Attachments::from([
        (AttachmentId(1), bytes(schema)),
        (
            AttachmentId(2),
            bytes(json!(format!(
                "PREDICT users.{target} FOR EACH users.user_id"
            ))),
        ),
        (
            AttachmentId(3),
            arrow(vec![("index", ints([key, 8]))], false),
        ),
        (AttachmentId(5), bytes(options)),
        (
            AttachmentId(7),
            bytes(
                json!({"tables":[{"source":{"source":["users"]},"rows":8},{"source":{"source":["orders"]},"rows":9}]}),
            ),
        ),
        (
            AttachmentId(8),
            arrow(
                vec![
                    ("user_id", ints(1..=8)),
                    (
                        "age",
                        Arc::new(Int32Array::from((21..29).collect::<Vec<_>>())),
                    ),
                    ("category", text(categories, large)),
                    ("note", note(8)),
                ],
                variant == "chunks",
            ),
        ),
        (
            AttachmentId(9),
            arrow(
                vec![
                    ("order_id", ints(0..64)),
                    ("user_id", ints((0..64).map(|i| i / 8 + 1))),
                    ("marker", ints(0..64)),
                    ("note", note(64)),
                ],
                variant == "chunks",
            ),
        ),
    ])
}

fn decoded(out: &Outgoing) -> Value {
    serde_json::to_value(sdm_host::read(out).unwrap()).unwrap()
}

fn terminal(run: &HostRun) -> Result<Materialized, Outgoing> {
    loop {
        let out = run.next()?;
        match sdm_host::read(&out).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Materialize(value),
                ..
            }) => return Ok(value),
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                let _ = run.take_warnings().unwrap();
            }
            other => panic!(
                "text materialization must not fetch or ask a host to select rows: {other:?}"
            ),
        }
    }
}

fn checkpoint(run: &HostRun) -> (Value, Attachments) {
    let state = decoded(&run.take_sampler_state().unwrap().unwrap())["state"].clone();
    let manifest_id = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let manifest = run.take_attachment(manifest_id).unwrap().unwrap();
    let Attachment::Bytes(raw) = &manifest else {
        panic!("manifest")
    };
    let parsed: Value = serde_json::from_slice(raw).unwrap();
    let mut owned = Attachments::new();
    for id in [
        state["memo"].as_u64().unwrap(),
        parsed["metadata"].as_u64().unwrap(),
    ] {
        let id = AttachmentId(id as u32);
        owned.insert(id, run.take_attachment(id).unwrap().unwrap());
    }
    for table in parsed["tables"].as_array().unwrap() {
        let id = AttachmentId(table["rows"].as_u64().unwrap() as u32);
        owned.insert(id, run.take_attachment(id).unwrap().unwrap());
    }
    owned.insert(manifest_id, manifest);
    (state, owned)
}

fn rebound(
    state: &Value,
    owned: &Attachments,
    rechunk: bool,
    change: bool,
) -> (Vec<u8>, Attachments) {
    let old_manifest = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&old_manifest] else {
        panic!("manifest")
    };
    let manifest: Value = serde_json::from_slice(raw).unwrap();
    let mut attached = inputs("age", "ordinary", false, 7);
    attached.remove(&AttachmentId(8));
    attached.remove(&AttachmentId(9));
    attached.insert(AttachmentId(7), bytes(json!({"tables":[]})));
    attached.insert(
        AttachmentId(20),
        owned[&AttachmentId(state["memo"].as_u64().unwrap() as u32)].clone(),
    );
    attached.insert(
        AttachmentId(21),
        owned[&AttachmentId(manifest["metadata"].as_u64().unwrap() as u32)].clone(),
    );
    let mut bindings = Vec::new();
    for (number, table) in manifest["tables"].as_array().unwrap().iter().enumerate() {
        let previous = AttachmentId(table["rows"].as_u64().unwrap() as u32);
        let id = AttachmentId(22 + number as u32);
        let Attachment::Arrow(data) = &owned[&previous] else {
            panic!("text checkpoint rows")
        };
        let mut batches = Vec::new();
        for batch in &data.batches {
            let batch = if change && table["source"]["source"][0] == "orders" {
                let mut columns = batch.columns().to_vec();
                let at = batch.schema().index_of("note").unwrap();
                // Same byte length as "red": changes logical values without
                // relying on buffer length, offsets, validity or row count.
                let array = &columns[at];
                let large = matches!(array.data_type(), arrow_schema::DataType::LargeUtf8);
                let values: Vec<_> = if large {
                    array.as_string::<i64>().iter().collect()
                } else {
                    array.as_string::<i32>().iter().collect()
                };
                columns[at] = text(
                    values
                        .into_iter()
                        .map(|v| if v == Some("red") { Some("RED") } else { v })
                        .collect(),
                    large,
                );
                RecordBatch::try_new(batch.schema(), columns).unwrap()
            } else {
                batch.clone()
            };
            if rechunk {
                for row in 0..batch.num_rows() {
                    batches.push(batch.slice(row, 1));
                }
            } else {
                batches.push(batch);
            }
        }
        attached.insert(
            id,
            Attachment::Arrow(ArrowData::new(data.schema.clone(), batches)),
        );
        bindings.push(json!({"source":table["source"],"rows":id.0}));
    }
    attached.insert(
        AttachmentId(30),
        bytes(json!({"metadata":21,"tables":bindings})),
    );
    let mut start: Value = serde_json::from_slice(&materialize_message()).unwrap();
    start["operation"]["sampler_state"] = json!({"version":state["version"],"memo":20,"frame_checkpoint":{"version":1,"manifest":30}});
    (serde_json::to_vec(&start).unwrap(), attached)
}

fn payloads(run: &HostRun, materialized: &Materialized) -> (Vec<Value>, Attachments) {
    let mut attached = Attachments::new();
    let mut payloads = Vec::new();
    for id in std::iter::once(materialized.state.payload).chain(
        materialized
            .state
            .batches
            .iter()
            .flatten()
            .map(|batch| batch.payload),
    ) {
        let value = run.take_attachment(id).unwrap().unwrap();
        let Attachment::Bytes(raw) = &value else {
            panic!("materialized payload")
        };
        payloads.push(serde_json::from_slice(raw).unwrap());
        attached.insert(id, value);
    }
    let id = materialized.state.query;
    attached.insert(id, run.take_attachment(id).unwrap().unwrap());
    (payloads, attached)
}

fn predict(run: &HostRun, mut materialized: Materialized, target: &str, expected: &Value) {
    let (payloads, attached) = payloads(run, &materialized);
    assert_eq!(json!(payloads), expected["payloads"]);
    if let Some(batches) = &mut materialized.state.batches {
        for (i, batch) in batches.iter_mut().enumerate() {
            batch.request_id = expected["sent"][i + 1]["metadata"]["request_id"]
                .as_str()
                .unwrap()
                .into();
        }
    }
    let start = message(
        json!({"op":"predict","state":materialized.state,"model":"kumo-relational","request_id":expected["sent"][0]["metadata"]["request_id"],"outputs":[],"default_outputs":true}),
    );
    let prediction = HostRun::start(&sdm_host::default_root(), &start, attached).unwrap();
    let mut called = 0;
    loop {
        let out = prediction
            .next()
            .unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
        match sdm_host::read(&out).unwrap() {
            ControlMessage::Step(Step::CallModel(call)) => {
                let Attachment::Bytes(raw) =
                    prediction.take_attachment(call.body).unwrap().unwrap()
                else {
                    panic!("model body")
                };
                let body: Value = serde_json::from_slice(&raw).unwrap();
                assert_eq!(body, expected["sent"][called]);
                called += 1;
                let table = &body["predict"]["instance_table"];
                let at = table["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|v| v == "instance_id")
                    .unwrap();
                let rows: Vec<_> = table["rows"].as_array().unwrap().iter().enumerate().map(|(i,row)| json!({"row_index":i,"id":row[at].as_i64().unwrap().to_string(),"prediction":if target == "category" {json!("red")} else {json!(100.0+called as f64)}})).collect();
                let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
                    id: call.id,
                    outcome: ModelOutcome::Response {
                        status: 200,
                        content_type: "application/json".into(),
                        body: AttachmentId(99),
                    },
                }))
                .unwrap();
                prediction.reply(&reply,Attachments::from([(AttachmentId(99),bytes(json!({"id":"text-feature-reference","model":"kumo-relational","metadata":{},"predictions":rows})))] )).unwrap();
            }
            ControlMessage::Step(Step::Done {
                result: OperationResult::Predict(result),
                ..
            }) => {
                let Attachment::Arrow(data) =
                    prediction.take_attachment(result.result).unwrap().unwrap()
                else {
                    panic!("result rows")
                };
                let batch = &data.batches[0];
                assert_eq!(batch.num_rows(), 2);
                assert_eq!(
                    batch
                        .column_by_name("row_index")
                        .unwrap()
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .values()
                        .as_ref(),
                    &[0, 1]
                );
                assert_eq!(
                    batch
                        .column_by_name("id")
                        .unwrap()
                        .as_primitive::<arrow_array::types::Int64Type>()
                        .values()
                        .as_ref(),
                    &[7, 8]
                );
                let column = batch.column_by_name("prediction").unwrap();
                if target == "category" {
                    assert_eq!(
                        column.as_string::<i32>().iter().collect::<Vec<_>>(),
                        vec![Some("red"), Some("red")]
                    );
                } else {
                    assert_eq!(
                        column
                            .as_primitive::<arrow_array::types::Float64Type>()
                            .values()
                            .as_ref(),
                        if called == 1 {
                            &[101., 101.]
                        } else {
                            &[101., 102.]
                        }
                    );
                }
                assert_eq!(called, expected["sent"].as_array().unwrap().len());
                break;
            }
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                let _ = prediction.take_warnings().unwrap();
            }
            other => panic!("prediction can only call the scripted model: {other:?}"),
        }
    }
}

#[test]
fn text_physical_families_reuse_scalar_labels_sampling_payloads_and_results() {
    for large in [false, true] {
        for target in ["age", "category"] {
            for variant in ["ordinary", "chunks", "allnull", "batched", "prose"] {
                let run = HostRun::start(
                    &sdm_host::default_root(),
                    &materialize_message(),
                    inputs(target, variant, large, 7),
                )
                .unwrap();
                let materialized =
                    terminal(&run).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
                predict(&run, materialized, target, &oracle(target, variant));
            }
        }
    }
}

#[test]
fn text_checkpoints_keep_values_across_fresh_ids_and_chunk_boundaries() {
    for large in [false, true] {
        let run = HostRun::start(
            &sdm_host::default_root(),
            &materialize_message(),
            inputs("age", "ordinary", large, 7),
        )
        .unwrap();
        terminal(&run).unwrap();
        let (state, owned) = checkpoint(&run);
        drop(run);
        let (message, attached) = rebound(&state, &owned, true, false);
        let resumed = HostRun::start(&sdm_host::default_root(), &message, attached).unwrap();
        let materialized =
            terminal(&resumed).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
        assert_eq!(
            json!(payloads(&resumed, &materialized).0),
            oracle("age", "ordinary")["payloads"]
        );
        let (message, attached) = rebound(&state, &owned, false, true);
        let failed = HostRun::start(&sdm_host::default_root(), &message, attached)
            .err()
            .expect("mutated text snapshot");
        assert_eq!(decoded(&failed)["error"]["code"], "INVALID_REQUEST");
        assert_eq!(
            decoded(&failed)["error"]["drop_held"],
            json!(["sampler_state"])
        );
    }
}

#[test]
fn unknown_entity_failure_retains_the_text_snapshot_before_neighbor_draws() {
    let run = HostRun::start(
        &sdm_host::default_root(),
        &materialize_message(),
        inputs("age", "ordinary", true, 9999),
    )
    .unwrap();
    let failed = terminal(&run).expect_err("unknown entity");
    assert_eq!(decoded(&failed)["error"]["code"], "INVALID_REQUEST");
    let (state, owned) = checkpoint(&run);
    let (message, attached) = rebound(&state, &owned, true, false);
    let resumed = HostRun::start(&sdm_host::default_root(), &message, attached).unwrap();
    let materialized = terminal(&resumed).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
    assert_eq!(
        json!(payloads(&resumed, &materialized).0),
        oracle("age", "ordinary")["payloads"]
    );
}

#[test]
fn text_identifying_columns_stay_explicitly_unfinished_for_both_offsets() {
    for large in [false, true] {
        let mut attached = inputs("age", "ordinary", large, 7);
        let Attachment::Bytes(raw) = &attached[&AttachmentId(1)] else {
            panic!("schema")
        };
        let mut schema: Value = serde_json::from_slice(raw).unwrap();
        schema["tables"][0]["columns"][0]["dtype"] = json!("string");
        attached.insert(AttachmentId(1), bytes(schema));
        let failed = HostRun::start(&sdm_host::default_root(), &materialize_message(), attached);
        let error = match failed {
            Ok(run) => terminal(&run).expect_err("text key remains unfinished"),
            Err(error) => error,
        };
        assert_eq!(
            decoded(&error)["error"]["fields"]["rule"],
            json!({"t":"utf8","v":"frame_text_key"})
        );
    }
}
