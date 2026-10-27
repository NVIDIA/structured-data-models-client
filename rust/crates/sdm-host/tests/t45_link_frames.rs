// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain Arrow twins of actual frozen public LINK calls, without Python.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Materialized, Operation, OperationResult};
use sdm_contract::run::{ModelOutcome, ModelReply, StartRun};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing};
use serde_json::{Value, json};

fn oracle(shape: &str) -> Value {
    let all: Value = serde_json::from_str(include_str!(
        "../../../../packages/structured-data-models-client/tests/t45_frames_pandas3_reference.json"
    ))
    .unwrap();
    all["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["kind"] == "int64" && case["shape"] == shape)
        .unwrap()["expected"]
        .clone()
}

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn arrow(columns: Vec<(&str, ArrayRef)>, chunks: bool) -> Attachment {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    let batches = if chunks {
        vec![batch.slice(0, 2), batch.slice(2, batch.num_rows() - 2)]
    } else {
        vec![batch.clone()]
    };
    Attachment::Arrow(ArrowData::new(batch.schema(), batches))
}

fn ints(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values.into_iter().collect::<Vec<_>>()))
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

fn inputs(shape: &str, chunks: bool) -> Attachments {
    let query = match shape {
        "reverse" => "PREDICT LIST_DISTINCT(buys.user_id, 0, 30, days) FOR EACH items.item_id",
        "empty-one-neighborhood" => {
            "PREDICT LIST_DISTINCT(buys.item_id WHERE buys.user_id != 0, 0, 30, days) FOR EACH users.user_id"
        }
        _ => "PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) FOR EACH users.user_id",
    };
    let schema = json!({"tables":[
        {"name":"users","source":{"source":["users"]},"columns":[column("user_id","int64","ID"),column("age","int64","numerical")],"primary_key":["user_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"items","source":{"source":["items"]},"columns":[column("item_id","int64","ID"),column("amount","float64","numerical")],"primary_key":["item_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"buys","source":{"source":["buys"]},"columns":[column("buy_id","int64","ID"),column("user_id","int64","ID"),column("item_id","int64","ID"),column("ts","date","timestamp")],"primary_key":["buy_id"],"frame_key_form":"scalar","time_column":"ts"}],
        "links":[[{"table":"buys","column":"user_id"},{"table":"users","column":"user_id"}],[{"table":"buys","column":"item_id"},{"table":"items","column":"item_id"}]]});
    let indices = if shape == "201-repeated" {
        (0..201).map(|row| row % 8).collect()
    } else {
        vec![0, 1]
    };
    Attachments::from([
        (AttachmentId(1), bytes(schema)),
        (AttachmentId(2), bytes(json!(query))),
        (
            AttachmentId(3),
            arrow(vec![("index", ints(indices))], false),
        ),
        (AttachmentId(5), bytes(json!({"random_seed":7}))),
        (
            AttachmentId(7),
            bytes(
                json!({"tables":[{"source":{"source":["users"]},"rows":8},{"source":{"source":["items"]},"rows":9},{"source":{"source":["buys"]},"rows":10}]}),
            ),
        ),
        (
            AttachmentId(8),
            arrow(vec![("user_id", ints(0..8)), ("age", ints(20..28))], chunks),
        ),
        (
            AttachmentId(9),
            arrow(
                vec![
                    ("item_id", ints(0..4)),
                    (
                        "amount",
                        Arc::new(Float64Array::from(vec![3.0, 4.0, 5.0, 6.0])),
                    ),
                ],
                chunks,
            ),
        ),
        (
            AttachmentId(10),
            arrow(
                vec![
                    ("buy_id", ints(0..120)),
                    ("user_id", ints((0..120).map(|row| row % 8))),
                    ("item_id", ints((0..120).map(|row| (row * 3) % 4))),
                    (
                        "ts",
                        Arc::new(TimestampMicrosecondArray::from(
                            (0..120)
                                .map(|row| 1_672_531_200_000_000 + row * 5 * 86_400_000_000)
                                .collect::<Vec<_>>(),
                        )),
                    ),
                ],
                chunks,
            ),
        ),
    ])
}

fn decoded(out: &Outgoing) -> Value {
    serde_json::to_value(sdm_host::read(out).unwrap()).unwrap()
}

fn materialize(shape: &str, chunks: bool) -> (HostRun, Materialized) {
    let start = message(
        json!({"op":"materialize","provider":{"provider":"frames","manifest":7},"schema":1,"query":2,"indices":3,"seed":"0x7","values":5,"options":[],"anchor_time":{"t":"ts","unit":"ns","v":"1677628800000000000"}}),
    );
    let run = HostRun::start(&sdm_host::default_root(), &start, inputs(shape, chunks)).unwrap();
    loop {
        let out = run
            .next()
            .unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
        match sdm_host::read(&out).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Materialize(value),
            }) => return (run, value),
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                let _ = run.take_warnings().unwrap();
            }
            other => panic!("plain Arrow LINK must not request host selection or I/O: {other:?}"),
        }
    }
}

fn check_prediction(run: HostRun, mut materialized: Materialized, expected: &Value) {
    let mut attached = Attachments::new();
    let payload_ids = std::iter::once(materialized.state.payload).chain(
        materialized
            .state
            .batches
            .iter()
            .flatten()
            .map(|batch| batch.payload),
    );
    let mut payloads = Vec::new();
    for id in payload_ids {
        let value = run.take_attachment(id).unwrap().unwrap();
        let Attachment::Bytes(raw) = &value else {
            panic!("LINK payload bytes")
        };
        payloads.push(serde_json::from_slice::<Value>(raw).unwrap());
        attached.insert(id, value);
    }
    assert_eq!(json!(payloads), expected["payloads"]);
    attached.insert(
        materialized.state.query,
        run.take_attachment(materialized.state.query)
            .unwrap()
            .unwrap(),
    );
    if let Some(batches) = &mut materialized.state.batches {
        for (index, batch) in batches.iter_mut().enumerate() {
            batch.request_id = Some(
                expected["sent"][index + 1]["metadata"]["request_id"]
                    .as_str()
                    .unwrap()
                    .into(),
            );
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
                    panic!("LINK model body")
                };
                let body: Value = serde_json::from_slice(&raw).unwrap();
                assert_eq!(body, expected["sent"][called]);
                called += 1;
                let table = &body["predict"]["instance_table"];
                let at = table["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .position(|name| name == "instance_id")
                    .unwrap();
                let rows = table["rows"].as_array().unwrap().iter().enumerate().map(|(index, row)| json!({"row_index":index,"id":row[at].as_i64().unwrap().to_string(),"prediction":null,"rankings":[{"item":"display-rhs","score":0.9}]})).collect::<Vec<_>>();
                let reply = sdm_host::encode(&ControlMessage::ModelReply(ModelReply {
                    id: call.id,
                    outcome: ModelOutcome::Response {
                        status: 200,
                        content_type: "application/json".into(),
                        body: AttachmentId(99),
                    },
                }))
                .unwrap();
                prediction.reply(&reply, Attachments::from([(AttachmentId(99),bytes(json!({"id":"frozen-full-link","model":"kumo-relational","metadata":{},"predictions":rows})))])).unwrap();
            }
            ControlMessage::Step(Step::Done {
                result: OperationResult::Predict(result),
            }) => {
                assert_eq!(called, expected["sent"].as_array().unwrap().len());
                let Attachment::Arrow(data) =
                    prediction.take_attachment(result.result).unwrap().unwrap()
                else {
                    panic!("LINK result rows")
                };
                let expected_rows = expected["result"]["data"].as_array().unwrap();
                let rows = data
                    .batches
                    .iter()
                    .flat_map(|batch| {
                        let positions = batch
                            .column_by_name("row_index")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int64Type>();
                        let ids = batch
                            .column_by_name("id")
                            .unwrap()
                            .as_primitive::<arrow_array::types::Int64Type>();
                        let rankings = batch.column_by_name("rankings").unwrap().as_list::<i32>();
                        (0..batch.num_rows()).map(move |row| {
                            let values = rankings.value(row);
                            let items = values
                                .as_string::<i32>()
                                .iter()
                                .map(|item| serde_json::from_str::<Value>(item.unwrap()).unwrap())
                                .collect::<Vec<_>>();
                            (positions.value(row), ids.value(row), items)
                        })
                    })
                    .collect::<Vec<_>>();
                assert_eq!(rows.len(), expected_rows.len());
                for ((position, id, rankings), expected) in rows.iter().zip(expected_rows) {
                    assert_eq!(
                        (*position, *id),
                        (expected[0].as_i64().unwrap(), expected[1].as_i64().unwrap())
                    );
                    assert_eq!(json!(rankings), expected[2]);
                }
                break;
            }
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                let _ = prediction.take_warnings().unwrap();
            }
            other => panic!("LINK prediction crossed an unexpected host boundary: {other:?}"),
        }
    }
}

#[test]
fn plain_arrow_link_materializes_and_predicts_as_the_frozen_client() {
    for shape in [
        "ordinary",
        "reverse",
        "empty-one-neighborhood",
        "201-repeated",
    ] {
        for chunks in [false, true] {
            let (run, materialized) = materialize(shape, chunks);
            check_prediction(run, materialized, &oracle(shape));
        }
    }
}
