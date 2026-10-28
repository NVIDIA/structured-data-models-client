// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plain Arrow twins of actual frozen public LINK calls, without Python.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, TimestampMicrosecondArray, UInt64Array,
};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::frames::materialize::{FramePreparation, PreparedFrameValue};
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
    let mut schema = json!({"tables":[
        {"name":"users","source":{"source":["users"]},"columns":[column("user_id","int64","ID"),column("age","int64","numerical")],"primary_key":["user_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"items","source":{"source":["items"]},"columns":[column("item_id","int64","ID"),column("amount","float64","numerical")],"primary_key":["item_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"buys","source":{"source":["buys"]},"columns":[column("buy_id","int64","ID"),column("user_id","int64","ID"),column("item_id","int64","ID"),column("ts","date","timestamp")],"primary_key":["buy_id"],"frame_key_form":"scalar","time_column":"ts"}],
        "links":[[{"table":"buys","column":"user_id"},{"table":"users","column":"user_id"}],[{"table":"buys","column":"item_id"},{"table":"items","column":"item_id"}]]});
    if shape == "composite-history" {
        schema["tables"][2]["columns"]
            .as_array_mut()
            .unwrap()
            .push(column("part", "int64", "ID"));
        schema["tables"][2]["primary_key"] = json!(["buy_id", "part"]);
        schema["tables"][2]["frame_key_form"] = json!("sequence");
    }
    let mut history = vec![
        ("buy_id", ints(0..120)),
        ("user_id", ints((0..120).map(|row| row % 8))),
        ("item_id", ints((0..120).map(|row| (row * 3) % 4))),
        (
            "ts",
            Arc::new(TimestampMicrosecondArray::from(
                (0..120)
                    .map(|row| 1_672_531_200_000_000 + row * 5 * 86_400_000_000)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ];
    if shape == "composite-history" {
        history.push(("part", ints(std::iter::repeat_n(0, 120))));
    }
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
        (AttachmentId(10), arrow(history, chunks)),
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

/// Each independent run receives exactly the owners its typed input declares.
/// Retaining and rebinding a useful value must not narrow native list children.
struct FrameStages {
    held: Attachments,
}

impl FrameStages {
    fn stage(&mut self, value: Value) -> Result<PreparedFrameValue, Value> {
        let input: FramePreparation = serde_json::from_value(value).unwrap();
        let mut attached: Attachments = input
            .attachments()
            .into_iter()
            .map(|(id, _)| (id, self.held[&id].clone()))
            .collect();
        if matches!(input, FramePreparation::Graph { .. }) {
            for id in [8, 9, 10].map(AttachmentId) {
                attached.insert(id, self.held[&id].clone());
            }
        }
        let id = AttachmentId(self.held.keys().map(|id| id.0).max().unwrap_or(0) + 100);
        attached.insert(id, Attachment::Bytes(serde_json::to_vec(&input).unwrap()));
        let start = message(json!({"op":"prepare_frames","input":id}));
        let run = HostRun::start(&sdm_host::default_root(), &start, attached)
            .map_err(|error| decoded(&error))?;
        let mut value = loop {
            match sdm_host::read(&run.next().map_err(|error| decoded(&error))?).unwrap() {
                ControlMessage::Step(Step::Done {
                    result: OperationResult::PrepareFrames(out),
                }) => {
                    let Some(Attachment::Bytes(bytes)) = run.take_attachment(out.value).unwrap()
                    else {
                        panic!("typed frame result")
                    };
                    break serde_json::from_slice::<PreparedFrameValue>(&bytes).unwrap();
                }
                ControlMessage::Warnings { .. } => {}
                ControlMessage::Step(Step::Warnings) => {
                    let _ = run.take_warnings().unwrap();
                }
                other => panic!("unexpected native frame I/O: {other:?}"),
            }
        };
        // Output ids belong to this run, whose allocator may reuse 1/2/3.
        // Preserve aliases while rebinding actual returned owners into the
        // fixture namespace; inherited already-rebound ids remain unchanged.
        let mut aliases = std::collections::BTreeMap::new();
        rebind_frame_product(&mut value, &mut |id| {
            let old = *id;
            if let Some(rebound) = aliases.get(&old) {
                *id = *rebound;
                return;
            }
            let rebound = if let Some(owner) = run.take_attachment(old).unwrap() {
                let fresh = AttachmentId(
                    (self.held.keys().map(|id| id.0).max().unwrap_or(0) + 1).max(1000),
                );
                self.held.insert(fresh, owner);
                fresh
            } else {
                assert!(
                    self.held.contains_key(&old),
                    "missing inherited owner {old:?}"
                );
                old
            };
            aliases.insert(old, rebound);
            *id = rebound;
        });
        Ok(value)
    }
}

/// Explicit typed resource fields only. Counts, ordinals and native source
/// positions are data and can never be treated as attachment ids here.
fn rebind_frame_product(value: &mut PreparedFrameValue, keep: &mut impl FnMut(&mut AttachmentId)) {
    use sdm_contract::frames::materialize::{
        CompletedFrameTask, FrameRequestPlan, LabelledFrameTask, PreparedFrameGraph,
    };

    fn graph(value: &mut PreparedFrameGraph, keep: &mut impl FnMut(&mut AttachmentId)) {
        keep(&mut value.owner);
    }
    fn labels(value: &mut LabelledFrameTask, keep: &mut impl FnMut(&mut AttachmentId)) {
        graph(&mut value.graph, keep);
        for id in &mut value.context {
            keep(id);
        }
    }
    fn task(value: &mut CompletedFrameTask, keep: &mut impl FnMut(&mut AttachmentId)) {
        labels(&mut value.labelled, keep);
        for id in &mut value.prediction {
            keep(id);
        }
        for id in &mut value.owner {
            keep(id);
        }
    }
    fn plan(value: &mut FrameRequestPlan, keep: &mut impl FnMut(&mut AttachmentId)) {
        task(&mut value.task, keep);
        keep(&mut value.sampling_values);
    }
    match value {
        PreparedFrameValue::SourceMeaning(_)
        | PreparedFrameValue::TableMeaning(_)
        | PreparedFrameValue::KeyMeaning(_)
        | PreparedFrameValue::KeyNulls(_)
        | PreparedFrameValue::ColumnMeaning(_)
        | PreparedFrameValue::IntegerColumn(_) => {}
        PreparedFrameValue::Graph(value) => graph(value, keep),
        PreparedFrameValue::Population(value) => {
            graph(&mut value.graph, keep);
            keep(&mut value.owner);
        }
        PreparedFrameValue::LabelBatch(value) => keep(&mut value.owner),
        PreparedFrameValue::Labels(value) => labels(value, keep),
        PreparedFrameValue::Task(value) => task(value, keep),
        PreparedFrameValue::Plan(value) => plan(value, keep),
        PreparedFrameValue::References(value) => {
            plan(&mut value.plan, keep);
            keep(&mut value.owner);
        }
        PreparedFrameValue::Context(value) => {
            plan(&mut value.plan, keep);
            keep(&mut value.anchors);
            for table in &mut value.tables {
                keep(&mut table.rows);
                keep(&mut table.source_rows);
            }
            for link in &mut value.links {
                keep(&mut link.pairs);
            }
        }
        PreparedFrameValue::Part(value) => keep(&mut value.payload),
        other => panic!("unexpected frame fixture product: {other:?}"),
    }
}

#[test]
fn unsigned_link_native_rows_survive_owner_rebinding_until_the_real_request_boundary() {
    use arrow_array::types::UInt64Type;
    let mut baseline_labels = None;
    let mut baseline_keys = None;
    for base in [0, (1_u64 << 53) + 1, (i64::MAX as u64) + 1] {
        let mut held = inputs("empty-one-neighborhood", false);
        // Same declared schema and graph shape as the frozen public case.
        // Only the actual physical key/FK buffers are unsigned.
        for (id, column) in [(AttachmentId(9), "item_id"), (AttachmentId(10), "item_id")] {
            let Attachment::Arrow(data) = &held[&id] else {
                panic!("raw frame")
            };
            let batch = &data.batches[0];
            let columns = batch
                .schema()
                .fields()
                .iter()
                .zip(batch.columns())
                .map(|(field, array)| {
                    let array = if field.name() == column {
                        Arc::new(UInt64Array::from(
                            array
                                .as_primitive::<arrow_array::types::Int64Type>()
                                .values()
                                .iter()
                                .map(|value| base + *value as u64)
                                .collect::<Vec<_>>(),
                        )) as ArrayRef
                    } else {
                        Arc::clone(array)
                    };
                    (field.name().clone(), array)
                })
                .collect::<Vec<_>>();
            let batch = RecordBatch::try_from_iter(columns).unwrap();
            held.insert(
                id,
                Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
            );
        }
        held.insert(AttachmentId(20), bytes(json!({"num_context_samples":3,"num_hops":1,"num_neighbors":[3],"rows_per_call":200})));
        let mut stages = FrameStages { held };
        let native_tables = [("users", vec!["user_id", "age"]), ("items", vec!["item_id", "amount"]), ("buys", vec!["buy_id", "user_id", "item_id", "ts"])].map(|(name, columns)| {
            json!({"source":{"source":[name]},"columns":columns.into_iter().map(|name| json!({"name":name,"declared":null,"logical_text":null,"strings":null})).collect::<Vec<_>>()})
        });
        let PreparedFrameValue::Graph(graph) = stages.stage(json!({"stage":"graph","schema":1,"frames":7,"checkpoint":null,"retained":null,"native_tables":native_tables,"values":5,"options":[],"prediction_rows":2,"limits":{}})).unwrap() else { panic!("graph") };
        let PreparedFrameValue::Population(population) = stages.stage(json!({"stage":"population","graph":graph,"query":2,"values":20,"seed":"0x7","entropy":7,"anchor_time":{"t":"ts","unit":"ns","v":"1677628800000000000"},"anchor_entity":false,"num_context_samples":null,"limits":{}})).unwrap() else { panic!("population") };
        let mut batches = Vec::new();
        let mut native_labels = Vec::new();
        for start in (0..population.rows).step_by(population.batch_rows as usize) {
            let PreparedFrameValue::LabelBatch(batch) = stages.stage(json!({"stage":"label_batch","population":population,"range":{"start":start,"stop":population.rows.min(start + u64::from(population.batch_rows))},"previous":batches.last(),"converted":{"kind":"core_keys","values":null}})).unwrap() else { panic!("label batch") };
            let Attachment::FrameLabelBatch(owner) = &stages.held[&batch.owner] else {
                panic!("retained label rows")
            };
            let raw = owner.rows().unwrap();
            let values = raw.batches[0]
                .column_by_name("target")
                .unwrap()
                .as_list::<i32>()
                .values()
                .as_primitive::<UInt64Type>();
            assert!(!values.is_empty());
            assert!(
                values
                    .values()
                    .iter()
                    .all(|value| *value >= base && *value < base + 4)
            );
            native_labels.push(
                values
                    .values()
                    .iter()
                    .map(|value| value - base)
                    .collect::<Vec<_>>(),
            );
            let complete = batch.complete;
            batches.push(batch);
            if complete {
                break;
            }
        }
        if let Some(expected) = &baseline_labels {
            assert_eq!(&native_labels, expected);
        } else {
            baseline_labels = Some(native_labels);
        }
        let PreparedFrameValue::Labels(labelled) = stages
            .stage(json!({"stage":"labels","population":population,"batches":batches}))
            .unwrap()
        else {
            panic!("labels")
        };
        let Some(Attachment::Arrow(rows)) = labelled.context.map(|id| &stages.held[&id]) else {
            panic!("published context")
        };
        assert_eq!(
            rows.batches[0]
                .column_by_name("target")
                .unwrap()
                .as_list::<i32>()
                .values()
                .data_type(),
            &arrow_schema::DataType::UInt64
        );
        let PreparedFrameValue::Task(task) = stages.stage(json!({"stage":"task","labelled":labelled,"prediction":{"kind":"core_keys","values":3},"values":20})).unwrap() else { panic!("task") };
        let PreparedFrameValue::Plan(plan) = stages
            .stage(json!({"stage":"plan","task":task,"values":20}))
            .unwrap()
        else {
            panic!("plan")
        };
        let PreparedFrameValue::References(references) = stages.stage(json!({"stage":"references","plan":plan,"converted":{"kind":"core_keys","values":null}})).unwrap() else { panic!("references") };
        let PreparedFrameValue::Context(context) = stages.stage(json!({"stage":"context","references":references,"range":{"start":0,"stop":2},"converted":{"kind":"core_keys","values":null}})).unwrap() else { panic!("context") };
        let items = context
            .tables
            .iter()
            .find(|table| table.name == "items")
            .unwrap();
        let Attachment::Arrow(rows) = &stages.held[&items.rows] else {
            panic!("sampled unsigned keys")
        };
        let keys = rows.batches[0]
            .column_by_name("item_id")
            .unwrap()
            .as_primitive::<UInt64Type>();
        assert!(!keys.is_empty());
        assert!(
            keys.values()
                .iter()
                .all(|value| *value >= base && *value < base + 4)
        );
        let selected = keys
            .values()
            .iter()
            .map(|value| value - base)
            .collect::<Vec<_>>();
        if let Some(expected) = &baseline_keys {
            assert_eq!(&selected, expected);
        } else {
            baseline_keys = Some(selected);
        }
        let output = stages.stage(json!({"stage":"payload","context":context,"text":{"outcome":"converted","columns":[]},"values":null}));
        if base > i64::MAX as u64 {
            let error = output.unwrap_err();
            assert_eq!(error["error"]["sentence"], "past_int64");
            assert_eq!(
                error["error"]["fields"]["table"],
                json!({"t":"utf8","v":"items"})
            );
            assert_eq!(
                error["error"]["fields"]["column"],
                json!({"t":"utf8","v":"item_id"})
            );
        } else {
            assert!(matches!(output.unwrap(), PreparedFrameValue::Part(_)));
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
        "composite-history",
    ] {
        for chunks in [false, true] {
            let (run, materialized) = materialize(shape, chunks);
            check_prediction(run, materialized, &oracle(shape));
        }
    }
}

#[test]
fn constructed_key_facts_and_native_null_admission_are_closed_host_products() {
    let mut stages = FrameStages {
        held: inputs("composite-history", false),
    };
    let Attachment::Bytes(schema) = &stages.held[&AttachmentId(1)] else {
        panic!("schema")
    };
    let schema: Value = serde_json::from_slice(schema).unwrap();
    let table = schema["tables"][2].clone();
    let native = [
        ("buy_id", "int64"),
        ("user_id", "int64"),
        ("item_id", "int64"),
        ("ts", "timestamp"),
        ("part", "int64"),
    ]
    .into_iter()
    .map(|(name, kind)| {
        let physical = if kind == "timestamp" {
            json!({"type":"timestamp","unit":"us","zone":null})
        } else {
            json!({"type":kind})
        };
        json!({"name":name,"declared":null,"values":{"storage":"typed","physical":physical}})
    })
    .collect::<Vec<_>>();
    let PreparedFrameValue::TableMeaning(typed) = stages.stage(json!({"stage":"table_meaning","input":{"table":table,"native":native,"row_count":120}})).unwrap() else { panic!("table meaning") };
    let PreparedFrameValue::KeyMeaning(key) = stages
        .stage(json!({"stage":"table_key","table":typed,"primary_key":["buy_id","part"]}))
        .unwrap()
    else {
        panic!("key meaning")
    };
    let definition = key.composite.unwrap();
    assert_eq!(definition.derived, "__kumo_key_6275795f6964_70617274");
    assert_eq!(
        definition
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        ["buy_id", "part"]
    );
    let no_null = stages.stage(json!({"stage":"key_nulls","input":{"definition":definition,"column":"buy_id","missing":false}})).unwrap();
    assert!(matches!(no_null, PreparedFrameValue::KeyNulls(_)));
    let refused = stages.stage(json!({"stage":"key_nulls","input":{"definition":definition,"column":"part","missing":true}})).unwrap_err();
    assert_eq!(refused["error"]["code"], "INVALID_REQUEST");
    assert_eq!(
        refused["error"]["fields"]["native_key_reason"],
        json!({"t":"utf8","v":"component_null"})
    );
    let mut forged = serde_json::to_value(&definition).unwrap();
    forged["derived"] = json!("not-the-selected-name");
    let refused = stages.stage(json!({"stage":"key_nulls","input":{"definition":forged,"column":"part","missing":false}})).unwrap_err();
    assert_eq!(refused["error"]["code"], "INVALID_MESSAGE");
}

#[test]
fn native_row_identity_manifest_is_not_a_scalar_topology_or_cache_alias() {
    let mut input = inputs("composite-history", false);
    // The observed strings need not equal a standard rendering: actual native
    // replace/map/add operations are authoritative source observations.
    let identity: Vec<String> = (0..120).map(|row| format!("observed-{row}")).collect();
    input.insert(
        AttachmentId(11),
        arrow(
            vec![(
                "identity",
                Arc::new(arrow_array::StringArray::from(identity)),
            )],
            false,
        ),
    );
    let Attachment::Bytes(manifest) = input.get_mut(&AttachmentId(7)).unwrap() else {
        panic!("manifest")
    };
    let mut manifest_value: Value = serde_json::from_slice(manifest).unwrap();
    manifest_value["row_identities"] = json!([{
        "definition":{"table":"buys","source":{"source":["buys"]},"columns":[{"name":"buy_id","dtype":"int"},{"name":"part","dtype":"int"}],"derived":"__kumo_key_6275795f6964_70617274","rows":120,"codec":"escaped_text_v1"},
        "values":{"values":11,"encoding":"utf8"}
    }]);
    *manifest = serde_json::to_vec(&manifest_value).unwrap();
    let start = message(
        json!({"op":"materialize","provider":{"provider":"frames","manifest":7},"schema":1,"query":2,"indices":3,"seed":"0x7","values":5,"options":[],"anchor_time":{"t":"ts","unit":"ns","v":"1677628800000000000"}}),
    );
    let run = HostRun::start(&sdm_host::default_root(), &start, input).unwrap();
    loop {
        let outgoing = run
            .next()
            .unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
        match sdm_host::read(&outgoing).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Materialize(value),
            }) => {
                assert!(value.query.is_some());
                assert!(run.take_attachment(value.state.payload).unwrap().is_some());
                break;
            }
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                let _ = run.take_warnings().unwrap();
            }
            other => panic!("unexpected native identity host effect: {other:?}"),
        }
    }
}

#[test]
fn resolved_question_survives_final_time_metadata_change_and_rejects_wrong_query_owner() {
    let mut stages = FrameStages {
        held: inputs("composite-history", false),
    };
    let PreparedFrameValue::Question(question) = stages.stage(json!({"stage":"question","schema":1,"query":2,"candidates_given":false,"statistic_given":false,"top_k_given":false})).unwrap() else { panic!("resolved question") };
    let PreparedFrameValue::Questions(questions) = stages.stage(json!({"stage":"questions","input":{"question":question,"candidates":null,"values":5,"options":[],"option_facts":[],"prediction_rows":2,"limits":{}}})).unwrap() else { panic!("ordinary questions") };
    let owner = questions.question.expect("ordinary query meaning retained");
    let Attachment::Bytes(schema) = stages.held.get_mut(&AttachmentId(1)).unwrap() else {
        panic!("schema")
    };
    let mut value: Value = serde_json::from_slice(schema).unwrap();
    value["tables"][2]["time_column"] = Value::Null;
    *schema = serde_json::to_vec(&value).unwrap();
    let native = [("users", vec!["user_id", "age"]), ("items", vec!["item_id", "amount"]), ("buys", vec!["buy_id", "user_id", "item_id", "ts", "part"])].map(|(name, columns)| json!({"source":{"source":[name]},"columns":columns.into_iter().map(|name| json!({"name":name,"declared":null,"logical_text":null,"strings":null})).collect::<Vec<_>>()}));
    let PreparedFrameValue::Graph(graph) = stages.stage(json!({"stage":"graph","schema":1,"frames":7,"checkpoint":null,"retained":null,"native_tables":native,"values":5,"options":[],"prediction_rows":2,"limits":{}})).unwrap() else { panic!("final physical graph") };
    let population = json!({"stage":"population","graph":graph,"query":2,"question":owner,"values":5,"seed":"0x7","entropy":null,"anchor_time":{"t":"ts","unit":"ns","v":"1677628800000000000"},"anchor_entity":false,"num_context_samples":3,"limits":{}});
    let error = stages.stage(population.clone()).unwrap_err();
    assert_eq!(error["error"]["code"], "INVALID_REQUEST");
    assert_eq!(
        error["error"]["fields"]["native_key_reason"],
        json!({"t":"utf8","v":"time_bound_missing"})
    );
    assert_eq!(
        error["error"]["fields"]["table"],
        json!({"t":"utf8","v":"buys"})
    );
    stages.held.insert(
        AttachmentId(2),
        bytes(json!("PREDICT users.age FOR EACH users.user_id")),
    );
    let error = stages.stage(population).unwrap_err();
    assert_eq!(error["error"]["code"], "INVALID_MESSAGE");
}
