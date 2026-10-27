// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed temporal frame answers use shared Rust selection, without a fetch.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Int32Array, Int64Array, RecordBatch, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::TimeUnit;
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Materialized, Operation, OperationResult};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing};
use serde_json::{Value, json};

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn arrow(columns: Vec<(&str, ArrayRef)>) -> Attachment {
    let batch = RecordBatch::try_from_iter(columns).unwrap();
    Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
}

fn integer(values: Vec<i64>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn column(name: &str, dtype: &str, stype: &str) -> Value {
    json!({"name":name,"dtype":dtype,"stype":stype,"stype_before_key":null})
}

fn message(anchor: Option<i64>) -> Vec<u8> {
    let operation: Operation = serde_json::from_value(json!({
        "op":"materialize", "provider":{"provider":"frames","manifest":7},
        "schema":1,"query":2,"indices":3,"seed":"0x7","values":5,"options":[],
        "anchor_time":anchor.map(|value| json!({"t":"ts","unit":"ns","v":value.to_string()}))
    }))
    .unwrap();
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

fn inputs(unit: TimeUnit, missing: bool, zone: Option<&str>) -> Attachments {
    let counts = if unit == TimeUnit::Second {
        [1, -2, 0, -1, -1, 2, 0, 3]
    } else {
        [1200, -1800, 500, -1200, -1500, -700, 0, -500]
    };
    let multiple = match unit {
        TimeUnit::Second | TimeUnit::Millisecond => 1,
        TimeUnit::Microsecond => 1000,
        TimeUnit::Nanosecond => 1_000_000,
    };
    let times: Vec<Option<i64>> = (0..64)
        .map(|row| (!missing).then_some(counts[row % 8] * multiple))
        .collect();
    let times: ArrayRef = match unit {
        TimeUnit::Second => Arc::new(TimestampSecondArray::from(times).with_timezone_opt(zone)),
        TimeUnit::Millisecond => {
            Arc::new(TimestampMillisecondArray::from(times).with_timezone_opt(zone))
        }
        TimeUnit::Microsecond => {
            Arc::new(TimestampMicrosecondArray::from(times).with_timezone_opt(zone))
        }
        TimeUnit::Nanosecond => {
            Arc::new(TimestampNanosecondArray::from(times).with_timezone_opt(zone))
        }
    };
    let schema = json!({"tables":[
        {"name":"users","source":{"source":["users"]},"columns":[column("user_id","int64","ID"),column("age","int32","numerical")],"primary_key":["user_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"events","source":{"source":["events"]},"columns":[column("event_id","int64","ID"),column("user_id","int64","ID"),column("marker","int64","numerical"),column("amount","int64","numerical"),column("at","date","timestamp")],"primary_key":["event_id"],"frame_key_form":"scalar","time_column":"at"}],
        "links":[[{"table":"events","column":"user_id"},{"table":"users","column":"user_id"}]]});
    Attachments::from([
        (AttachmentId(1), bytes(schema)),
        (
            AttachmentId(2),
            bytes(json!("PREDICT users.age FOR EACH users.user_id")),
        ),
        (AttachmentId(3), arrow(vec![("index", integer(vec![7, 8]))])),
        (
            AttachmentId(5),
            bytes(
                json!({"num_context_samples":3,"num_neighbors":[2],"num_hops":1,"random_seed":7}),
            ),
        ),
        (
            AttachmentId(7),
            bytes(
                json!({"tables":[{"source":{"source":["users"]},"rows":8},{"source":{"source":["events"]},"rows":9}]}),
            ),
        ),
        (
            AttachmentId(8),
            arrow(vec![
                ("user_id", integer((1..=8).collect())),
                (
                    "age",
                    Arc::new(Int32Array::from((21..29).collect::<Vec<_>>())),
                ),
            ]),
        ),
        (
            AttachmentId(9),
            arrow(vec![
                ("event_id", integer((0..64).collect())),
                ("user_id", integer((0..64).map(|row| row / 8 + 1).collect())),
                ("marker", integer((0..64).collect())),
                ("amount", integer((0..64).collect())),
                ("at", times),
            ]),
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
            other => {
                panic!("temporal frames must not fetch or ask a host to select rows: {other:?}")
            }
        }
    }
}

fn payload(run: &HostRun, materialized: Materialized) -> Value {
    let Attachment::Bytes(raw) = run
        .take_attachment(materialized.state.payload)
        .unwrap()
        .unwrap()
    else {
        panic!("payload bytes")
    };
    serde_json::from_slice(&raw).unwrap()
}

fn values(payload: &Value, table: &str, name: &str) -> Value {
    let table = if table == "instance_table" {
        &payload["predict"][table]
    } else {
        &payload["predict"]["related_tables"][table]
    };
    let at = table["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|column| column == name)
        .unwrap();
    json!(
        table["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row[at].clone())
            .collect::<Vec<_>>()
    )
}

fn checkpoint(run: &HostRun) -> (Value, Attachments) {
    let state = decoded(&run.take_sampler_state().unwrap().unwrap())["state"].clone();
    let mut owned = Attachments::new();
    let memo_id = AttachmentId(state["memo"].as_u64().unwrap() as u32);
    owned.insert(memo_id, run.take_attachment(memo_id).unwrap().unwrap());
    let manifest_id = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let manifest = run.take_attachment(manifest_id).unwrap().unwrap();
    let Attachment::Bytes(raw) = &manifest else {
        panic!("manifest")
    };
    let parsed: Value = serde_json::from_slice(raw).unwrap();
    let metadata_id = AttachmentId(parsed["metadata"].as_u64().unwrap() as u32);
    owned.insert(
        metadata_id,
        run.take_attachment(metadata_id).unwrap().unwrap(),
    );
    for table in parsed["tables"].as_array().unwrap() {
        let id = AttachmentId(table["rows"].as_u64().unwrap() as u32);
        owned.insert(id, run.take_attachment(id).unwrap().unwrap());
    }
    owned.insert(manifest_id, manifest);
    (state, owned)
}

fn manifest(state: &Value, owned: &Attachments) -> Value {
    let id = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&id] else {
        panic!("manifest")
    };
    serde_json::from_slice(raw).unwrap()
}

#[test]
fn typed_units_keep_second_cuts_ties_and_original_feature_instants() {
    for unit in [
        TimeUnit::Second,
        TimeUnit::Millisecond,
        TimeUnit::Microsecond,
        TimeUnit::Nanosecond,
    ] {
        for (anchor, expected) in [
            (100_000_000, json!([50, 54, 58, 62])),
            (
                -800_000_000,
                if unit == TimeUnit::Second {
                    json!([51, 52, 59, 60])
                } else {
                    json!([53, 55, 61, 63])
                },
            ),
        ] {
            let run = HostRun::start(
                &sdm_host::default_root(),
                &message(Some(anchor)),
                inputs(unit, false, None),
            )
            .unwrap();
            let result = terminal(&run).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
            let payload = payload(&run, result);
            assert_eq!(values(&payload, "events", "marker"), expected);
            let rendered = if anchor < 0 {
                "1969-12-31T23:59:59.000000Z"
            } else {
                "1970-01-01T00:00:00.000000Z"
            };
            assert_eq!(
                values(&payload, "instance_table", "__kumo_anchor_time"),
                json!([rendered, rendered])
            );
            if unit != TimeUnit::Second && anchor > 0 {
                assert_eq!(
                    values(&payload, "events", "at"),
                    json!([
                        "1970-01-01T00:00:00.500000Z",
                        "1970-01-01T00:00:00.000000Z",
                        "1970-01-01T00:00:00.500000Z",
                        "1970-01-01T00:00:00.000000Z"
                    ])
                );
            }
        }
    }
}

#[test]
fn default_anchor_uses_valid_snapshot_bounds_and_retains_no_draws_on_timed_hops() {
    let run = HostRun::start(
        &sdm_host::default_root(),
        &message(None),
        inputs(TimeUnit::Nanosecond, false, None),
    )
    .unwrap();
    let result = terminal(&run).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
    let built = payload(&run, result);
    assert_eq!(values(&built, "events", "marker"), json!([54, 48, 62, 56]));
    assert_eq!(
        values(&built, "instance_table", "__kumo_anchor_time"),
        json!(["1970-01-01T00:00:01.000000Z", "1970-01-01T00:00:01.000000Z"])
    );
    let (state, owned) = checkpoint(&run);
    let manifest = manifest(&state, &owned);
    let id = AttachmentId(manifest["metadata"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&id] else {
        panic!("metadata")
    };
    let metadata: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(metadata["neighbors"]["position"], 624);
    assert_eq!(metadata["neighbors"]["words"][0], 7);
    let mut attached = inputs(TimeUnit::Nanosecond, true, None);
    attached.remove(&AttachmentId(8));
    attached.remove(&AttachmentId(9));
    attached.insert(AttachmentId(7), bytes(json!({"tables":[]})));
    attached.insert(
        AttachmentId(20),
        owned[&AttachmentId(state["memo"].as_u64().unwrap() as u32)].clone(),
    );
    attached.insert(AttachmentId(21), owned[&id].clone());
    let mut tables = Vec::new();
    for (number, table) in manifest["tables"].as_array().unwrap().iter().enumerate() {
        let old = AttachmentId(table["rows"].as_u64().unwrap() as u32);
        let new = AttachmentId(22 + number as u32);
        attached.insert(new, owned[&old].clone());
        tables.push(json!({"source":table["source"],"rows":new.0}));
    }
    attached.insert(
        AttachmentId(30),
        bytes(json!({"metadata":21,"tables":tables})),
    );
    let mut start: Value = serde_json::from_slice(&message(None)).unwrap();
    start["operation"]["sampler_state"] = json!({"version":state["version"],"memo":20,"frame_checkpoint":{"version":1,"manifest":30}});
    let next = HostRun::start(
        &sdm_host::default_root(),
        &serde_json::to_vec(&start).unwrap(),
        attached,
    )
    .unwrap();
    let result = terminal(&next).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
    assert_eq!(
        values(&payload(&next, result), "events", "marker"),
        json!([54, 48, 62, 56])
    );
}

#[test]
fn zoned_arrow_timestamps_retain_their_utc_instant() {
    for zone in ["UTC", "America/New_York"] {
        let run = HostRun::start(
            &sdm_host::default_root(),
            &message(Some(100_000_000)),
            inputs(TimeUnit::Nanosecond, false, Some(zone)),
        )
        .unwrap();
        let result = terminal(&run).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
        assert_eq!(
            values(&payload(&run, result), "events", "marker"),
            json!([50, 54, 58, 62])
        );
    }
}

#[test]
fn all_missing_times_refuse_before_snapshot_or_sanitization_notice() {
    let run = HostRun::start(
        &sdm_host::default_root(),
        &message(None),
        inputs(TimeUnit::Nanosecond, true, None),
    )
    .unwrap();
    let failed = terminal(&run).expect_err("a timed key map has no valid rows");
    let error = decoded(&failed);
    assert_eq!(error["error"]["sentence"], "frame_no_valid_rows");
    assert_eq!(
        error["error"]["fields"]["requirements"],
        json!({"t":"utf8","v":"primary key and time column pairs"})
    );
    assert!(run.take_warnings().unwrap().is_none());
    if let Some(state) = run.take_sampler_state().unwrap() {
        assert!(decoded(&state)["state"]["frame_checkpoint"].is_null());
    }
}

#[test]
fn precise_features_follow_frozen_wire_rounding_and_retain_original_checkpoint_counts() {
    let oracle: Value = serde_json::from_str(include_str!(
        "../../../../packages/structured-data-models-client/tests/t44_temporal_reference.json"
    ))
    .unwrap();
    for (name, unit) in [("us", TimeUnit::Microsecond), ("ns", TimeUnit::Nanosecond)] {
        for (side, anchor) in [("positive", 100_000_000), ("negative", -800_000_000)] {
            let observed = &oracle["submicro_cases"][format!("{name}-{side}")];
            let counts: Vec<i64> = observed["input_counts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_i64().unwrap())
                .collect();
            let times: ArrayRef = match unit {
                TimeUnit::Microsecond => Arc::new(TimestampMicrosecondArray::from(counts.clone())),
                TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::from(counts.clone())),
                _ => unreachable!(),
            };
            let mut attached = inputs(unit, false, None);
            let replacement = {
                let Attachment::Arrow(data) = &attached[&AttachmentId(9)] else {
                    panic!("original event rows")
                };
                let batch = &data.batches[0];
                let mut columns = batch.columns().to_vec();
                let at = batch.schema().index_of("at").unwrap();
                columns[at] = times;
                let batch = RecordBatch::try_new(batch.schema(), columns).unwrap();
                Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
            };
            attached.insert(AttachmentId(9), replacement);
            let run = HostRun::start(&sdm_host::default_root(), &message(Some(anchor)), attached)
                .unwrap();
            let result = terminal(&run).unwrap_or_else(|error| panic!("{:?}", decoded(&error)));
            let written = payload(&run, result);
            assert_eq!(
                values(&written, "events", "at"),
                values(&observed["sent"][0], "events", "at"),
            );
            let (state, owned) = checkpoint(&run);
            let bindings = manifest(&state, &owned);
            let event = bindings["tables"]
                .as_array()
                .unwrap()
                .iter()
                .find(|binding| binding["source"]["source"] == json!(["events"]))
                .unwrap();
            let id = AttachmentId(event["rows"].as_u64().unwrap() as u32);
            let Attachment::Arrow(data) = &owned[&id] else {
                panic!("original checkpoint event rows")
            };
            let original = data.batches[0].column_by_name("at").unwrap();
            let retained = match unit {
                TimeUnit::Microsecond => original
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .values()
                    .to_vec(),
                TimeUnit::Nanosecond => original
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .unwrap()
                    .values()
                    .to_vec(),
                _ => unreachable!(),
            };
            assert_eq!(retained, counts);
        }
    }
}
