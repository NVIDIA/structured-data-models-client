// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first frame materializer consumes owned Arrow and returns its
//! checkpoint even when a later public task check fails.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult, SamplerState};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Step, TimestampUnit};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing};
use serde_json::{Value, json};

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn data(columns: &[(&str, Vec<i64>)]) -> Attachment {
    let batch = RecordBatch::try_from_iter(columns.iter().map(|(name, values)| {
        (
            *name,
            Arc::new(Int64Array::from(values.clone())) as ArrayRef,
        )
    }))
    .unwrap();
    Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
}

fn column(name: &str, stype: &str) -> Value {
    json!({"name":name,"dtype":"int64","stype":stype,"stype_before_key":null})
}

fn message() -> Vec<u8> {
    let operation: Operation = serde_json::from_value(
        json!({"op":"materialize","provider":{"provider":"frames","manifest":7},
            "schema":1,"query":2,"indices":3,"anchor_time":null,"seed":"0x7",
            "values":5,"options":[]}),
    )
    .unwrap();
    sdm_host::encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                value: 1_709_251_200_000_000,
                unit: TimestampUnit::Us,
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

fn inputs(key: i64) -> Attachments {
    let schema = json!({"tables":[
        {"name":"users","source":{"source":["users"]},"columns":[column("user_id","ID"),column("age","numerical")],
         "primary_key":["user_id"],"frame_key_form":"scalar","time_column":null},
        {"name":"orders","source":{"source":["orders"]},"columns":[column("order_id","ID"),column("user_id","ID"),column("amount","numerical")],
         "primary_key":["order_id"],"frame_key_form":"scalar","time_column":null}],
        "links":[[{"table":"orders","column":"user_id"},{"table":"users","column":"user_id"}]]});
    Attachments::from([
        (AttachmentId(1), bytes(schema)),
        (
            AttachmentId(2),
            bytes(json!("PREDICT users.age FOR EACH users.user_id")),
        ),
        (AttachmentId(3), data(&[("index", vec![key])])),
        (
            AttachmentId(5),
            bytes(
                json!({"num_context_samples":1,"num_neighbors":[2],"num_hops":1,"random_seed":7}),
            ),
        ),
        (
            AttachmentId(7),
            bytes(
                json!({"tables":[{"source":{"source":["users"]},"rows":8},{"source":{"source":["orders"]},"rows":9}]}),
            ),
        ),
        (
            AttachmentId(8),
            data(&[("user_id", vec![1, 2, 3]), ("age", vec![20, 30, 40])]),
        ),
        (
            AttachmentId(9),
            data(&[
                ("order_id", (0..18).collect()),
                ("user_id", (0..18).map(|n| n / 6 + 1).collect()),
                ("amount", (1..19).collect()),
            ]),
        ),
    ])
}

fn decoded(out: &Outgoing) -> Value {
    serde_json::to_value(sdm_host::read(out).unwrap()).unwrap()
}

fn terminal(run: &HostRun) -> Result<sdm_contract::ops::Materialized, Outgoing> {
    loop {
        let out = run.next()?;
        match sdm_host::read(&out).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Materialize(result),
                ..
            }) => return Ok(result),
            ControlMessage::Warnings { .. } => {}
            ControlMessage::Step(Step::Warnings) => {
                let _ = run.take_warnings().unwrap();
            }
            other => panic!("frame materialization must not fetch or call a model: {other:?}"),
        }
    }
}

fn checkpoint(run: &HostRun) -> (Value, Attachments) {
    let out = run.take_sampler_state().unwrap().expect("frame checkpoint");
    let written = decoded(&out);
    let state: SamplerState = serde_json::from_value(written["state"].clone()).unwrap();
    let frame = written["state"]["frame_checkpoint"].clone();
    assert_eq!(frame["version"], 1);
    let manifest_id = AttachmentId(frame["manifest"].as_u64().unwrap() as u32);
    let manifest = run.take_attachment(manifest_id).unwrap().unwrap();
    let Attachment::Bytes(raw) = &manifest else {
        panic!("manifest bytes")
    };
    let manifest_value: Value = serde_json::from_slice(raw).unwrap();
    let metadata_id = AttachmentId(manifest_value["metadata"].as_u64().unwrap() as u32);
    let mut owned = Attachments::from([
        (
            state.memo,
            run.take_attachment(state.memo).unwrap().unwrap(),
        ),
        (manifest_id, manifest),
        (
            metadata_id,
            run.take_attachment(metadata_id).unwrap().unwrap(),
        ),
    ]);
    for binding in manifest_value["tables"].as_array().unwrap() {
        let id = AttachmentId(binding["rows"].as_u64().unwrap() as u32);
        assert!(matches!(
            run.take_attachment(id).unwrap().map(|a| {
                let arrow = matches!(&a, Attachment::Arrow(_));
                owned.insert(id, a);
                arrow
            }),
            Some(true)
        ));
    }
    (written["state"].clone(), owned)
}

#[test]
fn frame_consumer_finishes_without_a_fetch_and_exports_owned_checkpoint() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs(3))
        .unwrap_or_else(|out| panic!("{:?}", decoded(&out)));
    let result = terminal(&run).unwrap_or_else(|out| panic!("{:?}", decoded(&out)));
    assert!(matches!(
        run.take_attachment(result.state.payload).unwrap(),
        Some(Attachment::Bytes(_))
    ));
    let (_, owned) = checkpoint(&run);
    assert_eq!(
        owned
            .values()
            .filter(|a| matches!(a, Attachment::Arrow(_)))
            .count(),
        2
    );
    assert!(run.take_sampler_state().unwrap().is_none());
    drop(run);
    assert!(
        owned
            .values()
            .filter_map(|a| match a {
                Attachment::Arrow(a) =>
                    Some(a.batches.iter().map(RecordBatch::num_rows).sum::<usize>()),
                _ => None,
            })
            .any(|n| n == 18)
    );
}

#[test]
fn unknown_entity_failure_preserves_the_built_snapshot_without_neighbor_draws() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs(999))
        .unwrap_or_else(|out| panic!("{:?}", decoded(&out)));
    let failed = terminal(&run).expect_err("unknown entity");
    assert_eq!(decoded(&failed)["error"]["code"], "INVALID_REQUEST");
    let (state, owned) = checkpoint(&run);
    let manifest_id = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&manifest_id] else {
        panic!("manifest")
    };
    let manifest: Value = serde_json::from_slice(raw).unwrap();
    let metadata_id = AttachmentId(manifest["metadata"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&metadata_id] else {
        panic!("metadata")
    };
    let metadata: Value = serde_json::from_slice(raw).unwrap();
    assert_eq!(metadata["neighbors"]["position"], 624);
    assert_eq!(metadata["neighbors"]["words"][0], 123);
    assert!(!metadata.to_string().contains("attachment"));
}

fn metadata(state: &Value, owned: &Attachments) -> Value {
    let manifest_id = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&manifest_id] else {
        panic!("manifest bytes")
    };
    let manifest: Value = serde_json::from_slice(raw).unwrap();
    let id = AttachmentId(manifest["metadata"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&id] else {
        panic!("metadata bytes")
    };
    serde_json::from_slice(raw).unwrap()
}

fn rebound(state: &Value, owned: &Attachments, version: u32) -> (Vec<u8>, Attachments) {
    let old_manifest = AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw) = &owned[&old_manifest] else {
        panic!("manifest bytes")
    };
    let manifest: Value = serde_json::from_slice(raw).unwrap();
    let old_metadata = AttachmentId(manifest["metadata"].as_u64().unwrap() as u32);
    let mut attached = inputs(3);
    attached.remove(&AttachmentId(8));
    attached.remove(&AttachmentId(9));
    attached.insert(AttachmentId(7), bytes(json!({"tables":[]})));
    attached.insert(
        AttachmentId(20),
        owned[&AttachmentId(state["memo"].as_u64().unwrap() as u32)].clone(),
    );
    attached.insert(AttachmentId(21), owned[&old_metadata].clone());
    let mut bindings = Vec::new();
    for (number, table) in manifest["tables"].as_array().unwrap().iter().enumerate() {
        let previous = AttachmentId(table["rows"].as_u64().unwrap() as u32);
        let next = AttachmentId(22 + number as u32);
        attached.insert(next, owned[&previous].clone());
        bindings.push(json!({"source":table["source"],"rows":next.0}));
    }
    attached.insert(
        AttachmentId(30),
        bytes(json!({"metadata":21,"tables":bindings})),
    );
    let mut start: Value = serde_json::from_slice(&message()).unwrap();
    start["operation"]["sampler_state"] = json!({"version":state["version"],"memo":20,"frame_checkpoint":{"version":version,"manifest":30}});
    (serde_json::to_vec(&start).unwrap(), attached)
}

#[test]
fn checkpoint_rebinding_uses_logical_snapshot_identity_not_attachment_ids() {
    let first = HostRun::start(&sdm_host::default_root(), &message(), inputs(3)).unwrap();
    terminal(&first).unwrap();
    let (state, owned) = checkpoint(&first);
    let original = metadata(&state, &owned);
    let (message, attached) = rebound(&state, &owned, 1);
    let next = HostRun::start(&sdm_host::default_root(), &message, attached).unwrap();
    terminal(&next).unwrap();
    let (state, owned) = checkpoint(&next);
    let after = metadata(&state, &owned);
    assert_eq!(original["source"], after["source"]);
    assert_eq!(original["neighbors"], after["neighbors"]);
}

#[test]
fn unknown_frame_checkpoint_version_is_rejected_and_dropped() {
    let first = HostRun::start(&sdm_host::default_root(), &message(), inputs(3)).unwrap();
    terminal(&first).unwrap();
    let (state, owned) = checkpoint(&first);
    let (message, attached) = rebound(&state, &owned, 99);
    let refused = HostRun::start(&sdm_host::default_root(), &message, attached)
        .err()
        .expect("unknown checkpoint version");
    let refused = decoded(&refused);
    assert_eq!(
        refused["error"]["fields"]["field"],
        json!({"t":"utf8","v":"sampler_state.frame_checkpoint.version"})
    );
    assert_eq!(refused["error"]["drop_held"], json!(["sampler_state"]));
}

#[test]
fn exported_arrow_owner_survives_run_close_and_releases_with_checkpoint() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs(3)).unwrap();
    terminal(&run).unwrap();
    let (_, owned) = checkpoint(&run);
    let arrow = owned
        .values()
        .find_map(|value| match value {
            Attachment::Arrow(data) => Some(data),
            _ => None,
        })
        .unwrap();
    let weak = Arc::downgrade(arrow.batches[0].column(0));
    drop(run);
    assert!(weak.upgrade().is_some());
    drop(owned);
    assert!(weak.upgrade().is_none());
}

#[test]
fn cancellation_before_graph_construction_exports_no_snapshot() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs(3)).unwrap();
    run.cancel().unwrap();
    assert!(run.take_sampler_state().unwrap().is_none());
    assert!(run.next().is_err());
}

#[test]
fn a_cap_failure_after_frame_walk_keeps_the_consumed_stream() {
    let mut attached = inputs(3);
    let Attachment::Bytes(raw) = &attached[&AttachmentId(1)] else {
        panic!("schema")
    };
    let mut schema: Value = serde_json::from_slice(raw).unwrap();
    let Attachment::Bytes(raw) = &attached[&AttachmentId(7)] else {
        panic!("manifest")
    };
    let mut manifest: Value = serde_json::from_slice(raw).unwrap();
    for ordinal in 0..14 {
        let name = format!("child{ordinal}");
        let mut table = schema["tables"][1].clone();
        table["name"] = json!(name);
        table["source"] = json!({"source":[name]});
        schema["tables"].as_array_mut().unwrap().push(table);
        schema["links"]
            .as_array_mut()
            .unwrap()
            .push(json!([{"table":name,"column":"user_id"},{"table":"users","column":"user_id"}]));
        let id = AttachmentId(10 + ordinal);
        attached.insert(id, attached[&AttachmentId(9)].clone());
        manifest["tables"]
            .as_array_mut()
            .unwrap()
            .push(json!({"source":{"source":[name]},"rows":id.0}));
    }
    attached.insert(AttachmentId(1), bytes(schema));
    attached.insert(AttachmentId(7), bytes(manifest));
    let run = HostRun::start(&sdm_host::default_root(), &message(), attached).unwrap();
    let failed = terminal(&run).expect_err("sixteen sampled tables");
    assert_eq!(
        decoded(&failed)["error"]["sentence"],
        "subgraph_past_tables"
    );
    let (state, owned) = checkpoint(&run);
    assert_ne!(metadata(&state, &owned)["neighbors"]["position"], 624);
}

#[test]
fn native_count_dtype_failure_keeps_the_reset_but_unconsumed_stream() {
    let mut attached = inputs(3);
    attached.insert(
        AttachmentId(5),
        bytes(json!({
            "num_context_samples": 1, "num_neighbors": [3.0],
            "num_hops": 1, "random_seed": 7,
        })),
    );
    let run = HostRun::start(&sdm_host::default_root(), &message(), attached).unwrap();
    let failed = terminal(&run).expect_err("the actual native sampler refuses float counts");
    assert_eq!(decoded(&failed)["error"]["code"], "INTERNAL_ERROR");
    assert_eq!(
        decoded(&failed)["error"]["sentence"],
        "frame_neighbor_dtype"
    );
    let (state, owned) = checkpoint(&run);
    let state = metadata(&state, &owned);
    assert_eq!(state["neighbors"]["position"], 624);
    assert_eq!(state["neighbors"]["words"][0], 7);
}

#[test]
fn unseeded_entity_draw_accepts_the_full_host_entropy_domain() {
    let mut attached = inputs(3);
    attached.insert(
        AttachmentId(8),
        data(&[
            ("user_id", (1..=100).collect()),
            ("age", (20..120).collect()),
        ]),
    );
    attached.insert(
        AttachmentId(5),
        bytes(json!({
            "num_context_samples": 1, "num_neighbors": [2],
            "num_hops": 1, "random_seed": null,
        })),
    );
    let Attachment::Bytes(raw) = &attached[&AttachmentId(7)] else {
        panic!("manifest")
    };
    let mut manifest: Value = serde_json::from_slice(raw).unwrap();
    manifest["entropy"] = json!(u64::MAX);
    attached.insert(AttachmentId(7), bytes(manifest));
    let mut start: Value = serde_json::from_slice(&message()).unwrap();
    start["operation"]["seed"] = Value::Null;
    start["operation"]["unseeded"] = json!(true);
    let run = HostRun::start(
        &sdm_host::default_root(),
        &serde_json::to_vec(&start).unwrap(),
        attached,
    )
    .unwrap();
    terminal(&run).unwrap_or_else(|out| panic!("full-width OS entropy: {:?}", decoded(&out)));
}

#[test]
fn resumed_early_failure_and_pre_next_cancel_retain_the_incoming_checkpoint() {
    for cancellation in [false, true] {
        let first = HostRun::start(&sdm_host::default_root(), &message(), inputs(3)).unwrap();
        terminal(&first).unwrap();
        let (state, owned) = checkpoint(&first);
        let before = metadata(&state, &owned);
        let (message, mut attached) = rebound(&state, &owned, 1);
        attached.insert(AttachmentId(5), bytes(json!({"num_context_samples":0})));
        let resumed = HostRun::start(&sdm_host::default_root(), &message, attached).unwrap();
        if cancellation {
            resumed.cancel().unwrap();
        } else {
            terminal(&resumed).expect_err("early context-size refusal");
        }
        let (state, owned) = checkpoint(&resumed);
        assert_eq!(metadata(&state, &owned), before);
        let (message, mut attached) = rebound(&state, &owned, 1);
        let mut start: Value = serde_json::from_slice(&message).unwrap();
        start["operation"]["seed"] = Value::Null;
        start["operation"]["unseeded"] = json!(true);
        attached.insert(
            AttachmentId(7),
            bytes(json!({"tables": [], "entropy": 123456789_u64})),
        );
        attached.insert(
            AttachmentId(5),
            bytes(json!({"num_context_samples":1,"num_neighbors":[2],"random_seed":null})),
        );
        let next = HostRun::start(
            &sdm_host::default_root(),
            &serde_json::to_vec(&start).unwrap(),
            attached,
        )
        .unwrap();
        terminal(&next).unwrap();
        let (state, owned) = checkpoint(&next);
        let after = metadata(&state, &owned);
        assert_eq!(after["source"], before["source"]);
        assert_ne!(after["neighbors"], before["neighbors"]);
    }
}

#[test]
fn sanitization_warning_boundary_can_cancel_before_any_neighbor_draw() {
    let mut attached = inputs(3);
    attached.insert(
        AttachmentId(8),
        data(&[
            ("user_id", vec![1, 1, 2, 3]),
            ("age", vec![20, 999, 30, 40]),
        ]),
    );
    let run = HostRun::start(&sdm_host::default_root(), &message(), attached).unwrap();
    assert!(matches!(
        sdm_host::read(&run.next().unwrap()).unwrap(),
        ControlMessage::Step(Step::Warnings)
    ));
    let warning = run
        .take_warnings()
        .unwrap()
        .expect("real sanitization warning");
    let ControlMessage::Warnings { warnings } = sdm_host::read(&warning).unwrap() else {
        panic!("warnings")
    };
    assert_eq!(warnings.len(), 1);
    assert_eq!(
        warnings[0].sentence,
        sdm_contract::Sentence::FrameRowsDropped
    );
    assert_eq!(
        warnings[0].fields["detail"],
        sdm_contract::Detail::text("[users: 1 of 4 rows dropped (1 with a repeated primary key)]")
    );
    run.cancel().unwrap();
    let (state, owned) = checkpoint(&run);
    let metadata = metadata(&state, &owned);
    assert_eq!(metadata["neighbors"]["position"], 624);
    assert_eq!(metadata["neighbors"]["words"][0], 123);
}

#[test]
fn boolean_key_capability_is_refused_before_a_frame_run_is_created() {
    use arrow_array::BooleanArray;
    let mut attached = inputs(3);
    let Attachment::Bytes(raw) = &attached[&AttachmentId(1)] else {
        panic!("schema")
    };
    let mut schema: Value = serde_json::from_slice(raw).unwrap();
    schema["tables"][0]["columns"][0]["dtype"] = json!("bool");
    attached.insert(AttachmentId(1), bytes(schema));
    let batch = RecordBatch::try_from_iter([
        (
            "user_id",
            Arc::new(BooleanArray::from(vec![false, true])) as ArrayRef,
        ),
        ("age", Arc::new(Int64Array::from(vec![20, 30])) as ArrayRef),
    ])
    .unwrap();
    attached.insert(
        AttachmentId(8),
        Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch])),
    );
    let failed = HostRun::start(&sdm_host::default_root(), &message(), attached)
        .err()
        .expect("unsupported numeric-key capability");
    assert_eq!(decoded(&failed)["error"]["code"], "UNSUPPORTED_REQUEST");
    assert_eq!(
        decoded(&failed)["error"]["fields"]["rule"],
        json!({"t":"utf8","v":"frame_boolean_key"})
    );
}
