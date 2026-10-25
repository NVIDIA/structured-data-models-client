// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The first frame materializer consumes owned Arrow and returns its
//! checkpoint even when a later public task check fails.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use sdm_contract::data::AttachmentId;
use sdm_contract::ops::{OperationResult, SamplerState};
use sdm_contract::{CONTRACT_VERSION, ControlMessage, Step};
use sdm_host::{ArrowData, Attachment, Attachments, HostRun, Outgoing};
use serde_json::{Value, json};

fn bytes(value: Value) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(&value).unwrap())
}

fn data(columns: &[(&str, Vec<i64>)]) -> Attachment {
    let batch = RecordBatch::try_from_iter(columns.iter().map(|(name, values)| {
        (*name, Arc::new(Int64Array::from(values.clone())) as ArrayRef)
    }))
    .unwrap();
    Attachment::Arrow(ArrowData::new(batch.schema(), vec![batch]))
}

fn column(name: &str, stype: &str) -> Value {
    json!({"name":name,"dtype":"int64","stype":stype,"stype_before_key":null})
}

fn message() -> Vec<u8> {
    serde_json::to_vec(&json!({
        "type":"start","contract":CONTRACT_VERSION,
        "facts":{"now":{"t":"ts","v":"1709251200000000000","unit":"ns","zone":"UTC"}},
        "operation":{"op":"materialize","provider":{"provider":"frames","manifest":7},
            "schema":1,"query":2,"indices":3,"anchor_time":null,"seed":"0x7",
            "values":5,"options":[]}
    }))
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
        (AttachmentId(2), bytes(json!("PREDICT users.age FOR EACH users.user_id"))),
        (AttachmentId(3), data(&[("index", vec![key])])),
        (AttachmentId(5), bytes(json!({"num_context_samples":1,"num_neighbors":[2],"num_hops":1,"random_seed":7}))),
        (AttachmentId(7), bytes(json!({"tables":[{"source":{"source":["users"]},"rows":8},{"source":{"source":["orders"]},"rows":9}]}))),
        (AttachmentId(8), data(&[("user_id",vec![1,2,3]),("age",vec![20,30,40])])),
        (AttachmentId(9), data(&[("order_id",(0..18).collect()),("user_id",(0..18).map(|n|n/6+1).collect()),("amount",(1..19).collect())])),
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
    let Attachment::Bytes(raw) = &manifest else { panic!("manifest bytes") };
    let manifest_value: Value = serde_json::from_slice(raw).unwrap();
    let metadata_id = AttachmentId(manifest_value["metadata"].as_u64().unwrap() as u32);
    let mut owned = Attachments::from([
        (state.memo, run.take_attachment(state.memo).unwrap().unwrap()),
        (manifest_id, manifest),
        (metadata_id, run.take_attachment(metadata_id).unwrap().unwrap()),
    ]);
    for binding in manifest_value["tables"].as_array().unwrap() {
        let id = AttachmentId(binding["rows"].as_u64().unwrap() as u32);
        assert!(matches!(run.take_attachment(id).unwrap().map(|a| {
            let arrow = matches!(&a, Attachment::Arrow(_));
            owned.insert(id,a);
            arrow
        }), Some(true)));
    }
    (written["state"].clone(), owned)
}

#[test]
fn frame_consumer_finishes_without_a_fetch_and_exports_owned_checkpoint() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs(3))
        .unwrap_or_else(|out| panic!("{:?}", decoded(&out)));
    let result = terminal(&run).unwrap_or_else(|out| panic!("{:?}", decoded(&out)));
    assert!(matches!(run.take_attachment(result.state.payload).unwrap(), Some(Attachment::Bytes(_))));
    let (_, owned) = checkpoint(&run);
    assert_eq!(owned.values().filter(|a| matches!(a,Attachment::Arrow(_))).count(),2);
    assert!(run.take_sampler_state().unwrap().is_none());
    drop(run);
    assert!(owned.values().filter_map(|a| match a { Attachment::Arrow(a)=>Some(a.batches.iter().map(RecordBatch::num_rows).sum::<usize>()), _=>None }).any(|n| n==18));
}

#[test]
fn unknown_entity_failure_preserves_the_built_snapshot_without_neighbor_draws() {
    let run = HostRun::start(&sdm_host::default_root(), &message(), inputs(999))
        .unwrap_or_else(|out| panic!("{:?}",decoded(&out)));
    let failed=terminal(&run).expect_err("unknown entity");
    assert_eq!(decoded(&failed)["error"]["kind"],"INVALID_REQUEST");
    let (state,owned)=checkpoint(&run);
    let manifest_id=AttachmentId(state["frame_checkpoint"]["manifest"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw)=&owned[&manifest_id] else { panic!("manifest") };
    let manifest:Value=serde_json::from_slice(raw).unwrap();
    let metadata_id=AttachmentId(manifest["metadata"].as_u64().unwrap() as u32);
    let Attachment::Bytes(raw)=&owned[&metadata_id] else { panic!("metadata") };
    let metadata:Value=serde_json::from_slice(raw).unwrap();
    assert_eq!(metadata["neighbors"]["position"],624);
    assert_eq!(metadata["neighbors"]["words"][0],123);
    assert!(!metadata.to_string().contains("attachment"));
}
