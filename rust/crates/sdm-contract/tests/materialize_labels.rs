// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed materialization data products account for every actual owner.

use sdm_contract::data::AttachmentId;
use sdm_contract::materialize::MaterializePreparation;
use sdm_contract::ops::{AttachmentKind, Materialize, Operation};
use serde_json::{Value, json};

fn population() -> Value {
    json!({"source":"source-1","identity":"population-1","owner":11,"rows":12000,"batch_rows":10000,"target":{"source":{"source":["buys"]},"column":"item_id"}})
}

fn source() -> Value {
    json!({"source":"source-1","identity":"source-meaning-1","owner":10,
        "target":{"source":{"source":["buys"]},"column":"item_id"},
        "tables":[{"source":{"source":["buys"]},"columns":["buy_id","user_id","item_id","ts"]}]})
}

#[test]
fn task_input_reuses_the_real_materialize_attachment_contract() {
    let request: Materialize = serde_json::from_value(json!({
        "provider":{"provider":"sql","dialect":"duckdb"},"schema":1,"query":2,
        "indices":3,"seed":"0x7","anchor_time":null,"values":5,"options":[]
    }))
    .unwrap();
    let expected = Operation::Materialize(request.clone()).attachments();
    let input = MaterializePreparation::Task { request };
    assert_eq!(input.attachments(), expected);
}

#[test]
fn actual_groups_completion_names_typed_population_groups_and_arrow_rows() {
    let value = json!({"stage":"complete_label_batch","population":population(),
        "groups":{"population":"population-1","range":{"start":0,"stop":10000},"owner":12,"native_target":{"source":{"source":["buys"]},"column":"item_id"},"target_encoding":null},
        "restored":{"rows":13,"encoding":null}});
    let input: MaterializePreparation = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(11), AttachmentKind::MaterializePopulation),
            (AttachmentId(12), AttachmentKind::MaterializeGroups),
            (AttachmentId(13), AttachmentKind::Arrow),
        ]
    );
    for field in ["complete", "accepted_rows", "next_action", "callback"] {
        let mut malformed = value.clone();
        malformed["groups"][field] = json!(true);
        assert!(serde_json::from_value::<MaterializePreparation>(malformed).is_err());
    }
}

#[test]
fn completed_chain_owns_only_actual_batches_and_does_not_reinterpret_counts_as_ids() {
    let input: MaterializePreparation = serde_json::from_value(json!({
        "stage":"request","population":population(),"batches":[
            {"population":"population-1","range":{"start":0,"stop":10000},"owner":12,"complete":false},
            {"population":"population-1","range":{"start":10000,"stop":12000},"owner":13,"complete":true}
        ]
    })).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(11), AttachmentKind::MaterializePopulation),
            (AttachmentId(12), AttachmentKind::MaterializeBatch),
            (AttachmentId(13), AttachmentKind::MaterializeBatch),
        ]
    );
}

#[test]
fn source_conversion_facts_are_closed_data_and_keep_full_original_decimal_metadata() {
    for (bits, scale) in [(32, -129), (64, 129), (128, 0), (256, i32::MAX)] {
        let value = json!({"stage":"source_column","source":source(),"column":{"source":{"source":["buys"]},"column":"item_id"},"input":{
            "storage":"typed_columns","declaration":{"declaration":"decimal","native":{"type":format!("decimal{bits}"),"precision":7,"scale":scale}}
        }});
        let input: MaterializePreparation = serde_json::from_value(value.clone()).unwrap();
        let roundtrip = serde_json::to_value(input).unwrap();
        assert_eq!(roundtrip["input"], value["input"]);
        let mut extra = value;
        extra["input"]["provider"] = json!("pretend-native");
        assert!(serde_json::from_value::<MaterializePreparation>(extra).is_err());
    }
}
