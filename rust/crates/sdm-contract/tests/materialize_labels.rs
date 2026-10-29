// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed materialization data products account for every actual owner.

use sdm_contract::data::AttachmentId;
use sdm_contract::frames::materialize::FramePreparation;
use sdm_contract::materialize::MaterializePreparation;
use sdm_contract::ops::{AttachmentKind, Materialize, Operation};
use serde_json::{Value, json};

fn population() -> Value {
    json!({"source":"source-1","identity":"population-1","owner":11,"rows":12000,"batch_rows":10000,"target":{"source":{"source":["buys"]},"column":"item_id"}})
}

fn source() -> Value {
    json!({"source":"source-1","identity":"source-meaning-1","owner":10,
        "target":{"source":{"source":["buys"]},"column":"item_id"},
        "tables":[{"source":{"source":["buys"]},"read":{"source":["buys"]},"columns":["buy_id","user_id","item_id","ts"]}]})
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
fn native_window_time_values_are_typed_arrow_not_an_opaque_or_policy_flag() {
    let input: MaterializePreparation = serde_json::from_value(json!({
        "stage":"target_column",
        "inputs":{"population":"population-1","range":{"start":0,"stop":3},
            "owner":10,"target":{"source":{"source":["buys"]},"column":"item_id"},
            "time":{"source":{"source":["buys"]},"column":"ts"},
            "form":"plain","condition_columns":[]},
        "converted":{"rows":11,"presence":12,"encoding":null,"times":13}
    }))
    .unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(10), AttachmentKind::LabelInputs),
            (AttachmentId(11), AttachmentKind::Arrow),
            (AttachmentId(12), AttachmentKind::Arrow),
            (AttachmentId(13), AttachmentKind::Arrow),
        ]
    );
    let mut malformed = serde_json::to_value(input).unwrap();
    malformed["converted"]["times"] = json!(true);
    assert!(serde_json::from_value::<MaterializePreparation>(malformed).is_err());
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

#[test]
fn actual_target_completion_declares_the_source_owner_values_and_full_presence_fact() {
    let value = json!({"stage":"target_column","inputs":{
        "population":"population-1","range":{"start":0,"stop":10000},"owner":21,
        "target":{"source":{"source":["buys"]},"column":"item_id"},"form":"plain","condition_columns":[]
    },"converted":{"rows":22,"presence":23,"encoding":null}});
    let input: FramePreparation = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(21), AttachmentKind::LabelInputs),
            (AttachmentId(22), AttachmentKind::Arrow),
            (AttachmentId(23), AttachmentKind::Arrow)
        ]
    );
    for field in ["accepted", "next_action", "callback"] {
        let mut malformed = value.clone();
        malformed["inputs"][field] = json!(true);
        assert!(serde_json::from_value::<FramePreparation>(malformed).is_err());
    }
}

#[test]
fn condition_history_names_actual_column_views_and_completed_value_owners() {
    let view = json!({"population":"population-1","range":{"start":0,"stop":2},"owner":21,
        "target":{"source":{"source":["users"]},"column":"age"},
        "form":"condition","condition_columns":[],"source_role":"entity"});
    let mut completed = view.clone();
    completed["owner"] = json!(22);
    let value = json!({"stage":"eligibility_column","inputs":{
        "population":"population-1","range":{"start":0,"stop":2},"owner":20,
        "target":{"source":{"source":["buys"]},"column":"item_id"},
        "form":"plain","condition_columns":[],
        "eligibility":{"columns":[view.clone()],"values":[]}
    },"column":{"input":completed,
        "owner":23,"derivation":"column","rows":2,"condition_values":[]}});
    let input: MaterializePreparation = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(20), AttachmentKind::LabelInputs),
            (AttachmentId(21), AttachmentKind::LabelInputs),
            (AttachmentId(22), AttachmentKind::LabelInputs),
            (AttachmentId(23), AttachmentKind::TargetColumn),
        ]
    );
    for field in ["next_action", "accepted", "callback"] {
        let mut bad = value.clone();
        bad["inputs"]["eligibility"][field] = json!(true);
        assert!(serde_json::from_value::<MaterializePreparation>(bad).is_err());
    }
}

#[test]
fn restored_group_data_and_late_condition_prefix_keep_distinct_typed_owners() {
    let groups = json!({"population":"population-1","range":{"start":0,"stop":10000},"owner":12,
        "native_target":{"source":{"source":["buys"]},"column":"item_id"},"target_encoding":null});
    let input:MaterializePreparation=serde_json::from_value(json!({"stage":"restore_label_groups","groups":groups,"restored":{"rows":13,"encoding":null}})).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(12), AttachmentKind::MaterializeGroups),
            (AttachmentId(13), AttachmentKind::Arrow)
        ]
    );
    let input:MaterializePreparation=serde_json::from_value(json!({"stage":"complete_label_batch","population":population(),
        "groups":{"restored":{"rows":14,"encoding":null},"what_if":{"columns":[],"values":[]},
            "population":"population-1","range":{"start":0,"stop":10000},"owner":12,
            "native_target":{"source":{"source":["buys"]},"column":"item_id"},"target_encoding":null},
        "restored":null})).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(11), AttachmentKind::MaterializePopulation),
            (AttachmentId(12), AttachmentKind::MaterializeGroups),
            (AttachmentId(14), AttachmentKind::Arrow)
        ]
    );
}

#[test]
fn selected_target_completion_keeps_the_actual_predecessor_and_never_walks_row_counts_as_ids() {
    let value = json!({"stage":"windowed_target","column":{
        "input":{"population":"population-1","range":{"start":0,"stop":10000},"owner":21,
            "target":{"source":{"source":["buys"]},"column":"item_id"},"form":"filtered","condition_columns":[]},
        "owner":24,"derivation":"filtered","rows":137,"condition_values":[]
    },"converted":{"rows":25,"presence":26,"encoding":"utf8_surrogatepass"}});
    let input: FramePreparation = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(21), AttachmentKind::LabelInputs),
            (AttachmentId(24), AttachmentKind::TargetColumn),
            (AttachmentId(25), AttachmentKind::Arrow),
            (AttachmentId(26), AttachmentKind::Arrow)
        ]
    );
    let mut extra = value;
    extra["column"]["origin"] = json!("native-library");
    assert!(serde_json::from_value::<FramePreparation>(extra).is_err());
}

#[test]
fn condition_completion_keeps_complete_actual_columns_for_repeated_occurrences() {
    let binding = json!({"source":{"source":["buys"]},"column":"amount"});
    let value = json!({"stage":"condition_column","column":{
        "input":{"population":"population-1","range":{"start":0,"stop":10000},"owner":21,
            "target":{"source":{"source":["buys"]},"column":"item_id"},"form":"filtered","condition_columns":[binding.clone(),binding.clone()]},
        "owner":24,"derivation":"column","rows":137,
        "condition_values":[{"column":binding.clone(),"converted":{"rows":25,"presence":26,"encoding":null}}]
    },"operand":{"column":binding,"converted":{"rows":27,"presence":28,"encoding":null}}});
    let input: FramePreparation = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(
        input.attachments(),
        vec![
            (AttachmentId(21), AttachmentKind::LabelInputs),
            (AttachmentId(24), AttachmentKind::TargetColumn),
            (AttachmentId(25), AttachmentKind::Arrow),
            (AttachmentId(26), AttachmentKind::Arrow),
            (AttachmentId(27), AttachmentKind::Arrow),
            (AttachmentId(28), AttachmentKind::Arrow)
        ]
    );
    let mut wrong = value;
    wrong["column"]["next_operand"] = json!(1);
    assert!(serde_json::from_value::<FramePreparation>(wrong).is_err());
}
