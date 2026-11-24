// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A built-in source as a host names it to its kit, and a fetch's
//! statements as the kit hands them to the host's driver, read and written
//! as JSON.

use sdm_contract::Scalar;
use sdm_contract::data::{AttachmentId, ExpectedSchema, KeySet};
use sdm_sql::plan::{Dialect, SqlChoice, SqlParam, SqlPlan, SqlStatement};

/// A source's namespace: left out, it reads as none and is written back
/// left out; each part a name or `null`, the source's current one (a
/// Databricks source given its catalog and not its schema); anything else
/// refused.
#[test]
fn a_source_namespace_takes_names_and_null_parts() {
    for (text, namespace) in [
        (r#"{"provider":"sql","dialect":"postgres"}"#, vec![]),
        (
            r#"{"provider":"sql","dialect":"postgres","namespace":["sales"]}"#,
            vec![Some("sales".to_owned())],
        ),
        (
            r#"{"provider":"sql","dialect":"databricks","namespace":["main",null]}"#,
            vec![Some("main".to_owned()), None],
        ),
    ] {
        let read: SqlChoice = serde_json::from_str(text).unwrap();
        assert_eq!(read.namespace, namespace, "{text}");
        let written: serde_json::Value = serde_json::to_value(&read).unwrap();
        let given: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(written, given, "{text}");
    }
    for refused in [
        r#"{"provider":"sql","dialect":"postgres","namespace":"sales"}"#,
        r#"{"provider":"sql","dialect":"postgres","namespace":[1]}"#,
        r#"{"provider":"host","dialect":"postgres"}"#,
        r#"{"dialect":"postgres"}"#,
    ] {
        assert!(
            serde_json::from_str::<SqlChoice>(refused).is_err(),
            "{refused}"
        );
    }
}

/// A fetch's statements as a driver reads them, tagged `sql`, round-trip.
#[test]
fn a_plan_round_trips_as_the_driver_reads_it() {
    let plan = SqlPlan {
        statements: vec![SqlStatement {
            dialect: Dialect::new("duckdb"),
            text: "SELECT id FROM orders WHERE id IN (SELECT id FROM keys) LIMIT ?".into(),
            params: vec![
                SqlParam::KeySet {
                    keys: KeySet {
                        attachment: AttachmentId(4),
                        columns: Vec::new(),
                    },
                    relation: "keys".into(),
                },
                SqlParam::Value {
                    value: Scalar::UInt64(u64::MAX),
                },
            ],
        }],
        expect: ExpectedSchema::Discover {},
        result_conversions: Vec::new(),
    };
    let text = serde_json::to_value(&plan).unwrap();
    assert_eq!(text["plan"], "sql");
    assert_eq!(text["statements"][0]["params"][0]["param"], "key_set");
    assert_eq!(serde_json::from_value::<SqlPlan>(text).unwrap(), plan);
    assert_eq!(plan.key_sets(), vec![AttachmentId(4)]);
}

/// A key set's JSON bound as a one-cell text attachment names it, and
/// the plan lists it with the key sets it binds, in order.
#[test]
fn a_text_parameter_names_its_attachment_and_the_plan_lists_it() {
    let param: SqlParam =
        serde_json::from_value(serde_json::json!({"param": "text", "attachment": 7})).unwrap();
    assert_eq!(
        param,
        SqlParam::Text {
            attachment: AttachmentId(7)
        }
    );
    assert_eq!(
        serde_json::to_value(&param).unwrap(),
        serde_json::json!({"param": "text", "attachment": 7})
    );
    let plan = SqlPlan {
        result_conversions: Vec::new(),
        statements: vec![SqlStatement {
            dialect: Dialect::new("postgres"),
            text: "SELECT 1".into(),
            params: vec![
                SqlParam::KeySet {
                    keys: KeySet {
                        attachment: AttachmentId(3),
                        columns: vec![],
                    },
                    relation: "keys".into(),
                },
                param,
            ],
        }],
        expect: ExpectedSchema::Discover {},
    };
    assert_eq!(plan.key_sets(), vec![AttachmentId(3), AttachmentId(7)]);
}
