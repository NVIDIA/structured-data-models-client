// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.28's contract follow-up: a plan's attachments renamed by one method
//! beside `FetchPlan::key_sets`, so the core names a bound fetch's batches
//! by the run's ids without reading any plan's shape.

use std::collections::BTreeMap;

use sdm_contract::Scalar;
use sdm_contract::data::{
    AttachmentId, Dialect, ExpectedSchema, FetchPlan, KeySet, SqlParam, SqlStatement,
};

fn key_set(id: u32) -> SqlParam {
    SqlParam::KeySet {
        keys: KeySet {
            attachment: AttachmentId(id),
            columns: vec![],
        },
        relation: "keys".into(),
    }
}

fn text(id: u32) -> SqlParam {
    SqlParam::Text {
        attachment: AttachmentId(id),
    }
}

fn plan(params: Vec<Vec<SqlParam>>) -> FetchPlan {
    FetchPlan::Sql {
        result_conversions: Vec::new(),
        statements: params
            .into_iter()
            .map(|params| SqlStatement {
                dialect: Dialect::new("databricks"),
                text: "SELECT ?".into(),
                params,
            })
            .collect(),
        expect: ExpectedSchema::Discover {},
    }
}

#[test]
fn each_attachment_a_plan_binds_is_renamed_where_the_map_names_it() {
    let value = SqlParam::Value {
        value: Scalar::Int64(9),
    };
    let mut renamed = plan(vec![
        vec![text(1), value.clone()],
        vec![key_set(4), text(1), key_set(2)],
    ]);
    renamed.rename_attachments(&BTreeMap::from([
        (AttachmentId(1), AttachmentId(17)),
        (AttachmentId(4), AttachmentId(18)),
    ]));
    assert_eq!(
        renamed,
        plan(vec![
            vec![text(17), value],
            vec![key_set(18), text(17), key_set(2)],
        ])
    );
    assert_eq!(
        renamed.key_sets(),
        [17, 18, 17, 2].map(AttachmentId).to_vec()
    );
}

#[test]
fn a_host_plan_binds_nothing_to_rename() {
    let mut host = FetchPlan::Host {
        expect: ExpectedSchema::Discover {},
    };
    host.rename_attachments(&BTreeMap::from([(AttachmentId(1), AttachmentId(2))]));
    assert_eq!(
        host,
        FetchPlan::Host {
            expect: ExpectedSchema::Discover {}
        }
    );
}
