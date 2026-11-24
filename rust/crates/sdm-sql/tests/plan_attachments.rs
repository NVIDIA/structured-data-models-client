// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A SQL plan's attachments renamed by one method beside
//! `SqlPlan::key_sets`, so a bound fetch's batches are named by the ids the
//! kit hands its host.

use std::collections::BTreeMap;

use sdm_contract::Scalar;
use sdm_contract::data::{AttachmentId, ExpectedSchema, KeySet};
use sdm_sql::plan::{Dialect, SqlParam, SqlPlan, SqlStatement};

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

fn plan(params: Vec<Vec<SqlParam>>) -> SqlPlan {
    SqlPlan {
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
