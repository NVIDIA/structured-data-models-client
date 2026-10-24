// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Catalog result sharing is a fact of the complete concrete read plan,
//! never of an answer schema alone.

use sdm_contract::Scalar;
use sdm_contract::data::{Dialect, ExpectedSchema, FetchPlan, SqlParam, SqlStatement};

#[test]
fn sharing_a_catalog_reply_requires_the_same_complete_concrete_read() {
    let statement = SqlStatement {
        dialect: Dialect::new("snowflake"),
        text: "DESCRIBE TABLE DB.SHOP.USERS".into(),
        params: vec![],
    };
    let plan = FetchPlan::Sql {
        statements: vec![statement.clone()],
        expect: ExpectedSchema::Discover {},
    };
    assert!(plan.same_catalog_read(&plan.clone()));
    let mut changed = statement.clone();
    changed.text = "DESCRIBE TABLE DB.SHOP.ORDERS".into();
    let mut parameterized = statement.clone();
    parameterized.params.push(SqlParam::Value {
        value: Scalar::Utf8("another table".into()),
    });
    let mut dialect = statement.clone();
    dialect.dialect = Dialect::new("databricks");
    for statements in [
        vec![changed],
        vec![parameterized],
        vec![dialect],
        vec![statement.clone(), statement.clone()],
    ] {
        let different = FetchPlan::Sql {
            statements,
            expect: ExpectedSchema::Discover {},
        };
        assert!(!plan.same_catalog_read(&different));
        assert!(!different.same_catalog_read(&plan));
    }
    assert!(!plan.same_catalog_read(&FetchPlan::Sql {
        statements: vec![statement],
        expect: ExpectedSchema::Schema { fields: vec![] },
    }));
    // Host plans carry only the expected schema, not the request being read.
    // Equal host plans therefore do not establish equal catalog answers.
    let host = FetchPlan::Host {
        expect: ExpectedSchema::Discover {},
    };
    assert_eq!(host, host.clone());
    assert!(!host.same_catalog_read(&host));
    assert!(!host.same_catalog_read(&plan));
    assert!(!plan.same_catalog_read(&host));
}
