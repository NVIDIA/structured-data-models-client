// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KO3-44 bounds retain each native SQL language's original placement.
//! These unit requests do not add warehouse acceptance cases or fixtures.

use sdm_contract::data::{
    DataLayer, DataRequest, Dialect, FetchPlan, NeighborLimit, ProviderChoice,
};
use sdm_data::Data;

fn rendered(dialect: &str, limit: NeighborLimit) -> String {
    let table = match dialect {
        "databricks" | "snowflake" => vec!["c", "s", "orders"],
        "postgres" => vec!["public", "orders"],
        _ => vec!["orders"],
    };
    let mut request: DataRequest = serde_json::from_value(serde_json::json!({
        "request": "neighbors", "table": {"source": table},
        "key": ["user_id"], "columns": ["order_id", "user_id"],
        "seeds": {"attachment": 1, "columns": [
            {"name": "key", "nullable": true, "type": {"type": "int64"}},
            {"name": "batch", "nullable": false, "type": {"type": "int64"}}
        ]},
        "window": null, "per_seed": 32,
        "order": [{"column": "order_id", "as_timestamp": false,
                   "descending": false, "nulls_last": false}]
    }))
    .unwrap();
    let DataRequest::Neighbors { per_seed, .. } = &mut request else {
        unreachable!()
    };
    *per_seed = Some(limit);
    let provider = ProviderChoice::Sql {
        dialect: Dialect::new(dialect),
        namespace: vec![],
        view: None,
        timestamp_encoding: None,
    };
    let FetchPlan::Sql { statements, .. } = Data::new().plan(&request, &provider).unwrap() else {
        panic!("not SQL")
    };
    statements
        .into_iter()
        .map(|statement| statement.text)
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn signed_bounds_and_expressions_retain_native_precedence_and_unsigned_width() {
    for dialect in ["duckdb", "sqlite", "postgres", "databricks", "snowflake"] {
        let placement = if matches!(dialect, "sqlite" | "postgres") {
            "LIMIT "
        } else {
            ") <= "
        };
        for (limit, literal) in [
            (NeighborLimit::Negative { value: sdm_contract::data::NegativeNeighborBound::new("-1".into()).unwrap() }, "-1"),
            (
                NeighborLimit::Negative {
                    value: sdm_contract::data::NegativeNeighborBound::new("-0.5".into()).unwrap(),
                },
                "-0.5",
            ),
            (
                NeighborLimit::Expression {
                    text: "3 OR TRUE".into(),
                },
                "3 OR TRUE",
            ),
            (NeighborLimit::Unsigned(u64::MAX), "18446744073709551615"),
        ] {
            let text = rendered(dialect, limit);
            assert!(
                text.contains(&format!("{placement}{literal}\n")),
                "{dialect}: {text}"
            );
            assert!(
                !text.contains("(3 OR TRUE)"),
                "parenthesizing changes predicate precedence"
            );
        }
    }
}
