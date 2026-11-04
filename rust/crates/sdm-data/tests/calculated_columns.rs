// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider SQL keeps physical expressions, aliases and row filters together.
use sdm_contract::data::{
    DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice, Relation, RelationColumn,
};
use sdm_data::Data;

#[test]
fn projections_preserve_filters_and_quoted_names_on_every_sql_provider() {
    for dialect in ["duckdb", "sqlite", "postgres", "databricks", "snowflake"] {
        let namespace = match dialect {
            "postgres" => vec![Some("public".into())],
            "databricks" | "snowflake" => vec![Some("catalog".into()), Some("schema".into())],
            _ => vec![],
        };
        let request = DataRequest::Preview {
            table: Relation {
                source: vec!["orders".into()],
                found: false,
                filter: Some("quantity > 2".into()),
                columns: vec![
                    RelationColumn {
                        name: "entity".into(),
                        column: Some("id".into()),
                        expression: None,
                    },
                    RelationColumn {
                        name: "line total".into(),
                        column: None,
                        expression: Some("price * quantity".into()),
                    },
                ],
            },
            columns: vec!["entity".into(), "line total".into()],
            limit: 10,
        };
        let choice = ProviderChoice::Sql {
            dialect: Dialect::new(dialect),
            namespace,
            view: None,
            timestamp_encoding: None,
        };
        let FetchPlan::Sql { statements, .. } = Data::new().plan(&request, &choice).unwrap() else {
            panic!("not SQL")
        };
        let sql = &statements.last().unwrap().text;
        assert!(sql.contains("price * quantity"), "{dialect}: {sql}");
        assert!(sql.contains("quantity > 2"), "{dialect}: {sql}");
        let quote = if dialect == "databricks" { '`' } else { '"' };
        assert!(
            sql.contains(&format!("AS {quote}entity{quote}")),
            "{dialect}: {sql}"
        );
        assert!(
            sql.contains(&format!("{quote}line total{quote}")),
            "{dialect}: {sql}"
        );
        if dialect == "postgres" {
            assert!(sql.contains("AS \"__sdm_projection\""));
        }
    }
}
