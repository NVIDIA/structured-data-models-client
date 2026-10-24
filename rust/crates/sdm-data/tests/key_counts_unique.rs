// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A key the source keeps unique (`TableKey::unique`, contract v28) is
//! counted on every dialect as the reference's `_key_count_sql` counts one,
//! the rows less those missing part of it, and any other key by its
//! distinct values, as before (carried row P2-T2.2-3).

use sdm_contract::data::{DataLayer, DataRequest, Dialect, FetchPlan, ProviderChoice, TableKey};
use sdm_data::Data;

fn statement(dialect: &str, unique: bool) -> String {
    let request = DataRequest::KeyCounts {
        keys: vec![TableKey {
            table: "users".into(),
            columns: vec!["user_id".into()],
            label: None,
            unique,
        }],
    };
    // Postgres reads every table under a schema (T3.7).
    let namespace = if dialect == "postgres" {
        vec![Some("public".to_owned())]
    } else {
        Vec::new()
    };
    let choice = ProviderChoice::Sql {
        namespace,
        dialect: Dialect::new(dialect),
        view: None,
    };
    let Ok(FetchPlan::Sql { statements, .. }) = Data::new().plan(&request, &choice) else {
        panic!("{dialect}: key_counts is not planned as SQL");
    };
    statements.last().expect("a statement").text.clone()
}

#[test]
fn a_unique_key_is_counted_as_its_rows_less_its_nulls_on_every_dialect() {
    for (dialect, quote) in [
        ("sqlite", '"'),
        ("duckdb", '"'),
        ("postgres", '"'),
        ("databricks", '`'),
    ] {
        let key = format!("{quote}user_id{quote}");
        let nulls = format!("COUNT(CASE WHEN {key} IS NULL THEN 1 END)");
        let unique = statement(dialect, true);
        assert!(
            unique.contains(&format!("COUNT(*) - {nulls} AS d, {nulls} AS z")),
            "{dialect}: {unique}"
        );
        let other = statement(dialect, false);
        assert!(
            other.contains(&format!("COUNT(DISTINCT {key}) AS d, {nulls} AS z")),
            "{dialect}: {other}"
        );
    }
}
