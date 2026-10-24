// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.27: a SQL source read through a semantic model is refused by name
//! until a dialect reads views, never read as if its logical tables were the
//! source's own.

use sdm_contract::data::{DataLayer, Dialect, ProviderChoice, ViewChoice};
use sdm_contract::{Detail, ErrorCode, Sentence};
use sdm_data::Data;

#[test]
fn a_sql_source_naming_a_view_is_refused_by_name() {
    for dialect in ["duckdb", "sqlite", "postgres", "databricks"] {
        let choice = ProviderChoice::Sql {
            timestamp_encoding: None,
            dialect: Dialect::new(dialect),
            namespace: Vec::new(),
            view: Some(ViewChoice {
                name: "sales".into(),
                trust_expressions: false,
            }),
        };
        let refused = Data::new().capabilities(&choice).unwrap_err();
        assert_eq!(refused.sentence, Sentence::ViewsUnread, "{dialect}");
        assert_eq!(refused.code, ErrorCode::UnsupportedRequest);
        assert_eq!(refused.fields["rule"], Detail::text("view"));
    }
}
