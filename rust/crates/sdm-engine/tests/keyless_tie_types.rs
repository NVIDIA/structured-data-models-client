// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::ColumnType;
use sdm_contract::data::{NullOrder, Relation};
use sdm_contract::ops::{DescribedColumn, Dtype, Stype};
use sdm_engine::{graph::Table, sample::tie_break};

#[test]
fn keyless_ties_preserve_types_without_reinterpreting_keyed_tables() {
    let column = |name: &str, dtype| DescribedColumn {
        name: name.into(),
        dtype,
        received: ColumnType::Utf8,
        stype: Stype::Categorical,
        stype_before_key: None,
        nullable: true,
        unique: false,
        declared_type: None,
        catalog: None,
        exact_number: None,
    };
    let mut table = Table {
        name: "notes".into(),
        source: Relation {
            source: vec!["notes".into()],
            columns: vec![],
            filter: None,
            found: false,
        },
        columns: vec![
            column("text", Dtype::String),
            column("number", Dtype::Int),
            column("at", Dtype::String),
            column("tags", Dtype::Stringlist),
        ],
        primary_key: None,
        time_column: Some("at".into()),
        end_time_column: None,
    };
    let terms = tie_break(&table, "number", NullOrder::Last).unwrap().0;
    assert_eq!(
        terms
            .iter()
            .map(|t| (t.column.as_str(), t.string_like))
            .collect::<Vec<_>>(),
        [("at", false), ("number", false), ("text", true)]
    );
    assert!(terms[0].as_timestamp && terms[0].descending);
    assert!(terms.iter().all(|t| t.nulls_last));
    table.primary_key = Some("text".into());
    let terms = tie_break(&table, "number", NullOrder::First).unwrap().0;
    assert_eq!(
        terms
            .iter()
            .map(|t| (t.column.as_str(), t.string_like))
            .collect::<Vec<_>>(),
        [("at", false), ("text", false)]
    );
    assert!(terms.iter().all(|t| !t.nulls_last));
}
