// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Warehouse and Spark metric-view descriptions, with explicit expected facts.
//! Preserve section boundaries, duplicate handling, location and typed refusals.

use sdm_data::view_expr::describe::{self, DescribeRefusal};

/// Python's white space (`str.isspace`), which the core hands in from
/// `sdm-pql`: the 29 characters Python 3.12 counts, with the class the
/// patterns are compiled with derived from them.
fn spaces() -> &'static Spaces {
    static SPACES: std::sync::LazyLock<Spaces> = std::sync::LazyLock::new(|| {
        Spaces::new(|c| {
            matches!(
                c,
                '\t'..='\r'
                    | '\u{1c}'..='\u{20}'
                    | '\u{85}'
                    | '\u{a0}'
                    | '\u{1680}'
                    | '\u{2000}'..='\u{200a}'
                    | '\u{2028}'
                    | '\u{2029}'
                    | '\u{202f}'
                    | '\u{205f}'
                    | '\u{3000}'
            )
        })
    });
    &SPACES
}

use sdm_data::view_expr::Spaces;

#[test]
fn warehouse_and_spark_descriptions_preserve_columns_parameters_and_location() {
    for width in [3, 4] {
        for blank in [None, Some(String::new())] {
            let mut rows = vec![
                row("# col_name", ""),
                row("id", "bigint"),
                row("total", "double measure"),
                row("category", "string"),
                row("id", "int"),
                vec![blank.clone(); width],
                row("ignored", "not a column"),
                row("# Parameters", ""),
                row("# col_name", ""),
                row("start", "date"),
                row("limit", "int"),
                row("# Detailed Table Information", ""),
                row("Type", "METRIC_VIEW"),
                row("View Text", "source: orders"),
                row("Catalog", "sales"),
                row("Database", "public"),
            ];
            for cells in &mut rows {
                cells.resize(width, blank.clone());
            }
            let found = describe::describe_view(&rows, spaces()).unwrap();
            assert_eq!(found.definition, "source: orders");
            assert_eq!(
                found.dtypes,
                [
                    ("id".into(), "int".into()),
                    ("category".into(), "string".into())
                ]
            );
            assert_eq!(
                found.parameters,
                [
                    ("start".into(), "date".into()),
                    ("limit".into(), "int".into())
                ]
            );
            assert_eq!(found.catalog.as_deref(), Some("sales"));
            assert_eq!(found.schema.as_deref(), Some("public"));
        }
    }
}

#[test]
fn ordinary_tables_and_views_are_not_metric_views() {
    for kind in ["MANAGED", "VIEW", "metric_view"] {
        let rows = [
            row("# Detailed Table Information", ""),
            row("Type", kind),
            row("View Text", "source: orders"),
        ];
        assert_eq!(
            describe::describe_view(&rows, spaces()),
            Err(DescribeRefusal::NotAMetricView {
                got: Some(kind.into())
            })
        );
    }
}

fn row(name: &str, value: &str) -> Vec<Option<String>> {
    vec![Some(name.to_owned()), Some(value.to_owned()), None]
}

/// Rows 23 and 24 by name: no `Type`, and a metric view whose `View Text`
/// is empty; and a detail named twice, the first kept.
#[test]
fn a_description_missing_its_type_or_text_is_refused() {
    let header = |name: &str| vec![Some(name.to_owned()), Some(String::new()), None];
    let no_type = vec![
        row("a", "int"),
        vec![None, None, None],
        header("# Detailed Table Information"),
        row("View Text", "source: t"),
    ];
    assert_eq!(
        describe::describe_view(&no_type, spaces()).unwrap_err(),
        DescribeRefusal::NotAMetricView { got: None }
    );
    let no_text = vec![
        header("# Detailed Table Information"),
        row("Type", "METRIC_VIEW"),
        row("View Text", ""),
    ];
    assert_eq!(
        describe::describe_view(&no_text, spaces()).unwrap_err(),
        DescribeRefusal::NoDefinition
    );
    let twice = vec![
        header("# Detailed Table Information"),
        row("Type", "METRIC_VIEW"),
        row("View Text", "source: t"),
        row("View Text", "source: u"),
        row("Catalog", "c"),
    ];
    let read = describe::describe_view(&twice, spaces()).unwrap();
    assert_eq!(read.definition, "source: t");
    assert_eq!(read.catalog.as_deref(), Some("c"));
    assert_eq!(read.schema, None);
}
