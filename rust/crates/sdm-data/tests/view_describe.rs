// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What `DESCRIBE TABLE EXTENDED` says of a metric view (T3.5, M1,
//! old-behaviour rows 21 to 24), read from its rows as the old client reads
//! them (`DB/metric_view.py:1389-1475`), against every description its own
//! tests made: four cells from a SQL warehouse, three from Spark, an empty
//! cell `''` or null, Spark's `# col_name` rows, a relation that is no metric
//! view, and one with no definition.

use sdm_data::view_expr::describe::{self, DescribeRefusal};
use serde_json::Value as Json;

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

const VECTORS: &str = include_str!("../../../../contract/view-step-vectors.json");

fn pairs(value: &Json) -> Vec<(String, String)> {
    value
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, entry)| (name.clone(), entry.as_str().unwrap().to_owned()))
        .collect()
}

#[test]
fn a_description_is_read_as_the_old_client_read_it() {
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    let vectors = document["describe"].as_array().unwrap();
    assert!(vectors.len() >= 20, "{} descriptions", vectors.len());
    let mut wrong = Vec::new();
    for vector in vectors {
        let rows: Vec<Vec<Option<String>>> = vector["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|cell| cell.as_str().map(str::to_owned))
                    .collect()
            })
            .collect();
        let read = describe::describe_view(&rows, spaces());
        match (read, vector.get("error")) {
            (Ok(description), None) => {
                let expected = (
                    vector["definition"].as_str().unwrap().to_owned(),
                    pairs(&vector["dtypes"]),
                    pairs(&vector["parameters"]),
                    vector["catalog"].as_str().map(str::to_owned),
                    vector["schema"].as_str().map(str::to_owned),
                );
                let got = (
                    description.definition,
                    description.dtypes,
                    description.parameters,
                    description.catalog,
                    description.schema,
                );
                if got != expected {
                    wrong.push(format!("{}: {got:?} where {expected:?}", vector["name"]));
                }
            }
            (Err(refusal), Some(error)) => {
                let message = error["message"].as_str().unwrap();
                let same = match &refusal {
                    DescribeRefusal::NotAMetricView { got } => message.contains(&format!(
                        "(got type '{}')",
                        got.as_deref().unwrap_or("UNKNOWN")
                    )),
                    DescribeRefusal::NoDefinition => {
                        message.starts_with("Could not read the definition")
                    }
                };
                if !same {
                    wrong.push(format!("{}: {refusal:?} where {message}", vector["name"]));
                }
            }
            (read, error) => wrong.push(format!("{}: {read:?} where {error:?}", vector["name"])),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
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
