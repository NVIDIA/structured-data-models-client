// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Array and timestamp options preserve mode-specific validation contracts.
use sdm_pql::dates::Reader;
use sdm_pql::schema::{Column, ColumnKey, ColumnKeyGroup, SchemaDefinition, TableDefinition};
use sdm_pql::types::{Dtype, Stype};
use sdm_pql::validator::predictive_query::PredictiveQueryValidator;
use sdm_pql::{Filters, Mode, parse};

fn schema() -> SchemaDefinition {
    let table = |key: &str, time: Option<&str>, cols: &[(&str, &str, &str)]| TableDefinition {
        cols: cols
            .iter()
            .map(|(name, dtype, stype)| Column {
                name: (*name).into(),
                dtype: Dtype::parse(dtype).unwrap(),
                stype: Stype::parse(stype).unwrap(),
            })
            .collect(),
        pkey: Some(key.into()),
        time_col: time.map(str::to_owned),
        end_time_col: None,
    };
    SchemaDefinition {
        tables: [
            (
                "users".into(),
                table(
                    "user_id",
                    None,
                    &[
                        ("user_id", "int", "ID"),
                        ("age", "int", "numerical"),
                        ("country", "string", "categorical"),
                        ("vip", "bool", "categorical"),
                        ("bio", "string", "text"),
                        ("signup", "time", "timestamp"),
                    ],
                ),
            ),
            (
                "products".into(),
                table(
                    "product_id",
                    None,
                    &[
                        ("product_id", "int", "ID"),
                        ("category", "string", "categorical"),
                        ("price", "float", "numerical"),
                        ("tags", "stringlist", "multicategorical"),
                    ],
                ),
            ),
            (
                "orders".into(),
                table(
                    "order_id",
                    Some("ordered_at"),
                    &[
                        ("order_id", "int", "ID"),
                        ("user_id", "int", "ID"),
                        ("product_id", "int", "ID"),
                        ("quantity", "int", "numerical"),
                        ("amount", "float", "numerical"),
                        ("status", "string", "categorical"),
                        ("ordered_at", "time", "timestamp"),
                    ],
                ),
            ),
            (
                "reviews".into(),
                table(
                    "review_id",
                    Some("created_at"),
                    &[
                        ("review_id", "int", "ID"),
                        ("user_id", "int", "ID"),
                        ("product_id", "int", "ID"),
                        ("rating", "int", "numerical"),
                        ("created_at", "time", "timestamp"),
                    ],
                ),
            ),
        ]
        .into_iter()
        .collect(),
        col_groups: vec![
            ColumnKeyGroup {
                columns: vec![
                    ColumnKey::new("orders", "user_id"),
                    ColumnKey::new("reviews", "user_id"),
                    ColumnKey::new("users", "user_id"),
                ],
            },
            ColumnKeyGroup {
                columns: vec![
                    ColumnKey::new("orders", "product_id"),
                    ColumnKey::new("products", "product_id"),
                    ColumnKey::new("reviews", "product_id"),
                ],
            },
        ],
    }
}

// Expected diagnostics are semantic rules, independent of validator execution.
// Span columns refer to the authored query text below; every span is on row 1.
type Diagnostic = (&'static str, Option<(usize, usize)>);
fn expected(kind: &str, training: bool, arrays: bool, timestamps: bool) -> Vec<Diagnostic> {
    if kind == "ids-ranked" || (kind == "ids" && !training) {
        return vec![];
    }
    if kind == "ids" {
        return vec![("PQL_TASK_PROBLEM_TYPE_REQUIRED", Some((63, 63)))];
    }
    if kind == "categories" || kind == "categories-classified" {
        return if !training {
            vec![("PQL_MODEL_NO_MULTILABEL", None)]
        } else if kind == "categories" {
            vec![("PQL_TASK_PROBLEM_TYPE_REQUIRED", Some((59, 59)))]
        } else {
            vec![]
        };
    }
    if kind == "compared" {
        return if arrays {
            vec![("PQL_TYPE_CONSTANT_MISMATCH", Some((8, 52)))]
        } else if training {
            vec![]
        } else {
            vec![
                ("PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION", Some((8, 52))),
                ("PQL_MODEL_AGGREGATION_NEEDS_FOREIGN_KEY", Some((8, 48))),
            ]
        };
    }
    assert!(matches!(kind, "timestamps" | "timestamps-ranked"));
    if !arrays {
        return if !training {
            vec![("PQL_MODEL_NO_MULTILABEL", None)]
        } else if kind == "timestamps" {
            vec![("PQL_TASK_PROBLEM_TYPE_REQUIRED", Some((63, 63)))]
        } else {
            vec![]
        };
    }
    let mut errors = vec![];
    if !timestamps {
        errors.push((
            "PQL_TYPE_AGGREGATION_ON_TIMESTAMP_UNSUPPORTED",
            Some((8, 52)),
        ));
    }
    errors.push(("PQL_TYPE_TARGET_DTYPE_NOT_PREDICTABLE", None));
    errors
}

#[test]
fn array_options_preserve_all_query_and_mode_combinations() {
    let schema = schema();
    let queries = [
        ("timestamps", "orders.ordered_at", ""),
        ("timestamps-ranked", "orders.ordered_at", " RANK TOP 3"),
        ("ids-ranked", "orders.product_id", " RANK TOP 5"),
        ("ids", "orders.product_id", ""),
        ("categories", "orders.status", ""),
        ("categories-classified", "orders.status", " CLASSIFY"),
        ("compared", "orders.status", " = 'shipped'"),
    ];
    for training in [false, true] {
        let mode = if training {
            Mode::Training
        } else {
            Mode::Inference
        };
        for arrays in [false, true] {
            for timestamps in [false, true] {
                for (kind, target, suffix) in queries {
                    let query = format!(
                        "PREDICT LIST_DISTINCT({target}, 0, 30, days){suffix} FOR EACH users.user_id"
                    );
                    let parsed = parse(
                        &query,
                        mode,
                        Reader {
                            year: Some(2026),
                            pads_year: false,
                        },
                    )
                    .outcome
                    .unwrap();
                    let validator = PredictiveQueryValidator {
                        allow_array_targets: arrays,
                        allow_timestamp_arrays: timestamps,
                        ..PredictiveQueryValidator::new(&schema, mode, Filters::default())
                    };
                    let (resolved, response) =
                        validator.validate_predictive_query(parsed.query).unwrap();
                    let want = expected(kind, training, arrays, timestamps);
                    assert_eq!(
                        resolved.is_some() && response.ok(),
                        want.is_empty(),
                        "{kind}/{training}/{arrays}/{timestamps}"
                    );
                    let got: Vec<_> = response
                        .errors
                        .iter()
                        .map(|e| {
                            let span = e.span.map(|s| {
                                assert_eq!((s.start_row, s.end_row), (1, 1));
                                (s.start_col, s.end_col)
                            });
                            (e.code, span)
                        })
                        .collect();
                    assert_eq!(got, want, "{kind}/{training}/{arrays}/{timestamps}");
                }
            }
        }
    }
}
