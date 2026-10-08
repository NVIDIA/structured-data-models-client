// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a validated query offers its callers beyond its JSON: writing it
//! back, its autoregressive form, its keys, windows and entity IDs. Each
//! expectation is what today's Python gives for the same query.

use std::path::Path;

use sdm_pql::types::{Dtype, QueryType, Stype, TaskType};
use sdm_pql::validator::predictive_query::PredictiveQueryValidator;
use sdm_pql::{Profile, SchemaDefinition, ValidatedQuery, validate_in};

/// The shop schema of the PQL corpus.
fn shop() -> SchemaDefinition {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/parity/pql/valid.json");
    let corpus: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("the corpus")).expect("JSON");
    SchemaDefinition::from_value(&corpus["cases"][0]["schema"]).expect("a schema")
}

fn resolve(query: &str, profile: &str) -> ValidatedQuery {
    let profile = Profile::by_name(profile).expect("a profile");
    let host = sdm_pql::dates::Host {
        year: 2026,
        pads_year: false,
    };
    let validation = validate_in(query, &shop(), &profile, host);
    validation
        .query
        .unwrap_or_else(|| panic!("{query} refused: {:?}", validation.response.errors))
}

#[test]
fn a_query_writes_back_plain_and_rich() {
    let q = resolve(
        "EXPLAIN PREDICT NOT COUNT(orders.* WHERE orders.status IS NULL, 0, 4, weeks) > 2 FOR \
         users.user_id IN (1, 2) WHERE users.age > 3 ASSUMING SUM(orders.amount, 0, 30, days) > 1.5",
        "RELATIONAL",
    );
    assert_eq!(
        q.query.to_string_with(false, false),
        "EXPLAIN PREDICT NOT (COUNT(orders.* WHERE orders.status IS NULL, 0, 4, weeks) > 2) FOR \
         users.user_id IN (1, 2) WHERE users.age > 3 ASSUMING SUM(orders.amount, 0, 30, days) > 1.5"
    );
    assert_eq!(
        q.query.to_string_with(true, false),
        "EXPLAIN [bold]PREDICT[/bold] [bold]NOT[/bold] ([bold]COUNT[/bold](orders.* \
         [bold]WHERE[/bold] orders.status [bold]IS[/bold] [bold]NULL[/bold], 0, 4, weeks) > 2) \
         [bold]FOR[/bold] users.user_id IN (1, 2) [bold]WHERE[/bold] users.age > 3 \
         [bold]ASSUMING[/bold] [bold]SUM[/bold](orders.amount, 0, 30, days) > 1.5"
    );
    assert_eq!(
        q.query
            .to_autoregressive_query()
            .to_string_with(false, false),
        "EXPLAIN PREDICT COUNT(orders.* WHERE orders.status IS NULL, 0, 4, weeks) > 2 FOR \
         users.user_id IN (1, 2)"
    );
    let ids: Vec<String> = q
        .get_rfm_entity_id_list()
        .unwrap()
        .iter()
        .map(|v| v.py_repr())
        .collect();
    assert_eq!(ids, ["1", "2"]);
    assert_eq!(
        q.get_single_target_fkey(false).as_deref(),
        Some("orders.user_id")
    );
    assert_eq!(
        q.get_single_target_fkey(true).as_deref(),
        Some("users.user_id")
    );
    assert_eq!(q.query_type(), QueryType::Temporal);
    assert_eq!(q.entity_dtype(), Some(Dtype::Int));
    assert_eq!(q.target_dtype(), Some(Dtype::Bool));
    assert_eq!(q.target_stype(), Some(Stype::Categorical));
    let range = q.get_combined_date_offset_range().unwrap();
    assert_eq!((range.borrow().start, range.borrow().end), (Some(0), 4));
}

#[test]
fn a_ranking_writes_back_its_spec_and_checks_batch_filters() {
    let q = resolve(
        "PREDICT LIST_DISTINCT(orders.product_id WHERE orders.amount > 3, 0, 30, days) RANK TOP 5 \
         FOR EACH users.user_id",
        "FINE_TUNING",
    );
    assert_eq!(
        q.query.to_string_with(true, false),
        "[bold]PREDICT[/bold] [bold]LIST_DISTINCT[/bold](orders.product_id [bold]WHERE[/bold] \
         orders.amount > 3, 0, 30, days) RANK [bold]TOP 5[/bold] [bold]FOR EACH[/bold] \
         users.user_id"
    );
    assert_eq!(
        q.query.to_string_with(false, true),
        "LIST_DISTINCT(orders.product_id WHERE orders.amount > 3, 0, 30, days) RANK TOP 5 FOR \
         EACH users.user_id"
    );
    assert_eq!(q.target_dtype(), Some(Dtype::String));
    let schema = shop();
    let validator = PredictiveQueryValidator::new(
        &schema,
        sdm_pql::Mode::Training,
        sdm_pql::Capabilities::default(),
    );
    let response = validator
        .validate_bp_filter_overrides(&q, TaskType::TemporalLinkPrediction)
        .unwrap();
    assert_eq!(response.errors.len(), 1);
    assert_eq!(response.errors[0].code, "PQL_REWRITE_TABLE_NOT_AVAILABLE");
    assert_eq!(
        response.errors[0].message,
        "row 1, column 46: Table orders appears in the target filter edit orders.amount > 3. \
         During batch prediction, only conditions that refer to products table can be added."
    );
    let none = validator
        .validate_bp_filter_overrides(&q, TaskType::Regression)
        .unwrap();
    assert!(none.ok());
}

#[test]
fn a_static_target_names_its_columns() {
    let q = resolve("PREDICT users.age FOR EACH users.user_id", "FINE_TUNING");
    let excluded = q.get_exclude_cols_dict();
    assert_eq!(
        excluded.get("users").map(Vec::as_slice),
        Some(&["age".to_string()][..])
    );
    assert_eq!(q.get_single_target_fkey(false), None);
    assert_eq!(q.target_column(), "users.age");
    assert_eq!(q.query_type(), QueryType::Static);
    assert_eq!(
        q.get_combined_date_offset_range().unwrap_err(),
        "Cannot compute combined time range because target does not have any aggregations."
    );
}

#[test]
fn merging_target_windows_rewrites_them_as_python_does() {
    let q = resolve(
        "PREDICT COUNT(orders.*, 0, 2, weeks) > 1 OR COUNT(reviews.*, 0, 3, days) > 0 FOR EACH \
         users.user_id",
        "FINE_TUNING",
    );
    let range = q.get_combined_date_offset_range().unwrap();
    assert_eq!((range.borrow().start, range.borrow().end), (Some(0), 14));
    assert_eq!(
        q.target_column(),
        "(COUNT(orders.*, 0, 14, days) > 1) OR (COUNT(reviews.*, 0, 3, days) > 0)"
    );
    let forecast = resolve(
        "PREDICT SUM(orders.amount, 0, 7, days) FORECAST 3 TIMEFRAMES FOR EACH users.user_id",
        "FINE_TUNING",
    );
    assert_eq!(
        forecast.query.to_string_with(true, false),
        "[bold]PREDICT[/bold] [bold]SUM[/bold](orders.amount, 0, 7, days) [bold]FORECAST[/bold] \
         3 [bold]TIMEFRAMES[/bold] [bold]FOR EACH[/bold] users.user_id"
    );
}

#[test]
fn validate_returns_the_warnings_each_call_gives() {
    // Today's Python reads the date twice, and pandas warns each time.
    let day_first = "Parsing dates in %d/%m/%Y format when dayfirst=False (the default) was \
                     specified. Pass `dayfirst=True` or specify a format to silence this warning.";
    let profile = Profile::by_name("RELATIONAL").expect("a profile");
    let dated = "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE users.signup > \
                 13/01/2024";
    for _ in 0..2 {
        let validation = sdm_pql::validate(dated, &shop(), &profile);
        assert!(validation.query.is_some(), "{:?}", validation.response);
        assert_eq!(validation.warnings, [day_first, day_first]);
    }
    let undated = sdm_pql::validate(
        "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id",
        &shop(),
        &profile,
    );
    assert!(undated.warnings.is_empty());
}
