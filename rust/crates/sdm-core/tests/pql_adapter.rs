// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapter's own rules: a model that cannot answer a time range refuses
//! before resolution; the two filter capabilities reach resolution; date
//! literals are read by D1 to D7 and D9, and a day-first one warns (D4),
//! while one with a run of five digits or more (D8) is refused by name
//! (T1.6c); and an accepted query is handed on in the validated-query
//! format.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: times the bounded-time refusals against a wall clock"
)]

use std::collections::BTreeMap;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Capability, ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::pql::{Aggregation, EntityClause, NodeKind, PqlLiteral, QueryMode};
use sdm_contract::{CoreError, CoreWarning, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Unit};
use sdm_core::pql::{DATE_LITERAL_DAY_FIRST, resolve, validate};
use sdm_core::profiles::Profile;
use serde_json::json;
/// Zones that hold nothing: these facts name no local zone, so none is
/// resolved.
struct NoZones;

impl sdm_core::Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        Err(sdm_core::UnknownZone)
    }
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    }
}

/// The inventory's `shop` tables, as the reference describes them.
fn shop() -> QuerySchema {
    serde_json::from_value(json!({
        "tables": [
            {"name": "orders", "columns": [
                {"name": "order_id", "dtype": "int", "stype": "ID"},
                {"name": "user_id", "dtype": "int", "stype": "ID"},
                {"name": "price", "dtype": "float", "stype": "numerical"},
                {"name": "ts", "dtype": "time", "stype": "timestamp"}],
             "primary_key": "order_id", "time_column": "ts", "end_time_column": null},
            {"name": "users", "columns": [
                {"name": "user_id", "dtype": "int", "stype": "ID"},
                {"name": "age", "dtype": "int", "stype": "numerical"},
                {"name": "country", "dtype": "string", "stype": "categorical"}],
             "primary_key": "user_id", "time_column": null, "end_time_column": null}],
        "links": [[{"table": "orders", "column": "user_id"}, {"table": "users", "column": "user_id"}]],
    }))
    .unwrap()
}

fn profile(name: ProfileName, capabilities: &[(Capability, bool)]) -> Profile {
    Profile::resolve(&ProfileSpec {
        name,
        model: None,
        capabilities: capabilities.iter().copied().collect::<BTreeMap<_, _>>(),
    })
}

fn codes_of(query: &str, profile: &Profile) -> Vec<String> {
    resolve(query, &shop(), profile, &facts(), &NoZones)
        .unwrap()
        .diagnostics
        .into_iter()
        .map(|d| d.code)
        .collect()
}

#[test]
fn a_time_range_is_refused_before_resolution() {
    // A tabular model predicts values as they stand: the window is the
    // reason no schema can fix, and it is the only one given.
    let query = "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id";
    let validation = resolve(
        query,
        &shop(),
        &profile(ProfileName::Tabular, &[]),
        &facts(),
        &NoZones,
    )
    .unwrap();
    assert!(validation.query.is_none());
    assert_eq!(validation.diagnostics.len(), 1);
    assert_eq!(
        validation.diagnostics[0].code,
        "PQL_MODEL_NO_TEMPORAL_TARGET"
    );
    assert_eq!(validation.diagnostics[0].span, None);
    // With the capability granted, the same query resolves and meets the
    // checks of the resolved query instead.
    let granted = profile(ProfileName::Tabular, &[(Capability::Temporal, true)]);
    assert_eq!(
        codes_of(query, &granted),
        ["PQL_MODEL_SINGLE_TABLE_ONLY", "PQL_MODEL_NO_DERIVED_TARGET"]
    );
}

#[test]
fn the_filter_capabilities_reach_resolution() {
    let relational = profile(ProfileName::Relational, &[]);
    let on_key =
        "PREDICT COUNT(orders.* WHERE orders.user_id = 3, 0, 30, days) FOR EACH users.user_id";
    let on_time = "PREDICT COUNT(orders.* WHERE orders.ts > 2024-01-01, 0, 30, days) FOR EACH \
                   users.user_id";
    assert_eq!(codes_of(on_key, &relational), Vec::<String>::new());
    assert_eq!(codes_of(on_time, &relational), Vec::<String>::new());

    let no_keys = profile(
        ProfileName::Relational,
        &[(Capability::KeyColumnsInFilters, false)],
    );
    assert!(codes_of(on_key, &no_keys).contains(&"PQL_MODEL_NO_KEY_COLUMN_FILTER".to_owned()));
    let no_times = profile(
        ProfileName::Relational,
        &[(Capability::TimestampColumnsInFilters, false)],
    );
    assert_eq!(
        codes_of(on_time, &no_times),
        ["PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION"]
    );
}

/// The time constant of a query `PREDICT COUNT(orders.* WHERE orders.ts >
/// <literal>, ...)`: its reading and offset, and the warnings the reading
/// gave.
fn read_literal(literal: &str) -> (Option<(String, Option<i32>)>, Vec<CoreWarning>) {
    let query = format!(
        "PREDICT COUNT(orders.* WHERE orders.ts > {literal}, 0, 30, days) FOR EACH users.user_id"
    );
    let validation = resolve(
        &query,
        &shop(),
        &profile(ProfileName::Relational, &[]),
        &facts(),
        &NoZones,
    )
    .unwrap();
    let Some(query) = validation.query else {
        return (None, validation.warnings);
    };
    let mut found = None;
    let mut visit = vec![&query.target];
    while let Some(node) = visit.pop() {
        match &node.kind {
            NodeKind::Constant {
                literal: PqlLiteral::Time { value, offset, .. },
            } => found = Some((value.clone(), *offset)),
            NodeKind::Join { target, .. } | NodeKind::Aggregation { target, .. } => {
                visit.push(target);
            }
            NodeKind::Filter { target, condition } => visit.extend([&**target, &**condition]),
            NodeKind::Condition { target, value, .. } => visit.extend([&**target, &**value]),
            _ => {}
        }
    }
    (found, validation.warnings)
}

/// `PREDICT COUNT(orders.* WHERE <condition>, 0, 30, days) FOR EACH
/// users.user_id`.
fn filtered(condition: &str) -> String {
    format!("PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) FOR EACH users.user_id")
}

/// The core's refusal of `query`, which must refuse rather than answer.
fn refusal_of(query: &str, facts: &HostFacts) -> CoreError {
    resolve(
        query,
        &shop(),
        &profile(ProfileName::Relational, &[]),
        facts,
        &NoZones,
    )
    .expect_err(query)
}

/// A text field of `error`.
fn text_field(error: &CoreError, name: &str) -> Option<String> {
    match error.fields.get(name) {
        Some(Detail::Scalar(Scalar::Utf8(text))) => Some(text.clone()),
        _ => None,
    }
}

/// T1.6c's refusal by name of a D8 literal: `INVALID_REQUEST`, kind
/// invalid request, naming `operation` `validate_query` and `rule`
/// `date_literal_packed`, with the literal's place as D4's warning gives
/// it (`start_row`, `start_col`, `end_row`, `end_col`), and never the
/// literal's text, in its message or any field.
fn assert_refused_as_packed(error: &CoreError, literal: &str) {
    assert_eq!(
        (error.kind, error.code),
        (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest),
        "{literal}: {error:?}"
    );
    assert_eq!(
        text_field(error, "operation").as_deref(),
        Some("validate_query"),
        "{literal}: {error:?}"
    );
    assert_eq!(
        text_field(error, "rule").as_deref(),
        Some("date_literal_packed"),
        "{literal}: {error:?}"
    );
    for name in ["start_row", "start_col", "end_row", "end_col"] {
        assert!(
            matches!(
                error.fields.get(name),
                Some(Detail::Scalar(Scalar::UInt64(_)))
            ),
            "{literal}: no {name} in {error:?}"
        );
    }
    let written = serde_json::to_string(error).unwrap();
    assert!(!written.contains(literal), "{literal}: {written}");
    for run in literal
        .split([' ', '/', '-', '.', ':'])
        .filter(|r| r.len() >= 5)
    {
        assert!(!written.contains(run), "{literal}: {written}");
    }
}

/// The D8 literals the T1.6c tests refuse: a packed first field of six,
/// eight, twelve and fourteen digits, a long field in the time of day, a
/// long zero-padded field, and ones the reference refuses (a year past
/// 9999, an offset of a day, a long last field). Register DT-9's two
/// queries are the second and fourth.
const PACKED: &[&str] = &[
    "230203-06-09",
    "20240101-01-01",
    "202401011230-1-1",
    "20240101123456-1-1",
    "20240101/00/123456",
    "2024-01-01 00000:00:00",
    "00001/1/2",
    "10000-01-01",
    "12345-1-1",
    "20240101-01-24",
    "1-1-20240101",
];

/// T1.6c brief, Scope: "a D8 literal is refused with a named code and rule
/// ... naming `date_literal_packed`", through `resolve` and `validate`, the
/// adapter behind `validate_query` and `materialize`: each shape is
/// refused by name, those the reference reads and those it refuses alike,
/// never read and never `PQL_VALUE_INVALID_LITERAL`.
#[test]
fn a_packed_date_or_long_field_is_refused_by_name() {
    for literal in PACKED {
        let query = filtered(&format!("orders.ts > {literal}"));
        assert_refused_as_packed(&refusal_of(&query, &facts()), literal);
        let validated = validate(
            &query,
            &shop(),
            &profile(ProfileName::Relational, &[]),
            &facts(),
            &NoZones,
        );
        assert_refused_as_packed(&validated.expect_err(literal), literal);
    }
}

/// T1.6c: the refusal names the literal's place as D4's warning names a
/// literal written at the same place with as many characters, the first
/// row's 41st column after the filter's `orders.ts > `.
#[test]
fn the_packed_refusal_names_the_literal_s_place_as_d4_s_warning_does() {
    let error = refusal_of(&filtered("orders.ts > 13/1/20240"), &facts());
    assert_refused_as_packed(&error, "13/1/20240");
    let (found, warnings) = read_literal("13/01/2024");
    assert_eq!(found, Some(("2024-01-13T00:00:00".to_owned(), None)));
    let [warning] = &warnings[..] else {
        panic!("one warning: {warnings:?}");
    };
    for name in ["start_row", "start_col", "end_row", "end_col"] {
        assert_eq!(error.fields.get(name), warning.fields.get(name), "{name}");
    }
    let at = |n: u64| Detail::Scalar(Scalar::UInt64(n));
    assert_eq!(error.fields["start_row"], at(1));
    assert_eq!(error.fields["start_col"], at(41));
}

/// T1.6c: a D8 literal in an array (`IN (...)`) is refused by name too.
#[test]
fn a_packed_date_in_an_array_is_refused_by_name() {
    let plain = filtered("orders.ts IN (2024-01-01, 2024-02-01)");
    assert!(
        resolve(
            &plain,
            &shop(),
            &profile(ProfileName::Relational, &[]),
            &facts(),
            &NoZones,
        )
        .is_ok(),
        "an array of dates is answered"
    );
    for literal in ["20240101-01-01", "2024-01-01 00000:00:00"] {
        let query = filtered(&format!("orders.ts IN (2024-01-01, {literal})"));
        assert_refused_as_packed(&refusal_of(&query, &facts()), literal);
    }
}

/// T1.6c brief: "keep everything else in D1 to D9 unchanged". Literals
/// whose every run has four digits or fewer read as before (a date, a time
/// of day, a two-digit year, the year 0, four-digit runs where five would
/// be D8), a day-first one still warns (D4), and D9's still give the
/// diagnostic `PQL_VALUE_INVALID_LITERAL`, never the named rule. Each value
/// is the reference's (`pd.to_datetime`, pandas 3.0.6).
#[test]
fn a_literal_with_no_run_of_five_digits_reads_as_before() {
    for (literal, value) in [
        ("2024-01-31", "2024-01-31T00:00:00"),
        ("2024-01-31 23:59:59", "2024-01-31T23:59:59"),
        ("1/2/24", "2024-01-02T00:00:00"),
        ("0000-01-01", "0000-01-01T00:00:00"),
        ("0001/1/2", "0001-01-02T00:00:00"),
        ("2024-01-01 0000:00:00", "2024-01-01T00:00:00"),
        ("01.02.2024", "2024-01-02T00:00:00"),
    ] {
        let (found, warnings) = read_literal(literal);
        assert_eq!(found, Some((value.to_owned(), None)), "{literal}");
        assert!(warnings.is_empty(), "{literal}");
    }
    let (found, warnings) = read_literal("31/12/2024");
    assert_eq!(found, Some(("2024-12-31T00:00:00".to_owned(), None)));
    let codes: Vec<&str> = warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, [DATE_LITERAL_DAY_FIRST]);
    for literal in ["2023-02-29", "123-4-567", "2024-01-01 24:00:00"] {
        let refused = codes_of(
            &filtered(&format!("orders.ts > {literal}")),
            &profile(ProfileName::Relational, &[]),
        );
        assert_eq!(refused, ["PQL_VALUE_INVALID_LITERAL"], "{literal}");
    }
}

/// T1.6c: node order decides, as for any refusal of a node. A syntax
/// error, which the grammar finds before any node is built, and a D9
/// literal built before the D8 one still give their diagnostic; a D8
/// literal built first ends the parse, so a D9 literal after it gives none.
#[test]
fn an_earlier_refusal_wins_and_the_first_packed_literal_ends_the_parse() {
    let relational = profile(ProfileName::Relational, &[]);
    let syntax = format!("{} )", filtered("orders.ts > 20240101-01-01"));
    assert_eq!(codes_of(&syntax, &relational), ["PQL_PARSE_SYNTAX"]);
    let d9_first = filtered("orders.ts > 2023-02-29 AND orders.ts < 20240101-01-01");
    assert_eq!(
        codes_of(&d9_first, &relational),
        ["PQL_VALUE_INVALID_LITERAL"]
    );
    let d8_first = filtered("orders.ts > 20240101-01-01 AND orders.ts < 2023-02-29");
    assert_refused_as_packed(&refusal_of(&d8_first, &facts()), "20240101-01-01");
    let two = filtered("orders.ts > 230203-06-09 AND orders.ts < 20240101-01-01");
    assert_refused_as_packed(&refusal_of(&two, &facts()), "230203-06-09");
    // The first one's place.
    let error = refusal_of(&two, &facts());
    assert_eq!(
        error.fields["start_col"],
        Detail::Scalar(Scalar::UInt64(41))
    );
}

#[test]
fn a_day_first_literal_warns_with_its_format_and_place_never_its_text() {
    let (found, warnings) = read_literal("29.02.2024 10:00:00");
    assert_eq!(found, Some(("2024-02-29T10:00:00".to_owned(), None)));
    let [warning] = &warnings[..] else {
        panic!("one warning: {warnings:?}");
    };
    assert_eq!(warning.code, DATE_LITERAL_DAY_FIRST);
    assert_eq!(warning.fields["format"], Detail::text("%d.%m.%Y %H:%M:%S"));
    let at = |n: u64| Detail::Scalar(Scalar::UInt64(n));
    assert_eq!(warning.fields["start_row"], at(1));
    assert_eq!(warning.fields["start_col"], at(41));
    let written = serde_json::to_string(warning).unwrap();
    assert!(!written.contains("29.02"), "{written}");
    // A month-first reading, and a day-first one whose format pandas cannot
    // name (a two-digit year), give none; two day-first literals give two.
    assert!(read_literal("01.02.2024").1.is_empty());
    assert!(read_literal("13/1/24").1.is_empty());
    let both = resolve(
        "PREDICT COUNT(orders.* WHERE orders.ts > 13/01/2024 AND orders.ts < 31/01/2024, 0, 30, \
         days) FOR EACH users.user_id",
        &shop(),
        &profile(ProfileName::Relational, &[]),
        &facts(),
        &NoZones,
    )
    .unwrap();
    assert!(both.query.is_some());
    assert_eq!(both.warnings.len(), 2);
    // A query refused after its literal was read still warns, as the
    // reference warns when it reads it.
    let refused = resolve(
        "PREDICT COUNT(orders.* WHERE orders.nope > 13/01/2024, 0, 30, days) FOR EACH \
         users.user_id",
        &shop(),
        &profile(ProfileName::Relational, &[]),
        &facts(),
        &NoZones,
    )
    .unwrap();
    assert!(refused.query.is_none());
    assert_eq!(refused.warnings.len(), 1);
}

#[test]
fn an_accepted_query_is_handed_on_in_the_validated_query_format() {
    let validation = resolve(
        "PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id",
        &shop(),
        &profile(ProfileName::Relational, &[]),
        &facts(),
        &NoZones,
    )
    .unwrap();
    assert!(validation.diagnostics.is_empty());
    assert!(validation.to_result().accepted);
    let query = validation.query.unwrap();
    assert_eq!(query.entity_clause, EntityClause::ForEach);
    assert_eq!(query.mode, QueryMode::Inference);
    let NodeKind::Join {
        target, lhs_key, ..
    } = &query.target.kind
    else {
        panic!("the target is not joined to users: {:?}", query.target.kind);
    };
    assert_eq!(
        (lhs_key.table.as_str(), lhs_key.column.as_str()),
        ("users", "user_id")
    );
    let NodeKind::Aggregation {
        aggregation,
        time_column,
        ..
    } = &target.kind
    else {
        panic!("the joined target is not an aggregation");
    };
    assert_eq!(*aggregation, Aggregation::Sum);
    assert_eq!(time_column.as_ref().unwrap().column, "ts");
}

/// The parser's nesting limit (`parser::MAX_DEPTH`).
const MAX_DEPTH: usize = 1000;

fn nots(depth: usize) -> String {
    format!(
        "PREDICT {}COUNT(orders.*, 0, 30, days) > 1 FOR EACH users.user_id",
        "NOT ".repeat(depth)
    )
}

fn parentheses(depth: usize) -> String {
    format!(
        "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {}users.age > 3{}",
        "(".repeat(depth),
        ")".repeat(depth)
    )
}

/// A filter of `links` + 1 conditions joined by `op`.
fn chain(op: &str, links: usize) -> String {
    format!(
        "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE users.age > 3{}",
        format!(" {op} users.age > 3").repeat(links)
    )
}

/// The deepest queries the parser takes, then the same one level past
/// them: 999 `NOT`s or parentheses are accepted and 1,000 refused; a chain
/// of 998 `AND` or `OR` links is accepted and 999 refused, each link a
/// level below the first condition's.
fn deepest() -> Vec<String> {
    let deepest = MAX_DEPTH - 1;
    let mut out = Vec::new();
    for past in [0, 1] {
        out.push(nots(deepest + past));
        out.push(parentheses(deepest + past));
        out.push(chain("AND", deepest - 1 + past));
        out.push(chain("OR", deepest - 1 + past));
    }
    out
}

fn on_a_stack_of(bytes: usize) {
    std::thread::Builder::new()
        .stack_size(bytes)
        .spawn(|| {
            let relational = profile(ProfileName::Relational, &[]);
            let answers: Vec<bool> = deepest()
                .iter()
                .map(|query| {
                    validate(query, &shop(), &relational, &facts(), &NoZones)
                        .unwrap()
                        .0
                        .accepted
                })
                .collect();
            // The deepest are answered, accepted; one level deeper, refused.
            assert_eq!(
                answers,
                [true, true, true, true, false, false, false, false]
            );
        })
        .unwrap()
        .join()
        .expect("answered");
}

#[test]
fn the_deepest_queries_validate_on_a_spawned_threads_default_stack() {
    on_a_stack_of(2 << 20);
}

#[test]
fn the_deepest_queries_validate_on_a_small_thread_stack() {
    on_a_stack_of(512 << 10);
}

#[test]
fn a_long_and_or_or_chain_is_refused_at_the_depth_limit_in_bounded_time() {
    // Each link counts against MAX_DEPTH, so a chain of 10,000 is refused
    // where the limit falls, as a syntax error, before any later pass
    // recurses through it: on a small stack, without overflowing, and
    // quickly (it once took seconds and aborted the process).
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| {
            let relational = profile(ProfileName::Relational, &[]);
            for op in ["AND", "OR"] {
                let started = std::time::Instant::now();
                let validation =
                    validate(&chain(op, 10_000), &shop(), &relational, &facts(), &NoZones)
                        .unwrap()
                        .0;
                assert!(!validation.accepted, "{op}");
                assert_eq!(validation.diagnostics.len(), 1, "{op}");
                assert_eq!(validation.diagnostics[0].code, "PQL_PARSE_SYNTAX", "{op}");
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(30),
                    "{op}: {:?}",
                    started.elapsed()
                );
            }
        })
        .unwrap()
        .join()
        .expect("answered");
}

/// A filter whose tree is tall although the parser never nests deeply:
/// `(NOT ... NOT c) AND c AND ...`, the first operand below every link.
fn negated_chain(nots: usize, links: usize) -> String {
    format!(
        "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE ({}users.age > 3){}",
        "NOT ".repeat(nots),
        " AND users.age > 3".repeat(links)
    )
}

/// `groups` parenthesized chains of `links` links, each the first operand
/// of the next: tall by `groups * (links + 1)` levels.
fn nested_groups(groups: usize, links: usize) -> String {
    let mut condition = "users.age > 3".to_owned();
    for _ in 0..groups {
        condition = format!("({condition}{})", " AND users.age > 3".repeat(links));
    }
    format!("PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {condition}")
}

#[test]
fn a_tree_taller_than_the_limit_is_refused_whatever_its_shape_in_bounded_time() {
    // The tree's height is bounded, not only the parser's nesting: these
    // nest the parser at most about 1,000 levels, and build trees of about
    // 2,000 and 19,500 (the reference gives up near 500 ANDs or 400 NOTs,
    // so this refuses nothing it accepts).
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| {
            let relational = profile(ProfileName::Relational, &[]);
            for query in [
                negated_chain(998, 998),
                nested_groups(20, 975),
                negated_chain(500, 499),
            ] {
                let started = std::time::Instant::now();
                let (validation, _) =
                    validate(&query, &shop(), &relational, &facts(), &NoZones).unwrap();
                assert!(!validation.accepted);
                assert_eq!(validation.diagnostics.len(), 1);
                assert_eq!(validation.diagnostics[0].code, "PQL_PARSE_SYNTAX");
                assert!(
                    started.elapsed() < std::time::Duration::from_secs(30),
                    "{:?}",
                    started.elapsed()
                );
            }
            // At the limit (1,000 levels; one more is refused above), of
            // either shape, accepted.
            for query in [negated_chain(500, 498), nested_groups(10, 98)] {
                let (validation, _) =
                    validate(&query, &shop(), &relational, &facts(), &NoZones).unwrap();
                assert!(validation.accepted, "{:?}", validation.diagnostics);
            }
        })
        .unwrap()
        .join()
        .expect("answered");
}

#[test]
fn resolution_s_capability_diagnostics_come_in_the_reference_s_order() {
    // E11: the two filter capabilities are consulted during resolution, as
    // the reference consults them, so which diagnostic comes first follows
    // the filter's order, and a type error refuses before either is asked.
    // Each answer is the reference's (rust-core-reference b413e9a).
    let neither = profile(
        ProfileName::Relational,
        &[
            (Capability::KeyColumnsInFilters, false),
            (Capability::TimestampColumnsInFilters, false),
        ],
    );
    let no_keys = profile(
        ProfileName::Relational,
        &[(Capability::KeyColumnsInFilters, false)],
    );
    let cases: [(&str, &Profile, &[&str]); 3] = [
        (
            "PREDICT COUNT(orders.* WHERE orders.ts > 2024-01-01 AND orders.user_id = 3, 0, 30, \
             days) FOR EACH users.user_id",
            &neither,
            &[
                "PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION",
                "PQL_MODEL_NO_KEY_COLUMN_FILTER",
                "PQL_MODEL_NO_KEY_COLUMN_REFERENCE",
            ],
        ),
        (
            "PREDICT COUNT(orders.* WHERE orders.user_id = 3 AND orders.ts > 2024-01-01, 0, 30, \
             days) FOR EACH users.user_id",
            &neither,
            &[
                "PQL_MODEL_NO_KEY_COLUMN_FILTER",
                "PQL_MODEL_UNSUPPORTED_COLUMN_OPERATION",
                "PQL_MODEL_NO_KEY_COLUMN_REFERENCE",
            ],
        ),
        (
            "PREDICT COUNT(orders.* WHERE orders.user_id = 3 AND orders.price > 'x', 0, 30, \
             days) FOR EACH users.user_id",
            &no_keys,
            &["PQL_TYPE_CONSTANT_MISMATCH"],
        ),
    ];
    for (query, profile, want) in cases {
        assert_eq!(codes_of(query, profile), want, "{query}");
    }
}

/// Zones from the database bundled with `jiff`.
struct Bundled;

impl sdm_core::Zones for Bundled {
    fn zone(&self, name: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        jiff::tz::TimeZoneDatabase::bundled()
            .get(name)
            .map_err(|_| sdm_core::UnknownZone)
    }
}

#[test]
fn two_digit_years_follow_the_year_in_the_host_s_local_zone() {
    // `sdm-pql/tests/dates.json`'s `local_year`: 2026-12-31T23:30 in New
    // York, 2027-01-01T04:30Z, where the reference's dateutil places a
    // two-digit year by the local year, 2026, not UTC's.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../sdm-pql/tests/dates.json");
    let vectors: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let local = &vectors["local_year"];
    assert_eq!(local["now"], "2027-01-01T04:30:00Z");
    assert_eq!(local["zone"], "America/New_York");
    assert_eq!(local["year"], 2026);
    let at = |zone: Option<sdm_contract::facts::LocalZone>| HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_798_777_800_000_000,
        },
        local_zone: zone,
        ..facts()
    };
    let new_york = at(Some(sdm_contract::facts::LocalZone::Iana(
        "America/New_York".into(),
    )));
    assert_eq!(
        sdm_core::pql::local_year(&new_york, &Bundled).unwrap(),
        2026
    );
    assert_eq!(
        sdm_core::pql::local_year(&at(None), &Bundled).unwrap(),
        2027
    );
    let rows = local["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 8);
    let mut moved = 0;
    let mut packed = 0;
    for row in rows {
        let literal = row[0].as_str().unwrap();
        let query = format!(
            "PREDICT COUNT(orders.* WHERE orders.ts > {literal}, 0, 30, days) FOR EACH \
             users.user_id"
        );
        // From T1.6c a packed date (`760101-01-01`) is refused by name in
        // either year; the reference's reading stays in the row.
        if literal.split(['/', '-', '.']).any(|run| run.len() >= 5) {
            packed += 1;
            for facts in [&new_york, &at(None)] {
                let error = resolve(
                    &query,
                    &shop(),
                    &profile(ProfileName::Relational, &[]),
                    facts,
                    &Bundled,
                )
                .expect_err(literal);
                assert_refused_as_packed(&error, literal);
            }
            continue;
        }
        let read = |facts: &HostFacts| {
            let validation = resolve(
                &query,
                &shop(),
                &profile(ProfileName::Relational, &[]),
                facts,
                &Bundled,
            )
            .unwrap();
            let target = format!("{:?}", validation.query.expect("accepted").target);
            let start = target.find("value: \"").unwrap() + 8;
            target[start..start + 19].to_owned()
        };
        assert_eq!(read(&new_york), row[1].as_str().unwrap(), "{literal}");
        moved += usize::from(read(&at(None)) != read(&new_york));
    }
    // In UTC's year, 76 is 2076, not 1976: the zone decides.
    assert!(moved >= 1, "no literal reads otherwise in UTC's year");
    assert_eq!(packed, 1, "the one packed date of the rows");
}
