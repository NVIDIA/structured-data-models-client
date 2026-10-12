// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapter's own rules: a model that cannot answer a time range refuses
//! before resolution; the two filter capabilities reach resolution; date
//! literals are read by D1 to D9, packed dates and long fields included,
//! and a day-first one warns (D4); and an accepted query is handed on in
//! the validated-query format.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: times the bounded-time refusals against a wall clock"
)]

use std::collections::BTreeMap;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Capability, ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::pql::{Aggregation, EntityClause, NodeKind, PqlLiteral, QueryMode};
use sdm_contract::{CoreWarning, Detail, HostFacts, Scalar, Unit};
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

#[test]
fn packed_dates_and_long_fields_are_read_by_d8() {
    // Register DT-9's two queries, refused as unsupported before T1.6b,
    // and D8's examples, each as the reference reads it (b413e9a).
    for (literal, value, offset) in [
        ("20240101-01-01", "2024-01-01T01:00:00", Some(-3600)),
        ("2024-01-01 00000:00:00", "2024-01-01T00:00:00", None),
        ("20240101/00/123456", "2024-01-01T12:34:56", None),
        ("230203-06-09", "2003-02-23T06:00:00", Some(-32_400)),
        ("00001/1/2", "0001-01-02T00:00:00", None),
    ] {
        let (found, warnings) = read_literal(literal);
        assert_eq!(found, Some((value.to_owned(), offset)), "{literal}");
        assert!(warnings.is_empty(), "{literal}");
    }
    // A long field that names no date is refused as the reference refuses
    // it: PQL_VALUE_INVALID_LITERAL, never unsupported.
    let refused = codes_of(
        "PREDICT COUNT(orders.* WHERE orders.ts > 20240101-01-24, 0, 30, days) FOR EACH \
         users.user_id",
        &profile(ProfileName::Relational, &[]),
    );
    assert_eq!(refused, ["PQL_VALUE_INVALID_LITERAL"]);
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
    for row in rows {
        let literal = row[0].as_str().unwrap();
        let query = format!(
            "PREDICT COUNT(orders.* WHERE orders.ts > {literal}, 0, 30, days) FOR EACH \
             users.user_id"
        );
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
}
