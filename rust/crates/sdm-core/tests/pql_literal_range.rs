// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host's date literal range (T1.20 fix round 1): a PQL date literal
//! whose reading lies outside the instants the host declares it can hold
//! (`HostFacts::date_literal_range`) is refused exactly as an unreadable
//! literal (D9) is, with one error diagnostic `PQL_VALUE_INVALID_LITERAL`,
//! no span, and the reference's message. It is decided where the literal's
//! node is built, so it wins over any later node and over every capability
//! and resolution check. With no range declared, every literal the core
//! reads is accepted.
//!
//! The range used here is pandas 2's, `Timestamp.min` to `Timestamp.max`
//! (1677-09-21 00:12:43.145224193 to 2262-04-11 23:47:16.854775807), and
//! each answer is the reference's at b413e9a on pandas 2.2.3.

use std::collections::BTreeMap;

use sdm_contract::facts::{InstantRange, PlatformFacts};
use sdm_contract::ops::{ProfileName, ProfileSpec, QuerySchema, Severity};
use sdm_contract::pql::{NodeKind, PqlLiteral};
use sdm_contract::{CoreWarning, HostFacts, Scalar, TimestampUnit};
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

/// pandas 2's `Timestamp.min.value` and `Timestamp.max.value`.
fn pandas_2() -> InstantRange {
    InstantRange {
        min_ns: -9_223_372_036_854_775_807,
        max_ns: 9_223_372_036_854_775_807,
    }
}

/// The facts of `pql_adapter.rs` (now 2026-01-01 UTC, no local zone),
/// declaring `range`.
fn facts(range: Option<InstantRange>) -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: TimestampUnit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: range,
    }
}

/// The inventory's `shop` tables, as `pql_adapter.rs` describes them.
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

fn profile(name: ProfileName) -> Profile {
    Profile::resolve(&ProfileSpec {
        name,
        model: None,
        capabilities: BTreeMap::new(),
    })
}

/// `PREDICT COUNT(orders.* WHERE <condition>, 0, 30, days) FOR EACH
/// users.user_id`.
fn filtered(condition: &str) -> String {
    format!("PREDICT COUNT(orders.* WHERE {condition}, 0, 30, days) FOR EACH users.user_id")
}

/// The literal as a target comparison, which fine-tuning takes (a model
/// already trained refuses an aggregation of a timestamp column).
fn compared(literal: &str) -> String {
    format!("PREDICT MAX(orders.ts, 0, 30, days) > {literal} FOR EACH users.user_id")
}

/// Each boundary literal and whether pandas 2 holds it. pandas 3 holds
/// them all.
const BOUNDARY: &[(&str, bool)] = &[
    ("0001-01-01", false),
    ("1600-01-01", false),
    ("1677-09-20", false),
    ("1677-09-21", false),
    ("1677-09-21 00:12:43", false),
    ("1677-09-21 00:12:44", true),
    ("1677-09-22", true),
    ("2262-04-10", true),
    ("2262-04-11", true),
    ("2262-04-11 23:47:16", true),
    ("2262-04-11 23:47:17", false),
    ("2262-04-12", false),
    ("2263-01-01", false),
    // Month-first, year 1.
    ("1/2/0001", false),
];

/// Asserts `query` is answered not accepted, with exactly one diagnostic:
/// the error `PQL_VALUE_INVALID_LITERAL`, with no span, naming `literal` in
/// the reference's words.
fn assert_refused_as_unreadable(query: &str, profile: &Profile, facts: &HostFacts, literal: &str) {
    let (validation, _) = validate(query, &shop(), profile, facts, &NoZones)
        .unwrap_or_else(|error| panic!("{query}: refused as {error:?}, not answered"));
    assert!(!validation.accepted, "{query}");
    let [diagnostic] = &validation.diagnostics[..] else {
        panic!("{query}: one diagnostic: {:?}", validation.diagnostics);
    };
    assert_eq!(diagnostic.code, "PQL_VALUE_INVALID_LITERAL", "{query}");
    assert_eq!(diagnostic.severity, Severity::Error, "{query}");
    assert_eq!(diagnostic.span, None, "{query}");
    assert_eq!(
        diagnostic.message,
        format!("{literal} is not a valid instance of time"),
        "{query}"
    );
    // `resolve` answers the same, with no validated query.
    let resolved = resolve(query, &shop(), profile, facts, &NoZones).unwrap();
    assert!(resolved.query.is_none(), "{query}");
    assert_eq!(resolved.diagnostics, validation.diagnostics, "{query}");
}

/// Asserts `query` is accepted with no diagnostics.
fn assert_accepted(query: &str, profile: &Profile, facts: &HostFacts) {
    let (validation, _) = validate(query, &shop(), profile, facts, &NoZones)
        .unwrap_or_else(|error| panic!("{query}: refused as {error:?}"));
    assert!(validation.accepted, "{query}: {:?}", validation.diagnostics);
    assert!(validation.diagnostics.is_empty(), "{query}");
}

#[test]
fn a_boundary_literal_is_answered_as_pandas_2_answers_it_in_a_filter() {
    let relational = profile(ProfileName::Relational);
    let declared = facts(Some(pandas_2()));
    for &(literal, held) in BOUNDARY {
        let query = filtered(&format!("orders.ts > {literal}"));
        if held {
            assert_accepted(&query, &relational, &declared);
        } else {
            assert_refused_as_unreadable(&query, &relational, &declared, literal);
        }
    }
}

#[test]
fn a_boundary_literal_is_answered_as_pandas_2_answers_it_in_a_target_comparison() {
    let fine_tuning = profile(ProfileName::FineTuning);
    let declared = facts(Some(pandas_2()));
    for &(literal, held) in BOUNDARY {
        let query = compared(literal);
        if held {
            assert_accepted(&query, &fine_tuning, &declared);
        } else {
            assert_refused_as_unreadable(&query, &fine_tuning, &declared, literal);
        }
    }
}

#[test]
fn with_no_range_declared_every_boundary_literal_is_accepted() {
    let relational = profile(ProfileName::Relational);
    let fine_tuning = profile(ProfileName::FineTuning);
    let undeclared = facts(None);
    for &(literal, _) in BOUNDARY {
        assert_accepted(
            &filtered(&format!("orders.ts > {literal}")),
            &relational,
            &undeclared,
        );
        assert_accepted(&compared(literal), &fine_tuning, &undeclared);
    }
}

#[test]
fn the_range_s_own_ends_are_held_to_the_nanosecond() {
    // A range of exactly one day, 2024-01-01 inclusive to the last
    // nanosecond before 2024-01-02: its first and last instants a literal
    // can name are held, and the second either side of them is not.
    const DAY_NS: i64 = 86_400 * 1_000_000_000;
    let start = 1_704_067_200 * 1_000_000_000;
    let day = facts(Some(InstantRange {
        min_ns: start,
        max_ns: start + DAY_NS - 1,
    }));
    let relational = profile(ProfileName::Relational);
    for literal in ["2024-01-01", "2024-01-01 00:00:00", "2024-01-01 23:59:59"] {
        assert_accepted(
            &filtered(&format!("orders.ts > {literal}")),
            &relational,
            &day,
        );
    }
    for literal in ["2023-12-31 23:59:59", "2024-01-02", "2024-01-02 00:00:00"] {
        assert_refused_as_unreadable(
            &filtered(&format!("orders.ts > {literal}")),
            &relational,
            &day,
            literal,
        );
    }
}

#[test]
fn an_array_with_one_element_out_of_range_is_refused_by_that_element() {
    let relational = profile(ProfileName::Relational);
    let declared = facts(Some(pandas_2()));
    assert_accepted(
        &filtered("orders.ts IN (1677-09-22, 2262-04-11)"),
        &relational,
        &declared,
    );
    for (array, named) in [
        ("2024-01-01, 1600-01-01", "1600-01-01"),
        ("1600-01-01, 2024-01-01", "1600-01-01"),
        ("2024-01-01, 2263-01-01", "2263-01-01"),
        // The first built of two is named.
        ("2262-04-12, 1600-01-01", "2262-04-12"),
    ] {
        assert_refused_as_unreadable(
            &filtered(&format!("orders.ts IN ({array})")),
            &relational,
            &declared,
            named,
        );
    }
    // The same arrays are answered with no range declared.
    assert_accepted(
        &filtered("orders.ts IN (2262-04-12, 1600-01-01)"),
        &relational,
        &facts(None),
    );
}

#[test]
fn a_literal_out_of_range_wins_over_resolution_and_capability_checks() {
    let declared = facts(Some(pandas_2()));
    let relational = profile(ProfileName::Relational);
    // An unknown column or table, before or after the literal: resolution
    // never runs, so only the literal's error is given.
    for query in [
        filtered("orders.nope > 3 AND orders.ts > 1600-01-01"),
        filtered("orders.ts > 1600-01-01 AND orders.nope > 3"),
        filtered("missing.ts > 1600-01-01"),
        "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE users.nope > \
         1600-01-01"
            .to_owned(),
    ] {
        assert_refused_as_unreadable(&query, &relational, &declared, "1600-01-01");
    }
    // A tabular model refuses a time range before resolution; the literal,
    // refused as it is built, comes first.
    assert_refused_as_unreadable(
        &filtered("orders.ts > 1600-01-01"),
        &profile(ProfileName::Tabular),
        &declared,
        "1600-01-01",
    );
}

#[test]
fn node_order_decides_between_an_out_of_range_and_another_bad_literal() {
    let declared = facts(Some(pandas_2()));
    let relational = profile(ProfileName::Relational);
    // A D9 literal built first is named; an out-of-range one built first
    // is named.
    assert_refused_as_unreadable(
        &filtered("orders.ts > 2023-02-29 AND orders.ts < 1600-01-01"),
        &relational,
        &declared,
        "2023-02-29",
    );
    assert_refused_as_unreadable(
        &filtered("orders.ts > 1600-01-01 AND orders.ts < 2023-02-29"),
        &relational,
        &declared,
        "1600-01-01",
    );
    assert_refused_as_unreadable(
        &filtered("orders.ts > 1600-01-01 AND orders.ts < 2263-01-01"),
        &relational,
        &declared,
        "1600-01-01",
    );
    // An out-of-range literal built before a D8 one ends the parse, so the
    // D8 one is never refused by name.
    assert_refused_as_unreadable(
        &filtered("orders.ts > 1600-01-01 AND orders.ts < 20240101-01-01"),
        &relational,
        &declared,
        "1600-01-01",
    );
    // A D8 literal built first is still refused by name (T1.6c).
    let error = validate(
        &filtered("orders.ts > 20240101-01-01 AND orders.ts < 1600-01-01"),
        &shop(),
        &relational,
        &declared,
        &NoZones,
    )
    .expect_err("a packed date built first is refused by name");
    assert_eq!(
        error.fields.get("rule"),
        Some(&sdm_contract::Detail::text("date_literal_packed"))
    );
}

/// The time constant of `orders.ts > <literal>` read under `facts`, and the
/// warnings the reading gave.
fn read_literal(literal: &str, facts: &HostFacts) -> (Option<String>, Vec<CoreWarning>) {
    let validation = resolve(
        &filtered(&format!("orders.ts > {literal}")),
        &shop(),
        &profile(ProfileName::Relational),
        facts,
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
                literal: PqlLiteral::Time { value, .. },
            } => found = Some(value.clone()),
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
fn a_two_digit_year_or_day_first_literal_in_range_reads_as_before() {
    let declared = facts(Some(pandas_2()));
    for (literal, value) in [
        ("1/2/24", "2024-01-02T00:00:00"),
        ("13/1/24", "2024-01-13T00:00:00"),
        ("01.02.2024", "2024-01-02T00:00:00"),
        ("1677-09-21 00:12:44", "1677-09-21T00:12:44"),
        ("2262-04-11 23:47:16", "2262-04-11T23:47:16"),
    ] {
        let (found, warnings) = read_literal(literal, &declared);
        assert_eq!(found.as_deref(), Some(value), "{literal}");
        assert!(warnings.is_empty(), "{literal}: {warnings:?}");
        assert_eq!(read_literal(literal, &facts(None)).0, found, "{literal}");
    }
    // A day-first literal in range still warns (D4).
    let (found, warnings) = read_literal("31/12/2024", &declared);
    assert_eq!(found.as_deref(), Some("2024-12-31T00:00:00"));
    let codes: Vec<&str> = warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, [DATE_LITERAL_DAY_FIRST]);
}
