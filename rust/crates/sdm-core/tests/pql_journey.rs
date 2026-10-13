// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The queries the first S0 cases and the Phase 1b journey ask
//! (`tests/pql/journey.json`): every inventory query that reaches PQL,
//! against the schema the reference's `Schema.from_source` gave it, gives
//! the reference's answer through the adapter, accepted, or refused with
//! the same error codes and spans.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the journey file"
)]

use std::collections::BTreeMap;
use std::path::Path;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{ProfileSpec, QuerySchema, Severity, Span};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::pql::resolve;
use sdm_core::profiles::Profile;
use serde_json::Value;
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

#[test]
fn every_journey_query_gives_the_reference_answer() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pql/journey.json");
    let journey: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cases = journey["cases"].as_array().unwrap();
    let ids: Vec<&str> = cases.iter().map(|c| c["id"].as_str().unwrap()).collect();
    // Of the inventory's 40 cases, 34 reach PQL; the six the reference's
    // `Schema.from_source` refuses never do, and are recorded by name.
    assert_eq!(cases.len(), 34, "{ids:?}");
    let skipped: Vec<&str> = journey["skipped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_str().unwrap())
        .collect();
    assert_eq!(
        skipped,
        [
            "nested.fixed-array",
            "nested.struct",
            "nested.map",
            "nested.union",
            "empty.fact-table",
            "time.feature-interval",
        ]
    );
    for id in ["base.regression", "refuse.unknown-column"] {
        assert!(ids.contains(&id), "the journey has no {id}");
    }
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let schema: QuerySchema = serde_json::from_value(case["schema"].clone()).unwrap();
        let profile = Profile::resolve(&ProfileSpec {
            name: serde_json::from_value(case["profile"].clone()).unwrap(),
            model: None,
            capabilities: BTreeMap::new(),
        });
        let got = resolve(
            case["query"].as_str().unwrap(),
            &schema,
            &profile,
            &facts(),
            &NoZones,
        )
        .unwrap_or_else(|e| panic!("{id}: {e:?}"));
        assert_eq!(
            got.query.is_some(),
            case["accepted"] == true,
            "{id}: {:?}",
            got.diagnostics
        );
        let want: Vec<(String, Option<Span>)> = case["errors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                (
                    e["code"].as_str().unwrap().to_owned(),
                    serde_json::from_value(e["span"].clone()).unwrap(),
                )
            })
            .collect();
        let found: Vec<(String, Option<Span>)> = got
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .map(|d| (d.code.clone(), d.span))
            .collect();
        assert_eq!(found, want, "{id}");
    }
}
