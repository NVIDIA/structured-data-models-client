// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent timezone observations through the actual host ZoneResolver.
//! Seasonal, historical, post-2038 and far-future cases cover both directions,
//! all pinned zones and every recorded gap/fold refusal.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the vector file and the vendored zone directory"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use jiff::Timestamp;
use jiff::civil::DateTime;
use jiff::tz::{AmbiguousOffset, TimeZone};
use sdm_host::zones::{ZoneResolver, ZoneRoot, ZoneSource};
use serde_json::{Value, json};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn vectors() -> &'static Value {
    static V: OnceLock<Value> = OnceLock::new();
    V.get_or_init(|| {
        let text =
            std::fs::read_to_string(testdata().join("native-admission-vectors.json")).unwrap();
        serde_json::from_str(&text).unwrap()
    })
}

fn tasks(kinds: &[&str]) -> Vec<&'static Value> {
    vectors()["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| kinds.contains(&t["kind"].as_str().unwrap()))
        .collect()
}

fn meta(key: &str) -> &'static str {
    vectors()["meta"][key].as_str().unwrap()
}

/// `Timestamp.isoformat` of a civil datetime: microseconds only when
/// there are any.
fn isoformat(t: DateTime) -> String {
    let base = t.strftime("%Y-%m-%dT%H:%M:%S").to_string();
    let us = t.subsec_nanosecond() / 1000;
    if us == 0 {
        base
    } else {
        format!("{base}.{us:06}")
    }
}

fn naive(text: &str) -> DateTime {
    DateTime::strptime("%Y-%m-%dT%H:%M:%S", text).unwrap()
}

fn utc_datetime(ts: Timestamp) -> DateTime {
    ts.to_zoned(TimeZone::UTC).datetime()
}

// --- zones --------------------------------------------------------------------

fn zone_answer(resolver: &ZoneResolver, task: &Value) -> Value {
    let input = &task["input"];
    let tz = resolver.resolve(input["zone"].as_str().unwrap()).unwrap();
    match task["kind"].as_str().unwrap() {
        "zone_from_utc" => {
            let utc = naive(input["utc"].as_str().unwrap())
                .to_zoned(TimeZone::UTC)
                .unwrap()
                .timestamp();
            let local = utc.to_zoned(tz);
            json!({"local": isoformat(local.datetime()), "offset": local.offset().seconds()})
        }
        _ => {
            // pandas' tz_localize defaults: a gap or a fold is refused.
            let wall = naive(input["wall"].as_str().unwrap());
            match tz.to_ambiguous_zoned(wall).offset() {
                AmbiguousOffset::Unambiguous { offset } => {
                    let utc = wall.to_zoned(TimeZone::fixed(offset)).unwrap().timestamp();
                    json!({"utc": isoformat(utc_datetime(utc))})
                }
                AmbiguousOffset::Gap { .. } => json!({"error": "NonExistentTimeError"}),
                AmbiguousOffset::Fold { .. } => json!({"error": "AmbiguousTimeError"}),
            }
        }
    }
}

/// The refusals of the corpus, by kind. The reference raised `ValueError`
/// for both, so its answer does not say which; these are read off the
/// inputs: the spring transitions of London and Dublin, and the calendar
/// day Samoa skipped, are gaps; their autumn transitions are folds.
const GAPS: [&str; 7] = [
    "l2z/Europe/London/2036-03-30 01:30",
    "l2z/Europe/London/2087-03-30 01:30",
    "l2z/Europe/London/2200-03-30 01:30",
    "l2z/Europe/Dublin/2036-03-30 01:30",
    "l2z/Europe/Dublin/2087-03-30 01:30",
    "l2z/Europe/Dublin/2200-03-30 01:30",
    "l2z/Pacific/Apia/2011-12-30 10:00",
];
const FOLDS: [&str; 6] = [
    "l2z/Europe/London/2036-10-26 01:30",
    "l2z/Europe/London/2087-10-26 01:30",
    "l2z/Europe/London/2200-10-26 01:30",
    "l2z/Europe/Dublin/2036-10-26 01:30",
    "l2z/Europe/Dublin/2087-10-26 01:30",
    "l2z/Europe/Dublin/2200-10-26 01:30",
];

/// The reference's answer, with a refusal named by its kind (a gap or a
/// fold) from the lists above, so a gap answered as a fold, or the
/// reverse, is a difference.
fn expected(task: &Value) -> Value {
    if task["ref"].get("error").is_none() {
        return task["ref"].clone();
    }
    let id = task["id"].as_str().unwrap();
    let error = if GAPS.contains(&id) {
        "NonExistentTimeError"
    } else if FOLDS.contains(&id) {
        "AmbiguousTimeError"
    } else {
        panic!("{id}: a refusal of the corpus is in neither list")
    };
    json!({ "error": error })
}

fn check_zones(resolver: &ZoneResolver) -> usize {
    let mut differences = Vec::new();
    let mut refusals = 0;
    let tasks = tasks(&["zone_from_utc", "zone_to_utc"]);
    for task in &tasks {
        let got = zone_answer(resolver, task);
        let want = expected(task);
        refusals += usize::from(want.get("error").is_some());
        if got != want {
            differences.push(format!("{}: want {want}, got {got}", task["id"]));
        }
    }
    assert!(differences.is_empty(), "{differences:#?}");
    assert_eq!(
        refusals,
        GAPS.len() + FOLDS.len(),
        "every listed refusal is in the corpus"
    );
    tasks.len()
}

#[test]
fn zones_agree_over_the_vendored_2026c_directory() {
    let dir = testdata().join("tzdata-2026c");
    let version = std::fs::read_to_string(dir.join("+VERSION")).unwrap();
    assert_eq!(version.trim(), meta("reference_tzdata"));
    let resolver = ZoneResolver::new([ZoneRoot {
        path: dir.clone(),
        version: Some(version.trim().to_owned()),
    }]);
    assert_eq!(check_zones(&resolver), 229);
    for (name, source) in resolver.sources() {
        let want = if name == "UTC" {
            ZoneSource::Fixed
        } else {
            ZoneSource::HostRoot {
                index: 0,
                path: dir.clone(),
                version: Some(meta("reference_tzdata").to_owned()),
            }
        };
        assert_eq!(source, want, "{name}");
    }
}

#[test]
fn zones_agree_over_the_bundled_database() {
    // The expectations were taken over 2026c; they hold for the bundled
    // version named in the vector file, and a new bundled version is
    // reviewed against them before this line changes.
    assert_eq!(
        ZoneResolver::bundled_version(),
        Some(meta("bundled_tzdata_checked"))
    );
    let resolver = ZoneResolver::new([]);
    assert_eq!(check_zones(&resolver), 229);
    for (name, source) in resolver.sources() {
        if name != "UTC" {
            assert!(matches!(source, ZoneSource::Bundled { .. }), "{name}");
        }
    }
}

#[test]
fn the_vendored_directory_is_only_the_corpus_zones() {
    // Guards the fixture: the zones the corpus names, and +VERSION.
    let zones: BTreeSet<&str> = tasks(&["zone_from_utc", "zone_to_utc"])
        .iter()
        .map(|t| t["input"]["zone"].as_str().unwrap())
        .collect();
    assert_eq!(zones.len(), 18);
    let dir = testdata().join("tzdata-2026c");
    for zone in &zones {
        assert!(dir.join(zone).is_file(), "{zone} is not vendored");
    }
}
