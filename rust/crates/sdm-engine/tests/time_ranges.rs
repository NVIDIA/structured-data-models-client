// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The range-end vectors (design, "Time"): `sdm_engine::time`'s zone
//! answers at both ends of the int64 microsecond and nanosecond ranges, at
//! 9999-12-30T22:00Z (where `jiff`'s timestamps stop), at 9999-12-31 and
//! at 0001-01-01, in every vendored zone (`rust/testdata/tzdata-2026c`),
//! from `time-ranges.json`, recorded by `time_ranges_record.py` with
//! Python's `zoneinfo` over the same files (a protected path, F4). An
//! instant outside the years Python holds is read there by whole 400-year
//! cycles, a rule this test holds the core's own moves to.
//!
//! [`check`] fails closed: a perturbed answer, and evidence missing, empty
//! or retyped, each fail (seeded below).

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the vendored zone directory"
)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use jiff::tz::TimeZone;
use sdm_engine::time::{Ambiguity, Civil, Instant, localize};
use serde_json::{Value, json};

const VECTORS: &str = include_str!("time-ranges.json");

fn zones_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tzdata-2026c")
}

fn zone(name: &str, zones: &mut BTreeMap<String, TimeZone>) -> Result<TimeZone, String> {
    if let Some(zone) = zones.get(name) {
        return Ok(zone.clone());
    }
    let bytes =
        std::fs::read(zones_dir().join(name)).map_err(|_| format!("{name}: not vendored"))?;
    let zone = TimeZone::tzif(name, &bytes).map_err(|_| format!("{name}: not a zone"))?;
    zones.insert(name.into(), zone.clone());
    Ok(zone)
}

fn text<'a>(value: &'a Value, at: &str) -> Result<&'a str, String> {
    value.as_str().ok_or_else(|| format!("{at}: not text"))
}

/// The number of answers each kind compared.
fn check(document: &Value) -> Result<BTreeMap<&'static str, usize>, String> {
    if document["format"] != "sdm-time-ranges/1" || document["tzdata"] != "2026c" {
        return Err("not the 2026c range vectors".into());
    }
    let vectors = document["vectors"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("no vectors")?;
    let mut zones = BTreeMap::new();
    let mut counts = BTreeMap::new();
    let mut differences = Vec::new();
    for (i, vector) in vectors.iter().enumerate() {
        let at = format!("vector {i}");
        let tz = zone(text(&vector["zone"], &at)?, &mut zones)?;
        match text(&vector["kind"], &at)? {
            "to_local" => {
                let nanos: i128 = text(&vector["instant"], &at)?
                    .parse()
                    .map_err(|_| format!("{at}: not an instant"))?;
                let local = Instant::from_nanos(nanos).to_local(&tz);
                let got = json!({"local": local.civil.to_string(), "offset": local.offset_seconds});
                let want = &vector["want"];
                if !want["local"].is_string() || !want["offset"].is_i64() {
                    return Err(format!("{at}: not an answer"));
                }
                if got != *want {
                    differences.push(format!("{at} {}: want {want}, got {got}", vector["zone"]));
                }
                *counts.entry("to_local").or_default() += 1;
            }
            "localize" => {
                let wall = vector["wall"]
                    .as_array()
                    .filter(|w| w.len() == 7)
                    .ok_or_else(|| format!("{at}: not a wall clock"))?;
                let n = |k: usize| {
                    wall[k]
                        .as_i64()
                        .ok_or_else(|| format!("{at}: not a number"))
                };
                let civil = Civil::new(
                    n(0)?,
                    n(1)? as u8,
                    n(2)? as u8,
                    n(3)? as u8,
                    n(4)? as u8,
                    n(5)? as u8,
                    n(6)? as u32,
                )
                .ok_or_else(|| format!("{at}: not a date"))?;
                let got = match localize(&civil, &tz) {
                    Ok(instant) => json!(instant.nanos().to_string()),
                    Err(Ambiguity::Gap) => json!("gap"),
                    Err(Ambiguity::Fold) => json!("fold"),
                };
                let want = &vector["want"];
                if !want.is_string() {
                    return Err(format!("{at}: not an answer"));
                }
                if got != *want {
                    differences.push(format!("{at} {}: want {want}, got {got}", vector["zone"]));
                }
                *counts.entry("localize").or_default() += 1;
            }
            _ => return Err(format!("{at}: a kind not known")),
        }
    }
    if differences.is_empty() {
        Ok(counts)
    } else {
        Err(format!("{differences:#?}"))
    }
}

fn document() -> Value {
    serde_json::from_str(VECTORS).unwrap()
}

#[test]
fn the_range_ends_answer_by_the_zone_rules_in_every_vendored_zone() {
    let counts = check(&document()).unwrap();
    // 18 zones: nine instants and five wall clocks each.
    assert_eq!(counts["to_local"], 18 * 9);
    assert_eq!(counts["localize"], 18 * 5);
}

#[test]
fn the_check_fails_closed() {
    let doc = document();
    let mut perturbed = doc.clone();
    perturbed["vectors"][1]["want"]["offset"] = json!(1);
    assert!(check(&perturbed).is_err());
    let mut perturbed = doc.clone();
    let wall = perturbed["vectors"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|v| v["kind"] == "localize")
        .unwrap();
    wall["want"] = json!("gap");
    assert!(check(&perturbed).is_err());
    for broken in [
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c"}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c", "vectors": []}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026e", "vectors": doc["vectors"]}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c", "vectors": [{"kind": "to_local", "zone": "UTC", "instant": "0"}]}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c", "vectors": [{"kind": "to_local", "zone": "UTC", "instant": 0, "want": {"local": "x", "offset": 0}}]}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c", "vectors": [{"kind": "localize", "zone": "UTC", "wall": [1, 1], "want": "0"}]}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c", "vectors": [{"kind": "other", "zone": "UTC"}]}),
        json!({"format": "sdm-time-ranges/1", "tzdata": "2026c", "vectors": [{"kind": "to_local", "zone": "Mars/Base", "instant": "0", "want": {"local": "x", "offset": 0}}]}),
    ] {
        assert!(check(&broken).is_err(), "{broken}");
    }
}
