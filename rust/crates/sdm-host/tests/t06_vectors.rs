// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The T0.6 date corpus (`rust/testdata/t06-vectors.json`) run against
//! `jiff` (decision A7).
//!
//! - Zone answers go through the host's resolver, once over the vendored
//!   2026c directory and once over the bundled database; every answer must
//!   equal the reference's, and each run first checks that its database is
//!   the version the expectations are tied to.
//! - `DateOffset` answers must equal the reference's.
//! - Unit conversions, single values and PQL literals repeat the T0.6
//!   probe's own reading with `jiff` in place of `chrono`, and must equal the
//!   probe's `chrono` answers; any difference is listed in
//!   `docs/design/rust-core/phase-1/dependencies.md` and named here.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use jiff::civil::DateTime;
use jiff::tz::{AmbiguousOffset, TimeZone};
use jiff::{SignedDuration, Span, Timestamp};
use regex::Regex;
use sdm_host::zones::{ZoneResolver, ZoneRoot, ZoneSource};
use serde_json::{Map, Value, json};

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn vectors() -> &'static Value {
    static V: OnceLock<Value> = OnceLock::new();
    V.get_or_init(|| {
        let text = std::fs::read_to_string(testdata().join("t06-vectors.json")).unwrap();
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

/// `Timestamp.isoformat()` of a civil datetime: microseconds only when
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

/// The reference's answer, with the exception class it raised reduced to
/// "refused", which is what is promised.
fn refused_or(answer: &Value) -> Value {
    if answer.get("error").is_some() {
        json!("refused")
    } else {
        answer.clone()
    }
}

fn check_zones(resolver: &ZoneResolver) -> usize {
    let mut differences = Vec::new();
    let tasks = tasks(&["zone_from_utc", "zone_to_utc"]);
    for task in &tasks {
        let got = zone_answer(resolver, task);
        if refused_or(&got) != refused_or(&task["ref"]) {
            differences.push(format!("{}: want {}, got {got}", task["id"], task["ref"]));
        }
    }
    assert!(differences.is_empty(), "{differences:#?}");
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
    assert_eq!(check_zones(&resolver), 2808);
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
    assert_eq!(check_zones(&resolver), 2808);
    for (name, source) in resolver.sources() {
        if name != "UTC" {
            assert!(matches!(source, ZoneSource::Bundled { .. }), "{name}");
        }
    }
}

// --- DateOffset ---------------------------------------------------------------

fn span(offset: &Map<String, Value>, k: i64) -> Span {
    let g = |n: &str| offset.get(n).and_then(Value::as_i64).unwrap_or(0) * k;
    Span::new()
        .years(g("years"))
        .months(g("months"))
        .weeks(g("weeks"))
        .days(g("days"))
        .hours(g("hours"))
        .minutes(g("minutes"))
        .seconds(g("seconds"))
}

#[test]
fn date_offsets_agree() {
    let tasks = tasks(&["offset"]);
    let mut differences = Vec::new();
    for task in &tasks {
        let base = naive(task["input"]["base"].as_str().unwrap());
        let offset = task["input"]["offset"].as_object().unwrap();
        let apply = |k| {
            base.checked_add(span(offset, k))
                .map(isoformat)
                .unwrap_or_else(|_| "error".into())
        };
        let got = json!({"plus": apply(1), "minus": apply(-1), "times3": apply(3)});
        if got != task["ref"] {
            differences.push(format!("{}: want {}, got {got}", task["id"], task["ref"]));
        }
    }
    assert!(differences.is_empty(), "{differences:#?}");
    assert_eq!(tasks.len(), 208);
}

// --- the T0.6 probe's reading, with jiff in place of chrono -------------------

const ZONE: &str = r"(?:Z|[+-]\d{2}(?::?\d{2})?|[A-Za-z][A-Za-z0-9_+/-]*)";

fn year_first() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(&format!(
            r"^(\d{{4}})[-/](\d{{2}})[-/](\d{{2}})(?:[ T](\d{{2}}):(\d{{2}})(?::(\d{{2}})(?:\.(\d+))?)?(?: ?({ZONE}))?)?$"
        ))
        .unwrap()
    })
}

fn two_part() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(&format!(
            r"^(\d{{1,2}})[-/](\d{{1,2}})[-/](\d{{2,4}})(?:[ T](\d{{1,2}}):(\d{{2}})(?::(\d{{2}})(?:\.(\d+))?)?( ?{ZONE})?)?$"
        ))
        .unwrap()
    })
}

/// A two-digit year moved into the hundred years around the probe's year,
/// pinned here as the host's clock will pin it.
fn pivot(year: i32) -> i32 {
    let now = vectors()["meta"]["chrono"]["today_year"].as_i64().unwrap() as i32;
    let mut y = year + now / 100 * 100;
    if y >= now + 50 {
        y -= 100;
    } else if y < now - 50 {
        y += 100;
    }
    y
}

fn zone_offset(zone: &str) -> Option<i32> {
    let zone = zone.trim();
    if matches!(zone, "Z" | "UTC" | "GMT") {
        return Some(0);
    }
    let sign = match zone.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let digits: String = zone[1..].chars().filter(|c| *c != ':').collect();
    let (h, m) = match digits.len() {
        2 => (digits.parse::<i32>().ok()?, 0),
        4 => (
            digits[..2].parse::<i32>().ok()?,
            digits[2..].parse::<i32>().ok()?,
        ),
        _ => return None,
    };
    if m >= 60 {
        return None;
    }
    Some(sign * (h * 3600 + m * 60))
}

fn micros_of_fraction(frac: Option<&str>) -> Option<i32> {
    match frac {
        None => Some(0),
        Some(f) => {
            let mut s: String = f.chars().take(6).collect();
            while s.len() < 6 {
                s.push('0');
            }
            s.parse().ok()
        }
    }
}

/// The civil datetime, or `None` when it does not exist; the one place the
/// probe asked `chrono` (`NaiveDate::from_ymd_opt`, `from_hms_micro_opt`).
fn build(y: i32, mo: u32, d: u32, h: u32, mi: u32, s: u32, us: i32) -> Option<DateTime> {
    let narrow = |v: u32| i8::try_from(v).ok();
    let date = jiff::civil::Date::new(i16::try_from(y).ok()?, narrow(mo)?, narrow(d)?).ok()?;
    let time = jiff::civil::Time::new(narrow(h)?, narrow(mi)?, narrow(s)?, us * 1000).ok()?;
    Some(date.to_datetime(time))
}

struct Parsed {
    local: DateTime,
    offset: Option<i32>,
}

fn parse_value(raw: &str, dayfirst: bool) -> Option<Parsed> {
    let text = raw.trim();
    let num = |c: Option<regex::Match>| c.map(|m| m.as_str().parse::<u32>().unwrap_or(99));
    let (y, mo, d, c) = if let Some(c) = year_first().captures(text) {
        let y: i32 = c[1].parse().ok()?;
        (y, c[2].parse().ok()?, c[3].parse().ok()?, c)
    } else {
        let c = two_part().captures(text)?;
        let a: u32 = c[1].parse().ok()?;
        let b: u32 = c[2].parse().ok()?;
        let ytext = &c[3];
        let mut y: i32 = ytext.parse().ok()?;
        if ytext.len() <= 2 {
            y = pivot(y);
        }
        let (mo, d) = if dayfirst {
            if b > 12 && a <= 12 { (a, b) } else { (b, a) }
        } else if a > 12 && b <= 12 {
            (b, a)
        } else {
            (a, b)
        };
        (y, mo, d, c)
    };
    let local = build(
        y,
        mo,
        d,
        num(c.get(4)).unwrap_or(0),
        num(c.get(5)).unwrap_or(0),
        num(c.get(6)).unwrap_or(0),
        micros_of_fraction(c.get(7).map(|m| m.as_str()))?,
    )?;
    let offset = match c.get(8) {
        Some(z) => Some(zone_offset(z.as_str())?),
        None => None,
    };
    Some(Parsed { local, offset })
}

/// The epoch as a civil datetime. Instants are counted from it with civil
/// arithmetic, because `jiff::Timestamp` stops at 9999-12-30T22:00:00Z while
/// pandas reads 9999-12-31; see `jiff_timestamps_stop_before_pandas_does`.
const EPOCH: DateTime = DateTime::constant(1970, 1, 1, 0, 0, 0, 0);

fn utc_micros(p: &Parsed) -> Option<i64> {
    let utc = p
        .local
        .checked_sub(SignedDuration::from_secs(i64::from(p.offset.unwrap_or(0))))
        .ok()?;
    i64::try_from(utc.duration_since(EPOCH).as_micros()).ok()
}

fn pql_literal(lit: &str) -> Value {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"^(\d+)([/.-])(\d+)([/.-])(\d+)(?: (\d+):(\d+):(\d+))?$").unwrap()
    });
    let Some(c) = re.captures(lit) else {
        return json!({"refused": "shape"});
    };
    if c[2] != c[4] {
        return json!({"refused": "separators"});
    }
    let f = [&c[1], &c[3], &c[5]];
    if f.iter().any(|x| matches!(x.len(), 6 | 8 | 12 | 14)) {
        return json!({"refused": "packed (D8, not modelled)"});
    }
    let n: Vec<u64> = f
        .iter()
        .map(|x| x.parse::<u64>().unwrap_or(u64::MAX))
        .collect();
    let long = f.iter().filter(|x| x.len() >= 3).count();
    if long >= 2 {
        return json!({"refused": "two long fields"});
    }
    let (yi, mi, di, warn) = if n[0] > 31 || f[0].len() >= 3 {
        (0, 1, 2, false)
    } else if n[0] > 12 {
        (2, 1, 0, true)
    } else {
        (2, 0, 1, false)
    };
    let mut year = n[yi] as i64;
    if long == 0 && year < 100 {
        year = pivot(year as i32) as i64;
    }
    if year > 9999 {
        return json!({"refused": "year"});
    }
    if year == 0 && !(yi == 0 && f[0] == "0000") {
        return json!({"refused": "year 0"});
    }
    let (h, m, s) = match c.get(6) {
        Some(_) => (
            c[6].parse::<u32>().unwrap_or(99),
            c[7].parse::<u32>().unwrap_or(99),
            c[8].parse::<u32>().unwrap_or(99),
        ),
        None => (0, 0, 0),
    };
    let (mo, d) = (
        u32::try_from(n[mi]).unwrap_or(u32::MAX),
        u32::try_from(n[di]).unwrap_or(u32::MAX),
    );
    let Some(t) = build(year as i32, mo, d, h, m, s, 0) else {
        return json!({"refused": "invalid"});
    };
    let warned =
        warn && f[yi].len() == 4 && f.iter().enumerate().all(|(i, x)| i == yi || x.len() <= 2);
    json!({"value": t.strftime("%Y-%m-%dT%H:%M:%S").to_string(), "warned": warned})
}

fn unit_ns(ns: i64) -> Value {
    let secs = ns.div_euclid(1_000_000_000);
    let t = utc_datetime(Timestamp::from_nanosecond(i128::from(ns)).unwrap());
    let wire = format!(
        "{}.{:06}Z",
        t.strftime("%Y-%m-%dT%H:%M:%S"),
        t.subsec_nanosecond() / 1000
    );
    let back = utc_datetime(Timestamp::from_second(secs).unwrap());
    json!({"seconds_floor": secs, "wire": wire, "seconds_back": isoformat(back)})
}

fn unit_ms(ms: i64) -> Value {
    match EPOCH.checked_add(SignedDuration::from_millis(ms)) {
        Ok(t) => json!(isoformat(t)),
        Err(_) => Value::Null,
    }
}

/// Task ids where `jiff` and `chrono` differ in the probe's reading. Each
/// one is listed, with its reason, in dependencies.md.
const CHRONO_DIFFERENCES: &[&str] = &[];

#[test]
fn the_probe_reading_on_jiff_matches_chrono() {
    let tasks = tasks(&["unit", "unit_ms", "parse_value", "pql_literal"]);
    let mut differences = BTreeSet::new();
    let mut report = Vec::new();
    for task in &tasks {
        let input = &task["input"];
        let got = match task["kind"].as_str().unwrap() {
            "unit" => unit_ns(input["ns"].as_i64().unwrap()),
            "unit_ms" => unit_ms(input["ms"].as_i64().unwrap()),
            "parse_value" => {
                match parse_value(
                    input["value"].as_str().unwrap(),
                    input["dayfirst"].as_bool().unwrap(),
                ) {
                    Some(p) => utc_micros(&p).map_or(Value::Null, |us| json!(us)),
                    None => Value::Null,
                }
            }
            _ => pql_literal(input["literal"].as_str().unwrap()),
        };
        if got != task["chrono"] {
            let id = task["id"].as_str().unwrap();
            differences.insert(id);
            report.push(format!("{id}: chrono {}, jiff {got}", task["chrono"]));
        }
    }
    assert_eq!(tasks.len(), 16 + 8 + 2068 + 45);
    let expected: BTreeSet<&str> = CHRONO_DIFFERENCES.iter().copied().collect();
    assert_eq!(differences, expected, "{report:#?}");
}

#[test]
fn jiff_timestamps_stop_before_pandas_does() {
    // The three corpus answers a Timestamp cannot give, which the civil
    // arithmetic above gives exactly as chrono and pandas do.
    assert!(Timestamp::from_millisecond(253_402_300_799_999).is_err());
    assert!(
        DateTime::constant(9999, 12, 31, 0, 0, 0, 0)
            .to_zoned(TimeZone::UTC)
            .is_err()
    );
    assert_eq!(Timestamp::MAX.to_string(), "9999-12-30T22:00:00.999999999Z");
    assert_eq!(
        unit_ms(253_402_300_799_999),
        json!("9999-12-31T23:59:59.999000")
    );
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
