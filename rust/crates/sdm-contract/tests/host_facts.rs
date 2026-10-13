// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host facts' date literal range (T1.20 fix round 1) on the wire:
//! `"date_literal_range": {"min_ns": ..., "max_ns": ...}` when declared, no
//! key when not, and an absent key reads as none, alone and in a run's
//! start.

use sdm_contract::data::AttachmentId;
use sdm_contract::facts::{InstantRange, LocalZone, PlatformFacts};
use sdm_contract::ops::{Operation, ProfileName, ProfileSpec, ValidateQuery};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, HostFacts, Scalar, Unit};
use serde_json::{Value, json};

fn facts(range: Option<InstantRange>) -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_790_000_000_000_000,
        },
        local_zone: Some(LocalZone::Iana("Europe/Berlin".into())),
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: true,
        },
        date_literal_range: range,
    }
}

/// pandas 2's `Timestamp.min.value` and `Timestamp.max.value`.
fn pandas_2() -> InstantRange {
    InstantRange {
        min_ns: -9_223_372_036_854_775_807,
        max_ns: 9_223_372_036_854_775_807,
    }
}

fn start(facts: HostFacts) -> ControlMessage {
    ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts,
        operation: Operation::ValidateQuery(ValidateQuery {
            query: AttachmentId(1),
            schema: AttachmentId(2),
            profile: ProfileSpec {
                name: ProfileName::Relational,
                model: None,
                capabilities: Default::default(),
            },
        }),
    }))
}

#[test]
fn a_declared_range_is_written_as_its_two_instants_and_read_back() {
    let declared = facts(Some(pandas_2()));
    let written = serde_json::to_value(&declared).unwrap();
    assert_eq!(
        written["date_literal_range"],
        json!({"min_ns": -9_223_372_036_854_775_807_i64, "max_ns": 9_223_372_036_854_775_807_i64})
    );
    let back: HostFacts = serde_json::from_value(written).unwrap();
    assert_eq!(back, declared);
    // The instants survive as text exactly, past f64's precision.
    let text = serde_json::to_string(&declared).unwrap();
    assert!(text.contains(r#""min_ns":-9223372036854775807"#), "{text}");
    assert!(text.contains(r#""max_ns":9223372036854775807"#), "{text}");
    assert_eq!(serde_json::from_str::<HostFacts>(&text).unwrap(), declared);
}

#[test]
fn no_range_writes_no_key_and_an_absent_key_reads_as_none() {
    let undeclared = facts(None);
    let written = serde_json::to_value(&undeclared).unwrap();
    let Value::Object(fields) = &written else {
        panic!("an object: {written}");
    };
    assert!(!fields.contains_key("date_literal_range"), "{written}");
    let back: HostFacts = serde_json::from_value(written.clone()).unwrap();
    assert_eq!(back, undeclared);
    assert_eq!(back.date_literal_range, None);
    // An explicit null reads as none too.
    let mut with_null = written;
    with_null["date_literal_range"] = Value::Null;
    let back: HostFacts = serde_json::from_value(with_null).unwrap();
    assert_eq!(back.date_literal_range, None);
}

#[test]
fn a_run_s_start_carries_the_range_or_omits_it() {
    for range in [Some(pandas_2()), None] {
        let declared = range.is_some();
        let message = start(facts(range));
        let written = serde_json::to_value(&message).unwrap();
        assert_eq!(
            written["facts"].get("date_literal_range").is_some(),
            declared,
            "{written}"
        );
        let back: ControlMessage = serde_json::from_value(written).unwrap();
        assert_eq!(back, message);
    }
}

#[test]
fn a_range_with_a_missing_or_unknown_field_is_refused() {
    let mut written = serde_json::to_value(facts(None)).unwrap();
    for range in [
        json!({"min_ns": 0}),
        json!({"max_ns": 0}),
        json!({"min_ns": 0, "max_ns": 1, "unit": "ns"}),
        json!({"min_ns": "0", "max_ns": 1}),
    ] {
        written["date_literal_range"] = range.clone();
        assert!(
            serde_json::from_value::<HostFacts>(written.clone()).is_err(),
            "{range}"
        );
    }
}
