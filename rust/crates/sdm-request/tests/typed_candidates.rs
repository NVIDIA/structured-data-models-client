// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed ranking candidates preserve distinctions and explicitly supplied equality groups.

use sdm_contract::ops::GivenCandidate;
use sdm_request::ranking::{self, Candidate};

/// A typed candidate as plain data.
fn typed(kind: &str, text: &str) -> serde_json::Value {
    serde_json::json!({"typed": {"text": text, "type": kind}})
}

/// A candidate the host holds equal to one given before it, as plain data.
fn repeat() -> serde_json::Value {
    serde_json::json!("repeat")
}

/// The positions `distinct` keeps of the candidates `given`, written as
/// JSON and read as the core reads them.
fn kept(given: &[serde_json::Value]) -> Vec<u32> {
    let written = serde_json::Value::Array(given.to_vec()).to_string();
    let candidates: Vec<GivenCandidate> =
        serde_json::from_str(&written).expect("plain-data candidates read");
    let read: Vec<Candidate> = candidates.iter().map(ranking::read).collect();
    ranking::distinct(&read)
        .into_iter()
        .map(|(position, _)| position)
        .collect()
}

#[test]
fn two_of_one_kind_and_text_are_one_candidate() {
    // Without a `repeat`: the core's rule by kind and text.
    assert_eq!(
        kept(&[
            typed("date", "2023-01-01"),
            typed("date", "2023-01-01"),
            typed("date", "2023-01-02"),
        ]),
        [0, 2]
    );
    assert_eq!(
        kept(&[typed("time", "12:00:00"), typed("time", "12:00:00")]),
        [0]
    );
    assert_eq!(
        kept(&[
            typed("datetime", "2023-01-01 00:00:00"),
            typed("datetime", "2023-01-01 00:00:00"),
        ]),
        [0]
    );
    // Two texts are two, whatever instant they name: grouping them is
    // the host's (`repeat`).
    assert_eq!(
        kept(&[
            typed("datetime_zoned", "2023-01-01 00:00:00+00:00"),
            typed("datetime_zoned", "2023-01-01 01:00:00+01:00"),
        ]),
        [0, 1]
    );
}

#[test]
fn a_date_and_a_timestamp_of_one_day_are_two() {
    assert_eq!(
        kept(&[
            typed("date", "2023-01-01"),
            typed("datetime", "2023-01-01 00:00:00"),
        ]),
        [0, 1]
    );
}

#[test]
fn aware_timestamps_of_one_instant_are_one_candidate() {
    // With `repeat`: the host's grouping, Python's equality.
    assert_eq!(
        kept(&[
            typed("datetime_zoned", "2023-01-01 00:00:00+00:00"),
            repeat(),
            typed("datetime_zoned", "2023-01-02 00:00:00+00:00"),
        ]),
        [0, 2]
    );
    // Aware times of day at two offsets (time(0, UTC) and
    // time(1, +01:00)), and a zone's repeated hour at fold 0 and
    // fold 1 (New York's 2023-11-05 01:30): one each.
    assert_eq!(
        kept(&[
            typed("time", "00:00:00+00:00"),
            repeat(),
            typed("time", "02:00:00+00:00"),
        ]),
        [0, 2]
    );
    assert_eq!(
        kept(&[
            typed("datetime_zoned", "2023-11-05 01:30:00-04:00"),
            repeat(),
            typed("datetime_zoned", "2023-11-06 01:30:00-05:00"),
        ]),
        [0, 2]
    );
}

#[test]
fn a_naive_and_an_aware_timestamp_are_two() {
    assert_eq!(
        kept(&[
            typed("datetime", "2023-01-01 00:00:00"),
            typed("datetime_zoned", "2023-01-01 00:00:00+00:00"),
        ]),
        [0, 1]
    );
    // A zone's wall time at fold 0 beside 05:30 UTC (the instant of its
    // fold 1), which Python holds unequal: two, as the host groups them.
    assert_eq!(
        kept(&[
            typed("datetime_zoned", "2023-11-05 01:30:00-04:00"),
            typed("datetime_zoned", "2023-11-05 05:30:00+00:00"),
            typed("datetime_zoned", "2023-11-06 01:30:00-05:00"),
        ]),
        [0, 1, 2]
    );
}

#[test]
fn a_kind_outside_the_set_is_refused() {
    // fix 7 dropped `first`: a typed candidate carrying it is refused.
    let with_first = serde_json::json!(
        {"typed": {"text": "2023-01-01", "type": "date", "first": 0}}
    );
    assert!(serde_json::from_str::<GivenCandidate>(&with_first.to_string()).is_err());
    for kind in ["week", "Date", ""] {
        let written = typed(kind, "2023-01-01").to_string();
        assert!(
            serde_json::from_str::<GivenCandidate>(&written).is_err(),
            "a typed candidate of kind {kind:?} is read"
        );
    }
    // Control: each kind of the set is read, and a repeat.
    assert!(serde_json::from_str::<GivenCandidate>(&repeat().to_string()).is_ok());
    for kind in ["date", "datetime", "datetime_zoned", "time"] {
        let written = typed(kind, "2023-01-01");
        assert!(
            serde_json::from_str::<GivenCandidate>(&written.to_string()).is_ok(),
            "a typed candidate of kind {kind:?} is refused: {written}"
        );
    }
}
