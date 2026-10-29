// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual native statistics retain the original fixed constructor ordering.

use sdm_contract::ops::{Dtype, Stype};
use sdm_schema::frames::native_column::Constructor;

#[test]
fn identity_and_fixed_types_do_not_read_future_samples() {
    for (name, dtype, expected) in [
        ("user_id", Dtype::String, Stype::Id),
        ("event", Dtype::Date, Stype::Timestamp),
        ("flag", Dtype::Bool, Stype::Categorical),
        ("age", Dtype::Int, Stype::Numerical),
    ] {
        let mut value = Constructor::new(name.into(), dtype);
        assert_eq!(value.meaning(), Some(expected));
        assert!(!value.timestamp_sample());
        assert!(!value.separated_sample());
        assert!(!value.distinct_sample());
        assert!(value.timestamp_counts(1, 1).is_err());
        assert!(value.distinct_count(1).is_err());
    }
}

#[test]
fn reached_timestamp_counts_short_circuit_later_native_reads() {
    let mut named = Constructor::new("event_ts".into(), Dtype::String);
    assert!(named.timestamp_sample());
    named.timestamp_counts(7, 10).unwrap();
    assert_eq!(named.meaning(), Some(Stype::Timestamp));
    assert!(!named.separated_sample());
    let mut unnamed = Constructor::new("part".into(), Dtype::String);
    unnamed.timestamp_counts(7, 10).unwrap();
    assert!(unnamed.separated_sample());
    assert!(unnamed.timestamp_counts(1, 1).is_err());
    let mut malformed = Constructor::new("part".into(), Dtype::String);
    assert!(malformed.timestamp_counts(2, 1).is_err());
}

#[test]
fn native_joined_text_selects_first_separator_on_ties_then_real_split_counts() {
    let mut value = Constructor::new("part".into(), Dtype::String);
    value.timestamp_counts(0, 10).unwrap();
    assert!(value.separated_sample());
    assert_eq!(value.separator(&[124, 59, 124, 59]).unwrap(), Some('|'));
    assert!(value.distinct_count(1).is_err());
    value.separated_counts(16, 10).unwrap();
    assert_eq!(value.meaning(), Some(Stype::Multicategorical));
    assert!(value.separated_counts(1, 1).is_err());
}

#[test]
fn categorical_distinct_short_circuit_does_not_require_a_length_read() {
    let mut value = Constructor::new("part".into(), Dtype::String);
    value.timestamp_counts(0, 0).unwrap();
    assert_eq!(value.separator(&[65, 66]).unwrap(), None);
    assert!(value.distinct_sample());
    assert!(!value.distinct_count(19).unwrap());
    assert_eq!(value.meaning(), Some(Stype::Categorical));
    assert!(value.population_count(20).is_err());
    let mut value = Constructor::new("part".into(), Dtype::String);
    value.timestamp_counts(0, 0).unwrap();
    value.separator(&[]).unwrap();
    assert!(value.distinct_count(20).unwrap());
    assert!(value.population_count(19).is_err());
    value.population_count(40).unwrap();
    assert_eq!(value.meaning(), Some(Stype::Categorical));
}

#[test]
fn string_binary_and_numeric_share_thresholds_keep_their_own_defaults() {
    for (dtype, rows, expected) in [
        (Dtype::String, 39, Stype::Text),
        (Dtype::Binary, 40, Stype::Categorical),
        (Dtype::Int, 399, Stype::Numerical),
        (Dtype::Int, 400, Stype::Categorical),
        (Dtype::Timedelta, 400, Stype::Categorical),
    ] {
        let mut value = Constructor::new("part".into(), dtype);
        if value.timestamp_sample() {
            value.timestamp_counts(0, 0).unwrap();
        }
        if value.separated_sample() {
            value.separator(&[]).unwrap();
        }
        if value.element_sample() {
            value.element_count(5).unwrap();
        }
        assert!(value.distinct_count(20).unwrap());
        value.population_count(rows).unwrap();
        assert_eq!(value.meaning(), Some(expected));
    }
}

#[test]
fn fixed_constructor_equivalence_normalizes_both_observation_and_source_aliases() {
    use sdm_schema::frames::native_column::equivalent;
    for source in [Dtype::Date, Dtype::Time] {
        assert!(equivalent(source, Dtype::Date));
    }
    assert!(equivalent(Dtype::Int64, Dtype::Int));
    assert!(equivalent(Dtype::Float32, Dtype::Float));
    assert!(!equivalent(Dtype::Date, Dtype::String));
    assert!(!equivalent(Dtype::Int, Dtype::String));
}
