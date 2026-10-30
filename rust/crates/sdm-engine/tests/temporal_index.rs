// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::tabular::{
    OrderedSampledTabularRows as Outcome, TabularIndexFacts, TabularIndexScalar as Atom,
    TabularIndexValues, TabularOrderError, TabularSortProfile, TemporalUnit,
};
use sdm_engine::index_sort::order_sampled;

fn instant(
    ticks: i64,
    unit: TemporalUnit,
    year: i64,
    month: u8,
    day: u8,
    micros: u64,
    nanosecond: u16,
) -> Atom {
    Atom::Timestamp {
        ticks,
        unit,
        year,
        month,
        day,
        micros,
        nanosecond,
        timezone: None,
        alternate_offset_micros: None,
    }
}
fn facts(values: Vec<Atom>) -> TabularIndexFacts {
    TabularIndexFacts {
        missing: vec![false; values.len()],
        values: TabularIndexValues::TemporalObject { values },
        profile: TabularSortProfile::Scalar,
    }
}
fn assert_order(values: Vec<Atom>, positions: Vec<usize>, already_ordered: bool) {
    assert_eq!(
        order_sampled(&facts(values)).unwrap(),
        Outcome::Sorted {
            positions,
            already_ordered
        }
    );
}

#[test]
fn widened_instants_preserve_nanoseconds_and_full_second_range() {
    let early = instant(
        1_577_836_800_000_000_001,
        TemporalUnit::Nanosecond,
        2020,
        1,
        1,
        0,
        1,
    );
    let late = instant(
        1_577_836_800_000_000_002,
        TemporalUnit::Nanosecond,
        2020,
        1,
        1,
        0,
        2,
    );
    assert_order(vec![late, early], vec![1, 0], false);
    let wide = instant(316_516_204_800, TemporalUnit::Second, 12000, 1, 1, 0, 0);
    let coarse = instant(
        16_725_225_600_000,
        TemporalUnit::Millisecond,
        2500,
        1,
        1,
        0,
        0,
    );
    assert_order(vec![wide, coarse], vec![1, 0], false);
}

#[test]
fn negative_epoch_submicrosecond_remainder_uses_floor() {
    let value = instant(
        -1,
        TemporalUnit::Nanosecond,
        1969,
        12,
        31,
        86_399_999_999,
        999,
    );
    let epoch = instant(0, TemporalUnit::Second, 1970, 1, 1, 0, 0);
    assert_order(vec![epoch, value], vec![1, 0], false);
}

fn fold(timezone: u64) -> Atom {
    Atom::Timestamp {
        ticks: 1_604_208_600,
        unit: TemporalUnit::Second,
        year: 2020,
        month: 11,
        day: 1,
        micros: 5_400_000_000,
        nanosecond: 0,
        timezone: Some(timezone),
        alternate_offset_micros: Some(-18_000_000_000),
    }
}
#[test]
fn same_identity_uses_wall_time_but_distinct_ambiguous_identity_is_not_equal() {
    let same = Atom::ZonedDateTime {
        ordinal: 737_730,
        micros: 5_400_000_000,
        timezone: 1,
        offset_micros: -18_000_000_000,
        alternate_offset_micros: -14_400_000_000,
    };
    assert_order(vec![fold(1), same], vec![0, 1], true);
    let distinct = Atom::ZonedDateTime {
        ordinal: 737_730,
        micros: 5_400_000_000,
        timezone: 2,
        offset_micros: -14_400_000_000,
        alternate_offset_micros: -18_000_000_000,
    };
    assert_order(vec![fold(1), distinct], vec![0, 1], false);
}

#[test]
fn instant_pairs_ignore_same_timezone_wall_fold_equality() {
    let early = fold(1);
    let mut late = early.clone();
    if let Atom::Timestamp {
        ticks,
        alternate_offset_micros,
        ..
    } = &mut late
    {
        *ticks += 3600;
        *alternate_offset_micros = Some(-14_400_000_000);
    }
    assert_order(vec![late, early], vec![1, 0], false);
}

#[test]
fn wide_civil_conversion_is_reached_and_preserves_sort_operand_positions() {
    let wide = instant(316_516_204_800, TemporalUnit::Second, 12000, 1, 1, 0, 0);
    let civil = Atom::DateTime {
        ordinal: 737_425,
        micros: 0,
        offset_micros: None,
    };
    assert!(matches!(
        order_sampled(&facts(vec![wide.clone(), civil])).unwrap(),
        Outcome::Refused {
            reason: TabularOrderError::TemporalYearOutOfRange,
            left: 1,
            right: 0,
            ..
        }
    ));
    assert!(matches!(
        order_sampled(&facts(vec![wide, Atom::Date { ordinal: 737_425 }])).unwrap(),
        Outcome::Refused {
            reason: TabularOrderError::TimestampDateMismatch,
            ..
        }
    ));
}

#[test]
fn malformed_temporal_facts_are_protocol_errors() {
    let wrong_remainder = instant(
        -1,
        TemporalUnit::Nanosecond,
        1969,
        12,
        31,
        86_399_999_999,
        0,
    );
    assert!(order_sampled(&facts(vec![wrong_remainder])).is_err());
    let impossible_date = instant(0, TemporalUnit::Second, 1970, 2, 30, 0, 0);
    assert!(order_sampled(&facts(vec![impossible_date])).is_err());
    let inconsistent_local = instant(0, TemporalUnit::Second, 1970, 1, 2, 0, 0);
    assert!(order_sampled(&facts(vec![inconsistent_local])).is_err());
}

#[test]
fn multi_level_range_error_propagates_with_level_local_owners() {
    let wide = instant(316_516_204_800, TemporalUnit::Second, 12000, 1, 1, 0, 0);
    let civil = Atom::DateTime {
        ordinal: 737_425,
        micros: 0,
        offset_micros: None,
    };
    let index = TabularIndexFacts {
        values: TabularIndexValues::Multi {
            levels: vec![facts(vec![wide.clone(), civil])],
            codes: vec![vec![0, 1]],
        },
        missing: vec![false; 2],
        profile: TabularSortProfile::Stable,
    };
    assert_eq!(
        order_sampled(&index).unwrap(),
        Outcome::Refused {
            reason: TabularOrderError::TemporalYearOutOfRange,
            levels: vec![0],
            left: 1,
            right: 0,
        }
    );
    let mut type_error = index;
    if let TabularIndexValues::Multi { levels, .. } = &mut type_error.values {
        levels[0] = facts(vec![wide, Atom::Date { ordinal: 737_425 }]);
    }
    assert!(matches!(
        order_sampled(&type_error).unwrap(),
        Outcome::Sorted { .. }
    ));
}
