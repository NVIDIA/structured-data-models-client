// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen frame nomination, fresh adoption, and later time observations.

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int64Array, RecordBatch,
    TimestampMicrosecondArray,
};
use sdm_contract::data::Relation;
use sdm_contract::ops::Stype;
use sdm_contract::{Detail, Scalar, Sentence};
use sdm_schema::frames::{adopt, choose_time, nominate};
use std::sync::Arc;

fn rows(key: ArrayRef, feature: ArrayRef, target: ArrayRef) -> RecordBatch {
    RecordBatch::try_from_iter([("user_id", key), ("feature", feature), ("target", target)])
        .unwrap()
}
fn integer(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}
fn original() -> RecordBatch {
    rows(
        integer(
            (0..100)
                .map(|n| if n == 0 { None } else { Some(n) })
                .collect(),
        ),
        integer((0..100).map(Some).collect()),
        integer((100..200).map(Some).collect()),
    )
}

#[test]
fn nomination_and_original_meaning_survive_fresh_adoption_and_later_changes() {
    let original = original();
    let (nomination, missing) = nominate(
        "instance",
        &Relation::named("instance"),
        &original.schema(),
        std::slice::from_ref(&original),
    )
    .unwrap();
    assert_eq!(nomination.nominated_key, vec!["user_id"]);
    assert_eq!(
        missing.iter().map(|w| w.sentence).collect::<Vec<_>>(),
        vec![Sentence::KeyNoValueColumn]
    );
    assert_eq!(nomination.table.columns[0].stype, Stype::Numerical);
    for (mode, duplicates, warning) in [
        ("create", 1, Some(Sentence::FrameKeyRepeatsComposite)),
        ("no-remedy", 1, Some(Sentence::FrameKeyRepeats)),
        ("unusable", 1, Some(Sentence::FrameKeyRepeatsUnusable)),
        ("bool", 98, Some(Sentence::FrameKeyRepeatsComposite)),
        ("empty", 0, None),
        ("absent", 0, None),
    ] {
        let mut keys: Vec<_> = (0..100)
            .map(|n| if n == 0 { None } else { Some(n) })
            .collect();
        keys[2] = Some(1);
        let key = if mode == "bool" {
            Arc::new(BooleanArray::from(
                (0..100).map(|n| n % 2 == 0).collect::<Vec<_>>(),
            )) as ArrayRef
        } else {
            integer(keys)
        };
        let mut feature: Vec<_> = (0..100).map(Some).collect();
        let mut target: Vec<_> = (100..200).map(Some).collect();
        if mode == "no-remedy" {
            feature[2] = feature[1];
            target[2] = target[1];
        }
        if mode == "unusable" {
            feature[0] = None;
            target[0] = None;
        }
        let fresh = rows(key, integer(feature), integer(target));
        let fresh = match mode {
            "empty" => fresh.slice(0, 0),
            "absent" => fresh.project(&[1, 2]).unwrap(),
            _ => fresh,
        };
        let (adopted, advice) = adopt(
            nomination.clone(),
            &fresh.schema(),
            std::slice::from_ref(&fresh),
        )
        .unwrap();
        assert_eq!(adopted.table.primary_key, vec!["user_id"]);
        assert_eq!(
            adopted.table.duplicate_keys,
            Some(Scalar::UInt64(duplicates)),
            "{mode}"
        );
        assert_eq!(adopted.table.columns[0].stype, Stype::Id);
        assert_eq!(
            adopted.table.columns[0].dtype,
            nomination.table.columns[0].dtype
        );
        assert_eq!(advice.first().map(|w| w.sentence), warning, "{mode}");
        if mode == "bool" {
            assert_eq!(
                advice[0].fields["example"],
                Detail::List(vec![Scalar::Bool(true)])
            );
        }
        // A repeat-warning handler can repair values. The adopted count is
        // already public and remains fixed while the later phase reads time.
        let completed =
            choose_time(adopted, &original.schema(), std::slice::from_ref(&original)).unwrap();
        assert_eq!(
            completed.table.duplicate_keys,
            Some(Scalar::UInt64(duplicates))
        );
    }
}

#[test]
fn only_earlier_timestamp_candidates_are_read_after_repeat_advice() {
    let original = original();
    let mut fields = original.columns().to_vec();
    fields.push(Arc::new(TimestampMicrosecondArray::from_iter_values(
        (0..100).map(|n| 1_700_000_000_000_000 + n),
    )));
    let stamped = RecordBatch::try_from_iter([
        ("user_id", fields[0].clone()),
        ("feature", fields[1].clone()),
        ("target", fields[2].clone()),
        ("created_at", fields[3].clone()),
    ])
    .unwrap();
    let (nomination, _) = nominate(
        "instance",
        &Relation::named("instance"),
        &stamped.schema(),
        std::slice::from_ref(&stamped),
    )
    .unwrap();
    let (adopted, _) = adopt(
        nomination,
        &stamped.schema(),
        std::slice::from_ref(&stamped),
    )
    .unwrap();
    let nulls = RecordBatch::try_from_iter([
        ("user_id", fields[0].clone()),
        ("feature", fields[1].clone()),
        ("target", fields[2].clone()),
        (
            "created_at",
            Arc::new(TimestampMicrosecondArray::from(vec![None; 100])) as ArrayRef,
        ),
    ])
    .unwrap();
    let result = choose_time(
        adopted.clone(),
        &nulls.schema(),
        std::slice::from_ref(&nulls),
    )
    .unwrap();
    assert_eq!(result.table.time_column, None);
    let numeric = RecordBatch::try_from_iter([
        ("user_id", fields[0].clone()),
        ("feature", fields[1].clone()),
        ("target", fields[2].clone()),
        ("created_at", integer((0..100).map(Some).collect())),
    ])
    .unwrap();
    let result = choose_time(
        adopted.clone(),
        &numeric.schema(),
        std::slice::from_ref(&numeric),
    )
    .unwrap();
    assert_eq!(result.table.time_column.as_deref(), Some("created_at"));
    let error =
        choose_time(adopted, &original.schema(), std::slice::from_ref(&original)).unwrap_err();
    assert_eq!(error.sentence, Sentence::DirectSchemaMissing);
    let (plain, _) = nominate(
        "instance",
        &Relation::named("instance"),
        &original.schema(),
        std::slice::from_ref(&original),
    )
    .unwrap();
    let (plain, _) = adopt(plain, &original.schema(), std::slice::from_ref(&original)).unwrap();
    let result = choose_time(plain, &stamped.schema(), std::slice::from_ref(&stamped)).unwrap();
    assert_eq!(result.table.time_column, None);
}

#[test]
fn fresh_time_candidates_read_numeric_values_at_their_physical_float_width() {
    for two in [false, true] {
        for narrow in [false, true] {
            for nullable in [false, true] {
                let key = integer((0..100).map(|n| (n != 0).then_some(n)).collect());
                let mut fields = vec![
                    ("user_id", key.clone()),
                    (
                        "event_a",
                        Arc::new(TimestampMicrosecondArray::from_iter_values(
                            (0..100).map(|n| 1_700_000_000_000_000 + n * 86_400_000_000),
                        )) as ArrayRef,
                    ),
                ];
                if two {
                    fields.push((
                        "event_b",
                        Arc::new(TimestampMicrosecondArray::from_iter_values(
                            (0..100).map(|n| 1_800_000_000_000_000 + n * 86_400_000_000),
                        )) as ArrayRef,
                    ));
                }
                let original = RecordBatch::try_from_iter(fields).unwrap();
                let (nomination, missing) = nominate(
                    "instance",
                    &Relation::named("instance"),
                    &original.schema(),
                    std::slice::from_ref(&original),
                )
                .unwrap();
                assert_eq!(missing[0].sentence, Sentence::KeyNoValueColumn);
                let offsets = (0..100)
                    .map(|n| (!nullable || n != 0).then_some(1.7e18 + f64::from(n) * 86400e9));
                let earlier: ArrayRef = if narrow {
                    Arc::new(Float32Array::from(
                        offsets.map(|n| n.map(|v| v as f32)).collect::<Vec<_>>(),
                    ))
                } else {
                    Arc::new(Float64Array::from(offsets.collect::<Vec<_>>()))
                };
                let mut fields = vec![("user_id", key), ("event_a", earlier)];
                if two {
                    fields.push((
                        "event_b",
                        Arc::new(Float64Array::from_iter_values(
                            (0..100).map(|n| 1.8e18 + f64::from(n) * 86400e9),
                        )) as ArrayRef,
                    ));
                }
                let fresh = RecordBatch::try_from_iter(fields).unwrap();
                let (adopted, _) =
                    adopt(nomination, &fresh.schema(), std::slice::from_ref(&fresh)).unwrap();
                let selected = choose_time(
                    adopted,
                    &fresh.schema(),
                    &[fresh.slice(0, 37), fresh.slice(37, 63)],
                )
                .unwrap();
                assert_eq!(selected.table.time_column.as_deref(), Some("event_a"));
            }
        }
    }
}
