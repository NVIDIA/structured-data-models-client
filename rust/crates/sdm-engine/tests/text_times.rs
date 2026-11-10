// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text-time format, ordering and inference policies with independent expected values.
use sdm_engine::time::text::{Reader, Share, Unread};

#[test]
fn supported_date_time_spellings_have_independent_epoch_values() {
    // 2024-01-15 midnight UTC, computed from the Gregorian calendar.
    const MIDNIGHT: i128 = 1_705_276_800_000_000_000;
    let reader = Reader::new(2026);
    for date in [
        "2024-01-15",
        "2024/01/15",
        "01/15/2024",
        "15/01/2024",
        "1/15/24",
        "15/1/24",
        "01-15-2024",
        "15-01-2024",
    ] {
        for (time, nanos) in [
            ("", 0),
            ("T10:30", 37_800_000_000_000),
            (" 10:30:15", 37_815_000_000_000),
            (" 10:30:15.123456789", 37_815_123_456_789),
        ] {
            for (zone, offset) in [
                ("", 0),
                ("Z", 0),
                (" UTC", 0),
                ("GMT", 0),
                ("+01", 3_600_000_000_000),
                ("+0130", 5_400_000_000_000),
                (" -08:00", -28_800_000_000_000),
            ] {
                if time.is_empty() && !zone.is_empty() {
                    continue;
                }
                let text = format!("{date}{time}{zone}");
                let values = [None, Some(text.as_str()), Some("  ")];
                for utc in [false, true] {
                    let got = reader
                        .column(&values, utc)
                        .unwrap_or_else(|e| panic!("{text}: {e:?}"));
                    assert_eq!(
                        got.into_iter()
                            .map(|v| v.map(|v| v.nanos()))
                            .collect::<Vec<_>>(),
                        [None, Some(MIDNIGHT + nanos - offset), None],
                        "{text}"
                    );
                }
                assert_eq!(
                    reader
                        .each(&values)
                        .unwrap()
                        .into_iter()
                        .map(|v| v.map(|v| v.nanos()))
                        .collect::<Vec<_>>(),
                    [None, Some(MIDNIGHT + nanos - offset), None]
                );
                assert_eq!(
                    reader.share(&[Some(&text)]),
                    Ok(Share {
                        fewest: 1,
                        listed: 1,
                        most: 1,
                        unshaped: 0
                    }),
                    "{text}"
                );
            }
        }
    }
}

#[test]
fn whole_column_evidence_decides_order_before_reading_any_row() {
    let reader = Reader::new(2026);
    for values in [
        [Some("02/01/2024"), Some("15/03/2024")],
        [Some("15/03/2024"), Some("02/01/2024")],
    ] {
        let expected = if values[0] == Some("02/01/2024") {
            vec![1_704_153_600_000_000_000, 1_710_460_800_000_000_000]
        } else {
            vec![1_710_460_800_000_000_000, 1_704_153_600_000_000_000]
        };
        // Day-first means January 2, not February 1.
        for utc in [false, true] {
            assert_eq!(
                reader
                    .column(&values, utc)
                    .unwrap()
                    .into_iter()
                    .map(|v| v.unwrap().nanos())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
    for values in [
        [Some("03/15/99"), Some("15/11/00")],
        [Some("01/02/2024"), None],
    ] {
        assert_eq!(reader.column(&values, false), Err(Unread::Order));
    }
    assert_eq!(
        reader.column(&[Some("2024-01-05"), Some("2021-13-45")], false),
        Err(Unread::Outside)
    );
}

#[test]
fn inference_shares_distinguish_invalid_shapes_order_and_missing_values() {
    let reader = Reader::new(2026);
    for (values, expected) in [
        (
            vec![None, Some(""), Some("  ")],
            Share {
                fewest: 0,
                listed: 0,
                most: 0,
                unshaped: 0,
            },
        ),
        (
            vec![Some("03/15/99"), Some("15/11/00")],
            Share {
                fewest: 1,
                listed: 2,
                most: 2,
                unshaped: 0,
            },
        ),
        (
            vec![Some("2024-01-05"), Some("2021-13-45"), Some("not a date")],
            Share {
                fewest: 1,
                listed: 1,
                most: 2,
                unshaped: 1,
            },
        ),
        (
            vec![Some("01/02/2024")],
            Share {
                fewest: 1,
                listed: 1,
                most: 1,
                unshaped: 0,
            },
        ),
    ] {
        assert_eq!(reader.share(&values), Ok(expected), "{values:?}");
    }
    assert_eq!(
        reader.day_first_format(&[Some("15-03-24 10:30 +01:00")]),
        Some("%d-%m-%y %H:%M %z".into())
    );
    assert_eq!(
        reader.day_first_format(&[Some("15/03/2024")]),
        Some("%d/%m/%Y".into())
    );
    assert_eq!(reader.day_first_format(&[Some("01/02/2024")]), None);
}

#[test]
fn timezone_normalization_preserves_null_positions_and_fractional_seconds() {
    let reader = Reader::new(2026);
    let values = [
        None,
        Some("1970-01-01 01:00+01:00"),
        Some("1970-01-01 00:00:00.000000001"),
        Some(""),
    ];
    assert_eq!(reader.column(&values, false), Err(Unread::MixedZones));
    assert_eq!(reader.share(&values), Err(Unread::MixedZones));
    assert_eq!(
        reader
            .column(&values, true)
            .unwrap()
            .into_iter()
            .map(|v| v.map(|v| v.nanos()))
            .collect::<Vec<_>>(),
        [None, Some(0), Some(1), None]
    );
    assert_eq!(
        reader.column(
            &[
                Some("1970-01-01 00:00+01:00"),
                Some("1970-01-01 00:00-01:00")
            ],
            false
        ),
        Err(Unread::MixedZones)
    );
}

#[test]
fn clock_pivot_and_calendar_range_use_wide_instants() {
    for (clock, text, expected) in [
        (2026, "1/15/75", 3_314_736_000_000_000_000i128),
        (2026, "1/15/76", 190_512_000_000_000_000),
        (2027, "1/15/76", 3_346_272_000_000_000_000),
        (2026, "1/15/124", -58_252_953_600_000_000_000),
        (2026, "0001-01-01", -62_135_596_800_000_000_000),
        (2026, "9999-01-01", 253_370_764_800_000_000_000),
    ] {
        assert_eq!(
            Reader::new(clock).column(&[Some(text)], false).unwrap()[0]
                .unwrap()
                .nanos(),
            expected,
            "{text}"
        );
    }
}

#[test]
fn whitespace_missing_and_refusal_precedence_are_explicit() {
    let reader = Reader::new(2026);
    let values = [None, Some("\u{3000}"), Some("\u{2003}1970-01-01\u{a0}")];
    assert_eq!(
        reader
            .each(&values)
            .unwrap()
            .into_iter()
            .map(|v| v.map(|v| v.nanos()))
            .collect::<Vec<_>>(),
        [None, None, Some(0)]
    );
    assert_eq!(
        reader.column(&[Some("01/02/2024"), Some("invalid")], false),
        Err(Unread::Outside)
    );
    assert_eq!(
        reader.column(
            &[Some("01/02/2024 10:00Z"), Some("01/02/2024 10:00+01")],
            false
        ),
        Err(Unread::Order)
    );
    assert_eq!(
        reader.column(&[Some("1970-01-15 9:30")], false),
        Err(Unread::Outside)
    );
    assert_eq!(
        reader.column(&[Some("15/1/1970 9:30")], false).unwrap()[0]
            .unwrap()
            .nanos(),
        1_243_800_000_000_000
    );
}

fn infer_text(
    name: &str,
    values: Vec<Option<&str>>,
) -> Result<sdm_contract::ops::Stype, sdm_contract::CoreError> {
    use arrow_array::{ArrayRef, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Utf8, true)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(values)) as ArrayRef],
    )
    .unwrap();
    let catalog = vec![sdm_contract::catalog::CatalogColumn {
        name: name.into(),
        declared_type: String::new(),
        not_null: false,
        primary_key: false,
        key_nullable: false,
        unique: false,
        dtype: None,
    }];
    let table = sdm_schema::infer::table(
        "events",
        &"events".into(),
        &catalog,
        &[],
        &schema,
        &[batch],
        &Reader::new(2026),
    )?;
    Ok(table.column(name).unwrap().stype)
}

#[test]
fn real_reader_inference_respects_name_weight_and_bounded_raw_samples() {
    use sdm_contract::ops::Stype;
    for (name, valid, timestamp) in [
        ("ts", 7, true),
        ("ts", 6, false),
        ("description", 7, false),
        ("description", 10, true),
    ] {
        let mut values = vec![Some("2024-01-15"); valid];
        values.extend(vec![Some("not a date"); 10 - valid]);
        assert_eq!(
            infer_text(name, values).unwrap() == Stype::Timestamp,
            timestamp,
            "{name}/{valid}"
        );
    }
    for (values, rule) in [
        (vec![Some("03/15/99"), Some("15/11/00")], "text_time_order"),
        (
            vec![Some("2024-01-15 10:00+01"), Some("2024-01-15 10:00+02")],
            "text_time_zones",
        ),
    ] {
        let error = infer_text("ts", values).unwrap_err();
        assert_eq!(
            error.fields.get("rule"),
            Some(&sdm_contract::Detail::text(rule))
        );
    }
    for blank in [None, Some("")] {
        let mut values = vec![blank; 100];
        values.push(Some("2024-01-15"));
        assert_ne!(infer_text("ts", values).unwrap(), Stype::Timestamp);
        let mut values = vec![blank; 99];
        values.extend([Some("2024-01-15"), Some("not a date")]);
        assert_eq!(infer_text("ts", values).unwrap(), Stype::Timestamp);
    }
    assert_ne!(
        infer_text("ts", vec![Some("code"), Some("0000-01-01")]).unwrap(),
        Stype::Timestamp
    );
}

#[test]
fn day_first_warning_keeps_the_first_shaped_values_format() {
    let reader = Reader::new(2026);
    for (values, expected) in [
        (
            vec![Some("15/03/2024T10:30:15.25Z")],
            "%d/%m/%YT%H:%M:%S.%f%z",
        ),
        (vec![Some("15-03-24 10:30:15 UTC")], "%d-%m-%y %H:%M:%S %Z"),
        (vec![Some("15/03/2024 10:30GMT")], "%d/%m/%Y %H:%M%Z"),
        (
            vec![
                None,
                Some("01/02/2024T10:30:15.25+01:00"),
                Some("15/03/2024"),
            ],
            "%d/%m/%YT%H:%M:%S.%f%z",
        ),
    ] {
        assert_eq!(
            reader.day_first_format(&values).as_deref(),
            Some(expected),
            "{values:?}"
        );
    }
}
