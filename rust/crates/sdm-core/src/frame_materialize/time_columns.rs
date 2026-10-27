// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-source text-time preparation before frame sanitization and sampling.
//!
//! The caller joins the original column's chunks before this read, preserving
//! row order and every value, including rows whose keys are later dropped.
//! Designated text time columns have the client's UTC reading boundary;
//! timestamp features retain the existing engine reader's offset check.
//! Both consume the same approved fixed-format Reader, without native parsing.

use std::sync::Arc;

use arrow_array::{ArrayRef, TimestampNanosecondArray};
use arrow_schema::{DataType, TimeUnit};
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::ops::{DescribedColumn, DescribedTable, Dtype, Stype};
use sdm_contract::{CoreError, Scalar, Sentence};
use sdm_engine::strings::StringColumn;
use sdm_engine::time::text::{Reader, Unread};

/// Converted values and the existing warning's actual raw missingness facts.
/// Native presentation and warning order remain with the parent frame read.
pub(crate) struct ConvertedTimeColumn {
    pub values: ArrayRef,
    pub unread: u64,
    pub examples: Vec<Scalar>,
}

fn protocol(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// Convert one complete original native text column when its schema names
/// time meaning. Already typed columns retain the existing typed consumer.
pub(crate) fn convert(
    table: &DescribedTable,
    column: &DescribedColumn,
    array: ArrayRef,
    strings: Option<StringEncoding>,
    reader: Reader,
) -> Result<Option<ConvertedTimeColumn>, CoreError> {
    let designated = table.time_column.as_deref() == Some(column.name.as_str());
    if !designated
        && column.stype != Stype::Timestamp
        && !matches!(column.dtype, Dtype::Date | Dtype::Time)
    {
        return Ok(None);
    }
    if !matches!(
        array.data_type(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary
    ) {
        return Ok(None);
    }
    let strings = strings
        .ok_or_else(|| protocol("native frame text time lacks its explicit source encoding"))?;
    let original = StringColumn::new(array.as_ref(), strings)
        .map_err(|_| protocol("native frame time encoding does not match its source column"))?;
    let mut values = Vec::with_capacity(original.len());
    for row in 0..original.len() {
        let value = original
            .get(row)
            .map_err(|_| protocol("native frame time contains malformed source text encoding"))?;
        values.push(match value {
            None => None,
            Some(value) => Some(
                value
                    .utf8()
                    .ok_or_else(|| crate::materialize::text_unread(Unread::Outside))?,
            ),
        });
    }
    // The complete original column decides day/month order once. Filtering,
    // key sanitization and temporal bounds all consume this typed result.
    let read = reader
        .column(&values, designated)
        .map_err(crate::materialize::text_unread)?;
    let mut unread = 0_u64;
    let mut examples = Vec::new();
    for (value, read) in values.iter().zip(&read) {
        if let (Some(value), None) = (value, read) {
            unread += 1;
            let example = Scalar::Utf8((*value).to_owned());
            if examples.len() < 3 && !examples.contains(&example) {
                examples.push(example);
            }
        }
    }
    let nanos = read
        .into_iter()
        .map(|at| {
            at.map(|at| at.to_count(TimeUnit::Nanosecond, &column.name))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(ConvertedTimeColumn {
        values: Arc::new(TimestampNanosecondArray::from(nanos)),
        unread,
        examples,
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use arrow_array::cast::AsArray;
    use arrow_array::types::TimestampNanosecondType;
    use arrow_array::{BinaryArray, LargeStringArray, StringArray, TimestampSecondArray};
    use sdm_contract::Detail;
    use sdm_contract::ops::HeldSchema;
    use serde_json::json;

    use super::*;

    fn described(designated: bool, timestamp: bool) -> DescribedTable {
        let held: HeldSchema = serde_json::from_value(json!({
            "tables": [{
                "name": "events", "source": {"source": ["original.events"]},
                "columns": [{
                    "name": "when",
                    "dtype": if timestamp {"timestamp[us]"} else {"string"},
                    "stype": if timestamp {"timestamp"} else {"categorical"}
                }],
                "primary_key": [],
                "time_column": if designated {Some("when")} else {None}
            }],
            "links": []
        }))
        .unwrap();
        sdm_schema::held::read(held, &BTreeMap::new(), true)
            .unwrap()
            .description
            .tables
            .remove(0)
    }

    fn converted(table: &DescribedTable, values: ArrayRef) -> ConvertedTimeColumn {
        convert(
            table,
            &table.columns[0],
            values,
            Some(StringEncoding::Utf8),
            Reader::new(2026),
        )
        .unwrap()
        .unwrap()
    }

    fn counts(value: &ConvertedTimeColumn) -> Vec<Option<i64>> {
        value
            .values
            .as_primitive::<TimestampNanosecondType>()
            .iter()
            .collect()
    }

    fn refused(table: &DescribedTable, values: ArrayRef, rule: &str) {
        let error = match convert(
            table,
            &table.columns[0],
            values,
            Some(StringEncoding::Utf8),
            Reader::new(2026),
        ) {
            Err(error) => error,
            Ok(_) => panic!("the complete column did not refuse {rule}"),
        };
        assert_eq!(error.fields.get("rule"), Some(&Detail::text(rule)));
    }

    #[test]
    fn the_last_original_chunk_decides_all_earlier_ambiguous_dates() {
        let table = described(true, true);
        let first = LargeStringArray::from(vec!["01/02/2024"; 1004]);
        let last = LargeStringArray::from(vec!["15/03/2024"]);
        let original = arrow_select::concat::concat(&[&first, &last]).unwrap();
        let result = converted(&table, original);
        let read = counts(&result);
        // Actual full-column frozen public readings on both runtime cells.
        assert_eq!(read.len(), 1005);
        assert_eq!(read[0], Some(1_706_745_600_000_000_000));
        assert_eq!(read[1004], Some(1_710_460_800_000_000_000));
        assert_eq!(result.unread, 0);
        assert!(result.examples.is_empty());
    }

    #[test]
    fn a_late_conflict_cannot_be_hidden_by_row_selection_or_a_key_mask() {
        let table = described(true, true);
        let mut original = vec!["01/02/2024"; 1005];
        original[1003] = "15/03/2024";
        original[1004] = "03/15/2024";
        // The frozen public input also refuses when the final row's key is
        // missing: the parent must call this before Snapshot drops that row.
        refused(
            &table,
            Arc::new(StringArray::from(original)),
            "text_time_order",
        );
    }

    #[test]
    fn designated_time_normalizes_offsets_while_features_keep_the_offset_guard() {
        let values = || {
            Arc::new(StringArray::from(vec![
                "2024-02-01T01:00:00+01:00",
                "2024-02-01T00:00:00Z",
                "2024-02-01T00:00:00",
            ])) as ArrayRef
        };
        assert_eq!(
            counts(&converted(&described(true, true), values())),
            vec![Some(1_706_745_600_000_000_000); 3]
        );
        // Existing DT-11/DT-16 fixes the feature reading independently of
        // the native runtime's format guess and native exception class.
        refused(&described(false, true), values(), "text_time_zones");
    }

    #[test]
    fn a_timestamp_feature_keeps_submicroseconds_even_on_an_untimed_graph() {
        let table = described(false, true);
        let result = converted(
            &table,
            Arc::new(StringArray::from(vec![
                Some("2024-02-01T00:00:00.123456789Z"),
                None,
            ])),
        );
        assert!(table.time_column.is_none());
        assert_eq!(counts(&result), [Some(1_706_745_600_123_456_789), None]);
        assert_eq!(result.unread, 0);
    }

    #[test]
    fn nulls_and_raw_blanks_keep_distinct_warning_facts_before_sanitization() {
        let result = converted(
            &described(true, true),
            Arc::new(StringArray::from(vec![
                Some("2024-02-01"),
                None,
                Some(""),
                Some(" "),
                Some(""),
            ])),
        );
        assert_eq!(
            counts(&result),
            [Some(1_706_745_600_000_000_000), None, None, None, None]
        );
        assert_eq!(result.unread, 3);
        assert_eq!(
            result.examples,
            [Scalar::Utf8("".into()), Scalar::Utf8(" ".into())]
        );
        let all_null = converted(
            &described(false, true),
            Arc::new(LargeStringArray::from(vec![None::<&str>; 2])),
        );
        assert_eq!(counts(&all_null), [None, None]);
        assert_eq!(all_null.unread, 0);
    }

    #[test]
    fn ordinary_text_and_already_typed_timestamps_do_not_acquire_a_text_parser() {
        let table = described(false, false);
        let ordinary: ArrayRef = Arc::new(StringArray::from(vec!["not a date"]));
        assert!(
            convert(
                &table,
                &table.columns[0],
                ordinary,
                Some(StringEncoding::Utf8),
                Reader::new(2026)
            )
            .unwrap()
            .is_none()
        );
        let table = described(false, true);
        let typed: ArrayRef =
            Arc::new(TimestampSecondArray::from(vec![Some(1), None]).with_timezone("Europe/Paris"));
        assert!(
            convert(&table, &table.columns[0], typed, None, Reader::new(2026))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn out_of_format_text_is_named_and_exceptional_native_text_is_never_repaired() {
        let table = described(true, true);
        refused(
            &table,
            Arc::new(StringArray::from(vec!["not a date"])),
            "text_time",
        );
        let exceptional: ArrayRef = Arc::new(BinaryArray::from(vec![
            b"2024-02-01\xed\xa0\x80".as_slice(),
        ]));
        let error = match convert(
            &table,
            &table.columns[0],
            exceptional,
            Some(StringEncoding::Utf8Surrogatepass),
            Reader::new(2026),
        ) {
            Err(error) => error,
            Ok(_) => panic!("surrogate text cannot be one of the approved date formats"),
        };
        assert_eq!(error.fields.get("rule"), Some(&Detail::text("text_time")));
    }
}
