// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::{ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array};
use arrow_array::{UInt8Array, new_null_array};
use arrow_schema::DataType;
use sdm_engine::index_codes::{CodeWidth, IndexCodes, InvalidCodes};
use std::sync::Arc;

#[test]
fn code_width_and_unused_dictionary_survive_selection() {
    let families: [(ArrayRef, CodeWidth, usize); 7] = [
        (
            Arc::new(Int8Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I8,
            126,
        ),
        (
            Arc::new(Int16Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I16,
            127,
        ),
        (
            Arc::new(Int16Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I16,
            128,
        ),
        (
            Arc::new(Int16Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I16,
            32766,
        ),
        (
            Arc::new(Int32Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I32,
            32767,
        ),
        (
            Arc::new(Int32Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I32,
            32768,
        ),
        (
            Arc::new(Int64Array::from(vec![3, -1, 0, 3])),
            CodeWidth::I64,
            2147483647,
        ),
    ];
    for (input, width, dictionary_rows) in families {
        let facts = IndexCodes::admit(input.clone(), dictionary_rows, 4).unwrap();
        assert_eq!(facts.width(), width);
        assert_eq!(facts.dictionary_rows(), dictionary_rows);
        assert_eq!(facts.source_rows(), 4);
        assert!(Arc::ptr_eq(facts.values(), &input));
        assert_eq!(
            facts.selected(&[3, 1, 0, 3, 2]).unwrap(),
            [Some(3), None, Some(3), Some(3), Some(0)]
        );
        assert_eq!(facts.selected(&[]).unwrap(), []);
    }
}

#[test]
fn admission_checks_unselected_rows_too() {
    let facts = IndexCodes::admit(Arc::new(Int8Array::from(vec![0, 1, 4])), 4, 3);
    assert!(matches!(
        facts,
        Err(InvalidCodes::DictionaryBounds {
            row: 2,
            code: 4,
            dictionary_rows: 4
        })
    ));
    for code in [-2, i64::MIN, i64::MAX] {
        let error = IndexCodes::admit(Arc::new(Int64Array::from(vec![code])), 4, 1);
        assert!(matches!(
            error,
            Err(InvalidCodes::DictionaryBounds { row: 0, .. })
        ));
    }
}

#[test]
fn missing_codes_and_empty_dictionaries_have_one_encoding() {
    let facts = IndexCodes::admit(Arc::new(Int8Array::from(vec![-1, -1])), 0, 2).unwrap();
    assert_eq!(facts.selected(&[1, 0]).unwrap(), [None, None]);
    let empty = IndexCodes::admit(Arc::new(Int16Array::from(Vec::<i16>::new())), 0, 0).unwrap();
    assert_eq!(empty.source_rows(), 0);
    let invalid = IndexCodes::admit(new_null_array(&DataType::Int8, 2), 1, 2);
    assert!(matches!(invalid, Err(InvalidCodes::NullCode { row: 0 })));
    let invalid = IndexCodes::admit(Arc::new(Int8Array::from(vec![0])), 0, 1);
    assert!(matches!(
        invalid,
        Err(InvalidCodes::DictionaryBounds {
            row: 0,
            code: 0,
            dictionary_rows: 0
        })
    ));
}

#[test]
fn mismatched_frames_and_out_of_bounds_rows_are_rejected() {
    let input: ArrayRef = Arc::new(Int8Array::from(vec![0, 1]));
    let invalid = IndexCodes::admit(input.clone(), 2, 3);
    assert!(matches!(
        invalid,
        Err(InvalidCodes::RowCount {
            expected: 3,
            actual: 2
        })
    ));
    let facts = IndexCodes::admit(input, 2, 2).unwrap();
    assert_eq!(
        facts.selected(&[0, 2]),
        Err(InvalidCodes::RowBounds {
            row: 2,
            source_rows: 2
        })
    );
    let invalid = IndexCodes::admit(Arc::new(UInt8Array::from(vec![0, 1])), 2, 2);
    assert!(matches!(
        invalid,
        Err(InvalidCodes::PhysicalType(DataType::UInt8))
    ));
}
