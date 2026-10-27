// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Borrowed native string data, pending coordinated contract35 linking.
//! Binary buffers mean text only under an explicit closed data descriptor.
//! Ordinary provider binary admission and key casting remain unchanged.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};
    use sdm_contract::data::strings::StringEncoding;
    use sdm_contract::scalar::unicode::UnicodeError;

    use super::{StringColumn, StringDataError};

    #[test]
    fn offset_widths_slices_nulls_and_native_units_keep_their_owners() {
        let native = [0xed, 0xa0, 0xbd, 0xed, 0xb8, 0xba];
        let scalar = "😺".as_bytes();
        for source in [
            Arc::new(BinaryArray::from(vec![
                Some(b"discard".as_slice()), None, Some(b"".as_slice()),
                Some(native.as_slice()), Some(scalar), Some(b"a\0b".as_slice()),
            ])) as ArrayRef,
            Arc::new(LargeBinaryArray::from(vec![
                Some(b"discard".as_slice()), None, Some(b"".as_slice()),
                Some(native.as_slice()), Some(scalar), Some(b"a\0b".as_slice()),
            ])) as ArrayRef,
        ] {
            let sliced = source.slice(1, 5);
            let column = StringColumn::new(sliced.as_ref(), StringEncoding::Utf8Surrogatepass)
                .unwrap();
            assert_eq!(column.len(), 5);
            assert!(column.get(0).unwrap().is_none());
            assert_eq!(column.get(1).unwrap().unwrap().as_bytes(), []);
            let pair = column.get(2).unwrap().unwrap();
            let scalar = column.get(3).unwrap().unwrap();
            assert_eq!(pair.points().collect::<Vec<_>>(), [0xd83d, 0xde3a]);
            assert_eq!(scalar.points().collect::<Vec<_>>(), [0x1f63a]);
            assert_ne!(pair, scalar);
            let points = column.get(4).unwrap().unwrap().points().collect::<Vec<_>>();
            assert_eq!(points, [97, 0, 98]);
            // Values borrow the original Arrow allocation, including after
            // slicing a column with an offset. No per-value Vec is made.
            let owner = source.to_data();
            let start = owner.buffers()[1].as_ptr() as usize;
            let end = start + owner.buffers()[1].len();
            assert!((start..end).contains(&(pair.as_bytes().as_ptr() as usize)));
        }
    }

    #[test]
    fn ordinary_strings_need_no_binary_or_host_reencoding() {
        for source in [
            Arc::new(StringArray::from(vec![Some("discard"), None, Some(""), Some("é\0😺")]))
                as ArrayRef,
            Arc::new(LargeStringArray::from(vec![Some("discard"), None, Some(""), Some("é\0😺")]))
                as ArrayRef,
        ] {
            let sliced = source.slice(1, 3);
            let column = StringColumn::new(sliced.as_ref(), StringEncoding::Utf8).unwrap();
            assert!(column.get(0).unwrap().is_none());
            assert_eq!(column.get(1).unwrap().unwrap().utf8(), Some(""));
            assert_eq!(column.get(2).unwrap().unwrap().utf8(), Some("é\0😺"));
        }
    }

    #[test]
    fn binary_text_requires_explicit_matching_meaning_and_valid_native_encoding() {
        let binary = BinaryArray::from(vec![Some(b"ordinary".as_slice()), Some(&[0xed, 0xa0])]);
        assert!(matches!(
            StringColumn::new(&binary, StringEncoding::Utf8),
            Err(StringDataError::Physical)
        ));
        let text = StringArray::from(vec![Some("ordinary")]);
        assert!(matches!(
            StringColumn::new(&text, StringEncoding::Utf8Surrogatepass),
            Err(StringDataError::Physical)
        ));
        let column = StringColumn::new(&binary, StringEncoding::Utf8Surrogatepass).unwrap();
        assert_eq!(column.get(0).unwrap().unwrap().utf8(), Some("ordinary"));
        assert_eq!(
            column.get(1).unwrap_err(),
            StringDataError::Encoding { row: 1, error: UnicodeError::Byte(0) }
        );
        assert_eq!(
            column.get(2).unwrap_err(),
            StringDataError::Index { row: 2, rows: 2 }
        );
    }
}
