// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Borrowed native string data, pending coordinated contract35 linking.
//! Binary buffers mean text only under an explicit closed data descriptor.
//! Ordinary provider binary admission and key casting remain unchanged.

use arrow_array::{Array, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray};
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::scalar::unicode::{UnicodeError, UnicodeSlice};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringDataError {
    Physical,
    Index { row: usize, rows: usize },
    Encoding { row: usize, error: UnicodeError },
}

/// Borrowed physical data with its already matched explicit meaning.
/// No per-value or whole-column ownership copies are necessary.
pub struct StringColumn<'a> {
    values: Values<'a>,
}

enum Values<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Surrogatepass(&'a BinaryArray),
    LargeSurrogatepass(&'a LargeBinaryArray),
}

impl<'a> StringColumn<'a> {
    pub fn new(array: &'a dyn Array, encoding: StringEncoding) -> Result<Self, StringDataError> {
        let values = match encoding {
            StringEncoding::Utf8 => array
                .as_any()
                .downcast_ref::<StringArray>()
                .map(Values::Utf8)
                .or_else(|| array.as_any().downcast_ref::<LargeStringArray>().map(Values::LargeUtf8)),
            StringEncoding::Utf8Surrogatepass => array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .map(Values::Surrogatepass)
                .or_else(|| array.as_any().downcast_ref::<LargeBinaryArray>().map(Values::LargeSurrogatepass)),
        }
        .ok_or(StringDataError::Physical)?;
        Ok(Self { values })
    }

    pub fn len(&self) -> usize {
        match &self.values {
            Values::Utf8(values) => values.len(),
            Values::LargeUtf8(values) => values.len(),
            Values::Surrogatepass(values) => values.len(),
            Values::LargeSurrogatepass(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, row: usize) -> Result<Option<UnicodeSlice<'a>>, StringDataError> {
        if row >= self.len() {
            return Err(StringDataError::Index { row, rows: self.len() });
        }
        macro_rules! ordinary {
            ($values:expr) => {
                Ok((!$values.is_null(row)).then(|| UnicodeSlice::from($values.value(row))))
            };
        }
        macro_rules! native {
            ($values:expr) => {
                if $values.is_null(row) {
                    Ok(None)
                } else {
                    UnicodeSlice::from_utf8_surrogatepass($values.value(row))
                        .map(Some)
                        .map_err(|error| StringDataError::Encoding { row, error })
                }
            };
        }
        match &self.values {
            Values::Utf8(values) => ordinary!(values),
            Values::LargeUtf8(values) => ordinary!(values),
            Values::Surrogatepass(values) => native!(values),
            Values::LargeSurrogatepass(values) => native!(values),
        }
    }
}

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
