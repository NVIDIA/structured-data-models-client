// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless native Unicode data, pending serialized frame contract35 linking.
//! UTF8 with surrogatepass preserves every Python string codepoint, including
//! adjacent surrogate codepoints that are distinct from a supplementary scalar.
//! No normalization, replacement, casing or model policy belongs here.

#[cfg(test)]
mod tests {
    use super::{UnicodeError, UnicodeText};

    #[test]
    fn explicit_surrogate_pairs_and_supplementary_scalars_are_distinct_values() {
        let pair = UnicodeText::from_points([0xd83d, 0xde3a]).unwrap();
        let scalar = UnicodeText::from("😺");
        assert_ne!(pair, scalar);
        assert_eq!(pair.as_bytes(), [0xed, 0xa0, 0xbd, 0xed, 0xb8, 0xba]);
        assert_eq!(scalar.as_bytes(), [0xf0, 0x9f, 0x98, 0xba]);
        assert_eq!(pair.points().collect::<Vec<_>>(), [0xd83d, 0xde3a]);
        assert_eq!(scalar.points().collect::<Vec<_>>(), [0x1f63a]);
        assert_eq!(pair.points().count(), 2);
        assert_eq!(scalar.points().count(), 1);
        assert!(pair.utf8().is_none());
        assert_eq!(scalar.utf8(), Some("😺"));
    }

    #[test]
    fn native_unicode_keeps_nul_empty_and_every_surrogate_boundary() {
        for points in [
            vec![],
            vec![0],
            vec![0xd7ff, 0xd800, 0xdbff, 0xdc00, 0xdfff, 0xe000],
            vec![0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x10000, 0x10ffff],
            vec![0x391, 0xd800, 0x3a3, 0x130, 0xdfff],
        ] {
            let value = UnicodeText::from_points(points.clone()).unwrap();
            assert_eq!(value.points().collect::<Vec<_>>(), points);
            assert_eq!(
                UnicodeText::from_utf8_surrogatepass(value.as_bytes().to_vec()).unwrap(),
                value
            );
        }
        assert_eq!(UnicodeText::from("").as_bytes(), []);
        assert_eq!(
            UnicodeText::from("a\0b").points().collect::<Vec<_>>(),
            [97, 0, 98]
        );
    }

    #[test]
    fn malformed_native_encodings_do_not_enter_a_lossless_value() {
        for (bytes, offset) in [
            (vec![0x80], 0),
            (vec![0xc0, 0x80], 0),
            (vec![0xe0, 0x80, 0x80], 0),
            (vec![0xed, 0xa0], 0),
            (vec![0xf0, 0x80, 0x80, 0x80], 0),
            (vec![0xf4, 0x90, 0x80, 0x80], 0),
            (vec![b'a', 0xed, 0xa0, 0x80, 0xff], 4),
            (vec![b'a', 0xe2, 0x82], 1),
        ] {
            assert_eq!(
                UnicodeText::from_utf8_surrogatepass(bytes).unwrap_err(),
                UnicodeError::Byte(offset)
            );
        }
        assert_eq!(
            UnicodeText::from_points([0x61, 0x110000]).unwrap_err(),
            UnicodeError::Point(1)
        );
    }

    #[test]
    fn unicode_order_matches_exact_native_codepoints_without_merging_pairs() {
        let points = [
            0, 0x7f, 0x80, 0x7ff, 0x800, 0xd7ff, 0xd800, 0xdfff, 0xe000, 0xffff, 0x10000, 0x10ffff,
        ];
        let values: Vec<_> = points
            .into_iter()
            .map(|point| UnicodeText::from_points([point]).unwrap())
            .collect();
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        assert_ne!(UnicodeText::from("é"), UnicodeText::from("e\u{301}"));
        let ordinary = "ΟΣ İ é 東京 😺";
        assert_eq!(
            UnicodeText::from(ordinary),
            UnicodeText::from_points(ordinary.chars().map(u32::from)).unwrap()
        );
    }
}
