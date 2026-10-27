// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless native Unicode data, pending serialized frame contract35 linking.
//! UTF8 with surrogatepass preserves every Python string codepoint, including
//! adjacent surrogate codepoints that are distinct from a supplementary scalar.
//! No normalization, replacement, casing or model policy belongs here.

/// Exact native string codepoints, stored as UTF8 with surrogatepass.
/// Validation permits surrogate codepoints but rejects overlong encodings,
/// malformed UTF8 and values past U+10FFFF. It never combines adjacent units.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnicodeText {
    bytes: Vec<u8>,
}

/// Location in the supplied representation, with no caller value echoed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnicodeError {
    Byte(usize),
    Point(usize),
}

impl std::fmt::Display for UnicodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Byte(offset) => write!(
                formatter,
                "invalid native Unicode encoding at byte {offset}"
            ),
            Self::Point(offset) => write!(
                formatter,
                "invalid native Unicode point at position {offset}"
            ),
        }
    }
}

impl std::error::Error for UnicodeError {}

impl UnicodeText {
    /// Take an owned native encoding without another full buffer copy.
    pub fn from_utf8_surrogatepass(bytes: Vec<u8>) -> Result<Self, UnicodeError> {
        let mut at = 0;
        while at < bytes.len() {
            match std::str::from_utf8(&bytes[at..]) {
                Ok(_) => break,
                Err(error) => {
                    at += error.valid_up_to();
                    // UTF8 rejects exactly these canonical three-byte
                    // surrogate encodings, which surrogatepass retains.
                    if bytes.get(at) != Some(&0xed)
                        || !matches!(bytes.get(at + 1), Some(0xa0..=0xbf))
                        || !matches!(bytes.get(at + 2), Some(0x80..=0xbf))
                    {
                        return Err(UnicodeError::Byte(at));
                    }
                    at += 3;
                }
            }
        }
        Ok(Self { bytes })
    }

    pub fn from_points(points: impl IntoIterator<Item = u32>) -> Result<Self, UnicodeError> {
        let mut bytes = Vec::new();
        for (position, point) in points.into_iter().enumerate() {
            if let Some(character) = char::from_u32(point) {
                let mut encoded = [0; 4];
                bytes.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
            } else if (0xd800..=0xdfff).contains(&point) {
                bytes.extend_from_slice(&[
                    0xe0 | ((point >> 12) as u8),
                    0x80 | (((point >> 6) & 0x3f) as u8),
                    0x80 | ((point & 0x3f) as u8),
                ]);
            } else {
                return Err(UnicodeError::Point(position));
            }
        }
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// An ordinary string view only where no surrogate codepoint occurs.
    pub fn utf8(&self) -> Option<&str> {
        std::str::from_utf8(&self.bytes).ok()
    }

    pub fn points(&self) -> UnicodePoints<'_> {
        UnicodePoints {
            remaining: &self.bytes,
        }
    }
}

impl From<&str> for UnicodeText {
    fn from(text: &str) -> Self {
        Self {
            bytes: text.as_bytes().to_vec(),
        }
    }
}

impl From<String> for UnicodeText {
    fn from(text: String) -> Self {
        Self {
            bytes: text.into_bytes(),
        }
    }
}

pub struct UnicodePoints<'a> {
    remaining: &'a [u8],
}

impl Iterator for UnicodePoints<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let first = *self.remaining.first()?;
        let (width, mask) = match first {
            0..=0x7f => (1, 0x7f),
            0xc2..=0xdf => (2, 0x1f),
            0xe0..=0xef => (3, 0x0f),
            _ => (4, 0x07),
        };
        // Constructors validate this private buffer once. Decoding its
        // codepoints neither revalidates the tail nor merges surrogate pairs.
        let encoded = self
            .remaining
            .get(..width)
            .expect("validated Unicode width");
        let mut point = u32::from(first & mask);
        for byte in &encoded[1..] {
            point = (point << 6) | u32::from(byte & 0x3f);
        }
        self.remaining = &self.remaining[width..];
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining.len().div_ceil(4), Some(self.remaining.len()))
    }
}

impl std::iter::FusedIterator for UnicodePoints<'_> {}

#[cfg(test)]
mod tests {
    use super::{UnicodeError, UnicodeSlice, UnicodeText};

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

    #[test]
    fn borrowed_native_text_keeps_arrow_owned_bytes_without_a_value_copy() {
        let source = [b'A', 0xed, 0xa0, 0xbd, 0xed, 0xb8, 0xba, b'B'];
        let borrowed = UnicodeSlice::from_utf8_surrogatepass(&source).unwrap();
        assert_eq!(borrowed.as_bytes().as_ptr(), source.as_ptr());
        assert_eq!(borrowed.points().collect::<Vec<_>>(), [65, 0xd83d, 0xde3a, 66]);
        assert!(borrowed.utf8().is_none());
        let ordinary = "é 😺";
        let borrowed = UnicodeSlice::from(ordinary);
        assert_eq!(borrowed.as_bytes().as_ptr(), ordinary.as_ptr());
        assert_eq!(borrowed.utf8(), Some(ordinary));
        let owned = UnicodeText::from_points([0xd800, 65]).unwrap();
        let borrowed = owned.as_slice();
        assert_eq!(borrowed.as_bytes().as_ptr(), owned.as_bytes().as_ptr());
        assert_eq!(borrowed.points().collect::<Vec<_>>(), [0xd800, 65]);
        assert_eq!(
            UnicodeSlice::from_utf8_surrogatepass(&[65, 0xed, 0xa0]).unwrap_err(),
            UnicodeError::Byte(1)
        );
    }

}
