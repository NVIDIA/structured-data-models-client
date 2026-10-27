// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless native request text, pending coordinated contract35 linking.
//! Ordinary UTF8 retains the existing JSON writer. Surrogate codepoints are
//! escaped separately; adjacent native surrogate points are never merged in
//! the logical value before writing model JSON.

#[cfg(test)]
mod tests {
    use sdm_contract::scalar::unicode::UnicodeText;

    use super::write;
    use crate::json::{Value, read_answer};

    fn encoded(points: &[u32]) -> Vec<u8> {
        let text = UnicodeText::from_points(points.iter().copied()).unwrap();
        let mut bytes = Vec::new();
        write(&text, &mut bytes);
        bytes
    }

    #[test]
    fn request_text_preserves_lone_units_and_explicit_pairs_until_json_reading() {
        assert_eq!(encoded(&[0xd800]), br#""\ud800""#);
        assert_eq!(encoded(&[0xdfff]), br#""\udfff""#);
        let pair = encoded(&[0xd83d, 0xde3a]);
        assert_eq!(pair, br#""\ud83d\ude3a""#);
        let scalar = encoded(&[0x1f63a]);
        assert_eq!(scalar, Value::Text("😺".into()).to_bytes());
        // Frozen native inspection distinguishes these values. JSON readers
        // combine the escaped pair, while the registered answer replacement
        // policy continues to replace only a lone unit with U+FFFD.
        assert_eq!(read_answer(&pair).unwrap(), read_answer(&scalar).unwrap());
        assert_eq!(
            read_answer(&encoded(&[0xd800])).unwrap(),
            Value::Text("�".into())
        );
    }

    #[test]
    fn exceptional_request_text_uses_the_existing_escape_policy_between_units() {
        assert_eq!(
            encoded(&[0x22, 0x5c, 0x0a, 0xd800, 0x00, 0x1f63a, 0xdfff]),
            "\"\\\"\\\\\\n\\ud800\\u0000😺\\udfff\"".as_bytes()
        );
        let mut output = b"prefix:".to_vec();
        write(
            &UnicodeText::from_points([0xd800, 0x22, 0xdfff]).unwrap(),
            &mut output,
        );
        assert_eq!(output, br#"prefix:"\ud800\"\udfff""#);
    }

    #[test]
    fn every_ordinary_scalar_keeps_the_existing_provider_and_request_encoding() {
        for point in 0..=0x10ffff {
            let Some(character) = char::from_u32(point) else {
                continue;
            };
            let ordinary = character.to_string();
            let mut output = Vec::new();
            write(&UnicodeText::from(ordinary.as_str()), &mut output);
            assert_eq!(output, Value::Text(ordinary).to_bytes(), "U+{point:04X}");
        }
    }
}
