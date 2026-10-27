// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless native request text, pending coordinated contract35 linking.
//! Ordinary UTF8 retains the existing JSON writer. Surrogate codepoints are
//! escaped separately; adjacent native surrogate points are never merged in
//! the logical value before writing model JSON.

use sdm_contract::scalar::unicode::UnicodeText;

/// Append one JSON string using the existing writer for ordinary text and
/// each ordinary run. Only native surrogate points require a new escape.
/// The owned logical string remains unchanged for native inspection.
pub(super) fn write(text: &UnicodeText, output: &mut Vec<u8>) {
    if let Some(ordinary) = text.utf8() {
        super::write_text(ordinary, output);
        return;
    }
    let bytes = text.as_bytes();
    output.push(b'"');
    let mut at = 0;
    while at < bytes.len() {
        match std::str::from_utf8(&bytes[at..]) {
            Ok(ordinary) => {
                fragment(ordinary, output);
                break;
            }
            Err(error) => {
                let end = at + error.valid_up_to();
                let ordinary =
                    std::str::from_utf8(&bytes[at..end]).expect("validated native ordinary run");
                fragment(ordinary, output);
                // UnicodeText has already validated canonical UTF8 with
                // surrogatepass. This is exactly one three-byte unit.
                let encoded = &bytes[end..end + 3];
                let point = (u32::from(encoded[0] & 0x0f) << 12)
                    | (u32::from(encoded[1] & 0x3f) << 6)
                    | u32::from(encoded[2] & 0x3f);
                debug_assert!((0xd800..=0xdfff).contains(&point));
                output.extend_from_slice(b"\\u");
                for shift in [12, 8, 4, 0] {
                    output.push(b"0123456789abcdef"[((point >> shift) & 0xf) as usize]);
                }
                at = end + 3;
            }
        }
    }
    output.push(b'"');
}

fn fragment(text: &str, output: &mut Vec<u8>) {
    if text.is_empty() {
        return;
    }
    let start = output.len();
    super::write_text(text, output);
    let end = output.len();
    // Remove only the two quotes of this run in the destination buffer.
    // There is no second whole input or whole payload allocation.
    output.copy_within(start + 1..end - 1, start);
    output.truncate(end - 2);
}

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
    fn ordinary_controls_keep_the_existing_provider_and_request_encoding() {
        for ordinary in [
            "",
            "\0",
            "\u{8}\t\n\u{c}\r",
            "\"\\/",
            "é 東京 😺",
            "\u{10ffff}",
        ] {
            let ordinary = ordinary.to_owned();
            let mut output = Vec::new();
            write(&UnicodeText::from(ordinary.as_str()), &mut output);
            assert_eq!(output, Value::Text(ordinary).to_bytes());
        }
    }
}
