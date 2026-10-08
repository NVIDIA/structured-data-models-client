// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integers too wide to come back unchanged.
//!
//! serde_json reads an integer outside both `i64` and `u64` as the nearest
//! `f64`, so it would come back as a float with other digits. Its
//! `arbitrary_precision` feature would keep it, but cannot be used with
//! `flatten` and `untagged`, which the open objects and unions rely on. A
//! payload holding such an integer is refused instead.

use serde::de::Error as _;

/// Refuse `json` if it holds an integer that does not fit in 64 bits.
///
/// Only numbers are looked at, outside strings; whether `json` is valid JSON
/// is left to the parser.
pub(crate) fn check(json: &str) -> Result<(), serde_json::Error> {
    let bytes = json.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                while i < bytes.len()
                    && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    i += 1;
                }
                let number = &json[start..i];
                if is_wide_integer(number) {
                    return Err(serde_json::Error::custom(format!(
                        "the integer {number} does not fit in 64 bits, so it \
                         would not come back unchanged"
                    )));
                }
            }
            _ => i += 1,
        }
    }
    Ok(())
}

fn is_wide_integer(number: &str) -> bool {
    let digits = number.strip_prefix('-').unwrap_or(number);
    !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && number.parse::<i64>().is_err()
        && number.parse::<u64>().is_err()
}

#[cfg(test)]
mod tests {
    use super::check;

    #[test]
    fn integers_within_64_bits_pass() {
        for json in [
            "0",
            "-0",
            "[9223372036854775807, -9223372036854775808]",
            "18446744073709551615",
            r#"{"a": [1, -2, {"b": 3}]}"#,
        ] {
            assert!(check(json).is_ok(), "{json} was refused");
        }
    }

    #[test]
    fn wider_integers_are_refused() {
        for json in [
            "18446744073709551616",
            "-9223372036854775809",
            r#"{"a": {"b": [1, 123456789012345678901234567890]}}"#,
        ] {
            let error = check(json).expect_err(json);
            assert!(
                error.to_string().contains("does not fit in 64 bits"),
                "{error}"
            );
        }
    }

    #[test]
    fn floats_and_strings_are_not_integers() {
        for json in [
            "123456789012345678901234567890.0",
            "1e30",
            "-1E+30",
            r#""18446744073709551616""#,
            r#"{"18446744073709551616": "a \" 18446744073709551616 \\"}"#,
            r#"["\\", 1]"#,
        ] {
            assert!(check(json).is_ok(), "{json} was refused");
        }
    }

    #[test]
    fn what_is_not_json_is_left_to_the_parser() {
        for json in ["-", "--1", "1-2", "\"unterminated 1", "\"\\"] {
            assert!(check(json).is_ok(), "{json} was refused");
        }
    }
}
