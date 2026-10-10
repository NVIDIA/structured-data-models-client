// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The wire's JSON: numbers kept as the tokens written, members in order,
//! a member named twice keeping its first place and last value, and what
//! JSON does not allow refused by position and category, never by text.

use sdm_wire::json::{Value, is_integer_token, read, whole};

#[test]
fn a_value_read_and_written_again_is_the_same_text() {
    let text = r#"{"b":[1,-0,97.0,1e-05,1E+16,123456789012345678901234567890],"a":{"x":"é\"\n","y":null,"z":[true,false]}}"#;
    let value = read(text.as_bytes()).unwrap();
    assert_eq!(
        String::from_utf8(value.to_bytes()).unwrap(),
        r#"{"b":[1,-0,97.0,1e-05,1E+16,123456789012345678901234567890],"a":{"x":"é\"\n","y":null,"z":[true,false]}}"#
    );
    let Value::Object(members) = &value else {
        panic!("an object");
    };
    assert_eq!(members[0].0, "b");
    assert_eq!(
        value.get("b"),
        Some(&Value::Array(
            [
                "1",
                "-0",
                "97.0",
                "1e-05",
                "1E+16",
                "123456789012345678901234567890"
            ]
            .map(|t| Value::Number(t.to_owned()))
            .to_vec()
        ))
    );
}

#[test]
fn a_member_named_twice_keeps_its_place_and_its_last_value() {
    let value = read(br#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
    assert_eq!(
        value,
        Value::Object(vec![
            ("a".into(), Value::Number("3".into())),
            ("b".into(), Value::Number("2".into())),
        ])
    );
}

#[test]
fn what_json_does_not_allow_is_refused_by_position_and_category() {
    for (text, category) in [
        (&b"{\"a\": NaN}"[..], "syntax"),
        (b"[Infinity]", "syntax"),
        (b"[-Infinity]", "syntax"),
        (b"{\"a\": 1", "eof"),
        (b"{\"secret-value\" 1}", "syntax"),
        (b"", "eof"),
    ] {
        let refused = read(text).unwrap_err();
        assert_eq!(
            refused.category,
            category,
            "{}",
            String::from_utf8_lossy(text)
        );
        assert!(refused.line >= 1);
    }
}

#[test]
fn a_whole_number_is_an_integer_or_a_float_with_no_fraction() {
    assert_eq!(whole("0"), Some(0));
    assert_eq!(whole("-12"), Some(-12));
    assert_eq!(whole("1.0"), Some(1));
    assert_eq!(whole("1e3"), Some(1000));
    assert_eq!(whole("-0.0"), Some(0));
    assert_eq!(whole("1.5"), None);
    assert_eq!(whole("1e400"), None);
    assert_eq!(whole("1e-400"), Some(0));
    assert_eq!(whole("123456789012345678901234567890123456789012"), None);
    assert!(is_integer_token("-12"));
    assert!(!is_integer_token("1.0"));
    assert!(!is_integer_token("1e3"));
    assert!(!is_integer_token("-"));
}
