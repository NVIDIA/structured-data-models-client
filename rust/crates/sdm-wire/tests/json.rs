// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The wire's JSON: numbers kept as the tokens written, members in order,
//! a member named twice keeping its first place and last value, and what
//! JSON does not allow refused by position and category, never by text.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reruns its own binary for the deep-text test"
)]

use sdm_wire::json::{MAX_DEPTH, Value, is_integer_token, read, whole};

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
        (&b"{\"a\": NaN}"[..], "not_finite"),
        (b"[Infinity]", "not_finite"),
        (b"[-Infinity]", "not_finite"),
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

fn nested(depth: usize) -> Vec<u8> {
    let mut text = "[".repeat(depth).into_bytes();
    text.push(b'0');
    text.extend("]".repeat(depth).bytes());
    text
}

/// Text nested past the reader's depth is refused by category, before any
/// value is built or dropped; the deepest it takes reads and drops whole.
#[test]
fn text_nested_past_the_depth_is_refused_before_anything_is_built() {
    let deepest = read(&nested(MAX_DEPTH)).unwrap();
    drop(deepest);
    for depth in [MAX_DEPTH + 1, 10_000, 200_000] {
        let refused = read(&nested(depth)).unwrap_err();
        assert_eq!(refused.category, "depth", "{depth}");
        assert_eq!((refused.line, refused.column), (1, MAX_DEPTH + 1));
    }
    // In an object member, past brackets and quotes held in text.
    let mut text = br#"{"a": "[[[{{{\"", "b": "#.to_vec();
    text.extend(nested(10_000));
    text.push(b'}');
    assert_eq!(read(&text).unwrap_err().category, "depth");
    let within = format!(
        r#"{{"a": "{}", "b": {}}}"#,
        "[".repeat(500),
        String::from_utf8(nested(10)).unwrap()
    );
    assert!(read(within.as_bytes()).is_ok());
    println!("deep text refused");
}

/// The same check in a process of its own: it must end with a refusal, not
/// abort on an overflowed stack (T1.11d review round 1).
#[test]
fn deep_text_ends_in_a_refusal_in_a_process_of_its_own() {
    let child = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "text_nested_past_the_depth_is_refused_before_anything_is_built",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    assert!(
        child.status.success(),
        "{:?}\n{stdout}\n{}",
        child.status,
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(stdout.contains("deep text refused"), "{stdout}");
}

/// A wide object reads in n log n: a hundred thousand members, each named
/// once, read well inside a second, where finding each name by a scan of
/// those before it took minutes at this width (T1.11d review round 2).
#[test]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: times the read"
)]
fn a_wide_object_reads_in_n_log_n() {
    let members: Vec<String> = (0..100_000).map(|i| format!("\"k{i}\":{i}")).collect();
    let text = format!("{{{},\"k5\":\"last\"}}", members.join(","));
    let started = std::time::Instant::now();
    let value = read(text.as_bytes()).unwrap();
    let took = started.elapsed();
    let Value::Object(read_members) = &value else {
        panic!("an object");
    };
    assert_eq!(read_members.len(), 100_000);
    assert_eq!(value.get("k5"), Some(&Value::Text("last".into())));
    assert!(took.as_secs_f64() < 3.0, "{took:?}");
}
