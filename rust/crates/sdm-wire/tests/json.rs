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

use sdm_wire::json::{MAX_DEPTH, Value, answer_text, is_integer_token, read, read_answer, whole};

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

/// A wide object reads in about n log n: reading eight times as many
/// members takes far less than the sixty-four times a scan of the names
/// before each one would (T1.11d review round 2). Measured as a ratio of
/// the two sizes, each the best of three reads, so a slow debug build or a
/// busy runner moves both alike.
#[test]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: times the read"
)]
fn a_wide_object_reads_in_n_log_n() {
    let text = |n: usize| {
        let members: Vec<String> = (0..n).map(|i| format!("\"k{i}\":{i}")).collect();
        format!("{{{},\"k5\":\"last\"}}", members.join(","))
    };
    let best = |text: &str| {
        (0..3)
            .map(|_| {
                let started = std::time::Instant::now();
                let value = read(text.as_bytes()).unwrap();
                let took = started.elapsed().as_secs_f64();
                assert_eq!(value.get("k5"), Some(&Value::Text("last".into())));
                took
            })
            .fold(f64::INFINITY, f64::min)
    };
    let (small, large) = (text(20_000), text(160_000));
    let (small, large) = (best(&small), best(&large));
    assert!(large / small.max(1e-6) < 24.0, "{small} s, then {large} s");
}

/// An endpoint's answer (T1.28b): `NaN`, `Infinity` and `-Infinity`
/// tokens outside text read as numbers of the token written, as Python's
/// reader takes them, and written back as written; text that holds those
/// words, or the marker the reader stands them in with, stays text; what
/// `read` refuses otherwise is refused alike, the depth bound included,
/// tokens or not, deep or wide.
#[test]
fn an_answer_reads_non_finite_tokens_as_python_s_reader_does() {
    let text = r#"{"a":[NaN,-Infinity,Infinity,1e400],"b":"NaN Infinity","c":"\uE000NaN"}"#;
    assert_eq!(read(text.as_bytes()).unwrap_err().category, "not_finite");
    let value = read_answer(text.as_bytes()).unwrap();
    assert_eq!(
        value.get("a"),
        Some(&Value::Array(vec![
            Value::Number("NaN".into()),
            Value::Number("-Infinity".into()),
            Value::Number("Infinity".into()),
            Value::Number("1e400".into()),
        ]))
    );
    assert_eq!(value.get("b"), Some(&Value::Text("NaN Infinity".into())));
    assert_eq!(value.get("c"), Some(&Value::Text("\u{e000}NaN".into())));
    // Written back, the tokens are as written.
    assert!(
        String::from_utf8(value.to_bytes())
            .unwrap()
            .starts_with(r#"{"a":[NaN,-Infinity,Infinity,1e400],"b":"NaN Infinity","c":"#)
    );
    // The marker's own escaped spelling in the answer moves the marker on:
    // that text stays text.
    let spelled = br#"{"a":"\uE000NaN","b":NaN}"#;
    // A value after a member's name, in nested objects and arrays, is one.
    let inner = read_answer(br#"{"a":{"b":[NaN,{"c":-Infinity}]},"d":Infinity}"#).unwrap();
    assert_eq!(inner.get("d"), Some(&Value::Number("Infinity".into())));
    let value = read_answer(spelled).unwrap();
    assert_eq!(value.get("a"), Some(&Value::Text("\u{e000}NaN".into())));
    assert_eq!(value.get("b"), Some(&Value::Number("NaN".into())));
    // Faults read refuses stay refused: a bare word that is not a token,
    // and a token glued to another.
    for bad in [
        &b"[Nan]"[..],
        b"[NaNx]",
        b"[-NaN]",
        b"[Infinityy]",
        // A token where a member's name goes, as Python's reader refuses.
        b"{NaN: 1}",
        b"{\"a\": 1, Infinity: 2}",
    ] {
        assert!(
            read_answer(bad).is_err(),
            "{}",
            String::from_utf8_lossy(bad)
        );
    }
    // Deep: refused by its depth, with tokens or not; wide: read.
    let mut deep = b"[NaN,".to_vec();
    deep.extend(nested(10_000));
    deep.push(b']');
    assert_eq!(read_answer(&deep).unwrap_err().category, "depth");
    let wide: Vec<String> = (0..50_000).map(|i| format!("\"k{i}\":NaN")).collect();
    let wide = format!("{{{}}}", wide.join(","));
    let value = read_answer(wide.as_bytes()).unwrap();
    assert_eq!(value.get("k49999"), Some(&Value::Number("NaN".into())));
}

/// An answer's bytes are decoded as Python's `json.loads` decodes bytes:
/// UTF-8 with or without its byte order mark, and UTF-16 and UTF-32 of
/// either byte order, with or without one, all read as the same value;
/// UTF-8 without a mark is the bytes as they are; what does not decode,
/// a surrogate code point included, is refused as category `encoding`.
#[test]
fn an_answer_is_decoded_as_python_s_reader_decodes_bytes() {
    let text = r#"{"a":[1.5,NaN],"b":"é😀"}"#;
    let expected = read_answer(text.as_bytes()).unwrap();
    let utf16 = |big: bool| -> Vec<u8> {
        text.encode_utf16()
            .flat_map(|u| {
                if big {
                    u.to_be_bytes()
                } else {
                    u.to_le_bytes()
                }
            })
            .collect()
    };
    let utf32 = |big: bool| -> Vec<u8> {
        text.chars()
            .flat_map(|c| {
                let n = u32::from(c);
                if big {
                    n.to_be_bytes()
                } else {
                    n.to_le_bytes()
                }
            })
            .collect()
    };
    let marked = |mark: &[u8], body: Vec<u8>| [mark, &body].concat();
    let spellings = [
        marked(&[0xEF, 0xBB, 0xBF], text.as_bytes().to_vec()),
        utf16(true),
        utf16(false),
        marked(&[0xFE, 0xFF], utf16(true)),
        marked(&[0xFF, 0xFE], utf16(false)),
        utf32(true),
        utf32(false),
        marked(&[0, 0, 0xFE, 0xFF], utf32(true)),
        marked(&[0xFF, 0xFE, 0, 0], utf32(false)),
    ];
    for bytes in &spellings {
        assert_eq!(&*answer_text(bytes).unwrap(), text.as_bytes());
        assert_eq!(read_answer(bytes).unwrap(), expected);
    }
    // UTF-8 without a mark is not copied; two-byte text is UTF-16 by its
    // zero byte, as Python's detection has it; three bytes are UTF-8.
    assert!(matches!(
        answer_text(text.as_bytes()).unwrap(),
        std::borrow::Cow::Borrowed(_)
    ));
    assert_eq!(read_answer(&[b'1', 0]).unwrap(), Value::Number("1".into()));
    assert_eq!(read_answer(&[0, b'1']).unwrap(), Value::Number("1".into()));
    assert_eq!(read_answer(b"[1]").unwrap(), read(b"[1]").unwrap());
    // A second mark is a character, which JSON text cannot start with.
    let twice = marked(&[0xEF, 0xBB, 0xBF, 0xEF, 0xBB, 0xBF], b"{}".to_vec());
    assert_eq!(read_answer(&twice).unwrap_err().category, "syntax");
    for bad in [
        // An odd byte count in UTF-16, a count not a multiple of four in
        // UTF-32, and a UTF-32 code point past Unicode.
        marked(&[0xFE, 0xFF], vec![0, b'{', 0, b'}', 0]),
        marked(&[0, 0, 0xFE, 0xFF], vec![0, 0, 0, b'1', 0, 0]),
        vec![0, 0, 0, b'"', 0, 0x11, 0, 0, 0, 0, 0, b'"'],
    ] {
        assert_eq!(
            read_answer(&bad).unwrap_err().category,
            "encoding",
            "{bad:?}"
        );
    }
    // A lone surrogate in UTF-16, as a UTF-32 code point, and written in
    // UTF-8 as Python's `surrogatepass` keeps it, is read as U+FFFD, as its
    // escape is (register P1B-24, KO2-93).
    let replaced = Value::Text("\u{FFFD}".into());
    for lone in [
        vec![0, b'"', 0xD8, 0, 0, b'"'],
        vec![0, 0, 0, b'"', 0, 0, 0xD8, 0, 0, 0, 0, b'"'],
        b"\"\xED\xA0\x80\"".to_vec(),
    ] {
        assert_eq!(read_answer(&lone).unwrap(), replaced, "{lone:?}");
    }
}
