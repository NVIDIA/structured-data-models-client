// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The conversions a view's parameter default goes through (T3.5), held to
//! the old client's answers for each text default in
//! `contract/view-step-vectors.json` (`default_text`), among them the
//! second reviews' timestamps and numbers: Python 3.12's `int`, `Decimal`
//! and `fromisoformat` read them there.

use sdm_pql::conversions::{
    LONGEST_TEXT, Unconverted, date_text, datetime_text, decimal_digit, decimal_text, int_text,
    not_an_integer_text, strip, too_many_digits_text,
};
use serde_json::Value as Json;

const STEPS: &str = include_str!("../../../../contract/view-step-vectors.json");

#[test]
fn each_text_default_converts_as_python_converted_it() {
    let document: Json = serde_json::from_str(STEPS).unwrap();
    let mut wrong = Vec::new();
    let mut held = 0;
    for vector in document["default_text"].as_array().unwrap() {
        if vector["default"]["t"] != "str" {
            continue;
        }
        let text = vector["default"]["v"].as_str().unwrap();
        let data_type = vector["data_type"].as_str().unwrap();
        let converted = match data_type {
            "TIMESTAMP" => datetime_text(strip(text)),
            "DATE" => date_text(strip(text)),
            "DOUBLE" | "DECIMAL(10,2)" => decimal_text(text),
            "INT" => int_text(text),
            _ => continue,
        };
        held += 1;
        let error = &vector["error"];
        let expected = match vector["text"].as_str() {
            Some(text) => Ok(text.to_owned()),
            None if error["type"] == "MemoryError" => Err(Unconverted::TooLong),
            None => match error["message"].as_str().unwrap() {
                message if message.starts_with("Exceeds the limit") => {
                    let digits = text.chars().filter(|&c| decimal_digit(c).is_some()).count();
                    assert_eq!(message, too_many_digits_text(digits));
                    Err(Unconverted::TooManyDigits(digits))
                }
                message if message.starts_with("invalid literal for int()") => {
                    assert_eq!(message, not_an_integer_text(text));
                    Err(Unconverted::NotAnInteger)
                }
                _ => Err(Unconverted::Invalid),
            },
        };
        if converted != expected {
            wrong.push(format!(
                "{data_type} {text:?}: {converted:?} where {expected:?}"
            ));
        }
    }
    assert!(held > 10_000, "{held} defaults");
    assert!(
        wrong.is_empty(),
        "{} differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// No text is built past the bound: a decimal whose digits would fill
/// more is refused before any is written.
#[test]
fn a_decimal_past_the_bound_is_refused_by_name() {
    let longest = format!("1e{}", LONGEST_TEXT - 1);
    assert_eq!(decimal_text(&longest).map(|t| t.len()), Ok(LONGEST_TEXT));
    assert_eq!(
        decimal_text(&format!("1e{LONGEST_TEXT}")),
        Err(Unconverted::TooLong)
    );
    assert_eq!(
        decimal_text("1e-999999999999999999"),
        Err(Unconverted::TooLong)
    );
    assert_eq!(
        decimal_text("1e9223372036854775807"),
        Err(Unconverted::Invalid)
    );
    assert_eq!(
        decimal_text("1e99999999999999999999999"),
        Err(Unconverted::Invalid)
    );
}

/// No text panics a conversion (T3.5 fix 2, the owner's stop rule): a few
/// seconds of texts made of the pieces of dates, times, offsets and
/// numbers, digits of other scripts and white space among them.
#[test]
fn no_text_panics_a_conversion() {
    const PIECES: &[&str] = &[
        "2024",
        "-",
        "W",
        "01",
        "1",
        "0",
        "9",
        "T",
        " ",
        ":",
        ".",
        ",",
        "+",
        "Z",
        "z",
        "e",
        "E",
        "_",
        "\u{661}",
        "\u{ff10}",
        "\u{1d7ce}",
        "é",
        "\u{3000}",
        "\u{1c}",
        "\0",
        "inf",
        "nan",
        "s",
        "99999999999999999999",
        "-9223372036854775808",
        "\u{d7ff}",
    ];
    let mut state: u64 = 0x5851_f42d_4c95_7f2d;
    let mut next = |n: usize| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % n as u64) as usize
    };
    for _ in 0..200_000 {
        let length = 1 + next(12);
        let text: String = (0..length).map(|_| PIECES[next(PIECES.len())]).collect();
        let _ = int_text(&text);
        let _ = decimal_text(&text);
        let _ = date_text(&text);
        let _ = datetime_text(&text);
    }
}
