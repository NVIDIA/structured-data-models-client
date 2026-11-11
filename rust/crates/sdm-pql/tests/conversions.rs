// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent expectations for numeric and ISO date/time parameter conversion.

use sdm_pql::conversions::{
    LONGEST_TEXT, Unconverted, date_text, datetime_text, decimal_text, int_text, strip,
};
#[test]
fn integers_normalize_signs_zeros_and_unicode_decimal_digits() {
    for (input, expected) in [
        (" +0012 ", "12"),
        ("-000", "0"),
        ("-٣٢", "-32"),
        ("１２", "12"),
    ] {
        assert_eq!(int_text(input), Ok(expected.into()), "{input:?}");
    }
    for input in ["", "1.0", "1e2", "1_000", "+", "²"] {
        assert_eq!(int_text(input), Err(Unconverted::Invalid), "{input:?}");
    }
    assert_eq!(int_text(&"9".repeat(4300)).map(|v| v.len()), Ok(4300));
    assert_eq!(
        int_text(&"9".repeat(4301)),
        Err(Unconverted::TooManyDigits(4301))
    );
    for c in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
        let input = format!("{c}12{c}");
        assert_eq!(strip(&input), "12");
        assert_eq!(int_text(&input), Err(Unconverted::NotAnInteger));
    }
}

#[test]
fn decimals_keep_exact_scale_and_expand_exponents() {
    for (input, expected) in [
        ("1.2300", "1.2300"),
        ("1.25e2", "125"),
        ("1e-3", "0.001"),
        ("-0.00", "-0.00"),
        (" ١٢.٥ ", "12.5"),
        ("1_000.25", "1000.25"),
        ("9007199254740993", "9007199254740993"),
    ] {
        assert_eq!(decimal_text(input), Ok(expected.into()), "{input:?}");
    }
    for input in ["", "NaN", "sNaN", "Infinity", "-inf", "1.2.3"] {
        assert_eq!(decimal_text(input), Err(Unconverted::Invalid), "{input:?}");
    }
}

#[test]
fn dates_validate_calendar_and_iso_week_boundaries() {
    for (input, expected) in [
        ("20240229", "2024-02-29"),
        ("2020-W53-7", "2021-01-03"),
        ("0001-01-01", "0001-01-01"),
        ("9999-12-31", "9999-12-31"),
    ] {
        assert_eq!(date_text(input), Ok(expected.into()), "{input:?}");
    }
    for input in [
        "2023-02-29",
        "2024-13-01",
        "0000-01-01",
        "2021-W53-1",
        "2024-01-01junk",
    ] {
        assert_eq!(date_text(input), Err(Unconverted::Invalid), "{input:?}");
    }
}

#[test]
fn timestamps_preserve_offsets_and_normalize_fraction_precision() {
    for (input, expected) in [
        ("2024-02-29", "2024-02-29 00:00:00"),
        ("2024-02-29T12:34:56Z", "2024-02-29 12:34:56+00:00"),
        (
            "2024-02-29T12:34:56.1234567+05:30",
            "2024-02-29 12:34:56.123456+05:30",
        ),
        (
            "2024-02-29T12:34:56,1-04:00",
            "2024-02-29 12:34:56.100000-04:00",
        ),
        (
            "2024-02-29T12:34:56+00:00:30",
            "2024-02-29 12:34:56+00:00:30",
        ),
    ] {
        assert_eq!(datetime_text(input), Ok(expected.into()), "{input:?}");
    }
    for input in [
        "2024-01-01T24:00:00",
        "2024-01-01T12:60:00",
        "2024-01-01T12:00:60",
        "2024-01-01T12:00:00+24:00",
    ] {
        assert_eq!(datetime_text(input), Err(Unconverted::Invalid), "{input:?}");
    }
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

/// Deterministic generated inputs exercise dates, times, offsets and
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
