// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Numeric response text accepts decimal syntax, supported Unicode digits and
//! whitespace, while rejecting malformed separators and incomplete numbers.

use sdm_wire::response::numeric_text;

#[test]
fn decimal_spellings_have_independent_numeric_values() {
    for text in [
        "0.5",
        " 0.5 ",
        "\t0.5\n",
        "0_0.5",
        ".5",
        "+.5",
        "\u{0660}.\u{0665}",
        "\u{ff10}.\u{ff15}",
        "\u{00a0}0.5\u{3000}",
    ] {
        assert_eq!(numeric_text(text), Some(0.5), "{text:?}");
    }
    for (text, value) in [
        ("1_000.25", 1000.25_f64),
        ("5.", 5.0),
        ("-1e-3", -0.001),
        ("1E+05", 100_000.0),
        ("1e5_0", 1e50),
        ("\u{0661}\u{0662}", 12.0),
        ("\u{0660}_\u{0665}", 5.0),
        ("\u{0085} 2", 2.0),
        ("-0", -0.0),
    ] {
        assert_eq!(
            numeric_text(text).map(f64::to_bits),
            Some(value.to_bits()),
            "{text:?}"
        );
    }
}

#[test]
fn special_values_preserve_sign_and_nan_semantics() {
    for text in ["inf", "1e400"] {
        assert_eq!(numeric_text(text), Some(f64::INFINITY), "{text:?}");
    }
    assert_eq!(numeric_text("-Infinity"), Some(f64::NEG_INFINITY));
    for text in ["nan", "NaN", "+nan"] {
        assert!(numeric_text(text).is_some_and(f64::is_nan), "{text:?}");
    }
}

#[test]
fn malformed_numeric_text_is_refused() {
    for text in [
        "1__0",
        "_1",
        "1_",
        "1_.5",
        "1._5",
        "1e_5",
        "0.5\u{200b}",
        "0x1",
        "1,5",
        "",
        "  ",
        "1.2.3",
        "e5",
        ".",
        "1e",
        "--1",
        "0.5 0",
        "1\u{001c}",
    ] {
        assert_eq!(numeric_text(text), None, "{text:?}");
    }
}

/// The digits every supported Python takes, and no more ( fix round
/// 2): Kawi's, which Unicode 15.0 added and the reference's Python 3.12
/// takes, are refused, as Python 3.10, the floor, refuses them (register
/// P1B-25); Arabic-Indic and fullwidth digits, in Unicode 13.0, are read.
#[test]
fn only_the_digits_every_supported_python_reads_are_read() {
    assert_eq!(numeric_text("\u{11f50}.\u{11f55}"), None);
    assert_eq!(numeric_text("\u{16ac0}"), None);
    assert_eq!(numeric_text("\u{0660}.\u{0665}"), Some(0.5));
    assert_eq!(numeric_text("\u{ff10}.\u{ff15}"), Some(0.5));
}
