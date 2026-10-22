// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python's equality of dict keys, written once (carried row P2-WR-6): two
//! values Python's `==` holds equal, as a `dict` keys them, get the same
//! key here, and two it holds unequal different ones. A number is keyed by
//! its exact value, whatever its type (`1`, `1.0`, `True` and
//! `Decimal('1.00')` are one key), text by its text, and a missing value
//! (`None`) by itself. The response reader keys a mapping's members by it
//! (`response::mapping`), and `sdm-request` a ranking's candidates
//! (`dict.fromkeys`); each decides what a NaN is, which equals nothing,
//! not even itself.

/// The key of a missing value (`None`).
pub const MISSING: &str = "z";

/// The key of text.
pub fn text(text: &str) -> String {
    format!("s{text}")
}

/// The key of a boolean: the number it equals (`True == 1`).
pub fn boolean(value: bool) -> String {
    format!("n{}", u8::from(value))
}

/// The key of the exact decimal number `fixed` writes in fixed point
/// (`-012.50`: an optional `-`, digits, and an optional fraction): its
/// value with no leading or trailing zero, and zero unsigned (`-12.5`,
/// `0`), so every spelling of one value is one key.
pub fn exact(fixed: &str) -> String {
    let (negative, digits) = match fixed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, fixed),
    };
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    let whole = whole.trim_start_matches('0');
    let fraction = fraction.trim_end_matches('0');
    let value = match (whole.is_empty(), fraction.is_empty()) {
        (true, true) => return "n0".to_owned(),
        (_, true) => whole.to_owned(),
        (true, false) => format!("0.{fraction}"),
        (false, false) => format!("{whole}.{fraction}"),
    };
    format!("n{}{value}", if negative { "-" } else { "" })
}

/// The key of a double: an infinity by its sign, any other by its exact
/// value ([`exact`]), so a whole double is its integer's key; `None` for a
/// NaN, which the caller keys as Python meets it.
pub fn float(value: f64) -> Option<String> {
    if value.is_nan() {
        None
    } else if value.is_infinite() {
        Some(format!("i{}", value.is_sign_negative()))
    } else {
        // 1,100 places hold every double's exact value (the smallest has
        // 1,074).
        Some(exact(&format!("{value:.1100}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_spelling_of_a_value_is_one_key_and_other_values_are_others() {
        assert_eq!(exact("1"), boolean(true));
        assert_eq!(exact("0"), boolean(false));
        assert_eq!(exact("-0"), exact("0.000"));
        assert_eq!(exact("007"), exact("7.0"));
        assert_eq!(exact("-12.50"), exact("-12.5"));
        assert_eq!(float(1.0), Some(exact("1")));
        assert_eq!(float(-0.0), Some(exact("0")));
        assert_eq!(float(0.5), Some(exact("0.50")));
        // A whole double is its integer's key, however large.
        assert_eq!(float(2f64.powi(70)), Some(exact("1180591620717411303424")));
        assert_eq!(
            float(1e300).map(|k| k.len()),
            Some("n1".len() + 300),
            "1e300 is a whole number of 301 digits"
        );
        assert_eq!(float(f64::NAN), None);
        assert_ne!(float(f64::INFINITY), float(f64::NEG_INFINITY));
        // 0.1 is not the decimal 0.1, as Python's `0.1 == Decimal('0.1')`
        // is false.
        assert_ne!(float(0.1), Some(exact("0.1")));
        assert_ne!(exact("1"), exact("-1"));
        assert_ne!(exact("10"), exact("1"));
        assert_ne!(exact("0.01"), exact("0.1"));
        assert_ne!(text("1"), exact("1"));
        assert_ne!(text(""), MISSING);
    }
}
