// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How Python writes values into text: `repr` of a string and of a float,
//! which several messages print a list of.

use crate::printable::UNPRINTABLE;

/// Whether Python's `str.isprintable` holds for `c`.
pub fn is_printable(c: char) -> bool {
    let cp = c as u32;
    UNPRINTABLE
        .binary_search_by(|&(lo, hi)| {
            if hi < cp {
                std::cmp::Ordering::Less
            } else if lo > cp {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_err()
}

/// `repr(s)` for a Python `str`.
pub fn str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c => {
                let cp = c as u32;
                if cp < 0x100 {
                    out.push_str(&format!("\\x{cp:02x}"));
                } else if cp < 0x10000 {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    out.push_str(&format!("\\U{cp:08x}"));
                }
            }
        }
    }
    out.push(quote);
    out
}

/// `repr(x)` (and `str(x)`) for a Python `float`.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // As many digits as the shortest that round-trip, as Python's repr
    // uses. Of the decimals of that many digits that round-trip, Python
    // writes the nearest, a tie to the even one, which exact formatting
    // gives where it round-trips; the shortest formatting may break a tie
    // the other way. Next to a power of two the nearest may not round-trip,
    // and the shortest is the one that does.
    let shortest = format!("{:e}", x.abs());
    let precision = shortest
        .split_once('e')
        .map_or(0, |(m, _)| m.chars().filter(char::is_ascii_digit).count())
        .saturating_sub(1);
    let nearest = format!("{:.*e}", precision, x.abs());
    let sci = if nearest.parse::<f64>() == Ok(x.abs()) {
        nearest
    } else {
        shortest
    };
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = match digits.trim_end_matches('0') {
        "" => "0".to_string(),
        kept => kept.to_string(),
    };
    // Python's decpt: the value is 0.DIGITS * 10**decpt.
    let decpt = exponent + 1;
    let sign = if x.is_sign_negative() { "-" } else { "" };
    if x == 0.0 {
        return format!("{sign}0.0");
    }
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let mut text = digits[..1].to_string();
        if n > 1 {
            text.push('.');
            text.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        let esign = if e < 0 { '-' } else { '+' };
        return format!("{sign}{text}e{esign}{:02}", e.abs());
    }
    let text = if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt >= n {
        format!("{}{}.0", digits, "0".repeat((decpt - n) as usize))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    format!("{sign}{text}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        for (x, want) in [
            (1.5, "1.5"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5e-7, "-2.5e-07"),
            (123.0, "123.0"),
            (0.1, "0.1"),
            (1.2345678901234567e20, "1.2345678901234567e+20"),
            (-0.0, "-0.0"),
            (f64::INFINITY, "inf"),
            // Exactly halfway between two shortest decimals: the even one.
            (f64::from(943.7_f32), "943.7000122070312"),
            (-690_600_950_624_650.0 - 0.25, "-690600950624650.2"),
            (5e-324, "5e-324"),
            // Below a power of two, where the nearest does not round-trip.
            (-5.960_464_477_539_063e-8, "-5.960464477539063e-08"),
            (1e22, "1e+22"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn strings_print_as_python_does() {
        assert_eq!(str_repr("a'b"), "\"a'b\"");
        assert_eq!(str_repr("a'b\""), "'a\\'b\"'");
        assert_eq!(str_repr("t\tn\u{200b}é😀\u{7f}"), "'t\\tn\\u200bé😀\\x7f'");
        assert_eq!(str_repr("\u{e0001}"), "'\\U000e0001'");
    }

    #[test]
    fn printable_follows_python() {
        for c in [' ', 'a', 'é', '😀', '\u{3000}'] {
            assert_eq!(is_printable(c), c == ' ' || c != '\u{3000}', "{c:?}");
        }
        for c in ['\n', '\u{85}', '\u{a0}', '\u{200b}', '\u{feff}', '\u{e000}'] {
            assert!(!is_printable(c), "{c:?}");
        }
    }
}
