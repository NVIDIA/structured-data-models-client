// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The text Python gives a float, the shortest that reads back to it, in
//! Python's layout: one of the two emulation modules this crate may hold
//! (`scripts/rust_core/emulation-allowlist.txt`, Phase 1 decisions A4).
//! The reference writes a float constant built from a value with it
//! (`Constant.from_value`); here it writes a float literal's value into
//! messages, and decides no validity (Phase 1 decisions, E9).

/// The text Python's `repr` gives `x`.
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
            // A float32's double, as a float target's classes spell it.
            (f64::from(3.0_f32), "3.0"),
            (f64::from(0.1_f32), "0.10000000149011612"),
            (f64::from(1e20_f32), "1.0000000200408773e+20"),
            (f64::from(1e-7_f32), "1.0000000116860974e-07"),
            (f64::from(1e-4_f32), "9.999999747378752e-05"),
            (f64::from(123_456_789_f32), "123456792.0"),
            (
                f64::from(f32::from_bits(0x0001_16c2)),
                "9.99994610111476e-41",
            ),
            (f64::from(-0.0_f32), "-0.0"),
            (f64::NAN, "nan"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }
}
