// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact numbers as Python holds them: `decimal.Decimal` and integers of any
//! size, which the wire's rules about what a column can carry read exactly.
//!
//! A finite [`Dec`] is `sign * coefficient * 10**exponent`, the coefficient
//! kept as its decimal digits, as `Decimal` keeps it: `Decimal('1.50')` and
//! `Decimal('1.5')` are equal and written differently. An integer is a
//! `Dec` of exponent zero.

use std::cmp::Ordering;

/// What a `Decimal` is besides a finite number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Special {
    Finite,
    Infinity,
    NaN,
    SNaN,
}

/// A `decimal.Decimal`, or an `int` of any size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dec {
    pub negative: bool,
    /// The coefficient's digits, without leading zeros (`"0"` for zero).
    pub digits: String,
    pub exponent: i64,
    pub special: Special,
}

impl Dec {
    /// The integer `value`.
    pub fn from_i128(value: i128) -> Self {
        Self {
            negative: value < 0,
            digits: value.unsigned_abs().to_string(),
            exponent: 0,
            special: Special::Finite,
        }
    }

    /// The integer written `text` (`-?[0-9]+`).
    pub fn integer(text: &str) -> Option<Self> {
        let (negative, body) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text.strip_prefix('+').unwrap_or(text)),
        };
        if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some(Self::finite(negative, body, 0))
    }

    fn finite(negative: bool, digits: &str, exponent: i64) -> Self {
        let trimmed = digits.trim_start_matches('0');
        Self {
            negative,
            digits: if trimmed.is_empty() {
                "0".to_string()
            } else {
                trimmed.to_string()
            },
            exponent,
            special: Special::Finite,
        }
    }

    /// `Decimal(text)`, or `None` where Python raises `InvalidOperation`.
    pub fn parse(text: &str) -> Option<Self> {
        let t = text.trim();
        let (negative, body) = match t.as_bytes().first() {
            Some(b'-') => (true, &t[1..]),
            Some(b'+') => (false, &t[1..]),
            _ => (false, t),
        };
        let lower = body.to_ascii_lowercase();
        let special = match lower.as_str() {
            "inf" | "infinity" => Some(Special::Infinity),
            "nan" => Some(Special::NaN),
            "snan" => Some(Special::SNaN),
            _ => None,
        };
        if let Some(special) = special {
            return Some(Self {
                negative,
                digits: "0".to_string(),
                exponent: 0,
                special,
            });
        }
        let body = body.replace('_', "");
        let (mantissa, exponent) = match body.find(['e', 'E']) {
            Some(i) => (&body[..i], body[i + 1..].parse::<i64>().ok()?),
            None => (body.as_str(), 0),
        };
        let (int, frac) = match mantissa.split_once('.') {
            Some((a, b)) => (a, b),
            None => (mantissa, ""),
        };
        if int.is_empty() && frac.is_empty() {
            return None;
        }
        if !int.bytes().all(|b| b.is_ascii_digit()) || !frac.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some(Self::finite(
            negative,
            &format!("{int}{frac}"),
            exponent - frac.len() as i64,
        ))
    }

    /// `Decimal(repr(x))` of a finite float.
    pub fn from_f64(x: f64) -> Self {
        Self::parse(&sdm_pql::pyrepr::float_repr(x)).unwrap_or_else(|| Self::from_i128(0))
    }

    pub fn is_finite(&self) -> bool {
        self.special == Special::Finite
    }

    pub fn is_nan(&self) -> bool {
        matches!(self.special, Special::NaN | Special::SNaN)
    }

    pub fn is_zero(&self) -> bool {
        self.is_finite() && self.digits == "0"
    }

    /// The digits and exponent with trailing zeros of the coefficient moved
    /// into the exponent: the same value, written shortest.
    fn reduced(&self) -> (String, i64) {
        if self.digits == "0" {
            return ("0".to_string(), 0);
        }
        let kept = self.digits.trim_end_matches('0');
        (
            kept.to_string(),
            self.exponent + (self.digits.len() - kept.len()) as i64,
        )
    }

    /// `value == value.to_integral_value()`, for a finite value.
    pub fn is_integral(&self) -> bool {
        if !self.is_finite() {
            return false;
        }
        self.reduced().1 >= 0
    }

    /// `Decimal.adjusted()`: the exponent of the most significant digit.
    pub fn adjusted(&self) -> i64 {
        self.exponent + self.digits.len() as i64 - 1
    }

    /// `int(value)` of a finite value: truncated toward zero, as digits.
    pub fn trunc_digits(&self) -> String {
        if self.exponent >= 0 {
            if self.digits == "0" {
                return "0".to_string();
            }
            return format!("{}{}", self.digits, "0".repeat(self.exponent as usize));
        }
        let cut = -self.exponent as usize;
        if cut >= self.digits.len() {
            return "0".to_string();
        }
        let kept = &self.digits[..self.digits.len() - cut];
        let trimmed = kept.trim_start_matches('0');
        if trimmed.is_empty() {
            "0".to_string()
        } else {
            trimmed.to_string()
        }
    }

    /// `int(value)` of a finite value.
    pub fn trunc(&self) -> Dec {
        let digits = self.trunc_digits();
        Self::finite(self.negative && digits != "0", &digits, 0)
    }

    /// The value as an `i128`, where it is an integer that fits.
    pub fn to_i128(&self) -> Option<i128> {
        if !self.is_integral() {
            return None;
        }
        let digits = self.trunc_digits();
        if digits.len() > 39 {
            return None;
        }
        let magnitude: u128 = digits.parse().ok()?;
        if self.negative {
            0i128.checked_sub_unsigned(magnitude)
        } else {
            i128::try_from(magnitude).ok()
        }
    }

    /// The value as an `i64`, where it is an integer that fits.
    pub fn to_i64(&self) -> Option<i64> {
        self.to_i128().and_then(|v| i64::try_from(v).ok())
    }

    /// `float(value)`: correctly rounded, `inf` past the largest double.
    pub fn to_f64(&self) -> f64 {
        let sign = if self.negative { -1.0 } else { 1.0 };
        match self.special {
            Special::Infinity => sign * f64::INFINITY,
            Special::NaN | Special::SNaN => f64::NAN,
            Special::Finite => {
                let (digits, exponent) = self.reduced();
                let magnitude = if digits == "0" {
                    0.0
                } else if exponent + digits.len() as i64 > 400 {
                    f64::INFINITY
                } else if exponent + (digits.len() as i64) < -400 {
                    0.0
                } else {
                    format!("{digits}e{exponent}").parse().unwrap_or(f64::NAN)
                };
                sign * magnitude
            }
        }
    }

    /// Compare two finite values.
    pub fn cmp_finite(&self, other: &Dec) -> Ordering {
        let a_zero = self.digits == "0";
        let b_zero = other.digits == "0";
        let a_neg = self.negative && !a_zero;
        let b_neg = other.negative && !b_zero;
        match (a_neg, b_neg) {
            (false, true) => return Ordering::Greater,
            (true, false) => return Ordering::Less,
            _ => {}
        }
        let magnitude = cmp_magnitude(self, other);
        if a_neg {
            magnitude.reverse()
        } else {
            magnitude
        }
    }

    /// Whether two values are equal as Python compares them: a NaN equals
    /// nothing.
    pub fn equals(&self, other: &Dec) -> bool {
        match (self.special, other.special) {
            (Special::Finite, Special::Finite) => self.cmp_finite(other) == Ordering::Equal,
            (Special::Infinity, Special::Infinity) => self.negative == other.negative,
            _ => false,
        }
    }

    /// `format(value, 'f')`.
    pub fn format_f(&self) -> String {
        let sign = if self.negative { "-" } else { "" };
        match self.special {
            Special::Infinity => return format!("{sign}Infinity"),
            Special::NaN => return format!("{sign}NaN"),
            Special::SNaN => return format!("{sign}sNaN"),
            Special::Finite => {}
        }
        let body = if self.exponent >= 0 {
            if self.digits == "0" {
                "0".to_string()
            } else {
                format!("{}{}", self.digits, "0".repeat(self.exponent as usize))
            }
        } else {
            let places = (-self.exponent) as usize;
            let padded = if self.digits.len() <= places {
                format!(
                    "{}{}",
                    "0".repeat(places + 1 - self.digits.len()),
                    self.digits
                )
            } else {
                self.digits.clone()
            };
            let point = padded.len() - places;
            format!("{}.{}", &padded[..point], &padded[point..])
        };
        format!("{sign}{body}")
    }

    /// `str(value)`: Python's to-scientific-string.
    pub fn to_sci(&self) -> String {
        let sign = if self.negative { "-" } else { "" };
        match self.special {
            Special::Infinity => return format!("{sign}Infinity"),
            Special::NaN => return format!("{sign}NaN"),
            Special::SNaN => return format!("{sign}sNaN"),
            Special::Finite => {}
        }
        let digits = &self.digits;
        let exponent = self.exponent;
        let adjusted = exponent + digits.len() as i64 - 1;
        if exponent <= 0 && adjusted >= -6 {
            if exponent == 0 {
                return format!("{sign}{digits}");
            }
            let places = (-exponent) as usize;
            let padded = if digits.len() <= places {
                format!("{}{}", "0".repeat(places + 1 - digits.len()), digits)
            } else {
                digits.clone()
            };
            let point = padded.len() - places;
            return format!("{sign}{}.{}", &padded[..point], &padded[point..]);
        }
        let mut out = format!("{sign}{}", &digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('E');
        if adjusted >= 0 {
            out.push('+');
        }
        out.push_str(&adjusted.to_string());
        out
    }

    /// `repr(value)` of a `Decimal`.
    pub fn repr(&self) -> String {
        format!("Decimal('{}')", self.to_sci())
    }

    /// `value.normalize()` under the default context of 28 digits.
    pub fn normalize(&self) -> Dec {
        if !self.is_finite() {
            return self.clone();
        }
        let (mut digits, mut exponent) = self.reduced();
        if digits.len() > 28 {
            let rounded = round_half_even(&digits, digits.len() - 28);
            exponent += (digits.len() - 28) as i64;
            digits = rounded;
            let kept = digits.trim_end_matches('0').to_string();
            exponent += (digits.len() - kept.len()) as i64;
            digits = kept;
        }
        if digits == "0" {
            return Self::finite(self.negative, "0", 0);
        }
        Self::finite(self.negative, &digits, exponent)
    }

    /// `Decimal(x)` of a finite float: its exact binary value in decimal.
    pub fn exact_f64(x: f64) -> Dec {
        if x == 0.0 {
            return Self::finite(x.is_sign_negative(), "0", 0);
        }
        let bits = x.abs().to_bits();
        let raw_exponent = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1 << 52) - 1);
        let (mantissa, exponent) = if raw_exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), raw_exponent - 1075)
        };
        let mut value = BigUint::from_u64(mantissa);
        if exponent >= 0 {
            for _ in 0..exponent {
                value.mul_small(2);
            }
            return Self::finite(x < 0.0, &value.to_string(), 0);
        }
        // m * 2**-k is m * 5**k * 10**-k.
        for _ in 0..(-exponent) {
            value.mul_small(5);
        }
        let mut digits = value.to_string();
        let mut exponent = exponent;
        while exponent < 0 && digits.len() > 1 && digits.ends_with('0') {
            digits.pop();
            exponent += 1;
        }
        Self::finite(x < 0.0, &digits, exponent)
    }

    /// The decimal digits of the integer this float holds exactly.
    pub fn of_whole_f64(x: f64) -> Option<Dec> {
        if !x.is_finite() || x.fract() != 0.0 {
            return None;
        }
        if x.abs() < 1e18 {
            return Some(Dec::from_i128(x as i128));
        }
        let bits = x.abs().to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64 - 1075;
        let mantissa = (bits & ((1 << 52) - 1)) | (1 << 52);
        let mut value = BigUint::from_u64(mantissa);
        for _ in 0..exponent.max(0) {
            value.mul_small(2);
        }
        Some(Self::finite(x < 0.0, &value.to_string(), 0))
    }
}

fn cmp_magnitude(a: &Dec, b: &Dec) -> Ordering {
    let (ad, ae) = a.reduced();
    let (bd, be) = b.reduced();
    if ad == "0" || bd == "0" {
        return (ad != "0").cmp(&(bd != "0"));
    }
    let a_top = ae + ad.len() as i64;
    let b_top = be + bd.len() as i64;
    if a_top != b_top {
        return a_top.cmp(&b_top);
    }
    // Same magnitude: compare digit by digit, padding the shorter.
    let width = ad.len().max(bd.len());
    let pa = format!("{ad:0<width$}");
    let pb = format!("{bd:0<width$}");
    pa.cmp(&pb)
}

/// `digits` with its last `drop` digits rounded away, half to even.
fn round_half_even(digits: &str, drop: usize) -> String {
    let keep = &digits[..digits.len() - drop];
    let rest = &digits[digits.len() - drop..];
    let first = rest.as_bytes()[0];
    let tail_nonzero = rest[1..].bytes().any(|b| b != b'0');
    let last_odd = keep.bytes().last().is_some_and(|b| (b - b'0') % 2 == 1);
    let up = first > b'5' || (first == b'5' && (tail_nonzero || last_odd));
    if !up {
        return keep.to_string();
    }
    let mut bytes = keep.as_bytes().to_vec();
    let mut i = bytes.len();
    loop {
        if i == 0 {
            bytes.insert(0, b'1');
            break;
        }
        i -= 1;
        if bytes[i] == b'9' {
            bytes[i] = b'0';
        } else {
            bytes[i] += 1;
            break;
        }
    }
    String::from_utf8(bytes).unwrap_or_default()
}

/// An unsigned integer of any size, base 10**9.
struct BigUint {
    limbs: Vec<u32>,
}

impl BigUint {
    fn from_u64(value: u64) -> Self {
        let mut limbs = Vec::new();
        let mut v = value;
        while v > 0 {
            limbs.push((v % 1_000_000_000) as u32);
            v /= 1_000_000_000;
        }
        Self { limbs }
    }

    fn mul_small(&mut self, factor: u32) {
        let mut carry: u64 = 0;
        for limb in &mut self.limbs {
            let product = u64::from(*limb) * u64::from(factor) + carry;
            *limb = (product % 1_000_000_000) as u32;
            carry = product / 1_000_000_000;
        }
        while carry > 0 {
            self.limbs.push((carry % 1_000_000_000) as u32);
            carry /= 1_000_000_000;
        }
    }
}

impl std::fmt::Display for BigUint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Some(top) = self.limbs.last() else {
            return f.write_str("0");
        };
        write!(f, "{top}")?;
        for limb in self.limbs.iter().rev().skip(1) {
            write!(f, "{limb:09}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(text: &str) -> Dec {
        Dec::parse(text).expect("a decimal")
    }

    #[test]
    fn decimals_read_and_write_as_python_does() {
        assert_eq!(d("1E+400").format_f(), format!("1{}", "0".repeat(400)));
        assert_eq!(d("1.50").format_f(), "1.50");
        assert_eq!(d("1E-3").format_f(), "0.001");
        assert_eq!(d("12.3456").to_sci(), "12.3456");
        assert_eq!(d("1E+400").to_sci(), "1E+400");
        assert_eq!(d("0.0000001").to_sci(), "1E-7");
        assert_eq!(d("123").repr(), "Decimal('123')");
        assert!(d("1.00").is_integral());
        assert!(!d("1.5").is_integral());
        assert_eq!(d("1.00").cmp_finite(&d("1")), Ordering::Equal);
        assert_eq!(d("-2").cmp_finite(&d("1")), Ordering::Less);
        assert_eq!(d("9223372036854775808").to_i64(), None);
        assert_eq!(d("9223372036854775807").to_i64(), Some(i64::MAX));
        assert_eq!(d("1E+400").to_f64(), f64::INFINITY);
        assert_eq!(d("12.3456").to_f64(), 12.3456);
        assert_eq!(d("1.50").normalize().to_sci(), "1.5");
        assert_eq!(d("100").normalize().to_sci(), "1E+2");
    }

    #[test]
    fn whole_floats_give_their_exact_digits() {
        assert_eq!(
            Dec::of_whole_f64(1e22).map(|v| v.format_f()),
            Some("10000000000000000000000".to_string())
        );
        assert_eq!(
            Dec::of_whole_f64(9007199254740992.0).map(|v| v.format_f()),
            Some("9007199254740992".to_string())
        );
    }
}
