// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `decimal.Decimal` as an Arrow `decimal128` holds it: its unscaled digits
//! and its scale.
//!
//! Python compares and hashes a decimal exactly against ints, floats and
//! other decimals, and the core's keys and labels must find equal what
//! Python finds equal, so the rules live here once: [`identity`],
//! [`compare`] and [`compare_float`]. [`cast_int`], [`cast_float`] and
//! [`cast_text`] are pyarrow's casts to `decimal128`, which
//! `pd.Series(values, dtype=pd.ArrowDtype(pa.decimal128(p, s)))` makes.

use std::cmp::Ordering;

/// What a decimal equals, and hashes as: the integer it equals, else the
/// float it equals, else itself without trailing zeros (`hash(Decimal)`
/// agrees with `hash(int)` and `hash(float)` of the same value).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Identity {
    Int(i128),
    Float(f64),
    /// Digits and a scale, without trailing zeros (as far as an `i8` scale
    /// holds them).
    Decimal(i128, i8),
}

/// The [`Identity`] of `digits` at `scale`.
pub fn identity(digits: i128, scale: i8) -> Identity {
    let (mut digits, mut scale) = (digits, i32::from(scale));
    while scale > 0 && digits % 10 == 0 {
        digits /= 10;
        scale -= 1;
    }
    let short = |scale: i32| i8::try_from(scale).expect("within an i8 throughout");
    if scale <= 0 {
        let power = scale.unsigned_abs();
        if let Some(v) = 10i128
            .checked_pow(power)
            .and_then(|f| digits.checked_mul(f))
        {
            return Identity::Int(v);
        }
        // An integer past i128 is a float when its odd part fits in 53
        // bits: digits * 10**power = odd * 2**twos.
        let twos = digits.trailing_zeros().min(127);
        if let Some(odd) = 5i128
            .checked_pow(power)
            .and_then(|f| (digits >> twos).checked_mul(f))
            && odd.unsigned_abs() < (1u128 << 53)
        {
            let value = odd as f64 * 2f64.powi((twos + power) as i32);
            if value.is_finite() {
                return Identity::Float(value);
            }
        }
        // Its trailing zeros go into the scale, as far as an `i8` holds it.
        while digits != 0 && digits % 10 == 0 && scale > i32::from(i8::MIN) {
            digits /= 10;
            scale -= 1;
        }
        return Identity::Decimal(digits, short(scale));
    }
    // Without trailing zeros, digits / 10**scale is a float only when
    // 5**scale divides the digits, leaving an odd numerator over 2**scale
    // that fits in 53 bits.
    if let Some(five) = 5i128.checked_pow(scale.unsigned_abs())
        && digits % five == 0
        && (digits / five).unsigned_abs() < (1u128 << 53)
    {
        return Identity::Float((digits / five) as f64 / 2f64.powi(scale));
    }
    Identity::Decimal(digits, short(scale))
}

/// Two decimals in order, each digits at a scale, exactly.
pub fn compare(a: (i128, i8), b: (i128, i8)) -> Ordering {
    let scale = a.1.max(b.1);
    let lift = |(v, s): (i128, i8)| {
        10i128
            .checked_pow(u32::try_from(i32::from(scale) - i32::from(s)).ok()?)
            .and_then(|f| v.checked_mul(f))
    };
    match (lift(a), lift(b)) {
        (Some(x), Some(y)) => x.cmp(&y),
        _ => Exact::scaled(a.0, a.1).cmp(&Exact::scaled(b.0, b.1)),
    }
}

/// A decimal against a float, exactly, as Python compares them; `None`
/// against `NaN`, which no decimal is equal to and which Python refuses to
/// order a decimal against (`decimal.InvalidOperation`).
pub fn compare_float(a: (i128, i8), f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    if f.is_infinite() {
        return Some(if f > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }
    Some(Exact::scaled(a.0, a.1).cmp(&Exact::float(f)))
}

/// Whether `x` has at most `precision` digits.
fn fits(x: i128, precision: u8) -> bool {
    10u128
        .checked_pow(u32::from(precision))
        .is_none_or(|limit| x.unsigned_abs() < limit)
}

/// `pa.array([v], type=pa.decimal128(precision, scale))` of a Python int:
/// the unscaled digits, or `None` where pyarrow refuses the value
/// (`ArrowInvalid`). For an `int64` and a scale that is not negative,
/// `pa.array([v]).cast(pa.decimal128(precision, scale))` is the same;
/// with a negative scale, that cast refuses every value.
pub fn cast_int(v: i128, precision: u8, scale: i8) -> Option<i128> {
    let x = if scale >= 0 {
        v.checked_mul(10i128.checked_pow(scale.unsigned_abs().into())?)?
    } else {
        let f = 10i128.checked_pow(scale.unsigned_abs().into())?;
        if v % f != 0 {
            return None;
        }
        v / f
    };
    fits(x, precision).then_some(x)
}

/// `pa.array([f]).cast(pa.decimal128(precision, scale))` of a float: the
/// unscaled digits, or `None` where pyarrow refuses it (`ArrowInvalid`), an
/// infinity or `NaN` included.
///
/// This is Arrow's `Decimal128::FromReal` step by step, rounding where it
/// rounds, so it gives what pyarrow gives rather than the closest decimal.
/// The two differ in two places:
/// - a scale above 22 makes Arrow multiply by ten and shift right a few
///   digits at a time, rounding each time, so the last digits can be off:
///   `0.1` as `decimal128(38, 38)` ends `...78270210` in Arrow, where
///   `0.1 * 10**38` rounded half to even ends `...78270212`;
/// - a negative scale makes Arrow multiply in `f64` by the `f64` nearest
///   `10**scale`, then round that product half to even.
///
/// For any other scale it is `f * 10**scale` rounded half to even, exactly.
pub fn cast_float(f: f64, precision: u8, scale: i8) -> Option<i128> {
    if !f.is_finite() {
        return None;
    }
    let x = from_positive_real(f.abs(), precision, scale)?;
    Some(if f < 0.0 { -x } else { x })
}

/// The `f64` nearest `10**exponent`, as Arrow's table of powers of ten
/// holds it.
fn power_of_ten(exponent: i32) -> Option<f64> {
    format!("1e{exponent}").parse().ok()
}

/// Arrow's `DecimalRealConversion::FromPositiveReal` for `Decimal128` and
/// `double` (`arrow/util/decimal.cc`).
fn from_positive_real(real: f64, precision: u8, scale: i8) -> Option<i128> {
    // Arrow's largest decimal128 precision, and the decimal digits a double's
    // mantissa is given.
    const MAX_PRECISION: i32 = 38;
    const MANTISSA_DIGITS: i32 = 16;
    const MANTISSA_BITS: i32 = 53;
    let (precision, scale) = (i32::from(precision), i32::from(scale));
    if scale < 0 {
        // `FromPositiveRealApprox`.
        let x = (real * power_of_ten(scale)?).round_ties_even();
        let limit = power_of_ten(precision)?;
        if x <= -limit || x >= limit {
            return None;
        }
        // An integral double below 10**38 converts exactly.
        return Some(x as i128);
    }
    // Arrow refuses a magnitude above 10**(precision - scale) as a double,
    // which can be below the exact power (1e23 < 10**23).
    if real > power_of_ten(precision - scale)? {
        return None;
    }
    // real = mantissa * 2**k, the mantissa an integer of 53 bits (fewer
    // below the normal range), as `frexp` and `ldexp` give it.
    let (mantissa, k) = if real == 0.0 {
        (0u128, -MANTISSA_BITS)
    } else {
        let bits = real.to_bits();
        let biased = i32::try_from(bits >> 52).ok()?;
        let fraction = bits & ((1u64 << 52) - 1);
        let (m, e) = if biased == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1u64 << 52), biased - 1075)
        };
        // `frexp` leaves the mantissa's top bit at bit 52 and `ldexp`
        // brings it back, so a subnormal keeps its trailing zeros in the
        // exponent rather than in the mantissa.
        let shift = i32::try_from(m.leading_zeros()).ok()? - 11;
        (u128::from(m) << shift, e - shift)
    };
    let ten = |e: i32| 10u128.checked_pow(u32::try_from(e).ok()?);
    let mut x = mantissa;
    if k < 0 {
        let mut right_shift_by = -k;
        let mut mul_by_ten_to = scale;
        let safe = MAX_PRECISION - MANTISSA_DIGITS;
        if mul_by_ten_to <= safe {
            x = rounded_right_shift(x.checked_mul(ten(mul_by_ten_to)?)?, right_shift_by);
        } else {
            x = x.checked_mul(ten(safe)?)?;
            mul_by_ten_to -= safe;
            let mul_step = (MAX_PRECISION - precision).max(1);
            let (mut total_exp, mut total_shift) = (0, 0);
            while mul_by_ten_to > 0 && right_shift_by > 0 {
                let exp = mul_by_ten_to.min(mul_step);
                total_exp += exp;
                let bits = right_shift_by.min(ceil_log2_power_of_ten(total_exp)? - total_shift);
                total_shift += bits;
                x = rounded_right_shift(x, bits);
                right_shift_by -= bits;
                x = x.checked_mul(ten(exp)?)?;
                mul_by_ten_to -= exp;
            }
            if mul_by_ten_to > 0 {
                x = x.checked_mul(ten(mul_by_ten_to)?)?;
            }
            if right_shift_by > 0 {
                x = rounded_right_shift(x, right_shift_by);
            }
        }
    } else {
        x = x
            .checked_mul(ten(scale)?)?
            .checked_mul(1u128.checked_shl(u32::try_from(k).ok()?)?)?;
    }
    let x = i128::try_from(x).ok()?;
    fits(x, u8::try_from(precision).ok()?).then_some(x)
}

/// `x / 2**bits`, rounded half to even.
fn rounded_right_shift(x: u128, bits: i32) -> u128 {
    let Ok(bits) = u32::try_from(bits) else {
        return x;
    };
    if bits == 0 {
        return x;
    }
    if bits >= 128 {
        // All of x is shifted out: it is below half of 2**bits.
        return 0;
    }
    let (q, r) = (x >> bits, x & ((1u128 << bits) - 1));
    let half = 1u128 << (bits - 1);
    if r > half || (r == half && q % 2 == 1) {
        q + 1
    } else {
        q
    }
}

/// The least `n` with `2**n >= 10**exponent`: Arrow's
/// `kCeilLog2PowersOfTen`.
fn ceil_log2_power_of_ten(exponent: i32) -> Option<i32> {
    if exponent == 0 {
        return Some(0);
    }
    // 10**exponent is no power of two, so this is its bit length.
    let power = 10u128.checked_pow(u32::try_from(exponent).ok()?)?;
    i32::try_from(128 - power.leading_zeros()).ok()
}

/// `pa.array([s]).cast(pa.decimal128(precision, scale))`: the unscaled
/// digits, or `None` where pyarrow refuses the text (`ArrowInvalid`): it is
/// not a decimal number, rescaling it would lose digits, or it does not fit.
///
/// Arrow reads the digits into 128 bits without a check, so text whose
/// digits pass `i128` comes out as some other number there; it is refused
/// here.
pub fn cast_text(s: &str, precision: u8, scale: i8) -> Option<i128> {
    let parsed = Parsed::read(s.as_bytes())?;
    let mut digits: i128 = 0;
    for d in parsed.whole.iter().chain(parsed.fraction) {
        digits = digits.checked_mul(10)?.checked_add(i128::from(d - b'0'))?;
    }
    let mut parsed_scale = i64::try_from(parsed.fraction.len()).ok()? - i64::from(parsed.exponent);
    if parsed_scale < 0 {
        if parsed_scale < -38 {
            return None;
        }
        digits = digits.checked_mul(10i128.checked_pow(u32::try_from(-parsed_scale).ok()?)?)?;
        parsed_scale = 0;
    }
    let delta = i64::from(scale) - parsed_scale;
    let x = if delta >= 0 {
        digits.checked_mul(10i128.checked_pow(u32::try_from(delta).ok()?)?)?
    } else {
        let f = 10i128.checked_pow(u32::try_from(-delta).ok()?)?;
        if digits % f != 0 {
            return None;
        }
        digits / f
    };
    let x = if parsed.negative { -x } else { x };
    fits(x, precision).then_some(x)
}

/// A decimal number's text as Arrow's parser reads it: a sign, digits, a
/// point, digits and an exponent.
struct Parsed<'a> {
    negative: bool,
    whole: &'a [u8],
    fraction: &'a [u8],
    exponent: i32,
}

impl<'a> Parsed<'a> {
    fn read(s: &'a [u8]) -> Option<Self> {
        let digits_from =
            |start: usize| start + s[start..].iter().take_while(|b| b.is_ascii_digit()).count();
        let mut pos = 0;
        let negative = s.first() == Some(&b'-');
        if matches!(s.first(), Some(b'-' | b'+')) {
            pos = 1;
        }
        let end = digits_from(pos);
        let whole = &s[pos..end];
        pos = end;
        let mut fraction: &[u8] = &[];
        if s.get(pos) == Some(&b'.') {
            let end = digits_from(pos + 1);
            fraction = &s[pos + 1..end];
            pos = end;
        }
        if whole.is_empty() && fraction.is_empty() {
            return None;
        }
        let mut exponent = 0;
        if matches!(s.get(pos), Some(b'e' | b'E')) {
            pos += 1;
            if s.get(pos) == Some(&b'+') {
                pos += 1;
            }
            let text = std::str::from_utf8(&s[pos..]).ok()?;
            let unsigned = text.strip_prefix('-').unwrap_or(text);
            if unsigned.is_empty() || !unsigned.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            exponent = text.parse().ok()?;
            pos = s.len();
        }
        (pos == s.len()).then_some(Parsed {
            negative,
            whole,
            fraction,
            exponent,
        })
    }
}

/// A number written out exactly: digits times ten to an exponent.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Exact {
    negative: bool,
    /// Most significant first, without leading or trailing zeros; empty for
    /// zero.
    digits: Vec<u8>,
    exponent: i32,
}

impl Exact {
    fn new(negative: bool, mut digits: Vec<u8>, mut exponent: i32) -> Self {
        while digits.last() == Some(&0) {
            digits.pop();
            exponent += 1;
        }
        let lead = digits.iter().take_while(|&&d| d == 0).count();
        digits.drain(..lead);
        Exact {
            negative: negative && !digits.is_empty(),
            digits,
            exponent,
        }
    }

    fn scaled(v: i128, scale: i8) -> Self {
        let digits = v
            .unsigned_abs()
            .to_string()
            .bytes()
            .map(|b| b - b'0')
            .collect();
        Exact::new(v < 0, digits, -i32::from(scale))
    }

    /// A finite float's exact value: its mantissa times a power of two,
    /// which is the mantissa times 5**-power over 10**-power.
    fn float(f: f64) -> Self {
        let bits = f.to_bits();
        let biased = i32::try_from((bits >> 52) & 0x7ff).expect("eleven bits");
        let fraction = bits & ((1u64 << 52) - 1);
        let (mantissa, power) = if biased == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1u64 << 52), biased - 1075)
        };
        let mut digits: Vec<u8> = mantissa
            .to_string()
            .bytes()
            .rev()
            .map(|b| b - b'0')
            .collect();
        let (base, chunk, step) = if power >= 0 {
            (2u64, 1u64 << 30, 30)
        } else {
            (5u64, 5u64.pow(13), 13)
        };
        let mut left = power.unsigned_abs();
        while left >= step {
            times(&mut digits, chunk);
            left -= step;
        }
        times(&mut digits, base.pow(left));
        digits.reverse();
        Exact::new(f.is_sign_negative(), digits, power.min(0))
    }

    /// The power of ten the number's magnitude is below, and not a tenth
    /// of.
    fn top(&self) -> i64 {
        i64::try_from(self.digits.len()).unwrap_or(i64::MAX) + i64::from(self.exponent)
    }

    fn cmp_magnitude(&self, other: &Exact) -> Ordering {
        match (self.digits.is_empty(), other.digits.is_empty()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self
                .top()
                .cmp(&other.top())
                .then_with(|| self.digits.cmp(&other.digits)),
        }
    }
}

impl PartialOrd for Exact {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Exact {
    fn cmp(&self, other: &Self) -> Ordering {
        let sign = |e: &Exact| match (e.digits.is_empty(), e.negative) {
            (true, _) => 0,
            (false, true) => -1,
            (false, false) => 1,
        };
        match sign(self).cmp(&sign(other)) {
            Ordering::Equal if self.negative => self.cmp_magnitude(other).reverse(),
            Ordering::Equal => self.cmp_magnitude(other),
            unequal => unequal,
        }
    }
}

/// Little-endian decimal digits times `k`.
fn times(digits: &mut Vec<u8>, k: u64) {
    let mut carry: u64 = 0;
    for d in digits.iter_mut() {
        let v = u64::from(*d) * k + carry;
        *d = u8::try_from(v % 10).expect("a digit");
        carry = v / 10;
    }
    while carry > 0 {
        digits.push(u8::try_from(carry % 10).expect("a digit"));
        carry /= 10;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact_text(e: &Exact) -> String {
        let digits: String = e.digits.iter().map(|d| char::from(b'0' + d)).collect();
        format!(
            "{}{}e{}",
            if e.negative { "-" } else { "" },
            digits,
            e.exponent
        )
    }

    #[test]
    fn a_float_is_written_out_exactly() {
        // `Decimal(x).as_tuple()` of each float, from Python.
        let cases = [
            (
                0.1,
                "1000000000000000055511151231257827021181583404541015625e-55",
            ),
            (-2.5, "-25e-1"),
            (1e23, "99999999999999991611392e0"),
        ];
        for (f, want) in cases {
            assert_eq!(exact_text(&Exact::float(f)), want, "{f}");
        }
        // 751 digits, starting 494065645841246544, at exponent -1074.
        let tiny = Exact::float(5e-324);
        assert_eq!(tiny.digits.len(), 751);
        assert_eq!(tiny.exponent, -1074);
        assert_eq!(exact_text(&tiny)[..18], *"494065645841246544");
    }

    #[test]
    fn identities_match_python_hashes() {
        // Python: the int, float or Decimal each equals and hashes as.
        assert_eq!(identity(1200, 2), Identity::Int(12));
        assert_eq!(identity(0, 5), Identity::Int(0));
        assert_eq!(identity(0, -3), Identity::Int(0));
        assert_eq!(identity(-15, 1), Identity::Float(-1.5));
        assert_eq!(identity(1, 1), Identity::Decimal(1, 1));
        assert_eq!(identity(5, -3), Identity::Int(5000));
        // 2**100 * 10**10 = 5**10 * 2**110 is a float past i128.
        let two_100 = 1i128 << 100;
        assert_eq!(
            identity(two_100, -10),
            Identity::Float(9_765_625.0 * 2f64.powi(110))
        );
        // 3 * 10**40 is not a float: 3 * 5**40 needs 95 bits.
        assert_eq!(identity(3, -40), Identity::Decimal(3, -40));
        // Decimal(10) * 10**39 == Decimal(1) * 10**40, and they hash alike.
        assert_eq!(identity(10, -39), Identity::Decimal(1, -40));
        assert_eq!(identity(300, -38), Identity::Decimal(3, -40));
    }

    #[test]
    fn decimals_compare_exactly() {
        assert_eq!(compare((10, 1), (1, 0)), Ordering::Equal);
        assert_eq!(compare((-5, 0), (3, 0)), Ordering::Less);
        // Past i128 when brought to one scale.
        let big = (i128::MAX, 0);
        assert_eq!(compare(big, (1, 38)), Ordering::Greater);
        assert_eq!(compare((1, -40), big), Ordering::Greater);
        assert_eq!(compare((-1, -40), (1, 38)), Ordering::Less);
        assert_eq!(compare((-1, -40), (-1, -39)), Ordering::Less);
    }

    #[test]
    fn decimals_compare_with_floats_as_python_does() {
        // Python: Decimal('0.1') == 0.1 is False, and Decimal('0.1') < 0.1.
        assert_eq!(compare_float((1, 1), 0.1), Some(Ordering::Less));
        assert_eq!(compare_float((5, 1), 0.5), Some(Ordering::Equal));
        assert_eq!(compare_float((-5, 1), -0.5), Some(Ordering::Equal));
        assert_eq!(compare_float((0, 3), -0.0), Some(Ordering::Equal));
        assert_eq!(compare_float((1, -100), 1.0), Some(Ordering::Greater));
        assert_eq!(compare_float((1, 0), f64::INFINITY), Some(Ordering::Less));
        assert_eq!(
            compare_float((1, 0), f64::NEG_INFINITY),
            Some(Ordering::Greater)
        );
        assert_eq!(compare_float((1, 0), f64::NAN), None);
        // 2**53 + 1 against the float 2**53.
        assert_eq!(
            compare_float(((1i128 << 53) + 1, 0), 2f64.powi(53)),
            Some(Ordering::Greater)
        );
        assert_eq!(compare_float((1, 0), 5e-324), Some(Ordering::Greater));
        assert_eq!(compare_float((0, 0), 5e-324), Some(Ordering::Less));
    }

    #[test]
    fn ints_cast_as_pyarrow_casts_them() {
        // pd.Series([v], dtype=pd.ArrowDtype(pa.decimal128(p, s))), from
        // Python.
        assert_eq!(cast_int(9_999_999_999, 10, 0), Some(9_999_999_999));
        assert_eq!(cast_int(10_000_000_000, 10, 0), None);
        assert_eq!(cast_int(99_999_999, 10, 2), Some(9_999_999_900));
        assert_eq!(cast_int(100_000_000, 10, 2), None);
        assert_eq!(cast_int(-99_999_999, 10, 2), Some(-9_999_999_900));
        assert_eq!(cast_int(i64::MAX.into(), 10, 0), None);
        assert_eq!(cast_int(i64::MIN.into(), 10, 2), None);
        assert_eq!(cast_int(i64::MIN.into(), 19, 0), Some(i64::MIN.into()));
        // Past int64, as `pa.array([v], type=...)` takes a Python int.
        assert_eq!(cast_int(1 << 63, 38, 0), Some(1 << 63));
        assert_eq!(cast_int(u64::MAX.into(), 20, 0), Some(u64::MAX.into()));
        assert_eq!(cast_int(u64::MAX.into(), 19, 0), None);
        // A negative scale: "Rescaling Decimal value would cause data loss".
        assert_eq!(cast_int(100, 10, -2), Some(1));
        assert_eq!(cast_int(150, 10, -2), None);
    }

    #[test]
    fn floats_cast_as_pyarrow_casts_them() {
        let cases: [(f64, u8, i8, Option<i128>); 17] = [
            (1.5, 10, 0, Some(2)),
            (2.5, 10, 0, Some(2)),
            (0.5, 10, 0, Some(0)),
            (1.234, 10, 0, Some(1)),
            (-0.0, 10, 0, Some(0)),
            (5e-324, 10, 0, Some(0)),
            (9_999_999_998.5, 10, 0, Some(9_999_999_998)),
            (9_999_999_999.5, 10, 0, None),
            (-9_999_999_999.5, 10, 0, None),
            (1e20, 10, 0, None),
            (1e300, 10, 0, None),
            (f64::INFINITY, 10, 0, None),
            (0.005, 10, 2, Some(1)),
            (0.015, 10, 2, Some(1)),
            (0.025, 10, 2, Some(3)),
            (1.005, 10, 2, Some(100)),
            (99_999_999.996, 10, 2, None),
        ];
        for (f, p, s, want) in cases {
            assert_eq!(cast_float(f, p, s), want, "{f} as decimal128({p}, {s})");
        }
        assert_eq!(cast_float(99_999_999.994, 10, 2), Some(9_999_999_999));
        assert_eq!(cast_float(9_999_999_998.5, 10, 2), None);
        assert_eq!(cast_float(1.5, 12, 2), Some(150));
        assert_eq!(cast_float(-1.5, 12, 2), Some(-150));
        assert_eq!(cast_float(f64::NAN, 10, 0), None);
        assert_eq!(cast_float(5e-324, 38, 38), Some(0));
    }

    #[test]
    fn floats_cast_as_arrow_rounds_them_at_large_and_negative_scales() {
        // pd.Series([f], dtype=pd.ArrowDtype(pa.decimal128(p, s))), from
        // Python. Up to scale 22 Arrow rounds f * 10**s once, exactly.
        assert_eq!(
            cast_float(0.1, 38, 23),
            Some(10_000_000_000_000_000_555_112)
        );
        // Above it, Arrow rounds as it goes: the exact result of these ends
        // ...270212 and ...936476.
        assert_eq!(
            cast_float(0.1, 38, 38),
            Some(10_000_000_000_000_000_555_111_512_312_578_270_210)
        );
        assert_eq!(
            cast_float(0.620_973_404_611_029_5, 38, 34),
            Some(6_209_734_046_110_294_558_573_627_909_936_477)
        );
        // A negative scale multiplies in f64 by 1e-1, which is not a tenth:
        // the exact result is ...975.
        assert_eq!(
            cast_float(22_767_301_833_479_750.0, 30, -1),
            Some(2_276_730_183_347_976)
        );
        assert_eq!(cast_float(150.0, 10, -2), Some(2));
        assert_eq!(cast_float(1.5, 10, -2), Some(0));
        assert_eq!(cast_float(-250.0, 10, -2), Some(-2));
        assert_eq!(cast_float(1e12, 10, -2), None);
    }

    #[test]
    fn texts_cast_as_pyarrow_casts_them() {
        let cases: [(&str, u8, i8, Option<i128>); 26] = [
            ("1", 10, 0, Some(1)),
            ("1.5", 10, 0, None),
            ("1e2", 10, 0, Some(100)),
            ("1E+2", 10, 2, Some(10_000)),
            (" 7 ", 10, 0, None),
            ("abc", 10, 0, None),
            ("", 10, 0, None),
            ("1_0", 10, 0, None),
            ("nan", 10, 0, None),
            ("inf", 10, 0, None),
            ("0x10", 10, 0, None),
            ("\u{661}", 10, 0, None),
            ("12345678901", 10, 0, None),
            ("0000000000001", 10, 0, Some(1)),
            ("1.10", 10, 0, None),
            ("1.0e-1", 10, 0, None),
            ("-0", 10, 0, Some(0)),
            ("1e-400", 10, 0, None),
            ("1e400", 10, 0, None),
            ("99999999.995", 10, 2, None),
            ("1.", 10, 2, Some(100)),
            (".5", 10, 2, Some(50)),
            ("1.000", 10, 2, Some(100)),
            ("1.001", 10, 2, None),
            ("-1", 10, 2, Some(-100)),
            ("+1", 10, 2, Some(100)),
        ];
        for (s, p, scale, want) in cases {
            assert_eq!(
                cast_text(s, p, scale),
                want,
                "{s:?} as decimal128({p}, {scale})"
            );
        }
        assert_eq!(cast_text("1.0e-1", 10, 2), Some(10));
        assert_eq!(cast_text(".", 10, 2), None);
        assert_eq!(cast_text("1e", 10, 2), None);
        assert_eq!(cast_text("-", 10, 2), None);
    }
}
