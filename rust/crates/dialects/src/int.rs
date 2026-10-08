// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python integers, which have no size limit: seeds and keys can exceed 64
//! bits. They are kept as their decimal digits, and only the arithmetic the
//! dialects do is offered.

use std::cmp::Ordering;
use std::fmt;

/// An integer of any size.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Int {
    negative: bool,
    /// Without leading zeros; `"0"` for zero, which is never negative.
    digits: String,
}

impl Int {
    /// `int(text)` for text of the form `-?[0-9]+`.
    pub fn parse(text: &str) -> Option<Self> {
        let (negative, body) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        Some(Self::from_parts(negative, body))
    }

    fn from_parts(negative: bool, digits: &str) -> Self {
        let trimmed = digits.trim_start_matches('0');
        let digits = if trimmed.is_empty() { "0" } else { trimmed };
        Self {
            negative: negative && digits != "0",
            digits: digits.to_string(),
        }
    }

    pub fn from_i128(value: i128) -> Self {
        Self {
            negative: value < 0,
            digits: value.unsigned_abs().to_string(),
        }
    }

    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// The value, if it fits an `i64`.
    pub fn to_i64(&self) -> Option<i64> {
        self.to_i128().and_then(|v| i64::try_from(v).ok())
    }

    /// The value, if it fits an `i128`.
    pub fn to_i128(&self) -> Option<i128> {
        let magnitude: u128 = self.digits.parse().ok()?;
        if self.negative {
            0i128.checked_sub_unsigned(magnitude)
        } else {
            i128::try_from(magnitude).ok()
        }
    }

    /// `self % modulus` as Python computes it for a positive modulus: never
    /// negative.
    pub fn rem_euclid(&self, modulus: u64) -> u64 {
        let modulus = u128::from(modulus);
        let rem = self
            .digits
            .bytes()
            .fold(0u128, |acc, b| (acc * 10 + u128::from(b - b'0')) % modulus);
        if self.negative && rem != 0 {
            (modulus - rem) as u64
        } else {
            rem as u64
        }
    }

    /// The magnitude as 32-bit words, least significant first, with no
    /// high zero words: none for zero.
    pub fn magnitude_words(&self) -> Vec<u32> {
        let mut digits: Vec<u32> = self.digits.bytes().map(|b| u32::from(b - b'0')).collect();
        let mut words = Vec::new();
        while !(digits.is_empty() || digits == [0]) {
            // One long division of the decimal digits by 2**32.
            let mut quotient = Vec::with_capacity(digits.len());
            let mut rem: u64 = 0;
            for digit in digits {
                let acc = rem * 10 + u64::from(digit);
                let q = acc >> 32;
                rem = acc & 0xffff_ffff;
                if !(quotient.is_empty() && q == 0) {
                    quotient.push(q as u32);
                }
            }
            words.push(rem as u32);
            digits = quotient;
        }
        words
    }
}

impl From<i64> for Int {
    fn from(value: i64) -> Self {
        Self::from_i128(i128::from(value))
    }
}

impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        let magnitude = self
            .digits
            .len()
            .cmp(&other.digits.len())
            .then_with(|| self.digits.cmp(&other.digits));
        match (self.negative, other.negative) {
            (false, false) => magnitude,
            (true, true) => magnitude.reverse(),
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
        }
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negative {
            f.write_str("-")?;
        }
        f.write_str(&self.digits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_and_remainders() {
        let big = Int::parse("18446744073709551617").unwrap();
        assert_eq!(big.magnitude_words(), [1, 0, 1]);
        assert_eq!(
            Int::parse("0").unwrap().magnitude_words(),
            Vec::<u32>::new()
        );
        assert_eq!(Int::parse("4294967296").unwrap().magnitude_words(), [0, 1]);
        assert_eq!(Int::parse("-3").unwrap().rem_euclid(1 << 31), (1 << 31) - 3);
        assert_eq!(big.rem_euclid(1 << 31), 1);
        assert_eq!(Int::parse("-0").unwrap().to_string(), "0");
        assert_eq!(big.to_i64(), None);
        assert_eq!(big.to_i128(), Some(18_446_744_073_709_551_617));
        assert!(Int::from(-5) < Int::from(3));
    }
}
