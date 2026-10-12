// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integers as Python reads them: any size, from a query's digits, refused
//! past Python's limit of 4,300 digits. This is one of the two emulation
//! modules this crate may hold (`scripts/rust_core/emulation-allowlist.txt`,
//! Phase 1 decisions A4), because the limit decides whether a query with a
//! longer integer is valid.
//!
//! PQL only compares integers with small bounds and prints them, so they are
//! kept as their decimal digits, exactly.

use std::cmp::Ordering;
use std::fmt;

/// Python refuses to read more digits than this into an `int`
/// (`sys.get_int_max_str_digits()`), counting leading zeros, and so does the
/// reference's parser.
const MAX_STR_DIGITS: usize = 4300;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyInt {
    negative: bool,
    /// Without leading zeros; `"0"` for zero, which is never negative.
    digits: String,
}

impl PyInt {
    /// `int(text)` for text of the grammar's `INT` shape, `-?[0-9]+`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let (negative, body) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        if body.is_empty() || !body.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!("'{text}' is not an integer"));
        }
        if body.len() > MAX_STR_DIGITS {
            return Err(format!(
                "{} digits is more than the {MAX_STR_DIGITS} an integer may have",
                body.len()
            ));
        }
        let trimmed = body.trim_start_matches('0');
        let digits = if trimmed.is_empty() { "0" } else { trimmed };
        Ok(Self {
            negative: negative && digits != "0",
            digits: digits.to_string(),
        })
    }

    pub fn from_i64(value: i64) -> Self {
        Self {
            negative: value < 0,
            digits: value.unsigned_abs().to_string(),
        }
    }

    /// The value, if it fits an `i64`.
    pub fn to_i64(&self) -> Option<i64> {
        let text = if self.negative {
            format!("-{}", self.digits)
        } else {
            self.digits.clone()
        };
        text.parse().ok()
    }

    pub fn cmp_i64(&self, other: i64) -> Ordering {
        self.cmp(&Self::from_i64(other))
    }

    pub fn lt(&self, other: i64) -> bool {
        self.cmp_i64(other) == Ordering::Less
    }

    pub fn gt(&self, other: i64) -> bool {
        self.cmp_i64(other) == Ordering::Greater
    }
}

impl Ord for PyInt {
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

impl PartialOrd for PyInt {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for PyInt {
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
    fn reads_and_compares_as_python() {
        let big = PyInt::parse("99999999999999999999999").unwrap();
        assert!(big.gt(i64::MAX));
        assert_eq!(big.to_string(), "99999999999999999999999");
        assert_eq!(PyInt::parse("-0").unwrap().to_string(), "0");
        assert_eq!(PyInt::parse("007").unwrap().to_i64(), Some(7));
        assert!(PyInt::parse("-9223372036854775809").unwrap().lt(i64::MIN));
        assert!(PyInt::parse(&"0".repeat(4301)).is_err());
        assert!(PyInt::parse(&format!("-{}", "1".repeat(4300))).is_ok());
    }
}
