// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact arbitrary-precision native Decimal comparison facts.

use sdm_contract::frames::{FrameDecimalExponent, FrameScalar};
use sdm_contract::{CoreError, Sentence};

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// A native Decimal equality key, normalized without arithmetic/context
/// rounding. This supports heterogeneous exponents and arbitrary precision;
/// it is not a new wire scalar or a replacement for native presentation.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Key {
    Finite {
        negative: bool,
        digits: String,
        exponent: i128,
    },
    Infinity {
        negative: bool,
    },
    SignalingNan {
        negative: bool,
        digits: String,
    },
    Native {
        row: usize,
    },
    Text(sdm_contract::scalar::unicode::UnicodeText),
    Float(u64),
}

impl Key {
    pub(super) fn fits_int64(&self) -> bool {
        let Self::Finite {
            negative,
            digits,
            exponent,
        } = self
        else {
            return false;
        };
        if digits == "0" {
            return true;
        }
        if *exponent < 0 {
            return false;
        }
        let length = digits.len() as i128 + exponent;
        if length != 19 {
            return length < 19;
        }
        let limit = if *negative {
            b"9223372036854775808"
        } else {
            b"9223372036854775807"
        };
        digits
            .bytes()
            .chain(std::iter::repeat_n(b'0', *exponent as usize))
            .cmp(limit.iter().copied())
            .is_le()
    }

    pub(super) fn fractional(&self) -> bool {
        matches!(
            self,
            Self::Infinity { .. }
                | Self::SignalingNan { .. }
                | Self::Finite {
                    exponent: ..=-1,
                    ..
                }
        )
    }
}

/// Read the already-existing Decimal scalar vocabulary as a comparison key.
/// Quiet NaNs carry native missingness, while a signaling NaN cannot be
/// manufactured into a quiet value by source admission.
pub(crate) fn key(value: FrameScalar) -> Result<Option<Key>, CoreError> {
    let FrameScalar::Decimal {
        mut negative,
        digits,
        exponent,
    } = value
    else {
        return Err(invalid(
            "an object Decimal cell requires a native Decimal tuple",
        ));
    };
    match exponent {
        FrameDecimalExponent::Nan => Ok(None),
        FrameDecimalExponent::SignalingNan => Ok(Some(Key::SignalingNan { negative, digits })),
        FrameDecimalExponent::Infinity => Ok(Some(Key::Infinity { negative })),
        FrameDecimalExponent::Finite { value } => {
            if digits.is_empty() || !digits.bytes().all(|digit| digit.is_ascii_digit()) {
                return Err(invalid("a finite native Decimal requires decimal digits"));
            }
            let leading = digits.trim_start_matches('0');
            let mut exponent = i128::from(value);
            let normalized = if leading.is_empty() {
                negative = false;
                exponent = 0;
                "0".to_owned()
            } else {
                let trimmed = leading.trim_end_matches('0');
                exponent += (leading.len() - trimmed.len()) as i128;
                trimmed.to_owned()
            };
            Ok(Some(Key::Finite {
                negative,
                digits: normalized,
                exponent,
            }))
        }
    }
}
