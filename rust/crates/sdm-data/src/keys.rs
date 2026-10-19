// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A key's text where an engine stores it as text: a decimal key on SQLite,
//! whose column is declared `TEXT` since one declared `DECIMAL` takes
//! numeric affinity and rounds what it stores (Phase 2 design, choice 2;
//! KO2-35). The contract loader (`contract/runners/databases.py`) stores a
//! decimal as this text and a key binder binds one as it, so equal values
//! of any scale meet: `contract/type-map-vectors.json` holds both to it
//! (`tests/type_map_vectors.rs` here).

/// A decimal's one text, `value` ten-to-the-`scale`ths: plain positional
/// notation, no exponent, no trailing fractional zeros, no trailing point,
/// `-0` written `0`.
pub fn canonical_decimal_text(value: i128, scale: i8) -> String {
    let (mut value, mut scale) = (value, scale);
    if value == 0 {
        return "0".to_owned();
    }
    while scale > 0 && value % 10 == 0 {
        value /= 10;
        scale -= 1;
    }
    let sign = if value < 0 { "-" } else { "" };
    let digits = value.unsigned_abs().to_string();
    if scale <= 0 {
        let zeros = "0".repeat(usize::from(scale.unsigned_abs()));
        return format!("{sign}{digits}{zeros}");
    }
    let scale = usize::from(scale.unsigned_abs());
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{fraction}")
}

#[cfg(test)]
mod tests {
    use super::canonical_decimal_text;

    #[test]
    fn the_extremes_and_a_negative_scale() {
        assert_eq!(canonical_decimal_text(i128::MIN, 0), i128::MIN.to_string());
        assert_eq!(
            canonical_decimal_text(i128::MIN, 38),
            "-1.70141183460469231731687303715884105728"
        );
        assert_eq!(canonical_decimal_text(-5, 1), "-0.5");
        assert_eq!(canonical_decimal_text(12, -2), "1200");
        assert_eq!(canonical_decimal_text(0, -2), "0");
    }
}
