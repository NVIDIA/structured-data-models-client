// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Unicode words, pending coordinated contract35 linking.
//! Builtin conversion has already completed over the whole selected value.
//! Clipping, casing and word policy remain owned by the engine.

use sdm_contract::scalar::unicode::{UnicodeSlice, UnicodeText};

use super::{CasingProfile, MAX_CHARS, casing, normalize_with_profile, split};

/// Normalize an already converted raw native value. No cast failure is
/// hidden by clipping here, and no host-lowered or tokenized data is used.
pub fn normalize_native(
    value: Option<UnicodeSlice<'_>>,
    profile: CasingProfile,
) -> Vec<UnicodeText> {
    let Some(value) = value else {
        return Vec::new();
    };
    if let Some(ordinary) = value.utf8() {
        return normalize_with_profile(Some(ordinary), profile)
            .into_iter()
            .map(UnicodeText::from)
            .collect();
    }
    let clipped: Vec<_> = value.points().take(MAX_CHARS).collect();
    let lower = casing::lower_points(&clipped, char::from_u32, profile);
    split::ranges(&lower, char::from_u32, casing::space)
        .into_iter()
        .map(|range| {
            UnicodeText::from_points(lower[range].iter().copied())
                .expect("lowercase retains valid native codepoints")
        })
        .collect()
}

/// Unbounded native lowercasing for diagnostic text. This shares the actual
/// versioned casing kernel but has no feature clipping or token policy.
pub fn lower_native(value: UnicodeSlice<'_>, profile: CasingProfile) -> UnicodeText {
    let points = value.points().collect::<Vec<_>>();
    UnicodeText::from_points(casing::lower_points(&points, char::from_u32, profile))
        .expect("lowercasing preserves valid native Unicode points")
}
