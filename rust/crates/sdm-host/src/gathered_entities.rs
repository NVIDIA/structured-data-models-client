// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed native value decoding for the shared gathered entity converter.

use sdm_contract::scalar::native_value::NativeValue;
use sdm_contract::scalar::unicode::UnicodeText;
use sdm_contract::{CoreError, Sentence};

pub fn gathered_entity_bytes(
    source: &[u8], observed: &[u8], dtype: &[u32],
) -> Result<Option<Vec<u32>>, CoreError> {
    let invalid = || CoreError::protocol(
        Sentence::InvalidMessage, "gathered entity conversion facts are malformed",
    );
    let source: NativeValue = serde_json::from_slice(source).map_err(|_| invalid())?;
    let observed: NativeValue = serde_json::from_slice(observed).map_err(|_| invalid())?;
    let dtype = UnicodeText::from_points(dtype.iter().copied()).map_err(|_| invalid())?;
    Ok(sdm_core::gathered_entity_text(&source, &observed, &dtype)?
        .map(|value| value.points().collect()))
}
