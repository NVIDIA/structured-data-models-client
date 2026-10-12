// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What resolution needs of a profile: the mode, and the two filter
//! capabilities the model validator consults while it resolves. The other
//! capabilities are asked by `sdm-core` of the resolved query.

/// Whether a query defines a task or asks one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Training,
    Inference,
}

/// Whether a filter may compare a key column, or a time column, of the
/// tables a pretrained model reads (`key_columns_in_filters` and
/// `timestamp_columns_in_filters`). Both default to permitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Filters {
    pub key_columns: bool,
    pub timestamp_columns: bool,
}

impl Default for Filters {
    fn default() -> Self {
        Filters {
            key_columns: true,
            timestamp_columns: true,
        }
    }
}
