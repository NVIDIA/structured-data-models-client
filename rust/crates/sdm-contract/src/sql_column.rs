// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A named provider-dialect projection added to an existing source schema.
use crate::data::{AttachmentId, ProviderChoice};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SqlColumn {
    pub provider: ProviderChoice,
    pub schema: AttachmentId,
    pub input: AttachmentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SqlColumnCall {
    pub table: String,
    pub name: String,
    pub expression: String,
}

/// Caller-owned SQL projection, reapplied after refreshing the physical source.
#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct CalculatedColumn {
    pub name: String,
    pub expression: String,
}
