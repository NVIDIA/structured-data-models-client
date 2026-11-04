// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Refresh existing declarations through a replacement caller-owned source.
use crate::data::{AttachmentId, ProviderChoice};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RebindSource {
    pub provider: ProviderChoice,
    pub previous_provider: ProviderChoice,
    pub source_kind: String,
    pub previous_source_kind: String,
    pub schema: AttachmentId,
}
