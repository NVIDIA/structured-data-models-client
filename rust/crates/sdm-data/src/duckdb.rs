// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DuckDB dialect. Its capabilities are the optional requests it has a
//! rendering for, so a request it cannot render is refused by name before
//! it is planned; none has one yet (T1.11b renders the catalog requests).

use std::collections::BTreeSet;

use sdm_contract::CoreError;
use sdm_contract::data::{
    DataRequest, Determinism, Dialect, FetchPlan, OptionalRequest, ProviderCapabilities,
};

/// The optional requests this dialect renders.
const RENDERED: &[OptionalRequest] = &[];

/// What a DuckDB source answers through this layer.
pub(crate) fn capabilities() -> ProviderCapabilities {
    ProviderCapabilities {
        optional: RENDERED.iter().copied().collect::<BTreeSet<_>>(),
        max_keys_per_request: None,
        // DuckDB's reservoir sampling repeats under a seed (REPEATABLE).
        sampling: Determinism::SeededStable,
    }
}

/// The statements answering `request` on DuckDB.
pub(crate) fn plan(request: &DataRequest) -> Result<FetchPlan, CoreError> {
    Err(crate::not_rendered(request, Dialect::Duckdb))
}
