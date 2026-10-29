// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test-only provider selection for complete independently rendered witnesses.

use sdm_contract::data::{FetchPlan, ProviderChoice, SqlParam};

/// Select only dialects whose complete Arrow-key plans can be re-rendered.
/// The caller must still prove full plan equality through its composition
/// root; this selection alone never authorizes rewriting any SQL.
pub fn fetch_witness_provider(plan: &FetchPlan) -> Option<ProviderChoice> {
    let FetchPlan::Sql { statements, .. } = plan else {
        return None;
    };
    let first = statements.first()?;
    let dialect = first.dialect.name();
    if (dialect != crate::sqlite::NAME && dialect != crate::duckdb::NAME)
        || statements
            .iter()
            .any(|statement| statement.dialect != first.dialect)
        || statements
            .iter()
            .flat_map(|statement| &statement.params)
            .any(|param| matches!(param, SqlParam::Text { .. }))
    {
        return None;
    }
    Some(ProviderChoice::Sql {
        dialect: first.dialect.clone(),
        namespace: vec![],
        view: None,
        timestamp_encoding: None,
    })
}
