// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `PROVIDERS`, the staging rule's marker (T2.1a), is exactly the dialects
//! this build renders: each listed one answers its capabilities, and the
//! corpus's other databases are refused as dialects it does not render.

use sdm_contract::data::{DataLayer, Dialect, ProviderChoice};
use sdm_data::{Data, PROVIDERS};

fn sql(name: &str) -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new(name),
    }
}

#[test]
fn every_listed_provider_renders_and_no_other_does() {
    assert!(PROVIDERS.contains(&"duckdb"));
    for name in ["duckdb", "sqlite", "postgres"] {
        let renders = Data::new().capabilities(&sql(name)).is_ok();
        assert_eq!(renders, PROVIDERS.contains(&name), "{name}");
    }
}
