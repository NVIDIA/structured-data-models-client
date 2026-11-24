// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A second data layer (plan section 3, "Data access"): the data
//! port (`sdm_contract::data::DataLayer`) implemented apart
//! from `sdm-sql`, composed at the composition root in place of
//! `sdm_host::default_root`'s ([`root`]). Dev-only.
//!
//! It reads no database: every answer is held to the shape the contract
//! fixes for it, and no semantic model is read (the trait's defaults). A
//! run takes the capabilities its provider declared, each request is
//! passed to the provider as itself, and a `describe` answer, in the
//! contract's shape (`catalog_columns`), is read by the core as every
//! source's is. It is built from the contract's public types alone (this
//! crate takes `sdm-contract` and `sdm-host`, no domain crate but the
//! contract), so the same prediction running on it shows the core needs
//! no change to swap its data layer.

use std::sync::Arc;

use sdm_contract::data::DataLayer;
use sdm_host::CompositionRoot;

/// The data layer of host providers alone.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostLayer;

/// The composition root over [`HostLayer`], in place of
/// `sdm_host::default_root`'s, over `sdm-sql`.
pub fn root() -> CompositionRoot {
    CompositionRoot::new(Arc::new(HostLayer))
}

impl DataLayer for HostLayer {}
