// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A semantic model's definition, read into one dialect-free [`ViewSpec`]
//! (T3.5, design S5): [`metric`] reads a metric view's YAML, [`semantic`] a
//! semantic view's, both through [`yaml`], which gives YAML 1.1's values as
//! the old client's PyYAML does. Expressions stay as written, opaque text:
//! which columns one reads, whether it may run, and how it is rendered are
//! the data layer's to say, and the core asks each step in the old client's
//! order. [`joins`] walks a metric view's joins in that order, and
//! [`model`] holds the logical model the view converts into. Nothing here
//! reads a warehouse.

pub mod joins;
pub mod metric;
pub mod model;
pub mod semantic;
pub mod spec;
pub mod yaml;

pub use spec::ViewSpec;
