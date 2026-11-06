// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model request preparation and prediction-result interpretation.
//! This crate turns sampled data into typed payloads, validates inference options,
//! and interprets model responses for relational and single-table tasks.
//! It contains no SQL or connector-specific dispatch.
//!
//! [`payload`] lays out relational requests, [`tabular`] handles single-table data,
//! and [`predict`] prepares calls and reads their answers.
//! [`frame_session`] prepares in-memory session operations, [`gradients`] interprets
//! explanations, and [`serving`] handles model-serving envelopes.

#![forbid(unsafe_code)]

pub mod batching;
pub mod inference;
pub mod options;
pub mod payload;
pub mod predict;
pub mod ranking;
pub mod serving;

pub mod direct;
pub mod tabular;

pub mod frame_session;

pub mod result;

pub mod split;

pub mod items;

pub mod gradients;

pub mod columns;
