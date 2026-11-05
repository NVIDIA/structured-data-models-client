// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed model payloads, responses and lossless JSON representation.
//! [`json`] preserves numeric tokens and object ordering.
//! [`payload`] defines request bodies and their generated schema;
//! [`response`] validates prediction answers.

#![forbid(unsafe_code)]

pub mod json;
pub mod key;
pub mod payload;
pub mod response;
