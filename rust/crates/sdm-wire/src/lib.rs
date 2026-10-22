// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The wire of the model API: its JSON ([`json`]), read with every number
//! kept as the token written and every object in the order written, the
//! request state's payload as typed structures ([`payload`]), whose derived
//! schema is the contract's locked definition of it (T1.28w), and the
//! response to a prediction ([`response`]).
//!
//! As built at T1.11d the response is thin: it reads what a regression's
//! answer carries, as the client's bindings (`sdm_api.generated.tfm_api`,
//! generated from the pinned OpenAPI file) read it, and names the fields it
//! does not read yet. T1.28a generates it from the pinned OpenAPI file.

#![forbid(unsafe_code)]

pub mod json;
pub mod key;
pub mod payload;
pub mod response;
