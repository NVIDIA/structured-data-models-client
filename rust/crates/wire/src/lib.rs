// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Structured Data API's wire types, generated from its contract.
//!
//! `generated.rs` is written by `scripts/generate_wire_rs.py` from the
//! contract revision `PROVENANCE.json` names, the same one `sdm-api` is
//! generated from. Parsing a payload with [`parse`] and serializing it back
//! gives the same JSON: absent keys stay absent, integers stay integers, and
//! open objects keep the keys the contract does not name. An integer wider
//! than 64 bits, which could not stay one, is refused.
//!
//! The types also derive serde's `Deserialize`, but deserializing them
//! directly, with `serde_json::from_str` or `from_value`, skips that check
//! and turns a wider integer into a float. So every payload this crate reads
//! goes through [`parse`], directly or through a function built on it, like
//! [`parse_prediction_request`].

mod fields;
mod generated;
mod integers;

pub use generated::*;
pub use serde_json::{Map, Number, Value};

use serde::de::DeserializeOwned;

/// Parse a payload into one of the wire types.
///
/// An integer that does not fit in 64 bits is refused, because it would come
/// back as a float.
pub fn parse<T: DeserializeOwned>(json: &str) -> Result<T, serde_json::Error> {
    integers::check(json)?;
    serde_json::from_str(json)
}

/// Parse the body of `POST /v1/predictions`, with [`parse`].
pub fn parse_prediction_request(json: &str) -> Result<PredictionRequest, serde_json::Error> {
    parse(json)
}
