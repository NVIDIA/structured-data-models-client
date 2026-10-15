// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract's locked definition of the request state's payload is the
//! schema the typed payload derives (T1.28w, decision E74): the copy
//! `sdm-contract` embeds, `schema/state-payload.json`, is the one these
//! types generate, so a change to them fails here until it is regenerated,
//! and then fails the contract lock until the version moves with it.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads and, with SDM_CONTRACT_WRITE, rewrites the committed payload schema"
)]

use std::path::PathBuf;

use sdm_wire::payload::schema_json;

fn committed() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sdm-contract/schema/state-payload.json")
}

#[test]
fn the_committed_payload_schema_is_the_one_the_types_derive() {
    let generated = schema_json();
    if std::env::var_os("SDM_CONTRACT_WRITE").is_some() {
        std::fs::write(committed(), &generated).unwrap();
    }
    let committed = std::fs::read_to_string(committed()).unwrap_or_default();
    assert!(
        committed == generated,
        "sdm-contract/schema/state-payload.json differs from the schema the payload's types \
         derive; regenerate it with SDM_CONTRACT_WRITE=1 cargo test -p sdm-wire --test \
         state_payload_schema, then the contract's schema and lock"
    );
}
