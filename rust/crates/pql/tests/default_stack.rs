// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `validate` recurses as deep as a query nests. Called on a thread with a
//! default stack (a spawned Rust thread's 2 MiB, or less), the deepest
//! queries the parser takes are answered, not a stack overflow that aborts
//! the process.

use std::path::Path;

use sdm_pql::parser::MAX_DEPTH;
use sdm_pql::{Profile, SchemaDefinition, validate};

fn shop() -> SchemaDefinition {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tests/parity/pql/valid.json");
    let corpus: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("the corpus")).expect("JSON");
    SchemaDefinition::from_value(&corpus["cases"][0]["schema"]).expect("a schema")
}

/// The deepest queries the parser takes, and one level past them.
fn deepest() -> Vec<String> {
    let mut out = Vec::new();
    for depth in [MAX_DEPTH - 2, MAX_DEPTH + 1] {
        out.push(format!(
            "PREDICT {}COUNT(orders.*, 0, 30, days) > 1 FOR EACH users.user_id",
            "NOT ".repeat(depth)
        ));
        out.push(format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {}users.age > \
             3{}",
            "(".repeat(depth),
            ")".repeat(depth)
        ));
    }
    out
}

fn on_a_stack_of(bytes: usize) {
    let schema = shop();
    let profile = Profile::by_name("RELATIONAL").expect("a profile");
    std::thread::Builder::new()
        .stack_size(bytes)
        .spawn(move || {
            for query in deepest() {
                let validation = validate(&query, &schema, &profile);
                let _ = validation.response.ok();
            }
        })
        .expect("a thread")
        .join()
        .expect("answered");
}

#[test]
fn the_deepest_queries_validate_on_a_spawned_threads_default_stack() {
    on_a_stack_of(2 << 20);
}

/// A thread pthreads makes on macOS without saying how big: 512 KiB.
#[test]
fn the_deepest_queries_validate_on_a_secondary_pthreads_default_stack() {
    on_a_stack_of(512 << 10);
}
