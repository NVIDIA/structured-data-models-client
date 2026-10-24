// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The reads plan B adds to a run's statements (the owner's decision of
//! 2026-10-24, register P3A-1): each `materialize` over a source's tables
//! reads each table as it stands at that run, its catalog, keys and
//! preview, which the old client read once, at `from_source`, and kept.
//! Every comparison of a run's statements with the old client's takes
//! them out by this one rule and holds the rest to the old client's
//! exactly; `scripts/rust_core/compare.py`'s `reads_as_it_stands` is the
//! same rule for the Python tools.

use serde_json::Value;

/// How many of `statements`, a capture's in order, each with its `stage`
/// and `sql`, are the reads plan B adds: the leading statements of the
/// `materialize` stage that each repeat, word for word, a statement the
/// `from_source` stage ran.
pub fn reads_as_it_stands(statements: &[Value]) -> usize {
    let described: Vec<&Value> = statements
        .iter()
        .filter(|s| s["stage"] == "from_source")
        .map(|s| &s["sql"])
        .collect();
    statements
        .iter()
        .filter(|s| s["stage"] == "materialize")
        .take_while(|s| described.contains(&&s["sql"]))
        .count()
}

/// `statements` without the reads plan B adds ([`reads_as_it_stands`]):
/// the statements the old client's run is compared with.
pub fn as_the_old_client_ran(statements: &[Value]) -> Vec<Value> {
    let mut skipped = reads_as_it_stands(statements);
    statements
        .iter()
        .filter(|s| {
            if skipped > 0 && s["stage"] == "materialize" {
                skipped -= 1;
                false
            } else {
                true
            }
        })
        .cloned()
        .collect()
}

/// The request kinds of a run's `fetches`, each with its `request`, without
/// the reads plan B adds, seen as requests: the leading reads of a table's
/// catalog (its columns and keys, and the index columns a key names) and
/// its preview, each `materialize` making them before anything the old
/// client's run made.
pub fn requests_as_the_old_client_made(fetches: &[Value]) -> Vec<Value> {
    let read_as_it_stands = |fetch: &Value| {
        let request = &fetch["request"];
        match request["request"].as_str() {
            Some("describe") => matches!(
                request["part"].as_str(),
                Some("columns" | "keys" | "index_columns")
            ),
            Some("preview" | "column_ranges") => true,
            _ => false,
        }
    };
    fetches
        .iter()
        .skip_while(|fetch| read_as_it_stands(fetch))
        .cloned()
        .collect()
}
