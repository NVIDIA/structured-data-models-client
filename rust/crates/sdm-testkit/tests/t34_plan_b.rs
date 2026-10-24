// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.4, round 16: the reads plan B adds to a core run are taken out of a
//! comparison with the old client by the mark the core puts on them, the
//! fetch's purpose (`locate`), and by nothing else.

use sdm_testkit::plan_b::as_the_old_client_ran;
use serde_json::{Value, json};

fn statement(stage: &str, sql: &str, parameters: Value, purpose: Option<&str>) -> Value {
    let mut statement = json!({
        "handle": 1, "stage": stage, "sql": sql, "parameters": parameters,
        "results": [], "error": null,
    });
    if let Some(purpose) = purpose {
        statement["purpose"] = json!(purpose);
    }
    statement
}

/// Codex's round 16 probe: a statement repeating one `from_source` ran,
/// bound to another schema and not marked, is compared.
#[test]
fn a_repeated_statement_bound_to_another_table_is_kept() {
    let catalog = "SELECT table_name FROM information_schema.tables WHERE table_schema = ?";
    let statements = [
        statement("from_source", catalog, json!(["public"]), None),
        statement("materialize", catalog, json!(["wrong_schema"]), None),
        statement("materialize", "SAMPLE X", Value::Null, None),
    ];
    let kept = as_the_old_client_ran(&statements);
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[1]["parameters"], json!(["wrong_schema"]));
}

/// Claude's round 16 probe: a statement repeating one `from_source` ran is
/// kept unless the core marked it, so a run that reads something else in
/// its place is not taken for the old client's.
#[test]
fn an_unmarked_repeat_of_a_description_s_statement_is_kept() {
    let statements = [
        statement(
            "from_source",
            "PRAGMA index_list(\"users\")",
            Value::Null,
            None,
        ),
        statement(
            "materialize",
            "PRAGMA index_list(\"users\")",
            Value::Null,
            None,
        ),
        statement("materialize", "SAMPLE X", Value::Null, None),
    ];
    assert_eq!(as_the_old_client_ran(&statements).len(), 3);
}

/// The reads the core marks as plan B's are taken out, whatever their text.
#[test]
fn the_reads_the_core_marks_are_taken_out() {
    let statements = [
        statement(
            "from_source",
            "PRAGMA table_info(\"users\")",
            Value::Null,
            None,
        ),
        statement(
            "materialize",
            "PRAGMA table_info(\"users\")",
            Value::Null,
            Some("locate"),
        ),
        statement(
            "materialize",
            "SELECT \"user_id\" FROM \"users\" LIMIT 1000",
            Value::Null,
            Some("locate"),
        ),
        statement("materialize", "SAMPLE X", Value::Null, None),
    ];
    let kept = as_the_old_client_ran(&statements);
    assert_eq!(
        kept.iter()
            .map(|s| s["sql"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["PRAGMA table_info(\"users\")", "SAMPLE X"]
    );
}
