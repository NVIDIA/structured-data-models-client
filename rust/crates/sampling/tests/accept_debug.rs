// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! P75.debug, inside the crate (task 6b): the two inputs the Phase 7 audit
//! found panicking only in a debug build, asked of `sdm_sampling::fixture_op`
//! in this test's own build, which `cargo test` makes a debug one.
//!
//! - A temporal neighborhood whose `colptr` starts at `i64::MIN` (the
//!   sampler's `end - count`): an answer, not a panic. Today's `relationallib`
//!   reads out of bounds here, so the answer itself is compared with the
//!   release build by `accept_driver_debug.py` instead.
//! - An offset of `n = 2**62` with weeks of `-2**63` (`RelativeDelta::fix`'s
//!   carry): pandas' `OverflowError`.
//!
//! And `sampling.date_offset_to_seconds` writes a result past int64 as the
//! exact integer today's fixture runner gives (REVIEW-r3 minor 3).

use serde_json::{Value, json};

fn ask(operation: &str, input: &Value) -> Value {
    let text = sdm_sampling::fixture_op(operation, &input.to_string())
        .unwrap_or_else(|e| panic!("{operation} refused its input: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{operation} gave {text}: {e}"))
}

fn ints(values: &[i64]) -> Value {
    json!({"$ndarray": {"dtype": "int64", "values": values}})
}

#[test]
fn a_neighborhood_starting_at_i64_min_is_answered() {
    let input = json!({
        "sampler": {
            "node_types": ["A", "B"],
            "edge_types": [["A", "to", "B"], ["B", "to", "A"]],
            "colptr": {"A__to__B": ints(&[0, 2, 4]), "B__to__A": ints(&[i64::MIN, 0, 1, 2])},
            "row": {"A__to__B": ints(&[0, 2, 0, 2]), "B__to__A": ints(&[0, 1, 1])},
            "node_time": {"A": ints(&[1, 0, 3]), "B": ints(&[2, 1])},
        },
        "calls": [{
            "num_neighbors": {"A__to__B": ints(&[1, 1]), "B__to__A": ints(&[1, 1])},
            "sampling_time_offsets": {},
            "entity_table": "B",
            "seed_node": ints(&[0, 1]),
            "seed_time": ints(&[1, 1]),
        }],
    });
    let got = ask("sampling.neighbor_sample", &input);
    let object = got.as_object().expect("an object");
    assert!(
        object.len() == 1 && (object.contains_key("output") || object.contains_key("error")),
        "gave {got}"
    );
}

#[test]
fn an_offset_carrying_past_i128_is_pandas_overflow() {
    let input = json!({
        "timestamps": [1_667_260_800_000_000_000_i64],
        "offset": {
            "n": 1_i64 << 62,
            "kwds": {"days": 86_400, "weeks": i64::MIN, "seconds": -12},
        },
        "sign": 1,
    });
    let got = ask("sampling.date_offset_apply", &input);
    let want = json!({"output": [{"error": {
        "code": "OverflowError",
        "message": "Python int too large to convert to C int",
    }}]});
    assert_eq!(got, want);
}

#[test]
fn seconds_past_int64_are_exact_integers() {
    let cases = [
        (
            json!({"years": 1_000_000_000_000_i64}),
            "earliest",
            "31536000000000000000",
        ),
        (
            json!({"years": 1_000_000_000_000_i64}),
            "latest",
            "31622400000000000000",
        ),
        (
            json!({"days": 1_i64 << 62}),
            "earliest",
            "398449671992126314905600",
        ),
        (
            json!({"months": -(1_i64 << 62)}),
            "latest",
            "-11156590815779536817356800",
        ),
    ];
    for (kwds, bound, want) in cases {
        let input = json!({"offset": {"n": 1, "kwds": kwds}, "bound": bound});
        let text = sdm_sampling::fixture_op("sampling.date_offset_to_seconds", &input.to_string())
            .unwrap_or_else(|e| panic!("refused {input}: {e}"));
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(compact, format!("{{\"output\":{want}}}"), "for {input}");
    }
}
