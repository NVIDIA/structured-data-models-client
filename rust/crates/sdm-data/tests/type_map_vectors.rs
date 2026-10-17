// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The key binders' canonical decimal text against
//! `contract/type-map-vectors.json`, the vector the Python loader runs too
//! (`tests/test_t21a_type_map.py`; T2.1a): each literal at its column's
//! scale gives the vector's text, equal values of different scales give
//! one text, and the SQLite pair and the join keys give theirs.

use std::path::PathBuf;

use sdm_data::canonical_decimal_text;
use serde_json::Value;

fn vectors() -> Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../contract/type-map-vectors.json");
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

/// `literal`, a decimal's text, as its unscaled value at `scale`, as Arrow
/// holds a value of a column of that scale.
fn unscaled(literal: &str, scale: i8) -> i128 {
    let (sign, digits) = match literal.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, literal),
    };
    let (whole, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    let scale = usize::try_from(scale).unwrap();
    assert!(fraction.len() <= scale, "{literal} past its scale");
    let text = format!("{whole}{fraction:0<scale$}");
    sign * text.parse::<i128>().unwrap()
}

/// `DECIMAL(p,s)`'s scale.
fn scale_of(duckdb_type: &str) -> i8 {
    duckdb_type
        .trim_end_matches(')')
        .rsplit(',')
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

#[test]
fn each_decimal_has_the_vector_s_canonical_text() {
    let document = vectors();
    assert_eq!(document["format"], "sdm-type-map-vectors/1");
    let texts = document["canonical_text"].as_array().unwrap();
    assert!(texts.len() >= 15);
    for vector in texts {
        let scale = scale_of(vector["duckdb_type"].as_str().unwrap());
        let literal = vector["literal"].as_str().unwrap();
        assert_eq!(
            canonical_decimal_text(unscaled(literal, scale), scale),
            vector["canonical"].as_str().unwrap(),
            "{literal}"
        );
    }
}

#[test]
fn equal_values_of_any_scale_have_one_text() {
    let document = vectors();
    for group in document["one_text"].as_array().unwrap() {
        for literal in group["literals"].as_array().unwrap() {
            let literal = literal.as_str().unwrap();
            for scale in 4..=10 {
                assert_eq!(
                    canonical_decimal_text(unscaled(literal, scale), scale),
                    group["canonical"].as_str().unwrap(),
                    "{literal} at scale {scale}"
                );
            }
        }
    }
    let pair = &document["sqlite_distinct_keys"];
    let join = &document["join_across_scales"];
    for (keys, duckdb_type, canonical) in [
        (&pair["keys"], &pair["duckdb_type"], &pair["canonical"]),
        (
            &join["left"]["keys"],
            &join["left"]["duckdb_type"],
            &join["left"]["canonical"],
        ),
        (
            &join["right"]["keys"],
            &join["right"]["duckdb_type"],
            &join["right"]["canonical"],
        ),
    ] {
        let scale = scale_of(duckdb_type.as_str().unwrap());
        let texts: Vec<String> = keys
            .as_array()
            .unwrap()
            .iter()
            .map(|key| canonical_decimal_text(unscaled(key.as_str().unwrap(), scale), scale))
            .collect();
        assert_eq!(serde_json::json!(texts), *canonical);
    }
}
