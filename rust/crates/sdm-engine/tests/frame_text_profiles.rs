// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kernel expectations taken from full frozen public frame model bodies.
//! The public host and HostRun stages will use these same immutable receipts.

use sdm_engine::words::{CasingProfile, UnicodeDataVersion, normalize_with_profile};
use serde_json::{Value, json};

fn column<'a>(table: &'a Value, name: &str) -> usize {
    table["columns"]
        .as_array()
        .unwrap()
        .iter()
        .position(|column| column == name)
        .unwrap()
}

fn user_input(id: i64) -> String {
    match id {
        1 => "ΟΣ ΟΣΑ AΣ\u{301} AΣ1 AΣ-A".into(),
        7 => "ΟΣ İ AΣ\u{301} ΟΣΑ".into(),
        8 => format!("{}İΣx", "é".repeat(298)),
        other => panic!("unexpected frozen selected user {other}"),
    }
}

#[test]
fn effective_native_profiles_reproduce_the_frozen_public_text_columns() {
    let arrow: Value = serde_json::from_str(include_str!(
        "../../../../packages/structured-data-models-client/tests/t44_native_text_boundaries_pandas3_reference.json"
    )).unwrap();
    // This is the installed library's own metadata, not its package version.
    assert_eq!(arrow["utf8proc_metadata"][0]["unicode_version"], "17.0.0");
    for recorded in [
        include_str!(
            "../../../../packages/structured-data-models-client/tests/t44_native_text_profile_pandas2_reference.json"
        ),
        include_str!(
            "../../../../packages/structured-data-models-client/tests/t44_native_text_profile_pandas3_reference.json"
        ),
    ] {
        let observed: Value = serde_json::from_str(recorded).unwrap();
        assert_eq!(observed["versions"]["unicode"], "15.0.0");
        for (name, record) in observed["records"].as_object().unwrap() {
            let casts: Vec<_> = record["trace"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|event| event["stage"] == "actual-astype-str")
                .collect();
            let bodies = record["sent"].as_array().unwrap();
            assert_eq!(casts.len(), 2 * bodies.len(), "{name}");
            for (body, cast) in bodies.iter().zip(casts.chunks_exact(2)) {
                assert_eq!(cast[0]["result_array"], cast[1]["result_array"], "{name}");
                let profile = match cast[0]["result_array"].as_str().unwrap() {
                    "ArrowStringArray" => CasingProfile::Simple(UnicodeDataVersion::V17_0),
                    "StringArray" | "NumpyExtensionArray" => {
                        CasingProfile::FullRoot(UnicodeDataVersion::V15_0)
                    }
                    other => panic!("unproved native target array {other}"),
                };
                for side in ["context", "predict"] {
                    let table = &body[side]["related_tables"]["users"];
                    let id = column(table, "user_id");
                    let note = column(table, "note");
                    for row in table["rows"].as_array().unwrap() {
                        let input = user_input(row[id].as_i64().unwrap());
                        assert_eq!(
                            json!(normalize_with_profile(Some(&input), profile)),
                            row[note],
                            "{name} {side}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn full_lowercase_keeps_context_and_expansion_after_the_character_cut() {
    let profile = CasingProfile::FullRoot(UnicodeDataVersion::V15_0);
    assert_eq!(
        normalize_with_profile(Some("ΟΣ ΟΣΑ AΣ\u{301}"), profile),
        ["ος", "οσα", "aς\u{301}"]
    );
    assert_eq!(
        normalize_with_profile(Some(&format!("{}İΣx", "é".repeat(298))), profile),
        [format!("{}i\u{307}ς", "é".repeat(298))]
    );
    assert_eq!(normalize_with_profile(None, profile), Vec::<String>::new());
    assert_eq!(
        normalize_with_profile(Some(""), profile),
        Vec::<String>::new()
    );
}

#[test]
fn the_simple_profile_and_existing_provider_default_are_kept_distinct_from_full() {
    let simple = CasingProfile::Simple(UnicodeDataVersion::V17_0);
    assert_eq!(normalize_with_profile(Some("ΟΣ İ"), simple), ["οσ", "i"]);
    assert_eq!(
        normalize_with_profile(Some("ΟΣ İ"), simple),
        sdm_engine::words::normalize(Some("ΟΣ İ"))
    );
    assert_ne!(
        normalize_with_profile(Some("ΟΣ İ"), simple),
        normalize_with_profile(
            Some("ΟΣ İ"),
            CasingProfile::FullRoot(UnicodeDataVersion::V15_0)
        )
    );
}
