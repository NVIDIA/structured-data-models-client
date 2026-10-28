// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Observed casing semantics are portable data, without a library taxonomy.

use sdm_contract::frames::materialize::NativeTextFacts;
use serde_json::json;

#[test]
fn casing_facts_roundtrip_only_their_relevant_identity() {
    for (fact, wire) in [
        (
            NativeTextFacts::FullRoot {
                unicode_version: "15.0.0".into(),
            },
            json!({"mapping":"full_root","unicode_version":"15.0.0"}),
        ),
        (
            NativeTextFacts::Simple {
                identity: "iσⱟᲊ꟏".into(),
            },
            json!({"mapping":"simple","identity":"iσⱟᲊ꟏"}),
        ),
    ] {
        assert_eq!(serde_json::to_value(&fact).unwrap(), wire);
        assert_eq!(serde_json::from_value::<NativeTextFacts>(wire).unwrap(), fact);
    }
}

#[test]
fn casing_alternatives_reject_missing_mixed_or_unknown_facts() {
    for malformed in [
        json!({"mapping":"full_root"}),
        json!({"mapping":"simple"}),
        json!({"mapping":"full_root","unicode_version":"15.0.0","identity":"extra"}),
        json!({"mapping":"simple","identity":"iσⱟᲊ꟏","unicode_version":"15.0.0"}),
        json!({"mapping":"other","unicode_version":"15.0.0"}),
        json!({"mapping":"full_root","unicode_version":15}),
        json!({"mapping":"simple","identity":17}),
    ] {
        assert!(serde_json::from_value::<NativeTextFacts>(malformed).is_err());
    }
}
