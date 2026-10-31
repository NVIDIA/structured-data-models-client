// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_contract::session::{SessionKey, SessionKeyPart};
use sdm_request::tabular::session::{canonical_key, filter_holdout, overlap_count};

fn number(value: &str) -> SessionKey {
    vec![SessionKeyPart::Numeric {
        decimal: value.into(),
    }]
}

fn text(value: &str) -> SessionKey {
    vec![SessionKeyPart::Spelling {
        value: value.into(),
    }]
}

#[test]
fn ordinary_numeric_spelling_matches_frozen_reference_matrix() {
    let equivalent = [number("1"), number("1.0"), number("1.00"), text("1")];
    for left in &equivalent {
        for right in &equivalent {
            assert_eq!(canonical_key(left).unwrap(), canonical_key(right).unwrap());
        }
        assert_ne!(canonical_key(left).unwrap(), canonical_key(&text("1.00")).unwrap());
    }
}

#[test]
fn missing_components_never_match_or_remove_rows() {
    let missing = vec![SessionKeyPart::Missing];
    let mut composite = text("present");
    composite.push(SessionKeyPart::Missing);
    assert_eq!(canonical_key(&missing).unwrap(), None);
    assert_eq!(canonical_key(&composite).unwrap(), None);
    let context = vec![missing.clone(), number("1"), composite.clone()];
    let held = filter_holdout(&context, &[missing.clone(), missing, composite]).unwrap();
    assert_eq!(held.retained_rows, vec![0, 1, 2]);
    assert_eq!(held.removed_rows, 0);
    assert_eq!(held.named_keys, 3);
    assert_eq!(overlap_count(&context, &context).unwrap(), 1);
}

#[test]
fn holdout_preserves_source_order_and_overlap_counts_duplicate_rows() {
    let context = vec![number("2"), number("1"), number("1.0"), number("3")];
    let held = filter_holdout(&context, &[text("1"), number("1.00")]).unwrap();
    assert_eq!(held.retained_rows, vec![0, 3]);
    assert_eq!(held.removed_rows, 2);
    assert_eq!(held.named_keys, 1);
    assert_eq!(overlap_count(&[text("1")], &context).unwrap(), 2);
}

#[test]
fn composites_keep_boundaries_and_boolean_spelling() {
    let composite = vec![
        SessionKeyPart::Numeric { decimal: "1.0".into() },
        SessionKeyPart::Boolean { value: true },
    ];
    let same = vec![
        SessionKeyPart::Spelling { value: "1".into() },
        SessionKeyPart::Spelling { value: "True".into() },
    ];
    assert_eq!(canonical_key(&composite).unwrap(), canonical_key(&same).unwrap());
    assert_ne!(canonical_key(&composite).unwrap(), canonical_key(&text("1,True")).unwrap());
    assert_ne!(canonical_key(&vec![SessionKeyPart::Boolean { value: true }]).unwrap(), canonical_key(&number("1")).unwrap());
}

#[test]
fn exact_numeric_identity_does_not_round_high_precision_keys() {
    let first = number("1.12345678901234567890123456781");
    let second = number("1.12345678901234567890123456782");
    assert_ne!(canonical_key(&first).unwrap(), canonical_key(&second).unwrap());
    assert_eq!(filter_holdout(&[first], &[second]).unwrap().removed_rows, 0);
}

#[test]
fn decimal_format_boundaries_match_only_canonical_text() {
    for (numeric, spelling) in [
        ("-0.000", "0"),
        ("001.2000", "1.2"),
        ("1.00E+3", "1000"),
        ("1.20E-6", "0.0000012"),
        ("1.20E-7", "1.2E-7"),
        ("-1.20E-7", "-1.2E-7"),
        ("-100.0500", "-100.05"),
    ] {
        assert_eq!(canonical_key(&number(numeric)).unwrap(), canonical_key(&text(spelling)).unwrap());
    }
    for noncanonical in ["-0", "+1", "01", "1e0", "1E+0", "1.0"] {
        let normalized = if noncanonical == "-0" { "0" } else { "1" };
        assert_ne!(canonical_key(&text(noncanonical)).unwrap(), canonical_key(&number(normalized)).unwrap());
    }
}

#[test]
fn huge_exponents_remain_compact_and_never_expand_to_compare_short_text() {
    let first = number("1E+999999999999999999");
    let same = number("10E+999999999999999998");
    assert_eq!(canonical_key(&first).unwrap(), canonical_key(&same).unwrap());
    assert_ne!(canonical_key(&first).unwrap(), canonical_key(&text("1E+999999999999999999")).unwrap());
    assert_eq!(canonical_key(&number("1E-999999999999999999")).unwrap(), canonical_key(&text("1E-999999999999999999")).unwrap());
}

#[test]
fn nonfinite_numeric_spellings_and_malformed_facts_are_explicit() {
    assert_eq!(canonical_key(&number("Infinity")).unwrap(), canonical_key(&text("Infinity")).unwrap());
    assert_ne!(canonical_key(&number("Infinity")).unwrap(), canonical_key(&number("inf")).unwrap());
    for invalid in ["", "1e", "--1", "one", "NaN"] {
        assert!(canonical_key(&number(invalid)).is_err());
    }
}
