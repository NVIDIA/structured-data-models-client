// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The payload limit counts the bytes the request sends (register PD-2,
//! ): the builder's UTF-8 bytes, as written, so a payload of exactly
//! the limit is taken and one byte more refused, where the reference
//! counted `json.dumps`' text, which writes each character past ASCII as a
//! six-byte escape. Held at the limit with ASCII text and with text of two-,
//! three- and four-byte characters, which the reference's count would
//! have taken for three to twelve times its size.
//!
//! The payloads here score one row, so a payload past the limit cannot be
//! split into smaller calls, and the context alone is past it.

use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, StringArray};
use sdm_contract::limits::MAX_REQUEST_BYTES;
use sdm_contract::ops::Stype;
use sdm_contract::{ErrorCode, Sentence};
use sdm_engine::labels::Labels;
use sdm_request::options::{Kind, Options, RunMode};
use sdm_request::payload::{Built, Sampled, SampledColumn, SampledTable, build, check_bytes};

const PAYLOAD_BYTES: usize = MAX_REQUEST_BYTES.default as usize;

/// The payload built and held to the default limit.
fn checked(sampled: &Sampled) -> Result<Built, sdm_contract::CoreError> {
    let built = build(sampled)?;
    check_bytes(&built, MAX_REQUEST_BYTES.default)?;
    Ok(built)
}

const ANCHOR: i64 = 1_709_251_200_000_000_000;

/// One entity, one context example and one to predict, each carrying
/// `note` as a categorical text feature.
fn sampled(note: &str) -> Sampled {
    let column = |a: ArrayRef| SampledColumn::Arrow(a);
    Sampled {
        num_train: 1,
        num_test: 1,
        anchors: vec![ANCHOR, ANCHOR],
        kind: Kind::Regression,
        labels: Labels::Float(vec![1.0]),
        windowed: true,
        tables: vec![SampledTable {
            name: "users".into(),
            primary_key: Some("user_id".into()),
            batch: vec![0, 1],
            columns: vec![
                (
                    "user_id".into(),
                    column(Arc::new(Int64Array::from(vec![0, 1]))),
                ),
                (
                    "note".into(),
                    column(Arc::new(StringArray::from(vec![note, note]))),
                ),
            ],
            stypes: vec![("note".into(), Stype::Categorical)],
        }],
        links: vec![],
        entity_table: "users".into(),
        source_query: String::new(),
        run_mode: RunMode::Fast,
        use_prediction_time: false,
        return_embeddings: false,
        options: Options::default(),
        statistic: None,
        top_k: None,
        lags: vec![],
        float_text: |x| format!("{x:?}"),
    }
}

/// The payload's length with `n` copies of `unit` in each note.
fn length(unit: &str, n: usize) -> Result<usize, ErrorCode> {
    checked(&sampled(&unit.repeat(n)))
        .map(|built| built.bytes.len())
        .map_err(|e| e.code)
}

#[test]
fn the_limit_counts_the_bytes_sent_at_the_limit() {
    for unit in ["a", "\u{e9}", "\u{20ac}", "\u{1f600}"] {
        let width = unit.len();
        // Each note is written twice (the context's row and the predicted
        // row), so the payload grows by twice a unit's bytes per unit.
        let base = length(unit, 0).unwrap();
        let step = length(unit, 1).unwrap() - base;
        assert_eq!(step, 2 * width, "{unit:?}");
        // The most units whose payload is within the limit, and the slack
        // left, which ASCII padding of the bytes that remain fills.
        let n = (PAYLOAD_BYTES - base) / step;
        let slack = PAYLOAD_BYTES - base - n * step;
        let at = format!("{}{}", unit.repeat(n), "a".repeat(slack / 2));
        let taken = checked(&sampled(&at)).expect("a payload within the limit is taken");
        let exact = base + n * step + 2 * (slack / 2);
        assert_eq!(taken.bytes.len(), exact, "{unit:?}");
        assert!(exact <= PAYLOAD_BYTES && PAYLOAD_BYTES - exact <= 1);
        // The bytes counted are the bytes written: UTF-8, each character
        // its own bytes, not an escape.
        let text = String::from_utf8(taken.bytes.clone()).unwrap();
        assert!(!text.contains("\\u"), "{unit:?}");
        // Two units more (four bytes or more past it) are refused, as the
        // reference refuses a payload past its limit.
        let past = format!("{}{}", unit.repeat(n + 2), "a".repeat(slack / 2));
        let refused = checked(&sampled(&past)).expect_err("a payload past the limit");
        assert_eq!(refused.code, ErrorCode::InvalidRequest, "{unit:?}");
        assert_eq!(refused.sentence, Sentence::PayloadNoSmallerCall, "{unit:?}");
    }
    // With ASCII the payload reaches the limit exactly, one byte step past
    // it refused: the count is of bytes, to the byte.
    let base = length("a", 0).unwrap();
    let fill = |extra: usize| {
        let half = (PAYLOAD_BYTES - base + extra) / 2;
        checked(&sampled(&"a".repeat(half)))
    };
    let even = (PAYLOAD_BYTES - base).is_multiple_of(2);
    let at = fill(0).expect("at the limit");
    assert_eq!(
        at.bytes.len(),
        if even {
            PAYLOAD_BYTES
        } else {
            PAYLOAD_BYTES - 1
        }
    );
    assert_eq!(fill(2).unwrap_err().code, ErrorCode::InvalidRequest);
}

#[test]
fn a_raised_limit_takes_a_payload_past_the_default_and_names_its_setting_past_it() {
    let base = length("a", 0).unwrap();
    let past = sampled(&"a".repeat((PAYLOAD_BYTES - base) / 2 + 8));
    let built = build(&past).unwrap();
    let refused = check_bytes(&built, MAX_REQUEST_BYTES.default).unwrap_err();
    assert_eq!(
        refused.fields["setting"],
        sdm_contract::Detail::text("max_request_bytes")
    );
    assert_eq!(
        refused.fields["env"],
        sdm_contract::Detail::text("SDM_CLIENT_MAX_REQUEST_BYTES")
    );
    let context = match &refused.fields["context_bytes"] {
        sdm_contract::Detail::Scalar(sdm_contract::Scalar::UInt64(n)) => *n,
        other => panic!("{other:?}"),
    };
    assert!(context < built.model_bytes as u64);
    check_bytes(&built, 2 * MAX_REQUEST_BYTES.default).expect("a raised limit takes it");
}
