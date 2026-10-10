// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RS-1 as far as T1.11c reaches it, against numpy (`rs1.json`, recorded
//! from numpy's own `default_rng`): the seed sequence's state, the raw
//! 64-bit draws, and a run of shuffles on one generator, which carries the
//! 32-bit draw's kept half from one shuffle to the next, for seeds from 0
//! to past 2**200 (more entropy words than the pool holds).

use sdm_engine::random::{Entropy, SeedSequence, default_rng};
use serde_json::Value;

fn hex(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().unwrap().trim_start_matches("0x"), 16).unwrap()
}

#[test]
fn the_streams_are_numpy_s() {
    let document: Value = serde_json::from_str(include_str!("rs1.json")).unwrap();
    assert_eq!(document["format"], "sdm-rs1/1");
    let cases = document["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 11);
    for case in cases {
        let seed = case["seed"].as_str().unwrap().trim_start_matches("0x");
        let entropy = Entropy::from_hex(seed).unwrap();
        let state: Vec<u64> = case["state"].as_array().unwrap().iter().map(hex).collect();
        assert_eq!(
            SeedSequence::new(&entropy).generate_state_u64(4),
            state,
            "{seed}"
        );
        let mut rng = default_rng(&entropy);
        for raw in case["raw"].as_array().unwrap() {
            assert_eq!(rng.next_u64(), hex(raw), "{seed}");
        }
        for shuffled in case["shuffles"].as_array().unwrap() {
            let want: Vec<u64> = shuffled
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            let mut got: Vec<u64> = (0..want.len() as u64).collect();
            rng.shuffle(&mut got);
            assert_eq!(got, want, "{seed}");
        }
    }
}

#[test]
fn entropy_reads_any_size() {
    assert_eq!(Entropy::from_hex("0").unwrap().words(), &[0]);
    assert_eq!(Entropy::from_hex("2a").unwrap().words(), &[42]);
    // 2**70: three words.
    assert_eq!(
        Entropy::from_hex("400000000000000000").unwrap().words(),
        &[0, 0, 0x40]
    );
    assert_eq!(
        Entropy::from_u128(1 << 70),
        Entropy::from_hex("400000000000000000").unwrap()
    );
    assert!(Entropy::from_hex("").is_none());
    assert!(Entropy::from_hex("-1").is_none());
    assert!(Entropy::from_hex("0x2a").is_none());
}
