// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_engine::words` against the reference's sampler, from `words.json`,
//! recorded by `words_record.py` in the reference's environment (a
//! protected path: never recorded again to pass a dependency bump, F4):
//! the lower-casing of every code point the pipeline changes, and the words
//! `_normalize_text` gives for punctuation, every whitespace code point,
//! line breaks, long texts and random mixtures of them.
//!
//! Every code point lowers to the reference's, and every value gives the
//! reference's words. [`check`] fails closed: a perturbed answer, and
//! evidence missing, empty or retyped, each fail (seeded below).

use sdm_engine::words::{lower_case, normalize};
use serde_json::{Value, json};

const VECTORS: &str = include_str!("words.json");

/// The code points whose lowering agreed, and the values whose words did.
#[derive(Debug, PartialEq, Eq)]
struct Counts {
    lowered: usize,
    values: usize,
}

fn text(value: &Value, at: &str) -> Result<Option<String>, String> {
    match value {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        _ => Err(format!("{at}: a value is neither text nor null")),
    }
}

fn check(vectors: &Value) -> Result<Counts, String> {
    let lower = vectors["lower"]
        .as_object()
        .filter(|l| !l.is_empty())
        .ok_or("no lower-casing recorded")?;
    let mut changed = std::collections::BTreeMap::new();
    for (point, wanted) in lower {
        let c = point
            .parse::<u32>()
            .ok()
            .and_then(char::from_u32)
            .ok_or_else(|| format!("{point}: not a code point"))?;
        let wanted: Vec<char> = wanted
            .as_array()
            .filter(|w| !w.is_empty())
            .ok_or_else(|| format!("{point}: no lowering"))?
            .iter()
            .map(|p| {
                p.as_u64()
                    .and_then(|p| u32::try_from(p).ok())
                    .and_then(char::from_u32)
            })
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{point}: a lowering is not code points"))?;
        changed.insert(c, wanted);
    }
    // Every code point: lowered as recorded, or, where nothing was
    // recorded, left as it is.
    for c in (0..=0x10_FFFF_u32).filter_map(char::from_u32) {
        let found: Vec<char> = lower_case(c).collect();
        let wanted = changed.get(&c).cloned().unwrap_or_else(|| vec![c]);
        if found != wanted {
            return Err(format!(
                "U+{:04X}: lowered otherwise than the reference",
                c as u32
            ));
        }
    }
    let values = vectors["values"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("no values recorded")?;
    for (at, pair) in values.iter().enumerate() {
        let at = format!("value {at}");
        let [value, words] = pair
            .as_array()
            .map(Vec::as_slice)
            .ok_or_else(|| format!("{at}: not a pair"))?
        else {
            return Err(format!("{at}: not a pair"));
        };
        let value = text(value, &at)?;
        let words: Vec<String> = words
            .as_array()
            .ok_or_else(|| format!("{at}: the words are not a list"))?
            .iter()
            .map(|w| w.as_str().map(str::to_owned))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("{at}: a word is not text"))?;
        if normalize(value.as_deref()) != words {
            return Err(format!("{at}: words otherwise than the reference's"));
        }
    }
    Ok(Counts {
        lowered: lower.len(),
        values: values.len(),
    })
}

/// A seeded fault: what it changes in the vectors.
type Seed = Box<dyn Fn(&mut Value)>;

fn vectors() -> Value {
    serde_json::from_str(VECTORS).unwrap()
}

#[test]
fn words_are_the_reference_s() {
    let counts = check(&vectors()).unwrap();
    assert_eq!(
        counts,
        Counts {
            lowered: 1488,
            values: 772
        }
    );
}

#[test]
fn the_check_fails_closed() {
    let seeded: Vec<(&str, Seed)> = vec![
        (
            "a perturbed word",
            Box::new(|v| v["values"][4][1][0] = json!("Sport-100")),
        ),
        (
            "a perturbed lowering",
            Box::new(|v| v["lower"]["65"] = json!([66])),
        ),
        ("no values", Box::new(|v| v["values"] = json!([]))),
        ("values missing", Box::new(|v| v["values"] = Value::Null)),
        ("no lowering", Box::new(|v| v["lower"] = json!({}))),
        (
            "a retyped value",
            Box::new(|v| v["values"][4][0] = json!(4)),
        ),
        (
            "a retyped word",
            Box::new(|v| v["values"][4][1][0] = json!(1)),
        ),
        (
            "a retyped lowering",
            Box::new(|v| v["lower"]["65"] = json!("a")),
        ),
    ];
    for (fault, seed) in seeded {
        let mut vectors = vectors();
        seed(&mut vectors);
        assert!(check(&vectors).is_err(), "{fault} passed");
    }
}
