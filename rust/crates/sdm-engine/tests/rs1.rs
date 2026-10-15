// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RS-1 against numpy, from two files of vectors recorded from numpy 2.5.3
//! (protected paths: they are never recorded again to pass a dependency
//! bump, F4):
//!
//! - `rs1.json` (T1.11c): the seed sequence's state, the raw 64-bit draws,
//!   and a run of shuffles on one generator, which carries the 32-bit
//!   draw's kept half from one shuffle to the next, for seeds from 0 to
//!   past 2**200 (more entropy words than the pool holds);
//! - `rs1-draws.json` (T1.22, `rs1_record.py`): T0.6's model of the example
//!   shuffle for each of its seeds, 0, 42, 123456789 and 2**70 (a fresh
//!   generator's state, raw draws and `shuffle(arange(50))`), and numpy's
//!   bounded draw itself, `random_interval`, over maxima that cross the
//!   32-bit path's edge and reach the 64-bit path's end, called from
//!   numpy's compiled library.
//!
//! [`check`] compares a document with the port and counts what it
//! compared; it fails closed: a perturbed draw, and evidence missing,
//! empty or retyped, each fail (seeded below).

use sdm_engine::random::{Entropy, SeedSequence, default_rng};
use serde_json::Value;

const RS1: &str = include_str!("rs1.json");
const DRAWS: &str = include_str!("rs1-draws.json");

/// T0.6's seeds for the example shuffle, in hex.
const MODEL_SEEDS: [&str; 4] = ["0x0", "0x2a", "0x75bcd15", "0x400000000000000000"];

fn text<'a>(value: &'a Value, at: &str) -> Result<&'a str, String> {
    value.as_str().ok_or_else(|| format!("{at}: not text"))
}

fn list<'a>(value: &'a Value, at: &str) -> Result<&'a [Value], String> {
    match value.as_array() {
        Some(items) if !items.is_empty() => Ok(items),
        Some(_) => Err(format!("{at}: empty")),
        None => Err(format!("{at}: not a list")),
    }
}

fn hex(value: &Value, at: &str) -> Result<u64, String> {
    let digits = text(value, at)?
        .strip_prefix("0x")
        .ok_or_else(|| format!("{at}: not 0x text"))?;
    u64::from_str_radix(digits, 16).map_err(|_| format!("{at}: not a u64"))
}

fn seed(value: &Value, at: &str) -> Result<Entropy, String> {
    let digits = text(value, at)?
        .strip_prefix("0x")
        .ok_or_else(|| format!("{at}: not 0x text"))?;
    Entropy::from_hex(digits).ok_or_else(|| format!("{at}: not a seed"))
}

fn same(got: u64, want: u64, at: &str) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!("{at}: the port draws {got:#x}, numpy {want:#x}"))
    }
}

/// The state and raw draws of `case`'s seed on a fresh generator, which is
/// left after them; the number of values compared.
fn state_and_raw(case: &Value, at: &str) -> Result<(sdm_engine::random::Pcg64, usize), String> {
    let entropy = seed(&case["seed"], &format!("{at}/seed"))?;
    let state = list(&case["state"], &format!("{at}/state"))?;
    let got = SeedSequence::new(&entropy).generate_state_u64(4);
    if state.len() != got.len() {
        return Err(format!("{at}/state: not four words"));
    }
    for (i, want) in state.iter().enumerate() {
        same(got[i], hex(want, &format!("{at}/state/{i}"))?, at)?;
    }
    let mut rng = default_rng(&entropy);
    let raw = list(&case["raw"], &format!("{at}/raw"))?;
    for (i, want) in raw.iter().enumerate() {
        same(rng.next_u64(), hex(want, &format!("{at}/raw/{i}"))?, at)?;
    }
    Ok((rng, state.len() + raw.len()))
}

/// `shuffle(arange(len))` on `rng`, compared with `shuffled`.
fn shuffle(
    rng: &mut sdm_engine::random::Pcg64,
    shuffled: &Value,
    at: &str,
) -> Result<usize, String> {
    let want: Vec<u64> = list(shuffled, at)?
        .iter()
        .map(|v| v.as_u64().ok_or_else(|| format!("{at}: not a position")))
        .collect::<Result<_, _>>()?;
    let mut got: Vec<u64> = (0..want.len() as u64).collect();
    rng.shuffle(&mut got);
    if got != want {
        return Err(format!("{at}: the port's shuffle is not numpy's"));
    }
    Ok(want.len())
}

/// Compares `document` with the port: the number of values compared, or
/// the first difference or fault in the evidence.
fn check(document: &Value) -> Result<usize, String> {
    let mut compared = 0;
    match document["format"].as_str() {
        Some("sdm-rs1/1") => {
            for (n, case) in list(&document["cases"], "cases")?.iter().enumerate() {
                let at = format!("cases/{n}");
                let (mut rng, count) = state_and_raw(case, &at)?;
                compared += count;
                let shuffles = list(&case["shuffles"], &format!("{at}/shuffles"))?;
                for (i, shuffled) in shuffles.iter().enumerate() {
                    compared += shuffle(&mut rng, shuffled, &format!("{at}/shuffles/{i}"))?;
                }
            }
        }
        Some("sdm-rs1-draws/1") => {
            for (n, case) in list(&document["model"], "model")?.iter().enumerate() {
                let at = format!("model/{n}");
                let (_, count) = state_and_raw(case, &at)?;
                compared += count;
                let mut fresh = default_rng(&seed(&case["seed"], &at)?);
                compared += shuffle(&mut fresh, &case["shuffle"], &format!("{at}/shuffle"))?;
            }
            for (n, case) in list(&document["intervals"], "intervals")?
                .iter()
                .enumerate()
            {
                let at = format!("intervals/{n}");
                let mut rng = default_rng(&seed(&case["seed"], &format!("{at}/seed"))?);
                for (i, pair) in list(&case["draws"], &format!("{at}/draws"))?
                    .iter()
                    .enumerate()
                {
                    let at = format!("{at}/draws/{i}");
                    let [max, draw] = list(pair, &at)? else {
                        return Err(format!("{at}: not a pair"));
                    };
                    let max = hex(max, &at)?;
                    let draw = hex(draw, &at)?;
                    if draw > max {
                        return Err(format!("{at}: numpy's draw is past its maximum"));
                    }
                    same(rng.interval(max), draw, &at)?;
                    compared += 1;
                }
            }
        }
        _ => return Err("not a format of RS-1's vectors".into()),
    }
    Ok(compared)
}

fn parsed(text: &str) -> Value {
    serde_json::from_str(text).expect("the vectors are JSON")
}

#[test]
fn the_streams_are_numpy_s() {
    let document = parsed(RS1);
    assert_eq!(document["cases"].as_array().unwrap().len(), 11);
    // 11 seeds: four state words, five raw draws, and the positions of
    // shuffles of 1, 2, 3, 8, 10, 33 and 1000 each.
    assert_eq!(check(&document), Ok(11 * (4 + 5 + 1057)));
}

#[test]
fn t06_s_model_and_the_bounded_draws_are_numpy_s() {
    let document = parsed(DRAWS);
    let seeds = |section: &str| -> Vec<String> {
        document[section]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| case["seed"].as_str().unwrap().to_owned())
            .collect()
    };
    // Every seed of T0.6's model, 2**70 among them, in both sections.
    assert_eq!(seeds("model"), MODEL_SEEDS);
    assert_eq!(seeds("intervals"), MODEL_SEEDS);
    // Both of the bounded draw's paths, and each edge between them.
    let maxima: Vec<u64> = document["intervals"][0]["draws"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| hex(&pair[0], "max").unwrap())
        .collect();
    for edge in [0, u64::from(u32::MAX), 1 << 32, u64::MAX] {
        assert!(maxima.contains(&edge), "{edge:#x}");
    }
    // A 32-bit draw after a 64-bit one: the kept half outlives it.
    let first_wide = maxima
        .iter()
        .position(|m| *m > u64::from(u32::MAX))
        .unwrap();
    assert!(
        maxima[first_wide..]
            .iter()
            .any(|m| *m <= u64::from(u32::MAX))
    );
    let draws: usize = maxima.len() * MODEL_SEEDS.len();
    assert_eq!(check(&document), Ok(4 * (4 + 5 + 50) + draws));
}

/// `document` with `edit` applied, which must fail [`check`] with a message
/// holding `expected`.
fn fails(document: &Value, edit: impl FnOnce(&mut Value), expected: &str) {
    let mut changed = document.clone();
    edit(&mut changed);
    assert_ne!(&changed, document, "the seeded fault changed nothing");
    match check(&changed) {
        Err(message) => assert!(message.contains(expected), "{message}"),
        Ok(count) => panic!("a seeded fault passed, comparing {count} values"),
    }
}

fn bump(value: &mut Value) {
    let n = hex(value, "seeded").unwrap();
    *value = Value::String(format!("{:#x}", n ^ 1));
}

#[test]
fn a_perturbed_draw_fails() {
    let rs1 = parsed(RS1);
    let draws = parsed(DRAWS);
    let differs = "numpy";
    fails(&rs1, |d| bump(&mut d["cases"][3]["state"][2]), differs);
    fails(&rs1, |d| bump(&mut d["cases"][10]["raw"][4]), differs);
    fails(
        &rs1,
        |d| {
            d["cases"][8]["shuffles"][6]
                .as_array_mut()
                .unwrap()
                .swap(0, 999)
        },
        "shuffle is not numpy's",
    );
    fails(&draws, |d| bump(&mut d["model"][3]["raw"][0]), differs);
    fails(
        &draws,
        |d| d["model"][2]["shuffle"].as_array_mut().unwrap().swap(3, 4),
        "shuffle is not numpy's",
    );
    // Each path of the bounded draw: a 32-bit maximum, the 32-bit edge, a
    // 64-bit maximum, and the last 32-bit draw after the 64-bit ones.
    for at in [5, 14, 21, 27] {
        fails(
            &draws,
            |d| bump(&mut d["intervals"][1]["draws"][at][1]),
            differs,
        );
    }
    // A draw past its maximum is numpy's evidence broken, not the port.
    fails(
        &draws,
        |d| d["intervals"][0]["draws"][1][1] = Value::String("0x2".into()),
        "past its maximum",
    );
}

#[test]
fn evidence_missing_empty_or_retyped_fails() {
    let rs1 = parsed(RS1);
    let draws = parsed(DRAWS);
    // Missing.
    fails(&rs1, |d| d["cases"] = Value::Null, "cases: not a list");
    fails(
        &draws,
        |d| d["intervals"] = Value::Null,
        "intervals: not a list",
    );
    fails(
        &draws,
        |d| {
            d["model"][0].as_object_mut().unwrap().remove("shuffle");
        },
        "model/0/shuffle: not a list",
    );
    fails(
        &draws,
        |d| d["format"] = Value::String("sdm-rs1-draws/2".into()),
        "not a format",
    );
    // Empty.
    fails(&rs1, |d| d["cases"] = Value::Array(vec![]), "cases: empty");
    fails(
        &draws,
        |d| d["model"] = Value::Array(vec![]),
        "model: empty",
    );
    fails(
        &draws,
        |d| d["intervals"][2]["draws"] = Value::Array(vec![]),
        "intervals/2/draws: empty",
    );
    fails(
        &rs1,
        |d| d["cases"][0]["raw"] = Value::Array(vec![]),
        "raw: empty",
    );
    // Retyped.
    fails(
        &rs1,
        |d| d["cases"][0]["raw"][0] = Value::from(0xa30f_ebcf_u64),
        "not text",
    );
    fails(
        &draws,
        |d| d["intervals"][0]["draws"][4] = Value::String("0x7".into()),
        "not a list",
    );
    fails(
        &draws,
        |d| {
            d["intervals"][0]["draws"][4].as_array_mut().unwrap().pop();
        },
        "not a pair",
    );
    fails(
        &draws,
        |d| d["model"][1]["shuffle"][0] = Value::String("37".into()),
        "not a position",
    );
    fails(
        &draws,
        |d| d["model"][1]["seed"] = Value::String("42".into()),
        "not 0x text",
    );
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
