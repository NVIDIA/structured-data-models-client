// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A bounded PQL fuzz test: a fixed number of inputs from a fixed seed,
//! each validated through the adapter, which must answer every one (an
//! acceptance, PQL's diagnostics, or a `CoreError`) without a panic, within
//! a time bound. The inputs are the corpus's queries mutated (characters
//! dropped, repeated, swapped or inserted from the grammar's alphabet),
//! queries made of the grammar's tokens at random, and date literals of
//! random digit runs, separators and times of day (D1 to D9).

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: reads the corpus and bounds the run against a wall clock"
)]

mod pql_compare;

use std::path::Path;
use std::time::{Duration, Instant};

use pql_compare::{NoZones, facts, profile, query_schema};
use sdm_core::pql::validate;
use serde_json::Value;

/// The seed and the number of inputs of each kind.
const SEED: u64 = 0x5d47_0cb6;
const MUTATED: usize = 6_000;
const TOKENS: usize = 3_000;
const DATES: usize = 6_000;
/// The whole run's bound, in a debug build.
const BOUND: Duration = Duration::from_secs(300);

/// xorshift64*: a fixed stream, so a failure names an input that repeats.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % (n as u64)).unwrap()
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

const ALPHABET: &[char] = &[
    '(', ')', ',', '.', '*', '=', '<', '>', '!', '\'', '"', '`', '#', '-', '+', '/', ':', ' ',
    '\n', '0', '1', '9', 'a', 'Z', '_', '%', '\\', 'é', '\u{3000}', '\u{0}',
];

const TOKENS_OF: &[&str] = &[
    "PREDICT",
    "FOR",
    "EACH",
    "WHERE",
    "ASSUMING",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "LIKE",
    "CLASSIFY",
    "RANK",
    "TOP",
    "FORECAST",
    "TIMEFRAMES",
    "COUNT",
    "SUM",
    "AVG",
    "LIST_DISTINCT",
    "FIRST",
    "LAST",
    "days",
    "hours",
    "-INF",
    "(",
    ")",
    ",",
    "=",
    ">",
    "<=",
    "!=",
    "users.user_id",
    "users.age",
    "orders.*",
    "orders.ts",
    "orders.price",
    "3",
    "-7",
    "2.5",
    "'x'",
    "TRUE",
    "13/01/2024",
    "20240101-01-01",
    "EXPLAIN",
    "EVALUATE",
    "STARTS WITH",
];

fn mutated(rng: &mut Rng, query: &str) -> String {
    let mut chars: Vec<char> = query.chars().collect();
    for _ in 0..=rng.below(4) {
        if chars.is_empty() {
            chars.push(*rng.pick(ALPHABET));
            continue;
        }
        let at = rng.below(chars.len());
        match rng.below(4) {
            0 => {
                chars.remove(at);
            }
            1 => {
                let end = (at + 1 + rng.below(12)).min(chars.len());
                let run: Vec<char> = chars[at..end].to_vec();
                for (i, c) in run.into_iter().enumerate() {
                    chars.insert(end + i, c);
                }
            }
            2 => {
                let other = rng.below(chars.len());
                chars.swap(at, other);
            }
            _ => chars.insert(at, *rng.pick(ALPHABET)),
        }
    }
    chars.into_iter().collect()
}

fn tokens(rng: &mut Rng) -> String {
    let n = 1 + rng.below(30);
    let mut words = vec!["PREDICT"];
    for _ in 0..n {
        words.push(rng.pick(TOKENS_OF));
    }
    words.join(" ")
}

fn digits(rng: &mut Rng) -> String {
    let n = 1 + match rng.below(4) {
        0 => rng.below(25),
        _ => rng.below(5),
    };
    (0..n)
        .map(|_| char::from(b'0' + u8::try_from(rng.below(10)).unwrap()))
        .collect()
}

fn date(rng: &mut Rng) -> String {
    let sep = *rng.pick(&['/', '-', '.']);
    let mut literal = format!("{}{sep}{}{sep}{}", digits(rng), digits(rng), digits(rng));
    if rng.below(3) == 0 {
        literal = format!("{literal} {}:{}:{}", digits(rng), digits(rng), digits(rng));
    }
    format!(
        "PREDICT COUNT(orders.* WHERE orders.ordered_at > {literal}, 0, 30, days) FOR EACH \
         users.user_id"
    )
}

#[test]
fn the_parser_answers_every_fuzzed_input_without_a_panic() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/pql/corpus.json");
    let corpus: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let schemas = corpus["schemas"].as_object().unwrap();
    let cases = corpus["cases"].as_array().unwrap();
    let mut rng = Rng(SEED);
    // Each input with the schema and profile it is validated with: a
    // mutated case keeps its own; the rest take the first schema, which
    // holds `orders.ordered_at`, and a named profile.
    let mut inputs: Vec<(String, &str, &str)> = Vec::new();
    for _ in 0..MUTATED {
        let case = rng.pick(cases);
        let query = mutated(&mut rng, case["query"].as_str().unwrap());
        inputs.push((
            query,
            case["schema"].as_str().unwrap(),
            case["profile"].as_str().unwrap(),
        ));
    }
    for _ in 0..TOKENS {
        let profile = *rng.pick(&["RELATIONAL", "TABULAR", "FINE_TUNING"]);
        inputs.push((tokens(&mut rng), "schema-1", profile));
    }
    for _ in 0..DATES {
        inputs.push((date(&mut rng), "schema-1", "RELATIONAL"));
    }
    let started = Instant::now();
    let (mut accepted, mut refused, mut errors) = (0, 0, 0);
    for (input, schema, name) in &inputs {
        let schema = query_schema(&schemas[*schema]);
        match validate(input, &schema, &profile(name), &facts(), &NoZones) {
            Ok((answer, _)) if answer.accepted => accepted += 1,
            Ok(_) => refused += 1,
            Err(_) => errors += 1,
        }
    }
    let took = started.elapsed();
    println!(
        "PQL fuzz: {} inputs, {accepted} accepted, {refused} refused, {errors} errors, {took:?}",
        inputs.len()
    );
    assert!(took < BOUND, "{took:?}");
    assert_eq!(inputs.len(), MUTATED + TOKENS + DATES);
    // Every kind of answer is reached, so the inputs are not all refused at
    // the first character. From T1.6c a date literal with a run of five
    // digits or more (D8) is refused by name, a `CoreError`, where it was
    // read or refused with a diagnostic: about 4,800 inputs, most of them
    // date inputs, end so, and are counted among the errors.
    assert!(
        accepted > 150 && refused > 8_000 && errors > 2_000,
        "{accepted} accepted, {refused} refused, {errors} errors"
    );
}
