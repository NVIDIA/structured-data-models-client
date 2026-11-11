// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A bounded PQL fuzz test: a fixed number of inputs from a fixed seed,
//! each validated through the adapter, which must answer every one (an
//! acceptance, PQL's diagnostics, or a `CoreError`) without a panic, within
//! a time bound. The inputs are authored queries mutated (characters
//! dropped, repeated, swapped or inserted from the grammar's alphabet),
//! queries made of the grammar's tokens at random, and date literals of
//! random digit runs, separators and times of day (D1 to D9).

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test code: bounds the run against a wall clock"
)]

mod pql_support;

use std::time::{Duration, Instant};

use pql_support::{NoZones, facts, profile, schema};
use sdm_core::pql::validate;

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
    let seeds = [
        (
            "PREDICT SUM(orders.quantity, 0, 30, days) FOR EACH users.user_id",
            "schema-1",
            "RELATIONAL",
        ),
        (
            "PREDICT users.age FOR EACH users.user_id",
            "schema-1",
            "TABULAR",
        ),
        (
            "PREDICT LIST_DISTINCT(orders.product_id, 0, 30, days) RANK TOP 5 FOR EACH users.user_id",
            "schema-1",
            "RELATIONAL",
        ),
        (
            "PREDICT COUNT(orders.*, 0, 1, days) FORECAST 3 TIMEFRAMES FOR EACH users.user_id",
            "schema-1",
            "RELATIONAL",
        ),
        (
            "PREDICT COUNT(orders.*, 0, 30, days) > 0 OR COUNT(orders.*, 0, 7, days) > 2 AND NOT users.vip = true FOR EACH users.user_id",
            "schema-1",
            "FINE_TUNING",
        ),
        (
            "PREDICT SUM(`Order Lines`.`Unit Price`, 0, 30, days) FOR EACH Customers.`Customer ID`",
            "schema-2",
            "RELATIONAL",
        ),
        (
            "PREDICT COUNT(orders.* WHERE orders.ordered_at > 2024-02-29, 0, 30, days) FOR EACH users.user_id",
            "schema-1",
            "RELATIONAL",
        ),
        (
            "PREDICT COUNT(orders.* WHERE orders.ordered_at > 12345-1-1, 0, 30, days) FOR EACH users.user_id",
            "schema-1",
            "RELATIONAL",
        ),
        ("PREDICT (", "schema-1", "RELATIONAL"),
    ];
    // Explicit controls prevent a broken generator from exercising only refusals.
    for ((input, schema_name, profile_name), accepted) in
        [(seeds[0], true), (seeds[1], true), (seeds[8], false)]
    {
        assert_eq!(
            validate(
                input,
                &schema(schema_name),
                &profile(profile_name),
                &facts(),
                &NoZones
            )
            .unwrap()
            .0
            .accepted,
            accepted
        );
    }
    let mut rng = Rng(SEED);
    // Each input with the schema and profile it is validated with: a
    // mutated case keeps its own; the rest take the first schema, which
    // holds `orders.ordered_at`, and a named profile.
    let mut inputs: Vec<(String, &str, &str)> = Vec::new();
    for _ in 0..MUTATED {
        let (query, schema, profile) = rng.pick(&seeds);
        inputs.push((mutated(&mut rng, query), schema, profile));
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
    for (input, schema_name, name) in &inputs {
        let schema = schema(schema_name);
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
    // Keep acceptance, diagnostic refusal and structured-error paths exercised;
    // exact proportions depend on the authored mutation seeds.
    assert!(
        accepted > 0 && refused > 0 && errors > 0,
        "{accepted} accepted, {refused} refused, {errors} errors"
    );
}
