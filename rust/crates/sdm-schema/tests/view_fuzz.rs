// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No text panics the view definitions' readers (T3.5 fix 2, the owner's
//! stop rule): a few seconds of texts made of YAML's pieces and the view
//! sections' names, and of the recorded texts cut and spliced, each read
//! as YAML, as a metric view with its joins walked, and as a semantic view
//! read whole. A reader may refuse any of them; none may panic.

use sdm_schema::view::joins;
use sdm_schema::view::yaml::{self, Rules};
use sdm_schema::view::{metric, semantic};
use serde_json::Value as Json;

const VECTORS: &str = include_str!("../../../../contract/view-yaml-vectors.json");

const RULES: Rules = Rules { most_digits: 4300 };

/// The pieces the texts are made of.
const PIECES: &[&str] = &[
    "v",
    ":",
    ": ",
    " ",
    "  ",
    "\n",
    "\n  ",
    "\n    ",
    "- ",
    "-",
    "[",
    "]",
    "{",
    "}",
    ",",
    "&a ",
    "*a",
    "&b ",
    "*b",
    "|",
    ">",
    "|+",
    ">-",
    "|2",
    "#",
    " # c",
    "'",
    "\"",
    "!",
    "!!str ",
    "<<",
    "? ",
    "0x1f",
    "017",
    "1",
    "9",
    ".",
    "e",
    "~",
    "yes",
    "on",
    "\t",
    "\r",
    "\r\n",
    "...",
    "---",
    "é",
    "\u{85}",
    "\u{2028}",
    "%YAML 1.1\n",
    "2024-01-01",
    "2024-01-01 10:00:00+05",
    "1:30",
    ".nan",
    "=",
    "joins",
    "name",
    "source",
    "c.s.t",
    "tables",
    "base_table",
    "table",
    "schema",
    "database",
    "relationship_columns",
    "left_column",
    "right_column",
    "left_table",
    "right_table",
    "relationships",
    "primary_key",
    "columns",
    "using",
    "on",
    "cardinality",
    "many_to_one",
    "dimensions",
    "time_dimensions",
    "facts",
    "measures",
    "parameters",
    "default",
    "data_type",
    "expr",
    "filter",
    "window",
    "order",
    "materialization",
    "fields",
    "0",
    "\u{feff}",
    "\\",
    "\\u0000",
    "\0",
];

/// A deterministic xorshift generator, so a failure repeats.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Reads `text` every way a view definition is read.
fn read_every_way(text: &str) {
    let _ = yaml::load(text, &RULES);
    if let Ok(mut spec) = metric::parse(text, &RULES) {
        let mut flip = false;
        let _ = joins::walk(
            &mut spec,
            |_, _| {
                flip = !flip;
                if flip { Ok(()) } else { Err(()) }
            },
            |_, _| Ok::<(), ()>(()),
        );
    }
    if let Ok(mut spec) = semantic::parse(text, &RULES) {
        let _ = semantic::read_all(&mut spec);
        for item in spec.relationships.clone() {
            let _ = spec.relationship(&item);
        }
        let _ = joins::walk(&mut spec, |_, _| Ok::<(), ()>(()), |_, _| Ok::<(), ()>(()));
    }
}

#[test]
fn no_text_panics_a_reader() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for _ in 0..20_000 {
        let length = 1 + rng.below(40);
        let text: String = (0..length)
            .map(|_| PIECES[rng.below(PIECES.len())])
            .collect();
        read_every_way(&text);
    }
    let document: Json = serde_json::from_str(VECTORS).unwrap();
    let texts: Vec<&str> = document["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|vector| vector["yaml"].as_str())
        .collect();
    for _ in 0..10_000 {
        let a = texts[rng.below(texts.len())];
        let b = texts[rng.below(texts.len())];
        let cut = |text: &str, at: usize| {
            let at = at % (text.len() + 1);
            (0..=at)
                .rev()
                .find(|&i| text.is_char_boundary(i))
                .unwrap_or(0)
        };
        let (i, j) = (cut(a, rng.below(64)), cut(b, rng.below(64)));
        let piece = PIECES[rng.below(PIECES.len())];
        read_every_way(&format!("{}{piece}{}", &a[..i], &b[j..]));
    }
}
