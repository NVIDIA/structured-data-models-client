// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! No definition panics the core's view steps and words (T3.5 fix 2, the
//! owner's stop rule): a few seconds of metric views whose parameters,
//! joins and dimensions are made of random pieces, each read, its
//! parameters' defaults converted for every type, its joins walked, and
//! its values quoted. Each may refuse, by name; none may panic, and no
//! text is built past its bound (Codex review 2, P1-5).

use sdm_core::pql::yaml_rules;
use sdm_core::{view_steps, view_words};
use sdm_data::view_expr::metric::DefaultRefusal;
use sdm_schema::view::metric;
use sdm_schema::view::spec::ColumnItem;
use sdm_schema::view::yaml;

const TYPES: &[&str] = &[
    "STRING",
    "INT",
    "BIGINT",
    "DOUBLE",
    "DECIMAL(10,2)",
    "BOOLEAN",
    "DATE",
    "TIMESTAMP",
    "TIMESTAMP_NTZ",
    "ARRAY<INT>",
    "VARCHAR(3)",
];

const VALUES: &[&str] = &[
    "1",
    "-0",
    "007",
    "1_000",
    "1.5",
    "1e3",
    "1.0e+3",
    ".inf",
    ".nan",
    "0x1f",
    "1:30",
    "yes",
    "~",
    "''",
    "'2024-01-01T12.5'",
    "2024-01-01",
    "2024-01-01 10:00:00+05",
    "'2024-W01-2'",
    "'\u{661}\u{662}'",
    "'1e999999999999999999'",
    "'1e-999999999999999999'",
    "'1e9223372036854775807'",
    "'1.0e-9223372036854775808'",
    "'nan'",
    "'+01:60'",
    "'2024-01-01T10:00:00 Z'",
    "[1, 2]",
    "{a: 1}",
    "&a [*a]",
    "|\n    #hello",
    "\"\\x1c7\\x1f\"",
    "'9999-W52-7'",
    "'20240101T1030'",
    "'\u{0}'",
    "'2024-01-01T24:00'",
];

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[(self.next() % from.len() as u64) as usize]
    }
}

/// Reads `definition` through every step and sentence of the core.
fn convert_every_way(definition: &str) {
    let Ok(mut spec) = metric::parse(definition, &yaml_rules()) else {
        return;
    };
    let _ = view_steps::fact_source(&spec);
    for parameter in spec.parameters.clone() {
        let Some(default) = parameter.default else {
            continue;
        };
        for data_type in TYPES {
            if let Err(refusal) = view_steps::default_text(data_type, &spec.document, default) {
                let _ =
                    view_words::default_refusal_text(&refusal, data_type, &spec.document, default);
            }
        }
    }
    for (_, parameter) in view_steps::parameters(&spec, &[]) {
        let _ = view_words::parameter_text(&parameter, &spec.document, None);
    }
    let _ = view_words::value_repr(&spec.document, spec.document.root());
    let _ = view_words::value_text(&spec.document, spec.document.root());
    for column in spec.columns.clone() {
        match column {
            ColumnItem::NotAMapping { cfg } => {
                let _ = view_words::dimension_not_a_mapping(&spec, cfg);
            }
            ColumnItem::Column(column) => {
                let _ = view_words::dimension_misses_text(&spec, column.cfg);
            }
        }
    }
    if let Ok(walk) = view_steps::walk_joins(&mut spec) {
        for message in &walk.messages {
            let _ = view_words::join_message_text(&spec, message);
        }
        for skipped in &walk.skipped {
            let _ = view_words::skip_reason_text(&spec, &skipped.alias, &skipped.reason);
        }
    }
}

#[test]
fn no_definition_panics_the_core() {
    let mut rng = Rng(0x0dd_ba11_cafe_f00d);
    let joins = [
        "joins: [{name: u, source: c.s.u, on: source.a = u.a}]\n",
        "joins: [{name: u, source: c.s.u, using: [a], cardinality: [1]}]\n",
        "joins: &j [{name: u, source: c.s.u, joins: *j}]\n",
        "joins: [1, {source: x}, {name: v, source: 'select 1'}]\n",
        "",
    ];
    for _ in 0..3_000 {
        let definition = format!(
            "source: {}\n{}parameters:\n- name: p\n  data_type: {}\n  default: {}\n\
             dimensions:\n- name: x\n  expr: p\n- {}\n",
            rng.pick(&["c.s.t", "'select 1'", "[a]", "x.y.z.w"]),
            rng.pick(&joins),
            rng.pick(TYPES),
            rng.pick(VALUES),
            rng.pick(VALUES),
        );
        convert_every_way(&definition);
    }
}

/// A value too large to quote is refused by name, not written: an alias
/// bomb as a default, and a decimal whose digits would pass the bound.
#[test]
fn a_value_too_large_to_write_is_refused_by_name() {
    let mut bomb = String::from("a0: &a0 [xxxxxxxxxxxxxxxx]\n");
    for i in 1..19 {
        bomb.push_str(&format!("a{i}: &a{i} [*a{}, *a{}]\n", i - 1, i - 1));
    }
    let document = yaml::load(&bomb, &yaml_rules()).unwrap();
    let root = document.root();
    assert!(view_words::value_repr(&document, root).is_err());
    assert_eq!(
        view_steps::default_text("STRING", &document, root),
        Err(DefaultRefusal::TooLong)
    );
    let document = yaml::load("v: '1e999999999999999999'", &yaml_rules()).unwrap();
    let node = document.get(document.root(), "v").unwrap();
    assert_eq!(
        view_steps::default_text("DOUBLE", &document, node),
        Err(DefaultRefusal::TooLong)
    );
    assert_eq!(
        view_words::default_refusal_text(&DefaultRefusal::TooLong, "DOUBLE", &document, node)
            .unwrap(),
        "its default is too long to write as a literal of type DOUBLE"
    );
}

/// The metric view's steps on a semantic view's definition answer, not
/// panic: it has no metric view's source, and no joins.
#[test]
fn a_semantic_view_passes_through_the_metric_steps() {
    use sdm_schema::view::semantic;
    let source = "tables:\n- name: A\n  base_table: {database: D, schema: S, table: T}\n\
                  - name: B\n  base_table: {database: D, schema: S, table: U}\n";
    let mut spec = semantic::parse(source, &yaml_rules()).unwrap();
    assert_eq!(view_steps::fact_source(&spec), Err(None));
    let walk = view_steps::walk_joins(&mut spec).unwrap();
    assert!(walk.kept.is_empty() && walk.skipped.is_empty());
    semantic::read_all(&mut spec).unwrap();
    assert_eq!(view_steps::fact_source(&spec), Err(None));
    let walk = view_steps::walk_joins(&mut spec).unwrap();
    assert!(walk.kept.is_empty() && walk.skipped.is_empty());
}

/// The bound on a written value is its characters, exactly (Codex review
/// 3, P1-4): a default of 2,100,000 `é` is read whole; a quoted text of
/// 1,100,000 escaped NULs, 4,400,002 characters, and 200,000 aliases of a
/// date are refused by name, never built past the bound.
#[test]
fn the_bound_counts_the_characters_written() {
    let wide = format!("v: \"{}\"", "é".repeat(2_100_000));
    let document = yaml::load(&wide, &yaml_rules()).unwrap();
    let node = document.get(document.root(), "v").unwrap();
    let text = view_steps::default_text("STRING", &document, node).unwrap();
    assert_eq!(text.chars().count(), 2_100_000);
    let nuls = format!("v: \"{}\"", "\\0".repeat(1_100_000));
    let document = yaml::load(&nuls, &yaml_rules()).unwrap();
    let node = document.get(document.root(), "v").unwrap();
    assert!(view_words::value_repr(&document, node).is_err());
    assert_eq!(
        view_words::value_text(&document, node).map(|t| t.len()),
        Ok(1_100_000)
    );
    let mut dates = String::from("v: [&d 2024-01-01");
    dates.push_str(&", *d".repeat(200_000));
    dates.push(']');
    let document = yaml::load(&dates, &yaml_rules()).unwrap();
    let node = document.get(document.root(), "v").unwrap();
    assert!(view_words::value_repr(&document, node).is_err());
    assert!(view_words::value_text(&document, node).is_err());
}

/// The class the data layer's patterns read `\s` by is derived from the
/// parser's one rule of white space: it holds exactly the code points the
/// rule holds, every one of them tested (Codex review 4, P1).
#[test]
fn the_patterns_white_space_is_the_one_rule() {
    let spaces = sdm_core::pql::spaces();
    let mut members = Vec::new();
    for range in spaces.class().split("\\x{").skip(1) {
        let (code, rest) = range.split_once('}').unwrap();
        let code = u32::from_str_radix(code, 16).unwrap();
        if rest == "-" {
            members.push((code, None));
        } else {
            match members.last_mut() {
                Some((_, last @ None)) => *last = Some(code),
                _ => members.push((code, Some(code))),
            }
        }
    }
    let held = |code: u32| {
        members
            .iter()
            .any(|&(first, last)| first <= code && code <= last.unwrap_or(first))
    };
    for code in 0..=u32::from(char::MAX) {
        let Some(c) = char::from_u32(code) else {
            continue;
        };
        assert_eq!(held(code), spaces.holds(c), "U+{code:04X}");
    }
}
