// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The date reader held to the reference's readings: T0.6's 45 PQL
//! literals (`rust/testdata/t06-vectors.json`), the generated literals of
//! `dates.json` (D8's packed dates and long fields, and a grid of D2 to D7
//! with D4's warning; `dates_record.py`), and the first attempt's
//! `padded-years.tsv`, where Linux's `strftime` and macOS's differ only in
//! D4's warning.
//!
//! Each literal compares its date and time, its offset, and D4's warning's
//! format; every kind of answer must be held, and each file must hold the
//! rows it promises, so a vector set read short or empty fails.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the vector files"
)]

use std::path::Path;

use sdm_pql::dates::{Reader, Reading};
use serde_json::Value;

fn here(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
    std::fs::read_to_string(path).unwrap()
}

/// `[value, offset, format]` as a row records them.
type Answer = (Option<String>, Option<i64>, Option<String>);

fn answer(reader: Reader, literal: &str) -> Answer {
    match reader.read(literal) {
        Reading::Read { stamp, day_first } => (
            Some(stamp.at.to_string()),
            stamp.offset.map(i64::from),
            day_first.map(|d| d.format()),
        ),
        Reading::Refused => (None, None, None),
    }
}

/// What the rows' answers held: reads, refusals, offsets and warnings.
#[derive(Default, Debug)]
struct Held {
    read: usize,
    refused: usize,
    offset: usize,
    warned: usize,
}

impl Held {
    fn count(&mut self, answer: &Answer) {
        match answer {
            (None, ..) => self.refused += 1,
            (Some(_), offset, format) => {
                self.read += 1;
                self.offset += usize::from(offset.is_some());
                self.warned += usize::from(format.is_some());
            }
        }
    }
}

/// Each `[literal, value, offset, format]` row against the reader.
fn check(reader: Reader, rows: &[Value], failures: &mut Vec<String>) -> Held {
    let mut held = Held::default();
    for row in rows {
        let literal = row[0].as_str().unwrap();
        let want: Answer = (
            row[1].as_str().map(str::to_owned),
            row[2].as_i64(),
            row[3].as_str().map(str::to_owned),
        );
        // A refused literal gives no warning here: the reference warns as it
        // guesses a format, before it refuses.
        let want = match want {
            (None, ..) => (None, None, None),
            read => read,
        };
        let got = answer(reader, literal);
        if got != want {
            failures.push(format!("{literal}: {got:?}, the reference {want:?}"));
        }
        held.count(&want);
    }
    held
}

#[test]
fn the_t06_literals_read_as_the_reference_reads_them() {
    let vectors: Value = serde_json::from_str(&here("../../testdata/t06-vectors.json")).unwrap();
    let year = vectors["meta"]["chrono"]["today_year"].as_i64().unwrap();
    let reader = Reader {
        year: i16::try_from(year).unwrap(),
        pads_year: true,
    };
    let mut failures = Vec::new();
    let mut held = Held::default();
    let mut rows = 0;
    for task in vectors["tasks"].as_array().unwrap() {
        if task["kind"] != "pql_literal" {
            continue;
        }
        rows += 1;
        let literal = task["input"]["literal"].as_str().unwrap();
        let reference = &task["ref"];
        // The reference writes an offset after the time: `...T06:00:00-09:00`.
        let want: Answer = match reference["value"].as_str() {
            None => (None, None, None),
            Some(value) => {
                let (at, offset) = value.split_at(19);
                let offset = (!offset.is_empty()).then(|| {
                    let sign = if offset.starts_with('-') { -1 } else { 1 };
                    let hours: i64 = offset[1..3].parse().unwrap();
                    let minutes: i64 = offset[4..6].parse().unwrap();
                    sign * (hours * 3600 + minutes * 60)
                });
                let warned = reference["warned"] == true;
                (Some(at.to_owned()), offset, warned.then_some(String::new()))
            }
        };
        let got = answer(reader, literal);
        // T0.6 records whether a warning was given, not its format.
        let got = (got.0, got.1, got.2.map(|_| String::new()));
        if got != want {
            failures.push(format!("{literal}: {got:?}, the reference {want:?}"));
        }
        held.count(&want);
    }
    assert_eq!(rows, 45, "T0.6's 45 literals");
    assert!(
        held.read > 0 && held.refused > 0 && held.offset > 0 && held.warned > 0,
        "{held:?}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_generated_literals_read_as_the_reference_reads_them() {
    let vectors: Value = serde_json::from_str(&here("tests/dates.json")).unwrap();
    assert_eq!(vectors["format"], "sdm-pql-dates/1");
    let reader = Reader {
        year: i16::try_from(vectors["year"].as_i64().unwrap()).unwrap(),
        pads_year: vectors["pads_year"].as_bool().unwrap(),
    };
    let mut failures = Vec::new();
    let packed = vectors["packed"]["rows"].as_array().unwrap();
    let held = check(reader, packed, &mut failures);
    // D8's rows: every one has a run of five digits or more.
    for row in packed {
        let literal = row[0].as_str().unwrap();
        assert!(
            literal
                .split([' ', '/', '-', '.', ':'])
                .any(|run| run.len() >= 5),
            "{literal}"
        );
    }
    assert!(packed.len() >= 7_000, "{}", packed.len());
    assert!(
        held.read >= 4_000 && held.refused >= 3_000 && held.offset >= 300,
        "{held:?}"
    );
    let plain = vectors["plain"].as_array().unwrap();
    let held = check(reader, plain, &mut failures);
    assert!(plain.len() >= 9_000, "{}", plain.len());
    assert!(
        held.read >= 1_000 && held.refused >= 1_000 && held.warned >= 1_000,
        "{held:?}"
    );
    assert!(
        failures.is_empty(),
        "{} differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_padded_year_warns_only_where_strftime_pads_it() {
    let text = here("tests/padded-years.tsv");
    let mut failures = Vec::new();
    let mut rows = 0;
    for line in text.lines() {
        if line.starts_with('#') || line.starts_with("literal\t") {
            continue;
        }
        let [literal, value, warning] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("a row of three: {line}");
        };
        rows += 1;
        let format = warning
            .strip_prefix("Parsing dates in ")
            .and_then(|rest| rest.split_once(" format when dayfirst=False"))
            .map(|(format, _)| format.to_owned());
        assert!(format.is_some(), "{literal}: every row warns where %Y pads");
        for (pads, want) in [(true, format.clone()), (false, None)] {
            let reader = Reader {
                year: 2026,
                pads_year: pads,
            };
            let got = answer(reader, literal);
            if got != (Some(value.to_owned()), None, want.clone()) {
                failures.push(format!("{literal} (pads {pads}): {got:?}"));
            }
        }
    }
    assert_eq!(rows, 885, "the 885 literals the platforms read apart");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Register PQL-4 (proposed), diagnostic: a day-first literal whose middle
/// field is written as `strftime` writes its year (`13/0001/1`) makes pandas
/// guess `%d/%Y/%m` and warn with it; the reader reads the same date and
/// names no format, as D4 states. Every other answer of the set agrees.
#[test]
fn a_format_pandas_guesses_from_a_year_written_as_a_month_is_not_named() {
    let vectors: Value = serde_json::from_str(&here("tests/dates.json")).unwrap();
    let reader = Reader {
        year: i16::try_from(vectors["year"].as_i64().unwrap()).unwrap(),
        pads_year: vectors["pads_year"].as_bool().unwrap(),
    };
    let rows = vectors["format_guess"].as_array().unwrap();
    let mut guessed = 0;
    let mut agreed = 0;
    for row in rows {
        let literal = row[0].as_str().unwrap();
        let got = answer(reader, literal);
        assert_eq!(got.0.as_deref(), row[1].as_str(), "{literal}");
        assert_eq!(got.1, row[2].as_i64(), "{literal}");
        match row[3].as_str() {
            Some(format) if format[2..].starts_with(&format!("{}%Y", &format[2..3])) => {
                guessed += 1;
                assert_eq!(got.2, None, "{literal}");
            }
            want => {
                agreed += 1;
                assert_eq!(got.2.as_deref(), want, "{literal}");
            }
        }
    }
    assert_eq!(guessed, 144, "the literals pandas guesses %Y for");
    assert!(agreed > 800, "{agreed}");
}
