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
//!
//! From T1.6c a literal with a run of five digits or more (D8) is refused
//! by name ([`Reading::Packed`]) instead of read: its rows keep the
//! reference's reading, as evidence of the old behaviour, and are held to
//! `Packed` here instead; every other row is held to the reference's
//! reading as before.

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
        Reading::Unplaced => panic!("{literal}: unplaced, with a year given"),
        Reading::Packed => panic!("{literal}: packed, with no run of five digits or more"),
    }
}

/// Whether `literal` has a run of five digits or more in its `DATE` or
/// `TIME` token: D8's literals, refused by name from T1.6c.
fn packed(literal: &str) -> bool {
    literal
        .split([' ', '/', '-', '.', ':'])
        .any(|run| run.len() >= 5)
}

/// What the rows' answers held: reads, refusals, offsets and warnings.
#[derive(Default, Debug)]
struct Held {
    read: usize,
    refused: usize,
    offset: usize,
    warned: usize,
    /// Rows the reader refused by name as packed (D8, from T1.6c).
    packed: usize,
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

/// Each `[literal, value, offset, format]` row against the reader: a D8
/// literal must be refused by name as packed, whatever the reference read
/// (T1.6c); every other must read as the reference reads it. `held` counts
/// the reference's answers, so the file's promise is checked as before, and
/// the packed refusals beside them.
fn check(reader: Reader, rows: &[Value], failures: &mut Vec<String>) -> Held {
    let mut held = Held::default();
    for row in rows {
        let literal = row[0].as_str().unwrap();
        let want: Answer = (
            row[1].as_str().map(str::to_owned),
            row[2].as_i64(),
            row[3].as_str().map(str::to_owned),
        );
        held.count(&want);
        if packed(literal) {
            match reader.read(literal) {
                Reading::Packed => held.packed += 1,
                other => failures.push(format!("{literal}: {other:?}, not refused as packed")),
            }
            continue;
        }
        let got = answer(reader, literal);
        if got != want {
            failures.push(format!("{literal}: {got:?}, the reference {want:?}"));
        }
    }
    held
}

/// T0.6's literals: the four with a run of five digits or more
/// (`230203-06-09`, `20240101/00/123456`, `00001/1/2`, which the reference
/// reads, and `10000-01-01`, which it refuses) are refused by name as packed
/// (T1.6c brief: "a D8 literal is refused with a named code and rule");
/// the other 41 read as the reference reads them (D1 to D7 and D9 unchanged).
#[test]
fn the_t06_literals_read_as_the_reference_reads_them() {
    let vectors: Value = serde_json::from_str(&here("../../testdata/t06-vectors.json")).unwrap();
    let year = vectors["meta"]["chrono"]["today_year"].as_i64().unwrap();
    let reader = Reader {
        year: Some(i16::try_from(year).unwrap()),
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
        if packed(literal) {
            match reader.read(literal) {
                Reading::Packed => held.packed += 1,
                other => failures.push(format!("{literal}: {other:?}, not refused as packed")),
            }
            continue;
        }
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
    assert_eq!(
        held.packed, 4,
        "T0.6's four literals with a run of five digits"
    );
    assert!(
        held.read > 0 && held.refused > 0 && held.warned > 0,
        "{held:?}"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// `dates.json`: every one of D8's 7,749 `packed` rows is refused by name
/// as packed, those the reference reads and those it refuses alike (T1.6c
/// brief: "remove the D8 reading path"); every `plain` row (every run four
/// digits or fewer) still reads as the reference reads it, D4's warnings
/// and D9's refusals included ("keep everything else in D1 to D9
/// unchanged").
#[test]
fn the_generated_literals_read_as_the_reference_reads_them() {
    let vectors: Value = serde_json::from_str(&here("tests/dates.json")).unwrap();
    assert_eq!(vectors["format"], "sdm-pql-dates/1");
    let reader = Reader {
        year: Some(i16::try_from(vectors["year"].as_i64().unwrap()).unwrap()),
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
    // The reference's own readings, kept as evidence of the old behaviour:
    // it read most and refused the rest.
    assert!(
        held.read >= 4_000 && held.refused >= 3_000 && held.offset >= 300,
        "{held:?}"
    );
    assert_eq!(held.packed, packed.len(), "every D8 row refused as packed");
    let plain = vectors["plain"].as_array().unwrap();
    let held = check(reader, plain, &mut failures);
    assert!(plain.len() >= 9_000, "{}", plain.len());
    assert!(
        held.read >= 1_000 && held.refused >= 1_000 && held.warned >= 1_000,
        "{held:?}"
    );
    assert_eq!(held.packed, 0, "no plain row has a run of five digits");
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
                year: Some(2026),
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
        year: Some(i16::try_from(vectors["year"].as_i64().unwrap()).unwrap()),
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

/// T1.6c brief, "a D8 literal is refused ... `date_literal_packed`", at the
/// reader: a run of five digits or more anywhere, in any of the three date
/// fields or the three time fields, is `Packed` before anything is read,
/// so it needs no current year and is never `Unplaced` (its six-digit
/// packed dates included, which D5 would place); a run of four is read as
/// before.
#[test]
fn a_run_of_five_digits_is_refused_as_packed_and_needs_no_year() {
    let no_year = Reader {
        year: None,
        pads_year: false,
    };
    let in_2026 = Reader {
        year: Some(2026),
        pads_year: true,
    };
    for literal in [
        // A packed first field of six, eight, twelve and fourteen digits.
        "230203-06-09",
        "760101-01-01",
        "20240101-01-01",
        "20240101/00/123456",
        "202401011230-1-1",
        "20240101123456-1-1",
        // A long field elsewhere: zero-padded, in the time, last.
        "00001/1/2",
        "2024-01-01 00000:00:00",
        "1/2/3 00001:0:0",
        "2024-01-01 1:2:00003",
        "1-1-20240101",
        "13/1/20240",
        // Ones the reference refuses: a year past 9999, an offset of a day.
        "10000-01-01",
        "12345-1-1",
        "20240101-01-24",
        // A run past what any field holds.
        "1.2.1234567890123456789012345",
    ] {
        assert!(packed(literal), "{literal}");
        for reader in [no_year, in_2026] {
            assert_eq!(reader.read(literal), Reading::Packed, "{literal}");
        }
    }
    // Four digits are no run of five: read, by D2 to D7, as the reference
    // reads them.
    for (literal, value) in [
        ("0001/1/2", "0001-01-02T00:00:00"),
        ("2024-01-01 0000:00:00", "2024-01-01T00:00:00"),
        ("0000-01-01", "0000-01-01T00:00:00"),
        ("2024-01-31 23:59:59", "2024-01-31T23:59:59"),
    ] {
        assert_eq!(
            answer(in_2026, literal),
            (Some(value.to_owned()), None, None),
            "{literal}"
        );
        assert_eq!(answer(no_year, literal).0.as_deref(), Some(value));
    }
    // A day-first literal still warns (D4), a two-digit year still needs
    // the current year (D5), and D9's literals are still refused.
    assert_eq!(answer(in_2026, "13/01/2024").2.as_deref(), Some("%d/%m/%Y"));
    assert_eq!(no_year.read("1/2/76"), Reading::Unplaced);
    assert_eq!(
        answer(in_2026, "1/2/76").0.as_deref(),
        Some("1976-01-02T00:00:00")
    );
    for literal in ["2023-02-29", "123-4-567", "1/2/3 24:00:00"] {
        assert_eq!(in_2026.read(literal), Reading::Refused, "{literal}");
    }
}
