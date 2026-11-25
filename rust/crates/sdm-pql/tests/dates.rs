// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Date parsing semantics, explicit calendar bounds and retained cross-platform
//! padded-year interoperability inputs.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the vector files"
)]

use std::path::Path;

use sdm_pql::dates::{Reader, Reading};

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
/// `TIME` token: D8's literals, refused by name.
fn packed(literal: &str) -> bool {
    literal
        .split([' ', '/', '-', '.', ':'])
        .any(|run| run.len() >= 5)
}

#[test]
fn short_years_ordering_and_warnings_have_explicit_meanings() {
    let reader = Reader {
        year: Some(2026),
        pads_year: true,
    };
    for (literal, datetime, warning) in [
        ("01/02/50", "2050-01-02T00:00:00", None),
        ("01/02/75", "2075-01-02T00:00:00", None),
        ("01/02/76", "1976-01-02T00:00:00", None),
        ("02/01/2024", "2024-02-01T00:00:00", None),
        ("15/03/2024", "2024-03-15T00:00:00", Some("%d/%m/%Y")),
        ("50/1/2 3:4:5", "2050-01-02T03:04:05", None),
        ("2038-01-20", "2038-01-20T00:00:00", None),
        ("9999-12-31", "9999-12-31T00:00:00", None),
    ] {
        assert_eq!(
            answer(reader, literal),
            (Some(datetime.into()), None, warning.map(str::to_owned)),
            "{literal}"
        );
    }
    let unplaced = Reader {
        year: None,
        pads_year: true,
    };
    assert!(matches!(unplaced.read("01/02/50"), Reading::Unplaced));
    for literal in [
        "230203-06-09",
        "20240101/00/123456",
        "00001/1/2",
        "10000-01-01",
    ] {
        assert!(
            matches!(unplaced.read(literal), Reading::Packed),
            "{literal}"
        );
    }
    assert!(matches!(reader.read("2024-01/02"), Reading::Refused));
}

/// Calendar meanings and refusals are independent of separator spelling.
#[test]
fn calendar_dates_and_time_bounds_have_explicit_meanings() {
    for pads_year in [false, true] {
        let reader = Reader {
            year: Some(2026),
            pads_year,
        };
        for separator in ["/", "-", "."] {
            for (fields, time, expected) in [
                (["2024", "02", "29"], "", Some("2024-02-29T00:00:00")),
                (["2023", "02", "29"], "", None),
                (["2024", "04", "31"], "", None),
                (["2024", "01", "00"], "", None),
                (["2024", "13", "01"], "", None),
                (
                    ["2024", "01", "02"],
                    " 23:59:59",
                    Some("2024-01-02T23:59:59"),
                ),
                (["2024", "01", "02"], " 24:00:00", None),
                (["2024", "01", "02"], " 23:60:00", None),
                (["2024", "01", "02"], " 23:59:60", None),
                (["0000", "01", "01"], "", Some("0000-01-01T00:00:00")),
                (["9999", "12", "31"], "", Some("9999-12-31T00:00:00")),
            ] {
                let literal = fields.join(separator) + time;
                let got = answer(reader, &literal);
                assert_eq!(got, (expected.map(str::to_owned), None, None), "{literal}");
            }
        }
    }
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

#[test]
fn year_zero_requires_leading_year_and_short_time_fields() {
    for pads_year in [false, true] {
        let reader = Reader {
            year: Some(2026),
            pads_year,
        };
        assert_eq!(
            answer(reader, "0000-01-01 00:00:00"),
            (Some("0000-01-01T00:00:00".into()), None, None)
        );
        for separator in ["/", "-", "."] {
            assert!(matches!(
                reader.read(&["1", "2", "0000"].join(separator)),
                Reading::Refused
            ));
        }
        for width in [3, 4] {
            for position in 0..3 {
                let mut fields = ["00".to_owned(), "00".to_owned(), "00".to_owned()];
                fields[position] = "0".repeat(width);
                let literal = format!("0000-01-01 {}", fields.join(":"));
                assert!(
                    matches!(reader.read(&literal), Reading::Refused),
                    "{literal}"
                );
            }
        }
    }
}

/// An unusual middle field must not produce an unsupported warning format.
#[test]
fn middle_field_year_spellings_do_not_invent_warning_formats() {
    for pads_year in [false, true] {
        let reader = Reader {
            year: Some(2026),
            pads_year,
        };
        for separator in ["/", "-", "."] {
            for (fields, expected) in [
                (["13", "0001", "1"], "0001-01-13T00:00:00"),
                (["13", "0001", "2"], "0002-01-13T00:00:00"),
                (["02", "015", "75"], "0075-02-15T00:00:00"),
            ] {
                let literal = fields.join(separator);
                assert_eq!(
                    answer(reader, &literal),
                    (Some(expected.into()), None, None),
                    "{literal}"
                );
            }
            assert!(matches!(
                reader.read(&["1", "2024", "5"].join(separator)),
                Reading::Refused
            ));
        }
    }
}

/// brief, "a D8 literal is refused... `date_literal_packed`", at the
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
