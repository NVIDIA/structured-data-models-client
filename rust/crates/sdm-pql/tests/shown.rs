// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-compatible diagnostic rendering of explicit typed values.

use sdm_pql::float_repr::{Shown, text_repr};

#[test]
fn typed_values_have_explicit_repr_and_text() {
    for (value, repr, text) in [
        (
            Shown::DateTime {
                year: 2024,
                month: 1,
                day: 2,
                hour: 3,
                minute: 4,
                second: 0,
                microsecond: 42,
                offset: Some(-19800),
            },
            "datetime.datetime(2024, 1, 2, 3, 4, 0, 42, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=66600)))",
            "2024-01-02 03:04:00.000042-05:30",
        ),
        (Shown::None, "None", "None"),
        (Shown::Bool(true), "True", "True"),
        (Shown::Bool(false), "False", "False"),
        (
            Shown::Int("123456789012345678901234567890".into()),
            "123456789012345678901234567890",
            "123456789012345678901234567890",
        ),
        (Shown::Float(-0.0), "-0.0", "-0.0"),
        (Shown::Float(1e20), "1e+20", "1e+20"),
        (Shown::Float(1e-7), "1e-07", "1e-07"),
        (Shown::Float(f64::NAN), "nan", "nan"),
        (Shown::Float(f64::INFINITY), "inf", "inf"),
        (Shown::Float(f64::NEG_INFINITY), "-inf", "-inf"),
        (Shown::Text("hello".into()), "'hello'", "hello"),
        (Shown::List(vec![]), "[]", "[]"),
        (Shown::Dict(vec![]), "{}", "{}"),
        (
            Shown::Dict(vec![(
                Shown::Text("x".into()),
                Shown::List(vec![Shown::None, Shown::Bool(true)]),
            )]),
            "{'x': [None, True]}",
            "{'x': [None, True]}",
        ),
        (
            Shown::Date {
                year: 2024,
                month: 2,
                day: 29,
            },
            "datetime.date(2024, 2, 29)",
            "2024-02-29",
        ),
        (
            Shown::DateTime {
                year: 2024,
                month: 1,
                day: 2,
                hour: 3,
                minute: 4,
                second: 5,
                microsecond: 123456,
                offset: None,
            },
            "datetime.datetime(2024, 1, 2, 3, 4, 5, 123456)",
            "2024-01-02 03:04:05.123456",
        ),
        (
            Shown::DateTime {
                year: 2024,
                month: 1,
                day: 2,
                hour: 3,
                minute: 4,
                second: 5,
                microsecond: 0,
                offset: Some(19800),
            },
            "datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(seconds=19800)))",
            "2024-01-02 03:04:05+05:30",
        ),
        (
            Shown::DateTime {
                year: 2024,
                month: 1,
                day: 2,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
                offset: Some(0),
            },
            "datetime.datetime(2024, 1, 2, 0, 0, tzinfo=datetime.timezone.utc)",
            "2024-01-02 00:00:00+00:00",
        ),
    ] {
        assert_eq!(value.repr(), repr);
        assert_eq!(value.text(), text);
    }
}

/// Python's `repr` of a text: its quotes, escapes, and characters it
/// does not print.
#[test]
fn a_text_is_quoted_as_python_quotes_it() {
    assert_eq!(text_repr("a"), "'a'");
    assert_eq!(text_repr("it's"), "\"it's\"");
    assert_eq!(text_repr("it's \"x\""), "'it\\'s \"x\"'");
    assert_eq!(text_repr("a\\b\tc\n"), "'a\\\\b\\tc\\n'");
    assert_eq!(text_repr("é"), "'é'");
    assert_eq!(text_repr("\u{85}"), "'\\x85'");
    assert_eq!(text_repr("\u{2028}"), "'\\u2028'");
    assert_eq!(text_repr("\u{E000}"), "'\\ue000'");
    assert_eq!(text_repr("$$"), "'$$'");
    assert_eq!(text_repr("'"), "\"'\"");
}

/// A collection inside itself, as Python writes it.
#[test]
fn cycles_are_written_as_python_writes_them() {
    let text = |t: &str| Shown::Text(t.into());
    assert_eq!(Shown::List(vec![Shown::Recursive(true)]).repr(), "[[...]]");
    assert_eq!(
        Shown::Dict(vec![(text("k"), Shown::Recursive(false))]).repr(),
        "{'k': {...}}"
    );
    let big = "1".repeat(50);
    assert_eq!(Shown::Int(big.clone()).text(), big);
}

/// A text is written up to its bound exactly: one character over, it is
/// refused, and nothing past the bound is built.
#[test]
fn a_text_is_written_up_to_its_bound_exactly() {
    let value = Shown::List(vec![
        Shown::Text("é".repeat(3)),
        Shown::Text("\0".into()),
        Shown::Date {
            year: 2024,
            month: 1,
            day: 1,
        },
    ]);
    let repr = value.repr();
    assert_eq!(repr, "['ééé', '\\x00', datetime.date(2024, 1, 1)]");
    let length = repr.chars().count();
    assert_eq!(value.repr_within(length).as_deref(), Some(repr.as_str()));
    assert_eq!(value.repr_within(length - 1), None);
    let text = Shown::Text("é".repeat(5));
    assert_eq!(text.text_within(5).as_deref(), Some("ééééé"));
    assert_eq!(text.text_within(4), None);
}

/// Python 3.12 escapes an unassigned code point, a format character and a
/// soft hyphen in a `repr`.
#[test]
fn python_s_printability_is_its_own_table() {
    assert_eq!(text_repr("\u{378}"), "'\\u0378'");
    assert_eq!(text_repr("\u{e0001}"), "'\\U000e0001'");
    assert_eq!(text_repr("\u{ad}"), "'\\xad'");
    assert_eq!(text_repr("\u{2066}"), "'\\u2066'");
    assert_eq!(text_repr("\u{1F600}"), "'\u{1F600}'");
}
