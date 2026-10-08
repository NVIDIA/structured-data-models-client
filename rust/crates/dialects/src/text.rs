// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What Python's built-ins make of a value, where a dialect calls them:
//! `int()`, `float()`, `str()` with `strip()` and `upper()`, and `zip`.

use crate::Error;
use crate::int::Int;
use crate::py::Py;
use crate::pyjson::Dumps;
use crate::unicode::{decimal_value, strip};

/// `json.dumps(list(values), ...)` of text.
pub fn json_text_list(values: &[String], style: Dumps) -> String {
    let list = Py::List(values.iter().cloned().map(Py::Str).collect());
    style
        .dumps(&list)
        .unwrap_or_else(|_| unreachable!("a list of text is always JSON"))
}

/// `[list(row) for row in zip(*columns)]`: as many rows as the shortest
/// column has values.
pub fn zip_rows(columns: &[Vec<Py>]) -> Vec<Py> {
    let rows = columns.iter().map(Vec::len).min().unwrap_or(0);
    (0..rows)
        .map(|i| Py::List(columns.iter().map(|c| c[i].clone()).collect()))
        .collect()
}

/// What `int(value)` and `float(value)` refuse: `TypeError` or
/// `ValueError`, which a caller may catch, or any other error, which
/// propagates.
#[derive(Debug)]
pub enum Refusal {
    /// `TypeError` or `ValueError`.
    Refused(Error),
    /// `OverflowError`, or a value this port does not read.
    Raised(Error),
}

impl From<Refusal> for Error {
    fn from(refusal: Refusal) -> Self {
        match refusal {
            Refusal::Refused(error) | Refusal::Raised(error) => error,
        }
    }
}

/// The digits of `int(text)`: whitespace around, a sign, and decimal digits
/// of any script with single underscores between them.
fn int_text(text: &str) -> Option<Int> {
    let body = strip(text);
    let (negative, body) = match body.chars().next()? {
        '-' => (true, &body[1..]),
        '+' => (false, &body[1..]),
        _ => (false, body),
    };
    let mut digits = String::new();
    let mut previous_underscore = true;
    for c in body.chars() {
        if c == '_' {
            if previous_underscore {
                return None;
            }
            previous_underscore = true;
            continue;
        }
        let value = decimal_value(c)?;
        digits.push(char::from_digit(value, 10)?);
        previous_underscore = false;
    }
    if digits.is_empty() || previous_underscore {
        return None;
    }
    Int::parse(&format!("{}{digits}", if negative { "-" } else { "" }))
}

/// `int(Decimal(text))`: the value truncated toward zero.
fn int_of_decimal(text: &str) -> Result<Int, Refusal> {
    let lower = text.to_ascii_lowercase();
    let unsigned = lower.trim_start_matches(['-', '+']);
    if unsigned.contains("nan") {
        return Err(Refusal::Refused(Error::new(
            "cannot convert NaN to integer",
        )));
    }
    if unsigned.starts_with("inf") {
        return Err(Refusal::Raised(Error::new(
            "cannot convert Infinity to integer",
        )));
    }
    let negative = lower.starts_with('-');
    let (mantissa, exponent) = match unsigned.split_once('e') {
        Some((m, e)) => (m, e.parse::<i64>().map_err(|_| bad_decimal(text))?),
        None => (unsigned, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    let point = whole.len() as i64 + exponent;
    let kept = if point <= 0 {
        "0".to_string()
    } else if point as usize >= digits.len() {
        format!("{digits}{}", "0".repeat(point as usize - digits.len()))
    } else {
        digits[..point as usize].to_string()
    };
    Int::parse(&format!("{}{kept}", if negative { "-" } else { "" }))
        .ok_or_else(|| bad_decimal(text))
}

fn bad_decimal(text: &str) -> Refusal {
    Refusal::Raised(Error::new(format!("a bad decimal {text:?}")))
}

/// `int(value)`.
pub fn py_int(value: &Py) -> Result<Int, Refusal> {
    match value {
        Py::Np(_, inner) => py_int(inner),
        Py::Int(int) => Ok(int.clone()),
        Py::Bool(b) => Ok(Int::from(i64::from(*b))),
        Py::Float(x) => {
            if x.is_nan() {
                Err(Refusal::Refused(Error::new(
                    "cannot convert float NaN to integer",
                )))
            } else if x.is_infinite() {
                Err(Refusal::Raised(Error::new(
                    "cannot convert float infinity to integer",
                )))
            } else {
                Int::parse(&format!("{:.0}", x.trunc()))
                    .ok_or_else(|| Refusal::Raised(Error::new("a float too large")))
            }
        }
        Py::Str(text) => int_text(text).ok_or_else(|| {
            Refusal::Refused(Error::new(format!(
                "invalid literal for int() with base 10: {text:?}"
            )))
        }),
        Py::Decimal(text) => int_of_decimal(text),
        other => Err(Refusal::Refused(Error::new(format!(
            "int() can't convert {other:?}"
        )))),
    }
}

/// `float(text)` for text Python reads as a float: whitespace around, a
/// sign, digits with a point and an exponent, `inf`, `infinity` or `nan`.
fn float_text(text: &str) -> Option<f64> {
    let body = strip(text);
    if body.contains('_') || body.is_empty() {
        return None;
    }
    let ascii: String = body
        .chars()
        .map(|c| match decimal_value(c) {
            Some(v) if !c.is_ascii_digit() => char::from_digit(v, 10).unwrap_or(c),
            _ => c,
        })
        .collect();
    let unsigned = ascii.trim_start_matches(['-', '+']).to_ascii_lowercase();
    if !matches!(unsigned.as_str(), "inf" | "infinity" | "nan")
        && !unsigned
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '-' | '+'))
    {
        return None;
    }
    ascii.parse().ok()
}

/// `float(value)`.
pub fn py_float(value: &Py) -> Result<f64, Refusal> {
    match value {
        Py::Np(_, inner) => py_float(inner),
        Py::Float(x) => Ok(*x),
        Py::Bool(b) => Ok(f64::from(u8::from(*b))),
        Py::Int(int) => {
            let x: f64 = int
                .to_string()
                .parse()
                .map_err(|_| Refusal::Raised(Error::new("a bad integer")))?;
            if x.is_infinite() {
                return Err(Refusal::Raised(Error::new(
                    "int too large to convert to float",
                )));
            }
            Ok(x)
        }
        Py::Str(text) => float_text(text).ok_or_else(|| {
            Refusal::Refused(Error::new(format!(
                "could not convert string to float: {text:?}"
            )))
        }),
        Py::Decimal(text) => {
            let lower = text.to_ascii_lowercase();
            if lower.contains("snan") {
                return Err(Refusal::Refused(Error::new(
                    "cannot convert signaling NaN to float",
                )));
            }
            let x: f64 = if lower.contains("nan") {
                if lower.starts_with('-') {
                    -f64::NAN
                } else {
                    f64::NAN
                }
            } else {
                lower
                    .parse()
                    .map_err(|_| Refusal::Raised(Error::new("a bad decimal")))?
            };
            Ok(x)
        }
        other => Err(Refusal::Refused(Error::new(format!(
            "float() can't convert {other:?}"
        )))),
    }
}

/// `str(value).strip().upper()`.
pub fn folded(value: &Py) -> Result<String, Error> {
    Ok(strip(&value.py_str()?).to_uppercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_as_python() {
        let int = |v: Py| py_int(&v).ok().map(|i| i.to_string());
        assert_eq!(int(Py::str(" 1_000 ")), Some("1000".into()));
        assert_eq!(int(Py::str("٤٢")), Some("42".into()));
        assert_eq!(int(Py::str("1__0")), None);
        assert_eq!(int(Py::str("many")), None);
        assert_eq!(int(Py::Float(-2.7)), Some("-2".into()));
        assert_eq!(int(Py::Float(1e20)), Some("100000000000000000000".into()));
        assert_eq!(int(Py::Decimal("1.9E+2".into())), Some("190".into()));
        assert_eq!(int(Py::Decimal("-0.5".into())), Some("0".into()));
        assert!(matches!(
            py_int(&Py::Float(f64::INFINITY)),
            Err(Refusal::Raised(_))
        ));
    }

    #[test]
    fn float_as_python() {
        assert_eq!(py_float(&Py::str(" -1.5e3 ")).ok(), Some(-1500.0));
        assert!(py_float(&Py::str("INF")).unwrap().is_infinite());
        assert!(py_float(&Py::str("1_0.5")).is_err());
        assert_eq!(
            py_float(&Py::Decimal("-0".into())).unwrap().to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(
            py_float(&Py::Int(Int::parse("9007199254740993").unwrap())).ok(),
            Some(9_007_199_254_740_992.0)
        );
    }
}
