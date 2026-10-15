// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a constant stands for: [`Literal`], the reading of a query's
//! literals, in place of the first attempt's Python value type.

use std::fmt;

use crate::dates::Stamp;
use crate::float_repr::float_repr;
use crate::pyint::PyInt;

/// The value a constant stands for. Integers keep their digits exactly;
/// a float is the number its text reads as.
#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Int(PyInt),
    Float(f64),
    Str(String),
    Time(Stamp),
    List(Vec<Literal>),
}

impl fmt::Display for Literal {
    /// The value as a message writes it: strings quoted, lists bracketed.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null => f.write_str("NULL"),
            Literal::Bool(b) => write!(f, "{b}"),
            Literal::Int(i) => write!(f, "{i}"),
            Literal::Float(x) => f.write_str(&float_repr(*x)),
            Literal::Str(s) => write!(f, "'{s}'"),
            Literal::Time(t) => write!(f, "{t}"),
            Literal::List(items) => {
                f.write_str("[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{item}")?;
                }
                f.write_str("]")
            }
        }
    }
}

/// The number a `DECIMAL` token stands for: digits, a point and an
/// exponent, with at most one sign. The grammar's token also takes `--3.5`
/// and `1e+-5`, which are not numbers and make the literal unreadable.
pub fn read_float(text: &str) -> Result<f64, String> {
    let t = text.trim_matches(crate::syntax::is_space);
    let lower = t.to_ascii_lowercase();
    let body = lower.trim_start_matches(['+', '-']);
    // Rust also reads spellings Python refuses, and the reverse; the
    // grammar's numbers are digits, a point and an exponent.
    let plain = !body.is_empty()
        && body
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-'));
    let signs = t.len() - t.trim_start_matches(['+', '-']).len();
    if !plain || signs > 1 {
        return Err(format!("{text} is not a number"));
    }
    t.parse::<f64>()
        .map_err(|_| format!("{text} is not a number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_take_one_sign() {
        assert_eq!(read_float("-1.5e3"), Ok(-1500.0));
        assert_eq!(read_float("1e-5"), Ok(1e-5));
        assert!(read_float("--3.5").is_err());
        assert!(read_float("1e+-5").is_err());
    }

    #[test]
    fn literals_write_into_messages() {
        let list = Literal::List(vec![
            Literal::Float(1.5),
            Literal::Str("a".into()),
            Literal::Null,
            Literal::Bool(true),
            Literal::Int(PyInt::parse("007").unwrap()),
        ]);
        assert_eq!(list.to_string(), "[1.5, 'a', NULL, true, 7]");
    }
}
