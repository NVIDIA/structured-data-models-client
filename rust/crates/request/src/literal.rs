// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `decompose._as_literal`: a ranking's candidate as the query language
//! writes it, or refused where it cannot name a row.

use sdm_schema::pyval::PyVal;

use crate::error::{R, invalid};
use crate::num::Dec;
use crate::pyops::{decimal_value, item, py_repr, py_str};
use crate::series::{Dtype, Series};
use crate::wire::{decimal_dtype, is_missing};

/// `_decimal_as_typed(value, dtype)`: the Python value a decimal candidate
/// is written as, by how its column travels.
enum Typed {
    Int(Dec),
    Float(f64),
    Text(String),
}

fn decimal_as_typed(value: &Dec, original: &PyVal, dtype: Option<&str>) -> R<Typed> {
    let dtype = match dtype {
        Some(d) => d.to_string(),
        None => decimal_dtype(
            &Series::new(None, Dtype::Object, vec![original.clone()]),
            None,
        ),
    };
    Ok(match dtype.as_str() {
        "int64" => Typed::Int(crate::pyops::py_int(original)?),
        "float64" => Typed::Float(value.to_f64()),
        _ => Typed::Text(value.format_f()),
    })
}

/// `_as_literal(value, dtype)`.
pub fn as_literal(value: &PyVal, dtype: Option<&str>) -> R<String> {
    let value = item(value);
    if is_missing(&value) {
        return Err(invalid(
            "a candidate names a row, and a missing value names none",
        ));
    }
    if matches!(value, PyVal::Bool(_)) {
        return Err(invalid(
            "a candidate names a row, and True and False do not",
        ));
    }
    let mut value = value;
    if let Some(dec) = decimal_value(&value) {
        let written = decimal_as_typed(&dec, &value, dtype)?;
        let differs = match &written {
            Typed::Int(n) => !n.equals(&dec),
            Typed::Float(x) => !x.is_finite() || !Dec::exact_f64(*x).equals(&dec),
            Typed::Text(_) => false,
        };
        if differs {
            let shown = match &written {
                Typed::Int(n) => n.format_f(),
                Typed::Float(x) => crate::pyops::float_text(*x),
                Typed::Text(t) => t.clone(),
            };
            return Err(invalid(format!(
                "the candidate {} names no row of a column of {}: written as one it becomes \
                 {shown}, which is a different row",
                dec.to_sci(),
                dtype.unwrap_or("numbers")
            )));
        }
        value = match written {
            Typed::Int(n) => crate::wire::dec_to_pyval(&n),
            Typed::Float(x) => PyVal::Float(x),
            Typed::Text(t) => PyVal::Str(t),
        };
    }
    if dtype == Some("string") {
        match &value {
            PyVal::Float(x) if !x.is_finite() => {
                return Err(invalid(format!(
                    "{} cannot name a row to rank",
                    py_repr(&value)
                )));
            }
            PyVal::Float(x) => value = PyVal::Str(Dec::from_f64(*x).format_f()),
            PyVal::Int(_) | PyVal::BigInt(_) => value = PyVal::Str(py_str(&value)),
            _ => {}
        }
    }
    match &value {
        PyVal::Int(_) | PyVal::BigInt(_) => return Ok(py_str(&value)),
        PyVal::Float(x) => {
            if !x.is_finite() || x.fract() != 0.0 {
                return Err(invalid(format!(
                    "{} cannot name a row to rank",
                    py_repr(&value)
                )));
            }
            return Ok(Dec::of_whole_f64(*x).map_or_else(String::new, |d| d.format_f()));
        }
        _ => {}
    }
    let text = py_str(&value);
    if text.contains(['"', '\'', '\\', ',', '(', ')']) || text.trim().is_empty() {
        return Err(invalid(format!(
            "the candidate {} cannot be written into a query: a key holding a quote, a comma, a \
             bracket or nothing at all would read as part of the question rather than as the row \
             it names",
            py_repr(&value)
        )));
    }
    Ok(format!("\"{text}\""))
}
