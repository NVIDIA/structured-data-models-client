// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keys as pandas hashes and compares them: `pd.unique`, `pd.factorize`,
//! `Index.get_indexer`, `drop_duplicates` and `np.unique` over a column of
//! primary or foreign keys.
//!
//! pandas finds `1`, `1.0` and `True` the same key, and a `NaN` the same as
//! another `NaN`; text is compared as text.
//!
//! What the sampler reaches is reproduced: integer, float, decimal and text
//! keys, and nulls. A decimal key is the `decimal.Decimal` an Arrow
//! `decimal128` column yields, equal to an integer or a float of the same
//! value, as Python's `==` and `hash` find it. A key of any other type (a
//! timestamp, bytes) is refused with a `TypeError` rather than hashed some
//! other way.

use std::cmp::Ordering;
use std::collections::HashMap;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Decimal128Type;
use arrow_schema::DataType;
use sdm_schema::decimal;

use crate::driver::Raised;
use crate::frame::{Series, floats, ints, texts};

/// One key, as Python holds it.
#[derive(Clone, Debug, PartialEq)]
pub enum Key {
    Null,
    Int(i64),
    Float(f64),
    /// A `decimal.Decimal`: its unscaled digits and its scale, as a
    /// `decimal128` holds it.
    Decimal(i128, i8),
    Text(String),
}

/// A key's identity: what pandas' hash tables find equal is equal here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeyId {
    Null,
    Int(i128),
    Float(u64),
    Nan,
    /// A decimal no integer or float equals: its digits and scale, without
    /// trailing zeros.
    Decimal(i128, i8),
    Text(String),
}

impl Key {
    pub fn id(&self) -> KeyId {
        match self {
            Key::Null => KeyId::Null,
            Key::Int(v) => KeyId::Int(i128::from(*v)),
            Key::Float(v) => {
                if v.is_nan() {
                    KeyId::Nan
                } else if v.fract() == 0.0 && v.abs() < 1.7e38 {
                    KeyId::Int(*v as i128)
                } else {
                    KeyId::Float((*v + 0.0).to_bits())
                }
            }
            Key::Decimal(digits, scale) => decimal_id(*digits, *scale),
            Key::Text(v) => KeyId::Text(v.clone()),
        }
    }

    /// `pd.isna(key)`.
    pub fn is_na(&self) -> bool {
        matches!(self, Key::Null) || matches!(self, Key::Float(v) if v.is_nan())
    }

    /// The key as Python's `repr` writes it in a list.
    pub fn repr(&self) -> String {
        match self {
            Key::Null => "None".to_string(),
            Key::Int(v) => v.to_string(),
            Key::Float(v) => sdm_schema::py::float_repr(*v),
            Key::Decimal(digits, scale) => format!(
                "Decimal('{}')",
                sdm_schema::frame::decimal_text(*digits, *scale)
            ),
            Key::Text(v) => sdm_schema::py::str_repr(v),
        }
    }

    /// The order `np.unique` sorts keys of one kind in.
    fn sort_cmp(&self, other: &Key) -> Ordering {
        match (self, other) {
            (Key::Int(a), Key::Int(b)) => a.cmp(b),
            (Key::Text(a), Key::Text(b)) => a.chars().cmp(b.chars()),
            (Key::Decimal(..) | Key::Int(_), Key::Decimal(..) | Key::Int(_)) => {
                match (self.exact(), other.exact()) {
                    (Some(a), Some(b)) => decimal::compare(a, b),
                    _ => Ordering::Equal,
                }
            }
            // Python orders a decimal against a float exactly; `np_unique`
            // refuses decimals beside a `NaN` before it sorts.
            (Key::Decimal(digits, scale), Key::Float(f)) => {
                decimal::compare_float((*digits, *scale), *f).unwrap_or(Ordering::Equal)
            }
            (Key::Float(f), Key::Decimal(digits, scale)) => {
                decimal::compare_float((*digits, *scale), *f)
                    .map_or(Ordering::Equal, Ordering::reverse)
            }
            // Python orders an int against a float exactly too; against
            // `NaN`, which no order holds for, as floats.
            (Key::Int(i), Key::Float(f)) => decimal::compare_float((i128::from(*i), 0), *f)
                .unwrap_or_else(|| (*i as f64).total_cmp(f)),
            (Key::Float(f), Key::Int(i)) => decimal::compare_float((i128::from(*i), 0), *f)
                .map_or_else(|| f.total_cmp(&(*i as f64)), Ordering::reverse),
            (a, b) => {
                let (x, y) = (a.number(), b.number());
                match (x, y) {
                    (Some(x), Some(y)) => x.total_cmp(&y),
                    _ => Ordering::Equal,
                }
            }
        }
    }

    fn number(&self) -> Option<f64> {
        match self {
            Key::Int(v) => Some(*v as f64),
            Key::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// An integer or a decimal as unscaled digits and a scale.
    fn exact(&self) -> Option<(i128, i8)> {
        match self {
            Key::Int(v) => Some((i128::from(*v), 0)),
            Key::Decimal(digits, scale) => Some((*digits, *scale)),
            _ => None,
        }
    }
}

/// A decimal's identity: the integer it equals, else the float it equals,
/// else itself (`sdm_schema::decimal::identity`).
fn decimal_id(digits: i128, scale: i8) -> KeyId {
    match decimal::identity(digits, scale) {
        decimal::Identity::Int(v) => KeyId::Int(v),
        decimal::Identity::Float(v) => Key::Float(v).id(),
        decimal::Identity::Decimal(digits, scale) => KeyId::Decimal(digits, scale),
    }
}

fn unsupported(data_type: &DataType) -> Raised {
    Raised::new(
        "TypeError",
        format!("keys of Arrow type {data_type} are not supported by the core"),
    )
}

/// The keys a column holds: `Null` where pandas' `isna` holds.
pub fn keys_of(series: &Series) -> Result<Vec<Key>, Raised> {
    let array = series.array.as_ref();
    if let Some(values) = ints(array) {
        return Ok(values
            .into_iter()
            .map(|v| v.map_or(Key::Null, Key::Int))
            .collect());
    }
    if let Some(values) = floats(array) {
        return Ok(values
            .into_iter()
            .map(|v| v.map_or(Key::Null, Key::Float))
            .collect());
    }
    if let Some(values) = texts(array) {
        return Ok(values
            .into_iter()
            .map(|v| v.map_or(Key::Null, Key::Text))
            .collect());
    }
    if let DataType::Decimal128(_, scale) = array.data_type() {
        let values = array.as_primitive::<Decimal128Type>();
        return Ok((0..array.len())
            .map(|i| {
                if array.is_valid(i) {
                    Key::Decimal(values.value(i), *scale)
                } else {
                    Key::Null
                }
            })
            .collect());
    }
    if matches!(array.data_type(), DataType::Null) {
        return Ok(vec![Key::Null; array.len()]);
    }
    Err(unsupported(array.data_type()))
}

/// `pd.unique(keys)`: each key once, in the order first met.
pub fn unique(keys: &[Key]) -> Vec<Key> {
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    for key in keys {
        if seen.insert(key.id(), ()).is_none() {
            out.push(key.clone());
        }
    }
    out
}

/// `pd.Index(keys)`, for `get_indexer`.
pub struct Index {
    positions: HashMap<KeyId, i64>,
}

impl Index {
    /// An index of keys that are each unique.
    pub fn new(keys: &[Key]) -> Self {
        let mut positions = HashMap::with_capacity(keys.len());
        for (i, key) in keys.iter().enumerate() {
            positions.entry(key.id()).or_insert(i as i64);
        }
        Self { positions }
    }

    /// `index.get_indexer(values)`: each value's position, or -1.
    pub fn get_indexer(&self, values: &[Key]) -> Vec<i64> {
        values
            .iter()
            .map(|v| self.positions.get(&v.id()).copied().unwrap_or(-1))
            .collect()
    }

    pub fn get(&self, key: &Key) -> Option<i64> {
        self.positions.get(&key.id()).copied()
    }
}

/// `pd.factorize(keys)`: each key's code, and the uniques in the order first
/// met; a null is coded -1 and left out of the uniques.
pub fn factorize(keys: &[Key]) -> (Vec<i64>, Vec<usize>) {
    let mut codes = Vec::with_capacity(keys.len());
    let mut first: Vec<usize> = Vec::new();
    let mut seen: HashMap<KeyId, i64> = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        if key.is_na() {
            codes.push(-1);
            continue;
        }
        let next = first.len() as i64;
        let code = *seen.entry(key.id()).or_insert_with(|| {
            first.push(i);
            next
        });
        codes.push(code);
    }
    (codes, first)
}

/// `np.unique(keys, return_index=True, return_inverse=True)` of keys of one
/// kind and no nulls: the sorted keys' first positions, and each key's
/// place among them.
///
/// Numbers with a null are the `float64` numpy makes of them, the null NaN:
/// sorted with NaN last, and every NaN one value, as numpy 2 counts them
/// (`equal_nan=True`). Text with a null numpy cannot sort; it raises a
/// `TypeError` whose text turns on the order its sort compares values in,
/// so the core raises one of its own.
pub fn np_unique(keys: &[Key]) -> Result<(Vec<usize>, Vec<i64>), Raised> {
    if keys.iter().any(Key::is_na) {
        if keys.iter().any(|k| matches!(k, Key::Text(_))) {
            return Err(Raised::new(
                "TypeError",
                "np.unique of text keys with a null is not supported by the core",
            ));
        }
        if keys.iter().any(|k| matches!(k, Key::Decimal(..))) {
            return Err(Raised::new(
                "TypeError",
                "np.unique of decimal keys with a null is not supported by the core",
            ));
        }
        let values: Vec<f64> = keys
            .iter()
            .map(|k| match k {
                Key::Int(v) => *v as f64,
                Key::Float(v) => *v,
                _ => f64::NAN,
            })
            .collect();
        return Ok(np_unique_floats(&values));
    }
    let mut order: Vec<usize> = (0..keys.len()).collect();
    // A stable sort, so of equal keys the first comes first.
    order.sort_by(|&a, &b| keys[a].sort_cmp(&keys[b]).then(a.cmp(&b)));
    let mut index = Vec::new();
    let mut inverse = vec![0i64; keys.len()];
    let mut last: Option<KeyId> = None;
    for &i in &order {
        let id = keys[i].id();
        if last.as_ref() != Some(&id) {
            index.push(i);
            last = Some(id);
        }
        inverse[i] = index.len() as i64 - 1;
    }
    Ok((index, inverse))
}

/// `df.drop_duplicates()` of pairs: the positions of each pair's first row.
pub fn first_of_pairs(a: &[Key], b: &[i64]) -> Vec<usize> {
    let mut seen = HashMap::new();
    let mut out = Vec::new();
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if seen.insert((x.id(), *y), ()).is_none() {
            out.push(i);
        }
    }
    out
}

/// `series.duplicated()` of integers: whether each repeats an earlier one.
pub fn duplicated(values: &[i64]) -> Vec<bool> {
    let mut seen = HashMap::new();
    values
        .iter()
        .map(|v| seen.insert(*v, ()).is_some())
        .collect()
}

/// Keys as a Python list's `repr`, as `series.tolist()` shows them.
pub fn list_repr(keys: &[Key]) -> String {
    format!(
        "[{}]",
        keys.iter().map(Key::repr).collect::<Vec<_>>().join(", ")
    )
}

/// Every key of the array at once, for a lookup that needs them all valid.
pub fn len_of(series: &Series) -> usize {
    series.array.len()
}

/// `np.unique(values, return_index=True, return_inverse=True)` of
/// `float64` values: a stable sort with NaN last, equal values (and every
/// NaN) one.
fn np_unique_floats(values: &[f64]) -> (Vec<usize>, Vec<i64>) {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (values[a], values[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        }
        .then(a.cmp(&b))
    });
    let mut index = Vec::new();
    let mut inverse = vec![0i64; values.len()];
    let mut previous: Option<f64> = None;
    for &i in &order {
        let value = values[i];
        let same = previous.is_some_and(|p| p == value || (p.is_nan() && value.is_nan()));
        if !same {
            index.push(i);
            previous = Some(value);
        }
        inverse[i] = index.len() as i64 - 1;
    }
    (index, inverse)
}

/// A decimal's text, as Python's `str(Decimal)` writes one, as its digits
/// and scale: `1.50` is `(150, 2)`, `1E+3` is `(1000, 0)`.
pub fn parse_decimal(text: &str) -> Option<(i128, i8)> {
    let text = text.trim();
    let (mantissa, exponent) = match text.find(['e', 'E']) {
        Some(at) => (&text[..at], text[at + 1..].parse::<i32>().ok()?),
        None => (text, 0),
    };
    let (negative, unsigned) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa.strip_prefix('+').unwrap_or(mantissa)),
    };
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    if whole.is_empty() && fraction.is_empty() {
        return None;
    }
    if !whole
        .chars()
        .chain(fraction.chars())
        .all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let mut digits: i128 = format!("{whole}{fraction}").parse().ok()?;
    let mut scale = i32::try_from(fraction.len()).ok()? - exponent;
    while scale < 0 {
        digits = digits.checked_mul(10)?;
        scale += 1;
    }
    if negative {
        digits = -digits;
    }
    Some((digits, i8::try_from(scale).ok()?))
}

/// A decimal that is a whole number, as one: `Decimal('1.0')` is `1`;
/// `None` for one that is not, or does not fit an `int64`.
pub fn whole_decimal(digits: i128, scale: i8) -> Option<i64> {
    let divisor = 10_i128.checked_pow(u32::try_from(scale.max(0)).ok()?)?;
    if digits % divisor != 0 {
        return None;
    }
    i64::try_from(digits / divisor).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_with_a_null_are_unique_as_numpy_floats() {
        let keys = [Key::Int(5), Key::Null, Key::Int(5), Key::Int(7)];
        assert_eq!(np_unique(&keys).unwrap(), (vec![0, 3, 1], vec![0, 2, 0, 1]));
        // Past 2**53 two integers are one float, so one key.
        let big = [Key::Int((1 << 53) + 1), Key::Null, Key::Int(1 << 53)];
        assert_eq!(np_unique(&big).unwrap(), (vec![0, 1], vec![0, 1, 0]));
        let text = [Key::Text("b".into()), Key::Null];
        assert_eq!(np_unique(&text).unwrap_err().class, "TypeError");
    }

    #[test]
    fn one_and_one_point_zero_are_one_key() {
        let index = Index::new(&[Key::Int(1), Key::Text("a".into()), Key::Float(2.5)]);
        assert_eq!(
            index.get_indexer(&[Key::Float(1.0), Key::Text("a".into()), Key::Int(2)]),
            [0, 1, -1]
        );
    }

    #[test]
    fn decimals_are_the_numbers_they_equal() {
        // Decimal('5.00') == 5, Decimal('1.5') == 1.5, Decimal('0.1') != 0.1.
        assert_eq!(Key::Decimal(500, 2).id(), Key::Int(5).id());
        assert_eq!(Key::Decimal(15, 1).id(), Key::Float(1.5).id());
        assert_ne!(Key::Decimal(1, 1).id(), Key::Float(0.1).id());
        assert_eq!(Key::Decimal(10, 2).id(), Key::Decimal(1, 1).id());
        assert_eq!(Key::Decimal(3, -2).id(), Key::Int(300).id());
        assert_eq!(Key::Decimal(0, 3).id(), Key::Int(0).id());
        assert_eq!(Key::Decimal(1250, 2).repr(), "Decimal('12.50')");
        let index = Index::new(&[Key::Decimal(400, 2), Key::Decimal(100, 2)]);
        assert_eq!(
            index.get_indexer(&[Key::Int(1), Key::Decimal(4, 0)]),
            [1, 0]
        );
    }

    #[test]
    fn decimals_sort_by_value_whatever_their_scale() {
        let keys = [
            Key::Decimal(1200, 2),
            Key::Decimal(5, 0),
            Key::Decimal(125, 1),
            Key::Decimal(50, 1),
        ];
        assert_eq!(np_unique(&keys).unwrap(), (vec![1, 0, 2], vec![1, 0, 2, 0]));
        let with_null = [Key::Decimal(1, 0), Key::Null];
        assert_eq!(np_unique(&with_null).unwrap_err().class, "TypeError");
    }

    #[test]
    fn decimals_and_floats_sort_exactly() {
        // np.unique([0.1, Decimal('0.1'), Decimal('0.5'), 0.5]) of objects,
        // with return_index and return_inverse, from Python.
        let keys = [
            Key::Float(0.1),
            Key::Decimal(1, 1),
            Key::Decimal(5, 1),
            Key::Float(0.5),
        ];
        assert_eq!(np_unique(&keys).unwrap(), (vec![1, 0, 2], vec![1, 0, 2, 2]));
    }

    #[test]
    fn ints_and_floats_sort_exactly() {
        // np.unique of these as objects, with return_index and
        // return_inverse, from Python.
        let big = (1i64 << 53) + 1;
        let two_53 = 2f64.powi(53);
        assert_eq!(
            np_unique(&[Key::Int(big), Key::Float(two_53)]).unwrap(),
            (vec![1, 0], vec![1, 0])
        );
        assert_eq!(
            np_unique(&[Key::Float(two_53), Key::Int(big)]).unwrap(),
            (vec![0, 1], vec![0, 1])
        );
        assert_eq!(
            np_unique(&[Key::Int(big), Key::Float(two_53), Key::Int(1 << 53)]).unwrap(),
            (vec![1, 0], vec![1, 0, 0])
        );
        assert_eq!(
            np_unique(&[Key::Int(3), Key::Float(2.5), Key::Float(3.0)]).unwrap(),
            (vec![1, 0], vec![1, 0, 1])
        );
    }

    #[test]
    fn factorize_and_unique_keep_first_order() {
        let keys = [Key::Int(5), Key::Null, Key::Int(3), Key::Int(5)];
        assert_eq!(factorize(&keys), (vec![0, -1, 1, 0], vec![0, 2]));
        let (index, inverse) = np_unique(&[Key::Int(5), Key::Int(3), Key::Int(5)]).unwrap();
        assert_eq!((index, inverse), (vec![1, 0], vec![1, 0, 1]));
    }

    #[test]
    fn a_decimal_reads_as_its_digits_and_scale() {
        assert_eq!(parse_decimal("1"), Some((1, 0)));
        assert_eq!(parse_decimal("1.50"), Some((150, 2)));
        assert_eq!(parse_decimal("-2.5"), Some((-25, 1)));
        assert_eq!(parse_decimal("1E+3"), Some((1000, 0)));
        assert_eq!(parse_decimal("1.5E-2"), Some((15, 3)));
        assert_eq!(parse_decimal("NaN"), None);
        assert_eq!(whole_decimal(10, 1), Some(1));
        assert_eq!(whole_decimal(15, 1), None);
    }
}
