// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `base/mapper.py::Mapper`: `(pkey, batch)` pairs numbered in the order
//! they are first added, as today's Python numbers them.
//!
//! The arithmetic is pandas' and numpy's on the raw keys: a pair is the
//! number `num_examples * key + batch`, where `key` is the key itself for
//! numbers (`int64`, wrapping as numpy wraps, or `float64`, inexact past
//! `2**53` as numpy is) and its position among the text keys met so far for
//! text. So two pairs whose numbers collide are one pair, as they are today.
//!
//! Decimal keys are `object` arrays of `decimal.Decimal`s in numpy, so the
//! number is a `Decimal` too, rounded as the default context rounds (28
//! digits, half to even), and pairs are found by Python's `==`.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::driver::Raised;
use crate::frame::Series;
use crate::keys::{Index, Key, KeyId, keys_of, unique};

/// A numpy array of the numbers the mapper computes: `int64`, `float64`,
/// or `object` (of `Decimal`s, and what joins them).
#[derive(Clone, Debug)]
enum Numbers {
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Objects(Vec<Key>),
}

impl Numbers {
    fn len(&self) -> usize {
        match self {
            Numbers::Ints(v) => v.len(),
            Numbers::Floats(v) => v.len(),
            Numbers::Objects(v) => v.len(),
        }
    }

    fn floats(&self) -> Vec<f64> {
        match self {
            Numbers::Ints(v) => v.iter().map(|&x| x as f64).collect(),
            Numbers::Floats(v) => v.clone(),
            Numbers::Objects(_) => Vec::new(),
        }
    }

    /// The numbers as the Python objects an `object` array holds.
    fn objects(&self) -> Vec<Key> {
        match self {
            Numbers::Ints(v) => v.iter().map(|&x| Key::Int(x)).collect(),
            Numbers::Floats(v) => v.iter().map(|&x| Key::Float(x)).collect(),
            Numbers::Objects(v) => v.clone(),
        }
    }
}

/// The digits the default `decimal` context keeps.
const CONTEXT_DIGITS: u32 = 28;

/// `digits * 10**exponent` rounded to the context's digits, half to even.
fn round_to_context(digits: i128, exponent: i32) -> (i128, i32) {
    let length = digits.unsigned_abs().checked_ilog10().map_or(1, |l| l + 1);
    if length <= CONTEXT_DIGITS {
        return (digits, exponent);
    }
    let drop = length - CONTEXT_DIGITS;
    let factor = 10i128.pow(drop);
    let (mut kept, rest) = (digits / factor, (digits % factor).abs());
    let half = factor / 2;
    if rest > half || (rest == half && kept % 2 != 0) {
        kept += digits.signum();
    }
    let mut exponent = exponent + drop as i32;
    if kept.unsigned_abs() == 10u128.pow(CONTEXT_DIGITS) {
        kept /= 10;
        exponent += 1;
    }
    (kept, exponent)
}

/// `n * key + batch` of a `Decimal` key, as `Decimal` arithmetic gives it.
fn decimal_pair(n: i64, digits: i128, scale: i8, batch: i64) -> Option<Key> {
    let (product, exponent) =
        round_to_context(digits.checked_mul(i128::from(n))?, -i32::from(scale));
    // The sum is exact at the smaller exponent, then rounded.
    let (left, right, exponent) = if exponent <= 0 {
        let lift = 10i128.checked_pow(exponent.unsigned_abs())?;
        (product, i128::from(batch).checked_mul(lift)?, exponent)
    } else {
        let lift = 10i128.checked_pow(exponent.unsigned_abs())?;
        (product.checked_mul(lift)?, i128::from(batch), 0)
    };
    let (sum, exponent) = round_to_context(left.checked_add(right)?, exponent);
    Some(Key::Decimal(sum, i8::try_from(-exponent).ok()?))
}

/// An object's identity in a `pd.Index`, as a number's.
fn object_id(key: &Key) -> NumberId {
    NumberId::Object(key.id())
}

/// The identity a number of an `object` array finds in an index of
/// `int64` or `float64` numbers: an integer or a float equal to it.
fn numeric_id(key: &Key) -> Option<NumberId> {
    match key.id() {
        KeyId::Int(v) => i64::try_from(v).ok().map(NumberId::Int),
        KeyId::Float(bits) => Some(NumberId::Float(bits)),
        KeyId::Nan => Some(NumberId::Nan),
        _ => None,
    }
}

/// A number's identity in a `pd.Index`: equal numbers are one, an integral
/// float the integer it equals.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NumberId {
    Int(i64),
    Float(u64),
    Nan,
    /// A number of an `object` index, by Python's `==`.
    Object(KeyId),
}

fn int_id(value: i64) -> NumberId {
    NumberId::Int(value)
}

fn float_id(value: f64) -> NumberId {
    if value.is_nan() {
        NumberId::Nan
    } else if value.fract() == 0.0 && value >= -(2f64.powi(63)) && value < 2f64.powi(63) {
        NumberId::Int(value as i64)
    } else {
        NumberId::Float((value + 0.0).to_bits())
    }
}

/// `pd.Index(pd.unique(numbers))`.
struct NumberIndex {
    positions: HashMap<NumberId, i64>,
    values: Numbers,
}

impl NumberIndex {
    fn new(numbers: Numbers) -> Self {
        let mut positions = HashMap::new();
        let values = match numbers {
            Numbers::Ints(values) => {
                let mut kept = Vec::new();
                for v in values {
                    if let Entry::Vacant(slot) = positions.entry(int_id(v)) {
                        slot.insert(kept.len() as i64);
                        kept.push(v);
                    }
                }
                Numbers::Ints(kept)
            }
            Numbers::Floats(values) => {
                let mut kept = Vec::new();
                for v in values {
                    if let Entry::Vacant(slot) = positions.entry(float_id(v)) {
                        slot.insert(kept.len() as i64);
                        kept.push(v);
                    }
                }
                Numbers::Floats(kept)
            }
            Numbers::Objects(values) => {
                let mut kept = Vec::new();
                for v in values {
                    if let Entry::Vacant(slot) = positions.entry(object_id(&v)) {
                        slot.insert(kept.len() as i64);
                        kept.push(v);
                    }
                }
                Numbers::Objects(kept)
            }
        };
        Self { positions, values }
    }

    fn get_indexer(&self, numbers: &Numbers) -> Vec<i64> {
        let ids: Vec<Option<NumberId>> = match (&self.values, numbers) {
            (Numbers::Objects(_), numbers) => numbers
                .objects()
                .iter()
                .map(|k| Some(object_id(k)))
                .collect(),
            (_, Numbers::Objects(keys)) => keys.iter().map(numeric_id).collect(),
            (_, Numbers::Ints(v)) => v.iter().map(|&x| Some(int_id(x))).collect(),
            (_, Numbers::Floats(v)) => v.iter().map(|&x| Some(float_id(x))).collect(),
        };
        ids.iter()
            .map(|id| {
                id.as_ref()
                    .and_then(|id| self.positions.get(id).copied())
                    .unwrap_or(-1)
            })
            .collect()
    }
}

/// Maps `(pkey, batch)` pairs to contiguous node ids.
pub struct Mapper {
    num_examples: i64,
    /// The text keys met so far, once the keys are text.
    pkey: Option<(Vec<Key>, Index)>,
    indices: Vec<Numbers>,
    index: Option<NumberIndex>,
}

impl Mapper {
    pub fn new(num_examples: usize) -> Self {
        Self {
            num_examples: num_examples as i64,
            pkey: None,
            indices: Vec::new(),
            index: None,
        }
    }

    /// The numbers of the pairs: `num_examples * index + batch`.
    fn numbers(&self, pkey: &Series, batch: &[i64]) -> Result<Numbers, Raised> {
        let n = self.num_examples;
        let index: Numbers = match &self.pkey {
            Some((_, categories)) => Numbers::Ints(categories.get_indexer(&keys_of(pkey)?)),
            None => {
                let keys = keys_of(pkey)?;
                let refused = || {
                    Raised::new(
                        "TypeError",
                        "keys that are neither numbers without nulls nor text are not \
                         supported by the core's mapper",
                    )
                };
                if pkey.is_integer_dtype() && keys.iter().any(Key::is_na) {
                    // `to_numpy()` of integers with a null, Arrow's or
                    // pandas' `Int64`, is `float64`, the null NaN.
                    Numbers::Floats(
                        keys.iter()
                            .map(|k| match k {
                                Key::Int(v) => *v as f64,
                                _ => f64::NAN,
                            })
                            .collect(),
                    )
                } else if pkey.is_integer_dtype() {
                    // `to_numpy()`, then `astype('int64')`, which wraps.
                    Numbers::Ints(
                        keys.iter()
                            .map(|k| match k {
                                Key::Int(v) => Ok(*v),
                                _ => Err(refused()),
                            })
                            .collect::<Result<_, _>>()?,
                    )
                } else if keys.iter().all(|k| matches!(k, Key::Decimal(..))) && !keys.is_empty() {
                    // `to_numpy()` of decimals: `Decimal` objects.
                    Numbers::Objects(keys)
                } else if crate::frame::floats(pkey.array.as_ref()).is_some() {
                    // A missing float is NaN in numpy.
                    Numbers::Floats(
                        keys.iter()
                            .map(|k| match k {
                                Key::Float(v) => *v,
                                _ => f64::NAN,
                            })
                            .collect(),
                    )
                } else {
                    return Err(refused());
                }
            }
        };
        if index.len() != batch.len() {
            return Err(Raised::new(
                "ValueError",
                format!(
                    "operands could not be broadcast together with shapes ({},) ({},) ",
                    index.len(),
                    batch.len()
                ),
            ));
        }
        Ok(match index {
            Numbers::Ints(values) => Numbers::Ints(
                values
                    .iter()
                    .zip(batch)
                    .map(|(&v, &b)| n.wrapping_mul(v).wrapping_add(b))
                    .collect(),
            ),
            Numbers::Floats(values) => Numbers::Floats(
                values
                    .iter()
                    .zip(batch)
                    .map(|(&v, &b)| n as f64 * v + b as f64)
                    .collect(),
            ),
            Numbers::Objects(values) => Numbers::Objects(
                values
                    .iter()
                    .zip(batch)
                    .map(|(v, &b)| match v {
                        Key::Decimal(digits, scale) => decimal_pair(n, *digits, *scale, b)
                            .ok_or_else(|| {
                                Raised::new(
                                    "TypeError",
                                    "a decimal key past 38 digits is not supported by the \
                                     core's mapper",
                                )
                            }),
                        _ => Err(Raised::internal("an object number that is not a decimal")),
                    })
                    .collect::<Result<_, _>>()?,
            ),
        })
    }

    /// `add(pkey, batch)`.
    pub fn add(&mut self, pkey: &Series, batch: &[i64]) -> Result<(), Raised> {
        if let Some((categories, _)) = &self.pkey {
            let mut all = categories.clone();
            all.extend(keys_of(pkey)?);
            let category = unique(&all);
            let index = Index::new(&category);
            self.pkey = Some((category, index));
        } else if pkey.is_string_dtype() {
            let category = unique(&keys_of(pkey)?);
            let index = Index::new(&category);
            self.pkey = Some((category, index));
        }
        let numbers = self.numbers(pkey, batch)?;
        self.indices.push(numbers);
        Ok(())
    }

    /// `get(pkey, batch)`: each pair's node id, or -1.
    pub fn get(&mut self, pkey: &Series, batch: &[i64]) -> Result<Vec<i64>, Raised> {
        if self.index.is_none() && self.indices.is_empty() {
            return Ok(vec![-1; pkey.len()]);
        }
        if let Some(index) = self.index.take() {
            if self.indices.is_empty() {
                self.index = Some(index);
            } else {
                self.indices.insert(0, index.values);
            }
        }
        if self.index.is_none() {
            let all = concatenate(&self.indices);
            self.index = Some(NumberIndex::new(all));
            self.indices.clear();
        }
        let numbers = self.numbers(pkey, batch)?;
        Ok(self
            .index
            .as_ref()
            .map(|index| index.get_indexer(&numbers))
            .unwrap_or_default())
    }
}

/// `np.concatenate(arrays)`: `object` if any is, else `float64` if any is.
fn concatenate(arrays: &[Numbers]) -> Numbers {
    if arrays.iter().any(|a| matches!(a, Numbers::Objects(_))) {
        return Numbers::Objects(arrays.iter().flat_map(Numbers::objects).collect());
    }
    if arrays.iter().any(|a| matches!(a, Numbers::Floats(_))) {
        Numbers::Floats(arrays.iter().flat_map(Numbers::floats).collect())
    } else {
        Numbers::Ints(
            arrays
                .iter()
                .flat_map(|a| match a {
                    Numbers::Ints(v) => v.clone(),
                    Numbers::Floats(_) | Numbers::Objects(_) => Vec::new(),
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int64Array, StringArray};

    use super::*;
    use crate::frame::Dtype;

    fn ints(values: &[i64]) -> Series {
        Series::new(
            Dtype::Numpy("int64".into()),
            Arc::new(Int64Array::from(values.to_vec())),
        )
    }

    #[test]
    fn pairs_are_numbered_in_the_order_added() {
        let mut mapper = Mapper::new(3);
        mapper.add(&ints(&[10, 20]), &[0, 1]).unwrap();
        mapper.add(&ints(&[10, 30]), &[2, 1]).unwrap();
        assert_eq!(
            mapper.get(&ints(&[30, 10, 10, 99]), &[1, 0, 1, 0]).unwrap(),
            [3, 0, -1, -1]
        );
        // A later add puts the built index first, as Python does.
        mapper.add(&ints(&[99]), &[0]).unwrap();
        assert_eq!(mapper.get(&ints(&[99, 20]), &[0, 1]).unwrap(), [4, 1]);
    }

    #[test]
    fn integer_keys_with_a_null_are_numbered_as_floats() {
        let keys = |v: Vec<Option<i64>>| Series::new(Dtype::Arrow, Arc::new(Int64Array::from(v)));
        let mut mapper = Mapper::new(2);
        mapper.add(&keys(vec![Some(4), None]), &[0, 1]).unwrap();
        // A null is NaN, `2 * NaN + batch` NaN whatever the batch, and pandas
        // finds NaN where it holds one.
        assert_eq!(
            mapper
                .get(&keys(vec![None, Some(4), Some(4), None]), &[0, 0, 1, 1])
                .unwrap(),
            [1, 0, -1, 1]
        );
    }

    #[test]
    fn decimal_pairs_are_decimals_rounded_to_the_context() {
        // 33 * Decimal('3.00') + 1 is Decimal('100.00').
        assert_eq!(decimal_pair(33, 300, 2, 1), Some(Key::Decimal(10_000, 2)));
        // Past 28 digits the product rounds half to even, then the sum does:
        // 33 * Decimal('123456789012345678901234567.5') + 1 and
        // 33 * Decimal('9999999999999999999999999999') + 5, as Python gives
        // them.
        assert_eq!(
            decimal_pair(33, 1_234_567_890_123_456_789_012_345_675, 1, 1),
            Some(Key::Decimal(4_074_074_037_407_407_403_740_740_729, 0))
        );
        assert_eq!(
            decimal_pair(33, 9_999_999_999_999_999_999_999_999_999, 0, 5),
            Some(Key::Decimal(3_300_000_000_000_000_000_000_000_000, -2))
        );
    }

    #[test]
    fn decimal_keys_are_numbered_by_value() {
        use arrow_array::Decimal128Array;

        let keys = |v: &[i128]| {
            Series::new(
                Dtype::Arrow,
                Arc::new(
                    Decimal128Array::from(v.to_vec())
                        .with_precision_and_scale(12, 2)
                        .unwrap(),
                ),
            )
        };
        let mut mapper = Mapper::new(2);
        mapper.add(&keys(&[300, 100]), &[0, 1]).unwrap();
        assert_eq!(
            mapper.get(&keys(&[100, 300, 300]), &[1, 0, 1]).unwrap(),
            [1, 0, -1]
        );
    }

    #[test]
    fn text_keys_are_numbered_by_their_place() {
        let mut mapper = Mapper::new(2);
        let text = |v: &[&str]| Series::new(Dtype::Arrow, Arc::new(StringArray::from(v.to_vec())));
        mapper.add(&text(&["b", "a"]), &[0, 0]).unwrap();
        assert_eq!(mapper.get(&text(&["a", "c"]), &[0, 0]).unwrap(), [1, -1]);
    }
}
