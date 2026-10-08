// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The random generators through Python (`tests/rust_core/sampling/CONTRACT.md`,
//! Random numbers): numpy's legacy `RandomState` and C++'s `std::mt19937`.
//! Test-only until Phase 9 switches Python over to the core.

use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyInt, PyList, PyTuple};
use sdm_random::{Mt19937, RandomState};

use crate::caught;

/// A Python `int` argument in `T`'s range: a `TypeError` for anything but an
/// integer, as Python's own functions raise, and a `ValueError` saying
/// `out_of_range` for an integer outside it, where pyo3 would raise
/// `OverflowError`.
pub(crate) fn int_arg<T: TryFrom<i128>>(
    value: &Bound<'_, PyAny>,
    out_of_range: &str,
) -> PyResult<T> {
    let wide: i128 = match value.extract() {
        Ok(wide) => wide,
        Err(error) if error.is_instance_of::<PyOverflowError>(value.py()) => {
            return Err(PyValueError::new_err(out_of_range.to_string()));
        }
        Err(error) => return Err(error),
    };
    T::try_from(wide).map_err(|_| PyValueError::new_err(out_of_range.to_string()))
}

/// A size numpy takes as an array's length: a `ValueError` for a negative
/// one and for one past any array, as numpy's.
fn size_arg(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    let size: i128 = int_arg(value, "Maximum allowed dimension exceeded")?;
    if size < 0 {
        return Err(PyValueError::new_err("negative dimensions are not allowed"));
    }
    usize::try_from(size).map_err(|_| PyValueError::new_err("Maximum allowed dimension exceeded"))
}

/// A Python integer as its sign (`true` when negative) and the 32-bit words
/// of its magnitude, least significant first, with no high zero words.
pub(crate) fn int_words(value: &Bound<'_, PyAny>) -> PyResult<(bool, Vec<u32>)> {
    if !value.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "expected an integer, not {}",
            value.get_type().name()?
        )));
    }
    let negative = value.lt(0)?;
    let magnitude = value.call_method0("__abs__")?;
    let bits: usize = magnitude.call_method0("bit_length")?.extract()?;
    let bytes: Vec<u8> = magnitude
        .call_method1("to_bytes", (bits.div_ceil(8), "little"))?
        .extract()?;
    let words = bytes
        .chunks(4)
        .map(|chunk| {
            chunk
                .iter()
                .rev()
                .fold(0u32, |word, &byte| (word << 8) | u32::from(byte))
        })
        .collect();
    Ok((negative, words))
}

/// A Python integer that must fit 32 bits unsigned, as numpy seeds by.
fn seed_word(value: &Bound<'_, PyAny>) -> PyResult<u32> {
    let (negative, words) = int_words(value)?;
    match (negative, words.as_slice()) {
        (false, []) => Ok(0),
        (false, [word]) => Ok(*word),
        _ => Err(PyValueError::new_err(
            "Seed must be between 0 and 2**32 - 1",
        )),
    }
}

/// numpy's legacy `RandomState`.
#[pyclass(name = "RandomState", module = "sdm_core._native")]
pub(crate) struct PyRandomState {
    inner: RandomState,
}

/// What numpy raises: a `MemoryError` for an array no allocation can hold,
/// and otherwise a `ValueError`.
fn refused(error: sdm_random::RandomError) -> PyErr {
    match error.code {
        "MemoryError" => PyMemoryError::new_err(error.message),
        _ => PyValueError::new_err(error.message),
    }
}

#[pymethods]
impl PyRandomState {
    /// `RandomState(seed)`: an `int` from 0 to `2**32 - 1`, or a non-empty
    /// list of such ints, as `RandomState(np.array(seed, dtype=np.uint32))`
    /// takes it: by number when it holds one, by `init_by_array` otherwise.
    #[new]
    fn new(seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        caught(|| {
            if seed.is_instance_of::<PyBool>() || seed.is_instance_of::<PyInt>() {
                return Ok(Self {
                    inner: RandomState::new(seed_word(seed)?),
                });
            }
            if seed.is_instance_of::<PyList>() || seed.is_instance_of::<PyTuple>() {
                let words = seed
                    .try_iter()?
                    .map(|item| seed_word(&item?))
                    .collect::<PyResult<Vec<u32>>>()?;
                // numpy squeezes a one-word array to a number, and seeds by it.
                if let [word] = words.as_slice() {
                    return Ok(Self {
                        inner: RandomState::new(*word),
                    });
                }
                return RandomState::from_words(&words)
                    .map(|inner| Self { inner })
                    .map_err(refused);
            }
            Err(PyTypeError::new_err(format!(
                "RandomState takes an int or a list of ints, not {}",
                seed.get_type().name()?
            )))
        })
    }

    /// `randint(0, 2**32, dtype=np.uint32)`.
    fn next_u32(&mut self) -> PyResult<u32> {
        caught(|| Ok(self.inner.next_u32()))
    }

    /// `randint(0, 2**64, dtype=np.uint64)`.
    fn next_u64(&mut self) -> PyResult<u64> {
        caught(|| Ok(self.inner.next_u64()))
    }

    /// `randint(0, max + 1, dtype=np.int64)`, the draw `shuffle` makes, for
    /// `0 <= max < 2**63 - 1`; numpy's `ValueError` outside that.
    fn interval(&mut self, max: &Bound<'_, PyAny>) -> PyResult<u64> {
        caught(|| {
            let max: i128 = int_arg(max, "high is out of bounds for int64")?;
            if max < 0 {
                return Err(PyValueError::new_err("low >= high"));
            }
            if max >= i128::from(i64::MAX) {
                return Err(PyValueError::new_err("high is out of bounds for int64"));
            }
            Ok(self.inner.interval(max as u64))
        })
    }

    /// `randint(low, high, size=size, dtype=np.int64)`.
    fn randint(
        &mut self,
        low: &Bound<'_, PyAny>,
        high: &Bound<'_, PyAny>,
        size: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<i64>> {
        caught(|| {
            let low = int_arg(low, "low is out of bounds for int64")?;
            let high = int_arg(high, "high is out of bounds for int64")?;
            let size = size_arg(size)?;
            self.inner.randint(low, high, size).map_err(refused)
        })
    }

    /// `values`, a list of `int64` values, in the order `shuffle` leaves
    /// them.
    fn shuffle(&mut self, values: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
        caught(|| {
            if values.is_instance_of::<pyo3::types::PyString>() {
                return Err(PyTypeError::new_err(
                    "shuffle takes a list of ints, not str",
                ));
            }
            let mut values = values
                .try_iter()?
                .map(|value| int_arg(&value?, "Python int too large to convert to int64"))
                .collect::<PyResult<Vec<i64>>>()?;
            self.inner.shuffle(&mut values);
            Ok(values)
        })
    }

    /// `permutation(n)`: empty for a negative `n`, as `np.arange(n)` is.
    fn permutation(&mut self, n: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
        caught(|| {
            let n: i128 = int_arg(n, "Maximum allowed size exceeded")?;
            let n = usize::try_from(n.max(0))
                .map_err(|_| PyValueError::new_err("Maximum allowed size exceeded"))?;
            self.inner.permutation(n).map_err(refused)
        })
    }

    /// `choice(a, size=size, replace=False)`.
    fn choice(&mut self, a: &Bound<'_, PyAny>, size: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
        caught(|| {
            let a = int_arg(a, "a is out of bounds for int64")?;
            let size = size_arg(size)?;
            self.inner.choice(a, size).map_err(refused)
        })
    }
}

/// `base/utils.random_state(seed)`: the stream of any seed from 0 up.
#[pyfunction]
pub(crate) fn random_state(seed: &Bound<'_, PyAny>) -> PyResult<PyRandomState> {
    caught(|| {
        let (negative, words) = int_words(seed)?;
        if negative {
            return Err(PyValueError::new_err(
                "Seed must be between 0 and 2**32 - 1",
            ));
        }
        sdm_random::random_state(&words)
            .map(|inner| PyRandomState { inner })
            .map_err(refused)
    })
}

/// C++'s `std::mt19937`, drawn as `relationallib`'s neighbor sampler draws.
#[pyclass(name = "Mt19937", module = "sdm_core._native")]
pub(crate) struct PyMt19937 {
    inner: Mt19937,
}

#[pymethods]
impl PyMt19937 {
    /// `std::mt19937(seed)`, for `0 <= seed < 2**64`.
    #[new]
    fn new(seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        caught(|| {
            let seed = int_arg(seed, "a seed is an integer from 0 to 2**64 - 1")?;
            Ok(Self {
                inner: Mt19937::new(seed),
            })
        })
    }

    /// `rng()`.
    fn next_u32(&mut self) -> PyResult<u32> {
        caught(|| Ok(self.inner.next_u32()))
    }

    /// The offsets the sampler picks from a neighborhood of `population`
    /// when it asks for `count`, `1 <= count < population`.
    fn floyd(
        &mut self,
        population: &Bound<'_, PyAny>,
        count: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<i64>> {
        caught(|| {
            let population: i64 = int_arg(population, "a population is an int64")?;
            let count: i64 = int_arg(count, "a count is an int64")?;
            if count < 1 || count >= population {
                return Err(PyValueError::new_err(format!(
                    "floyd picks 1 <= count < population, not {count} of {population}"
                )));
            }
            self.inner.floyd(population, count).map_err(refused)
        })
    }
}
