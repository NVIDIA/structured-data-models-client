// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sampling's building blocks through Python (`tests/rust_core/sampling/CONTRACT.md`):
//! `DateOffset` arithmetic, text normalization, the in-memory graph store
//! and neighbor sampler, and the sampler fixtures. Test-only until Phase 9
//! switches Python over to the core. Input it can't read is a `ValueError`,
//! and so is a panic, which would be a bug in the port ([`caught`]).

use arrow_array::{ArrayRef, new_empty_array};
use pyo3::create_exception;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use sdm_labels::date_offset::{Bound as TimeBound, DateOffset, OffsetError, pooled_offset};
use sdm_sampling::graph_store::{GraphSpec, KeyValue, MissingKey};
use sdm_sampling::neighbor::{
    EdgeType, NeighborSampler, SampleError, Seed, TimeRanges, cast_error, rel_type,
};
use serde_json::{Value, json};

use crate::caught;
use crate::random::int_arg;

/// `f`'s answer, with its error message and any panic as a `ValueError`.
pub(crate) fn guarded<T>(f: impl FnOnce() -> Result<T, String>) -> PyResult<T> {
    caught(|| f().map_err(PyValueError::new_err))
}

fn parse(text: &str, what: &str) -> Result<Value, String> {
    serde_json::from_str(text).map_err(|e| format!("{what} is not JSON: {e}"))
}

/// `pd.DateOffset(**offset)` of an offset's JSON: a `ValueError` for JSON
/// that isn't one, and the error making it raises, such as `OverflowError`
/// for a `timedelta` past its range.
fn offset(text: &str) -> PyResult<DateOffset> {
    let offset = parse(text, "the offset")
        .and_then(|value| DateOffset::from_json(&value))
        .map_err(PyValueError::new_err)?;
    offset.constructed().map_err(offset_error)?;
    Ok(offset)
}

fn offset_error(error: OffsetError) -> PyErr {
    raised(error.code, error.message, None)
}

fn error_json(error: &OffsetError) -> Value {
    json!({"error": {"code": error.code, "message": error.message}})
}

/// `pd.Timestamp(t, unit='ns') + offset` (`sign` 1) or `- offset` (`sign`
/// -1) for each timestamp of the JSON list: a JSON list of `{"ns": N}` or
/// `{"error": {"code", "message"}}`.
#[pyfunction]
pub(crate) fn date_offset_apply(
    timestamps_json: &str,
    offset_json: &str,
    sign: &Bound<'_, PyAny>,
) -> PyResult<String> {
    caught(|| {
        let sign: i64 = int_arg(sign, "sign is 1 or -1")?;
        if sign != 1 && sign != -1 {
            return Err(PyValueError::new_err(format!(
                "sign is 1 or -1, not {sign}"
            )));
        }
        let offset = offset(offset_json)?;
        let Value::Array(timestamps) =
            parse(timestamps_json, "the timestamps").map_err(PyValueError::new_err)?
        else {
            return Err(PyValueError::new_err("the timestamps are a JSON list"));
        };
        let out = timestamps
            .iter()
            .map(|ts| {
                let ts = ts.as_i64().ok_or_else(|| {
                    PyValueError::new_err(format!("a timestamp is an integer, not {ts}"))
                })?;
                Ok(match offset.apply(ts, sign) {
                    Ok(ns) => json!({"ns": ns}),
                    Err(error) => error_json(&error),
                })
            })
            .collect::<PyResult<Vec<Value>>>()?;
        Ok(Value::Array(out).to_string())
    })
}

/// `base/utils.date_offset_to_seconds(offset, bound)`, as exact as Python's
/// integers.
#[pyfunction]
pub(crate) fn date_offset_to_seconds<'py>(
    py: Python<'py>,
    offset_json: &str,
    bound: &str,
) -> PyResult<Bound<'py, PyAny>> {
    caught(|| {
        let bound = TimeBound::parse(bound).ok_or_else(|| {
            PyValueError::new_err(format!("bound is 'earliest' or 'latest', not {bound:?}"))
        })?;
        let seconds = offset(offset_json)?
            .to_seconds(bound)
            .map_err(offset_error)?;
        if let Some(seconds) = seconds.to_i128() {
            return Ok(seconds.into_pyobject(py)?.into_any());
        }
        let (high, low) = seconds.parts();
        high.into_pyobject(py)?
            .call_method1("__lshift__", (64,))?
            .call_method1("__add__", (low,))
    })
}

/// `rfm._date_offset_to_ns(offset)`: `{"ns": N}` or the error it raises,
/// making the offset included.
#[pyfunction]
pub(crate) fn date_offset_to_ns(offset_json: &str) -> PyResult<String> {
    guarded(|| {
        let offset =
            parse(offset_json, "the offset").and_then(|value| DateOffset::from_json(&value))?;
        Ok(
            match offset.constructed().and_then(|()| offset.to_ns_from_2020()) {
                // Past int64 for offsets of centuries, so written as digits.
                Ok(ns) => format!("{{\"ns\": {ns}}}"),
                Err(error) => error_json(&error).to_string(),
            },
        )
    })
}

/// `base/sampler._pooled_offset(offsets, earliest=earliest)`, as
/// `{"n", "kwds"}`.
#[pyfunction]
pub(crate) fn date_offset_pool(offsets_json: &str, earliest: bool) -> PyResult<String> {
    caught(|| {
        let Value::Array(given) =
            parse(offsets_json, "the offsets").map_err(PyValueError::new_err)?
        else {
            return Err(PyValueError::new_err("the offsets are a JSON list"));
        };
        let offsets = given
            .iter()
            .map(|value| offset(&value.to_string()))
            .collect::<PyResult<Vec<_>>>()?;
        if offsets.is_empty() {
            return Err(PyValueError::new_err("there is no offset to pool"));
        }
        let pooled = pooled_offset(&offsets, earliest).map_err(offset_error)?;
        Ok(pooled.to_json().to_string())
    })
}

/// `base/sampler._normalize_text(pd.Series(values, dtype=object), max_words)`
/// as a list of word lists. `max_words` is any `int`, a negative one slicing
/// from the end as it does in Python; the cut to `6 * max_words` characters
/// takes a C `long`, so past that it is pyarrow's `OverflowError`, unless
/// there is nothing to cut.
#[pyfunction]
#[pyo3(signature = (values, max_words))]
pub(crate) fn normalize_text(
    values: Vec<Option<String>>,
    max_words: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Vec<String>>> {
    caught(|| {
        let max_words = match max_words {
            Some(value) if !value.is_none() => {
                // A `TypeError` for anything but an integer.
                crate::random::int_words(value)?;
                let cut = int_arg::<i64>(value, "")
                    .ok()
                    .filter(|words| words.checked_mul(6).is_some());
                if cut.is_none() && !values.is_empty() {
                    return Err(pyo3::exceptions::PyOverflowError::new_err(
                        "Python int too large to convert to C long",
                    ));
                }
                cut
            }
            _ => None,
        };
        Ok(sdm_sampling::text::normalize_text(&values, max_words))
    })
}

// --- The neighbor sampler ----------------------------------------------------

/// `py::array` of a value, as pybind11 casts one: `numpy.asarray`.
fn as_array<'py>(
    numpy: &Bound<'py, PyModule>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    numpy.call_method1("asarray", (value,))
}

/// `validate_arr`: a `RuntimeError` for an array of another rank, or, when
/// it holds anything and `int64` is asked, of another dtype.
fn check_array(array: &Bound<'_, PyAny>, name: &str, dim: usize, int64: bool) -> PyResult<()> {
    let ndim: usize = array.getattr("ndim")?.extract()?;
    if ndim != dim {
        return Err(PyRuntimeError::new_err(format!("{name} is not {dim}D.")));
    }
    let size: usize = array.getattr("size")?.extract()?;
    if size == 0 {
        return Ok(());
    }
    if int64 {
        let dtype = array.getattr("dtype")?;
        let kind: String = dtype.getattr("kind")?.extract()?;
        let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
        if kind != "i" || itemsize != 8 {
            return Err(PyRuntimeError::new_err(format!(
                "{name} must have dtype int64."
            )));
        }
    }
    Ok(())
}

/// A checked one-dimensional array's values; an empty one of any dtype is
/// empty.
fn int_values(array: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    let size: usize = array.getattr("size")?.extract()?;
    if size == 0 {
        return Ok(Vec::new());
    }
    array.call_method0("tolist")?.extract()
}

/// `validate_arr_dict`: each key one of `keys`, each value an array of rank
/// `dim` (and `int64` when asked), in the dict's order.
fn checked_dict<'py>(
    numpy: &Bound<'py, PyModule>,
    dict: &Bound<'py, PyDict>,
    name: &str,
    keys: &[String],
    dim: usize,
    int64: bool,
) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
    let mut out = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = key.str()?.to_string();
        if !keys.contains(&key) {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid type '{key}' found in {name}."
            )));
        }
        let array = as_array(numpy, &value)?;
        check_array(&array, &format!("Type '{key}' in {name}"), dim, int64)?;
        out.push((key, array));
    }
    Ok(out)
}

fn int_dict(entries: Vec<(String, Bound<'_, PyAny>)>) -> PyResult<Vec<(String, Vec<i64>)>> {
    entries
        .into_iter()
        .map(|(key, array)| Ok((key, int_values(&array)?)))
        .collect()
}

/// A seed as the sampler takes it: any integer from `-2**31` up.
fn sampler_seed(seed: &Bound<'_, PyAny>) -> PyResult<Seed> {
    let index = seed
        .py()
        .import("operator")?
        .call_method1("index", (seed,))?;
    let (negative, words) = crate::random::int_words(&index)?;
    if negative {
        let value = words.first().copied().map_or(0, i64::from);
        if words.len() > 1 || value > 1 << 31 {
            return Err(PyTypeError::new_err(format!(
                "a seed is an integer from -2**31 up, not {index}"
            )));
        }
        return Ok(Seed::Word(-value));
    }
    Ok(match words.as_slice() {
        [] => Seed::Word(0),
        [word] => Seed::Word(i64::from(*word)),
        _ => Seed::Words(words),
    })
}

fn sample_error(error: SampleError) -> PyErr {
    match error {
        SampleError::Runtime(message) => PyRuntimeError::new_err(message),
        SampleError::Index(message) => pyo3::exceptions::PyIndexError::new_err(message),
        SampleError::Key(key) => PyKeyError::new_err(key),
    }
}

/// The constructor's `seed`: when it isn't passed, `relationallib`'s default,
/// 123; when it is, even as `None`, it is read as `seed()` reads one, so
/// `None` is refused as `relationallib` refuses it.
enum SeedArg<'py> {
    Default,
    Given(Bound<'py, PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for SeedArg<'py> {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(SeedArg::Given(obj.to_owned()))
    }
}

/// `relationallib.NeighborSampler`, its signatures, results and errors.
#[pyclass(name = "NeighborSampler", module = "sdm_core._native")]
pub(crate) struct PyNeighborSampler {
    inner: NeighborSampler,
    rel_types: Vec<String>,
}

fn int64_array<'py>(numpy: &Bound<'py, PyModule>, values: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let bytes = PyBytes::new(numpy.py(), &bytes);
    numpy
        .call_method1("frombuffer", (bytes, numpy.getattr("int64")?))?
        .call_method0("copy")
}

fn int64_dict<'py>(
    numpy: &Bound<'py, PyModule>,
    entries: &[(String, Vec<i64>)],
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(numpy.py());
    for (key, values) in entries {
        dict.set_item(key, int64_array(numpy, values)?)?;
    }
    Ok(dict)
}

/// A sequence of three strings, as pybind11 reads a `std::tuple`.
fn edge_type(value: &Bound<'_, PyAny>) -> PyResult<EdgeType> {
    let parts: Vec<String> = value.extract()?;
    match <[String; 3]>::try_from(parts) {
        Ok([dst, rel, src]) => Ok((dst, rel, src)),
        Err(_) => Err(PyTypeError::new_err("an edge type is three strings")),
    }
}

#[pymethods]
impl PyNeighborSampler {
    #[new]
    #[pyo3(signature = (node_types, edge_types, colptr, row, node_time, seed=SeedArg::Default))]
    fn new(
        node_types: Vec<String>,
        edge_types: &Bound<'_, PyAny>,
        colptr: &Bound<'_, PyDict>,
        row: &Bound<'_, PyDict>,
        node_time: &Bound<'_, PyAny>,
        seed: SeedArg<'_>,
    ) -> PyResult<Self> {
        caught(|| Self::build(node_types, edge_types, colptr, row, node_time, seed))
    }

    /// Restart the generator.
    fn seed(&mut self, seed: &Bound<'_, PyAny>) -> PyResult<()> {
        caught(|| {
            let seed = sampler_seed(seed)?;
            self.inner.seed(&seed);
            Ok(())
        })
    }

    /// `(row, col, node_id, batch, num_sampled_nodes, num_sampled_edges)`.
    #[pyo3(signature = (num_neighbors, sampling_time_offsets, entity_table, seed_node, seed_time))]
    fn sample<'py>(
        &mut self,
        num_neighbors: &Bound<'py, PyDict>,
        sampling_time_offsets: &Bound<'py, PyDict>,
        entity_table: String,
        seed_node: &Bound<'py, PyAny>,
        seed_time: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        caught(|| {
            self.sampled(
                num_neighbors,
                sampling_time_offsets,
                &entity_table,
                seed_node,
                seed_time,
            )
        })
    }
}

impl PyNeighborSampler {
    fn build(
        node_types: Vec<String>,
        edge_types: &Bound<'_, PyAny>,
        colptr: &Bound<'_, PyDict>,
        row: &Bound<'_, PyDict>,
        node_time: &Bound<'_, PyAny>,
        seed: SeedArg<'_>,
    ) -> PyResult<Self> {
        let py = edge_types.py();
        let edge_types = edge_types
            .try_iter()?
            .map(|edge| edge_type(&edge?))
            .collect::<PyResult<Vec<_>>>()?;
        let node_time = match node_time.cast::<PyDict>() {
            Ok(dict) => dict.clone(),
            Err(_) => py
                .get_type::<PyDict>()
                .call1((node_time,))?
                .cast_into::<PyDict>()?,
        };
        let seed = match seed {
            SeedArg::Given(seed) => sampler_seed(&seed)?,
            SeedArg::Default => Seed::Word(123),
        };
        let numpy = py.import("numpy")?;
        let rel_types: Vec<String> = edge_types.iter().map(rel_type).collect();
        let colptr = int_dict(checked_dict(&numpy, colptr, "colptr", &rel_types, 1, true)?)?;
        let row = int_dict(checked_dict(&numpy, row, "row", &rel_types, 1, true)?)?;
        let node_time = int_dict(checked_dict(
            &numpy,
            &node_time,
            "node_time",
            &node_types,
            1,
            true,
        )?)?;
        let inner = NeighborSampler::new(node_types, edge_types, colptr, row, node_time, &seed)
            .map_err(sample_error)?;
        Ok(Self { inner, rel_types })
    }

    fn sampled<'py>(
        &mut self,
        num_neighbors: &Bound<'py, PyDict>,
        sampling_time_offsets: &Bound<'py, PyDict>,
        entity_table: &str,
        seed_node: &Bound<'py, PyAny>,
        seed_time: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let py = num_neighbors.py();
        let numpy = py.import("numpy")?;
        // pybind11 takes a `py::array` argument only as a `numpy.ndarray`,
        // unlike a dict's values, which it converts.
        let ndarray = numpy.getattr("ndarray")?;
        for (name, value) in [("seed_node", seed_node), ("seed_time", seed_time)] {
            if !value.is_instance(&ndarray)? {
                return Err(PyTypeError::new_err(format!(
                    "sample(): incompatible function arguments: {name} is a numpy.ndarray, not {}",
                    value.get_type().name()?
                )));
            }
        }
        let counts = int_dict(checked_dict(
            &numpy,
            num_neighbors,
            "num_neighbors",
            &self.rel_types,
            1,
            true,
        )?)?;
        let ranges = checked_dict(
            &numpy,
            sampling_time_offsets,
            "sampling_time_offsets",
            &self.rel_types,
            2,
            false,
        )?
        .into_iter()
        .map(|(key, array)| Ok((key, time_ranges(&array)?)))
        .collect::<PyResult<Vec<_>>>()?;
        check_array(seed_node, "seed_node", 1, true)?;
        check_array(seed_time, "seed_time", 1, true)?;
        let seed_node = int_values(seed_node)?;
        let seed_time = int_values(seed_time)?;
        let sampled = self
            .inner
            .sample(&counts, &ranges, entity_table, &seed_node, &seed_time)
            .map_err(sample_error)?;
        PyTuple::new(
            py,
            [
                int64_dict(&numpy, &sampled.row)?,
                int64_dict(&numpy, &sampled.col)?,
                int64_dict(&numpy, &sampled.node_id)?,
                int64_dict(&numpy, &sampled.batch)?,
                int64_dict(&numpy, &sampled.num_sampled_nodes)?,
                int64_dict(&numpy, &sampled.num_sampled_edges)?,
            ],
        )
    }
}

/// A checked two-dimensional `sampling_time_offsets` array: its rows as
/// `(start, end)`, a `None` start open, when it is two wide. A value that
/// can't be cast is kept as the error, which the sampler raises when it
/// reaches that hop, where `relationallib` casts it.
fn time_ranges(array: &Bound<'_, PyAny>) -> PyResult<TimeRanges> {
    let shape: Vec<usize> = array.getattr("shape")?.extract()?;
    let (rows, width) = (shape[0], shape[1]);
    if width != 2 {
        return Ok(TimeRanges {
            width,
            ranges: vec![Ok((None, 0)); rows],
        });
    }
    let mut ranges = Vec::with_capacity(rows);
    for row in array.try_iter()? {
        let row = row?;
        let start = row.get_item(0)?;
        let start = if start.is_none() {
            Ok(None)
        } else {
            cast_int(&start)?.map(Some)
        };
        let end = cast_int(&row.get_item(1)?)?;
        ranges.push(start.and_then(|start| Ok((start, end?))));
    }
    Ok(TimeRanges { width, ranges })
}

/// `cast<int64_t>()` of a Python object, or the text of pybind11's error:
/// it refuses a float, and an integer past `int64`.
fn cast_int(value: &Bound<'_, PyAny>) -> PyResult<Result<i64, String>> {
    if !value.is_instance_of::<pyo3::types::PyFloat>()
        && let Ok(value) = value.extract::<i64>()
    {
        return Ok(Ok(value));
    }
    Ok(Err(cast_error(&value.get_type().str()?.to_string())))
}

// --- The graph store -----------------------------------------------------------

create_exception!(
    sdm_core._native,
    OutOfBoundsDatetime,
    PyValueError,
    "pandas' OutOfBoundsDatetime, raised where pandas raises it."
);
create_exception!(
    sdm_core._native,
    OutOfBoundsTimedelta,
    PyValueError,
    "pandas' OutOfBoundsTimedelta, raised where pandas raises it."
);
create_exception!(
    sdm_core._native,
    DateParseError,
    PyValueError,
    "pandas' DateParseError, raised where pandas raises it."
);

/// The exception today's Python raises: `KeyError(key)` where the error
/// kept the key, whose `repr` its message already is, and otherwise by its
/// class name.
pub(crate) fn raised(code: &str, message: String, key: Option<String>) -> PyErr {
    match key {
        Some(key) => PyKeyError::new_err(key),
        None => python_error(code, message),
    }
}

/// The exception today's Python raises, by its class name.
fn python_error(code: &str, message: String) -> PyErr {
    match code {
        "KeyError" => PyKeyError::new_err(message),
        "TypeError" => PyTypeError::new_err(message),
        "RuntimeError" => PyRuntimeError::new_err(message),
        "OverflowError" => pyo3::exceptions::PyOverflowError::new_err(message),
        "AssertionError" => pyo3::exceptions::PyAssertionError::new_err(message),
        "AttributeError" => pyo3::exceptions::PyAttributeError::new_err(message),
        "IndexError" => pyo3::exceptions::PyIndexError::new_err(message),
        "MemoryError" => pyo3::exceptions::PyMemoryError::new_err(message),
        "NotImplementedError" => pyo3::exceptions::PyNotImplementedError::new_err(message),
        "ZeroDivisionError" => pyo3::exceptions::PyZeroDivisionError::new_err(message),
        "OutOfBoundsDatetime" => OutOfBoundsDatetime::new_err(message),
        "OutOfBoundsTimedelta" => OutOfBoundsTimedelta::new_err(message),
        "DateParseError" => DateParseError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

/// A `TypeError` unless `table` is a `pyarrow.Table`.
pub(crate) fn require_table(table: &Bound<'_, PyAny>) -> PyResult<()> {
    let pa_table = table.py().import("pyarrow")?.getattr("Table")?;
    if table.is_instance(&pa_table)? {
        return Ok(());
    }
    Err(PyTypeError::new_err(format!(
        "a table is a pyarrow.Table, not {}",
        table.get_type().name()?
    )))
}

/// A `pyarrow.Table`'s columns, by name, each one Arrow array.
pub(crate) fn table_columns(table: &Bound<'_, PyAny>) -> PyResult<Vec<(String, ArrayRef)>> {
    require_table(table)?;
    let table = table.call_method0("combine_chunks")?;
    let (schema, batches) = crate::import_table(&table)?;
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| {
            let array = match batches.as_slice() {
                [] => new_empty_array(field.data_type()),
                [batch] => batch.column(i).clone(),
                _ => {
                    return Err(PyValueError::new_err(
                        "a combined table came in more than one batch",
                    ));
                }
            };
            Ok((field.name().clone(), array))
        })
        .collect()
}

fn key_object<'py>(py: Python<'py>, key: &KeyValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match key {
        KeyValue::Int(v) => v.into_pyobject(py)?.into_any(),
        KeyValue::Float(v) => v.into_pyobject(py)?.into_any(),
        KeyValue::Text(v) => v.into_pyobject(py)?.into_any(),
    })
}

/// `LocalGraphStore(graph)` of `pa.Table.from_pandas` tables and the graph's
/// JSON: its report, masks, key maps, times and CSC arrays.
#[pyfunction]
pub(crate) fn graph_store<'py>(
    py: Python<'py>,
    tables: &Bound<'py, PyDict>,
    graph_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    caught(|| store_dict(py, tables, graph_json))
}

fn store_dict<'py>(
    py: Python<'py>,
    tables: &Bound<'py, PyDict>,
    graph_json: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let host = crate::schema::host(py)?;
    let graph = GraphSpec::from_json(graph_json).map_err(PyValueError::new_err)?;
    let mut data = Vec::with_capacity(tables.len());
    for (name, table) in tables.iter() {
        data.push((name.extract::<String>()?, table_columns(&table)?));
    }
    let store =
        sdm_sampling::graph_store::graph_store(&graph, &data, &host).map_err(|e| match e.key {
            Some(MissingKey::Name(key)) => raised(&e.code, e.message, Some(key)),
            Some(MissingKey::None) => PyKeyError::new_err((None::<String>,)),
            None => raised(&e.code, e.message, None),
        })?;
    let out = PyDict::new(py);
    let report = PyDict::new(py);
    for (name, counts) in &store.report {
        let entry = PyDict::new(py);
        entry.set_item("input_rows", counts.input_rows)?;
        entry.set_item("output_rows", counts.output_rows)?;
        entry.set_item("null_primary_key_rows", counts.null_primary_key_rows)?;
        entry.set_item(
            "duplicate_primary_key_rows",
            counts.duplicate_primary_key_rows,
        )?;
        entry.set_item("null_time_rows", counts.null_time_rows)?;
        report.set_item(name, entry)?;
    }
    out.set_item("report", report)?;
    let masks = PyDict::new(py);
    for (name, mask) in &store.masks {
        masks.set_item(name, mask.clone())?;
    }
    out.set_item("masks", masks)?;
    let keys = PyDict::new(py);
    for (name, values) in &store.keys {
        let values = values
            .iter()
            .map(|key| key_object(py, key))
            .collect::<PyResult<Vec<_>>>()?;
        keys.set_item(name, PyList::new(py, values)?)?;
    }
    out.set_item("keys", keys)?;
    let ints = |entries: &[(String, Vec<i64>)]| -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (name, values) in entries {
            dict.set_item(name, values.clone())?;
        }
        Ok(dict)
    };
    out.set_item("node_ids", ints(&store.node_ids)?)?;
    out.set_item("time", ints(&store.time)?)?;
    let ranges = PyDict::new(py);
    for (name, (lo, hi)) in &store.time_range {
        ranges.set_item(name, vec![*lo, *hi])?;
    }
    out.set_item("time_range", ranges)?;
    let by_edge = |entries: &[(EdgeType, Vec<i64>)]| -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (edge, values) in entries {
            dict.set_item(rel_type(edge), values.clone())?;
        }
        Ok(dict)
    };
    out.set_item("colptr", by_edge(&store.colptr)?)?;
    out.set_item("row", by_edge(&store.row)?)?;
    Ok(out)
}

/// A `sampling.*` fixture operation: `{"output"}` or `{"error"}` as JSON
/// text; `ValueError` for an unknown operation or input that isn't a JSON
/// object.
#[pyfunction]
pub(crate) fn sampling_op(op: &str, input_json: &str) -> PyResult<String> {
    guarded(|| sdm_sampling::fixture_op(op, input_json))
}
