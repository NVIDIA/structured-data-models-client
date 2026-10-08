// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The label executor through Python (`tests/rust_core/sampling/CONTRACT.md`,
//! Labels): the `labels.*` fixture operations and `labels_execute` over
//! Arrow. Test-only until Phase 9 switches Python over to the core.

use std::collections::HashMap;

use arrow_array::{ArrayRef, make_array};
use arrow_data::ArrayData;
use arrow_pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sdm_labels::arrow::{column as arrow_column, pandas_dtypes, to_arrow};
use sdm_labels::column::Column;
use sdm_labels::executor::{Batch, Frame};

use crate::caught;
use crate::sampling::{guarded, raised, require_table, table_columns};

/// A `labels.*` fixture operation: `{"output"}` or `{"error"}` as JSON
/// text; `ValueError` for an unknown operation or input that isn't a JSON
/// object.
#[pyfunction]
pub(crate) fn labels_op(py: Python<'_>, op: &str, input_json: &str) -> PyResult<String> {
    caught(|| {
        let host = crate::schema::host(py)?;
        let (output, warnings) =
            guarded(|| sdm_labels::fixture_op_with_warnings(op, input_json, &host))?;
        if let Err(error) = give_warnings(py, warnings) {
            let value = error.value(py);
            return Ok(serde_json::json!({"error": {
                "code": value.get_type().name()?.to_string(),
                "message": value.str()?.to_string(),
            }})
            .to_string());
        }
        Ok(output)
    })
}

fn give_warnings(py: Python<'_>, warnings: Vec<sdm_labels::Warning>) -> PyResult<()> {
    for warning in warnings {
        match warning {
            sdm_labels::Warning::User(message) => {
                py.import("warnings")?.call_method1(
                    "warn",
                    (message, py.get_type::<pyo3::exceptions::PyUserWarning>()),
                )?;
            }
            sdm_labels::Warning::Float32Overflow(value) => {
                py.import("numpy")?
                    .call_method1("array", (vec![value],))?
                    .call_method1("astype", ("float32",))?;
            }
            sdm_labels::Warning::MaskedFloat32Overflow(value) => {
                let keywords = PyDict::new(py);
                keywords.set_item("dtype", "Float64")?;
                py.import("pandas")?
                    .getattr("Series")?
                    .call((vec![value],), Some(&keywords))?
                    .call_method1("astype", ("float32",))?;
            }
        }
    }
    Ok(())
}

/// The exception the executor raised, as today's Python raises it.
fn label_error(error: sdm_labels::LabelError) -> PyErr {
    raised(&error.code, error.message, error.key)
}

/// An Arrow array from Python: an `Array`, or a `ChunkedArray` combined.
fn import_array(array: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let array = if array.hasattr("combine_chunks")? {
        array.call_method0("combine_chunks")?
    } else {
        array.clone()
    };
    Ok(make_array(ArrayData::from_pyarrow_bound(&array)?))
}

/// `PQueryPandasExecutor().execute_query(query, feat_dict, time_dict,
/// batch_dict, anchor_time)` over Arrow: `query_json` the resolved query's
/// tree, `feat` tables as `pa.Table.from_pandas` gives them, `times` and
/// `batches` arrays by table, `anchor` the examples' anchor times. Gives
/// `(values, mask)`: the labels as a `pyarrow.Array`, and a list of bools.
#[pyfunction]
pub(crate) fn labels_execute<'py>(
    py: Python<'py>,
    query_json: &str,
    feat: &Bound<'py, PyDict>,
    times: &Bound<'py, PyDict>,
    batches: &Bound<'py, PyDict>,
    anchor: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Vec<bool>)> {
    caught(|| execute(py, query_json, feat, times, batches, anchor))
}

fn execute<'py>(
    py: Python<'py>,
    query_json: &str,
    feat: &Bound<'py, PyDict>,
    times: &Bound<'py, PyDict>,
    batches: &Bound<'py, PyDict>,
    anchor: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Vec<bool>)> {
    let mut batch = Batch {
        host: crate::schema::host(py)?,
        ..Batch::default()
    };
    for (name, table) in feat.iter() {
        let name: String = name.extract()?;
        require_table(&table)?;
        let metadata: HashMap<String, String> =
            match table.getattr("schema")?.getattr("metadata")? {
                meta if meta.is_none() => HashMap::new(),
                meta => meta
                    .cast::<PyDict>()?
                    .iter()
                    .map(|(k, v)| {
                        let k: Vec<u8> = k.extract()?;
                        let v: Vec<u8> = v.extract()?;
                        Ok((
                            String::from_utf8_lossy(&k).into_owned(),
                            String::from_utf8_lossy(&v).into_owned(),
                        ))
                    })
                    .collect::<PyResult<_>>()?,
            };
        let dtypes = pandas_dtypes(&metadata);
        let columns = table_columns(&table)?
            .into_iter()
            .map(|(column, array)| {
                let dtype = dtypes.get(&column).map(String::as_str);
                Ok((column, arrow_column(array.as_ref(), dtype)?))
            })
            .collect::<Result<Vec<_>, sdm_labels::LabelError>>()
            .map_err(label_error)?;
        batch.feat.push((name, Frame { columns }));
    }
    for (name, array) in times.iter() {
        let array = import_array(&array)?;
        let column = arrow_column(array.as_ref(), None).map_err(label_error)?;
        batch.time.push((name.extract()?, column));
    }
    for (name, array) in batches.iter() {
        let array = import_array(&array)?;
        let column = arrow_column(array.as_ref(), Some("int64")).map_err(label_error)?;
        let Column::I64(rows) = column else {
            return Err(PyValueError::new_err("a batch is not int64"));
        };
        batch.batch.push((name.extract()?, rows));
    }
    let anchor = import_array(anchor)?;
    batch.anchor = arrow_column(anchor.as_ref(), None).map_err(label_error)?;
    let result = sdm_labels::execute(query_json, &batch);
    give_warnings(py, batch.warnings.into_inner())?;
    let (values, mask) = result
        .and_then(|(out, mask)| Ok((to_arrow(&out)?, mask)))
        .map_err(label_error)?;
    Ok((values.to_data().to_pyarrow(py)?, mask))
}
