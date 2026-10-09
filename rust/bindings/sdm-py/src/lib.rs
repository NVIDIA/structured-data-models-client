// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Python binding over sdm-host, the module `sdm_client._native`.
//!
//! A thin layer over [`HostRun`]: control messages cross as `bytes` both
//! ways, with the attachments each names in a `dict` by id; an error comes
//! back as the `error` control message `HostRun` wrote, raised as the one
//! [`CoreError`]. It makes no semantic call into Python.
//!
//! - Arrow crosses through the PyCapsule interface only (decision E16): an
//!   Arrow attachment in is any object with `__arrow_c_stream__`, and one
//!   out is an [`ArrowStream`] that has it. pyarrow's private
//!   `_export_to_c` and `_import_from_c` are never reached.
//! - The GIL is released for every core call and held only while touching
//!   Python objects; the core's `tracing` events reach Python's `logging`
//!   through `pyo3-log`, installed at the first core call (decision H8).
//! - Importing the module starts no thread and installs no logger, signal
//!   handler or global pool; it reads no environment variable.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::sync::Arc;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_pyarrow::FromPyArrow;
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, SchemaRef};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyMapping};
use sdm_contract::data::AttachmentId;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, ErrorCode, ErrorKind};
use sdm_core::{ArrowData, Attachment, Attachments, CompositionRoot};
use sdm_host::{HostRun, Outgoing, Unreadable, read};

create_exception!(
    sdm_client._native,
    CoreError,
    PyException,
    "An error the core returned. `control` holds its `error` control \
     message and `attachments` the attachments that message names; `kind` \
     and `code` are read from it."
);

/// The bridge from `log` records to Python's `logging`, installed at the
/// first core call; `None` when another `log` logger was there first.
static LOGGING: PyOnceLock<Option<pyo3_log::ResetHandle>> = PyOnceLock::new();

/// Installs the logging bridge, once; every core call goes through here
/// first, with the GIL held.
fn logging(py: Python<'_>) -> Option<&pyo3_log::ResetHandle> {
    LOGGING
        .get_or_init(py, || pyo3_log::try_init().ok())
        .as_ref()
}

/// Forgets the logger levels the bridge has cached, so a change to the
/// `logging` configuration takes effect; the Python host calls it at each
/// run start (decision H8). Before the first core call there is no cache.
#[pyfunction]
fn reset_log_cache(py: Python<'_>) {
    if let Some(handle) = LOGGING.get(py).and_then(Option::as_ref) {
        handle.reset();
    }
}

/// The batches of an Arrow attachment, ready to export again; each export
/// shares their buffers.
#[pyclass(frozen, module = "sdm_client._native")]
struct ArrowStream {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
}

#[pymethods]
impl ArrowStream {
    /// The PyCapsule stream protocol. A requested schema is not supported:
    /// the data is exported as the core holds it.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        if requested_schema.is_some_and(|schema| !schema.is_none()) {
            return Err(PyValueError::new_err(
                "an Arrow attachment is exported in its own schema; none can be requested",
            ));
        }
        let batches: Vec<Result<RecordBatch, ArrowError>> =
            self.batches.iter().cloned().map(Ok).collect();
        let reader: Box<dyn RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(batches, Arc::clone(&self.schema)));
        // The capsule owns the stream until a consumer moves it out; one
        // never consumed is released when the capsule is.
        PyCapsule::new(
            py,
            FFI_ArrowArrayStream::new(reader),
            Some(CString::from(c"arrow_array_stream")),
        )
    }

    /// The PyCapsule schema protocol.
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let schema = FFI_ArrowSchema::try_from(self.schema.as_ref())
            .map_err(|_| PyValueError::new_err("the schema cannot be exported"))?;
        PyCapsule::new(py, schema, Some(CString::from(c"arrow_schema")))
    }
}

/// One attachment as Python sees it: `bytes`, or an [`ArrowStream`].
fn to_python(py: Python<'_>, attachment: Attachment) -> PyResult<Py<PyAny>> {
    Ok(match attachment {
        Attachment::Bytes(bytes) => PyBytes::new(py, &bytes).into_any().unbind(),
        Attachment::Arrow(data) => Py::new(
            py,
            ArrowStream {
                schema: data.schema,
                batches: data.batches,
            },
        )?
        .into_any(),
    })
}

fn attachments_to_python<'py>(
    py: Python<'py>,
    attachments: Attachments,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (id, attachment) in attachments {
        dict.set_item(id.0, to_python(py, attachment)?)?;
    }
    Ok(dict)
}

/// A control message the core wrote, as `(control, attachments)`.
fn outgoing<'py>(py: Python<'py>, out: Outgoing) -> PyResult<(Py<PyBytes>, Py<PyDict>)> {
    let control = PyBytes::new(py, &out.control).unbind();
    Ok((
        control,
        attachments_to_python(py, out.attachments)?.unbind(),
    ))
}

/// The [`CoreError`] raising the `error` control message `out`. Its kind
/// and code are read from the message; its text is theirs alone, so a
/// traceback carries no message the core wrote (decision H17).
fn core_error(py: Python<'_>, out: Outgoing) -> PyErr {
    let (kind, code) = match read(&out) {
        Ok(ControlMessage::Error { error }) => (error.kind, error.code),
        // HostRun gives only error messages here.
        _ => (ErrorKind::Internal, ErrorCode::InternalError),
    };
    let (kind, code) = (kind.as_str(), code.as_str());
    let err = CoreError::new_err(format!("{kind}: {code}"));
    let set = || -> PyResult<()> {
        let value = err.value(py);
        value.setattr("kind", kind)?;
        value.setattr("code", code)?;
        value.setattr("control", PyBytes::new(py, &out.control))?;
        value.setattr("attachments", attachments_to_python(py, out.attachments)?)?;
        Ok(())
    };
    match set() {
        Ok(()) => err,
        Err(failed) => failed,
    }
}

/// Why the attachments a Python caller sent could not be read.
enum Unread {
    /// A Python error that is not the data's: a wrong type, an interrupt.
    Python(PyErr),
    /// An Arrow attachment the binding could not read into Arrow data,
    /// which `sdm-host` turns into the run's error.
    Arrow(AttachmentId, Unreadable),
}

impl From<PyErr> for Unread {
    fn from(error: PyErr) -> Self {
        Unread::Python(error)
    }
}

/// Reads one Arrow attachment through `__arrow_c_stream__`. A failure of
/// the producer, an exception from its `__arrow_c_stream__` or an error
/// its stream reports, is the provider's; a batch Arrow cannot build in
/// the stream's own schema, such as one with nulls in a column declared
/// without them, fails check (a). Nothing of either error's text is kept,
/// since it may quote values (decision H17).
fn read_arrow(py: Python<'_>, id: u32, value: &Bound<'_, PyAny>) -> Result<ArrowData, Unread> {
    let producer = |error: PyErr| {
        if error.is_instance_of::<PyException>(py) {
            Unread::Arrow(AttachmentId(id), Unreadable::Producer)
        } else {
            Unread::Python(error)
        }
    };
    let reader = ArrowArrayStreamReader::from_pyarrow_bound(value).map_err(producer)?;
    let schema = reader.schema();
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|error| {
            let failure = match error {
                ArrowError::CDataInterface(_) => Unreadable::Producer,
                _ => Unreadable::Declared,
            };
            Unread::Arrow(AttachmentId(id), failure)
        })?);
    }
    Ok(ArrowData::new(schema, batches))
}

/// The attachments a Python caller sends, read with the GIL held: `bytes`,
/// or an object with `__arrow_c_stream__`, by id.
fn attachments_from_python(attachments: &Bound<'_, PyMapping>) -> Result<Attachments, Unread> {
    let py = attachments.py();
    let mut read: Attachments = BTreeMap::new();
    for item in attachments.items()?.try_iter()? {
        let (id, value): (u32, Bound<'_, PyAny>) = item?.extract()?;
        let attachment = if let Ok(bytes) = value.cast::<PyBytes>() {
            Attachment::Bytes(bytes.as_bytes().to_vec())
        } else if value.hasattr("__arrow_c_stream__")? {
            // Checked first, so arrow-pyarrow takes the capsule path and
            // never pyarrow's private export.
            Attachment::Arrow(read_arrow(py, id, &value)?)
        } else {
            return Err(Unread::Python(PyTypeError::new_err(format!(
                "attachment {id} is a {}: an attachment is bytes or an object with __arrow_c_stream__",
                value.get_type().name()?
            ))));
        };
        read.insert(AttachmentId(id), attachment);
    }
    Ok(read)
}

/// One core run; see [`HostRun`]. Calls may come from any thread; a call
/// while another is in the run raises `RUN_BUSY`.
#[pyclass(frozen, module = "sdm_client._native")]
struct Run {
    run: HostRun,
}

#[pymethods]
impl Run {
    /// Starts the run a `start` control message asks for, with the
    /// attachments it names.
    #[staticmethod]
    fn start(py: Python<'_>, message: &[u8], attachments: &Bound<'_, PyMapping>) -> PyResult<Run> {
        logging(py);
        let attachments = match attachments_from_python(attachments) {
            Ok(attachments) => attachments,
            Err(Unread::Python(error)) => return Err(error),
            Err(Unread::Arrow(id, failure)) => {
                return Err(core_error(py, HostRun::start_unreadable(id, failure)));
            }
        };
        let message = message.to_vec();
        py.detach(|| HostRun::start(&CompositionRoot::default(), &message, attachments))
            .map(|run| Run { run })
            .map_err(|out| core_error(py, out))
    }

    /// The run's next step, as `(control, attachments)`.
    fn next(&self, py: Python<'_>) -> PyResult<(Py<PyBytes>, Py<PyDict>)> {
        logging(py);
        match py.detach(|| self.run.next()) {
            Ok(out) => outgoing(py, out),
            Err(out) => Err(core_error(py, out)),
        }
    }

    /// Hands the run a `fetch_reply` or `model_reply`, with the attachments
    /// it names.
    fn reply(
        &self,
        py: Python<'_>,
        message: &[u8],
        attachments: &Bound<'_, PyMapping>,
    ) -> PyResult<()> {
        logging(py);
        let message = message.to_vec();
        let result = match attachments_from_python(attachments) {
            Ok(attachments) => py.detach(|| self.run.reply(&message, attachments)),
            Err(Unread::Python(error)) => return Err(error),
            // An Arrow attachment that could not be read still ends the
            // fetch it answers, as sdm-host decides (decision H16).
            Err(Unread::Arrow(id, failure)) => {
                py.detach(|| self.run.reply_unreadable(&message, id, failure))
            }
        };
        result.map_err(|out| core_error(py, out))
    }

    /// Cancels the run; its attachments are released.
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        logging(py);
        py.detach(|| self.run.cancel())
            .map_err(|out| core_error(py, out))
    }

    /// Takes an attachment the run made and a step named: `bytes`, an
    /// [`ArrowStream`], or `None` when the run holds no such id.
    fn take_attachment(&self, py: Python<'_>, id: u32) -> PyResult<Option<Py<PyAny>>> {
        logging(py);
        match py.detach(|| self.run.take_attachment(AttachmentId(id))) {
            Ok(Some(attachment)) => to_python(py, attachment).map(Some),
            Ok(None) => Ok(None),
            Err(out) => Err(core_error(py, out)),
        }
    }

    /// The warnings given since the last call, as one `warnings` message
    /// `(control, attachments)`, or `None` when there are none.
    fn take_warnings(&self, py: Python<'_>) -> PyResult<Option<(Py<PyBytes>, Py<PyDict>)>> {
        logging(py);
        match py.detach(|| self.run.take_warnings()) {
            Ok(Some(out)) => outgoing(py, out).map(Some),
            Ok(None) => Ok(None),
            Err(out) => Err(core_error(py, out)),
        }
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add("CONTRACT_VERSION", CONTRACT_VERSION)?;
    module.add("TEST_OPS", cfg!(feature = "test-ops"))?;
    module.add("CoreError", py.get_type::<CoreError>())?;
    module.add_class::<ArrowStream>()?;
    module.add_class::<Run>()?;
    module.add_function(wrap_pyfunction!(reset_log_cache, module)?)?;
    Ok(())
}
