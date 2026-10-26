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

use arrow_array::ffi::{FFI_ArrowArray, from_ffi_and_data_type};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{
    RecordBatch, RecordBatchIterator, RecordBatchOptions, RecordBatchReader, StructArray,
};
use arrow_data::{BufferSpec, layout};
use arrow_schema::ffi::FFI_ArrowSchema;
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyMapping, PyTuple};
use sdm_contract::data::AttachmentId;
use sdm_contract::{CONTRACT_VERSION, ControlMessage, ErrorCode, ErrorKind};
use sdm_host::{
    ArrowData, Attachment, Attachments, HostRun, Outgoing, Unreadable, Unreadables, default_root,
    read,
};

create_exception!(
    sdm_client._native,
    CoreError,
    PyException,
    "An error the core returned. `control` holds its `error` control \
     message and `attachments` the attachments that message names; `kind`, \
     `code` and `retryable` are read from it."
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

/// Runs a core call with the GIL released. The logging bridge calls
/// Python's `logging` from inside it, and pyo3-log leaves an exception a
/// handler raised set in the thread's error indicator rather than
/// returning it, so after the call:
///
/// - an `Exception` is reported as `logging` reports its own handler
///   failures, through `sys.unraisablehook`, and the call's answer is
///   delivered; the exception is the handler's and holds nothing the core
///   wrote but the event;
/// - anything else, such as `KeyboardInterrupt` or `SystemExit`, is raised
///   in place of the answer, which is dropped: the run stays where the
///   call left it, so a step `next` gave is gone and the next call meets
///   the run's following state.
fn core_call<T: Send>(py: Python<'_>, call: impl FnOnce() -> T + Send) -> PyResult<T> {
    let answer = py.detach(call);
    core_result(py, answer)
}

/// Apply the same pending-exception policy whether ownership permits
/// releasing the GIL during the core call or requires holding it.
fn core_result<T>(py: Python<'_>, answer: T) -> PyResult<T> {
    match PyErr::take(py) {
        None => Ok(answer),
        Some(raised) if raised.is_instance_of::<PyException>(py) => {
            raised.write_unraisable(py, None);
            Ok(answer)
        }
        Some(interrupt) => Err(interrupt),
    }
}

/// The table of what every error's and warning's sentence says, as its JSON
/// text (`sdm_contract::wording`, T3.17): the host reads it once and fills
/// each sentence from it.
#[pyfunction]
fn templates() -> String {
    sdm_host::templates()
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

/// Whether `text` is a POSIX `TZ` string a run takes as its local zone, as
/// `HostRun::start` checks it; the Python host names a local zone only when
/// it is, so no `TZ` setting refuses a run.
#[pyfunction]
fn is_posix_zone(text: &str) -> bool {
    sdm_host::is_posix_zone(text)
}

/// The machine's files as Python reads them, for the local zone's rule:
/// `os.path.isfile` and `pathlib.Path.resolve(strict=True)`, the calls the
/// Python host made before the rule moved into `sdm-host` (T3.12), so the
/// binding does no I/O of its own. The first exception Python raised that
/// the rule cannot read as an answer is kept, and raised after.
struct PyZoneFiles<'py> {
    isfile: Bound<'py, PyAny>,
    path: Bound<'py, PyAny>,
    raised: std::cell::RefCell<Option<PyErr>>,
}

impl PyZoneFiles<'_> {
    fn kept<T>(&self, answer: PyResult<T>, otherwise: T) -> T {
        answer.unwrap_or_else(|error| {
            self.raised.borrow_mut().get_or_insert(error);
            otherwise
        })
    }
}

impl sdm_host::ZoneFiles for PyZoneFiles<'_> {
    fn is_file(&self, path: &std::path::Path) -> bool {
        let answer = self
            .isfile
            .call1((path,))
            .and_then(|found| found.is_truthy());
        self.kept(answer, false)
    }

    fn real_path(&self, path: &std::path::Path) -> Option<std::path::PathBuf> {
        let py = self.path.py();
        let resolved = self.path.call1((path,)).and_then(|made| {
            let keywords = PyDict::new(py);
            keywords.set_item("strict", true)?;
            made.call_method("resolve", (), Some(&keywords))
        });
        match resolved {
            Ok(found) => self.kept(found.str().map(|text| Some(text.to_string().into())), None),
            // A path that does not resolve (OSError) names no zone.
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyOSError>(py) => None,
            Err(error) => self.kept(Err(error), None),
        }
    }
}

/// The local zone a host's facts name, by the one rule every host reads it
/// with (`sdm_host::local_zone`, T3.12): `setting` is `TZ` when set, and
/// `roots` the zone lookup roots in the host's zone library's order;
/// `('iana', name)`, `('posix', text)`, or `None`. The rule's questions of
/// the machine's files are answered by Python (`os.path.isfile`,
/// `pathlib.Path.resolve`).
#[pyfunction]
#[pyo3(signature = (setting, roots))]
fn local_zone(
    py: Python<'_>,
    setting: Option<&str>,
    roots: Vec<String>,
) -> PyResult<Option<(&'static str, String)>> {
    let files = PyZoneFiles {
        isfile: py.import("os.path")?.getattr("isfile")?,
        path: py.import("pathlib")?.getattr("Path")?,
        raised: std::cell::RefCell::new(None),
    };
    let roots: Vec<std::path::PathBuf> = roots.into_iter().map(Into::into).collect();
    let zone = sdm_host::local_zone(setting, &roots, &files);
    if let Some(error) = files.raised.into_inner() {
        return Err(error);
    }
    Ok(zone.map(|zone| match zone {
        sdm_contract::facts::LocalZone::Iana(name) => ("iana", name),
        sdm_contract::facts::LocalZone::Posix(text) => ("posix", text),
    }))
}

/// What each model supports, as the client describes it to a caller, as
/// the core's JSON (`sdm_host::capabilities`, T3.12).
#[pyfunction]
fn capabilities() -> String {
    sdm_host::capabilities()
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
    let (kind, code, retryable) = match read(&out) {
        Ok(ControlMessage::Error { error }) => (error.kind, error.code, error.retryable),
        // HostRun gives only error messages here.
        _ => (ErrorKind::Internal, ErrorCode::InternalError, false),
    };
    let (kind, code) = (kind.as_str(), code.as_str());
    let err = CoreError::new_err(format!("{kind}: {code}"));
    let set = || -> PyResult<()> {
        let value = err.value(py);
        value.setattr("kind", kind)?;
        value.setattr("code", code)?;
        value.setattr("retryable", retryable)?;
        value.setattr("control", PyBytes::new(py, &out.control))?;
        value.setattr("attachments", attachments_to_python(py, out.attachments)?)?;
        Ok(())
    };
    match set() {
        Ok(()) => err,
        Err(failed) => failed,
    }
}

/// Reads one Arrow attachment through `__arrow_c_stream__`. A failure of
/// the producer, an exception from its `__arrow_c_stream__` or an error
/// its stream reports, is the provider's; a batch Arrow cannot build in
/// the stream's own schema, such as one with nulls in a column declared
/// without them, fails check (a). Nothing of either error's text is kept,
/// since it may quote values (decision H17).
fn read_arrow(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Result<ArrowData, Unreadable>> {
    let stream = match take_arrow_stream(value) {
        Ok(stream) => stream,
        // Not the data's: an interrupt goes on up.
        Err(error) if !error.is_instance_of::<PyException>(py) => return Err(error),
        Err(_) => return Ok(Err(Unreadable::Producer)),
    };
    Ok(read_arrow_stream(stream))
}

/// Moves the public capsule's stream into Arrow's RAII owner, exactly as
/// arrow-pyarrow does. No producer data is read or copied here.
#[allow(
    unsafe_code,
    reason = "the public Arrow capsule transfers ownership of a valid C stream"
)]
fn take_arrow_stream(value: &Bound<'_, PyAny>) -> PyResult<FFI_ArrowArrayStream> {
    let capsule = value
        .call_method0("__arrow_c_stream__")?
        .cast_into::<PyCapsule>()?;
    let pointer = capsule.pointer_checked(Some(c"arrow_array_stream"))?;
    let pointer = pointer.cast::<FFI_ArrowArrayStream>().as_ptr();
    if !pointer.is_aligned() {
        return Err(PyValueError::new_err("unaligned Arrow stream capsule"));
    }
    // SAFETY: the named Arrow capsule owns an initialized, aligned C stream.
    // from_raw moves it and marks the capsule's copy released, so its destructor
    // cannot release our stream a second time. Standard Arrow Drop owns it now.
    Ok(unsafe { FFI_ArrowArrayStream::from_raw(pointer) })
}

/// Checks raw buffer addresses before Arrow's importer can dereference typed
/// offsets or construct scalar slices. Arrow's own layout is the single source
/// of physical alignments; child and dictionary buffers are checked as well.
fn arrow_buffers_aligned(array: &FFI_ArrowArray, data_type: &DataType) -> bool {
    let mut pending = vec![(array, data_type)];
    while let Some((array, data_type)) = pending.pop() {
        if array.length < 0 || array.offset < 0 || array.n_buffers < 0 || array.n_children < 0 {
            return false;
        }
        let Some(extent) = array.len().checked_add(array.offset()) else {
            return false;
        };
        let physical = layout(data_type);
        let first = usize::from(physical.can_contain_null_mask);
        let minimum = first + physical.buffers.len() + usize::from(physical.variadic);
        if array.num_buffers() < minimum
            || (!physical.variadic && array.num_buffers() != minimum)
            || (array.num_buffers() != 0 && array.buffers.is_null())
        {
            return false;
        }
        let offsets = matches!(
            data_type,
            DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::List(_)
                | DataType::LargeList(_)
                | DataType::Map(_, _)
        );
        for (index, buffer) in physical.buffers.iter().enumerate() {
            // Offset arrays carry length+1 entries, including on an empty
            // array. Other zero-byte buffers may legally be dangling; the
            // Arrow importer replaces those without dereferencing them.
            if let BufferSpec::FixedWidth { alignment, .. } = buffer
                && (extent != 0 || (offsets && index == 0))
            {
                let pointer = array.buffer(first + index);
                if pointer.is_null() || pointer.align_offset(*alignment) != 0 {
                    return false;
                }
            }
        }
        if physical.variadic && array.num_buffers() > minimum {
            // The final C buffer stores i64 lengths for the variadic byte
            // buffers. The importer reads it before constructing ArrayData.
            let pointer = array.buffer(array.num_buffers() - 1);
            if pointer.is_null() || pointer.align_offset(std::mem::align_of::<i64>()) != 0 {
                return false;
            }
        }
        let children: Vec<&DataType> = match data_type {
            DataType::List(field)
            | DataType::LargeList(field)
            | DataType::FixedSizeList(field, _)
            | DataType::ListView(field)
            | DataType::LargeListView(field)
            | DataType::Map(field, _) => {
                vec![field.data_type()]
            }
            DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
            DataType::Union(fields, _) => {
                fields.iter().map(|(_, field)| field.data_type()).collect()
            }
            DataType::RunEndEncoded(ends, values) => vec![ends.data_type(), values.data_type()],
            _ => Vec::new(),
        };
        if array.num_children() != children.len()
            || (!children.is_empty() && array.children.is_null())
        {
            return false;
        }
        for (index, child_type) in children.into_iter().enumerate() {
            pending.push((array.child(index), child_type));
        }
        match (array.dictionary(), data_type) {
            (Some(dictionary), DataType::Dictionary(_, values)) => {
                pending.push((dictionary, values))
            }
            (None, DataType::Dictionary(_, _)) | (Some(_), _) => return false,
            _ => {}
        }
    }
    true
}

/// Standard C-stream consumption, with alignment checked before Arrow's typed
/// importer. FFI owners release streams and arrays on every early return; on
/// success Arrow's imported buffers retain the array owner without a copy.
#[allow(
    unsafe_code,
    reason = "Arrow's C callbacks and importer require valid owned FFI objects"
)]
fn read_arrow_stream(mut stream: FFI_ArrowArrayStream) -> Result<ArrowData, Unreadable> {
    if stream.release.is_none() {
        return Err(Unreadable::Producer);
    }
    let Some(get_schema) = stream.get_schema else {
        return Err(Unreadable::Producer);
    };
    let Some(get_next) = stream.get_next else {
        return Err(Unreadable::Producer);
    };
    let mut raw_schema = FFI_ArrowSchema::empty();
    // SAFETY: the owned live stream supplies its callback and initializes the
    // empty output according to the C stream ABI, as Arrow's reader requires.
    if unsafe { get_schema(&mut stream, &mut raw_schema) } != 0 {
        return Err(Unreadable::Producer);
    }
    let schema = Arc::new(Schema::try_from(&raw_schema).map_err(|_| Unreadable::Producer)?);
    let data_type = DataType::Struct(schema.fields().clone());
    let mut batches = Vec::new();
    loop {
        let mut array = FFI_ArrowArray::empty();
        // SAFETY: as for get_schema, the live stream initializes an empty owned
        // FFI output. Its release callback remains attached on every exit path.
        if unsafe { get_next(&mut stream, &mut array) } != 0 {
            return Err(Unreadable::Producer);
        }
        if array.is_released() {
            break;
        }
        if !arrow_buffers_aligned(&array, &data_type) {
            return Err(Unreadable::Declared);
        }
        // SAFETY: the producer's C array is valid for its declared schema and
        // owns its buffers. Every typed buffer in the complete array tree was
        // checked for alignment before this importer can read offsets or sizes.
        let data = unsafe { from_ffi_and_data_type(array, data_type.clone()) }.map_err(
            |error| match error {
                ArrowError::CDataInterface(_) => Unreadable::Producer,
                _ => Unreadable::Declared,
            },
        )?;
        let length = data.len();
        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            StructArray::from(data).into_parts().1,
            &RecordBatchOptions::new().with_row_count(Some(length)),
        )
        .map_err(|_| Unreadable::Declared)?;
        batches.push(batch);
    }
    Ok(ArrowData::new(schema, batches))
}

/// The attachments a Python caller sends, read with the GIL held: `bytes`,
/// or an object with `__arrow_c_stream__`, by id, with the Arrow ones that
/// could not be read named apart, so `sdm-host` checks the message's whole
/// set before it decides what an unreadable one means.
fn attachments_from_python(
    attachments: &Bound<'_, PyMapping>,
) -> PyResult<(Attachments, Unreadables)> {
    let py = attachments.py();
    let mut read: Attachments = BTreeMap::new();
    let mut unreadable = Unreadables::new();
    for item in attachments.items()?.try_iter()? {
        let (id, value): (u32, Bound<'_, PyAny>) = item?.extract()?;
        let attachment = if let Ok(bytes) = value.cast::<PyBytes>() {
            Attachment::Bytes(bytes.as_bytes().to_vec())
        } else if value.hasattr("__arrow_c_stream__")? {
            // Only the public capsule path, never pyarrow's private export.
            match read_arrow(py, &value)? {
                Ok(data) => Attachment::Arrow(data),
                Err(failure) => {
                    unreadable.insert(AttachmentId(id), failure);
                    continue;
                }
            }
        } else {
            return Err(PyTypeError::new_err(format!(
                "attachment {id} is a {}: an attachment is bytes or an object with __arrow_c_stream__",
                value.get_type().name()?
            )));
        };
        read.insert(AttachmentId(id), attachment);
    }
    Ok((read, unreadable))
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
        let (attachments, unreadable) = attachments_from_python(attachments)?;
        let message = message.to_vec();
        // A valid frame start copies foreign buffers before inference. Keep
        // the GIL until that copy finishes: dropping the GIL first would let
        // another Python thread mutate a NumPy producer while we borrow it.
        // The typed envelope is inspected only to choose ownership timing;
        // HostRun retains the sole manifest/attachment validation path.
        let frames = matches!(sdm_host::decode(&message), Ok(ControlMessage::Start(start))
        if match &start.operation {
            sdm_contract::ops::Operation::DescribeSource(request) => matches!(
                request.provider, sdm_contract::data::ProviderChoice::Frames { .. }),
            sdm_contract::ops::Operation::Materialize(request) => matches!(
                request.provider, sdm_contract::data::ProviderChoice::Frames { .. }),
            _ => false,
        });
        let result = if frames {
            core_result(
                py,
                HostRun::start_reading(&default_root(), &message, attachments, &unreadable),
            )?
        } else {
            core_call(py, || {
                HostRun::start_reading(&default_root(), &message, attachments, &unreadable)
            })?
        };
        result
            .map(|run| Run { run })
            .map_err(|out| core_error(py, out))
    }

    /// The run's next step, as `(control, attachments)`.
    fn next(&self, py: Python<'_>) -> PyResult<(Py<PyBytes>, Py<PyDict>)> {
        logging(py);
        match core_call(py, || self.run.next())? {
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
        // An Arrow attachment that could not be read still answers the
        // fetch it was sent for, as sdm-host decides (decision H16).
        let (attachments, unreadable) = attachments_from_python(attachments)?;
        let message = message.to_vec();
        core_call(py, || {
            self.run.reply_reading(&message, attachments, &unreadable)
        })?
        .map_err(|out| core_error(py, out))
    }

    /// Cancels the run; its attachments are released.
    fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        logging(py);
        core_call(py, || self.run.cancel())?.map_err(|out| core_error(py, out))
    }

    /// Takes an attachment the run made and a step named: `bytes`, an
    /// [`ArrowStream`], or `None` when the run holds no such id.
    fn take_attachment(&self, py: Python<'_>, id: u32) -> PyResult<Option<Py<PyAny>>> {
        logging(py);
        match core_call(py, || self.run.take_attachment(AttachmentId(id)))? {
            Ok(Some(attachment)) => to_python(py, attachment).map(Some),
            Ok(None) => Ok(None),
            Err(out) => Err(core_error(py, out)),
        }
    }

    /// The warnings given since the last call, as one `warnings` message
    /// `(control, attachments)`, or `None` when there are none.
    fn take_warnings(&self, py: Python<'_>) -> PyResult<Option<(Py<PyBytes>, Py<PyDict>)>> {
        logging(py);
        match core_call(py, || self.run.take_warnings())? {
            Ok(Some(out)) => outgoing(py, out).map(Some),
            Ok(None) => Ok(None),
            Err(out) => Err(core_error(py, out)),
        }
    }

    /// The sampler state the run left, as one `sampler_state` message
    /// `(control, attachments)`, or `None` when it left none (decision
    /// E52); its memo is an attachment the message names.
    fn take_sampler_state(&self, py: Python<'_>) -> PyResult<Option<(Py<PyBytes>, Py<PyDict>)>> {
        logging(py);
        match core_call(py, || self.run.take_sampler_state())? {
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
    // The databases the core provides for, the staging rule's private
    // marker (T2.1a); never public API.
    module.add("PROVIDERS", PyTuple::new(py, sdm_host::providers())?)?;
    module.add(
        "ERROR_CODES",
        PyTuple::new(py, ErrorCode::ALL.map(ErrorCode::as_str))?,
    )?;
    // Each meaning a column of each data type can carry, as the language
    // answers it, the client's check of a column's meaning (T3.16, rule 11).
    module.add(
        "MEANINGS",
        PyTuple::new(
            py,
            sdm_host::meanings()
                .into_iter()
                .map(|(stype, dtype)| PyTuple::new(py, [stype, dtype]))
                .collect::<PyResult<Vec<_>>>()?,
        )?,
    )?;
    // Each data type name a host may give a column, and the contract's name
    // of the type it names: the one table of them (T3.16).
    module.add(
        "DTYPE_NAMES",
        PyTuple::new(
            py,
            sdm_host::dtype_names()
                .into_iter()
                .map(|(name, dtype)| PyTuple::new(py, [name.to_owned(), dtype]))
                .collect::<PyResult<Vec<_>>>()?,
        )?,
    )?;
    module.add("CoreError", py.get_type::<CoreError>())?;
    module.add_class::<ArrowStream>()?;
    module.add_class::<Run>()?;
    module.add_function(wrap_pyfunction!(reset_log_cache, module)?)?;
    module.add_function(wrap_pyfunction!(templates, module)?)?;
    module.add_function(wrap_pyfunction!(is_posix_zone, module)?)?;
    module.add_function(wrap_pyfunction!(local_zone, module)?)?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    Ok(())
}
