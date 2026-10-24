// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Java binding over sdm-host, the native library of `com.nvidia.sdm`
//! (design choice 10).
//!
//! A thin layer over [`HostRun`], as `sdm-py` is: control messages cross as
//! `byte[]` both ways, never as `String`, so no text meets JNI's modified
//! UTF-8; the attachments a message names cross beside it by id; an error
//! comes back as the `error` control message `HostRun` wrote, thrown as the
//! one `NativeError`. It makes no semantic call into Java.
//!
//! - Arrow crosses through the Arrow C Data interface only, with no copy:
//!   an Arrow attachment in is an `ArrowArrayStream` struct Java exported
//!   its data into and hands over by address, which this layer moves out
//!   of; one out is a [`NativeArrow`] handle, which Java exports into a
//!   struct of its own when it reads it ([`export_arrow`]).
//! - Every native resource has one owner and an explicit release: a run is
//!   a `Box<HostRun>` Java's `Run` owns and releases once
//!   ([`Java_com_nvidia_sdm_internal_NativeCore_release`]); an Arrow
//!   attachment out is a `Box<ArrowData>` Java's `NativeArrow` owns and
//!   releases once. Java guards each handle so no call reaches a released
//!   one.
//! - Every exported entry contains a panic: the core's own are contained by
//!   `sdm-host` (`contain`) and come back as `INTERNAL_ERROR` with
//!   `fields.panic`, and one in this layer is caught at the entry and
//!   thrown the same way, so no unwind crosses into the JVM.
//! - The core's `tracing` events reach SLF4J through the `log` facade
//!   ([`logging`]), installed when Java first loads the library.
//! - Loading the library starts no thread; it reads no environment
//!   variable.

mod logging;

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::ArrowError;
use jni::errors::ErrorPolicy;
use jni::objects::{Global, JByteArray, JClass, JIntArray, JLongArray, JObject, JObjectArray};
use jni::sys::{jboolean, jint, jlong};
use jni::{Env, EnvUnowned, JValue, jni_sig, jni_str};
use sdm_contract::data::AttachmentId;
use sdm_contract::{CONTRACT_VERSION, CoreError, Detail, ErrorCode, Scalar, Sentence};
use sdm_host::{
    ArrowData, Attachment, Attachments, HostRun, Outgoing, Unreadable, Unreadables, default_root,
    write_error,
};

/// The Java classes this layer makes objects of, handed over by
/// `NativeCore`'s static initializer ([`Java_com_nvidia_sdm_internal_NativeCore_init`])
/// so no lookup depends on a thread's class loader.
struct Classes {
    message: Global<JClass<'static>>,
    arrow: Global<JClass<'static>>,
    error: Global<JClass<'static>>,
}

static CLASSES: OnceLock<Classes> = OnceLock::new();

/// Why an entry did not answer.
enum Failure {
    /// The core's error, as the `error` control message it wrote.
    Core(Outgoing),
    /// A JNI call failed; the JVM holds the exception it raised, if any.
    Jni,
    /// Java called the entry wrongly: the attachment arrays disagree, or a
    /// handle is null. A defect of the Java host, thrown as
    /// `IllegalArgumentException`.
    Usage(&'static str),
}

impl From<jni::errors::Error> for Failure {
    fn from(_: jni::errors::Error) -> Self {
        Failure::Jni
    }
}

/// The `error` control message for a panic in this layer: `INTERNAL_ERROR`
/// with `fields.panic`, as `sdm-host` writes a contained one, with nothing
/// of the panic's message (decision H17).
fn panicked() -> Outgoing {
    let error = CoreError::new(
        Sentence::Internal,
        "the Java binding panicked; the call did not finish",
    )
    .with_field("panic", Detail::Scalar(Scalar::Bool(true)))
    .with_field("location", Detail::text("sdm-jni"));
    write_error(&error, || Ok(AttachmentId(1)))
}

/// Throws `NativeError` carrying `out`; `false` when it could not be made,
/// with whatever the JVM raised making it left pending.
fn throw_core(env: &mut Env<'_>, out: Outgoing) -> bool {
    let thrown = (|| -> Result<(), jni::errors::Error> {
        let classes = CLASSES
            .get()
            .ok_or(jni::errors::Error::NullPtr("classes"))?;
        let message = message_object(env, out)?;
        let error = env.new_object(
            &classes.error,
            jni_sig!("(Lcom/nvidia/sdm/internal/NativeMessage;)V"),
            &[JValue::Object(&message)],
        )?;
        let error = env.cast_local::<jni::objects::JThrowable>(error)?;
        env.throw(error)
    })();
    // `throw` reports the exception it raised as an error.
    env.exception_check() || thrown.is_ok()
}

/// The policy every entry resolves its outcome with: the core's error and
/// a panic as `NativeError`, a usage error as `IllegalArgumentException`,
/// and a JNI failure as the exception the JVM raised (or
/// `IllegalStateException`, when it raised none).
struct Raise;

impl<T: Default> ErrorPolicy<T, Failure> for Raise {
    type Captures<'unowned_env_local: 'native_method, 'native_method> = ();

    fn on_error<'unowned_env_local: 'native_method, 'native_method>(
        env: &mut Env<'unowned_env_local>,
        _captures: &mut Self::Captures<'unowned_env_local, 'native_method>,
        failure: Failure,
    ) -> jni::errors::Result<T> {
        match failure {
            Failure::Core(out) => {
                if !throw_core(env, out) {
                    let _ = env.throw_new(
                        jni::jni_str!("java/lang/IllegalStateException"),
                        jni::jni_str!("the core's error could not be raised"),
                    );
                }
            }
            Failure::Usage(what) => {
                if !env.exception_check() {
                    let _ = env.throw_new(
                        jni::jni_str!("java/lang/IllegalArgumentException"),
                        jni::strings::JNIString::from(what),
                    );
                }
            }
            Failure::Jni => {
                if !env.exception_check() {
                    let _ = env.throw_new(
                        jni::jni_str!("java/lang/IllegalStateException"),
                        jni::jni_str!("a JNI call in the Java binding failed"),
                    );
                }
            }
        }
        Ok(T::default())
    }

    fn on_panic<'unowned_env_local: 'native_method, 'native_method>(
        env: &mut Env<'unowned_env_local>,
        _captures: &mut Self::Captures<'unowned_env_local, 'native_method>,
        payload: Box<dyn std::any::Any + Send + 'static>,
    ) -> jni::errors::Result<T> {
        // The payload may quote input; it is dropped unread, and a panic in
        // its drop is caught and the payload leaked.
        if let Err(again) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(payload)))
        {
            std::mem::forget(again);
        }
        if env.exception_check() {
            env.exception_clear();
        }
        if !throw_core(env, panicked()) {
            let _ = env.throw_new(
                jni::jni_str!("java/lang/IllegalStateException"),
                jni::jni_str!("the Java binding panicked"),
            );
        }
        Ok(T::default())
    }
}

/// Runs an entry's body with its environment, contained, and resolves its
/// outcome with [`Raise`].
fn entry<'caller, T: Default>(
    unowned: &mut EnvUnowned<'caller>,
    body: impl FnOnce(&mut Env<'caller>) -> Result<T, Failure>,
) -> T {
    unowned.with_env(body).resolve::<Raise>()
}

/// A run's handle as Java holds it: the address of a `Box<HostRun>`.
fn run_of<'a>(handle: jlong) -> Result<&'a HostRun, Failure> {
    if handle == 0 {
        return Err(Failure::Usage("a released run"));
    }
    #[allow(unsafe_code)]
    // SAFETY: a non-zero handle is one `start` returned from
    // `Box::into_raw`, and Java's `Run` passes it only while it holds it:
    // its close waits for every call in flight, then releases it once and
    // passes it no more (one owner, an explicit release).
    let run = unsafe { &*(handle as *const HostRun) };
    Ok(run)
}

/// The Arrow data a `NativeArrow` handle holds: the address of a
/// `Box<ArrowData>`.
fn arrow_of<'a>(handle: jlong) -> Result<&'a ArrowData, Failure> {
    if handle == 0 {
        return Err(Failure::Usage("a released Arrow attachment"));
    }
    #[allow(unsafe_code)]
    // SAFETY: as for `run_of`: a non-zero handle is a `Box<ArrowData>` this
    // layer made, which Java's `NativeArrow` owns and releases once.
    let data = unsafe { &*(handle as *const ArrowData) };
    Ok(data)
}

/// Reads the Arrow stream Java exported into the struct at `address`,
/// moving it out, so the struct is left released and Java's close of it
/// releases nothing twice. The batches keep Java's buffers, with no copy,
/// and release them through the stream's callbacks when dropped. A failure
/// of the producer is the provider's; a batch Arrow cannot build in the
/// stream's own schema fails check (a); neither keeps any text (H17).
fn read_stream(address: jlong) -> Result<Result<ArrowData, Unreadable>, Failure> {
    if address == 0 {
        return Err(Failure::Usage(
            "an Arrow attachment's stream address is null",
        ));
    }
    #[allow(unsafe_code)]
    // SAFETY: Java hands over the address of an `ArrowArrayStream` struct
    // it allocated and exported its data into (Arrow Java's C Data
    // module), live for this call; `from_raw` moves the struct out and
    // leaves it marked released, as the C stream interface's move says.
    let stream = unsafe { FFI_ArrowArrayStream::from_raw(address as *mut FFI_ArrowArrayStream) };
    let reader = match ArrowArrayStreamReader::try_new(stream) {
        Ok(reader) => reader,
        Err(_) => return Ok(Err(Unreadable::Producer)),
    };
    let schema = reader.schema();
    let mut batches = Vec::new();
    for batch in reader {
        match batch {
            Ok(batch) => batches.push(batch),
            Err(ArrowError::CDataInterface(_)) => return Ok(Err(Unreadable::Producer)),
            Err(_) => return Ok(Err(Unreadable::Declared)),
        }
    }
    Ok(Ok(ArrowData::new(schema, batches)))
}

/// The attachments a Java caller sends: `ids[i]` names `bytes[i]` when it
/// is not null, else the Arrow stream at `streams[i]`. The Arrow ones that
/// could not be read are named apart, so `sdm-host` checks the message's
/// whole set before it decides what an unreadable one means.
fn attachments_from_java(
    env: &mut Env<'_>,
    ids: &JIntArray<'_>,
    bytes: &JObjectArray<'_, JByteArray<'_>>,
    streams: &JLongArray<'_>,
) -> Result<(Attachments, Unreadables), Failure> {
    let count = ids.len(env)?;
    if bytes.len(env)? != count || streams.len(env)? != count {
        return Err(Failure::Usage(
            "an attachment's id, bytes and stream arrays differ in length",
        ));
    }
    let mut id_values = vec![0; count];
    ids.get_region(env, 0, &mut id_values)?;
    let mut addresses = vec![0; count];
    streams.get_region(env, 0, &mut addresses)?;
    let mut read: Attachments = BTreeMap::new();
    let mut unreadable = Unreadables::new();
    for (index, (&id, &address)) in id_values.iter().zip(&addresses).enumerate() {
        let id = u32::try_from(id).map_err(|_| Failure::Usage("a negative attachment id"))?;
        let id = AttachmentId(id);
        let element = bytes.get_element(env, index)?;
        let attachment = if !element.is_null() {
            if address != 0 {
                return Err(Failure::Usage("an attachment is both bytes and a stream"));
            }
            Attachment::Bytes(env.convert_byte_array(&element)?)
        } else {
            match read_stream(address)? {
                Ok(data) => Attachment::Arrow(data),
                Err(failure) => {
                    unreadable.insert(id, failure);
                    continue;
                }
            }
        };
        if read.insert(id, attachment).is_some() {
            return Err(Failure::Usage("two attachments share an id"));
        }
    }
    Ok((read, unreadable))
}

/// One attachment as Java sees it: `byte[]`, or a `NativeArrow` owning the
/// data.
fn attachment_object<'local>(
    env: &mut Env<'local>,
    attachment: Attachment,
) -> Result<JObject<'local>, jni::errors::Error> {
    Ok(match attachment {
        Attachment::Bytes(bytes) => env.byte_array_from_slice(&bytes)?.into(),
        Attachment::Arrow(data) => {
            let classes = CLASSES
                .get()
                .ok_or(jni::errors::Error::NullPtr("classes"))?;
            let handle = Box::into_raw(Box::new(data));
            match env.new_object(
                &classes.arrow,
                jni_sig!("(J)V"),
                &[JValue::Long(handle as jlong)],
            ) {
                Ok(object) => object,
                Err(error) => {
                    #[allow(unsafe_code)]
                    // SAFETY: `handle` was made just above and no Java
                    // object took it.
                    drop(unsafe { Box::from_raw(handle) });
                    return Err(error);
                }
            }
        }
    })
}

/// A control message the core wrote, as a `NativeMessage`: the control
/// bytes, and its attachments by id.
fn message_object<'local>(
    env: &mut Env<'local>,
    out: Outgoing,
) -> Result<JObject<'local>, jni::errors::Error> {
    let classes = CLASSES
        .get()
        .ok_or(jni::errors::Error::NullPtr("classes"))?;
    let control = env.byte_array_from_slice(&out.control)?;
    let count = out.attachments.len();
    let ids = env.new_int_array(count)?;
    let values = JObjectArray::<JObject>::new(env, count, JObject::null())?;
    let mut id_values = Vec::with_capacity(count);
    for (index, (id, attachment)) in out.attachments.into_iter().enumerate() {
        id_values.push(jint::try_from(id.0).map_err(|_| jni::errors::Error::NullPtr("id"))?);
        let value = attachment_object(env, attachment)?;
        values.set_element(env, index, &value)?;
        env.delete_local_ref(value);
    }
    ids.set_region(env, 0, &id_values)?;
    env.new_object(
        &classes.message,
        jni_sig!("([B[I[Ljava/lang/Object;)V"),
        &[
            JValue::Object(&control),
            JValue::Object(&ids),
            JValue::Object(&values),
        ],
    )
}

/// Takes the Java classes this layer makes objects of, and installs the
/// logging bridge to `log`'s class; called once, from `NativeCore`'s
/// static initializer.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_init<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    message: JClass<'caller>,
    arrow: JClass<'caller>,
    error: JClass<'caller>,
    log: JClass<'caller>,
) {
    entry(&mut unowned, |env| {
        let classes = Classes {
            message: env.new_global_ref(&message)?,
            arrow: env.new_global_ref(&arrow)?,
            error: env.new_global_ref(&error)?,
        };
        // A second load of the library in the same class loader finds them
        // set; the first's stand.
        let _ = CLASSES.set(classes);
        logging::install(env, &log)?;
        Ok(())
    });
}

/// What the library is, as JSON bytes: the contract version, whether the
/// test-only operations are built in, the databases the core provides for
/// (the staging rule's private marker; never public API), and the error
/// codes the core gives.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_info0<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
) -> JByteArray<'caller> {
    entry(&mut unowned, |env| {
        let info = serde_json::json!({
            "contract": CONTRACT_VERSION,
            "test_ops": cfg!(feature = "test-ops"),
            "providers": sdm_host::providers(),
            "error_codes": ErrorCode::ALL.map(ErrorCode::as_str),
            // Each data type's name on the client's wire, by the contract's
            // name of it: what a schema holds a described column's type as.
            "wire_names": sdm_host::wire_names()
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>(),
        });
        let bytes =
            serde_json::to_vec(&info).map_err(|_| Failure::Usage("the library's description"))?;
        Ok(env.byte_array_from_slice(&bytes)?)
    })
}

/// The table of what every error's and warning's sentence says, as its
/// JSON bytes (`sdm_host::templates`, T3.17), the twin of `sdm-py`'s
/// `templates()`: the host reads it once and fills each sentence from it.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_templates0<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
) -> JByteArray<'caller> {
    entry(&mut unowned, |env| {
        Ok(env.byte_array_from_slice(sdm_host::templates().as_bytes())?)
    })
}

/// Whether the UTF-8 `text` is a POSIX `TZ` string a run takes as its
/// local zone, as `HostRun::start` checks it.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_isPosixZone<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    text: JByteArray<'caller>,
) -> jboolean {
    entry(&mut unowned, |env| {
        let bytes = env.convert_byte_array(&text)?;
        Ok(std::str::from_utf8(&bytes).is_ok_and(sdm_host::is_posix_zone))
    })
}

/// The machine's files as Java reads them, for the local zone's rule:
/// `NativeCore.isRegularFile` and `NativeCore.realPath` (`Files.isRegularFile`
/// and `Path.toRealPath`), so the binding does no I/O of its own (T3.12).
/// A JNI failure is kept, and the entry fails with it after the rule.
struct JavaZoneFiles<'local, 'caller> {
    env: std::cell::RefCell<&'local mut Env<'caller>>,
    class: &'local JClass<'caller>,
    failed: std::cell::Cell<bool>,
}

impl JavaZoneFiles<'_, '_> {
    fn path_bytes(path: &std::path::Path) -> Vec<u8> {
        path.to_string_lossy().into_owned().into_bytes()
    }
}

impl sdm_host::ZoneFiles for JavaZoneFiles<'_, '_> {
    fn is_file(&self, path: &std::path::Path) -> bool {
        let mut env = self.env.borrow_mut();
        let answer = (|| -> jni::errors::Result<bool> {
            let bytes = env.byte_array_from_slice(&Self::path_bytes(path))?;
            env.call_static_method(
                self.class,
                jni_str!("isRegularFile"),
                jni_sig!("([B)Z"),
                &[JValue::Object(&bytes)],
            )?
            .z()
        })();
        answer.unwrap_or_else(|_| {
            self.failed.set(true);
            false
        })
    }

    fn real_path(&self, path: &std::path::Path) -> Option<std::path::PathBuf> {
        let mut env = self.env.borrow_mut();
        let answer = (|| -> jni::errors::Result<Option<Vec<u8>>> {
            let bytes = env.byte_array_from_slice(&Self::path_bytes(path))?;
            let found = env
                .call_static_method(
                    self.class,
                    jni_str!("realPath"),
                    jni_sig!("([B)[B"),
                    &[JValue::Object(&bytes)],
                )?
                .l()?;
            if found.is_null() {
                return Ok(None);
            }
            let found = env.cast_local::<JByteArray>(found)?;
            Ok(Some(env.convert_byte_array(&found)?))
        })();
        match answer {
            Ok(found) => found
                .and_then(|bytes| String::from_utf8(bytes).ok())
                .map(Into::into),
            Err(_) => {
                self.failed.set(true);
                None
            }
        }
    }
}

/// The local zone a host's facts name, by the one rule every host reads it
/// with (`sdm_host::local_zone`, T3.12), the twin of `sdm-py`'s
/// `local_zone`: `setting` is `TZ`'s UTF-8 bytes, or null when unset, and
/// `roots` the JSON array of the zone lookup roots' paths in the host's
/// order; the JSON bytes of `{"iana": name}`, `{"posix": text}` or `null`.
/// A setting that is not UTF-8 names no zone. The rule's questions of the
/// machine's files are answered by Java ([`JavaZoneFiles`]).
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_localZone0<'caller>(
    mut unowned: EnvUnowned<'caller>,
    class: JClass<'caller>,
    setting: JByteArray<'caller>,
    roots: JByteArray<'caller>,
) -> JByteArray<'caller> {
    entry(&mut unowned, |env| {
        let roots = env.convert_byte_array(&roots)?;
        let roots: Vec<String> =
            serde_json::from_slice(&roots).map_err(|_| Failure::Usage("the zone lookup roots"))?;
        let roots: Vec<std::path::PathBuf> = roots.into_iter().map(Into::into).collect();
        let setting = if setting.is_null() {
            None
        } else {
            Some(env.convert_byte_array(&setting)?)
        };
        let zone = {
            let files = JavaZoneFiles {
                env: std::cell::RefCell::new(&mut *env),
                class: &class,
                failed: std::cell::Cell::new(false),
            };
            let zone = match setting.as_deref().map(std::str::from_utf8) {
                None => sdm_host::local_zone(None, &roots, &files),
                Some(Ok(text)) => sdm_host::local_zone(Some(text), &roots, &files),
                Some(Err(_)) => None,
            };
            if files.failed.get() {
                return Err(Failure::Jni);
            }
            zone
        };
        let bytes = serde_json::to_vec(&zone).map_err(|_| Failure::Usage("the local zone"))?;
        Ok(env.byte_array_from_slice(&bytes)?)
    })
}

/// What each model supports, as the client describes it to a caller, as
/// the core's JSON bytes (`sdm_host::capabilities`, T3.12), the twin of
/// `sdm-py`'s `capabilities()`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_capabilities0<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
) -> JByteArray<'caller> {
    entry(&mut unowned, |env| {
        Ok(env.byte_array_from_slice(sdm_host::capabilities().as_bytes())?)
    })
}

/// A Postgres `json` or `jsonb` value, given as the server's UTF-8 text, as
/// the reference's provider hands it to the core
/// (`sdm_host::json_value_text`, T3.12 fix 1): its UTF-8 text, or null
/// for a JSON null; an exception for text that is not JSON.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_jsonText0<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    text: JByteArray<'caller>,
) -> JByteArray<'caller> {
    entry(&mut unowned, |env| {
        let given = env.convert_byte_array(&text)?;
        let given = String::from_utf8(given)
            .map_err(|_| Failure::Usage("a JSON value is not UTF-8 text"))?;
        match sdm_host::json_value_text(&given) {
            Some(Some(written)) => Ok(env.byte_array_from_slice(written.as_bytes())?),
            Some(None) => Ok(JByteArray::default()),
            None => Err(Failure::Usage("a JSON value the server wrote is not JSON")),
        }
    })
}

/// Sets the most verbose level the logging bridge passes on: 0 none, 1
/// error, 2 warn, 3 info, 4 debug, 5 trace. Java sets it at each run start
/// from SLF4J's configuration, so a change there takes effect.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_setLogLevel<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    level: jint,
) {
    entry(&mut unowned, |_env| {
        logging::set_level(level);
        Ok(())
    });
}

/// Starts the run a `start` control message asks for, with the attachments
/// it names; the run's handle, which the caller owns and releases once.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_start<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    message: JByteArray<'caller>,
    ids: JIntArray<'caller>,
    bytes: JObjectArray<'caller, JByteArray<'caller>>,
    streams: JLongArray<'caller>,
) -> jlong {
    entry(&mut unowned, |env| {
        let message = env.convert_byte_array(&message)?;
        let (attachments, unreadable) = attachments_from_java(env, &ids, &bytes, &streams)?;
        let run = HostRun::start_reading(&default_root(), &message, attachments, &unreadable)
            .map_err(Failure::Core)?;
        Ok(Box::into_raw(Box::new(run)) as jlong)
    })
}

/// The run's next step, as a `NativeMessage`.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_next<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
) -> JObject<'caller> {
    entry(&mut unowned, |env| {
        let out = run_of(run)?.next().map_err(Failure::Core)?;
        Ok(message_object(env, out)?)
    })
}

/// Hands the run a `fetch_reply` or `model_reply`, with the attachments it
/// names. An Arrow attachment that could not be read still answers the
/// fetch it was sent for, as sdm-host decides (decision H16).
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_reply<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
    message: JByteArray<'caller>,
    ids: JIntArray<'caller>,
    bytes: JObjectArray<'caller, JByteArray<'caller>>,
    streams: JLongArray<'caller>,
) {
    entry(&mut unowned, |env| {
        let run = run_of(run)?;
        let message = env.convert_byte_array(&message)?;
        let (attachments, unreadable) = attachments_from_java(env, &ids, &bytes, &streams)?;
        run.reply_reading(&message, attachments, &unreadable)
            .map_err(Failure::Core)
    });
}

/// Cancels the run; its attachments are released.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_cancel<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
) {
    entry(&mut unowned, |_env| {
        run_of(run)?.cancel().map_err(Failure::Core)
    });
}

/// Takes an attachment the run made and a step named: `byte[]`, a
/// `NativeArrow`, or null when the run holds no such id.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_takeAttachment<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
    id: jint,
) -> JObject<'caller> {
    entry(&mut unowned, |env| {
        let id = u32::try_from(id).map_err(|_| Failure::Usage("a negative attachment id"))?;
        match run_of(run)?
            .take_attachment(AttachmentId(id))
            .map_err(Failure::Core)?
        {
            Some(attachment) => Ok(attachment_object(env, attachment)?),
            None => Ok(JObject::null()),
        }
    })
}

/// The warnings given since the last call, as one `warnings` message, or
/// null when there are none.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_takeWarnings<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
) -> JObject<'caller> {
    entry(&mut unowned, |env| {
        match run_of(run)?.take_warnings().map_err(Failure::Core)? {
            Some(out) => Ok(message_object(env, out)?),
            None => Ok(JObject::null()),
        }
    })
}

/// The sampler state the run left, as one `sampler_state` message, or null
/// when it left none (decision E52).
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_takeSamplerState<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
) -> JObject<'caller> {
    entry(&mut unowned, |env| {
        match run_of(run)?.take_sampler_state().map_err(Failure::Core)? {
            Some(out) => Ok(message_object(env, out)?),
            None => Ok(JObject::null()),
        }
    })
}

/// Releases a run: its handle is invalid from here on. Java's `Run` calls
/// it once, after every call in flight has returned.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_release<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    run: jlong,
) {
    entry(&mut unowned, |_env| {
        if run != 0 {
            // SAFETY: as for `run_of`; this is the handle's one release.
            drop(unsafe { Box::from_raw(run as *mut HostRun) });
        }
        Ok(())
    });
}

/// Exports the Arrow data a `NativeArrow` holds into the `ArrowArrayStream`
/// struct at `address`, which Java allocated empty and imports from next.
/// The batches are shared, not copied; the data stays the handle's too.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_exportArrow<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    arrow: jlong,
    address: jlong,
) {
    entry(&mut unowned, |_env| {
        let data = arrow_of(arrow)?;
        if address == 0 {
            return Err(Failure::Usage("the stream's address is null"));
        }
        export_arrow(data, address as *mut FFI_ArrowArrayStream);
        Ok(())
    });
}

/// Writes a stream of `data`'s batches into `target`, sharing their
/// buffers.
fn export_arrow(data: &ArrowData, target: *mut FFI_ArrowArrayStream) {
    let batches: Vec<Result<RecordBatch, ArrowError>> =
        data.batches.iter().cloned().map(Ok).collect();
    let reader: Box<dyn RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(batches, Arc::clone(&data.schema)));
    #[allow(unsafe_code)]
    // SAFETY: `target` is the address of an `ArrowArrayStream` struct Java
    // allocated, empty (released), live for this call; writing over it
    // drops nothing, and Java's import moves the stream out and releases
    // it through its callback.
    unsafe {
        std::ptr::write(target, FFI_ArrowArrayStream::new(reader));
    }
}

/// Releases a `NativeArrow`'s data: its handle is invalid from here on.
#[allow(unsafe_code)]
#[unsafe(no_mangle)]
pub extern "system" fn Java_com_nvidia_sdm_internal_NativeCore_releaseArrow<'caller>(
    mut unowned: EnvUnowned<'caller>,
    _class: JClass<'caller>,
    arrow: jlong,
) {
    entry(&mut unowned, |_env| {
        if arrow != 0 {
            // SAFETY: as for `arrow_of`; this is the handle's one release.
            drop(unsafe { Box::from_raw(arrow as *mut ArrowData) });
        }
        Ok(())
    });
}
