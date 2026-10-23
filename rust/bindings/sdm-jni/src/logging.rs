// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bridge from the core's `tracing` events to SLF4J (design choice 10,
//! KO3-30), as `pyo3-log` is Python's: `tracing`'s `log` feature writes each
//! event to the `log` facade, and this logger hands it to
//! `com.nvidia.sdm.internal.NativeLog.log(level, target, message)`, which
//! writes it to the SLF4J logger the target names. The JAR declares the
//! SLF4J 2 API and binds no implementation: the application chooses one.
//!
//! The level filter is `log`'s own, set from Java at each run start
//! ([`set_level`]), so an event SLF4J would drop never crosses JNI. A
//! failure to log (the class gone, an exception in a handler) is cleared
//! and dropped, never raised into the core's call.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use jni::objects::{Global, JClass, JValue};
use jni::vm::JavaVM;
use jni::{Env, jni_sig, jni_str};
use log::{Level, LevelFilter, Log, Metadata, Record};

/// Where events go: the class holding `log`, kept from the thread that
/// loaded the library.
struct JavaLog {
    class: Global<JClass<'static>>,
}

static TARGET: OnceLock<JavaLog> = OnceLock::new();

/// Whether this logger is the `log` facade's: false when another logger
/// in the process took it first, which then gets the events.
static INSTALLED: AtomicBool = AtomicBool::new(false);

struct Bridge;

static BRIDGE: Bridge = Bridge;

/// Keeps `class` as the target of events and makes this the `log`
/// facade's logger, off until Java sets a level.
pub(crate) fn install(env: &mut Env<'_>, class: &JClass<'_>) -> jni::errors::Result<()> {
    let target = JavaLog {
        class: env.new_global_ref(class)?,
    };
    if TARGET.set(target).is_ok() && log::set_logger(&BRIDGE).is_ok() {
        INSTALLED.store(true, Ordering::Release);
        log::set_max_level(LevelFilter::Off);
    }
    Ok(())
}

/// Sets the most verbose level passed on (0 none to 5 trace); nothing
/// when another logger holds the facade.
pub(crate) fn set_level(level: i32) {
    if !INSTALLED.load(Ordering::Acquire) {
        return;
    }
    log::set_max_level(match level {
        1 => LevelFilter::Error,
        2 => LevelFilter::Warn,
        3 => LevelFilter::Info,
        4 => LevelFilter::Debug,
        5 => LevelFilter::Trace,
        _ => LevelFilter::Off,
    });
}

fn number(level: Level) -> i32 {
    match level {
        Level::Error => 1,
        Level::Warn => 2,
        Level::Info => 3,
        Level::Debug => 4,
        Level::Trace => 5,
    }
}

impl Log for Bridge {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let (Some(target), Ok(vm)) = (TARGET.get(), JavaVM::singleton()) else {
            return;
        };
        // The thread is the caller's, attached already; another is attached
        // for the call. An exception Java raised is caught by the
        // attachment and dropped here.
        let _ = vm.attach_current_thread(|env| -> jni::errors::Result<()> {
            let name = env.new_string(record.target())?;
            let text = env.new_string(record.args().to_string())?;
            env.call_static_method(
                &target.class,
                jni_str!("log"),
                jni_sig!("(ILjava/lang/String;Ljava/lang/String;)V"),
                &[
                    JValue::Int(number(record.level())),
                    JValue::Object(&name),
                    JValue::Object(&text),
                ],
            )?;
            Ok(())
        });
    }

    fn flush(&self) {}
}
