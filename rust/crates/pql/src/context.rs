// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What one run shares, as Python's warnings module and `dateutil`'s parser
//! info share it: the Python process dates are read as in, and the
//! `UserWarning`s reading them gives.
//!
//! [`run`] is the Rust counterpart of `warnings.catch_warnings(record=True)`:
//! it collects the warnings of everything it calls, and hands them back.
//! [`crate::validate`] runs inside one of its own; call [`run`] directly
//! only to collect the warnings of a resolved query's helpers that read
//! dates afterwards, such as `get_rfm_entity_id_list`.

use std::cell::RefCell;

use crate::dates::Host;

struct Run {
    host: Host,
    warnings: Vec<String>,
}

thread_local! {
    static RUN: RefCell<Option<Run>> = const { RefCell::new(None) };
}

/// Puts back the run that was current when [`run`] began, however `work`
/// ends, so that runs nest and a panic leaves nothing behind.
struct Restore(Option<Run>);

impl Drop for Restore {
    fn drop(&mut self) {
        let outer = self.0.take();
        RUN.with(|r| *r.borrow_mut() = outer);
    }
}

/// Calls `work` with dates read as in `host`, and returns what it returns
/// with the warnings it gave, in order. A run inside another collects its
/// own warnings, which the outer run does not see.
pub fn run<R>(host: Host, work: impl FnOnce() -> R) -> (R, Vec<String>) {
    let outer = RUN.with(|r| {
        r.borrow_mut().replace(Run {
            host,
            warnings: Vec::new(),
        })
    });
    let _restore = Restore(outer);
    let out = work();
    let warnings = RUN.with(|r| {
        r.borrow_mut()
            .as_mut()
            .map(|run| std::mem::take(&mut run.warnings))
            .unwrap_or_default()
    });
    (out, warnings)
}

/// The Python process dates are read as in: the current run's, else
/// [`Host::assumed`].
pub fn host() -> Host {
    RUN.with(|r| r.borrow().as_ref().map(|run| run.host))
        .unwrap_or_else(Host::assumed)
}

/// A `UserWarning`, as Python's `warnings.warn` would give it. Outside a
/// [`run`] nobody is collecting, and it is dropped.
pub fn warn(message: String) {
    RUN.with(|r| {
        if let Some(run) = r.borrow_mut().as_mut() {
            run.warnings.push(message);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(year: i64) -> Host {
        Host {
            year,
            pads_year: false,
        }
    }

    #[test]
    fn a_run_collects_its_warnings_and_leaves_nothing_behind() {
        let (year, warnings) = run(host(1999), || {
            warn("one".to_string());
            warn("two".to_string());
            super::host().year
        });
        assert_eq!(year, 1999);
        assert_eq!(warnings, ["one", "two"]);
        assert!(RUN.with(|r| r.borrow().is_none()));
        warn("dropped".to_string());
        let ((), warnings) = run(host(1999), || {});
        assert!(warnings.is_empty());
    }

    #[test]
    fn runs_nest() {
        let ((inner_year, inner), outer) = run(host(1999), || {
            warn("outer".to_string());
            let inner = run(host(2042), || {
                warn("inner".to_string());
                super::host().year
            });
            assert_eq!(super::host().year, 1999);
            inner
        });
        assert_eq!(inner_year, 2042);
        assert_eq!(inner, ["inner"]);
        assert_eq!(outer, ["outer"]);
    }

    #[test]
    fn a_panic_ends_the_run() {
        let caught = std::panic::catch_unwind(|| {
            run(host(1999), || {
                warn("lost".to_string());
                panic!("inside a run");
            })
        });
        assert!(caught.is_err());
        assert!(RUN.with(|r| r.borrow().is_none()));
    }
}
