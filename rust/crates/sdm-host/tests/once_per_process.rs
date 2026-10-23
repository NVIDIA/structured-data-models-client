// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The warnings the reference gives once in a process (decision E52): the
//! core gives `CONTEXT_COVERAGE` on every run that meets it, holding no
//! state across runs, and `HostRun::take_warnings`, which every binding
//! takes its warnings through, gives it the first time a run of the
//! process gives it and drops it after, any other warning untouched. One
//! test, in a file of its own, so its process holds no other run.

use std::sync::Arc;

use sdm_contract::{ControlMessage, CoreError, CoreWarning, Sentence};
use sdm_core::Run;
use sdm_core::materialize::{CONTEXT_COVERAGE, MISSING_INDICES, ONCE_PER_PROCESS};
use sdm_host::{HostRun, ZoneResolver, read};

/// A run giving `codes`, in order, then failing, as a refused
/// `materialize` that met them does.
fn giving(codes: &'static [&'static str]) -> HostRun {
    let run = Run::new(move |asker| async move {
        for code in codes {
            // A sentence of the code, the first the table lists.
            let sentence = Sentence::ALL
                .iter()
                .find(|s| s.code().as_str() == *code)
                .expect("a sentence of the code");
            asker.warn(CoreWarning::new(*sentence, "given"));
        }
        Err(CoreError::new(
            Sentence::DataRequestRefused,
            "refused after its warnings",
        ))
    });
    HostRun::with(run, Arc::new(ZoneResolver::new([])))
}

/// The codes of the warnings the host gives for a run of `codes`.
fn shown(codes: &'static [&'static str]) -> Vec<String> {
    let run = giving(codes);
    assert!(run.next().is_err());
    let Some(out) = run.take_warnings().unwrap() else {
        return vec![];
    };
    let ControlMessage::Warnings { warnings } = read(&out).unwrap() else {
        panic!("not warnings");
    };
    warnings.into_iter().map(|w| w.code).collect()
}

#[test]
fn a_once_a_process_warning_is_given_once_in_the_process() {
    assert_eq!(ONCE_PER_PROCESS, [CONTEXT_COVERAGE]);
    let both: &'static [&'static str] = &[MISSING_INDICES, CONTEXT_COVERAGE];
    assert_eq!(shown(both), [MISSING_INDICES, CONTEXT_COVERAGE]);
    // A second run, and a third: the coverage warning is dropped, the
    // other given as before.
    assert_eq!(shown(both), [MISSING_INDICES]);
    assert_eq!(
        shown(&[CONTEXT_COVERAGE, MISSING_INDICES]),
        [MISSING_INDICES]
    );
    assert!(shown(&[CONTEXT_COVERAGE]).is_empty());
}
