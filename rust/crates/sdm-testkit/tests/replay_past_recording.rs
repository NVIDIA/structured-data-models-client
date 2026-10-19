// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Postgres replay asked a fetch past its recording (T2.8, from
//! `s0.boundary.depth-999@postgres` on `cross-host (postgres)`): for a
//! diagnostic case, which gates nothing, the replay answers that fetch with
//! a provider error naming it ("past the recording"), so the core refuses
//! the case and rust-host writes its line as a refusal, every core leg
//! observing every case the reference observed (the staging rule,
//! `run_python.py legs`), and the run goes on to the next case; an
//! acceptance case still fails the run by name, so a case that gates is
//! never judged on rows it was not recorded with.
//!
//! `sdm-contract-run --replay` runs three cases over
//! `tests/replay-past-recording.jsonl`, a recording of one fetch:
//! `t28.replay.answered`, whose one fetch (the describe's list of tables,
//! answered with none) is recorded, so it runs to its end and writes its
//! line; and two cases recorded with no fetch at all, which the core asks
//! one past at once: `t28.replay.diagnostic-past-its-recording` and
//! `t28.replay.acceptance-past-its-recording`. Each case is `s0.accept.
//! sum-window`'s, its id and status its own.
//!
//! - The diagnostic case before the answered one: the run succeeds and
//!   writes both lines, the diagnostic case's a refusal, `PROVIDER_ERROR`.
//! - The acceptance case before the answered one: the run fails, naming
//!   the case and the fetch past the recording, and writes no file.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: writes the cases and reads the run's output"
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// `s0.accept.sum-window`'s case as `id`, with `status`, written into
/// `dir`.
fn case(dir: &Path, id: &str, status: &str) -> PathBuf {
    let text = std::fs::read_to_string(root().join("contract/cases/s0/s0.accept.sum-window.json"))
        .unwrap();
    let mut case: Value = serde_json::from_str(&text).unwrap();
    case["id"] = Value::String(id.to_owned());
    case["status"] = Value::String(status.to_owned());
    let path = dir.join(format!("{id}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&case).unwrap()).unwrap();
    path
}

/// `sdm-contract-run --replay` over the recording and `cases`, in order,
/// writing to `out`.
fn replay(cases: &[PathBuf], out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_sdm-contract-run"))
        .arg("--replay")
        .arg(root().join("rust/crates/sdm-testkit/tests/replay-past-recording.jsonl"))
        .arg("--out")
        .arg(out)
        .args(cases)
        .current_dir(root())
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

/// The observation lines the run wrote at `out`.
fn lines(out: &Path) -> Vec<Value> {
    std::fs::read_to_string(out)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// The case keys of the observation lines the run wrote at `out`.
fn keys(out: &Path) -> Vec<String> {
    lines(out)
        .iter()
        .map(|o| o["case"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn a_diagnostic_case_past_its_recording_is_refused_and_an_acceptance_case_fails_the_run() {
    let dir = std::env::temp_dir().join(format!("t28-replay-past-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let answered = case(&dir, "t28.replay.answered", "acceptance");
    let diagnostic = case(
        &dir,
        "t28.replay.diagnostic-past-its-recording",
        "diagnostic",
    );
    let acceptance = case(
        &dir,
        "t28.replay.acceptance-past-its-recording",
        "acceptance",
    );

    // Control: a case whose fetches are all recorded runs to its end and
    // writes its line.
    let out = dir.join("answered.jsonl");
    let done = replay(std::slice::from_ref(&answered), &out);
    assert!(
        done.status.success(),
        "the recorded case alone: {}",
        String::from_utf8_lossy(&done.stderr)
    );
    assert_eq!(keys(&out), ["t28.replay.answered@postgres"]);

    // A diagnostic case past its recording is refused by the core, its
    // fetch answered with the replay's named provider error, and writes
    // its line as a refusal; the run goes on to the next case.
    let out = dir.join("diagnostic.jsonl");
    let done = replay(&[diagnostic, answered.clone()], &out);
    assert!(
        done.status.success(),
        "a diagnostic case past its recording failed the run: {}",
        String::from_utf8_lossy(&done.stderr)
    );
    assert_eq!(
        keys(&out),
        [
            "t28.replay.diagnostic-past-its-recording@postgres",
            "t28.replay.answered@postgres"
        ]
    );
    let refused = &lines(&out)[0]["outcome"];
    assert_eq!(
        (&refused["result"], &refused["code"]),
        (&Value::from("refusal"), &Value::from("PROVIDER_ERROR")),
        "the diagnostic case past its recording: {refused}"
    );

    // An acceptance case past its recording fails the run by name, and no
    // observation file is left.
    let out = dir.join("acceptance.jsonl");
    let done = replay(&[acceptance, answered], &out);
    let said = String::from_utf8_lossy(&done.stderr);
    assert!(
        !done.status.success(),
        "an acceptance case past its recording passed"
    );
    assert!(
        said.contains("t28.replay.acceptance-past-its-recording@postgres")
            && said.contains("past the 0 recorded"),
        "{said}"
    );
    assert!(!out.exists(), "a failed run left {}", out.display());

    std::fs::remove_dir_all(&dir).ok();
}
