// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm-contract-run`: the pure-Rust host over contract cases.
//!
//! ```text
//! sdm-contract-run --out FILE CASE_OR_DIRECTORY...
//! ```
//!
//! Runs every case through `HostRun` and writes one observation per case,
//! one JSON object per line, for `scripts/rust_core/compare.py`. A runner
//! failure (a case it cannot load or run, a step S0 never takes) exits
//! non-zero and leaves no output file, an earlier run's included;
//! observations are written only when every case was recorded. The
//! identity's commit is the checkout the command runs in.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Result, bail};
use sdm_testkit::{case, host, observation};

fn run(arguments: &[String]) -> Result<(PathBuf, usize)> {
    let (out, cases) = match arguments {
        [flag, out, cases @ ..] if flag == "--out" && !cases.is_empty() => (
            PathBuf::from(out),
            cases.iter().map(PathBuf::from).collect::<Vec<_>>(),
        ),
        _ => bail!("usage: sdm-contract-run --out FILE CASE_OR_DIRECTORY..."),
    };
    let mut partial = out.clone().into_os_string();
    partial.push(".partial");
    // A failed run leaves no output, not an earlier run's.
    for stale in [out.as_os_str(), partial.as_os_str()] {
        match std::fs::remove_file(stale) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(error.into());
            }
            _ => {}
        }
    }
    let loaded = case::load(&cases)?;
    if loaded.is_empty() {
        bail!("no cases");
    }
    let identity = observation::Identity::new(&std::env::current_dir()?)?;
    let mut lines = String::new();
    for each in &loaded {
        let observed = host::run(&each.case)?;
        lines.push_str(&observation::write(&identity, each, &observed).to_string());
        lines.push('\n');
    }
    std::fs::write(&partial, lines)?;
    std::fs::rename(&partial, &out)?;
    Ok((out, loaded.len()))
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match run(&arguments) {
        Ok((_, count)) => {
            println!("rust-host: {count} cases recorded");
            ExitCode::SUCCESS
        }
        Err(failure) => {
            eprintln!("runner failure: {failure:#}");
            ExitCode::FAILURE
        }
    }
}
