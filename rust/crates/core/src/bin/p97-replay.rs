// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{Read, Write};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments != ["--format", "p97-snapshot-v1"] {
        return Err("expected --format p97-snapshot-v1".into());
    }
    let mut request = String::new();
    std::io::stdin().read_to_string(&mut request)?;
    let output = sdm_core::snapshot::replay_snapshot_json(&request)?;
    std::io::stdout().write_all(output.as_bytes())?;
    std::io::stdout().write_all(b"\n")?;
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
