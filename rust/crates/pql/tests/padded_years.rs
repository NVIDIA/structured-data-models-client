// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The date literals pandas reads differently on Linux than on macOS
//! (`padded_years.tsv`, recorded by `padded_years.py`): the core reads each
//! as pandas does where Python's `strftime` pads `%Y`, with the day-first
//! warning, and as it does where it does not, without it.

use sdm_pql::dates::{Host, to_datetime};

fn rows() -> Vec<(String, String, String)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/padded_years.tsv");
    let text = std::fs::read_to_string(&path).expect("padded_years.tsv");
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| {
            let cells: Vec<&str> = line.split('\t').collect();
            assert_eq!(cells.len(), 3, "{line}");
            (cells[0].into(), cells[1].into(), cells[2].into())
        })
        .collect()
}

#[test]
fn each_literal_reads_as_on_its_platform() {
    let rows = rows();
    assert!(rows.len() >= 800, "only {} rows", rows.len());
    let mut failures = Vec::new();
    for (literal, value, warning) in &rows {
        for pads_year in [true, false] {
            let reading = to_datetime(
                literal,
                Host {
                    year: 2026,
                    pads_year,
                },
            );
            let want = if pads_year {
                vec![warning.clone()]
            } else {
                Vec::new()
            };
            let got = reading.value.map(|s| s.iso());
            if got.as_deref() != Some(value.as_str()) || reading.warnings != want {
                failures.push(format!(
                    "{literal} (pads_year {pads_year}): {got:?} {:?}",
                    reading.warnings
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
