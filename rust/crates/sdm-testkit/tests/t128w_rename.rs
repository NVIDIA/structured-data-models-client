// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T1.28w exit 4, held from outside the implementation (the package's
//! independent test author): the parser crate is `sdm-pql`, and nothing
//! in the code or the manifests names its interim name. The docs may (the
//! rename's history); every other tracked kind of file may not.

use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The interim name, in both spellings, built so this file does not name
/// it.
fn interim() -> [String; 2] {
    let stem = ["sdm", "pql", "interim"];
    [stem.join("-"), stem.join("_")]
}

/// The files to search: code, manifests, guard data and CI, below `root`;
/// the docs, build output and environments left out.
fn files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if matches!(
                name.as_str(),
                "docs" | "target" | ".git" | ".venv" | "venv" | "node_modules" | "__pycache__"
            ) || name.starts_with('.') && name != ".github"
            {
                continue;
            }
            files(&path, out);
        } else if [
            ".rs", ".toml", ".lock", ".py", ".sh", ".yml", ".yaml", ".json", ".txt", ".cfg",
        ]
        .iter()
        .any(|ext| name.ends_with(ext))
        {
            out.push(path);
        }
    }
}

#[test]
fn the_parser_crate_is_sdm_pql_and_nothing_outside_the_docs_names_the_interim_one() {
    let root = repo();
    let manifest = std::fs::read_to_string(root.join("rust/crates/sdm-pql/Cargo.toml"))
        .expect("the crate's manifest is at rust/crates/sdm-pql");
    assert!(manifest.contains("name = \"sdm-pql\""), "{manifest}");
    let lock = std::fs::read_to_string(root.join("rust/Cargo.lock")).unwrap();
    assert!(lock.contains("name = \"sdm-pql\""));
    let mut found = Vec::new();
    files(&root, &mut found);
    assert!(found.len() > 200, "{} files searched", found.len());
    let names = interim();
    let mut naming = Vec::new();
    for path in found {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        if names.iter().any(|n| text.contains(n.as_str())) {
            naming.push(
                path.strip_prefix(&root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
    assert!(naming.is_empty(), "the interim name is named in {naming:?}");
}
