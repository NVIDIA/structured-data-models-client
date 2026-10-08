// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The contract schema's regeneration and lock checks.
//!
//! `SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock -- generated`
//! writes the schema of the current version and adds its lock entry; it
//! refuses to rewrite an entry that exists. That entries never change across
//! history is checked by `scripts/rust_core/contract_lock.py`.

use std::path::{Path, PathBuf};

use sdm_contract::{CONTRACT_VERSION, schema_json};
use sha2::{Digest, Sha256};

fn schema_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schema")
}

fn schema_file(version: u32) -> String {
    format!("sdm-contract-{version}.json")
}

fn sha256(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The lock's entries: `(version, sha256, file)`, in file order.
fn parse_lock(text: &str) -> Result<Vec<(u32, String, String)>, String> {
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let [version, hash, file] = parts[..] else {
            return Err(format!("not a lock entry: {line:?}"));
        };
        let version: u32 = version
            .parse()
            .map_err(|_| format!("not a version: {version:?}"))?;
        let hash = hash
            .strip_prefix("sha256:")
            .ok_or_else(|| format!("not a sha256: {hash:?}"))?;
        if entries.iter().any(|(v, _, _)| *v == version) {
            return Err(format!("version {version} is locked twice"));
        }
        entries.push((version, hash.to_owned(), file.to_owned()));
    }
    Ok(entries)
}

/// Checks that the current version is locked to `schema`.
fn verify(schema: &str, lock: &str, version: u32) -> Result<(), String> {
    let entries = parse_lock(lock)?;
    let Some((_, hash, file)) = entries.iter().find(|(v, _, _)| *v == version) else {
        return Err(format!(
            "contract version {version} has no lock entry; a schema change needs a version bump and a new entry"
        ));
    };
    if *file != schema_file(version) {
        return Err(format!("version {version} is locked to {file}"));
    }
    if *hash != sha256(schema) {
        return Err(format!(
            "the schema of contract version {version} differs from its lock entry; \
             bump CONTRACT_VERSION and add an entry instead of changing a locked version"
        ));
    }
    Ok(())
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn the_committed_schema_is_the_generated_one() {
    let generated = schema_json();
    let path = schema_dir().join(schema_file(CONTRACT_VERSION));
    if std::env::var_os("SDM_CONTRACT_WRITE").is_some() {
        let lock_path = schema_dir().join("contract.lock");
        let lock = std::fs::read_to_string(&lock_path).unwrap_or_default();
        let entries = parse_lock(&lock).unwrap();
        if let Some((_, hash, _)) = entries.iter().find(|(v, _, _)| *v == CONTRACT_VERSION) {
            assert_eq!(
                *hash,
                sha256(&generated),
                "version {CONTRACT_VERSION} is locked; bump CONTRACT_VERSION instead"
            );
        } else {
            let mut lock = if lock.is_empty() {
                "# The contract schema lock: one line per contract version, never rewritten.\n"
                    .to_owned()
            } else {
                lock
            };
            lock.push_str(&format!(
                "{CONTRACT_VERSION} sha256:{} {}\n",
                sha256(&generated),
                schema_file(CONTRACT_VERSION)
            ));
            std::fs::write(&lock_path, lock).unwrap();
        }
        std::fs::write(&path, &generated).unwrap();
    }
    assert!(
        read(&path) == generated,
        "{} differs from the schema the types generate; regenerate it with \
         SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock -- generated",
        path.display()
    );
}

#[test]
fn every_locked_version_matches_its_committed_schema() {
    let lock = read(&schema_dir().join("contract.lock"));
    let entries = parse_lock(&lock).unwrap();
    assert!(entries.iter().any(|(v, _, _)| *v == CONTRACT_VERSION));
    for (version, _, file) in &entries {
        let schema = read(&schema_dir().join(file));
        verify(&schema, &lock, *version).unwrap();
    }
}

#[test]
fn a_schema_change_without_a_version_bump_fails() {
    let lock = read(&schema_dir().join("contract.lock"));
    let changed = schema_json().replace("\"step\"", "\"stage\"");
    assert_ne!(changed, schema_json());
    let err = verify(&changed, &lock, CONTRACT_VERSION).unwrap_err();
    assert!(err.contains("bump CONTRACT_VERSION"), "{err}");
    // A bump with no entry fails too.
    let err = verify(&changed, &lock, CONTRACT_VERSION + 1).unwrap_err();
    assert!(err.contains("no lock entry"), "{err}");
}

#[test]
fn a_malformed_or_doubled_lock_fails() {
    assert!(parse_lock("1 abc sdm-contract-1.json").is_err());
    assert!(
        parse_lock("1 sha256:ab sdm-contract-1.json\n1 sha256:cd sdm-contract-1.json").is_err()
    );
    assert!(parse_lock("one sha256:ab sdm-contract-1.json").is_err());
}
