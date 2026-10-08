// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The regeneration and lock checks of the two versioned schemas: the
//! contract's (`contract.lock`) and the PQL validated-query format's
//! (`validated-query.lock`).
//!
//! `SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock -- generated`
//! writes the schema of each current version and adds its lock entry; it
//! refuses to rewrite an entry that exists. That entries never change across
//! history is checked by `scripts/rust_core/contract_lock.py`.

use std::path::{Path, PathBuf};

use sdm_contract::{
    CONTRACT_VERSION, VALIDATED_QUERY_VERSION, schema_json, validated_query_schema_json,
};
use sha2::{Digest, Sha256};

/// One versioned schema and its lock.
struct Format {
    /// The schema file's name before `-<version>.json`.
    prefix: &'static str,
    lock: &'static str,
    version: u32,
    generate: fn() -> String,
}

const CONTRACT: Format = Format {
    prefix: "sdm-contract",
    lock: "contract.lock",
    version: CONTRACT_VERSION,
    generate: schema_json,
};

const VALIDATED_QUERY: Format = Format {
    prefix: "pql-validated-query",
    lock: "validated-query.lock",
    version: VALIDATED_QUERY_VERSION,
    generate: validated_query_schema_json,
};

const FORMATS: [Format; 2] = [CONTRACT, VALIDATED_QUERY];

fn schema_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schema")
}

impl Format {
    fn schema_file(&self, version: u32) -> String {
        format!("{}-{version}.json", self.prefix)
    }
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

/// Checks that `version` of `format` is locked to `schema`.
fn verify(format: &Format, schema: &str, lock: &str, version: u32) -> Result<(), String> {
    let entries = parse_lock(lock)?;
    let Some((_, hash, file)) = entries.iter().find(|(v, _, _)| *v == version) else {
        return Err(format!(
            "{} version {version} has no lock entry; a schema change needs a version bump and a new entry",
            format.prefix
        ));
    };
    if *file != format.schema_file(version) {
        return Err(format!("version {version} is locked to {file}"));
    }
    if *hash != sha256(schema) {
        return Err(format!(
            "the schema of {} version {version} differs from its lock entry; \
             bump the version and add an entry instead of changing a locked version",
            format.prefix
        ));
    }
    Ok(())
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Writes the schema of `format`'s current version and adds its lock
/// entry, refusing to rewrite one that exists.
fn write(format: &Format, generated: &str) {
    let lock_path = schema_dir().join(format.lock);
    let lock = std::fs::read_to_string(&lock_path).unwrap_or_default();
    let entries = parse_lock(&lock).unwrap();
    if let Some((_, hash, _)) = entries.iter().find(|(v, _, _)| *v == format.version) {
        assert_eq!(
            *hash,
            sha256(generated),
            "{} version {} is locked; bump the version instead",
            format.prefix,
            format.version
        );
    } else {
        let mut lock = if lock.is_empty() {
            format!(
                "# The {} schema lock: one line per version, never rewritten.\n",
                format.prefix
            )
        } else {
            lock
        };
        lock.push_str(&format!(
            "{} sha256:{} {}\n",
            format.version,
            sha256(generated),
            format.schema_file(format.version)
        ));
        std::fs::write(&lock_path, lock).unwrap();
    }
    std::fs::write(
        schema_dir().join(format.schema_file(format.version)),
        generated,
    )
    .unwrap();
}

#[test]
fn the_committed_schemas_are_the_generated_ones() {
    for format in &FORMATS {
        let generated = (format.generate)();
        if std::env::var_os("SDM_CONTRACT_WRITE").is_some() {
            write(format, &generated);
        }
        let path = schema_dir().join(format.schema_file(format.version));
        assert!(
            read(&path) == generated,
            "{} differs from the schema the types generate; regenerate it with \
             SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock -- generated",
            path.display()
        );
    }
}

#[test]
fn every_locked_version_matches_its_committed_schema() {
    for format in &FORMATS {
        let lock = read(&schema_dir().join(format.lock));
        let entries = parse_lock(&lock).unwrap();
        assert!(entries.iter().any(|(v, _, _)| *v == format.version));
        for (version, _, file) in &entries {
            let schema = read(&schema_dir().join(file));
            verify(format, &schema, &lock, *version).unwrap();
        }
    }
}

#[test]
fn a_schema_change_without_a_version_bump_fails() {
    for (format, word) in [(&CONTRACT, "\"step\""), (&VALIDATED_QUERY, "\"entity\"")] {
        let lock = read(&schema_dir().join(format.lock));
        let generated = (format.generate)();
        let changed = generated.replace(word, "\"renamed\"");
        assert_ne!(changed, generated);
        let err = verify(format, &changed, &lock, format.version).unwrap_err();
        assert!(err.contains("bump the version"), "{err}");
        // A bump with no entry fails too.
        let err = verify(format, &changed, &lock, format.version + 1).unwrap_err();
        assert!(err.contains("no lock entry"), "{err}");
    }
}

#[test]
fn a_malformed_or_doubled_lock_fails() {
    assert!(parse_lock("1 abc sdm-contract-1.json").is_err());
    assert!(
        parse_lock("1 sha256:ab sdm-contract-1.json\n1 sha256:cd sdm-contract-1.json").is_err()
    );
    assert!(parse_lock("one sha256:ab sdm-contract-1.json").is_err());
}
