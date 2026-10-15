// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `contract/bases.yaml`, the named DuckDB bases a case builds its data
//! from, read without a YAML library: the file holds one form only, a
//! `format` line and, under `bases:`, each base's name followed by its
//! statements as plain one-line list items, which YAML reads as the text
//! written. Anything else in it is refused, so a file this reader would
//! read differently from the Python runner's YAML reader never passes.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// The format this reader reads.
pub const FORMAT: &str = "sdm-bases/1";

/// Each base's statements, by name.
pub type Bases = BTreeMap<String, Vec<String>>;

/// The bases of the file at `path`.
pub fn load(path: &Path) -> Result<Bases> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&text)
}

/// The bases `text` holds.
pub fn parse(text: &str) -> Result<Bases> {
    let mut bases = Bases::new();
    let mut format = None;
    let mut in_bases = false;
    let mut current: Option<String> = None;
    for (number, line) in text.lines().enumerate() {
        let number = number + 1;
        let trimmed = line.trim_end();
        if trimmed.is_empty() || trimmed.trim_start().starts_with('#') {
            continue;
        }
        if let Some(value) = trimmed.strip_prefix("format: ") {
            format = Some(value.to_owned());
            in_bases = false;
        } else if trimmed == "bases:" {
            in_bases = true;
        } else if let Some(statement) = trimmed.strip_prefix("  - ") {
            let Some(name) = current.as_ref().filter(|_| in_bases) else {
                bail!("bases.yaml:{number}: a statement outside a base");
            };
            if !plain(statement) {
                bail!("bases.yaml:{number}: not a plain one-line statement");
            }
            bases
                .get_mut(name)
                .expect("the current base exists")
                .push(statement.to_owned());
        } else if let Some(name) = trimmed
            .strip_prefix("  ")
            .and_then(|rest| rest.strip_suffix(':'))
            .filter(|name| is_name(name))
        {
            if !in_bases || bases.contains_key(name) {
                bail!("bases.yaml:{number}: a base outside bases, or named twice");
            }
            bases.insert(name.to_owned(), vec![]);
            current = Some(name.to_owned());
        } else {
            bail!("bases.yaml:{number}: a line this reader does not read");
        }
    }
    if format.as_deref() != Some(FORMAT) {
        bail!("bases.yaml: not {FORMAT}");
    }
    if bases.is_empty() || bases.values().any(Vec::is_empty) {
        bail!("bases.yaml: no bases, or a base without statements");
    }
    Ok(bases)
}

/// A base's name, as the case schema writes one.
fn is_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// A statement YAML reads as the text written: a plain scalar on one line,
/// starting with a letter, holding no `: ` or ` #`.
fn plain(statement: &str) -> bool {
    statement
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && !statement.contains(": ")
        && !statement.contains(" #")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_bases_read() {
        let bases =
            load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../contract/bases.yaml"))
                .unwrap();
        assert_eq!(
            bases.keys().collect::<Vec<_>>(),
            ["adventureworks", "declared", "shop"]
        );
        assert_eq!(bases["shop"].len(), 2);
        assert!(bases["shop"][0].starts_with("CREATE TABLE users AS SELECT"));
    }

    #[test]
    fn other_forms_are_refused() {
        for text in [
            "format: sdm-bases/1\nbases:\n  a:\n  - 'quoted'\n",
            "format: sdm-bases/1\nbases:\n  a:\n  - SELECT 1: 2\n",
            "format: sdm-bases/1\nbases:\n  a:\n    - SELECT 1\n",
            "format: sdm-bases/2\nbases:\n  a:\n  - SELECT 1\n",
            "format: sdm-bases/1\nbases:\n  a:\n",
            "format: sdm-bases/1\nbases:\n  a:\n  - SELECT 1\n  a:\n  - SELECT 2\n",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
        assert!(parse("format: sdm-bases/1\nbases:\n  a:\n  - SELECT 1 # x\n").is_err());
    }
}
