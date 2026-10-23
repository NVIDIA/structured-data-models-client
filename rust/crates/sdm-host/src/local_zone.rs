// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local zone a host's facts name (contract `HostFacts.local_zone`),
//! by one rule every binding asks, so two hosts on one machine send the
//! same zone (T3.12; the rule the Python host's `_rust/facts.py` and the
//! Java host's `HostFacts` each held before).
//!
//! The rule reads two things of the machine's files, which it asks the
//! host for through [`ZoneFiles`]: whether a path names a regular file,
//! and where a link resolves to. The host answers them in its own
//! language, so this crate does no I/O of its own (the core's ban on it,
//! `rust/clippy.toml`, and its one exemption, decision E24, stay as they
//! are); each host reads `TZ` and its zone lookup roots, in its own zone
//! library's order, and passes them in.

use std::path::{Component, Path, PathBuf};

use sdm_contract::facts::LocalZone;

use crate::runs::is_posix_zone;

/// The link the platform's local zone is named by, when `TZ` is unset.
pub const LOCALTIME: &str = "/etc/localtime";

/// What [`local_zone`] asks of the machine's files, answered by the host.
pub trait ZoneFiles {
    /// Whether `path` names a regular file, links followed; false when it
    /// cannot be read.
    fn is_file(&self, path: &Path) -> bool;
    /// `path` with every link resolved, absolute; none when it does not
    /// resolve (it, or a link on the way, is missing).
    fn real_path(&self, path: &Path) -> Option<PathBuf>;
}

/// The local zone, as the platform names it.
///
/// `setting` is `TZ` when set: one leading `:` is dropped, and the name
/// left is an IANA name when it is `UTC` in any case, or a regular file
/// under one of `roots` and neither empty, absolute nor holding a `..`
/// part; else `TZ` as given is a POSIX `TZ` string when a run reads it as
/// one ([`is_posix_zone`]); else none. With `setting` none, the zone
/// [`LOCALTIME`] resolves to, named by its path after the last `zoneinfo`
/// part; none when it does not resolve or names none.
pub fn local_zone(
    setting: Option<&str>,
    roots: &[PathBuf],
    files: &dyn ZoneFiles,
) -> Option<LocalZone> {
    if let Some(setting) = setting {
        let name = setting.strip_prefix(':').unwrap_or(setting);
        if iana(name, roots, files) {
            return Some(LocalZone::Iana(name.to_owned()));
        }
        if is_posix_zone(setting) {
            return Some(LocalZone::Posix(setting.to_owned()));
        }
        return None;
    }
    let target = files.real_path(Path::new(LOCALTIME))?;
    let parts: Vec<String> = target
        .components()
        .filter_map(|part| match part {
            Component::Normal(name) => Some(name.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let index = parts.iter().rposition(|part| part == "zoneinfo")?;
    let name = parts[index + 1..].join("/");
    (!name.is_empty()).then_some(LocalZone::Iana(name))
}

/// Whether `name` is an IANA name the host's roots hold, or `UTC`.
fn iana(name: &str, roots: &[PathBuf], files: &dyn ZoneFiles) -> bool {
    if name.to_uppercase() == "UTC" {
        return true;
    }
    if name.is_empty() || name.starts_with('/') || name.split('/').any(|part| part == "..") {
        return false;
    }
    roots.iter().any(|root| files.is_file(&root.join(name)))
}
