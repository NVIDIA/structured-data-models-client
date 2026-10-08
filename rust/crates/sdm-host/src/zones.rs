// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Zone resolution (decision A7): each zone resolves through the host's
//! lookup roots in order, then the bundled copy, and the source and version
//! used for each zone are recorded.
//!
//! The host passes its roots in, in the order its own zone library searches
//! them, with the version it read for each; nothing here reads the
//! environment or asks `jiff` for its global database.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use jiff::tz::{TimeZone, TimeZoneDatabase};

/// One zone lookup root the host names, with the database version it read
/// there (for example from `+VERSION` or `tzdata.zi`), if any.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZoneRoot {
    pub path: PathBuf,
    pub version: Option<String>,
}

/// Where a resolved zone came from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ZoneSource {
    /// The host's root at this position in its lookup order.
    HostRoot {
        index: usize,
        path: PathBuf,
        version: Option<String>,
    },
    /// The database bundled into the core, used when no root has the zone.
    Bundled { version: Option<String> },
    /// `UTC`, whose rules no database version changes.
    Fixed,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ZoneError {
    #[error("unknown time zone {name:?}: no lookup root and not the bundled database has it")]
    Unknown { name: String },
}

/// Resolves zone names for one host's facts, and records the source of
/// every zone it resolved.
pub struct ZoneResolver {
    roots: Vec<(ZoneRoot, Option<TimeZoneDatabase>)>,
    bundled: TimeZoneDatabase,
    used: Mutex<BTreeMap<String, ZoneSource>>,
}

impl ZoneResolver {
    /// A resolver over `roots`, searched in the order given; a root that
    /// does not hold a zone database is skipped.
    pub fn new(roots: impl IntoIterator<Item = ZoneRoot>) -> Self {
        let roots = roots
            .into_iter()
            .map(|root| {
                let db = TimeZoneDatabase::from_dir(&root.path).ok();
                (root, db)
            })
            .collect();
        Self {
            roots,
            bundled: TimeZoneDatabase::bundled(),
            used: Mutex::new(BTreeMap::new()),
        }
    }

    /// The version of the bundled database, fixed by `Cargo.lock`.
    pub fn bundled_version() -> Option<&'static str> {
        jiff_tzdb::VERSION
    }

    /// The zone `name`, from the first root that has it, else the bundled
    /// database; the source used is recorded.
    pub fn resolve(&self, name: &str) -> Result<TimeZone, ZoneError> {
        let (zone, source) = self.lookup(name)?;
        self.used
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(name.to_owned(), source);
        Ok(zone)
    }

    /// Every zone resolved so far, by the name asked for, with its source.
    pub fn sources(&self) -> BTreeMap<String, ZoneSource> {
        self.used
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn lookup(&self, name: &str) -> Result<(TimeZone, ZoneSource), ZoneError> {
        // jiff answers these two from no database at all. UTC is fixed by
        // definition; Etc/Unknown is jiff's own name, not an IANA zone, and
        // Python's zoneinfo refuses it, so it is refused here too.
        if name.eq_ignore_ascii_case("utc") {
            return Ok((TimeZone::UTC, ZoneSource::Fixed));
        }
        if name.eq_ignore_ascii_case("etc/unknown") {
            return Err(ZoneError::Unknown {
                name: name.to_owned(),
            });
        }
        for (index, (root, db)) in self.roots.iter().enumerate() {
            let Some(db) = db else { continue };
            if let Ok(zone) = db.get(name) {
                let source = ZoneSource::HostRoot {
                    index,
                    path: root.path.clone(),
                    version: root.version.clone(),
                };
                return Ok((zone, source));
            }
        }
        match self.bundled.get(name) {
            Ok(zone) => Ok((
                zone,
                ZoneSource::Bundled {
                    version: Self::bundled_version().map(str::to_owned),
                },
            )),
            Err(_) => Err(ZoneError::Unknown {
                name: name.to_owned(),
            }),
        }
    }
}
