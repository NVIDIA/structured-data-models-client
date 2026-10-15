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
//!
//! Names are matched without regard to case, as the reference takes them
//! (pandas reads `utc` on every platform, and pandas 2 finds
//! `america/new_york` through pytz), and a zone found under any spelling is
//! known by its canonical one, the IANA name [`TimeZone::iana_name`] gives:
//! its source is recorded under that name, so two spellings of one zone are
//! one entry (Phase 1 decisions, E3; register DT-8).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use jiff::tz::{TimeZone, TimeZoneDatabase};
use sdm_core::{UnknownZone, Zones};

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
/// every zone it resolved. It is the zones a run's core resolves with
/// ([`Zones`], decision H14).
pub struct ZoneResolver {
    databases: Arc<Databases>,
    used: Mutex<BTreeMap<String, ZoneSource>>,
}

/// The zone databases of one list of lookup roots, in order, and the
/// bundled one: shared by every resolver over the same roots and versions.
struct Databases {
    roots: Vec<(ZoneRoot, Option<TimeZoneDatabase>)>,
    bundled: TimeZoneDatabase,
}

/// The lookup roots and versions a list of databases was read for.
type RootsKey = Vec<(PathBuf, Option<String>)>;

/// The databases read so far, by the lookup roots and versions they were
/// read for, so runs with the same host facts share them (design, "Data
/// access"). Each resolver keeps its own record of the zones it resolved.
/// The lock is held only while a list is looked up or read, never across a
/// `tracing` event or an Arrow release, and a binding takes it with the
/// GIL released.
static CACHE: Mutex<BTreeMap<RootsKey, Arc<Databases>>> = Mutex::new(BTreeMap::new());

fn databases(roots: Vec<ZoneRoot>) -> Arc<Databases> {
    let key: RootsKey = roots
        .iter()
        .map(|root| (root.path.clone(), root.version.clone()))
        .collect();
    let mut cache = CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    let read = cache.entry(key).or_insert_with(|| {
        let roots = roots
            .into_iter()
            .map(|root| {
                // The one file-system read in the domain and shipped crates:
                // the host names its lookup roots in HostFacts, and the
                // resolver reads the zone database each one holds (E24).
                #[allow(
                    clippy::disallowed_methods,
                    reason = "reads the zone lookup roots the host names (E24)"
                )]
                let db = TimeZoneDatabase::from_dir(&root.path).ok();
                (root, db)
            })
            .collect();
        Arc::new(Databases {
            roots,
            bundled: TimeZoneDatabase::bundled(),
        })
    });
    Arc::clone(read)
}

impl ZoneResolver {
    /// A resolver over `roots`, searched in the order given; a root that
    /// does not hold a zone database is skipped. The databases are read
    /// once per list of roots and versions, and shared.
    pub fn new(roots: impl IntoIterator<Item = ZoneRoot>) -> Self {
        Self {
            databases: databases(roots.into_iter().collect()),
            used: Mutex::new(BTreeMap::new()),
        }
    }

    /// The version of the bundled database, fixed by `Cargo.lock`.
    pub fn bundled_version() -> Option<&'static str> {
        jiff_tzdb::VERSION
    }

    /// The zone `name`, in any case, from the first root that has it, else
    /// the bundled database; the source used is recorded under the zone's
    /// canonical name, whatever spelling asked for it.
    pub fn resolve(&self, name: &str) -> Result<TimeZone, ZoneError> {
        let (zone, source) = self.lookup(name)?;
        let canonical = zone.iana_name().unwrap_or(name).to_owned();
        self.used
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(canonical, source);
        Ok(zone)
    }

    /// Every zone resolved so far, by its canonical name, with its source.
    pub fn sources(&self) -> BTreeMap<String, ZoneSource> {
        self.used
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    fn lookup(&self, name: &str) -> Result<(TimeZone, ZoneSource), ZoneError> {
        // jiff answers these two from no database at all. UTC is fixed by
        // definition, in any case; Etc/Unknown is jiff's own name, not an
        // IANA zone, and the reference refuses it in any case, so it is
        // refused here too.
        if name.eq_ignore_ascii_case("UTC") {
            return Ok((TimeZone::UTC, ZoneSource::Fixed));
        }
        if name.eq_ignore_ascii_case("Etc/Unknown") {
            return Err(ZoneError::Unknown {
                name: name.to_owned(),
            });
        }
        // jiff's databases match names without regard to case, and give the
        // zone under the spelling they hold.
        for (index, (root, db)) in self.databases.roots.iter().enumerate() {
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
        match self.databases.bundled.get(name) {
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

/// The core resolves through the host's resolver, which records each
/// zone's source; the core is told only that a name is unknown, never the
/// name, which can come from the data (decision H17).
impl Zones for ZoneResolver {
    fn zone(&self, name: &str) -> Result<TimeZone, UnknownZone> {
        self.resolve(name).map_err(|_| UnknownZone)
    }
}
