// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ordered host lookup and the recorded per-zone source, on fixture
//! directories built from the vendored 2026c files; the runner's own zone
//! database is never read.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use jiff::Timestamp;
use sdm_host::zones::{ZoneError, ZoneResolver, ZoneRoot, ZoneSource};

fn vendored() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tzdata-2026c")
}

/// A fresh fixture directory holding `files`, each `(zone name, vendored
/// zone whose bytes it gets)`.
fn fixture(files: &[(&str, &str)]) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "zone-roots-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    for (name, from) in files {
        let target = dir.join(name);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(vendored().join(from), target).unwrap();
    }
    dir
}

fn root(path: &Path, version: &str) -> ZoneRoot {
    ZoneRoot {
        path: path.to_owned(),
        version: Some(version.to_owned()),
    }
}

/// The UTC offset in seconds of `zone` at a summer instant of 2024.
fn summer_offset(zone: &jiff::tz::TimeZone) -> i32 {
    let at: Timestamp = "2024-07-01T12:00:00Z".parse().unwrap();
    zone.to_offset(at).seconds()
}

#[test]
fn the_first_root_holding_a_zone_wins() {
    // The first root's America/New_York carries Tokyo's rules, so the
    // offset shows which file was read.
    let first = fixture(&[("America/New_York", "Asia/Tokyo")]);
    let second = fixture(&[
        ("America/New_York", "America/New_York"),
        ("Europe/Berlin", "Europe/Berlin"),
    ]);
    let resolver = ZoneResolver::new([root(&first, "first"), root(&second, "second")]);

    let new_york = resolver.resolve("America/New_York").unwrap();
    assert_eq!(summer_offset(&new_york), 9 * 3600);
    let berlin = resolver.resolve("Europe/Berlin").unwrap();
    assert_eq!(summer_offset(&berlin), 2 * 3600);

    let sources = resolver.sources();
    assert_eq!(
        sources["America/New_York"],
        ZoneSource::HostRoot {
            index: 0,
            path: first.clone(),
            version: Some("first".into())
        }
    );
    assert_eq!(
        sources["Europe/Berlin"],
        ZoneSource::HostRoot {
            index: 1,
            path: second.clone(),
            version: Some("second".into())
        }
    );
}

#[test]
fn a_zone_no_root_holds_comes_from_the_bundled_database() {
    let only = fixture(&[("Europe/Berlin", "Europe/Berlin")]);
    let resolver = ZoneResolver::new([root(&only, "2026c")]);
    let tokyo = resolver.resolve("Asia/Tokyo").unwrap();
    assert_eq!(summer_offset(&tokyo), 9 * 3600);
    assert_eq!(
        resolver.sources()["Asia/Tokyo"],
        ZoneSource::Bundled {
            version: ZoneResolver::bundled_version().map(str::to_owned)
        }
    );
    assert!(ZoneResolver::bundled_version().is_some());
}

#[test]
fn a_missing_or_empty_root_is_skipped() {
    let missing = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no-such-zone-root");
    let empty = fixture(&[]);
    fs::create_dir_all(&empty).unwrap();
    let real = fixture(&[("Europe/London", "Europe/London")]);
    let resolver = ZoneResolver::new([
        root(&missing, "missing"),
        root(&empty, "empty"),
        root(&real, "2026c"),
    ]);
    resolver.resolve("Europe/London").unwrap();
    assert_eq!(
        resolver.sources()["Europe/London"],
        ZoneSource::HostRoot {
            index: 2,
            path: real,
            version: Some("2026c".into())
        }
    );
}

#[test]
fn no_roots_means_the_bundled_database() {
    let resolver = ZoneResolver::new([]);
    resolver.resolve("America/Santiago").unwrap();
    assert!(matches!(
        resolver.sources()["America/Santiago"],
        ZoneSource::Bundled { .. }
    ));
}

#[test]
fn utc_is_fixed_and_etc_unknown_is_refused() {
    let resolver = ZoneResolver::new([root(&fixture(&[]), "2026c")]);
    assert_eq!(resolver.resolve("UTC").unwrap(), jiff::tz::TimeZone::UTC);
    assert_eq!(resolver.sources()["UTC"], ZoneSource::Fixed);
    assert_eq!(
        resolver.resolve("Etc/Unknown").unwrap_err(),
        ZoneError::Unknown {
            name: "Etc/Unknown".into()
        }
    );
}

#[test]
fn an_unknown_zone_is_refused_by_name_and_not_recorded() {
    let resolver = ZoneResolver::new([root(&vendored(), "2026c")]);
    let err = resolver.resolve("Mars/Olympus_Mons").unwrap_err();
    assert_eq!(
        err,
        ZoneError::Unknown {
            name: "Mars/Olympus_Mons".into()
        }
    );
    assert!(err.to_string().contains("Mars/Olympus_Mons"));
    assert!(resolver.sources().is_empty());
}

#[test]
fn only_the_canonical_spelling_resolves() {
    // The root holds the zone, and so does the bundled database; jiff would
    // find both under any case, and neither is taken.
    let only = fixture(&[("America/New_York", "America/New_York")]);
    let resolver = ZoneResolver::new([root(&only, "2026c")]);
    for name in [
        "america/new_york",
        "AMERICA/NEW_YORK",
        "utc",
        "Utc",
        "etc/unknown",
    ] {
        assert_eq!(
            resolver.resolve(name).unwrap_err(),
            ZoneError::Unknown { name: name.into() },
            "{name}"
        );
    }
    // A zone only the bundled database has is refused the same way.
    assert!(resolver.resolve("asia/tokyo").is_err());
    assert!(resolver.sources().is_empty());

    resolver.resolve("America/New_York").unwrap();
    resolver.resolve("Asia/Tokyo").unwrap();
    resolver.resolve("UTC").unwrap();
    let names: Vec<String> = resolver.sources().into_keys().collect();
    assert_eq!(names, ["America/New_York", "Asia/Tokyo", "UTC"]);
}
