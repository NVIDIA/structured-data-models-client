// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The local zone a host's facts name, by the one rule every binding asks
//! (T3.12; T3.11's review: the rule the Python host's `_rust/facts.py` and
//! the Java host's `HostFacts` each held), over a fake of the machine's
//! files, so no test reads the file system.

use std::path::{Path, PathBuf};

use sdm_contract::facts::LocalZone;
use sdm_host::{LOCALTIME, ZoneFiles, is_posix_zone, local_zone};

use std::cell::RefCell;
use std::collections::BTreeMap;

/// The machine's files, as a host would answer: the regular files, and
/// where each link resolves; every path asked, in order.
#[derive(Default)]
struct Files {
    regular: Vec<PathBuf>,
    links: BTreeMap<PathBuf, PathBuf>,
    asked: RefCell<Vec<PathBuf>>,
}

impl ZoneFiles for Files {
    fn is_file(&self, path: &Path) -> bool {
        self.asked.borrow_mut().push(path.to_owned());
        self.regular.iter().any(|file| file == path)
    }

    fn real_path(&self, path: &Path) -> Option<PathBuf> {
        self.asked.borrow_mut().push(path.to_owned());
        self.links.get(path).cloned()
    }
}

fn holding(files: &[&str]) -> Files {
    Files {
        regular: files.iter().map(PathBuf::from).collect(),
        ..Files::default()
    }
}

fn iana_zone(name: &str) -> Option<LocalZone> {
    Some(LocalZone::Iana(name.to_owned()))
}

fn roots(paths: &[&str]) -> Vec<PathBuf> {
    paths.iter().map(PathBuf::from).collect()
}

#[test]
fn a_name_a_root_holds_is_its_iana_name() {
    let files = holding(&["/zi/Europe/Berlin"]);
    let roots = roots(&["/zi"]);
    assert_eq!(
        local_zone(Some("Europe/Berlin"), &roots, &files),
        iana_zone("Europe/Berlin")
    );
    // One leading colon is dropped, as glibc reads TZ.
    assert_eq!(
        local_zone(Some(":Europe/Berlin"), &roots, &files),
        iana_zone("Europe/Berlin")
    );
}

#[test]
fn utc_is_named_in_any_case_without_a_root() {
    let files = holding(&[]);
    assert_eq!(local_zone(Some("utc"), &[], &files), iana_zone("utc"));
    assert_eq!(local_zone(Some(":UtC"), &[], &files), iana_zone("UtC"));
    assert!(files.asked.borrow().is_empty());
}

#[test]
fn every_root_is_searched_in_order() {
    let files = holding(&["/second/Asia/Tokyo"]);
    let roots = roots(&["/first", "/second"]);
    assert_eq!(
        local_zone(Some("Asia/Tokyo"), &roots, &files),
        iana_zone("Asia/Tokyo")
    );
    assert_eq!(
        *files.asked.borrow(),
        [
            PathBuf::from("/first/Asia/Tokyo"),
            PathBuf::from("/second/Asia/Tokyo")
        ]
    );
}

#[test]
fn a_name_leaving_the_root_is_never_asked_of_the_files() {
    let files = holding(&["/outside", "/zi/Europe/Berlin"]);
    let roots = roots(&["/zi"]);
    for setting in [
        "../outside",
        "Europe/../../outside",
        "/zi/Europe/Berlin",
        "",
        ":",
    ] {
        assert_eq!(local_zone(Some(setting), &roots, &files), None, "{setting}");
    }
    assert!(files.asked.borrow().is_empty());
}

#[test]
fn a_posix_string_is_kept_as_given() {
    let files = holding(&[]);
    assert_eq!(
        local_zone(Some("EST5EDT,M3.2.0,M11.1.0"), &[], &files),
        Some(LocalZone::Posix("EST5EDT,M3.2.0,M11.1.0".to_owned()))
    );
    // An IANA name a root holds wins over the POSIX reading.
    let held = holding(&["/zi/EST5EDT"]);
    assert_eq!(
        local_zone(Some("EST5EDT"), &roots(&["/zi"]), &held),
        iana_zone("EST5EDT")
    );
    // The colon is TZ's, kept in a POSIX string as the setting gave
    // it; a run's parser decides whether it reads it.
    let expected = is_posix_zone(":EST5EDT").then(|| LocalZone::Posix(":EST5EDT".to_owned()));
    assert_eq!(local_zone(Some(":EST5EDT"), &[], &files), expected);
}

#[test]
fn an_unknown_setting_names_no_zone() {
    let files = holding(&[]);
    for setting in ["Not/AZone", "EST", "<+03", "EST5EDT,garbage,garbage"] {
        assert_eq!(
            local_zone(Some(setting), &roots(&["/zi"]), &files),
            None,
            "{setting}"
        );
    }
}

#[test]
fn without_tz_the_link_is_named_after_the_last_zoneinfo() {
    let mut files = holding(&[]);
    files.links.insert(
        PathBuf::from(LOCALTIME),
        PathBuf::from("/usr/share/zoneinfo/posix/zoneinfo/America/New_York"),
    );
    assert_eq!(local_zone(None, &[], &files), iana_zone("America/New_York"));
    assert_eq!(*files.asked.borrow(), [PathBuf::from(LOCALTIME)]);
}

#[test]
fn without_tz_a_link_naming_no_zone_is_none() {
    // Missing.
    assert_eq!(local_zone(None, &[], &holding(&[])), None);
    for target in ["/etc/plain", "/usr/share/zoneinfo", "/usr/share/zoneinfo/"] {
        let mut files = holding(&[]);
        files
            .links
            .insert(PathBuf::from(LOCALTIME), PathBuf::from(target));
        assert_eq!(local_zone(None, &[], &files), None, "{target}");
    }
}

#[test]
fn a_setting_wins_over_the_link() {
    let mut files = holding(&[]);
    files.links.insert(
        PathBuf::from(LOCALTIME),
        PathBuf::from("/usr/share/zoneinfo/Asia/Tokyo"),
    );
    assert_eq!(local_zone(Some("UTC"), &[], &files), iana_zone("UTC"));
    assert!(files.asked.borrow().is_empty());
}
