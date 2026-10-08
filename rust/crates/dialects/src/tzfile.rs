// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A time zone's UTC offset as pyarrow reads it, to write a zoned timestamp
//! as `pyarrow.compute.strftime` does.
//!
//! pyarrow reads the operating system's time zone database (its vendored
//! `date` library, built to use the OS database) and takes only the
//! transitions each zone's file lists: it does not extend them with the
//! rule the file's footer gives for later years. Files compiled "fat", as
//! macOS and Debian ship them, list transitions to 2037, so after its last
//! transition a zone keeps that transition's offset: New York stays on
//! standard time from November 2037 on. This reads the files the same way.
//!
//! The database is `/usr/share/zoneinfo`, where pyarrow finds it on Linux
//! and, through the link `/etc/localtime` names, on macOS; like pyarrow, this
//! does not read `$TZDIR`. A zone written as an offset, such as `+05:30`,
//! needs none.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::Error;
use crate::time::parse_offset;

/// One zone's transitions, in order, each with the offset from then on,
/// and the offset before the first.
#[derive(Clone, Debug)]
struct Zone {
    transitions: Vec<(i64, i32)>,
    initial: i32,
}

impl Zone {
    fn offset_at(&self, seconds: i64) -> i32 {
        match self.transitions.partition_point(|&(at, _)| at <= seconds) {
            0 => self.initial,
            n => self.transitions[n - 1].1,
        }
    }
}

fn zones() -> &'static Mutex<HashMap<String, Zone>> {
    static ZONES: OnceLock<Mutex<HashMap<String, Zone>>> = OnceLock::new();
    ZONES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn database() -> PathBuf {
    PathBuf::from("/usr/share/zoneinfo")
}

/// The UTC offset, in seconds, the Arrow time zone `zone` has at
/// `seconds` after the epoch.
pub fn zone_offset(zone: &str, seconds: i64) -> Result<i32, Error> {
    if let Some(offset) = parse_offset(zone) {
        return Ok(offset);
    }
    let mut cache = zones()
        .lock()
        .map_err(|_| Error::new("the time zone cache is poisoned"))?;
    if !cache.contains_key(zone) {
        let parsed = load(zone)?;
        cache.insert(zone.to_string(), parsed);
    }
    Ok(cache[zone].offset_at(seconds))
}

fn load(zone: &str) -> Result<Zone, Error> {
    let unknown = || Error::new(format!("unknown time zone {zone:?}"));
    let safe = !zone.is_empty()
        && !zone.starts_with('/')
        && zone
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
        && zone
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-' | '+'));
    if !safe {
        return Err(unknown());
    }
    let bytes = std::fs::read(database().join(zone)).map_err(|_| unknown())?;
    parse(&bytes).ok_or_else(|| Error::new(format!("cannot read the time zone file of {zone:?}")))
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let slice = self.bytes.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_be_bytes(self.take(4)?.try_into().ok()?))
    }
}

/// The counts a TZif header gives.
struct Counts {
    isutcnt: usize,
    isstdcnt: usize,
    leapcnt: usize,
    timecnt: usize,
    typecnt: usize,
    charcnt: usize,
}

fn header(reader: &mut Reader<'_>) -> Option<(u8, Counts)> {
    if reader.take(4)? != b"TZif" {
        return None;
    }
    let version = reader.take(1)?[0];
    reader.take(15)?;
    let mut next = || reader.u32().map(|n| n as usize);
    Some((
        version,
        Counts {
            isutcnt: next()?,
            isstdcnt: next()?,
            leapcnt: next()?,
            timecnt: next()?,
            typecnt: next()?,
            charcnt: next()?,
        },
    ))
}

/// A TZif file's transitions, from its 64-bit data where it has some,
/// as the `date` library loads them.
fn parse(bytes: &[u8]) -> Option<Zone> {
    let mut reader = Reader { bytes, at: 0 };
    let (version, mut counts) = header(&mut reader)?;
    let mut time_size = 4;
    if version >= b'2' {
        // Skip the 32-bit data, and read the 64-bit data after it.
        let skip = counts.timecnt * 5
            + counts.typecnt * 6
            + counts.charcnt
            + counts.leapcnt * 8
            + counts.isstdcnt
            + counts.isutcnt;
        reader.take(skip)?;
        counts = header(&mut reader)?.1;
        time_size = 8;
    }
    let mut times = Vec::with_capacity(counts.timecnt);
    for _ in 0..counts.timecnt {
        let raw = reader.take(time_size)?;
        times.push(if time_size == 8 {
            i64::from_be_bytes(raw.try_into().ok()?)
        } else {
            i64::from(i32::from_be_bytes(raw.try_into().ok()?))
        });
    }
    let indices = reader.take(counts.timecnt)?.to_vec();
    let mut types = Vec::with_capacity(counts.typecnt);
    for _ in 0..counts.typecnt {
        let raw = reader.take(6)?;
        let offset = i32::from_be_bytes(raw[..4].try_into().ok()?);
        types.push((offset, raw[4] != 0));
    }
    if types.is_empty() {
        return None;
    }
    let transitions = times
        .into_iter()
        .zip(indices)
        .map(|(at, index)| Some((at, types.get(usize::from(index))?.0)))
        .collect::<Option<Vec<_>>>()?;
    // Before the first transition, the first type that is not daylight
    // saving time, or else the first type.
    let initial = types.iter().find(|(_, dst)| !dst).unwrap_or(&types[0]).0;
    Some(Zone {
        transitions,
        initial,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_as_pyarrow_reads_them() {
        // pyarrow's offsets of New York in July 2037 and July 2038.
        assert_eq!(
            zone_offset("America/New_York", 2_130_062_400).unwrap(),
            -14_400
        );
        assert_eq!(
            zone_offset("America/New_York", 2_161_598_400).unwrap(),
            -18_000
        );
        assert_eq!(zone_offset("Asia/Kolkata", 0).unwrap(), 19_800);
        assert_eq!(zone_offset("UTC", 0).unwrap(), 0);
        assert_eq!(zone_offset("-03:00", 0).unwrap(), -10_800);
        assert!(zone_offset("../etc/passwd", 0).is_err());
        assert!(zone_offset("No/Such_Zone", 0).is_err());
    }
}
