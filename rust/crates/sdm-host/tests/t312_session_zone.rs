// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.12 fix 3 (Codex r3 P0-1): the zone a psycopg session reads a
//! `timestamptz` in, as Python's `ZoneInfo` finds it: `UTC` itself; a zone
//! by exactly its name, from the lookup roots, then the bundled copy (no
//! roots here); none for a name Python finds no zone under (another
//! spelling, a fixed offset), where psycopg reads in UTC.

use sdm_host::session_zone_offset;

/// 1900-01-01T00:00Z and 2024-07-01T00:00Z, in microseconds.
const IN_1900: i64 = -2_208_988_800_000_000;
const IN_2024: i64 = 1_719_792_000_000_000;

#[test]
fn a_session_zone_is_read_as_python_s_zoneinfo_reads_it() {
    assert_eq!(session_zone_offset("UTC", &[], IN_2024), Some(0));
    assert_eq!(
        session_zone_offset("Asia/Kolkata", &[], IN_2024),
        Some(19_800)
    );
    assert_eq!(
        session_zone_offset("America/New_York", &[], IN_2024),
        Some(-14_400)
    );
    // A historic offset, to the second, as the bundled copy holds it.
    assert_eq!(
        session_zone_offset("Asia/Kolkata", &[], IN_1900),
        Some(19_270)
    );
    for unknown in ["asia/kolkata", "GMT+3", "+03:00", "Nowhere/Else", ""] {
        assert_eq!(
            session_zone_offset(unknown, &[], IN_2024),
            None,
            "{unknown}"
        );
    }
}

/// 9999-12-31T23:59:59Z, a common "no end" value, past the last instant
/// jiff holds (9999-12-30T22:00Z), in microseconds.
const END_OF_TIME: i64 = 253_402_300_799_000_000;

/// Claude T3.12 r5 P0-1: an instant past the zone data's last one is in its
/// zone still (no zone changes offset in the last day of 9999), never read
/// as an unknown zone's UTC.
#[test]
fn an_instant_at_the_end_of_time_is_in_its_zone() {
    for (zone, offset) in [
        ("America/Los_Angeles", -28_800),
        ("America/New_York", -18_000),
        ("Pacific/Honolulu", -36_000),
        ("Asia/Kolkata", 19_800),
        ("UTC", 0),
    ] {
        assert_eq!(
            session_zone_offset(zone, &[], END_OF_TIME),
            Some(offset),
            "{zone}"
        );
    }
    assert_eq!(session_zone_offset("GMT+3", &[], END_OF_TIME), None);
}
