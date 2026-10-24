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
