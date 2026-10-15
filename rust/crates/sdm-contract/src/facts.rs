// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host facts: what the core would otherwise read from the machine, passed in
//! with every operation so tests can pin it.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::scalar::Scalar;

/// The facts a host passes with every operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HostFacts {
    /// The clock: a `ts` scalar in UTC. Two-digit years and "now" read it.
    pub now: Scalar,
    /// The host's local zone, if it knows one.
    pub local_zone: Option<LocalZone>,
    /// The zone lookup roots, in the order the host's own zone library
    /// searches them; the core falls back to its bundled database.
    pub zone_roots: Vec<ZoneRootFact>,
    pub platform: PlatformFacts,
    /// The instants the host can hold a PQL date literal as, when it holds
    /// fewer than the core reads (years 0 to 9999); none when it holds them
    /// all. A literal outside them is refused as one that cannot be read
    /// (`PQL_VALUE_INVALID_LITERAL`), as the reference refuses it on such a
    /// host: pandas 2 holds a literal as a nanosecond `Timestamp`.
    ///
    /// A bridge fact: it is needed only because the Python host's S0
    /// shadow check runs the old pandas validator on every query the core
    /// accepts. It is deleted with that check in Phase 5, and the core then
    /// reads years 0 to 9999 on every host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub date_literal_range: Option<InstantRange>,
}

/// Instants, inclusive at both ends, as nanoseconds from the Unix epoch on
/// a literal's own wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InstantRange {
    pub min_ns: i64,
    pub max_ns: i64,
}

/// The host's local zone, as the host's own platform names it: an IANA
/// name, or a POSIX `TZ` string where the platform has no IANA name for it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum LocalZone {
    /// An IANA zone name, such as `Europe/Berlin`.
    Iana(String),
    /// A POSIX `TZ` string, such as `EST5EDT,M3.2.0,M11.1.0`.
    Posix(String),
}

/// One zone lookup root, with the database version the host read there.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ZoneRootFact {
    pub path: String,
    pub version: Option<String>,
}

/// Platform behavior the reference depends on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlatformFacts {
    /// The operating system, as the host names it (`linux`, `darwin`,
    /// `windows`).
    pub os: String,
    /// Whether the platform's `strftime` pads years below 1000 to four
    /// digits (`date-rules.md`).
    pub strftime_pads_year: bool,
}
