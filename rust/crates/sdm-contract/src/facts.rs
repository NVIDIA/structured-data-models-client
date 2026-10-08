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
