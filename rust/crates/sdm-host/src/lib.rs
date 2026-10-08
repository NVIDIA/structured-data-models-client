// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host layer every binding shares, plain Rust: the control codecs and
//! the 64 KB control cap ([`codec`]), Arrow admission ([`admission`]), run
//! ownership with panic containment and busy handling ([`runs`]), and zone
//! resolution through the host's lookup roots ([`zones`]). It schedules no
//! domain work; that is the core's.

#![forbid(unsafe_code)]

pub mod codec;
pub mod zones;

pub use codec::{CONTROL_CAP, decode, encode, encode_error, encode_warnings};
pub use zones::{ZoneError, ZoneResolver, ZoneRoot, ZoneSource};
