// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading dates out of text as pandas 3.0.6 does (DV20): `to_datetime`
//! over a column, with the format guess, `strptime`, the ISO 8601 reader and
//! `dateutil` it runs on.

pub mod civil;
pub mod dateutil;
pub mod iso;
pub mod pandas;
pub mod strftime;
pub mod strptime;
pub mod zone;

pub use pandas::{Call, ColumnTz, Converted, Host, PyErr, to_datetime};
