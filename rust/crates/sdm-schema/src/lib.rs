// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The canonical schema and its inference over Arrow (Phase 1b). It holds no
//! SQL and matches on no provider: it reads the catalog's answers as the
//! data layer read them, in no dialect's shape (`sdm_contract::catalog`,
//! decision E54), the key counts in the shape the contract fixes
//! (`sdm_contract::data::results`) and the rows a preview returned, as
//! Arrow.
//!
//! As built at T1.11b it is thin: it infers what `Schema.from_source` needs
//! for the plain tables of `base.regression`, by the reference's rules
//! (the relational engine's metadata inference over its preview, and
//! `predictive_query.time_columns`), and refuses with `UNSUPPORTED_REQUEST`,
//! naming the rule ([`not_yet`]), whatever those rules decide by a path it
//! does not port yet, rather than guess (register entry P1B-2): a column
//! type outside the plain set, text that may hold dates or several values,
//! a tie between key candidates, a time column chosen among several,
//! refused for its name or values or a time of day, a preview of a whole 1,000 rows, an
//! empty table. T1.23 and T1.25 complete it.
//!
//! - [`catalog`] reads the key counts.
//! - [`infer`] infers one table: each column's data and semantic types,
//!   its primary key and its time column.

#![forbid(unsafe_code)]

pub mod catalog;
pub mod infer;
mod names;
mod values;

use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};

/// The refusal of an input whose inference takes a rule this build does not
/// port yet, naming the rule, never a value or a name the input holds
/// (decision H17).
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        ErrorKind::Unsupported,
        ErrorCode::UnsupportedRequest,
        "describing this source takes an inference rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("describe_source"))
    .with_field("rule", Detail::text(rule))
}
