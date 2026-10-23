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
//! refused for its values or a time of day, a preview of a whole 1,000
//! rows, an empty table. T1.23 and T1.25 complete it. From T1.21d
//! (decision E55) it holds the schema's assembly the reference client
//! makes after the engine's reading, which the Python bridge held before:
//! a timestamp passed over by its name, links inferred by name and kept
//! where they join, a lone edit stamp settled by them, the columns joined
//! on marked, the warnings, and the consistency check of the schema
//! `materialize` is handed. From T1.23 (decision E59) it holds the rest of
//! the time column's choice, a list column left out, and the keys, time
//! columns and relationships a caller names; its warnings are codes and
//! fields, which each host words. From T1.32e (decision E83) a preview of a
//! whole 1,000 rows no longer refuses: the time column's choice reads its
//! candidates over every row, in the answer to the one request the core
//! asks the source, as the reference's choice reads them.
//!
//! - [`catalog`] reads the key counts.
//! - [`infer`] infers one table: each column's data and semantic types,
//!   its primary key and its time column.
//! - [`assemble`] assembles the tables into the schema, and holds the
//!   schema `materialize` is handed to it.
//! - [`edit`] checks and makes an edit to a schema, as the reference's
//!   setters do (decision E67, T1.24b).
//! - [`held`] reads the schema as a host holds it, as plain data, and
//!   writes it back (T3.16).

#![forbid(unsafe_code)]

pub mod assemble;
pub mod catalog;
pub mod edit;
pub mod held;
pub mod infer;
pub mod keys;
mod names;
mod values;

use sdm_contract::{CoreError, Detail, Sentence};

/// The refusal of an input whose inference takes a rule this build does not
/// port yet, naming the rule, never a value or a name the input holds
/// (decision H17).
pub fn not_yet(rule: &'static str) -> CoreError {
    CoreError::new(
        Sentence::NotYetDescribe,
        "describing this source takes an inference rule this core does not hold yet",
    )
    .with_field("operation", Detail::text("describe_source"))
    .with_field("rule", Detail::text(rule))
}
