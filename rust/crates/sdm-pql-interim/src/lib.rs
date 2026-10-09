// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The interim PQL parser: the first attempt's lexer, parser, visitor,
//! syntax tree and seven validators (`archive/rust-core-attempt-1`,
//! `rust/crates/pql`), ported behind `sdm-core`'s adapter.
//!
//! Only `sdm_core::pql` uses this crate. The parser is permanent (decision
//! E18): T1.6b completes it and gives the crate a permanent name. It parses
//! a query ([`parse`]) and resolves it against a schema ([`resolve`]);
//! whether the model on the other end can answer the resolved query is
//! asked by the core, not here, except for the two filter capabilities
//! resolution itself consults ([`Filters`]).
//!
//! Python behavior is emulated only in the modules listed in
//! `scripts/rust_core/emulation-allowlist.txt`: [`pyint`], Python's limit of
//! 4,300 digits on reading an integer, and [`float_repr`], the text Python
//! gives a float (Phase 1 decisions, A4), and [`escapes`], how the reference
//! reads a string literal's escapes (E9). Message text is not promised, so
//! nothing else copies Python's printing.

#![forbid(unsafe_code)]

mod api;
pub mod ast;
pub mod codes;
pub mod dates;
pub mod dor;
pub mod escapes;
pub mod float_repr;
pub mod lexer;
pub mod literal;
pub mod names;
pub mod parser;
pub mod profiles;
pub mod pyint;
pub mod query;
pub mod response;
pub mod schema;
pub mod syntax;
pub mod types;
pub mod validator;
pub mod visitor;

pub use api::{Interrupted, Parsed, Resolved, parse, resolve};
pub use profiles::{Filters, Mode};
pub use query::{ParsedQuery, ValidatedQuery};
pub use response::{Span, ValidationError, ValidationResponse, ValidationWarning};
pub use schema::SchemaDefinition;
