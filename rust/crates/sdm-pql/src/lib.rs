// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The PQL parser: the lexer, parser, visitor, syntax tree and seven
//! validators, first ported from the first attempt
//! (`archive/rust-core-attempt-1`, `rust/crates/pql`) and this client's own
//! from then on (decision E18), behind `sdm-core`'s adapter.
//!
//! Only `sdm_core::pql` uses this crate. It parses a query ([`parse`]) and
//! resolves it against a schema ([`resolve`]); whether the model on the
//! other end can answer the resolved query is asked by the core, not here,
//! except for the two filter capabilities resolution itself consults
//! ([`Filters`]).
//!
//! Python behavior is emulated only in the modules listed in
//! `scripts/rust_core/emulation-allowlist.txt`: [`pyint`], Python's limit of
//! 4,300 digits on reading an integer, and [`float_repr`], the text Python
//! gives a float (Phase 1 decisions, A4), and [`escapes`], how the reference
//! reads a string literal's escapes (E9), and from T3.5 [`conversions`], the
//! conversions a view's parameter default goes through. Message text is not
//! promised, so nothing else copies Python's printing.

#![forbid(unsafe_code)]

mod api;
pub mod ast;
pub mod codes;
pub mod conversions;
pub mod dates;
pub mod dor;
pub mod escapes;
pub mod float_repr;
pub mod lexer;
pub mod literal;
pub mod names;
pub mod parser;
mod printable;
pub mod profiles;
pub mod pyint;
pub mod query;
pub mod response;
pub mod schema;
pub mod syntax;
pub mod types;
pub mod validator;
pub mod visitor;

pub use api::{
    DateWarning, LiteralWarning, Parse, Parsed, Resolved, Unparsed, parse, parse_held, resolve,
};
pub use profiles::{Filters, Mode};
pub use query::{ParsedQuery, ValidatedQuery};
pub use response::{Span, ValidationError, ValidationResponse, ValidationWarning};
pub use schema::SchemaDefinition;
