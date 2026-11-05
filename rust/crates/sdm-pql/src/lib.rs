// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PQL parsing, schema resolution and validation.
//! [`parse`] builds the query representation and [`resolve`] resolves it against a
//! schema; the core determines whether the selected model supports the result.
//! Resolution consults [`Filters`] for filter-specific capabilities.
//!
//! Compatibility-sensitive literal conversion is limited to the modules listed in
//! `scripts/rust_core/emulation-allowlist.txt`: integer bounds, float formatting,
//! string escapes and semantic-view parameter conversions.

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
