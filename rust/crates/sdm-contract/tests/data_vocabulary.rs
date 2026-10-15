// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data-request vocabulary carries no expression and no SQL (decision
//! E28): the structural guard that replaced the L0 scan's weak SQL tier.
//!
//! A `DataRequest` names what it wants with typed fields; a provider, or
//! `sdm-data` for a SQL provider, decides how to fetch it. So no field of
//! the vocabulary, at any depth, may hold a free-text expression or a SQL
//! string. The test reads the generated JSON Schema of `DataRequest` and
//! classifies every property's schema recursively: through local `$ref`s
//! (each definition once per path, so a cycle ends), `anyOf`, `oneOf` and
//! `allOf`, list items at any depth, nested objects and map values. It
//! holds:
//!
//! - no property is named as SQL or an expression (`DENIED`), anywhere,
//!   the typed values included, matched against the whole name and against
//!   each of its `_`-separated parts, so `predicate_expr` is refused;
//! - a property typed as a reviewed typed value is refused also when its
//!   name, or a part of it, is `filter` or `order` (`DENIED_FOR_VALUES`):
//!   a scalar or a field there would be a condition or an ordering in
//!   disguise, where the vocabulary has typed structures for both;
//! - text (a string, at any depth of lists, maps, options or newtypes) is
//!   held only by a property named as an identifier the request refers to
//!   (`IDENTIFIERS`), the nearest property naming it;
//! - an any-typed schema (`true` or `{}`, as `serde_json::Value` gives, or a
//!   map of anything) is refused, since it can hold text of any shape;
//! - the typed values (`TYPED_VALUES`: a scalar, an Arrow field and its
//!   type) are reviewed: a reference to one of them is accepted at that
//!   reference, and only the name check goes inside.
//!
//! A new text field fails until it is added to `IDENTIFIERS` in review, so
//! it is a name the request refers to, never text a provider interprets.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use sdm_contract::data::DataRequest;
use serde_json::{Map, Value};

/// Property names that hold SQL or an expression; never allowed.
const DENIED: &[&str] = &[
    "sql",
    "text",
    "query",
    "statement",
    "expression",
    "expr",
    "predicate",
    "condition",
    "clause",
    "where",
    "having",
    "order_by",
    "raw",
    "dialect",
];

/// Names refused, beyond `DENIED`, for a property typed as a reviewed typed
/// value.
const DENIED_FOR_VALUES: &[&str] = &["filter", "filters", "order", "orders"];

/// Text properties that name what a request refers to.
const IDENTIFIERS: &[&str] = &[
    "table",
    "tables",
    "column",
    "columns",
    "key",
    // A column of a `neighbors` request's seed key set (decision H15).
    "seed_column",
];

/// Definitions that are reviewed typed values, not request structure: their
/// text is a scalar's encoded value, a field's name, type or Arrow metadata.
const TYPED_VALUES: &[&str] = &["Scalar", "Field", "ColumnType"];

/// Keywords that annotate a schema without constraining it.
const ANNOTATIONS: &[&str] = &[
    "description",
    "title",
    "examples",
    "default",
    "$schema",
    "$id",
];

#[derive(Debug, Clone, PartialEq, Eq)]
enum Finding {
    /// Text under a property named as an identifier: allowed.
    Identifier(String),
    /// Text under any other property, or with no property naming it.
    Text(String),
    /// A schema that accepts anything.
    Any(String),
    /// A property named as SQL or an expression.
    Denied(String),
}

struct Walk<'a> {
    defs: &'a Map<String, Value>,
    found: Vec<Finding>,
}

/// Whether `name`, whole or by any `_`-separated part, is one of `denied`
/// (or, for a typed value, of `DENIED_FOR_VALUES` too).
fn denied(name: &str, typed_value: bool) -> bool {
    let name = name.to_ascii_lowercase();
    let listed =
        |word: &str| DENIED.contains(&word) || (typed_value && DENIED_FOR_VALUES.contains(&word));
    listed(&name) || name.split('_').any(listed)
}

/// The definition a local `$ref` names.
fn ref_name(reference: &str) -> Option<&str> {
    reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))
}

fn types(schema: &Map<String, Value>) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(ts)) => ts.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

impl Walk<'_> {
    /// Classify `schema`, reached at `path`, under the nearest property
    /// `name` (None at the root); `seen` holds the definitions entered on
    /// this path.
    fn classify(&mut self, schema: &Value, path: &str, name: Option<&str>, seen: &[String]) {
        let map = match schema {
            Value::Bool(true) => return self.found.push(Finding::Any(path.into())),
            Value::Bool(false) => return,
            Value::Object(map) => map,
            _ => return self.found.push(Finding::Any(path.into())),
        };
        if map.keys().all(|k| ANNOTATIONS.contains(&k.as_str())) {
            return self.found.push(Finding::Any(path.into()));
        }
        if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
            let Some(def) = ref_name(reference) else {
                return self
                    .found
                    .push(Finding::Any(format!("{path} ({reference})")));
            };
            let at = format!("{path} -> {def}");
            if seen.iter().any(|s| s == def) {
                return; // a cycle: this definition is classified on this path
            }
            let mut inner = seen.to_vec();
            inner.push(def.to_owned());
            match self.defs.get(def) {
                None => self.found.push(Finding::Any(format!("{at} (missing)"))),
                Some(target) if TYPED_VALUES.contains(&def) => {
                    // The property holding a typed value is named for what
                    // it holds; a condition or an ordering never is one.
                    if let Some(property) = name
                        && denied(property, true)
                    {
                        self.found.push(Finding::Denied(at.clone()));
                    }
                    self.names_only(target, &at, &inner)
                }
                Some(target) => self.classify(target, &at, name, &inner),
            }
        }
        for key in ["anyOf", "oneOf", "allOf"] {
            for (i, branch) in map
                .get(key)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .enumerate()
            {
                self.classify(branch, &format!("{path}/{key}/{i}"), name, seen);
            }
        }
        let closed = map.contains_key("const") || map.contains_key("enum");
        let kinds = types(map);
        if kinds.contains(&"string") && !closed {
            let at = path.to_owned();
            self.found.push(match name {
                Some(n) if IDENTIFIERS.contains(&n) => Finding::Identifier(at),
                _ => Finding::Text(at),
            });
        }
        if kinds.contains(&"array") || map.contains_key("items") {
            match map.get("items") {
                None => self.found.push(Finding::Any(format!("{path}/items"))),
                Some(Value::Array(items)) => {
                    for (i, item) in items.iter().enumerate() {
                        self.classify(item, &format!("{path}/items/{i}"), name, seen);
                    }
                }
                Some(items) => self.classify(items, &format!("{path}/items"), name, seen),
            }
            if let Some(more) = map.get("prefixItems").and_then(Value::as_array) {
                for (i, item) in more.iter().enumerate() {
                    self.classify(item, &format!("{path}/prefixItems/{i}"), name, seen);
                }
            }
        }
        if let Some(properties) = map.get("properties").and_then(Value::as_object) {
            for (property, child) in properties {
                let at = format!("{path}/{property}");
                if denied(property, false) {
                    self.found.push(Finding::Denied(at.clone()));
                }
                self.classify(child, &at, Some(property), seen);
            }
        }
        match map.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(values) => self.classify(values, &format!("{path}/*"), name, seen),
        }
    }

    /// Inside a reviewed typed value: only the name check.
    fn names_only(&mut self, schema: &Value, path: &str, seen: &[String]) {
        match schema {
            Value::Object(map) => {
                for (key, child) in map {
                    if key == "properties" {
                        for (property, _) in child.as_object().into_iter().flatten() {
                            if denied(property, false) {
                                self.found
                                    .push(Finding::Denied(format!("{path}/{property}")));
                            }
                        }
                    }
                    if key == "$ref" {
                        if let Some(def) = child.as_str().and_then(ref_name)
                            && !seen.iter().any(|s| s == def)
                            && let Some(target) = self.defs.get(def)
                        {
                            let mut inner = seen.to_vec();
                            inner.push(def.to_owned());
                            self.names_only(target, &format!("{path} -> {def}"), &inner);
                        }
                        continue;
                    }
                    self.names_only(child, &format!("{path}/{key}"), seen);
                }
            }
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    self.names_only(item, &format!("{path}/{i}"), seen);
                }
            }
            _ => {}
        }
    }
}

fn schema<T: JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("a schema serializes")
}

fn findings(schema: &Value) -> Vec<Finding> {
    let empty = Map::new();
    let defs = schema
        .get("$defs")
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let mut walk = Walk {
        defs,
        found: Vec::new(),
    };
    walk.classify(schema, "#", None, &[]);
    walk.found
}

/// What in a vocabulary's schema could carry SQL or an expression.
fn problems(schema: &Value) -> Vec<Finding> {
    findings(schema)
        .into_iter()
        .filter(|f| !matches!(f, Finding::Identifier(_)))
        .collect()
}

#[test]
fn data_requests_hold_no_expression_or_sql() {
    let found = problems(&schema::<DataRequest>());
    assert!(found.is_empty(), "{found:#?}");
}

#[test]
fn the_lists_do_not_overlap() {
    let refused: BTreeSet<_> = DENIED.iter().chain(DENIED_FOR_VALUES).collect();
    assert!(IDENTIFIERS.iter().all(|name| !refused.contains(name)));
    assert!(IDENTIFIERS.iter().all(|name| !denied(name, true)));
}

#[test]
fn the_vocabulary_has_identifier_text() {
    // The walk sees the request's own text fields; were it blind, the first
    // test would pass on anything.
    let found = findings(&schema::<DataRequest>());
    let at = |field: &str| {
        found
            .iter()
            .any(|f| matches!(f, Finding::Identifier(p) if p.contains(field)))
    };
    assert!(at("/table") && at("/columns/items"), "{found:#?}");
}

// Controls: identifiers and the reviewed typed values pass.

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct Allowed {
    table: String,
    columns: Vec<String>,
    key: Option<Vec<String>>,
    seed: sdm_contract::Scalar,
    fields: Vec<sdm_contract::Field>,
    limit: Option<u64>,
    flag: bool,
}

#[test]
fn identifiers_and_typed_values_pass() {
    let found = problems(&schema::<Allowed>());
    assert!(found.is_empty(), "{found:#?}");
}

// Seeded faults: shapes the guard must refuse, each alone.

fn refused<T: JsonSchema>(expected: &str) {
    let found = problems(&schema::<T>());
    assert!(
        found.iter().any(|f| format!("{f:?}").contains(expected)),
        "{expected} in {found:#?}"
    );
}

#[derive(JsonSchema)]
#[serde(tag = "request")]
#[allow(dead_code, reason = "only its schema is read")]
enum WithSql {
    Scan { table: String, sql: String },
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct FilterText(String);

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct Newtype {
    table: String,
    filter: FilterText,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct NestedLists {
    table: String,
    fragments: Vec<Vec<String>>,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct AnyValue {
    table: String,
    opts: serde_json::Value,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct OptionalNewtype {
    table: String,
    maybe: Option<FilterText>,
}

#[derive(JsonSchema)]
#[serde(tag = "choice")]
#[allow(dead_code, reason = "only its schema is read")]
enum Choice {
    Plain { column: String },
    Ordered { column: String, note: String },
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct OptionalVariant {
    table: String,
    inner: Option<Choice>,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct MapOfText {
    table: String,
    options: std::collections::BTreeMap<String, String>,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct Recursive {
    column: String,
    label: String,
    next: Option<Box<Recursive>>,
}

#[test]
fn a_sql_field_is_refused() {
    refused::<WithSql>("Denied(\"#/oneOf/0/sql\")");
}

#[test]
fn a_newtype_of_text_is_refused() {
    refused::<Newtype>("Text(\"#/filter -> FilterText\")");
}

#[test]
fn nested_lists_of_text_are_refused() {
    refused::<NestedLists>("Text(\"#/fragments/items/items\")");
}

#[test]
fn an_any_typed_value_is_refused() {
    refused::<AnyValue>("Any(\"#/opts\")");
}

#[test]
fn an_optional_newtype_of_text_is_refused() {
    refused::<OptionalNewtype>("FilterText\")");
}

#[test]
fn text_in_a_referenced_optional_variant_is_refused() {
    refused::<OptionalVariant>("Choice/oneOf/1/note");
}

#[test]
fn a_map_of_text_is_refused() {
    refused::<MapOfText>("Text(\"#/options/*\")");
}

#[test]
fn a_cycle_ends_and_its_text_is_refused() {
    refused::<Recursive>("/label");
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct Tuple {
    table: String,
    pair: (u8, String),
}

#[derive(JsonSchema)]
#[serde(untagged)]
#[allow(dead_code, reason = "only its schema is read")]
enum Untagged {
    Count(u64),
    Words(String),
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct UntaggedField {
    table: String,
    bound: Untagged,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct Flattened {
    table: String,
    #[serde(flatten)]
    rest: Newtype,
}

#[test]
fn text_in_a_tuple_is_refused() {
    refused::<Tuple>("Text(\"#/pair/prefixItems/1\")");
}

#[test]
fn text_in_an_untagged_variant_is_refused() {
    refused::<UntaggedField>("Untagged/anyOf/1");
}

#[test]
fn text_in_a_flattened_struct_is_refused() {
    refused::<Flattened>("FilterText");
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct FilterScalar {
    table: String,
    filter: sdm_contract::Scalar,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct PredicateField {
    table: String,
    predicate_expr: sdm_contract::Field,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct OrderValues {
    table: String,
    sort_order: Option<Vec<sdm_contract::Scalar>>,
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct RawSql {
    table: String,
    raw_sql: Vec<u8>,
}

#[test]
fn a_typed_value_named_as_a_filter_is_refused() {
    refused::<FilterScalar>("Denied(\"#/filter -> Scalar\")");
}

#[test]
fn a_typed_value_named_with_a_part_of_an_expression_is_refused() {
    refused::<PredicateField>("Denied(\"#/predicate_expr\")");
    refused::<PredicateField>("Denied(\"#/predicate_expr -> Field\")");
}

#[test]
fn typed_values_under_an_order_are_refused_at_any_depth() {
    refused::<OrderValues>("Denied(\"#/sort_order/items -> Scalar\")");
}

#[test]
fn a_part_of_a_name_is_read_on_any_property() {
    refused::<RawSql>("Denied(\"#/raw_sql\")");
}

#[test]
fn the_vocabulary_s_own_order_and_filters_are_typed_structures() {
    // `order` holds a tie-break of named columns and flags, and `filters`
    // the typed filters; neither is a typed value, so both pass.
    let found = findings(&schema::<DataRequest>());
    assert!(
        found
            .iter()
            .all(|f| !matches!(f, Finding::Denied(_) | Finding::Text(_))),
        "{found:#?}"
    );
    let at = |field: &str| {
        found
            .iter()
            .any(|f| matches!(f, Finding::Identifier(p) if p.contains(field)))
    };
    assert!(at("/order") && at("/seed_column"), "{found:#?}");
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct FiltersScalars {
    table: String,
    filters: Vec<sdm_contract::Scalar>,
}

#[test]
fn typed_values_named_as_filters_are_refused() {
    refused::<FiltersScalars>("Denied(\"#/filters/items -> Scalar\")");
}
