// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data-request vocabulary carries typed requests rather than core-built
//! SQL (E28), with precisely reviewed opaque caller-expression seams.
//! KO3-44 extends the existing relation seam only to a closed neighbor limit.
//!
//! A `DataRequest` names what it wants with typed fields; a provider, or
//! `sdm-data` for a SQL provider, decides how to fetch it. So no field of
//! the vocabulary, at any depth, may hold a free-text expression or a SQL
//! string outside those exact seams. The test reads the generated JSON
//! Schema of `DataRequest` and
//! classifies every property's schema recursively: through local `$ref`s
//! (each definition once per path, so a cycle ends), `anyOf`, `oneOf` and
//! `allOf`, list items at any depth, nested objects and map values. It
//! holds:
//!
//! - outside the exact reviewed seams, no property is named as SQL or an
//!   expression (`DENIED`), anywhere,
//!   the typed values included, matched against the whole name and against
//!   each of its `_`-separated parts, so `predicate_expr` is refused;
//! - a property typed as a reviewed typed value is refused also when its
//!   name, or a part of it, is `filter` or `order` (`DENIED_FOR_VALUES`):
//!   a scalar or a field there would be a condition or an ordering in
//!   disguise, where the vocabulary has typed structures for both;
//! - text (a string, at any depth of lists, maps, options or newtypes) is
//!   held only by a property named as an identifier the request refers to
//!   (`IDENTIFIERS`), the nearest property naming it, or by the exact
//!   checked numeric-token value or scoped opaque-expression seam below;
//! - an any-typed schema (`true` or `{}`, as `serde_json::Value` gives, or a
//!   map of anything) is refused, since it can hold text of any shape;
//! - the typed values (`TYPED_VALUES`: a scalar, an Arrow field and its
//!   type) are reviewed: a reference to one of them is accepted at that
//!   reference, and only the name check goes inside.
//!
//! A new text field fails until it is added to `IDENTIFIERS` in review, so
//! it is a name the request refers to, never text a provider interprets.
//! The reviewed exceptions retain their exact definition, property and shape.

use std::collections::BTreeSet;

use schemars::JsonSchema;
use sdm_contract::data::DataRequest;
use serde_json::{Map, Value};

/// SQL or expression property names, refused outside the exact reviewed seams.
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
    // A relation's name parts, each quoted alone (design choice 12,
    // contract v26): the source a request reads, never text interpreted.
    "source",
    // The name a derived relation gives a column (contract v28).
    "name",
];

/// The reviewed exceptions of contract v28 (Phase 3, design choice 3): a
/// derived relation's filter and expression columns, SQL text of the
/// source's own that the core carries opaque, never reading or writing it,
/// which only `sdm-data`'s dialects render and only for a provider
/// declaring `derived_relations`. Each is allowed only as that property of
/// that definition.
const OPAQUE_SQL: &[(&str, &str)] = &[("Relation", "filter"), ("RelationColumn", "expression")];

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
    /// A checked lossless numeric token, never arbitrary text.
    Number(String),
    /// Caller code only in the precisely closed reviewed neighbor seam.
    Opaque(String),
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

/// P3D-7: only `Window.source_type` holds a catalog declaration as plain
/// source metadata. The dialect matches fixed DATE/TIMESTAMP spellings;
/// it never interpolates the declaration into SQL. Do not allow this name
/// elsewhere, or an array, map, reference or any-typed payload here.
fn window_catalog_type(schema: &Value) -> bool {
    let Some(map) = schema.as_object() else {
        return false;
    };
    let kinds = types(map);
    map.keys()
        .all(|key| key == "type" || ANNOTATIONS.contains(&key.as_str()))
        && kinds.contains(&"string")
        && kinds.iter().all(|kind| matches!(*kind, "string" | "null"))
}

/// Exact schema keywords, allowing only harmless annotations beside them.
fn exact_schema(schema: &Value, expected: &[(&str, Value)]) -> bool {
    let Some(map) = schema.as_object() else {
        return false;
    };
    map.keys().all(|key| {
        ANNOTATIONS.contains(&key.as_str())
            || expected.iter().any(|(name, _)| key.as_str() == *name)
    }) && expected
        .iter()
        .all(|(key, value)| map.get(*key) == Some(value))
}

fn checked_negative_number(schema: &Value) -> bool {
    exact_schema(
        schema,
        &[
            ("type", serde_json::json!("string")),
            (
                "pattern",
                serde_json::json!(
                    r"^-(?:[1-9][0-9]*(?:\.[0-9]+)?|0\.0*[1-9][0-9]*)(?:[eE][+-]?[0-9]+)?$"
                ),
            ),
        ],
    )
}

/// One required property, no additional fields, and its exact child schema.
fn closed_bound(schema: &Value, property: &str, child: &[(&str, Value)]) -> bool {
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return false;
    };
    properties.len() == 1
        && properties
            .get(property)
            .is_some_and(|value| exact_schema(value, child))
        && exact_schema(
            schema,
            &[
                ("type", serde_json::json!("object")),
                ("properties", Value::Object(properties.clone())),
                ("required", serde_json::json!([property])),
                ("additionalProperties", Value::Bool(false)),
            ],
        )
}

/// KO3-44: preserve unsigned counts, checked negative numeric text and one
/// distinct caller-expression property. Never admit a whole unreviewed type.
fn closed_neighbor_limit(schema: &Value, defs: &Map<String, Value>) -> bool {
    let Some(branches) = schema.get("anyOf").and_then(Value::as_array) else {
        return false;
    };
    branches.len() == 3
        && exact_schema(schema, &[("anyOf", Value::Array(branches.clone()))])
        && exact_schema(
            &branches[0],
            &[
                ("type", serde_json::json!("integer")),
                ("format", serde_json::json!("uint64")),
                ("minimum", serde_json::json!(0)),
            ],
        )
        && closed_bound(
            &branches[1],
            "negative",
            &[("$ref", serde_json::json!("#/$defs/NegativeNeighborBound"))],
        )
        && defs
            .get("NegativeNeighborBound")
            .is_some_and(checked_negative_number)
        && closed_bound(
            &branches[2],
            "sql_expression",
            &[("type", serde_json::json!("string"))],
        )
}

impl Walk<'_> {
    /// Classify `schema`, reached at `path`, under the nearest property
    /// `name` (None at the root); `seen` holds the definitions entered on
    /// this path. `definition` names only this exact definition root,
    /// never an inline descendant reached through its structure.
    fn classify(
        &mut self,
        schema: &Value,
        path: &str,
        name: Option<&str>,
        seen: &[String],
        definition: Option<&str>,
    ) {
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
                Some(target)
                    if def == "NegativeNeighborBound" && checked_negative_number(target) =>
                {
                    if name.is_some_and(|property| denied(property, true)) {
                        self.found.push(Finding::Denied(at.clone()));
                    }
                    self.found.push(Finding::Number(at));
                }
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
                Some(target) => self.classify(target, &at, name, &inner, Some(def)),
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
                if definition == Some("NeighborLimit")
                    && key == "anyOf"
                    && i == 2
                    && closed_neighbor_limit(schema, self.defs)
                {
                    // The full closed branch has exactly this one string.
                    // Other branches still traverse the ordinary walker.
                    self.found
                        .push(Finding::Opaque(format!("{path}/{key}/{i}/sql_expression")));
                } else {
                    self.classify(branch, &format!("{path}/{key}/{i}"), name, seen, None);
                }
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
                        self.classify(item, &format!("{path}/items/{i}"), name, seen, None);
                    }
                }
                Some(items) => self.classify(items, &format!("{path}/items"), name, seen, None),
            }
            if let Some(more) = map.get("prefixItems").and_then(Value::as_array) {
                for (i, item) in more.iter().enumerate() {
                    self.classify(item, &format!("{path}/prefixItems/{i}"), name, seen, None);
                }
            }
        }
        if let Some(properties) = map.get("properties").and_then(Value::as_object) {
            for (property, child) in properties {
                let at = format!("{path}/{property}");
                if definition == Some("Window")
                    && property == "source_type"
                    && window_catalog_type(child)
                {
                    self.found.push(Finding::Identifier(at));
                    continue;
                }
                if seen
                    .last()
                    .is_some_and(|def| OPAQUE_SQL.iter().any(|(d, p)| d == def && p == property))
                {
                    self.found.push(Finding::Identifier(at));
                    continue;
                }
                if denied(property, false) {
                    self.found.push(Finding::Denied(at.clone()));
                }
                self.classify(child, &at, Some(property), seen, None);
            }
        }
        match map.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(values) => self.classify(values, &format!("{path}/*"), name, seen, None),
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
    walk.classify(schema, "#", None, &[], None);
    walk.found
}

/// What in a vocabulary's schema could carry SQL or an expression.
fn problems(schema: &Value) -> Vec<Finding> {
    findings(schema)
        .into_iter()
        .filter(|f| {
            !matches!(
                f,
                Finding::Identifier(_) | Finding::Number(_) | Finding::Opaque(_)
            )
        })
        .collect()
}

#[test]
fn data_requests_hold_no_expression_or_sql() {
    let found = problems(&schema::<DataRequest>());
    assert!(found.is_empty(), "{found:#?}");
}

#[test]
fn the_window_catalog_type_is_a_scoped_source_fact() {
    for plain in [
        serde_json::json!({"type": "string"}),
        serde_json::json!({"type": ["string", "null"]}),
    ] {
        let mut request = schema::<DataRequest>();
        request["$defs"]["Window"]["properties"]["source_type"] = plain;
        let found = findings(&request);
        assert!(
            found.iter().any(|finding| {
                matches!(finding, Finding::Identifier(path) if path.ends_with(" -> Window/source_type"))
            }),
            "{found:#?}"
        );
        assert!(problems(&request).is_empty(), "{found:#?}");
    }
}

#[test]
fn inline_descendants_of_window_cannot_inherit_the_catalog_exception() {
    let nested = serde_json::json!({
        "type": "object",
        "properties": {"source_type": {"type": ["string", "null"]}},
        "additionalProperties": false,
    });
    for descendant in [
        nested.clone(),
        serde_json::json!({"type": "array", "items": nested.clone()}),
        serde_json::json!({"type": "object", "additionalProperties": nested.clone()}),
        serde_json::json!({"allOf": [nested]}),
    ] {
        let mut request = schema::<DataRequest>();
        request["$defs"]["Window"]["properties"]["unreviewed"] = descendant;
        let found = problems(&request);
        assert!(
            found.iter().any(|finding| {
                matches!(finding, Finding::Text(path) if path.contains(" -> Window/unreviewed/") && path.ends_with("/source_type"))
            }),
            "{found:#?}"
        );
    }
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct UnreviewedSourceType {
    source_type: Option<String>,
}

#[test]
fn a_source_type_name_outside_the_window_is_not_an_identifier() {
    refused::<UnreviewedSourceType>("Text(\"#/source_type\")");
}

#[test]
fn a_window_catalog_type_cannot_expand_to_structured_or_untyped_data() {
    for widened in [
        serde_json::json!({"type": "array", "items": {"type": "string"}}),
        serde_json::json!({"type": "object", "additionalProperties": {"type": "string"}}),
        Value::Bool(true),
    ] {
        let mut request = schema::<DataRequest>();
        request["$defs"]["Window"]["properties"]["source_type"] = widened;
        let found = problems(&request);
        assert!(
            found.iter().any(|finding| {
                matches!(finding, Finding::Text(path) | Finding::Any(path) if path.contains(" -> Window/source_type"))
            }),
            "{found:#?}"
        );
    }
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

#[test]
fn only_the_closed_neighbor_seam_carries_checked_numbers_and_opaque_expressions() {
    let request = schema::<DataRequest>();
    assert!(problems(&request).is_empty());
    let found = findings(&request);
    assert!(
        found
            .iter()
            .any(|f| matches!(f, Finding::Number(path) if path.contains("NegativeNeighborBound"))),
        "{found:#?}"
    );
    assert!(
        found
            .iter()
            .any(|f| matches!(f, Finding::Opaque(path) if path.ends_with("/sql_expression"))),
        "{found:#?}"
    );
    assert!(!found.iter().any(|f| matches!(f, Finding::Identifier(path) if path.contains("NeighborLimit") || path.contains("NegativeNeighborBound"))));
}

#[test]
fn the_numeric_token_schema_cannot_widen_or_become_arbitrary_text() {
    for widened in [
        serde_json::json!({"type":"string"}),
        serde_json::json!({"type":"string","pattern":".*"}),
        serde_json::json!({"type":"array","items":{"type":"string"}}),
        Value::Bool(true),
    ] {
        let mut request = schema::<DataRequest>();
        request["$defs"]["NegativeNeighborBound"] = widened;
        assert!(!problems(&request).is_empty(), "{request:#?}");
    }
}

#[test]
fn neighbor_expression_shape_drift_never_inherits_the_reviewed_seam() {
    let original = schema::<DataRequest>();
    let object = serde_json::json!({"type":"object", "properties":{"sql_expression":{"type":"string"}}, "required":["sql_expression"], "additionalProperties":false});
    let mut versions = Vec::new();
    let mut extra = original.clone();
    extra["$defs"]["NeighborLimit"]["anyOf"][2]["properties"]["note"] =
        serde_json::json!({"type":"string"});
    versions.push(extra);
    for branch in [1, 2] {
        let mut open = original.clone();
        open["$defs"]["NeighborLimit"]["anyOf"][branch]
            .as_object_mut()
            .unwrap()
            .remove("additionalProperties");
        versions.push(open);
        let mut optional = original.clone();
        optional["$defs"]["NeighborLimit"]["anyOf"][branch]
            .as_object_mut()
            .unwrap()
            .remove("required");
        versions.push(optional);
    }
    for widened in [
        Value::Bool(true),
        serde_json::json!({"type":"array","items":{"type":"string"}}),
        serde_json::json!({"type":"object","additionalProperties":{"type":"string"}}),
    ] {
        let mut request = original.clone();
        request["$defs"]["NeighborLimit"]["anyOf"][2]["properties"]["sql_expression"] = widened;
        versions.push(request);
    }
    for nested in [
        serde_json::json!({"type":"array","items":object.clone()}),
        serde_json::json!({"type":"object","additionalProperties":object.clone()}),
        serde_json::json!({"allOf":[object]}),
    ] {
        let mut request = original.clone();
        request["$defs"]["NeighborLimit"]["anyOf"][2] = nested;
        versions.push(request);
    }
    let mut number = original.clone();
    number["$defs"]["NeighborLimit"]["anyOf"][0]["type"] = Value::String("number".into());
    versions.push(number);
    let mut extra_branch = original.clone();
    extra_branch["$defs"]["NeighborLimit"]["anyOf"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({"type":"string"}));
    versions.push(extra_branch);
    for request in versions {
        let found = problems(&request);
        assert!(!found.is_empty(), "{request:#?}");
        assert!(
            !findings(&request)
                .iter()
                .any(|f| matches!(f, Finding::Opaque(_))),
            "{found:#?}"
        );
    }
}

#[derive(JsonSchema)]
#[allow(dead_code, reason = "only its schema is read")]
struct CheckedNumberAsFilter {
    table: String,
    filter: sdm_contract::data::NegativeNeighborBound,
}

#[test]
fn a_checked_numeric_token_cannot_be_repurposed_as_a_filter() {
    refused::<CheckedNumberAsFilter>("Denied(\"#/filter -> NegativeNeighborBound\")");
}
