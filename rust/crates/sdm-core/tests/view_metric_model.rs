// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric-view definitions through schema parsing and dialect-aware resolution.
//! Authored inputs assert joins, parameters, dimensions, measures and refusals.

use sdm_contract::CoreError;
use sdm_contract::data::{DataLayer, ProviderChoice};
use sdm_contract::view::{ViewTextAnswer, ViewTextAsk};
use sdm_core::pql::{spaces, yaml_rules};
use sdm_core::view::definition::{self as view_definition, Dimension};
use sdm_core::view_steps::{self, JoinJudgement};
use sdm_core::view_words;
use sdm_data::Data;
use sdm_data::view_expr::metric as data;
use sdm_data::view_expr::strip_comments;
use sdm_schema::view::joins::JoinWalk;
use sdm_schema::view::metric;
use sdm_schema::view::spec::{Cardinality, JoinKeys, RefusalReason, Source, TableItem, ViewSpec};
use sdm_schema::view::yaml::Value;
use serde_json::Value as Json;

/// The data layer's answers in Databricks's dialect, as the routine asks
/// them (`DataLayer::view_text`).
fn ask(question: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
    Data::reading_views(spaces()).view_text(&ProviderChoice::sql("databricks"), question)
}

/// `unquote_ident`: a name in backticks, unquoted, else as written.
fn unquoted(name: &str) -> String {
    data::unquote(name, spaces())
}

/// Each kept join as the old answer writes it.
fn kept_joins(spec: &ViewSpec, walk: &JoinWalk<JoinJudgement>) -> Vec<Json> {
    walk.kept
        .iter()
        .map(|&at| {
            let TableItem::Table(table) = &spec.tables[at] else {
                unreachable!()
            };
            let mut path = vec![table.alias.clone()];
            let mut parent = table.parent;
            while let Some(index) = parent.filter(|&p| p > 0) {
                let TableItem::Table(up) = &spec.tables[index] else {
                    unreachable!()
                };
                path.insert(0, up.alias.clone());
                parent = up.parent;
            }
            let parent_alias = if path.len() > 1 {
                path[path.len() - 2].clone()
            } else {
                metric::SOURCE_ALIAS.to_owned()
            };
            let join = table.join.as_ref().unwrap();
            let Source::Name(Some(source)) = table.source else {
                unreachable!()
            };
            let parts = data::table_reference(spec.text(source).unwrap(), spaces()).unwrap();
            let condition = match join.keys {
                JoinKeys::Using(column) => {
                    let column = unquoted(&view_words::value_text(&spec.document, column).unwrap());
                    serde_json::json!([
                        {"qualifier": parent_alias, "column": column},
                        {"qualifier": table.alias, "column": column},
                    ])
                }
                JoinKeys::On(Some(on)) => {
                    let on = strip_comments(spec.text(on).unwrap(), spaces()).unwrap();
                    let [a, b] = data::join_condition(&on, spaces()).unwrap();
                    serde_json::json!([
                        {"qualifier": a.qualifier, "column": a.column},
                        {"qualifier": b.qualifier, "column": b.column},
                    ])
                }
                _ => unreachable!(),
            };
            serde_json::json!({
                "alias": table.alias,
                "table": parts,
                "parent_alias": parent_alias,
                "path": path,
                "condition": condition,
                "cardinality": match join.cardinality {
                    Cardinality::ManyToOne => "many_to_one",
                    Cardinality::OneToMany => "one_to_many",
                    Cardinality::Unsupported(_) | Cardinality::Unhashable => unreachable!(),
                },
            })
        })
        .collect()
}

/// The structure the data layer reads further, kept as written: a join's
/// source, `on` (or the key `true` a bare `on:` is, row 31), `using` and
/// cardinality; the filter; a parameter's type and default.
#[test]
fn a_join_keeps_what_it_says_as_written() {
    let spec = metric::parse(
        "source: c.s.orders\n\
         filter: source.price > 50 -- a comment\n\
         parameters:\n\
         \x20 - name: since\n\
         \x20   data_type: DATE\n\
         \x20   default: 2024-01-01\n\
         joins:\n\
         \x20 - name: users\n\
         \x20   source: c.s.users\n\
         \x20   on: source.user_id = users.user_id\n\
         \x20   joins:\n\
         \x20     - name: countries\n\
         \x20       source: c.s.countries\n\
         \x20       using: country\n\
         \x20       cardinality: one_to_many\n\
         \x20 - name: shops\n\
         \x20   source: c.s.shops\n\
         \x20   using: [a, b]\n\
         \x20   cardinality: many_to_many\n",
        &yaml_rules(),
    )
    .unwrap();
    assert_eq!(
        spec.text(spec.filter.unwrap()),
        Some("source.price > 50 -- a comment")
    );
    assert!(matches!(
        spec.value(spec.parameters[0].default.unwrap()),
        Value::Date(_)
    ));
    let tables: Vec<&str> = spec
        .tables
        .iter()
        .map(|t| match t {
            TableItem::Table(t) => t.alias.as_str(),
            _ => "?",
        })
        .collect();
    assert_eq!(tables, ["source", "users", "countries", "shops"]);
    let TableItem::Table(countries) = &spec.tables[2] else {
        panic!()
    };
    assert_eq!(countries.parent, Some(1));
    let join = countries.join.as_ref().unwrap();
    assert_eq!(join.cardinality, Cardinality::OneToMany);
    let JoinKeys::Using(column) = join.keys else {
        panic!()
    };
    assert_eq!(spec.text(column), Some("country"));
    let TableItem::Table(shops) = &spec.tables[3] else {
        panic!()
    };
    assert_eq!(shops.join.as_ref().unwrap().keys, JoinKeys::UsingSeveral);
}

/// Each dimension as the old answer writes it.
#[test]
fn metric_definition_resolves_parameter_filter_and_joined_dimensions() {
    let mut spec = metric::parse(
        r#"
source: catalog.sales.orders
parameters:
  - name: threshold
    data_type: INT
    default: 50
filter: source.price > threshold
joins:
  - name: users
    source: catalog.sales.users
    'on': source.user_id = users.user_id
dimensions:
  - name: order_id
    expr: source.order_id
  - name: country
    expr: users.country
measures:
  - name: revenue
    expr: SUM(source.price)
"#,
        &yaml_rules(),
    )
    .unwrap();
    assert!(view_steps::fact_source(&spec, &ask).unwrap().is_ok());
    let parameters = view_steps::parameters(&spec, &[], &ask).unwrap();
    let read = view_definition::read(&mut spec, &parameters, &ask, "databricks")
        .unwrap()
        .unwrap();
    assert_eq!(
        read.filter.as_deref(),
        Some("source.price > CAST('50' AS INT)")
    );
    assert_eq!(read.notes, [("threshold".to_owned(), "50".to_owned())]);
    assert_eq!(
        read.dimensions,
        [
            Dimension::Column {
                name: "order_id".into(),
                alias: "source".into(),
                expr: "order_id".into()
            },
            Dimension::Column {
                name: "country".into(),
                alias: "users".into(),
                expr: "country".into()
            },
        ]
    );
    assert!(read.skipped.is_empty());
    assert_eq!(read.joins.len(), 1);
    let join = &read.joins[0];
    assert_eq!(join.alias, "users");
    assert_eq!(join.parent_alias, "source");
    assert_eq!(join.path, ["users"]);
    assert_eq!(join.table, ["catalog", "sales", "users"]);
    assert!(join.many_to_one);
    assert_eq!(join.condition[0].qualifier.as_deref(), Some("source"));
    assert_eq!(join.condition[0].column, "user_id");
    assert_eq!(join.condition[1].qualifier.as_deref(), Some("users"));
    assert_eq!(join.condition[1].column, "user_id");
    assert_eq!(read.measures.len(), 1);
}

#[test]
fn metric_join_walk_preserves_nested_paths_and_reports_invalid_entries() {
    use sdm_schema::view::joins::{JoinMessage, SkipReason};
    let mut spec = metric::parse(
        r#"
source: c.s.orders
joins:
  - name: users
    source: c.s.users
    using: user_id
    joins:
      - name: countries
        source: c.s.countries
        using: country_id
  - name: rejected
    source: c.s.bad
    cardinality: many_to_many
    joins:
      - name: hidden
        source: c.s.hidden
        using: id
  - name: composite
    source: c.s.composite
    using: [a, b]
  - name: users
    source: c.s.duplicate
    using: user_id
  - invalid
  - source: c.s.nameless
"#,
        &yaml_rules(),
    )
    .unwrap();
    let walk = view_steps::walk_joins(&mut spec, &ask).unwrap().unwrap();
    let kept = kept_joins(&spec, &walk);
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0]["alias"], "users");
    assert_eq!(kept[1]["path"], serde_json::json!(["users", "countries"]));
    assert_eq!(kept[1]["parent_alias"], "users");
    assert_eq!(kept[1]["table"], serde_json::json!(["c", "s", "countries"]));
    assert_eq!(
        kept[1]["condition"],
        serde_json::json!([
            {"qualifier":"users","column":"country_id"},
            {"qualifier":"countries","column":"country_id"}
        ])
    );
    assert_eq!(walk.skipped.len(), 3);
    assert_eq!(walk.skipped[0].alias, "rejected");
    assert!(matches!(walk.skipped[0].reason, SkipReason::Cardinality(_)));
    assert_eq!(walk.skipped[0].nested, ["hidden"]);
    assert_eq!(walk.skipped[1].alias, "composite");
    assert!(matches!(walk.skipped[1].reason, SkipReason::UsingSeveral));
    assert_eq!(walk.skipped[2].alias, "users");
    assert!(matches!(walk.skipped[2].reason, SkipReason::NameInUse));
    assert!(matches!(
        walk.messages.as_slice(),
        [JoinMessage::NotAMapping(_), JoinMessage::Nameless(_)]
    ));
}

#[test]
fn metric_fields_fallback_and_dimension_precedence_are_explicit() {
    for (dimensions, expected, ignored) in [
        ("", "legacy_id", false),
        (
            "dimensions:\n  - name: current_id\n    expr: source.order_id\n",
            "current_id",
            true,
        ),
    ] {
        let text = format!(
            "source: c.s.orders\nfields:\n  - name: legacy_id\n    expr: source.order_id\n{dimensions}"
        );
        let mut spec = metric::parse(&text, &yaml_rules()).unwrap();
        let read = view_definition::read(&mut spec, &[], &ask, "databricks")
            .unwrap()
            .unwrap();
        assert_eq!(
            read.dimensions,
            [Dimension::Column {
                name: expected.into(),
                alias: "source".into(),
                expr: "order_id".into(),
            }]
        );
        assert_eq!(
            spec.ignored
                .contains(&sdm_schema::view::spec::Ignored::Fields),
            ignored
        );
    }
}

#[test]
fn metric_measures_keep_input_metadata_and_deduplicate_window_orders() {
    let mut spec = metric::parse(
        r#"
source: c.s.orders
measures:
  - name: revenue
    expr: SUM(source.price)
  - name: count
    expr: COUNT(*)
  - name: customers
    expr: COUNT(DISTINCT source.user_id)
  - name: moving
    expr: SUM(source.price)
    window:
      - order: '`ts`'
      - order: ts
      - order: order_id
"#,
        &yaml_rules(),
    )
    .unwrap();
    let read = view_definition::read(&mut spec, &[], &ask, "databricks")
        .unwrap()
        .unwrap();
    assert_eq!(read.window_orders, ["ts", "order_id"]);
    let metadata: Vec<_> = read
        .measures
        .iter()
        .map(|m| (m.name.as_str(), m.inputs.clone(), m.aggregation.clone()))
        .collect();
    assert_eq!(
        metadata,
        vec![
            (
                "revenue",
                vec![("source".into(), "price".into())],
                Some(("SUM".into(), Some("price".into())))
            ),
            ("count", vec![], Some(("COUNT".into(), None))),
            (
                "customers",
                vec![("source".into(), "user_id".into())],
                Some(("COUNT_DISTINCT".into(), Some("user_id".into())))
            ),
            ("moving", vec![("source".into(), "price".into())], None),
        ]
    );
}

#[test]
fn metric_source_and_filter_refusals_are_named() {
    assert!(matches!(
        metric::parse("[]", &yaml_rules()).unwrap_err().reason,
        RefusalReason::NotAMapping
    ));
    assert!(matches!(
        metric::parse("dimensions: []", &yaml_rules())
            .unwrap_err()
            .reason,
        RefusalReason::MissesSource
    ));
    for source in ["123", "[one, two]", "SELECT * FROM orders"] {
        let spec = metric::parse(&format!("source: {source}"), &yaml_rules()).unwrap();
        assert!(
            view_steps::fact_source(&spec, &ask).unwrap().is_err(),
            "{source}"
        );
    }
    let mut spec = metric::parse(
        "source: c.s.orders\nfilter: source.price > 0; DROP TABLE orders",
        &yaml_rules(),
    )
    .unwrap();
    let refusal = view_definition::read(&mut spec, &[], &ask, "databricks")
        .unwrap()
        .unwrap_err();
    assert_eq!(
        refusal.error().sentence,
        sdm_contract::Sentence::ViewFilterRefused
    );
    assert_eq!(refusal.filter, "'source.price > 0; DROP TABLE orders'");
    assert!(!refusal.reason.is_empty());
}

#[test]
fn metric_parameters_use_described_types_and_report_missing_defaults() {
    use sdm_contract::view::{ParameterRead, Unread};
    let spec = metric::parse(
        r#"
source: c.s.orders
parameters:
  - name: Threshold
    data_type: STRING
    default: 50
  - name: required
    data_type: INT
  - name: untyped
    default: 5
"#,
        &yaml_rules(),
    )
    .unwrap();
    let parameters =
        view_steps::parameters(&spec, &[("THRESHOLD".into(), "INT".into())], &ask).unwrap();
    assert_eq!(
        parameters
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["threshold", "required", "untyped"]
    );
    assert_eq!(parameters[0].1.name, "Threshold");
    assert_eq!(
        parameters[0].1.read,
        Ok(ParameterRead {
            literal: "CAST('50' AS INT)".into(),
            text: "50".into()
        })
    );
    assert_eq!(parameters[1].1.read, Err(Unread::NoDefault));
    assert_eq!(parameters[2].1.read, Err(Unread::NoType));
}

#[test]
fn metric_wildcard_exclusions_and_skipped_descendant_dimensions_are_preserved() {
    let mut spec = metric::parse(
        r#"
source: c.s.orders
joins:
  - name: rejected
    source: c.s.bad
    cardinality: many_to_many
    joins:
      - name: hidden
        source: c.s.hidden
        using: id
dimensions:
  - name: all_columns
    expr: source.* EXCEPT (secret, internal)
  - name: unavailable
    expr: hidden.value
"#,
        &yaml_rules(),
    )
    .unwrap();
    let read = view_definition::read(&mut spec, &[], &ask, "databricks")
        .unwrap()
        .unwrap();
    assert_eq!(
        read.dimensions,
        [Dimension::Wildcard {
            alias: "source".into(),
            excluded: vec!["secret".into(), "internal".into()],
        }]
    );
    assert!(read.joins.is_empty());
    assert_eq!(read.skipped.len(), 1);
    assert_eq!(read.skipped[0].alias, "rejected");
    assert_eq!(read.skipped[0].nested, ["hidden"]);
    assert_eq!(read.skipped[0].dimensions, ["unavailable"]);
}

#[test]
fn metric_filter_refusals_preserve_context_and_remediation() {
    use sdm_core::view_words::FilterHatch;
    for (filter, parameters, hatch, reason_fragment) in [
        ("users.age > 21", "", FilterHatch::None, "users"),
        (
            "source.price > threshold",
            "parameters:\n  - name: threshold\n    data_type: DOUBLE\n",
            FilterHatch::Default,
            "threshold",
        ),
    ] {
        let text = format!(
            "source: c.s.orders\n{parameters}filter: {filter}\njoins:\n  - name: users\n    source: c.s.users\n    using: user_id\n"
        );
        let mut spec = metric::parse(&text, &yaml_rules()).unwrap();
        let parameters = view_steps::parameters(&spec, &[], &ask).unwrap();
        let refused = view_definition::read(&mut spec, &parameters, &ask, "databricks")
            .unwrap()
            .unwrap_err();
        assert_eq!(refused.hatch, hatch);
        assert_eq!(
            refused.error().sentence,
            if hatch == FilterHatch::Default {
                sdm_contract::Sentence::ViewFilterRefusedNoDefault
            } else {
                sdm_contract::Sentence::ViewFilterRefused
            }
        );
        assert_eq!(refused.filter, format!("'{filter}'"));
        assert!(refused.reason.contains(reason_fragment));
    }
}

#[test]
fn metric_omissions_emit_specific_user_warnings() {
    use sdm_contract::Sentence;
    let mut spec = metric::parse(
        r#"
source: c.s.orders
materialization: {schedule: daily}
fields:
  - name: unused
    expr: source.price
dimensions:
  - broken
  - name: not_text
    expr: 123
  - name: valid
    expr: source.order_id
"#,
        &yaml_rules(),
    )
    .unwrap();
    let read = view_definition::read(&mut spec, &[], &ask, "databricks")
        .unwrap()
        .unwrap();
    assert_eq!(
        read.dimensions,
        [Dimension::Column {
            name: "valid".into(),
            alias: "source".into(),
            expr: "order_id".into(),
        }]
    );
    let sentences: Vec<_> = read
        .messages
        .iter()
        .map(|warning| warning.sentence)
        .collect();
    assert_eq!(
        sentences,
        [
            Sentence::ViewMaterializationIgnored,
            Sentence::ViewFieldsIgnored,
            Sentence::ViewDimensionNotAMapping,
            Sentence::ViewDimensionMisses
        ]
    );
    assert!(
        read.messages
            .iter()
            .all(|warning| warning.code == "VIEW_READ_IN_PART")
    );
}

#[test]
fn metric_unreadable_join_source_and_condition_skip_without_losing_valid_joins() {
    use sdm_schema::view::joins::SkipReason;
    let mut spec = metric::parse(
        r#"
source: c.s.orders
joins:
  - name: query
    source: SELECT * FROM users
    using: user_id
  - name: multiple
    source: c.s.users
    'on': source.user_id = multiple.user_id AND source.country = multiple.country
  - name: valid
    source: c.s.users
    using: user_id
"#,
        &yaml_rules(),
    )
    .unwrap();
    let walk = view_steps::walk_joins(&mut spec, &ask).unwrap().unwrap();
    assert_eq!(
        walk.skipped
            .iter()
            .map(|skip| skip.alias.as_str())
            .collect::<Vec<_>>(),
        ["query", "multiple"]
    );
    assert!(
        walk.skipped
            .iter()
            .all(|skip| matches!(skip.reason, SkipReason::Judged(_)))
    );
    let kept = kept_joins(&spec, &walk);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0]["alias"], "valid");
}
