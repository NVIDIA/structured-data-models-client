// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Metric-view parsing and substitution with explicit expected facts.
use sdm_data::view_expr::metric::{
    self, JoinRef, KeyRef, MeasureInput, MeasureReason, Parameter, ParameterRead, Unread,
};

/// Python's white space (`str.isspace`), which the core hands in from
/// `sdm-pql`: the 29 characters Python 3.12 counts, with the class the
/// patterns are compiled with derived from them.
fn spaces() -> &'static Spaces {
    static SPACES: std::sync::LazyLock<Spaces> = std::sync::LazyLock::new(|| {
        Spaces::new(|c| {
            matches!(
                c,
                '\t'..='\r'
                    | '\u{1c}'..='\u{20}'
                    | '\u{85}'
                    | '\u{a0}'
                    | '\u{1680}'
                    | '\u{2000}'..='\u{200a}'
                    | '\u{2028}'
                    | '\u{2029}'
                    | '\u{202f}'
                    | '\u{205f}'
                    | '\u{3000}'
            )
        })
    });
    &SPACES
}

use sdm_data::view_expr::Spaces;

#[test]
fn identifiers_and_join_conditions_preserve_quoted_parts() {
    for input in ["cat.sch.`order lines`", " cat.sch.`order lines` "] {
        assert_eq!(
            metric::table_reference(input, spaces()),
            Some(vec!["cat".into(), "sch".into(), "order lines".into()])
        );
        assert_eq!(
            metric::view_name(input, spaces()),
            Some(vec!["cat".into(), "sch".into(), "order lines".into()])
        );
    }
    for input in ["SELECT * FROM t", "a.b.c.d", "a . b", "", "a;drop"] {
        assert_eq!(metric::table_reference(input, spaces()), None, "{input}");
    }
    assert_eq!(metric::unquote("`a``b`", spaces()), "a`b");
    let expected = [
        KeyRef {
            qualifier: Some("source".into()),
            column: "user_id".into(),
        },
        KeyRef {
            qualifier: Some("users".into()),
            column: "id".into(),
        },
    ];
    assert_eq!(
        metric::join_condition("source.user_id = users.id", spaces()),
        Some(expected)
    );
    for condition in ["a > b", "a = b AND c = d", "a = sum(b)"] {
        assert_eq!(metric::join_condition(condition, spaces()), None);
    }
}

#[test]
fn alias_and_derived_column_rules_distinguish_literals_and_expressions() {
    assert_eq!(
        metric::references_alias("users.id + 'users.id'", &["users"], spaces()),
        Ok(true)
    );
    assert_eq!(
        metric::references_alias("'users.id'", &["users"], spaces()),
        Ok(false)
    );
    assert_eq!(
        metric::strip_alias("users.id + 'users.id'", &["users"], spaces()),
        Ok("id + 'users.id'".into())
    );
    assert_eq!(
        metric::derived_column("DATE_TRUNC('MONTH', event_time)", spaces()),
        Some("event_time".into())
    );
    assert_eq!(
        metric::derived_column("CAST(event_time AS DATE)", spaces()),
        Some("event_time".into())
    );
    assert_eq!(metric::derived_column("event_time", spaces()), None);
    assert_eq!(
        metric::renamed_column("`event time`", spaces()),
        Some("event time".into())
    );
    assert_eq!(metric::renamed_column("event_time + 1", spaces()), None);
    assert_eq!(
        metric::wildcard("users.* EXCEPT (`private`, id)", spaces()),
        Some((vec!["users".into()], vec!["private".into(), "id".into()]))
    );
}

/// A type's class, as `_default_text` tells types apart: its words in
/// any case and spacing.
#[test]
fn a_parameter_s_type_is_classed_as_the_old_client_classed_it() {
    use metric::TypeClass;
    for (data_type, class) in [
        ("BOOLEAN", TypeClass::Boolean),
        ("bigint", TypeClass::Integer),
        ("INT", TypeClass::Integer),
        ("DOUBLE", TypeClass::Decimal),
        ("decimal( 10 , 2 )", TypeClass::Decimal),
        ("DEC", TypeClass::Decimal),
        ("STRING", TypeClass::String),
        ("varchar(3)", TypeClass::String),
        ("DATE", TypeClass::Date),
        ("timestamp_ntz", TypeClass::Timestamp),
        ("TIMESTAMP\u{1c}", TypeClass::Timestamp),
        ("ARRAY<INT>", TypeClass::NoLiteral),
        ("INTERVAL DAY", TypeClass::NoLiteral),
    ] {
        assert_eq!(
            metric::type_class(data_type, spaces()),
            class,
            "{data_type}"
        );
    }
}

#[test]
fn parameters_replace_bare_names_once_without_touching_literals_or_qualified_names() {
    let parameters = vec![
        (
            "limit".into(),
            Parameter {
                name: "Limit".into(),
                read: Ok(ParameterRead {
                    literal: "12".into(),
                    text: "12".into(),
                }),
            },
        ),
        (
            "missing".into(),
            Parameter {
                name: "Missing".into(),
                read: Err(Unread::NoDefault),
            },
        ),
    ];
    assert_eq!(
        metric::substitute_parameters(
            "limit + LIMIT + source.limit + 'limit' + missing",
            &parameters,
            spaces()
        ),
        Ok((
            "12 + 12 + source.limit + 'limit' + missing".into(),
            vec!["Limit".into(), "Missing".into()]
        ))
    );
    assert!(metric::substitute_parameters("'unclosed", &parameters, spaces()).is_err());
}

#[test]
fn field_substitution_preserves_precedence_and_qualified_source_names() {
    let fields = vec![
        ("total".into(), Some("price * quantity".into())),
        ("bad".into(), None),
    ];
    assert_eq!(
        metric::substitute_fields(
            "total + source.total + 'total'",
            &fields,
            &[vec!["source".into()]],
            spaces()
        ),
        Ok(("(price * quantity) + source.total + 'total'".into(), None))
    );
    assert_eq!(
        metric::substitute_fields("BAD + 1", &fields, &[], spaces()),
        Ok(("BAD + 1".into(), Some("BAD".into())))
    );
    assert_eq!(metric::reads_star("struct(*)", spaces()), Ok(true));
    assert_eq!(metric::reads_star("COUNT(*)", spaces()), Ok(false));
    assert_eq!(metric::reads_star("price * quantity", spaces()), Ok(false));
}

#[test]
fn filters_cannot_reach_joined_tables_or_run_statements() {
    let candidates = vec![(vec!["users".into()], "users".into())];
    assert_eq!(
        metric::check_filter(Some("  price > 0  "), &candidates, spaces()),
        Ok("price > 0".into())
    );
    assert_eq!(
        metric::check_filter(None, &candidates, spaces()),
        Err(metric::FilterRefusal::NotACondition)
    );
    assert_eq!(
        metric::check_filter(Some("users.id > 0"), &candidates, spaces()),
        Err(metric::FilterRefusal::JoinedTable("users".into()))
    );
    assert!(matches!(
        metric::check_filter(Some("x; DROP TABLE t"), &candidates, spaces()),
        Err(metric::FilterRefusal::Unsafe(_))
    ));
}

#[test]
fn measures_capture_inputs_and_refuse_references_to_rejected_dimensions() {
    let measures = [
        MeasureInput {
            name: "revenue".into(),
            expr: Some("SUM(amount)".into()),
            window: false,
        },
        MeasureInput {
            name: "rows".into(),
            expr: Some("COUNT(*)".into()),
            window: false,
        },
        MeasureInput {
            name: "bad".into(),
            expr: Some("SUM(rejected)".into()),
            window: false,
        },
        MeasureInput {
            name: "empty".into(),
            expr: None,
            window: false,
        },
    ];
    let (read, refused) = metric::read_measures(
        &measures,
        &[],
        &["amount".into()],
        &[],
        &["rejected".into()],
        &[],
        spaces(),
    );
    assert_eq!(read.len(), 2);
    assert_eq!(read[0].name, "revenue");
    assert_eq!(read[0].inputs, [("source".into(), "amount".into())]);
    assert_eq!(
        read[0].aggregation,
        Some(("SUM".into(), Some("amount".into())))
    );
    assert_eq!(read[1].name, "rows");
    assert_eq!(read[1].aggregation, Some(("COUNT".into(), None)));
    assert_eq!(refused.len(), 1);
    assert_eq!(refused[0].name, "bad");
    assert_eq!(
        refused[0].reason,
        MeasureReason::NamesRefused("rejected".into())
    );
}

#[test]
fn join_key_resolution_respects_sides_case_and_child_first_ambiguity() {
    let key = |qualifier: Option<&str>, column: &str| KeyRef {
        qualifier: qualifier.map(str::to_owned),
        column: column.into(),
    };
    let mut join = JoinRef {
        alias: "users".into(),
        parent_alias: "source".into(),
        path: vec!["users".into()],
        condition: [key(Some("SOURCE"), "user_id"), key(Some("users"), "id")],
    };
    let expected = Some(("source".into(), "user_id".into(), "id".into()));
    assert_eq!(
        metric::resolve_join_keys(&join, &["id"], &["user_id"]),
        expected
    );
    join.condition.swap(0, 1);
    assert_eq!(
        metric::resolve_join_keys(&join, &["id"], &["user_id"]),
        expected
    );
    join.condition = [key(None, "user_id"), key(None, "id")];
    assert_eq!(
        metric::resolve_join_keys(&join, &["id"], &["user_id"]),
        expected
    );
    assert_eq!(
        metric::resolve_join_keys(&join, &["user_id", "id"], &["user_id"]),
        None
    );
    join.condition[0] = key(Some("unknown"), "id");
    assert_eq!(
        metric::resolve_join_keys(&join, &["id"], &["user_id"]),
        None
    );
}

#[test]
fn measures_do_not_claim_simple_aggregation_for_windows_or_cross_table_inputs() {
    let measure = |name: &str, expr: &str, window| MeasureInput {
        name: name.into(),
        expr: Some(expr.into()),
        window,
    };
    let measures = [
        measure("windowed", "SUM(amount)", true),
        measure("cross", "SUM(source.amount + users.credit)", false),
        measure("joined", "SUM(users.credit)", false),
    ];
    let candidates = [
        (vec!["source".into()], "source".into()),
        (vec!["users".into()], "users".into()),
    ];
    let (read, refused) = metric::read_measures(
        &measures,
        &candidates,
        &["users".into()],
        &[],
        &[],
        &[],
        spaces(),
    );
    assert!(refused.is_empty());
    assert_eq!(read[0].inputs, [("source".into(), "amount".into())]);
    assert!(read[0].aggregation.is_none());
    assert!(read[1].inputs.is_empty());
    assert!(read[1].aggregation.is_none());
    assert_eq!(read[2].inputs, [("users".into(), "credit".into())]);
    assert!(read[2].aggregation.is_none());
}

#[test]
fn dimensions_parameters_and_measure_names_have_distinct_namespaces() {
    let measure = |name: &str, expr: &str| MeasureInput {
        name: name.into(),
        expr: Some(expr.into()),
        window: false,
    };
    let dimensions = [(
        "revenue".into(),
        metric::DimensionRef {
            name: "Revenue".into(),
            alias: "source".into(),
            expr: "price * quantity".into(),
        },
    )];
    let measures = [
        measure("total", "SUM(revenue)"),
        measure("unique", "COUNT(DISTINCT revenue)"),
        measure("adjusted", "total + tax"),
        measure("non_count_distinct", "SUM(DISTINCT revenue)"),
    ];
    let (read, refused) = metric::read_measures(
        &measures,
        &[],
        &[],
        &dimensions,
        &["total".into(), "tax".into()],
        &["tax".into()],
        spaces(),
    );
    assert!(refused.is_empty());
    assert_eq!(read.len(), 4);
    assert!(read[0].inputs.is_empty()); // A dimension is resolved separately from raw columns.
    assert_eq!(
        read[0].aggregation,
        Some(("SUM".into(), Some("Revenue".into())))
    );
    assert_eq!(
        read[1].aggregation,
        Some(("COUNT_DISTINCT".into(), Some("Revenue".into())))
    );
    assert!(read[2].inputs.is_empty()); // Neither a parameter nor another measure is a source column.
    assert!(read[2].aggregation.is_none());
    assert!(read[3].aggregation.is_none());
}
