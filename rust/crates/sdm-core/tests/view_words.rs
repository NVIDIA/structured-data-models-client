// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! User-facing view conversion messages. Expression refusals have direct
//! expectations for expressions, YAML values and parameter conversions.

use sdm_contract::CoreError;
use sdm_contract::data::{DataLayer, ProviderChoice};
use sdm_contract::view::{ViewKind, ViewTextAnswer, ViewTextAsk};
use sdm_core::pql::{spaces, yaml_rules};
use sdm_core::{view_steps, view_words};
use sdm_data::Data;
use sdm_data::view_expr::metric::{self as data, DefaultRefusal, FilterRefusal};
use sdm_data::view_expr::{self, ExprDialect};
use sdm_schema::view::metric;
use sdm_schema::view::model::{self, Edge, MeasureNote, TimeColumnNote};
use sdm_schema::view::yaml;

/// The data layer's answers in Databricks's dialect, as the routine asks
/// them (`DataLayer::view_text`).
fn ask(question: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
    Data::reading_views(spaces()).view_text(&ProviderChoice::sql("databricks"), question)
}

/// The dialect's own words for a refusal's reason, as the routine asks.
fn dialect_words(reason: &view_expr::ExprRefusal) -> Option<String> {
    match ask(ViewTextAsk::ReasonWords {
        reason: reason.clone(),
    })
    .unwrap()
    {
        ViewTextAnswer::Words(words) => words,
        _ => panic!("another answer"),
    }
}

#[test]
fn yaml_values_render_with_explicit_type_and_collection_rules() {
    for (input, repr, text) in [
        (
            "2024-01-02 03:04:00.000042-05:30",
            "datetime.datetime(2024, 1, 2, 3, 4, 0, 42, tzinfo=datetime.timezone(datetime.timedelta(days=-1, seconds=66600)))",
            "2024-01-02 03:04:00.000042-05:30",
        ),
        ("null", "None", "None"),
        ("yes", "True", "True"),
        ("017", "15", "15"),
        (
            "123456789012345678901234567890",
            "123456789012345678901234567890",
            "123456789012345678901234567890",
        ),
        ("-0.0", "-0.0", "-0.0"),
        (".nan", "nan", "nan"),
        (".inf", "inf", "inf"),
        ("-.inf", "-inf", "-inf"),
        ("hello", "'hello'", "hello"),
        ("[]", "[]", "[]"),
        ("{}", "{}", "{}"),
        (
            "{x: [null, yes]}",
            "{'x': [None, True]}",
            "{'x': [None, True]}",
        ),
        ("&a [*a]", "[[...]]", "[[...]]"),
        ("&a {x: *a}", "{'x': {...}}", "{'x': {...}}"),
        ("2024-02-29", "datetime.date(2024, 2, 29)", "2024-02-29"),
        (
            "2024-01-02 03:04:05.123456",
            "datetime.datetime(2024, 1, 2, 3, 4, 5, 123456)",
            "2024-01-02 03:04:05.123456",
        ),
        (
            "2024-01-02 03:04:05+05:30",
            "datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(seconds=19800)))",
            "2024-01-02 03:04:05+05:30",
        ),
    ] {
        let document = yaml::load(input, &yaml_rules()).unwrap();
        assert_eq!(
            view_words::value_repr(&document, document.root()).unwrap(),
            repr,
            "{input}"
        );
        assert_eq!(
            view_words::value_text(&document, document.root()).unwrap(),
            text,
            "{input}"
        );
    }
}

/// The message of a column whose expression is refused, as the core words
/// it: `column` as the old client's `_view_expr_message` names it.
fn expression_message(
    column: &str,
    expr: &str,
    view: ViewKind,
    dialect: ExprDialect,
    trust_expressions: bool,
    columns: &[String],
) -> Option<String> {
    let verdict =
        view_expr::expression_verdict(expr, dialect, trust_expressions, columns, spaces())?;
    let (name, table) = match column.split_once("' of table '") {
        Some((name, table)) => (
            name.trim_start_matches("column '"),
            Some(table.trim_end_matches('\'')),
        ),
        None => (
            column.trim_start_matches("column '").trim_end_matches('\''),
            None,
        ),
    };
    let reason = view_words::expr_reason(&verdict.reason, dialect_words(&verdict.reason));
    Some(view_words::expression_dropped(view, name, table, &reason, verdict.untrusted).message)
}

#[test]
fn trusted_calls_and_unsafe_sql_have_distinct_messages() {
    for (view, dialect) in [
        (ViewKind::Metric, ExprDialect::Databricks),
        (ViewKind::Semantic, ExprDialect::Snowflake),
    ] {
        assert_eq!(
            expression_message("column 'x'", "abs(x)", view, dialect, false, &["x".into()]),
            None
        );
        assert_eq!(
            expression_message(
                "column 'x'",
                "custom_function(x)",
                view,
                dialect,
                true,
                &["x".into()]
            ),
            None
        );
        let message = expression_message(
            "column 'x'",
            "custom_function(x)",
            view,
            dialect,
            false,
            &["x".into()],
        )
        .unwrap();
        assert!(message.contains("custom_function"));
        assert!(message.contains("not a pure built-in SQL function"));
        for trust in [false, true] {
            let message = expression_message(
                "column 'x'",
                "x; DROP TABLE t",
                view,
                dialect,
                trust,
                &["x".into()],
            )
            .unwrap();
            assert!(message.contains("it contains ';'"));
            assert!(message.contains("warehouse credentials"));
        }
    }
}

/// Rows 64 to 66 by name.
#[test]
fn unsafe_text_and_untrusted_calls_are_worded_by_name() {
    let reason = |expr: &str| {
        view_expr::unsafe_reason(expr, Some(ExprDialect::Databricks), spaces())
            .map(|r| view_words::expr_reason(&r, dialect_words(&r)))
    };
    assert_eq!(
        reason("x; DROP TABLE t").as_deref(),
        Some("it contains ';'")
    );
    assert_eq!(
        reason("trim(FROM users.country)").as_deref(),
        Some("it contains a sub-query")
    );
    assert_eq!(
        reason("é + 1").as_deref(),
        Some("it has 'é' outside quotes, which is not ASCII")
    );
    assert_eq!(
        reason("'a").as_deref(),
        Some("it leaves \"'\" unpaired, so which part of it is quoted text cannot be established")
    );
    assert_eq!(
        expression_message(
            "column 'x'",
            "trim(FROM users.country)",
            ViewKind::Metric,
            ExprDialect::Databricks,
            true,
            &["country".to_owned()],
        )
        .as_deref(),
        Some(
            "Failed to add column 'x' since it contains a sub-query, and expressions taken \
             from the metric view are executed as SQL under your warehouse credentials"
        )
    );
    let call = view_expr::untrusted_call_reason(
        "secret('s', 'k')",
        ExprDialect::Databricks,
        &["x"],
        spaces(),
    )
    .map(|r| view_words::expr_reason(&r, dialect_words(&r)));
    assert_eq!(
        call.as_deref(),
        Some("it calls 'secret', which is not a pure built-in SQL function")
    );
}

#[test]
fn every_expression_refusal_reason_keeps_its_wording() {
    for (expr, dialect, expected) in [
        (
            "\u{1c}select\u{1c}1",
            None,
            "it contains the SQL keyword 'select'",
        ),
        ("x --", None, "it contains '--'"),
        ("x)", None, "it closes a '(' it never opened"),
        ("(x", None, "it leaves a '(' unclosed"),
        (
            "a$$ + secret(x) + b$$",
            None,
            "it leaves '$$' unpaired, so which part of it is quoted text cannot be established",
        ),
        (
            "$$x$$",
            Some(ExprDialect::Databricks),
            "it contains '$$', which Databricks SQL does not quote with",
        ),
        (
            "\"MY\\COL\" * 2",
            Some(ExprDialect::Snowflake),
            "it has a backslash inside a double-quoted name",
        ),
    ] {
        let reason = view_expr::unsafe_reason(expr, dialect, spaces())
            .map(|r| view_words::expr_reason(&r, dialect_words(&r)));
        assert_eq!(reason.as_deref(), Some(expected), "{expr:?}");
    }
    let wide = ["a", "col", "d", "name", "order_date", "x", "y", "z"];
    for (expr, dialect, columns, expected) in [
        (
            "cat.sch.f(x)",
            ExprDialect::Databricks,
            &wide[..],
            "it calls the qualified function 'f'",
        ),
        (
            "`f`(x)",
            ExprDialect::Databricks,
            &wide[..],
            "it calls a function by a quoted name",
        ),
        (
            "a[0](x)",
            ExprDialect::Databricks,
            &wide[..],
            "it calls the result of an expression",
        ),
        (
            "identifier(x)",
            ExprDialect::Databricks,
            &["x"][..],
            "it names a function through IDENTIFIER()",
        ),
        (
            "user",
            ExprDialect::Databricks,
            &["x"][..],
            "it reads the session value 'user'",
        ),
        (
            "`user`",
            ExprDialect::Databricks,
            &["x"][..],
            "it reads the session value `user`",
        ),
        (
            "\"user\"",
            ExprDialect::Snowflake,
            &["x"][..],
            "it reads the session value \"user\"",
        ),
    ] {
        let reason = view_expr::untrusted_call_reason(expr, dialect, columns, spaces())
            .map(|r| view_words::expr_reason(&r, dialect_words(&r)));
        assert_eq!(reason.as_deref(), Some(expected), "{expr:?}");
    }
}

#[test]
fn yaml_defaults_have_explicit_values_and_refusals() {
    for (kind, input, expected) in [
        ("INT", "'0012'", "12"),
        ("DOUBLE", "'1.2500'", "1.2500"),
        ("DATE", "'20240229'", "2024-02-29"),
        (
            "TIMESTAMP",
            "'2024-02-29T12:00:00Z'",
            "2024-02-29 12:00:00+00:00",
        ),
    ] {
        let document = yaml::load(&format!("v: {input}"), &yaml_rules()).unwrap();
        let value = document.get(document.root(), "v").unwrap();
        let class = view_steps::type_class(&ask, kind).unwrap();
        assert_eq!(
            view_steps::default_text(class, &document, value),
            Ok(expected.into())
        );
    }
    let document = yaml::load("v: wrong", &yaml_rules()).unwrap();
    let value = document.get(document.root(), "v").unwrap();
    assert_eq!(
        view_words::default_refusal_text(&DefaultRefusal::NotOfType, "INT", &document, value)
            .unwrap(),
        "its default 'wrong' is not of type INT"
    );
    assert_eq!(
        view_words::default_refusal_text(&DefaultRefusal::NoLiteral, "ARRAY", &document, value)
            .unwrap(),
        "its type 'ARRAY' has no literal here"
    );
    assert_eq!(
        view_words::default_refusal_text(&DefaultRefusal::TooLong, "DOUBLE", &document, value)
            .unwrap(),
        "its default is too long to write as a literal of type DOUBLE"
    );
}

#[test]
fn parameter_defaults_and_missing_information_are_reported() {
    let spec = metric::parse("source: c.s.t\nparameters:\n  - name: Limit\n    default: 12\n  - name: MissingDefault\n  - name: MissingType\n    default: 1\n", &yaml_rules()).unwrap();
    let read = view_steps::parameters(
        &spec,
        &[
            ("Limit".into(), "INT".into()),
            ("MissingDefault".into(), "INT".into()),
        ],
        &ask,
    )
    .unwrap();
    let texts: Vec<_> = read
        .iter()
        .map(|(name, parameter)| {
            let default = spec
                .parameters
                .iter()
                .find(|p| p.name.to_lowercase() == *name)
                .and_then(|p| p.default);
            (
                name.as_str(),
                view_words::parameter_text(parameter, &spec.document, default).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        texts,
        [
            ("limit", "12".into()),
            ("missingdefault", "has no default".into()),
            ("missingtype", "has no type".into())
        ]
    );
    assert_eq!(
        view_words::parameter_note_text("Limit", "12"),
        "Read the parameter 'Limit' as its default '12', as a query of the view without arguments does"
    );
}

#[test]
fn filter_and_measure_messages_identify_the_rejected_reference() {
    assert_eq!(
        view_words::filter_reason_text(&FilterRefusal::NotACondition, None),
        "it is not a SQL condition"
    );
    assert_eq!(
        view_words::filter_reason_text(&FilterRefusal::JoinedTable("users".into()), None),
        "it references the joined table 'users'"
    );
    let refusal = data::MeasureRefusal {
        name: "revenue".into(),
        reason: data::MeasureReason::NamesRefused("amount".into()),
    };
    assert_eq!(
        view_words::measure_refusal_text(&refusal),
        "Failed to add measure 'revenue' since it names the dimension 'amount', which is refused"
    );
}

#[test]
fn graph_notes_explain_joins_aggregations_and_time_selection() {
    assert_eq!(
        view_words::skipped_join_text("users", "missing key", &[], &[]),
        "Failed to add join 'users' since missing key"
    );
    assert_eq!(
        view_words::measure_note_text(&MeasureNote {
            measure: "rows".into(),
            table: "orders".into(),
            function: "COUNT".into(),
            column: None
        }),
        "'rows' is COUNT(orders.*, 0, 30, days) over the next 30 days"
    );
    assert_eq!(
        view_words::measure_note_text(&MeasureNote {
            measure: "revenue".into(),
            table: "orders".into(),
            function: "SUM".into(),
            column: Some("amount".into())
        }),
        "'revenue' is SUM(orders.amount, 0, 30, days) over the next 30 days"
    );
    let mut note = TimeColumnNote {
        table: "orders".into(),
        column: "event_time".into(),
        instead_of: vec!["day".into()],
        source_column: None,
        added: true,
        window_order: false,
    };
    assert_eq!(
        view_words::time_note_text(&note),
        "Took 'event_time' as the time column of table 'orders', and added it as a column, since 'day' truncates or casts it, and it says when a row happened more precisely"
    );
    note.window_order = true;
    note.instead_of.push("month".into());
    assert_eq!(
        view_words::time_note_reasons(&note),
        "a window measure of the view orders by it and 'day' and 'month' truncate or cast it, and it says when a row happened more precisely"
    );
}

/// `_edge_error`'s and `_join_key_reason`'s texts by name, the vectors
/// holding few.
#[test]
fn each_edge_reason_and_key_reason_is_worded_by_name() {
    let edge = Edge {
        source: "orders".into(),
        key: "placed".into(),
        target: "users".into(),
    };
    let text = |reason| view_words::edge_reason_text(&edge, &reason);
    assert_eq!(
        text(model::EdgeReason::NoPrimaryKey),
        "Edge Edge(src_table='orders', fkey='placed', dst_table='users') is invalid since table \
         'users' does not have a primary key. Add either a primary key or remove the link before \
         proceeding."
    );
    assert_eq!(
        text(model::EdgeReason::NotNumberOrString {
            dtype: sdm_contract::ops::Dtype::Date
        }),
        "Edge(src_table='orders', fkey='placed', dst_table='users') is invalid as foreign key must \
         be a number or string (got 'date')"
    );
    assert_eq!(
        view_words::key_reason_text(
            &model::KeyReason::NotUnique {
                repeats: 1200,
                rows: 34567
            },
            "users",
            "id",
            Some("'a'"),
        ),
        "its key 'id' is not unique in table 'users': 1,200 of 34,567 rows repeat a value, e.g. \
         'a'. A graph table needs one row per key, so point the join at a table or view with one \
         row per 'id'"
    );
}

/// Long function names must survive error formatting without truncation or panic.
#[test]
fn long_untrusted_function_names_are_preserved_in_the_reason() {
    let name = "custom_".to_owned() + &"x".repeat(1024);
    let reason = view_expr::untrusted_call_reason(
        &format!("{name}(x)"),
        ExprDialect::Databricks,
        &["x"],
        spaces(),
    )
    .unwrap();
    assert_eq!(
        view_words::expr_reason(&reason, dialect_words(&reason)),
        format!("it calls '{name}', which is not a pure built-in SQL function")
    );
}

#[test]
fn yaml_default_types_do_not_coerce_booleans_into_numbers() {
    for (kind, input, expected) in [
        ("BOOLEAN", "true", Ok("true".to_owned())),
        ("BOOLEAN", "' FALSE '", Ok("false".to_owned())),
        ("BOOLEAN", "1", Err(DefaultRefusal::NotOfType)),
        ("INT", "true", Err(DefaultRefusal::NotOfType)),
        ("DOUBLE", "true", Err(DefaultRefusal::NotOfType)),
        ("STRING", "true", Ok("true".to_owned())),
        ("STRING", "2024-02-29", Ok("2024-02-29".to_owned())),
        ("DATE", "2024-02-29", Ok("2024-02-29".to_owned())),
        ("ARRAY<INT>", "[1, 2]", Err(DefaultRefusal::NoLiteral)),
    ] {
        let doc = yaml::load(&format!("v: {input}"), &yaml_rules()).unwrap();
        assert_eq!(
            view_steps::default_text(
                view_steps::type_class(&ask, kind).unwrap(),
                &doc,
                doc.get(doc.root(), "v").unwrap()
            ),
            expected,
            "{kind}: {input}"
        );
    }
}

#[test]
fn described_parameter_type_overrides_yaml_and_last_duplicate_wins_in_place() {
    let spec = metric::parse("source: t\nparameters:\n  - name: Limit\n    data_type: STRING\n    default: '12'\n  - name: other\n    data_type: INT\n    default: 3\n  - name: LIMIT\n    data_type: STRING\n    default: '24'\n", &yaml_rules()).unwrap();
    let read = view_steps::parameters(&spec, &[("limit".into(), "INT".into())], &ask).unwrap();
    assert_eq!(
        read.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        ["limit", "other"]
    );
    assert_eq!(read[0].1.name, "LIMIT");
    assert_eq!(
        read[0].1.read.as_ref().unwrap().literal,
        "CAST('24' AS INT)"
    );
    assert_eq!(read[1].1.read.as_ref().unwrap().literal, "CAST('3' AS INT)");
    assert_eq!(
        view_words::skipped_join_text(
            "users",
            "missing key",
            &["addresses".into()],
            &["city".into(), "country".into()]
        ),
        "Failed to add join 'users', and with it the join 'addresses' nested in it and the dimensions 'city' and 'country', since missing key"
    );
}
