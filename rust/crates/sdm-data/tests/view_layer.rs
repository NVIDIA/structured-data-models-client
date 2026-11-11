// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Semantic-view questions, failures and Arrow-backed definitions through the data port.
//! A layer without whitespace rules refuses view interpretation explicitly.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::ErrorKind;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, FailureFacts, FetchPlan, ProviderChoice,
    ProviderError, Relation, RelationColumn, Unusable, ViewChoice,
};
use sdm_contract::ops::Dtype;
use sdm_contract::view::{
    DeclaredType, ExprRefusal, KeyRef, ViewKind, ViewTextAnswer, ViewTextAsk,
};
use sdm_contract::{Detail, Sentence};
use sdm_data::Data;
use sdm_data::view_expr::Spaces;

/// Python's white space (`str.isspace`), which the core hands in from
/// `sdm-pql`: the 29 characters Python 3.12 counts.
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

fn databricks() -> ProviderChoice {
    ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new("databricks"),
        namespace: vec![Some("cat".into()), Some("sch".into())],
        view: Some(ViewChoice {
            name: "sales_mv".into(),
            trust_expressions: false,
        }),
    }
}

fn ask(question: ViewTextAsk) -> ViewTextAnswer {
    Data::reading_views(spaces())
        .view_text(&databricks(), question)
        .unwrap()
}

/// Each question is the helper it is named after, in Databricks's dialect.
#[test]
fn a_question_of_a_view_s_text_is_answered_in_the_dialect() {
    assert_eq!(
        ask(ViewTextAsk::TableReference {
            name: "cat.sch.`order lines`".into()
        }),
        ViewTextAnswer::Names(Some(vec!["cat".into(), "sch".into(), "order lines".into()]))
    );
    assert_eq!(
        ask(ViewTextAsk::TableReference {
            name: "SELECT * FROM t".into()
        }),
        ViewTextAnswer::Names(None)
    );
    assert_eq!(
        ask(ViewTextAsk::JoinCondition {
            condition: "source.user_id = users.id".into()
        }),
        ViewTextAnswer::Condition(Some([
            KeyRef {
                qualifier: Some("source".into()),
                column: "user_id".into()
            },
            KeyRef {
                qualifier: Some("users".into()),
                column: "id".into()
            },
        ]))
    );
    let ViewTextAnswer::Expression(Some(verdict)) = ask(ViewTextAsk::Expression {
        expr: "secret('s', 'k')".into(),
        trust_expressions: false,
        columns: vec!["x".into()],
    }) else {
        panic!("an untrusted call is refused")
    };
    assert!(verdict.untrusted);
    assert_eq!(verdict.reason, ExprRefusal::NotPure("secret".into()));
    assert_eq!(
        ask(ViewTextAsk::Expression {
            expr: "secret('s', 'k')".into(),
            trust_expressions: true,
            columns: vec![],
        }),
        ViewTextAnswer::Expression(None)
    );
    // Unsafe text whatever the trust.
    let ViewTextAnswer::Expression(Some(verdict)) = ask(ViewTextAsk::Expression {
        expr: "trim(FROM x)".into(),
        trust_expressions: true,
        columns: vec![],
    }) else {
        panic!("a sub-query is refused")
    };
    assert!(!verdict.untrusted);
    assert_eq!(
        ask(ViewTextAsk::ReasonWords {
            reason: ExprRefusal::DollarQuotes
        }),
        ViewTextAnswer::Words(Some(
            "it contains '$$', which Databricks SQL does not quote with".into()
        ))
    );
    assert_eq!(
        ask(ViewTextAsk::ReasonWords {
            reason: ExprRefusal::SubQuery
        }),
        ViewTextAnswer::Words(None)
    );
    assert_eq!(
        ask(ViewTextAsk::DeclaredType {
            data_type: "decimal(10,0)".into()
        }),
        ViewTextAnswer::DeclaredType(DeclaredType::Known(Dtype::Int))
    );
    assert_eq!(
        ask(ViewTextAsk::DeclaredType {
            data_type: "map<string,int>".into()
        }),
        ViewTextAnswer::DeclaredType(DeclaredType::Unsupported)
    );
    let columns = vec![RelationColumn {
        name: "total".into(),
        column: None,
        expression: Some("price * quantity".into()),
    }];
    assert_eq!(
        ask(ViewTextAsk::Derive {
            table: Relation::of(vec!["cat".into(), "sch".into(), "orders".into()]),
            columns: columns.clone(),
            filter: Some("source.status = 'VALID'".into()),
        }),
        ViewTextAnswer::Relation(Relation {
            source: vec!["cat".into(), "sch".into(), "orders".into()],
            columns,
            filter: Some("source.status = 'VALID'".into()),
            found: false,
        })
    );
}

/// A layer given no white space, or a dialect that reads no semantic
/// model, refuses one by name.
#[test]
fn a_layer_that_reads_no_view_refuses_one_by_name() {
    let refused = Data::new()
        .view_text(
            &databricks(),
            ViewTextAsk::StripComments { expr: "x".into() },
        )
        .unwrap_err();
    assert_eq!(refused.sentence, Sentence::ViewsUnread);
    let refused = Data::new().capabilities(&databricks()).unwrap_err();
    assert_eq!(refused.sentence, Sentence::ViewsUnread);
    assert!(
        Data::reading_views(spaces())
            .capabilities(&databricks())
            .is_ok()
    );
    let duckdb = ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new("duckdb"),
        namespace: vec![],
        view: None,
    };
    let refused = Data::reading_views(spaces())
        .view_text(&duckdb, ViewTextAsk::StripComments { expr: "x".into() })
        .unwrap_err();
    assert_eq!(refused.sentence, Sentence::ViewsUnread);
}

fn failed(sqlstate: Option<&str>, class: Option<&str>) -> ProviderError {
    ProviderError {
        driver_message: None,
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: sqlstate.map(str::to_owned),
        vendor_code: None,
        error_class: class.map(str::to_owned),
    }
}

/// What a Databricks failure says (`DB/errors.py:66-99`), each fact on its
/// own; another dialect's failures say nothing.
#[test]
fn a_databricks_failure_says_its_facts() {
    let facts = |state, class| Data::new().failure(&databricks(), &failed(state, class));
    assert_eq!(
        facts(Some("42P01"), None),
        FailureFacts {
            not_found: false,
            refuses_expression: true,
            unusable: Some(Unusable::Missing),
        }
    );
    assert_eq!(
        facts(Some("42501"), None).unusable,
        Some(Unusable::Forbidden)
    );
    assert_eq!(
        facts(None, Some("INSUFFICIENT_PERMISSIONS.NO_SELECT")).unusable,
        Some(Unusable::Forbidden)
    );
    assert_eq!(
        facts(None, Some("PERMISSION_DENIED")).unusable,
        Some(Unusable::Forbidden)
    );
    assert!(facts(Some("22012"), None).refuses_expression);
    assert!(facts(Some("42703"), Some("UNRESOLVED_COLUMN.WITH_SUGGESTION")).refuses_expression);
    assert_eq!(facts(Some("08001"), None), FailureFacts::default());
    assert_eq!(facts(None, None), FailureFacts::default());
    let duckdb = ProviderChoice::sql("duckdb");
    assert_eq!(
        Data::new().failure(&duckdb, &failed(Some("42P01"), None)),
        FailureFacts::default()
    );
}

fn definition_request(parts: &[&str]) -> DataRequest {
    DataRequest::Describe {
        tables: vec![Relation::of(
            parts.iter().map(|p| (*p).to_owned()).collect(),
        )],
        part: DescribePart::ViewDefinition,
    }
}

/// The metric view's description is asked as the old client asks it, each
/// part of its name in backticks (`DB/metric_view.py:1440-1450`).
#[test]
fn a_metric_view_s_definition_is_asked_by_its_quoted_name() {
    let FetchPlan::Sql { statements, .. } = Data::reading_views(spaces())
        .plan(
            &definition_request(&["cat", "s`ch", "sales_mv"]),
            &databricks(),
        )
        .unwrap()
    else {
        panic!("a SQL plan")
    };
    let texts: Vec<&str> = statements.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["DESCRIBE TABLE EXTENDED `cat`.`s``ch`.`sales_mv`"]);
}

/// The semantic lookup argument is bound text, not a fully qualified table
/// rendered in a FROM clause. Snowflake resolves missing namespace parts.
#[test]
fn semantic_definition_names_are_bound_without_physical_table_qualification() {
    use sdm_contract::Scalar;
    use sdm_contract::data::SqlParam;

    let provider = ProviderChoice::sql("snowflake");
    for (parts, expected) in [
        (vec!["V"], "V"),
        (vec!["DB", "V"], "DB.V"),
        (vec!["SHOP", "V"], "SHOP.V"),
        (vec!["DB", "SHOP", "V"], "DB.SHOP.V"),
        (vec!["DB", "SHOP", "SHOP.V"], "SHOP.V"),
        (
            vec!["DB", "SHOP", "\"Sales View\""],
            "DB.SHOP.\"Sales View\"",
        ),
        (vec!["DB", "SHOP", "\"Sales.View\""], "\"Sales.View\""),
    ] {
        let FetchPlan::Sql { statements, .. } = Data::reading_views(spaces())
            .plan(&definition_request(&parts), &provider)
            .unwrap()
        else {
            panic!("a SQL plan")
        };
        assert_eq!(statements.len(), 1);
        assert_eq!(
            statements[0].text,
            "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)"
        );
        assert_eq!(
            statements[0].params,
            vec![SqlParam::Value {
                value: Scalar::Utf8(expected.into()),
            }]
        );
    }
    for name in ["V", "SHOP.V", "DB.SHOP.\"Sales View\"", "\"Sales.View\""] {
        assert_eq!(
            Data::reading_views(spaces())
                .view_text(&provider, ViewTextAsk::ViewName { name: name.into() })
                .unwrap(),
            ViewTextAnswer::Names(Some(vec![name.into()]))
        );
    }
    for (namespace, expected) in [
        (vec![None, None], "V"),
        (vec![Some("DB".into()), None], "DB.V"),
        (vec![None, Some("SHOP".into())], "SHOP.V"),
        (vec![Some("DB".into()), Some("SHOP".into())], "DB.SHOP.V"),
    ] {
        let source = ProviderChoice::Sql {
            timestamp_encoding: None,
            dialect: Dialect::new("snowflake"),
            namespace,
            view: None,
        };
        let FetchPlan::Sql { statements, .. } = Data::reading_views(spaces())
            .plan(&definition_request(&["V"]), &source)
            .unwrap()
        else {
            panic!("a SQL plan")
        };
        assert_eq!(
            statements[0].params,
            vec![SqlParam::Value {
                value: Scalar::Utf8(expected.into()),
            }]
        );
    }
    // Only definition lookups bypass physical qualification.
    assert!(
        Data::new()
            .plan(
                &DataRequest::Preview {
                    table: Relation::of(vec!["V".into()]),
                    columns: vec!["id".into()],
                    limit: 1,
                },
                &provider
            )
            .is_err()
    );
}

/// Rows as the describe gives them, each cell text or null, as many
/// columns as the row has.
fn batch(rows: &[Vec<Option<&str>>]) -> RecordBatch {
    let width = rows.iter().map(Vec::len).max().unwrap_or(2);
    let fields: Vec<Field> = (0..width)
        .map(|at| Field::new(format!("c{at}"), DataType::Utf8, true))
        .collect();
    let columns: Vec<ArrayRef> = (0..width)
        .map(|at| {
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.get(at).copied().flatten().map(str::to_owned))
                    .collect::<Vec<_>>(),
            )) as ArrayRef
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

/// Arrow batches from both warehouse and Spark preserve the same definition facts.
#[test]
fn metric_definition_is_read_through_the_data_layer_port() {
    let layer = Data::reading_views(spaces());
    for width in [3, 4] {
        let mut rows = vec![
            vec![Some("id"), Some("bigint")],
            vec![Some("total"), Some("double measure")],
            vec![None, None],
            vec![Some("# Parameters"), None],
            vec![Some("start"), Some("date")],
            vec![Some("# Detailed Table Information"), None],
            vec![Some("Type"), Some("METRIC_VIEW")],
            vec![Some("View Text"), Some("source: orders")],
            vec![Some("Catalog"), Some("sales")],
            vec![Some("Database"), Some("public")],
        ];
        for row in &mut rows {
            row.resize(width, None);
        }
        let answer = layer
            .read(
                &definition_request(&["cat", "sch", "sales_mv"]),
                &databricks(),
                &[batch(&rows)],
            )
            .unwrap();
        let CatalogAnswer::ViewDefinition(found) = answer else {
            panic!("expected a view definition")
        };
        assert_eq!(found.kind, ViewKind::Metric);
        assert_eq!(found.text, "source: orders");
        assert_eq!(found.columns, [("id".into(), "bigint".into())]);
        assert_eq!(found.parameters, [("start".into(), "date".into())]);
        assert_eq!(found.located, [Some("sales".into()), Some("public".into())]);
    }
}

#[test]
fn invalid_definitions_report_the_specific_reason_and_qualified_view_name() {
    for (kind, text, expected) in [
        ("VIEW", "source: orders", Sentence::ViewNotAMetricView),
        ("METRIC_VIEW", "", Sentence::ViewNoDefinition),
    ] {
        let rows = vec![
            vec![Some("# Detailed Table Information"), None],
            vec![Some("Type"), Some(kind)],
            vec![Some("View Text"), Some(text)],
        ];
        let error = Data::reading_views(spaces())
            .read(
                &definition_request(&["cat", "sch", "sales_mv"]),
                &databricks(),
                &[batch(&rows)],
            )
            .unwrap_err();
        assert_eq!(error.sentence, expected);
        assert_eq!(error.fields["view"], Detail::text("`cat`.`sch`.`sales_mv`"));
    }
}

#[test]
fn a_views_kind_is_known_before_its_definition_is_fetched() {
    let data = Data::reading_views(spaces());
    assert_eq!(
        data.view_kind(&ProviderChoice::sql("databricks")),
        Some(ViewKind::Metric)
    );
    assert_eq!(data.view_kind(&ProviderChoice::sql("duckdb")), None);
    assert_eq!(
        data.view_kind(&ProviderChoice::sql("snowflake")),
        Some(ViewKind::Semantic)
    );
}
