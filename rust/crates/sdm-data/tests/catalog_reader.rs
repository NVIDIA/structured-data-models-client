// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data layer's reader of catalog answers (decision E54): DuckDB's own
//! catalog answers and a host provider's answers in the contract's shape,
//! for the same catalog, read into the same dialect-neutral answer; the
//! reading of DuckDB's constraint types and index expressions, which only
//! `sdm-data` holds; and a reader that refuses, rather than panics, on an
//! answer not in its checked shape.

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, BooleanArray, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::catalog::{
    CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogIndex, CatalogKey, KeyKind,
};
use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Determinism, Dialect, ExpectedSchema, FetchPlan,
    ProviderCapabilities, ProviderChoice,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind};
use sdm_data::Data;

fn duckdb() -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
    }
}

fn host() -> ProviderChoice {
    ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            optional: Default::default(),
            max_keys_per_request: None,
            max_seed: None,
            text_times: None,
            sampling: Determinism::SeededStable,
        },
    }
}

fn describe(part: DescribePart) -> DataRequest {
    DataRequest::Describe {
        tables: match part {
            DescribePart::Tables | DescribePart::Indexes => vec![],
            _ => vec!["orders".into()],
        },
        part,
    }
}

fn strings(values: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}

fn lists(item: &str, rows: &[&[&str]]) -> ArrayRef {
    let mut builder =
        ListBuilder::new(StringBuilder::new()).with_field(Field::new(item, DataType::Utf8, true));
    for row in rows {
        for value in *row {
            builder.values().append_value(value);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

/// One batch of these columns, each field typed as its array and named as
/// given, in the schema the plan for `request` on `provider` expects.
fn answer(
    request: &DataRequest,
    provider: &ProviderChoice,
    columns: Vec<ArrayRef>,
) -> Vec<RecordBatch> {
    let ExpectedSchema::Schema { fields } = Data::new()
        .plan(request, provider)
        .unwrap()
        .expect()
        .clone()
    else {
        panic!("a catalog answer has a fixed shape");
    };
    assert_eq!(fields.len(), columns.len());
    let schema = Schema::new(
        fields
            .iter()
            .zip(&columns)
            .map(|(f, a)| Field::new(&f.name, a.data_type().clone(), f.nullable))
            .collect::<Vec<_>>(),
    );
    vec![RecordBatch::try_new(Arc::new(schema), columns).unwrap()]
}

fn read(
    request: &DataRequest,
    provider: &ProviderChoice,
    batches: &[RecordBatch],
) -> CatalogAnswer {
    Data::new().read(request, provider, batches).unwrap()
}

/// The same catalog, as DuckDB answers it and as a host provider answers
/// it in the contract's shape: each part reads to the same answer.
#[test]
fn duckdb_s_catalog_and_a_host_s_read_to_the_same_answer() {
    let tables = describe(DescribePart::Tables);
    let expected = CatalogAnswer::Tables(vec!["orders".into(), "users".into()]);
    let names = strings(&[Some("orders"), Some("users")]);
    assert_eq!(
        read(
            &tables,
            &duckdb(),
            &answer(&tables, &duckdb(), vec![names.clone()])
        ),
        expected
    );
    assert_eq!(
        read(&tables, &host(), &answer(&tables, &host(), vec![names])),
        expected
    );

    let columns = describe(DescribePart::Columns);
    let expected = CatalogAnswer::Columns(vec![
        CatalogColumn {
            name: "order_id".into(),
            declared_type: "BIGINT".into(),
            not_null: true,
            primary_key: true,
        },
        CatalogColumn {
            name: "ts".into(),
            declared_type: "TIMESTAMP".into(),
            not_null: false,
            primary_key: false,
        },
    ]);
    let name = strings(&[Some("order_id"), Some("ts")]);
    let declared = strings(&[Some("BIGINT"), Some("TIMESTAMP")]);
    let not_null: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
    let key: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
    let pragma = vec![
        Arc::new(Int32Array::from(vec![0, 1])) as ArrayRef,
        name.clone(),
        declared.clone(),
        not_null.clone(),
        strings(&[None, Some("now()")]),
        key.clone(),
    ];
    assert_eq!(
        read(&columns, &duckdb(), &answer(&columns, &duckdb(), pragma)),
        expected
    );
    let neutral = vec![name, declared, not_null, key];
    assert_eq!(
        read(&columns, &host(), &answer(&columns, &host(), neutral)),
        expected
    );

    let keys = describe(DescribePart::Keys);
    let expected = CatalogAnswer::Keys(vec![
        CatalogKey {
            kind: KeyKind::Primary,
            columns: vec!["order_id".into()],
        },
        CatalogKey {
            kind: KeyKind::Unique,
            columns: vec!["a".into(), "b".into()],
        },
        CatalogKey {
            kind: KeyKind::Other,
            columns: vec!["user_id".into()],
        },
    ]);
    let members: &[&[&str]] = &[&["order_id"], &["a", "b"], &["user_id"]];
    let constraints = vec![
        strings(&[Some("PRIMARY KEY"), Some("UNIQUE"), Some("FOREIGN KEY")]),
        lists("l", members),
    ];
    assert_eq!(
        read(&keys, &duckdb(), &answer(&keys, &duckdb(), constraints)),
        expected
    );
    let neutral = vec![
        strings(&[Some("primary"), Some("unique"), Some("check")]),
        lists("item", members),
    ];
    assert_eq!(
        read(&keys, &host(), &answer(&keys, &host(), neutral)),
        expected
    );

    let foreign = describe(DescribePart::ForeignKeys);
    let expected = CatalogAnswer::ForeignKeys(vec![CatalogForeignKey {
        key: 3,
        referenced_table: "users".into(),
        columns: vec!["user_id".into()],
        referenced_columns: vec!["user_id".into()],
    }]);
    let pair = |item: &str| {
        vec![
            Arc::new(Int64Array::from(vec![3])) as ArrayRef,
            strings(&[Some("users")]),
            lists(item, &[&["user_id"]]),
            lists(item, &[&["user_id"]]),
        ]
    };
    assert_eq!(
        read(&foreign, &duckdb(), &answer(&foreign, &duckdb(), pair("l"))),
        expected
    );
    assert_eq!(
        read(&foreign, &host(), &answer(&foreign, &host(), pair("item"))),
        expected
    );

    let indexes = describe(DescribePart::Indexes);
    let expected = CatalogAnswer::Indexes(vec![
        CatalogIndex {
            table: "orders".into(),
            columns: vec!["user_id".into()],
        },
        CatalogIndex {
            table: "orders".into(),
            columns: vec!["a".into(), "b".into()],
        },
    ]);
    let expressions = vec![
        strings(&[Some("orders"), Some("orders"), Some("users")]),
        strings(&[Some("[user_id]"), Some("[a, b]"), None]),
    ];
    assert_eq!(
        read(
            &indexes,
            &duckdb(),
            &answer(&indexes, &duckdb(), expressions)
        ),
        expected
    );
    let neutral = vec![
        strings(&[Some("orders"), Some("orders")]),
        lists("item", &[&["user_id"], &["a", "b"]]),
    ];
    assert_eq!(
        read(&indexes, &host(), &answer(&indexes, &host(), neutral)),
        expected
    );
}

/// DuckDB's index expressions, read as the reference's `_indexed_column`
/// reads them: one column, its quotes taken off, only from a bracketed
/// list without a comma.
#[test]
fn duckdb_s_index_expressions_read_as_the_reference_reads_them() {
    let indexes = describe(DescribePart::Indexes);
    let read_one = |expressions: &str| -> Vec<String> {
        let batches = answer(
            &indexes,
            &duckdb(),
            vec![strings(&[Some("t")]), strings(&[Some(expressions)])],
        );
        match read(&indexes, &duckdb(), &batches) {
            CatalogAnswer::Indexes(found) => found.into_iter().next().unwrap().columns,
            other => panic!("{other:?}"),
        }
    };
    assert_eq!(read_one("[user_id]"), ["user_id"]);
    assert_eq!(read_one(" ['\"a\"\"b\"'] "), ["a\"b"]);
    assert_eq!(read_one("['user id']"), ["user id"]);
    // The reference's slice of a lone double quote is the empty name.
    assert_eq!(read_one("['\"']"), [""]);
    // A comma: never one column, as the reference reads none.
    assert_eq!(read_one("[a, b]"), ["a", "b"]);
    assert_eq!(read_one("['\"a,b\"']").len(), 2);
    // Not a bracketed list: no column.
    assert!(read_one("a").is_empty());
    assert!(read_one("[a").is_empty());
}

fn internal(err: &CoreError) -> (ErrorKind, ErrorCode) {
    (err.kind, err.code)
}

/// The reader refuses with `INTERNAL_ERROR`, never panics, on what the
/// core's admission would have refused, and reads only a catalog answer.
#[test]
fn an_answer_not_in_its_checked_shape_is_refused_not_a_panic() {
    let columns = describe(DescribePart::Columns);
    for provider in [duckdb(), host()] {
        // A column missing.
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("other", DataType::Utf8, true)])),
            vec![strings(&[Some("x")])],
        )
        .unwrap();
        let err = Data::new().read(&columns, &provider, &[batch]).unwrap_err();
        assert_eq!(
            internal(&err),
            (ErrorKind::Internal, ErrorCode::InternalError)
        );
        assert!(err.fields.contains_key("column"));
    }
    // A column of another type than its checked one, in each reader.
    let keys = describe(DescribePart::Keys);
    for (provider, names) in [
        (duckdb(), ["constraint_type", "constraint_column_names"]),
        (host(), ["kind", "columns"]),
    ] {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(names[0], DataType::Utf8, true),
                Field::new(names[1], DataType::Utf8, true),
            ])),
            vec![strings(&[Some("UNIQUE")]), strings(&[Some("a")])],
        )
        .unwrap();
        let err = Data::new().read(&keys, &provider, &[batch]).unwrap_err();
        assert_eq!(err.code, ErrorCode::InternalError);
        assert_eq!(err.fields["column"], Detail::text(names[1]));
    }
    let err = Data::new()
        .read(
            &DataRequest::Preview {
                table: "t".into(),
                columns: vec!["c".into()],
                limit: 1,
            },
            &duckdb(),
            &[],
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InternalError);
    assert_eq!(err.fields["request"], Detail::text("preview"));
    // A dialect this build renders nothing for reads nothing either.
    let err = Data::new()
        .read(
            &columns,
            &ProviderChoice::Sql {
                dialect: Dialect::new("sqlite"),
            },
            &[],
        )
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnsupportedRequest);
    assert_eq!(err.fields["dialect"], Detail::text("sqlite"));
}

/// The plan's expected shape is what the reader reads: an answer in the
/// contract's host shape is not DuckDB's, so a DuckDB plan would refuse it
/// at admission (check (b)), and the reverse.
#[test]
fn a_duckdb_plan_and_a_host_plan_expect_different_catalog_shapes() {
    for part in [
        DescribePart::Tables,
        DescribePart::Columns,
        DescribePart::Keys,
        DescribePart::ForeignKeys,
        DescribePart::Indexes,
    ] {
        let request = describe(part);
        let duckdb = Data::new().plan(&request, &duckdb()).unwrap();
        let host = Data::new().plan(&request, &host()).unwrap();
        assert!(matches!(duckdb, FetchPlan::Sql { .. }));
        assert_eq!(
            host,
            FetchPlan::Host {
                expect: request.expected()
            }
        );
        assert_ne!(duckdb.expect(), host.expect(), "{part:?}");
    }
}
