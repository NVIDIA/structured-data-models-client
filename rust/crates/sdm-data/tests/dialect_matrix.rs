// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQL request coverage across the five supported dialects.
//! Small independent assertions protect admission, schemas, seed ownership,
//! identifier handling, aggregates, windows and ordering without SQL recordings.

use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, Filter, KeySet, ProviderChoice, Relation,
    SEED_COUNT_UNREAD, SEED_KEY_TYPE, SEED_KEY_UNIQUE, SEED_STORED_TIME, SeedBound, TableColumn,
    TableKey, TieBreak, TieTerm, Window, field,
};
use sdm_contract::ops::Dtype;
use sdm_contract::{ColumnType, Field, Scalar, Unit};
use sdm_data::Data;
use serde_json::{Value, json};

const RENDERED: [&str; 5] = ["duckdb", "sqlite", "postgres", "databricks", "snowflake"];

fn relation(parts: &[&str]) -> Relation {
    Relation {
        columns: Vec::new(),
        filter: None,
        found: false,
        source: parts.iter().map(|p| (*p).to_owned()).collect(),
    }
}

fn names(names: &[&str]) -> Vec<String> {
    names.iter().map(|n| (*n).to_owned()).collect()
}

fn not_null(column: &str) -> Filter {
    Filter::NotNull {
        column: column.to_owned(),
    }
}

fn with(mut field: Field, key: &str, value: &str) -> Field {
    field.metadata.insert(key.to_owned(), value.to_owned());
    field
}

fn timestamp() -> ColumnType {
    ColumnType::Timestamp {
        unit: Unit::Ns,
        zone: None,
    }
}

fn decimal_key() -> ColumnType {
    ColumnType::Decimal128 {
        precision: 38,
        scale: 0,
    }
}

/// A key set of `keys` (its key columns, as `seed_key_columns` names them,
/// and their fields), the batch, and `extra` seed columns after them.
fn seeds(keys: Vec<Field>, extra: Vec<Field>) -> KeySet {
    let mut columns = keys;
    columns.push(field("batch", ColumnType::Int64, false));
    columns.extend(extra);
    KeySet {
        attachment: sdm_contract::data::AttachmentId(1),
        columns,
    }
}

fn one_key(column_type: ColumnType) -> Vec<Field> {
    vec![field("key", column_type, true)]
}

fn two_keys() -> Vec<Field> {
    vec![
        field("key_1", ColumnType::Int64, true),
        field("key_2", ColumnType::Utf8, true),
    ]
}

fn term(column: &str, as_timestamp: bool, descending: bool, nulls_last: bool) -> TieTerm {
    TieTerm {
        string_like: false,
        column: column.to_owned(),
        as_timestamp,
        descending,
        nulls_last,
    }
}

fn bound(seed_column: &str, inclusive: bool) -> Option<SeedBound> {
    Some(SeedBound {
        seed_column: seed_column.to_owned(),
        inclusive,
    })
}

/// The relations a request reads, each depth and awkward name: one part,
/// a dot or a quote inside one, two and three parts, none, four, and the
/// seeds' own names in two cases.
fn relations() -> Vec<Relation> {
    vec![
        relation(&["t"]),
        relation(&["a.b"]),
        relation(&["we\"ird"]),
        relation(&["it's"]),
        relation(&["back\\slash"]),
        relation(&["main", "t"]),
        relation(&["db", "s", "t"]),
        relation(&[]),
        relation(&["a", "b", "c", "d"]),
        relation(&["__sdm_seeds__"]),
        relation(&["__SDM_SEEDS__"]),
    ]
}

fn describes() -> Vec<DataRequest> {
    let parts = [
        DescribePart::Tables,
        DescribePart::Columns,
        DescribePart::Keys,
        DescribePart::ForeignKeys,
        DescribePart::Indexes,
        DescribePart::IndexColumns,
        DescribePart::Namespace,
    ];
    let mut tables: Vec<Vec<Relation>> = vec![vec![]];
    tables.extend(relations().into_iter().map(|r| vec![r]));
    tables.push(vec![relation(&["t"]), relation(&["u"])]);
    tables.push(vec![relation(&["t"]), relation(&["main", "u"])]);
    let mut found = Vec::new();
    for part in parts {
        for tables in &tables {
            found.push(DataRequest::Describe {
                tables: tables.clone(),
                part,
            });
        }
    }
    found
}

fn reads() -> Vec<DataRequest> {
    let mut found = Vec::new();
    for table in relations() {
        found.push(DataRequest::Preview {
            table: table.clone(),
            columns: names(&["id", "we\"ird"]),
            limit: 1000,
        });
        found.push(DataRequest::Scan {
            table: table.clone(),
            columns: names(&["id"]),
            filters: vec![],
            limit: None,
        });
        found.push(DataRequest::CountRows {
            table: table.clone(),
            filters: vec![],
        });
        found.push(DataRequest::ColumnRanges {
            table: table.clone(),
            columns: names(&["at"]),
        });
        found.push(DataRequest::RepeatedKey {
            table: table.clone(),
            key: names(&["id"]),
        });
        found.push(DataRequest::KeyCounts {
            keys: vec![TableKey {
                unique: false,
                table: table.clone(),
                columns: names(&["id"]),
                label: None,
            }],
        });
        found.push(DataRequest::TimeBounds {
            columns: vec![TableColumn {
                table: table.clone(),
                column: "at".to_owned(),
            }],
        });
    }
    let t = relation(&["t"]);
    let time_range = Filter::TimeRange {
        column: "at".to_owned(),
        lower: None,
        upper: None,
    };
    for (columns, filters, limit) in [
        (names(&["id", "x"]), vec![not_null("id")], Some(5)),
        (
            names(&["id"]),
            vec![not_null("id"), not_null("we\"ird")],
            None,
        ),
        (names(&["id"]), vec![time_range.clone()], None),
        (names(&[]), vec![], None),
    ] {
        found.push(DataRequest::Scan {
            table: t.clone(),
            columns,
            filters,
            limit,
        });
    }
    found.push(DataRequest::Preview {
        table: t.clone(),
        columns: vec![],
        limit: 10,
    });
    for filters in [
        vec![not_null("id")],
        vec![not_null("id"), not_null("x")],
        vec![time_range],
    ] {
        found.push(DataRequest::CountRows {
            table: t.clone(),
            filters,
        });
    }
    for columns in [names(&["a", "b"]), names(&[])] {
        found.push(DataRequest::ColumnRanges {
            table: t.clone(),
            columns,
        });
    }
    for key in [names(&["a", "b"]), names(&[])] {
        found.push(DataRequest::RepeatedKey {
            table: t.clone(),
            key,
        });
    }
    found.push(DataRequest::KeyCounts {
        keys: vec![
            TableKey {
                unique: false,
                table: t.clone(),
                columns: names(&["id"]),
                label: None,
            },
            TableKey {
                unique: false,
                table: relation(&["u"]),
                columns: names(&["a", "b"]),
                label: Some(4),
            },
            TableKey {
                unique: false,
                table: relation(&["main", "v"]),
                columns: names(&["x", "y", "z"]),
                label: None,
            },
        ],
    });
    found.push(DataRequest::KeyCounts {
        keys: vec![
            TableKey {
                unique: false,
                table: t.clone(),
                columns: names(&["id"]),
                label: None,
            },
            TableKey {
                unique: false,
                table: relation(&[]),
                columns: names(&["id"]),
                label: None,
            },
        ],
    });
    found.push(DataRequest::TimeBounds {
        columns: vec![
            TableColumn {
                table: t.clone(),
                column: "at".to_owned(),
            },
            TableColumn {
                table: relation(&["main", "u"]),
                column: "when".to_owned(),
            },
        ],
    });
    found
}

fn samples() -> Vec<DataRequest> {
    let mut found = Vec::new();
    let sample = |table: Relation,
                  key: &[&str],
                  seed: Scalar,
                  columns: &[&str],
                  filters: Vec<Filter>,
                  key_dtype: Option<Dtype>| {
        DataRequest::SampleEntities {
            rows: None,
            table,
            key: names(key),
            n: 10000,
            seed,
            columns: names(columns),
            filters,
            key_dtype,
        }
    };
    for table in relations() {
        found.push(sample(
            table,
            &["id"],
            Scalar::UInt64(42),
            &["id", "x"],
            vec![],
            Some(Dtype::Int),
        ));
    }
    let t = || relation(&["t"]);
    for dtype in [
        None,
        Some(Dtype::Int),
        Some(Dtype::Int64),
        Some(Dtype::String),
        Some(Dtype::Float),
    ] {
        found.push(sample(
            t(),
            &["id"],
            Scalar::UInt64(42),
            &["id"],
            vec![not_null("id")],
            dtype,
        ));
    }
    found.push(sample(
        relation(&["main", "t"]),
        &["id"],
        Scalar::UInt64(7),
        &["id"],
        vec![],
        Some(Dtype::Int),
    ));
    for seed in [
        Scalar::UInt64(0),
        Scalar::Int64(-1),
        Scalar::Int32(5),
        Scalar::UInt64((1 << 31) - 1),
        Scalar::UInt64(1 << 31),
        Scalar::UInt64(u64::MAX),
        Scalar::Decimal128 {
            precision: 38,
            scale: 0,
            value: 123,
        },
        Scalar::Decimal128 {
            precision: 38,
            scale: 2,
            value: 123,
        },
        Scalar::Float64(1.5),
        Scalar::Utf8("42".to_owned()),
    ] {
        found.push(sample(t(), &["id"], seed, &["id"], vec![], None));
    }
    found.push(sample(
        t(),
        &["a", "b"],
        Scalar::UInt64(42),
        &["a", "b", "c"],
        vec![not_null("a"), not_null("b")],
        None,
    ));
    found.push(sample(t(), &["id"], Scalar::UInt64(42), &[], vec![], None));
    found.push(sample(
        t(),
        &["id"],
        Scalar::UInt64(42),
        &["x"],
        vec![],
        None,
    ));
    found.push(sample(t(), &[], Scalar::UInt64(42), &["x"], vec![], None));
    found.push(sample(
        t(),
        &["id"],
        Scalar::UInt64(42),
        &["id"],
        vec![Filter::TimeRange {
            column: "at".to_owned(),
            lower: None,
            upper: None,
        }],
        None,
    ));
    found.push(sample(
        t(),
        &["id"],
        Scalar::Float64(1.5),
        &["id"],
        vec![Filter::TimeRange {
            column: "at".to_owned(),
            lower: None,
            upper: None,
        }],
        None,
    ));
    found
}

/// The seed key fields each dialect binds differently: integers, a
/// `decimal128(38, 0)` with and without the key's declared type, an
/// unsigned 64-bit integer, a float and a text.
fn key_fields() -> Vec<Vec<Field>> {
    let declared = |name: &str| vec![with(field("key", decimal_key(), true), SEED_KEY_TYPE, name)];
    vec![
        one_key(ColumnType::Int64),
        one_key(ColumnType::Int32),
        one_key(ColumnType::Uint64),
        one_key(decimal_key()),
        declared("BIGINT"),
        declared(" hugeint "),
        declared("VARINT"),
        declared("BIGNUM"),
        declared("DECIMAL(20,0)"),
        declared("DECIMAL(10,2)"),
        declared("DOUBLE"),
        vec![with(
            field("key", ColumnType::Int64, true),
            SEED_KEY_TYPE,
            "BIGINT",
        )],
        one_key(ColumnType::Float64),
        one_key(ColumnType::Utf8),
    ]
}

fn neighbors() -> Vec<DataRequest> {
    let mut found = Vec::new();
    let read = |table: Relation,
                key: &[&str],
                columns: &[&str],
                seeds: KeySet,
                window: Option<Window>,
                per_seed: Option<u64>,
                order: Option<TieBreak>| DataRequest::Neighbors {
        table,
        key: names(key),
        columns: names(columns),
        seeds,
        window,
        per_seed: per_seed.map(Into::into),
        order,
    };
    let t = || relation(&["t"]);
    for table in relations() {
        found.push(read(
            table,
            &["fk"],
            &["fk", "x"],
            seeds(one_key(ColumnType::Int64), vec![]),
            None,
            None,
            None,
        ));
    }
    for keys in key_fields() {
        found.push(read(
            t(),
            &["fk"],
            &["fk"],
            seeds(keys, vec![]),
            None,
            None,
            None,
        ));
    }
    let times = |stored: Option<&str>| {
        let time = |name: &str| {
            let f = field(name, timestamp(), false);
            match stored {
                Some(s) => with(f, SEED_STORED_TIME, s),
                None => f,
            }
        };
        vec![time("end"), time("start")]
    };
    let windows = [
        Window {
            source_type: None,
            column: "at".to_owned(),
            lower: bound("start", false),
            upper: bound("end", true),
        },
        Window {
            source_type: None,
            column: "at".to_owned(),
            lower: bound("start", true),
            upper: bound("end", false),
        },
        Window {
            source_type: None,
            column: "at".to_owned(),
            lower: None,
            upper: bound("end", true),
        },
        Window {
            source_type: None,
            column: "at".to_owned(),
            lower: bound("start", false),
            upper: None,
        },
    ];
    let orders = [
        TieBreak(vec![
            term("at", true, true, true),
            term("fk", false, false, true),
        ]),
        TieBreak(vec![
            term("at", true, true, false),
            term("id", false, false, false),
        ]),
        TieBreak(vec![
            term("x", false, true, true),
            term("y", false, false, false),
        ]),
    ];
    for stored in [
        None,
        Some("2022-06-30 00:00:00"),
        Some("2022-06-30T00:00:00.000000Z"),
        Some("2022-06-30 00:00:00.123456789012+00:00"),
    ] {
        for window in &windows {
            found.push(read(
                t(),
                &["fk"],
                &["fk", "at"],
                seeds(one_key(ColumnType::Int64), times(stored)),
                Some(window.clone()),
                None,
                None,
            ));
        }
    }
    for order in &orders {
        for window in [None, Some(windows[0].clone())] {
            found.push(read(
                t(),
                &["fk"],
                &["fk", "at", "x"],
                seeds(
                    one_key(ColumnType::Int64),
                    times(Some("2022-06-30 00:00:00")),
                ),
                window,
                Some(3),
                Some(order.clone()),
            ));
        }
    }
    // A key of several columns, windowed and limited per seed.
    found.push(read(
        relation(&["main", "t"]),
        &["a", "b"],
        &["a", "b", "at"],
        seeds(two_keys(), times(None)),
        Some(windows[0].clone()),
        Some(2),
        Some(orders[0].clone()),
    ));
    found.push(read(
        t(),
        &["a", "b"],
        &["a", "b"],
        seeds(two_keys(), vec![]),
        None,
        None,
        None,
    ));
    // A limit without an order, and an order without a limit.
    found.push(read(
        t(),
        &["fk"],
        &["fk"],
        seeds(one_key(ColumnType::Int64), vec![]),
        None,
        Some(3),
        None,
    ));
    found.push(read(
        t(),
        &["fk"],
        &["fk"],
        seeds(one_key(ColumnType::Int64), vec![]),
        None,
        None,
        Some(orders[2].clone()),
    ));
    // No columns; seeds missing the key; a seed column past the key, the
    // batch and zone-less times.
    found.push(read(
        t(),
        &["fk"],
        &[],
        seeds(one_key(ColumnType::Int64), vec![]),
        None,
        None,
        None,
    ));
    found.push(read(
        t(),
        &["a", "b"],
        &["a"],
        seeds(one_key(ColumnType::Int64), vec![]),
        None,
        None,
        None,
    ));
    found.push(read(
        t(),
        &["fk"],
        &["fk"],
        seeds(
            one_key(ColumnType::Int64),
            vec![field("other", ColumnType::Utf8, true)],
        ),
        None,
        None,
        None,
    ));
    found.push(read(
        t(),
        &["fk"],
        &["fk"],
        seeds(
            one_key(ColumnType::Int64),
            vec![field("plain_time", timestamp(), true)],
        ),
        None,
        None,
        None,
    ));
    found
}

fn lookups() -> Vec<DataRequest> {
    let mut found = Vec::new();
    let lookup =
        |table: Relation, key: &[&str], columns: &[&str], seeds: KeySet| DataRequest::Lookup {
            table,
            key: names(key),
            columns: names(columns),
            seeds,
        };
    let t = || relation(&["t"]);
    for table in relations() {
        found.push(lookup(
            table,
            &["id"],
            &["id", "x"],
            seeds(one_key(ColumnType::Int64), vec![]),
        ));
    }
    for keys in key_fields() {
        found.push(lookup(t(), &["id"], &["id"], seeds(keys, vec![])));
    }
    let unique = |name: &str, column_type: ColumnType| {
        with(field(name, column_type, true), SEED_KEY_UNIQUE, "true")
    };
    let unread = |mut set: KeySet| {
        for f in &mut set.columns {
            if f.name == "batch" {
                f.metadata
                    .insert(SEED_COUNT_UNREAD.to_owned(), "true".to_owned());
            }
        }
        set
    };
    found.push(lookup(
        t(),
        &["id"],
        &["id", "x"],
        seeds(vec![unique("key", ColumnType::Int64)], vec![]),
    ));
    found.push(lookup(
        t(),
        &["id"],
        &["id", "x"],
        unread(seeds(one_key(ColumnType::Int64), vec![])),
    ));
    found.push(lookup(
        t(),
        &["id"],
        &["id"],
        unread(seeds(vec![unique("key", decimal_key())], vec![])),
    ));
    found.push(lookup(
        relation(&["main", "t"]),
        &["a", "b"],
        &["a", "b", "c"],
        seeds(two_keys(), vec![]),
    ));
    found.push(lookup(
        t(),
        &["a", "b"],
        &["a", "b"],
        seeds(
            vec![
                unique("key_1", ColumnType::Int64),
                unique("key_2", ColumnType::Utf8),
            ],
            vec![],
        ),
    ));
    found.push(lookup(
        t(),
        &["a", "b"],
        &["a", "b"],
        seeds(
            vec![
                unique("key_1", ColumnType::Int64),
                field("key_2", ColumnType::Utf8, true),
            ],
            vec![],
        ),
    ));
    found.push(lookup(
        t(),
        &["id"],
        &[],
        seeds(one_key(ColumnType::Int64), vec![]),
    ));
    found.push(lookup(
        t(),
        &["a", "b"],
        &["a"],
        seeds(one_key(ColumnType::Int64), vec![]),
    ));
    found
}

/// Every request of the matrix, each kind in the shapes above.
fn requests() -> Vec<DataRequest> {
    let mut found = describes();
    found.extend(reads());
    found.extend(samples());
    found.extend(neighbors());
    found.extend(lookups());
    found
}

fn provider(dialect: &str, namespace: &[Option<&str>]) -> ProviderChoice {
    ProviderChoice::Sql {
        timestamp_encoding: None,
        dialect: Dialect::new(dialect),
        namespace: namespace.iter().map(|p| p.map(str::to_owned)).collect(),
        view: None,
    }
}

#[test]
fn the_matrix_holds_every_request_kind_and_catalog_part_on_every_dialect() {
    let requests = requests();
    let kinds: std::collections::BTreeSet<&str> = requests.iter().map(DataRequest::name).collect();
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        [
            "column_ranges",
            "count_rows",
            "describe",
            "key_counts",
            "lookup",
            "neighbors",
            "preview",
            "repeated_key",
            "sample_entities",
            "scan",
            "time_bounds",
        ]
    );
    let parts: std::collections::BTreeSet<String> = requests
        .iter()
        .filter_map(|r| match r {
            DataRequest::Describe { part, .. } => Some(format!("{part:?}")),
            _ => None,
        })
        .collect();
    assert_eq!(parts.len(), 7);
    // Each dialect plans some of each kind, and refuses some.
    for dialect in RENDERED {
        let p = provider(dialect, &[]);
        for kind in [
            "column_ranges",
            "count_rows",
            "describe",
            "key_counts",
            "lookup",
            "neighbors",
            "preview",
            "repeated_key",
            "sample_entities",
            "scan",
            "time_bounds",
        ] {
            let answers: Vec<_> = requests
                .iter()
                .filter(|r| r.name() == kind)
                .map(|r| Data::new().plan(r, &p))
                .collect();
            assert!(
                answers.iter().any(Result::is_ok),
                "{dialect} plans no {kind}"
            );
            assert!(
                answers.iter().any(Result::is_err),
                "{dialect} refuses no {kind}"
            );
        }
    }
}

#[test]
fn provider_capabilities_and_namespace_probes_have_explicit_contracts() {
    let data = Data::new();
    for dialect in RENDERED {
        let p = provider(dialect, &[]);
        let capabilities = data.capabilities(&p).unwrap();
        let names: Vec<_> = capabilities
            .optional
            .iter()
            .map(|kind| kind.name())
            .collect();
        assert_eq!(
            names,
            [
                "preview",
                "count_rows",
                "key_counts",
                "time_bounds",
                "column_ranges",
                "repeated_key",
                "sample_entities",
                "neighbors",
                "lookup"
            ]
        );
        assert_eq!(capabilities.max_keys_per_request, None);
        assert_eq!(
            capabilities.max_seed,
            if dialect == "duckdb" {
                i32::MAX as u64
            } else {
                u64::MAX
            }
        );
        assert_eq!(
            serde_json::to_value(capabilities.sampling).unwrap(),
            if dialect == "snowflake" {
                "unseeded"
            } else {
                "seeded_stable"
            }
        );
        assert_eq!(
            serde_json::to_value(capabilities.text_times).unwrap(),
            "value_by_value"
        );
        assert_eq!(
            serde_json::to_value(capabilities.null_order).unwrap(),
            if matches!(dialect, "sqlite" | "databricks") {
                "first"
            } else {
                "last"
            }
        );
        assert_eq!(
            serde_json::to_value(capabilities.index_advice).unwrap(),
            match dialect {
                "duckdb" => json!("single_column"),
                "sqlite" => json!("leading_column"),
                _ => Value::Null,
            }
        );
        assert!(capabilities.derived_relations);
        assert_eq!(capabilities.sample_rows, dialect == "databricks");
        assert_eq!(
            capabilities.neighbor_lookback,
            matches!(dialect, "databricks" | "snowflake")
        );
        assert_eq!(capabilities.keys_optional, dialect == "databricks");
        assert_eq!(capabilities.key_enforcement, dialect == "snowflake");
        assert_eq!(
            capabilities.opaque_vendor.as_deref(),
            (dialect == "duckdb").then_some("DuckDB")
        );
        assert_eq!(data.source_database(&p), Some(dialect));
        assert_eq!(
            data.read_as(&provider(dialect, &[Some("s")]), &relation(&["t"]))
                .source,
            ["s", "t"]
        );
        assert_eq!(
            data.read_as(&provider(dialect, &[Some("s")]), &relation(&["other", "t"]))
                .source,
            ["other", "t"]
        );
    }
    let unknown = provider("nope", &[]);
    assert!(data.source_database(&unknown).is_none());
    let error = data.capabilities(&unknown).unwrap_err();
    assert_eq!(error.code, sdm_contract::ErrorCode::UnsupportedRequest);
    assert_eq!(
        data.plan(&requests()[0], &unknown).unwrap_err().code,
        sdm_contract::ErrorCode::UnsupportedRequest
    );
}

#[test]
fn scans_and_counts_preserve_projection_filters_and_limits() {
    use sdm_contract::data::{ExpectedSchema, FetchPlan};
    for dialect in RENDERED {
        let quote = if dialect == "databricks" { '`' } else { '"' };
        let q = |name: &str| format!("{quote}{name}{quote}");
        let table = relation(&["t"]);
        let cases = [
            (
                DataRequest::Scan {
                    table: table.clone(),
                    columns: names(&["value", "id"]),
                    filters: vec![not_null("id")],
                    limit: Some(5),
                },
                format!(
                    "SELECT {}, {} FROM {} WHERE {} IS NOT NULL LIMIT 5",
                    q("value"),
                    q("id"),
                    if dialect == "postgres" {
                        format!("{}.{}", q("public"), q("t"))
                    } else if dialect == "snowflake" {
                        format!("{}.{}.{}", q("db"), q("public"), q("t"))
                    } else {
                        q("t")
                    },
                    q("id")
                ),
            ),
            (
                DataRequest::CountRows {
                    table,
                    filters: vec![not_null("id")],
                },
                format!(
                    "SELECT COUNT(*){} FROM {} WHERE {} IS NOT NULL",
                    if dialect == "databricks" {
                        String::new()
                    } else {
                        format!(" AS {}", q("rows"))
                    },
                    if dialect == "postgres" {
                        format!("{}.{}", q("public"), q("t"))
                    } else if dialect == "snowflake" {
                        format!("{}.{}.{}", q("db"), q("public"), q("t"))
                    } else {
                        q("t")
                    },
                    q("id")
                ),
            ),
        ];
        for (request, expected_sql) in cases {
            let FetchPlan::Sql {
                statements,
                expect,
                result_conversions,
            } = Data::new()
                .plan(
                    &request,
                    &provider(
                        dialect,
                        if dialect == "postgres" {
                            &[Some("public")]
                        } else if dialect == "snowflake" {
                            &[Some("db"), Some("public")]
                        } else {
                            &[]
                        },
                    ),
                )
                .unwrap()
            else {
                panic!("{dialect}: expected SQL");
            };
            assert_eq!(statements.len(), 1, "{dialect}");
            assert_eq!(statements[0].text, expected_sql, "{dialect}");
            assert!(
                statements[0].params.is_empty(),
                "unkeyed reads bind no seeds"
            );
            assert_eq!(expect, ExpectedSchema::Discover {});
            assert_eq!(
                result_conversions,
                if dialect == "postgres" {
                    vec![sdm_contract::data::SqlResultConversion::ZonedTextUtc]
                } else {
                    vec![]
                }
            );
        }
    }
}

#[test]
fn lookup_retains_typed_seed_attachment_and_field_order() {
    use sdm_contract::data::{FetchPlan, SqlParam};
    for dialect in RENDERED {
        for key_fields in [
            one_key(ColumnType::Int64),
            one_key(ColumnType::Utf8),
            two_keys(),
        ] {
            let key_names = if key_fields.len() == 1 {
                names(&["id"])
            } else {
                names(&["id", "tenant"])
            };
            let supplied = seeds(key_fields, vec![]);
            let request = DataRequest::Lookup {
                table: relation(&["t"]),
                key: key_names,
                columns: names(&["id", "value"]),
                seeds: supplied.clone(),
            };
            let planned = Data::new().plan(
                &request,
                &provider(
                    dialect,
                    if dialect == "postgres" {
                        &[Some("public")]
                    } else if dialect == "snowflake" {
                        &[Some("db"), Some("public")]
                    } else {
                        &[]
                    },
                ),
            );
            if supplied.columns.len() == 3 && matches!(dialect, "databricks" | "snowflake") {
                // Warehouse seed transport currently admits one key only.
                let error = planned.unwrap_err();
                assert_eq!(error.code, sdm_contract::ErrorCode::UnsupportedRequest);
                assert_eq!(
                    serde_json::to_value(&error).unwrap()["fields"]["field"],
                    json!({"t": "utf8", "v": "key"})
                );
                continue;
            }
            let FetchPlan::Sql { statements, .. } = planned.unwrap() else {
                panic!("{dialect}: expected SQL");
            };
            let bound: Vec<_> = statements
                .iter()
                .flat_map(|s| &s.params)
                .filter_map(|p| {
                    if let SqlParam::KeySet { keys, .. } = p {
                        Some(keys)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(
                bound,
                [&supplied],
                "{dialect}: preserve seed attachment, field order, types and metadata"
            );
        }
    }
}

fn settled_provider(dialect: &str) -> ProviderChoice {
    provider(
        dialect,
        match dialect {
            "postgres" => &[Some("public")],
            "snowflake" => &[Some("db"), Some("public")],
            _ => &[],
        },
    )
}

#[test]
fn malformed_requests_refuse_the_named_field_before_rendering() {
    let cases = [
        (DataRequest::KeyCounts { keys: vec![] }, "keys"),
        (DataRequest::TimeBounds { columns: vec![] }, "columns"),
        (
            DataRequest::ColumnRanges {
                table: relation(&["t"]),
                columns: vec![],
            },
            "columns",
        ),
        (
            DataRequest::RepeatedKey {
                table: relation(&["t"]),
                key: vec![],
            },
            "key",
        ),
        (
            DataRequest::Lookup {
                table: relation(&["t"]),
                key: vec![],
                columns: names(&["id"]),
                seeds: seeds(one_key(ColumnType::Int64), vec![]),
            },
            "key",
        ),
        (
            DataRequest::Neighbors {
                table: relation(&["t"]),
                key: names(&["id"]),
                columns: names(&["id"]),
                seeds: seeds(one_key(ColumnType::Int64), vec![]),
                window: None,
                per_seed: Some(2.into()),
                order: None,
            },
            "order",
        ),
    ];
    for (request, field) in &cases {
        let error = request.check().unwrap_err();
        assert_eq!(
            error.code,
            sdm_contract::ErrorCode::InvalidRequest,
            "{request:?}"
        );
        let fields = serde_json::to_value(&error).unwrap()["fields"].clone();
        assert_eq!(fields["field"], json!({"t": "utf8", "v": field}));
        assert_eq!(fields["request"], json!({"t": "utf8", "v": request.name()}));
    }
}

#[test]
fn key_counts_and_time_bounds_have_independent_fixed_schemas() {
    use sdm_contract::data::ExpectedSchema;
    for dialect in RENDERED {
        for (request, fields) in [
            (
                DataRequest::KeyCounts {
                    keys: vec![TableKey {
                        table: relation(&["t"]),
                        columns: names(&["id"]),
                        unique: false,
                        label: None,
                    }],
                },
                vec![
                    field("k", ColumnType::Int32, false),
                    field("n", ColumnType::Int64, false),
                    field("d", ColumnType::Int64, false),
                    field("z", ColumnType::Int64, false),
                ],
            ),
            (
                DataRequest::TimeBounds {
                    columns: vec![TableColumn {
                        table: relation(&["t"]),
                        column: "at".into(),
                    }],
                },
                vec![
                    field("table_index", ColumnType::Int32, false),
                    field(
                        "min_date",
                        ColumnType::Timestamp {
                            unit: Unit::Us,
                            zone: None,
                        },
                        true,
                    ),
                    field(
                        "max_date",
                        ColumnType::Timestamp {
                            unit: Unit::Us,
                            zone: None,
                        },
                        true,
                    ),
                ],
            ),
        ] {
            let plan = Data::new()
                .plan(&request, &settled_provider(dialect))
                .unwrap();
            let expected = match (&request, dialect) {
                (
                    DataRequest::TimeBounds { .. },
                    "sqlite" | "postgres" | "databricks" | "snowflake",
                ) => ExpectedSchema::Discover {},
                (DataRequest::KeyCounts { .. }, "sqlite" | "snowflake") => ExpectedSchema::Schema {
                    fields: if dialect == "snowflake" {
                        ["K", "N", "D", "Z"]
                    } else {
                        ["k", "n", "d", "z"]
                    }
                    .into_iter()
                    .map(|name| field(name, ColumnType::Int64, false))
                    .collect(),
                },
                _ => ExpectedSchema::Schema { fields },
            };
            assert_eq!(plan.expect(), &expected, "{dialect}: {request:?}");
        }
    }
}

#[test]
fn identifiers_escape_quotes_without_splitting_literal_dots() {
    use sdm_contract::data::FetchPlan;
    for dialect in RENDERED {
        let (name, quoted) = if dialect == "databricks" {
            ("a.`b's", "`a.``b's`")
        } else {
            ("a.\"b's", "\"a.\"\"b's\"")
        };
        let request = DataRequest::Preview {
            table: relation(&[name]),
            columns: names(&[name]),
            limit: 3,
        };
        let FetchPlan::Sql { statements, .. } = Data::new()
            .plan(&request, &settled_provider(dialect))
            .unwrap()
        else {
            panic!("SQL");
        };
        let prefix = match dialect {
            "postgres" => "\"public\".",
            "snowflake" => "\"db\".\"public\".",
            _ => "",
        };
        assert_eq!(
            statements.last().unwrap().text,
            format!(
                "SELECT {quoted} FROM {prefix}{quoted}{} LIMIT 3",
                if dialect == "sqlite" {
                    " ORDER BY rowid"
                } else {
                    ""
                }
            )
        );
    }
}

#[test]
fn sample_seed_admission_matches_dialect_capabilities() {
    for dialect in RENDERED {
        for (seed, accepted) in [
            (Scalar::UInt64(42), true),
            (Scalar::UInt64(i32::MAX as u64), true),
            (Scalar::UInt64(1 << 31), dialect != "duckdb"),
            (Scalar::Int64(-1), !matches!(dialect, "duckdb" | "sqlite")),
            (Scalar::Float64(1.5), dialect == "snowflake"),
            (Scalar::Utf8("42".into()), dialect == "snowflake"),
        ] {
            let request = DataRequest::SampleEntities {
                rows: Some(100),
                table: relation(&["t"]),
                key: names(&["id"]),
                columns: names(&["id", "x"]),
                n: 5,
                seed,
                filters: vec![not_null("id")],
                key_dtype: Some(Dtype::Int),
            };
            let result = Data::new().plan(&request, &settled_provider(dialect));
            assert_eq!(
                result.is_ok(),
                accepted,
                "{dialect}: {request:?}: {result:?}"
            );
            if let Err(error) = result {
                assert_eq!(
                    serde_json::to_value(error).unwrap()["fields"]["field"],
                    json!({"t": "utf8", "v": "seed"})
                );
            }
        }
    }
}

#[test]
fn neighbor_windows_and_per_seed_order_keep_endpoint_semantics() {
    use sdm_contract::data::FetchPlan;
    for dialect in RENDERED {
        for (lower, upper) in [
            (Some(false), Some(true)),
            (Some(true), Some(false)),
            (None, Some(true)),
            (Some(false), None),
        ] {
            let request = DataRequest::Neighbors {
                table: relation(&["t"]),
                key: names(&["fk"]),
                columns: names(&["fk", "at", "x", "y"]),
                seeds: seeds(
                    one_key(ColumnType::Int64),
                    [("start", lower), ("end", upper)]
                        .into_iter()
                        .filter_map(|(name, present)| {
                            present.map(|_| field(name, timestamp(), false))
                        })
                        .collect(),
                ),
                window: Some(Window {
                    source_type: None,
                    column: "at".into(),
                    lower: lower.and_then(|inclusive| bound("start", inclusive)),
                    upper: upper.and_then(|inclusive| bound("end", inclusive)),
                }),
                per_seed: Some(3.into()),
                order: Some(TieBreak(vec![
                    term("x", false, true, true),
                    term("y", false, false, false),
                ])),
            };
            let FetchPlan::Sql { statements, .. } = Data::new()
                .plan(&request, &settled_provider(dialect))
                .unwrap()
            else {
                panic!("SQL");
            };
            let text = statements
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            // These operators occur only in the window predicates; the row bound is <= 3.
            for (inclusive, strict, closed) in [(lower, " > ", " >= "), (upper, " < ", " <= ")] {
                if let Some(inclusive) = inclusive {
                    let operator = if inclusive { closed } else { strict };
                    assert!(
                        text.lines().any(|line| line.contains(operator)
                            && !line.trim_start().starts_with(") <= 3")),
                        "{dialect}: missing {operator}: {text}"
                    );
                    if strict == " > " {
                        assert!(
                            !text.contains(if inclusive { strict } else { closed }),
                            "{text}"
                        );
                    }
                } else if strict == " > " {
                    assert!(!text.contains(strict) && !text.contains(closed), "{text}");
                } else {
                    assert!(!text.contains(strict), "{text}");
                    assert!(
                        !text.lines().any(|line| line.contains(closed)
                            && !line.trim_start().starts_with(") <= 3")),
                        "{text}"
                    );
                }
            }
            let (first, second) = match dialect {
                "databricks" => ("`x` DESC", "`y`"),
                "snowflake" => ("\"x\" DESC NULLS LAST", "\"y\" NULLS FIRST"),
                "sqlite" => ("\"x\" DESC", "\"y\""),
                _ => ("\"x\" DESC NULLS LAST", "\"y\" ASC NULLS FIRST"),
            };
            let order = text.split("ORDER BY ").nth(1).expect("per-seed order");
            assert!(
                order.find(first).unwrap() < order.find(second).unwrap(),
                "{dialect}: {order}"
            );
            if matches!(dialect, "sqlite" | "postgres") {
                assert!(text.contains("LIMIT 3"));
            } else {
                assert!(text.contains("PARTITION BY ") && text.contains(") <= 3"));
            }
        }
    }
}

#[test]
fn catalog_scope_is_explicit_for_each_dialect() {
    // Each pair says whether whole-catalog and single-table requests are supported.
    let parts = [
        DescribePart::Tables,
        DescribePart::Columns,
        DescribePart::Keys,
        DescribePart::ForeignKeys,
        DescribePart::Indexes,
        DescribePart::IndexColumns,
        DescribePart::Namespace,
    ];
    for (dialect, admitted) in [
        (
            "duckdb",
            [
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (true, false),
                (false, false),
                (true, true),
            ],
        ),
        (
            "sqlite",
            [
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (false, true),
                (true, true),
            ],
        ),
        (
            "postgres",
            [
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (true, false),
                (false, true),
                (true, true),
            ],
        ),
        (
            "databricks",
            [
                (true, true),
                (false, true),
                (false, true),
                (false, true),
                (false, false),
                (false, false),
                (true, false),
            ],
        ),
        (
            "snowflake",
            [
                (true, false),
                (false, true),
                (false, true),
                (false, true),
                (false, false),
                (false, false),
                (true, false),
            ],
        ),
    ] {
        for (part, (whole, single)) in parts.into_iter().zip(admitted) {
            for (tables, expected) in [(vec![], whole), (vec![relation(&["t"])], single)] {
                let request = DataRequest::Describe { tables, part };
                let plan = Data::new().plan(&request, &settled_provider(dialect));
                assert_eq!(plan.is_ok(), expected, "{dialect} {request:?}: {plan:?}");
                if let Err(error) = plan {
                    assert_eq!(error.code, sdm_contract::ErrorCode::UnsupportedRequest);
                    assert_eq!(
                        serde_json::to_value(error).unwrap()["fields"]["request"],
                        json!({"t":"utf8","v":"describe"})
                    );
                }
            }
        }
    }
}

#[test]
fn renderer_refuses_filters_it_cannot_preserve() {
    for dialect in RENDERED {
        let request = DataRequest::Scan {
            table: relation(&["t"]),
            columns: names(&["at"]),
            limit: None,
            filters: vec![Filter::TimeRange {
                column: "at".into(),
                lower: None,
                upper: None,
            }],
        };
        let error = Data::new()
            .plan(&request, &settled_provider(dialect))
            .unwrap_err();
        assert_eq!(error.code, sdm_contract::ErrorCode::UnsupportedRequest);
        assert_eq!(
            serde_json::to_value(error).unwrap()["fields"]["field"],
            json!({"t":"utf8","v":"filters"})
        );
    }
}

#[test]
fn ranges_and_repeated_composite_keys_preserve_aggregate_semantics() {
    use sdm_contract::data::FetchPlan;
    for dialect in RENDERED {
        let quote = if dialect == "databricks" { '`' } else { '"' };
        let q = |s: &str| format!("{quote}{s}{quote}");
        for request in [
            DataRequest::ColumnRanges {
                table: relation(&["t"]),
                columns: names(&["a", "b"]),
            },
            DataRequest::RepeatedKey {
                table: relation(&["t"]),
                key: names(&["a", "b"]),
            },
        ] {
            let FetchPlan::Sql { statements, .. } = Data::new()
                .plan(&request, &settled_provider(dialect))
                .unwrap()
            else {
                panic!("SQL")
            };
            let text = &statements.last().unwrap().text;
            if matches!(request, DataRequest::ColumnRanges { .. }) {
                let aggregates = format!(
                    "COUNT(*), MIN({a}), MAX({a}), COUNT({a}), MIN({b}), MAX({b}), COUNT({b})",
                    a = q("a"),
                    b = q("b")
                );
                assert!(text.contains(&aggregates), "{dialect}: {text}");
            } else {
                for fragment in [
                    format!("{} IS NOT NULL AND {} IS NOT NULL", q("a"), q("b")),
                    format!("GROUP BY {}, {}", q("a"), q("b")),
                    "HAVING COUNT(*) > 1".into(),
                    "LIMIT 1".into(),
                ] {
                    assert!(
                        text.contains(&fragment),
                        "{dialect}: missing {fragment}: {text}"
                    );
                }
            }
        }
    }
}
