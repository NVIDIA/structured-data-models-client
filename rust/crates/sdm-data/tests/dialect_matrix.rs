// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every request kind, on every dialect this build renders, in the shapes
//! that steer its rendering, held to what the dialects rendered before
//! they shared one SQL builder (T3.18, design fix E): the plan (each
//! statement's text and parameters, and the schema it expects) or the
//! refusal, byte for byte, and each dialect's capabilities and quirks.
//!
//! The goldens (`*_goldens.rs`) pin what the contract cases run; this
//! matrix reaches what they do not: every request kind (the goldens hold
//! no `column_ranges`, `repeated_key` or `count_rows`), each part of the
//! catalog, relations of every depth and of awkward names, namespaces,
//! composite keys, each window bound, each per-seed order, the seed key
//! types each dialect binds differently, and each refusal a rendering
//! makes. `dialect_matrix.jsonl` holds one line per case, recorded from
//! the dialects at the phase head the refactor started from (e6c3791b).
//!
//! A deliberate SQL change fails here, as it fails a golden: the failure
//! prints each changed line ready to paste, as the dialect now renders it.

use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, Filter, KeySet, ProviderChoice, Relation,
    SEED_COUNT_UNREAD, SEED_KEY_TYPE, SEED_KEY_UNIQUE, SEED_STORED_TIME, SeedBound, TableColumn,
    TableKey, TieBreak, TieTerm, Window, field,
};
use sdm_contract::ops::Dtype;
use sdm_contract::{ColumnType, Field, Scalar, Unit};
use sdm_data::Data;
use serde_json::{Value, json};

/// The recorded matrix, read at build time: the core's crates do no I/O,
/// their tests included.
const RECORDED: &str = include_str!("dialect_matrix.jsonl");

/// The dialects held: the five this build renders (Databricks from T3.4,
/// Snowflake from T3.6), and one it refuses, a name it does not know.
const RENDERED: [&str; 5] = ["duckdb", "sqlite", "postgres", "databricks", "snowflake"];
const REFUSED: [&str; 1] = ["nope"];

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
        per_seed,
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
            column: "at".to_owned(),
            lower: bound("start", false),
            upper: bound("end", true),
        },
        Window {
            column: "at".to_owned(),
            lower: bound("start", true),
            upper: bound("end", false),
        },
        Window {
            column: "at".to_owned(),
            lower: None,
            upper: bound("end", true),
        },
        Window {
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

/// The namespaces a request is planned under: none, and, for a read of the
/// table `t` and a whole catalog read, one schema and one past a part left
/// to the connection; for the table list and a preview, one with a part
/// left inside it and two parts too.
fn namespaces(request: &DataRequest) -> Vec<Vec<Option<&'static str>>> {
    let t = relation(&["t"]);
    let (read, catalog) = match request {
        DataRequest::Describe { tables, part } => (
            tables.is_empty() || tables == std::slice::from_ref(&t),
            *part == DescribePart::Tables,
        ),
        DataRequest::Preview { table, .. } => (*table == t, true),
        DataRequest::Scan { table, .. }
        | DataRequest::CountRows { table, .. }
        | DataRequest::SampleEntities { table, .. }
        | DataRequest::Neighbors { table, .. }
        | DataRequest::Lookup { table, .. } => (*table == t, false),
        DataRequest::KeyCounts { keys } => (keys.len() > 2, false),
        DataRequest::TimeBounds { columns } => (columns.len() > 1, false),
        _ => (false, false),
    };
    let mut found = vec![vec![]];
    if read {
        found.extend([vec![Some("s")], vec![None, Some("s")]]);
    }
    if catalog {
        found.extend([vec![Some("s"), None], vec![Some("db"), Some("s")]]);
    }
    found
}

fn provider(dialect: &str, namespace: &[Option<&str>]) -> ProviderChoice {
    ProviderChoice::Sql {
        dialect: Dialect::new(dialect),
        namespace: namespace.iter().map(|p| p.map(str::to_owned)).collect(),
        view: None,
    }
}

/// What the layer answers for `request` on `provider`: its plan, or its
/// refusal.
/// A key set a statement binds is written as the request's seeds where it
/// is they, as every one is today, so the line names each once.
fn answer(request: &DataRequest, provider: &ProviderChoice) -> Value {
    let mut plan = match Data::new().plan(request, provider) {
        Ok(plan) => json!(plan),
        Err(error) => return json!({ "error": error }),
    };
    let seeds = match request {
        DataRequest::Neighbors { seeds, .. } | DataRequest::Lookup { seeds, .. } => json!(seeds),
        _ => Value::Null,
    };
    if let Some(statements) = plan["statements"].as_array_mut() {
        for param in statements
            .iter_mut()
            .filter_map(|s| s["params"].as_array_mut())
            .flatten()
        {
            if param["keys"] == seeds {
                param["keys"] = json!("the request's seeds");
            }
        }
    }
    json!({ "plan": plan })
}

/// What the layer says of `dialect` itself: its capabilities and quirks.
fn probe(dialect: &str) -> Value {
    let data = Data::new();
    let p = provider(dialect, &[]);
    let capabilities = match data.capabilities(&p) {
        Ok(c) => json!({ "capabilities": c }),
        Err(error) => json!({ "error": error }),
    };
    json!({
        "capabilities": capabilities,
        "source_database": data.source_database(&p),
        "read_as": data.read_as(&provider(dialect, &[Some("s")]), &relation(&["t"])),
    })
}

/// The matrix as it renders today, one line per case.
fn rendered() -> Vec<String> {
    let mut lines = Vec::new();
    for dialect in RENDERED.iter().chain(REFUSED.iter()) {
        lines.push(json!({ "dialect": dialect, "probe": probe(dialect) }).to_string());
    }
    let requests = requests();
    for dialect in RENDERED {
        for request in &requests {
            for namespace in namespaces(request) {
                let p = provider(dialect, &namespace);
                lines.push(
                    json!({
                        "dialect": dialect,
                        "namespace": namespace,
                        "request": request,
                        "answer": answer(request, &p),
                    })
                    .to_string(),
                );
            }
        }
    }
    for dialect in REFUSED {
        let request = &requests[0];
        lines.push(
            json!({
                "dialect": dialect,
                "namespace": [],
                "request": request,
                "answer": answer(request, &provider(dialect, &[])),
            })
            .to_string(),
        );
    }
    lines
}

#[test]
fn every_request_kind_renders_as_recorded_on_every_dialect() {
    let now = rendered();
    let recorded: Vec<&str> = RECORDED.lines().filter(|l| !l.is_empty()).collect();
    if now.iter().map(String::as_str).eq(recorded.iter().copied()) {
        return;
    }
    let changed: Vec<&String> = now
        .iter()
        .filter(|line| !recorded.contains(&line.as_str()))
        .collect();
    let stale = recorded
        .iter()
        .filter(|line| !now.iter().any(|n| n == *line))
        .count();
    panic!(
        "the matrix renders {} line(s) not recorded, and {stale} recorded line(s) no longer render \
         (of {} now, {} recorded); each new line as the layer renders it:\n{}",
        changed.len(),
        now.len(),
        recorded.len(),
        changed
            .iter()
            .map(|l| l.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
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
            let answers: Vec<Value> = requests
                .iter()
                .filter(|r| r.name() == kind)
                .map(|r| answer(r, &p))
                .collect();
            assert!(
                answers.iter().any(|a| a.get("plan").is_some()),
                "{dialect} plans no {kind}"
            );
            assert!(
                answers.iter().any(|a| a.get("error").is_some()),
                "{dialect} refuses no {kind}"
            );
        }
    }
}
