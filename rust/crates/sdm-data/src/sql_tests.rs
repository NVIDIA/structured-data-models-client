// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The dialect trait's tests (T3.18, design fix E): each dialect this
//! build renders is found by its name and declares its own depth,
//! quoting, namespace, seed names, casts and quirks; and a dialect that
//! implements only the trait's required hooks gets every shared statement
//! shape from the one builder, as a warehouse dialect will (T3.4 onward).
//! The statements themselves are held by the goldens and by
//! `tests/dialect_matrix.rs`.

use arrow_array::RecordBatch;
use sdm_contract::catalog::CatalogAnswer;
use sdm_contract::data::{
    DataRequest, DescribePart, Determinism, ExpectedSchema, FetchPlan, Filter, KeySet,
    ProviderCapabilities, Relation, SeedBound, SqlParam, TableColumn, TableKey, TieTerm, Window,
    field, seed_integer,
};
use sdm_contract::{ColumnType, CoreError, Detail, Scalar, Unit};

use crate::PROVIDERS;
use crate::sql::{self, DIALECTS, Sample, SeedRead, SqlDialect};

fn dialect(name: &str) -> &'static dyn SqlDialect {
    sql::dialect(name).unwrap_or_else(|| panic!("no dialect {name}"))
}

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

#[test]
fn each_dialect_is_found_by_its_name_and_only_those() {
    let found: Vec<&str> = DIALECTS.iter().map(|d| d.name()).collect();
    assert_eq!(
        found,
        ["duckdb", "sqlite", "postgres", "databricks", "snowflake"]
    );
    for name in &found {
        assert_eq!(dialect(name).name(), *name);
    }
    // The providers by the names the runners use: a dialect's, or its legs
    // (Databricks's SQL warehouse and Spark session run one dialect).
    assert_eq!(
        PROVIDERS,
        [
            "duckdb",
            "sqlite",
            "postgres",
            "databricks-sql",
            "databricks-spark",
            "snowflake"
        ]
    );
    for name in ["Snowflake", "DuckDB", "", "nope"] {
        assert!(sql::dialect(name).is_none(), "{name}");
    }
}

#[test]
fn each_dialect_declares_its_depth_quoting_and_namespace() {
    for (name, most, namespace, max_seed) in [
        ("duckdb", 3, 0, (1 << 31) - 1),
        ("sqlite", 2, 0, u64::MAX),
        ("postgres", 2, 1, u64::MAX),
        ("databricks", 3, 2, u64::MAX),
        ("snowflake", 3, 2, u64::MAX),
    ] {
        let d = dialect(name);
        assert_eq!(d.most_parts(), most, "{name}");
        assert_eq!(d.namespace_depth(), namespace, "{name}");
        assert_eq!(d.capabilities().max_seed, max_seed, "{name}");
        let quoted = if name == "databricks" {
            "`a\"b.c`"
        } else {
            "\"a\"\"b.c\""
        };
        assert_eq!(d.identifier("a\"b.c"), quoted, "{name}");
    }
}

#[test]
fn each_dialect_names_its_seeds_past_a_table_of_that_name() {
    let request = DataRequest::Preview {
        table: relation(&["t"]),
        columns: names(&["id"]),
        limit: 1,
    };
    let duckdb = dialect("duckdb");
    assert_eq!(
        duckdb.seeds_for(&request, &relation(&["t"])),
        "__sdm_seeds__"
    );
    // DuckDB matches names without case.
    assert_eq!(
        duckdb.seeds_for(&request, &relation(&["main", "__SDM_SEEDS__"])),
        "__sdm_seeds___"
    );
    let postgres = dialect("postgres");
    assert_eq!(
        postgres.seeds_for(&request, &relation(&["__sdm_seeds__"])),
        "__sdm_seeds___"
    );
    // A CTE shadows only a name of one part in its own case.
    assert_eq!(
        postgres.seeds_for(&request, &relation(&["__SDM_SEEDS__"])),
        "__sdm_seeds__"
    );
    assert_eq!(
        postgres.seeds_for(&request, &relation(&["s", "__sdm_seeds__"])),
        "__sdm_seeds__"
    );
    // SQLite's name is the request's own, so each key set of a run has one.
    let sqlite = dialect("sqlite");
    let named = sqlite.seeds_for(&request, &relation(&["t"]));
    assert!(
        named.starts_with("__sdm_seeds_") && named.ends_with("__"),
        "{named}"
    );
    assert_eq!(named.len(), "__sdm_seeds_".len() + 16 + 2);
    let other = DataRequest::Preview {
        table: relation(&["u"]),
        columns: names(&["id"]),
        limit: 1,
    };
    assert_ne!(sqlite.seeds_for(&other, &relation(&["t"])), named);
    assert_eq!(
        sqlite.seeds_for(&request, &relation(&[&named.to_uppercase()])),
        format!("{named}_")
    );
    // Where each reads the relation bound.
    assert_eq!(duckdb.seeds_relation("x"), "\"x\"");
    assert_eq!(postgres.seeds_relation("x"), "\"x\"");
    assert_eq!(sqlite.seeds_relation("x"), "temp.\"x\"");
}

#[test]
fn each_dialect_casts_a_time_as_it_compares_one() {
    assert_eq!(
        dialect("duckdb").as_time("t.\"at\""),
        "CAST(t.\"at\" AS TIMESTAMP)"
    );
    assert_eq!(dialect("sqlite").as_time("t.\"at\""), "t.\"at\"");
    assert_eq!(dialect("postgres").as_time("t.\"at\""), "t.\"at\"");
}

#[test]
fn each_dialect_declares_its_quirks() {
    let quirks = |name: &str| {
        // In its capabilities since contract v28 (P2-MID2-2).
        let c = dialect(name).capabilities();
        (
            c.opaque_vendor,
            c.compares_stored_times,
            c.samples_by_key_dtype,
            c.reads_nan_as_missing,
        )
    };
    assert_eq!(
        quirks("duckdb"),
        (Some("DuckDB".to_owned()), false, false, false)
    );
    assert_eq!(quirks("sqlite"), (None, true, true, false));
    assert_eq!(quirks("postgres"), (None, false, false, true));
    assert_eq!(quirks("databricks"), (None, false, false, false));
    assert_eq!(quirks("snowflake"), (None, false, false, false));
}

/// A dialect of the trait's required hooks alone, quoting with backticks
/// and taking relations of one part, as a new dialect starts.
struct Plain;

impl SqlDialect for Plain {
    fn name(&self) -> &'static str {
        "plain"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            opaque_vendor: None,
            compares_stored_times: false,
            samples_by_key_dtype: false,
            reads_nan_as_missing: false,
            derived_relations: false,
            neighbor_lookback: false,
            keys_optional: false,
            key_enforcement: false,
            sample_rows: false,
            optional: Default::default(),
            max_keys_per_request: None,
            sampling: Determinism::SeededStable,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            fallback: None,
        }
    }

    fn most_parts(&self) -> usize {
        1
    }

    fn identifier(&self, name: &str) -> String {
        format!("`{}`", name.replace('`', "``"))
    }

    fn seeds_for(&self, _request: &DataRequest, _table: &Relation) -> String {
        "seeds".to_owned()
    }

    fn sample_seed(&self, request: &DataRequest, seed: &Scalar) -> Result<i128, CoreError> {
        seed_integer(seed).ok_or_else(|| sql::not_taken(request, "seed", "no such seed"))
    }

    fn sample(&self, sample: &Sample<'_>) -> Vec<String> {
        vec![format!(
            "SELECT {}\nFROM {}{}\nLIMIT {} -- {}",
            sample.projected, sample.table, sample.filtered, sample.n, sample.seed
        )]
    }

    fn neighbors_per_seed(&self, read: &SeedRead<'_>, limit: u64, terms: &[TieTerm]) -> String {
        let order: Vec<String> = terms
            .iter()
            .map(|t| self.order_term(t, "t.", None))
            .collect();
        format!(
            "{}\nORDER BY {}\nLIMIT {limit}",
            read.joined(),
            order.join(", ")
        )
    }

    fn lookup(&self, read: &SeedRead<'_>) -> String {
        read.joined()
    }

    fn describe(
        &self,
        _request: &DataRequest,
        tables: &[Relation],
        part: DescribePart,
        _namespace: &[String],
    ) -> Result<(String, Vec<SqlParam>), CoreError> {
        // A catalog read binding the name it looks for, as a warehouse's
        // does (Databricks's `lower(?)`, T3.4).
        Ok(match (part, tables) {
            (DescribePart::Columns, [table]) => (
                "SELECT column_name FROM information_schema.columns WHERE lower(table_name) = lower(?)"
                    .to_owned(),
                vec![SqlParam::Value {
                    value: Scalar::Utf8(table.source.last().cloned().unwrap_or_default()),
                }],
            ),
            _ => ("SHOW TABLES".to_owned(), vec![]),
        })
    }

    fn expected(&self, request: &DataRequest) -> ExpectedSchema {
        request.expected()
    }

    fn read(
        &self,
        _request: &DataRequest,
        part: DescribePart,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        crate::read::host(part, batches)
    }
}

/// `request`'s statements on [`Plain`].
fn plain(request: &DataRequest) -> Result<Vec<String>, CoreError> {
    match sql::plan(&Plain, request, &[], 0)? {
        FetchPlan::Sql { statements, .. } => {
            assert!(statements.iter().all(|s| s.dialect.name() == "plain"));
            Ok(statements.into_iter().map(|s| s.text).collect())
        }
        FetchPlan::Host { .. } => panic!("a dialect planned a host fetch"),
    }
}

#[test]
fn a_new_dialect_gets_every_shared_shape_by_implementing_the_trait() {
    let t = || relation(&["t"]);
    let one = |request: DataRequest| plain(&request).unwrap().join("\n;\n");
    assert_eq!(
        one(DataRequest::Preview {
            table: t(),
            columns: names(&["a", "b`c"]),
            limit: 10,
        }),
        "SELECT `a`, `b``c` FROM `t` LIMIT 10"
    );
    assert_eq!(
        one(DataRequest::Scan {
            table: t(),
            columns: names(&["a"]),
            filters: vec![
                Filter::NotNull { column: "a".into() },
                Filter::NotNull { column: "b".into() },
            ],
            limit: Some(5),
        }),
        "SELECT `a` FROM `t` WHERE `a` IS NOT NULL AND `b` IS NOT NULL LIMIT 5"
    );
    assert_eq!(
        one(DataRequest::CountRows {
            table: t(),
            filters: vec![],
        }),
        "SELECT COUNT(*) AS `rows` FROM `t`"
    );
    assert_eq!(
        one(DataRequest::ColumnRanges {
            table: t(),
            columns: names(&["a"]),
        }),
        "SELECT COUNT(*), MIN(`a`), MAX(`a`), COUNT(`a`)\nFROM `t`"
    );
    assert_eq!(
        one(DataRequest::RepeatedKey {
            table: t(),
            key: names(&["a", "b"]),
        }),
        "SELECT `a`, `b`\nFROM `t`\nWHERE `a` IS NOT NULL AND `b` IS NOT NULL\nGROUP BY `a`, `b`\nHAVING COUNT(*) > 1\nLIMIT 1"
    );
    assert_eq!(
        one(DataRequest::KeyCounts {
            keys: vec![
                TableKey {
                    unique: false,
                    table: t(),
                    columns: names(&["a"]),
                    label: None,
                },
                TableKey {
                    unique: false,
                    table: relation(&["u"]),
                    columns: names(&["a", "b"]),
                    label: Some(7),
                },
            ],
        }),
        "SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT `a`) AS d, COUNT(CASE WHEN `a` IS NULL THEN 1 END) AS z\nFROM `t`\nUNION ALL\nSELECT 7 AS k, COUNT(*) AS n, COUNT(DISTINCT (`a`, `b`)) FILTER (WHERE `a` IS NOT NULL AND `b` IS NOT NULL) AS d, COUNT(CASE WHEN `a` IS NULL OR `b` IS NULL THEN 1 END) AS z\nFROM `u`"
    );
    assert_eq!(
        one(DataRequest::TimeBounds {
            columns: vec![TableColumn {
                table: t(),
                column: "at".into(),
            }],
        }),
        "SELECT\n  0 as table_index,\n  MIN(`at`) as min_date,\n  MAX(`at`) as max_date\nFROM `t`"
    );
    assert_eq!(
        one(DataRequest::SampleEntities {
            rows: None,
            table: t(),
            key: names(&["a"]),
            n: 3,
            seed: Scalar::UInt64(42),
            columns: names(&["a"]),
            filters: vec![Filter::NotNull { column: "a".into() }],
            key_dtype: None,
        }),
        "SELECT `a`\nFROM `t`\nWHERE `a` IS NOT NULL\nLIMIT 3 -- 42"
    );
    let timed = |name: &str| {
        field(
            name,
            ColumnType::Timestamp {
                unit: Unit::Ns,
                zone: None,
            },
            false,
        )
    };
    let seeds = KeySet {
        attachment: sdm_contract::data::AttachmentId(1),
        columns: vec![
            field("key_1", ColumnType::Int64, true),
            field("key_2", ColumnType::Int64, true),
            field("batch", ColumnType::Int64, false),
            timed("end"),
        ],
    };
    let neighbors = DataRequest::Neighbors {
        table: t(),
        key: names(&["a", "b"]),
        columns: names(&["a", "at"]),
        seeds: seeds.clone(),
        window: Some(Window {
            source_type: None,
            column: "at".into(),
            lower: None,
            upper: Some(SeedBound {
                seed_column: "end".into(),
                inclusive: true,
            }),
        }),
        per_seed: None,
        order: None,
    };
    assert_eq!(
        one(neighbors),
        "SELECT s.`batch` AS `__kumo_batch__`, t.`a`, t.`at`\nFROM `seeds` s\nJOIN `t` t\n  ON t.`a` = s.`key_1` AND t.`b` = s.`key_2`\n AND t.`at` <= s.`end`"
    );
    assert_eq!(
        one(DataRequest::Neighbors {
            table: t(),
            key: names(&["a", "b"]),
            columns: names(&["a"]),
            seeds: seeds.clone(),
            window: None,
            per_seed: Some(2),
            order: Some(sdm_contract::data::TieBreak(vec![TieTerm {
                column: "at".into(),
                as_timestamp: true,
                descending: true,
                nulls_last: true,
            }])),
        }),
        "SELECT s.`batch` AS `__kumo_batch__`, t.`a`\nFROM `seeds` s\nJOIN `t` t\n  ON t.`a` = s.`key_1` AND t.`b` = s.`key_2`\nORDER BY t.`at` DESC NULLS LAST\nLIMIT 2"
    );
    assert_eq!(
        one(DataRequest::Lookup {
            table: t(),
            key: names(&["a", "b"]),
            columns: names(&["a"]),
            seeds,
        }),
        "SELECT s.`batch` AS `__kumo_batch__`, t.`a`\nFROM `seeds` s\nJOIN `t` t\n  ON t.`a` = s.`key_1` AND t.`b` = s.`key_2`"
    );
    assert_eq!(
        one(DataRequest::Describe {
            tables: vec![],
            part: DescribePart::Tables,
        }),
        "SHOW TABLES"
    );
}

#[test]
fn a_catalog_statement_keeps_the_parameters_its_dialect_binds() {
    let request = DataRequest::Describe {
        tables: vec![relation(&["Orders"])],
        part: DescribePart::Columns,
    };
    let FetchPlan::Sql { statements, .. } = sql::plan(&Plain, &request, &[], 0).unwrap() else {
        panic!("a dialect planned a host fetch");
    };
    let [statement] = statements.as_slice() else {
        panic!("one catalog statement, not {}", statements.len());
    };
    assert_eq!(
        statement.text,
        "SELECT column_name FROM information_schema.columns WHERE lower(table_name) = lower(?)"
    );
    assert_eq!(
        statement.params,
        vec![SqlParam::Value {
            value: Scalar::Utf8("Orders".to_owned()),
        }]
    );
    // A catalog statement binding nothing carries no parameter.
    let FetchPlan::Sql { statements, .. } = sql::plan(
        &Plain,
        &DataRequest::Describe {
            tables: vec![],
            part: DescribePart::Tables,
        },
        &[],
        0,
    )
    .unwrap() else {
        panic!("a dialect planned a host fetch");
    };
    assert!(statements.iter().all(|s| s.params.is_empty()));
}

#[test]
fn the_shared_builder_refuses_for_every_dialect_alike() {
    let field_of = |error: CoreError, name: &str| error.fields.get(name).cloned();
    // A relation past the dialect's depth, by the rule `relation`.
    let deep = plain(&DataRequest::Preview {
        table: relation(&["s", "t"]),
        columns: names(&["a"]),
        limit: 1,
    })
    .unwrap_err();
    assert_eq!(
        field_of(deep.clone(), "rule"),
        Some(Detail::text("relation"))
    );
    assert_eq!(
        field_of(deep, "most"),
        Some(Detail::Scalar(Scalar::UInt64(1)))
    );
    // A filter the renderings do not take, and a read of no columns.
    let filtered = plain(&DataRequest::Scan {
        table: relation(&["t"]),
        columns: names(&["a"]),
        filters: vec![Filter::TimeRange {
            column: "a".into(),
            lower: None,
            upper: None,
        }],
        limit: None,
    })
    .unwrap_err();
    assert_eq!(field_of(filtered, "field"), Some(Detail::text("filters")));
    let empty = plain(&DataRequest::Preview {
        table: relation(&["t"]),
        columns: vec![],
        limit: 1,
    })
    .unwrap_err();
    assert_eq!(field_of(empty, "field"), Some(Detail::text("columns")));
    // A sample's key outside its columns, before its seed is read.
    let key = plain(&DataRequest::SampleEntities {
        rows: None,
        table: relation(&["t"]),
        key: names(&["k"]),
        n: 1,
        seed: Scalar::Float64(1.5),
        columns: names(&["a"]),
        filters: vec![],
        key_dtype: None,
    })
    .unwrap_err();
    assert_eq!(field_of(key, "field"), Some(Detail::text("key")));
    // Each table of a describe is read through the dialect's depth first.
    let described = plain(&DataRequest::Describe {
        tables: vec![relation(&["s", "t"])],
        part: DescribePart::Columns,
    })
    .unwrap_err();
    assert_eq!(field_of(described, "field"), Some(Detail::text("tables")));
}
