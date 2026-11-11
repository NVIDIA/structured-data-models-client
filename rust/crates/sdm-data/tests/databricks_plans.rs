// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Databricks catalog binding and sampling contracts, without recorded SQL.

use sdm_contract::Scalar;
use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, FetchPlan, ProviderChoice, Relation, SqlParam,
    SqlStatement,
};
use sdm_data::Data;

fn relation(parts: &[&str]) -> Relation {
    Relation::of(parts.iter().map(|part| (*part).to_owned()).collect())
}

fn plan(request: &DataRequest) -> Vec<SqlStatement> {
    let provider = serde_json::from_value::<ProviderChoice>(serde_json::json!({
        "provider":"sql", "dialect":"databricks", "namespace":["main","shop"]
    }))
    .unwrap();
    let FetchPlan::Sql { statements, .. } = Data::new().plan(request, &provider).unwrap() else {
        panic!("SQL")
    };
    assert!(statements.iter().all(|s| s.dialect.name() == "databricks"));
    statements
}

#[test]
fn catalog_values_are_bound_and_foreign_key_roles_have_distinct_labels() {
    for part in [
        DescribePart::Tables,
        DescribePart::Columns,
        DescribePart::Keys,
        DescribePart::ForeignKeys,
    ] {
        let tables = if part == DescribePart::Tables {
            vec![]
        } else {
            vec![relation(&["main", "s'chema", "odd'table"])]
        };
        let statements = plan(&DataRequest::Describe { tables, part });
        let statement = statements.last().unwrap();
        let expected = if part == DescribePart::Tables {
            vec!["shop"]
        } else {
            vec!["s'chema", "odd'table"]
        };
        let bound: Vec<_> = statement
            .params
            .iter()
            .map(|p| match p {
                SqlParam::Value {
                    value: Scalar::Utf8(value),
                } => value.as_str(),
                other => panic!("unexpected catalog parameter: {other:?}"),
            })
            .collect();
        assert_eq!(bound, expected);
        assert!(!statement.text.contains("s'chema") && !statement.text.contains("odd'table"));
        assert!(statement.text.contains('`') && statement.text.contains('?'));
        if part == DescribePart::ForeignKeys {
            assert!(statement.text.contains("kcu.column_name AS source_column"));
            assert!(
                statement
                    .text
                    .contains("ccu.column_name AS referenced_column")
            );
        }
    }
}

#[test]
fn sampling_oversamples_then_caps_the_percentage_and_preserves_seed() {
    for (n, rows, percent, seed) in [
        (2, 100, "6.0", 0),
        (1, 1000, "0.30000000000000004", -1),
        (100, 10, "100.0", 2147483648),
    ] {
        let request = DataRequest::SampleEntities {
            rows: Some(rows),
            table: relation(&["main", "shop", "users"]),
            key: vec!["id".into()],
            columns: vec!["id".into()],
            n,
            seed: Scalar::Int64(seed),
            filters: vec![],
            key_dtype: None,
        };
        let statements = plan(&request);
        let text = &statements.last().unwrap().text;
        assert!(
            text.contains(&format!(
                "TABLESAMPLE ({percent} PERCENT) REPEATABLE ({seed})"
            )),
            "{text}"
        );
        assert!(text.ends_with(&format!("LIMIT {n}")), "{text}");
    }
}

#[test]
fn qualified_identifiers_preserve_each_part_and_escape_backticks() {
    let statements = plan(&DataRequest::Preview {
        table: relation(&["my-catalog", "sch.ema", "o`dd"]),
        columns: vec!["home.city".into(), "back`tick".into()],
        limit: 1000,
    });
    assert_eq!(
        statements.last().unwrap().text,
        "SELECT `home.city`, `back``tick` FROM `my-catalog`.`sch.ema`.`o``dd` LIMIT 1000"
    );
    for source in [vec![], vec!["a", "b", "c", "d"]] {
        let request = DataRequest::Preview {
            table: relation(&source),
            columns: vec!["x".into()],
            limit: 1,
        };
        assert!(
            Data::new()
                .plan(&request, &ProviderChoice::sql("databricks"))
                .is_err()
        );
    }
}

#[test]
fn keyed_reads_unpack_the_bound_json_parameter_by_position() {
    let cte = |columns: &str, element: &str| {
        format!(
            "WITH TMP AS (\n  SELECT CAST(pos AS BIGINT) AS `batch`, {columns}\n  FROM posexplode(from_json(?, 'array<{element}>'))\n)\n"
        )
    };
    let cases = [
        ("lookup", "int64", cte("col AS `key`", "bigint")),
        ("lookup", "float64", cte("col AS `key`", "double")),
        ("lookup", "utf8", cte("col AS `key`", "string")),
        (
            "neighbors",
            "int64",
            cte(
                "col.id AS `key`, col.e AS `end`",
                "struct<id:bigint,e:string>",
            ),
        ),
    ];
    for (kind, key_type, expected) in cases {
        let mut input = serde_json::json!({
            "request":kind,"table":{"source":["main","shop","users"]},
            "key":["id"],"columns":["id","at"],
            "seeds":{"attachment":7,"columns":[
                {"name":"key","nullable":false,"type":{"type":key_type}},
                {"name":"batch","nullable":false,"type":{"type":"int64"}}
            ]}
        });
        if kind == "neighbors" {
            input["seeds"]["columns"].as_array_mut().unwrap().push(serde_json::json!({
                "name":"end","nullable":false,"type":{"type":"timestamp","unit":"ns","zone":null}
            }));
            input["window"] = serde_json::json!({"column":"at","source_type":null,"lower":null,"upper":{"seed_column":"end","inclusive":true}});
            input["order"] = serde_json::Value::Null;
            input["per_seed"] = serde_json::Value::Null;
        }
        let request = serde_json::from_value(input).unwrap();
        let statements = plan(&request);
        let bound: Vec<_> = statements
            .iter()
            .filter(|s| {
                s.params
                    .iter()
                    .any(|p| matches!(p, SqlParam::KeySet { .. }))
            })
            .collect();
        assert_eq!(bound.len(), 1);
        let statement = bound[0];
        assert!(
            statement.text.starts_with(&expected),
            "{kind} {key_type}: {}",
            statement.text
        );
        let SqlParam::KeySet { keys, .. } = &statement.params[0] else {
            panic!("key parameter")
        };
        assert_eq!(keys.attachment.0, 7);
    }
}
