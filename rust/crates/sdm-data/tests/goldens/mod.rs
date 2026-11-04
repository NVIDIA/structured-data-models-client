// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! L4's two checks, shared by each database's goldens (T2.4; design,
//! locked check 3): `duckdb_goldens.rs`, `sqlite_goldens.rs` and
//! `postgres_goldens.rs` each pin, verbatim, every statement its dialect
//! renders for the contract cases, beside the request it answers.
//!
//! - [`render`]: each golden request, planned here, renders exactly its
//!   statements, so a change of the SQL without a golden update fails.
//! - [`run_is_pinned`]: every fetch a case run made on the database (the
//!   observations `SDM_L4_OBSERVATIONS` names, python-core's run in
//!   `contract-python (ref-pandas3, <database>)`, which executed each
//!   statement on that database and whose results the comparator held to
//!   the reference's) is a golden, its statements the golden's, so the
//!   goldens are the SQL the run executed and a request the dialect newly
//!   answers is never left unreviewed. CI's `l4-goldens (<database>)` names
//!   the run; with no run named, as in `cargo test --workspace`, there is
//!   nothing to hold and the check says so. A run named but missing,
//!   unreadable, empty, or holding no fetch on the database fails.
//! - [`seeded_faults`]: both checks shown to fail on their seeded faults.
//!
//! Coverage ignores only resolved identity projections and non-rendering type
//! decorations absent from old requests. SQLite scratch identifiers may be
//! consistently renamed across a statement sequence; literal values and every
//! other SQL byte remain pinned. PostgreSQL string-order flags remain part of
//! request identity, with one explicitly reviewed text-cast variant below.
//! A failure prints each missing golden ready to review, as the run made it.

use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, FetchPlan, ProviderChoice,
};
use sdm_data::Data;
use serde_json::Value;

/// Each request, as the contract writes it, and its statements in order.
pub type Goldens = &'static [(&'static str, &'static [&'static str])];

type Owned = Vec<(String, Vec<String>)>;

fn owned(database: &str, goldens: Goldens) -> Owned {
    let mut result: Owned = goldens
        .iter()
        .map(|(request, statements)| {
            (
                (*request).to_owned(),
                statements.iter().map(|s| (*s).to_owned()).collect(),
            )
        })
        .collect();
    if database == "postgres" {
        // Reviewed current string ordering for the one AdventureWorks request.
        // Keep the historical request/SQL, and derive only these eight casts
        // from its independently pinned text, never from the current renderer.
        let columns = [
            "AccountNumber",
            "CreditCardApprovalCode",
            "PurchaseOrderNumber",
            "SalesOrderNumber",
        ];
        let variants: Vec<_> = result
            .iter()
            .filter_map(|(request, statements)| {
                let mut request: Value = serde_json::from_str(request).unwrap();
                if request["request"] != "neighbors"
                    || !request["order"].is_array()
                    || request["window"]["source_type"] != "timestamp without time zone"
                    || request["table"]["source"]
                        != serde_json::json!(["public", "adventureworks_sales_order_headers"])
                {
                    return None;
                }
                for term in request["order"].as_array_mut().unwrap() {
                    if columns.iter().any(|column| term["column"] == *column) {
                        term["string_like"] = Value::Bool(true);
                    }
                }
                let statements = statements
                    .iter()
                    .map(|statement| {
                        let mut text = statement.clone();
                        for column in columns {
                            let before = format!("t.\"{column}\" ASC");
                            assert_eq!(text.matches(&before).count(), 2);
                            text =
                                text.replace(&before, &format!("CAST(t.\"{column}\" AS text) ASC"));
                        }
                        text
                    })
                    .collect();
                Some((request.to_string(), statements))
            })
            .collect();
        assert_eq!(variants.len(), 1, "one reviewed string-order variant");
        result.extend(variants);
    }
    result
}

/// A golden as the file writes it.
fn literal(request: &str, statements: &[String]) -> String {
    let texts: Vec<String> = statements.iter().map(|s| format!("{s:?}")).collect();
    format!("(r##\"{request}\"##, &[{}]),", texts.join(", "))
}

/// The namespace the contract cases' runs on `dialect` plan `request`
/// under: on Postgres the schema each case's database settles, `public`,
/// its session's current one, which the core reads once per run (the read
/// itself planned before, under none) and names in every statement (T3.7);
/// none for the others.
fn settled(dialect: &str, request: &DataRequest) -> Vec<Option<String>> {
    let session = matches!(
        request,
        DataRequest::Describe {
            part: DescribePart::Namespace,
            ..
        }
    );
    if dialect == "postgres" && !session {
        vec![Some("public".to_owned())]
    } else {
        Vec::new()
    }
}

/// The goldens whose request does not render exactly its statements.
fn misrendered(dialect: &str, goldens: &Owned) -> Vec<String> {
    let mut problems = Vec::new();
    for (request, statements) in goldens {
        let parsed: DataRequest = match serde_json::from_str(request) {
            Ok(parsed) => parsed,
            Err(error) => {
                problems.push(format!("{request}: not a request: {error}"));
                continue;
            }
        };
        let choice = ProviderChoice::Sql {
            timestamp_encoding: None,
            dialect: Dialect::new(dialect),
            namespace: settled(dialect, &parsed),
            view: None,
        };
        match Data::new().plan(&parsed, &choice) {
            Ok(FetchPlan::Sql {
                statements: planned,
                ..
            }) => {
                let texts: Vec<String> = planned.iter().map(|s| s.text.clone()).collect();
                if &texts != statements {
                    problems.push(format!("renders\n  {}", literal(request, &texts)));
                }
            }
            Ok(_) => problems.push(format!("{request}: not planned as SQL")),
            Err(error) => problems.push(format!("{request}: refused: {error:?}")),
        }
    }
    problems
}

/// Every golden request renders exactly its statements on `dialect`.
pub fn render(dialect: &str, goldens: Goldens) {
    assert!(!goldens.is_empty(), "no goldens on {dialect}");
    let problems = misrendered(dialect, &owned(dialect, goldens));
    assert!(
        problems.is_empty(),
        "{dialect}: {} golden(s) not what the dialect renders:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// Whether an observation's case key is on `database`: DuckDB's keep the
/// bare case id, the others are `<case>@<database>`.
fn on(database: &str, case: &str) -> bool {
    if database == "duckdb" {
        !case.contains('@')
    } else {
        case.strip_suffix(database)
            .is_some_and(|rest| rest.ends_with('@'))
    }
}

/// Coverage identity omits only resolved decorations absent from old requests.
/// SQL still compares exactly (apart from opaque SQLite scratch identifiers),
/// so source_type/string_like changes that affect rendering need a reviewed pin.
fn coverage_request(value: &mut Value, database: &str) {
    if !matches!(database, "duckdb" | "sqlite" | "postgres") {
        return;
    }
    match value {
        Value::Array(items) => {
            for item in items {
                coverage_request(item, database);
            }
        }
        Value::Object(fields) => {
            for item in fields.values_mut() {
                coverage_request(item, database);
            }
            if fields.contains_key("source") {
                if fields.get("found") == Some(&Value::Bool(true)) {
                    fields.remove("found");
                }
                if fields
                    .get("columns")
                    .and_then(Value::as_array)
                    .is_some_and(|columns| {
                        columns.iter().all(|column| {
                            column.as_object().is_some_and(|fields| {
                                fields.len() == 2
                                    && fields.contains_key("column")
                                    && fields.get("column") == fields.get("name")
                            })
                        })
                    })
                {
                    fields.remove("columns");
                }
                if database == "postgres"
                    && fields["source"]
                        .as_array()
                        .is_some_and(|parts| parts.len() == 2 && parts[0] == "public")
                {
                    fields["source"] = serde_json::json!([fields["source"][1]]);
                }
            }
            if fields.contains_key("column") && fields.contains_key("upper") {
                fields.remove("source_type");
            }
            if fields.contains_key("as_timestamp")
                && (database != "postgres" || fields.get("string_like") != Some(&Value::Bool(true)))
            {
                fields.remove("string_like");
            }
        }
        _ => {}
    }
}

/// Scratch identity is a bijection over the entire statement sequence.
/// Never rewrite quoted string values, comments, or ordinary identifiers.
fn coverage_statements(database: &str, statements: &[String]) -> Vec<String> {
    if database != "sqlite" {
        return statements.to_vec();
    }
    let token = fancy_regex::Regex::new(r#"'(?:[^']|'')*'|--[^\n]*|/\*[\s\S]*?\*/|(?P<scratch>"__sdm_seeds_[0-9a-f]{16}__")(?!")|"(?:[^"]|"")*""#).unwrap();
    let mut names = std::collections::BTreeMap::new();
    statements
        .iter()
        .map(|statement| {
            token
                .replace_all(statement, |capture: &fancy_regex::Captures<'_, str>| {
                    let text = capture.get(0).unwrap().as_str();
                    if capture.name("scratch").is_none() {
                        return text.to_owned();
                    }
                    let next = names.len();
                    let number = names.entry(text.to_owned()).or_insert(next);
                    format!("\"__sdm_seeds_digest_{number}__\"")
                })
                .into_owned()
        })
        .collect()
}

/// The fetches of `observations` (one JSON object per line) on `database`
/// that no golden holds with their statements, each ready to paste; an
/// error where the run is unreadable or holds no fetch on the database.
fn unpinned(database: &str, observations: &str, goldens: &Owned) -> Result<Vec<String>, String> {
    let mut pinned: Vec<(Value, Vec<String>)> = Vec::new();
    for (request, statements) in goldens {
        let mut value = serde_json::from_str(request)
            .map_err(|error| format!("a golden is not JSON: {error}: {request}"))?;
        coverage_request(&mut value, database);
        pinned.push((value, coverage_statements(database, statements)));
    }
    let (mut cases, mut fetches) = (0, 0);
    let mut problems: Vec<String> = Vec::new();
    for (number, line) in observations.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let observation: Value = serde_json::from_str(line)
            .map_err(|error| format!("line {}: not JSON: {error}", number + 1))?;
        let Some(case) = observation["case"].as_str() else {
            return Err(format!("line {}: names no case", number + 1));
        };
        if !on(database, case) {
            continue;
        }
        cases += 1;
        let Some(made) = observation["core"]["fetches"].as_array() else {
            continue;
        };
        for fetch in made {
            fetches += 1;
            let request = &fetch["request"];
            let Some(planned) = fetch["plan"]["statements"].as_array() else {
                problems.push(format!("{case}: a fetch planned no SQL: {request}"));
                continue;
            };
            let mut texts = Vec::new();
            for statement in planned {
                if statement["dialect"].as_str() != Some(database) {
                    problems.push(format!(
                        "{case}: a statement of dialect {}, not {database}",
                        statement["dialect"]
                    ));
                }
                texts.push(statement["text"].as_str().unwrap_or_default().to_owned());
            }
            // Deserialization rejects unknown fields and malformed decorations.
            serde_json::from_value::<DataRequest>(request.clone())
                .map_err(|error| format!("{case}: invalid observed request: {error}"))?;
            let mut identity = request.clone();
            coverage_request(&mut identity, database);
            let normalized = coverage_statements(database, &texts);
            if !pinned
                .iter()
                .any(|(r, s)| *r == identity && *s == normalized)
            {
                let entry = literal(&request.to_string(), &texts);
                if !problems.contains(&entry) {
                    problems.push(entry);
                }
            }
        }
    }
    if cases == 0 {
        return Err(format!("the run holds no observation on {database}"));
    }
    if fetches == 0 {
        return Err(format!(
            "the run's {cases} observation(s) on {database} hold no fetch"
        ));
    }
    Ok(problems)
}

/// The run `SDM_L4_OBSERVATIONS` names, held to the goldens.
#[allow(
    clippy::disallowed_methods,
    reason = "a test reading the run CI names, not the core: L4's evidence"
)]
pub fn run_is_pinned(database: &str, goldens: Goldens) {
    let Some(path) = std::env::var_os("SDM_L4_OBSERVATIONS").filter(|p| !p.is_empty()) else {
        eprintln!("{database}: SDM_L4_OBSERVATIONS names no run: nothing to hold the goldens to");
        return;
    };
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: unreadable: {error}", path.to_string_lossy()));
    match unpinned(database, &text, &owned(database, goldens)) {
        Ok(problems) => assert!(
            problems.is_empty(),
            "{database}: {} fetch(es) the run made are no golden; each as the run made it:\n{}",
            problems.len(),
            problems.join("\n")
        ),
        Err(error) => panic!("{database}: {error}"),
    }
}

/// A run of one case on `database` making each golden's fetch.
fn run_of(database: &str, goldens: &Owned) -> String {
    let case = if database == "duckdb" {
        "a.case".to_owned()
    } else {
        format!("a.case@{database}")
    };
    let fetches: Vec<Value> = goldens
        .iter()
        .map(|(request, statements)| {
            serde_json::json!({
                "request": serde_json::from_str::<Value>(request).unwrap(),
                "plan": {
                    "plan": "sql",
                    "statements": statements
                        .iter()
                        .map(|text| serde_json::json!({"dialect": database, "params": [], "text": text}))
                        .collect::<Vec<_>>(),
                },
            })
        })
        .collect();
    serde_json::json!({"case": case, "core": {"fetches": fetches}}).to_string()
}

/// Both checks fail on their seeded faults, on `goldens`.
pub fn seeded_faults(database: &str, goldens: Goldens) {
    let goldens = owned(database, goldens);
    // Controls: the goldens render, and a run of every golden is pinned.
    assert!(misrendered(database, &goldens).is_empty());
    let run = run_of(database, &goldens);
    assert_eq!(unpinned(database, &run, &goldens), Ok(Vec::new()));

    // A seeded SQL change without a golden update: the golden's text no
    // longer what the dialect renders, and a run of the changed SQL no
    // golden.
    let mut changed = goldens.clone();
    changed[0].1[0].push(' ');
    assert_eq!(misrendered(database, &changed).len(), 1);
    let ran = run_of(database, &changed);
    assert_eq!(unpinned(database, &ran, &goldens).map(|p| p.len()), Ok(1));

    if database == "postgres" {
        // A typed string-order request cannot pass using the older uncast SQL.
        let mut wrong = goldens.last().unwrap().clone();
        for statement in &mut wrong.1 {
            *statement =
                statement.replace("CAST(t.\"AccountNumber\" AS text)", "t.\"AccountNumber\"");
        }
        assert_eq!(
            unpinned(database, &run_of(database, &vec![wrong]), &goldens).map(|p| p.len()),
            Ok(1)
        );
    }

    // A request no golden holds.
    let fewer = goldens[1..].to_vec();
    assert_eq!(unpinned(database, &run, &fewer).map(|p| p.len()), Ok(1));

    // A run that holds nothing to judge fails, never passes: empty, of
    // another database, with no fetch, or not JSON.
    assert!(unpinned(database, "", &goldens).is_err());
    let other = if database == "duckdb" {
        "sqlite"
    } else {
        "duckdb"
    };
    assert!(unpinned(database, &run_of(other, &goldens), &goldens).is_err());
    assert!(unpinned(database, &run_of(database, &Vec::new()), &goldens).is_err());
    assert!(unpinned(database, "{", &goldens).is_err());
}

#[test]
fn resolved_request_metadata_does_not_hide_changed_relations_or_queries() {
    let request = serde_json::json!({
        "request": "lookup", "table": {"source": ["orders"]},
        "columns": ["id"], "key": ["id"],
        "seeds": {"attachment": 1, "columns": [
            {"name": "key", "nullable": false, "type": {"type": "int64"}},
            {"name": "batch", "nullable": false, "type": {"type": "int64"}}
        ]}
    });
    let sql = vec!["SELECT id FROM orders".to_owned()];
    let pins = vec![(request.to_string(), sql.clone())];
    for database in ["duckdb", "sqlite", "postgres"] {
        let mut resolved = request.clone();
        resolved["table"]["found"] = Value::Bool(true);
        resolved["table"]["columns"] = serde_json::json!([{"name": "id", "column": "id"}]);
        if database == "postgres" {
            resolved["table"]["source"] = serde_json::json!(["public", "orders"]);
        }
        let run = |request: &Value, sql: Vec<String>| {
            unpinned(
                database,
                &run_of(database, &vec![(request.to_string(), sql)]),
                &pins,
            )
        };
        assert_eq!(run(&resolved, sql.clone()), Ok(Vec::new()));
        for table in [
            serde_json::json!({"source": ["other", "orders"]}),
            serde_json::json!({"source": ["orders"], "found": false}),
            serde_json::json!({"source": ["orders"], "columns": [{"name": "id", "column": "other"}]}),
            serde_json::json!({"query": "SELECT other AS id FROM orders"}),
        ] {
            let mut changed = resolved.clone();
            changed["table"] = table;
            assert!(run(&changed, sql.clone()).map_or(true, |problems| !problems.is_empty()));
        }
        let mut changed = resolved.clone();
        changed["key"] = serde_json::json!(["other"]);
        assert!(run(&changed, sql.clone()).map_or(true, |problems| !problems.is_empty()));
        assert!(
            run(&resolved, vec!["SELECT other FROM orders".to_owned()])
                .map_or(true, |problems| !problems.is_empty())
        );
    }
}

#[test]
fn scratch_identity_preserves_sequence_links_literals_and_comments() {
    let a = "\"__sdm_seeds_0123456789abcdef__\"";
    let b = "\"__sdm_seeds_fedcba9876543210__\"";
    let c = "\"__sdm_seeds_1123456789abcdef__\"";
    let sequence = |first, second| {
        vec![
            format!("CREATE TABLE {first}(id INTEGER)"),
            format!("SELECT id FROM {second}"),
        ]
    };
    let canonical = |statements: Vec<String>| coverage_statements("sqlite", &statements);
    assert_eq!(canonical(sequence(a, a)), canonical(sequence(b, b)));
    assert_ne!(canonical(sequence(a, a)), canonical(sequence(b, c)));
    for (prefix, suffix) in [
        ("SELECT '", "'"),
        ("-- ", "\nSELECT 1"),
        ("/* ", " */ SELECT 1"),
    ] {
        assert_ne!(
            canonical(vec![format!("{prefix}{a}{suffix}")]),
            canonical(vec![format!("{prefix}{b}{suffix}")])
        );
    }
    assert_ne!(
        canonical(sequence(a, a)),
        canonical(vec![
            format!("CREATE TABLE {b}(id INTEGER)"),
            format!("SELECT id FROM main.{b}")
        ])
    );
}

#[test]
fn cloud_request_type_metadata_remains_exact() {
    let request = serde_json::json!({
        "window": {"column": "at", "upper": null, "source_type": "DATE"},
        "table": {"source": ["orders"], "found": true,
                  "columns": [{"name": "at", "column": "at"}]},
        "order": [{"column": "at", "as_timestamp": false, "string_like": true}]
    });
    for database in ["databricks", "snowflake"] {
        let mut actual = request.clone();
        coverage_request(&mut actual, database);
        assert_eq!(actual, request);
        actual["window"]["source_type"] = Value::String("TIMESTAMP".to_owned());
        coverage_request(&mut actual, database);
        assert_ne!(actual, request);
    }
}
