// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Databricks metric view's conversion (T3.5, M1 to M11) against what the
//! old client read and answered on a SQL warehouse (`contract/view-captures.json`,
//! from the recordings package T3.20, `contract/fixtures/views/captures.py`):
//! the core's `describe_source` over the view, each statement it plans
//! answered by the warehouse's recorded rows, in the old client's order, every
//! one asked, and the schema the old client gave.
//!
//! The statements are the old client's text, but for an expression's sample:
//! the old client read `SELECT {expr} AS name FROM {the table's rows}`, and
//! the core reads the expression inside the relation the data layer renders,
//! `SELECT name FROM (SELECT ({expr}) AS name FROM t AS source ...)`, the same
//! rows (register P3J-12); the test reads the core's as the old client's text
//! to find its recorded rows.

use std::sync::Arc;

use arrow_array::RecordBatch;
use base64::Engine;
use sdm_contract::data::{
    DataRequest, ExpectedSchema, FetchPlan, ProviderChoice, ProviderError, SqlParam,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{DescribeSource, Operation, OperationResult, SourceDescription};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, ColumnType, ErrorCode, HostFacts, Scalar, Step, Unit};
use sdm_core::{
    ArrowData, Attachment, Attachments, CompositionRoot, FetchResult, UnknownZone, Zones, start,
};
use serde_json::{Value as Json, json};

const CAPTURES: &str = include_str!("../../../../contract/view-captures.json");

struct Utc;

impl Zones for Utc {
    fn zone(&self, name: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        match name {
            "UTC" | "Etc/UTC" => Ok(jiff::tz::TimeZone::UTC),
            _ => Err(UnknownZone),
        }
    }
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_761_264_000_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    }
}

fn captures() -> Json {
    let document: Json = serde_json::from_str(CAPTURES).unwrap();
    assert_eq!(document["format"], "sdm-view-captures/1");
    document
}

fn rows(document: &Json, digest: &str) -> ArrowData {
    let encoded = document["tables"][digest].as_str().unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .unwrap();
    let reader =
        arrow_ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None).unwrap();
    let schema = reader.schema();
    let batches: Vec<RecordBatch> = reader.map(Result::unwrap).collect();
    ArrowData::new(schema, batches)
}

/// Rows the recording read as Python values (`fetchall`), whose integers
/// pyarrow types as 64-bit, in the type the request expects, as the
/// driver's Arrow gives them (a position, 32-bit).
fn as_expected(data: ArrowData, expect: &ExpectedSchema) -> ArrowData {
    let ExpectedSchema::Schema { fields } = expect else {
        return data;
    };
    let schema = Arc::new(arrow_schema::Schema::new(
        data.schema
            .fields()
            .iter()
            .zip(fields)
            .map(|(field, wanted)| {
                if wanted.column_type == ColumnType::Int32
                    && field.data_type() == &arrow_schema::DataType::Int64
                {
                    arrow_schema::Field::new(
                        field.name(),
                        arrow_schema::DataType::Int32,
                        field.is_nullable(),
                    )
                } else {
                    field.as_ref().clone()
                }
            })
            .collect::<Vec<_>>(),
    ));
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(schema.fields())
                .map(|(column, field)| {
                    if field.data_type() == &arrow_schema::DataType::Int32
                        && column.data_type() == &arrow_schema::DataType::Int64
                    {
                        let wide = column
                            .as_any()
                            .downcast_ref::<arrow_array::Int64Array>()
                            .unwrap();
                        Arc::new(arrow_array::Int32Array::from_iter(
                            wide.iter().map(|v| v.map(|v| i32::try_from(v).unwrap())),
                        )) as arrow_array::ArrayRef
                    } else {
                        Arc::clone(column)
                    }
                })
                .collect();
            RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
        })
        .collect();
    ArrowData::new(schema, batches)
}

/// `text` with each placeholder the recordings write for a private name
/// (`{catalog}`, `{schema}`, ...) as a plain name, which an unquoted part of
/// a name may hold, as the warehouse's own names did.
fn unplaced(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('{') {
        let (before, after) = rest.split_at(open);
        out.push_str(before);
        match after[1..].find('}') {
            Some(close)
                if after[1..=close]
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c == '_') =>
            {
                out.push_str("ph_");
                out.push_str(&after[1..=close]);
                rest = &after[close + 2..];
            }
            _ => {
                out.push('{');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// `data` with each text cell's placeholders as plain names ([`unplaced`]).
fn unplaced_rows(data: ArrowData) -> ArrowData {
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .map(
                    |column| match column.as_any().downcast_ref::<arrow_array::StringArray>() {
                        Some(text) => Arc::new(
                            text.iter()
                                .map(|v| v.map(unplaced))
                                .collect::<arrow_array::StringArray>(),
                        ) as arrow_array::ArrayRef,
                        None => Arc::clone(column),
                    },
                )
                .collect();
            RecordBatch::try_new(batch.schema(), columns).unwrap()
        })
        .collect();
    ArrowData::new(Arc::clone(&data.schema), batches)
}

/// `text` with every run of white space one space, as exact replay reads it.
fn normalised(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A statement's parameters as the recordings write them.
fn parameters(params: &[SqlParam]) -> Vec<Json> {
    params
        .iter()
        .map(|param| match param {
            SqlParam::Value {
                value: Scalar::Utf8(text),
            } => json!(text),
            SqlParam::Value { value } => json!(format!("{value:?}")),
            other => json!(format!("{other:?}")),
        })
        .collect()
}

/// The metric view the case reads, by the name its `DESCRIBE` names.
fn view_name(case: &Json) -> String {
    let first = unplaced(case["statements"][0]["text"].as_str().unwrap());
    let name = first.strip_prefix("DESCRIBE TABLE EXTENDED ").unwrap();
    name.split('.')
        .map(|part| part.trim_matches('`').replace("``", "`"))
        .collect::<Vec<_>>()
        .join(".")
}

/// The old client's text of an expression's sample the core asks as a
/// preview of a relation of expressions: the expressions over the rows the
/// old client read of the table, `from` (its last preview's `FROM`).
fn old_sample(request: &DataRequest, from: &str) -> Option<String> {
    let DataRequest::Preview { table, limit, .. } = request else {
        return None;
    };
    let renamed =
        |c: &sdm_contract::data::RelationColumn| c.column.as_ref().is_some_and(|s| *s != c.name);
    if table.columns.is_empty()
        || table
            .columns
            .iter()
            .any(|c| c.expression.is_none() && !renamed(c))
    {
        return None;
    }
    let projections: Vec<String> = table
        .columns
        .iter()
        .map(|c| {
            let read = match (&c.expression, &c.column) {
                (Some(expr), _) => expr.clone(),
                (None, Some(column)) => format!("`{}`", column.replace('`', "``")),
                (None, None) => String::new(),
            };
            format!("{read} AS `{}`", c.name.replace('`', "``"))
        })
        .collect();
    Some(format!(
        "SELECT {} FROM {from} LIMIT {limit}",
        projections.join(", ")
    ))
}

/// What a run over a case did: each recorded statement asked, in order, the
/// first statement it asked that the recording does not hold next, how it
/// ended, every request it asked and every warning it gave.
struct Replayed {
    asked: usize,
    stray: Option<String>,
    ended: Result<SourceDescription, sdm_contract::CoreError>,
    requests: Vec<DataRequest>,
    warnings: Vec<sdm_contract::CoreWarning>,
}

/// An answer of a test's own to the requests it names.
type Answer = Box<dyn Fn(&DataRequest) -> Option<FetchResult>>;

/// How a run departs from its recording: the view's definition rewritten
/// (`from`, `to`), and answers of its own for the requests it names, which
/// read no recorded statement.
#[derive(Default)]
struct Edits {
    definition: Option<(&'static str, &'static str)>,
    answer: Option<Answer>,
}

/// A provider's failure under `sqlstate`.
fn refusal(sqlstate: &str) -> FetchResult {
    FetchResult::Error(ProviderError {
        kind: sdm_contract::ErrorKind::DataSource,
        retryable: false,
        sqlstate: Some(sqlstate.to_owned()),
        vendor_code: None,
        error_class: None,
    })
}

fn replay(document: &Json, case: &Json) -> Replayed {
    replay_with(document, case, &Edits::default(), true)
}

/// The run of `case`, its statements answered by the recording: in order
/// where `strict`, else wherever the recording holds each.
fn replay_with(document: &Json, case: &Json, edits: &Edits, strict: bool) -> Replayed {
    replay_spec(document, case, edits, strict, json!({
        "tables": [], "primary_keys": {}, "time_columns": {}
    }))
}

fn replay_spec(document: &Json, case: &Json, edits: &Edits, strict: bool, spec: Json) -> Replayed {
    let trust = case["input"]["view"]["trust_expressions"]
        .as_bool()
        .unwrap_or(false);
    let provider: ProviderChoice = serde_json::from_value(json!({
        "provider": "sql", "dialect": "databricks",
        "view": {"name": view_name(case), "trust_expressions": trust},
    }))
    .unwrap();
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::reading_views(
        sdm_core::pql::spaces(),
    )));
    let mut run = start(
        &root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::DescribeSource(DescribeSource {
                provider,
                source: sdm_contract::data::AttachmentId(1),
                check_keys: false,
            }),
        },
        Attachments::from([(
            sdm_contract::data::AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&spec).unwrap()),
        )]),
        Arc::new(Utc),
    )
    .unwrap();
    let statements = case["statements"].as_array().unwrap();
    let mut at = 0;
    let mut requests = Vec::new();
    // The `FROM` of the last preview of each table's rows, by base table.
    let mut froms: Vec<(Vec<String>, String)> = Vec::new();
    let ended = |run: &mut sdm_core::Run, at, requests, stray, ended| Replayed {
        asked: at,
        stray,
        ended,
        requests,
        warnings: run.take_warnings(),
    };
    loop {
        let step = match run.next() {
            Ok(step) => step,
            Err(error) => return ended(&mut run, at, requests, None, Err(error)),
        };
        match step {
            Step::Done { result } => {
                let OperationResult::DescribeSource(described) = result else {
                    panic!("not a description");
                };
                let Some(Attachment::Bytes(bytes)) = run.take_attachment(described.description)
                else {
                    panic!("no description attached");
                };
                let description = serde_json::from_slice(&bytes).unwrap();
                return ended(&mut run, at, requests, None, Ok(description));
            }
            Step::Fetch {
                requests: asked_now,
            } => {
                for asked in asked_now {
                    requests.push(asked.request.clone());
                    if let Some(answer) = edits.answer.as_ref().and_then(|f| f(&asked.request)) {
                        run.submit_fetch(asked.id, answer).unwrap();
                        continue;
                    }
                    let FetchPlan::Sql {
                        statements: sql, ..
                    } = &asked.plan
                    else {
                        panic!("not SQL");
                    };
                    let last = sql.last().unwrap();
                    let mut text = normalised(&last.text);
                    if let DataRequest::Preview { table, .. } = &asked.request {
                        if let Some(from) = froms
                            .iter()
                            .rev()
                            .find(|(source, _)| *source == table.source)
                            .map(|(_, f)| f.clone())
                            && let Some(old) = old_sample(&asked.request, &from)
                        {
                            text = normalised(&old);
                        } else if old_sample(&asked.request, "").is_none() {
                            let from = text
                                .split_once(" FROM ")
                                .and_then(|(_, rest)| rest.rsplit_once(" LIMIT "))
                                .map(|(from, _)| from.to_owned())
                                .unwrap();
                            froms.push((table.source.clone(), from));
                        }
                    }
                    let holds = |r: &Json| {
                        normalised(&unplaced(r["text"].as_str().unwrap())) == text
                            && r["parameters"]
                                .as_array()
                                .cloned()
                                .unwrap_or_default()
                                .iter()
                                .map(|p| json!(unplaced(p.as_str().unwrap_or_default())))
                                .collect::<Vec<_>>()
                                == parameters(&last.params)
                    };
                    let recorded = if strict {
                        statements.get(at).filter(|r| holds(r))
                    } else {
                        statements.iter().find(|r| holds(r))
                    };
                    let Some(recorded) = recorded else {
                        let stray = format!(
                            "{text} {:?}, where the old client asked {}",
                            parameters(&last.params),
                            statements
                                .get(at)
                                .map_or("nothing".to_owned(), |r| r.to_string())
                        );
                        let error =
                            sdm_contract::CoreError::new(sdm_contract::Sentence::Internal, "stray");
                        return ended(&mut run, at, requests, Some(stray), Err(error));
                    };
                    let answer = match recorded["result"].as_str() {
                        Some(digest) => {
                            let mut data = unplaced_rows(rows(document, digest));
                            if let Some((from, to)) = edits.definition {
                                data = edited_rows(data, from, to);
                            }
                            FetchResult::Data(as_expected(data, asked.plan.expect()))
                        }
                        None => FetchResult::Error(ProviderError {
                            kind: sdm_contract::ErrorKind::Internal,
                            retryable: false,
                            sqlstate: None,
                            vendor_code: None,
                            error_class: None,
                        }),
                    };
                    at += 1;
                    run.submit_fetch(asked.id, answer).unwrap();
                }
            }
            Step::CallModel(_) => panic!("a description called a model"),
        }
    }
}

/// `data` with `from` replaced by `to` in each text cell.
fn edited_rows(data: ArrowData, from: &str, to: &str) -> ArrowData {
    let batches = data
        .batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .map(
                    |column| match column.as_any().downcast_ref::<arrow_array::StringArray>() {
                        Some(text) => Arc::new(
                            text.iter()
                                .map(|v| v.map(|v| v.replace(from, to)))
                                .collect::<arrow_array::StringArray>(),
                        ) as arrow_array::ArrayRef,
                        None => Arc::clone(column),
                    },
                )
                .collect();
            RecordBatch::try_new(batch.schema(), columns).unwrap()
        })
        .collect();
    ArrowData::new(Arc::clone(&data.schema), batches)
}

/// The recorded case `id`.
fn case<'d>(document: &'d Json, id: &str) -> &'d Json {
    document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["case"] == id && c["leg"] == "databricks-sql")
        .unwrap()
}

/// Whether `request` reads the base table `table` (its last name part).
fn reads(request: &DataRequest, table: &str) -> bool {
    request
        .relations()
        .iter()
        .any(|(_, relation)| relation.source.last().is_some_and(|t| t == table))
}

/// Whether `request` samples an expression.
fn samples(request: &DataRequest) -> bool {
    matches!(request, DataRequest::Preview { table, .. } if table.columns.iter().any(|c| c.expression.is_some()))
}

/// S5's first sequence: a filter the view cannot apply refuses it before
/// any joined table's columns are read (`RE/graph.py:1641-1646`).
#[test]
fn a_filter_is_refused_before_any_joined_table_is_read() {
    let document = captures();
    let edits = Edits {
        definition: Some((
            "filter: source.price > 50",
            "filter: my_udf(source.price) > 50",
        )),
        answer: None,
    };
    let replayed = replay_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    let Err(refused) = &replayed.ended else {
        panic!("a filter calling an untrusted function was applied");
    };
    assert_eq!(
        refused.sentence,
        sdm_contract::Sentence::ViewFilterRefusedUntrusted,
        "{refused:?}"
    );
    assert!(
        replayed.requests.iter().any(|r| reads(r, "orders")),
        "the source table's columns were not read: {:?}",
        replayed.requests
    );
    assert!(
        !replayed.requests.iter().any(|r| reads(r, "users")),
        "a joined table was read before the filter was refused: {:?}",
        replayed.requests
    );
}

/// S5's second sequence: a join dropped at its key's count drops the joins
/// nested in it before anything of theirs is sampled (`:1819-1825`).
#[test]
fn a_join_dropped_takes_its_children_before_their_samples() {
    let document = captures();
    let edits = Edits {
        definition: Some((
            "  - name: region\n    expr: users.countries.region",
            "  - name: region\n    expr: users.countries.region\n  - name: area\n    expr: UPPER(users.countries.region)",
        )),
        answer: Some(Box::new(|request: &DataRequest| match request {
            // `users`' key repeats, so its join is dropped.
            DataRequest::KeyCounts { keys } => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![
                    arrow_schema::Field::new("k", arrow_schema::DataType::Int32, false),
                    arrow_schema::Field::new("n", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("d", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("z", arrow_schema::DataType::Int64, false),
                ]));
                let n = keys.len();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(arrow_array::Int32Array::from_iter_values(0..n as i32)),
                        Arc::new(arrow_array::Int64Array::from(vec![10; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![5; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![0; n])),
                    ],
                )
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(schema, vec![batch])))
            }
            DataRequest::RepeatedKey { key, .. } => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    &key[0],
                    arrow_schema::DataType::Int64,
                    true,
                )]));
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(arrow_array::Int64Array::from(vec![1]))],
                )
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(schema, vec![batch])))
            }
            _ => None,
        })),
    };
    let replayed = replay_with(
        &document,
        case(&document, "view.metric.wildcards-nested-join"),
        &edits,
        false,
    );
    assert!(replayed.stray.is_none(), "{:?}", replayed.stray);
    let Ok(description) = &replayed.ended else {
        panic!("{:?}", replayed.ended.err());
    };
    let tables: Vec<&str> = description.tables.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(tables, ["orders"]);
    assert!(
        !replayed
            .requests
            .iter()
            .any(|r| samples(r) && reads(r, "countries")),
        "a dropped join's child was sampled: {:?}",
        replayed.requests
    );
    let said: Vec<&str> = replayed
        .warnings
        .iter()
        .map(|w| w.message.as_str())
        .collect();
    assert!(
        said.iter()
            .any(|w| w.starts_with("Failed to add join 'users'") && w.contains("is not unique")),
        "{said:?}"
    );
    assert!(
        said.iter()
            .any(|w| w.starts_with("Failed to add join 'countries'")
                && w.contains("'area'")
                && w.contains("its parent join 'users' was not added")),
        "{said:?}"
    );
}

/// S5's fourth sequence: an expression's sample the warehouse refuses with
/// `42501` (the caller may not read what it reads) leaves that column out,
/// retried alone first, as any class 22 or 42 refusal does (row 45).
#[test]
fn a_42501_on_an_expression_sample_leaves_that_column_out() {
    let document = captures();
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request: &DataRequest| {
            samples(request).then(|| refusal("42501"))
        })),
    };
    let replayed = replay_with(
        &document,
        case(&document, "view.metric.from-source"),
        &edits,
        false,
    );
    assert!(replayed.stray.is_none(), "{:?}", replayed.stray);
    let Ok(description) = &replayed.ended else {
        panic!("{:?}", replayed.ended.err());
    };
    let orders = description
        .tables
        .iter()
        .find(|t| t.name == "orders")
        .unwrap();
    assert!(
        !orders.columns.iter().any(|c| c.name == "price_band"),
        "{:?}",
        orders.columns
    );
    // Sampled with the batch, then alone.
    assert_eq!(replayed.requests.iter().filter(|r| samples(r)).count(), 2);
    let refused = replayed
        .warnings
        .iter()
        .find(|w| w.sentence == sdm_contract::Sentence::ViewColumnRefused)
        .expect("no warning of the column left out");
    assert_eq!(
        refused.message,
        "Failed to add column 'price_band' to table 'orders': SQLSTATE: 42501"
    );
}

/// The schema as the old client's observation writes it.
fn schema_json(description: &SourceDescription) -> Json {
    let tables: Vec<Json> = description
        .tables
        .iter()
        .map(|table| {
            json!({
                "name": table.name,
                "columns": table.columns.iter().map(|c| json!({
                    "name": c.name,
                    "dtype": c.dtype.wire_name(),
                    "stype": serde_json::to_value(c.stype).unwrap(),
                })).collect::<Vec<_>>(),
                "primary_key": table.primary_key.first(),
                "time_column": table.time_column,
            })
        })
        .collect();
    let links: Vec<Json> = description
        .links
        .iter()
        .map(|link| json!([link[0].table, link[0].column, link[1].table, link[1].column]))
        .collect();
    json!({"tables": tables, "links": links})
}

/// The old schema with the fields the core's description holds.
fn old_schema(schema: &Json) -> Json {
    let tables: Vec<Json> = schema["tables"]
        .as_array()
        .unwrap()
        .iter()
        .map(|table| {
            json!({
                "name": table["name"],
                "columns": table["columns"],
                "primary_key": table["primary_key"],
                "time_column": table["time_column"],
            })
        })
        .collect();
    json!({"tables": tables, "links": schema["links"]})
}

#[test]
fn each_metric_view_converts_as_the_old_client_converted_it_on_a_sql_warehouse() {
    let document = captures();
    let mut wrong = Vec::new();
    let mut converted = 0;
    for case in document["cases"].as_array().unwrap() {
        if case["leg"] != "databricks-sql" {
            continue;
        }
        let id = case["case"].as_str().unwrap();
        let replayed = replay(&document, case);
        let recorded = case["statements"].as_array().unwrap().len();
        if let Some(stray) = &replayed.stray {
            wrong.push(format!(
                "{id}: after {} statements, {stray}",
                replayed.asked
            ));
            continue;
        }
        if replayed.asked != recorded {
            wrong.push(format!(
                "{id}: asked {} of the {recorded} statements the old client asked",
                replayed.asked
            ));
        }
        match (&replayed.ended, case["outcome"]["result"].as_str()) {
            (Ok(description), Some("success")) => {
                converted += 1;
                let got = schema_json(description);
                let want = old_schema(&case["schema"]);
                if got != want {
                    wrong.push(format!("{id}: {got}\n  where the old client gave {want}"));
                }
            }
            (Err(error), Some("refusal")) => {
                if error.code != ErrorCode::InvalidRequest {
                    wrong.push(format!(
                        "{id}: refused with {:?}: {}",
                        error.code, error.message
                    ));
                }
            }
            (ended, outcome) => wrong.push(format!(
                "{id}: ended {:?} where the old client's ended {outcome:?}",
                ended.as_ref().map(|_| "described").map_err(|e| &e.message)
            )),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} differences:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    assert!(converted >= 14, "{converted} views converted");
}

/// The schema a host holds of `description`, each column with the facts
/// the core gave it.
fn held(description: &SourceDescription) -> sdm_contract::ops::HeldSchema {
    use sdm_contract::ops::{HeldColumn, HeldSchema, HeldTable, SourceFacts};
    HeldSchema {
        tables: description
            .tables
            .iter()
            .map(|table| HeldTable {
                name: table.name.clone(),
                source: table.source.clone(),
                columns: table
                    .columns
                    .iter()
                    .map(|c| HeldColumn {
                        name: c.name.clone(),
                        dtype: c.dtype.wire_name().into(),
                        stype: c.stype,
                        stype_before_key: c.stype_before_key,
                        source: Some(SourceFacts {
                            relation: table.source.clone(),
                            dtype: c.dtype,
                            received: c.received.clone(),
                            nullable: c.nullable,
                            unique: c.unique,
                            declared_type: c.declared_type.clone(),
                            exact_number: c.exact_number,
                        }),
                    })
                    .collect(),
                primary_key: table.primary_key.clone(),
                time_column: table.time_column.clone(),
                duplicate_keys: None,
            })
            .collect(),
        links: description.links.clone(),
    }
}

/// How a `materialize` over a view's schema read the view: the first
/// request past plan B's reads (the view read again), or how it ended
/// before.
enum Located {
    Past(DataRequest),
    Ended(sdm_contract::CoreError),
}

/// `materialize` of `held` over the view of `case`, its view read as the
/// recording (edited by `edits`) answers, to the first request past plan
/// B's reads.
fn materialize_over(
    document: &Json,
    case: &Json,
    held: &sdm_contract::ops::HeldSchema,
    edits: &Edits,
) -> Located {
    materialize_query(document, case, held, edits,
        "PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id")
}

fn materialize_query(
    document: &Json,
    case: &Json,
    held: &sdm_contract::ops::HeldSchema,
    edits: &Edits,
    query: &str,
) -> Located {
    use sdm_contract::data::AttachmentId;
    use sdm_contract::ops::Materialize;
    let provider: ProviderChoice = serde_json::from_value(json!({
        "provider": "sql", "dialect": "databricks",
        "view": {"name": view_name(case), "trust_expressions": false},
    }))
    .unwrap();
    let indices = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "indices",
            arrow_schema::DataType::Int64,
            false,
        )])),
        vec![Arc::new(arrow_array::Int64Array::from(vec![0, 1])) as arrow_array::ArrayRef],
    )
    .unwrap();
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::reading_views(
        sdm_core::pql::spaces(),
    )));
    let mut run = start(
        &root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation: Operation::Materialize(Materialize {
                provider,
                schema: AttachmentId(1),
                query: AttachmentId(2),
                indices: Some(AttachmentId(3)),
                anchor_time: Some(Scalar::Timestamp {
                    unit: Unit::Ns,
                    zone: None,
                    value: 1_709_251_200_000_000_000,
                }),
                seed: None,
                unseeded: false,
                anchor_entity: false,
                values: None,
                options: vec![],
                most_parts: None,
                most_entities: None,
                most_calls: None,
                candidates: None,
                sampler_state: None,
            }),
        },
        Attachments::from([
            (
                AttachmentId(1),
                Attachment::Bytes(serde_json::to_vec(held).unwrap()),
            ),
            (
                AttachmentId(2),
                Attachment::Bytes(
                    serde_json::to_vec(query)
                    .unwrap(),
                ),
            ),
            (
                AttachmentId(3),
                Attachment::Arrow(ArrowData::new(indices.schema(), vec![indices])),
            ),
        ]),
        Arc::new(Utc),
    )
    .unwrap();
    let statements = case["statements"].as_array().unwrap();
    loop {
        let step = match run.next() {
            Ok(step) => step,
            Err(error) => return Located::Ended(error),
        };
        let Step::Fetch { requests } = step else {
            panic!("materialize ended before it sampled: {step:?}");
        };
        for asked in requests {
            if asked.purpose.is_none() {
                return Located::Past(asked.request);
            }
            if let Some(answer) = edits.answer.as_ref().and_then(|f| f(&asked.request)) {
                run.submit_fetch(asked.id, answer).unwrap();
                continue;
            }
            let FetchPlan::Sql {
                statements: sql, ..
            } = &asked.plan
            else {
                panic!("not SQL");
            };
            let last = sql.last().unwrap();
            let text = normalised(&last.text);
            let recorded = statements
                .iter()
                .find(|r| {
                    normalised(&unplaced(r["text"].as_str().unwrap())) == text
                        && r["parameters"]
                            .as_array()
                            .cloned()
                            .unwrap_or_default()
                            .iter()
                            .map(|p| json!(unplaced(p.as_str().unwrap_or_default())))
                            .collect::<Vec<_>>()
                            == parameters(&last.params)
                })
                .or_else(|| {
                    // An expression's sample: its rows are the recorded sample's.
                    statements.iter().find(|r| {
                        samples(&asked.request)
                            && r["text"].as_str().unwrap().contains(" AS `price_band` ")
                    })
                })
                .unwrap_or_else(|| panic!("no recorded statement {text}"));
            let digest = recorded["result"].as_str().unwrap();
            let mut data = unplaced_rows(rows(document, digest));
            if let Some((from, to)) = edits.definition {
                data = edited_rows(data, from, to);
            }
            run.submit_fetch(
                asked.id,
                FetchResult::Data(as_expected(data, asked.plan.expect())),
            )
            .unwrap();
        }
    }
}

/// The schema `describe_source` gives of the view of `case`.
fn described(document: &Json, id: &str) -> SourceDescription {
    let replayed = replay(document, case(document, id));
    replayed.ended.unwrap()
}

/// Plan B: each `materialize` reads the view again, every probe included,
/// each table read as the relation the view gives it at this run: the
/// source table's rows under the view's filter, with its expressions.
#[test]
fn materialize_reads_each_table_as_the_view_gives_it_now() {
    let document = captures();
    let id = "view.metric.from-source";
    let schema = held(&described(&document, id));
    let Located::Past(request) =
        materialize_over(&document, case(&document, id), &schema, &Edits::default())
    else {
        panic!("materialize ended reading the view");
    };
    let orders = request
        .relations()
        .into_iter()
        .map(|(_, r)| r.clone())
        .find(|r| r.source.last().is_some_and(|t| t == "orders"));
    let orders = orders.unwrap_or_else(|| panic!("the first read is not of orders: {request:?}"));
    assert_eq!(orders.filter.as_deref(), Some("source.price > 50"));
    assert!(
        orders
            .columns
            .iter()
            .any(|c| c.name == "PRICE" && c.column.as_deref() == Some("price")),
        "{orders:?}"
    );
}

/// Plan B: a table the schema holds that the view no longer gives is
/// refused by name, with why.
#[test]
fn a_join_the_view_no_longer_gives_is_refused_by_name() {
    let document = captures();
    let id = "view.metric.from-source";
    let schema = held(&described(&document, id));
    // The join's key now repeats, so the view drops it at this run.
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request: &DataRequest| match request {
            DataRequest::KeyCounts { keys } => {
                let schema = Arc::new(arrow_schema::Schema::new(vec![
                    arrow_schema::Field::new("k", arrow_schema::DataType::Int32, false),
                    arrow_schema::Field::new("n", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("d", arrow_schema::DataType::Int64, false),
                    arrow_schema::Field::new("z", arrow_schema::DataType::Int64, false),
                ]));
                let n = keys.len();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(arrow_array::Int32Array::from_iter_values(0..n as i32)),
                        Arc::new(arrow_array::Int64Array::from(vec![10; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![0; n])),
                        Arc::new(arrow_array::Int64Array::from(vec![10; n])),
                    ],
                )
                .unwrap();
                Some(FetchResult::Data(ArrowData::new(schema, vec![batch])))
            }
            _ => None,
        })),
    };
    let Located::Ended(refused) = materialize_over(&document, case(&document, id), &schema, &edits)
    else {
        panic!("a table the view no longer gives was read");
    };
    assert_eq!(
        refused.sentence,
        sdm_contract::Sentence::ViewTableOmitted,
        "{refused:?}"
    );
    assert_eq!(
        refused.fields.get("reason"),
        Some(&sdm_contract::Detail::text(
            "its key 'user_id' holds no value in table 'users'"
        ))
    );
}

/// Plan B: a column the schema holds that the view no longer gives is
/// refused by name; an expression the view changed is read as it now
/// reads.
#[test]
fn a_dimension_removed_is_refused_and_one_changed_read_as_it_now_reads() {
    let document = captures();
    let id = "view.metric.from-source";
    let schema = held(&described(&document, id));
    let removed = Edits {
        definition: Some((
            "  - name: price_band\n    expr: CASE WHEN source.price > 100 THEN 'high' ELSE 'low' END\n",
            "",
        )),
        answer: None,
    };
    let Located::Ended(refused) =
        materialize_over(&document, case(&document, id), &schema, &removed)
    else {
        panic!("a column the view no longer gives was read");
    };
    assert_eq!(
        refused.sentence,
        sdm_contract::Sentence::ViewColumnOmitted,
        "{refused:?}"
    );
    assert_eq!(
        refused.fields.get("column"),
        Some(&sdm_contract::Detail::text("price_band"))
    );

    let changed = Edits {
        definition: Some(("source.price > 100 THEN", "source.price > 200 THEN")),
        answer: None,
    };
    let Located::Past(request) =
        materialize_over(&document, case(&document, id), &schema, &changed)
    else {
        panic!("materialize ended reading the view");
    };
    let expressions: Vec<Option<String>> = request
        .relations()
        .iter()
        .flat_map(|(_, relation)| relation.columns.iter())
        .filter(|column| column.name == "price_band")
        .map(|column| column.expression.clone())
        .collect();
    assert_eq!(
        expressions,
        [Some(
            "CASE WHEN price > 200 THEN 'high' ELSE 'low' END".to_owned()
        )],
        "{request:?}"
    );
}


#[test]
fn logical_table_aliases_and_swapped_names_survive_describe_and_materialize() {
    let document = captures();
    let recorded = case(&document, "view.metric.from-source");
    for (buyers, transactions) in [("buyers", "transactions"), ("orders", "users")] {
        let spec = json!({
            "tables": [
                {"name": buyers, "source": {"source": ["users"]}},
                {"name": transactions, "source": {"source": ["orders"]}}
            ],
            "primary_keys": {}, "time_columns": {}
        });
        let replayed = replay_spec(&document, recorded, &Edits::default(), true, spec);
        let description = replayed.ended.unwrap();
        assert_eq!(description.tables.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            [buyers, transactions]);
        assert_eq!(description.tables[0].source, sdm_contract::data::Relation::named("users"));
        assert_eq!(description.tables[0].primary_key, ["user_id"]);
        assert_eq!(description.tables[1].source, sdm_contract::data::Relation::named("orders"));
        assert_eq!(description.links[0][0].table, transactions);
        assert_eq!(description.links[0][1].table, buyers);
        let query = format!("PREDICT COUNT({transactions}.*, 0, 30, days) > 0 FOR EACH {buyers}.user_id");
        let Located::Past(request) = materialize_query(
            &document, recorded, &held(&description), &Edits::default(), &query,
        ) else { panic!("aliased view schema did not reach sampling"); };
        assert!(request.relations().iter().any(|(_, r)| r.source.last().is_some_and(|s| s == "orders")),
            "{request:?}");
    }
}

#[test]
fn a_views_caller_overrides_use_common_key_time_and_relationship_rules() {
    let document = captures();
    let recorded = case(&document, "view.metric.from-source");
    let spec = json!({
        "tables": [], "primary_keys": {"orders": ["order_id"]},
        "time_columns": {"orders": null}, "relationships": []
    });
    let description = replay_spec(&document, recorded, &Edits::default(), true, spec).ended.unwrap();
    let orders = description.tables.iter().find(|t| t.name == "orders").unwrap();
    assert_eq!(orders.primary_key, ["order_id"]);
    assert_eq!(orders.time_column, None);
    assert_eq!(orders.duplicate_keys, None);
    assert!(description.links.is_empty());
    assert_ne!(orders.columns.iter().find(|c| c.name == "user_id").unwrap().stype,
        sdm_contract::ops::Stype::Id);

    for (primary, time) in [(json!({"absent": ["id"]}), json!({})),
        (json!({"orders": ["absent"]}), json!({})),
        (json!({"orders": null}), json!({})),
        (json!({}), json!({"orders": "absent"}))] {
        let spec = json!({"tables": [], "primary_keys": primary, "time_columns": time});
        let error = replay_spec(&document, recorded, &Edits::default(), true, spec).ended.unwrap_err();
        assert_eq!(error.code, ErrorCode::InvalidRequest, "{error:?}");
    }
}

#[test]
fn view_refresh_checks_held_types_and_accepts_compatible_numeric_widths() {
    let document = captures();
    let recorded = case(&document, "view.metric.from-source");
    let original = held(&described(&document, "view.metric.from-source"));
    for dtype in ["string", "int32", "float32"] {
        let mut schema = original.clone();
        schema.tables.iter_mut().find(|t| t.name == "orders").unwrap()
            .columns.iter_mut().find(|c| c.name == "order_id").unwrap().dtype = dtype.into();
        let result = materialize_over(&document, recorded, &schema, &Edits::default());
        if dtype == "string" {
            let Located::Ended(error) = result else { panic!("incompatible type sampled"); };
            assert_eq!(error.sentence, sdm_contract::Sentence::ColumnTypeUnmatched);
            assert_eq!(error.fields.get("dtype"), Some(&sdm_contract::Detail::text("string")));
        } else {
            assert!(matches!(result, Located::Past(_)), "compatible numeric type refused");
        }
    }
    // The same logical dimension becomes numeric when the view's expression changes.
    let edits = Edits {
        definition: Some(("THEN 'high' ELSE 'low' END", "THEN 1 ELSE 0 END")),
        answer: Some(Box::new(|request| match request {
            DataRequest::Preview { columns, .. } if columns == &["price_band".to_owned()] => {
                let batch = RecordBatch::try_from_iter([("price_band",
                    Arc::new(arrow_array::Int64Array::from(vec![1, 0])) as arrow_array::ArrayRef)]).unwrap();
                Some(FetchResult::Data(ArrowData::new(batch.schema(), vec![batch])))
            }
            _ => None,
        })),
    };
    let Located::Ended(error) = materialize_over(&document, recorded, &original, &edits)
    else { panic!("changed incompatible dimension reached sampling"); };
    assert_eq!(error.sentence, sdm_contract::Sentence::ColumnTypeUnmatched);
    assert_eq!(error.fields.get("column"), Some(&sdm_contract::Detail::text("price_band")));
}

#[test]
fn a_refused_expression_keeps_the_warehouses_diagnosis() {
    let document = captures();
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request| samples(request).then(|| FetchResult::Error(
            serde_json::from_value(json!({"kind": "data_source", "retryable": false,
                "sqlstate": "22018", "driver_message": "Cannot cast price 'unknown' to DOUBLE"})).unwrap()
        )))),
    };
    let replayed = replay_with(&document, case(&document, "view.metric.from-source"), &edits, false);
    assert!(replayed.ended.is_ok(), "{:?}", replayed.ended);
    assert!(replayed.warnings.iter().any(|w| w.fields.values().any(|v|
        v == &sdm_contract::Detail::text("Cannot cast price 'unknown' to DOUBLE"))),
        "{:?}", replayed.warnings);
}

#[test]
fn a_failed_duplicate_example_keeps_joins_unchecked() {
    let document = captures();
    let edits = Edits {
        definition: None,
        answer: Some(Box::new(|request| match request {
            DataRequest::KeyCounts { keys } => {
                let n = keys.len();
                let batch = RecordBatch::try_from_iter([
                    ("k", Arc::new(arrow_array::Int32Array::from_iter_values(0..n as i32)) as arrow_array::ArrayRef),
                    ("n", Arc::new(arrow_array::Int64Array::from(vec![10; n])) as arrow_array::ArrayRef),
                    ("d", Arc::new(arrow_array::Int64Array::from(vec![9; n])) as arrow_array::ArrayRef),
                    ("z", Arc::new(arrow_array::Int64Array::from(vec![0; n])) as arrow_array::ArrayRef),
                ]).unwrap();
                Some(FetchResult::Data(ArrowData::new(batch.schema(), vec![batch])))
            }
            DataRequest::RepeatedKey { .. } => Some(refusal("42501")),
            _ => None,
        })),
    };
    let replayed = replay_with(&document, case(&document, "view.metric.from-source"), &edits, false);
    let description = replayed.ended.unwrap();
    assert_eq!(description.links.len(), 1);
    assert!(replayed.warnings.iter().any(|w| w.sentence == sdm_contract::Sentence::ViewJoinUnchecked));
}
