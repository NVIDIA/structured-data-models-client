// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dispatch, the composition root, profiles and model capability checks.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, DescribePart, Determinism, Dialect, ExpectedSchema,
    FetchPlan, OptionalRequest, ProviderCapabilities, ProviderChoice, SqlStatement,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    Capability, DescribeSource, Materialize, NamedTable, Operation, OperationResult, ProfileName,
    ProfileSpec, QuerySchema, Seed, SourceDescription, SourceSpec, Span, ValidateQuery,
};
use sdm_contract::run::StartRun;
use sdm_contract::{
    CONTRACT_VERSION, CoreError, ErrorCode, ErrorKind, HostFacts, Scalar, Step, Unit,
};
use sdm_core::profiles::{Mode, Profile, QueryFeatures, check_capabilities, codes as model};
use sdm_core::{
    Attachment, Attachments, CompositionRoot, RunState, UnknownZone, Zones,
    start as start_with_zones,
};

/// Zones that hold nothing: these runs resolve none.
struct NoZones;

impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
}

fn start(
    root: &CompositionRoot,
    start: StartRun,
    attachments: Attachments,
) -> Result<sdm_core::Run, CoreError> {
    start_with_zones(root, start, attachments, Arc::new(NoZones))
}

/// The root a binding gets, over `sdm-data`'s data layer.
fn root() -> CompositionRoot {
    CompositionRoot::new(Arc::new(sdm_data::Data::new()))
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 0,
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

/// A bytes attachment of `value` as JSON.
fn json(value: &impl serde::Serialize) -> Attachment {
    Attachment::Bytes(serde_json::to_vec(value).unwrap())
}

/// The attachments the operations below name.
fn attachments_of(operation: &Operation) -> Attachments {
    match operation {
        Operation::ValidateQuery(v) => Attachments::from([
            (v.query, json(&"PREDICT t.c FOR EACH t.id")),
            (
                v.schema,
                json(&QuerySchema {
                    tables: vec![],
                    links: vec![],
                }),
            ),
        ]),
        Operation::DescribeSource(d) => Attachments::from([(
            d.source,
            json(&SourceSpec {
                tables: vec![NamedTable {
                    name: "t".into(),
                    source: "t".into(),
                }],
                primary_keys: BTreeMap::new(),
                time_columns: BTreeMap::new(),
                relationships: None,
            }),
        )]),
        Operation::Materialize(m) => Attachments::from([
            (
                m.schema,
                json(&SourceDescription {
                    tables: vec![],
                    links: vec![],
                }),
            ),
            (m.query, json(&"PREDICT t.c FOR EACH t.id")),
        ]),
        other => panic!("no attachments for {}", other.name()),
    }
}

fn validate() -> Operation {
    Operation::ValidateQuery(ValidateQuery {
        query: AttachmentId(1),
        schema: AttachmentId(2),
        profile: ProfileSpec {
            name: ProfileName::Relational,
            model: None,
            capabilities: BTreeMap::new(),
        },
    })
}

fn describe(provider: ProviderChoice) -> Operation {
    Operation::DescribeSource(DescribeSource {
        provider,
        source: AttachmentId(7),
        check_keys: false,
    })
}

fn run_of(root: &CompositionRoot, operation: Operation) -> Result<sdm_core::Run, CoreError> {
    let attachments = attachments_of(&operation);
    start(
        root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        attachments,
    )
}

#[test]
fn another_contract_version_is_refused_before_a_run_exists() {
    let err = start(
        &root(),
        StartRun {
            contract: CONTRACT_VERSION + 1,
            facts: facts(),
            operation: validate(),
        },
        attachments_of(&validate()),
    )
    .err()
    .unwrap();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Protocol, ErrorCode::InvalidMessage)
    );
}

#[test]
fn a_clock_that_is_not_a_timestamp_in_utc_is_refused() {
    let at = |zone: Option<&str>| Scalar::Timestamp {
        unit: Unit::Us,
        zone: zone.map(str::to_owned),
        value: 0,
    };
    for now in [
        Scalar::Int64(0),
        at(None),
        at(Some("Europe/Berlin")),
        at(Some("utc")),
        at(Some("Etc/UTC")),
    ] {
        let mut bad = facts();
        bad.now = now.clone();
        let err = start(
            &root(),
            StartRun {
                contract: CONTRACT_VERSION,
                facts: bad,
                operation: validate(),
            },
            attachments_of(&validate()),
        )
        .err()
        .unwrap();
        assert_eq!(err.code, ErrorCode::InvalidRequest, "{now:?}");
    }
}

#[test]
fn every_operation_dispatches_to_a_run() {
    let root = root();
    let mut run = run_of(&root, validate()).unwrap();
    assert_eq!(run.state(), RunState::Ready);
    // The schema has no table t, which the parser behind the adapter says.
    let Step::Done {
        result: OperationResult::ValidateQuery(validation),
    } = run.next().unwrap()
    else {
        panic!("validate_query did not finish with its result");
    };
    assert!(!validation.accepted);
    assert_eq!(validation.diagnostics[0].code, "PQL_SCHEMA_UNKNOWN_TABLE");
    assert_eq!(run.state(), RunState::Finished);

    let host = ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            derived_relations: false,
            neighbor_lookback: false,
            sample_rows: false,
            optional: BTreeSet::new(),
            max_keys_per_request: None,
            max_seed: u64::MAX,
            text_times: None,
            null_order: None,
            index_advice: None,
            sampling: Determinism::Unseeded,
            fallback: None,
        },
    };
    // A table named is read by its source name, with no listing of the
    // source's tables first (T1.23): its columns and keys in one step.
    let Step::Fetch { requests } = run_of(&root, describe(host)).unwrap().next().unwrap() else {
        panic!("describe_source did not ask for the table named");
    };
    let asked: Vec<&DataRequest> = requests.iter().map(|step| &step.request).collect();
    assert_eq!(
        asked,
        [
            &DataRequest::Describe {
                tables: vec!["t".into()],
                part: DescribePart::Columns,
            },
            &DataRequest::Describe {
                tables: vec!["t".into()],
                part: DescribePart::Keys,
            },
        ]
    );
}

/// A data layer that plans every request as one statement, to show that the
/// root's one data layer is what every run plans through.
struct Fake;

impl DataLayer for Fake {
    fn capabilities(&self, _: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        Ok(capabilities(&[OptionalRequest::CountRows]))
    }

    fn plan(&self, _: &DataRequest, _: &ProviderChoice) -> Result<FetchPlan, CoreError> {
        Ok(FetchPlan::Sql {
            statements: vec![SqlStatement {
                dialect: Dialect::new("duckdb"),
                text: "SELECT 1".into(),
                params: vec![],
            }],
            expect: ExpectedSchema::Discover {},
        })
    }

    fn read(
        &self,
        _: &DataRequest,
        _: &ProviderChoice,
        _: &[arrow_array::RecordBatch],
    ) -> Result<sdm_contract::catalog::CatalogAnswer, CoreError> {
        Ok(sdm_contract::catalog::CatalogAnswer::Tables(vec![
            "fake".into(),
        ]))
    }
}

fn capabilities(optional: &[OptionalRequest]) -> ProviderCapabilities {
    ProviderCapabilities {
        derived_relations: false,
        neighbor_lookback: false,
        sample_rows: false,
        optional: optional.iter().copied().collect(),
        max_keys_per_request: None,
        max_seed: u64::MAX,
        text_times: None,
        null_order: None,
        index_advice: None,
        sampling: Determinism::SeededStable,
        fallback: None,
    }
}

fn count() -> DataRequest {
    DataRequest::CountRows {
        table: "t".into(),
        filters: vec![],
    }
}

fn scan() -> DataRequest {
    DataRequest::Scan {
        table: "t".into(),
        columns: vec![],
        filters: vec![],
        limit: None,
    }
}

fn named(err: &CoreError, field: &str) -> String {
    match err.fields.get(field) {
        Some(sdm_contract::Detail::Scalar(Scalar::Utf8(text))) => text.clone(),
        other => panic!("no {field}: {other:?} in {err:?}"),
    }
}

#[test]
fn the_root_s_data_layer_serves_a_built_in_dialect_or_a_host_provider() {
    let root = root();
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
    };
    let access = root.data_access(&duckdb).unwrap();
    // The dialect declares what sdm-data renders for it: every request
    // kind from T1.23, a count of rows and a scan among them (an
    // undeclared optional request is refused by name before it is
    // planned: a_host_provider_is_asked_only_for_what_it_declared).
    let FetchPlan::Sql { statements, .. } = access.plan(&count()).unwrap() else {
        panic!("DuckDB answers with SQL");
    };
    assert_eq!(statements[0].text, r#"SELECT COUNT(*) AS "rows" FROM "t""#);
    let err = access.plan(&scan()).unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidRequest);
    assert_eq!(named(&err, "field"), "columns");
    // It renders the catalog (T1.11b), each plan expecting the answer's
    // fixed shape.
    assert_eq!(
        access.capabilities().optional,
        BTreeSet::from([
            OptionalRequest::CountRows,
            OptionalRequest::Preview,
            OptionalRequest::KeyCounts,
            OptionalRequest::TimeBounds,
            OptionalRequest::ColumnRanges,
            OptionalRequest::RepeatedKey,
            OptionalRequest::SampleEntities,
            OptionalRequest::Neighbors,
            OptionalRequest::Lookup,
        ])
    );
    let tables = DataRequest::Describe {
        tables: vec![],
        part: DescribePart::Tables,
    };
    let FetchPlan::Sql { statements, expect } = access.plan(&tables).unwrap() else {
        panic!("DuckDB answers with SQL");
    };
    assert_eq!(statements.len(), 1);
    // A catalog answer is in the dialect's own catalog's shape, which the
    // data layer reads, not the contract's (decision E54).
    assert!(matches!(expect, ExpectedSchema::Schema { .. }));
    assert_ne!(expect, tables.expected());

    // A dialect this build knows but renders nothing for (Snowflake's;
    // SQLite's and Postgres's are served from T2.2 and T2.3).
    let snowflake = ProviderChoice::Sql {
        dialect: Dialect::new("snowflake"),
    };
    let err = root.data_access(&snowflake).err().unwrap();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
    );
    assert_eq!(named(&err, "dialect"), "snowflake");
    assert!(run_of(&root, describe(snowflake)).is_err());
    // A name the data layer does not know is refused the same way, and
    // not repeated: the contract names no dialect, so it lets any through.
    let unknown = ProviderChoice::Sql {
        dialect: Dialect::new("a dialect\nof no name"),
    };
    let err = root.data_access(&unknown).err().unwrap();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
    );
    assert!(!err.fields.contains_key("dialect"));

    // Swapping the data layer replaces the root's one value.
    let swapped = CompositionRoot::new(Arc::new(Fake));
    let access = swapped.data_access(&duckdb).unwrap();
    let FetchPlan::Sql { statements, .. } = access.plan(&count()).unwrap() else {
        panic!("not the swapped layer's plan");
    };
    assert_eq!(statements[0].text, "SELECT 1");
}

#[test]
fn a_host_provider_is_asked_only_for_what_it_declared() {
    let host = ProviderChoice::Host {
        capabilities: capabilities(&[OptionalRequest::Preview]),
    };
    let access = root().data_access(&host).unwrap();
    // A host provider is held to the shape the contract fixes for a
    // request's answer, as a rendering is (decision E36); a request whose
    // columns are the source's own discovers them.
    let pass = |request: DataRequest| {
        assert_eq!(
            access.plan(&request).unwrap(),
            FetchPlan::Host {
                expect: request.expected()
            }
        );
    };
    pass(scan());
    let tables = DataRequest::Describe {
        tables: vec![],
        part: DescribePart::Tables,
    };
    assert!(matches!(tables.expected(), ExpectedSchema::Schema { .. }));
    pass(tables);
    let preview = DataRequest::Preview {
        table: "t".into(),
        columns: vec![],
        limit: 10,
    };
    assert_eq!(preview.expected(), ExpectedSchema::Discover {});
    pass(preview);
    let err = access.plan(&count()).unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Unsupported, ErrorCode::UnsupportedRequest)
    );
    assert_eq!(named(&err, "request"), "count_rows");
}

#[test]
fn profiles_resolve_with_their_model_and_capabilities() {
    let tabular = Profile::named(ProfileName::Tabular);
    assert_eq!(tabular.mode, Mode::Inference);
    assert!(!tabular.capabilities.temporal && tabular.capabilities.static_);
    assert_eq!(Profile::named(ProfileName::FineTuning).mode, Mode::Training);
    assert!(!Profile::named(ProfileName::Timeseries).capabilities.static_);
    assert!(
        !Profile::named(ProfileName::Anomaly)
            .capabilities
            .forecasting
    );
    assert!(
        Profile::named(ProfileName::Relational)
            .capabilities
            .link_prediction
    );

    let resolved = Profile::resolve(&ProfileSpec {
        name: ProfileName::Tabular,
        model: Some("my-model".into()),
        capabilities: BTreeMap::from([(Capability::Temporal, true), (Capability::Static, false)]),
    });
    assert_eq!(resolved.capabilities.model, "my-model");
    assert!(resolved.capabilities.temporal && !resolved.capabilities.static_);
}

fn features() -> QueryFeatures {
    QueryFeatures {
        tables: BTreeSet::from(["users".to_owned()]),
        entity_table: "users".into(),
        target_span: Some(Span {
            start_row: 1,
            start_col: 8,
            end_row: 1,
            end_col: 17,
        }),
        ..QueryFeatures::default()
    }
}

#[test]
fn a_capability_only_ever_refuses() {
    let relational = Profile::named(ProfileName::Relational);
    let mut everything = features();
    everything.tables.insert("orders".into());
    everything.follows_foreign_key = true;
    everything.temporal_target = true;
    everything.derived_target = true;
    everything.entity_filter = true;
    everything.counterfactual = true;
    everything.ranks_foreign_key = true;
    everything.num_forecasts = 3;
    assert!(check_capabilities(&everything, &relational).is_empty());
}

#[test]
fn each_missing_capability_gives_its_code_in_order() {
    let tabular = Profile::named(ProfileName::Tabular);
    let mut query = features();
    query.tables.insert("orders".into());
    query.temporal_target = true;
    query.derived_target = true;
    query.entity_filter = true;
    query.counterfactual = true;
    query.ranks_foreign_key = true;
    query.num_forecasts = 2;
    let found: Vec<String> = check_capabilities(&query, &tabular)
        .into_iter()
        .map(|d| d.code)
        .collect();
    assert_eq!(
        found,
        [
            model::MODEL_SINGLE_TABLE_ONLY,
            model::MODEL_NO_TEMPORAL_TARGET,
            model::MODEL_NO_DERIVED_TARGET,
            model::MODEL_NO_ENTITY_FILTER,
            model::MODEL_NO_COUNTERFACTUAL,
            model::MODEL_CANNOT_RANK_FOREIGN_KEY,
            model::MODEL_NO_FORECASTING,
        ]
    );

    // One table that follows a key, against a model without relationships.
    let mut joined = features();
    joined.follows_foreign_key = true;
    let codes: Vec<String> = check_capabilities(&joined, &tabular)
        .into_iter()
        .map(|d| d.code)
        .collect();
    assert_eq!(codes, [model::MODEL_NO_RELATIONSHIPS]);

    // A static target, against a model that needs a window.
    let diagnostics = check_capabilities(&features(), &Profile::named(ProfileName::Timeseries));
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code, model::MODEL_NO_STATIC_TARGET);
    assert_eq!(
        diagnostics[0].span,
        Some(Span {
            start_row: 1,
            start_col: 8,
            end_row: 1,
            end_col: 17,
        })
    );
    assert!(diagnostics[0].message.contains("a time-series model"));
}

fn start_with(operation: Operation, attachments: Attachments) -> Result<sdm_core::Run, CoreError> {
    start(
        &root(),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        attachments,
    )
}

fn attachment_named(err: &CoreError) -> u32 {
    match err.fields.get("attachment") {
        Some(sdm_contract::Detail::Scalar(Scalar::UInt32(id))) => *id,
        other => panic!("no attachment named: {other:?} in {err:?}"),
    }
}

#[test]
fn a_start_attaches_exactly_what_it_names_as_the_kind_it_names() {
    let full = attachments_of(&validate());
    let mut missing = full.clone();
    missing.remove(&AttachmentId(2));
    let mut extra = full.clone();
    extra.insert(AttachmentId(9), Attachment::Bytes(vec![]));
    let mut arrow = full.clone();
    arrow.insert(
        AttachmentId(1),
        Attachment::Arrow(sdm_core::ArrowData::new(
            Arc::new(arrow_schema::Schema::empty()),
            vec![],
        )),
    );
    for (attachments, named) in [(missing, 2), (extra, 9), (arrow, 1)] {
        let err = start_with(validate(), attachments).err().unwrap();
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::InvalidMessage)
        );
        assert_eq!(attachment_named(&err), named);
    }
    // One attachment named for two fields.
    let Operation::ValidateQuery(mut twice) = validate() else {
        unreachable!()
    };
    twice.schema = twice.query;
    let err = start_with(
        Operation::ValidateQuery(twice),
        Attachments::from([(AttachmentId(1), json(&"q"))]),
    )
    .err()
    .unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    assert_eq!(attachment_named(&err), 1);
}

#[test]
fn an_attachment_that_is_not_its_json_is_refused_without_its_text() {
    let mut attachments = attachments_of(&validate());
    attachments.insert(
        AttachmentId(2),
        Attachment::Bytes(br#"{"tables": "CANARY-not-a-list", "links": []}"#.to_vec()),
    );
    let err = start_with(validate(), attachments).err().unwrap();
    assert_eq!(err.code, ErrorCode::InvalidMessage);
    assert_eq!(attachment_named(&err), 2);
    let written = serde_json::to_string(&err).unwrap();
    assert!(!written.contains("CANARY"), "{written}");
    assert_eq!(err.fields["category"], sdm_contract::Detail::text("data"));
}

fn materialize(seed: Option<Seed>) -> Operation {
    Operation::Materialize(Materialize {
        provider: ProviderChoice::Host {
            capabilities: ProviderCapabilities {
                derived_relations: false,
                neighbor_lookback: false,
                sample_rows: false,
                optional: BTreeSet::new(),
                max_keys_per_request: None,
                max_seed: u64::MAX,
                text_times: None,
                null_order: None,
                index_advice: None,
                sampling: Determinism::SeededStable,
                fallback: None,
            },
        },
        schema: AttachmentId(1),
        held: None,
        query: AttachmentId(2),
        indices: None,
        anchor_time: None,
        seed,
        unseeded: false,
        anchor_entity: false,
        values: None,
        options: vec![],
        most_parts: None,
        most_entities: None,
        most_calls: None,
        candidates: None,
        sampler_state: None,
    })
}

#[test]
fn a_negative_seed_is_refused_by_the_core_and_a_seed_of_any_size_is_taken() {
    let negative = materialize(Some(Seed::from(-1_i128)));
    let err = start_with(negative.clone(), attachments_of(&negative))
        .err()
        .unwrap();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest)
    );
    assert_eq!(err.fields["field"], sdm_contract::Detail::text("seed"));
    // 2**70 passes the core's entry, which takes a seed of any size; this
    // start names no entities, which the run refuses (tests/materialize.rs
    // follows 2**70 to the sample it seeds).
    for seed in [
        Some(Seed::from(1_u128 << 70)),
        Some(Seed::from(0_u128)),
        None,
    ] {
        let operation = materialize(seed);
        let mut run = start_with(operation.clone(), attachments_of(&operation)).unwrap();
        let err = run.next().unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest);
        assert_eq!(err.fields["field"], sdm_contract::Detail::text("indices"));
    }
}

/// A start of `describe_source` naming `tables` (`[name, source]`).
fn describe_named(tables: &[(&str, &str)]) -> Result<sdm_core::Run, CoreError> {
    let operation = describe(ProviderChoice::Sql {
        dialect: Dialect::new("duckdb"),
    });
    let spec = SourceSpec {
        tables: tables
            .iter()
            .map(|(name, source)| NamedTable {
                name: (*name).into(),
                source: (*source).into(),
            })
            .collect(),
        primary_keys: BTreeMap::new(),
        time_columns: BTreeMap::new(),
        relationships: None,
    };
    start(
        &root(),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        Attachments::from([(AttachmentId(7), json(&spec))]),
    )
}

#[test]
fn tables_named_wrongly_are_refused_before_anything_is_fetched() {
    // T1.23: as the reference refuses a mapping of tables (its
    // `_require_table_names`) and a name the schema may not give a table
    // (`_check_name`): each `INVALID_REQUEST` naming the field, never the
    // source's name held to the schema's rules.
    for tables in [
        vec![("", "t")],
        vec![("t", "")],
        vec![("t", "a"), ("t", "b")],
        vec![("a.b", "t")],
        vec![("a`b", "t")],
        vec![("a\nb", "t")],
        vec![("a\rb", "t")],
    ] {
        let err = describe_named(&tables).unwrap().next().unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidRequest, "{tables:?}");
        assert_eq!(named(&err, "field"), "tables", "{tables:?}");
    }
    // A source name holding those characters is the source's to hold: the
    // run asks for its columns.
    for source in ["a.b", "a`b", "a\nb", "\"q\""] {
        assert!(
            matches!(
                describe_named(&[("ok", source)]).unwrap().next(),
                Ok(Step::Fetch { .. })
            ),
            "{source:?}"
        );
    }
}

/// A `validate_query` run of `query` over one table `t` with a time column,
/// under `facts`.
fn validate_run(query: &str, facts: HostFacts) -> Result<sdm_core::Run, CoreError> {
    let schema: QuerySchema = serde_json::from_value(serde_json::json!({
        "tables": [{"name": "t", "columns": [
            {"name": "id", "dtype": "int", "stype": "ID"},
            {"name": "c", "dtype": "int", "stype": "categorical"},
            {"name": "ts", "dtype": "time", "stype": "timestamp"}],
         "primary_key": "id", "time_column": "ts", "end_time_column": null}],
        "links": [],
    }))
    .unwrap();
    let operation = validate();
    start(
        &root(),
        StartRun {
            contract: CONTRACT_VERSION,
            facts,
            operation,
        },
        Attachments::from([
            (AttachmentId(1), json(&query)),
            (AttachmentId(2), json(&schema)),
        ]),
    )
}

#[test]
fn a_day_first_date_literal_warns_on_the_run_s_warnings_channel() {
    let mut run = validate_run(
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 13/01/2024 AND t.ts < 01/02/2024",
        facts(),
    )
    .unwrap();
    let Step::Done {
        result: OperationResult::ValidateQuery(validation),
    } = run.next().unwrap()
    else {
        panic!("validate_query did not finish with its result");
    };
    assert!(validation.accepted, "{:?}", validation.diagnostics);
    // The warning is the run's, not a diagnostic of the query.
    assert!(validation.diagnostics.is_empty());
    let warnings = run.take_warnings();
    let codes: Vec<&str> = warnings.iter().map(|w| w.code.as_str()).collect();
    assert_eq!(codes, ["DATE_LITERAL_DAY_FIRST"]);
    assert_eq!(
        warnings[0].fields["format"],
        sdm_contract::Detail::text("%d/%m/%Y")
    );
    assert!(sdm_core::warning_codes().contains(&("DATE_LITERAL_DAY_FIRST", &[][..])));
}

/// T1.6c: a packed date (D8) is refused by name, not read, so it never
/// needs the current year: with a zone the run cannot resolve, a six-digit
/// packed date (which D5 would have placed) and an eight-digit one are
/// refused as packed, never for the zone; a two-digit year still is, and
/// whichever literal is built first decides.
#[test]
fn a_local_zone_the_run_cannot_resolve_refuses_only_a_year_it_must_place() {
    // The current year is read only for a two-digit year (D5): with a zone
    // the run cannot resolve, a query that places none is answered as
    // ever, and one that does is refused by name.
    let mut facts = facts();
    facts.local_zone = Some(sdm_contract::facts::LocalZone::Iana("Mars/Olympus".into()));
    for query in [
        "PREDICT t.c FOR EACH t.id",
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 2024-01-13",
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 13/01/2024",
    ] {
        let Step::Done {
            result: OperationResult::ValidateQuery(validation),
        } = validate_run(query, facts.clone()).unwrap().next().unwrap()
        else {
            panic!("{query}: no answer");
        };
        assert!(validation.accepted, "{query}: {:?}", validation.diagnostics);
    }
    for query in [
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 1/2/76",
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 1/2/76 AND t.ts < 760101-01-01",
    ] {
        let err = validate_run(query, facts.clone())
            .unwrap()
            .next()
            .unwrap_err();
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest),
            "{query}"
        );
        assert_eq!(named(&err, "field"), "facts.local_zone", "{query}");
    }
    for query in [
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 760101-01-01",
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 20240101-01-01",
        "PREDICT t.c FOR EACH t.id WHERE t.ts > 760101-01-01 AND t.ts < 1/2/76",
    ] {
        let err = validate_run(query, facts.clone())
            .unwrap()
            .next()
            .unwrap_err();
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest),
            "{query}"
        );
        assert_eq!(named(&err, "rule"), "date_literal_packed", "{query}");
        // The query's own refusal (field query, naming the query given and
        // its first packed literal, KO2-86), not the local zone's.
        assert_eq!(named(&err, "field"), "query", "{query}");
        assert_eq!(named(&err, "query"), query, "{query}");
        assert_eq!(
            named(&err, "literal"),
            query
                .split(" > ")
                .nth(1)
                .unwrap()
                .split(' ')
                .next()
                .unwrap(),
            "{query}"
        );
    }
}

/// T1.6c brief, Tests: "a D8 literal is refused by name" on the
/// `validate_query` operation, as a host runs it: the run ends with the
/// core's error, `INVALID_REQUEST`, naming `operation` `validate_query` and
/// `rule` `date_literal_packed`, with the literal's place and never its
/// text; a literal with no run of five digits or more in the same place is
/// answered as before, and D9's still with the diagnostic.
#[test]
fn a_packed_date_literal_ends_validate_query_with_the_core_s_error() {
    for literal in [
        "230203-06-09",
        "20240101/00/123456",
        "2024-01-01 00000:00:00",
        "10000-01-01",
    ] {
        let query = format!("PREDICT t.c FOR EACH t.id WHERE t.ts > {literal}");
        let mut run = validate_run(&query, facts()).unwrap();
        let err = run.next().unwrap_err();
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest),
            "{literal}"
        );
        assert_eq!(named(&err, "operation"), "validate_query", "{literal}");
        assert_eq!(named(&err, "rule"), "date_literal_packed", "{literal}");
        for name in ["start_row", "start_col", "end_row", "end_col"] {
            assert!(
                matches!(
                    err.fields.get(name),
                    Some(sdm_contract::Detail::Scalar(Scalar::UInt64(_)))
                ),
                "{literal}: no {name} in {err:?}"
            );
        }
        assert_eq!(
            err.fields["start_col"],
            sdm_contract::Detail::Scalar(Scalar::UInt64(39)),
            "{literal}"
        );
        // KO2-86: the literal and the query given are named, exactly, and
        // the literal is in no other field.
        assert_eq!(named(&err, "literal"), literal, "{literal}");
        assert_eq!(named(&err, "query"), query, "{literal}");
        let mut rest = err.clone();
        rest.fields.remove("literal");
        rest.fields.remove("query");
        let written = serde_json::to_string(&rest).unwrap();
        assert!(!written.contains(literal), "{written}");
    }
    for (literal, accepted) in [("2023-02-03", true), ("2023-02-29", false)] {
        let query = format!("PREDICT t.c FOR EACH t.id WHERE t.ts > {literal}");
        let Step::Done {
            result: OperationResult::ValidateQuery(validation),
        } = validate_run(&query, facts()).unwrap().next().unwrap()
        else {
            panic!("{literal}: no answer");
        };
        assert_eq!(validation.accepted, accepted, "{literal}");
        let codes: Vec<&str> = validation
            .diagnostics
            .iter()
            .map(|d| d.code.as_str())
            .collect();
        let want: &[&str] = if accepted {
            &[]
        } else {
            &["PQL_VALUE_INVALID_LITERAL"]
        };
        assert_eq!(codes, want, "{literal}");
    }
}
