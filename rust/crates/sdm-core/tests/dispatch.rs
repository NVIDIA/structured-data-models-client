// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dispatch, the composition root, profiles and model capability checks.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use sdm_contract::data::{
    DataRequest, Determinism, Dialect, OptionalRequest, ProviderCapabilities, ProviderChoice,
    SqlStatement,
};
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{
    Capability, DescribeSource, Operation, OperationResult, ProfileName, ProfileSpec, QuerySchema,
    Span, ValidateQuery,
};
use sdm_contract::run::StartRun;
use sdm_contract::{CONTRACT_VERSION, CoreError, ErrorKind, HostFacts, Scalar, Step, Unit, codes};
use sdm_core::compose::{DataAccess, SqlCompiler};
use sdm_core::profiles::{Mode, Profile, QueryFeatures, check_capabilities, codes as model};
use sdm_core::{CompositionRoot, RunState, start};

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
    }
}

fn validate() -> Operation {
    Operation::ValidateQuery(ValidateQuery {
        query: "PREDICT t.c FOR EACH t.id".into(),
        schema: QuerySchema {
            tables: vec![],
            links: vec![],
        },
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
        tables: vec!["t".into()],
        primary_keys: BTreeMap::new(),
        time_columns: BTreeMap::new(),
        check_keys: false,
    })
}

fn run_of(root: &CompositionRoot, operation: Operation) -> Result<sdm_core::Run, CoreError> {
    start(
        root,
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
    )
}

#[test]
fn another_contract_version_is_refused_before_a_run_exists() {
    let err = start(
        &CompositionRoot::default(),
        StartRun {
            contract: CONTRACT_VERSION + 1,
            facts: facts(),
            operation: validate(),
        },
    )
    .err()
    .unwrap();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Protocol, codes::INVALID_MESSAGE)
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
            &CompositionRoot::default(),
            StartRun {
                contract: CONTRACT_VERSION,
                facts: bad,
                operation: validate(),
            },
        )
        .err()
        .unwrap();
        assert_eq!(err.code, codes::INVALID_REQUEST, "{now:?}");
    }
}

#[test]
fn every_operation_dispatches_to_a_run() {
    let root = CompositionRoot::default();
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
            optional: BTreeSet::new(),
            max_keys_per_request: None,
            sampling: Determinism::Unseeded,
        },
    };
    let err = run_of(&root, describe(host)).unwrap().next().unwrap_err();
    assert_eq!(err.code, codes::UNSUPPORTED_REQUEST);
}

struct Fake;

impl SqlCompiler for Fake {
    fn compile(&self, _: &DataRequest) -> Result<SqlStatement, CoreError> {
        Ok(SqlStatement {
            dialect: Dialect::Duckdb,
            text: "SELECT 1".into(),
            params: vec![],
        })
    }
}

#[test]
fn the_composition_root_picks_a_built_in_dialect_or_a_host_provider() {
    let empty = CompositionRoot::default();
    let duckdb = ProviderChoice::Sql {
        dialect: Dialect::Duckdb,
    };
    let err = empty.data_access(&duckdb).err().unwrap();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Unsupported, codes::UNSUPPORTED_REQUEST)
    );
    assert!(run_of(&empty, describe(duckdb.clone())).is_err());

    let root = CompositionRoot::default().with_dialect(Dialect::Duckdb, Arc::new(Fake));
    let access = root.data_access(&duckdb).unwrap();
    let sql = access
        .prepare(&DataRequest::CountRows {
            table: "t".into(),
            filters: vec![],
        })
        .unwrap()
        .unwrap();
    assert_eq!(sql.text, "SELECT 1");
    let sqlite = ProviderChoice::Sql {
        dialect: Dialect::Sqlite,
    };
    assert!(
        root.data_access(&sqlite).is_err(),
        "only registered dialects"
    );
}

#[test]
fn a_host_provider_is_asked_only_for_what_it_declared() {
    let access = DataAccess::Host(ProviderCapabilities {
        optional: BTreeSet::from([OptionalRequest::Preview]),
        max_keys_per_request: None,
        sampling: Determinism::SeededStable,
    });
    let scan = DataRequest::Scan {
        table: "t".into(),
        columns: vec![],
        filters: vec![],
        limit: None,
    };
    assert_eq!(access.prepare(&scan).unwrap(), None, "scan is the minimum");
    let describe = DataRequest::Describe { tables: vec![] };
    assert_eq!(access.prepare(&describe).unwrap(), None);
    let preview = DataRequest::Preview {
        table: "t".into(),
        columns: vec![],
        limit: 10,
    };
    assert_eq!(access.prepare(&preview).unwrap(), None, "declared");
    let count = DataRequest::CountRows {
        table: "t".into(),
        filters: vec![],
    };
    let err = access.prepare(&count).unwrap_err();
    assert_eq!(
        (err.kind, err.code.as_str()),
        (ErrorKind::Unsupported, codes::UNSUPPORTED_REQUEST)
    );
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
