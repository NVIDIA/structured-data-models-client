// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use sdm_contract::data::ProviderError;
use sdm_contract::materialize::{MaterializeSourceRead, MaterializeSourceTable, SourceReadKind};

fn error() -> ProviderError {
    ProviderError {
        kind: sdm_contract::ErrorKind::DataSource,
        retryable: true,
        sqlstate: Some("08006".into()),
        vendor_code: Some("actual-driver-code".into()),
        error_class: None,
        driver_message: Some("private row values must never be public".into()),
    }
}

#[test]
fn reported_native_source_io_reuses_the_same_provider_policy_and_request_location() {
    let source = sdm_contract::data::Relation::named("events");
    let tables = vec![MaterializeSourceTable {
        temporal_columns: Vec::new(),
        time_column: None,
        source: source.clone(),
        read: source.clone(),
        columns: vec!["id".into(), "time".into()],
    }];
    for (kind, name) in [
        (SourceReadKind::SampleEntities, "sample_entities"),
        (SourceReadKind::Neighbors, "neighbors"),
        (SourceReadKind::Lookup, "lookup"),
        (SourceReadKind::Scan, "scan"),
    ] {
        let read = MaterializeSourceRead {
            kind,
            source: source.clone(),
            columns: vec!["time".into(), "id".into()],
        };
        let actual = reported_failure(&tables, &read, error()).unwrap_err();
        let expected = crate::describe::provider_error(error(), name);
        assert_eq!(actual, expected);
        let public = serde_json::to_string(&actual).unwrap();
        assert!(!public.contains("private row values"));
        assert_eq!(actual.fields["request"], Detail::text(name));
    }
}

#[test]
fn a_terminal_source_diagnostic_does_not_authorize_foreign_or_partial_source_meaning() {
    let source = sdm_contract::data::Relation::named("events");
    let tables = vec![MaterializeSourceTable {
        temporal_columns: Vec::new(),
        time_column: None,
        source: source.clone(),
        read: source.clone(),
        columns: vec!["id".into()],
    }];
    for read in [
        MaterializeSourceRead {
            kind: SourceReadKind::Lookup,
            source: sdm_contract::data::Relation::named("foreign"),
            columns: vec!["id".into()],
        },
        MaterializeSourceRead {
            kind: SourceReadKind::Lookup,
            source: source.clone(),
            columns: vec!["foreign".into()],
        },
        MaterializeSourceRead {
            kind: SourceReadKind::Lookup,
            source: source.clone(),
            columns: vec!["id".into(), "id".into()],
        },
    ] {
        assert_eq!(
            reported_failure(&tables, &read, error()).unwrap_err().code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
}

#[test]
fn an_actual_derived_read_is_distinct_from_its_logical_source_relation() {
    let source = sdm_contract::data::Relation::named("events");
    let mut actual = source.clone();
    actual.found = true;
    actual.columns.push(sdm_contract::data::RelationColumn {
        name: "id".into(),
        column: Some("id".into()),
        expression: None,
    });
    let tables = vec![MaterializeSourceTable {
        temporal_columns: Vec::new(),
        time_column: None,
        source: source.clone(),
        read: actual.clone(),
        columns: vec!["id".into()],
    }];
    let mut read = MaterializeSourceRead {
        kind: SourceReadKind::Lookup,
        source: actual,
        columns: vec!["id".into()],
    };
    assert_eq!(
        reported_failure(&tables, &read, error()).unwrap_err(),
        crate::describe::provider_error(error(), "lookup")
    );
    read.source = source;
    assert_eq!(
        reported_failure(&tables, &read, error()).unwrap_err().code,
        sdm_contract::ErrorCode::InvalidMessage
    );
}
