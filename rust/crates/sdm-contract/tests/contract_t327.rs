// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! T3.27, contract first: every contract change the remaining Phase 3
//! packages need (Snowflake, views, Lakebase, Java on Databricks), in one
//! version. Each addition round-trips and is optional, so a message written
//! before it reads the same; each new `DataLayer` method keeps today's
//! behaviour, or refuses by name, until a dialect answers it; and every
//! message site of the old client those packages port has its sentence.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the core's sources, to hold the proposed sentences unsaid"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema, SchemaRef};
use sdm_contract::catalog::{CatalogAnswer, ViewDefinition};
use sdm_contract::data::{
    AttachmentId, DataLayer, DataRequest, DescribePart, FailureFacts, FetchPlan, KeyData, KeySet,
    ProviderCapabilities, ProviderChoice, ProviderError, Relation, SqlParam, SqlStatement,
    ViewChoice,
};
use sdm_contract::ops::Dtype;
use sdm_contract::ops::{DescribedSource, ModelEndpoint, Predict, ViewNote};
use sdm_contract::run::{ModelAnswer, ModelOutcome};
use sdm_contract::view::{
    DeclaredType, ExprRefusal, ExprVerdict, FilterRefusal, ViewKind, ViewTextAnswer, ViewTextAsk,
};
use sdm_contract::wording::{self, Group};
use sdm_contract::{Code, CoreError, Detail, ErrorCode, ErrorKind, Sentence, WarningCode};
use serde_json::json;

fn read<T: serde::de::DeserializeOwned>(value: serde_json::Value) -> T {
    serde_json::from_value(value).unwrap()
}

fn written<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).unwrap()
}

// --- ProviderError: error_class in, refuses_expression out ---------------------------

#[test]
fn a_provider_error_carries_its_error_class_and_no_judgement() {
    let error: ProviderError = read(json!({
        "kind": "data_source", "retryable": false, "sqlstate": "42501",
        "vendor_code": null, "error_class": "INSUFFICIENT_PERMISSIONS"
    }));
    assert_eq!(
        error.error_class.as_deref(),
        Some("INSUFFICIENT_PERMISSIONS")
    );
    assert_eq!(written(&error)["error_class"], "INSUFFICIENT_PERMISSIONS");
    // Left out, none, and written as before.
    let plain: ProviderError = read(json!({
        "kind": "data_source", "retryable": true, "sqlstate": null, "vendor_code": "2003"
    }));
    assert_eq!(plain.error_class, None);
    assert!(written(&plain).get("error_class").is_none());
    // What a failure means is the data layer's to say (`DataLayer::failure`),
    // never the host's: a judgement the host sends is no field of the error.
    let judged = serde_json::from_value::<ProviderError>(json!({
        "kind": "data_source", "retryable": false, "sqlstate": "42703",
        "vendor_code": null, "refuses_expression": true
    }));
    assert!(judged.is_err());
}

// --- ProviderChoice: a semantic model ----------------------------------------------

#[test]
fn a_semantic_model_is_a_sql_provider_s_view_and_optional() {
    let choice: ProviderChoice = read(json!({
        "provider": "sql", "dialect": "databricks", "namespace": ["c", "s"],
        "view": {"name": "sales_metrics", "trust_expressions": true}
    }));
    let ProviderChoice::Sql { view, .. } = &choice else {
        unreachable!()
    };
    assert_eq!(
        view,
        &Some(ViewChoice {
            name: "sales_metrics".into(),
            trust_expressions: true,
        })
    );
    assert_eq!(written(&choice)["view"]["name"], "sales_metrics");
    let plain: ProviderChoice = read(json!({"provider": "sql", "dialect": "duckdb"}));
    assert_eq!(plain, ProviderChoice::sql("duckdb"));
    assert_eq!(
        written(&plain),
        json!({"provider": "sql", "dialect": "duckdb"})
    );
    // The flag is the caller's, always written.
    assert!(
        serde_json::from_value::<ProviderChoice>(json!({
            "provider": "sql", "dialect": "snowflake", "view": {"name": "v"}
        }))
        .is_err()
    );
}

// --- SqlParam::Text: a key set's JSON as a one-cell text attachment ------------------

#[test]
fn a_text_parameter_names_its_attachment_and_the_step_lists_it() {
    let param: SqlParam = read(json!({"param": "text", "attachment": 7}));
    assert_eq!(
        param,
        SqlParam::Text {
            attachment: AttachmentId(7)
        }
    );
    assert_eq!(written(&param), json!({"param": "text", "attachment": 7}));
    let plan = FetchPlan::Sql {
        result_conversions: Vec::new(),
        statements: vec![SqlStatement {
            dialect: sdm_contract::data::Dialect::new("postgres"),
            text: "SELECT 1".into(),
            params: vec![
                SqlParam::KeySet {
                    keys: KeySet {
                        attachment: AttachmentId(3),
                        columns: vec![],
                    },
                    relation: "keys".into(),
                },
                param,
            ],
        }],
        expect: sdm_contract::data::ExpectedSchema::Discover {},
    };
    // Every attachment a plan's statements bind, in order: what the fetch
    // step lists for the host to take.
    assert_eq!(plan.key_sets(), vec![AttachmentId(3), AttachmentId(7)]);
}

// --- DescribePart::ViewDefinition --------------------------------------------------

#[test]
fn a_view_s_definition_is_a_catalog_part() {
    let request: DataRequest = read(json!({
        "request": "describe", "tables": [{"source": ["c", "s", "v"]}],
        "part": "view_definition"
    }));
    let DataRequest::Describe { part, .. } = &request else {
        unreachable!()
    };
    assert_eq!(*part, DescribePart::ViewDefinition);
    assert_eq!(written(&request)["part"], "view_definition");
    let answer = CatalogAnswer::ViewDefinition(ViewDefinition {
        kind: ViewKind::Metric,
        text: "version: 0.1".into(),
        parameters: vec![("p".into(), "INT".into())],
        located: vec![Some("c".into()), Some("s".into())],
        columns: vec![("d".into(), "string".into())],
    });
    assert_eq!(answer.part(), DescribePart::ViewDefinition);
    // The kind is the data layer's answer, never read off the definition:
    // the same text, even none, is either kind.
    for kind in [ViewKind::Metric, ViewKind::Semantic] {
        let definition = ViewDefinition {
            kind,
            text: String::new(),
            parameters: vec![],
            located: vec![],
            columns: vec![],
        };
        assert_eq!(definition.kind, kind);
    }
    assert_eq!(ViewKind::Semantic.noun(), "semantic view");
}

// --- DescribedSource.notes ---------------------------------------------------------

#[test]
fn a_description_carries_its_typed_view_notes_and_none_by_default() {
    let described: DescribedSource = read(json!({
        "description": 4,
        "notes": [{
            "sentence": "view_parameter_note",
            "fields": {"parameter": {"t": "utf8", "v": "region"}, "default": {"t": "utf8", "v": "EMEA"}}
        }, {
            "sentence": "view_time_column_note",
            "fields": {
                "column": {"t": "utf8", "v": "ts"}, "table": {"t": "utf8", "v": "orders"},
                "reasons": {"t": "utf8", "v": "a window measure of the view orders by it"}
            },
            "table": "orders", "time_column": "ts"
        }]
    }));
    assert_eq!(described.notes.len(), 2);
    let note: &ViewNote = &described.notes[1];
    assert_eq!(note.sentence, Sentence::ViewTimeColumnNote);
    assert_eq!(note.table.as_deref(), Some("orders"));
    assert_eq!(note.time_column.as_deref(), Some("ts"));
    assert_eq!(note.fields["column"], Detail::text("ts"));
    assert_eq!(
        written(&described)["notes"][0]["sentence"],
        "view_parameter_note"
    );
    let plain: DescribedSource = read(json!({"description": 4}));
    assert!(plain.notes.is_empty());
    assert_eq!(written(&plain), json!({"description": 4}));
    // A note says a note's sentence.
    assert!(described.notes.iter().all(ViewNote::is_note));
    let mut crossed = described.notes[0].clone();
    crossed.sentence = Sentence::ViewJoinSkipped;
    assert!(!crossed.is_note());
}

// --- ModelEndpoint and ModelAnswer (T3.13) -------------------------------------------

#[test]
fn a_predict_names_its_serving_endpoint_and_none_is_today_s() {
    let mut predict: Predict = read(json!({
        "state": null, "model": "m", "request_id": "r", "outputs": []
    }));
    assert_eq!(predict.model_endpoint, None);
    assert!(written(&predict).get("model_endpoint").is_none());
    predict.model_endpoint = Some(ModelEndpoint::ServingEndpoint {
        name: "kumo-rfm".into(),
        max_request_bytes: 16_000_000,
    });
    let text = written(&predict);
    assert_eq!(
        text["model_endpoint"],
        json!({
            "endpoint": "serving_endpoint", "name": "kumo-rfm",
            "max_request_bytes": 16_000_000
        })
    );
    assert_eq!(read::<Predict>(text), predict);
}

#[test]
fn a_failed_model_call_hands_over_its_status_detail_and_codes() {
    let outcome: ModelOutcome = read(json!({
        "outcome": "failed",
        "answer": {
            "status": 400, "detail": 9, "provider_code": "BAD_REQUEST",
            "provider_error": "BadRequest"
        }
    }));
    let ModelOutcome::Failed { answer } = &outcome else {
        unreachable!()
    };
    assert_eq!(
        answer,
        &ModelAnswer {
            status: 400,
            detail: Some(AttachmentId(9)),
            provider_code: Some("BAD_REQUEST".into()),
            provider_error: Some("BadRequest".into()),
        }
    );
    // An SDK failure whose message holds no code still carries its code;
    // a REST body is the detail alone.
    let bare: ModelOutcome = read(json!({"outcome": "failed", "answer": {"status": 503}}));
    assert_eq!(
        written(&bare),
        json!({"outcome": "failed", "answer": {"status": 503}})
    );
}

// --- The DataLayer's new methods: each default today's, or a refusal by name ---------

/// A data layer that answers only what every layer must.
struct Bare;

impl DataLayer for Bare {
    fn capabilities(&self, _: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        unreachable!()
    }

    fn plan(&self, _: &DataRequest, _: &ProviderChoice) -> Result<FetchPlan, CoreError> {
        unreachable!()
    }

    fn read(
        &self,
        _: &DataRequest,
        _: &ProviderChoice,
        _: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        unreachable!()
    }
}

fn keys(values: &[i64]) -> KeyData {
    let schema: SchemaRef = Arc::new(Schema::new(vec![ArrowField::new(
        "key",
        DataType::Int64,
        false,
    )]));
    let column: ArrayRef = Arc::new(Int64Array::from(values.to_vec()));
    KeyData {
        batches: vec![RecordBatch::try_new(schema.clone(), vec![column]).unwrap()],
        schema,
    }
}

fn keyed_plan(attachments: &[u32]) -> FetchPlan {
    FetchPlan::Sql {
        result_conversions: Vec::new(),
        statements: vec![SqlStatement {
            dialect: sdm_contract::data::Dialect::new("postgres"),
            text: "SELECT 1".into(),
            params: attachments
                .iter()
                .map(|&id| SqlParam::KeySet {
                    keys: KeySet {
                        attachment: AttachmentId(id),
                        columns: vec![],
                    },
                    relation: "keys".into(),
                })
                .collect(),
        }],
        expect: sdm_contract::data::ExpectedSchema::Discover {},
    }
}

#[test]
fn a_failure_means_nothing_until_a_dialect_reads_it() {
    let error: ProviderError = read(json!({
        "kind": "data_source", "retryable": false, "sqlstate": "42P01",
        "vendor_code": "2003", "error_class": "TABLE_OR_VIEW_NOT_FOUND"
    }));
    let facts = Bare.failure(&ProviderChoice::sql("databricks"), &error);
    assert_eq!(facts, FailureFacts::default());
    assert!(!facts.not_found && !facts.refuses_expression && facts.unusable.is_none());
}

#[test]
fn a_relation_is_asked_as_given_until_a_dialect_spells_it_otherwise() {
    let relation = Relation::of(vec!["db".into(), "s".into(), "orders".into()]);
    let asked = Bare.spellings(&ProviderChoice::sql("snowflake"), &relation);
    assert_eq!(asked, vec![relation]);
}

#[test]
fn a_request_binds_in_one_fetch_as_the_core_made_its_key_sets() {
    // bind's default (T3.28 simplified it): the plan as it is, binding the
    // key sets the core made, under the ids it made them under.
    let request: DataRequest = read(json!({
        "request": "describe", "tables": [{"source": ["t"]}], "part": "tables"
    }));
    let made = BTreeMap::from([
        (AttachmentId(3), keys(&[1, 2])),
        (AttachmentId(5), keys(&[3])),
    ]);
    let plan = keyed_plan(&[3, 5, 3]);
    let bound = Bare
        .bind(
            &ProviderChoice::sql("duckdb"),
            &request,
            plan.clone(),
            &made,
            |x| x.to_string(),
        )
        .unwrap();
    assert_eq!(bound.fetches.len(), 1);
    assert_eq!(bound.fetches[0].plan, plan);
    assert!(bound.fetches[0].batches.is_empty());
}

/// A data layer that reads views, saying the clause a reason names its
/// warehouse in, as a dialect does (`it contains '$$', which Databricks SQL
/// does not quote with`, old-behaviour row 64).
struct Worded;

impl DataLayer for Worded {
    fn capabilities(&self, _: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        unreachable!()
    }

    fn plan(&self, _: &DataRequest, _: &ProviderChoice) -> Result<FetchPlan, CoreError> {
        unreachable!()
    }

    fn read(
        &self,
        _: &DataRequest,
        _: &ProviderChoice,
        _: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        unreachable!()
    }

    fn view_text(&self, _: &ProviderChoice, ask: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        Ok(match ask {
            ViewTextAsk::ReasonWords {
                reason: ExprRefusal::DollarQuotes,
            } => ViewTextAnswer::Words(Some(
                "it contains '$$', which Databricks SQL does not quote with".into(),
            )),
            ViewTextAsk::ReasonWords { .. } => ViewTextAnswer::Words(None),
            ViewTextAsk::DeclaredType { data_type } => {
                ViewTextAnswer::DeclaredType(match data_type.as_str() {
                    "BIGINT" => DeclaredType::Known(Dtype::Int),
                    "VARIANT" => DeclaredType::Unsupported,
                    _ => DeclaredType::Unknown,
                })
            }
            _ => unreachable!(),
        })
    }
}

#[test]
fn a_reason_that_names_the_warehouse_is_worded_by_the_dialect() {
    let databricks = ProviderChoice::sql("databricks");
    // The same clause, whether the filter or a dropped column met it.
    let filter = FilterRefusal::Unsafe(ExprRefusal::DollarQuotes);
    let column = ExprVerdict {
        reason: ExprRefusal::DollarQuotes,
        untrusted: false,
    };
    for reason in [
        match filter {
            FilterRefusal::Unsafe(reason) => reason,
            _ => unreachable!(),
        },
        column.reason,
    ] {
        let said = Worded
            .view_text(&databricks, ViewTextAsk::ReasonWords { reason })
            .unwrap();
        assert_eq!(
            said,
            ViewTextAnswer::Words(Some(
                "it contains '$$', which Databricks SQL does not quote with".into()
            ))
        );
    }
    // A reason in no warehouse's words is the core's to word.
    let plain = Worded
        .view_text(
            &databricks,
            ViewTextAsk::ReasonWords {
                reason: ExprRefusal::SubQuery,
            },
        )
        .unwrap();
    assert_eq!(plain, ViewTextAnswer::Words(None));
}

#[test]
fn a_view_column_s_declared_type_is_read_by_the_dialect_s_type_map() {
    let databricks = ProviderChoice::sql("databricks");
    let read = |data_type: &str| {
        Worded
            .view_text(
                &databricks,
                ViewTextAsk::DeclaredType {
                    data_type: data_type.into(),
                },
            )
            .unwrap()
    };
    // Known, unsupported (refused) and unknown (left to the sample) apart.
    assert_eq!(
        read("BIGINT"),
        ViewTextAnswer::DeclaredType(DeclaredType::Known(Dtype::Int))
    );
    assert_eq!(
        read("VARIANT"),
        ViewTextAnswer::DeclaredType(DeclaredType::Unsupported)
    );
    assert_eq!(
        read("TEXT"),
        ViewTextAnswer::DeclaredType(DeclaredType::Unknown)
    );
}

// --- A serving endpoint's refusal keeps its identity ------------------------------

#[test]
fn a_serving_endpoint_s_refusal_carries_its_status_and_its_own_code() {
    // As the old client's `NimRequestError(status, code=...)`: the status
    // and the endpoint's code, or none, which the host raises its
    // endpoint-refusal error with.
    for (sentence, status, code) in [
        (Sentence::ServingCallersFault, 400, Some("INVALID_REQUEST")),
        (
            Sentence::ServingRefused,
            503,
            Some("TEMPORARILY_UNAVAILABLE"),
        ),
        (Sentence::ServingFailed, 500, None),
        (
            Sentence::ServingFailedCoded,
            404,
            Some("RESOURCE_DOES_NOT_EXIST"),
        ),
    ] {
        assert_eq!(sentence.code(), Code::Error(ErrorCode::EndpointRefused));
        let error = CoreError::refused_by_endpoint(sentence, status, code, "m");
        assert_eq!(error.code, ErrorCode::EndpointRefused);
        assert_eq!(error.kind, ErrorKind::ModelEndpoint);
        assert!(error.holds_to_table());
        assert_eq!(
            error.fields[sdm_contract::error::REFUSED_STATUS],
            Detail::Scalar(sdm_contract::Scalar::UInt16(status))
        );
        assert_eq!(
            error.fields.get(sdm_contract::error::REFUSED_CODE),
            code.map(Detail::text).as_ref()
        );
    }
    assert_eq!(
        serde_json::to_value(ErrorCode::EndpointRefused).unwrap(),
        "ENDPOINT_REFUSED"
    );
}

#[test]
fn a_view_s_text_is_refused_by_name_until_a_dialect_reads_views() {
    let refused = Bare
        .view_text(
            &ProviderChoice::sql("duckdb"),
            ViewTextAsk::TableReference {
                name: "c.s.t".into(),
            },
        )
        .map(|_: ViewTextAnswer| ())
        .unwrap_err();
    assert_eq!(refused.sentence, Sentence::ViewsUnread);
    assert_eq!(refused.code, ErrorCode::UnsupportedRequest);
    assert_eq!(refused.kind, ErrorKind::Unsupported);
    assert_eq!(refused.fields["rule"], Detail::text("view"));
    assert!(refused.holds_to_table());
}

// --- Sentences: one per message site the remaining packages port -------------------

/// Every sentence T3.27 adds, by name, with its code: the old client's
/// message sites in `old-behaviour/T3.5-views.md` and `T3.6-snowflake.md`,
/// and `CL/core/databricks_serving.py`'s.
fn added() -> Vec<(&'static str, Code)> {
    use ErrorCode as E;
    use WarningCode as W;
    let e = Code::Error;
    let w = Code::Warning;
    vec![
        ("views_unread", e(E::UnsupportedRequest)),
        ("view_name_invalid", e(E::InvalidRequest)),
        ("view_not_found", e(E::InvalidRequest)),
        ("view_not_a_metric_view", e(E::InvalidRequest)),
        ("view_no_definition", e(E::InvalidRequest)),
        ("view_not_yaml", e(E::InvalidRequest)),
        ("view_not_a_mapping", e(E::InvalidRequest)),
        ("view_misses_source", e(E::InvalidRequest)),
        ("view_misses_entry", e(E::InvalidRequest)),
        ("view_not_a_list", e(E::InvalidRequest)),
        ("view_nests_itself", e(E::InvalidRequest)),
        ("view_tables_absent", e(E::InvalidRequest)),
        ("view_entry_unhashable", e(E::InvalidRequest)),
        ("view_entry_not_text", e(E::InvalidRequest)),
        ("view_value_too_large", e(E::InvalidRequest)),
        ("view_source_unsupported", e(E::InvalidRequest)),
        ("view_source_unqualified", e(E::InvalidRequest)),
        ("view_on_metric_view", e(E::InvalidRequest)),
        ("view_filter_refused", e(E::InvalidRequest)),
        ("view_filter_refused_no_default", e(E::InvalidRequest)),
        ("view_filter_refused_untrusted", e(E::InvalidRequest)),
        ("view_join_name_collides", e(E::InvalidRequest)),
        ("view_table_cannot_hold", e(E::InvalidRequest)),
        ("view_invalid", e(E::InvalidRequest)),
        ("view_semantic_not_found", e(E::InvalidRequest)),
        ("view_semantic_definition_unread", e(E::InvalidRequest)),
        ("view_table_omitted", e(E::InvalidRequest)),
        ("view_column_omitted", e(E::InvalidRequest)),
        ("view_materialization_ignored", w(W::ViewReadInPart)),
        ("view_fields_ignored", w(W::ViewReadInPart)),
        ("view_join_not_a_mapping", w(W::ViewReadInPart)),
        ("view_join_nameless", w(W::ViewReadInPart)),
        ("view_join_skipped", w(W::ViewReadInPart)),
        ("view_join_skipped_with_it", w(W::ViewReadInPart)),
        ("view_dimension_no_table", w(W::ViewReadInPart)),
        ("view_dimension_not_a_mapping", w(W::ViewReadInPart)),
        ("view_dimension_misses", w(W::ViewReadInPart)),
        ("view_dimension_refused", w(W::ViewReadInPart)),
        ("view_dimension_multiple_tables", w(W::ViewReadInPart)),
        ("view_measure_refused", w(W::ViewReadInPart)),
        ("view_measure_column_shadowed", w(W::ViewReadInPart)),
        ("view_column_refused", w(W::ViewReadInPart)),
        ("view_join_unchecked", w(W::ViewReadInPart)),
        ("view_derived_key_replaced", w(W::ViewReadInPart)),
        ("view_edge_dropped", w(W::ViewReadInPart)),
        ("view_composite_key", w(W::ViewReadInPart)),
        ("view_column_cross_table", w(W::ViewReadInPart)),
        ("view_columns_cross_table", w(W::ViewReadInPart)),
        ("view_expression_dropped", w(W::ViewReadInPart)),
        ("view_expression_dropped_untrusted", w(W::ViewReadInPart)),
        ("view_table_expression_dropped", w(W::ViewReadInPart)),
        (
            "view_table_expression_dropped_untrusted",
            w(W::ViewReadInPart),
        ),
        ("view_relationship_refused", w(W::ViewReadInPart)),
        ("view_measure_note", Code::Note),
        ("view_time_column_note", Code::Note),
        ("view_time_column_note_added", Code::Note),
        ("view_time_column_note_renamed", Code::Note),
        ("view_parameter_note", Code::Note),
        ("no_current_schema", e(E::InvalidRequest)),
        ("unspecified_schema", e(E::InvalidRequest)),
        ("no_current_database", e(E::InvalidRequest)),
        ("null_anchor", e(E::InvalidRequest)),
        ("table_no_columns", e(E::InvalidRequest)),
        ("seed_ignored", w(W::SeedIgnored)),
        ("serving_value_not_json", e(E::InvalidRequest)),
        ("serving_past_limit", e(E::InvalidRequest)),
        ("serving_not_json", e(E::TransportError)),
        ("serving_not_an_object", e(E::TransportError)),
        ("serving_refused", e(E::EndpointRefused)),
        ("serving_callers_fault", e(E::EndpointRefused)),
        ("serving_failed", e(E::EndpointRefused)),
        ("serving_failed_coded", e(E::EndpointRefused)),
        ("serving_prediction_row_count", e(E::TransportError)),
        ("serving_response_column_missing", e(E::TransportError)),
    ]
}

#[test]
fn every_message_site_the_remaining_packages_port_has_its_sentence() {
    let by_name: BTreeMap<&str, Sentence> =
        Sentence::ALL.iter().map(|s| (s.as_str(), *s)).collect();
    for (name, code) in added() {
        let sentence = by_name
            .get(name)
            .unwrap_or_else(|| panic!("no sentence {name}"));
        assert_eq!(sentence.code(), code, "{name}");
        assert!(!wording::template(*sentence).text.is_empty(), "{name}");
    }
    // Every message a semantic model is read in part for is said in one
    // warning, opened by the model as the host names it.
    for (name, code) in added() {
        if code == Code::Warning(WarningCode::ViewReadInPart) {
            assert_eq!(
                wording::template(by_name[name]).group,
                Some(Group::ViewReadInPart),
                "{name}"
            );
        }
    }
    let read_in_part = wording::grouping(Group::ViewReadInPart);
    assert_eq!(read_in_part.code, WarningCode::ViewReadInPart);
    assert_eq!(
        read_in_part.opening,
        "{host.semantic_model!r} has what a graph cannot hold, so it is read in part:\n- "
    );
    assert_eq!(read_in_part.between, "\n- ");
    assert_eq!(
        serde_json::to_value(WarningCode::ViewReadInPart).unwrap(),
        "VIEW_READ_IN_PART"
    );
}

/// The owner approvals recorded in the T35-OWNER-2026-10-24-ERRORS and
/// LOOKUP-ERRORS decision rows release these names, without changing their
/// templates, wire identities or the producer scan for pending proposals.
#[test]
fn owner_approved_named_refusals_are_no_longer_pending_proposals() {
    for sentence in [
        Sentence::ViewNotYaml,
        Sentence::ViewMissesEntry,
        Sentence::ViewNotAList,
        Sentence::ViewNestsItself,
        Sentence::ViewEntryUnhashable,
        Sentence::ViewEntryNotText,
        Sentence::ViewValueTooLarge,
        Sentence::ViewSemanticNotFound,
        Sentence::ViewSemanticDefinitionUnread,
        Sentence::NoCurrentDatabase,
    ] {
        assert!(
            !wording::PROPOSED.contains(&sentence),
            "{sentence:?} is owner-approved"
        );
    }
}

#[test]
fn a_proposed_improvement_s_sentence_is_marked_and_used_by_none() {
    let proposed: Vec<&str> = wording::PROPOSED.iter().map(|s| s.as_str()).collect();
    assert_eq!(proposed, vec!["null_anchor"]);
    // Said by no producer: no crate of the core names one.
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    for krate in [
        "sdm-core",
        "sdm-data",
        "sdm-schema",
        "sdm-engine",
        "sdm-request",
        "sdm-host",
    ] {
        let mut stack = vec![crates.join(krate).join("src")];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                for sentence in wording::PROPOSED {
                    let named = format!("Sentence::{sentence:?}");
                    assert!(!text.contains(&named), "{} says {named}", path.display());
                }
            }
        }
    }
}

#[test]
fn the_new_message_sites_say_the_old_client_s_words() {
    let said = |s: Sentence| wording::template(s).text;
    // `CL/schema.py:1220-1225`, view row 11.
    assert_eq!(
        said(Sentence::ViewTablesAbsent),
        "{absent!r} are not tables of {host.source!r}; it has {offered!r}"
    );
    // `SNOW/table.py:88-93`.
    assert_eq!(
        said(Sentence::UnspecifiedSchema),
        "Unspecified 'schema' for table '{table}' in database '{in_database}'"
    );
    // `CL/core/databricks_serving.py:489-500`.
    assert_eq!(
        said(Sentence::ServingPredictionRowCount),
        "serving endpoint {endpoint!r} did not return exactly one prediction row"
    );
    assert_eq!(
        said(Sentence::ServingResponseColumnMissing),
        "serving endpoint {endpoint!r} answered with a row that has no 'response_json' string \
         column"
    );
}

#[test]
fn the_view_sentences_say_the_old_client_s_words() {
    // `RE/graph.py` and `DB/metric_view.py` at 43e55f4f, the quoted values
    // the core's (`sdm-pql`'s Python text), the clauses the core's.
    let said = |s: Sentence| wording::template(s).text;
    assert_eq!(
        said(Sentence::ViewExpressionDroppedUntrusted),
        "Failed to add column '{column}' since {reason}, and expressions taken from the metric \
         view are executed as SQL under your warehouse credentials. Pass \
         {advice:trust_expressions()} to keep it if you trust whoever authored the view"
    );
    assert_eq!(
        said(Sentence::ViewTableExpressionDropped),
        "Failed to add column '{column}' of table '{table}' since {reason}, and expressions \
         taken from the semantic view are executed as SQL under your warehouse credentials"
    );
    assert_eq!(
        said(Sentence::ViewFilterRefusedUntrusted),
        "Cannot apply the metric view filter {filter} since {reason}. The graph would otherwise \
         read rows the view leaves out. Create a regular view with this WHERE clause and point \
         the metric view's source at it, or pass {advice:trust_expressions()} to apply it if you \
         trust whoever authored the view"
    );
    assert_eq!(
        said(Sentence::ViewJoinSkippedWithIt),
        "Failed to add join '{join}', and with it {lost}, since {reason}"
    );
    assert_eq!(
        said(Sentence::ViewColumnsCrossTable),
        "Failed to add columns '{columns!r}' of table '{table}' since their expressions \
         reference other tables"
    );
    assert_eq!(
        said(Sentence::ViewMeasureNote),
        "'{measure}' is {function}({table}.{column}, 0, 30, days) over the next 30 days"
    );
    assert_eq!(
        said(Sentence::NoCurrentSchema),
        "No current 'schema' set. Please specify the {database:backend} schema manually"
    );
    assert_eq!(
        said(Sentence::ServingPastLimit),
        "this request is {bytes:,} bytes once wrapped for serving endpoint {endpoint!r}, above \
         the {limit:,} byte limit it accepts. Send fewer rows per call: a smaller \
         num_context_samples, or fewer entities to score"
    );
    // The way out of an untrusted expression is the host's to spell.
    assert_eq!(
        wording::Advice::TrustExpressions.as_str(),
        "trust_expressions"
    );
}

/// A SQL choice's semantic model is read through its accessor (T3.5's
/// contract follow-up); a choice naming none, or a host provider, has none.
#[test]
fn a_choice_s_view_is_read_by_its_accessor() {
    let read: sdm_contract::data::ProviderChoice = serde_json::from_value(serde_json::json!({
        "provider": "sql", "dialect": "databricks",
        "view": {"name": "c.s.v", "trust_expressions": true},
    }))
    .unwrap();
    let view = read.view().unwrap();
    assert_eq!(
        (view.name.as_str(), view.trust_expressions),
        ("c.s.v", true)
    );
    assert!(
        sdm_contract::data::ProviderChoice::sql("databricks")
            .view()
            .is_none()
    );
}
