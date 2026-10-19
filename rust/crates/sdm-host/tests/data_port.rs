// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data port and the zones through `HostRun` (T1.11a, decision H14):
//! the zones a host's facts name are the ones the core resolves with, an
//! unknown zone reaches the core by no name, and a fetch the driver handed
//! over in several batches, or empty, is admitted as one attachment of its
//! schema.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: builds fixture zone directories from the vendored files"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{ArrayRef, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::data::{AttachmentId, DataRequest, ExpectedSchema, FetchPlan};
use sdm_contract::facts::{PlatformFacts, ZoneRootFact};
use sdm_contract::ops::{Operation, OperationResult};
use sdm_contract::run::{FetchOutcome, FetchReply, ReplyId, StartRun};
use sdm_contract::test_ops::{TestFetch, TestOperation, TestResult};
use sdm_contract::{
    CONTRACT_VERSION, ColumnType, ControlMessage, CoreError, ErrorCode, ErrorKind, HostFacts,
    Scalar, Step, Unit,
};
use sdm_core::{ArrowData, Attachment, Attachments};
use sdm_host::{HostRun, Outgoing, ZoneError, ZoneResolver, ZoneRoot, ZoneSource, encode, read};

fn vendored() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tzdata-2026c")
}

/// A fresh fixture directory holding `files`, each `(zone name, vendored
/// zone whose bytes it gets)`.
fn fixture(files: &[(&str, &str)]) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "data-port-roots-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&dir);
    for (name, from) in files {
        let target = dir.join(name);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(vendored().join(from), target).unwrap();
    }
    dir
}

fn start_message(operation: Operation, roots: &[&Path]) -> Vec<u8> {
    encode(&ControlMessage::Start(Box::new(StartRun {
        contract: CONTRACT_VERSION,
        facts: HostFacts {
            now: Scalar::Timestamp {
                unit: Unit::Us,
                zone: Some("UTC".into()),
                value: 1_767_225_600_000_000,
            },
            local_zone: None,
            zone_roots: roots
                .iter()
                .enumerate()
                .map(|(i, path)| ZoneRootFact {
                    path: path.to_str().unwrap().into(),
                    version: Some(format!("root-{i}")),
                })
                .collect(),
            platform: PlatformFacts {
                os: "linux".into(),
                strftime_pads_year: false,
            },
            date_literal_range: None,
        },
        operation,
    })))
    .unwrap()
}

fn error(out: Outgoing) -> CoreError {
    match read(&out).unwrap() {
        ControlMessage::Error { error } => error,
        other => panic!("not an error: {other:?}"),
    }
}

/// A summer instant of 2024, as a `ts` scalar.
fn summer() -> Scalar {
    Scalar::Timestamp {
        unit: Unit::Us,
        zone: Some("UTC".into()),
        value: 1_719_835_200_000_000,
    }
}

/// The offset the core's run gives for `zone` under a host whose lookup
/// roots are `roots`, and the run's record of where each zone came from.
fn offset(
    zone: &str,
    roots: &[&Path],
) -> (
    Result<i32, CoreError>,
    std::collections::BTreeMap<String, ZoneSource>,
) {
    let operation = Operation::Test(TestOperation::Zone {
        zone: zone.into(),
        at: summer(),
    });
    let run = HostRun::start(
        &sdm_host::default_root(),
        &start_message(operation, roots),
        Attachments::new(),
    )
    .map_err(error)
    .unwrap();
    let answer = match run.next() {
        Ok(out) => match read(&out).unwrap() {
            ControlMessage::Step(Step::Done {
                result: OperationResult::Test(TestResult { offset_seconds, .. }),
            }) => Ok(offset_seconds.unwrap()),
            other => panic!("not the zone's answer: {other:?}"),
        },
        Err(out) => Err(error(out)),
    };
    (answer, run.zone_sources())
}

#[test]
fn the_zones_a_host_passes_are_the_ones_the_core_resolves_with() {
    // The first root's America/New_York carries Tokyo's rules, so the
    // offset the core answers shows which root it read.
    let tokyo_rules = fixture(&[("America/New_York", "Asia/Tokyo")]);
    let new_york = fixture(&[("America/New_York", "America/New_York")]);
    let tokyo = 9 * 3600;
    let eastern = -4 * 3600;

    let (got, sources) = offset("America/New_York", &[&tokyo_rules]);
    assert_eq!(got.unwrap(), tokyo);
    assert_eq!(
        sources["America/New_York"],
        ZoneSource::HostRoot {
            index: 0,
            path: tokyo_rules.clone(),
            version: Some("root-0".into()),
        },
        "the host records the source of the zone the core resolved"
    );
    assert_eq!(offset("America/New_York", &[&new_york]).0.unwrap(), eastern);
    assert_eq!(
        offset("america/new_york", &[&tokyo_rules, &new_york])
            .0
            .unwrap(),
        tokyo,
        "in the host's order, in any case"
    );
    assert_eq!(
        offset("America/New_York", &[&new_york, &tokyo_rules])
            .0
            .unwrap(),
        eastern
    );
    // No root holds it: the bundled database.
    let (got, sources) = offset("America/New_York", &[]);
    assert_eq!(got.unwrap(), eastern);
    assert!(matches!(
        sources["America/New_York"],
        ZoneSource::Bundled { .. }
    ));
}

#[test]
fn runs_with_the_same_roots_keep_their_own_record() {
    let root = fixture(&[("Europe/Berlin", "Europe/Berlin")]);
    let (_, first) = offset("Europe/Berlin", &[&root]);
    let (_, second) = offset("UTC", &[&root]);
    assert!(first.contains_key("Europe/Berlin"));
    assert!(
        !second.contains_key("Europe/Berlin"),
        "a run records only the zones it resolved: {second:?}"
    );
}

#[test]
fn an_unknown_zone_reaches_the_core_by_no_name() {
    let canary = "Canary/Secret_7f3e";
    // The host's own error quotes the name, for the host's own use.
    let resolver = ZoneResolver::new([]);
    let ZoneError::Unknown { name } = resolver.resolve(canary).unwrap_err();
    assert_eq!(name, canary);
    assert!(
        resolver
            .resolve(canary)
            .unwrap_err()
            .to_string()
            .contains(canary)
    );
    // The core is told only that the zone is unknown (decision H17).
    let (got, _) = offset(canary, &[]);
    let err = got.unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest)
    );
    let text = serde_json::to_string(&err).unwrap();
    assert!(!text.contains("Secret_7f3e"), "{text}");
    // Directly too: the trait the core reads carries no name.
    let zones: Arc<dyn sdm_core::Zones> = Arc::new(ZoneResolver::new([ZoneRoot {
        path: fixture(&[]),
        version: None,
    }]));
    assert_eq!(zones.zone(canary).unwrap_err(), sdm_core::UnknownZone);
}

// --- a fetch in several batches, and an empty one ----------------------------------

fn schema() -> SchemaRef {
    // large_utf8 is admitted as utf8: the batches are admitted, not only
    // passed along.
    Arc::new(Schema::new(vec![
        Field::new("name", DataType::LargeUtf8, true),
        Field::new("n", DataType::Int64, false),
    ]))
}

fn batch(names: &[Option<&str>], ns: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(LargeStringArray::from(names.to_vec())) as ArrayRef,
            Arc::new(Int64Array::from(ns.to_vec())) as ArrayRef,
        ],
    )
    .unwrap()
}

/// The admitted data the core holds after a fetch answered with `data`,
/// echoed back by the test-only fetch, under the plan's `expect`.
fn admitted(data: ArrowData, expect: ExpectedSchema) -> ArrowData {
    let fetch = TestFetch {
        request: DataRequest::Scan {
            table: "t".into(),
            columns: vec!["name".into(), "n".into()],
            filters: vec![],
            limit: None,
        },
        plan: Some(FetchPlan::Host { expect }),
        provider: None,
        echo: true,
        undeclared: vec![],
        fallback: None,
    };
    let run = HostRun::start(
        &sdm_host::default_root(),
        &start_message(
            Operation::Test(TestOperation::Fetch {
                fetch: AttachmentId(1),
            }),
            &[],
        ),
        Attachments::from([(
            AttachmentId(1),
            Attachment::Bytes(serde_json::to_vec(&fetch).unwrap()),
        )]),
    )
    .map_err(error)
    .unwrap();
    run.next().map_err(error).unwrap();
    run.reply(
        &encode(&ControlMessage::FetchReply(FetchReply {
            id: ReplyId(1),
            outcome: FetchOutcome::Data {
                data: AttachmentId(9),
                continuation: None,
            },
        }))
        .unwrap(),
        Attachments::from([(AttachmentId(9), Attachment::Arrow(data))]),
    )
    .map_err(error)
    .unwrap();
    let ControlMessage::Step(Step::Done {
        result: OperationResult::Test(TestResult { data: Some(id), .. }),
    }) = read(&run.next().map_err(error).unwrap()).unwrap()
    else {
        panic!("the fetch echoes its data");
    };
    let Some(Attachment::Arrow(data)) = run.take_attachment(id).map_err(error).unwrap() else {
        panic!("the echo is Arrow");
    };
    data
}

/// The rows of `data` as one batch, its batch boundaries left out.
fn rows(data: &ArrowData) -> RecordBatch {
    arrow_select::concat::concat_batches(&data.schema, &data.batches).unwrap()
}

fn expected_fields() -> ExpectedSchema {
    let field = |name: &str, column_type, nullable| sdm_contract::Field {
        name: name.into(),
        column_type,
        nullable,
        metadata: Default::default(),
    };
    ExpectedSchema::Schema {
        fields: vec![
            field("name", ColumnType::Utf8, true),
            field("n", ColumnType::Int64, false),
        ],
    }
}

#[test]
fn a_fetch_in_several_batches_is_admitted_as_one_attachment_of_the_same_rows() {
    let parts = vec![
        batch(&[Some("a"), None], &[1, 2]),
        batch(&[], &[]),
        batch(&[Some("é"), Some(""), Some("c")], &[3, 4, 5]),
    ];
    let whole = arrow_select::concat::concat_batches(&schema(), &parts).unwrap();
    for expect in [ExpectedSchema::Discover {}, expected_fields()] {
        let several = admitted(ArrowData::new(schema(), parts.clone()), expect.clone());
        let one = admitted(ArrowData::new(schema(), vec![whole.clone()]), expect);
        assert_eq!(several.schema, one.schema);
        assert_eq!(several.received, one.received);
        assert_eq!(several.schema.field(0).data_type(), &DataType::Utf8);
        assert_eq!(several.received.field(0).data_type(), &DataType::LargeUtf8);
        assert_eq!(rows(&several), rows(&one));
        assert_eq!(rows(&several).num_rows(), 5);
    }
}

#[test]
fn an_empty_fetch_keeps_its_schema() {
    for expect in [ExpectedSchema::Discover {}, expected_fields()] {
        let no_batches = admitted(ArrowData::new(schema(), vec![]), expect.clone());
        let empty_batch = admitted(ArrowData::new(schema(), vec![batch(&[], &[])]), expect);
        for data in [&no_batches, &empty_batch] {
            assert_eq!(rows(data).num_rows(), 0);
            assert_eq!(data.schema.fields().len(), 2);
            assert_eq!(data.schema.field(0).data_type(), &DataType::Utf8);
            assert_eq!(data.received.as_ref(), schema().as_ref());
        }
        assert_eq!(no_batches.schema, empty_batch.schema);
        assert_eq!(rows(&no_batches), rows(&empty_batch));
    }
}
