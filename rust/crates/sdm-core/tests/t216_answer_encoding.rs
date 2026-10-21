// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The core reads a model answer in each encoding the reference's reader
//! takes (T2.16's test author's, from the whole-phase review's round two,
//! Codex P2).
//!
//! The reference reads an answer's bytes with `json.loads`, which decodes
//! them in the encoding `json.detect_encoding` finds: UTF-32 or UTF-16 by
//! their byte order marks, UTF-8 with its mark, or, with no mark, UTF-16 or
//! UTF-32 of either byte order by where the first four bytes are zero, and
//! UTF-8 otherwise. Since T2.14b the core reads the bytes the endpoint sent
//! (`sdm_wire::json::read_answer`), and refused every form but UTF-8 as
//! `INVALID_RESPONSE` (`field=response`, `reason=not_json`).
//!
//! One answer, holding text past ASCII (a character UTF-16 writes as a
//! surrogate pair among it) and a `NaN` token the reference's reader takes,
//! is written in each form of [`FORMS`]:
//!
//! - `reader.<form>`: `read_answer` reads it as the value it reads the
//!   UTF-8 text as;
//! - `run.<form>`: a predict run answered with it is done, with the
//!   result and the endpoint's metadata it gives for the UTF-8 text.
//!
//! And bytes no such encoding decodes, or holding a lone surrogate, which
//! the reference keeps and the core refuses (register P1B-24), are refused
//! ([`refused`]): `read_answer` refuses them with the category named, and
//! the run ends `INVALID_RESPONSE`, `field=response`, `reason=not_json`,
//! that `category`, holding nothing of the answer. Text under four bytes is
//! decoded as `json.detect_encoding` decodes it.
//!
//! The core's decoding is `sdm_wire::json::answer_text`, which
//! `read_answer` calls first (T2.16's fix).
//!
//! The same forms through the public Python calls against a localhost
//! endpoint are `contract/tests/test_t216_response_encoding.py`.
//!
//! Gating: `contract/tests/t216-pending.json`'s `pending` names, as
//! `rust:reader.<form>` and `rust:run.<form>`, the checks that failed when
//! written. Each must still fail (strict); once one passes it fails, naming
//! itself, until it is taken out. Written on the fix, so none is pending;
//! on `f133c4b8`, before it, every form's but `utf-8` failed.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the gating list"
)]

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

use sdm_contract::data::AttachmentId;
use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{Operation, OperationResult, Predict, REQUEST_STATE_VERSION, RequestState};
use sdm_contract::run::{Endpoint, StartRun};
use sdm_contract::{
    CONTRACT_VERSION, CoreError, Detail, ErrorCode, ErrorKind, HostFacts, Scalar, Step, Unit,
};
use sdm_core::{
    Attachment, Attachments, CompositionRoot, ModelResult, Run, UnknownZone, Zones, start,
};
use sdm_wire::json::read_answer;

const PENDING: &str = "contract/tests/t216-pending.json";
const CANARY: &str = "CANARY-t216";

/// The answer: the payload's two rows answered, the second's prediction a
/// `NaN` token, and metadata holding text past ASCII, one character of it
/// past the Basic Multilingual Plane.
const ANSWER: &str = concat!(
    r#"{"object":"prediction.response","id":"r9","model":"kumo-relational","#,
    "\"metadata\":{\"note\":\"caf\u{e9} \u{1F600} \u{4E2D}\"},",
    r#""predictions":[{"id":"2","row_index":0,"prediction":1.5},"#,
    r#"{"id":"3","row_index":1,"prediction":NaN}]}"#
);

fn utf16(text: &str, big: bool) -> Vec<u8> {
    text.encode_utf16()
        .flat_map(|unit| {
            if big {
                unit.to_be_bytes()
            } else {
                unit.to_le_bytes()
            }
        })
        .collect()
}

fn utf32(text: &str, big: bool) -> Vec<u8> {
    text.chars()
        .flat_map(|c| {
            let unit = u32::from(c);
            if big {
                unit.to_be_bytes()
            } else {
                unit.to_le_bytes()
            }
        })
        .collect()
}

fn marked(mark: &[u8], mut rest: Vec<u8>) -> Vec<u8> {
    let mut out = mark.to_vec();
    out.append(&mut rest);
    out
}

/// How a form writes a text.
type Write = fn(&str) -> Vec<u8>;

/// Each form the reference's reader takes, by the name
/// `contract/tools/t216_encoding_record.py` gives it, and how it writes a
/// text.
const FORMS: [(&str, Write); 10] = [
    ("utf-8", |t| t.as_bytes().to_vec()),
    ("utf-8-sig", |t| {
        marked(b"\xef\xbb\xbf", t.as_bytes().to_vec())
    }),
    ("utf-16-le-bom", |t| marked(b"\xff\xfe", utf16(t, false))),
    ("utf-16-be-bom", |t| marked(b"\xfe\xff", utf16(t, true))),
    ("utf-16-le", |t| utf16(t, false)),
    ("utf-16-be", |t| utf16(t, true)),
    ("utf-32-le-bom", |t| {
        marked(b"\xff\xfe\x00\x00", utf32(t, false))
    }),
    ("utf-32-be-bom", |t| {
        marked(b"\x00\x00\xfe\xff", utf32(t, true))
    }),
    ("utf-32-le", |t| utf32(t, false)),
    ("utf-32-be", |t| utf32(t, true)),
];

/// The answer with its id the canary followed by `#`, in UTF-8.
fn canary_answer() -> String {
    ANSWER.replacen("\"r9\"", &format!("\"{CANARY}#\""), 1)
}

/// `bytes` with the `unit` bytes that write `#` replaced by `with`.
fn swapped(mut bytes: Vec<u8>, unit: &[u8], with: &[u8]) -> Vec<u8> {
    let at = bytes.windows(unit.len()).position(|w| w == unit).unwrap();
    bytes.splice(at..at + unit.len(), with.iter().copied());
    bytes
}

/// Bytes the core refuses, each with the category its refusal names:
///
/// - no encoding the reference's reader finds decodes them, and
///   `json.loads` refuses them too (`UnicodeDecodeError`): UTF-16 with no
///   mark one byte short, UTF-16 with its mark one byte short, and UTF-8
///   holding a byte UTF-8 never holds;
/// - they hold a lone surrogate, which `json.loads` decodes
///   (`surrogatepass`) and keeps, and the core refuses (register P1B-24):
///   UTF-16 and UTF-32 with their marks, and UTF-8 (a Databricks
///   `response_json` holding one, as the serving transport encodes it).
fn refused() -> Vec<(&'static str, Vec<u8>, &'static str)> {
    let mut odd = utf16(ANSWER, false);
    odd.pop();
    let mut marked_odd = marked(b"\xff\xfe", utf16(ANSWER, false));
    marked_odd.pop();
    let text = canary_answer();
    vec![
        ("utf-16-odd", odd, "encoding"),
        ("utf-16-bom-odd", marked_odd, "encoding"),
        (
            "utf-8-invalid",
            swapped(text.clone().into_bytes(), b"#", b"\xff"),
            "syntax",
        ),
        (
            "utf-16-lone-surrogate",
            marked(
                b"\xff\xfe",
                swapped(utf16(&text, false), b"#\x00", b"\x00\xd8"),
            ),
            "encoding",
        ),
        (
            "utf-32-lone-surrogate",
            marked(
                b"\xff\xfe\x00\x00",
                swapped(utf32(&text, false), b"#\x00\x00\x00", b"\x00\xd8\x00\x00"),
            ),
            "encoding",
        ),
        (
            "utf-8-lone-surrogate",
            swapped(text.into_bytes(), b"#", b"\xed\xa0\x80"),
            "syntax",
        ),
    ]
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// The pending file's `list` names of this test, as `rust:` keys.
fn named(list: &str) -> BTreeSet<String> {
    let text = std::fs::read_to_string(root().join(PENDING)).unwrap();
    let document: serde_json::Value = serde_json::from_str(&text).unwrap();
    document[list]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|n| n.as_str().unwrap().strip_prefix("rust:"))
        .map(str::to_owned)
        .collect()
}

/// Runs `check`; a pending key must fail, any other must pass.
fn gated(key: &str, check: impl FnOnce() + std::panic::UnwindSafe) {
    let outcome = std::panic::catch_unwind(check);
    match (outcome, named("pending").contains(key)) {
        (Ok(()), true) => panic!("{key} passes now: take rust:{key} out of {PENDING}'s pending"),
        (Err(failure), false) => std::panic::resume_unwind(failure),
        _ => {}
    }
}

#[test]
fn the_pending_list_only_shrinks() {
    let authored: BTreeSet<String> = FORMS
        .iter()
        .flat_map(|(form, _)| [format!("reader.{form}"), format!("run.{form}")])
        .collect();
    assert!(named("pending").is_subset(&named("authored")));
    assert_eq!(named("authored"), authored);
}

struct NoZones;

impl Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, UnknownZone> {
        Err(UnknownZone)
    }
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

/// A regression's gathered payload, two rows to score, as
/// `tests/predict.rs` writes one.
fn payload_text() -> String {
    r#"{"model":"kumo-relational","task":{"kind":"regression","target":{"column_name":"TARGET","dtype":"float32"},"entity_table_names":["users"]},"schema":{"instance_table":{"columns":{"instance_id":{"dtype":"int64"},"__kumo_entity_ref_0":{"dtype":"int64"}}},"related_tables":{"users":{"columns":{"user_id":{"dtype":"int64"}}}},"relationships":[{"source_columns":["__kumo_entity_ref_0"],"target_columns":["user_id"],"target_table":"users"}]},"context":{"instance_table":{"format":"arrays","columns":["instance_id","TARGET"],"rows":[[0,1e-05],[1,2.5]]}},"predict":{"instance_table":{"format":"arrays","columns":["instance_id","__kumo_entity_ref_0"],"rows":[[2,7],[3,9]]}},"output":{"fields":["prediction"]},"inference":{"run_mode":"fast"},"metadata":{"num_context_examples":2,"source_query":"PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id"}}"#
        .to_owned()
}

fn predict_run() -> Run {
    let operation = Operation::Predict(Predict {
        state: RequestState {
            version: REQUEST_STATE_VERSION,
            payload: AttachmentId(1),
            query: AttachmentId(2),
            forecast: None,
            batches: None,
            ranking: None,
        },
        model: "kumo-relational".into(),
        request_id: "sdm-0123456789ab".into(),
        outputs: vec!["prediction".to_owned()],
        default_outputs: false,
        inference_asked: vec![],
    });
    let attachments = Attachments::from([
        (
            AttachmentId(1),
            Attachment::Bytes(payload_text().into_bytes()),
        ),
        (
            AttachmentId(2),
            Attachment::Bytes(
                serde_json::to_vec("PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id")
                    .unwrap(),
            ),
        ),
    ]);
    start(
        &CompositionRoot::new(Arc::new(sdm_data::Data::new())),
        StartRun {
            contract: CONTRACT_VERSION,
            facts: facts(),
            operation,
        },
        attachments,
        Arc::new(NoZones),
    )
    .unwrap()
}

/// The run's next step once the model answered 200 with `answer`.
fn answered(answer: &[u8]) -> (Run, Result<Step, CoreError>) {
    let mut run = predict_run();
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!("a model call");
    };
    assert_eq!(call.endpoint, Endpoint::Predict);
    run.submit_model(
        call.id,
        ModelResult::Response {
            status: 200,
            content_type: "application/json".into(),
            body: answer.to_vec(),
        },
    )
    .unwrap();
    let step = run.next();
    (run, step)
}

/// What a done run gives: the endpoint's metadata as the core wrote it,
/// and the result's batches as Arrow shows them.
fn given(answer: &[u8]) -> (Vec<u8>, String) {
    let (mut run, step) = answered(answer);
    let Step::Done {
        result: OperationResult::Predict(predicted),
    } = step.unwrap()
    else {
        panic!("done");
    };
    let Some(Attachment::Bytes(metadata)) = run.take_attachment(predicted.metadata) else {
        panic!("the metadata is attached");
    };
    let Some(Attachment::Arrow(result)) = run.take_attachment(predicted.result) else {
        panic!("the result is attached");
    };
    (metadata, format!("{:?}", result.batches))
}

#[test]
fn the_answer_s_utf_8_text_is_what_each_form_is_held_to() {
    // Control: the UTF-8 text reads, with its NaN and its text past ASCII,
    // and each form writes it as its name says (its first bytes).
    let read = read_answer(ANSWER.as_bytes()).unwrap();
    let written = String::from_utf8(read.to_bytes()).unwrap();
    assert!(
        written.contains("caf\u{e9} \u{1F600} \u{4E2D}"),
        "{written}"
    );
    assert!(written.contains(r#""prediction":NaN"#), "{written}");
    let (metadata, _) = given(ANSWER.as_bytes());
    assert_eq!(
        String::from_utf8(metadata).unwrap(),
        "{\"note\":\"caf\u{e9} \u{1F600} \u{4E2D}\"}"
    );
    let starts: Vec<(&str, Vec<u8>)> = FORMS
        .iter()
        .map(|(form, write)| (*form, write(ANSWER)[..4].to_vec()))
        .collect();
    assert_eq!(
        starts,
        [
            ("utf-8", b"{\"ob".to_vec()),
            ("utf-8-sig", b"\xef\xbb\xbf{".to_vec()),
            ("utf-16-le-bom", b"\xff\xfe{\x00".to_vec()),
            ("utf-16-be-bom", b"\xfe\xff\x00{".to_vec()),
            ("utf-16-le", b"{\x00\"\x00".to_vec()),
            ("utf-16-be", b"\x00{\x00\"".to_vec()),
            ("utf-32-le-bom", b"\xff\xfe\x00\x00".to_vec()),
            ("utf-32-be-bom", b"\x00\x00\xfe\xff".to_vec()),
            ("utf-32-le", b"{\x00\x00\x00".to_vec()),
            ("utf-32-be", b"\x00\x00\x00{".to_vec()),
        ]
    );
}

#[test]
fn the_reader_reads_each_form_as_the_utf_8_text() {
    let want = read_answer(ANSWER.as_bytes()).unwrap();
    for (form, write) in FORMS {
        let bytes = write(ANSWER);
        let want = want.clone();
        gated(&format!("reader.{form}"), move || {
            let got = read_answer(&bytes).unwrap_or_else(|e| panic!("{form}: refused as {e:?}"));
            assert_eq!(got, want, "{form}");
        });
    }
}

#[test]
fn a_run_answered_in_each_form_gives_what_the_utf_8_text_gives() {
    let want = given(ANSWER.as_bytes());
    for (form, write) in FORMS {
        let bytes = write(ANSWER);
        let want = want.clone();
        gated(&format!("run.{form}"), move || {
            assert_eq!(given(&bytes), want, "{form}");
        });
    }
}

#[test]
fn bytes_no_encoding_decodes_and_lone_surrogates_are_refused_by_name() {
    for (name, bytes, category) in refused() {
        let read = read_answer(&bytes).expect_err(name);
        assert_eq!(read.category, category, "{name}: read");
        let (_, step) = answered(&bytes);
        let error = step.expect_err(name);
        assert_eq!(
            (error.kind, error.code),
            (ErrorKind::ModelEndpoint, ErrorCode::InvalidResponse),
            "{name}"
        );
        assert_eq!(error.fields["field"], Detail::text("response"), "{name}");
        assert_eq!(error.fields["reason"], Detail::text("not_json"), "{name}");
        assert_eq!(error.fields["category"], Detail::text(category), "{name}");
        let written = format!("{}{}", serde_json::to_string(&error).unwrap(), error);
        assert!(!written.contains(CANARY), "{name}: {written}");
    }
}

#[test]
fn text_shorter_than_four_bytes_is_decoded_as_python_s_reader_decodes_it() {
    // `json.detect_encoding`: two bytes, one of them zero, are UTF-16 of
    // the byte order the zero names; any other text under four bytes is
    // UTF-8.
    for (bytes, token) in [
        (&b"1\x00"[..], "1"),
        (&b"\x001"[..], "1"),
        (&b"12"[..], "12"),
        (&b"123"[..], "123"),
    ] {
        assert_eq!(
            read_answer(bytes).unwrap(),
            sdm_wire::json::Value::Number(token.to_owned()),
            "{bytes:?}"
        );
    }
}
