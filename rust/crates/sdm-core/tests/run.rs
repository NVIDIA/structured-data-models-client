// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Every state of the Run state machine and every illegal transition.

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::Poll;

use arrow_array::RecordBatch;
use arrow_schema::Schema;
use sdm_contract::data::{
    DataRequest, DescribePart, Determinism, ProviderCapabilities, ProviderChoice, ProviderError,
};
use sdm_contract::ops::{OperationResult, QueryValidation};
use sdm_contract::run::{Endpoint, ReplyId};
use sdm_contract::{CoreError, CoreWarning, ErrorCode, ErrorKind, Step};
use sdm_core::{
    ArrowData, Asker, Attachment, CompositionRoot, FetchResult, ModelResult, Reply, Run, RunState,
};

fn describe(table: &str) -> DataRequest {
    DataRequest::Describe {
        tables: vec![table.to_owned()],
        part: DescribePart::Columns,
    }
}

/// A run whose data requests a host provider answers, through `sdm-data`'s
/// data layer.
fn data_run<F>(make: impl FnOnce(Asker) -> F) -> Run
where
    F: Future<Output = Result<OperationResult, CoreError>> + Send + 'static,
{
    let root = CompositionRoot::new(Arc::new(sdm_data::Data::new()));
    let host = ProviderChoice::Host {
        capabilities: ProviderCapabilities {
            optional: BTreeSet::new(),
            max_keys_per_request: None,
            sampling: Determinism::Unseeded,
        },
    };
    Run::with_data(root.data_access(&host).unwrap(), make)
}

fn rows(n: usize) -> FetchResult {
    let schema = Arc::new(Schema::empty());
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(&schema),
        vec![],
        &arrow_array::RecordBatchOptions::new().with_row_count(Some(n)),
    )
    .unwrap();
    FetchResult::Data(ArrowData::new(schema, vec![batch]))
}

fn failed() -> FetchResult {
    FetchResult::Error(ProviderError {
        kind: ErrorKind::DataSource,
        retryable: false,
        sqlstate: Some("42P01".into()),
        vendor_code: None,
    })
}

fn answered(status: u16, body: &[u8]) -> ModelResult {
    ModelResult::Response {
        status,
        content_type: "application/json".into(),
        body: body.to_vec(),
    }
}

fn row_count(reply: &Reply) -> usize {
    match reply {
        Reply::Fetch(FetchResult::Data(data)) => data.batches.iter().map(|b| b.num_rows()).sum(),
        _ => usize::MAX,
    }
}

fn warning(code: &str) -> CoreWarning {
    CoreWarning {
        code: code.into(),
        message: String::new(),
        fields: Default::default(),
    }
}

/// A result carrying what the body saw, as diagnostic codes.
fn done(seen: Vec<String>) -> OperationResult {
    OperationResult::ValidateQuery(sdm_contract::ops::QueryValidation {
        accepted: true,
        diagnostics: seen
            .into_iter()
            .map(|code| sdm_contract::ops::PqlDiagnostic {
                code,
                message: String::new(),
                severity: sdm_contract::ops::Severity::Warning,
                span: None,
            })
            .collect(),
    })
}

/// Two fetches awaited together, then one model call, then the result.
fn scripted() -> Run {
    data_run(|asker: Asker| async move {
        asker.warn(warning("FIRST"));
        let a = asker.fetch(describe("a"))?;
        let b = asker.fetch(describe("b"))?;
        let (a, b) = join(a, b).await;
        asker.warn(warning("SECOND"));
        let answer = asker
            .call_model(
                Endpoint::Predict,
                "application/json".into(),
                format!("[{},{}]", row_count(&a), row_count(&b)).into_bytes(),
            )
            .await;
        asker.warn(warning("THIRD"));
        let Reply::Model(ModelResult::Response { status, body, .. }) = answer else {
            return Err(CoreError::new(
                ErrorKind::ModelEndpoint,
                ErrorCode::TransportError,
                "no answer",
            ));
        };
        Ok(done(vec![
            status.to_string(),
            String::from_utf8(body).unwrap(),
        ]))
    })
}

async fn join<A: Future + Unpin, B: Future + Unpin>(mut a: A, mut b: B) -> (A::Output, B::Output) {
    let (mut ra, mut rb) = (None, None);
    std::future::poll_fn(|cx| {
        if ra.is_none()
            && let Poll::Ready(v) = Pin::new(&mut a).poll(cx)
        {
            ra = Some(v);
        }
        if rb.is_none()
            && let Poll::Ready(v) = Pin::new(&mut b).poll(cx)
        {
            rb = Some(v);
        }
        if ra.is_some() && rb.is_some() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;
    (ra.unwrap(), rb.unwrap())
}

fn codes_of(warnings: Vec<CoreWarning>) -> Vec<String> {
    warnings.into_iter().map(|w| w.code).collect()
}

fn fetch_ids(step: &Step) -> Vec<ReplyId> {
    match step {
        Step::Fetch { requests } => requests.iter().map(|r| r.id).collect(),
        other => panic!("expected a fetch, got {other:?}"),
    }
}

#[test]
fn a_scripted_operation_runs_to_done() {
    let mut run = scripted();
    assert_eq!(run.state(), RunState::Ready);

    let step = run.next().unwrap();
    let ids = fetch_ids(&step);
    assert_eq!(ids.len(), 2, "fetches awaited together are one step");
    assert_eq!(run.state(), RunState::AwaitingFetch(ids.clone()));
    assert_eq!(codes_of(run.take_warnings()), ["FIRST"]);

    // Replies may come in any order.
    run.submit_fetch(ids[1], rows(3)).unwrap();
    assert_eq!(run.state(), RunState::AwaitingFetch(vec![ids[0]]));
    run.submit_fetch(ids[0], rows(2)).unwrap();
    assert_eq!(run.state(), RunState::Ready);

    let Step::CallModel(call) = run.next().unwrap() else {
        panic!("expected a model call")
    };
    assert_eq!(run.state(), RunState::AwaitingModel(call.id));
    assert_eq!(codes_of(run.take_warnings()), ["SECOND"]);
    let Some(Attachment::Bytes(body)) = run.take_attachment(call.body) else {
        panic!("the body is an attachment")
    };
    assert_eq!(body, b"[2,3]");
    assert!(run.take_attachment(call.body).is_none(), "taken once");

    run.submit_model(call.id, answered(200, b"ok")).unwrap();
    assert_eq!(run.state(), RunState::Ready);
    let Step::Done { result } = run.next().unwrap() else {
        panic!("expected done")
    };
    assert_eq!(result, done(vec!["200".into(), "ok".into()]));
    assert_eq!(run.state(), RunState::Finished);
    assert_eq!(codes_of(run.take_warnings()), ["THIRD"]);
}

#[test]
fn a_finished_run_refuses_every_call() {
    let mut run = data_run(|_| async {
        Ok(OperationResult::ValidateQuery(QueryValidation {
            accepted: true,
            diagnostics: vec![],
        }))
    });
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
    for err in [
        run.next().unwrap_err(),
        run.submit_fetch(ReplyId(1), rows(0)).unwrap_err(),
        run.submit_model(ReplyId(1), answered(200, b""))
            .unwrap_err(),
        run.cancel().unwrap_err(),
    ] {
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::RunFinished)
        );
    }
    assert_eq!(run.state(), RunState::Finished);
}

#[test]
fn a_ready_run_refuses_replies() {
    let mut run = scripted();
    let err = run.submit_fetch(ReplyId(1), rows(0)).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply);
    let err = run
        .submit_model(ReplyId(1), answered(200, b""))
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply);
    assert_eq!(run.state(), RunState::Ready);
    // And it still runs.
    fetch_ids(&run.next().unwrap());
}

#[test]
fn awaiting_a_fetch_refuses_next_and_stray_replies() {
    let mut run = scripted();
    let ids = fetch_ids(&run.next().unwrap());

    let err = run.next().unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Protocol, ErrorCode::ReplyMissing)
    );

    let err = run.submit_fetch(ReplyId(99), rows(0)).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply, "an unknown id");

    let err = run.submit_model(ids[0], answered(200, b"")).unwrap_err();
    assert_eq!(err.code, ErrorCode::WrongReply, "a model reply to a fetch");

    run.submit_fetch(ids[0], rows(1)).unwrap();
    let err = run.submit_fetch(ids[0], rows(1)).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply, "a duplicate reply");

    let err = run.next().unwrap_err();
    assert_eq!(
        err.code,
        ErrorCode::ReplyMissing,
        "one fetch still unanswered"
    );
    assert_eq!(run.state(), RunState::AwaitingFetch(vec![ids[1]]));
}

#[test]
fn awaiting_a_model_call_refuses_stale_and_wrong_replies() {
    let mut run = scripted();
    let ids = fetch_ids(&run.next().unwrap());
    for id in &ids {
        run.submit_fetch(*id, rows(0)).unwrap();
    }
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!()
    };

    let err = run.submit_fetch(ids[0], rows(0)).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply, "a stale fetch id");
    let err = run.submit_fetch(call.id, rows(0)).unwrap_err();
    assert_eq!(
        err.code,
        ErrorCode::WrongReply,
        "a fetch reply to a model call"
    );
    let err = run.next().unwrap_err();
    assert_eq!(err.code, ErrorCode::ReplyMissing);

    run.submit_model(call.id, answered(500, b"down")).unwrap();
    let err = run.submit_model(call.id, answered(200, b"")).unwrap_err();
    assert_eq!(err.code, ErrorCode::UnknownReply, "a duplicate model reply");
    assert!(matches!(run.next().unwrap(), Step::Done { .. }));
}

/// A run whose body notes when it is dropped.
fn droppable(dropped: Arc<AtomicBool>) -> Run {
    struct Flag(Arc<AtomicBool>);
    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    data_run(move |asker: Asker| async move {
        let _flag = Flag(dropped);
        asker.fetch(describe("a"))?.await;
        Ok(done(vec![]))
    })
}

#[test]
fn cancelling_while_awaiting_a_fetch_drops_the_body() {
    let dropped = Arc::new(AtomicBool::new(false));
    let mut run = droppable(Arc::clone(&dropped));
    let ids = fetch_ids(&run.next().unwrap());
    run.cancel().unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(run.state(), RunState::Cancelled);
    for err in [
        run.next().unwrap_err(),
        run.submit_fetch(ids[0], rows(0)).unwrap_err(),
        run.submit_model(ids[0], answered(200, b"")).unwrap_err(),
        run.cancel().unwrap_err(),
    ] {
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Cancelled, ErrorCode::RunCancelled)
        );
    }
}

#[test]
fn cancelling_before_the_first_step_or_while_awaiting_a_model_call() {
    let mut run = scripted();
    run.cancel().unwrap();
    assert_eq!(run.next().unwrap_err().code, ErrorCode::RunCancelled);

    let mut run = scripted();
    for id in fetch_ids(&run.next().unwrap()) {
        run.submit_fetch(id, rows(0)).unwrap();
    }
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!()
    };
    run.cancel().unwrap();
    assert!(
        run.take_attachment(call.body).is_none(),
        "attachments go with the run"
    );
    assert_eq!(
        run.submit_model(call.id, answered(200, b""))
            .unwrap_err()
            .code,
        ErrorCode::RunCancelled
    );
}

#[test]
fn the_host_can_end_a_run_it_cannot_answer() {
    let dropped = Arc::new(AtomicBool::new(false));
    let mut run = droppable(Arc::clone(&dropped));
    let ids = fetch_ids(&run.next().unwrap());
    run.end().unwrap();
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(run.state(), RunState::Finished);
    for err in [
        run.next().unwrap_err(),
        run.submit_fetch(ids[0], rows(0)).unwrap_err(),
        run.end().unwrap_err(),
    ] {
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::RunFinished)
        );
    }
    // The attachments the run made stay until the host takes them.
    let mut run = scripted();
    for id in fetch_ids(&run.next().unwrap()) {
        run.submit_fetch(id, rows(0)).unwrap();
    }
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!()
    };
    run.end().unwrap();
    assert!(run.take_attachment(call.body).is_some());
}

#[test]
fn an_operation_error_finishes_the_run() {
    let mut run = data_run(|asker: Asker| async move {
        let reply = asker.fetch(describe("a"))?.await;
        match reply {
            Reply::Fetch(FetchResult::Error(e)) => Err(CoreError::provider(&e)),
            _ => Ok(done(vec![])),
        }
    });
    let ids = fetch_ids(&run.next().unwrap());
    run.submit_fetch(ids[0], failed()).unwrap();
    let err = run.next().unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::DataSource, ErrorCode::ProviderError)
    );
    assert_eq!(run.state(), RunState::Finished);
    assert_eq!(run.next().unwrap_err().code, ErrorCode::RunFinished);
}

#[test]
fn a_panicking_body_poisons_the_run() {
    let mut run = data_run(|asker: Asker| async move {
        asker.fetch(describe("a"))?.await;
        panic!("a defect");
    });
    let ids = fetch_ids(&run.next().unwrap());
    run.submit_fetch(ids[0], rows(0)).unwrap();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run.next()));
    assert!(panicked.is_err());
    assert_eq!(run.state(), RunState::Poisoned);
    for err in [
        run.next().unwrap_err(),
        run.submit_fetch(ids[0], rows(0)).unwrap_err(),
        run.submit_model(ids[0], answered(200, b"")).unwrap_err(),
        run.cancel().unwrap_err(),
    ] {
        assert_eq!(
            (err.kind, err.code),
            (ErrorKind::Protocol, ErrorCode::RunPoisoned)
        );
    }
}

#[test]
fn the_host_can_poison_a_run() {
    let mut run = scripted();
    fetch_ids(&run.next().unwrap());
    run.poison();
    assert_eq!(run.state(), RunState::Poisoned);
    assert_eq!(run.next().unwrap_err().code, ErrorCode::RunPoisoned);
}

#[test]
fn a_body_that_waits_on_nothing_or_on_too_much_is_an_internal_error() {
    let mut run = data_run(|_| async {
        std::future::pending::<()>().await;
        Ok(done(vec![]))
    });
    let err = run.next().unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::Internal, ErrorCode::InternalError)
    );
    assert_eq!(run.state(), RunState::Finished);

    let mut run = data_run(|asker: Asker| async move {
        let a = asker.fetch(describe("a"))?;
        let b = asker.call_model(Endpoint::Predict, "application/json".into(), vec![]);
        join(a, b).await;
        Ok(done(vec![]))
    });
    let err = run.next().unwrap_err();
    assert_eq!(err.kind, ErrorKind::Internal);
    assert_eq!(run.state(), RunState::Finished);
}

#[test]
fn each_part_of_the_body_runs_once_however_many_steps() {
    let resumed = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&resumed);
    let mut run = data_run(move |asker: Asker| async move {
        let mut seen = 0;
        for i in 0..200 {
            counter.fetch_add(1, Ordering::SeqCst);
            let reply = asker.fetch(describe(&format!("t{i}")))?.await;
            seen += row_count(&reply);
        }
        Ok(done(vec![seen.to_string()]))
    });
    let mut steps = 0;
    let result = loop {
        match run.next().unwrap() {
            Step::Done { result } => break result,
            step => {
                steps += 1;
                for id in fetch_ids(&step) {
                    run.submit_fetch(id, rows(1)).unwrap();
                }
            }
        }
    };
    assert_eq!(steps, 200);
    assert_eq!(resumed.load(Ordering::SeqCst), 200);
    assert_eq!(result, done(vec!["200".into()]));
}

#[test]
fn a_transport_error_reaches_the_body() {
    let mut run = data_run(|asker: Asker| async move {
        match asker
            .call_model(Endpoint::Predict, "application/json".into(), b"{}".to_vec())
            .await
        {
            Reply::Model(ModelResult::TransportError { retryable, .. }) => {
                Ok(done(vec![retryable.to_string()]))
            }
            _ => Ok(done(vec![])),
        }
    });
    let Step::CallModel(call) = run.next().unwrap() else {
        panic!()
    };
    run.submit_model(
        call.id,
        ModelResult::TransportError {
            kind: ErrorKind::ModelEndpoint,
            retryable: true,
        },
    )
    .unwrap();
    let Step::Done { result } = run.next().unwrap() else {
        panic!()
    };
    assert_eq!(result, done(vec!["true".into()]));
}

#[test]
fn a_key_set_travels_as_an_attachment() {
    let mut run = data_run(|asker: Asker| async move {
        let keys = asker.attach(Attachment::Bytes(vec![1, 2, 3]))?;
        asker
            .fetch(DataRequest::Describe {
                tables: vec![keys.0.to_string()],
                part: DescribePart::Columns,
            })?
            .await;
        Ok(done(vec![]))
    });
    let Step::Fetch { requests } = run.next().unwrap() else {
        panic!()
    };
    let DataRequest::Describe { tables, .. } = &requests[0].request else {
        panic!()
    };
    let id = sdm_contract::data::AttachmentId(tables[0].parse().unwrap());
    assert!(matches!(run.take_attachment(id), Some(Attachment::Bytes(b)) if b == [1, 2, 3]));
}
