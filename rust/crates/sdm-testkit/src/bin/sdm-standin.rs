// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm-standin`: the stand-in provider ([`sdm_testkit::standin`]) served
//! over a pipe to the Python and Java test harnesses (Phase 3 design,
//! choice 2; `contract/runners/standin_client.py`).
//!
//! ```text
//! sdm-standin --capture DIRECTORY
//! ```
//!
//! Every message, each way, is one frame: a 4-byte big-endian length, then
//! that many bytes. The stand-in first writes a JSON frame
//! `{"ready": true, "case", "leg"}`, or `{"ready": false, "message"}` and
//! exits 2 when it cannot serve the capture. Then, per fetch step, the
//! harness writes a JSON frame `{"step": <FetchStep>, "attachments": [id,
//! ...]}` and one Arrow IPC stream frame per attachment named, in order
//! (the key sets the step binds); the stand-in answers with a JSON frame
//! `{"id", "answer", "by", "flagged"}`, `answer` one of
//!
//! - `data`, followed by one Arrow IPC stream frame of every row;
//! - `error`, with `error`, the provider's error as the contract carries it;
//! - `live_only`, with `code`, the reason (`B1`, `B2`);
//! - `refused`, with `message`: the stand-in refuses the step by name (a
//!   recorded table or draw missing, a draw asked otherwise), a runner
//!   failure for the harness, never an answer.
//!
//! A frame it cannot read is answered `refused`, and the stand-in goes on.
//! It exits 0 when its input ends.

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Result, anyhow, bail};
use sdm_contract::data::AttachmentId;
use sdm_contract::run::FetchStep;
use sdm_testkit::duck::Fetched;
use sdm_testkit::standin::{self, Answer, Capture, StandIn};
use serde_json::{Value, json};

/// The largest frame read: a step and its key sets are small.
const LARGEST: usize = 1 << 30;

fn read_frame(input: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut length = [0u8; 4];
    match input.read_exact(&mut length) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let length = u32::from_be_bytes(length) as usize;
    if length > LARGEST {
        bail!("a frame of {length} bytes is past the largest read");
    }
    let mut bytes = vec![0u8; length];
    input.read_exact(&mut bytes)?;
    Ok(Some(bytes))
}

fn write_frame(output: &mut impl Write, bytes: &[u8]) -> Result<()> {
    let length = u32::try_from(bytes.len()).map_err(|_| anyhow!("a frame past 4 GiB"))?;
    output.write_all(&length.to_be_bytes())?;
    output.write_all(bytes)?;
    Ok(())
}

fn write_json(output: &mut impl Write, value: &Value) -> Result<()> {
    write_frame(output, &serde_json::to_vec(value)?)?;
    output.flush()?;
    Ok(())
}

/// A fetch step and the key sets it binds, by attachment.
type Request = (FetchStep, BTreeMap<AttachmentId, Fetched>);

/// One request: the step and its key sets, or why it cannot be read.
fn request(input: &mut impl Read, frame: &[u8]) -> Result<std::result::Result<Request, String>> {
    let message: Value = match serde_json::from_slice(frame) {
        Ok(message) => message,
        Err(_) => return Ok(Err("the frame is not a JSON request".to_owned())),
    };
    let ids: Vec<u32> = match message.get("attachments") {
        None | Some(Value::Null) => Vec::new(),
        Some(value) => match serde_json::from_value(value.clone()) {
            Ok(ids) => ids,
            Err(_) => return Ok(Err("attachments is not a list of ids".to_owned())),
        },
    };
    // The attachments' frames are read whatever the step holds, so the
    // next request starts where it should.
    let mut keys = BTreeMap::new();
    for id in &ids {
        let Some(bytes) = read_frame(input)? else {
            bail!("the input ended inside a request's attachments");
        };
        match standin::decode(&bytes) {
            Ok(fetched) => {
                keys.insert(AttachmentId(*id), fetched);
            }
            Err(_) => return Ok(Err(format!("attachment {id} is not an Arrow IPC stream"))),
        }
    }
    let step: FetchStep = match message.get("step").cloned().map(serde_json::from_value) {
        Some(Ok(step)) => step,
        Some(Err(error)) => return Ok(Err(format!("the step is not a fetch step: {error}"))),
        None => return Ok(Err("the request holds no step".to_owned())),
    };
    Ok(Ok((step, keys)))
}

fn serve(stand_in: &StandIn, input: &mut impl Read, output: &mut impl Write) -> Result<()> {
    while let Some(frame) = read_frame(input)? {
        let (step, keys) = match request(input, &frame)? {
            Ok(found) => found,
            Err(message) => {
                write_json(output, &json!({"answer": "refused", "message": message}))?;
                continue;
            }
        };
        let id = serde_json::to_value(step.id)?;
        match stand_in.fetch(&step, &keys) {
            Err(refused) => write_json(
                output,
                &json!({"id": id, "answer": "refused", "message": refused.to_string()}),
            )?,
            Ok(answered) => {
                let head = json!({"id": id, "by": answered.by, "flagged": answered.flagged});
                match answered.answer {
                    Answer::Data(fetched) => {
                        let bytes = standin::encode(&fetched)?;
                        let mut head = head;
                        head["answer"] = json!("data");
                        write_frame(output, &serde_json::to_vec(&head)?)?;
                        write_frame(output, &bytes)?;
                        output.flush()?;
                    }
                    Answer::Error(error) => {
                        let mut head = head;
                        head["answer"] = json!("error");
                        head["error"] = serde_json::to_value(error)?;
                        write_json(output, &head)?;
                    }
                    Answer::LiveOnly(code) => {
                        let mut head = head;
                        head["answer"] = json!("live_only");
                        head["code"] = json!(code);
                        write_json(output, &head)?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let mut arguments = std::env::args().skip(1);
    let capture = match (
        arguments.next().as_deref(),
        arguments.next(),
        arguments.next(),
    ) {
        (Some("--capture"), Some(path), None) => PathBuf::from(path),
        _ => {
            eprintln!("usage: sdm-standin --capture DIRECTORY");
            return ExitCode::from(2);
        }
    };
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let stand_in = match Capture::read(&capture).and_then(StandIn::open) {
        Ok(stand_in) => stand_in,
        Err(error) => {
            let _ = write_json(
                &mut output,
                &json!({"ready": false, "message": format!("{error:#}")}),
            );
            return ExitCode::from(2);
        }
    };
    let ready = json!({
        "ready": true,
        "case": stand_in.capture().case,
        "leg": stand_in.capture().leg,
    });
    if write_json(&mut output, &ready).is_err() {
        return ExitCode::from(2);
    }
    let stdin = io::stdin();
    let mut input = stdin.lock();
    match serve(&stand_in, &mut input, &mut output) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sdm-standin: {error:#}");
            ExitCode::from(1)
        }
    }
}
