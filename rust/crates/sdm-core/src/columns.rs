// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context-analysis operation and lifecycle adapter; no host computations.
use sdm_contract::{CoreError, columns::AnalyzeColumns, ops::OperationResult};
use crate::run::{Asker,Attachment,Attachments,internal};

pub(crate) fn run(asker:&Asker,input:AnalyzeColumns,mut attached:Attachments)->Result<OperationResult,CoreError>{
    if input.payloads.len()!=1{return Err(internal("column analysis expects one normalized context"));}
    let Some(Attachment::Bytes(bytes))=attached.remove(&input.payloads[0])else{return Err(internal("column analysis payload is not bytes"));};
    let payload=crate::predict::payload(&bytes)?;
    let result=sdm_request::columns::analyze(&payload)?;
    crate::direct::value(asker,&result)
}
