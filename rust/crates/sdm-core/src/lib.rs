// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The single Run state machine, operation dispatch, the composition root,
//! model capability checks and profiles, and the PQL adapter.

#![forbid(unsafe_code)]

pub mod attached;
pub mod big_integer;
pub mod compose;
mod describe;
mod discover;
pub mod dispatch;
mod edit;
pub mod exact_number;
mod fallback;
mod frame_materialize;
mod frames;
mod held;
mod lag;
pub mod materialize;
mod pages;
pub mod pql;
mod predict;
mod prepare;
mod prepare_response;
pub mod profiles;
mod ranking;
pub mod run;
mod subgraph;
mod tabular;
#[cfg(feature = "test-ops")]
mod test_ops;
pub mod view;
pub mod view_steps;
pub mod view_words;

pub use compose::{CompositionRoot, DataAccess};
pub use dispatch::start;
pub use edit::{dtype_names, meanings, wire_names};
pub use frame_materialize::RetainedFrameGraph;
pub use materialize::frame_values::{
    RetainedFrameLabelBatch, RetainedFrameLabelGroups, RetainedFramePayload,
    RetainedFramePopulation, RetainedFrameQuestion, RetainedFrameReferences, RetainedFrameTask,
};
pub use materialize::native_lookup::RetainedLookupRows;
pub use materialize::native_target::{RetainedLabelInputs, RetainedTargetColumn};
pub use materialize::source_values::{
    RetainedMaterializeBatch, RetainedMaterializeGroups, RetainedMaterializePopulation,
    RetainedMaterializeSource,
};
/// The text a float is spelled in, as the reference's language writes it
/// (Python's `repr`): for a host that hands the data layer a value to
/// write as the reference does (T3.12 fix 1).
pub use pql::float_text;
pub use run::{
    ArrowData, Ask, Asker, Attachment, AttachmentIds, Attachments, FetchResult, ModelResult, Reply,
    Run, RunState,
};
/// The zones port a host implements and passes to [`start`]; re-exported
/// so a host reaches it through the core, never through an edge to
/// `sdm-engine` (decision E30).
pub use sdm_engine::time::{UnknownZone, Zones};
/// What each model supports, as the client describes it to a caller
/// (`capabilities()`), re-exported so a host reaches the request crate's
/// table through the core (decision E30).
pub use sdm_request::predict::{Capabilities, capabilities};
pub use sdm_schema::frames::key_adoption::Adoption as FrameKeyAdoption;
pub use sdm_schema::frames::key_advice::{Advice as FrameKeyAdvice, sample as native_frame_sample};
pub use sdm_schema::frames::key_candidates::Candidates as FrameKeyCandidates;
pub use sdm_schema::frames::object_infer::check_cell as check_frame_cell;
/// A fixed native constructor source traversal, shared by all bindings.
pub use sdm_schema::frames::object_infer::{
    Constructor as FrameObjectConstructor, Content as FrameObjectContent, Read as FrameObjectRead,
};
/// Owned logical payload data, before JSON escaping or native presentation.
pub use sdm_wire::json::Value as LogicalValue;
pub use tabular::{
    TabularContext, TabularIndexSelection, TabularPopulation, TabularScoringKeys,
    tabular_labelled_rows,
};

/// Finite native sample facts complete one useful physical column meaning.
pub use sdm_schema::frames::native_column::{
    Constructor as FrameNativeConstructor, dtype as native_constructed_dtype,
};

/// Admit the fixed original physical declaration before semantic observations.
pub fn native_column_constructor(
    table: &str,
    name: String,
    values: &sdm_contract::frames::materialize::FrameConstructorValues,
    declared: Option<&sdm_contract::frames::materialize::FramePhysicalDeclaration>,
) -> Result<FrameNativeConstructor, sdm_contract::CoreError> {
    use sdm_contract::frames::materialize::{FrameConstructorValues, FramePhysicalDeclaration};
    if let Some(declared) = declared {
        frame_materialize::values::check_declaration(declared)?;
        if *declared == FramePhysicalDeclaration::LargeUtf8 {
            return Err(frame_materialize::values::unsupported_large_string(
                table, &name,
            ));
        }
        if let FramePhysicalDeclaration::Physical { physical } = declared
            && *values
                != (FrameConstructorValues::Typed {
                    physical: physical.clone(),
                })
        {
            return Err(sdm_contract::CoreError::protocol(
                sdm_contract::Sentence::InvalidMessage,
                "native constructor declaration contradicts its observed storage",
            ));
        }
    }
    Ok(FrameNativeConstructor::new(
        name,
        native_constructed_dtype(values)?,
    ))
}
