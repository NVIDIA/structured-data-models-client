// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable useful task and complete-reference ownership, without replay.
//! These retain actual domain rows and validation results, never a program
//! position, host object, callback, registry or a second graph index.

use super::*;

#[cfg(test)]
#[path = "owners_tests.rs"]
mod tests;

#[derive(Clone, PartialEq, Eq)]
struct Definition {
    identity: String,
    meaning: FrameTaskMeaning,
    prediction_type: FramePredictionType,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
    predictions: u64,
    per_call: u64,
    calls: u32,
}

impl Definition {
    fn of(value: &CompletedFrameTask) -> Self {
        Self {
            identity: value.identity.clone(),
            meaning: value.labelled.meaning.clone(),
            prediction_type: value.labelled.prediction_type.clone(),
            encoding: value.prediction_encoding,
            predictions: value.prediction_rows,
            per_call: value.rows_per_call,
            calls: value.calls,
        }
    }
}

struct TaskData {
    definition: Definition,
    graph: crate::RetainedFrameGraph,
    context: ArrowData,
    predicted: ArrowData,
    labelled: Arc<task::LabelledContext>,
    asked: Arc<Asked>,
    rows: Arc<TaskRows>,
}

/// Actual immutable context/prediction data and the already parsed task.
/// Core Task construction is the only native-owner factory.
#[derive(Clone)]
pub struct RetainedFrameTask {
    data: Arc<TaskData>,
}

impl std::fmt::Debug for RetainedFrameTask {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedFrameTask")
            .field("identity", &self.data.definition.identity)
            .finish_non_exhaustive()
    }
}

impl RetainedFrameTask {
    pub(super) fn new(bound: &BoundTask) -> Self {
        Self {
            data: Arc::new(TaskData {
                definition: Definition::of(&bound.value),
                graph: bound.labels.graph.retained(),
                context: bound.labels.rows.clone(),
                predicted: bound.predicted.clone(),
                labelled: Arc::clone(&bound.labels.labelled),
                asked: Arc::clone(&bound.labels.asked),
                rows: Arc::clone(&bound.rows),
            }),
        }
    }

    pub fn identity(&self) -> &str {
        &self.data.definition.identity
    }
    pub fn context_rows(&self) -> ArrowData {
        self.data.context.clone()
    }
    pub fn prediction_rows(&self) -> ArrowData {
        self.data.predicted.clone()
    }

    pub(super) fn bind(
        &self,
        graph: BoundGraph,
        value: CompletedFrameTask,
    ) -> Result<BoundTask, CoreError> {
        if Definition::of(&value) != self.data.definition
            || !graph.retained().same_owner(&self.data.graph)
        {
            return Err(protocol(
                "retained frame task no longer matches its immutable meaning or graph owner",
            ));
        }
        let labels = BoundLabels {
            graph,
            value: value.labelled.clone(),
            rows: self.data.context.clone(),
            labelled: Arc::clone(&self.data.labelled),
            asked: Arc::clone(&self.data.asked),
        };
        Ok(BoundTask {
            labels,
            value,
            predicted: self.data.predicted.clone(),
            rows: Arc::clone(&self.data.rows),
            owner: Some(self.clone()),
        })
    }
}

struct ReferencesData {
    identity: String,
    plan_identity: String,
    task: RetainedFrameTask,
    nodes: ArrayRef,
}

/// The complete node-reference result, validated before any neighbor draw.
/// Later native casts remain distinct and are resolved again per batch.
#[derive(Clone)]
pub struct RetainedFrameReferences {
    data: Arc<ReferencesData>,
}

impl std::fmt::Debug for RetainedFrameReferences {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedFrameReferences")
            .field("identity", &self.data.identity)
            .finish_non_exhaustive()
    }
}

impl RetainedFrameReferences {
    pub(super) fn new(plan: &BoundPlan, nodes: ArrayRef, identity: String) -> Self {
        Self {
            data: Arc::new(ReferencesData {
                identity,
                plan_identity: plan.value.identity.clone(),
                task: plan.task.retained(),
                nodes,
            }),
        }
    }

    pub fn identity(&self) -> &str {
        &self.data.identity
    }
    pub fn rows(&self) -> ArrowData {
        data(
            RecordBatch::try_from_iter([("node", Arc::clone(&self.data.nodes))])
                .expect("validated complete node references are one column"),
        )
    }

    pub(super) fn validate(&self, plan: &BoundPlan, identity: &str) -> Result<usize, CoreError> {
        if identity != self.data.identity
            || plan.value.identity != self.data.plan_identity
            || plan.task.value.identity != self.data.task.identity()
            || !plan
                .task
                .labels
                .graph
                .retained()
                .same_owner(&self.data.task.data.graph)
        {
            return Err(protocol(
                "retained complete frame references no longer match their task and plan",
            ));
        }
        Ok(self.data.nodes.len())
    }
}
