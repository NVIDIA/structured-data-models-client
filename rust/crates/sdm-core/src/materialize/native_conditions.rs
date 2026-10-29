// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Actual completed condition column views and existing mathematical truth.
//! The history is retained by its source/group owner, never an action cursor.

use super::native_target::{RetainedLabelInputs, RetainedTargetColumn};
use super::*;
use crate::run::Attachments;
use sdm_contract::materialize::{CompletedConditionValue, PreparedConditionValues};

pub(super) struct Context {
    pub(super) entity: ColumnRef,
    pub(super) filter: Option<Shape>,
    pub(super) what_if: Vec<Shape>,
}

pub(super) fn entity_condition(column: String, filter: Option<Shape>) -> Shape {
    // Presence is complete Boolean data, not an untested query column.
    // Use the existing predicate kernel and retain its exact false rows.
    let key = Shape::Condition {
        target: Box::new(Shape::Column(column)),
        test: Test::Compare(labels::RelOp::Eq, Some(labels::Constant::Bool(true))),
    };
    match filter {
        Some(filter) => Shape::Both {
            op: labels::BoolOp::And,
            left: Box::new(key),
            right: Box::new(filter),
        },
        None => key,
    }
}

#[derive(Clone)]
pub(super) struct Conditions {
    shape: Shape,
    views: Arc<Vec<RetainedLabelInputs>>,
    values: Vec<RetainedTargetColumn>,
    truth: native_target::conditions::ConditionInputs,
    entity_key: bool,
}

impl Conditions {
    pub(super) fn entity(input: &RetainedLabelInputs) -> Result<Self, CoreError> {
        let context = input.data.root.condition_context()?;
        let shape = entity_condition(context.entity.column, context.filter);
        Self::new(input, shape, true)
    }

    pub(super) fn what_if(input: &RetainedLabelInputs) -> Result<Option<Self>, CoreError> {
        let context = input.data.root.condition_context()?;
        let mut shapes = context.what_if.into_iter();
        let Some(mut shape) = shapes.next() else {
            return Ok(None);
        };
        for right in shapes {
            shape = Shape::Both {
                op: labels::BoolOp::And,
                left: Box::new(shape),
                right: Box::new(right),
            };
        }
        Self::new(input, shape, false).map(Some)
    }

    fn new(input: &RetainedLabelInputs, shape: Shape, entity_key: bool) -> Result<Self, CoreError> {
        let targets = native_target::conditions::operand_shapes(&shape)?;
        let views = targets
            .into_iter()
            .map(|shape| input.view(shape.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            shape,
            views: Arc::new(views),
            values: Vec::new(),
            truth: Default::default(),
            entity_key,
        })
    }

    pub(super) fn selection(&self) -> Option<&[bool]> {
        self.truth.truth().map(|truth| truth.values.as_slice())
    }

    pub(super) fn complete(&self, column: RetainedTargetColumn) -> Result<Self, CoreError> {
        let expected = self
            .views
            .get(self.values.len())
            .ok_or_else(|| invalid("native condition adds an extra occurrence"))?;
        if !Arc::ptr_eq(&expected.data, &column.input.data) {
            return Err(invalid(
                "native condition value belongs to another exact column view",
            ));
        }
        let target = if self.entity_key && self.values.is_empty() {
            labels::Target::Column(labels::Column::Bool(
                column.data.presence().iter().map(|&v| Some(v)).collect(),
            ))
        } else {
            column.data.condition_target()?
        };
        let truth = self.truth.complete_target(&self.shape, target)?;
        let mut values = self.values.clone();
        values.push(column);
        Ok(Self {
            shape: self.shape.clone(),
            views: Arc::clone(&self.views),
            values,
            truth,
            entity_key: self.entity_key,
        })
    }

    pub(super) fn publish(&self, asker: &Asker) -> Result<PreparedConditionValues, CoreError> {
        Ok(PreparedConditionValues {
            columns: self
                .views
                .iter()
                .map(|view| view.publish(asker))
                .collect::<Result<Vec<_>, _>>()?,
            values: self
                .values
                .iter()
                .map(|value| {
                    Ok(CompletedConditionValue {
                        column: value.input.data.meaning.target.clone(),
                        owner: asker.attach(Attachment::TargetColumn(value.clone()))?,
                    })
                })
                .collect::<Result<Vec<_>, CoreError>>()?,
        })
    }

    pub(super) fn matches(
        &self,
        value: PreparedConditionValues,
        attachments: &mut Attachments,
    ) -> Result<(), CoreError> {
        if value.columns.len() != self.views.len() || value.values.len() != self.values.len() {
            return Err(invalid(
                "native condition history changes its complete occurrence prefix",
            ));
        }
        for (actual, expected) in value.columns.into_iter().zip(self.views.iter()) {
            let actual = RetainedLabelInputs::bind(actual, attachments)?;
            if !Arc::ptr_eq(&actual.data, &expected.data) {
                return Err(invalid("native condition names a foreign source view"));
            }
        }
        for (actual, expected) in value.values.into_iter().zip(&self.values) {
            let Some(Attachment::TargetColumn(owner)) = attachments.remove(&actual.owner) else {
                return Err(invalid(
                    "native condition lacks its actual completed column",
                ));
            };
            if actual.column != expected.input.data.meaning.target
                || !Arc::ptr_eq(&owner.data, &expected.data)
                || !Arc::ptr_eq(&owner.input.data, &expected.input.data)
            {
                return Err(invalid(
                    "native condition rewrites its actual completed values or source",
                ));
            }
        }
        Ok(())
    }
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

#[derive(Clone)]
pub(super) struct RestoredGroups {
    pub(super) labels: Arc<labels::Labels>,
    pub(super) rows: RecordBatch,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
    conditions: Option<Conditions>,
}

impl RestoredGroups {
    pub(super) fn new(
        input: &RetainedLabelInputs,
        labels: labels::Labels,
        rows: RecordBatch,
        encoding: Option<sdm_contract::data::strings::StringEncoding>,
    ) -> Result<Self, CoreError> {
        Ok(Self {
            labels: Arc::new(labels),
            rows,
            encoding,
            conditions: Conditions::what_if(input)?,
        })
    }

    pub(super) fn publish(
        &self,
        asker: &Asker,
    ) -> Result<
        (
            sdm_contract::frames::materialize::RestoredFrameLabels,
            Option<PreparedConditionValues>,
        ),
        CoreError,
    > {
        let rows = asker.attach(Attachment::Arrow(ArrowData::new(
            self.rows.schema(),
            vec![self.rows.clone()],
        )))?;
        Ok((
            sdm_contract::frames::materialize::RestoredFrameLabels {
                rows,
                encoding: self.encoding,
            },
            self.conditions
                .as_ref()
                .map(|conditions| conditions.publish(asker))
                .transpose()?,
        ))
    }

    pub(super) fn matches(
        &self,
        restored: Option<sdm_contract::frames::materialize::RestoredFrameLabels>,
        conditions: Option<PreparedConditionValues>,
        attachments: &mut Attachments,
    ) -> Result<(), CoreError> {
        let restored =
            restored.ok_or_else(|| invalid("native groups drop their actual restored values"))?;
        let Some(Attachment::Arrow(rows)) = attachments.remove(&restored.rows) else {
            return Err(invalid("native restored groups lack their actual rows"));
        };
        if restored.encoding != self.encoding || one_batch(&rows)? != self.rows {
            return Err(invalid(
                "native group history rewrites its actual restored rows or encoding",
            ));
        }
        match (&self.conditions, conditions) {
            (None, None) => Ok(()),
            (Some(expected), Some(actual)) => expected.matches(actual, attachments),
            _ => Err(invalid(
                "native groups drop or invent their required what-if history",
            )),
        }
    }

    pub(super) fn complete(&self, column: RetainedTargetColumn) -> Result<Self, CoreError> {
        let before = self
            .conditions
            .as_ref()
            .ok_or_else(|| invalid("native restored groups have no what-if condition"))?;
        Ok(Self {
            labels: Arc::clone(&self.labels),
            rows: self.rows.clone(),
            encoding: self.encoding,
            conditions: Some(before.complete(column)?),
        })
    }

    pub(super) fn selection(&self) -> Result<Option<&[bool]>, CoreError> {
        self.conditions
            .as_ref()
            .map(|conditions| {
                conditions
                    .selection()
                    .ok_or_else(|| invalid("native batch precedes complete what-if observations"))
            })
            .transpose()
    }
}

pub(super) fn eligibility(
    asker: &Asker,
    inputs: sdm_contract::materialize::PreparedLabelInputs,
    column: sdm_contract::materialize::PreparedTargetColumn,
    attachments: &mut Attachments,
) -> Result<sdm_contract::materialize::PreparedLabelInputs, CoreError> {
    let input = RetainedLabelInputs::bind(inputs, attachments)?;
    let column = RetainedTargetColumn::bind(column, attachments)?;
    let before = input
        .data
        .eligibility
        .as_ref()
        .ok_or_else(|| invalid("this column view has no entity eligibility history"))?;
    let eligibility = before.complete(column)?;
    input.with_eligibility(eligibility).publish(asker)
}
