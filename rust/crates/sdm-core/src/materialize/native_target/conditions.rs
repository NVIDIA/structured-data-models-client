// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Completed mathematical predicate data over actual native column occurrences.
//! The existing validated Shape and label kernels retain all expression policy.

use super::*;

#[cfg(test)]
mod tests;

#[derive(Clone, Default)]
pub(in crate::materialize) struct ConditionInputs {
    leaves: Vec<Arc<labels::Truth>>,
    /// Actual completed subtree masks, keyed by their structural expression path.
    masks: BTreeMap<Vec<u8>, Arc<labels::Truth>>,
}

impl ConditionInputs {
    pub(in crate::materialize) fn completed(&self) -> usize {
        self.leaves.len()
    }

    pub(in crate::materialize) fn truth(&self) -> Option<&Arc<labels::Truth>> {
        self.masks.get(&Vec::new())
    }

    pub(in crate::materialize) fn complete(
        &self,
        filter: &Shape,
        values: labels::Column,
    ) -> Result<Self, CoreError> {
        self.complete_target(filter, labels::Target::Column(values))
    }

    pub(in crate::materialize) fn complete_target(
        &self,
        filter: &Shape,
        target: labels::Target,
    ) -> Result<Self, CoreError> {
        let operands = operands(filter)?;
        let operand = operands.get(self.completed()).ok_or_else(|| {
            invalid_target("condition completion adds a foreign or repeated occurrence")
        })?;
        let target = match operand.test {
            Some(test) => tested(target, test),
            None => target,
        };
        // Finish this actual left condition before the caller can convert
        // another native operand. A false value does not suppress that read.
        let truth = labels::truth(&target)?;
        let mut completed = self.clone();
        completed.leaves.push(Arc::new(truth));
        compose(
            filter,
            &completed.leaves,
            &mut 0,
            &mut Vec::new(),
            &mut completed.masks,
        )?;
        Ok(completed)
    }
}

struct Operand<'a> {
    target: &'a Shape,
    test: Option<&'a Test>,
}

fn operands(filter: &Shape) -> Result<Vec<Operand<'_>>, CoreError> {
    fn collect<'a>(shape: &'a Shape, into: &mut Vec<Operand<'a>>) -> Result<(), CoreError> {
        match shape {
            Shape::Column(_) | Shape::Aggregated(_) => into.push(Operand {
                target: shape,
                test: None,
            }),
            Shape::Condition { target, test } => {
                into.push(Operand {
                    target,
                    test: Some(test),
                });
            }
            Shape::Not(inner) => collect(inner, into)?,
            Shape::Both { left, right, .. } => {
                collect(left, into)?;
                collect(right, into)?;
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    collect(filter, &mut found)?;
    Ok(found)
}

pub(in crate::materialize) fn operand_columns(filter: &Shape) -> Result<Vec<&str>, CoreError> {
    operands(filter)?
        .into_iter()
        .map(|operand| match operand.target {
            Shape::Column(column) => Ok(column.as_str()),
            _ => Err(unvalidated("a non-column in a row condition")),
        })
        .collect()
}

pub(in crate::materialize) fn operand_shapes(filter: &Shape) -> Result<Vec<&Shape>, CoreError> {
    Ok(operands(filter)?
        .into_iter()
        .map(|operand| operand.target)
        .collect())
}

fn compose(
    shape: &Shape,
    leaves: &[Arc<labels::Truth>],
    at: &mut usize,
    path: &mut Vec<u8>,
    masks: &mut BTreeMap<Vec<u8>, Arc<labels::Truth>>,
) -> Result<Option<Arc<labels::Truth>>, CoreError> {
    if let Some(truth) = masks.get(path) {
        *at += operands(shape)?.len();
        return Ok(Some(Arc::clone(truth)));
    }
    let truth = match shape {
        Shape::Column(_) | Shape::Aggregated(_) | Shape::Condition { .. } => {
            let truth = leaves.get(*at).cloned();
            *at += 1;
            truth
        }
        Shape::Not(inner) => {
            path.push(0);
            let child = compose(inner, leaves, at, path, masks)?;
            path.pop();
            child.map(|truth| Arc::new(labels::negate_truth(&truth)))
        }
        Shape::Both { op, left, right } => {
            path.push(0);
            let left = compose(left, leaves, at, path, masks)?;
            path.pop();
            let Some(left) = left else {
                return Ok(None);
            };
            path.push(1);
            let right = compose(right, leaves, at, path, masks)?;
            path.pop();
            right
                .map(|right| labels::combine_truth(*op, &left, &right).map(Arc::new))
                .transpose()?
        }
    };
    if let Some(truth) = &truth {
        masks.insert(path.clone(), Arc::clone(truth));
    }
    Ok(truth)
}
