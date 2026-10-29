// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-table row selection over admitted primitive entity and validity facts.
//! Native candidate labels are input facts for sampled-subset ordering.

use std::collections::{HashMap, HashSet};

use sdm_contract::{CoreError, Detail, Scalar, Sentence};

use crate::random::Mt19937;

/// Scored source rows, or missing ordinals in the original requested sequence.
/// Missing values remain request ordinals so the host presents original values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSelection {
    pub scored: Vec<usize>,
    pub missing: Vec<usize>,
}

impl IndexSelection {
    /// Integer keys preserve Python dictionary equality, including repetitions.
    /// An absent requested integer is outside the admitted signed-64 domain.
    pub fn new(keys: &[i64], requested: &[Option<i64>]) -> Self {
        let mut positions: HashMap<i64, Vec<usize>> = HashMap::new();
        for (row, &key) in keys.iter().enumerate() {
            positions.entry(key).or_default().push(row);
        }
        let missing: Vec<usize> = requested
            .iter()
            .enumerate()
            .filter_map(|(ordinal, key)| {
                key.and_then(|key| positions.get(&key))
                    .is_none()
                    .then_some(ordinal)
            })
            .collect();
        if !missing.is_empty() {
            return Self {
                scored: Vec::new(),
                missing,
            };
        }
        let mut selected = Self {
            scored: Vec::new(),
            missing,
        };
        for (ordinal, key) in requested.iter().enumerate() {
            match key.and_then(|key| positions.get(&key)) {
                Some(rows) => selected.scored.extend_from_slice(rows),
                None => selected.missing.push(ordinal),
            }
        }
        selected
    }
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

/// Owned primitive scoring identities, before the source membership read.
#[derive(Debug, Clone)]
pub struct ScoringKeys {
    values: HashSet<i64>,
}

impl ScoringKeys {
    pub fn new(values: Vec<i64>) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }
    pub fn needs_source(&self) -> bool {
        !self.values.is_empty()
    }
    pub fn retain(&self, source: &[i64]) -> Vec<usize> {
        source
            .iter()
            .enumerate()
            .filter_map(|(row, key)| (!self.values.contains(key)).then_some(row))
            .collect()
    }
}

/// Exclude every held-out row before collecting labelled candidates.
pub fn labelled_rows(known: &[bool], excluded: &[usize]) -> Result<Vec<usize>, CoreError> {
    let mut held_out = vec![false; known.len()];
    for &row in excluded {
        let held = held_out
            .get_mut(row)
            .ok_or_else(|| invalid("tabular holdout row is outside the source population"))?;
        *held = true;
    }
    Ok(known
        .iter()
        .zip(held_out)
        .enumerate()
        .filter_map(|(row, (&known, held))| (known && !held).then_some(row))
        .collect())
}

/// A useful context population, captured before its optional cap advice.
/// It contains source ordinals only, never a native handle or sorted answer.
#[derive(Debug, Clone)]
pub struct ContextSelection {
    candidates: Vec<usize>,
    wanted: usize,
    automatic: bool,
}

impl ContextSelection {
    /// The composition root supplies the model's established context limit.
    pub fn new(
        candidates: Vec<usize>,
        requested: Option<usize>,
        maximum: usize,
    ) -> Result<Self, CoreError> {
        if candidates.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid("tabular candidates must retain original row order"));
        }
        let wanted = requested.unwrap_or(maximum);
        if wanted == 0 {
            return Err(CoreError::new(
                Sentence::NumContextSamplesBelowOne,
                "tabular context size is below one",
            )
            .with_field("given", Detail::text("0")));
        }
        if wanted > maximum {
            return Err(CoreError::new(
                Sentence::NumContextSamplesPastMost,
                "tabular context size exceeds the model limit",
            )
            .with_field("given", Detail::text(wanted.to_string()))
            .with_field("limit", Detail::Scalar(Scalar::UInt64(maximum as u64))));
        }
        Ok(Self {
            candidates,
            wanted,
            automatic: requested.is_none(),
        })
    }

    /// The filtered population named by automatic-cap advice, before any draw.
    pub fn notice_rows(&self) -> Option<usize> {
        (self.automatic && self.candidates.len() > self.wanted).then_some(self.candidates.len())
    }

    pub fn wanted(&self) -> usize {
        self.wanted
    }

    pub fn reduced(&self) -> bool {
        self.candidates.len() > self.wanted
    }

    /// Original source rows, after sampling and ordering by actual index facts.
    pub fn select(&self, labels: Option<&[i64]>) -> Result<Vec<usize>, CoreError> {
        Ok(match self.positions(labels)? {
            Some(positions) => positions
                .into_iter()
                .map(|row| self.candidates[row])
                .collect(),
            None => self.candidates.clone(),
        })
    }

    /// Candidate-frame positions: seed-42 sample prefix, then unique-label sort.
    /// Uncapped populations consume no labels and require no replacement take.
    pub fn positions(&self, labels: Option<&[i64]>) -> Result<Option<Vec<usize>>, CoreError> {
        if !self.reduced() {
            if labels.is_some() {
                return Err(invalid(
                    "uncapped tabular context does not consume index labels",
                ));
            }
            return Ok(None);
        }
        let labels = labels.ok_or_else(|| {
            invalid("sampled tabular context requires actual candidate index labels")
        })?;
        if labels.len() != self.candidates.len() {
            return Err(invalid(
                "tabular index label extent differs from its captured candidates",
            ));
        }
        let mut unique = HashSet::with_capacity(labels.len());
        if labels.iter().any(|label| !unique.insert(*label)) {
            return Err(invalid("tabular candidate index labels must be unique"));
        }
        let mut selected = Mt19937::seeded(42).permutation(self.candidates.len());
        selected.truncate(self.wanted);
        selected.sort_unstable_by_key(|&row| labels[row]);
        Ok(Some(selected))
    }
}
