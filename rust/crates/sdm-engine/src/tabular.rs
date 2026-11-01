// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tabular source-row selection. Entity equality is supplied by the admitted
//! key representation; index labels and sampled ordering are separate inputs.

use std::collections::{BTreeMap, BTreeSet};

/// Source positions in scoring order, or missing requested-key ordinals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoringRows {
    pub rows: Vec<usize>,
    pub missing: Vec<usize>,
}

/// Expand each requested key to every matching source row, retaining repeats.
/// Only use a key type whose equality agrees with the actual native producer.
/// In particular, missing native keys must not be collapsed into one null key.
pub fn expand_keys<K: Ord>(source: &[K], requested: &[K]) -> ScoringRows {
    let mut positions = BTreeMap::<&K, Vec<usize>>::new();
    for (row, key) in source.iter().enumerate() {
        positions.entry(key).or_default().push(row);
    }
    let missing: Vec<_> = requested
        .iter()
        .enumerate()
        .filter_map(|(ordinal, key)| (!positions.contains_key(key)).then_some(ordinal))
        .collect();
    if !missing.is_empty() {
        return ScoringRows {
            rows: Vec::new(),
            missing,
        };
    }
    let rows = requested
        .iter()
        .flat_map(|key| positions[key].iter().copied())
        .collect();
    ScoringRows { rows, missing }
}

/// Supplied-frame holdout by admitted value equality, preserving source order.
/// This is distinct from positional holdout on the indices entrypoint.
pub fn matching_rows<K: Ord>(source: &[K], scoring: &[K]) -> Vec<usize> {
    let keys: BTreeSet<_> = scoring.iter().collect();
    source
        .iter()
        .enumerate()
        .filter_map(|(row, key)| keys.contains(key).then_some(row))
        .collect()
}

/// Filter target validity and positional holdout without observing index labels.
/// Invalid positions are protocol failures for the composition layer to report.
pub fn candidates(known: &[bool], excluded: &[usize]) -> Option<Vec<usize>> {
    let mut held_out = vec![false; known.len()];
    for &row in excluded {
        *held_out.get_mut(row)? = true;
    }
    Some(
        known
            .iter()
            .enumerate()
            .filter_map(|(row, &valid)| (valid && !held_out[row]).then_some(row))
            .collect(),
    )
}

/// Capacity decision made before delivering any default-reduction warning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextDecision {
    pub wanted: usize,
    pub reduced: bool,
    pub advise_default_reduction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountError {
    Zero,
    AboveLimit,
    AbovePopulation,
}

/// The composition supplies the model limit; this engine owns no capability.
pub fn context_decision(
    population: usize,
    requested: Option<usize>,
    limit: usize,
) -> Result<ContextDecision, CountError> {
    let wanted = requested.unwrap_or(limit);
    if wanted == 0 {
        return Err(CountError::Zero);
    }
    if wanted > limit {
        return Err(CountError::AboveLimit);
    }
    let reduced = population > wanted;
    Ok(ContextDecision {
        wanted,
        reduced,
        advise_default_reduction: reduced && requested.is_none(),
    })
}

/// Draw from the retained candidate frame's population AFTER warning delivery.
/// The population may have changed; never reuse its pre-warning row count.
/// Returned offsets are in draw order, before sampled-index ordering.
pub fn sample_offsets(population: usize, wanted: usize) -> Result<Vec<usize>, CountError> {
    if wanted > population {
        return Err(CountError::AbovePopulation);
    }
    let mut drawn = crate::random::LegacyMt19937::seeded(42).permutation(population);
    drawn.truncate(wanted);
    Ok(drawn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduction_draws_after_warning_and_before_index_ordering() {
        let decision = context_decision(10, Some(3), 10).unwrap();
        assert!(decision.reduced);
        assert!(!decision.advise_default_reduction);
        assert_eq!(sample_offsets(10, decision.wanted).unwrap(), vec![8, 1, 5]);
        assert_eq!(
            sample_offsets(10, decision.wanted),
            sample_offsets(10, decision.wanted)
        );
        assert!(!context_decision(3, Some(3), 10).unwrap().reduced);
        assert!(
            context_decision(10, None, 3)
                .unwrap()
                .advise_default_reduction
        );
        assert_eq!(
            context_decision(0, Some(11), 10),
            Err(CountError::AboveLimit)
        );
        assert_eq!(
            sample_offsets(2, decision.wanted),
            Err(CountError::AbovePopulation)
        );
    }

    #[test]
    fn repeated_requests_expand_duplicate_source_keys() {
        assert_eq!(
            expand_keys(&[7, 3, 7, 9], &[7, 9, 7]),
            ScoringRows {
                rows: vec![0, 2, 3, 0, 2],
                missing: vec![],
            }
        );
        assert_eq!(expand_keys(&[7, 3], &[8, 7, 8]).missing, vec![0, 2]);
    }

    #[test]
    fn holdout_is_applied_before_context_selection() {
        let held = matching_rows(&[7, 3, 7, 9], &[7]);
        assert_eq!(candidates(&[true, false, true, true], &held), Some(vec![3]));
        assert_eq!(candidates(&[true], &[1]), None);
    }
}
