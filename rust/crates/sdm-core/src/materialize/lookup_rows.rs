// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scored lookup validation shared by raw and native source completion.
//! The caller supplies its actual task batch, never a host refusal flag.

use super::*;

#[cfg(test)]
mod tests;

/// Validate counted lookup rows before native business-column sanitation.
/// Return their actual seed positions without sorting or selecting rows.
/// Missing scored rows remain the later request completion's responsibility.
pub(super) fn validate(
    entity: &Table,
    key: &str,
    scored: &ArrayRef,
    num_train: usize,
    looked_up: &RecordBatch,
) -> Result<Vec<usize>, CoreError> {
    let positions = batches(looked_up, num_train + scored.len())?;
    // A key the source enforces unique is held by one row each: a
    // dialect may leave its count out (SEED_KEY_UNIQUE), and any
    // other count is read. Uncounted neighbor reads use another caller.
    let unique = entity.column(key).is_some_and(|column| column.unique);
    let matches = match looked_up.column_by_name(MATCHES_COLUMN) {
        None if unique => None,
        _ => Some(
            named(looked_up, MATCHES_COLUMN)?
                .as_primitive_opt::<Int64Type>()
                .ok_or_else(|| {
                    crate::run::schema_mismatch("expected", Some(MATCHES_COLUMN), "not int64")
                })?,
        ),
    };
    // Each repeated scored entity in caller order, once per caller value.
    // Repeated context keys do not cause this refusal.
    let mut repeated: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter(|&(row, &at)| at >= num_train && matches.is_some_and(|m| m.value(row) > 1))
        .map(|(_, &at)| at)
        .collect();
    if !repeated.is_empty() {
        repeated.sort_unstable();
        repeated.dedup();
        let mut keys: Vec<Scalar> = Vec::with_capacity(repeated.len());
        for at in repeated {
            let value = key_scalar(scored.as_ref(), at - num_train);
            if !keys.contains(&value) {
                keys.push(value);
            }
        }
        let (sentence, more) = match keys.len() {
            1 => (Sentence::HeldBySeveralRowsOne, 0),
            2..=5 => (Sentence::HeldBySeveralRows, 0),
            n => (Sentence::HeldBySeveralRowsMany, n - 5),
        };
        let mut refused = invalid(
            sentence,
            "an entity to predict for is held by more than one row, so which is meant \
             would be a guess",
            "indices",
        );
        if more > 0 {
            refused = refused
                .with_field("shown", Detail::List(keys[..5].to_vec()))
                .with_field("more", count(more));
        }
        return Err(refused
            .with_field("reason", Detail::text("held_by_several_rows"))
            .with_field("table", Detail::text(&entity.name))
            .with_field("key", Detail::text(key))
            .with_field("keys", Detail::List(keys)));
    }
    Ok(positions)
}
