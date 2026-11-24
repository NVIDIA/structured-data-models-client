// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A host provider's request asked in parts: the seeds of a `neighbors` or
//! a `lookup` holding more keys than the provider takes in one request
//! (`max_keys_per_request`) split into requests of at most that many seeds
//! each, whose answers, joined in order, are the request's
//! (the core's `AskState::Chunks`). Every host reads the same parts, so no
//! host splits a key set itself.

use std::collections::BTreeMap;

use sdm_contract::data::{AttachmentId, BoundFetch, BoundRequest, DataRequest, FetchPlan, KeyData};
use sdm_contract::{CoreError, Sentence};

/// The local name each part's seeds are bound by, until the run installs
/// them as an attachment of their own; a request split here binds no other
/// key set.
const PART: AttachmentId = AttachmentId(0);

/// `request` asked as parts of at most `max` seeds each, where it is a
/// `neighbors` or a `lookup`: each seed's rows are answered apart from
/// another's, and carry the seed's own batch position, so the parts'
/// answers, in the seeds' order, are the request's. Any other request, and
/// one whose seeds number no more than `max`, is asked whole.
pub fn bind(
    request: &DataRequest,
    plan: FetchPlan,
    keys: &BTreeMap<AttachmentId, KeyData>,
    max: u64,
) -> Result<BoundRequest, CoreError> {
    let (DataRequest::Neighbors { seeds, .. } | DataRequest::Lookup { seeds, .. }) = request else {
        return Ok(BoundRequest::whole(plan, request));
    };
    debug_assert_eq!(
        request.key_sets(),
        [seeds.attachment],
        "the seeds are its one key set"
    );
    let Some(data) = keys.get(&seeds.attachment) else {
        return Ok(BoundRequest::whole(plan, request));
    };
    let rows: usize = data.batches.iter().map(|b| b.num_rows()).sum();
    let max = usize::try_from(max.max(1)).unwrap_or(usize::MAX);
    if rows <= max {
        return Ok(BoundRequest::whole(plan, request));
    }
    let joined = arrow_select::concat::concat_batches(&data.schema, &data.batches)
        .map_err(|_| CoreError::new(Sentence::Internal, "a key set's batches do not join"))?;
    let renamed = BTreeMap::from([(seeds.attachment, PART)]);
    let mut fetches = Vec::new();
    let mut from = 0;
    while from < rows {
        let length = max.min(rows - from);
        let mut part = request.clone();
        part.rename_attachments(&renamed);
        let mut part_plan = plan.clone();
        part_plan.rename_attachments(&renamed);
        fetches.push(BoundFetch {
            plan: part_plan,
            batches: BTreeMap::from([(
                PART,
                KeyData {
                    schema: data.schema.clone(),
                    batches: vec![joined.slice(from, length)],
                },
            )]),
            request: part,
        });
        from += length;
    }
    Ok(BoundRequest { fetches })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use sdm_contract::data::{ExpectedSchema, Filter, KeySet, Relation};
    use sdm_contract::types::{ColumnType, Field as ContractField};

    use super::*;

    const SEEDS: AttachmentId = AttachmentId(5);

    fn key_set() -> KeySet {
        let field = |name: &str| ContractField {
            name: name.into(),
            column_type: ColumnType::Int64,
            nullable: false,
            metadata: BTreeMap::new(),
        };
        KeySet {
            attachment: SEEDS,
            columns: vec![field("key"), field("batch")],
        }
    }

    /// Seven seeds in two batches, each its key and its batch position.
    fn keys(rows: i64) -> BTreeMap<AttachmentId, KeyData> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("batch", DataType::Int64, false),
        ]));
        let batch = |from: i64, to: i64| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values((from..to).map(|i| i * 10))),
                    Arc::new(Int64Array::from_iter_values(from..to)),
                ],
            )
            .unwrap()
        };
        let middle = rows / 2;
        BTreeMap::from([(
            SEEDS,
            KeyData {
                schema: Arc::clone(&schema),
                batches: vec![batch(0, middle), batch(middle, rows)],
            },
        )])
    }

    fn lookup() -> DataRequest {
        DataRequest::Lookup {
            table: Relation::named("users"),
            key: vec!["id".into()],
            columns: vec!["age".into()],
            seeds: key_set(),
        }
    }

    fn scan(limit: Option<u64>) -> DataRequest {
        DataRequest::Scan {
            table: Relation::named("events"),
            columns: vec!["id".into()],
            filters: vec![Filter::KeyIn {
                columns: vec!["id".into()],
                keys: key_set(),
            }],
            limit,
        }
    }

    fn plan(request: &DataRequest) -> FetchPlan {
        FetchPlan {
            expect: ExpectedSchema::Discover {},
            key_sets: request.key_sets(),
        }
    }

    fn batches(fetch: &BoundFetch) -> Vec<i64> {
        let data = &fetch.batches[&PART];
        data.batches
            .iter()
            .flat_map(|b| {
                let column = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                column.values().to_vec()
            })
            .collect()
    }

    #[test]
    fn a_key_set_past_the_limit_is_asked_in_parts_of_at_most_the_limit_in_order() {
        let bound = bind(&lookup(), plan(&lookup()), &keys(7), 3).unwrap();
        let parts: Vec<Vec<i64>> = bound.fetches.iter().map(batches).collect();
        assert_eq!(parts, [vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
        for fetch in &bound.fetches {
            // Each part's request and plan name its own seeds.
            assert_eq!(fetch.request.key_sets(), [PART]);
            assert_eq!(fetch.plan.key_sets, [PART]);
        }
    }

    #[test]
    fn what_is_not_split_is_asked_whole() {
        let whole = |request: &DataRequest, rows: i64, max: u64| {
            let bound = bind(request, plan(request), &keys(rows), max).unwrap();
            bound.fetches.len() == 1
                && &bound.fetches[0].request == request
                && bound.fetches[0].plan.key_sets == [SEEDS]
                && bound.fetches[0].batches.is_empty()
        };
        // A scan's key filter: parts would answer a key twice that the
        // whole asks once, and keep no limit.
        assert!(whole(&scan(None), 7, 3));
        assert!(whole(&scan(Some(10)), 7, 3));
        // No more seeds than the provider takes.
        assert!(whole(&lookup(), 3, 3));
        // No limit declared.
        assert!(whole(&lookup(), 7, u64::MAX));
    }
}
