// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The data requests' own checks: every field is honored or the request is
//! refused with a typed error naming the request and the field, never a
//! value (design, "Data access"; decision H15).

#[allow(dead_code, reason = "the corpus serves several test files")]
mod corpus;

use corpus::seeds;
use sdm_contract::data::{
    DataRequest, DescribePart, ExpectedSchema, OptionalRequest, SeedBound, TableColumn, TableKey,
    TieBreak, TieTerm, Window,
};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

fn term(column: &str) -> TieTerm {
    TieTerm {
        column: column.into(),
        as_timestamp: false,
        descending: false,
        nulls_last: true,
    }
}

fn upper(seed_column: &str) -> Window {
    Window {
        column: "ts".into(),
        lower: None,
        upper: Some(SeedBound {
            seed_column: seed_column.into(),
            inclusive: true,
        }),
    }
}

fn neighbors(
    window: Option<Window>,
    per_seed: Option<u64>,
    order: Option<TieBreak>,
) -> DataRequest {
    DataRequest::Neighbors {
        table: "orders".into(),
        key: vec!["user_id".into()],
        columns: vec!["price".into()],
        seeds: seeds(&["anchor", "start"]),
        window,
        per_seed,
        order,
    }
}

fn refused(request: &DataRequest, field: &str) -> CoreError {
    let err = request.check().unwrap_err();
    assert_eq!(
        (err.kind, err.code),
        (ErrorKind::InvalidRequest, ErrorCode::InvalidRequest)
    );
    let text = |name: &str| match err.fields.get(name) {
        Some(Detail::Scalar(Scalar::Utf8(text))) => text.clone(),
        other => panic!("{name}: {other:?}"),
    };
    assert_eq!(text("request"), request.name());
    assert_eq!(text("field"), field);
    err
}

#[test]
fn both_neighbor_fetches_of_the_reference_are_expressible() {
    // _by_fkey: up to the anchor, inclusive, n rows per seed in tie order.
    let by_fkey = neighbors(
        Some(upper("anchor")),
        Some(32),
        Some(TieBreak(vec![term("ts"), term("order_id")])),
    );
    by_fkey.check().unwrap();
    // _by_time: after the start, exclusive, up to the end, inclusive; no
    // limit and no order.
    let by_time = neighbors(
        Some(Window {
            column: "ts".into(),
            lower: Some(SeedBound {
                seed_column: "start".into(),
                inclusive: false,
            }),
            upper: Some(SeedBound {
                seed_column: "anchor".into(),
                inclusive: true,
            }),
        }),
        None,
        None,
    );
    by_time.check().unwrap();
    // No time column: no window.
    neighbors(None, Some(1), Some(TieBreak(vec![term("order_id")])))
        .check()
        .unwrap();
    assert_eq!(by_fkey.optional(), Some(OptionalRequest::Neighbors));
}

#[test]
fn a_row_limit_and_an_order_come_together() {
    refused(&neighbors(None, Some(3), None), "order");
    refused(
        &neighbors(None, None, Some(TieBreak(vec![term("ts")]))),
        "order",
    );
    refused(&neighbors(None, Some(3), Some(TieBreak(vec![]))), "order");
}

#[test]
fn a_window_bound_names_a_zone_less_nanosecond_seed_column() {
    refused(&neighbors(Some(upper("missing")), None, None), "window");
    // `batch` is a seed column, but not a time.
    refused(&neighbors(Some(upper("batch")), None, None), "window");
    refused(
        &neighbors(
            Some(Window {
                column: "ts".into(),
                lower: None,
                upper: None,
            }),
            None,
            None,
        ),
        "window",
    );
}

#[test]
fn seeds_hold_the_key_and_batch_columns() {
    for missing in ["key", "batch"] {
        let mut keys = seeds(&[]);
        keys.columns.retain(|f| f.name != missing);
        refused(
            &DataRequest::Lookup {
                table: "users".into(),
                key: vec!["user_id".into()],
                columns: vec![],
                seeds: keys.clone(),
            },
            "seeds",
        );
        let DataRequest::Neighbors {
            table,
            key,
            columns,
            window,
            per_seed,
            order,
            ..
        } = neighbors(None, None, None)
        else {
            unreachable!()
        };
        refused(
            &DataRequest::Neighbors {
                table,
                key,
                columns,
                seeds: keys,
                window,
                per_seed,
                order,
            },
            "seeds",
        );
    }
}

#[test]
fn the_reference_s_union_reads_name_what_they_read() {
    let keys = DataRequest::KeyCounts {
        keys: vec![
            TableKey {
                table: "orders".into(),
                columns: vec!["order_id".into()],
            },
            TableKey {
                table: "users".into(),
                columns: vec!["user_id".into()],
            },
        ],
    };
    keys.check().unwrap();
    refused(&DataRequest::KeyCounts { keys: vec![] }, "keys");
    refused(
        &DataRequest::KeyCounts {
            keys: vec![TableKey {
                table: "orders".into(),
                columns: vec![],
            }],
        },
        "keys",
    );
    let bounds = DataRequest::TimeBounds {
        columns: vec![TableColumn {
            table: "orders".into(),
            column: "ts".into(),
        }],
    };
    bounds.check().unwrap();
    refused(&DataRequest::TimeBounds { columns: vec![] }, "columns");
    let ranges = DataRequest::ColumnRanges {
        table: "orders".into(),
        columns: vec!["ts".into()],
    };
    ranges.check().unwrap();
    assert_eq!(ranges.name(), "column_ranges");
    assert_eq!(
        ranges.optional().map(OptionalRequest::name),
        Some("column_ranges")
    );
    assert_eq!(ranges.expected(), ExpectedSchema::Discover {});
    refused(
        &DataRequest::ColumnRanges {
            table: "orders".into(),
            columns: vec![],
        },
        "columns",
    );
    let repeated = DataRequest::RepeatedKey {
        table: "orders".into(),
        key: vec!["order_id".into()],
    };
    repeated.check().unwrap();
    assert_eq!(repeated.name(), "repeated_key");
    assert_eq!(
        repeated.optional().map(OptionalRequest::name),
        Some("repeated_key")
    );
    assert_eq!(repeated.expected(), ExpectedSchema::Discover {});
    refused(
        &DataRequest::RepeatedKey {
            table: "orders".into(),
            key: vec![],
        },
        "key",
    );
}

#[test]
fn a_refusal_quotes_no_value_it_was_given() {
    let canary = "CANARY_7f3e_secret";
    let err = refused(&neighbors(Some(upper(canary)), None, None), "window");
    let text = serde_json::to_string(&err).unwrap();
    assert!(!text.contains(canary), "{text}");
}

#[test]
fn every_request_has_its_contract_name() {
    let describe = DataRequest::Describe {
        tables: vec![],
        part: DescribePart::ForeignKeys,
    };
    let value = serde_json::to_value(&describe).unwrap();
    assert_eq!(value["request"], describe.name());
    assert_eq!(value["part"], "foreign_keys");
    for request in [
        neighbors(None, None, None),
        DataRequest::Lookup {
            table: "users".into(),
            key: vec!["user_id".into()],
            columns: vec![],
            seeds: seeds(&[]),
        },
    ] {
        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["request"], request.name());
        let optional = request.optional().unwrap();
        assert_eq!(serde_json::to_value(optional).unwrap(), optional.name());
    }
}
