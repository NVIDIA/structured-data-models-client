// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[test]
fn contradictory_reached_overlap_counts_are_protocol_errors() {
    let relationship = [
        ColumnRef {
            table: "child".into(),
            column: "fk".into(),
        },
        ColumnRef {
            table: "parent".into(),
            column: "id".into(),
        },
    ];
    for (child, parent, intersects) in [
        (0, Some(0), None),
        (0, None, Some(false)),
        (1, None, None),
        (1, Some(0), Some(true)),
        (1, Some(1), None),
    ] {
        let result = relationship_overlap(vec![FrameRelationshipEvidence::Distinct {
            relationship: relationship.clone(),
            child,
            parent,
            intersects,
        }]);
        assert_eq!(
            result.unwrap_err().code,
            sdm_contract::ErrorCode::InvalidMessage
        );
    }
}
