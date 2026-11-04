// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_pql::dates::Reader;
use sdm_pql::{Mode, parse};

#[test]
fn forecast_entity_rendering_preserves_ids_and_filters() {
    for entity in [
        "FOR accounts.id = 3",
        "FOR accounts.id IN (3, 4)",
        "FOR EACH accounts.id WHERE accounts.value > 1",
    ] {
        let text = format!("PREDICT COUNT(orders.*, 0, 1, days) FORECAST 3 TIMEFRAMES {entity}");
        let parsed = parse(
            &text,
            Mode::Inference,
            Reader {
                year: Some(2026),
                pads_year: false,
            },
        )
        .outcome
        .unwrap();
        assert_eq!(parsed.query.render_entity(false), entity);
    }
}
