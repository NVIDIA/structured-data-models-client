// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Approved UTC windows on zoned warehouse timestamps, with wall clocks
//! preserved for NTZ and DATE. No session state changes are permitted.
use sdm_contract::data::{DataLayer, DataRequest, FetchPlan, ProviderChoice};
use sdm_data::Data;
use serde_json::json;

fn sql(dialect: &str, source_type: &str) -> String {
    let request: DataRequest = serde_json::from_value(json!({
        "request": "neighbors", "table": {"source": ["D", "S", "orders"]},
        "key": ["user_id"], "columns": ["ts"],
        "seeds": {"attachment": 1, "columns": [
            {"name": "key", "column_type": {"type": "int64"}, "nullable": false},
            {"name": "batch", "column_type": {"type": "int64"}, "nullable": false},
            {"name": "end", "column_type": {"type": "timestamp", "unit": "ns", "zone": null}, "nullable": true}
        ]},
        "window": {"column": "ts", "source_type": source_type, "lower": null,
                   "upper": {"seed_column": "end", "inclusive": true}},
        "per_seed": null, "order": null
    })).unwrap();
    let FetchPlan::Sql { statements, .. } = Data::new()
        .plan(&request, &ProviderChoice::sql(dialect)).unwrap() else { panic!("SQL") };
    statements.into_iter().map(|s| s.text).collect::<Vec<_>>().join("\n")
}

#[test]
fn snowflake_zoned_windows_compare_utc_and_wall_clocks_stay_unchanged() {
    for kind in ["TIMESTAMP_LTZ(6)", "TIMESTAMP_TZ(9)"] {
        let text = sql("snowflake", kind);
        assert!(text.contains("CONVERT_TIMEZONE('UTC', \"ts\")::TIMESTAMP_NTZ"), "{text}");
        assert!(!text.contains("ALTER SESSION"));
    }
    for kind in ["TIMESTAMP_NTZ(9)", "DATE"] {
        let text = sql("snowflake", kind);
        assert!(!text.contains("CONVERT_TIMEZONE"), "{text}");
    }
}

#[test]
fn databricks_zoned_bounds_name_utc_and_ntz_bounds_name_wall_time() {
    for kind in ["TIMESTAMP", "TIMESTAMP_LTZ"] {
        let text = sql("databricks", kind);
        assert!(text.contains("CAST(CONCAT(col.e, 'Z') AS TIMESTAMP)"), "{text}");
        assert!(!text.contains("SET TIME ZONE"));
    }
    for kind in ["TIMESTAMP_NTZ", "DATE"] {
        let text = sql("databricks", kind);
        assert!(text.contains("CAST(col.e AS TIMESTAMP_NTZ)"), "{text}");
        assert!(!text.contains("CONCAT(col.e, 'Z')"));
    }
}
