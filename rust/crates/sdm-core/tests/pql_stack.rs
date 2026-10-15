// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The adapter's stack and the deepest validated query: the adapter grows
//! one stack per call and the parser none of its own; the deepest
//! tree the parser accepts round-trips through the validated-query format,
//! and text one level deeper is refused by its reader.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads the crate's sources to find where stacks grow"
)]

use std::path::{Path, PathBuf};

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::pql::{
    LogicalOp, MAX_JSON_DEPTH, NodeKind, PqlNode, ReadError, ValidatedQuery, read_validated_query,
};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::pql::{on_adapter_stack, resolve};
use sdm_core::profiles::Profile;
use serde_json::json;
/// Zones that hold nothing: these facts name no local zone, so none is
/// resolved.
struct NoZones;

impl sdm_core::Zones for NoZones {
    fn zone(&self, _: &str) -> Result<jiff::tz::TimeZone, sdm_core::UnknownZone> {
        Err(sdm_core::UnknownZone)
    }
}

fn facts() -> HostFacts {
    HostFacts {
        now: Scalar::Timestamp {
            unit: Unit::Us,
            zone: Some("UTC".into()),
            value: 1_767_225_600_000_000,
        },
        local_zone: None,
        zone_roots: vec![],
        platform: PlatformFacts {
            os: "linux".into(),
            strftime_pads_year: false,
        },
        date_literal_range: None,
    }
}

fn shop() -> QuerySchema {
    serde_json::from_value(json!({
        "tables": [
            {"name": "orders", "columns": [
                {"name": "order_id", "dtype": "int", "stype": "ID"},
                {"name": "user_id", "dtype": "int", "stype": "ID"},
                {"name": "price", "dtype": "float", "stype": "numerical"},
                {"name": "ts", "dtype": "time", "stype": "timestamp"}],
             "primary_key": "order_id", "time_column": "ts", "end_time_column": null},
            {"name": "users", "columns": [
                {"name": "user_id", "dtype": "int", "stype": "ID"},
                {"name": "age", "dtype": "int", "stype": "numerical"}],
             "primary_key": "user_id", "time_column": null, "end_time_column": null}],
        "links": [[{"table": "orders", "column": "user_id"}, {"table": "users", "column": "user_id"}]],
    }))
    .unwrap()
}

/// The source files of a crate of this workspace.
fn sources(krate: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut dirs = vec![
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(krate)
            .join("src"),
    ];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn the_adapter_grows_one_stack_per_call_and_the_parser_none() {
    // Every call that grows a stack, in code, outside comments.
    let mut sites = Vec::new();
    for krate in ["sdm-core", "sdm-pql"] {
        for path in sources(krate) {
            let text = std::fs::read_to_string(&path).unwrap();
            for (n, line) in text.lines().enumerate() {
                let code = line.split("//").next().unwrap_or_default();
                if code.contains("stacker::") {
                    let name = path.file_name().unwrap().to_string_lossy().into_owned();
                    sites.push(format!("{krate}/{name}:{}: {}", n + 1, code.trim()));
                }
            }
        }
    }
    // One: `on_adapter_stack`, which `validate` and `resolve` each enter
    // once, and nothing below them enters again.
    assert_eq!(sites.len(), 1, "{sites:#?}");
    assert!(sites[0].starts_with("sdm-core/pql.rs:"), "{sites:#?}");
    assert!(
        sites[0].contains("stacker::grow(STACK_BYTES, f)"),
        "{sites:#?}"
    );
    let core =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/pql.rs")).unwrap();
    assert_eq!(core.matches("on_adapter_stack(||").count(), 2);
    // And the stack it gives is the grown one, whatever the caller's.
    let remaining = std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| on_adapter_stack(stacker::remaining_stack))
        .unwrap()
        .join()
        .unwrap()
        .unwrap();
    assert!(remaining > 200 << 20, "{remaining}");
}

/// The deepest queries the parser accepts, of each nesting form.
fn deepest_queries() -> Vec<String> {
    vec![
        format!(
            "PREDICT {}COUNT(orders.*, 0, 30, days) > 1 FOR EACH users.user_id",
            "NOT ".repeat(999)
        ),
        format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {}users.age > 3{}",
            "(".repeat(999),
            ")".repeat(999)
        ),
        format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE users.age > 3{}",
            " AND users.age > 3".repeat(998)
        ),
        format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE users.age > 3{}",
            " OR users.age > 3".repeat(998)
        ),
        // Mixed: NOTs under a chain, and chains nested in groups, each as
        // tall as the parser takes.
        format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE ({}users.age > 3){}",
            "NOT ".repeat(500),
            " AND users.age > 3".repeat(498)
        ),
        {
            let mut condition = "users.age > 3".to_owned();
            for _ in 0..10 {
                condition = format!("({condition}{})", " OR users.age > 3".repeat(98));
            }
            format!("PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {condition}")
        },
    ]
}

fn validated(query: &str) -> ValidatedQuery {
    let profile = Profile::resolve(&ProfileSpec {
        name: ProfileName::Relational,
        model: None,
        capabilities: Default::default(),
    });
    resolve(query, &shop(), &profile, &facts(), &NoZones)
        .unwrap()
        .query
        .expect("accepted")
}

/// The deepest nesting of objects and arrays in `text`, strings skipped.
fn json_depth(text: &str) -> usize {
    let value: serde_json::Value = {
        let mut de = serde_json::Deserializer::from_str(text);
        de.disable_recursion_limit();
        serde::Deserialize::deserialize(serde_stacker::Deserializer::new(&mut de)).unwrap()
    };
    fn depth(value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::Object(map) => 1 + map.values().map(depth).max().unwrap_or(0),
            serde_json::Value::Array(items) => 1 + items.iter().map(depth).max().unwrap_or(0),
            _ => 0,
        }
    }
    depth(&value)
}

#[test]
fn the_deepest_validated_query_round_trips_and_one_level_deeper_is_refused() {
    on_adapter_stack(|| {
        let mut deepest = 0;
        let mut deepest_query = None;
        for query in deepest_queries() {
            let tree = validated(&query);
            let text = serde_json::to_string(&tree).unwrap();
            let back = read_validated_query(&text).unwrap();
            assert_eq!(back, tree);
            let depth = json_depth(&text);
            if depth > deepest {
                deepest = depth;
                deepest_query = Some(tree);
            }
        }
        // The format's limit is the deepest accepted tree's, past
        // serde_json's own recursion limit.
        assert_eq!(deepest, MAX_JSON_DEPTH);
        let text = serde_json::to_string(deepest_query.as_ref().unwrap()).unwrap();
        assert!(serde_json::from_str::<ValidatedQuery>(&text).is_err());

        // One tree level deeper (a NOT around the deepest target): refused
        // before anything is built.
        let mut deeper = deepest_query.unwrap();
        let wrap = |inner: PqlNode| PqlNode {
            dtype: inner.dtype,
            stype: inner.stype,
            range: inner.range,
            span: inner.span,
            kind: NodeKind::Logical {
                op: LogicalOp::Not,
                left: Box::new(inner),
                right: None,
            },
        };
        deeper.target = wrap(deeper.target.clone());
        deeper.entity = wrap(deeper.entity.clone());
        if let Some(assuming) = deeper.assuming.take() {
            deeper.assuming = Some(wrap(assuming));
        }
        let text = serde_json::to_string(&deeper).unwrap();
        assert!(json_depth(&text) > MAX_JSON_DEPTH);
        assert!(matches!(
            read_validated_query(&text),
            Err(ReadError::TooDeep { .. })
        ));
    });
}
