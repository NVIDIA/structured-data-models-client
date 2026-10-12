// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The deepest query, end to end on a small stack, in a process of its own:
//! a query nested 999 levels is parsed, resolved, read off (its features
//! and its validated query traversed), and its trees dropped, all from a
//! thread of 256 KB, through the adapter's grown stack. The work runs in a
//! child process, so a stack overflow, which aborts a process, fails this
//! test with the child's exit status rather than ending the test run.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: runs its own binary as the child, told so by its environment"
)]

use std::process::Command;

use sdm_contract::facts::PlatformFacts;
use sdm_contract::ops::{ProfileName, ProfileSpec, QuerySchema};
use sdm_contract::{HostFacts, Scalar, Unit};
use sdm_core::pql::{features, on_adapter_stack, resolve, validate};
use sdm_core::profiles::Profile;
use serde_json::json;

/// Set in the child: run the work and report it.
const CHILD: &str = "SDM_PQL_DEEP_CHILD";
/// What the child prints once every step has run.
const DONE: &str = "deep query: parsed, resolved, traversed and dropped at depth 999";

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
    }
}

fn shop() -> QuerySchema {
    serde_json::from_value(json!({
        "tables": [
            {"name": "orders", "columns": [
                {"name": "order_id", "dtype": "int", "stype": "ID"},
                {"name": "user_id", "dtype": "int", "stype": "ID"},
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

/// 999 nested `NOT`s around the target, and 999 nested parentheses around
/// the entity's filter: the deepest the parser accepts.
fn deepest() -> [String; 2] {
    [
        format!(
            "PREDICT {}COUNT(orders.*, 0, 30, days) > 1 FOR EACH users.user_id",
            "NOT ".repeat(999)
        ),
        format!(
            "PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {}users.age > 3{}",
            "(".repeat(999),
            ")".repeat(999)
        ),
    ]
}

/// The work, on a thread of 256 KB: every step a deep tree recurses
/// through, each on the adapter's stack.
fn work() {
    std::thread::Builder::new()
        .stack_size(256 << 10)
        .spawn(|| {
            let profile = Profile::resolve(&ProfileSpec {
                name: ProfileName::Relational,
                model: None,
                capabilities: Default::default(),
            });
            for query in deepest() {
                // Parsed, resolved and dropped inside the call.
                let (answer, _) = validate(&query, &shop(), &profile, &facts(), &NoZones).unwrap();
                assert!(answer.accepted, "{:?}", answer.diagnostics);
                // Handed back, then traversed and dropped on the adapter's
                // stack, as a caller holding a deep query must.
                let validation = resolve(&query, &shop(), &profile, &facts(), &NoZones).unwrap();
                on_adapter_stack(move || {
                    let held = validation.query.expect("accepted");
                    let read = features(&held, &shop());
                    assert!(read.temporal_target || read.entity_filter);
                    let text = serde_json::to_string(&held).unwrap();
                    assert!(text.len() > 999);
                    drop(held);
                });
            }
        })
        .unwrap()
        .join()
        .expect("the work finished on its thread");
    println!("{DONE}");
}

#[test]
fn a_depth_999_query_is_parsed_traversed_and_dropped_on_a_256_kb_thread() {
    if std::env::var_os(CHILD).is_some() {
        work();
        return;
    }
    let child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "a_depth_999_query_is_parsed_traversed_and_dropped_on_a_256_kb_thread",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&child.stdout);
    assert!(
        child.status.success(),
        "the child ended {:?}: {stdout}{}",
        child.status,
        String::from_utf8_lossy(&child.stderr)
    );
    assert!(
        stdout.contains(DONE),
        "the child did not finish the work: {stdout}"
    );
}
