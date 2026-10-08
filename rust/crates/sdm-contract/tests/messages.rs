// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Control messages round-trip through JSON, and unknown fields are refused.

mod corpus;

use std::collections::BTreeSet;

use corpus::messages;
use sdm_contract::ControlMessage;
use sdm_contract::data::{
    Determinism, Dialect, OptionalRequest, ProviderCapabilities, ProviderChoice,
};

#[test]
fn every_message_round_trips() {
    for message in messages() {
        let text = serde_json::to_string(&message).unwrap();
        let back: ControlMessage = serde_json::from_str(&text).unwrap();
        assert_eq!(back, message, "{text}");
    }
}

#[test]
fn provider_choices_round_trip() {
    for choice in [
        ProviderChoice::Sql {
            dialect: Dialect::Duckdb,
        },
        ProviderChoice::Host {
            capabilities: ProviderCapabilities {
                optional: BTreeSet::from([OptionalRequest::Preview, OptionalRequest::CountRows]),
                max_keys_per_request: Some(10_000),
                sampling: Determinism::SeededStable,
            },
        },
    ] {
        let text = serde_json::to_string(&choice).unwrap();
        assert_eq!(
            serde_json::from_str::<ProviderChoice>(&text).unwrap(),
            choice
        );
    }
}

#[test]
fn unknown_fields_and_tags_are_refused() {
    for text in [
        r#"{"message":"cancel","extra":1}"#,
        r#"{"message":"step","step":"ask_host","what":"truthiness"}"#,
        r#"{"message":"fetch_reply","id":1,"outcome":{"outcome":"data","data":2,"rows":3}}"#,
        r#"{"message":"error","error":{"kind":"python","code":"X","message":""}}"#,
        r#"{"message":"step","step":"fetch","requests":[{"id":1,"request":{"request":"describe","tables":[]},"sql":null,"expect":{"expect":"discover","x":1}}]}"#,
        r#"{"message":"warnings","warnings":[{"code":"W","message":"","fields":{"f":{"t":"null","v":"1"}}}]}"#,
    ] {
        assert!(
            serde_json::from_str::<ControlMessage>(text).is_err(),
            "{text} was accepted"
        );
    }
}
