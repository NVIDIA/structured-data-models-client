// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The host layer every binding shares, plain Rust: the control codecs and
//! the 64 KB control cap ([`codec`]), Arrow admission ([`admission`]), run
//! ownership with panic containment and busy handling ([`runs`]), and zone
//! resolution through the host's lookup roots ([`zones`]). It schedules no
//! domain work; that is the core's.

#![forbid(unsafe_code)]

pub mod admission;
pub mod codec;
mod extensions;
mod frames;
pub mod local_zone;
pub mod runs;
pub mod zones;

pub use admission::{OFFSET_CAPACITY, admit, admit_from, admit_with_capacity, admitted_type};
pub use codec::{CONTROL_CAP, Outgoing, decode, encode, read, write, write_error};
pub use frames::detach as detach_frame;
pub use local_zone::{LOCALTIME, ZoneFiles, local_zone};
pub use runs::{Failure, HostRun, Unreadable, Unreadables, contain, is_posix_zone};
/// The types a binding hands across, so a binding depends on `sdm-host`
/// and on `sdm-contract` for types, never on `sdm-core` (decision E21);
/// `CompositionRoot` too, so a host composes its own data layer
/// (`CompositionRoot::new`) where it does not take [`default_root`]'s, as
/// the dev-only test host swaps one in (T2.7).
pub use sdm_core::{
    ArrowData, Attachment, Attachments, CompositionRoot, dtype_names, meanings, wire_names,
};
pub use zones::{ZoneError, ZoneResolver, ZoneRoot, ZoneSource, session_zone_offset};

/// A Postgres `json` or `jsonb` value, given as the server's text, as the
/// reference's provider hands it to the core (`sdm_data::json_text`, the
/// reference's float text): for a host whose driver gives the server's
/// text (T3.12 fix 1), so how the value is read stays in the data layer.
/// `None` where the text is not JSON; `Some(None)` for a JSON null.
pub fn json_value_text(text: &str) -> Option<Option<String>> {
    let read = || sdm_data::json_text(text, sdm_core::float_text);
    // The reader recurses once per level: a deep value is read on a stack
    // of its own, as the core reads a deep query, a shallow one on the
    // caller's (a JVM thread's stack is small).
    if nesting(text) > SHALLOW {
        sdm_core::pql::on_adapter_stack(read)
    } else {
        read()
    }
}

/// The deepest nesting read on the caller's stack.
const SHALLOW: usize = 64;

/// How deep `text`'s arrays and objects nest, its strings skipped.
fn nesting(text: &str) -> usize {
    let (mut depth, mut deepest, mut quoted, mut escaped) = (0usize, 0usize, false, false);
    for byte in text.bytes() {
        if quoted {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => quoted = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => quoted = true,
            b'[' | b'{' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    deepest
}

/// The table of what every sentence says, which a binding exports as the
/// JSON it is (`sdm_contract::wording`, T3.17): a host reads it once and
/// fills each error's and warning's sentence from it, formatting nothing
/// the table does not name.
pub fn templates() -> String {
    sdm_contract::wording::table_json_naming(sdm_data::DATABASE_NAMES)
}

/// What each model supports, as the client describes it to a caller
/// (`capabilities()`; Python's `ModelCapabilities`), as the JSON a binding
/// exports: `{"models": [...]}`, each model's `model`, `tasks` and
/// `outputs` (sorted), `single_table`, `multiple_tables`, `relationships`,
/// `explanation_formats`, `sessions` and `source` (`sdk`: the client side,
/// never the endpoint's answer), in the core's order of the models.
pub fn capabilities() -> String {
    let models: Vec<serde_json::Value> = sdm_core::capabilities()
        .into_iter()
        .map(|each| {
            serde_json::json!({
                "model": each.model,
                "tasks": each.tasks,
                "outputs": each.outputs,
                "single_table": each.single_table,
                "multiple_tables": each.multiple_tables,
                "relationships": each.relationships,
                "explanation_formats": each.explanation_formats,
                "sessions": each.sessions,
                "source": "sdk",
            })
        })
        .collect();
    serde_json::json!({ "models": models }).to_string()
}

/// The databases the data layer of [`default_root`] provides for
/// (`sdm_data::PROVIDERS`), which a binding reports as plain data.
pub fn providers() -> &'static [&'static str] {
    sdm_data::PROVIDERS
}

/// `query` as the PQL library writes it back (`sdm_core::pql::normalized`),
/// or `None` when it does not parse: the text a host sends materialize.
pub fn normalized_query(query: &str) -> Option<String> {
    sdm_core::pql::normalized(query)
}

/// The composition root a binding starts its runs with: the one data layer,
/// `sdm-data`'s, built here where every binding reaches it (decisions E21,
/// H14, E30).
pub fn default_root() -> CompositionRoot {
    CompositionRoot::new(std::sync::Arc::new(sdm_data::Data::reading_views(
        sdm_core::pql::spaces(),
    )))
}

#[cfg(test)]
mod tests {
    #[test]
    fn capabilities_are_every_models_in_the_clients_shape() {
        let read: serde_json::Value = serde_json::from_str(&super::capabilities()).unwrap();
        let models = read["models"].as_array().unwrap();
        let names: Vec<_> = models
            .iter()
            .map(|m| m["model"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["kumo-relational", "kumo-tabular"]);
        for model in models {
            let keys: Vec<_> = model.as_object().unwrap().keys().cloned().collect();
            assert_eq!(
                keys.len(),
                9,
                "model, tasks, outputs, three shapes, formats, sessions, source: {keys:?}"
            );
            assert_eq!(model["source"], "sdk");
            assert_eq!(model["single_table"], true);
        }
        assert_eq!(models[1]["multiple_tables"], false);
        assert_eq!(models[1]["relationships"], false);
    }
}
