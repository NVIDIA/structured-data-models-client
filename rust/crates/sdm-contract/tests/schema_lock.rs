// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The regeneration and lock checks of the two versioned schemas: the
//! contract's (`contract.lock`) and the PQL validated-query format's
//! (`validated-query.lock`).
//!
//! A lock entry holds the SHA-256 of a projection of the schema that strips
//! only the annotation keywords (`description`, `title`, `examples`), as
//! compact JSON with sorted keys, so a doc comment needs no new version
//! while a change of shape does (decision H7); `contract_lock.py` computes
//! the same.
//!
//! `SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock -- generated`
//! writes the schema of each current version and sets its lock entry: the
//! working entry of a phase is rewritten at each bump (decision H4). That
//! no entry merged into `rust-core` changes, and that a lock holds at most
//! one working entry, is checked by `scripts/rust_core/contract_lock.py`.
//!
//! From T1.22 the contract's schema holds the JSON of the attachments the
//! control messages name (`$defs/Attachments`): every field naming an
//! attachment is accounted for, and a field added to an attachment's
//! definitions fails the lock, each seeded below.

#![allow(
    clippy::disallowed_methods,
    reason = "test code: reads and, with SDM_CONTRACT_WRITE, rewrites the committed schema and lock"
)]

use std::path::{Path, PathBuf};

use sdm_contract::{
    CONTRACT_VERSION, VALIDATED_QUERY_VERSION, schema_json, validated_query_schema_json,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

/// One versioned schema and its lock.
struct Format {
    /// The schema file's name before `-<version>.json`.
    prefix: &'static str,
    lock: &'static str,
    version: u32,
    generate: fn() -> String,
}

const CONTRACT: Format = Format {
    prefix: "sdm-contract",
    lock: "contract.lock",
    version: CONTRACT_VERSION,
    generate: schema_json,
};

const VALIDATED_QUERY: Format = Format {
    prefix: "pql-validated-query",
    lock: "validated-query.lock",
    version: VALIDATED_QUERY_VERSION,
    generate: validated_query_schema_json,
};

const FORMATS: [Format; 2] = [CONTRACT, VALIDATED_QUERY];

fn schema_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("schema")
}

impl Format {
    fn schema_file(&self, version: u32) -> String {
        format!("{}-{version}.json", self.prefix)
    }
}

/// Keywords whose value is documentation only.
const ANNOTATIONS: [&str; 3] = ["description", "title", "examples"];
/// Keywords whose value maps names to schemas: the names are not keywords.
const NAMED: [&str; 5] = [
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];
/// Keywords whose value is data, not a schema.
const DATA: [&str; 3] = ["enum", "const", "default"];

/// The schema `node` without its annotation keywords.
fn project(node: Value) -> Value {
    match node {
        Value::Object(map) => {
            let mut out = Map::new();
            for (key, value) in map {
                if ANNOTATIONS.contains(&key.as_str()) {
                    continue;
                }
                let value = match value {
                    Value::Object(named) if NAMED.contains(&key.as_str()) => Value::Object(
                        named
                            .into_iter()
                            .map(|(name, schema)| (name, project(schema)))
                            .collect(),
                    ),
                    value if DATA.contains(&key.as_str()) => value,
                    value => project(value),
                };
                out.insert(key, value);
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(project).collect()),
        other => other,
    }
}

/// Sorts every object's keys, as Python's `sort_keys` does.
fn sorted(node: Value) -> Value {
    match node {
        Value::Object(map) => {
            let mut map: Map<String, Value> =
                map.into_iter().map(|(k, v)| (k, sorted(v))).collect();
            map.sort_keys();
            Value::Object(map)
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
        other => other,
    }
}

/// The lock hash of a schema's text.
fn sha256(text: &str) -> String {
    let schema: Value = serde_json::from_str(text).expect("a schema is JSON");
    let canonical = serde_json::to_string(&sorted(project(schema))).expect("JSON serializes");
    let digest = Sha256::digest(canonical.as_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The lock's entries: `(version, sha256, file)`, in file order.
fn parse_lock(text: &str) -> Result<Vec<(u32, String, String)>, String> {
    let mut entries = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let [version, hash, file] = parts[..] else {
            return Err(format!("not a lock entry: {line:?}"));
        };
        let version: u32 = version
            .parse()
            .map_err(|_| format!("not a version: {version:?}"))?;
        let hash = hash
            .strip_prefix("sha256:")
            .ok_or_else(|| format!("not a sha256: {hash:?}"))?;
        if entries.iter().any(|(v, _, _)| *v == version) {
            return Err(format!("version {version} is locked twice"));
        }
        entries.push((version, hash.to_owned(), file.to_owned()));
    }
    Ok(entries)
}

/// Checks that `version` of `format` is locked to `schema`.
fn verify(format: &Format, schema: &str, lock: &str, version: u32) -> Result<(), String> {
    let entries = parse_lock(lock)?;
    let Some((_, hash, file)) = entries.iter().find(|(v, _, _)| *v == version) else {
        return Err(format!(
            "{} version {version} has no lock entry; a schema change needs a version bump and a new entry",
            format.prefix
        ));
    };
    if *file != format.schema_file(version) {
        return Err(format!("version {version} is locked to {file}"));
    }
    if *hash != sha256(schema) {
        return Err(format!(
            "the schema of {} version {version} differs from its lock entry; \
             bump the version and add an entry instead of changing a locked version",
            format.prefix
        ));
    }
    Ok(())
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Writes the schema of `format`'s current version and sets its lock
/// entry, the phase's working entry (decision H4).
fn write(format: &Format, generated: &str) {
    let lock_path = schema_dir().join(format.lock);
    let lock = std::fs::read_to_string(&lock_path).unwrap_or_default();
    parse_lock(&lock).unwrap();
    let entry = format!(
        "{} sha256:{} {}",
        format.version,
        sha256(generated),
        format.schema_file(format.version)
    );
    let mut lines: Vec<String> = lock
        .lines()
        .filter(|line| line.split_whitespace().next() != Some(format.version.to_string().as_str()))
        .map(str::to_owned)
        .collect();
    if lines.is_empty() {
        lines.push(format!(
            "# The {} schema lock: one line per version; see contract_lock.py.",
            format.prefix
        ));
    }
    lines.push(entry);
    std::fs::write(&lock_path, lines.join("\n") + "\n").unwrap();
    std::fs::write(
        schema_dir().join(format.schema_file(format.version)),
        generated,
    )
    .unwrap();
}

#[test]
fn the_committed_schemas_are_the_generated_ones() {
    for format in &FORMATS {
        let generated = (format.generate)();
        if std::env::var_os("SDM_CONTRACT_WRITE").is_some() {
            write(format, &generated);
        }
        let path = schema_dir().join(format.schema_file(format.version));
        assert!(
            read(&path) == generated,
            "{} differs from the schema the types generate; regenerate it with \
             SDM_CONTRACT_WRITE=1 cargo test -p sdm-contract --test schema_lock -- generated",
            path.display()
        );
    }
}

#[test]
fn every_locked_version_matches_its_committed_schema() {
    for format in &FORMATS {
        let lock = read(&schema_dir().join(format.lock));
        let entries = parse_lock(&lock).unwrap();
        assert!(entries.iter().any(|(v, _, _)| *v == format.version));
        for (version, _, file) in &entries {
            let schema = read(&schema_dir().join(file));
            verify(format, &schema, &lock, *version).unwrap();
        }
    }
}

#[test]
fn a_schema_change_without_a_version_bump_fails() {
    for (format, word) in [(&CONTRACT, "\"step\""), (&VALIDATED_QUERY, "\"entity\"")] {
        let lock = read(&schema_dir().join(format.lock));
        let generated = (format.generate)();
        let changed = generated.replace(word, "\"renamed\"");
        assert_ne!(changed, generated);
        let err = verify(format, &changed, &lock, format.version).unwrap_err();
        assert!(err.contains("bump the version"), "{err}");
        // A bump with no entry fails too.
        let err = verify(format, &changed, &lock, format.version + 1).unwrap_err();
        assert!(err.contains("no lock entry"), "{err}");
    }
}

#[test]
fn a_malformed_or_doubled_lock_fails() {
    assert!(parse_lock("1 abc sdm-contract-1.json").is_err());
    assert!(
        parse_lock("1 sha256:ab sdm-contract-1.json\n1 sha256:cd sdm-contract-1.json").is_err()
    );
    assert!(parse_lock("one sha256:ab sdm-contract-1.json").is_err());
}

#[test]
fn a_doc_comment_change_keeps_the_hash_and_a_property_named_description_does_not() {
    let schema = r##"{
        "$defs": {
            "Note": {
                "description": "A note, as a doc comment describes it.",
                "type": "object",
                "properties": {
                    "description": {"type": "string", "title": "Description"},
                    "title": {"type": "string"}
                },
                "required": ["description", "title"],
                "examples": [{"description": "x", "title": "y"}]
            }
        },
        "oneOf": [{"$ref": "#/$defs/Note", "description": "one"}],
        "enum": [{"description": "data, not an annotation"}]
    }"##;
    let locked = sha256(schema);
    // Doc comments, titles and examples: the same hash.
    for edited in [
        schema.replace("A note, as a doc comment describes it.", "Reworded."),
        schema.replace(r#""title": "Description""#, r#""title": "Renamed""#),
        schema.replace(r#""description": "one""#, r#""description": "two""#),
        schema.replace(
            r#""examples": [{"description": "x", "title": "y"}]"#,
            r#""examples": []"#,
        ),
    ] {
        assert_ne!(edited, schema);
        assert_eq!(sha256(&edited), locked, "{edited}");
    }
    // A property called `description` or `title`, and data in `enum`, are
    // shape: a new hash.
    for edited in [
        schema.replace(
            r#""description": {"type": "string""#,
            r#""summary": {"type": "string""#,
        ),
        schema.replace(
            r#""title": {"type": "string"}"#,
            r#""label": {"type": "string"}"#,
        ),
        schema.replace(r#""required": ["description""#, r#""required": ["summary""#),
        schema.replace("data, not an annotation", "other data"),
    ] {
        assert_ne!(edited, schema);
        assert_ne!(sha256(&edited), locked, "{edited}");
    }
    // And on the real schema: a doc comment's text is not hashed.
    let generated = schema_json();
    let reworded = generated.replacen("\"description\": \"", "\"description\": \"Reworded. ", 1);
    assert_ne!(reworded, generated);
    assert_eq!(sha256(&reworded), sha256(&generated));
}

// --- The attachments the control messages name (T1.22) -----------------------

/// Whether `node` refers to an attachment id, directly, as one of a union's
/// members or as a list's items.
fn names_attachment(node: &Value) -> bool {
    if node.get("$ref").and_then(Value::as_str) == Some("#/$defs/AttachmentId") {
        return true;
    }
    ["anyOf", "oneOf"].iter().any(|key| {
        node.get(*key)
            .and_then(Value::as_array)
            .is_some_and(|members| members.iter().any(names_attachment))
    }) || node.get("items").is_some_and(names_attachment)
}

/// Every field of a control message that names an attachment, as
/// `<definition>[/<tag>].<field>`, the tag being the variant's constant.
fn attachment_fields(node: &Value, at: &str, found: &mut Vec<String>) {
    match node {
        Value::Object(map) => {
            if let Some(Value::Object(properties)) = map.get("properties") {
                let tag = properties
                    .values()
                    .find_map(|p| p.get("const").and_then(Value::as_str));
                for (name, property) in properties {
                    if names_attachment(property) {
                        found.push(match tag {
                            Some(tag) => format!("{at}/{tag}.{name}"),
                            None => format!("{at}.{name}"),
                        });
                    }
                }
            }
            for (key, value) in map {
                if key != "$defs" {
                    attachment_fields(value, at, found);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                attachment_fields(item, at, found);
            }
        }
        _ => {}
    }
}

/// The attachment fields of `schema` the classification below does not
/// account for, and the classified ones it no longer has.
fn unaccounted(schema: &Value) -> Result<(), String> {
    let mut found = Vec::new();
    attachment_fields(schema, "ControlMessage", &mut found);
    for (name, definition) in schema["$defs"].as_object().ok_or("no $defs")? {
        if name != "Attachments" {
            attachment_fields(definition, name, &mut found);
        }
    }
    let listed: Vec<&str> = schema["$defs"]["Attachments"]["properties"]
        .as_object()
        .ok_or("no attachments listed")?
        .keys()
        .map(String::as_str)
        .collect();
    let mut known: Vec<&str> = OTHER.iter().map(|(field, _)| *field).collect();
    for (field, attachment) in JSON {
        if !listed.contains(&attachment) {
            return Err(format!("{field}'s JSON ({attachment}) is not in the lock"));
        }
        known.push(field);
    }
    if listed.len() != JSON.len() {
        return Err("the lock lists an attachment no field names".into());
    }
    for field in &found {
        if !known.contains(&field.as_str()) {
            return Err(format!(
                "{field} names an attachment the lock does not account for"
            ));
        }
    }
    for field in known {
        if !found.iter().any(|f| f == field) {
            return Err(format!("{field} is accounted for and not in the schema"));
        }
    }
    Ok(())
}

/// The fields whose attachment is JSON of the contract's, with the entry of
/// `$defs/Attachments` that locks it.
const JSON: [(&str, &str); 9] = [
    ("Operation/validate_query.query", "ValidateQuery.query"),
    ("Operation/validate_query.schema", "ValidateQuery.schema"),
    ("Operation/describe_source.source", "DescribeSource.source"),
    ("Operation/materialize.schema", "Materialize.schema"),
    ("Operation/materialize.query", "Materialize.query"),
    ("RequestState.payload", "RequestState.payload"),
    ("RequestState.query", "RequestState.query"),
    (
        "OperationResult/describe_source.description",
        "DescribedSource.description",
    ),
    ("OperationResult/predict.metadata", "Predicted.metadata"),
];

/// The other fields naming an attachment, and why the lock needs no entry.
const OTHER: [(&str, &str); 7] = [
    (
        "ControlMessage/attached.attachment",
        "a control message: the root schema",
    ),
    ("ControlMessage/call_model.body", "the endpoint's bytes"),
    ("ModelOutcome/response.body", "the endpoint's bytes"),
    ("Operation/materialize.indices", "Arrow, with its schema"),
    ("FetchOutcome/data.data", "Arrow, with its schema"),
    ("KeySet.attachment", "Arrow, with its schema"),
    ("OperationResult/predict.result", "Arrow, with its schema"),
];

fn generated() -> Value {
    serde_json::from_str(&schema_json()).expect("the schema is JSON")
}

#[test]
fn the_lock_accounts_for_every_attachment_a_message_names() {
    assert_eq!(unaccounted(&generated()), Ok(()));
    // Seeded: a field naming a new attachment, unaccounted for, fails; so
    // does an attachment the lock lists that no field names, and a JSON
    // attachment the lock lost.
    let mut added = generated();
    added["$defs"]["SourceSpec"]["properties"]["extra"] =
        serde_json::json!({"$ref": "#/$defs/AttachmentId"});
    let err = unaccounted(&added).unwrap_err();
    assert!(err.contains("SourceSpec.extra"), "{err}");
    let mut listed = generated();
    listed["$defs"]["Attachments"]["properties"]["Spare.json"] = serde_json::json!({});
    assert!(unaccounted(&listed).unwrap_err().contains("no field names"));
    let mut lost = generated();
    lost["$defs"]["Attachments"]["properties"]
        .as_object_mut()
        .unwrap()
        .remove("DescribeSource.source");
    assert!(unaccounted(&lost).unwrap_err().contains("not in the lock"));
}

#[test]
fn a_field_added_to_an_attachment_fails_the_lock() {
    let lock = read(&schema_dir().join(CONTRACT.lock));
    let text = schema_json();
    verify(&CONTRACT, &text, &lock, CONTRACT.version).unwrap();
    // The JSON each locked attachment holds, and the definitions it is
    // made of: a field added to any of them is a new shape.
    for definition in [
        "SourceSpec",
        "SourceDescription",
        "DescribedTable",
        "DescribedColumn",
        "QuerySchema",
        "QueryTable",
        "QueryColumn",
        "ColumnRef",
        "StatePayload",
        "EndpointMetadata",
    ] {
        let mut schema = generated();
        let node = &mut schema["$defs"][definition];
        assert!(node.is_object(), "{definition} is not in the schema");
        node["properties"]["added"] = serde_json::json!({"type": "string"});
        let changed = serde_json::to_string_pretty(&schema).unwrap();
        let err = verify(&CONTRACT, &changed, &lock, CONTRACT.version).unwrap_err();
        assert!(err.contains("bump the version"), "{definition}: {err}");
    }
    // A new request state format is a new shape too.
    let mut schema = generated();
    schema["$defs"]["StatePayload"]["x-request-state-version"] = serde_json::json!(2);
    let changed = serde_json::to_string_pretty(&schema).unwrap();
    assert!(verify(&CONTRACT, &changed, &lock, CONTRACT.version).is_err());
}
