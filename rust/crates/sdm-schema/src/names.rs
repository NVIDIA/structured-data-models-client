// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a column's name says, by the reference's rules: whether it names an
//! identifier, a measure, a time, and whether a timestamp's name says when a
//! row was loaded, last changed, or someone was born
//! (`predictive_query.time_columns` and the relational engine's metadata
//! inference, ported word for word).

/// Whether `name` holds `word` as a whole `_`-separated part, in any case:
/// the reference's `(^|_)(word|…)(_|$)`, searched ignoring case.
pub(crate) fn has_part(name: &str, words: &[&str]) -> bool {
    name.split('_')
        .any(|part| words.iter().any(|word| part.eq_ignore_ascii_case(word)))
}

/// Name suffixes ending in `id` that do not make an identifier.
const IGNORED_ID_SUFFIXES: [&str; 16] = [
    "bid", "acid", "grid", "maid", "paid", "raid", "void", "avoid", "braid", "covid", "fluid",
    "rabid", "solid", "hybrid", "inlaid", "liquid",
];

/// Whether the name says identifier: `id`, `hash`, `key`, `code` or `uuid`
/// as a part, or a name ending in `id` that is not one of the words that
/// merely end so.
pub(crate) fn says_id(name: &str) -> bool {
    let lowered = name.to_lowercase();
    if has_part(&lowered, &["id", "hash", "key", "code", "uuid"]) {
        return true;
    }
    lowered.ends_with("id") && !IGNORED_ID_SUFFIXES.iter().any(|s| lowered.ends_with(s))
}

/// Whether the name says a time.
pub(crate) fn says_time(name: &str) -> bool {
    has_part(
        name,
        &[
            "date",
            "datetime",
            "dt",
            "time",
            "timedate",
            "timestamp",
            "ts",
            "created",
            "updated",
        ],
    )
}

/// Whether the name says a measure, which a number named so is, rather
/// than a category.
pub(crate) fn says_measure(name: &str) -> bool {
    has_part(
        name,
        &[
            "price", "sales", "amount", "quantity", "total", "cost", "score", "rating", "avg",
            "average", "recency", "age", "num", "pos", "number", "position",
        ],
    )
}

/// The lowercased words of a name, split on snake_case and camelCase, as
/// the reference's `[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+` finds
/// them: an acronym before a capitalised word, a word, an acronym, digits.
fn tokens(name: &str) -> Vec<String> {
    let chars: Vec<char> = name.chars().collect();
    let upper = |i: usize| chars.get(i).is_some_and(char::is_ascii_uppercase);
    let lower = |i: usize| chars.get(i).is_some_and(char::is_ascii_lowercase);
    let digit = |i: usize| chars.get(i).is_some_and(char::is_ascii_digit);
    let run = |mut i: usize, f: &dyn Fn(usize) -> bool| {
        while f(i) {
            i += 1;
        }
        i
    };
    let mut found = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = if upper(i) {
            let caps = run(i, &upper);
            if caps - i >= 2 && lower(caps) {
                // An acronym before a capitalised word: all but its last
                // capital.
                caps - 1
            } else if lower(i + 1) {
                run(i + 1, &lower)
            } else {
                caps
            }
        } else if lower(i) {
            run(i, &lower)
        } else if digit(i) {
            run(i, &digit)
        } else {
            i += 1;
            continue;
        };
        found.push(chars[i..end].iter().collect::<String>().to_lowercase());
        i = end;
    }
    found
}

/// A name in one case with no separators, as a catalog may fold one.
fn is_joined(name: &str) -> bool {
    let cased: Vec<char> = name
        .chars()
        .filter(|c| c.is_uppercase() || c.is_lowercase())
        .collect();
    !name.is_empty()
        && name.chars().all(char::is_alphanumeric)
        && !cased.is_empty()
        && (cased.iter().all(|c| c.is_uppercase()) || cased.iter().all(|c| c.is_lowercase()))
}

fn reads_as(name: &str, words: &[&str], prefixes: &[(&str, &[&str])]) -> bool {
    if is_joined(name) {
        let lowered = name.to_lowercase();
        for (prefix, other_words) in prefixes {
            if let Some(rest) = lowered.strip_prefix(prefix)
                && !other_words.iter().any(|w| rest.starts_with(w))
            {
                return true;
            }
        }
    }
    tokens(name).iter().any(|t| words.contains(&t.as_str()))
}

/// Whether the name says when a row was loaded, synced or refreshed.
pub(crate) fn is_load_stamp(name: &str) -> bool {
    reads_as(
        name,
        &[
            "synced",
            "sync",
            "loaded",
            "load",
            "reload",
            "reloaded",
            "ingest",
            "ingested",
            "ingestion",
            "batched",
            "extract",
            "extracted",
            "extraction",
            "etl",
            "refresh",
            "refreshdate",
            "refreshed",
        ],
        &[
            ("load", &["ing"]),
            ("reload", &["ing"]),
            ("sync", &["hron"]),
            ("etl", &[]),
            ("refresh", &["ing", "ment"]),
            ("ingest", &[]),
            ("extract", &[]),
            ("batched", &[]),
        ],
    )
}

/// Whether the name says when a row was last changed.
pub(crate) fn is_edit_stamp(name: &str) -> bool {
    let lowered = name.to_lowercase();
    if lowered.contains("modif") || lowered.contains("updat") {
        return true;
    }
    reads_as(
        name,
        &["last", "edit", "edited", "changed", "deleted", "deletion"],
        &[
            ("last", &["ing", "ly"]),
            ("edit", &["ion", "or"]),
            ("changed", &["ate"]),
            ("deleted", &["ate"]),
            ("deletion", &[]),
        ],
    )
}

/// Whether the name says when a row was created (`is_creation_stamp`):
/// it starts `create`, in any case, which keeps an all-capitals
/// `CREATEDDATE`, or holds `create`, `created` or `creation` as a word.
pub(crate) fn is_creation_stamp(name: &str) -> bool {
    name.to_lowercase().starts_with("create")
        || tokens(name)
            .iter()
            .any(|token| ["create", "created", "creation"].contains(&token.as_str()))
}

/// Whether the name says a date of birth.
pub(crate) fn is_birth_date(name: &str) -> bool {
    if is_joined(name) && name.to_lowercase().contains("birth") {
        return true;
    }
    reads_as(
        name,
        &["birth", "birthday", "birthdate", "dob", "born"],
        &[("dob", &[]), ("born", &[])],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_split_as_the_reference_splits() {
        assert_eq!(tokens("ETLLoadedAt"), ["etl", "loaded", "at"]);
        assert_eq!(tokens("LASTMODIFIEDDATE"), ["lastmodifieddate"]);
        assert_eq!(tokens("last_seen"), ["last", "seen"]);
        assert_eq!(tokens("lastingDate"), ["lasting", "date"]);
        assert_eq!(tokens("created_at2"), ["created", "at", "2"]);
        assert_eq!(tokens("A"), ["a"]);
        assert_eq!(tokens("ABc"), ["a", "bc"]);
    }

    #[test]
    fn stamps_read_as_the_reference_reads_them() {
        assert!(is_load_stamp("_fivetran_synced"));
        assert!(is_load_stamp("LOADDATE"));
        assert!(!is_load_stamp("LOADINGDATE"));
        assert!(is_edit_stamp("ModifiedDate"));
        assert!(is_edit_stamp("updated_at"));
        assert!(is_edit_stamp("last_seen"));
        assert!(!is_edit_stamp("lastingDate"));
        assert!(!is_edit_stamp("EDITIONDATE"));
        assert!(is_birth_date("DATEOFBIRTH"));
        assert!(is_birth_date("dob"));
        assert!(!is_load_stamp("ts"));
        assert!(!is_edit_stamp("ts"));
        assert!(!is_birth_date("ts"));
        assert!(!is_edit_stamp("created_at"));
        assert!(is_creation_stamp("CREATEDDATE"));
        assert!(is_creation_stamp("row_creation_ts"));
        assert!(is_creation_stamp("createdAt"));
        assert!(!is_creation_stamp("recreated"));
        assert!(!is_creation_stamp("ts"));
    }

    #[test]
    fn names_say_what_the_reference_reads_in_them() {
        assert!(says_id("order_id"));
        assert!(says_id("uid"));
        assert!(says_id("Customer_KEY"));
        assert!(!says_id("paid"));
        assert!(!says_id("country"));
        assert!(says_time("order_ts"));
        assert!(!says_time("tsx"));
        assert!(says_measure("age"));
        assert!(says_measure("Unit_Price"));
        assert!(!says_measure("ages"));
    }
}
