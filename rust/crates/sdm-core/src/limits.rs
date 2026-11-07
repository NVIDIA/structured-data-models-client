// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The size limits a run holds a request to, resolved from what the host
//! passed with its start ([`GivenLimits`]): the caller's value, else the
//! environment's, else the default ([`sdm_contract::limits::SETTINGS`]).
//! Every host reads its limits through this one rule.

use sdm_contract::limits::{
    GivenLimits, Limits, MAX_RELATIONAL_CONTEXT_ROWS, MAX_RELATIONAL_PREDICT_ROWS,
    MAX_REQUEST_BYTES, MAX_TABULAR_CONTEXT_ROWS, MAX_TEMPORAL_LINK_PREDICT_ROWS, SETTINGS, Setting,
};
use sdm_contract::{CoreError, Detail, Sentence};

/// The limits `given` names, or the refusal of an environment variable that
/// is not a whole number of at least one, naming the setting and the
/// variable. A name that is no setting's, or a value given below one, is
/// the host's fault, `INVALID_MESSAGE`: a host checks its caller's values.
pub fn resolve(given: &GivenLimits) -> Result<Limits, CoreError> {
    for (name, &value) in &given.given {
        if value == 0 || !SETTINGS.iter().any(|setting| setting.name == name) {
            return Err(malformed("limits.given", name));
        }
    }
    for env in given.environment.keys() {
        if !SETTINGS.iter().any(|setting| setting.env == env) {
            return Err(malformed("limits.environment", env));
        }
    }
    let pick = |setting: &Setting| -> Result<u64, CoreError> {
        if let Some(&value) = given.given.get(setting.name) {
            return Ok(value);
        }
        let Some(text) = given.environment.get(setting.env) else {
            return Ok(setting.default);
        };
        whole(text).ok_or_else(|| {
            setting.named(
                CoreError::new(
                    Sentence::LimitEnvironmentInvalid,
                    "a limit's environment variable is not a whole number of at least one",
                )
                .with_field("given", Detail::text(text)),
            )
        })
    };
    Ok(Limits {
        max_relational_context_rows: pick(&MAX_RELATIONAL_CONTEXT_ROWS)?,
        max_relational_predict_rows: pick(&MAX_RELATIONAL_PREDICT_ROWS)?,
        max_temporal_link_predict_rows: pick(&MAX_TEMPORAL_LINK_PREDICT_ROWS)?,
        max_tabular_context_rows: pick(&MAX_TABULAR_CONTEXT_ROWS)?,
        max_request_bytes: pick(&MAX_REQUEST_BYTES)?,
    })
}

/// `text` as a whole number of at least one: decimal digits, with `_`
/// allowed between them as Python and Java write a long number, and white
/// space around them; anything else, zero, or a number past `u64`, none.
fn whole(text: &str) -> Option<u64> {
    let digits = text.trim();
    let well_formed = digits.as_bytes().first().is_some_and(u8::is_ascii_digit)
        && digits.as_bytes().last().is_some_and(u8::is_ascii_digit)
        && digits.bytes().all(|b| b.is_ascii_digit() || b == b'_')
        && !digits.contains("__");
    if !well_formed {
        return None;
    }
    digits
        .replace('_', "")
        .parse::<u64>()
        .ok()
        .filter(|&value| value >= 1)
}

fn malformed(field: &str, name: &str) -> CoreError {
    CoreError::protocol(
        Sentence::InvalidMessage,
        "the start names a limit that is no setting's, or one below one",
    )
    .with_field("field", Detail::text(field))
    .with_field("name", Detail::text(name))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use sdm_contract::ErrorCode;

    use super::*;

    fn given(given: &[(&str, u64)], environment: &[(&str, &str)]) -> GivenLimits {
        GivenLimits {
            given: given.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
            environment: environment
                .iter()
                .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
                .collect::<BTreeMap<_, _>>(),
        }
    }

    #[test]
    fn nothing_given_is_every_default() {
        assert_eq!(resolve(&GivenLimits::default()).unwrap(), Limits::default());
    }

    #[test]
    fn the_caller_comes_before_the_environment_before_the_default() {
        let env = MAX_TABULAR_CONTEXT_ROWS.env;
        let from_env = resolve(&given(&[], &[(env, "200000")])).unwrap();
        assert_eq!(from_env.max_tabular_context_rows, 200_000);
        assert_eq!(from_env.max_request_bytes, MAX_REQUEST_BYTES.default);
        let both = resolve(&given(
            &[(MAX_TABULAR_CONTEXT_ROWS.name, 50_000)],
            &[(env, "200000")],
        ))
        .unwrap();
        assert_eq!(both.max_tabular_context_rows, 50_000);
    }

    #[test]
    fn environment_text_is_a_whole_number_of_at_least_one() {
        let env = MAX_REQUEST_BYTES.env;
        for (text, expected) in [
            ("1", Some(1)),
            (" 64000000 ", Some(64_000_000)),
            ("200_000", Some(200_000)),
            ("0", None),
            ("", None),
            ("-1", None),
            ("1e6", None),
            ("2.5", None),
            ("_1", None),
            ("1__0", None),
            ("ten", None),
            ("18446744073709551616", None),
        ] {
            let resolved = resolve(&given(&[], &[(env, text)]));
            match expected {
                Some(value) => assert_eq!(resolved.unwrap().max_request_bytes, value, "{text:?}"),
                None => {
                    let error = resolved.unwrap_err();
                    assert_eq!(
                        error.sentence,
                        Sentence::LimitEnvironmentInvalid,
                        "{text:?}"
                    );
                    assert_eq!(error.code, ErrorCode::InvalidConfiguration);
                    assert_eq!(error.fields["env"], Detail::text(env));
                    assert_eq!(
                        error.fields["setting"],
                        Detail::text(MAX_REQUEST_BYTES.name)
                    );
                    assert_eq!(error.fields["given"], Detail::text(text));
                }
            }
        }
    }

    #[test]
    fn a_given_zero_and_an_unknown_name_are_the_host_s_fault() {
        let error = resolve(&given(&[(MAX_REQUEST_BYTES.name, 0)], &[])).unwrap_err();
        assert_eq!(error.sentence, Sentence::InvalidMessage);
        let error = resolve(&given(&[("max_tables", 20)], &[])).unwrap_err();
        assert_eq!(error.sentence, Sentence::InvalidMessage);
        let error = resolve(&given(&[], &[("SDM_CLIENT_MAX_HOPS", "8")])).unwrap_err();
        assert_eq!(error.sentence, Sentence::InvalidMessage);
    }
}
