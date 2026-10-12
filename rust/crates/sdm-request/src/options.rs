// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The call's options that shape the payload, read by the reference's rules
//! (T1.28a): the run mode a context size selects, the regression's
//! statistic, and the inference options (`inference_config`), which the
//! engine reads into its config classes (`InferenceConfig`,
//! `ClassificationInferenceConfig`, `RegressionInferenceConfig`) and writes
//! into the payload (`_inference_config_to_json`).
//!
//! Each refusal is a code and fields naming the option and why, never the
//! value given; the host words it.

use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};
use sdm_wire::json::{Value, whole};
use sdm_wire::payload::{ClassificationConfig, InferenceConfig};

/// A task's kind, as the engine names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Regression,
    Binary,
    Multiclass,
}

impl Kind {
    /// The kind's name on the wire (`TaskType`'s value).
    pub fn wire_name(self) -> &'static str {
        match self {
            Kind::Regression => "regression",
            Kind::Binary => "binary_classification",
            Kind::Multiclass => "multiclass_classification",
        }
    }

    pub fn is_classification(self) -> bool {
        !matches!(self, Kind::Regression)
    }
}

/// How much context a run gathers, and how wide it samples
/// (`_MAX_CONTEXT_SIZE`, `_DEFAULT_NUM_NEIGHBORS`): the client's two modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunMode {
    Fast,
    Best,
}

impl RunMode {
    /// The mode's name on the wire.
    pub fn wire_name(self) -> &'static str {
        match self {
            RunMode::Fast => "fast",
            RunMode::Best => "best",
        }
    }

    /// The most context examples the mode gathers.
    pub fn context_examples(self) -> usize {
        match self {
            RunMode::Fast => 1_000,
            RunMode::Best => 10_000,
        }
    }

    /// The neighbors kept of each row at each of the default two hops.
    pub fn neighbors(self) -> [u64; 2] {
        match self {
            RunMode::Fast => [32, 32],
            RunMode::Best => [64, 64],
        }
    }
}

/// The run mode a context size asks for, as the client chooses it
/// (`_run_mode_for`, with `num_context_samples`' bound of
/// `_validate_materialize_args`): none or up to a fast run's examples, a
/// fast run; up to a best run's, a best one. Below 1 is `INVALID_REQUEST`;
/// past a best run's examples, `UNSUPPORTED_REQUEST`, as the endpoint takes
/// no more. `asked` is the size as the contract carries it, any integer:
/// its sign and its value, `None` past `u128`.
pub fn run_mode(asked: Option<(bool, Option<u128>)>) -> Result<RunMode, CoreError> {
    let Some((negative, value)) = asked else {
        return Ok(RunMode::Fast);
    };
    if negative || value == Some(0) {
        return Err(CoreError::new(
            ErrorKind::InvalidRequest,
            ErrorCode::InvalidRequest,
            "num_context_samples is at least 1",
        )
        .with_field("field", Detail::text("num_context_samples")));
    }
    match value {
        Some(n) if n <= RunMode::Fast.context_examples() as u128 => Ok(RunMode::Fast),
        Some(n) if n <= RunMode::Best.context_examples() as u128 => Ok(RunMode::Best),
        _ => Err(CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "num_context_samples is past the most context the endpoint takes",
        )
        .with_field("field", Detail::text("num_context_samples"))
        .with_field(
            "limit",
            Detail::Scalar(Scalar::UInt64(RunMode::Best.context_examples() as u64)),
        )),
    }
}

/// The most hops the endpoint reads (`hops_max`).
pub const HOPS_MAX: u128 = 6;

/// The hops the sampling options ask for, as the client bounds them
/// (`_check_shape`): `num_neighbors`' length where it is given, else
/// `num_hops`; none, or fewer than one, is `INVALID_REQUEST` (a request
/// that crosses no relationship), more than [`HOPS_MAX`]
/// `UNSUPPORTED_REQUEST`. Each is `(negative, value)` as [`run_mode`] takes
/// it.
pub fn hops(
    num_hops: Option<(bool, Option<u128>)>,
    num_neighbors: Option<(bool, Option<u128>)>,
) -> Result<(), CoreError> {
    let (field, (negative, value)) = match (num_neighbors, num_hops) {
        (Some(n), _) => ("num_neighbors", n),
        (None, Some(n)) => ("num_hops", n),
        (None, None) => return Ok(()),
    };
    if negative || value == Some(0) {
        return Err(invalid(field, "no_hop"));
    }
    if value.is_none_or(|n| n > HOPS_MAX) {
        return Err(CoreError::new(
            ErrorKind::Unsupported,
            ErrorCode::UnsupportedRequest,
            "the sampling options ask more hops than the endpoint reads",
        )
        .with_field("field", Detail::text(field))
        .with_field("limit", Detail::Scalar(Scalar::UInt64(HOPS_MAX as u64))));
    }
    Ok(())
}

/// The statistics a regression reports of its predicted distribution
/// (`PREDICTION_STATISTICS`).
pub const STATISTICS: [&str; 2] = ["mean", "median"];

fn invalid(field: &'static str, reason: &'static str) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "an option of the call is not one the engine reads",
    )
    .with_field("field", Detail::text(field))
    .with_field("reason", Detail::text(reason))
}

/// The statistic the caller chose, refused unless the endpoint reports it
/// (`_statistic_asked_for`).
pub fn statistic(given: Option<&str>) -> Result<Option<String>, CoreError> {
    match given {
        None => Ok(None),
        Some(name) if STATISTICS.contains(&name) => Ok(Some(name.to_owned())),
        Some(_) => Err(invalid("prediction_statistic", "not_a_statistic")),
    }
}

/// A statistic chosen for a task with no distribution to summarise
/// (`_record_the_statistic`): refused, as the reference refuses it.
pub fn statistic_for(kind: Kind, statistic: Option<&String>) -> Result<(), CoreError> {
    if statistic.is_some() && kind != Kind::Regression {
        return Err(invalid("prediction_statistic", "not_a_regression")
            .with_field("task", Detail::text(kind.wire_name())));
    }
    Ok(())
}

/// The inference options, read as the engine's config classes read them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub num_estimators: u64,
    pub column_shuffle: bool,
    pub category_shuffle: bool,
    pub hop_shuffle: bool,
    /// A classification's.
    pub class_shuffle: bool,
    /// A regression's.
    pub target_transforms: Vec<String>,
    /// A regression's.
    pub output_type: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            num_estimators: 1,
            column_shuffle: false,
            category_shuffle: false,
            hop_shuffle: false,
            class_shuffle: false,
            target_transforms: vec!["quantile".to_owned()],
            output_type: "median".to_owned(),
        }
    }
}

const TRANSFORMS: [&str; 3] = ["clip", "power", "quantile"];
const OUTPUT_TYPES: [&str; 3] = ["mean", "median", "quantiles"];

/// A refusal of the inference option `key`: `INVALID_REQUEST`, as the
/// engine's validation error is the caller's (a `ValueError`).
fn option_refused(key: &str, reason: &'static str) -> CoreError {
    invalid("inference_config", reason).with_field("key", Detail::text(key))
}

/// A value the engine's validation coerces from text, which this core does
/// not read (pydantic's lax text coercions): refused by name.
fn coerced_from_text(key: &str) -> CoreError {
    sdm_engine::not_yet("inference_config_text").with_field("key", Detail::text(key))
}

/// A boolean option: a JSON boolean, or a number equal to 0 or 1, as the
/// engine's validation takes one.
fn boolean(key: &str, value: &Value) -> Result<bool, CoreError> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Number(token) => {
            let n: f64 = token.parse().unwrap_or(f64::NAN);
            if n == 0.0 {
                Ok(false)
            } else if n == 1.0 {
                Ok(true)
            } else {
                Err(option_refused(key, "not_a_boolean"))
            }
        }
        Value::Text(_) => Err(coerced_from_text(key)),
        _ => Err(option_refused(key, "not_a_boolean")),
    }
}

/// `num_estimators`: a whole number from 1 to 4, given as an integer, a
/// float with no fraction or a boolean (as 0 or 1), as the engine's
/// validation takes it.
fn estimators(key: &str, value: &Value) -> Result<u64, CoreError> {
    let n = match value {
        Value::Bool(b) => i128::from(*b),
        Value::Number(token) => {
            whole(token).ok_or_else(|| option_refused(key, "not_a_whole_number"))?
        }
        Value::Text(_) => return Err(coerced_from_text(key)),
        _ => return Err(option_refused(key, "not_a_whole_number")),
    };
    if !(1..=4).contains(&n) {
        return Err(option_refused(key, "out_of_range"));
    }
    Ok(n as u64)
}

/// Reads the caller's `inference_config` for a task of `kind`: the keys its
/// config class holds, each value as the engine's validation takes it, the
/// rest the class's defaults; a key the class does not hold, or a value it
/// refuses, is `INVALID_REQUEST`. `None` is the defaults.
pub fn options(kind: Kind, given: Option<&Value>) -> Result<Options, CoreError> {
    let mut options = Options::default();
    let Some(given) = given else {
        return Ok(options);
    };
    let Value::Object(members) = given else {
        return Err(invalid("inference_config", "not_an_object"));
    };
    for (key, value) in members {
        match key.as_str() {
            "num_estimators" => options.num_estimators = estimators(key, value)?,
            "column_shuffle" => options.column_shuffle = boolean(key, value)?,
            "category_shuffle" => options.category_shuffle = boolean(key, value)?,
            "hop_shuffle" => options.hop_shuffle = boolean(key, value)?,
            "class_shuffle" if kind.is_classification() => {
                options.class_shuffle = boolean(key, value)?;
            }
            "target_transforms" if kind == Kind::Regression => {
                let Value::Array(items) = value else {
                    return Err(option_refused(key, "not_a_list"));
                };
                options.target_transforms = items
                    .iter()
                    .map(|item| match item {
                        // A null transform, which the engine's type takes
                        // and sends as null, is refused by name: the
                        // payload's transforms are names (register P1B-3).
                        Value::Null => Err(sdm_engine::not_yet("target_transform_null")),
                        Value::Text(name) if TRANSFORMS.contains(&name.as_str()) => {
                            Ok(name.clone())
                        }
                        _ => Err(option_refused(key, "not_a_transform")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "output_type" if kind == Kind::Regression => match value {
                Value::Text(name) if OUTPUT_TYPES.contains(&name.as_str()) => {
                    options.output_type = name.clone();
                }
                _ => return Err(option_refused(key, "not_an_output_type")),
            },
            _ => return Err(option_refused(key, "not_an_option")),
        }
    }
    Ok(options)
}

impl Options {
    /// A regression's options as the engine writes them.
    pub fn regression(&self) -> InferenceConfig {
        InferenceConfig {
            kind: "regression".to_owned(),
            num_estimators: self.num_estimators,
            column_shuffle: self.column_shuffle,
            category_shuffle: self.category_shuffle,
            hop_shuffle: self.hop_shuffle,
            class_shuffle: None,
            target_transforms: self.target_transforms.clone(),
            output_type: self.output_type.clone(),
        }
    }

    /// A classification's options as the engine writes them.
    pub fn classification(&self) -> ClassificationConfig {
        ClassificationConfig {
            kind: "classification".to_owned(),
            num_estimators: self.num_estimators,
            column_shuffle: self.column_shuffle,
            category_shuffle: self.category_shuffle,
            hop_shuffle: self.hop_shuffle,
            class_shuffle: self.class_shuffle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Value {
        sdm_wire::json::read(text.as_bytes()).unwrap()
    }

    fn rule(e: &CoreError) -> Option<String> {
        e.fields.get("rule").map(|d| format!("{d:?}"))
    }

    #[test]
    fn the_run_mode_is_the_client_s() {
        assert_eq!(run_mode(None).unwrap(), RunMode::Fast);
        assert_eq!(run_mode(Some((false, Some(1)))).unwrap(), RunMode::Fast);
        assert_eq!(run_mode(Some((false, Some(1_000)))).unwrap(), RunMode::Fast);
        assert_eq!(run_mode(Some((false, Some(1_001)))).unwrap(), RunMode::Best);
        assert_eq!(
            run_mode(Some((false, Some(10_000)))).unwrap(),
            RunMode::Best
        );
        for (asked, code) in [
            (Some((false, Some(0))), ErrorCode::InvalidRequest),
            (Some((true, Some(5))), ErrorCode::InvalidRequest),
            (Some((false, Some(10_001))), ErrorCode::UnsupportedRequest),
            (Some((false, None)), ErrorCode::UnsupportedRequest),
        ] {
            assert_eq!(run_mode(asked).unwrap_err().code, code, "{asked:?}");
        }
    }

    #[test]
    fn the_hops_are_bounded_as_the_client_bounds_them() {
        let some = |n: u128| Some((false, Some(n)));
        assert!(hops(None, None).is_ok());
        assert!(hops(some(6), None).is_ok());
        assert!(
            hops(some(9), some(2)).is_ok(),
            "the neighbors' length counts"
        );
        assert_eq!(
            hops(some(0), None).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        assert_eq!(
            hops(None, some(0)).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        assert_eq!(
            hops(Some((true, Some(1))), None).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        assert_eq!(
            hops(some(7), None).unwrap_err().code,
            ErrorCode::UnsupportedRequest
        );
        assert_eq!(
            hops(Some((false, None)), None).unwrap_err().code,
            ErrorCode::UnsupportedRequest
        );
    }

    #[test]
    fn the_options_are_read_as_the_engine_reads_them() {
        let o = options(Kind::Regression, Some(&read(r#"{"num_estimators": 4.0, "column_shuffle": 1, "target_transforms": ["power", "clip"], "output_type": "quantiles"}"#))).unwrap();
        assert_eq!(o.num_estimators, 4);
        assert!(o.column_shuffle);
        assert_eq!(
            o.target_transforms,
            vec!["power".to_owned(), "clip".to_owned()]
        );
        assert_eq!(o.output_type, "quantiles");
        assert_eq!(
            options(
                Kind::Binary,
                Some(&read(r#"{"num_estimators": true, "class_shuffle": 0.0}"#))
            )
            .unwrap()
            .num_estimators,
            1
        );
        for (kind, text) in [
            (Kind::Regression, r#"{"num_estimators": 5}"#),
            (Kind::Regression, r#"{"num_estimators": 0}"#),
            (Kind::Regression, r#"{"num_estimators": 4.5}"#),
            (Kind::Regression, r#"{"num_estimators": null}"#),
            (Kind::Regression, r#"{"column_shuffle": 2}"#),
            (Kind::Regression, r#"{"class_shuffle": true}"#),
            (Kind::Binary, r#"{"output_type": "mean"}"#),
            (Kind::Binary, r#"{"target_transforms": []}"#),
            (Kind::Regression, r#"{"target_transforms": "clip"}"#),
            (Kind::Regression, r#"{"target_transforms": ["x"]}"#),
            (Kind::Regression, r#"{"output_type": "x"}"#),
            (Kind::Regression, r#"{"foo": 1}"#),
            (Kind::Regression, r#"[1]"#),
        ] {
            let e = options(kind, Some(&read(text))).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidRequest, "{text}");
            assert_eq!(rule(&e), None, "{text}");
        }
        for text in [
            r#"{"column_shuffle": "yes"}"#,
            r#"{"num_estimators": "4"}"#,
            r#"{"target_transforms": [null]}"#,
        ] {
            let e = options(Kind::Regression, Some(&read(text))).unwrap_err();
            assert_eq!(e.code, ErrorCode::UnsupportedRequest, "{text}");
            assert!(rule(&e).is_some(), "{text}");
        }
    }

    #[test]
    fn a_wide_object_is_refused_at_its_first_key_the_engine_does_not_read() {
        // E49's width probe: a hundred thousand members, read once, refused
        // by name at the first the engine's config does not hold.
        let members: Vec<String> = (0..100_000).map(|i| format!(r#""k{i}": {i}"#)).collect();
        let wide = read(&format!("{{{}}}", members.join(",")));
        let e = options(Kind::Regression, Some(&wide)).unwrap_err();
        assert_eq!(e.code, ErrorCode::InvalidRequest);
        assert_eq!(e.fields["key"], Detail::text("k0"));
    }

    #[test]
    fn a_statistic_is_one_the_endpoint_reports_and_a_regression_s() {
        assert_eq!(statistic(Some("mean")).unwrap().as_deref(), Some("mean"));
        assert_eq!(statistic(None).unwrap(), None);
        assert_eq!(
            statistic(Some("mode")).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        let chosen = Some("median".to_owned());
        assert!(statistic_for(Kind::Regression, chosen.as_ref()).is_ok());
        assert_eq!(
            statistic_for(Kind::Binary, chosen.as_ref())
                .unwrap_err()
                .code,
            ErrorCode::InvalidRequest
        );
    }
}
