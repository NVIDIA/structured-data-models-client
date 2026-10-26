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

use sdm_contract::{CoreError, Detail, Scalar, Sentence};
use sdm_wire::json::{Value, is_integer_token, whole};
use sdm_wire::payload::{ClassificationConfig, InferenceConfig};

/// An integer the caller gave: its sign, and its magnitude, `None` past
/// `u128`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Integer {
    pub negative: bool,
    pub magnitude: Option<u128>,
}

/// The seed the call names (`random_seed`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AskedSeed {
    /// Not named: the host's `seed` and `unseeded`, or the default.
    Absent,
    /// Named as none: an unseeded sample.
    Unseeded,
    /// An integer of any size, as signed hexadecimal text (the contract's
    /// `Seed`).
    Given(String),
}

/// The `lag_timesteps` the call names (T2.9), as the reference's engine
/// meets it: it compares the value with zero once the context is labelled
/// (`_get_task_table`), and for a temporal query multiplies the target's
/// window by a positive one (`sample_lagged_target`), so each value is read
/// here and refused only where, and as, the engine refuses it
/// ([`AskedLag::refusal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskedLag {
    /// A whole number of steps (`false` and a float zero are none).
    Steps(u64),
    /// A whole number past `u64`, which the engine cannot multiply a window
    /// by (`OverflowError`).
    TooLarge,
    /// Below zero, an integer's or a float's: `'lag_timesteps' cannot be
    /// negative`.
    Negative,
    /// Above zero and not an integer (a float, or `true`): no window is
    /// multiplied by it (`TypeError`), so a temporal query is refused; a
    /// static one adds no lag, and takes it.
    NotWhole,
    /// Text, a null, a list or a mapping, which is not ordered against
    /// zero (`TypeError`), refused for any query.
    Unordered,
}

impl AskedLag {
    /// Reads the value given, as the engine meets it.
    fn of(value: &Value) -> AskedLag {
        match value {
            Value::Number(token) if sdm_wire::json::is_integer_token(token) => {
                let (negative, digits) = match token.strip_prefix('-') {
                    Some(digits) => (true, digits),
                    None => (false, token.as_str()),
                };
                let digits = digits.trim_start_matches('0');
                if digits.is_empty() {
                    AskedLag::Steps(0)
                } else if negative {
                    AskedLag::Negative
                } else {
                    digits.parse().map_or(AskedLag::TooLarge, AskedLag::Steps)
                }
            }
            // A float: below zero refused as negative, zero (either sign)
            // or a NaN no lag, which neither comparison takes, and any other
            // (an infinity too) not whole.
            Value::Number(token) => match token.parse::<f64>() {
                Ok(f) if f < 0.0 => AskedLag::Negative,
                Ok(f) if f > 0.0 => AskedLag::NotWhole,
                _ => AskedLag::Steps(0),
            },
            Value::Bool(true) => AskedLag::NotWhole,
            Value::Bool(false) => AskedLag::Steps(0),
            _ => AskedLag::Unordered,
        }
    }

    /// The engine's refusal of this value for a query that is `temporal`
    /// or not, as its client gives it (`INVALID_REQUEST`, naming
    /// `lag_timesteps` and why), or none: the steps a temporal query adds
    /// as lagged features, which are none for a static one.
    pub fn refusal(self, temporal: bool) -> Result<u64, CoreError> {
        let refused = |message: &str, reason: &'static str| {
            CoreError::new(Sentence::LagNegative, message)
                .with_field("field", Detail::text("lag_timesteps"))
                .with_field("reason", Detail::text(reason))
        };
        match self {
            AskedLag::Unordered => Err(mistyped(
                Sentence::LagNotANumber,
                "lag_timesteps",
                "not_a_number",
            )),
            AskedLag::Negative => Err(refused(
                "the past steps added as features cannot be negative",
                "negative",
            )),
            _ if !temporal => Ok(0),
            AskedLag::NotWhole => Err(mistyped(
                Sentence::LagNotWhole,
                "lag_timesteps",
                "not_a_whole_number",
            )),
            AskedLag::TooLarge => Err(CoreError::new(
                Sentence::LagPastTimes,
                "a time is outside what nanoseconds since the epoch hold",
            )
            .with_field("field", Detail::text("lag_timesteps"))),
            AskedLag::Steps(steps) => Ok(steps),
        }
    }
}

/// The context's anchor the call names (`context_anchor_time`, T2.11),
/// which the host writes as a `ts` scalar, the literal `entity`, or none,
/// having refused any other value as it refuses one for `anchor_time`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AskedAnchor {
    /// Not named, or named as none: the anchor less the target's window.
    Absent,
    /// Each entity's own time (`'entity'`).
    Entity,
    /// A time, nanoseconds since the epoch, its UTC instant.
    At(i64),
}

/// The call's option values the core reads ([`values`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Asked {
    pub num_context_samples: Option<Integer>,
    /// `num_context_samples` as the JSON text the host wrote it in, which
    /// its refusals name as the caller gave it (`given`).
    pub num_context_samples_given: Option<String>,
    pub num_hops: Option<Integer>,
    /// Whether `num_hops` was named as none, which the engine cannot
    /// compare with its bounds unless `num_neighbors` stands in for it.
    pub num_hops_none: bool,
    /// `num_hops` as its decimal digits, signed, where it is a number.
    pub num_hops_text: Option<String>,
    /// How many hops `num_neighbors` asks for: its length.
    pub num_neighbors: Option<Integer>,
    /// Each hop's neighbors `num_neighbors` asks for, in order, as given
    /// (T2.11), read by [`neighbors`] once the hops are bounded.
    pub neighbors: Option<Vec<Value>>,
    /// The sampling options the engine reads after the client's checks
    /// (T2.11), as given, each read where the reference reads it, so a
    /// refusal of one comes where the reference's does: `max_pq_iterations`
    /// and `context_anchor_time` with the task ([`task_table`]),
    /// `exclude_cols_dict` as the subgraph is sampled ([`excluded`]),
    /// `use_prediction_time` and `return_embeddings` as the request is
    /// built ([`request_flags`]).
    pub max_pq_iterations: Option<Value>,
    pub context_anchor_time: Option<Value>,
    pub exclude_cols_dict: Option<Value>,
    pub use_prediction_time: Option<Value>,
    pub return_embeddings: Option<Value>,
    pub prediction_statistic: Option<String>,
    pub inference_config: Option<Value>,
    pub seed: AskedSeed,
    /// How many candidates a ranking keeps (`top_k`), as given: read by
    /// [`top_k`] where the client reads it (T2.8).
    pub top_k: Option<Value>,
    /// The most rows one model call scores, as the caller lowered it
    /// (`rows_per_call`, `main`'s multi-call `predict`): a whole number of
    /// at least one, `u64::MAX` standing for any larger one, which lowers
    /// nothing ([`rows_per_call`]).
    pub rows_per_call: Option<u64>,
    /// The `lag_timesteps` given (T2.9), none where the call names none.
    pub lag_timesteps: Option<AskedLag>,
}

/// A value of the wrong type for its option, refused as the client
/// refuses it (`INVALID_REQUEST`), naming the option and the type it takes.
fn mistyped(sentence: Sentence, field: &str, reason: &'static str) -> CoreError {
    CoreError::new(
        sentence,
        "an option of the call is of a type it does not take",
    )
    .with_field("field", Detail::text(field))
    .with_field("reason", Detail::text(reason))
}

/// The sentence of an option that is not a whole number, by the option.
fn not_whole(field: &str) -> Sentence {
    match field {
        "num_context_samples" => Sentence::NumContextSamplesNotWhole,
        "num_hops" => Sentence::NumHopsNotWhole,
        "random_seed" => Sentence::RandomSeedNotWhole,
        "top_k" => Sentence::TopKNotWhole,
        "rows_per_call" => Sentence::RowsPerCall,
        "max_pq_iterations" => Sentence::MaxPqIterationsNotAWholeNumber,
        _ => Sentence::OptionNotWhole,
    }
}

/// A value the caller gave, as the JSON text the host wrote it in, which a
/// refusal names for the host's words (`given`).
fn given_text(value: &Value) -> Detail {
    Detail::text(String::from_utf8_lossy(&value.to_bytes()).into_owned())
}

/// An integer token's digits as decimal text, its sign apart.
fn integer(field: &str, value: &Value) -> Result<(bool, String), CoreError> {
    match value {
        Value::Number(token) if sdm_wire::json::is_integer_token(token) => {
            let (negative, digits) = match token.strip_prefix('-') {
                Some(digits) => (true, digits),
                None => (false, token.as_str()),
            };
            let digits = digits.trim_start_matches('0');
            Ok((negative && !digits.is_empty(), digits.to_owned()))
        }
        // A boolean, a float or anything else: not a whole number, as the
        // client's check (`numbers.Integral`, a bool excluded) refuses it,
        // naming the value given, for the host's words.
        _ => Err(mistyped(not_whole(field), field, "not_a_whole_number")
            .with_field("given", given_text(value))),
    }
}

fn magnitude(field: &str, value: &Value) -> Result<Integer, CoreError> {
    let (negative, digits) = integer(field, value)?;
    Ok(Integer {
        negative,
        magnitude: if digits.is_empty() {
            Some(0)
        } else {
            digits.parse().ok()
        },
    })
}

/// Decimal digits as lowercase hexadecimal, of any size.
fn hexadecimal(decimal: &str) -> String {
    let mut digits: Vec<u32> = decimal.bytes().map(|b| u32::from(b - b'0')).collect();
    let mut hex = Vec::new();
    while !digits.is_empty() {
        let mut remainder = 0;
        let mut quotient = Vec::with_capacity(digits.len());
        for d in digits {
            let value = remainder * 10 + d;
            let q = value / 16;
            remainder = value % 16;
            if !(quotient.is_empty() && q == 0) {
                quotient.push(q);
            }
        }
        hex.push(char::from_digit(remainder, 16).unwrap_or('0'));
        digits = quotient;
    }
    if hex.is_empty() {
        return "0".to_owned();
    }
    hex.iter().rev().collect()
}

/// Reads the call's option values (`Materialize.values`), each of the type
/// the reference's client takes, a member present only where the call
/// names the option; a value of another type is `INVALID_REQUEST` naming
/// the option, as the client's checks refuse it.
pub fn values(given: Option<&Value>) -> Result<Asked, CoreError> {
    let mut asked = Asked {
        num_context_samples: None,
        num_hops: None,
        num_hops_none: false,
        num_hops_text: None,
        num_context_samples_given: None,
        num_neighbors: None,
        neighbors: None,
        max_pq_iterations: None,
        context_anchor_time: None,
        exclude_cols_dict: None,
        use_prediction_time: None,
        return_embeddings: None,
        prediction_statistic: None,
        inference_config: None,
        seed: AskedSeed::Absent,
        top_k: None,
        rows_per_call: None,
        lag_timesteps: None,
    };
    let Some(given) = given else {
        return Ok(asked);
    };
    let Value::Object(members) = given else {
        return Err(mistyped(
            Sentence::OptionValuesUnread,
            "values",
            "not_an_object",
        ));
    };
    // Hosts preserve nonfinite numeric values in this one attachment.
    // KO3-44 admits only the reference's NaN lag, not other nonfinite
    // options or strings that happen to spell a nonfinite number.
    for (name, value) in members {
        let nan_lag =
            name == "lag_timesteps" && matches!(value, Value::Number(token) if token == "NaN");
        if !nan_lag && !value.all_finite() {
            return Err(CoreError::new(
                Sentence::OptionValueNotFinite,
                format!("{name} carries a value that cannot be sent as JSON (a NaN, an infinity, or a type JSON has no form for)"),
            )
            .with_field("field", Detail::text(name)));
        }
    }
    for (name, value) in members {
        match (name.as_str(), value) {
            // A size or a list of hops left as none is no size asked; a
            // count of hops named as none is the engine's to compare
            // ([`neighbors`]).
            ("num_context_samples" | "num_neighbors", Value::Null) => {}
            ("num_hops", Value::Null) => asked.num_hops_none = true,
            ("num_context_samples", v) => {
                asked.num_context_samples = Some(magnitude(name, v)?);
                asked.num_context_samples_given =
                    Some(String::from_utf8_lossy(&v.to_bytes()).into_owned());
            }
            ("num_hops", v) => {
                asked.num_hops = Some(magnitude(name, v)?);
                let (negative, digits) = integer(name, v)?;
                let digits = if digits.is_empty() { "0" } else { &digits };
                let sign = if negative { "-" } else { "" };
                asked.num_hops_text = Some(format!("{sign}{digits}"));
            }
            ("num_neighbors", Value::Array(hops)) => {
                asked.num_neighbors = Some(Integer {
                    negative: false,
                    magnitude: Some(hops.len() as u128),
                });
                // Each hop's size is read where the hops are bounded, so a
                // count past the endpoint's is refused as such first.
                asked.neighbors = Some(hops.clone());
            }
            ("num_neighbors", v) => {
                return Err(
                    mistyped(Sentence::NeighborsNotASequence, name, "not_a_sequence")
                        .with_field("given", given_text(v)),
                );
            }
            ("max_pq_iterations", v) => asked.max_pq_iterations = Some(v.clone()),
            ("context_anchor_time", v) => asked.context_anchor_time = Some(v.clone()),
            ("exclude_cols_dict", v) => asked.exclude_cols_dict = Some(v.clone()),
            ("use_prediction_time", v) => asked.use_prediction_time = Some(v.clone()),
            ("return_embeddings", v) => asked.return_embeddings = Some(v.clone()),
            ("prediction_statistic", Value::Text(text)) => {
                asked.prediction_statistic = Some(text.clone());
            }
            ("prediction_statistic", v) => {
                return Err(mistyped(Sentence::StatisticUnknown, name, "not_text")
                    .with_field("given", given_text(v)));
            }
            ("inference_config", Value::Null) => {}
            ("inference_config", v @ Value::Object(_)) => asked.inference_config = Some(v.clone()),
            ("inference_config", _) => {
                return Err(mistyped(
                    Sentence::InferenceConfigNotAnObject,
                    name,
                    "not_a_mapping",
                ));
            }
            ("rows_per_call", v) => asked.rows_per_call = rows_per_call(v)?,
            ("random_seed", Value::Null) => asked.seed = AskedSeed::Unseeded,
            ("random_seed", v) => {
                let (negative, digits) = integer(name, v)?;
                let sign = if negative { "-" } else { "" };
                asked.seed = AskedSeed::Given(format!("{sign}0x{}", hexadecimal(&digits)));
            }
            ("top_k", v) => asked.top_k = Some(v.clone()),
            // Any value, refused only where the engine refuses it, after the
            // context is labelled (T2.9).
            ("lag_timesteps", v) => asked.lag_timesteps = Some(AskedLag::of(v)),
            _ => return Err(mistyped(Sentence::OptionUnknown, name, "not_an_option")),
        }
    }
    Ok(asked)
}

/// The most candidates a ranking keeps (the language's `MAX_TOP_K`).
pub const MAX_TOP_K: u32 = 20;

/// How many candidates to keep, as the client's `_candidates_to_rank` reads
/// the call's `top_k`: a whole number (`numbers.Integral`, a boolean not
/// one) from 1 to [`MAX_TOP_K`], else `INVALID_REQUEST` naming `top_k`, why
/// (`not_a_whole_number`, `out_of_range`) and the limit.
pub fn top_k(given: &Value) -> Result<u32, CoreError> {
    let (negative, digits) = integer("top_k", given)?;
    match digits.parse::<u32>() {
        Ok(k) if !negative && (1..=MAX_TOP_K).contains(&k) => Ok(k),
        _ => Err(CoreError::new(
            Sentence::TopKOutOfRange,
            "top_k is not a number of candidates this model ranks",
        )
        .with_field("field", Detail::text("top_k"))
        .with_field("given", given_text(given))
        .with_field("reason", Detail::text("out_of_range"))
        .with_field("limit", Detail::Scalar(Scalar::UInt32(MAX_TOP_K)))),
    }
}

/// A sampling option the engine's request model holds to a boolean
/// (`RFMPredictRequest`, pydantic's lax mode, T2.11): a JSON boolean, a
/// number equal to 0 or 1, or text `0`, `off`, `f`, `false`, `n`, `no`,
/// `1`, `on`, `t`, `true`, `y` or `yes` in any case of letters; anything
/// else, none included, `INVALID_REQUEST` naming the option
/// (`not_a_boolean`).
fn lax_boolean(field: &str, value: &Value) -> Result<bool, CoreError> {
    const FALSE: [&str; 6] = ["0", "off", "f", "false", "n", "no"];
    const TRUE: [&str; 6] = ["1", "on", "t", "true", "y", "yes"];
    let refused = || mistyped(Sentence::OptionNotABoolean, field, "not_a_boolean");
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Number(token) => match token.parse::<f64>() {
            Ok(0.0) => Ok(false),
            Ok(1.0) => Ok(true),
            _ => Err(refused()),
        },
        Value::Text(text) if FALSE.iter().any(|w| w.eq_ignore_ascii_case(text)) => Ok(false),
        Value::Text(text) if TRUE.iter().any(|w| w.eq_ignore_ascii_case(text)) => Ok(true),
        _ => Err(refused()),
    }
}

/// The options the engine reads with the task (`_get_task_table`), in its
/// order: the context's anchor ([`context_anchor`]), then
/// `max_pq_iterations` ([`iterations`]).
pub fn task_table(asked: &Asked) -> Result<(AskedAnchor, Option<Iterations>), CoreError> {
    let anchor = match &asked.context_anchor_time {
        None => AskedAnchor::Absent,
        Some(given) => context_anchor(given)?,
    };
    let iterations = asked
        .max_pq_iterations
        .as_ref()
        .map(|v| iterations(v).map_err(|e| e.with_field("given", given_text(v))))
        .transpose()?;
    Ok((anchor, iterations))
}

/// The columns the caller keeps out of the request (`exclude_cols_dict`),
/// by table, in the order given, each table's names as given, as the
/// client reads them (`_columns_to_leave_out`): none or absent is none; a
/// mapping of sequences, anything else `INVALID_REQUEST` naming the option
/// (`not_a_mapping`, or `not_a_sequence` with the table).
pub fn excluded(asked: &Asked) -> Result<Vec<(String, Vec<Value>)>, CoreError> {
    const NAME: &str = "exclude_cols_dict";
    match &asked.exclude_cols_dict {
        None | Some(Value::Null) => Ok(vec![]),
        Some(Value::Object(tables)) => tables
            .iter()
            .map(|(table, columns)| match columns {
                Value::Array(columns) => Ok((table.clone(), columns.clone())),
                _ => Err(
                    mistyped(Sentence::ExcludedNotASequence, NAME, "not_a_sequence")
                        .with_field("table", Detail::text(table))
                        .with_field("given", given_text(columns)),
                ),
            })
            .collect(),
        Some(other) => Err(
            mistyped(Sentence::ExcludedNotAMapping, NAME, "not_a_mapping")
                .with_field("given", given_text(other)),
        ),
    }
}

/// The options the engine's request model holds to a boolean
/// (`RFMPredictRequest`), read as the request is built ([`lax_boolean`]):
/// whether the model is told each example's anchor time
/// (`use_prediction_time`), and whether the payload's outputs name
/// `embeddings` (`return_embeddings`, `_output_fields`), each false where
/// not named.
pub fn request_flags(asked: &Asked) -> Result<(bool, bool), CoreError> {
    let told = match &asked.use_prediction_time {
        None => false,
        Some(given) => lax_boolean("use_prediction_time", given)?,
    };
    let embeddings = match &asked.return_embeddings {
        None => false,
        Some(given) => lax_boolean("return_embeddings", given)?,
    };
    Ok((told, embeddings))
}

/// `max_pq_iterations` as given ([`iterations`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Iterations {
    /// A whole number, any past `u64` read as `u64::MAX`.
    Whole(u64),
    /// A float written with a fraction or an exponent, which the engine
    /// multiplies into a float count of candidates ([`Iterations::trials`]).
    Float(f64),
}

impl Iterations {
    /// The candidates tried for `examples` examples
    /// (`max_pq_iterations * num_train_examples`): a float's product as the
    /// reference's DuckDB sampler reads it in `reservoir(n ROWS)`, rounded
    /// to the nearest whole number, a half to the even one; `None` for a float where the count
    /// must be whole, a temporal query's (`temporal`), whose walk slices by
    /// it, as the engine's `TypeError` refuses it.
    pub fn trials(self, examples: usize, temporal: bool) -> Option<usize> {
        match self {
            Iterations::Whole(n) => Some(
                usize::try_from(n)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(examples),
            ),
            Iterations::Float(_) if temporal => None,
            // `as` saturates a float past `usize`.
            Iterations::Float(f) => Some((f * examples as f64).round_ties_even() as usize),
        }
    }
}

/// `max_pq_iterations` (T2.11): a number of at least one, as the engine
/// compares it (`_get_task_table`), a boolean as 0 or 1 as Python counts
/// one; below one is `INVALID_REQUEST` (`below_one`), as the engine
/// refuses it; text or none, which the engine cannot compare, mistyped
/// (`not_a_whole_number`), as its `TypeError` is the caller's.
fn iterations(given: &Value) -> Result<Iterations, CoreError> {
    let Integer {
        negative,
        magnitude,
    } = match given {
        Value::Bool(b) => Integer {
            negative: false,
            magnitude: Some(u128::from(*b)),
        },
        Value::Number(token) if !is_integer_token(token) => {
            let f: f64 = token.parse().map_err(|_| {
                mistyped(
                    Sentence::MaxPqIterationsNotAWholeNumber,
                    "max_pq_iterations",
                    "not_a_whole_number",
                )
            })?;
            if f.is_nan() || f < 1.0 {
                return Err(invalid(
                    Sentence::IterationsBelowOne,
                    "max_pq_iterations",
                    "below_one",
                ));
            }
            return Ok(Iterations::Float(f));
        }
        v => magnitude("max_pq_iterations", v)?,
    };
    match magnitude {
        _ if negative || magnitude == Some(0) => Err(invalid(
            Sentence::IterationsBelowOne,
            "max_pq_iterations",
            "below_one",
        )),
        Some(n) => Ok(Iterations::Whole(u64::try_from(n).unwrap_or(u64::MAX))),
        None => Ok(Iterations::Whole(u64::MAX)),
    }
}

/// What one hop of the walk keeps of the rows referring to each row it
/// visits ([`neighbors`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Hop {
    /// At most this many of each row's referring rows, latest first; none
    /// for a size between zero and one, as `ROW_NUMBER() <= 0.5` keeps none
    /// while the hop's lookups still run.
    Keep(u64),
    /// A negative bound reaches the provider; its native predicate or
    /// limit semantics decide the result (KO3-44).
    Negative(String),
    /// Caller-authored SQL expression, in the reference's original
    /// neighbor-bound position, not a parsed or normalized row count.
    SqlExpression(String),
    /// A size of zero: the walk ends before this hop (`neighbors == 0`).
    Stop,
}

/// The hops the walk takes, each with its size, as the engine resolves
/// them (`_resolve_options`): `num_neighbors` where it is given, each size
/// read as [`hop`] reads it; else the first `num_hops` of the run mode's sizes
/// (`_DEFAULT_NUM_NEIGHBORS`), two where it is not named, and a `num_hops`
/// named as none, which the engine cannot compare with its bounds,
/// `INVALID_REQUEST` (`not_a_whole_number`), as the engine's `TypeError`
/// is the caller's. Call it after [`hops`] has bounded the count.
pub fn neighbors(asked: &Asked, mode: RunMode) -> Result<Vec<Hop>, CoreError> {
    let Some(given) = &asked.neighbors else {
        if asked.num_hops_none {
            return Err(mistyped(
                Sentence::NumHopsNone,
                "num_hops",
                "not_a_whole_number",
            ));
        }
        let hops = asked
            .num_hops
            .and_then(|n| n.magnitude)
            .map_or(2, |n| usize::try_from(n).unwrap_or(usize::MAX));
        return Ok(mode
            .all_neighbors()
            .into_iter()
            .take(hops)
            .map(Hop::Keep)
            .collect());
    };
    given
        .iter()
        .enumerate()
        .map(|(at, size)| {
            hop(size).map_err(|e| e.with_field("hop", Detail::Scalar(Scalar::UInt64(at as u64))))
        })
        .collect()
}

/// One hop's size, as the reference's DuckDB sampler reads the value it
/// writes into `ROW_NUMBER() <= {size}`: a whole number, a float, a
/// boolean (`True` is one) or text spelling a number (digits, a sign, a
/// point and an exponent, white space around them, which the SQL reads as
/// that number); equal to zero
/// (as a number, never as text) ending the walk (`neighbors == 0`), below
/// one keeping no rows, else keeping its whole part, any past `u64` read
/// as `u64::MAX`. Other text is caller-authored SQL expression code and
/// reaches the provider unchanged, where the reference writes it. A
/// negative numeric bound keeps its sign and value for native provider
/// semantics (KO3-44); a list or none remains mistyped.
fn hop(given: &Value) -> Result<Hop, CoreError> {
    let refused = || mistyped(Sentence::HopNotWhole, "num_neighbors", "not_a_whole_number");
    let size = match given {
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Text(text) => {
            // A number DuckDB reads as written into the SQL: white space
            // around it, a sign, digits with a point and an exponent.
            let number = text.trim_ascii();
            let plain = number.bytes().any(|b| b.is_ascii_digit())
                && number
                    .bytes()
                    .all(|b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.' | b'e' | b'E'));
            let Some(size) = number
                .parse::<f64>()
                .ok()
                .filter(|size| plain && size.is_finite())
            else {
                return Ok(Hop::SqlExpression(text.clone()));
            };
            if size < 0.0 {
                let value = if serde_json::from_str::<serde::de::IgnoredAny>(number).is_ok() {
                    number.to_owned()
                } else {
                    size.to_string()
                };
                return Ok(Hop::Negative(value));
            }
            // `'0' == 0` is false: a zero spelled as text ends nothing.
            return Ok(if size >= 1.0 {
                Hop::Keep(size.floor() as u64)
            } else {
                Hop::Keep(0)
            });
        }
        Value::Number(token) => {
            if let Ok(Integer {
                negative,
                magnitude,
            }) = magnitude("num_neighbors", given)
            {
                return Ok(match magnitude {
                    Some(0) => Hop::Stop,
                    _ if negative => return Ok(Hop::Negative(token.clone())),
                    Some(n) => Hop::Keep(u64::try_from(n).unwrap_or(u64::MAX)),
                    None => Hop::Keep(u64::MAX),
                });
            }
            token.parse::<f64>().map_err(|_| refused())?
        }
        _ => return Err(refused()),
    };
    if !size.is_finite() {
        return Err(refused());
    }
    if size < 0.0 {
        let Value::Number(token) = given else {
            return Err(refused());
        };
        return Ok(Hop::Negative(token.clone()));
    }
    Ok(if size == 0.0 {
        Hop::Stop
    } else if size < 1.0 {
        Hop::Keep(0)
    } else {
        // `as` saturates a float past `u64` to `u64::MAX`.
        Hop::Keep(size.floor() as u64)
    })
}

/// Where the engine says it ignores `num_hops` (`_resolve_options`'s
/// warning), `num_hops` as Python writes it in the warning: `num_neighbors`
/// given, with `num_hops` named as anything but two (`None` for none).
pub fn hops_ignored(asked: &Asked) -> Option<String> {
    asked.neighbors.as_ref()?;
    if asked.num_hops_none {
        return Some("None".to_owned());
    }
    let n = asked.num_hops?;
    if !n.negative && n.magnitude == Some(2) {
        return None;
    }
    asked.num_hops_text.clone()
}

/// `context_anchor_time` as the host writes it ([`AskedAnchor`]): a `ts`
/// scalar read at its UTC instant, as the engine reads one
/// (`to_naive_utc`); a value of another form is refused as mistyped
/// (`not_a_time`), which no host that holds to the contract sends.
fn context_anchor(given: &Value) -> Result<AskedAnchor, CoreError> {
    let refused = || {
        mistyped(
            Sentence::ContextAnchorNotATime,
            "context_anchor_time",
            "not_a_time",
        )
    };
    match given {
        Value::Null => Ok(AskedAnchor::Absent),
        Value::Text(text) if text == "entity" => Ok(AskedAnchor::Entity),
        Value::Object(_) => {
            // The contract's `ts` scalar: `t`, the value `v` as decimal
            // text, `unit` and `zone`.
            if given.get("t").and_then(Value::as_text) != Some("ts") {
                return Err(refused());
            }
            let value: i64 = given
                .get("v")
                .and_then(Value::as_text)
                .and_then(|v| v.parse().ok())
                .ok_or_else(refused)?;
            match given.get("unit").and_then(Value::as_text) {
                Some("ns") => Ok(AskedAnchor::At(value)),
                Some("us") => value
                    .checked_mul(1_000)
                    .map(AskedAnchor::At)
                    .ok_or_else(refused),
                _ => Err(refused()),
            }
        }
        _ => Err(refused()),
    }
}

/// The batch size the caller chose (`rows_per_call_asked_for`): none for a
/// value of none, which chooses no size; a whole number of at least one,
/// any past `u64` read as `u64::MAX`, since the size only ever lowers the
/// endpoint's own (`rows_one_call_scores`); anything else
/// `INVALID_REQUEST` naming the field, a boolean, a float or text as
/// mistyped (`not_a_whole_number`) and a whole number below one as
/// `below_one`.
pub fn rows_per_call(given: &Value) -> Result<Option<u64>, CoreError> {
    if matches!(given, Value::Null) {
        return Ok(None);
    }
    let Integer {
        negative,
        magnitude,
    } = magnitude("rows_per_call", given)?;
    match magnitude {
        _ if negative || magnitude == Some(0) => {
            Err(invalid(Sentence::RowsPerCall, "rows_per_call", "below_one"))
        }
        Some(n) => Ok(Some(u64::try_from(n).unwrap_or(u64::MAX))),
        None => Ok(Some(u64::MAX)),
    }
}

/// The most rows one model call scores (`predict_rows_max`, the serving
/// layer's `TFM_MAX_TEST_ROWS`, which the reference's client and engine
/// hold at 10,000 from `main`'s 1fc5a686). A temporal link prediction,
/// which scores 200, is not a task this core gathers.
pub const PREDICT_ROWS: u64 = 10_000;

/// How many entities one call scores, as the caller asked or as the
/// endpoint allows (`rows_one_call_scores`): the caller's size only ever
/// lowers the endpoint's.
pub fn rows_one_call_scores(asked: Option<u64>) -> u64 {
    asked.map_or(PREDICT_ROWS, |n| n.min(PREDICT_ROWS))
}

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

    /// The neighbors kept of each row at each hop the endpoint reads, the
    /// first two being the default (`_DEFAULT_NUM_NEIGHBORS`).
    pub fn all_neighbors(self) -> [u64; 6] {
        match self {
            RunMode::Fast => [32, 32, 8, 8, 4, 4],
            RunMode::Best => [64, 64, 8, 8, 4, 4],
        }
    }
}

/// The run mode a context size asks for, as the client chooses it
/// (`_run_mode_for`, with `num_context_samples`' bound of
/// `_validate_materialize_args`): none or up to a fast run's examples, a
/// fast run; up to a best run's, a best one. Below 1 is `INVALID_REQUEST`;
/// past a best run's examples, `UNSUPPORTED_REQUEST`, as the endpoint takes
/// no more.
pub fn run_mode(asked: Option<Integer>, given: Option<&str>) -> Result<RunMode, CoreError> {
    // A refusal names the size as the caller gave it, its JSON text.
    let named = |refused: CoreError| match given {
        Some(given) => refused.with_field("given", Detail::text(given)),
        None => refused,
    };
    let Some(Integer {
        negative,
        magnitude: value,
    }) = asked
    else {
        return Ok(RunMode::Fast);
    };
    if negative || value == Some(0) {
        return Err(named(
            CoreError::new(
                Sentence::NumContextSamplesBelowOne,
                "num_context_samples is at least 1",
            )
            .with_field("field", Detail::text("num_context_samples")),
        ));
    }
    match value {
        Some(n) if n <= RunMode::Fast.context_examples() as u128 => Ok(RunMode::Fast),
        Some(n) if n <= RunMode::Best.context_examples() as u128 => Ok(RunMode::Best),
        _ => Err(named(
            CoreError::new(
                Sentence::NumContextSamplesPastMost,
                "num_context_samples is past the most context the endpoint takes",
            )
            .with_field("field", Detail::text("num_context_samples"))
            .with_field(
                "limit",
                Detail::Scalar(Scalar::UInt64(RunMode::Best.context_examples() as u64)),
            ),
        )),
    }
}

/// The most hops the endpoint reads (`hops_max`).
pub const HOPS_MAX: u128 = 6;

/// The hops the sampling options ask for, as the client bounds them
/// (`_check_shape`): `num_neighbors`' length where it is given, else
/// `num_hops`; none, or fewer than one, is `INVALID_REQUEST` (a request
/// that crosses no relationship), more than [`HOPS_MAX`]
/// `UNSUPPORTED_REQUEST`.
pub fn hops(num_hops: Option<Integer>, num_neighbors: Option<Integer>) -> Result<(), CoreError> {
    let (
        field,
        Integer {
            negative,
            magnitude: value,
        },
    ) = match (num_neighbors, num_hops) {
        (Some(n), _) => ("num_neighbors", n),
        (None, Some(n)) => ("num_hops", n),
        (None, None) => return Ok(()),
    };
    // How many hops were asked, as the host's words name them.
    let asked = value.map(|n| format!("{}{n}", if negative && n > 0 { "-" } else { "" }));
    let named = |error: CoreError| match &asked {
        Some(n) => error.with_field("hops", Detail::text(n)),
        None => error,
    };
    if negative || value == Some(0) {
        let sentence = if field == "num_hops" {
            Sentence::NoHopNumHops
        } else {
            Sentence::NoHopNumNeighbors
        };
        return Err(named(invalid(sentence, field, "no_hop"))
            .with_field("limit", Detail::Scalar(Scalar::UInt64(HOPS_MAX as u64))));
    }
    if value.is_none_or(|n| n > HOPS_MAX) {
        return Err(named(
            CoreError::new(
                if asked.is_some() {
                    Sentence::HopsPastEndpoint
                } else {
                    Sentence::HopsPastEndpointUncounted
                },
                "the sampling options ask more hops than the endpoint reads",
            )
            .with_field("field", Detail::text(field))
            .with_field("limit", Detail::Scalar(Scalar::UInt64(HOPS_MAX as u64))),
        ));
    }
    Ok(())
}

/// The statistics a regression reports of its predicted distribution
/// (`PREDICTION_STATISTICS`).
pub const STATISTICS: [&str; 2] = ["mean", "median"];

fn invalid(sentence: Sentence, field: &'static str, reason: &'static str) -> CoreError {
    CoreError::new(
        sentence,
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
        Some(name) => Err(invalid(
            Sentence::StatisticUnknown,
            "prediction_statistic",
            "not_a_statistic",
        )
        .with_field("given", given_text(&Value::Text(name.to_owned())))),
    }
}

/// A statistic chosen for a task with no distribution to summarise
/// (`_record_the_statistic`): refused, as the reference refuses it.
pub fn statistic_for(kind: Kind, statistic: Option<&String>) -> Result<(), CoreError> {
    if statistic.is_some() && kind != Kind::Regression {
        return Err(invalid(
            Sentence::StatisticNotARegression,
            "prediction_statistic",
            "not_a_regression",
        )
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
    /// A regression's; a null one is none.
    pub target_transforms: Vec<Option<String>>,
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
            target_transforms: vec![Some("quantile".to_owned())],
            output_type: "median".to_owned(),
        }
    }
}

const TRANSFORMS: [&str; 3] = ["clip", "power", "quantile"];
const OUTPUT_TYPES: [&str; 3] = ["mean", "median", "quantiles"];

/// A refusal of the inference option `key`: `INVALID_REQUEST`, as the
/// engine's validation error is the caller's (a `ValueError`).
fn option_refused(key: &str, reason: &'static str) -> CoreError {
    let sentence = match reason {
        "not_a_boolean" => Sentence::InferenceOptionNotABoolean,
        "not_a_whole_number" | "out_of_range" => Sentence::InferenceOptionNotOneToFour,
        "not_a_list" | "not_a_transform" => Sentence::InferenceOptionNotTransforms,
        "not_an_output_type" => Sentence::InferenceOptionNotAnOutputType,
        _ => Sentence::InferenceOptionUnknown,
    };
    invalid(sentence, "inference_config", reason).with_field("key", Detail::text(key))
}

/// A boolean option given as text, read as the engine's validation
/// (pydantic's lax mode) reads it: `0`, `off`, `f`, `false`, `n` and `no`
/// are false, `1`, `on`, `t`, `true`, `y` and `yes` true, in any case of
/// letters; any other text is refused, one with white space around it too.
fn boolean_text(text: &str) -> Option<bool> {
    const FALSE: [&str; 6] = ["0", "off", "f", "false", "n", "no"];
    const TRUE: [&str; 6] = ["1", "on", "t", "true", "y", "yes"];
    if FALSE.iter().any(|w| w.eq_ignore_ascii_case(text)) {
        Some(false)
    } else if TRUE.iter().any(|w| w.eq_ignore_ascii_case(text)) {
        Some(true)
    } else {
        None
    }
}

/// A whole number given as text, read as the engine's validation reads it:
/// pydantic-core 2.46.5's `str_as_int` (`src/input/shared.rs`), the one
/// path its lax `int` takes from text, ported step for step (T2.13 fix 4).
/// The text is first read as a JSON integer as it is; failing that, it is
/// cleaned ([`clean_int_text`]) and read so again. `None` for text it
/// refuses, and `Err` with the sign for a value past `i128`, which no
/// option takes.
fn whole_text(text: &str) -> Option<Result<i128, bool>> {
    json_int(text).or_else(|| json_int(&clean_int_text(text)?))
}

/// A JSON integer (jiter's `NumberInt`): an optional `-`, then `0` or a
/// digit from 1 to 9 and more digits; `Err` with the sign past `i128`.
fn json_int(text: &str) -> Option<Result<i128, bool>> {
    let (negative, digits) = match text.strip_prefix('-') {
        Some(digits) => (true, digits),
        None => (false, text),
    };
    let well_formed = !digits.is_empty()
        && digits.bytes().all(|b| b.is_ascii_digit())
        && (digits == "0" || !digits.starts_with('0'));
    if !well_formed {
        return None;
    }
    Some(match digits.parse::<i128>() {
        Ok(n) if negative => Ok(-n),
        Ok(n) => Ok(n),
        Err(_) => Err(negative),
    })
}

/// pydantic-core's `clean_int_str`: white space around the text dropped
/// (`str::trim`, as there), a `+` dropped where no `-` follows it, a `-`
/// set aside where no sign follows it, the leading zeros dropped with any
/// underscores among them ([`strip_leading_zeros`]: `0__3` is `3`), a point
/// followed only by zeros dropped with them, then the underscores, each
/// between two characters and none doubled (`strip_underscores`); `None`
/// where nothing was dropped, or the leading zeros are malformed.
fn clean_int_text(text: &str) -> Option<String> {
    let len_before = text.len();
    let mut s = text.trim();
    if let Some(suffix) = s.strip_prefix('+') {
        if suffix.starts_with('-') {
            return None;
        }
        s = suffix;
    }
    let mut negative = false;
    if let Some(suffix) = s.strip_prefix('-') {
        if suffix.starts_with('-') || suffix.starts_with('+') {
            return None;
        }
        negative = true;
        s = suffix;
    }
    s = strip_leading_zeros(s)?;
    if let Some(i) = s.find('.') {
        let decimal = &s[i + 1..];
        if !decimal.is_empty() && decimal.chars().all(|c| c == '0') {
            s = &s[..i];
        }
    }
    let sign = if negative { "-" } else { "" };
    if !(s.starts_with('_') || s.ends_with('_') || !s.contains('_') || s.contains("__")) {
        return Some(format!("{sign}{}", s.replace('_', "")));
    }
    if len_before == s.len() {
        return None;
    }
    Some(format!("{sign}{s}"))
}

/// pydantic-core's `strip_leading_zeros`: text starting with a zero loses
/// its zeros and the underscores among them, up to the first other digit
/// or `-`, or keeps the last zero before a point; text of zeros and
/// underscores alone keeps its last character; text starting with a digit
/// from 1 to 9 or `-` is kept; anything else is refused.
fn strip_leading_zeros(s: &str) -> Option<&str> {
    let mut chars = s.char_indices();
    match chars.next() {
        Some((_, '0')) => (),
        Some((_, c)) if c.is_ascii_digit() || c == '-' => return Some(s),
        _ => return None,
    }
    for (i, c) in chars {
        match c {
            '0' | '_' => (),
            '1'..='9' | '-' => return Some(&s[i..]),
            '.' => return Some(&s[i - 1..]),
            _ => return None,
        }
    }
    Some(&s[s.len() - 1..])
}

/// A boolean option: a JSON boolean, a number equal to 0 or 1, or text
/// [`boolean_text`] reads, as the engine's validation takes one.
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
        Value::Text(text) => boolean_text(text).ok_or_else(|| option_refused(key, "not_a_boolean")),
        _ => Err(option_refused(key, "not_a_boolean")),
    }
}

/// `num_estimators`: a whole number from 1 to 4, given as an integer, a
/// float with no fraction, a boolean (as 0 or 1) or text [`whole_text`]
/// reads, as the engine's validation takes it.
fn estimators(key: &str, value: &Value) -> Result<u64, CoreError> {
    let n = match value {
        Value::Bool(b) => i128::from(*b),
        Value::Number(token) => {
            whole(token).ok_or_else(|| option_refused(key, "not_a_whole_number"))?
        }
        Value::Text(text) => match whole_text(text) {
            Some(Ok(n)) => n,
            Some(Err(_)) => return Err(option_refused(key, "out_of_range")),
            None => return Err(option_refused(key, "not_a_whole_number")),
        },
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
        return Err(invalid(
            Sentence::InferenceConfigNotAnObject,
            "inference_config",
            "not_an_object",
        ));
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
                        // (`TargetTransform | None`) and writes as null.
                        Value::Null => Ok(None),
                        Value::Text(name) if TRANSFORMS.contains(&name.as_str()) => {
                            Ok(Some(name.clone()))
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
            class_shuffle: Some(self.class_shuffle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::ErrorCode;

    fn read(text: &str) -> Value {
        sdm_wire::json::read(text.as_bytes()).unwrap()
    }

    fn rule(e: &CoreError) -> Option<String> {
        e.fields.get("rule").map(|d| format!("{d:?}"))
    }

    fn int(negative: bool, magnitude: Option<u128>) -> Option<Integer> {
        Some(Integer {
            negative,
            magnitude,
        })
    }

    #[test]
    fn the_run_mode_is_the_client_s() {
        assert_eq!(run_mode(None, None).unwrap(), RunMode::Fast);
        assert_eq!(run_mode(int(false, Some(1)), None).unwrap(), RunMode::Fast);
        assert_eq!(
            run_mode(int(false, Some(1_000)), None).unwrap(),
            RunMode::Fast
        );
        assert_eq!(
            run_mode(int(false, Some(1_001)), None).unwrap(),
            RunMode::Best
        );
        assert_eq!(
            run_mode(int(false, Some(10_000)), None).unwrap(),
            RunMode::Best
        );
        for (asked, code) in [
            (int(false, Some(0)), ErrorCode::InvalidRequest),
            (int(true, Some(5)), ErrorCode::InvalidRequest),
            (int(false, Some(10_001)), ErrorCode::UnsupportedRequest),
            (int(false, None), ErrorCode::UnsupportedRequest),
        ] {
            assert_eq!(run_mode(asked, None).unwrap_err().code, code, "{asked:?}");
        }
    }

    #[test]
    fn the_hops_are_bounded_as_the_client_bounds_them() {
        let some = |n: u128| int(false, Some(n));
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
            hops(int(true, Some(1)), None).unwrap_err().code,
            ErrorCode::InvalidRequest
        );
        assert_eq!(
            hops(some(7), None).unwrap_err().code,
            ErrorCode::UnsupportedRequest
        );
        assert_eq!(
            hops(int(false, None), None).unwrap_err().code,
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
            vec![Some("power".to_owned()), Some("clip".to_owned())]
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
        // A null transform is taken, and written as null (T2.13).
        let o = options(
            Kind::Regression,
            Some(&read(r#"{"target_transforms": ["quantile", null]}"#)),
        )
        .unwrap();
        assert_eq!(o.target_transforms, vec![Some("quantile".to_owned()), None]);
    }

    /// A whole number's underscores as pydantic 2.13.5's lax `int` reads
    /// them, each answer probed on the reference (T2.13 fix 2): one between
    /// two digits of the whole part, none leading, trailing, doubled, after
    /// the sign or in the fraction; a grouped number past 4 is out of range,
    /// a misplaced underscore no number.
    #[test]
    fn a_whole_number_s_underscores_are_read_as_the_engine_reads_them() {
        let refused = |text: &str| {
            let given = Value::Object(vec![("num_estimators".into(), Value::Text(text.into()))]);
            let error = options(Kind::Regression, Some(&given)).unwrap_err();
            assert_eq!(
                error.fields["key"],
                Detail::text("num_estimators"),
                "{text:?}"
            );
            error.fields["reason"].clone()
        };
        for text in [
            "1_0",
            "1_000",
            "-0_3",
            "-1_0",
            "3_0.0",
            "1_0.0",
            "9_223_372_036_854_775_808",
            // pydantic-core's quirks, each a number out of range: a `-` after
            // the leading zeros is the number's sign (`0-3` is -3), and
            // zeros alone keep one (`0__0` is 0).
            "0-3",
            "0_-3",
            "-0__3",
            "0__0",
            "0__5",
            "-0_0",
            "0_0.00",
        ] {
            assert_eq!(refused(text), Detail::text("out_of_range"), "{text:?}");
        }
        for text in [
            "_3",
            "3_",
            "3__0",
            "1__",
            "+_3",
            "-_3",
            "3_.0",
            "3._0",
            "3.0_",
            "3.0_0",
            "3.000_0",
            "0_1.0_0",
            "_",
            "__",
            "3_\u{663}",
            "1__0",
            "0__1__0",
            "0__",
            "__0",
            "0_0_",
            "0_.0",
            "0._0",
            "_0_3",
            "0_",
            "00_",
            "3__",
            "+_0_3",
            "0 _3",
        ] {
            assert_eq!(
                refused(text),
                Detail::text("not_a_whole_number"),
                "{text:?}"
            );
        }
    }

    /// Text read as pydantic 2.13's lax mode reads it into the engine's
    /// config classes, each answer probed on the reference (T2.13).
    #[test]
    fn an_option_given_as_text_is_read_as_the_engine_reads_it() {
        let shuffle = |text: &str| {
            let given = Value::Object(vec![("column_shuffle".into(), Value::Text(text.into()))]);
            options(Kind::Regression, Some(&given)).map(|o| o.column_shuffle)
        };
        for (text, wanted) in [
            ("true", true),
            ("True", true),
            ("TRUE", true),
            ("tRuE", true),
            ("yes", true),
            ("YES", true),
            ("YeS", true),
            ("on", true),
            ("ON", true),
            ("oN", true),
            ("t", true),
            ("T", true),
            ("y", true),
            ("Y", true),
            ("1", true),
            ("false", false),
            ("False", false),
            ("no", false),
            ("off", false),
            ("oFf", false),
            ("f", false),
            ("F", false),
            ("n", false),
            ("N", false),
            ("0", false),
        ] {
            assert_eq!(shuffle(text).unwrap(), wanted, "{text:?}");
        }
        for text in [
            " true",
            "true ",
            " 1",
            "1 ",
            "1.0",
            "0.0",
            "2",
            "01",
            "00",
            "+1",
            "-0",
            "",
            " ",
            "truee",
            "true\0",
            "\u{ff54}rue",
            "\u{131}",
            "N\u{43e}",
        ] {
            let e = shuffle(text).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidRequest, "{text:?}");
            assert_eq!(rule(&e), None, "{text:?}");
        }
        let estimators = |text: &str| {
            let given = Value::Object(vec![("num_estimators".into(), Value::Text(text.into()))]);
            options(Kind::Binary, Some(&given)).map(|o| o.num_estimators)
        };
        let leading = format!("{}1", "0".repeat(5_000));
        for (text, wanted) in [
            ("1", 1),
            (" 1", 1),
            ("1 ", 1),
            ("1.0", 1),
            ("2", 2),
            ("01", 1),
            ("+1", 1),
            ("3", 3),
            ("+3", 3),
            (" 3 ", 3),
            ("3.0", 3),
            ("03", 3),
            ("4", 4),
            ("\t2\n", 2),
            ("2.00", 2),
            ("00001", 1),
            ("3.0000", 3),
            ("3.0 ", 3),
            ("\u{a0}3", 3),
            ("3\u{a0}", 3),
            ("\u{3000}3", 3),
            ("\u{2028}3", 3),
            ("\u{b}3", 3),
            ("\u{c}3", 3),
            ("\u{85}3", 3),
            ("3\r", 3),
            ("0003.000", 3),
            ("+03.0", 3),
            ("4.00", 4),
            ("3.0\n", 3),
            ("\n3.0", 3),
            (leading.as_str(), 1),
            // An underscore between two digits (T2.13 fix 2).
            ("0_3", 3),
            ("0_1.0", 1),
            ("+0_3", 3),
            (" 0_3 ", 3),
            ("0_0_3", 3),
            ("00_3", 3),
            ("0_3.00", 3),
            ("+0_0_0_3", 3),
            // Underscores among the leading zeros, doubled too, which
            // pydantic-core's `strip_leading_zeros` drops with them (T2.13
            // fix 4).
            ("0__3", 3),
            ("00__3", 3),
            ("0___3", 3),
            ("+0__3", 3),
            ("0__1.0", 1),
            ("0__0__3", 3),
            ("0__2.00", 2),
            ("0__4", 4),
            ("00_1", 1),
        ] {
            assert_eq!(estimators(text).unwrap(), wanted, "{text:?}");
        }
        let past = format!("1{}", "0".repeat(40));
        for text in [
            "0",
            "0.0",
            "00",
            "-0",
            "+0",
            "1_0",
            "1e0",
            "",
            " ",
            "\u{ff11}",
            "\u{663}",
            "-3",
            "3.5",
            "0x3",
            "3_",
            "_3",
            "0b1",
            "5",
            "1_000",
            "1.",
            ".5e1",
            "5e-1",
            "2e0",
            "inf",
            "nan",
            "4.0000000000000001",
            "3.9999999999999999",
            ".0",
            "3.",
            "3.0.0",
            "- 3",
            "+-3",
            "3 .0",
            "\u{1c}3",
            "2.0e0",
            "4.",
            "-1.0",
            "1.00000000000000000000000000000001",
            "1000000000000000000000000000000",
            "Infinity",
            "\u{663}.0",
            "true",
            past.as_str(),
        ] {
            let e = estimators(text).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidRequest, "{text:?}");
            assert_eq!(rule(&e), None, "{text:?}");
        }
    }

    #[test]
    fn the_call_s_values_are_read_by_the_types_the_client_takes() {
        let asked = values(Some(&read(
            r#"{"num_context_samples": 10, "num_hops": null, "num_neighbors": [8, 8, 4], "prediction_statistic": "mean", "inference_config": {"num_estimators": 2}, "random_seed": 1180591620717411303424}"#,
        )))
        .unwrap();
        assert_eq!(asked.num_context_samples, int(false, Some(10)));
        assert_eq!(asked.num_hops, None);
        assert_eq!(asked.num_neighbors, int(false, Some(3)));
        assert_eq!(asked.prediction_statistic.as_deref(), Some("mean"));
        assert!(asked.inference_config.is_some());
        // 2**70, as the contract's seed writes it.
        assert_eq!(asked.seed, AskedSeed::Given("0x400000000000000000".into()));
        let seed = |text: &str| values(Some(&read(text))).unwrap().seed;
        assert_eq!(seed(r#"{"random_seed": null}"#), AskedSeed::Unseeded);
        assert_eq!(
            seed(r#"{"random_seed": -42}"#),
            AskedSeed::Given("-0x2a".into())
        );
        assert_eq!(
            seed(r#"{"random_seed": 0}"#),
            AskedSeed::Given("0x0".into())
        );
        assert_eq!(seed("{}"), AskedSeed::Absent);
        assert_eq!(
            values(Some(&read(r#"{"num_context_samples": -3}"#)))
                .unwrap()
                .num_context_samples,
            int(true, Some(3))
        );
        for text in [
            r#"{"num_context_samples": 2.5}"#,
            r#"{"num_context_samples": true}"#,
            r#"{"num_hops": "2"}"#,
            r#"{"num_hops": 2.0}"#,
            r#"{"num_neighbors": "ab"}"#,
            r#"{"num_neighbors": {"a": 1}}"#,
            r#"{"prediction_statistic": 3}"#,
            r#"{"prediction_statistic": null}"#,
            r#"{"inference_config": [1]}"#,
            r#"{"random_seed": 1.5}"#,
            r#"{"random_seed": false}"#,
            r#"{"no_such": 1}"#,
            r#"[1]"#,
        ] {
            let e = values(Some(&read(text))).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidRequest, "{text}");
        }
    }

    /// `lag_timesteps` read as the engine meets each value (T2.9): refused
    /// for any query where it is negative or not ordered against zero, for
    /// a temporal one where it is not whole or past `u64`, and otherwise
    /// the steps a temporal query adds, none for a static one.
    #[test]
    fn only_the_legacy_nan_lag_is_admitted_from_nonfinite_option_tokens() {
        let given = sdm_wire::json::read_options(br#"{"lag_timesteps":NaN}"#).unwrap();
        assert_eq!(
            values(Some(&given)).unwrap().lag_timesteps,
            Some(AskedLag::Steps(0))
        );
        for (text, field) in [
            (br#"{"num_hops":NaN}"#.as_slice(), "num_hops"),
            (br#"{"lag_timesteps":Infinity}"#.as_slice(), "lag_timesteps"),
            (
                br#"{"inference_config":{"num_estimators":NaN}}"#.as_slice(),
                "inference_config",
            ),
        ] {
            let given = sdm_wire::json::read_options(text).unwrap();
            let error = values(Some(&given)).unwrap_err();
            assert_eq!(error.code, ErrorCode::InvalidRequest);
            assert_eq!(error.fields["field"], Detail::text(field));
        }
        let text = asked(r#"{"lag_timesteps":"NaN"}"#);
        assert!(text.lag_timesteps.unwrap().refusal(true).is_err());
    }

    #[test]
    fn lag_timesteps_is_refused_where_the_engine_refuses_it() {
        let lag = |text: &str| {
            values(Some(&read(&format!(r#"{{"lag_timesteps": {text}}}"#))))
                .unwrap()
                .lag_timesteps
                .unwrap()
        };
        let taken = |text: &str, temporal: bool| lag(text).refusal(temporal);
        for (text, steps) in [
            ("0", 0),
            ("-0", 0),
            ("3", 3),
            ("0.0", 0),
            ("-0.0", 0),
            ("false", 0),
            ("18446744073709551615", u64::MAX),
        ] {
            assert_eq!(taken(text, true).unwrap(), steps, "{text}");
            assert_eq!(taken(text, false).unwrap(), 0, "{text}");
        }
        for (text, reason) in [
            ("-1", "negative"),
            ("-1.5", "negative"),
            ("-1e400", "negative"),
            ("-99999999999999999999999", "negative"),
            (r#""2""#, "not_a_number"),
            ("null", "not_a_number"),
            ("[1]", "not_a_number"),
            (r#"{"a": 1}"#, "not_a_number"),
        ] {
            for temporal in [true, false] {
                let e = taken(text, temporal).unwrap_err();
                assert_eq!(e.code, ErrorCode::InvalidRequest, "{text}");
                assert_eq!(e.fields["field"], Detail::text("lag_timesteps"));
                assert_eq!(e.fields["reason"], Detail::text(reason), "{text}");
            }
        }
        for text in ["1.5", "2.0", "1e400", "true"] {
            let e = taken(text, true).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidRequest, "{text}");
            assert_eq!(e.fields["reason"], Detail::text("not_a_whole_number"));
            assert_eq!(taken(text, false).unwrap(), 0, "{text}");
        }
        let past = taken("18446744073709551616", true).unwrap_err();
        assert_eq!(past.code, ErrorCode::ValueOutOfRange);
        assert_eq!(past.fields["field"], Detail::text("lag_timesteps"));
        assert_eq!(taken("18446744073709551616", false).unwrap(), 0);
        assert_eq!(values(Some(&read("{}"))).unwrap().lag_timesteps, None);
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

    fn asked(text: &str) -> Asked {
        values(Some(&read(text))).unwrap()
    }

    fn reason(e: &CoreError) -> Detail {
        e.fields["reason"].clone()
    }

    #[test]
    fn the_hops_are_the_run_mode_s_first_num_hops_two_by_default() {
        let fast = |text: &str| neighbors(&asked(text), RunMode::Fast).unwrap();
        assert_eq!(fast("{}"), vec![Hop::Keep(32), Hop::Keep(32)]);
        assert_eq!(
            fast(r#"{"num_hops": 4}"#),
            vec![Hop::Keep(32), Hop::Keep(32), Hop::Keep(8), Hop::Keep(8)]
        );
        assert_eq!(
            neighbors(&asked(r#"{"num_hops": 3}"#), RunMode::Best).unwrap(),
            vec![Hop::Keep(64), Hop::Keep(64), Hop::Keep(8)]
        );
        // num_hops named as none, with no num_neighbors, is the engine's
        // TypeError: the caller's.
        let e = neighbors(&asked(r#"{"num_hops": null}"#), RunMode::Fast).unwrap_err();
        assert_eq!(e.code, ErrorCode::InvalidRequest);
        assert_eq!(e.fields["field"], Detail::text("num_hops"));
    }

    #[test]
    fn num_neighbors_gives_each_hop_s_size_a_zero_ending_the_walk() {
        let given = asked(r#"{"num_neighbors": [5, 1, 0, 3], "num_hops": 2}"#);
        assert_eq!(
            neighbors(&given, RunMode::Fast).unwrap(),
            vec![Hop::Keep(5), Hop::Keep(1), Hop::Stop, Hop::Keep(3)]
        );
        // KO3-44 retains the signed bound for the provider's semantics.
        for (sizes, negative) in [
            ("[4, -1]", "-1"),
            ("[4, -0.5]", "-0.5"),
            (r#"[4, "-2"]"#, "-2"),
        ] {
            assert_eq!(
                neighbors(
                    &asked(&format!(r#"{{"num_neighbors": {sizes}}}"#)),
                    RunMode::Fast
                )
                .unwrap(),
                vec![Hop::Keep(4), Hop::Negative(negative.into())],
            );
        }
        assert_eq!(hops_ignored(&given), None);
        let wide = asked(r#"{"num_neighbors": [100000000000000000000000000000]}"#);
        assert_eq!(
            neighbors(&wide, RunMode::Fast).unwrap(),
            vec![Hop::Keep(u64::MAX)]
        );
        let odd = asked(r#"{"num_neighbors": [2.0, 2.5, true, false, 0.5, -0.0, 0.0, 1e30]}"#);
        assert_eq!(
            neighbors(&odd, RunMode::Fast).unwrap(),
            vec![
                Hop::Keep(2),
                Hop::Keep(2),
                Hop::Keep(1),
                Hop::Stop,
                Hop::Keep(0),
                Hop::Stop,
                Hop::Stop,
                Hop::Keep(u64::MAX)
            ]
        );
        let spelled = asked(r#"{"num_neighbors": ["3", "-0", "0", "2.5"]}"#);
        assert_eq!(
            neighbors(&spelled, RunMode::Fast).unwrap(),
            vec![Hop::Keep(3), Hop::Keep(0), Hop::Keep(0), Hop::Keep(2)]
        );
        let spaced = asked(r#"{"num_neighbors": [" 3 ", "1e1", ".5"]}"#);
        assert_eq!(
            neighbors(&spaced, RunMode::Fast).unwrap(),
            vec![Hop::Keep(3), Hop::Keep(10), Hop::Keep(0)]
        );
        for size in ["null", "[1]"] {
            let e = neighbors(
                &asked(&format!(r#"{{"num_neighbors": [{size}]}}"#)),
                RunMode::Fast,
            )
            .unwrap_err();
            assert_eq!(e.fields["field"], Detail::text("num_neighbors"), "{size}");
            assert_eq!(reason(&e), Detail::text("not_a_whole_number"), "{size}");
        }
    }

    #[test]
    fn owner_admitted_neighbor_values_keep_their_distinct_meanings() {
        let values = asked(
            r#"{"num_neighbors": [-1, -0.5, "-2", "3 OR TRUE", "nonsense", 18446744073709551615]}"#,
        );
        assert_eq!(
            neighbors(&values, RunMode::Fast).unwrap(),
            vec![
                Hop::Negative("-1".into()),
                Hop::Negative("-0.5".into()),
                Hop::Negative("-2".into()),
                Hop::SqlExpression("3 OR TRUE".into()),
                Hop::SqlExpression("nonsense".into()),
                Hop::Keep(u64::MAX),
            ]
        );
    }

    #[test]
    fn num_hops_is_said_to_be_ignored_beside_num_neighbors_unless_two() {
        let ignored = |text: &str| hops_ignored(&asked(text));
        assert_eq!(
            ignored(r#"{"num_neighbors": [4], "num_hops": 3}"#).as_deref(),
            Some("3")
        );
        assert_eq!(
            ignored(r#"{"num_neighbors": [4], "num_hops": -0}"#).as_deref(),
            Some("0")
        );
        assert_eq!(
            ignored(r#"{"num_neighbors": [4], "num_hops": null}"#).as_deref(),
            Some("None")
        );
        assert_eq!(ignored(r#"{"num_neighbors": [4], "num_hops": 2}"#), None);
        assert_eq!(ignored(r#"{"num_neighbors": [4]}"#), None);
        assert_eq!(ignored(r#"{"num_hops": 3}"#), None);
    }

    #[test]
    fn max_pq_iterations_is_a_whole_number_of_at_least_one() {
        let pq = |text: &str| task_table(&asked(text)).unwrap().1.unwrap();
        assert_eq!(pq(r#"{"max_pq_iterations": 3}"#), Iterations::Whole(3));
        assert_eq!(pq(r#"{"max_pq_iterations": true}"#), Iterations::Whole(1));
        assert_eq!(
            pq(r#"{"max_pq_iterations": 100000000000000000000000}"#),
            Iterations::Whole(u64::MAX)
        );
        let float = pq(r#"{"max_pq_iterations": 1.5}"#);
        assert_eq!(float, Iterations::Float(1.5));
        assert_eq!(float.trials(1_000, false), Some(1_500));
        assert_eq!(float.trials(1_000, true), None);
        assert_eq!(Iterations::Float(1.0006).trials(1_000, false), Some(1_001));
        assert_eq!(Iterations::Float(1.0005).trials(1_000, false), Some(1_000));
        assert_eq!(Iterations::Whole(3).trials(1_000, true), Some(3_000));
        for (given, why) in [
            ("0", "below_one"),
            ("-2", "below_one"),
            ("false", "below_one"),
            ("0.5", "below_one"),
            (r#""3""#, "not_a_whole_number"),
            ("null", "not_a_whole_number"),
        ] {
            let e =
                task_table(&asked(&format!(r#"{{"max_pq_iterations": {given}}}"#))).unwrap_err();
            assert_eq!(e.code, ErrorCode::InvalidRequest, "{given}");
            assert_eq!(e.fields["field"], Detail::text("max_pq_iterations"));
            assert_eq!(reason(&e), Detail::text(why), "{given}");
        }
    }

    #[test]
    fn use_prediction_time_and_return_embeddings_are_lax_booleans() {
        assert!(
            request_flags(&asked(r#"{"use_prediction_time": true}"#))
                .unwrap()
                .0
        );
        assert_eq!(request_flags(&asked("{}")).unwrap(), (false, false));
        for (given, read_as) in [
            ("1", true),
            ("0.0", false),
            (r#""YES""#, true),
            (r#""off""#, false),
            (r#""t""#, true),
        ] {
            let text = format!(r#"{{"use_prediction_time": {given}}}"#);
            assert_eq!(request_flags(&asked(&text)).unwrap().0, read_as, "{given}");
        }
        for name in ["use_prediction_time", "return_embeddings"] {
            for given in ["null", "2", r#""x""#, r#"" yes""#, "[1]"] {
                let e = request_flags(&asked(&format!(r#"{{"{name}": {given}}}"#))).unwrap_err();
                assert_eq!(e.fields["field"], Detail::text(name), "{given}");
                assert_eq!(reason(&e), Detail::text("not_a_boolean"), "{given}");
            }
        }
        assert_eq!(
            request_flags(&asked(r#"{"return_embeddings": "true"}"#)).unwrap(),
            (false, true)
        );
    }

    #[test]
    fn exclude_cols_dict_maps_tables_to_lists_kept_in_order() {
        let given = asked(r#"{"exclude_cols_dict": {"b": ["y", "x"], "a": []}}"#);
        let given = excluded(&given).unwrap();
        assert_eq!(given.len(), 2);
        assert_eq!(given[0].0, "b");
        assert_eq!(given[0].1, vec![Value::text("y"), Value::text("x")]);
        assert!(
            excluded(&asked(r#"{"exclude_cols_dict": null}"#))
                .unwrap()
                .is_empty()
        );
        for (given, why) in [
            (r#"["a"]"#, "not_a_mapping"),
            (r#""a""#, "not_a_mapping"),
            (r#"{"a": "x"}"#, "not_a_sequence"),
            (r#"{"a": null}"#, "not_a_sequence"),
        ] {
            let e = excluded(&asked(&format!(r#"{{"exclude_cols_dict": {given}}}"#))).unwrap_err();
            assert_eq!(
                e.fields["field"],
                Detail::text("exclude_cols_dict"),
                "{given}"
            );
            assert_eq!(reason(&e), Detail::text(why), "{given}");
        }
    }

    #[test]
    fn context_anchor_time_is_a_ts_scalar_entity_or_none() {
        assert_eq!(task_table(&asked("{}")).unwrap().0, AskedAnchor::Absent);
        assert_eq!(
            task_table(&asked(r#"{"context_anchor_time": null}"#))
                .unwrap()
                .0,
            AskedAnchor::Absent
        );
        assert_eq!(
            task_table(&asked(r#"{"context_anchor_time": "entity"}"#))
                .unwrap()
                .0,
            AskedAnchor::Entity
        );
        assert_eq!(
            task_table(&asked(r#"{"context_anchor_time": {"t": "ts", "v": "1709251200000000000", "unit": "ns", "zone": null}}"#)).unwrap().0,
            AskedAnchor::At(1_709_251_200_000_000_000)
        );
        assert_eq!(
            task_table(&asked(
                r#"{"context_anchor_time": {"t": "ts", "v": "5", "unit": "us", "zone": "UTC"}}"#
            ))
            .unwrap()
            .0,
            AskedAnchor::At(5_000)
        );
        for given in [
            r#""2024-01-01""#,
            r#""unreadable""#,
            "5",
            r#"{"t": "i64", "v": "5"}"#,
            r#"{"t": "ts", "v": "9223372036854775807", "unit": "us", "zone": null}"#,
        ] {
            let e =
                task_table(&asked(&format!(r#"{{"context_anchor_time": {given}}}"#))).unwrap_err();
            assert_eq!(reason(&e), Detail::text("not_a_time"), "{given}");
        }
    }
}
