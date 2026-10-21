// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A ranking asked as the questions the model serves, and its answers put
//! back together (T2.8): the client's `decompose._per_candidate` and
//! `_as_literal`, and `SDMClient._ranked` and `_how_sure`.
//!
//! The endpoint has no ranking task. A ranking over candidates is asked as
//! one binary question per distinct candidate, whether the entity reaches it
//! in the window, written from the query's final aggregation
//! ([`Ranked::question`]), each candidate written into its question by the
//! type of the column ranked ([`literal`]). The answers are put back
//! together as one row per entity asked about, each candidate scored by the
//! probability of its question's positive class, best first, at most `top_k`
//! ([`rank`]), laid out as an Arrow table of candidates (each the JSON text
//! of its value, and its position among those given) and scores
//! ([`table`]); a host gives each candidate back as the value it was given
//! as, found by its position.
//!
//! Each refusal is a code and fields naming the candidate by its position
//! among those given, never its value; the host words it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::builder::{Float64Builder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use sdm_contract::ops::{GivenCandidate, TypedKind};
use sdm_contract::{CoreError, Detail, ErrorCode, ErrorKind, Scalar};

use crate::predict::{Entity, Part, id_columns};

/// A window of the final aggregation, as the query writes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Window {
    /// `None` for an open start, which the client cannot write a question
    /// of.
    pub start: Option<i64>,
    pub end: i64,
    /// The unit's name as the query writes it (`days`).
    pub unit: String,
}

/// What every candidate's question is written from: the final
/// aggregation's target, as the query writes it, filter and all, and the
/// table and column it names; its window; and the entity clause, filter and
/// all (`_for_each`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ranked {
    pub target: String,
    /// The table and the column ranked, each unquoted.
    pub table: String,
    pub column: String,
    pub window: Option<Window>,
    pub entity: String,
}

impl Ranked {
    /// Whether the rows ranked are chosen by a filter (`' WHERE ' in
    /// ranked`), which the client cannot combine with each candidate's own.
    pub fn filtered(&self) -> bool {
        self.target.contains(" WHERE ")
    }

    /// The question asked of one candidate, written as `literal`: whether
    /// the entity reaches a row of the ranked table holding it in the
    /// window, `PREDICT COUNT(<table>.* WHERE <target>=<literal>, <start>,
    /// <end>, <unit>) > 0 FOR EACH ...`; `quote` spells the table as the
    /// query language names it. None where the window has no start, which
    /// the client cannot write (a `TypeError` there).
    pub fn question(&self, literal: &str, quote: fn(&str) -> String) -> Option<String> {
        let Window {
            start: Some(start),
            end,
            unit,
        } = self.window.as_ref()?
        else {
            return None;
        };
        Some(format!(
            "PREDICT COUNT({}.* WHERE {}={literal}, {start}, {end}, {unit}) > 0 {}",
            quote(&self.table),
            self.target,
            self.entity,
        ))
    }
}

/// One candidate as the host gave it.
#[derive(Debug, Clone, PartialEq)]
pub enum Candidate {
    /// A null, or a float NaN: no row is named.
    Missing,
    Bool(bool),
    /// An integer of any width, by its digits ([`Number`]).
    Int(Number),
    /// A float of any Arrow float type, as the double it is.
    Float(f64),
    /// An exact decimal of any precision: its unscaled digits
    /// ([`Number`]) and its scale, the digits after the point.
    Decimal {
        unscaled: Number,
        scale: i32,
    },
    Text(String),
    /// A value of a type no scalar holds, by the text the reference writes
    /// into its question, which the core holds equal to no other candidate
    /// (`GivenCandidate::Spelled`).
    Spelled(String),
    /// A value of a type no scalar holds, by its text and its kind's name
    /// (`GivenCandidate::Typed`): written as [`Candidate::Spelled`] is,
    /// and in one group with another the host names the same `first` of
    /// (the host's language's equality), or, where it names none, with
    /// another of one kind and one text.
    Typed {
        text: String,
        kind: TypedKind,
        first: Option<u32>,
    },
    /// A value the host holds equal to one given before it, left out
    /// (`GivenCandidate::Repeat`).
    Repeat,
    /// A value of another type, by the type's name.
    Other(String),
    /// A value the host could not look up by value (`GivenCandidate::
    /// Unreadable`).
    Unreadable,
}

/// A whole number of any width: its sign, and its digits with no leading
/// zero (`0` for zero), which keep a negative zero's sign as the reference
/// writes it (`Decimal('-0.00')`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Number {
    pub negative: bool,
    pub digits: String,
}

impl Number {
    /// The number `text` writes: `-` before a negative one, then its
    /// decimal digits; none for anything else.
    pub fn parse(text: &str) -> Option<Number> {
        let (negative, digits) = match text.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, text),
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let digits = digits.trim_start_matches('0');
        Some(Number {
            negative,
            digits: if digits.is_empty() { "0" } else { digits }.to_owned(),
        })
    }

    fn sign(&self) -> &'static str {
        if self.negative { "-" } else { "" }
    }

    /// Its text as an integer's, a zero with no sign.
    fn whole_text(&self) -> String {
        if self.digits == "0" {
            "0".to_owned()
        } else {
            format!("{}{}", self.sign(), self.digits)
        }
    }
}

/// The decimal `text` writes in fixed point (`-12.50`), as its unscaled
/// digits and scale; none for anything else.
fn fixed_point(text: &str) -> Option<(Number, i32)> {
    let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
    if !fraction.bytes().all(|b| b.is_ascii_digit()) || whole.trim_start_matches('-').is_empty() {
        return None;
    }
    let scale = i32::try_from(fraction.len()).ok()?;
    Some((Number::parse(&format!("{whole}{fraction}"))?, scale))
}

impl Candidate {
    /// A key equal for two candidates Python's `dict.fromkeys` holds as
    /// one: a number by its exact value, whatever its type (`1`, `1.0`,
    /// `True` and `Decimal('1.00')` are one), text by its text, a missing
    /// value one key; none for a NaN or a value of another type, which no
    /// other equals.
    fn key(&self) -> Option<String> {
        let trimmed = |text: String| -> String {
            let text = if text.contains('.') {
                text.trim_end_matches('0').trim_end_matches('.').to_owned()
            } else {
                text
            };
            if text == "-0" { "0".to_owned() } else { text }
        };
        match self {
            Candidate::Missing => Some("z".to_owned()),
            Candidate::Bool(b) => Some(format!("n{}", u8::from(*b))),
            Candidate::Int(n) => Some(format!("n{}", n.whole_text())),
            Candidate::Float(f) if !f.is_finite() => {
                (!f.is_nan()).then(|| format!("i{}", f.is_sign_negative()))
            }
            Candidate::Float(f) => Some(format!("n{}", trimmed(format!("{f:.1100}")))),
            Candidate::Decimal { unscaled, scale } => {
                Some(format!("n{}", trimmed(decimal_text(unscaled, *scale))))
            }
            Candidate::Text(text) => Some(format!("s{text}")),
            Candidate::Typed { text, kind, first } => Some(match first {
                Some(first) => format!("g{first}"),
                None => format!("t{kind:?}\u{0}{text}"),
            }),
            Candidate::Spelled(_)
            | Candidate::Repeat
            | Candidate::Other(_)
            | Candidate::Unreadable => None,
        }
    }
}

/// A candidate as the host gave it ([`GivenCandidate`]), from its contract
/// scalar or its digits: a null or a NaN missing; a scalar of a type a host
/// does not write a candidate as, or digits that write no number, another
/// type, by its name.
pub fn read(given: &GivenCandidate) -> Candidate {
    let scalar = match given {
        GivenCandidate::Value(scalar) => scalar,
        GivenCandidate::Integer(text) => {
            return Number::parse(text)
                .map_or_else(|| Candidate::Other("integer".to_owned()), Candidate::Int);
        }
        GivenCandidate::Decimal(text) => {
            return fixed_point(text).map_or_else(
                || Candidate::Other("decimal".to_owned()),
                |(unscaled, scale)| Candidate::Decimal { unscaled, scale },
            );
        }
        GivenCandidate::Spelled(text) => return Candidate::Spelled(text.clone()),
        GivenCandidate::Typed(typed) => {
            return Candidate::Typed {
                text: typed.text.clone(),
                kind: typed.kind,
                first: typed.first,
            };
        }
        GivenCandidate::Repeat => return Candidate::Repeat,
        GivenCandidate::Unreadable => return Candidate::Unreadable,
    };
    let int = |digits: String| {
        Number::parse(&digits)
            .map_or_else(|| Candidate::Other("integer".to_owned()), Candidate::Int)
    };
    let decimal = |digits: String, scale: i8| {
        Number::parse(&digits).map_or_else(
            || Candidate::Other("decimal".to_owned()),
            |unscaled| Candidate::Decimal {
                unscaled,
                scale: scale.into(),
            },
        )
    };
    match scalar {
        Scalar::Null => Candidate::Missing,
        Scalar::Bool(b) => Candidate::Bool(*b),
        Scalar::Int8(v) => int(v.to_string()),
        Scalar::Int16(v) => int(v.to_string()),
        Scalar::Int32(v) => int(v.to_string()),
        Scalar::Int64(v) => int(v.to_string()),
        Scalar::UInt8(v) => int(v.to_string()),
        Scalar::UInt16(v) => int(v.to_string()),
        Scalar::UInt32(v) => int(v.to_string()),
        Scalar::UInt64(v) => int(v.to_string()),
        Scalar::Float32(v) => float((*v).into()),
        Scalar::Float64(v) => float(*v),
        Scalar::Decimal128 { scale, value, .. } => decimal(value.to_string(), *scale),
        Scalar::Decimal256 { scale, value, .. } => decimal(value.to_string(), *scale),
        Scalar::Utf8(text) => Candidate::Text(text.clone()),
        Scalar::Binary(_) => Candidate::Other("binary".to_owned()),
        Scalar::Date32(_) => Candidate::Other("date32".to_owned()),
        Scalar::Timestamp { .. } => Candidate::Other("ts".to_owned()),
        Scalar::Duration { .. } => Candidate::Other("duration".to_owned()),
        Scalar::Time64 { .. } => Candidate::Other("time64".to_owned()),
    }
}

/// A float, a NaN missing (`pd.isna`).
fn float(value: f64) -> Candidate {
    if value.is_nan() {
        Candidate::Missing
    } else {
        Candidate::Float(value)
    }
}

/// A candidate as the result's `rankings` names it (T2.8): the JSON text of
/// its value, an integer's digits, a float as Python's `repr` writes it
/// (`float_repr`), a decimal's digits at its scale, text, or a value
/// spelled as text, as a JSON string; none for a value no question is asked
/// of (missing, a boolean, another type).
pub fn json_text(candidate: &Candidate, float_repr: fn(f64) -> String) -> Option<String> {
    match candidate {
        Candidate::Int(n) => Some(n.whole_text()),
        Candidate::Float(f) if f.is_finite() => Some(float_repr(*f)),
        Candidate::Decimal { unscaled, scale } => Some(decimal_text(unscaled, *scale)),
        Candidate::Text(text) | Candidate::Spelled(text) | Candidate::Typed { text, .. } => Some(
            String::from_utf8_lossy(&sdm_wire::json::Value::text(text).to_bytes()).into_owned(),
        ),
        _ => None,
    }
}

/// The distinct candidates, each with its position among those given (its
/// first), in the order first given (`dict.fromkeys`); one the host holds a
/// repeat left out.
pub fn distinct(candidates: &[Candidate]) -> Vec<(u32, &Candidate)> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut kept: Vec<(u32, &Candidate)> = Vec::new();
    for (position, candidate) in (0u32..).zip(candidates) {
        // A value with no key equals none but itself, and is kept.
        let new = !matches!(candidate, Candidate::Repeat)
            && candidate.key().is_none_or(|key| seen.insert(key));
        if new {
            kept.push((position, candidate));
        }
    }
    kept
}

/// A candidate refused, by rule, naming it by its position.
fn refused(rule: &'static str, position: u32) -> CoreError {
    CoreError::new(
        ErrorKind::InvalidRequest,
        ErrorCode::InvalidRequest,
        "a candidate cannot be written into the question asked of it",
    )
    .with_field("field", Detail::text("candidates"))
    .with_field("rule", Detail::text(rule))
    .with_field("candidate", Detail::Scalar(Scalar::UInt32(position)))
}

/// The refusal of a candidate the host could not look up by value
/// (`candidate_unreadable`, reason `not_a_key`), which names no row: the
/// reference's `dict.fromkeys` fails on it before anything is written.
pub fn unreadable(position: u32) -> CoreError {
    refused("candidate_unreadable", position).with_field("reason", Detail::text("not_a_key"))
}

/// Whether Python's `str.isspace` holds of `c`: Unicode white space, and
/// the four separators U+001C to U+001F.
fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// A decimal's text as `format(value, 'f')` writes it: its digits, at its
/// scale, a negative zero's sign kept.
fn decimal_text(unscaled: &Number, scale: i32) -> String {
    let sign = unscaled.sign();
    let digits = &unscaled.digits;
    if scale <= 0 {
        if digits == "0" {
            return format!("{sign}0");
        }
        let zeros = "0".repeat(usize::try_from(scale.unsigned_abs()).unwrap_or(0));
        return format!("{sign}{digits}{zeros}");
    }
    let scale = usize::try_from(scale.unsigned_abs()).unwrap_or(0);
    let padded = format!("{digits:0>width$}", width = scale + 1);
    let (whole, fraction) = padded.split_at(padded.len() - scale);
    format!("{sign}{whole}.{fraction}")
}

/// The whole number a decimal is, as an integer's text; none for one with a
/// fraction.
fn whole_of(unscaled: &Number, scale: i32) -> Option<String> {
    let text = decimal_text(unscaled, scale);
    let (whole, fraction) = text.split_once('.').unwrap_or((&text, ""));
    fraction
        .bytes()
        .all(|b| b == b'0')
        .then(|| Number::parse(whole).map(|n| n.whole_text()))
        .flatten()
}

/// Whether `float` is exactly the decimal `text` writes
/// (`Decimal(float(value)) == value`), from the double's exact expansion.
fn exactly(float: f64, text: &str) -> bool {
    let trimmed = |text: &str| -> String {
        let text = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.')
        } else {
            text
        };
        match text {
            "-0" => "0".to_owned(),
            other => other.to_owned(),
        }
    };
    trimmed(&format!("{float:.1100}")) == trimmed(text)
}

/// A float's text as `format(Decimal(str(value)), 'f')` writes it, from its
/// shortest repr as Python writes it (`repr`): positional, with no
/// exponent.
fn positional(repr: &str) -> String {
    let Some((mantissa, exponent)) = repr.split_once('e') else {
        return repr.to_owned();
    };
    let Ok(exponent) = exponent.parse::<i64>() else {
        return repr.to_owned();
    };
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{whole}{fraction}");
    // Where the point falls, counted in digits from the left.
    let point = i64::try_from(whole.len()).unwrap_or(0) + exponent;
    let len = i64::try_from(digits.len()).unwrap_or(0);
    if point <= 0 {
        let zeros = "0".repeat(usize::try_from(-point).unwrap_or(0));
        format!("{sign}0.{zeros}{digits}")
    } else if point >= len {
        let zeros = "0".repeat(usize::try_from(point - len).unwrap_or(0));
        format!("{sign}{digits}{zeros}")
    } else {
        let at = usize::try_from(point).unwrap_or(0);
        format!("{sign}{}.{}", &digits[..at], &digits[at..])
    }
}

/// A whole double's digits, as `str(int(value))` writes them.
fn whole_text(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        format!("{value:.0}")
    }
}

/// The candidate at `position` as the query language writes it, by the
/// dtype of the column ranked (`dtype`, its name on the wire, none where
/// the schema does not hold it), as the client's `_as_literal` writes it:
/// an integer as its digits; a whole float as its integer's; a decimal as
/// the column's type takes it (an integer column's integer, a float
/// column's double, else its digits as text), refused where that is
/// another value; text, or a value spelled as text, in double quotes; a
/// number on a text column as its digits in double quotes. Refused (`INVALID_REQUEST`, field
/// `candidates`, naming the rule and the candidate): a missing value
/// (`candidate_missing`), a boolean (`candidate_bool`), a decimal its
/// column's type does not hold exactly (`candidate_not_exact`), a float not
/// finite or not whole (`candidate_not_whole`), and text holding a quote, a
/// comma, a bracket or a backslash, or only white space
/// (`candidate_unwritable`); a value of any other type, which this core does
/// not write yet (`UNSUPPORTED_REQUEST`, rule `candidate_type`).
/// `float_repr` is Python's `repr` of a double.
pub fn literal(
    candidate: &Candidate,
    position: u32,
    dtype: Option<&str>,
    float_repr: fn(f64) -> String,
) -> Result<String, CoreError> {
    let mut value = candidate.clone();
    match candidate {
        Candidate::Missing | Candidate::Repeat => {
            return Err(refused("candidate_missing", position));
        }
        Candidate::Bool(_) => return Err(refused("candidate_bool", position)),
        Candidate::Unreadable => return Err(unreadable(position)),
        Candidate::Other(name) => {
            return Err(CoreError::new(
                ErrorKind::Unsupported,
                ErrorCode::UnsupportedRequest,
                "a candidate is of a type this core does not write into a question yet",
            )
            .with_field("field", Detail::text("candidates"))
            .with_field("rule", Detail::text("candidate_type"))
            .with_field("candidate", Detail::Scalar(Scalar::UInt32(position)))
            .with_field("type", Detail::text(name)));
        }
        Candidate::Decimal { unscaled, scale } => {
            let dtype = dtype.map_or_else(|| decimal_dtype(unscaled, *scale), str::to_owned);
            let text = decimal_text(unscaled, *scale);
            let not_exact = || {
                refused("candidate_not_exact", position).with_field("dtype", Detail::text(&dtype))
            };
            value = match dtype.as_str() {
                "int64" => Candidate::Int(
                    whole_of(unscaled, *scale)
                        .and_then(|whole| Number::parse(&whole))
                        .ok_or_else(not_exact)?,
                ),
                "float64" => match text.parse::<f64>() {
                    Ok(float) if exactly(float, &text) => Candidate::Float(float),
                    _ => return Err(not_exact()),
                },
                _ => Candidate::Text(text),
            };
        }
        _ => {}
    }
    if dtype == Some("string") {
        match &value {
            Candidate::Float(f) if !f.is_finite() => {
                return Err(refused("candidate_not_whole", position));
            }
            Candidate::Float(f) => value = Candidate::Text(positional(&float_repr(*f))),
            Candidate::Int(n) => value = Candidate::Text(n.whole_text()),
            _ => {}
        }
    }
    match value {
        Candidate::Int(n) => Ok(n.whole_text()),
        Candidate::Float(f) if !f.is_finite() || f.fract() != 0.0 => {
            Err(refused("candidate_not_whole", position))
        }
        Candidate::Float(f) => Ok(whole_text(f)),
        Candidate::Text(text) | Candidate::Spelled(text) | Candidate::Typed { text, .. } => {
            if text.chars().any(|c| "\"'\\,()".contains(c)) || text.chars().all(is_space) {
                return Err(refused("candidate_unwritable", position));
            }
            Ok(format!("\"{text}\""))
        }
        _ => Err(refused("candidate_missing", position)),
    }
}

/// The dtype a decimal's column travels as where the schema does not say
/// (`_decimal_dtype` of the one value): `float64` for a fraction, `int64`
/// for a whole number int64 holds, else `string`.
fn decimal_dtype(unscaled: &Number, scale: i32) -> String {
    match whole_of(unscaled, scale) {
        None => "float64",
        Some(whole) if whole.parse::<i64>().is_ok() => "int64",
        Some(_) => "string",
    }
    .to_owned()
}

/// One row of a ranking: the entity asked about on it, and its candidates,
/// best first, at most `top_k`, each by its position among those given and
/// as the result names it ([`json_text`]), with its score.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub row_index: i64,
    pub id: Entity,
    pub ranked: Vec<(u32, String, f64)>,
}

/// How sure the endpoint was that the row's entity reaches the candidate:
/// the probability of the positive class, spelled `true`, `True`, `TRUE` or
/// `1` (`_how_sure`); none where the answer gives no such class.
fn how_sure(probabilities: Option<&Vec<(String, f64)>>) -> Option<f64> {
    let given = probabilities?;
    ["true", "True", "TRUE", "1"]
        .iter()
        .find_map(|spelling| given.iter().find(|(class, _)| class == spelling))
        .map(|(_, p)| *p)
}

/// The answers to a ranking's questions put back together (`_ranked`):
/// each candidate's answer, by its position among those given and as the
/// result names it, in the order asked. One row per row asked about, keyed by its row and entity, in the
/// order the answers first name it; each candidate scored by how sure its
/// answer was ([`how_sure`]), an answer that does not say refused
/// (`INTERNAL_ERROR`, naming the candidate and the classes it gave); sorted
/// best first, ties in the order asked, a NaN last (register PD-1), and
/// cut to `top_k`.
pub fn rank(answers: &[(u32, String, Part)], top_k: Option<u32>) -> Result<Vec<Row>, CoreError> {
    let mut rows: Vec<Row> = Vec::new();
    let mut at: BTreeMap<(i64, String), usize> = BTreeMap::new();
    for (candidate, named, part) in answers {
        for answer in &part.rows {
            let Some(score) = how_sure(answer.probabilities.as_ref()) else {
                let mut classes: Vec<&str> = answer
                    .probabilities
                    .iter()
                    .flatten()
                    .map(|(class, _)| class.as_str())
                    .collect();
                classes.sort_unstable();
                return Err(CoreError::new(
                    ErrorKind::Internal,
                    ErrorCode::InternalError,
                    "the endpoint answered a candidate's question without saying how sure it was",
                )
                .with_field("field", Detail::text("probabilities"))
                .with_field("candidate", Detail::Scalar(Scalar::UInt32(*candidate)))
                .with_field(
                    "classes",
                    Detail::List(
                        classes
                            .into_iter()
                            .map(|c| Scalar::Utf8(c.to_owned()))
                            .collect(),
                    ),
                ));
            };
            let key = (answer.row_index, format!("{:?}", answer.id));
            let row = *at.entry(key).or_insert_with(|| {
                rows.push(Row {
                    row_index: answer.row_index,
                    id: answer.id.clone(),
                    ranked: Vec::new(),
                });
                rows.len() - 1
            });
            rows[row].ranked.push((*candidate, named.clone(), score));
        }
    }
    for row in &mut rows {
        // Stable, best first, as Python's sort with `reverse=True` keeps
        // equal scores in the order asked; a NaN last (PD-1).
        row.ranked
            .sort_by(|(_, _, a), (_, _, b)| match (a.is_nan(), b.is_nan()) {
                (false, false) => b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal),
                (a_nan, b_nan) => a_nan.cmp(&b_nan),
            });
        if let Some(k) = top_k
            && k > 0
        {
            row.ranked.truncate(k as usize);
        }
    }
    Ok(rows)
}

/// A ranking's result: one row per entity asked about, ordered by
/// `row_index` (stable), with `row_index`, `id` (and `id_as_written`) as
/// [`crate::predict::table`] writes them, `rankings` (list of utf8, each
/// candidate as the result names it, best first), `scores` (list of
/// float64, each one's score, in the same order), `candidates` (list of
/// int64, each one's position among those given, in the same order) and
/// `answer_index`, the row's place in the order the answers named it.
pub fn table(rows: &[Row], text_ids: bool) -> Result<RecordBatch, CoreError> {
    let internal = || {
        CoreError::new(
            ErrorKind::Internal,
            ErrorCode::InternalError,
            "the ranking does not lay out",
        )
    };
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].row_index);
    let mut fields = vec![Field::new("row_index", DataType::Int64, false)];
    let mut columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from_iter_values(
        order.iter().map(|&i| rows[i].row_index),
    ))];
    let ids: Vec<&Entity> = order.iter().map(|&i| &rows[i].id).collect();
    let (id_fields, id_columns) = id_columns(&ids, text_ids);
    fields.extend(id_fields);
    columns.extend(id_columns);
    let mut rankings = ListBuilder::new(StringBuilder::new());
    let mut scores = ListBuilder::new(Float64Builder::new());
    let mut positions = ListBuilder::new(Int64Builder::new());
    for &i in &order {
        for (position, candidate, score) in &rows[i].ranked {
            rankings.values().append_value(candidate);
            scores.values().append_value(*score);
            positions.values().append_value(i64::from(*position));
        }
        rankings.append(true);
        scores.append(true);
        positions.append(true);
    }
    let rankings = rankings.finish();
    let scores = scores.finish();
    let positions = positions.finish();
    fields.push(Field::new("rankings", rankings.data_type().clone(), false));
    columns.push(Arc::new(rankings));
    fields.push(Field::new("scores", scores.data_type().clone(), false));
    columns.push(Arc::new(scores));
    fields.push(Field::new(
        "candidates",
        positions.data_type().clone(),
        false,
    ));
    columns.push(Arc::new(positions));
    fields.push(Field::new("answer_index", DataType::Int64, false));
    columns.push(Arc::new(Int64Array::from_iter_values(
        order.iter().map(|&i| i64::try_from(i).unwrap_or(i64::MAX)),
    )));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|_| internal())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repr(x: f64) -> String {
        // Enough of Python's repr for these tests.
        match x {
            1.0 => "1.0".into(),
            1e16 => "1e+16".into(),
            1.5e-7 => "1.5e-07".into(),
            _ => format!("{x}"),
        }
    }

    fn int(digits: &str) -> Candidate {
        Candidate::Int(Number::parse(digits).unwrap())
    }

    fn decimal(unscaled: &str, scale: i32) -> Candidate {
        Candidate::Decimal {
            unscaled: Number::parse(unscaled).unwrap(),
            scale,
        }
    }

    fn written(candidate: Candidate, dtype: Option<&str>) -> Result<String, String> {
        literal(&candidate, 0, dtype, repr).map_err(|e| {
            e.fields
                .get("rule")
                .map(|r| format!("{r:?}"))
                .unwrap_or_default()
        })
    }

    #[test]
    fn a_candidate_is_written_as_the_client_writes_it() {
        assert_eq!(written(int("7"), Some("int64")), Ok("7".into()));
        assert_eq!(written(int("7"), Some("string")), Ok("\"7\"".into()));
        assert_eq!(
            written(Candidate::Float(3.0), Some("int64")),
            Ok("3".into())
        );
        assert_eq!(
            written(Candidate::Float(-0.0), Some("int64")),
            Ok("0".into())
        );
        assert_eq!(
            written(Candidate::Float(1e20), Some("int64")),
            Ok("100000000000000000000".into())
        );
        assert!(written(Candidate::Float(2.5), Some("int64")).is_err());
        assert_eq!(
            written(Candidate::Float(1.0), Some("string")),
            Ok("\"1.0\"".into())
        );
        assert_eq!(
            written(Candidate::Float(1e16), Some("string")),
            Ok("\"10000000000000000\"".into())
        );
        assert_eq!(
            written(Candidate::Float(1.5e-7), Some("string")),
            Ok("\"0.00000015\"".into())
        );
        assert_eq!(
            written(Candidate::Text("a b".into()), Some("string")),
            Ok("\"a b\"".into())
        );
        assert!(written(Candidate::Text("a,b".into()), None).is_err());
        assert!(written(Candidate::Text(" \u{1f}".into()), None).is_err());
        assert!(written(Candidate::Missing, None).is_err());
        assert!(written(Candidate::Bool(true), None).is_err());
        assert_eq!(written(decimal("500", 2), Some("int64")), Ok("5".into()));
        assert!(written(decimal("510", 2), Some("int64")).is_err());
        assert_eq!(written(decimal("500", 2), Some("float64")), Ok("5".into()));
        assert!(written(decimal("1", 1), Some("float64")).is_err());
        assert!(written(decimal("5", 1), Some("float64")).is_err());
        assert_eq!(
            written(decimal("150", 2), Some("string")),
            Ok("\"1.50\"".into())
        );
        assert_eq!(
            written(decimal("-5", 3), Some("string")),
            Ok("\"-0.005\"".into())
        );
        assert_eq!(
            written(decimal("-0", 2), Some("string")),
            Ok("\"-0.00\"".into())
        );
        assert_eq!(written(decimal("5", 0), None), Ok("5".into()));
        assert_eq!(written(decimal("5", -2), None), Ok("500".into()));
        assert_eq!(
            written(Candidate::Spelled("2023-01-01".into()), Some("int64")),
            Ok("\"2023-01-01\"".into())
        );
    }

    #[test]
    fn a_number_past_any_scalar_is_written_by_its_every_digit() {
        let wide = format!("1{}", "0".repeat(38));
        // Past int64 a whole decimal travels as its digits where the column
        // is not known; on a column of integers it is its integer.
        assert_eq!(written(decimal(&wide, 0), None), Ok(format!("\"{wide}\"")));
        assert_eq!(written(decimal(&wide, 0), Some("int64")), Ok(wide.clone()));
        assert_eq!(
            written(decimal(&wide, 0), Some("string")),
            Ok(format!("\"{wide}\""))
        );
        // A double holds 1e38 only near it.
        assert!(written(decimal(&wide, 0), Some("float64")).is_err());
        let huge = format!("1{}123456789", "0".repeat(68));
        assert_eq!(written(int(&huge), Some("int64")), Ok(huge.clone()));
        assert_eq!(written(int(&huge), Some("float64")), Ok(huge.clone()));
        assert_eq!(
            written(int(&huge), Some("string")),
            Ok(format!("\"{huge}\""))
        );
        let tiny = format!("{}1", "0".repeat(29));
        assert_eq!(
            written(decimal("1", 30), Some("string")),
            Ok(format!("\"0.{tiny}\""))
        );
    }

    #[test]
    fn a_candidate_reads_from_every_form_a_host_gives() {
        use sdm_contract::ops::GivenCandidate as G;
        // 10**39, past i128, as a host writes it in a Decimal256.
        let wide = format!("1{}", "0".repeat(39));
        let scalar: Scalar =
            serde_json::from_str(&format!(r#"{{"t":"d256","v":"{wide}","p":76,"s":0}}"#)).unwrap();
        assert_eq!(read(&G::Value(scalar)), decimal(&wide, 0));
        assert_eq!(read(&G::Integer("-007".into())), int("-7"));
        assert_eq!(read(&G::Decimal("-0.00".into())), decimal("-0", 2));
        assert_eq!(read(&G::Decimal("12.50".into())), decimal("1250", 2));
        assert_eq!(
            read(&G::Decimal("1e5".into())),
            Candidate::Other("decimal".into())
        );
        assert_eq!(
            read(&G::Integer("1.5".into())),
            Candidate::Other("integer".into())
        );
        assert_eq!(
            read(&G::Spelled("x".into())),
            Candidate::Spelled("x".into())
        );
        assert_eq!(read(&G::Repeat), Candidate::Repeat);
    }

    #[test]
    fn candidates_are_kept_once_in_the_order_first_given() {
        let given = vec![
            int("2"),
            int("1"),
            int("2"),
            Candidate::Missing,
            Candidate::Missing,
        ];
        let kept: Vec<u32> = distinct(&given).into_iter().map(|(p, _)| p).collect();
        assert_eq!(kept, vec![0, 1, 3]);
        let floats = vec![Candidate::Float(0.0), Candidate::Float(-0.0)];
        assert_eq!(distinct(&floats).len(), 1);
    }

    #[test]
    fn candidates_equal_as_the_reference_holds_them_are_one() {
        let given = vec![
            int("1"),
            Candidate::Float(1.0),
            Candidate::Bool(true),
            decimal("100", 2),
            Candidate::Text("1".into()),
            Candidate::Float(0.5),
            decimal("5", 1),
            Candidate::Float(f64::NAN),
            Candidate::Float(f64::NAN),
            decimal("-0", 3),
            int("0"),
        ];
        let kept: Vec<u32> = distinct(&given).into_iter().map(|(p, _)| p).collect();
        assert_eq!(kept, vec![0, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn a_value_spelled_as_text_is_no_text_and_a_repeat_is_left_out() {
        let given = vec![
            Candidate::Spelled("2023-01-01".into()),
            Candidate::Text("2023-01-01".into()),
            Candidate::Repeat,
            Candidate::Spelled("2023-01-01".into()),
            int("1"),
        ];
        let kept: Vec<u32> = distinct(&given).into_iter().map(|(p, _)| p).collect();
        assert_eq!(kept, vec![0, 1, 3, 4]);
    }

    #[test]
    fn a_candidate_is_named_by_the_json_text_of_its_value() {
        assert_eq!(json_text(&int("-3"), repr), Some("-3".into()));
        assert_eq!(
            json_text(&Candidate::Float(1e16), repr),
            Some("1e+16".into())
        );
        assert_eq!(json_text(&decimal("150", 2), repr), Some("1.50".into()));
        assert_eq!(
            json_text(&Candidate::Text("a\"b".into()), repr),
            Some("\"a\\\"b\"".into())
        );
        assert_eq!(json_text(&Candidate::Missing, repr), None);
    }

    fn answered(scores: &[(i64, f64)]) -> Part {
        Part {
            model: "m".into(),
            request_id: "r".into(),
            metadata: sdm_wire::json::Value::Object(vec![]),
            rows: scores
                .iter()
                .map(|(row, p)| crate::predict::Row {
                    row_index: *row,
                    forecast_step: None,
                    answer_index: 0,
                    id: Entity::Integer(*row),
                    prediction: crate::predict::Answer::Bool(true),
                    probabilities: Some(vec![("False".into(), 1.0 - p), ("True".into(), *p)]),
                    scores: None,
                    rankings: None,
                    explanation: None,
                    metadata: None,
                    quantiles: None,
                    embeddings: None,
                })
                .collect(),
            answers: crate::predict::AnswerType::Bool,
            text_ids: false,
            explanation: None,
        }
    }

    #[test]
    fn a_ranking_is_best_first_ties_in_order_asked_a_nan_last_cut_to_k() {
        let answers = vec![
            (0, "\"a\"".to_owned(), answered(&[(0, 0.2), (1, f64::NAN)])),
            (1, "\"b\"".to_owned(), answered(&[(0, 0.9), (1, 0.5)])),
            (2, "\"c\"".to_owned(), answered(&[(0, 0.2), (1, 0.7)])),
        ];
        let rows = rank(&answers, Some(2)).unwrap();
        fn named(row: &Row) -> Vec<&str> {
            row.ranked.iter().map(|(_, c, _)| c.as_str()).collect()
        }
        assert_eq!(named(&rows[0]), ["\"b\"", "\"a\""]);
        assert_eq!(named(&rows[1]), ["\"c\"", "\"b\""]);
        let all = rank(&answers, None).unwrap();
        assert_eq!(named(&all[1]), ["\"c\"", "\"b\"", "\"a\""]);
        // An answer with no positive class's probability is refused.
        let mut silent = answered(&[(0, 0.5)]);
        silent.rows[0].probabilities = Some(vec![("x".into(), 0.5)]);
        let error = rank(&[(4, "4".into(), silent)], None).unwrap_err();
        assert_eq!(error.code, ErrorCode::InternalError);
    }

    #[test]
    fn a_float_is_written_without_an_exponent() {
        assert_eq!(positional("1e+16"), "10000000000000000");
        assert_eq!(positional("-1.5e+16"), "-15000000000000000");
        assert_eq!(positional("1.5e-07"), "0.00000015");
        assert_eq!(positional("1.25e+01"), "12.5");
        assert_eq!(positional("0.1"), "0.1");
    }
}
