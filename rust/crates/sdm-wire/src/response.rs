// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The model API's prediction response, read as the client's generated
//! bindings read it (`PredictionResponse.from_dict` and
//! `PredictionItem.from_dict` of `sdm_api.generated.tfm_api`), for the
//! fields it reads: the response's `id`, `model`, `predictions` and
//! `metadata`, and each item's `id`, `row_index` and `forecast_step`, each
//! required and typed as the bindings require it, checked in their order.
//! For those fields a response the bindings refuse is refused here, and
//! one they read is read (`contract/response-vectors.json` holds the two
//! together). A refusal names the field and why, never a value.
//!
//! From T1.28b an item's `probabilities` is read as the bindings'
//! `_float_dict` reads it: absent or `null` is none, any other value not an
//! object is refused (`not_an_object`), and each member's value is a
//! number, or text a float reads, not a boolean (`not_a_number`) and
//! finite (`not_finite`), kept in the order written.
//!
//! From T2.8 an item's `scores` is read as the bindings' `_float_tuple`
//! reads it (an array, each member read as a probability is), and its
//! `rankings` as their `_mapping_tuple` reads it (each member a mapping,
//! [`mappings`]).
//!
//! From T2.13 an item's `explanation` and `metadata` are each read as the
//! bindings' `_dict_or_none` reads them, Python's `dict` of the value
//! ([`mapping`], the one reader T2.8 lands).
//!
//! From T2.9 an item's `quantiles` is read as the bindings' `_float_dict`
//! reads it, as `probabilities` is (an object, each member's value a
//! finite number, each refusal naming `prediction.quantiles`), and kept as
//! the [`mapping`] it is, each value written as the double it reads as.
//!
//! From T2.11 an item's `embeddings` is read as the bindings'
//! `_float_tuple` reads it, as `scores` is: absent or `null` is none, any
//! value not an array refused (`not_a_sequence`), each member a number.
//!
//! As built at T1.11d the item's `prediction` is taken as written, as the
//! bindings take it. From T2.11 every other field the bindings read is
//! read, so [`PredictionItem::carried`] names none. Members the bindings
//! do not read are ignored, as there.

use crate::json::{Value, is_integer_token, whole};

/// Why a response is not one the contract describes: the field, and the
/// fault, never the value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refused {
    /// The field, as `prediction.row_index` or `metadata`.
    pub field: &'static str,
    /// `missing`, `null`, `not_text`, `not_whole`, `out_of_range`,
    /// `not_an_object` or `not_a_sequence`.
    pub reason: &'static str,
}

fn refused(field: &'static str, reason: &'static str) -> Refused {
    Refused { field, reason }
}

/// The fields of an item this build does not read, in the bindings' order.
pub const UNREAD_ITEM_FIELDS: [&str; 0] = [];

/// One answer.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionItem {
    /// Which row of the request this answers; a whole number, of any size
    /// a response may write.
    pub row_index: i128,
    pub id: Option<String>,
    pub forecast_step: Option<i128>,
    /// The answer as the response wrote it; absent and `null` alike are
    /// `None`, as the bindings read both.
    pub prediction: Option<Value>,
    /// Each class's probability, in the order written; absent and `null`
    /// alike are `None`.
    pub probabilities: Option<Vec<(String, f64)>>,
    /// The item's scores, in the order written; absent and `null` alike are
    /// `None` (T2.8).
    pub scores: Option<Vec<f64>>,
    /// The item's rankings, each a mapping as a JSON object, in the order
    /// written; absent and `null` alike are `None` (T2.8).
    pub rankings: Option<Vec<Value>>,
    /// The item's explanation, a mapping as [`mapping`] lays one out;
    /// absent and `null` alike are `None` (T2.13).
    pub explanation: Option<Value>,
    /// The item's own metadata, a mapping as [`mapping`] lays one out;
    /// absent and `null` alike are `None` (T2.13).
    pub metadata: Option<Value>,
    /// The item's quantiles, a mapping as [`mapping`] lays one out, each
    /// value the double it reads as; absent and `null` alike are `None`
    /// (T2.9).
    pub quantiles: Option<Value>,
    /// The item's embedding, in the order written; absent and `null` alike
    /// are `None` (T2.11).
    pub embeddings: Option<Vec<f64>>,
    /// The fields of [`UNREAD_ITEM_FIELDS`] this item carries, not null.
    pub carried: Vec<&'static str>,
}

/// The response to one prediction request.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResponse {
    pub id: String,
    pub model: String,
    pub predictions: Vec<PredictionItem>,
    /// The endpoint's `metadata` object, as written.
    pub metadata: Value,
}

impl PredictionResponse {
    /// Reads `value` as the bindings read a response.
    pub fn read(value: &Value) -> Result<PredictionResponse, Refused> {
        let Value::Object(_) = value else {
            return Err(refused("response", "not_an_object"));
        };
        let id = required_text(value, "id", "id")?;
        let model = required_text(value, "model", "model")?;
        let predictions = match value.get("predictions") {
            None => return Err(refused("predictions", "missing")),
            Some(Value::Array(items)) => items
                .iter()
                .map(PredictionItem::read)
                .collect::<Result<_, _>>()?,
            // An object iterates over its names, each of which is text and
            // so not an item; with none, there are no items.
            Some(Value::Object(members)) if members.is_empty() => vec![],
            Some(Value::Object(_)) => return Err(refused("prediction", "not_an_object")),
            Some(_) => return Err(refused("predictions", "not_a_sequence")),
        };
        let metadata = match value.get("metadata") {
            None => return Err(refused("metadata", "missing")),
            Some(object @ Value::Object(_)) => object.clone(),
            Some(_) => return Err(refused("metadata", "not_an_object")),
        };
        Ok(PredictionResponse {
            id,
            model,
            predictions,
            metadata,
        })
    }
}

impl PredictionItem {
    fn read(value: &Value) -> Result<PredictionItem, Refused> {
        let Value::Object(_) = value else {
            return Err(refused("prediction", "not_an_object"));
        };
        let id = match value.get("id") {
            None | Some(Value::Null) => None,
            Some(Value::Text(text)) => Some(text.clone()),
            Some(_) => return Err(refused("prediction.id", "not_text")),
        };
        let row_index = match value.get("row_index") {
            None => return Err(refused("prediction.row_index", "missing")),
            Some(Value::Null) => return Err(refused("prediction.row_index", "null")),
            Some(number) => whole_of(number, "prediction.row_index")?,
        };
        let forecast_step = match value.get("forecast_step") {
            None | Some(Value::Null) => None,
            Some(number) => Some(whole_of(number, "prediction.forecast_step")?),
        };
        let prediction = match value.get("prediction") {
            None | Some(Value::Null) => None,
            Some(answer) => Some(answer.clone()),
        };
        let probabilities = match value.get("probabilities") {
            None | Some(Value::Null) => None,
            Some(Value::Object(members)) => Some(
                members
                    .iter()
                    .map(|(class, p)| Ok((class.clone(), probability(p)?)))
                    .collect::<Result<_, Refused>>()?,
            ),
            Some(_) => return Err(refused("prediction.probabilities", "not_an_object")),
        };
        let scores = match value.get("scores") {
            None | Some(Value::Null) => None,
            Some(Value::Array(members)) => Some(
                members
                    .iter()
                    .map(|v| number(v, "prediction.scores"))
                    .collect::<Result<_, Refused>>()?,
            ),
            Some(_) => return Err(refused("prediction.scores", "not_a_sequence")),
        };
        let rankings = match value.get("rankings") {
            None | Some(Value::Null) => None,
            Some(given) => Some(mappings(given)?),
        };
        let explanation = match value.get("explanation") {
            None | Some(Value::Null) => None,
            Some(given) => Some(mapping(given, "prediction.explanation")?),
        };
        let metadata = match value.get("metadata") {
            None | Some(Value::Null) => None,
            Some(given) => Some(mapping(given, "prediction.metadata")?),
        };
        let quantiles = match value.get("quantiles") {
            None | Some(Value::Null) => None,
            Some(given @ Value::Object(_)) => Some(numbers(
                mapping(given, "prediction.quantiles")?,
                "prediction.quantiles",
            )?),
            Some(_) => return Err(refused("prediction.quantiles", "not_an_object")),
        };
        let embeddings = match value.get("embeddings") {
            None | Some(Value::Null) => None,
            Some(Value::Array(members)) => Some(
                members
                    .iter()
                    .map(|v| number(v, "prediction.embeddings"))
                    .collect::<Result<_, Refused>>()?,
            ),
            Some(_) => return Err(refused("prediction.embeddings", "not_a_sequence")),
        };
        let carried = UNREAD_ITEM_FIELDS
            .into_iter()
            .filter(|name| !matches!(value.get(name), None | Some(Value::Null)))
            .collect();
        Ok(PredictionItem {
            row_index,
            id,
            forecast_step,
            prediction,
            probabilities,
            scores,
            rankings,
            explanation,
            metadata,
            quantiles,
            embeddings,
            carried,
        })
    }
}

fn required_text(value: &Value, name: &str, field: &'static str) -> Result<String, Refused> {
    match value.get(name) {
        None => Err(refused(field, "missing")),
        Some(Value::Text(text)) => Ok(text.clone()),
        Some(_) => Err(refused(field, "not_text")),
    }
}

/// A whole number, as the bindings' `_whole` takes one: an integer, or a
/// finite float with no fraction; never a boolean or text. A whole number
/// past `i128` (an integer token, or a finite float with no fraction, at
/// or past 2**127 in size) is one the bindings take, refused as
/// `out_of_range`, as a row past the request's rows is; a fraction, an
/// infinity or anything else as `not_whole`.
fn whole_of(value: &Value, field: &'static str) -> Result<i128, Refused> {
    let Value::Number(token) = value else {
        return Err(refused(field, "not_whole"));
    };
    if let Some(number) = whole(token) {
        return Ok(number);
    }
    let past = is_integer_token(token)
        || token
            .parse::<f64>()
            .is_ok_and(|float| float.is_finite() && float.fract() == 0.0);
    Err(refused(
        field,
        if past { "out_of_range" } else { "not_whole" },
    ))
}

/// A probability as the bindings' `_finite` reads one: a number, or text
/// Python's `float` reads ([`numeric_text`]), never a boolean, and finite.
fn probability(value: &Value) -> Result<f64, Refused> {
    number(value, "prediction.probabilities")
}

/// A mapping whose every value is a number, as the bindings' `_float_dict`
/// holds one: each value read as [`number`] reads it (refused naming
/// `field`), and written as the double it reads as, so text a float reads
/// and an integer alike are that double.
fn numbers(mapping: Value, field: &'static str) -> Result<Value, Refused> {
    let Value::Object(members) = mapping else {
        return Err(refused(field, "not_an_object"));
    };
    members
        .into_iter()
        .map(|(name, v)| {
            let read = number(&v, field)?;
            Ok((
                name,
                Value::float(read).ok_or(refused(field, "not_finite"))?,
            ))
        })
        .collect::<Result<_, Refused>>()
        .map(Value::Object)
}

/// One mapping of a response field, as Python's `dict` makes one of the
/// value the transport read: of an object as it is, of a list of pairs, and
/// of empty text none. A pair is a list of two whose first member a dict
/// may key by (text, a number, a boolean or null), text of two characters
/// (a character to a character), or an object of two members (its first
/// name to its second, as `dict` iterates an object's names). A key given
/// twice keeps its first place and spelling and takes the last value, keys
/// equal as Python's are ([`key_of`]). Anything else is refused
/// (`not_a_mapping`, naming `field`), as `dict` refuses it.
///
/// The one reader of a mapping-valued response field (T2.8; the
/// explanation, the item metadata and the quantiles read it too): the core
/// lays a mapping out as JSON object text in a Utf8 column, a list of
/// mappings as a list of those texts. A mapping keyed by text alone is the
/// JSON object; one with another key (a number, a boolean or null) is the
/// JSON list of its pairs, `[[key, value], ...]`, as a JSON object holds no
/// other key. Both forms are read here, keys equal as Python's are in both.
pub fn mapping(member: &Value, field: &'static str) -> Result<Value, Refused> {
    let not = || refused(field, "not_a_mapping");
    // Two characters, as `dict` reads a pair from text.
    let pair_of = |text: &str| -> Option<(Value, Value)> {
        let mut chars = text.chars();
        let (Some(a), Some(b), None) = (chars.next(), chars.next(), chars.next()) else {
            return None;
        };
        Some((Value::Text(a.to_string()), Value::Text(b.to_string())))
    };
    match member {
        Value::Object(_) => Ok(member.clone()),
        Value::Text(text) if text.is_empty() => Ok(Value::Object(vec![])),
        Value::Array(pairs) => {
            let mut built: Vec<(Value, Value)> = Vec::new();
            for pair in pairs {
                let (key, value) = match pair {
                    Value::Array(two) => match two.as_slice() {
                        [Value::Array(_) | Value::Object(_), _] => return Err(not()),
                        [key, value] => (key.clone(), value.clone()),
                        _ => return Err(not()),
                    },
                    Value::Text(text) => pair_of(text).ok_or_else(not)?,
                    Value::Object(two) => match two.as_slice() {
                        [(first, _), (second, _)] => {
                            (Value::Text(first.clone()), Value::Text(second.clone()))
                        }
                        _ => return Err(not()),
                    },
                    _ => return Err(not()),
                };
                let named = key_of(&key);
                match built
                    .iter_mut()
                    .find(|(k, _)| named.is_some() && key_of(k) == named)
                {
                    Some(kept) => kept.1 = value,
                    None => built.push((key, value)),
                }
            }
            if built.iter().all(|(k, _)| matches!(k, Value::Text(_))) {
                Ok(Value::Object(
                    built
                        .into_iter()
                        .filter_map(|(k, v)| match k {
                            Value::Text(name) => Some((name, v)),
                            _ => None,
                        })
                        .collect(),
                ))
            } else {
                Ok(Value::Array(
                    built
                        .into_iter()
                        .map(|(k, v)| Value::Array(vec![k, v]))
                        .collect(),
                ))
            }
        }
        _ => Err(not()),
    }
}

/// A rankings field as the bindings' `_mapping_tuple` reads it, `tuple(
/// dict(item) for item in value)`: each member a [`mapping`]; text iterates
/// its characters and an object its names, each made a mapping as a member
/// is. Anything else is refused (`not_a_mapping`).
pub fn mappings(value: &Value) -> Result<Vec<Value>, Refused> {
    const FIELD: &str = "prediction.rankings";
    let one = |member: &Value| mapping(member, FIELD);
    match value {
        Value::Array(members) => members.iter().map(one).collect(),
        Value::Text(text) => text
            .chars()
            .map(|c| one(&Value::Text(c.to_string())))
            .collect(),
        Value::Object(members) => members
            .iter()
            .map(|(name, _)| one(&Value::Text(name.clone())))
            .collect(),
        _ => Err(refused(FIELD, "not_a_mapping")),
    }
}

/// A dict key as Python's equality holds it ([`crate::key`]), which the
/// reference's client keys a response's mappings by (`dict`), so two keys
/// the reference holds one are one here: every NaN one key. Two NaN
/// objects are two keys to Python's `dict`, but the client reads a
/// response with `json.loads`, which gives every `NaN` token the same NaN
/// object, and `dict` finds a key that is the same object before it
/// compares, so the reference keeps one key for them all (T2.13: the
/// reference's `PredictionItem.from_dict` makes `[[NaN, 1], [NaN, 2]]`
/// `{NaN: 2}`).
fn key_of(key: &Value) -> Option<String> {
    match key {
        Value::Text(text) => Some(crate::key::text(text)),
        Value::Null => Some(crate::key::MISSING.to_owned()),
        Value::Bool(b) => Some(crate::key::boolean(*b)),
        Value::Number(token) if is_integer_token(token) => Some(crate::key::exact(token)),
        Value::Number(token) => {
            let float: f64 = token.parse().ok()?;
            Some(crate::key::float(float).unwrap_or_else(|| "x".to_owned()))
        }
        _ => None,
    }
}

/// A number as the bindings' `_finite` reads one, for `field`: a number,
/// or text Python's `float` reads ([`numeric_text`]), never a boolean, and
/// finite.
fn number(value: &Value, field: &'static str) -> Result<f64, Refused> {
    let read = match value {
        Value::Number(token) => token.parse::<f64>().ok(),
        Value::Text(text) => numeric_text(text),
        _ => None,
    };
    match read {
        None => Err(refused(field, "not_a_number")),
        Some(p) if !p.is_finite() => Err(refused(field, "not_finite")),
        Some(p) => Ok(p),
    }
}

/// The first code point of each run of ten Unicode decimal digits (0 to 9)
/// that every supported Python's `float` takes as the digits they stand for:
/// Unicode 13.0's (`unicodedata.decimal` of Python 3.10, the floor), a
/// portable policy (T1.28b fix round 2): a digit later Unicode versions add,
/// which the reference's Python 3.12 and later take, is refused (register
/// P1B-25).
const DECIMAL_ZEROS: [u32; 65] = [
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x16A60, 0x16B50, 0x1D7CE,
    0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E950, 0x1FBF0,
];

/// The white space every supported Python's `float` strips around a
/// number: what `str.isspace` takes as white space but the four separators
/// 0x1C to 0x1F, which `float` refuses (measured on Python 3.10 to 3.14).
const WHITE_SPACE: [u32; 25] = [
    0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
    0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
];

/// Text as every supported Python's `float` and `int` see it before they
/// read it: each white space a space and each Unicode decimal digit the
/// ASCII digit it stands for, the spaces around it trimmed, and a single
/// underscore between two digits dropped; `None` for text holding any other
/// character outside ASCII, or an underscore anywhere else.
fn ascii_digits(text: &str) -> Option<String> {
    let mut ascii = String::with_capacity(text.len());
    for c in text.chars() {
        let code = u32::from(c);
        if WHITE_SPACE.contains(&code) {
            ascii.push(' ');
        } else if let Some(zero) = DECIMAL_ZEROS
            .iter()
            .find(|zero| (**zero..**zero + 10).contains(&code))
        {
            ascii.push(char::from(b'0' + u8::try_from(code - zero).ok()?));
        } else if c.is_ascii() {
            ascii.push(c);
        } else {
            return None;
        }
    }
    let trimmed = ascii.trim_matches(' ');
    let bytes = trimmed.as_bytes();
    // A single underscore between two digits, and nowhere else.
    for (at, byte) in bytes.iter().enumerate() {
        if *byte == b'_'
            && !(at > 0
                && bytes[at - 1].is_ascii_digit()
                && bytes.get(at + 1).is_some_and(u8::is_ascii_digit))
        {
            return None;
        }
    }
    Some(trimmed.chars().filter(|c| *c != '_').collect())
}

/// Text as Python's `int` reads it (the grammar its documentation gives for
/// a string argument in base 10), as `ascii_digits` sees it: a sign, then
/// decimal digits; the integer's digits, with no leading zero and a minus
/// sign only for a negative one, or `None` for any other text.
pub fn integer_text(text: &str) -> Option<String> {
    let plain = ascii_digits(text)?;
    let negative = plain.starts_with('-');
    let digits = plain.strip_prefix(['+', '-']).unwrap_or(&plain);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    Some(match (digits.is_empty(), negative) {
        (true, _) => "0".to_owned(),
        (false, true) => format!("-{digits}"),
        (false, false) => digits.to_owned(),
    })
}

/// Text as Python's `float` reads it (the grammar its documentation gives
/// for a string argument): white space around it, a sign, then `inf`,
/// `infinity` or `nan` in any case, or a decimal number (digits with an
/// optional fraction, or a fraction alone, then an optional exponent), any
/// Unicode decimal digit read as the digit it stands for and a single
/// underscore allowed between two digits; `None` for any other text.
pub fn numeric_text(text: &str) -> Option<f64> {
    let plain = ascii_digits(text)?;
    let unsigned = plain.strip_prefix(['+', '-']).unwrap_or(&plain);
    let lower = unsigned.to_ascii_lowercase();
    if matches!(lower.as_str(), "inf" | "infinity" | "nan") {
        let negative = plain.starts_with('-');
        return Some(match (lower.as_str(), negative) {
            ("nan", _) => f64::NAN,
            (_, false) => f64::INFINITY,
            (_, true) => f64::NEG_INFINITY,
        });
    }
    // digits [. [digits]] | . digits, then [e [sign] digits]
    let (mantissa, exponent) = match unsigned.find(['e', 'E']) {
        Some(at) => (&unsigned[..at], Some(&unsigned[at + 1..])),
        None => (unsigned, None),
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let (whole, fraction) = match mantissa.split_once('.') {
        Some((whole, fraction)) => (whole, Some(fraction)),
        None => (mantissa, None),
    };
    let mantissa_ok = match fraction {
        None => digits(whole),
        Some(fraction) => {
            (digits(whole) && (fraction.is_empty() || digits(fraction)))
                || (whole.is_empty() && digits(fraction))
        }
    };
    let exponent_ok = exponent.is_none_or(|e| digits(e.strip_prefix(['+', '-']).unwrap_or(e)));
    if !(mantissa_ok && exponent_ok) {
        return None;
    }
    plain.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(text: &str) -> Value {
        crate::json::read(text.as_bytes()).unwrap()
    }

    /// A response as the transport reads one, `NaN` and the infinities
    /// taken (`json::read_answer`).
    fn answer(text: &str) -> Value {
        crate::json::read_answer(text.as_bytes()).unwrap()
    }

    /// T2.13: `dict(value)` as the bindings' `_dict_or_none` makes it, each
    /// answer recorded through the reference's `PredictionItem.from_dict`
    /// (an item's explanation), a key kept as first written.
    #[test]
    fn a_mapping_is_made_as_python_s_dict_makes_one() {
        for (given, wanted) in [
            ("{}", "{}"),
            (r#"{"a":1,"b":[1,2]}"#, r#"{"a":1,"b":[1,2]}"#),
            (r#""""#, "{}"),
            ("[]", "{}"),
            (r#"[["a",1],["b",2]]"#, r#"{"a":1,"b":2}"#),
            (r#"[["a",1],["a",2]]"#, r#"{"a":2}"#),
            (r#"["ab","cd"]"#, r#"{"a":"b","c":"d"}"#),
            (r#"["ab","ac"]"#, r#"{"a":"c"}"#),
            (r#"[{"x":1,"y":2}]"#, r#"{"x":"y"}"#),
            (r#"[[1,"one"],[1.0,"uno"],[true,"si"]]"#, r#"[[1,"si"]]"#),
            ("[[null,1]]", "[[null,1]]"),
            (r#"[[1,2],["1",3]]"#, r#"[[1,2],["1",3]]"#),
            (r#"[["a",1],[2,3]]"#, r#"[["a",1],[2,3]]"#),
            (r#"[[-0,"z"],[0.0,"y"]]"#, r#"[[-0,"y"]]"#),
            ("[[NaN,1],[NaN,2]]", "[[NaN,2]]"),
            ("[[1e400,1],[1e999,2]]", "[[1e400,2]]"),
            (r#"["\u00e9\u00e8"]"#, r#"{"\u00e9":"\u00e8"}"#),
            (
                "[[9007199254740993,1],[9007199254740992.0,2]]",
                "[[9007199254740993,1],[9007199254740992.0,2]]",
            ),
            ("[[100,1],[1e2,2]]", "[[100,2]]"),
        ] {
            assert_eq!(
                mapping(&answer(given), "prediction.explanation").unwrap(),
                answer(wanted),
                "{given}"
            );
        }
        for given in [
            r#""ab""#,
            r#""a""#,
            r#"[{"x":1}]"#,
            "[[[1],2]]",
            r#"[[{"a":1},2]]"#,
            r#"[["a"]]"#,
            r#"[["a",1,2]]"#,
            "[1]",
            "[null]",
            "1",
            "true",
        ] {
            let refused = mapping(&answer(given), "prediction.metadata").unwrap_err();
            assert_eq!(
                (refused.field, refused.reason),
                ("prediction.metadata", "not_a_mapping"),
                "{given}"
            );
        }
    }

    /// Every NaN is one key, as the reference holds them: `json.loads` gives
    /// every `NaN` token the same object, which `dict` finds as one key
    /// (T2.13); the first place is kept and the last value taken, and a NaN
    /// is no other number's key.
    #[test]
    fn every_nan_key_is_one_key() {
        let read = |text: &str| mapping(&answer(text), "prediction.explanation").unwrap();
        assert_eq!(read("[[NaN,1],[NaN,2]]"), answer("[[NaN,2]]"));
        assert_eq!(read("[[NaN,1],[0,2],[NaN,3]]"), answer("[[NaN,3],[0,2]]"));
        assert_eq!(
            read("[[NaN,1],[Infinity,2],[-Infinity,3]]"),
            answer("[[NaN,1],[Infinity,2],[-Infinity,3]]")
        );
        assert_eq!(
            read(r#"[[NaN,1],["NaN",2]]"#),
            answer(r#"[[NaN,1],["NaN",2]]"#)
        );
    }

    #[test]
    fn a_mapping_is_read_as_python_s_dict_reads_it() {
        let made = |text: &str| {
            mapping(&read(text), "f").map(|v| String::from_utf8(v.to_bytes()).unwrap())
        };
        assert_eq!(made(r#"{"a":1}"#), Ok(r#"{"a":1}"#.to_owned()));
        // A pair of each form: a list of two, two characters, and an object
        // of two members, its first name to its second.
        assert_eq!(
            made(r#"[["a",1],"bc",{"d":0,"e":0}]"#),
            Ok(r#"{"a":1,"b":"c","d":"e"}"#.to_owned())
        );
        // Keys equal as Python's are one, the first kept, the last value.
        assert_eq!(
            made(r#"[[1,"x"],[1.0,"y"],[true,"z"]]"#),
            Ok(r#"[[1,"z"]]"#.to_owned())
        );
        assert!(made(r#"[{"a":0}]"#).is_err());
        assert!(made(r#"[[[1],2]]"#).is_err());
        assert!(made("3").is_err());
    }
}
