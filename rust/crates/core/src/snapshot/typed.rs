// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_schema::pyval::PyVal;
use serde_json::{Map, Value, json};

use super::{Error, Result, array, decode_typed, object, require, typed_observation};

pub(super) fn value(input: &Value) -> Result<PyVal> {
    let mapping = object(input)?;
    require(mapping.len() == 1, "invalid typed input")?;
    let (kind, contents) = mapping.iter().next().ok_or(Error("invalid typed input"))?;
    Ok(match kind.as_str() {
        "scalar" | "float" => {
            PyVal::decode(&decode_typed(input)?).map_err(|_| Error("invalid typed scalar"))?
        }
        "list" => PyVal::List(array(contents)?.iter().map(value).collect::<Result<_>>()?),
        "tuple" => PyVal::Tuple(array(contents)?.iter().map(value).collect::<Result<_>>()?),
        "mapping" => {
            let mut pairs = Vec::new();
            for entry in array(contents)? {
                let entry = array(entry)?;
                require(entry.len() == 2, "invalid typed mapping pair")?;
                let key = value(&entry[0])?;
                require(
                    !pairs.iter().any(|(previous, _)| previous == &key),
                    "duplicate typed mapping key",
                )?;
                pairs.push((key, value(&entry[1])?));
            }
            PyVal::Dict(pairs)
        }
        "fixture_scalar" => PyVal::decode(contents).map_err(|_| Error("invalid fixture scalar"))?,
        _ => return Err(Error("unsupported typed value")),
    })
}

fn members(value: &PyVal) -> Result<&[(PyVal, PyVal)]> {
    if let PyVal::Dict(pairs) = value {
        Ok(pairs)
    } else {
        Err(Error("expected typed mapping"))
    }
}

pub(super) fn sequence_json(value: &PyVal) -> Result<Value> {
    Ok(match value {
        PyVal::Tuple(values) | PyVal::List(values) => {
            Value::Array(values.iter().map(sequence_json).collect::<Result<_>>()?)
        }
        PyVal::Dict(pairs) => {
            let mut mapping = Map::new();
            for (key, value) in pairs {
                let PyVal::Str(key) = key else {
                    return Err(Error("nontext sampling key"));
                };
                require(
                    mapping.insert(key.clone(), sequence_json(value)?).is_none(),
                    "duplicate sampling key",
                )?;
            }
            Value::Object(mapping)
        }
        PyVal::None => Value::Null,
        PyVal::Bool(value) => json!(value),
        PyVal::Str(value) => json!(value),
        PyVal::Int(value) => {
            if let Ok(value) = i64::try_from(*value) {
                json!(value)
            } else if let Ok(value) = u64::try_from(*value) {
                json!(value)
            } else {
                return Err(Error("wide integer requires lossless replay codec"));
            }
        }
        PyVal::Float(value) if value.is_finite() => json!(value),
        _ => return Err(Error("unsupported sampling scalar")),
    })
}

fn member<'a>(value: &'a PyVal, name: &str) -> Result<&'a PyVal> {
    members(value)?
        .iter()
        .find_map(|(key, value)| matches!(key, PyVal::Str(key) if key == name).then_some(value))
        .ok_or(Error("missing typed field"))
}

fn sequence(value: &PyVal) -> Result<&[PyVal]> {
    match value {
        PyVal::List(values) | PyVal::Tuple(values) => Ok(values),
        _ => Err(Error("expected typed sequence")),
    }
}

fn strings(value: &PyVal) -> Result<Vec<String>> {
    sequence(value)?
        .iter()
        .map(|value| match value {
            PyVal::Str(value) => Ok(value.clone()),
            _ => Err(Error("expected typed text")),
        })
        .collect()
}

pub(super) fn result(input: &PyVal) -> Result<sdm_request::result::Result_> {
    let mut normalized = Map::new();
    for (key, value) in members(input)? {
        let PyVal::Str(key) = key else {
            return Err(Error("invalid result field"));
        };
        let name = match key.as_str() {
            "_explanation" => "explanation",
            "_unscored_candidates" => "unscored_candidates",
            "_names_its_rows" => "names_its_rows",
            "_names_its_step" => "names_its_step",
            name => name,
        };
        let encoded = match name {
            "outputs" => json!(strings(value)?),
            "unscored_candidates" | "items" => Value::Array(
                sequence(value)?
                    .iter()
                    .map(sdm_request::layout::encode_deep)
                    .collect(),
            ),
            _ => sdm_request::layout::encode_deep(value),
        };
        require(
            normalized.insert(name.to_owned(), encoded).is_none(),
            "duplicate result field",
        )?;
    }
    sdm_request::layout::decode_result(&Value::Object(normalized))
        .map_err(|_| Error("invalid result input"))
}

pub(super) fn absent(input: &Value) -> Result<Value> {
    let input = super::host::input_value(input)?;
    let result = result(member(&input, "result")?)?;
    let requested = strings(member(&input, "requested")?)?;
    let absent = result.absent(&requested);
    typed_observation(
        json!({"tuple": absent.into_iter().map(|name| json!({"scalar": name})).collect::<Vec<_>>()}),
        Value::Null,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_accepts_existing_timestamp_input_metadata() {
        let scalar = json!({"pandas_scalar": {
            "class": "Timestamp",
            "value": {"numpy_scalar": {"dtype": "<M8[ns]", "bytes_hex": "0100000000000000"}},
            "timezone": null,
            "fold": 0
        }});
        let input = json!({"mapping": [
            [{"scalar": "result"}, {"mapping": [
                [{"scalar": "outputs"}, {"tuple": [{"scalar": "prediction"}]}],
                [{"scalar": "items"}, {"list": [{"mapping": [
                    [{"scalar": "row_index"}, {"scalar": 0}],
                    [{"scalar": "id"}, scalar],
                    [{"scalar": "prediction"}, {"scalar": 1}]
                ]}]}]
            ]}],
            [{"scalar": "requested"}, {"list": [{"scalar": "embeddings"}]}]
        ]});
        assert_eq!(
            absent(&input).unwrap()["observation"]["result"],
            json!({"tuple": [{"scalar": "embeddings"}]})
        );
        for malformed in [
            json!({"pandas_scalar": {"class": "Other"}}),
            json!({"numpy_scalar": {"dtype": "<M8[ns]", "bytes_hex": "00"}}),
            json!({"unknown_scalar": 1}),
        ] {
            let mut changed = input.clone();
            changed["mapping"][0][1]["mapping"][1][1]["list"][0]["mapping"][1][1] = malformed;
            assert!(absent(&changed).is_err());
        }
    }

    #[test]
    fn containers_keep_their_original_types_and_keys() {
        assert!(matches!(
            value(&json!({"tuple": []})).unwrap(),
            PyVal::Tuple(_)
        ));
        assert!(matches!(
            value(&json!({"list": []})).unwrap(),
            PyVal::List(_)
        ));
        let mapping =
            value(&json!({"mapping": [[{"scalar": 1}, {"tuple": [{"scalar": true}]}]]})).unwrap();
        assert_eq!(
            mapping,
            PyVal::Dict(vec![(PyVal::Int(1), PyVal::Tuple(vec![PyVal::Bool(true)]))])
        );
        assert!(value(&json!({"mapping": [[{"scalar": 1}, {"scalar": 2}], [{"scalar": 1}, {"scalar": 3}]]})).is_err());
    }
}
