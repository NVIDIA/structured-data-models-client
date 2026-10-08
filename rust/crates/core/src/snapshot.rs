// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::Deserialize;
use serde_json::{Map, Value, json};

mod columns;
mod dates;
mod exact;
mod host;
mod labels;
mod sampling;
mod semantic;
mod sequences;
mod typed;

#[derive(Debug)]
pub struct Error(&'static str);

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for Error {}

type Result<T> = std::result::Result<T, Error>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    files: BTreeMap<String, String>,
    cases: Vec<String>,
    routes: BTreeMap<String, Route>,
    native_sha256: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Route {
    kind: String,
    native_functions: Vec<String>,
}

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error(message))
    }
}

fn object(value: &Value) -> Result<&Map<String, Value>> {
    value.as_object().ok_or(Error("expected an object"))
}

fn array(value: &Value) -> Result<&[Value]> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or(Error("expected an array"))
}

fn text(value: &Value) -> Result<&str> {
    value.as_str().ok_or(Error("expected text"))
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    object(value)?.get(name).ok_or(Error("missing input field"))
}

fn fields(value: &Value, names: &[&str]) -> Result<()> {
    let mapping = object(value)?;
    require(
        mapping.len() == names.len() && names.iter().all(|name| mapping.contains_key(*name)),
        "unexpected input fields",
    )
}

fn checked_path(path: &str) -> Result<()> {
    require(
        !path.is_empty()
            && !path.contains(['\\', ':', '\0'])
            && path.split('/').all(|part| !matches!(part, "" | "." | "..")),
        "invalid snapshot path",
    )
}

fn unhex(encoded: &str) -> Result<Vec<u8>> {
    require(encoded.len().is_multiple_of(2), "invalid snapshot bytes")?;
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => Ok(byte - b'0'),
                b'a'..=b'f' => Ok(byte - b'a' + 10),
                _ => Err(Error("invalid snapshot bytes")),
            };
            Ok(digit(pair[0])? * 16 + digit(pair[1])?)
        })
        .collect()
}

fn decode_typed(value: &Value) -> Result<Value> {
    let mapping = object(value)?;
    require(mapping.len() == 1, "invalid typed input")?;
    let (kind, contents) = mapping.iter().next().ok_or(Error("invalid typed input"))?;
    match kind.as_str() {
        "scalar" => {
            require(
                contents.is_null()
                    || contents.is_string()
                    || contents.is_boolean()
                    || contents.as_i64().is_some()
                    || contents.as_u64().is_some(),
                "invalid typed scalar",
            )?;
            Ok(contents.clone())
        }
        "list" => Ok(Value::Array(
            array(contents)?
                .iter()
                .map(decode_typed)
                .collect::<Result<_>>()?,
        )),
        "float" => {
            let encoded = text(contents)?;
            match encoded {
                "nan" | "inf" | "-inf" => Ok(json!({"$float": encoded})),
                _ => {
                    let negative = encoded.starts_with('-');
                    let unsigned = encoded.strip_prefix('-').unwrap_or(encoded);
                    let (significand, exponent) = unsigned
                        .strip_prefix("0x")
                        .and_then(|value| value.split_once('p'))
                        .ok_or(Error("invalid typed float"))?;
                    let (whole, fraction) = significand
                        .split_once('.')
                        .ok_or(Error("invalid typed float"))?;
                    require(
                        matches!(whole, "0" | "1") && !fraction.is_empty() && fraction.len() <= 13,
                        "invalid typed float",
                    )?;
                    let fraction_value = u64::from_str_radix(fraction, 16)
                        .map_err(|_| Error("invalid typed float"))?;
                    let exponent: i32 =
                        exponent.parse().map_err(|_| Error("invalid typed float"))?;
                    require((-1022..=1023).contains(&exponent), "invalid typed float")?;
                    let significand = if whole == "1" { 1.0 } else { 0.0 }
                        + fraction_value as f64 / (1u64 << (4 * fraction.len())) as f64;
                    let number =
                        significand * 2f64.powi(exponent) * if negative { -1.0 } else { 1.0 };
                    require(float_hex(number) == encoded, "noncanonical typed float")?;
                    Ok(json!(number))
                }
            }
        }
        "mapping" => {
            let mut decoded = Map::new();
            for entry in array(contents)? {
                let pair = array(entry)?;
                require(pair.len() == 2, "invalid typed mapping pair")?;
                let key = decode_typed(&pair[0])?;
                let key = text(&key)?;
                require(!decoded.contains_key(key), "duplicate typed mapping key")?;
                decoded.insert(key.to_owned(), decode_typed(&pair[1])?);
            }
            Ok(Value::Object(decoded))
        }
        _ => Err(Error("unsupported typed input")),
    }
}

fn float_hex(number: f64) -> String {
    let bits = number.to_bits();
    let sign = if bits >> 63 == 0 { "" } else { "-" };
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 && fraction == 0 {
        format!("{sign}0x0.0p+0")
    } else if exponent == 0 {
        format!("{sign}0x0.{fraction:013x}p-1022")
    } else {
        format!("{sign}0x1.{fraction:013x}p{:+}", exponent - 1023)
    }
}

fn encode_typed(value: Value) -> Result<Value> {
    match value {
        Value::Object(mapping) => Ok(json!({"mapping": mapping.into_iter().map(|(key, value)| {
            Ok(json!([{"scalar": key}, encode_typed(value)?]))
        }).collect::<Result<Vec<_>>>()?})),
        Value::Array(values) => {
            Ok(json!({"list": values.into_iter().map(encode_typed).collect::<Result<Vec<_>>>()?}))
        }
        Value::Number(ref number) if number.is_f64() => {
            Ok(json!({"float": float_hex(number.as_f64().ok_or(Error("invalid typed output"))?)}))
        }
        scalar => Ok(json!({"scalar": scalar})),
    }
}

fn generator(case: &Value) -> Result<Value> {
    let input = field(case, "replay_input")?;
    fields(
        input,
        if input.get("host_inputs").is_some() {
            &["generator", "generator_input", "host_inputs"]
        } else {
            &["generator", "generator_input"]
        },
    )?;
    let coordinate = field(input, "generator")?;
    fields(coordinate, &["name", "stream_seed", "ordinal"])?;
    let name = text(field(coordinate, "name")?)?;
    require(
        field(coordinate, "stream_seed")?.as_u64().is_some()
            && field(coordinate, "ordinal")?.as_u64().is_some(),
        "invalid generator coordinate",
    )?;
    if let Some(facts) = input.get("host_inputs") {
        if name == "rfm.sequences" {
            return sequences::replay(field(input, "generator_input")?, Some(facts));
        }
        if name == "schema.date_columns" {
            return dates::replay(field(input, "generator_input")?, facts);
        }
        if name == "labels.question_batches" {
            return labels::questions(field(input, "generator_input")?, Some(facts));
        }
        if name == "sampling.hashseed_array_jobs" {
            let arguments = typed::sequence_json(&typed::value(field(input, "generator_input")?)?)?;
            return sampling::replay(name, &arguments, Some(facts));
        }
        if name == "pql.date_literals" {
            fields(facts, &["strftime_year_953"])?;
            let rendered = text(field(facts, "strftime_year_953")?)?;
            require(
                matches!(rendered, "953" | "0953"),
                "invalid date runtime capability",
            )?;
            let arguments = decode_typed(field(input, "generator_input")?)?;
            require(
                field(&arguments, "operation")? == "pql.date",
                "invalid date host operation",
            )?;
            return pql_generator(&arguments, Some(rendered == "0953"));
        }
        require(
            matches!(name, "request.results" | "request.merge"),
            "unsupported host witness route",
        )?;
        return host::replay(name, field(input, "generator_input")?, Some(facts));
    }
    if matches!(name, "request.results" | "request.merge") {
        return host::replay(name, field(input, "generator_input")?, None);
    }
    if name == "rfm.sequences" {
        return sequences::replay(field(input, "generator_input")?, None);
    }
    if name == "request.absent" {
        return typed::absent(field(input, "generator_input")?);
    }
    if name == "labels.question_batches" {
        return labels::questions(field(input, "generator_input")?, None);
    }
    if matches!(
        name,
        "sampling.random_state"
            | "sampling.random_draws"
            | "sampling.graphs"
            | "sampling.graph_store"
            | "sampling.graph_store_calls"
            | "sampling.offset_pairs"
            | "sampling.offset_pooling"
            | "sampling.debug_jobs"
            | "sampling.hashseed_array_jobs"
            | "sampling.hashseed_python_jobs"
    ) {
        let arguments = typed::sequence_json(&typed::value(field(input, "generator_input")?)?)?;
        return sampling::replay(name, &arguments, None);
    }
    let arguments = decode_typed(field(input, "generator_input")?)?;
    let result = match name {
        "dialects.view_expressions" => {
            fields(
                &arguments,
                &[
                    "expression",
                    "dialects",
                    "functions",
                    "untrusted_only_if_safe",
                ],
            )?;
            require(
                field(&arguments, "dialects")? == &json!([null, "databricks", "snowflake"])
                    && field(&arguments, "functions")?
                        == &json!(["_unsafe_expr_reason", "_untrusted_call_reason"])
                    && field(&arguments, "untrusted_only_if_safe")? == &Value::Bool(true),
                "invalid view expression policy",
            )?;
            let expression = text(field(&arguments, "expression")?)?;
            let mut results = Vec::new();
            for dialect in [None, Some("databricks"), Some("snowflake")] {
                let reason = sdm_dialects::view_expr::unsafe_expr_reason(expression, dialect);
                results.push(json!({"tuple": [
                    {"scalar": "_unsafe_expr_reason"}, {"scalar": dialect}, {"scalar": reason}
                ]}));
                if reason.is_none()
                    && let Some(dialect) = dialect
                {
                    let reason =
                        sdm_dialects::view_expr::untrusted_call_reason(expression, dialect, &[])
                            .map_err(|_| Error("invalid view expression dialect"))?;
                    results.push(json!({"tuple": [
                        {"scalar": "_untrusted_call_reason"}, {"scalar": dialect}, {"scalar": reason}
                    ]}));
                }
            }
            return typed_observation(json!({"list": results}), Value::Null);
        }
        "dialects.random_state" => {
            fields(&arguments, &["calls"])?;
            let calls = array(field(&arguments, "calls")?)?;
            require(!calls.is_empty(), "empty generator calls")?;
            Value::Array(calls.iter().map(dialect_generator).collect::<Result<_>>()?)
        }
        "dialects.json" => dialect_generator(&arguments)?,
        "labels.aggregation" | "labels.aggregation_type" => {
            let (output, warnings) = sdm_labels::fixture_op_with_warnings(
                name,
                &arguments.to_string(),
                &sdm_schema::dt::Host::assumed(),
            )
            .map_err(|_| Error("invalid labels input"))?;
            let output =
                serde_json::from_str(&output).map_err(|_| Error("invalid labels output"))?;
            let mut observation = generator_observation(output, Value::Null)?;
            observation["observation"]["warnings"] = json!(warnings.into_iter().filter(|warning| !warning.masked_float32_cast()).map(|warning| json!({"class": warning.category(), "code": null, "message": warning.message(), "details": null})).collect::<Vec<_>>());
            return Ok(observation);
        }
        "sampling.text" => {
            let values = array(field(&arguments, "values")?)?
                .iter()
                .map(|value| {
                    if value.is_null() {
                        Ok(None)
                    } else {
                        Ok(Some(text(value)?.to_owned()))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Value::Array(
                array(field(&arguments, "max_words_variants")?)?
                    .iter()
                    .map(|maximum| {
                        let maximum = if maximum.is_null() {
                            None
                        } else {
                            Some(maximum.as_i64().ok_or(Error("invalid text limit"))?)
                        };
                        Ok(json!(sdm_sampling::text::normalize_text(&values, maximum)))
                    })
                    .collect::<Result<_>>()?,
            )
        }
        "sampling.random_word_lists" => {
            let words = array(field(&arguments, "words")?)?
                .iter()
                .map(|word| {
                    word.as_u64()
                        .and_then(|word| u32::try_from(word).ok())
                        .ok_or(Error("invalid RNG word"))
                })
                .collect::<Result<Vec<_>>>()?;
            let mut state = if let [word] = words.as_slice() {
                sdm_random::RandomState::new(*word)
            } else {
                sdm_random::RandomState::from_words(&words)
                    .map_err(|_| Error("invalid RNG words"))?
            };
            random_draws(&mut state, field(&arguments, "operations")?)?
        }
        "sampling.mt19937" => {
            let seed = field(&arguments, "seed")?
                .as_u64()
                .ok_or(Error("invalid MT seed"))?;
            let mut state = sdm_random::Mt19937::new(seed);
            let mut result = Vec::new();
            for operation in array(field(&arguments, "operations")?)? {
                let operation = array(operation)?;
                match operation.first().and_then(Value::as_str) {
                    Some("u") if operation.len() == 2 => {
                        let count = count(&operation[1])?;
                        result.extend((0..count).map(|_| json!(state.next_u32())));
                    }
                    Some("f") if operation.len() == 3 => {
                        let population = integer(&operation[1])?;
                        let count = integer(&operation[2])?;
                        result.extend(
                            state
                                .floyd(population, count)
                                .map_err(|_| Error("invalid MT draw"))?
                                .into_iter()
                                .map(Value::from),
                        );
                    }
                    _ => return Err(Error("invalid MT operation")),
                }
            }
            Value::Array(result)
        }
        "labels.questions" => {
            let output = sdm_pql::validate_json(
                text(field(&arguments, "query")?)?,
                &field(&arguments, "schema")?.to_string(),
                "RELATIONAL",
                None,
            )
            .map_err(|_| Error("invalid label query"))?;
            let output: Value =
                serde_json::from_str(&output).map_err(|_| Error("invalid label query output"))?;
            json!({"resolved": output.get("tree")})
        }
        "request.payload_size" => {
            let size = |key| {
                sdm_request::size::payload_size_bytes(&field(&arguments, key)?.to_string())
                    .map_err(|_| Error("invalid payload size input"))
            };
            json!({"parsed": size("parsed")?, "original": size("payload")?})
        }
        "schema.metric_views" => {
            fields(&arguments, &["definition", "trust_expressions"])?;
            let definition = text(field(&arguments, "definition")?)?;
            let trust = field(&arguments, "trust_expressions")?
                .as_bool()
                .ok_or(Error("invalid expression trust"))?;
            serde_json::from_str(&sdm_schema::metric_view::parse_json(definition, trust))
                .map_err(|_| Error("invalid metric view output"))?
        }
        "pql.queries" | "pql.date_literals" => return pql_generator(&arguments, None),
        "request.ranked" | "request.ranked_nan" => {
            let output = crate::fixture_op("client.ranked", &arguments.to_string())
                .map_err(|_| Error("invalid ranked input"))?;
            let output: Value =
                serde_json::from_str(&output).map_err(|_| Error("invalid ranked output"))?;
            field(&output, "output")?.clone()
        }
        _ => return Err(Error("unsupported generator")),
    };
    generator_observation(result, Value::Null)
}

fn generator_observation(result: Value, error: Value) -> Result<Value> {
    typed_observation(encode_typed(result)?, error)
}

fn generator_exact(input: &exact::Value) -> Result<exact::Value> {
    let structure = input.structure();
    fields(
        &structure,
        if structure.get("host_inputs").is_some() {
            &["generator", "generator_input", "host_inputs"]
        } else {
            &["generator", "generator_input"]
        },
    )?;
    let coordinate = field(&structure, "generator")?;
    fields(coordinate, &["name", "stream_seed", "ordinal"])?;
    require(
        field(coordinate, "stream_seed")?.as_u64().is_some()
            && field(coordinate, "ordinal")?.as_u64().is_some(),
        "invalid generator coordinate",
    )?;
    let name = text(field(coordinate, "name")?)?;
    if matches!(name, "sampling.offset_pairs" | "sampling.debug_jobs") {
        require(
            structure.get("host_inputs").is_none(),
            "unsupported host witness route",
        )?;
        let arguments =
            typed::sequence_json(&typed::value(&input.field("generator_input")?.narrow()?)?)?;
        return sampling::replay_exact(name, &arguments);
    }
    if !matches!(
        name,
        "sampling.random_state" | "sampling.random_draws" | "sampling.hashseed_python_jobs"
    ) {
        if name == "rfm.sequences" {
            let input = input.narrow()?;
            return sequences::replay_exact(
                field(&input, "generator_input")?,
                input.get("host_inputs"),
            );
        }
        return generator(&json!({"replay_input": input.narrow()?})).map(exact::Value::from_json);
    }
    let typed_input = input.field("generator_input")?;
    let mut arguments = Map::new();
    let mut seed = None;
    for entry in typed_input.field("mapping")?.array()? {
        let entry = entry.array()?;
        require(entry.len() == 2, "invalid typed mapping pair")?;
        fields(&entry[0].structure(), &["scalar"])?;
        let key = entry[0].field("scalar")?.text()?;
        require(
            !arguments.contains_key(key) && !(key == "seed" && seed.is_some()),
            "duplicate typed mapping key",
        )?;
        if key == "seed" {
            let scalar = &entry[1];
            fields(&scalar.structure(), &["scalar"])?;
            seed = Some(scalar.field("scalar")?.integer()?.to_owned());
        } else {
            arguments.insert(
                key.to_owned(),
                typed::sequence_json(&typed::value(&entry[1].narrow()?)?)?,
            );
        }
    }
    if name == "sampling.hashseed_python_jobs"
        && arguments.get("kind") != Some(&json!("random_state"))
    {
        fields(&typed_input.structure(), &["mapping"])?;
        require(
            structure.get("host_inputs").is_none(),
            "unsupported host witness route",
        )?;
        return sampling::replay_exact(name, &Value::Object(arguments));
    }
    fields(&typed_input.structure(), &["mapping"])?;
    let seed = seed.ok_or(Error("missing random seed"))?;
    require(
        !seed.is_empty() && seed.bytes().all(|digit| digit.is_ascii_digit()),
        "invalid random seed",
    )?;
    let mut words = vec![0u32];
    for digit in seed.bytes() {
        let mut carry = u64::from(digit - b'0');
        for word in &mut words {
            let expanded = u64::from(*word) * 10 + carry;
            *word = expanded as u32;
            carry = expanded >> 32;
        }
        if carry != 0 {
            words.push(carry as u32);
        }
    }
    if words == [0] {
        words.clear();
    }
    let arguments = Value::Object(arguments);
    let mut utility = sdm_random::random_state(&words).map_err(|_| Error("invalid random seed"))?;
    let result = if name == "sampling.hashseed_python_jobs" {
        fields(&arguments, &["kind", "ops"])?;
        let answer = random_draws(&mut utility, field(&arguments, "ops")?)?;
        json!({"answers": [answer], "safe_python": [], "purpose": "native-hashseed-consistency"})
    } else {
        if name == "sampling.random_state" {
            fields(&arguments, &["operations", "modes", "words"])?;
            require(
                field(&arguments, "words")? == &json!(words),
                "random seed words disagree",
            )?;
        } else {
            fields(&arguments, &["operations", "modes"])?;
        }
        require(
            field(&arguments, "modes")?
                == &json!(["utils.random_state", "RandomState.integer-or-words"]),
            "invalid random modes",
        )?;
        let mut direct = if words.len() <= 1 {
            sdm_random::RandomState::new(words.first().copied().unwrap_or(0))
        } else {
            sdm_random::RandomState::from_words(&words).map_err(|_| Error("invalid random seed"))?
        };
        json!([
            random_draws(&mut utility, field(&arguments, "operations")?)?,
            random_draws(&mut direct, field(&arguments, "operations")?)?
        ])
    };
    generator_observation(result, Value::Null).map(exact::Value::from_json)
}

fn typed_observation(result: Value, error: Value) -> Result<Value> {
    Ok(json!({"family": "original-generator-v1", "observation": {
        "status": if error.is_null() { "success" } else { "refused" },
        "schema": [], "payloads": [], "sql": [],
        "result": result, "error": error, "warnings": []
    }}))
}

fn typed_observation_exact(result: exact::Value, error: Value) -> Result<exact::Value> {
    let mut observation = exact::Value::from_json(typed_observation(Value::Null, error)?);
    *observation.field_mut("observation")?.field_mut("result")? = result;
    Ok(observation)
}

fn integer(value: &Value) -> Result<i64> {
    value.as_i64().ok_or(Error("expected integer"))
}

fn count(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or(Error("expected count"))
}

fn random_draws(state: &mut sdm_random::RandomState, operations: &Value) -> Result<Value> {
    let mut outputs = Vec::new();
    for operation in array(operations)? {
        let operation = array(operation)?;
        let output = match operation.first().and_then(Value::as_str) {
            Some("u32") if operation.len() == 2 => json!(
                (0..count(&operation[1])?)
                    .map(|_| state.next_u32())
                    .collect::<Vec<_>>()
            ),
            Some("u64") if operation.len() == 2 => json!(
                (0..count(&operation[1])?)
                    .map(|_| state.next_u64())
                    .collect::<Vec<_>>()
            ),
            Some("interval") if operation.len() == 2 => {
                json!(state.interval(operation[1].as_u64().ok_or(Error("invalid RNG interval"))?))
            }
            Some("randint") if operation.len() == 4 => json!(
                state
                    .randint(
                        integer(&operation[1])?,
                        integer(&operation[2])?,
                        count(&operation[3])?
                    )
                    .map_err(|_| Error("invalid RNG range"))?
            ),
            Some("shuffle") if operation.len() == 2 => {
                let mut values = array(&operation[1])?.to_vec();
                state.shuffle(&mut values);
                Value::Array(values)
            }
            Some("permutation") if operation.len() == 2 => json!(
                state
                    .permutation(count(&operation[1])?)
                    .map_err(|_| Error("invalid RNG permutation"))?
            ),
            Some("choice") if operation.len() == 3 => json!(
                state
                    .choice(integer(&operation[1])?, count(&operation[2])?)
                    .map_err(|_| Error("invalid RNG choice"))?
            ),
            _ => return Err(Error("invalid RNG operation")),
        };
        outputs.push(output);
    }
    Ok(Value::Array(outputs))
}

fn ordered_pql(value: Value) -> Value {
    if let Some(errors) = value.get("errors") {
        json!({"warnings": value.get("warnings"), "errors": errors})
    } else {
        value
    }
}

fn pql_generator(arguments: &Value, pads_year: Option<bool>) -> Result<Value> {
    fields(arguments, &["operation", "arguments"])?;
    let given = field(arguments, "arguments")?;
    let operation = text(field(arguments, "operation")?)?;
    let output = match operation {
        "pql.date" => sdm_pql::parse_date_json_in(
            text(field(given, "literal")?)?,
            sdm_pql::dates::Host {
                pads_year: pads_year.unwrap_or(sdm_pql::dates::Host::assumed().pads_year),
                ..sdm_pql::dates::Host::assumed()
            },
        ),
        "pql.validate" => sdm_pql::validate_json(
            text(field(given, "query")?)?,
            &field(given, "schema")?.to_string(),
            text(field(given, "profile")?)?,
            object(given)?
                .get("mode")
                .filter(|value| !value.is_null())
                .map(text)
                .transpose()?,
        )
        .map_err(|_| Error("invalid PQL input"))?,
        "pql.profile" | "pql.options" => {
            let query = text(field(given, "query")?)?;
            let schema = sdm_pql::SchemaDefinition::from_json(&field(given, "schema")?.to_string())
                .map_err(|_| Error("invalid PQL schema"))?;
            let mode = sdm_pql::Mode::parse(text(field(given, "mode")?)?)
                .ok_or(Error("invalid PQL mode"))?;
            if operation == "pql.profile" {
                let variant = field(given, "capabilities")?;
                fields(variant, &["base", "set", "model"])?;
                let model = text(field(variant, "model")?)?;
                let base = match text(field(variant, "base")?)? {
                    "all" => sdm_pql::Capabilities {
                        model: model.to_owned(),
                        ..Default::default()
                    },
                    "nothing" => sdm_pql::Capabilities::nothing(model),
                    _ => return Err(Error("invalid PQL capabilities base")),
                };
                let settings = object(field(variant, "set")?)?
                    .iter()
                    .map(|(name, value)| {
                        Ok((
                            name.as_str(),
                            value.as_bool().ok_or(Error("invalid PQL capability"))?,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let capabilities = base
                    .also(&settings)
                    .map_err(|_| Error("invalid PQL capability"))?;
                sdm_pql::validate(query, &schema, &sdm_pql::Profile { mode, capabilities })
                    .to_json()
                    .render()
            } else {
                let mut validator =
                    sdm_pql::validator::predictive_query::PredictiveQueryValidator::new(
                        &schema,
                        mode,
                        sdm_pql::Capabilities::default(),
                    );
                validator.allow_array_targets = field(given, "allow_array_targets")?
                    .as_bool()
                    .ok_or(Error("invalid PQL array option"))?;
                validator.allow_timestamp_arrays = field(given, "allow_timestamp_arrays")?
                    .as_bool()
                    .ok_or(Error("invalid PQL timestamp option"))?;
                let (answer, warnings) =
                    sdm_pql::context::run(sdm_pql::dates::Host::assumed(), || {
                        let parsed =
                            sdm_pql::visitor::parse_query(query, mode).map_err(|failure| {
                                let class = if failure.code == sdm_pql::codes::VALUE_INVALID_LITERAL
                                {
                                    "builtins.TypeError"
                                } else {
                                    "builtins.ValueError"
                                };
                                (class, failure.message)
                            })?;
                        validator
                            .validate_predictive_query(parsed)
                            .map_err(|failure| ("builtins.ValueError", failure.0))
                    });
                match answer {
                    Ok((query, response)) => sdm_pql::Validation {
                        query,
                        response,
                        warnings,
                    }
                    .to_json()
                    .render(),
                    Err((class, message)) => {
                        return generator_observation(
                            Value::Null,
                            json!({
                                "class": class, "message": message, "code": null, "details": null
                            }),
                        );
                    }
                }
            }
        }
        _ => return Err(Error("unsupported PQL operation")),
    };
    let output = serde_json::from_str(&output).map_err(|_| Error("invalid PQL output"))?;
    generator_observation(ordered_pql(output), Value::Null)
}

fn dialect(backend: &str, function: &str, arguments: &Value) -> Result<Value> {
    let rendered = sdm_dialects::render_json(backend, function, &arguments.to_string())
        .map_err(|_| Error("dialect replay refused its input"))?;
    serde_json::from_str(&rendered).map_err(|_| Error("invalid dialect output"))
}

fn dialect_generator(call: &Value) -> Result<Value> {
    fields(call, &["backend", "function", "arguments"])?;
    dialect(
        text(field(call, "backend")?)?,
        text(field(call, "function")?)?,
        field(call, "arguments")?,
    )
}

enum Case {
    Generator(exact::Value),
    Semantic {
        input: Value,
        directory: String,
        dependencies: BTreeSet<String>,
    },
    Public {
        input: Value,
        directory: String,
        dependencies: BTreeSet<String>,
    },
    Dialect {
        backend: String,
        function: String,
        arguments: Value,
    },
    Fixture {
        operation: String,
        input: Value,
    },
    Pql {
        query: String,
        schema: Value,
        profile: String,
        invalid: bool,
    },
}

pub struct PreparedSnapshot {
    files: BTreeSet<String>,
    cases: BTreeMap<String, Case>,
    arrow: BTreeMap<String, Vec<u8>>,
}

impl PreparedSnapshot {
    pub fn from_json(files_json: &str) -> Result<Self> {
        let files = serde_json::from_value(unique_json(files_json.as_bytes())?)
            .map_err(|_| Error("invalid snapshot files"))?;
        Self::from_encoded_files(&files)
    }

    fn from_encoded_files(files: &BTreeMap<String, String>) -> Result<Self> {
        require(!files.is_empty(), "empty snapshot")?;
        let mut cases = BTreeMap::new();
        let mut arrow = BTreeMap::new();
        for (path, encoded) in files {
            checked_path(path)?;
            let bytes = unhex(encoded)?;
            if path.ends_with(".arrow") {
                arrow.insert(path.clone(), bytes);
            } else if path.starts_with("tests/parity/frozen-public/")
                && path.ends_with(".json")
                && !path.ends_with("/manifest.json")
            {
                let document = unique_json(&bytes)?;
                fields(&document, &["format", "cases"])?;
                require(
                    text(field(&document, "format")?)? == "p97-public-v2",
                    "unsupported public format",
                )?;
                let (directory, _) = path.rsplit_once('/').ok_or(Error("invalid public path"))?;
                for case in array(field(&document, "cases")?)? {
                    let identity = text(field(case, "id")?)?;
                    require(
                        !identity.is_empty() && !cases.contains_key(identity),
                        "duplicate public identity",
                    )?;
                    let mut seen = BTreeSet::new();
                    for dependency in array(field(case, "depends_on")?)? {
                        let dependency = text(dependency)?;
                        require(
                            files.contains_key(dependency)
                                && dependency != path
                                && seen.insert(dependency),
                            "invalid public dependency",
                        )?;
                    }
                    cases.insert(
                        identity.into(),
                        Case::Public {
                            input: field(case, "replay_input")?.clone(),
                            directory: directory.into(),
                            dependencies: seen.into_iter().map(str::to_owned).collect(),
                        },
                    );
                }
            } else if path.starts_with("tests/parity/frozen-generators/") && path.ends_with(".json")
            {
                let exact_document = exact::Value::parse(&bytes)?;
                let document = exact_document.structure();
                fields(&document, &["format", "cases"])?;
                require(
                    text(field(&document, "format")?)? == "p97-generators-v1",
                    "unsupported generator format",
                )?;
                for (case, exact_case) in array(field(&document, "cases")?)?
                    .iter()
                    .zip(exact_document.field("cases")?.array()?)
                {
                    let identity = text(field(case, "id")?)?;
                    require(
                        !identity.is_empty() && !cases.contains_key(identity),
                        "duplicate or empty snapshot case",
                    )?;
                    let dependencies = array(field(case, "depends_on")?)?;
                    let mut seen = BTreeSet::new();
                    for dependency in dependencies {
                        let dependency = text(dependency)?;
                        require(
                            files.contains_key(dependency)
                                && dependency != path
                                && seen.insert(dependency),
                            "invalid snapshot dependency",
                        )?;
                    }
                    let input = exact_case.field("replay_input")?;
                    let replay = if input.field("generator")?.field("name")?.narrow()?
                        == "schema.semantic_views"
                    {
                        let (directory, _) = path
                            .rsplit_once('/')
                            .ok_or(Error("invalid semantic generator path"))?;
                        Case::Semantic {
                            input: input.narrow()?,
                            directory: directory.into(),
                            dependencies: seen.into_iter().map(str::to_owned).collect(),
                        }
                    } else {
                        Case::Generator(input.clone())
                    };
                    cases.insert(identity.to_owned(), replay);
                }
            } else if path.starts_with("tests/parity/dialects/") && path.ends_with(".json") {
                let document = unique_json(&bytes)?;
                fields(&document, &["format", "backend", "calls"])?;
                require(
                    field(&document, "format")? == &json!(1),
                    "unsupported dialect format",
                )?;
                let backend = text(field(&document, "backend")?)?;
                for (ordinal, call) in array(field(&document, "calls")?)?.iter().enumerate() {
                    let identity = format!("{path}::call-{ordinal}");
                    require(!cases.contains_key(&identity), "duplicate snapshot case")?;
                    cases.insert(
                        identity,
                        Case::Dialect {
                            backend: backend.to_owned(),
                            function: text(field(call, "function")?)?.to_owned(),
                            arguments: field(call, "args")?.clone(),
                        },
                    );
                }
            } else if path.starts_with("tests/parity/fixtures/") && path.ends_with(".json") {
                let document = unique_json(&bytes)?;
                fields(&document, &["source", "cases"])?;
                for case in array(field(&document, "cases")?)? {
                    let identity = format!("{path}::{}", text(field(case, "id")?)?);
                    require(!cases.contains_key(&identity), "duplicate snapshot case")?;
                    cases.insert(
                        identity,
                        Case::Fixture {
                            operation: text(field(case, "operation")?)?.to_owned(),
                            input: field(case, "input")?.clone(),
                        },
                    );
                }
            } else if matches!(
                path.as_str(),
                "tests/parity/pql/valid.json" | "tests/parity/pql/invalid.json"
            ) {
                let document = unique_json(&bytes)?;
                fields(&document, &["cases"])?;
                for case in array(field(&document, "cases")?)? {
                    let identity = format!("{path}::{}", text(field(case, "id")?)?);
                    require(!cases.contains_key(&identity), "duplicate snapshot case")?;
                    cases.insert(
                        identity,
                        Case::Pql {
                            query: text(field(case, "query")?)?.to_owned(),
                            schema: field(case, "schema")?.clone(),
                            profile: text(field(case, "profile")?)?.to_owned(),
                            invalid: path.ends_with("/invalid.json"),
                        },
                    );
                }
            }
        }
        Ok(Self {
            files: files.keys().cloned().collect(),
            cases,
            arrow,
        })
    }

    pub fn replay_case(&self, identity: &str) -> Result<Value> {
        let case = self
            .cases
            .get(identity)
            .ok_or(Error("snapshot case is missing or unsupported"))?;
        match case {
            Case::Generator(input) => generator_exact(input)?.narrow(),
            Case::Semantic {
                input,
                directory,
                dependencies,
            } => {
                let blobs = self
                    .arrow
                    .iter()
                    .filter(|(name, _)| dependencies.contains(*name))
                    .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
                    .collect();
                semantic::replay(input, &blobs, directory)
            }
            Case::Public {
                input,
                directory,
                dependencies,
            } => {
                let blobs = self
                    .arrow
                    .iter()
                    .filter(|(name, _)| dependencies.contains(*name))
                    .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
                    .collect();
                let output = crate::public_replay::replay(input, &blobs, directory)
                    .map_err(|_| Error("public replay refused its input"))?;
                Ok(json!({"family": "public-observation-v1", "observation": output}))
            }
            Case::Dialect {
                backend,
                function,
                arguments,
            } => Ok(
                json!({"family": "dialect-call-v1", "observation": dialect(backend, function, arguments)?}),
            ),
            Case::Fixture { operation, input } => {
                let input = input.to_string();
                let output = if operation.starts_with("labels.") {
                    sdm_labels::fixture_op(operation, &input)
                        .map_err(|_| Error("invalid labels fixture input"))?
                } else if operation.starts_with("sampling.") {
                    sdm_sampling::fixture_op(operation, &input)
                        .map_err(|_| Error("invalid sampling fixture input"))?
                } else if operation.starts_with("dates.")
                    || operation.starts_with("composite.")
                    || operation == "time_column.choose"
                {
                    sdm_schema::fixture_op(operation, &input)
                        .map_err(|_| Error("invalid schema fixture input"))?
                } else {
                    crate::fixture_op(operation, &input)
                        .map_err(|_| Error("unsupported fixture operation"))?
                };
                let output: Value =
                    serde_json::from_str(&output).map_err(|_| Error("invalid fixture output"))?;
                Ok(json!({"family": "fixture-operation-v1", "observation": {
                    "operation": operation, "result": output.get("output"), "error": output.get("error"),
                    "warnings": output.get("warnings").cloned().unwrap_or_else(|| json!([]))
                }}))
            }
            Case::Pql {
                query,
                schema,
                profile,
                invalid,
            } => {
                let mut output: Value = serde_json::from_str(
                    &sdm_pql::validate_json(query, &schema.to_string(), profile, None)
                        .map_err(|_| Error("invalid PQL input"))?,
                )
                .map_err(|_| Error("invalid PQL output"))?;
                if *invalid {
                    output
                        .as_object_mut()
                        .ok_or(Error("invalid PQL output"))?
                        .remove("warnings");
                }
                Ok(json!({"family": "pql-compiler-v1", "observation": output}))
            }
        }
    }

    fn replay_exact(&self, identity: &str) -> Result<exact::Value> {
        match self.cases.get(identity) {
            Some(Case::Generator(input)) => generator_exact(input),
            _ => self.replay_case(identity).map(exact::Value::from_json),
        }
    }

    pub fn replay_case_json(&self, identity: &str) -> Result<String> {
        self.replay_exact(identity).map(|value| value.to_json())
    }
}

pub fn replay_snapshot(request_json: &str) -> Result<Value> {
    replay_snapshot_exact(request_json)?.narrow()
}

pub fn replay_snapshot_json(request_json: &str) -> Result<String> {
    replay_snapshot_exact(request_json).map(|value| value.to_json())
}

fn replay_snapshot_exact(request_json: &str) -> Result<exact::Value> {
    let request: Request = serde_json::from_value(unique_json(request_json.as_bytes())?)
        .map_err(|_| Error("invalid snapshot request"))?;
    require(!request.cases.is_empty(), "empty snapshot")?;
    if let Some(digest) = &request.native_sha256 {
        require(
            digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid native binding digest",
        )?;
    }
    let selected: BTreeSet<_> = request.cases.iter().collect();
    require(
        selected.len() == request.cases.len() && selected == request.routes.keys().collect(),
        "invalid snapshot case routes",
    )?;
    let snapshot = PreparedSnapshot::from_encoded_files(&request.files)?;
    let mut observations = Vec::new();
    for identity in &request.cases {
        let route = &request.routes[identity];
        require(
            route.kind == "native_semantic" && route.native_functions == ["replay_snapshot"],
            "unsupported snapshot route",
        )?;
        observations.push((identity.clone(), snapshot.replay_exact(identity)?));
    }
    Ok(exact::Value::Object(vec![
        ("observations".into(), exact::Value::Object(observations)),
        ("host_dispositions".into(), exact::Value::Object(Vec::new())),
        (
            "files_verified".into(),
            exact::Value::from_json(json!(snapshot.files)),
        ),
    ]))
}

struct Unique(Value);

impl<'de> Deserialize<'de> for Unique {
    fn deserialize<Deserializer: serde::Deserializer<'de>>(
        deserializer: Deserializer,
    ) -> std::result::Result<Self, Deserializer::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Unique;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("JSON without duplicate keys")
            }
            fn visit_unit<Error: serde::de::Error>(self) -> std::result::Result<Unique, Error> {
                Ok(Unique(Value::Null))
            }
            fn visit_bool<Error: serde::de::Error>(
                self,
                value: bool,
            ) -> std::result::Result<Unique, Error> {
                Ok(Unique(value.into()))
            }
            fn visit_i64<Error: serde::de::Error>(
                self,
                value: i64,
            ) -> std::result::Result<Unique, Error> {
                Ok(Unique(value.into()))
            }
            fn visit_u64<Error: serde::de::Error>(
                self,
                value: u64,
            ) -> std::result::Result<Unique, Error> {
                Ok(Unique(value.into()))
            }
            fn visit_f64<Error: serde::de::Error>(
                self,
                value: f64,
            ) -> std::result::Result<Unique, Error> {
                serde_json::Number::from_f64(value)
                    .map(|number| Unique(Value::Number(number)))
                    .ok_or_else(|| Error::custom("invalid number"))
            }
            fn visit_str<Error: serde::de::Error>(
                self,
                value: &str,
            ) -> std::result::Result<Unique, Error> {
                Ok(Unique(value.into()))
            }
            fn visit_seq<Access: serde::de::SeqAccess<'de>>(
                self,
                mut access: Access,
            ) -> std::result::Result<Unique, Access::Error> {
                let mut values = Vec::new();
                while let Some(Unique(value)) = access.next_element()? {
                    values.push(value);
                }
                Ok(Unique(Value::Array(values)))
            }
            fn visit_map<Access: serde::de::MapAccess<'de>>(
                self,
                mut access: Access,
            ) -> std::result::Result<Unique, Access::Error> {
                let mut values = Map::new();
                while let Some((key, Unique(value))) = access.next_entry::<String, Unique>()? {
                    if values.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate key"));
                    }
                }
                Ok(Unique(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

fn unique_json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice::<Unique>(bytes)
        .map(|value| value.0)
        .map_err(|_| Error("invalid or duplicate JSON"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(seed: u64) -> Value {
        json!({"id": "generated", "depends_on": [], "expected": "not an answer", "replay_input": {
            "generator": {"name": "dialects.random_state", "stream_seed": 20261004, "ordinal": 0},
            "generator_input": encode_typed(json!({"calls": [{
                "backend": "databricks", "function": "utils.random_state.randint",
                "arguments": {"random_seed": seed, "low": 0, "high": 2147483647, "size": 1}
            }]})).unwrap()
        }})
    }

    fn request(cases: Vec<Value>) -> Value {
        let bytes = json!({"format": "p97-generators-v1", "cases": cases}).to_string();
        let encoded: String = bytes.bytes().map(|byte| format!("{byte:02x}")).collect();
        json!({"files": {"tests/parity/frozen-generators/example.json": encoded},
            "cases": ["generated"], "routes": {"generated": {"kind": "native_semantic", "native_functions": ["replay_snapshot"]}}})
    }

    #[test]
    fn label_cast_warning_survives_snapshot_without_expected_feedback() {
        let mut input = case(0);
        input["replay_input"]["generator"]["name"] = json!("labels.aggregation_type");
        input["replay_input"]["generator_input"] = encode_typed(json!({
            "op": "MAX", "feat": {"dtype": "float64", "values": [1e100]},
            "batch": [0], "batch_size": 1, "filter_na": false,
        }))
        .unwrap();
        let output = generator(&input).unwrap();
        assert_eq!(
            output["observation"]["warnings"],
            json!([{
                "class": "builtins.RuntimeWarning", "code": null,
                "message": "overflow encountered in cast", "details": null,
            }])
        );
        let original = replay_snapshot(&request(vec![input.clone()]).to_string()).unwrap();
        input["expected"] = json!({"warnings": [], "result": "incorrect"});
        assert_eq!(
            original,
            replay_snapshot(&request(vec![input]).to_string()).unwrap()
        );
    }

    #[test]
    fn date_snapshot_uses_explicit_runtime_year_format_capability() {
        let mut input = case(0);
        input["replay_input"]["generator"]["name"] = json!("pql.date_literals");
        input["replay_input"]["generator_input"] = encode_typed(json!({
            "operation": "pql.date", "arguments": {"literal": "30/8/0953"},
        }))
        .unwrap();
        let original_input = input["replay_input"]["generator_input"].clone();
        let without = generator(&input).unwrap();
        input["replay_input"]["host_inputs"] = json!({"strftime_year_953": "0953"});
        let observed = generator(&input).unwrap();
        let result = decode_typed(&observed["observation"]["result"]).unwrap();
        assert_eq!(result["value"], "0953-08-30T00:00:00");
        assert_eq!(
            result["warnings"],
            json!([{
                "category": "builtins.UserWarning",
                "message": "Parsing dates in %d/%m/%Y format when dayfirst=False (the default) was specified. Pass `dayfirst=True` or specify a format to silence this warning.",
            }])
        );
        input["expected"] = json!("wrong outcome");
        assert_eq!(generator(&input).unwrap(), observed);
        assert_eq!(input["replay_input"]["generator_input"], original_input);
        input["replay_input"]["host_inputs"] = json!({"strftime_year_953": "953"});
        assert_eq!(generator(&input).unwrap(), without);
        for invalid in [
            json!({"strftime_year_953": true}),
            json!({"strftime_year_953": "0953", "expected": true}),
        ] {
            input["replay_input"]["host_inputs"] = invalid;
            assert!(generator(&input).is_err());
        }
        input["replay_input"]["host_inputs"] = json!({"strftime_year_953": "0953"});
        input["replay_input"]["generator"]["name"] = json!("pql.queries");
        assert!(generator(&input).is_err());
    }

    #[test]
    fn nullable_cast_snapshot_suppresses_only_masked_runtime_warnings() {
        let mut input = case(0);
        input["replay_input"]["generator"]["name"] = json!("labels.aggregation_type");
        input["replay_input"]["generator_input"] = encode_typed(json!({
            "op": "MAX", "feat": {"dtype": "Float64", "values": [1e100, {"$na": null}]},
            "batch": [0, 0], "batch_size": 1, "filter_na": false,
        }))
        .unwrap();
        let output = generator(&input).unwrap();
        assert_eq!(output["observation"]["status"], "success");
        assert_eq!(output["observation"]["warnings"], json!([]));
        let original = replay_snapshot(&request(vec![input.clone()]).to_string()).unwrap();
        input["expected"] = json!({"result": "incorrect", "warnings": ["incorrect"]});
        assert_eq!(
            original,
            replay_snapshot(&request(vec![input]).to_string()).unwrap()
        );
    }

    #[test]
    fn generator_adapters_compute_typed_inputs_and_ignore_expected_fields() {
        let inputs = [
            (
                "sampling.text",
                json!({"values": ["Hello WORLD", null], "max_words_variants": [1, null]}),
            ),
            (
                "sampling.random_word_lists",
                json!({"words": [42], "operations": [["u32", 2], ["permutation", 3]]}),
            ),
            (
                "sampling.mt19937",
                json!({"seed": 42, "operations": [["u", 2], ["f", 5, 2]]}),
            ),
            (
                "request.payload_size",
                json!({"payload": {"text": "é"}, "text": "unused spelling", "parsed": {"text": "é"}}),
            ),
        ];
        for (name, arguments) in inputs {
            let mut input = case(0);
            input["replay_input"]["generator"]["name"] = json!(name);
            input["replay_input"]["generator_input"] = encode_typed(arguments).unwrap();
            let output = generator(&input).unwrap();
            assert_eq!(output["observation"]["status"], "success");
            input["expected"] = json!({"result": "not computed"});
            assert_eq!(generator(&input).unwrap(), output);
        }
    }

    #[test]
    fn semantic_snapshot_uses_original_source_inputs_and_preserves_tuple_results() {
        let source = json!({
            "backend": "snow", "namespace": {"database": "TEST", "schema": "PARITY_SHOP"},
            "semantic_model": "SV", "trust_expressions": false,
        });
        let options = json!({"tables": null, "primary_keys": null, "time_columns": null,
            "relationships": null, "check_keys": true});
        let mut semantic = case(0);
        semantic["replay_input"] = json!({
            "generator": {"name": "schema.semantic_views", "stream_seed": 20261009, "ordinal": 0},
            "generator_input": encode_typed(json!({"definition": "{}", "tables": [], "trust_expressions": false})).unwrap(),
            "host_inputs": {
                "host": {"year": 2026, "today": [2026, 10, 6], "pads_year": true,
                    "tznames": ["UTC"], "timestamp_series_units": ["s", "ms", "us", "ns"]},
                "external": {
                    "initial_schema": {"function": "schema_source_run", "args": ["snow", source.to_string(), options.to_string()]},
                    "queries": [{"sql": "SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)", "params": ["TEST.PARITY_SHOP.SV"],
                        "outcome": {"error": {"message": "source unavailable"}}}],
                },
            },
        });
        let mut original = semantic["replay_input"]["generator_input"].clone();
        original.sort_all_objects();
        use sha2::Digest;
        semantic["replay_input"]["host_inputs"]["external"]["original_input_sha256"] =
            json!(format!(
                "{:x}",
                sha2::Sha256::digest(serde_json::to_vec(&original).unwrap())
            ));
        let output = replay_snapshot(&request(vec![semantic.clone()]).to_string()).unwrap();
        let result = &output["observations"]["generated"]["observation"]["result"];
        assert!(result["tuple"][0]["mapping"].is_array());
        assert!(result["tuple"][1]["list"][0]["tuple"].is_array());
        semantic["expected"] = json!({"schema": "forged expectation"});
        assert_eq!(
            output,
            replay_snapshot(&request(vec![semantic.clone()]).to_string()).unwrap()
        );
        semantic["replay_input"]["host_inputs"]["schema"] = json!({});
        assert!(replay_snapshot(&request(vec![semantic.clone()]).to_string()).is_err());
        semantic["replay_input"]["host_inputs"]
            .as_object_mut()
            .unwrap()
            .remove("schema");
        semantic["replay_input"]["generator_input"] =
            encode_typed(json!({"definition": "{}", "tables": [], "trust_expressions": true}))
                .unwrap();
        assert!(replay_snapshot(&request(vec![semantic]).to_string()).is_err());
    }

    #[test]
    fn custom_pql_options_report_native_parser_failure() {
        let arguments = json!({"operation": "pql.options", "arguments": {
            "query": "wrong", "schema": {"tables": {}, "col_groups": []}, "mode": "INFERENCE",
            "allow_array_targets": false, "allow_timestamp_arrays": true
        }});
        let output = pql_generator(&arguments, None).unwrap();
        assert_eq!(output["observation"]["status"], "refused");
        assert_eq!(
            output["observation"]["error"]["class"],
            "builtins.ValueError"
        );
        assert!(
            output["observation"]["error"]["message"]
                .as_str()
                .unwrap()
                .contains("PREDICT")
        );
    }

    #[test]
    fn custom_pql_options_preserve_invalid_literal_exception_kind() {
        let arguments = json!({"operation": "pql.options", "arguments": {
            "query": "PREDICT users.when > 99/4/2013 FOR EACH users.id",
            "schema": {"tables": {}, "col_groups": []}, "mode": "INFERENCE",
            "allow_array_targets": false, "allow_timestamp_arrays": true
        }});
        let observed = pql_generator(&arguments, None).unwrap();
        assert_eq!(observed["observation"]["status"], "refused");
        assert_eq!(
            observed["observation"]["error"],
            json!({
                "class": "builtins.TypeError", "code": null, "details": null,
                "message": "99/4/2013 is not a valid instance of time",
            })
        );
    }

    #[test]
    fn computes_inputs_without_consulting_expectations() {
        let original = replay_snapshot(&request(vec![case(0)]).to_string()).unwrap();
        assert_eq!(
            original["observations"]["generated"]["observation"]["result"],
            json!({"list": [{"mapping": [[{"scalar": "result"}, {"list": [{"scalar": 209652396}]}]]}]})
        );
        let mut changed = case(0);
        changed["expected"] = json!({"status": "refused", "result": "broken expectation"});
        assert_eq!(
            original,
            replay_snapshot(&request(vec![changed]).to_string()).unwrap()
        );
        assert_ne!(
            original,
            replay_snapshot(&request(vec![case(1)]).to_string()).unwrap()
        );
    }

    #[test]
    fn exact_random_keys_have_only_one_typed_tag() {
        let mut input = json!({
            "generator": {"name": "sampling.random_draws", "stream_seed": 1, "ordinal": 0},
            "generator_input": {"mapping": [
                [{"scalar": "seed"}, {"scalar": 7}],
                [{"scalar": "operations"}, {"list": []}],
                [{"scalar": "modes"}, {"list": [{"scalar": "utils.random_state"}, {"scalar": "RandomState.integer-or-words"}]}],
            ]},
        });
        assert!(generator_exact(&exact::Value::from_json(input.clone())).is_ok());
        input["generator_input"]["mapping"][0][0]["unknown_tag"] = json!(true);
        assert!(generator_exact(&exact::Value::from_json(input)).is_err());
    }

    #[test]
    fn rejects_missing_duplicate_and_unclosed_cases() {
        assert!(replay_snapshot(&request(vec![case(0), case(1)]).to_string()).is_err());
        let mut missing = request(vec![case(0)]);
        missing["cases"] = json!(["missing"]);
        assert!(replay_snapshot(&missing.to_string()).is_err());
        let mut unclosed = case(0);
        unclosed["depends_on"] = json!(["missing.arrow"]);
        assert!(replay_snapshot(&request(vec![unclosed]).to_string()).is_err());
        let mut traversal = request(vec![case(0)]);
        traversal["files"]["../outside"] = json!("00");
        assert!(replay_snapshot(&traversal.to_string()).is_err());
        let mut route = request(vec![case(0)]);
        route["routes"]["generated"]["kind"] = json!("public_host_refusal");
        assert!(replay_snapshot(&route.to_string()).is_err());
    }

    #[test]
    fn rejects_ambiguous_or_unsupported_encodings() {
        assert!(unique_json(br#"{"key": 1, "key": 2}"#).is_err());
        assert!(decode_typed(&json!({"scalar": 1.5})).is_err());
        assert!(decode_typed(&json!({"tuple": []})).is_err());
        assert!(decode_typed(&json!({"mapping": [[{"scalar": "key"}, {"scalar": 1}], [{"scalar": "key"}, {"scalar": 2}]]})).is_err());
        assert!(unhex("abc").is_err());
        assert!(unhex("zz").is_err());
    }

    #[test]
    fn typed_float_preserves_bits_at_exponent_boundaries() {
        for number in [
            0.0,
            -0.0,
            0.1,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            f64::MAX,
            -f64::MAX,
        ] {
            let encoded = encode_typed(json!(number)).unwrap();
            let decoded = decode_typed(&encoded).unwrap();
            assert_eq!(decoded.as_f64().unwrap().to_bits(), number.to_bits());
        }
        assert_eq!(float_hex(0.1), "0x1.999999999999ap-4");
        for encoded in ["0x1.0p+1024", "0x1.0p-1074", "0x1.0p+0", "invalid"] {
            assert!(decode_typed(&json!({"float": encoded})).is_err());
        }
    }
}
