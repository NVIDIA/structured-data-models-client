// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use arrow_schema::{DataType, TimeUnit};
use sdm_labels::date_offset::{Bound, DateOffset, OffsetError, pooled_offset};
use sdm_sampling::graph_store::{GraphSpec, KeyValue};
use sdm_sampling::neighbor::{NeighborSampler, SampleError, Seed, TimeRanges};
use serde_json::{Map, Value, json};
use std::io::Cursor;

use super::exact::Value as Exact;
use super::{
    Error, Result, array, count, field, generator_observation, integer, object, random_draws,
    require, text,
};

pub(super) fn replay_exact(name: &str, arguments: &Value) -> Result<Exact> {
    let result = match name {
        "sampling.offset_pairs" => {
            super::fields(
                arguments,
                &[
                    "offset",
                    "sign",
                    "repeat",
                    "timestamps",
                    "seconds_bounds",
                    "nanoseconds",
                ],
            )?;
            offset_pairs(arguments)?
        }
        "sampling.debug_jobs" => {
            super::fields(arguments, &["calls"])?;
            let answers = array(field(arguments, "calls")?)?
                .iter()
                .map(debug_job)
                .collect::<Result<_>>()?;
            Exact::Object(vec![
                ("answers".into(), Exact::Array(answers)),
                (
                    "purpose".into(),
                    Exact::String("native-debug-release-consistency".into()),
                ),
            ])
        }
        "sampling.hashseed_python_jobs" => Exact::Object(vec![
            ("answers".into(), Exact::Array(vec![python_job(arguments)?])),
            ("safe_python".into(), Exact::Array(Vec::new())),
            (
                "purpose".into(),
                Exact::String("native-hashseed-consistency".into()),
            ),
        ]),
        _ => return Err(Error("unsupported exact sampling generator")),
    };
    super::typed_observation_exact(result.typed()?, Value::Null)
}

pub(super) fn replay(name: &str, arguments: &Value, facts: Option<&Value>) -> Result<Value> {
    require(
        facts.is_none()
            || (name == "sampling.hashseed_array_jobs" && field(arguments, "kind")? == "execute"),
        "unsupported sampling host facts",
    )?;
    if matches!(
        name,
        "sampling.offset_pairs" | "sampling.debug_jobs" | "sampling.hashseed_python_jobs"
    ) {
        return replay_exact(name, arguments)?.narrow();
    }
    match name {
        "sampling.random_state" => {
            super::fields(arguments, &["seed", "operations", "modes", "words"])
        }
        "sampling.random_draws" => super::fields(arguments, &["seed", "operations", "modes"]),
        "sampling.graphs" => super::fields(
            arguments,
            &[
                "node_types",
                "edge_types",
                "colptr",
                "row",
                "node_time",
                "seed",
                "calls",
            ],
        ),
        "sampling.graph_store" => super::fields(arguments, &["graph_json", "tables"]),
        "sampling.offset_pairs" => super::fields(
            arguments,
            &[
                "offset",
                "sign",
                "repeat",
                "timestamps",
                "seconds_bounds",
                "nanoseconds",
            ],
        ),
        "sampling.offset_pooling" => super::fields(arguments, &["offsets", "earliest_variants"]),
        "sampling.debug_jobs" => super::fields(arguments, &["calls"]),
        _ => Ok(()),
    }?;
    let result = match name {
        "sampling.random_state" | "sampling.random_draws" => {
            let words = seed_words(field(arguments, "seed")?)?;
            require(
                field(arguments, "modes")?
                    == &json!(["utils.random_state", "RandomState.integer-or-words"]),
                "invalid random modes",
            )?;
            if name == "sampling.random_state" {
                require(
                    field(arguments, "words")? == &json!(words),
                    "random seed words disagree",
                )?;
            }
            let mut utility =
                sdm_random::random_state(&words).map_err(|_| Error("invalid random seed"))?;
            let mut direct = if words.len() <= 1 {
                sdm_random::RandomState::new(words.first().copied().unwrap_or(0))
            } else {
                sdm_random::RandomState::from_words(&words)
                    .map_err(|_| Error("invalid random seed"))?
            };
            let operations = field(arguments, "operations")?;
            json!([
                random_draws(&mut utility, operations)?,
                random_draws(&mut direct, operations)?
            ])
        }
        "sampling.graphs" => graph(arguments)?,
        "sampling.graph_store" => store(arguments)?,
        "sampling.graph_store_calls" => {
            let input = field(arguments, "store")?;
            let stored = store(input)?;
            let sampled = if stored.get("error").is_some() {
                Value::Null
            } else {
                sample_stored(input, &stored, field(arguments, "sampler")?)?
            };
            json!({"store": stored, "sampled": sampled})
        }
        "sampling.hashseed_array_jobs" => match text(field(arguments, "kind")?)? {
            "store" => store(arguments)?,
            "execute" => {
                let query: Value = serde_json::from_str(text(field(arguments, "query_json")?)?)
                    .map_err(|_| Error("invalid labels query"))?;
                let answer = super::labels::execute(&query, arguments, facts)?;
                return super::typed_observation(
                    json!({"mapping": [
                        [{"scalar": "answers"}, {"list": [answer]}],
                        [{"scalar": "safe_python"}, {"list": []}],
                        [{"scalar": "purpose"}, {"scalar": "native-hashseed-consistency"}],
                    ]}),
                    Value::Null,
                );
            }
            _ => return Err(Error("invalid sampling array job")),
        },
        "sampling.offset_pooling" => {
            let offsets = offsets(field(arguments, "offsets")?)?;
            Value::Array(
                array(field(arguments, "earliest_variants")?)?
                    .iter()
                    .map(|earliest| {
                        let earliest = earliest.as_bool().ok_or(Error("invalid offset bound"))?;
                        Ok(match pooled_offset(&offsets, earliest) {
                            Ok(offset) => sorted_offset(offset),
                            Err(error) => offset_error(error),
                        })
                    })
                    .collect::<Result<_>>()?,
            )
        }
        _ => return Err(Error("unsupported sampling generator")),
    };
    let result = if name == "sampling.debug_jobs" {
        json!({"answers": result, "purpose": "native-debug-release-consistency"})
    } else if name.starts_with("sampling.hashseed_") {
        json!({"answers": [result], "safe_python": [], "purpose": "native-hashseed-consistency"})
    } else {
        result
    };
    generator_observation(result, Value::Null)
}

fn seed_words(seed: &Value) -> Result<Vec<u32>> {
    let decimal = seed.to_string();
    require(
        !decimal.is_empty() && decimal.bytes().all(|digit| digit.is_ascii_digit()),
        "invalid random seed",
    )?;
    let mut words = vec![0u32];
    for digit in decimal.bytes() {
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
    Ok(words)
}

fn sampler_seed(value: &Value) -> Result<Seed> {
    if let Some(seed) = value
        .as_i64()
        .filter(|seed| (-2147483648..=4294967295).contains(seed))
    {
        return Ok(Seed::Word(seed));
    }
    Ok(Seed::Words(seed_words(value)?))
}

fn integers(value: &Value) -> Result<Vec<i64>> {
    array(value)?.iter().map(integer).collect()
}

fn integer_map(value: &Value) -> Result<Vec<(String, Vec<i64>)>> {
    object(value)?
        .iter()
        .map(|(name, values)| Ok((name.clone(), integers(values)?)))
        .collect()
}

fn sample_error(error: SampleError) -> Value {
    let code = match &error {
        SampleError::Runtime(_) => "RuntimeError",
        SampleError::Key(_) => "KeyError",
        SampleError::Index(_) => "IndexError",
    };
    json!({"error": {"code": code, "message": error.to_string()}})
}

fn ranges(value: &Value) -> Result<Vec<(String, TimeRanges)>> {
    object(value)?
        .iter()
        .map(|(name, values)| {
            let rows = array(values)?;
            let width = rows
                .first()
                .map(array)
                .transpose()?
                .map_or(0, <[Value]>::len);
            let ranges = rows
                .iter()
                .map(|row| {
                    let row = array(row)?;
                    require(row.len() == width, "ragged sampling time ranges")?;
                    if width != 2 {
                        return Ok(Ok((None, 0)));
                    }
                    let cast = |value: &Value| -> std::result::Result<i64, String> {
                        value.as_i64().ok_or_else(|| {
                            let kind = if value.is_null() {
                                "NoneType"
                            } else if value.is_f64() {
                                "float"
                            } else if value.is_string() {
                                "str"
                            } else {
                                "int"
                            };
                            sdm_sampling::neighbor::cast_error(&format!("<class '{kind}'>"))
                        })
                    };
                    let start = if row[0].is_null() {
                        Ok(None)
                    } else {
                        cast(&row[0]).map(Some)
                    };
                    Ok(start.and_then(|start| cast(&row[1]).map(|end| (start, end))))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((name.clone(), TimeRanges { width, ranges }))
        })
        .collect()
}

fn graph(arguments: &Value) -> Result<Value> {
    let nodes = array(field(arguments, "node_types")?)?
        .iter()
        .map(|node| text(node).map(str::to_owned))
        .collect::<Result<Vec<_>>>()?;
    let edges = array(field(arguments, "edge_types")?)?
        .iter()
        .map(|edge| {
            let edge = array(edge)?;
            require(edge.len() == 3, "invalid sampling edge")?;
            Ok((
                text(&edge[0])?.to_owned(),
                text(&edge[1])?.to_owned(),
                text(&edge[2])?.to_owned(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut sampler = match NeighborSampler::new(
        nodes,
        edges,
        integer_map(field(arguments, "colptr")?)?,
        integer_map(field(arguments, "row")?)?,
        integer_map(field(arguments, "node_time")?)?,
        &sampler_seed(field(arguments, "seed")?)?,
    ) {
        Ok(sampler) => sampler,
        Err(error) => return Ok(json!([sample_error(error)])),
    };
    let mut outputs = Vec::new();
    for call in array(field(arguments, "calls")?)? {
        let reseed = field(call, "reseed")?;
        if !reseed.is_null() {
            sampler.seed(&sampler_seed(reseed)?);
        }
        match sampler.sample(
            &integer_map(field(call, "num_neighbors")?)?,
            &ranges(field(call, "offsets")?)?,
            text(field(call, "entity")?)?,
            &integers(field(call, "seed_node")?)?,
            &integers(field(call, "seed_time")?)?,
        ) {
            Ok(sampled) => outputs.push(Value::Array(
                [
                    sampled.row,
                    sampled.col,
                    sampled.node_id,
                    sampled.batch,
                    sampled.num_sampled_nodes,
                    sampled.num_sampled_edges,
                ]
                .into_iter()
                .map(|part| {
                    Value::Array(
                        part.into_iter()
                            .map(|(name, values)| json!([name, "int64", values]))
                            .collect(),
                    )
                })
                .collect(),
            )),
            Err(error) => outputs.push(sample_error(error)),
        }
    }
    Ok(Value::Array(outputs))
}

fn sample_stored(input: &Value, stored: &Value, request: &Value) -> Result<Value> {
    let specification = GraphSpec::from_json(text(field(input, "graph_json")?)?)
        .map_err(|_| Error("invalid graph input"))?;
    let nodes: Vec<_> = specification
        .tables
        .into_iter()
        .map(|table| table.name)
        .collect();
    let edges: Vec<_> = specification
        .edges
        .into_iter()
        .flat_map(|edge| {
            let edge = (edge.src_table, edge.fkey, edge.dst_table);
            let reverse = sdm_sampling::graph_store::rev_edge_type(&edge);
            [edge, reverse]
        })
        .collect();
    graph(&json!({
        "node_types": nodes,
        "edge_types": edges,
        "colptr": field(stored, "colptr")?,
        "row": field(stored, "row")?,
        "node_time": field(stored, "time")?,
        "seed": field(request, "seed")?,
        "calls": field(request, "calls")?,
    }))
}

fn offset(value: &Value) -> Result<DateOffset> {
    DateOffset::from_json(value).map_err(|_| Error("invalid date offset"))
}

fn offsets(value: &Value) -> Result<Vec<DateOffset>> {
    array(value)?.iter().map(offset).collect()
}

fn offset_error(error: OffsetError) -> Value {
    json!({"error": {"code": error.code, "message": error.message}})
}

fn exact_integer(decimal: &str) -> Result<Exact> {
    let value = Exact::parse(decimal.as_bytes())?;
    require(value.integer()? == decimal, "invalid exact integer")?;
    Ok(value)
}

fn seconds(offset: &DateOffset, bound: &Value) -> Result<Exact> {
    let bound = Bound::parse(text(bound)?).ok_or(Error("invalid offset bound"))?;
    match offset.constructed().and_then(|()| offset.to_seconds(bound)) {
        Ok(seconds) => exact_integer(&seconds.to_decimal()),
        Err(error) => Ok(Exact::from_json(offset_error(error))),
    }
}

fn nanoseconds(offset: &DateOffset) -> Result<Exact> {
    match offset.constructed().and_then(|()| offset.to_ns_from_2020()) {
        Ok(value) => Ok(Exact::Object(vec![(
            "ns".into(),
            exact_integer(&value.to_string())?,
        )])),
        Err(error) => Ok(Exact::from_json(offset_error(error))),
    }
}

fn offset_pairs(arguments: &Value) -> Result<Exact> {
    let offset = offset(field(arguments, "offset")?)?;
    let sign = integer(field(arguments, "sign")?)?;
    require(matches!(sign, -1 | 1), "invalid offset sign")?;
    let mut current = integers(field(arguments, "timestamps")?)?;
    let mut alive: Vec<usize> = (0..current.len()).collect();
    let mut steps = Vec::new();
    for _ in 0..count(field(arguments, "repeat")?)? {
        let mut remaining = Vec::new();
        let mut outcomes = Vec::new();
        for index in &alive {
            match offset
                .constructed()
                .and_then(|()| offset.apply(current[*index], sign))
            {
                Ok(value) => {
                    current[*index] = value;
                    remaining.push(*index);
                    outcomes.push(json!({"ns": value}));
                }
                Err(error) => outcomes.push(offset_error(error)),
            }
        }
        steps.push(json!({"indices": alive, "outcomes": outcomes}));
        alive = remaining;
        if alive.is_empty() {
            break;
        }
    }
    let bounds = array(field(arguments, "seconds_bounds")?)?
        .iter()
        .map(|bound| seconds(&offset, bound))
        .collect::<Result<Vec<_>>>()?;
    Ok(Exact::Object(vec![
        ("steps".into(), Exact::from_json(json!(steps))),
        ("seconds".into(), Exact::Array(bounds)),
        ("nanoseconds".into(), nanoseconds(&offset)?),
    ]))
}

fn sorted_offset(offset: DateOffset) -> Value {
    let mut keywords: Vec<_> = offset
        .kwds
        .iter()
        .map(|(unit, amount)| (unit.name(), *amount))
        .collect();
    keywords.sort_by_key(|(name, _)| *name);
    let keywords: Map<String, Value> = keywords
        .into_iter()
        .map(|(name, amount)| (name.to_owned(), json!(amount)))
        .collect();
    json!({"n": offset.n, "kwds": keywords})
}

fn apply(arguments: &Value) -> Result<Value> {
    let offset = offset(field(arguments, "offset")?)?;
    let sign = integer(field(arguments, "sign")?)?;
    require(matches!(sign, -1 | 1), "invalid offset sign")?;
    Ok(Value::Array(
        integers(field(arguments, "timestamps")?)?
            .into_iter()
            .map(|timestamp| {
                match offset
                    .constructed()
                    .and_then(|()| offset.apply(timestamp, sign))
                {
                    Ok(value) => json!({"ns": value}),
                    Err(error) => offset_error(error),
                }
            })
            .collect(),
    ))
}

fn python_job(arguments: &Value) -> Result<Exact> {
    let value = match text(field(arguments, "kind")?)? {
        "random_state" => {
            let mut state = sdm_random::random_state(&seed_words(field(arguments, "seed")?)?)
                .map_err(|_| Error("invalid random seed"))?;
            random_draws(&mut state, field(arguments, "ops")?)
        }
        "mt" => {
            let job = array(field(arguments, "job")?)?;
            require(job.len() == 2, "invalid MT job")?;
            let seed = job[0].as_u64().ok_or(Error("invalid MT seed"))?;
            let mut state = sdm_random::Mt19937::new(seed);
            let mut outputs = Vec::new();
            for operation in array(&job[1])? {
                let operation = array(operation)?;
                match operation.first().and_then(Value::as_str) {
                    Some("u") if operation.len() == 2 => {
                        outputs.extend((0..count(&operation[1])?).map(|_| json!(state.next_u32())))
                    }
                    Some("f") if operation.len() == 3 => outputs.extend(
                        state
                            .floyd(integer(&operation[1])?, integer(&operation[2])?)
                            .map_err(|_| Error("invalid MT draw"))?
                            .into_iter()
                            .map(Value::from),
                    ),
                    _ => return Err(Error("invalid MT operation")),
                }
            }
            Ok(Value::Array(outputs))
        }
        "text" => {
            let values = array(field(arguments, "values")?)?
                .iter()
                .map(|value| {
                    if value.is_null() {
                        Ok(None)
                    } else {
                        text(value).map(|value| Some(value.to_owned()))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let maximum = field(arguments, "max_words")?;
            let maximum = if maximum.is_null() {
                None
            } else {
                Some(integer(maximum)?)
            };
            Ok(json!(sdm_sampling::text::normalize_text(&values, maximum)))
        }
        "offset_apply" => {
            if let Err(error) = offset(field(arguments, "offset")?)?.constructed() {
                return Ok(Exact::from_json(
                    json!({"raised": error.code, "text": error.message}),
                ));
            }
            apply(arguments)
        }
        "offset_seconds" => {
            let value = seconds(
                &offset(field(arguments, "offset")?)?,
                field(arguments, "bound")?,
            )?;
            return if let Ok(error) = value.field("error") {
                Ok(Exact::Object(vec![
                    ("raised".into(), error.field("code")?.clone()),
                    ("text".into(), error.field("message")?.clone()),
                ]))
            } else {
                Ok(value)
            };
        }
        "offset_ns" => return nanoseconds(&offset(field(arguments, "offset")?)?),
        "offset_pool" => {
            let earliest = field(arguments, "earliest")?
                .as_bool()
                .ok_or(Error("invalid offset bound"))?;
            Ok(
                match pooled_offset(&offsets(field(arguments, "offsets")?)?, earliest) {
                    Ok(value) => sorted_offset(value),
                    Err(error) => offset_error(error),
                },
            )
        }
        "sampler" => graph(field(arguments, "graph")?),
        "labels_op" | "sampling_op" => {
            let operation = text(field(arguments, "op")?)?;
            let input = field(arguments, "input")?.to_string();
            let output = if operation.starts_with("labels.") {
                sdm_labels::fixture_op(operation, &input)
            } else {
                sdm_sampling::fixture_op(operation, &input)
            }
            .map_err(|_| Error("invalid sampling fixture job"))?;
            return Exact::parse(output.as_bytes());
        }
        _ => Err(Error("unsupported sampling job")),
    }?;
    Ok(Exact::from_json(value))
}

pub(super) fn arrow_table(value: &Value) -> Result<RecordBatch> {
    super::fields(value, &["arrow_ipc_base64"])?;
    let encoded = text(field(value, "arrow_ipc_base64")?)?;
    require(encoded.len().is_multiple_of(4), "invalid Arrow base64")?;
    let body = encoded.trim_end_matches('=');
    require(
        encoded.len() - body.len() <= 2
            && body
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/')),
        "invalid Arrow base64",
    )?;
    let bytes = match sdm_schema::pyval::PyVal::decode(&json!({"$bytes": encoded}))
        .map_err(|_| Error("invalid Arrow bytes"))?
    {
        sdm_schema::pyval::PyVal::Bytes(bytes) => bytes,
        _ => return Err(Error("invalid Arrow bytes")),
    };
    let (schema, batches) = if bytes.starts_with(b"ARROW1") {
        let reader = arrow_ipc::reader::FileReader::try_new(Cursor::new(&bytes), None)
            .map_err(|_| Error("invalid Arrow file"))?;
        (
            reader.schema(),
            reader
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| Error("invalid Arrow batch"))?,
        )
    } else {
        let reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(&bytes), None)
            .map_err(|_| Error("invalid Arrow stream"))?;
        (
            reader.schema(),
            reader
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| Error("invalid Arrow batch"))?,
        )
    };
    arrow_select::concat::concat_batches(&schema, &batches)
        .map_err(|_| Error("invalid Arrow chunks"))
}

fn source_time(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 19
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b' '
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return false;
    }
    if !bytes
        .iter()
        .enumerate()
        .all(|(index, byte)| matches!(index, 4 | 7 | 10 | 13 | 16) || byte.is_ascii_digit())
    {
        return false;
    }
    let numbers = [
        &text[..4],
        &text[5..7],
        &text[8..10],
        &text[11..13],
        &text[14..16],
        &text[17..],
    ]
    .map(|digits| digits.parse::<u32>().unwrap_or(u32::MAX));
    let [year, month, day, hour, minute, second] = numbers;
    let valid = (1899..=2024).contains(&year)
        && (1..=12).contains(&month)
        && day >= 1
        && day <= sdm_labels::date_offset::days_in_month(i64::from(year), month)
        && hour < 24
        && minute < 60
        && second < 60;
    if !valid {
        return false;
    }
    let seconds = sdm_labels::date_offset::days_from_civil(i64::from(year), month, day) * 86400
        + i64::from(hour * 3600 + minute * 60 + second);
    for (year, month, day, clock) in [(2024, 1, 1, 0), (1969, 12, 31, 86399), (1900, 3, 1, 0)] {
        let base = (sdm_labels::date_offset::days_from_civil(year, month, day) * 86400 + clock)
            * 1_000_000_000;
        for step in [1_000_000_000, 86_400_000_000_000, 1, 999_999_999] {
            for multiple in -5..=30 {
                for extra in [0, 1, 500_000_000] {
                    if (base + multiple * step + extra).div_euclid(1_000_000_000) == seconds {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn host_independent_time(array: &dyn Array) -> bool {
    match array.data_type() {
        DataType::Timestamp(TimeUnit::Nanosecond, None) | DataType::Null => true,
        DataType::Utf8 => array.as_string::<i32>().iter().flatten().all(source_time),
        DataType::LargeUtf8 => array.as_string::<i64>().iter().flatten().all(source_time),
        _ => false,
    }
}

fn store(arguments: &Value) -> Result<Value> {
    let graph = GraphSpec::from_json(text(field(arguments, "graph_json")?)?)
        .map_err(|_| Error("invalid graph input"))?;
    let tables = object(field(arguments, "tables")?)?
        .iter()
        .map(|(name, table)| {
            let table = arrow_table(table)?;
            let columns = table
                .schema()
                .fields()
                .iter()
                .zip(table.columns())
                .map(|(field, array)| (field.name().clone(), array.clone()))
                .collect::<Vec<_>>();
            if let Some(specification) = graph.tables.iter().find(|table| table.name == *name) {
                for (name, array) in &columns {
                    let timed = specification.time_column.as_ref() == Some(name)
                        || specification.columns.iter().any(|column| {
                            column.name == *name
                                && (column.dtype == "time" || column.stype == "timestamp")
                        });
                    require(
                        !timed || host_independent_time(array.as_ref()),
                        "sampling date input requires explicit host facts",
                    )?;
                }
            }
            Ok((name.clone(), columns))
        })
        .collect::<Result<Vec<_>>>()?;
    let stored = match sdm_sampling::graph_store::graph_store(
        &graph,
        &tables,
        &sdm_schema::dt::Host::assumed(),
    ) {
        Ok(stored) => stored,
        Err(error) => return Ok(json!({"error": {"code": error.code, "message": error.message}})),
    };
    let report: Map<String, Value> = stored.report.into_iter().map(|(name, report)| (name, json!({"input_rows": report.input_rows, "output_rows": report.output_rows, "null_primary_key_rows": report.null_primary_key_rows, "duplicate_primary_key_rows": report.duplicate_primary_key_rows, "null_time_rows": report.null_time_rows}))).collect();
    let masks: Map<String, Value> = stored
        .masks
        .into_iter()
        .map(|(name, values)| (name, json!(values)))
        .collect();
    let keys: Map<String, Value> = stored
        .keys
        .into_iter()
        .map(|(name, values)| {
            let values = values
                .into_iter()
                .map(|value| match value {
                    KeyValue::Int(value) => json!(value),
                    KeyValue::Float(value) => json!(value),
                    KeyValue::Text(value) => json!(value),
                })
                .collect::<Vec<_>>();
            (name, json!(values))
        })
        .collect();
    let integers = |entries: Vec<(String, Vec<i64>)>| -> Value {
        Value::Object(
            entries
                .into_iter()
                .map(|(name, values)| (name, json!(values)))
                .collect(),
        )
    };
    let ranges: Map<String, Value> = stored
        .time_range
        .into_iter()
        .map(|(name, (lower, upper))| (name, json!([lower, upper])))
        .collect();
    let edges = |entries: Vec<(sdm_sampling::neighbor::EdgeType, Vec<i64>)>| -> Value {
        Value::Object(
            entries
                .into_iter()
                .map(|(edge, values)| (sdm_sampling::neighbor::rel_type(&edge), json!(values)))
                .collect(),
        )
    };
    Ok(
        json!({"report": report, "masks": masks, "keys": keys, "node_ids": integers(stored.node_ids), "time": integers(stored.time), "time_range": ranges, "colptr": edges(stored.colptr), "row": edges(stored.row)}),
    )
}

fn debug_job(job: &Value) -> Result<Exact> {
    let arguments = array(field(job, "args")?)?;
    let value = match text(field(job, "call")?)? {
        "sampling_op" if arguments.len() == 2 => {
            let result = sdm_sampling::fixture_op(text(&arguments[0])?, text(&arguments[1])?)
                .map_err(|_| Error("invalid debug sampling operation"))?;
            return Exact::parse(result.as_bytes());
        }
        "date_offset_apply" if arguments.len() == 3 => {
            let timestamps: Value = serde_json::from_str(text(&arguments[0])?)
                .map_err(|_| Error("invalid debug timestamps"))?;
            let specification: Value = serde_json::from_str(text(&arguments[1])?)
                .map_err(|_| Error("invalid debug offset"))?;
            if let Err(error) = offset(&specification)?.constructed() {
                return Ok(Exact::from_json(
                    json!({"raised": {"code": error.code, "message": error.message}}),
                ));
            }
            apply(&json!({"timestamps": timestamps, "offset": specification, "sign": arguments[2]}))
        }
        "date_offset_to_seconds" if arguments.len() == 2 => {
            let value: Value = serde_json::from_str(text(&arguments[0])?)
                .map_err(|_| Error("invalid debug offset"))?;
            let result = seconds(&offset(&value)?, &arguments[1])?;
            return if let Ok(error) = result.field("error") {
                Ok(Exact::Object(vec![("raised".into(), error.clone())]))
            } else {
                Ok(Exact::Object(vec![("value".into(), result)]))
            };
        }
        _ => Err(Error("invalid debug sampling job")),
    }?;
    Ok(Exact::from_json(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_integer_transport_preserves_wide_exact_results() {
        for decimal in [
            "18446744073709551616",
            "-18446744073709551617",
            "123456789012345678901234567890",
        ] {
            let number = exact_integer(decimal).unwrap();
            assert_eq!(number.to_json(), decimal);
            let encoded = number.typed().unwrap();
            assert_eq!(encoded.to_json(), format!("{{\"scalar\":{decimal}}}"));
            assert_eq!(Exact::parse(encoded.to_json().as_bytes()).unwrap(), encoded);
        }
        for malformed in ["01", "1.5", "1e3", "null", "{}", "1 2"] {
            assert!(exact_integer(malformed).is_err());
        }
    }

    #[test]
    fn sampling_generator_routes_preserve_computed_wide_results() {
        let offset = json!({"n": 22, "kwds": {"months": 424, "days": 1483}});
        for (name, arguments) in [
            (
                "sampling.offset_pairs",
                json!({
                    "offset": offset, "sign": -1, "repeat": 0, "timestamps": [],
                    "seconds_bounds": ["earliest", "latest"], "nanoseconds": true
                }),
            ),
            (
                "sampling.hashseed_python_jobs",
                json!({"kind": "offset_ns", "offset": offset}),
            ),
        ] {
            let input = Exact::from_json(json!({
                "generator": {"name": name, "stream_seed": 1, "ordinal": 0},
                "generator_input": super::super::encode_typed(arguments).unwrap()
            }));
            let result = super::super::generator_exact(&input).unwrap();
            let expected_result = if name == "sampling.offset_pairs" {
                r#"{"mapping":[[{"scalar":"steps"},{"list":[]}],[{"scalar":"seconds"},{"list":[{"scalar":25385184000},{"scalar":27803001600}]}],[{"scalar":"nanoseconds"},{"mapping":[[{"scalar":"ns"},{"scalar":27349056000000000000}]]}]]}"#
            } else {
                r#"{"mapping":[[{"scalar":"answers"},{"list":[{"mapping":[[{"scalar":"ns"},{"scalar":27349056000000000000}]]}]}],[{"scalar":"safe_python"},{"list":[]}],[{"scalar":"purpose"},{"scalar":"native-hashseed-consistency"}]]}"#
            };
            let expected = format!(
                r#"{{"family":"original-generator-v1","observation":{{"status":"success","schema":[],"payloads":[],"sql":[],"result":{expected_result},"error":null,"warnings":[]}}}}"#
            );
            assert_eq!(result, Exact::parse(expected.as_bytes()).unwrap());
            assert_eq!(Exact::parse(result.to_json().as_bytes()).unwrap(), result);
            assert!(result.narrow().is_err());
        }
        let offset = json!({"n": 16, "kwds": {"years": -9223372036854775807_i64, "months": 2, "hours": 2147483646}});
        let expected = seconds(&super::offset(&offset).unwrap(), &json!("latest")).unwrap();
        assert!(expected.narrow().is_err());
        let calls = json!({"calls": [
            {"call": "sampling_op", "args": ["sampling.date_offset_to_seconds", json!({"offset": offset, "bound": "latest"}).to_string()]},
            {"call": "date_offset_to_seconds", "args": [offset.to_string(), "latest"]}
        ]});
        let input = Exact::from_json(json!({
            "generator": {"name": "sampling.debug_jobs", "stream_seed": 1, "ordinal": 0},
            "generator_input": super::super::encode_typed(calls).unwrap()
        }));
        let result = super::super::generator_exact(&input).unwrap();
        assert_eq!(result.to_json().matches(&expected.to_json()).count(), 2);
        assert_eq!(Exact::parse(result.to_json().as_bytes()).unwrap(), result);
    }

    #[test]
    fn ordinary_numeric_transport_keeps_integer_and_float_types() {
        for integer in [i64::MIN, -1, 0, 1, i64::MAX] {
            let original = json!(integer);
            let roundtrip: Value = serde_json::from_str(&original.to_string()).unwrap();
            assert_eq!(roundtrip.as_i64(), Some(integer));
            assert_eq!(
                super::super::encode_typed(roundtrip).unwrap(),
                json!({"scalar": integer})
            );
        }
        assert_eq!(
            serde_json::from_str::<Value>(&u64::MAX.to_string())
                .unwrap()
                .as_u64(),
            Some(u64::MAX)
        );
        for (number, hexadecimal) in [
            (0.0_f64, "0x0.0p+0"),
            (-0.0, "-0x0.0p+0"),
            (1.5, "0x1.8000000000000p+0"),
        ] {
            let original = json!(number);
            let roundtrip: Value = serde_json::from_str(&original.to_string()).unwrap();
            assert!(roundtrip.is_f64());
            assert_eq!(roundtrip.as_f64().unwrap().to_bits(), number.to_bits());
            assert_eq!(
                super::super::encode_typed(roundtrip).unwrap(),
                json!({"float": hexadecimal})
            );
            assert_eq!(
                Exact::from_json(original)
                    .typed()
                    .unwrap()
                    .narrow()
                    .unwrap(),
                json!({"float": hexadecimal})
            );
        }
        for source in ["1e3", "1E+03", "-0.0", "1.25", "1", "-1"] {
            let ordinary = serde_json::from_str(source).unwrap();
            assert_eq!(
                Exact::parse(source.as_bytes())
                    .unwrap()
                    .typed()
                    .unwrap()
                    .narrow()
                    .unwrap(),
                super::super::encode_typed(ordinary).unwrap()
            );
        }
    }

    #[test]
    fn sampling_uses_new_store_arrays_and_graph_edges() {
        let mut specification = json!({
            "tables": [{"name": "events", "columns": []}, {"name": "users", "columns": []}],
            "edges": [{"src_table": "events", "fkey": "user", "dst_table": "users"}],
        });
        let input = json!({"graph_json": specification.to_string()});
        let mut stored = json!({
            "colptr": {"events__user__users": [0, 1, 2], "users____###REV###__user__events": [0, 1, 2]},
            "row": {"events__user__users": [0, 1], "users____###REV###__user__events": [0, 1]},
            "time": {},
        });
        let request = json!({"seed": 7, "row": "recorded arrays must not answer", "calls": [{
            "reseed": null, "entity": "users", "seed_node": [0], "seed_time": [],
            "offsets": {}, "num_neighbors": {"events__user__users": [-1], "users____###REV###__user__events": [-1]},
        }]});
        let original = sample_stored(&input, &stored, &request).unwrap();
        assert!(original[0].is_array());
        stored["row"]["events__user__users"] = json!([1, 0]);
        let changed = sample_stored(&input, &stored, &request).unwrap();
        assert!(changed[0].is_array());
        assert_ne!(original, changed);
        specification["edges"][0]["fkey"] = json!("different");
        let changed_edge = sample_stored(
            &json!({"graph_json": specification.to_string()}),
            &stored,
            &request,
        )
        .unwrap();
        assert!(changed_edge[0].get("error").is_some());
    }

    #[test]
    fn seeds_keep_every_word_and_refuse_nonintegers() {
        assert!(seed_words(&json!(0)).unwrap().is_empty());
        assert_eq!(seed_words(&json!(4294967296u64)).unwrap(), [0, 1]);
        assert_eq!(seed_words(&json!(u64::MAX)).unwrap(), [u32::MAX, u32::MAX]);
        for malformed in [json!(-1), json!(1.5), json!(true), json!("12")] {
            assert!(seed_words(&malformed).is_err());
        }
    }

    #[test]
    fn host_guard_excludes_ambiguous_and_out_of_range_dates() {
        for accepted in [
            "1900-03-01 00:00:00",
            "1969-12-31 23:59:59",
            "2024-01-01 00:00:00",
        ] {
            assert!(source_time(accepted));
        }
        for refused in [
            "9999-01-01 00:00:00",
            "1899-12-31 23:59:59",
            "2024-02-29 12:34:56",
            "01-02-24",
            "12:34:56",
            "2024-01-01 00:00:00 UTC",
            "2024-01-01 00:00:00.1",
            "1900-02-29 00:00:00",
            "2024-13-01 00:00:00",
            "2024-01-01 24:00:00",
            "2025-01-01 00:00:00",
        ] {
            assert!(!source_time(refused));
        }
    }

    #[test]
    fn complete_source_dates_ignore_host_year_zone_and_unit_inference() {
        let mut earlier = sdm_schema::dt::Host::assumed();
        earlier.year = 1900;
        earlier.today = (1900, 1, 1);
        earlier.pads_year = false;
        earlier.tznames = vec!["UTC".into()];
        earlier.timestamp_series_units =
            sdm_schema::dt::pandas::TimestampSeriesUnits::from_names(["ns"; 4]).unwrap();
        let mut later = earlier.clone();
        later.year = 2100;
        later.today = (2100, 12, 31);
        later.pads_year = true;
        later.tznames = vec!["EST".into(), "EDT".into()];
        later.timestamp_series_units =
            sdm_schema::dt::pandas::TimestampSeriesUnits::from_names(["us", "us", "us", "ns"])
                .unwrap();
        let mut values = vec![None];
        for year in [1899, 1900, 1969, 1970, 2023, 2024] {
            for month in 1..=12 {
                for day in 1..=sdm_labels::date_offset::days_in_month(year, month) {
                    values.push(Some(format!("{year:04}-{month:02}-{day:02} 23:59:59")));
                }
            }
        }
        let call = sdm_schema::dt::Call {
            coerce: true,
            ..Default::default()
        };
        let earlier =
            sdm_schema::dt::pandas::objects_to_datetime_public(&values, call, &earlier).unwrap();
        let later =
            sdm_schema::dt::pandas::objects_to_datetime_public(&values, call, &later).unwrap();
        assert_eq!(earlier.values, later.values);
        assert!(earlier.values.iter().skip(1).all(Option::is_some));
    }
}
