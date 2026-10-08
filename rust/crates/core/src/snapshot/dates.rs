// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use sdm_schema::dates::{self, TimeColumn};
use serde_json::{Value, json};

use super::{
    Error, Result, array, decode_typed, encode_typed, field, fields, require, text,
    typed_observation,
};

pub(super) fn replay(input: &Value, facts: &Value) -> Result<Value> {
    fields(facts, &["host", "primitives"])?;
    require(
        array(field(facts, "primitives")?)?.is_empty(),
        "unused date host primitives",
    )?;
    let host = super::host::descriptor(field(facts, "host")?)?;
    let input = decode_typed(input)?;
    fields(
        &input,
        &["column", "operations", "successful_conversion_read"],
    )?;
    require(
        field(&input, "successful_conversion_read")? == true,
        "date conversion read must remain linked",
    )?;
    let operations = [
        "dates.day_comes_first",
        "dates.looks_unambiguous",
        "dates.shapes_written",
        "dates.reads_as_a_date",
        "dates.conversion_that_works",
        "time_column.read",
        "infer.timestamp",
    ];
    require(
        field(&input, "operations")? == &json!(operations),
        "unsupported date operation sequence",
    )?;
    let column = field(&input, "column")?;
    fields(column, &["values", "only_plausible_years", "column"])?;
    let raw_values = field(column, "values")?;
    let values = array(raw_values)?
        .iter()
        .map(|value| {
            if value.is_null() {
                Ok(None)
            } else {
                text(value).map(|value| Some(value.to_owned()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let mut output = Vec::new();
    let mut warnings = Vec::new();
    for operation in operations {
        let mut given = json!({"values": raw_values});
        let mut conversion = None;
        let result = match operation {
            "dates.day_comes_first" => json!({"output": dates::day_comes_first(&values)}),
            "dates.looks_unambiguous" => json!({"output": dates::looks_unambiguous(&values)}),
            "dates.shapes_written" => {
                let mut names = dates::shapes_written(&values, &host)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>();
                names.sort_unstable();
                json!({"output": names})
            }
            "dates.reads_as_a_date" => json!({"output": dates::reads_as_a_date(&values, &host)}),
            "dates.conversion_that_works" => {
                let plausible = field(column, "only_plausible_years")?
                    .as_bool()
                    .ok_or(Error("invalid date year policy"))?;
                given["only_plausible_years"] = json!(plausible);
                conversion = dates::conversion_that_works(&values, plausible, false, &host);
                json!({"output": conversion.as_ref().map(|conversion| conversion.arguments())})
            }
            "time_column.read" => match dates::time_column(&values, "t.when", &host) {
                Ok(TimeColumn::Untouched(length)) => {
                    json!({"output": {"dtype": "object", "values": vec![Value::Null; length]}})
                }
                Ok(TimeColumn::Read(read)) => {
                    json!({"output": {"dtype": read.dtype(), "values": (0..values.len()).map(|position| read.isoformat(position)).collect::<Vec<_>>()}})
                }
                Err(error) => json!({"error": {"code": error.code, "message": error.message}}),
            },
            "infer.timestamp" => {
                given["column"] = field(column, "column")?.clone();
                let (result, emitted) =
                    dates::contains_timestamp(&values, text(field(column, "column")?)?, &host);
                warnings.extend(emitted.into_iter().map(|message| json!({"class": "builtins.UserWarning", "code": null, "message": message, "details": null})));
                match result {
                    Ok(found) => json!({"output": found}),
                    Err(error) => json!({"error": {"code": error.class, "message": error.message}}),
                }
            }
            _ => return Err(Error("unknown date operation")),
        };
        output.push(
            json!({"tuple": [{"scalar": operation}, encode_typed(given)?, encode_typed(result)?]}),
        );
        if let Some(conversion) = conversion {
            let given = json!({"values": raw_values, "conversion": {"utc": conversion.utc, "dayfirst": conversion.dayfirst, "mixed": conversion.mixed}});
            let result = match dates::read(&values, &conversion, &host) {
                Ok(read) => {
                    json!({"output": (0..values.len()).map(|position| read.isoformat(position)).collect::<Vec<_>>()})
                }
                Err(error) => json!({"error": {"code": error.class, "message": error.message}}),
            };
            output.push(json!({"tuple": [{"scalar": "dates.read"}, encode_typed(given)?, encode_typed(result)?]}));
        }
    }
    let mut observation = typed_observation(json!({"list": output}), Value::Null)?;
    observation["observation"]["warnings"] = json!(warnings);
    Ok(observation)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Value {
        json!({"host": {"year": 2026, "today": [2026, 10, 6], "pads_year": false, "tznames": ["UTC", "UTC"], "timestamp_series_units": ["s", "ms", "us", "ns"]}, "primitives": []})
    }

    fn input() -> Value {
        json!({"column": {"values": ["29.02.2024"], "only_plausible_years": false, "column": "created"}, "operations": ["dates.day_comes_first", "dates.looks_unambiguous", "dates.shapes_written", "dates.reads_as_a_date", "dates.conversion_that_works", "time_column.read", "infer.timestamp"], "successful_conversion_read": true})
    }

    #[test]
    fn dates_preserve_actual_warning_and_linked_conversion() {
        let output = replay(&encode_typed(input()).unwrap(), &facts()).unwrap();
        assert_eq!(
            output["observation"]["warnings"].as_array().unwrap().len(),
            1
        );
        let results = output["observation"]["result"]["list"].as_array().unwrap();
        assert!(
            results
                .iter()
                .any(|entry| entry["tuple"][0]["scalar"] == "dates.read")
        );
    }

    #[test]
    fn dates_refuse_unused_facts_and_unlinked_conversion() {
        let raw = encode_typed(input()).unwrap();
        let mut host = facts();
        host["primitives"] = json!([{}]);
        assert!(replay(&raw, &host).is_err());
        host = facts();
        host["host"]["today"] = json!([2026, 0, 1]);
        assert!(replay(&raw, &host).is_err());
        host = facts();
        host["host"]["timestamp_series_units"] = json!(["s", "s", "us", "ns"]);
        assert!(replay(&raw, &host).is_err());
        let mut given = input();
        given["successful_conversion_read"] = json!(false);
        assert!(replay(&encode_typed(given).unwrap(), &facts()).is_err());
    }
}
