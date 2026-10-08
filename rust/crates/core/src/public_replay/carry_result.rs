// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::RefCell;
use std::collections::VecDeque;

use sdm_schema::pyval::PyVal;
use serde_json::{Value, json};

use super::{Result, exact_fields, field, text};

pub(super) fn replay(input: &Value, primitives: &Value) -> Result<Value> {
    let result = sdm_request::layout::decode_result(input)?;
    let pending = RefCell::new(
        primitives
            .as_array()
            .ok_or("invalid result primitives")?
            .iter()
            .collect::<VecDeque<_>>(),
    );
    let held = RefCell::new(Vec::<Value>::new());
    let take = |operation: &str, arguments: Value| -> Result<Value> {
        let row = pending
            .borrow_mut()
            .pop_front()
            .ok_or("missing result primitive")?;
        exact_fields(row, &["operation", "input", "output"])?;
        if field(row, "operation")? != operation || field(row, "input")? != &arguments {
            return Err("result primitive operation or input differs".into());
        }
        Ok(field(row, "output")?.clone())
    };
    let layout = sdm_request::layout::host_layout_with(
        &result,
        |cells| -> Result<(String, usize)> {
            let arguments = sdm_request::values::encode(&PyVal::List(cells.to_vec()));
            let column = take("infer_column", arguments)?;
            exact_fields(&column, &["dtype", "cells", "index"])?;
            let dtype = text(field(&column, "dtype")?)?;
            let values = field(&column, "cells")?
                .as_array()
                .ok_or("invalid primitive cells")?;
            let indices = field(&column, "index")?
                .as_array()
                .ok_or("invalid primitive index")?;
            if !matches!(
                dtype,
                "object" | "int64" | "float64" | "complex128" | "str" | "string"
            ) || values.len() != cells.len()
                || indices
                    != &(0..cells.len())
                        .map(|index| json!(index))
                        .collect::<Vec<_>>()
            {
                return Err("invalid result column primitive".into());
            }
            let reference = held.borrow().len().to_string();
            held.borrow_mut().push(column);
            Ok((reference, cells.len()))
        },
        |reference, stable, ascending, missing_last| -> Result<Vec<usize>> {
            let index = reference
                .parse::<usize>()
                .map_err(|_| "invalid result reference")?;
            let column = held
                .borrow()
                .get(index)
                .cloned()
                .ok_or("missing result column")?;
            let positions = take(
                "column_order",
                json!({"column": column, "stable": stable, "ascending": ascending, "missing_last": missing_last}),
            )?;
            positions
                .as_array()
                .ok_or("invalid result permutation")?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| "invalid result position".to_owned())
                })
                .collect()
        },
        str::to_owned,
    )?;
    if !pending.borrow().is_empty() {
        return Err("unused result primitives".into());
    }
    let columns = field(&layout, "columns")?
        .as_array()
        .ok_or("invalid computed layout")?;
    let rows = field(&layout, "rows")?
        .as_u64()
        .ok_or("invalid computed row count")? as usize;
    let mut names = Vec::new();
    let mut dtypes = Vec::new();
    let mut values = vec![Vec::new(); rows];
    for column in columns {
        names.push(field(column, "name")?.clone());
        let reference = text(field(column, "reference")?)?
            .parse::<usize>()
            .map_err(|_| "invalid computed reference")?;
        let held = held.borrow();
        let actual = held.get(reference).ok_or("missing computed column")?;
        dtypes.push(field(actual, "dtype")?.clone());
        let cells = field(actual, "cells")?
            .as_array()
            .ok_or("invalid held cells")?;
        for (row, position) in field(column, "positions")?
            .as_array()
            .ok_or("invalid computed positions")?
            .iter()
            .enumerate()
        {
            let position = position.as_u64().ok_or("invalid computed position")? as usize;
            values[row].push(
                cells
                    .get(position)
                    .ok_or("computed position outside column")?
                    .clone(),
            );
        }
    }
    Ok(
        json!({"status": "success", "schema": [], "payloads": [], "sql": [], "error": null, "warnings": [], "result": {"frame": {"columns": names, "dtypes": dtypes, "index": (0..rows).collect::<Vec<_>>(), "values": values}}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (Value, Value) {
        let cell = json!({"$np": {"type": "bytes48", "value": {"$bytes": "YWJjZGVm"}}});
        let index = json!({"dtype": "int64", "cells": [{"$np": {"type": "int64", "value": 0}}], "index": [0]});
        let input = json!({"items": [{"row_index": 0, "prediction": cell}], "outputs": [], "metadata": {}, "explanation": null, "model": null, "request_id": null, "names_its_rows": false, "names_its_step": false, "unscored_candidates": []});
        let primitives = json!([
            {"operation": "infer_column", "input": [0], "output": index},
            {"operation": "infer_column", "input": [cell], "output": {"dtype": "object", "cells": [cell], "index": [0]}},
            {"operation": "column_order", "input": {"column": index, "stable": true, "ascending": true, "missing_last": true}, "output": [0]},
        ]);
        (input, primitives)
    }

    #[test]
    fn result_layout_uses_computed_columns_and_observed_host_primitives() {
        let (input, primitives) = inputs();
        let output = replay(&input, &primitives).unwrap();
        assert_eq!(
            output["result"]["frame"]["columns"],
            json!(["row_index", "prediction"])
        );
        assert_eq!(
            output["result"]["frame"]["dtypes"],
            json!(["int64", "object"])
        );
        assert_eq!(
            output["result"]["frame"]["values"][0][1],
            input["items"][0]["prediction"]
        );
    }

    #[test]
    fn result_primitives_reject_missing_extra_wrong_input_and_permutation() {
        let (input, primitives) = inputs();
        let mut missing = primitives.clone();
        missing.as_array_mut().unwrap().pop();
        assert!(replay(&input, &missing).is_err());
        let mut extra = primitives.clone();
        extra.as_array_mut().unwrap().push(primitives[0].clone());
        assert!(replay(&input, &extra).is_err());
        let mut wrong_input = primitives.clone();
        wrong_input[0]["input"] = json!([99]);
        assert!(replay(&input, &wrong_input).is_err());
        let mut wrong_order = primitives.clone();
        wrong_order[2]["output"] = json!([1]);
        assert!(replay(&input, &wrong_order).is_err());
    }
}
