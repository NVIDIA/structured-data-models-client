// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generated direct decoder order and selected native conversion facts.

use sdm_wire::json::{self, Value};
use sdm_wire::response::{NativeFault, PredictionResponse};

fn response(fields: &str) -> Value {
    json::read_answer(format!(r#"{{"id":"server","model":"kumo-tabular","metadata":{{}},"predictions":[{{"row_index":0,{fields}}}]}}"#).as_bytes()).unwrap()
}

#[test]
fn direct_generated_order_and_gathered_order_are_explicit() {
    for (fields, direct, gathered) in [
        (
            r#""embeddings":"bad","quantiles":[]"#,
            "prediction.embeddings",
            "prediction.quantiles",
        ),
        (
            r#""embeddings":"bad","explanation":1"#,
            "prediction.embeddings",
            "prediction.explanation",
        ),
        (
            r#""quantiles":[],"explanation":1"#,
            "prediction.quantiles",
            "prediction.explanation",
        ),
    ] {
        let value = response(fields);
        let fault = PredictionResponse::read_direct(&value).unwrap_err();
        assert_eq!(fault.field, direct);
        assert_eq!(fault.item, Some(0));
        assert!(fault.value.is_some());
        assert_eq!(
            PredictionResponse::read(&value).unwrap_err().field,
            gathered
        );
    }
}

#[test]
fn nested_fault_selects_the_consumed_value_and_native_operation() {
    let fault =
        PredictionResponse::read_direct(&response(r#""embeddings":[1,"bad"]"#)).unwrap_err();
    assert_eq!(fault.reason, "not_a_number");
    assert_eq!(fault.value, Some(Value::Text("bad".into())));
    let fault = PredictionResponse::read_direct(&response(r#""rankings":1"#)).unwrap_err();
    assert_eq!(fault.native, Some(NativeFault::Iter));
    assert_eq!(fault.value, Some(Value::int(1)));
    let fault = PredictionResponse::read_direct(&response(r#""rankings":["bad"]"#)).unwrap_err();
    assert_eq!(fault.native, Some(NativeFault::Dict));
    assert_eq!(fault.value, Some(Value::Text("bad".into())));
}

#[test]
fn whole_direct_steps_keep_exact_digits_and_numeric_overflow_keeps_its_origin() {
    let digits = "1".to_owned() + &"0".repeat(200);
    let value = response(&format!(r#""forecast_step":{digits}"#));
    let direct = PredictionResponse::read_direct(&value).unwrap();
    assert_eq!(
        direct.predictions[0].forecast_step_written.as_deref(),
        Some(digits.as_str())
    );
    assert!(direct.predictions[0].forecast_step.is_none());
    assert_eq!(
        PredictionResponse::read(&value).unwrap_err().reason,
        "out_of_range"
    );
    let digits = "1".to_owned() + &"0".repeat(400);
    for (member, reason) in [
        (digits.clone(), "not_a_number"),
        (format!("\"{digits}\""), "not_finite"),
    ] {
        let fault = PredictionResponse::read_direct(&response(&format!(r#""scores":[{member}]"#)))
            .unwrap_err();
        assert_eq!(fault.reason, reason);
    }
}

#[test]
fn generated_float_conversion_fault_is_distinct_from_boolean_and_nonfinite_refusal() {
    let wide = "1".to_owned() + &"0".repeat(400);
    for member in ["\"bad\"".to_owned(), "null".into(), "[]".into(), wide] {
        for fields in [
            format!(r#""probabilities":{{"x":{member}}}"#),
            format!(r#""scores":[{member}]"#),
            format!(r#""embeddings":[{member}]"#),
            format!(r#""quantiles":{{"x":{member}}}"#),
        ] {
            let value = response(&fields);
            let fault = PredictionResponse::read_direct(&value).unwrap_err();
            assert_eq!(fault.native, Some(NativeFault::Float));
            assert_eq!(fault.reason, "not_a_number");
            // Gathered faults retain their established diagnostic policy.
            assert_ne!(
                PredictionResponse::read(&value).unwrap_err().native,
                Some(NativeFault::Float)
            );
        }
    }
    for (member, reason) in [("true", "not_a_number"), (r#""NaN""#, "not_finite")] {
        let fault = PredictionResponse::read_direct(&response(&format!(r#""scores":[{member}]"#)))
            .unwrap_err();
        assert_eq!(fault.reason, reason);
        assert_eq!(fault.native, None);
    }
}
