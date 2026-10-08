// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `sdm_client/materializers/tabular.py`: single-table requests built from
//! the rows of a schema's table.

use sdm_protocol::Diagnostic;
use sdm_schema::pyval::PyVal;

use crate::error::R;
use crate::pyops::{decimal_value, is_integral_type, is_real_type, py_float};
use crate::series::{Series, is_na};

/// `_is_whole(value)`.
pub fn is_whole(value: &PyVal) -> bool {
    if matches!(value, PyVal::Bool(_)) || is_integral_type(value) {
        return true;
    }
    if let Some(dec) = decimal_value(value) {
        return dec.is_finite() && dec.is_integral();
    }
    if is_real_type(value) {
        return py_float(value).is_ok_and(|x| x.is_finite() && x.fract() == 0.0);
    }
    false
}

/// `target.dropna().unique()`: the distinct present values, in order.
pub fn distinct_present(series: &Series) -> R<Vec<PyVal>> {
    if let crate::series::Dtype::Arrow(_) = &series.dtype
        && let Some(message) = sdm_schema::frame::unique_kernel_refusal(&series.dtype.name())
    {
        return Err(Diagnostic::refusal(
            "pyarrow.lib.ArrowNotImplementedError",
            None,
            message,
        ));
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for value in &series.cells {
        if is_na(value) {
            continue;
        }
        let key = value.eq_key();
        if seen.insert(key) {
            out.push(value.clone());
        }
    }
    Ok(out)
}

/// `_holds_two_whole_numbers(target)`.
pub fn holds_two_whole_numbers(series: &Series) -> R<bool> {
    let values = distinct_present(series)?;
    Ok(values.len() == 2 && values.iter().all(is_whole))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::series::Dtype;

    #[test]
    fn two_number_target_preserves_arrow_unique_refusals() {
        for dtype in ["decimal32(8, 2)", "decimal64(16, 2)"] {
            for cells in [
                vec![],
                vec![PyVal::Na],
                vec![PyVal::Decimal("1.00".into()), PyVal::Decimal("2.00".into())],
            ] {
                let target = Series::new(Some("target".into()), Dtype::Arrow(dtype.into()), cells);
                let error = holds_two_whole_numbers(&target).unwrap_err();
                assert_eq!(
                    error.class.as_deref(),
                    Some("pyarrow.lib.ArrowNotImplementedError")
                );
                assert_eq!(
                    error.message,
                    format!("Function 'unique' has no kernel matching input types ({dtype})")
                );
                assert_eq!(error.code, None);
            }
        }
        for dtype in ["decimal128(20, 2)", "decimal256(40, 2)"] {
            let target = Series::new(
                Some("target".into()),
                Dtype::Arrow(dtype.into()),
                vec![
                    PyVal::Na,
                    PyVal::Decimal("1.00".into()),
                    PyVal::Decimal("1.00".into()),
                    PyVal::Decimal("2.00".into()),
                ],
            );
            assert!(holds_two_whole_numbers(&target).unwrap());
        }
    }
}
