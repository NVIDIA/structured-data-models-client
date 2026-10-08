// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The SQL fragments every dialect shares (`rfm/base/utils.py`), and
//! identifier quoting (`sdm_connectors/sql.py::quote_ident`).

use crate::Error;
use crate::py::{Dtype, Fraction};

/// `quote_ident(ident, char)`: `ident` between `char`s, each `char` in it
/// doubled.
pub fn quote_ident(ident: &str, char: &str) -> String {
    let doubled = format!("{char}{char}");
    format!("{char}{}{char}", ident.replace(char, &doubled))
}

/// `ordered_sample_sql`: `projections` of the first `num_rows` rows of
/// `source` in the order `order_terms` give, the terms selected first as
/// `__kumo_order_<i>__` and ordered by position.
pub fn ordered_sample_sql(
    source: &str,
    projections: &[String],
    order_terms: &[String],
    num_rows: i64,
) -> String {
    if order_terms.is_empty() {
        return format!(
            "SELECT {} FROM {source} LIMIT {num_rows}",
            projections.join(", ")
        );
    }
    let selected: Vec<String> = order_terms
        .iter()
        .enumerate()
        .map(|(i, term)| format!("{term} AS __kumo_order_{i}__"))
        .chain(projections.iter().cloned())
        .collect();
    let positions: Vec<String> = (1..=order_terms.len()).map(|i| i.to_string()).collect();
    format!(
        "SELECT {} FROM {source} ORDER BY {} LIMIT {num_rows}",
        selected.join(", "),
        positions.join(", ")
    )
}

/// The largest fraction a block prefilter is applied at.
pub const MAX_PREFILTER_FRACTION: Fraction = Fraction {
    numerator: 1,
    denominator: 8,
};

/// `prefilter_fraction(wanted)`: `wanted` rounded up to a power of two, or
/// `None` where that is above an eighth or `wanted` is not positive.
pub fn prefilter_fraction(wanted: Fraction) -> Result<Option<Fraction>, Error> {
    if wanted.numerator <= 0 {
        return Ok(None);
    }
    // The smallest power of two at least `wanted`: 2**exponent >= n / d.
    let (n, d) = (wanted.numerator, wanted.denominator);
    let mut exponent: i32 = 0;
    let fits = |e: i32| -> Result<bool, Error> {
        let overflow = || Error::new("a fraction too large");
        if e >= 0 {
            let scaled = d.checked_mul(1i128.checked_shl(e as u32).ok_or_else(overflow)?);
            Ok(scaled.ok_or_else(overflow)? >= n)
        } else {
            let scaled = n.checked_mul(1i128.checked_shl((-e) as u32).ok_or_else(overflow)?);
            Ok(scaled.ok_or_else(overflow)? <= d)
        }
    };
    while !fits(exponent)? {
        exponent += 1;
    }
    while fits(exponent - 1)? {
        exponent -= 1;
    }
    let fraction = if exponent >= 0 {
        Fraction::integer(1i128 << exponent)
    } else {
        Fraction::new(1, 1i128 << (-exponent))?
    };
    if fraction > MAX_PREFILTER_FRACTION {
        return Ok(None);
    }
    Ok(Some(fraction))
}

/// `percent_literal(fraction)`: `repr(100 * float(fraction))`.
///
/// `float(fraction)` is the correctly rounded quotient, which one division
/// of two floats gives where both are exact: below 2**53, or a power of two.
pub fn percent_literal(fraction: Fraction) -> Result<String, Error> {
    const EXACT: u128 = 1 << 53;
    let (n, d) = (fraction.numerator, fraction.denominator);
    let exact = |v: i128| v.unsigned_abs() < EXACT || v.unsigned_abs().is_power_of_two();
    if !(exact(n) && exact(d)) {
        return Err(Error::new(format!(
            "no exact float of the fraction {}",
            fraction.text()
        )));
    }
    let value = n as f64 / d as f64;
    Ok(sdm_pql::pyrepr::float_repr(100.0 * value))
}

/// `key_count_select`: one `SELECT` of a key count's `UNION ALL`.
pub fn key_count_select(
    label: i64,
    source: &str,
    columns: &[String],
    unique: bool,
    count_distinct: &str,
) -> String {
    let missing: Vec<String> = columns.iter().map(|c| format!("{c} IS NULL")).collect();
    let nulls = format!("COUNT(CASE WHEN {} THEN 1 END)", missing.join(" OR "));
    let distinct = if unique {
        format!("COUNT(*) - {nulls}")
    } else {
        count_distinct.to_string()
    };
    format!("SELECT {label} AS k, COUNT(*) AS n, {distinct} AS d, {nulls} AS z\nFROM {source}")
}

/// `count_distinct_values`: the distinct values of a key, as Spark and
/// Snowflake count several columns.
pub fn count_distinct_values(columns: &[String]) -> String {
    format!("COUNT(DISTINCT {})", columns.join(", "))
}

/// `repeated_key_select`: one value of the key `columns` more than one row
/// of `source` holds.
pub fn repeated_key_select(source: &str, columns: &[String]) -> String {
    let quoted = columns.join(", ");
    let present: Vec<String> = columns.iter().map(|c| format!("{c} IS NOT NULL")).collect();
    format!(
        "SELECT {quoted}\nFROM {source}\nWHERE {}\nGROUP BY {quoted}\nHAVING COUNT(*) > 1\nLIMIT 1",
        present.join(" AND ")
    )
}

/// How many columns the inference sample of a table without a declared
/// primary key hashes, at most.
const NUM_SAMPLE_HASH_COLUMNS: usize = 8;

/// A table's source column, as `sample_order_terms` reads it.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceColumn {
    pub name: String,
    pub dtype: Option<Dtype>,
}

/// `sample_order_terms`: the terms that pick and order the rows a warehouse
/// table's types are inferred from.
pub fn sample_order_terms(
    columns: &[SourceColumn],
    primary_key: Option<&str>,
    quote: impl Fn(&str) -> String,
    hash_function: &str,
) -> Vec<String> {
    let scalars: Vec<(&str, Dtype)> = columns
        .iter()
        .filter_map(|c| match c.dtype {
            Some(dtype) if dtype != Dtype::Unsupported && !dtype.is_list() => {
                Some((c.name.as_str(), dtype))
            }
            _ => None,
        })
        .collect();
    if scalars.is_empty() {
        return Vec::new();
    }
    let refs: Vec<String> = scalars.iter().map(|(name, _)| quote(name)).collect();
    if let Some(key) = primary_key
        && let Some(index) = scalars.iter().position(|(name, _)| *name == key)
    {
        let key = &refs[index];
        let mut terms = vec![format!("{hash_function}({key})"), key.clone()];
        terms.extend(refs.iter().filter(|r| *r != key).cloned());
        return terms;
    }
    let mut by_cost: Vec<usize> = (0..scalars.len()).collect();
    by_cost.sort_by_key(|&i| (scalars[i].1.is_string(), i));
    let mut hashed: Vec<usize> = by_cost.into_iter().take(NUM_SAMPLE_HASH_COLUMNS).collect();
    hashed.sort_unstable();
    let hashed: Vec<&str> = hashed.iter().map(|&i| refs[i].as_str()).collect();
    let mut terms = vec![format!("{hash_function}({})", hashed.join(", "))];
    terms.extend(refs.iter().cloned());
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_round_up_to_powers_of_two() {
        let f = |n, d| Fraction::new(n, d).unwrap();
        assert_eq!(prefilter_fraction(f(1, 8)).unwrap(), Some(f(1, 8)));
        assert_eq!(prefilter_fraction(f(1, 7)).unwrap(), None);
        assert_eq!(prefilter_fraction(f(2048, 16385)).unwrap(), Some(f(1, 8)));
        assert_eq!(
            prefilter_fraction(f(2048, 10_i128.pow(9))).unwrap(),
            Some(f(1, 262_144))
        );
        assert_eq!(prefilter_fraction(f(0, 1)).unwrap(), None);
        assert_eq!(percent_literal(f(1, 128)).unwrap(), "0.78125");
        assert_eq!(
            percent_literal(f(1, 1 << 40)).unwrap(),
            "9.094947017729282e-11"
        );
    }

    #[test]
    fn quoting_doubles_the_quote() {
        assert_eq!(quote_ident("a\"b", "\""), "\"a\"\"b\"");
        assert_eq!(quote_ident("it's", "'"), "'it''s'");
    }
}
