// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The decisions the tables and samplers make about SQL, as pure functions
//! of what they decide on: how a time bound is written, whether a column
//! holds zoned times, and the scan caps that sample a large table's blocks.
//!
//! Each takes what today's Python reads off its objects and its catalog
//! queries, already run, and gives what the Python attribute returns.

use crate::fragments::{percent_literal, prefilter_fraction, quote_ident};
use crate::int::Int;
use crate::py::{Dtype, Fraction, Py, Series};
use crate::text::{Refusal, folded, py_int};
use crate::time::Stamp;
use crate::unicode::decimal_value;
use crate::{Error, postgres, snow};

/// `DatabricksSampler._bound_format`: a bound on a `STRING` column is
/// written bare, any other with `+00:00`.
pub fn databricks_bound_format(dtype: Option<Dtype>) -> &'static str {
    if dtype == Some(Dtype::String) {
        "%Y-%m-%d %H:%M:%S"
    } else {
        "%Y-%m-%d %H:%M:%S+00:00"
    }
}

/// `PostgresSampler._utc_bounds`: bounds are UTC instants unless the column
/// holds text.
pub fn postgres_utc_bounds(dtype: Option<Dtype>) -> bool {
    dtype != Some(Dtype::String)
}

/// `SnowSampler._zone`: `LTZ` or `TZ` for a zoned source column, `TZ` for a
/// zoned expression column, `None` otherwise.
pub fn snow_zone(
    source_column: bool,
    zoned_expr: bool,
    source_type: Option<&str>,
) -> Option<&'static str> {
    if !source_column {
        return zoned_expr.then_some("TZ");
    }
    snow::zone_of(source_type)
}

/// `SnowSampler._bound_format`: with `+00:00` for a zoned column.
pub fn snow_bound_format(
    source_column: bool,
    zoned_expr: bool,
    source_type: Option<&str>,
) -> &'static str {
    match snow_zone(source_column, zoned_expr, source_type) {
        None => snow::NAIVE_BOUND_FORMAT,
        Some(_) => snow::ZONED_BOUND_FORMAT,
    }
}

/// `DuckDBTable._is_zoned_time`: a source column by its declared type, an
/// expression column by the type its sample was read as.
pub fn duckdb_is_zoned_time(
    source_column: bool,
    is_source: bool,
    source_type: Option<&str>,
    zoned_expr: bool,
) -> bool {
    if source_column && is_source {
        return matches!(
            source_type.unwrap_or(""),
            "TIMESTAMP WITH TIME ZONE" | "TIMESTAMPTZ"
        );
    }
    zoned_expr
}

/// What `DESCRIBE TABLE EXTENDED` says of a Databricks table, by key.
pub type Described = Vec<(String, String)>;

fn described_get<'a>(described: &'a Described, key: &str) -> &'a str {
    described
        .iter()
        .find(|(k, _)| k == key)
        .map_or("", |(_, v)| v.as_str())
}

/// The bytes `re.match(r'(\d+) bytes', text)` reads, if it matches.
fn statistics_bytes(text: &str) -> Option<Int> {
    let digits: String = text
        .chars()
        .map_while(|c| decimal_value(c).and_then(|v| char::from_digit(v, 10)))
        .collect();
    let width: usize = text
        .chars()
        .take(digits.chars().count())
        .map(char::len_utf8)
        .sum();
    if digits.is_empty() || !text[width..].starts_with(" bytes") {
        return None;
    }
    Int::parse(&digits)
}

/// `DatabricksTable._is_large`: whether the inference sample reads a bounded
/// subset of the table. `detail_size` is the `sizeInBytes` `DESCRIBE DETAIL`
/// gave, if any; `described` what `DESCRIBE TABLE EXTENDED` said, read only
/// without a size, `None` where it failed.
pub fn databricks_is_large(
    detail_size: &Py,
    described: Option<&Described>,
    min_bytes: &Int,
) -> Result<bool, Error> {
    let size = match detail_size {
        Py::None => {
            let Some(described) = described else {
                return Ok(false);
            };
            if described_get(described, "Type").to_uppercase() == "VIEW" {
                return Ok(true);
            }
            match statistics_bytes(described_get(described, "Statistics")) {
                Some(size) => size,
                None => return Ok(false),
            }
        }
        size => py_int(size)?,
    };
    Ok(&size >= min_bytes)
}

/// `PostgresTable._prefiltered_sample_source`: the table with a repeatable
/// sample of its blocks, if it is large and has no primary key. `relation`
/// is what `relation_size` returned, `None` where it failed.
pub fn postgres_prefiltered_sample_source(
    schema: &str,
    source_name: &str,
    primary_key: Option<&str>,
    relation: Option<&[Py]>,
    min_blocks: &Int,
    target_blocks: &Int,
) -> Result<Option<String>, Error> {
    if primary_key.is_some() {
        return Ok(None);
    }
    let Some(found) = relation else {
        return Ok(None);
    };
    let Some(row) = found.first() else {
        return Ok(None);
    };
    let (kind, blocks) = match row {
        Py::Tuple(items) | Py::List(items) if items.len() == 2 => (&items[0], &items[1]),
        _ => return Err(Error::new("a relation_size row is not (kind, blocks)")),
    };
    if !matches!(kind.py_str()?.as_str(), "r" | "m" | "p") || blocks.is_none() {
        return Ok(None);
    }
    let blocks = py_int(blocks)?;
    if &blocks < min_blocks {
        return Ok(None);
    }
    let fraction = Fraction::new(fits(target_blocks)?, fits(&blocks)?)?;
    let Some(fraction) = prefilter_fraction(fraction)? else {
        return Ok(None);
    };
    let table = format!(
        "{}.{}",
        quote_ident(schema, "\""),
        quote_ident(source_name, "\"")
    );
    Ok(Some(postgres::block_sample(
        &table,
        &percent_literal(fraction)?,
    )))
}

fn fits(value: &Int) -> Result<i128, Error> {
    value
        .to_i128()
        .ok_or_else(|| Error::new("an integer too large for a fraction"))
}

/// What `SHOW TABLES` says of a table, by field.
pub type Listing = Vec<(String, Py)>;

fn listing_get<'a>(listing: &'a Listing, field: &str) -> Result<&'a Py, Error> {
    listing
        .iter()
        .find(|(k, _)| k == field)
        .map(|(_, v)| v)
        .ok_or_else(|| Error::new(format!("KeyError: {field:?}")))
}

/// `SnowTable._prefilter`: the clause that samples the table's
/// micro-partitions so a read still sees `rows` rows, or `None` to read it
/// whole.
pub fn snow_prefilter(
    rows: &Int,
    seed: &Int,
    listing: Option<&Listing>,
    sampled_kinds: &[String],
    prefilter_partitions: &Int,
    micro_partition_bytes: &Int,
) -> Result<Option<String>, Error> {
    let Some(listed) = listing else {
        return Ok(None);
    };
    if !sampled_kinds.contains(&folded(listing_get(listed, "kind")?)?) {
        return Ok(None);
    }
    for flag in ["is_external", "is_hybrid", "is_iceberg", "is_dynamic"] {
        if folded(listing_get(listed, flag)?)? == "Y" {
            return Ok(None);
        }
    }
    let read = |field: &str| -> Result<Option<Int>, Error> {
        match py_int(listing_get(listed, field)?) {
            Ok(value) => Ok(Some(value)),
            Err(Refusal::Refused(_)) => Ok(None),
            Err(Refusal::Raised(error)) => Err(error),
        }
    };
    let Some(num_rows) = read("rows")? else {
        return Ok(None);
    };
    let Some(num_bytes) = read("bytes")? else {
        return Ok(None);
    };
    if num_rows <= Int::from(0) {
        return Ok(None);
    }
    let one = Fraction::integer(1);
    let partitions = Fraction::new(fits(&num_bytes)?, fits(micro_partition_bytes)?)?;
    let partitions = if one >= partitions { one } else { partitions };
    let by_rows = Fraction::new(fits(rows)?, fits(&num_rows)?)?;
    let by_partitions = Fraction::new(
        fits(prefilter_partitions)?
            .checked_mul(partitions.denominator)
            .ok_or_else(|| Error::new("a fraction too large"))?,
        partitions.numerator,
    )?;
    let wanted = if by_rows >= by_partitions {
        by_rows
    } else {
        by_partitions
    };
    let Some(fraction) = prefilter_fraction(wanted)? else {
        return Ok(None);
    };
    let seed = Int::from(seed.rem_euclid(1 << 31) as i64);
    Ok(Some(snow::block_sample(&percent_literal(fraction)?, &seed)))
}

/// `SnowTable._listing`: the fields of the one row `SHOW TABLES` lists for
/// `source_name`, or `None`. `listed` is every row it listed, `None` where
/// it failed.
pub fn snow_listing(
    source_name: &str,
    listed: Option<&[Listing]>,
    fields: &[String],
) -> Result<Option<Listing>, Error> {
    let Some(rows) = listed else {
        return Ok(None);
    };
    let name = |row: &Listing| {
        row.iter()
            .find(|(k, _)| k == "name")
            .map(|(_, v)| v.clone())
    };
    let mut named: Vec<&Listing> = rows
        .iter()
        .filter(|row| matches!(name(row), Some(Py::Str(s)) if s == source_name))
        .collect();
    if named.is_empty() {
        let folded_name = source_name.to_uppercase();
        named = Vec::new();
        for row in rows {
            let text = name(row).unwrap_or(Py::None).py_str()?;
            if text.to_uppercase() == folded_name {
                named.push(row);
            }
        }
    }
    let [row] = named.as_slice() else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for field in fields {
        match row.iter().find(|(k, _)| k == field) {
            Some((_, value)) => out.push((field.clone(), value.clone())),
            None => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// How one SQLite table writes its timestamps as text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TimeSpelling {
    pub separator: String,
    pub digits: i64,
    pub suffix: String,
}

impl Default for TimeSpelling {
    fn default() -> Self {
        Self {
            separator: " ".into(),
            digits: 0,
            suffix: String::new(),
        }
    }
}

/// The class `_TimeSpelling` is, for its record.
pub const TIME_SPELLING_CLASS: &str =
    "sdm_relational_engine.rfm.backend.sqlite.sampler._TimeSpelling";

/// `^\d{4}-\d{2}-\d{2}([T ])\d{2}:\d{2}:\d{2}(?:\.(\d+))?(.*)$` against
/// `text`: the separator, the fraction's digits and what follows.
fn stored_time(text: &str) -> Option<(char, usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut at = 0usize;
    let digits = |at: &mut usize, n: usize| -> Option<()> {
        for _ in 0..n {
            decimal_value(*chars.get(*at)?)?;
            *at += 1;
        }
        Some(())
    };
    let literal =
        |at: &mut usize, c: char| -> Option<()> { (chars.get(*at) == Some(&c)).then(|| *at += 1) };
    digits(&mut at, 4)?;
    literal(&mut at, '-')?;
    digits(&mut at, 2)?;
    literal(&mut at, '-')?;
    digits(&mut at, 2)?;
    let separator = *chars.get(at).filter(|c| matches!(c, 'T' | ' '))?;
    at += 1;
    digits(&mut at, 2)?;
    literal(&mut at, ':')?;
    digits(&mut at, 2)?;
    literal(&mut at, ':')?;
    digits(&mut at, 2)?;
    let mut fraction = 0usize;
    if chars.get(at) == Some(&'.') && chars.get(at + 1).and_then(|&c| decimal_value(c)).is_some() {
        at += 1;
        while chars.get(at).and_then(|&c| decimal_value(c)).is_some() {
            fraction += 1;
            at += 1;
        }
    }
    let mut rest: Vec<char> = chars[at..].to_vec();
    // `.` stops at a newline, and `$` matches before a final one.
    if rest.last() == Some(&'\n') {
        rest.pop();
    }
    if rest.contains(&'\n') {
        return None;
    }
    Some((separator, fraction, rest.into_iter().collect()))
}

/// `SQLiteSampler._probe_time_spelling`: how the table writes its times,
/// from one stored value, `found`, the rows the probe returned, `None` where
/// it failed.
pub fn sqlite_probe_time_spelling(time_column: Option<&str>, found: Option<&[Py]>) -> TimeSpelling {
    if time_column.is_none() {
        return TimeSpelling::default();
    }
    let Some(Py::Tuple(row) | Py::List(row)) = found.and_then(|rows| rows.first()) else {
        return TimeSpelling::default();
    };
    let Some(Py::Str(value)) = row.first() else {
        return TimeSpelling::default();
    };
    let Some((separator, digits, suffix)) = stored_time(value) else {
        return TimeSpelling::default();
    };
    TimeSpelling {
        separator: separator.to_string(),
        digits: digits as i64,
        suffix: if matches!(suffix.as_str(), "" | "Z" | "+00:00" | "+0000") {
            suffix
        } else {
            String::new()
        },
    }
}

/// `_TimeSpelling.spell(times)`: each time written as the table writes its
/// own, a missing one as NaN.
pub fn sqlite_spell(spelling: &TimeSpelling, times: &Series) -> Result<Series, Error> {
    let digits = usize::try_from(spelling.digits).unwrap_or(0);
    let values = times
        .values
        .iter()
        .map(|value| {
            let stamp: Stamp = match value {
                Py::NaT | Py::None => return Ok(Py::Float(f64::NAN)),
                Py::Timestamp(stamp) => *stamp,
                other => return Err(Error::new(format!("spell of {other:?}"))),
            };
            let c = stamp.civil;
            if c.year < 1000 {
                return Err(Error::new("a year before 1000 is not ported"));
            }
            let mut text = format!("{}{}{}", c.date_text(), spelling.separator, c.time_text());
            if digits > 0 {
                let fraction = format!("{:09}", c.nanos);
                let fraction = format!("{fraction:0<digits$}");
                text.push('.');
                text.extend(fraction.chars().take(digits));
            }
            text.push_str(&spelling.suffix);
            Ok(Py::Str(text))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(Series {
        dtype: "str".into(),
        values,
        name: times.name.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_times_as_python_reads_them() {
        assert_eq!(
            stored_time("2024-01-01T10:00:00.123Z"),
            Some(('T', 3, "Z".into()))
        );
        assert_eq!(
            stored_time("2024-01-01 10:00:00.5 UTC\n"),
            Some((' ', 1, " UTC".into()))
        );
        assert_eq!(stored_time("2024-01-01 10:00:00\nmore"), None);
        assert_eq!(
            stored_time("2024-01-01 10:00:00."),
            Some((' ', 0, ".".into()))
        );
        assert_eq!(stored_time("2024-01-01"), None);
    }

    #[test]
    fn statistics_read_as_python() {
        assert_eq!(statistics_bytes("12 bytes, 3 rows"), Some(Int::from(12)));
        assert_eq!(statistics_bytes("about 5 bytes"), None);
        assert_eq!(statistics_bytes("12bytes"), None);
    }
}
