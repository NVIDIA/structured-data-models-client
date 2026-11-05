// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Associations between labelled context and its target, independent of model inference.
//! The engine owns eligibility, graph interpretation, cohorts and scores; hosts render rows.
use sdm_contract::{
    CoreError, Sentence,
    columns::{ColumnAnalysis, ColumnRow},
};
use sdm_wire::json::Value;
use std::collections::{BTreeMap, BTreeSet};

const COHORTS: usize = 10;
const CATEGORIES: usize = 500;
const CLASSES: usize = 30;
const OTHER: &str = "OTHER(*)";
const NA: &str = "N/A";
fn invalid(reason: &str) -> CoreError {
    CoreError::new(
        Sentence::InvalidMessage,
        format!("invalid column analysis input: {reason}"),
    )
}
fn field<'a>(v: &'a Value, key: &str) -> Result<&'a Value, CoreError> {
    v.get(key).ok_or_else(|| invalid("missing field"))
}
fn array(v: &Value) -> Result<&[Value], CoreError> {
    match v {
        Value::Array(v) => Ok(v),
        _ => Err(invalid("expected array")),
    }
}
fn members(v: &Value) -> Result<&[(String, Value)], CoreError> {
    match v {
        Value::Object(v) => Ok(v),
        _ => Err(invalid("expected object")),
    }
}
fn text(v: &Value) -> Result<&str, CoreError> {
    v.as_text().ok_or_else(|| invalid("expected text"))
}
fn encoded(v: &Value) -> String {
    String::from_utf8(v.to_bytes()).expect("wire JSON is UTF-8")
}
fn number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) | Value::Text(n) => n.parse::<f64>().ok().filter(|n| n.is_finite()),
        Value::Bool(v) => Some(u8::from(*v).into()),
        _ => None,
    }
}
fn key(v: &Value) -> Result<String, CoreError> {
    crate::result::candidate_key(&crate::result::candidate(v.clone()))?
        .ok_or_else(|| invalid("invalid identity"))
}
fn missing(v: &Value) -> bool {
    matches!(v, Value::Null)
        || matches!(v,Value::Number(n) if n.parse::<f64>().is_ok_and(f64::is_nan))
}
fn display(v: &Value) -> String {
    match v {
        Value::Text(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        _ => encoded(v),
    }
}
fn absent(v: &Value) -> bool {
    missing(v) || matches!(v,Value::Text(s) if s.is_empty())
}
fn listed(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::Text(s)) => vec![s.clone()],
        Some(Value::Array(v)) => v
            .iter()
            .filter_map(Value::as_text)
            .map(str::to_owned)
            .collect(),
        _ => vec![],
    }
}

struct Table<'a> {
    names: Vec<&'a str>,
    rows: Vec<&'a [Value]>,
}
impl<'a> Table<'a> {
    fn read(v: &'a Value) -> Result<Self, CoreError> {
        let names = array(field(v, "columns")?)?
            .iter()
            .map(text)
            .collect::<Result<Vec<_>, _>>()?;
        if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
            return Err(invalid("duplicate columns"));
        }
        let rows = array(field(v, "rows")?)?
            .iter()
            .map(array)
            .collect::<Result<Vec<_>, _>>()?;
        if rows.iter().any(|r| r.len() != names.len()) {
            return Err(invalid("row width"));
        }
        Ok(Self { names, rows })
    }
    fn at(&self, name: &str) -> Result<usize, CoreError> {
        self.names
            .iter()
            .position(|n| *n == name)
            .ok_or_else(|| invalid("column missing from table"))
    }
}

/// Stable frequency order, with exact numeric identity and a defined order for mixed types.
/// Unlike pandas sorting, mixed primitive class labels do not invoke host comparisons.
fn frequencies(values: impl IntoIterator<Item = Value>) -> Result<Vec<Value>, CoreError> {
    let mut counts: BTreeMap<String, (Value, usize)> = BTreeMap::new();
    for v in values {
        if missing(&v) {
            continue;
        }
        let k = key(&v)?;
        let entry = counts.entry(k).or_insert((v, 0));
        entry.1 += 1;
    }
    let mut counts = counts.into_values().collect::<Vec<_>>();
    counts.sort_by(|(a, n), (b, m)| m.cmp(n).then_with(|| scalar_order(a, b)));
    Ok(counts.into_iter().map(|(v, _)| v).collect())
}
fn scalar_order(a: &Value, b: &Value) -> std::cmp::Ordering {
    // Exact integer strings are compared without rounding; finite nonintegral numbers use f64.
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            if sdm_wire::json::is_integer_token(x) && sdm_wire::json::is_integer_token(y) {
                let parts = |v: &str| {
                    let neg = v.starts_with('-');
                    let digits = v.trim_start_matches('-').trim_start_matches('0');
                    (neg && !digits.is_empty(), digits.to_owned())
                };
                let (nx, dx) = parts(x);
                let (ny, dy) = parts(y);
                return nx.cmp(&ny).reverse().then_with(|| {
                    let order = dx.len().cmp(&dy.len()).then_with(|| dx.cmp(&dy));
                    if nx { order.reverse() } else { order }
                });
            }
            number(a)
                .zip(number(b))
                .map(|(x, y)| x.total_cmp(&y))
                .unwrap_or_else(|| x.cmp(y))
                .then_with(|| x.cmp(y))
        }
        (Value::Text(x), Value::Text(y)) => x.cmp(y),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => encoded(a).cmp(&encoded(b)),
    }
}
#[derive(Default)]
struct Cohorts {
    labels: Vec<String>,
    counts: Vec<f64>,
    means: Vec<Vec<f64>>,
    score: f64,
}
fn score(means: &[Vec<f64>], counts: &[f64]) -> f64 {
    let total: f64 = counts.iter().sum();
    if total <= 1.0 || means.is_empty() {
        return 0.0;
    }
    let width = means[0].len();
    if width == 0 {
        return 0.0;
    }
    (0..width)
        .map(|j| {
            let center = means.iter().zip(counts).map(|(m, c)| m[j] * c).sum::<f64>() / total;
            (means
                .iter()
                .zip(counts)
                .map(|(m, c)| c * (m[j] - center).powi(2))
                .sum::<f64>()
                / (total - 1.0))
                .sqrt()
        })
        .sum::<f64>()
        / width as f64
}
fn short(v: f64) -> String {
    let s = format!("{v:.2}");
    s.strip_suffix(".00").unwrap_or(&s).to_owned()
}
fn ranges(values: &[Option<f64>], count: bool, outcome: &[Vec<f64>]) -> Option<Cohorts> {
    let mut sorted = values.iter().flatten().copied().collect::<Vec<_>>();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    let mut bins = (0..=COHORTS)
        .map(|i| {
            let at = i as f64 * (sorted.len() - 1) as f64 / COHORTS as f64;
            let lo = at.floor() as usize;
            let hi = at.ceil() as usize;
            let v = sorted[lo] * (1.0 - (at - lo as f64)) + sorted[hi] * (at - lo as f64);
            if count { v.ceil() } else { v }
        })
        .collect::<Vec<_>>();
    bins.dedup();
    if bins.len() == 1 {
        bins.push(bins[0]);
    }
    let zero = count && bins[0] == 0.0;
    let boundaries = if zero {
        &bins[..bins.len() - 1]
    } else {
        &bins[1..bins.len() - 1]
    };
    let width = if zero { bins.len() } else { bins.len() - 1 };
    let mut counts = vec![0.0; width];
    let mut sums = vec![vec![0.0; outcome[0].len()]; width];
    for (v, y) in values.iter().zip(outcome) {
        if let Some(v) = v {
            let i = boundaries.partition_point(|b| b < v);
            counts[i] += 1.0;
            for (sum, y) in sums[i].iter_mut().zip(y) {
                *sum += y;
            }
        }
    }
    let mut result = Cohorts::default();
    let mut start = bins[0];
    for (i, (&n, sum)) in counts.iter().zip(sums).enumerate() {
        if n == 0.0 {
            continue;
        }
        let end = bins[if zero { i } else { i + 1 }];
        result.labels.push(format!(
            "{}{} - {}]",
            if result.labels.is_empty() { "[" } else { "(" },
            short(start),
            short(end)
        ));
        result.counts.push(n);
        result.means.push(sum.into_iter().map(|s| s / n).collect());
        start = end;
    }
    if count && let Some(label) = result.labels.last_mut() {
        label.pop();
        label.push_str("+]");
    }
    result.score = score(&result.means, &result.counts);
    Some(result)
}
fn categories(
    shares: &[Vec<f64>],
    labels: &[String],
    other: Option<usize>,
    outcome: &[Vec<f64>],
) -> Option<Cohorts> {
    let mut result = Cohorts::default();
    let mut scoring = Vec::new();
    let mut scoring_counts = Vec::new();
    for (i, label) in labels.iter().enumerate() {
        let count = shares.iter().map(|r| r[i]).sum::<f64>();
        if count == 0.0 {
            continue;
        }
        let mean = (0..outcome[0].len())
            .map(|j| {
                shares
                    .iter()
                    .zip(outcome)
                    .map(|(s, y)| s[i] * y[j])
                    .sum::<f64>()
                    / count
            })
            .collect::<Vec<_>>();
        if other != Some(i) {
            scoring.push(mean.clone());
            scoring_counts.push(count);
        }
        result.labels.push(label.clone());
        result.counts.push(count);
        result.means.push(mean);
    }
    if result.labels.is_empty() {
        return None;
    }
    result.score = score(&scoring, &scoring_counts);
    if result.labels.len() > COHORTS {
        let split = COHORTS - 1;
        let total = result.counts[split..].iter().sum::<f64>();
        let means = (0..outcome[0].len())
            .map(|j| {
                result.counts[split..]
                    .iter()
                    .zip(&result.means[split..])
                    .map(|(c, m)| c * m[j])
                    .sum::<f64>()
                    / total
            })
            .collect();
        result.labels.truncate(split);
        result.labels.push(OTHER.into());
        result.counts.truncate(split);
        result.counts.push(total);
        result.means.truncate(split);
        result.means.push(means);
    }
    Some(result)
}
enum TableRole<'a> {
    Instance,
    Related { anchor: Option<&'a [Option<f64>]> },
}

struct Analysis {
    outcome: Vec<Vec<f64>>,
    classes: Vec<Value>,
    columns: Vec<(String, u32, String, Option<String>, Cohorts)>,
}
impl Analysis {
    fn add(
        &mut self,
        table: &str,
        hop: usize,
        column: &str,
        stype: Option<&str>,
        cohorts: Option<Cohorts>,
    ) {
        if let Some(cohorts) = cohorts {
            self.columns.push((
                table.into(),
                hop as u32,
                column.into(),
                stype.map(str::to_owned),
                cohorts,
            ));
        }
    }
    fn finish(mut self) -> ColumnAnalysis {
        if self.columns.iter().any(|c| {
            !c.4.score.is_finite()
                || c.4.counts.iter().any(|v| !v.is_finite())
                || c.4.means.iter().flatten().any(|v| !v.is_finite())
        }) {
            return ColumnAnalysis::Failed {
                reason: "nonfinite_statistics".into(),
            };
        }
        let scale = self.columns.iter().map(|c| c.4.score).fold(0.0, f64::max);
        if scale > 0.0 {
            let total = self.columns.iter().map(|c| c.4.score / scale).sum::<f64>();
            for c in &mut self.columns {
                c.4.score = (c.4.score / scale) / total;
            }
        }
        self.columns.sort_by(|a, b| b.4.score.total_cmp(&a.4.score));
        let mut rows = Vec::new();
        for (table, hop, column, stype, c) in self.columns {
            let total = c.counts.iter().sum::<f64>();
            for (i, cohort) in c.labels.iter().enumerate() {
                for (j, class) in self.classes.iter().enumerate() {
                    rows.push(ColumnRow {
                        table: table.clone(),
                        hop,
                        column: column.clone(),
                        stype: stype.clone(),
                        cohort: cohort.clone(),
                        class_json: encoded(class),
                        share: c.counts[i] / total,
                        mean: c.means[i][j],
                        score: c.score,
                    });
                }
            }
        }
        ColumnAnalysis::Ready { rows }
    }
    fn table(
        &mut self,
        name: &str,
        table: &Table<'_>,
        described: &Value,
        selected: &[(usize, usize, usize)],
        excluded: &BTreeSet<String>,
        role: TableRole<'_>,
    ) -> Result<(), CoreError> {
        let n = self.outcome.len();
        let hops = selected
            .iter()
            .map(|(_, _, hop)| *hop)
            .collect::<BTreeSet<_>>();
        let stypes = members(field(described, "columns")?)?;
        for hop in hops {
            let rows = selected
                .iter()
                .filter(|(_, _, h)| *h == hop)
                .map(|(r, e, _)| (*r, *e))
                .collect::<Vec<_>>();
            if matches!(role, TableRole::Related { .. }) {
                let mut counts = vec![Some(0.0); n];
                for (_, e) in &rows {
                    *counts[*e].as_mut().unwrap() += 1.0;
                }
                self.add(
                    name,
                    hop,
                    "COUNT(*)",
                    None,
                    ranges(&counts, true, &self.outcome),
                );
            }
            for (column, spec) in stypes {
                if excluded.contains(column) {
                    continue;
                }
                let Some(stype) = spec.get("stype").and_then(Value::as_text) else {
                    continue;
                };
                if !matches!(
                    stype,
                    "numerical" | "categorical" | "ID" | "multicategorical" | "timestamp"
                ) {
                    continue;
                }
                let Ok(at) = table.at(column) else {
                    continue;
                };
                let values = selected
                    .iter()
                    .map(|(r, _, _)| table.rows[*r][at].clone())
                    .collect::<Vec<_>>();
                if stype == "numerical" {
                    let mut sums = vec![0.0; n];
                    let mut counts = vec![0; n];
                    for (r, e) in &rows {
                        if let Some(v) = number(&table.rows[*r][at]) {
                            sums[*e] += v;
                            counts[*e] += 1;
                        }
                    }
                    let means = sums
                        .iter()
                        .zip(counts)
                        .map(|(sum, n)| (n > 0).then(|| sum / n as f64))
                        .collect::<Vec<_>>();
                    self.add(
                        name,
                        hop,
                        column,
                        Some(stype),
                        ranges(&means, false, &self.outcome),
                    );
                } else if stype == "timestamp" {
                    if let TableRole::Related {
                        anchor: Some(anchor),
                    } = role
                    {
                        self.times(name, hop, column, table, at, &rows, anchor)?;
                    }
                } else {
                    self.category(name, hop, column, stype, table, at, &rows, &values)?;
                }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn category(
        &mut self,
        name: &str,
        hop: usize,
        column: &str,
        stype: &str,
        table: &Table<'_>,
        at: usize,
        rows: &[(usize, usize)],
        all: &[Value],
    ) -> Result<(), CoreError> {
        let present = all
            .iter()
            .filter(|v| !absent(v))
            .cloned()
            .collect::<Vec<_>>();
        if present.is_empty() {
            return Ok(());
        }
        let multi = stype == "multicategorical";
        let separator = if multi {
            let mut counts: Vec<(char, usize)> = Vec::new();
            for value in present.iter().take(1000) {
                for c in display(value).chars() {
                    if ",;:|\t".contains(c) {
                        if let Some((_, n)) = counts.iter_mut().find(|(k, _)| *k == c) {
                            *n += 1;
                        } else {
                            counts.push((c, 1));
                        }
                    }
                }
            }
            counts
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(std::cmp::Ordering::Greater))
                .map(|(c, _)| c)
        } else {
            None
        };
        let cells = |v: &Value| -> Vec<Value> {
            if absent(v) {
                vec![]
            } else if let Some(sep) = separator {
                display(v)
                    .split(sep)
                    .filter(|s| !s.is_empty())
                    .map(|s| Value::Text(s.into()))
                    .collect()
            } else {
                vec![v.clone()]
            }
        };
        let mut mapping = frequencies(if multi {
            present.iter().flat_map(&cells).collect()
        } else {
            present.clone()
        })?;
        if multi && mapping.len() > CATEGORIES {
            return Ok(());
        }
        if !multi && present.len() >= 100 {
            let unique = mapping.len();
            let n = present.len();
            if stype == "ID" && unique > 4000 {
                if unique as f64 > 0.8 * n as f64 {
                    return Ok(());
                }
            } else {
                let mut freq = BTreeMap::<String, usize>::new();
                for v in &present {
                    *freq.entry(key(v)?).or_default() += 1;
                }
                let min = (unique as f64).sqrt().floor().max(2.0) as usize;
                let passing = freq.values().filter(|&&v| v > min).count();
                let threshold = (0.3 * (n as f64).sqrt()).floor() as usize;
                if passing > 4000 || passing > 20 * threshold {
                    return Ok(());
                }
            }
        }
        let other = (!multi && mapping.len() > CATEGORIES).then_some(CATEGORIES - 1);
        if other.is_some() {
            mapping.truncate(CATEGORIES - 1);
        }
        let mut labels = mapping.iter().map(display).collect::<Vec<_>>();
        if other.is_some() {
            labels.push(OTHER.into());
        }
        let map = mapping
            .iter()
            .enumerate()
            .map(|(i, v)| Ok((key(v)?, i)))
            .collect::<Result<BTreeMap<_, _>, CoreError>>()?;
        let na = labels.len();
        if !multi {
            labels.push(NA.into());
        }
        let mut shares = vec![vec![0.0; labels.len()]; self.outcome.len()];
        let mut counts = vec![0; self.outcome.len()];
        for (r, e) in rows {
            counts[*e] += 1;
            let v = &table.rows[*r][at];
            let values = if multi { cells(v) } else { vec![v.clone()] };
            for v in values {
                let i = if absent(&v) {
                    na
                } else {
                    *map.get(&key(&v)?).unwrap_or(&other.unwrap_or(na))
                };
                if i < labels.len() {
                    shares[*e][i] += 1.0;
                }
            }
        }
        for (row, n) in shares.iter_mut().zip(counts) {
            if n > 0 {
                for v in row {
                    *v /= n as f64;
                }
            }
        }
        self.add(
            name,
            hop,
            column,
            Some(stype),
            categories(&shares, &labels, other, &self.outcome),
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn times(
        &mut self,
        name: &str,
        hop: usize,
        column: &str,
        table: &Table<'_>,
        at: usize,
        rows: &[(usize, usize)],
        anchor: &[Option<f64>],
    ) -> Result<(), CoreError> {
        let ages = rows
            .iter()
            .filter_map(|(r, e)| Some((*e, anchor[*e]? - seconds(&table.rows[*r][at])?)))
            .filter(|(_, v)| *v >= 0.0)
            .collect::<Vec<_>>();
        let oldest = ages.iter().map(|(_, v)| *v).fold(0.0, f64::max).floor();
        let (unit, scale) = if oldest == 0.0 {
            ("months", 2592000.0)
        } else if oldest / 3600.0 < 70.0 {
            ("hours", 3600.0)
        } else if oldest / 86400.0 < 50.0 {
            ("days", 86400.0)
        } else if oldest / 604800.0 < 30.0 {
            ("weeks", 604800.0)
        } else {
            ("months", 2592000.0)
        };
        let window = (oldest / (scale * COHORTS as f64)).ceil().max(1.0);
        let labels = (0..COHORTS)
            .map(|i| {
                format!(
                    "[{} {unit}, {} {unit}{}",
                    i as f64 * window,
                    (i + 1) as f64 * window,
                    if i == COHORTS - 1 { "]" } else { ")" }
                )
            })
            .collect::<Vec<_>>();
        let mut shares = vec![vec![0.0; COHORTS]; self.outcome.len()];
        let mut counts = vec![0; self.outcome.len()];
        for (e, age) in ages {
            shares[e][((age / (scale * window)).floor() as usize).min(COHORTS - 1)] += 1.0;
            counts[e] += 1;
        }
        for (s, n) in shares.iter_mut().zip(counts) {
            if n > 0 {
                for v in s {
                    *v /= n as f64;
                }
            }
        }
        self.add(
            name,
            hop,
            column,
            Some("timestamp"),
            categories(&shares, &labels, None, &self.outcome),
        );
        Ok(())
    }
}
fn seconds(value: &Value) -> Option<f64> {
    match value {
        Value::Number(v) => v
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(|v| (v / 1e9).floor()),
        Value::Text(v)
            if v.as_bytes()
                .get(4)
                .is_some_and(|b| *b == b'-' || *b == b'/')
                && v.as_bytes()[..4].iter().all(u8::is_ascii_digit) =>
        {
            sdm_engine::time::text::Reader::new(2000)
                .column(&[Some(v)], true)
                .ok()?
                .first()?
                .map(|v| v.nanos().div_euclid(1_000_000_000) as f64)
        }
        _ => None,
    }
}

/// Analyze a single already normalized request. Invalid structure is an input error.
/// Empty labelled context produces a valid empty result, including classification.
pub fn analyze(payload: &Value) -> Result<ColumnAnalysis, CoreError> {
    let task = field(payload, "task")?;
    let kind = text(field(task, "kind")?)?;
    if !matches!(
        kind,
        "regression" | "classification" | "binary_classification" | "multiclass_classification"
    ) {
        return Ok(ColumnAnalysis::Unsupported {
            reason: "column explanations require regression or classification".into(),
        });
    }
    let schema = field(payload, "schema")?;
    let context = field(payload, "context")?;
    let instance = Table::read(field(context, "instance_table")?)?;
    let target = text(field(field(task, "target")?, "column_name")?)?;
    let at = instance.at(target)?;
    let kept = instance
        .rows
        .iter()
        .enumerate()
        .filter(|(_, r)| !missing(&r[at]))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if kept.is_empty() {
        return Ok(ColumnAnalysis::Ready { rows: vec![] });
    }
    let targets = kept
        .iter()
        .map(|&i| instance.rows[i][at].clone())
        .collect::<Vec<_>>();
    let classes = if kind == "regression" {
        vec![Value::Null]
    } else {
        frequencies(targets.clone())?
            .into_iter()
            .take(CLASSES)
            .collect()
    };
    let class_keys = classes.iter().map(key).collect::<Result<Vec<_>, _>>()?;
    if kind == "regression" && targets.iter().any(|v| number(v).is_none()) {
        return Ok(ColumnAnalysis::Failed {
            reason: "target_not_finite_numeric".into(),
        });
    }
    let outcome = targets
        .iter()
        .map(|v| {
            if kind == "regression" {
                number(v)
                    .map(|v| vec![v])
                    .ok_or_else(|| invalid("regression target is not finite numeric"))
            } else {
                let k = key(v)?;
                Ok(class_keys
                    .iter()
                    .map(|c| if *c == k { 1.0 } else { 0.0 })
                    .collect())
            }
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    let mut analysis = Analysis {
        outcome,
        classes,
        columns: vec![],
    };
    if let Some(related) = context.get("related_tables") {
        let hops = crate::gradients::context_hops(schema, context)?;
        let id = instance.at("instance_id")?;
        let mut entities = BTreeMap::new();
        for (e, &r) in kept.iter().enumerate() {
            if entities.insert(key(&instance.rows[r][id])?, e).is_some() {
                return Err(invalid("duplicate instance identity"));
            }
        }
        let anchor = task
            .get("anchor_time_column")
            .and_then(Value::as_text)
            .and_then(|name| instance.at(name).ok())
            .map(|i| {
                kept.iter()
                    .map(|&r| seconds(&instance.rows[r][i]))
                    .collect::<Vec<_>>()
            });
        let mut excluded: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        if let Some(links) = schema.get("relationships") {
            for link in array(links)? {
                for side in ["source", "target"] {
                    if let Some(name) = link.get(&format!("{side}_table")).and_then(Value::as_text)
                    {
                        excluded
                            .entry(name.into())
                            .or_default()
                            .extend(listed(link.get(&format!("{side}_columns"))));
                    }
                }
            }
        }
        for (name, value) in members(related)? {
            let table = Table::read(value)?;
            let id = table.at("instance_id")?;
            let described = field(field(schema, "related_tables")?, name)?;
            let mut exclusions = excluded.remove(name).unwrap_or_default();
            exclusions.insert("instance_id".into());
            exclusions.extend(listed(described.get("primary_key")));
            let distances = hops
                .get(name.as_str())
                .ok_or_else(|| invalid("missing related distances"))?;
            let mut selected = vec![];
            for (r, row) in table.rows.iter().enumerate() {
                if let (Some(hop), Some(&entity)) = (distances[r], entities.get(&key(&row[id])?)) {
                    selected.push((r, entity, hop));
                }
            }
            analysis.table(
                name,
                &table,
                described,
                &selected,
                &exclusions,
                TableRole::Related {
                    anchor: anchor.as_deref(),
                },
            )?;
        }
    } else {
        let described = field(schema, "instance_table")?;
        let mut excluded = listed(described.get("primary_key"))
            .into_iter()
            .collect::<BTreeSet<_>>();
        excluded.insert(target.into());
        let selected = kept
            .iter()
            .enumerate()
            .map(|(e, &r)| (r, e, 0))
            .collect::<Vec<_>>();
        analysis.table(
            "instance_table",
            &instance,
            described,
            &selected,
            &excluded,
            TableRole::Instance,
        )?;
    }
    Ok(analysis.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(s: &str) -> Value {
        sdm_wire::json::read(s.as_bytes()).unwrap()
    }
    fn payload(rows: &str, kind: &str) -> Value {
        read(&format!(
            r#"{{"task":{{"kind":"{kind}","target":{{"column_name":"y"}}}},"schema":{{"instance_table":{{"columns":{{"x":{{"stype":"categorical"}},"y":{{"stype":"numerical"}}}}}}}},"context":{{"instance_table":{{"columns":["x","y"],"rows":{rows}}}}}}}"#
        ))
    }
    #[test]
    fn independent_two_cohort_oracle() {
        let ColumnAnalysis::Ready { rows } = analyze(&payload(
            r#"[["a",0],["a",2],["b",8],["b",10]]"#,
            "regression",
        ))
        .unwrap() else {
            panic!()
        };
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cohort, "a");
        assert_eq!(rows[0].mean, 1.0);
        assert_eq!(rows[1].mean, 9.0);
        assert_eq!(rows[0].share, 0.5);
        assert_eq!(rows[0].score, 1.0);
        assert!(
            (score(&[vec![1.0], vec![9.0]], &[2.0, 2.0]) - (64.0_f64 / 3.0).sqrt()).abs() < 1e-12
        );
    }
    #[test]
    fn empty_classification_is_ready_empty() {
        assert_eq!(
            analyze(&payload(r#"[["a",null]]"#, "classification")).unwrap(),
            ColumnAnalysis::Ready { rows: vec![] }
        );
    }
    #[test]
    fn mixed_classes_are_exact_and_deterministic() {
        let ColumnAnalysis::Ready { rows } = analyze(&payload(
            r#"[["a",1],["a","1"],["b",9007199254740993],["b",9007199254740992]]"#,
            "classification",
        ))
        .unwrap() else {
            panic!()
        };
        let labels = rows[..4]
            .iter()
            .map(|r| r.class_json.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            labels,
            BTreeSet::from(["1", "\"1\"", "9007199254740992", "9007199254740993"])
        );
    }
    #[test]
    fn constant_score_is_zero() {
        let ColumnAnalysis::Ready { rows } =
            analyze(&payload(r#"[["a",2],["b",2]]"#, "regression")).unwrap()
        else {
            panic!()
        };
        assert!(rows.iter().all(|r| r.score == 0.0));
    }
    #[test]
    fn exact_wide_negative_frequency_ties() {
        let a = Value::Number("-10000000000000000000000000000000000000001".into());
        let b = Value::Number("-10000000000000000000000000000000000000002".into());
        assert_eq!(frequencies(vec![a.clone(), b.clone()]).unwrap(), vec![b, a]);
    }
    #[test]
    fn invalid_regression_target_is_recoverable() {
        assert_eq!(
            analyze(&payload(r#"[["a","not a number"]]"#, "regression")).unwrap(),
            ColumnAnalysis::Failed {
                reason: "target_not_finite_numeric".into()
            }
        );
    }
    #[test]
    fn finite_extreme_targets_never_emit_nonfinite_statistics() {
        let result = analyze(&payload(r#"[["a",1e308],["b",-1e308]]"#, "regression")).unwrap();
        let raw = serde_json::to_string(&result).unwrap();
        validate(&raw).unwrap();
        assert!(matches!(result, ColumnAnalysis::Failed { .. }));
    }
    #[test]
    fn tied_separator_uses_first_encounter() {
        let mut p = payload(r#"[["a|b,c",0],["d|e,f",1]]"#, "regression");
        let mut schema = p.get("schema").unwrap().clone();
        let mut t = schema.get("instance_table").unwrap().clone();
        let mut cols = t.get("columns").unwrap().clone();
        cols.set("x", read(r#"{"stype":"multicategorical"}"#));
        t.set("columns", cols);
        schema.set("instance_table", t);
        p.set("schema", schema);
        let ColumnAnalysis::Ready { rows } = analyze(&p).unwrap() else {
            panic!()
        };
        assert!(rows.iter().any(|r| r.cohort == "b,c"));
        assert!(!rows.iter().any(|r| r.cohort == "a|b"));
    }
    #[test]
    fn malformed_analysis_is_not_absence() {
        assert!(validate(r#"{"status":"mystery"}"#).is_err());
        assert!(validate(r#"{"status":"ready","rows":[{"table":"t","hop":0,"column":"x","stype":null,"cohort":"a","class_json":"[]","share":0.5,"mean":1,"score":1}]}"#).is_err());
    }
}

/// Check an analysis supplied by a host before using it in a semantic merge.
pub fn validate(raw: &str) -> Result<ColumnAnalysis, CoreError> {
    let analysis: ColumnAnalysis =
        serde_json::from_str(raw).map_err(|_| invalid("malformed analysis"))?;
    if let ColumnAnalysis::Ready { rows } = &analysis {
        for row in rows {
            let label = sdm_wire::json::read(row.class_json.as_bytes())
                .map_err(|_| invalid("invalid class scalar"))?;
            if matches!(label, Value::Array(_) | Value::Object(_))
                || !row.share.is_finite()
                || !(0.0..=1.0).contains(&row.share)
                || !row.mean.is_finite()
                || !row.score.is_finite()
                || !(0.0..=1.0).contains(&row.score)
            {
                return Err(invalid("invalid cohort statistics"));
            }
        }
    }
    Ok(analysis)
}

/// Restore only excluded rows, validating the delta before any analysis or model call.
pub fn restored(payload: &Value) -> Result<Value, CoreError> {
    let Some(delta) = payload.get("client_analysis_delta") else {
        return Ok(payload.clone());
    };
    if delta.get("version") != Some(&Value::int(1)) {
        return Err(CoreError::protocol(
            Sentence::StateUnread,
            "unsupported column context delta",
        ));
    }
    if text(field(delta, "schema_json")?)? != encoded(field(payload, "schema")?) {
        return Err(CoreError::protocol(
            Sentence::StateUnread,
            "column context delta schema differs",
        ));
    }
    fn restore(table: &Value, delta: &Value) -> Result<Value, CoreError> {
        let removed = field(delta, "removed")?;
        if field(table, "columns")? != field(removed, "columns")? {
            return Err(CoreError::protocol(
                Sentence::StateUnread,
                "column context delta columns differ",
            ));
        }
        let total = match field(delta, "original_rows")? {
            Value::Number(n) => n.parse::<usize>().map_err(|_| invalid("delta row count"))?,
            _ => return Err(invalid("delta row count")),
        };
        let positions = array(field(delta, "positions")?)?
            .iter()
            .map(|v| match v {
                Value::Number(n) => n.parse::<usize>().map_err(|_| invalid("delta position")),
                _ => Err(invalid("delta position")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let before = Table::read(table)?;
        let dropped = Table::read(removed)?;
        if positions.len() != dropped.rows.len()
            || before.rows.len().checked_add(positions.len()) != Some(total)
            || positions.windows(2).any(|p| p[0] >= p[1])
            || positions.last().is_some_and(|p| *p >= total)
        {
            return Err(CoreError::protocol(
                Sentence::StateUnread,
                "invalid column context delta positions",
            ));
        }
        let mut kept = before.rows.into_iter();
        let mut dropped = dropped.rows.into_iter();
        let mut at = positions.into_iter().peekable();
        let mut rows = Vec::with_capacity(total);
        for i in 0..total {
            let row = if at.peek() == Some(&i) {
                at.next();
                dropped.next()
            } else {
                kept.next()
            }
            .ok_or_else(|| invalid("delta rows exhausted"))?;
            rows.push(Value::Array(row.to_vec()));
        }
        let mut result = table.clone();
        result.set("rows", Value::Array(rows));
        Ok(result)
    }
    let context = field(payload, "context")?;
    let mut combined = context.clone();
    combined.set(
        "instance_table",
        restore(
            field(context, "instance_table")?,
            field(delta, "instance_table")?,
        )?,
    );
    let original = members(field(context, "related_tables")?)?;
    let deltas = members(field(delta, "related_tables")?)?;
    if original.len() != deltas.len() || original.iter().zip(deltas).any(|((a, _), (b, _))| a != b)
    {
        return Err(CoreError::protocol(
            Sentence::StateUnread,
            "column context delta tables differ",
        ));
    }
    combined.set(
        "related_tables",
        Value::Object(
            original
                .iter()
                .zip(deltas)
                .map(|((name, t), (_, d))| Ok((name.clone(), restore(t, d)?)))
                .collect::<Result<_, CoreError>>()?,
        ),
    );
    let mut result = payload.clone();
    result.set("context", combined);
    result.remove("client_analysis_delta");
    Ok(result)
}

/// Union forecast evidence before cohort selection. Steps namespace instance identity.
pub fn forecast(payloads: &[Value]) -> Result<Value, CoreError> {
    let Some(first) = payloads.first() else {
        return Err(invalid("empty analysis request"));
    };
    let mut out = first.clone();
    let mut contexts = Vec::new();
    let mut next_id = 0i64;
    let mut schema = field(first, "schema")?.clone();
    for payload in payloads {
        if field(field(payload, "task")?, "kind")? != field(field(first, "task")?, "kind")?
            || field(field(payload, "task")?, "target")? != field(field(first, "task")?, "target")?
        {
            return Err(invalid("forecast tasks differ"));
        }
        let mut context = field(payload, "context")?.clone();
        let instance = Table::read(field(&context, "instance_table")?)?;
        if let Ok(id) = instance.at("instance_id") {
            let mut ids = BTreeMap::new();
            for row in &instance.rows {
                let k = key(&row[id])?;
                if ids.insert(k, next_id).is_some() {
                    return Err(invalid("duplicate forecast instance"));
                }
                next_id = next_id
                    .checked_add(1)
                    .ok_or_else(|| invalid("forecast too large"))?;
            }
            fn rename(table: &mut Value, ids: &BTreeMap<String, i64>) -> Result<(), CoreError> {
                let read = Table::read(table)?;
                let id = read.at("instance_id")?;
                let rows = read
                    .rows
                    .iter()
                    .map(|r| {
                        let mut row = r.to_vec();
                        row[id] = Value::int(
                            *ids.get(&key(&r[id])?)
                                .ok_or_else(|| invalid("unknown forecast instance"))?,
                        );
                        Ok(Value::Array(row))
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?;
                table.set("rows", Value::Array(rows));
                Ok(())
            }
            let mut table = field(&context, "instance_table")?.clone();
            rename(&mut table, &ids)?;
            context.set("instance_table", table);
            if let Some(related) = context.get("related_tables") {
                let tables = members(related)?
                    .iter()
                    .map(|(name, t)| {
                        let mut t = t.clone();
                        rename(&mut t, &ids)?;
                        Ok((name.clone(), t))
                    })
                    .collect::<Result<_, CoreError>>()?;
                context.set("related_tables", Value::Object(tables));
            }
        }
        let step_schema = field(payload, "schema")?;
        if step_schema.get("instance_table") != schema.get("instance_table")
            || field(payload, "task")?.get("anchor_time_column")
                != field(first, "task")?.get("anchor_time_column")
        {
            return Err(invalid("forecast instance schema or anchor differs"));
        }
        for name in ["related_tables"] {
            if let Some(value) = step_schema.get(name) {
                let mut combined = schema.get(name).cloned().unwrap_or(Value::Object(vec![]));
                for (table, desc) in members(value)? {
                    if let Some(existing) = combined.get(table) {
                        if existing != desc {
                            return Err(invalid("forecast table schemas differ"));
                        }
                    } else {
                        combined.set(table, desc.clone());
                    }
                }
                schema.set(name, combined);
            }
        }
        if let Some(value) = step_schema.get("relationships") {
            let mut links = schema
                .get("relationships")
                .map(array)
                .transpose()?
                .unwrap_or(&[])
                .to_vec();
            for link in array(value)? {
                if !links.contains(link) {
                    links.push(link.clone());
                }
            }
            schema.set("relationships", Value::Array(links));
        }
        contexts.push(context);
    }
    fn append(target: &mut Value, source: &Value) -> Result<(), CoreError> {
        if field(target, "columns")? != field(source, "columns")? {
            return Err(invalid("forecast columns differ"));
        }
        let mut rows = array(field(target, "rows")?)?.to_vec();
        rows.extend_from_slice(array(field(source, "rows")?)?);
        target.set("rows", Value::Array(rows));
        Ok(())
    }
    let mut combined = contexts.remove(0);
    for context in contexts {
        let mut table = field(&combined, "instance_table")?.clone();
        append(&mut table, field(&context, "instance_table")?)?;
        combined.set("instance_table", table);
        if let Some(related) = context.get("related_tables") {
            let mut tables = combined
                .get("related_tables")
                .cloned()
                .unwrap_or(Value::Object(vec![]));
            for (name, table) in members(related)? {
                let mut target = tables.get(name).cloned().unwrap_or_else(|| {
                    let mut t = table.clone();
                    t.set("rows", Value::Array(vec![]));
                    t
                });
                append(&mut target, table)?;
                tables.set(name, target);
            }
            combined.set("related_tables", tables);
        }
    }
    out.set("context", combined);
    out.set("schema", schema);
    Ok(out)
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    fn parse(text: &str) -> Value {
        sdm_wire::json::read(text.as_bytes()).unwrap()
    }

    #[test]
    fn delta_restores_exact_positions_with_duplicate_values_and_unicode_schema() {
        let mut p = parse(
            r#"{"schema":{"instance_table":{"columns":{"é\n":{"stype":"categorical"}}}},"context":{"instance_table":{"columns":["é\n"],"rows":[["kept"],["kept"]]},"related_tables":{}}}"#,
        );
        let mut delta = parse(
            r#"{"version":1,"instance_table":{"original_rows":5,"positions":[0,2,4],"removed":{"columns":["é\n"],"rows":[["first"],["middle"],["last"]]}},"related_tables":{}}"#,
        );
        delta.set(
            "schema_json",
            Value::Text(encoded(p.get("schema").unwrap())),
        );
        p.set("client_analysis_delta", delta);
        let restored = restored(&p).unwrap();
        assert_eq!(
            restored
                .get("context")
                .unwrap()
                .get("instance_table")
                .unwrap()
                .get("rows")
                .unwrap(),
            &parse(r#"[["first"],["kept"],["middle"],["kept"],["last"]]"#)
        );
        assert!(restored.get("client_analysis_delta").is_none());
    }

    #[test]
    fn forecast_rejects_changed_instance_semantics_or_anchor() {
        let a = parse(
            r#"{"task":{"kind":"regression","target":{"column_name":"y"}},"schema":{"instance_table":{"columns":{"x":{"stype":"categorical"},"y":{"stype":"numerical"}}}},"context":{"instance_table":{"columns":["x","y"],"rows":[["a",1]]}}}"#,
        );
        let mut b = a.clone();
        let mut task = b.get("task").unwrap().clone();
        task.set("anchor_time_column", Value::Text("x".into()));
        b.set("task", task);
        assert!(forecast(&[a.clone(), b]).is_err());
        let mut b = a.clone();
        b.set("schema", parse(r#"{"instance_table":{"columns":{"x":{"stype":"numerical"},"y":{"stype":"numerical"}}}}"#));
        assert!(forecast(&[a, b]).is_err());
    }
}
