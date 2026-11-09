// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Associations between labelled context and its target, independent of model inference.
//! The engine owns eligibility, graph interpretation and cohorts; hosts render rows.
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
const ENTITY_REF: &str = "__kumo_entity_ref";
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
}
fn short(v: f64, places: usize) -> String {
    let s = format!("{v:.places$}");
    match s.split_once('.') {
        Some((whole, fraction)) if fraction.bytes().all(|b| b == b'0') => whole.to_owned(),
        _ => s,
    }
}
/// The fewest decimals, from 2 to 6, at which adjacent distinct edges print
/// apart.
fn places(edges: &[f64]) -> usize {
    (2..=6)
        .find(|&places| {
            edges
                .windows(2)
                .all(|w| w[0] == w[1] || short(w[0], places) != short(w[1], places))
        })
        .unwrap_or(6)
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
    let places = places(&bins);
    let mut start = bins[0];
    for (i, (&n, sum)) in counts.iter().zip(sums).enumerate() {
        if n == 0.0 {
            continue;
        }
        let end = bins[if zero { i } else { i + 1 }];
        result.labels.push(format!(
            "{}{} - {}]",
            if result.labels.is_empty() { "[" } else { "(" },
            short(start, places),
            short(end, places)
        ));
        result.counts.push(n);
        result.means.push(sum.into_iter().map(|s| s / n).collect());
        start = end;
    }
    if count && let Some(label) = result.labels.last_mut() {
        label.pop();
        label.push_str("+]");
    }
    Some(result)
}
fn categories(shares: &[Vec<f64>], labels: &[String], outcome: &[Vec<f64>]) -> Option<Cohorts> {
    let mut result = Cohorts::default();
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
        result.labels.push(label.clone());
        result.counts.push(count);
        result.means.push(mean);
    }
    if result.labels.is_empty() {
        return None;
    }
    if result.labels.len() > COHORTS {
        // Missing values keep a cohort of their own, outside the merge.
        let na = usize::from(result.labels.last().is_some_and(|l| l == NA));
        let split = COHORTS - 1 - na;
        let end = result.labels.len() - na;
        let total = result.counts[split..end].iter().sum::<f64>();
        let means = (0..outcome[0].len())
            .map(|j| {
                result.counts[split..end]
                    .iter()
                    .zip(&result.means[split..end])
                    .map(|(c, m)| c * m[j])
                    .sum::<f64>()
                    / total
            })
            .collect();
        result.labels.splice(split..end, [OTHER.to_owned()]);
        result.counts.splice(split..end, [total]);
        result.means.splice(split..end, [means]);
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
    fn finish(self) -> ColumnAnalysis {
        if self.columns.iter().any(|c| {
            c.4.counts.iter().any(|v| !v.is_finite())
                || c.4.means.iter().flatten().any(|v| !v.is_finite())
        }) {
            return ColumnAnalysis::Failed {
                reason: "nonfinite_statistics".into(),
            };
        }
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
            categories(&shares, &labels, &self.outcome),
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
            categories(&shares, &labels, &self.outcome),
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
    }
    #[test]
    fn adjacent_numeric_edges_get_distinct_labels() {
        for (values, first, last) in [
            (
                (1..=11).map(|i| f64::from(i) / 1000.0).collect::<Vec<_>>(),
                "[0.001 - 0.002]",
                "(0.010 - 0.011]",
            ),
            (
                (1..=11).map(|i| 10000.0 + f64::from(i) / 1000.0).collect(),
                "[10000.001 - 10000.002]",
                "(10000.010 - 10000.011]",
            ),
        ] {
            let values = values.into_iter().map(Some).collect::<Vec<_>>();
            let outcome = vec![vec![1.0]; values.len()];
            let labels = ranges(&values, false, &outcome).unwrap().labels;
            assert_eq!(labels.first().unwrap(), first);
            assert_eq!(labels.last().unwrap(), last);
            assert_eq!(labels.iter().collect::<BTreeSet<_>>().len(), labels.len());
        }
    }
    #[test]
    fn missing_values_keep_their_own_cohort_beside_the_merged_tail() {
        let rows = (0..12)
            .map(|i| format!(r#"["{}",{i}]"#, char::from(b'a' + i)))
            .chain([r#"[null,20]"#.to_owned()])
            .collect::<Vec<_>>()
            .join(",");
        let ColumnAnalysis::Ready { rows } =
            analyze(&payload(&format!("[{rows}]"), "regression")).unwrap()
        else {
            panic!()
        };
        let cohorts = rows.iter().map(|r| r.cohort.as_str()).collect::<Vec<_>>();
        assert_eq!(cohorts.len(), COHORTS);
        assert_eq!(cohorts[COHORTS - 2..], [OTHER, NA]);
        assert_eq!(rows[COHORTS - 1].mean, 20.0);
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
        let result = analyze(&payload(
            r#"[["a",1e308],["a",1e308],["b",-1e308],["b",-1e308]]"#,
            "regression",
        ))
        .unwrap();
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
        assert!(validate(r#"{"status":"ready","rows":[{"table":"t","hop":0,"column":"x","stype":null,"cohort":"a","class_json":"[]","share":0.5,"mean":1}]}"#).is_err());
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
            {
                return Err(invalid("invalid cohort statistics"));
            }
        }
    }
    Ok(analysis)
}

/// The labelled rows several calls of one request send, as one context:
/// each instance, keyed by its entity and anchor, is read once, from the
/// first call that sends it, under an id of its own, since calls gathered
/// apart can use one instance id for different entities.
pub fn sent(payloads: &[Value]) -> Result<Value, CoreError> {
    let Some(first) = payloads.first() else {
        return Err(invalid("empty analysis request"));
    };
    let entity_of = |table: &Table<'_>| table.names.iter().position(|c| c.starts_with(ENTITY_REF));
    let first_context = field(first, "context")?;
    if payloads.len() == 1
        || entity_of(&Table::read(field(first_context, "instance_table")?)?).is_none()
    {
        return Ok(first.clone());
    }
    let mut seen = BTreeSet::new();
    let mut next_id = 0i64;
    let mut instances = Vec::new();
    let mut related: Vec<(String, Value)> = Vec::new();
    let mut schema = field(first, "schema")?.clone();
    for payload in payloads {
        let context = field(payload, "context")?;
        let table = Table::read(field(context, "instance_table")?)?;
        let id = table.at("instance_id")?;
        let entity = entity_of(&table).ok_or_else(|| invalid("a call names no entity"))?;
        let anchor = field(payload, "task")?
            .get("anchor_time_column")
            .and_then(Value::as_text)
            .and_then(|name| table.at(name).ok());
        let mut keys = Vec::new();
        let mut kept = BTreeMap::new();
        for row in &table.rows {
            let at = (
                key(&row[entity])?,
                anchor.map(|a| key(&row[a])).transpose()?,
            );
            if !seen.contains(&at) {
                kept.insert(key(&row[id])?, next_id);
                let mut row = row.to_vec();
                row[id] = Value::int(next_id);
                instances.push(Value::Array(row));
                next_id = next_id
                    .checked_add(1)
                    .ok_or_else(|| invalid("analysis too large"))?;
            }
            keys.push(at);
        }
        seen.extend(keys);
        if let Some(tables) = context.get("related_tables") {
            for (name, rows) in members(tables)? {
                let read = Table::read(rows)?;
                let at = read.at("instance_id")?;
                let held = match related.iter_mut().find(|(n, _)| n == name) {
                    Some((_, held)) => held,
                    None => {
                        let mut empty = rows.clone();
                        empty.set("rows", Value::Array(vec![]));
                        related.push((name.clone(), empty));
                        &mut related.last_mut().expect("just pushed").1
                    }
                };
                // A call may send a column another did not, such as a key
                // to a table only it reached: rows take every column, a
                // cell their call did not send left null.
                let mut names = array(field(held, "columns")?)?.to_vec();
                for column in array(field(rows, "columns")?)? {
                    if !names.contains(column) {
                        names.push(column.clone());
                    }
                }
                let widen = |row: &[Value], from: &[Value]| {
                    names
                        .iter()
                        .map(|name| {
                            from.iter()
                                .position(|f| f == name)
                                .map_or(Value::Null, |i| row[i].clone())
                        })
                        .collect::<Vec<_>>()
                };
                let before = array(field(held, "columns")?)?.to_vec();
                let mut all = array(field(held, "rows")?)?
                    .iter()
                    .map(|row| Ok(Value::Array(widen(array(row)?, &before))))
                    .collect::<Result<Vec<_>, CoreError>>()?;
                let from = array(field(rows, "columns")?)?;
                for row in &read.rows {
                    if let Some(&new) = kept.get(&key(&row[at])?) {
                        let mut row = row.to_vec();
                        row[at] = Value::int(new);
                        all.push(Value::Array(widen(&row, from)));
                    }
                }
                held.set("columns", Value::Array(names.clone()));
                held.set("rows", Value::Array(all));
            }
        }
        let step = field(payload, "schema")?;
        if let Some(value) = step.get("related_tables") {
            let mut combined = schema
                .get("related_tables")
                .cloned()
                .unwrap_or(Value::Object(vec![]));
            for (name, described) in members(value)? {
                match combined.get(name).cloned() {
                    None => combined.set(name, described.clone()),
                    Some(mut held) => {
                        let mut columns = field(&held, "columns")?.clone();
                        for (column, spec) in members(field(described, "columns")?)? {
                            if columns.get(column).is_none() {
                                columns.set(column, spec.clone());
                            }
                        }
                        held.set("columns", columns);
                        combined.set(name, held);
                    }
                }
            }
            schema.set("related_tables", combined);
        }
        if let Some(value) = step.get("relationships") {
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
    }
    let mut instance_table = field(first_context, "instance_table")?.clone();
    instance_table.set("rows", Value::Array(instances));
    let mut context = Value::Object(vec![]);
    context.set("instance_table", instance_table);
    if payloads.iter().any(|p| {
        p.get("context")
            .is_some_and(|c| c.get("related_tables").is_some())
    }) {
        context.set("related_tables", Value::Object(related));
    }
    let mut out = first.clone();
    out.set("schema", schema);
    out.set("context", context);
    Ok(out)
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

    fn step(labels: [f64; 3], table: &str, column: &str, values: [f64; 3]) -> Value {
        let instances = (0..3)
            .map(|i| format!("[{i},{},{}]", i + 1, labels[i]))
            .collect::<Vec<_>>()
            .join(",");
        let related = (0..3)
            .map(|i| format!("[{i},{},{}]", i + 1, values[i]))
            .collect::<Vec<_>>()
            .join(",");
        parse(&format!(
            r#"{{"task":{{"kind":"regression","target":{{"column_name":"y"}}}},
            "schema":{{"instance_table":{{"columns":{{"instance_id":{{"stype":"ID"}},"ref":{{"stype":"ID"}},"y":{{"stype":"numerical"}}}}}},
            "related_tables":{{"{table}":{{"primary_key":["instance_id","id"],"columns":{{"instance_id":{{"stype":"ID"}},"id":{{"stype":"ID"}},"{column}":{{"stype":"numerical"}}}}}}}},
            "relationships":[{{"source_columns":["ref"],"target_table":"{table}","target_columns":["id"]}}]}},
            "context":{{"instance_table":{{"columns":["instance_id","ref","y"],"rows":[{instances}]}},
            "related_tables":{{"{table}":{{"columns":["instance_id","id","{column}"],"rows":[{related}]}}}}}}}}"#
        ))
    }

    #[test]
    fn forecast_analysis_covers_every_step_and_tables_only_a_later_step_reached() {
        let combined = forecast(&[
            step([0.0, 10.0, 0.0], "orders", "amount", [1.0, 2.0, 3.0]),
            step(
                [100.0, 200.0, 300.0],
                "events",
                "seconds",
                [10.0, 20.0, 30.0],
            ),
        ])
        .unwrap();
        let ColumnAnalysis::Ready { rows } = analyze(&combined).unwrap() else {
            panic!()
        };
        let cohorts = |column: &str| {
            rows.iter()
                .filter(|r| r.column == column)
                .map(|r| (r.cohort.as_str(), r.mean))
                .collect::<Vec<_>>()
        };
        let counted = rows
            .iter()
            .filter(|r| r.table == "orders" && r.column == "COUNT(*)")
            .map(|r| (r.cohort.as_str(), r.mean))
            .collect::<Vec<_>>();
        assert_eq!(counted, [("[0 - 0]", 200.0), ("(0 - 1+]", 10.0 / 3.0)]);
        assert_eq!(
            cohorts("amount"),
            [("[1 - 1.20]", 0.0), ("(1.20 - 2]", 10.0), ("(2 - 3]", 0.0)]
        );
        assert_eq!(
            cohorts("seconds"),
            [
                ("[10 - 12]", 100.0),
                ("(12 - 20]", 200.0),
                ("(20 - 30]", 300.0)
            ]
        );
    }

    #[test]
    fn calls_of_one_request_are_analysed_over_each_sent_instance_once() {
        let call = |rows: &str| {
            parse(&format!(
                r#"{{"task":{{"kind":"regression","target":{{"column_name":"y"}}}},
                "schema":{{"instance_table":{{"columns":{{"instance_id":{{"stype":"ID"}},"__kumo_entity_ref_0":{{"stype":"ID"}},"x":{{"stype":"categorical"}},"y":{{"stype":"numerical"}}}}}}}},
                "context":{{"instance_table":{{"columns":["instance_id","__kumo_entity_ref_0","x","y"],"rows":[{rows}]}}}}}}"#
            ))
        };
        let first = call(r#"[0,1,"p",0],[2,7,"p",0],[3,2,"q",1],[4,3,"q",0]"#);
        let second = call(r#"[0,1,"p",0],[1,5,"p",1],[3,2,"q",1],[4,3,"q",0]"#);
        let ColumnAnalysis::Ready { rows } = analyze(&sent(&[first, second]).unwrap()).unwrap()
        else {
            panic!()
        };
        let cohorts = rows
            .iter()
            .filter(|r| r.column == "x")
            .map(|r| (r.cohort.as_str(), r.share, r.mean))
            .collect::<Vec<_>>();
        assert_eq!(cohorts, [("p", 0.6, 1.0 / 3.0), ("q", 0.4, 0.5)]);
    }

    #[test]
    fn calls_gathered_apart_keep_each_instance_and_its_rows() {
        let call = |instances: &str, users: &str| {
            parse(&format!(
                r#"{{"task":{{"kind":"regression","target":{{"column_name":"y"}}}},
                "schema":{{"instance_table":{{"columns":{{"instance_id":{{"stype":"ID"}},"__kumo_entity_ref_0":{{"stype":"ID"}},"y":{{"stype":"numerical"}}}}}},
                "related_tables":{{"users":{{"primary_key":["instance_id","uid"],"columns":{{"instance_id":{{"stype":"ID"}},"uid":{{"stype":"ID"}},"age":{{"stype":"numerical"}}}}}}}},
                "relationships":[{{"source_columns":["__kumo_entity_ref_0"],"target_table":"users","target_columns":["uid"]}}]}},
                "context":{{"instance_table":{{"columns":["instance_id","__kumo_entity_ref_0","y"],"rows":[{instances}]}},
                "related_tables":{{"users":{{"columns":["instance_id","uid","age"],"rows":[{users}]}}}}}}}}"#
            ))
        };
        let first = call("[0,1,0],[1,2,1]", "[0,1,30],[1,2,40]");
        let second = call("[0,3,1],[1,4,1]", "[0,3,50],[1,4,60]");
        let combined = sent(&[first, second]).unwrap();
        let context = field(&combined, "context").unwrap();
        let instances = Table::read(field(context, "instance_table").unwrap()).unwrap();
        let entity = instances
            .rows
            .iter()
            .map(|r| (key(&r[0]).unwrap(), key(&r[1]).unwrap()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(entity.len(), 4);
        let users = field(field(context, "related_tables").unwrap(), "users").unwrap();
        for row in &Table::read(users).unwrap().rows {
            assert_eq!(entity[&key(&row[0]).unwrap()], key(&row[1]).unwrap());
        }
        assert!(matches!(
            analyze(&combined).unwrap(),
            ColumnAnalysis::Ready { .. }
        ));
    }

    #[test]
    fn calls_that_reached_different_tables_are_analysed_together() {
        let users = r#""users":{"primary_key":["instance_id","uid"],"columns":{"instance_id":{"stype":"ID"},"uid":{"stype":"ID"},"age":{"stype":"numerical"}}}"#;
        let orders = |key: &str| {
            format!(
                r#""orders":{{"primary_key":["instance_id","oid"],"columns":{{"instance_id":{{"stype":"ID"}},"oid":{{"stype":"ID"}},"uid":{{"stype":"ID"}}{key}}}}}"#
            )
        };
        let link = |table: &str, column: &str, target: &str| {
            format!(
                r#"{{"source_table":"{table}","source_columns":["{column}"],"target_table":"{target}","target_columns":["{column}"]}}"#
            )
        };
        let call = |instances: &str, extra: &str, product: bool| {
            let entity = r#"{"source_columns":["__kumo_entity_ref_0"],"target_table":"users","target_columns":["uid"]}"#;
            let (key, products, product_link, columns, rows) = if product {
                (
                    r#","pid":{"stype":"ID"}"#,
                    r#","products":{"primary_key":["instance_id","pid"],"columns":{"instance_id":{"stype":"ID"},"pid":{"stype":"ID"},"price":{"stype":"numerical"}}}"#,
                    format!(",{}", link("orders", "pid", "products")),
                    r#"["instance_id","oid","uid","pid"]"#,
                    r#","products":{"columns":["instance_id","pid","price"],"rows":[[1,9,5.0]]}"#,
                )
            } else {
                ("", "", String::new(), r#"["instance_id","oid","uid"]"#, "")
            };
            let order_rows = if product { "[[1,8,2,9]]" } else { "[[0,7,1]]" };
            parse(&format!(
                r#"{{"task":{{"kind":"regression","target":{{"column_name":"y"}}}},
                "schema":{{"instance_table":{{"columns":{{"instance_id":{{"stype":"ID"}},"__kumo_entity_ref_0":{{"stype":"ID"}},"y":{{"stype":"numerical"}}}}}},
                "related_tables":{{{users},{}{products}}},
                "relationships":[{entity},{}{product_link}]}},
                "context":{{"instance_table":{{"columns":["instance_id","__kumo_entity_ref_0","y"],"rows":[{instances}]}},
                "related_tables":{{"users":{{"columns":["instance_id","uid","age"],"rows":[{extra}]}},
                "orders":{{"columns":{columns},"rows":{order_rows}}}{rows}}}}}}}"#,
                orders(key),
                link("orders", "uid", "users"),
            ))
        };
        let first = call("[0,1,0],[1,2,1]", "[0,1,30],[1,2,40]", false);
        let second = call("[0,1,0],[1,2,1]", "[0,1,30],[1,2,40]", true);
        assert!(matches!(
            analyze(&sent(&[first, second]).unwrap()).unwrap(),
            ColumnAnalysis::Ready { .. }
        ));
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
