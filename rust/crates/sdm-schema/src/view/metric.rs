// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A metric view's definition, read into its structure (design S5, T3.5's
//! M2): its source, parameters, joins with their nested joins, filter,
//! dimensions and measures, every expression and name kept as written.
//!
//! This is the part of the old client's `parse_metric_view`
//! (`DB/metric_view.py:727-1082`) that reads the YAML's structure; what it
//! decided by reading an expression or a name (a join's source and
//! condition, a dimension's tables and wildcards, the filter, a
//! parameter's default against its type) the data layer decides, and the
//! core asks in the old client's order. So a section is read here as the
//! old client iterates it (`for x in section or []`, a list's items, a
//! mapping's keys, a text's characters), an entry that is no mapping or
//! misses a name is kept in its place with its value, which the core's
//! message quotes, and nothing is dropped.

use std::fmt;

use super::yaml::{self, Value, YamlError};

/// A metric view's definition, its expressions as written.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricView {
    /// The `source`, never null.
    pub source: Value,
    /// Whether the definition has a `materialization` section, which the
    /// old client ignores with a message (`:794-798`).
    pub materialization: bool,
    /// Each entry of `parameters` that is a mapping with a text `name`, in
    /// order, a name given twice included (`:569-605`).
    pub parameters: Vec<Parameter>,
    /// The `joins`, in order, each with its nested joins (`:825-909`).
    pub joins: Vec<JoinItem>,
    /// The `filter`, unless absent or null (`:924`).
    pub filter: Option<Value>,
    /// The `dimensions`, or, without any, the `fields` (`:946-957`).
    pub dimensions: Vec<DimensionItem>,
    /// Whether `fields` was passed over for `dimensions`, which the old
    /// client says (`:953-957`).
    pub fields_ignored: bool,
    /// Each entry of `measures` that is a mapping with a text `name`, in
    /// order (`:1111-1126`).
    pub measures: Vec<Measure>,
    /// The `order` of each window of each measure, as written and in order
    /// (`:1222-1235`); the data layer unquotes them and keeps each once.
    pub window_orders: Vec<String>,
}

/// A parameter of the view: its name, and its `data_type` and `default`
/// as YAML gave them, null when absent. The type Databricks describes wins
/// over this one (`:589`).
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub data_type: Value,
    pub default: Value,
}

/// An entry of a `joins` section.
#[derive(Clone, Debug, PartialEq)]
pub enum JoinItem {
    /// No mapping, with its value (`:828-832`).
    NotAMapping {
        cfg: Value,
    },
    /// A mapping with no text `name` (`:834-839`), and its nested joins,
    /// which a skipped parent still names (`:817-823`).
    Nameless {
        cfg: Value,
        joins: Vec<JoinItem>,
    },
    Join(Join),
}

/// A join with a name.
#[derive(Clone, Debug, PartialEq)]
pub struct Join {
    pub name: String,
    /// Its `source`, null when absent.
    pub source: Value,
    pub cardinality: Cardinality,
    pub keys: JoinKeys,
    /// Its nested joins.
    pub joins: Vec<JoinItem>,
}

/// A join's `cardinality` (`:862-865`), `many_to_one` when absent.
#[derive(Clone, Debug, PartialEq)]
pub enum Cardinality {
    ManyToOne,
    OneToMany,
    /// Any other value.
    Unsupported(Value),
}

/// What a join is joined on (`:867-891`).
#[derive(Clone, Debug, PartialEq)]
pub enum JoinKeys {
    /// `using` with one column, as given (Python's `str` of it names the
    /// column).
    Using(Value),
    /// `using` with any other number of columns, or no list.
    UsingSeveral,
    /// No `using`: the `on`, or the value of the key `true` a bare `on:`
    /// is in YAML 1.1 (`:883`), null when neither is there.
    On(Value),
}

/// An entry of the dimensions.
#[derive(Clone, Debug, PartialEq)]
pub enum DimensionItem {
    /// No mapping, with its value (`:959-963`).
    NotAMapping { cfg: Value },
    /// A mapping: its `name` and `expr`, null when absent, and the mapping,
    /// which the message of one that misses either quotes (`:991-996`).
    Dimension {
        name: Value,
        expr: Value,
        cfg: Value,
    },
}

/// A measure: its name, its `expr` (null when absent), and whether it has
/// a `window` (`:1200`).
#[derive(Clone, Debug, PartialEq)]
pub struct Measure {
    pub name: String,
    pub expr: Value,
    pub window: bool,
}

/// Why a metric view's definition is refused before anything of it is
/// read further.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetricRefusal {
    /// The text is no YAML (the old client let PyYAML's error through).
    Yaml(YamlError),
    /// `:774-777`.
    NotAMapping,
    /// `:779-783`.
    MissesSource,
    /// A section the old client iterates is a value that cannot be
    /// iterated, a number say, where it failed with a `TypeError`.
    NotAList { section: &'static str },
}

impl fmt::Display for MetricRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricRefusal::Yaml(error) => {
                write!(
                    f,
                    "The metric view definition is not readable YAML: {error}"
                )
            }
            MetricRefusal::NotAMapping => {
                f.write_str("The metric view definition is not a YAML mapping")
            }
            MetricRefusal::MissesSource => {
                f.write_str("The metric view definition misses a 'source'")
            }
            MetricRefusal::NotAList { section } => {
                write!(f, "The metric view definition's '{section}' is not a list")
            }
        }
    }
}

impl std::error::Error for MetricRefusal {}

/// Reads a metric view's definition (`DB/metric_view.py:773-783`).
pub fn parse(definition: &str) -> Result<MetricView, MetricRefusal> {
    let cfg = yaml::load(definition).map_err(MetricRefusal::Yaml)?;
    if !matches!(cfg, Value::Map(_)) {
        return Err(MetricRefusal::NotAMapping);
    }
    let source = match cfg.get("source") {
        None | Some(Value::Null) => return Err(MetricRefusal::MissesSource),
        Some(source) => source.clone(),
    };
    let dimensions_cfg = cfg.get("dimensions").filter(|v| v.is_set());
    let fields_cfg = cfg.get("fields").filter(|v| v.is_set());
    let (section, dimension_cfgs) = match (dimensions_cfg, fields_cfg) {
        (Some(dimensions), _) => ("dimensions", Some(dimensions)),
        (None, fields) => ("fields", fields),
    };
    Ok(MetricView {
        source,
        materialization: cfg.has("materialization"),
        parameters: parameters(cfg.get("parameters")),
        joins: joins(cfg.get("joins"))?,
        filter: cfg.get("filter").filter(|v| **v != Value::Null).cloned(),
        dimensions: entries(dimension_cfgs, section)?
            .into_iter()
            .map(|entry| match &entry {
                Value::Map(_) => DimensionItem::Dimension {
                    name: entry.get("name").cloned().unwrap_or(Value::Null),
                    expr: entry.get("expr").cloned().unwrap_or(Value::Null),
                    cfg: entry.clone(),
                },
                _ => DimensionItem::NotAMapping { cfg: entry.clone() },
            })
            .collect(),
        fields_ignored: dimensions_cfg.is_some() && fields_cfg.is_some(),
        measures: measures(cfg.get("measures")),
        window_orders: window_orders(cfg.get("measures")),
    })
}

/// The entries `for entry in section or []` visits: none for a null or an
/// empty value, a list's items, a mapping's keys, a text's characters, a
/// byte string's numbers; any other value cannot be iterated.
fn entries(section: Option<&Value>, name: &'static str) -> Result<Vec<Value>, MetricRefusal> {
    let Some(value) = section.filter(|value| value.is_set()) else {
        return Ok(Vec::new());
    };
    match value {
        Value::Seq(items) => Ok(items.clone()),
        Value::Map(pairs) => Ok(pairs.iter().map(|(key, _)| key.clone()).collect()),
        Value::Str(text) => Ok(text.chars().map(|c| Value::Str(c.to_string())).collect()),
        Value::Bytes(bytes) => Ok(bytes.iter().map(|&b| Value::Int(i128::from(b))).collect()),
        _ => Err(MetricRefusal::NotAList { section: name }),
    }
}

fn parameters(section: Option<&Value>) -> Vec<Parameter> {
    let Some(Value::Seq(items)) = section else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|cfg| {
            let name = cfg.get("name")?.as_str()?;
            Some(Parameter {
                name: name.to_owned(),
                data_type: cfg.get("data_type").cloned().unwrap_or(Value::Null),
                default: cfg.get("default").cloned().unwrap_or(Value::Null),
            })
        })
        .collect()
}

fn joins(section: Option<&Value>) -> Result<Vec<JoinItem>, MetricRefusal> {
    entries(section, "joins")?
        .into_iter()
        .map(|cfg| {
            if !matches!(cfg, Value::Map(_)) {
                return Ok(JoinItem::NotAMapping { cfg });
            }
            let nested = joins(cfg.get("joins"))?;
            let Some(name) = cfg.get("name").and_then(Value::as_str) else {
                return Ok(JoinItem::Nameless {
                    cfg: cfg.clone(),
                    joins: nested,
                });
            };
            let cardinality = match cfg.get("cardinality") {
                None => Cardinality::ManyToOne,
                Some(Value::Str(text)) if text == "many_to_one" => Cardinality::ManyToOne,
                Some(Value::Str(text)) if text == "one_to_many" => Cardinality::OneToMany,
                Some(other) => Cardinality::Unsupported(other.clone()),
            };
            let keys = match cfg.get("using") {
                None | Some(Value::Null) => JoinKeys::On(on(&cfg)),
                Some(column @ Value::Str(_)) => JoinKeys::Using(column.clone()),
                Some(Value::Seq(columns)) if columns.len() == 1 => {
                    JoinKeys::Using(columns[0].clone())
                }
                Some(_) => JoinKeys::UsingSeveral,
            };
            Ok(JoinItem::Join(Join {
                name: name.to_owned(),
                source: cfg.get("source").cloned().unwrap_or(Value::Null),
                cardinality,
                keys,
                joins: nested,
            }))
        })
        .collect()
}

/// `join_cfg.get('on', join_cfg.get(True))`: the `on` entry if there is
/// one, else the entry of a key Python finds equal to `True` (the key
/// YAML 1.1 reads a bare `on:` as, or `1`).
fn on(cfg: &Value) -> Value {
    if let Some(value) = cfg.get("on") {
        return value.clone();
    }
    let Value::Map(pairs) = cfg else {
        return Value::Null;
    };
    pairs
        .iter()
        .find(|(key, _)| match key {
            Value::Bool(flag) => *flag,
            Value::Int(n) => *n == 1,
            Value::Float(x) => *x == 1.0,
            _ => false,
        })
        .map_or(Value::Null, |(_, value)| value.clone())
}

impl Join {
    /// The names of every join nested in this one, at any depth, in order,
    /// as the old client lists those a skipped join takes with it
    /// (`_collect_aliases`, `:817-823`), through nested joins with no name.
    pub fn nested_names(&self) -> Vec<String> {
        fn collect(items: &[JoinItem], names: &mut Vec<String>) {
            for item in items {
                match item {
                    JoinItem::NotAMapping { .. } => {}
                    JoinItem::Nameless { joins, .. } => collect(joins, names),
                    JoinItem::Join(join) => {
                        names.push(join.name.clone());
                        collect(&join.joins, names);
                    }
                }
            }
        }
        let mut names = Vec::new();
        collect(&self.joins, &mut names);
        names
    }
}

fn measures(section: Option<&Value>) -> Vec<Measure> {
    let Some(Value::Seq(items)) = section else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|cfg| {
            let name = cfg.get("name")?.as_str()?;
            Some(Measure {
                name: name.to_owned(),
                expr: cfg.get("expr").cloned().unwrap_or(Value::Null),
                window: cfg.has("window"),
            })
        })
        .collect()
}

/// Each window's `order` text, a measure's `window` being one mapping or a
/// list of them (`_window_orders`, `:1222-1235`).
fn window_orders(section: Option<&Value>) -> Vec<String> {
    let Some(Value::Seq(items)) = section else {
        return Vec::new();
    };
    let mut orders = Vec::new();
    for cfg in items {
        let windows = match cfg.get("window") {
            Some(window @ Value::Map(_)) => vec![window.clone()],
            Some(Value::Seq(windows)) => windows.clone(),
            _ => Vec::new(),
        };
        for window in &windows {
            if let Some(order) = window.get("order").and_then(Value::as_str) {
                orders.push(order.to_owned());
            }
        }
    }
    orders
}
