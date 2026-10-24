// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A metric view's definition read as the old client's `parse_metric_view`
//! reads it (`DB/metric_view.py:727-1081`, T3.5's M2), once its parameters
//! are read: the joins kept and skipped, the filter, the dimensions and the
//! measures, in that order, each question of text asked of the data layer
//! in the run provider's dialect ([`Ask`]). Nothing here reads a warehouse;
//! the routine ([`super::metric`]) reads the source table's columns first
//! where the view has parameters, as the old client does, and hands the
//! parameters in.

use sdm_contract::view::{
    DimensionRef, FilterRefusal, JoinRef, KeyRef, MeasureInput, MeasureRead, Parameter, Unreadable,
    ViewTextAnswer, ViewTextAsk,
};
use sdm_contract::{CoreError, CoreWarning, Detail, Sentence};
use sdm_schema::view::joins::{JoinMessage, SkipReason};
use sdm_schema::view::metric::SOURCE_ALIAS;
use sdm_schema::view::spec::{
    Cardinality, ColumnItem, Ignored, JoinKeys, Source, TableItem, ViewSpec,
};
use sdm_schema::view::yaml::NodeId;

use crate::view_steps::{self, Ask, JoinJudgement};
use crate::view_words::{self, FilterHatch, TooLarge};

/// A join the conversion keeps: its alias, the alias it is declared on,
/// its path from the source table, its table's name parts as written, its
/// cardinality and its condition's two sides.
#[derive(Clone, Debug, PartialEq)]
pub struct Join {
    pub alias: String,
    pub parent_alias: String,
    pub path: Vec<String>,
    pub table: Vec<String>,
    pub many_to_one: bool,
    pub condition: [KeyRef; 2],
}

impl Join {
    /// The join as the data layer resolves its keys.
    pub fn as_ref(&self) -> JoinRef {
        JoinRef {
            alias: self.alias.clone(),
            parent_alias: self.parent_alias.clone(),
            path: self.path.clone(),
            condition: self.condition.clone(),
        }
    }
}

/// A join left out, with the joins nested in it and the dimensions that go
/// with it.
#[derive(Clone, Debug, PartialEq)]
pub struct Skipped {
    pub alias: String,
    pub reason: String,
    pub nested: Vec<String>,
    pub dimensions: Vec<String>,
}

/// A dimension the conversion reads.
#[derive(Clone, Debug, PartialEq)]
pub enum Dimension {
    /// `alias.*`, every column of that table but `excluded`.
    Wildcard {
        alias: String,
        excluded: Vec<String>,
    },
    /// A column on the table `alias`, its expression with the table's
    /// qualifier taken out.
    Column {
        name: String,
        alias: String,
        expr: String,
    },
}

/// A metric view's definition, read.
#[derive(Clone, Debug, PartialEq)]
pub struct Definition {
    pub joins: Vec<Join>,
    pub skipped: Vec<Skipped>,
    /// Every join path or alias a column can be qualified by, with the alias
    /// it resolves to, longest first.
    pub candidates: Vec<(Vec<String>, String)>,
    /// The filter, its parameters read as their defaults, checked against
    /// joined tables and unsafe text; not yet against the source table's
    /// columns.
    pub filter: Option<String>,
    pub dimensions: Vec<Dimension>,
    pub measures: Vec<MeasureRead>,
    /// What the definition loses, in the old client's order, before the
    /// skipped joins ([`Definition::warnings`]).
    pub messages: Vec<CoreWarning>,
    /// Each parameter read as its default, by its name, with its text, in
    /// the order first read.
    pub notes: Vec<(String, String)>,
    /// The dimensions window measures order by, unquoted, each once.
    pub window_orders: Vec<String>,
}

impl Definition {
    /// What the definition loses, as the old client says it: its messages,
    /// then each join skipped (`[*spec.messages, *map(str, skipped)]`).
    pub fn warnings(&self) -> Vec<CoreWarning> {
        let mut warnings = self.messages.clone();
        warnings.extend(self.skipped.iter().map(skipped_warning));
        warnings
    }
}

/// The warning of a join left out (`SkippedJoin.__str__`).
pub fn skipped_warning(skipped: &Skipped) -> CoreWarning {
    let text = view_words::skipped_join_text(
        &skipped.alias,
        &skipped.reason,
        &skipped.nested,
        &skipped.dimensions,
    );
    match view_words::lost_text(&skipped.nested, &skipped.dimensions) {
        Some(lost) => CoreWarning::new(Sentence::ViewJoinSkippedWithIt, text)
            .with_field("join", Detail::text(&skipped.alias))
            .with_field("lost", Detail::text(lost))
            .with_field("reason", Detail::text(&skipped.reason)),
        None => CoreWarning::new(Sentence::ViewJoinSkipped, text)
            .with_field("join", Detail::text(&skipped.alias))
            .with_field("reason", Detail::text(&skipped.reason)),
    }
}

/// Why the view is refused while its definition is read: its filter.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterRefused {
    /// The filter as the old client quoted it, Python's `repr`.
    pub filter: String,
    pub reason: String,
    pub hatch: FilterHatch,
}

impl FilterRefused {
    /// The refusal, as the old client words it (`_refuse_filter`).
    pub fn error(&self) -> CoreError {
        let sentence = match self.hatch {
            FilterHatch::None => Sentence::ViewFilterRefused,
            FilterHatch::Default => Sentence::ViewFilterRefusedNoDefault,
            FilterHatch::Trust => Sentence::ViewFilterRefusedUntrusted,
        };
        CoreError::new(
            sentence,
            view_words::filter_refused_text(&self.filter, &self.reason, self.hatch),
        )
        .with_field("filter", Detail::text(&self.filter))
        .with_field("reason", Detail::text(&self.reason))
    }
}

/// The reason a dimension reading every column through `*` is refused
/// (`_STAR_REASON`).
pub const STAR_REASON: &str =
    "it reads every column of a table through `*`, which the graph cannot reproduce";

fn mismatched(asked: &str) -> CoreError {
    CoreError::new(
        Sentence::Internal,
        "the data layer answered a question of a view's text with another kind of answer",
    )
    .with_field("asked", Detail::text(asked))
}

/// A value too large to quote, refused by name.
pub fn too_large(_: TooLarge) -> CoreError {
    CoreError::new(
        Sentence::ViewValueTooLarge,
        "The metric view definition holds a value too large to write in a message or a statement",
    )
    .with_field("view_kind", Detail::text("metric view"))
}

fn text(
    ask: Ask<'_>,
    question: ViewTextAsk,
    asked: &str,
) -> Result<Result<String, Unreadable>, CoreError> {
    match ask(question)? {
        ViewTextAnswer::Text(text) => Ok(text),
        _ => Err(mismatched(asked)),
    }
}

fn flag(
    ask: Ask<'_>,
    question: ViewTextAsk,
    asked: &str,
) -> Result<Result<bool, Unreadable>, CoreError> {
    match ask(question)? {
        ViewTextAnswer::Flag(flag) => Ok(flag),
        _ => Err(mismatched(asked)),
    }
}

/// `expr` with its comments dropped, as Databricks drops them (`_sql`).
fn stripped(ask: Ask<'_>, expr: &str) -> Result<Result<String, Unreadable>, CoreError> {
    text(
        ask,
        ViewTextAsk::StripComments {
            expr: expr.to_owned(),
        },
        "strip_comments",
    )
}

/// Whether `expr` holds text no trust accepts (`_unsafe_expr_reason`).
fn unsafe_text(ask: Ask<'_>, expr: &str) -> Result<bool, CoreError> {
    match ask(ViewTextAsk::Expression {
        expr: expr.to_owned(),
        trust_expressions: true,
        columns: Vec::new(),
    })? {
        ViewTextAnswer::Expression(verdict) => Ok(verdict.is_some()),
        _ => Err(mismatched("expression")),
    }
}

/// Why an expression naming `referenced` parameters cannot be read, the
/// first with no literal (`_unusable_parameter`).
fn unusable_parameter(
    spec: &ViewSpec,
    parameters: &[(String, Parameter)],
    referenced: &[String],
) -> Result<Option<String>, CoreError> {
    for name in referenced {
        let Some((_, parameter)) = parameters.iter().find(|(folded, _)| folded == name) else {
            continue;
        };
        if parameter.read.is_err() {
            let default = spec
                .parameters
                .iter()
                .find(|p| p.name == parameter.name)
                .and_then(|p| p.default);
            let said = view_words::parameter_text(parameter, &spec.document, default)
                .map_err(too_large)?;
            return Ok(Some(format!(
                "it references the parameter '{}', which {said}",
                parameter.name
            )));
        }
    }
    Ok(None)
}

/// `expr` with each parameter that has a literal replaced by it, and the
/// parameters it names (`_substitute_parameters`).
fn substituted(
    ask: Ask<'_>,
    expr: &str,
    parameters: &[(String, Parameter)],
) -> Result<Result<(String, Vec<String>), Unreadable>, CoreError> {
    match ask(ViewTextAsk::SubstituteParameters {
        expr: expr.to_owned(),
        parameters: parameters.to_vec(),
    })? {
        ViewTextAnswer::Substituted(read) => Ok(read),
        _ => Err(mismatched("substitute_parameters")),
    }
}

/// Records each parameter of `referenced` as read, once, in order.
fn used(
    notes: &mut Vec<(String, String)>,
    parameters: &[(String, Parameter)],
    referenced: &[String],
) {
    for name in referenced {
        let Some((_, parameter)) = parameters.iter().find(|(folded, _)| folded == name) else {
            continue;
        };
        let Ok(read) = &parameter.read else {
            continue;
        };
        if !notes.iter().any(|(n, _)| *n == parameter.name) {
            notes.push((parameter.name.clone(), read.text.clone()));
        }
    }
}

fn warning(sentence: Sentence, text: String, fields: &[(&str, &str)]) -> CoreWarning {
    fields.iter().fold(
        CoreWarning::new(sentence, text),
        |warning, (name, value)| warning.with_field(name, Detail::text(*value)),
    )
}

/// The joins kept, each read for what the conversion needs of it.
fn kept_joins(spec: &ViewSpec, kept: &[usize], ask: Ask<'_>) -> Result<Vec<Join>, CoreError> {
    let mut joins = Vec::with_capacity(kept.len());
    for &at in kept {
        let TableItem::Table(table) = &spec.tables[at] else {
            continue;
        };
        let mut path = vec![table.alias.clone()];
        let mut parent = table.parent;
        while let Some(index) = parent.filter(|&p| p > 0) {
            let TableItem::Table(up) = &spec.tables[index] else {
                break;
            };
            path.insert(0, up.alias.clone());
            parent = up.parent;
        }
        let parent_alias = if path.len() > 1 {
            path[path.len() - 2].clone()
        } else {
            SOURCE_ALIAS.to_owned()
        };
        let (Some(join), Source::Name(Some(source))) = (&table.join, &table.source) else {
            continue;
        };
        let Some(written) = spec.text(*source) else {
            continue;
        };
        let Some(parts) = view_steps::names(
            ask,
            ViewTextAsk::TableReference {
                name: written.to_owned(),
            },
        )?
        else {
            continue;
        };
        let condition = match join.keys {
            JoinKeys::Using(column) => {
                let written = view_words::value_text(&spec.document, column).map_err(too_large)?;
                // `unquote_ident`: one name, in backticks or bare.
                let column = match view_steps::names(
                    ask,
                    ViewTextAsk::TableReference {
                        name: written.clone(),
                    },
                )? {
                    Some(parts) if parts.len() == 1 => parts[0].clone(),
                    _ => written.trim().to_owned(),
                };
                [
                    KeyRef {
                        qualifier: Some(parent_alias.clone()),
                        column: column.clone(),
                    },
                    KeyRef {
                        qualifier: Some(table.alias.clone()),
                        column,
                    },
                ]
            }
            JoinKeys::On(Some(on)) => {
                let Some(on) = spec.text(on) else { continue };
                let Ok(on) = stripped(ask, on)? else { continue };
                match ask(ViewTextAsk::JoinCondition { condition: on })? {
                    ViewTextAnswer::Condition(Some(condition)) => condition,
                    ViewTextAnswer::Condition(None) => continue,
                    _ => return Err(mismatched("join_condition")),
                }
            }
            _ => continue,
        };
        joins.push(Join {
            alias: table.alias.clone(),
            parent_alias,
            path,
            table: parts,
            many_to_one: !matches!(join.cardinality, Cardinality::OneToMany),
            condition,
        });
    }
    Ok(joins)
}

/// Reads `spec`, a metric view's definition, given its `parameters` as the
/// routine read them, a source column of a parameter's name already taken
/// out (`DB/metric_view.py:799-802`); the source table's name is read
/// before ([`view_steps::fact_source`]); `database` names the warehouse a
/// message names (`{database:backend}`). A filter the view cannot apply
/// refuses it ([`FilterRefused`]); a definition the old client failed on
/// is refused by name.
pub fn read(
    spec: &mut ViewSpec,
    parameters: &[(String, Parameter)],
    ask: Ask<'_>,
    database: &str,
) -> Result<Result<Definition, FilterRefused>, CoreError> {
    let mut messages: Vec<CoreWarning> = Vec::new();
    if spec.ignored.contains(&Ignored::Materialization) {
        messages.push(warning(
            Sentence::ViewMaterializationIgnored,
            view_words::ignored_text(Ignored::Materialization),
            &[],
        ));
    }
    let walk = match view_steps::walk_joins(spec, ask)? {
        Ok(walk) => walk,
        Err(refusal) => {
            return Err(CoreError::new(Sentence::ViewInvalid, refusal.to_string())
                .with_field("view_kind", Detail::text("metric view")));
        }
    };
    for message in &walk.messages {
        let text = view_words::join_message_text(spec, message).map_err(too_large)?;
        let (sentence, cfg) = match message {
            JoinMessage::NotAMapping(cfg) => (Sentence::ViewJoinNotAMapping, *cfg),
            JoinMessage::Nameless(cfg) => (Sentence::ViewJoinNameless, *cfg),
        };
        let join = view_words::value_repr(&spec.document, cfg).map_err(too_large)?;
        messages.push(warning(sentence, text, &[("join", &join)]));
    }
    let joins = kept_joins(spec, &walk.kept, ask)?;
    let kept: Vec<String> = joins.iter().map(|j| j.alias.clone()).collect();
    let mut skipped = Vec::with_capacity(walk.skipped.len());
    for skip in &walk.skipped {
        let reason: &SkipReason<JoinJudgement> = &skip.reason;
        skipped.push(Skipped {
            alias: skip.alias.clone(),
            reason: view_words::skip_reason_text(spec, &skip.alias, reason).map_err(too_large)?,
            nested: skip.nested.clone(),
            dimensions: Vec::new(),
        });
    }

    // Every join path or alias a column can be qualified by, longest first.
    let mut candidates: Vec<(Vec<String>, String)> = Vec::new();
    for join in &joins {
        candidates.push((join.path.clone(), join.alias.clone()));
        if join.path.len() > 1 {
            candidates.push((vec![join.alias.clone()], join.alias.clone()));
        }
    }
    // `aliases - kept`, a set in the old client, so its order is Python's
    // and no reading depends on it: each alias met and not kept.
    for alias in &walk.aliases {
        if !kept.contains(alias) {
            candidates.push((vec![alias.clone()], alias.clone()));
        }
    }
    candidates.push((vec![SOURCE_ALIAS.to_owned()], SOURCE_ALIAS.to_owned()));
    // A stable sort, longest first, as Python's `sort(reverse=True)` keeps
    // equal lengths in order.
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.0.len()));
    let mut qualifiers: Vec<Vec<String>> = Vec::new();
    for (path, _) in &candidates {
        let folded: Vec<String> = path.iter().map(|p| p.to_lowercase()).collect();
        if !qualifiers.contains(&folded) {
            qualifiers.push(folded);
        }
    }

    let mut notes: Vec<(String, String)> = Vec::new();
    let filter = match spec.filter {
        None => None,
        Some(node) => match read_filter(spec, node, parameters, &candidates, &mut notes, ask)? {
            Ok(filter) => Some(filter),
            Err(refused) => return Ok(Err(refused)),
        },
    };
    if spec.ignored.contains(&Ignored::Fields) {
        messages.push(warning(
            Sentence::ViewFieldsIgnored,
            view_words::ignored_text(Ignored::Fields),
            &[("database", database)],
        ));
    }

    let lose = |skipped: &mut Vec<Skipped>, alias: &str, dimension: &str| {
        if let Some((_, at)) = walk.skipped_with.iter().find(|(name, _)| name == alias) {
            skipped[*at].dimensions.push(dimension.to_owned());
        }
    };
    // Each field's expression by folded name, `None` for one refused.
    let mut defined: Vec<(String, Option<String>)> = Vec::new();
    let define = |defined: &mut Vec<(String, Option<String>)>, name: &str, expr: Option<String>| {
        let folded = name.to_lowercase();
        match defined.iter_mut().find(|(n, _)| *n == folded) {
            Some(slot) => slot.1 = expr,
            None => defined.push((folded, expr)),
        }
    };
    let mut dimensions: Vec<Dimension> = Vec::new();
    for item in spec.columns.clone() {
        let column = match item {
            ColumnItem::NotAMapping { cfg } => {
                let text = view_words::dimension_not_a_mapping(spec, cfg).map_err(too_large)?;
                let dimension = view_words::value_repr(&spec.document, cfg).map_err(too_large)?;
                messages.push(warning(
                    Sentence::ViewDimensionNotAMapping,
                    text,
                    &[("dimension", &dimension)],
                ));
                continue;
            }
            ColumnItem::Column(column) => column,
        };
        let name = column.name.and_then(|n| spec.text(n)).map(str::to_owned);
        let expr = match column.expr.and_then(|e| spec.text(e)) {
            None => None,
            Some(written) => Some(stripped(ask, written)?),
        };
        // An expression whose comments could not be read is refused as one
        // that cannot be read.
        if let Some(Ok(written)) = &expr
            && let ViewTextAnswer::Wildcard(Some((path, excluded))) = ask(ViewTextAsk::Wildcard {
                expr: written.clone(),
            })?
        {
            let folded: Vec<String> = path.iter().map(|p| p.to_lowercase()).collect();
            let alias = candidates
                .iter()
                .rev()
                .find(|(p, _)| {
                    p.iter()
                        .map(|x| x.to_lowercase())
                        .eq(folded.iter().cloned())
                })
                .map(|(_, alias)| alias.clone());
            let shown = view_words::stripped_text(written);
            match alias {
                None => messages.push(warning(
                    Sentence::ViewDimensionNoTable,
                    format!(
                        "Failed to add dimension '{shown}' since it references no table of the view"
                    ),
                    &[("dimension", &shown)],
                )),
                Some(alias) if alias != SOURCE_ALIAS && !kept.contains(&alias) => {
                    lose(&mut skipped, &alias, &shown);
                }
                Some(alias) => dimensions.push(Dimension::Wildcard { alias, excluded }),
            }
            continue;
        }
        let (Some(name), Some(expr)) = (name, expr) else {
            let text = view_words::dimension_misses_text(spec, column.cfg).map_err(too_large)?;
            let dimension =
                view_words::value_repr(&spec.document, column.cfg).map_err(too_large)?;
            messages.push(warning(
                Sentence::ViewDimensionMisses,
                text,
                &[("dimension", &dimension)],
            ));
            continue;
        };
        let refuse = |messages: &mut Vec<CoreWarning>, reason: String| {
            messages.push(warning(
                Sentence::ViewDimensionRefused,
                format!("Failed to add dimension '{name}' since {reason}"),
                &[("dimension", &name), ("reason", &reason)],
            ));
        };
        let mut expr = match expr {
            Ok(expr) => expr,
            Err(unreadable) => {
                refuse(&mut messages, unreadable.to_string());
                define(&mut defined, &name, None);
                continue;
            }
        };
        if unsafe_text(ask, &expr)? {
            // Left for the expression's check to drop, saying why.
            define(&mut defined, &name, None);
        } else {
            let read = substituted(ask, &expr, parameters)?;
            let (substituted_expr, referenced) = match read {
                Ok(read) => read,
                Err(unreadable) => {
                    refuse(&mut messages, unreadable.to_string());
                    define(&mut defined, &name, None);
                    continue;
                }
            };
            let fields = match ask(ViewTextAsk::SubstituteFields {
                expr: substituted_expr,
                fields: defined.clone(),
                qualifiers: qualifiers.clone(),
            })? {
                ViewTextAnswer::Fields(read) => read,
                _ => return Err(mismatched("substitute_fields")),
            };
            let (fielded, refused) = match fields {
                Ok(read) => read,
                Err(unreadable) => {
                    refuse(&mut messages, unreadable.to_string());
                    define(&mut defined, &name, None);
                    continue;
                }
            };
            let star = match flag(
                ask,
                ViewTextAsk::ReadsStar {
                    expr: fielded.clone(),
                },
                "reads_star",
            )? {
                Ok(star) => star,
                Err(unreadable) => {
                    refuse(&mut messages, unreadable.to_string());
                    define(&mut defined, &name, None);
                    continue;
                }
            };
            let mut reason = unusable_parameter(spec, parameters, &referenced)?;
            if reason.is_none()
                && let Some(refused) = refused
            {
                reason = Some(format!(
                    "it names the dimension '{refused}', which is refused"
                ));
            }
            if reason.is_none() && star {
                reason = Some(STAR_REASON.to_owned());
            }
            if let Some(reason) = reason {
                refuse(&mut messages, reason);
                define(&mut defined, &name, None);
                continue;
            }
            used(&mut notes, parameters, &referenced);
            define(&mut defined, &name, Some(fielded.clone()));
            expr = fielded;
        }

        let mut refs: Vec<String> = Vec::new();
        let mut unreadable = None;
        for (path, alias) in &candidates {
            match flag(
                ask,
                ViewTextAsk::ReferencesAlias {
                    expr: expr.clone(),
                    path: path.clone(),
                },
                "references_alias",
            )? {
                Ok(true) => {
                    if !refs.contains(alias) {
                        refs.push(alias.clone());
                    }
                    match text(
                        ask,
                        ViewTextAsk::StripAlias {
                            expr: expr.clone(),
                            path: path.clone(),
                        },
                        "strip_alias",
                    )? {
                        Ok(rest) => expr = rest,
                        Err(why) => {
                            unreadable = Some(why);
                            break;
                        }
                    }
                }
                Ok(false) => {}
                Err(why) => {
                    unreadable = Some(why);
                    break;
                }
            }
        }
        if let Some(why) = unreadable {
            refuse(&mut messages, why.to_string());
            continue;
        }
        if refs.len() > 1 {
            messages.push(warning(
                Sentence::ViewDimensionMultipleTables,
                format!(
                    "Failed to add dimension '{name}' since its expression references multiple tables"
                ),
                &[("dimension", &name)],
            ));
            continue;
        }
        let alias = refs.pop().unwrap_or_else(|| SOURCE_ALIAS.to_owned());
        if alias != SOURCE_ALIAS && !kept.contains(&alias) {
            lose(&mut skipped, &alias, &name);
            continue;
        }
        dimensions.push(Dimension::Column { name, alias, expr });
    }

    // The dimensions by folded name, a later one of a name in the earlier's
    // place, as a dict keeps it.
    let mut by_name: Vec<(String, DimensionRef)> = Vec::new();
    for dimension in &dimensions {
        let Dimension::Column { name, alias, expr } = dimension else {
            continue;
        };
        let folded = name.to_lowercase();
        let entry = DimensionRef {
            name: name.clone(),
            alias: alias.clone(),
            expr: expr.clone(),
        };
        match by_name.iter_mut().find(|(n, _)| *n == folded) {
            Some(slot) => slot.1 = entry,
            None => by_name.push((folded, entry)),
        }
    }
    let refused: Vec<String> = defined
        .iter()
        .filter(|(name, field)| field.is_none() || !by_name.iter().any(|(n, _)| n == name))
        .map(|(name, _)| name.clone())
        .collect();
    let measure_inputs: Vec<MeasureInput> = spec
        .measures
        .iter()
        .map(|m| MeasureInput {
            name: m.name.clone(),
            expr: m.expr.and_then(|e| spec.text(e)).map(str::to_owned),
            window: m.window,
        })
        .collect();
    let (measures, left_out) = match ask(ViewTextAsk::Measures {
        measures: measure_inputs,
        candidates: candidates.clone(),
        kept: kept.clone(),
        dimensions: by_name,
        refused,
        parameters: parameters
            .iter()
            .map(|(folded, _)| folded.clone())
            .collect(),
    })? {
        ViewTextAnswer::Measures(read, left_out) => (read, left_out),
        _ => return Err(mismatched("measures")),
    };
    for refusal in &left_out {
        let text = view_words::measure_refusal_text(refusal);
        let reason = view_words::measure_reason_text(&refusal.reason);
        messages.push(warning(
            Sentence::ViewMeasureRefused,
            text,
            &[("measure", &refusal.name), ("reason", &reason)],
        ));
    }
    let mut window_orders: Vec<String> = Vec::new();
    for order in &spec.window_orders {
        let unquoted = match view_steps::names(
            ask,
            ViewTextAsk::TableReference {
                name: order.clone(),
            },
        )? {
            Some(parts) if parts.len() == 1 => parts[0].clone(),
            _ => order.trim().to_owned(),
        };
        if !window_orders.contains(&unquoted) {
            window_orders.push(unquoted);
        }
    }
    Ok(Ok(Definition {
        joins,
        skipped,
        candidates,
        filter,
        dimensions,
        measures,
        messages,
        notes,
        window_orders,
    }))
}

/// The filter read (`DB/metric_view.py:924-944`): its comments dropped,
/// its parameters read as their defaults, then checked against joined
/// tables and unsafe text (`_check_filter`).
fn read_filter(
    spec: &ViewSpec,
    node: NodeId,
    parameters: &[(String, Parameter)],
    candidates: &[(Vec<String>, String)],
    notes: &mut Vec<(String, String)>,
    ask: Ask<'_>,
) -> Result<Result<String, FilterRefused>, CoreError> {
    let written = spec.text(node);
    let row_filter = match written {
        Some(written) => match stripped(ask, written)? {
            Ok(text) => Some(text),
            Err(unreadable) => {
                return Ok(Err(FilterRefused {
                    filter: view_words::quote(written),
                    reason: unreadable.to_string(),
                    hatch: FilterHatch::None,
                }));
            }
        },
        None => None,
    };
    // The filter as the old client quoted it: the text it read, or the
    // value as written where it is no text.
    let shown = match &row_filter {
        Some(text) => view_words::quote(text),
        None => view_words::value_repr(&spec.document, node).map_err(too_large)?,
    };
    let refused = |reason: String, hatch: FilterHatch| {
        Ok(Err(FilterRefused {
            filter: shown.clone(),
            reason,
            hatch,
        }))
    };
    let mut checked = row_filter.clone();
    if let Some(filter) = &row_filter
        && !unsafe_text(ask, filter)?
    {
        let (substituted_filter, referenced) = match substituted(ask, filter, parameters)? {
            Ok(read) => read,
            Err(unreadable) => return refused(unreadable.to_string(), FilterHatch::None),
        };
        let star = match flag(
            ask,
            ViewTextAsk::ReadsStar {
                expr: filter.clone(),
            },
            "reads_star",
        )? {
            Ok(star) => star,
            Err(unreadable) => return refused(unreadable.to_string(), FilterHatch::None),
        };
        if let Some(reason) = unusable_parameter(spec, parameters, &referenced)? {
            return refused(reason, FilterHatch::Default);
        }
        if star {
            return refused(STAR_REASON.to_owned(), FilterHatch::None);
        }
        used(notes, parameters, &referenced);
        checked = Some(substituted_filter);
    }
    // A refusal of the check quotes the filter as checked, its parameters
    // read (`_check_filter(row_filter, ...)`).
    let shown = match &checked {
        Some(text) => view_words::quote(text),
        None => shown,
    };
    match ask(ViewTextAsk::Filter {
        filter: checked,
        candidates: candidates.to_vec(),
    })? {
        ViewTextAnswer::Filter(Ok(filter)) => Ok(Ok(filter)),
        ViewTextAnswer::Filter(Err(refusal)) => {
            let words = match &refusal {
                FilterRefusal::Unsafe(reason) => match ask(ViewTextAsk::ReasonWords {
                    reason: reason.clone(),
                })? {
                    ViewTextAnswer::Words(words) => words,
                    _ => return Err(mismatched("reason_words")),
                },
                _ => None,
            };
            Ok(Err(FilterRefused {
                filter: shown,
                reason: view_words::filter_reason_text(&refusal, words),
                hatch: FilterHatch::None,
            }))
        }
        _ => Err(mismatched("filter")),
    }
}
