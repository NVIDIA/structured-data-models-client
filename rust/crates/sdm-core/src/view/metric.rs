// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Databricks metric view's conversion, M1 to M11 of T3.5, in the old
//! client's order (`RE/graph.py:1369-1966`): the definition read
//! ([`super::read_definition`], M1), parsed (M2, the source table's columns
//! read first where the view has parameters), the source table's columns
//! (M3), the filter checked against them (M4), the dimensions on their
//! tables, each joined table's columns read as first needed (M5), the
//! measures' columns (M6), the source table's rows and its expressions'
//! samples (M7), each joined table's rows (M8), one count of every join's
//! key (M9), each join added or dropped (M10), then the notes, the time
//! columns and the model's checks (M11).

use std::collections::BTreeMap;

use sdm_contract::data::{DataRequest, Relation, TableKey, Unusable, ViewChoice};
use sdm_contract::view::{DeclaredType, ExprVerdict, ViewTextAnswer, ViewTextAsk};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_schema::infer::Previewed;
use sdm_schema::view::metric::SOURCE_ALIAS;
use sdm_schema::view::model::{
    self, JoinStep, KeyCheck, MeasureColumn, ModelColumn, ModelTable, ViewModel,
};
use sdm_schema::view::spec::ViewSpec;

use super::definition::{self, Definition, Dimension, Join, Skipped};
use super::{
    Added, Converted, Described, ReadTask, Reading, ViewTable, failed, failure_text, model_columns,
    preview,
};
use crate::discover;
use crate::pql::yaml_rules;
use crate::run::internal;
use crate::view_steps;
use crate::view_words::{self, FilterHatch};

/// A view refused while its definition is read, as `sdm-schema` reads it.
pub(super) fn refused(refusal: &sdm_schema::view::spec::ViewRefusal) -> CoreError {
    use sdm_schema::view::spec::RefusalReason as R;
    let kind = Detail::text(refusal.kind.noun());
    let error = |sentence| {
        CoreError::new(sentence, refusal.to_string()).with_field("view_kind", kind.clone())
    };
    // The old client's own refusals; where it failed unnamed, the view is
    // refused as one that cannot be converted, with why (the named
    // refusals P3J-1 to P3J-4 and P3J-11 propose wait for the owner).
    let reason = match &refusal.reason {
        R::NotAMapping => return error(Sentence::ViewNotAMapping),
        R::MissesSource => return error(Sentence::ViewMissesSource),
        R::Yaml(yaml) => format!("its definition is not readable YAML: {yaml}"),
        R::Misses(entry) => format!("its definition misses '{entry}'"),
        R::NotAList(section) => format!("its definition's '{section}' is not a list"),
        R::NestsItself => "its definition has a join nested in itself".to_owned(),
        R::Unhashable(entry) => format!("its definition's '{entry}' is a list or a mapping"),
        R::NotText(entry) => format!("its definition's '{entry}' is not text"),
    };
    super::unconvertible(refusal.kind.noun(), reason)
}

/// Why a joined table cannot be used, as a clause (`unusable_table_reason`,
/// `DB/errors.py:66-89`).
fn unusable_text(unusable: Unusable) -> &'static str {
    match unusable {
        Unusable::Missing => "its table does not exist or you may not see it",
        Unusable::Forbidden => "you may not read its table",
    }
}

const METRIC_JOIN: &str = "its table is a metric view; a metric view built on another metric \
                           view is not supported, point the join's source at the underlying table";
const EMPTY_JOIN: &str = "its table is empty";

/// The state of the conversion: the tables read so far, by alias, and why
/// each joined table that cannot be used cannot.
struct Metric<'r, 'a> {
    reading: &'r Reading<'a>,
    view: ViewChoice,
    located: Vec<Option<String>>,
    tables: Vec<(String, ViewTable)>,
    unusable: Vec<(String, String)>,
    joins: Vec<Join>,
    /// Whether each joined table's rows were read.
    rows_read: Vec<String>,
}

impl<'r, 'a> Metric<'r, 'a> {
    fn ask(&self, question: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        self.reading.ask(question)
    }

    fn table(&self, alias: &str) -> Option<&ViewTable> {
        self.tables.iter().find(|(a, _)| a == alias).map(|(_, t)| t)
    }

    fn table_mut(&mut self, alias: &str) -> Option<&mut ViewTable> {
        self.tables
            .iter_mut()
            .find(|(a, _)| a == alias)
            .map(|(_, t)| t)
    }

    /// A table's name parts filled with the view's own catalog and schema
    /// (`_table_parts`, `RE/graph.py:1523-1535`).
    fn qualified(&self, parts: &[String]) -> Result<Relation, CoreError> {
        let catalog = self.located.first().cloned().flatten();
        let schema = self.located.get(1).cloned().flatten();
        let mut full: Vec<String> = parts.to_vec();
        match (full.len(), &catalog, &schema) {
            (2, Some(catalog), _) => full.insert(0, catalog.clone()),
            (1, Some(catalog), Some(schema)) if !catalog.is_empty() && !schema.is_empty() => {
                full.splice(0..0, [catalog.clone(), schema.clone()]);
            }
            _ => {}
        }
        if full.len() != 3 {
            let table = parts.join(".");
            return Err(CoreError::new(
                Sentence::ViewSourceUnqualified,
                format!(
                    "Could not fully qualify table '{table}' referenced by metric view '{}'",
                    self.view.name
                ),
            )
            .with_field("table", Detail::text(table))
            .with_field("view", Detail::text(&self.view.name)));
        }
        Ok(Relation::of(full))
    }

    /// M3: the source table, its columns read (`_fact_table`). A source that
    /// is itself a metric view refuses the view.
    async fn fact_table(&mut self, parts: &[String]) -> Result<(), CoreError> {
        if self.table(SOURCE_ALIAS).is_some() {
            return Ok(());
        }
        let base = self.qualified(parts)?;
        let name = base.source.last().cloned().unwrap_or_default();
        let read = discover::columns(self.reading.asker, &base).await;
        let catalog = match read {
            Ok(Ok(Some(catalog))) => catalog,
            Ok(Ok(None)) => return Err(discover::not_held(self.reading.asker, &base).await),
            Ok(Err(error)) => return Err(failed(&error, "describe")),
            Err(error) if error.sentence == Sentence::MetricViewAsTable => {
                return Err(CoreError::new(
                    Sentence::ViewOnMetricView,
                    format!(
                        "'{name}' is a metric view; a metric view built on another metric view \
                         is not supported, point its source at the underlying table"
                    ),
                )
                .with_field("table", Detail::text(name)));
            }
            Err(error) => return Err(error),
        };
        self.tables
            .push((SOURCE_ALIAS.to_owned(), ViewTable::new(name, base, catalog)));
        Ok(())
    }

    /// Whether the joined table `alias` can be used, its columns read, and
    /// its rows if `rows` (`_read_join`): a join under one that cannot be
    /// used goes with it, never read.
    async fn read_join(&mut self, alias: &str, rows: bool) -> Result<bool, CoreError> {
        let mut chain = vec![alias.to_owned()];
        loop {
            let last = chain.last().cloned().unwrap_or_default();
            let parent = self
                .joins
                .iter()
                .find(|j| j.alias == last)
                .map(|j| j.parent_alias.clone())
                .ok_or_else(|| internal("a view's join names no join"))?;
            if parent == SOURCE_ALIAS {
                break;
            }
            chain.push(parent);
        }
        // Parents first, as the old client's recursion reads them.
        for at in (0..chain.len()).rev() {
            let alias = chain[at].clone();
            if self.unusable.iter().any(|(a, _)| *a == alias) {
                return Ok(false);
            }
            if !self.read_one(&alias, rows).await? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// One joined table's columns, and its rows if `rows`, read once.
    async fn read_one(&mut self, alias: &str, rows: bool) -> Result<bool, CoreError> {
        if self.table(alias).is_none() {
            let join = self
                .joins
                .iter()
                .find(|j| j.alias == alias)
                .cloned()
                .ok_or_else(|| internal("a view's join names no join"))?;
            let base = self.qualified(&join.table)?;
            let read = discover::columns(self.reading.asker, &base).await;
            let catalog = match read {
                Ok(Ok(Some(catalog))) => catalog,
                Ok(Ok(None)) => {
                    return self.unusable(alias, unusable_text(Unusable::Missing));
                }
                Ok(Err(error)) => match self.reading.data.failure(&error).unusable {
                    Some(why) => return self.unusable(alias, unusable_text(why)),
                    None => return Err(failed(&error, "describe")),
                },
                Err(error) if error.sentence == Sentence::MetricViewAsTable => {
                    return self.unusable(alias, METRIC_JOIN);
                }
                Err(error) => return Err(error),
            };
            self.tables.push((
                alias.to_owned(),
                ViewTable::new(alias.to_owned(), base, catalog),
            ));
        }
        if rows && !self.rows_read.iter().any(|a| a == alias) {
            match self.source_rows(alias).await? {
                Ok(()) => self.rows_read.push(alias.to_owned()),
                Err(why) => return self.unusable(alias, why),
            }
        }
        Ok(true)
    }

    fn unusable(&mut self, alias: &str, why: &str) -> Result<bool, CoreError> {
        self.unusable.push((alias.to_owned(), why.to_owned()));
        Ok(false)
    }

    /// A table's rows, every source column under the view's name
    /// (`_source_sample_df`): an empty table, or a failure that says the
    /// table cannot be used, is why (a joined table's); any other failure
    /// is the run's.
    async fn source_rows(&mut self, alias: &str) -> Result<Result<(), &'static str>, CoreError> {
        let table = self
            .table(alias)
            .ok_or_else(|| internal("a view's table was read before its columns"))?;
        let relation = table.source_relation();
        let columns = table.source_names();
        let base = table.base.clone();
        let read = match preview(self.reading.asker, relation, columns).await? {
            Ok(read) => read,
            Err(error) => {
                return match self.reading.data.failure(&error).unusable {
                    Some(why) if alias != SOURCE_ALIAS => Ok(Err(unusable_text(why))),
                    _ => Err(failed(&error, "preview")),
                };
            }
        };
        let rows: usize = read.batches.iter().map(|b| b.num_rows()).sum();
        if rows == 0 {
            if alias == SOURCE_ALIAS {
                return Err(sdm_schema::infer::empty_table(&base));
            }
            return Ok(Err(EMPTY_JOIN));
        }
        if let Some(table) = self.table_mut(alias) {
            table.rows = Some(read);
        }
        Ok(Ok(()))
    }

    /// The columns of the table `alias`, under the view's names; none for a
    /// joined table that cannot be used (`_columns_of`).
    async fn columns_of(&mut self, alias: &str) -> Result<Vec<String>, CoreError> {
        if alias != SOURCE_ALIAS && !self.read_join(alias, false).await? {
            return Ok(Vec::new());
        }
        Ok(self
            .table(alias)
            .map(ViewTable::source_names)
            .unwrap_or_default())
    }

    /// Why an expression of the view is refused, or none to keep it.
    fn verdict(&self, expr: &str, columns: Vec<String>) -> Result<Option<ExprVerdict>, CoreError> {
        match self.ask(ViewTextAsk::Expression {
            expr: expr.to_owned(),
            trust_expressions: self.view.trust_expressions,
            columns,
        })? {
            ViewTextAnswer::Expression(verdict) => Ok(verdict),
            _ => Err(internal(
                "the data layer read an expression as another answer",
            )),
        }
    }

    /// The refusal's clause, in the data layer's words where it has its own.
    fn reason(&self, verdict: &ExprVerdict) -> Result<String, CoreError> {
        let words = match self.ask(ViewTextAsk::ReasonWords {
            reason: verdict.reason.clone(),
        })? {
            ViewTextAnswer::Words(words) => words,
            _ => return Err(internal("the data layer worded a reason as another answer")),
        };
        Ok(view_words::expr_reason(&verdict.reason, words))
    }

    /// Each column of `columns` added to the table `alias`, leaving out each
    /// one the warehouse refuses, and the warning of each
    /// (`_add_view_columns`): the expressions sampled in one read, and, if
    /// the warehouse refuses it, one at a time.
    async fn add_columns(
        &mut self,
        alias: &str,
        columns: Vec<Added>,
        warnings: &mut Vec<CoreWarning>,
        omitted: &mut Vec<(String, String)>,
    ) -> Result<(), CoreError> {
        match self.try_add(alias, &columns).await? {
            None => return Ok(()),
            Some(_) if columns.len() == 1 => {}
            Some(_) => {}
        }
        let table_name = self
            .table(alias)
            .map(|t| t.name.clone())
            .unwrap_or_default();
        for column in columns {
            if let Some(why) = self.try_add(alias, std::slice::from_ref(&column)).await? {
                warnings.push(
                    CoreWarning::new(
                        Sentence::ViewColumnRefused,
                        format!(
                            "Failed to add column '{}' to table '{table_name}': {why}",
                            column.name
                        ),
                    )
                    .with_field("column", Detail::text(&column.name))
                    .with_field("table", Detail::text(&table_name))
                    .with_field("failure", Detail::text(&why)),
                );
                omitted.push((column.name.clone(), why));
            }
        }
        Ok(())
    }

    /// `columns` added to the table `alias`, or none of them and why
    /// (`_try_add_columns`): a refusal the warehouse gives an expression's
    /// sample, or a column the table cannot hold; any other failure is the
    /// run's.
    async fn try_add(
        &mut self,
        alias: &str,
        columns: &[Added],
    ) -> Result<Option<String>, CoreError> {
        if columns.is_empty() {
            return Ok(None);
        }
        let table = self
            .table(alias)
            .ok_or_else(|| internal("a view's columns were added to a table not read"))?;
        for column in columns {
            if table.added.iter().any(|a| a.name == column.name)
                || columns.iter().filter(|c| c.name == column.name).count() > 1
            {
                return Err(CoreError::new(
                    Sentence::ViewInvalid,
                    format!(
                        "Column '{}' already exists in table '{}'",
                        column.name, table.name
                    ),
                )
                .with_field("view_kind", Detail::text("metric view"))
                .with_field("view", Detail::text(&self.view.name))
                .with_field(
                    "reason",
                    Detail::text(format!(
                        "its column '{}' is named twice in table '{}'",
                        column.name, table.name
                    )),
                ));
            }
            if column.expr.is_none() {
                let declared = table.catalog_column(&column.name);
                match declared {
                    None => {
                        return Ok(Some(format!(
                            "Column '{}' does not exist in the underlying source table",
                            column.name
                        )));
                    }
                    Some(c) if c.dtype == Some(sdm_contract::ops::Dtype::Unsupported) => {
                        return Ok(Some(format!(
                            "Encountered unsupported data type for column '{}' in table '{}'. \
                             Please either change the column's data type or remove the column \
                             from this table.",
                            column.name, table.name
                        )));
                    }
                    Some(_) => {}
                }
            }
        }
        let expressions: Vec<(String, String)> = columns
            .iter()
            .filter_map(|c| c.expr.clone().map(|e| (c.name.clone(), e)))
            .collect();
        let mut sampled = Vec::new();
        if !expressions.is_empty() {
            let relation = table.expression_relation(&expressions);
            let names: Vec<String> = expressions.iter().map(|(n, _)| n.clone()).collect();
            let read = match preview(self.reading.asker, relation, names.clone()).await? {
                Ok(read) => read,
                Err(error) => {
                    if self.reading.data.failure(&error).refuses_expression {
                        return Ok(Some(failure_text(&error, ReadTask::Expression)));
                    }
                    return Err(failed(&error, "preview"));
                }
            };
            let rows: usize = read.batches.iter().map(|b| b.num_rows()).sum();
            if rows == 0 {
                return Ok(Some(format!(
                    "Table '{}' is empty",
                    table.base.source.join(".")
                )));
            }
            for name in &names {
                let index = read
                    .schema
                    .index_of(name)
                    .map_err(|_| internal("an expression's sample lacks its column"))?;
                let arrays: Vec<&dyn arrow_array::Array> = read
                    .batches
                    .iter()
                    .map(|b| b.column(index).as_ref())
                    .collect();
                let joined = arrow_select::concat::concat(&arrays)
                    .map_err(|_| internal("an expression's sample does not concatenate"))?;
                sampled.push((name.clone(), read.schema.field(index).clone(), joined));
            }
        }
        let reading = self.reading;
        let table = self
            .table_mut(alias)
            .ok_or_else(|| internal("a view's columns were added to a table not read"))?;
        let before = table.sampled.len();
        table.sampled.extend(sampled);
        // A column whose type the engine cannot read is refused with the
        // batch, as `add_columns` refuses it.
        if let Err(error) = table.infer(reading, columns) {
            table.sampled.truncate(before);
            if error.sentence == Sentence::Internal {
                return Err(error);
            }
            return Ok(Some(error.message));
        }
        table.added.extend(columns.iter().cloned());
        Ok(None)
    }
}

/// The view's described type of a dimension (`DatabricksTable._to_dtype`).
fn declared(
    reading: &Reading<'_>,
    described: &[(String, String)],
    name: &str,
) -> Result<Option<sdm_contract::ops::Dtype>, CoreError> {
    let Some((_, data_type)) = described.iter().find(|(n, _)| n == name) else {
        return Ok(None);
    };
    match reading.ask(ViewTextAsk::DeclaredType {
        data_type: data_type.clone(),
    })? {
        ViewTextAnswer::DeclaredType(DeclaredType::Known(dtype)) => Ok(Some(dtype)),
        ViewTextAnswer::DeclaredType(DeclaredType::Unsupported) => {
            Ok(Some(sdm_contract::ops::Dtype::Unsupported))
        }
        ViewTextAnswer::DeclaredType(DeclaredType::Unknown) => Ok(None),
        _ => Err(internal("the data layer read a type as another answer")),
    }
}

/// `name` among `columns` without regard to case, the first of a folded
/// name (`by_folded_name`).
fn folded<'n>(columns: &'n [String], name: &str) -> Option<&'n String> {
    let name = name.to_lowercase();
    columns.iter().find(|c| c.to_lowercase() == name)
}

/// The keys of `join`: the table it keys and its key, then the table
/// linking to that one and its key (`_join_ends`), each key as its table's
/// source spells it; none for a condition that resolves into no key.
fn join_ends(
    metric: &Metric<'_, '_>,
    join: &Join,
) -> Result<Option<[(String, String); 2]>, CoreError> {
    let columns = |alias: &str| {
        metric
            .table(alias)
            .map(ViewTable::source_names)
            .unwrap_or_default()
    };
    let resolved = match metric.ask(ViewTextAsk::JoinKeys {
        join: join.as_ref(),
        child_columns: columns(&join.alias),
        parent_columns: columns(&join.parent_alias),
    })? {
        ViewTextAnswer::JoinKeys(resolved) => resolved,
        _ => return Err(internal("the data layer resolved a join as another answer")),
    };
    let Some((other_alias, other_key, child_key)) = resolved else {
        return Ok(None);
    };
    let real = |alias: &str, key: &str| {
        let names = columns(alias);
        if names.iter().any(|n| n == key) {
            return key.to_owned();
        }
        folded(&names, key)
            .cloned()
            .unwrap_or_else(|| key.to_owned())
    };
    let other = (other_alias.clone(), real(&other_alias, &other_key));
    let child = (join.alias.clone(), real(&join.alias, &child_key));
    Ok(Some(if join.many_to_one {
        [child, other]
    } else {
        [other, child]
    }))
}

/// A scalar as Python's `repr` writes it, in a message.
fn scalar_repr(value: &Scalar) -> String {
    match value {
        Scalar::Utf8(text) => view_words::quote(text),
        Scalar::Int64(n) => n.to_string(),
        Scalar::Int32(n) => n.to_string(),
        Scalar::UInt64(n) => n.to_string(),
        Scalar::Float64(x) => crate::pql::float_text(*x),
        Scalar::Bool(b) => if *b { "True" } else { "False" }.to_owned(),
        Scalar::Null => "None".to_owned(),
        other => format!("{other:?}"),
    }
}

/// Converts a metric view (M2 to M11), its definition read (M1).
pub(super) async fn convert(
    reading: &Reading<'_>,
    view: &ViewChoice,
    described: Described,
) -> Result<Converted, CoreError> {
    let ask = |question: ViewTextAsk| reading.ask(question);
    // M2: the definition parsed, its source a plain table name.
    let mut spec: ViewSpec = sdm_schema::view::metric::parse(&described.text, &yaml_rules())
        .map_err(|refusal| refused(&refusal))?;
    let source = match view_steps::fact_source(&spec, &ask)? {
        Ok(parts) => parts,
        Err(Some(node)) => {
            let shown =
                view_words::value_repr(&spec.document, node).map_err(definition::too_large)?;
            let text =
                view_words::unsupported_source_text(&spec, node).map_err(definition::too_large)?;
            return Err(CoreError::new(Sentence::ViewSourceUnsupported, text)
                .with_field("source", Detail::text(shown)));
        }
        Err(None) => return Err(internal("a metric view's definition gives no source")),
    };
    let mut metric = Metric {
        reading,
        view: view.clone(),
        located: described.located.clone(),
        tables: Vec::new(),
        unusable: Vec::new(),
        joins: Vec::new(),
        rows_read: Vec::new(),
    };
    let mut parameters = view_steps::parameters(&spec, &described.parameters, &ask)?;
    if !parameters.is_empty() {
        // The source table's columns first, a column shadowing a parameter
        // of its folded name (`DB/metric_view.py:799-802`).
        metric.fact_table(&source).await?;
        let columns = metric
            .table(SOURCE_ALIAS)
            .map(ViewTable::source_names)
            .unwrap_or_default();
        parameters.retain(|(name, _)| !columns.iter().any(|c| c.to_lowercase() == *name));
    }
    let database = reading.asker.source_database().unwrap_or_default();
    let read: Definition = match definition::read(&mut spec, &parameters, &ask, database)? {
        Ok(read) => read,
        Err(refused) => return Err(refused.error()),
    };
    let mut warnings: Vec<CoreWarning> = read.messages.clone();
    let mut skipped: Vec<Skipped> = read.skipped.clone();
    metric.joins = read.joins.clone();

    // M3: the source table.
    metric.fact_table(&source).await?;
    let fact_name = metric
        .table(SOURCE_ALIAS)
        .map(|t| t.name.clone())
        .unwrap_or_default();
    if metric.joins.iter().any(|j| j.alias == fact_name) {
        return Err(CoreError::new(
            Sentence::ViewJoinNameCollides,
            format!(
                "Cannot convert metric view '{}' into a graph since the join name '{fact_name}' \
                 collides with the name of the source table",
                view.name
            ),
        )
        .with_field("view", Detail::text(&view.name))
        .with_field("table", Detail::text(&fact_name)));
    }

    // M4: the filter, attached before any row is read.
    if let Some(filter) = &read.filter {
        if !view.trust_expressions {
            let columns = metric.columns_of(SOURCE_ALIAS).await?;
            if let Some(verdict) = metric.verdict(filter, columns)? {
                let reason = metric.reason(&verdict)?;
                let refused = definition::FilterRefused {
                    filter: view_words::quote(filter),
                    reason,
                    hatch: FilterHatch::Trust,
                };
                return Err(refused.error());
            }
        }
        if let Some(fact) = metric.table_mut(SOURCE_ALIAS) {
            fact.filter = Some(filter.clone());
        }
    }

    // M5: the dimensions, each on its table.
    let mut by_alias: Vec<(String, Vec<Added>)> = Vec::new();
    let push = |by_alias: &mut Vec<(String, Vec<Added>)>, alias: &str, added: Added| match by_alias
        .iter_mut()
        .find(|(a, _)| a == alias)
    {
        Some((_, columns)) => columns.push(added),
        None => by_alias.push((alias.to_owned(), vec![added])),
    };
    let mut renames: Vec<(String, Vec<(String, String)>)> = Vec::new();
    let rename = |renames: &mut Vec<(String, Vec<(String, String)>)>,
                  alias: &str,
                  from: String,
                  to: String| {
        match renames.iter_mut().find(|(a, _)| a == alias) {
            Some((_, names)) => {
                if !names.iter().any(|(f, _)| *f == from) {
                    names.push((from, to));
                }
            }
            None => renames.push((alias.to_owned(), vec![(from, to)])),
        }
    };
    let renamed_from = |renames: &[(String, Vec<(String, String)>)], alias: &str, column: &str| {
        renames
            .iter()
            .find(|(a, _)| a == alias)
            .is_some_and(|(_, names)| names.iter().any(|(f, _)| f == column))
    };
    let mut omitted_columns: Vec<(String, String, String)> = Vec::new();
    for dimension in &read.dimensions {
        match dimension {
            Dimension::Wildcard { alias, excluded } => {
                let excluded: Vec<String> = excluded.iter().map(|e| e.to_lowercase()).collect();
                for name in metric.columns_of(alias).await? {
                    if !excluded.contains(&name.to_lowercase()) {
                        push(
                            &mut by_alias,
                            alias,
                            Added {
                                column: Some(name.clone()),
                                name,
                                expr: None,
                                declared: None,
                                declared_type: None,
                            },
                        );
                    }
                }
            }
            Dimension::Column { name, alias, expr } => {
                let columns = metric.columns_of(alias).await?;
                let named = match metric.ask(ViewTextAsk::RenamedColumn { expr: expr.clone() })? {
                    ViewTextAnswer::Column(named) => named,
                    _ => return Err(internal("the data layer read a column as another answer")),
                };
                let real = named
                    .as_deref()
                    .and_then(|named| folded(&columns, named))
                    .cloned();
                if let Some(real) = real.filter(|r| r.to_lowercase() == name.to_lowercase()) {
                    // A column named as itself in any case is that column,
                    // named as the view names it.
                    if real != *name {
                        rename(&mut renames, alias, real.clone(), name.clone());
                    }
                    push(
                        &mut by_alias,
                        alias,
                        Added {
                            name: name.clone(),
                            column: Some(real),
                            expr: None,
                            declared: None,
                            declared_type: None,
                        },
                    );
                } else if let Some(verdict) = metric.verdict(expr, columns)? {
                    let reason = metric.reason(&verdict)?;
                    warnings.push(view_words::expression_dropped(
                        sdm_contract::view::ViewKind::Metric,
                        name,
                        None,
                        &reason,
                        verdict.untrusted,
                    ));
                    let table = metric
                        .table(alias)
                        .map_or_else(|| alias.clone(), |t| t.name.clone());
                    omitted_columns.push((table, name.clone(), reason));
                } else {
                    push(
                        &mut by_alias,
                        alias,
                        Added {
                            name: name.clone(),
                            column: None,
                            expr: Some(expr.clone()),
                            declared: declared(reading, &described.columns, name)?,
                            declared_type: described
                                .columns
                                .iter()
                                .find(|(n, _)| n == name)
                                .map(|(_, t)| t.clone()),
                        },
                    );
                }
            }
        }
    }

    // M6: the columns a measure reads, added unless a dimension stands for
    // them or holds something else under their name.
    for measure in &read.measures {
        for (alias, name) in &measure.inputs {
            let columns = metric.columns_of(alias).await?;
            let Some(column) = folded(&columns, name).cloned() else {
                continue;
            };
            let taken: Vec<Added> = by_alias
                .iter()
                .find(|(a, _)| a == alias)
                .map(|(_, c)| c.clone())
                .unwrap_or_default();
            let mut stands = false;
            for spec in taken.iter().filter(|s| s.expr.is_some()) {
                let renamed = match metric.ask(ViewTextAsk::RenamedColumn {
                    expr: spec.expr.clone().unwrap_or_default(),
                })? {
                    ViewTextAnswer::Column(renamed) => renamed,
                    _ => return Err(internal("the data layer read a column as another answer")),
                };
                if renamed.is_some_and(|r| r.to_lowercase() == column.to_lowercase()) {
                    stands = true;
                    break;
                }
            }
            if stands {
                continue;
            }
            match taken
                .iter()
                .find(|s| s.name.to_lowercase() == column.to_lowercase())
            {
                None => push(
                    &mut by_alias,
                    alias,
                    Added {
                        name: column.clone(),
                        column: Some(column.clone()),
                        expr: None,
                        declared: None,
                        declared_type: None,
                    },
                ),
                Some(same) if same.expr.is_some() => warnings.push(
                    CoreWarning::new(
                        Sentence::ViewMeasureColumnShadowed,
                        format!(
                            "Did not add the column '{column}', which the measure '{}' reads, \
                             since the dimension '{}' has its name",
                            measure.name, same.name
                        ),
                    )
                    .with_field("column", Detail::text(&column))
                    .with_field("measure", Detail::text(&measure.name))
                    .with_field("dimension", Detail::text(&same.name)),
                ),
                Some(_) => {}
            }
        }
    }

    // Each key a join names, as a dimension, wildcard or measure names it,
    // else as the join's condition spells it.
    for join in metric.joins.clone() {
        let child = metric.columns_of(&join.alias).await?;
        let parent = metric.columns_of(&join.parent_alias).await?;
        let resolved = match metric.ask(ViewTextAsk::JoinKeys {
            join: join.as_ref(),
            child_columns: child,
            parent_columns: parent,
        })? {
            ViewTextAnswer::JoinKeys(resolved) => resolved,
            _ => return Err(internal("the data layer resolved a join as another answer")),
        };
        let Some((other_alias, other_key, child_key)) = resolved else {
            continue;
        };
        for (alias, key) in [(join.alias.clone(), child_key), (other_alias, other_key)] {
            let columns = metric.columns_of(&alias).await?;
            let real = folded(&columns, &key).cloned();
            let named = by_alias
                .iter()
                .find(|(a, _)| *a == alias)
                .is_some_and(|(_, c)| {
                    c.iter()
                        .any(|s| s.name.to_lowercase() == key.to_lowercase())
                });
            if let Some(real) = real
                && real != key
                && !renamed_from(&renames, &alias, &real)
                && !named
            {
                rename(&mut renames, &alias, real, key);
            }
        }
    }
    // Before a row is read, so every read names them as the view does.
    for (alias, names) in &renames {
        if metric.unusable.iter().any(|(a, _)| a == alias) {
            continue;
        }
        if let Some(table) = metric.table_mut(alias) {
            table.renames.clone_from(names);
        }
    }
    let mut model = ViewModel::new();
    // M7: the source table's rows, then its expressions' samples.
    if let Err(why) = metric.source_rows(SOURCE_ALIAS).await? {
        return Err(internal(why));
    }
    let fact_columns = by_alias
        .iter()
        .find(|(a, _)| a == SOURCE_ALIAS)
        .map(|(_, c)| c.clone())
        .unwrap_or_default();
    let mut refused_columns = Vec::new();
    metric
        .add_columns(
            SOURCE_ALIAS,
            fact_columns,
            &mut warnings,
            &mut refused_columns,
        )
        .await?;
    for (column, why) in refused_columns.drain(..) {
        omitted_columns.push((fact_name.clone(), column, why));
    }
    let mut inferred: BTreeMap<String, Previewed> = BTreeMap::new();
    {
        let fact = metric
            .table(SOURCE_ALIAS)
            .ok_or_else(|| internal("a metric view's source table was not read"))?;
        let previewed = fact.infer(reading, &[])?;
        model
            .add_table(ModelTable {
                name: fact.name.clone(),
                columns: model_columns(fact, &previewed),
                primary_key: None,
            })
            .map_err(|_| internal("a metric view's source table was added twice"))?;
        inferred.insert(fact.name.clone(), previewed);
    }

    // M8 and M9: each joined table's rows, then one count of every join's
    // key.
    let mut keyed: Vec<(String, String)> = Vec::new();
    for join in metric.joins.clone() {
        if !metric.read_join(&join.alias, true).await? {
            continue;
        }
        let Some([one, _]) = join_ends(&metric, &join)? else {
            continue;
        };
        if !keyed
            .iter()
            .any(|(a, k)| *a == one.0 && k.to_lowercase() == one.1.to_lowercase())
        {
            keyed.push(one);
        }
    }
    let mut counts: Option<Vec<(KeyCheckRead, Option<String>)>> = None;
    let mut uncounted = String::new();
    if !keyed.is_empty() {
        let keys: Vec<TableKey> = keyed
            .iter()
            .map(|(alias, key)| TableKey {
                table: metric
                    .table(alias)
                    .map(ViewTable::source_relation)
                    .unwrap_or_else(|| Relation::named(alias.clone())),
                columns: vec![key.clone()],
                label: None,
                unique: false,
            })
            .collect();
        match crate::describe::attempt(reading.asker, DataRequest::KeyCounts { keys: keys.clone() })
            .await?
        {
            Ok(answered) => {
                let mut read = sdm_schema::catalog::key_counts(&answered.batches)?;
                read.sort_by_key(|c| c.position);
                if read.len() != keys.len() {
                    return Err(internal("a key count does not hold one row per key"));
                }
                let mut held = Vec::with_capacity(read.len());
                for (count, key) in read.iter().zip(&keys) {
                    let number = |n: i64| u64::try_from(n).unwrap_or(0);
                    let (rows, distinct, missing) = (
                        number(count.rows),
                        number(count.distinct),
                        number(count.missing),
                    );
                    let repeats = rows.saturating_sub(distinct + missing);
                    let mut example = None;
                    if repeats > 0 {
                        let found = crate::describe::attempt(
                            reading.asker,
                            DataRequest::RepeatedKey {
                                table: key.table.clone(),
                                key: key.columns.clone(),
                            },
                        )
                        .await?;
                        let found = match found {
                            Ok(found) => found,
                            Err(error) => {
                                uncounted = failure_text(&error, ReadTask::JoinKeys);
                                held.clear();
                                break;
                            }
                        };
                        example = sdm_schema::keys::example(&found.batches)
                            .and_then(|row| row.first().map(scalar_repr));
                    }
                    held.push((
                        KeyCheckRead {
                            rows,
                            distinct,
                            repeats,
                        },
                        example,
                    ));
                }
                if held.len() == keys.len() {
                    counts = Some(held);
                }
            }
            Err(error) => uncounted = failure_text(&error, ReadTask::JoinKeys),
        }
    }

    // M10: each join, in order, added or dropped with what goes with it.
    let mut added_aliases = vec![SOURCE_ALIAS.to_owned()];
    let mut omitted_tables: Vec<(String, String)> = Vec::new();
    for join in metric.joins.clone() {
        let dimensions: Vec<String> = by_alias
            .iter()
            .find(|(a, _)| *a == join.alias)
            .map(|(_, c)| c.iter().map(|s| s.name.clone()).collect())
            .unwrap_or_default();
        let mut drop = |warnings: &mut Vec<CoreWarning>, reason: String| {
            let skip = Skipped {
                alias: join.alias.clone(),
                reason: reason.clone(),
                nested: Vec::new(),
                dimensions: dimensions.clone(),
            };
            warnings.push(definition::skipped_warning(&skip));
            omitted_tables.push((join.alias.clone(), reason));
        };
        if !added_aliases.contains(&join.parent_alias) {
            drop(
                &mut warnings,
                format!("its parent join '{}' was not added", join.parent_alias),
            );
            continue;
        }
        if !metric.read_join(&join.alias, true).await? {
            let why = metric
                .unusable
                .iter()
                .find(|(a, _)| *a == join.alias)
                .map(|(_, w)| w.clone())
                .unwrap_or_default();
            drop(&mut warnings, why);
            continue;
        }
        let columns = by_alias
            .iter()
            .find(|(a, _)| *a == join.alias)
            .map(|(_, c)| c.clone())
            .unwrap_or_default();
        let mut join_warnings = Vec::new();
        let mut refused_here = Vec::new();
        metric
            .add_columns(&join.alias, columns, &mut join_warnings, &mut refused_here)
            .await?;
        let Some([one, many]) = join_ends(&metric, &join)? else {
            drop(
                &mut warnings,
                format!(
                    "its join condition could not be resolved into a key reference between \
                     '{}' and '{}'",
                    join.parent_alias, join.alias
                ),
            );
            continue;
        };
        let table = metric
            .table(&join.alias)
            .ok_or_else(|| internal("a join's table was not read"))?;
        let previewed = table.infer(reading, &[])?;
        let name_of = |alias: &str| {
            metric
                .table(alias)
                .map_or_else(|| alias.to_owned(), |t| t.name.clone())
        };
        let (one_table, many_table) = (name_of(&one.0), name_of(&many.0));
        let count = match &counts {
            None => KeyCheck::Failed,
            Some(held) => {
                let at = keyed
                    .iter()
                    .position(|(a, k)| *a == one.0 && k.to_lowercase() == one.1.to_lowercase());
                match at.and_then(|at| held.get(at)) {
                    Some((read, _)) => KeyCheck::Counted {
                        rows: Some(read.rows),
                        distinct: Some(read.distinct),
                        repeats: read.repeats,
                    },
                    None => KeyCheck::Counted {
                        rows: None,
                        distinct: None,
                        repeats: 0,
                    },
                }
            }
        };
        let example = counts.as_ref().and_then(|held| {
            keyed
                .iter()
                .position(|(a, k)| *a == one.0 && k.to_lowercase() == one.1.to_lowercase())
                .and_then(|at| held.get(at))
                .and_then(|(_, example)| example.clone())
        });
        let mut source_columns = Vec::new();
        for (alias, key) in [&one, &many] {
            if let Some(column) = source_model_column(&metric, reading, alias, key)? {
                source_columns.push((name_of(alias), column));
            }
        }
        let step = JoinStep {
            table: ModelTable {
                name: table.name.clone(),
                columns: model_columns(table, &previewed),
                primary_key: None,
            },
            one: (one_table.clone(), one.1.clone()),
            many: (many_table, many.1.clone()),
            count,
            source_columns,
        };
        match model.add_join(step) {
            Ok(notes) => {
                warnings.extend(join_warnings);
                for (column, why) in refused_here {
                    omitted_columns.push((join.alias.clone(), column, why));
                }
                for note in notes {
                    let text = view_words::join_note_text(&note, &join.alias, &uncounted);
                    warnings.push(match &note {
                        model::JoinNote::KeptUnchecked { key } => {
                            CoreWarning::new(Sentence::ViewJoinUnchecked, text)
                                .with_field("join", Detail::text(&join.alias))
                                .with_field("key", Detail::text(key))
                                .with_field("failure", Detail::text(&uncounted))
                        }
                        model::JoinNote::ReplacedDerived { key, table } => {
                            CoreWarning::new(Sentence::ViewDerivedKeyReplaced, text)
                                .with_field("key", Detail::text(key))
                                .with_field("table", Detail::text(table))
                                .with_field("join", Detail::text(&join.alias))
                        }
                    });
                }
                inferred.insert(table.name.clone(), previewed);
                added_aliases.push(join.alias.clone());
            }
            Err(refusal) => {
                drop(
                    &mut warnings,
                    view_words::join_refusal_text(&refusal, example.as_deref()),
                );
            }
        }
    }
    for skip in &skipped {
        omitted_tables.push((skip.alias.clone(), skip.reason.clone()));
        for nested in &skip.nested {
            omitted_tables.push((nested.clone(), skip.reason.clone()));
        }
    }
    warnings.extend(skipped.drain(..).map(|s| definition::skipped_warning(&s)));

    // M11: the measures' notes, each table's time column, the model's
    // checks.
    let mut notes = Vec::new();
    let fact = metric
        .table(SOURCE_ALIAS)
        .ok_or_else(|| internal("a metric view's source table was not read"))?;
    let mut measure_columns = Vec::new();
    for added in &fact.added {
        let renames = match &added.expr {
            None => None,
            Some(expr) => match reading.ask(ViewTextAsk::RenamedColumn { expr: expr.clone() })? {
                ViewTextAnswer::Column(renamed) => renamed,
                _ => return Err(internal("the data layer read a column as another answer")),
            },
        };
        measure_columns.push(MeasureColumn {
            name: added.name.clone(),
            expression: added.expr.is_some(),
            renames,
        });
    }
    let measures: Vec<(String, Option<model::Aggregation>)> = read
        .measures
        .iter()
        .map(|m| (m.name.clone(), m.aggregation.clone()))
        .collect();
    for note in model::measure_notes(
        &fact.name,
        &measure_columns,
        &fact.source_names(),
        &measures,
    ) {
        notes.push(super::note(
            Sentence::ViewMeasureNote,
            view_words::measure_note_text(&note),
            &[
                ("measure", note.measure.clone()),
                ("function", note.function.clone()),
                ("table", view_words::pql_name(&note.table)),
                (
                    "column",
                    note.column
                        .as_deref()
                        .map_or("*".to_owned(), view_words::pql_name),
                ),
            ],
            Some(&note.table),
            None,
        ));
    }
    // Each table as the model holds it now: a key a join added from the
    // source, or put in place of an expression, read as the source's column.
    for logical in &model.tables {
        let Some(table) = metric
            .tables
            .iter_mut()
            .map(|(_, t)| t)
            .find(|t| t.name == logical.name)
        else {
            continue;
        };
        super::sync(table, logical);
    }
    let mut times: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    for logical in model.tables.clone() {
        let table = metric
            .tables
            .iter_mut()
            .map(|(_, t)| t)
            .find(|t| t.name == logical.name)
            .ok_or_else(|| internal("a view's model holds a table not read"))?;
        let taken = super::time_column(reading, table, &read.window_orders).await?;
        if let Some(note) = &taken.note {
            notes.push(super::time_note(note));
        }
        if let Some(model_table) = model.tables.iter_mut().find(|t| t.name == logical.name) {
            for column in model_columns(table, &taken.previewed) {
                if model_table.column(&column.name).is_none() {
                    model_table.columns.push(column);
                }
            }
        }
        inferred.insert(logical.name.clone(), taken.previewed);
        times.insert(
            logical.name.clone(),
            (taken.choice.column, taken.choice.fallback),
        );
    }
    for (edge, reason) in model.drop_invalid_edges() {
        warnings.push(
            CoreWarning::new(
                Sentence::ViewEdgeDropped,
                view_words::dropped_edge_text(&edge, &reason),
            )
            .with_field("source_table", Detail::text(&edge.source))
            .with_field("target_table", Detail::text(&edge.target))
            .with_field(
                "reason",
                Detail::text(view_words::edge_reason_text(&edge, &reason)),
            ),
        );
    }
    let time_choices = super::time_choices(&model, &times);
    let times = super::settle_times(&model, times);
    model
        .validate()
        .map_err(|refusal| super::invalid(view, "metric view", &refusal))?;
    for (name, text) in &read.notes {
        notes.push(super::note(
            Sentence::ViewParameterNote,
            view_words::parameter_note_text(name, text),
            &[("parameter", name.clone()), ("default", text.clone())],
            None,
            None,
        ));
    }
    let tables: Vec<ViewTable> = metric.tables.into_iter().map(|(_, t)| t).collect();
    let (description, relations) = super::describe(&model, &tables, &inferred, &times)?;
    Ok(Converted {
        description,
        relations,
        time_choices,
        notes,
        warnings,
        omitted_tables,
        omitted_columns,
    })
}

/// A key's count, read.
#[derive(Clone, Copy)]
struct KeyCheckRead {
    rows: u64,
    distinct: u64,
    repeats: u64,
}

/// The column `key` of the table `alias` as its source holds it, for a key
/// a join adds or replaces (`key_table.add_column(key)`).
fn source_model_column(
    metric: &Metric<'_, '_>,
    reading: &Reading<'_>,
    alias: &str,
    key: &str,
) -> Result<Option<ModelColumn>, CoreError> {
    let Some(table) = metric.table(alias) else {
        return Ok(None);
    };
    let Some(column) = table.source_of(key) else {
        return Ok(None);
    };
    let added = Added {
        name: key.to_owned(),
        column: Some(column),
        expr: None,
        declared: None,
        declared_type: None,
    };
    let mut alone = ViewTable::new(
        table.name.clone(),
        table.base.clone(),
        table.catalog.clone(),
    );
    alone.renames.clone_from(&table.renames);
    alone.rows.clone_from(&table.rows);
    let previewed = alone.infer(reading, std::slice::from_ref(&added))?;
    Ok(previewed.table().column(key).map(|c| ModelColumn {
        name: c.name.clone(),
        expr: None,
        dtype: c.dtype,
        stype: c.stype,
    }))
}
