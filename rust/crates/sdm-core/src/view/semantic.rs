// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A Snowflake semantic view's conversion, V1 to V7 of T3.5, in the old
//! client's order (`RE/graph.py:1967-2275`): the definition read
//! ([`super::read_definition`], V1); then each table in turn, finished
//! before the next is read (V2 to V5): its base table and key (V2), each
//! column, an expression naming another table dropped and any other checked
//! against the base table's columns, which are read then (V3), every column
//! added at once, the expressions sampled in one read and then the base
//! table's rows (V4), and its time column (V5); then the relationships
//! (V6), and the model's checks (V7).
//!
//! A column whose expression the warehouse refuses fails the read, as in
//! the old client, where a metric view leaves it out (old-behaviour row 57;
//! proposed to leave it out on both).

use std::collections::BTreeMap;

use sdm_contract::data::{Relation, ViewChoice};
use sdm_contract::view::{DeclaredType, ViewTextAnswer, ViewTextAsk};
use sdm_contract::{CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_schema::infer::Previewed;
use sdm_schema::view::model::{
    LinkFailure, ModelColumn, ModelTable, RelationshipRefusal, RelationshipStep, ViewModel,
};
use sdm_schema::view::semantic;
use sdm_schema::view::spec::{PrimaryKey, Section, Source, TableSpec, ViewSpec};
use sdm_schema::view::yaml::NodeId;

use super::definition::too_large;
use super::metric::refused;
use super::{Added, Converted, Described, Reading, ViewTable, failed, model_columns, preview};
use crate::discover::{self, Catalog};
use crate::pql::yaml_rules;
use crate::run::internal;
use crate::view_words;

/// A value of the definition as Python's `str` writes it.
fn text(spec: &ViewSpec, node: NodeId) -> Result<String, CoreError> {
    view_words::value_text(&spec.document, node).map_err(too_large)
}

/// The base table's catalog, read once, its name tried in each spelling the
/// data layer gives until one resolves (design S2: Snowflake's as written,
/// then upper-cased), that spelling kept for the run.
async fn base_catalog(
    reading: &Reading<'_>,
    written: &Relation,
) -> Result<(Relation, Catalog), CoreError> {
    let spellings = reading.data.spellings(written);
    let last = spellings.len().saturating_sub(1);
    for (at, spelling) in spellings.into_iter().enumerate() {
        match discover::columns(reading.asker, &spelling).await? {
            Ok(Some(catalog)) => return Ok((spelling, catalog)),
            Ok(None) if at < last => {}
            Ok(None) => break,
            Err(error) if at < last && reading.data.failure(&error).not_found => {}
            Err(error) if reading.data.failure(&error).not_found => break,
            Err(error) => return Err(failed(&error, "describe")),
        }
    }
    Err(discover::not_held(reading.asker, written).await)
}

/// A table of the view being converted: its base table, read only once a
/// step needs it.
struct Pending {
    name: String,
    written: Relation,
    table: Option<ViewTable>,
}

impl Pending {
    async fn read(&mut self, reading: &Reading<'_>) -> Result<&mut ViewTable, CoreError> {
        if self.table.is_none() {
            let (base, catalog) = base_catalog(reading, &self.written).await?;
            self.table = Some(ViewTable::new(self.name.clone(), base, catalog));
        }
        self.table
            .as_mut()
            .ok_or_else(|| internal("a semantic view's table was not read"))
    }
}

/// Why a table cannot hold what the view declares, refusing the view
/// (`RE/graph.py:2174-2189`).
fn cannot_hold(view: &ViewChoice, table: &str, reason: &str) -> CoreError {
    CoreError::new(
        Sentence::ViewTableCannotHold,
        format!(
            "Cannot convert semantic view '{}' into a graph since its table '{table}' cannot \
             hold what it declares: {reason}",
            view.name
        ),
    )
    .with_field("view", Detail::text(&view.name))
    .with_field("table", Detail::text(table))
    .with_field("reason", Detail::text(reason))
}

/// The base table's rows, every column (`_source_sample_df`), read once.
async fn source_rows(reading: &Reading<'_>, table: &mut ViewTable) -> Result<(), CoreError> {
    if table.rows.is_some() {
        return Ok(());
    }
    let read = match preview(reading.asker, table.source_relation(), table.source_names()).await? {
        Ok(read) => read,
        Err(error) => return Err(failed(&error, "preview")),
    };
    if read.batches.iter().map(|b| b.num_rows()).sum::<usize>() == 0 {
        return Err(sdm_schema::infer::empty_table(&table.base));
    }
    table.rows = Some(read);
    Ok(())
}

/// Every column of `columns` added to `table` at once (`Table.add_columns`):
/// the expressions sampled in one read, then the base table's rows; a
/// column the table cannot hold is why, refusing the view.
async fn add_columns(
    reading: &Reading<'_>,
    view: &ViewChoice,
    table: &mut ViewTable,
    columns: Vec<Added>,
) -> Result<(), CoreError> {
    if columns.is_empty() {
        return Ok(());
    }
    let expressions: Vec<(String, String)> = columns
        .iter()
        .filter_map(|c| c.expr.clone().map(|e| (c.name.clone(), e)))
        .collect();
    if !expressions.is_empty() {
        let names: Vec<String> = expressions.iter().map(|(n, _)| n.clone()).collect();
        let read = match preview(
            reading.asker,
            table.expression_relation(&expressions),
            names.clone(),
        )
        .await?
        {
            Ok(read) => read,
            // A driver's error is not caught here (row 57).
            Err(error) => return Err(failed(&error, "preview")),
        };
        if read.batches.iter().map(|b| b.num_rows()).sum::<usize>() == 0 {
            return Err(sdm_schema::infer::empty_table(&table.base));
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
            table
                .sampled
                .push((name.clone(), read.schema.field(index).clone(), joined));
        }
    }
    for column in &columns {
        if table.added.iter().any(|a| a.name == column.name) {
            return Err(cannot_hold(
                view,
                &table.name,
                &format!(
                    "Column '{}' already exists in table '{}'",
                    column.name, table.name
                ),
            ));
        }
        if column.expr.is_none() {
            if table.catalog_column(&column.name).is_none() {
                return Err(cannot_hold(
                    view,
                    &table.name,
                    &format!(
                        "Column '{}' does not exist in the underlying source table",
                        column.name
                    ),
                ));
            }
            source_rows(reading, table).await?;
        }
    }
    if let Err(error) = table.infer(reading, &columns) {
        if error.sentence == Sentence::Internal {
            return Err(error);
        }
        return Err(cannot_hold(view, &table.name, &error.message));
    }
    table.added.extend(columns);
    Ok(())
}

/// Converts a semantic view (V2 to V7), its definition read (V1).
pub(super) async fn convert(
    reading: &Reading<'_>,
    view: &ViewChoice,
    described: Described,
) -> Result<Converted, CoreError> {
    let mut spec = semantic::parse(&described.text, &yaml_rules()).map_err(|r| refused(&r))?;
    let mut warnings: Vec<CoreWarning> = Vec::new();
    let mut omitted_columns: Vec<(String, String, String)> = Vec::new();
    let mut model = ViewModel::new();
    let mut tables: Vec<ViewTable> = Vec::new();
    let mut inferred: BTreeMap<String, Previewed> = BTreeMap::new();
    let mut times: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    let mut names: Vec<String> = Vec::new();
    for item in &spec.tables {
        if let sdm_schema::view::spec::TableItem::Declared { name, .. } = item {
            names.push(text(&spec, *name)?);
        }
    }
    for index in 0..spec.tables.len() {
        // V2: the table, its base table and key.
        let declared: TableSpec = semantic::table(&mut spec, index).map_err(|r| refused(&r))?;
        let name = declared.alias.clone();
        let Source::Parts {
            database,
            schema,
            table,
        } = declared.source
        else {
            return Err(internal("a semantic view's table has no base table"));
        };
        let written = Relation::of(vec![
            text(&spec, database)?,
            text(&spec, schema)?,
            text(&spec, table)?,
        ]);
        let primary_key = match declared.primary_key {
            PrimaryKey::None => None,
            PrimaryKey::One(node) => Some(text(&spec, node)?),
            PrimaryKey::Composite => {
                warnings.push(
                    CoreWarning::new(
                        Sentence::ViewCompositeKey,
                        format!(
                            "Failed to add primary key for table '{name}' since composite \
                             primary keys are not yet supported"
                        ),
                    )
                    .with_field("table", Detail::text(&name)),
                );
                None
            }
        };
        let others: Vec<String> = names.iter().filter(|n| **n != name).cloned().collect();
        let mut pending = Pending {
            name: name.clone(),
            written,
            table: None,
        };

        // V3: each column, in its section's order.
        let mut columns: Vec<Added> = Vec::new();
        let mut time_dimensions: Vec<String> = Vec::new();
        let mut cross: Vec<String> = Vec::new();
        for item in semantic::columns(&mut spec, index) {
            let sdm_schema::view::spec::ColumnItem::Column(column) =
                item.map_err(|r| refused(&r))?
            else {
                continue;
            };
            let column_name = match column.name {
                Some(node) => text(&spec, node)?,
                None => continue,
            };
            if column.section == Section::TimeDimension {
                time_dimensions.push(column_name.clone());
            }
            let Some(expr_node) = column.expr else {
                columns.push(source_column(&column_name));
                continue;
            };
            let written = text(&spec, expr_node)?;
            let expr = match reading.ask(ViewTextAsk::StripQualifier {
                table: name.clone(),
                expr: written,
            })? {
                ViewTextAnswer::Text(Ok(expr)) => expr,
                ViewTextAnswer::Text(Err(unreadable)) => {
                    let reason = unreadable.to_string();
                    warnings.push(view_words::expression_dropped(
                        sdm_contract::view::ViewKind::Semantic,
                        &column_name,
                        Some(&name),
                        &reason,
                        false,
                    ));
                    omitted_columns.push((name.clone(), column_name, reason));
                    continue;
                }
                _ => {
                    return Err(internal(
                        "the data layer read a qualifier as another answer",
                    ));
                }
            };
            if expr == column_name {
                columns.push(source_column(&column_name));
                continue;
            }
            let crosses = match reading.ask(ViewTextAsk::CrossTable {
                expr: expr.clone(),
                others: others.clone(),
            })? {
                ViewTextAnswer::Flag(Ok(crosses)) => crosses,
                ViewTextAnswer::Flag(Err(_)) => true,
                _ => {
                    return Err(internal(
                        "the data layer read a cross-table test as another answer",
                    ));
                }
            };
            if crosses {
                cross.push(column_name);
                continue;
            }
            let base_columns = pending.read(reading).await?.source_names();
            let verdict = match reading.ask(ViewTextAsk::Expression {
                expr: expr.clone(),
                trust_expressions: view.trust_expressions,
                columns: base_columns,
            })? {
                ViewTextAnswer::Expression(verdict) => verdict,
                _ => {
                    return Err(internal(
                        "the data layer read an expression as another answer",
                    ));
                }
            };
            if let Some(verdict) = verdict {
                let words = match reading.ask(ViewTextAsk::ReasonWords {
                    reason: verdict.reason.clone(),
                })? {
                    ViewTextAnswer::Words(words) => words,
                    _ => return Err(internal("the data layer worded a reason as another answer")),
                };
                let reason = view_words::expr_reason(&verdict.reason, words);
                warnings.push(view_words::expression_dropped(
                    sdm_contract::view::ViewKind::Semantic,
                    &column_name,
                    Some(&name),
                    &reason,
                    verdict.untrusted,
                ));
                omitted_columns.push((name.clone(), column_name, reason));
                continue;
            }
            let declared_type = column.data_type.map(|node| text(&spec, node)).transpose()?;
            let declared = match &declared_type {
                None => None,
                Some(data_type) => match reading.ask(ViewTextAsk::DeclaredType {
                    data_type: data_type.clone(),
                })? {
                    ViewTextAnswer::DeclaredType(DeclaredType::Known(dtype)) => Some(dtype),
                    ViewTextAnswer::DeclaredType(DeclaredType::Unsupported) => {
                        Some(sdm_contract::ops::Dtype::Unsupported)
                    }
                    ViewTextAnswer::DeclaredType(DeclaredType::Unknown) => None,
                    _ => return Err(internal("the data layer read a type as another answer")),
                },
            };
            columns.push(Added {
                name: column_name,
                column: None,
                expr: Some(expr),
                declared,
                declared_type,
            });
        }
        match cross.as_slice() {
            [] => {}
            [one] => {
                warnings.push(
                    CoreWarning::new(
                        Sentence::ViewColumnCrossTable,
                        format!(
                            "Failed to add column '{one}' of table '{name}' since its expression \
                             references other tables"
                        ),
                    )
                    .with_field("column", Detail::text(one))
                    .with_field("table", Detail::text(&name)),
                );
                omitted_columns.push((
                    name.clone(),
                    one.clone(),
                    "its expression references other tables".into(),
                ));
            }
            several => {
                let listed = format!(
                    "[{}]",
                    several
                        .iter()
                        .map(|c| view_words::quote(c))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                warnings.push(
                    CoreWarning::new(
                        Sentence::ViewColumnsCrossTable,
                        format!(
                            "Failed to add columns '{listed}' of table '{name}' since their \
                             expressions reference other tables"
                        ),
                    )
                    .with_field(
                        "columns",
                        Detail::List(several.iter().cloned().map(Scalar::Utf8).collect()),
                    )
                    .with_field("table", Detail::text(&name)),
                );
                for column in several {
                    omitted_columns.push((
                        name.clone(),
                        column.clone(),
                        "its expression references other tables".into(),
                    ));
                }
            }
        }

        // V4: every column added at once, then the key.
        if !columns.is_empty() {
            let table = pending.read(reading).await?;
            add_columns(reading, view, table, columns).await?;
        }
        if let Some(key) = &primary_key {
            let table = pending.read(reading).await?;
            if !table.added.iter().any(|a| a.name == *key) {
                add_columns(reading, view, table, vec![source_column(key)]).await?;
            }
        }
        let Some(mut table) = pending.table.take() else {
            // A table of no column and no key reads nothing.
            model
                .add_table(ModelTable {
                    name: name.clone(),
                    columns: Vec::new(),
                    primary_key: None,
                })
                .map_err(|e| super::invalid(view, "semantic view", &e))?;
            tables.push(ViewTable::new(
                name.clone(),
                pending.written.clone(),
                Catalog {
                    columns: Vec::new(),
                },
            ));
            continue;
        };
        let previewed = table.infer(reading, &[])?;
        let mut columns_model: Vec<ModelColumn> = model_columns(&table, &previewed);
        if let Some(key) = &primary_key {
            let Some(column) = columns_model.iter_mut().find(|c| c.name == *key) else {
                return Err(internal("a semantic view's key was not added"));
            };
            if !matches!(
                column.dtype,
                sdm_contract::ops::Dtype::Int
                    | sdm_contract::ops::Dtype::Float
                    | sdm_contract::ops::Dtype::String
            ) {
                return Err(cannot_hold(
                    view,
                    &name,
                    &format!(
                        "Primary key '{key}' in table '{name}' must be of type int, float or \
                         string"
                    ),
                ));
            }
            column.stype = sdm_contract::ops::Stype::Id;
        }

        // V5: the time column, among the time dimensions first.
        let timestamps = |only: Option<&[String]>| -> Vec<String> {
            columns_model
                .iter()
                .filter(|c| c.stype == sdm_contract::ops::Stype::Timestamp)
                .filter(|c| only.is_none_or(|names| names.contains(&c.name)))
                .map(|c| c.name.clone())
                .collect()
        };
        let mut candidates = timestamps(Some(&time_dimensions));
        if candidates.is_empty() {
            candidates = timestamps(None);
        }
        let mut choice = (None, None);
        if !candidates.is_empty() {
            let taken = super::choose_among(reading, &mut table, &previewed, &candidates).await?;
            choice = (taken.column, taken.fallback);
        }
        model
            .add_table(ModelTable {
                name: name.clone(),
                columns: columns_model,
                primary_key: primary_key.clone(),
            })
            .map_err(|e| super::invalid(view, "semantic view", &e))?;
        inferred.insert(name.clone(), previewed);
        times.insert(name.clone(), choice);
        tables.push(table);
    }

    // V6: the relationships, each of one column to the right table's key.
    for item in semantic::relationships(&mut spec) {
        let item = item.map_err(|r| refused(&r))?;
        let relationship = text(&spec, item.name)?;
        let refuse = |reason: String| {
            CoreWarning::new(
                Sentence::ViewRelationshipRefused,
                format!("Failed to add relationship '{relationship}' since {reason}"),
            )
            .with_field("relationship", Detail::text(&relationship))
            .with_field("reason", Detail::text(reason))
        };
        if item.pairs != 1 {
            warnings.push(refuse(
                "composite key references are not yet supported".to_owned(),
            ));
            continue;
        }
        let read = spec.relationship(&item).map_err(|r| refused(&r))?;
        let left_table = text(&spec, read.left_table)?;
        let left_key = text(&spec, read.left_column)?;
        let right_table = text(&spec, read.right_table)?;
        let right_key = text(&spec, read.right_column)?;
        let mut step = RelationshipStep {
            left_table: left_table.clone(),
            left_key: left_key.clone(),
            right_table,
            right_key,
            left_key_column: None,
        };
        let mut result = model.add_relationship(step.clone());
        if matches!(
            result,
            Err(RelationshipRefusal::NotLinked(
                LinkFailure::NoSourceColumn { .. }
            ))
        ) {
            step.left_key_column = match tables.iter_mut().find(|t| t.name == left_table) {
                Some(table) if table.catalog_column(&left_key).is_some() => {
                    source_rows(reading, table).await?;
                    table
                        .infer(reading, &[source_column(&left_key)])
                        .ok()
                        .and_then(|p| {
                            p.table().column(&left_key).map(|c| ModelColumn {
                                name: left_key.clone(),
                                dtype: c.dtype,
                                stype: c.stype,
                                expr: None,
                            })
                        })
                }
                _ => None,
            };
            result = model.add_relationship(step);
        }
        if let Err(refusal) = result {
            warnings.push(refuse(view_words::relationship_reason_text(&refusal)));
        }
    }

    // V7: the model's checks.
    for logical in &model.tables {
        if let Some(table) = tables.iter_mut().find(|t| t.name == logical.name) {
            super::sync(table, logical);
        }
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
        .map_err(|refusal| super::invalid(view, "semantic view", &refusal))?;
    for table in &mut tables {
        if let Ok(previewed) = table.infer(reading, &[]) {
            inferred.insert(table.name.clone(), previewed);
        }
    }
    let (description, relations) = super::describe(&model, &tables, &inferred, &times)?;
    Ok(Converted {
        description,
        relations,
        time_choices,
        notes: Vec::new(),
        warnings,
        omitted_tables: Vec::new(),
        omitted_columns,
    })
}

/// A column of the base table under its own name.
fn source_column(name: &str) -> Added {
    Added {
        name: name.to_owned(),
        column: Some(name.to_owned()),
        expr: None,
        declared: None,
        declared_type: None,
    }
}
