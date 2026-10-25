// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A semantic model's conversion (T3.5, design S5): one routine in the old
//! client's read order, a Databricks metric view's M1 to M11
//! ([`metric`]) and a Snowflake semantic view's V1 to V7 ([`semantic`]), at
//! every `describe_source` and `materialize` (the owner's plan B: a view is
//! read as it stands at each run, nothing kept between runs).
//!
//! Each step reads through the run's asker and sees its answer before the
//! next, as the old client did; every question of the view's text is the
//! data layer's (`DataLayer::view_text`), and every failure is read by the
//! fact that fits what was asked (`DataLayer::failure`, design S3). The
//! logical model is `sdm-schema`'s ([`sdm_schema::view::model`]); each
//! logical table reads a relation the data layer renders, which lives in
//! the run alone ([`Converted::relations`]).
//!
//! At `describe_source` the conversion answers the schema it gives, its
//! notes and what it left out (one warning of the group "read in part");
//! at `materialize` it is read again, and a table or column the schema
//! holds that the view no longer gives is refused by name, with this run's
//! reason ([`located`]).

pub mod definition;
mod metric;
mod semantic;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field, Schema};
use sdm_contract::catalog::{CatalogAnswer, CatalogColumn};
use sdm_contract::data::{
    DataRequest, DescribePart, ProviderError, Relation, RelationColumn, ViewChoice,
};
use sdm_contract::ops::{
    ColumnRef, DescribedColumn, DescribedSource, DescribedTable, Dtype, HeldSchema,
    OperationResult, SourceDescription, SourceSpec, Stype, ViewNote,
};
use sdm_contract::view::{ViewKind, ViewTextAnswer, ViewTextAsk};
use sdm_contract::{ColumnType, CoreError, CoreWarning, Detail, Scalar, Sentence};
use sdm_engine::time::text::Reader;
use sdm_schema::infer::{self, Floats, PREVIEW_ROWS, Previewed, WholeTable};
use sdm_schema::view::model::{
    self, Choice, ModelColumn, ModelRefusal, ModelTable, TimeColumnNote, TimeInput, ViewModel,
};

use crate::compose::DataAccess;
use crate::describe::attempt;
use crate::discover::Catalog;
use crate::run::{ArrowData, Asker, Attachment, internal};

/// What converting a view gave.
pub(crate) struct Converted {
    /// The schema the view converts into, each table read as the relation
    /// of one part named after it, as a host names it.
    pub description: SourceDescription,
    /// Each logical table's relation, as this run reads it.
    pub relations: Vec<(String, Relation)>,
    /// Time choices before schema selection and caller relationship overrides.
    pub time_choices: BTreeMap<String, infer::TimeChoice>,
    /// The notes of the conversion (`Schema.conversion_notes`).
    pub notes: Vec<ViewNote>,
    /// What the view loses, in order, each a warning of the group "read in
    /// part" (`CL/materializers/sources.py:285-294`).
    pub warnings: Vec<CoreWarning>,
    /// Each logical table the view names that the conversion left out, with
    /// why, as a clause.
    pub omitted_tables: Vec<(String, String)>,
    /// Each column the view names that the conversion left out, by table,
    /// with why, as a clause.
    pub omitted_columns: Vec<(String, String, String)>,
}

/// The data access of a run converting a view, and its choice.
fn access(asker: &Asker) -> Result<(&DataAccess, ViewChoice), CoreError> {
    let data = asker
        .data()
        .ok_or_else(|| internal("a semantic model's conversion names no provider"))?;
    let view = data
        .view()
        .ok_or_else(|| internal("a semantic model's conversion names no view"))?;
    Ok((data, view))
}

/// The data layer's answer to a question of the view's text.
fn ask_text(data: &DataAccess, question: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
    data.view_text(question)
}

/// Converts the view the run's provider names, in its kind's order.
pub(crate) async fn convert(asker: &Asker, reader: &Reader) -> Result<Converted, CoreError> {
    let (data, view) = access(asker)?;
    let floats = if asker.reads_nan_as_missing() {
        Floats::NanMissing
    } else {
        Floats::NanValue
    };
    let reading = Reading {
        asker,
        data,
        reader,
        floats,
    };
    let definition = read_definition(&reading, &view).await?;
    let converted = match definition.kind {
        ViewKind::Metric => metric::convert(&reading, &view, definition).await,
        ViewKind::Semantic => semantic::convert(&reading, &view, definition).await,
    };
    // A refusal that could not name the view where it was found names it.
    converted.map_err(|error| {
        if error.sentence == Sentence::ViewInvalid && !error.fields.contains_key("view") {
            let reason = match error.fields.get("reason") {
                Some(Detail::Scalar(Scalar::Utf8(reason))) => reason.clone(),
                _ => String::new(),
            };
            let kind = match error.fields.get("view_kind") {
                Some(Detail::Scalar(Scalar::Utf8(kind))) => kind.clone(),
                _ => String::new(),
            };
            return CoreError {
                message: format!(
                    "Cannot convert {kind} '{}' into a graph since {reason}",
                    view.name
                ),
                ..error
            }
            .with_field("view", Detail::text(&view.name));
        }
        error
    })
}

/// A view that cannot be converted, with why; the routine names the view
/// ([`convert`]).
pub(crate) fn unconvertible(kind: &str, reason: String) -> CoreError {
    CoreError::new(
        Sentence::ViewInvalid,
        format!("Cannot convert {kind} into a graph since {reason}"),
    )
    .with_field("view_kind", Detail::text(kind))
    .with_field("reason", Detail::text(reason))
}

/// What every step of a conversion reads through.
pub(crate) struct Reading<'a> {
    pub asker: &'a Asker,
    pub data: &'a DataAccess,
    pub reader: &'a Reader,
    pub floats: Floats,
}

impl Reading<'_> {
    /// The data layer's answer to a question of the view's text.
    pub fn ask(&self, question: ViewTextAsk) -> Result<ViewTextAnswer, CoreError> {
        ask_text(self.data, question)
    }
}

/// The view's definition as the warehouse describes it.
pub(crate) struct Described {
    pub kind: ViewKind,
    pub text: String,
    pub parameters: Vec<(String, String)>,
    pub located: Vec<Option<String>>,
    pub columns: Vec<(String, String)>,
}

/// M1 and V1: the view's definition, read once at this run.
async fn read_definition(reading: &Reading<'_>, view: &ViewChoice) -> Result<Described, CoreError> {
    let parts = match reading.ask(ViewTextAsk::ViewName {
        name: view.name.clone(),
    })? {
        ViewTextAnswer::Names(Some(parts)) => parts,
        ViewTextAnswer::Names(None) => {
            return Err(CoreError::new(
                Sentence::ViewNameInvalid,
                format!("Invalid metric view name '{}'", view.name),
            )
            .with_field("name", Detail::text(&view.name)));
        }
        _ => {
            return Err(internal(
                "the data layer read a view's name as another answer",
            ));
        }
    };
    let relation = Relation::of(parts);
    // A metric definition describes a relation under its source namespace.
    // A semantic definition instead binds opaque lookup text: its data
    // layer qualifies that argument, leaving missing parts to the warehouse.
    let relation = match reading.data.view_kind() {
        Some(ViewKind::Semantic) => relation,
        _ => reading.asker.read_as(&relation),
    };
    let request = DataRequest::Describe {
        tables: vec![relation.clone()],
        part: DescribePart::ViewDefinition,
    };
    let answer = match attempt(reading.asker, request.clone()).await? {
        Ok(data) => reading.asker.read(&request, &data)?,
        Err(failed) => {
            if reading.data.view_kind() == Some(ViewKind::Semantic) {
                let name = relation.dotted();
                return Err(CoreError::new(
                    Sentence::ViewSemanticNotFound,
                    format!("Semantic view '{name}' does not exist or you may not read it"),
                )
                .with_field("view", Detail::text(name)));
            }
            let facts = reading.data.failure(&failed);
            if facts.unusable == Some(sdm_contract::data::Unusable::Missing) || facts.not_found {
                return Err(view_not_found(&relation));
            }
            return Err(
                CoreError::provider(&failed).with_field("request", Detail::text("describe"))
            );
        }
    };
    let CatalogAnswer::ViewDefinition(definition) = answer else {
        return Err(crate::describe::another_part());
    };
    Ok(Described {
        kind: definition.kind,
        text: definition.text,
        parameters: definition.parameters,
        located: definition.located,
        columns: definition.columns,
    })
}

/// A view the warehouse does not hold, or the caller may not see
/// (`DB/metric_view.py:1440-1450`), named as the old client quoted it.
fn view_not_found(relation: &Relation) -> CoreError {
    let view = quoted(relation);
    CoreError::new(
        Sentence::ViewNotFound,
        format!("Metric view '{view}' does not exist or you may not see it"),
    )
    .with_field("view", Detail::text(view))
}

/// A metric view's name as the old client quoted it in its messages
/// (`RE/graph.py:1512-1518`): each part in backticks, a backtick doubled.
/// Only a message names it so; every statement's name is the data layer's.
fn quoted(relation: &Relation) -> String {
    relation
        .source
        .iter()
        .map(|part| format!("`{}`", part.replace('`', "``")))
        .collect::<Vec<_>>()
        .join(".")
}

/// A column a logical table holds, as the conversion added it.
#[derive(Clone, Debug)]
pub(crate) struct Added {
    pub name: String,
    /// The source column it reads, as the source spells it; none for an
    /// expression.
    pub column: Option<String>,
    pub expr: Option<String>,
    /// The type the view declares for it, if any.
    pub declared: Option<Dtype>,
    /// The original declaration retained for temporal and marked source facts.
    pub declared_type: Option<String>,
}

/// A logical table as the conversion builds it: the base table it reads,
/// its catalog, the names the view gives its columns, the view's filter on
/// it, the rows read of it, and the columns added.
pub(crate) struct ViewTable {
    pub name: String,
    pub base: Relation,
    pub catalog: Catalog,
    /// Each source column the view names otherwise, as the source spells it
    /// and as the view does (`_name_columns`).
    pub renames: Vec<(String, String)>,
    pub filter: Option<String>,
    /// The source's rows, every column under the view's name for it.
    pub rows: Option<ArrowData>,
    /// Each expression column's sampled values, by name.
    pub sampled: Vec<(String, Field, ArrayRef)>,
    pub added: Vec<Added>,
}

impl ViewTable {
    pub fn new(name: String, base: Relation, catalog: Catalog) -> ViewTable {
        ViewTable {
            name,
            base,
            catalog,
            renames: Vec::new(),
            filter: None,
            rows: None,
            sampled: Vec::new(),
            added: Vec::new(),
        }
    }

    /// A copy of the table for reading its columns with others added.
    pub fn clone_for_reading(&self) -> ViewTable {
        ViewTable {
            name: self.name.clone(),
            base: self.base.clone(),
            catalog: self.catalog.clone(),
            renames: self.renames.clone(),
            filter: self.filter.clone(),
            rows: self.rows.clone(),
            sampled: self.sampled.clone(),
            added: self.added.clone(),
        }
    }

    /// The view's name for the source column `column`.
    pub fn view_name(&self, column: &str) -> String {
        self.renames
            .iter()
            .find(|(from, _)| from == column)
            .map_or_else(|| column.to_owned(), |(_, to)| to.clone())
    }

    /// The source's columns, each under the view's name for it, in the
    /// catalog's order.
    pub fn source_names(&self) -> Vec<String> {
        self.catalog
            .columns
            .iter()
            .map(|c| self.view_name(&c.name))
            .collect()
    }

    /// The source column the view names `name`, as the source spells it.
    pub fn source_of(&self, name: &str) -> Option<String> {
        self.catalog
            .columns
            .iter()
            .find(|c| self.view_name(&c.name) == name)
            .map(|c| c.name.clone())
    }

    /// The catalog's column the view names `name`.
    pub fn catalog_column(&self, name: &str) -> Option<&CatalogColumn> {
        let source = self.source_of(name)?;
        self.catalog.columns.iter().find(|c| c.name == source)
    }

    /// The relation every read of the source's rows names
    /// (`_quoted_source_name`, `DB/table.py:170-195`): the base table, or,
    /// where the view renames a column or filters the table, its rows as
    /// the view reads them.
    pub fn source_relation(&self) -> Relation {
        let mut relation = self.base.clone();
        if !self.renames.is_empty() {
            relation.columns = self
                .catalog
                .columns
                .iter()
                .map(|c| RelationColumn {
                    name: self.view_name(&c.name),
                    column: Some(c.name.clone()),
                    expression: None,
                })
                .collect();
        }
        relation.filter.clone_from(&self.filter);
        relation
    }

    /// The relation of `expressions` read over the source's rows as the view
    /// reads them: each expression evaluated over the base table, under the
    /// view's filter.
    pub fn expression_relation(&self, expressions: &[(String, String)]) -> Relation {
        let mut relation = self.base.clone();
        relation.columns = expressions
            .iter()
            .map(|(name, expr)| RelationColumn {
                name: name.clone(),
                column: None,
                expression: Some(expr.clone()),
            })
            .collect();
        relation.filter.clone_from(&self.filter);
        relation
    }

    /// The relation a time column's choice reads `asked` over every row of
    /// (`_full_range`): the source's rows as the view reads them where each
    /// is a source column under its view name, else those rows with each
    /// other column asked, an expression or a source column under another
    /// name, `extra` the columns being added.
    pub fn ranges_relation(&self, asked: &[String], extra: &[Added]) -> Relation {
        let sources = self.source_names();
        let others: Vec<&Added> = self
            .added
            .iter()
            .chain(extra)
            .filter(|a| asked.contains(&a.name) && !sources.contains(&a.name))
            .collect();
        let mut relation = self.source_relation();
        if others.is_empty() {
            return relation;
        }
        if relation.columns.is_empty() {
            relation.columns = self
                .catalog
                .columns
                .iter()
                .map(|c| RelationColumn {
                    name: c.name.clone(),
                    column: Some(c.name.clone()),
                    expression: None,
                })
                .collect();
        }
        relation
            .columns
            .extend(others.into_iter().map(|a| RelationColumn {
                name: a.name.clone(),
                column: a.column.clone(),
                expression: a.expr.clone(),
            }));
        relation
    }

    /// The field sampled or read for the column `name`.
    pub fn field(&self, name: &str) -> Option<&Field> {
        if let Some((_, field, _)) = self.sampled.iter().find(|(n, _, _)| n == name) {
            return Some(field);
        }
        let rows = self.rows.as_ref()?;
        Some(rows.schema.field(rows.schema.index_of(name).ok()?))
    }

    /// The values read of a column, from its sample or the source rows.
    pub fn values(&self, name: &str) -> Option<(Field, ArrayRef)> {
        if let Some((_, field, array)) = self.sampled.iter().find(|(n, _, _)| n == name) {
            return Some((field.clone(), Arc::clone(array)));
        }
        let rows = self.rows.as_ref()?;
        let index = rows.schema.index_of(name).ok()?;
        let arrays: Vec<ArrayRef> = rows
            .batches
            .iter()
            .map(|b| Arc::clone(b.column(index)))
            .collect();
        let field = self.field(name)?.clone();
        let joined =
            arrow_select::concat::concat(&arrays.iter().map(|a| a.as_ref()).collect::<Vec<_>>())
                .ok()?;
        Some((field, joined))
    }

    /// The table's columns, `extra` added, inferred from the rows read of
    /// them, as the old client's `add_columns` infers each: the type the
    /// catalog or the view declares, else the values'; the meaning by name
    /// and values. Each column's values are its own read, cut to the
    /// fewest rows any holds (`_update_expr_sample_df`).
    pub fn infer(&self, reading: &Reading<'_>, extra: &[Added]) -> Result<Previewed, CoreError> {
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        let mut columns = Vec::new();
        for added in self.added.iter().chain(extra) {
            let Some((field, array)) = self.values(&added.name) else {
                return Err(internal("a view's column was added with no values read"));
            };
            let declared = self
                .catalog_column(&added.name)
                .filter(|_| added.expr.is_none());
            columns.push(CatalogColumn {
                name: added.name.clone(),
                declared_type: declared
                    .map(|c| c.declared_type.clone())
                    .unwrap_or_default(),
                not_null: declared.is_some_and(|c| c.not_null),
                primary_key: false,
                key_nullable: false,
                unique: false,
                dtype: added.declared.or_else(|| declared.and_then(|c| c.dtype)),
            });
            fields.push(Field::new(
                added.name.clone(),
                field.data_type().clone(),
                true,
            ));
            arrays.push(array);
        }
        // A table with no column yet holds the rows its source does.
        let held = self
            .rows
            .as_ref()
            .map_or(0, |r| r.batches.iter().map(RecordBatch::num_rows).sum());
        let rows = arrays.iter().map(|a| a.len()).min().unwrap_or(held);
        let arrays: Vec<ArrayRef> = arrays.iter().map(|a| a.slice(0, rows)).collect();
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            arrays,
            &arrow_array::RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .map_err(|_| internal("a view's rows do not make one batch"))?;
        infer::view_table(
            &self.name,
            &Relation::named(self.name.clone()),
            &columns,
            &schema,
            &[batch],
            reading.reader,
            reading.floats,
        )
    }
}

/// The model's columns of `table` from its inferred columns.
pub(crate) fn model_columns(table: &ViewTable, previewed: &Previewed) -> Vec<ModelColumn> {
    previewed
        .table()
        .columns
        .iter()
        .map(|c| ModelColumn {
            name: c.name.clone(),
            expr: table
                .added
                .iter()
                .find(|a| a.name == c.name)
                .and_then(|a| a.expr.clone()),
            dtype: c.dtype,
            stype: c.stype,
        })
        .collect()
}

/// The relations, by logical table, and the schema the model gives, its
/// tables named after the logical tables (design S6).
pub(crate) fn describe(
    model: &ViewModel,
    tables: &[ViewTable],
    inferred: &BTreeMap<String, Previewed>,
    times: &BTreeMap<String, (Option<String>, Option<String>)>,
) -> Result<(SourceDescription, Vec<(String, Relation)>), CoreError> {
    let mut described = Vec::with_capacity(model.tables.len());
    let mut relations = Vec::with_capacity(model.tables.len());
    for logical in &model.tables {
        let table = tables
            .iter()
            .find(|t| t.name == logical.name)
            .ok_or_else(|| internal("a view's model holds a table the conversion did not read"))?;
        let previewed = inferred.get(&logical.name);
        let mut columns = Vec::with_capacity(logical.columns.len());
        let mut relation = table.base.clone();
        relation.filter.clone_from(&table.filter);
        for column in &logical.columns {
            let read = previewed.and_then(|p| p.table().column(&column.name));
            let (received, nullable, before) = match read {
                Some(read) => (read.received.clone(), read.nullable, read.stype),
                None => (ColumnType::Utf8, true, column.stype),
            };
            let added = table.added.iter().find(|a| a.name == column.name);
            relation.columns.push(RelationColumn {
                name: column.name.clone(),
                column: match added {
                    Some(added) => added.column.clone(),
                    None => table.source_of(&column.name),
                }
                .filter(|_| column.expr.is_none()),
                expression: column.expr.clone(),
            });
            columns.push(DescribedColumn {
                name: column.name.clone(),
                received,
                dtype: column.dtype,
                stype: column.stype,
                stype_before_key: (column.stype == Stype::Id && before != Stype::Id)
                    .then_some(before),
                nullable,
                unique: false,
                declared_type: table.field(&column.name).and_then(|field| {
                    let declared =
                        added.and_then(|a| a.declared_type.as_deref()).or_else(|| {
                            table
                                .catalog_column(&column.name)
                                .map(|c| c.declared_type.as_str())
                        })?;
                    crate::discover::retained_source_type(field, declared).map(str::to_owned)
                }),
                exact_number: None,
            });
        }
        let (time_column, inferred_time) = times.get(&logical.name).cloned().unwrap_or_default();
        described.push(DescribedTable {
            name: logical.name.clone(),
            source: Relation::named(logical.name.clone()),
            columns,
            frame_key_form: None,
            primary_key: logical.primary_key.iter().cloned().collect(),
            time_column,
            time_column_inferred: inferred_time,
            duplicate_keys: None,
        });
        relations.push((logical.name.clone(), relation));
    }
    let links = model
        .edges
        .iter()
        .filter_map(|edge| {
            let target = model.table(&edge.target)?.primary_key.clone()?;
            Some(vec![
                ColumnRef {
                    table: edge.source.clone(),
                    column: edge.key.clone(),
                },
                ColumnRef {
                    table: edge.target.clone(),
                    column: target,
                },
            ])
        })
        .collect();
    Ok((
        SourceDescription {
            tables: described,
            links,
        },
        relations,
    ))
}

/// The provider's failure, as the run's error naming what it asked.
pub(crate) fn failed(error: &ProviderError, request: &str) -> CoreError {
    CoreError::provider(error).with_field("request", Detail::text(request))
}

/// The caller-visible operation a failed conversion read was performing.
#[derive(Clone, Copy)]
pub(crate) enum ReadTask {
    Expression,
    JoinKeys,
}

/// A finite diagnosis from known codes, with the operation the core requested.
/// Driver messages and unrecognized free-text fields remain internal (H17,
/// KO2-86). Table and column context is added by the existing message site.
pub(crate) fn failure_text(error: &ProviderError, task: ReadTask) -> String {
    let class = error
        .error_class
        .as_deref()
        .and_then(|c| c.split('.').next());
    let state = error.sqlstate.as_deref();
    let reason = match (class, state) {
        (Some("CAST_INVALID_INPUT" | "CANNOT_PARSE_TIMESTAMP"), _)
        | (_, Some("22018" | "22007")) => Some("a value could not be converted"),
        (Some("DIVIDE_BY_ZERO" | "INTERVAL_DIVIDED_BY_ZERO"), _) | (_, Some("22012")) => {
            Some("division by zero occurred")
        }
        (Some("CAST_OVERFLOW" | "NUMERIC_VALUE_OUT_OF_RANGE" | "ARITHMETIC_OVERFLOW"), _)
        | (_, Some("22003" | "22008")) => Some("a value was outside the supported range"),
        (Some("UNRESOLVED_COLUMN" | "COLUMN_NOT_FOUND" | "UNRESOLVED_FIELD"), _)
        | (_, Some("42703" | "42S22")) => Some("a referenced column could not be resolved"),
        (Some("TABLE_OR_VIEW_NOT_FOUND"), _) | (_, Some("42P01" | "42S02")) => {
            Some("a referenced table or view could not be resolved")
        }
        (Some("INSUFFICIENT_PERMISSIONS" | "PERMISSION_DENIED"), _) | (_, Some("42501")) => {
            Some("permission was denied")
        }
        (Some("PARSE_SYNTAX_ERROR"), _) | (_, Some("42601")) => {
            Some("the expression's syntax was not accepted")
        }
        (Some("DATATYPE_MISMATCH"), _) | (_, Some("42804")) => {
            Some("the expression used incompatible types")
        }
        _ => None,
    };
    let (during, action) = match task {
        ReadTask::Expression => ("evaluating the expression", "evaluate the expression"),
        ReadTask::JoinKeys => ("counting the join keys", "count the join keys"),
    };
    let mut explanation = match reason {
        Some(reason) => format!("{reason} while {during}"),
        None => format!("the warehouse could not {action}"),
    };
    // SQLSTATE is a five-character code. Vendor codes here are numeric;
    // neither free-form field may become a second route for driver text.
    let mut codes = Vec::new();
    if let Some(state) = state.filter(|s| {
        s.len() == 5
            && s.bytes()
                .all(|b| b.is_ascii_digit() || b.is_ascii_uppercase())
    }) {
        codes.push(format!("SQLSTATE: {state}"));
    }
    if let Some(code) = error
        .vendor_code
        .as_deref()
        .filter(|s| !s.is_empty() && s.len() <= 20 && s.bytes().all(|b| b.is_ascii_digit()))
    {
        codes.push(format!("vendor code: {code}"));
    }
    if !codes.is_empty() {
        explanation.push_str(&format!(" ({})", codes.join(", ")));
    }
    explanation
}

/// `describe_source` of a semantic model (T3.5): the view converted, the
/// tables the caller names kept (all where none), its notes and what it
/// left out given here alone.
pub(crate) async fn describe_source(
    asker: Asker,
    spec: SourceSpec,
    reader: &Reader,
) -> Result<OperationResult, CoreError> {
    // Validate schema names through the ordinary source boundary before reads.
    let named = crate::describe::named_tables(&spec)?;
    let converted = convert(&asker, reader).await?;
    let mut description = converted.description;
    let chosen = if named.is_empty() {
        description
            .tables
            .iter()
            .map(|table| sdm_contract::ops::NamedTable {
                name: table.name.clone(),
                source: table.source.clone(),
                columns: None,
            })
            .collect()
    } else {
        named
    };
    let mut absent: Vec<_> = chosen
        .iter()
        .filter(|named| !description.tables.iter().any(|t| t.source == named.source))
        .map(|named| named.source.dotted())
        .collect();
    if !absent.is_empty() {
        absent.sort();
        absent.dedup();
        let mut offered: Vec<_> = description.tables.iter().map(|t| t.name.clone()).collect();
        offered.sort();
        return Err(CoreError::new(
            Sentence::ViewTablesAbsent,
            "tables named are not tables of the semantic model",
        )
        .with_field(
            "absent",
            Detail::List(absent.into_iter().map(Scalar::Utf8).collect()),
        )
        .with_field(
            "offered",
            Detail::List(offered.into_iter().map(Scalar::Utf8).collect()),
        ));
    }
    // The last alias of each model table names its links and notes, exactly
    // as the reference's as_named map; table copies retain caller order.
    let aliases: BTreeMap<_, _> = chosen
        .iter()
        .map(|named| (named.source.dotted(), named.name.clone()))
        .collect();
    let tables = chosen
        .iter()
        .map(|named| {
            let mut table = description
                .tables
                .iter()
                .find(|t| t.source == named.source)
                .expect("the logical source was found")
                .clone();
            table.name.clone_from(&named.name);
            table
        })
        .collect();
    description.tables = tables;
    description.links = description
        .links
        .into_iter()
        .filter_map(|link| {
            link.into_iter()
                .map(|end| {
                    Some(ColumnRef {
                        table: aliases.get(&end.table)?.clone(),
                        column: end.column,
                    })
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect();
    let choices = chosen
        .iter()
        .map(|named| {
            converted
                .time_choices
                .get(&named.source.dotted())
                .cloned()
                .unwrap_or_default()
        })
        .collect();
    let assembled = sdm_schema::assemble::assemble_converted(
        description,
        choices,
        &spec.primary_keys,
        &spec.time_columns,
        spec.relationships.as_deref(),
    )?;
    description = assembled.description;
    let notes = converted
        .notes
        .into_iter()
        .filter_map(|mut note| {
            if let Some(table) = &note.table {
                let alias = aliases.get(table)?;
                note.table = Some(alias.clone());
                let shown = if note.sentence == Sentence::ViewMeasureNote {
                    crate::view_words::pql_name(alias)
                } else {
                    alias.clone()
                };
                note.fields.insert("table".to_owned(), Detail::text(shown));
            }
            Some(note)
        })
        .collect();
    for warning in assembled.warnings {
        asker.warn(warning);
    }
    for warning in converted.warnings {
        asker.warn(warning);
    }
    // A note of a table not taken, or of a time column the table no longer
    // has, is the host's to drop (`CL/schema.py:1401-1433`).
    let bytes = serde_json::to_vec(&description)
        .map_err(|_| internal("a source description does not serialize"))?;
    let id = asker.attach(Attachment::Bytes(bytes))?;
    Ok(OperationResult::DescribeSource(DescribedSource {
        frame_rows: None,
        description: id,
        notes,
    }))
}

/// `materialize` over a semantic model (plan B): the view read again, as it
/// stands at this run, and `held` read by it: each table the schema holds
/// read as the relation the view gives it now, each column by the type and
/// meaning the view gives it now, the caller's meanings kept. A table or
/// column the schema holds that the view no longer gives is refused by
/// name, with this run's reason; nothing of what the view leaves out is
/// said here.
pub(crate) async fn located(
    asker: &Asker,
    held: &HeldSchema,
    reader: &Reader,
) -> Result<SourceDescription, CoreError> {
    let converted = convert(asker, reader).await?;
    let mut located = held.clone();
    let mut found = BTreeMap::new();
    for table in &held.tables {
        let Some(fresh) = converted
            .description
            .tables
            .iter()
            .find(|t| t.source == table.source)
        else {
            let reason = converted
                .omitted_tables
                .iter()
                .find(|(name, _)| Some(name.as_str()) == table.source.single())
                .map_or_else(
                    || "the view no longer names it".to_owned(),
                    |(_, why)| why.clone(),
                );
            return Err(CoreError::new(
                Sentence::ViewTableOmitted,
                format!(
                    "the semantic model no longer gives the table '{}' this schema holds: {reason}",
                    table.name
                ),
            )
            .with_field("table", Detail::text(&table.name))
            .with_field("reason", Detail::text(reason)));
        };
        let relation = converted
            .relations
            .iter()
            .find(|(name, _)| Some(name.as_str()) == table.source.single())
            .map(|(_, r)| r.clone())
            .ok_or_else(|| internal("a view's table has no relation"))?;
        for column in &table.columns {
            let Some(now) = fresh.columns.iter().find(|c| c.name == column.name) else {
                let reason = converted
                    .omitted_columns
                    .iter()
                    .find(|(t, c, _)| {
                        Some(t.as_str()) == table.source.single() && *c == column.name
                    })
                    .map_or_else(
                        || "the view no longer names it".to_owned(),
                        |(_, _, why)| why.clone(),
                    );
                return Err(CoreError::new(
                    Sentence::ViewColumnOmitted,
                    format!(
                        "the semantic model no longer gives the column '{}' of table '{}' this \
                         schema holds: {reason}",
                        column.name, table.name
                    ),
                )
                .with_field("table", Detail::text(&table.name))
                .with_field("column", Detail::text(&column.name))
                .with_field("reason", Detail::text(reason)));
            };
            found.insert(
                (table.name.clone(), column.name.clone()),
                sdm_schema::held::facts_of(&relation, now),
            );
        }
        let current = located
            .tables
            .iter_mut()
            .find(|t| t.name == table.name)
            .ok_or_else(|| internal("a held logical table disappeared during refresh"))?;
        current.source = relation;
    }
    sdm_schema::held::check_source_types(held, &found)?;
    Ok(sdm_schema::held::read(located, &found, false)?.by_the_source())
}

/// The rows a preview of `relation` reads, `columns` in order, or the
/// provider's failure whole.
pub(crate) async fn preview(
    asker: &Asker,
    relation: Relation,
    columns: Vec<String>,
) -> Result<Result<ArrowData, ProviderError>, CoreError> {
    attempt(
        asker,
        DataRequest::Preview {
            table: relation,
            columns,
            limit: PREVIEW_ROWS,
        },
    )
    .await
}

/// A note of the conversion (`Schema.conversion_notes`): its sentence, the
/// fields its template names, and the table and time column it is about.
pub(crate) fn note(
    sentence: Sentence,
    _text: String,
    fields: &[(&str, String)],
    table: Option<&str>,
    time_column: Option<&str>,
) -> ViewNote {
    ViewNote {
        sentence,
        fields: fields
            .iter()
            .map(|(name, value)| ((*name).to_owned(), Detail::text(value.clone())))
            .collect(),
        table: table.map(str::to_owned),
        time_column: time_column.map(str::to_owned),
    }
}

/// The note of a time column taken where the view's own columns would have
/// led elsewhere (`TimeColumnNote`).
pub(crate) fn time_note(taken: &TimeColumnNote) -> ViewNote {
    let reasons = crate::view_words::time_note_reasons(taken);
    let mut fields = vec![
        ("column", taken.column.clone()),
        ("table", taken.table.clone()),
        ("reasons", reasons),
    ];
    let sentence = match &taken.source_column {
        Some(source) => {
            fields.push(("source", source.clone()));
            Sentence::ViewTimeColumnNoteRenamed
        }
        None if taken.added => Sentence::ViewTimeColumnNoteAdded,
        None => Sentence::ViewTimeColumnNote,
    };
    note(
        sentence,
        crate::view_words::time_note_text(taken),
        &fields,
        Some(&taken.table),
        Some(&taken.column),
    )
}

/// A view whose model cannot be used (`Graph.validate`), refused by name.
pub(crate) fn invalid(view: &ViewChoice, kind: &str, refusal: &ModelRefusal) -> CoreError {
    let reason = match refusal {
        ModelRefusal::NoTable => "it gives no table".to_owned(),
        ModelRefusal::TableTaken(name) => format!("two of its tables are named '{name}'"),
        ModelRefusal::Edge(edge, why) => crate::view_words::edge_reason_text(edge, why),
    };
    CoreError::new(
        Sentence::ViewInvalid,
        format!(
            "Cannot convert {kind} '{}' into a graph since {reason}",
            view.name
        ),
    )
    .with_field("view_kind", Detail::text(kind))
    .with_field("view", Detail::text(&view.name))
    .with_field("reason", Detail::text(reason))
}

/// Preserve the conversion's choices for common schema assembly after aliases,
/// table selection, keys and explicit relationship replacement are known.
fn time_choices(
    model: &ViewModel,
    times: &BTreeMap<String, (Option<String>, Option<String>)>,
) -> BTreeMap<String, infer::TimeChoice> {
    times
        .iter()
        .map(|(name, (column, fallback))| {
            let key = model.table(name).and_then(|t| t.primary_key.as_ref());
            let permitted = |candidate: &Option<String>| {
                candidate.as_ref().filter(|c| Some(*c) != key).cloned()
            };
            (
                name.clone(),
                infer::TimeChoice {
                    column: permitted(column),
                    fallback: permitted(fallback),
                    ..infer::TimeChoice::default()
                },
            )
        })
        .collect()
}

/// Each table's time column once the links are known
/// (`_settle_time_columns`): a lone stamp of the last edit taken only in a
/// table nothing links to; the time column and the one inference took, by
/// table.
pub(crate) fn settle_times(
    model: &ViewModel,
    times: BTreeMap<String, (Option<String>, Option<String>)>,
) -> BTreeMap<String, (Option<String>, Option<String>)> {
    times
        .into_iter()
        .map(|(table, (column, fallback))| {
            let keyed = model.table(&table).and_then(|t| t.primary_key.clone());
            let settled = match &fallback {
                Some(stamp) if model.linked(&table) && column.as_ref() == Some(stamp) => {
                    (None, Some(stamp.clone()))
                }
                _ if column.is_some() && column == keyed => (None, column.clone()),
                _ => (column, None),
            };
            (table, settled)
        })
        .collect()
}

/// `table`'s columns as the model holds them: a key a join or relationship
/// added read as the source's column, and a derived column a join put the
/// source's in place of read so (`RE/graph.py:1856-1871`).
pub(crate) fn sync(table: &mut ViewTable, logical: &ModelTable) {
    for column in &logical.columns {
        let source = table.source_of(&column.name);
        match table.added.iter_mut().find(|a| a.name == column.name) {
            Some(added) => {
                if column.expr.is_none() && added.expr.is_some() {
                    added.expr = None;
                    added.column = source;
                    added.declared_type = None;
                    table.sampled.retain(|(name, _, _)| *name != column.name);
                }
            }
            None => table.added.push(Added {
                name: column.name.clone(),
                column: source,
                expr: None,
                declared: None,
                declared_type: None,
            }),
        }
    }
    table.added.retain(|a| logical.column(&a.name).is_some());
}

/// A table's time column, as taken (`_infer_view_time_column`).
pub(crate) struct TimeTaken {
    pub choice: Choice,
    pub note: Option<TimeColumnNote>,
    /// The table's columns inferred, the column taken added if it was.
    pub previewed: Previewed,
}

/// Takes the time column of a view's table (`RE/graph.py:274-373`,
/// `RE/base/table.py:859-893`): a truncated or cast date gives way to its
/// source column, added under a name of its own where a dimension has its
/// name, and a dimension a window measure orders by is offered alone
/// first; where the rows read may not be all the table's, the candidates'
/// ranges are read over the whole table, once per choice, as the old
/// client's choice reads them.
pub(crate) async fn time_column(
    reading: &Reading<'_>,
    table: &mut ViewTable,
    window_orders: &[String],
) -> Result<TimeTaken, CoreError> {
    let previewed = table.infer(reading, &[])?;
    let mut inputs = Vec::new();
    for column in &previewed.table().columns {
        let added = table.added.iter().find(|a| a.name == column.name);
        let expr = added.and_then(|a| a.expr.clone());
        let derived = match &expr {
            None => None,
            Some(expr) => match reading.ask(ViewTextAsk::DerivedColumn { expr: expr.clone() })? {
                ViewTextAnswer::Column(column) => column,
                _ => return Err(internal("the data layer read a column as another answer")),
            },
        };
        inputs.push(TimeInput {
            name: column.name.clone(),
            timestamp: column.stype == Stype::Timestamp,
            source: expr.is_none(),
            derived,
        });
    }
    let source_columns: Vec<(String, bool)> = table
        .catalog
        .columns
        .iter()
        .map(|c| {
            let name = table.view_name(&c.name);
            let timelike = match c.dtype {
                Some(dtype) => matches!(dtype, Dtype::Date | Dtype::Time),
                None => table.values(&name).is_some_and(|(field, _)| {
                    matches!(
                        field.data_type(),
                        arrow_schema::DataType::Date32
                            | arrow_schema::DataType::Date64
                            | arrow_schema::DataType::Timestamp(_, _)
                            | arrow_schema::DataType::Time64(_)
                            | arrow_schema::DataType::Time32(_)
                    )
                }),
            };
            (name, timelike)
        })
        .collect();
    // Each whole-table reading asked, by the columns it read.
    let mut wholes: Vec<(Vec<String>, Option<WholeTable>)> = Vec::new();
    let mut raws: Vec<(String, Field, ArrayRef)> = Vec::new();
    loop {
        match time_pass(
            reading,
            table,
            &inputs,
            &source_columns,
            window_orders,
            &wholes,
            &raws,
        )? {
            Pass::ReadRaw { name, column } => {
                let mut relation = table.base.clone();
                relation.columns = vec![RelationColumn {
                    name: name.clone(),
                    column: Some(column),
                    expression: None,
                }];
                relation.filter.clone_from(&table.filter);
                let read = match preview(reading.asker, relation, vec![name.clone()]).await? {
                    Ok(read)
                        if read
                            .batches
                            .iter()
                            .map(RecordBatch::num_rows)
                            .sum::<usize>()
                            > 0 =>
                    {
                        read
                    }
                    // Refused or empty, the column is not added.
                    _ => {
                        raws.push((
                            name.clone(),
                            Field::new(name, arrow_schema::DataType::Null, true),
                            Arc::new(arrow_array::NullArray::new(0)) as ArrayRef,
                        ));
                        continue;
                    }
                };
                let arrays: Vec<&dyn arrow_array::Array> =
                    read.batches.iter().map(|b| b.column(0).as_ref()).collect();
                let joined = arrow_select::concat::concat(&arrays)
                    .map_err(|_| internal("a column's sample does not concatenate"))?;
                raws.push((name, read.schema.field(0).clone(), joined));
            }
            Pass::Read { asked, relation } => {
                // The candidates' ranges over every row, in one read, as
                // the sample reads them (`_full_range`); a failure leaves
                // the choice to the rows read, saying so.
                let request = DataRequest::ColumnRanges {
                    table: relation,
                    columns: asked.clone(),
                };
                let whole = match attempt(reading.asker, request).await? {
                    Ok(read) => Some(infer::whole_table(&asked, &read.batches, reading.reader)?),
                    Err(_) => None,
                };
                wholes.push((asked, whole));
            }
            Pass::Done {
                outcome,
                kept,
                choice,
            } => {
                for (added, field, array) in kept {
                    table.sampled.push((added.name.clone(), field, array));
                    table.added.push(added);
                }
                let previewed = table.infer(reading, &[])?;
                let choice = Choice {
                    column: outcome.chosen.clone(),
                    fallback: choice.fallback.filter(|_| outcome.chosen.is_some()),
                };
                return Ok(TimeTaken {
                    choice,
                    note: outcome.note,
                    previewed,
                });
            }
        }
    }
}

/// One pass of a time column's choice: done, or a whole-table reading it
/// needs first.
enum Pass {
    Read {
        asked: Vec<String>,
        relation: Relation,
    },
    /// The source column `column`, under the name `name`, read as the
    /// sample reads it (`ColumnSpec(name, expr=column)`).
    ReadRaw { name: String, column: String },
    Done {
        outcome: model::TimeOutcome,
        /// The columns added and kept, with their values.
        kept: Vec<(Added, Field, ArrayRef)>,
        choice: Choice,
    },
}

/// The time column's choice over what was read, `wholes` the whole-table
/// readings asked so far, or the first reading it still needs.
fn time_pass(
    reading: &Reading<'_>,
    table: &ViewTable,
    inputs: &[TimeInput],
    source_columns: &[(String, bool)],
    window_orders: &[String],
    wholes: &[(Vec<String>, Option<WholeTable>)],
    raws: &[(String, Field, ArrayRef)],
) -> Result<Pass, CoreError> {
    let pending: RefCell<Option<Vec<String>>> = RefCell::new(None);
    let pending_raw: RefCell<Option<(String, String)>> = RefCell::new(None);
    let failed: RefCell<Option<CoreError>> = RefCell::new(None);
    let extra: RefCell<Vec<(Added, Field, ArrayRef)>> = RefCell::new(Vec::new());
    let last: RefCell<Choice> = RefCell::new(Choice::default());
    let infer_with = |extra: &[(Added, Field, ArrayRef)]| {
        let mut alone = table.clone_for_reading();
        for (added, field, array) in extra {
            alone
                .sampled
                .push((added.name.clone(), field.clone(), Arc::clone(array)));
        }
        let adds: Vec<Added> = extra.iter().map(|(a, _, _)| a.clone()).collect();
        alone.infer(reading, &adds)
    };
    let outcome = model::time_column(
        &table.name,
        inputs,
        source_columns,
        window_orders,
        |raw| {
            let refused = model::Added {
                added: false,
                timestamp: false,
            };
            if pending_raw.borrow().is_some() {
                return refused;
            }
            let Some(column) = table.source_of(&raw.raw) else {
                return refused;
            };
            // A source column under its own name is read with the source's
            // rows; under another, it is read alone, as the old client
            // samples an expression.
            let values = if raw.name == raw.raw {
                table.values(&raw.raw)
            } else {
                match raws.iter().find(|(name, _, _)| *name == raw.name) {
                    Some((_, field, array)) => Some((field.clone(), Arc::clone(array))),
                    None => {
                        *pending_raw.borrow_mut() = Some((raw.name.clone(), column));
                        return refused;
                    }
                }
            };
            let Some((field, array)) = values else {
                return refused;
            };
            let added = Added {
                name: raw.name.clone(),
                column: Some(column),
                expr: None,
                declared: None,
                declared_type: None,
            };
            let mut tried = extra.borrow().clone();
            tried.push((added.clone(), field.clone(), Arc::clone(&array)));
            match infer_with(&tried) {
                Ok(read) => {
                    let timestamp = read
                        .table()
                        .column(&raw.name)
                        .is_some_and(|c| c.stype == Stype::Timestamp);
                    extra.borrow_mut().push((added, field, array));
                    model::Added {
                        added: true,
                        timestamp,
                    }
                }
                Err(_) => refused,
            }
        },
        |among| {
            if pending.borrow().is_some()
                || pending_raw.borrow().is_some()
                || failed.borrow().is_some()
            {
                return Choice::default();
            }
            let read = match infer_with(&extra.borrow()) {
                Ok(read) => read,
                Err(error) => {
                    failed.borrow_mut().get_or_insert(error);
                    return Choice::default();
                }
            };
            let asked = read.whole_table_columns_among(among);
            let whole = if asked.is_empty() {
                Some((None, false))
            } else {
                wholes
                    .iter()
                    .find(|(columns, _)| *columns == asked)
                    .map(|(_, whole)| (whole.clone(), whole.is_none()))
            };
            let Some((whole, unread)) = whole else {
                *pending.borrow_mut() = Some(asked);
                return Choice::default();
            };
            match infer::choose_among(&read, among, whole.as_ref(), unread) {
                Ok(choice) => {
                    let chosen = Choice {
                        column: choice.column.clone(),
                        fallback: choice.fallback.clone(),
                    };
                    *last.borrow_mut() = chosen.clone();
                    chosen
                }
                Err(error) => {
                    failed.borrow_mut().get_or_insert(error);
                    Choice::default()
                }
            }
        },
    );
    if let Some(error) = failed.into_inner() {
        return Err(error);
    }
    if let Some((name, column)) = pending_raw.into_inner() {
        return Ok(Pass::ReadRaw { name, column });
    }
    if let Some(asked) = pending.into_inner() {
        let extra: Vec<Added> = extra.borrow().iter().map(|(a, _, _)| a.clone()).collect();
        let relation = table.ranges_relation(&asked, &extra);
        return Ok(Pass::Read { asked, relation });
    }
    let kept = extra
        .into_inner()
        .into_iter()
        .filter(|(added, _, _)| outcome.kept.contains(&added.name))
        .collect();
    Ok(Pass::Done {
        outcome,
        kept,
        choice: last.into_inner(),
    })
}

/// The time column of a view's table chosen among `candidates`
/// (`_infer_time_column_among`): over the rows read, and, where they may
/// not be all the table's, the candidates' ranges over every row, read once.
pub(crate) async fn choose_among(
    reading: &Reading<'_>,
    table: &mut ViewTable,
    previewed: &Previewed,
    candidates: &[String],
) -> Result<Choice, CoreError> {
    let asked = previewed.whole_table_columns_among(candidates);
    let (whole, unread) = if asked.is_empty() {
        (None, false)
    } else {
        let request = DataRequest::ColumnRanges {
            table: table.ranges_relation(&asked, &[]),
            columns: asked.clone(),
        };
        match attempt(reading.asker, request).await? {
            Ok(read) => (
                Some(infer::whole_table(&asked, &read.batches, reading.reader)?),
                false,
            ),
            Err(_) => (None, true),
        }
    };
    let choice = infer::choose_among(previewed, candidates, whole.as_ref(), unread)?;
    Ok(Choice {
        column: choice.column,
        fallback: choice.fallback,
    })
}
