// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Admit one provider-computed column without re-inferring existing declarations.
use crate::run::{Asker, Attachment, Attachments, Run, internal};
use sdm_contract::data::{DataRequest, RelationColumn};
use sdm_contract::ops::{
    DescribedColumn, DescribedSource, HeldSchema, NamedTable, OperationResult,
};
use sdm_contract::sql_column::{SqlColumn, SqlColumnCall};
use sdm_contract::{CoreError, Detail, HostFacts, Sentence};

fn invalid(reason: &str) -> CoreError {
    CoreError::new(Sentence::SchemaRefinement, reason).with_field("reason", Detail::text(reason))
}

pub(crate) fn start(
    root: &crate::compose::CompositionRoot,
    request: SqlColumn,
    mut attachments: Attachments,
    facts: HostFacts,
) -> Result<Run, CoreError> {
    let held: HeldSchema = crate::attached::take_json(&mut attachments, request.schema)?;
    let call: SqlColumnCall = crate::attached::take_json(&mut attachments, request.input)?;
    let data = root.data_access(&request.provider)?;
    Ok(Run::with_data(data, move |asker| async move {
        run(asker, held, call, facts).await
    }))
}

async fn run(
    asker: Asker,
    held: HeldSchema,
    call: SqlColumnCall,
    facts: HostFacts,
) -> Result<OperationResult, CoreError> {
    if call.name.is_empty() || call.expression.trim().is_empty() {
        return Err(invalid(
            "the column name and SQL expression must be nonempty",
        ));
    }
    if call.name.starts_with(sdm_engine::graph::RESERVED_PREFIX)
        || sdm_request::payload::RESERVED.contains(&call.name.as_str())
    {
        return Err(invalid(
            "the calculated column name is reserved by the engine",
        ));
    }
    let original = held.clone();
    let mut description = crate::describe::initialize(&asker, Some(&held), &facts)
        .await?
        .ok_or_else(|| internal("SQL column source was not initialized"))?;
    let table = description
        .tables
        .iter_mut()
        .find(|table| table.name == call.table)
        .ok_or_else(|| invalid("the schema has no such table"))?;
    if table.columns.iter().any(|column| column.name == call.name) {
        return Err(invalid("a calculated column must have a new name"));
    }
    let mut physical = table.source.clone();
    physical.columns.clear();
    physical.found = false;
    // Validate the expression in physical-source scope alone. This keeps newly
    // calculated aliases out of scope even on dialects with lateral aliases.
    let mut expression = physical;
    expression.columns.push(RelationColumn {
        name: call.name.clone(),
        column: None,
        expression: Some(call.expression.clone()),
    });
    let values = crate::describe::ask(
        &asker,
        DataRequest::Preview {
            table: expression.clone(),
            columns: vec![call.name.clone()],
            limit: sdm_schema::infer::PREVIEW_ROWS,
        },
    )
    .await?;
    let catalog = vec![sdm_contract::catalog::CatalogColumn {
        name: call.name.clone(),
        declared_type: String::new(),
        not_null: false,
        primary_key: false,
        key_nullable: true,
        unique: false,
        dtype: None,
    }];
    let reader = crate::describe::text_reader(&facts)?;
    let inferred = sdm_schema::infer::view_table(
        &table.name,
        &expression,
        &catalog,
        &values.schema,
        &values.batches,
        &reader,
        if asker.reads_nan_as_missing() {
            sdm_schema::infer::Floats::NanMissing
        } else {
            sdm_schema::infer::Floats::NanValue
        },
    )?;
    let column = inferred
        .table()
        .columns
        .first()
        .ok_or_else(|| internal("SQL column preview inferred no column"))?;
    let mut relation = table.source.clone();
    if relation.columns.is_empty() || relation.found {
        relation.columns = table
            .columns
            .iter()
            .map(|column| RelationColumn {
                name: column.name.clone(),
                column: Some(column.name.clone()),
                expression: None,
            })
            .collect();
    }
    relation.found = false;
    relation.columns.push(RelationColumn {
        name: call.name.clone(),
        column: None,
        expression: Some(call.expression.clone()),
    });
    // Validate the complete retained projection before replacing any metadata.
    crate::discover::table(
        &asker,
        &NamedTable {
            name: table.name.clone(),
            source: relation.clone(),
            columns: Some(
                relation
                    .columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect(),
            ),
        },
        true,
        None,
    )
    .await?;
    table.source = relation;
    table
        .calculated_columns
        .push(sdm_contract::sql_column::CalculatedColumn {
            name: call.name,
            expression: call.expression.clone(),
        });
    table.columns.push(DescribedColumn {
        name: column.name.clone(),
        received: column.received.clone(),
        dtype: column.dtype,
        stype: column.stype,
        stype_before_key: None,
        nullable: column.nullable,
        unique: false,
        declared_type: None,
        exact_number: column.exact_number,
    });
    // The source identity stays logical; only the declarations are durable.
    for table in &mut description.tables {
        table.source = original
            .tables
            .iter()
            .find(|t| t.name == table.name)
            .ok_or_else(|| internal("SQL column lost original table"))?
            .source
            .clone();
    }
    let bytes = serde_json::to_vec(&description)
        .map_err(|_| internal("SQL column schema serialization"))?;
    Ok(OperationResult::DescribeSource(DescribedSource {
        description: asker.attach(Attachment::Bytes(bytes))?,
        notes: vec![],
    }))
}

/// Rebuild caller projections over a freshly resolved base, never cached SQL.
pub(crate) fn project(
    base: &sdm_contract::data::Relation,
    names: impl IntoIterator<Item = String>,
    calculated: &[sdm_contract::sql_column::CalculatedColumn],
) -> Result<sdm_contract::data::Relation, CoreError> {
    if calculated.is_empty() {
        return Ok(base.clone());
    }
    let mut relation = base.clone();
    if relation.columns.is_empty() || relation.found {
        relation.columns = names
            .into_iter()
            .map(|name| RelationColumn {
                column: Some(name.clone()),
                name,
                expression: None,
            })
            .collect();
    }
    relation.found = false;
    for column in calculated {
        if column.name.is_empty()
            || column.expression.trim().is_empty()
            || column.name.starts_with(sdm_engine::graph::RESERVED_PREFIX)
            || sdm_request::payload::RESERVED.contains(&column.name.as_str())
            || relation.columns.iter().any(|c| c.name == column.name)
        {
            return Err(invalid(
                "invalid or duplicate calculated column declaration",
            ));
        }
        relation.columns.push(RelationColumn {
            name: column.name.clone(),
            column: None,
            expression: Some(column.expression.clone()),
        });
    }
    Ok(relation)
}
