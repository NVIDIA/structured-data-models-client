// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A catalog answer in the standard shape every source answers `describe`
//! in ([`sdm_contract::data::catalog_columns`]), the optional facts
//! included (contract v76): what a built-in source answers through the kit
//! once its dialect's own catalog rows are read, so the core reads every
//! source's catalog one way; and every other answer whose shape the
//! contract fixes ([`fixed`]), in that shape.

use std::sync::Arc;

use arrow_array::builder::{BooleanBuilder, Int64Builder, ListBuilder, StringBuilder};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use sdm_contract::catalog::{CatalogAnswer, IndexCoverage, KeyKind};
use sdm_contract::data::catalog_columns as c;
use sdm_contract::data::{DataRequest, ExpectedSchema};
use sdm_contract::{CoreError, Detail, Sentence};

fn names_type() -> DataType {
    DataType::List(Arc::new(Field::new(c::LIST_ITEM, DataType::Utf8, true)))
}

fn names_builder() -> ListBuilder<StringBuilder> {
    ListBuilder::new(StringBuilder::new()).with_field(Arc::new(Field::new(
        c::LIST_ITEM,
        DataType::Utf8,
        true,
    )))
}

fn push_names(builder: &mut ListBuilder<StringBuilder>, names: &[String]) {
    for name in names {
        builder.values().append_value(name);
    }
    builder.append(true);
}

fn texts(values: impl IntoIterator<Item = Option<String>>) -> ArrayRef {
    let mut builder = StringBuilder::new();
    for value in values {
        builder.append_option(value);
    }
    Arc::new(builder.finish())
}

fn flags(values: impl IntoIterator<Item = bool>) -> ArrayRef {
    let mut builder = BooleanBuilder::new();
    for value in values {
        builder.append_value(value);
    }
    Arc::new(builder.finish())
}

fn kind_name(kind: KeyKind) -> &'static str {
    match kind {
        KeyKind::Primary => c::KIND_PRIMARY,
        KeyKind::Unique => c::KIND_UNIQUE,
        KeyKind::PartialUnique => c::KIND_PARTIAL_UNIQUE,
        KeyKind::Foreign => c::KIND_FOREIGN,
        KeyKind::Other => "other",
    }
}

fn batch(
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
) -> Result<(SchemaRef, RecordBatch), CoreError> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).map_err(|_| {
        CoreError::new(
            Sentence::Internal,
            "a catalog answer could not be written in the standard shape",
        )
    })?;
    Ok((schema, batch))
}

/// `answer` in the standard shape of its part; `None` for a semantic
/// model's definition, which is not written in it.
pub(crate) fn catalog(
    answer: &CatalogAnswer,
) -> Result<Option<(SchemaRef, RecordBatch)>, CoreError> {
    let written = match answer {
        CatalogAnswer::Tables(names) | CatalogAnswer::KeyEnforcement(names) => batch(
            vec![Field::new(c::TABLE, DataType::Utf8, false)],
            vec![texts(names.iter().cloned().map(Some))],
        )?,
        CatalogAnswer::Namespace(parts) => batch(
            vec![Field::new(c::NAME, DataType::Utf8, false)],
            vec![texts(parts.iter().cloned().map(Some))],
        )?,
        CatalogAnswer::IndexColumns(names) => batch(
            vec![Field::new(c::NAME, DataType::Utf8, true)],
            vec![texts(names.iter().cloned())],
        )?,
        CatalogAnswer::Columns(columns) => {
            let dtypes = columns
                .iter()
                .map(|column| match column.dtype {
                    Some(dtype) => match serde_json::to_value(dtype) {
                        Ok(serde_json::Value::String(name)) => Ok(Some(name)),
                        _ => Err(CoreError::new(
                            Sentence::Internal,
                            "a column's dtype has no name",
                        )),
                    },
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>, _>>()?;
            batch(
                vec![
                    Field::new(c::NAME, DataType::Utf8, false),
                    Field::new(c::TYPE, DataType::Utf8, false),
                    Field::new(c::NOT_NULL, DataType::Boolean, false),
                    Field::new(c::PRIMARY_KEY, DataType::Boolean, false),
                    Field::new(c::KEY_NULLABLE, DataType::Boolean, false),
                    Field::new(c::UNIQUE, DataType::Boolean, false),
                    Field::new(c::DTYPE, DataType::Utf8, true),
                ],
                vec![
                    texts(columns.iter().map(|column| Some(column.name.clone()))),
                    texts(
                        columns
                            .iter()
                            .map(|column| Some(column.declared_type.clone())),
                    ),
                    flags(columns.iter().map(|column| column.not_null)),
                    flags(columns.iter().map(|column| column.primary_key)),
                    flags(columns.iter().map(|column| column.key_nullable)),
                    flags(columns.iter().map(|column| column.unique)),
                    texts(dtypes),
                ],
            )?
        }
        CatalogAnswer::Keys(keys) => {
            let mut columns = names_builder();
            for key in keys {
                push_names(&mut columns, &key.columns);
            }
            batch(
                vec![
                    Field::new(c::KIND, DataType::Utf8, false),
                    Field::new(c::COLUMNS, names_type(), false),
                    Field::new(c::INDEX, DataType::Utf8, true),
                    Field::new(c::ENFORCED, DataType::Boolean, false),
                ],
                vec![
                    texts(keys.iter().map(|key| Some(kind_name(key.kind).to_owned()))),
                    Arc::new(columns.finish()),
                    texts(keys.iter().map(|key| key.index.clone())),
                    flags(keys.iter().map(|key| key.enforced)),
                ],
            )?
        }
        CatalogAnswer::ForeignKeys(keys) => {
            let mut columns = names_builder();
            let mut referenced = names_builder();
            let mut numbers = Int64Builder::new();
            for key in keys {
                numbers.append_value(key.key);
                push_names(&mut columns, &key.columns);
                push_names(&mut referenced, &key.referenced_columns);
            }
            batch(
                vec![
                    Field::new(c::KEY, DataType::Int64, false),
                    Field::new(c::REFERENCED_TABLE, DataType::Utf8, false),
                    Field::new(c::COLUMNS, names_type(), false),
                    Field::new(c::REFERENCED_COLUMNS, names_type(), false),
                ],
                vec![
                    Arc::new(numbers.finish()),
                    texts(keys.iter().map(|key| Some(key.referenced_table.clone()))),
                    Arc::new(columns.finish()),
                    Arc::new(referenced.finish()),
                ],
            )?
        }
        CatalogAnswer::Indexes(indexes) => {
            let mut columns = names_builder();
            for index in indexes {
                match &index.columns {
                    IndexCoverage::Resolved(names) => push_names(&mut columns, names),
                    IndexCoverage::Unresolved | IndexCoverage::Named(_) => {
                        push_names(&mut columns, &[])
                    }
                }
            }
            batch(
                vec![
                    Field::new(c::TABLE, DataType::Utf8, false),
                    Field::new(c::COLUMNS, names_type(), false),
                    Field::new(c::INDEX, DataType::Utf8, true),
                    Field::new(c::RESOLVED, DataType::Boolean, false),
                ],
                vec![
                    texts(indexes.iter().map(|index| Some(index.table.clone()))),
                    Arc::new(columns.finish()),
                    texts(indexes.iter().map(|index| match &index.columns {
                        IndexCoverage::Named(name) => Some(name.clone()),
                        _ => None,
                    })),
                    flags(
                        indexes
                            .iter()
                            .map(|index| !matches!(index.columns, IndexCoverage::Unresolved)),
                    ),
                ],
            )?
        }
        CatalogAnswer::ViewDefinition(_) => return Ok(None),
    };
    Ok(Some(written))
}

/// The answer to a request other than a `describe` whose shape the
/// contract fixes ([`DataRequest::expected`]: the key counts), written in
/// that shape: each column, by its position, under the contract's name and
/// cast to its type, as a dialect's statement answers it under the names
/// and types its engine gives (SQLite's integers all `int64`, Snowflake's
/// names upper-cased). `None` for a request of no fixed shape, or an
/// answer already in it; refused by name where the answer has another
/// number of columns, or a value its column's type does not hold.
pub fn fixed(
    request: &DataRequest,
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<Option<(SchemaRef, Vec<RecordBatch>)>, CoreError> {
    if matches!(request, DataRequest::Describe { .. }) {
        return Ok(None);
    }
    let ExpectedSchema::Schema { fields, .. } = request.expected() else {
        return Ok(None);
    };
    let standard = Arc::new(Schema::new(
        fields.iter().map(|f| f.to_arrow()).collect::<Vec<_>>(),
    ));
    if schema.as_ref() == standard.as_ref() {
        return Ok(None);
    }
    let mismatch = |reason: &str| {
        CoreError::new(
            Sentence::ProviderSchemaMismatch,
            format!("the provider's data does not match the expected schema: {reason}"),
        )
        .with_field("check", Detail::text("expected"))
        .with_field("reason", Detail::text(reason))
    };
    if schema.fields().len() != standard.fields().len() {
        return Err(mismatch("the answer has another number of columns"));
    }
    let options = arrow_cast::CastOptions {
        safe: false,
        ..Default::default()
    };
    let written = batches
        .iter()
        .map(|batch| {
            let columns = batch
                .columns()
                .iter()
                .zip(standard.fields())
                .map(|(column, field)| {
                    arrow_cast::cast_with_options(column, field.data_type(), &options)
                        .map_err(|_| mismatch("a value its column's type does not hold"))
                })
                .collect::<Result<Vec<ArrayRef>, CoreError>>()?;
            RecordBatch::try_new(Arc::clone(&standard), columns)
                .map_err(|_| mismatch("a column holds a null where none is allowed"))
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    Ok(Some((standard, written)))
}

#[cfg(test)]
mod tests {
    use sdm_contract::catalog::{
        CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogIndex, CatalogKey, IndexCoverage,
        KeyKind,
    };
    use sdm_contract::data::DescribePart;
    use sdm_contract::ops::Dtype;

    /// Every fact of `answer` survives the standard shape: written, then
    /// read back by the reader every source's answer is read by.
    fn round_trip(part: DescribePart, answer: CatalogAnswer) {
        let (_, batch) = super::catalog(&answer).unwrap().unwrap();
        assert_eq!(
            sdm_contract::catalog_read::standard(part, &[batch]).unwrap(),
            answer
        );
    }

    #[test]
    fn every_catalog_fact_survives_the_standard_shape() {
        round_trip(
            DescribePart::Columns,
            CatalogAnswer::Columns(vec![
                CatalogColumn {
                    name: "id".into(),
                    declared_type: "INTEGER".into(),
                    not_null: false,
                    primary_key: true,
                    key_nullable: false,
                    unique: true,
                    dtype: Some(Dtype::Int64),
                },
                CatalogColumn {
                    name: "code".into(),
                    declared_type: "TEXT".into(),
                    not_null: false,
                    primary_key: true,
                    key_nullable: true,
                    unique: false,
                    dtype: None,
                },
            ]),
        );
        round_trip(
            DescribePart::Keys,
            CatalogAnswer::Keys(vec![
                CatalogKey {
                    kind: KeyKind::Primary,
                    columns: vec!["id".into()],
                    index: None,
                    enforced: false,
                },
                CatalogKey {
                    kind: KeyKind::PartialUnique,
                    columns: Vec::new(),
                    index: Some("ix_code".into()),
                    enforced: true,
                },
            ]),
        );
        round_trip(
            DescribePart::Indexes,
            CatalogAnswer::Indexes(vec![
                CatalogIndex {
                    table: "t".into(),
                    columns: IndexCoverage::Resolved(vec!["a".into(), "b".into()]),
                },
                CatalogIndex {
                    table: "t".into(),
                    columns: IndexCoverage::Unresolved,
                },
                CatalogIndex {
                    table: "t".into(),
                    columns: IndexCoverage::Named("ix".into()),
                },
            ]),
        );
        round_trip(
            DescribePart::ForeignKeys,
            CatalogAnswer::ForeignKeys(vec![CatalogForeignKey {
                key: 0,
                referenced_table: "users".into(),
                columns: vec!["user_id".into()],
                referenced_columns: vec!["id".into()],
            }]),
        );
        round_trip(
            DescribePart::Tables,
            CatalogAnswer::Tables(vec!["users".into(), "orders".into()]),
        );
        round_trip(
            DescribePart::Namespace,
            CatalogAnswer::Namespace(vec!["main".into(), "shop".into()]),
        );
        round_trip(
            DescribePart::IndexColumns,
            CatalogAnswer::IndexColumns(vec![Some("a".into()), None]),
        );
    }
}
