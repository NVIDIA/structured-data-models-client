// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A second data layer (T2.7, plan section 3, "Data access"): the data
//! port (`sdm_contract::data::DataLayer`, decision H14) implemented apart
//! from `sdm-data`, for host providers alone, composed at the composition
//! root in place of `sdm-data`'s ([`root`]). Dev-only.
//!
//! It renders no SQL and serves no dialect: a run naming one is refused
//! before it exists, `UNSUPPORTED_REQUEST` naming the dialect. A host
//! provider's run takes the capabilities the provider declared, each
//! request is passed to the provider as itself (`FetchPlan::Host`) with
//! the schema the contract fixes for its answer, and a `describe` answer,
//! in the contract's shape for a host provider (`catalog_columns`), is
//! read here into the catalog's answer. Nothing else is declared: no
//! opaque vendor, no stored-time comparison, no sampling by a key's type
//! (the trait's defaults). It is built from the contract's public types
//! alone (this crate takes `sdm-contract` and `sdm-host`, no domain crate
//! but the contract), so the same prediction running on it shows the
//! core needs no change to swap its data layer.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, ArrayRef, RecordBatch};
use sdm_contract::catalog::{
    CatalogAnswer, CatalogColumn, CatalogForeignKey, CatalogIndex, CatalogKey, IndexCoverage,
    KeyKind,
};
use sdm_contract::data::{
    DataLayer, DataRequest, DescribePart, Dialect, FetchPlan, ProviderCapabilities, ProviderChoice,
    catalog_columns as c, views_unread,
};
use sdm_contract::{CoreError, Detail, Sentence};
use sdm_host::CompositionRoot;

/// The data layer of host providers alone.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostLayer;

/// The composition root over [`HostLayer`], in place of
/// `sdm_host::default_root`'s, over `sdm-data`.
pub fn root() -> CompositionRoot {
    CompositionRoot::new(Arc::new(HostLayer))
}

/// The refusal of a SQL dialect, which this layer renders none of.
fn no_dialect(dialect: &Dialect) -> CoreError {
    CoreError::new(
        Sentence::NoDialectNamed,
        "this data layer renders no SQL dialect",
    )
    .with_field("dialect", Detail::text(dialect.name()))
}

impl DataLayer for HostLayer {
    fn capabilities(&self, provider: &ProviderChoice) -> Result<ProviderCapabilities, CoreError> {
        match provider {
            ProviderChoice::Host { capabilities } => Ok(capabilities.clone()),
            ProviderChoice::Sql { dialect, .. } => Err(no_dialect(dialect)),
        }
    }

    fn plan(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
    ) -> Result<FetchPlan, CoreError> {
        match provider {
            ProviderChoice::Host { .. } => Ok(FetchPlan::Host {
                expect: request.expected(),
            }),
            ProviderChoice::Sql { dialect, .. } => Err(no_dialect(dialect)),
        }
    }

    fn read(
        &self,
        request: &DataRequest,
        provider: &ProviderChoice,
        batches: &[RecordBatch],
    ) -> Result<CatalogAnswer, CoreError> {
        if let ProviderChoice::Sql { dialect, .. } = provider {
            return Err(no_dialect(dialect));
        }
        let DataRequest::Describe { part, .. } = request else {
            return Err(CoreError::new(
                Sentence::Internal,
                "only a catalog answer is read by the data layer",
            )
            .with_field("request", Detail::text(request.name())));
        };
        let mut rows = Rows::default();
        for batch in batches {
            rows.read(*part, batch)?;
        }
        rows.answer(*part)
    }
}

/// An answer the layer cannot read, which the core's check of its shape
/// (decision H16, check (b)) should have refused first.
fn unread_answer() -> CoreError {
    CoreError::new(
        Sentence::Internal,
        "a catalog answer is not in the shape its plan expects",
    )
}

/// The column `name` of `batch`.
fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, CoreError> {
    batch
        .column_by_name(name)
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))
}

/// Row `row` of the text column `name`, `None` where null.
fn text(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<String>, CoreError> {
    let strings = column(batch, name)?
        .as_string_opt::<i32>()
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))?;
    Ok(strings.is_valid(row).then(|| strings.value(row).to_owned()))
}

/// Row `row` of the Boolean column `name`, false where null.
fn flag(batch: &RecordBatch, name: &str, row: usize) -> Result<bool, CoreError> {
    let flags = column(batch, name)?
        .as_boolean_opt()
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))?;
    Ok(flags.is_valid(row) && flags.value(row))
}

/// Row `row` of the list-of-names column `name`: its names in order, a
/// null name dropped and the others kept, none where the list is null, as
/// `sdm-data`'s catalog reader reads a key's and a foreign key's columns
/// (P2-T2.7-1).
fn kept_names(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<String>, CoreError> {
    let lists = column(batch, name)?
        .as_list_opt::<i32>()
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))?;
    if lists.is_null(row) {
        return Ok(Vec::new());
    }
    let items = lists.value(row);
    let strings = items
        .as_string_opt::<i32>()
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))?;
    Ok((0..strings.len())
        .filter(|&i| strings.is_valid(i))
        .map(|i| strings.value(i).to_owned())
        .collect())
}

/// Row `row` of the list-of-names column `name`: its names in order, or
/// `None` where the list, or any name in it, is null, as `sdm-data` reads
/// an index's columns (unresolved, never the names left over).
fn names(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<Vec<String>>, CoreError> {
    let lists = column(batch, name)?
        .as_list_opt::<i32>()
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))?;
    if lists.is_null(row) {
        return Ok(None);
    }
    let items = lists.value(row);
    let strings = items
        .as_string_opt::<i32>()
        .ok_or_else(|| unread_answer().with_field("column", Detail::text(name)))?;
    if strings.null_count() > 0 {
        return Ok(None);
    }
    Ok(Some(
        (0..strings.len())
            .map(|i| strings.value(i).to_owned())
            .collect(),
    ))
}

/// What the answer's batches held, read.
#[derive(Default)]
struct Rows {
    tables: Vec<String>,
    columns: Vec<CatalogColumn>,
    keys: Vec<CatalogKey>,
    foreign_keys: Vec<CatalogForeignKey>,
    indexes: Vec<CatalogIndex>,
    index_columns: Vec<Option<String>>,
    namespace: Vec<String>,
}

impl Rows {
    fn read(&mut self, part: DescribePart, batch: &RecordBatch) -> Result<(), CoreError> {
        for row in 0..batch.num_rows() {
            match part {
                DescribePart::ViewDefinition => return Err(views_unread()),
                DescribePart::Tables => self.tables.extend(text(batch, c::TABLE, row)?),
                DescribePart::Columns => {
                    if let Some(name) = text(batch, c::NAME, row)? {
                        self.columns.push(CatalogColumn {
                            name,
                            declared_type: text(batch, c::TYPE, row)?.unwrap_or_default(),
                            not_null: flag(batch, c::NOT_NULL, row)?,
                            primary_key: flag(batch, c::PRIMARY_KEY, row)?,
                            key_nullable: false,
                            unique: false,
                            dtype: None,
                        });
                    }
                }
                DescribePart::Keys => self.keys.push(CatalogKey {
                    kind: match text(batch, c::KIND, row)?.as_deref() {
                        Some(c::KIND_PRIMARY) => KeyKind::Primary,
                        Some(c::KIND_UNIQUE) => KeyKind::Unique,
                        Some(c::KIND_FOREIGN) => KeyKind::Foreign,
                        _ => KeyKind::Other,
                    },
                    columns: kept_names(batch, c::COLUMNS, row)?,
                    index: None,
                    enforced: true,
                }),
                DescribePart::ForeignKeys => {
                    let key = column(batch, c::KEY)?
                        .as_primitive_opt::<Int64Type>()
                        .ok_or_else(|| unread_answer().with_field("column", Detail::text(c::KEY)))?
                        .value(row);
                    self.foreign_keys.push(CatalogForeignKey {
                        key,
                        referenced_table: text(batch, c::REFERENCED_TABLE, row)?
                            .unwrap_or_default(),
                        columns: kept_names(batch, c::COLUMNS, row)?,
                        referenced_columns: kept_names(batch, c::REFERENCED_COLUMNS, row)?,
                    });
                }
                DescribePart::Indexes => {
                    if let Some(table) = text(batch, c::TABLE, row)? {
                        self.indexes.push(CatalogIndex {
                            table,
                            columns: names(batch, c::COLUMNS, row)?
                                .map_or(IndexCoverage::Unresolved, IndexCoverage::Resolved),
                        });
                    }
                }
                DescribePart::IndexColumns => {
                    self.index_columns.push(text(batch, c::NAME, row)?);
                }
                DescribePart::Namespace => self.namespace.extend(text(batch, c::NAME, row)?),
            }
        }
        Ok(())
    }

    fn answer(self, part: DescribePart) -> Result<CatalogAnswer, CoreError> {
        Ok(match part {
            // A host provider is never asked for a view's definition (T3.27).
            DescribePart::ViewDefinition => return Err(views_unread()),
            DescribePart::Tables => CatalogAnswer::Tables(self.tables),
            DescribePart::Columns => CatalogAnswer::Columns(self.columns),
            DescribePart::Keys => CatalogAnswer::Keys(self.keys),
            DescribePart::ForeignKeys => CatalogAnswer::ForeignKeys(self.foreign_keys),
            DescribePart::Indexes => CatalogAnswer::Indexes(self.indexes),
            DescribePart::IndexColumns => CatalogAnswer::IndexColumns(self.index_columns),
            DescribePart::Namespace => CatalogAnswer::Namespace(self.namespace),
        })
    }
}
