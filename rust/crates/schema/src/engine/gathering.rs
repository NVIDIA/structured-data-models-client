// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What gathering a request reads of an engine table over a source, past
//! what describing it read: a copy of the table for one gather
//! (`EngineTables._copy_for_one_gather`), the row counts each backend's
//! sampler takes (`_get_num_rows`), and what its statements depend on
//! (`DuckDBTable._is_zoned_time`, `PostgresTable._postgres_type`, and
//! Snowflake's types and listing).

use std::collections::BTreeMap;

use sdm_dialects::Statement;

use super::{Backend, Ctx, ER, EngineTable, PyExc, int_of, rows, text_of};
use crate::pyval::PyVal;

/// A row of Snowflake's `SHOW TABLES`, by field.
pub type Listed = Vec<(String, PyVal)>;

impl EngineTable {
    /// `_copy_for_one_gather(table, name, columns)`: a copy under `name`,
    /// with no key and no time columns, keeping only `columns`.
    pub fn copy_for_one_gather(&self, name: &str, columns: &[String]) -> ER<EngineTable> {
        let mut copied = self.clone();
        copied.name = name.to_string();
        copied.set_end_time_column(None)?;
        copied.set_time_column(None)?;
        copied.set_primary_key(None)?;
        let leaving: Vec<String> = copied
            .columns
            .iter()
            .map(|c| c.name.clone())
            .filter(|c| !columns.contains(c))
            .collect();
        for column in leaving {
            copied.remove_column(&column)?;
        }
        Ok(copied)
    }

    /// `table[name].stype = stype`.
    pub fn set_column_stype(&mut self, name: &str, stype: sdm_pql::types::Stype) -> ER<()> {
        match self.column_mut(name) {
            Some(column) => column.set_stype(stype),
            None => Err(self.not_found(name)),
        }
    }

    /// `DuckDBTable._is_zoned_time(name)`: whether a column holds
    /// `TIMESTAMP WITH TIME ZONE` values. An expression column read zoned is
    /// not one a gather over plain tables has.
    pub fn is_zoned_time(&self, name: &str) -> bool {
        let is_source = self
            .source_columns
            .as_ref()
            .is_some_and(|cs| cs.iter().any(|c| c.name == name))
            && self.column(name).is_some_and(|c| c.is_source());
        if is_source {
            let spelled = self.spelled.get(name).map_or("", String::as_str);
            return matches!(spelled, "TIMESTAMP WITH TIME ZONE" | "TIMESTAMPTZ");
        }
        false
    }

    /// `PostgresTable._postgres_type(column)`: PostgreSQL's own cast
    /// spelling of a source column.
    pub fn postgres_type(&self, column: &str) -> Option<String> {
        self.spelled.get(column).cloned()
    }

    /// `SnowTable._source_snowflake_type_dict`.
    pub fn snowflake_types(&self) -> BTreeMap<String, String> {
        self.spelled.clone()
    }

    /// Snowflake's `_listed`, where it was read already: what `SHOW TABLES`
    /// listed of the tables named like this one.
    pub fn listed_rows(&self) -> Option<&Vec<Listed>> {
        match &self.listed {
            Some(Some(rows)) => Some(rows),
            _ => None,
        }
    }

    /// Snowflake's `_listed`, read now where it was not.
    async fn snow_listed(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<Listed>> {
        if let Some(Some(rows)) = &self.listed {
            return Ok(rows.clone());
        }
        let stmt = sdm_dialects::snow::tables_like(
            self.place.catalog.as_deref().unwrap_or_default(),
            self.place.schema.as_deref().unwrap_or_default(),
            &self.raw_source_name,
        );
        let table = ctx
            .db
            .run(&stmt)
            .await
            .map_err(|e| PyExc::new("ProgrammingError", e.message))?;
        let fields: Vec<String> = table
            .column_names()
            .iter()
            .map(|f| f.to_lowercase())
            .collect();
        let rows: Vec<Listed> = table
            .rows()
            .into_iter()
            .map(|r| fields.iter().cloned().zip(r).collect())
            .collect();
        self.listed = Some(Some(rows.clone()));
        Ok(rows)
    }

    /// Snowflake's `_named()`: the listing of this table, named exactly or
    /// else alike but for case.
    async fn snow_named(&mut self, ctx: &mut Ctx<'_>) -> ER<Vec<Listed>> {
        let rows = self.snow_listed(ctx).await?;
        let get = |row: &Listed, f: &str| row.iter().find(|(k, _)| k == f).map(|(_, v)| v.clone());
        let exact: Vec<Listed> = rows
            .iter()
            .filter(|r| matches!(get(r, "name"), Some(PyVal::Str(s)) if s == self.raw_source_name))
            .cloned()
            .collect();
        if !exact.is_empty() {
            return Ok(exact);
        }
        let folded = self.raw_source_name.to_uppercase();
        Ok(rows
            .into_iter()
            .filter(|r| get(r, "name").unwrap_or(PyVal::None).str_().to_uppercase() == folded)
            .collect())
    }

    fn count_statement(&self) -> Option<Statement> {
        let source = self.quoted_source_name();
        match self.backend {
            Backend::Duckdb => Some(sdm_dialects::duckdb::count_rows(&source)),
            Backend::Postgres => Some(sdm_dialects::postgres::count_rows(&source)),
            Backend::Databricks => Some(sdm_dialects::databricks::count_rows(&source)),
            Backend::Sqlite | Backend::Snow => None,
        }
    }

    /// `_get_num_rows()`: the table's rows, as each backend counts them;
    /// `None` on SQLite, which does not.
    pub async fn num_rows(&mut self, ctx: &mut Ctx<'_>) -> ER<Option<i64>> {
        if self.backend == Backend::Snow {
            let named = self.snow_named(ctx).await?;
            if named.is_empty() {
                return Err(PyExc::new(
                    "RuntimeError",
                    format!(
                        "Could not count the rows of '{}': SHOW TABLES does not list it",
                        self.source_name()
                    ),
                ));
            }
            if named.len() > 1 {
                let mut names: Vec<String> = named
                    .iter()
                    .map(|r| {
                        r.iter()
                            .find(|(k, _)| k == "name")
                            .map_or_else(|| "None".to_string(), |(_, v)| v.str_())
                    })
                    .collect();
                names.sort();
                return Err(PyExc::new(
                    "RuntimeError",
                    format!(
                        "Could not count the rows of '{}': it names each of the tables {}. Spell \
                         the name exactly.",
                        self.source_name(),
                        crate::py::str_list_repr(&names)
                    ),
                ));
            }
            let rows = named[0]
                .iter()
                .find(|(k, _)| k == "rows")
                .map(|(_, v)| v.clone());
            return match rows {
                None | Some(PyVal::None) => Ok(None),
                Some(v) => {
                    let n = int_of(&v);
                    if n == Some(0) {
                        return Err(self.empty_table());
                    }
                    Ok(n)
                }
            };
        }
        let Some(stmt) = self.count_statement() else {
            return Ok(None);
        };
        let found = rows(ctx.db, self.backend, &stmt).await?;
        let n = found
            .first()
            .and_then(|r| r.first())
            .and_then(int_of)
            .ok_or_else(|| PyExc::new("IndexError", "list index out of range"))?;
        if n == 0 {
            return Err(self.empty_table());
        }
        Ok(Some(n))
    }

    fn empty_table(&self) -> PyExc {
        super::empty_table(&self.source_name())
    }

    /// The text of a listed field, as the listing holds it.
    pub fn listed_text(value: &PyVal) -> String {
        text_of(value)
    }
}

impl EngineTable {
    /// `_SQL_TEXT_TYPE` and `_SQL_CHR_FUNCTION` of the table's backend.
    fn sql_text_type(&self) -> (&'static str, &'static str) {
        match self.backend {
            Backend::Duckdb | Backend::Snow => ("VARCHAR", "CHR"),
            Backend::Postgres => ("TEXT", "CHR"),
            Backend::Databricks => ("STRING", "CHR"),
            Backend::Sqlite => ("TEXT", "CHAR"),
        }
    }

    /// `_set_composite_primary_key(names)`: the key's columns folded, in
    /// the source, into one derived identity column the table keys on.
    pub async fn set_composite_primary_key(
        &mut self,
        ctx: &mut Ctx<'_>,
        names: &[String],
    ) -> ER<()> {
        let missing: Vec<String> = names
            .iter()
            .filter(|n| !self.has_column(n))
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(PyExc::new(
                "ValueError",
                format!(
                    "Cannot use {} as a composite primary key of table '{}': column(s) {} are \
                     not present",
                    crate::py::str_list_repr(names),
                    self.name,
                    crate::py::str_list_repr(&missing)
                ),
            ));
        }
        for name in names {
            if Some(name.as_str()) == self.time_column.as_deref()
                || Some(name.as_str()) == self.end_time_column.as_deref()
            {
                return Err(PyExc::new(
                    "ValueError",
                    format!(
                        "Cannot use column '{name}' in a composite primary key since it is \
                         already defined to be a time column"
                    ),
                ));
            }
        }
        for name in names {
            let dtype = self.column(name).map(|c| c.dtype);
            if let Some(dtype) = dtype
                && matches!(
                    dtype,
                    sdm_pql::types::Dtype::Float
                        | sdm_pql::types::Dtype::Float32
                        | sdm_pql::types::Dtype::Float64
                        | sdm_pql::types::Dtype::Floatlist
                )
            {
                return Err(PyExc::new(
                    "ValueError",
                    format!(
                        "composite key column '{name}' is {}; a floating-point column cannot \
                         be part of a row's identity because its text is not reproducible. Use \
                         an exact column, or add a column holding the identity you intend.",
                        dtype.value()
                    ),
                ));
            }
        }
        let encoded = crate::composite::encode_identity(names)
            .map_err(|message| PyExc::new("ValueError", message))?;
        let derived = format!("{}{encoded}", crate::composite::DERIVED_PREFIX);
        // `_materialize_derived_key`: the fold, as an expression column.
        let (text_type, chr) = self.sql_text_type();
        let expression =
            crate::composite::sql_expression(names, |n| self.quote_key_column(n), text_type, chr);
        if !self.has_column(&derived) {
            self.add_columns(
                ctx,
                vec![super::ColumnSpec {
                    name: derived.clone(),
                    expr: Some(expression),
                    dtype: None,
                    stype: None,
                }],
            )
            .await?;
        }
        self.set_column_stype(&derived, sdm_pql::types::Stype::Id)?;
        self.assert_warehouse_folds_like_pandas(ctx, &derived, names)
            .await?;
        self.set_primary_key(Some(&derived))?;
        self.composite_key = Some(names.to_vec());
        Ok(())
    }

    /// `_assert_warehouse_folds_like_pandas(derived, names)`: the source's
    /// fold of the sample, compared with this client's.
    async fn assert_warehouse_folds_like_pandas(
        &mut self,
        ctx: &mut Ctx<'_>,
        derived: &str,
        names: &[String],
    ) -> ER<()> {
        let source = self.source_sample(ctx).await?;
        let folded = match source.column(derived) {
            Some(c) => Some(c.cells.clone()),
            None => self.expr_sample.column(derived).map(|c| c.cells.clone()),
        };
        let Some(folded) = folded else {
            return Ok(());
        };
        if names.iter().any(|n| !source.has(n)) {
            return Ok(());
        }
        let warehouse: std::collections::BTreeSet<String> = folded
            .iter()
            .filter(|v| !crate::frame::missing(v))
            .map(text_of)
            .collect();
        let frame: Vec<(String, Vec<PyVal>)> = source
            .columns
            .iter()
            .map(|c| (c.name.clone(), c.cells.clone()))
            .collect();
        let locally: std::collections::BTreeSet<String> =
            crate::composite::encode_frame(&frame, names)
                .map_err(|message| PyExc::new("ValueError", message))?
                .into_iter()
                .collect();
        if warehouse.is_empty() || warehouse == locally {
            return Ok(());
        }
        let differing: Vec<String> = warehouse.difference(&locally).take(1).cloned().collect();
        Err(PyExc::new(
            "ValueError",
            format!(
                "'{}' folds {} into an identity differently in the warehouse than this client \
                 does, so a prediction seeded here would match no row: the warehouse produced \
                 {}. Add a column holding the identity you intend and declare that instead.",
                self.name,
                crate::py::str_list_repr(names),
                crate::py::str_list_repr(&differing)
            ),
        ))
    }
}
