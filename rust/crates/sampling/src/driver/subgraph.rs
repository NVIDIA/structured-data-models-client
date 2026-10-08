// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Sampler.sample_subgraph`: the columns each table is read with, the
//! backend's traversal, and its output shaped into a `SampledSubgraph`:
//! end times past the anchor cleared, text normalized, and each link stored
//! as the reverse of another (`REV`), as pairs (`COO`) or compressed
//! (`CSC`).

use std::sync::Arc;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::{Array, TimestampNanosecondArray};
use arrow_schema::DataType;
use sdm_schema::dt::Host;
use sdm_schema::pyval::PyVal;
use serde_json::Value;

use super::inputs::{Backend, Inputs, int};
use super::output::{SampledLink, SampledTable, SubgraphDone};
use super::sampler::Sampler;
use super::sql::SamplerOutput;
use super::target::{Columns, seed, times_of};
use super::{Db, Raised};
use crate::frame::{Dtype, Frame, Series, texts};
use crate::graph_store::rev_edge_type;

fn string_list(value: &Value, what: &str) -> Result<Vec<String>, Raised> {
    value
        .as_array()
        .ok_or_else(|| Raised::value_error(format!("{what} is not a list")))?
        .iter()
        .map(|v| super::inputs::text(v, what))
        .collect()
}

/// The anchor times a call is given, as `to_naive_utc` then
/// `astype('datetime64[ns]')` make them.
pub(crate) fn anchor_times(series: &Series) -> Result<Vec<Option<i64>>, Raised> {
    if let Some(values) = crate::frame::ints(series.array.as_ref()) {
        return Ok(values);
    }
    times_of(series)
}

/// `_normalize_text(series)`: each text as a list of its lower-case words.
fn normalize(series: &Series) -> Result<Series, Raised> {
    if series.is_empty() {
        return Ok(series.clone());
    }
    let objects = series.object_values().or_else(|| {
        matches!(
            series.array.data_type(),
            DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..)
        )
        .then(|| {
            (0..series.len())
                .map(|position| sdm_schema::engine::result::value_at(&series.array, position))
                .collect()
        })
    });
    if objects.as_ref().is_some_and(|values| {
        matches!(
            values.first(),
            Some(PyVal::List(_) | PyVal::Tuple(_) | PyVal::NdArray(..))
        )
    }) {
        return Ok(series.clone());
    }
    let values = objects
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    if value.is_missing() {
                        None
                    } else {
                        Some(value.str_())
                    }
                })
                .collect()
        })
        .or_else(|| texts(series.array.as_ref()))
        .ok_or_else(|| {
            Raised::new(
                "TypeError",
                format!(
                    "text of Arrow type {} is not normalized by the core",
                    series.array.data_type()
                ),
            )
        })?;
    let words = crate::text::normalize_text(&values, Some(50));
    let mut builder = ListBuilder::new(StringBuilder::new());
    for list in words {
        for word in list {
            builder.values().append_value(word);
        }
        builder.append(true);
    }
    Ok(Series::new(Dtype::Object, Arc::new(builder.finish())))
}

/// `Sampler.sample_subgraph`.
pub async fn sample_subgraph(inputs: Inputs, db: Db, host: Host) -> Result<SubgraphDone, Raised> {
    let mut sampler = Sampler::new(&inputs, host)?;
    let get = |name: &str| inputs.arguments.get(name).unwrap_or(&Value::Null);
    let entity_table_names = string_list(get("entity_table_names"), "entity_table_names")?;
    let entity_pkey = inputs.series_argument("entity_pkey", "entity_pkey_dtype")?;
    let anchor_time: Option<Vec<Option<i64>>> = match get("anchor_time") {
        Value::String(s) if s == "entity" => None,
        _ => Some(anchor_times(
            &inputs.series_argument("anchor_time", "anchor_time_dtype")?,
        )?),
    };
    let num_neighbors: Vec<i64> = get("num_neighbors")
        .as_array()
        .ok_or_else(|| Raised::value_error("num_neighbors is not a list"))?
        .iter()
        .map(|v| int(v, "num_neighbors"))
        .collect::<Result<_, _>>()?;
    let random_seed = seed(inputs.arguments.get("random_seed"))?;
    let scored: Option<Vec<i64>> = match get("scored") {
        Value::Null => None,
        v => Some(
            v.as_array()
                .ok_or_else(|| Raised::value_error("scored is not a list"))?
                .iter()
                .map(|i| int(i, "scored"))
                .collect::<Result<_, _>>()?,
        ),
    };

    sample_subgraph_with(
        &mut sampler,
        &db,
        &entity_table_names,
        &entity_pkey,
        anchor_time.as_deref(),
        &num_neighbors,
        get("exclude_cols_dict"),
        random_seed.as_ref(),
        scored.as_deref(),
    )
    .await
}

/// `Sampler.sample_subgraph` of a sampler and arguments already read. What
/// it raises carries the warnings reading the rows gave before it raised.
#[allow(clippy::too_many_arguments)]
pub async fn sample_subgraph_with(
    sampler: &mut Sampler,
    db: &Db,
    entity_table_names: &[String],
    entity_pkey: &Series,
    anchor_time: Option<&[Option<i64>]>,
    num_neighbors: &[i64],
    exclude_cols_dict: &Value,
    random_seed: Option<&sdm_dialects::int::Int>,
    scored: Option<&[i64]>,
) -> Result<SubgraphDone, Raised> {
    let sampled = sample_and_shape(
        sampler,
        db,
        entity_table_names,
        entity_pkey,
        anchor_time,
        num_neighbors,
        exclude_cols_dict,
        random_seed,
        scored,
    )
    .await;
    match sampled {
        Ok(done) => Ok(done),
        Err(raised) => Err(raised.after(sampler.take_warnings())),
    }
}

#[allow(clippy::too_many_arguments)]
async fn sample_and_shape(
    sampler: &mut Sampler,
    db: &Db,
    entity_table_names: &[String],
    entity_pkey: &Series,
    anchor_time: Option<&[Option<i64>]>,
    num_neighbors: &[i64],
    exclude_cols_dict: &Value,
    random_seed: Option<&sdm_dialects::int::Int>,
    scored: Option<&[i64]>,
) -> Result<SubgraphDone, Raised> {
    let entity_table_names = entity_table_names.to_vec();
    // Exclude the columns that leak the target.
    let mut table_stype_dict = sampler.table_stype_dict.clone();
    if let Value::Object(exclude) = exclude_cols_dict {
        for (table, columns) in exclude {
            let Some((_, stypes)) = table_stype_dict.iter_mut().find(|(t, _)| t == table) else {
                let mut names: Vec<&String> =
                    sampler.table_stype_dict.iter().map(|(t, _)| t).collect();
                names.sort();
                return Err(Raised::value_error(format!(
                    "'exclude_cols_dict' names table '{table}', which is not in the graph. \
                     Available tables: {}",
                    sdm_schema::py::str_list_repr(&names)
                )));
            };
            for column in string_list(columns, "an excluded column")? {
                let Some(i) = stypes.iter().position(|(c, _)| *c == column) else {
                    let mut names: Vec<&String> = stypes.iter().map(|(c, _)| c).collect();
                    names.sort();
                    return Err(Raised::value_error(format!(
                        "'exclude_cols_dict' names column '{column}' of table '{table}', which \
                         is not one of its feature columns (primary and foreign keys are never \
                         sent as features). Excludable columns: {}",
                        sdm_schema::py::str_list_repr(&names)
                    )));
                };
                stypes.remove(i);
            }
        }
    }

    // The columns read as features, and each entity table's key.
    let mut columns_dict = Columns(
        table_stype_dict
            .iter()
            .map(|(t, stypes)| (t.clone(), stypes.iter().map(|(c, _)| c.clone()).collect()))
            .collect(),
    );
    for table in &entity_table_names {
        let key = sampler.primary_key(table)?.clone();
        columns_dict.add(table, &key);
    }
    let entity_table = entity_table_names
        .first()
        .ok_or_else(|| Raised::new("IndexError", "tuple index out of range"))?
        .clone();

    let out: SamplerOutput = match sampler.backend {
        Backend::Local => super::local::sample_subgraph(
            sampler,
            &entity_table,
            entity_pkey,
            anchor_time,
            &columns_dict,
            num_neighbors,
            random_seed,
        )?,
        _ => {
            super::sql::sample_subgraph(
                sampler,
                db,
                &entity_table,
                entity_pkey,
                anchor_time,
                &columns_dict,
                num_neighbors,
                scored,
            )
            .await?
        }
    };
    shape(sampler, &entity_table_names, &table_stype_dict, out)
}

/// The sampler's output as a `SampledSubgraph`.
fn shape(
    sampler: &Sampler,
    entity_table_names: &[String],
    table_stype_dict: &[(String, Vec<(String, String)>)],
    out: SamplerOutput,
) -> Result<SubgraphDone, Raised> {
    let mut tables: Vec<SampledTable> = Vec::new();
    for (table, batch) in &out.batch_dict {
        if batch.is_empty() {
            continue;
        }
        let primary_key = if entity_table_names.contains(table) {
            Some(sampler.primary_key(table)?.clone())
        } else {
            None
        };
        let mut df: Frame = out
            .df_dict
            .iter()
            .find(|(t, _)| t == table)
            .map(|(_, f)| f.clone())
            .ok_or_else(|| Raised::key_error(table))?;
        if let Some(end_time_column) = sampler.end_time_column(table) {
            if out.inverse_dict.iter().any(|(t, _)| t == table) {
                return Err(Raised::new("AssertionError", ""));
            }
            let series = df.column(end_time_column)?.clone();
            let times = times_of(&series)?;
            let cleared: Vec<Option<i64>> = times
                .iter()
                .zip(batch)
                .map(|(t, &b)| {
                    let anchor = out.anchor_time.get(b as usize).copied().unwrap_or(i64::MIN);
                    match t {
                        Some(v) if *v > anchor => None,
                        other => *other,
                    }
                })
                .collect();
            df.set(
                end_time_column,
                Series::new(
                    series.dtype.clone(),
                    Arc::new(TimestampNanosecondArray::from(cleared)),
                ),
            );
        }
        let stypes = table_stype_dict
            .iter()
            .find(|(t, _)| t == table)
            .map(|(_, s)| s.clone())
            .ok_or_else(|| Raised::key_error(table))?;
        for (column, stype) in &stypes {
            if stype == "text" {
                let normalized = normalize(df.column(column)?)?;
                df.set(column, normalized);
            }
        }
        let num_sampled_nodes = out
            .num_sampled_nodes_dict
            .iter()
            .find(|(t, _)| t == table)
            .map(|(_, n)| n.clone())
            .ok_or_else(|| Raised::key_error(table))?;
        tables.push(SampledTable {
            name: table.clone(),
            df,
            row: out
                .inverse_dict
                .iter()
                .find(|(t, _)| t == table)
                .map(|(_, r)| r.clone()),
            batch: batch.clone(),
            num_sampled_nodes,
            stype_dict: stypes,
            primary_key,
        });
    }

    let mut links: Vec<SampledLink> = Vec::new();
    for (edge_type, row) in &out.row_dict {
        let col = out
            .col_dict
            .iter()
            .find(|(e, _)| e == edge_type)
            .map(|(_, c)| c.clone())
            .ok_or_else(|| Raised::key_error(&format!("{edge_type:?}")))?;
        if row.is_empty() {
            continue;
        }
        let num_sampled_edges = out
            .num_sampled_edges_dict
            .iter()
            .find(|(e, _)| e == edge_type)
            .map(|(_, n)| n.clone())
            .unwrap_or_default();
        let rev = rev_edge_type(edge_type);
        if links.iter().any(|l| l.edge_type == rev) {
            let rev_col = out.col_dict.iter().find(|(e, _)| *e == rev).map(|(_, c)| c);
            let rev_row = out.row_dict.iter().find(|(e, _)| *e == rev).map(|(_, r)| r);
            if rev_col == Some(row) && rev_row == Some(&col) {
                links.push(SampledLink {
                    edge_type: edge_type.clone(),
                    layout: "REV",
                    row: None,
                    col: None,
                    num_sampled_edges,
                });
                continue;
            }
        }
        let is_arange = |values: &[i64]| values.iter().enumerate().all(|(i, &v)| v == i as i64);
        let mut layout = "COO";
        let row_out = (!is_arange(row)).then(|| row.clone());
        let mut col_out = (!is_arange(&col)).then(|| col.clone());
        let num_cols: i64 = tables
            .iter()
            .find(|t| t.name == edge_type.2)
            .map(|t| t.num_sampled_nodes.iter().sum())
            .ok_or_else(|| Raised::key_error(&edge_type.2))?;
        if let Some(c) = &col_out
            && c.len() as i64 > num_cols + 1
            && c.windows(2).all(|w| w[1] - w[0] >= 0)
        {
            layout = "CSC";
            let mut counts = vec![0i64; num_cols.max(0) as usize];
            for &v in c {
                let v = usize::try_from(v).map_err(|_| {
                    Raised::value_error("'list' argument must have no negative elements")
                })?;
                if v >= counts.len() {
                    return Err(Raised::value_error(
                        "provided out is the wrong size for the reduction",
                    ));
                }
                counts[v] += 1;
            }
            let mut compressed = Vec::with_capacity(counts.len() + 1);
            compressed.push(0);
            let mut total = 0;
            for count in counts {
                total += count;
                compressed.push(total);
            }
            col_out = Some(compressed);
        }
        links.push(SampledLink {
            edge_type: edge_type.clone(),
            layout,
            row: row_out,
            col: col_out,
            num_sampled_edges,
        });
    }

    Ok(SubgraphDone {
        anchor_time: out.anchor_time,
        tables,
        links,
        warnings: sampler.take_warnings(),
    })
}

#[cfg(test)]
mod text_tests {
    use super::*;

    #[test]
    fn normalization_uses_the_first_sampled_cell_and_preserves_nulls() {
        let series = Series::new(
            Dtype::Object,
            crate::frame::encoded_objects(vec![
                r#""Some TEXT!""#.into(),
                r#"{"$tuple":["New York","B!"]}"#.into(),
                "null".into(),
            ]),
        );
        let sampled = series.take(&[1, 2, 1]).unwrap();
        let kept = normalize(&sampled).unwrap();
        assert_eq!(
            crate::frame::object_values(&kept.array),
            crate::frame::object_values(&sampled.array)
        );
        let null_first = normalize(&series.take(&[2, 1]).unwrap()).unwrap();
        assert_eq!(
            sdm_schema::engine::result::value_at(&null_first.array, 0),
            PyVal::List(vec![])
        );
        assert_eq!(
            sdm_schema::engine::result::value_at(&null_first.array, 1),
            PyVal::List(vec![
                PyVal::Str("new".into()),
                PyVal::Str("york".into()),
                PyVal::Str("b".into())
            ])
        );
        let text_first = normalize(&series).unwrap();
        assert_eq!(
            sdm_schema::engine::result::value_at(&text_first.array, 0),
            PyVal::List(vec![PyVal::Str("some".into()), PyVal::Str("text".into())])
        );
    }

    #[test]
    fn a_null_first_list_normalizes_the_remaining_lists() {
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.append(false);
        builder.values().append_value("A!");
        builder.append(true);
        let series = Series::new(Dtype::Object, Arc::new(builder.finish()));
        let normalized = normalize(&series).unwrap();
        assert_eq!(
            sdm_schema::engine::result::value_at(&normalized.array, 0),
            PyVal::List(vec![])
        );
        assert_eq!(
            sdm_schema::engine::result::value_at(&normalized.array, 1),
            PyVal::List(vec![
                PyVal::Str("[".into()),
                PyVal::Str("a".into()),
                PyVal::Str("]".into())
            ])
        );
    }
}
