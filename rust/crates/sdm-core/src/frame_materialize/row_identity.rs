// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned complete row identities for native graph construction and checkpoints.

use super::*;

#[derive(Clone)]
pub(crate) struct OwnedIdentity {
    pub definition: FrameCompositeDefinition,
    pub values: ArrowData,
    pub encoding: StringEncoding,
}

pub(super) fn bind(
    bindings: Vec<FrameRowIdentity>,
    attachments: &mut Attachments,
) -> Result<Vec<OwnedIdentity>, CoreError> {
    bindings
        .into_iter()
        .map(|binding| {
            crate::materialize::frame_values::source_keys::definition(&binding.definition)?;
            let Some(Attachment::Arrow(values)) = attachments.remove(&binding.values.values) else {
                return Err(invalid("native row identity lacks its owned Arrow values"));
            };
            let batch = crate::materialize::one_batch(&values)?;
            if batch.num_columns() != 1 || batch.column(0).null_count() != 0 {
                return Err(invalid(
                    "native row identity is not a complete aligned nonmissing column",
                ));
            }
            let strings = sdm_engine::strings::StringColumn::new(
                batch.column(0).as_ref(),
                binding.values.encoding,
            )
            .map_err(|_| invalid("native row identity encoding differs from its values"))?;
            for row in 0..strings.len() {
                strings
                    .get(row)
                    .map_err(|_| invalid("native row identity contains invalid encoded text"))?;
            }
            Ok(OwnedIdentity {
                definition: binding.definition,
                values,
                encoding: binding.values.encoding,
            })
        })
        .collect()
}

pub(super) fn complete(
    bindings: &mut Vec<OwnedIdentity>,
    tables: &[(Relation, ArrowData)],
    restored: bool,
    description: &SourceDescription,
    graph: &NativeGraph,
    strings: Option<&BTreeMap<String, BTreeMap<String, StringEncoding>>>,
) -> Result<(), CoreError> {
    let mut supplied = BTreeMap::new();
    for identity in std::mem::take(bindings) {
        if supplied
            .insert(identity.definition.table.clone(), identity)
            .is_some()
        {
            return Err(invalid("native source repeats a compound identity binding"));
        }
    }
    let mut complete = Vec::new();
    for table in &description.tables {
        if table.primary_key.len() < 2 {
            continue;
        }
        let (_, data) = tables
            .iter()
            .find(|(source, _)| source == &table.source)
            .ok_or_else(|| invalid("native row identity lacks its complete source"))?;
        let rows = crate::materialize::one_batch(data)?;
        let identity = match supplied.remove(&table.name) {
            Some(identity) => identity,
            None if restored => {
                return Err(invalid(
                    "restored native source lacks its compound identity",
                ));
            }
            None => {
                let encodings = strings
                    .and_then(|strings| strings.get(&table.name))
                    .cloned()
                    .unwrap_or_default();
                let converted =
                    sdm_engine::frame_identity::encode(&rows, &table.primary_key, &encodings)
                        .map_err(|error| error.with_field("table", Detail::text(&table.name)))?;
                let columns = table
                    .primary_key
                    .iter()
                    .map(|name| {
                        let array = rows
                            .column_by_name(name)
                            .ok_or_else(|| invalid("native identity lacks a source component"))?;
                        let dtype = if encodings.contains_key(name)
                            || matches!(array.data_type(), DataType::Utf8 | DataType::LargeUtf8)
                        {
                            Dtype::String
                        } else if array.data_type().is_integer() {
                            Dtype::Int
                        } else if array.data_type() == &DataType::Boolean {
                            Dtype::Bool
                        } else {
                            return Err(invalid(
                                "raw compound identity has an unsupported primitive component",
                            ));
                        };
                        Ok(FrameIdentityComponent {
                            name: name.clone(),
                            dtype,
                        })
                    })
                    .collect::<Result<Vec<_>, CoreError>>()?;
                let definition = FrameCompositeDefinition {
                    table: table.name.clone(),
                    source: table.source.clone(),
                    columns,
                    derived: sdm_engine::graph::folded_name(&table.primary_key),
                    codec: FrameIdentityCodec::EscapedTextV1,
                };
                let batch = RecordBatch::try_from_iter([("identity", converted.values)])
                    .map_err(|_| internal("converted native identity does not align"))?;
                OwnedIdentity {
                    definition,
                    values: ArrowData::new(batch.schema(), vec![batch]),
                    encoding: converted.encoding,
                }
            }
        };
        let identity_rows = crate::materialize::one_batch(&identity.values)?;
        if identity_rows.num_rows() != rows.num_rows() {
            return Err(invalid(
                "native row identity is not aligned to every original source row",
            ));
        }
        let definition = &identity.definition;
        if definition.source != table.source
            || definition.table != table.name
            || definition
                .columns
                .iter()
                .map(|column| &column.name)
                .ne(table.primary_key.iter())
            || definition.derived
                != graph
                    .topology
                    .table(&table.name)
                    .and_then(|table| table.primary_key.as_ref())
                    .cloned()
                    .unwrap_or_default()
        {
            return Err(invalid(
                "native row identity contradicts its complete source declaration",
            ));
        }
        crate::materialize::frame_values::source_keys::definition(definition)?;
        complete.push(identity);
    }
    if !supplied.is_empty() {
        return Err(invalid(
            "native identity bindings name an absent or scalar-keyed table",
        ));
    }
    *bindings = complete;
    Ok(())
}
