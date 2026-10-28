// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native source conversion consumes each reached live column declaration.
//! Rust owns meaning, integrality admission and the complete fixed time read;
//! actual native boxing/cast effects and dtype owners remain in the host.

use super::*;
use sdm_contract::data::strings::StringData;
use sdm_contract::ops::{HeldSchema, HeldTable};

use crate::frame_materialize::native_projection;

pub(super) fn consistent_native(
    description: &mut SourceDescription,
    types: &sdm_schema::assemble::HostTypes,
) -> Result<(), CoreError> {
    let mut projected = native_projection(description);
    consistent(&mut projected, types)?;
    for (actual, marked) in description.tables.iter_mut().zip(projected.tables) {
        for (column, marked) in actual.columns.iter_mut().zip(marked.columns) {
            column.stype = marked.stype;
        }
    }
    Ok(())
}

pub(super) fn kept_native(description: &SourceDescription) -> Result<(), CoreError> {
    kept_checks(&native_projection(description))
}

pub(super) fn source(
    mut schema: sdm_contract::ops::HeldSchema,
    typed_tables: Vec<FrameTableMeaning>,
) -> Result<FrameSourceMeaning, CoreError> {
    if schema.tables.len() != typed_tables.len() {
        return Err(protocol(
            "post-cast source declarations lack their typed tables",
        ));
    }
    for (live, typed) in schema.tables.iter_mut().zip(typed_tables) {
        let constructed = typed.constructed;
        let typed = typed.table;
        if typed
            .columns
            .iter()
            .map(|column| &column.name)
            .ne(constructed.iter().map(|column| &column.name))
            || constructed
                .iter()
                .map(|column| &column.name)
                .collect::<BTreeSet<_>>()
                .len()
                != constructed.len()
            || constructed
                .iter()
                .any(|column| !sdm_schema::frames::native_column::canonical(column.dtype))
        {
            return Err(protocol(
                "native constructed columns differ from their complete table meaning",
            ));
        }
        if live.name != typed.name
            || live.source != typed.source
            || live.columns.len() != typed.columns.len()
        {
            return Err(protocol(
                "post-cast source declaration differs from its typed table",
            ));
        }
        for (column, frozen) in live.columns.iter_mut().zip(typed.columns) {
            if column.name != frozen.name {
                return Err(protocol(
                    "post-cast source declaration differs from its typed column",
                ));
            }
            // Native casts have already consumed the earlier declaration.
            // Key/time/stype remain the live graph facts observed only after
            // all casts, while a later dtype edit cannot recast typed rows.
            let fixed = constructed
                .iter()
                .find(|fixed| fixed.name == column.name)
                .ok_or_else(|| protocol("native source lacks its fixed constructed column"))?;
            column.dtype = fixed.dtype.wire_name().into();
        }
    }
    let read = sdm_schema::held::read(schema.clone(), &BTreeMap::new(), true)?;
    let mut description = native_projection(&read.description);
    sdm_schema::assemble::mark_keys(&mut description);
    sdm_schema::assemble::check_link_structure(&description, &read.types)?;
    reserved(&description)?;
    kept_checks(&description).map_err(super::source_keys::metadata_error)?;
    Ok(FrameSourceMeaning { schema })
}

pub(super) fn table(input: FrameTableInputs) -> Result<FrameTableMeaning, CoreError> {
    if input.empty {
        return Err(super::source_keys::native(
            "empty_frame",
            &input.table.name,
            "",
        ));
    }
    let mut names = BTreeSet::new();
    for column in &input.native {
        if !names.insert(&column.name) {
            return Err(protocol(
                "native table type facts repeat a physical source column",
            ));
        }
        if let Some(declared) = &column.declared {
            crate::frame_materialize::values::check_declaration(declared)?;
        }
        if let Some(FramePhysicalDeclaration::Physical { physical }) = &column.declared
            && column.values
                != (FrameConstructorValues::Typed {
                    physical: physical.clone(),
                })
        {
            return Err(protocol(
                "fixed native declaration contradicts its constructed value storage",
            ));
        }
        if column.declared == Some(FramePhysicalDeclaration::LargeUtf8) {
            return Err(crate::frame_materialize::values::unsupported_large_string(
                &input.table.name,
                &column.name,
            ));
        }
    }
    let constructed = input
        .native
        .iter()
        .map(|column| {
            if column.meaning.name != column.name
                || column.meaning.dtype != super::source_keys::constructed(&column.values)?
                || !crate::edit::supports(column.meaning.stype, column.meaning.dtype)
            {
                return Err(protocol(
                    "constructed native meaning contradicts its observed type",
                ));
            }
            Ok(column.meaning.clone())
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    Ok(FrameTableMeaning {
        table: input.table,
        constructed,
    })
}

pub(super) fn meaning(input: FrameColumnInputs) -> FrameColumnMeaning {
    let facts = &input.facts;
    let text = (facts.object || facts.string) && facts.inferred == Some(FrameObservedContent::Text);
    let decimal = facts.arrow_decimal
        || (facts.object && facts.inferred == Some(FrameObservedContent::Decimal));
    let declared = &input.column.dtype;
    let kind = if input.designated_time && text {
        FrameColumnKind::TextTime
    } else if decimal {
        if declared.starts_with("float") {
            FrameColumnKind::DecimalFloat
        } else if declared.starts_with("int") {
            FrameColumnKind::DecimalInteger
        } else {
            FrameColumnKind::DecimalString
        }
    } else if facts.object
        && declared.starts_with("timestamp")
        && facts.inferred == Some(FrameObservedContent::Temporal)
    {
        FrameColumnKind::NativeTime
    } else if facts.timestamp
        && (input.column.stype == Stype::Timestamp
            || sdm_schema::held::wire_dtype(declared) == sdm_contract::ops::Dtype::Time)
    {
        FrameColumnKind::TypedTime
    } else if text
        && (input.column.stype == Stype::Timestamp
            || matches!(
                sdm_schema::held::wire_dtype(declared),
                sdm_contract::ops::Dtype::Date | sdm_contract::ops::Dtype::Time
            ))
    {
        FrameColumnKind::TextFeatureTime
    } else {
        FrameColumnKind::Original
    };
    FrameColumnMeaning { input, kind }
}

fn checked(value: FrameColumnMeaning) -> Result<FrameColumnMeaning, CoreError> {
    if meaning(value.input.clone()) != value {
        return Err(protocol(
            "frame source column meaning no longer matches its live declaration and native facts",
        ));
    }
    Ok(value)
}

pub(super) fn integer(input: FrameIntegerColumn) -> Result<FrameIntegerColumn, CoreError> {
    let meaning = checked(input.meaning.clone())?;
    if meaning.kind != FrameColumnKind::DecimalInteger
        || input.fractional != input.example.is_some()
    {
        return Err(protocol(
            "native decimal integrality facts do not match their source meaning",
        ));
    }
    if let Some(example) = &input.example {
        let mut error = CoreError::new(
            Sentence::FrameDecimalFraction,
            "a native Decimal column contains a fractional value under an integer declaration",
        )
        .with_field(
            "where",
            Detail::text(format!(
                "{}.{}",
                meaning.input.table, meaning.input.column.name
            )),
        )
        .with_field("dtype", Detail::text(meaning.input.column.dtype))
        .with_field("example", Detail::text(example));
        error.public_details = Some(Box::default());
        return Err(error);
    }
    Ok(input)
}

pub(super) fn time(
    asker: &Asker,
    value: FrameColumnMeaning,
    values: StringData,
    facts: &HostFacts,
    attachments: &mut Attachments,
) -> Result<TypedFrameSourceColumn, CoreError> {
    let value = checked(value)?;
    if !matches!(
        value.kind,
        FrameColumnKind::TextTime | FrameColumnKind::TextFeatureTime
    ) {
        return Err(protocol(
            "a native text-time source column lacks resolved time meaning",
        ));
    }
    let source = &value.input;
    let held = HeldSchema {
        tables: vec![HeldTable {
            name: source.table.clone(),
            source: source.source.clone(),
            columns: vec![source.column.clone()],
            primary_key: vec![],
            frame_key_form: None,
            time_column: source.designated_time.then(|| source.column.name.clone()),
            duplicate_keys: None,
        }],
        links: vec![],
    };
    let read = sdm_schema::held::read(held, &BTreeMap::new(), true)?;
    let table = &read.description.tables[0];
    let array = take_strings(attachments, &values)?;
    let rows = array.len();
    let converted = crate::frame_materialize::time_columns::convert(
        table,
        &table.columns[0],
        array,
        Some(values.encoding),
        crate::describe::text_reader(facts)?,
    )?
    .ok_or_else(|| {
        protocol("a native text-time source did not produce its complete typed column")
    })?;
    if converted.unread > 0 {
        asker.warn(CoreWarning {
            fields: [
                ("table".to_owned(), Detail::text(&source.table)),
                ("column".to_owned(), Detail::text(&source.column.name)),
                (
                    "unread".to_owned(),
                    Detail::Scalar(Scalar::UInt64(converted.unread)),
                ),
                (
                    "rows".to_owned(),
                    Detail::Scalar(Scalar::UInt64(u64::try_from(rows).unwrap_or(u64::MAX))),
                ),
                ("examples".to_owned(), Detail::List(converted.examples)),
            ]
            .into(),
            ..CoreWarning::new(
                Sentence::TimeValuesUnread,
                "a time column's values were read as no time and dropped",
            )
        });
    }
    Ok(TypedFrameSourceColumn {
        values: attach_array(asker, &source.column.name, converted.values)?,
        strings: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sdm_contract::ops::HeldColumn;

    fn typed(table: HeldTable) -> FrameTableMeaning {
        let constructed = table
            .columns
            .iter()
            .map(|column| FrameConstructedColumn {
                name: column.name.clone(),
                dtype: sdm_schema::held::wire_dtype(&column.dtype),
                stype: column.stype,
            })
            .collect();
        FrameTableMeaning { table, constructed }
    }

    fn declared(dtype: &str) -> FrameColumnInputs {
        FrameColumnInputs {
            table: "items".to_owned(),
            source: sdm_contract::data::Relation::named("items"),
            column: HeldColumn {
                name: "id".to_owned(),
                untyped: false,
                stype_declared: false,
                dtype: dtype.to_owned(),
                stype: Stype::Id,
                stype_before_key: None,
                source: None,
            },
            designated_time: false,
            facts: FrameColumnFacts {
                object: true,
                string: false,
                arrow_decimal: false,
                timestamp: false,
                inferred: Some(FrameObservedContent::Decimal),
            },
        }
    }

    #[test]
    fn graph_meaning_uses_post_cast_logical_metadata_and_cast_time_dtypes() {
        let mut key = declared("int64").column;
        key.name = "id".into();
        let mut feature = declared("int64").column;
        feature.name = "feature".into();
        feature.stype = Stype::Numerical;
        let mut time = declared("timestamp[ns]").column;
        time.name = "when".into();
        time.stype = Stype::Timestamp;
        let table = HeldTable {
            name: "items".into(),
            source: sdm_contract::data::Relation::named("items"),
            columns: vec![key, feature, time],
            primary_key: vec!["id".into()],
            frame_key_form: None,
            time_column: None,
            duplicate_keys: None,
        };
        let typed = typed(table.clone());
        let mut live = table;
        live.columns[1].stype = Stype::Categorical;
        live.columns[1].dtype = "string".into();
        live.time_column = Some("when".into());
        let resolved = source(
            HeldSchema {
                tables: vec![live],
                links: vec![],
            },
            vec![typed],
        )
        .unwrap()
        .schema;
        assert_eq!(resolved.tables[0].columns[1].dtype, "int64");
        assert_eq!(resolved.tables[0].columns[1].stype, Stype::Categorical);
        assert_eq!(resolved.tables[0].time_column.as_deref(), Some("when"));
    }

    #[test]
    fn each_native_decimal_column_resolves_its_current_declaration() {
        let first = meaning(declared("float64"));
        let mut later = declared("int64");
        later.column.dtype = "string".to_owned();
        assert_eq!(first.kind, FrameColumnKind::DecimalFloat);
        assert_eq!(meaning(later).kind, FrameColumnKind::DecimalString);
        assert_eq!(
            meaning(declared("int64")).kind,
            FrameColumnKind::DecimalInteger
        );
    }

    #[test]
    fn actual_typed_timestamps_retain_their_native_sanitization_boundary() {
        let mut input = declared("timestamp[us]");
        input.column.stype = Stype::Timestamp;
        input.facts = FrameColumnFacts {
            object: false,
            string: false,
            arrow_decimal: false,
            timestamp: true,
            inferred: None,
        };
        assert_eq!(meaning(input).kind, FrameColumnKind::TypedTime);
    }

    #[test]
    fn fractional_decimal_refusal_precedes_whole_native_cast() {
        let input = FrameIntegerColumn {
            meaning: meaning(declared("int64")),
            fractional: true,
            example: Some("Decimal('0.70')".to_owned()),
        };
        let error = integer(input).unwrap_err();
        assert_eq!(error.sentence, Sentence::FrameDecimalFraction);
        assert_eq!(error.public_details.as_deref(), Some(&BTreeMap::new()));
        let good = FrameIntegerColumn {
            meaning: meaning(declared("int64")),
            fractional: false,
            example: None,
        };
        assert_eq!(integer(good.clone()).unwrap(), good);
    }

    #[test]
    fn forged_decimal_meaning_does_not_admit_a_different_native_cast() {
        let mut input = FrameIntegerColumn {
            meaning: meaning(declared("int64")),
            fractional: false,
            example: None,
        };
        input.meaning.input.column.dtype = "float64".to_owned();
        assert!(integer(input).is_err());
    }

    #[test]
    fn closed_observed_contents_preserve_the_existing_conversion_branch_order() {
        for (observed, designated, declared_type, expected) in [
            (
                Some(FrameObservedContent::Text),
                true,
                "string",
                FrameColumnKind::TextTime,
            ),
            (
                Some(FrameObservedContent::Decimal),
                false,
                "float64",
                FrameColumnKind::DecimalFloat,
            ),
            (
                Some(FrameObservedContent::Decimal),
                false,
                "int64",
                FrameColumnKind::DecimalInteger,
            ),
            (
                Some(FrameObservedContent::Decimal),
                false,
                "string",
                FrameColumnKind::DecimalString,
            ),
            (
                Some(FrameObservedContent::Temporal),
                false,
                "timestamp[us]",
                FrameColumnKind::NativeTime,
            ),
            (
                Some(FrameObservedContent::Other),
                true,
                "timestamp[us]",
                FrameColumnKind::Original,
            ),
            (None, true, "timestamp[us]", FrameColumnKind::Original),
            (
                Some(FrameObservedContent::Temporal),
                false,
                "string",
                FrameColumnKind::Original,
            ),
        ] {
            let mut input = declared(declared_type);
            input.designated_time = designated;
            input.facts.inferred = observed;
            assert_eq!(meaning(input).kind, expected);
        }
        let mut input = declared("timestamp[us]");
        input.facts.inferred = Some(FrameObservedContent::Text);
        assert_eq!(
            meaning(input.clone()).kind,
            FrameColumnKind::TextFeatureTime
        );
        // A typed Decimal observation wins even when native inference was not
        // performed; typed time remains separate from boxed temporal content.
        input.facts.object = false;
        input.facts.arrow_decimal = true;
        input.facts.inferred = None;
        assert_eq!(meaning(input).kind, FrameColumnKind::DecimalString);
    }

    #[test]
    fn fixed_large_utf8_table_refusal_is_neutral_and_names_its_reached_column() {
        let input = FrameTableInputs {
            empty: false,
            table: HeldTable {
                name: "items".into(),
                source: "input_items".into(),
                columns: vec![declared("string").column],
                primary_key: vec![],
                frame_key_form: None,
                time_column: None,
                duplicate_keys: None,
            },
            native: vec![FrameTableColumnFacts {
                name: "id".into(),
                declared: Some(FramePhysicalDeclaration::LargeUtf8),
                values: FrameConstructorValues::Typed {
                    physical: sdm_contract::ColumnType::Utf8,
                },
                meaning: FrameConstructedColumn {
                    name: "id".into(),
                    dtype: Dtype::String,
                    stype: Stype::Id,
                },
            }],
        };
        let error = table(input).unwrap_err();
        assert_eq!(error.sentence, Sentence::FrameNativeInternal);
        assert_eq!(
            error.fields["native_declaration_reason"],
            Detail::text("fixed_large_utf8")
        );
        assert_eq!(error.fields["table"], Detail::text("items"));
        assert_eq!(error.fields["column"], Detail::text("id"));
        assert!(!error.fields.contains_key("native_cause_message"));
        assert!(!error.fields.contains_key("message_points"));
        assert_eq!(error.public_details.as_deref(), Some(&BTreeMap::new()));
    }

    #[test]
    fn post_cast_source_uses_existing_native_edge_kind_refusal() {
        let mut key = declared("float64").column;
        key.name = "item_id".into();
        let items = HeldTable {
            name: "items".into(),
            source: "items".into(),
            columns: vec![key],
            primary_key: vec!["item_id".into()],
            frame_key_form: None,
            time_column: None,
            duplicate_keys: None,
        };
        let mut foreign = declared("string").column;
        foreign.name = "item_id".into();
        let buys = HeldTable {
            name: "buys".into(),
            source: "buys".into(),
            columns: vec![foreign],
            primary_key: vec![],
            frame_key_form: None,
            time_column: None,
            duplicate_keys: None,
        };
        let tables = vec![items, buys];
        let error = source(
            HeldSchema {
                tables: tables.clone(),
                links: vec![vec![
                    ColumnRef {
                        table: "buys".into(),
                        column: "item_id".into(),
                    },
                    ColumnRef {
                        table: "items".into(),
                        column: "item_id".into(),
                    },
                ]],
            },
            tables.into_iter().map(typed).collect(),
        )
        .unwrap_err();
        assert_eq!(error.sentence, Sentence::EdgeKeyTypes);
        assert_eq!(error.fields["source_dtype"], Detail::text("string"));
        assert_eq!(error.fields["target_dtype"], Detail::text("float"));
        assert_eq!(error.fields["source_table"], Detail::text("buys"));
        assert_eq!(error.fields["target_table"], Detail::text("items"));
    }
}
