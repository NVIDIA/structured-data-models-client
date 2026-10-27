// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native source conversion consumes each reached live column declaration.
//! Rust owns meaning, integrality admission and the complete fixed time read;
//! actual native boxing/cast effects and dtype owners remain in the host.

use super::*;
use sdm_contract::data::strings::StringData;
use sdm_contract::ops::{HeldSchema, HeldTable};

pub(super) fn meaning(input: FrameColumnInputs) -> FrameColumnMeaning {
    let facts = &input.facts;
    let text = (facts.object || facts.string) && facts.inferred.as_deref() == Some("string");
    let decimal =
        facts.arrow_decimal || (facts.object && facts.inferred.as_deref() == Some("decimal"));
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
        && matches!(
            facts.inferred.as_deref(),
            Some("datetime" | "datetime64" | "date")
        )
    {
        FrameColumnKind::NativeTime
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
        error.public_details = Some(BTreeMap::new());
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
                (
                    "examples".to_owned(),
                    Detail::List(converted.examples.into_iter().map(Detail::Scalar).collect()),
                ),
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
                inferred: Some("decimal".to_owned()),
            },
        }
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
    fn fractional_decimal_refusal_precedes_whole_native_cast() {
        let input = FrameIntegerColumn {
            meaning: meaning(declared("int64")),
            fractional: true,
            example: Some("Decimal('0.70')".to_owned()),
        };
        let error = integer(input).unwrap_err();
        assert_eq!(error.sentence, Sentence::FrameDecimalFraction);
        assert_eq!(error.public_details, Some(BTreeMap::new()));
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
}
