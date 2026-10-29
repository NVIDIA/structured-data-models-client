// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Complete native target row maps, independently of native library identity.

use super::*;
use crate::run::Attachments;
use sdm_contract::data::strings::StringEncoding;
use sdm_contract::materialize::{
    NativeConditionColumn, NativeTargetValues, PreparedLabelInputs, PreparedTargetColumn,
    TargetDerivation as Derivation,
};

pub(super) mod conditions;
pub(super) use conditions::operand_columns;

#[derive(Clone)]
pub(super) enum LabelRoot {
    Frame {
        population: super::frame_values::RetainedFramePopulation,
        previous: Option<super::frame_values::RetainedFrameLabelBatch>,
        graph: crate::RetainedFrameGraph,
    },
    Source {
        population: super::source_values::RetainedMaterializePopulation,
        previous: Option<super::source_values::RetainedMaterializeBatch>,
        memo: SamplerMemo,
    },
}

impl LabelRoot {
    pub(super) fn condition_context(&self) -> Result<super::native_conditions::Context, CoreError> {
        match self {
            Self::Frame { population, .. } => population.condition_context(),
            Self::Source { population, .. } => population.condition_context(),
        }
    }

    fn binding(
        &self,
        table: &str,
        column: &str,
    ) -> Result<sdm_contract::frames::materialize::FrameNativeColumnRef, CoreError> {
        match self {
            Self::Frame { graph, .. } if column != "*" => graph.native_binding(&ColumnRef {
                table: table.into(),
                column: column.into(),
            }),
            Self::Frame { graph, .. } => {
                let prepared = graph.prepared();
                let table = prepared
                    .graph
                    .table(table)
                    .ok_or_else(|| invalid_target("native condition source is absent"))?;
                Ok(sdm_contract::frames::materialize::FrameNativeColumnRef {
                    source: table.source.clone(),
                    column: column.into(),
                })
            }
            Self::Source { population, .. } => population.column_binding(table, column),
        }
    }
}

pub(super) struct LabelInputData {
    pub(super) meaning: PreparedLabelInputs,
    pub(super) root: LabelRoot,
    pub(super) inputs: Arc<task::LabelInputs>,
    pub(super) target: Shape,
    pub(super) eligibility: Option<super::native_conditions::Conditions>,
}

#[derive(Clone)]
pub struct RetainedLabelInputs {
    pub(super) data: Arc<LabelInputData>,
}

impl std::fmt::Debug for RetainedLabelInputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedLabelInputs")
            .field("population", &self.data.meaning.population)
            .field("range", &self.data.meaning.range)
            .finish_non_exhaustive()
    }
}

impl RetainedLabelInputs {
    pub(super) fn new(
        meaning: PreparedLabelInputs,
        root: LabelRoot,
        inputs: Arc<task::LabelInputs>,
    ) -> Result<Self, CoreError> {
        let target = inputs.target.clone();
        let input = Self {
            data: Arc::new(LabelInputData {
                meaning,
                root,
                inputs,
                target,
                eligibility: None,
            }),
        };
        let eligibility = super::native_conditions::Conditions::entity(&input)?;
        Ok(input.with_eligibility(eligibility))
    }

    pub(super) fn with_eligibility(
        &self,
        eligibility: super::native_conditions::Conditions,
    ) -> Self {
        Self {
            data: Arc::new(LabelInputData {
                meaning: self.data.meaning.clone(),
                root: self.data.root.clone(),
                inputs: Arc::clone(&self.data.inputs),
                target: self.data.target.clone(),
                eligibility: Some(eligibility),
            }),
        }
    }

    pub(super) fn publish(&self, asker: &Asker) -> Result<PreparedLabelInputs, CoreError> {
        let mut value = self.data.meaning.clone();
        value.eligibility = self
            .data
            .eligibility
            .as_ref()
            .map(|condition| condition.publish(asker))
            .transpose()?;
        value.owner = asker.attach(Attachment::LabelInputs(self.clone()))?;
        Ok(value)
    }

    pub(super) fn view(&self, target: Shape) -> Result<Self, CoreError> {
        use sdm_contract::materialize::{ConditionSourceRole, TargetColumnForm};
        let (table, column, form, source_role, filter) = match &target {
            Shape::Column(column) => {
                let context = self.data.root.condition_context()?;
                (
                    context.entity.table,
                    column.clone(),
                    TargetColumnForm::Condition,
                    ConditionSourceRole::Entity,
                    None,
                )
            }
            Shape::Aggregated(window) => (
                window.table.clone(),
                window.column.clone().unwrap_or_else(|| "*".into()),
                if window.filter.is_some() {
                    TargetColumnForm::Filtered
                } else {
                    TargetColumnForm::Plain
                },
                ConditionSourceRole::Windowed,
                window.filter.as_deref(),
            ),
            _ => {
                return Err(invalid_target(
                    "condition view names no validated column or aggregation",
                ));
            }
        };
        let binding = self.data.root.binding(&table, &column)?;
        let condition_columns = filter
            .map(|filter| {
                operand_columns(filter)?
                    .into_iter()
                    .map(|column| self.data.root.binding(&table, column))
                    .collect::<Result<Vec<_>, CoreError>>()
            })
            .transpose()?
            .unwrap_or_default();
        let source_read = match source_role {
            ConditionSourceRole::Entity => self.data.inputs.population.source_read,
            ConditionSourceRole::Windowed => self
                .data
                .inputs
                .fetched
                .get(&table)
                .and_then(|rows| rows.source_read),
        };
        let meaning = PreparedLabelInputs {
            population: self.data.meaning.population.clone(),
            range: self.data.meaning.range,
            owner: AttachmentId(0),
            source_read,
            target: binding,
            time: match &target {
                Shape::Aggregated(window) => Some(self.data.root.binding(
                    &window.table,
                    &target_source(&self.data.inputs, window)?.time_column,
                )?),
                _ => None,
            },
            form,
            condition_columns,
            source_role,
            eligibility: None,
        };
        Ok(Self {
            data: Arc::new(LabelInputData {
                meaning,
                root: self.data.root.clone(),
                inputs: Arc::clone(&self.data.inputs),
                target,
                eligibility: None,
            }),
        })
    }

    pub(crate) fn carry(&self, asker: &Asker) {
        match &self.data.root {
            LabelRoot::Frame { graph, .. } => asker.keep_retained_frame(graph.clone()),
            LabelRoot::Source { memo, .. } => asker.keep(memo.clone()),
        }
    }

    fn source_rows(&self) -> Result<(RecordBatch, Vec<usize>), CoreError> {
        match &self.data.target {
            Shape::Column(_) => {
                let input = &self.data.inputs;
                let source = input
                    .native_rows
                    .as_ref()
                    .unwrap_or(&input.population.sampled);
                Ok((
                    take_batch(source, &input.entity_rows)?,
                    input.entity_positions.clone(),
                ))
            }
            Shape::Aggregated(window) => {
                let source = target_source(&self.data.inputs, window)?;
                Ok((
                    source.rows.clone(),
                    source
                        .source_positions
                        .clone()
                        .unwrap_or_else(|| (0..source.rows.num_rows()).collect()),
                ))
            }
            _ => Err(invalid_target(
                "native source view is not a column or aggregation",
            )),
        }
    }

    fn column_values(&self, rows: &RecordBatch) -> Result<ArrayRef, CoreError> {
        if self.data.meaning.target.column == "*" {
            Ok(Arc::new(arrow_array::Int64Array::from(vec![
                1;
                rows.num_rows()
            ])))
        } else {
            named(rows, &self.data.meaning.target.column).map(Arc::clone)
        }
    }

    pub fn rows(&self) -> Result<ArrowData, CoreError> {
        let (rows, source_rows) = self.source_rows()?;
        let count = rows.num_rows();
        let positions: ArrayRef = Arc::new(arrow_array::UInt64Array::from_iter_values(
            (0..count).map(|row| row as u64),
        ));
        let source_rows: ArrayRef = Arc::new(arrow_array::UInt64Array::from_iter_values(
            source_rows.into_iter().map(|row| row as u64),
        ));
        let values = self.column_values(&rows)?;
        let mut columns = vec![
            ("position", positions),
            ("source_row", source_rows),
            ("target", values),
        ];
        if let Shape::Aggregated(window) = &self.data.target {
            let source = target_source(&self.data.inputs, window)?;
            columns.push((
                "time",
                Arc::clone(named(&source.rows, &source.time_column)?),
            ));
        }
        let batch = RecordBatch::try_from_iter(columns)
            .map_err(|_| invalid_target("native column view source rows do not align"))?;
        Ok(ArrowData::new(batch.schema(), vec![batch]))
    }

    pub(super) fn bind(
        value: PreparedLabelInputs,
        attachments: &mut Attachments,
    ) -> Result<Self, CoreError> {
        let Some(Attachment::LabelInputs(owner)) = attachments.remove(&value.owner) else {
            return Err(invalid_target(
                "native target input lacks its actual source owner",
            ));
        };
        match (&owner.data.eligibility, value.eligibility.clone()) {
            (Some(condition), Some(value)) => condition.matches(value, attachments)?,
            (None, None) => {}
            _ => {
                return Err(invalid_target(
                    "native input drops or invents its required eligibility data",
                ));
            }
        }
        let mut expected = owner.data.meaning.clone();
        expected.owner = value.owner;
        expected.eligibility = value.eligibility.clone();
        if expected != value {
            return Err(invalid_target(
                "native target input does not match its complete source and range meaning",
            ));
        }
        Ok(owner)
    }
}

#[derive(Clone)]
pub struct RetainedTargetColumn {
    pub(super) input: RetainedLabelInputs,
    pub(super) data: Arc<TargetColumn>,
}

impl std::fmt::Debug for RetainedTargetColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedTargetColumn")
            .field("derivation", &self.data.derivation)
            .field("rows", &self.data.positions.len())
            .finish_non_exhaustive()
    }
}

impl RetainedTargetColumn {
    pub(crate) fn carry(&self, asker: &Asker) {
        self.input.carry(asker);
    }

    pub fn rows(&self) -> Result<ArrowData, CoreError> {
        let positions: ArrayRef = Arc::new(arrow_array::UInt64Array::from_iter_values(
            self.data.positions.iter().map(|&r| r as u64),
        ));
        let batch = RecordBatch::try_from_iter([
            ("position", positions),
            ("target", Arc::clone(&self.data.native_values)),
        ])
        .map_err(|_| invalid_target("completed native target values do not align"))?;
        Ok(ArrowData::new(batch.schema(), vec![batch]))
    }

    pub fn selection(&self) -> Result<ArrowData, CoreError> {
        let values: ArrayRef =
            Arc::new(arrow_array::BooleanArray::from(self.data.selection.clone()));
        let batch = RecordBatch::try_from_iter([("selected", values)])
            .map_err(|_| invalid_target("native target selection does not form a column"))?;
        Ok(ArrowData::new(batch.schema(), vec![batch]))
    }

    pub(super) fn bind(
        value: PreparedTargetColumn,
        attachments: &mut Attachments,
    ) -> Result<Self, CoreError> {
        let input = RetainedLabelInputs::bind(value.input, attachments)?;
        let Some(Attachment::TargetColumn(owner)) = attachments.remove(&value.owner) else {
            return Err(invalid_target(
                "completed target lacks its actual predecessor owner",
            ));
        };
        if !Arc::ptr_eq(&input.data, &owner.input.data)
            || !Arc::ptr_eq(&owner.data.inputs, &input.data.inputs)
            || value.derivation != owner.data.derivation
            || value.rows != owner.data.positions.len() as u64
            || value.condition_values.len() != owner.data.condition_values.len()
            || owner
                .data
                .previous
                .as_ref()
                .is_some_and(|previous| !Arc::ptr_eq(&previous.inputs, &owner.data.inputs))
        {
            return Err(invalid_target(
                "completed target does not match its source, predecessor and complete rows",
            ));
        }
        for (value, expected) in value
            .condition_values
            .into_iter()
            .zip(&owner.data.condition_values)
        {
            let actual = conversion(value.converted, attachments)?;
            if value.column != expected.column
                || actual.encoding != expected.converted.encoding
                || actual.positions != expected.converted.positions
                || actual.presence != expected.converted.presence
                || actual.values.to_data() != expected.converted.values.to_data()
            {
                return Err(invalid_target(
                    "condition evidence no longer matches its actual completed occurrence",
                ));
            }
        }
        Ok(owner)
    }
}

struct ConvertedColumn {
    rows: RecordBatch,
    values: ArrayRef,
    encoding: Option<StringEncoding>,
    positions: Vec<usize>,
    presence: Option<Vec<bool>>,
    times: Option<ArrayRef>,
}

struct ConditionColumnData {
    column: sdm_contract::frames::materialize::FrameNativeColumnRef,
    converted: ConvertedColumn,
}

fn conversion(
    value: NativeTargetValues,
    attachments: &mut Attachments,
) -> Result<ConvertedColumn, CoreError> {
    let Some(Attachment::Arrow(data)) = attachments.remove(&value.rows) else {
        return Err(invalid_target(
            "native target conversion lacks its actual Arrow rows",
        ));
    };
    let rows = one_batch(&data)?;
    if rows.num_columns() != 2
        || rows.schema().fields()[0].name() != "position"
        || rows.schema().fields()[1].name() != "target"
    {
        return Err(invalid_target(
            "native target conversion needs complete position and target columns",
        ));
    }
    let positions = rows
        .column(0)
        .as_primitive_opt::<UInt64Type>()
        .ok_or_else(|| invalid_target("native target source positions are not uint64"))?;
    let positions = positions
        .iter()
        .map(|row| {
            row.and_then(|row| usize::try_from(row).ok())
                .ok_or_else(|| {
                    invalid_target(
                        "native target position is missing or outside this host's extent",
                    )
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let presence = value
        .presence
        .map(|id| {
            let Some(Attachment::Arrow(data)) = attachments.remove(&id) else {
                return Err(invalid_target(
                    "native target lacks its complete missingness column",
                ));
            };
            let rows = one_batch(&data)?;
            if rows.num_columns() != 1 || rows.column(0).null_count() != 0 {
                return Err(invalid_target(
                    "native target missingness must be one complete nonnullable column",
                ));
            }
            let present = rows
                .column(0)
                .as_boolean_opt()
                .ok_or_else(|| invalid_target("native target missingness is not Boolean"))?;
            Ok(present.values().iter().collect::<Vec<_>>())
        })
        .transpose()?;
    Ok(ConvertedColumn {
        values: Arc::clone(rows.column(1)),
        rows,
        encoding: value.encoding,
        positions,
        presence,
        times: value
            .times
            .map(|id| {
                let Some(Attachment::Arrow(data)) = attachments.remove(&id) else {
                    return Err(invalid_target(
                        "native window lacks its actual captured time values",
                    ));
                };
                let rows = one_batch(&data)?;
                if rows.num_columns() != 1 {
                    return Err(invalid_target(
                        "native time values require one complete source column",
                    ));
                }
                Ok(Arc::clone(rows.column(0)))
            })
            .transpose()?,
    })
}

fn publish(
    asker: &Asker,
    input: RetainedLabelInputs,
    data: Arc<TargetColumn>,
) -> Result<PreparedTargetColumn, CoreError> {
    let input_value = input.publish(asker)?;
    let condition_values = data
        .condition_values
        .iter()
        .map(|value| {
            let rows = asker.attach(Attachment::Arrow(ArrowData::new(
                value.converted.rows.schema(),
                vec![value.converted.rows.clone()],
            )))?;
            let presence = value.converted.presence.as_ref().ok_or_else(|| {
                invalid_target("completed condition lacks its actual native presence")
            })?;
            let presence: ArrayRef = Arc::new(arrow_array::BooleanArray::from(presence.clone()));
            let presence = RecordBatch::try_from_iter([("present", presence)]).map_err(|_| {
                invalid_target("completed condition presence does not form a column")
            })?;
            let presence = asker.attach(Attachment::Arrow(ArrowData::new(
                presence.schema(),
                vec![presence],
            )))?;
            Ok(NativeConditionColumn {
                column: value.column.clone(),
                converted: NativeTargetValues {
                    rows,
                    presence: Some(presence),
                    encoding: value.converted.encoding,
                    times: None,
                },
            })
        })
        .collect::<Result<Vec<_>, CoreError>>()?;
    let value = PreparedTargetColumn {
        input: input_value,
        owner: AttachmentId(0),
        derivation: data.derivation,
        rows: data.positions.len() as u64,
        condition_values,
    };
    let owner = asker.attach(Attachment::TargetColumn(RetainedTargetColumn {
        input,
        data,
    }))?;
    Ok(PreparedTargetColumn { owner, ..value })
}

pub(super) fn column(
    asker: &Asker,
    input_value: PreparedLabelInputs,
    converted: NativeTargetValues,
    attachments: &mut Attachments,
) -> Result<PreparedTargetColumn, CoreError> {
    let input = RetainedLabelInputs::bind(input_value.clone(), attachments)?;
    let ConvertedColumn {
        values,
        encoding,
        positions,
        presence,
        times,
        ..
    } = conversion(converted, attachments)?;
    let presence = presence.ok_or_else(|| {
        invalid_target(
            "native column completion requires its complete first missingness observation",
        )
    })?;
    if input
        .data
        .eligibility
        .as_ref()
        .is_some_and(|condition| condition.selection().is_none())
    {
        return Err(invalid_target(
            "native target precedes complete entity eligibility",
        ));
    }
    let data = TargetColumn::column(
        Arc::clone(&input.data.inputs),
        input.data.target.clone(),
        values,
        encoding,
        positions,
        &presence,
        times,
    )?;
    publish(asker, input, data)
}

pub(super) fn selected(
    asker: &Asker,
    value: PreparedTargetColumn,
    converted: NativeTargetValues,
    window: bool,
    attachments: &mut Attachments,
) -> Result<PreparedTargetColumn, CoreError> {
    let previous = RetainedTargetColumn::bind(value, attachments)?;
    let ConvertedColumn {
        values,
        encoding,
        positions,
        presence,
        times,
        ..
    } = conversion(converted, attachments)?;
    let data = if window {
        if times.is_some() {
            return Err(invalid_target(
                "window application cannot replace its captured time data",
            ));
        }
        let presence = presence.ok_or_else(|| {
            invalid_target(
                "native window completion requires its complete second missingness observation",
            )
        })?;
        TargetColumn::windowed(previous.data, values, encoding, positions, &presence)?
    } else {
        if presence.is_some() {
            return Err(invalid_target(
                "condition-selected target cannot claim another missingness observation",
            ));
        }
        TargetColumn::filtered(previous.data, values, encoding, positions, times)?
    };
    publish(asker, previous.input, data)
}

pub(super) fn condition(
    asker: &Asker,
    value: PreparedTargetColumn,
    operand: NativeConditionColumn,
    attachments: &mut Attachments,
) -> Result<PreparedTargetColumn, CoreError> {
    let previous = RetainedTargetColumn::bind(value, attachments)?;
    let expected = previous
        .input
        .data
        .meaning
        .condition_columns
        .get(previous.data.condition.completed())
        .ok_or_else(|| invalid_target("native condition adds an extra or foreign occurrence"))?;
    if &operand.column != expected {
        return Err(invalid_target(
            "native condition column does not match its actual resolved occurrence",
        ));
    }
    let converted = conversion(operand.converted, attachments)?;
    let data = TargetColumn::condition(previous.data, operand.column, converted)?;
    publish(asker, previous.input, data)
}

#[cfg(test)]
mod tests;

/// Actual target data with its root and exact completed predecessor.
/// Values are snapshots at handover; the native Series remains host-owned.
pub(super) struct TargetColumn {
    pub(super) inputs: Arc<task::LabelInputs>,
    target: Shape,
    pub(super) previous: Option<Arc<TargetColumn>>,
    pub(super) derivation: Derivation,
    pub(super) positions: Vec<usize>,
    pub(super) values: ArrayRef,
    /// Actual native completion, retained independently from exact mathematical values.
    native_values: ArrayRef,
    pub(super) encoding: Option<StringEncoding>,
    /// Complete condition/time selection over the actual predecessor values.
    pub(super) selection: Vec<bool>,
    first_presence: Vec<bool>,
    times: Option<ArrayRef>,
    condition_values: Vec<Arc<ConditionColumnData>>,
    condition: conditions::ConditionInputs,
}

impl TargetColumn {
    pub(super) fn presence(&self) -> &[bool] {
        &self.first_presence
    }

    pub(super) fn condition_target(&self) -> Result<labels::Target, CoreError> {
        if let Shape::Column(_) = &self.target {
            if self.derivation != Derivation::Column {
                return Err(invalid_target("row condition carries a target window"));
            }
            let every = (0..self.values.len()).collect::<Vec<_>>();
            native_condition_column(&self.values, &every, self.encoding).map(labels::Target::Column)
        } else {
            self.groups()
        }
    }
    pub(super) fn group_type(&self) -> Result<DataType, CoreError> {
        group_value_type(&self.values, self.encoding)
    }

    pub(super) fn native_group_type(&self) -> Result<DataType, CoreError> {
        group_value_type(&self.native_values, self.encoding)
    }

    pub(super) fn native_groups(&self) -> Result<labels::Target, CoreError> {
        self.groups_from(&self.native_values)
    }

    pub(super) fn group_encoding(&self) -> Option<StringEncoding> {
        self.encoding.or_else(|| {
            matches!(
                self.values.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            .then_some(StringEncoding::Utf8)
        })
    }
    pub(super) fn column(
        inputs: Arc<task::LabelInputs>,
        target: Shape,
        values: ArrayRef,
        encoding: Option<StringEncoding>,
        positions: Vec<usize>,
        present: &[bool],
        times: Option<ArrayRef>,
    ) -> Result<Arc<Self>, CoreError> {
        if let Shape::Column(column) = &target {
            if times.is_some() {
                return Err(invalid_target(
                    "row conditions cannot name a target time window",
                ));
            }
            let source = inputs
                .native_rows
                .as_ref()
                .unwrap_or(&inputs.population.sampled);
            let before = take(named(source, column)?, &inputs.entity_rows)?;
            let expected = column_positions(inputs.entity_rows.len(), present, false)?;
            validate_positions(&expected, &positions, values.len())?;
            validate_condition_values(
                &before,
                inputs.strings.get(column).copied(),
                &values,
                encoding,
            )?;
            let native_values = Arc::clone(&values);
            let values = mathematical_values(&before, &positions, &values)?;
            return Ok(Arc::new(Self {
                inputs,
                target,
                previous: None,
                derivation: Derivation::Column,
                positions,
                values,
                native_values,
                encoding,
                selection: Vec::new(),
                first_presence: present.to_vec(),
                times: None,
                condition_values: Vec::new(),
                condition: Default::default(),
            }));
        }
        let window = target_window(&target)?;
        let fetched = target_source(&inputs, window)?;
        let expected = column_positions(fetched.rows.num_rows(), present, window.filter.is_none())?;
        validate_positions(&expected, &positions, values.len())?;
        let before = match &window.column {
            Some(column) => Arc::clone(named(&fetched.rows, column)?),
            None => Arc::new(arrow_array::Int64Array::from(vec![
                1;
                fetched.rows.num_rows()
            ])) as ArrayRef,
        };
        validate_values(
            &before,
            fetched
                .strings
                .get(window.column.as_deref().unwrap_or("*"))
                .copied(),
            &values,
            encoding,
        )?;
        let selection = if window.filter.is_some() {
            if times.is_some() {
                return Err(invalid_target(
                    "filtered target observes its time values after its condition",
                ));
            }
            Vec::new()
        } else {
            window_selection(&inputs, window, &positions, times.as_ref())?
        };
        let native_values = Arc::clone(&values);
        let values = mathematical_values(&before, &positions, &values)?;
        Ok(Arc::new(Self {
            inputs,
            target,
            previous: None,
            derivation: Derivation::Column,
            positions,
            values,
            native_values,
            encoding,
            selection,
            first_presence: present.to_vec(),
            times,
            condition_values: Vec::new(),
            condition: conditions::ConditionInputs::default(),
        }))
    }

    fn condition(
        previous: Arc<Self>,
        column: sdm_contract::frames::materialize::FrameNativeColumnRef,
        converted: ConvertedColumn,
    ) -> Result<Arc<Self>, CoreError> {
        if converted.times.is_some() {
            return Err(invalid_target(
                "condition operands cannot replace target time data",
            ));
        }
        let window = target_window(&previous.target)?;
        let filter = window
            .filter
            .as_deref()
            .ok_or_else(|| invalid_target("a plain target has no condition occurrence"))?;
        if previous.derivation != Derivation::Column {
            return Err(invalid_target(
                "native condition lacks its actual incomplete column predecessor",
            ));
        }
        let source = target_source(&previous.inputs, window)?;
        let rows = source.rows.num_rows();
        let expected = (0..rows).collect::<Vec<_>>();
        validate_positions(&expected, &converted.positions, converted.values.len())?;
        let presence = converted.presence.as_deref().ok_or_else(|| {
            invalid_target("native condition lacks its complete missingness observation")
        })?;
        column_positions(rows, presence, false)?;
        let names = conditions::operand_columns(filter)?;
        let name = names
            .get(previous.condition.completed())
            .ok_or_else(|| invalid_target("native condition adds an extra occurrence"))?;
        validate_condition_values(
            named(&source.rows, name)?,
            source.strings.get(*name).copied(),
            &converted.values,
            converted.encoding,
        )?;
        let semantic =
            mathematical_values(named(&source.rows, name)?, &expected, &converted.values)?;
        let values = native_condition_column(&semantic, &expected, converted.encoding)?;
        let condition = previous.condition.complete(filter, values)?;
        let (derivation, selection) = match condition.truth() {
            Some(truth) => {
                if truth.values.len() != previous.first_presence.len() {
                    return Err(invalid_target(
                        "complete condition and target presence do not align",
                    ));
                }
                (
                    Derivation::Conditioned,
                    truth
                        .values
                        .iter()
                        .zip(&previous.first_presence)
                        .map(|(&a, &b)| a && b)
                        .collect(),
                )
            }
            None => (Derivation::Column, Vec::new()),
        };
        let mut condition_values = previous.condition_values.clone();
        condition_values.push(Arc::new(ConditionColumnData { column, converted }));
        Ok(Arc::new(Self {
            inputs: Arc::clone(&previous.inputs),
            target: previous.target.clone(),
            positions: previous.positions.clone(),
            values: Arc::clone(&previous.values),
            native_values: Arc::clone(&previous.native_values),
            encoding: previous.encoding,
            first_presence: previous.first_presence.clone(),
            times: previous.times.clone(),
            previous: Some(previous),
            derivation,
            selection,
            condition_values,
            condition,
        }))
    }

    pub(super) fn filtered(
        previous: Arc<Self>,
        values: ArrayRef,
        encoding: Option<StringEncoding>,
        positions: Vec<usize>,
        times: Option<ArrayRef>,
    ) -> Result<Arc<Self>, CoreError> {
        let window = target_window(&previous.target)?;
        if previous.derivation != Derivation::Conditioned || window.filter.is_none() {
            return Err(invalid_target(
                "condition-selected target lacks its actual filtered column predecessor",
            ));
        }
        let expected = selected_positions(&previous.positions, &previous.selection)?;
        validate_positions(&expected, &positions, values.len())?;
        validate_values(&previous.values, previous.encoding, &values, encoding)?;
        let selection = window_selection(&previous.inputs, window, &positions, times.as_ref())?;
        let offsets = selected_positions(
            &(0..previous.values.len()).collect::<Vec<_>>(),
            &previous.selection,
        )?;
        let native_values = Arc::clone(&values);
        let values = mathematical_values(&previous.values, &offsets, &values)?;
        Ok(Arc::new(Self {
            inputs: Arc::clone(&previous.inputs),
            target: previous.target.clone(),
            first_presence: previous.first_presence.clone(),
            times,
            condition_values: previous.condition_values.clone(),
            condition: previous.condition.clone(),
            previous: Some(previous),
            derivation: Derivation::Filtered,
            positions,
            values,
            native_values,
            encoding,
            selection,
        }))
    }

    pub(super) fn windowed(
        previous: Arc<Self>,
        values: ArrayRef,
        encoding: Option<StringEncoding>,
        positions: Vec<usize>,
        present: &[bool],
    ) -> Result<Arc<Self>, CoreError> {
        let window = target_window(&previous.target)?;
        if previous.derivation
            != if window.filter.is_some() {
                Derivation::Filtered
            } else {
                Derivation::Column
            }
        {
            return Err(invalid_target(
                "window-selected target lacks its actual column/filter predecessor",
            ));
        }
        let in_window = selected_positions(&previous.positions, &previous.selection)?;
        let expected = selected_positions(&in_window, present)?;
        validate_positions(&expected, &positions, values.len())?;
        validate_values(&previous.values, previous.encoding, &values, encoding)?;
        let offsets = selected_positions(
            &(0..previous.values.len()).collect::<Vec<_>>(),
            &previous.selection,
        )?;
        let offsets = selected_positions(&offsets, present)?;
        let native_values = Arc::clone(&values);
        let values = mathematical_values(&previous.values, &offsets, &values)?;
        Ok(Arc::new(Self {
            inputs: Arc::clone(&previous.inputs),
            target: previous.target.clone(),
            first_presence: previous.first_presence.clone(),
            times: previous.times.clone(),
            condition_values: previous.condition_values.clone(),
            condition: previous.condition.clone(),
            previous: Some(previous),
            derivation: Derivation::Windowed,
            positions,
            values,
            native_values,
            encoding,
            selection: Vec::new(),
        }))
    }

    pub(super) fn groups(&self) -> Result<labels::Target, CoreError> {
        self.groups_from(&self.values)
    }

    fn groups_from(&self, values: &ArrayRef) -> Result<labels::Target, CoreError> {
        if self.derivation != Derivation::Windowed {
            return Err(invalid_target(
                "grouping requires the complete actual window-selected target",
            ));
        }
        let window = target_window(&self.target)?;
        let source = target_source(&self.inputs, window)?;
        let batch = self
            .positions
            .iter()
            .map(|&row| source.batch[row])
            .collect::<Vec<_>>();
        let every = (0..values.len()).collect::<Vec<_>>();
        let examples = self.inputs.population.examples.index.len();
        // Window policy has already selected and bound every actual source
        // position. This reuses only the existing stable pair/group kernel.
        if window.aggregation == labels::Aggregation::ListDistinct {
            let values = list_distinct_column(values, &every, self.encoding)?;
            labels::distinct_selected(examples, &batch, &values).map(labels::Target::Listed)
        } else {
            let values = label_values(values)?;
            labels::aggregate_selected(window.aggregation, examples, &batch, &values)
                .map(labels::Target::Aggregated)
        }
    }
}

fn target_window(target: &Shape) -> Result<&Windowed, CoreError> {
    match target {
        Shape::Aggregated(window) => Ok(window),
        _ => Err(invalid_target(
            "native target data names no complete aggregated column",
        )),
    }
}

fn target_source<'a>(
    inputs: &'a task::LabelInputs,
    window: &Windowed,
) -> Result<&'a Fetched, CoreError> {
    inputs
        .fetched
        .get(&window.table)
        .ok_or_else(|| invalid_target("native target source was not read in this label input"))
}

fn window_selection(
    inputs: &task::LabelInputs,
    window: &Windowed,
    positions: &[usize],
    times: Option<&ArrayRef>,
) -> Result<Vec<bool>, CoreError> {
    let source = target_source(inputs, window)?;
    let starts = shifted(
        &inputs.population.examples.anchors,
        window.start,
        "anchor_time",
    )?;
    let ends = shifted(
        &inputs.population.examples.anchors,
        window.end,
        "anchor_time",
    )?;
    let times = times
        .ok_or_else(|| invalid_target("native window lacks its reached captured time values"))?;
    if times.len() != source.rows.num_rows() {
        return Err(invalid_target(
            "captured native time values omit or add source rows",
        ));
    }
    if !matches!(times.data_type(), DataType::Timestamp(_, _)) {
        return Err(invalid_target(
            "native time completion is not a timestamp column",
        ));
    }
    let unzoned = timestamp_without_zone(times);
    let times = times_ns(unzoned.as_ref().unwrap_or(times), &source.time_column)?;
    Ok(positions
        .iter()
        .map(|&row| {
            let batch = source.batch[row];
            times[row].is_some_and(|time| time > starts[batch] && time <= ends[batch])
        })
        .collect())
}

/// Select mathematical values using the actual retained type and exact row map.
/// Native completions remain authoritative for every unprotected scalar family.
fn mathematical_values(
    before: &ArrayRef,
    offsets: &[usize],
    completed: &ArrayRef,
) -> Result<ArrayRef, CoreError> {
    if offsets.len() != completed.len() || offsets.iter().any(|&row| row >= before.len()) {
        return Err(invalid_target(
            "native mathematical projection lacks its complete ordered row map",
        ));
    }
    if matches!(
        before.data_type(),
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _)
    ) {
        take(before, offsets)
    } else {
        Ok(Arc::clone(completed))
    }
}

fn validate_values(
    before: &ArrayRef,
    before_encoding: Option<StringEncoding>,
    values: &ArrayRef,
    encoding: Option<StringEncoding>,
) -> Result<(), CoreError> {
    let family = |value: &ArrayRef, text: Option<StringEncoding>| {
        if text.is_some() {
            return Some(0);
        }
        Some(match value.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => 0,
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => 1,
            DataType::Float32 | DataType::Float64 => 2,
            DataType::Boolean => 3,
            DataType::Decimal128(_, _) => 4,
            _ => return None,
        })
    };
    let exact_decimal = matches!(before.data_type(), DataType::Decimal128(_, _))
        && before_encoding.is_none()
        && encoding.is_none()
        && matches!(family(values, None), Some(1 | 2 | 4));
    if family(before, before_encoding).is_none()
        || (!exact_decimal && family(before, before_encoding) != family(values, encoding))
    {
        return Err(invalid_target(
            "native target completion changes its admitted scalar family",
        ));
    }
    if let Some(encoding) = encoding {
        sdm_engine::strings::owned(values, encoding).map_err(|_| {
            invalid_target("native target text does not match its declared complete encoding")
        })?;
    }
    let rows = (0..values.len()).collect::<Vec<_>>();
    list_distinct_column(values, &rows, encoding)?;
    Ok(())
}

fn validate_condition_values(
    before: &ArrayRef,
    before_encoding: Option<StringEncoding>,
    values: &ArrayRef,
    encoding: Option<StringEncoding>,
) -> Result<(), CoreError> {
    if matches!(before.data_type(), DataType::Timestamp(_, _)) {
        if encoding.is_some() || !matches!(values.data_type(), DataType::Timestamp(_, _)) {
            return Err(invalid_target(
                "native condition changes its temporal scalar family",
            ));
        }
        let every = (0..values.len()).collect::<Vec<_>>();
        native_condition_column(values, &every, encoding)?;
        return Ok(());
    }
    validate_values(before, before_encoding, values, encoding)
}

fn native_condition_column(
    values: &ArrayRef,
    rows: &[usize],
    encoding: Option<StringEncoding>,
) -> Result<labels::Column, CoreError> {
    let canonical = timestamp_without_zone(values);
    condition_column(canonical.as_ref().unwrap_or(values), rows, encoding)
}

fn group_value_type(
    values: &ArrayRef,
    encoding: Option<StringEncoding>,
) -> Result<DataType, CoreError> {
    if let Some(encoding) = encoding {
        sdm_engine::strings::owned(values, encoding).map_err(|_| {
            invalid_target("native group scalar encoding does not match its completed values")
        })?;
        return Ok(match encoding {
            StringEncoding::Utf8 => DataType::Utf8,
            StringEncoding::Utf8Surrogatepass => DataType::Binary,
        });
    }
    Ok(match values.data_type() {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => DataType::Int64,
        DataType::UInt64 => DataType::UInt64,
        DataType::Float32 | DataType::Float64 => DataType::Float32,
        DataType::Boolean => DataType::Boolean,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DataType::Utf8,
        DataType::Decimal128(_, scale) => DataType::Decimal128(38, *scale),
        _ => {
            return Err(invalid_target(
                "native group values have no admitted scalar representation",
            ));
        }
    })
}

pub(super) fn column_positions(
    rows: usize,
    present: &[bool],
    drop_missing: bool,
) -> Result<Vec<usize>, CoreError> {
    if rows != present.len() {
        return Err(invalid_target(
            "native missingness does not cover its complete fetched column",
        ));
    }
    Ok(present
        .iter()
        .enumerate()
        .filter_map(|(row, &present)| (!drop_missing || present).then_some(row))
        .collect())
}

pub(super) fn selected_positions(
    positions: &[usize],
    selected: &[bool],
) -> Result<Vec<usize>, CoreError> {
    if positions.len() != selected.len() {
        return Err(invalid_target(
            "native target selection does not cover its complete predecessor",
        ));
    }
    Ok(positions
        .iter()
        .zip(selected)
        .filter_map(|(&row, &selected)| selected.then_some(row))
        .collect())
}

pub(super) fn validate_positions(
    expected: &[usize],
    positions: &[usize],
    values: usize,
) -> Result<(), CoreError> {
    if positions != expected || values != positions.len() {
        return Err(invalid_target(
            "native target completion omits, reorders or misaligns selected rows",
        ));
    }
    Ok(())
}

fn invalid_target(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}
