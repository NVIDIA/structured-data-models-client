// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Owned provider-backed source and actual completed native label data.
//! Sampling and request policy are the same kernels used by raw Materialize.
//! No owner retains an asker, suspended future, cursor or host callback.

use super::*;
use crate::attached::take_option_values_if;
use crate::compose::DataAccess;
use crate::frame_materialize::identity::Identity;
use crate::run::Attachments;
use sdm_contract::frames::materialize::{
    FrameBatchRange, FrameNativeColumnRef, RestoredFrameLabels,
};
use sdm_contract::materialize::*;
use std::sync::Weak;

pub(super) const BATCH_ROWS: usize = 10_000;

#[cfg(test)]
mod tests;

/// Native materialization wraps an empty-context ValueError at this reached boundary.
/// Keep the shared sentence/private facts and the existing native presentation recipe.
fn native_context_diagnostic(mut error: CoreError) -> CoreError {
    if matches!(
        error.sentence,
        Sentence::ContextTooRestrictive
            | Sentence::NoEntityAtAnchor
            | Sentence::NoEntityAtAnchorLater
            | Sentence::NoEntityPassed
            | Sentence::NoEntityPassedSeveral
            | Sentence::AnchorTooLate
            | Sentence::AnchorTooLateSeveral
    ) {
        error
            .fields
            .insert("native_key_reason".into(), Detail::text("task_validation"));
        error.public_details = Some(Box::default());
    }
    error
}

fn invalid(message: &str) -> CoreError {
    CoreError::protocol(Sentence::InvalidMessage, message)
}

struct SourceData {
    meaning: PreparedMaterializeSource,
    source: Arc<prepared::PreparedSource>,
    access: DataAccess,
    query: Option<QueryFacts>,
}

#[derive(Clone)]
pub struct RetainedMaterializeSource(Arc<SourceData>);

impl std::fmt::Debug for RetainedMaterializeSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedMaterializeSource")
            .field("identity", &self.0.meaning.identity)
            .finish_non_exhaustive()
    }
}

struct PopulationData {
    meaning: PreparedMaterializePopulation,
    source: RetainedMaterializeSource,
    task: prepared::PreparedTask,
}

#[derive(Clone)]
pub struct RetainedMaterializePopulation(Arc<PopulationData>);

#[derive(Clone)]
struct GroupsData {
    range: FrameBatchRange,
    previous: Option<RetainedMaterializeBatch>,
    groups: Arc<task::EvaluatedLabelGroups>,
    native_input: native_target::RetainedLabelInputs,
    restored: Option<super::native_conditions::RestoredGroups>,
    encoding: Option<sdm_contract::data::strings::StringEncoding>,
    scalar: DataType,
    native_groups: task::LabelledContext,
    native_scalar: DataType,
    memo: SamplerMemo,
}

#[derive(Clone)]
pub struct RetainedMaterializeGroups {
    population: RetainedMaterializePopulation,
    data: Arc<GroupsData>,
}

impl std::fmt::Debug for RetainedMaterializeGroups {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedMaterializeGroups")
            .field("range", &self.data.range)
            .finish_non_exhaustive()
    }
}

impl RetainedMaterializeGroups {
    pub fn rows(&self) -> Result<ArrowData, CoreError> {
        if let Some(restored) = &self.data.restored {
            return Ok(ArrowData::new(
                restored.rows.schema(),
                vec![restored.rows.clone()],
            ));
        }
        let rows =
            native_labels::group_rows(&self.data.native_groups, Some(&self.data.native_scalar))?;
        Ok(ArrowData::new(rows.schema(), vec![rows]))
    }
}

struct BatchData {
    range: FrameBatchRange,
    previous: Option<Weak<BatchData>>,
    accepted: usize,
    complete: bool,
    labelled: task::LabelledContext,
    memo: SamplerMemo,
}

#[derive(Clone)]
pub struct RetainedMaterializeBatch {
    population: RetainedMaterializePopulation,
    data: Arc<BatchData>,
}

impl std::fmt::Debug for RetainedMaterializeBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedMaterializeBatch")
            .field("range", &self.data.range)
            .finish_non_exhaustive()
    }
}

fn bind_batch(
    population: &RetainedMaterializePopulation,
    value: PreparedMaterializeBatch,
    attachments: &mut Attachments,
) -> Result<RetainedMaterializeBatch, CoreError> {
    let Some(Attachment::MaterializeBatch(owner)) = attachments.remove(&value.owner) else {
        return Err(invalid(
            "native label prefix lacks its actual completed batch owner",
        ));
    };
    if !population.same_owner(&owner.population)
        || value.population != population.0.meaning.identity
        || value.range != owner.data.range
        || value.complete != owner.data.complete
    {
        return Err(invalid(
            "native label batch does not match its actual source population and range",
        ));
    }
    Ok(owner)
}

fn candidate_range(start: u64, rows: usize) -> Option<FrameBatchRange> {
    let start_ = usize::try_from(start).ok()?;
    (start_ < rows && start_ % BATCH_ROWS == 0).then(|| FrameBatchRange {
        start,
        stop: rows.min(start_.saturating_add(BATCH_ROWS)) as u64,
    })
}

impl std::fmt::Debug for RetainedMaterializePopulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetainedMaterializePopulation")
            .field("identity", &self.0.meaning.identity)
            .finish_non_exhaustive()
    }
}

fn target(source: &prepared::PreparedSource) -> Result<FrameNativeColumnRef, CoreError> {
    let Shape::Aggregated(window) = &source.asked.target else {
        return Err(invalid(
            "native list source lacks its resolved aggregated target",
        ));
    };
    if window.aggregation != labels::Aggregation::ListDistinct {
        return Err(invalid("native list source has another aggregation"));
    }
    let table = source
        .graph
        .table(&window.table)
        .ok_or_else(|| invalid("native list source lost its resolved target table"))?;
    let column = window
        .column
        .clone()
        .filter(|column| table.column(column).is_some())
        .ok_or_else(|| invalid("native list source lost its resolved target column"))?;
    Ok(FrameNativeColumnRef {
        source: table.source.clone(),
        column,
    })
}

impl RetainedMaterializeSource {
    fn bind(
        value: PreparedMaterializeSource,
        attachments: &mut Attachments,
    ) -> Result<Self, CoreError> {
        let Some(Attachment::MaterializeSource(owner)) = attachments.remove(&value.owner) else {
            return Err(invalid(
                "native source preparation lacks its actual source owner",
            ));
        };
        let mut expected = owner.0.meaning.clone();
        expected.owner = value.owner;
        if expected != value {
            return Err(invalid(
                "native source meaning does not match its immutable owner",
            ));
        }
        Ok(owner)
    }

    fn restore(&self, asker: Asker, memo: &SamplerMemo) -> Asker {
        let asker = asker.with_access(self.0.access.clone());
        asker.describe_fetch_partitions();
        asker.locate(self.0.source.description.clone());
        for table in &self.0.source.description.tables {
            if let Some(read) = self
                .0
                .meaning
                .tables
                .iter()
                .find(|captured| captured.source == table.source)
            {
                asker.pin(&table.source, read.read.clone());
            }
            asker.guard(
                &table.source,
                table
                    .columns
                    .iter()
                    .map(|column| column.name.clone())
                    .collect(),
            );
        }
        asker.keep(memo.clone());
        asker
    }

    fn column(
        &self,
        column: &FrameNativeColumnRef,
    ) -> Result<&sdm_contract::ops::DescribedColumn, CoreError> {
        self.0
            .source
            .description
            .tables
            .iter()
            .find(|table| table.source == column.source)
            .and_then(|table| {
                table
                    .columns
                    .iter()
                    .find(|field| field.name == column.column)
            })
            .ok_or_else(|| {
                invalid("native source column does not belong to its complete source meaning")
            })
    }
}

impl RetainedMaterializePopulation {
    pub(super) fn condition_context(&self) -> Result<super::native_conditions::Context, CoreError> {
        let source = &self.0.task.source;
        let entity = source
            .graph
            .table(&source.asked.entity_table)
            .ok_or_else(|| invalid("native condition lost its entity table"))?;
        let key = entity
            .primary_key
            .as_ref()
            .ok_or_else(|| invalid("native condition lost its entity key"))?;
        let filtered = source.entity_filter.is_some();
        Ok(super::native_conditions::Context {
            entity: ColumnRef {
                table: entity.name.clone(),
                column: key.clone(),
            },
            filter: if filtered {
                source.asked.choosing.first().cloned()
            } else {
                None
            },
            what_if: source
                .asked
                .choosing
                .iter()
                .skip(usize::from(filtered))
                .cloned()
                .collect(),
        })
    }

    pub(super) fn column_binding(
        &self,
        table: &str,
        column: &str,
    ) -> Result<FrameNativeColumnRef, CoreError> {
        let table = self
            .0
            .task
            .source
            .graph
            .table(table)
            .filter(|table| column == "*" || table.column(column).is_some())
            .ok_or_else(|| invalid("native condition lost its captured source column"))?;
        Ok(FrameNativeColumnRef {
            source: table.source.clone(),
            column: column.into(),
        })
    }

    pub(super) fn temporal_columns(&self, table: &str) -> Result<&[String], CoreError> {
        let source = self.column_binding(table, "*")?.source;
        self.0
            .source
            .0
            .meaning
            .tables
            .iter()
            .find(|table| table.source == source)
            .map(|table| table.temporal_columns.as_slice())
            .ok_or_else(|| invalid("complete temporal source lost its captured declaration"))
    }

    fn bind(
        value: PreparedMaterializePopulation,
        attachments: &mut Attachments,
    ) -> Result<Self, CoreError> {
        let Some(Attachment::MaterializePopulation(owner)) = attachments.remove(&value.owner)
        else {
            return Err(invalid(
                "native label input lacks its actual candidate population",
            ));
        };
        let mut expected = owner.0.meaning.clone();
        expected.owner = value.owner;
        if expected != value {
            return Err(invalid(
                "native candidate meaning does not match its actual complete population",
            ));
        }
        Ok(owner)
    }

    fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

fn publish(asker: &Asker, value: PreparedMaterializeValue) -> Result<OperationResult, CoreError> {
    Ok(OperationResult::PrepareMaterialize(PreparedMaterialize {
        value: asker
            .attach(Attachment::Bytes(serde_json::to_vec(&value).map_err(
                |_| internal("native materialization value does not serialize"),
            )?))?,
    }))
}

pub(crate) async fn task(asker: Asker, inputs: Inputs) -> Result<OperationResult, CoreError> {
    let (steps, ranked, query) = resolved(&inputs);
    if steps.is_some() || ranked.is_some() || inputs.candidates.is_some() {
        let result = super::run(asker.clone(), inputs).await?;
        return nested(&asker, result);
    }
    let source = Arc::new(prepare_source(&asker, inputs).await?);
    if !matches!(&source.asked.target, Shape::Aggregated(window) if window.aggregation == labels::Aggregation::ListDistinct)
    {
        let task = prepared::population(&asker, source).await?;
        let labelled = task.context(&asker).await?;
        let gathered = prepared::finish(&asker, &task, labelled).await?;
        let result = gathered_result(&asker, gathered, query)?;
        return nested(&asker, result);
    }
    let target = target(&source)?;
    let tables = source
        .description
        .tables
        .iter()
        .map(|table| MaterializeSourceTable {
            source: table.source.clone(),
            read: asker.relation_for_read(&table.source),
            columns: table
                .columns
                .iter()
                .map(|column| column.name.clone())
                .collect(),
            temporal_columns: table
                .columns
                .iter()
                .filter(|column| {
                    (!column.untyped && column.dtype == Dtype::Time)
                        || column.stype == Stype::Timestamp
                })
                .map(|column| column.name.clone())
                .collect(),
            time_column: table.time_column.clone(),
        })
        .collect();
    let mut identity = Identity::new("provider-backed-native-source");
    identity.value(&source.description)?;
    identity.value(&source.source_query)?;
    identity.value(&source.seed)?;
    identity.value(&(source.anchor, source.context_anchor))?;
    identity.bytes(source.options_identity.as_bytes());
    let mut meaning = PreparedMaterializeSource {
        source: source.source.clone(),
        identity: identity.finish(),
        owner: AttachmentId(0),
        target,
        tables,
    };
    let access = asker
        .data()
        .cloned()
        .ok_or_else(|| invalid("a native provider source has no actual data access"))?;
    let owner = RetainedMaterializeSource(Arc::new(SourceData {
        meaning: meaning.clone(),
        source,
        access,
        query,
    }));
    meaning.owner = asker.attach(Attachment::MaterializeSource(owner))?;
    publish(&asker, PreparedMaterializeValue::Source(meaning))
}

fn nested(asker: &Asker, result: OperationResult) -> Result<OperationResult, CoreError> {
    match result {
        OperationResult::Materialize(value) => {
            publish(asker, PreparedMaterializeValue::Request(value))
        }
        _ => Err(invalid(
            "native preparation completed with another operation result",
        )),
    }
}

async fn population(
    asker: Asker,
    value: PreparedMaterializeSource,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    let source = RetainedMaterializeSource::bind(value, attachments)?;
    let asker = source.restore(asker, &source.0.source.memo);
    let task = prepared::population(&asker, Arc::clone(&source.0.source)).await?;
    let mut identity = Identity::new("provider-backed-native-population");
    identity.bytes(source.0.meaning.identity.as_bytes());
    identity.value(&task.population.examples.index)?;
    identity.value(&task.population.examples.anchors)?;
    identity.arrow(&ArrowData::new(
        task.population.sampled.schema(),
        vec![task.population.sampled.clone()],
    ))?;
    let mut meaning = PreparedMaterializePopulation {
        source: source.0.meaning.source.clone(),
        identity: identity.finish(),
        owner: AttachmentId(0),
        rows: task.population.examples.index.len() as u64,
        batch_rows: BATCH_ROWS as u32,
        target: source.0.meaning.target.clone(),
    };
    let owner = RetainedMaterializePopulation(Arc::new(PopulationData {
        meaning: meaning.clone(),
        source,
        task,
    }));
    meaning.owner = asker.attach(Attachment::MaterializePopulation(owner))?;
    publish(&asker, PreparedMaterializeValue::Population(meaning))
}

fn column_meaning(
    owner: &RetainedMaterializeSource,
    column: FrameNativeColumnRef,
    input: MaterializeSourceFacts,
) -> Result<MaterializeColumnMeaning, CoreError> {
    let declared = owner.column(&column)?;
    let dtype = (!declared.untyped).then_some(declared.dtype);
    let scalar = native_source::scalar_meaning(dtype, &input)?;
    let scalar = native_source::timestamp_meaning(dtype, declared.stype, &input).unwrap_or(scalar);
    let notice = native_source::time_notice(dtype, declared.stype, &column.column, &input)?;
    Ok(MaterializeColumnMeaning {
        source: owner.0.meaning.identity.clone(),
        column,
        scalar,
        input,
        notice,
    })
}

async fn label_inputs(
    asker: Asker,
    value: PreparedMaterializePopulation,
    range: FrameBatchRange,
    previous: Option<PreparedMaterializeBatch>,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    use super::native_target::{LabelRoot, RetainedLabelInputs};
    let population = RetainedMaterializePopulation::bind(value, attachments)?;
    let previous = previous
        .map(|value| bind_batch(&population, value, attachments))
        .transpose()?;
    let task = &population.0.task;
    if range.start != previous.as_ref().map_or(0, |batch| batch.data.range.stop)
        || candidate_range(range.start, task.population.examples.index.len()) != Some(range)
        || previous.as_ref().is_some_and(|batch| batch.data.complete)
    {
        return Err(invalid(
            "native SQL candidate range does not follow its actual completed prefix",
        ));
    }
    let memo = previous
        .as_ref()
        .map_or(&task.memo, |batch| &batch.data.memo);
    let asker = population.0.source.restore(asker, memo);
    let mut memo = Memo {
        asker: &asker,
        frames: None,
        memo: memo.clone(),
    };
    let plan = prepared::Draw::new(&task.source)?;
    let inputs = task::read_label_inputs(
        &asker,
        &mut memo,
        plan.draw(&task.source.read_times),
        task.population
            .slice(range.start as usize..range.stop as usize)?,
        None,
        true,
    )
    .await?;
    keep_stored_times(&mut memo, &task.source.read_times);
    let Shape::Aggregated(window) = &task.source.asked.target else {
        return Err(invalid(
            "native SQL target lost its captured list aggregation",
        ));
    };
    let fetched = inputs
        .fetched
        .get(&window.table)
        .ok_or_else(|| invalid("native SQL target lacks its actual fetched source rows"))?;
    let source_read = fetched
        .source_read
        .ok_or_else(|| invalid("native SQL target source has no actual issued-read identity"))?;
    let table = task
        .source
        .graph
        .table(&window.table)
        .ok_or_else(|| invalid("native SQL target lost its captured table"))?;
    let condition_columns = window
        .filter
        .as_deref()
        .map(|filter| {
            native_target::operand_columns(filter)?
                .into_iter()
                .map(|column| {
                    if table.column(column).is_none() {
                        return Err(invalid(
                            "native SQL condition lost a captured source column",
                        ));
                    }
                    Ok(FrameNativeColumnRef {
                        source: table.source.clone(),
                        column: column.into(),
                    })
                })
                .collect::<Result<Vec<_>, CoreError>>()
        })
        .transpose()?
        .unwrap_or_default();
    let meaning = PreparedLabelInputs {
        population: population.0.meaning.identity.clone(),
        range,
        owner: AttachmentId(0),
        source_read: Some(source_read),
        target: population.0.meaning.target.clone(),
        time: Some(
            population.column_binding(&window.table, &inputs.fetched[&window.table].time_column)?,
        ),
        form: if window.filter.is_some() {
            TargetColumnForm::Filtered
        } else {
            TargetColumnForm::Plain
        },
        condition_columns,
        source_role: Default::default(),
        eligibility: None,
    };
    let root = LabelRoot::Source {
        population: population.clone(),
        previous,
        memo: memo.memo.clone(),
    };
    let owner = RetainedLabelInputs::new(meaning, root, inputs)?;
    publish(
        &asker,
        PreparedMaterializeValue::LabelInputs(owner.publish(&asker)?),
    )
}

fn native_groups(
    asker: &Asker,
    value: PreparedTargetColumn,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    use native_target::{LabelRoot, RetainedTargetColumn};
    let column = RetainedTargetColumn::bind(value, attachments)?;
    let LabelRoot::Source {
        population,
        previous,
        memo,
    } = &column.input.data.root
    else {
        return Err(invalid(
            "SQL groups require their actual provider-backed population root",
        ));
    };
    let task = &population.0.task;
    let plan = prepared::Draw::new(&task.source)?;
    let mut groups = task::evaluate_target_groups(
        asker,
        plan.draw(&task.source.read_times),
        Arc::clone(&column.input.data.inputs),
        column.data.groups()?,
    )?;
    groups.entity_selection = column
        .input
        .data
        .eligibility
        .as_ref()
        .and_then(|condition| condition.selection().map(<[bool]>::to_vec));
    let native = groups.projected_native_labels(column.data.native_groups()?)?;
    let native_scalar = column.data.native_group_type()?;
    let encoding =
        native_labels::label_encoding(&native.labels).or_else(|| column.data.group_encoding());
    let scalar = column.data.group_type()?;
    let mut value = PreparedMaterializeGroups {
        population: population.0.meaning.identity.clone(),
        range: column.input.data.meaning.range,
        owner: AttachmentId(0),
        native_target: Some(population.0.meaning.target.clone()),
        target_encoding: encoding,
        restored: None,
        what_if: None,
    };
    let owner = RetainedMaterializeGroups {
        population: population.clone(),
        data: Arc::new(GroupsData {
            range: value.range,
            previous: previous.clone(),
            groups: Arc::new(groups),
            native_input: column.input.clone(),
            restored: None,
            encoding,
            scalar,
            native_groups: native,
            native_scalar,
            memo: memo.clone(),
        }),
    };
    value.owner = asker.attach(Attachment::MaterializeGroups(owner))?;
    publish(asker, PreparedMaterializeValue::LabelGroups(value))
}

fn bind_groups(
    value: PreparedMaterializeGroups,
    attachments: &mut Attachments,
) -> Result<RetainedMaterializeGroups, CoreError> {
    let Some(Attachment::MaterializeGroups(owner)) = attachments.remove(&value.owner) else {
        return Err(invalid("native groups lack their actual owner"));
    };
    if value.population != owner.population.0.meaning.identity
        || value.range != owner.data.range
        || value.native_target.as_ref() != Some(&owner.population.0.meaning.target)
        || value.target_encoding != owner.data.encoding
    {
        return Err(invalid(
            "native groups rewrite their actual source and range",
        ));
    }
    match &owner.data.restored {
        Some(restored) => restored.matches(value.restored, value.what_if, attachments)?,
        None if value.restored.is_none() && value.what_if.is_none() => {}
        None => {
            return Err(invalid(
                "native groups invent unobserved restored or condition values",
            ));
        }
    }
    Ok(owner)
}

fn publish_groups(
    asker: &Asker,
    owner: RetainedMaterializeGroups,
) -> Result<OperationResult, CoreError> {
    let (restored, what_if) = match &owner.data.restored {
        Some(restored) => {
            let (rows, conditions) = restored.publish(asker)?;
            (Some(rows), conditions)
        }
        None => (None, None),
    };
    let mut value = PreparedMaterializeGroups {
        population: owner.population.0.meaning.identity.clone(),
        range: owner.data.range,
        owner: AttachmentId(0),
        native_target: Some(owner.population.0.meaning.target.clone()),
        target_encoding: owner.data.encoding,
        restored,
        what_if,
    };
    asker.keep(owner.data.memo.clone());
    value.owner = asker.attach(Attachment::MaterializeGroups(owner))?;
    publish(asker, PreparedMaterializeValue::LabelGroups(value))
}

fn restore_groups(
    asker: &Asker,
    value: PreparedMaterializeGroups,
    restored: RestoredFrameLabels,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    let owner = bind_groups(value, attachments)?;
    if owner.data.restored.is_some() {
        return Err(invalid(
            "native groups already contain their restored values",
        ));
    }
    let Some(Attachment::Arrow(rows)) = attachments.remove(&restored.rows) else {
        return Err(invalid("native group restoration lacks its actual rows"));
    };
    let rows = one_batch(&rows)?;
    let expected = one_batch(&owner.rows()?)?;
    let before = owner
        .data
        .groups
        .native_labels()?
        .ok_or_else(|| invalid("native groups have no pre-mask list values"))?;
    let labels = native_labels::restored_with_meaning(
        &before,
        &expected,
        &rows,
        owner.data.encoding,
        restored.encoding,
        Some(&owner.data.scalar),
    )?;
    let restored = super::native_conditions::RestoredGroups::new(
        &owner.data.native_input,
        labels,
        rows,
        restored.encoding,
    )?;
    let mut data = owner.data.as_ref().clone();
    data.restored = Some(restored);
    publish_groups(
        asker,
        RetainedMaterializeGroups {
            population: owner.population,
            data: Arc::new(data),
        },
    )
}

fn what_if(
    asker: &Asker,
    value: PreparedMaterializeGroups,
    column: PreparedTargetColumn,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    let owner = bind_groups(value, attachments)?;
    let column = native_target::RetainedTargetColumn::bind(column, attachments)?;
    let restored = owner
        .data
        .restored
        .as_ref()
        .ok_or_else(|| invalid("what-if input precedes actual native restoration"))?
        .complete(column)?;
    let mut data = owner.data.as_ref().clone();
    data.restored = Some(restored);
    publish_groups(
        asker,
        RetainedMaterializeGroups {
            population: owner.population,
            data: Arc::new(data),
        },
    )
}

fn complete_batch(
    asker: &Asker,
    value: PreparedMaterializePopulation,
    groups: PreparedMaterializeGroups,
    restored: Option<RestoredFrameLabels>,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    let population = RetainedMaterializePopulation::bind(value, attachments)?;
    let owner = bind_groups(groups.clone(), attachments)?;
    if !population.same_owner(&owner.population) || restored.is_some() {
        return Err(invalid(
            "native SQL batch does not consume its actual population and restored history",
        ));
    }
    let restored = owner
        .data
        .restored
        .as_ref()
        .ok_or_else(|| invalid("native SQL batch precedes actual restored group completion"))?;
    let task = &population.0.task;
    let plan = prepared::Draw::new(&task.source)?;
    let labelled = task::complete_observed_label_groups(
        asker,
        plan.draw(&task.source.read_times),
        &owner.data.groups,
        restored.labels.as_ref().clone(),
        restored.selection()?,
    )?;
    let previous = owner.data.previous.as_ref();
    let accepted = previous
        .map_or(0, |batch| batch.data.accepted)
        .checked_add(labelled.keys.len())
        .ok_or_else(|| invalid("completed native SQL labels exceed the declared host capacity"))?;
    let complete = accepted >= task.source.context_examples;
    let mut value = PreparedMaterializeBatch {
        population: population.0.meaning.identity.clone(),
        range: groups.range,
        owner: AttachmentId(0),
        complete,
    };
    let batch = RetainedMaterializeBatch {
        population,
        data: Arc::new(BatchData {
            range: groups.range,
            previous: previous.map(|batch| Arc::downgrade(&batch.data)),
            accepted,
            complete,
            labelled,
            memo: owner.data.memo.clone(),
        }),
    };
    value.owner = asker.attach(Attachment::MaterializeBatch(batch))?;
    publish(asker, PreparedMaterializeValue::LabelBatch(value))
}

async fn request(
    asker: Asker,
    population: PreparedMaterializePopulation,
    values: Vec<PreparedMaterializeBatch>,
    current: Option<AttachmentId>,
    attachments: &mut Attachments,
) -> Result<OperationResult, CoreError> {
    let population = RetainedMaterializePopulation::bind(population, attachments)?;
    let mut batches: Vec<RetainedMaterializeBatch> = Vec::with_capacity(values.len());
    for value in values {
        let batch = bind_batch(&population, value, attachments)?;
        let previous = batches.last();
        let follows = match (&batch.data.previous, previous) {
            (None, None) => true,
            (Some(found), Some(expected)) => found.ptr_eq(&Arc::downgrade(&expected.data)),
            _ => false,
        };
        if !follows
            || batch.data.range.start != previous.map_or(0, |batch| batch.data.range.stop)
            || previous.is_some_and(|batch| batch.data.complete)
            || previous
                .map_or(0, |batch| batch.data.accepted)
                .checked_add(batch.data.labelled.keys.len())
                != Some(batch.data.accepted)
        {
            return Err(invalid(
                "native SQL request does not retain its complete actual ordered batch lineage",
            ));
        }
        batches.push(batch);
    }
    let task = &population.0.task;
    let exhausted = batches.last().map_or(0, |batch| batch.data.range.stop)
        == task.population.examples.index.len() as u64;
    if !exhausted && !batches.last().is_some_and(|batch| batch.data.complete) {
        return Err(invalid(
            "native SQL request lacks its complete reached candidate prefix",
        ));
    }
    let labelled = match batches.first() {
        Some(first) => {
            let keys = arrow_select::concat::concat(
                &batches
                    .iter()
                    .map(|batch| batch.data.labelled.keys.as_ref())
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| invalid("native SQL completed batch keys do not concatenate"))?;
            let mut labels = first.data.labelled.labels.clone();
            for batch in batches.iter().skip(1) {
                labels.extend(&batch.data.labelled.labels)?;
            }
            task::LabelledContext {
                keys,
                anchors: batches
                    .iter()
                    .flat_map(|batch| batch.data.labelled.anchors.iter().copied())
                    .collect(),
                labels,
            }
        }
        None => task::LabelledContext {
            keys: task.population.entity_key.slice(0, 0),
            anchors: vec![],
            labels: labels::Labels::Lists(vec![]),
        },
    };
    let memo = batches.last().map_or(&task.memo, |batch| &batch.data.memo);
    let asker = population.0.source.restore(asker, memo);
    let mut memo = Memo {
        asker: &asker,
        frames: None,
        memo: memo.clone(),
    };
    let plan = prepared::Draw::new(&task.source)?;
    let labelled = task::finish_context(
        &asker,
        &mut memo,
        plan.draw(&task.source.read_times),
        labelled,
        task.population.kept.len(),
    )
    .await?;
    keep_stored_times(&mut memo, &task.source.read_times);
    let final_task = prepared::PreparedTask {
        source: Arc::clone(&task.source),
        population: task.population.clone(),
        indices: Arc::clone(&task.indices),
        memo: memo.memo,
    };
    let current = take_option_values_if(attachments, current)?;
    let gathered = prepared::finish_native(&asker, &final_task, labelled, current.as_ref()).await?;
    let result = gathered_result(&asker, gathered, population.0.source.0.query.clone())?;
    nested(&asker, result)
}

pub(crate) async fn run(
    asker: Asker,
    input: MaterializePreparation,
    attachments: Attachments,
) -> Result<OperationResult, CoreError> {
    run_inner(asker, input, attachments)
        .await
        .map_err(native_context_diagnostic)
}

async fn run_inner(
    asker: Asker,
    input: MaterializePreparation,
    mut attachments: Attachments,
) -> Result<OperationResult, CoreError> {
    match input {
        MaterializePreparation::LookupRows { lookup, helpers } => {
            let Some(Attachment::LookupRows(owner)) = attachments.remove(&lookup.owner) else {
                return Err(invalid("counted lookup lacks its actual task batch owner"));
            };
            if !owner.matches(&lookup) {
                return Err(invalid(
                    "counted lookup does not match its issued read and task range",
                ));
            }
            let Some(Attachment::Arrow(rows)) = attachments.remove(&helpers) else {
                return Err(invalid("counted lookup lacks its actual raw helper rows"));
            };
            let positions = owner.validate(&one_batch(&rows)?)?;
            let positions: ArrayRef = Arc::new(arrow_array::UInt64Array::from_iter_values(
                positions.into_iter().map(|row| row as u64),
            ));
            let rows = RecordBatch::try_from_iter([("position", positions)])
                .map_err(|_| invalid("lookup positions do not form a column"))?;
            let positions =
                asker.attach(Attachment::Arrow(ArrowData::new(rows.schema(), vec![rows])))?;
            publish(&asker, PreparedMaterializeValue::LookupRows { positions })
        }
        MaterializePreparation::Request {
            population,
            batches,
            values,
        } => request(asker, population, batches, values, &mut attachments).await,
        MaterializePreparation::LabelInputs {
            population,
            range,
            previous,
        } => label_inputs(asker, population, range, previous, &mut attachments).await,
        MaterializePreparation::TargetColumn { inputs, converted } => {
            let value = native_target::column(&asker, inputs, converted, &mut attachments)?;
            publish(&asker, PreparedMaterializeValue::TargetColumn(value))
        }
        MaterializePreparation::EligibilityColumn { inputs, column } => {
            let value =
                super::native_conditions::eligibility(&asker, inputs, column, &mut attachments)?;
            publish(&asker, PreparedMaterializeValue::LabelInputs(value))
        }
        MaterializePreparation::ConditionColumn { column, operand } => {
            let value = native_target::condition(&asker, column, operand, &mut attachments)?;
            publish(&asker, PreparedMaterializeValue::TargetColumn(value))
        }
        MaterializePreparation::FilteredTarget { column, converted } => {
            let value =
                native_target::selected(&asker, column, converted, false, &mut attachments)?;
            publish(&asker, PreparedMaterializeValue::TargetColumn(value))
        }
        MaterializePreparation::WindowedTarget { column, converted } => {
            let value = native_target::selected(&asker, column, converted, true, &mut attachments)?;
            publish(&asker, PreparedMaterializeValue::TargetColumn(value))
        }
        MaterializePreparation::NativeLabelGroups { column } => {
            native_groups(&asker, column, &mut attachments)
        }
        MaterializePreparation::RestoreLabelGroups { groups, restored } => {
            restore_groups(&asker, groups, restored, &mut attachments)
        }
        MaterializePreparation::WhatIfColumn { groups, column } => {
            what_if(&asker, groups, column, &mut attachments)
        }
        MaterializePreparation::CompleteLabelBatch {
            population,
            groups,
            restored,
        } => complete_batch(&asker, population, groups, restored, &mut attachments),
        MaterializePreparation::Population { source } => {
            population(asker, source, &mut attachments).await
        }
        MaterializePreparation::SourceColumn {
            source,
            column,
            input,
        } => {
            let source = RetainedMaterializeSource::bind(source, &mut attachments)?;
            publish(
                &asker,
                PreparedMaterializeValue::ColumnMeaning(column_meaning(&source, column, input)?),
            )
        }
        MaterializePreparation::SourceFallback {
            source,
            mut meaning,
            failure,
        } => {
            let source = RetainedMaterializeSource::bind(source, &mut attachments)?;
            let expected = column_meaning(&source, meaning.column.clone(), meaning.input.clone())?;
            if expected != meaning
                || !matches!(
                    meaning.scalar,
                    MaterializeScalarMeaning::Decimal { .. }
                        | MaterializeScalarMeaning::ExactDecimal { .. }
                )
                || !(failure.arithmetic
                    || failure.builtin.ancestry.iter().any(|class| {
                        matches!(
                            class,
                            sdm_contract::prepare::BuiltinFailureClass::TypeError
                                | sdm_contract::prepare::BuiltinFailureClass::ValueError
                        )
                    }))
            {
                return Err(invalid(
                    "native scalar fallback was not reached by its actual Decimal conversion",
                ));
            }
            meaning.scalar = MaterializeScalarMeaning::Original {};
            publish(&asker, PreparedMaterializeValue::ColumnMeaning(meaning))
        }
        MaterializePreparation::SourceFailure {
            source,
            read,
            error,
        } => {
            let source = RetainedMaterializeSource::bind(source, &mut attachments)?;
            native_source::reported_failure(&source.0.meaning.tables, &read, error)?;
            Err(internal(
                "a source failure returned without its terminal diagnostic",
            ))
        }
        _ => Err(invalid(
            "native materialization product has no matching completed-data consumer",
        )),
    }
}
