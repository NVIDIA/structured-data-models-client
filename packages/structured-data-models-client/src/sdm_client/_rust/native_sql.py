# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native SQL source and target handovers over closed core products.

The core selects the source, candidate slices, targets, masks and request.
This adapter retains actual native data only for the reached conversions.
Every operation is an explicit domain preparation, without a callback RunStep.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import pandas as pd
import pyarrow as pa

from sdm_client._rust import codec, results
from sdm_client._rust.frame_native import NativeFrameFailure, guard_failure
from sdm_client._rust.native_label_stage import (
    ColumnExport,
    complete_eligibility,
    complete_target,
    complete_what_if,
)
from sdm_client._rust.native_products import Bindings, Product
from sdm_client._rust.native_source_completion import (
    NativeSourceTable,
    SourceCompletion,
)
from sdm_client._rust.native_source_rows import SourceReadFailure
from sdm_client._rust.native_sql_products import adopt, bind, read, request
from sdm_client._rust.native_sql_sources import complete_source_times
from sdm_client._rust.native_target import export_groups
from sdm_client._rust.run import Finished, _driver, run


class _FetchedColumns:
    """The actual frame of this one producing Run and its selected source read."""

    def __init__(
        self, table: NativeSourceTable, preparation: _Preparation
    ) -> None:
        self.table = table
        try:
            self.times = (
                None
                if table.time_column is None
                else table.frame[table.time_column]
            )
        except Exception as error:
            preparation.native_failure(NativeFrameFailure.capture(error))

    def target_series(
        self, binding: dict[str, Any], rows: list[int]
    ) -> pd.Series:
        if binding['source'] != self.table.source:
            raise codec.CodecError('native target names another source read')
        if binding['column'] == '*':
            import numpy as np

            return pd.Series(np.ones(len(rows), dtype='int64'))
        return self.table.frame[binding['column']].iloc[rows]

    def time_series(self, binding: dict[str, Any]) -> pd.Series:
        if (
            binding['source'] != self.table.source
            or binding['column'] != self.table.time_column
            or self.times is None
        ):
            raise codec.CodecError(
                'native time values name another captured source'
            )
        return self.times


class _Sources:
    def __init__(
        self,
        entity: SourceCompletion,
        windowed: SourceCompletion,
        preparation: _Preparation,
    ) -> None:
        self.entity = {
            first: _FetchedColumns(table, preparation)
            for first, table in entity.tables.items()
        }
        self.windowed = {
            first: _FetchedColumns(table, preparation)
            for first, table in windowed.tables.items()
        }

    def columns(self, view: Product) -> _FetchedColumns:
        producing = (
            self.entity
            if view.data.get('source_role') == 'entity'
            else self.windowed
        )
        return producing[view.data['source_read']]


class _Preparation:
    def __init__(self, provider: Any, state: dict[str, Any]) -> None:
        self.provider = provider
        self.state = state
        self.source: Product | None = None
        self.completion: SourceCompletion | None = None

    def prepare(
        self,
        value: dict[str, Any],
        bindings: Bindings,
        *,
        driver_class: str | None = None,
    ) -> Finished:
        attachments = {1: codec.attachment(value), **bindings.attachments}
        completion = None
        if self.source is not None:
            completion = SourceCompletion(self.provider, self.source, self)
        try:
            finished = run(
                {'op': 'prepare_materialize', 'input': 1},
                attachments,
                provider=self.provider,
                state=self.state,
                source_provider=completion,
                driver_class=driver_class,
            )
            self.completion = completion
            return finished
        except Exception as error:
            if completion is not None and error is completion.native_error:
                self.native_failure(NativeFrameFailure.capture(error))
            raise
        finally:
            attachments.clear()
            bindings.attachments.clear()
            if completion is not None and completion is not self.completion:
                completion.tables.clear()

    def simple(
        self, value: dict[str, Any], bindings: Bindings, expected: str
    ) -> Product:
        return adopt(self.prepare(value, bindings), expected)

    def column(
        self, binding: dict[str, Any], facts: dict[str, Any]
    ) -> dict[str, Any]:
        bindings = Bindings()
        if self.source is None:
            raise codec.CodecError(
                'native source conversion lacks its source owner'
            )
        return self.simple(
            {
                'stage': 'source_column',
                'source': bind(self.source, bindings),
                'column': binding,
                'input': facts,
            },
            bindings,
            'column_meaning',
        ).data

    def fallback(
        self, meaning: dict[str, Any], failure: NativeFrameFailure
    ) -> dict[str, Any]:
        bindings = Bindings()
        if self.source is None:
            raise codec.CodecError(
                'native source fallback lacks its source owner'
            )
        return self.simple(
            {
                'stage': 'source_fallback',
                'source': bind(self.source, bindings),
                'meaning': meaning,
                'failure': {
                    'builtin': failure.facts.as_wire()['builtin'],
                    'arithmetic': ArithmeticError
                    in type.__getattribute__(type(failure.error), '__mro__'),
                },
            },
            bindings,
            'column_meaning',
        ).data

    def lookup(
        self, meaning: dict[str, Any], helpers: pa.Table, owner: object
    ) -> pa.Table:
        bindings = Bindings()
        lookup = bind(
            Product('lookup', meaning, {meaning['owner']: owner}), bindings
        )
        product = self.simple(
            {
                'stage': 'lookup_rows',
                'lookup': lookup,
                'helpers': bindings.add(helpers),
            },
            bindings,
            'lookup_rows',
        )
        try:
            return pa.table(product.owners[product.data['positions']])
        finally:
            product.owners.clear()

    def read_failure(self, failure: SourceReadFailure) -> NoReturn:
        if self.source is None:
            raise codec.CodecError(
                'native source failure lacks its source owner'
            )
        observed = failure.step['request']
        fields, driver_class = _driver(failure.error, self.provider)
        bindings = Bindings()
        self.prepare(
            {
                'stage': 'source_failure',
                'source': bind(self.source, bindings),
                'read': {
                    'kind': observed['request'],
                    'source': observed['table'],
                    'columns': observed['columns'],
                },
                'error': fields,
            },
            bindings,
            driver_class=driver_class,
        )
        raise codec.CodecError(
            'an actual source failure returned successful data'
        )

    def target_column(
        self, inputs: Product, converted: ColumnExport
    ) -> Product:
        bindings = Bindings()
        return self.simple(
            {
                'stage': 'target_column',
                'inputs': bind(inputs, bindings),
                'converted': converted.bind(bindings),
            },
            bindings,
            'target_column',
        )

    def eligibility_column(self, inputs: Product, column: Product) -> Product:
        bindings = Bindings()
        return self.simple(
            {
                'stage': 'eligibility_column',
                'inputs': bind(inputs, bindings),
                'column': bind(column, bindings),
            },
            bindings,
            'label_inputs',
        )

    def what_if_column(self, groups: Product, column: Product) -> Product:
        bindings = Bindings()
        return self.simple(
            {
                'stage': 'what_if_column',
                'groups': bind(groups, bindings),
                'column': bind(column, bindings),
            },
            bindings,
            'label_groups',
        )

    def condition_column(
        self,
        column: Product,
        occurrence: dict[str, Any],
        converted: ColumnExport,
    ) -> Product:
        bindings = Bindings()
        return self.simple(
            {
                'stage': 'condition_column',
                'column': bind(column, bindings),
                'operand': {
                    'column': occurrence,
                    'converted': converted.bind(bindings),
                },
            },
            bindings,
            'target_column',
        )

    def filtered_target(
        self, column: Product, converted: ColumnExport
    ) -> Product:
        bindings = Bindings()
        return self.simple(
            {
                'stage': 'filtered_target',
                'column': bind(column, bindings),
                'converted': converted.bind(bindings),
            },
            bindings,
            'target_column',
        )

    def windowed_target(
        self, column: Product, converted: ColumnExport
    ) -> Product:
        bindings = Bindings()
        return self.simple(
            {
                'stage': 'windowed_target',
                'column': bind(column, bindings),
                'converted': converted.bind(bindings),
            },
            bindings,
            'target_column',
        )

    def native_failure(self, failure: NativeFrameFailure) -> NoReturn:
        guard_failure(failure, origin='engine')


def gather(
    operation: dict[str, Any],
    attachments: Mapping[int, Any],
    provider: Any,
    state: dict[str, Any],
    options: dict[str, Any],
) -> tuple[Finished, results.Gathered]:
    """Gather the actual task, returning its original terminal Run reader.

    Non-list and decomposed questions can return their actual nested request
    immediately. List populations retain source and completed native owners
    only for this call, until the request consumes their completed batch chain.
    """
    preparation = _Preparation(provider, state)
    products: list[Product] = []
    source_completions: list[SourceCompletion] = []
    bindings = Bindings()
    task = dict(operation)
    task.pop('op')
    task_attachments = dict(attachments)
    held = state.get('held')
    if held is not None:
        key = max(task_attachments, default=0) + 1
        task['sampler_state'] = {'version': held['version'], 'memo': key}
        task_attachments[key] = held['memo']
    # Preserve the original explicit task IDs while choosing a distinct
    # input-sized preparation attachment for the enclosing operation.
    input_id = max(task_attachments, default=0) + 1
    task_attachments[input_id] = codec.attachment(
        {'stage': 'task', 'request': task}
    )
    try:
        finished = run(
            {'op': 'prepare_materialize', 'input': input_id},
            task_attachments,
            provider=provider,
            state=state,
        )
        value = read(finished)
        if isinstance(value, results.Gathered):
            return finished, value
        if value.kind != 'source':
            raise codec.CodecError(
                'native SQL task returned no source or request'
            )
        products.append(value)
        preparation.source = value
        bindings = Bindings()
        population = preparation.simple(
            {'stage': 'population', 'source': bind(value, bindings)},
            bindings,
            'population',
        )
        products.append(population)
        entity_sources = preparation.completion
        if entity_sources is None:
            raise codec.CodecError(
                'native population has no producing source Run'
            )
        source_completions.append(entity_sources)
        batches: list[Product] = []
        start = 0
        while start < population.data['rows']:
            stop = min(
                start + population.data['batch_rows'], population.data['rows']
            )
            bindings = Bindings()
            inputs = preparation.simple(
                {
                    'stage': 'label_inputs',
                    'population': bind(population, bindings),
                    'range': {'start': start, 'stop': stop},
                    'previous': None
                    if not batches
                    else bind(batches[-1], bindings),
                },
                bindings,
                'label_inputs',
            )
            products.append(inputs)
            completion = preparation.completion
            if completion is None:
                raise codec.CodecError(
                    'native label input has no producing source Run'
                )
            source_completions.append(completion)
            complete_source_times(
                inputs, entity_sources, completion, preparation
            )
            sources = _Sources(entity_sources, completion, preparation)
            eligible = complete_eligibility(inputs, sources, preparation)
            products.append(eligible)
            column, native = complete_target(
                eligible, sources.columns(eligible), preparation
            )
            products.append(column)
            bindings = Bindings()
            groups = preparation.simple(
                {
                    'stage': 'native_label_groups',
                    'column': bind(column, bindings),
                },
                bindings,
                'label_groups',
            )
            products.append(groups)
            try:
                grouped = pa.table(groups.owners[groups.data['owner']].rows())
                target = grouped.column('target')
                values = target.to_pylist()
                if groups.data['target_encoding'] == 'utf8_surrogatepass':
                    values = [
                        [
                            value.decode('utf-8', 'surrogatepass')
                            for value in row
                        ]
                        for row in values
                    ]
                actual = native.group_result(values)
                restored_target, encoding = export_groups(
                    actual,
                    target.combine_chunks(),
                    text=groups.data['target_encoding'] is not None,
                )
                restored = grouped.set_column(
                    2, pa.field('target', restored_target.type), restored_target
                )
            except Exception as error:
                preparation.native_failure(NativeFrameFailure.capture(error))
            bindings = Bindings()
            restored_groups = preparation.simple(
                {
                    'stage': 'restore_label_groups',
                    'groups': bind(groups, bindings),
                    'restored': {
                        'rows': bindings.add(restored),
                        'encoding': encoding,
                    },
                },
                bindings,
                'label_groups',
            )
            products.append(restored_groups)
            final_groups = complete_what_if(
                restored_groups, sources, preparation
            )
            if final_groups is not restored_groups:
                products.append(final_groups)
            bindings = Bindings()
            batch = preparation.simple(
                {
                    'stage': 'complete_label_batch',
                    'population': bind(population, bindings),
                    'groups': bind(final_groups, bindings),
                    'restored': None,
                },
                bindings,
                'label_batch',
            )
            products.append(batch)
            batches.append(batch)
            for consumed in (
                inputs,
                eligible,
                column,
                groups,
                restored_groups,
                final_groups,
            ):
                consumed.owners.clear()
            completion.tables.clear()
            if batch.data['complete']:
                break
            start = stop
        bindings = Bindings()
        prepared = {
            'stage': 'request',
            'population': bind(population, bindings),
            'batches': [bind(batch, bindings) for batch in batches],
            'values': bindings.json(
                {'inference_config': options['inference_config']}
                if 'inference_config' in options
                else {},
                options=True,
            ),
        }
        finished = preparation.prepare(prepared, bindings)
        return finished, request(finished)
    finally:
        task_attachments.clear()
        bindings.attachments.clear()
        for product in products:
            product.owners.clear()
        for completion in source_completions:
            completion.tables.clear()
        preparation.source = None
        preparation.completion = None
