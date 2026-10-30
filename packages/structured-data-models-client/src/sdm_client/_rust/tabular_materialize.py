# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native frame owners around the core's tabular row-selection policies."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._rust.direct import _call
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.materializers.tabular import _asking, _is_whole, _rows
from sdm_client.request import Request


def _integer(value: int) -> dict[str, Any]:
    return {'kind': 'integer', 'value': value}


def _expand(source: pd.Series, indices: list[Any], entity: str) -> list[int]:
    # Dictionary equality and hashing consume the actual native key owners.
    # Its integer group identities are observations, not sampled positions.
    groups: dict[Any, int] = {}
    source_groups = [groups.setdefault(value, len(groups)) for value in source]
    present = [value in groups for value in indices]
    missing = _call('observe_tabular_membership', {'present': present})[
        'missing'
    ]
    if missing:
        values = [indices[ordinal] for ordinal in missing]
        shown = values[:10]
        count = len(values)
        tail = '' if count == len(shown) else ', ...'
        raise StructuredDataError(
            f'{count} of the {len(indices)} indices are not in '
            f'{entity}: {shown}{tail}. Pass predict=<frame> '
            f'to score rows not present in the table',
            code='INVALID_REQUEST',
        )
    # This second native lookup pass occurs only after missing-key refusal.
    requested = [groups[value] for value in indices]
    result = _call(
        'expand_tabular_keys',
        {
            'source': [_integer(value) for value in source_groups],
            'requested': [_integer(value) for value in requested],
        },
    )
    return result['rows']


def _filtered(
    frame: pd.DataFrame,
    known: Any,
    excluded: list[int],
    *,
    assign: bool = False,
) -> pd.DataFrame:
    if (
        type(frame) is pd.DataFrame
        and type(known) is pd.Series
        and known.dtype == np.dtype('bool')
    ):
        selected = _call(
            'filter_tabular_rows',
            {
                'known': known.tolist(),
                'excluded': excluded,
            },
        )['positions']
        # An ordinary Boolean Series has no user hooks; retain its native owner
        # and the reference's Boolean __getitem__ frame boundary.
        values = np.zeros(len(known), dtype=bool)
        values[selected] = True
        known.iloc[:] = values
    elif assign:
        # These positions already came from Rust's scoring expansion. Native
        # mask assignment must precede any inspection of a subclass mask.
        known.iloc[excluded] = False
    return frame[known]


def _context(
    candidates: pd.DataFrame, wanted: int | None, table: str, remedy: str
) -> pd.DataFrame:
    from sdm_client._rust.tabular_index import observe
    from sdm_client.models import TABULAR_LIMITS

    limit = TABULAR_LIMITS['context_rows_max']
    over_limit = wanted is not None and bool(wanted > limit)
    accepted = _call('check_tabular_cap', {'over_limit': over_limit})[
        'accepted'
    ]
    if not accepted:
        raise StructuredDataError(
            f'num_context_samples is {wanted:,}, and the '
            f'endpoint accepts at most {limit:,} context rows',
            code='UNSUPPORTED_REQUEST',
        )
    count = int(wanted) if isinstance(wanted, np.integer) else wanted
    decision = _call(
        'decide_tabular_context',
        {
            'population': len(candidates),
            'requested': count,
        },
    )
    if not decision['reduced']:
        return candidates
    if decision['advise_default_reduction']:
        count = decision['wanted']
        warnings.warn(
            f'{table} has {len(candidates):,} rows that could be context and '
            f'the endpoint accepts {count:,}, so {count:,} were '
            f'sampled. {remedy}',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    sampling_axis = candidates._get_axis_number(0)
    sampled = _call(
        'sample_tabular_rows',
        {
            'population': candidates.shape[sampling_axis],
            'wanted': decision['wanted'],
        },
    )
    if sampled['kind'] == 'refused':
        raise ValueError(
            "Cannot take a larger sample than population when 'replace=False'"
        )
    subset = candidates.take(
        np.asarray(sampled['offsets'], dtype=np.intp), axis=sampling_axis
    )
    ordering = _call('order_sampled_tabular_rows', observe(subset.index))
    if ordering['kind'] == 'refused':
        left = type(subset.index[ordering['left']]).__name__
        right = type(subset.index[ordering['right']]).__name__
        raise TypeError(
            f"'<' not supported between instances of '{left}' and '{right}'"
        )
    if ordering.get('already_ordered', False):
        return subset.copy(deep=False)
    # Native frame reconstruction mirrors sort_index after Rust has selected
    # its permutation. No native comparison or sorting policy runs here.
    axis = subset._get_block_manager_axis(0)
    data = subset._mgr.take(
        np.asarray(ordering['positions'], dtype=np.intp),
        axis=axis,
        verify=False,
    )
    data.set_axis(axis, data.axes[axis]._sort_levels_monotonic())
    result = subset._constructor_from_mgr(data, axes=data.axes)
    return result.__finalize__(subset, method='sort_index')


def _task(query: Any, target: pd.Series, stated: str | None) -> str:
    from sdm_client.materializers.tabular import TABULAR_TASK_KINDS

    stype = query.target_stype.value
    answer = _call(
        'decide_tabular_task', {'stage': 'target', 'target_stype': stype}
    )
    facts: dict[str, Any] = {
        'stage': 'kind',
        'target_stype': stype,
        'stated': None,
        'binary_whole': None,
    }
    if answer['decision'] == 'target_supported':
        if stated is not None:
            facts['stated'] = (
                isinstance(stated, str) and stated in TABULAR_TASK_KINDS
            )
        answer = _call('decide_tabular_task', facts)
    if answer['decision'] == 'explicit':
        assert stated is not None
        return stated
    if answer['decision'] == 'inspect_target':
        values = target.dropna().unique()
        facts['binary_whole'] = len(values) == 2 and all(
            _is_whole(value) for value in values
        )
        answer = _call('decide_tabular_task', facts)
    if answer['decision'] == 'refused':
        if answer['reason'] == 'unsupported_target':
            raise StructuredDataError(
                f'tabular materialization requires a categorical or '
                f'numerical target; {query.target_column!r} has semantic '
                f'type {query.target_stype.value!r}',
                code='UNSUPPORTED_REQUEST',
            )
        supported = ['classification', 'regression']
        raise StructuredDataError(
            f'the tabular model does not serve the task kind '
            f'{stated!r}; it serves {supported}',
            code='UNSUPPORTED_REQUEST',
            details={'task': stated, 'supported': supported},
        )
    if answer['warn_binary']:
        warnings.warn(
            f'{query.target_column} holds two whole numbers, which is how '
            f'a binary label is usually written, so this is being asked as '
            f'a classification. If they are quantities rather than labels, '
            f'pass task="regression" to ask for one; pass '
            f'task="classification" to say this was meant and silence '
            f'this.',
            category=SDMWarning,
            stacklevel=caller_stacklevel(),
        )
    return answer['kind']


def materialize(
    schema: Any,
    query: Any,
    *,
    indices: Any = None,
    predict: pd.DataFrame | None = None,
    num_context_samples: int | None = None,
    task: str | None = None,
) -> Request:
    from predictive_query.validator.utils import col_name, table_name

    table = table_name(query.entity_column)
    entity = col_name(query.entity_column)
    target = col_name(query.target_column)
    frame = _rows(schema[table])
    from sdm_client._rust.tabular_index import has_custom_sorter

    if has_custom_sorter(frame):
        # Owner-approved exception: preserve the entire existing custom-sort
        # journey, including hooks that may alter a subsequent prediction.
        from sdm_client.materializers.tabular import TabularMaterializer

        original = TabularMaterializer(schema)
        if predict is None:
            return original.indices(query, indices, num_context_samples, task)
        return original.table(query=query, predict=predict, task=task)
    if predict is None:
        requested = list(indices)
        scored = _expand(frame[entity], requested, query.entity_column)
        predict = frame.iloc[scored].drop(columns=[target])
        candidates = _filtered(
            frame, frame[target].notna(), scored, assign=True
        )
        remedy = 'Pass num_context_samples to choose how many, or select the rows yourself.'
    else:
        candidates = frame
        if entity in frame.columns and entity in predict.columns:
            keys = set(predict[entity])
            if keys:
                candidates = _filtered(frame, ~frame[entity].isin(keys), [])
        if target in candidates:
            candidates = _filtered(candidates, candidates[target].notna(), [])
        remedy = (
            'This route takes no num_context_samples, so choose the '
            'context by building the schema from the rows you want.'
        )
    context = _context(candidates, num_context_samples, table, remedy)
    return _asking(
        Request(
            context=context,
            predict=predict,
            target=target,
            task=_task(query, context[target], task),
            schema=schema.snapshot(),
            entity_table=table,
        ),
        query,
    )
