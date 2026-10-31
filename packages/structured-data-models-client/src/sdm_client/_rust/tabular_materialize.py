# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native frame owners around the core's tabular row-selection policies."""

from __future__ import annotations

import warnings
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust.direct import _call
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError
from sdm_client.materializers.tabular import _asking, _is_whole, _rows
from sdm_client.request import Request


def _integer(value: int) -> dict[str, Any]:
    return {'kind': 'integer', 'value': str(value)}


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


def _passive_index(index: pd.Index) -> bool:
    if type(index) is pd.RangeIndex:
        return True
    if type(index) is pd.MultiIndex:
        return all(_passive_index(level) for level in index._levels)
    if type(index) not in (
        pd.Index,
        pd.DatetimeIndex,
        pd.TimedeltaIndex,
        pd.PeriodIndex,
        pd.CategoricalIndex,
    ):
        return False
    # Stored array identity only: do not iterate labels or invoke array hooks.
    return type(index._data) in (
        np.ndarray,
        pd.arrays.IntegerArray,
        pd.arrays.FloatingArray,
        pd.arrays.BooleanArray,
        pd.arrays.StringArray,
        pd.arrays.DatetimeArray,
        pd.arrays.TimedeltaArray,
        pd.arrays.PeriodArray,
        pd.Categorical,
    )


def _passive_frame(frame: pd.DataFrame) -> bool:
    return (
        type(frame) is pd.DataFrame
        and not frame.attrs
        and _passive_index(frame._mgr.axes[1])
    )


def _ordinary_selection(
    frame: pd.DataFrame,
    entity: str,
    target: str,
    requested: list[Any] | None,
    predict: pd.DataFrame | None,
    count: Any,
) -> dict[str, Any] | None:
    # A bulk read is permitted only where no native user operation separates
    # these observations. Everything else retains the staged native path.
    if not _passive_frame(frame) or type(frame.columns) is not pd.Index:
        return None
    if not all(type(name) is str for name in frame.columns):
        return None
    if not all(type(block.values) is np.ndarray for block in frame._mgr.blocks):
        return None
    if count is not None and (
        type(count) is not int or not 0 <= count <= np.iinfo(np.intp).max
    ):
        return None
    if entity not in frame.columns or target not in frame.columns:
        return None
    keys, labels = frame[entity], frame[target]
    if keys.dtype.kind not in 'iu' or labels.dtype.kind not in 'biuf':
        return None
    source = [_integer(value) for value in keys.tolist()]
    if predict is None:
        assert requested is not None
        if not all(
            type(value) in (int, bool) and -(2**127) <= value < 2**127
            for value in requested
        ):
            return None
        mode = {
            'kind': 'indices',
            'source': source,
            'requested': [_integer(int(value)) for value in requested],
        }
    else:
        if not _passive_frame(predict) or type(predict.columns) is not pd.Index:
            return None
        if not all(type(name) is str for name in predict.columns):
            return None
        if entity not in predict.columns or not all(
            type(block.values) is np.ndarray for block in predict._mgr.blocks
        ):
            return None
        scoring = predict[entity]
        if scoring.dtype != keys.dtype:
            return None
        mode = {
            'kind': 'supplied',
            'source': source,
            'scoring': [_integer(value) for value in scoring.tolist()],
        }
    return _call(
        'select_tabular_rows',
        {
            'mode': mode,
            'known': labels.notna().tolist(),
            'requested_context': count,
        },
    )


def _missing(indices: list[Any], ordinals: list[int], entity: str) -> None:
    values = [indices[ordinal] for ordinal in ordinals]
    shown = values[:10]
    tail = '' if len(values) == len(shown) else ', ...'
    raise StructuredDataError(
        f'{len(values)} of the {len(indices)} indices are not in '
        f'{entity}: {shown}{tail}. Pass predict=<frame> '
        f'to score rows not present in the table',
        code='INVALID_REQUEST',
    )


def _filtered(
    frame: pd.DataFrame,
    known: Any,
    excluded: list[int],
    *,
    assign: bool = False,
) -> pd.DataFrame:
    if (
        _passive_frame(frame)
        and type(known) is pd.Series
        and not known.attrs
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
    candidates: pd.DataFrame,
    wanted: int | None,
    table: str,
    remedy: str,
    decision: dict[str, Any] | None = None,
) -> pd.DataFrame:
    from sdm_client._rust.tabular_index import bind_profile, observe
    from sdm_client.models import TABULAR_LIMITS

    limit = TABULAR_LIMITS['context_rows_max']
    if decision is None:
        over_limit = wanted is not None and bool(wanted > limit)
        accepted = _call('check_tabular_cap', {'over_limit': over_limit})[
            'accepted'
        ]
        if not accepted:
            decision = {'kind': 'refused'}
        else:
            count = int(wanted) if isinstance(wanted, np.integer) else wanted
            decision = _call(
                'decide_tabular_context',
                {
                    'population': len(candidates),
                    'requested': count,
                },
            )
    if decision['kind'] == 'refused':
        raise StructuredDataError(
            f'num_context_samples is {wanted:,}, and the '
            f'endpoint accepts at most {limit:,} context rows',
            code='UNSUPPORTED_REQUEST',
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
    observation = observe(subset.index)
    ordering = _call('order_sampled_tabular_rows', observation)
    if ordering['kind'] == 'needs_profile':
        ordering = _call(
            'order_sampled_tabular_rows', bind_profile(observation)
        )
    if ordering['kind'] == 'refused':
        owner_index = subset.index
        for level in ordering.get('levels', []):
            if (
                type(owner_index) is not pd.MultiIndex
                or not 0 <= level < owner_index.nlevels
            ):
                raise AssertionError(
                    'comparison refusal has invalid level address'
                )
            owner_index = owner_index.levels[level]
        if ordering['reason'] == 'integer_to_float_overflow':
            raise OverflowError('int too large to convert to float')
        if ordering['reason'] == 'native_effects_pending':
            raise RouteError(
                'this NumPy comparison effect is not yet represented by the core'
            )
        if ordering['reason'] in (
            'timestamp_date_mismatch',
            'timestamp_awareness_mismatch',
            'temporal_year_out_of_range',
        ):
            # Rust selected the reached failure, including operand order.
            # Exact admitted owners supply only the runtime's diagnostic.
            left_owner = owner_index[ordering['left']]
            right_owner = owner_index[ordering['right']]
            if (
                type(left_owner) is not pd.Timestamp
                and type(right_owner) is not pd.Timestamp
            ):
                raise AssertionError('temporal refusal has invalid owners')
            if ordering['reason'] == 'temporal_year_out_of_range':
                timestamp = (
                    left_owner
                    if type(left_owner) is pd.Timestamp
                    else right_owner
                )
                timestamp.to_pydatetime()
            else:
                _ = left_owner < right_owner
            raise AssertionError('temporal comparison refusal did not raise')
        if ordering['reason'] == 'date_time_mismatch':
            # Rust selected the failing pair. Exact native owners present the
            # interpreter's diagnostic, which changed in Python 3.13.
            left_owner = owner_index[ordering['left']]
            right_owner = owner_index[ordering['right']]
            if {type(left_owner), type(right_owner)} != {date, datetime}:
                raise AssertionError(
                    'date comparison refusal has invalid owners'
                )
            _ = left_owner < right_owner
            raise AssertionError('date comparison refusal did not raise')
        if ordering['reason'] == 'naive_aware_datetime':
            raise TypeError(
                "can't compare offset-naive and offset-aware datetimes"
            )

        def native_name(position: int) -> str:
            owner = type(owner_index[position])
            # These exact built-in types use a qualified native type name in
            # Python's comparison diagnostic. Custom owners are not admitted.
            if owner.__module__ in ('decimal', 'datetime'):
                return f'{owner.__module__}.{owner.__name__}'
            return owner.__name__

        left = native_name(ordering['left'])
        right = native_name(ordering['right'])
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
    requested = list(indices) if predict is None else None
    selected = _ordinary_selection(
        frame, entity, target, requested, predict, num_context_samples
    )
    decision = None
    if selected is not None:
        if selected['kind'] == 'missing':
            assert requested is not None
            _missing(requested, selected['ordinals'], query.entity_column)
        decision = selected['context']
    if predict is None:
        assert requested is not None
        scored = (
            selected['scored']
            if selected is not None
            else _expand(frame[entity], requested, query.entity_column)
        )
        predict = frame.iloc[scored].drop(columns=[target])
        if selected is None:
            candidates = _filtered(
                frame, frame[target].notna(), scored, assign=True
            )
        else:
            mask = np.zeros(len(frame), dtype=bool)
            mask[selected['candidates']] = True
            candidates = frame[mask]
        remedy = 'Pass num_context_samples to choose how many, or select the rows yourself.'
    else:
        candidates = frame
        if selected is not None:
            mask = np.zeros(len(frame), dtype=bool)
            mask[selected['candidates']] = True
            candidates = frame[mask]
        elif entity in frame.columns and entity in predict.columns:
            keys = set(predict[entity])
            if keys:
                candidates = _filtered(frame, ~frame[entity].isin(keys), [])
        if selected is None and target in candidates:
            candidates = _filtered(candidates, candidates[target].notna(), [])
        remedy = (
            'This route takes no num_context_samples, so choose the '
            'context by building the schema from the rows you want.'
        )
    context = _context(candidates, num_context_samples, table, remedy, decision)
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
