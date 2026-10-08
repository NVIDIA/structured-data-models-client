# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticate generator inputs by independently invoking the retained producers."""

from __future__ import annotations

import csv
import io
import math
import sys
import warnings

import generator_consumers
import generator_descriptor
import generator_host
import generator_native
import generator_oracles
import generator_pql
import generator_primitives
import generator_sampling_extra
import generator_semantic
import generator_syntax
from support import (
    INVESTIGATIONS,
    Rejected,
    checked_path,
    require,
    same_json,
    sha,
)

PLAN = INVESTIGATIONS / 'p97-base-contract-staging/generator-plan.tsv'
DEPENDENCIES = (
    {
        'tests/rust_core/pql/_pql.py',
        'rust/crates/pql/tests/custom_profiles.py',
        'rust/crates/pql/tests/custom_profiles.json',
        'tests/rust_core/dialects/_dialects.py',
        'tests/rust_core/request/_request.py',
        'tests/parity/_fixture_ops.py',
        'tests/parity/pql/valid.json',
        'tests/parity/pql/invalid.json',
        'scripts/parity/_parity/values.py',
        'tests/rust_core/schema/accept_dates_series.py',
        'tests/rust_core/schema/accept_views.py',
        'tests/rust_core/schema/_schema.py',
        'tests/rust_core/schema/_warehouse.py',
        'tests/rust_core/sampling/accept_rng.py',
        'tests/rust_core/sampling/accept_offsets.py',
        'tests/rust_core/sampling/accept_text.py',
        'tests/rust_core/sampling/accept_hashseed.py',
        'tests/rust_core/sampling/mt19937_reference.cpp',
        'tests/rust_core/sampling/driver/_debug_run.py',
    }
    | set(generator_syntax.SOURCES)
    | generator_host.SOURCES
    | generator_descriptor.SOURCES
    | generator_semantic.SOURCES
)


def plan():
    return {
        row['id']: row
        for row in csv.DictReader(io.StringIO(PLAN.read_text()), delimiter='\t')
    }


def source_bindings(root):
    paths = DEPENDENCIES | {row['source'] for row in plan().values()}
    return {
        path: sha(checked_path(root, path).read_bytes())
        for path in sorted(paths)
    }


def typed(value):
    if isinstance(value, dict):
        return {
            'mapping': [
                [typed(key), typed(item)] for key, item in value.items()
            ]
        }
    if type(value) in {list, tuple}:
        return {
            'list' if type(value) is list else 'tuple': [
                typed(item) for item in value
            ]
        }
    if value is None or type(value) in {str, bool, int}:
        return {'scalar': value}
    if type(value) is float:
        return {'float': value.hex() if math.isfinite(value) else str(value)}
    import numpy as np
    import pandas as pd

    if isinstance(value, np.generic):
        return {
            'numpy_scalar': {
                'dtype': value.dtype.str,
                'bytes_hex': value.tobytes().hex(),
            }
        }
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return {
            'pandas_scalar': {
                'class': type(value).__name__,
                'value': typed(value.asm8),
                'timezone': str(value.tz)
                if isinstance(value, pd.Timestamp) and value.tz is not None
                else None,
                'fold': value.fold if isinstance(value, pd.Timestamp) else None,
            }
        }
    if isinstance(value, pd.Index):
        return {
            'index': {
                'class': type(value).__name__,
                'dtype': str(value.dtype),
                'names': typed(list(value.names)),
                'values': typed(value.tolist()),
                'levels': [typed(level) for level in value.levels]
                if isinstance(value, pd.MultiIndex)
                else None,
                'codes': [typed(code) for code in value.codes]
                if isinstance(value, pd.MultiIndex)
                else None,
            }
        }
    if isinstance(value, pd.DataFrame):
        return {
            'frame': {
                'columns': typed(value.columns),
                'index': typed(value.index),
                'attrs': typed(value.attrs),
                'series': [
                    typed(value.iloc[:, ordinal])
                    for ordinal in range(len(value.columns))
                ],
            }
        }
    if isinstance(value, pd.Series):
        return {
            'series': {
                'dtype': str(value.dtype),
                'name': typed(value.name),
                'index': typed(value.index),
                'attrs': typed(value.attrs),
                'values': typed(value.tolist()),
                'categories': typed(value.cat.categories)
                if isinstance(value.dtype, pd.CategoricalDtype)
                else None,
                'ordered': value.cat.ordered
                if isinstance(value.dtype, pd.CategoricalDtype)
                else None,
            }
        }
    if isinstance(value, np.ndarray):
        return {
            'ndarray': {
                'dtype': value.dtype.str,
                'descriptor': typed(value.dtype.descr),
                'shape': list(value.shape),
                'values': typed(value.tolist()),
            }
        }
    import _fixture_ops as operations

    return {'fixture_scalar': operations.encode(value)}


def consume(root, name, value, observer, native_variant):
    if name in generator_native.NATIVE_STREAMS:
        return generator_native.consume(
            root, name, value, observer, native_variant
        )
    if name.startswith('pql.'):
        return generator_oracles.pql(root, value, observer)
    if name == 'rfm.sequences':
        return generator_oracles.sequences(root, value, observer)
    family = name.partition('.')[0]
    require(
        family in {'dialects', 'schema', 'request', 'labels', 'sampling'},
        'generator.original_consumer_not_bound',
    )
    return getattr(generator_oracles, family)(root, name, value, observer)


def construct(
    root,
    name,
    stream_seed,
    ordinals,
    bindings,
    *,
    array_count=None,
    observe_outcomes=False,
    native_variant=None,
):
    inventory = plan()
    require(name in inventory, 'generator.original_family')
    require(
        type(stream_seed) is int
        and stream_seed
        in {int(value) for value in inventory[name]['stream_seeds'].split(',')},
        'generator.original_seed',
    )
    require(
        bool(ordinals)
        and all(type(value) is int and value >= 0 for value in ordinals),
        'generator.original_ordinal',
    )
    require(
        same_json(source_bindings(root), bindings),
        'generator.original_source_closure',
    )
    if name.startswith('pql.'):
        generated = generator_pql.samples(
            root,
            name,
            stream_seed,
            ordinals,
            {
                path: bindings[path]
                for path in generator_pql.source_bindings(root)
            },
        )['samples']
    elif name.startswith('dialects.'):
        generated = generator_primitives.dialects(
            root, name, stream_seed, ordinals
        )
    elif name.startswith('schema.'):
        generated = generator_primitives.schema(
            root, name, stream_seed, ordinals
        )
    elif name in {'sampling.graph_store', 'sampling.graph_store_calls'}:
        generated = generator_sampling_extra.stores(
            root, name, stream_seed, ordinals
        )
    elif name.startswith('sampling.hashseed_'):
        generated = generator_sampling_extra.hashseed(
            root, name, stream_seed, ordinals, array_count=array_count
        )
    elif name.startswith('sampling.'):
        generated = generator_primitives.sampling(
            root, name, stream_seed, ordinals
        )
    elif name.startswith('request.'):
        generated = generator_consumers.request(
            root, name, stream_seed, ordinals
        )
    elif name.startswith('labels.'):
        generated = generator_consumers.labels(
            root, name, stream_seed, ordinals
        )
    else:
        require(name == 'rfm.sequences', 'generator.original_family')
        generated = generator_consumers.sequences(root, stream_seed, ordinals)
    result = observe_samples(
        root,
        name,
        generated,
        observe_outcomes=observe_outcomes,
        native_variant=native_variant,
    )
    require(
        same_json(source_bindings(root), bindings),
        'generator.original_source_changed',
    )
    return result


def observe_samples(
    root, name, generated, *, observe_outcomes=True, native_variant=None
):
    result = {}
    for ordinal, value in generated.items():
        result[ordinal] = {'input': typed(value)}
        if name in generator_native.NATIVE_STREAMS:
            result[ordinal]['raw_input'] = value
        if observe_outcomes:
            observer = generator_oracles.OriginalCalls(root)
            actual, error = None, None
            captured_warnings = []
            try:
                with warnings.catch_warnings(record=True) as captured_warnings:
                    warnings.simplefilter('always')
                    actual = consume(
                        root, name, value, observer, native_variant
                    )
            except (Rejected, AssertionError, OSError):
                raise
            except Exception as raised:
                if name in generator_native.NATIVE_STREAMS:
                    raise
                sys.path.insert(
                    0, str(INVESTIGATIONS / 'p97-real-adapter-staging')
                )
                import _fixture_ops as operations
                from common import diagnostic

                error = diagnostic(raised, operations.encode)
            require(
                bool(observer.events)
                and (
                    name in generator_native.NATIVE_STREAMS
                    or all(
                        event['native_calls'] == 0 for event in observer.events
                    )
                ),
                'generator.original_consumer_route',
            )
            sys.path.insert(0, str(INVESTIGATIONS / 'p97-real-adapter-staging'))
            import _fixture_ops as operations
            from common import diagnostic

            outcome = {
                'status': 'success' if error is None else 'refused',
                'schema': [],
                'payloads': [],
                'sql': [],
                'result': typed(actual),
                'error': error,
                'warnings': [
                    diagnostic(item.message, operations.encode)
                    for item in captured_warnings
                ],
            }
            source_warnings = outcome['warnings']
            if name in generator_native.NATIVE_STREAMS:
                outcome['warnings'] = []
            result[ordinal].update(
                outcome=outcome,
                source_consumer_events=observer.events,
                public_calls=0,
                consumer_kind=generator_native.consumer_kind(name),
                source_warnings=source_warnings,
                safe_python=getattr(observer, 'safe_python', []),
            )
            if (
                name in generator_descriptor.FAMILIES
                and outcome['status'] == 'success'
            ):
                result[ordinal]['host_inputs_witness'] = (
                    generator_descriptor.witness(root, name, value)
                )
            if (
                name in generator_host.FAMILIES
                and outcome['status'] == 'success'
            ):
                result[ordinal]['host_inputs_witness'] = generator_host.witness(
                    root, name, value
                )
            if name == 'pql.queries':
                syntax = generator_syntax.witness(root, value, outcome)
                if syntax is not None:
                    result[ordinal]['pql_syntax_witness'] = syntax
    return result
