# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute original source consumers without claiming public sweep calls."""

from __future__ import annotations

import inspect
import os
import sys
import types
from functools import cache
from pathlib import Path

from generator_primitives import module
from support import require, sha


@cache
def source_codes(raw, filename):
    def children(code):
        yield code
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                yield from children(constant)

    return tuple(children(compile(raw, filename, 'exec', dont_inherit=True)))


class OriginalCalls:
    def __init__(self, root):
        self.root = root.resolve()
        self.events = []

    def invoke(self, function, *arguments, **keywords):
        code = function.__code__
        path = Path(code.co_filename).resolve()
        canonical = path
        if not path.is_relative_to(self.root):
            loaded = sys.modules.get(function.__module__)
            copied = Path(os.environ['P97_LEGACY_COPY_ROOT']).resolve()
            require(
                function.__module__.startswith('sdm_relational_engine.')
                and path.is_relative_to(copied)
                and loaded is not None
                and Path(loaded.__file__).resolve() == path
                and Path(loaded.__spec__.origin).resolve() == path,
                'generator.consumer_copy_identity',
            )
            matches = list(
                self.root.glob(
                    'packages/*/src/'
                    + function.__module__.replace('.', '/')
                    + '.py'
                )
            )
            require(
                len(matches) == 1
                and matches[0].read_bytes() == path.read_bytes(),
                'generator.consumer_copy_source',
            )
            canonical = matches[0]
        raw = path.read_bytes()
        source_hash = sha(raw)
        require(
            code in source_codes(raw, code.co_filename),
            'generator.consumer_code',
        )
        calls = 0
        native_calls = 0
        native = sys.modules.get('sdm_core._native')
        exports = (
            tuple(
                value
                for value in vars(native).values()
                if inspect.isbuiltin(value)
            )
            if native is not None
            else ()
        )
        previous = sys.getprofile()
        require(previous is None, 'generator.consumer_profile_busy')

        def observed(frame, event, value):
            nonlocal calls, native_calls
            if event == 'call' and frame.f_code is code:
                calls += 1
            elif event == 'c_call' and any(
                value is function for function in exports
            ):
                native_calls += 1

        sys.setprofile(observed)
        try:
            return function(*arguments, **keywords)
        finally:
            sys.setprofile(previous)
            require(
                calls > 0
                and sha(path.read_bytes()) == source_hash
                and sha(canonical.read_bytes()) == source_hash,
                'generator.consumer_not_observed_or_changed',
            )
            self.events.append(
                {
                    'path': canonical.relative_to(self.root).as_posix(),
                    'function': code.co_qualname
                    if hasattr(code, 'co_qualname')
                    else code.co_name,
                    'line': code.co_firstlineno,
                    'source_sha256': source_hash,
                    'operation_calls': calls,
                    'native_calls': native_calls,
                }
            )


def pql(root, value, observer):
    operation, arguments = value['operation'], value['arguments']
    if operation in {'pql.date', 'pql.validate'}:
        consumer = module(root, 'tests/rust_core/pql/_pql.py')
        function = (
            consumer.python_date
            if operation == 'pql.date'
            else consumer.python_result
        )
        return observer.invoke(function, **arguments)
    require(
        operation in {'pql.profile', 'pql.options'}, 'generator.pql_consumer'
    )
    consumer = module(root, 'rust/crates/pql/tests/custom_profiles.py')
    if operation == 'pql.profile':
        return observer.invoke(
            consumer.profile_answer,
            arguments['query'],
            arguments['schema'],
            arguments['mode'],
            arguments['capabilities'],
        )
    return observer.invoke(consumer.option_answer, **arguments)


def dialects(root, name, value, observer):
    if name == 'dialects.view_expressions':
        consumer = module(root, 'tests/rust_core/dialects/accept_view_expr.py')
        return observer.invoke(consumer._python, value['expression'])
    consumer = module(root, 'tests/rust_core/dialects/_dialects.py')
    if name == 'dialects.json':
        return observer.invoke(
            consumer.python_call,
            value['backend'],
            value['function'],
            value['arguments'],
        )
    return [
        observer.invoke(consumer.python_randint, call['arguments'])
        if call['function'] == 'utils.random_state.randint'
        else observer.invoke(
            consumer.python_call,
            call['backend'],
            call['function'],
            call['arguments'],
        )
        for call in value['calls']
    ]


def schema(root, name, value, observer):
    if name == 'schema.date_columns':
        consumer = module(root, 'tests/rust_core/schema/accept_dates_series.py')
        return observer.invoke(consumer._with_read, value['column'])
    if name == 'schema.metric_views':
        consumer = module(root, 'tests/rust_core/schema/_views.py')
        return observer.invoke(
            consumer.python_metric_view,
            value['definition'],
            value['trust_expressions'],
        )
    consumer = module(root, 'tests/rust_core/schema/accept_views.py')
    tables = {
        tuple(item['identity']): item['columns'] for item in value['tables']
    }
    return observer.invoke(
        consumer._python_semantic,
        value['definition'],
        tables,
        value['trust_expressions'],
    )


def restore_result(value):
    from sdm_api.generated.tfm_api import PredictionItem
    from sdm_client.result import PredictionResult

    result = PredictionResult(
        model=value['model'],
        request_id=value['request_id'],
        items=[PredictionItem(**item) for item in value['items']],
        metadata=value['metadata'],
        explanation=value['_explanation'],
        outputs=value['outputs'],
        unscored_candidates=value['_unscored_candidates'],
    )
    result._names_its_rows = value['_names_its_rows']
    result._names_its_step = value['_names_its_step']
    return result


def request(root, name, value, observer):
    if name in {'request.ranked', 'request.ranked_nan'}:
        import _fixture_ops as operations

        return observer.invoke(operations.run, 'client.ranked', value)
    if name == 'request.payload_size':
        from sdm_relational_engine.rfm.inference.payload import (
            payload_size_bytes,
        )

        return {
            'parsed': observer.invoke(payload_size_bytes, value['parsed']),
            'original': observer.invoke(payload_size_bytes, value['payload']),
        }
    consumer = module(
        root, 'tests/rust_core/request/accept_request_generated.py'
    )
    if name == 'request.results':
        return observer.invoke(consumer._frame_or_error, restore_result(value))
    if name == 'request.absent':
        return observer.invoke(
            restore_result(value['result']).absent, value['requested']
        )
    require(name == 'request.merge', 'generator.request_consumer')
    from sdm_client.result import PredictionResult

    merged = observer.invoke(
        PredictionResult.merge,
        [restore_result(part) for part in value['parts']],
    )
    return {
        'attributes': observer.invoke(consumer._attributes, merged),
        'frame': observer.invoke(consumer._frame_or_error, merged),
    }


def labels(root, name, value, observer):
    consumer = module(root, 'tests/rust_core/labels/_labels.py')
    if name in {'labels.aggregation_type', 'labels.aggregation'}:
        return observer.invoke(consumer.python_op, name, value)
    resolved = observer.invoke(
        consumer.resolve, value['query'], value['schema']
    )
    if name == 'labels.questions':
        if resolved is None:
            return {'resolved': None}
        from predictive_query.schema import json_serde

        return {'resolved': observer.invoke(json_serde.to_json_dict, resolved)}
    require(
        name == 'labels.question_batches' and resolved is not None,
        'generator.label_consumer',
    )
    return [
        observer.invoke(consumer.python_execute, resolved, batch)
        for batch in value['batches']
    ]


def sampling(root, name, value, observer):
    consumer = module(root, 'tests/rust_core/sampling/_sampling.py')
    if name in {'sampling.random_state', 'sampling.random_draws'}:
        import numpy as np
        from sdm_relational_engine.rfm.base.utils import random_state

        seed = value['seed']
        states = [
            observer.invoke(random_state, seed),
            np.random.RandomState(
                seed
                if seed < 2**32
                else np.array(consumer.seed_words(seed), dtype=np.uint32)
            ),
        ]
        return [
            observer.invoke(consumer.numpy_draws, state, value['operations'])
            for state in states
        ]
    if name == 'sampling.random_word_lists':
        import numpy as np

        return observer.invoke(
            consumer.numpy_draws,
            np.random.RandomState(np.array(value['words'], dtype=np.uint32)),
            value['operations'],
        )
    if name == 'sampling.mt19937':
        return observer.invoke(
            consumer.cpp_draws, [[value['seed'], value['operations']]]
        )[0]
    if name == 'sampling.graphs':
        return observer.invoke(
            consumer.run_sampler,
            consumer.relationallib().NeighborSampler,
            value,
        )
    if name == 'sampling.graph_store':
        return observer.invoke(consumer.python_store, value.original_graph)
    if name == 'sampling.graph_store_calls':
        store = observer.invoke(
            consumer.python_store, value['store'].original_graph
        )
        if 'error' in store:
            return {'store': store, 'sampled': None}
        return {
            'store': store,
            'sampled': observer.invoke(
                consumer.run_sampler,
                consumer.relationallib().NeighborSampler,
                value['sampler'],
            ),
        }
    if name == 'sampling.offset_pairs':
        from sdm_relational_engine.rfm.base.utils import date_offset_to_seconds

        current = list(value['timestamps'])
        alive = list(range(len(current)))
        steps = []
        for _ordinal in range(value['repeat']):
            outcomes = [
                observer.invoke(
                    consumer.python_apply,
                    current[index],
                    value['offset'],
                    value['sign'],
                )
                for index in alive
            ]
            steps.append({'indices': list(alive), 'outcomes': outcomes})
            remaining = []
            for index, outcome in zip(alive, outcomes, strict=True):
                if 'ns' in outcome:
                    current[index] = outcome['ns']
                    remaining.append(index)
            alive = remaining
            if not alive:
                break
        return {
            'steps': steps,
            'seconds': [
                observer.invoke(
                    date_offset_to_seconds,
                    consumer.offset_of(value['offset']),
                    bound,
                )
                for bound in value['seconds_bounds']
            ],
            'nanoseconds': observer.invoke(
                consumer.python_to_ns, value['offset']
            ),
        }
    if name == 'sampling.offset_pooling':
        return [
            observer.invoke(consumer.python_pool, value['offsets'], earliest)
            for earliest in value['earliest_variants']
        ]
    require(name == 'sampling.text', 'generator.sampling_consumer')
    return [
        observer.invoke(consumer.python_normalize, value['values'], maximum)
        for maximum in value['max_words_variants']
    ]


def sequences(root, value, observer):
    consumer = module(root, 'tests/rust_core/rfm/test_sequences.py')
    return [observer.invoke(consumer._python, call) for call in value['calls']]
