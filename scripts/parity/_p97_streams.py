# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input projections over retained source-owned parity generators."""

from __future__ import annotations

import base64
import importlib.util
import itertools
import json
import random
import sys
import tempfile
from pathlib import Path


def module(root, relative):
    path = root / relative
    for directory in (
        path.parent,
        root / 'tests/rust_core',
        root / 'tests/parity',
        root / 'scripts/parity',
    ):
        sys.path.insert(0, str(directory))
    name = '_p97_input_' + relative.replace('/', '_').replace('.', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('original input producer unavailable')
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def check_seed(actual, expected):
    if actual != expected:
        raise ValueError('stream seed differs from original producer')


def dialects(root, name, seed):
    filename, constructor = {
        'dialects.json': ('accept_json.py', 'cases'),
        'dialects.view_expressions': ('accept_view_expr.py', 'expressions'),
    }[name]
    producer = module(root, 'tests/rust_core/dialects/' + filename)
    check_seed(seed, producer.SEED)
    for value in getattr(producer, constructor)():
        if name == 'dialects.json':
            yield {
                'backend': value[0],
                'function': value[1],
                'arguments': value[2],
            }
        else:
            yield {
                'expression': value,
                'dialects': [None, 'databricks', 'snowflake'],
                'functions': ['_unsafe_expr_reason', '_untrusted_call_reason'],
                'untrusted_only_if_safe': True,
            }


def schema(root, name, seed, count):
    if name == 'schema.date_columns':
        producer = module(root, 'tests/rust_core/schema/_dates.py')
        check_seed(seed, producer.SEED)
        if count > producer.COUNT:
            raise ValueError('date-column population exceeded')
        for column in producer.columns(count, seed):
            yield {
                'column': column,
                'operations': list(producer.OPERATIONS),
                'successful_conversion_read': True,
            }
        return
    producer = module(root, 'tests/rust_core/schema/_views.py')
    if count > 10000:
        raise ValueError('view population exceeded')
    if name == 'schema.metric_views':
        check_seed(seed, producer.METRIC_SEED)
        for definition, trusted in producer.metric_views(count, seed):
            yield {'definition': definition, 'trust_expressions': trusted}
    elif name == 'schema.semantic_views':
        check_seed(seed, producer.SEMANTIC_SEED)
        for definition, tables, trusted in producer.semantic_views(count, seed):
            yield {
                'definition': definition,
                'tables': [
                    {'identity': list(key), 'columns': columns}
                    for key, columns in tables.items()
                ],
                'trust_expressions': trusted,
            }
    else:
        raise ValueError('unsupported schema stream')


def sampling(root, name, seed, count):
    producer = module(root, 'tests/rust_core/sampling/_sampling.py')
    if name in {'sampling.random_state', 'sampling.random_draws'}:
        check_seed(
            seed,
            producer.RNG_SEED + (7 if name == 'sampling.random_draws' else 0),
        )
        draws = random.Random(producer.RNG_SEED + 7)
        for value in producer.seeds():
            sample = {
                'seed': value,
                'operations': producer.rng_ops(draws),
                'modes': ['utils.random_state', 'RandomState.integer-or-words'],
            }
            if name == 'sampling.random_state':
                sample['words'] = producer.seed_words(value)
            yield sample
    elif name == 'sampling.random_word_lists':
        check_seed(seed, producer.RNG_SEED + 3)
        rng = random.Random(seed)
        words = [
            [
                rng.randrange(2**32)
                for _ in range(rng.choice((1, 2, 623, 624, 625, 1000)))
            ]
            for _ in range(300)
        ]
        words.extend([[0], [2**32 - 1], [0] * 700, [1, 0, 0]])
        draws = random.Random(producer.RNG_SEED + 7)
        for value in words:
            yield {'words': value, 'operations': producer.rng_ops(draws)}
    elif name == 'sampling.mt19937':
        check_seed(seed, producer.RNG_SEED + 1)
        for value, operations in producer.mt_jobs(producer.seeds()):
            yield {'seed': value, 'operations': operations}
    elif name == 'sampling.graphs':
        check_seed(seed, producer.GRAPH_SEED)
        if count > producer.NUM_GRAPHS:
            raise ValueError('graph population exceeded')
        yield from producer.graphs(count, seed)
    elif name == 'sampling.offset_pairs':
        check_seed(seed, producer.OFFSET_SEED)
        for value in producer.offset_pairs():
            yield {
                **value,
                'seconds_bounds': ['earliest', 'latest'],
                'nanoseconds': True,
            }
    elif name == 'sampling.offset_pooling':
        check_seed(seed, producer.OFFSET_SEED + 1)
        rng = random.Random(seed)
        for _ in range(5000):
            group = [
                producer.random_offset(rng) for _ in range(rng.randint(1, 4))
            ]
            if rng.random() < 0.3:
                group.append(dict(group[0]))
            yield {'offsets': group, 'earliest_variants': [True, False]}
    elif name == 'sampling.text':
        check_seed(seed, producer.TEXT_SEED)
        for value in producer.texts():
            yield {'values': [value], 'max_words_variants': [50, 3, None]}
    elif name == 'sampling.debug_jobs':
        debug = module(
            root, 'tests/rust_core/sampling/driver/accept_driver_debug.py'
        )
        check_seed(seed, debug.SEED)
        jobs = debug.jobs()
        prefix = 2 + len(debug.PAST_INT64) * 2
        for job in jobs[:prefix]:
            yield {'calls': [job]}
        for start in range(prefix, prefix + debug.NUM_OFFSETS * 4, 4):
            yield {'calls': jobs[start : start + 4]}
        for job in jobs[prefix + debug.NUM_OFFSETS * 4 :]:
            yield {'calls': [job]}
    else:
        raise ValueError('unsupported sampling stream')


def result_input(result):
    return {
        **{key: value for key, value in vars(result).items() if key != 'items'},
        'items': [vars(item).copy() for item in result.items],
    }


def requests(root, name, seed):
    if name == 'request.ranked_nan':
        producer = module(root, 'tests/rust_core/test_ranked_nan_order.py')
        check_seed(seed, producer.SEED)
        yield from producer._rankings()
        return
    if name == 'request.payload_size':
        producer = module(
            root, 'tests/rust_core/request/accept_request_size.py'
        )
        check_seed(seed, producer.SEED)
        rng = random.Random(seed)
        for _ in range(producer.COUNT):
            payload = producer._object(rng, 0)
            text = producer.spell(rng, payload)
            yield {'payload': payload, 'text': text, 'parsed': json.loads(text)}
        return
    producer = module(
        root, 'tests/rust_core/request/accept_request_generated.py'
    )
    if name == 'request.ranked':
        check_seed(seed, producer.RANKED_SEED)
        yield from producer._ranked_inputs()
        return
    results = producer.results.__wrapped__()
    if name == 'request.results':
        check_seed(seed, producer.RESULTS_SEED)
        yield from map(result_input, results)
    elif name == 'request.absent':
        check_seed(seed, producer.RESULTS_SEED + 1)
        rng = random.Random(seed)
        fields = [*producer.FIELDS, 'unknown_field', 'id', 'prediction']
        for result in results:
            yield {
                'result': result_input(result),
                'requested': [
                    rng.choice(fields) for _ in range(rng.randint(0, 6))
                ],
            }
    elif name == 'request.merge':
        check_seed(seed, producer.RESULTS_SEED + 2)
        rng = random.Random(seed)
        groups = {}
        for result in results:
            groups.setdefault((result.model, result.outputs), []).append(result)
        alike = [group for group in groups.values() if len(group) > 1]
        for _ in range(producer.MERGES):
            if rng.random() < 0.8 and alike:
                group = rng.choice(alike)
                parts = rng.sample(group, rng.randint(1, min(4, len(group))))
            else:
                parts = rng.sample(results, rng.randint(0, 3))
            yield {'parts': list(map(result_input, parts))}
    else:
        raise ValueError('unsupported request stream')


def labels(root, name, seed, count):
    producer = module(root, 'tests/rust_core/labels/_labels.py')
    if name == 'labels.aggregation_type':
        check_seed(seed, producer.LABELS_SEED)
        yield from producer.aggregation_type_inputs()
    elif name == 'labels.aggregation':
        check_seed(seed, producer.LABELS_SEED + 1)
        if count > 3000:
            raise ValueError('aggregation population exceeded')
        yield from producer.aggregation_inputs(count)
    elif name == 'labels.questions':
        check_seed(seed, producer.LABELS_SEED + 2)
        if count > 1500:
            raise ValueError('question population exceeded')
        schema = producer.shop_schema()
        for query in producer.generated_questions(count):
            yield {'query': query, 'schema': schema}
    elif name == 'labels.question_batches':
        check_seed(seed, producer.LABELS_SEED + 3)
        consumer = module(root, 'tests/rust_core/labels/accept_generated.py')
        from predictive_query.schema import json_serde

        rng = random.Random(seed)
        for query, schema, resolved in consumer._questions():
            yield {
                'query': query,
                'schema': schema,
                'resolved': json_serde.to_json_dict(resolved),
                'batches': [
                    producer.sample_data(rng, schema, resolved.entity_table)
                    for _ in range(2)
                ],
            }
    else:
        raise ValueError('unsupported label stream')


def pql(root, name, seed, count):
    producer = module(root, 'tests/rust_core/pql/_generate.py')
    if name == 'pql.date_literals':
        check_seed(seed, 20261003)
        if count > 100000:
            raise ValueError('date-literal population exceeded')
        for literal in producer.date_literals(seed, count):
            yield {'operation': 'pql.date', 'arguments': {'literal': literal}}
        return
    if name != 'pql.queries' or seed not in {20261002, 31, 32} or count > 50000:
        raise ValueError('unsupported query stream seed or population')
    helper = module(root, 'tests/rust_core/pql/_pql.py')
    generated = producer.queries(seed, count, helper.schemas())
    custom = (
        module(root, 'rust/crates/pql/tests/custom_profiles.py')
        if seed in {31, 32}
        else None
    )
    rng = random.Random(seed)
    for arguments in generated:
        operation = 'pql.validate'
        if custom is not None:
            arguments = {
                'query': arguments['query'],
                'schema': arguments['schema'],
                'mode': rng.choice(custom.MODES),
            }
            if seed == 31:
                operation = 'pql.profile'
                arguments['capabilities'] = {
                    'base': 'nothing',
                    'set': {
                        capability: bool(rng.getrandbits(1))
                        for capability in custom.CAPABILITIES
                    },
                    'model': custom.MODEL,
                }
            else:
                operation = 'pql.options'
                arguments.update(
                    allow_array_targets=bool(rng.getrandbits(1)),
                    allow_timestamp_arrays=bool(rng.getrandbits(1)),
                )
        yield {'operation': operation, 'arguments': arguments}


def sequences(root, seed):
    producer = module(root, 'tests/rust_core/rfm/test_sequences.py')
    check_seed(seed, producer.SEED)
    for value in producer._rows():
        yield {
            'calls': [
                {
                    'series': 'arrow',
                    'values': [value, '[1]'],
                    'dtype': dtype,
                    'stype': stype,
                }
                for dtype, stype in (
                    ('string', 'sequence'),
                    ('intlist', None),
                    ('stringlist', None),
                )
            ]
        }


def graph_input(producer, graph):
    import pyarrow as pa

    tables = {}
    for name, table in producer.arrow_tables(graph).items():
        buffer = pa.BufferOutputStream()
        with pa.ipc.new_stream(buffer, table.schema) as writer:
            writer.write_table(table)
        tables[name] = {
            'arrow_ipc_base64': base64.b64encode(
                buffer.getvalue().to_pybytes()
            ).decode('ascii')
        }
    return {'graph_json': producer.graph_json(graph), 'tables': tables}


def stores(root, name, seed):
    producer = module(root, 'tests/rust_core/sampling/_sampling.py')
    consumer = module(root, 'tests/rust_core/sampling/accept_graph_store.py')
    if name == 'sampling.graph_store':
        check_seed(seed, producer.STORE_SEED)
        for graph in consumer._graphs():
            yield graph_input(producer, graph)
        return
    check_seed(seed, producer.STORE_SEED + 1)
    rng = random.Random(seed)
    for graph in consumer._graphs():
        original = producer.python_store(graph)
        value = {'store': graph_input(producer, graph)}
        if 'error' in original:
            value.update(store_refused=True, calls=[])
        else:
            calls = consumer._calls(rng, original)
            value['sampler'] = {
                **producer.store_sampler_args(original, graph),
                'seed': 7,
                'calls': calls,
            }
        yield value


def inline_arrow(value, scratch):
    if isinstance(value, dict):
        return {key: inline_arrow(item, scratch) for key, item in value.items()}
    if isinstance(value, list):
        return [inline_arrow(item, scratch) for item in value]
    if isinstance(value, str) and value.startswith(str(scratch) + '/'):
        path = Path(value)
        if not path.is_file() or not path.resolve().is_relative_to(scratch):
            raise ValueError('Arrow input escapes original job directory')
        return {
            'arrow_ipc_base64': base64.b64encode(path.read_bytes()).decode()
        }
    return value


def hashseed(root, name, seed, array_count):
    producer = module(root, 'tests/rust_core/sampling/_jobs.py')
    sampling_source = module(root, 'tests/rust_core/sampling/_sampling.py')
    if name == 'sampling.hashseed_python_jobs':
        check_seed(seed, sampling_source.RNG_SEED + 11)
        yield from producer.python_jobs(500)
        return
    labels_source = module(root, 'tests/rust_core/labels/_labels.py')
    if seed not in {sampling_source.STORE_SEED, labels_source.LABELS_SEED + 9}:
        raise ValueError('unsupported Arrow-job stream seed')
    if type(array_count) is not int or array_count < 1280:
        raise ValueError('Arrow-job population requires at least 1280 jobs')
    kind = 'store' if seed == sampling_source.STORE_SEED else 'execute'
    with tempfile.TemporaryDirectory(prefix='p97-arrow-inputs-') as temporary:
        scratch = Path(temporary).resolve()
        for job in producer.array_jobs(array_count, scratch):
            if job['kind'] == kind:
                yield inline_arrow(job, scratch)


def population_size(root, name, seed, *, array_count=None):
    if name.startswith('dialects.'):
        filename, attribute = {
            'dialects.json': ('accept_json.py', 'cases'),
            'dialects.view_expressions': ('accept_view_expr.py', 'expressions'),
        }[name]
        producer = module(root, 'tests/rust_core/dialects/' + filename)
        check_seed(seed, producer.SEED)
        return len(getattr(producer, attribute)())
    if name == 'schema.date_columns':
        return module(root, 'tests/rust_core/schema/_dates.py').COUNT
    if name.startswith('schema.'):
        return module(root, 'tests/rust_core/schema/accept_views.py').COUNT
    if name.startswith('sampling.hashseed_'):
        return sum(1 for _ in hashseed(root, name, seed, array_count))
    if name.startswith('sampling.'):
        producer = module(root, 'tests/rust_core/sampling/_sampling.py')
        if name in {
            'sampling.random_state',
            'sampling.random_draws',
            'sampling.mt19937',
        }:
            return len(producer.seeds())
        if name == 'sampling.offset_pairs':
            return len(producer.offset_pairs())
        if name == 'sampling.debug_jobs':
            debug = module(
                root, 'tests/rust_core/sampling/driver/accept_driver_debug.py'
            )
            return (
                2
                + len(debug.PAST_INT64) * 2
                + debug.NUM_OFFSETS
                + debug.NUM_GRAPHS
            )
        if name == 'sampling.random_word_lists':
            return sum(1 for _ in sampling(root, name, seed, 0))
        if name == 'sampling.offset_pooling':
            return sum(1 for _ in sampling(root, name, seed, 0))
        return {
            'sampling.graphs': producer.NUM_GRAPHS,
            'sampling.graph_store': producer.NUM_STORES,
            'sampling.graph_store_calls': producer.NUM_STORES,
            'sampling.text': producer.NUM_TEXTS,
        }[name]
    if name.startswith('labels.'):
        producer = module(root, 'tests/rust_core/labels/_labels.py')
        if name == 'labels.aggregation_type':
            return len(producer.aggregation_type_inputs())
        if name == 'labels.aggregation':
            return len(producer.aggregation_inputs())
        if name == 'labels.questions':
            return len(producer.generated_questions(1500))
        return len(
            module(
                root, 'tests/rust_core/labels/accept_generated.py'
            )._questions()
        )
    if name.startswith('pql.'):
        return 100000 if name == 'pql.date_literals' else 50000
    if name == 'request.ranked_nan':
        return module(root, 'tests/rust_core/test_ranked_nan_order.py').CASES
    if name == 'request.payload_size':
        return module(
            root, 'tests/rust_core/request/accept_request_size.py'
        ).COUNT
    if name.startswith('request.'):
        producer = module(
            root, 'tests/rust_core/request/accept_request_generated.py'
        )
        attribute = {
            'request.results': 'RESULTS',
            'request.absent': 'RESULTS',
            'request.merge': 'MERGES',
            'request.ranked': 'RANKED',
        }[name]
        return getattr(producer, attribute)
    if name == 'rfm.sequences':
        return module(root, 'tests/rust_core/rfm/test_sequences.py').ROWS
    raise ValueError('unsupported original stream population')


def samples(root, name, seed, ordinals, *, array_count=None):
    count = max(ordinals) + 1
    family = name.partition('.')[0]
    if name in {'sampling.graph_store', 'sampling.graph_store_calls'}:
        generated = stores(root, name, seed)
    elif name.startswith('sampling.hashseed_'):
        generated = hashseed(root, name, seed, array_count)
    elif family == 'dialects':
        generated = dialects(root, name, seed)
    elif family == 'schema':
        generated = schema(root, name, seed, count)
    elif family == 'sampling':
        generated = sampling(root, name, seed, count)
    elif family == 'request':
        generated = requests(root, name, seed)
    elif family == 'labels':
        generated = labels(root, name, seed, count)
    elif family == 'pql':
        generated = pql(root, name, seed, count)
    elif name == 'rfm.sequences':
        generated = sequences(root, seed)
    else:
        raise ValueError('unsupported original stream')
    wanted = set(ordinals)
    selected = {
        ordinal: value
        for ordinal, value in enumerate(itertools.islice(generated, count))
        if ordinal in wanted
    }
    if set(selected) != wanted:
        raise ValueError('coordinates exceed original stream population')
    return selected
