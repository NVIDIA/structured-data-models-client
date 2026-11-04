#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Python benchmark child. Import/provenance probes run in separate processes."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

ENTRY_EPOCH = time.time()


def probe():
    import sdm_client

    versions = {
        dist.metadata['Name']: dist.version
        for dist in importlib.metadata.distributions()
    }
    packages = {}
    for name in (
        'sdm_client',
        'sdm_api',
        'sdm_connectors',
        'predictive_query',
        'sdm_relational_engine',
    ):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            packages[name] = str(Path(spec.origin).parent)
    spec = importlib.util.find_spec('sdm_client._native')
    native = (
        hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        if spec
        else None
    )
    return {
        'versions': versions,
        'dependency_sha256': hashlib.sha256(
            json.dumps(versions, sort_keys=True).encode()
        ).hexdigest(),
        'packages': packages,
        'native_sha256': native,
        'python': platform.python_version(),
        'client_file': sdm_client.__file__,
        'legacy_extensions': {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for package in packages.values()
            for path in Path(package).rglob('*.so')
        },
    }


def gradient_capable(client):
    """The public view export survives instance-only result attributes."""
    view = getattr(client, 'GradientExplanations', None)
    return view is not None and callable(getattr(view, 'to_pandas', None))


def run(args):
    imported = time.perf_counter()
    import pandas as pd
    import sdm_client
    from sdm_client import (
        KumoRelational,
        KumoTabular,
        PredictionResult,
        PredictiveQuery,
        Relationship,
        Request,
        Schema,
        SDMClient,
    )

    import_seconds = time.perf_counter() - imported
    fixed = os.environ['SDM_BENCH_FIXED'] == '1'
    model = (
        KumoTabular()
        if args.workload == 'tabular-session'
        else KumoRelational()
    )
    outputs = (
        ['prediction', 'explanation'] if args.gradients else ['prediction']
    )
    if args.gradients and not gradient_capable(sdm_client):
        print(
            json.dumps(
                {
                    'entry_epoch_seconds': ENTRY_EPOCH,
                    'status': 'unsupported',
                    'reason': 'This pinned client has no gradient result API',
                    'import_seconds': import_seconds,
                }
            )
        )
        return
    setup = time.perf_counter()
    users = pd.DataFrame(
        {
            'id': range(args.rows),
            **{
                f'x{column}': [
                    float((row + column) % 97) for row in range(args.rows)
                ]
                for column in range(args.width)
            },
            'y': [float(row % 31) for row in range(args.rows)],
        }
    )
    score_start = (
        args.rows
        if args.workload == 'tabular-session'
        else args.rows - args.scores
    )
    scoring = pd.DataFrame(
        {
            'id': range(score_start, score_start + args.scores),
            **{
                f'x{column}': [
                    float((row + column) % 97)
                    for row in range(score_start, score_start + args.scores)
                ]
                for column in range(args.width)
            },
        }
    )
    events = (
        None
        if args.workload == 'tabular-session'
        else pd.DataFrame(
            {
                'event_id': range(args.rows * args.degree),
                'id': [
                    row // args.degree for row in range(args.rows * args.degree)
                ],
                'value': [
                    float(row % 13) for row in range(args.rows * args.degree)
                ],
            }
        )
    )
    source = connection = None
    if args.workload == 'relational-sqlite':
        import adbc_driver_sqlite.dbapi
        import pyarrow as pa
        from sdm_connectors import SQLiteSource

        connection = adbc_driver_sqlite.dbapi.connect()
        with connection.cursor() as cursor:
            for name, frame in [('users', users), ('events', events)]:
                cursor.adbc_ingest(
                    name,
                    pa.Table.from_pandas(frame, preserve_index=False),
                    mode='create',
                )
        connection.commit()
        source = SQLiteSource(connection)
    client = SDMClient(url=os.environ['SDM_BENCH_URL'])
    setup_seconds = time.perf_counter() - setup
    results = []
    try:
        for iteration in range(1 + args.warmups + args.loops):
            values = {}

            def timed(name, call, values=values):
                start = time.perf_counter()
                result = call()
                values[name] = time.perf_counter() - start
                return result

            if args.workload == 'tabular-session':
                request = timed(
                    'request_prepare',
                    lambda: Request(
                        users, scoring, target='y', task='regression'
                    ),
                )
                fitted = timed(
                    'fit_http',
                    lambda request=request: client.fit(
                        request, model=model, outputs=outputs
                    ),
                )
                try:
                    result = timed(
                        'score_http_decode',
                        lambda fitted=fitted, request=request: fitted.predict(
                            request
                        ),
                    )
                finally:
                    timed('session_close_http', fitted.close)
            else:

                def schema_for_inputs():
                    schema = (
                        Schema.from_source(
                            source,
                            {'users': 'users', 'events': 'events'},
                            primary_keys={'users': 'id', 'events': 'event_id'},
                        )
                        if source
                        else Schema.from_data(
                            {'users': users, 'events': events},
                            primary_keys={'users': 'id', 'events': 'event_id'},
                        )
                    )
                    schema.relationships = [
                        Relationship('events', 'id', 'users', 'id')
                    ]
                    schema['users']['y'].stype = 'numerical'
                    for column in range(args.width):
                        schema['users'][f'x{column}'].stype = 'numerical'
                    return schema

                schema = timed('schema', schema_for_inputs)
                query = timed(
                    'pql',
                    lambda schema=schema: PredictiveQuery(
                        'PREDICT users.y FOR EACH users.id', schema
                    ),
                )
                request = timed(
                    'materialize_sample_encode',
                    lambda query=query: query.materialize(
                        model=model,
                        indices=list(range(args.rows - args.scores, args.rows)),
                        num_context_samples=min(args.rows - args.scores, 32),
                        random_seed=7,
                        num_neighbors=[args.degree],
                        num_hops=1,
                    ),
                )
                result = timed(
                    'predict_http_decode',
                    lambda request=request: client.predict(
                        request, model=model, outputs=outputs
                    ),
                )
            frame = timed('result_view', result.to_pandas)
            if len(frame) != args.scores or (
                fixed and not (frame['prediction'] == 1.0).all()
            ):
                raise AssertionError(
                    'fixed response row/value contract differs'
                )
            gradient_cells = 0
            if args.gradients:
                explanations = timed(
                    'gradient_view',
                    lambda result=result: result.explanations.to_pandas(),
                )
                gradient_cells = len(explanations)
                if (
                    not gradient_cells
                    or len(result.explanations) != args.scores
                    or (fixed and not (explanations['gradient'] == 0.25).all())
                ):
                    raise AssertionError('gradient response contract differs')
            if args.gradients:
                values['result_merge'] = None
            else:
                merged = timed(
                    'result_merge',
                    lambda result=result: PredictionResult.merge(
                        [result, result]
                    ),
                )
                if len(merged) != 2 * args.scores:
                    raise AssertionError('merge row count differs')
            results.append(
                {
                    'phase': 'cold-first'
                    if iteration == 0
                    else 'warmup'
                    if iteration <= args.warmups
                    else 'warm',
                    'seconds': values,
                    'gradient_cells': gradient_cells,
                }
            )
    finally:
        start = time.perf_counter()
        client.close()
        close_seconds = time.perf_counter() - start
        if connection:
            connection.close()
    highwater = None
    if platform.system() == 'Linux':
        import resource

        highwater = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    print(
        json.dumps(
            {
                'entry_epoch_seconds': ENTRY_EPOCH,
                'status': 'passed',
                'import_seconds': import_seconds,
                'setup_seconds': setup_seconds,
                'client_close_seconds': close_seconds,
                'iterations': results,
                'unavailable_phases': {
                    'result_merge': 'Duplicate-result gradient merging is outside the shared public scope of the pinned clients'
                }
                if args.gradients
                else {},
                'inputs': vars(args),
                'input_rows': {
                    'users': args.rows,
                    'scores': args.scores,
                    'events': 0
                    if args.workload == 'tabular-session'
                    else args.rows * args.degree,
                },
                'process_highwater_rss_bytes': highwater,
                'native_load': 'included in import or first public native operation; not separately observable',
                'sql_query_count': None if source else 0,
                'native_call_count': None,
                'counter_note': 'No production instrumentation; SQL/native counts unavailable for source execution',
                'gc': 'runtime default, no forced collection',
                'trace': 'disabled',
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--workload', default='tabular-session')
    for name, default in [
        ('rows', 1000),
        ('scores', 10),
        ('width', 4),
        ('degree', 4),
        ('warmups', 3),
        ('loops', 8),
        ('gradients', 0),
    ]:
        parser.add_argument('--' + name, type=int, default=default)
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe()))
    else:
        run(args)


if __name__ == '__main__':
    main()
