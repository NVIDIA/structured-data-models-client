# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one installed public Python journey; credentials stay in private config."""

import argparse
import contextlib
import copy
import datetime
import functools
import hashlib
import importlib
import json
import math
import pickle
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path

from shop_fixtures import load_fixture

from contract import (
    HERE,
    digest,
    effective_case,
    expectation,
    fixture_hash,
    manifest,
)


class Incomplete(Exception):
    pass


def scalar(value):
    import pandas as pd

    if value is pd.NA or value is pd.NaT:
        return None
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        if value.is_finite() and value == value.to_integral_value():
            return int(value)
        if value.is_finite() and Decimal.from_float(float(value)) == value:
            return float(value)
        return {'decimal': str(value)}
    if isinstance(value, float):
        return value if math.isfinite(value) else {'float': str(value)}
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar(v) for v in value]
    if hasattr(value, 'tolist') and getattr(value, 'ndim', 0) > 0:
        return scalar(value.tolist())
    if hasattr(value, 'item'):
        return scalar(value.item())
    raise TypeError(f'unsupported observation scalar: {type(value).__name__}')


def reader_schema(data):
    import pyarrow as pa

    def logical(dtype):
        if (
            pa.types.is_list(dtype)
            or pa.types.is_large_list(dtype)
            or pa.types.is_fixed_size_list(dtype)
        ):
            child = logical(dtype.value_type) + (
                '' if dtype.value_field.nullable else ' not null'
            )
            if pa.types.is_fixed_size_list(dtype):
                return f'fixed_size_list<item: {child}>[{dtype.list_size}]'
            kind = 'large_list' if pa.types.is_large_list(dtype) else 'list'
            return f'{kind}<item: {child}>'
        if pa.types.is_large_string(dtype):
            return 'string'
        if pa.types.is_large_binary(dtype):
            return 'binary'
        return str(dtype)

    return [
        [field.name, logical(field.type)]
        for field in pa.Schema.from_pandas(data, preserve_index=False)
    ]


def frame(data):
    return {
        'columns': list(data.columns),
        'rows': [
            [scalar(v) for v in row]
            for row in data.itertuples(index=False, name=None)
        ],
    }


@functools.lru_cache
def artifact(commit, phase, provenance=None):
    if phase != 'candidate':
        raise Incomplete('only current installed clients are supported')
    import sdm_client

    package = Path(sdm_client.__file__).resolve().parent
    # Hash the installed source, not a claimed checkout path or version string.
    sha = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        sha.update(str(path.relative_to(package)).encode())
        sha.update(path.read_bytes())
    out = {'commit': commit, 'package_hash': sha.hexdigest()}
    from sdm_client import _native

    out.update(
        native_hash=hashlib.sha256(
            Path(_native.__file__).read_bytes()
        ).hexdigest(),
        contract=_native.CONTRACT_VERSION,
    )
    if provenance is None:
        raise Incomplete('candidate requires verified wheel provenance')
    path = Path(provenance)
    identity = json.loads(path.read_text())
    if (
        identity.get('commit') != commit
        or identity.get('tree_clean') is not True
        or identity.get('features') != []
    ):
        raise Incomplete(
            'candidate provenance is not the requested clean production build'
        )
    wheel = path.parent / identity['wheel']
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != identity.get(
        'sha256'
    ) or out['native_hash'] != identity.get('native_sha256'):
        raise Incomplete('wheel/native artifact identity mismatch')
    wheel_python = hashlib.sha256()
    with zipfile.ZipFile(wheel) as archive:
        for name in sorted(
            name
            for name in archive.namelist()
            if name.startswith('sdm_client/') and name.endswith('.py')
        ):
            wheel_python.update(name.removeprefix('sdm_client/').encode())
            wheel_python.update(archive.read(name))
    if wheel_python.hexdigest() != out['package_hash']:
        raise Incomplete('installed Python differs from the verified wheel')
    return out


def tables_of(fixture):
    import pandas as pd

    tables = {
        name: pd.DataFrame(table['rows'], columns=table['columns'])
        for name, table in fixture['tables'].items()
    }
    # Explicit source types are applied only while constructing the fixture,
    # never to values returned by a public reader.
    import pyarrow as pa

    for name, table in fixture['tables'].items():
        for column, spec in table.get('types', {}).items():
            values = [
                row[table['columns'].index(column)] for row in table['rows']
            ]
            if spec.startswith('list<'):
                dtype = pa.list_(pa.type_for_alias(spec[5:-1]))
            elif spec.startswith('decimal128('):
                precision, scale = map(int, spec[11:-1].split(','))
                dtype = pa.decimal128(precision, scale)
                values = [
                    None if v is None else Decimal(str(v)) for v in values
                ]
            elif spec.startswith('timestamp['):
                parts = spec[10:-1].split(',', 1)
                dtype = pa.timestamp(
                    parts[0], tz=parts[1] if len(parts) == 2 else None
                )
                values = [
                    None if v is None else pd.Timestamp(v) for v in values
                ]
            elif spec == 'date32':
                dtype = pa.date32()
                values = [
                    datetime.date.fromisoformat(v) if v is not None else None
                    for v in values
                ]
            else:
                dtype = pa.type_for_alias(spec)
            tables[name][column] = pd.Series(
                pd.arrays.ArrowExtensionArray(pa.array(values, type=dtype))
            )
    time_column = fixture['tables']['events']['columns'][3]
    tables['events'][time_column] = pd.to_datetime(
        tables['events'][time_column]
    )
    return tables


def source_tables(stack, connector, config, fixture):
    """Remote fixtures are pre-provisioned; this runner never changes them."""
    import sdm_connectors as sources

    tables = tables_of(fixture)
    if connector == 'memory':
        return None, tables, None
    if connector in ('csv', 'parquet', 'zip', 's3-csv', 's3-parquet'):
        from sdm_connectors import read

        settings = config.get('sources', {}).get(connector, {})
        if connector.startswith('s3-') or settings.get('paths'):
            paths = settings.get('paths')
            if not paths:
                raise Incomplete('missing configured S3 fixture paths')
            tables = {
                name: read(
                    path, storage_options=settings.get('storage_options')
                )
                for name, path in paths.items()
            }
        else:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            for name, table in tables.items():
                path = directory / (
                    name
                    + (
                        '.parquet'
                        if connector == 'parquet'
                        else '.csv.zip'
                        if connector == 'zip'
                        else '.csv'
                    )
                )
                if connector == 'parquet':
                    import pyarrow as pa
                    import pyarrow.parquet as pq

                    # Provision a language-neutral file before the public read.
                    # Pandas list-extension metadata is not portable to its own
                    # numpy-nullable reader; the Arrow schema retains the type.
                    if any(
                        spec.startswith('list<')
                        for spec in fixture['tables'][name]
                        .get('types', {})
                        .values()
                    ):
                        data = pa.Table.from_pandas(table, preserve_index=False)
                        pq.write_table(data.replace_schema_metadata(None), path)
                    else:
                        table.to_parquet(path, index=False)
                else:
                    table.to_csv(path, index=False)
                tables[name] = read(path)
        return None, tables, None
    settings = config.get('sources', {}).get(connector, {})
    if connector == 'duckdb':
        connection = importlib.import_module('duckdb').connect()
        source_class = sources.DuckDBSource
    elif connector == 'sqlite':
        connection = importlib.import_module(
            'adbc_driver_sqlite.dbapi'
        ).connect()
        source_class = sources.SQLiteSource
    else:
        drivers = {
            'postgres': ('psycopg', sources.PostgresSource),
            'lakebase': ('psycopg', sources.PostgresSource),
            'databricks-sql': ('databricks.sql', sources.DatabricksSource),
            'databricks-metric': ('databricks.sql', sources.DatabricksSource),
            'snowflake': ('snowflake.connector', sources.SnowflakeSource),
            'snowflake-semantic': (
                'snowflake.connector',
                sources.SnowflakeSource,
            ),
        }
        if not settings.get('snapshot'):
            raise Incomplete('missing verified source snapshot configuration')
        if connector == 'databricks-spark':
            from databricks.connect import DatabricksSession

            builder = DatabricksSession.builder
            connect = settings.get('connect', {})
            if connect.get('profile'):
                builder = builder.profile(connect['profile'])
            if connect.get('cluster_id'):
                builder = builder.clusterId(connect['cluster_id'])
            connection = builder.getOrCreate()
            source_class = sources.DatabricksSource
        else:
            driver, source_class = drivers[connector]
            connection = importlib.import_module(driver).connect(
                **settings.get('connect', {})
            )
    if hasattr(connection, 'close'):
        stack.callback(connection.close)
    if connector in ('duckdb', 'sqlite'):
        for name, table in tables.items():
            if connector == 'sqlite':
                types = {
                    'id': 'INTEGER',
                    'event': 'INTEGER',
                    'item': 'INTEGER',
                    'value': 'DOUBLE',
                    'feature': 'DOUBLE',
                    'price': 'DOUBLE',
                    'class': 'TEXT',
                    'flag': 'INTEGER',
                    'event_time': 'TIMESTAMP',
                }
                with connection.cursor() as cursor:
                    columns = ', '.join(
                        '"' + column + '" ' + types[column]
                        for column in table.columns
                    )
                    cursor.execute(f'CREATE TABLE "{name}" ({columns})')
                    cursor.executemany(
                        f'INSERT INTO "{name}" VALUES ('
                        + ','.join('?' for _ in table.columns)
                        + ')',
                        frame(table)['rows'],
                    )
                connection.commit()
            else:
                connection.register('fixture_input', table)
                connection.execute(
                    f'CREATE TABLE "{name}" AS SELECT * FROM fixture_input'
                )
                connection.unregister('fixture_input')
    source = source_class(connection, **settings.get('source', {}))
    if connector in ('databricks-metric', 'snowflake-semantic'):
        source = source.semantic_model(
            settings['view'],
            trust_expressions=settings.get('trust_expressions', False),
        )
    return source, tables, connection


def fixture_nulls(value, declared):
    """Canonicalize only fixture-declared missing slots in detached evidence."""
    if declared is None and (value is None or value == {'float': 'nan'}):
        return None
    if (
        isinstance(declared, list)
        and isinstance(value, list)
        and len(value) == len(declared)
    ):
        return [
            fixture_nulls(v, d) for v, d in zip(value, declared, strict=True)
        ]
    return value


def check_snapshot(tables, fixture, *, raw_reader=False):
    # Independent source reads must agree with the fixture, including row
    # order normalized by the explicit primary key, before claiming its hash.
    for name, expected in tables_of(fixture).items():
        if name not in tables:
            raise Incomplete('fixture table missing: ' + name)
        key = fixture['primary_keys'][name]
        actual = tables[name].sort_values(key).reset_index(drop=True)
        wanted = expected.sort_values(key).reset_index(drop=True)
        observed, reference = frame(actual), frame(wanted)
        if raw_reader and observed['columns'] == reference['columns']:
            # Source identity proof only: these detached observations are not
            # passed to the schema or model, and raw reader evidence is retained.
            import pandas as pd

            source = fixture['tables'][name]
            key_at = source['columns'].index(key)
            declared_rows = {r[key_at]: r for r in source['rows']}
            for row, expected_row in zip(
                observed['rows'], reference['rows'], strict=True
            ):
                declared = declared_rows[expected_row[key_at]]
                for at in range(len(row)):
                    row[at] = fixture_nulls(row[at], declared[at])
                    expected_row[at] = fixture_nulls(
                        expected_row[at], declared[at]
                    )
                for at, (value, expected) in enumerate(
                    zip(row, expected_row, strict=True)
                ):
                    column = observed['columns'][at]
                    if (
                        column.lower() == 'class'
                        and isinstance(value, int)
                        and not isinstance(value, bool)
                    ):
                        if str(value) == expected:
                            row[at] = expected
                    elif (
                        column.lower() == 'flag'
                        and type(value) is int
                        and value in (0, 1)
                    ):
                        if type(expected) is bool and bool(value) == expected:
                            row[at] = expected
                    elif (
                        column.lower() == 'event_time'
                        or source.get('types', {})
                        .get(column, '')
                        .startswith('timestamp[')
                    ) and isinstance(value, str):
                        if pd.Timestamp(value) == pd.Timestamp(expected):
                            row[at] = expected
        if observed != reference:
            raise Incomplete(
                'source values differ from shared fixture: ' + name
            )


def request_observation(request):
    # Exercise public inspection without replacing the raw prediction evidence.
    for name, rows in [
        ('context', request.context_rows),
        ('predict', request.predict_rows),
    ]:
        split = getattr(request, name)
        if len(split.instance_table) != rows:
            raise AssertionError('inspected request row count differs: ' + name)
        for table in split.related_tables.values():
            frame(table)
    if request.payloads is not None:
        return scalar(request.payloads)
    return {
        'context': frame(request.context.instance_table),
        'predict': frame(request.predict.instance_table),
        'target': request.task.target.column_name,
        'task': request.task.kind,
    }


def journey(case, connector, model, config, mode, record):
    from sdm_client import (
        KumoRelational,
        KumoTabular,
        PredictiveQuery,
        Request,
        Schema,
        SDMClient,
    )
    from sdm_connectors import quote_ident, read_table

    fixture = load_fixture(HERE / case['fixture'])
    config = {
        **config,
        'sources': {
            **config.get('sources', {}),
            **config.get('fixture_sources', {}).get(case['fixture'], {}),
        },
    }
    user_key, target, category, flag = fixture['tables']['users']['columns'][:4]
    time_column = fixture['tables']['events']['columns'][3]
    chosen = KumoRelational() if model == 'relational' else KumoTabular()
    with contextlib.ExitStack() as stack:
        record['stage'] = 'source'
        source, tables, connection = source_tables(
            stack, connector, config, fixture
        )
        mappings = (
            config.get('sources', {})
            .get(connector, {})
            .get('tables', {name: name for name in tables})
        )
        if connection is not None:
            if connector == 'databricks-spark':
                namespace = config['sources'][connector].get('source', {})

                def qualified(actual):
                    parts = [
                        namespace.get('catalog'),
                        namespace.get('schema'),
                        actual,
                    ]
                    return '.'.join(
                        '`' + part.replace('`', '``') + '`'
                        for part in parts
                        if part
                    )

                tables = {
                    name: connection.table(qualified(actual)).toPandas()
                    for name, actual in mappings.items()
                }
            else:
                tables = {
                    name: read_table(
                        connection,
                        query='SELECT * FROM '
                        + quote_ident(
                            actual,
                            '`' if connector.startswith('databricks') else '"',
                        ),
                    )
                    for name, actual in mappings.items()
                }
        record['reader'] = {
            name: {
                'schema': reader_schema(table),
                'raw_schema': [
                    [str(column), str(dtype)]
                    for column, dtype in table.dtypes.items()
                ],
                'data': frame(table),
            }
            for name, table in tables.items()
        }
        if case.get('input_mode') != 'raw-reader':
            if not tables['users'][flag].isin([False, True, 0, 1]).all():
                raise Incomplete(
                    'fixture flag has values other than boolean/0/1'
                )
            tables['users'][category] = tables['users'][category].astype(str)
            tables['users'][flag] = tables['users'][flag].astype(bool)
            if time_column in tables.get('events', {}):
                tables['events'][time_column] = __import__(
                    'pandas'
                ).to_datetime(tables['events'][time_column])
            check_snapshot(tables, fixture)
            tables = {
                name: table.sort_values(
                    fixture['primary_keys'][name]
                ).reset_index(drop=True)
                for name, table in tables.items()
            }
        else:
            # Verification reads values but must not replace the reader's tables.
            check_snapshot(tables, fixture, raw_reader=True)
        record['source_snapshot'] = digest(fixture)
        record['stage'] = 'schema'
        keys = fixture['primary_keys']
        if model == 'tabular' and case['entry'] != 'pql-relational-schema':
            schema = Schema.from_data(
                {'users': tables['users']}, primary_keys={'users': user_key}
            )
        elif (
            source is None
            or model == 'tabular'
            or (
                case.get('input_mode') == 'raw-reader'
                and case.get('schema_input') != 'source'
            )
        ):
            schema = Schema.from_data(
                tables, primary_keys=keys, time_columns={'events': time_column}
            )
        else:
            schema = Schema.from_source(
                source,
                mappings,
                primary_keys=keys,
                time_columns={'events': time_column},
            )
        schema['users'][target].stype = 'numerical'
        for qualified, stype in case.get('semantic_types', {}).items():
            table_name, column = qualified.split('.', 1)
            schema[table_name][column].stype = stype
        # These links are explicit fixture facts, independent of catalog FK
        # enforcement and name-guessing differences across connectors.
        if model == 'relational' or case['entry'] == 'pql-relational-schema':
            from sdm_client import Relationship

            schema.relationships = [
                Relationship(*link[0].split('.'), *link[1].split('.'))
                for link in fixture['links']
            ]
        semantic = {
            'schema': {
                name: {
                    'primary_key': scalar(table.primary_key),
                    'time_column': table.time_column,
                    'columns': [
                        [col, table[col].dtype, table[col].stype]
                        for col in table
                    ],
                }
                for name, table in schema.tables.items()
            },
            'lifecycle': [],
            'results': [],
            'request': None,
        }
        record['semantic'] = semantic
        record['stage'] = 'materialize'
        if case['entry'] == 'direct':
            context = tables['users']
            prediction = context.iloc[:2].drop(columns=[target])
            request = Request(
                context,
                prediction,
                target=target,
                task='regression',
                entity_table='users',
                schema=schema,
            )
            semantic['request'] = request_observation(request)
        else:
            options = dict(case['options'])
            if model == 'tabular':
                options.pop('random_seed', None)
            request = PredictiveQuery(case['pql'], schema).materialize(
                model=chosen, **options
            )
            semantic['request'] = request_observation(request)
        semantic['lifecycle'].append('materialize')
        if case['journey'] == 'restore-predict':
            record['stage'] = 'restore'
            request = pickle.loads(pickle.dumps(request))
            semantic['lifecycle'].append('restore')
            semantic['request'] = request_observation(request)
        if mode == 'local':
            record['status'] = 'passed'
            return
        endpoint = (
            config.get('endpoints_by_case', {})
            .get(case['id'], {})
            .get(model, config.get('endpoints', {}).get(model))
        )
        if (
            not endpoint
            or endpoint.get('kind') != 'live'
            or not endpoint.get('version')
        ):
            raise Incomplete(
                'live endpoint/model-version configuration missing'
            )
        record['endpoint'] = {
            key: endpoint[key] for key in ('kind', 'version', 'transport')
        }
        record['endpoint']['model'] = endpoint.get('model', 'kumo-' + model)
        if endpoint['transport'] == 'databricks':
            client = SDMClient.for_databricks_serving(**endpoint['client'])
        elif endpoint['transport'] in ('nim', 'private-bridge'):
            client = SDMClient(**endpoint['client'])
        else:
            raise Incomplete('unsupported public endpoint transport')
        stack.callback(client.close)
        supplemental = case.get('supplement')
        before = (
            copy.deepcopy(request_observation(request))
            if supplemental
            else None
        )
        if supplemental:
            from supplements import (
                assert_holdout,
                holdout_key,
                observe_transport,
                result_details,
            )

            semantic['result_details'] = []
            semantic['transport'] = []
            stack.enter_context(
                observe_transport(
                    endpoint['client']['url'], semantic['transport']
                )
            )

        if case.get('input_mode') == 'raw-reader' and not supplemental:
            from supplements import observe_transport

            semantic['wire_calls'] = []
            stack.enter_context(
                observe_transport(
                    endpoint['client']['url'], semantic['wire_calls']
                )
            )

        def record_result(answer):
            semantic['results'].append(frame(answer.to_pandas()))
            if supplemental:
                assert not answer.absent(case['outputs']), (
                    'requested output is absent'
                )
                semantic['result_details'].append(
                    result_details(answer, scalar)
                )

        outputs = {'outputs': case['outputs']} if 'outputs' in case else {}
        if case['journey'] == 'fit-predict-close':
            held = [holdout_key(request)] if supplemental else [1, 2]
            if supplemental:
                semantic['holdout_keys'] = scalar(held)
            record['stage'] = 'fit'
            fitted = client.fit(request, model=chosen, holdout=held, **outputs)
            semantic['lifecycle'].append('fit')
            try:
                for _ in range(2):
                    record['stage'] = 'predict'
                    record_result(fitted.predict(request))
                    semantic['lifecycle'].append('predict')
            finally:
                record['stage'] = 'close'
                fitted.close()
                semantic['lifecycle'].append('close')
            if supplemental:
                assert_holdout(request, held, semantic['transport'])
        else:
            record['stage'] = 'predict'
            record_result(client.predict(request, model=chosen, **outputs))
            semantic['lifecycle'].append('predict')
            if supplemental:
                assert [
                    call['operation'] for call in semantic['transport']
                ] == ['predict'] * supplemental['prediction_calls']
                if supplemental['kind'] == 'batch':
                    assert (
                        len(request.payloads)
                        == supplemental['prediction_calls']
                    )
        if supplemental:
            assert request_observation(request) == before, (
                'prediction mutated the request'
            )
        record['status'] = 'passed'


def run(case, connector, model, config, phase, mode, commit, provenance=None):
    record = {
        'version': 1,
        'case': case['id'],
        'connector': connector,
        'model': model,
        'language': 'python',
        'phase': phase,
        'status': 'incomplete',
        'execution': mode,
        'fixture_hash': fixture_hash(case),
        'artifact': artifact(commit, phase, provenance),
        'source_snapshot': None,
        'endpoint': None,
        'semantic': None,
        'error': None,
    }
    try:
        journey(case, connector, model, config, mode, record)
    except ImportError:
        record['status'] = 'incomplete'
        record['reason'] = (
            'optional connector or runtime dependency is unavailable'
        )
    except Incomplete as error:
        record['status'] = 'incomplete'
        record['reason'] = str(error)
    except Exception as error:
        record['status'] = 'refused' if hasattr(error, 'code') else 'failed'
        fields = getattr(error, 'details', {}) or {}
        record['error'] = {
            'stage': record.get('stage', 'setup'),
            'code': getattr(error, 'code', type(error).__name__),
            'rule': fields.get('rule') if isinstance(fields, dict) else None,
        }
        # Exception text may contain source credentials or raw data; never log it.
    record.pop('stage', None)
    return record


def expected_outcome(case, record):
    expected = expectation(
        case,
        record['model'],
        'python',
        live=record.get('execution') != 'local',
    )
    if expected['status'] == 'supported':
        return record['status'] == 'passed'
    return record['status'] == expected.get('outcome', 'refused') and (
        record.get('error')
        == {key: expected[key] for key in ('stage', 'code', 'rule')}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=HERE / 'manifest.json')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['candidate'], default='candidate')
    parser.add_argument('--mode', choices=['local', 'live'], default='local')
    parser.add_argument('--case', action='append')
    parser.add_argument('--connector', action='append')
    parser.add_argument(
        '--model', choices=['relational', 'tabular'], action='append'
    )
    parser.add_argument('--commit', required=True)
    parser.add_argument('--provenance', type=Path)
    args = parser.parse_args()
    data = manifest(args.manifest)
    config = json.loads(args.config.read_text()) if args.config else {}
    results = []
    with args.out.open('w') as output:
        for case in data['scenarios']:
            if args.case and case['id'] not in args.case:
                continue
            for connector in case['connectors']:
                if args.connector and connector not in args.connector:
                    continue
                for model in case['models']:
                    if args.model and model not in args.model:
                        continue
                    record = run(
                        effective_case(case, connector),
                        connector,
                        model,
                        config,
                        args.phase,
                        args.mode,
                        args.commit,
                        str(args.provenance) if args.provenance else None,
                    )
                    output.write(
                        json.dumps(record, sort_keys=True, allow_nan=False)
                        + '\n'
                    )
                    output.flush()
                    results.append(
                        expected_outcome(
                            effective_case(case, connector), record
                        )
                    )
    return 0 if results and all(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
