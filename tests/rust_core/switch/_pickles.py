# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P9.pickle: schemas, requests and results through pickle and copy.

The objects are those the public calls of a few in-memory transcripts make,
in order, answered by the stand-in model. Each is described through the
public API only (``describe``), as it is, after ``pickle``, ``copy.copy``,
``copy.deepcopy`` and, for a schema, ``snapshot()``; a schema's copies are
then edited (their relationships cleared), and the original and the copy
described again. What it does is described too: a schema materializes the
transcript's question, a request is predicted, a result is turned into a
frame.

    python _pickles.py make --out DIR   # at BASE: the pickles, expected.json
    python _pickles.py fresh --out FILE # today: the same, made afresh
    python _pickles.py old --out FILE   # today: BASE's pickles, read back
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _switch import (  # noqa: E402
    TRANSCRIPTS,
    add_paths,
    canonical,
    digest,
    ported_loaded,
    qualname,
)

PICKLES = HERE / 'pickles'
EXPECTED = PICKLES / 'expected.json'

#: The transcripts whose objects are made, each in memory.
SOURCES = (
    'memory/classification',
    'memory/forecast--explanations-by-step',
    'memory/ranking--unscored-candidate',
    'memory/link-prediction',
    'tabular/direct-classification',
    'tabular/batches',
    'tabular/pql-predict-frame',
)


def _attempt(fn: Any) -> Any:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return {'value': fn()}
    except Exception as error:  # noqa: BLE001 - an outcome like any other
        return {'raised': qualname(type(error)), 'message': str(error)[:500]}


def describe_frame(frame: Any) -> Any:
    return {
        'columns': [str(c) for c in frame.columns],
        'dtypes': [str(t) for t in frame.dtypes],
        'index': digest(list(frame.index)),
        'values': digest(frame.to_dict('split')['data']),
    }


def describe_schema(schema: Any) -> Any:
    tables = []
    for name, table in schema.tables.items():
        described = {
            'name': name,
            'type': qualname(type(table)),
            'primary_key': canonical(table.primary_key),
            'time_column': table.time_column,
            'duplicate_keys': _attempt(
                lambda t=table: canonical(t.duplicate_keys)
            ),
            'columns': [
                [
                    str(cname),
                    str(col.dtype),
                    str(col.stype),
                    _attempt(lambda c=col: canonical(c.stype_declared)),
                    _attempt(lambda c=col: canonical(c.untyped)),
                ]
                for cname, col in table.columns.items()
            ],
        }
        if hasattr(table, 'data'):
            described['data'] = describe_frame(table.data)
        if hasattr(table, 'source_name'):
            described['source_name'] = table.source_name
        tables.append(described)
    return {
        'type': qualname(type(schema)),
        'tables': tables,
        'relationships': [
            [r.source_table, r.source_column, r.target_table, r.target_column]
            for r in schema.relationships
        ],
        'source_is_none': schema.source is None,
        'repr': digest(repr(schema)),
        'check_consistent': _attempt(
            lambda: canonical(schema.check_consistent())
        ),
    }


def _normalized(payloads: Any) -> Any:
    out = json.loads(json.dumps(list(payloads), default=repr))
    for payload in out:
        metadata = (
            payload.get('metadata') if isinstance(payload, dict) else None
        )
        if isinstance(metadata, dict) and 'request_id' in metadata:
            metadata['request_id'] = 'request'
    return out


def _split(split: Any) -> Any:
    return {
        'instance_table': describe_frame(split.instance_table),
        'related_tables': {
            str(k): describe_frame(v) for k, v in split.related_tables.items()
        },
    }


def describe_request(request: Any) -> Any:
    def decomposition() -> Any:
        d = request.decomposition
        if d is None:
            return None
        return {'kind': d.kind, 'top': d.top, 'parts': canonical(list(d.parts))}

    return {
        'type': qualname(type(request)),
        'context': _attempt(lambda: _split(request.context)),
        'predict': _attempt(lambda: _split(request.predict)),
        'context_rows': _attempt(lambda: canonical(request.context_rows)),
        'predict_rows': _attempt(lambda: canonical(request.predict_rows)),
        'asked': _attempt(lambda: canonical(request.asked)),
        'is_relational': _attempt(lambda: request.is_relational),
        'nothing_to_score': _attempt(
            lambda: canonical(request.nothing_to_score)
        ),
        'query': _attempt(lambda: canonical(request.query)),
        'target_moves_with_time': _attempt(
            lambda: canonical(request.target_moves_with_time)
        ),
        'decomposition': _attempt(decomposition),
        'parts': _attempt(
            lambda: None if request.parts is None else len(request.parts)
        ),
        'payloads': _attempt(
            lambda: (
                None
                if request.payloads is None
                else digest(_normalized(request.payloads))
            )
        ),
    }


def describe_result(result: Any) -> Any:
    return {
        'type': qualname(type(result)),
        'model': result.model,
        'request_id': result.request_id,
        'metadata': canonical(result.metadata),
        'outputs': _attempt(lambda: canonical(result.outputs)),
        'unscored_candidates': canonical(list(result.unscored_candidates)),
        'explanation': canonical(result.explanation),
        'frame': _attempt(lambda: describe_frame(result.to_pandas())),
    }


def describe(obj: Any) -> Any:
    from sdm_client import PredictionResult, Request, Schema

    if isinstance(obj, Schema):
        return describe_schema(obj)
    if isinstance(obj, Request):
        return describe_request(obj)
    if isinstance(obj, PredictionResult):
        return describe_result(obj)
    raise AssertionError(f'cannot describe a {type(obj).__name__}')


# --- What each object does ---------------------------------------------------


class Made:
    r"""One object a transcript's calls made, and what to do with it."""

    def __init__(
        self,
        name: str,
        obj: Any,
        source: str,
        transcript: dict[str, Any],
        at: int,
    ) -> None:
        self.name = name
        self.obj = obj
        self.source = source
        self.transcript = transcript
        self.at = at


_RUNS: list[Any] = []


def _source_runs() -> Any:
    r"""The process's ``Source._run`` counter."""
    if not _RUNS:
        from _switch import SourceRuns

        runs = SourceRuns()
        runs.install()
        _RUNS.append(runs)
    return _RUNS[0]


def _client(transcript: dict[str, Any]) -> Any:
    from _parity.nim import SyntheticSession
    from sdm_client import SDMClient

    session = SyntheticSession(transcript.get('stand_in'))
    client = SDMClient(url='http://localhost:8000')
    client._transport._session.close()
    client._transport._session = session
    return client, session


def behavior(made: Made, obj: Any) -> Any:
    r"""What the object does: a schema asks the transcript's next question,
    a request is predicted, a result becomes a frame."""
    from sdm_client import PredictionResult, PredictiveQuery, Request, Schema

    calls = made.transcript['calls']
    if isinstance(obj, Schema):
        query = next(
            (c for c in calls[made.at :] if c['call'] == 'PredictiveQuery'),
            None,
        )
        materialize = next(
            (
                c
                for c in calls[made.at :]
                if c['call'] == 'PredictiveQuery.materialize'
            ),
            None,
        )
        if query is None or materialize is None:
            return None
        interp = _interpreter(made.transcript)

        def ask() -> Any:
            args = {k: interp.value(v) for k, v in materialize['args'].items()}
            runs = _source_runs()
            runs.calls = []
            request = PredictiveQuery(query['args']['query'], obj).materialize(
                **args
            )
            return {
                'payloads': digest(_normalized(request.payloads or ())),
                'source_runs': len(runs.calls),
            }

        return _attempt(ask)
    if isinstance(obj, Request):
        predict = next(
            (c for c in calls[made.at :] if c['call'] == 'SDMClient.predict'),
            None,
        )
        if predict is None:
            return None

        def send() -> Any:
            client, session = _client(made.transcript)
            try:
                result = client.predict(
                    obj,
                    model=predict['args']['model'],
                    outputs=predict['args'].get('outputs'),
                )
            finally:
                client.close()
            return {
                'bodies': digest(_normalized(session.payloads)),
                'result': describe_result(result),
            }

        return _attempt(send)
    if isinstance(obj, PredictionResult):
        return _attempt(
            lambda: describe_result(PredictionResult.merge([obj, obj]))
        )
    return None


def _interpreter(transcript: dict[str, Any]) -> Any:
    from _parity.runner import _Interpreter

    client, session = _client(transcript)
    client.close()
    return _Interpreter(transcript, None, session)


def made_objects() -> list[Made]:
    r"""Every schema, request and result the source transcripts' calls make,
    answered by the stand-in model."""
    from _parity.transcript import load

    out = []
    for name in SOURCES:
        transcript = load(TRANSCRIPTS / name / 'transcript.json')
        interp = _interpreter(transcript)
        client, _ = _client(transcript)
        interp.client = client
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            seen: set[int] = set()
            for at, call in enumerate(transcript['calls']):
                args = {
                    k: interp.value(v) for k, v in call.get('args', {}).items()
                }
                getattr(interp, '_' + call['call'].replace('.', '_'))(**args)
                for kind in ('schema', 'request', 'result'):
                    obj = getattr(interp, kind)
                    if obj is not None and id(obj) not in seen:
                        seen.add(id(obj))
                        out.append(
                            Made(
                                f'{name}#{at}-{kind}',
                                obj,
                                name,
                                transcript,
                                at + 1,
                            )
                        )
            interp.close()
    out.append(_from_source())
    return out


def _from_source() -> Made:
    r"""A schema read from a DuckDB holding the shop of
    ``memory/classification``: what it caches of the source is what keeps
    the reads of its first question down (research risk 6)."""
    import duckdb
    from _parity.transcript import load
    from sdm_client import Schema
    from sdm_connectors import DuckDBSource

    name = 'memory/classification'
    transcript = load(TRANSCRIPTS / name / 'transcript.json')
    interp = _interpreter(transcript)
    connection = duckdb.connect()
    connection.execute('SET threads TO 1')
    for table in ('users', 'products', 'orders'):
        connection.register('frame', interp.frames[table])
        connection.execute(f'CREATE TABLE {table} AS SELECT * FROM frame')
        connection.unregister('frame')
    runs = _source_runs()
    runs.calls = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
    made = Made('duckdb-source#0-schema', schema, name, transcript, 1)
    made.reads = len(runs.calls)  # type: ignore[attr-defined]
    return made


def round_trips(made: Made) -> dict[str, Any]:
    from sdm_client import Schema

    obj = made.obj
    out: dict[str, Any] = {
        'original': describe(obj),
        'reads': getattr(made, 'reads', None),
        'behavior': behavior(made, obj),
    }
    makers = {
        'pickle': lambda: pickle.loads(pickle.dumps(obj)),
        'copy': lambda: copy.copy(obj),
        'deepcopy': lambda: copy.deepcopy(obj),
    }
    if isinstance(obj, Schema):
        makers['snapshot'] = obj.snapshot
    for how, make in makers.items():
        made_copy = _attempt(make)
        if 'raised' in made_copy:
            out[how] = made_copy
            continue
        twin = made_copy['value']
        entry: dict[str, Any] = {
            'described': describe(twin),
            'behavior': behavior(made, twin),
        }
        if isinstance(obj, Schema) and twin.relationships:

            def clear(t: Any = twin) -> Any:
                t.relationships = []

            entry['cleared'] = _attempt(clear)
            entry['original_after'] = describe(obj)
            entry['copy_after'] = describe(twin)
        out[how] = entry
    return out


def make(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    for made in made_objects():
        file: str | None = (
            made.name.replace('/', '__').replace('#', '--') + '.pkl'
        )
        try:
            data = pickle.dumps(made.obj)
        except Exception:  # noqa: BLE001 - such an object has no old pickle
            file = None
        else:
            (out / str(file)).write_bytes(data)
        first = round_trips(made)
        again = round_trips(made)
        assert first == again, f'{made.name} is not described the same twice'
        expected[made.name] = {
            'file': file,
            'transcript': made.source,
            'at': made.at,
            **first,
        }
    (out / 'expected.json').write_text(
        json.dumps(expected, indent=1, sort_keys=True) + '\n'
    )


def fresh() -> dict[str, Any]:
    got = {}
    for made in made_objects():
        got[made.name] = round_trips(made)
    return {'objects': got, 'modules': ported_loaded()}


def old() -> dict[str, Any]:
    expected = json.loads(EXPECTED.read_text())
    from _parity.transcript import load

    got = {}
    for name, entry in expected.items():
        if entry['file'] is None:
            continue
        transcript = load(TRANSCRIPTS / entry['transcript'] / 'transcript.json')
        loaded = _attempt(
            lambda e=entry: pickle.loads((PICKLES / e['file']).read_bytes())
        )
        if 'raised' in loaded:
            got[name] = loaded
            continue
        obj = loaded['value']
        made = Made(name, obj, entry['transcript'], transcript, entry['at'])
        got[name] = {
            'described': describe(obj),
            'behavior': behavior(made, obj),
            'copy': _attempt(lambda o=obj: describe(copy.deepcopy(o))),
        }
    return {'objects': got}


def main(argv: list[str] | None = None) -> int:
    add_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('make', 'fresh', 'old'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    if args.command == 'make':
        make(Path(args.out))
        return 0
    data = fresh() if args.command == 'fresh' else old()
    Path(args.out).write_text(json.dumps(data, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
