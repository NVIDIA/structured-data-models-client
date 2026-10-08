# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The recording tools beyond what the acceptance checks pin down.

* The scrubber reads Arrow strings at any depth, and the places where live
  results name people and storage.
* ``record`` stores only a recording that replays, one scenario that fails
  does not stop the others, and two PostgreSQL recordings at once do not
  collide.

Secret-shaped values are assembled at run time, so no literal here trips a
scanner.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'parity'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _parity.scrubber import Scrubber  # noqa: E402

EMAIL = 'jane.doe' + '@' + 'corp.example.com'
BUCKET = 's3://' + 'acme-private-bucket/warehouse/orders'
ABFSS = 'abfss://' + 'container' + '@' + 'acmestore.dfs.core.windows.net/tables'
BLOB_HOST = 'acmestore' + '.blob.core.windows.net'


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [*value, *(s for v in value.values() for s in _strings(v))]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def _leaks(table: pa.Table) -> list[str]:
    texts = _strings(table.to_pylist())
    texts += [str(table.schema), str(table.schema.metadata)]
    texts += [str(field.metadata) for field in table.schema]
    return [
        secret
        for secret in (EMAIL, BUCKET, ABFSS, BLOB_HOST, 'ROLE_OWNER_X')
        if any(secret in text for text in texts)
    ]


def test_arrow_strings_are_scrubbed_at_any_depth() -> None:
    owner_field = pa.field(
        'comment', pa.string(), metadata={b'by': EMAIL.encode()}
    )
    table = pa.Table.from_arrays(
        [
            pa.array([[f'see {BUCKET}', 'north'], []]),
            pa.array([{'path': ABFSS, 'n': 1}, {'path': 'kept', 'n': 2}]),
            pa.array(
                [[('location', 'anywhere'), ('k', 'kept')], [('k', BLOB_HOST)]],
                type=pa.map_(pa.string(), pa.string()),
            ),
            pa.array([f'mail {EMAIL}', 'kept']).dictionary_encode(),
            pa.array(['ROLE_OWNER_X', None]),
            pa.array(['kept', 'kept']),
        ],
        schema=pa.schema(
            [
                pa.field('notes', pa.list_(pa.string())),
                pa.field(
                    'where',
                    pa.struct([('path', pa.string()), ('n', pa.int64())]),
                ),
                pa.field('props', pa.map_(pa.string(), pa.string())),
                pa.field('said', pa.dictionary(pa.int32(), pa.string())),
                pa.field('owner', pa.string()),
                owner_field,
            ],
            metadata={b'source': BUCKET.encode()},
        ),
    )
    assert len(_leaks(table)) == 5

    scrubbed, found = Scrubber({}).table(table)

    assert not _leaks(scrubbed), _leaks(scrubbed)
    assert found
    assert scrubbed.schema.names == table.schema.names
    assert [f.type for f in scrubbed.schema] == [f.type for f in table.schema]
    rows = scrubbed.to_pylist()
    assert rows[0]['notes'][1] == 'north'
    assert rows[1]['where'] == {'path': 'kept', 'n': 2}
    assert rows[0]['props'][1] == ('k', 'kept')
    assert rows[1]['said'] == 'kept'
    assert rows[1]['owner'] is None
    assert [r['comment'] for r in rows] == ['kept', 'kept']


def test_key_value_rows_name_what_is_secret() -> None:
    r"""``DESCRIBE TABLE EXTENDED`` answers in rows of name and value."""
    owner = 'svc-' + '5f1c2e3a-0000-4b6c-9d8e-123456789abc'
    table = pa.table(
        {
            'col_name': ['user_id', '', 'Owner', 'Location', 'Type'],
            'data_type': ['bigint', '', owner, ABFSS, 'MANAGED'],
            'comment': [None, None, None, None, None],
        }
    )
    scrubbed, _ = Scrubber({}).table(table)
    values = scrubbed.column('data_type').to_pylist()
    assert values[0] == 'bigint'
    assert values[4] == 'MANAGED'
    assert owner not in values and ABFSS not in values
    assert (
        scrubbed.column('col_name').to_pylist()
        == table.column('col_name').to_pylist()
    )


def test_a_scrubbed_result_scans_clean() -> None:
    r"""What scrubbing leaves is not found again, keys with spaces included:
    Databricks' ``DESCRIBE TABLE EXTENDED`` of a metric view has a
    ``Created By`` row, whose mark the scan once took for a secret."""
    table = pa.table(
        {
            'col_name': ['Created By', 'Last Modified By', 'Owner', 'Type'],
            'data_type': ['Spark', 'someone', 'someone', 'METRIC_VIEW'],
        }
    )
    scrubbed, found = Scrubber({}).table(table)
    assert found
    assert scrubbed.column('data_type').to_pylist() == [
        '<redacted:created-by>',
        '<redacted:last-modified-by>',
        '<redacted:owner>',
        'METRIC_VIEW',
    ]
    again, found = Scrubber({}).table(scrubbed)
    assert found == []
    assert again.equals(scrubbed)
    value, found = Scrubber({}).json({'Created By': 'someone'})
    assert value == {'Created By': '<redacted:created-by>'}
    assert Scrubber({}).json(value)[1] == []


def test_shapes_of_people_and_storage_are_found_in_json() -> None:
    value = {
        'error': f'{EMAIL} may not read {BUCKET} or {ABFSS} at {BLOB_HOST}',
        'kept': 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
    }
    scrubbed, found = Scrubber({}).json(value)
    assert {f.what for f in found} >= {'email', 'storage-url', 'storage-host'}
    assert not any(
        secret in scrubbed['error']
        for secret in (EMAIL, BUCKET, ABFSS, BLOB_HOST)
    )
    assert scrubbed['kept'] == value['kept']


#: The three variables of a Snowflake personal access token's file, with
#: values no shape-based rule would find.
SNOWFLAKE_FILE = {
    'SNOWFLAKE_URL': 'https://' + 'parity-planted-acct.example.org',
    'SNOWFLAKE_USERNAME': 'PARITY' + '_PLANTED_USER',
    'SNOWFLAKE_ACCESS_TOKEN': 'pat' + '-planted-0123456789-abcdef',
}


def _snowflake_sample(tmp_path: Path) -> Path:
    r"""A recording carrying each value in SQL, an error, a payload and Arrow."""
    url, user, token = SNOWFLAKE_FILE.values()
    root = tmp_path / 'snowflake' / 'count'
    root.mkdir(parents=True)
    transcript = {
        'name': 'snowflake/count',
        'steps': [
            {
                'op': 'execute',
                'sql': f"SELECT '{user}' AS who, '{url}/console' AS at",
                'params': [token],
                'result': 'step-0000.arrow',
                'error': {
                    'type': 'snowflake.connector.errors.ProgrammingError',
                    'message': f'{user} could not log in to {url} with {token}',
                },
            }
        ],
        'payloads': [{'metadata': {'note': f'{user}@{url}'}}],
    }
    (root / 'transcript.json').write_text(json.dumps(transcript))
    table = pa.table(
        {
            'who': [user, 'kept'],
            'notes': [[f'token {token}'], []],
            'where': [{'url': url}, {'url': 'kept'}],
        }
    )
    with pa.OSFile(str(root / 'step-0000.arrow'), 'wb') as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    return tmp_path


def _holds_any(root: Path, secrets: Any) -> list[str]:
    from _parity.scrubber import _read_arrow

    texts = []
    for path in root.rglob('*'):
        if path.suffix == '.json':
            texts.append(path.read_text('utf-8'))
        elif path.suffix == '.arrow':
            table = _read_arrow(path)[0]
            texts += [json.dumps(table.to_pylist()), str(table.schema)]
    return [s for s in secrets if any(s in text for text in texts)]


def test_the_snowflake_token_files_values_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""``SNOWFLAKE_URL``, ``SNOWFLAKE_USERNAME`` and ``SNOWFLAKE_ACCESS_TOKEN``
    are redacted wherever they appear, and ``scan`` finds them, when set.

    No shape-based rule finds these values, so without the variables set
    the scan passes them; with them set, it finds each, and the scrubbed
    copy holds none.
    """
    import scrub

    sample = _snowflake_sample(tmp_path / 'in')
    secrets = list(SNOWFLAKE_FILE.values())
    assert _holds_any(sample, secrets) == secrets
    for name in SNOWFLAKE_FILE:
        monkeypatch.delenv(name, raising=False)
    assert scrub.main(['scan', str(sample)]) == 0

    for name, value in SNOWFLAKE_FILE.items():
        monkeypatch.setenv(name, value)
    assert scrub.main(['scan', str(sample)]) == 1
    target = tmp_path / 'out'
    assert scrub.main(['scrub', str(sample), str(target)]) == 0

    assert _holds_any(target, secrets) == []
    assert scrub.main(['scan', str(target)]) == 0
    stored = json.loads(
        (target / 'snowflake' / 'count' / 'transcript.json').read_text()
    )
    step = stored['steps'][0]
    assert step['params'] == ['<redacted:snowflake_access_token>']
    assert step['sql'] == (
        "SELECT '<redacted:snowflake_username>' AS who, "
        "'<redacted:snowflake_url>/console' AS at"
    )


def test_the_metric_view_names_a_catalog_needing_backticks() -> None:
    r"""A catalog such as ``example-catalog`` is written in backticks, which
    YAML takes in no plain value: the live run's metric view failed so."""
    yaml = pytest.importorskip('yaml')
    from _parity.live import METRIC_VIEW

    qualified = '`example-catalog`.parity_shop'
    view = yaml.safe_load(METRIC_VIEW.format(q=qualified))
    assert view['source'] == f'{qualified}.orders'
    assert [join['source'] for join in view['joins']] == [
        f'{qualified}.users',
        f'{qualified}.products',
    ]


class _Statements:
    r"""A cursor that keeps the SQL it is given."""

    def __init__(self) -> None:
        self.sql: list[str] = []

    def execute(self, sql: str) -> None:
        self.sql.append(sql)


def _creates(backend: str, quoted: bool) -> list[str]:
    from _parity import data as datagen
    from _parity.live import _load
    from _parity.scenarios import SHOP_SEED

    cursor = _Statements()
    frames = datagen.generate('shop', SHOP_SEED)
    _load(cursor, 'D.S', frames, backend, quoted=quoted)
    return [sql for sql in cursor.sql if sql.startswith('CREATE')]


def test_snowflake_tables_keep_the_shops_lower_case_names() -> None:
    r"""Snowflake folds an unquoted name to upper case, so the shop's tables
    are written quoted there: the live run found no ``users.user_id``."""
    creates = _creates('snowflake', quoted=True)
    assert creates[0].startswith(
        'CREATE OR REPLACE TABLE D.S."users" ("user_id" BIGINT, "age" BIGINT'
    )
    unquoted = _creates('snowflake', quoted=False)
    assert unquoted[0].startswith(
        'CREATE OR REPLACE TABLE D.S.users (user_id BIGINT, age BIGINT'
    )
    assert _creates('databricks', quoted=False)[0].startswith(
        'CREATE OR REPLACE TABLE D.S.users (user_id BIGINT'
    )


def test_a_snowflake_semantic_view_is_asked_in_upper_case() -> None:
    r"""The engine reads a semantic view's quoted names with their quotes,
    so the semantic view is over the shop in upper case, and its questions
    name the tables and columns so; every other live scenario asks the
    offline question as it is."""
    from _parity.scenarios import QUESTIONS, live

    for scenario in live():
        query = scenario.calls[1]['args']['query']
        asked = QUESTIONS[scenario.task_kind][0]
        if scenario.data.get('view') == 'semantic-view':
            assert query != asked
            assert query.lower() == asked.lower()
            assert query.startswith('PREDICT ')
            assert ('days' in query) == ('days' in asked)
        else:
            assert query == asked
    count = next(s for s in live() if s.name == 'snowflake/semantic-view/count')
    assert count.calls[1]['args']['query'] == (
        'PREDICT COUNT(ORDERS.*, 0, 30, days) FOR EACH USERS.USER_ID'
    )


# Recording.


def test_a_recording_that_does_not_replay_is_not_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _parity import corpus
    from _parity.runner import CannotRecord
    from _parity.scenarios import by_name

    save = corpus.save

    def broken(directory: Path, transcript: dict[str, Any], *rest: Any) -> Any:
        # As a scrub that touched a payload would leave it.
        transcript['payloads'][0]['metadata']['model'] = '<redacted:host>'
        return save(directory, transcript, *rest)

    monkeypatch.setattr(corpus, 'save', broken)
    with pytest.raises(CannotRecord, match='does not replay.*payloads'):
        corpus.record_one(by_name()['tabular/direct-regression'], tmp_path)
    assert not any(tmp_path.iterdir())


def test_one_failing_scenario_does_not_stop_the_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import run_corpus

    record_one = run_corpus.record_one

    def flaky(scenario: Any, corpus: Path) -> Path:
        if scenario.name == 'tabular/direct-regression':
            raise TypeError('a driver call went wrong')
        return record_one(scenario, corpus)

    monkeypatch.setattr(run_corpus, 'record_one', flaky)
    status = run_corpus.main(
        [
            'record',
            '--corpus',
            str(tmp_path),
            '--only',
            'tabular/direct-regression,tabular/direct-classification',
        ]
    )
    assert status != 0
    err = capsys.readouterr().err
    assert (
        'cannot record tabular/direct-regression: TypeError: a driver call'
        in err
    )
    stored = [
        json.loads(p.read_text())['name']
        for p in tmp_path.rglob('transcript.json')
    ]
    assert stored == ['tabular/direct-classification']


def test_two_postgres_recordings_at_once_do_not_collide(
    tmp_path: Path,
) -> None:
    r"""Each recording loads its data into a schema named in its SQL, so two
    at once in one database (a gate and a reviewer, say) take turns."""
    if not os.getenv('SDM_POSTGRES_TEST_DSN'):
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    runs = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPTS / 'run_corpus.py'),
                'record',
                '--corpus',
                str(tmp_path / str(i)),
                '--only',
                'postgres/count',
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for i in range(2)
    ]
    for run in runs:
        out, err = run.communicate(timeout=600)
        assert run.returncode == 0, f'{out}\n{err}'
    first, second = (
        (
            tmp_path / str(i) / 'postgres' / 'count' / 'transcript.json'
        ).read_text()
        for i in range(2)
    )
    assert first == second


# --- The Snowflake sign-in ------------------------------------------------------


_SIGN_IN = (
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_TOKEN',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_WAREHOUSE',
    'PARITY_SNOWFLAKE_DATABASE',
)


class _FakeCursor:
    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, *args: Any, **kwargs: Any) -> _FakeCursor:
        return self


class _FakeConnection:
    def cursor(self) -> _FakeCursor:
        return _FakeCursor()

    def close(self) -> None:
        return None


@pytest.fixture
def snowflake_sign_ins(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    r"""What each ``snowflake.connector.connect`` was given, with a stand-in
    ``snowflake.connector`` in place of the driver, which CI doesn't
    install, and dummy credentials set; nothing is reached."""
    package = types.ModuleType('snowflake')
    connector = types.ModuleType('snowflake.connector')
    package.connector = connector  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'snowflake', package)
    monkeypatch.setitem(sys.modules, 'snowflake.connector', connector)
    for name in _SIGN_IN:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SNOWFLAKE_ACCOUNT', 'dummy-account')
    monkeypatch.setenv('SNOWFLAKE_USER', 'dummy-user')
    monkeypatch.setenv('SNOWFLAKE_WAREHOUSE', 'dummy-warehouse')
    monkeypatch.setenv('PARITY_SNOWFLAKE_DATABASE', 'DUMMY_DB')
    given: list[dict[str, Any]] = []

    def connect(**kwargs: Any) -> _FakeConnection:
        given.append(kwargs)
        return _FakeConnection()

    connector.connect = connect  # type: ignore[attr-defined]
    return given


def _sign_in_to_snowflake() -> None:
    from _parity import data as datagen
    from _parity import live

    path = Path(__file__).parent / 'transcripts' / 'snowflake' / 'count'
    transcript = json.loads((path / 'transcript.json').read_text())
    prepared = live.prepare(transcript)
    data = prepared['data']
    with live.connection(
        prepared, datagen.generate(data['generator'], data['seed'])
    ):
        pass


def test_a_snowflake_token_is_signed_in_with_in_place_of_a_password(
    snowflake_sign_ins: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SNOWFLAKE_TOKEN', 'dummy-token')
    monkeypatch.setenv('SNOWFLAKE_PASSWORD', 'dummy-password')
    _sign_in_to_snowflake()
    assert snowflake_sign_ins == [
        {
            'account': 'dummy-account',
            'user': 'dummy-user',
            'warehouse': 'dummy-warehouse',
            'authenticator': 'PROGRAMMATIC_ACCESS_TOKEN',
            'token': 'dummy-token',
        }
    ]


def test_a_snowflake_password_is_signed_in_with_when_there_is_no_token(
    snowflake_sign_ins: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('SNOWFLAKE_PASSWORD', 'dummy-password')
    _sign_in_to_snowflake()
    assert snowflake_sign_ins == [
        {
            'account': 'dummy-account',
            'user': 'dummy-user',
            'warehouse': 'dummy-warehouse',
            'password': 'dummy-password',
        }
    ]


def test_snowflake_is_not_recorded_without_a_token_or_a_password(
    snowflake_sign_ins: list[dict[str, Any]],
) -> None:
    from _parity.runner import CannotRecord

    with pytest.raises(
        CannotRecord, match='SNOWFLAKE_TOKEN or SNOWFLAKE_PASSWORD set'
    ):
        _sign_in_to_snowflake()
    assert snowflake_sign_ins == []
