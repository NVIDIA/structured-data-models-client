# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact shared artifacts, complete shards and bounded readiness probes."""

import hashlib
import json
import sys
import zipfile
from types import SimpleNamespace

import ci_artifacts
import pytest
import resource_preflight
from ci_shards import combine


@pytest.mark.parametrize(
    'fault', [None, 'sha', 'workflow', 'path', 'partial', 'failed', 'pending']
)
def test_only_successful_exact_full_ci_is_reusable(fault):
    workflow = {'id': 7, 'path': '.github/workflows/ci.yml'}
    run = {
        'id': 11,
        'workflow_id': 7,
        'path': workflow['path'],
        'head_sha': 'candidate',
        'status': 'completed',
        'conclusion': 'success',
        'display_title': 'CI',
    }
    patches = {
        'sha': {'head_sha': 'other'},
        'workflow': {'workflow_id': 8},
        'path': {'path': 'evil.yml'},
        'partial': {
            'display_title': 'Targeted Arm PQL acceptance (not full CI)'
        },
        'failed': {'conclusion': 'failure'},
        'pending': {'status': 'in_progress'},
    }
    run.update(patches.get(fault, {}))
    if fault:
        with pytest.raises(ValueError):
            ci_artifacts.validate_run(run, workflow, 'candidate')
    else:
        assert ci_artifacts.validate_run(run, workflow, 'candidate') == 11


def artifacts(root):
    for name in ('wheels', 'java', 'native/linux-x86_64', 'evidence'):
        (root / name).mkdir(parents=True)
    wheel = root / 'wheels/client.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('sdm_client/_native.abi3.so', b'python native')
    identity = {
        'commit': 'candidate',
        'tree_clean': True,
        'features': [],
        'wheel': wheel.name,
        'sha256': ci_artifacts.digest(wheel),
        'native_sha256': hashlib.sha256(b'python native').hexdigest(),
    }
    sidecar = root / 'wheels/client.whl.json'
    sidecar.write_text(json.dumps(identity))
    jar = root / 'java/client.jar'
    with zipfile.ZipFile(jar, 'w') as archive:
        archive.writestr(ci_artifacts.NATIVE, b'java native')
    pom = root / 'java/client.pom'
    pom.write_text('pom')
    (root / 'java/java-artifacts.json').write_text(
        json.dumps(
            {
                'commit': 'candidate',
                'files': {p.name: ci_artifacts.digest(p) for p in (jar, pom)},
            }
        )
    )
    (root / 'native/linux-x86_64/libsdm_jni.so').write_bytes(b'java native')
    return sidecar, identity


@pytest.mark.parametrize(
    'fault',
    [
        None,
        'commit',
        'dirty',
        'testops',
        'wheel',
        'wheel-native',
        'jar',
        'pom',
        'jni',
        'missing',
        'duplicate',
    ],
)
def test_downloaded_production_artifacts_are_byte_verified(tmp_path, fault):
    sidecar, identity = artifacts(tmp_path)
    if fault in ('commit', 'dirty', 'testops', 'wheel-native'):
        key, value = {
            'commit': ('commit', 'wrong'),
            'dirty': ('tree_clean', False),
            'testops': ('features', ['test-ops']),
            'wheel-native': ('native_sha256', 'wrong'),
        }[fault]
        identity[key] = value
        sidecar.write_text(json.dumps(identity))
    elif fault in ('wheel', 'jar', 'pom', 'jni'):
        path = {
            'wheel': 'wheels/client.whl',
            'jar': 'java/client.jar',
            'pom': 'java/client.pom',
            'jni': 'native/linux-x86_64/libsdm_jni.so',
        }[fault]
        with (tmp_path / path).open('ab') as out:
            out.write(b'tampered')
    elif fault == 'missing':
        sidecar.unlink()
    elif fault == 'duplicate':
        (tmp_path / 'wheels/extra.whl').write_bytes(b'extra')
    if fault:
        with pytest.raises((ValueError, OSError)):
            ci_artifacts.verify(tmp_path, 'candidate', 11)
        assert not (tmp_path / 'evidence/artifacts.json').exists()
    else:
        ci_artifacts.verify(tmp_path, 'candidate', 11)
        receipt = json.loads((tmp_path / 'evidence/artifacts.json').read_text())
        assert receipt['ci_run_id'] == 11 and receipt['build'] == 'production'


@pytest.mark.parametrize(
    'fault',
    [
        None,
        'missing-shard',
        'missing-language',
        'empty',
        'wrong-shard',
        'wrong-language',
        'wrong-run',
        'wrong-commit',
        'mixed-artifacts',
        'wrong-row-native',
        'wrong-row-package',
        'missing-row-artifact',
    ],
)
def test_aggregation_requires_all_declared_shards_and_identities(
    tmp_path, fault
):
    root = tmp_path / 'shards'
    root.mkdir()
    for connector in ('sqlite', 'duckdb'):
        folder = root / ('live-shard-' + connector)
        folder.mkdir()
        receipt = {
            'commit': 'candidate',
            'ci_run_id': 11,
            'build': 'production',
            'native_sha256': 'same',
            'python': {
                'commit': 'candidate',
                'package_hash': 'python',
                'native_hash': 'py-native',
                'contract': 70,
            },
            'java': {
                'commit': 'candidate',
                'package_hash': 'java',
                'native_hash': 'jni-native',
                'contract': 70,
            },
        }
        if connector == 'duckdb':
            receipt.update(
                {
                    'wrong-run': {'ci_run_id': 12},
                    'wrong-commit': {'commit': 'other'},
                    'mixed-artifacts': {'native_sha256': 'other'},
                }.get(fault, {})
            )
        (folder / 'artifacts.json').write_text(json.dumps(receipt))
        for language in ('python', 'java'):
            row = {
                'connector': connector,
                'language': language,
                'artifact': dict(receipt[language]),
            }
            if fault == 'wrong-row-native':
                row['artifact']['native_hash'] = 'other'
            elif fault == 'wrong-row-package':
                row['artifact']['package_hash'] = 'other'
            elif fault == 'missing-row-artifact':
                del row['artifact']
            if connector == 'duckdb' and language == 'java':
                row.update(
                    {
                        'wrong-shard': {'connector': 'sqlite'},
                        'wrong-language': {'language': 'python'},
                    }.get(fault, {})
                )
            (folder / (language + '.jsonl')).write_text(json.dumps(row) + '\n')
    target = root / 'live-shard-duckdb/java.jsonl'
    if fault == 'missing-language':
        target.unlink()
    elif fault == 'empty':
        target.write_text('')
    elif fault == 'missing-shard':
        import shutil

        shutil.rmtree(root / 'live-shard-duckdb')
    out = tmp_path / 'out.jsonl'
    if fault:
        with pytest.raises((ValueError, OSError)):
            combine(root, ['sqlite', 'duckdb'], 'candidate', 11, out)
        assert not out.exists()
    else:
        combine(root, ['sqlite', 'duckdb'], 'candidate', 11, out)
        assert len(out.read_text().splitlines()) == 4


def test_database_probe_reads_one_row_per_configured_fixture_table(monkeypatch):
    queries = []
    monkeypatch.setattr(
        resource_preflight, 'source_tables', lambda *args: (None, {}, object())
    )
    monkeypatch.setitem(
        sys.modules,
        'sdm_connectors',
        SimpleNamespace(
            quote_ident=lambda value, quote: quote + value + quote,
            read_table=lambda connection, query: queries.append(query) or [1],
        ),
    )
    resource_preflight.probe_source(
        'postgres',
        {
            'sources': {
                'postgres': {
                    'source': {'schema': 'fixture_namespace'},
                    'tables': {
                        'users': 'fixture_users',
                        'events': 'fixture_events',
                    },
                }
            }
        },
        {},
    )
    assert queries == [
        'SELECT * FROM "fixture_namespace"."fixture_users" LIMIT 1',
        'SELECT * FROM "fixture_namespace"."fixture_events" LIMIT 1',
    ]


def test_spark_probe_applies_limit_before_collect(monkeypatch):
    calls = []

    class Table:
        def limit(self, rows):
            calls.append(('limit', rows))
            return self

        def toPandas(self):
            calls.append(('collect',))
            return [1]

    connection = SimpleNamespace(
        table=lambda name: calls.append(('table', name)) or Table()
    )
    monkeypatch.setattr(
        resource_preflight,
        'source_tables',
        lambda *args: (None, {}, connection),
    )
    resource_preflight.probe_source(
        'databricks-spark',
        {
            'sources': {
                'databricks-spark': {
                    'tables': {'users': 'users'},
                    'source': {'catalog': 'catalog', 'schema': 'fixture'},
                }
            }
        },
        {},
    )
    assert calls == [
        ('table', '`catalog`.`fixture`.`users`'),
        ('limit', 1),
        ('collect',),
    ]


@pytest.mark.parametrize(
    'ready,listing,expected',
    [
        (True, ['model'], 'observed-nonempty'),
        (True, [], 'observed-empty'),
        (True, RuntimeError('secret URL'), 'unavailable'),
        (False, ['model'], None),
    ],
)
def test_readiness_required_listing_optional_and_private(
    monkeypatch, ready, listing, expected
):
    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def health_ready(self):
            return ready

        def models(self):
            if isinstance(listing, Exception):
                raise listing
            return listing

    monkeypatch.setitem(
        sys.modules, 'sdm_client', SimpleNamespace(SDMClient=Client)
    )
    endpoint = {'transport': 'nim', 'client': {'url': 'secret URL'}}
    if expected is None:
        with pytest.raises(ValueError):
            resource_preflight.probe_endpoint(endpoint)
    else:
        result = resource_preflight.probe_endpoint(endpoint)
        assert result == {'ready': True, 'model_listing': expected}
        assert 'secret' not in json.dumps(result)


def test_preflight_checks_effective_fixture_variants(monkeypatch, tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'endpoints': {}}))
    calls = []
    monkeypatch.setattr(resource_preflight, 'environment', lambda *args: {})
    monkeypatch.setattr(resource_preflight, 'preflight', lambda env: [])
    monkeypatch.setattr(
        resource_preflight, 'model_identities', lambda config: {}
    )
    monkeypatch.setattr(
        resource_preflight,
        'probe_source',
        lambda connector, config, fixture: calls.append(connector),
    )
    result = resource_preflight.check(config)
    pairs = {
        (entry['connector'], entry['fixture']) for entry in result['sources']
    }
    assert (
        'snowflake-semantic',
        'fixtures/shop-snowflake-semantic.json',
    ) in pairs
    assert ('duckdb', 'fixtures/shop-temporal-micro.json') in pairs
    assert len(calls) == len(pairs)


def test_failed_preflight_never_serializes_connection_errors(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / 'result.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'preflight',
            '--config',
            str(tmp_path / 'private.json'),
            '--out',
            str(output),
        ],
    )

    def fail(*args, **kwargs):
        raise RuntimeError('private-host credential-token')

    monkeypatch.setattr(resource_preflight, 'check', fail)
    with pytest.raises(SystemExit):
        resource_preflight.main()
    assert json.loads(output.read_text()) == {
        'status': 'failed',
        'error_type': 'RuntimeError',
    }
    assert 'private-host' not in capsys.readouterr().out + output.read_text()
