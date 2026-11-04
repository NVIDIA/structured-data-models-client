# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live requirements must resolve and remain incomplete without resources."""

import copy
import json

import pytest
from api_contracts import groups, validate_live_targets
from live_api import CONFIG_VARIABLES, environment, preflight

from contract import manifest


def test_entire_live_registry_resolves():
    data = manifest()
    validate_live_targets(
        data, {group['id'] for group in data['api_contracts']}
    )
    selected = groups(data, 'live')
    assert len(selected) == len(data['live_api_tests'])
    assert {
        node
        for group in selected
        for nodes in group['tests'].values()
        for node in nodes
    } == {record['test'] for record in data['live_api_tests']}


@pytest.mark.parametrize(
    'fault',
    ['missing-file', 'missing-method', 'unknown-capability', 'duplicate'],
)
def test_invalid_live_registry_fails(fault):
    data = copy.deepcopy(manifest())
    entry = data['live_api_tests'][0]
    if fault == 'missing-file':
        entry['test'] = 'java/missing.java#missing'
    elif fault == 'missing-method':
        entry['test'] += 'Missing'
    elif fault == 'unknown-capability':
        entry['api_contract'] = 'absent'
    else:
        data['live_api_tests'].append(copy.deepcopy(entry))
    with pytest.raises(AssertionError):
        validate_live_targets(
            data, {group['id'] for group in data['api_contracts']}
        )


def test_configuration_mapping_never_claims_execution(tmp_path):
    config = tmp_path / 'private.json'
    config.write_text('{}')
    settings = {'endpoints': {'relational': {'client': {'url': 'private-url'}}}}
    env = environment(config, settings, {})
    assert all(env[name] == str(config) for name in CONFIG_VARIABLES)
    assert env['SDM_RELATIONAL_URL'] == 'private-url'
    missing = preflight(env)
    assert 'SDM_GRADIENT_URL' in missing
    assert 'SDM_RUN_LOCAL_MODEL=1' in missing
    assert 'private-url' not in json.dumps(missing)


def test_arbitrary_environment_injection_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='unknown'):
        environment(
            tmp_path, {'live_api': {'environment': {'PYTHONPATH': 'bad'}}}, {}
        )


def test_preflight_cli_retains_incomplete_sanitized_evidence(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from live_api import REQUIRED_VARIABLES

    config = tmp_path / 'config.json'
    config.write_text(
        json.dumps(
            {
                'endpoints': {
                    'relational': {'client': {'url': 'secret-url-sentinel'}}
                }
            }
        )
    )
    output = tmp_path / 'evidence' / 'live-api.json'
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name('live_api.py')),
            '--config',
            str(config),
            '--out',
            str(output),
            '--private-dir',
            str(tmp_path / 'private'),
            '--commit',
            'candidate',
            '--preflight',
        ],
        env={
            key: value
            for key, value in os.environ.items()
            if key not in REQUIRED_VARIABLES
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    evidence = json.loads(output.read_text())
    assert evidence['status'] == 'incomplete'
    assert evidence['candidate_commit'] == 'candidate'
    assert evidence['results'] == {}
    assert 'immutable model version/transport metadata' in evidence['missing']
    assert (
        'secret-url-sentinel'
        not in output.read_text() + result.stdout + result.stderr
    )


@pytest.mark.parametrize(
    'fault',
    [
        None,
        'commit',
        'wheel',
        'python-native',
        'java-native',
        'java-jar',
        'source',
    ],
)
def test_candidate_artifact_proof_rejects_mismatches(
    tmp_path, monkeypatch, fault
):
    import hashlib
    import sys
    import types
    import zipfile

    import live_api
    from python_runner import Incomplete, artifact

    monkeypatch.setattr(
        live_api.importlib.metadata, 'version', lambda name: 'test-runtime'
    )
    package = tmp_path / 'sdm_client'
    package.mkdir()
    source = package / '__init__.py'
    source.write_text('VALUE = 1\n')
    native = package / 'native.so'
    native.write_bytes(b'native')
    module = types.ModuleType('sdm_client')
    module.__file__ = str(source)
    module._native = types.SimpleNamespace(
        __file__=str(native), CONTRACT_VERSION=1
    )
    monkeypatch.setitem(sys.modules, 'sdm_client', module)
    wheel = tmp_path / 'client.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('sdm_client/__init__.py', source.read_bytes())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    provenance = tmp_path / 'wheel.json'
    provenance.write_text(
        json.dumps(
            {
                'commit': 'candidate',
                'tree_clean': True,
                'features': [],
                'wheel': wheel.name,
                'sha256': digest(wheel),
                'native_sha256': digest(native),
            }
        )
    )
    jar = tmp_path / 'client.jar'
    jar.write_bytes(b'jar')
    jni = tmp_path / 'jni.so'
    jni.write_bytes(b'jni')
    java = tmp_path / 'java.json'
    java.write_text(
        json.dumps(
            {
                'commit': 'candidate',
                'package_hash': digest(jar),
                'native_hash': digest(jni),
                'contract': 1,
            }
        )
    )
    monkeypatch.setattr(
        live_api.subprocess,
        'check_output',
        lambda command, **kwargs: (
            ('wrong' if fault == 'source' else 'candidate')
            if 'rev-parse' in command
            else ''
        ),
    )
    if fault in ('wheel', 'python-native', 'java-native', 'java-jar'):
        {
            'wheel': wheel,
            'python-native': native,
            'java-native': jni,
            'java-jar': jar,
        }[fault].write_bytes(b'changed')
    artifact.cache_clear()
    try:
        if fault:
            with pytest.raises((ValueError, Incomplete)):
                live_api.artifact_identities(
                    'wrong' if fault == 'commit' else 'candidate',
                    provenance,
                    java,
                    jar,
                    jni,
                )
        else:
            identities = live_api.artifact_identities(
                'candidate', provenance, java, jar, jni
            )
            assert (
                identities['python']['commit']
                == identities['java']['commit']
                == 'candidate'
            )
    finally:
        artifact.cache_clear()


def test_model_identity_requires_versions_and_omits_addresses():
    from live_api import model_identities

    config = {
        'endpoints': {
            model: {
                'version': 'v1',
                'transport': 'nim',
                'client': {'url': 'private-url'},
            }
            for model in ('tabular', 'relational')
        },
        'live_api': {
            'model_versions': {
                'pretrained': 'weights-v1',
                'gradients': 'weights-v1',
            }
        },
    }
    assert 'private-url' not in json.dumps(model_identities(config))
    del config['live_api']['model_versions']['gradients']
    with pytest.raises(ValueError, match='versions'):
        model_identities(config)
