# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live credentials and release acceptance cannot be bypassed by partial CI."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.load(
        (ROOT / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader
    )


def test_live_credentials_only_on_approved_runs():
    data = workflow('parity-live.yml')
    assert set(data['on']) == {'workflow_run', 'workflow_dispatch'}
    assert data['on']['workflow_run']['branches'] == ['main']
    assert data['on']['workflow_run']['workflows'] == ['CI']
    assert (
        'head_repository.full_name == github.repository'
        in data['jobs']['preflight']['if']
    )
    assert data['jobs']['shards']['environment'] == 'parity-live'
    assert data['permissions'] == {
        'contents': 'read',
        'actions': 'read',
        'packages': 'read',
    }
    assert data['jobs']['shards']['services']['postgres']['credentials'] == {
        'username': '${{ github.actor }}',
        'password': '${{ secrets.GITHUB_TOKEN }}',
    }
    assert 'env' not in data['jobs']['models']
    for name, value in data['jobs']['shards']['env'].items():
        if 'secrets.' in value:
            assert "matrix.service == 'real-warehouse'" in value
        assert not name.startswith('AWS_')
    for job in data['jobs'].values():
        for step in job['steps']:
            if step.get('uses', '').startswith('actions/upload-artifact@'):
                assert step['with']['path'].endswith('/parity/evidence/')
                assert 'private' not in step['with']['path']


@pytest.mark.parametrize(
    'conclusion', ['success', 'failure', 'cancelled', 'missing', 'skipped']
)
def test_release_requires_success_for_exact_commit(tmp_path, conclusion):
    step = next(
        step
        for step in workflow('publish.yml')['jobs']['build']['steps']
        if step.get('name')
        == 'Require hosted acceptance for the release commit'
    )
    fake = tmp_path / 'gh'
    fake.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$@" > "$PARITY_ARGS"\n'
        'printf "%s\\n" "$PARITY_CONCLUSION"\n'
    )
    fake.chmod(0o755)
    args = tmp_path / 'args'
    result = subprocess.run(
        ['bash', '-c', step['run']],
        env={
            **os.environ,
            'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
            'GITHUB_SHA': 'exact-release-commit',
            'PARITY_ARGS': str(args),
            'PARITY_CONCLUSION': conclusion,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (conclusion == 'success')
    actual = args.read_text().splitlines()
    assert actual[actual.index('--workflow') + 1] == 'parity-live.yml'
    assert actual[actual.index('--commit') + 1] == 'exact-release-commit'
    assert actual[actual.index('--branch') + 1] == 'main'
    assert actual[actual.index('--status') + 1] == 'completed'


def test_optional_giant_acceptance_is_sequential_and_private():
    job = workflow('parity-stress.yml')['jobs']['parity']
    assert 'PARITY_RUNNER_LABELS' in job['runs-on']
    setup = next(
        step
        for step in job['steps']
        if step.get('name') == 'Require dedicated live configuration'
    )
    assert '24 * 1024 * 1024' in setup['run']
    giant = next(
        step
        for step in job['steps']
        if step.get('name')
        == 'Validate giant values against the real model sequentially'
    )
    assert 'if' not in giant and 'continue-on-error' not in giant
    script = giant['run']
    assert "('baseline'," not in script
    assert script.index("('python',") < script.index("('java',")
    assert 'for language, command in commands:' in script
    assert 'subprocess.run(command, stdout=log' in script
    assert (
        "if any(report['status'] != 'passed' for report in reports):" in script
    )
    assert "'evidence/giant.json'" in script
    assert "root / ('giant-' + language + '.log')" in script
    assert giant['env']['SDM_T136_BIG'] == '1'


@pytest.mark.parametrize(
    'case_xml, expected',
    [
        ('<testcase/>', 'passed'),
        ('<testcase><skipped/></testcase>', 'failed'),
        ('<testcase><failure/></testcase>', 'failed'),
        ('<testcase><error/></testcase>', 'failed'),
        ('', 'failed'),
        ('<testcase/><testcase/>', 'failed'),
    ],
)
def test_giant_reports_require_exactly_one_passed_case(
    tmp_path, case_xml, expected
):
    import json

    step = next(
        step
        for step in workflow('parity-stress.yml')['jobs']['parity']['steps']
        if step.get('name')
        == 'Validate giant values against the real model sequentially'
    )
    script = step['run'].split("<<'PYGIANT'\n", 1)[1].rsplit('PYGIANT', 1)[0]
    # Exercise the actual workflow report parser with a successful fake process.
    start = script.rfind('\n', 0, script.index("counts = {'tests'")) + 1
    end = script.rfind('\n', 0, script.index("(root / 'evidence/giant.json')"))
    import textwrap

    parser = textwrap.dedent(script[start:end])
    report = tmp_path / 'result.xml'
    report.write_text('<testsuite>' + case_xml + '</testsuite>')
    import types
    import xml.etree.ElementTree as ET

    context = {
        'report': report,
        'ET': ET,
        'completed': types.SimpleNamespace(returncode=0),
        'reports': [],
        'language': 'candidate',
    }
    exec(parser, context)
    assert context['reports'][0]['status'] == expected
    assert 'testcase' not in json.dumps(context['reports'])


def test_live_workflow_validates_current_clients_without_frozen_install():
    scripts = '\n'.join(
        step.get('run', '')
        for job in workflow('parity-live.yml')['jobs'].values()
        for step in job['steps']
    )
    runner = (ROOT / 'tests/parity/hosted_runner.py').read_text()
    aggregate = (ROOT / 'tests/parity/hosted_aggregate.py').read_text()
    assert "comparison='current-clients'" in aggregate
    assert "'candidate'" in runner
    assert '-Dsdm.parity.mode=local' in runner
    for retired in (
        '--phase baseline',
        '--baseline',
        'BASELINE_COMMIT',
        'git worktree add',
        'structured-data-models-relational-engine',
        'structured-data-models-pql',
    ):
        assert retired not in scripts


def test_ordinary_hosted_separates_cpu_smoke_without_giant_prerequisites():
    data = workflow('parity-live.yml')
    scripts = '\n'.join(
        step.get('run', '')
        for job in data['jobs'].values()
        for step in job['steps']
    )
    assert 'tests/parity/hosted_model_smoke.py' in scripts
    assert 'tests/parity/hosted_runner.py' in scripts
    assert data['jobs']['aggregate']['needs'] == [
        'preflight',
        'shards',
        'models',
    ]
    assert 'MemAvailable' not in scripts
    assert 'PYGIANT' not in scripts
    stress = workflow('parity-stress.yml')
    assert set(stress['on']) == {'workflow_dispatch'}
    assert stress['jobs']['parity']['environment'] == 'parity-live'


def test_live_lanes_share_resources_and_exact_artifacts_without_rebuilds():
    live = workflow('parity-live.yml')
    stress = workflow('parity-stress.yml')
    assert (
        live['concurrency']
        == stress['concurrency']
        == {'group': 'parity-shared-resources', 'cancel-in-progress': 'false'}
    )
    assert live['jobs']['shards']['strategy']['max-parallel'] == '2'
    assert live['jobs']['shards']['strategy']['fail-fast'] == 'false'
    assert live['jobs']['shards']['needs'] == 'preflight'
    assert live['jobs']['models']['needs'] == 'preflight'
    for data in (live, stress):
        for job in data['jobs'].values():
            scripts = '\n'.join(step.get('run', '') for step in job['steps'])
            for forbidden in (
                'cargo build',
                'native_wheel.py',
                '-DskipTests package',
            ):
                assert forbidden not in scripts
    for name in ('shards', 'models'):
        setup = next(
            step
            for step in live['jobs'][name]['steps']
            if step.get('uses') == './.github/actions/parity-artifacts'
        )
        assert (
            setup['with']['ci-run-id']
            == '${{ needs.preflight.outputs.ci_run_id }}'
        )
    aggregate = '\n'.join(
        step.get('run', '') for step in live['jobs']['aggregate']['steps']
    )
    assert 'hosted_aggregate.py' in aggregate
    assert '--connector' not in aggregate and '--language' not in aggregate
    assert 'test "$SHARDS_RESULT" = success' in aggregate
    assert 'test "$MODELS_RESULT" = success' in aggregate


def test_installed_api_failure_retains_accounting_and_diagnostics():
    steps = workflow('ci.yml')['jobs']['installed-wheel']['steps']
    run = next(
        step
        for step in steps
        if step.get('name')
        == 'Required Python API cases and collected result identities'
    )
    assert 'set -o pipefail' in run['run']
    assert '--run --out "$RUNNER_TEMP/python-api-results.json"' in run['run']
    assert 'tee "$RUNNER_TEMP/python-api.log"' in run['run']
    upload = next(
        step
        for step in steps
        if step.get('with', {}).get('name') == 'installed-wheel-api-results'
    )
    assert upload['if'] == 'always()'
    assert 'python-api-results.json' in upload['with']['path']
    assert 'python-api.log' in upload['with']['path']
