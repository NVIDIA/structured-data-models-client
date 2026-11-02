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


def test_live_credentials_only_on_trusted_refs():
    data = workflow('parity-live.yml')
    assert set(data['on']) == {'schedule', 'workflow_dispatch'}
    job = data['jobs']['parity']
    assert job['environment'] == 'parity-live'
    assert job['if'] == (
        "github.ref == 'refs/heads/main' || "
        "github.ref == 'refs/heads/rust-core'"
    )
    assert data['permissions'] == {'contents': 'read'}
    evidence = next(
        s
        for s in job['steps']
        if s.get('with', {}).get('name', '').startswith('live-parity-')
    )
    assert evidence['with']['path'].endswith('/parity/evidence/')
    assert 'config.json' not in evidence['with']['path']


@pytest.mark.parametrize(
    'conclusion', ['success', 'failure', 'cancelled', 'missing', 'skipped']
)
def test_release_requires_success_for_exact_commit(tmp_path, conclusion):
    step = next(
        step
        for step in workflow('publish.yml')['jobs']['build']['steps']
        if step.get('name')
        == 'Require complete live parity for the release commit'
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


def test_giant_acceptance_is_mandatory_sequential_and_private():
    job = workflow('parity-live.yml')['jobs']['parity']
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
        for step in workflow('parity-live.yml')['jobs']['parity']['steps']
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
    job = workflow('parity-live.yml')['jobs']['parity']
    scripts = '\n'.join(step.get('run', '') for step in job['steps'])
    assert '--comparison current-clients' in scripts
    assert '--phase candidate' in scripts
    assert '-Dsdm.parity.mode=live' in scripts
    for retired in (
        '--phase baseline',
        '--baseline',
        'BASELINE_COMMIT',
        'git worktree add',
        'structured-data-models-relational-engine',
        'structured-data-models-pql',
    ):
        assert retired not in scripts
