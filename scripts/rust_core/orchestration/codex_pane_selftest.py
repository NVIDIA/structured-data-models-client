#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-tests for codex_pane.py against a fake herdr and a throwaway repository.

No network, no real herdr, no Codex. Run: python3 codex_pane_selftest.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / 'codex_pane.py'

# The fake herdr keeps its world in $FAKE_STATE. FAKE_MODE picks how the
# "Codex" in a pane answers a prompt: report, changes, capacity-once,
# capacity-always, silent, blocked, working.
FAKE = textwrap.dedent(
    r"""
    #!/usr/bin/env python3
    import json, os, re, sys
    from pathlib import Path
    state = Path(os.environ['FAKE_STATE'])
    w = json.loads(state.read_text()) if state.exists() else {'ws': [], 'tabs': 0, 'agents': {}, 'log': []}
    a = sys.argv[1:]
    w['log'].append(a)
    def out(result):
        print(json.dumps({'result': result}))
    def save():
        state.write_text(json.dumps(w))
    def agent(target):
        for pane, ag in w['agents'].items():
            if pane == target or ag.get('name') == target:
                return pane, ag
        print(json.dumps({'error': {'code': 'agent_not_found', 'message': target}}), file=sys.stderr)
        save(); sys.exit(1)
    def answer(ag, prompt):
        mode = os.environ.get('FAKE_MODE', 'report')
        report = re.search(r'Write your report to (\S+?):', prompt).group(1)
        sha = re.search(r'at commit ([0-9a-f]{40})', prompt).group(1)
        if mode == 'capacity-once' and not ag.get('failed'):
            ag['failed'] = True; ag['status'] = 'idle'; ag['text'] = 'stream error: model at capacity, try again later'
        elif mode == 'capacity-always':
            ag['status'] = 'idle'; ag['text'] = 'error 429 Too Many Requests'
        elif mode == 'silent':
            ag['status'] = 'idle'; ag['text'] = 'I looked around.'
        elif mode == 'blocked':
            ag['status'] = 'blocked'
        elif mode == 'working':
            ag['status'] = 'working'
        else:
            verdict = 'CHANGES REQUIRED' if mode == 'changes' else 'APPROVE'
            Path(report).write_text(f'{verdict}\n{sha}\nP3 nothing much\n')
            ag['status'] = 'done'; ag['text'] = 'Report written.'
    cmd = a[:2]
    if cmd == ['workspace', 'list']:
        out({'workspaces': w['ws']})
    elif cmd == ['workspace', 'create']:
        ws = {'workspace_id': f'w{len(w["ws"]) + 1}', 'label': a[a.index('--label') + 1]}
        w['ws'].append(ws); out({'workspace': ws, 'tab': {}, 'root_pane': {}})
    elif cmd == ['tab', 'create']:
        w['tabs'] += 1; n = w['tabs']
        w.setdefault('tab_of', {})[f'w9:p{n}'] = f'w9:t{n}'
        w.setdefault('tab_labels', {})[f'w9:t{n}'] = a[a.index('--label') + 1]
        if os.environ.get('FAKE_FAIL_RUN') == '1':
            w['fail_run'] = True
        out({'tab': {'tab_id': f'w9:t{n}'}, 'root_pane': {'pane_id': f'w9:p{n}', 'terminal_title': 'user@host'}})
    elif cmd == ['tab', 'list']:
        out({'tabs': [{'tab_id': t, 'label': l} for t, l in w.get('tab_labels', {}).items()]})
    elif cmd == ['pane', 'rename']:
        out({'pane': {'terminal_title': 'user@host'}})
        if w.pop('fail_run', False):
            save(); print('boom', file=sys.stderr); sys.exit(1)
    elif cmd == ['pane', 'run']:
        if os.environ.get('FAKE_NO_CODEX') != '1':
            assert a[3].startswith('cx -m gpt-6-astra -c model_reasoning_effort=high -C '), a
            w['agents'][a[2]] = {'name': None, 'status': 'idle', 'text': '', 'tab': w['tab_of'][a[2]]}
        # the real herdr prints nothing here
    elif cmd == ['agent', 'list']:
        out({'agents': [{'name': ag['name'], 'agent': 'codex', 'agent_status': ag['status'], 'pane_id': p,
                         'tab_id': ag.get('tab'), 'terminal_title': 'user@host'} for p, ag in w['agents'].items()]})
    elif cmd == ['agent', 'rename']:
        if os.environ.get('FAKE_FAIL_AGENT_RENAME') == '1':
            save(); print('{"error": {"code": "x", "message": "rename failed"}}', file=sys.stderr); sys.exit(1)
        pane, ag = agent(a[2]); ag['name'] = a[3]; out({'agent': {'name': a[3]}})
    elif cmd == ['agent', 'prompt']:
        pane, ag = agent(a[2]); ag['prompts'] = ag.get('prompts', 0) + 1; answer(ag, a[3]); out({})
    elif cmd == ['agent', 'read']:
        pane, ag = agent(a[2]); print(ag.get('text', ''))
    elif cmd == ['agent', 'kill']:
        w['agents'].pop(agent(a[2])[0]); out({})
    else:
        print(json.dumps({'error': {'code': 'unknown', 'message': ' '.join(a)}}), file=sys.stderr); save(); sys.exit(2)
    save()
    """
).lstrip()

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = '') -> None:
    print(
        ('ok   ' if condition else 'FAIL ')
        + name
        + (f'  ({detail})' if detail and not condition else '')
    )
    if not condition:
        FAILURES.append(name)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        origin, repo, state = root / 'origin.git', root / 'repo', root / 'state'
        # The commits' identity from the environment, so no git configuration is written.
        identity = {
            **os.environ,
            'GIT_AUTHOR_NAME': 'Test',
            'GIT_AUTHOR_EMAIL': 'test@example.invalid',
            'GIT_COMMITTER_NAME': 'Test',
            'GIT_COMMITTER_EMAIL': 'test@example.invalid',
            'GIT_TERMINAL_PROMPT': '0',
        }

        def run(*c: str, cwd: Path = root) -> str:
            done = subprocess.run(
                c,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                env=identity,
            )
            return done.stdout.strip()

        run('git', 'init', '--quiet', '--bare', str(origin))
        run('git', 'clone', '--quiet', str(origin), str(repo))
        (repo / 'a.txt').write_text('one\n')
        run('git', 'add', '.', cwd=repo)
        run('git', 'commit', '--quiet', '-m', 'one', cwd=repo)
        run('git', 'push', '--quiet', 'origin', 'HEAD:rust-core', cwd=repo)
        first = run('git', 'rev-parse', 'HEAD', cwd=repo)
        (repo / 'a.txt').write_text('two\n')
        run('git', 'commit', '--quiet', '-am', 'two', cwd=repo)
        run(
            'git',
            'push',
            '--quiet',
            'origin',
            'HEAD:rust-core-phase-2-lane-a',
            cwd=repo,
        )
        second = run('git', 'rev-parse', 'HEAD', cwd=repo)
        run('git', 'fetch', '--quiet', 'origin', cwd=repo)
        # The handoff stamps a review needs (T3.0), as handoff.py writes
        # them, for the two SHAs reviewed below. A SHA with none is refused
        # (tests/test_t30_process.py holds that).
        stamps = repo / '.git' / 'sdm-handoff'
        stamps.mkdir()
        for sha in (first, second):
            (stamps / f'{sha}.json').write_text(
                json.dumps({'sha': sha, 'package': 'T2.3', 'ok': True})
            )
        fake = root / 'herdr'
        fake.write_text(FAKE)
        fake.chmod(0o755)
        brief = root / 'brief.md'
        brief.write_text('Review it.\n')
        env = {
            **os.environ,
            'SDM_ORCH_STATE': str(state),
            'SDM_REPO': str(repo),
            'SDM_HERDR': str(fake),
            'FAKE_STATE': str(root / 'fake.json'),
            'SDM_CODEX_POLL': '0.05',
            'SDM_CODEX_GRACE': '0',
            'SDM_CODEX_BACKOFF': '0,0',
            'SDM_CODEX_CAP': '2',
        }

        def tool(*args: str, **extra: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, str(TOOL), *args],
                env={**env, **extra},
                capture_output=True,
                text=True,
                timeout=120,
            )

        def fake_world() -> dict:
            return json.loads((root / 'fake.json').read_text())

        # open
        r = tool('open', '--package', 'T2.3')
        check(
            'open starts Codex and names it',
            r.returncode == 0 and 'OPEN T2.3 cx-t2-3' in r.stdout,
            r.stderr,
        )
        check(
            'open makes the review worktree',
            (state / 'review-worktrees/T2.3/a.txt').is_file(),
        )
        check(
            'open creates the codex-reviews workspace',
            fake_world()['ws'][0]['label'] == 'codex-reviews',
        )
        check(
            'nothing prints herdr raw output',
            'user@host' not in r.stdout + r.stderr,
        )
        r = tool('open', '--package', 'T2.3')
        check(
            'open again reuses the live pane',
            r.returncode == 0 and 'already running' in r.stdout,
        )
        check(
            'open refuses a non-package ID',
            tool('open', '--package', 'rust-core').returncode == 2,
        )
        r = tool(
            'open',
            '--package',
            'T2.9',
            FAKE_NO_CODEX='1',
            SDM_CODEX_OPEN_TIMEOUT='1',
        )
        check(
            'open fails cleanly when Codex never comes up',
            r.returncode == 2 and 'did not come up' in r.stderr,
            r.stderr,
        )
        r = tool('open', '--package', 'T2.9')
        check(
            'a tab left with no Codex is not reused blindly',
            r.returncode == 2 and 'inspect it' in r.stderr,
            r.stderr,
        )
        tabs_before = fake_world()['tabs']
        r = tool('open', '--package', 'T2.8', FAKE_FAIL_RUN='1')
        check(
            'a herdr failure is a clean FAILED line, no traceback',
            r.returncode == 2 and 'Traceback' not in r.stderr,
            r.stderr,
        )
        r = tool('open', '--package', 'T2.7', FAKE_FAIL_AGENT_RENAME='1')
        tabs_before = fake_world()['tabs']
        r = tool('open', '--package', 'T2.7')
        check(
            'open adopts Codex left running by a half-finished open',
            r.returncode == 0 and fake_world()['tabs'] == tabs_before,
            r.stderr,
        )

        # review + wait: approve
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            second,
            '--brief',
            str(brief),
        )
        check(
            'review sends round 1',
            r.returncode == 0 and f'SENT T2.3 r1 {second[:12]}' in r.stdout,
            r.stderr,
        )
        wt_head = run(
            'git', 'rev-parse', 'HEAD', cwd=state / 'review-worktrees/T2.3'
        )
        check(
            'review checks the worktree out at the exact SHA', wt_head == second
        )
        r = tool('wait', '--package', 'T2.3')
        check(
            'wait reports APPROVE',
            r.returncode == 0 and r.stdout.startswith('REPORT T2.3 r1 APPROVE'),
            r.stdout,
        )

        # fix round in the same pane, changes required
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            first,
            '--brief',
            str(brief),
            FAKE_MODE='changes',
        )
        check(
            'round 2 goes to the same pane',
            r.returncode == 0 and 'SENT T2.3 r2' in r.stdout,
            r.stderr,
        )
        r = tool('wait', '--package', 'T2.3')
        check(
            'wait reports CHANGES REQUIRED',
            'REPORT T2.3 r2 CHANGES REQUIRED' in r.stdout,
            r.stdout,
        )

        # a fresh report file per round
        used = state / 'codex-reports/T2.3-r2.md'
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            first,
            '--brief',
            str(brief),
            '--report',
            str(used),
        )
        check(
            'an existing report file is refused',
            r.returncode == 2 and 'already exists' in r.stderr,
        )

        # capacity: retried once, then reported
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            second,
            '--brief',
            str(brief),
            FAKE_MODE='capacity-once',
        )
        r = tool('wait', '--package', 'T2.3', FAKE_MODE='capacity-once')
        check(
            'a capacity error is retried and the review completes',
            'REPORT T2.3 r3 APPROVE' in r.stdout,
            r.stdout,
        )
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            second,
            '--brief',
            str(brief),
            FAKE_MODE='capacity-always',
        )
        r = tool('wait', '--package', 'T2.3', FAKE_MODE='capacity-always')
        check(
            'repeated capacity errors end as CAPACITY',
            r.returncode == 5 and r.stdout.startswith('CAPACITY'),
            r.stdout,
        )

        # silent and blocked
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            second,
            '--brief',
            str(brief),
            FAKE_MODE='silent',
        )
        r = tool('wait', '--package', 'T2.3', FAKE_MODE='silent')
        check(
            'idle with no report is IDLE-NO-REPORT',
            r.returncode == 5 and 'IDLE-NO-REPORT' in r.stdout,
            r.stdout,
        )
        r = tool(
            'review',
            '--package',
            'T2.3',
            '--sha',
            second,
            '--brief',
            str(brief),
            FAKE_MODE='blocked',
        )
        r = tool('wait', '--package', 'T2.3')
        check(
            'a dialog is BLOCKED',
            r.returncode == 4 and 'BLOCKED' in r.stdout,
            r.stdout,
        )

        # cap: T2.3 is still reviewing (blocked); cap is 2
        tool('open', '--package', 'T2.4')
        tool('open', '--package', 'T2.5')
        r = tool(
            'review',
            '--package',
            'T2.4',
            '--sha',
            second,
            '--brief',
            str(brief),
            FAKE_MODE='working',
        )
        check(
            'a second concurrent review is allowed under the cap',
            r.returncode == 0,
            r.stderr + r.stdout,
        )
        r = tool(
            'review',
            '--package',
            'T2.4',
            '--sha',
            second,
            '--brief',
            str(brief),
        )
        check(
            'a new round while one is still working is refused',
            r.returncode == 2 and 'still reviewing' in r.stderr,
            r.stderr,
        )
        r = tool(
            'review',
            '--package',
            'T2.5',
            '--sha',
            second,
            '--brief',
            str(brief),
        )
        check(
            'a third concurrent review exits AT-CAP',
            r.returncode == 3 and r.stdout.startswith('AT-CAP 2/2'),
            r.stdout,
        )
        r = tool('wait', '--package', 'T2.4', '--timeout-min', '0.001')
        check(
            'a long review times out without settling',
            r.returncode == 7 and 'TIMEOUT' in r.stdout,
            r.stdout,
        )

        # tracked changes in the worktree are not overwritten
        (state / 'review-worktrees/T2.5/a.txt').write_text(
            'edited by a reviewer\n'
        )
        tool('wait', '--package', 'T2.4', '--timeout-min', '0.001')
        r = tool(
            'review',
            '--package',
            'T2.5',
            '--sha',
            first,
            '--brief',
            str(brief),
            SDM_CODEX_CAP='9',
        )
        check(
            'a dirty review worktree is refused',
            r.returncode == 2 and 'tracked changes' in r.stderr,
            r.stderr,
        )

        # a pane that went away
        subprocess.run(
            [str(fake), 'agent', 'kill', 'cx-t2-4'],
            env=env,
            capture_output=True,
        )
        r = tool('wait', '--package', 'T2.4')
        check(
            'a closed pane is GONE',
            r.returncode == 6 and r.stdout.startswith('GONE'),
            r.stdout,
        )
        r = tool(
            'review',
            '--package',
            'T2.4',
            '--sha',
            second,
            '--brief',
            str(brief),
        )
        check(
            'review to a gone pane says to open again',
            r.returncode == 2 and 'open again' in r.stderr,
            r.stderr,
        )

        # list
        r = tool('list')
        check(
            'list shows every package',
            all(p in r.stdout for p in ('T2.3', 'T2.4', 'T2.5')),
            r.stdout,
        )
        check('no herdr raw output anywhere', 'user@host' not in r.stdout)
        log = fake_world()['log']
        check(
            'nothing closes a pane or tab',
            not any(
                c[:2] in (['pane', 'close'], ['tab', 'close']) for c in log
            ),
        )
        prompts = [c for c in log if c[:2] == ['agent', 'prompt']]
        check(
            'every prompt names the brief, the SHA and the report',
            all(
                str(brief) in c[3]
                and re.search('[0-9a-f]{40}', c[3])
                and 'Write your report to' in c[3]
                for c in prompts
            ),
        )
    print(f'{len(FAILURES)} failed' if FAILURES else 'all passed')
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
