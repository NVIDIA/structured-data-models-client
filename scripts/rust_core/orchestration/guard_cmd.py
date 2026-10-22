#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Blocks the banned commands before they run (T3.0, item 6), and a reviewer
launched on a SHA with no handoff stamp (item 4).

A Claude Code PreToolUse hook, wired in the repository's
``.claude/settings.json`` for the Bash and Agent tools. It reads the hook's
JSON on standard input. When the call is allowed it prints nothing and
exits 0; when it is refused it prints

    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
     "permissionDecision": "deny", "permissionDecisionReason": "..."}}

and exits 0, the reason naming the rule and the allowed alternative.

The Bash rules (the owner's, 2026-10-12 and 2026-10-21; the handbook's
"Hard rules for every agent"), each matched on the parsed command, its
``&&``/``;``/``|`` parts, ``$( )`` and backquotes, ``bash -c`` strings, and
behind ``env``, ``timeout``, ``xargs``, ``nice``, ``heavy`` and the
``perl -e 'alarm ...; exec @ARGV'`` probe wrapper; heredoc bodies are text,
never commands:

- ``rm -r`` (any spelling: ``-r``, ``-R``, ``-rf``, ``--recursive``) of a
  path outside the agent's own scratch: the session's scratchpad
  (``/tmp/claude-*/<project>/<session id>/``), a directory listed in
  ``SDM_SCRATCH`` (colon-separated), or a directory right under
  ``/mnt/localssd/tmp/`` or ``/mnt/localssd/target/`` that this session
  made with ``mkdir`` when it did not exist yet (recorded in
  ``$SDM_ORCH_STATE/scratch-claims.json``); another agent's directory
  there needs the owner's token. A target with an unexpanded variable
  or a command substitution cannot be checked and is refused;
- ``git push`` that updates or deletes ``rust-core``, ``main`` or
  ``rust-core-reference``, however spelled (the owner, Q14);
- ``git push`` with ``--force``, ``-f``, ``--force-with-lease``,
  ``--force-if-includes`` or a ``+`` refspec; ``git push --delete``,
  ``-d``, ``--prune`` or a ``:branch`` refspec;
- ``git worktree remove --force``/``-f``, ``docker rm -f``/``--force``;
- ``--no-verify`` on any git command (``git commit -n`` too), and
  ``git -c core.hooksPath=...``;
- ``sudo``;
- ``git config``, except its reads (``--get``, ``--get-all``,
  ``--get-regexp``, ``--list``/``-l``, ``git config get``/``list``);
- ``git stash``, ``git clean -f``, ``cargo clean``;
- ``git branch -D`` (or ``--delete --force``);
- ``gh run watch``;
- ``git fetch --prune``/``-p``, ``git pull --prune``, ``git remote prune``;
- ``prepush.sh --full``.

An approval token lets an owner-approved one-off through: the simple command
starts with ``SDM_APPROVED=<id>``, ``<id>`` the slug of a heading in an
owner-answers file (``$SDM_ORCH_STATE/phase-*/owner-answers*.md``, or the
files ``SDM_OWNER_ANSWERS`` lists, colon-separated). The command and token
are appended to ``$SDM_ORCH_STATE/logs/approved-commands.log``.
``guard_cmd.py --tokens`` prints the valid ids.

The Agent rule: a prompt that names a round brief (``round-briefs/...``)
or the reviewer brief (``reviewer.md``, not ``design-reviewer.md``) starts
a review round. Its candidate, the SHA on a ``Candidate: <full SHA>``
line in the prompt or the round brief it points at, needs a passing
handoff stamp (``handoff.py``) in the clone the hook runs in, else the
launch is refused naming ``handoff.py run``; other SHAs a brief names (the
reference, the previous round) are context. A round brief that
cannot be read or has none is refused. A prompt naming only ``reviewer.md``
and no candidate line has its full SHAs not yet landed checked, and with
none is docs work, and passes.

Refusals are appended to ``$SDM_ORCH_STATE/logs/guard-denied.log``. Codex
panes have no PreToolUse hook; ``rule_watch.py`` keeps watching them.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

STATE = Path(
    os.environ.get(
        'SDM_ORCH_STATE', Path.home() / '.local/share/sdm-orchestration'
    )
).expanduser()
TOKEN_VAR = 'SDM_APPROVED'
SEPARATORS = {';', '&&', '||', '|', '&', '|&', '(', ')', ';;', '\n'}
SHELLS = {'bash', 'sh', 'zsh', 'dash'}
#: The roots under which a directory is some agent's build or temp scratch.
SCRATCH_ROOTS = ('/mnt/localssd/tmp', '/mnt/localssd/target')
SHA = re.compile(r'(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])')
REVIEWER_BRIEF = re.compile(r'(?<![\w-])reviewer\.md\b')


@dataclass
class Verdict:
    rule: str
    instead: str

    def reason(self) -> str:
        return f'BLOCKED ({self.rule}); instead: {self.instead}'


# --- parsing --------------------------------------------------------------------

HEREDOC = re.compile(r'<<-?\s*([\'"]?)([A-Za-z_][A-Za-z0-9_]*)\1')


def strip_heredocs(text: str) -> str:
    """The command without heredoc bodies, which are data, not commands."""
    lines = text.split('\n')
    out: list[str] = []
    pending: list[str] = []
    for line in lines:
        if pending:
            if line.strip() == pending[0]:
                pending.pop(0)
            continue
        out.append(line)
        pending.extend(m.group(2) for m in HEREDOC.finditer(line))
    return '\n'.join(out)


def substitutions(text: str) -> Iterator[str]:
    """The text of each ``$( ... )`` and backquoted command, outermost first;
    the ones inside are found when that text is parsed in turn. Text in
    single quotes, and a character after a backslash, is literal, as the
    shell reads it (Codex r1: a quoted commit message or search pattern is
    no command)."""
    i, quoted, doubled = 0, False, False
    while i < len(text):
        ch = text[i]
        if quoted:
            quoted = ch != "'"
            i += 1
        elif ch == "'" and not doubled:
            quoted = True
            i += 1
        elif ch == '"':
            doubled = not doubled
            i += 1
        elif ch == '\\':
            i += 2
        elif text.startswith('$(', i) and not text.startswith('$((', i):
            depth, j = 1, i + 2
            while j < len(text) and depth:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            yield text[i + 2 : j - 1]
            i = j
        elif ch == '`':
            j = text.find('`', i + 1)
            if j < 0:
                return
            yield text[i + 1 : j]
            i = j + 1
        else:
            i += 1


def plain(text: str) -> str:
    """``text`` with each comment dropped to the end of its own line and
    each line end outside quotes made a ``;``, as the shell reads them
    (Claude r1 P2-2: one comment hid every later line)."""
    out: list[str] = []
    i, single, double = 0, False, False
    while i < len(text):
        ch = text[i]
        if single:
            single = ch != "'"
        elif double:
            if ch == '\\':
                out.append(text[i : i + 2])
                i += 2
                continue
            double = ch != '"'
        elif ch == '\\':
            out.append(text[i : i + 2])
            i += 2
            continue
        elif ch == "'":
            single = True
        elif ch == '"':
            double = True
        elif ch == '#' and (i == 0 or text[i - 1] in ' \t\n;&|()'):
            end = text.find('\n', i)
            i = len(text) if end < 0 else end
            continue
        elif ch == '\n':
            out.append(' ; ')
            i += 1
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def tokens(text: str) -> list[str]:
    lexer = shlex.shlex(plain(text), posix=True, punctuation_chars=';&|()')
    lexer.whitespace_split = True
    lexer.commenters = ''
    return list(lexer)


#: Shell keywords and grouping words a command may follow on its line.
KEYWORDS = {'do', 'then', 'else', 'elif', 'if', 'while', 'until', '{', '}', '!'}
#: A redirection word: ``>``, ``2>``, ``>>``, ``2>/dev/null``, ``<in``.
REDIRECT = re.compile(r'[0-9]*(>>?|<)(.*)')


def without_redirections(argv: list[str]) -> list[str]:
    """``argv`` without its redirections, which name no argument (Claude r1
    P2-3: ``rm -rf <scratch> 2>/dev/null`` read the redirection as a
    target)."""
    out: list[str] = []
    skip = False
    for word in argv:
        if skip:
            skip = False
            continue
        found = REDIRECT.fullmatch(word)
        if found and out:
            skip = not found.group(2)
            continue
        out.append(word)
    return out


def simple_commands(text: str, depth: int = 0) -> Iterator[list[str]]:
    """Every simple command in ``text``, as its words, nested ones included.
    Raises ValueError when the text cannot be parsed."""
    if depth > 8:
        raise ValueError('nested too deeply')
    # Comments go before substitutions are read (Codex r2): a commented
    # `git config` or $(sudo ...) is text, not a command.
    text = plain(strip_heredocs(text))
    for inner in substitutions(text):
        yield from simple_commands(inner, depth + 1)
    words: list[str] = []
    for token in tokens(text):
        if token in SEPARATORS or set(token) <= set(';&|()'):
            if words:
                yield words
            words = []
        else:
            words.append(token)
    if words:
        yield words


ASSIGNMENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*=')


def unwrap(words: list[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """The command behind its wrappers, the variables assigned before it, and
    the strings a nested shell runs (``bash -c``, ``eval``)."""
    env: dict[str, str] = {}
    nested: list[str] = []
    w = list(words)
    while w:
        head = os.path.basename(w[0])
        if w[0] in KEYWORDS:
            w = w[1:]
        elif ASSIGNMENT.match(w[0]):
            key, _, value = w[0].partition('=')
            env[key] = value
            w = w[1:]
        elif head == 'env':
            w = w[1:]
            while w and (w[0].startswith('-') or ASSIGNMENT.match(w[0])):
                if ASSIGNMENT.match(w[0]):
                    key, _, value = w[0].partition('=')
                    env[key] = value
                elif w[0] in ('-u', '--unset', '-C', '--chdir', '-S'):
                    w = w[1:]
                w = w[1:]
        elif head in ('nohup', 'time', 'command', 'builtin', 'exec', 'heavy'):
            w = w[1:]
            while w and w[0].startswith('-') and head in ('time', 'exec'):
                w = w[1:]
        elif head == 'nice':
            w = w[1:]
            if w and w[0] == '-n':
                w = w[2:]
            elif w and w[0].startswith('-'):
                w = w[1:]
        elif head == 'stdbuf':
            w = w[1:]
            while w and w[0].startswith('-'):
                w = w[1:]
        elif head == 'timeout':
            w = w[1:]
            while w and w[0].startswith('-'):
                takes = w[0] in ('-s', '--signal', '-k', '--kill-after')
                w = w[2:] if takes else w[1:]
            w = w[1:]  # the duration
        elif head == 'xargs':
            w = w[1:]
            while w and w[0].startswith('-'):
                takes = w[0] in ('-n', '-I', '-P', '-L', '-d', '-a', '-E', '-s')
                w = w[2:] if takes else w[1:]
        elif head == 'perl' and len(w) >= 3 and w[1] == '-e' and 'exec' in w[2]:
            # perl -e 'alarm shift; exec @ARGV' 600 CMD...
            w = w[3:]
            if w and re.fullmatch(r'[0-9]+', w[0]):
                w = w[1:]
        elif head in SHELLS:
            rest = w[1:]
            for n, word in enumerate(rest):
                if word == '-c' or (
                    word.startswith('-')
                    and not word.startswith('--')
                    and 'c' in word
                ):
                    if n + 1 < len(rest):
                        nested.append(rest[n + 1])
                    return [], env, nested
                if not word.startswith('-'):
                    # bash SCRIPT ARGS: the script is the command.
                    w = rest[n:]
                    break
            else:
                return [], env, nested
        elif head == 'eval':
            nested.append(' '.join(w[1:]))
            return [], env, nested
        else:
            break
    return w, env, nested


# --- the rules ------------------------------------------------------------------

GIT_GLOBAL_WITH_VALUE = {
    '-C',
    '-c',
    '--config-env',
    '--git-dir',
    '--work-tree',
    '--namespace',
}


def git_split(argv: list[str]) -> tuple[list[str], str | None, list[str]]:
    """git's global options, its subcommand and the subcommand's arguments."""
    opts: list[str] = []
    i = 1
    while i < len(argv) and argv[i].startswith('-'):
        opts.append(argv[i])
        if argv[i] in GIT_GLOBAL_WITH_VALUE and i + 1 < len(argv):
            opts.append(argv[i + 1])
            i += 1
        i += 1
    if i >= len(argv):
        return opts, None, []
    return opts, argv[i], argv[i + 1 :]


def short_cluster(arg: str, letter: str) -> bool:
    """``arg`` is a cluster of short options holding ``letter`` (``-rf``)."""
    return bool(re.fullmatch(r'-[A-Za-z]+', arg)) and letter in arg[1:]


def options(args: Iterable[str]) -> list[str]:
    out = []
    for arg in args:
        if arg == '--':
            break
        out.append(arg)
    return [a for a in out if a.startswith('-')]


#: The branches only the owner's merge pushes (the owner, Q14).
PROTECTED = ('rust-core', 'main', 'rust-core-reference')
#: git push options that take a value as the next word.
PUSH_VALUED = {'-o', '--push-option', '--repo', '--receive-pack', '--exec'}


def current_branch(cwd: str, opts: list[str]) -> str:
    where = cwd
    if '-C' in opts and opts.index('-C') + 1 < len(opts):
        given = opts[opts.index('-C') + 1]
        where = given if os.path.isabs(given) else os.path.join(cwd, given)
    try:
        done = subprocess.run(
            ['git', '-C', where, 'symbolic-ref', '--short', '-q', 'HEAD'],
            capture_output=True, text=True, timeout=30,
            stdin=subprocess.DEVNULL,
        )  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return done.stdout.strip()


def push_targets(args: list[str], branch: str) -> list[str]:
    """The branch names a ``git push`` updates or deletes: each refspec's
    destination (``src:dst``, ``+src:dst``, ``:dst``, ``refs/heads/x``),
    a bare name, ``HEAD`` as the branch checked out, and, with no refspec,
    the branch checked out; ``--all`` and ``--mirror`` push every branch."""
    positional: list[str] = []
    skip = False
    remote_given = False
    for arg in args:
        if skip:
            skip = False
        elif arg in PUSH_VALUED:
            skip = True
            remote_given = remote_given or arg == '--repo'
        elif arg.startswith('--repo='):
            remote_given = True
        elif not arg.startswith('-'):
            positional.append(arg)
    if '--all' in args or '--mirror' in args or '--branches' in args:
        return list(PROTECTED)
    # The remote is the first word, unless --repo gave it (Codex r3 P3).
    refspecs = positional if remote_given else positional[1:]
    if not refspecs:
        return [branch] if branch else ['HEAD']
    found = []
    for spec in refspecs:
        spec = spec.lstrip('+')
        dst = spec.rpartition(':')[2] if ':' in spec else spec
        if dst in ('HEAD', '@', ''):
            dst = branch or 'HEAD'
        found.append(dst.removeprefix('refs/').removeprefix('heads/'))
    return found


def check_git(argv: list[str], cwd: str = '.') -> Verdict | None:
    opts, sub, args = git_split(argv)
    for n, opt in enumerate(opts):
        value = (
            opts[n + 1]
            if opt in ('-c', '--config-env') and n + 1 < len(opts)
            else ''
        )
        if (
            opt.startswith('-c') or opt.startswith('--config-env')
        ) and 'hookspath' in (value or opt).lower():
            return Verdict(
                'git -c core.hooksPath skips the hooks',
                'let the hooks run; fix what they report',
            )
    flags = options(args)
    if '--no-verify' in flags or (
        sub == 'commit' and any(short_cluster(f, 'n') for f in flags)
    ):
        return Verdict(
            '--no-verify skips the hooks',
            'let the hooks run and fix what they report; a slow pre-push '
            'is the cheap tier (prepush.sh), never skipped',
        )
    if sub == 'push':
        hit = sorted(
            set(push_targets(args, current_branch(cwd, opts))) & set(PROTECTED)
        )
        if hit:
            return Verdict(
                f"git push to {', '.join(hit)}, which only the owner's merge "
                'pushes (the owner, Q14)',
                "push your phase, lane or topic branch; the owner's own merge "
                f'push runs under {TOKEN_VAR}=<owner-answers id>',
            )
        if any(
            f in ('--force', '--force-if-includes')
            or f.startswith('--force-with-lease')
            or short_cluster(f, 'f')
            for f in flags
        ):
            return Verdict(
                'git push --force rewrites a shared branch',
                'push a new branch name (a topic or archive branch) or ask '
                f'the owner and run it under {TOKEN_VAR}=<owner-answers id>',
            )
        positional = [a for a in args if not a.startswith('-')]
        if any(a.startswith('+') for a in positional):
            return Verdict(
                "a '+' refspec force-pushes",
                'push without the +, to a new branch name if it is not a '
                'fast-forward',
            )
        if any(
            f in ('--delete', '--prune', '--mirror') or short_cluster(f, 'd')
            for f in flags
        ) or any(a.startswith(':') and len(a) > 1 for a in positional[1:]):
            return Verdict(
                'git push --delete removes a remote branch',
                'leave the branch; deleting one needs the owner: run it under '
                f'{TOKEN_VAR}=<owner-answers id>',
            )
    if (
        sub == 'worktree'
        and args[:1] == ['remove']
        and any(
            f == '--force' or short_cluster(f, 'f') for f in options(args[1:])
        )
    ):
        return Verdict(
            'git worktree remove --force discards work',
            'commit or move the work, then `git worktree remove` without '
            '--force, or leave it for clean_tmp.sh',
        )
    if sub == 'clean' and any(
        f in ('--force',) or short_cluster(f, 'f') for f in flags
    ):
        return Verdict(
            'git clean -f deletes untracked files and build output',
            'remove only your own scratch, written out, or ask the owner',
        )
    if sub == 'stash':
        return Verdict(
            'git stash is shared by every worktree of the clone',
            'make a temporary WIP commit on your own branch instead',
        )
    if sub == 'config':
        reads = {
            '--get',
            '--get-all',
            '--get-regexp',
            '--list',
            '-l',
            '--get-urlmatch',
            '--get-color',
            '--get-colorbool',
        }
        writes = {
            '--unset',
            '--unset-all',
            '--add',
            '--replace-all',
            '--edit',
            '-e',
            '--rename-section',
            '--remove-section',
        }
        positional = [a for a in args if not a.startswith('-')]
        # ``git config KEY`` alone reads it.
        lone_key = len(positional) == 1 and not set(flags) & writes
        if not (
            set(flags) & reads or args[:1] in (['get'], ['list']) or lone_key
        ):
            return Verdict(
                'git config changes a configuration the clone shares',
                'pass a one-off setting with `git -c key=value <command>`',
            )
    if sub == 'branch' and (
        any(f == '-D' or (short_cluster(f, 'D')) for f in flags)
        or (
            any(f in ('--delete', '-d') for f in flags)
            and any(f in ('--force', '-f') for f in flags)
        )
    ):
        return Verdict(
            'git branch -D deletes unmerged work',
            'leave the branch; deleting it needs the owner: run it under '
            f'{TOKEN_VAR}=<owner-answers id>',
        )
    if sub in ('fetch', 'pull') and any(
        f in ('--prune', '--prune-tags')
        or short_cluster(f, 'p')
        or short_cluster(f, 'P')
        for f in flags
    ):
        return Verdict(
            f'git {sub} --prune deletes remote-tracking refs other '
            'worktrees read',
            f'git {sub} without --prune',
        )
    if sub == 'remote' and args[:1] == ['prune']:
        return Verdict(
            'git remote prune deletes remote-tracking refs other worktrees '
            'read',
            'leave them; git fetch without --prune',
        )
    return None


def scratch_roots(session: str | None) -> list[str]:
    roots = [
        os.path.realpath(p)
        for p in os.environ.get('SDM_SCRATCH', '').split(':')
        if p
    ]
    if session and re.fullmatch(r'[A-Za-z0-9_-]+', session):
        roots.extend(
            os.path.realpath(str(p))
            for p in Path('/tmp').glob(f'claude-*/*/{session}')
        )
    return roots


def inside_scratch(path: str, roots: list[str]) -> bool:
    real = os.path.realpath(path)
    for root in roots:
        if real == root or real.startswith(root.rstrip('/') + '/'):
            return True
    return False


CLAIMS = 'scratch-claims.json'


def claims() -> dict[str, str]:
    try:
        found = json.loads((STATE / CLAIMS).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def claimed(session: str | None) -> list[str]:
    """The directories under the shared build and temp roots this session
    made (:func:`claim`)."""
    return [d for d, owner in claims().items() if session and owner == session]


def claim(argv: list[str], cwd: str, session: str | None) -> None:
    """A ``mkdir`` of a new directory right under ``/mnt/localssd/tmp/`` or
    ``/mnt/localssd/target/`` makes it this session's: ``rm -r`` inside it
    is then allowed for this session alone. A directory that exists already
    (another agent's, or one made before) is never claimed."""
    if not session:
        return
    found = claims()
    changed = False
    for arg in argv[1:]:
        if arg.startswith('-'):
            continue
        path = os.path.expandvars(os.path.expanduser(arg))
        if '$' in path:
            continue
        path = os.path.realpath(
            path if os.path.isabs(path) else os.path.join(cwd, path)
        )
        for root in SCRATCH_ROOTS:
            root = os.path.realpath(root)
            rel = os.path.relpath(path, root)
            if rel.startswith('..') or rel == '.' or os.path.isabs(rel):
                continue
            top = os.path.join(root, rel.split(os.sep)[0])
            if top not in found and not os.path.exists(top):
                found[top] = session
                changed = True
    if changed:
        try:
            STATE.mkdir(parents=True, exist_ok=True)
            (STATE / CLAIMS).write_text(json.dumps(found, indent=1) + '\n')
        except OSError:
            pass


def check_rm(argv: list[str], cwd: str, session: str | None) -> Verdict | None:
    recursive = False
    targets: list[str] = []
    ended = False
    for arg in argv[1:]:
        if not ended and arg == '--':
            ended = True
        elif not ended and arg.startswith('--'):
            recursive = recursive or arg == '--recursive'
        elif not ended and arg.startswith('-') and len(arg) > 1:
            recursive = recursive or 'r' in arg[1:] or 'R' in arg[1:]
        else:
            targets.append(arg)
    if not recursive:
        return None
    if not targets:
        return Verdict(
            'rm -r of paths it is handed (xargs, find), which cannot be checked',
            'remove only inside your own scratch, its path written out',
        )
    return check_targets(targets, cwd, session, 'rm -r')


def check_targets(
    targets: list[str], cwd: str, session: str | None, what: str
) -> Verdict | None:
    roots = scratch_roots(session) + claimed(session)
    alternative = (
        "remove only inside your own scratch (the session's scratchpad, a "
        'directory in SDM_SCRATCH, or one this session made with mkdir under '
        '/mnt/localssd/{tmp,target}/), written out literally; another '
        "agent's directory needs the owner's token or clean_tmp.sh"
    )
    for target in targets:
        expanded = os.path.expandvars(os.path.expanduser(target))
        if '$' in expanded or '`' in expanded:
            return Verdict(
                f'{what} of {target!r}, which cannot be checked '
                '(an unexpanded variable or substitution)',
                alternative,
            )
        literal = re.split(r'[*?\[]', expanded, maxsplit=1)[0]
        if literal != expanded:
            literal = (
                os.path.dirname(literal)
                if not literal.endswith('/')
                else literal
            )
        path = literal if os.path.isabs(literal) else os.path.join(cwd, literal)
        if not inside_scratch(path or '/', roots):
            return Verdict(
                f'{what} of {target!r}, outside your scratch', alternative
            )
    return None


def check_simple(
    argv: list[str], cwd: str, session: str | None
) -> Verdict | None:
    head = os.path.basename(argv[0])
    if head == 'sudo':
        return Verdict('sudo', 'ask the owner; nothing here needs root')
    if head == 'rm':
        return check_rm(argv, cwd, session)
    if head == 'git':
        return check_git(argv, cwd)
    if head == 'find' and (
        '-delete' in argv
        or any(
            a in ('-exec', '-execdir', '-ok')
            and n + 1 < len(argv)
            and os.path.basename(argv[n + 1]) == 'rm'
            for n, a in enumerate(argv)
        )
    ):
        starts = []
        for arg in argv[1:]:
            if arg.startswith('-') or arg in ('(', '!'):
                break
            starts.append(arg)
        return check_targets(starts or ['.'], cwd, session, 'find -delete')
    if head == 'gh' and argv[1:3] == ['run', 'watch']:
        return Verdict(
            'gh run watch has hung agents',
            'poll `gh run view <id> --json status,conclusion` under a '
            'timeout, from the orchestrator, in the background',
        )
    if (
        head == 'docker'
        and (argv[1:2] == ['rm'] or argv[1:3] == ['container', 'rm'])
        and any(
            f == '--force' or short_cluster(f, 'f') for f in options(argv[2:])
        )
    ):
        return Verdict(
            'docker rm --force kills a running container',
            'stop it first, then docker rm without --force',
        )
    cargo = [a for a in argv[1:] if not a.startswith('+')]
    if head == 'cargo' and cargo[:1] == ['clean']:
        return Verdict(
            'cargo clean deletes a build other agents may share',
            'leave the build; clean_tmp.sh removes stale targets after a dry run',
        )
    if head == 'prepush.sh' and '--full' in argv[1:]:
        return Verdict(
            'prepush.sh --full runs the slow tier locally',
            'prepush.sh (the cheap tier); CI runs the rest',
        )
    return None


# --- approval tokens --------------------------------------------------------------


def slug(heading: str) -> str:
    text = re.sub(r'[^a-z0-9 -]', '', heading.lower().replace('–', '-'))
    return re.sub(r'-+', '-', re.sub(r'\s+', '-', text.strip())).strip('-')


def answer_files() -> list[Path]:
    given = os.environ.get('SDM_OWNER_ANSWERS')
    if given:
        return [Path(p) for p in given.split(':') if p]
    return sorted(STATE.glob('phase-*/owner-answers*.md'))


def valid_tokens() -> set[str]:
    found = set()
    for path in answer_files():
        try:
            text = path.read_text(encoding='utf-8')
        except OSError:
            continue
        for line in text.splitlines():
            match = re.match(r'#{1,6}\s+(.+)', line)
            if match:
                found.add(slug(match.group(1)))
    found.discard('')
    return found


def log(name: str, line: str) -> None:
    try:
        (STATE / 'logs').mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            '%Y-%m-%dT%H:%M:%SZ'
        )
        with open(STATE / 'logs' / name, 'a', encoding='utf-8') as handle:
            handle.write(f'{stamp} {line}\n')
    except OSError:
        pass


# --- the hook ---------------------------------------------------------------------


def check_command(
    command: str, cwd: str, session: str | None
) -> Verdict | None:
    try:
        commands = list(simple_commands(command))
    except ValueError as error:
        if re.search(
            r'\brm\s+-\w*[rR]|--force|--no-verify|\bsudo\b|git\s+(stash|config)|'
            r'branch\s+-D|gh\s+run\s+watch|--prune',
            command,
        ):
            return Verdict(
                f'a command the guard cannot parse ({error}) that looks like a '
                'banned one',
                'split it into plain commands',
            )
        return None
    tokens_cache: set[str] | None = None
    queue = list(commands)
    while queue:
        words = queue.pop(0)
        argv, env, nested = unwrap(words)
        argv = without_redirections(argv)
        for text in nested:
            try:
                queue.extend(simple_commands(text))
            except ValueError:
                return Verdict(
                    'a nested shell command the guard cannot parse',
                    'run the commands directly',
                )
        if not argv:
            continue
        if os.path.basename(argv[0]) == 'git' and any(
            name.startswith('GIT_CONFIG') and 'hookspath' in str(value).lower()
            for name, value in env.items()
        ):
            # The hooks path set through git's configuration variables
            # (GIT_CONFIG_PARAMETERS, GIT_CONFIG_KEY_<n>; T3.2).
            return Verdict(
                'git with core.hooksPath set in its environment skips the hooks',
                'let the hooks run; fix what they report',
            )
        if os.path.basename(argv[0]) == 'mkdir':
            claim(argv, cwd, session)
        if os.path.basename(argv[0]) == 'cd' and len(argv) > 1:
            target = os.path.expandvars(os.path.expanduser(argv[1]))
            cwd = target if os.path.isabs(target) else os.path.join(cwd, target)
            continue
        verdict = check_simple(argv, cwd, session)
        if verdict is None:
            continue
        token = env.get(TOKEN_VAR)
        if token:
            if tokens_cache is None:
                tokens_cache = valid_tokens()
            if token in tokens_cache:
                log(
                    'approved-commands.log',
                    f'{TOKEN_VAR}={token} {" ".join(argv)}',
                )
                continue
            return Verdict(
                f'{verdict.rule}, and {TOKEN_VAR}={token} names no heading of '
                'an owner-answers file',
                "use the slug of the owner's answer approving it "
                '(guard_cmd.py --tokens lists them)',
            )
        return verdict
    return None


def git_common_dir(cwd: str) -> Path | None:
    try:
        done = subprocess.run(
            [
                'git',
                '-C',
                cwd,
                'rev-parse',
                '--path-format=absolute',
                '--git-common-dir',
            ],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = done.stdout.strip()
    return Path(out) if done.returncode == 0 and out else None


#: A round brief the orchestrator writes for a review round.
ROUND_BRIEF = re.compile(r'[^\s`\'"()]*round-briefs/[^\s`\'"()]+')


def brief_text(name: str) -> str | None:
    """The round brief ``name`` names, read, or None when it cannot be."""
    name = name.rstrip('.,;:')
    path = name.replace('${SDM_ORCH_STATE}', str(STATE)).replace(
        '$SDM_ORCH_STATE', str(STATE)
    )
    try:
        return Path(os.path.expanduser(path)).read_text(encoding='utf-8')
    except (OSError, ValueError):
        return None


#: The candidate line every round brief carries (Claude r3): the one SHA
#: the round reviews, which needs the stamp. Other SHAs a brief names (the
#: reference, the previous round, the base) are context.
CANDIDATE = re.compile(r'^\W*Candidate:\W*([0-9a-f]{40})\b', re.MULTILINE)


def landed(sha: str, cwd: str) -> bool:
    """Whether ``sha`` is on the epic or a phase branch as fetched."""
    try:
        done = subprocess.run(
            ['git', '-C', cwd, 'branch', '-r', '--contains', sha],
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and any(
        re.fullmatch(r'origin/rust-core(-phase-[0-9]+)?', line.strip())
        for line in done.stdout.splitlines()
    )


def check_agent(prompt: str, cwd: str) -> Verdict | None:
    """A review launch: a prompt naming a round brief (``round-briefs/``)
    or the reviewer brief (``briefs/reviewer.md``). The candidate it
    reviews is the SHA on a ``Candidate: <full SHA>`` line, in the prompt
    or the round brief, and needs a passing stamp. A round brief that
    cannot be read, or has no candidate line, is refused. A prompt naming
    only ``reviewer.md`` with no candidate line has its full SHAs not yet
    landed checked instead, and with none is docs work, and passes."""
    briefs = ROUND_BRIEF.findall(prompt)
    if not briefs and not REVIEWER_BRIEF.search(prompt):
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import handoff  # noqa: PLC0415

    texts = [prompt]
    for brief in briefs:
        read = brief_text(brief)
        if read is None:
            return Verdict(
                f'a review round whose brief {brief} cannot be read',
                'write the round brief before launching the round, at a path '
                'the hook can read',
            )
        texts.append(read)
    candidates = sorted(set(CANDIDATE.findall('\n'.join(texts))))
    if not candidates and briefs:
        return Verdict(
            'a review round whose brief has no candidate line',
            "start the round brief with 'Candidate: <the full SHA under review>'",
        )
    if not candidates:
        candidates = [
            s for s in sorted(set(SHA.findall(prompt))) if not landed(s, cwd)
        ]
    if not candidates:
        return None
    common = git_common_dir(cwd)
    if common is None:
        return Verdict(
            f'a review round launched outside a git clone ({cwd})',
            'launch it from the clone holding the handoff stamps',
        )
    missing = [s for s in candidates if handoff.stamp_problem(common, s)]
    if not missing:
        return None
    return Verdict(
        'a review round on a SHA with no handoff stamp ('
        + ', '.join(s[:12] for s in missing)
        + ')',
        'the implementer or test author runs `scripts/rust_core/handoff.py '
        'run --package <ID>` on that SHA first, then reports ready',
    )


def decide(event: dict) -> Verdict | None:
    tool = event.get('tool_name')
    data = event.get('tool_input') or {}
    cwd = event.get('cwd') or os.getcwd()
    if tool == 'Bash':
        return check_command(
            str(data.get('command') or ''), cwd, event.get('session_id')
        )
    if tool in ('Agent', 'Task'):
        return check_agent(str(data.get('prompt') or ''), cwd)
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--tokens']:
        for token in sorted(valid_tokens()):
            print(token)
        return 0
    if argv[:1] in (['-h'], ['--help']):
        print(__doc__)
        return 0
    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0
    if not isinstance(event, dict):
        return 0
    verdict = decide(event)
    if verdict is None:
        return 0
    data = event.get('tool_input') or {}
    shown = str(data.get('command') or data.get('description') or '')[:300]
    log('guard-denied.log', f'{verdict.rule}: {shown!r}')
    json.dump(
        {
            'hookSpecificOutput': {
                'hookEventName': 'PreToolUse',
                'permissionDecision': 'deny',
                'permissionDecisionReason': verdict.reason(),
            }
        },
        sys.stdout,
    )
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
