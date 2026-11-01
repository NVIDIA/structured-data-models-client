# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-journey policy tripwires, including fixtures and worker threads."""

import contextlib
import importlib
import os
import platform
import shutil
import sys
import threading
from pathlib import Path
from unittest.mock import patch

_ACTIVE = None


def _audit(event, args):
    if _ACTIVE is not None:
        _ACTIVE.audit(event, args)


sys.addaudithook(_audit)

TARGETS = {
    'sdm_client.materializers.relational.RelationalMaterializer': (
        '_gather',
        '_get_model',
        '_build_graph',
    ),
    'sdm_client.materializers.tabular.TabularMaterializer': (
        'indices',
        'table',
    ),
    'sdm_client.adapters.relational.RelationalAdapter': (
        'prediction_plan',
        'session_plan',
        'session_prediction_plan',
        'normalize',
    ),
    'sdm_client.adapters.tabular.TabularAdapter': (
        'prediction_plan',
        'session_plan',
        'session_prediction_plan',
        'normalize',
    ),
    'sdm_client.schema.Schema': (
        '_settle',
        '_mark_key_columns',
        '_check_relationships',
    ),
    'sdm_client.schema.Column': ('_check_stype', '_check_supports_dtype'),
    'sdm_client.schema.Table': (
        '_check_key',
        '_adopt_key',
        '_check_time_column',
    ),
    'sdm_client.schema.SourceTable': (
        '_check_key',
        '_adopt_key',
        '_check_time_column',
    ),
    'sdm_client.client': ('_held_back', '_ranked', '_by_step'),
    'sdm_client.request': ('_split_from',),
    'predictive_query': ('validate',),
}


def owner(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        module, attr = name.rsplit('.', 1)
        return getattr(importlib.import_module(module), attr)


def snowflake_platform_probe(event, args):
    """Recognize only the observed driver's stdlib hardware-name probe."""
    if event != 'subprocess.Popen' or len(args) != 4:
        return False
    executable, command, cwd, env = args
    if (
        executable != 'uname'
        or command != ['uname', '-p']
        or cwd is not None
        or env is not None
    ):
        return False
    program = shutil.which('uname')
    if program is None or Path(program).resolve() != Path('/usr/bin/uname'):
        return False
    description = sys.modules.get('snowflake.connector.description')
    if description is None:
        return False
    origin = getattr(description, '__file__', None)
    if not origin:
        return False
    stdlib = platform._Processor.from_subprocess.__code__
    saw_stdlib = saw_driver = False
    frame = sys._getframe(1)
    while frame is not None:
        saw_stdlib |= frame.f_code is stdlib
        saw_driver |= (
            frame.f_code.co_name == '<module>'
            and frame.f_globals is description.__dict__
            and Path(frame.f_code.co_filename).resolve()
            == Path(origin).resolve()
        )
        frame = frame.f_back
    return saw_stdlib and saw_driver


class LegacyGuard(contextlib.AbstractContextManager):
    def __init__(self, targets=None, *, raise_caught=True):
        self.targets = TARGETS if targets is None else targets
        self.report = {
            'installed': False,
            'scope': 'whole-journey',
            'workers': [],
            'legacy_hits': [],
            'unobserved_subprocesses': [],
            'observed_subprocesses': [],
        }
        self.stack = contextlib.ExitStack()
        self.codes = {}
        self.workers = set()
        self.active = False
        self.raise_caught = raise_caught

    def __enter__(self):
        global _ACTIVE
        if os.environ.get('SDM_CORE_ROUTES'):
            raise AssertionError('parity requires unmodified default routes')
        for name, methods in self.targets.items():
            obj = owner(name)
            for method in methods:
                original = getattr(
                    obj, method
                )  # Missing tripwires fail closed.
                label = name + '.' + method
                if hasattr(original, '__code__'):
                    self.codes[original.__code__] = label

                def fail(*args, _label=label, **kwargs):
                    self.report['legacy_hits'].append(_label)
                    raise AssertionError('legacy policy executed: ' + _label)

                self.stack.enter_context(patch.object(obj, method, fail))
        self.previous = sys.getprofile()
        self.thread_previous = threading.getprofile()
        self.workers.add(f'{os.getpid()}:{threading.get_ident()}')
        sys.setprofile(self.trace)
        threading.setprofile(self.trace)
        # Python 3.12 includes already-running worker threads too. Patching
        # function objects above applies process-wide on older interpreters.
        self.all_threads = getattr(threading, 'setprofile_all_threads', None)
        if self.all_threads:
            self.all_threads(self.trace)
        self.active = True
        _ACTIVE = self
        self.report['installed'] = True
        return self

    def audit(self, event, args):
        if self.active and event in (
            'subprocess.Popen',
            'os.fork',
            'os.posix_spawn',
            'os.posix_spawnp',
        ):
            if snowflake_platform_probe(event, args):
                self.report['observed_subprocesses'].append(
                    {
                        'pid': 'not-yet-spawned',
                        'command': ['uname', '-p'],
                        'origin': 'snowflake.connector.description -> platform._Processor.from_subprocess',
                    }
                )
                return
            # Do not retain arguments: they may include credentials. A child
            # without its own observation cannot establish route coverage.
            self.report['unobserved_subprocesses'].append(event)

    def trace(self, frame, event, arg):
        if event != 'call':
            return
        self.workers.add(f'{os.getpid()}:{threading.get_ident()}')
        label = self.codes.get(frame.f_code)
        if label:
            self.report['legacy_hits'].append(label)
            raise AssertionError('legacy policy alias executed: ' + label)

    def __exit__(self, *error):
        global _ACTIVE
        _ACTIVE = None
        self.active = False
        if self.all_threads:
            self.all_threads(self.thread_previous)
        threading.setprofile(self.thread_previous)
        sys.setprofile(self.previous)
        self.stack.close()
        self.report['workers'] = sorted(self.workers)
        if os.environ.get('SDM_CORE_ROUTES'):
            raise AssertionError('journey changed default routes')
        # Errors caught inside an API still fail the journey.
        if (
            self.raise_caught
            and self.report['legacy_hits']
            and error[0] is None
        ):
            raise AssertionError('legacy policy was invoked and caught')
        return False
