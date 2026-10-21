# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import sys
import tempfile

from setuptools import setup

# The retained wire types are first-party code and the query language comes
# from ``predictive_query``, so the client carries no external ``kumo-api``
# dependency. ``pydantic`` used to arrive transitively through it; it is now
# declared directly.

if bool(int(os.getenv('WITH_RELATIONALLIB', '1'))):
    kwargs = dict(cmake_source_dir='.')
else:
    kwargs = dict()

if sys.platform == 'win32':
    # MSBuild tracks file access through `.tlog` files written beneath the
    # build directory, and their names are long. Under this package's own path
    # they pass Windows' 260-character limit, and the compiler check fails
    # before any source here is read. Build somewhere short instead.
    #
    # Named for where it is building from, because `build/` is per checkout
    # and this is not: without the digest, two checkouts on one machine would
    # share a CMake cache and a set of `.tlog` files.
    _here = os.path.dirname(os.path.abspath(__file__))
    _tag = hashlib.sha256(_here.encode('utf-8')).hexdigest()[:8]
    kwargs['options'] = {
        'build': {
            'build_base': os.path.join(tempfile.gettempdir(), f'sdm-{_tag}')
        }
    }

setup(
    install_requires=[
        # The floors every distribution that names these shares on the Rust
        # core's epic branch (decision E16). Databricks Runtime 14.3 LTS and
        # serverless environment 1 ship older ones, which the install
        # upgrades; pyarrow 8 failed there on import before any floor.
        'numpy>=1.23.5',
        'pandas>=2.2.2',
        'pyarrow>=14.0.1',
        'python-dateutil',
        'requests>=2.28.2',
        'urllib3',
        'typing_extensions>=4.5.0',
        'pydantic>=2.7',
        'structured-data-models-api==1.0.0',
        'structured-data-models-pql==1.0.0',
        'structured-data-models-connectors==1.0.0',
        'antlr4-python3-runtime==4.13.2',
        'rich>=9.0.0',
        'jinja2',
        'tabulate',
    ],
    **kwargs,
)
