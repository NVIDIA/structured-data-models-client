# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A real compiled Java API addition must require a ledger mapping."""

import os
import runpy
import subprocess
from pathlib import Path

import pytest
from ledger import validate_api_inventory

ROOT = Path(__file__).resolve().parents[2]


def test_compiled_public_addition_requires_mapping(tmp_path):
    api = runpy.run_path(str(ROOT / 'scripts/snapshot_java_api.py'))
    source = tmp_path / 'Example.java'
    source.write_text(
        'package com.nvidia.sdm; public class Example { public void existing() {} }'
    )
    java_home = os.environ.get('JAVA_HOME')
    javac = str(Path(java_home) / 'bin/javac') if java_home else 'javac'
    javap = str(Path(java_home) / 'bin/javap') if java_home else 'javap'
    subprocess.run([javac, '-d', str(tmp_path), str(source)], check=True)
    before = api['collect'](tmp_path, javap)
    mapping = {symbol: {'capability': 'example'} for symbol in before}
    validate_api_inventory(before, mapping)
    source.write_text(
        source.read_text().replace(
            'public void existing()',
            'public void added() {} public void existing()',
        )
    )
    subprocess.run([javac, '-d', str(tmp_path), str(source)], check=True)
    with pytest.raises(ValueError, match='added'):
        validate_api_inventory(api['collect'](tmp_path, javap), mapping)
