# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The parity corpus: recording today's Python, and replaying it.

The format these modules write is described in ``tests/parity/FORMAT.md``.
The command-line tools beside this package are the entry points:
``run_corpus.py``, ``diff.py``, ``scrub.py`` and ``coverage.py``.
"""

from __future__ import annotations

from pathlib import Path

#: The repository root.
REPO = Path(__file__).resolve().parents[3]
#: Where the parity files live.
PARITY = REPO / 'tests' / 'parity'
#: The default corpus directory.
TRANSCRIPTS = PARITY / 'transcripts'
#: The transcript format this code writes and reads.
FORMAT_VERSION = 1
