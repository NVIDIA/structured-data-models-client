# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Records ``rs1-draws.json``, RS-1's draws as numpy makes them (T1.22).

Run once, by hand, under the reference's numpy (2.5.3); CI never runs it,
and the vectors are not recorded again to pass a dependency bump (F4):

    python rust/crates/sdm-engine/tests/rs1_record.py OUT.json

It writes two sections, each from a fresh ``default_rng(seed)``:

- ``model``: T0.6's model of the example shuffle, for its four seeds (0,
  42, 123456789 and 2**70): the seed sequence's state, five raw 64-bit
  draws, then ``shuffle(arange(50))`` on a second fresh generator, as
  ``sample_target`` shuffles.
- ``intervals``: numpy's bounded draw itself, ``random_interval`` in
  ``distributions.c``, called in order on one generator for a list of
  maxima that crosses the 32-bit path's edge (``2**32 - 1``) and the 64-bit
  path to ``2**64 - 1``, with 32-bit draws after 64-bit ones, so the kept
  half of a 64-bit draw is seen to outlive them. Python has no binding for
  ``random_interval``, so this compiles a three-line C shim against the
  ``libnpyrandom.a`` numpy ships for extension authors and calls it through
  ``ctypes`` on the generator's own ``bitgen_t``: the draws are numpy's
  compiled code, not a model of it.
"""

from __future__ import annotations

import ctypes
import json
import pathlib
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

SEEDS = (0, 42, 123456789, 2**70)
SHUFFLED = 50
MAXIMA = (
    0,
    1,
    2,
    3,
    7,
    100,
    255,
    256,
    1000,
    65535,
    65536,
    2**31 - 1,
    2**31,
    2**32 - 2,
    2**32 - 1,
    2**32,
    2**33 + 5,
    2**48,
    5,
    2**63 - 1,
    2**63,
    2**64 - 2,
    2**64 - 1,
    9,
    2**32 - 1,
    2**40,
    1,
    6,
)

SHIM = """
#include <numpy/random/distributions.h>
uint64_t shim_interval(bitgen_t *state, uint64_t max) {
    return random_interval(state, max);
}
"""


def _shim(work: pathlib.Path) -> ctypes.CDLL:
    root = pathlib.Path(np.__file__).parent
    include = root / '_core' / 'include'
    library = root / 'random' / 'lib' / 'libnpyrandom.a'
    source = work / 'shim.c'
    source.write_text(SHIM, encoding='ascii')
    shared = work / 'shim.so'
    subprocess.run(
        [
            'cc',
            '-shared',
            '-fPIC',
            '-O2',
            f'-I{include}',
            f'-I{sysconfig.get_paths()["include"]}',
            str(source),
            str(library),
            '-o',
            str(shared),
            # libnpyrandom's other objects name Python's symbols, which
            # the interpreter loading the shim provides.
            *(
                ['-undefined', 'dynamic_lookup']
                if sys.platform == 'darwin'
                else []
            ),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(shared))
    lib.shim_interval.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.shim_interval.restype = ctypes.c_uint64
    return lib


def _hex(value: int) -> str:
    return hex(int(value))


def record() -> dict[str, object]:
    model = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        state = np.random.SeedSequence(seed).generate_state(4, np.uint64)
        raw = rng.bit_generator.random_raw(5)
        shuffled = np.arange(SHUFFLED)
        np.random.default_rng(seed).shuffle(shuffled)
        model.append(
            {
                'seed': _hex(seed),
                'state': [_hex(v) for v in state],
                'raw': [_hex(v) for v in raw],
                'shuffle': [int(v) for v in shuffled],
            }
        )
    intervals = []
    with tempfile.TemporaryDirectory() as work:
        lib = _shim(pathlib.Path(work))
        for seed in SEEDS:
            bitgen = np.random.PCG64(seed)
            pointer = bitgen.ctypes.bit_generator.value
            draws = [
                [_hex(top), _hex(lib.shim_interval(pointer, top))]
                for top in MAXIMA
            ]
            intervals.append({'seed': _hex(seed), 'draws': draws})
    return {
        'format': 'sdm-rs1-draws/1',
        'source': (
            f'numpy {np.__version__}: T0.6 model seeds; per seed a fresh '
            'default_rng(seed): SeedSequence(seed).generate_state(4, '
            'uint64), bit_generator.random_raw(5), and on another fresh '
            f'generator shuffle(arange({SHUFFLED})); intervals: '
            'random_interval(bitgen, max) from libnpyrandom on one fresh '
            'PCG64(seed) per seed, for each max in order, each pair '
            '[max, draw]'
        ),
        'model': model,
        'intervals': intervals,
    }


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: rs1_record.py OUT.json')
    text = json.dumps(record(), separators=(',', ':')) + '\n'
    pathlib.Path(sys.argv[1]).write_text(text, encoding='ascii')
