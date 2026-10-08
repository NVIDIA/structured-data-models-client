# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in for ``torch`` that sorts ties the other way round.

Only the determinism checks put this directory on ``sys.path``. It implements
the small part of the tensor API that sorting code reaches for. Its stable sort
is a true stable sort, so code that asks for one gets the same order as
numpy's. Its default sort is unstable on purpose: equal values come out in
*descending* position, which numpy's default sort does not do. Any output
that depends on which unstable sort ran therefore changes when this is
importable.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

__version__ = '0.0.0+determinism-fake'

IS_DETERMINISM_FAKE = True


class Tensor:
    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)

    def argsort(
        self, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> Tensor:
        return Tensor(_argsort(self._data, descending, stable))

    def sort(
        self, dim: int = -1, descending: bool = False, stable: bool = False
    ) -> _Sorted:
        index = _argsort(self._data, descending, stable)
        return _Sorted(Tensor(self._data[index]), Tensor(index))

    def numpy(self) -> np.ndarray:
        return self._data

    def __getitem__(self, key: Any) -> Tensor:
        if isinstance(key, Tensor):
            key = key._data
        return Tensor(self._data[key])

    def __len__(self) -> int:
        return len(self._data)


class _Sorted(NamedTuple):
    values: Tensor
    indices: Tensor


def _argsort(data: np.ndarray, descending: bool, stable: bool) -> np.ndarray:
    if data.ndim != 1:
        raise NotImplementedError('the determinism fake sorts 1-D data only')
    _, rank = np.unique(data, return_inverse=True)
    rank = rank.reshape(-1).astype(np.int64)
    position = np.arange(len(data), dtype=np.int64)
    primary = -rank if descending else rank
    secondary = position if stable else -position
    return np.lexsort((secondary, primary))


def from_numpy(data: np.ndarray) -> Tensor:
    return Tensor(data)


def as_tensor(data: Any, **_: Any) -> Tensor:
    return Tensor(data)


def tensor(data: Any, **_: Any) -> Tensor:
    return Tensor(data)


def argsort(
    input: Tensor, dim: int = -1, descending: bool = False, stable: bool = False
) -> Tensor:
    return input.argsort(dim=dim, descending=descending, stable=stable)


def sort(
    input: Tensor, dim: int = -1, descending: bool = False, stable: bool = False
) -> _Sorted:
    return input.sort(dim=dim, descending=descending, stable=stable)
