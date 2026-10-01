# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd


class Mapper:
    r"""A mapper to map ``(pkey, batch)`` pairs to contiguous node IDs.

    Args:
        num_examples: The maximum number of examples to add/retrieve.
    """

    def __init__(self, num_examples: int) -> None:
        self._pkey: pd.Index | None = None
        self._indices: list[np.ndarray] = []
        self._index: pd.Index | None = None
        self._num_examples = num_examples

    def add(self, pkey: pd.Series, batch: np.ndarray) -> None:
        r"""Adds a set of ``(pkey, batch)`` pairs to the mapper.

        Args:
            pkey: The primary keys.
            batch: The batch vector.
        """
        if self._pkey is not None:
            category = np.concatenate(
                [
                    self._pkey.to_numpy(),
                    pkey.to_numpy(),
                ],
                axis=0,
            )
            category = pd.unique(category)  # Preserves ordering.
            self._pkey = pd.Index(category)
        elif pd.api.types.is_string_dtype(pkey):
            category = pd.unique(pkey)  # Preserves ordering.
            self._pkey = pd.Index(category)

        if self._pkey is not None:
            index = self._pkey.get_indexer(pkey)
        else:
            index = pkey.to_numpy()
        if np.issubdtype(index.dtype, np.integer):
            index = index.astype('int64', copy=False)
        index = self._num_examples * index + batch
        self._indices.append(index)

    def get(self, pkey: pd.Series, batch: np.ndarray) -> np.ndarray:
        r"""Retrieves the node IDs for a set of ``(pkey, batch)`` pairs.

        Returns ``-1`` for any pair not registered in the mapping.

        Args:
            pkey: The primary keys.
            batch: The batch vector.
        """
        if self._index is None and len(self._indices) == 0:
            return np.full(len(pkey), -1, dtype=np.int64)

        if self._index is not None and len(self._indices) > 0:
            self._indices = [self._index.to_numpy()] + self._indices
            self._index = None

        if self._index is None:  # Lazy build index:
            category = pd.unique(np.concatenate(self._indices))
            self._index = pd.Index(category)
            self._indices = []

        if self._pkey is not None:
            index = self._pkey.get_indexer(pkey)
        else:
            index = pkey.to_numpy()
        if np.issubdtype(index.dtype, np.integer):
            index = index.astype('int64', copy=False)
        index = self._num_examples * index + batch

        out = self._index.get_indexer(index)
        out = out.astype('int64', copy=False)

        return out
