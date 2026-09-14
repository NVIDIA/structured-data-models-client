# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sdm_relational_engine.api.rfm.context import Subgraph
from sdm_relational_engine.rfm.backend.local import LocalTable
from sdm_relational_engine.rfm.base import Table
from sdm_relational_engine.rfm.diagnostics import (
    GraphSanitizationReport,
    SanitizationStatus,
    TableSanitizationReport,
    TaskReferenceError,
)
from sdm_relational_engine.utils import ProgressLogger

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph


class LocalGraphStore:
    def __init__(
        self,
        graph: 'Graph',
        verbose: bool | ProgressLogger = True,
    ) -> None:

        if not isinstance(verbose, ProgressLogger):
            verbose = ProgressLogger.default(
                msg='Materializing graph',
                verbose=verbose,
            )

        with verbose as logger:
            self.df_dict, self.mask_dict = self.sanitize(graph)
            logger.log('Sanitized input data')

            self.pkey_map_dict = self.get_pkey_map_dict(graph)
            num_pkeys = sum(t.has_primary_key() for t in graph.tables.values())
            if num_pkeys > 1:
                logger.log(f'Collected primary keys from {num_pkeys} tables')
            else:
                logger.log(f'Collected primary key from {num_pkeys} table')

            self.time_dict, self.min_max_time_dict = self.get_time_data(graph)
            if len(self.min_max_time_dict) > 0:
                min_time = min(t for t, _ in self.min_max_time_dict.values())
                max_time = max(t for _, t in self.min_max_time_dict.values())
                logger.log(
                    f'Identified temporal graph from '
                    f'{min_time.date()} to {max_time.date()}'
                )
            else:
                logger.log('Identified static graph without timestamps')

            self.row_dict, self.colptr_dict = self.get_csc(graph)
            num_nodes = sum(len(df) for df in self.df_dict.values())
            num_edges = sum(len(row) for row in self.row_dict.values())
            logger.log(
                f'Created graph with {num_nodes:,} nodes and '
                f'{num_edges:,} edges'
            )

    def get_node_id(self, table_name: str, pkey: pd.Series) -> np.ndarray:
        r"""Returns the node ID given primary keys.

        Args:
            table_name: The table name.
            pkey: The primary keys to receive node IDs for.
        """
        if table_name not in self.df_dict.keys():
            raise KeyError(f"Table '{table_name}' does not exist")

        if table_name not in self.pkey_map_dict.keys():
            raise ValueError(
                f"Table '{table_name}' does not have a primary key"
            )

        if len(pkey) == 0:
            raise KeyError(f"No primary keys passed for table '{table_name}'")

        pkey_map = self.pkey_map_dict[table_name]

        # The Python type of the index's first element, then the index's own
        # dtype. The first is what this always did, and it is right for a
        # numpy-backed index -- `type(np.int64(3))` is `np.int64`, which pandas
        # accepts -- and for an object index of one uniform type, where it
        # coerces a key that arrived as text. But an object index holds
        # whatever is in cell zero, and that is a class rather than a dtype: a
        # column of `Decimal` raised `dtype '<class 'decimal.Decimal'>' not
        # understood` here. Falling back to the index's dtype answers that
        # case without giving up the coercion in the other one.
        failure: Exception | None = None
        for target in (type(pkey_map.index[0]), pkey_map.index.dtype):
            try:
                pkey = pkey.astype(target)
                break
            except (ValueError, TypeError) as e:
                # `TypeError` as well as `ValueError`: the cast raised it for
                # exactly the case the message below describes, so the message
                # never ran and the raw pandas complaint escaped instead.
                failure = e
        else:
            raise ValueError(
                f'Could not cast primary keys '
                f'{pkey.tolist()} to the expected data '
                f"type '{pkey_map.index.dtype}'"
            ) from failure

        try:
            return pkey_map.loc[pkey]['arange'].to_numpy()
        except KeyError as e:
            missing = ~np.isin(pkey, pkey_map.index)
            raise KeyError(
                f'The primary keys {pkey[missing].tolist()} do '
                f"not exist in the '{table_name}' table"
            ) from e

    def validate_entity_references(
        self,
        table_name: str,
        pkey: pd.Series,
    ) -> None:
        r"""Validate task entity references without exposing key values."""
        if table_name not in self.pkey_map_dict:
            return
        index = self.pkey_map_dict[table_name].index
        key_type = type(index[0])
        try:
            converted = pkey.astype(key_type)
            missing = ~converted.isin(index).to_numpy()
        except (TypeError, ValueError, OverflowError):
            converted_values: list[object] = []
            invalid = np.zeros(len(pkey), dtype=bool)
            for position, value in enumerate(pkey):
                try:
                    converted_values.append(key_type(value))
                except (TypeError, ValueError, OverflowError):
                    converted_values.append(None)
                    invalid[position] = True
            converted = pd.Series(converted_values)
            missing = invalid | ~converted.isin(index).to_numpy()
        if missing.any():
            raise TaskReferenceError(table_name, int(missing.sum()))

    def sanitize(
        self,
        graph: 'Graph',
    ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
        r"""Sanitizes raw data according to table schema definition:

        In particular, it:
        * converts timestamp data to `pd.Datetime`
        * drops timezone information from timestamps
        * drops duplicate primary keys
        * removes rows with missing primary keys or time values
        """
        df_dict: dict[str, pd.DataFrame] = {}
        for table_name, table in graph.tables.items():
            assert isinstance(table, LocalTable)
            df_dict[table_name] = Table._sanitize(
                df=table._data.copy(deep=False).reset_index(drop=True),
                dtype_dict={
                    column.name: column.dtype for column in table.columns
                },
                stype_dict={
                    column.name: column.stype for column in table.columns
                },
            )

        mask_dict: dict[str, np.ndarray] = {}
        reports: dict[str, TableSanitizationReport] = {}
        for table in graph.tables.values():
            df = df_dict[table.name]
            null_pkey = np.zeros(len(df), dtype=bool)
            duplicate_pkey = np.zeros(len(df), dtype=bool)
            null_time = np.zeros(len(df), dtype=bool)

            if table._primary_key is not None:
                ser = df[table._primary_key]
                null_pkey = ser.isna().to_numpy()
                duplicate_pkey = ser.duplicated().to_numpy() & ~null_pkey

            if table._time_column is not None:
                ser = df[table._time_column]
                null_time = ser.isna().to_numpy() & ~null_pkey & ~duplicate_pkey

            mask = ~(null_pkey | duplicate_pkey | null_time)
            if not mask.all():
                mask_dict[table.name] = mask

            reports[table.name] = TableSanitizationReport(
                table_name=table.name,
                input_rows=len(df),
                output_rows=int(mask.sum()),
                null_primary_key_rows=int(null_pkey.sum()),
                duplicate_primary_key_rows=int(duplicate_pkey.sum()),
                null_time_rows=int(null_time.sum()),
            )

        self.sanitization_report = GraphSanitizationReport(
            status=SanitizationStatus.AVAILABLE,
            tables=reports,
        )
        return df_dict, mask_dict

    def get_pkey_map_dict(
        self,
        graph: 'Graph',
    ) -> dict[str, pd.DataFrame]:
        pkey_map_dict: dict[str, pd.DataFrame] = {}

        for table in graph.tables.values():
            if table._primary_key is None:
                continue

            pkey = self.df_dict[table.name][table._primary_key]
            pkey_map = pd.DataFrame(
                dict(arange=range(len(pkey))),
                index=pkey,
            )
            if table.name in self.mask_dict:
                pkey_map = pkey_map[self.mask_dict[table.name]]

            if len(pkey_map) == 0:
                error_msg = f"Found no valid rows in table '{table.name}'. "
                if table.has_time_column():
                    error_msg += (
                        'Please make sure that there exists valid '
                        'non-N/A primary key and time column pairs '
                        'in this table.'
                    )
                else:
                    error_msg += (
                        'Please make sure that there exists valid '
                        'non-N/A primary keys in this table.'
                    )
                raise ValueError(error_msg)

            pkey_map_dict[table.name] = pkey_map

        return pkey_map_dict

    def get_time_data(
        self,
        graph: 'Graph',
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    ]:
        time_dict: dict[str, np.ndarray] = {}
        min_max_time_dict: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for table in graph.tables.values():
            if table._time_column is None:
                continue

            time = self.df_dict[table.name][table._time_column]
            time_dict[table.name] = time.astype(int).to_numpy() // 1000**3

            if table.name in self.mask_dict.keys():
                time = time[self.mask_dict[table.name]]
            if len(time) > 0:
                min_max_time_dict[table.name] = (time.min(), time.max())
            else:
                min_max_time_dict[table.name] = (
                    pd.Timestamp.max,
                    pd.Timestamp.min,
                )

        return time_dict, min_max_time_dict

    def get_csc(
        self,
        graph: 'Graph',
    ) -> tuple[
        dict[tuple[str, str, str], np.ndarray],
        dict[tuple[str, str, str], np.ndarray],
    ]:
        # A mapping from raw primary keys to node indices (0 to N-1):
        map_dict: dict[str, pd.Index] = {}
        # A dictionary to manage offsets of node indices for invalid rows:
        offset_dict: dict[str, np.ndarray] = {}
        for table_name in {edge.dst_table for edge in graph.edges}:
            ser = self.df_dict[table_name][graph[table_name]._primary_key]
            if table_name in self.mask_dict.keys():
                mask = self.mask_dict[table_name]
                ser = ser[mask]
                offset_dict[table_name] = np.cumsum(~mask)[mask]
            map_dict[table_name] = pd.Index(ser)

        # Build CSC graph representation:
        row_dict: dict[tuple[str, str, str], np.ndarray] = {}
        colptr_dict: dict[tuple[str, str, str], np.ndarray] = {}
        for src_table, fkey, dst_table in graph.edges:
            src_df = self.df_dict[src_table]
            dst_df = self.df_dict[dst_table]

            src = np.arange(len(src_df))
            dst = map_dict[dst_table].get_indexer(src_df[fkey])
            dst = dst.astype('int64', copy=False)
            mask = dst >= 0
            if dst_table in offset_dict:
                dst = dst + offset_dict[dst_table][dst]
            if src_table in self.mask_dict.keys():
                mask = mask & self.mask_dict[src_table]
            src, dst = src[mask], dst[mask]

            # Sort by destination/column (and time within neighborhoods):
            # `lexsort` is expensive (especially in numpy) so avoid it if
            # possible by grouping `time` and `node_id` together:
            if len(dst) > 0 and src_table in self.time_dict:
                src_time = self.time_dict[src_table][src]
                min_time = int(src_time.min())
                max_time = int(src_time.max())
                offset = (max_time - min_time) + 1
                if offset * len(dst_df) <= np.iinfo(np.int64).max:
                    index = dst * offset + (src_time - min_time)
                    perm = _argsort(index)
                else:  # Safe route to avoid `int64` overflow:
                    perm = _lexsort([src_time, dst])
            else:
                perm = _argsort(dst)

            row, col = src[perm], dst[perm]

            # Convert into compressed representation:
            colcount = np.bincount(col, minlength=len(dst_df))
            colptr = np.empty(len(colcount) + 1, dtype=colcount.dtype)
            colptr[0] = 0
            np.cumsum(colcount, out=colptr[1:])
            edge_type = (src_table, fkey, dst_table)
            row_dict[edge_type] = row
            colptr_dict[edge_type] = colptr

            # Reverse connection - no sort and no time handling needed since
            # the reverse mapping is 1-to-many.
            row, col = dst, src
            colcount = np.bincount(col, minlength=len(src_df))
            colptr = np.empty(len(colcount) + 1, dtype=colcount.dtype)
            colptr[0] = 0
            np.cumsum(colcount, out=colptr[1:])
            edge_type = Subgraph.rev_edge_type(edge_type)
            row_dict[edge_type] = row
            colptr_dict[edge_type] = colptr

        return row_dict, colptr_dict


def _argsort(input: np.ndarray) -> np.ndarray:
    if find_spec('torch') is None:
        return np.argsort(input)
    torch = import_module('torch')
    return torch.from_numpy(input).argsort().numpy()


def _lexsort(inputs: list[np.ndarray]) -> np.ndarray:
    assert len(inputs) >= 1

    try:
        torch = import_module('torch')
        out = torch.from_numpy(inputs[0]).argsort(stable=True)
    except Exception:
        return np.lexsort(inputs)  # PyTorch<1.9 without `stable` support.

    for input in inputs[1:]:
        index = torch.from_numpy(input)[out]
        index = index.argsort(stable=True)
        out = out[index]

    return out.numpy()
