# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Per-column cohort statistics over a request's context rows."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sdm_client.explanations.decoder import GradientDecoder


@dataclass(frozen=True)
class _Cohorts:
    table: str
    hop: int
    column: str
    stype: str | None
    cohorts: list[str]
    shares: np.ndarray  # [cohorts]
    means: np.ndarray  # [cohorts, outputs]
    score: float


class ColumnAnalysis:
    r"""Column analysis of one payload's context, as kumo-ml's RFM computes it.

    Each analysed column is split into cohorts: numerical values and
    ``COUNT(*)`` (rows linked to an entity) into quantile bins, categories into
    one cohort each, timestamps into time-before-anchor buckets. Each cohort
    carries its population share and its mean outcome: one mean per class for
    classification (classes ordered by frequency in the context labels),
    the plain mean for regression. A column's score is the population-weighted
    standard deviation of its cohort means, averaged over classes, then
    normalized so all scores sum to 1.

    Relational columns are analysed per entity: an entity's related rows at one
    hop are averaged first, then entities are grouped. The task table itself is
    not analysed, nor are keys, ``instance_id`` or the target.
    """

    NUM_COHORTS = 10
    MAX_CATEGORIES = 500
    MAX_CLASSES = 30
    COUNT = 'COUNT(*)'
    OTHER = 'OTHER(*)'
    NA = 'N/A'
    #: The one table a tabular payload carries is reported under this name.
    TABULAR_TABLE = 'instance_table'

    _INSTANCE_ID = 'instance_id'
    _STYPES = frozenset(
        {'numerical', 'categorical', 'ID', 'multicategorical', 'timestamp'}
    )
    _SECONDS = {
        'hours': 60 * 60,
        'days': 60 * 60 * 24,
        'weeks': 60 * 60 * 24 * 7,
        'months': 60 * 60 * 24 * 30,
    }
    _SEPARATORS = frozenset({',', ';', ':', '|', '\t'})

    def __init__(self, payload: Mapping[str, Any]) -> None:
        task = payload['task']
        context = payload['context']
        instance = self._frame(context['instance_table'])
        target = instance[task['target']['column_name']]
        self._classes: list[Any] | None = None
        if task['kind'] == 'regression':
            outcome = pd.to_numeric(target).to_numpy(float)[:, None]
        else:
            self._classes = self._by_frequency(target)[: self.MAX_CLASSES]
            outcome = np.stack(
                [(target == label).to_numpy(float) for label in self._classes],
                axis=1,
            )
        self._outcome: np.ndarray = outcome
        self._entities = len(instance)
        self._columns: list[_Cohorts] = []
        if 'related_tables' in context:
            self._relational(payload, instance)
        else:
            self._tabular(payload, instance)
        total = sum(column.score for column in self._columns)
        if total > 0:
            self._columns = [
                _Cohorts(**{**vars(column), 'score': column.score / total})
                for column in self._columns
            ]
        self._columns.sort(key=lambda column: column.score, reverse=True)

    def to_pandas(self) -> pd.DataFrame:
        r"""One row per (table, hop, column, cohort, class).

        Columns by normalized score, highest first (ties keep context order:
        tables as sent, hops ascending, ``COUNT(*)`` then columns as sent);
        cohorts in their own order; classes by frequency. ``class`` is
        ``None`` for regression.
        """
        classes = self._classes if self._classes is not None else [None]
        rows = [
            (
                column.table,
                column.hop,
                column.column,
                column.stype,
                cohort,
                label,
                float(column.shares[at]),
                float(column.means[at, k]),
                column.score,
            )
            for column in self._columns
            for at, cohort in enumerate(column.cohorts)
            for k, label in enumerate(classes)
        ]
        return pd.DataFrame(
            rows,
            columns=[
                'table',
                'hop',
                'column',
                'stype',
                'cohort',
                'class',
                'share',
                'mean',
                'score',
            ],
        )

    # Payload ##################################################################

    def _tabular(
        self, payload: Mapping[str, Any], instance: pd.DataFrame
    ) -> None:
        described = payload['schema']['instance_table']
        excluded = {
            payload['task']['target']['column_name'],
            *self._listed(described.get('primary_key')),
        }
        stypes = {
            column: spec.get('stype')
            for column, spec in described['columns'].items()
            if column in instance.columns and column not in excluded
        }
        rows = len(instance)
        # Each row is its own entity at hop 0; there is no anchor.
        self._table(
            self.TABULAR_TABLE,
            instance,
            np.zeros(rows, dtype=int),
            np.arange(rows),
            stypes,
            anchor=None,
            count=False,
        )

    def _relational(
        self, payload: Mapping[str, Any], instance: pd.DataFrame
    ) -> None:
        schema = payload['schema']
        context = payload['context']
        # The decoder's BFS reads the half under ``predict``; point it at ours.
        hops = GradientDecoder._hops_from_entity(
            {'schema': schema, 'predict': context}
        )
        entity_of = {
            value: at
            for at, value in enumerate(instance[self._INSTANCE_ID].tolist())
        }
        anchor_column = payload['task'].get('anchor_time_column')
        anchor = None
        if anchor_column and anchor_column in instance.columns:
            anchor = self._seconds(instance[anchor_column])
        keys: dict[str, set[str]] = {}
        for link in schema.get('relationships') or []:
            for side in ('source', 'target'):
                if link.get(f'{side}_table') is not None:
                    keys.setdefault(link[f'{side}_table'], set()).update(
                        link.get(f'{side}_columns') or []
                    )
        for name, table in context['related_tables'].items():
            frame = self._frame(table)
            hop = pd.Series(hops[name], dtype=object)
            reached = hop.notna().to_numpy()
            if not reached.any():
                continue
            frame = frame[reached].reset_index(drop=True)
            described = schema['related_tables'][name]
            excluded = {
                self._INSTANCE_ID,
                *self._listed(described.get('primary_key')),
                *keys.get(name, ()),
            }
            stypes = {
                column: spec.get('stype')
                for column, spec in described['columns'].items()
                if column in frame.columns and column not in excluded
            }
            self._table(
                name,
                frame,
                hop[reached].to_numpy(int),
                frame[self._INSTANCE_ID].map(entity_of).to_numpy(int),
                stypes,
                anchor=anchor,
                count=True,
            )

    # Collection ###############################################################

    def _table(
        self,
        name: str,
        frame: pd.DataFrame,
        hops: np.ndarray,
        entity: np.ndarray,
        stypes: Mapping[str, str | None],
        *,
        anchor: np.ndarray | None,
        count: bool,
    ) -> None:
        r"""Every analysed column of one table, per hop the table is reached at.

        Category mappings and keep/drop decisions are made once over the
        table's rows at all hops, as kumo-ml's encoders are.
        """
        prepared: dict[str, tuple[str, Any]] = {}
        for column, stype in stypes.items():
            if stype not in self._STYPES:
                continue
            if stype == 'timestamp' and anchor is None:
                continue
            feature = self._prepare(frame[column], stype)
            if feature is not None:
                prepared[column] = feature
        for hop in sorted(set(hops.tolist())):
            at = hops == hop
            linked = entity[at]
            if count:
                links = np.bincount(linked, minlength=self._entities)
                self._add(
                    name, hop, self.COUNT, None, self._ranges(links, True)
                )
            for column, (kind, data) in prepared.items():
                if kind == 'numerical':
                    result = self._ranges(self._means(data[at], linked), False)
                elif kind == 'timestamp':
                    assert anchor is not None
                    result = self._times(data[at], linked, anchor)
                else:
                    index, labels, other = data
                    if kind == 'categorical':
                        pairs = (linked, index[at])
                    else:
                        rows = np.flatnonzero(at)
                        keep = np.isin(index[0], rows)
                        position = np.searchsorted(rows, index[0][keep])
                        pairs = (linked[position], index[1][keep])
                    shares = self._shares(pairs, linked, len(labels))
                    result = self._categories(shares, labels, other)
                self._add(name, hop, column, stypes[column], result)

    def _prepare(self, values: pd.Series, stype: str) -> tuple[str, Any] | None:
        r"""A column's per-row data, or ``None`` if kumo-ml's encoder drops it.

        A categorical becomes ``(index, labels, other)``: each row's cohort,
        the cohort labels (N/A last) and which label is the merged tail. A
        multicategorical's index is ``(row, cohort)`` pairs.
        """
        if stype == 'numerical':
            numbers = pd.to_numeric(values, errors='coerce').to_numpy(float)
            numbers = np.where(np.isfinite(numbers), numbers, np.nan)
            if np.isnan(numbers).all():
                return None
            return 'numerical', numbers
        if stype == 'timestamp':
            seconds = self._seconds(values)
            if np.isnan(seconds).all():
                return None
            return 'timestamp', seconds
        present = values[values.notna() & (values != '')]
        if present.empty:
            return None
        if stype == 'multicategorical':
            return self._multicategorical(values, present)
        if not self._categorical_kept(present, stype):
            return None
        mapping = self._by_frequency(present)
        labels = [str(value) for value in mapping]
        other = np.zeros(len(labels) + 1, dtype=bool)
        tail = None
        if len(mapping) > self.MAX_CATEGORIES:
            mapping = mapping[: self.MAX_CATEGORIES - 1]
            labels = [*labels[: self.MAX_CATEGORIES - 1], self.OTHER]
            tail = self.MAX_CATEGORIES - 1
            other = np.zeros(self.MAX_CATEGORIES + 1, dtype=bool)
            other[tail] = True
        labels.append(self.NA)
        position = {value: at for at, value in enumerate(mapping)}
        index = np.array(
            [
                self._category(value, position, tail, len(labels) - 1)
                for value in values.tolist()
            ],
            dtype=int,
        )
        return 'categorical', (index, labels, other)

    @staticmethod
    def _category(
        value: Any, position: Mapping[Any, int], tail: int | None, na: int
    ) -> int:
        if value is None or value == '' or pd.isna(value):
            return na
        if value in position:
            return position[value]
        # Only reached for categories cut by the cap.
        return tail if tail is not None else na

    def _multicategorical(
        self, values: pd.Series, present: pd.Series
    ) -> tuple[str, Any] | None:
        separator = self._separator(present)
        cells = [self._split(value, separator) for value in values.tolist()]
        exploded = pd.Series([item for cell in cells for item in cell])
        mapping = self._by_frequency(exploded)
        if len(mapping) > self.MAX_CATEGORIES:
            return None
        position = {value: at for at, value in enumerate(mapping)}
        rows = [at for at, cell in enumerate(cells) for _ in cell]
        index = [position[item] for cell in cells for item in cell]
        labels = [str(value) for value in mapping]
        other = np.zeros(len(labels), dtype=bool)
        return 'multicategorical', (
            (np.array(rows, dtype=int), np.array(index, dtype=int)),
            labels,
            other,
        )

    @staticmethod
    def _split(value: Any, separator: str | None) -> list[Any]:
        if value is None or value == '' or pd.isna(value):
            return []
        if separator is None:
            return [value]
        return [item for item in str(value).split(separator) if item != '']

    @classmethod
    def _separator(cls, present: pd.Series) -> str | None:
        r"""The most frequent separator character in the first 1000 cells."""
        text = '\n'.join(str(value) for value in present.iloc[:1000])
        counts: dict[str, int] = {}
        for char in text:
            if char in cls._SEPARATORS:
                counts[char] = counts.get(char, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda char: counts[char])

    @staticmethod
    def _categorical_kept(present: pd.Series, stype: str) -> bool:
        r"""kumo-ml's ``infer_categorical_encoder`` / ``infer_id_encoder``."""
        count = len(present)
        frequency = present.value_counts()
        unique = len(frequency)
        if stype == 'ID' and unique > 4000:
            return unique <= 0.8 * count
        if count < 100:
            return True
        threshold = int(count * (0.3 / math.sqrt(count)))
        min_occ = max(int(math.sqrt(unique)), 2)
        passing = int((frequency > min_occ).sum())
        return passing <= 4000 and passing <= 20 * threshold

    def _means(self, values: np.ndarray, entity: np.ndarray) -> np.ndarray:
        r"""Per entity, the mean of its finite values; NaN if it has none."""
        valid = ~np.isnan(values)
        sums = np.bincount(
            entity[valid], weights=values[valid], minlength=self._entities
        )
        counts = np.bincount(entity[valid], minlength=self._entities)
        means = np.full(self._entities, np.nan)
        np.divide(sums, counts, out=means, where=counts > 0)
        return means

    def _shares(
        self,
        pairs: tuple[np.ndarray, np.ndarray],
        rows: np.ndarray,
        width: int,
    ) -> np.ndarray:
        r"""Per entity, the share of its ``rows`` in each cohort; 0 if none."""
        sums = np.zeros((self._entities, width))
        np.add.at(sums, pairs, 1.0)
        counts = np.bincount(rows, minlength=self._entities)[:, None]
        return np.divide(
            sums, counts, out=np.zeros_like(sums), where=counts > 0
        )

    def _times(
        self, times: np.ndarray, entity: np.ndarray, anchor: np.ndarray
    ) -> tuple[list[str], np.ndarray, np.ndarray, float] | None:
        r"""Time before the entity's anchor, in ``NUM_COHORTS`` buckets."""
        known = times[~np.isnan(times)]
        low, high = (
            (int(known.min()), int(known.max())) if known.size else (0, 0)
        )
        window, unit = self._time_window(high - low)
        before = (anchor[entity] - times) / self._SECONDS[unit]
        valid = before >= 0  # NaN and future-looking rows drop out.
        edges = np.arange(0, (self.NUM_COHORTS + 1) * window, window)
        index = np.searchsorted(edges[1:-1], before[valid], side='left')
        shares = self._shares(
            (entity[valid], index), entity[valid], self.NUM_COHORTS
        )
        labels = [
            f'[{start} {unit}, {end} {unit})'
            for start, end in zip(edges[:-1], edges[1:], strict=True)
        ]
        return self._categories(
            shares, labels, np.zeros(len(labels), dtype=bool)
        )

    @classmethod
    def _time_window(cls, seconds: int) -> tuple[int, str]:
        r"""kumo-ml's ``get_time_cohort``: bucket width and unit for a span."""
        if seconds == 0:
            return 1, 'months'
        hours = seconds // (60 * 60)
        days = hours // 24
        weeks = days // 7
        months = days // 30
        n = cls.NUM_COHORTS
        if hours < 7 * n:
            return max(hours // n, 1), 'hours'
        if days < 5 * n:
            return max(days // n, 1), 'days'
        if weeks < 3 * n:
            return max(weeks // n, 1), 'weeks'
        return max(months // n, 1), 'months'

    # Aggregation ##############################################################

    def _ranges(
        self, values: np.ndarray, is_count: bool
    ) -> tuple[list[str], np.ndarray, np.ndarray, float] | None:
        r"""kumo-ml's ``aggregate_range``: entities in quantile bins."""
        values = values.astype(float)
        valid = ~np.isnan(values)
        values, outcome = values[valid], self._outcome[valid]
        if values.size == 0:
            return None
        bins = np.quantile(values, np.linspace(0, 1, self.NUM_COHORTS + 1))
        if is_count:
            bins = np.ceil(bins)
        bins = np.unique(bins)
        if bins.size == 1:
            bins = np.concatenate([bins, bins])
        # A zero count keeps a bin of its own.
        zero_bin = is_count and bins[0] == 0
        boundaries = bins[:-1] if zero_bin else bins[1:-1]
        width = bins.size if zero_bin else bins.size - 1
        index = np.searchsorted(boundaries, values, side='left')
        counts = np.bincount(index, minlength=width).astype(float)
        sums = np.zeros((width, outcome.shape[1]))
        np.add.at(sums, index, outcome)
        keep = counts > 0
        counts = counts[keep]
        means = sums[keep] / counts[:, None]
        edges = np.concatenate(
            [bins[:1], (bins if zero_bin else bins[1:])[keep]]
        )
        text = [f'{value:.4g}' for value in edges.tolist()]
        cohorts = []
        for at, (start, end) in enumerate(
            zip(text[:-1], text[1:], strict=True)
        ):
            if is_count and at == len(text) - 2:
                end = f'{end}+'  # Links beyond the sampled ones may exist.
            cohorts.append(
                f'[{start} - {end}]' if at == 0 else f'({start} - {end}]'
            )
        return cohorts, counts / counts.sum(), means, self._score(means, counts)

    def _categories(
        self, shares: np.ndarray, labels: list[str], other: np.ndarray
    ) -> tuple[list[str], np.ndarray, np.ndarray, float] | None:
        r"""kumo-ml's ``aggregate_category``: entities weighted by share."""
        counts = shares.sum(axis=0)
        keep = counts > 0
        if not keep.any():
            return None
        sums = shares.T @ self._outcome
        counts, other = counts[keep], other[keep]
        means = sums[keep] / counts[:, None]
        cohorts = [
            label for label, kept in zip(labels, keep, strict=True) if kept
        ]
        score = self._score(means[~other], counts[~other])
        if len(cohorts) > self.NUM_COHORTS:
            split = self.NUM_COHORTS - 1
            weight = counts[split:, None] / counts[split:].sum()
            cohorts = [*cohorts[:split], self.OTHER]
            means = np.concatenate(
                [means[:split], (weight * means[split:]).sum(0, keepdims=True)]
            )
            counts = np.concatenate([counts[:split], [counts[split:].sum()]])
        return cohorts, counts / counts.sum(), means, score

    @staticmethod
    def _score(means: np.ndarray, counts: np.ndarray) -> float:
        r"""Population-weighted std (Bessel) of cohort means, class-averaged."""
        total = counts.sum()
        if total <= 1:
            return 0.0
        center = (means * counts[:, None]).sum(0) / total
        spread = (counts[:, None] * (means - center) ** 2).sum(0) / (total - 1)
        return float(np.sqrt(spread).mean())

    def _add(
        self,
        table: str,
        hop: int,
        column: str,
        stype: str | None,
        result: tuple[list[str], np.ndarray, np.ndarray, float] | None,
    ) -> None:
        if result is None:
            return
        cohorts, shares, means, score = result
        self._columns.append(
            _Cohorts(table, hop, column, stype, cohorts, shares, means, score)
        )

    # Values ###################################################################

    @staticmethod
    def _frame(table: Mapping[str, Any]) -> pd.DataFrame:
        r"""A wire ``arrays`` table as a frame, values as sent."""
        return pd.DataFrame(
            table.get('rows') or [],
            columns=list(table['columns']),
            dtype=object,
        )

    @staticmethod
    def _seconds(values: pd.Series) -> np.ndarray:
        r"""Whole seconds since the epoch; NaN where missing."""
        stamps = pd.to_datetime(values, utc=True, errors='coerce')
        seconds = np.full(len(values), np.nan)
        known = stamps.notna().to_numpy()
        seconds[known] = np.floor(
            [stamp.timestamp() for stamp in stamps[known]]
        )
        return seconds

    @staticmethod
    def _by_frequency(values: pd.Series) -> list[Any]:
        r"""Distinct non-null values, most frequent first, ties ascending."""
        counts = values.dropna().value_counts()
        ordered = (
            counts.rename_axis('category')
            .reset_index(name='count')
            .sort_values(
                by=['count', 'category'],
                ascending=[False, True],
                kind='mergesort',
            )
        )
        return ordered['category'].tolist()

    @staticmethod
    def _listed(key: Any) -> list[str]:
        if key is None:
            return []
        return [key] if isinstance(key, str) else list(key)
