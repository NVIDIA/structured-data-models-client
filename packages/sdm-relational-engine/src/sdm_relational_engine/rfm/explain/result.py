# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, overload

import pandas as pd
import sdm_relational_engine.utils.display as display
from rich.console import Console
from rich.markdown import Markdown
from sdm_relational_engine.rfm.base.casting import CastMixin
from sdm_relational_engine.rfm.explain.gradient import GraphGradientScore
from sdm_relational_engine.utils.environment import in_notebook


@dataclass(repr=False)
class ExplainConfig(CastMixin):
    r"""Configuration for explainability.

    .. warning::

        **The natural-language summary is generated off-machine.** With
        ``skip_summary=False`` (the default), the ``openai`` package installed
        (``pip install 'sdm-relational-engine[explain]'``), and a discoverable
        API key, the client sends the predictive query, returned predictions,
        cohort analysis, and subgraph attribution to an OpenAI-compatible
        chat-completions endpoint. The attribution includes the
        **raw cell values** of the explained entity's subgraph. Unless
        ``KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL`` points elsewhere, the endpoint
        is OpenAI's ``https://api.openai.com/v1/``, a non-NVIDIA service. The
        key comes only from ``KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY``. Setting
        that variable enables the call; no other key does.

        Pass ``skip_summary=True`` (e.g.
        ``predict(..., explain=dict(skip_summary=True))``) to turn it off. The
        structured explanation on ``.cohorts`` / ``.subgraphs`` is computed by
        the NIM and is unaffected. Nothing leaves the machine when no API key
        is discoverable or ``openai`` is not installed.

    Args:
        skip_summary: Whether to skip generating a human-readable summary of
            the explanation. The summary's LLM endpoint is configured once via
            the environment: the API key from
            ``KUMO_RELATIONAL_EXPLAIN_LLM_API_KEY``,
            ``KUMO_RELATIONAL_EXPLAIN_LLM_BASE_URL`` (for any OpenAI-compatible
            endpoint, including a self-hosted one),
            ``KUMO_RELATIONAL_EXPLAIN_LLM_MODEL`` (default
            ``gpt-4.1-mini-2025-04-14``), and
            ``KUMO_RELATIONAL_EXPLAIN_LLM_TIMEOUT`` (default 20s).
    """

    skip_summary: bool = False


@dataclass(repr=False)
class Explanation:
    r"""A prediction together with an account of what drove it.

    Attributes:
        prediction: The prediction this explains.
        summary: The account in prose.
        details: The explanation payload, structured attribution included
            where the model provides it. An answer that is not a mapping
            is wrapped as ``{'format': 'unknown', 'value': ...}`` rather
            than dropped. Present even when ``skip_summary`` left
            ``summary`` empty.
        warning: What the explanation could not establish, when
            anything; :obj:`None` when nothing stood in the way.
    """

    prediction: pd.DataFrame
    summary: str
    details: Any
    warning: str | None = None

    def _structured_details(self) -> dict[str, Any]:
        r"""Return the inner cohort/subgraph payload for a ``kumo_rfm_v2_1``
        explanation, or an empty mapping for any other format.
        """
        if (
            isinstance(self.details, dict)
            and self.details.get('format') == 'kumo_rfm_v2_1'
        ):
            inner = self.details.get('details')
            if isinstance(inner, dict):
                return inner
        return {}

    @property
    def cohorts(self) -> list[Any]:
        r"""Column-level cohort analysis (global view), when available."""
        value = self._structured_details().get('cohorts')
        return list(value) if isinstance(value, list) else []

    @property
    def subgraphs(self) -> list[Any]:
        r"""Cell-level subgraph attribution (local view), when available."""
        value = self._structured_details().get('subgraphs')
        return list(value) if isinstance(value, list) else []

    @property
    def feature_importance(self) -> GraphGradientScore:
        r"""Gradient feature importance, as in the old client.

        Sums the explained entity's per-cell subgraph attribution into
        per-table, per-column scores and returns a :class:`GraphGradientScore`:
        ``fi[table][column]`` -> score, ``fi[table].total_score``,
        ``fi.print_summary(normalize=...)``, ``fi.normalize()``,
        ``fi.to_pandas()``. Empty when no subgraph attribution is present.
        """
        scores = GraphGradientScore()
        for table, column, cell in self._iter_cells():
            value = cell.get('score')
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            table_scores = scores[str(table)]
            table_scores.columns[column] = table_scores.columns.get(
                column, 0.0
            ) + float(value)
        return scores

    def _iter_cells(
        self,
    ) -> Iterator[tuple[str, str, dict[str, Any]]]:
        r"""Yield ``(table, column, cell)`` over every attributed cell.

        The NIM returns each subgraph as ``{'tables': {table: {node_id:
        {'cells': {column: {'value', 'score'}}}}}}``; the table name is the key
        under ``tables`` and a table may span several nodes.
        """
        for entry in self.subgraphs:
            if not isinstance(entry, dict):
                continue
            tables = entry.get('tables')
            if not isinstance(tables, dict):
                continue
            for table, nodes in tables.items():
                if not isinstance(nodes, dict):
                    continue
                for node in nodes.values():
                    if not isinstance(node, dict):
                        continue
                    cells = node.get('cells')
                    if not isinstance(cells, dict):
                        continue
                    for column, cell in cells.items():
                        if isinstance(cell, dict):
                            yield table, column, cell

    @overload
    def __getitem__(self, index: Literal[0]) -> pd.DataFrame:
        pass

    @overload
    def __getitem__(self, index: Literal[1]) -> str:
        pass

    def __getitem__(self, index: int) -> pd.DataFrame | str:
        if index == 0:
            return self.prediction
        if index == 1:
            return self.summary
        raise IndexError('Index out of range')

    def __iter__(self) -> Iterator[pd.DataFrame | str]:
        return iter((self.prediction, self.summary))

    def __repr__(self) -> str:
        return str((self.prediction, self.summary))

    def __str__(self) -> str:
        console = Console(soft_wrap=True)
        with console.capture() as cap:
            if self.warning is not None:
                console.print(f'[bold yellow]Warning:[/] {self.warning}\n')
            console.print(display.to_rich_table(self.prediction))
            console.print(Markdown(self.summary))
        return cap.get()[:-1]

    def print(self) -> None:
        r"""Prints the explanation."""
        if in_notebook():
            if self.warning is not None:
                display.message(f'**Warning:** {self.warning}')
            display.dataframe(self.prediction)
            display.message(self.summary)
        else:
            print(self)

    def _ipython_display_(self) -> None:
        self.print()


def extract_explanation(
    prediction: pd.DataFrame,
    explain_config: ExplainConfig,
) -> tuple[pd.DataFrame, str, dict[str, Any], str | None]:
    # The SaaS backend returned ``summary``/``warning`` at the top level; the
    # Universal TFM NIM wraps the native driver output as ``{format, details}``
    # and emits structured attribution only. Read both shapes so a server-side
    # summary is used as soon as one is available.
    if 'EXPLANATION' not in prediction or prediction.empty:
        raise RuntimeError(
            'Prediction response did not include requested explanation.'
        )
    raw_explanation = prediction['EXPLANATION'].iloc[0]
    if isinstance(raw_explanation, dict):
        details = dict(raw_explanation)
    else:
        details = {
            'format': 'unknown',
            'value': raw_explanation,
        }

    nested = details.get('details')
    nested = nested if isinstance(nested, dict) else {}

    summary = ''
    if not explain_config.skip_summary:
        maybe_summary = details.get('summary')
        if not isinstance(maybe_summary, str):
            maybe_summary = nested.get('summary')
        if isinstance(maybe_summary, str):
            summary = maybe_summary

    warning = details.get('warning')
    if warning is None:
        warning = nested.get('warning')
    if warning is not None:
        warning = str(warning)
    return prediction.drop(columns=['EXPLANATION']), summary, details, warning
