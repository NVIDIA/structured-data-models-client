# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from NVIDIA/Kumo-TS, Kumo-Forecast/sdk/forecasting.py
# at af74798bbf7d020d3298f545fb9be383560a18ce: the pages of the forecast
# report, the artifacts written beside it, and the explanation as JSON.

r"""Drawing a forecast explanation, as Kumo-TS draws it."""

from __future__ import annotations

import json
import logging
import types
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sdm_client.explain._kumo_ts.channel_stability import (
    PerChannelEmbeddingStabilityReport,
)
from sdm_client.explain._kumo_ts.interpretability import (
    ForecastExplanation,
    TrajectoryStabilityReport,
)

# Module scope in the file these were taken from, and dropped by the move.
# Without them the integrated-gradients page and every warning path here
# raise NameError the moment they run.
logger = logging.getLogger(__name__)


def _save_lag_horizon_artifacts(
    out_dir: Path,
    attributions: np.ndarray,
    scores: np.ndarray | None = None,
    *,
    na_rep: str = 'nan',
) -> Path | None:
    """Write lag x horizon attribution CSVs and a heatmap PNG.

    Returns the path to the heatmap PNG if matplotlib is available, else None.
    """
    K, H = attributions.shape

    lag_cols = [f'horizon_{h}' for h in range(H)]
    df_matrix = pd.DataFrame(attributions, columns=lag_cols)
    df_matrix.insert(0, 'lag', [f'lag_{j}' for j in range(K)])
    df_matrix.to_csv(
        out_dir / 'lag_horizon_attributions.csv', index=False, na_rep=na_rep
    )

    rows = []
    for j in range(K):
        for h in range(H):
            row = {
                'lag': j,
                'horizon': h,
                'attribution': float(attributions[j, h]),
            }
            if scores is not None:
                row['score'] = float(scores[j, h])
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        out_dir / 'lag_horizon_long.csv', index=False, na_rep=na_rep
    )

    try:
        import matplotlib as mpl

        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(max(8, H * 0.15), max(6, K * 0.05)))
    im = ax.imshow(
        attributions, aspect='auto', cmap='viridis', interpolation='nearest'
    )
    ax.set_xlabel('Forecast horizon')
    ax.set_ylabel('Lag (past time step)')
    ax.set_xticks(np.linspace(0, H - 1, min(12, H), dtype=int))
    ax.set_xticklabels(np.linspace(0, H - 1, min(12, H), dtype=int))
    ax.set_yticks(np.linspace(0, K - 1, min(12, K), dtype=int))
    ax.set_yticklabels(np.linspace(0, K - 1, min(12, K), dtype=int))
    plt.colorbar(im, ax=ax, label='Attribution')
    plt.tight_layout()
    heatmap_path = out_dir / 'lag_horizon_heatmap.png'
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    return heatmap_path


def _topk_lag_steps_per_horizon(
    scores: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[np.ndarray | None, list[list[str]]]:
    """Compute marginal per-step lag weights and produce top-k summary rows."""
    if scores.ndim != 2 or scores.shape[0] < 2 or scores.shape[1] < 1:
        return None, []
    step_scores = np.vstack([scores[0:1, :], np.diff(scores, axis=0)])
    step_scores = step_scores - np.max(step_scores, axis=0, keepdims=True)
    step_probs = np.exp(step_scores)
    step_probs = step_probs / np.maximum(
        np.sum(step_probs, axis=0, keepdims=True), 1e-12
    )

    H = step_probs.shape[1]
    rows: list[list[str]] = []
    for h in range(H):
        order = np.argsort(-step_probs[:, h])[:top_k]
        row: list[str] = [f'horizon_{h}']
        for idx in order:
            lag_step = int(idx) + 1
            w = float(step_probs[idx, h])
            row.append(str(lag_step))
            row.append(f'{w:.4f}')
        while len(row) < 1 + 2 * top_k:
            row.extend(['', ''])
        rows.append(row)
    return step_probs, rows


def _semantic_flow_segments(
    flow: np.ndarray,
    *,
    context_len: int,
    forecast_horizon: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split flow_magnitudes into (history, forecast) using the same convention as
    ``interpretability._flow_segment_ratios``.

    Returns ``(history_flow, forecast_flow, boundary_index)`` where ``boundary_index``
    is the transition index that marks the first window touching the forecast segment.
    """
    f = np.asarray(flow, dtype=np.float32).reshape(-1)
    T_trans = int(f.shape[0])
    L = int(context_len)
    H = int(forecast_horizon)
    boundary = max(0, L - 1)
    hist = f[0:boundary] if T_trans > 0 else f
    fcst = f[boundary : min(T_trans, L + H - 1)] if T_trans > 0 else f
    return hist, fcst, boundary


def _save_semantic_flow_csv(
    out_dir: Path,
    flow: np.ndarray,
    *,
    context_len: int,
    forecast_horizon: int,
    na_rep: str = 'nan',
) -> Path:
    """Persist per-transition latent flow magnitudes with a segment label.

    Columns: ``transition_index, segment, flow_magnitude`` where ``segment`` is
    ``"history"`` for transitions whose window sits inside the input context and
    ``"forecast"`` for transitions whose window extends into the model-generated
    future. Transitions outside both segments (only possible when ``T-1 > L+H-1``)
    are labeled ``"tail"``.
    """
    f = np.asarray(flow, dtype=np.float32).reshape(-1)
    T_trans = int(f.shape[0])
    L = int(context_len)
    H = int(forecast_horizon)
    boundary = max(0, L - 1)
    fcst_end = min(T_trans, L + H - 1)

    segments = np.full((T_trans,), 'tail', dtype=object)
    segments[0:boundary] = 'history'
    segments[boundary:fcst_end] = 'forecast'

    df = pd.DataFrame(
        {
            'transition_index': np.arange(T_trans, dtype=np.int64),
            'segment': segments,
            'flow_magnitude': f,
        }
    )
    out_path = out_dir / 'semantic_flow.csv'
    df.to_csv(out_path, index=False, na_rep=na_rep)
    return out_path


def _semantic_flow_page(
    pdf: Any,
    plt: Any,
    *,
    explanation: ForecastExplanation,
    context_len: int,
    forecast_horizon: int,
) -> None:
    """Append a single PDF page summarizing semantic flow magnitudes.

    Layout mirrors the trajectory-stability page: title, gray blurb, a primary
    visual (line chart of flow over transitions with the history/forecast split
    annotated), two side-by-side tables (per-segment summary + diagnostics), and
    a monospace reading guide.
    """
    if getattr(explanation, 'flow_magnitudes', None) is None:
        return
    flow = np.asarray(explanation.flow_magnitudes, dtype=np.float32).reshape(-1)
    T_trans = int(flow.shape[0])
    if T_trans == 0:
        return

    hist, fcst, boundary = _semantic_flow_segments(
        flow,
        context_len=context_len,
        forecast_horizon=forecast_horizon,
    )

    def _stats(arr: np.ndarray) -> tuple[str, str, str, str, str, str]:
        a = np.asarray(arr, dtype=np.float32)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return ('--', '--', '--', '--', '--', '0')
        return (
            f'{float(np.mean(a)):.4f}',
            f'{float(np.median(a)):.4f}',
            f'{float(np.percentile(a, 95)):.4f}',
            f'{float(np.max(a)):.4f}',
            f'{float(np.var(a)):.4f}',
            f'{int(a.size)}',
        )

    hist_s = _stats(hist)
    fcst_s = _stats(fcst)

    def _fmt(x: Any) -> str:
        if x is None:
            return 'n/a'
        try:
            xv = float(x)
        except (TypeError, ValueError):
            return 'n/a'
        return f'{xv:.4f}' if np.isfinite(xv) else 'n/a'

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        'Semantic flow magnitudes',
        ha='center',
        fontsize=14,
        fontweight='bold',
    )
    fig.text(
        0.06,
        0.90,
        r'Per-step latent flow is $m_t = \Vert \mathbf{z}_{t+1} - \mathbf{z}_t \Vert_2$'
        ' over the trajectory built on\n'
        '[history; forecast]. This is the temporal signal that drives the lag x horizon\n'
        'heatmap; bigger spikes contribute more to attribution. Compare the history\n'
        'segment against the forecast segment to gauge representation-level stability.',
        ha='left',
        va='top',
        fontsize=9,
        fontfamily='DejaVu Sans',
        math_fontfamily='dejavusans',
        color='gray',
    )

    ax = fig.add_axes((0.08, 0.55, 0.84, 0.23))
    x_axis = np.arange(T_trans)
    ax.plot(
        x_axis, flow, linewidth=1.0, color='#1f77b4', label='flow magnitude'
    )
    if 0 < boundary < T_trans:
        ax.axvspan(boundary, T_trans - 1, color='#f0f0f0', alpha=0.7, zorder=0)
        ax.axvline(
            boundary,
            color='black',
            linestyle='--',
            linewidth=0.9,
            label='history / forecast split',
        )
    if hist.size > 0:
        hist_mean = float(np.nanmean(hist))
        ax.hlines(
            hist_mean,
            0,
            max(0, boundary - 1),
            colors='#2ca02c',
            linestyles=':',
            linewidth=1.2,
            label=f'history mean ({hist_mean:.3f})',
        )
    if fcst.size > 0:
        fcst_mean = float(np.nanmean(fcst))
        fcst_end_idx = max(
            boundary,
            min(T_trans, int(context_len) + int(forecast_horizon) - 1) - 1,
        )
        ax.hlines(
            fcst_mean,
            boundary,
            fcst_end_idx,
            colors='#d62728',
            linestyles=':',
            linewidth=1.2,
            label=f'forecast mean ({fcst_mean:.3f})',
        )
    ax.set_xlabel(r'Latent transition index ($t \rightarrow t+1$)')
    ax.set_ylabel(r'$\Vert \mathbf{z}_{t+1} - \mathbf{z}_t \Vert_2$')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    seg_rows = [
        ['Mean', hist_s[0], fcst_s[0]],
        ['Median', hist_s[1], fcst_s[1]],
        ['P95', hist_s[2], fcst_s[2]],
        ['Max', hist_s[3], fcst_s[3]],
        ['Variance', hist_s[4], fcst_s[4]],
        ['Transitions', hist_s[5], fcst_s[5]],
    ]
    ax_t1 = fig.add_axes((0.06, 0.32, 0.42, 0.18))
    ax_t1.axis('off')
    table_seg = ax_t1.table(
        cellText=seg_rows,
        colLabels=['Metric', 'History', 'Forecast'],
        colWidths=[0.4, 0.3, 0.3],
        loc='upper center',
        cellLoc='center',
    )
    table_seg.auto_set_font_size(False)
    table_seg.set_fontsize(8)
    table_seg.scale(1.0, 1.3)
    for c in range(3):
        cell = table_seg[(0, c)]
        cell.set_facecolor('#dddddd')
        cell.set_text_props(weight='bold')

    diag_rows = [
        [
            'Flow ratio (forecast/history)',
            _fmt(explanation.flow_ratio_forecast_vs_history),
            r'$\approx 1$ healthy, $>1.5$ OOD-volatile',
        ],
        [
            'Flow variance ratio',
            _fmt(explanation.flow_variance_ratio_forecast_vs_history),
            r'$\approx 1$ healthy, $>2$ noisy',
        ],
        [
            'Curvature ratio',
            _fmt(explanation.curvature_ratio_forecast_vs_history),
            r'$\approx 1$ healthy, $>1.5$ jaggy',
        ],
        [
            'Latent diag-Mahalanobis ratio',
            _fmt(explanation.latent_diag_mahalanobis_ratio_forecast_vs_history),
            r'$\approx 1$ healthy, $\gg 1$ OOD shift',
        ],
    ]
    ax_t2 = fig.add_axes((0.52, 0.32, 0.42, 0.18))
    ax_t2.axis('off')
    table_diag = ax_t2.table(
        cellText=diag_rows,
        colLabels=['Diagnostic', 'Value', 'Heuristic'],
        colWidths=[0.42, 0.18, 0.40],
        loc='upper center',
        cellLoc='center',
    )
    table_diag.auto_set_font_size(False)
    table_diag.set_fontsize(8)
    table_diag.scale(1.0, 1.3)
    for c in range(3):
        cell = table_diag[(0, c)]
        cell.set_facecolor('#dddddd')
        cell.set_text_props(weight='bold')

    interp_text = (
        'How to read these metrics\n'
        '-------------------------\n'
        '  - Per-segment table: mean / median / p95 / max / variance of latent\n'
        '    flow magnitudes split into history (transitions fully inside the\n'
        '    input window) and forecast (transitions whose window touches the\n'
        '    model-generated future).\n'
        '  - Flow ratio: average forecast flow divided by average history flow. Values\n'
        "    near 1 mean the model's latent dynamics in the OOD segment match history.\n"
        '    Large values flag attributions in that region as likely noisy.\n'
        '  - Flow variance ratio: same comparison on variance.\n'
        '  - Curvature ratio: second-difference energy of Z; spikes when the\n'
        '    forecast segment becomes much jaggier than history.\n'
        '  - Latent diag-Mahalanobis ratio: per-dim distance of forecast latents\n'
        '    from the history mean (scaled by history variance). Values well above 1 indicate a\n'
        '    representation-level OOD shift.\n'
        '\n'
        'These scalars are also under explanation.diagnostics in explanation.json,\n'
        'and the full series is exported as semantic_flow.csv.'
    )
    fig.text(
        0.06,
        0.30,
        interp_text,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        color='#333333',
        linespacing=1.05,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _save_channel_axis_artifacts(
    out_dir: Path,
    channel_horizon_attributions: np.ndarray,
    channel_labels: list[str] | None = None,
    *,
    na_rep: str = 'nan',
) -> Path | None:
    """Write the [C, H] channel x horizon attribution CSV and a heatmap PNG.

    Returns the path to the heatmap PNG if matplotlib is available, else None.
    """
    A = np.asarray(channel_horizon_attributions, dtype=np.float32)
    if A.ndim != 2:
        return None
    C, H = A.shape
    labels = (
        channel_labels
        if (channel_labels and len(channel_labels) == C)
        else [f'channel_{c}' for c in range(C)]
    )

    df = pd.DataFrame(A, columns=[f'horizon_{h}' for h in range(H)])
    df.insert(0, 'channel', labels)
    df.to_csv(
        out_dir / 'channel_horizon_attributions.csv', index=False, na_rep=na_rep
    )

    try:
        import matplotlib as mpl

        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(max(8, H * 0.15), max(4, C * 0.4)))
    im = ax.imshow(A, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Forecast horizon')
    ax.set_ylabel('Input channel')
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, H - 1, min(12, H), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    plt.colorbar(im, ax=ax, label='Attribution')
    plt.tight_layout()
    heatmap_path = out_dir / 'channel_horizon_heatmap.png'
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    return heatmap_path


def _channel_axis_page(
    pdf: Any,
    plt: Any,
    *,
    explanation: ForecastExplanation,
    channel_labels: list[str] | None = None,
) -> None:
    """Append one PDF page for the feature-axis (channel x horizon) attribution.

    Primary visual: a [C, H] heatmap of how much each input channel contributes to
    each forecast step. A side bar summarizes each channel's total attribution
    (summed over horizons) as a single feature-importance score.
    """
    A = getattr(explanation, 'channel_horizon_attributions', None)
    if A is None:
        return
    A = np.asarray(A, dtype=np.float32)
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        return
    C, H = A.shape
    labels = (
        channel_labels
        if (channel_labels and len(channel_labels) == C)
        else [f'channel_{c}' for c in range(C)]
    )
    importance = A.sum(axis=1)
    method = getattr(explanation, 'channel_flow_method', None) or 'jacobian'
    resid = getattr(explanation, 'channel_flow_residual_ratio_mean', None)
    target_jacobian = method == 'target_input_jacobian'
    intro = (
        'Which input channel drives each target forecast step. Each cell is the magnitude of\n'
        "the target forecast's local directional response when that channel moves from its\n"
        'local-mean replacement to the observed input. Rows are channels and columns are\n'
        'forecast steps; brighter = a stronger target-specific effect.'
        if target_jacobian
        else "Which input channel (feature) drives each forecast step. The model's response is\n"
        'decomposed along the feature axis into per-channel contributions, then aggregated\n'
        'per horizon. Rows are input channels, columns are forecast steps; brighter = that\n'
        'channel matters more there.'
    )

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        'Feature-axis attribution (channel x horizon)',
        ha='center',
        fontsize=14,
        fontweight='bold',
    )
    fig.text(
        0.06,
        0.90,
        intro,
        ha='left',
        va='top',
        fontsize=9,
        color='gray',
    )

    ax = fig.add_axes((0.08, 0.40, 0.56, 0.42))
    im = ax.imshow(A, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Forecast horizon (0 = first predicted step)')
    ax.set_ylabel('Input channel')
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, H - 1, min(12, H), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels(xt)
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label='Directional effect' if target_jacobian else 'Attribution',
    )

    axb = fig.add_axes((0.72, 0.40, 0.22, 0.42))
    y = np.arange(C)
    axb.barh(y, importance, color='#1f77b4')
    axb.set_yticks(y)
    axb.set_yticklabels([])
    axb.invert_yaxis()  # channel 0 at top, aligned with the heatmap rows
    axb.set_title('Channel importance\n(sum over horizons)', fontsize=9)
    axb.set_xlabel(
        'Total effect' if target_jacobian else 'Total attribution', fontsize=8
    )
    axb.tick_params(axis='both', labelsize=7)

    top_c = int(np.argmax(importance))
    resid_line = (
        f'  - Estimator: {method}. Jacobian trust ratio (mean residual): {resid:.3f}\n'
        '    (lower = more reliable first-order split; high values flag strong\n'
        '    cross-channel interaction the first-order decomposition cannot capture).\n'
        if resid is not None
        else f'  - Estimator: {method}.\n'
    )
    heatmap_note = (
        "  - Heatmap cell (channel c, horizon h): magnitude of the target forecast's\n"
        '    first-order directional replacement effect; columns are not normalized.\n'
        if target_jacobian
        else '  - Heatmap cell (channel c, horizon h): share of the forecast at step h\n'
        '    attributable to input channel c (each horizon column sums to 1 over channels).\n'
    )
    bar_note = 'total effect' if target_jacobian else 'total attribution'
    note = (
        'How to read this page\n'
        '---------------------\n'
        + heatmap_note
        + f"  - Right bar: each channel's {bar_note} summed across horizons -- a single\n"
        '    feature-importance score per input channel.\n'
        f'  - Most influential channel overall: {labels[top_c]}.\n'
        + resid_line
        + '\n'
        'Full channel-by-horizon values are exported as channel_horizon_attributions.csv.'
    )
    fig.text(
        0.06,
        0.30,
        note,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        color='#333333',
        linespacing=1.05,
    )

    pdf.savefig(fig)
    plt.close(fig)


def _feature_axis_embedding_stability_frame(
    report: PerChannelEmbeddingStabilityReport,
    channel_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return the complete feature-axis embedding-stability report."""
    n_channels = int(report.n_channels)
    labels = (
        channel_labels
        if channel_labels is not None and len(channel_labels) == n_channels
        else [f'channel_{c}' for c in range(n_channels)]
    )
    return pd.DataFrame(
        {
            'feature': labels,
            'lip_ratio_mean': np.asarray(
                report.lip_ratio_mean, dtype=np.float32
            ),
            'lip_ratio_p50': np.asarray(report.lip_ratio_p50, dtype=np.float32),
            'lip_ratio_p95': np.asarray(report.lip_ratio_p95, dtype=np.float32),
            'lip_ratio_max': np.asarray(report.lip_ratio_max, dtype=np.float32),
            'n_trials_per_channel': np.asarray(
                report.n_trials_per_channel, dtype=np.int64
            ),
            'n_unique_windows': int(report.n_unique_windows),
            'step_delta_norm_mean': float(report.step_delta_norm_mean),
        }
    )


def _save_feature_axis_embedding_stability_csv(
    out_dir: Path,
    report: PerChannelEmbeddingStabilityReport,
    channel_labels: list[str] | None = None,
) -> Path:
    """Write the complete feature-axis embedding-stability report."""
    out_path = out_dir / 'feature_axis_embedding_stability.csv'
    _feature_axis_embedding_stability_frame(report, channel_labels).to_csv(
        out_path, index=False, na_rep='nan'
    )
    return out_path


def _feature_axis_embedding_stability_page(
    pdf: Any,
    plt: Any,
    *,
    report: PerChannelEmbeddingStabilityReport,
    channel_labels: list[str] | None = None,
) -> None:
    """Append the feature-axis embedding-stability summary."""
    frame = _feature_axis_embedding_stability_frame(report, channel_labels)
    shown = frame.sort_values(
        'lip_ratio_p95', ascending=False, na_position='last'
    ).head(10)

    def _fmt(value: Any) -> str:
        numeric = float(value)
        return f'{numeric:.4f}' if np.isfinite(numeric) else 'n/a'

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        'Feature-axis embedding stability',
        ha='center',
        fontsize=14,
        fontweight='bold',
    )
    fig.text(
        0.06,
        0.90,
        'This shows the delta between two embedding points after a small, controlled change to one feature.\n'
        r'Each sensitivity score is $\Vert \Delta \mathbf{z} \Vert_2'
        r'\,/\, \Vert \Delta \mathbf{x}_c \Vert_2$.'
        '\nLower values mean less latent movement per unit feature change; compare features\n'
        'within the same model and dataset because embedding scales are model-specific.',
        ha='left',
        va='top',
        fontsize=9,
        fontfamily='DejaVu Sans',
        math_fontfamily='dejavusans',
        color='gray',
    )

    table_rows = [
        [
            row.feature,
            _fmt(row.lip_ratio_mean),
            _fmt(row.lip_ratio_p50),
            _fmt(row.lip_ratio_p95),
            _fmt(row.lip_ratio_max),
            str(int(row.n_trials_per_channel)),
        ]
        for row in shown.itertuples(index=False)
    ]
    table_ax = fig.add_axes((0.06, 0.50, 0.88, 0.28))
    table_ax.axis('off')
    table = table_ax.table(
        cellText=table_rows,
        colLabels=['Feature', 'Mean', 'P50', 'P95', 'Max', 'Trials'],
        colWidths=[0.30, 0.14, 0.14, 0.14, 0.14, 0.14],
        loc='upper center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    for c in range(6):
        cell = table[(0, c)]
        cell.set_facecolor('#dddddd')
        cell.set_text_props(weight='bold')

    interp_text = (
        'How to read these metrics\n'
        '-------------------------\n'
        '  - Mean / P50 / P95 / Max: summaries of the sensitivity score across\n'
        '    small feature changes. Lower values mean the embedding\n'
        '    moves less per unit change in that feature.\n'
        '  - P95: conservative tail sensitivity. Rows are ranked by P95; the highest\n'
        '    values identify features whose small changes can move the embedding most.\n'
        '  - Trials: valid sensitivity measurements retained for that feature.\n'
        f'  - Sample coverage: {int(report.n_unique_windows)} unique windows. Mean\n'
        f'    baseline latent step: {_fmt(report.step_delta_norm_mean)} (reference only).\n'
        '  - Compare values only within the same model and dataset because embedding\n'
        '    scales are model-specific. A larger ratio is sensitivity, not by itself a defect.\n'
        '\n'
        'All features and raw report fields are exported to\n'
        'feature_axis_embedding_stability.csv and explanation.json.'
    )
    fig.text(
        0.06,
        0.43,
        interp_text,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        color='#333333',
    )

    pdf.savefig(fig)
    plt.close(fig)


@contextmanager
def _normalization_statistics_gradient(
    model: torch.nn.Module, *, enabled: bool
):
    """Temporarily include supported normalization statistics in autograd."""
    if not enabled:
        yield 0
        return

    restores: list[tuple[object, object, bool]] = []
    for module in model.modules():
        if not all(
            hasattr(module, name)
            for name in ('_get_statistics', '_normalize', 'eps')
        ):
            continue

        original = module._get_statistics
        had_instance_override = '_get_statistics' in vars(module)

        def _get_statistics(
            self: Any, x: torch.Tensor, mask: torch.Tensor | None = None
        ) -> None:
            if mask is None:
                mask = torch.ones((x.shape[0], x.shape[-1]), device=x.device)
            expanded = (
                mask.to(device=x.device)
                .unsqueeze(1)
                .expand(-1, x.shape[1], -1)
                .bool()
            )
            masked_x = torch.where(
                expanded, x, torch.full_like(x, float('nan'))
            )
            mean = torch.nanmean(masked_x, dim=-1, keepdim=True)
            variance = (masked_x - mean).square().nanmean(dim=-1, keepdim=True)
            raw_stdev = variance.sqrt()
            eps_sq = torch.as_tensor(
                float(self.eps) ** 2,
                dtype=variance.dtype,
                device=variance.device,
            )
            safe_stdev = variance.clamp_min(eps_sq).sqrt()
            self.mean = mean
            self.stdev = (
                raw_stdev.detach() + safe_stdev - safe_stdev.detach() + self.eps
            )

        module._get_statistics = types.MethodType(_get_statistics, module)
        restores.append((module, original, had_instance_override))

    try:
        yield len(restores)
    finally:
        for module, original, had_instance_override in reversed(restores):
            if had_instance_override:
                module._get_statistics = original
            else:
                delattr(module, '_get_statistics')


def _compute_integrated_gradients_report(
    *,
    model: torch.nn.Module,
    x_context_ct: np.ndarray,
    input_mask_l: np.ndarray,
    device: torch.device,
    channel_labels: list[str],
    baseline: Any,
    n_steps: int,
    n_baselines: int,
    internal_batch_size: int | None,
    reduce: str,
    seed: int,
    grad_through_norm: bool,
) -> dict[str, Any]:
    """Run embedding integrated gradients for the current SDK context window."""
    from sdm_client.explain._kumo_ts.integrated_gradients import (
        integrated_gradients_embedding,
    )

    x_tensor = torch.as_tensor(
        np.asarray(x_context_ct, dtype=np.float32), device=device
    )
    input_mask = torch.as_tensor(
        np.asarray(input_mask_l, dtype=np.int64), device=device
    )

    def embed_fn(x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f'SDK embedding input must be [batch, channels, context], got {tuple(x.shape)}'
            )
        mask = (
            input_mask.reshape(1, -1).expand(x.shape[0], -1).to(device=x.device)
        )
        with _normalization_statistics_gradient(
            model, enabled=grad_through_norm
        ):
            output = model.embed(x_enc=x, input_mask=mask)
        embeddings = (
            output.embeddings if hasattr(output, 'embeddings') else output
        )
        return embeddings.reshape(embeddings.shape[0], -1)

    attribution = integrated_gradients_embedding(
        embed_fn,
        x_tensor,
        baseline=baseline,
        n_steps=n_steps,
        n_baselines=n_baselines,
        internal_batch_size=internal_batch_size,
        reduce=reduce,
        seed=seed,
        device=device,
    )

    A = np.asarray(attribution.attribution, dtype=np.float32)
    if A.ndim != 2:
        raise ValueError(
            f'integrated gradients attribution must be [C, L], got shape {A.shape}'
        )

    C, _ = A.shape
    labels = (
        channel_labels
        if len(channel_labels) == C
        else [f'channel_{c}' for c in range(C)]
    )
    abs_A = np.abs(A)
    channel_abs_effect = abs_A.sum(axis=1).astype(np.float32)
    channel_signed_effect = A.sum(axis=1).astype(np.float32)
    total_abs = float(channel_abs_effect.sum())
    channel_abs_share = (
        (channel_abs_effect / total_abs).astype(np.float32)
        if total_abs > 1e-12
        else np.zeros_like(channel_abs_effect, dtype=np.float32)
    )
    time_abs_effect = abs_A.sum(axis=0).astype(np.float32)
    top_order = np.argsort(-channel_abs_effect)[: min(10, C)]

    return {
        'attribution': A,
        'channel_labels': labels,
        'channel_abs_effect': channel_abs_effect,
        'channel_signed_effect': channel_signed_effect,
        'channel_abs_share': channel_abs_share,
        'time_abs_effect': time_abs_effect,
        'top_channels': [
            {
                'channel': labels[int(i)],
                'abs_effect': float(channel_abs_effect[int(i)]),
                'signed_effect': float(channel_signed_effect[int(i)]),
                'abs_share': float(channel_abs_share[int(i)]),
            }
            for i in top_order
        ],
        'overall_effect': float(attribution.overall_effect),
        'abs_effect': float(attribution.abs_effect),
        'embedding_delta': float(attribution.embedding_delta),
        'embedding_delta_abs_mean': float(attribution.embedding_delta_abs_mean),
        'convergence_delta': float(attribution.convergence_delta),
        'n_steps': int(attribution.n_steps),
        'n_baselines': int(attribution.n_baselines),
        'baseline': str(attribution.baseline),
        'reduce': str(attribution.reduce),
        'embedding_dim': int(attribution.embedding_dim),
        'grad_through_norm': bool(grad_through_norm),
    }


def _save_integrated_gradients_artifacts(
    out_dir: Path,
    integrated_gradients_report: dict[str, Any],
    *,
    na_rep: str = 'nan',
) -> Path | None:
    """Write integrated-gradients CSVs and a signed heatmap PNG."""
    A = np.asarray(
        integrated_gradients_report.get('attribution'), dtype=np.float32
    )
    if A.ndim != 2:
        return None
    C, L = A.shape
    labels = integrated_gradients_report.get('channel_labels')
    labels = (
        labels
        if isinstance(labels, list) and len(labels) == C
        else [f'channel_{c}' for c in range(C)]
    )

    df_matrix = pd.DataFrame(A.T, columns=labels)
    df_matrix.insert(0, 'lag_from_last', np.arange(L - 1, -1, -1, dtype=int))
    df_matrix.insert(0, 'context_index', np.arange(L, dtype=int))
    df_matrix.to_csv(
        out_dir / 'integrated_gradients_attributions.csv',
        index=False,
        na_rep=na_rep,
    )

    summary = pd.DataFrame(
        {
            'channel': labels,
            'signed_effect': np.asarray(
                integrated_gradients_report['channel_signed_effect'],
                dtype=np.float32,
            ),
            'abs_effect': np.asarray(
                integrated_gradients_report['channel_abs_effect'],
                dtype=np.float32,
            ),
            'abs_share': np.asarray(
                integrated_gradients_report['channel_abs_share'],
                dtype=np.float32,
            ),
        }
    )
    summary.to_csv(
        out_dir / 'integrated_gradients_channel_summary.csv',
        index=False,
        na_rep=na_rep,
    )

    try:
        import matplotlib as mpl

        mpl.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    scale = (
        float(np.nanpercentile(np.abs(A), 99)) if np.isfinite(A).any() else 0.0
    )
    if scale <= 1e-12:
        scale = float(np.nanmax(np.abs(A))) if np.isfinite(A).any() else 1.0
    if scale <= 1e-12:
        scale = 1.0

    fig, ax = plt.subplots(figsize=(max(8, L * 0.02), max(4, C * 0.35)))
    im = ax.imshow(
        A,
        aspect='auto',
        cmap='coolwarm',
        interpolation='nearest',
        vmin=-scale,
        vmax=scale,
    )
    ax.set_xlabel('Input context lag (0 = most recent)')
    ax.set_ylabel('Input channel')
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, L - 1, min(12, L), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels((L - 1 - xt).astype(int))
    plt.colorbar(im, ax=ax, label='Signed IG attribution')
    plt.tight_layout()
    heatmap_path = out_dir / 'integrated_gradients_heatmap.png'
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    return heatmap_path


def _integrated_gradients_page(
    pdf: Any,
    plt: Any,
    *,
    integrated_gradients_report: dict[str, Any] | None,
) -> None:
    """Append one PDF page for embedding integrated gradients."""
    if integrated_gradients_report is None:
        return
    A = np.asarray(
        integrated_gradients_report.get('attribution'), dtype=np.float32
    )
    if A.ndim != 2 or A.shape[0] < 1 or A.shape[1] < 1:
        return
    C, L = A.shape
    labels = integrated_gradients_report.get('channel_labels')
    labels = (
        labels
        if isinstance(labels, list) and len(labels) == C
        else [f'channel_{c}' for c in range(C)]
    )
    abs_effect = np.asarray(
        integrated_gradients_report.get('channel_abs_effect'), dtype=np.float32
    )
    if abs_effect.shape != (C,):
        abs_effect = np.abs(A).sum(axis=1)

    scale = (
        float(np.nanpercentile(np.abs(A), 99)) if np.isfinite(A).any() else 0.0
    )
    if scale <= 1e-12:
        scale = float(np.nanmax(np.abs(A))) if np.isfinite(A).any() else 1.0
    if scale <= 1e-12:
        scale = 1.0

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.95,
        'Integrated gradients (embedding sensitivity)',
        ha='center',
        fontsize=14,
        fontweight='bold',
    )
    fig.text(
        0.06,
        0.90,
        'Signed input attributions for the model embedding along the baseline-to-input path.\n'
        'Rows are input channels and columns are the context timesteps; lag 0 is the most\n'
        'recent input. Positive and negative colors show direction, while magnitude shows strength.',
        ha='left',
        va='top',
        fontsize=9,
        color='gray',
    )

    ax = fig.add_axes((0.08, 0.38, 0.58, 0.43))
    im = ax.imshow(
        A,
        aspect='auto',
        cmap='coolwarm',
        interpolation='nearest',
        vmin=-scale,
        vmax=scale,
    )
    ax.set_xlabel('Input context lag (0 = most recent)')
    ax.set_ylabel('Input channel')
    ax.set_yticks(range(C))
    ax.set_yticklabels(labels, fontsize=8)
    xt = np.linspace(0, L - 1, min(12, L), dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels((L - 1 - xt).astype(int))
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label='Signed attribution'
    )

    axb = fig.add_axes((0.74, 0.38, 0.22, 0.43))
    y = np.arange(C)
    axb.barh(y, abs_effect, color='#2a9d8f')
    axb.set_yticks(y)
    axb.set_yticklabels([])
    axb.invert_yaxis()
    axb.set_title(
        r'Channel effect' '\n' r'($\sum |\mathrm{IG}|$)',
        fontsize=9,
        math_fontfamily='dejavusans',
    )
    axb.set_xlabel('Absolute effect', fontsize=8)
    axb.tick_params(axis='both', labelsize=7)

    top_c = int(np.argmax(abs_effect))
    baseline = integrated_gradients_report.get('baseline')
    reduce = integrated_gradients_report.get('reduce')
    steps = integrated_gradients_report.get('n_steps')
    n_baselines = integrated_gradients_report.get('n_baselines')
    convergence = _scalar_to_jsonable(
        integrated_gradients_report.get('convergence_delta')
    )
    emb_delta = _scalar_to_jsonable(
        integrated_gradients_report.get('embedding_delta')
    )
    grad_norm = bool(integrated_gradients_report.get('grad_through_norm'))
    note = (
        'How to read this page\n'
        '---------------------\n'
        '  - Heatmap cell (channel c, lag l): signed contribution of that input value\n'
        '    to the scalar embedding objective used by integrated gradients.\n'
        '  - Right bar: per-channel absolute effect, summed over all context timesteps.\n'
        f'  - Most influential channel by absolute IG: {labels[top_c]}.\n'
        f'  - Config: baseline={baseline}, steps={steps}, baselines={n_baselines}, reduce={reduce}.\n'
        f'  - RevIN/statistics gradient path enabled: {grad_norm}.\n'
        f'  - Embedding delta: {emb_delta}; convergence delta: {convergence}.\n'
        '\n'
        'Full values are exported as integrated_gradients_attributions.csv and\n'
        'integrated_gradients_channel_summary.csv.'
    )
    fig.text(
        0.06,
        0.30,
        note,
        ha='left',
        va='top',
        fontsize=9,
        family='monospace',
        color='#333333',
    )

    pdf.savefig(fig)
    plt.close(fig)


def _build_pdf_report(
    pdf_path: Path,
    *,
    dataset_name: str | None = None,
    forecast_df: pd.DataFrame,
    explanation: ForecastExplanation,
    target_column: str,
    timestamp_column: str,
    heatmap_path: Path | None,
    topk_rows: list[list[str]],
    top_k: int,
    trajectory_report: TrajectoryStabilityReport | None = None,
    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport
    | None = None,
    context_len: int | None = None,
    forecast_horizon: int | None = None,
    channel_labels: list[str] | None = None,
    integrated_gradients_report: dict[str, Any] | None = None,
) -> Path | None:
    """Compose a multi-page PDF with the forecast and attribution heatmap."""
    try:
        import matplotlib as mpl

        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return None

    attrib = np.asarray(explanation.lag_horizon_attributions)
    K, H = attrib.shape
    flow_magnitudes = getattr(explanation, 'flow_magnitudes', None)
    show_heatmap = heatmap_path is not None and heatmap_path.exists()
    show_topk = bool(topk_rows)
    show_semantic_flow = (
        context_len is not None
        and forecast_horizon is not None
        and flow_magnitudes is not None
        and np.asarray(flow_magnitudes).size > 0
    )
    show_trajectory_stability = trajectory_report is not None
    show_feature_axis = (
        getattr(explanation, 'channel_horizon_attributions', None) is not None
    )
    show_feature_axis_stability = (
        feature_axis_embedding_stability_report is not None
    )
    show_integrated_gradients = integrated_gradients_report is not None

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(
            0.5,
            0.94,
            'Kumo-Forecast Interpretability Report',
            ha='center',
            fontsize=18,
            fontweight='bold',
        )
        fig.text(
            0.5,
            0.905,
            datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
            ha='center',
            fontsize=10,
            color='gray',
        )

        sections = [
            section
            for available, section in [
                (
                    True,
                    [
                        'Forecast preview: line chart of predicted target values.'
                    ],
                ),
                (
                    show_heatmap,
                    [
                        'Lag x Horizon attribution heatmap: how much each past step',
                        '   contributes to each forecast step.',
                        '   a) lag=0 is the most recent input.',
                        '   b) Softmax-normalized weights; brighter cells = time steps',
                        '      that contribute more to the forecast.',
                    ],
                ),
                (
                    show_topk,
                    ['Top-k lag steps per horizon (marginal contributions).'],
                ),
                (
                    show_semantic_flow,
                    [
                        'Semantic flow magnitudes: per-transition latent flow and',
                        '   forecast-vs-history diagnostics.',
                    ],
                ),
                (
                    show_trajectory_stability,
                    [
                        'Latent trajectory stability: temporal-smoothness metrics',
                        '   over the context window.',
                    ],
                ),
                (
                    show_feature_axis,
                    [
                        'Feature-axis attribution: which input channel drives each',
                        '   forecast step (multivariate inputs only).',
                    ],
                ),
                (
                    show_feature_axis_stability,
                    [
                        'Feature-axis embedding stability: per-channel sensitivity',
                        '   to small, controlled input changes (multivariate inputs only).',
                    ],
                ),
                (
                    show_integrated_gradients,
                    [
                        'Integrated gradients: signed input-channel/time',
                        '   sensitivity for the model embedding.',
                    ],
                ),
            ]
            if available
        ]

        summary = [
            'Overview',
            '--------',
            f'Dataset: {dataset_name or ""}',
            f'Target column: {target_column}',
            f'Timestamp column: {timestamp_column}',
            f'Forecast steps (H): {H}',
            f'Lag context size (K): {K}',
            '',
            'What is in this report',
            '----------------------',
        ]
        for number, section in enumerate(sections, start=1):
            summary.extend([f'{number}. {section[0]}', *section[1:]])
        fig.text(
            0.08,
            0.85,
            '\n'.join(summary),
            ha='left',
            va='top',
            fontsize=10,
            family='monospace',
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 6))
        if (
            timestamp_column in forecast_df.columns
            and f'{target_column}_forecast' in forecast_df.columns
        ):
            ax.plot(
                forecast_df[timestamp_column],
                forecast_df[f'{target_column}_forecast'],
                marker='o',
                linewidth=1.5,
            )
            ax.set_xlabel(timestamp_column)
            ax.set_ylabel(f'{target_column} (forecast, original scale)')
            ax.set_title(f'Forecast preview: {target_column}')
            fig.autofmt_xdate()
        else:
            ax.text(
                0.5, 0.5, 'Forecast columns not found', ha='center', va='center'
            )
            ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if show_heatmap:
            try:
                from matplotlib.image import imread
            except ImportError:
                imread = None
            fig = plt.figure(figsize=(11, 8.5))
            if imread is not None:
                ax = fig.add_axes((0.05, 0.18, 0.9, 0.72))
                ax.imshow(imread(str(heatmap_path)))
                ax.axis('off')
            else:
                ax = fig.add_axes((0.05, 0.18, 0.9, 0.72))
                ax.text(
                    0.5,
                    0.5,
                    'Heatmap PNG could not be embedded.',
                    ha='center',
                    va='center',
                )
                ax.axis('off')
            fig.text(
                0.5,
                0.94,
                'Lag x Horizon attribution heatmap',
                ha='center',
                fontsize=14,
                fontweight='bold',
            )
            fig.text(
                0.5,
                0.10,
                'Vertical axis: lag back from the last observation (0 = most recent).\n'
                'Horizontal axis: forecast step (0 = first predicted step).\n'
                'Brighter = that past step has more influence on that forecast step.',
                ha='center',
                va='center',
                fontsize=9,
                color='gray',
            )
            pdf.savefig(fig)
            plt.close(fig)

        if show_topk:
            header = ['Horizon']
            for r in range(1, top_k + 1):
                header.extend([f'Rank-{r}\nLag', f'Rank-{r}\nProb'])

            rows_per_page = 30
            chunks = [
                topk_rows[i : i + rows_per_page]
                for i in range(0, len(topk_rows), rows_per_page)
            ]
            num_cols = 1 + 2 * top_k
            horizon_col = 0.12
            data_col = max(0.05, (1.0 - horizon_col) / max(1, num_cols - 1))
            col_widths = [horizon_col] + [data_col] * (num_cols - 1)

            for idx, chunk in enumerate(chunks, start=1):
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(
                    0.5,
                    0.95,
                    f'Top-{top_k} lag steps per horizon (page {idx}/{len(chunks)})',
                    ha='center',
                    fontsize=14,
                    fontweight='bold',
                )
                fig.text(
                    0.06,
                    0.91,
                    'Horizon: which forecast step (0 = first predicted step).\n'
                    'Rank-r Lag: how many steps back from the last observation '
                    'for the r-th most influential past step\n'
                    '    (1 = most recent past step). Ranked from highest to '
                    'lowest contribution.\n'
                    'Rank-r Prob: marginal softmax weight of that ranked lag step '
                    'on that horizon\n'
                    '    -- higher means it contributed more to the forecast.',
                    ha='left',
                    va='top',
                    fontsize=9,
                    color='gray',
                )

                ax = fig.add_axes((0.04, 0.04, 0.92, 0.78))
                ax.axis('off')
                table = ax.table(
                    cellText=chunk,
                    colLabels=header,
                    colWidths=col_widths,
                    loc='upper center',
                    cellLoc='center',
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.0, 1.25)
                for c in range(len(header)):
                    cell = table[(0, c)]
                    cell.set_facecolor('#dddddd')
                    cell.set_text_props(weight='bold')
                    cell.set_height(cell.get_height() * 1.6)

                pdf.savefig(fig)
                plt.close(fig)

        if show_semantic_flow:
            try:
                _semantic_flow_page(
                    pdf,
                    plt,
                    explanation=explanation,
                    context_len=int(context_len),
                    forecast_horizon=int(forecast_horizon),
                )
            except Exception as e:
                logger.warning('Semantic flow page skipped: %s', e)

        if show_trajectory_stability:
            r = trajectory_report
            rows = [
                [
                    'Zero-crossing rate (mean / p95)',
                    f'{r.zero_crossing_rate_mean:.4f}',
                    f'{r.zero_crossing_rate_p95:.4f}',
                ],
                [
                    'Direction-flip rate (mean / p95)',
                    f'{r.direction_flip_rate_mean:.4f}',
                    f'{r.direction_flip_rate_p95:.4f}',
                ],
                [
                    'Relative jitter (mean / p95)',
                    f'{r.relative_jitter_mean:.4f}',
                    f'{r.relative_jitter_p95:.4f}',
                ],
                [
                    'Occupancy positive / negative',
                    f'{r.occupancy_positive_mean:.4f}',
                    f'{r.occupancy_negative_mean:.4f}',
                ],
                [
                    'Latent shape (T / D)',
                    f'{int(r.n_time_steps)}',
                    f'{int(r.n_dimensions)}',
                ],
            ]

            fig = plt.figure(figsize=(11, 8.5))
            fig.text(
                0.5,
                0.95,
                'Latent trajectory stability',
                ha='center',
                fontsize=14,
                fontweight='bold',
            )
            fig.text(
                0.06,
                0.90,
                'Per-dimension temporal-smoothness metrics for the latent trajectory.\n'
                'Lower zero-crossing, direction-flip, and relative-jitter values indicate\n'
                "a smoother embedding (supports the framework's stability assumption).\n"
                r'Relative jitter compares $\mathrm{mean}(|\Delta z|)$ with '
                r'$\mathrm{mean}(|z-z_{\mathrm{center}}|)$.',
                ha='left',
                va='top',
                fontsize=9,
                fontfamily='DejaVu Sans',
                math_fontfamily='dejavusans',
                color='gray',
            )

            ax = fig.add_axes((0.08, 0.45, 0.84, 0.35))
            ax.axis('off')
            table = ax.table(
                cellText=rows,
                colLabels=['Metric', 'Value 1', 'Value 2'],
                colWidths=[0.5, 0.25, 0.25],
                loc='upper center',
                cellLoc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.4)
            for c in range(3):
                cell = table[(0, c)]
                cell.set_facecolor('#dddddd')
                cell.set_text_props(weight='bold')

            interp_text = (
                'How to read these metrics\n'
                '-------------------------\n'
                '  - Zero-crossing rate: fraction of consecutive steps where the centered\n'
                '    latent value flips sign. Lower => trajectory stays on one side of the\n'
                '    reference for longer (less oscillation).\n'
                '  - Direction-flip rate: fraction of steps where the latent step direction\n'
                '    reverses sign. Lower => monotone, smoother dynamics.\n'
                '  - Relative jitter: average latent step size divided by average displacement\n'
                '    from the reference. Lower => small smooth steps\n'
                '    relative to overall amplitude.\n'
                '  - Occupancy positive / negative: fraction of time the centered trajectory\n'
                '    sits above / below the deadband. Asymmetry can flag regime drift.\n'
                '  - Latent shape (T / D): number of latent time steps and embedding dim;\n'
                '    sanity-check that T matches the context length used for this run.\n'
                '\n'
                'Pair these with the diagnostics block in explanation.json\n'
                '(flow_ratio_forecast_vs_history, curvature_ratio_forecast_vs_history,\n'
                'latent_diag_mahalanobis_ratio_forecast_vs_history) for the forecast-vs-\n'
                'history comparison.'
            )
            fig.text(
                0.06,
                0.40,
                interp_text,
                ha='left',
                va='top',
                fontsize=9,
                family='monospace',
                color='#333333',
            )
            pdf.savefig(fig)
            plt.close(fig)

        # Feature-axis page last (and only for multivariate inputs), matching
        # its position in the cover summary.
        if show_feature_axis:
            try:
                _channel_axis_page(
                    pdf,
                    plt,
                    explanation=explanation,
                    channel_labels=channel_labels,
                )
            except Exception as e:
                logger.warning('Feature-axis page skipped: %s', e)

        if show_feature_axis_stability:
            try:
                _feature_axis_embedding_stability_page(
                    pdf,
                    plt,
                    report=feature_axis_embedding_stability_report,
                    channel_labels=channel_labels,
                )
            except Exception as e:
                logger.warning(
                    'Feature-axis embedding-stability page skipped: %s', e
                )

        if show_integrated_gradients:
            try:
                _integrated_gradients_page(
                    pdf,
                    plt,
                    integrated_gradients_report=integrated_gradients_report,
                )
            except Exception as e:
                logger.warning('Integrated-gradients page skipped: %s', e)

    return pdf_path


def _array_to_jsonable(arr: Any) -> Any:
    """Convert a numpy array (or torch tensor / scalar / None) into JSON-friendly data."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return (
            np.where(np.isfinite(arr), arr, None).tolist()
            if arr.dtype.kind == 'f'
            else arr.tolist()
        )
    try:
        return _array_to_jsonable(np.asarray(arr))
    except Exception:
        return None


def _scalar_to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _trajectory_report_to_dict(
    report: TrajectoryStabilityReport | None,
) -> dict[str, Any] | None:
    """Convert a TrajectoryStabilityReport into a JSON-friendly dict (or None)."""
    if report is None:
        return None
    return {
        'zero_crossing_rate_mean': _scalar_to_jsonable(
            report.zero_crossing_rate_mean
        ),
        'zero_crossing_rate_p95': _scalar_to_jsonable(
            report.zero_crossing_rate_p95
        ),
        'direction_flip_rate_mean': _scalar_to_jsonable(
            report.direction_flip_rate_mean
        ),
        'direction_flip_rate_p95': _scalar_to_jsonable(
            report.direction_flip_rate_p95
        ),
        'relative_jitter_mean': _scalar_to_jsonable(
            report.relative_jitter_mean
        ),
        'relative_jitter_p95': _scalar_to_jsonable(report.relative_jitter_p95),
        'occupancy_positive_mean': _scalar_to_jsonable(
            report.occupancy_positive_mean
        ),
        'occupancy_negative_mean': _scalar_to_jsonable(
            report.occupancy_negative_mean
        ),
        'n_time_steps': int(report.n_time_steps),
        'n_dimensions': int(report.n_dimensions),
    }


def _feature_axis_embedding_stability_to_dict(
    report: PerChannelEmbeddingStabilityReport | None,
    *,
    channel_labels: list[str] | None = None,
) -> dict[str, Any] | None:
    """Convert a feature-axis embedding-stability report to JSON."""
    if report is None:
        return None
    frame = _feature_axis_embedding_stability_frame(report, channel_labels)
    return {
        'channel_labels': frame['feature'].tolist(),
        'lip_ratio_mean': _array_to_jsonable(
            frame['lip_ratio_mean'].to_numpy()
        ),
        'lip_ratio_p50': _array_to_jsonable(frame['lip_ratio_p50'].to_numpy()),
        'lip_ratio_p95': _array_to_jsonable(frame['lip_ratio_p95'].to_numpy()),
        'lip_ratio_max': _array_to_jsonable(frame['lip_ratio_max'].to_numpy()),
        'n_trials_per_channel': _array_to_jsonable(
            frame['n_trials_per_channel'].to_numpy()
        ),
        'n_unique_windows': int(report.n_unique_windows),
        'step_delta_norm_mean': _scalar_to_jsonable(
            report.step_delta_norm_mean
        ),
    }


def _integrated_gradients_to_dict(
    integrated_gradients_report: dict[str, Any] | None,
    *,
    include_full_arrays: bool = True,
) -> dict[str, Any] | None:
    """Convert an integrated-gradients report into a JSON-friendly dict."""
    if integrated_gradients_report is None:
        return None

    payload: dict[str, Any] = {
        'baseline': integrated_gradients_report.get('baseline'),
        'reduce': integrated_gradients_report.get('reduce'),
        'n_steps': int(integrated_gradients_report.get('n_steps', 0)),
        'n_baselines': int(integrated_gradients_report.get('n_baselines', 0)),
        'embedding_dim': int(
            integrated_gradients_report.get('embedding_dim', 0)
        ),
        'grad_through_norm': bool(
            integrated_gradients_report.get('grad_through_norm', False)
        ),
        'overall_effect': _scalar_to_jsonable(
            integrated_gradients_report.get('overall_effect')
        ),
        'abs_effect': _scalar_to_jsonable(
            integrated_gradients_report.get('abs_effect')
        ),
        'embedding_delta': _scalar_to_jsonable(
            integrated_gradients_report.get('embedding_delta')
        ),
        'embedding_delta_abs_mean': _scalar_to_jsonable(
            integrated_gradients_report.get('embedding_delta_abs_mean')
        ),
        'convergence_delta': _scalar_to_jsonable(
            integrated_gradients_report.get('convergence_delta')
        ),
        'channel_labels': list(
            integrated_gradients_report.get('channel_labels', [])
        ),
        'channel_signed_effect': _array_to_jsonable(
            integrated_gradients_report.get('channel_signed_effect')
        ),
        'channel_abs_effect': _array_to_jsonable(
            integrated_gradients_report.get('channel_abs_effect')
        ),
        'channel_abs_share': _array_to_jsonable(
            integrated_gradients_report.get('channel_abs_share')
        ),
        'top_channels': integrated_gradients_report.get('top_channels', []),
    }
    if include_full_arrays:
        payload['attribution'] = _array_to_jsonable(
            integrated_gradients_report.get('attribution')
        )
        payload['time_abs_effect'] = _array_to_jsonable(
            integrated_gradients_report.get('time_abs_effect')
        )
    return payload


def _explanation_to_dict(
    forecast_df: pd.DataFrame,
    explanation: ForecastExplanation,
    *,
    target_column: str,
    timestamp_column: str = 'timestamp',
    dataset_name: str | None = None,
    include_full_arrays: bool = True,
    trajectory_report: TrajectoryStabilityReport | None = None,
    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport
    | None = None,
    channel_labels: list[str] | None = None,
    integrated_gradients_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render the (forecast, explanation) pair as a JSON-serializable dict."""
    fc = forecast_df.copy()
    if timestamp_column in fc.columns and pd.api.types.is_datetime64_any_dtype(
        fc[timestamp_column]
    ):
        fc[timestamp_column] = fc[timestamp_column].dt.strftime(
            '%Y-%m-%dT%H:%M:%S'
        )
    forecast_records = fc.to_dict(orient='records')

    base = np.asarray(explanation.baseline_forecast)
    scores = np.asarray(explanation.lag_horizon_scores)
    attrib = np.asarray(explanation.lag_horizon_attributions)
    C, H = base.shape if base.ndim == 2 else (None, None)
    K = attrib.shape[0] if attrib.ndim == 2 else None

    surrogate_block: dict[str, Any] | None
    if explanation.surrogate_coef is not None:
        surrogate_block = {
            'coef': _array_to_jsonable(explanation.surrogate_coef),
            'intercept': _array_to_jsonable(explanation.surrogate_intercept),
            'feature_layout': explanation.surrogate_feature_layout,
        }
    else:
        surrogate_block = None

    explanation_block: dict[str, Any] = {
        'shapes': {
            'C_channels': int(C) if C is not None else None,
            'H_horizon': int(H) if H is not None else None,
            'K_lags': int(K) if K is not None else None,
        },
        'baseline_forecast': _array_to_jsonable(base),
        'lag_horizon_scores': _array_to_jsonable(scores),
        'lag_horizon_attributions': _array_to_jsonable(attrib),
        'surrogate': surrogate_block,
        'diagnostics': {
            'flow_ratio_forecast_vs_history': _scalar_to_jsonable(
                explanation.flow_ratio_forecast_vs_history
            ),
            'flow_variance_ratio_forecast_vs_history': _scalar_to_jsonable(
                explanation.flow_variance_ratio_forecast_vs_history
            ),
            'curvature_ratio_forecast_vs_history': _scalar_to_jsonable(
                explanation.curvature_ratio_forecast_vs_history
            ),
            'latent_diag_mahalanobis_ratio_forecast_vs_history': _scalar_to_jsonable(
                explanation.latent_diag_mahalanobis_ratio_forecast_vs_history
            ),
            'latent_trajectory_shape': (
                list(np.asarray(explanation.latent_trajectory).shape)
                if getattr(explanation, 'latent_trajectory', None) is not None
                else None
            ),
        },
        'trajectory_stability': _trajectory_report_to_dict(trajectory_report),
    }

    # Feature-axis (channel) attribution -- only present for multivariate inputs.
    chan_attrib = getattr(explanation, 'channel_horizon_attributions', None)
    stability_block = _feature_axis_embedding_stability_to_dict(
        feature_axis_embedding_stability_report,
        channel_labels=channel_labels,
    )
    if chan_attrib is not None or stability_block is not None:
        feature_axis: dict[str, Any] = {}
        if chan_attrib is not None:
            feature_axis.update(
                {
                    'method': getattr(explanation, 'channel_flow_method', None),
                    'channel_horizon_attributions': _array_to_jsonable(
                        np.asarray(chan_attrib)
                    ),
                    'residual_ratio_mean': _scalar_to_jsonable(
                        explanation.channel_flow_residual_ratio_mean
                    ),
                    'residual_ratio_p95': _scalar_to_jsonable(
                        explanation.channel_flow_residual_ratio_p95
                    ),
                }
            )
        if (
            include_full_arrays
            and getattr(explanation, 'per_channel_flow', None) is not None
        ):
            feature_axis['per_channel_flow'] = _array_to_jsonable(
                np.asarray(explanation.per_channel_flow)
            )
        if stability_block is not None:
            feature_axis['embedding_stability'] = stability_block
        explanation_block['feature_axis'] = feature_axis

    if include_full_arrays:
        explanation_block['flow_magnitudes'] = _array_to_jsonable(
            explanation.flow_magnitudes
        )
        explanation_block['latent_trajectory'] = _array_to_jsonable(
            explanation.latent_trajectory
        )

    ig_block = _integrated_gradients_to_dict(
        integrated_gradients_report,
        include_full_arrays=include_full_arrays,
    )
    if ig_block is not None:
        explanation_block['integrated_gradients'] = ig_block

    return {
        'metadata': {
            'dataset_name': dataset_name,
            'target_column': target_column,
            'timestamp_column': timestamp_column,
            'generated_at': datetime.now(tz=timezone.utc).strftime(
                '%Y-%m-%dT%H:%M:%SZ'
            ),
            'include_full_arrays': bool(include_full_arrays),
        },
        'forecast': forecast_records,
        'explanation': explanation_block,
    }


def _save_explanation_json(
    *,
    forecast_df: pd.DataFrame,
    explanation: ForecastExplanation,
    path: str | Path,
    target_column: str,
    timestamp_column: str = 'timestamp',
    dataset_name: str | None = None,
    include_full_arrays: bool = True,
    indent: int | None = 2,
    trajectory_report: TrajectoryStabilityReport | None = None,
    feature_axis_embedding_stability_report: PerChannelEmbeddingStabilityReport
    | None = None,
    channel_labels: list[str] | None = None,
    integrated_gradients_report: dict[str, Any] | None = None,
) -> Path:
    """Persist the (forecast, explanation) pair as a JSON file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _explanation_to_dict(
        forecast_df,
        explanation,
        target_column=target_column,
        timestamp_column=timestamp_column,
        dataset_name=dataset_name,
        include_full_arrays=include_full_arrays,
        trajectory_report=trajectory_report,
        feature_axis_embedding_stability_report=feature_axis_embedding_stability_report,
        channel_labels=channel_labels,
        integrated_gradients_report=integrated_gradients_report,
    )
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=indent, allow_nan=False)
    return out_path
