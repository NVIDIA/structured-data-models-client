# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained Mermaid visualization pipeline."""

import base64
import html
import io
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph, LocalTable, viz
from sdm_relational_engine.rfm import graph as graph_module


@pytest.fixture
def sample_graph() -> Graph:
    users = pd.DataFrame(
        {
            'user_id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
        }
    )
    orders = pd.DataFrame(
        {
            'order_id': [10, 11],
            'user_id': [1, 2],
            'amount': [9.5, 3.2],
        }
    )
    graph = Graph(
        tables=[
            LocalTable(users, name='users', primary_key='user_id'),
            LocalTable(orders, name='orders', primary_key='order_id'),
        ]
    )
    graph.link('orders', 'user_id', 'users')
    return graph


@pytest.fixture
def no_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, 'in_streamlit_notebook', lambda: False)
    monkeypatch.setattr(graph_module, 'in_jupyter_notebook', lambda: False)


def test_bundled_mermaid_js() -> None:
    js = viz._mermaid_js()
    assert len(js) > 1_000_000
    assert 'globalThis["mermaid"]' in js


def test_bundled_mermaid_is_not_vulnerable() -> None:
    """Mermaid below 11.15.0 injects HTML from `classDef` (GHSA-ghcm-xqfw-q4vr).

    The bundle is vendored, so nothing else would notice it being rolled back
    to a version carrying the advisory.
    """
    found = re.findall(r'version:"(\d+)\.(\d+)\.(\d+)"', viz._mermaid_js())
    assert found, 'no version string in the bundle'
    assert max(tuple(int(p) for p in v) for v in found) >= (11, 15, 0)


def test_to_html_is_self_contained(sample_graph: Graph) -> None:
    source = sample_graph._to_mermaid()
    doc = viz.to_html(source)

    assert doc.startswith('<!DOCTYPE html>')
    assert html.escape(source) in doc
    assert 'mermaid.initialize' in doc
    assert 'globalThis["mermaid"]' in doc

    # No external fetches: every script is inlined.
    assert '<script src=' not in doc
    assert 'cdn.' not in doc
    assert 'mermaid.ink' not in doc


def test_to_iframe_round_trip(sample_graph: Graph) -> None:
    source = sample_graph._to_mermaid()
    iframe = viz.to_iframe(source, height=320)

    assert iframe.startswith('<iframe srcdoc="')
    assert 'height="320"' in iframe
    assert 'sandbox="allow-scripts"' in iframe

    srcdoc = re.search(r'srcdoc="(.*)" width', iframe, re.DOTALL).group(1)
    assert html.unescape(srcdoc) == viz.to_html(source)


def test_render_image_url_encoding(requests_mock) -> None:
    requests_mock.get(
        re.compile(r'https://mermaid\.ink/.*'), content=b'png-bytes'
    )
    out = viz.render_image('erDiagram', 'png')
    assert out == b'png-bytes'

    url = requests_mock.request_history[0].url
    assert url.startswith('https://mermaid.ink/img/base64:')
    encoded = url.removeprefix('https://mermaid.ink/img/base64:')
    encoded = encoded.split('?')[0]
    state = json.loads(base64.urlsafe_b64decode(encoded))
    assert state['code'] == 'erDiagram'


def test_render_image_svg_route(requests_mock) -> None:
    requests_mock.get(
        re.compile(r'https://mermaid\.ink/svg/.*'), content=b'<svg/>'
    )
    assert viz.render_image('erDiagram', 'svg') == b'<svg/>'


def test_render_image_unreachable(requests_mock) -> None:
    requests_mock.get(re.compile(r'https://mermaid\.ink/.*'), status_code=503)
    with pytest.raises(RuntimeError, match='offline rendering'):
        viz.render_image('erDiagram', 'png')


def test_render_image_bad_format() -> None:
    with pytest.raises(ValueError, match='png'):
        viz.render_image('erDiagram', 'pdf')


def test_visualize_html_file(sample_graph: Graph, tmp_path: Path) -> None:
    path = tmp_path / 'graph.html'
    sample_graph.visualize(path=path)

    doc = path.read_text(encoding='utf-8')
    assert 'erDiagram' in doc
    assert 'mermaid.initialize' in doc
    assert 'orders' in doc


def test_visualize_mmd_file(sample_graph: Graph, tmp_path: Path) -> None:
    path = tmp_path / 'graph.mmd'
    sample_graph.visualize(path=path)

    source = path.read_text(encoding='utf-8')
    assert source.startswith('erDiagram')
    assert 'users o|--o{ orders : user_id' in source


def test_visualize_png_file(
    sample_graph: Graph, tmp_path: Path, requests_mock
) -> None:
    requests_mock.get(
        re.compile(r'https://mermaid\.ink/.*'), content=b'png-bytes'
    )
    path = tmp_path / 'graph.png'
    sample_graph.visualize(path=path)
    assert path.read_bytes() == b'png-bytes'


def test_visualize_bytes_io(sample_graph: Graph, requests_mock) -> None:
    requests_mock.get(
        re.compile(r'https://mermaid\.ink/.*'), content=b'png-bytes'
    )
    buffer = io.BytesIO()
    sample_graph.visualize(path=buffer)
    assert buffer.getvalue() == b'png-bytes'


def test_visualize_rejects_unknown_suffix(
    sample_graph: Graph, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match='not supported'):
        sample_graph.visualize(path=tmp_path / 'graph.pdf')


def test_visualize_rejects_missing_suffix(
    sample_graph: Graph, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match='Missing file extension'):
        sample_graph.visualize(path=tmp_path / 'graph')


def test_visualize_jupyter_display(
    sample_graph: Graph, no_notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, 'in_jupyter_notebook', lambda: True)

    displayed: list = []
    fake_display = types.ModuleType('IPython.display')
    fake_display.HTML = lambda data: ('HTML', data)
    fake_display.display = displayed.append
    fake_ipython = types.ModuleType('IPython')
    fake_ipython.display = fake_display
    monkeypatch.setitem(sys.modules, 'IPython', fake_ipython)
    monkeypatch.setitem(sys.modules, 'IPython.display', fake_display)

    sample_graph.visualize(height=222)

    assert len(displayed) == 1
    kind, payload = displayed[0]
    assert kind == 'HTML'
    assert payload.startswith('<iframe srcdoc="')
    assert 'height="222"' in payload


def test_visualize_streamlit_display(
    sample_graph: Graph, no_notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, 'in_streamlit_notebook', lambda: True)

    calls: list = []
    fake_st = types.ModuleType('streamlit')
    fake_st.components = types.SimpleNamespace(
        v1=types.SimpleNamespace(
            html=lambda body, height, scrolling: calls.append((body, height))
        )
    )
    fake_st.code = lambda body: calls.append(('code', body))
    monkeypatch.setitem(sys.modules, 'streamlit', fake_st)

    sample_graph.visualize(height=333)

    assert len(calls) == 1
    body, height = calls[0]
    assert 'mermaid.initialize' in body
    assert height == 333


def test_visualize_streamlit_fallback(
    sample_graph: Graph,
    no_notebook,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_module, 'in_streamlit_notebook', lambda: True)

    calls: list = []
    fake_st = types.ModuleType('streamlit')
    fake_st.components = types.SimpleNamespace(
        v1=types.SimpleNamespace(html=None)
    )  # Simulate missing custom component support.
    fake_st.code = lambda body: calls.append(body)
    monkeypatch.setitem(sys.modules, 'streamlit', fake_st)

    sample_graph.visualize()

    assert len(calls) == 1
    assert calls[0].startswith('erDiagram')


def test_visualize_terminal_prints_source(
    sample_graph: Graph,
    no_notebook,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list = []
    monkeypatch.setattr('builtins.print', lambda *args: printed.append(args))

    with pytest.warns(UserWarning, match='graph.html'):
        sample_graph.visualize()

    assert len(printed) == 1
    out = printed[0][0]
    assert out.startswith('erDiagram')
    assert 'users o|--o{ orders : user_id' in out
