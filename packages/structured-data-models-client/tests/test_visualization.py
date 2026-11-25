# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualization must be metadata-only, offline and safe for untrusted names."""

from pathlib import Path

import pytest

from sdm_client import Schema


def test_empty_schema_public_journey(tmp_path):
    schema = Schema({})
    assert schema.to_mermaid().startswith('flowchart LR')
    document = schema.to_html()
    assert '<!doctype html>' in document
    assert 'securityLevel' in document
    assert '<script src=' not in document
    schema.visualize(tmp_path / 'schema.html')
    assert (tmp_path / 'schema.html').read_text() == document
    schema.visualize(tmp_path / 'schema.mmd')
    assert (tmp_path / 'schema.mmd').read_text().strip() == schema.to_mermaid()


@pytest.mark.parametrize('path', ['schema', 'schema.pdf'])
def test_bad_export_path(path):
    with pytest.raises(ValueError):
        Schema({}).visualize(path)


def test_options():
    with pytest.raises(TypeError):
        Schema({}).to_mermaid(1)
    with pytest.raises(ValueError):
        Schema({}).visualize(height=0)
    with pytest.raises(TypeError):
        Schema({}).visualize(height=True)


@pytest.fixture
def metadata_schema():
    """Raw metadata tests escaping independently of stricter schema validation."""
    import json
    from types import SimpleNamespace

    from sdm_client import Column, Relationship

    fixture = (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/visualization/schema.json'
    )
    value = json.loads(fixture.read_text())
    schema = object.__new__(Schema)
    object.__setattr__(
        schema,
        '_tables',
        {
            table['name']: SimpleNamespace(
                columns={c['name']: Column(**c) for c in table['columns']},
                primary_key=table['primary_key'],
                time_column=table['time_column'],
            )
            for table in value['tables']
        },
    )
    object.__setattr__(
        schema, 'relationships', [Relationship(**r) for r in value['links']]
    )
    return schema


def _decoded(source):
    import re

    return re.sub(r'#(\d+);', lambda m: chr(int(m[1])), source)


def test_metadata_keys_edges_and_safe_labels(metadata_schema, monkeypatch):
    from sdm_client._rust import run

    monkeypatch.setattr(
        run, 'native', lambda: pytest.fail('presentation called Rust')
    )
    source = metadata_schema.to_mermaid()
    decoded = _decoded(source)
    assert decoded.count('[PK]') == 2
    assert '[FK]' in decoded and '[TIME]' in decoded
    assert 'customer_id → id (many to one)' in decoded
    assert source.count(' -->|') == 1
    assert '</script>' not in source and 'attacker' not in source
    assert 'attacker' not in metadata_schema.to_mermaid(False)
    assert metadata_schema.to_mermaid() == source


def test_image_errors_do_not_disclose_schema(monkeypatch):
    import traceback

    import requests

    from sdm_client import _visualization

    def fail(*args, **kwargs):
        raise requests.ConnectionError('https://mermaid.ink/SECRET-SCHEMA')

    monkeypatch.setattr(requests, 'get', fail)
    with pytest.raises(RuntimeError) as caught:
        _visualization.image('secret-schema', '.png')
    assert 'SECRET-SCHEMA' not in ''.join(
        traceback.format_exception(caught.value)
    )


def test_image_export_and_buffer(monkeypatch, tmp_path):
    import io

    from sdm_client import _visualization

    calls = []
    monkeypatch.setattr(
        _visualization,
        'image',
        lambda source, suffix: calls.append(suffix) or b'image',
    )
    schema = Schema({})
    stream = io.BytesIO()
    schema.visualize(stream)
    schema.visualize(tmp_path / 'schema.svg')
    assert stream.getvalue() == b'image' and not stream.closed
    assert (tmp_path / 'schema.svg').read_bytes() == b'image'
    assert calls == ['.png', '.svg']


def test_vendored_renderer_identity():
    import hashlib
    from importlib.resources import files

    assets = files('sdm_client').joinpath('assets')
    assert (
        hashlib.sha256(
            assets.joinpath('mermaid.min.js').read_bytes()
        ).hexdigest()
        == '70137e77bb273bb2ef972b86e8b0400cca8be53cb25bfc45911a186dc98665de'
    )
    assert (
        'MIT License'
        in assets.joinpath('THIRD-PARTY-NOTICES-MERMAID.txt').read_text()
    )


@pytest.mark.parametrize('language', ['python', 'java'])
def test_real_offline_renderer(metadata_schema, tmp_path, language):
    """Optional browser check: install playwright and its Chromium to execute."""
    browser_api = pytest.importorskip('playwright.sync_api')
    target = tmp_path / 'schema.html'
    if language == 'python':
        metadata_schema.visualize(target)
    else:
        target = (
            Path(__file__).resolve().parents[3]
            / 'java/target/visualization-fixture.html'
        )
        if not target.exists():
            pytest.skip('run Java SchemaVisualizationTest first')
        assert (
            target.with_suffix('.mmd').read_text()
            == metadata_schema.to_mermaid()
        )
    with browser_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            network = []
            page.on('request', lambda request: network.append(request.url))
            page.goto(target.as_uri())
            page.wait_for_selector('svg .node')
            assert page.locator('svg .node').count() == 2
            assert page.locator('.error-text').count() == 0
            labels = ''.join(page.locator('svg .node').all_text_contents())
            assert '</script><script>window.pwned=1</script>' in labels
            assert '[PK]' in labels and '[FK]' in labels and '[TIME]' in labels
            assert 'customer_id → id (many to one)' in ''.join(
                page.locator('.edgeLabel').all_text_contents()
            )
            assert page.evaluate('window.pwned') is None
            assert all(url.startswith('file:') for url in network)
        finally:
            browser.close()


def test_notebook_display_is_sandboxed(monkeypatch):
    import sys
    import types

    from sdm_client import _visualization

    displayed = []
    monkeypatch.setitem(sys.modules, 'streamlit', None)
    monkeypatch.setitem(
        sys.modules,
        'IPython',
        types.SimpleNamespace(
            get_ipython=lambda: types.SimpleNamespace(kernel=True)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'IPython.display',
        types.SimpleNamespace(HTML=lambda text: text, display=displayed.append),
    )
    monkeypatch.setattr(
        _visualization,
        'image',
        lambda *args: pytest.fail('notebook used network'),
    )
    Schema({}).visualize(height=123)
    assert len(displayed) == 1
    assert 'sandbox="allow-scripts"' in displayed[0]
    assert 'allow-same-origin' not in displayed[0]
    assert 'height="123"' in displayed[0]


def test_streamlit_display_and_restricted_fallback(monkeypatch):
    import sys
    import types

    rendered = []
    fallback = []
    component = types.SimpleNamespace(
        html=lambda *args, **kwargs: rendered.append((args, kwargs))
    )
    monkeypatch.setitem(
        sys.modules,
        'streamlit',
        types.SimpleNamespace(
            components=types.SimpleNamespace(v1=component), code=fallback.append
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        'streamlit.runtime.scriptrunner',
        types.SimpleNamespace(get_script_run_ctx=lambda **kwargs: object()),
    )
    schema = Schema({})
    schema.visualize()
    assert rendered[0][1]['scrolling'] is True

    def unavailable(*args, **kwargs):
        raise RuntimeError('disabled by notebook host')

    component.html = unavailable
    schema.visualize()
    assert fallback == [schema.to_mermaid()]


def test_image_request_explicit_format_timeout_and_metadata(monkeypatch):
    import base64
    import json
    from types import SimpleNamespace

    import requests

    from sdm_client import _visualization

    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, content=b'image')

    monkeypatch.setattr(requests, 'get', get)
    assert _visualization.image('flowchart LR', '.png') == b'image'
    url, options = calls[0]
    assert url.startswith('https://mermaid.ink/img/base64:')
    assert url.endswith('?type=png') and options['timeout'] == 30
    payload = json.loads(
        base64.urlsafe_b64decode(url.split('base64:', 1)[1].split('?')[0])
    )
    assert payload['code'] == 'flowchart LR'
    assert payload['mermaid']['securityLevel'] == 'strict'
