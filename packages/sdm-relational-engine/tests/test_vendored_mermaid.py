# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""Checks that the vendored mermaid bundle is upstream bytes, unmodified.

The third-party attribution in the root LICENSE is enumerated from the mermaid
release named in VENDORED.md. That enumeration is only truthful while the file
on disk is the file upstream published, so the digest is pinned and checked
here rather than trusted.

This exists because a repo-wide identifier rename once rewrote two strings
inside the minified bundle, turning a vendored artifact into a modified one
without changing its stated provenance.
"""

import hashlib
import re
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / (
    'src/sdm_relational_engine/rfm/assets'
)
BUNDLE = ASSETS / 'mermaid.min.js'
VENDORED = ASSETS / 'VENDORED.md'
NOTICE_DIGEST = (
    '957479759ea620fab38a31a961c21c7ae23bee82be7e302ad44a3db0079cf20b'
)


def _declared(field: str) -> str:
    match = re.search(rf'^- {field}: `?([^`\n(]+)', VENDORED.read_text(), re.M)
    assert match, f'VENDORED.md declares no {field}'
    return match.group(1).strip()


def test_bundle_matches_the_digest_vendored_md_declares() -> None:
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert digest == _declared('SHA-256'), (
        'mermaid.min.js does not match the SHA-256 in VENDORED.md. Either it '
        'was edited in place, which breaks its third-party attribution, or it '
        'was upgraded without updating VENDORED.md and the LICENSE '
        'enumeration.'
    )


def test_license_enumerates_the_vendored_mermaid_version() -> None:
    version = _declared('Version').split()[0]
    licence = (Path(__file__).resolve().parents[3] / 'LICENSE').read_text()
    assert f'mermaid {version}' in licence, (
        f'The root LICENSE does not enumerate mermaid {version}. The bundled '
        'components must be re-enumerated from the manifest of the version '
        'actually vendored.'
    )


def test_bundle_carries_no_first_party_identifiers() -> None:
    text = BUNDLE.read_text(encoding='utf-8', errors='replace')
    for token in ('Kumo', 'kumo', 'Relational', 'Nemotron', 'nemotron'):
        assert token not in text, (
            f'{token!r} appears in the vendored mermaid bundle, so a rename '
            'has rewritten upstream bytes.'
        )


def _notice_blocks() -> str:
    r"""Upstream's own license notices, as the bundle carries them."""
    bundle = BUNDLE.read_text(encoding='utf-8', errors='replace')
    blocks = [
        block
        for block in re.findall(r'/\*![\s\S]{0,4000}?\*/', bundle)
        if 'license' in block.lower()
    ]
    assert blocks, 'the bundle carries no license notices'
    return '\n'.join(blocks)


def _versions(text: str) -> set[str]:
    return set(re.findall(r'(?<![\w.])(\d+\.\d+\.\d+)(?![\w.])', text))


def test_bundled_license_notices_are_unchanged() -> None:
    r"""Pins upstream's notices so any change to them is caught.

    The notices name components in shapes no single pattern reads reliably:
    ``DOMPurify 3.4.0``, ``lodash/lodash#4.18.1`` and ``Embeddable Minimum
    Strictly-Compliant Promises/A+ 1.1.1 Thenable`` all appear. Rather than
    claim to parse each one, this pins their digest. An upgrade that changes
    any notice fails here, which is the signal to re-enumerate the LICENSE by
    hand from the new bundle.
    """
    digest = hashlib.sha256(_notice_blocks().encode()).hexdigest()
    assert digest == NOTICE_DIGEST, (
        "the bundle's license notices changed. Re-read them and re-enumerate "
        'the bundled components in the root LICENSE before updating this '
        'digest.'
    )


def test_license_states_no_version_the_notices_do_not() -> None:
    r"""Every version the LICENSE claims must be one the bundle declares.

    This is the direction that matters: attributing a component at a version
    upstream does not ship is a false statement about what we distribute. It is
    how ``js-yaml 4.1.0`` survived the upgrade to a bundle carrying 4.1.1. The
    reverse is not asserted, because the LICENSE deliberately attributes some
    components without pinning a version.
    """
    licence = (Path(__file__).resolve().parents[3] / 'LICENSE').read_text()
    section = licence[licence.index('THIRD-PARTY SOFTWARE BUNDLED') :]
    declared = _versions(_notice_blocks()) | {_declared('Version').split()[0]}

    unsupported = _versions(section) - declared
    assert not unsupported, (
        f'the LICENSE attributes bundled components at {sorted(unsupported)}, '
        'which neither the bundle notices nor VENDORED.md declare.'
    )
