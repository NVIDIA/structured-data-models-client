# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the maintained architecture and bindings SVGs with Python alone."""

from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent


def diagram(
    name: str, title: str, boxes: list, arrows: list, notes: list
) -> None:
    """Write a fixed-layout diagram; coordinates and labels are the source."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="760" '
        'viewBox="0 0 900 760" role="img" aria-labelledby="title" '
        'font-family="Arial, sans-serif">',
        '<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION &amp; '
        'AFFILIATES. SPDX-License-Identifier: Apache-2.0 -->',
        f'<title id="title">{escape(title)}</title>',
        '<rect width="900" height="760" fill="white"/>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/></marker></defs>',
        f'<text x="450" y="38" text-anchor="middle" font-size="24" '
        f'font-weight="bold" fill="#222">{escape(title)}</text>',
    ]
    colors = {'host': ('#eaf2fc', '#3d72b8'), 'core': ('#fdf3e6', '#c9781c')}
    for x, y, width, height, kind, lines in boxes:
        fill, stroke = colors.get(kind, ('#f7f7f7', '#666'))
        parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        for i, line in enumerate(lines):
            weight = 'bold' if i == 0 else 'normal'
            parts.append(
                f'<text x="{x + width / 2}" y="{y + 29 + i * 25}" '
                f'text-anchor="middle" font-size="17" font-weight="{weight}" '
                f'fill="#222">{escape(line)}</text>'
            )
    for x1, y1, x2, y2, label, tx, ty in arrows:
        parts.append(
            f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#555" '
            'stroke-width="2" marker-end="url(#arrow)"/>'
        )
        if label:
            parts.append(
                f'<text x="{tx}" y="{ty}" text-anchor="middle" '
                f'font-size="15" fill="#444">{escape(label)}</text>'
            )
    for y, line in notes:
        parts.append(
            f'<text x="450" y="{y}" text-anchor="middle" '
            f'font-size="16" fill="#444">{escape(line)}</text>'
        )
    parts.append('</svg>')
    (HERE / name).write_text('\n'.join(parts) + '\n')


def main() -> None:
    diagram(
        'architecture.svg',
        'Prediction flow: engine decisions, host I/O',
        [
            (
                160,
                75,
                580,
                105,
                'host',
                [
                    'Python / Java public API',
                    'Schema, query or direct input; materialize / fit / predict',
                    'Connections and conversion to typed inputs stay here',
                ],
            ),
            (
                70,
                245,
                440,
                180,
                'core',
                [
                    'Rust engine',
                    'Validate schema and PQL; select eligible context',
                    'Sample, label and construct model requests',
                    'Interpret results and explanation values',
                    'Return typed steps and results',
                ],
            ),
            (
                585,
                265,
                260,
                125,
                'host',
                [
                    'Host I/O loop',
                    'Run SQL / read provider data',
                    'Send model requests',
                    'Return Arrow / response bytes',
                ],
            ),
            (
                585,
                460,
                260,
                105,
                'outside',
                [
                    'External systems',
                    'Database / files',
                    'Model endpoint',
                ],
            ),
            (
                70,
                510,
                440,
                105,
                'host',
                [
                    'Python / Java result',
                    'Native errors, pandas / Arrow views',
                    'Display and export',
                ],
            ),
        ],
        [
            (290, 180, 290, 245, 'typed operation + data', 390, 215),
            (510, 290, 585, 290, 'steps', 548, 277),
            (585, 370, 510, 370, 'replies', 548, 357),
            (715, 390, 715, 460, '', 0, 0),
            (290, 425, 290, 510, 'typed result', 355, 474),
        ],
        [
            (
                670,
                'The Rust core opens no database connection or network socket.',
            ),
            (
                698,
                'In-memory inputs use the existing native graph/data path; no warehouse is required.',
            ),
            (
                726,
                'Rust owns engine meaning, not every helper shared by Python and Java.',
            ),
        ],
    )
    diagram(
        'bindings.svg',
        'Bindings and ownership',
        [
            (
                70,
                80,
                355,
                130,
                'host',
                [
                    'Python',
                    'PyO3 / sdm-py',
                    'maturin wheel, abi3',
                    'Arrow admission through PyCapsule',
                ],
            ),
            (
                475,
                80,
                355,
                130,
                'host',
                [
                    'Java',
                    'JNI / sdm-jni',
                    'JAR + platform native library',
                    'Arrow Java C Data admission',
                ],
            ),
            (
                165,
                280,
                570,
                155,
                'core',
                [
                    'sdm-host: native boundary machinery',
                    'Versioned messages, Arrow admission and resource ownership',
                    'Run lifetime, busy-state checks and panic containment',
                    'Bindings depend on sdm-host and sdm-contract',
                ],
            ),
            (
                165,
                500,
                570,
                105,
                'core',
                [
                    'sdm-core and engine crates',
                    'Run state machine, schema, sampling, requests and results',
                    'Current PQL implementation: repository Rust sdm-pql',
                ],
            ),
        ],
        [
            (247, 210, 247, 280, '', 0, 0),
            (652, 210, 652, 280, '', 0, 0),
            (450, 435, 450, 500, 'typed operations', 535, 472),
        ],
        [
            (
                650,
                'Typed attachments retain their owners until consumed or released.',
            ),
            (
                678,
                'Arrow exchange can share buffers; input normalization and algorithms may allocate.',
            ),
            (
                706,
                'Bindings adapt runtime objects and logging; engine policy has no Python/Java dependency.',
            ),
            (
                734,
                'No user sorting callbacks or external C++ PQL runtime are part of this engine.',
            ),
        ],
    )


if __name__ == '__main__':
    main()
