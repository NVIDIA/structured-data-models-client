# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record successful scenarios from actual frozen R1/R2 receipts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

PRODUCERS = {
    1: '14e01a43eb9d176a3419bf4c428d986e039657732be4d6318cee59ecd81c5eed',
    2: 'f10c51a9f2fdb2bcaeba8d9e602c215bd72400f95328f93a0a13b6fb803dd5ea',
}
SOURCES = {
    1: 'b6574b5536f6129e350d9919f23122836cca2dc48c488f297c844e596b130523',
    2: '8f66dfc1bafc25eee45f318952da6b6d2f2af53ed1a1dba402a5b959c21256d5',
}
REFERENCE = '43e55f4f58924384c3cb52f67c8ed2e73ba45fa1'
CAPTURES = {
    (
        1,
        '310',
    ): '95ef583a6c9c0fcaae358d7d88cca8ebbcdca33e2431a8e908e58913439c7b5a',
    (
        1,
        '312',
    ): '4f6bbd0e2d3ad2d081582ccf75ecbcd269b5a717bb11348200e8934bdda38086',
    (
        2,
        '310',
    ): '47cbc9a3c7b9d506901960c1086d3f9fda55684be28bab56e3aa54b64ddcf8a5',
    (
        2,
        '312',
    ): 'bc01cd74769dbc9cdecad2b64296a2a6fc9cc7aed13f9d80bdf975aaa367cd37',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FoldFactory(ast.NodeTransformer):
    """Specialize only the measured fold case, without running its code."""

    def visit_Compare(self, node):
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == 'case'
            and len(node.ops) == len(node.comparators) == 1
            and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.comparators[0], ast.Constant)
        ):
            return ast.Constant(node.comparators[0].value == 'fold-entity')
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.test, ast.Constant)
            and type(node.test.value) is bool
        ):
            return node.body if node.test.value else node.orelse
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if (
            isinstance(node.test, ast.Constant)
            and type(node.test.value) is bool
        ):
            return node.body if node.test.value else node.orelse
        return node


def scenario_definitions(path):
    return {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in {'question', 'Scripted', 'observe'}
    }


def same_fold_source(packet):
    first = scenario_definitions(packet / 'scenario_r1.py')
    second = scenario_definitions(packet / 'scenario_r2.py')
    assert set(first) == set(second) == {'question', 'Scripted', 'observe'}
    for name in first:
        left, right = first[name], second[name]
        if name == 'question':
            left, right = FoldFactory().visit(left), FoldFactory().visit(right)
        assert ast.dump(left) == ast.dump(right), name


def read(packet, round_number, cell):
    assert (
        digest(packet / f'probe_r{round_number}.py') == PRODUCERS[round_number]
    )
    assert (
        digest(packet / f'scenario_r{round_number}.py') == SOURCES[round_number]
    )
    terminal_name = (
        'execution-terminal.private.json'
        if round_number == 1
        else 'execution-r2-terminal.private.json'
    )
    terminal = json.loads((packet / terminal_name).read_text())
    assert terminal['exit_code'] == 0
    assert terminal['producer_sha256'] == PRODUCERS[round_number]
    assert [item['cell'] for item in terminal['cells']] == ['310', '312']
    assert all(item['exit_code'] == 0 for item in terminal['cells'])
    receipt = packet / f'frozen-fold-keyless-r{round_number}-{cell}.json'
    assert digest(receipt) == CAPTURES[round_number, cell]
    captured = json.loads(receipt.read_text())
    assert captured['reference'] == REFERENCE
    assert captured['producer_sha256'] == PRODUCERS[round_number]
    assert captured['scenario_source_sha256'] == SOURCES[round_number]
    assert captured['cell'] == cell
    assert captured['network_attempts'] == []
    return captured, digest(receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    tests = root / 'packages/structured-data-models-client/tests'
    assert digest(tests / 'test_t44_fold_keyless_frames.py') == SOURCES[2]
    same_fold_source(args.packet)
    for cell, runtime in [('310', 'pandas2'), ('312', 'pandas3')]:
        first, first_digest = read(args.packet, 1, cell)
        second, second_digest = read(args.packet, 2, cell)
        assert set(first['records']) == {'fold-entity', 'keyless-history'}
        assert set(second['records']) == {'keyless-history'}
        assert first['records']['keyless-history']['outcome'] == 'error'
        assert first['records']['keyless-history']['stage'] == 'schema'
        records = {
            'fold-entity': first['records']['fold-entity'],
            'keyless-history': second['records']['keyless-history'],
        }
        assert all(
            record['outcome'] == 'success' for record in records.values()
        )
        output = {
            'reference': REFERENCE,
            'cell': cell,
            'captures': [
                {
                    'round': round_number,
                    'producer_sha256': PRODUCERS[round_number],
                    'scenario_source_sha256': SOURCES[round_number],
                    'capture_sha256': captured_digest,
                }
                for round_number, captured_digest in [
                    (1, first_digest),
                    (2, second_digest),
                ]
            ],
            'records': records,
        }
        path = tests / f't44_fold_keyless_{runtime}_reference.json'
        with path.open('x') as stream:
            json.dump(
                output, stream, indent=2, ensure_ascii=True, allow_nan=False
            )
            stream.write('\n')


if __name__ == '__main__':
    main()
