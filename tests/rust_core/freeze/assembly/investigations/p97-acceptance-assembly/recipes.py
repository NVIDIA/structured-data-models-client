# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless ordered recipe envelope, including order-significant frame columns."""

from __future__ import annotations

from support import json_bytes, require, sha, unique_json


def encode(value):
    if isinstance(value, dict):
        require(all(type(key) is str for key in value), 'recipe.key')
        return {
            'mapping': [[key, encode(child)] for key, child in value.items()]
        }
    if isinstance(value, list):
        return {'sequence': [encode(child) for child in value]}
    require(type(value) in {str, int, float, bool, type(None)}, 'recipe.scalar')
    json_bytes(value)
    return {'scalar': value}


def decode(value):
    require(type(value) is dict and len(value) == 1, 'recipe.node')
    if 'mapping' in value:
        require(type(value['mapping']) is list, 'recipe.mapping')
        result = {}
        for pair in value['mapping']:
            require(
                type(pair) is list
                and len(pair) == 2
                and type(pair[0]) is str
                and pair[0] not in result,
                'recipe.mapping_entry',
            )
            result[pair[0]] = decode(pair[1])
        return result
    if 'sequence' in value:
        require(type(value['sequence']) is list, 'recipe.sequence')
        return [decode(child) for child in value['sequence']]
    require('scalar' in value, 'recipe.tag')
    require(encode(value['scalar']) == value, 'recipe.scalar')
    return value['scalar']


def seal(recipe: dict) -> dict:
    tree = encode(recipe)
    raw = json_bytes(tree)
    require(encode(decode(unique_json(raw))) == tree, 'recipe.roundtrip')
    return {
        'format': 'p97-ordered-recipe-v1',
        'tree': tree,
        'input_sha256': sha(raw),
    }


def open_recipe(envelope: dict) -> dict:
    require(
        set(envelope) == {'format', 'tree', 'input_sha256'}
        and envelope['format'] == 'p97-ordered-recipe-v1',
        'recipe.envelope',
    )
    recipe = decode(envelope['tree'])
    require(type(recipe) is dict and seal(recipe) == envelope, 'recipe.binding')
    return recipe
