#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Generate the Rust wire types from the Structured Data API contract.

Every schema under ``components.schemas`` becomes one Rust item of the same
name, with serde attributes that make a parse and a re-serialization give
back the same JSON: an absent key stays absent, a null stays null, an integer
stays an integer and a float a float, and an object the contract leaves open
keeps the keys it does not name. Objects the contract closes refuse unknown
keys, and enums refuse unknown values.

The output is written already formatted the way ``rustfmt`` would, so this
needs no Rust toolchain, and ``--check`` compares bytes.

Beside the output, ``PROVENANCE.json`` records it the way
``scripts/generate_sdm_api.py`` records ``tfm_api.py``: the generated file's
and this generator's hashes, and the contract's path, revision and bytes. The
generator is self-contained on purpose, so that its hash covers everything
that decides the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = Path('rust/crates/wire/src/generated.rs')
GENERATOR = 'scripts/generate_wire_rs.py'
SOURCE_REPOSITORY = 'structured-data-api'

# rustfmt's defaults, `max_width` and `attr_fn_like_width`.
MAX_WIDTH = 100
ATTR_ARGS_WIDTH = 70
INDENT = '    '

RUST_KEYWORDS = {
    'as', 'async', 'await', 'break', 'const', 'continue', 'dyn', 'else',
    'enum', 'extern', 'false', 'fn', 'for', 'if', 'impl', 'in', 'let',
    'loop', 'match', 'mod', 'move', 'mut', 'pub', 'ref', 'return', 'static',
    'struct', 'trait', 'true', 'type', 'unsafe', 'use', 'where', 'while',
    'abstract', 'become', 'box', 'do', 'final', 'gen', 'macro', 'override',
    'priv', 'try', 'typeof', 'unsized', 'virtual', 'yield',
}  # fmt: skip
# Keywords a raw identifier cannot spell.
NOT_RAW = {'crate', 'self', 'Self', 'super', '_'}

# What the generator understands. A keyword outside these could change what
# a value may be, so it stops the generator rather than being dropped.
SHAPE_KEYS = {
    '$ref', 'type', 'format', 'enum', 'properties', 'required',
    'additionalProperties', 'items', 'oneOf', 'anyOf', 'discriminator',
    'nullable',
}  # fmt: skip
# The keywords that only an object has.
OBJECT_KEYS = {'properties', 'required', 'additionalProperties'}
# Constraints the wire types do not enforce: they hold values the contract
# bounds, and checking the bounds is the request's job, not the parser's.
ANNOTATION_KEYS = {
    'description', 'title', 'example', 'examples', 'default', 'deprecated',
    'readOnly', 'writeOnly', 'maxLength', 'minLength', 'pattern', 'maximum',
    'minimum', 'exclusiveMaximum', 'exclusiveMinimum', 'multipleOf',
    'maxItems', 'minItems', 'uniqueItems', 'maxProperties', 'minProperties',
    'externalDocs', 'xml',
}  # fmt: skip


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _string_map(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(v, str) for v in value.values()
    )


# The kind of value each structural keyword takes.
VALUE_KINDS = (
    ('type', lambda v: isinstance(v, str), 'a string'),
    ('properties', lambda v: isinstance(v, dict), 'a map of schemas'),
    ('required', _strings, 'a list of names'),
    ('enum', lambda v: isinstance(v, list) and bool(v), 'a list of values'),
    ('nullable', lambda v: isinstance(v, bool), 'a boolean'),
    (
        'discriminator',
        lambda v: (
            isinstance(v, dict)
            and isinstance(v.get('propertyName'), str)
            and _string_map(v.get('mapping', {}))
        ),
        'a propertyName and a mapping of names to references',
    ),
)

PRESENT = 'deserialize_with = "fields::present"'
NULLABLE = 'deserialize_with = "fields::nullable"'
OPTIONAL = ('default', 'skip_serializing_if = "Option::is_none"', PRESENT)

OPEN_OBJECT = 'Map<String, Value>'


class ContractError(ValueError):
    r"""The contract uses something the generator cannot express."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--spec',
        required=True,
        help='Path to the contract, nim-sd.openapi.yaml, in a checkout of '
        'the contract repository at the revision to record.',
    )
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT),
        help=f'The generated Rust file. Defaults to {DEFAULT_OUTPUT}.',
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Write nothing; exit non-zero unless the generated file and its '
        'record are what this would write.',
    )
    parser.add_argument(
        '--no-record',
        dest='record',
        action='store_false',
        help='Neither write nor check PROVENANCE.json. For a contract that is '
        'a test fixture, which has no revision of its own to record.',
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec_bytes = spec_path.read_bytes()
    spec = load_spec(spec_bytes, spec_path)
    spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    try:
        code = generate(spec, source=spec_path.name, source_sha256=spec_sha256)
    except ContractError as error:
        print(f'cannot generate the wire types: {error}', file=sys.stderr)
        return 1

    output = Path(args.output)
    files = {output: code}
    if args.record:
        files[output.parent / 'PROVENANCE.json'] = provenance(
            output,
            code,
            spec_path,
            spec_sha256,
            revision=spec_revision(spec_path),
        )

    if args.check:
        stale = [
            str(path)
            for path, text in files.items()
            if not path.is_file() or path.read_text() != text
        ]
        if stale:
            print(
                f'not up to date with {spec_path}: {", ".join(stale)}; run '
                f'python {GENERATOR} --spec {spec_path}',
                file=sys.stderr,
            )
            return 1
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        path.write_text(text)
    return 0


def load_spec(data: bytes, path: Path) -> dict[str, Any]:
    if path.suffix == '.json':
        spec = json.loads(data)
    else:
        import yaml

        spec = yaml.safe_load(data)
    if not isinstance(spec, dict) or 'components' not in spec:
        raise SystemExit(f'{path} is not an OpenAPI document with components')
    return spec


def provenance(
    output: Path,
    code: str,
    spec_path: Path,
    spec_sha256: str,
    *,
    revision: str,
) -> str:
    record = {
        'generated_file': output.name,
        'generated_sha256': hashlib.sha256(code.encode()).hexdigest(),
        'generator': GENERATOR,
        'generator_sha256': hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        'source_path': spec_path.name,
        'source_repository': SOURCE_REPOSITORY,
        'source_revision': revision,
        'source_sha256': spec_sha256,
    }
    return json.dumps(record, indent=2, sort_keys=True) + '\n'


def spec_revision(spec_path: Path) -> str:
    r"""The contract's git revision, read from the checkout it is in."""
    try:
        return subprocess.check_output(
            ['git', '-C', str(spec_path.resolve().parent), 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return ''


# --- The model ---------------------------------------------------------------


@dataclass
class Field:
    json_name: str
    rust_name: str
    rust_type: str
    attrs: list[str]
    doc: str | None


@dataclass
class Struct:
    name: str
    doc: str | None
    closed: bool
    fields: list[Field] = field(default_factory=list)


@dataclass
class Enum:
    name: str
    doc: str | None
    values: list[tuple[str, str]]


@dataclass
class Union:
    name: str
    doc: str | None
    variants: list[tuple[str, str | None]]  # (variant, type); None is null
    tag: str | None = None
    tags: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Alias:
    name: str
    doc: str | None
    rust_type: str


Item = Struct | Enum | Union | Alias


class Generator:
    def __init__(self, schemas: dict[str, Any]) -> None:
        self.schemas = schemas
        self.items: list[Item] = []
        self.names: set[str] = set()

    def run(self) -> list[Item]:
        for name, schema in self.schemas.items():
            if not re.fullmatch(r'[A-Z][A-Za-z0-9]*', name):
                raise ContractError(f'schema name {name!r} is not a Rust type')
            self.top_level(name, schema)
        return self.items

    def claim(self, name: str) -> str:
        if name in self.names:
            raise ContractError(f'two items would be named {name}')
        self.names.add(name)
        return name

    def top_level(self, name: str, schema: dict[str, Any]) -> None:
        _known_keys(schema, name)
        doc = schema.get('description')
        named = 'oneOf' in schema or 'anyOf' in schema or 'enum' in schema
        if schema.get('nullable') and (named or _is_struct(schema)):
            raise ContractError(f'{name}: a nullable named type')
        if 'oneOf' in schema or 'anyOf' in schema:
            self.union(name, schema)
        elif 'enum' in schema:
            self.enum(name, schema)
        elif _is_struct(schema):
            self.struct(name, schema)
        else:
            self.items.append(
                Alias(self.claim(name), doc, self.type_of(schema, name))
            )

    def type_of(self, schema: dict[str, Any], context: str) -> str:
        r"""The Rust type of a value of ``schema``, naming inline items
        after ``context``: an ``Option`` when the value may be null."""
        rust_type = self.shape_of(schema, context)
        if schema.get('nullable') and rust_type != 'Value':
            return f'Option<{rust_type}>'
        return rust_type

    def shape_of(self, schema: dict[str, Any], context: str) -> str:
        r"""The Rust type of a value of ``schema`` that is not null."""
        _known_keys(schema, context)
        if '$ref' in schema:
            return self.ref(schema['$ref'])
        if 'oneOf' in schema or 'anyOf' in schema:
            return self.union(context, schema)
        if 'enum' in schema:
            return self.enum(context, schema)
        if _is_struct(schema):
            return self.struct(context, schema)
        kind = schema.get('type')
        if kind == 'object':
            return self.map_of(
                schema.get('additionalProperties', True), context
            )
        if kind == 'array':
            if 'items' not in schema:
                raise ContractError(f'{context}: an array without items')
            return f'Vec<{self.type_of(schema["items"], context + "Item")}>'
        if kind == 'string':
            return 'String'
        if kind == 'integer':
            return 'i32' if schema.get('format') == 'int32' else 'i64'
        if kind == 'number':
            return 'Number'
        if kind == 'boolean':
            return 'bool'
        if kind is None:
            return 'Value'
        raise ContractError(f'{context}: unknown type {kind!r}')

    def ref(self, ref: str) -> str:
        prefix = '#/components/schemas/'
        name = ref[len(prefix) :] if ref.startswith(prefix) else None
        if name not in self.schemas:
            raise ContractError(f'unresolved reference {ref}')
        return name

    def enum(self, name: str, schema: dict[str, Any]) -> str:
        values = schema['enum']
        if schema.get('type', 'string') != 'string' or not all(
            isinstance(v, str) for v in values
        ):
            raise ContractError(f'{name}: only string enums are supported')
        variants = []
        for value in values:
            variant = _pascal(value)
            if variant in {v for v, _ in variants}:
                raise ContractError(f'{name}: two values would be {variant}')
            variants.append((variant, value))
        self.items.append(
            Enum(self.claim(name), schema.get('description'), variants)
        )
        return name

    def union(self, name: str, schema: dict[str, Any]) -> str:
        if 'oneOf' in schema and 'anyOf' in schema:
            raise ContractError(f'{name}: both oneOf and anyOf')
        members = schema.get('oneOf', schema.get('anyOf'))
        if 'type' in schema:
            raise ContractError(f'{name}: a union with its own shape')
        if not isinstance(members, list) or not members:
            raise ContractError(f'{name}: a union is a list of schemas')
        for member in members:
            _known_keys(member, name)
        self.claim(name)
        item = Union(name, schema.get('description'), [])
        discriminator = schema.get('discriminator')
        if discriminator is not None:
            item.tag = discriminator['propertyName']
            if any('$ref' not in m for m in members):
                raise ContractError(f'{name}: a tagged member is not a $ref')
            targets = [self.ref(m['$ref']) for m in members]
            mapping = {
                tag: self.ref(ref)
                for tag, ref in discriminator.get('mapping', {}).items()
            }
            for tag, target in mapping.items():
                if target not in targets:
                    raise ContractError(f'{name}: {tag} maps outside oneOf')
            # A member the mapping leaves out is selected by its own name.
            for target in targets:
                if target not in mapping.values():
                    mapping[target] = target
            item.tags = list(mapping.items())
            item.variants = [(t, t) for t in dict.fromkeys(targets)]
            self.items.append(item)
            return name
        nullable = False
        for member in members:
            nullable = nullable or bool(member.get('nullable'))
            if '$ref' in member:
                variant = self.ref(member['$ref'])
            else:
                variant = _pascal(member.get('type') or 'value')
            if variant in {v for v, _ in item.variants} | {'Null'}:
                raise ContractError(f'{name}: two members would be {variant}')
            item.variants.append(
                (variant, self.shape_of(member, name + variant))
            )
        if nullable:
            item.variants.append(('Null', None))
        self.items.append(item)
        return name

    def map_of(self, extra: bool | dict[str, Any], context: str) -> str:
        r"""The Rust type of an object's keys beyond those it names, by its
        ``additionalProperties``, which is not ``false``."""
        if extra is True or extra == {}:
            return OPEN_OBJECT
        return f'IndexMap<String, {self.type_of(extra, context + "Value")}>'

    def struct(self, name: str, schema: dict[str, Any]) -> str:
        extra = schema.get('additionalProperties', True)
        item = Struct(
            self.claim(name), schema.get('description'), extra is False
        )
        self.items.append(item)
        properties = schema.get('properties', {})
        required = schema.get('required', [])
        unknown = [r for r in required if r not in properties]
        names = {}
        for json_name in [*properties, *unknown]:
            prop = properties.get(json_name, {})
            rust_name = _snake(json_name)
            if rust_name in names:
                raise ContractError(
                    f'{name}: {json_name!r} and {names[rust_name]!r} would '
                    f'both be {rust_name}'
                )
            names[rust_name] = json_name
            rust_type = self.type_of(prop, name + _pascal(json_name))
            attrs = []
            if rust_name.removeprefix('r#') != json_name:
                attrs.append(f'rename = {_rust_str(json_name)}')
            if json_name not in required:
                # Absent is None, present is Some, even when it is null.
                rust_type = f'Option<{rust_type}>'
                attrs.extend(OPTIONAL)
            elif rust_type.startswith('Option<'):
                # serde would take an absent key for null; it is refused.
                attrs.append(NULLABLE)
            item.fields.append(
                Field(
                    json_name,
                    rust_name,
                    rust_type,
                    attrs,
                    prop.get('description'),
                )
            )
        if extra is not False:
            rust_name = 'additional_properties'
            while rust_name in names:
                rust_name += '_'
            item.fields.append(
                Field(
                    '',
                    rust_name,
                    self.map_of(extra, name),
                    ['flatten'],
                    'Keys the contract does not name, kept as they came.',
                )
            )
        return name


def _known_keys(schema: dict[str, Any], where: str) -> None:
    r"""Stop unless every keyword of ``schema`` is one the generator models,
    on a shape it applies to, so that none is dropped."""
    if not isinstance(schema, dict):
        raise ContractError(f'{where}: a schema is an object, not {schema!r}')
    unknown = set(schema) - SHAPE_KEYS - ANNOTATION_KEYS
    unknown = {k for k in unknown if not k.startswith('x-')}
    if unknown:
        raise ContractError(f'{where}: unsupported keywords {sorted(unknown)}')
    # A keyword's value of the wrong kind would be read wrongly, not refused:
    # a string of required names, say, would be taken letter by letter.
    for key, valid, what in VALUE_KINDS:
        if key in schema and not valid(schema[key]):
            raise ContractError(
                f'{where}: {key} is {schema[key]!r}, not {what}'
            )
    kind = schema.get('type')
    if 'discriminator' in schema and not (
        'oneOf' in schema or 'anyOf' in schema
    ):
        raise ContractError(f'{where}: a discriminator without oneOf or anyOf')
    if 'items' in schema and kind != 'array':
        raise ContractError(f'{where}: items on a schema that is no array')
    # Without a type, a schema also takes values that are not objects, so
    # its object keywords cannot make it a struct or a map.
    object_keys = sorted(set(schema) & OBJECT_KEYS)
    if object_keys and kind is None:
        raise ContractError(f'{where}: {object_keys} without type object')
    if object_keys and kind != 'object':
        raise ContractError(f'{where}: {object_keys} on a {kind} schema')
    extra = schema.get('additionalProperties', True)
    if not isinstance(extra, bool | dict):
        raise ContractError(
            f'{where}: additionalProperties is {extra!r}, not a boolean or '
            'a schema'
        )


def _is_struct(schema: dict[str, Any]) -> bool:
    return schema.get('type') == 'object' and (
        'properties' in schema
        or 'required' in schema
        or schema.get('additionalProperties', True) is False
    )


def _pascal(value: str) -> str:
    words = re.findall(r'[A-Za-z0-9]+', value)
    name = ''.join(w[:1].upper() + w[1:] for w in words)
    if not name or name[0].isdigit():
        name = 'V' + name
    return name


def _snake(value: str) -> str:
    name = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', value)
    name = re.sub(r'[^A-Za-z0-9_]+', '_', name).strip('_').lower() or 'field'
    if name[0].isdigit():
        name = 'f_' + name
    if name in NOT_RAW:
        return name + '_'
    if name in RUST_KEYWORDS:
        return 'r#' + name
    return name


# --- Rendering ---------------------------------------------------------------


def generate(spec: dict[str, Any], *, source: str, source_sha256: str) -> str:
    schemas = spec['components'].get('schemas') or {}
    items = Generator(schemas).run()
    used = '\n'.join(_render(item) for item in items)
    imports = ['use serde::{Deserialize, Serialize};']
    if 'IndexMap<' in used:
        imports.insert(0, 'use indexmap::IndexMap;')
    values = sorted(
        name
        for name in ('Map', 'Number', 'Value')
        if re.search(rf'\b{name}\b', used)
    )
    if any(isinstance(i, Union) and i.tag for i in items):
        imports[-1] = 'use serde::{Deserialize, Deserializer, Serialize};'
    if values:
        joined = ', '.join(values)
        imports.append(
            f'use serde_json::{{{joined}}};' if len(values) > 1 else
            f'use serde_json::{joined};'
        )  # fmt: skip
    if 'fields::' in used:
        imports.append('')
        imports.append('use crate::fields;')
    header = [
        '// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION '
        '& AFFILIATES.',
        '// All rights reserved.',
        '// SPDX-License-Identifier: Apache-2.0',
        '',
        f'// Code generated by {GENERATOR}; DO NOT EDIT.',
        f'// Source: {source}',
        f'// Source SHA256: {source_sha256}',
        '',
        "//! The types of the contract's `components.schemas`, one item each.",
        '',
        *imports,
        '',
    ]
    return '\n'.join(header) + '\n' + used


def _doc(text: str | None, indent: str) -> list[str]:
    if not text:
        return []
    lines = text.strip('\n').split('\n')
    return [f'{indent}///{" " + line.rstrip() if line.strip() else ""}'
            for line in lines]  # fmt: skip


def _attr(items: list[str] | tuple[str, ...], indent: str) -> list[str]:
    r"""A serde attribute, laid out as rustfmt lays it out: on one line while
    its arguments fit ``attr_fn_like_width``, else one argument a line."""
    args = ', '.join(items)
    one = f'{indent}#[serde({args})]'
    if len(args) <= ATTR_ARGS_WIDTH and len(one) <= MAX_WIDTH:
        return [one]
    inner = indent + INDENT
    lines = [f'{indent}#[serde(']
    lines.extend(f'{inner}{item},' for item in items[:-1])
    lines.append(f'{inner}{items[-1]}')
    lines.append(f'{indent})]')
    return lines


def _render(item: Item) -> str:
    lines: list[str] = []
    doc = _doc(item.doc, '')
    if isinstance(item, Alias):
        lines += doc
        lines.append(f'pub type {item.name} = {item.rust_type};')
    elif isinstance(item, Enum):
        lines += doc
        lines.append(
            '#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, '
            'Deserialize)]'
        )
        lines.append(f'pub enum {item.name} {{')
        for variant, value in item.values:
            if variant != value:
                lines += _attr([f'rename = {_rust_str(value)}'], INDENT)
            lines.append(f'{INDENT}{variant},')
        lines.append('}')
    elif isinstance(item, Struct):
        lines += doc
        lines.append(
            '#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]'
        )
        if item.closed:
            lines.append('#[serde(deny_unknown_fields)]')
        if not item.fields:
            lines.append(f'pub struct {item.name} {{}}')
        else:
            lines.append(f'pub struct {item.name} {{')
            for f in item.fields:
                lines += _doc(f.doc, INDENT)
                if f.attrs:
                    lines += _attr(f.attrs, INDENT)
                lines.append(f'{INDENT}pub {f.rust_name}: {f.rust_type},')
            lines.append('}')
    else:
        lines += _render_union(item, doc)
    return '\n'.join(lines) + '\n'


def _render_union(item: Union, doc: list[str]) -> list[str]:
    lines = list(doc)
    lines.append('#[derive(Debug, Clone, PartialEq, Serialize)]')
    if item.tag is None:
        lines[-1] = '#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]'
    lines.append('#[serde(untagged)]')
    lines.append(f'pub enum {item.name} {{')
    for variant, rust_type in item.variants:
        if rust_type is None:
            lines.append(f'{INDENT}{variant},')
        else:
            lines.append(f'{INDENT}{variant}({rust_type}),')
    lines.append('}')
    if item.tag is None:
        return lines
    tags = ', '.join(_rust_str(tag) for tag, _ in item.tags)
    tag_name = _rust_str(item.tag)
    i2, i3 = INDENT * 2, INDENT * 3
    lines += [
        '',
        f'impl {item.name} {{',
        f'{INDENT}/// The values of `{item.tag}` that select a variant.',
        f'{INDENT}pub const TAGS: &[&str] = &[{tags}];',
        '}',
        '',
        f"impl<'de> Deserialize<'de> for {item.name} {{",
        f'{INDENT}fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>',
        f'{INDENT}where',
        f"{i2}D: Deserializer<'de>,",
        f'{INDENT}{{',
        f'{i2}let (tag, value) = fields::tagged(deserializer, {tag_name})?;',
        f'{i2}match tag.as_str() {{',
    ]
    for tag, target in item.tags:
        lines.append(
            f'{i3}{_rust_str(tag)} => '
            f'fields::variant(value).map(Self::{target}),'
        )
    lines += [
        f'{i3}_ => Err(fields::unknown_tag({tag_name}, &tag, Self::TAGS)),',
        f'{i2}}}',
        f'{INDENT}}}',
        '}',
    ]
    return lines


def _rust_str(value: str) -> str:
    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
    return f'"{escaped}"'


if __name__ == '__main__':
    sys.exit(main())
