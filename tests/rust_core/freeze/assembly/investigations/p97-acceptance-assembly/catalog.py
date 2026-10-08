# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map only byte-identical, module-identified legacy copies to original spans."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from support import require, sha


class VettedCatalog:
    def __init__(self, original, canonical: Path, copied: Path):
        self.original = original
        self.mapping = {}
        self.bindings = {}
        canonical = canonical.resolve()
        copied = copied.resolve()
        originals = {
            path.relative_to(canonical) for path in canonical.rglob('*.py')
        }
        copies = {path.relative_to(copied) for path in copied.rglob('*.py')}
        require(originals == copies and bool(originals), 'catalog.file_set')
        for relative in sorted(originals):
            source, target = canonical / relative, copied / relative
            require(
                not source.is_symlink() and not target.is_symlink(),
                'catalog.symlink',
            )
            raw = source.read_bytes()
            require(target.read_bytes() == raw, 'catalog.bytes')
            parts = relative.with_suffix('').parts
            module = '.'.join(('sdm_relational_engine', *parts))
            if module.endswith('.__init__'):
                module = module[:-9]
            self.mapping[str(target)] = (str(source), module)
            self.bindings[str(target)] = (source, sha(raw))
        self.copied_calls = 0

    def verify(self):
        for target, (source, expected) in self.bindings.items():
            require(
                sha(source.read_bytes()) == expected
                and sha(Path(target).read_bytes()) == expected,
                'catalog.changed',
            )

    def symbol(self, code):
        filename = str(Path(code.co_filename).resolve())
        mapped = self.mapping.get(filename)
        if mapped is None:
            return self.original.symbol_resolved(code, filename)
        canonical, name = mapped
        module = sys.modules.get(name)
        require(
            module is not None
            and getattr(module, '__spec__', None) is not None
            and str(Path(module.__file__).resolve()) == filename
            and str(Path(module.__spec__.origin).resolve()) == filename,
            'catalog.module_identity',
        )
        proxy = SimpleNamespace(
            co_filename=canonical,
            co_name=code.co_name,
            co_firstlineno=code.co_firstlineno,
        )
        symbol = self.original.symbol(proxy)
        if symbol:
            self.copied_calls += 1
        return symbol
