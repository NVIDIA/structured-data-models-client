# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta
from typing import Any, ClassVar


class Singleton(ABCMeta):
    r"""A per-process singleton definition."""

    # Deliberately shared across every class using this metaclass: the
    # registry of live singletons is the state the metaclass exists for.
    _instances: ClassVar[dict[type, Any]] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            # Calls the `__init__` method of the subclass and returns a
            # reference, which is stored to prevent multiple instantiations.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            return instance
        return cls._instances[cls]

    def clear(cls) -> None:
        r"""Clears the singleton class instance, so the next construction
        will re-initialize the clas.
        """
        try:
            del Singleton._instances[cls]
        except KeyError:
            pass
