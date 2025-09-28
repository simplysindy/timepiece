"""Compatibility helpers for numpy module layout changes."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np


def ensure_numpy_core_alias() -> None:
    """Expose numpy.core as numpy._core for legacy pickled models."""
    if getattr(np, "_core", None) is not None and "numpy._core" in sys.modules:
        return

    try:
        core_module: ModuleType = importlib.import_module("numpy.core")
    except ImportError:
        return

    # Register alias so ``import numpy._core`` resolves to ``numpy.core``
    sys.modules.setdefault("numpy._core", core_module)
    setattr(np, "_core", core_module)

    # Ensure common submodules are discoverable via the alias
    for submodule_name in list(sys.modules):
        if not submodule_name.startswith("numpy.core"):
            continue
        alias = submodule_name.replace("numpy.core", "numpy._core", 1)
        sys.modules.setdefault(alias, sys.modules[submodule_name])
