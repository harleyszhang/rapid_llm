"""Declare a test module's import-time needs in the module itself.

The pattern mirrors sglang's ``python/sglang/test/ci/ci_register.py``: the
helpers here do nothing at runtime, and the tool that needs the information
reads it out of the file's source. ``tests/conftest.py`` finds the
declarations by parsing each file with :mod:`ast` *before* importing it, so a
module whose import would crash on a machine without CUDA never gets imported
there. A pytest marker cannot stand in for this: markers are evaluated only
after the import has already run -- and failed.

Usage, next to the imports of a module whose module-scope imports reach the
Triton/CUDA runtime::

    from tests.registry import import_needs_cuda

    import_needs_cuda()
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Name of the declaration understood by :func:`declares` and ``conftest.py``.
IMPORT_NEEDS_CUDA = "import_needs_cuda"

__all__ = ["IMPORT_NEEDS_CUDA", "declares", "import_needs_cuda"]


def import_needs_cuda() -> None:
    """Declare that importing this module requires a CUDA machine.

    A no-op at runtime. ``tests/conftest.py`` reads it from the source and
    removes the module from collection on a machine without CUDA. Declare it
    when a *module-scope* import reaches something only a CUDA/Triton machine
    carries (``triton``, ``flashinfer``, a CUDA-only extension), or when
    import-time code evaluates device state.

    Tests that merely *run* on a CUDA device want the ``gpu`` marker instead --
    see :func:`tests.distributed.tp_harness.needs_gpus`, which applies it for
    TP payloads.
    """


def declares(path: Path, name: str) -> bool:
    """Whether the Python source at ``path`` calls the ``name`` declaration."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if called == name:
            return True
    return False
