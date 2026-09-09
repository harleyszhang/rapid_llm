"""Environment variable lookup with legacy-name compatibility.

The project was renamed from ``lite_llama`` to ``rapid_llm``; every
``RAPID_LLM_*`` variable used to be spelled ``LITE_LLAMA_*``. Reads prefer
the new name and fall back to the legacy one, warning once per legacy
variable so existing deployments keep working while scripts migrate.
"""

from __future__ import annotations

import os
import warnings

_LEGACY_PREFIX = "LITE_LLAMA_"
_NEW_PREFIX = "RAPID_LLM_"

#: Legacy variables already warned about in this process (warn once each).
_warned: set[str] = set()


def getenv(name: str, default: str | None = None) -> str | None:
    """Read *name* from the environment, falling back to its legacy spelling.

    ``RAPID_LLM_FOO`` wins; when unset, the legacy ``LITE_LLAMA_FOO`` is
    honoured and a :class:`DeprecationWarning` is emitted once per process.
    """
    value = os.environ.get(name)
    if value is not None:
        return value
    if name.startswith(_NEW_PREFIX):
        legacy = _LEGACY_PREFIX + name[len(_NEW_PREFIX) :]
        value = os.environ.get(legacy)
        if value is not None and legacy not in _warned:
            _warned.add(legacy)
            warnings.warn(
                f"Environment variable {legacy} is deprecated and will be removed; "
                f"rename it to {name}.",
                DeprecationWarning,
                stacklevel=2,
            )
    return value if value is not None else default


def env_flag(name: str, *, default: bool = False) -> bool:
    """Read *name* as an opt-in switch: anything but ``0``/``false``/``off`` is on.

    The spelling every optional-feature switch shares (the overlap policies, the
    sequence-parallel pass), in one place so the accepted words cannot drift
    between them. An empty value reads as *off* — an exported-but-blank variable
    is how a shell script spells "not set".

    Args:
        name: Variable name, resolved through :func:`getenv` (legacy spelling
            still honoured).
        default: Value used when the variable is absent.
    """
    raw = (getenv(name, "1" if default else "0") or "").strip().lower()
    return raw not in ("", "0", "false", "off")
