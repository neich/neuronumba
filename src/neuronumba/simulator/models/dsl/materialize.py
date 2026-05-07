"""
Materialize generated source to a real on-disk module.

Numba's `cache=True` requires a real file path (the cache key is the source
file path). Synthesized filenames via `linecache` do not work. Writing to a
real `.py` file under `tempfile.gettempdir()` (or a user-specified directory)
and importing it as a module gives us:
  - working numba cache,
  - real tracebacks pointing into the file,
  - debuggable code you can inspect with `inspect.getsource`.

Same approach SymPy's `lambdify` and TVB's RateML take.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import time
from typing import Any, Dict


_GENERATED_MODULES: Dict[str, Any] = {}  # cache by source-hash


def get_cache_dir() -> str:
    """Return the directory where generated module files are written.

    Defaults to ``$TMPDIR/nndsl_generated/``. Override with the
    ``NEURONUMBA_DSL_CACHE_DIR`` environment variable — useful on shared
    compute clusters where ``$TMPDIR`` is per-job and gets cleaned, defeating
    numba's compile cache between runs.

    Resolved fresh on each call so tests (and users who change the env var
    mid-session) see the current value.
    """
    return os.environ.get(
        "NEURONUMBA_DSL_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "nndsl_generated"),
    )


def cleanup_cache(max_age_days: float = 7.0) -> int:
    """Delete generated ``.py`` files older than ``max_age_days``.

    Useful when iterating in Jupyter: every spec change creates a new module
    file, and stale ones accumulate. Returns the number of files removed.

    Files currently held in the in-process import cache are NOT removed even
    if they're old, because deleting them out from under Python could break
    further use of already-built model classes.
    """
    cache_dir = get_cache_dir()
    if not os.path.isdir(cache_dir):
        return 0

    cutoff = time.time() - max_age_days * 86400.0
    in_use = {getattr(mod, "__file__", None) for mod in _GENERATED_MODULES.values()}
    removed = 0
    for entry in os.listdir(cache_dir):
        if not entry.endswith(".py"):
            continue
        path = os.path.join(cache_dir, entry)
        if path in in_use:
            continue
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            # File may have been removed by another process between listdir
            # and stat — fine to ignore.
            pass
    return removed


def _materialize_module(spec_name: str, src: str) -> Any:
    """Write `src` (a complete Python file) to disk and import it as a module.
    Returns the imported module. Same source -> cached module reuse."""
    key = hashlib.md5(src.encode("utf-8")).hexdigest()[:12]
    if key in _GENERATED_MODULES:
        return _GENERATED_MODULES[key]

    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    mod_name = f"nndsl_gen_{spec_name}_{key}"
    fname = os.path.join(cache_dir, f"{mod_name}.py")
    if not os.path.exists(fname):
        with open(fname, "w", encoding="utf-8") as f:
            f.write(src)

    spec = importlib.util.spec_from_file_location(mod_name, fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _GENERATED_MODULES[key] = mod
    return mod
