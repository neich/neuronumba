"""Tests for the Phase 5 UX-polish helpers and the configurable cache dir.

Covers:
  - get_cache_dir() honors NEURONUMBA_DSL_CACHE_DIR (Phase 0.3, deferred here)
  - get_source_file() returns a real, readable on-disk file for built models
  - cleanup_cache() removes old files but leaves in-use and recent files alone
  - The numba-error wrapping in get_numba_dfun is wired up (smoke check)
"""
import os
import time

import pytest

from neuronumba.simulator.models.dsl import (
    cleanup_cache, get_cache_dir, get_source_file,
)


def test_get_cache_dir_env_var_respected(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURONUMBA_DSL_CACHE_DIR", str(tmp_path))
    assert get_cache_dir() == str(tmp_path)


def test_get_cache_dir_default_when_unset(monkeypatch):
    monkeypatch.delenv("NEURONUMBA_DSL_CACHE_DIR", raising=False)
    cache = get_cache_dir()
    # Default lives under the system tempdir.
    import tempfile
    assert cache.startswith(tempfile.gettempdir())
    assert cache.endswith("nndsl_generated")


def test_get_source_file_returns_existing_path(hopf_dsl_cls):
    path = get_source_file(hopf_dsl_cls)
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    # Sanity: the dfun factory we expect for this spec must be present.
    assert "make_HopfDSL_dfun" in src


def test_get_source_file_rejects_non_dsl_class():
    from neuronumba.simulator.models import Hopf
    with pytest.raises(TypeError, match="DSL-built model"):
        get_source_file(Hopf)


def test_cleanup_cache_removes_old_only(monkeypatch, tmp_path):
    """cleanup_cache deletes stale files but spares recent ones."""
    monkeypatch.setenv("NEURONUMBA_DSL_CACHE_DIR", str(tmp_path))

    old = tmp_path / "old_module.py"
    fresh = tmp_path / "fresh_module.py"
    not_a_py = tmp_path / "stray.txt"
    old.write_text("# old")
    fresh.write_text("# fresh")
    not_a_py.write_text("ignored")

    # Backdate `old` by 30 days; leave `fresh` at now.
    old_time = time.time() - 30 * 86400
    os.utime(str(old), (old_time, old_time))

    removed = cleanup_cache(max_age_days=7.0)
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()
    assert not_a_py.exists()  # only .py files are touched


def test_numba_error_wrapping_smoke(hopf_dsl_cls, weights):
    """The error-wrapping path is not exercised in the happy case, but we
    verify the wrapper is in place by monkeypatching numba to raise."""
    import numba as nb

    m = hopf_dsl_cls(g=0.5).configure(weights=weights)

    # Monkeypatch nb.njit to raise the kind of error users might hit.
    # (We can't easily produce a real TypingError with our whitelisted
    # equations; this confirms the wrapper exists and reports the right
    # context.)
    real_njit = nb.njit
    def _broken_njit(*args, **kwargs):
        def _decorator(fn):
            raise nb.errors.TypingError("synthetic typing failure")
        return _decorator
    nb.njit = _broken_njit
    try:
        with pytest.raises(RuntimeError) as excinfo:
            m.get_numba_dfun()
    finally:
        nb.njit = real_njit

    msg = str(excinfo.value)
    assert "HopfDSL" in msg              # spec name surfaced
    assert "Generated source:" in msg    # path surfaced
    assert "synthetic typing failure" in msg  # original error preserved
