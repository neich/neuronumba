"""Generate runtime neuronumba Model subclasses from a ModelDef.

Two entry points:

- ``validate_equations(md)``: pure-Python smoke test (no numba, no neuronumba).
  Runs the equation body on tiny stub arrays and checks that every required
  ``d<state>`` and observable name was assigned with the right shape.
- ``load_model_class(md)``: writes a real ``.py`` under ``_generated/`` and
  imports it, returning the generated ``Model`` subclass. The file path is
  content-hashed so numba's cache works normally.
"""
from __future__ import annotations

import hashlib
import importlib.util
import keyword
import math
import textwrap
from pathlib import Path

import numpy as np

from model_def import ModelDef


_CACHE_DIR = Path(__file__).parent / "_generated"
_INDENT_BODY = " " * 12   # def fn(...) inside def get_numba_dfun() inside class

_RESERVED_NAMES = frozenset({
    # Set by Model / LinearCouplingModel / generated code:
    "weights", "weights_t", "n_rois", "g", "m", "m_aux", "P", "P_aux",
    # Used as identifiers inside the generated body:
    "np", "nb", "self", "state", "coupling",
})


def _is_valid_identifier(s: str) -> bool:
    return s.isidentifier() and not keyword.iskeyword(s)


def _fmt_float(v: float) -> str:
    if math.isinf(v):
        return "math.inf" if v > 0 else "-math.inf"
    return repr(float(v))


def _sanitize_class_name(name: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in name) or "MyModel"
    if s[0].isdigit():
        s = "M_" + s
    return s


def _check_names(md: ModelDef) -> str | None:
    state_names = [v.name for v in md.state_vars]
    obs_names = [o.name for o in md.observable_vars]
    param_names = [p.name for p in md.params]
    dep_names = [d.name for d in md.dependants]
    all_names = state_names + obs_names + param_names + dep_names

    bad = [n for n in all_names if not _is_valid_identifier(n)]
    if bad:
        return f"Invalid Python identifier(s): {', '.join(bad)}"

    clashes = [n for n in all_names if n in _RESERVED_NAMES]
    if clashes:
        return (
            f"Reserved name(s) used: {', '.join(sorted(set(clashes)))} — "
            "these are set by neuronumba or the generated code."
        )

    seen: set[str] = set()
    for n in all_names:
        if n in seen:
            return f"Duplicate name across state/observable/param/dependant: {n!r}"
        seen.add(n)

    if not state_names:
        return "At least one state variable is required."
    return None


def validate_equations(md: ModelDef, n_rois: int = 2) -> tuple[bool, str]:
    """Run the equation body as plain Python on stub arrays. No numba."""
    err = _check_names(md)
    if err:
        return False, err

    namespace: dict = {"np": np}

    rng = np.random.default_rng(0)
    stub_weights = rng.uniform(0.0, 1.0, size=(n_rois, n_rois))
    np.fill_diagonal(stub_weights, 0.0)
    namespace["weights"] = stub_weights
    namespace["n_rois"] = n_rois
    namespace["g"] = 1.0

    for p in md.params:
        namespace[p.name] = (np.full(n_rois, p.default)
                             if p.tag == "regional" else p.default)

    for dep in md.dependants:
        try:
            dep_code = compile(dep.formula or "pass", f"<dependant:{dep.name}>", "exec")
        except SyntaxError as e:
            return False, f"Dependant {dep.name!r} syntax error on line {e.lineno}: {e.msg}"
        try:
            exec(dep_code, namespace)
        except Exception as e:
            return False, f"Dependant {dep.name!r}: {type(e).__name__}: {e}"
        if dep.name not in namespace:
            return False, (
                f"Dependant {dep.name!r}: formula did not assign to {dep.name!r}."
            )

    for v in md.state_vars:
        namespace[v.name] = np.full(n_rois, v.initial)
    for name in md.coupling_var_names:
        namespace[f"cpl_{name}"] = np.zeros(n_rois)

    try:
        code = compile(md.equations or "pass", "<equations>", "exec")
    except SyntaxError as e:
        return False, f"Syntax error on line {e.lineno}: {e.msg}"

    try:
        exec(code, namespace)
    except Exception as e:  # the user's code can raise anything
        return False, f"{type(e).__name__}: {e}"

    missing_d = [f"d{v.name}" for v in md.state_vars if f"d{v.name}" not in namespace]
    if missing_d:
        return False, f"Missing derivative assignment(s): {', '.join(missing_d)}"

    missing_obs = [o.name for o in md.observable_vars if o.name not in namespace]
    if missing_obs:
        return False, f"Missing observable assignment(s): {', '.join(missing_obs)}"

    for v in md.state_vars:
        arr = np.asarray(namespace[f"d{v.name}"], dtype=float)
        if arr.shape != (n_rois,):
            return False, (
                f"d{v.name} has shape {arr.shape}, expected ({n_rois},) — "
                "make sure the expression is a per-region 1-D array."
            )

    return True, (
        f"OK — {len(md.dependants)} dependant(s), "
        f"{len(md.state_vars)} state derivative(s) and "
        f"{len(md.observable_vars)} observable(s) computed correctly."
    )


def _list_repr(items: list[str]) -> str:
    return "[" + ", ".join(f"'{i}'" for i in items) + "]"


def _bounds_src(md: ModelDef) -> str:
    bounds = {v.name: (v.lo, v.hi) for v in md.state_vars if v.has_bounds}
    if not bounds:
        return "{}"
    parts = ", ".join(
        f"'{n}': ({_fmt_float(lo)}, {_fmt_float(hi)})" for n, (lo, hi) in bounds.items()
    )
    return "{" + parts + "}"


def _attr_lines(md: ModelDef) -> list[str]:
    lines: list[str] = []
    for p in md.params:
        tag = {
            "regional": "Model.Tag.REGIONAL",
            "plain": "Model.Tag.UNKNOWN",
        }.get(p.tag, "Model.Tag.UNKNOWN")
        doc = repr(p.doc) if p.doc else "''"
        lines.append(
            f"    {p.name} = Attr(default={_fmt_float(p.default)}, "
            f"attributes={tag}, doc={doc})"
        )
    for d in md.dependants:
        doc = repr(d.doc) if d.doc else "''"
        lines.append(f"    {d.name} = Attr(dependant=True, doc={doc})")
    return lines


def _init_dependant_method(md: ModelDef) -> str:
    """Return the source for ``_init_dependant`` or an empty string."""
    if not md.dependants:
        return ""
    lines = [
        "    def _init_dependant(self):",
        "        super()._init_dependant()",
        "        weights = self.weights",
        "        n_rois = self.n_rois",
        "        g = self.g",
    ]
    for p in md.params:
        lines.append(f"        {p.name} = self.{p.name}")
    for dep in md.dependants:
        body = textwrap.dedent(dep.formula or "").strip()
        if body:
            lines.append(textwrap.indent(body, "        "))
        lines.append(f"        self.{dep.name} = {dep.name}")
    return "\n".join(lines) + "\n"


def _dependant_capture_lines(md: ModelDef) -> list[str]:
    return [f"        {d.name} = self.{d.name}" for d in md.dependants]


def _unpack_lines(md: ModelDef) -> list[str]:
    lines: list[str] = []
    for i, v in enumerate(md.state_vars):
        lines.append(f"{_INDENT_BODY}{v.name} = state[{i}, :]")
    for i, name in enumerate(md.coupling_var_names):
        lines.append(f"{_INDENT_BODY}cpl_{name} = coupling[{i}, :]")
    for p in md.params:
        if p.tag == "regional":
            lines.append(f"{_INDENT_BODY}{p.name} = m[np.intp(P.{p.name})]")
    return lines


def _equations_src(md: ModelDef) -> str:
    body = textwrap.dedent(md.equations).strip()
    if not body:
        return f"{_INDENT_BODY}pass"
    return textwrap.indent(body, _INDENT_BODY)


def _return_src(md: ModelDef) -> str:
    n_state = len(md.state_vars)
    d_names = ", ".join(f"d{v.name}" for v in md.state_vars)
    state_expr = f"np.stack(({d_names},))" if n_state == 1 else f"np.stack(({d_names}))"

    obs_count = len(md.observable_vars)
    obs_names = ", ".join(o.name for o in md.observable_vars)
    if obs_count == 0:
        obs_expr = "np.empty((1, 1))"
    elif obs_count == 1:
        obs_expr = f"np.stack(({obs_names},))"
    else:
        obs_expr = f"np.stack(({obs_names}))"

    return f"{_INDENT_BODY}return {state_expr}, {obs_expr}"


def _initial_state_src(md: ModelDef) -> str:
    lines = ["        state = np.empty((self.n_state_vars, n_rois))"]
    for i, v in enumerate(md.state_vars):
        lines.append(f"        state[{i}] = {_fmt_float(v.initial)}")
    lines.append("        return state")
    return "\n".join(lines)


def _noise_template_src(md: ModelDef) -> str:
    if not md.state_vars:
        return "        return np.array([0.0])"
    sigmas = ", ".join(_fmt_float(v.sigma) for v in md.state_vars)
    return f"        return np.array([{sigmas}])"


def generate_source(md: ModelDef) -> str:
    cls_name = _sanitize_class_name(md.name)
    state_names = [v.name for v in md.state_vars]
    obs_names = [o.name for o in md.observable_vars]
    coupling_names = md.coupling_var_names

    attr_block = "\n".join(_attr_lines(md)) or "    pass"
    init_dep_block = _init_dependant_method(md)
    unpack_block = "\n".join(_unpack_lines(md))
    eq_block = _equations_src(md)
    return_block = _return_src(md)
    dep_capture_block = "\n".join(_dependant_capture_lines(md))

    return f'''"""Generated by examples/neuromodel — do not edit by hand."""
from __future__ import annotations

import math

import numba as nb
import numpy as np

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL
from neuronumba.simulator.models import LinearCouplingModel, Model


class {cls_name}(LinearCouplingModel):
    _state_var_names = {_list_repr(state_names)}
    _coupling_var_names = {_list_repr(coupling_names)}
    _observable_var_names = {_list_repr(obs_names)}
    _state_var_bounds = {_bounds_src(md)}

{attr_block}

{init_dep_block}
    def initial_state(self, n_rois):
{_initial_state_src(md)}

    def get_noise_template(self):
{_noise_template_src(md)}

    def get_numba_dfun(self):
        m = self.m.copy()
        P = self.P
{dep_capture_block}

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def {cls_name}_dfun(state, coupling):
{unpack_block}

{eq_block}

{return_block}

        return {cls_name}_dfun
'''


def write_module(md: ModelDef) -> tuple[Path, str]:
    """Write the generated source to disk and return (path, class_name)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    init = _CACHE_DIR / "__init__.py"
    if not init.exists():
        init.write_text("")
    src = generate_source(md)
    digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:10]
    cls_name = _sanitize_class_name(md.name)
    path = _CACHE_DIR / f"_gen_{cls_name}_{digest}.py"
    if not path.exists():
        path.write_text(src)
    return path, cls_name


def load_model_class(md: ModelDef):
    path, cls_name = write_module(md)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls_name)
