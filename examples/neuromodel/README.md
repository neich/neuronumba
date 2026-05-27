# neuromodel — a Qt app for building neuronumba models

A small PySide6 application for prototyping `neuronumba` `Model` subclasses
without hand-editing source files. Each tab is one model under construction;
within a tab there are sub-tabs for the model's name, its variables, its
parameters, the equations and a simulation panel.

## Run

From the repository root, with `neuronumba` installed:

```bash
pip install -r examples/neuromodel/requirements.txt
python examples/neuromodel/main.py
```

The first run JIT-compiles the generated model — expect a few seconds of delay
on the first **Run simulation** click. Subsequent runs reuse the numba cache.

## Workflow

1. **Pick a template** in the toolbar (`Blank`, `Deco2014`, `Hopf`,
   `Naskar2021`) and press *Add tab*, or *Open JSON…* to reload a previously
   saved spec.
2. **Model tab** — set the model name; save the spec to JSON / load JSON /
   export the generated Python module.
3. **Variables tab** — add or remove state variables (each row: name, initial
   value, low/high bounds, noise sigma, "Coupled" flag) and observable
   variables. Bounds use `inf` / `-inf` for unbounded.
4. **Parameters tab** — add or remove parameters (name, default, tag, doc).
   * Tag `regional` exposes the parameter as a per-ROI scalar inside the numba
     dfun.
   * Tag `plain` keeps the parameter as a normal Python attribute (not
     accessible inside the equations).
5. **Equations tab** — edit the body of the dfun in plain Python. Available
   identifiers:
     - state variables (`S_e`, `S_i`, …)
     - coupling inputs (`cpl_S_e`, …)
     - regional parameters by name
     - `np` (NumPy)
   You must define `d<name>` for every state variable and assign every
   observable name. Click **Validate equations** to run a pure-Python smoke
   test on a 2-region stub — errors are reported with Python-level messages,
   so this is much friendlier than waiting for numba to fail.
6. **Simulation tab** — choose the number of regions (or load a connectivity
   matrix), `dt`, total simulation time, warm-up, sampling period, integrator,
   global coupling `g`, and the variable to plot, then press **Run
   simulation**.

## How the generated code looks

`Validate equations` is pure Python and writes nothing to disk.
`Run simulation` (and *Export Python…*) write a real Python module that
subclasses `LinearCouplingModel`:

```python
class Deco2014(LinearCouplingModel):
    _state_var_names = ['S_e', 'S_i']
    _coupling_var_names = ['S_e']
    _observable_var_names = ['Ie', 're']
    _state_var_bounds = {'S_e': (0.0, 1.0), 'S_i': (0.0, 1.0)}

    tau_e = Attr(default=100.0, attributes=Model.Tag.REGIONAL, doc='…')
    # … one Attr per parameter …

    def initial_state(self, n_rois):
        state = np.empty((self.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        return state

    def get_noise_template(self):
        return np.array([0.001, 0.001])

    def get_numba_dfun(self):
        m = self.m.copy(); P = self.P

        @nb.njit(nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
                 cache=NUMBA_CACHE, fastmath=NUMBA_FASTMATH, nogil=NUMBA_NOGIL)
        def Deco2014_dfun(state, coupling):
            S_e = state[0, :]
            S_i = state[1, :]
            cpl_S_e = coupling[0, :]
            tau_e = m[np.intp(P.tau_e)]
            # … etc …

            # equations from the editor go here, indented to match …
            dS_e = -S_e / tau_e + gamma_e * (1.0 - S_e) * re
            dS_i = -S_i / tau_i + gamma_i * ri

            return np.stack((dS_e, dS_i)), np.stack((Ie, re))

        return Deco2014_dfun
```

`Run simulation` writes the file under
`examples/neuromodel/_generated/_gen_<Name>_<hash>.py`, imports it via
`importlib.util.spec_from_file_location`, and runs the standard neuronumba
`Simulator`. Numba `cache`, `fastmath` and `nogil` flags are honoured (see
`neuronumba.numba_tools.config`). The directory is excluded from git via
`.gitignore` and can be deleted at any time.

## Limitations of v1

- All generated models subclass `LinearCouplingModel` (linear coupling
  `g * W.T @ state`). Models with non-linear coupling (the real Hopf, for
  example) can be approximated; for an exact reproduction, export the Python
  file and override `get_numba_coupling()`.
- Parameter values are single floats that broadcast across regions. For
  per-ROI heterogeneity, load the JSON and set per-region arrays
  programmatically, or edit the exported `.py`.
- Connectivity tract lengths are random — the simulation uses
  `HistoryNoDelays`, so axonal delays are not modelled in this app.
- Only one observable variable can be plotted at a time (the underlying
  simulator currently supports a single monitor).
