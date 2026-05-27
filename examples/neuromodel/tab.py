"""One tab = one neuronumba model under construction."""
from __future__ import annotations

import time
import traceback

import numpy as np

from PySide6.QtWidgets import (
    QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QMessageBox, QPushButton,
    QTabWidget, QVBoxLayout, QWidget,
)

from builder import generate_source, load_model_class, validate_equations
from dependants_widget import DependantsWidget
from equations_widget import EquationsWidget
from model_def import ModelDef
from params_widget import ParamsWidget
from sim_widget import SimWidget
from vars_widget import VarsWidget


class ModelTab(QWidget):
    def __init__(self, md: ModelDef):
        super().__init__()

        self.name_edit = QLineEdit(md.name)
        meta_form = QFormLayout()
        meta_form.addRow("Model name:", self.name_edit)

        save_btn = QPushButton("Save JSON…")
        save_btn.clicked.connect(self._on_save_json)
        load_btn = QPushButton("Load JSON…")
        load_btn.clicked.connect(self._on_load_json)
        export_btn = QPushButton("Export Python…")
        export_btn.clicked.connect(self._on_export_py)

        meta_btns = QHBoxLayout()
        meta_btns.addWidget(save_btn)
        meta_btns.addWidget(load_btn)
        meta_btns.addWidget(export_btn)
        meta_btns.addStretch()

        meta_widget = QWidget()
        meta_layout = QVBoxLayout(meta_widget)
        meta_layout.addLayout(meta_form)
        meta_layout.addLayout(meta_btns)
        meta_layout.addStretch()

        self.params_widget = ParamsWidget(md.params)
        self.vars_widget = VarsWidget()
        self.vars_widget.set_state_vars(md.state_vars)
        self.vars_widget.set_observable_vars(md.observable_vars)

        self.dependants_widget = DependantsWidget(md.dependants)

        self.eq_widget = EquationsWidget(md.equations)
        self.eq_widget.validate_requested.connect(self._on_validate)

        self.sim_widget = SimWidget()
        self.sim_widget.run_requested.connect(self._on_run)
        self.sim_widget.observe.currentTextChanged.connect(self._on_observe_changed)

        self._sim_results: dict[str, np.ndarray] = {}
        self._sim_sampling: float = 0.0

        inner = QTabWidget()
        inner.addTab(meta_widget, "Model")
        inner.addTab(self.vars_widget, "Variables")
        inner.addTab(self.params_widget, "Parameters")
        inner.addTab(self.dependants_widget, "Dependants")
        inner.addTab(self.eq_widget, "Equations")
        inner.addTab(self.sim_widget, "Simulation")
        inner.currentChanged.connect(lambda _: self._refresh_observe_choices())

        layout = QVBoxLayout(self)
        layout.addWidget(inner)

        self._refresh_observe_choices()

    def current_def(self) -> ModelDef:
        return ModelDef(
            name=self.name_edit.text().strip() or "MyModel",
            state_vars=self.vars_widget.get_state_vars(),
            observable_vars=self.vars_widget.get_observable_vars(),
            params=self.params_widget.get_params(),
            dependants=self.dependants_widget.get_dependants(),
            equations=self.eq_widget.text(),
        )

    def _refresh_observe_choices(self):
        md = self.current_def()
        names = [v.name for v in md.state_vars] + [o.name for o in md.observable_vars]
        self.sim_widget.set_observable_choices(names)

    def _on_save_json(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model JSON", f"{self.name_edit.text()}.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w") as f:
            f.write(self.current_def().to_json())
        self.eq_widget.set_status(f"Saved to {path}", ok=True)

    def _on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load model JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                md = ModelDef.from_json(f.read())
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self.name_edit.setText(md.name)
        self.vars_widget.set_state_vars(md.state_vars)
        self.vars_widget.set_observable_vars(md.observable_vars)
        self.params_widget.set_params(md.params)
        self.dependants_widget.set_dependants(md.dependants)
        self.eq_widget.set_text(md.equations)
        self._refresh_observe_choices()

    def _on_export_py(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Python module", f"{self.name_edit.text()}.py", "Python (*.py)")
        if not path:
            return
        with open(path, "w") as f:
            f.write(generate_source(self.current_def()))
        self.eq_widget.set_status(f"Exported to {path}", ok=True)

    def _on_validate(self):
        md = self.current_def()
        ok, msg = validate_equations(md)
        self.eq_widget.set_status(msg, ok=ok)

    def _on_run(self):
        try:
            self._run_simulation()
        except Exception as e:
            traceback.print_exc()
            self.sim_widget.set_status(f"{type(e).__name__}: {e}", ok=False)

    def _run_simulation(self):
        md = self.current_def()
        ok, msg = validate_equations(md)
        if not ok:
            self.sim_widget.set_status(f"Validate failed: {msg}", ok=False)
            return

        from neuronumba.simulator.connectivity import Connectivity
        from neuronumba.simulator.history import HistoryNoDelays
        from neuronumba.simulator.integrators import EulerDeterministic, EulerStochastic
        from neuronumba.simulator.monitors import RawSubSample
        from neuronumba.simulator.simulator import Simulator

        cls = load_model_class(md)

        weights = self.sim_widget.get_weights()
        n_rois = weights.shape[0]
        if n_rois != self.sim_widget.n_rois.value():
            self.sim_widget.n_rois.setValue(n_rois)

        dt = self.sim_widget.dt.value()
        tmax = self.sim_widget.tmax.value()
        twarmup = self.sim_widget.twarmup.value()
        sampling = self.sim_widget.sampling.value()
        g = self.sim_widget.g.value()
        integrator_name = self.sim_widget.integrator.currentText()
        obs_var = self.sim_widget.observe.currentText()
        if not obs_var:
            self.sim_widget.set_status("Pick a variable to plot first.", ok=False)
            return

        model = cls(weights=weights, g=g)

        if integrator_name == "EulerDeterministic":
            integrator = EulerDeterministic(dt=dt)
        else:
            sigmas = np.array([v.sigma for v in md.state_vars], dtype=np.float64)
            integrator = EulerStochastic(dt=dt, sigmas=sigmas)

        rng = np.random.default_rng(0)
        lengths = rng.uniform(1.0, 11.0, size=(n_rois, n_rois))
        con = Connectivity(weights=weights, lengths=lengths, speed=1.0)
        history = HistoryNoDelays()
        all_var_names = (
            [v.name for v in md.state_vars]
            + [o.name for o in md.observable_vars]
        )
        monitor = RawSubSample(period=sampling, monitor_vars=model.get_var_info(all_var_names))
        sim = Simulator(connectivity=con, model=model, history=history,
                       integrator=integrator, monitors=[monitor])

        self.sim_widget.set_status(
            f"Running {tmax + twarmup:.0f} ms on n_rois={n_rois}…", ok=True)
        self.sim_widget.run_btn.setEnabled(False)
        self.sim_widget.run_btn.repaint()
        try:
            t0 = time.perf_counter()
            sim.run(0, twarmup + tmax)
            elapsed = time.perf_counter() - t0
        finally:
            self.sim_widget.run_btn.setEnabled(True)

        self._sim_results = {}
        self._sim_sampling = sampling
        for name in all_var_names:
            data = monitor.data(name)
            # RawSubSample over-allocates its buffer by one row; when n_steps
            # is a multiple of the sampling interval (the typical case) the
            # trailing row is never written and stays at zeros.
            if data.shape[0] >= 2:
                data = data[:-1]
            from_idx = int(data.shape[0] * twarmup / (twarmup + tmax))
            self._sim_results[name] = data[from_idx:, :].copy()

        self._plot_current_var()
        shape = self._sim_results[obs_var].shape
        self.sim_widget.set_status(
            f"Done: data {shape} in {elapsed:.2f}s. "
            "Switch the dropdown to replot any variable.", ok=True)

    def _on_observe_changed(self, _name: str) -> None:
        self._plot_current_var()

    def _plot_current_var(self) -> None:
        name = self.sim_widget.observe.currentText()
        if not name or name not in self._sim_results:
            return
        self.sim_widget.plot(self._sim_results[name], self._sim_sampling, name)
