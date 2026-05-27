"""Simulation parameter form + matplotlib plot."""
from __future__ import annotations

import numpy as np

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)


class SimWidget(QWidget):
    run_requested = Signal()

    def __init__(self):
        super().__init__()

        self.n_rois = QSpinBox()
        self.n_rois.setRange(1, 1024)
        self.n_rois.setValue(4)

        self.dt = QDoubleSpinBox()
        self.dt.setDecimals(4)
        self.dt.setRange(1e-4, 100.0)
        self.dt.setValue(0.1)
        self.dt.setSuffix(" ms")

        self.tmax = QDoubleSpinBox()
        self.tmax.setDecimals(1)
        self.tmax.setRange(1.0, 1e7)
        self.tmax.setValue(5000.0)
        self.tmax.setSuffix(" ms")

        self.twarmup = QDoubleSpinBox()
        self.twarmup.setDecimals(1)
        self.twarmup.setRange(0.0, 1e7)
        self.twarmup.setValue(1000.0)
        self.twarmup.setSuffix(" ms")

        self.sampling = QDoubleSpinBox()
        self.sampling.setDecimals(2)
        self.sampling.setRange(0.01, 1000.0)
        self.sampling.setValue(1.0)
        self.sampling.setSuffix(" ms")

        self.integrator = QComboBox()
        self.integrator.addItems(["EulerDeterministic", "EulerStochastic"])
        self.integrator.setCurrentIndex(1)

        self.g = QDoubleSpinBox()
        self.g.setDecimals(4)
        self.g.setRange(0.0, 100.0)
        self.g.setValue(0.0)

        self.observe = QComboBox()

        self.run_btn = QPushButton("Run simulation")
        self.run_btn.clicked.connect(self.run_requested.emit)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self._weights: np.ndarray | None = None
        self.weights_label = QLabel("(random)")
        load_btn = QPushButton("Load…")
        load_btn.clicked.connect(self._on_load_weights)
        rnd_btn = QPushButton("Random")
        rnd_btn.clicked.connect(self._on_random_weights)
        w_row = QHBoxLayout()
        w_row.addWidget(self.weights_label, 1)
        w_row.addWidget(load_btn)
        w_row.addWidget(rnd_btn)
        w_wrapper = QWidget()
        w_wrapper.setLayout(w_row)

        form_box = QGroupBox("Simulation parameters")
        form = QFormLayout(form_box)
        form.addRow("Brain regions (n_rois):", self.n_rois)
        form.addRow("Integration step dt:", self.dt)
        form.addRow("Total simulation time:", self.tmax)
        form.addRow("Warm-up time (discarded):", self.twarmup)
        form.addRow("Sampling period:", self.sampling)
        form.addRow("Integrator:", self.integrator)
        form.addRow("Global coupling g:", self.g)
        form.addRow("Connectivity weights:", w_wrapper)
        form.addRow("Variable to plot:", self.observe)

        run_row = QHBoxLayout()
        run_row.addWidget(self.run_btn)
        run_row.addStretch()

        self.figure = Figure(figsize=(6, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._ax = None

        self.ylim_auto = QCheckBox("Auto")
        self.ylim_auto.setChecked(True)
        self.ylim_min = QDoubleSpinBox()
        self.ylim_max = QDoubleSpinBox()
        for s in (self.ylim_min, self.ylim_max):
            s.setDecimals(4)
            s.setRange(-1e9, 1e9)
            s.setSingleStep(0.1)
            s.setKeyboardTracking(False)
            s.setEnabled(False)
        self.ylim_auto.toggled.connect(self._on_ylim_auto)
        self.ylim_min.valueChanged.connect(self._apply_ylim)
        self.ylim_max.valueChanged.connect(self._apply_ylim)

        ylim_row = QHBoxLayout()
        ylim_row.addWidget(QLabel("Y axis:"))
        ylim_row.addWidget(self.ylim_auto)
        ylim_row.addWidget(QLabel("min"))
        ylim_row.addWidget(self.ylim_min)
        ylim_row.addWidget(QLabel("max"))
        ylim_row.addWidget(self.ylim_max)
        ylim_row.addStretch()

        plot_box = QGroupBox("Time series")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addLayout(ylim_row)
        plot_layout.addWidget(self.canvas)

        left = QVBoxLayout()
        left.addWidget(form_box)
        left.addLayout(run_row)
        left.addWidget(self.status)
        left.addStretch()
        left_w = QWidget()
        left_w.setLayout(left)
        left_w.setMaximumWidth(420)

        layout = QHBoxLayout(self)
        layout.addWidget(left_w)
        layout.addWidget(plot_box, 1)

    def _on_load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load connectivity weights", "",
            "Data files (*.csv *.tsv *.mat *.npy *.npz);;All files (*)")
        if not path:
            return
        from neuronumba.tools.loader import load_2d_matrix
        try:
            w = load_2d_matrix(path)
        except Exception as e:
            self.set_status(f"Failed to load: {e}", ok=False)
            return
        if w.ndim != 2 or w.shape[0] != w.shape[1]:
            self.set_status(f"Expected square matrix, got shape {w.shape}", ok=False)
            return
        self._weights = w.astype(np.float64, copy=False)
        name = path.rsplit("/", 1)[-1]
        self.weights_label.setText(f"{name} ({w.shape[0]}x{w.shape[0]})")
        self.n_rois.setValue(w.shape[0])
        self.set_status(f"Loaded weights from {name}.", ok=True)

    def _on_random_weights(self):
        self._weights = None
        self.weights_label.setText("(random)")

    def get_weights(self) -> np.ndarray:
        if self._weights is not None:
            return self._weights
        n = self.n_rois.value()
        rng = np.random.default_rng(42)
        w = rng.uniform(0.0, 1.0, size=(n, n))
        np.fill_diagonal(w, 0.0)
        return w

    def set_observable_choices(self, names: list[str]):
        current = self.observe.currentText()
        self.observe.clear()
        self.observe.addItems(names)
        if current in names:
            self.observe.setCurrentText(current)

    def set_status(self, msg: str, ok: bool):
        color = "#22863a" if ok else "#b31d28"
        self.status.setText(f"<span style='color:{color}'>{msg}</span>")

    def plot(self, data: np.ndarray, sampling_ms: float, title: str):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        t = np.arange(data.shape[0]) * sampling_ms / 1000.0
        ax.plot(t, data, lw=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        self._ax = ax
        if self.ylim_auto.isChecked():
            self._sync_ylim_spins_from_axes()
        else:
            self._apply_ylim()
        self.canvas.draw_idle()

    def _sync_ylim_spins_from_axes(self) -> None:
        if self._ax is None:
            return
        lo, hi = self._ax.get_ylim()
        span = max(abs(hi - lo), 1.0)
        step = span / 20.0
        for s, v in ((self.ylim_min, lo), (self.ylim_max, hi)):
            s.blockSignals(True)
            s.setSingleStep(step)
            s.setValue(v)
            s.blockSignals(False)

    def _on_ylim_auto(self, checked: bool) -> None:
        self.ylim_min.setEnabled(not checked)
        self.ylim_max.setEnabled(not checked)
        if self._ax is None:
            return
        if checked:
            self._ax.relim()
            self._ax.autoscale(axis="y")
            self._sync_ylim_spins_from_axes()
        else:
            self._apply_ylim()
        self.canvas.draw_idle()

    def _apply_ylim(self) -> None:
        if self._ax is None or self.ylim_auto.isChecked():
            return
        lo = self.ylim_min.value()
        hi = self.ylim_max.value()
        if hi <= lo:
            hi = lo + 1e-9
        self._ax.set_ylim(lo, hi)
        self.canvas.draw_idle()
