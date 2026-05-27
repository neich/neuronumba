"""State + observable variable editor."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from model_def import ObservableVarDef, StateVarDef


STATE_COLUMNS = ("Name", "Initial", "Lo", "Hi", "Noise σ", "Coupled")
OBS_COLUMNS = ("Name", "Doc")

_NOISE_COL = 4
_NOISE_TOOLTIP = (
    "Standard deviation of additive Gaussian noise for this state variable.\n"
    "Only used when the integrator is EulerStochastic (Simulation tab).\n"
    "Set to 0 to make this variable deterministic."
)


def _fmt_bound(v: float) -> str:
    if math.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:g}"


def _parse_bound(s: str, default: float) -> float:
    t = s.strip().lower()
    if t in ("", "none"):
        return default
    if t in ("inf", "+inf"):
        return math.inf
    if t == "-inf":
        return -math.inf
    try:
        return float(t)
    except ValueError:
        return default


def _cell_text(table: QTableWidget, row: int, col: int) -> str:
    item = table.item(row, col)
    return item.text().strip() if item else ""


def _safe_float(table: QTableWidget, row: int, col: int, default: float) -> float:
    txt = _cell_text(table, row, col)
    try:
        return float(txt) if txt else default
    except ValueError:
        return default


class VarsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.state_table = self._build_table(STATE_COLUMNS)
        noise_header = self.state_table.horizontalHeaderItem(_NOISE_COL)
        if noise_header is not None:
            noise_header.setToolTip(_NOISE_TOOLTIP)
        self.obs_table = self._build_table(OBS_COLUMNS)

        layout = QVBoxLayout(self)
        layout.addWidget(self._state_group())
        layout.addWidget(self._obs_group())

    @staticmethod
    def _build_table(headers: tuple[str, ...]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        return t

    def _state_group(self) -> QGroupBox:
        add = QPushButton("Add state var")
        add.clicked.connect(lambda: self._append_state(
            StateVarDef(name=f"v{self.state_table.rowCount()}")))
        rm = QPushButton("Remove selected")
        rm.clicked.connect(lambda: self._remove(self.state_table))

        btns = QHBoxLayout()
        btns.addWidget(add)
        btns.addWidget(rm)
        btns.addStretch()

        box = QGroupBox("State variables (rows = order in state array)")
        v = QVBoxLayout(box)
        v.addWidget(self.state_table)
        v.addLayout(btns)
        v.addWidget(QLabel(
            "Bounds use 'inf' / '-inf' for unbounded. "
            "<b>Noise σ</b> is the standard deviation of additive Gaussian "
            "noise per variable, applied only when the integrator is "
            "<b>EulerStochastic</b> (set σ=0 to make a variable deterministic). "
            "<b>Coupled</b> marks which variables enter inter-region coupling."))
        return box

    def _obs_group(self) -> QGroupBox:
        add = QPushButton("Add observable")
        add.clicked.connect(lambda: self._append_obs(
            ObservableVarDef(name=f"o{self.obs_table.rowCount()}")))
        rm = QPushButton("Remove selected")
        rm.clicked.connect(lambda: self._remove(self.obs_table))

        btns = QHBoxLayout()
        btns.addWidget(add)
        btns.addWidget(rm)
        btns.addStretch()

        box = QGroupBox("Observable variables (computed each step, not integrated)")
        v = QVBoxLayout(box)
        v.addWidget(self.obs_table)
        v.addLayout(btns)
        return box

    @staticmethod
    def _remove(table: QTableWidget):
        rows = sorted({i.row() for i in table.selectedIndexes()}, reverse=True)
        for r in rows:
            table.removeRow(r)

    def _append_state(self, v: StateVarDef):
        r = self.state_table.rowCount()
        self.state_table.insertRow(r)
        self.state_table.setItem(r, 0, QTableWidgetItem(v.name))
        self.state_table.setItem(r, 1, QTableWidgetItem(f"{v.initial:g}"))
        self.state_table.setItem(r, 2, QTableWidgetItem(_fmt_bound(v.lo)))
        self.state_table.setItem(r, 3, QTableWidgetItem(_fmt_bound(v.hi)))
        self.state_table.setItem(r, 4, QTableWidgetItem(f"{v.sigma:g}"))
        chk = QTableWidgetItem()
        chk.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsSelectable
        )
        chk.setCheckState(Qt.CheckState.Checked if v.is_coupling else Qt.CheckState.Unchecked)
        chk.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_table.setItem(r, 5, chk)

    def _append_obs(self, o: ObservableVarDef):
        r = self.obs_table.rowCount()
        self.obs_table.insertRow(r)
        self.obs_table.setItem(r, 0, QTableWidgetItem(o.name))
        self.obs_table.setItem(r, 1, QTableWidgetItem(o.doc))

    def set_state_vars(self, vars_: list[StateVarDef]):
        self.state_table.setRowCount(0)
        for v in vars_:
            self._append_state(v)

    def set_observable_vars(self, vars_: list[ObservableVarDef]):
        self.obs_table.setRowCount(0)
        for v in vars_:
            self._append_obs(v)

    def get_state_vars(self) -> list[StateVarDef]:
        out: list[StateVarDef] = []
        for r in range(self.state_table.rowCount()):
            name = _cell_text(self.state_table, r, 0)
            if not name:
                continue
            chk = self.state_table.item(r, 5)
            is_coupling = (chk.checkState() == Qt.CheckState.Checked) if chk else False
            out.append(StateVarDef(
                name=name,
                initial=_safe_float(self.state_table, r, 1, 0.0),
                lo=_parse_bound(_cell_text(self.state_table, r, 2), -math.inf),
                hi=_parse_bound(_cell_text(self.state_table, r, 3), math.inf),
                sigma=_safe_float(self.state_table, r, 4, 0.0),
                is_coupling=is_coupling,
            ))
        return out

    def get_observable_vars(self) -> list[ObservableVarDef]:
        out: list[ObservableVarDef] = []
        for r in range(self.obs_table.rowCount()):
            name = _cell_text(self.obs_table, r, 0)
            if not name:
                continue
            out.append(ObservableVarDef(name=name, doc=_cell_text(self.obs_table, r, 1)))
        return out
