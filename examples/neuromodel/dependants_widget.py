"""Editor for dependent attributes (a la neuronumba's Attr(dependant=True)).

Each dependant has a name and a Python formula evaluated at configure time,
with access to: ``weights``, ``n_rois``, ``g``, ``np``, every regional and
plain parameter, and every earlier dependant in the list. The formula must
assign to a variable matching the dependant's name (e.g. for ``weights_t``
write ``weights_t = weights.T.copy()``).
"""
from __future__ import annotations

from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from model_def import DependantDef


_HINT = (
    "Dependants are values computed at <i>configure time</i> from "
    "<code>weights</code>, <code>n_rois</code>, <code>g</code>, "
    "<code>np</code>, the model's parameters and any earlier dependants. "
    "Each formula must assign to a variable with the dependant's name "
    "(e.g. for <code>weights_t</code> write "
    "<code>weights_t = weights.T.copy()</code>). The resulting values are "
    "stored on the model and made available inside the equation body."
)


class DependantsWidget(QWidget):
    def __init__(self, deps: list[DependantDef] | None = None):
        super().__init__()
        self._deps: list[DependantDef] = []
        self._suppress = False

        self.listw = QListWidget()
        self.listw.currentRowChanged.connect(self._on_select_changed)

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add)
        rm_btn = QPushButton("Remove")
        rm_btn.clicked.connect(self._on_remove)

        list_btns = QHBoxLayout()
        list_btns.addWidget(add_btn)
        list_btns.addWidget(rm_btn)

        left = QVBoxLayout()
        left.addWidget(self.listw, 1)
        left.addLayout(list_btns)
        left_w = QWidget()
        left_w.setLayout(left)

        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._on_name_changed)
        self.formula_edit = QPlainTextEdit()
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(11)
        self.formula_edit.setFont(mono)
        self.formula_edit.setTabStopDistance(
            self.formula_edit.fontMetrics().horizontalAdvance("    "))
        self.formula_edit.textChanged.connect(self._on_formula_changed)
        self.doc_edit = QLineEdit()
        self.doc_edit.textChanged.connect(self._on_doc_changed)

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Formula:", self.formula_edit)
        form.addRow("Doc:", self.doc_edit)
        right_w = QWidget()
        right_w.setLayout(form)

        splitter = QSplitter()
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 720])

        hint = QLabel(_HINT)
        hint.setWordWrap(True)
        hint.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(splitter, 1)

        if deps:
            self.set_dependants(deps)
        else:
            self._set_editor_enabled(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        self.name_edit.setEnabled(enabled)
        self.formula_edit.setEnabled(enabled)
        self.doc_edit.setEnabled(enabled)

    def _on_add(self) -> None:
        d = DependantDef(name=f"d{len(self._deps)}", formula="", doc="")
        self._deps.append(d)
        self.listw.addItem(d.name)
        self.listw.setCurrentRow(len(self._deps) - 1)

    def _on_remove(self) -> None:
        r = self.listw.currentRow()
        if r < 0 or r >= len(self._deps):
            return
        self._deps.pop(r)
        self.listw.takeItem(r)
        if self._deps:
            self.listw.setCurrentRow(min(r, len(self._deps) - 1))
        else:
            self._on_select_changed(-1)

    def _on_select_changed(self, row: int) -> None:
        self._suppress = True
        try:
            if row < 0 or row >= len(self._deps):
                self.name_edit.setText("")
                self.formula_edit.setPlainText("")
                self.doc_edit.setText("")
                self._set_editor_enabled(False)
            else:
                d = self._deps[row]
                self.name_edit.setText(d.name)
                self.formula_edit.setPlainText(d.formula)
                self.doc_edit.setText(d.doc)
                self._set_editor_enabled(True)
        finally:
            self._suppress = False

    def _current_dep(self) -> DependantDef | None:
        r = self.listw.currentRow()
        if r < 0 or r >= len(self._deps):
            return None
        return self._deps[r]

    def _on_name_changed(self, text: str) -> None:
        if self._suppress:
            return
        d = self._current_dep()
        if d is None:
            return
        d.name = text.strip()
        item = self.listw.item(self.listw.currentRow())
        if item is not None:
            item.setText(d.name or "(unnamed)")

    def _on_formula_changed(self) -> None:
        if self._suppress:
            return
        d = self._current_dep()
        if d is None:
            return
        d.formula = self.formula_edit.toPlainText()

    def _on_doc_changed(self, text: str) -> None:
        if self._suppress:
            return
        d = self._current_dep()
        if d is None:
            return
        d.doc = text

    def set_dependants(self, deps: list[DependantDef]) -> None:
        self._suppress = True
        try:
            self._deps = [
                DependantDef(name=d.name, formula=d.formula, doc=d.doc) for d in deps
            ]
            self.listw.clear()
            for d in self._deps:
                self.listw.addItem(d.name or "(unnamed)")
        finally:
            self._suppress = False
        if self._deps:
            self.listw.setCurrentRow(0)
        else:
            self._on_select_changed(-1)

    def get_dependants(self) -> list[DependantDef]:
        return [
            DependantDef(name=d.name, formula=d.formula, doc=d.doc)
            for d in self._deps
        ]
