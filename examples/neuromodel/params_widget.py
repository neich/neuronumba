"""Parameter editor: a table with Name / Default / Tag / Doc columns."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QHeaderView, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from model_def import ParamDef


COLUMNS = ("Name", "Default", "Tag", "Doc")
TAG_CHOICES = ("regional", "plain")


def _cell_text(table: QTableWidget, row: int, col: int) -> str:
    item = table.item(row, col)
    return item.text().strip() if item else ""


class ParamsWidget(QWidget):
    def __init__(self, params: list[ParamDef] | None = None):
        super().__init__()
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        add = QPushButton("Add parameter")
        add.clicked.connect(self._on_add)
        rm = QPushButton("Remove selected")
        rm.clicked.connect(self._on_remove)

        btns = QHBoxLayout()
        btns.addWidget(add)
        btns.addWidget(rm)
        btns.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(btns)

        if params:
            self.set_params(params)

    def _on_add(self):
        self._append(ParamDef(name=f"p{self.table.rowCount()}", default=0.0, tag="regional"))

    def _on_remove(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _append(self, p: ParamDef):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(p.name))
        self.table.setItem(r, 1, QTableWidgetItem(repr(p.default)))
        cb = QComboBox()
        cb.addItems(TAG_CHOICES)
        cb.setCurrentText(p.tag if p.tag in TAG_CHOICES else "regional")
        self.table.setCellWidget(r, 2, cb)
        self.table.setItem(r, 3, QTableWidgetItem(p.doc))

    def set_params(self, params: list[ParamDef]):
        self.table.setRowCount(0)
        for p in params:
            self._append(p)

    def get_params(self) -> list[ParamDef]:
        out: list[ParamDef] = []
        for r in range(self.table.rowCount()):
            name = _cell_text(self.table, r, 0)
            if not name:
                continue
            try:
                default = float(_cell_text(self.table, r, 1) or "0")
            except ValueError:
                default = 0.0
            cb = self.table.cellWidget(r, 2)
            tag = cb.currentText() if cb else "regional"
            out.append(ParamDef(
                name=name, default=default, tag=tag,
                doc=_cell_text(self.table, r, 3)))
        return out
