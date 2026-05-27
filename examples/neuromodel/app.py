"""Main window with one tab per model under construction."""
from __future__ import annotations

import sys

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QPushButton, QTabWidget, QToolBar,
)

from model_def import ModelDef
from tab import ModelTab
from templates import TEMPLATES


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuronumba — Model Builder")
        self.resize(1320, 880)

        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.tabCloseRequested.connect(self._on_close_tab)
        self.setCentralWidget(self.tabs)

        tb = QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" New tab from template: "))
        self.tmpl_combo = QComboBox()
        self.tmpl_combo.addItems(list(TEMPLATES.keys()))
        self.tmpl_combo.setCurrentText("Deco2014")
        tb.addWidget(self.tmpl_combo)

        new_btn = QPushButton("Add tab")
        new_btn.clicked.connect(self._on_new_tab)
        tb.addWidget(new_btn)

        tb.addSeparator()

        open_btn = QPushButton("Open JSON…")
        open_btn.clicked.connect(self._on_open_json)
        tb.addWidget(open_btn)

        new_action = QAction("New tab", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._on_new_tab)
        self.addAction(new_action)

        self._add_from_template("Deco2014")

    def _on_new_tab(self):
        self._add_from_template(self.tmpl_combo.currentText())

    def _add_from_template(self, key: str):
        md = TEMPLATES[key]()
        self._add_tab(ModelTab(md), md.name)

    def _add_tab(self, tab: ModelTab, label: str):
        idx = self.tabs.addTab(tab, label)
        self.tabs.setCurrentIndex(idx)
        tab.name_edit.textChanged.connect(lambda txt, t=tab: self._rename_tab(t, txt))

    def _rename_tab(self, tab: ModelTab, text: str):
        idx = self.tabs.indexOf(tab)
        if idx >= 0:
            self.tabs.setTabText(idx, text or "Untitled")

    def _on_close_tab(self, index: int):
        if self.tabs.count() <= 1:
            return
        widget = self.tabs.widget(index)
        self.tabs.removeTab(index)
        widget.deleteLater()

    def _on_open_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open model JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                md = ModelDef.from_json(f.read())
        except Exception as e:
            QMessageBox.warning(self, "Open failed", str(e))
            return
        self._add_tab(ModelTab(md), md.name)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
