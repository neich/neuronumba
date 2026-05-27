"""Equation editor with monospace font, minimal syntax highlighting, validate."""
from __future__ import annotations

import re

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)


class _Highlighter(QSyntaxHighlighter):
    KEYWORDS = (
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
        "pass", "raise", "return", "try", "while", "with", "yield", "True",
        "False", "None",
    )

    def __init__(self, doc):
        super().__init__(doc)
        self.kw_fmt = QTextCharFormat()
        self.kw_fmt.setForeground(QColor("#005cc5"))
        self.kw_fmt.setFontWeight(QFont.Weight.Bold)

        self.np_fmt = QTextCharFormat()
        self.np_fmt.setForeground(QColor("#6f42c1"))

        self.comment_fmt = QTextCharFormat()
        self.comment_fmt.setForeground(QColor("#6a737d"))
        self.comment_fmt.setFontItalic(True)

        self._kw_re = re.compile(r"\b(?:" + "|".join(self.KEYWORDS) + r")\b")
        self._np_re = re.compile(r"\bnp\.\w+\b")
        self._cmt_re = re.compile(r"#[^\n]*")

    def highlightBlock(self, text: str) -> None:
        for m in self._kw_re.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self.kw_fmt)
        for m in self._np_re.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self.np_fmt)
        for m in self._cmt_re.finditer(text):
            self.setFormat(m.start(), m.end() - m.start(), self.comment_fmt)


class EquationsWidget(QWidget):
    validate_requested = Signal()

    def __init__(self, initial: str = ""):
        super().__init__()
        self.editor = QPlainTextEdit()
        self.editor.setPlainText(initial)
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(11)
        self.editor.setFont(mono)
        self.editor.setTabStopDistance(self.editor.fontMetrics().horizontalAdvance("    "))
        self._hl = _Highlighter(self.editor.document())

        self.validate_btn = QPushButton("Validate equations")
        self.validate_btn.clicked.connect(self.validate_requested.emit)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        btns = QHBoxLayout()
        btns.addWidget(self.validate_btn)
        btns.addStretch()

        hint = QLabel(
            "Define <code>d&lt;name&gt;</code> for every state variable and "
            "assign every observable name. Available identifiers: state "
            "variables (<code>S_e</code>, …), coupling inputs "
            "(<code>cpl_S_e</code>, …), regional parameters, and <code>np</code>."
        )
        hint.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(hint)
        layout.addWidget(self.editor, 1)
        layout.addLayout(btns)
        layout.addWidget(self.status)

    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, txt: str) -> None:
        self.editor.setPlainText(txt)

    def set_status(self, msg: str, ok: bool) -> None:
        color = "#22863a" if ok else "#b31d28"
        self.status.setText(f"<span style='color:{color}'>{msg}</span>")
