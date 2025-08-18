from __future__ import annotations
from pathlib import Path

from PySide6.QtCore     import QSize, QRectF
from PySide6.QtGui      import QPainter
from PySide6.QtSvg      import QSvgRenderer
from PySide6.QtWidgets  import (
    QSizePolicy,
    QWidget,
)
import sys

def resource_path(*parts: str) -> Path:
    base = getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)  # ../app
    return Path(base).joinpath(*parts)

class SvgLogo(QWidget):
    def __init__(self, svg_path: Path, preferred_height: int = 40, keep_aspect: bool = True, parent=None):
        super().__init__(parent)
        
        self.renderer = QSvgRenderer(str(svg_path))
        self._pref_h = preferred_height
        self.keep_aspect = keep_aspect

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def setSvg(self, svg_path: Path):
        self.renderer = QSvgRenderer(str(svg_path))
        self.updateGeometry()
        self.update()

    def sizeHint(self) -> QSize:
        if not self.renderer.isValid():
            return QSize(int(self._pref_h), self._pref_h)
        ds = self.renderer.defaultSize()
        if ds.height() <= 0:
            return QSize(int(self._pref_h), self._pref_h)
        ratio = ds.width() / ds.height()
        return QSize(int(self._pref_h * ratio), self._pref_h)

    def setPreferredHeight(self, h: int):
        self._pref_h = h
        self.updateGeometry()
        self.update()

    def paintEvent(self, event):
        if not self.renderer.isValid():
            return
        p = QPainter(self)
        p.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing, True
        )

        if not self.keep_aspect:
            # Растянуть на весь прямоугольник
            self.renderer.render(p, QRectF(0, 0, self.width(), self.height()))
            return

        # Сохранение пропорций и центрирование
        ds = self.renderer.defaultSize()
        if ds.height() <= 0:
            self.renderer.render(p, QRectF(0, 0, self.width(), self.height()))
            return

        ratio = ds.width() / ds.height()
        target_w = min(self.width(), self.height() * ratio)
        target_h = target_w / ratio
        if target_h > self.height():
            target_h = self.height()
            target_w = target_h * ratio

        x = (self.width() - target_w) / 2
        y = (self.height() - target_h) / 2
        self.renderer.render(p, QRectF(x, y, target_w, target_h))
