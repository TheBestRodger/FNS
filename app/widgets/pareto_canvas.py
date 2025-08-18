# app/widgets/pareto_canvas.py
from __future__ import annotations

from typing import Tuple, Sequence
import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

# Матплотлиб иногда болтает про интерактивный бекенд — приглушим
try:
    import matplotlib
    matplotlib.set_loglevel("warning")
except Exception:
    pass


class ParetoCanvas(QWidget):
    """
    QWidget-обёртка над FigureCanvas.
    Экспортируемые методы:
      - setData(populations, pareto_front)
      - filter_eff(threshold: int)
    Сигналы:
      - pointSelected(object)  — если подключите обработчик клика по точке (заготовка оставлена).
    """
    pointSelected = Signal(object)

    def __init__(self, populations: Tuple, pareto_front: Tuple, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Матем
        self._populations: Tuple = populations or ([], [], tuple())
        self._pareto: Tuple = pareto_front or ([], [], tuple())
        self._eff_threshold: int = 0  # фильтр эффективности (если применяете)
        

        # Графика
        self.fig = Figure(figsize=(4, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
                # взаимодействие: выбор точки, панорамирование, зум колесом
        self._dragging = False
        self._drag_start: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

        # первичная отрисовка
        self._redraw()
        # обработчик выбора точки на диаграмме Парето
        self.canvas.mpl_connect("button_press_event", self._on_click)

    # -------------------- ПУБЛИЧНЫЙ API --------------------

    def setData(self, populations: Tuple, pareto_front: Tuple) -> None:
        self._populations = populations or ([], [], tuple())
        self._pareto = pareto_front or ([], [], tuple())
        self._redraw()

    def filter_eff(self, threshold: int) -> None:
        self._eff_threshold = int(threshold)
        self._redraw()

    # -------------------- ВНУТРЕННЕЕ -----------------------

    def _unpack(self, data: Tuple) -> Tuple[np.ndarray, np.ndarray, tuple]:
        """Ожидаем формат (xs, ys, meta), но терпимо относимся к пустым/сюрпризам."""
        if not data or len(data) < 2:
            return np.array([]), np.array([]), tuple()
        xs = np.asarray(data[0]) if data[0] is not None else np.array([])
        ys = np.asarray(data[1]) if data[1] is not None else np.array([])
        meta = data[2] if len(data) > 2 else tuple()
        return xs, ys, meta

    def _apply_eff_filter(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Если фильтр эффективности включён, отбрасываем точки с y < threshold."""
        if xs.size == 0 or ys.size == 0:
            return xs, ys
        thr = self._eff_threshold
        if thr <= 0:
            return xs, ys
        mask = ys >= thr
        return xs[mask], ys[mask]

    def _redraw(self) -> None:
        self.ax.clear()

        xs, ys, _ = self._unpack(self._populations)
        p_xs, p_ys, _ = self._unpack(self._pareto)

        # применим фильтр, если он осмыслен для ваших данных
        xs_f, ys_f = self._apply_eff_filter(xs, ys)
        p_xs_f, p_ys_f = self._apply_eff_filter(p_xs, p_ys)

        # рисуем только если есть точки
        artists = []
        if xs_f.size and ys_f.size:
            a = self.ax.scatter(xs_f, ys_f, s=12, alpha=0.5, label="Популяции")
            artists.append(a)
        if p_xs_f.size and p_ys_f.size:
            b = self.ax.scatter(p_xs_f, p_ys_f, s=18, alpha=0.9, marker="x", label="Парето")
            artists.append(b)

        self.ax.set_xlabel("Нагрузка")
        self.ax.set_ylabel("Эффективность")
        self.ax.grid(True, alpha=0.3)

        if artists:
            self.ax.legend(loc="best")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        """Обработка клика по точке Парето"""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        # координаты и соответствующие им особи
        p_xs, p_ys, meta = self._unpack(self._pareto)
        if p_xs.size == 0 or p_ys.size == 0 or len(meta) == 0:
            return

        meta_arr = np.asarray(meta, dtype=object)
        if self._eff_threshold > 0:
            mask = p_ys >= self._eff_threshold
            p_xs, p_ys, meta_arr = p_xs[mask], p_ys[mask], meta_arr[mask]
        if p_xs.size == 0:
            return

        # найти ближайшую точку
        dists = np.hypot(p_xs - event.xdata, p_ys - event.ydata)
        idx = int(dists.argmin())
        self.pointSelected.emit(meta_arr[idx])
     # --------- интерактивность ---------

    def _on_press(self, event) -> None:
        """ЛКМ — выбор точки, ПКМ — начало панорамирования."""
        if event.inaxes != self.ax:
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            self._emit_nearest(event.xdata, event.ydata)
        elif event.button == 3 and event.xdata is not None and event.ydata is not None:
            self._dragging = True
            self._drag_start = (event.xdata, event.ydata, self.ax.get_xlim(), self.ax.get_ylim())

    def _on_motion(self, event) -> None:
        if not self._dragging or event.inaxes != self.ax or self._drag_start is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0, xlim0, ylim0 = self._drag_start
        dx = event.xdata - x0
        dy = event.ydata - y0
        self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if event.button == 3:
            self._dragging = False
            self._drag_start = None

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 1.2 if event.step > 0 else 1 / 1.2
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        new_x = [event.xdata - (event.xdata - xlim[0]) * scale,
                 event.xdata + (xlim[1] - event.xdata) * scale]
        new_y = [event.ydata - (event.ydata - ylim[0]) * scale,
                 event.ydata + (ylim[1] - event.ydata) * scale]
        self.ax.set_xlim(new_x)
        self.ax.set_ylim(new_y)
        self.canvas.draw_idle()

    def _emit_nearest(self, x: float, y: float) -> None:
        """Найти ближайшую точку Парето и эмитить её мета-данные."""
        p_xs, p_ys, meta = self._unpack(self._pareto)
        if p_xs.size == 0 or p_ys.size == 0 or len(meta) == 0:
            return
        meta_arr = np.asarray(meta, dtype=object)
        dists = np.hypot(p_xs - x, p_ys - y)
        idx = int(dists.argmin())
        self.pointSelected.emit(meta_arr[idx])


