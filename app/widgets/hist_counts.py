# hist_counts.py
from __future__ import annotations
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HistCountsWidget(QWidget):
    """
    Гистограмма распределения по целым значениям: сколько задач у инспектора.
    Аналог вашей _draw_load_distribution(counts, order).
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._fig = Figure(figsize=(3, 2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def update_counts(self, counts: np.ndarray) -> None:
        """
        counts — ndarray длиной M, значения — число задач на инспектора.
        Строим гистограмму по целым бинам.
        """
        self._ax.clear()

        if counts is None or counts.size == 0:
            self._ax.set_title("Данные отсутствуют")
            self._canvas.draw_idle()
            return

        max_tasks = int(counts.max())
        # Бины по целым: -0.5, 0.5, 1.5, ..., max+0.5
        bin_edges = np.arange(max_tasks + 2) - 0.5

        n, edges, _ = self._ax.hist(counts, bins=bin_edges, edgecolor="white")
        # Центры бинов для тиков
        centers = (edges[:-1] + edges[1:]) / 2
        self._ax.set_xticks(centers)
        self._ax.set_xticklabels([str(i) for i in range(max_tasks + 1)])

        self._ax.set_title("Распределение нагрузки (кол-во задач)")
        self._ax.set_xlabel("Задач у инспектора")
        self._ax.set_ylabel("Число инспекторов")
        self._fig.tight_layout()
        self._canvas.draw_idle()
