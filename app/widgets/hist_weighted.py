# hist_weighted.py
from __future__ import annotations
import numpy as np
from typing import Sequence
from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ForecastCountsWidget(QWidget):
    """
    _draw_future_load: один бар-чарт по отсортированным counts.
    Показывает эффективность в заголовке, если передана.
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._fig = Figure(figsize=(3, 2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def update_counts(self,
                      inspectors_index: Sequence[int],
                      counts: np.ndarray | None,
                      eff: float | None = None) -> None:
        """
        inspectors_index — индексы 0..M-1 (или любая последовательность для оси X)
        counts           — массив длиной <= len(inspectors_index)
        eff              — эффективность (%) для заголовка
        """
        self._ax.clear()

        if counts is None or len(counts) == 0:
            self._ax.set_title("Нет данных для прогноза")
            self._canvas.draw_idle()
            return

        # Приводим длину X к длине данных (на всякий случай)
        n = min(len(inspectors_index), len(counts))
        x = list(inspectors_index)[:n]
        y = np.sort(counts[:n])[::-1]

        self._ax.bar(x, y, width=0.8, alpha=0.7, label="Распределённая нагрузка")
        self._ax.set_xlabel("Инспекторы (отсортированы по числу задач)")
        self._ax.set_ylabel("Количество задач")
        self._ax.grid(True, alpha=0.3)
        self._ax.legend()

        title = "Прогноз распределения"
        if eff is not None:
            title += f" — эффективность {eff:.2f}%"
        self._ax.set_title(title)

        self._fig.tight_layout()
        self._canvas.draw_idle()


class WeightedCompareWidget(QWidget):
    """
    _draw_pareto_distribution:
    сравнивает базовое распределение и новое (оба одномерные ряды одной размерности).
    Рисует среднюю и ±1σ по базовому ряду.
    ВАЖНО: не смешивайте 'counts' и '%'. Передавайте данные в одних единицах!
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._fig = Figure(figsize=(3, 2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._canvas)

    def update_series(self,
                      inspectors_index: Sequence[int],
                      baseline: np.ndarray,
                      candidate: np.ndarray,
                      eff: float | None = None,
                      y_label: str = "Нагрузка (одинаковые единицы)") -> None:
        """
        baseline  — базовый ряд (например, будущая «реальная» нагрузка) -> будет отсортирован по убыванию
        candidate — новый ряд для сравнения (та же метрика)
        """
        self._ax.clear()

        if baseline is None or baseline.size == 0:
            self._ax.set_title("Нет базового ряда для сравнения")
            self._canvas.draw_idle()
            return

        # Сортируем оба ряда по значению baseline (убывание), чтобы индексы соответствовали
        order = np.argsort(baseline)[::-1]
        base_sorted = baseline[order]
        cand_sorted = candidate[order] if (candidate is not None and candidate.size == baseline.size) \
                      else np.zeros_like(base_sorted)

        inspectors_arr = np.asarray(inspectors_index)
        if inspectors_arr.size == 0:
            inspectors_sorted = np.arange(base_sorted.size)
            valid_mask = np.ones_like(base_sorted, dtype=bool)
        else:
            valid_mask = order < inspectors_arr.size
            inspectors_sorted = inspectors_arr[order[valid_mask]]

        base_display = base_sorted[valid_mask]
        cand_display = cand_sorted[valid_mask]

        n = min(inspectors_sorted.size, base_display.size)
        x = inspectors_sorted[:n]
        base_display = base_display[:n]
        cand_display = cand_display[:n]
        positions = np.arange(n)

        self._ax.bar(positions, base_display, width=0.8, alpha=0.7, label="Базовое распределение")
        self._ax.bar(positions, cand_display, width=0.8, alpha=0.7, label="Новое распределение")

        mean = float(base_sorted.mean())
        std  = float(base_sorted.std())
        self._ax.axhline(y=mean, linestyle="--", alpha=0.7, label=f"Среднее = {mean:.2f}")
        self._ax.fill_between(positions, [mean-std]*n, [mean+std]*n, alpha=0.2, label=f"±1σ ({std:.2f})")

        self._ax.set_xticks(positions)
        self._ax.set_xticklabels(x.tolist() if isinstance(x, np.ndarray) else list(x))

        self._ax.set_xlabel("Инспекторы (отсортированы по базовому ряду)")
        self._ax.set_ylabel(y_label)
        self._ax.grid(True, alpha=0.3)
        self._ax.legend()

        title = "Сравнение распределений"
        if eff is not None:
            title += f" — эффективность {eff:.2f}%"
        self._ax.set_title(title)

        self._fig.tight_layout()
        self._canvas.draw_idle()
