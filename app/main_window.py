# main_window.py (ключевые изменения для асинхронной загрузки)
from __future__ import annotations
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (QApplication,
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QGroupBox, QSlider, QSizePolicy, QTableWidget, QTableWidgetItem,
    QPushButton, QStackedLayout, QButtonGroup, QFileDialog, QMessageBox,
    QProgressDialog, QStatusBar, QProgressBar
)

from core.deap_optim import _evaluation

from style.svg_logo import SvgLogo, resource_path
from style.colors import STYLE_FNS

from widgets.pareto_canvas import ParetoCanvas
from widgets.hist_counts import HistCountsWidget
from widgets.hist_weighted import ForecastCountsWidget, WeightedCompareWidget
from widgets.table_assignment import AssignmentTable            # <-- замена QTableWidget на QTableView

from models.state import DataState

from services.loader import LoadWorker                   # <-- новый воркер

DATA_DIR = Path("")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Анализ Парето-фронта")

        # ----------- состояние (чтобы методы компилировались до первой загрузки)
        self.data_dir: Path = Path(DATA_DIR)
        self.populations = ([], [], tuple())
        self.pareto_front = ([], [], tuple())
        self.assignment_df = pd.DataFrame()
        self.current_inspectors = pd.DataFrame()
        self.current_individ: list[int] = []
        self.future_individ: list[int] = []
        self.future_eff: float | None = None
        self.sorted_future_load = np.array([], dtype=float)
        self.sorted_future_counts = np.array([], dtype=int)
        self.inspectors_index: list[int] = []
        self.task_prob = np.array([])
        self.N = self.M = 0
        self.task_cat = np.array([])
        self.den_TNO: dict = {}

        # ----------- UI
        central = QWidget()
        self.setCentralWidget(central)
        self.root_layout = QVBoxLayout(central)
        self.root_layout.setContentsMargins(0, 0, 0, 0)
        self.root_layout.setSpacing(0)

        self._build_header()

        hline = QFrame(); hline.setFrameShape(QFrame.HLine); hline.setFrameShadow(QFrame.Sunken)
        self.root_layout.addWidget(hline)

        self.plot = ParetoCanvas(self.populations, self.pareto_front, self)
        self.root_layout.addWidget(self.plot, stretch=4)

        # filter_box = QGroupBox("Фильтр эффективности")
        # self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 100); self.slider.setValue(0)
        # self.slider_label = QLabel("≥ 0 %")
        # self.slider.valueChanged.connect(self._on_slider)
        # flay = QVBoxLayout(filter_box); flay.addWidget(self.slider); flay.addWidget(self.slider_label, alignment=Qt.AlignCenter)
        # self.root_layout.addWidget(filter_box)

        bottom = QHBoxLayout(); self.root_layout.addLayout(bottom, stretch=3)

        # Левый столбец
        left_pane = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_pane.addWidget(self.table, stretch=1)

        # ЗАМЕНА: таблица назначений теперь QTableView с моделью
        self.result_table = AssignmentTable(self)
        left_pane.addWidget(self.result_table, stretch=2)
        bottom.addLayout(left_pane, stretch=3)

        # Правый столбец
        self.hist_counts = HistCountsWidget(self)
        self.hist_forecast = ForecastCountsWidget(self)
        self.hist_compare = WeightedCompareWidget(self)

        self.hist_stack = QStackedLayout()
        self.hist_stack.addWidget(self.hist_counts)
        self.hist_stack.addWidget(self.hist_forecast)
        self.hist_stack.addWidget(self.hist_compare)

        hist_container = QWidget(); hist_container.setLayout(self.hist_stack)
        btn_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)
        for text, idx in [("Распределение", 0), ("Прогноз", 1), ("Нагрузка", 2)]:
            btn = QPushButton(text); btn.setCheckable(True); btn.setChecked(idx == 0)
            btn.clicked.connect(lambda _=False, x=idx: self.hist_stack.setCurrentIndex(x))
            self.btn_group.addButton(btn); btn_layout.addWidget(btn)
        right_pane = QVBoxLayout(); right_pane.addLayout(btn_layout); right_pane.addWidget(hist_container, stretch=1)
        bottom.addLayout(right_pane, stretch=2)

        # Меню
        save_act = self.menuBar().addAction("Save PNG…")
        save_act.triggered.connect(self.save_png)
        load_act = self.menuBar().addAction("Load CSVs…")
        load_act.triggered.connect(self.choose_csv_dir)

        # Статус-бар + индикатор прогресса (постоянный)
        sb = QStatusBar(self); self.setStatusBar(sb)
        self.sb_progress = QProgressBar(self); self.sb_progress.setMaximumWidth(160)
        self.sb_progress.setRange(0, 100); self.sb_progress.setValue(0); self.sb_progress.setVisible(False)
        sb.addPermanentWidget(self.sb_progress)

        # Диалог прогресса для «длинной» загрузки (модальный)
        self.progress_dlg: QProgressDialog | None = None

        # Сигнал из ParetoCanvas
        self.plot.pointSelected.connect(self.show_params)

        # ---- ПЕРВИЧНАЯ ЗАГРУЗКА (асинхронно) ----
        self.start_load(self.data_dir)

        self.resize(1020, 720)

    # ------------------------ Асинхронная загрузка ---------------------------
    def start_load(self, data_dir: Path) -> None:
        """Запустить загрузку данных в отдельном потоке."""
        self._set_ui_enabled(False)

        # Статус-бар прогресс
        # сбрасываем и показываем индикатор на каждую новую загрузку
        self.sb_progress.setRange(0, 100)
        self.sb_progress.reset()
        self.sb_progress.setVisible(True)

        # Модальный прогресс-диалог (информативные подписи этапов)
        self.progress_dlg = QProgressDialog("Загрузка данных...", "Отмена", 0, 100, self)
        self.progress_dlg.setWindowTitle("Пожалуйста, подождите")
        self.progress_dlg.setAutoClose(False)
        self.progress_dlg.setAutoReset(False)
        self.progress_dlg.setMinimumDuration(0)  # показать сразу
        self.progress_dlg.canceled.connect(self._cancel_load)  # просто закрывает диалог

        # Поток + воркер
        self.loader_thread = QThread(self)
        self.loader_worker = LoadWorker(Path(data_dir))
        self.loader_worker.moveToThread(self.loader_thread)

        # wiring
        self.loader_thread.started.connect(self.loader_worker.run)
        self.loader_worker.progress.connect(self._on_load_progress)
        self.loader_worker.finished.connect(self._on_load_finished)
        self.loader_worker.error.connect(self._on_load_error)

        # life-cycle
        self.loader_worker.finished.connect(self.loader_thread.quit)
        self.loader_worker.finished.connect(self.loader_worker.deleteLater)
        self.loader_thread.finished.connect(self.loader_thread.deleteLater)

        self.loader_thread.start()

    def _update_progress(self, percent: int, message: str) -> None:
        """Helper to update both progress indicators and process UI events."""
        if self.progress_dlg:
            self.progress_dlg.setLabelText(message)
            self.progress_dlg.setValue(percent)
        self.sb_progress.setValue(percent)
        self.statusBar().showMessage(message)
        QApplication.processEvents()

    def _on_load_progress(self, percent: int, message: str) -> None:
        # Логика загрузки занимает первую половину прогресса (0-50)
        self._update_progress(percent // 2, message)

    def _on_load_finished(self, state: DataState) -> None:
        # После загрузки используем вторую половину прогресса для отрисовки
        self._update_progress(50, "Обработка результатов...")
        self._apply_state(state)

        # Завершить прогресс
        if self.progress_dlg:
            self.progress_dlg.close()
            self.progress_dlg = None
        self.sb_progress.reset()
        self.sb_progress.setVisible(False)
        self.statusBar().clearMessage()
        self._set_ui_enabled(True)

    def _on_load_error(self, text: str) -> None:
        if self.progress_dlg:
            self.progress_dlg.close()
            self.progress_dlg = None
        self.sb_progress.reset()
        self.sb_progress.setVisible(False)
        self.statusBar().clearMessage()
        self._set_ui_enabled(True)
        QMessageBox.critical(self, "Ошибка загрузки", text)

    def _cancel_load(self) -> None:
        # Жёсткой отмены нет (поток занят I/O); просто скрываем диалог
        # (при желании можно реализовать флаг отмены внутри воркера)
        if self.progress_dlg:
            self.progress_dlg.hide()

    def _set_ui_enabled(self, enabled: bool) -> None:
        self.menuBar().setEnabled(enabled)
        # self.slider.setEnabled(enabled)
        for btn in self.btn_group.buttons():
            btn.setEnabled(enabled)

    # ------------------------ Применение состояния ----------------------------
    def _apply_state(self, s: DataState) -> None:
        self.data_dir = s.data_dir

        # Pareto
        self._update_progress(60, "Отрисовка Парето-фронта...")
        self.populations = s.populations
        self.pareto_front = s.pareto_front
        if hasattr(self.plot, "setData"):
            self.plot.setData(self.populations, self.pareto_front)
        else:
            # fallback: пересоздать ParetoCanvas
            old_index = self.root_layout.indexOf(self.plot)
            self.root_layout.removeWidget(self.plot)
            self.plot.deleteLater()
            self.plot = ParetoCanvas(self.populations, self.pareto_front, self)
            self.root_layout.insertWidget(old_index, self.plot, stretch=4)
            self.plot.pointSelected.connect(self.show_params)

        self._update_progress(70, "Обновление таблицы...")

        # Таблица назначений
        self.assignment_df = s.assignment_df
        self.result_table.set_dataframe(self.assignment_df)

        # Текущие/будущие
        self.current_inspectors = s.current_inspectors
        self.current_individ = s.current_individ
        self.future_individ = s.future_individ

        self.task_prob = s.task_prob
        self.N, self.M = s.N, s.M
        self.task_cat = s.task_cat
        self.den_TNO = s.den_TNO

        # Оси/метрики
        self.inspectors_index = s.inspectors_index
        self.sorted_future_counts = s.future_counts_sorted
        self.sorted_future_load = s.future_load_sorted
        self.future_eff = s.future_eff

        self._update_progress(90, "Построение гистограмм...")

        # Перерисовка правых графиков
        counts_all = self._task_counts(self.current_individ + self.future_individ)
        self.hist_counts.update_counts(counts_all)

        self.hist_forecast.update_counts(
            inspectors_index=self.inspectors_index,
            counts=self.sorted_future_counts,
            eff=self.future_eff,
        )

        self.hist_compare.update_series(
            inspectors_index=self.inspectors_index,
            baseline=self.sorted_future_load,
            candidate=self.sorted_future_load.copy(),
            eff=self.future_eff,
            y_label="Взвешенная нагрузка (ед.)",
        )

        self._update_progress(100, "Готово")
 #svg_path = Path(__file__).with_name("style").joinpath("logo.svg")
    # ------------------------ Остальное без изменений -------------------------
    def _build_header(self) -> None:
        svg_path = resource_path("style", "logo.svg")

        logo = SvgLogo(svg_path, preferred_height=80)
        header_widget = QWidget(); header_widget.setObjectName("Header")
        header_widget.setStyleSheet(f"#Header {{ background-color: {STYLE_FNS}; border: none; }}")
        header = QHBoxLayout(header_widget); header.setContentsMargins(16, 8, 16, 8); header.setSpacing(12)
        title_layout = QVBoxLayout(); title_layout.setContentsMargins(0, 0, 0, 0); title_layout.setSpacing(4)
        subtitle = QLabel("НОЦ ФНС России и МГТУ им. Н. Э. Баумана"); subtitle.setStyleSheet("font-size: 14px; color: #333;")
        title_layout.addWidget(logo); title_layout.addWidget(subtitle)
        header.addLayout(title_layout); header.addStretch(1)
        self.root_layout.addWidget(header_widget)

    def _task_counts(self, individual: Sequence[int]) -> np.ndarray:
        counts = np.zeros(self.M, dtype=int)
        for idx in individual:
            if isinstance(idx, (int, np.integer)) and 0 <= idx < self.M:
                counts[idx] += 1
        return counts

    def show_params(self, ind: Sequence[int]) -> None:
        ind = list(ind)
        full_ind = ind + self.current_individ
        counts = self._task_counts(full_ind)
        order = np.argsort(counts)[::-1]
        top10_idx = order[:10]
        self.table.setRowCount(len(top10_idx))
        for row, idx in enumerate(top10_idx):
            self.table.setItem(row, 0, QTableWidgetItem(f"Сотрудник {idx}"))
            self.table.setItem(row, 1, QTableWidgetItem(str(int(counts[idx]))))

        self.hist_counts.update_counts(counts)

        loads_candidate, eff = _evaluation(  # если у вас обёртка — используйте её
            ind, self.den_TNO, self.task_cat, self.task_prob, self.M,
            results=True, current_individ=self.current_individ,
        )

        task_counts_candidate = self._task_counts(ind)
        self.hist_forecast.update_counts(self.inspectors_index, task_counts_candidate, eff)

        sorted_candidate_loads = np.sort(loads_candidate)[::-1]
        self.hist_compare.update_series(
            inspectors_index=self.inspectors_index,
            baseline=self.sorted_future_load,
            candidate=sorted_candidate_loads,
            eff=eff, y_label="Взвешенная нагрузка (ед.)",
        )

    # def _on_slider(self, value: int) -> None:
    #     self.plot.filter_eff(value)
    #     self.slider_label.setText(f"≥ {value} %")

    def save_png(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG (*.png)")
        if filename:
            self.plot.fig.savefig(filename, dpi=300)
            QMessageBox.information(self, "Сохранено", f"Изображение сохранено:\n{Path(filename).name}")

    def choose_csv_dir(self) -> None:
        new_dir = QFileDialog.getExistingDirectory(self, "Выбрать папку с CSV", str(self.data_dir))
        if not new_dir:
            return

        # Пересоздаём ParetoCanvas после загрузки — тогда, когда будут данные
        self.start_load(Path(new_dir))
        # (в _on_load_finished->_apply_state данные придут, дальше просто setData())
        self.result_table.set_dataframe(self.assignment_df)
    # def choose_csv_dir(self) -> None:
    #     """Выбор новой директории с CSV и обновление всех виджетов."""
    #     new_dir = QFileDialog.getExistingDirectory(self, "Выбрать папку с CSV", str(self.data_dir))
    #     if not new_dir:
    #         return

    #     # 1) Перезагрузка данных
    #     self._load_data(Path(new_dir))
    #     self.result_table.set_dataframe(self.assignment_df)
    #     # 2) Пересоздать ParetoCanvas (он зависит от populations/pareto_front)
    #     old_index = self.root_layout.indexOf(self.plot)
    #     self.root_layout.removeWidget(self.plot)
    #     self.plot.deleteLater()

    #     self.plot = ParetoCanvas(self.populations, self.pareto_front, self)
    #     self.root_layout.insertWidget(old_index, self.plot, stretch=4)
    #     self.plot.pointSelected.connect(self.show_params)

    #     # 3) Обновить таблицу результатов и графики
    #     self._after_data_loaded_initial()