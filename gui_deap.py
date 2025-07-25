import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QSlider,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QFrame,
)
from PySide6.QtGui import QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

from deap_optim import (
    get_results,
    get_assignment_table,
    _get_dataframe,
    _filter_by_no,
    _start_data_prep,
    _evaluation,
)

DATA_DIR = Path("data")
NO_CODE = 3700


class ParetoCanvas(QWidget):
    """Scatter plot + интерактивный Парето‑фронт."""

    pointSelected = Signal(object)  # → list[int]

    def __init__(self, pops, pfront, parent=None):
        super().__init__(parent)
        self.loads, self.effs, self.inds = [np.asarray(a, dtype=float) for a in pops]
        pf_loads, pf_effs, self.pf_inds = pfront

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

        # Scatter points
        self.scatter = self.ax.scatter(
            self.loads,
            self.effs,
            s=30,
            c="skyblue",
            alpha=0.65,
            picker=True,
            label="All",
        )
        self.pf_scatter = self.ax.scatter(
            pf_loads,
            pf_effs,
            s=70,
            color="crimson",
            picker=True,
            label="Pareto Front",
        )

        # Tooltip
        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.annot.set_visible(False)

        # Axes formatting
        self.ax.set_xlabel("Max load, %")
        self.ax.set_ylabel("Efficiency, %")
        self.ax.legend()
        self.fig.tight_layout()

        # mpl events
        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    # Utility helpers

    @staticmethod
    def _build_counts(individual):
        counts = {}
        for idx in individual:
            counts[idx] = counts.get(idx, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    # Hover tooltip
    def _on_hover(self, event):
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.pf_scatter.contains(event)
            if cont:
                idx = ind["ind"][0]
                items = self._build_counts(self.pf_inds[idx])[:10]
                text = "\n".join(f"Сотрудник {emp}: {load} задач" for emp, load in items)
                self.annot.xy = self.pf_scatter.get_offsets()[idx]
                self.annot.set_text(text)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
                return
        if vis:
            self.annot.set_visible(False)
            self.canvas.draw_idle()

    # Pick event
    def _on_pick(self, ev):
        idx = ev.ind[0]
        data = self.pf_inds[idx] if ev.artist is self.pf_scatter else self.inds[idx]
        self.pointSelected.emit(data)

    # Scroll zoom
    def _on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        base_scale = 1.2
        scale = 1 / base_scale if event.button == "up" else base_scale

        ax = self.ax
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        x_range = (x_right - x_left) * scale
        y_range = (y_top - y_bottom) * scale
        relx = (event.xdata - x_left) / (x_right - x_left)
        rely = (event.ydata - y_bottom) / (y_top - y_bottom)
        ax.set_xlim(event.xdata - relx * x_range, event.xdata + (1 - relx) * x_range)
        ax.set_ylim(event.ydata - rely * y_range, event.ydata + (1 - rely) * y_range)
        self.canvas.draw_idle()

    # Efficiency slider
    def filter_eff(self, min_eff):
        mask = self.effs >= min_eff
        self.scatter.set_offsets([[x, y] for x, y, m in zip(self.loads, self.effs, mask) if m])
        self.canvas.draw_idle()


class MainWindow(QMainWindow):
    def _load_data(self, data_dir: Path):
        """Load optimisation and helper datasets from ``data_dir``.

        If the required CSV files do not exist, all data attributes are cleared
        so that the graphs appear empty.
        """

        self.data_dir = Path(data_dir)

        try:
            inspector_df, new_df, inwork_df = _get_dataframe(self.data_dir)
        except FileNotFoundError:
            # CSVs missing → show empty graphs
            self.populations = ([], [], tuple())
            self.pareto_front = ([], [], tuple())
            self.assignment_df = pd.DataFrame()
            self.current_inspectors = pd.DataFrame()
            self.current_individ = []
            self.future_individ = []
            self.future_eff = None
            self.sorted_future_load = np.array([])
            self.future_counts = np.array([])
            self.sorted_future_counts = np.array([])
            self.inspectors_index = []
            self.task_prob = np.array([])
            self.N = self.M = 0
            self.task_cat = np.array([])
            self.den_TNO = {}
            return

        # Optimisation results
        self.populations, self.pareto_front = get_results(data_dir=self.data_dir)
        self.assignment_df = get_assignment_table(data_dir=self.data_dir)

        # Additional data for histograms

        self.current_inspectors, current_df = _filter_by_no(NO_CODE, inspector_df, inwork_df)
        current_df = current_df.merge(
            self.current_inspectors.reset_index()[["Инспектор, сменивший статус", "Inspector index"]],
            how="left",
            left_on="Инспектор, сменивший статус",
            right_on="Инспектор, сменивший статус",
        )
        self.current_individ = current_df["Inspector index"].to_list()

        no_inspectors, no_df = _filter_by_no(NO_CODE, inspector_df, new_df)

        self.task_prob, self.N, self.M, self.task_cat, self.den_TNO = _start_data_prep(
            no_inspectors, no_df, current_df
        )

        future_individ = no_df[[
            "Статус РСЗ",
            "Тип",
            "Потенциальный ущерб, руб",
            "ИНН НП",
            "Инспектор, сменивший статус",
        ]].copy()
        future_individ = future_individ.merge(
            no_inspectors.reset_index()[["Инспектор, сменивший статус", "Inspector index"]],
            how="left",
            left_on="Инспектор, сменивший статус",
            right_on="Инспектор, сменивший статус",
        )
        self.future_individ = future_individ["Inspector index"].to_list()

        future_load, self.future_eff = _evaluation(
            self.future_individ,
            self.den_TNO,
            self.task_cat,
            self.task_prob,
            self.M,
            results=True,
            current_individ=self.current_individ,
        )

        self.sorted_future_load = np.sort(future_load)[::-1]
        self.future_counts = self._task_counts(self.future_individ)
        self.sorted_future_counts = np.sort(self.future_counts)[::-1]
        self.inspectors_index = list(range(len(self.current_inspectors)))

    def __init__(self):
        super().__init__()
        self.setWindowTitle("РСЗ – Анализ Парето‑фронта")

        # Load optimisation data
        self._load_data(DATA_DIR)

        central = QWidget()
        self.setCentralWidget(central)
        self.root_layout = QVBoxLayout(central)

        # Header with logo and controls
        logo = QLabel()
        logo_path = Path(__file__).with_name("style").joinpath("logo.png")
        logo.setPixmap(QPixmap(str(logo_path)).scaledToHeight(40, Qt.SmoothTransformation))

        header = QHBoxLayout()
        title_layout = QVBoxLayout()
        title = QLabel("Проект РСЗ")
        subtitle = QLabel("НОЦ ФНС России и МГТУ им. Н. Э. Баумана")
        title_layout.addWidget(logo)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout)
        header.addStretch(1)

        self.save_btn = QPushButton("Save PNG…")
        self.save_btn.clicked.connect(self.save_png)
        self.load_btn = QPushButton("Load CSVs…")
        self.load_btn.clicked.connect(self.choose_csv_dir)
        header.addWidget(self.save_btn)
        header.addWidget(self.load_btn)

   
        self.root_layout.addLayout(header)
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setFrameShadow(QFrame.Sunken)
        self.root_layout.addWidget(hline)

        self.plot = ParetoCanvas(self.populations, self.pareto_front, self)
        self.root_layout.addWidget(self.plot, stretch=4)

        filter_box = QGroupBox("Фильтр эффективности")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        lbl = QLabel("≥ 0 %")
        self.slider_label = lbl
        self.slider.valueChanged.connect(self._on_slider)
        flay = QVBoxLayout(filter_box)
        flay.addWidget(self.slider)
        flay.addWidget(lbl, alignment=Qt.AlignCenter)
        self.root_layout.addWidget(filter_box)


        bottom = QHBoxLayout()
        self.root_layout.addLayout(bottom, stretch=3)


        left_pane = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_pane.addWidget(self.table, stretch=1)

        self.result_table = QTableWidget()
        self._populate_result_table(self.assignment_df)
        left_pane.addWidget(self.result_table, stretch=2)

        bottom.addLayout(left_pane, stretch=3)

        self.bar_fig = Figure(figsize=(3, 2), dpi=100)
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_canvas = FigureCanvas(self.bar_fig)

        self.pred_fig = Figure(figsize=(3, 2), dpi=100)
        self.pred_ax = self.pred_fig.add_subplot(111)
        self.pred_canvas = FigureCanvas(self.pred_fig)

        self.pareto_fig = Figure(figsize=(3, 2), dpi=100)
        self.pareto_ax = self.pareto_fig.add_subplot(111)
        self.pareto_canvas = FigureCanvas(self.pareto_fig)


        self.hist_stack = QStackedLayout()
        self.hist_stack.addWidget(self.bar_canvas)      # 0 – Распределение
        self.hist_stack.addWidget(self.pred_canvas)     # 1 – Прогноз
        self.hist_stack.addWidget(self.pareto_canvas)   # 2 – Нагрузка

        hist_container = QWidget()  # wrapper for QStackedLayout
        hist_container.setLayout(self.hist_stack)

        btn_layout = QHBoxLayout()
        self.btn_group = QButtonGroup(self)
        buttons = [("Распределение", 0), ("Прогноз", 1), ("Нагрузка", 2)]
        for text, idx in buttons:
            btn = QPushButton(text)
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
            btn.clicked.connect(lambda _=False, x=idx: self.hist_stack.setCurrentIndex(x))
            self.btn_group.addButton(btn)
            btn_layout.addWidget(btn)

        right_pane = QVBoxLayout()
        right_pane.addLayout(btn_layout)
        right_pane.addWidget(hist_container, stretch=1)
        bottom.addLayout(right_pane, stretch=2)

        self.plot.pointSelected.connect(self.show_params)
        self._draw_future_load()  # initialise second histogram

        save_act = self.menuBar().addAction("Save PNG…")
        save_act.triggered.connect(self.save_png)

        load_act = self.menuBar().addAction("Load CSVs…")
        load_act.triggered.connect(self.choose_csv_dir)

        self.resize(1020, 720)

    def show_params(self, ind):
        ind = list(ind)
        counts = self.plot._build_counts(ind)
        top10 = sorted(counts[:10], key=lambda x: x[1], reverse=True)
        self.table.setRowCount(len(top10))
        for i, (emp, load) in enumerate(top10):
            self.table.setItem(i, 0, QTableWidgetItem(f"Сотрудник {emp}"))
            self.table.setItem(i, 1, QTableWidgetItem(str(load)))
        self.table.sortItems(1, Qt.DescendingOrder)
        self._draw_load_distribution(ind)
        loads, eff = _evaluation(
            ind,
            self.den_TNO,
            self.task_cat,
            self.task_prob,
            self.M,
            results=True,
            current_individ=self.current_individ,
        )
        task_counts = self._task_counts(ind)
        self._draw_future_load(task_counts, eff)
        self._draw_pareto_distribution(loads, eff)

    def _on_slider(self, value):
        """Handle efficiency slider."""
        self.plot.filter_eff(value)
        self.slider_label.setText(f"≥ {value} %")

    def _draw_load_distribution(self, individual):
        counts = self.plot._build_counts(individual)[:10]
        inspectors = [emp for emp, _ in counts]
        tasks = [load for _, load in counts]
        positions = range(len(inspectors))

        self.bar_ax.clear()
        self.bar_ax.bar(positions, tasks, color="#77B7F7")
        self.bar_ax.set_title("Нагрузка (кол-во задач)")
        self.bar_ax.set_xlabel("Inspector idx")
        self.bar_ax.set_ylabel("Tasks")
        self.bar_ax.set_xticks(positions)
        self.bar_ax.set_xticklabels(inspectors, rotation=45)
        self.bar_fig.tight_layout()
        self.bar_canvas.draw_idle()

    def _task_counts(self, individual):
        """Return task counts per inspector for a given individual."""
        counts = np.zeros(self.M, dtype=int)
        for idx in individual:
            counts[idx] += 1
        return counts

    def _draw_future_load(self, loads=None, eff=None):
        """Draw forecasted load distribution (task counts)."""

        if loads is None:
            loads = self.sorted_future_counts
            eff = self.future_eff
        else:
            loads = np.sort(loads)[::-1]

        self.pred_ax.clear()
        self.pred_ax.bar(
            self.inspectors_index,
            loads,
            width=0.8,
            label="Распределённая нагрузка",
            color="#5FA7F0",
            alpha=0.7,
        )
        self.pred_ax.set_xlabel("Инспекторы (отсортированы по числу задач)")
        self.pred_ax.set_ylabel("Количество задач")
        self.pred_ax.grid(True, alpha=0.3)
        self.pred_ax.legend()
        if eff is not None:
            self.pred_ax.set_title(f"Эффективность {round(eff, 2)}%")
        self.pred_fig.tight_layout()
        self.pred_canvas.draw_idle()

    def _draw_pareto_distribution(self, loads, eff):
        """Visualise comparison with baseline distribution."""

        sorted_load = np.sort(loads)[::-1]
        current_mean = loads.mean()
        std = loads.std()

        self.pareto_ax.clear()
        self.pareto_ax.bar(
            self.inspectors_index,
            self.sorted_future_load,
            width=0.8,
            label="Реальная будущая нагрузка",
            color="#FB4F00",
            alpha=0.7,
        )
        self.pareto_ax.bar(
            self.inspectors_index,
            sorted_load,
            width=0.8,
            label="Новая распределённая нагрузка",
            color="lightblue",
            alpha=0.7,
        )
        self.pareto_ax.axhline(y=current_mean, color="red", linestyle="--", alpha=0.7)
        self.pareto_ax.fill_between(
            self.inspectors_index,
            current_mean - std,
            current_mean + std,
            color="#4dff7f",
            alpha=0.2,
            label=f"±1 σ ({std:.2f} %)",
        )
        self.pareto_ax.set_xlabel("Инспекторы (отсортированы по итоговой нагрузке)")
        self.pareto_ax.set_ylabel("Взвешенная нагрузка (%)")
        self.pareto_ax.legend()
        self.pareto_ax.grid(True, alpha=0.3)
        self.pareto_ax.set_title(f"Эффективность {round(eff, 2)}%")
        self.pareto_fig.tight_layout()
        self.pareto_canvas.draw_idle()

    def _populate_result_table(self, df):
        self.result_table.setColumnCount(len(df.columns))
        self.result_table.setRowCount(len(df))
        self.result_table.setHorizontalHeaderLabels(df.columns.tolist())
        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                self.result_table.setItem(i, j, QTableWidgetItem(str(df.iloc[i, j])))
        self.result_table.resizeColumnsToContents()

    def save_png(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG (*.png)")
        if filename:
            self.plot.fig.savefig(filename, dpi=300)
            QMessageBox.information(self, "Сохранено", f"Изображение сохранено:\n{Path(filename).name}")

    def choose_csv_dir(self):
        """Select new directory with CSV files and reload data."""
        new_dir = QFileDialog.getExistingDirectory(self, "Выбрать папку с CSV", str(self.data_dir))
        if new_dir:
            # reload data and refresh UI
            self._load_data(Path(new_dir))

            # rebuild scatter plot
            self.root_layout.removeWidget(self.plot)
            self.plot.deleteLater()
            self.plot = ParetoCanvas(self.populations, self.pareto_front, self)
            self.root_layout.insertWidget(0, self.plot, stretch=4)
            self.plot.pointSelected.connect(self.show_params)

            # update tables and histograms
            self._populate_result_table(self.assignment_df)
            self._draw_future_load()


# QWidget { background-color: #001f3f; color: #e0f0ff; }
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(
        """
        QPushButton { background-color: #003366; color: white; padding: 4px 8px; border: none; border-radius: 4px; }
        QPushButton:hover { background-color: #004c8c; }
        QGroupBox { border: 1px solid #004c8c; margin-top: 6px; }
        QGroupBox:title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }
        """
    )
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
