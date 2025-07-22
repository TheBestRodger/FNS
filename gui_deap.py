import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QFileDialog,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

from deap_optim import get_results, get_assignment_table

populations, pareto_front = get_results()
assignment_df = get_assignment_table()


class ParetoCanvas(QWidget):
    """Scatter‑plot с точками популяции и Парето‑фронтом."""

    pointSelected = Signal(object)  # → список индексов инспекторов

    def __init__(self, pops, pfront, parent=None):
        import numpy as np

        super().__init__(parent)
        self.loads, self.effs, self.inds = pops
        self.loads = np.asarray(self.loads, dtype=float)
        self.effs = np.asarray(self.effs, dtype=float)

        pf_loads, pf_effs, self.pf_inds = pfront

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

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

        self.annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        self.annot.set_visible(False)

        self.ax.set_xlabel("Max load, %")
        self.ax.set_ylabel("Efficiency, %")
        self.ax.legend()
        self.fig.tight_layout()

        self.canvas.mpl_connect("pick_event", self._on_pick)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

        
    @staticmethod
    def _build_counts(individual):
        counts = {}
        for idx in individual:
            counts[idx] = counts.get(idx, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

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

    # Pick
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

    def filter_eff(self, min_eff):
        mask = self.effs >= min_eff
        self.scatter.set_offsets([[x, y] for x, y, m in zip(self.loads, self.effs, mask) if m])
        self.canvas.draw_idle()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pareto‑Front Viewer")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.plot = ParetoCanvas(populations, pareto_front, self)
        root.addWidget(self.plot, stretch=4)


        filter_box = QGroupBox("Фильтр эффективности")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        lbl = QLabel("≥ 0 %")
        slider.valueChanged.connect(lambda v: (self.plot.filter_eff(v), lbl.setText(f"≥ {v} %")))
        flay = QVBoxLayout(filter_box)
        flay.addWidget(slider)
        flay.addWidget(lbl, alignment=Qt.AlignCenter)
        root.addWidget(filter_box)


        bottom = QHBoxLayout()


        left_pane = QVBoxLayout()

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        left_pane.addWidget(self.table, stretch=1)

        self.result_table = QTableWidget()
        self._populate_result_table(assignment_df)
        left_pane.addWidget(self.result_table, stretch=2)

        bottom.addLayout(left_pane, stretch=3)


        self.bar_fig = Figure(figsize=(3, 2), dpi=100)
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_canvas = FigureCanvas(self.bar_fig)
        bottom.addWidget(self.bar_canvas, stretch=2)

        root.addLayout(bottom, stretch=3)

 
        self.plot.pointSelected.connect(self.show_params)

        save_act = self.menuBar().addAction("Save PNG…")
        save_act.triggered.connect(self.save_png)

        self.resize(900, 700)

    def show_params(self, ind):
        counts = self.plot._build_counts(ind)
        top10 = sorted(counts[:10], key=lambda x: x[1], reverse=True)
        self.table.setRowCount(len(top10))
        for i, (emp, load) in enumerate(top10):
            self.table.setItem(i, 0, QTableWidgetItem(f"Сотрудник {emp}"))
            self.table.setItem(i, 1, QTableWidgetItem(str(load)))
        self.table.sortItems(1, Qt.DescendingOrder)
        self._draw_load_distribution(ind)

    def _draw_load_distribution(self, individual):
        counts = self.plot._build_counts(individual)[:10]
        inspectors = [emp for emp, _ in counts]
        tasks = [load for _, load in counts]
        positions = range(len(inspectors))

        self.bar_ax.clear()
        self.bar_ax.bar(positions, tasks, color="#ffa600")
        self.bar_ax.set_title("Нагрузка (кол-во задач)")
        self.bar_ax.set_xlabel("Inspector idx")
        self.bar_ax.set_ylabel("Tasks")
        self.bar_ax.set_xticks(positions)
        self.bar_ax.set_xticklabels(inspectors, rotation=45)
        self.bar_fig.tight_layout()
        self.bar_canvas.draw_idle()

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



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
