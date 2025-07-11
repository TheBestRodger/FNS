import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QSlider, QLabel, QTableWidget,
    QTableWidgetItem, QSizePolicy, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure


from deap_optim import get_results     

populations, pareto_front = get_results() 


class ParetoCanvas(QWidget):
    pointSelected = Signal(object)        
    
    def __init__(self, pops, pfront, parent=None):
        import numpy as np
        super().__init__(parent)
        self.loads, self.effs, self.inds = pops
        self.loads = np.asarray(self.loads, dtype=float)
        self.effs  = np.asarray(self.effs,  dtype=float)
        #self.loads, self.effs, self.inds = pops
        pf_loads, pf_effs, self.pf_inds = pfront

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)


        # lay = QVBoxLayout(self)
        # lay.setContentsMargins(0, 0, 0, 0)
        # lay.addWidget(self.canvas)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.canvas.mpl_connect("pick_event",  self._on_pick)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.scatter = self.ax.scatter(
            self.loads, self.effs,
            s=30, c="skyblue", alpha=.65, picker=True, label="All"
        )
        # парето
        self.pf_scatter = self.ax.scatter(
            pf_loads, pf_effs,
            s=70, color="crimson", label="Pareto Front", picker=True  # только точки
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
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        # z = np.polyfit(self.loads, self.effs, deg=1)
        # p = np.poly1d(z)
        # x_line = np.linspace(min(self.loads), max(self.loads), 100)
        # self.ax.plot(x_line, p(x_line), color="black", lw=1.5, label="Trend")
        self.ax.set_xlabel("Max load, %")
        self.ax.set_ylabel("Efficiency, %")
        self.ax.legend()
        self.fig.tight_layout()

        self.canvas.mpl_connect("pick_event", self._on_pick)

    # фильтр по min-Efficiency
    def filter_eff(self, min_eff):
        mask = self.effs >= min_eff
        self.scatter.set_offsets([[x, y] for x, y, m in zip(self.loads, self.effs, mask) if m])
        self.canvas.draw_idle()

    def _build_counts(self, individual):
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
                text = "\n".join(f"{emp}: {load}" for emp, load in items)
                self.annot.xy = self.pf_scatter.get_offsets()[idx]
                self.annot.set_text(text)
                self.annot.set_visible(True)
                self.canvas.draw_idle()
                return
        if vis:
            self.annot.set_visible(False)
            self.canvas.draw_idle()

    def _on_pick(self, ev):
        idx = ev.ind[0]
        if ev.artist is self.pf_scatter:
            data = self.pf_inds[idx]
        else:
            data = self.inds[idx]
        self.pointSelected.emit(data)
    def _on_scroll(self, event):

        if event.xdata is None or event.ydata is None:
            return

        base_scale = 1.2           
        scale = 1 / base_scale if event.button == "up" else base_scale

        ax = self.ax
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()

        x_range = (x_right - x_left) * scale
        y_range = (y_top   - y_bottom) * scale

        relx = (event.xdata - x_left)   / (x_right - x_left)
        rely = (event.ydata - y_bottom) / (y_top   - y_bottom)

        ax.set_xlim(event.xdata - relx * x_range,
                    event.xdata + (1 - relx) * x_range)
        ax.set_ylim(event.ydata - rely * y_range,
                    event.ydata + (1 - rely) * y_range)

        self.canvas.draw_idle()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pareto-Front Viewer")

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # график
        self.plot = ParetoCanvas(populations, pareto_front, self)
        vbox.addWidget(self.plot, stretch=4)
        bottom = QHBoxLayout()
        # слайдер эффективности
        controls = QGroupBox("Фильтр эффективности")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        lbl = QLabel("≥ 0 %")
        slider.valueChanged.connect(lambda v: (
            self.plot.filter_eff(v),
            lbl.setText(f"≥ {v} %")
        ))
        lay = QVBoxLayout(controls)
        lay.addWidget(slider)
        lay.addWidget(lbl, alignment=Qt.AlignCenter)
        vbox.addWidget(controls)


        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom.addWidget(self.table, stretch=3)

        self.bar_fig = Figure(figsize=(3, 2), dpi=100)
        self.bar_ax = self.bar_fig.add_subplot(111)
        self.bar_canvas = FigureCanvas(self.bar_fig)
        bottom.addWidget(self.bar_canvas, stretch=2)

        vbox.addLayout(bottom, stretch=2)
        self.plot.pointSelected.connect(self.show_params)

        save_act = self.menuBar().addAction("Save PNG…")
        save_act.triggered.connect(self.save_png)

        self.resize(900, 700)
        
    def show_params(self, ind):
        counts = self.plot._build_counts(ind)
        top = counts[:10]
        self.table.setRowCount(len(top))
        for i, (emp, load) in enumerate(top):
            self.table.setItem(i, 0, QTableWidgetItem(f"Сотрудник {emp}"))
            self.table.setItem(i, 1, QTableWidgetItem(str(load)))
        self._draw_load_distribution(ind)

    def save_png(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить график", "", "PNG (*.png)")
        if fn:
            self.plot.fig.savefig(fn, dpi=300)
            QMessageBox.information(self, "Сохранено", f"Изображение сохранено:\n{Path(fn).name}")
    def _draw_load_distribution(self, individual):
        # individual — список индексов инспекторов
        counts = self.plot._build_counts(individual)[:10]
        inspectors = [emp for emp, _ in counts]
        tasks = [load for _, load in counts]

        self.bar_ax.clear()
        self.bar_ax.bar(inspectors, tasks, color="#ffa600")
        self.bar_ax.set_title("Нагрузка (кол-во задач)")
        self.bar_ax.set_xlabel("Inspector idx")
        self.bar_ax.set_ylabel("Tasks")
        self.bar_ax.tick_params(axis='x', rotation=45)
        self.bar_fig.tight_layout()
        self.bar_canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
