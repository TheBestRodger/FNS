# table_assignment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QRegularExpression,
    Signal,
)
from PySide6.QtGui import QAction, QBrush, QClipboard
from PySide6.QtWidgets import (
    QHeaderView,
    QMenu,
    QTableView,
    QWidget,
    QFileDialog,
    QMessageBox,
)


# ----------------------------- УТИЛИТЫ ФОРМАТИРОВАНИЯ -------------------------

@dataclass
class ColumnFormat:
    fmt: Optional[str] = None
    align_right: bool = True
    nan_text: str = ""
    func: Optional[callable] = None


def _default_format_for_dtype(dtype: Any) -> ColumnFormat:
    if pd.api.types.is_numeric_dtype(dtype):
        return ColumnFormat(fmt="{:,.0f}", align_right=True)
    return ColumnFormat(fmt=None, align_right=False)


# --------------------------------- МОДЕЛЬ -------------------------------------

class PandasTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._df: pd.DataFrame = df.copy() if df is not None else pd.DataFrame()
        self._col_formats: Dict[str, ColumnFormat] = {}
        self._init_default_formats()

    # ---------- публичные API ----------
    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.copy() if df is not None else pd.DataFrame()
        self._init_default_formats()
        self.endResetModel()

    def set_column_formats(self, mapping: Dict[str, ColumnFormat]) -> None:
        self.beginResetModel()
        for col, cfg in mapping.items():
            if col in self._df.columns:
                self._col_formats[col] = cfg
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    # ---------- QAbstractTableModel ----------
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  
        if parent.isValid():
            return 0
        return 0 if self._df is None else len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int: 
        if parent.isValid():
            return 0
        return 0 if self._df is None else self._df.shape[1]

    def headerData( 
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> Any:
        if role != Qt.DisplayRole:
            return None
        if self._df is None:
            return None
        if orientation == Qt.Horizontal:
            try:
                return str(self._df.columns[section])
            except IndexError:
                return ""
        else:
            # индекс строки (1-based удобнее глазу)
            return str(section + 1)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or self._df is None:
            return None

        row = index.row()
        col = index.column()
        col_name = self._df.columns[col]
        value = self._df.iat[row, col]

        if role == Qt.DisplayRole:
            cfg = self._col_formats.get(col_name)
            if pd.isna(value):
                return cfg.nan_text if cfg else ""
            if cfg and cfg.func is not None:
                try:
                    return cfg.func(value)
                except Exception:
                    return str(value)
            if cfg and cfg.fmt:
                try:
                    # безопасное форматирование чисел/строк
                    return cfg.fmt.format(value)
                except Exception:
                    return str(value)
            return str(value)

        if role == Qt.TextAlignmentRole:
            cfg = self._col_formats.get(col_name)
            if cfg and cfg.align_right:
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)

        # пример лёгкой подсветки NaN
        if role == Qt.ForegroundRole and pd.isna(value):
            return QBrush(Qt.gray)

        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlags: 
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable  # read-only


    def _init_default_formats(self) -> None:
        self._col_formats.clear()
        if self._df is None or self._df.empty:
            return
        for col in self._df.columns:
            self._col_formats[col] = _default_format_for_dtype(self._df[col].dtype)


# -------------------------------- ПРОКСИ МОДЕЛЬ СОРТИРОВКИ И ФИЛЬТРА -------------------------------

class NumericAwareSortProxy(QSortFilterProxyModel):
    """
    Прокси добавляет:
      - корректную сортировку чисел, даже если они отформатированы как строки,
      - простой фильтр по подстроке (регистронезависимый).
    """
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._filter_text = ""
        self.setDynamicSortFilter(True)
        self.setFilterCaseSensitivity(Qt.CaseInsensitive)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        model = self.sourceModel()
        if not isinstance(model, PandasTableModel):
            return super().lessThan(left, right)

        df = model.dataframe()
        col = left.column()
        col_name = df.columns[col]
        lv = df.iat[left.row(), col]
        rv = df.iat[right.row(), col]

        # Пытаемся сравнить как числа
        try:
            lvn = float(lv) if not pd.isna(lv) else float("-inf")
            rvn = float(rv) if not pd.isna(rv) else float("-inf")
            return lvn < rvn
        except Exception:
            # иначе — как строки (с учётом None/NaN)
            ls = "" if pd.isna(lv) else str(lv)
            rs = "" if pd.isna(rv) else str(rv)
            return ls < rs

    def setFilterText(self, text: str) -> None:
        self._filter_text = text.strip()
        if self._filter_text:
            # простой contains по всем столбцам
            regex = QRegularExpression(self._filter_text.replace("*", ".*"))
            self.setFilterRegularExpression(regex)
        else:
            self.setFilterRegularExpression(QRegularExpression())

    def filterAcceptsRow(
        self, source_row: int, source_parent: QModelIndex
    ) -> bool:
        if not self._filter_text:
            return True
        model = self.sourceModel()
        if not isinstance(model, PandasTableModel):
            return True
        df = model.dataframe()
        for c in range(df.shape[1]):
            val = df.iat[source_row, c]
            if pd.isna(val):
                continue
            if self._filter_text.lower() in str(val).lower():
                return True
        return False


# -------------------------------- ВИДЖЕТ ТАБЛИЦЫ ------------------------------

class AssignmentTable(QTableView):
    """
    assignment_df:
      - set_dataframe(df)
      - сортировка по клику на заголовке
      - авто-подгон ширины столбцов
      - контекстное меню: копировать, экспорт CSV
    """
    requestCopy = Signal()
    requestExport = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._model = PandasTableModel()
        self._proxy = NumericAwareSortProxy(self)
        self._proxy.setSourceModel(self._model)
        self.setModel(self._proxy)

        # Внешний вид
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        self.setSortingEnabled(True)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.setWordWrap(False)

        # Контекстное меню
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_context_menu)
        self.requestCopy.connect(self.copy_selection_to_clipboard)
        self.requestExport.connect(self.export_to_csv_dialog)

    # ---------- Публичные API ----------
    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._model.set_dataframe(df)
        self._autosize_columns()

    def set_column_formats(self, mapping: Dict[str, ColumnFormat]) -> None:
        self._model.set_column_formats(mapping)
        self._autosize_columns()

    def dataframe(self) -> pd.DataFrame:
        return self._model.dataframe()

    def copy_selection_to_clipboard(self) -> None:
        """Копировать выделенные строки в буфер обмена (как TSV)."""
        sel = self.selectionModel().selectedRows()
        if not sel:
            return

        # Берём видимые строки через proxy
        rows = sorted([ix.row() for ix in sel])
        df = self._model.dataframe()

        # Преобразуем индексы proxy -> source
        src_rows = [self._proxy.mapToSource(self._proxy.index(r, 0)).row() for r in rows]
        sub = df.iloc[src_rows, :]

        # Форматируем значения так же, как в DisplayRole
        out = self._format_dataframe_for_copy(sub)
        text = out.to_csv(sep="\t", index=False)
        QApplication = type(self.parent()).__mro__[-2]  # lazy import safety
        QApplication.instance().clipboard().setText(text)

    def export_to_csv_dialog(self) -> None:
        """Экспорт видимых (отфильтрованных) строк в CSV."""
        filename, _ = QFileDialog.getSaveFileName(self, "Сохранить как CSV", "", "CSV (*.csv)")
        if not filename:
            return
        df = self._visible_dataframe()
        try:
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            QMessageBox.information(self, "Экспорт завершён", f"Сохранено: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", str(e))

    # ---------- Вспомогательные ----------
    def _autosize_columns(self) -> None:
        self.resizeColumnsToContents()
        # Немного добавочного пространства (для заголовков)
        extra = 12
        for c in range(self.model().columnCount()):
            w = self.columnWidth(c)
            self.setColumnWidth(c, w + extra)

    def _open_context_menu(self, pos) -> None:
        menu = QMenu(self)
        act_copy = QAction("Копировать выделенные строки", self)
        act_copy.triggered.connect(self.requestCopy)
        menu.addAction(act_copy)

        act_export = QAction("Экспорт видимых в CSV…", self)
        act_export.triggered.connect(self.requestExport)
        menu.addAction(act_export)

        menu.exec(self.viewport().mapToGlobal(pos))

    def _visible_dataframe(self) -> pd.DataFrame:
        """Текущая «видимая» таблица (сорт/фильтр уже применены)."""
        src = self._model.dataframe()
        if src.empty:
            return src

        rows = []
        for r in range(self._proxy.rowCount()):
            src_row = self._proxy.mapToSource(self._proxy.index(r, 0)).row()
            rows.append(src_row)
        return src.iloc[rows, :].reset_index(drop=True)

    def _format_dataframe_for_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применить те же правила форматирования, что и в DisplayRole."""
        out = pd.DataFrame(index=df.index)
        for col in df.columns:
            cfg = self._model._col_formats.get(col)  # доступ к конфигу
            series = df[col]
            if cfg and cfg.func:
                out[col] = series.map(lambda v: "" if pd.isna(v) else str(cfg.func(v)))
            elif cfg and cfg.fmt:
                def _fmt(v):
                    if pd.isna(v):
                        return cfg.nan_text
                    try:
                        return cfg.fmt.format(v)
                    except Exception:
                        return str(v)
                out[col] = series.map(_fmt)
            else:
                out[col] = series.map(lambda v: "" if pd.isna(v) else str(v))
        return out
