# workers/loader.py
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

import numpy as np
import pandas as pd

from models.state import DataState
from models.metrics import (
    task_counts,
    sort_desc,
    safe_series_to_int_list,
    evaluate_distribution,
)

# Доменные функции/константы — подправьте пути импорта под ваш проект
from core.deap_optim import _get_dataframe, get_results, get_assignment_table
from core.deap_optim import _filter_by_no, _start_data_prep
from core.deap_optim import _evaluation  # ваша оценочная функция


class LoadWorker(QObject):
    """
    Загружает и подготавливает данные в отдельном потоке.
    Сигналы:
      - progress(int, str)  — % и подпись этапа
      - finished(DataState) — успешный результат
      - error(str)          — текст ошибки
    """
    progress = Signal(int, str)
    finished = Signal(DataState)
    error = Signal(str)

    def __init__(self, data_dir: Path, no_code: int) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.no_code = no_code

    def run(self) -> None:
        try:
            self.progress.emit(5, "Чтение CSV...")
            inspector_df, new_df, inwork_df = _get_dataframe(self.data_dir)
            counts = new_df['Код НО инспектора, сменившего стат'].value_counts()
            available_tnos = counts[counts > 1].index.astype(int).tolist()

            self.progress.emit(25, "Чтение результатов оптимизации...")
            populations, pareto_front = get_results(data_dir=self.data_dir, no_code=self.no_code)
            assignment_df = get_assignment_table(data_dir=self.data_dir, no_code=self.no_code)

            self.progress.emit(45, "Подготовка текущих данных...")
            current_inspectors, current_df = _filter_by_no(self.no_code, inspector_df, inwork_df)
            current_df = current_df.merge(
                current_inspectors.reset_index()[["Инспектор, сменивший статус", "Inspector index"]],
                how="left", on="Инспектор, сменивший статус"
            )
            current_individ = safe_series_to_int_list(current_df["Inspector index"], default=-1)

            self.progress.emit(60, "Подготовка новых заданий...")
            no_inspectors, no_df = _filter_by_no(self.no_code, inspector_df, new_df)
            task_prob, N, M, task_cat, den_TNO = _start_data_prep(no_inspectors, no_df, current_df)

            future_df = no_df[[
                "Статус РСЗ", "Тип", "Потенциальный ущерб, руб",
                "ИНН НП", "Инспектор, сменивший статус"
            ]].copy()
            future_df = future_df.merge(
                no_inspectors.reset_index()[["Инспектор, сменивший статус", "Inspector index"]],
                how="left", on="Инспектор, сменивший статус"
            )
            future_individ = safe_series_to_int_list(future_df["Inspector index"], default=-1)

            self.progress.emit(80, "Оценка распределения...")
            future_load, future_eff = evaluate_distribution(
                individual=future_individ,
                den_TNO=den_TNO,
                task_cat=task_cat,
                task_prob=task_prob,
                M=M,
                current_individ=current_individ,
                evaluation_func=_evaluation,
            )

            self.progress.emit(90, "Формирование состояния...")
            future_cnts_sorted = sort_desc(task_counts(future_individ, M))
            future_load_sorted = sort_desc(future_load)
            inspectors_index = list(range(M))

            state = DataState(
                data_dir=self.data_dir,
                populations=populations,
                pareto_front=pareto_front,
                assignment_df=assignment_df,
                current_inspectors=current_inspectors,
                current_individ=current_individ,
                task_prob=task_prob,
                N=N, M=M, task_cat=task_cat, den_TNO=den_TNO,
                future_individ=future_individ,
                future_counts_sorted=future_cnts_sorted,
                future_eff=future_eff,
                future_load_sorted=future_load_sorted,
                inspectors_index=inspectors_index,
                available_tnos=available_tnos,
            )

            self.progress.emit(100, "Готово")
            self.finished.emit(state)

        except FileNotFoundError:
            # Вернём пустое состояние вместо падения
            empty = DataState(
                data_dir=self.data_dir,
                populations=([], [], tuple()),
                pareto_front=([], [], tuple()),
                assignment_df=pd.DataFrame(),
                current_inspectors=pd.DataFrame(),
                current_individ=[],
                task_prob=np.array([]),
                N=0, M=0, task_cat=np.array([]), den_TNO={},
                future_individ=[],
                future_counts_sorted=np.array([], dtype=int),
                future_eff=None,
                future_load_sorted=np.array([], dtype=float),
                inspectors_index=[],
                available_tnos=[],
            )
            self.progress.emit(100, "Пустые данные")
            self.finished.emit(empty)
        except Exception as e:
            self.error.emit(str(e))
