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
from core.deap_optim import (
    _get_dataframe,
    _get_new_type_data,
    _create_inspetors_df,
    get_results,
    get_assignment_table,
)
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
            exp_inspectors_df, df, auto_stats_df = _get_dataframe(self.data_dir)
            new_tasks_df, inwork_tasks_df, finish_tasks_df = _get_new_type_data(df, auto_stats_df)

            def _collect_codes(series: pd.Series) -> set[int]:
                return set(pd.to_numeric(series, errors="coerce").dropna().astype(int))

            task_tnos = _collect_codes(new_tasks_df["Код НО инспектора, сменившего стат"])
            inwork_tnos = _collect_codes(inwork_tasks_df["Код НО инспектора, сменившего стат"])
            df_tnos = _collect_codes(df["Код НО инспектора, сменившего стат"])
            available_tnos = sorted(task_tnos & inwork_tnos & df_tnos)
            if self.no_code not in available_tnos:
                available_tnos = sorted({*available_tnos, self.no_code})

            self.progress.emit(25, "Чтение результатов оптимизации...")
            populations, pareto_front = get_results(
                data_dir=self.data_dir,
                no_code=self.no_code,
            )
            assignment_df = get_assignment_table(
                data_dir=self.data_dir,
                no_code=self.no_code,
            )

            self.progress.emit(45, "Подготовка текущих данных...")
            no_df, inwork_no_df, new_no_df = _filter_by_no(
                self.no_code, df, inwork_tasks_df, new_tasks_df
            )
            main_inspectors_df, inspectors_direction_df = _create_inspetors_df(
                no_df, exp_inspectors_df, finish_tasks_df
            )

            task_prob, N, M, task_cat, task_dir, den_TNO, maps = _start_data_prep(
                new_no_df, inspectors_direction_df, inwork_no_df
            )

            inspector_to_index = maps.get("inspector_id_to_index", {})

            current_df = inwork_no_df.copy()
            current_df["Inspector index"] = current_df[
                "Инспектор, сменивший статус"
            ].map(inspector_to_index)
            current_individ = safe_series_to_int_list(
                current_df["Inspector index"], default=-1
            )

            self.progress.emit(60, "Подготовка новых заданий...")
            future_df = new_no_df[[
                "Статус РСЗ",
                "Тип",
                "Потенциальный ущерб, руб",
                "ИНН НП",
                "Инспектор, сменивший статус",
            ]].copy()
            future_df["Inspector index"] = future_df[
                "Инспектор, сменивший статус"
            ].map(inspector_to_index)
            future_individ = safe_series_to_int_list(
                future_df["Inspector index"], default=-1
            )

            self.progress.emit(80, "Оценка распределения...")
            if M > 0:
                future_load, future_eff = evaluate_distribution(
                    individual=future_individ,
                    den_TNO=den_TNO,
                    task_cat=task_cat,
                    task_dir=task_dir,
                    task_prob=task_prob,
                    M=M,
                    current_individ=current_individ,
                    evaluation_func=_evaluation,
                )
            else:
                # нет инспекторов для выбранного ТНО — возвращаем пустые метрики
                future_load = np.array([], dtype=float)
                future_eff = None

            self.progress.emit(90, "Формирование состояния...")
            future_cnts_sorted = sort_desc(task_counts(future_individ, M))
            future_load_sorted = sort_desc(future_load)
            inspectors_index = list(range(M))

            state = DataState(
                data_dir=self.data_dir,
                populations=populations,
                pareto_front=pareto_front,
                assignment_df=assignment_df,
                current_inspectors=main_inspectors_df,
                current_individ=current_individ,
                task_prob=task_prob,
                N=N,
                M=M,
                task_cat=task_cat,
                task_dir=task_dir,
                den_TNO=den_TNO,
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
                N=0,
                M=0,
                task_cat=np.array([]),
                task_dir=np.array([]),
                den_TNO={},
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
