from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Tuple
from style import NO_CODE
from app.core.deap_optim import (
    get_results,
    get_assignment_table,
    _get_dataframe,
    _get_new_type_data,
    _create_inspetors_df,
    _filter_by_no,
    _start_data_prep,
    _evaluation,
)
from app.models.metrics import _task_counts
from app.models.state import DataState

class DataService:
    def load(self, data_dir: Path) -> DataState:
        exp_inspectors_df, df, auto_stats_df = _get_dataframe(data_dir)
        new_tasks_df, inwork_tasks_df, finish_tasks_df = _get_new_type_data(df, auto_stats_df)

        no_df, inwork_no_df, new_no_df = _filter_by_no(NO_CODE, df, inwork_tasks_df, new_tasks_df)
        curr_inspectors, inspectors_direction_df = _create_inspetors_df(
            no_df, exp_inspectors_df, finish_tasks_df
        )

        populations, pareto = get_results(data_dir=data_dir, no_code=NO_CODE)
        assignment = get_assignment_table(
            data_dir=data_dir,
            no_code=NO_CODE,
            pareto_front=pareto,
            new_tasks_df=new_no_df,
            inspectors_direction_df=inspectors_direction_df,
        )

        task_prob, N, M, task_cat, task_dir, den_TNO, maps = _start_data_prep(
            new_no_df, inspectors_direction_df, inwork_no_df
        )

        inspector_to_index = maps.get("inspector_id_to_index", {})

        current_df = inwork_no_df.copy()
        current_df["Inspector index"] = current_df[
            "Инспектор, сменивший статус"
        ].map(inspector_to_index)
        current_individ = current_df["Inspector index"].astype("Int64").fillna(-1).astype(int).tolist()

        fut = new_no_df[[
            "Статус РСЗ",
            "Тип",
            "Потенциальный ущерб, руб",
            "ИНН НП",
            "Инспектор, сменивший статус",
        ]].copy()
        fut["Inspector index"] = fut[
            "Инспектор, сменивший статус"
        ].map(inspector_to_index)
        future_individ = fut["Inspector index"].astype("Int64").fillna(-1).astype(int).tolist()
        future_load, future_eff = _evaluation(
            future_individ,
            den_TNO,
            task_cat,
            task_dir,
            task_prob,
            M,
            results=True,
            current_individ=current_individ,
        )

        future_counts_sorted = np.sort(_task_counts(future_individ, M))[::-1]
        future_load_sorted   = np.sort(future_load)[::-1]
        inspectors_index = list(range(M))

        return DataState(
            data_dir=data_dir,
            populations=populations,
            pareto_front=pareto,
            assignment_df=assignment,
            current_inspectors=curr_inspectors,
            current_individ=current_individ,
            task_prob=task_prob,
            N=N,
            M=M,
            task_cat=task_cat,
            task_dir=task_dir,
            den_TNO=den_TNO,
            future_individ=future_individ,
            future_counts_sorted=future_counts_sorted,
            future_eff=future_eff,
            future_load_sorted=future_load_sorted,
            inspectors_index=inspectors_index
        )
