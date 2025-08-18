from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Tuple
from style import NO_CODE
from app.core.deap_optim import (
    get_results,
    get_assignment_table,
    _get_dataframe,
    _filter_by_no,
    _start_data_prep,
    _evaluation,
)
from app.models.metrics import _task_counts
from app.models.state import DataState

class DataService:
    def load(self, data_dir: Path) -> DataState:
        inspector_df, new_df, inwork_df = _get_dataframe(data_dir)
        populations, pareto = get_results(data_dir)
        assignment = get_assignment_table(data_dir)

        curr_inspectors, current_df = _filter_by_no(NO_CODE, inspector_df, inwork_df)
        current_df = current_df.merge(
            curr_inspectors.reset_index()[["Инспектор, сменивший статус","Inspector index"]],
            how="left", on="Инспектор, сменивший статус"
        )
        current_individ = current_df["Inspector index"].tolist()

        no_inspectors, no_df = _filter_by_no(NO_CODE, inspector_df, new_df)
        task_prob, N, M, task_cat, den_TNO = _start_data_prep(no_inspectors, no_df, current_df)

        fut = no_df[["Статус РСЗ","Тип","Потенциальный ущерб, руб","ИНН НП","Инспектор, сменивший статус"]].copy()
        fut = fut.merge(
            no_inspectors.reset_index()[["Инспектор, сменивший статус","Inspector index"]],
            how="left", on="Инспектор, сменивший статус"
        )
        future_individ = fut["Inspector index"].fillna(-1).astype(int).tolist()
        future_load, future_eff = _evaluation(future_individ, den_TNO, task_cat, task_prob, M,
                                             results=True, current_individ=current_individ)

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
            task_prob=task_prob, N=N, M=M, task_cat=task_cat, den_TNO=den_TNO,
            future_individ=future_individ,
            future_counts_sorted=future_counts_sorted,
            future_eff=future_eff,
            future_load_sorted=future_load_sorted,
            inspectors_index=inspectors_index
        )
