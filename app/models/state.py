import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataState:
    data_dir: Path
    populations: tuple
    pareto_front: tuple
    assignment_df: pd.DataFrame

    current_inspectors: pd.DataFrame
    current_individ: list[int]

    task_prob: np.ndarray
    N: int
    M: int
    task_cat: np.ndarray
    task_dir: np.ndarray
    den_TNO: dict

    future_individ: list[int]
    future_counts_sorted: np.ndarray
    future_eff: float | None
    future_load_sorted: np.ndarray
    inspectors_index: list[int]
    available_tnos: list[int] = field(default_factory=list)