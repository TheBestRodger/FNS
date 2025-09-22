# metrics.py
from __future__ import annotations

from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd


def task_counts(individual: Sequence[int], M: int) -> np.ndarray:
    """
    Подсчитать число задач на каждого инспектора.
    Безопасно обрабатывает неверные индексы.
    """
    counts = np.zeros(M, dtype=int)
    for idx in individual:
        if isinstance(idx, (int, np.integer)) and 0 <= idx < M:
            counts[idx] += 1
    return counts


def sort_desc(arr: np.ndarray) -> np.ndarray:
    """Отсортировать numpy-вектор по убыванию, не модифицируя исходник."""
    if arr is None or arr.size == 0:
        return np.array([], dtype=arr.dtype if arr is not None else float)
    return np.sort(arr)[::-1]


def safe_series_to_int_list(series: pd.Series, default: int = -1) -> list[int]:
    """
    Преобразовать Series с возможными NaN к списку int.
    Все NaN -> default (по умолчанию -1), затем к int.
    """
    return series.astype("Int64").fillna(default).astype(int).tolist()


def evaluate_distribution(
    individual: Sequence[int],
    *,
    den_TNO,
    task_cat: np.ndarray,
    task_dir: np.ndarray,
    task_prob: np.ndarray,
    M: int,
    current_individ: Sequence[int],
    evaluation_func,
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Обёртка над вашей функцией оценки (например, _evaluation из проекта).
    Возвращает (loads, eff), где
      - loads: массив «взвешенной нагрузки» (любой вашей метрики),
      - eff:   эффективность (float | None).
    """
    loads, eff = evaluation_func(
        list(individual),
        den_TNO,
        task_cat,
        task_dir,
        task_prob,
        M,
        results=True,
        current_individ=list(current_individ),
    )
    loads = np.asarray(loads)
    return loads, eff
