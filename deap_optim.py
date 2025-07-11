"""
pareto_data.py  ─ Library to run the inspector-task multi-objective
optimisation and provide only the artefacts that нужны для визуализации:

    • populations   – (loads, efficiencies, individuals)
    • pareto_front  – (loads_pf, efficiencies_pf, pareto_individuals)

The heavy GA расчёт выполняется один раз, результат кэшируется в pickle
рядом с модулем.  Повторный импорт или вызов `get_results()` мгновенный.

Usage
-----
>>> from pareto_data import get_results
>>> pops, pf = get_results()               # берём готовые данные
>>> x_all, y_all, _ = pops                 # распаковка, как в ноутбуке

Если нужно пересчитать (например, изменили CSV или настройки алгоритма):
>>> pops, pf = get_results(recompute=True)

Optional
~~~~~~~~
`data_path`  – путь к *Automated_RSZ_distribution_enc.csv*; если не указан,
               ожидается файл рядом с этим модулем.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Sequence
from functools import partial

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random

# ---------------------------------------------------------------------------
#  Files & caching
# ---------------------------------------------------------------------------
_CACHE_FILE = Path(__file__).with_suffix(".pkl")
_DEFAULT_DATA = Path(__file__+"data/").with_name("Automated_RSZ_distribution_enc.csv")

# ---------------------------------------------------------------------------
#  Core pipeline (directly перенесено из ноутбука)
# ---------------------------------------------------------------------------

def _prepare_dataframe(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Load & preprocess source CSV.  Returns (inspectors_df, new_df, distribution_new_tasks)."""
    df = pd.read_csv(csv_path, sep=';')

    # -- Чистка / типы -------------------------------------------------------
    df = df.drop(index=df[(df['Инспектор, сменивший статус'].isna()) & (df['Статус РСЗ'] != 'Новое')].index)
    df.loc[df['Код НО инспектора, сменившего стат']=='000n', 'Код НО инспектора, сменившего стат'] = 0
    df.drop_duplicates(inplace=True)
    df['Дата выявления'] = pd.to_datetime(df['Дата выявления'], format='%Y-%m-%d %H:%M:%S')
    df['Дата изменения статуса РСЗ'] = pd.to_datetime(df['Дата изменения статуса РСЗ'], format='%Y-%m-%d %H:%M:%S')
    df['Код НО инспектора, сменившего стат'] = df['Код НО инспектора, сменившего стат'].astype(int)
    df['Тип'] = df['Тип'].replace({
        'Риск_долго': 'RISK_LONG',
        'Риск_быстро': 'RISK_SHORT',
        'Схема': 'SCHEMA',
        'Задание': 'TASK'
    }, regex=True)

    # -- Подсчёт Auto_Stats (зашито как в ноутбуке) --------------------------
    Auto_Stats = pd.DataFrame({
        'Новое':                                                   [0, 0, 0, 0],
        'В работе':                                                [0, 0, 0, 0],
        'Нарушение подтверждено. Устранено уточнением сведений':   [100, 86, 46, 0],
        'Нарушение не подтверждено':                               [50, 43, 23 , 0],
        'Нарушение подтверждено. Доказано':                        [75, 65, 35, 0],
        'Установлены отрицательные изменения':                     [0, 0, 0, 0],
        'Нарушение подтверждено. Не доказано':                     [40, 34, 18, 0],
        'Не доказано':                                             [40, 34, 18, 0],
        'Декларация аннулирована':                                 [60, 52, 28, 0],
        'Разакцептование':                                         [0, 0, 0, 0],
        'Архив':                                                   [0, 0, 0, 0],
        'Ожидает обработки':                                       [0, 0, 0, 0],
        'Разакцептован':                                           [0, 0, 0, 0],
        'На проверке':                                             [0, 0, 0, 0],
        'Выполнено':                                               [0, 0, 0, 3],
        'Не выполнено':                                            [0, 0, 0, 0],
    }).transpose()
    Auto_Stats.columns = ['SCHEMA', 'RISK_LONG', 'RISK_SHORT', 'TASK']
    all_scores = [Auto_Stats.loc[stat, df.iloc[i, 1]] for i, stat in enumerate(df['Статус РСЗ'])]
    df['Score'] = all_scores
    # -- Inspectors table ----------------------------------------------------
    N_ended_works = df[df['Score'] > 0].groupby('Инспектор, сменивший статус')['Score'].count()
    inspectors_df = pd.DataFrame(df.groupby('Инспектор, сменивший статус')['Код НО инспектора, сменившего стат'].last())
    inspectors_efficiency = (
        df[df['Score'] > 0]
          .groupby('Инспектор, сменивший статус')['Score']
          .sum() / N_ended_works / 100
    )
    inspectors_df = pd.DataFrame(df.groupby('Инспектор, сменивший статус')['Код НО инспектора, сменившего стат'].last())
    inspectors_efficiency = df[df['Score'] > 0].groupby('Инспектор, сменивший статус')['Score'].sum()/N_ended_works/100
    inspectors_df = inspectors_df.merge(inspectors_efficiency, how='left', left_index=True, right_index=True)
    inspectors_df.fillna(0, inplace=True)
    inspectors_df['Experience'] = [np.random.random()/2 for i in range(len(inspectors_df))]
    inspectors_df['Qualification'] = inspectors_df['Score'] + inspectors_df['Experience']

    mask = inspectors_df['Qualification'] >= 0.71

    # Используем np.where для каждого столбца
    inspectors_df['p_SCHEMA'] = np.where(mask, 0.5, 0)
    inspectors_df['p_RISK_LONG'] = np.where(mask, 0.5, 0)
    inspectors_df['p_RISK_SHORT'] = np.where(mask, 0, 0.5)
    inspectors_df['p_TASK'] = np.where(mask, 0, 0.5)

    df.sort_values(['№ схемы/риска', 'Дата изменения статуса РСЗ'])
    grouped_df = df.groupby('№ схемы/риска')[['Статус РСЗ', 'Тип', 'Инспектор, сменивший статус']].last()
    new_df = grouped_df[grouped_df['Статус РСЗ'] == 'Новое']

    distribution_new_tasks = {'RISK_SHORT': 1, 'SCHEMA': 1, 'RISK_LONG': 1, 'TASK': 1}
    distribution_new_tasks.update((k, new_df['Тип'].value_counts().to_dict()[k])
                                  for k in set(distribution_new_tasks) & set(new_df['Тип'].value_counts()))
    print("Distribution of new tasks:", distribution_new_tasks)
    return inspectors_df, new_df, distribution_new_tasks

# ---------------------------------------------------------------------------
#  Helper functions from notebook (unchanged)
# ---------------------------------------------------------------------------

def _filter_by_no(no_code: int, inspectors_df: pd.DataFrame, dataframe: pd.DataFrame):
    no_inspectors_df = inspectors_df[inspectors_df['Код НО инспектора, сменившего стат'] == no_code].copy()
    no_inspectors_df["Inspector index"] = np.arange(len(no_inspectors_df))

    # Привязываем задачи к НО
    df2 = dataframe.merge(inspectors_df[['Код НО инспектора, сменившего стат']],
                          how='left',
                          left_on='Инспектор, сменивший статус',
                          right_index=True)
    df2['Код НО инспектора, сменившего стат'] = df2['Код НО инспектора, сменившего стат'].fillna(0).astype(int)
    no_df = df2[df2['Код НО инспектора, сменившего стат'] == no_code].copy()
    return no_inspectors_df, no_df


def _start_data_prep(inspectors_df: pd.DataFrame, df: pd.DataFrame):
    # Расчёт вероятности передачи определённого типа задачи сотруднику --> np.array(сотрудники, типы задач)
    task_prob = [inspectors_df['p_SCHEMA'].to_list(), inspectors_df['p_RISK_LONG'].to_list(), inspectors_df['p_RISK_SHORT'].to_list(), inspectors_df['p_TASK'].to_list(),]
    task_prob = np.array([*zip(*task_prob)])
    
    # Определения числа распределяемых задач и числа сотрудников --> int, int
    N, M = df.__len__(), inspectors_df.__len__()
    
    # Создание массива хранящего в себе тип задачи по порядку --> np.array(число распределяемых задач)
    task_cat = df['Тип'].to_numpy()
    
    # Создание словаря хранящего количество распределяемых задач данного типа --> dict()
    den_TNO = {'RISK_SHORT': 1, 'SCHEMA': 1, 'RISK_LONG': 1, 'TASK': 1}
    den_TNO.update((k, df['Тип'].value_counts().to_dict()[k]) for k in set(den_TNO) & set(df['Тип'].value_counts().to_dict()))
    
    return task_prob, N, M, task_cat, den_TNO


def _evaluation(individual: list, den_TNO: dict, task_cat: np.array, task_prob: np.array, M: int, results: bool = False):
    K1, K2, K3, K4 = 100, 80, 46, 3
    K_SUM = K1 + K2 + K3 + K4

    cnt = {cat: np.zeros(M, dtype=int) for cat in den_TNO}
    for j, emp_idx in enumerate(individual):
        cnt[task_cat[j]][emp_idx] += 1

    nu = (
        cnt["SCHEMA"]     / den_TNO["SCHEMA"]     * K1 +
        cnt["RISK_LONG"]  / den_TNO["RISK_LONG"]  * K2 +
        cnt["RISK_SHORT"] / den_TNO["RISK_SHORT"] * K3 +
        cnt["TASK"]       / den_TNO["TASK"]       * K4
    ) / K_SUM * 100

    efficiency = (
        cnt["SCHEMA"]     * task_prob[:, 0] / den_TNO["SCHEMA"] +
        cnt["RISK_LONG"]  * task_prob[:, 1] / den_TNO["RISK_LONG"] +
        cnt["RISK_SHORT"] * task_prob[:, 2] / den_TNO["RISK_SHORT"] +
        cnt["TASK"]       * task_prob[:, 3] / den_TNO["TASK"]
    ).sum() * 100

    load = nu.max()
    return (load, efficiency) if not results else (nu, efficiency)


def _multi_optimization(task_prob, N, M, task_cat, den_TNO,
                        population_size=100, epoches=50, p_crossing=0.5, p_mutation=0.2):
    random.seed(42)
    np.random.seed(42)

    func = partial(_evaluation, den_TNO=den_TNO, task_cat=task_cat, task_prob=task_prob, M=M)

    creator.create('FintesMulti', base.Fitness, weights=(-1, 1)) # load ↓, efficiency ↑
    creator.create('Individual', list, fitness=creator.FintesMulti)
    
    toolbox = base.Toolbox()
    fit1_stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    fit2_stats = tools.Statistics(key=lambda ind: ind.fitness.values[1])

    mstats = tools.MultiStatistics(fit1=fit1_stats, fit2=fit2_stats)
    mstats.register("min", np.min)
    mstats.register("avg", np.mean)
    mstats.register("max", np.max)

    toolbox.register('attr_int', np.random.randint, 0, M)
    toolbox.register('individual', tools.initRepeat, creator.Individual, 
                    toolbox.attr_int, N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutUniformInt,  low=0, up=M-1, indpb=0.05)
    toolbox.register('select', tools.selNSGA2)
    
    hall_of_fame = tools.HallOfFame(np.inf)
    population = toolbox.population(population_size)
    pop, logbook = algorithms.eaMuPlusLambda(population, 
                                    toolbox, 
                                    mu=population_size,
                                    lambda_=population_size,
                                    cxpb=p_crossing, mutpb=p_mutation, ngen=epoches, 
                                    halloffame=hall_of_fame,
                                    stats=mstats, verbose=True)
    
    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    uniq_pareto = []
    for ind in pareto:
        if ind not in uniq_pareto:
            uniq_pareto.append(ind)
            
    print("Len of Pareto set:", len(uniq_pareto))
    return hall_of_fame, uniq_pareto

# ---------------------------------------------------------------------------
#  Main worker: run evolution & return artefacts -----------------------------
# ---------------------------------------------------------------------------

def _run_evolution(csv_path: Path) -> Tuple[Tuple, Tuple]:
    inspectors_df, new_df, _ = _prepare_dataframe(csv_path)
    print("Inspectors DataFrame:", inspectors_df.head())
    print("New Tasks DataFrame:", new_df.head())
    no_inspectors_df, no_df = _filter_by_no(3700, inspectors_df, new_df)
    task_prob, N, M, task_cat, den_TNO = _start_data_prep(no_inspectors_df, no_df)

    hof, uniq_pareto = _multi_optimization(task_prob, N, M, task_cat, den_TNO)

    all_generations = hof.items
    populations_xy = list(zip(*[ind.fitness.values for ind in all_generations]))
    populations_xy.append(tuple(all_generations))

    pareto_xy = list(zip(*[ind.fitness.values for ind in uniq_pareto]))
    pareto_xy.append(tuple(uniq_pareto))

    return tuple(populations_xy), tuple(pareto_xy)

# ---------------------------------------------------------------------------
#  Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_results(*, data_path: str | Path | None = None, recompute: bool = False):
    """Return *(populations, pareto_front)*.

    Parameters
    ----------
    data_path : str | Path | None
        CSV path.  If None → берём файл рядом с модулем.
    recompute : bool, default False
        Ignore cached pickle and перезапустить GA.
    """
    csv_path = "data/Automated_RSZ_distribution_enc.csv"#Path(data_path) if data_path is not None else _DEFAULT_DATA

    if not recompute and _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open('rb') as fh:
                return pickle.load(fh)
        except Exception:
            pass  # повреждённый кеш → пересчитаем

    populations, pareto_front = _run_evolution(csv_path)

    # save cache (ignore errors silently)
    try:
        with _CACHE_FILE.open('wb') as fh:
            pickle.dump((populations, pareto_front), fh)
    except OSError:
        pass

    return populations, pareto_front

# ---------------------------------------------------------------------------
#  Quick manual test ---------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pop, pf = get_results(recompute=True)
    print(f"Populations: {len(pop[0])} individuals → first 5: {list(zip(pop[0], pop[1]))[:5]}")
    print(f"ParetoFront: {len(pf[0])} individuals → {list(zip(pf[0], pf[1]))}")
