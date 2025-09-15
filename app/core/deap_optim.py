from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple, Sequence
from functools import partial

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random


_CACHE_FILE = Path(__file__).with_suffix(".pkl")
# _DEFAULT_DATA = Path(__file__+"data/").with_name("Automated_RSZ_distribution_enc.csv")
# get_results И get_assignment_table используют статичный путь к CSV

def _filter_by_no(no_code: int, inspectors_df: pd.DataFrame, dataframe: pd.DataFrame):
    no_inspectors_df = inspectors_df[inspectors_df['Код НО инспектора, сменившего стат'] == no_code]
    no_inspectors_df["Inspector index"] = [i for i in range(len(no_inspectors_df))]

    mask = no_inspectors_df['Qualification'] >= 0.71 * no_inspectors_df['Qualification'].max()

    # Используем np.where для каждого столбца
    no_inspectors_df['p_SCHEMA'] = np.where(mask, 0.5, 0)
    no_inspectors_df['p_RISK_LONG'] = np.where(mask, 0.5, 0)
    no_inspectors_df['p_RISK_SHORT'] = np.where(mask, 0, 0.5)
    no_inspectors_df['p_TASK'] = np.where(mask, 0, 0.5)
    
    # Заспределяем новые задачи по НО    
    # dataframe = dataframe.merge(inspectors_df[['Код НО инспектора, сменившего стат']], how='left', left_on='Инспектор, сменивший статус', right_index=True)
    dataframe['Код НО инспектора, сменившего стат'] = dataframe['Код НО инспектора, сменившего стат'].fillna(0).astype(int)
    no_df = dataframe[dataframe['Код НО инспектора, сменившего стат'] == no_code]

    return no_inspectors_df, no_df


def _start_data_prep(inspectors_df: pd.DataFrame, df: pd.DataFrame, in_work: pd.DataFrame = None):
    # Расчёт вероятности передачи определённого типа задачи сотруднику --> np.array(сотрудники, типы задач)
    task_prob = [inspectors_df['p_SCHEMA'].to_list(), inspectors_df['p_RISK_LONG'].to_list(), inspectors_df['p_RISK_SHORT'].to_list(), inspectors_df['p_TASK'].to_list(),]
    task_prob = np.array([*zip(*task_prob)])
    
    # Определения числа распределяемых задач и числа сотрудников --> int, int
    N, M = df.__len__(), inspectors_df.__len__()
    
    # Создание массива хранящего в себе тип задачи по порядку --> np.array(число распределяемых задач)
    if in_work is not None:
        concate_df = pd.concat([df, in_work])
    else:
        concate_df = df
    task_cat = concate_df['Тип'].to_numpy()
    
    # Создание словаря хранящего количество распределяемых задач данного типа --> dict()
    den_TNO = {'RISK_SHORT': 1, 'SCHEMA': 1, 'RISK_LONG': 1, 'TASK': 1}
    den_TNO.update((k, concate_df['Тип'].value_counts().to_dict()[k]) for k in set(den_TNO) & set(concate_df['Тип'].value_counts().to_dict()))
    
    return task_prob, N, M, task_cat, den_TNO


def _evaluation(individual: list, den_TNO: dict, task_cat: np.array, task_prob: np.array, M: int, results: bool = False, current_individ: list = []):
    full_ind = individual + current_individ
    K1, K2, K3, K4 = 100, 80, 46, 3
    K_SUM = K1 + K2 + K3 + K4
    cnt = {cat: np.zeros(M, dtype=int) for cat in den_TNO} 
    for j, emp_idx in enumerate(full_ind):
        cnt[task_cat[j]][emp_idx] += 1
        
    nu = (
        cnt["SCHEMA"]     / den_TNO["SCHEMA"]     * K1 +
        cnt["RISK_LONG"]  / den_TNO["RISK_LONG"]  * K2 +
        cnt["RISK_SHORT"] / den_TNO["RISK_SHORT"] * K3 +
        cnt["TASK"]       / den_TNO["TASK"]       * K4
    ) / K_SUM * 100
    
    
    efficiency = (
        cnt["SCHEMA"]     * task_prob[:, 0]   / den_TNO["SCHEMA"]     +
        cnt["RISK_LONG"]  * task_prob[:, 1]   / den_TNO["RISK_LONG"]  +
        cnt["RISK_SHORT"] * task_prob[:, 2]   / den_TNO["RISK_SHORT"] +
        cnt["TASK"]       * task_prob[:, 3]   / den_TNO["TASK"]       
    ).sum() *100

    load = nu.std()  # равномерность загрузки инспекторов (чем меньше, тем лучше)
    return (load, efficiency) if not results else (nu, efficiency)


def _multi_optimization(task_prob, N, M, task_cat, den_TNO, in_work_indiv: list = [],
                        population_size=100, epoches=50, p_crossing=0.5, p_mutation=0.4):
    random.seed(42)
    np.random.seed(42)

    func = partial(_evaluation, den_TNO=den_TNO, task_cat=task_cat, task_prob=task_prob, M=M, current_individ=in_work_indiv)

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

def _get_dataframe(data_dir: str | Path = ""):
    """Load required CSV files from ``data_dir``.

    Parameters
    ----------
    data_dir : str or Path, optional
        Path to directory containing ``inspectors_df.csv``, ``new_tasks_df.csv``
        and ``inwork_tasks_df.csv``. Defaults to ``"data"``.
    """

    base = Path(data_dir)
    inspectors_df = pd.read_csv(base / "inspectors_df.csv")
    new_df = pd.read_csv(base / "new_tasks_df.csv")
    inwork_df = pd.read_csv(base / "inwork_tasks_df.csv")
    return inspectors_df, new_df, inwork_df

def _run_evolution(data_dir: str | Path) -> Tuple[Tuple, Tuple]:
    """Run optimisation pipeline using CSVs from ``data_dir``."""

    inspectors_df, new_df, inwork_df = _get_dataframe(data_dir)
    print("Inspectors DataFrame:", inspectors_df.head())
    print("New Tasks DataFrame:", new_df.head())
    no_inspectors_df, no_df = _filter_by_no(3700, inspectors_df, new_df)
    task_prob, N, M, task_cat, den_TNO = _start_data_prep(no_inspectors_df, no_df)

    if N == 0 or M == 0:
        # нет данных для оптимизации
        empty = ([], [], tuple())
        return empty, empty

    in_work_indiv = in_work_no_df.merge(no_inspectors_df[['Inspector index', 'Инспектор, сменивший статус']], left_on='Инспектор, сменивший статус', right_on='Инспектор, сменивший статус', how='left')
    in_work_indiv = in_work_indiv['Inspector index'].to_list()
        
    hof, uniq_pareto = _multi_optimization(task_prob, N, M, task_cat, den_TNO, in_work_indiv)

    all_generations = hof.items
    populations_xy = list(zip(*[ind.fitness.values for ind in all_generations]))
    populations_xy.append(tuple(all_generations))

    pareto_xy = list(zip(*[ind.fitness.values for ind in uniq_pareto]))
    pareto_xy.append(tuple(uniq_pareto))

    return tuple(populations_xy), tuple(pareto_xy)


def get_results(*, data_dir: str | Path = "", recompute: bool = False):

    return _run_evolution(data_dir)

# получаем таблицу назначений задач инспекторам
# (используется в GUI)
def get_assignment_table(*,
                         pareto_index: int = 0,
                         no_code: int = 3700,
                         data_dir: str | Path = "",
                         recompute: bool = False) -> pd.DataFrame:

    ppopulations, pareto_front = get_results(data_dir=data_dir)

    inspectors_df, new_df, inwork_df = _get_dataframe(data_dir)
    no_inspectors_df, no_df = _filter_by_no(no_code, inspectors_df, new_df)

    uniq_pareto = pareto_front[2]
    if len(uniq_pareto) == 0:
        return pd.DataFrame()

    result_df = no_df[
        ['Статус РСЗ', 'Тип', 'Потенциальный ущерб, руб', 'ИНН НП',
         'Инспектор, сменивший статус']
    ].copy()
    result_df['New Inspector index'] = uniq_pareto[pareto_index]
    result_df.reset_index(inplace=True)
    result_df['Статус РСЗ'] = result_df['Статус РСЗ'].replace('Новое', 'В работе')
    result_df = result_df.merge(
        no_inspectors_df.reset_index()[
            ['Инспектор, сменивший статус', 'Inspector index']
        ],
        how='left',
        left_on='New Inspector index',
        right_on='Inspector index'
    )
    return result_df
# Тесты, чтоб проверить отдельные функции без GUI
if __name__ == "__main__":
    pop, pf = get_results(recompute=True)
    print(
        f"Populations: {len(pop[0])} individuals → first 5:"
        f" {list(zip(pop[0], pop[1]))[:5]}"
    )
    print(
        f"ParetoFront: {len(pf[0])} individuals →"
        f" {list(zip(pf[0], pf[1]))}"
    )
    print("Sample assignment table:")
    print(get_assignment_table())
