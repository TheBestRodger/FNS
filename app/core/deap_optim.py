from __future__ import annotations

from pathlib import Path
from typing import Tuple, Sequence
from functools import partial
from threading import RLock

import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from datetime import datetime, timedelta
import random
import time

# _DEFAULT_DATA = Path(__file__+"data/").with_name("Automated_RSZ_distribution_enc.csv")
# get_results И get_assignment_table используют статичный путь к CSV


_DATAFRAME_CACHE: dict[Path, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
_RESULTS_CACHE: dict[tuple[Path, int], tuple[Tuple, Tuple]] = {}
_CACHE_LOCK = RLock()

def _get_new_type_data(df: pd.DataFrame, Auto_Stats: pd.DataFrame, day: str = '2024-06-27', history_new: bool = True):
    df = df.sort_values(['№ схемы/риска', 'Дата изменения статуса РСЗ'])
    day += ' 00:00:00'
    date_format = '%Y-%m-%d %H:%M:%S'
    day_str = datetime.strptime(day, date_format)
    day_before_str = day_str - timedelta(days=1)
    quartal_day_str = day_str - timedelta(days=90)

    day_df = df[df['Дата изменения статуса РСЗ'] <= day_str]
    columns = ['Статус РСЗ', 'Тип', 'Дата изменения статуса РСЗ', 'Потенциальный ущерб, руб', 'ИНН НП', 'Вид налога', 'Тип декларации', 'Рег. номер декларации', 'КПП', 'Инспектор, сменивший статус', 'Код НО инспектора, сменившего стат']

    cutten_day_df = day_df[day_df['Дата изменения статуса РСЗ'] > day_before_str]
    new_tasks_df = cutten_day_df[cutten_day_df['Статус РСЗ'] == 'Новое'].groupby('№ схемы/риска')[columns].last()

    inwork_tasks_df = df[df['Дата изменения статуса РСЗ'] <= day_before_str].groupby('№ схемы/риска')[columns].last()
    # Одна из настроек позволяющая подтянуть новые задачи со старых дат
    new_tasks_df = pd.concat([new_tasks_df, inwork_tasks_df[inwork_tasks_df['Статус РСЗ'] == 'Новое']]) if history_new else new_tasks_df
    new_tasks_df.drop(columns=['Инспектор, сменивший статус', 'Код НО инспектора, сменившего стат'], inplace=True)
    inwork_tasks_df = inwork_tasks_df[inwork_tasks_df['Статус РСЗ'] != 'Новое']
    finish_tasks_df = inwork_tasks_df[inwork_tasks_df['Статус РСЗ'] != 'В работе']
    inwork_tasks_df = inwork_tasks_df[inwork_tasks_df['Статус РСЗ'] == 'В работе']

    insp_df = df.groupby('№ схемы/риска')[['Инспектор, сменивший статус', 'Код НО инспектора, сменившего стат']].last()
    new_tasks_df = new_tasks_df.merge(insp_df, how='left', left_index=True, right_index=True)

    all_scores = [Auto_Stats.loc[stat, df.iloc[i, 1]] for i, stat in enumerate(finish_tasks_df['Статус РСЗ'])]
    finish_tasks_df['Score'] = all_scores
    
    # Динамическая обрезка по результативности
    finish_tasks_df = finish_tasks_df[finish_tasks_df['Дата изменения статуса РСЗ'] >= quartal_day_str]

    return new_tasks_df, inwork_tasks_df, finish_tasks_df

# Функция для разбиванию инспекторов и РСЗ по НО
def _filter_by_no(no_code: int, df: pd.DataFrame, inwork_tasks_df, new_tasks_df):

    no_df = df[df['Код НО инспектора, сменившего стат'] == no_code]

    # Заспределяем новые или текущие задачи задачи по НО
    inwork_tasks_df['Код НО инспектора, сменившего стат'] = inwork_tasks_df['Код НО инспектора, сменившего стат'].fillna(0).astype(int)
    no_inwork_tasks_df = inwork_tasks_df[inwork_tasks_df['Код НО инспектора, сменившего стат'] == no_code]

    new_tasks_df['Код НО инспектора, сменившего стат'] = new_tasks_df['Код НО инспектора, сменившего стат'].fillna(0).astype(int)
    no_new_tasks_df = new_tasks_df[new_tasks_df['Код НО инспектора, сменившего стат'] == no_code]

    return no_df, no_inwork_tasks_df, no_new_tasks_df

# Создание датасета об инспекторах
def _create_inspetors_df(
    df: pd.DataFrame,
    exp_inspectors_df: pd.DataFrame,
    finish_tasks_df: pd.DataFrame | None = None,
    *,
    insp_col: str = "Инспектор, сменивший статус",
    dir_col: str = "Вид налога",                 # колонка-направление в df/finish_tasks_df
    score_col: str = "Score",                    # баллы из матрицы начисления
    code_no_col: str = "Код НО инспектора, сменившего стат",
    experience_col: str = "Стаж",                # в exp_inspectors_df
    course_prefix: str = "Курс_",                # префикс для курсов в exp_inspectors_df
    time_col: str = "T_days",                    # для формулы v1 (суммарные дни по задачам)
    c_mode: str = "pow",                         # "pow": 1.125**N  |  "log": 1 + log(1+N)
    strong_threshold: float = 0.71,              # порог сильной компетенции (0.71*K_max(dir))
    alpha: float = 0.5,                          # коэфициенты соотношения опыта и стажа
    beta: float = 0.5,
    reduction: float = 0.0
):
    assert insp_col in df.columns, f"В df нет столбца '{insp_col}'"
    assert dir_col in df.columns,  f"В df нет столбца '{dir_col}'"
    assert code_no_col in df.columns, f"В df нет столбца '{code_no_col}'"
    assert experience_col in exp_inspectors_df.columns, f"В exp_inspectors_df нет '{experience_col}'"


    # Master по инспекторам: Code_NO, Стаж
    main_inspectors = (
        df.sort_values([insp_col, "Дата изменения статуса РСЗ"])
          .groupby(insp_col)[code_no_col].last()
          .rename("Code_NO").to_frame()
    )
    if exp_inspectors_df.index.name != insp_col:
        exp_inspectors_df = exp_inspectors_df.set_index(insp_col)
    main_inspectors[experience_col] = exp_inspectors_df[experience_col].reindex(main_inspectors.index).fillna(0)

    main_inspectors["Inspector index"] = (
    main_inspectors.groupby(insp_col).ngroup())

    #Полное множество (инспектор × направление)

    all_inspectors = main_inspectors.index.unique()
    all_dirs = pd.Index(sorted(df[dir_col].dropna().unique()))

    base = (
        pd.MultiIndex.from_product([all_inspectors, all_dirs], names=[insp_col, dir_col])
          .to_frame(index=False)
    )

    # Добавление курсов
    courses_long = (
            exp_inspectors_df.reset_index()
                            .melt(id_vars=[insp_col],
                                value_vars=all_dirs,
                                var_name="course_col",
                                value_name="N_courses")
        )
    
    courses_long = courses_long.rename(columns={"course_col": dir_col})
    
    inspectors_direction_df = base.merge(courses_long, how="left", on=[insp_col, dir_col])
    inspectors_direction_df["N_courses"] = inspectors_direction_df["N_courses"].fillna(0).astype(int)

    # Добавление Code NO и Стажа
    inspectors_direction_df = inspectors_direction_df.merge(
        main_inspectors[["Code_NO", "Inspector index", experience_col]],
        how="left", left_on=insp_col, right_index=True
    )
    inspectors_direction_df[experience_col] = inspectors_direction_df[experience_col].fillna(0)



    # Статистика по всем задачам
    all_stats = df.groupby([insp_col, dir_col]).size().rename("N_all_tasks").reset_index()
    inspectors_direction_df = inspectors_direction_df.merge(all_stats, how="left", on=[insp_col, dir_col])
    inspectors_direction_df["N_all_tasks"] = inspectors_direction_df["N_all_tasks"].fillna(0).astype(int)

    for col in (insp_col, dir_col, score_col):
        assert col in finish_tasks_df.columns, f"В finish_tasks_df нет столбца '{col}'"

    fin_g = finish_tasks_df.groupby([insp_col, dir_col])
    n_done = fin_g.size().rename("N_done").reset_index()
    sum_score = fin_g[score_col].sum().rename("sum_score").reset_index()

    inspectors_direction_df = (
        inspectors_direction_df
        .merge(n_done,    how="left", on=[insp_col, dir_col])
        .merge(sum_score, how="left", on=[insp_col, dir_col])
    )

    inspectors_direction_df[["N_done", "sum_score"]] = inspectors_direction_df[["N_done", "sum_score"]].fillna(0)
    inspectors_direction_df["N_done"] = inspectors_direction_df["N_done"].astype(int)


    inspectors_direction_df["P"] = np.where(
    inspectors_direction_df["N_done"] > 0,
    (inspectors_direction_df["sum_score"] / 100.0) / inspectors_direction_df["N_done"] * 100, 0.0)


    if c_mode == "pow":
        cN = np.power(1.125, inspectors_direction_df["N_courses"])
    elif c_mode == "log":
        cN = 1.0 + np.log1p(inspectors_direction_df["N_courses"])
    else:
        raise ValueError("c_mode должен быть 'pow' или 'log'")


    inspectors_direction_df["cN"] = cN
    inspectors_direction_df["Omega"] = np.log1p(inspectors_direction_df[experience_col]) * inspectors_direction_df["cN"]

    inspectors_direction_df["P_norm"] = (inspectors_direction_df["P"] / inspectors_direction_df["P"].max()).clip(0.0, 1.0)
    inspectors_direction_df["Omega_norm"] = (inspectors_direction_df["Omega"] / inspectors_direction_df["Omega"].max()).fillna(0.0).clip(0.0, 1.0)
    ALPHA, BETA = alpha, beta   # можно параметризовать
    inspectors_direction_df["K"] = (ALPHA * inspectors_direction_df["Omega_norm"] + BETA * inspectors_direction_df["P_norm"])


    K_max_dir = inspectors_direction_df.groupby(dir_col)["K"].transform("max")
    strong = inspectors_direction_df["K"] >= (strong_threshold * K_max_dir.fillna(0))

    inspectors_direction_df["p_SCHEMA"]     = np.where(strong, 1.0, 0.0)
    inspectors_direction_df["p_RISK_LONG"]  = np.where(strong, 1.0, 0.0)
    inspectors_direction_df["p_RISK_SHORT"] = np.where(strong, 0.0, 1.0)
    inspectors_direction_df["p_TASK"]       = np.where(strong, 0.0, 1.0)


    out_cols = [
        insp_col, dir_col, "Code_NO", "Inspector index", experience_col, "N_courses",
        "N_all_tasks", "N_done", "sum_score", "P", "P_norm", "Omega", "Omega_norm", "K",
        "p_SCHEMA", "p_RISK_LONG", "p_RISK_SHORT", "p_TASK"
    ]

    inspectors_direction_df = inspectors_direction_df[out_cols].sort_values([insp_col, dir_col]).reset_index(drop=True)
    red = inspectors_direction_df.groupby("Инспектор, сменивший статус")["P"].max().sort_values(ascending=False).index[:int(len(all_inspectors) * (1 - reduction))]
    inspectors_direction_df[inspectors_direction_df["Инспектор, сменивший статус"].isin(red)]
    return main_inspectors, inspectors_direction_df


# Подсчёт статистик
def _start_data_prep(new_tasks_df: pd.DataFrame, no_inspectors_df: pd.DataFrame, in_work_df: pd.DataFrame = None,
                        *,
                        insp_col: str = "Инспектор, сменивший статус",
                        dir_col: str = "Вид налога",
                        type_col: str = "Тип"):
    """
    Готовит данные для ГА с учётом направлений:
      - p: тензор вероятностей формы (M, 4, m). М - колво инспекторов, m - колво направлений
      - N, M: число новых задач и число уникальных инспекторов
      - task_type: массив типов задач (код 0..3) длиной N+K
      - task_dir: массив направлений (код 0..m-1) длиной N+K
      - den_TNO: словарь общих объёмов по типам (в целом по ТНО)
      - maps: словари соответствий индексов инспекторов и направлений

    Ожидается, что no_inspectors_df содержит строки по (инспектор × направление) и колонки p_SCHEMA, p_RISK_LONG, p_RISK_SHORT, p_TASK.
    """


    # Создание массива хранящего в себе тип задачи по порядку --> np.array(число распределяемых задач)
    if in_work_df is not None:
        concat_df = pd.concat([new_tasks_df, in_work_df], ignore_index=True)
    else:
        concat_df = new_tasks_df

    # Уникальные инспекторы из no_inspectors_df (именно по нему строим M)
    inspectors = pd.Index(no_inspectors_df[insp_col].dropna().unique())
    M = len(inspectors)
    N = len(new_tasks_df)
    insp_to_idx = {ins: i for i, ins in enumerate(inspectors)}

    # Уникальные направления — из задач (чтобы точно покрыть все используемые в оценке)
    directions = pd.Index(concat_df[dir_col].dropna().unique())
    m = len(directions)
    dir_to_idx = {d: j for j, d in enumerate(directions)}

    print(f"Number of directions for new and in work tasks: {m}")
    print(f"Number of inspectors: {M}")
    print(f"Number of new tasks: {N}")


     # ---------------- коды типов задач (фиксируем порядок) ----------------
    type_order = ["SCHEMA", "RISK_LONG", "RISK_SHORT", "TASK"]
    p_type_order = ["p_SCHEMA", "p_RISK_LONG", "p_RISK_SHORT", "p_TASK"]
    type_to_code = {t: k for k, t in enumerate(type_order)}

    # Массивы типов и направлений для всех задач (новые + текущие) если в данных типы/направления — уже строчки, мапим в int-коды:
    task_type = concat_df[type_col].map(type_to_code).to_numpy()
    task_dir = concat_df[dir_col].map(dir_to_idx).to_numpy()



    # ---------------- тензор вероятностей p: (M, 4, m) ----------------
    # Инициализация нулями: если где-то нет пары (инспектор, направление), вероятности = 0
    p = np.zeros((M, 4, m), dtype=float)

    #  Приведём таблицу вероятностей к нужным индексам
    # Оставляем только строки по направлениям, которые реально встречаются в задачах
    tmp = no_inspectors_df[[insp_col, dir_col, *p_type_order]].copy()
    tmp = tmp[tmp[dir_col].isin(directions)]


    # Мапим индексы
    tmp["i"] = tmp[insp_col].map(insp_to_idx)
    tmp["j"] = tmp[dir_col].map(dir_to_idx)

    # Защита от пропусков (инспектора из другого НО и т.п.)
    tmp = tmp.dropna(subset=["i", "j"])
    tmp["i"] = tmp["i"].astype(int)
    tmp["j"] = tmp["j"].astype(int)

    # Заполняем тензор p
    # p_SCHEMA->0, p_RISK_LONG->1, p_RISK_SHORT->2, p_TASK->3
    for row in tmp.itertuples(index=False):
        i, j = row.i, row.j
        p[i, 0, j] = getattr(row, p_type_order[0])  # p_SCHEMA
        p[i, 1, j] = getattr(row, p_type_order[1])  # p_RISK_LONG
        p[i, 2, j] = getattr(row, p_type_order[2])  # p_RISK_SHORT
        p[i, 3, j] = getattr(row, p_type_order[3])  # p_TASK

    # Элемент p[i, t, j] — «вероятность/вес» того, что инспектор i подходит для типа t по направлению j.

    # ---------------- N и словарь объёмов по типам (den_TNO) ----------------

    den_TNO = {'RISK_SHORT': 1, 'SCHEMA': 1, 'RISK_LONG': 1, 'TASK': 1}
    counts = concat_df[type_col].value_counts().to_dict()
    for k in den_TNO.keys() & counts.keys():
        den_TNO[k] = counts[k]

    # ---------------- возврат ----------------
    maps = {
        "inspector_index_to_id": inspectors.to_list(),
        "inspector_id_to_index": insp_to_idx,
        "dir_index_to_name": directions.to_list(),
        "dir_name_to_index": dir_to_idx,
        "type_order": type_order,
        "type_to_code": type_to_code,
    }

    return p, N, M, task_type, task_dir, den_TNO, maps


def _evaluation(individual: list,
            den_TNO: dict,
            task_cat: np.ndarray,      # КОДЫ типов задач (0:SCHEMA,1:RISK_LONG,2:RISK_SHORT,3:TASK) для всех задач (N+K)
            task_dir: np.ndarray,      # КОДЫ направлений (0..m-1) для всех задач (N+K)
            task_prob: np.ndarray,     # ТЕНЗОР p формы (M, 4, m) — вероятности для (инспектор, тип, направление)
            M: int,
            results: bool = False,
            current_individ: list | None = None,
            load_metric: str = "gini"   # <-- новый параметр

        ):
    """
    Возвращает:
      - если results=False:  (load, efficiency)
      - если results=True:   (nu, efficiency), где nu — профиль нагрузок по инспекторам (%)

    Примечания по входам:
      * individual        — назначение N НОВЫХ задач (индексы инспекторов 0..M-1)
      * current_individ   — уже назначенные K задач (если есть)
      * task_cat          — коды типов для всех задач (по порядку: новые + текущие)
      * task_dir          — коды направлений для всех задач (тот же порядок)
      * task_prob (p)     — тензор p[i, t, j], где i — инспектор, t — тип, j — направление
      * den_TNO           — объёмы задач по типам в целом по ТНО (для формулы нагрузки)
    """

    if current_individ is None:
        current_individ = []
    full_ind = individual + current_individ  # L = N + K

    L = len(full_ind) # всего задач
    m = task_prob.shape[2]  # число направлений

    assert L == len(task_cat) == len(task_dir), "Длины full_ind, task_cat (типы) и task_dir (направления) должны совпадать"
    assert task_prob.ndim == 3 and task_prob.shape[0] == M and task_prob.shape[1] == 4, \
        "task_prob (тензор p) должен иметь форму (M, 4, m)"


    # == нагрузка ==

    #  3D-счётчики назначений: cnt[type, dir, emp]
    cnt = np.zeros((4, m, M), dtype=int)
    np.add.at(cnt, (task_cat, task_dir, np.asarray(full_ind, dtype=int)), 1)

    # свернуть по направлениям -> (4, M)
    cnt_type_emp = cnt.sum(axis=1)    

    K1, K2, K3, K4 = 100, 80, 46, 3
    # K_SUM = K1 + K2 + K3 + K4

    # делим на общие объёмы по типам (защита от нулей)
    # nu = (
    #     cnt_type_emp[0] / max(den_TNO["SCHEMA"],     1) * K1 +
    #     cnt_type_emp[1] / max(den_TNO["RISK_LONG"],  1) * K2 +
    #     cnt_type_emp[2] / max(den_TNO["RISK_SHORT"], 1) * K3 +
    #     cnt_type_emp[3] / max(den_TNO["TASK"],       1) * K4
    # ) / K_SUM * 100.0
    # второй вариант
    nu = (
        cnt_type_emp[0] * K1 +
        cnt_type_emp[1] * K2 +
        cnt_type_emp[2] * K3 +
        cnt_type_emp[3] * K4
    ) / (den_TNO["SCHEMA"] * K1 + den_TNO["RISK_LONG"] * K2 + den_TNO["RISK_SHORT"] * K3 + den_TNO["TASK"] * K4) * 100.0

  # === выбор метрики нагрузки ===
    if load_metric == "std":
        load = float(nu.std())
    elif load_metric == "range":
        load = float(nu.max() - nu.min())
    elif load_metric == "gini":
        sorted_nu = np.sort(nu)
        n = len(sorted_nu)
        cum = np.cumsum(sorted_nu)
        gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0.0
        load = float(gini * 100)  # можно в процентах
    else:
        raise ValueError(f"Неизвестная метрика нагрузки: {load_metric}")




    # == эффективность ==
    # T[dir, emp] — всего задач у инспектора по направлению
    T = cnt.sum(axis=0)  # (m, M)

    # доли типов внутри направления у инспектора; если T=0 -> 0 ТУТ ПРОБЛЕМА
    share = np.divide(
        cnt,
        T[None, :, :],
        out=np.zeros_like(cnt, dtype=float),
        where=T[None, :, :] > 0
    )
    
    

    # согласуем оси p: (M,4,m) -> (4,m,M)
    p_43M = np.transpose(task_prob, (1, 2, 0))
    
    count_dir = np.count_nonzero(T, axis=0)
    
    # print(share)
    # print(count_dir)
    # print(p_43M)
    # print(share * p_43M)    
    # print((share * p_43M).sum())

    efficiency = (share * p_43M).sum(axis=0)
    efficiency = float(np.divide(efficiency,
                                 count_dir[None, :],
                                 out=np.zeros_like(efficiency, dtype=float),
                                 where=count_dir[None, :] > 0
                                 ).sum() / M * 100) # максимизация

    if not results:
        return load, efficiency
    return nu, efficiency


def _multi_optimization(
    in_work_indiv: list,          # K уже назначенных задач (индексы инспекторов 0..M-1)
    task_prob: np.ndarray,                # тензор вероятностей (M, 4, m)
    N: int,                       # число НОВЫХ задач (длина индивида)
    M: int,                       # число инспекторов
    task_type: np.ndarray,        # коды типов задач (0..3) длиной N+K (новые + текущие)
    task_dir:  np.ndarray,        # коды направлений (0..m-1) длиной N+K
    den_TNO: dict,
    population_size: int = 200,
    epoches: int = 100,
    p_crossing: float = 0.5,
    p_mutation: float = 0.25
):
    random.seed(42)
    np.random.seed(42)


    # assert p.ndim == 3 and p.shape[0] == M and p.shape[1] == 4, "p должен иметь форму (M,4,m)"
    assert len(task_type) == len(task_dir) == N + len(in_work_indiv), "task_type/task_dir длиной N+K"
    assert 0 <= p_crossing <= 1 and 0 <= p_mutation <= 1

    for k in ["SCHEMA", "RISK_LONG", "RISK_SHORT", "TASK"]:
        assert k in den_TNO, f"В den_TNO отсутствует ключ {k}"



    t0_total = time.perf_counter()



    func = partial(
        _evaluation,                # новая версия
        den_TNO=den_TNO,
        task_cat=task_type,
        task_dir=task_dir,
        task_prob=task_prob,               # тензор p (M,4,m)
        M=M,
        current_individ=in_work_indiv,
        load_metric = 'std'
    )


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
    toolbox.register('mutate', tools.mutUniformInt,  low=0, up=M-1, indpb=0.1)
    toolbox.register('select', tools.selNSGA2)


    # тайминг: инициализация популяции
    t0_init = time.perf_counter()
    population = toolbox.population(population_size)
    t1_init = time.perf_counter()



    hall_of_fame = tools.HallOfFame(np.inf)

    t0_evo = time.perf_counter()

    pop, logbook = algorithms.eaMuPlusLambda(population,
                                    toolbox,
                                    mu=population_size//2,
                                    lambda_=population_size//2,
                                    cxpb=p_crossing, mutpb=p_mutation, ngen=epoches,
                                    halloffame=hall_of_fame,
                                    stats=mstats, verbose=False) #True)

    t1_evo = time.perf_counter()


    # pareto = tools.sortNondominated(hall_of_fame.items, len(hall_of_fame.items), first_front_only=True)[0]
    pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    uniq_pareto = []
    for ind in pareto:
        if ind not in uniq_pareto:
            uniq_pareto.append(ind)



    t1_total = time.perf_counter()

    # оценим число вызовов evaluation: стартовая популяция + потомки на каждом поколении
    est_evals = population_size + epoches * (population_size // 2)

    # сводка времени
    init_time_s = t1_init - t0_init
    evo_time_s  = t1_evo - t0_evo
    total_time_s = t1_total - t0_total
    per_eval_ms = (evo_time_s / max(est_evals, 1)) * 1e3

    print(f"\n=== Сводка по времени ===")
    print(f"Init population: {init_time_s:,.3f} s")
    print(f"Evolution:       {evo_time_s:,.3f} s over ~{est_evals} evaluations (~{per_eval_ms:,.3f} ms/eval)")
    print(f"Total wall time: {total_time_s:,.3f} s")
    print(f"Len of Pareto set: {len(uniq_pareto)}")
    return hall_of_fame, uniq_pareto

def _resolve_dir(data_dir: str | Path) -> Path:
    """Convert ``data_dir`` to an absolute :class:`Path`."""

    return Path(data_dir).expanduser().resolve()


def _get_dataframe(data_dir: str | Path = ""):
    """Load required CSV files from ``data_dir`` with basic caching."""

    base = _resolve_dir(data_dir)
    with _CACHE_LOCK:
        cached = _DATAFRAME_CACHE.get(base)
    if cached is not None:
        exp_cached, df_cached, stats_cached = cached
        return (
            exp_cached.copy(deep=False),
            df_cached.copy(deep=False),
            stats_cached.copy(deep=False),
        )

    exp_inspectors_df = pd.read_csv(base / "exp_inspectors_df.csv", index_col=0)
    df = pd.read_csv(base / "Automated_RSZ_distribution_enc.csv", sep=';', low_memory=False)
    df = df.sort_values(['№ схемы/риска', 'Дата изменения статуса РСЗ'])

    drop_list = df[(df['Инспектор, сменивший статус'].isna()) & (df['Статус РСЗ'] != 'Новое')]['№ схемы/риска'].unique()
    df = df.drop(index=df[df['№ схемы/риска'].isin(drop_list)].index)
    df.loc[df['Код НО инспектора, сменившего стат']=='000n', 'Код НО инспектора, сменившего стат'] = 0

    df.drop_duplicates(inplace=True)
    df['Дата выявления'] = pd.to_datetime(df['Дата выявления'], format='%Y-%m-%d %H:%M:%S')
    df['Дата изменения статуса РСЗ'] = pd.to_datetime(df['Дата изменения статуса РСЗ'], format='%Y-%m-%d %H:%M:%S')
    df['Код НО инспектора, сменившего стат'] = df['Код НО инспектора, сменившего стат'].astype(int)

    df['Тип'] = df['Тип'].replace(
        to_replace={
            'Риск_долго': 'RISK_LONG',
            'Риск_быстро': 'RISK_SHORT',
            'Схема': 'SCHEMA',
            'Задание': 'Task'
        },
        regex=True)

    Auto_Stats_df = pd.read_csv(base / "Auto_Stats.csv", index_col=0)

    with _CACHE_LOCK:
        _DATAFRAME_CACHE[base] = (exp_inspectors_df, df, Auto_Stats_df)

    return (
        exp_inspectors_df.copy(deep=False),
        df.copy(deep=False),
        Auto_Stats_df.copy(deep=False),
    )

def _run_evolution(data_dir: str | Path, *, no_code: int) -> Tuple[Tuple, Tuple]:
    """Run optimisation pipeline using CSVs from ``data_dir``."""

    exp_inspectors_df, df, Auto_Stats_df = _get_dataframe(data_dir)
    new_tasks_df, inwork_tasks_df, finish_tasks_df = _get_new_type_data(df, Auto_Stats_df, '2024-06-27')
    print("Inspectors DataFrame:", exp_inspectors_df.head())
    print("New Tasks DataFrame:", new_tasks_df.head())
    no_df, no_inwork_tasks_df, no_new_tasks_df = _filter_by_no(no_code, df, inwork_tasks_df, new_tasks_df)
    main_inspectors_df, inspectors_direction_df  = _create_inspetors_df(no_df, exp_inspectors_df, finish_tasks_df)
    in_work_indiv = no_inwork_tasks_df.merge(main_inspectors_df[['Inspector index']], left_on='Инспектор, сменивший статус', right_index=True, how='left')['Inspector index'].to_list()
    new_tasks_current_indiv = no_new_tasks_df.merge(main_inspectors_df[['Inspector index']], left_on='Инспектор, сменивший статус', right_index=True, how='left')['Inspector index'].to_list()

    task_prob, N, M, task_type, task_dir, den_TNO, maps = _start_data_prep(no_new_tasks_df, inspectors_direction_df, no_inwork_tasks_df)
    
    if N == 0 or M == 0:
        # нет данных для оптимизации
        empty = ([], [], tuple())
        return empty, empty

    hof, uniq_pareto = _multi_optimization(
        in_work_indiv=in_work_indiv,
        task_prob=task_prob,
        N=N,
        M=M,
        task_type=task_type,
        task_dir=task_dir,
        den_TNO=den_TNO,
        population_size=200,
        epoches=200,
        p_crossing=0.5,
        p_mutation=0.25
    )

    all_generations = hof.items
    populations_xy = list(zip(*[ind.fitness.values for ind in all_generations]))
    populations_xy.append(tuple(all_generations))

    pareto_xy = list(zip(*[ind.fitness.values for ind in uniq_pareto]))
    pareto_xy.append(tuple(uniq_pareto))

    return tuple(populations_xy), tuple(pareto_xy)


def get_results(*, data_dir: str | Path = "", no_code: int = 3700, recompute: bool = False):

    base = _resolve_dir(data_dir)
    cache_key = (base, no_code)
    if not recompute:
        with _CACHE_LOCK:
            cached = _RESULTS_CACHE.get(cache_key)
        if cached is not None:
            return cached

    result = _run_evolution(base, no_code=no_code)
    with _CACHE_LOCK:
        _RESULTS_CACHE[cache_key] = result
    return result

# получаем таблицу назначений задач инспекторам
# (используется в GUI)
def get_assignment_table(
    *,
    pareto_index: int = 0,
    no_code: int = 3700,
    data_dir: str | Path = "",
    recompute: bool = False,
    pareto_front: Tuple | None = None,
    new_tasks_df: pd.DataFrame | None = None,
    inspectors_direction_df: pd.DataFrame | None = None,
) -> pd.DataFrame:

    base = _resolve_dir(data_dir)
    if pareto_front is None or recompute:
        _, pareto_front = get_results(
            data_dir=base,
            no_code=no_code,
            recompute=recompute,
        )

    uniq_pareto = pareto_front[2] if pareto_front else tuple()
    if len(uniq_pareto) == 0:
        return pd.DataFrame()

    if new_tasks_df is None or inspectors_direction_df is None:
        exp_inspectors_df, df, Auto_Stats_df = _get_dataframe(base)
        new_tasks_df_full, inwork_tasks_df, finish_tasks_df = _get_new_type_data(
            df,
            Auto_Stats_df,
            '2024-06-27',
        )
        no_df, _, default_new_tasks_df = _filter_by_no(
            no_code,
            df,
            inwork_tasks_df,
            new_tasks_df_full,
        )
        _, inspectors_direction_df = _create_inspetors_df(
            no_df,
            exp_inspectors_df,
            finish_tasks_df,
        )
        new_tasks_df = default_new_tasks_df

    result_df = new_tasks_df[
        ['Статус РСЗ', 'Тип', 'Потенциальный ущерб, руб', 'ИНН НП',
         'Инспектор, сменивший статус']
    ].copy()

    assignments = list(uniq_pareto[pareto_index])
    if len(assignments) != len(result_df):
        raise ValueError(
            f"Pareto individual length ({len(assignments)}) does not match number of new tasks ({len(result_df)})."
        )

    result_df['New Inspector index'] = assignments
    result_df.reset_index(inplace=True)
    result_df['Статус РСЗ'] = result_df['Статус РСЗ'].replace('Новое', 'В работе')
    inspector_lookup = inspectors_direction_df.reset_index()[
        ['Инспектор, сменивший статус', 'Inspector index']
    ].drop_duplicates('Inspector index')
    result_df = result_df.merge(
        inspector_lookup,
        how='left',
        left_on='New Inspector index',
        right_on='Inspector index'
    )
    return result_df
# Тесты, чтоб проверить отдельные функции без GUI
if __name__ == "__main__":
    pop, pf = get_results(data_dir="./data/", no_code=3700)
    print(
        f"Populations: {len(pop[0])} individuals -> first 5:"
        f" {list(zip(pop[0], pop[1]))[:5]}"
    )
    print(
        f"ParetoFront: {len(pf[0])} individuals ->"
        f" {list(zip(pf[0], pf[1]))}"
    )
    print("Sample assignment table:")
    print(get_assignment_table(data_dir="./data/", no_code=3700))
