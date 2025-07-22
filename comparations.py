"""Compare Pareto fronts from DEAP and ACO optimizers."""

from __future__ import annotations

import pandas as pd

import deap_optim
import instects


def main() -> None:
    # Получаем результаты оптимизации из обоих модулей
    _, deap_pf = deap_optim.get_results()
    _, aco_pf = instects.get_results()

    deap_loads, deap_effs, _ = deap_pf
    aco_loads, aco_effs, _ = aco_pf

    print("DEAP Pareto efficiency values:")
    print(list(deap_effs))
    print("ACO Pareto efficiency values:")
    print(list(aco_effs))

    df_deap = pd.DataFrame({
        "Algorithm": "DEAP",
        "Load": deap_loads,
        "Efficiency": deap_effs,
    })
    df_aco = pd.DataFrame({
        "Algorithm": "ACO",
        "Load": aco_loads,
        "Efficiency": aco_effs,
    })

    table = pd.concat([df_deap, df_aco], ignore_index=True)
    print("Pareto comparison table:")
    print(table)


if __name__ == "__main__":
    main()

