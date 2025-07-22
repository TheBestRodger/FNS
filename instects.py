import random
from typing import Callable, Dict, List, Tuple, Any

import numpy as np

from deap_optim import (
    _prepare_dataframe,
    _filter_by_no,
    _start_data_prep,
    _evaluation,
)


class Solution:
    def __init__(self, params: List[int], scores: Tuple[float, ...]):
        self.params = params
        self.scores = scores

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Solution) and tuple(self.params) == tuple(other.params)

    def __hash__(self) -> int:
        return hash(tuple(self.params))

    def dominates(self, other: "Solution") -> bool:
        better_or_eq = all(a <= b for a, b in zip(self.scores, other.scores))
        better = any(a < b for a, b in zip(self.scores, other.scores))
        return better_or_eq and better


class ParetoArchive:
    def __init__(self) -> None:
        self.archive: set[Solution] = set()

    def update(self, candidates: List[Solution]) -> None:
        for cand in candidates:
            if any(cand.scores == a.scores for a in self.archive):
                continue            # пропускаем точную копию по метрикам
        for cand in candidates:
            to_remove = set()
            dominated = False
            for arch in self.archive:
                if cand.dominates(arch):
                    to_remove.add(arch)
                elif arch.dominates(cand):
                    dominated = True
                    break
            if not dominated:
                self.archive.difference_update(to_remove)
                self.archive.add(cand)

    def get(self) -> List[Solution]:
        return list(self.archive)


class AntColonyOptimizer:
    def __init__(
        self,
        func: Callable[[List[int]], Tuple[float, ...]],
        param_ranges: List[Tuple[int, int]],
        n_ants: int = 30,
        n_iter: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        track: bool = True, 
        verbose: bool = False
    ) -> None:
        self.func = func
        self.dim = len(param_ranges)
        self.ranges = param_ranges
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromones = [np.ones(r[1] - r[0] + 1) for r in param_ranges]
        self.solutions: List[Solution] = []
        self.track   = track      # следить за обучением?
        self.verbose = verbose
        self.history = []         # сюда пишем статистику по поколениям

    def sample_ant(self) -> List[int]:
        params = []
        for d in range(self.dim):
            tau = self.pheromones[d]
            possible_vals = list(range(self.ranges[d][0], self.ranges[d][1] + 1))
            eta = np.ones_like(tau)
            probs = (tau**self.alpha) * (eta**self.beta)
            probs = probs / probs.sum()
            val = random.choices(possible_vals, weights=probs)[0]
            params.append(val)
        return params

    def update_pheromones(self, pareto_solutions: List[Solution]) -> None:
        for d in range(self.dim):
            self.pheromones[d] *= 1 - self.evaporation
        for sol in pareto_solutions:
            for d, val in enumerate(sol.params):
                self.pheromones[d][val - self.ranges[d][0]] += 1

    def run(self) -> Dict[str, Any]:
        archive = ParetoArchive()
        for gen in range(self.n_iter):
            candidates = []
            for _ in range(self.n_ants):
                params = self.sample_ant()
                scores = self.func(params)
                sol = Solution(params, scores)
                candidates.append(sol)
                self.solutions.append(sol)
            archive.update(candidates)
            self.update_pheromones(archive.get())
        
            # ── LOGGING ────────────────────────────────────────────
            if self.track:
                loads, effs = zip(*(s.scores for s in archive.get()))
                self.history.append({
                    "gen": gen,
                    "pareto_size": len(loads),
                    "best_load":  min(loads),
                    "best_eff":   max(effs),        
                    "avg_load":   sum(loads)/len(loads),
                    "avg_eff":    sum(effs)/len(effs),
                })
            if self.verbose and gen % 5 == 0:
                h = self.history[-1]
                print(f"Gen {gen:3d}: Pareto {h['pareto_size']:3d}  "
                      f"best_load={h['best_load']:.3f} best_eff={h['best_eff']:.3f}")
        return {
            "explored": [(sol.params, sol.scores) for sol in self.solutions],
            "pareto_front": [(sol.params, sol.scores) for sol in archive.get()],
            "history": self.history,  # <── новое
        }


def _load_data(no_code: int = 3700):
    inspectors_df, new_df, _ = _prepare_dataframe(
        "data/Automated_RSZ_distribution_enc.csv"
    )
    no_inspectors_df, no_df = _filter_by_no(no_code, inspectors_df, new_df)
    task_prob, N, M, task_cat, den_TNO = _start_data_prep(no_inspectors_df, no_df)
    return task_prob, N, M, task_cat, den_TNO


def main() -> None:
    task_prob, N, M, task_cat, den_TNO = _load_data()
    if N == 0 or M == 0:
        print("No data for optimization")
        return

    def evaluate(params: List[int]) -> Tuple[float, float]:
        load, eff = _evaluation(params, den_TNO, task_cat, task_prob, M)
        return float(load), float(eff)

    param_ranges = [(0, M - 1)] * N
    optimizer = AntColonyOptimizer(
        func=evaluate,
        param_ranges=param_ranges,
        n_ants=30,
        n_iter=40,
    )
    result = optimizer.run()
    import matplotlib.pyplot as plt

    def plot_learning(history):
        gens = [h["gen"] for h in history]
        fig, ax1 = plt.subplots()

        ax1.set_xlabel("Поколение")
        ax1.set_ylabel("Best load / Avg load")
        ax1.plot(gens, [h["best_load"] for h in history], label="best load", linestyle="--")
        ax1.plot(gens, [h["avg_load"]  for h in history], label="avg load")
        ax1.invert_yaxis()                     # если меньший load лучше
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Best / Avg efficiency")
        ax2.plot(gens, [h["best_eff"] for h in history], label="best eff", color="tab:red", linestyle="--")
        ax2.plot(gens, [h["avg_eff"]  for h in history], label="avg eff",  color="tab:red")
        ax2.legend(loc="upper right")

        plt.title("Процесс обучения ACO")
        plt.tight_layout()
        plt.show()

    plot_learning(result["history"])

    print("Pareto front size:", len(result["pareto_front"]))
    for params, scores in result["pareto_front"]:
        print(f"load={scores[0]:.2f}, eff={scores[1]:.2f}")



if __name__ == "__main__":
    main()
