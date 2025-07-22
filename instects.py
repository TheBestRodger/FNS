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
        for _ in range(self.n_iter):
            candidates = []
            for _ in range(self.n_ants):
                params = self.sample_ant()
                scores = self.func(params)
                sol = Solution(params, scores)
                candidates.append(sol)
                self.solutions.append(sol)
            archive.update(candidates)
            self.update_pheromones(archive.get())
        return {
            "explored": [(sol.params, sol.scores) for sol in self.solutions],
            "pareto_front": [(sol.params, sol.scores) for sol in archive.get()],
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

    print("Pareto front size:", len(result["pareto_front"]))
    for params, scores in result["pareto_front"]:
        print(f"load={scores[0]:.2f}, eff={scores[1]:.2f}")


if __name__ == "__main__":
    main()
