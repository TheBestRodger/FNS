import numpy as np
import random
from typing import List, Tuple, Callable, Dict, Any

class Solution:
    def __init__(self, params: List[int], scores: Tuple[float, ...]):
        self.params = params
        self.scores = scores

    def __eq__(self, other):
        return tuple(self.params) == tuple(other.params)

    def __hash__(self):
        return hash(tuple(self.params))

    def dominates(self, other) -> bool:
        better_or_eq = all(a <= b for a, b in zip(self.scores, other.scores))
        better = any(a < b for a, b in zip(self.scores, other.scores))
        return better_or_eq and better

class ParetoArchive:
    def __init__(self):
        self.archive = set()

    def update(self, candidates: List[Solution]):
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
        evaporation: float = 0.5
    ):
        self.func = func
        self.dim = len(param_ranges)
        self.ranges = param_ranges
        self.n_ants = n_ants
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromones = [np.ones(r[1] - r[0] + 1) for r in param_ranges]
        self.solutions = []

    def sample_ant(self) -> List[int]:
        params = []
        for d in range(self.dim):
            tau = self.pheromones[d]
            possible_vals = list(range(self.ranges[d][0], self.ranges[d][1] + 1))
            eta = np.ones_like(tau)
            probs = (tau ** self.alpha) * (eta ** self.beta)
            probs = probs / probs.sum()
            val = random.choices(possible_vals, weights=probs)[0]
            params.append(val)
        return params

    def update_pheromones(self, pareto_solutions: List[Solution]):
        for d in range(self.dim):
            self.pheromones[d] *= (1 - self.evaporation)
        for sol in pareto_solutions:
            for d, val in enumerate(sol.params):
                self.pheromones[d][val - self.ranges[d][0]] += 1

    def run(self) -> Dict[str, Any]:
        archive = ParetoArchive()
        for it in range(self.n_iter):
            candidates = []
            for ant in range(self.n_ants):
                params = self.sample_ant()
                scores = self.func(params)
                sol = Solution(params, scores)
                candidates.append(sol)
                self.solutions.append(sol)
            archive.update(candidates)
            self.update_pheromones(archive.get())
        return {
            "explored": [(sol.params, sol.scores) for sol in self.solutions],
            "pareto_front": [(sol.params, sol.scores) for sol in archive.get()]
        }
def test_func(params: List[int]) -> Tuple[float, float]:
    # Минимизируем сумму квадратов (критерий 1) и сумму отклонений от [1,2,3,4] (критерий 2)
    crit1 = sum(x ** 2 for x in params)
    crit2 = sum((x - target) ** 2 for x, target in zip(params, [1, 2, 3, 4]))
    return (crit1, crit2)

param_ranges = [(-10, 10)] * 4

optimizer = AntColonyOptimizer(
    func=test_func,
    param_ranges=param_ranges,
    n_ants=30,
    n_iter=60
)
result = optimizer.run()

print("Все изученные параметры и их оценки (первые 10):")
for params, scores in result["explored"][:10]:
    print(f"params={params}, f={scores}")
print(f"\nВсего исследовано: {len(result['explored'])}")
print("\nМножество не-доминируемых решений (Парето):")
for params, scores in result["pareto_front"]:
    print(f"params={params}, f={scores}")