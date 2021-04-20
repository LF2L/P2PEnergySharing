import numpy as np

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.model.problem import Problem
from pymoo.optimize import minimize


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=8,
                         n_obj=1,
                         n_constr=0,
                         elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        to_int = (x * [128, 64, 32, 16, 8, 4, 2, 1]).sum()
        out["F"] = - np.sin(to_int / 256 * np.pi)


problem = MyProblem()

algorithm = GA(
    pop_size=10,
    sampling=get_sampling("bin_random"),
    crossover=get_crossover("bin_ux"),
    mutation=get_mutation("bin_bitflip"))

res = minimize(problem,
               algorithm,
               ("n_gen", 10),
               seed=1,
               verbose=True)

print("X", res.X.astype(np.int))
print("F", - res.F[0])