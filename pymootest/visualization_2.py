import numpy as np
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, X, out, *args, **kwargs):
        f1 = X[:,0]**2 + X[:,1]**2
        f2 = (X[:,0]-1)**2 + X[:,1]**2

        g1 = 2*(X[:, 0]-0.1) * (X[:, 0]-0.9) / 0.18
        g2 = - 20*(X[:, 0]-0.4) * (X[:, 0]-0.6) / 4.8

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 40)

if __name__ == '__main__':
    vectorized_problem = MyProblem()

    # functional interface 
    res = minimize(vectorized_problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

    n_evals = []    # corresponding number of function evaluations\
    F = []          # the objective space values in each generation
    cv = []         # constraint violation in each generation


    # iterate over the deepcopies of algorithms
    for algorithm in res.history:

        # store the number of function evaluations
        n_evals.append(algorithm.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algorithm.opt

        # store the least contraint violation in this generation
        cv.append(opt.get("CV").min())

        # filter out only the feasible and append
        feas = np.where(opt.get("feasible"))[0]
        _F = opt.get("F")[feas]
        F.append(_F)