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
        print("_init_")

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

termination = get_termination("n_gen", 50)

if __name__ == '__main__':
    vectorized_problem = MyProblem()

    # functional interface 
    res = minimize(vectorized_problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

    plot = Scatter(title = "Objective Space")
    plot.add(res.F, color="red")
    plot.show()

    print(res.X)
    
    plot = Scatter(title = "Results Space")
    plot.add(res.X, color="red")
    plot.show()