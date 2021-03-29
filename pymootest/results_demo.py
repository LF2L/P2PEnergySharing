from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_problem
from pymoo.optimize import minimize


problem = get_problem("sphere")
algorithm = GA(pop_size=5)
res = minimize(problem,algorithm,('n_gen', 30),seed=1)
