import abc
import numpy as np
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from P2PSystemSim.OptimisationProblem import CommonProblem

class OptimisationAlgorithm(metaclass=abc.ABCMeta):
    def __init__(self, *args, ** kwargs):
        if type(self) == OptimisationAlgorithm:
            raise TypeError(" OptimisationALgorithm is an interface, it cannot be instantiated. Try with a concrete algorithm, e.g.: NSGAII")

    @abc.abstractmethod
    def operate(self):
        pass

class NSGAII(OptimisationAlgorithm):
    def __init__(self, commonProblem):
        self._commonProblem = commonProblem
    def operate(self):
        algorithm = NSGA2(
            pop_size=60,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 100)

        res = minimize(self._commonProblem,
                       algorithm,
                       termination,
                       seed=1,
                       pf=self._commonProblem.pareto_front(use_cache=False),
                       save_history=True,
                       verbose=True)
        return res.pop.get["X"][0]
class G_A(OptimisationAlgorithm):
    def __init__(self, commonProblem):
        self._commonProblem = commonProblem
    def operate(self):
        algorithm = GA(pop_size=100, eliminate_duplicates=True)

        res = minimize(self._commonProblem,
                       algorithm,
                       termination=('n_gen', 100),
                       seed=1,
                       verbose=True)
        # todo: changer avec le meilleur individu!!!!!!!!!!!!!!!
        return res.pop.get("X")[0]

class Optimiser:
    def __init__(self, algorithm):
        self._algorithm = algorithm
    def optimise(self):
        return self._algorithm.operate()
        #todo : add visualisation or other manipulations


