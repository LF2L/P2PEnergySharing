import numpy as np
import matplotlib.pyplot as plt
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
#from P2PSystemSim.OptimisationProblem import CommonProblem
from abc import ABC, abstractmethod

class OptimisationAlgorithm(ABC):
    def __init__(self, optimisationProblem, *args, ** kwargs):
        self._problem = optimisationProblem
        if type(self) == OptimisationAlgorithm:
            raise TypeError(" OptimisationALgorithm is an interface, it cannot be instantiated. Try with a concrete algorithm, e.g.: NSGAII")

    @abstractmethod
    def operate(self, **param):
        pass

class NSGAII(OptimisationAlgorithm):
    def __init__(self, optimisationProblem):
        super().__init__(optimisationProblem)

    def operate(self, **param):
        algorithm = NSGA2(
            pop_size=60,
            n_offsprings=10,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )

        termination = get_termination("n_gen", 100)

        res = minimize(self._problem,
                       algorithm,
                       termination,
                       seed=1,
                       pf=self._problem.pareto_front(use_cache=False),
                       save_history=True,
                       verbose=True)
        return res.pop.get["X"][0]

class G_A(OptimisationAlgorithm):
    def __init__(self, optimisationProblem):
        super().__init__(optimisationProblem)

    def operate(self, **param):
        algorithm = GA(pop_size= param['pop_size'] if hasattr(param, 'pop_size') else 100, eliminate_duplicates=True)

        res = minimize(self._problem,
                       algorithm,
                       termination=('n_gen', param['termination'] if hasattr(param, 'termination') else 100),
                       seed=1,
                       verbose=param['verbose'] if hasattr(param, 'verbose') else False)
        # todo: changer avec le meilleur individu!!!!!!!!!!!!!!!
        return res.pop.get("X")[0]

class Optimiser:

    def __init__(self, algorithm):
        self._algorithm = algorithm

    def optimise(self, **optiParam ):
        self.optimisationResults = self._algorithm.operate(param= optiParam)
        self.displayGraph()
        return self.optimisationResults
        #todo : add visualisation or other manipulations

    def displayGraph(self, title = 'Power exchange optimisation'):
        fig, ax = plt.subplots()
        ax.plot(range(0,len(self.optimisationResults)), self.optimisationResults, label="power (Wh)")
        # ax.plot(range(0,len(self._loadForecast)), self._loadForecast, label="Load forecast")
        #ax.plot(len(self._REgeneration), self._REgeneration, len(self._loadForecast), self._loadForecast)
        ax.set(xlabel='timeslots', ylabel='Power (Wh)', title='{}'.format( title))
        ax.grid()
        ax.legend( loc='upper left', borderaxespad=0.)
        plt.show()



