import numpy as np
import random
from pymoo.model.problem import Problem
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.running_metric import RunningMetric
import matplotlib.pyplot as plt

nbOfStepInOneDay = 144

FeedInTariff = 0.035 * np.ones(nbOfStepInOneDay) 
#gridPrices = [random.random()/1000 for i in range(nbOfStepInOneDay)]
gridPrices = 0.040 * np.ones(nbOfStepInOneDay)

#loadForecast = [16000.0] #Wh
loadForecast = [10570.79861419492, 9896.136453664507, 9885.244932495456, 9885.244932495456, 10733.818947452446, 9896.136453664507, 9885.244932495456, 9885.244932495456, 10733.818947452446, 8918.82180835372, 9743.490562944387, 9885.244932495456, 8907.93028718467, 8897.038766015621, 9743.490562944387, 9885.244932495456, 9885.244932495456, 9885.244932495456, 9885.244932495456, 9885.244932495456, 8907.93028718467, 9743.490562944387, 9885.244932495456, 9885.244932495456, 9885.244932495456, 11561.12699870298, 14369.091818969626, 15685.460869992732, 11905.223350131322, 10884.342620144342, 9896.136453664507, 8907.93028718467, 9743.490562944387, 9885.244932495456, 9885.244932495456, 10733.818947452446, 10873.451098975293, 10873.451098975293, 9896.136453664507, 9885.244932495456, 10733.818947452446, 10873.451098975293, 12551.455383211036, 15359.420203477683, 11905.223350131322, 11861.65726545513, 10884.342620144342, 11724.147331960503, 11920.27268862229, 12120.150059480364, 13304.278504751923, 12905.504220659457, 13149.880355598103, 14294.088742882088, 12818.199215771041, 14506.632590702733, 17294.67675088221, 17857.94400153461, 18344.208616415784, 15988.263211221112, 14534.377234459544, 14098.00513576139, 14594.999257616153, 14885.890687194164, 14395.215017521745, 12855.501398197042, 13693.320253029862, 14120.642344036496, 15719.500928579606, 16570.09712519503, 17149.02447075413, 17204.923583033593, 15015.832599743377, 15602.883295143325, 15546.921144321353, 14666.169950383946, 14206.595665473073, 15481.18509767805, 14423.18116934674, 12328.175976375349, 11748.139795773293, 11057.696582498775, 13299.069013058113, 13737.38076521029, 12525.931042167682, 12395.195755169774, 12568.419850020959, 13550.085113882718, 12584.219850398022, 13301.99197547371, 10274.63187140532, 14880.811605879908, 16119.018648432104, 14615.385901606984, 11373.267644870246, 11916.11690733929, 10641.240610853065, 11956.245794796056, 9998.543873889286, 9886.22477730591, 10733.818947452446, 10873.451098975293, 10873.451098975293, 9896.136453664507, 9885.244932495456, 11561.12699870298, 10884.342620144342, 10873.451098975293, 11724.147331960503, 10884.342620144342, 11724.147331960503, 10884.342620144342, 10873.451098975293, 11724.147331960503, 10884.342620144342, 10873.451098975293, 9896.136453664507, 9885.244932495456, 9885.244932495456, 10733.818947452446, 9896.136453664507, 9885.244932495456, 9885.244932495456, 9885.244932495456, 9885.244932495456, 13215.743101204045, 15522.440536735208, 14837.167286063685, 10373.264878026266, 9885.244932495456, 9885.244932495456, 8907.93028718467, 9743.490562944387, 8907.93028718467, 9743.490562944387, 8907.93028718467, 9743.490562944387, 9885.244932495456, 9885.244932495456, 8907.93028718467, 9743.490562944387, 10733.818947452446, 10873.451098975293, 9896.136453664507]
#REgenerationForecast = [2448.0] #W
REgenerationForecast = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.6, 38.4, 86.39999999999999, 139.2, 163.2, 182.4, 192.0, 182.4, 225.60000000000002, 307.2, 432.0, 441.59999999999997, 427.20000000000005, 456.0, 528.0, 595.2, 595.2, 624.0, 696.0, 840.0, 897.6, 883.1999999999999, 782.4000000000001, 624.0, 513.5999999999999, 580.8, 566.4, 513.5999999999999, 432.0, 422.4, 432.0, 432.0, 580.8, 686.3999999999999, 561.5999999999999, 408.0, 312.0, 398.4, 456.0, 451.20000000000005, 326.4, 312.0, 336.0, 336.0, 297.6, 244.79999999999998, 168.0, 86.39999999999999, 24.0, 19.2, 9.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# battery 
SOCmin = 0.2
SOCmax = 0.8
selfDischarge = 0 # sigma
charge_efficiency = 1
discharge_efficiency = 1
nominalCapacity = 1800 # kWh
previous_value = 900 #kWh
current_capa = 1000 #kWh

#problem parameters
timeslot_duration = 10*60 # seconds 
timeslot_duration_hours = 10/60 # hours 

# pymoo problem
populationSize = 1000
populationSize = 500
# lower_bound = np.array([-50000, SOCmin*nominalCapacity]) # lower bound imported energy  == exportation 
# upper_bound = np.array([100000, SOCmax*nominalCapacity]) # upper bound imported energy  == imporation  

lower_bound = -50000 * np.ones(nbOfStepInOneDay)
upper_bound = 100000 * np.ones(nbOfStepInOneDay)

threshold = 1e-4

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=144,
                         n_obj=1,
                         n_constr=1,
                         xl=lower_bound,
                         xu=upper_bound)

    def _evaluate(self, X, out, *args, **kwargs):
        #print(X.shape)
        #f1 = np.where(X[:, 0]<0,X[:, 0],0)*FeedInTariff + np.where(X[:, 0]>0,X[:, 0],0) * gridPrices
        f1 = (np.where(X[:,:144]<0,X[:,:144],0)*FeedInTariff + np.where(X[:,:144]>0,X[:,:144],0) * gridPrices ).sum(axis=1)
        #print(f1.shape)

        
        # f1 = X[:,0]**2 + X[:,1]**2
        # f2 = (X[:,0]-1)**2 + X[:,1]**2

        # g1 = 2*(X[:, 0]-0.1) * (X[:, 0]-0.9) / 0.18
        # g2 = - 20*(X[:, 0]-0.4) * (X[:, 0]-0.6) / 4.8
        #balancing_requerements =  np.sum([X[:, 0], loadForecast , REgenerationForecast])
        # min constraint induced by the battery
        #g1 =  SOCmin * nominalCapacity - previous_value *(1- selfDischarge) + (np.where(X[:, 1]>0,X[:, 1],0) * charge_efficiency + np.where(X[:, 1]<0,X[:, 1],0) * discharge_efficiency ) * timeslot_duration_hours
        
        # max contraint induced by the battery
        #g2 = previous_value *(1- selfDischarge) + (np.where(X[:, 1]>0,X[:, 1],0) * charge_efficiency + np.where(X[:, 1]<0,X[:, 1],0) * discharge_efficiency ) * timeslot_duration_hours - SOCmax * nominalCapacity

        #g3 = np.sum([X[:, 0], loadForecast , REgenerationForecast, current_capa]) 
        #g3 = X[:, 0] + loadForecast + REgenerationForecast + current_capa + threshold 
        #g3 = (X[:, 0] - loadForecast - REgenerationForecast - current_capa)**2 - threshold
        #g3 = (X[:, 0] + loadForecast + REgenerationForecast + X[:, 1])**2 - threshold
        g3 = (X[:,:144] - loadForecast - REgenerationForecast )**2 - threshold
        #g3 = (X + loadForecast + REgenerationForecast)**2 - threshold

        out["F"] = np.column_stack([f1])
        out["G"] = np.column_stack([g3])

if __name__ == '__main__':

    # populationSize = param['pop_size'] if hasattr(param, 'pop_size') else 100
    termination = get_termination("n_gen", 300)
    algorithm = GA(pop_size= populationSize, eliminate_duplicates=True)
    # algorithm = NSGA2(pop_size= populationSize, eliminate_duplicates=True)

    vectorized_problem = MyProblem()

    res = minimize(vectorized_problem,
                       algorithm,
                       termination,
                       return_least_infeasible=True,
                       seed=1,
                       save_history=True,
                       verbose= True)

    print(res.X)
    print(res.F)
    
    # plot = Scatter()
    # plot.add(res.X, color="red")
    # plot.show()

    plot = Scatter(title = "Objective Space")
    plot.add(res.F)
    plot.show()

    # plot = Scatter()
    # plot.add(res.X, color="orange")
    # plot.show()

    balance_check = res.X - REgenerationForecast - loadForecast
    print(balance_check)

    fig, axs = plt.subplots(1,2)
    axs[0].plot(range(0,len(REgenerationForecast)), REgenerationForecast, label="RE generation")
    axs[0].plot(range(0,len(loadForecast)), loadForecast, label="Load forecast")
    axs[0].plot(range(0,len(res.X)), res.X, label="Power exchanged")

    axs[0].set(xlabel='timeslots', ylabel='Power (Wh)', title='Power consumption and production ')
    axs[0].grid()
    axs[0].legend()
    #fig.savefig("test.png")
    axs[1].plot(range(0,len(balance_check)), balance_check, label="Balance")
    axs[1].set(xlabel='timeslots', ylabel='Power (Wh)', title='Balance for each timeslot')
    axs[1].legend()
    #lines, labels = fig.axes[-1].get_legend_handles_labels()
    #fig.legend(lines,labels, loc='upper right')
    plt.show()

    # running = RunningMetric(delta_gen=10,
    #                     n_plots=5,
    #                     only_if_n_plots=True,
    #                     key_press=False,
    #                     do_show=True)

    # for algorithm in res.history[:50]:
    #     running.notify(algorithm)