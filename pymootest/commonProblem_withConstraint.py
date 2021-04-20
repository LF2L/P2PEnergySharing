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

intialize = [ 9.73866292e+03, 9.91058508e+03, 1.01783511e+04, 1.02120353e+04, 1.07155201e+04, 9.39147996e+03, 1.02231931e+04, 9.92657678e+03, 1.10126540e+04, 8.37499275e+03, 9.82597517e+03, 1.00794181e+04, 9.20061763e+03, 8.89005371e+03, 9.98110480e+03, 9.79998475e+03, 1.01741315e+04, 1.02469587e+04, 1.01607629e+04, 9.99768359e+03, 8.58401433e+03, 9.97627929e+03, 9.83293494e+03, 1.02252002e+04, 9.66792122e+03, 1.17606314e+04, 1.45682787e+04, 1.60168386e+04, 1.20651353e+04, 1.03805988e+04, 9.58702700e+03, 9.10718327e+03, 9.67007792e+03, 1.00378795e+04, 9.10718989e+03, 1.09298617e+04, 1.03288421e+04, 1.10988575e+04, 1.00024361e+04, 9.67572326e+03, 1.07285194e+04, 1.12256446e+04, 1.21433862e+04, 1.53133058e+04, 1.21248602e+04, 1.16658324e+04, 1.06699181e+04, 1.08533716e+04, 1.22521270e+04, 1.24308186e+04, 1.33556376e+04, 1.29146117e+04, 1.33389131e+04, 1.42819636e+04, 1.28043653e+04, 1.42455220e+04, 1.70658028e+04, 1.78633712e+04, 1.80258420e+04, 1.59102399e+04, 1.41085709e+04, 1.37792181e+04, 1.44247337e+04, 1.43779643e+04, 1.36297830e+04, 1.21750709e+04, 1.30191347e+04, 1.27956556e+04, 1.44153272e+04, 1.56610996e+04, 1.55001069e+04, 1.67733271e+04, 1.47947905e+04, 1.45696328e+04, 1.51298999e+04, 1.42453911e+04, 1.38332571e+04, 1.43617005e+04, 1.38258130e+04, 1.20211557e+04, 1.12162338e+04, 1.07375574e+04, 1.15592902e+04, 1.21596773e+04, 1.19641445e+04, 1.23547742e+04, 1.19710131e+04, 1.24709305e+04, 1.20722737e+04, 1.32144960e+04, 9.93255060e+03, 1.44894921e+04, 1.59928635e+04, 1.45014126e+04, 1.11299274e+04, 1.18857695e+04, 1.09629612e+04, 1.18439213e+04, 1.00250168e+04, 1.00074954e+04, 9.59950754e+03, 1.09951957e+04, 1.07540346e+04, 9.55572190e+03, 9.93953336e+03, 1.11807750e+04, 1.09008244e+04, 1.07467943e+04, 1.05787039e+04, 1.11580788e+04, 1.19593699e+04, 1.10903665e+04, 1.05772586e+04, 1.15040302e+04, 1.07569433e+04, 1.09216073e+04, 8.67284900e+03, 9.68593755e+03, 9.61479569e+03, 1.10000950e+04, 9.59094701e+03, 1.01067009e+04, 9.55808200e+03, 1.01380518e+04, 9.61155474e+03, 1.30468296e+04, 1.55189131e+04, 1.46584316e+04, 1.04312221e+04, 9.65098114e+03, 9.71893920e+03, 8.92272268e+03, 9.83877093e+03, 9.07814501e+03, 9.10977722e+03, 8.92714220e+03, 9.77690370e+03, 9.79598293e+03, 9.69089331e+03, 8.67546364e+03, 9.79082780e+03, 1.07576348e+04, 1.09983052e+04, 1.00018229e+04, 8.45582876e+02, -8.47450453e+00, -2.87758653e+02, -3.27180923e+02, 1.27250371e+01, 5.01665179e+02, -3.39659291e+02, -4.89255682e+01, -2.90963779e+02, 5.34761873e+02, -9.87100789e+01, -1.97772064e+02, -2.93294458e+02, -1.69390472e+00, -2.48445357e+02, 8.75874626e+01, -2.91510413e+02, -3.57130675e+02, -2.81536380e+02, -1.13947363e+02, 3.12729619e+02, -2.24683771e+02, 4.71082714e+01, -3.48349042e+02, 2.19511341e+02, -2.07258973e+02, -2.26219423e+02, -3.42246716e+02, -1.72976987e+02, 5.19887831e+02, 3.13210727e+02, -2.15042935e+02, 6.03299247e+01, -1.66707693e+02, 7.67158645e+02, -2.03787681e+02, 5.43084524e+02, -2.44004919e+02, -1.10658672e+02, 2.25821580e+02, 8.85538176e+00, -3.59646494e+02, 4.08931959e+02, 4.27487965e+01, -2.31178678e+02, 1.72976297e+02, 2.05917883e+02, 8.63687824e+02, -3.48858352e+02, -3.54823510e+02, -1.51295374e+02, -1.44101221e+02, -3.51515841e+02, -1.84628192e+02, -1.82379579e+02, 5.63389554e+01, 1.44452509e+01, -3.24561162e+02, -1.09854370e+02, -3.55702627e+02, -1.51425143e+01, -1.35839356e+02, -3.57028486e+02, -1.08187768e+02, 1.58461895e+02, 3.83222705e+01, -3.95321553e+01, 4.81008443e+02, 4.01103904e+02, 2.69841091e+01, 8.30914686e+02, -2.14712154e+02, -3.15600396e+02, 4.17985638e+02, -1.81136073e+02, -1.36844194e+02, -4.22959899e+01, 6.79492903e+02, 1.50654885e+02, -1.18206045e+02, -6.59339738e+01, -3.57636904e+02, 1.15750770e+03, 1.15515197e+03, 2.29053571e+02, -3.57738988e+02, 1.15698069e+02, 6.17735111e+02, 1.69332936e+02, -2.44318841e+02, -2.31388486e+01, 2.61396777e+01, -1.85811725e+02, -1.56883937e+02, 5.80713332e+01, -6.00141030e+01, -3.51500035e+02, 8.22419656e+01, -5.05520583e+01, -1.30328042e+02, 1.14272931e+03, -1.20939275e+02, 1.21773533e+02, 3.27688360e+02, -7.05494170e+01, 3.65298187e+02, -3.95915249e+01, 1.04955252e+02, 1.14259958e+03, -2.83566210e+02, -2.43477853e+02, -2.18074445e+02, 2.88218357e+02, 2.09012378e+02, 1.26441581e+02, -6.78331361e+01, 1.23189778e+03, 1.80979439e+02, 2.71967772e+02, -2.67976491e+02, 2.93146202e+02, -2.27280244e+02, 3.14638303e+02, -2.61972984e+02, 2.68798810e+02, 1.65674635e+02, 4.99451131e+00, 1.72068341e+02, -5.73850935e+01, 2.32692576e+02, 1.60842984e+02, -1.62668950e+01, -8.83570542e+01, -1.77303373e+02, 6.30042555e+02, -1.69016565e+01, -3.32411125e+01, 9.46326268e+01, 2.04761543e+02, 2.26700772e+02, -5.01632276e+01, -2.76245297e+01, -1.23241188e+02, -9.94871938e+01]

# battery 
SOCmin = 0.2
SOCmax = 0.8
selfDischarge = 0 # sigma
charge_efficiency = 1
discharge_efficiency = 1
nominalCapacity = 1800 # kWh
initial_value = 900 #kWh
current_capa = 1000 #kWh

#problem parameters
timeslot_duration = 10*60 # seconds 
timeslot_duration_hours = 10/60 # hours 

# pymoo problem
populationSize = 600
# lower_bound = np.array([-50000, SOCmin*nominalCapacity]) # lower bound imported energy  == exportation 
# upper_bound = np.array([100000, SOCmax*nominalCapacity]) # upper bound imported energy  == imporation  

battery_lower_bound = - SOCmax*nominalCapacity *np.ones(nbOfStepInOneDay)
battery_upper_bound = SOCmax*nominalCapacity *np.ones(nbOfStepInOneDay)

lower_bound_powerExchange = -max(REgenerationForecast) * np.ones(nbOfStepInOneDay)
upper_bound_powerExchange = max(loadForecast) * np.ones(nbOfStepInOneDay)

lower_bound = np.concatenate((lower_bound_powerExchange,battery_lower_bound), axis=None)
upper_bound = np.concatenate((upper_bound_powerExchange, battery_upper_bound), axis=None)

threshold = 1e-1

class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=288,
                         n_obj=1,
                         n_constr=3,
                         xl=lower_bound,
                         xu=upper_bound)

    def _evaluate(self, X, out, *args, **kwargs):
        
        #f1 = np.where(X[:, 0]<0,X[:, 0],0)*FeedInTariff + np.where(X[:, 0]>0,X[:, 0],0) * gridPrices
        f1 = (np.where(X[:,:144]<0,X[:,:144],0)*FeedInTariff + np.where(X[:,:144]>0,X[:,:144],0) * gridPrices ).sum(axis=1)

        
        # f1 = X[:,0]**2 + X[:,1]**2
        # f2 = (X[:,0]-1)**2 + X[:,1]**2

        # g1 = 2*(X[:, 0]-0.1) * (X[:, 0]-0.9) / 0.18
        # g2 = - 20*(X[:, 0]-0.4) * (X[:, 0]-0.6) / 4.8
        #balancing_requerements =  np.sum([X[:, 0], loadForecast , REgenerationForecast])

        # power_battery_levels = np.array([  sum(X[:,144:index+1]) for index in range(0, len(X[:,144:]))])
        # min constraint induced by the battery
        prev_values = np.roll(X,1,axis=1)
        prev_values[:,144] = initial_value
        # g1 =  SOCmin * nominalCapacity - prev_values[:,144:] *(1- selfDischarge) - (np.where(X[:,144:]<0,X[:,144:],0) * charge_efficiency - np.where(X[:,144:]>0,X[:,144:],0) * discharge_efficiency ) * timeslot_duration_hours
        # g1 = SOCmin * nominalCapacity - prev_values[:,144:] + X[:,144:] 
        g1 = SOCmin * nominalCapacity - np.cumsum(prev_values[:,144:],axis=1) + np.cumsum(X[:,144:], axis =1)

        # max contraint induced by the battery
        # g2 = prev_values[:,144:] *(1- selfDischarge) + (np.where(X[:,144:]<0,X[:,144:],0) * charge_efficiency - np.where(X[:,144:]>0,X[:,144:],0) * discharge_efficiency ) * timeslot_duration_hours - SOCmax * nominalCapacity
        # g2 = prev_values[:,144:] - X[:,144:] - SOCmax * nominalCapacity
        g2 = np.cumsum(prev_values[:,144:],axis=1)  - np.cumsum(X[:,144:], axis=1) - SOCmax * nominalCapacity

        #g3 = np.sum([X[:, 0], loadForecast , REgenerationForecast, current_capa]) 
        #g3 = X[:, 0] + loadForecast + REgenerationForecast + current_capa + threshold 
        #g3 = (X[:, 0] - loadForecast - REgenerationForecast - current_capa)**2 - threshold
        #g3 = (X[:, 0] + loadForecast + REgenerationForecast + X[:, 1])**2 - threshold
        g3 = (X[:,:144] - loadForecast + REgenerationForecast +  X[:,144:])**2 - threshold
        # g4 = (X[:,144] - initial_value)**2 - threshold
        #g3 = (X + loadForecast + REgenerationForecast)**2 - threshold
        #g5 = SOCmin * nominalCapacity - np.cumsum(X[:,144:],axis=1) #- initial_value
        #g6 = np.cumsum(X[:,144:],axis=1)  - SOCmax * nominalCapacity #+ initial_value

        out["F"] = np.column_stack([f1])
        out["G"] = np.column_stack([g1, g2, g3])

if __name__ == '__main__':
    init = np.sum([[loadForecast], [REgenerationForecast]], axis=0)
    print(init.shape)

    # populationSize = param['pop_size'] if hasattr(param, 'pop_size') else 100
    termination = get_termination("n_gen", 2000)

    vectorized_problem = MyProblem()


    init_design_space = np.zeros((populationSize,vectorized_problem.n_var))
    init_design_space[:,:144] = init
    init_design_space[:,144] = initial_value

    algorithm = GA(pop_size= populationSize, eliminate_duplicates=True, sampling=init_design_space)
    # algorithm = GA(pop_size= populationSize, eliminate_duplicates=True)
    # algorithm = NSGA2(pop_size= populationSize, eliminate_duplicates=True)

    res = minimize(vectorized_problem,
                       algorithm,
                       termination= termination,
                       return_least_infeasible=True,
                       seed=1,
                       save_history=True,
                       verbose= True)

    print(res.X)
    # print(res.F)
    print(res.G)
    print(res.G.shape)
    
    # plot = Scatter()
    # plot.add(res.X, color="red")
    # plot.show()

    plot = Scatter(title = "Objective Space")
    plot.add(res.F)
    plot.show()

    # plot = Scatter()
    # plot.add(res.X, color="orange")
    # plot.show()

    balance_check = res.X[:144] + REgenerationForecast - loadForecast + res.X[144:]
    print(balance_check)

    # battery_mvt = res.X[144:].tolist()
    # battery_power_level = [ (initial_value + sum(battery_mvt[:index+1])) for index in range(0, len(battery_mvt))]
    # battery_power_level_init = [  sum(battery_mvt[:index+1]) for index in range(0, len(battery_mvt))]

    battery_power_level = np.cumsum(res.X[144:])


    fig, axs = plt.subplots(1,4)
    axs[0].plot(range(0,len(REgenerationForecast)), REgenerationForecast, label="RE generation")
    axs[0].plot(range(0,len(loadForecast)), loadForecast, label="Load forecast")
    axs[0].plot(range(0,len(res.X[:144])), res.X[:144], label="Power exchanged")

    axs[0].set(xlabel='timeslots', ylabel='Power (Wh)', title='Power consumption and production ')
    axs[0].grid()
    axs[0].legend()
    #fig.savefig("test.png")
    axs[1].plot(range(0,len(balance_check)), balance_check, label="Balance")
    axs[1].set(xlabel='timeslots', ylabel='Power (Wh)', title='Balance for each timeslot')
    axs[1].legend()

    axs[2].plot(range(0,len(res.X[144:])), res.X[144:], label="battery power transfer")
    axs[2].set(xlabel='timeslots', ylabel='Power (Wh)', title='Battery charge and discharge')
    axs[2].legend()

    axs[3].plot(range(0,len(battery_power_level)), battery_power_level, label="battery state")
    # axs[3].plot(range(0,len(balance_check)), balance_check, label="battery state")
    axs[3].hlines(y=SOCmin*nominalCapacity, xmin = 0 , xmax = len(res.X[144:]), label="level Min")
    axs[3].hlines(y=SOCmax*nominalCapacity, xmin = 0 , xmax = len(res.X[144:]), label="level Min")
    axs[3].set(xlabel='timeslots', ylabel='Power (Wh)', title='Battery State')
    axs[3].legend()

    # axs[3].plot(range(0,len(res.X[144:])), res.X[144:], label="Balance")
    # axs[3].set(xlabel='timeslots', ylabel='Power (Wh)', title='Battery charge and discharge')
    # axs[3].legend()
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