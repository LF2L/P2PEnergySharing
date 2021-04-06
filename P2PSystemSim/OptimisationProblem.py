
#from Coordinator import Prosumer
#from DReventDispatch import *
from P2PSystemSim.Assets import *
from P2PSystemSim.Optimiser import *
#from utils import *
from multiprocessing import Pool
#from copy import deepcopy
from functools import partial
import numpy as np

class CommonProblem(Problem): # extends the problem object from Pymoo

    def __init__(self,loadForecast, REgenerationForecast, **kwargs):
        assert(len(loadForecast) == len(REgenerationForecast))
        self.BatteryList = kwargs["batteryList"]
        totalNominalCapacity = sum([battery._nominalCapacity for battery in self.BatteryList])
        self.loadForecast = loadForecast
        self.REgenerationForecast = REgenerationForecast
        self.Parameters = kwargs
        self.nbTimeSlots = len(self.loadForecast)
        """
        :param loadForecast: list of predicted energy loads with any granularity
        :param REgenerationForecast: list of predicted RE generation capacity with any granularity
        :param kwargs: dictionary that includes: batteryList (objects of class Battery), nominalCapacity (battery size), grid energy prices and feed-in
        tariffs for each time slot, battery charge and discharge efficiencies and battery self-discharge
        """
        timeSlot = 86400/self.nbTimeSlots #timeslot duraction in seconds
        xl = np.zeros(self.nbTimeSlots)
        xu = np.zeros(self.nbTimeSlots)
        for i in range(0, self.nbTimeSlots):
            xu[i] = self.loadForecast[i] - self.REgenerationForecast[i] + totalNominalCapacity/self.nbTimeSlots # the upper bound
            # (positive value) represents the total needed load minus the RE generation (which is equivalent to the energy
            # demand that wasn't met by RE generation) plus the battery size. In other words: we cover energy demand
            # that wasn't met through RE generation and charge the battery to its maximum capacity admitting it was fully uncharged
            xl[i] = - (totalNominalCapacity + self.REgenerationForecast[i])
            # the maximum i can sell is the maximum i can have available,
            # which is the maximum I can have in storage plus all i can produce
        super().__init__(n_var=len(loadForecast), n_obj=1, n_constr=1, xl=xl, xu=xu, elementwise_evaluation=False)

    # def calcCost(self, x, **kwargs):
    #     """
    #     :param x: one solution vector (size = nb of timeslot in a day)
    #     :param kwargs1:
    #     :return: real: total cost induced by this solution vector
    #     """

    #     s = 0
    #     for i in range(len(x)):
            
    #         if x[i] >= 0:
    #             # if transation is positive, apply grid price 
    #             s = s + x[i] * kwargs["gridPrices"][i]
    #         else:
    #             # if transation is nagtive, apply Feed in tariff 
    #             s = s + x[i] * kwargs["FIT"][i]

    #     return (s)

    def calcConstr(self, x, **kwargs2):
            """
            this function computes the constraint functions values (144 = 3*48 constraints)
            :param x: one solution vector, i.e. transactions with the grid for each time-slot of the day
            :param kwargs2: needed parameters like the REgenerationForecast vector, the loadForecast and others
            :return: constraint function value = temp array of how much pbd was not met /how much pbc couldn't fit for each time slot
            """

            # calculate vector that only records grid energy loads (feed-in values, i.e. negative values, are turned into zeros)
            #energy_import_from_grid  = np.where(x>0,x,0)
            #X = [x[i] if x[i]>=0 else 0 for i in range (len(x))]
            # calculate vector that only records feed-in energy in positive form (grid loads, i.e. positive values, are turned into zeros)
            #Y = [-x[i] if x[i]<=0 else 0 for i in range(len(x))]
            #energy_exported_to_grid = np.where(x<0,x,0)

            # Charge and discharge power calculation
            # in this model we cannot charge and discharge in the same time slot bc Y[i] isn't systematically drawn from battery
            # Pbc = np.zeros(len(x))
            # Pbd = np.zeros(len(x))
            # for i in range(len(x)):
            #     # surplus = X[i] + kwargs2["REgenerationForecast"][i] - kwargs2["loadForecast"][i] - Y[i]
                
            #     # surplus is internal sources (RE generation) and external (grid load = x(i)) minus destinations (feed-ins and net loads)
            #     if surplus>=0:
            #         Pbc[i] = surplus
            #     else:
            #         Pbd[i] = -surplus
            surplus = np.where(x>0,x,0) + kwargs2["REgenerationForecast"] - kwargs2["loadForecast"] + np.where(x<0,x,0)
            # power_battery_charing = np.where(surplus>0, surplus,0)
            # power_battery_discharing = np.where(surplus<0, - surplus,0)

            #create aggregation of batteries and attempt to charge/discharge with Pbc and Pbd temp arrays
            batteryagg = BatteryAggregation(kwargs2["batteryList"])
            operator = batteryagg.operator()
            G = operator.loadProcessing(surplus)
            # print(f"G: {G}")
            # G1 = operator.charge(power_battery_charing)
            # G2 = operator.discharge(power_battery_discharing)
            # G = np.zeros(self.nbTimeSlots)
            # for i in range(len(G)):
            #     G[i] = G1[i] if G1[i] != 0 else G2[i]

            return G

    def _evaluate(self, X, out, *args, **kwargs):
        #print(f"Number of variables: {len(X)}")
        """
        :param x: all solutions because elemenwise_evaluation = False
        :param out:
        :param args:
        :param kwargs:
        :return:
        """

        # generate the objectives for the optimization 
        # f1 = [self.calcCost(xi, gridPrices=self.Parameters["gridPrices"], FIT = self.Parameters["FIT"]) for xi in X] # return a vector

        # function nb 7 in the article
        # f1 = np.sum(np.where(X<0,X,0)*self.Parameters["FIT"] + np.where(X>0,X,0) * self.Parameters["gridPrices"] ) # return a scalar 
        f1 = np.sum(np.where(X<0,X,0)*self.Parameters["FIT"] + np.where(X>0,X,0) * self.Parameters["gridPrices"] ) * np.ones(len(X))
        out["F"] =  np.column_stack([f1])

        # generate the constraints for the optimization 
        g1 = [self.calcConstr(xi, REgenerationForecast = self.REgenerationForecast, loadForecast = self.loadForecast, batteryList=self.BatteryList) for xi in X]
        out["G"] = np.column_stack([g1])

class BiddingProblem(Problem):
    def __init__(self, loadForecast: list, REgenerationForecast: list, prosumerAgent, sellPrices, buyPrices):
        
        self.loadForecast = loadForecast
        self.REgenerationForecast = REgenerationForecast
        self.prosumer = prosumerAgent._prosumer
        self.proID= prosumerAgent._prosumer._ID
        self.shiftableLoadMatrix = self.prosumer._get_shiftableLoadMatrix()
        self.nbTimeSlots= len(loadForecast)
        self.buyPrices = buyPrices
        self.sellPrices = sellPrices
        xl = np.zeros(self.nbTimeSlots)
        xu = np.zeros(self.nbTimeSlots)
        for i in range(0, self.nbTimeSlots):
            # xl : lower variable boundaries
            xl[i] = self.loadForecast[i]*self.shiftableLoadMatrix[i] if self.shiftableLoadMatrix[i] <0 else self.loadForecast[i]   #if we can move some of the current power load to another time slot
            # xu : upper variable boundaries
            xu[i] = self.loadForecast[i]*self.shiftableLoadMatrix[i] if self.shiftableLoadMatrix[i]> 0 else self.loadForecast[i] #if we can add some more power load to the current time slot
        super().__init__(n_var=self.nbTimeSlots, n_obj=1, n_constr=1, xl=xl, xu=xu, elementwise_evaluation=False)
    
    def calcost(self, x,  **kwargs):
        """
        :param x: one solution = one temp list of netLoads after shifting
        :param prosumer:
        :return:
        """
        prosumer = self.prosumer
        prosumerProblem = CommonProblem(loadForecast=prosumer._get_loadForecast(), REgenerationForecast=
        prosumer._get_REgeneration(), batteryList= [prosumer._get_Battery()], gridPrices = self.buyPrices, FIT = self.sellPrices)
        prosumerOptimiser = Optimiser(G_A(optimisationProblem=prosumerProblem))
        best = prosumerOptimiser.optimise(pop_size=20, termination=5, verbose= False)
        #print(f"length best: {best}")

        #compute price
        s = 0
        for i in range(len(best)):
            
            if x[i] >= 0:
                # if transation is positive, apply grid price 
                s = s + x[i] * self.sellPrices[i]
            else:
                # if transation is nagtive, apply Feed in tariff 
                s = s + x[i] * self.buyPrices[i]

        return (s)

        #return calcCost(best, gridPrices = self.buyPrices, FIT = self.sellPrices)

    def _evaluate(self, X, out, *args, **kwargs):
        """
        :param x: all solutions because elementwise_evaluation = False
        :param out:
        :param args:
        :param kwargs:
        :return:
        """
        
        g1 = np.sum(X) - (np.sum(self.loadForecast) * np.ones(len(X))) -(1e-3)
        # generate the constraints for the optimization 
        out["G"] = np.column_stack([g1])

        #for every solution call calcobj using parallel processes
        #l = len(X) #number of solution vectors
        p = Pool(8) #one process for every solution
        #func = partial()
        f1 = p.map(self.calcost, X) #list of costs for every solution
        # F = [calcostBP(bidProblem= self, x= xi) for xi in x]

        # generate the objectives for the optimization 
        out["F"] =  np.column_stack([f1])


