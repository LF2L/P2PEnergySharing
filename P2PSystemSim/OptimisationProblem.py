
#from Coordinator import Prosumer
#from DReventDispatch import *
from P2PSystemSim.Assets import *
from P2PSystemSim.Optimiser import *
#from utils import *
from multiprocessing import Pool
from copy import deepcopy

def calcCost(x, **kwargs):
    """
    :param x: one solution vector
    :param kwargs1:
    :return: real: total cost induced by this solution vector
    """
    # print(x)
    s = 0
    for i in range(len(x)):

        if x[i] >= 0:
            s = s + x[i] * kwargs["gridPrices"][i]
        else:
            s = s + x[i] * kwargs["FIT"][i]

    return (s)



class CommonProblem(Problem): # extends the problem object from Pymoo

    def __init__(self,loadForecast, REgenerationForecast, **kwargs):
        assert(len(loadForecast) == len(REgenerationForecast))
        self.BatteryList = deepcopy(kwargs["batteryList"])
        totalNominalCapacity = sum([b._get_nominalCapacity() for b in self.BatteryList])
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
            xu[i] = self.loadForecast[i] - self.REgenerationForecast[i] + totalNominalCapacity/timeSlot # the upper bound
            # (positive value) represents the total needed load minus the RE generation (which is equivalent to the energy
            # demand that wasn't met by RE generation) plus the battery size. In other words: we cover energy demand
            # that wasn't met through RE generation and charge the battery to its maximum capacity admitting it was fully uncharged
            xl[i] = - (totalNominalCapacity + self.REgenerationForecast[i])
            # the maximum i can sell is the maximum i can have available,
            # which is the maximum I can have in storage plus all i can produce
        super().__init__(n_var=len(loadForecast), n_obj=1, n_constr=len(loadForecast), xl=xl, xu=xu, elementwise_evaluation=False)


    def _evaluate(self, x, out, *args, **kwargs):
        """

        :param x: all solutions because elemenwise_evaluation = False
        :param out:
        :param args:
        :param kwargs:
        :return:
        """

        f = [calcCost(xi, gridPrices=self.Parameters["gridPrices"], FIT = self.Parameters["FIT"]) for xi in x]
        out["F"] =  np.array(f)

        def calcConstr(x, **kwargs2):
            """
            this function computes the constraint functions values (144 = 3*48 constraints)
            :param x: one solution vector, i.e. transactions with the grid for each time-slot of the day
            :param kwargs2: needed parameters like the REgenerationForecast vector, the loadForecast and others
            :return: constraint function value = temp array of how much pbd was not met /how much pbc couldn't fit for each time slot
            """

            # calculate vector that only records grid energy loads (feed-in values, i.e. negative values, are turned into zeros)
            X = [x[i] if x[i]>=0 else 0 for i in range (len(x))]
            # calculate vector that only records feed-in energy in positive form (grid loads, i.e. positive values, are turned into zeros)
            Y = [-x[i] if x[i]<=0 else 0 for i in range(len(x))]

            # Charge and discharge power calculation
            # in this model we cannot charge and discharge in the same time slot bc Y[i] isn't systematically drawn from battery
            Pbc = np.zeros(len(x))
            Pbd = np.zeros(len(x))
            for i in range(len(x)):
                surplus = X[i] + kwargs2["REgenerationForecast"][i] - kwargs2["loadForecast"][i] - Y[i]
                # surplus is internal sources (RE generation) and external (grid load = x(i)) minus destinations (feed-ins and net loads)
                if surplus>=0:
                    Pbc[i] = surplus
                else:
                    Pbd[i] = -surplus

            #create aggregation of batteries and attempt to charge/discharge with Pbc and Pbd temp arrays
            batteryagg = BatteryAggregation(kwargs2["batteryList"])
            operator = batteryagg.operator()
            G1 = operator.charge(Pbc)
            G2 = operator.discharge(Pbd)
            G = np.zeros(self.nbTimeSlots)
            for i in range(len(G)):
                G[i] = G1[i] if G1[i] != 0 else G2[i]

            return G

        G = [calcConstr(xi, REgenerationForecast = self.REgenerationForecast, loadForecast = self.loadForecast, batteryList=self.BatteryList) for xi in x]
        out["G"] = np.column_stack([G])


class BiddingProblem(Problem):
    def __init__(self, loadForecast: list, REgenerationForecast: list, prosumer):
        self.loadForecast = loadForecast
        self.REgenerationForecast = REgenerationForecast
        self.prosumer = prosumer
        self.shiftableLoadMatrix = prosumer._get_shiftableLoadMatrix()
        self.nbTimeSlots= len(loadForecast)
        xl = np.zeros(self.nbTimeSlots)
        xu = np.zeros(self.nbTimeSlots)
        for i in range(0, self.nbTimeSlots):
            xl[i] = loadForecast[i]*self.shiftableLoadMatrix[i] if self.shiftableLoadMatrix[i]<0 else loadForecast[i]  #if we can move some of the current power load to another time slot
            xu[i] = loadForecast[i]*self.shiftableLoadMatrix[i] if self.shiftableLoadMatrix[i]> 0 else loadForecast[i] #if we can add some more power load to the current time slot
        super().__init__(n_var=len(loadForecast), n_obj=1, n_constr=1, xl=xl, xu=xu, elementwise_evaluation=False)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        :param x: all solutions because elementwise_evaluation = False
        :param out:
        :param args:
        :param kwargs:
        :return:
        """
        def calcconstr(x, loadForecast):
            """
            :param x: one solution
            :param loadForecast: the prosumer's load forecast
            :return: real that must be < 0
            """
            return sum(x) -sum(loadForecast) -(1e-5) #turning equality constraint into inequality constraint
        G = [calcconstr(xi, self.loadForecast) for xi in x]
        out["G"] = G
        def calcost(x, prosumer, **kwargs):
            """
            :param x: one solution = one temp list of netLoads after shifting
            :param prosumer:
            :return:
            """

            problem = CommonProblem(loadForecast=prosumer._get_loadForecast(), REgenerationForecast=
            prosumer._get_REgeneration(), batteryList= [prosumer._get_Battery()], gridPrices = kwargs["gridPrices"], FIT = kwargs["FIT"])
            optimiser = Optimiser(G_A(commonProblem=problem))
            best = optimiser.optimise()
            #compute price
            return calcCost(best, gridPrices = kwargs["gridPrices"], FIT = kwargs["FIT"])

        #for every solution call calcobj using parallel processes
        l = len(x) #number of solution vectors
        p = Pool(l) #one process for every solution
        F = p.map(calcost, x) #list of costs for every solution
        out["F"] = F


