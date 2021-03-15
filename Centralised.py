import abc
from Optimiser import Optimiser
from DReventDispatch import *
from Battery import *
from Optimiser import *
from OptimisationProblem import *
from PriceCalculator import *

class Prosumer:
    def __init__(self, id, loadForecast : list, REgeneration: list, battery : Battery = None, shiftableLoadMatrix=None):
        assert(len(loadForecast) == len(REgeneration))
        self._ID = id
        self.battery = None
        if battery is not None:
            self.set_battery(battery)
        self._loadForecast = loadForecast
        self._REgeneration = REgeneration
        self._shiftableLoadMatrix = shiftableLoadMatrix
    def set_battery(self, battery: Battery):
        self.battery = battery
        battery._set_owner(self)
    def _get_ID(self):
        return self._ID
    def _get_Battery(self)-> Battery:
        return self._battery
    def _get_loadForecast(self)-> list:
        return self._loadForecast
    def _get_REgeneration(self)-> list:
        return self._REgeneration
    def _get_shiftableLoadMatrix(self) -> list:
        return self._shiftableLoadMatrix
    def actionDR(self) -> bool:
        return random.choice([0, 1])

class Coordinator(metaclass=abc.ABCMeta): # equivalent to java abstract class

    def __init__(self, **kwargs):
        self._prosumers = kwargs["prosumerList"]
        self._gridPrices = kwargs["gridPrices"]
        self._FIT =  kwargs["FIT"]
        self.algorithm = kwargs["algorithm"] #string
    def _get_loadHistory(self):
        return self._loadHistory

    def _set_loadHistory(self, loadhistory):
        self._loadHistory = loadhistory

    def _get_solarRadiance(self):
        return self._solarRadiance

    def _set_solarRadiance(self, solarradiance):
        self._solarRadiance = solarradiance

    def _get_temperatures(self):
        return self._temperatures

    def _set_temperatures(self, temp):
        self._temperatures = temp

    def _get_feedInTariffs(self):
        return self._feedInTariffs

    def _set_feedInTariffs(self, feedintariffs):
        self._feedInTariffs = feedintariffs

    def _get_gridPrices(self):
        return self._gridPrices

    def _set_gridPrices(self, gp):
        self._gridPrices = gp

    def forecastLoads(self):
        #todo: complete with forecast
        return None

    def computeREgenerationForecast(self):
        return self._loadHistory
        #todo : change to real forecast formula

    @abc.abstractmethod
    def optimise(self, algorithm):
        pass

    @abc.abstractmethod
    def generateDRrequest(self):
        pass

    @abc.abstractmethod
    def sendDRrequest(self):
        pass

    @abc.abstractmethod
    def computeCommunityPricing(self, priceSceheme):
        pass
    @abc.abstractmethod
    def run(self,loadForecast, REforecast, gridPrices, FIT, batteryList):
        pass

class DRcoordinator(Coordinator):
    def optimise(self, algorithm):
        return None

    def generateDRrequest(self):
        return None

    def sendDRrequest(self) -> list:
        return list()

    def computeCommunityPricing(self, priceScheme):
        return None
    def run(self,loadForecast, REforecast, gridPrices, FIT, batteryList ):
        return None

class RegularCoordinator(Coordinator):
    def generateDRrequest(self):
        """ this class doesn't redefine method generateDRrequest because it isn't its responsibility"""
        pass

    def sendDRrequest(self):
        """ this class doesn't redefine method sendDRrequest because it isn't its responsibility"""
        pass

    def optimise(self, algorithm):
        # instantiate Optimiser with the OptimisationAlgorithm
        optimiser = Optimiser(algorithm=algorithm)
        return optimiser.optimise()


    def computeCommunityPricing(self, priceScheme):
        priceCalc = PriceCalculator(priceScheme)

        return priceCalc.generatePrices()
    def run(self):
        nbtimeslots = len(self._prosumers[0]._get_loadForecast())
        totalLoadForecast = np.zeros(nbtimeslots)
        totalREforecast = np.zeros(nbtimeslots)
        for prosumer in self._prosumers:
            totalLoadForecast = np.add(totalLoadForecast, prosumer._get_loadForecast())
            totalREforecast = np.add(totalREforecast, prosumer._get_REgeneration())

        # create batteryAggregation
        batteryList = []
        for prosumer in self._prosumers:
            batteryList.append(prosumer.battery)
        batteryAggregation = BatteryAggregation(batteryList)
        # create CommonProblem
        optProblem = CommonProblem(loadForecast = totalLoadForecast, REgenerationForecast=totalREforecast, batteryList=batteryList, FIT=self._FIT, gridPrices = self._gridPrices )
        # instantiate OptimisationAlgorithm
        if self.algorithm=="NSGAII":
            algorithm = NSGAII(optProblem)
        if self.algorithm=="GA":
            algorithm = G_A(optProblem)
        # optimise -> optimal solution
        res = self.optimise(algorithm)
        #instantiate pricing scheme
        pricingscheme = SDR(gridSellPrices=self._gridPrices, FIT=self._FIT ,listProsumers=self._prosumers )
        # instantiate price calculator
        #calculator = PriceCalculator(pricingscheme)
        # compute costs for every prosumer
        pricingscheme.generatePrices()

        resultPriceDic = pricingscheme.applyPrices()

        return res, resultPriceDic



