from abc import ABC, abstractmethod
#from DReventDispatch import *
from P2PSystemSim.Optimiser import *
from P2PSystemSim.OptimisationProblem import *
from P2PSystemSim.PricingSystem import *


class Coordinator(ABC): 

    def __init__(self, prosumerList, gridPrices, FIT, algorithm):
        self._prosumers = prosumerList
        self._gridPrices = gridPrices
        self._FIT =  FIT
        self.algorithm = algorithm #string

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

    def calculateSelfSufficiencyIndex(self):
        assert (self.totalLoadForecast)
        return (self.totalLoadForecast - GridPowerLoad) / self.totalLoadForecast


    @abstractmethod
    def optimise(self, algorithm):
        pass

    @abstractmethod
    def generateDRrequest(self):
        pass

    @abstractmethod
    def sendDRrequest(self):
        pass

    @abstractmethod
    def computeCommunityPricing(self, priceSceheme):
        pass
    @abstractmethod
    def run(self,loadForecast, REforecast, gridPrices, FIT, batteryList):
        pass

class DRcoordinator(Coordinator):

    def __init__(self, prosumerList, gridPrices, FIT, algorithm):
        super().__init__(prosumerList, gridPrices, FIT, algorithm)
        
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

    def __init__(self, prosumerList, gridPrices, FIT, algorithm):
        super().__init__(prosumerList, gridPrices, FIT, algorithm)

    
    def generateDRrequest(self):
        """ this class doesn't redefine method generateDRrequest because it isn't its responsibility"""
        pass

    def sendDRrequest(self):
        """ this class doesn't redefine method sendDRrequest because it isn't its responsibility"""
        pass

    def optimise(self, algorithm):
        # instantiate Optimiser with the OptimisationAlgorithm
        optimiser = Optimiser(algorithm=algorithm)
        return optimiser.optimise(verbose= False)


    def computeCommunityPricing(self, priceScheme):
        priceCalc = PriceCalculator(priceScheme)
        return priceCalc.generatePrices()

    def run(self):
        nbtimeslots = len(self._prosumers[0]._get_loadForecast())
        self.totalLoadForecast = np.zeros(nbtimeslots)
        totalREforecast = np.zeros(nbtimeslots)
        for prosumer in self._prosumers:
            self.totalLoadForecast = np.add(self.totalLoadForecast, prosumer._get_loadForecast())
            totalREforecast = np.add(totalREforecast, prosumer._get_REgeneration())

        # create batteryAggregation
        batteryList = []
        for prosumer in self._prosumers:
            batteryList.append(prosumer._battery)
        batteryAggregation = BatteryAggregation(batteryList)

        # create CommonProblem
        optProblem = CommonProblem(loadForecast = self.totalLoadForecast, REgenerationForecast=totalREforecast, batteryList=batteryList, FIT=self._FIT, gridPrices = self._gridPrices )
        # instantiate OptimisationAlgorithm
        if self.algorithm=="NSGAII":
            algorithm = NSGAII(optProblem)
        if self.algorithm=="GA":
            algorithm = G_A(optProblem)
        # optimise -> optimal solution
        self.powerFromGrid= self.optimise(algorithm)
        #instantiate pricing scheme
        pricingscheme = SDR(gridSellPrices=self._gridPrices, FIT=self._FIT ,listProsumers=self._prosumers )
        # instantiate price calculator
        #calculator = PriceCalculator(pricingscheme)
        # compute costs for every prosumer
        pricingscheme.generatePrices()

        resultPriceDic = pricingscheme.applyPrices()

        return self.powerFromGrid, resultPriceDic

    def displayProsumers(self):
        fig, axs = plt.subplots(1, len(self._prosumers))
        fig.suptitle('Production and consumption of each prosumer')
        for i, prosumer in enumerate(self._prosumers): 
            axs[i].plot(range(0,len(prosumer._REgeneration)), prosumer._REgeneration, label="RE generation")
            axs[i].plot(range(0,len(prosumer._loadForecast)), prosumer._loadForecast, label="Load forecast")
            axs[i].set(xlabel='timeslots', ylabel='Power (Wh)', title='Prosumer {}'.format(prosumer._ID))
            # axs[i].legend()
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines,labels, loc='upper right')
        plt.show()

    def calculateSelfSufficiency(self):
        if hasattr(self, 'totalLoadForecast') and hasattr(self, 'powerFromGrid') :
            communityImportFromGrid = abs(np.sum(self.powerFromGrid))
            communityTotalLoad = np.sum(self.totalLoadForecast)
            selfSuficiency = (communityTotalLoad - communityImportFromGrid ) / communityTotalLoad
            return selfSuficiency
        else:
            raise "The coordinator has to run() before calculate the self sufficiency indicator"

        

