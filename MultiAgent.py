from OptimisationProblem import *
from ConvergenceControler import *

class ProsumerAgent():
    def __init__(self, **kwargs):
        self._prosumer = kwargs["prosumer"]
        self._shiftableLoadMatrix = kwargs["shiftableLoadMatrix"]
        #todo: continue
        pass
    def _get_prosumer(self):
        return self._prosumer
    def _get_shiftableLoadMatrix(self):
        return self._shiftableLoadMatrix
    def generateBid(self, **kwargs) -> list:
        """
        :param kwargs: sellPrices = temp list of selling prices for each time slot buyPrices = buying prices for every time slot
        :return: temp list of the bid = demand/offer of energy
        """
        buyPrices = kwargs["buyPrices"]
        sellPrices = kwargs["sellPrices"]
        prosumerAgent = ProsumerAgent(prosumer=self._prosumer, shiftableLoadMatrix=self._shiftableLoadMatrix)
        prob = BiddingProblem(prosumeragent=prosumerAgent, sellPrices=sellPrices, buyPrices=buyPrices)
        optimiser = Optimiser(algorithm=G_A(problem=prob, verbose=True))
        res = optimiser.optimise(pop_size=20, termination=10)
        self._prosumer._set_loadForecast(res)
        return res

class CoordinatorAgent:
    def __init__(self, **kwargs):
        self.CAsellPrices = kwargs["gridPrices"] #will vary
        self.CAbuyPrices = kwargs["FIT"] #will vary
        self.gridPrices =kwargs["gridPrices"]  #stays the same
        self.FIT = kwargs["FIT"] #stays the same
        self.prosumerAgents = kwargs["prosumerAgentList"]
    def computePrices(self, pricingScheme: PricingScheme, bids= None):
        """
        :param bids: list of (lists<- bids of prosumers = temp list of energy offer/demand)
        :param pricingScheme:  object of class Pricing Scheme
        :return: two temp lists: buy and sell prices in the energy pool
        """
        prosumers = [PA._get_prosumer() for PA in self.prosumerAgents]
        pricecalc = PriceCalculator(pricingScheme=pricingScheme)
        Psell, Pbuy, _= pricecalc.generatePrices(gridBuyPrice=self.CAsellPrices, gridSellPrice= self.CAbuyPrices, listProsumers=prosumers)
        return Psell, Pbuy

    def getBids(self):
        bids = []
        for PA in self.prosumerAgents:
            bid = PA.generateBid(buyPrices = self.CAsellPrices, sellPrices = self.CAbuyPrices)
            bids.append(bid)
        return(bids)
    def run(self, convergenceModel: ConvergenceControlModel, pricingScheme:PricingScheme):
        controler = ConvergenceControler(convergenceModel)
        while controler.control():
            bids = self.getBids()
            self.CAsellPrices, self.CAbuyPrices = self.computePrices(pricingScheme=pricingScheme, bids=bids)
        for PA in self.prosumerAgents:
            print(PA._get_prosumer._get_loadForecast())

