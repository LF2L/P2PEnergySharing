from P2PSystemSim.OptimisationProblem import *
from ConvergenceControler import *
from P2PSystemSim.PricingSystem import *

class ProsumerAgent():
    def __init__(self, prosumer,  **kwargs):
        self._prosumer = prosumer
        #self._shiftableLoadMatrix = shiftableLoadMatrix
        #todo: continue
        pass

    def _get_prosumer(self):
        return self._prosumer

    # def _get_shiftableLoadMatrix(self):
    #     return self._shiftableLoadMatrix

    def generateBid(self, buyPrices, sellPrices, **kwargs) -> list:
        """
        :param kwargs: sellPrices = temp list of selling prices for each time slot buyPrices = buying prices for every time slot
        :return: temp list of the bid = demand/offer of energy
        """
        #prosumerAgent = ProsumerAgent(prosumer=self._prosumer, shiftableLoadMatrix=self._prosumer._shiftableLoadMatrix)
        prob = BiddingProblem(prosumerAgent=self, loadForecast = self._prosumer._loadForecast, REgenerationForecast= self._prosumer._REgeneration, sellPrices=sellPrices, buyPrices=buyPrices)
        optimiser = Optimiser(algorithm=G_A(optimisationProblem=prob))
        res = optimiser.optimise(pop_size=20, termination=10, verbose= True)
        self._prosumer._set_loadForecast(res)
        return res

class CoordinatorAgent:
    def __init__(self, gridPrices, FIT, prosumerAgents, **kwargs):
        self.CAsellPrices = gridPrices #will vary
        self.CAbuyPrices = FIT #will vary
        self.gridPrices = gridPrices  #stays the same
        self.FIT = FIT #stays the same
        self.prosumerAgents = prosumerAgents

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
        for prosumAgent in self.prosumerAgents:
            bid = prosumAgent.generateBid(buyPrices = self.CAsellPrices, sellPrices = self.CAbuyPrices)
            bids.append(bid)
        return(bids)

    def run(self, convergenceModel: ConvergenceControlModel, pricingScheme:PricingScheme):
        controler = ConvergenceControler(convergenceModel)
        while controler.control():
            bids = self.getBids()
            self.CAsellPrices, self.CAbuyPrices = self.computePrices(pricingScheme=pricingScheme, bids=bids)
        for PA in self.prosumerAgents:
            print(PA._get_prosumer._get_loadForecast())

