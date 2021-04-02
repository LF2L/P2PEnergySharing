from P2PSystemSim.OptimisationProblem import *
from P2PSystemSim.ConvergenceControler import *
from P2PSystemSim.PricingSystem import *

class ProsumerAgent():
    def __init__(self, prosumer,  **kwargs):
        self._prosumer = prosumer
        #self._shiftableLoadMatrix = shiftableLoadMatrix
        pass

    # def _get_prosumer(self):
    #     return self._prosumer

    # def _get_shiftableLoadMatrix(self):
    #     return self._shiftableLoadMatrix

    def generateBid(self, buyPrices, sellPrices, **kwargs) -> list:
        """
        :param kwargs: 
        sellPrices = temp list of selling prices for each time slot 
        buyPrices = buying prices for every time slot
        :return: temp list of the bid = demand/offer of energy
        """
        #prosumerAgent = ProsumerAgent(prosumer=self._prosumer, shiftableLoadMatrix=self._prosumer._shiftableLoadMatrix)
        prob = BiddingProblem(prosumerAgent=self, loadForecast = self._prosumer._loadForecast, REgenerationForecast= self._prosumer._REgeneration, sellPrices=sellPrices, buyPrices=buyPrices)
        bidOptimiser = Optimiser(G_A(optimisationProblem=prob))
        optimisdBids = bidOptimiser.optimise()
        #print(optimisdBids.pop.get())
        #res = optimiser.optimise()
        self._prosumer._loadForecast= optimisdBids
        self._prosumer.loadHistoric.insert(0, optimisdBids)
        return optimisdBids

class CoordinatorAgent:
    def __init__(self, gridPrices, FIT, prosumerAgents, **kwargs):
        self.CAsellPrices = gridPrices #will vary
        self.CAbuyPrices = FIT #will vary
        self.gridPrices = gridPrices  #stays the same
        self.FIT = FIT #stays the same
        self.prosumerAgents = prosumerAgents

        self.sellPricesHistoric = []
        self.sellPricesHistoric.append(FIT)
        self.buyPricesHistoric = []
        self.buyPricesHistoric.append(gridPrices)

    def computePrices(self, pricingScheme: PricingScheme, bids= None):
        """
        :param bids: list of (lists<- bids of prosumers = temp list of energy offer/demand)
        :param pricingScheme:  object of class Pricing Scheme
        :return: two temp lists: buy and sell prices in the energy pool
        """
        prosumers = [PA._prosumer for PA in self.prosumerAgents]
        pricecalc = PriceCalculator(pricingScheme=pricingScheme)
        Psell, Pbuy, _= pricecalc.generatePrices(gridBuyPrice=self.CAsellPrices, gridSellPrice= self.CAbuyPrices, listProsumers=prosumers)
        return Psell, Pbuy

    def getBids(self):
        bids = []
        for prosumAgent in self.prosumerAgents:
            print(f"prosumer: {prosumAgent._prosumer._ID}")
            bid = prosumAgent.generateBid(buyPrices = self.CAsellPrices, sellPrices = self.CAbuyPrices)
            bids.append(bid)
        return(bids)

    def run(self, convergenceModel: ConvergenceControlModel, pricingScheme:PricingScheme):
        controler = ConvergenceControler(convergenceModel)
        while controler.control():
            print(f"iteration: {controler.controlModel.nbIterations}")
            self.bids = self.getBids()
            self.CAsellPrices, self.CAbuyPrices = self.computePrices(pricingScheme=pricingScheme, bids=self.bids)
        for PA in self.prosumerAgents:
            print(PA._prosumer._loadForecast)

    def displayProsumers(self):
        fig, axs = plt.subplots(1, len(self.prosumerAgents))
        fig.suptitle('Production and consumption of each prosumer')
        for i, prosumerAgent in enumerate(self.prosumerAgents): 
            axs[i].plot(range(0,len(prosumerAgent._prosumer._REgeneration)), prosumerAgent._prosumer._REgeneration, label="RE generation")
            for j, load in enumerate(prosumerAgent._prosumer.loadHistoric):
                if j == 0:
                    label= "Last load optimisation"
                elif j == (len(prosumerAgent._prosumer.loadHistoric)-1):
                    label = "Initial load"
                else:
                    label = f"Load optimisation {len(prosumerAgent._prosumer.loadHistoric) - j}"
                axs[i].plot(range(0,len(load)), load, label=label)
            axs[i].set(xlabel='timeslots', ylabel='Power (Wh)', title='Prosumer {}'.format(prosumerAgent._prosumer._ID))
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

    def displayPricingEvolution(self):
        fig, axs = plt.subplots()
        fig.suptitle('Pricing evolution')
        for i, sellPrices in enumerate(self.sellPricesHistoric): 
            if i == 0:
                label= "Last sell price optimisation"
            elif i == (len(self.sellPricesHistoric)-1):
                label = "Initial sell price"
            else:
                label = f"Sell price optimisation {len(self.sellPricesHistoric) - i}"
            axs.plot(range(0,len(sellPrices)), sellPrices, label=label)
            #axs[i].set(xlabel='timeslots', ylabel='Power (Wh)', title='Prosumer {}'.format(prosumerAgent._prosumer._ID))
            # axs[i].legend()
        for j, buyPrices in enumerate(self.buyPricesHistoric):
            if j == 0:
                label= "Last buy prices optimisation"
            elif j == (len(self.buyPricesHistoric)-1):
                label = "Initial buy prices"
            else:
                label = f"Buy prices optimisation {len(self.buyPricesHistoric) - i}"
            axs.plot(range(0,len(buyPrices)), buyPrices, label=label)
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines,labels, loc='upper right')
        plt.show()