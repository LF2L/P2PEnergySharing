from abc import ABC, abstractmethod
from copy import deepcopy
#from DReventDispatch import *
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------
def SDarray(prosumers):
    DSarray = []
    for prosumer in prosumers:
        netLoad = prosumer._get_loadForecast()
        REgen = prosumer._get_REgeneration()
        SD = np.zeros(len(netLoad))
        battery = deepcopy(prosumer.battery)
        for j in range(len(netLoad)):
            resultPower = netLoad[j] - REgen[j]
            if resultPower > 0:  # if REgeneration wasn't able to cover net load
                SD[j] = battery.discharge(resultPower)  # remains zero if the battery can handle it, otherwise it
                # will be equal to the remaining energy that couldn't be sourced
            if resultPower < 0:  # if REgeneration was enough and there is surplus
                SD[j] = -battery.charge(resultPower)  # negative value
        DSarray.append(SD)
    return DSarray

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

#
class PricingScheme(ABC): #equivalent to java interface
        def __init__(self, FIT,  gridSellPrices, listProsumers):
            self.prosumers = listProsumers
            self.gridSellPrices = gridSellPrices
            self.gridBuyPrices = FIT
            if type(self) == PricingScheme:
                raise TypeError(
                    " PricingScheme is an interface, it cannot be instantiated. Try with a concrete scheme, e.g.: SDR")

        @abstractmethod
        def generatePrices(self, **kwargs):
            pass
        @abstractmethod
        def applyPrices(self,**kwargs):
            pass

class BillSharing(PricingScheme):
    
    def __init__(self, FIT,  gridSellPrices, listProsumers, **kwargs):
        super().__init__(FIT,  gridSellPrices, listProsumers)
        #self.x = kwargs["x"] #solution optimale
        #pass
        #self.x = None
        #self.gridSellPrices = kwargs["gridPrices"]  # temporal 1D array
        #self.gridBuyPrices = kwargs["FIT"]
        # temporal 1D array
        #self.prosumers = None  # list<Prosumer>

    def applyPrices(self, x, gridPrices, FIT, listProsumers):
        DSarray = np.array(SDarray(listProsumers))
        totalSD = [sum(DSarray[:,i]) for i in range(len(DSarray[0]))]
        SDweights = [] #contains weights of every prosumer for every time slot
        for i in range(len(DSarray)): #for every prosumer
            pros_i_weights= []
            for j in range(len(totalSD)): #for every time slot
                pros_i_weights.append(DSarray[i][j]/totalSD[j])
            SDweights.append(pros_i_weights)
        #SDweights=np.array(SDweights)
        totalCost = calcCost(x, gridPrices=gridPrices, FIT=FIT)
        gridPriceWeights = [gridPrices[i] / sum(gridPrices) for i in range(len(gridPrices))]
        #gridPriceWeights = np.array(gridPriceWeights)
        costWeights = []
        for p in range(len(DSarray)): #for each prosumer
            #prosCostlist = [SDweights[prosumer][k]*gridPriceWeights[k] for k in range(len(totalSD))]
            prosCostlist = []
            for k in range(len(totalSD)): #for each time slot:
                prosCostlist.append(SDweights[p][k]*gridPriceWeights[k])
            pros_cost_weight = sum(prosCostlist) #k is time slot
            costWeights.append(pros_cost_weight)

        d = dict()
        for i in range(len(DSarray)): #for each prosumer
            d[listProsumers[i]._get_ID()] = costWeights[i]*totalCost

        return d

class SDR(PricingScheme):
    def __init__(self, FIT,  gridSellPrices, listProsumers, **kwargs):
        super().__init__(FIT,  gridSellPrices, listProsumers)
        
    def SDarray(self, prosumers):
        return SDarray(prosumers)

    def generatePrices(self, **kwargs):

        # gridSellPrice = kwargs["gridPrices"]  # temporal 1D array
        # gridBuyPrice = kwargs["FIT"]  # temporal 1D array
        # prosumers = kwargs["listProsumers"]  # list<Prosumer>
        Psell=[]
        Pbuy = []
        tempSDarray = self.SDarray(self.prosumers)
        l = len(tempSDarray)
        for j in range(len(tempSDarray[0])): #for each timeslot
            tempDarray_h = [tempSDarray[h][j] if tempSDarray[h][j]>0 else 0 for h in range(len(tempSDarray))]
            tempSarray_h = [tempSDarray[h][j] if tempSDarray[h][j]<0 else 0 for h in range(len(tempSDarray))]
            TDP_h = sum(tempDarray_h)   # total demand  power for time slot h
            TSP_h = - sum(tempSarray_h) # total supply power for time slot h made positive
            SDR_h = TSP_h/TDP_h if TDP_h!= 0 else np.infty
            Psell_h = self.gridSellPrices[j] if SDR_h>1 else  (self.gridSellPrices[j] * self.gridBuyPrices[j])/((self.gridBuyPrices[j] - self.gridSellPrices[j])*SDR_h +self.gridSellPrices[j])
            Pbuy_h  = self.gridSellPrices[j] if SDR_h>1 else  Psell_h*SDR_h + self.gridBuyPrices[j]*(1 - SDR_h)
            Psell.append(Psell_h)
            Pbuy.append(Pbuy_h)

        return Psell, Pbuy, tempSDarray

    def applyPrices(self, **kwargs):
        SDarray = self.SDarray(prosumers=self.prosumers)
        Psell, Pbuy,_ = self.generatePrices(gridPrices=self.gridSellPrices, FIT=self.gridBuyPrices, listProsumers=self.prosumers)
        assert (len(SDarray) == len(self.prosumers))
        """
        :param SDarray: list of (lists <-temp list of supply and demand of prosumer i taking into consideration their REgeneration, net loads and battery energy levels at each timeslot)
        :return: dictionary of prosumer ID : total price for the day (price to pay or to recieve)
        """
        resultDic = dict()
        for i in range(len(SDarray)):  # for every prosumer
            totalprice = 0
            for j in range(len(SDarray[i])):  # for every time-slot
                if SDarray[i][j] > 0:
                    totalprice = totalprice + SDarray[i][j] * Psell[j]
                if SDarray[i][j] < 0:
                    totalprice = totalprice + SDarray[i][j] * Pbuy[j]
            resultDic[self.prosumers[i]._get_ID()] = totalprice

        return resultDic


# class PriceCalculator:
#     def __init__(self, pricingScheme: PricingScheme):
#         self._pricingScheme = pricingScheme
#         #self.SDarray = None
#         self.Pbuy = None
#         self.Psell = None
#     def generatePrices(self):
#         # todo: find common manipulations for all possible schemes (multi-agent bidding included)
#         Psell, Pbuy, SDarray = self._pricingScheme.generatePrices()
#         self.Psell = Psell
#         self.Pbuy = Pbuy
#         self.SDarray = SDarray
#         return Psell, Pbuy, SDarray
#     def applyPrices(self, x):
#         #assert(len(self.SDarray)==len(listProsumers))
#         """
#         :param SDarray: list of (lists <-temp list of supply and demand of prosumer i taking into consideration their REgeneration, net loads and battery energy levels at each timeslot)
#         :return: dictionary of prosumer ID : total price for the day (price to pay or to recieve)
#         """

#         return self._pricingScheme.applyPrices(x=x, gridPrices= gridPrices, FIT=FIT, listProsumers=listProsumers)






