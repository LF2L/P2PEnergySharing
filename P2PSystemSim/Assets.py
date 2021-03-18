from abc import ABC, abstractmethod
import numpy as np
import sys
import pandas as pd

class Battery:
    def __init__(self, nominalCapacity, SOCmin, SOCmax, selfDischarge, chargeEfficiency, dischargeEfficiency, initialEnergy=None, timeSlot = 86400/48):
        if initialEnergy is not None:
            assert (SOCmin * nominalCapacity <= initialEnergy <= SOCmax * nominalCapacity)
        self._nominalCapacity = nominalCapacity
        self._SOCmin = SOCmin
        self._SOCmax = SOCmax
        self._selfDischarge = selfDischarge
        self._chargeEfficiency = chargeEfficiency
        self._dischargeEfficiency = dischargeEfficiency
        #initialising energy level
        self._energyLevel = initialEnergy if initialEnergy is not None else self._SOCmin*self._nominalCapacity
        #initialising owner
        self._owner = None
        self._timeSlot = timeSlot

    #getters
    def _get_nominalCapacity(self):
        return self._nominalCapacity
    def _get_SOCmin(self):
        return self._SOCmin
    def _get_SOCmax(self):
        return self._get_SOCmax()
    def _get_selfDischarge(self):
        return self._selfDischarge
    def _get_chargeEfficiency(self):
        return self._chargeEfficiency
    def _get_dischargeEfficiency(self):
        return self._dischargeEfficiency
    def _get_energyLevel(self):
        return self._energyLevel
    def _get_owner(self):
        return self._owner
    def _get_timeSlot(self):
        return self._timeSlot
    def _set_owner(self, prosumer):
        self._owner = prosumer
    def _set_energyLevel(self, energy):
        self._energyLevel = energy
    def _set_timeSlot(self, timeSlot):
        self._timeSlot = timeSlot

    #normal functions
    def charge(self, pbc):
        """
        :param pbc: positive value
        :return: positive value = result excess power that couldn't enter battery (or zero)
        """
        timeSlot = self._timeSlot
        W_before = self._energyLevel
        sigma = self._selfDischarge
        muC_i = self._chargeEfficiency
        Pc_max = self.maximumPc()
        if pbc < Pc_max:
            self._energyLevel = (W_before * (1 + sigma) + pbc * muC_i * timeSlot)
            return 0
        else:
            self._energyLevel = (W_before * (1 + sigma) + Pc_max * muC_i * timeSlot)
            return pbc - Pc_max


    def discharge(self, pbd):
        """
        :param pbd: positive value
        :return: positive value = result needed power that couldn't be drawn from the battery (or zero)
        """
        timeSlot = self._timeSlot
        W_before = self._energyLevel
        sigma = self._selfDischarge
        muD_i = self._dischargeEfficiency
        Pd_max = self.maximumPd()
        if pbd < Pd_max:
            self._set_energyLevel(W_before*(1-sigma) - pbd*(1/muD_i)*timeSlot)
            return 0
        else:
            self._set_energyLevel(W_before * (1 - sigma) - Pd_max * (1 / muD_i) * timeSlot)
            return pbd - Pd_max

    def maximumPc(self):
        return (self._SOCmax * self._nominalCapacity- self._energyLevel * (1 - self._selfDischarge)) / (self._chargeEfficiency * self._timeSlot)

    def maximumPd(self):
        return (self._dischargeEfficiency/self._timeSlot)*(self._energyLevel*(1 - self._selfDischarge) - self._SOCmin*self._nominalCapacity)



#iteration/collection design pattern
class Operator(ABC):
    def __init__(self, *args, **kwargs):
        if type(self) == Operator:
            raise TypeError(
                " Operator is an interface, it cannot be instantiated. Try with a concrete operator, e.g.: AggregationIterator")
    @abstractmethod
    def charge(self, Pbc):
        pass
    @abstractmethod
    def discharge(self, Pbd):
        pass
    @abstractmethod
    def maximumPc(self):
        pass
    @abstractmethod
    def maximumPd(self):
        pass

class AggregationOperator(Operator):
    def __init__(self,listBattery):
        self._listBattery = listBattery
    def maximumPc(self):
        return sum([battery.maximumPc() for battery in self._listBattery])
    def maximumPd(self):
        return sum([battery.maximumPd() for battery in self._listBattery])
    def charge(self, Pbc):
        """
        :param Pbc: temporal list of charging power
        :return: temp list of constraint function values
        """
        for b in self._listBattery:
            b._set_timeSlot(86400/len(Pbc))
        G = np.zeros(len(Pbc))

        for i in range(len(Pbc)): #for each time slot
            if Pbc[i] != 0:
                totalMaximumPc = self.maximumPc()
                gi = Pbc[i] - totalMaximumPc
                G[i] = gi
                if gi > 0:
                    # if constraint is broken one time there is no need to continue for other time slots since we don't know how the batteries are going to be charged
                    # we just replicate the last constraint value for the next time slots
                    for j in range(i+1, len(Pbc)):
                        G[i] = gi
                    break
                else:
                    #if the constraint isn't broken we charge batteries with Pbc[i] and continue for the next time slot

                    for battery in self._listBattery:
                        Pbc[i] = battery.charge(Pbc[i]) #charge until reaching the max and return what couldn't be charged so it will be passed to next battery
        return G


    def discharge(self, Pbd):
        """
        :param Pbd: temporal list of discharging power
        :return: temp list of constraint function values
        """
        for b in self._listBattery:
            b._set_timeSlot(86400/len(Pbd))
        G = np.zeros(len(Pbd))
        for i in range(len(Pbd)):  # for each time slot
            if Pbd[i]!= 0:
                totalMaximumPd = self.maximumPd()
                gi = Pbd[i] - totalMaximumPd
                G[i] = gi
                if gi > 0:
                    # if constraint is broken no need to continue for the next time slots
                    for j in range(i + 1, len(Pbd)):
                        G[i] = gi
                    break
                else:

                    for battery in self._listBattery:
                        Pbd[i] = battery.discharge(Pbd[i])  # discharge until reaching the max and return what couldn't be discharged so it will be passed to next battery
                        #at one point pbd will be = zero
        return G

class BatteryCollection(ABC):
    def __init__(self, *args, **kwargs):
        if type(self) == BatteryCollection:
            raise TypeError(
                " BatteryCollection is an interface, it cannot be instantiated. Try with a concrete collection, e.g.: BatteryAggregation")
    @abstractmethod
    def operator(self):
        pass

class BatteryAggregation(BatteryCollection):
    def __init__(self, listBattery):
        self._listBattery = listBattery
    def addItem(self, item):
        self._listBattery.append(item)
    def operator(self):
        return AggregationOperator(self._listBattery)

class PhotovoltaicPanel:
    def __init__(self, surface, efficiency):
        self._surface = surface
        self._efficiency = efficiency

    def elecProduction(self, GHIfilepath):
        df = pd.read_csv(GHIfilepath)
        GHIlist = df["GHI"].tolist()
        return [ghi*self._surface*self._efficiency for ghi in GHIlist]
