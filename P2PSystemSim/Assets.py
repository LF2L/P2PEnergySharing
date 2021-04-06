from abc import ABC, abstractmethod
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

class Battery:
    def __init__(self, nominalCapacity, SOCmin, SOCmax, selfDischarge, chargeEfficiency, dischargeEfficiency, initialEnergy=None, timeSlot_duraction = 86400/48):
        if initialEnergy is not None:
            assert (SOCmin * nominalCapacity <= initialEnergy <= SOCmax * nominalCapacity)
        self._nominalCapacity = nominalCapacity
        self._SOCmin = SOCmin # unit: ratio 0<=X<=1
        self._SOCmax = SOCmax # unit: ratio 0<=X<=1
        self._selfDischarge = selfDischarge # unit: ratio 0<=X<=1
        self._chargeEfficiency = chargeEfficiency # unit: ratio 0<=X<=1
        self._dischargeEfficiency = dischargeEfficiency # unit: ratio 0<=X<=1
        #initialising energy level
        self._energyLevel = initialEnergy if initialEnergy is not None else self._SOCmin*self._nominalCapacity
        #initialising owner
        self._owner = None
        self._timeSlot_duration_in_sec = timeSlot_duraction

        self._actualSOC = initialEnergy/self._nominalCapacity

        self._SOChistoric = [] # percentage
        self._SOChistoric.append(self._actualSOC)

        self._energLevelHistoric= []
        self._energLevelHistoric.append(self._energyLevel)
        

    #normal functions
    def charge(self, power_charging):
        """
        :param power_charging: positive value
        :return: positive value = result excess power that couldn't enter battery (or zero)
        """
        timeSlot_duration = self._timeSlot_duration_in_sec
        W_before = self._energyLevel
        sigma = self._selfDischarge
        muC_i = self._chargeEfficiency
        Pc_max = self.maximumPc()
        if power_charging < Pc_max:
            self._energyLevel = (W_before * (1 + sigma) + power_charging * muC_i * timeSlot_duration)
            return 0
        else:
            self._energyLevel = (W_before * (1 + sigma) + Pc_max * muC_i * timeSlot_duration)
            return power_charging - Pc_max


    def discharge(self, pbd):
        """
        :param pbd: positive value
        :return: positive value = result needed power that couldn't be drawn from the battery (or zero)
        """
        timeSlot_duration = self._timeSlot_duration_in_sec
        W_before = self._energyLevel
        sigma = self._selfDischarge
        muD_i = self._dischargeEfficiency
        Pd_max = self.maximumPd()
        if pbd < Pd_max:
            self._energyLevel = (W_before*(1-sigma) - pbd*(1/muD_i)*timeSlot_duration)
            return 0
        else:
            self._energyLevel = (W_before * (1 - sigma) - Pd_max * (1 / muD_i) * timeSlot_duration)
            return pbd - Pd_max

    def balance(self, power_to_battery):
        print(f"balance : {power_to_battery}")
        timeSlot_duration = self._timeSlot_duration_in_sec
        W_before = self._energyLevel
        sigma = self._selfDischarge
        muD_i = self._dischargeEfficiency
        muC_i = self._chargeEfficiency
        Pc_max = self.maximumPc()
        Pd_max = self.maximumPd()

        if power_to_battery > 0 and power_to_battery < Pc_max:
            self._energyLevel = (W_before * (1 + sigma) + power_to_battery * muC_i * timeSlot_duration)
            self._SOChistoric.append(self._energyLevel/self._nominalCapacity)
            self._energLevelHistoric.append(self._energyLevel)
            return 0
        elif power_to_battery > 0 :
            self._energyLevel = (W_before * (1 + sigma) + Pc_max * muC_i * timeSlot_duration)
            self._SOChistoric.append(self._energyLevel/self._nominalCapacity)
            self._energLevelHistoric.append(self._energyLevel)
            return power_to_battery - Pc_max

        elif power_to_battery < 0 and 0 < Pd_max + power_to_battery: # in case power in negative = discharging battery
            self._energyLevel = (W_before*(1-sigma) - power_to_battery* (1/muD_i)*timeSlot_duration)
            self._SOChistoric.append(self._energyLevel/self._nominalCapacity)
            self._energLevelHistoric.append(self._energyLevel)
            return 0
        elif power_to_battery < 0:
            self._energyLevel = (W_before * (1 - sigma) - Pd_max * (1 / muD_i) * timeSlot_duration)
            self._SOChistoric.append(self._energyLevel/self._nominalCapacity)
            self._energLevelHistoric.append(self._energyLevel)
            return Pd_max - power_to_battery
        else:
            self._energyLevel=0
            self._SOChistoric.append(self._energyLevel/self._nominalCapacity)
            self._energLevelHistoric.append(self._energyLevel)
            return 0


    def maximumPc(self):
        return (self._SOCmax * self._nominalCapacity - self._energyLevel * (1 - self._selfDischarge)) / (self._chargeEfficiency * self._timeSlot_duration_in_sec)

    def maximumPd(self):
        # return (self._dischargeEfficiency/self._timeSlot)*(self._energyLevel*(1 - self._selfDischarge) - self._SOCmin*self._nominalCapacity)
        return (self._energyLevel*(1 - self._selfDischarge) - self._SOCmin*self._nominalCapacity) / (self._dischargeEfficiency * self._timeSlot_duration_in_sec)


    def displaySOCGraph(self):
        fig, ax = plt.subplots()
        ax.hlines(y=self._SOCmin, label="SOC Min")
        ax.hlines(y=self._SOCmax, label="SOC Max")
        ax.plot(range(0,len(self._SOChistoric)), self._SOChistoric, label="SOC")
        #ax.plot(len(self._REgeneration), self._REgeneration, len(self._loadForecast), self._loadForecast)
        ax.set(xlabel='timeslots', ylabel='State of Charge (%)', title='Battery State of charge evolution from prosumer {}'.format(self._owner._ID))
        ax.grid()
        #ax.legend(loc='upper left', borderaxespad=0.)
        plt.show()

    def getSOCGraph(self):
        fig, ax = plt.subplots()
        ax.hlines(y=self._SOCmin, xmin = 0 , xmax = len(self._SOChistoric), label="SOC Min")
        ax.hlines(y=self._SOCmax, xmin = 0 , xmax = len(self._SOChistoric), label="SOC Max")
        ax.plot(range(0,len(self._SOChistoric)), self._SOChistoric, label="SOC")
        #ax.plot(len(self._REgeneration), self._REgeneration, len(self._loadForecast), self._loadForecast)
        ax.set(xlabel='timeslots', ylabel='State of Charge (%)', title='Battery State of charge evolution from prosumer {}'.format(self._owner._ID))
        ax.grid()
        #ax.legend(loc='upper left', borderaxespad=0.)
        return ax

    def displayPowerGraph(self):
        fig, ax = plt.subplots()
        ax.hlines(y=self._SOCmin* self._nominalCapacity, label="SOC Min")
        ax.hlines(y=self._SOCmax * self._nominalCapacity, label="SOC Max")
        ax.plot(range(0,len(self._SOChistoric)), self._SOChistoric, label="SOC")
        #ax.plot(len(self._REgeneration), self._REgeneration, len(self._loadForecast), self._loadForecast)
        ax.set(xlabel='timeslots', ylabel='Power (Wh)', title='Power in the battery of the prosumer {}'.format(self._owner._ID))
        ax.grid()
        #ax.legend(loc='upper left', borderaxespad=0.)
        plt.show()

#iteration/collection design pattern
class Operator(ABC):
    def __init__(self, *args, **kwargs):
        if type(self) == Operator:
            raise TypeError(
                " Operator is an interface, it cannot be instantiated. Try with a concrete operator, e.g.: AggregationIterator")
    @abstractmethod
    def charge(self, power_charging):
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

    @abstractmethod
    def loadProcessing(self, power_to_battery):
        pass

class AggregationOperator(Operator):
    def __init__(self,listBattery):
        self._listBattery = listBattery
    def maximumPc(self):
        return sum([battery.maximumPc() for battery in self._listBattery])
    def maximumPd(self):
        return sum([battery.maximumPd() for battery in self._listBattery])
    def charge(self, power_charging):
        """
        :param power_charging: temporal list of charging power
        :return: temp list of constraint function values
        """
        for b in self._listBattery:
            b._timeSlot = 86400/len(power_charging)
        G = np.zeros(len(power_charging))

        for i in range(len(power_charging)): #for each time slot
            if power_charging[i] != 0:
                totalMaximumPc = self.maximumPc()
                gi = power_charging[i] - totalMaximumPc
                G[i] = gi
                if gi > 0:
                    # if constraint is broken one time there is no need to continue for other time slots since we don't know how the batteries are going to be charged
                    # we just replicate the last constraint value for the next time slots
                    for j in range(i+1, len(power_charging)):
                        G[i] = gi
                    break
                else:
                    #if the constraint isn't broken we charge batteries with power_charging[i] and continue for the next time slot

                    for battery in self._listBattery:
                        power_charging[i] = battery.charge(power_charging[i]) #charge until reaching the max and return what couldn't be charged so it will be passed to next battery
        return G


    def discharge(self, Pbd):
        """
        :param Pbd: temporal list of discharging power
        :return: temp list of constraint function values
        """
        for b in self._listBattery:
            b._timeSlot = 86400/len(Pbd)
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

    def loadProcessing(self, power_to_battery):
        """
        :param power_to_battery: it is the energy requested (if negative value) or provided (if positive value)
        :return: return exceeded power
        """
        # print(f"data processing: {power_to_battery}")
        G = np.zeros(len(power_to_battery))
        for i in range(len(power_to_battery)): #for each time slot
            if power_to_battery[i] > 0:
                totalMaximumPc = self.maximumPc()
                gi = power_to_battery[i] - totalMaximumPc
                # G[i] = gi
                if gi > 0:
                    # if constraint is broken one time there is no need to continue for other time slots since we don't know how the batteries are going to be charged
                    # we just replicate the last constraint value for the next time slots
                    # for j in range(i+1, len(power_charging)):
                    #     G[i] = gi
                    # break
                    G[i] = gi # positive 
                else:
                    #if the constraint isn't broken we charge batteries with power_charging[i] and continue for the next time slot

                    for battery in self._listBattery:
                        power_to_battery[i] = battery.balance(power_to_battery[i])
                    G[i] = 0 

            elif power_to_battery[i]<0:
                totalMaximumPd = self.maximumPd()
                gi = totalMaximumPd + power_to_battery[i] # power_to_battery[i] is negative
                # G[i] = gi
                if gi < 0:
                    # if constraint is broken no need to continue for the next time slots
                    # for j in range(i + 1, len(Pbd)):
                    #     G[i] = gi
                    # break
                    G[i] = gi
                else:

                    for battery in self._listBattery:
                        power_to_battery[i] = battery.balance(power_to_battery[i])  # discharge until reaching the max and return what couldn't be discharged so it will be passed to next battery
                        #at one point pbd will be = zero
                    G[i] = 0 
            else: 
                G[i] = 0
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
