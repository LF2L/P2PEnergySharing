import numpy as np
import random
import sys

nbOfStepInOneDay = 1

FeedInTariff = 0.035 * np.ones(nbOfStepInOneDay) 
#gridPrices = [random.random()/1000 for i in range(nbOfStepInOneDay)]
gridPrices = 0.040 * np.ones(nbOfStepInOneDay)

loadForecast = [16000.0] #kWh
REgenerationForecast = [2448.0] #W




# battery 
SOCmin = 0.2
SOCmax = 0.8
selfDischarge = 0 # sigma
charge_efficiency = 1
discharge_efficiency = 1
nominalCapacity = 1800 # kWh
previous_value = 900 #kWh

#problem parameters
timeslot_duration = 10*60 # seconds 

# pymoo problem
populationSize = 100
lower_bound = np.array([-200000000])
upper_bound = np.array([300000])

def min(xi):
    balancing_requerements =  np.sum([[float(xi)], loadForecast , REgenerationForecast])
    g1 =  SOCmin * nominalCapacity - previous_value *(1- selfDischarge) + (np.where(balancing_requerements>0,balancing_requerements,0) * charge_efficiency + np.where(balancing_requerements<0,balancing_requerements,0) * discharge_efficiency ) * (timeslot_duration/3600)
    return g1

def max(xi):
    balancing_requerements =  np.sum([[float(xi)], loadForecast , REgenerationForecast])
    # max contraint induced by the battery
    g2 = previous_value *(1- selfDischarge) + (np.where(balancing_requerements>0,balancing_requerements,0) * charge_efficiency + np.where(balancing_requerements<0,balancing_requerements,0) * discharge_efficiency ) * (timeslot_duration/3600) - SOCmax * nominalCapacity
    return g2

if __name__ == "__main__":
   print(min(sys.argv[1]))
   print(f"max: {max(sys.argv[1])}")
   