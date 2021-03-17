import random
import numpy as np
from P2PSystemSim.Assets import *
from P2PSystemSim.Prosumer import *
from P2PSystemSim.CoordinationSytem import *
from utils import calcREgeneration

if __name__ == '__main__':
    # -------------------------------------problem definition----------------------------------------------------------
    stepSize = 10 # minutes 
    nbOfStepInOneDay= int(1440/ stepSize) # entire time in minutes divided by the size of one step

    gridPrices = [random.random()/1000 for i in range(nbOfStepInOneDay)]
    FeedInTariff = 0.035 * np.ones(nbOfStepInOneDay) # define the FeedInTariff as constant over the day (based on the tarif defined in data.gouv.fr) - 10c€/kWh
    with open('Forecasted/30001480014107') as f1:
        loadForecat1  = f1.read().splitlines()
    loadForecat1 = [float(lf) for lf in loadForecat1]


    with open('Forecasted/30001480282717') as f2:
        loadForecat2  = f2.read().splitlines()
    loadForecat2 = [float(lf) for lf in loadForecat2]

    with open('Forecasted/30001480640919') as f3:
        loadForecat3  = f3.read().splitlines()
    loadForecat3 = [float(lf) for lf in loadForecat3]


    with open('Forecasted/50083502116836') as f4:
        loadForecat4 = f4.read().splitlines()
    loadForecat4 = [float(lf) for lf in loadForecat4]

    REgeneration1 = calcREgeneration("DonnéesIrradianceSolaire/03-01-2020", SPefficiency=0.16, SParea=180)

    REgeneration2 = calcREgeneration("DonnéesIrradianceSolaire/03-01-2020", SPefficiency=0.153, SParea=650)

    REgeneration3 = calcREgeneration("DonnéesIrradianceSolaire/03-01-2020", SPefficiency=0.144, SParea=250)

    REgeneration4 = calcREgeneration("DonnéesIrradianceSolaire/03-01-2020", SPefficiency=0.16, SParea=150)



    battery1 = Battery(nominalCapacity=1000 * 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=200 * 600)
    battery2 = Battery(nominalCapacity=500* 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=100 * 600)
    battery3 = Battery(nominalCapacity=500* 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=100 * 600)
    battery4 = Battery(nominalCapacity=500* 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=100 * 600)
    prosumer1 = Prosumer(1, loadForecat1, REgeneration1,battery1)
    prosumer2 = Prosumer(2, loadForecat2, REgeneration2,battery2)
    prosumer3 = Prosumer(3, loadForecat3, REgeneration3, battery3)
    prosumer4 = Prosumer(4, loadForecat4, REgeneration4, battery4)
    #-------------------------------------------------------------------------------------------------------------------
    # instantiate RegularCoordinator with the prosumer list, grid prices and FeedInTariff
    coordinator = RegularCoordinator(prosumerList=[prosumer1, prosumer2, prosumer3, prosumer4],gridPrices=gridPrices, FIT=FeedInTariff, algorithm="GA")
    #run optimisation and price calculation
    res, pricedic = coordinator.run()

    # display results
    prosumer1.displayGraph()
    print(pricedic)