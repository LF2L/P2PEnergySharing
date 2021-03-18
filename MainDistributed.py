import random
import numpy as np
from P2PSystemSim.MultiAgent import ProsumerAgent, CoordinatorAgent
from P2PSystemSim.Prosumer import *
from P2PSystemSim.Assets import *
from P2PSystemSim.PricingSystem import *
from P2PSystemSim.ConvergenceControler import NumberIterations


if __name__ == '__main__':
    # -------------------------------------problem definition----------------------------------------------------------

    gridPrices = [random.random()/1000 for i in range(144)]
    FIT = 0.035 * np.ones(144)
    with open('Forecasted/30001480014107') as f1:
        loadForecat1  = f1.read().splitlines()
        f1.close()
    loadForecat1 = [float(lf) for lf in loadForecat1]


    with open('Forecasted/30001480282717') as f2:
        loadForecat2  = f2.read().splitlines()
        f2.close()
    loadForecat2 = [float(lf) for lf in loadForecat2]

    with open('Forecasted/30001480640919') as f3:
        loadForecat3  = f3.read().splitlines()
        f3.close()
    loadForecat3 = [float(lf) for lf in loadForecat3]


    with open('Forecasted/50083502116836') as f4:
        loadForecat4 = f4.read().splitlines()
        f4.close()
    loadForecat4 = [float(lf) for lf in loadForecat4]

    PV1 = PhotovoltaicPanel(surface=180, efficiency=0.16).elecProduction("DonnéesIrradianceSolaire/03-01-2020")
    PV2 = PhotovoltaicPanel(surface=650, efficiency=0.153).elecProduction("DonnéesIrradianceSolaire/03-01-2020")
    PV3 = PhotovoltaicPanel(surface=250, efficiency=0.144).elecProduction("DonnéesIrradianceSolaire/03-01-2020")
    PV4 = PhotovoltaicPanel(surface=150, efficiency=0.16).elecProduction("DonnéesIrradianceSolaire/03-01-2020")

    battery1 = Battery(nominalCapacity=1000 * 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=200 * 600)
    battery2 = Battery(nominalCapacity=500* 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=100 * 600)
    battery3 = Battery(nominalCapacity=500* 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=100 * 600)
    battery4 = Battery(nominalCapacity=500* 600, SOCmin=0.2, SOCmax=0.8, selfDischarge=0, chargeEfficiency=1,
                       dischargeEfficiency=1, initialEnergy=100 * 600)

    prosumerAgent1 = ProsumerAgent(prosumer=Prosumer(1, loadForecat1, PV1, battery1, [random.randrange(-1,1) for i in range(144)]) )
    prosumerAgent2 = ProsumerAgent(prosumer=Prosumer(2, loadForecat2, PV2, battery2, [random.randrange(-1,1) for i in range(144)]) )
    prosumerAgent3 = ProsumerAgent(prosumer=Prosumer(3, loadForecat3, PV3, battery3, [random.randrange(-1,1) for i in range(144)]) )
    prosumerAgent4 = ProsumerAgent(prosumer=Prosumer(4, loadForecat4, PV4, battery4, [random.randrange(-1,1) for i in range(144)]) )

    coordinatorAgent = CoordinatorAgent(prosumerAgents=[prosumerAgent1, prosumerAgent2, prosumerAgent3, prosumerAgent4], FIT=FIT, gridPrices=gridPrices)
    coordinatorAgent.run(pricingScheme = SDR(gridSellPrices=gridPrices, FIT=FIT, listProsumers=[prosumerAgent1, prosumerAgent2, prosumerAgent3, prosumerAgent4] ), convergenceModel=NumberIterations(maxIterations=8))



    #-------------------------------------------------------------------------------------------------------------------