from mesa import Agent, Model
from mesa.time import RandomActivation
import pandas as pd
import numpy as np
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
from math import *
from prosumer_opti import *


def global_self_sufficiency(model):
    total_load = 0
    total_prod = 0
    for prosumer in model.schedule.agents:
        total_load += prosumer.load
    for prosumer in model.schedule.agents:
        total_prod += prosumer.production
    return total_prod / total_load if total_load != 0 else 0

def global_self_sufficiency2(model):
    total_load = 0
    total_import = 0
    for prosumer in model.schedule.agents:
        total_load += prosumer.load
    for prosumer in model.schedule.agents:
        total_import += prosumer.powerFromGrid
    return (total_load - total_import) / total_load if total_load != 0 else 0

class ProsumerAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, loadProfil, buyPrice, spotPrice,  GHIprofil, PVsurface, efficiency, stp_duration, batNomCapacity, model, initialSOC=0.5, SOCmin=0.2, SOCmax=0.9, selfDischarge=0.00, chargeEfficiency=1, dischargeEfficiency=1):
        super().__init__(unique_id, model)

        self.stp = 0
        self.step_duration = stp_duration/60  # scalar unit: hour

        # PV settings
        self.pv_surface = PVsurface  # scalar unit: m^2
        self.pv_efficiency = efficiency  # scalar unit: NaN
        self.GHIList = GHIprofil  # array unit: w/m^2

        # battery settings
        self.b_nominalCapacity = batNomCapacity * 1000 # unit: Wh
        self.b_SOCmin = SOCmin  # unit: ratio 0<=X<=1
        self.b_SOCmax = SOCmax  # unit: ratio 0<=X<=1
        self.b_SOC = initialSOC  # unit: ratio 0<=X<=1
        self.b_selfDischarge = selfDischarge  # unit: ratio 0<=X<=1
        self.b_chargeEfficiency = chargeEfficiency  # unit: ratio 0<=X<=1
        self.b_dischargeEfficiency = dischargeEfficiency  # unit: ratio 0<=X<=1
        self.b_energyLevel = initialSOC * self.b_nominalCapacity  # unit: Wh

        # prosumer setting
        self.loadProfil = loadProfil  # array unit : W
        self.load = self.loadProfil[self.stp] * self.step_duration  # scalar Wh
        self.loadPred = [ load* self.step_duration  for load in self.loadProfil] # scalar Wh
        self.production = self.solarProduction(self.GHIList[self.stp])  # scalar Wh
        self.PVprod = [ self.solarProduction(GHI)  for GHI in self.GHIList] # array
        self.power_need = self.load - self.production
        self.powerFromGrid = self.power_need if self.power_need > 0 else 0  # scalar unit: Wh
        self.powerToGrid = abs(self.power_need)  if self.power_need < 0 else 0  # scalar unit: Wh
        self.buyPrices = [price/1000 for price in buyPrice] # array unit: oginally €/kWh --> €/Wh
        self.spotPrices = [price/1000 for price in spotPrice] # array  unit : oginally €/kWh --> €/Wh
        self.cost = self.powerFromGrid * self.buyPrices[self.stp]
        self.profit = self.powerToGrid * self.spotPrices[self.stp]


        populationSize = 600

        opti_problem = MyProblem(self.spotPrices, self.buyPrices, self.b_SOCmin, self.b_SOCmax, self.b_selfDischarge, self.b_chargeEfficiency, self.b_dischargeEfficiency, self.b_nominalCapacity, self.PVprod, self.loadPred, self.b_energyLevel)

        init = np.sum([[self.loadPred], [self.PVprod]], axis=0)
        init_design_space = np.zeros((populationSize, opti_problem.n_var))
        init_design_space[:, :144] = init
        init_design_space[:, 144] = self.b_energyLevel

        termination = MultiObjectiveDefaultTermination(
            x_tol=1e-8,
            cv_tol=1e-2,
            f_tol=0.0025,
            nth_gen=5,
            n_last=10,
            n_max_gen=1600,
            n_max_evals=10000000
        )
        algorithm = GA(pop_size=populationSize,
                    eliminate_duplicates=True, sampling=init_design_space)

        res = minimize(opti_problem,
                        algorithm,
                        termination= termination,
                        return_least_infeasible=True,
                        seed=1,
                        save_history=True,
                        verbose= True)

        self.b_predicted_energy_level = res.X[144:]
        self.powerFromGrid_opti_pred = np.where(res.X[:144]>0,res.X[:144],0)
        self.powerToGrid_opti_pred =  np.where(res.X[:144]<0,abs(res.X[:144]),0) 


    def solarProduction(self, ghi):
        return ghi*self.pv_surface*self.pv_efficiency  # * self.step_duration

    def step(self):
        
        self.load = self.loadProfil[self.stp] * self.step_duration
        self.production = self.solarProduction(self.GHIList[self.stp])
        self.power_need= self.load - self.production 

        # battery prediction as constraint 
        b_energy_provided=self.b_energyLevel - self.b_predicted_energy_level[self.stp]
        if(b_energy_provided> 0 ):
            # discharging the battery 
            if (self.b_predicted_energy_level[self.stp] >= self.b_nominalCapacity * self.b_SOCmin):
                self.b_energyLevel = self.b_predicted_energy_level[self.stp]
                self.powerFromGrid = self.power_need - b_energy_provided if self.power_need - b_energy_provided >0 else 0
                self.powerToGrid = self.power_need - b_energy_provided if self.power_need - b_energy_provided <0 else 0
            else:
                self.b_energyLevel = self.b_nominalCapacity * self.b_SOCmin
                self.powerFromGrid = self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmin) if self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmin) >0 else 0
                self.powerToGrid = self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmin) if self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmin) <0 else 0
            # if (self.power_need>0):
            #     # need energy 
            #     self.powerFromGrid = self.power_need - b_energy_provided
            #     self.powerToGrid = 0
            # else:
            #     # produce excedent energy 
            #     self.powerFromGrid = 0 
            #     self.powerToGrid = self.power_need + b_energy_provided
        else:
            # charging the battery 
            if (self.b_predicted_energy_level[self.stp] <= self.b_nominalCapacity * self.b_SOCmax):
                self.b_energyLevel = self.b_predicted_energy_level[self.stp]
                self.powerFromGrid = self.power_need - b_energy_provided if self.power_need - b_energy_provided >0 else 0
                self.powerToGrid = self.power_need - b_energy_provided if self.power_need - b_energy_provided <0 else 0
            else:
                self.b_energyLevel = self.b_nominalCapacity * self.b_SOCmax
                self.powerFromGrid = self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmax) if self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmax) >0 else 0
                self.powerToGrid = self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmax)  if self.power_need - (self.b_energyLevel - self.b_nominalCapacity * self.b_SOCmax)  <0 else 0

            # self.b_energyLevel = self.b_predicted_energy_level[self.stp]
            # if (self.power_need>0):
            #     self.powerFromGrid = self.power_need - b_energy_provided
            #     self.powerToGrid = 0
            # else:
            #     # produce excedent energy 
            #     self.powerFromGrid = 0 
            #     self.powerToGrid = self.power_need + b_energy_provided


        self.cost = self.powerFromGrid * self.buyPrices[self.stp] if self.powerFromGrid > 0 else 0
        self.profit = self.powerToGrid * self.spotPrices[self.stp] if self.powerToGrid > 0 else 0

        # battery usage
        self.b_SOC = self.b_energyLevel / self.b_nominalCapacity

        self.stp += 1

class P2PEnergyTradingModel(Model):
    """A model with some number of agents."""

    def __init__(self, prosumerDataList, step_duration):
        self.num_agents = len(prosumerDataList)
        self.schedule = RandomActivation(self)  # SimultaneousActivation(self)
        # self.global_battery

        for i in range(self.num_agents):
            a = ProsumerAgent(i, prosumerDataList[i]["load"], prosumerDataList[i]["buyPrice"], prosumerDataList[i]["FIT"],  prosumerDataList[i]["GHI"],
                              prosumerDataList[i]["PV_sufrace"], prosumerDataList[i]["PV_efficiency"], step_duration, prosumerDataList[i]["nominalCapacity"],  self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={
                "global_self_sufficiency": global_self_sufficiency, "global_self_sufficiency2": global_self_sufficiency2},
            agent_reporters={"SOC": "b_SOC", "cost": "cost", "profit": "profit", "load": "load", "production": "production","powerNeed":"power_need", "importFromGrid": "powerFromGrid", "exportToGrid": "powerToGrid", "SOCmin": "b_SOCmin", "SOCmax": "b_SOCmax"})

    def step(self):
        # do the optimisation
        # self.schedule.agents

        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':

    stepSize = 10  # minutes
    # day time in minutes divided by the size of one step --> number of timeslot in a day
    nbOfStepInOneDay = int(1440 / stepSize)

    # define the FeedInTariff as constant over the day (based on the tarif defined in data.gouv.fr) - 10c€/kWh
    FeedInTariff = 0.10 * np.ones(nbOfStepInOneDay)

    data = []

    with open('Forecasted/30001480014107') as f1:
        loadForecat1 = f1.read().splitlines()
    loadForecat1 = [float(lf) for lf in loadForecat1]

    with open('Forecasted/30001480282717') as f2:
        loadForecat2 = f2.read().splitlines()
    loadForecat2 = [float(lf) for lf in loadForecat2]

    with open('Forecasted/30001480640919') as f3:
        loadForecat3 = f3.read().splitlines()
    loadForecat3 = [float(lf) for lf in loadForecat3]

    with open('Forecasted/50083502116836') as f4:
        loadForecat4 = f4.read().splitlines()
    loadForecat4 = [float(lf) for lf in loadForecat4]

    df = pd.read_csv('DonnéesIrradianceSolaire/03-01-2020')
    GHIlist = df["GHI"].tolist()

    spotPricesDF = pd.read_csv(
        'DonneesEnergyConsometers/spotPrice.csv', sep=";")
    spotPrices = spotPricesDF["kwhPrice"].tolist()

    prosumerData1 = {'buyPrice': spotPrices, 'FIT': FeedInTariff.tolist(), 'load': loadForecat1,
                     'GHI': GHIlist, 'PV_sufrace': 180, 'PV_efficiency': 0.16, 'nominalCapacity': 3.99}
    data.append(prosumerData1)
    prosumerData2 = {'buyPrice': spotPrices, 'FIT': FeedInTariff.tolist(), 'load': loadForecat2,
                     'GHI': GHIlist, 'PV_sufrace': 650, 'PV_efficiency': 0.153, 'nominalCapacity': 2.98}
    data.append(prosumerData2)
    prosumerData3 = {'buyPrice': spotPrices, 'FIT': FeedInTariff.tolist(), 'load': loadForecat3,
                     'GHI': GHIlist, 'PV_sufrace': 250, 'PV_efficiency': 0.144, 'nominalCapacity': 29.4}
    data.append(prosumerData3)
    prosumerData4 = {'buyPrice': spotPrices, 'FIT': FeedInTariff.tolist(), 'load': loadForecat4,
                     'GHI': GHIlist, 'PV_sufrace': 150, 'PV_efficiency': 0.16, 'nominalCapacity': 18.83}
    data.append(prosumerData4)

    # --------------- RUN THE MODEL ------------------------------
    model = P2PEnergyTradingModel(data, stepSize)
    for i in range(nbOfStepInOneDay):
        model.step()

    # --------------- PRINT RESULTS -------------------------

    model_data = model.datacollector.get_model_vars_dataframe()
    print(f"self sufficuiency 1: {model_data['global_self_sufficiency'].sum()} %")
    print(f"self sufficuiency 2: {model_data['global_self_sufficiency2'].sum()} %")

    agent_data = model.datacollector.get_agent_vars_dataframe()
    # agent_data.xs(0, level="AgentID")["load"].plot()
    # plt.show()
    nb_graph_horizontal = 2

    fig, axs = plt.subplots(nb_graph_horizontal, ceil(
        len(data)/nb_graph_horizontal))
    fig.suptitle('Production and consumption of each prosumer')
    for i in range(len(data)):
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["production"], label="Energy produced by PV")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["load"], label="Energy consumed by prosumer")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(i, level="AgentID")[
            "importFromGrid"], label="Energy from grid")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(i, level="AgentID")[
            "exportToGrid"], label="Energy to grid")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(i, level="AgentID")[
            "powerNeed"], label="Energy Need")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].set(xlabel='timeslots', ylabel='Power (Wh)',
                                                                 title='Prosumer {}'.format(i))
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].legend()
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper right')
    plt.show()

    fig, axs = plt.subplots(2, ceil(len(data)/nb_graph_horizontal))
    fig.suptitle('Energy cost of each prosumer')
    for i in range(len(data)):
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["cost"], label="Energy cost")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["profit"], label="Energy selling profit")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].set(xlabel='timeslots', ylabel='Cost (€)',
                                                                 title='Prosumer {}'.format(i))
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].legend()
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper right')
    plt.show()

    fig, axs = plt.subplots(2, ceil(len(data)/nb_graph_horizontal))
    fig.suptitle('Battery state for each prosumer')
    for i in range(len(data)):
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["SOC"], label="State of charge")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].set(xlabel='timeslots', ylabel='State (%)',
                                                                 title='Prosumer {}'.format(i))
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].hlines(y=agent_data.xs(
            i, level="AgentID")["SOCmin"], xmin = 0 , xmax = nbOfStepInOneDay, label="Level Min")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].hlines(y=agent_data.xs(
            i, level="AgentID")["SOCmax"], xmin = 0 , xmax = nbOfStepInOneDay, label="Level Max")
        axs[i//nb_graph_horizontal][i % nb_graph_horizontal].legend()
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper right')
    plt.show()
