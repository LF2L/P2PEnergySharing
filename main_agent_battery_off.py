from mesa import Agent, Model
from mesa.time import RandomActivation
import pandas as pd
import numpy as np
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt
from math import *


def global_self_sufficiency(model):
    total_load = 0
    total_prod = 0
    for prosumer in model.schedule.agents:
        total_load += prosumer.load
    for prosumer in model.schedule.agents:
        total_prod += prosumer.production
    return total_prod / total_load if total_prod != 0 else 0


class ProsumerAgent(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, loadProfil, buyPrice, spotPrice,  GHIprofil, PVsurface, efficiency, stp_duration, batNomCapacity, model, initialSOC = 0.5, SOCmin=0.2, SOCmax=0.9, selfDischarge=0.00, chargeEfficiency=1, dischargeEfficiency=1):
        super().__init__(unique_id, model)

        self.stp = 0
        self.step_duration = stp_duration/60  # scalar unit: hour
        
        # PV settings
        self.pv_surface = PVsurface  # scalar unit: m^2
        self.pv_efficiency = efficiency  # scalar unit: NaN
        self.GHIList = GHIprofil  # array unit: w/m^2

        # battery settings
        self.b_nominalCapacity = batNomCapacity  # unit: kWh
        self.b_SOCmin = SOCmin  # unit: ratio 0<=X<=1
        self.b_SOCmax = SOCmax  # unit: ratio 0<=X<=1
        self.b_SOC = initialSOC # unit: ratio 0<=X<=1
        self.b_selfDischarge = selfDischarge  # unit: ratio 0<=X<=1
        self.b_chargeEfficiency = chargeEfficiency  # unit: ratio 0<=X<=1
        self.b_dischargeEfficiency = dischargeEfficiency  # unit: ratio 0<=X<=1
        self.b_energyLevel = initialSOC * batNomCapacity # unit: kWh

        # prosumer setting
        self.loadProfil = loadProfil  # array unit : W
        self.load = self.loadProfil[self.stp] * self.step_duration  # scalar Wh
        self.production = self.solarProduction(self.GHIList[self.stp])  # scalar Wh
        self.powerFromGrid = self.load - self.production # scalar unit: Wh
        self.buyPrices=  [price/1000 for price in buyPrice] # scalar unit: oginally €/kWh --> €/Wh
        self.spotPrices = spotPrice
        self.cost = self.powerFromGrid * self.buyPrices[self.stp] if self.powerFromGrid > 0 else 0
        self.profit = self.powerFromGrid * self.spotPrices[self.stp] if self.powerFromGrid < 0 else 0


    def solarProduction(self, ghi):
        return ghi*self.pv_surface*self.pv_efficiency # * self.step_duration

    def optimise(self):
         

    def step(self):
        self.load = self.loadProfil[self.stp] * self.step_duration
        self.production = self.solarProduction(self.GHIList[self.stp])
        self.powerFromGrid = self.load - self.production - self.b_energyLevel *(1- self.b_SOCmin)
        self.cost = self.powerFromGrid * self.buyPrices[self.stp] if self.powerFromGrid > 0 else 0
        self.profit = self.powerFromGrid * self.spotPrices[self.stp] if self.powerFromGrid < 0 else 0
        self.b_energyLevel = self.b_energyLevel* (1- self.b_selfDischarge) 
        self.b_SOC = self.b_energyLevel / self.b_nominalCapacity
        
        self.stp += 1
        # The agent's step will go here.
        # if self.wealth == 0:
        #     return
        # other_agent = self.random.choice(self.model.schedule.agents)
        # other_agent.wealth += 1
        # self.wealth -= 1


class P2PEnergyTradingModel(Model):
    """A model with some number of agents."""

    def __init__(self, prosumerDataList, step_duration):
        self.num_agents = len(prosumerDataList)
        self.schedule = RandomActivation(self) # SimultaneousActivation(self)
        # self.global_battery


        for i in range(self.num_agents):
            a = ProsumerAgent(i, prosumerDataList[i]["load"], prosumerDataList[i]["buyPrice"], prosumerDataList[i]["FIT"],  prosumerDataList[i]["GHI"],
                              prosumerDataList[i]["PV_sufrace"], prosumerDataList[i]["PV_efficiency"], step_duration, prosumerDataList[i]["nominalCapacity"],  self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={
                "global_self_sufficiency": global_self_sufficiency},
            agent_reporters={"SOC": "b_SOC", "cost": "cost", "load": "load", "production": "production", "importFromGrid": "powerFromGrid"})

    def step(self):
        # do the optimisation
        self.schedule.agents


        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':

    stepSize = 10  # minutes
    # day time in minutes divided by the size of one step --> number of timeslot in a day
    nbOfStepInOneDay = int(1440 / stepSize)

    # define the FeedInTariff as constant over the day (based on the tarif defined in data.gouv.fr) - 10c€/kWh
    FeedInTariff = 0.035 * np.ones(nbOfStepInOneDay)

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

    spotPricesDF = pd.read_csv('DonneesEnergyConsometers/spotPrice.csv',sep=";")
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
    print(f"{model_data} %")

    agent_data = model.datacollector.get_agent_vars_dataframe()
    # agent_data.xs(0, level="AgentID")["load"].plot()
    # plt.show()
    nb_graph_horizontal = 2

    fig, axs = plt.subplots(nb_graph_horizontal, ceil(len(data)/nb_graph_horizontal))
    fig.suptitle('Production and consumption of each prosumer')
    for i in range(len(data)):
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["production"], label="Energy produced by PV")
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["load"], label="Energy consumed by prosumer")
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(i, level="AgentID")[
                    "importFromGrid"], label="Energy from grid")
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].set(xlabel='timeslots', ylabel='Power (Wh)',
                   title='Prosumer {}'.format(i))
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].legend()
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper right')
    plt.show()

    fig, axs = plt.subplots(2, ceil(len(data)/nb_graph_horizontal))
    fig.suptitle('Energy cost of each prosumer')
    for i in range(len(data)):
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["cost"], label="Energy cost")
        # axs[i].plot(range(0, nbOfStepInOneDay), agent_data.xs(
        #     i, level="AgentID")["load"], label="Load forecast")
        # axs[i].plot(range(0, nbOfStepInOneDay), agent_data.xs(i, level="AgentID")[
        #             "importFromGrid"], label="Importation from grid")
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].set(xlabel='timeslots', ylabel='Cost (€)',
                   title='Prosumer {}'.format(i))
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].legend()
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper right')
    plt.show()

    fig, axs = plt.subplots(2, ceil(len(data)/nb_graph_horizontal))
    fig.suptitle('Battery state for each prosumer')
    for i in range(len(data)):
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].plot(range(0, nbOfStepInOneDay), agent_data.xs(
            i, level="AgentID")["SOC"], label="State of charge")
        # axs[i].plot(range(0, nbOfStepInOneDay), agent_data.xs(
        #     i, level="AgentID")["load"], label="Load forecast")
        # axs[i].plot(range(0, nbOfStepInOneDay), agent_data.xs(i, level="AgentID")[
        #             "importFromGrid"], label="Importation from grid")
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].set(xlabel='timeslots', ylabel='State (%)',
                   title='Prosumer {}'.format(i))
        axs[i//nb_graph_horizontal][i%nb_graph_horizontal].legend()
    # lines, labels = fig.axes[-1].get_legend_handles_labels()
    # fig.legend(lines, labels, loc='upper right')
    plt.show()