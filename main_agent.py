from mesa import Agent, Model
from mesa.time import RandomActivation
import pandas as pd
from mesa.datacollection import DataCollector
import matplotlib.pyplot as plt

def global_self_sufficiency(model):
    total_load=0
    total_prod=0
    for prosumer in model.schedule.agents:
        total_load += prosumer.load
    for prosumer in model.schedule.agents:
        total_prod += prosumer.production
    return total_prod / total_load if total_prod != 0 else 0

class ProsumerAgent(Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, loadProfil, GHIprofil, PVsurface, efficiency, stp_duration, model):
        super().__init__(unique_id, model)
        self.loadProfil = loadProfil # array unit : W
        # self.productionProfil # array unit: kW
        self.GHIList = GHIprofil # array unit: w/m^2
        self.load = 0 # scalar kWh
        self.production = 0 # scalar kWh
        self.pv_surface = PVsurface # scalar unit: m^2
        self.pv_efficiency =  efficiency # scalar unit: NaN
        self.step_duration = stp_duration # scalar unit: hour
        self.stp = 0 
        self.powerFromGrid = 0 # scalar unit: kWh
        # self.shiftWilligness
        # self.b_nominalCapacity #kWh
        # self.b_nominalCapacity #kWh

    def solarProduction(self, ghi):
        return ghi*self.pv_surface*self.pv_efficiency* self.step_duration


    def step(self):
        self.load = self.loadProfil[self.stp] * self.step_duration
        self.production = self.solarProduction(self.GHIList[self.stp])
        self.powerFromGrid = self.load - self.production
        self.stp +=1
        # The agent's step will go here.
        # if self.wealth == 0:
        #     return
        # other_agent = self.random.choice(self.model.schedule.agents)
        # other_agent.wealth += 1
        # self.wealth -= 1

class P2PEnergyTradingModel(Model):
    """A model with some number of agents."""
    def __init__(self, prosumerDataList, step_duration ):
        self.num_agents = len(prosumerDataList)
        self.schedule = RandomActivation(self)

        for i in range(self.num_agents):
            a = ProsumerAgent(i, prosumerDataList[i]["load"] , prosumerDataList[i]["GHI"], prosumerDataList[i]["PV_sufrace"], prosumerDataList[i]["PV_efficiency"], step_duration, self)
            self.schedule.add(a)

        self.datacollector = DataCollector(
            model_reporters={"global_self_sufficiency": global_self_sufficiency},
            agent_reporters={"load": "load","production": "production", "importFromGrid": "powerFromGrid"})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()


if __name__ == '__main__':

    stepSize = 10 # minutes 
    nbOfStepInOneDay= int(1440/ stepSize) # day time in minutes divided by the size of one step --> number of timeslot in a day

    data = []

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

    df = pd.read_csv('DonnéesIrradianceSolaire/03-01-2020')
    GHIlist = df["GHI"].tolist()

    prosumerData1 = {'load': loadForecat1, 'GHI': GHIlist, 'PV_sufrace': 180, 'PV_efficiency': 0.16}
    data.append(prosumerData1)
    prosumerData2 = {'load': loadForecat2, 'GHI': GHIlist, 'PV_sufrace': 650, 'PV_efficiency': 0.153}
    data.append(prosumerData2)
    prosumerData3 = {'load': loadForecat3, 'GHI': GHIlist, 'PV_sufrace': 250, 'PV_efficiency': 0.144}
    data.append(prosumerData3)
    prosumerData4 = {'load': loadForecat4, 'GHI': GHIlist, 'PV_sufrace': 150, 'PV_efficiency': 0.16}
    data.append(prosumerData4)


    ## --------------- RUN THE MODEL ------------------------------
    model = P2PEnergyTradingModel(data, stepSize)
    for i in range(nbOfStepInOneDay):
        model.step()

    ## --------------- PRINT RESULTS -------------------------

    model_data = model.datacollector.get_model_vars_dataframe()
    print(f"{model_data.sum()} %")


    agent_data = model.datacollector.get_agent_vars_dataframe()
    # agent_data.xs(0, level="AgentID")["load"].plot()
    # plt.show()

    fig, axs = plt.subplots(1, len(data) )
    fig.suptitle('Production and consumption of each prosumer')
    for i in range(len(data)): 
        axs[i].plot(range(0,nbOfStepInOneDay), agent_data.xs(i, level="AgentID")["production"], label="RE generation")
        axs[i].plot(range(0,nbOfStepInOneDay), agent_data.xs(i, level="AgentID")["load"], label="Load forecast")
        axs[i].plot(range(0,nbOfStepInOneDay), agent_data.xs(i, level="AgentID")["importFromGrid"], label="Importation from grid")
        axs[i].set(xlabel='timeslots', ylabel='Power (Wh)', title='Prosumer {}'.format(i))
        # axs[i].legend()
    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines,labels, loc='upper right')
    plt.show()