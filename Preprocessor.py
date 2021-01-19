import abc
import pandas as pd
from operator import *
from multiprocessing import current_process
from functools import partial
from utils import *
from datetime import datetime as dt

#-----------------------------------------------------------------------------------------------------------------------
# this function will be used by parallel processes so it must be out of the class definition
#-----------------------------------------------------------------------------------------------------------------------
def job(partition: pd.DataFrame, stepLength, historyLength, debug=False):
    print(len(partition))
    start_index = 0
    finish_index = historyLength
    columns = list()
    columns.append('time')
    for i in range(historyLength):
        columns.append('t-' + str(historyLength - i))
    columns.append("t")
    trData = pd.DataFrame([], columns=columns)
    while finish_index < len(partition):
        #print(current_process())
        progress(finish_index, len(partition), status='preparing training data')
        liste = list()
        needed = partition[start_index:finish_index + 1]
        if needed is not None:
            #time = needed[historyLength:historyLength + 1].index[0][1]
            time = needed[historyLength:historyLength +1]["time"].tolist().pop()
            liste.append(time)
            history_and_current = needed["netLoad"].tolist()
            liste.extend(history_and_current)
            df1 = pd.DataFrame([liste], columns=columns)
            trData = trData.append(df1)
            start_index = start_index + stepLength
            finish_index = finish_index + stepLength
    #if debug:
    print(current_process())
    print("trainindData")
    print(trData)

    return trData
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
from Macros import *

class Preprocessor(metaclass=abc.ABCMeta): # equivalent to java abstract class
    def __init__(self, energyLoadfilePath, GHIfilePath, SPefficiency, SParea):
        """

        :param energyLoadfilePath: path to file containing energy load data
        :param SParea: area of the solar panel
        :param SPefficiency: solar panel efficiency
        """
        self._area = SParea
        self._efficiency  = SPefficiency
        self._directLoadFilePath = energyLoadfilePath
        self._GHIfilePath  = GHIfilePath
        self._directLoad = None
    def directLoad(self):
        """
        :return: object pandas.DataFrame with only the desired fields and with dates and times rewritten in desired format
        """
        df = pd.read_csv(self._directLoadFilePath, delimiter=";")
        #df = df.set_index(['date', 'time'])
        df = df.drop(columns="status")
        df.dropna(subset = ["powerLoad"], inplace=True)
        df = df.reset_index(drop=True)
        self._directLoad = df
        #print(df[0:1]["date"].tolist().pop())
        return df
    @abc.abstractmethod
    def REgeneration(self):
        pass

    def netLoad(self, loadData, REgenerationData):
        if type(self)==Preprocessor:
            #there is no REgeneration in the generic case so net load is the  grid power load
            netLoad = loadData["powerLoad"].tolist()
            se = pd.Series(netLoad)
            loadData["netLoad"] = se.values
            return loadData

    def recursive_build(self, netLoad: pd.DataFrame, historyLength: int = 145, stepLength: int = 145, nbProcesses = 8, debug = False):
        """
        :param netLoad: pd.DataFrame ["date", "time", "powerLoad", "netLoad"]
        :param historyLength: how many time slots we want to include in variables array X (to predict Y)
        :param stepLength: step length between first batch of selected data from the netload and next batch. Eatch batch makes up one row of the training data
        :param nbProcesses: nuber of parallel processes to share the job
        :return:training data: pandas.DataFrame X =['timeSlotStart', 'temperature', 't-N', 't-(N-1)', ...., 't-1', 't'] where Y = ['t'] for many timeslots...days
                     prediction data: pandas.DataFrame X =['timeSlotStart', 'temperature', 't-N', 't-(N-1)', ...., 't-1'] for [historyLength] nb of time slots
        """
        #creating partitions of the netLoad Dataframe to assign to different jobs---------------------------------------
        l = len(netLoad) // nbProcesses
        partitions = []
        for i in range(0, nbProcesses):
            begin = i * l
            if begin != (nbProcesses - 1) * l:
                end = begin + l - 1
            else:
                end = len(netLoad)
            partition = netLoad[begin:end + 1]
            print("partition")
            print(partition)
            partitions.append(partition)
        assert sum([len(part) for part in partitions])==len(netLoad)
        #creating jobs--------------------------------------------------------------------------------------------------
        #JOB = partial(job, stepLength = stepLength, historyLength=historyLength,debug=debug )
        #F = pool.map(JOB, partitions)
        #reassembling results of every process--------------------------------------------------------------------------
        #trainingData=pd.DataFrame(list(), columns=F[0].columns)
        #for f in F:
        #    trainingData = trainingData.append(f)
        #print("len partition")
        #print(len(partitions[0]))
        trainDat = job(netLoad, historyLength= historyLength, stepLength = stepLength)
        #creating prediction data---------------------------------------------------------------------------------------
        prediciton_need = netLoad[len(netLoad)-historyLength-145: len(netLoad)-1]
        # what's enough to build prediction data for the last day. data is from t-2879...t and i need
        # to predict t+1 for each time slot
        predictionData = job(prediciton_need, stepLength=1, historyLength=historyLength)
        predictionData = predictionData.drop([predictionData.columns[1]], axis=1)
        if debug:
            print("trainingData")
            print(predictionData)
        #---------------------------------------------------------------------------------------------------------------
        return trainDat, predictionData


    def preprocess(self, historyLength=20 * 144,stepLength=145, debug=False):
        """
            :param debug: True if you want to print the prediction data
            :param historyLength: number of time slots whose energy net load we would use as variables for training
            :return: training data: pandas.DataFrame X =['timeSlotStart', 'temperature', 't-N', 't-(N-1)', ...., 't-1', 't'] where Y = ['t'] for many timeslots...days
                     prediction data: pandas.DataFrame X =['timeSlotStart', 'temperature', 't-N', 't-(N-1)', ...., 't-1'] for [historyLength] nb of time slots
        """

        loadData = self.directLoad()
        REgenerationData = self.REgeneration()
        net_load = self.netLoad(loadData=loadData, REgenerationData=REgenerationData)
        #net_load = net_load.apply(pd.to_numeric, errors='coerce')
        #net_load = net_load.dropna()
        return self.recursive_build(net_load, historyLength=historyLength, stepLength=stepLength,debug=debug)


class REpreprocessor(Preprocessor):
    def REgeneration(self):
        """
        :param filePath: path to csv file with solar irradiance data (GHI). columns are ["date", "time", "GHI"]
        :return: pandas.DataFrame with RE generation power for every time slot (same dimension as load forecasts)
        """
        solarData = pd.read_csv(self._GHIfilePath, delimiter = ",")
        #solarData = solarData.set_index(["PeriodEnd", "PeriodStart"])
        solarData["REgeneration"] = solarData["GHI"]*self._area*self._efficiency
        solarData = solarData.drop(["GHI"], axis=1)
        startDate = dt.strptime(self._directLoad[0:1]["date"].tolist().pop(),"%d-%m-%Y")
        startTime = dt.strptime(self._directLoad[0:1]["time"].tolist().pop(),"%H:%M")
        for i in range(len(solarData)):
            stPeriod = solarData[i:i+1]["PeriodStart"].tolist().pop()
            stDate = dt.strptime(stPeriod.split("T")[0],"%Y-%m-%d")
            stTime = dt.strptime(stPeriod.split("T")[1].split("Z")[0],"%H:%M:%S")
            if stDate == startDate and stTime == startTime:
                neededSolar = solarData[i:i+len(self._directLoad)]
                assert len(neededSolar)==len(self._directLoad)

        return neededSolar


    def netLoad(self, loadData, REgenerationData):
        """
        :param loadData: pandas.DataFrame with columns["date", "time", "powerLoad"]
        :param REgenerationData: pandas.DataFrame with columns["date","time","REgeneration"]
        :return: pandas.DataFrame with columns ["date", "time", "netLoad"]
        """
        netLoad = list( map(add,  loadData["powerLoad"].tolist(), REgenerationData["REgeneration"].tolist()))
        se = pd.Series(netLoad)
        loadData['netLoad'] = se.values
        #loadData = loadData.drop(["powerLoad"], columns = 1)
        #print("netLoad")
        #print(loadData)
        return loadData









if __name__ == '__main__':

 preprocessor1 = REpreprocessor(energyLoadfilePath="DonnéesEnergyConsometers/Enedis_M021_CDC_A04NZ5SK_30001480014107.csv",
                                  GHIfilePath="DonnéesIrradianceSolaire/47.75_-3.3667_Solcast_PT10M.csv", SParea=180,SPefficiency=0.1666)
 preprocessor2 = REpreprocessor(energyLoadfilePath="DonnéesEnergyConsometers/Enedis_M021_CDC_A04NZ5SK_30001480282717.csv",
 GHIfilePath="DonnéesIrradianceSolaire/47.75_-3.3667_Solcast_PT10M.csv", SParea = 650, SPefficiency=0.1538)
 preprocessor3 = REpreprocessor(energyLoadfilePath="DonnéesEnergyConsometers/Enedis_M021_CDC_A04NZ5SK_50083502116836.csv",
 GHIfilePath="DonnéesIrradianceSolaire/47.75_-3.3667_Solcast_PT10M.csv",SParea = 250, SPefficiency=0.1440)
 preprocessor4 = REpreprocessor(energyLoadfilePath="DonnéesEnergyConsometers/Enedis_M021_CDC_A04NZ5SK_30001480640919.csv",
 GHIfilePath="DonnéesIrradianceSolaire/47.75_-3.3667_Solcast_PT10M.csv", SParea = 150, SPefficiency=0.1666)
 trainingData1, predicitonData1 = preprocessor1.preprocess(historyLength=144*20,stepLength=145, debug = False)
 trainingData2, predicitonData2 = preprocessor2.preprocess(historyLength=144*20,stepLength=145, debug = False)
 trainingData3, predicitonData3 = preprocessor3.preprocess(historyLength=144*20,stepLength=145, debug = False)
 trainingData4, predicitonData4 = preprocessor4.preprocess(historyLength=144*20,stepLength=145, debug = False)

 trainingData1.to_csv("Forecasted/trainingData1",index=False)
 trainingData2.to_csv("Forecasted/trainingData2", index=False)
 trainingData3.to_csv("Forecasted/trainingData3", index=False)
 trainingData4.to_csv("Forecasted/trainingData4", index=False)

 predicitonData1.to_csv("Forecasted/predictionData1.csv", index=False)
 predicitonData2.to_csv("Forecasted/predictionData2", index=False)
 predicitonData3.to_csv("Forecasted/predictionData3", index=False)
 predicitonData4.to_csv("Forecasted/predictionData4", index=False)




