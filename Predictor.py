from sklearn import preprocessing
from pprint import pprint, PrettyPrinter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPRegressor
from Preprocessor import *
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.metrics import make_scorer, mean_squared_error

#feature selection methods----------------------------------------------------------------------------------------------
class FeatureSelectionMethod(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        if type(self) == FeatureSelectionMethod:
            raise TypeError(
                "FeatureSelectionMethod is an interface, it cannot be instantiated. Try a concrete method, e.g.: RFE"
            )
    @abc.abstractmethod
    def featureSelection(self, trainingData: pd.DataFrame, predictionData: pd.DataFrame, debug=False):
        pass
class UnivariateMethod(FeatureSelectionMethod):
    def __init__(self):
        self._trainingData = None #instantiation
    def featureSelection(self, trainingData: pd.DataFrame, predictionData: pd.DataFrame, debug=False):
        """
            :param predictionData: trainingData: pandas.DataFrame X =['timeSlotStart', 't-(N-1)', ...., 't-1','t']
            :param trainingData: pandas.DataFrame X =['timeSlotStart', 't-N', 't-(N-1)', ...., 't-1','t'] for many days where Y = ['t']
            :param debug: True if you want prints. False by default
            :return new_X, Y
        """
        self._trainingData = trainingData
        y = pd.DataFrame(self._trainingData["t"].tolist(), columns=["t"])
        X = self._trainingData.drop(["t"], axis=1)
        print("nb of initial features")
        print(X.shape)
        selector = SelectKBest(f_regression, k=300).fit(X, y.values.ravel())
        print("im here")
        X_new = selector.transform(X)
        X_predict = selector.transform(predictionData)
        #X_new = SelectKBest(f_regression, k=2).fit_transform(X, y)
        print(X_new.shape)
        if debug:
            # ------------------------------------------------------Debug prints---------------------------------------------
            print("grid scores: ----------------------------------------------------------------------------------------")
            print(selector.pvalues_)
            print("ranking ----------------------------------------------------------------------------------------------")
            ranking = []
            for i in range(len(selector.get_support())):
                if selector.get_support()[i] == True:
                    ranking.append(i)
            print(ranking)
        d = dict()

        for i in range((X.shape[1]-1)//144+1):
            d[i] = []
        for nb in ranking:
            s=nb//144
            d[s].append(nb)
        otherd = dict()
        for key in d.keys():
            l=[]
            for ts in d[key]:
                l.append(ts - 144*key)
            otherd[key] = l
        #PrettyPrinter(depth=2).pprint(d)
        print("sorted")
        for elem in sorted(d.keys()):
            print([elem, otherd[elem]])

        xx=[]
        yy=[]
        for key in sorted(otherd.keys()):
            for t in otherd[key]:
                xx.append(key)
                yy.append(t)
        plt.scatter(xx, yy, s=10)
        oother = otherd
        for key in oother.keys():
            oother[key] = len(oother[key])
        pprint(oother)
        plt.xlabel("Previous days")
        plt.ylabel("Time slots")
        plt.savefig("fig/fig")

        return X_new, y, X_predict


class RFEmethod(FeatureSelectionMethod):
    def __init__(self):
        self._trainingData = None #instantiate with None



    def featureSelection(self, trainingData: pd.DataFrame, predictionData: pd.DataFrame, debug=False):
        """
        :param trainingData: pandas.DataFrame X =['timeSlotStart', 'temperature', 't-N', 't-(N-1)', ...., 't-1','t'] for many days where Y = ['t']
        :param debug: True if you want prints. False by default
        :return X_new, y
        """
        self._trainingData = trainingData
        y = pd.DataFrame(self._trainingData["t"].tolist(), columns = ["t"])
        X = self._trainingData.drop(["t"], axis=1)
        print(type(X))
        print(type(y))
        # Create the RFE object and compute a cross-validated score. Here we use support machine regression
        svr = svm.SVR(kernel="linear")
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        selector = RFECV(estimator=svr, step=1, cv=KFold(),
                      scoring=scorer)
        #svc = svm.SVC(kernel="linear")

        #selector = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
         #scoring='accuracy')
        #print("------------------------------------")
        #print(selector)
        #print("------------------------------------")

        X_new = selector.fit(X, y.values.ravel())
        if debug:
        #------------------------------------------------------Debug prints---------------------------------------------
            print("Optimal number of features : %d" % selector.n_features_)
            print("grid scores: ----------------------------------------------------------------------------------------")
            print(selector.grid_scores_)
            print("ranking ----------------------------------------------------------------------------------------------")
            print(selector.ranking_)
            print("X ----------------------------------------------------------------------------------------------------")
            print(X)
            print("after transform -------------------------------------------------------------------------------------")
            print(selector.transform(X))
        #---------------------------------------------------------------------------------------------------------------

        return X_new, y

#-----------------------------------------------------------------------------------------------------------------------





#forecast algorithms ---------------------------------------------------------------------------------------------------
class ForecastAlgorithm(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        if type(self) == ForecastAlgorithm:
            raise TypeError(
                " ForecastALgorithm is an interface, it cannot be instantiated. Try with a concrete algorithm, e.g.: SVR")

    @abc.abstractmethod
    def forecast(self,X_train, y_train, X_predict):
        pass

class SVR(ForecastAlgorithm):
    def __init__(self):
        pass

    def forecast(self,X_train, y_train, X_predict):
        """
        :param
        :return:
        """
        #todo: continue
        return None

class MLPregression(ForecastAlgorithm):
    def __init__(self):
        pass

    def forecast(self,X_train, y_train, X_predict):
        """
        :param X_predict: pd.Dataframe
        :param Y_train: pd.Dataframe
        :param X_train: pd.Dataframe

        :param selector: sklearn.feature_selector.* -> selector that has been fit to the training data to select best features
        :return: list (or array) of predicted variables
        """
        print("X_train")
        print(X_train.shape)
        features = X_train
        print("features")
        print(features.shape)
        regr = MLPRegressor(random_state=1, max_iter=500).fit(features, y_train.values.ravel())
        print("X_predict_features")
        print(X_predict.shape)
        Y_predict = regr.predict(X_predict)
        return Y_predict
#-----------------------------------------------------------------------------------------------------------------------


class Predictor:
    def __init__(self, algorithm: ForecastAlgorithm, FSmethod: FeatureSelectionMethod):
        self._algorithm = algorithm
        self._FSmethod = FSmethod # feature selection method

    def REgenerationForecast(self, GHIdata: pd.DataFrame, SParea, SPefficiency):
        """"
        :param GHIdata: pandas.DataFrame ["PeriodEnd", "PeriodStart", "GHI"] for one day
        return: temp list of RE power generation forecast for one prosumer for one day
        """
        GHI = GHIdata["GHI"].tolist()
        REgenForecast = GHI*(SParea*SPefficiency)
        return REgenForecast

    def featureSelection(self, trainingData, predictionData, debug):
        """
        :param trainingData: pandas.DataFrame ["PeriodEnd, "PeriodStart", "powerLoad", "netLoad"]
        :return: sklearn.feature_selection.*
        """
        X_train, y_train, X_predict = self._FSmethod.featureSelection(trainingData, predictionData, debug = debug)
        return X_train, y_train, X_predict


    def loadForecast(self, X_train, y_train, X_predict):
        """

        :return
        """

        result = self._algorithm.forecast(X_train=X_train, y_train=y_train, X_predict=X_predict )
        return result


    def predict(self, trainingData, predictionData, debug = False):
        """
        :param trainingData: pd.Dataframe
        :param predictionData: pd.Dataframe
        :return: list (or array) of predicted variables
        """
        l = preprocessing.LabelEncoder()
        l.fit(trainingData["time"].tolist())
        trainingData["time"] = l.transform(trainingData["time"].tolist())
        predictionData["time"] = l.transform(predictionData["time"].tolist())
        X_train, y_train, X_predict = self.featureSelection(trainingData=trainingData, predictionData = predictionData, debug = debug)
        return self.loadForecast(X_train = X_train, y_train=y_train, X_predict=X_predict)



if __name__ == '__main__':

        trainingData1 = pd.read_csv("Forecasted/trainingData4")

       # trainingData2 = pd.read_csv("Forecasted/trainingData2")

        #trainingData3 = pd.read_csv("Forecasted/trainingData3")

        #trainingData4 = pd.read_csv("Forecasted/trainingData4")

        predictionData1 = pd.read_csv("Forecasted/predictionData4")

        #predictionData2 = pd.read_csv("Forecasted/predictionData2")

        #predictionData3 = pd.read_csv("Forecasted/predictionData3")

        #predictionData4 = pd.read_csv("Forecasted/predictionData4")

        files = ["30001480014107", "30001480282717", "50083502116836", "30001480640919"]

        prediction1 = Predictor(MLPregression(), UnivariateMethod()).predict(trainingData=trainingData1,predictionData = predictionData1, debug = True)
        #prediction2 = Predictor(MLPregression(), RFEmethod()).predict(trainingData=trainingData2,predictionData = predictionData2, deubg=True)


        #prediction3 = Predictor(MLPregression(), RFEmethod()).predict(trainingData=trainingData3,predictionData = predictionData3, debug = True)
        #prediction4 = Predictor(MLPregression(), RFEmethod()).predict(trainingData=trainingData4,predictionData = predictionData4, debug=True)
        #predictions = [prediction1, prediction2, prediction3, prediction4]

        #for i in range(len(files)):
        #    with open('Forecasted/' + files[i], 'w') as filehandle:
        #        for value in predictions[i]:
        #            filehandle.write('%s\n' % value)



