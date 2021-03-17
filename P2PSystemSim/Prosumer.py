import matplotlib.pyplot as plt
from P2PSystemSim.Assets import *

class Prosumer:
    def __init__(self, id, loadForecast : list, REgeneration: list, battery : Battery = None, shiftableLoadMatrix=None):
        assert(len(loadForecast) == len(REgeneration))
        self._ID = id
        self.battery = None
        if battery is not None:
            self.set_battery(battery)
        self._loadForecast = loadForecast
        self._REgeneration = REgeneration
        self._shiftableLoadMatrix = shiftableLoadMatrix

    def set_battery(self, battery: Battery):
        self.battery = battery
        battery._set_owner(self)
    def _get_ID(self):
        return self._ID
    def _get_Battery(self)-> Battery:
        return self._battery
    def _get_loadForecast(self)-> list:
        return self._loadForecast
    def _get_REgeneration(self)-> list:
        return self._REgeneration
    def _get_shiftableLoadMatrix(self) -> list:
        return self._shiftableLoadMatrix
    def actionDR(self) -> bool:
        return random.choice([0, 1])
    def displayGraph(self):
        fig, ax = plt.subplots()
        ax.plot(range(0,len(self._REgeneration)), self._REgeneration, label="RE generation")
        ax.plot(range(0,len(self._loadForecast)), self._loadForecast, label="Load forecast")
        #ax.plot(len(self._REgeneration), self._REgeneration, len(self._loadForecast), self._loadForecast)
        ax.set(xlabel='timeslots', ylabel='Power (Wh)', title='Power consumption and production prosumer {}'.format(self._ID))
        ax.grid()
        #fig.savefig("test.png")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.show()