import abc
import random
from Battery import Battery

class DRevent:
    """this is meant to be an interface. However, there is no method in python to differentiate between abstract
    classes and interfaces so i'm explicitely not allowing direct instantiation of objects of class Devent to make sure
    it sticks to the java definition of an interface"""
    def __init__(self, *args, ** kargs):
        if type(self) == DRevent:
            raise TypeError(" DRevent is an interface, it cannot be instantiated. Try ConcreteDRevent")

    @abc.abstractmethod
    def execute(self) -> bool:
        pass


class ConcreteDRevent(DRevent):
    def __init__(self, DRrequestArray, prosumer):
        """
        :param DRrequestArray: array of reals that describes the DRevent
        :param prosumer: object of class Prosumer
        """
        self.prosumer = prosumer
        self.DRrequestArray = DRrequestArray

    def execute(self) -> bool:
        return self.prosumer.actionDR(self)


class DReventDispatcher:
    def __init__(self, cDRev):
        """
        :param cDRev: object of class ConcreteDRevent
        """
        self.concreteDRevent = cDRev

    def execute(self) -> bool:
        return self.concreteDRevent.prosumer.actionDR()




