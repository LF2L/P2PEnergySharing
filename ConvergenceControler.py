from abc import ABC, abstractmethod

class ConvergenceControlModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def control(self, **kwargs):
        pass

class StepLength(ConvergenceControlModel):
    def __init__(self, **kwargs):
        pass
        #todo: continue
    def control(self, **kwargs):
        pass
       #todo: complete

class NumberIterations(ConvergenceControlModel):
    def __init__(self, maxIterations, **kwargs):
        self.maxIterations = maxIterations
        self.nbIterations = 0

    def control(self, **kwargs):
        self.nbIterations = self.nbIterations+1
        if self.nbIterations>self.maxIterations:
            return False
        else:
            return True

class ConvergenceControler:
    def __init__(self, controlModel: ConvergenceControlModel):
        self.controlModel = controlModel

    def control(self, **kwargs)-> bool:
        return self.controlModel.control()
        #todo: complete
        return False