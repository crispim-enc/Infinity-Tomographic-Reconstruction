#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: easyPETBased
# * AUTHOR: Pedro Encarnação
# * DATE: 28/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Brief description of the file.
"""
import numpy as np
from .PETModuleGeneric import PETModule
from src.DetectionLayout.Photodetectors.Crystals import LYSOCrystal


class easyPETModule(PETModule):
    """
    Class that represents a easyPETBased module. It contains the information about the module geometry and the detectors that compose it.
    Methods:

    """
    def __init__(self, module_id=1, model="easyPET.3D_32"):
        super(easyPETModule, self).__init__()
        self._idModule = module_id
        if model == "easyPET.3D_32":
            self.model32()

    def model32(self):
        """
        Create the easyPETBased module with 32 detectors.
        """
        self._numberVisibleLightSensorsX = int(1)
        self._numberVisibleLightSensorsY = int(1)
        self._numberHighEnergyLightDetectorsX = int(2)
        self._numberHighEnergyLightDetectorsY = int(16)
        self._numberHighEnergyLightDetectors = None
        self._totalNumberVisibleLightSensors = self._numberVisibleLightSensorsX * self._numberVisibleLightSensorsY
        self._totalNumberHighEnergyLightDetectors = self._numberHighEnergyLightDetectorsX * self._numberHighEnergyLightDetectorsY
        self._reflectorThicknessX = 0.28
        self._reflectorThicknessY = 0.35
        self._modelVisibleLightSensors = [HamamatsuS14161Series(i) for i in range(self._totalNumberVisibleLightSensors)]

        self._shiftXBetweenVisibleAndHighEnergy = 0
        self._shiftYBetweenVisibleAndHighEnergy = 0
        self._shiftZBetweenVisibleAndHighEnergy = 0
        self._modelHighEnergyLightDetectors = [LYSOCrystal(i) for i in range(self._totalNumberHighEnergyLightDetectors)]
        for i in range(self._totalNumberHighEnergyLightDetectors):
            self._modelHighEnergyLightDetectors[i].setCrystalID(i)
            self._modelHighEnergyLightDetectors[i].setCristalSize(2, 2, 30)
