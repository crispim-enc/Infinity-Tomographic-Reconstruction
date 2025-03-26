#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: SPIRALModule
# * AUTHOR: Pedro Encarnação
# * DATE: 28/02/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************


__version__ = "1.0.0"

from src.DetectionLayout.Photodetectors.SiPM import HamamatsuS14161Series
from src.DetectionLayout.Photodetectors.Crystals import LYSO, GenericCrystal
from src.DetectionLayout.Modules import PETModule


class SPIRALModule2025A(PETModule):
    def __init__(self):
        """
        SPIRAL Module 2025A
        """
        super(SPIRALModule2025A, self).__init__()
        self._numberOfLayers = 4
        self._numberVisibleLightSensorsX = 16
        self._numberVisibleLightSensorsY = 1
        self._totalNumberVisibleLightSensors = ((self._numberVisibleLightSensorsX * self._numberVisibleLightSensorsY) *
                                                self._numberOfLayers)

        self._numberVisibleHighEnergyLightDetectorsX = 16
        self._numberVisibleHighEnergyLightDetectorsY = 1
        self._totalNumberHighEnergyLightDetectors = ((self._numberVisibleHighEnergyLightDetectorsX *
                                                            self._numberVisibleHighEnergyLightDetectorsY) *
                                                            self._numberOfLayers)

    def setDefaultModelHighEnergyLightDetectors(self,width=1.5,height=1.5,depth=1.5):
        """
        Set the models of the high energy light detectors.
        """
        self._modelHighEnergyLightDetectors = []
        for i in range(self._numberOfLayers):
            for j in range(self._numberVisibleHighEnergyLightDetectorsX):
                for k in range(self._numberVisibleHighEnergyLightDetectorsY):
                    index = i * self._numberVisibleHighEnergyLightDetectorsX * self._numberVisibleHighEnergyLightDetectorsY + \
                            j * self._numberVisibleHighEnergyLightDetectorsY + k

                    self._modelHighEnergyLightDetectors.append(GenericCrystal(index))

                    depth = width*j+ width
                    self._modelHighEnergyLightDetectors[index].setCristalSize()
        self._modelHighEnergyLightDetectors = GenericCrystal(0)




    def setVisibleEnergyLightDetectorBlock(self):
        """
        Set the visible energy light detector block.
        """
        self._visibleEnergyLightDetectorBlock = HamamatsuS14161Series()


    def setHighEnergyLightDetectorBlock(self):

        """
        Set the high energy light detector block.
        """
        self._highEnergyLightDetectorBlock = LYSO()



if __name__ == "__main__":
    spiralModule = SPIRALModule2025A()
    print(spiralModule)