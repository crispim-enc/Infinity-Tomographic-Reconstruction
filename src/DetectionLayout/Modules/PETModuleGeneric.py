import numpy as np
from src.DetectionLayout.Photodetectors.SiPM import HamamatsuS14161Series
from src.DetectionLayout.Photodetectors.Crystals import LYSOCrystal


class PETModule:
    """
    Class that represents a PET module. It contains the information about the module geometry and the detectors that compose it.
    Methods:
        rotateAndTranslateModule: Rotates and translates the module according to the given angles and translations.
    Attributes:
        moduleID (int): The module ID.
        numberVisibleLightSensorsX (int): The number of visible light sensors in the x axis.
        numberVisibleLightSensorsY (int): The number of visible light sensors in the y axis.
        numberHighEnergyLightDetectors (int): The number of high energy light detectors.
        modelVisibleLightSensors (str): The model of the visible light sensors.
        modelHighEnergyLightDetectors (str): The model of the high energy light detectors.
        visibleLightSensorObject (object): The object that represents the visible light sensors.

    """

    def __init__(self, module_id=None, ):

        if module_id is None:
            module_id = int(1)
        self._idModule = module_id
        self._numberVisibleLightSensorsX = int(1)
        self._numberVisibleLightSensorsY = int(1)
        self._numberHighEnergyLightDetectorsX = int(16)
        self._numberHighEnergyLightDetectorsY = int(16)
        self._numberHighEnergyLightDetectors = None
        self._totalNumberVisibleLightSensors = self._numberVisibleLightSensorsX * self._numberVisibleLightSensorsY
        self._totalNumberHighEnergyLightDetectors = self._numberHighEnergyLightDetectorsX * self._numberHighEnergyLightDetectorsY
        self._reflectorThicknessX = 0.1
        self._reflectorThicknessY = 0.1
        self._modelVisibleLightSensors = [HamamatsuS14161Series(i) for i in range(self._totalNumberVisibleLightSensors)]
        self._modelHighEnergyLightDetectors = [LYSOCrystal(i) for i in range(self._totalNumberHighEnergyLightDetectors)]
        self._visibleLightSensorObject = None
        self._highEnergyLightDetectorObject = None
        self._shiftXBetweenVisibleAndHighEnergy = 0
        self._shiftYBetweenVisibleAndHighEnergy = 5
        self._shiftZBetweenVisibleAndHighEnergy = 0
        self._alphaRotation = 0
        self._betaRotation = 0
        self._sigmaRotation = 0
        self._xTranslation = 0
        self._yTranslation = 0
        self._zTranslation = 0
        self._detectorsPosition = None
        self.setVisibleEnergyLightDetectorBlock(True)
        self.setHighEnergyLightDetectorBlock(True)

    def setInitialGeometry(self):
        """
        Sets the initial geometry of the module.

        """
        self.setHighEnergyLightDetectorBlock()
        self.setVisibleEnergyLightDetectorBlock()

    @property
    def highEnergyLightDetectorBlock(self):
        return self._modelHighEnergyLightDetectors

    def setVisibleEnergyLightDetectorBlock(self, shiftSiPM=False):
        for i in range(self._totalNumberVisibleLightSensors):

            center_to_rotate = self._modelVisibleLightSensors[i].centerSiPMModule
            if shiftSiPM:
                center_to_rotate[1] = (center_to_rotate[1] - self._modelHighEnergyLightDetectors[0].crystalSizeZ / 2 -
                                       self._shiftYBetweenVisibleAndHighEnergy)

            self._modelVisibleLightSensors[i].setCenterSiPMModule(center_to_rotate)
            self._modelVisibleLightSensors[i].setChannelOriginalCentrePosition()
            new_center = self.rotateAndTranslateModule(point=center_to_rotate, alpha=self._alphaRotation,
                                          beta = self._betaRotation,
                                          sigma = self._sigmaRotation, x=self._xTranslation,
                                          y = self._yTranslation,
                                          z = self._zTranslation)
            self._modelVisibleLightSensors[i].setCenterSiPMModule(new_center)
            centers_channels = self._modelVisibleLightSensors[i].channelCentrePosition
            for channel in range(len(self._modelVisibleLightSensors[i].channelCentrePosition)):
                channel_center = self._modelVisibleLightSensors[i].channelCentrePosition[channel]
                new_channel_center = self.rotateAndTranslateModule(point=channel_center, alpha=self._alphaRotation,
                                              beta = self._betaRotation,
                                              sigma=self._sigmaRotation, x=self._xTranslation,
                                              y=self._yTranslation,
                                              z=self._zTranslation)
                centers_channels[channel] = new_channel_center

            self._modelVisibleLightSensors[i].setChannelCentrePosition(centers_channels)
            self._modelVisibleLightSensors[i].setAlphaRotation(self._alphaRotation)
            self._modelVisibleLightSensors[i].setBetaRotation(self._betaRotation)
            self._modelVisibleLightSensors[i].setSigmaRotation(self._sigmaRotation)
            self._modelVisibleLightSensors[i].setXTranslation(self._xTranslation)
            self._modelVisibleLightSensors[i].setYTranslation(self._yTranslation)
            self._modelVisibleLightSensors[i].setZTranslation(self._zTranslation)

    def setHighEnergyLightDetectorBlock(self, block_creation=False):
        if block_creation:
            x_step = self._modelHighEnergyLightDetectors[0].crystalSizeX + self._reflectorThicknessX
            x_range = np.arange(0, self._numberHighEnergyLightDetectorsX * x_step, x_step) - (
                        self._numberHighEnergyLightDetectorsX - 1) * x_step / 2
            z_step = self._modelHighEnergyLightDetectors[0].crystalSizeY + self._reflectorThicknessY
            z_range = np.arange(0, self._numberHighEnergyLightDetectorsY * z_step, z_step) - (
                        self._numberHighEnergyLightDetectorsY - 1) * z_step / 2
            xx, zz = np.meshgrid(x_range, z_range)
            # xx, zz = np.meshgrid(np.arange(self._numberVisibleLightSensorsX) * (
            #             self._modelHighEnergyLightDetectors[0].crystalSizeX + self._reflectorThicknessX),
            # np.arange(self._numberVisibleLightSensorsY) * (
            #             self._modelHighEnergyLightDetectors[0].crystalSizeY + self._reflectorThicknessY))
            x_flat = xx.flatten()
            y_flat = np.zeros(self._numberHighEnergyLightDetectorsX * self._numberHighEnergyLightDetectorsY)
            z_flat = zz.flatten()
        for i in range(self._totalNumberHighEnergyLightDetectors):
            self._modelHighEnergyLightDetectors[i].setCrystalID(i + self._idModule *
                                                                self._totalNumberHighEnergyLightDetectors)
            # print("Crystal ID: ", self._modelHighEnergyLightDetectors[i].crystalID)
            if block_creation:
                center_to_rotate = np.array([x_flat[i], y_flat[i], z_flat[i]])
            else:
                center_to_rotate = self._modelHighEnergyLightDetectors[i].centroid
            # print("Center to rotate: ", center_to_rotate)
            # self._modelHighEnergyLightDetectors[i].setCentroid(center_to_rotate)
            new_center = self.rotateAndTranslateModule(point=center_to_rotate, alpha=self._alphaRotation,
                                                       beta=self._betaRotation,
                                                       sigma=self._sigmaRotation, x=self._xTranslation,
                                                       y=self._yTranslation,
                                                       z=self._zTranslation)
            # print("New center: ", new_center)
            self._modelHighEnergyLightDetectors[i].setCentroid(new_center)
            self._modelHighEnergyLightDetectors[i].setAlphaRotation(self._alphaRotation)
            self._modelHighEnergyLightDetectors[i].setBetaRotation(self._betaRotation)
            self._modelHighEnergyLightDetectors[i].setSigmaRotation(self._sigmaRotation)
            self._modelHighEnergyLightDetectors[i].setXTranslation(self._xTranslation)
            self._modelHighEnergyLightDetectors[i].setYTranslation(self._yTranslation)
            self._modelHighEnergyLightDetectors[i].setZTranslation(self._zTranslation)

            self._modelHighEnergyLightDetectors[i].setVerticesCrystalCoordinateSystem()

            # Rotate and translate the crystal vertices
            vertexes = self._modelHighEnergyLightDetectors[i].vertices
            for vertex in range(len(self._modelHighEnergyLightDetectors[i].vertices)):
                new_vertex = self.rotateAndTranslateModule(point=vertexes[vertex], alpha=self._alphaRotation,
                                                           beta=self._betaRotation, sigma=self._sigmaRotation,
                                                           x=self._xTranslation, y=self._yTranslation,
                                                           z=self._zTranslation)
                vertexes[vertex] = new_vertex
            self._modelHighEnergyLightDetectors[i].setVertices(vertexes)



    def rotateAndTranslateModule(self, point=None, alpha=0, beta=0, sigma=0, x=0, y=0, z=0, angunit="deg"):

        """
            Rotates and translates the initial matrix according to the given angles and translations.

          Args:
              alpha (float): The angle of rotation around the x axis.
              beta (float): The angle of rotation around the y axis.
              sigma (float): The angle of rotation around the z axis.
              x (float): The translation in the x axis.
              y (float): The translation in the y axis.
              z (float): The translation in the z axis.
              angunit (str): The unit of the angles. Default is "deg" for degrees.
                      """

        if angunit == "deg":
            alpha = np.deg2rad(alpha)
            beta = np.deg2rad(beta)
            sigma = np.deg2rad(sigma)

        A = np.array([[np.cos(sigma) * np.cos(beta),
                       -np.sin(sigma) * np.cos(alpha) + np.cos(sigma) * np.sin(beta) * np.sin(alpha),
                       np.sin(sigma) * np.sin(alpha) + np.cos(sigma) * np.sin(beta) * np.cos(alpha),
                       x],

                      [np.sin(sigma) * np.cos(beta),
                       np.cos(sigma) * np.cos(alpha) + np.sin(sigma) * np.sin(beta) * np.sin(alpha),
                       -np.cos(sigma) * np.sin(alpha) + np.sin(sigma) * np.sin(beta) * np.cos(alpha),
                       y],

                      [-np.sin(beta),
                       np.cos(beta) * np.sin(alpha),
                       np.cos(beta) * np.cos(alpha),
                       z],

                      [0,
                       0,
                       0,
                       1]], dtype=np.float32)


        B = np.ones(4)
        B[0:3] = point

        return np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2] + A[0, 3] * B[3],
                         A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2] + A[1, 3] * B[3],
                         A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2] + A[2, 3] * B[3]],
                        dtype=np.float32)

    @property
    def moduleID(self):
        """
        Returns the module ID.
        """
        return self._idModule

    def setModuleID(self, value: int):
        """
        Sets the module ID.
        """
        self._idModule = value

    @property
    def detectorsPosition(self):
        """
        Returns the position of the detectors.
        """
        return self._detectorsPosition

    @property
    def alphaRotation(self):
        """
        Returns the alpha rotation.
        """
        return self._alphaRotation

    def setAlphaRotation(self, value):
        """
        Sets the alpha rotation.
        """
        self._alphaRotation = value

    @property
    def betaRotation(self):
        """
        Returns the beta rotation.
        """
        return self._betaRotation

    def setBetaRotation(self, value):
        """
        Sets the beta rotation.
        """
        self._betaRotation = value

    @property
    def sigmaRotation(self):
        """
        Returns the sigma rotation.
        """
        return self._sigmaRotation

    def setSigmaRotation(self, value):
        """
        Sets the sigma rotation.
        """
        self._sigmaRotation = value

    @property
    def xTranslation(self):
        """
        Returns the x translation.
        """
        return self._xTranslation

    def setXTranslation(self, value):
        """
        Sets the x translation.
        """
        self._xTranslation = value

    @property
    def yTranslation(self):
        """
        Returns the y translation.
        """
        return self._yTranslation

    def setYTranslation(self, value):
        """
        Sets the y translation.
        """
        self._yTranslation = value

    @property
    def zTranslation(self):
        """
        Returns the z translation.
        """
        return self._zTranslation

    def setZTranslation(self, value):
        """
        Sets the z translation.
        """
        self._zTranslation = value

    @property
    def visibleLightSensorObject(self):
        """
        Returns the visible light sensor object.
        """
        return self._visibleLightSensorObject

    def updateVisibleLightSensorObject(self, value):
        """
        Updates the visible light sensor object.
        """
        if self._visibleLightSensorObject != value:
            self._visibleLightSensorObject = value

        return self._visibleLightSensorObject

    @property
    def numberVisibleLightSensorsX(self):
        """
        Number of visible light sensors in the X direction
        """
        return self._numberVisibleLightSensorsX

    def updateNumberVisibleLightSensorsX(self, value) -> int:
        """
        Update the number of visible light sensors in the X direction

        """
        if self._numberVisibleLightSensorsX != value:
            self._numberVisibleLightSensorsX = value

        return self._numberVisibleLightSensorsX

    @property
    def numberVisibleLightSensorsY(self):
        """
        Number of visible light sensors in the Y direction
        """
        return self._numberVisibleLightSensorsY

    def updateNumberVisibleLightSensorsY(self, value):
        """
        Update the number of visible light sensors in the Y direction
        """
        if self._numberVisibleLightSensorsY != value:
            self._numberVisibleLightSensorsY = value

        return self._numberVisibleLightSensorsY

    @property
    def numberHighEnergyLightDetectorsX(self):
        """
        Number of high energy light detectors in the X direction
        """
        return self._numberHighEnergyLightDetectors

    @property
    def numberHighEnergyLightDetectorsY(self):
        """
        Number of high energy light detectors in the Y direction
        """
        return self._numberHighEnergyLightDetectors

    def updateNumberHighEnergyLightDetectors(self, valueX, valueY):
        """
        Update the number of high energy light detectors in the X and Y direction
        """
        # if self._numberHighEnergyLightDetectorsX != value:
        self._numberHighEnergyLightDetectorsX = valueX
        self._numberHighEnergyLightDetectorsY = valueY

    @property
    def modelVisibleLightSensors(self):
        return self._modelVisibleLightSensors

    @property
    def modelHighEnergyLightDetectors(self):
        """
        Returns the model of the high energy light detectors.
        """
        return self._modelHighEnergyLightDetectors

    def setModelHighEnergyLightDetectors(self, value):
        """
        Sets the model of the high energy light detectors.
        """
        self._modelHighEnergyLightDetectors = value

    def setModelVisibleLightSensors(self, value):
        """
        Sets the model of the visible light sensors.
        Available options are:

        """
        self._modelVisibleLightSensors = value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from src.Designer import DeviceDesignerStandalone
    petModules = []
    for i in range(2):
        petModule = PETModule(i)
        petModules.append(petModule)
        petModule.setXTranslation(30*np.cos(np.deg2rad(45*i)))
        petModule.setYTranslation(30*np.sin(np.deg2rad(45*i)))
        # petModule.setYTranslation(30)
        # petModule.setbe(45*i)
        petModule.setSigmaRotation(90+45*i)
        petModule.setInitialGeometry()
        centers = [i.centroid for i in petModule.modelHighEnergyLightDetectors]
        centers = np.array(centers).T
        # plt.figure()
        # plt.plot(centers[0], centers[1], 'o')

        # plt.figure()
        # plt.plot(centers[0], centers[1], 'o')

        plt.plot(centers[1], centers[2], 'o')

    designer = DeviceDesignerStandalone(device=petModules[1])
    designer.addModule()
    # plt.show()

