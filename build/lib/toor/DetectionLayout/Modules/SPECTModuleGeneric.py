import os


class SPECTHeadGeneric(object):

    def __init__(self, moduleObject, collimatorObject):
        """
        Initialize the head geometry
        :param numberHead: number of the head
        """

        self._numberOfModules = 5
        self._numberHead = 0
        self._detectionModule = [moduleObject() for i in range(self._numberOfModules)]
        self._collimators = [collimatorObject() for i in range(self._numberOfModules)]
        self._spacingCZTtoCollimator = 1.9  # mm
        self.alphaRotation = 0
        self.betaRotation = 0
        self.sigmaRotation = 0
        self.xTranslation = 0
        self.yTranslation = 0
        self.zTranslation = 0
        self._projectorHeadPointsMap = None

    def calculateInitialGeometry(self, alpha=0, beta=0, sigma=0, x=0, y=0, z=0):
        """
        Calculate the initial geometry of the head
        :param alpha: rotation around the x axis
        :param beta: rotation around the y axis
        :param sigma: Rotation around z axis
        :param x: translation along x axis
        :param y: translation along y axis
        :param z: translation along z axis
        :return:
        """
        self.alphaRotation = alpha
        self.betaRotation = beta
        self.sigmaRotation = sigma
        self.xTranslation = x
        self.yTranslation = y
        self.zTranslation = z
        # import matplotlib.pyplot as plt
        for i in range(self._numberOfModules):
            self._detectionModule[i].updateModuleNumber(i)
            self._detectionModule[i].calculateInitialMatrix()
            # self._CZTModules[i].rotateAndTranslate(0, 0, 0,x =self.xTranslation, y=self.yTranslation)
            self._detectionModule[i].rotateAndTranslate(0, 0, 0,
                                                        z=self._detectionModule[i].detectorSizeX * (i - (self._numberOfModules - 1) / 2))

            self._collimators[i].calculateInitialPyramidalVertex(self._detectionModule[i], alpha=alpha, beta=beta, sigma=sigma,
                                                                 x=x, y=y, z=z, angunit="rad")

            self._detectionModule[i].rotateAndTranslate(alpha, beta, sigma,
                                                        x=x,
                                                        y=y, z=z, angunit="rad")

    @property
    def CZTModules(self):
        """
        Return the list of objects of the CZT modules
        :return:
        """
        return self._detectionModule

    @property
    def numberOfModules(self):
        """
        Return the number of modules
        :return:
        """
        return self._numberOfModules

    @property
    def collimators(self):
        """
        Return the list of objects of the collimators
        :return:
        """
        return self._collimators

    def setNumberHead(self, value: int):
        """
        Set the number of the head
        :param value:
        :return:
        """
        if value != self._numberHead:
            self._numberHead = value

    @property
    def numberHead(self):
        """
        Return the number of the head
        :return:
        """
        return self._numberHead

    @property
    def projectorHeadPointsMap(self):
        """
        Return the points of the projector head
        :return:
        """
        return self._projectorHeadPointsMap

    def setProjectorHeadPointsMap(self, points):
        """
        Set the points of the projector head
        :param points:
        :return:
        """

        self._projectorHeadPointsMap = points

    def saveVarsHead(self, deviceDirectory):

        """
        Save the variables of the head to a txt file
        :return:

        """
        # save the variables to a file for each head

        file = open(os.path.join(deviceDirectory, "headVars{}.txt".format(self._numberHead)), "w")
        file.write("Head Number: " + str(self._numberHead) + "\n")
        file.write("Number of Modules: " + str(self._numberOfModules) + "\n")
        file.write("Spacing CZT to Collimator: " + str(self._spacingCZTtoCollimator) + "\n")
        file.write("Alpha Rotation: " + str(self.alphaRotation) + "\n")
        file.write("Beta Rotation: " + str(self.betaRotation) + "\n")
        file.write("Sigma Rotation: " + str(self.sigmaRotation) + "\n")
        file.write("X Translation: " + str(self.xTranslation) + "\n")
        file.write("Y Translation: " + str(self.yTranslation) + "\n")
        file.write("Z Translation: " + str(self.zTranslation) + "\n")
        file.close()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Designer import GeometryDesignerObject
    spectGeometry = SPECTHeadGeneric()
    spectGeometry.calculateInitialGeometry()
    gd = GeometryDesignerObject()
    for i in range(spectGeometry.numberOfModules):
        Czt1 = spectGeometry.CZTModules[i]
        plt.plot(Czt1.initialMatrix[0], Czt1.initialMatrix[1], ".")

        gd.drawDetectors(geometryVector=[Czt1.initialMatrix[0], Czt1.initialMatrix[1], Czt1.initialMatrix[2]],
                         moduleNumber=Czt1.moduleNumber)

    gd.renderWin.Render()
    gd.renderInteractor.Start()
    plt.show()
