import numpy as np


class CZTModule(object):
    def __init__(self):
        """
            This class defines the geometry of one CZT Module in its coordinating system.

            Attributes:
                _origin (np.ndarray): The origin of the coordinating system, represented as a numpy array of three floats.
                _moduleNumber (int): The number of the CZT module.
                _numberOfPixels (np.ndarray): The number of pixels in the module, represented as a numpy array of two integers.
                _pixelSize (np.ndarray): The size of the pixels in meters, represented as a numpy array of two floats.
                _pixelSizeFronteir (np.ndarray): The size of the pixels in the frontiers in meters, represented as a numpy array of two floats.
                _pixelSpacingX (float): The spacing between the pixels in the x axis.
                _pixelSpacingY (float): The spacing between the pixels in the y axis.
                _edgeSpacingX (float): The spacing between the edges of the detector and the pixels in the x axis.
                _edgeSpacingY (float): The spacing between the edges of the detector and the pixels in the y axis.
                _detectorSizeX (float): The size of the detector in the x axis.
                _detectorSizeY (float): The size of the detector in the y axis.
                _cztThickness (float): The thickness of the CZT detector.
                _initialMatrix (np.ndarray): A matrix representing the initial coordinates of the pixels in the module.
                alphaRotation (float): The rotation angle around the x axis.
                betaRotation (float): The rotation angle around the y axis.
                sigmaRotation (float): The rotation angle around the z axis.
                xTranslation (float): The translation distance in the x axis.
                yTranslation (float): The translation distance in the y axis.
                zTranslation (float): The translation distance in the z axis.
            """
        self._origin = np.array([0, 0, 0], dtype=np.float32) # the origin of the module
        self._moduleNumber = int(1) # the number of the module
        self._numberOfPixels = np.array([16, 16], dtype=int) # the number of pixels in the module
        self._pixelSize = np.array([1.5, 1.5], dtype=np.float32)  # the size of the pixels in milimeters
        self._pixelSizeFronteir = np.array([1.3, 1.3], dtype=np.float32)  # the size of the pixels in the frontiers in milimeters
        self._pixelSpacingX = float(0.1)  # the spacing between the pixels in the x axis
        self._pixelSpacingY = float(0.1)  # the spacing between the pixels in the y axis
        self._edgeSpacingX = float(0.15)  # the spacing between the edges of the detectors modules
        self._edgeSpacingY = float(0.15)    # the spacing between the edges of the detectors modules
        self._detectorSizeX = (self.numberOfPixels[0] - 2) * self.pixelSize[0] + (
                    self.numberOfPixels[0] - 1) * self.pixelSpacingX + \
                              2 * (self._edgeSpacingX + self._pixelSizeFronteir[0])

        # self._detectorSizeX = 1.6 * 16

        self._detectorSizeY = (self.numberOfPixels[1] - 2) * self.pixelSize[1] + (
                    self.numberOfPixels[1] - 1) * self.pixelSpacingX + \
                              2 * (self._edgeSpacingX + self._pixelSizeFronteir[1])
        self._cztThickness = float(5)  # mm
        self._initialMatrix = None
        self.alphaRotation = 0
        self.betaRotation = 0
        self.sigmaRotation = 0
        self.xTranslation = 0
        self.yTranslation = 0
        self.zTranslation = 0

    def calculateInitialMatrix(self):
        """
             Calculates the initial matrix representing the coordinates of the pixels in the module.
             The matrix is stored in the `initialMatrix` attribute.

             Returns:
                 np.ndarray: The initial matrix of the CZT module.
             """
        x_step = (self._pixelSize[0] + self._pixelSpacingX) # the step in the x axis
        y_step = (self._pixelSize[1] + self._pixelSpacingY) # the step in the y axis
        y = np.arange(0, self._numberOfPixels[0] * x_step, x_step) - (self._numberOfPixels[0] - 1) * x_step / 2
        # corrections for the fronteirs
        y[0] += np.abs(self.pixelSize[0] - self._pixelSizeFronteir[0]) / 2
        y[-1] -= np.abs(self.pixelSize[0] - self._pixelSizeFronteir[0]) / 2

        z = np.arange(0, self._numberOfPixels[1] * y_step, y_step) - (self._numberOfPixels[1] - 1) * y_step / 2
        # corrections for the fronteirs
        z[0] += np.abs(self.pixelSize[1] - self._pixelSizeFronteir[1]) / 2
        z[-1] -= np.abs(self.pixelSize[1] - self._pixelSizeFronteir[1]) / 2
        yy, zz = np.meshgrid(y, z)
        # zz, yy = np.meshgrid(z, y)
        y = np.reshape(yy, yy.shape[0] * yy.shape[1])
        x = np.zeros(y.shape)
        z = np.reshape(zz, zz.shape[0] * zz.shape[1])
        self._initialMatrix = np.array([x, y, z], dtype=np.float32)
        return self._initialMatrix

    @property
    def initialMatrix(self):
        return self._initialMatrix

    def rotateAndTranslate(self, alpha=0, beta=0, sigma=0, x=0, y=0, z=0, angunit="deg"):
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
        self.alphaRotation += alpha
        self.betaRotation += beta
        self.sigmaRotation += sigma
        self.xTranslation += x
        self.yTranslation += y
        self.zTranslation += z
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

        B = np.ones((4, self._initialMatrix.shape[1]))
        B[0:3] = self._initialMatrix

        self._initialMatrix = np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2] + A[0, 3] * B[3],
                                        A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2] + A[1, 3] * B[3],
                                        A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2] + A[2, 3] * B[3]],
                                       dtype=np.float32)

    @property
    def origin(self):
        return self._origin

    def updateOrigin(self, value):
        if self._origin != value:
            if isinstance(value, list):
                value = np.array(value)
            else:
                raise TypeError
            self._origin = value
        return self._origin

    @property
    def moduleNumber(self):
        return self._moduleNumber

    def updateModuleNumber(self, value):
        if self._moduleNumber != value:
            self._moduleNumber = value

        return self._moduleNumber

    @property
    def numberOfPixels(self):
        return self._numberOfPixels

    def updateNumberOfPixels(self, xValue, yValue):
        arr = np.array([xValue, yValue])
        if self._numberOfPixels != arr:
            self._numberOfPixels = arr

        return self._numberOfPixels

    @property
    def pixelSize(self):
        return self._pixelSize

    def updatePixelSize(self, width, height):
        arr = np.array([width, height])
        if self._pixelSize != arr:
            self._pixelSize = arr

        return self._pixelSize

    @property
    def pixelSpacingX(self):
        return self._pixelSpacingX

    def updatepixelSpacingX(self, value):
        if self._pixelSpacingX != value:
            self._pixelSpacingX = value

        return self._pixelSpacingX

    @property
    def pixelSpacingY(self):
        return self._pixelSpacingY

    def updatepixelSpacingY(self, value):
        if self._pixelSpacingY != value:
            self._pixelSpacingY = value

        return self._pixelSpacingX

    @property
    def cztThickness(self):
        return self._cztThickness

    def updatecztThickness(self, value):
        if self._cztThickness != value:
            self._cztThickness = value

        return self._cztThickness

    @property
    def detectorSizeX(self):
        return self._detectorSizeX

    def updateDetectorSizeX(self, value):
        if self._detectorSizeX != value:
            self._detectorSizeX = value

        return self._detectorSizeX

    @property
    def detectorSizeY(self):
        return self._detectorSizeY

    def updateDetectorSizeY(self, value):
        if self._detectorSizeY != value:
            self._detectorSizeY = value

        return self._detectorSizeY
