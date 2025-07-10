# *******************************************************
# * FILE: collimatorGeneric.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np


class CollimatorGeneric(object):
    def __init__(self):
        """
        Collimator geometry
        Bore size: 1.28 x 1.28 mm
        Septa length: 2.3 cm
        """
        self._boreSize = [1.28, 1.28]
        self._septaWidth = 0.32
        self._septaLength = 23
        self._distanceToDetector = 0 #1.9
        self._yShiftCenterOfDetectorToCollimatorCenter = 0
        self._zShiftCenterOfDetectorToCollimatorCenter = 0
        self._collimatorDensity = None
        self._collimatorShape = "Square parallel-hole"
        self._collimatorMaterial = "Tungsten-alloy"
        if self._collimatorShape == "Square parallel-hole":
            self._areaSepta = self._boreSize[0] * self._boreSize[1]

        self._vertex1 = None
        self._vertex2 = None
        self._vertex3 = None
        self._vertex4 = None
        self._focalPoint = None
        self._centerOutsideFace = None
        self._initialMatrix = None

    def joinPyramidPointsIntoArray(self):
        """
        Join the points of the pyramid into a numpy array
        :return:
        """
        return np.array([self._focalPoint[0], self._focalPoint[1], self._focalPoint[2],
                         self._vertex1[0], self._vertex1[1], self._vertex1[2],
                         self._vertex2[0], self._vertex2[1], self._vertex2[2],
                         self._vertex3[0], self._vertex3[1], self._vertex3[2],
                         self._vertex4[0], self._vertex4[1], self._vertex4[2]], dtype=np.float32).T

    def calculateInitialPyramidalVertex(self, CztModule, alpha=0, beta=0, sigma=0, x=0, y=0, z=0, angunit="deg"):
        """
        Calculate the initial farest vertices of the collimator
        :param CztModule object
        :return:
        """
        distanceFromPoint = [-self._septaLength, self._boreSize[0] / 2, self._boreSize[1] / 2]
        shifts = [self._distanceToDetector, -self._yShiftCenterOfDetectorToCollimatorCenter,
                  -self._zShiftCenterOfDetectorToCollimatorCenter]
        self._vertex1 = CollimatorGeneric.vectorVertex(CztModule.initialMatrix, distanceFromPoint, shifts)

        distanceFromPoint = [-self._septaLength, -self._boreSize[0] / 2, self._boreSize[1] / 2]
        self._vertex2 = CollimatorGeneric.vectorVertex(CztModule.initialMatrix, distanceFromPoint, shifts)

        distanceFromPoint = [-self._septaLength, -self._boreSize[0] / 2, -self._boreSize[1] / 2]
        self._vertex3 = CollimatorGeneric.vectorVertex(CztModule.initialMatrix, distanceFromPoint, shifts)

        distanceFromPoint = [-self._septaLength, self._boreSize[0] / 2, -self._boreSize[1] / 2]
        self._vertex4 = CollimatorGeneric.vectorVertex(CztModule.initialMatrix, distanceFromPoint, shifts)

        self._focalPoint = CztModule.initialMatrix
        # self._focalPoint[0] -= 23/2

        self._vertex1 = self.rotateAndTranslate(self._vertex1, alpha=alpha, beta=beta, sigma=sigma, x=x, y=y, z=z, angunit=angunit)
        self._vertex2 = self.rotateAndTranslate(self._vertex2, alpha=alpha, beta=beta, sigma=sigma, x=x, y=y, z=z, angunit=angunit)
        self._vertex3 = self.rotateAndTranslate(self._vertex3, alpha=alpha, beta=beta, sigma=sigma, x=x, y=y, z=z, angunit=angunit)
        self._vertex4 = self.rotateAndTranslate(self._vertex4, alpha=alpha, beta=beta, sigma=sigma, x=x, y=y, z=z, angunit=angunit)
        self._focalPoint = self.rotateAndTranslate(self._focalPoint, alpha=alpha, beta=beta, sigma=sigma, x=x, y=y, z=z, angunit=angunit)

    # generate a function to calculate the intersection of two vectors
    def rotateAndTranslate(self, point, alpha=0, beta=0, sigma=0, x=0, y=0, z=0, angunit="deg"):
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
        # self.alphaRotation += alpha
        # self.betaRotation += beta
        # self.sigmaRotation += sigma
        # self.xTranslation += x
        # self.yTranslation += y
        # self.zTranslation += z
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

        B = np.ones((4, point.shape[1]))
        B[0:3] = point

        return np.array([A[0, 0] * B[0] + A[0, 1] * B[1] + A[0, 2] * B[2] + A[0, 3] * B[3],
                                        A[1, 0] * B[0] + A[1, 1] * B[1] + A[1, 2] * B[2] + A[1, 3] * B[3],
                                        A[2, 0] * B[0] + A[2, 1] * B[1] + A[2, 2] * B[2] + A[2, 3] * B[3]],
                                       dtype=np.float32)
    def calculateFocalPoint(self, CztModule):
        """
        Calculate the focal point of the collimator
        :param CztModule object
        :return:
        """
        # calculate the focal point
        # calculate the center of the outside face

        pass

    @staticmethod
    def vectorVertex(centerpoint, distanceFromPoint, shift):
        """
        Calculate the vector of the vertice

        :param centerpoint:
        :param distanceFromPoint:
        :param shift:
        :return:
        """
        point = np.zeros(centerpoint.shape)
        for i in range(3):
            point[i] = centerpoint[i] + distanceFromPoint[i] + shift[i]
        return point

    @property
    def focalPoint(self):
        """
        Return the focal point of the collimator
        :return:
        """
        return self._focalPoint

    @property
    def vertex1(self):
        """
        Return the vertice 1
        :return:
        """
        return self._vertex1

    @property
    def vertex2(self):
        """
        Return the vertice 2
        :return:
        """
        return self._vertex2

    @property
    def vertex3(self):
        """
        Return the vertice 3
        :return:
        """
        return self._vertex3

    @property
    def vertex4(self):
        """
        Return the vertice 4
        :return:
        """
        return self._vertex4
