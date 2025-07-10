# *******************************************************
# * FILE: genericCrystal.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np


class GenericCrystal:
    """
    Class that represents a LYSO crystal. It contains the information about the crystal geometry and the detectors that compose it.
    Methods:


    """
    def __init__(self, crystal_id=1):
        self._density = 7.4
        self._crystalID = crystal_id
        self._crystalSizeX = 20 # mm
        self._crystalSizeY = 1.6  # mm
        self._crystalSizeZ = 1.6
        self._centroid = [0, 0, 0]
        self._vertices = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]])

        self._originalVertices = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]])
        self._alphaRotation = 0
        self._betaRotation = 0
        self._sigmaRotation = 0
        self._xTranslation = 0
        self._yTranslation = 0
        self._zTranslation = 0
        self._volume = self._crystalSizeX * self._crystalSizeY * self._crystalSizeZ * 1e-3
        self._mass = self._density * self._volume

    def setCrystalID(self, value):
        """
        Sets the crystal ID.
        """
        if value != self._crystalID:
            self._crystalID = value

    @property
    def crystalID(self):
        """
        Returns the crystal ID.
        """
        return self._crystalID

    def setVerticesCrystalCoordinateSystem(self):
        """
        Sets the crystal vertices.
        """
        self._vertices = np.array([[self._centroid[0] - self._crystalSizeX / 2,
                                    self._centroid[1] - self._crystalSizeY / 2,
                                    self._centroid[2] - self._crystalSizeZ / 2],
                                   [self._centroid[0] + self._crystalSizeX / 2,
                                    self._centroid[1] - self._crystalSizeY / 2,
                                    self._centroid[2] - self._crystalSizeZ / 2],
                                   [self._centroid[0] + self._crystalSizeX / 2,
                                    self._centroid[1] + self._crystalSizeY / 2,
                                    self._centroid[2] - self._crystalSizeZ / 2],
                                   [self._centroid[0] - self._crystalSizeX / 2,
                                    self._centroid[1] + self._crystalSizeY / 2,
                                    self._centroid[2] - self._crystalSizeZ / 2],
                                   [self._centroid[0] - self._crystalSizeX / 2,
                                    self._centroid[1] - self._crystalSizeY / 2,
                                    self._centroid[2] + self._crystalSizeZ / 2],
                                   [self._centroid[0] + self._crystalSizeX / 2,
                                    self._centroid[1] - self._crystalSizeY / 2,
                                    self._centroid[2] + self._crystalSizeZ / 2],
                                   [self._centroid[0] + self._crystalSizeX / 2,
                                    self._centroid[1] + self._crystalSizeY / 2,
                                    self._centroid[2] + self._crystalSizeZ / 2],
                                   [self._centroid[0] - self._crystalSizeX / 2,
                                    self._centroid[1] + self._crystalSizeY / 2,
                                    self._centroid[2] + self._crystalSizeZ / 2]])

    @property
    def vertices(self):
        """
        Returns the crystal vertices.
        """
        return self._vertices

    def setVertices(self, value):
        """
        Sets the crystal vertices.
        """
        self._vertices = value

    @property
    def centroid(self):
        """
        Returns the crystal centroid.
        """
        return self._centroid

    def setCentroid(self, value):
        """
        Sets the crystal centroid.
        """

        self._centroid = value

    def setCristalSize(self, sizex, sizey, sizez):
        self._crystalSizeX = sizex
        self._crystalSizeY = sizey
        self._crystalSizeZ = sizez
        self._volume = self._crystalSizeX * self._crystalSizeY * self._crystalSizeZ * 1e-3
        self._mass = self._density * self._volume
        self.setVerticesCrystalCoordinateSystem()
        # self.setVertices()

    def getCrystalShape(self):
        return [self._crystalSizeX, self._crystalSizeY, self._crystalSizeZ]

    @property
    def crystalSizeX(self):
        return self._crystalSizeX

    @property
    def crystalSizeY(self):
        return self._crystalSizeY

    @property
    def crystalSizeZ(self):
        return self._crystalSizeZ

    @property
    def mass(self):
        return self._mass

    @property
    def density(self):
        return self._density

    def setDensity(self, value):
        if self._density != value:
            self._density = value
            self._mass = self._density * self._volume

    @property
    def volume(self):
        return self._volume

    @property
    def alphaRotation(self):
        return self._alphaRotation

    def setAlphaRotation(self, value):
        self._alphaRotation = value

    @property
    def betaRotation(self):
        return self._betaRotation

    def setBetaRotation(self, value):
        self._betaRotation = value

    @property
    def sigmaRotation(self):
        return self._sigmaRotation

    def setSigmaRotation(self, value):
        self._sigmaRotation = value

    @property
    def xTranslation(self):
        return self._xTranslation

    def setXTranslation(self, value):
        self._xTranslation = value

    @property
    def yTranslation(self):
        return self._yTranslation

    def setYTranslation(self, value):
        self._yTranslation = value

    @property
    def zTranslation(self):
        return self._zTranslation

    def setZTranslation(self, value):
        self._zTranslation = value
