#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: dualrotationsystem
# * AUTHOR: Pedro Encarnação
# * DATE: 29/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Brief description of the file.
"""
from src.Device import Device
import numpy as np


class DualRotationSystem(Device):
    def __init__(self, detector_moduleA=None, detector_moduleB=None):
        super().__init__()
        self._geometryType = "dualRotationSystemGeneric"
        self._detectorModuleObjA = detector_moduleA
        self._detectorModuleObjB = detector_moduleB
        self._distanceBetweenMotors = 30
        self._distanceFanMotorToDetectorModulesOnSideA = 0
        self._distanceFanMotorToDetectorModulesOnSideB = 60
        self._originSystemWZ = np.array([0, 0, 0])
        self._numberOfDetectorModulesSideA = 2
        self._numberOfDetectorModulesSideB = 2

        if self._detectorModuleObjA is not None:
            self._detectorModulesSideA = [self._detectorModuleObjA(i) for i in range(self._numberOfDetectorModulesSideA)]

        if self._detectorModuleObjB is not None:
            self._detectorModulesSideB = [self._detectorModuleObjB(i) for i in range(self._numberOfDetectorModulesSideB)]

    def generateInitialCoordinates(self):
        """
        Generate the initial coordinates of the system
        """
        if len(self._detectorModulesSideA) != 0:
        # Detector Module A
            for i in range(self._numberOfDetectorModulesSideA):
                self._detectorModulesSideA[i].setInitialGeometry()
        try:
            if len(self._detectorModulesSideB) != 0:
                # Detector Module B
                for i in range(self._numberOfDetectorModulesSideB):
                    self._detectorModulesSideB[i].setInitialGeometry()
        except AttributeError:
            pass

    def detectorSideACoordinatesAfterMovement(self, axialMotorAngle, fanMotorAngle, uniqueIdDetectorheaderA=None):
        """
        Load the list mode data np.array
        """

    def detectorSideBCoordinatesAfterMovement(self, axialMotorAngle, fanMotorAngle, uniqueIdDetectorheaderB=None):
        """

        """

    @staticmethod
    def _rotatePoint(angle, initial_point):
        cos_angle = np.cos(angle, dtype=np.float32)
        sin_angle = np.sin(angle, dtype=np.float32)
        rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]],
                                   dtype=np.float32)

        return np.array([rotation_matrix[0, 0] * initial_point[0] + rotation_matrix[0, 1] * initial_point[1],
                         rotation_matrix[1, 0] * initial_point[0] + rotation_matrix[1, 1] * initial_point[1]],
                        dtype=np.float32)

    @staticmethod
    def applyDualRotation(angle1, radius1, angle2, radius2, ZCoordinate):
        B = np.array([np.cos(angle1) * radius1,
                      np.sin(angle1) * radius1,
                      np.ones(len(angle2))], dtype=np.float32)

        A00 = np.cos(angle2).astype(np.float32)
        A01 = np.sin(angle2).astype(np.float32)

        x_corner = A00 * B[0] - A01 * B[1] + radius2 * A00 * B[2]
        y_corner = A01 * B[0] + A00 * B[1] + radius2 * A01 * B[2]
        z_corner = ZCoordinate

        return np.array([x_corner, y_corner, z_corner], dtype=np.float32).T

    @property
    def originSystemWZ(self):
        return self._originSystemWZ

    @property
    def detectorModulesSideA(self):
        return self._detectorModulesSideA

    @property
    def numberOfDetectorModulesSideA(self):
        return self._numberOfDetectorModulesSideA

    def setNumberOfDetectorModulesSideA(self, value):
        self._numberOfDetectorModulesSideA = value

        if self._detectorModuleObjA is not None:
            self._detectorModulesSideA = [self._detectorModuleObjA(i) for i in range(self._numberOfDetectorModulesSideA)]
        else:
            raise AttributeError("Detector Module A is not defined")

    @property
    def detectorModulesSideB(self):
        return self._detectorModulesSideB

    @property
    def numberOfDetectorModulesSideB(self):
        return self._numberOfDetectorModulesSideB

    def setNumberOfDetectorModulesSideB(self, value):
        self._numberOfDetectorModulesSideB = value

        if self._detectorModuleObjB is not None:
            self._detectorModulesSideB = [self._detectorModuleObjB(i) for i in range(self._numberOfDetectorModulesSideB)]
        else:
            raise AttributeError("Detector Module B is not defined")
