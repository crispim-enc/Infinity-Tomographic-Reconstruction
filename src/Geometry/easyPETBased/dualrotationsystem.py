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
        if detector_moduleA is None:
            raise ValueError("Detector module is not defined. Please provice a detectorModule")
        super().__init__()
        self._detectorModuleA = detector_moduleA
        self._detectorModuleB = detector_moduleB
        self._distanceBetweenMotors = 30
        self._distanceAxialMotorToDetectorModule = 30  # probably not needed
        self._distanceAxialMotorToRadioactiveSource = 30
        self._distanceFanMotorToDetectorModuleFaceA = 0
        self._distanceFanMotorToDetectorModuleFaceB = 60
        self._translationRadialModuleA = 15
        self._translationTangentialModuleA = 0
        self._translationAxialModuleA = 72.68/2
        self._originSystemWZ = np.array([0, 0, 0])

    def generateInitialCoordinates(self):
        """
        Generate the initial coordinates of the system
        """
        # Detector Module A
        self._detectorModuleA.generateInitialCoordinates()
        # Detector Module B
        self._detectorModuleB.generateInitialCoordinates()

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
        # rotation_matrix = np.array([[np.cos(angle, dtype=np.float32), -np.sin(angle, dtype=np.float32)],
        #                             [np.sin(angle, dtype=np.float32), np.cos(angle, dtype=np.float32)]],
        #                            dtype=np.float32)

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
