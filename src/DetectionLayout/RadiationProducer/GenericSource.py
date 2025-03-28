#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: GenericSource
# * AUTHOR: Pedro Encarnação
# * DATE: 28/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
This functions create a generic source object for example x-ray emission.
"""
import os
import json5
import numpy as np


class GenericRadiativeSource:
    def __init__(self, device=None):
        # if device is None:
        #     raise ValueError("Device cannot be None. Please provide a Device object.")
        self._device = device
        self._sourceName = "Am-241"
        self._sourceHalfLife = 432.2  # years
        self._sourceActivity = 1.0 * 37000
        self._focalSpotInitialPositionWKSystem = np.array([0, 0, 0], dtype=np.float32)  # mm
        self._focalSpotInitialPositionXYSystem = np.array([0, 0, 0], dtype=np.float32)
        self._focalSpotDiameter = 1  # mm
        self._shieldingShape = "Cylinder"
        self._shieldingMaterial = "Lead"
        self._shieldingDensity = 11.34 # g/cm3
        self._shieldingThickness = 0.5
        self._shieldingHeight = 3
        self._shieldingRadius = 1.25
        # self._gantryAngle = 30  # degrees
        self._mainEmissions = {1: {"energy": 59.54, "intensity": 0.36},
                               2: {"energy": 26.34, "intensity": 0.024},
                               }

    def writeSourceInformation(self):
        """
        Write the source information json file under the device directory.
        """
        file_name = os.path.join(self._device.deviceDirectory, "RadiativeSourceInformation.txt")
        # create a json file with the source information
        with open(file_name, "w") as file:
            json5.dump({"sourceName": self._sourceName,
                        "sourceActivity": self._sourceActivity,
                        "focalSpot": self._focalSpotInitialPositionXYSystem,
                        "focalSpotDiameter": self._focalSpotDiameter,
                        "shieldingShape": self._shieldingShape,
                        "shieldingMaterial": self._shieldingMaterial,
                        "shieldingDensity": self._shieldingDensity,
                        "shieldingThickness": self._shieldingThickness,
                        "shieldingHeight": self._shieldingHeight,
                        "shieldingRadius": self._shieldingRadius,
                        # "gantryAngle": self._gantryAngle,
                        "mainEmissions": self._mainEmissions
                        }, file)

    @property
    def sourceName(self):
        """
        Returns the source name.
        """
        return self._sourceName

    def setSourceName(self, value):
        """
        Sets the source name.
        """
        if value != self._sourceName:
            self._sourceName = value

    def setSourceActivity(self, value):
        """
        Sets the source activity.
        param: float(), units in Bq
        """
        if value != self._sourceActivity:
            self._sourceActivity = value

    @property
    def sourceActivity(self):
        """
        Returns the source activity.
        """
        return self._sourceActivity

    @property
    def focalSpotInitialPositionWKSystem(self):
        """
        Returns the focal spot.
        """
        return self._focalSpotInitialPositionWKSystem

    def setFocalSpotInitialPositionWKSystem(self, value):
        """
        Sets the focal spot. For easyCT geometries  the focal spot is set to the fan motor geometry
        param: list() or np.array(), units in mm
        """
        # if list convert to np.array
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)

            self._focalSpotInitialPositionWKSystem = value
        elif isinstance(value, np.ndarray):
            self._focalSpotInitialPositionWKSystem = value

        else:
            raise ValueError("Focal spot initial position must be a list or np.array.")

    @property
    def focalSpotInitialPositionXYSystem(self):
        """
        Returns the focal spot.
        """
        return self._focalSpotInitialPositionXYSystem

    def setFocalSpotInitialPositionXYSystem(self, value):
        """
        Sets the focal spot. For easyCT geometries  the focal spot is set to the fan motor geometry
        param: list() or np.array(), units in mm
        """
        # if list convert to np.array
        if isinstance(value, list):
            value = np.array(value, dtype=np.float32)

            self._focalSpotInitialPositionXYSystem = value
        elif isinstance(value, np.ndarray):
            self._focalSpotInitialPositionXYSystem = value
            print("Focal spot initial position set to: ", self._focalSpotInitialPositionXYSystem)

        else:
            raise ValueError("Focal spot initial position must be a list or np.array.")

    @property
    def shieldingShape(self):
        """
        Returns the shielding shape.
        """
        return self._shieldingShape

    def setShieldingShape(self, value):
        """
        Sets the shielding shape.
        """
        if value != self._shieldingShape:
            self._shieldingShape = value

    @property
    def shieldingMaterial(self):
        """
        Returns the shielding material.
        """
        return self._shieldingMaterial

    def setShieldingMaterial(self, value):
        """
        Sets the shielding material.
        """
        if value != self._shieldingMaterial:
            self._shieldingMaterial = value

    @property
    def shieldingDensity(self):
        """
        Returns the shielding density.
        """
        return self._shieldingDensity

    def setShieldingDensity(self, value):
        """
        Sets the shielding density.
        """
        if value != self._shieldingDensity:
            self._shieldingDensity = value

    @property
    def shieldingThickness(self):
        """
        Returns the shielding thickness.
        """
        return self._shieldingThickness

    def setShieldingThickness(self, value):
        """
        Sets the shielding thickness.
        """
        if value != self._shieldingThickness:
            self._shieldingThickness = value

    @property
    def shieldingHeight(self):
        """
        Returns the shielding height.
        """
        return self._shieldingHeight

    def setShieldingHeight(self, value):
        """
        Sets the shielding height.
        """
        if value != self._shieldingHeight:
            self._shieldingHeight = value

    @property
    def shieldingRadius(self):
        """
        Returns the shielding radius in mm.
        """
        return self._shieldingRadius

    def setShieldingRadius(self, value):
        """
        Sets the shielding radius.
        param: float(), units in mm
        """
        if value != self._shieldingRadius:
            self._shieldingRadius = value

    @property
    def mainEmissions(self):
        """
        Returns the main emissions.
        """
        return self._mainEmissions

    def setMainEmissions(self, value):
        """
        Sets the main emissions.
        """
        if value != self._mainEmissions:
            self._mainEmissions = value

    @property
    def focalSpotDiameter(self):
        """
        Returns the focal spot diameter.
        """
        return self._focalSpotDiameter

    def setFocalSpotDiameter(self, value):
        """
        Sets the focal spot diameter.
        """
        if value != self._focalSpotDiameter:
            self._focalSpotDiameter = value
