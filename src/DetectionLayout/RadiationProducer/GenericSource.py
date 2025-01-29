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


class GenericRadiativeSource:
    def __init__(self, device=None):
        if device is None:
            raise ValueError("Device cannot be None. Please provide a Device object.")
        self._device = device
        self._sourceName = "Am-241"
        self._sourceHalfLife = 432.2  # years
        self._sourceActivity = 1.0 * 37000
        self._focalSpotInitialPosition = [0, 0, 0]  # mm
        self._focalSpotDiameter = 2  # mm
        self._shieldingShape = "Cylinder"
        self._shieldingMaterial = "Lead"
        self._shieldingDensity = 11.34 # g/cm3
        self._shieldingThickness = 0.5
        self._shieldingHeight = 2
        self._shieldingRadius = 2
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
                        "focalSpot": self._focalSpot,
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
    def focalSpotInitialPosition(self):
        """
        Returns the focal spot.
        """
        return self._focalSpotInitialPosition

    def setFocalSpotInitialPosition(self, value):
        """
        Sets the focal spot. For easyCT geometries  the focal spot is set to the fan motor geometry
        param: list(), units in mm
        """
        if value != self._focalSpotInitialPosition:
            self._focalSpotInitialPosition = value

