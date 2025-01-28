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


class GenericRadiativeSource:
    def __init__(self, device=None):
        if device is None:
            raise ValueError("Device cannot be None. Please provide a Device object.")
        self._device = device
        self._sourceName = "Am-241"
        self._sourceHalfLife = 432.2  # years
        self._sourceActivity = 1.0 * 37000
        self._sourceActivityUnit = "Bq"
        self._focalSpot = [0, 0, 0] # mm
        self._focalSpotDiameter = 2 # mm
        self._shieldingShape = "Cylinder"
        self._shieldingMaterial = "Lead"
        self._shieldingDensity = 11.34
        self._shieldingThickness = 0.5
        self._shieldingHeight = 2
        self._shieldingRadius = 2
        self._gantryAngle = 30 # degrees
        self._mainEmissions = {1: {"energy": 59.54, "intensity": 0.36},
                                2: {"energy": 26.34, "intensity": 0.024},
                            }

    def writeSourceInformation(self):
        """
        Write the source information to a file
        :param file_name: name of the file
        :type file_name: str
        """
        file_name = os.path.join(self._device.deviceDirectory, "RadiativeSourceInformation.txt")
        with open(file_name, "w") as file:
            file.write("Radioactive Source Name: {}\n".format(self._sourceName))
            file.write("Radioative  Source Activity: {} {}\n".format(self._sourceActivity, self._sourceActivityUnit))
            file.write("Focal Spot: {} mm\n".format(self._focalSpotDiameter))
            file.write("Shielding Shape: {}\n".format(self._shieldingShape))
            file.write("Shielding Material: {}\n".format(self._shieldingMaterial))
            file.write("Shielding Density: {} g/cm3\n".format(self._shieldingDensity))
            file.write("Shielding Thickness: {} mm\n".format(self._shieldingThickness))
            file.write("Shielding Height: {} mm\n".format(self._shieldingHeight))
            file.write("Shielding Radius: {} mm\n".format(self._shieldingRadius))
            file.write("Gantry Angle: {} degrees\n".format(self._gantryAngle))
            file.write("Main Emissions:\n")
            for key, value in self._mainEmissions.items():
                file.write("Energy: {} keV, Intensity: {}\n".format(value["energy"], value["intensity"]))


    def readSourceInformation(self):
        """
        Read the source information from a file
        :param file_name: name of the file
        :type file_name: str
        """
        file_name = os.path.join(self._device.deviceDirectory, "RadiativeSourceInformation.txt")
        with open(file_name, "r") as file:
            for line in file:
                print(line)


    @property
    def sourceName(self):
        """
        Returns the source name.
        """
        return self._sourceName


    def setSourceActivity(self, value):
        """
        Sets the source activity.
        """
        if value != self._sourceActivity:
            self._sourceActivity = value

