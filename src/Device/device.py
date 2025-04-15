"""
    Author: "P. M. C. C. Encarnação"
    Email: "pedro.encarnacao@ua.pt"
    Date: "2023-10-10"
    Description: "This file is used to create a device object.
    This object is used to store the information about the device.

"""

#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# Methods:
# --------
# getDeviceUUID: returns the device UUID
# setDeviceUUID: sets the device UUID
# getDeviceName: returns the device name
# setDeviceName: sets the device name
# getDeviceType: returns the device type
# setDeviceType: sets the device type
# getDeviceStatus: returns the device status
# setDeviceStatus: sets the device status
# getDeviceDirectory: returns the device directory
# setDeviceDirectory: sets the device directory
#
# Properties:
# -----------
# deviceUUID: returns the device UUID
# deviceName: returns the device name
# deviceType: returns the device type
# deviceStatus: returns the device status
# deviceDirectory: returns the device directory

import os
import uuid
import json5


class Device:
    """
    Class that represents a device. It contains the information about the device.

    example:

    print("Device UUID: ", device.deviceUUID)

    """
    def __init__(self):
        self._deviceUUID = None
        self._deviceName = None
        self._deviceType = None
        self._deviceDirectory = None
        self._geometryObject = None
        self._geometryType = None
        self._energyResolutionFunction = None


    def readDeviceProperties(self, objectName=None):
        """
        Read the device properties from a file

        :param objectName: name of the object to read the properties from (folder name)
        :type objectName: str
        Deprecated
        """
        if objectName is None:
            print("Error: objectName is None")
            return

        # check if folder of device exists
        mainDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._deviceDirectory = os.path.join(mainDirectory, "configurations", self._deviceUUID)
        if not os.path.exists(self._deviceDirectory):
            print("Error: device directory does not exist")
            return

        # read text deviceID.txt
        file = open(os.path.join(self._deviceDirectory, "deviceID.txt"), "r")
        lines = file.readlines()
        for line in lines:
            if "deviceUUID" in line:
                self._deviceUUID = line.split(":")[1].strip()
            elif "deviceName" in line:
                self._deviceName = line.split(":")[1].strip()
            elif "deviceType" in line:
                self._deviceType = line.split(":")[1].strip()

    @property
    def deviceUUID(self):
        """ Returns the device UUID

             :return: deviceUUID
             """
        return self._deviceUUID

    def generateDeviceUUID(self):
        """
        Generate a device Universally Unique Identifier (UUID) according to version 4

        :return:
        """
        self._deviceUUID = str(uuid.uuid4())

    def setDeviceUUID(self, deviceUUID):
        """
        Sets the device UUID

        :param deviceUUID:

        :type deviceUUID: str

        """
        self._deviceUUID = deviceUUID

    @property
    def deviceName(self):
        return self._deviceName

    def getDeviceName(self):
        return self._deviceName

    def setDeviceName(self, deviceName):
        self._deviceName = deviceName

    @property
    def deviceType(self):
        return self._deviceType

    def setDeviceType(self, deviceType):
        self._deviceType = deviceType

    @property
    def deviceDirectory(self):
        return self._deviceDirectory

    @property
    def geometryType(self):
        return self._geometryType

    def setGeometryType(self, geometry_type):
        self._geometryType = geometry_type

    def createDirectory(self):
        """
        Create the directory for the device

        :return:
        """
        # create the directory if it does not exist in configurations with the id and name of the device
        mainDirectory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._deviceDirectory = os.path.join(mainDirectory, "configurations", self._deviceUUID + "_" + self._deviceName)
        if not os.path.exists(self._deviceDirectory):
            os.makedirs(self._deviceDirectory)

    @property
    def geometryObject(self):
        return self._geometryObject

    def setGeometryObject(self, geometryObject):
        """
        Set the geometry object (ex. SPECTGeometry object)

        :param geometryObject:
        :type geometryObject: object

        :return:
        """
        self._geometryObject = geometryObject

    @property
    def energyResolutionFunction(self):
        return self._energyResolutionFunction

    def setEnergyResolutionFunction(self, energyResolutionFunction):
        """
        """
        self._energyResolutionFunction = energyResolutionFunction

    def getFWHMSystemEnergyResponse(self, energy):
        if self._energyResolutionFunction is None:
            raise ValueError("No energy resolution function has been set.")
        return self._energyResolutionFunction.run(energy)

