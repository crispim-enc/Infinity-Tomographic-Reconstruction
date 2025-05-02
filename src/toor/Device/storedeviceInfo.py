#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: createdevice
# * AUTHOR: Pedro Encarnação
# * DATE: 25/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************
import pickle
import os


class StoreDeviceInFo:
    def __init__(self, device_directory=None):
        self.device_directory = device_directory

    def createDeviceInDirectory(self, object):
        """
        Create a device
        :param device_directory: directory of the device
        :type device_directory: str
        :param object: object device
        :type object: Device
        """
        with open(os.path.join(self.device_directory,"DeviceInfo"), 'wb') as f:
            pickle.dump(object, f)

        print("Device created successfully")

    def readDeviceFromDirectory(self):
        """
        Read a device from a directory
        :param device_directory: directory of the device
        :type device_directory: str
        """
        with open(os.path.join(self.device_directory,"DeviceInfo"), 'rb') as f:
            device = pickle.load(f)
        return device

