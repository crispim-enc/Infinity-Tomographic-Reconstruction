#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: calibration_wrapper
# * AUTHOR: Pedro Encarnação
# * DATE: 29/04/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

class CalibrationWrapper:
    """
    This class is a wrapper for the calibration process.
    It contains the methods to calibrate the device and to create the TOR file.
    """
    def __init__(self):

        self._systemSensitivity = None

    @property
    def systemSensitivity(self):
        """
        Get the system sensitivity.
        :return: The system sensitivity.
        """
        return self._systemSensitivity

    def setSystemSensitivity(self, systemSensitivity):
        """
        Set the system sensitivity.
        :param systemSensitivity: system sensitivity
        :type systemSensitivity: SystemSensitivity
        """
        self._systemSensitivity = systemSensitivity
