#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: calibration
# * AUTHOR: Pedro Encarnação
# * DATE: 09/04/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

import numpy as np


class DetectorSensitivityResponse:
    """
    This class is used to calculate the detector sensitivity of a system.
    """

    def __init__(self, device=None, TORFile=None):
        """
        Initialize the DetectorSensitivity class.
        :param device: The device to be used for the calculation.
        :param file_path: The path to the file where the results will be saved.
        """
        self._device = device
        self._energyRegions = None
        self._torFile = TORFile
        self._detectorSensitivity = None

    @property
    def device(self):
        return self._device

    def setDevice(self, value):
        self._device = value







