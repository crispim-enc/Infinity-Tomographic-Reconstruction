#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: calibration
# * AUTHOR: Pedro Encarnação
# * DATE: 09/04/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

import numpy as np
import matplotlib.pyplot as plt


class DetectorSensitivityResponse:
    """
    This class is used to calculate the detector sensitivity of a system.
    """

    def __init__(self, TORFile=None, use_detector_energy_resolution=True):
        """
        Initialize the DetectorSensitivity class.
        :param device: The device to be used for the calculation.
        :param file_path: The path to the file where the results will be saved.
        """
        self._useDetectorEnergyResolution = use_detector_energy_resolution
        self._energyWindows = None
        self._energyPeaks = None
        self._fields = None
        self._torFile = TORFile
        self._detectorSensitivity = None
        self._probabilityOfDetection = None

    @property
    def fields(self):
        """
        Get the fields.
        :return: The fields.
        """
        return self._fields

    def setFields(self):
        """
        Set the fields for the detector sensitivity.
        :param fields: The fields to be used for the calculation.
        """

        self._fields = [str(i) for i in self._energyPeaks]

    @property
    def energyPeaks(self):
        """
        Get the energy peaks.
        :return: The energy peaks.
        """
        return self._energyPeaks

    def setEnergyPeaks(self, energyPeaks=None):
        """
        Set the energy peaks for the detector sensitivity.
        :param energyPeaks: The energy peaks to be used for the calculation.
        """
        if not isinstance(energyPeaks, list) and not isinstance(energyPeaks, np.ndarray):
            raise ValueError("The energy peaks must be a list or numpy array.")
        self._energyPeaks = energyPeaks
        self.setFields()

    @property
    def energyWindows(self):
        """
        Get the energy windows.
        :return: The energy windows.
        """
        return self._energyWindows

    def setEnergyWindows(self, energyWindows=None):
        """
        Set the energy regions for the detector sensitivity.
        :param energyRegions: The energy regions to be used for the calculation.
        """
        if self._useDetectorEnergyResolution:
            # check if energyresolution function is defined
            # if not hasattr(self._torFile.systemInfo, "getEnergyResolution"):
            #     raise ValueError("The energy resolution function is not defined in the sensitivity file. Run the TOR file generator with a device with this function set.")
            energyWindows = np.zeros((len(self._energyPeaks), 2))
            low = self._energyPeaks - self._energyPeaks * self._torFile.systemInfo.getFWHMSystemEnergyResponse(self._energyPeaks)
            high = self._energyPeaks + self._energyPeaks * self._torFile.systemInfo.getFWHMSystemEnergyResponse(self._energyPeaks)
            energyWindows[:, 0] = np.clip(low, 0, None)
            energyWindows[:, 1] = np.clip(high, 0, None)
            energyWindows = np.round(energyWindows, 2)
            print("Energy windows: ", energyWindows)
        else:
            if not isinstance(energyWindows, list) or not isinstance(energyWindows, np.ndarray):
                raise ValueError("The energy regions must be a list or numpy array.")
        self._energyWindows = energyWindows

    def setDetectorSensitivity(self):
        numberOfEnergyCuts = len(self._energyPeaks)

        if self._torFile.systemInfo.deviceType == "PET":
            # "Not developed yet"
            numberOfModules = len(self._torFile.systemInfo.detectorModules)
            pass
        else:
            # plt.figure()
            column = self._torFile.fileBodyData.listmodeFields.index("IDB")
            self._probabilityOfDetection = np.zeros((len(self._energyPeaks), len(self._torFile.fileBodyData.uniqueValues[column])))
            for i in range(numberOfEnergyCuts):
                energyWindow = self._energyWindows[i]
                energyPeak = self._energyPeaks[i]
                # Get the number of events in the energy window
                #cut listmode
                energy_array = self._torFile.fileBodyData["ENERGYB"]
                indexes_active = np.where((energy_array >= energyWindow[0]) & (energy_array <= energyWindow[1]))[0]
                id_array = self._torFile.fileBodyData["IDB"][indexes_active]
                # get collumn of IDA

                probability = np.histogram(id_array, bins=len(self._torFile.fileBodyData.uniqueValues[column]), density=True)[0]
                self._probabilityOfDetection[i] = probability
            #     plt.plot(probability, label=f"Energy window {i + 1} ({energyWindow[0]}-{energyWindow[1]})")
            # plt.legend()
            # plt.show()

    @property
    def probabilityOfDetection(self):
        """
        Get the detector sensitivity.
        :return: The detector sensitivity.
        """
        return self._probabilityOfDetection

    def __str__(self):
        """
        Return the string representation of the class.
        :return: The string representation of the class.
        """
        return f"DetectorSensitivityResponse(energyPeaks={self._energyPeaks}, energyWindows={self._energyWindows})"

    def __repr__(self):
        """
        Return the string representation of the class.
        :return: The string representation of the class.
        """
        return self.__str__()

    def __getitem__(self, item):
        """
        Get the item from the class.
        :param item: The item to be get.
        :return: The item.
        """
        return getattr(self, item)

    def __iter__(self):
        """
        Return the iterator of the class.
        :return: The iterator of the class.
        """
        return iter(self.__dict__.items())

    def __len__(self):
        """
        Return the length of the class.
        :return: The length of the class.
        """
        return len(self._probabilityOfDetection)

