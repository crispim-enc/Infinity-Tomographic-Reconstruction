# *******************************************************
# * FILE: torfiles.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import numpy as np
import pickle


class ToRFile:
    def __init__(self, filepath=None, file_with_calibrations=True):
        if filepath is None:
            raise FileNotFoundError("Filepath is required")

        self._systemInfoSize = None
        self._filePath = filepath
        self._version = "1.0"
        self._acquisitionInfo = None
        self._systemInfo = None
        self._fileBodyData = None
        self._calibrations = None
        self._fileWithCalibrationData = file_with_calibrations

    def read(self, filePath=None):
        print("Reading file: {}".format(filePath) if filePath is not None else "Reading file: {}".format(self._filePath))
        if filePath is None:
            filePath = self._filePath
        with open(filePath, 'rb') as input_file:
            self._systemInfo = pickle.load(input_file)
            self._acquisitionInfo = pickle.load(input_file)
            if self._fileWithCalibrationData:
                try:
                    self._calibrations = pickle.load(input_file)
                except EOFError:
                    print("No calibration data found in file")
                    self._calibrations = None

            else:
                self._calibrations = None
                # read the calibration data

            self._fileBodyData = pickle.load(input_file)

        input_file.close()

    def write(self):
        # records = np.rec.fromarrays((self._fileBodyData), names=self._listModeFields)
        with open(self._filePath, 'wb') as output_file:
            # size_acquisition_info = np.fromfile(output_file, dtype=np.int32, count=1)
            # self._listModeFields.tofile(output_file)
            pickle.dump(self._systemInfo, output_file)
            pickle.dump(self._acquisitionInfo, output_file)
            if self._fileWithCalibrationData:
                pickle.dump(self._calibrations, output_file)
            else:
                self._calibrations = None



            # self._fileBodyData.tofile(output_file, dtype=[('energyA', np.float64), ('energyB', np.float64),
            #                                                 ('IDA', np.float64), ('IDB', np.float64),
            #                                                 ('AXIAL_MOTOR', np.float64), ('FAN_MOTOR', np.float64),
            #                                                 ('TIME', np.float64)])
            pickle.dump(self._fileBodyData, output_file)



            # self._version.tofile(output_file)
            # self._sizeHeader.tofile(output_file)
            # self._acquisitionInfoSize.tofile(output_file)
            # self._systemInfoSize.tofile(output_file)
            # self._acquisitionInfo.tofile(output_file)
            # self._systemInfo.tofile(output_file)
        output_file.close()

    @property
    def version(self):
        return self._version

    @property
    def systemInfo(self):
        return self._systemInfo

    @property
    def fileBodyData(self):
        return self._fileBodyData

    @property
    def acquisitionInfo(self):
        return self._acquisitionInfo

    def setAcquisitionInfo(self, acquisitionInfo):
        self._acquisitionInfo = acquisitionInfo

    @property
    def calibrations(self):
        return self._calibrations

    def setCalibrations(self, calibrations):
        self._calibrations = calibrations

    def setSystemInfo(self, systemInfo):
        self._systemInfo = systemInfo

    def setfileBodyData(self, data):
        self._fileBodyData = data

