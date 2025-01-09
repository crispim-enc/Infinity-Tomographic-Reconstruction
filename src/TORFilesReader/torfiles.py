import numpy as np


class ToRFile:
    def __init__(self):

        self._systemInfoSize = None
        self._filePath = None
        self._version = "1.0"
        self._sizeHeader = None
        self._acquisitionInfoSize = None
        self._acquisitionInfo = None
        self._systemInfo = None
        self._listModeFields = None
        self._listMode = None

    def read(self, filePath):
        self._filePath = filePath
        with open(filePath, 'rb') as input_file:
            self._version = np.fromfile(input_file, dtype=np.uint32, count=1)

        input_file.close()
        self._acquisitionInfo = acquisition_info
        self._systemInfo = systemConfigurations_info
        self._listMode = records

    def write(self):
        records = np.rec.fromarrays((), names=self._listModeFields)
        with open(self.filename, 'wb') as output_file:
            self._version.tofile(output_file)
            self._sizeHeader.tofile(output_file)
            self._acquisitionInfoSize.tofile(output_file)
            self._systemInfoSize.tofile(output_file)
            self._acquisitionInfo.tofile(output_file)
            self._systemInfo.tofile(output_file)

        output_file.close()