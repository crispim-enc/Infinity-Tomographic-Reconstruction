import numpy as np
import pickle


class ToRFile:
    def __init__(self, filepath=None):
        if filepath is None:
            raise FileNotFoundError("Filepath is required")

        self._systemInfoSize = None
        self._filePath = filepath
        self._version = "1.0"
        self._acquisitionInfo = None
        self._systemInfo = None
        self._listModeFields = None
        self._listMode = None

    def read(self, filePath = None):
        if filePath is None:
            filePath = self._filePath
        with open(filePath, 'rb') as input_file:
            self._systemInfo = pickle.load(input_file)
            self._acquisitionInfo = pickle.load(input_file)
            self._listMode = np.load(input_file)
            # self._listMode = np.fromfile(input_file, dtype=[('energyA', np.float64), ('energyB', np.float64),
            #                                                 ('IDA', np.float64), ('IDB', np.float64),
            #                                                 ('AXIAL_MOTOR', np.float64), ('FAN_MOTOR', np.float64),
            #                                                 ('TIME', np.float64)])

            # self._listModeFields = records.dtype.names
            # self._version = np.fromfile(input_file, dtype=np.uint32, count=1)

        input_file.close()

    def write(self):
        # records = np.rec.fromarrays((self._listMode), names=self._listModeFields)
        with open(self._filePath, 'wb') as output_file:
            # size_acquisition_info = np.fromfile(output_file, dtype=np.int32, count=1)
            # self._listModeFields.tofile(output_file)
            pickle.dump(self._systemInfo, output_file)
            pickle.dump(self._acquisitionInfo, output_file)
            # self._listMode.tofile(output_file, dtype=[('energyA', np.float64), ('energyB', np.float64),
            #                                                 ('IDA', np.float64), ('IDB', np.float64),
            #                                                 ('AXIAL_MOTOR', np.float64), ('FAN_MOTOR', np.float64),
            #                                                 ('TIME', np.float64)])
            np.save(output_file, self._listMode)


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
    def listModeFields(self):
        return self._listModeFields

    @property
    def systemInfo(self):
        return self._systemInfo

    @property
    def listMode(self):
        return self._listMode

    @property
    def acquisitionInfo(self):
        return self._acquisitionInfo

    def setAcquisitionInfo(self, acquisitionInfo):
        self._acquisitionInfo = acquisitionInfo

    def setSystemInfo(self, systemInfo):
        self._systemInfo = systemInfo

    def setListMode(self, listMode):
        self._listMode = listMode

    def setListModeFields(self, listModeFields):
        self._listModeFields = listModeFields