import glob
import os
import uproot
import numpy as np
from src.ToRFilesReader import ToRFile
from src.GateLink.RootToTor import GenerateCoincidencesAndAnhnilationsPositions


class GenerateToRFileFromMultiFiles(ToRFile):
    def __init__(self, file_path=None, arrays_keys=None):
        super().__init__()
        # find root files in the folder
        self._rootFiles = glob.glob(os.path.join(file_path, "*.root"))
        self._rootFiles.sort()
        # make dir for temp coincidences
        self._tempDir = os.path.join(os.path.dirname(__file__), "temp")
        if not os.path.exists(self._tempDir):
            os.makedirs(self._tempDir)
        self._listModeFields = None
        self.arrays_keys = arrays_keys

    def readRootFiles(self, method='time_diff', coincidenceWindow=0.1):
        i = 0
        for file in self._rootFiles:
            coinc = GenerateCoincidencesAndAnhnilationsPositions(filename=file, method=method, coincidenceWindow=coincidenceWindow)
            coinc.readRoot()
            coinc.setPartNumber(i)
            # arrays_keys = ['time', 'baseID','runID','eventID', 'energy', 'level1ID', 'level2ID', 'level3ID', 'level4ID']
            # coinc.setArraysToConvert(coinc.singlesScanner1, arrays_keys)
            # coinc.setArraysToConvert(coinc.singlesScanner2, arrays_keys)
            coinc.setArraysToConvert(coinc.singles, self.arrays_keys)
            coinc.findCoincidencesBigArrays(array_keys=self.arrays_keys)
            coinc.saveCoincidencesAsNumpyRecordsArray(self.arrays_keys)
            i += 1

    def joinRootFiles(self):
        # find root files in the folder
        rootFiles = glob.glob(os.path.join(os.path.dirname(__file__), "*.npy"))
        # join the root files


if __name__ == "__main__":
    arrays_keys = ['time', 'baseID','runID','eventID', 'energy', 'level1ID', 'level2ID', 'level3ID', 'level4ID']
    generate = GenerateToRFileFromMultiRootFiles(arrays_keys=arrays_keys)
    generate.readRootFiles(method='time_diff', coincidenceWindow=0.1)
    generate.joinRootFiles()