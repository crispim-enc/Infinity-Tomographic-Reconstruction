import time
import glob
import numpy as np
import uproot
import os
import json
import matplotlib.pyplot as plt
try:
    from src.GateLink.RootToTor import ReadRootFile
except ModuleNotFoundError:
    from readroot import ReadRootFile


class GenerateCoincidencesAndAnhnilationsPositions(ReadRootFile):
    """
    This class is used to generate the coincidences and anhnilations positions from the root file data.
    The coincidences algorithm is based on the time of the events.
    It assumed that GATE digitizer is used to join multiple events that occur the same detector.
    For more information about the GATE digitizer, please refer to the GATE documentation.
    args:
        filename: the path to the root file
        method: the method used to generate the coincidences. The options are:
            SameID: the coincidence is generated when the events have the same ID
            SameTime: the coincidence is generated when the events have the same time
        coincidenceWindow: the coincidence window in nano seconds
        doubleScannerFormat: if the root file returns two arrays for the singles, one for each scanner

    parameters:
        _filename: the path to the root file
        _method: the method used to generate the coincidences. The options are:
            SameID: the coincidence is generated when the events have the same ID
            SameTime: the coincidence is generated when the events have the same time
        _coincidenceWindow: the coincidence window in nano seconds
        doubleScannerFormat: if the root file returns two arrays for the singles, one for each scanner
        _fileBodyData: the list mode data
        _coincidences: the coincidences data
        _anhnilations: the anhnilations data
        _singlesScanner1: the singles data for the first module
        _singlesScanner2: the singles data for the second module
    """
    def __init__(self, filename=None, method='time_diff', coincidenceWindow=0.1):
        # if filename is None:
        #     return
        super().__init__(filename=filename)
        self._method = method
        self._coincidenceWindow = coincidenceWindow
        self._numberOfSinglesDetected = None
        self._numberOfCoincidencesFiltered = None
        self._numberOfRealCoincidences = None

    def findCoincidences(self, coincidenceWindow=0.1, array_keys=None):  # coincidenceWindow in ns
        """
        Slower version: This method is used to find the coincidences in the singles data.
        args:
            coincidenceWindow: the coincidence window in nano seconds

        """
        print("Finding coincidences...")
        coincidenceWindow *= 10 ** -9

        if self._doubleScannerFormat:

            totalArraysSingleScanner1 = np.array([getattr(self._singlesScanner1, array_keys[i]) for i in range(len(array_keys))], dtype=np.float64)
            totalArraysSingleScanner2 = np.array([getattr(self._singlesScanner2, array_keys[i]) for i in range(len(array_keys))], dtype=np.float64)

            totalArrays = np.hstack((totalArraysSingleScanner1, totalArraysSingleScanner2))


            del totalArraysSingleScanner1, totalArraysSingleScanner2

        else:
            totalArrays = np.array([getattr(self._singles, array_keys[i]) for i in range(len(array_keys))], dtype=np.float64)

        # globalListMode = np.zeros((len(timeTotal), 10))
        print("Number of singles: ", len(totalArrays[0]))
        print("sorting...")
        # Sort the arrays by time
        index_ordered = totalArrays[0].argsort()
        totalArrays = totalArrays[:, index_ordered]

        print("Coinc from EVentID{}".format(len(np.where(np.diff(totalArrays[3]) == 0)[0])))
        diffTime = np.diff(totalArrays[0])
        # Mask to remove events that occur in the same time
        mask = diffTime <= coincidenceWindow
        mask = np.append(mask, True)
        mask[np.where(diffTime <= coincidenceWindow)[0]+1] = True

        totalArrays = totalArrays[:, mask]
        print("Singles inside coincidence window: ", len(totalArrays[0]))
        # timeTotal = timeTotal[mask]
        # baseIDTotal = baseIDTotal[mask]

        # Mask to remove events that occur in the same module

        maskID = np.zeros(len(totalArrays[0]), dtype=bool)
        indexes = np.where((totalArrays[1][1::2] - totalArrays[1][::2]) != 0)[0]*2
        print(len(indexes))
        maskID[indexes] = True
        maskID[indexes+1] = True

        totalArrays = totalArrays[:, maskID]

        singlesScanner1 = totalArrays[:, totalArrays[1] == 0]
        singlesScanner2 = totalArrays[:, totalArrays[1] == 1]
        print("Singles after coincidence window and same module: ", len(totalArrays[0]))
        print("singlesScanner1", len(singlesScanner1[0]))
        print("singlesScanner2", len(singlesScanner2[0]))
        self.setArraysToConvert(self._singlesScanner1, keys=array_keys, arr=singlesScanner1)
        self.setArraysToConvert(self._singlesScanner2, keys=array_keys, arr=singlesScanner2)
        # timeTotal = timeTotal[maskID]
        # baseIDTotal = baseIDTotal[maskID]
        print("Coincidences found")

    def findCoincidencesTrueEventID(self, array_keys=None):
        # if self._doubleScannerFormat:
        totalArraysSingleScanner1 = [getattr(self._singlesScanner1, array_keys[i]) for i in range(len(array_keys))]
        totalArraysSingleScanner2 = [getattr(self._singlesScanner2, array_keys[i]) for i in range(len(array_keys))]
        totalArrays = [np.hstack((totalArraysSingleScanner1[i], totalArraysSingleScanner2[i])) for i in range(len(array_keys))]

        eventID_index = np.lexsort((totalArrays[3], totalArrays[2]))
        totalArrays = [totalArrays[i][eventID_index] for i in range(len(array_keys))]

        diffeventID = np.diff(totalArrays[3])
        coincidences_detected = np.where(diffeventID == 0)
        print("Total events: ", len(coincidences_detected[0]))
        singlesScanner1 = [totalArrays[i][coincidences_detected[0]] for i in range(len(array_keys))]
        singlesScanner2 = [totalArrays[i][coincidences_detected[0] + 1] for i in range(len(array_keys))]
        # if eventID is equal is a coincidence
        print("singlesScanner1", len(singlesScanner1[0]))
        print("singlesScanner2", len(singlesScanner2[0]))
        self.setArraysToConvert(self._singlesScanner1, keys=array_keys, arr=singlesScanner1)
        self.setArraysToConvert(self._singlesScanner2, keys=array_keys, arr=singlesScanner2)

    def findCoincidencesBigArrays(self, coincidenceWindow=1, array_keys=None):
        """
        This method is used to find the coincidences in the singles data.

        """
        coincidenceWindow *= 10 ** -9

        if self._doubleScannerFormat:
            totalArraysSingleScanner1 = [getattr(self._singlesScanner1, array_keys[i]) for i in range(len(array_keys))]
            totalArraysSingleScanner2 = [getattr(self._singlesScanner2, array_keys[i]) for i in range(len(array_keys))]
            totalArrays = [np.hstack((totalArraysSingleScanner1[i], totalArraysSingleScanner2[i])) for i in range(len(array_keys))]

        else:
            totalArrays = [getattr(self._singles, array_keys[i]) for i in range(len(array_keys))]
        try:
            index_time = array_keys.index("time")
        except ValueError:
            print("No time array")
            return
        try:
            index_module = array_keys.index("baseID")
        except ValueError:
            print("No baseID array")
            return
        print("Number of singles: ", len(totalArrays[0]))
        index_ordered = totalArrays[index_time].argsort()
        for i in range(len(array_keys)):
            totalArrays[i] = totalArrays[i][index_ordered]

        if array_keys[3] == "eventID":
            self._numberOfRealCoincidences = len(np.where(np.diff(totalArrays[3]) == 0)[0])
            print("Coinc from EventID: {}".format(self._numberOfRealCoincidences))

        diffTime = np.diff(totalArrays[0])
        # Mask to remove events that occur in the same time
        mask = diffTime <= coincidenceWindow
        mask[np.where(diffTime <= coincidenceWindow)[0] + 1] = True
        if len(mask == True) % 2 != 0:
            mask = np.append(mask, True)
        else:
            mask = np.append(mask, False)

        for i in range(len(array_keys)):
            totalArrays[i] = totalArrays[i][mask]
        print("Singles inside coincidence window: ", len(totalArrays[0]))

        # Mask to remove events that occur in the same module
        maskID = np.zeros(len(totalArrays[0]), dtype=bool)

        indexes = np.where((totalArrays[index_module][1::2] - totalArrays[index_module][::2]) != 0)[0] * 2

        maskID[indexes] = True
        maskID[indexes + 1] = True

        for i in range(len(array_keys)):
            totalArrays[i] = totalArrays[i][maskID]

        if self._doubleScannerFormat:
            singlesScanner1 = [totalArrays[i][totalArrays[index_module] == 0] for i in range(len(array_keys))]
            singlesScanner2 = [totalArrays[i][totalArrays[index_module] == 1] for i in range(len(array_keys))]

            print("singlesScanner1", len(singlesScanner1[0]))
            print("singlesScanner2", len(singlesScanner2[0]))
            self.setArraysToConvert(self._singlesScanner1, keys=array_keys, arr=singlesScanner1)
            self.setArraysToConvert(self._singlesScanner2, keys=array_keys, arr=singlesScanner2)

        print("Singles after coincidence window and same module: ", len(totalArrays[0]))
        self.setArraysToConvert(self._singles, keys=array_keys, arr=totalArrays)

        print("Coincidences found")

    def saveCoincidencesAsRootFile(self, array_keys=None, filename=None):
        print("Saving coincidences as root file...")
        if filename is None:
            filename = os.path.join(os.path.dirname(self._filename), f"coincidences_{self._partNumber}.root")
        elif not filename.endswith(".root"):
            filename = os.path.join(filename, f"coincidences_{self._partNumber}.root")
        if self.doubleScannerFormat:
            dictToRootSinglesScanner1 = {array_keys[i]: getattr(self._singlesScanner1, array_keys[i]) for i in range(len(array_keys))}
            dictToRootSinglesScanner2 = {array_keys[i]: getattr(self._singlesScanner2, array_keys[i]) for i in range(len(array_keys))}
        else:
            dictToSingles = {array_keys[i]: getattr(self._singles, array_keys[i]) for i in range(len(array_keys))}

        with uproot.recreate(filename) as root_file:
            if self.doubleScannerFormat:
                root_file["SinglesScanner1"] = dictToRootSinglesScanner1
                root_file["SinglesScanner2"] = dictToRootSinglesScanner2
            else:
                root_file["Singles"] = dictToSingles

    def saveCoincidencesAsNumpyRecordsArray(self, array_keys=None):
        """

        """
        print("Saving coincidences as numpy records array...")
        filename = os.path.join(os.path.dirname(self._filename), "numpy_files", f"coincidences_{self._partNumber}.npy")
        file_path = os.path.dirname(filename)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        if self.doubleScannerFormat:
            records = np.rec.fromarrays((getattr(self._singlesScanner1, array_keys[i]) for i in range(len(array_keys))),
                                        names=array_keys)
            records = np.hstack((records, np.rec.fromarrays((getattr(self._singlesScanner2, array_keys[i]) for i in range(len(array_keys))),
                                        names=array_keys)))
        else:
            records = np.rec.fromarrays((getattr(self._singles, array_keys[i]) for i in range(len(array_keys))),
                                        names=array_keys)
        list_of_tuples = records.dtype.descr
        with open(filename, 'wb') as output_file:
            output_file.write('_'.join(f'{tup[0]} {tup[1]}' for tup in list_of_tuples).encode('utf-8'))
            output_file.write('\n'.encode('utf-8'))
            records.tofile(output_file)

    def readCoincidencesNumpyRecordsArray(self):
        """

        """
        print("Reading coincidences as numpy records array...")
        filename = os.path.join(os.path.dirname(self._filename), "numpy_files", f"coincidences_{self._partNumber}.npy")
        with open(filename, 'rb') as input_file:
            structType = input_file.readline().decode('utf-8').split('\n')[0].split('_')
            structType = [tuple(tup.split(' ')) for tup in structType]
            records = np.rec.fromfile(input_file, dtype=structType)
        return records

    def printCoincidenceStats(self):
        with open(os.path.join(os.path.dirname(self._filename), "coincidences_stats.txt"), "w") as file:
            file.write(f"Part number: {self._partNumber}")

        print("Part number: ", self._partNumber)
        print("Extract from File: ", self._filename)
        print("Initial Time: ", self._singles.time.min())
        print("Final Time: ", self._singles.time.max())
        print("Total Time: ", self._singles.time.max() - self._singles.time.min())
        print("Module 0: ", len(self._singles.time[self._singles.baseID == 0]))
        print("Module 1: ", len(self._singles.time[self._singles.baseID == 1]))
        print("Energy: ", self._singles.energy.min(), self._singles.energy.max())
        print("Energy mean: ", self._singles.energy.mean())
        print("Energy std: ", self._singles.energy.std())
        print("level1ID: ", self._singles.level1ID.min(), self._singles.level1ID.max())
        print("level2ID: ", self._singles.level2ID.min(), self._singles.level2ID.max())
        print("level3ID: ", self._singles.level3ID.min(), self._singles.level3ID.max())




if __name__ == '__main__':
    tic = time.time()
    import tkinter as tk
    from tkinter import filedialog
    file_path = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\" \
                "FOV-UniformSource\\20-December-2022_17h23_8turn_0p005s_1p80bot_0p23top_range108\\" \
                "easyPET_part0_filtered.root"
    file_path = ("C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\StaticAquisition\\"
                 "17-Jan-2023_14h30_1turn_0p25s_180p0bot_0p9top_range0.9\\easyPET_part0.root")
    # file_path = '/home/crispim/Documentos/Simulations/easyPET_part0_copy.root'
    coinc = GenerateCoincidencesAndAnhnilationsPositions(filename=file_path)

    # coinc._partNumber = 0
    coinc.setPartNumber(0)
    coinc.setDoubleScannerFormat(True)
    coinc.readRoot()
    # arrays_keys = ['time', 'baseID', 'runID','eventID', 'energy', 'level1ID', 'level2ID', 'level3ID']
    arrays_keys = ['time', 'baseID', 'runID', 'eventID', 'sourcePosX', 'sourcePosY', 'sourcePosZ', 'energy',
                   'level3ID']
    coinc.setArraysToConvert(coinc.singlesScanner1, arrays_keys)
    coinc.setArraysToConvert(coinc.singlesScanner2, arrays_keys)
    # coinc.setArraysToConvert(coinc.singles, arrays_keys)
    tec = time.time()
    # # #
    # coinc.findCoincidencesBigArrays(array_keys=arrays_keys,coincidenceWindow=0.1)
    coinc.findCoincidencesTrueEventID(array_keys=arrays_keys)
    coinc.saveCoincidencesAsRootFile(array_keys=arrays_keys)
    coinc.saveCoincidencesAsNumpyRecordsArray(array_keys=arrays_keys)


    # # # # coinc.saveCoincidencesAsRootFile(arrays_keys)
    # coinc.saveCoincidencesAsNumpyRecordsArray(arrays_keys)
    # records_ = coinc.readCoincidencesNumpyRecordsArray()

    toc = time.time()
    print("Time elapsed: ", toc-tic)
    print("Time finding coincidences: ", toc-tec)
    # import matplotlib.pyplot as plt
    #
    # plt.hist(coinc.singlesScanner1.time, bins=100)
