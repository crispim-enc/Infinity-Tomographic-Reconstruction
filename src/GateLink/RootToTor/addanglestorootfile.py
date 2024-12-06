import os
import numpy as np
import uproot
try:
    from src.GateLink.RootToTor import ReadRootFile
except ModuleNotFoundError:
    from readroot import ReadRootFile


class AddMotorInfoToRoot(ReadRootFile):
    def __init__(self, filename=None, doubleScannerFormat=True, path_to_macs=None):
        super().__init__(filename=filename)
        if filename is None:
            return
        # self._filename = filename
        if path_to_macs is None:
            self._directory  = os.path.dirname(filename)
        else:
            self._directory = path_to_macs
        self._flat_tree = None
        self._bottomMotor = None
        self._topMotor = None
        self._bottomMotorFileName = os.path.join(self._directory, "positionsScanner1.mac")
        self._topMotorFileName = os.path.join(self._directory, "positions.mac")
        self._bottomMotorToSaveSinglesScanner1 = None
        self._bottomMotorToSaveSinglesScanner2 = None
        self._topMotorToSaveSinglesScanner1 = None
        self._topMotorToSaveSinglesScanner2 = None
        self._bottomMotorToSaveSingles = None
        self._topMotorToSaveSingles = None
        # self._singlesScanner1 = None
        # self._singlesScanner2 = None
        # self._singles = None
        self._timeincrement = 0.005

    def readMotorFiles(self):
        print("Reading motor files: {} and {}".format(self._bottomMotorFileName, self._topMotorFileName))
        with open(self._bottomMotorFileName, "r") as f:
            self._bottomMotor = f.readlines()
            self._bottomMotor = [np.array(self._bottomMotor[i].split(" "), dtype=np.float32) for i in
                                 range(8, len(self._bottomMotor))]
            self._bottomMotor = np.array(self._bottomMotor)
            self._bottomMotor = self._bottomMotor[:, :2]
            self._bottomMotor[:, 1] = np.round(self._bottomMotor[:, 1], 3)

        self._timeincrement = self._bottomMotor[1, 0] - self._bottomMotor[0, 0]

        with open(self._topMotorFileName, "r") as f:
            self._topMotor = f.readlines()
            self._topMotor = [np.array(self._topMotor[i].split(" "), dtype=np.float32) for i in
                              range(8, len(self._topMotor))]
            self._topMotor = np.array(self._topMotor)
            self._topMotor = self._topMotor[:, :2]
            self._topMotor[:, 1] = np.round(self._topMotor[:, 1], 3)

    def createMotorArrays(self):
        print("Creating motor arrays")
        if self.doubleScannerFormat:
            indexesSingleScanner1 = (self._singlesScanner1.time / self._timeincrement-1).astype(np.int32)
            indexesSingleScanner2 = (self._singlesScanner2.time / self._timeincrement-1).astype(np.int32)

            self._bottomMotorToSaveSinglesScanner1 = self._bottomMotor[indexesSingleScanner1, 1]
            self._topMotorToSaveSinglesScanner1 = self._topMotor[indexesSingleScanner1, 1]
            self._bottomMotorToSaveSinglesScanner2 = self._bottomMotor[indexesSingleScanner2, 1]
            self._topMotorToSaveSinglesScanner2 = self._topMotor[indexesSingleScanner2, 1]

        else:
            indexesSingle = (self._singles.time / self._timeincrement).astype(np.int32)
            self._bottomMotorToSaveSingles = self._bottomMotor[indexesSingle, 1]
            self._topMotorToSaveSingles = self._topMotor[indexesSingle, 1]

    def saveMotorArraysIntoRoot(self, array_keys):
        print("Saving motor arrays into root file: {}".format(self._filename))
        if self.doubleScannerFormat:
            temp_scanner1 = [getattr(self._singlesScanner1, array_keys[i]) for i in range(len(array_keys))]
            temp_scanner2 = [getattr(self._singlesScanner2, array_keys[i]) for i in range(len(array_keys))]
            dictToSingles = {array_keys[i]: np.hstack((temp_scanner1[i], temp_scanner2[i])) for i in range(len(array_keys))}
            dictToSingles["level1ID"] = np.hstack((self._bottomMotorToSaveSinglesScanner1, self._bottomMotorToSaveSinglesScanner2))
            dictToSingles["level2ID"] = np.hstack((self._topMotorToSaveSinglesScanner1, self._topMotorToSaveSinglesScanner2))

        else:
            dictToSingles = {array_keys[i]: getattr(self._singles, array_keys[i]) for i in range(len(array_keys))}
            dictToSingles["level1ID"] = self._bottomMotorToSaveSingles
            dictToSingles["level2ID"] = self._topMotorToSaveSingles

        name_folder = os.path.join(os.path.dirname(self._filename),"motors_added")
        if not os.path.exists(name_folder):
            os.mkdir(name_folder)
        name_file = os.path.basename(self._filename).split(".")[0]
        file = os.path.join(name_folder, f"{name_file}_copy.root")
        self._filename = file
        # self._filename = "easyPET_part0_copy.root"
        with uproot.recreate(self._filename) as root_file:

            # if self.doubleScannerFormat:
            #     root_file["SinglesScanner1"] = dictToRootSinglesScanner1
            #     root_file["SinglesScanner2"] = dictToRootSinglesScanner2
            # else:
            root_file["Singles"] = dictToSingles


if __name__ == "__main__":
    file_path = "/home/crispim/Documentos/Simulations/"
    root_file = "easyPET_part0(1).root"
    file_path = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\StaticAquisition\\17-Jan-2023_14h30_1turn_0p25s_180p0bot_0p9top_range0.9"
    root_file = "coincidences_0.root"
    arrays_keys = ['time', 'baseID', 'runID', 'eventID', 'sourcePosX', 'sourcePosY', 'sourcePosZ', 'energy',
                   'globalPosX', 'globalPosY',
                   'globalPosZ', 'level1ID', 'level2ID', 'level3ID', 'level4ID']

    rootFile = AddMotorInfoToRoot(filename=os.path.join(file_path, root_file))
    rootFile.readRoot()

    rootFile.setArraysToConvert(rootFile.singlesScanner1, arrays_keys)
    rootFile.setArraysToConvert(rootFile.singlesScanner2, arrays_keys)
    rootFile.readMotorFiles()
    rootFile.createMotorArrays()
    rootFile.saveMotorArraysIntoRoot(arrays_keys)
    import matplotlib.pyplot as plt

    plt.hist2d(rootFile._bottomMotorToSaveSinglesScanner1, rootFile._topMotorToSaveSinglesScanner1,
               bins=[np.unique(rootFile._bottomMotorToSaveSinglesScanner1),
                     np.unique(rootFile._topMotorToSaveSinglesScanner1)])
    plt.show()
    # rootFile.saveMotorArraysIntoRoot()
