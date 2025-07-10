# *******************************************************
# * FILE: rootjoiner.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import uproot
from uproot.exceptions import KeyInFileError
import glob
import os


class RootJoiner:
    def __init__(self, file_path,  path_to_save=None, tree_name="Singles", double_scanner_format=False):
        self.file_path = file_path
        self.path_to_save = path_to_save
        if self.path_to_save is None:
            self.path_to_save = self.file_path
        self.tree_name = tree_name
        self.double_scanner_format = double_scanner_format
        self._rootFiles = glob.glob(os.path.join(file_path, "*.root"))
        self._rootFiles.sort()
        if self.double_scanner_format:
            self._rootFiles_scanner1= [f"{el}:SinglesScanner1" for el in self._rootFiles]
            self._rootFiles_scanner2 = [f"{el}:SinglesScanner2" for el in self._rootFiles]
            self._globalArray_scanner1 = None
            self._globalArray_scanner2 = None

        else:
            self._rootFiles = [f"{el}:{tree_name}" for el in self._rootFiles]
            self._globalArray = None

    def chooseJustFilteredData(self):
        # filter files which has filtered in the late part of the name
        self._rootFiles = [el for el in self._rootFiles if "filtered" in el]
        
    def join(self):
        print("Joining root files")
        print("Number of files: ", len(self._rootFiles))
        if self.double_scanner_format:
            print("Double scanner format")
            self._globalArray_scanner1 = uproot.concatenate(self._rootFiles_scanner1)
            self._globalArray_scanner2 = uproot.concatenate(self._rootFiles_scanner2)
            print("Saving joined root file")
            # try:
            #
            #     del self._globalArray_scanner1["level1ID"]
            #     del self._globalArray_scanner1["level2ID"]
            #     del self._globalArray_scanner2["level1ID"]
            #     del self._globalArray_scanner2["level2ID"]
            #
            # except KeyInFileError:
            #     pass
            root = uproot.recreate(os.path.join(self.path_to_save, "joined.root"))
            root["SinglesScanner1"] = self._globalArray_scanner1
            root["SinglesScanner2"] = self._globalArray_scanner2
            print("End of join")
        else:
            self._globalArray = uproot.concatenate(self._rootFiles)
            # try:
            #     del self._globalArray["comptVolName"]
            #     del self._globalArray["RayleighVolName"]
            #
            # except AttributeError:
            #     pass
            print("Saving joined root file")
            # try:

            root = uproot.create(os.path.join(self.path_to_save, "joined.root"))
            # except OSError:
            #     # delete the file and create a new one
            #     os.remove(os.path.join(self.path_to_save, "joined.root"))
            #     root = uproot.create(os.path.join(self.path_to_save, "joined.root"))

            # root[self.tree_name] = self._globalArray
            root["Singles"] = self._globalArray
            print("End of join")

        root.close()

    @property
    def globalArray(self):
        return self._globalArray

    @property
    def globalArray_scanner1(self):
        return self._globalArray_scanner1

    @property
    def globalArray_scanner2(self):
        return self._globalArray_scanner2


if __name__ == "__main__":
    filePath = "C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\SimulacoesGATE\\EasyPET3D64\\FOV-UniformSource\\20-December-2022_17h23_8turn_0p005s_1p80bot_0p23top_range108\\"
    rootJoiner = RootJoiner(file_path=filePath, tree_name="SinglesScanner1")
    rootJoiner.chooseJustFilteredData()
    rootJoiner.join()
