import numpy as np


class GroundTruth:
    # Explain the Method
    """
    This class is used to read the listmode data from the simulation and generate the ground truth image.
    The grouth truth image is generated
    The data is filtered in energy and for different CTRs

    """
    def __init__(self, filePath=None, pixelSize=None):
        if filePath is None:
            return
        self._filePath = filePath
        self._listMode = None
        self._energyWindow = [300, 700]
        self._groundTruth = None
        self._pixelSize = pixelSize

    def readAnhnilationData(self, listMode=None):
        if listMode is None:
            self._listMode = np.load(self._filePath)
        else:
            self._listMode = listMode
        return self._listMode

    def applyEnergyWindow(self, window=None):
        if window is None:
            window = self._energyWindow
        self._energyWindow = window
        self._listMode = self._listMode[(self._listMode >= window[0] & self._listMode < window[1]), 6]
        self._listMode = self._listMode[(self._listMode >= window[0] & self._listMode < window[1]), 7]

    @property
    def groundTruth(self):
        return self._groundTruth

    def generateImageSameSize(self, image):
        self._groundTruth = np.histogramdd((self._listMode[2, :], self._listMode[3, :], self._listMode[4, :]),
                                           bins=image.shape, range=[[-image.shape[0] * self._pixelSize[0] / 2,
                                                                     image.shape[0] * self._pixelSize[0] / 2],
                                                                    [-image.shape[1] * self._pixelSize[1] / 2,
                                                                        image.shape[1] * self._pixelSize[1] / 2],
                                                                    [-image.shape[2] * self._pixelSize[2] / 2,
                                                                        image.shape[2] * self._pixelSize[2] / 2]])


if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    import matplotlib.pyplot as plt
    from src.GateLink.RootToTor import GenerateCoincidencesAndAnhnilationsPositions

    root = tk.Tk()
    root.withdraw()

    # fp = filedialog.askopenfilename()
    fp = '/home/crispim/Documentos/Simulations/easyPET_part0_copy.root'
    fp = '/media/crispim/Storage/big_simulations/15-December-2022_0h11_40turn_0p005s_1p80bot_0p23top_range108.root'
    coinc = GenerateCoincidencesAndAnhnilationsPositions(filename=fp)
    coinc.readRoot()
    arrays_keys = ['sourcePosX', 'sourcePosY', 'sourcePosZ', 'time', 'baseID', 'energy', 'globalPosX', 'globalPosY',
                   'globalPosZ', 'level1ID', 'level2ID', 'level3ID', 'level4ID']
    coinc.setArraysToConvert(coinc.singlesScanner1, arrays_keys)
    listmode = np.array([coinc.singlesScanner1.energy, coinc.singlesScanner1.energy, coinc.singlesScanner1.sourcePosX,
                        coinc.singlesScanner1.sourcePosY, coinc.singlesScanner1.sourcePosZ, coinc.singlesScanner1.time])

    listmode = np.array([coinc.singlesScanner1.energy, coinc.singlesScanner1.energy, coinc.singlesScanner1.sourcePosX,
                         coinc.singlesScanner1.sourcePosY, coinc.singlesScanner1.sourcePosZ,
                         coinc.singlesScanner1.time, coinc.singlesScanner1.level1ID, coinc.singlesScanner1.level2ID, coinc.singlesScanner1.level3ID])

    # listmode = np.array([coinc.singlesScanner2.energy, coinc.singlesScanner2.energy, coinc.singlesScanner2.sourcePosX,
    #                         coinc.singlesScanner2.sourcePosY, coinc.singlesScanner2.sourcePosZ,
    #                         coinc.singlesScanner2.time, coinc.singlesScanner2.level1ID, coinc.singlesScanner2.level2ID, coinc.singlesScanner2.level3ID])
    #
    # listmode = listmode[:,listmode[6, :] >-5]
    # listmode = listmode[:,listmode[7, :] >= 50]
    # listmode = listmode[:,listmode[8, :] == 30]
    print(f"Number of anhnilations:{listmode.shape[1]} ")
    gt = GroundTruth(filePath=fp)
    gt.readAnhnilationData(listMode=listmode)

    pixelSize = [0.25, 0.25, 0.25]
    # imageOriginal = np.ones((int(2/pixelSize[0]), int(2/pixelSize[1]), int(30/pixelSize[2])))
    imageOriginal = np.ones((100, 100, 68))
    gt.generateImageSameSize(image=imageOriginal)
    plt.imshow(np.mean(gt.groundTruth[0], axis=2), extent=[0, imageOriginal.shape[0]*pixelSize[0],
                                                            0, imageOriginal.shape[1]*pixelSize[1]])
    # plt.imshow(np.mean(gt.groundTruth[0], axis=0))
    plt.show()
