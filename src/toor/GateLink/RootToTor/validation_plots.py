# *******************************************************
# * FILE: validation_plots.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import os
import matplotlib.pyplot as plt
import numpy as np


class CoincidenceValidationPlots:
    def __init__(self, listmode_file, array_keys, directory):
        self.listModeFile = listmode_file
        # if type(self.listModeFile) ==
        self.array_keys = array_keys
        self.directory = directory

    def generateCharts(self):
        print("Plotting coincidence validation plots")
        for key in self.array_keys:
            plt.figure()
            plt.hist(self.listModeFile[key], bins= 100, label=key)
            plt.xlabel(key)
            plt.ylabel("Coincidences")
            plt.legend()
            plt.savefig(os.path.join(self.directory, key + ".png"))
            plt.close()
