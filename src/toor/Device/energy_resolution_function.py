#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: energy_resolution_function
# * AUTHOR: Pedro Encarnação
# * DATE: 14/04/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Brief description of the file.
"""
import numpy as np


class EnergyResolutionFunction:
    def __init__(self, p1=None, p2=None):
        """
        Initialize the EnergyResolutionFunction class.
        :param energy_resolution_function: The energy resolution function to be used for the calculation.       """
        self._p1 = p1
        self._p2 = p2

    def run(self, E):
        fwhm = np.sqrt((self._p1 / E) ** 2 + self._p2 ** 2)

        return fwhm / E