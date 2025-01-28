#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: easyPET
# * AUTHOR: Pedro Encarnação
# * DATE: 28/01/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Brief description of the file.
"""
import numpy as np
from .PETModuleGeneric import PETModule


class easyPETModule(PETModule):
    """
    Class that represents a easyPET module. It contains the information about the module geometry and the detectors that compose it.
    Methods:


    """
    def __init__(self, module_id=1):
        super(easyPETModule, self).__init__()

        self._idModule = module_id

        # self._vertices = np.array([[
