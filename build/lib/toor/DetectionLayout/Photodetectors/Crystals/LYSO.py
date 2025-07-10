# *******************************************************
# * FILE: LYSO.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

from toor.DetectionLayout.Photodetectors.Crystals import GenericCrystal


class LYSOCrystal(GenericCrystal):
    """
    Class that represents a LYSO crystal. It contains the information about the crystal geometry and the detectors that compose it.
    Methods:


    """
    def __init__(self, crystal_id=1):
        super(LYSOCrystal, self).__init__()
        self._density = 7.4
        self._crystalID = crystal_id

