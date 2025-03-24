#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: Radioisotope
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Brief description of the file.
"""


class RadioisotopeInfo:
    def __init__(self):
        self._tracers = None
        self._halfLifes = None
        self._positronFractions = None
        self._decayFactor = None
        self._decayTypes = None
        self._decayEnergies = None
        self._decayPercentage = None
        self._decayNumber = None
        self._residualActivity = None
        self._totalDose = None
        self._route = None
        self._startDate = None
        self._endDate = None
        self._injectionDate = None
        self._batchCodeValue = None
        self._batchCodingSchemeDesignator = None
        self._batchCodingSchemeVersion = None
        self._batchCodeMeaning = None
        self._batchLongCodeValue = None

    @property
    def tracers(self):
        return self._tracers

    def setTracers(self, tracers):
        """
        Set a list of tracers
        example: ['F18', 'C11']
        """
        self._tracers = tracers
