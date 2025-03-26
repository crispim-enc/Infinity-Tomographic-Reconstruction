#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: scanning_body
# * AUTHOR: Pedro Encarnação
# * DATE: 26/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
ListMode body of the TOR file
"""


class ListModeBody:
    """
    Class to store the statistics of the listmode data
    Attributes:
    listmode: numpy array
    listmodeFields: list of strings
    mean: numpy array with the mean for each field
    std: numpy array with the standard deviation for each field
    min: numpy array with the minimum for each field
    max: numpy array with the maximum for each field
    median: numpy array with the median for each field
    numberOfEvents: int
    numberOfEventsPerSecond: int
    numberOfEventsPerFrame: list of int


    """
    def __init__(self):
        self._listmode = None
        self._listmodeFields = None
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._median = None
        self._numberOfEvents = None
        self._numberOfEventsPerSecond = None
        self._numberOfEventsPerFrame = None

    def setListmode(self, listmode):
        """
        Set the listmode data
        """
        self._listmode = listmode

    def resetListmode(self):
        """
        Call this method before store the listmode data to reduce memory usage
        """
        self._listmode = None

    @property
    def listmodeFields(self):
        return self._listmodeFields

    def setListmodeFields(self, listmodeFields):
        self._listmodeFields = listmodeFields



