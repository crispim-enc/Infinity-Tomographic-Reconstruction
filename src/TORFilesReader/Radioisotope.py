#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: Radioisotope
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

__author__ = "Pedro Encarnação"
__date__ = "24/03/2025"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.1"

import time


class RadioisotopeInfo:
    """
    This class contains the information about the radioisotope used in the scan.
    Allows to add the multiple tracers.
    Atributes:


    Methods:
        setTracers() -> list:
            Set the list of tracers

        setHalfLifes() -> list:
            Set the list of half lifes

        setPositronFractions() -> list:
            Set the list of positron fractions

        setDecayFactor() -> list:
            Set the decay factor

        setDecayTypes() -> list:
            Set the decay types

        setDecayEnergies() -> list:
            Set the decay energies

        setDecayPercentage() -> list:
            Set the decay percentage

        setResidualActivity() -> list:
            Set the residual activity

        setTotalDose() -> list:
            Set the total dose

        setRoute() -> list:
            Set the route

        setStartDate() -> list:
            Set the start date

        setEndDate() -> list:
            Set the end date

        setInjectionDateTime() -> list:
            Set the injection date time

        setBatchCodeValue() -> list:
            Set the batch code value

        setBatchCodingSchemeDesignator() -> list:
            Set the batch coding scheme designator

        setBatchCodingSchemeVersion() -> list:
            Set the batch coding scheme version

        setBatchCodeMeaning() -> list:
            Set the batch code meaning

        setBatchLongCodeValue() -> list:
            Set the batch long code value


    Not implemented

    """
    def __init__(self):
        self._tracers = None
        self._halfLifes = None
        self._positronFractions = None
        self._decayFactor = None
        self._decayTypes = None
        self._decayEnergies = None
        self._decayPercentage = None
        self._residualActivity = None
        self._totalDose = None
        self._route = None
        self._startDateTime = None
        self._endDateTime = None
        self._injectionDateTime = None
        self._batchCodeValue = None
        self._batchCodingSchemeDesignator = None
        self._batchCodingSchemeVersion = None
        self._batchCodeMeaning = None
        self._batchLongCodeValue = None

    @property
    def tracers(self):
        """
        Get the list of tracers
        return: list: ['F18', 'C11']
        """
        return self._tracers

    def setTracers(self, tracers: list):
        """
        Set a list of tracers
        example: ['F18', 'C11']
        """
        self._tracers = tracers

    @property
    def halfLifes(self):
        """
        Get the list of half lifes
        Returns: list: e.g. [6586.26, 1224] in seconds
        """
        return self._halfLifes

    def setHalfLifes(self, halfLifes: list):
        """
        Set a list of half lifes
        Example: [6586.26, 1224] in seconds. Each value needs to be a float
        """
        self._halfLifes = halfLifes
        # verify if the list of half lifes is the same size as the list of tracers
        if len(self._tracers) != len(self._halfLifes):
            raise ValueError("The list of half lifes needs to have the same size as the list of tracers")

        # verify if the half lifes are positive
        for halfLife in self._halfLifes:
            if halfLife <= 0:
                raise ValueError("The half life needs to be a positive value")

        # verify if the half lifes are float
        for halfLife in self._halfLifes:
            if not isinstance(halfLife, float):
                raise ValueError("The half life needs to be a float value")

    @property
    def positronFractions(self):
        """
        Get the list of positron fractions
        Returns: list: [0.97]
        """
        return self._positronFractions

    def setPositronFractions(self, positronFractions: list):
        """
        Set the positron fraction
        Example: [0.97]
        """
        self._positronFractions = positronFractions
        # verify if the positron fraction is between 0 and 1
        for positronFraction in self._positronFractions:
            if positronFraction < 0 or positronFraction > 1:
                raise ValueError("The positron fraction needs to be between 0 and 1")

        for positronFraction in self._positronFractions:
            if not isinstance(positronFraction, float):
                raise ValueError("The positron fraction needs to be a float value")

    @property
    def decayFactor(self):
        """
        Get the decay factor
        """
        return self._decayFactor

    def setDecayFactor(self, decayFactor: list):
        """
        Set the decay factor
        Example:[1.0]
        """
        self._decayFactor = decayFactor
        # verify if the decay factor is a float
        if not isinstance(self._decayFactor, list):
            raise ValueError("The decay factor needs to be a float value")

    @property
    def decayTypes(self):
        """
        Get the decay types
        Returns: list: ['BetaPlus']  # BetaPlus, BetaMinus, Alpha, Gamma
        """
        return self._decayTypes

    def setDecayTypes(self, decayTypes: list):
        """
        Set the decay types
        Example: ['BetaPlus', 'BetaMinus', 'Alpha', 'Gamma']
        """
        self._decayTypes = decayTypes
        # verify if the decay types are a list
        if not isinstance(self._decayTypes, list):
            raise ValueError("The decay types needs to be a list")

        # verify if the decay types are one of the following: BetaPlus, BetaMinus, Alpha, Gamma

        for decayType in self._decayTypes:
            if decayType not in ["BetaPlus", "BetaMinus", "Alpha", "Gamma"]:
                raise ValueError("The decay types needs to be one of the following: BetaPlus, BetaMinus, Alpha, Gamma")

    @property
    def decayEnergies(self):
        """
        Get the decay energies
        Returns: list: [511] in KeV
        """
        return self._decayEnergies

    def setDecayEnergies(self, decayEnergies: list):
        """
        Set the decay energies
        Example: [511] in KeV
        """
        self._decayEnergies = decayEnergies
        # verify if the decay energies are a list
        if not isinstance(self._decayEnergies, list):
            raise ValueError("The decay energies needs to be a list")

        # verify if the decay energies are float
        for decayEnergy in self._decayEnergies:
            if not isinstance(decayEnergy, float) and not isinstance(decayEnergy, int):
                raise ValueError("The decay energies needs to be a float value")

    @property
    def decayPercentage(self):
        """
        Get the decay percentage
        Returns: list: [0.97]
        """
        return self._decayPercentage

    def setDecayPercentage(self, decayPercentage: list):
        """
        Set the decay percentage
        Example: [0.97]
        """
        self._decayPercentage = decayPercentage
        # verify if the decay percentage is between 0 and 1
        for decayPercentage in self._decayPercentage:
            if decayPercentage < 0 or decayPercentage > 1:
                raise ValueError("The decay percentage needs to be between 0 and 1")

        for decayPercentage in self._decayPercentage:
            if not isinstance(decayPercentage, float):
                raise ValueError("The decay percentage needs to be a float value")

    @property
    def residualActivity(self):
        """
        Get the residual activity
        Returns: list: [10] in MBq
        """
        return self._residualActivity

    def setResidualActivity(self, residualActivity: list):
        """
        Set the residual activity
        Example: [10] in MBq
        """
        self._residualActivity = residualActivity
        # verify if the residual activity is a list
        if not isinstance(self._residualActivity, list):
            raise ValueError("The residual activity needs to be a list")

        # verify if the residual activity is float
        for residualActivity in self._residualActivity:
            if not isinstance(residualActivity, float):
                raise ValueError("The residual activity needs to be a float value")

    @property
    def totalDose(self):
        """
        Get the total dose
        Returns: list: [10] in MBq
        """
        return self._totalDose

    def setTotalDose(self, totalDose: list):
        """
        Set the total dose
        Example: [10] in MBq
        """
        self._totalDose = totalDose
        # verify if the total dose is a list
        if not isinstance(self._totalDose, list):
            raise ValueError("The total dose needs to be a list")

        # verify if the total dose is float
        for totalDose in self._totalDose:
            if not isinstance(totalDose, float):
                raise ValueError("The total dose needs to be a float value")

    @property
    def route(self):
        """
        Get the route
        Returns: list: ['IV']
        """
        return self._route

    def setRoute(self, route: list):
        """
        Set the route
        Example: ['IV']
        """
        self._route = route
        # verify if the route is a list
        if not isinstance(self._route, list):
            raise ValueError("The route needs to be a list")

    @property
    def startDateTime(self):
        """
        Get the start date time
        Returns: list: ['2025-03-24 00:00:00']
        """
        return self._startDateTime

    def setStartDate(self, startDateTime: time.time()):
        """
        Set the start date time
        Example: ['2025-03-24 00:00:00']
        """
        self._startDateTime = startDateTime
        # verify if the start date time is a list
        if not isinstance(self._startDateTime, list):
            raise ValueError("The start date time needs to be a list")

    @property
    def endDateTime(self):
        """
        Get the end date time
        Returns: list: ['2025-03-24 00:00:00']
        """
        return self._endDateTime

    def setEndDate(self, endDateTime: time.time()):
        """
        Set the end date time
        Example: ['2025-03-24 00:00:00']
        """
        self._endDateTime = endDateTime
        # verify if the end date time is a list
        if not isinstance(self._endDateTime, list):
            raise ValueError("The end date time needs to be a list")

