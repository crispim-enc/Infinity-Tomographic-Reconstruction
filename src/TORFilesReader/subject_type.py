#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: animal
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

class PhantomType:
    def __init__(self):
        self._phantomDescription = None


class AnimalType:
    def __init__(self):
        self._healthy = None
        self._diagnosis = None
        self._observations = None
        self._speciesDescription = None
        self._speciesCodeValue = None
        self._speciesCodingSchemeDesignator = None
        self._speciesCodingSchemeVersion = None
        self._speciesCodeMeaning = None
        self._speciesLongCodeValue = None
        self._speciesURNCodeValue = None
        self._breedDescription = None
        self._breedCodeValue = None
        self._breedCodingSchemeDesignator = None
        self._breedCodingSchemeVersion = None
        self._breedCodeMeaning = None
        self._breedLongCodeValue = None
        self._breedURNCodeValue = None
        self._breedResgNumber = None
        self._breedResgCodeValue = None
        self._breedResgCodingSchemeDesignator = None
        self._breedResgCodingSchemeVersion = None
        self._breedResgCodeMeaning = None
        self._fasting = None
        self._fastingStart = None
        self._fastingDuration = None
        self._glucoseLevel = None
        self._glucoseTime = None
        self._weight = None
        self._size = None
        self._animalTemp = None
        self._environmentTemp = None
        self._position = None
        self._typeOfAnesthesia = None
        self._anestheticAgent = None
        self._anestheticVolume = None
        self._anestheticPercentage = None
        self._gasTransport = None
        self._gasQuantity = None
        self._routeA = None

    @property
    def healthy(self):
        return self._healthy

    def setHealthy(self, healthy):
        """
        bool: Healthy or not
        """
        # if not a bool raise
        if not isinstance(healthy, bool):
            raise ValueError("Healthy must be a bool")

        self._healthy = healthy

    @property
    def diagnosis(self):
        return self._diagnosis

    def setDiagnosis(self, diagnosis):
        """
        str: Diagnosis
        """
        # if not a string raise
        if not isinstance(diagnosis, str):
            raise ValueError("Diagnosis must be a string")
        self._diagnosis = diagnosis

    @property
    def observations(self):
        return self._observations

    def setObservations(self, observations):
        """
        str: Observations
        """
        # if not a string raise
        if not isinstance(observations, str):
            raise ValueError("Observations must be a string")
        self._observations = observations



