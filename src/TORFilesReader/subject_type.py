#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: animal
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

version = "0.1.0alpha"


class PhantomType:
    def __init__(self):
        self._phantomName = None
        self._phantomPurpose = None
        self._phantomDescription = None
        self._digitalPhantomCopy = None

    @property
    def phantomName(self):
        """
        Return: Name of the phantom
        """
        return self._phantomName

    def setPhantomName(self, phantomName):
        """
        Set the name of the phantom
        """
        if not isinstance(phantomName, str):
            raise ValueError("Phantom name must be a string")
        self._phantomName = phantomName

    @property
    def phantomPurpose(self):
        """
        Return: Purpose of the phantom
        """
        return self._phantomPurpose

    def setPhantomPurpose(self, phantomPurpose):
        """
        Set the purpose of the phantom
        """
        if not isinstance(phantomPurpose, str):
            raise ValueError("Phantom purpose must be a string")
        self._phantomPurpose = phantomPurpose

    @property
    def phantomDescription(self):
        """
        Return: Description of the phantom
        """
        return self._phantomDescription

    def setPhantomDescription(self, phantomDescription):
        """
        Set the description of the phantom
        """
        if not isinstance(phantomDescription, str):
            raise ValueError("Phantom description must be a string")
        self._phantomDescription = phantomDescription

    @property
    def digitalPhantomCopy(self):
        """
        Return: Digital phantom copy (look for phantom package to find
        """
        return self._digitalPhantomCopy

    def setDigitalPhantomCopy(self, digitalPhantomCopy):
        """
        Set the digital phantom copy
        """
        self._digitalPhantomCopy = digitalPhantomCopy




class AnimalType:
    """
    AnimalType class
    Attributes:
    - healthy: bool
    - diagnosis: str
    - observations: str
    - speciesDescription: str
    - speciesCodeValue: str
    - speciesCodingSchemeDesignator: str
    - speciesCodingSchemeVersion: str
    - speciesCodeMeaning: str
    - speciesLongCodeValue: str
    - speciesURNCodeValue: str
    - breedDescription: str
    - breedCodeValue: str
    - breedCodingSchemeDesignator: str
    - breedCodingSchemeVersion: str
    - breedCodeMeaning: str
    - breedLongCodeValue: str
    - breedURNCodeValue: str
    - breedResgNumber: str
    - breedResgCodeValue: str
    - breedResgCodingSchemeDesignator: str
    - breedResgCodingSchemeVersion: str
    - breedResgCodeMeaning: str
    - fasting: bool
    - fastingStart: time.time() or str
    - fastingDuration: int
    - glucoseLevel: float
    - glucoseTime: time.time() or str
    - weight: float
    - size: float
    - animalTemp: float
    - environmentTemp: float
    - position: str
    - typeOfAnesthesia: str
    - anestheticAgent: str
    - anestheticVolume: float
    - anestheticPercentage: float
    - gasTransport: str
    - gasQuantity: float
    - routeA: str
    """
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



