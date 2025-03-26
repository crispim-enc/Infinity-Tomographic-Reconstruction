#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: scanning_header
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

__author__ = "Pedro Encarnação"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.1.0alpha"


import time


class AcquisitionInfo:
    def __init__(self):
        self._subject = None
        self._tecnhician = None
        self._date = None
        self._radioisotope = None
        self._scanType = None
        self._id = None
        self._numberOfFrames = 0
        self._indexesOfFrames = []
        self._instanceUID = None
        self._studyInstanceUID = None
        self._frameOfReferenceUID = None

    @property
    def subject(self):
        return self._subject

    def setSubject(self, subject):
        self._subject = subject

    @property
    def id(self):
        return self._id

    def setId(self, id):
        self._id = id

    @property
    def numberOfFrames(self):
        return self._numberOfFrames

    def setNumberOfFrames(self, numberOfFrames):
        self._numberOfFrames = numberOfFrames
        if not isinstance(self._numberOfFrames, int):
            raise ValueError("Number of indexes must be a int")

    @property
    def indexesOfFrames(self):
        return self._indexesOfFrames

    def setIndexesOfFrames(self, indexesOfFrames):
        self._indexesOfFrames = indexesOfFrames
        if not isinstance(self._indexesOfFrames, list):
            raise ValueError("Indexes of frames must be a list")
        self.setNumberOfFrames(int(len(self._indexesOfFrames)-1))

    @property
    def instanceUID(self):
        return self._instanceUID

    def setInstanceUID(self, instanceUID):
        self._instanceUID = instanceUID

    @property
    def studyInstanceUID(self):
        return self._studyInstanceUID

    def setStudyInstanceUID(self, studyInstanceUID):
        self._studyInstanceUID = studyInstanceUID

    @property
    def frameOfReferenceUID(self):
        return self._frameOfReferenceUID

    def setFrameOfReferenceUID(self, frameOfReferenceUID):
        self._frameOfReferenceUID = frameOfReferenceUID

    def setRadioisotope(self, radioisotope):
        self._radioisotope = radioisotope

    @property
    def radioisotope(self):
        return self._radioisotope

    @property
    def scanType(self):
        return self._scanType

    def setScanType(self, scanType):
        self._scanType = scanType

    def setTecnhician(self, tecnichian):
        self._tecnhician = tecnichian

    @property
    def tecnichian(self):
        return self._tecnhician

    @property
    def date(self):
        return self._date

    def setDate(self, date):
        self._date = date

        if  not isinstance(self._date, str):
            raise ValueError("Date must be a time.time() obj or a string")










