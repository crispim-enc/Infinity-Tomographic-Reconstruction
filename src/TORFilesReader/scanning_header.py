#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: scanning_header
# * AUTHOR: Pedro Encarnação
# * DATE: 24/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************


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


class Statistics:
    def __init__(self):
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._median = None
        self._mode = None
        self._percentile = None
        self._histogram = None
        self._histogramBins = None







