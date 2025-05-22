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
import numpy as np


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
        self._numberOfEventsPerFramePerSecond = []
        self._minDiff = []
        self._uniqueValues = []
        self._uniqueValuesCount = []
        self._numberOfMotors = None
        self._frameStartIndexes = None
        self._coincidenceTimeDiff = None
        self._globalDetectorID = None
        self._countsPerGlobalID = None
        self._indexCountsPerGlobalID = None

    @property
    def countsPerGlobalID(self):
        return self._countsPerGlobalID

    @property
    def indexCountsPerGlobalID(self):
        return self._indexCountsPerGlobalID

    def setCountsPerGlobalID(self):
        globalID, indexCountsPerGlobalID, counts = np.unique(self._globalDetectorID, return_index=True, return_counts=True)
        self._indexCountsPerGlobalID = indexCountsPerGlobalID
        self._countsPerGlobalID = np.ascontiguousarray(counts, dtype=np.int32)

    def setListModeHistogramHybridMode(self):
        originalListmode = self._listmode[self._indexCountsPerGlobalID, :]
        self._listmode = np.zeros((self._countsPerGlobalID.shape[0], len(self._listmodeFields) + 1))
        self._listmode[:, :-1] = originalListmode
        self._listmode[:, -1] = self._countsPerGlobalID
        self._listmodeFields.append("Counts")

    @property
    def globalDetectorID(self):
        return self._globalDetectorID

    def setGlobalDetectorID(self, globalDetectorID=None):
        # Needs to run after generateStatistics
        if globalDetectorID is None:
            print("Global detector ID not set. Automatically setting is going to run. Note that incomplete data could set a wrong global ID leading to incorrect reconstruction")
            print("Number of motors detected: ", self._numberOfMotors)

            acceptableFields = ["motor", "Motor", "MOTOR"]
            keys_with_motor = [key for key in self._listmodeFields if any(acceptableField in key for acceptableField in acceptableFields)]
            step_size = []
            range_motor = []
            for motor in keys_with_motor:
                step_ = self._minDiff[self._listmodeFields.index(motor)]
                range_ = self.uniqueValuesCount[self._listmodeFields.index(motor)]
                print(motor, " step", step_)
                print(motor, " range", range_)
                step_size.append(step_)
                range_motor.append(range_)

            acceptableFields = ["ID", "id", "Id"]
            keys_with_ID = [key for key in self._listmodeFields if any(acceptableField in key for acceptableField in acceptableFields)]
            range_ID = []
            for ID in keys_with_ID:
                range_ = self.uniqueValuesCount[self._listmodeFields.index(ID)]
                print(ID, " number of detectors", range_)
                range_ID.append(range_)

            motor_angle_normalized_0 = self._listmode[:, self._listmodeFields.index(keys_with_motor[0])] + np.abs(self._listmode[:, self._listmodeFields.index(keys_with_motor[0])].min())
            motor_angle_normalized_1 = self._listmode[:, self._listmodeFields.index(keys_with_motor[1])] + np.abs(self._listmode[:, self._listmodeFields.index(keys_with_motor[1])].min())
            globalID = (motor_angle_normalized_0 / step_size[0] +
                        range_motor[0] * motor_angle_normalized_1 / step_size[1] +
                        range_motor[1] * range_motor[0] * self._listmode[:, self._listmodeFields.index(keys_with_ID[0])] +
                        range_ID[0] * range_motor[1] * range_motor[0] * (self._listmode[:, self._listmodeFields.index(keys_with_ID[1])]))
            self._globalDetectorID = globalID.astype(int)

            print("GlobalID_maximum: ", self._globalDetectorID.max())
            print("GlobalID_minimum: ", self._globalDetectorID.min())
            print("Expected GlobalID maximum: ", range_motor[0]*range_motor[1]*range_ID[0]*range_ID[1])


    @property
    def listmode(self):
        return self._listmode

    def setListmode(self, listmode, regenerateStats=False):
        """
        Set the listmode data
        """
        self._listmode = listmode
        if regenerateStats:
            self.regenerateStats()

    def regenerateStats(self):
        # listModeBody.setFrameStartIndexes(scanHeader.indexesOfFrames) " FuTure implementation
        self.generateStatistics()
        self.printStatistics()
        self.setGlobalDetectorID()
        self.setCountsPerGlobalID()

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

    @property
    def mean(self):
        return self._mean

    def setMeanValues(self):
        self._mean = np.mean(self._listmode, axis=0)

    @property
    def std(self):
        return self._std

    def setStdValues(self):
        self._std = np.std(self._listmode, axis=0)

    @property
    def min(self):
        return self._min

    def setMinValues(self):
        self._min = np.min(self._listmode, axis=0)

    @property
    def max(self):
        return self._max

    def setMaxValues(self):
        self._max = np.max(self._listmode, axis=0)

    @property
    def median(self):
        return self._median

    def setMedianValues(self):
        self._median = np.median(self._listmode, axis=0)

    @property
    def numberOfEvents(self):
        return self._numberOfEvents

    def setNumberOfEvents(self):
        self._numberOfEvents = len(self._listmode)

    @property
    def numberOfEventsPerSecond(self):
        # find indexes of the collumn time array in self._listmodefields

        return self._numberOfEventsPerSecond

    def setNumberOfEventsPerSecond(self):
        acceptableFields = ["TIME", "time", "Time"]
        timeIndex = None
        for i, field in enumerate(self._listmodeFields):
            if field in acceptableFields:
                timeIndex = i
                break
        if timeIndex is None:
            raise ValueError("Time field not found in listmode fields")
        self._numberOfEventsPerSecond = self._numberOfEvents / self._listmode[-1, timeIndex]

    @property
    def numberOfEventsPerFramePerSecond(self):

        return self._numberOfEventsPerFramePerSecond

    def setNumberOfEventsPerFramePerSecond(self):
        if self._frameStartIndexes is None:
            raise ValueError("Frame start indexes not set")

        for i in range(len(self._frameStartIndexes) - 1):
            time_frame = self._listmode[self._frameStartIndexes[i + 1], self._listmodeFields.index("TIME")] - \
                         self._listmode[self._frameStartIndexes[i], self._listmodeFields.index("TIME")]
            self._numberOfEventsPerFramePerSecond.append((self._frameStartIndexes[i + 1] - self._frameStartIndexes[
                i]) / time_frame)

    @property
    def frameStartIndexes(self) -> list:
        return self._frameStartIndexes

    def setFrameStartIndexes(self, frameStartIndexes):
        self._frameStartIndexes = frameStartIndexes

    @property
    def numberOfMotors(self):
        return self._numberOfMotors

    def setNumberOfMotors(self):
        # check if th fields include "motor" string
        self._numberOfMotors = 0
        acceptableFields = ["motor", "Motor", "MOTOR"]
        for field in self._listmodeFields:
            for acceptableField in acceptableFields:
                if acceptableField in field:
                    self._numberOfMotors += 1

    @property
    def minDiff(self):
        return self._minDiff

    def setMinDiff(self):
        for i in range(len(self._listmodeFields)):
            if len(self._uniqueValues[i]) > 1:
                self._minDiff.append(np.abs(np.min(np.diff(self._uniqueValues[i]))))
            elif len(self._uniqueValues[i]) <= 1:
                self._minDiff.append(None)

    @property
    def uniqueValues(self):
        return self._uniqueValues

    def setUniqueValues(self):
        for i in range(len(self._listmodeFields)):
            unique_values, counts = np.unique(np.round(self._listmode[:, i], 4), return_counts=True)
            self._uniqueValues.append(unique_values)

    @property
    def uniqueValuesCount(self):
        return self._uniqueValuesCount

    def setUniqueValuesCount(self):
        for i in range(len(self._listmodeFields)):
            self._uniqueValuesCount.append(len(self._uniqueValues[i]))


    def generateStatistics(self):
        """
        Generate statistics for the listmode data
        """
        self.setMeanValues()
        self.setStdValues()
        self.setMinValues()
        self.setMaxValues()
        self.setMedianValues()
        self.setNumberOfEvents()
        self.setNumberOfMotors()
        self.setUniqueValues()
        self.setUniqueValuesCount()
        self.setMinDiff()

        self.setNumberOfEventsPerFramePerSecond()
        self.setNumberOfEventsPerSecond()

    def printStatistics(self):
        """
        Print the statistics of the listmode data
        """
        for field in self._listmodeFields:
            print(f"Field: {field}")
            print("...............................")
            print(f"Mean: {np.round(self._mean[self._listmodeFields.index(field)],4)}")
            print(f"Std: {np.round(self._std[self._listmodeFields.index(field)], 4)}")
            print(f"Range: {np.round(self._min[self._listmodeFields.index(field)],5)} to {np.round(self._max[self._listmodeFields.index(field)],5)}")
            print(f"Median: {np.round(self._median[self._listmodeFields.index(field)],5)}")
            if self._minDiff[self._listmodeFields.index(field)] is not None:
                print(f"Min diff: {np.round(self._minDiff[self._listmodeFields.index(field)],10)}")
            else:
                print("Min diff: None")
            # print(f"Unique values: {self._uniqueValues[self._listmodeFields.index(field)]}")
            print(f"Number of unique values: {self._uniqueValuesCount[self._listmodeFields.index(field)]}")
            print("-------------------------------\n")

        print(f"Number of events: {self._numberOfEvents}")
        print(f"Number of events per second: {self._numberOfEventsPerSecond}")
        print(f"Number of events per frame per second: {self._numberOfEventsPerFramePerSecond}")
        print(f"Number of motors: {self._numberOfMotors}")

    def saveVarsToFile(self, filename):
        """

        """
        with open(filename, "w") as file:
            self.printStatistics()

    def __str__(self):
        """
        String representation of the ListModeBody object
        """
        return f"ListModeBody with {self.numberOfEvents} events"

    def __repr__(self):
        return f"ListModeBody with {self.numberOfEvents} events"

    def __len__(self):
        """
        Get the number of events in the listmode data
        """
        return self.numberOfEvents

    def __getitem__(self, key):
        """
        Get an item from the listmode data
        """
        if isinstance(key, str):
            return self._listmode[:, self._listmodeFields.index(key)]
        return self._listmode[key]

    def __iter__(self):
        """
        Iterate over the listmode data
        """
        return iter(self._listmode)

    def __contains__(self, item):
        """
        Check if the item is in the listmode data
        """
        return item in self._listmode
