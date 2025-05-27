import numpy as np
import uproot


class ExtractSingles:
    """
    This class is used to extract the singles data from the root file and to convert into numpy array.
    args:
        parent: the uproot object containing the singles data
    parameters:
        _parent: the uproot object containing the singles data
        _runID: the run ID
        _eventID: the event ID
        _sourceID: the source ID
        _sourcePosX: the source position in x
        _sourcePosY: the source position in y
        _sourcePosZ: the source position in z
        _globalPosX: the global position in x
        _globalPosY: the global position in y
        _globalPosZ: the global position in z
        _time: the time
        _energy: the energy
        _baseID: the base ID
        _level1ID: the level 1 ID (moduleID)
        _level2ID: the level 2 ID (submoduleID)
        _level3ID: the level 3 ID (crystalID
        _level4ID: the level 4 ID (pixelID)
        _level5ID: the level 5 ID (?)
        _comptonPhantom: the compton phantom
        _comptonCrystal: the compton crystal
        _comptonDetector: the compton detector
        _comptonWorld: the compton world
    methods:
        _createAllTreeArrays: create all the tree arrays from the root file to numpy arrays
        runID: get the run ID
        setRunID: set the run ID
        eventID: get the event ID
        setEventID: set the event ID
        sourceID: get the source ID
        setSourceID: set the source ID
        sourcePosX: get the source position in x
        setSourcePosX: set the source position in x
        sourcePosY: get the source position in y
        setSourcePosY: set the source position in y
        sourcePosZ: get the source position in z
        setSourcePosZ: set the source position in z
        globalPosX: get the global position in x
        setGlobalPosX: set the global position in x
        globalPosY: get the global position in y
        setGlobalPosY: set the global position in y
        globalPosZ: get the global position in z
        setGlobalPosZ: set the global position in z
        time: get the time
        setTime: set the time
        energy: get the energy
        setEnergy: set the energy
        baseID: get the base ID
        setBaseID: set the base ID
        level1ID: get the level 1 ID (moduleID)
        setLevel1ID: set the level 1 ID (moduleID)
        level2ID: get the level 2 ID (submoduleID)
        setLevel2ID: set the level 2 ID (submoduleID)
        level3ID: get the level 3 ID (crystalID)
        setLevel3ID: set the level 3 ID (crystalID)
        level4ID: get the level 4 ID (pixelID)
        setLevel4ID: set the level 4 ID (pixelID)
        level5ID: get the level 5 ID (?)
        setLevel5ID: set the level 5 ID (?)
        comptonPhantom: get the compton phantom
        setComptonPhantom: set the compton phantom
        comptonCrystal: get the compton crystal
        setComptonCrystal: set the compton crystal
        comptonDetector: get the compton detector
        setComptonDetector: set the compton detector
        comptonWorld: get the compton world

    """

    def __init__(self, parent=None):
        self._parent = parent
        self._runID = None
        self._eventID = None
        self._sourceID = None
        self._sourcePosX = None
        self._sourcePosY = None
        self._sourcePosZ = None
        self._globalPosX = None
        self._globalPosY = None
        self._globalPosZ = None
        self._time = None
        self._energy = None
        self._baseID = None
        self._level1ID = None  # moduleID
        self._level2ID = None  # submoduleID
        self._level3ID = None  # crystalID
        self._level4ID = None  # pixelID ?
        self._level5ID = None  # ?
        self._comptonPhantom = None
        self._comptonCrystal = None
        self._comptonDetector = None
        self._comptonWorld = None
        self.keys = self._parent.keys()

    @property
    def parent(self):
        return self._parent

    # def createAllTreeArrays(self):
    #     self.setRunID
    #     self.setEventID
    #     self.setSourceID
    #     self.setSourcePosX
    #     self.setSourcePosY
    #     self.setSourcePosZ
    #     self.setGlobalPosX
    #     self.setGlobalPosY
    #     self.setGlobalPosZ
    #     self.setTime
    #     self.setEnergy
    #     self.setBaseID
    #     self.setLevel1ID
    #     self.setLevel2ID
    #     self.setLevel3ID
    #     self.setLevel4ID
    #     self.setLevel5ID
    #     self.setComptonPhantom
    #     self.setComptonCrystal

    @property
    def runID(self):
        return self._runID

    @runID.setter
    def setRunID(self, arr=None, _type=np.int32):
        if arr is None:
            self._runID = np.array(self._parent['runID'], dtype=_type)
        else:
            self._runID = np.array(arr, dtype=_type)

    @property
    def eventID(self):
        return self._eventID

    @eventID.setter
    def setEventID(self, arr=None, _type=np.uint32):
        if arr is None:
            self._eventID = np.array(self._parent['eventID'], dtype=_type)
        else:
            self._eventID = np.array(arr, dtype=_type)

    @property
    def sourceID(self):
        return self._sourceID

    @sourceID.setter
    def setSourceID(self, arr=None, _type=np.int8):
        if arr is None:
            self._sourceID = np.array(self._parent['sourceID'], dtype=_type)
        else:
            self._sourceID = np.array(arr, dtype=_type)

    @property
    def sourcePosX(self):
        return self._sourcePosX

    @sourcePosX.setter
    def setSourcePosX(self, arr=None, _type=np.float32):
        if arr is None:
            self._sourcePosX = np.array(self._parent['sourcePosX'], dtype=_type)
        else:
            self._sourcePosX = np.array(arr, dtype=_type)

    @property
    def sourcePosY(self):
        return self._sourcePosY

    @sourcePosY.setter
    def setSourcePosY(self, arr=None, _type=np.float32):
        if arr is None:
            self._sourcePosY = np.array(self._parent['sourcePosY'], dtype=_type)
        else:
            self._sourcePosY = np.array(arr, dtype=_type)

    @property
    def sourcePosZ(self):
        return self._sourcePosZ

    @sourcePosZ.setter
    def setSourcePosZ(self, arr=None,  _type=np.float32):
        if arr is None:
            self._sourcePosZ = np.array(self._parent['sourcePosZ'], dtype=_type)
        else:
            self._sourcePosZ = np.array(arr, dtype=_type)

    @property
    def globalPosX(self):
        return self._globalPosX

    @globalPosX.setter
    def setGlobalPosX(self, arr=None, _type=np.float32):
        if arr is None:
            self._globalPosX = np.array(self._parent['globalPosX'], dtype=_type)
        else:
            self._globalPosX = np.array(arr, dtype=_type)

    @property
    def globalPosY(self):
        return self._globalPosY

    @globalPosY.setter
    def setGlobalPosY(self, arr=None, _type=np.float32):
        if arr is None:
            self._globalPosY = np.array(self._parent['globalPosY'], dtype=_type)
        else:
            self._globalPosY = np.array(arr, dtype=_type)

    @property
    def globalPosZ(self):
        return self._globalPosZ

    @globalPosZ.setter
    def setGlobalPosZ(self, arr=None, _type=np.float32):
        if arr is None:
            self._globalPosZ = np.array(self._parent['globalPosZ'], dtype=_type)
        else:
            self._globalPosZ = np.array(arr, dtype=_type)

    @property
    def baseID(self):
        return self._baseID

    @baseID.setter
    def setBaseID(self, arr=None, _type=np.int8):
        if arr is None:
            self._baseID = np.array(self._parent['baseID'], dtype=_type)
        else:
            self._baseID = np.array(arr, dtype=_type)

    @property
    def time(self):
        return self._time

    @time.setter
    def setTime(self, arr=None,  _type=np.float64):
        if arr is None:
            self._time = np.array(self._parent['time'], dtype=_type)
        else:
            self._time = np.array(arr, dtype=_type)

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def setEnergy(self, arr=None, _type=np.float32):
        if arr is None:
            self._energy = np.array(self._parent['energy'], dtype=_type)
        else:
            self._energy = np.array(arr, dtype=_type)

    @property
    def level1ID(self):
        return self._level1ID

    @level1ID.setter
    def setLevel1ID(self, arr=None, _type=np.float32):
        if arr is None:
            self._level1ID = np.array(self._parent['level1ID'], dtype=_type)
        else:
            self._level1ID = np.array(arr, dtype=_type)

    @property
    def level2ID(self):
        return self._level2ID

    @level2ID.setter
    def setLevel2ID(self, arr=None, _type=np.float32):
        if arr is None:
            self._level2ID = np.array(self._parent['level2ID'], dtype=_type)
        else:
            self._level2ID = np.array(arr, dtype=_type)

    @property
    def level3ID(self):
        return self._level3ID

    @level3ID.setter
    def setLevel3ID(self, arr=None,  _type=np.uint16):
        if arr is None:
            self._level3ID = np.array(self._parent['level3ID'], dtype=_type)
        else:
            self._level3ID = np.array(arr, dtype=_type)

    @property
    def level4ID(self):
        return self._level4ID

    @level4ID.setter
    def setLevel4ID(self, arr=None, _type=np.uint8):
        if arr is None:
            self._level4ID = np.array(self._parent['level4ID'], dtype=_type)
        else:
            self._level4ID = np.array(arr, dtype=_type)

    @property
    def level5ID(self):
        return self._level5ID

    @level5ID.setter
    def setLevel5ID(self, arr=None, _type=np.int8):
        if arr is None:
            self._level5ID = np.array(self._parent['level5ID'], dtype=_type)
        else:
            self._level5ID = np.array(arr, dtype=_type)

    @property
    def comptonPhantom(self):
        return self._comptonPhantom

    @comptonPhantom.setter
    def setComptonPhantom(self, arr=None, _type=np.int8):
        if arr is None:
            self._comptonPhantom = np.array(self._parent['comptonPhantom'], dtype=_type)
        else:
            self._comptonPhantom = np.array(arr, dtype=_type)

    @property
    def comptonCrystal(self):
        return self._comptonCrystal

    @comptonCrystal.setter
    def setComptonCrystal(self, arr=None, _type=np.int8):
        if arr is None:
            self._comptonCrystal = np.array(self._parent['comptonCrystal'], dtype=_type)
        else:
            self._comptonCrystal = np.array(arr, dtype=_type)



