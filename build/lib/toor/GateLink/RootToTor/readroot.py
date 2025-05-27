import uproot
try:
    from toor.GateLink.RootToTor import ExtractSingles
except ModuleNotFoundError:
    from singlesstruct import ExtractSingles


class ReadRootFile:
    """
    Class to read root files
    args:
        filename: str
            Path to the root file
    parameters:
        _filename: str
            Path to the root file
        _partNumber: int
            Part number of the root file
        _singlesScanner1: ExtractSingles
            ExtractSingles object for the singles of the first scanner
        _singlesScanner2: ExtractSingles
            ExtractSingles object for the singles of the second scanner
        _singles: ExtractSingles
            ExtractSingles object for the singles of the single scanner
        _keysUsed: list
            List of the keys used to create the singles
        _doubleScannerFormat: bool
            True if the root file is in double scanner format

    """
    def __init__(self, filename=None):

        self._filename = filename
        self._partNumber = None
        self._singlesScanner1 = None
        self._singlesScanner2 = None
        self._singles = None
        self._keysUsed = None
        self._doubleScannerFormat = True

    @property
    def partNumber(self):
        return self._partNumber

    def setPartNumber(self, partNumber):
        self._partNumber = partNumber

    @property
    def singles(self):
        return self._singles

    @property
    def singlesScanner1(self):
        return self._singlesScanner1

    @property
    def singlesScanner2(self):
        return self._singlesScanner2

    @property
    def doubleScannerFormat(self):
        return self._doubleScannerFormat

    def setDoubleScannerFormat(self, value):
        self._doubleScannerFormat = value

    def readRoot(self):
        flat_tree = uproot.open(self._filename)

        if self._doubleScannerFormat:
            try:
                self._singlesScanner1 = ExtractSingles(flat_tree["SinglesScanner1"])
                self._singlesScanner2 = ExtractSingles(flat_tree["SinglesScanner2"])
            except AttributeError:
                print("Double format is not available")
                pass
        else:
            try:
                self._singles = ExtractSingles(flat_tree["Singles"])
            except KeyError:
                print("Singles tree not found")
                pass

    def setArraysToConvert(self, obj=None, keys=None, arr=None):
        if obj is None:
            obj = self._singles
        # print(obj)
        if keys is None:
            keys = obj.keys()
        if arr is None:
            arr = [None for i in range(len(keys))]
        i = 0
        for key in keys:
            key_upper = key[0].capitalize() + key[1:]

            if hasattr(obj, f"set{key_upper}"):
                att_setter = setattr(obj, f"set{key_upper}", arr[i])
                print(f"{key}: Converted to numpy")
                att = getattr(obj, f"{key}")
                print(f"{key}: {att}")
                i += 1
