#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: interfile_writer
# * AUTHOR: Pedro Encarnação
# * DATE: 16/05/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

import numpy as np


class InterfileWriter:
    def __init__(self, file_name, data):
        self._fileName = file_name
        self._data = None
        self._header = None
        self._modality = 'CT'
        self._version = '3.3'
        self._typeOfData = 'Image'
        self._totalNumberOfImages = '1'
        self._imageScaleFactor = '1'
        self._imagedataByteOrder = 'LITTLEENDIAN'
        self._numberFormat = 'float'
        self._numberOfBytesPerPixel = '4'
        self._nameOfPatient = 'Patient'
        self._numberOfDimensions = '3'
        self._matrixAxisLabel = ['x', 'y', 'z']
        self._matrixSize = [str(data.shape[0]), str(data.shape[1]), str(data.shape[2])]
        self._voxelSize = ['1', '1', '1']

    @property
    def header(self):
        return self._header


    def saveHeaderFile(self):
        """
        Save the header file in the same directory as the data file.
        Example:
            !INTERFILE :=
            name of data file := teste.v

            !GENERAL DATA :=
            data description := CT scan of thorax
            imaging modality := CT

            !GENERAL ACQUISITION PARAMETERS :=
            patient name := John Doe
            patient ID := 123456
            patient sex := M
            patient birth date := 19700101
            study ID := CT teste
            acquisition date := 20250515
            acquisition time := 093015
            operator name := Dr. Smith
            institution name := Example Medical Center

            !IMAGE DATA DESCRIPTION :=
            number of dimensions := 3
            matrix size [1] := 71
            matrix size [2] := 71
            matrix size [3] := 73
            scaling factor (mm/pixel) [1] := 0.5
            scaling factor (mm/pixel) [2] := 0.5
            scaling factor (mm/pixel) [3] := 0.5
            !number format := float
            !number of bytes per pixel := 4
            imagedata byte order := LITTLEENDIAN
            number of time frames := 1

            !DATA ACQUISITION PARAMETERS :=
            energy window lower level (keV) := 80
            energy window upper level (keV) := 140
            slice thickness (mm) := 0.5
            reconstruction diameter (mm) := 250.0
            reconstruction algorithm := LM-MLEM

            !END OF INTERFILE :=
        """
        self._header
        # Open the file in write mode
        with open(f"{self._fileName}.h", 'w') as f:
            # Write the header to the file
            for key, value in self.header.items():
                f.write("{} :={}\n".format(key, value))
            # Write the end of file marker
            f.write("!END OF INTERFILE :=\n")

