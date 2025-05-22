#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: general_data
# * AUTHOR: Pedro Encarnação
# * DATE: 21/05/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************


class GeneralData:
    def __init__(self):
        self._dataDescription = None
        self._imagingModality = None
        self._patientName = None
        self._patientID = None
        self._patientSex = None
        self._patientBirthDate = None
        self._studyID = None
        self._acquisitionDate = None
        self._acquisitionTime = None
        self._operatorName = None
        self._institutionName = None

    def setDataDescription(self, dataDescription: str) -> None:
        """
        Set the data description.
        """
        self._dataDescription = dataDescription
        print(f"Data description set to: {self._dataDescription}")

    @property
    def dataDescription(self):
        """
        Get the data description.
        """
        return self._dataDescription

    def setImagingModality(self, imagingModality: str) -> None:
        """
        Set the imaging modality.
        """
        self._imagingModality = imagingModality
        print(f"Imaging modality set to: {self._imagingModality}")

    @property
    def imagingModality(self):
        """
        Get the imaging modality.
        """
        return self._imagingModality

