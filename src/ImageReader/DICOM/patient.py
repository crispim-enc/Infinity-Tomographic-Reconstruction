import logging

from pydicom.dataset import Dataset, FileDataset
# from typing import BinaryIO, Union
# from pydicom.filebase import DicomFileLike
# from pydicom.fileutil import PathType

from src.ImageReader.DICOM.submodule import MainAttribute, SubAttribute


class Patient(FileDataset):
    # filename_or_obj: Union[PathType, BinaryIO, DicomFileLike], dataset: _DatasetType, ds = None):
    # super().__init__(filename_or_obj, dataset)
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self._default_values()

    def _default_values(self):
        self.ds.PatientID = "Unknown"
        self.ds.PatientName = "Unknown"
        self.ds.IssuerOfPatientID = "Unknown"
        self.ds.TypeOfPatientID = "TEXT"
        self.ds.PatientBirthDate = ""
        self.ds.PatientSpeciesDescription = ""
        self.ds.PatientSex = ""

    def override(self, acquisitionInfo=None):
        if acquisitionInfo is None:
            return
        self.ds.PatientName = acquisitionInfo.Id
        self.ds.PatientID = acquisitionInfo.Id
        self.ds.IssuerOfPatientID = acquisitionInfo.TypeOfSubject
        self.ds.TypeOfPatientID = "TEXT"
        if acquisitionInfo.TypeOfSubject == 'Animal':
            self.ds.PatientBirthDate = ""
            self.ds.QualityControlSubject = "NO"
            self.ds.PatientSpeciesDescription = acquisitionInfo.SpeciesDescription
            self.ds.PatientSpeciesCodeSequence = [Dataset()]
            self.ds.PatientSpeciesCodeSequence[0].CodeValue = acquisitionInfo.SpeciesCodeValue
            self.ds.PatientSpeciesCodeSequence[0].CodingSchemeDesignator = acquisitionInfo.SpeciesCodingSchemeDesignator
            self.ds.PatientSpeciesCodeSequence[0].CodingSchemeVersion = acquisitionInfo.SpeciesCodingSchemeVersion
            self.ds.PatientSpeciesCodeSequence[0].CodeMeaning = acquisitionInfo.SpeciesCodeMeaning
            self.ds.PatientSpeciesCodeSequence[0].LongCodeValue = acquisitionInfo.SpeciesLongCodeValue
            self.ds.PatientSpeciesCodeSequence[0].URNCodeValue = acquisitionInfo.SpeciesUrnCodeValue

            self.ds.PatientBreedDescription = acquisitionInfo.BreedDescription
            self.ds.PatientBreedCodeSequence = [Dataset()]
            self.ds.PatientBreedCodeSequence[0].CodeValue = acquisitionInfo.BreedCodeValue
            self.ds.PatientBreedCodeSequence[0].CodingSchemeDesignator = acquisitionInfo.BreedCodingSchemeDesignator
            self.ds.PatientBreedCodeSequence[0].CodingSchemeVersion = acquisitionInfo.BreedCodingSchemeVersion
            self.ds.PatientBreedCodeSequence[0].CodeMeaning = acquisitionInfo.BreedCodeMeaning
            self.ds.PatientBreedCodeSequence[0].LongCodeValue = acquisitionInfo.BreedLongCodeValue
            self.ds.PatientBreedCodeSequence[0].URNCodeValue = acquisitionInfo.BreedUrnCodeValue

            self.ds.BreedRegistrationSequence = [Dataset()]
            self.ds.BreedRegistrationSequence[0].BreedRegistrationNumber = acquisitionInfo.BreedResgNumber
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence = [Dataset()]
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence[0].CodeValue = \
                acquisitionInfo.BreedResgCodeValue
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence[0].CodingSchemeDesignator = \
                acquisitionInfo.BreedResgCodingSchemeDesignator
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence[0].CodingSchemeVersion = \
                acquisitionInfo.BreedResgCodingSchemeVersion
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence[0].CodeMeaning = \
                acquisitionInfo.BreedResgCodeMeaning
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence[0].LongCodeValue =\
                acquisitionInfo.BreedResgLongCodeValue
            self.ds.BreedRegistrationSequence[0].BreedRegistryCodeSequence[0].URNCodeValue =\
                acquisitionInfo.BreedResgUrnCodeValue

            self.ds.ResponsiblePerson = acquisitionInfo.AnimalWelfareResponsiblePerson
            if acquisitionInfo.AnimalWelfareResponsiblePersonRole is None:
                acquisitionInfo.AnimalWelfareResponsiblePersonRole = "None"
            self.ds.ResponsiblePersonRole = acquisitionInfo.AnimalWelfareResponsiblePersonRole.upper()
            self.ds.ResponsibleOrganization = acquisitionInfo.AnimalWelfareResponsibleOrganization

        elif acquisitionInfo.TypeOfSubject == 'Phantom':
            self.ds.QualityControlSubject = "YES"

        self.ds.PatientPosition = acquisitionInfo.Position


class Patient_not:
    def __int__(self):
        """
        For future implementation in case you need to "
        This module specifies the Attributes of the Patient that describe and identify the Patient who is the subject
        of a Study. This Module contains Attributes of the Patient that are needed for interpretation of the Composite
        Instances and are common for all Studies performed on the Patient. It contains Attributes that are also included
        in the Patient Modules in Section C.2.
        """
        # Default values, protect members
        self._patientsName = MainAttribute(value="Mice Doe", tag="(0010,0022)", type="Optional", dependencies=None)
        self._patientID = MainAttribute(value="Name Given by the Institution", tag="(0010,0022)", type="Optional",
                                        dependencies=None)
        self._issuerOfPatientID = MainAttribute(value="Mouse", tag="(0010,0022)", type="Optional", dependencies=None)

        self._typeOfPatientID = MainAttribute(value="TEXT", tag="(0010,0022)", type="Optional", dependencies=None)

        self._issuerOfPatientIDQualifiersSequence = None
        self._sourceOfGroupIdentificionSequence = None
        self._groupOfPatientIdentificationSequence = None
        self._patientsBirthDate = None
        self._patientBirthTime = None
        self._patientSex = None
        self._qualityControlSubject = None
        self._strainDescription = None
        self._strainNomenclature = None
        self._strainStrockSequence = None
        self._strainAdditionalInformation = None
        self._strainCodeSequence = None
        self._geneticModificationsSequence = None
        self._otherPatientNames = None
        self._otherPatientIDsSequence = None
        self._patientSpeciesDescription = None
        self._patientSpeciesCodeSequence = SubAttribute(value="Mouse", tag="(0010,0022)", type_="Optional",
                                                        dependencies=None, codevalue="", codingschemedesignator="",
                                                        codingschemeversion="", codemeaning="",
                                                        longcodevalue=" ", urncodevalue="")
        self._patientBreedDescription = " "
        self._patientBreedCodeSequence = SubAttribute(value="Mouse", tag="(0010,0022)", type_="Optional",
                                                      dependencies=None, codevalue="", codingschemedesignator="",
                                                      codingschemeversion="", codemeaning="",
                                                      longcodevalue=" ", urncodevalue="")

        self._breedRegistrationSequence = SubAttribute(value="Mouse", tag="(0010,0022)", type_="Optional",
                                                       dependencies=None, codevalue="", codingschemedesignator="",
                                                       codingschemeversion="", codemeaning="",
                                                       longcodevalue=" ", urncodevalue="")
        self._responsiblePerson = None
        self._responsiblePersonRole = "Director"
        self._responsibleOrganization = None
        self._patientComents = None
        self._patientIdentityRemoved = "NO"
        self.typeOfSubject = None

    def updateVariables(self, imageInfo, acquisitionInfo):
        self._typeOf_subject = acquisitionInfo['Type of subject']
        # self.patientsName()

    def updateDsFile(self, ds, acquisitionInfo):
        ds.PatientID = self._patientID["value"]
        ds.PatientBirthDate = self._patientsBirthDate
        ds.PatientSex = self._patientSex
        if self._typeOf_subject == 'Animal':
            logging.info('Animal')
            ds.PatientSpeciesDescription = self._patientSpeciesDescription.value
            ds.PatientSpeciesCodeSequence = Dataset()
            ds.PatientSpeciesCodeSequence.CodeValue = self._patientSpeciesCodeSequence.codeValue()
            ds.PatientSpeciesCodeSequence.CodingSchemeDesignator = \
                self._patientSpeciesCodeSequence["CodingSchemeDesignator"]
            ds.PatientSpeciesCodeSequence.CodingSchemeVersion = self._patientSpeciesCodeSequence["CodingSchemeVersion"]
            ds.PatientSpeciesCodeSequence.CodeMeaning = self._patientSpeciesCodeSequence["CodeMeaning"]
            ds.PatientSpeciesCodeSequence.LongCodeValue = self._patientSpeciesCodeSequence["LongCodeValue"]
            ds.PatientSpeciesCodeSequence.URNCodeValue = self._patientSpeciesCodeSequence["URNCodeValue"]

            ds.PatientBreedDescription = self._patientBreedDescription

            ds.PatientBreedCodeSequence = Dataset()
            ds.PatientBreedCodeSequence.CodeValue = self._patientBreedCodeSequence["value"]["CodeValue"]
            ds.PatientBreedCodeSequence.CodingSchemeDesignator = \
                self._patientBreedCodeSequence["CodingSchemeDesignator"]
            ds.PatientBreedCodeSequence.CodingSchemeVersion = self._patientBreedCodeSequence["CodingSchemeVersion"]
            ds.PatientBreedCodeSequence.CodeMeaning = self._patientBreedCodeSequence["CodeMeaning"]
            ds.PatientBreedCodeSequence.LongCodeValue = self._patientBreedCodeSequence["LongCodeValue"]
            ds.PatientBreedCodeSequence.URNCodeValue = self._patientBreedCodeSequence["URNCodeValue"]

            ds.BreedRegistrationSequence = Dataset()
            ds.BreedRegistrationSequence.BreedRegistrationNumber = self._breedRegistrationSequence

            ds.BreedRegistrationSequence.BreedRegistryCodeSequence = Dataset()
            ds.BreedRegistrationSequence.BreedRegistryCodeSequence.CodeValue = acquisitionInfo[
                "Breed Resg Code Value "]
            ds.BreedRegistrationSequence.BreedRegistryCodeSequence.CodingSchemeDesignator = \
                acquisitionInfo["Breed Resg Coding Scheme Designator "]
            ds.BreedRegistrationSequence.BreedRegistryCodeSequence.CodingSchemeVersion = \
                acquisitionInfo["Breed Resg Coding Scheme version "]
            ds.BreedRegistrationSequence.BreedRegistryCodeSequence.CodeMeaning = acquisitionInfo[
                "Breed Resg Code Meaning "]
            ds.BreedRegistrationSequence.BreedRegistryCodeSequence.LongCodeValue = acquisitionInfo[
                "Breed Resg long Code Value "]
            ds.BreedRegistrationSequence.BreedRegistryCodeSequence.URNCodeValue = acquisitionInfo[
                "Breed Resg URN Code Value "]

            ds.ResponsiblePerson = acquisitionInfo['Animal welfare - responsible person']
            ds.ResponsiblePersonRole = acquisitionInfo['Animal welfare - responsible person role']
            ds.ResponsibleOrganization = acquisitionInfo['Animal welfare - responsible organization']

        return ds

    def patientsName(self, value=None):
        """Patient's full name."""
        self._patientsName.value(value=value)
        return self._patientsName

    def patientID(self, new_value=None):
        """
        Primary identifier for the Patient.
        Note: In the case of imaging a group of small animals simultaneously, the single value of this identifier
        corresponds to the
        identification of the entire group. See also Section C.7.1.4.1.1
        see: https://dicom.innolitics.com/ciods/pet-image/patient/00100020

        Tag	(0010,0020)
        Type	Required, Empty if Unknown (2)
        Keyword	PatientID
        Value Multiplicity	1
        Value Representation	Long String (LO)
        """

    def issuerOfPatientID(self):
        """Identifier of the Assigning Authority (system, organization, agency, or department) that issued
        the Patient ID."""

    def typeOfPatientID(self):
        """The type of identifier in the Patient ID (0010,0020).
        Tag	(0010,0022)
        Type	Optional (3)
        Keyword	TypeOfPatientID
        Value Multiplicity	1
        Value Representation	Code String (CS)"""


