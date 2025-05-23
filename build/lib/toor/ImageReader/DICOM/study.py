from pydicom.dataset import Dataset, FileDataset


class GeneralStudy(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.StudyInstanceUID = "1.2.840.10008.5.1.4.1.1.128"

    def override(self, seriesInfo):
        self.ds.SeriesInstanceUID = seriesInfo.SeriesInstanceUID


class PatientStudy(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self._default_values()

    def _default_values(self):
        self.ds.PatientAge = "000M"
        self.ds.PatientSize = "0.0"
        self.ds.PatientWeight = "0.0"
        self.ds.PatientBodyMassIndex = "0.0"
        self.ds.PatientSexNeutered = "INTACT"
        self.ds.AdditionalPatientHistory = " "

    def override(self, acquisitionInfo):
        """  """
        self.ds.PatientSex = acquisitionInfo.sex
        self.ds.PatientWeight = acquisitionInfo.weight
        self.ds.PatientSize = acquisitionInfo.size



