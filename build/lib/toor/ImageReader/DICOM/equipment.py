# *******************************************************
# * FILE: equipment.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

from pydicom.dataset import Dataset, FileDataset


class Equipment(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.Manufacturer = "RI-TE, Lda - Radiation Imaging Technologies"
        self.ds.InstitutionName = "Unknown"
        self.ds.InstitutionAddress = "Unknown"
        self.ds.StationName = "Unknown"
        self.ds.InstitutionalDepartmentName = "Unknown"
        self.ds.ManufacturerModelName = "EasyPET3D"
        self.ds.DeviceSerialNumber = "training0001_20_07_22"
        self.ds.SoftwareVersions = "ReconstructionModel: beta_0.1"
        self.ds.SpatialResolution = "1.0"
        self.ds.DateOfLastCalibration = "20220720"
        self.ds.TimeOfLastCalibration = "123200"


class Synchronization(FileDataset):
    def __init__(self, ds):
        self.ds = ds



