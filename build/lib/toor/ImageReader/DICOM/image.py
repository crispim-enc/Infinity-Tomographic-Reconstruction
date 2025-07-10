# *******************************************************
# * FILE: image.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

from pydicom.dataset import Dataset, FileDataset
from dateutil.parser import parse


class GeneralImage(FileDataset):
    def __init__(self, filename_or_obj, dataset,ds):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.ImageType = ['ORIGINAL', 'PRIMARY']
        self.ds.AcquisitionDate = "00000000"
        self.ds.AcquisitionTime = "000000"
        # self.ds.ContentTime = "000000"
        self.ds.PatientOrientation = ["LE","D"]
        self.ds.InstanceNumber = "0"
        # self.ds.TimeTrigger = 0
        # self.ds.ImagesInAcquisition = 0
        self.ds.ImageComments = ""

    def override(self, acquisitionInfo,seriesInfo):
        # self.ds.AcquisitionNumber = acquisitionInfo.AcquisitionNumber

        self.ds.AcquisitionDate = parse(acquisitionInfo.AcquisitionStartTime).strftime('%Y%m%d')
        self.ds.StudyDate = parse(acquisitionInfo.AcquisitionStartTime).strftime('%Y%m%d')
        # self.ds.StudyTime =
        self.ds.AcquisitionTime = parse(acquisitionInfo.AcquisitionStartTime).strftime("%H%M%S")
        self.ds.StudyTime = parse(acquisitionInfo.AcquisitionStartTime).strftime("%H%M%S")
        self.ds.StudyDescription = "PET_{}_{}".format(acquisitionInfo.Id,
                                                      parse(acquisitionInfo.AcquisitionStartTime))

        self.ds.Columns = seriesInfo.volume.shape[0]
        self.ds.Rows = seriesInfo.volume.shape[0]
        self.ds.PatientOrientation = ["LE", "D"]


class GeneralReference(FileDataset):
    """"""


class ImagePlane(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.ImageOrientationPatient = r"0\-1\0\1\0\0"
        self.ds.SliceThickness = 1
        self.ds.PixelSpacing = [1, 1]

    def update_image_position(self, x_y, z):
        self.ds.ImagePositionPatient = [-x_y, x_y, z]

    def override(self, imageInfo, seriesInfo):
        self.ds.PixelSpacing = [imageInfo.pixelSizeXY, imageInfo.pixelSizeXY]
        self.ds.SliceThickness = seriesInfo.sliceThickness


class ImagePixel(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.SamplesPerPixel = 1
        self.ds.PhotometricInterpretation = 'MONOCHROME2'
        self.ds.BitsAllocated = 16
        self.ds.BitsStored = 16
        self.ds.HighBit = 15
        self.ds.PixelRepresentation = 0
        self.ds.RescaleIntercept = 0

    def override(self):
        """Future Implementation - Not Included"""


class Device(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.DeviceSequence = [Dataset()]
        self.ds.DeviceSequence.CodeValue = ""
        self.ds.DeviceSequence.CodingSchemeDesignator = ""
        self.ds.DeviceSequence.CodingSchemeVersion = ""
        self.ds.DeviceSequence.CodeMeaning = ""
        self.ds.DeviceSequence.MappingResourceUID = ""
        self.ds.DeviceSequence.ContextGroupVersion = ""
        self.ds.ManufacturerModelName = "EasyPET3D"
        self.ds.DeviceSerialNumber = "training0000001"
        self.ds.DeviceID = "EasyPET3D-training0000001"
        self.ds.DeviceLength = str(32*2+31*0.28)
        self.ds.DeviceDiameter = str(48)
        self.ds.DeviceDiameterUnits = "mm"
        self.ds.DeviceVolume = str(48*32*2+31*0.28*0.001)
        self.ds.InterMarkerDistance = "72.68"
        self.ds.DeviceDescription = "EasyPET the portable PET"

    def override(self,):
        """Future Implementation"""


class Specimen(FileDataset):
    """Future Implementation - Not Included"""


class PETImage(FileDataset):
    """Future Implementation - Not Included"""
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.ImageType = ['ORIGINAL', 'PRIMARY', 'OTHER']
        # self.ds.AcquisitionDate = "" # Override in General Image
        # self.ds.AcquisitionTime = "" Override in General Image
        # Required for GATED images
        # self.ds.TriggerTime = "0"
        # self.ds.NominalInterval = "0"
        # self.ds.LowRRValue = "0"
        # self.ds.HighRRValue = "0"
        # self.ds.IntervalsAcquired = "0"
        # self.ds.IntervalsRejected = "0"

        # self.ds.SamplesPerPixel = 1 # Override in General Image
        # self.ds.PhotometricInterpretation = 'MONOCHROME2' # Override in General Image
        # self.ds.BitsAllocated = 16 # Override in General Image
        # self.ds.BitsStored = 16 # Override in General Image
        # self.ds.HighBit = 15 # Override in General Image
        # self.ds.PixelRepresentation = 0 # Override in General Image
        self.ds.RescaleIntercept = 0
        self.ds.RescaleSlope = 1
        self.ds.LossyImageCompression = "00"
        self.ds.FrameReferenceTime = 0
        self.ds.PrimaryPromptsCountsAccumulated = 0
        # self.ds.SliceSensitivityFactor = 1

    def override(self, seriesInfo):
        """Future Implementation"""
        self.ds.RescaleSlope = str(seriesInfo.fc)


class OverlayPlane(FileDataset):
    """Future Implementation - Not Included"""


class VoiLUT(FileDataset):
    """Future Implementation - Not Included"""


class AcquisitionContext(FileDataset):
    """Future Implementation - Not Included"""


class SOPCommon(FileDataset):
    """Future Implementation - Not Included"""


class CommonInstanceReference(FileDataset):
    """Future Implementation - Not Included"""