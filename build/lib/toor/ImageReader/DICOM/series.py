# *******************************************************
# * FILE: series.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

from dateutil.parser import parse
from pydicom.dataset import Dataset, FileDataset
import numpy as np


class GeneralSeries(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.SeriesDate = ""
        self.ds.SeriesTime = ""
        self.ds.Modality = "PT"
        self.ds.SeriesDescription = "Unknown"
        self.ds.PerformingPhysicianName = "Unknown"
        self.ds.OperatorsName = "Unknown"
        self.ds.AnatomicalOrientationType = 'QUADRUPED'
        self.ds.BodyPartExamined = "WHOLEBODY"
        self.ds.PatientPosition = "HFS"
        self.ds.SeriesInstanceUID = ""
        self.ds.SeriesNumber = "1"
        self.ds.Laterality = None
        self.ds.SmallestPixelValueInSeries = 0
        self.ds.LargestPixelValueInSeries = 0

    def override(self, acquisitionInfo, seriesInfo, imageInfo):
        """"""
        self.ds.SeriesDate = parse(acquisitionInfo.AcquisitionStartTime).strftime('%Y%m%d')
        self.ds.SeriesTime = parse(acquisitionInfo.AcquisitionStartTime).strftime("%H%M%S")
        self.ds.SeriesNumber = seriesInfo.seriesNumber
        self.ds.SeriesInstanceUID = seriesInfo.seriesInstanceUID
        seriesDescription = "Unknown"
        if seriesInfo.current_type_of_reconstruction == "DYNAMIC":
            seriesDescription = "{} {}".format(seriesInfo.current_type_of_reconstruction, imageInfo.number_cumulative_turns)
        elif seriesInfo.current_type_of_reconstruction == "WHOLE BODY":
            seriesDescription = "{} {} to {}s".format(seriesInfo.current_type_of_reconstruction,
                                                      imageInfo.remove_turns["Init time"], imageInfo.remove_turns["End time"])
        elif seriesInfo.current_type_of_reconstruction == "STATIC":
            seriesDescription = "{} organ".format(seriesInfo.current_type_of_reconstruction)

        self.ds.SeriesDescription = seriesDescription


class PETSeries(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        # self.ds.SeriesDate = "" # Override in General Series
        # self.ds.SeriesTime = "" # Override in SeriesTime
        self.ds.AcquisitionTerminationCondition = "MANU"  # if fixed turns Time
        self.ds.AcquisitionStartCondition = "MANU"
        self.ds.ReconstructionDiameter = "48"  # real fov
        self.ds.GantryDetectorTilt = "0"  # could not be true
        self.ds.GantryDetectorSlew = "0"  # could be different of 0

        self.ds.FieldOfViewShape = "CYLINDRICAL RING"
        self.ds.FieldOfViewDimensions = [int(48), int(32*2+31*0.28)]
        self.ds.CorrectedImage = ['DECY', 'NORM', 'DCAL', 'RADL','SCAT']
        self.ds.EnergyWindowRangeSequence = [Dataset()]
        self.ds.EnergyWindowRangeSequence[0].EnergyWindowLowerLimit = 320
        self.ds.EnergyWindowRangeSequence[0].EnergyWindowUpperLimit = 720
        # self.ds.NumberOfRRIntervals = 0
        self.ds.NumberOfTimeSlices = 1
        self.ds.NumberOfSlices = 1
        self.ds.TypeOfDetectorMotion = "CONTINUOUS"
        self.ds.SeriesType = ['STATIC', 'IMAGE']
        self.ds.Units = "BQML"
        self.ds.CountsSource = "EMISSION"
        self.ds.SUVType = "BW"
        self.ds.RandomsCorrectionMethod = "None"
        self.ds.AttenuationCorrectionMethod = "None"
        self.ds.DecayCorrection = "START"
        self.ds.ReconstructionMethod = "ListMode - Unknown"
        self.ds.ScatterCorrectionMethod = "Dual-energy window"
        self.ds.AxialAcceptance = np.degrees(np.arcsin(2.28/60))
        self.ds.DetectorElementSize = "2"
        self.ds.CoincidenceWindowWidth = "40"

    def override(self, imageInfo, acquisitionInfo, seriesInfo):
        """ """
        self.ds.ReconstructionMethod = imageInfo.algorithm
        self.ds.SeriesType = [seriesInfo.seriesType, seriesInfo.seriesTypeValue2]
        if seriesInfo.seriesType == 'DYNAMIC':
            # self.ds.NumberOfTimeSlices = int(imageInfo.dynamicFrames*seriesInfo.volume.shape[2])
            self.ds.NumberOfTimeSlices = int(imageInfo.dynamicFrames)
            self.ds.NumberOfSlices = int(seriesInfo.volume.shape[2])
        else:
            self.ds.NumberOfTimeSlices = int(1)
            self.ds.NumberOfSlices = int(seriesInfo.volume.shape[2])
            self.ds.FrameReferenceTime = 0
        self.ds.ReconstructionDiameter = str(imageInfo.fov)
        # self.ds.NumberOfSlices = int(acquisitionInfo["Number of turns"]) * int(imageInfo['NumberOfSlices'])


class PETIsotope(FileDataset):
    def __init__(self, filename_or_obj, dataset, ds=None):
        super().__init__(filename_or_obj, dataset)
        self.ds = ds
        self.ds.RadiopharmaceuticalInformationSequence = [Dataset()]
        self.RadioInformationSequence = self.ds.RadiopharmaceuticalInformationSequence[0]
        self.RadioInformationSequence.Radiopharmaceutical = "Unknown"
        self.RadioInformationSequence.RadiopharmaceuticalRoute = "Unknown"
        self.RadioInformationSequence.RadiopharmaceuticalVolume = "0"
        self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclideCodeSequence = [Dataset()]
        self.RadionuclideCodeSequence = self.ds.RadiopharmaceuticalInformationSequence[0].RadionuclideCodeSequence = [Dataset()]

        self.RadionuclideCodeSequence.CodeValue = ""

    def override(self, acquisitionInfo):
        """DICOM properties """
        dt = parse(acquisitionInfo.StartDateTime.split(' ')[1])

        # Required, Empty if Unknown (2)
        self.RadioInformationSequence.Radiopharmaceutical = acquisitionInfo.Tracer
        self.RadioInformationSequence.RadiopharmaceuticalRoute = acquisitionInfo.Route
        if acquisitionInfo.VolumeTracer is None:
            acquisitionInfo.VolumeTracer = "0"
        elif isinstance(acquisitionInfo.VolumeTracer, float):
            acquisitionInfo.VolumeTracer = str(acquisitionInfo.VolumeTracer)
        elif isinstance(acquisitionInfo.VolumeTracer, int):
            acquisitionInfo.VolumeTracer = str(acquisitionInfo.VolumeTracer)
        self.RadioInformationSequence.RadiopharmaceuticalVolume = acquisitionInfo.VolumeTracer.replace(",", ".")
        if (acquisitionInfo.TotalDose is not None) and (
                acquisitionInfo.ResidualActivity is not None):
            ## ERRADO
            self.RadioInformationSequence.RadionuclideTotalDose = \
                str(float(acquisitionInfo.TotalDose) - float(acquisitionInfo.ResidualActivity))
        self.RadioInformationSequence.RadiopharmaceuticalStartDateTime = dt.strftime("%H%M%S")
        self.RadioInformationSequence.RadiopharmaceuticalSpecificActivity = acquisitionInfo.TotalDose #errado
        self.RadioInformationSequence.RadionuclidePositronFraction = acquisitionInfo.PositronFraction

        self.RadioInformationSequence.RadionuclideHalfLife = acquisitionInfo.HalfLife

        self.RadionuclideCodeSequence.CodeValue = acquisitionInfo.BatchCodeValue
        self.RadionuclideCodeSequence.CodingSchemeDesignator = acquisitionInfo.BatchCodingSchemeDesignator
        self.RadionuclideCodeSequence.CodingSchemeVersion = acquisitionInfo.BatchCodingSchemeVersion
        self.RadionuclideCodeSequence.CodeMeaning = acquisitionInfo.BatchCodeMeaning
        self.RadionuclideCodeSequence.LongCodeValue = acquisitionInfo.BatchLongCodeValue
        self.RadionuclideCodeSequence.URNCodeValue = acquisitionInfo.BatchUrnCodeValue


class PETMultiGatedAcquisition(FileDataset):
    """Future Implementation"""
