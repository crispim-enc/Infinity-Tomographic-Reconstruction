import os
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian
from datetime import date, timedelta
import logging
from dateutil.parser import parse
import numpy as np

from ImageReader.DICOM.patient import Patient
from ImageReader.DICOM.image import GeneralImage, GeneralReference, ImagePlane, ImagePixel, Device, PETImage
from ImageReader.DICOM.frameofreference import FrameOfReference
from ImageReader.DICOM.series import GeneralSeries, PETIsotope, PETSeries
from ImageReader.DICOM.study import GeneralStudy, PatientStudy
from ImageReader.DICOM.equipment import Equipment


class WriteDicom:
    def  __init__(self, parent=None, convert_from_raw=False, path_dcm=None, volume=None):
        if parent is None:
            return

        self.current_type_of_reconstruction = parent.current_type_of_reconstruction
        self.acquisitionInfo = Dict2Class(parent.acquisitionInfo)
        self.systemConfigurationsInfo = Dict2Class(parent.systemConfigurations_info)
        self.convert_from_raw = convert_from_raw
        if convert_from_raw:
            self.reconstructionInfo = Dict2Class(parent.raw_data_heder_info)

        else:
            self.reconstructionInfo = parent

        if path_dcm is None:
            self.path_dcm = parent.path_dcm
        else:
            self.path_dcm = path_dcm

        if volume is None:
            self.volume = np.zeros((10, 10, 10))
        else:
            self.volume = volume

        self._seriesNumber = "1"
        self._load_series_number()
        self.seriesType = self.current_type_of_reconstruction
        self.seriesTypeValue2 = "IMAGE"
        self.seriesInstanceUID = pydicom.uid.generate_uid()
        self.sliceThickness = None
        self.fc = None
        self.updateVolume(self.volume)
        self._time_id_series = 0
        # t = np.around(volume, decimals=2)

        self.filename_endian = os.path.join(self.path_dcm, str(self._seriesNumber),
                                            "IM{}_f{}_s{}ID_{}{}.dcm".format(0, 0,
                                                                           self._seriesNumber,
                                                                           self.acquisitionInfo.Id,
                                                                           self.acquisitionInfo.Tracer))
        self.file_meta = FileMetaDataset()
        self.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        self.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.128'
        self.file_meta.MediaStorageSOPInstanceUID = '1.2.840.10008.1.3.10'
        self.file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.0.100.4.0'

        self.ds = FileDataset(self.filename_endian, {}, file_meta=self.file_meta, preamble=b"\0" * 128)

        self.patient = Patient(self.filename_endian, {}, self.ds)

        self.generalStudy = GeneralStudy(self.filename_endian, {}, self.ds)
        self.patientStudy = PatientStudy(self.filename_endian, {}, self.ds)

        self.generalSeries = GeneralSeries(self.filename_endian, {}, self.ds)
        self.petSeries = PETSeries(self.filename_endian, {}, self.ds)
        self.petIsotope = PETIsotope(self.filename_endian, {}, self.ds)

        self.frameReference = FrameOfReference(self.filename_endian, {}, self.ds)

        self.generalEquipment = Equipment(self.filename_endian, {}, self.ds)

        self.generalImage = GeneralImage(self.filename_endian, {}, self.ds)
        self.imagePlane = ImagePlane(self.filename_endian, {}, self.ds)
        self.imagePixel = ImagePixel(self.filename_endian, {}, self.ds)
        self.petImage = PETImage(self.filename_endian, {}, self.ds)
        self.device = Device(self.filename_endian, {}, self.ds)

    def updateVolume(self, volume):
        self.volume = volume
        self.fc = np.max(self.volume) / 65535  # int16
        self.volume /= self.fc
        self.volume = self.volume.astype(dtype=np.uint16)
        self.sliceThickness = (self.systemConfigurationsInfo.Array_crystal_x *
                               self.systemConfigurationsInfo.Crystal_pitch_x +
                               (self.systemConfigurationsInfo.Array_crystal_x - 1) * 2 *
                               self.systemConfigurationsInfo.Reflector_interior_a_x) / self.volume.shape[2]

    @property
    def time_id_series(self):
        return self._time_id_series

    def updatetimeIdSeries(self, new_id: int):
        if self._time_id_series != new_id:
            self._time_id_series = new_id

        return self._time_id_series

    @property
    def seriesNumber(self):
        return self._seriesNumber

    def updateSeriesNumber(self, new_number: str):
        if self._seriesNumber != new_number:
            self._seriesNumber = new_number

        return self._seriesNumber

    def _load_series_number(self):
        try:
            series_number = np.load(os.path.join(self.path_dcm, "series_number.npy"))
            self.updateSeriesNumber(str(series_number))
            print(f"Series: {self._seriesNumber}")
        except FileNotFoundError:
            self.updateSeriesNumber(str(1))
        np.save(os.path.join(self.path_dcm, "series_number.npy"), int(self.seriesNumber) + 1)

    def _generate_dicom_instance(self, slice_number=0, time_id=0):
        logging.debug('START')

        self.filename_endian = os.path.join(self.path_dcm, self._seriesNumber,
                                            "IM{}_f{}_s{}ID_{}{}.dcm".format(slice_number, time_id,
                                                                             self._seriesNumber,
                                                                             self.acquisitionInfo.Id,
                                                                             self.acquisitionInfo.Tracer))

        # self.ds = FileDataset(self.filename_endian, {}, file_meta=self.file_meta, preamble=b"\0" * 128)
        self.ds.__init__(self.filename_endian, {}, file_meta=self.file_meta)

        self.ds.is_little_endian = True
        self.ds.is_implicit_VR = True
        self.ds.SOPInstanceUID = pydicom.uid.generate_uid()
        self.patient = Patient(self.filename_endian, {}, self.ds)

        self.generalStudy = GeneralStudy(self.filename_endian, {}, self.ds)
        self.patientStudy = PatientStudy(self.filename_endian, {}, self.ds)

        self.generalSeries = GeneralSeries(self.filename_endian, {}, self.ds)
        self.petSeries = PETSeries(self.filename_endian, {}, self.ds)
        self.petIsotope = PETIsotope(self.filename_endian, {}, self.ds)

        self.frameReference = FrameOfReference(self.filename_endian, {}, self.ds)

        self.generalEquipment = Equipment(self.filename_endian, {}, self.ds)

        self.generalImage = GeneralImage(self.filename_endian, {}, self.ds)
        self.imagePlane = ImagePlane(self.filename_endian, {}, self.ds)
        self.imagePixel = ImagePixel(self.filename_endian, {}, self.ds)
        self.petImage = PETImage(self.filename_endian, {}, self.ds)
        self.device = Device(self.filename_endian, {}, self.ds)
        self.default_values()
        self.patient.override(self.acquisitionInfo)

        self.generalSeries.override(self.acquisitionInfo, self, self.reconstructionInfo)
        self.petSeries.override(self.reconstructionInfo, self.acquisitionInfo, self)
        self.petIsotope.override(self.acquisitionInfo)

        self.generalImage.override(self.acquisitionInfo, self)
        self.imagePlane.override(self.reconstructionInfo, self)
        self.imagePixel.override()
        self.petImage.override(self)
        self.device.override()
        self.frameReference.override(self.acquisitionInfo)

        # self.patient_module()
        self.pet_corrections()
        # self.series_modules()
        # self.pet_isotope()
        # self.general_image()
        # self.frame_reference()
        logging.debug('END')

    def default_values(self):
        # self.ds.SamplesPerPixel = 1
        # self.ds.PhotometricInterpretation = 'MONOCHROME2'
        # self.ds.BitsAllocated = 16
        # self.ds.BitsStored = 16
        # self.ds.HighBit = 15
        # self.ds.PixelRepresentation = 0
        # self.ds.RescaleIntercept = 0
        # self.ds.RescaleSlope = 1
        # self.ds.Modality = 'PT'
        # self.ds.CollimatorType = 'NONE'
        # self.ds.Units = 'BQML' # poderá vir outras unidades
        # self.ds.CountsSource = 'EMISSION'
        self.ds.ReprojectionMethod = ""
        self.ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.128'
        self.ds.StudyID = "01245405"  # STUDY = PET
        self.ds.ReferringPhysicianName = ""
        self.ds.AccessionNumber = ""
        self.ds.DecayCorrection = "START"
        self.ds.ActualFrameDuration = None

    def pet_corrections(self):
        logging.debug('START')
        self.ds.CorrectedImage = ['DECY', 'NORM', 'DCAL', 'RADL']
        # self.ds.AttenuationCorrectionMethod = self.imageInfo['attenuation_data']
        # Extra info taht not present in standard DICOM IOD
        # self.ds.RandomsCorrectionMethod = 'NONE' if(self.imageInfo['random_correction'] != True) else 'INCOMPLETO'
        # self.ds.ScatterCorrected = 'YES' if (self.imageInfo['scatter_angle_correction']) else 'NO'
        # self.ds.AttenuationCorrected = 'YES' if (self.imageInfo['attenuation_correction']) else 'NO'
        # self.ds.DeadTimeCorrected = 'YES' if (self.imageInfo['dead_time_correction']) else 'NO'
        # self.ds.PatientMotionCorrected = 'YES' if(self.imageInfo['respiratory_movement_correction'] or
        #                                      self.imageInfo['heart_movement_correction']) else 'NO'
        # self.ds.RandomsCorrected = 'YES' if(self.imageInfo['random_correction'] or
        #                                      self.imageInfo['random_correction']) else 'NO'
        self.ds.ScatterFractionFactor = 0
        self.ds.DeadTimeFactor = 1
        self.ds.DecayFactor = 1
        self.ds.DoseCalibrationFactor = 1
        logging.debug('END')

    def write_dicom_file(self):
        logging.debug('START')
        try:
            self.acquisitionInfo.TypeOfSubject
        except AttributeError:
            # QtWidgets.QMessageBox.warning(None, "WARNING!", "It is impossible to export data. "
            #                                                 "Acquisition wasn't done with software latest version.",
            #                               QtWidgets.QMessageBox.Ok)
            return
        if self.convert_from_raw:
            dicom_folder = os.path.join(self.path_dcm, 'dicom')
            if not os.path.exists(dicom_folder):
                os.mkdir(dicom_folder)
            file_path_dcm = os.path.join(dicom_folder, str(self._seriesNumber))
        else:
            file_path_dcm = os.path.join(self.path_dcm, str(self._seriesNumber))
        if not os.path.exists(file_path_dcm):
            os.mkdir(file_path_dcm)

        image_index = 0
        for i in range(0, self.volume.shape[2]):

            if self.current_type_of_reconstruction == "WHOLE BODY" or self.current_type_of_reconstruction == "STATIC":
                image_index = i

            elif self.current_type_of_reconstruction == "DYNAMIC":
                image_index = self.volume.shape[2] * self.time_id_series + i
                time_init_frame = self.reconstructionInfo.reading_data[
                                      self.acquisitionInfo.TurnEndIndex[self.time_id_series], 6] * 1000
                time_end_frame = self.reconstructionInfo.reading_data[
                                     self.acquisitionInfo.TurnEndIndex[self.time_id_series + 1], 6] * 1000
                average_time_frame = ((time_end_frame - time_init_frame) / 2 + time_init_frame)
                self.ds.FrameReferenceTime = average_time_frame
                self.ds.ActualFrameDuration = int(time_end_frame - time_init_frame)
                self.ds.AcquisitionTime = (parse(self.acquisitionInfo.AcquisitionStartTime) +
                                           timedelta(seconds=time_init_frame / 1000)).strftime("%H%M%S")
                self.ds.ContentTime = (parse(self.acquisitionInfo.AcquisitionStartTime) +
                                       timedelta(seconds=time_init_frame / 1000)).strftime("%H%M%S")

            self._generate_dicom_instance(i, self.time_id_series)
            self.ds.InstanceNumber = "{}".format(image_index)
            self.ds.ImageIndex = image_index
            # self.ds.TimeSliceIndex = self.time_id_series

            self.ds.PixelSpacing = [self.reconstructionInfo.pixelSizeXY, self.reconstructionInfo.pixelSizeXY]
            self.ds.SliceThickness = self.sliceThickness
            self.imagePlane.update_image_position(0, self.sliceThickness * i)
            # self.imagePlane.update_image_position(0, 0)
            slice_ = self.volume[:, :, i]
            if slice_.dtype != np.uint16:
                slice_ = slice_.astype(np.uint16)
            self.ds.PixelData = slice_.tobytes()
            self.ds.save_as(self.filename_endian)

        logging.debug('END')


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            key_update = key.split(" ")
            for k in range(len(key_update)):
                key_update[k] = key_update[k].capitalize()
                print(k)
            key_update = "".join(key_update)
            key_update = key_update.replace("-", "")
            key_update = key_update.replace("%", "")
            key_update = key_update.replace("(", "")
            key_update = key_update.replace(")", "")
            key_update = key_update.replace(",", "")
            setattr(self, key_update.replace(" ", ""), my_dict[key])
