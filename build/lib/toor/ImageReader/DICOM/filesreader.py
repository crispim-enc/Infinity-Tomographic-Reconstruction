# *******************************************************
# * FILE: filesreader.py
# * AUTHOR: Pedro Encarnação
# * DATE: 2025-07-10
# * LICENSE: CC BY-NC-SA 4.0
# *******************************************************

import os
import numpy as np
import vtk
import pydicom
from pydicom import dcmread
from pydicom.fileset import FileSet
from pydicom.filereader import read_dicomdir
from pydicom.uid import generate_uid
from pydicom import config
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from os import listdir, path


def fix_sop_class(elem, **kwargs):
    if elem.tag == 0x00020002:
        # DigitalXRayImageStorageForProcessing
        elem = elem._replace(value=b"1.2.840.10008.5.1.4.1.1.1.1.1")

    return elem


class DicomVTKReader():
    def __init__(self):
        base_name = os.path.dirname(os.path.abspath(__file__))
        name_file = os.path.join(base_name, 'DICOM_test_files', 'S97210','S30','I10.DCM')
        # name_file = 'C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\PhD\\Simulations\\SimulationIPET\\IPET_DerenzoSmall_500uCi_Na22_ListMode\\whole_body\\29\\IMG108_0_series20.dcm'
        name_file = 'C:\\Users\\pedro\\Universidade de Aveiro\\Fabiana Ribeiro - iCBR\\iCBR_Acquisitions\\2022\\2022-07_V64\\Easypet Scan 27 Jul 2022 - 15h 49m 04s\\whole_body\\1\\IM0_f0_s1ID_Animal 74[18F]-FDG.dcm'
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(os.path.dirname(name_file))
        # reader.SetDirectoryName(name_file)
        # reader.SetFileName(name_file)
        reader.Update()

        self.DICOMreader = reader
        self._imagePositionPatient = reader.GetImagePositionPatient()
        self._imageOrientationPatient = reader.GetImageOrientationPatient()
        self._transferSyntaxUID = reader.GetTransferSyntaxUID()
        self._studyID = reader.GetStudyID()
        self._descriptiveName = reader.GetDescriptiveName()
        


        print('test')


class DicomReader:
    def __init__(self, _file_init=None):
        self.path_ = _file_init
        # self.path_ = 'C:\\Users\\pedro\OneDrive - Universidade de Aveiro\\PhD\\' \
        #        'Simulations\\SimulationIPET\\IPET_DerenzoSmall_500uCi_Na22_ListMode\\whole_body\\70\\IM50_f0_s69ID_ListMode.npy[18F]-FDG.dcm'
        #
        # self.path_ = 'C:\\Users\\pedro\\OneDrive - Universidade de Aveiro\\Testes\\UA-V64\\V64_SR\\' \
        #              'Easypet Scan 09 Jun 2022 - 17h 16m 54s\\dynamic_image\\15\\IM0_f0_s15ID_SR-NEMA[22]-Na .dcm'
        self._dicomHeaders = None
        self._volumes = None

    @property
    def dicomHeaders(self):
        return self._dicomHeaders

    def volumes(self):
        return self._volumes

    def readDirectory(self):
        # config.data_element_callback = fix_sop_class
        file_folder = os.path.dirname(self.path_)
        dicom_files = [path.join(file_folder, _) for _ in listdir(file_folder) if _.endswith(".dcm")]
        self._dicomHeaders = [None for i in range(len(dicom_files))]
        fs = dcmread(dicom_files[0], force=True)
        print(fs)
        self._volumes = np.zeros((fs.Columns, fs.Rows, fs.NumberOfSlices, fs.NumberOfTimeSlices), dtype=np.float32)
        print("Image Memory:",
              self._volumes.size * self._volumes.itemsize)
        for file in dicom_files:
            fs = dcmread(file, force=True)
            order = fs.ImageIndex
            k = order % fs.NumberOfSlices
            l = order // fs.NumberOfSlices
            self._dicomHeaders[order] = fs
            self._volumes[:, :, k, l] = fs.pixel_array * fs.RescaleSlope

        # fs = read_dicomdir(path)7

        # print(fs.PixelData)
        # self._volumes *= fs.RescaleSlope
        # self._volumes = self._volumes.T



        # print(fs)
        # # root_dir = Path(fs.filename).resolve().parent
        # # print(f'Root directory: {root_dir}\n')
        # # A summary of the File-set's contents can be seen when printing
        # print(fs)
        # print()
        #
        # # Iterating over the FileSet yields FileInstance objects
        # for instance in fs:
        #     # Load the corresponding SOP Instance dataset
        #     ds = instance.load()
        #     # Do something with each dataset
        #
        # # We can search the File-set
        # patient_ids = fs.find_values("PatientID")
        # for patient_id in patient_ids:
        #     # Returns a list of FileInstance, where each one represents an available
        #     #   SOP Instance with a matching *Patient ID*
        #     result = fs.find(PatientID=patient_id)
        #     print(
        #         f"PatientName={result[0].PatientName}, "
        #         f"PatientID={result[0].PatientID}"
        #     )
        #
        #     # Search available studies
        #     study_uids = fs.find_values("StudyInstanceUID", instances=result)
        #     for study_uid in study_uids:
        #         result = fs.find(PatientID=patient_id, StudyInstanceUID=study_uid)
        #         print(
        #             f"  StudyDescription='{result[0].StudyDescription}', "
        #             f"StudyDate={result[0].StudyDate}"
        #         )
        #
        #         # Search available series
        #         series_uids = fs.find_values("SeriesInstanceUID", instances=result)
        #         for series_uid in series_uids:
        #             result = fs.find(
        #                 PatientID=patient_id,
        #                 StudyInstanceUID=study_uid,
        #                 SeriesInstanceUID=series_uid
        #             )
        #             plural = ['', 's'][len(result) > 1]
        #
        #             print(
        #                 f"    Modality={result[0].Modality} - "
        #                 f"{len(result)} SOP Instance{plural}"
        #             )
        #
        # # Of course you can just get the instances directly if you know what you want
        # series_uid = "1.3.6.1.4.1.5962.1.1.0.0.0.1196533885.18148.0.118"
        # result = fs.find(SeriesInstanceUID=series_uid)
        # print(f"\nFound {len(result)} instances for SeriesInstanceUID={series_uid}")
        #
        # # We can search the actual stored SOP Instances by using `load=True`
        # # This can be useful as the DICOMDIR's directory records only contain a
        # #   limited subset of the available elements, however its less efficient
        # result = fs.find(load=False, PhotometricInterpretation="MONOCHROME1")
        # result_load = fs.find(load=True, PhotometricInterpretation="MONOCHROME1")
        # print(
        #     f"Found {len(result)} instances with "
        #     f"PhotometricInterpretation='MONOCHROME1' without loading the stored "
        #     f"instances and {len(result_load)} instances with loading"
        # )

        # # We can remove and add instances to the File-set
        # fs.add(get_testdata_file("CT_small.dcm"))
        # fs.add(get_testdata_file("MR_small.dcm"))
        # result = fs.find(StudyDescription="'XR C Spine Comp Min 4 Views'")
        # fs.remove(result)
        #
        # # To edit the elements in the DICOMDIR's File-set Identification Module
        # #   (Part 3, Annex F.3.2.1) use the following properties:
        # # (0004,1130) File-set ID
        # fs.ID = "MY FILESET"
        # # Change the File-set's UID
        # fs.UID = generate_uid()
        # # (0004,1141) File-set Descriptor File ID
        # fs.descriptor_file_id = "README"
        # # (0004,1142) Specific Character Set of File-set Descriptor File
        # fs.descriptor_character_set = "ISO_IR 100"
        #
        # # Changes to the File-set are staged until write() is called
        # # Calling write() will update the File-set's directory structure to meet the
        # #   semantics used by pydicom File-sets (if required), add/remove instances and
        # #   and re-write the DICOMDIR file
        # # We don't do it here because it would overwrite your example data
        # # fs.write()
        #
        # # Alternatively, the File-set can be copied to a new root directory
        # #   This will apply any staged changes while leaving the original FileSet
        # #   object unchanged
        # tdir = TemporaryDirectory()
        # new_fileset = fs.copy(tdir.name)
        # print(f"\nOriginal File-set still at {fs.path}")
        # root = Path(new_fileset.path)
        # print(f"File-set copied to {root} and contains the following files:")
        # # Note how the original File-set directory layout has been changed to
        # #   the structure used by pydicom
        # for p in sorted(root.glob('**/*')):
        #     if p.is_file():
        #         print(f"  {p.relative_to(root)}")

if __name__ =="__main__":
    d = DicomReader()
    d.readDirectory()
    # DicomVTKReader()