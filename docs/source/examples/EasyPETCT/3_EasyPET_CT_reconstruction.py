#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: EasyPET_CT
# * AUTHOR: Pedro Encarnação
# * DATE: 17/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
Easy_CT Reconstruction
======================
This example shows how use the TOR package to build and reconstruct a easyPET/CT system.
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pycuda.driver as cuda
from toor.Corrections.CT.Projector import PyramidalProjector
from toor.Corrections.CT import NormalizationCT, DualRotationNormalizationSystem
from toor.Optimizer import GPUSharedMemoryMultipleKernel
# from toor.ImageReader.Interfile import InterfileWriter
from toor.ImageReader.RawData import RawDataSetter


class ReconstructionEasyPETCT:
    def __init__(self, file_path=None, file_path_output=None, iterations=25, subsets=1, algorithm="MLEM",
                 voxelSize=None, radial_fov_range=None, energyregion=None, fov=35, output_format="raw",
                 algorithm_options=None):
        """
        Reconstruction Manager
        :param file_path: Path to the file
        :param iterations: Number of iterations
        :param subsets: Number of subsets
        :param algorithm: Algorithm to use LM-MLEM or LM-MRP
        """
        if algorithm_options is None:
            algorithm_options = [0.1, 3]
        if radial_fov_range is None:
            radial_fov_range = [0, 25]
        self.radial_fov_range = radial_fov_range
        if file_path is None:
            raise FileExistsError
        if file_path_output is None:
            file_path_output = os.path.dirname(file_path)
        if voxelSize is None:
            voxelSize = [17, 0.8, 0.8]

        self.voxelSize = voxelSize

        self._normalizationMatrix = None
        self.ctx = None  # Context of the GPU
        self.device = None  # Device of the GPU
        self.cuda = cuda  # Cuda driver
        self.cuda.init()  # Initialize the cuda driver
        self.filePath = file_path  # Path to the file
        self.filePathOutput = file_path_output
        self._fov = fov  # Field of view
        self._fileOutputType = output_format  # Output file type
        self.iterations = iterations  # Number of iterations
        self.subsets = subsets  # Number of subsets
        self.algorithm = algorithm  # Algorithm to use
        self.algorithm_options = algorithm_options  # Algorithm options
        self.lastImageReconstructed = None  # Last image reconstructed
        self.saved_image_by_iteration = True
        self.energyRegion = energyregion

        self.ToRFile_reader = ToRFile(filepath=self.filePath)
        self.ToRFile_reader.read()

        folder = self.filePathOutput
        # find the bigger number in the folder ex."1"
        # check if a series numpy number exist in the folder. If not create a new one. If yes, create a new one with a higher number
        series_file = "seriesID.npy"

        series_file_path = os.path.join(folder, series_file)
        if os.path.exists(series_file_path):
            with open(series_file_path, "rb") as f:
                self._seriesNumber = np.load(f)
            self._seriesNumber += 1
            np.save(series_file_path, self._seriesNumber)
        else:
            self._seriesNumber = 0
            np.save(series_file_path, self._seriesNumber)
        folder = os.path.join(folder, str(self._seriesNumber))
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.filePathOutput = folder

        plt.figure()
        plt.hist(self.ToRFile_reader.fileBodyData["ENERGYB"], bins=500)

        energyRegionKeysAvailable = self.ToRFile_reader.calibrations.systemSensitivity.fields
        if not energyregion in energyRegionKeysAvailable:
            print("Energy Region: ", energyregion, " is not available in the file")
            print("Energy Regions available (Choose exactly: )",
                  self.ToRFile_reader.calibrations.systemSensitivity.fields)
            raise ValueError("Energy Region: ", energyregion, " is not available in the file")
        self._indexEnergyRegion = self.ToRFile_reader.calibrations.systemSensitivity.fields.index(energyregion)
        energyMask = (self.ToRFile_reader.fileBodyData["ENERGYB"] <=
                      self.ToRFile_reader.calibrations.systemSensitivity.energyWindows[self._indexEnergyRegion][1]) & (
                             self.ToRFile_reader.fileBodyData["ENERGYB"] >=
                             self.ToRFile_reader.calibrations.systemSensitivity.energyWindows[self._indexEnergyRegion][
                                 0])

        self.ToRFile_reader.fileBodyData.setListmode(self.ToRFile_reader.fileBodyData.listmode[energyMask], regenerateStats=True)
        self.ToRFile_reader.fileBodyData.setListModeHistogramHybridMode()
        plt.hist(self.ToRFile_reader.fileBodyData["ENERGYB"], bins=500)
        plt.savefig(os.path.join(self.filePathOutput, "energy_histogram.png"))
        plt.close()

        self.projector = PyramidalProjector(voxelSize=voxelSize, FovRadialStart=radial_fov_range[0],
                                                 FovRadialEnd=radial_fov_range[1], fov=self._fov, only_fov=True)

        self.lastImageReconstructed = None


        # self.ToRFile_reader.fileBodyData.saveVariablesTxT(filename=folder)

    def saveVariablesTxT(self):
        """
        Save the variables in a txt file
        :return:
        """

        # print variables of acquisition info
        folder = os.path.join(self.filePathOutput, "Info.txt")
        with open(folder, "w") as f:
            f.write("\t\tReconstruction Parameters\n")
            f.write("__________________________________________________\n")
            f.write("Iterations: " + str(self.iterations) + "\n")
            f.write("Subsets: " + str(self.subsets) + "\n")
            f.write("Algorithm: " + self.algorithm + "\n")
            f.write("Voxel Size: " + str(self.voxelSize) + "\n")
            f.write("Radial FOV Range: " + str(self._fov) + "\n")
            f.write("Energy Region: " + str(self.energyRegion) + "\n")
            f.write("Energy Window: " + str(self.ToRFile_reader.calibrations.systemSensitivity.energyWindows) + "\n")
            if self.lastImageReconstructed is not None:
                f.write("Image Shape: " + str(self.lastImageReconstructed.shape) + "\n")
            # Print the dict of the class
            f.write("\n\n")
            f.write("\t\tAcquisition Info\n")
            f.write("__________________________________________________\n")
            for key, value in self.ToRFile_reader.acquisitionInfo.__dict__.items():

                f.write(f"{key}: {value}\n")
                # if value is an object, print the dict of that class
                if hasattr(value, "__dict__"):
                    for key2, value2 in value.__dict__.items():
                        f.write(f"\t{key2}: {value2}\n")

            f.write("\n\n")
            f.write("\t\tCalibrations\n")
            f.write("__________________________________________________\n")
            for key3, value3 in self.ToRFile_reader.calibrations.__dict__.items():
                f.write(f"{key3}: {value3}\n")
                # if value is an object, print the dict of that class
                if hasattr(value3, "__dict__"):
                    for key4, value4 in value3.__dict__.items():
                        f.write(f"\t{key4}: {value4}\n")

            f.write("\n\n")
            f.write("\t\tSystem Info\n")
            f.write("__________________________________________________\n")
            for key5, value5 in self.ToRFile_reader.systemInfo.__dict__.items():
                f.write(f"{key5}: {value5}\n")
                # if value is an object, print the dict of that class
                if hasattr(value5, "__dict__"):
                    for key6, value6 in value5.__dict__.items():
                        f.write(f"\t{key6}: {value6}\n")


    def start(self):
        print("Starting reconstruction")
        print("________________________________")
        self.ctx = cuda.Device(0).make_context()  # Create the context
        self.device = self.ctx.get_device()
        self.generateNormalization()
        self.generateImage()
        self.ctx.detach()
        print("Context detached")
        print("Reconstruction finished")
        print("________________________________")

    def generateNormalization(self):
        """


        """
        print("Starting Normalization")
        print("________________________________")
        normalization = DualRotationNormalizationSystem(self.ToRFile_reader)
        normalization.printMotorVariables()
        normalization.setEnergyPeakKey(self.energyRegion)
        listModeForNormalization = normalization.normalizationLM()

        systemInfo = self.ToRFile_reader.systemInfo
        # systemInfo.xRayProducer.setFocalSpotInitialPositionWKSystem([12.55, 0, 0]) #Apagar
        # nb_eventstest = 2
        # normalization.reading_data = normalization.reading_data[:nb_eventstest]
        # normalization.reading_data[:, 0] = np.arange(0,360, 360/nb_eventstest)
        # normalization.reading_data[:, 1] = np.zeros(nb_eventstest)
        # normalization.reading_data[:, 2] = 0
        axialMotorData = listModeForNormalization[:,normalization.fieldsListMode.index("AXIAL_MOTOR")]
        fanMotorData = listModeForNormalization[:,normalization.fieldsListMode.index("FAN_MOTOR")]
        detectorIdData = listModeForNormalization[:,normalization.fieldsListMode.index("IDB")].astype(np.int32)
        systemInfo.sourcePositionAfterMovement(axialMotorData, fanMotorData)

        systemInfo.detectorSideBCoordinatesAfterMovement(axialMotorData, fanMotorData,
                                                                detectorIdData)

        self.projector.pointCenterList = systemInfo.sourceCenter
        self.projector.pointCorner1List = systemInfo.verticesB[:, 7]
        self.projector.pointCorner2List = systemInfo.verticesB[:, 3]
        self.projector.pointCorner3List = systemInfo.verticesB[:, 0]
        self.projector.pointCorner4List = systemInfo.verticesB[:, 4]
        self.projector.setAbsMinMin()
        self.projector.transformIntoPositivePoints()
        self.projector.amplifyPointsToGPUCoordinateSystem()
        self.projector.createVectorialSpace()
        self.projector.createPlanes()
        # self.projector.setCountsPerPosition(np.ones(systemInfo.sourceCenter.shape[0], dtype=np.int32))
        self.projector.setCountsPerPosition((normalization.tiledProbabilityOfDetection * 1000000).astype(np.int32)) # Maybe convert the GPU entry to float32 to be univer
        print("Normalization GPU")
        optimizer = GPUSharedMemoryMultipleKernel(parent=self, normalizationFlag=False)
        optimizer.number_of_iterations = 1
        optimizer.multipleKernel()

        geometric_normalization = optimizer.im
        geometric_normalization /= np.sum(geometric_normalization) #Comentar se não quiser normalização
        self._normalizationMatrix = geometric_normalization

        folder = self.filePathOutput
        file_name = "normalization.raw"
        if self._fileOutputType == "raw":
            rawData = RawDataSetter(file_name=os.path.join(folder, file_name), size_file_m=self._normalizationMatrix.shape)
            rawData.write(self._normalizationMatrix)
        # if not os.path.exists(folder):
        #     os.makedirs(folder)

        # exporter = InterfileWriter(file_name=os.path.join(folder, file_name), data=self._normalizationMatrix)
        # # exporter.generateInterfileHeader(voxel_size=self.voxelSize, name_subject=1)
        # exporter.write()

    def generateImage(self):
        listModeBody_read = self.ToRFile_reader.fileBodyData

        axialMotorData = listModeBody_read["AXIAL_MOTOR"]
        fanMotorData = listModeBody_read["FAN_MOTOR"]
        detectorIdData = listModeBody_read["IDB"].astype(np.int32)
        #filter in energy

        systemInfo = self.ToRFile_reader.systemInfo
        # systemInfo.xRayProducer.setFocalSpotInitialPositionWKSystem([12.55, 0, 0])
        systemInfo.sourcePositionAfterMovement(axialMotorData, fanMotorData)

        systemInfo.detectorSideBCoordinatesAfterMovement(axialMotorData,
                                                              fanMotorData,
                                                              detectorIdData) ## ID_A está trocado com ID_B

        self.projector.pointCenterList = systemInfo.sourceCenter
        self.projector.pointCorner1List = systemInfo.verticesB[:, 7]   #Só esta ordem funciona
        self.projector.pointCorner2List = systemInfo.verticesB[:, 3]
        self.projector.pointCorner3List = systemInfo.verticesB[:, 0]
        self.projector.pointCorner4List = systemInfo.verticesB[:, 4]
        self.projector.transformIntoPositivePoints()
        self.projector.amplifyPointsToGPUCoordinateSystem()
        # self.projector.createVectorialSpace()
        self.projector.createPlanes()
        # self.projector.setCountsPerPosition(np.ones(systemInfo.sourceCenter.shape[0], dtype=np.int32))
        self.projector.setCountsPerPosition(self.ToRFile_reader.fileBodyData.countsPerGlobalID)
        # self._normalizationM atrix = np.ones_like(self.projector.im_index_z)  # apagar depois
        # self._normalizationMatrix = self._normalizationMatrix.astype(np.float32)  # apagar depois
        # self._normalizationMatrix /= np.sum(self._normalizationMatrix)
        optimizer = GPUSharedMemoryMultipleKernel(parent=self,)
        optimizer.normalization_matrix = self._normalizationMatrix

        print(f"GPU being use {self.device}")
        optimizer.multipleKernel()
        self.lastImageReconstructed = optimizer.im  # Get the last image reconstructed

        folder = self.filePathOutput
        file_name = "wholebody.raw"
        if self._fileOutputType == "raw":
            rawData = RawDataSetter(file_name=os.path.join(folder, file_name),
                                    size_file_m=self.lastImageReconstructed.shape)
            rawData.write(self.lastImageReconstructed)
        # generate a folder called whole_body and save the image
        # folder = os.path.join(os.path.dirname(self.file_path), "whole_body")

        # exporter = InterfileWriter(file_name=os.path.join(folder, file_name), data=self.lastImageReconstructed,)
        # # exporter.generateInterfileHeader(voxel_size=self.voxelSize, name_subject=1)
        # exporter.write()


if __name__ == "__main__":
    from toor.TORFilesReader import ToRFile
    # filename = "../../allvalues.npy"
    filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\intelligent_scan-NewGeometries-CT\\allvalues.npy"
    filename = "E:\\easyCT study 1 frontal\\simulatedsinogram_matrix.tor"
    # output_path = "C:/Users/pedro/OneDrive/Ambiente de Trabalho/Iterations_test"
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    voxelSize = [0.25, 0.25, 0.5]
    energyregion = "59.6"
    # voxelSize =[1, 1, 1]

    # output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\all_values.tor"
    # output_path = "E:\\simulatedsinogram_matrix.tor"
    output_path = None

    r = ReconstructionEasyPETCT(filename, iterations=25, subsets=1, algorithm="LM-MRP",
                     voxelSize=voxelSize, radial_fov_range=None, energyregion=energyregion, file_path_output=output_path,
                                algorithm_options=[0.15, 3])
    r.start()
    r.saveVariablesTxT()

    # plt.imshow(np.mean(r.lastImageReconstructed, axis=2))
    # plt.show()
