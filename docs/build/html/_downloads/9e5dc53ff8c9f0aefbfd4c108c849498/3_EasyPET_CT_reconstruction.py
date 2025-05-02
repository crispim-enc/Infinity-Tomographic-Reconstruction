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
import numpy as np
import pycuda.driver as cuda
from Corrections import PyramidalProjector
from Corrections import NormalizationCT
from Optimizer import GPUSharedMemoryMultipleKernel


class ReconstructionEasyPETCT:
    def __init__(self, file_path=None, file_path_output=None, iterations=25, subsets=1, algorithm="MLEM",
                 voxelSize=None, radial_fov_range=None, energyregion=None):
        """
        Reconstruction Manager
        :param file_path: Path to the file
        :param iterations: Number of iterations
        :param subsets: Number of subsets
        :param algorithm: Algorithm to use LM-MLEM or LM-MRP
        """
        if radial_fov_range is None:
            radial_fov_range = [23, 40]
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
        self.file_path = file_path  # Path to the file
        self.file_path_output = file_path_output
        self.iterations = iterations  # Number of iterations
        self.subsets = subsets  # Number of subsets
        self.algorithm = algorithm  # Algorithm to use
        self.lastImageReconstructed = None  # Last image reconstructed
        self.saved_image_by_iteration = True
        self.energyRegion = energyregion

        self.ToRFile_reader = ToRFile(filepath=output_path)
        self.ToRFile_reader.read()
        self.ToRFile_reader.fileBodyData.setListModeHistogramHybridMode()


        radial_fov_range = [0, 23]
        self.projector = PyramidalProjector(voxelSize=voxelSize, FovRadialStart=radial_fov_range[0],
                                                 FovRadialEnd=radial_fov_range[1], fov=35, only_fov=True)
        self.lastImageReconstructed = None

    def start(self):
        print("Starting reconstruction")
        print("________________________________")
        self.ctx = cuda.Device(0).make_context()  # Create the context
        self.device = self.ctx.get_device()
        self.generateNormalization()
        self.generateImage()
        self.ctx.detach()

    def generateNormalization(self):
        """


        """
        normalization = NormalizationCT(number_of_crystals=[32, 1],
                                                    rangeTopMotor=108, begin_range_botMotor=0, end_rangeBotMotor=360,
                                                    stepTopmotor=0.225, stepBotMotor=1.8, recon_2D=False)
        normalization.normalization_LM()

        systemInfo = self.ToRFile_reader.systemInfo
        systemInfo.xRayProducer.setFocalSpotInitialPositionWKSystem([12.55, 0, 0])
        # nb_eventstest = 2
        # normalization.reading_data = normalization.reading_data[:nb_eventstest]
        # normalization.reading_data[:, 0] = np.arange(0,360, 360/nb_eventstest)
        # normalization.reading_data[:, 1] = np.zeros(nb_eventstest)
        # normalization.reading_data[:, 2] = 0
        systemInfo.sourcePositionAfterMovement(normalization.reading_data[:,0], normalization.reading_data[:,1])

        systemInfo.detectorSideBCoordinatesAfterMovement(normalization.reading_data[:,0], normalization.reading_data[:,1],
                                                                normalization.reading_data[:,2].astype(np.int32))


        # parametric_listMode = parametric_listMode.loc[0]
        # self.replacePointsInProjector(parametric_listMode)
        self.projector.pointCenterList = systemInfo.sourceCenter
        self.projector.pointCorner1List = systemInfo.verticesB[:, 7]
        self.projector.pointCorner2List = systemInfo.verticesB[:, 3]
        self.projector.pointCorner3List = systemInfo.verticesB[:, 0]
        self.projector.pointCorner4List = systemInfo.verticesB[:, 4]
        # for i in range(8):
        #     # print(systemInfo.verticesB[1, i, 0:2])
        #     plt.plot(systemInfo.verticesB[1, i, 1], systemInfo.verticesB[1, i, 2], "o", label=f"Corner {i}")
        #
        # plt.plot(systemInfo.sourceCenter[1, 1],systemInfo.sourceCenter[1, 2] , 'ro', label='Source Center')
        # plt.legend()
        # plt.show()

        self.projector.createVectorialSpace()
        self.projector.createPlanes()
        self.projector.setCountsPerPosition(np.ones(systemInfo.sourceCenter.shape[0], dtype=np.int32))
        print("Normalization GPU")
        optimizer = GPUSharedMemoryMultipleKernel(parent=self, normalizationFlag=False)
        optimizer.number_of_iterations = 1
        optimizer.multipleKernel()

        geometric_normalization = optimizer.im
        geometric_normalization /= np.sum(geometric_normalization) #Comentar se não quiser normalização
        self._normalizationMatrix = geometric_normalization
        # self._normalizationMatrix = np.ones_like(geometric_normalization)  # reset normalization matrix
        # self._normalizationMatrix *= geometric_normalization #Comentar se não quiser normalização
        # self._normalizationMatrix /= np.sum(self._normalizationMatrix)

        folder = self.file_path_output
        file_name = "normalization"
        # if not os.path.exists(folder):
        #     os.makedirs(folder)
        # exporter = InterfileWriter(file_name=os.path.join(folder, file_name), data=self._normalizationMatrix)
        # # exporter.generateInterfileHeader(voxel_size=self.voxelSize, name_subject=1)
        # exporter.write()

    def generateImage(self):
        listModeBody_read = self.ToRFile_reader.fileBodyData
        systemInfo = self.ToRFile_reader.systemInfo
        systemInfo.xRayProducer.setFocalSpotInitialPositionWKSystem([12.55, 0, 0])
        systemInfo.sourcePositionAfterMovement(listModeBody_read["AXIAL_MOTOR"], listModeBody_read["FAN_MOTOR"])

        systemInfo.detectorSideBCoordinatesAfterMovement(listModeBody_read["AXIAL_MOTOR"],
                                                              listModeBody_read["FAN_MOTOR"],
                                                              listModeBody_read["IDB"].astype(np.int32)) ## ID_A está trocado com ID_B

        self.projector.pointCenterList = systemInfo.sourceCenter
        self.projector.pointCorner1List = systemInfo.verticesB[:, 7]   #Só esta ordem funciona
        self.projector.pointCorner2List = systemInfo.verticesB[:, 3]
        self.projector.pointCorner3List = systemInfo.verticesB[:, 0]
        self.projector.pointCorner4List = systemInfo.verticesB[:, 4]
        var_1 = 0
        var_2 = 1

        self.projector.createVectorialSpace()
        self.projector.createPlanes()
        # self.projector.setCountsPerPosition(np.ones(systemInfo.sourceCenter.shape[0], dtype=np.int32))
        self.projector.setCountsPerPosition(self.ToRFile_reader.fileBodyData.countsPerGlobalID)
        # self._normalizationM atrix = np.ones_like(self.projector.im_index_z)  # apagar depois
        # self._normalizationMatrix = self._normalizationMatrix.astype(np.float32)  # apagar depois
        # self._normalizationMatrix /= np.sum(self._normalizationMatrix)
        optimizer = GPUSharedMemoryMultipleKernel(parent=self, )
        optimizer.normalization_matrix = self._normalizationMatrix
        # optimizer.im = np.ascontiguousarray(self._normalizationMatrix * self.projector.countsPerPosition.sum(), dtype=np.float32)
        print(f"GPU being use {self.device}")
        optimizer.multipleKernel()

        self.lastImageReconstructed = optimizer.im  # Get the last image reconstructed
        # generate a folder called whole_body and save the image
        # folder = os.path.join(os.path.dirname(self.file_path), "whole_body")
        folder = self.file_path_output
        file_name = "image"
        if not os.path.exists(folder):
            os.makedirs(folder)
        # exporter = InterfileWriter(file_name=os.path.join(folder, file_name), data=self.lastImageReconstructed,)
        # # exporter.generateInterfileHeader(voxel_size=self.voxelSize, name_subject=1)
        # exporter.write()


if __name__ == "__main__":
    from TORFilesReader import ToRFile
    # filename = "../../allvalues.npy"
    filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\intelligent_scan-NewGeometries-CT\\allvalues.npy"
    output_path = "C:/Users/pedro/OneDrive/Ambiente de Trabalho/Iterations_test"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    voxelSize =[0.25,0.25,1]
    # voxelSize =[1, 1, 1]


    output_path = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\all_values.tor"


    r = ReconstructionEasyPETCT(filename, iterations=20, subsets=1, algorithm="LM-MLEM",
                     voxelSize=voxelSize, radial_fov_range=None, energyregion=None, file_path_output=output_path)
    # r.start()

    # plt.imshow(np.mean(r.lastImageReconstructed, axis=2))
    # plt.show()
