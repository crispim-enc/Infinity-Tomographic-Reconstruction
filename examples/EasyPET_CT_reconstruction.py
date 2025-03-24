#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: EasyPET_CT
# * AUTHOR: Pedro Encarnação
# * DATE: 17/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

"""
This example shows how use the TOR package to build and reconstruct a easyPET/CT system.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pycuda.driver as cuda
from src.Geometry.easyPETBased import EasyCTGeometry, testSourceDistance
from src.DetectionLayout.Modules import PETModule, easyPETModule
from src.DetectionLayout.RadiationProducer import GenericRadiativeSource
from src.Designer import DeviceDesignerStandalone
from src.Corrections.CT.Projector import PyramidalProjector
from src.Optimizer import kernelManager



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
        crystals_geometry = [32, 1]
        listmode = np.load(file_path)
        # listmode[:,2] += 1
        # listmode[:,3] += 1
        listmode = listmode[listmode[:, 2] < 64]
        listmode = listmode[listmode[:, 3] < 64]
        listmode = listmode[listmode[:,6] < 481]

        idA = np.copy(listmode[:, 2])
        idB = np.copy(listmode[:, 3])
        listmode[:, 2] = idB
        listmode[:, 3] = idA

        print("Top min: {}".format(listmode[:,5].min()))
        print("Top max: {}".format(listmode[:,5].max()))
        print("Bot min: {}".format(listmode[:,4].min()))
        print("Bot max: {}".format(listmode[:,4].max()))
        print("IdA min: {}".format(listmode[:,2].min()))
        print("IdA max: {}".format(listmode[:,2].max()))
        print("IdB min: {}".format(listmode[:,3].min()))
        print("IdB max: {}".format(listmode[:,3].max()))


        #

        self.parametric_coordinates = SetParametricsPoints(listMode=self.listMode, geometry_file=self.geometry_file,
                                                      simulation_files=False)
        # pointCenterlist = self.parametric_coordinates.sourceCenter
        #  = self.parametric_coordinates.vertice1
        # pointCorner2List = self.parametric_coordinates.vertice2
        # pointCorner3List = self.parametric_coordinates.vertice3
        # pointCorner4List = self.parametric_coordinates.vertice4

        radial_fov_range = [0, 23]
        self.projector = PyramidalProjector(voxelSize=voxelSize, FovRadialStart=radial_fov_range[0],
                                                 FovRadialEnd=radial_fov_range[1], fov=self.fov)


        # self._normalizationMatrix = np.ones_like(self.projector.im_index_z)  # apagar depois
        # self._normalizationMatrix = self._normalizationMatrix.astype(np.float32)  # apagar depois
        # self._normalizationMatrix /= np.sum(self._normalizationMatrix)
        self.lastImageReconstructed = None

    def start(self):

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
        parametric_listMode = normalization.reading_data
        self.parametric_coordinates = SetParametricsPoints(listMode=parametric_listMode, geometry_file=self.geometry_file,
                                                           normalization=True)
        # parametric_listMode = parametric_listMode.loc[0]
        # self.replacePointsInProjector(parametric_listMode)
        self.projector.pointCenterList = self.parametric_coordinates.sourceCenter
        self.projector.pointCorner1List = self.parametric_coordinates.corner1list
        self.projector.pointCorner2List = self.parametric_coordinates.corner4list
        self.projector.pointCorner3List = self.parametric_coordinates.corner3list
        self.projector.pointCorner4List = self.parametric_coordinates.corner2list

        self.projector.createVectorialSpace()
        self.projector.createPlanes()
        print("Normalization GPU")
        optimizer = GPUSharedMemoryMultipleKernel(parent=self, normalizationFlag=False)
        optimizer.number_of_iterations = 1
        optimizer.multipleKernel()

        geometric_normalization = optimizer.im
        geometric_normalization /= np.sum(geometric_normalization) #Comentar se não quiser normalização
        self._normalizationMatrix = geometric_normalization
        self._normalizationMatrix = np.ones_like(geometric_normalization)  # reset normalization matrix
        # self._normalizationMatrix *= geometric_normalization #Comentar se não quiser normalização
        # self._normalizationMatrix /= np.sum(self._normalizationMatrix)

        folder = self.file_path_output
        file_name = "normalization"
        if not os.path.exists(folder):
            os.makedirs(folder)
        exporter = InterfileWriter(file_name=os.path.join(folder, file_name), data=self._normalizationMatrix)
        # exporter.generateInterfileHeader(voxel_size=self.voxelSize, name_subject=1)
        exporter.write()

    def generateImage(self):
        self.parametric_coordinates = SetParametricsPoints(listMode=self.listMode,
                                                           geometry_file=self.geometry_file,
                                                           simulation_files=False)
        self.projector.pointCenterList = self.parametric_coordinates.sourceCenter
        self.projector.pointCorner1List = self.parametric_coordinates.corner1list
        self.projector.pointCorner2List = self.parametric_coordinates.corner4list
        self.projector.pointCorner3List = self.parametric_coordinates.corner3list
        self.projector.pointCorner4List = self.parametric_coordinates.corner2list
        var_1 = 0
        var_2 = 1

        self.projector.createVectorialSpace()
        self.projector.createPlanes()
        optimizer = GPUSharedMemoryMultipleKernel(parent=self, )
        optimizer.normalization_matrix = self._normalizationMatrix
        print(f"GPU being use {self.device}")
        optimizer.multipleKernel()

        self.lastImageReconstructed = optimizer.im  # Get the last image reconstructed
        # generate a folder called whole_body and save the image
        # folder = os.path.join(os.path.dirname(self.file_path), "whole_body")
        folder = self.file_path_output
        file_name = "image"
        if not os.path.exists(folder):
            os.makedirs(folder)
        exporter = InterfileWriter(file_name=os.path.join(folder, file_name), data=self.lastImageReconstructed,)
        # exporter.generateInterfileHeader(voxel_size=self.voxelSize, name_subject=1)
        exporter.write()


# filename = "../../allvalues.npy"
filename = "C:\\Users\\pedro\\OneDrive\\Ambiente de Trabalho\\intelligent_scan-NewGeometries-CT\\allvalues.npy"
output_path = "C:/Users/pedro/OneDrive/Ambiente de Trabalho/Iterations_test"
if not os.path.exists(output_path):
    os.makedirs(output_path)

voxelSize =[0.5,0.5,0.5]

list_mode = np.load(filename)


# Set PET module type
_module = easyPETModule

# Set x-ray producer object
xrayproducer = GenericRadiativeSource()

# Set device
newDevice = EasyCTGeometry(detector_moduleA=_module, detector_moduleB=_module, x_ray_producer=xrayproducer)

# Set source
newDevice.xRayProducer.setFocalSpotInitialPositionWKSystem([-2, 0, 36.2 / 2])
newDevice.evaluateInitialSourcePosition()

# Set modules Side A
newDevice.setNumberOfDetectorModulesSideA(2)
moduleSideA_X_translation = np.array([-15, -15], dtype=np.float32)
moduleSideA_Y_translation = np.array([-2.175, 2.175], dtype=np.float32)
moduleSideA_Z_translation = np.array([36.2 / 2, 36.2 / 2], dtype=np.float32)
moduleSideA_alpha_rotation = np.array([0, 0], dtype=np.float32)
moduleSideA_beta_rotation = np.array([0, 0], dtype=np.float32)
moduleSideA_sigma_rotation = np.array([-90, -90], dtype=np.float32)

for i in range(newDevice.numberOfDetectorModulesSideA):
    newDevice.detectorModulesSideA[i].setXTranslation(moduleSideA_X_translation[i])
    newDevice.detectorModulesSideA[i].setYTranslation(moduleSideA_Y_translation[i])
    newDevice.detectorModulesSideA[i].setZTranslation(moduleSideA_Z_translation[i])
    newDevice.detectorModulesSideA[i].setAlphaRotation(moduleSideA_alpha_rotation[i])
    newDevice.detectorModulesSideA[i].setBetaRotation(moduleSideA_beta_rotation[i])
    newDevice.detectorModulesSideA[i].setSigmaRotation(moduleSideA_sigma_rotation[i])

newDevice.setNumberOfDetectorModulesSideB(2)
moduleSideB_X_translation = np.array([75, 75], dtype=np.float32)
moduleSideB_Y_translation = np.array([-2.175, 2.175], dtype=np.float32)
moduleSideB_Z_translation = np.array([36.2 / 2, 36.2 / 2], dtype=np.float32)
moduleSideB_alpha_rotation = np.array([0, 0], dtype=np.float32)
moduleSideB_beta_rotation = np.array([0, 0], dtype=np.float32)
moduleSideB_sigma_rotation = np.array([90, 90], dtype=np.float32)

for i in range(newDevice.numberOfDetectorModulesSideB):
    newDevice.detectorModulesSideB[i].setXTranslation(moduleSideB_X_translation[i])
    newDevice.detectorModulesSideB[i].setYTranslation(moduleSideB_Y_translation[i])
    newDevice.detectorModulesSideB[i].setZTranslation(moduleSideB_Z_translation[i])
    newDevice.detectorModulesSideB[i].setAlphaRotation(moduleSideB_alpha_rotation[i])
    newDevice.detectorModulesSideB[i].setBetaRotation(moduleSideB_beta_rotation[i])
    newDevice.detectorModulesSideB[i].setSigmaRotation(moduleSideB_sigma_rotation[i])

# S
# newDevice
newDevice.setDeviceName("EasyCT")
newDevice.setDeviceType("CT")
newDevice.generateInitialCoordinates()

from src.TORFilesReader import ToRFile



# newDevice.generateDeviceUUID()
# newDevice.createDirectory()
print(newDevice.deviceUUID)
print(newDevice.deviceName)


#TESTS
axial_motor_angles = np.deg2rad(np.arange(0, 360, 45))
fan_motor_angles = np.deg2rad(np.arange(-45, 60, 15))
# repeat the fan motor angles for each axial motor angle
fan_motor_angles = np.repeat(fan_motor_angles, len(axial_motor_angles))
axial_motor_angles = np.tile(axial_motor_angles, len(fan_motor_angles) // len(axial_motor_angles))
newDevice.sourcePositionAfterMovement(axial_motor_angles, fan_motor_angles)
testSourceDistance(newDevice.xRayProducer.focalSpotInitialPositionWKSystem, newDevice.sourceCenter,
                   newDevice.originSystemWZ.T)
index_fan_motor_angle_0 = np.where(fan_motor_angles == 0)
source_center_fan_motor_angle_0 = newDevice.sourceCenter[index_fan_motor_angle_0]
origin_fan_motor_angle_0 = newDevice.originSystemWZ.T[index_fan_motor_angle_0]

# Plots
# plt.plot(newDevice.originSystemWZ[0], newDevice.originSystemWZ[1], 'ro', label='Origin Fan Motor')
# # plot source center
# plt.plot(newDevice.sourceCenter[:, 0], newDevice.sourceCenter[:, 1], 'bo', label='Source Center')
# # plot a line from the origin to the source center at fan motor angle 0
# plt.plot(source_center_fan_motor_angle_0[:, 0], source_center_fan_motor_angle_0[:, 1], 'gx')
#
# plt.plot([origin_fan_motor_angle_0[:, 0], source_center_fan_motor_angle_0[:, 0]],
#          [origin_fan_motor_angle_0[:, 1], source_center_fan_motor_angle_0[:, 1]], '-')
# plt.legend()
# plt.title("Configuration Source side of detector module A")
# plt.title("Configuration Source in front module")
# plt.show()

# add visualization
designer = DeviceDesignerStandalone(device=newDevice)
designer.addDevice()
designer.addxRayProducerSource()
designer.startRender()

r = ReconstructionEasyPETCT(filename, iterations=10, subsets=1, algorithm="LM-MLEM",
                 voxelSize=voxelSize, radial_fov_range=None, energyregion=None, file_path_output=output_path)
r.start()

plt.imshow(np.mean(r.lastImageReconstructed, axis=2))
plt.show()
