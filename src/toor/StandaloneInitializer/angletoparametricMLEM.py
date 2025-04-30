import os
import pycuda.driver as cuda
import numpy as np
import matplotlib.pyplot as plt
from Corrections.PET.Projector import PyramidalProjector
from Geometry import SetParametricsPoints
from Geometry.easyPETBased import MatrixGeometryCorrection
from Optimizer import GPUSharedMemoryMultipleKernel
from ImageReader.Interfile import InterfileWriter
from Corrections.EasyPET.Normalization import NormalizationCT


class ReconstructionCT:
    def __init__(self, file_path=None, file_path_output=None, iterations=25, subsets=1, algorithm="MLEM",
                 voxelSize=None, radial_fov_range=None, energyregion=None, projector_type="Simulated", GPU_recon=False):
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
        self.GPU_recon = GPU_recon
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

        self.listMode = listmode


        MatrixCorrection = MatrixGeometryCorrection(operation='r',
                                                    file_path=os.path.join(os.path.dirname(
                                                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                        'system_configurations', 'x_{}__y_{}'.format(
                                                            crystals_geometry[0], crystals_geometry[1])))
        self.geometry_file = MatrixCorrection.coordinates
        self.fov = np.sin(
            np.radians((np.max(listmode[:, 5]) + np.abs(np.min(listmode[:, 5]))) / 2 +
                       4 * np.degrees(np.arctan(np.max(self.geometry_file[1]) /
                                                60)))) *  60
        height = 2
        crystal_width = 2
        reflector_y = 2 * 0.175
        # reflector_y = self.systemConfigurations_info["reflector_interior_A_y"]#simula
        # geometry_file[:crystals_geometry[0] * crystals_geometry[1], 1] += 1
        # geometry_file[:, 1] = np.tile(np.round(np.arange(0,crystals_geometry[1]-1,0.8)-2.4,3),crystals_geometry[0])
        # geometry_file[:, 1] = np.tile(np.round(np.arange(0, crystals_geometry[1] * crystal_width + 2 * reflector_y,
        #                                                  crystal_width + reflector_y) - (crystal_width + reflector_y) *
        #                                        (crystals_geometry[1] - 1) / 2, 3), crystals_geometry[0] * 2)
        self.geometry_file[0:32, 1] = 0
        # geometry_file[0:32, 1] = 0
        self.geometry_file[32:64, 1] = 0

        z = np.repeat(np.arange(0, crystals_geometry[0] * height, height), crystals_geometry[1])

        self.geometry_file[0:crystals_geometry[0] * crystals_geometry[1], 2] = 31
        # geometry_file[0:crystals_geometry[0] * crystals_geometry[1], 2] = 31
        ## add 1.5 for 2019 aqusitions
        # geometry_file[32:64, 2] = z + 2.5
        self.geometry_file[
        crystals_geometry[0] * crystals_geometry[1]:crystals_geometry[0] * crystals_geometry[1] * 2,
        2] = (z + 1)

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
        blblbl

        """


        normalization = NormalizationCT(number_of_crystals=[32,1],
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


filename = "../../allvalues.npy"
output_path = "C:/Users/pedro/OneDrive/Ambiente de Trabalho/Iterations_test"
if not os.path.exists(output_path):
    os.makedirs(output_path)
voxelSize = [0.5,0.5,0.5]

r = ReconstructionCT(filename, iterations=10, subsets=1, algorithm="LM-MLEM",
                 voxelSize=voxelSize, radial_fov_range=None, energyregion=None, file_path_output=output_path)
r.start()

plt.imshow(np.mean(r.lastImageReconstructed, axis=2))
plt.show()





