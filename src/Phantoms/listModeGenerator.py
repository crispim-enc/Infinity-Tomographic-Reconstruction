import numpy as np
import os
import pycuda.driver as cuda
from src.Corrections.EasyPET.Normalization import GenerateEveryPossibleVolumePositions
from src.Geometry import SetParametricCoordinates, MatrixGeometryCorrection, ParallelepipedProjector
from src.Optimizer import ROIEvents
from src.Phantoms import PhantomGenerator
from src.EasyPETLinkInitializer.EasyPETDataReader import binary_data


class ListModeGenerator:
    def __init__(self, PET_MAP=None, reading_data=None, planes_equations=None,
                 stepTopmotor=None, stepBotMotor=None, rangeTopMotor=None):
        cuda.init()
        self.ctx = cuda.Device(0).make_context()
        self.device = self.ctx.get_device()
        self.cuda_drv = cuda
        self.directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"dataFiles")

        self.stepTopmotor = stepTopmotor
        self.stepBotMotor = stepBotMotor
        self.rangeTopMotor = rangeTopMotor

        self.a = planes_equation.a
        self.b = planes_equation.b
        self.c = planes_equation.c
        self.d = planes_equation.d

        self.a_normal = planes_equation.a_normal
        self.b_normal = planes_equation.b_normal
        self.c_normal = planes_equation.c_normal
        self.d_normal = planes_equation.d_normal

        self.a_cf = planes_equation.a_cf
        self.b_cf = planes_equation.b_cf
        self.c_cf = planes_equation.c_cf
        self.d_cf = planes_equation.d_cf
        self.PET_MAP = PET_MAP

        self.im_index_x = planes_equation.im_index_x
        self.im_index_y = planes_equation.im_index_y
        self.im_index_z = planes_equation.im_index_z

        self.half_crystal_pitch_xy = planes_equation.half_crystal_pitch_xy
        self.half_crystal_pitch_z = planes_equation.half_crystal_pitch_z
        self.distance_between_array_pixel = planes_equations.distance_between_array_pixel
        PET_MAP =np.ascontiguousarray(np.zeros((self.im_index_x.shape)), dtype=np.float32)
        # PET_MAP[int(PET_MAP.shape[0]/2)-3:int(PET_MAP.shape[0]/2)+3,
        #         int(PET_MAP.shape[1]/2)-3:int(PET_MAP.shape[1]/2)+3,
        #         int(PET_MAP.shape[2]/2)-3:int(PET_MAP.shape[2]/2)+3] =1

        PET_MAP[int(PET_MAP.shape[0] / 2) - 3:int(PET_MAP.shape[0] / 2) + 3,
        int(PET_MAP.shape[1] / 2) - 3:int(PET_MAP.shape[1] / 2) + 3,
        0:3] = 1
        active_pixels = np.where(PET_MAP > 0)
        self.active_pixel_x = active_pixels[0]
        self.active_pixel_y = active_pixels[1]
        self.active_pixel_z = active_pixels[2]
        self.active_pixel_x = np.ascontiguousarray(self.active_pixel_x, dtype=np.int32)
        self.active_pixel_y = np.ascontiguousarray(self.active_pixel_y, dtype=np.int32)
        self.active_pixel_z = np.ascontiguousarray(self.active_pixel_z, dtype=np.int32)
        self.im = np.ascontiguousarray(np.zeros((self.im_index_x.shape)), dtype=np.float32)
        self.sum_pixel = np.ascontiguousarray(
            np.zeros(self.a.shape, dtype=np.float32))

        roievents = ROIEvents(self)
        valid_vor = roievents.pixel2Position()
        sum_vor = roievents.sum_vor[valid_vor == 1]

        self.ctx.detach()
        self.reading_data = reading_data[valid_vor == 1]
        self.reading_data = np.repeat(self.reading_data, sum_vor.astype(np.int32), axis=0)
        self.generate_file()

    def generate_file(self):
        """

        """
        gen_directory = os.path.join(self.directory, "listmode_generated")
        # file_gen = os.path.join(gen_directory, "listMode_gen.easypet")
        if not os.path.isdir(gen_directory):
            os.mkdir(gen_directory)
            os.mkdir(os.path.join(gen_directory, "static_image"))

        file = "C:\\Users\\pedro\\Downloads\\Easypet Scan 29 Nov 2021 - 09h 53m 05s\\Easypet Scan 29 Nov 2021 - 09h 53m 05s.easypet"
        [listMode, Version_binary, header, dates, otherinfo, acquisitionInfo, stringdata, systemConfigurations_info,
         energyfactor_info, peakMatrix_info] = binary_data().open(file_name=file)
        self.reading_data[:,3] = (self.reading_data[:,3]+self.rangeTopMotor/2)/self.stepTopmotor
        self.reading_data[:,2] = self.reading_data[:,2]/self.stepBotMotor
        header[0] = 1
        header[1] = self.stepBotMotor
        header[2] = 1
        header[3] = self.stepTopmotor
        header[4] = 1
        header[5] = self.rangeTopMotor
        EA_Corrected = np.ones(len(self.reading_data), dtype=np.float32)*511

        EB_Corrected = np.ones((len(self.reading_data)), dtype=np.float32)*511
        timestamp = np.ones(len(self.reading_data), dtype=np.float32)

        binary_data().save_listmode(gen_directory, gen_directory, self.reading_data[:,2:], self.reading_data[:,0:2],
                                    EA_Corrected, EB_Corrected,
                          timestamp, header, dates, False, stringdata, acquisitionInfo,
                          part_file_number=1, joining_files=True)

    def _open_dummy_file(self):
        """"""

        # plt.hist(reading_data[:,3],len(np.unique(reading_data[:,3])))
        # plt.show()


if __name__ == "__main__":
    crystals_geometry = [32, 2]
    crystal_width = 2
    crystal_height = 2
    FOV = 20
    distance_between_motors = 30
    distance_crystals = 60
    pixelSizeXY = 0.5
    pixelSizeXYZ = 0.5
    reflector_xy = 0.28
    reflector_z = 0.35
    topMotorStep = 0.9
    BotMotorStep = 3.6
    TopMotorRange = np.sin(np.radians(FOV))*distance_crystals*2
    total_listmode_c = GenerateEveryPossibleVolumePositions(stepTopmotor=topMotorStep, stepBotMotor=BotMotorStep,
                                                            rangeTopMotor=TopMotorRange)
    total_listmode_c.matrix_without_reduction()
    reading_data = total_listmode_c.every_possible_position_array
    print(len(reading_data))
    MatrixCorrection = MatrixGeometryCorrection(operation='r',
                                                file_path=os.path.join(os.path.dirname(
                                                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                    'system_configurations', 'x_{}__y_{}'.format(
                                                        crystals_geometry[0], crystals_geometry[1])))
    geometry_file = MatrixCorrection.coordinates
    z = np.repeat(np.arange(0, crystals_geometry[0] * 2, 2), 2)
    geometry_file[0:crystals_geometry[0] * crystals_geometry[1], 2] = (z + 1)
    ## add 1.5 for 2019 aqusitions
    # geometry_file[32:64, 2] = z + 2.5
    geometry_file[
    crystals_geometry[0] * crystals_geometry[1]:crystals_geometry[0] * crystals_geometry[1] * 2,
    2] = (z + 1)
#
    parametric_coordinates = SetParametricCoordinates(listMode=reading_data,
                                                      geometry_file=geometry_file,
                                                      simulation_files=True,
                                                      crystal_width=2,
                                                      shuffle=False, FoV= FOV,
                                                      distance_between_motors=distance_between_motors,
                                                      distance_crystals=distance_crystals,
                                                      correct_decay=False,
                                                      recon2D=False, number_of_neighbours="Auto",
                                                      generated_files=True)

    planes_equation = ParallelepipedProjector(parametric_coordinates, pixelSizeXY=pixelSizeXY,
                                              pixelSizeXYZ=pixelSizeXYZ,
                                              crystal_width=crystal_width,
                                              crystal_height=crystal_height,
                                              reflector_xy=reflector_xy,
                                              reflector_z=reflector_z,
                                              FoV=FOV,
                                              bool_consider_reflector_in_z_projection=False,
                                              bool_consider_reflector_in_xy_projection=False,
                                              distance_crystals=distance_crystals)

    p = PhantomGenerator()
    image = p.derenzo()
    image = p.apply_circular_mask_fov(image)
    for z in range(image.shape[2]):
        image[:, :, z] = np.max(image[:, :, z]) - image[:, :, z]
    image = p.apply_circular_mask_fov(image)

    ListModeGenerator(PET_MAP=image, reading_data=reading_data, planes_equations=planes_equation,
                      stepTopmotor=topMotorStep, stepBotMotor=BotMotorStep,
                                                            rangeTopMotor=TopMotorRange )
    print("size")
# time_correction = parametric_coordinates.decay_factor
#
# self.adaptativedoimap = AdaptativeDOIMapping(listMode=reading_data)
# self.adaptativedoimap.load_doi_files()
# self.adaptativedoimap.generate_listmode_doi_values()
#
# self.im = EM(algorithm=algorithm, algorithm_options=algorithm_options,
#                 normalization_matrix=self.normalization_matrix,
#                 time_correction=time_correction,
#                 planes_equation=planes_equation, number_of_iterations=self.number_of_iterations,
#                 number_of_subsets=number_of_subsets,
#                 directory=study_path, cuda_drv=cuda, pixeltoangle=pixeltoangle,
#                 easypetdata=Easypetdata, saved_image_by_iteration=False, multiple_kernel=multiple_kernel,
#                 entry_im=entry_im, signals_interface=signals_interface,
#                 current_info_step=current_info_step, doi_mapping=self.adaptativedoimap).im