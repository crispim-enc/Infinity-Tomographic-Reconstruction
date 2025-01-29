#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

import os
import glob
import copy
import numpy as np
import pycuda.driver as cuda
from array import array
from src.Corrections.EasyPET.Normalization import AdaptiveNormalizationMatrix
from src.Geometry.easyPETBased.matrixgeometrycorrection import MatrixGeometryCorrection
from src.Geometry.easyPETBased.easypet_parametric_points import SetParametricCoordinates
from src.Corrections.PET.Projector import ParallelepipedProjector
from src.Optimizer.GPUManager import EM
from src.AnalyticalReconstruction.initializer import AnalyticalReconstruction
from src.Corrections.PET.DOI import AdaptativeDOIMapping
from src.Corrections.PET.DecayCorrection import DecayCorrection
from src.Quantification import FactorQuantificationFromUniformPhantom
from src.ImageReader import RawDataSetter
from src.ImageReader.DICOM import WriteDicom


class ReconstructionInitializer:
    """     Links the reconstruction code with the interface or standalone code.
            Basically manages the diferent functions:
                - Images Corrections
                    Normalization
                    Attenuation
                    Decay Correction
                    DOI
                - Parametric coordinates
                - TOR planes
                - EM (Optimizer and CPU)
                - Save data (Saving in Raw Data (See ImageReader package) -->future implementation saves in DICOM)
                - Convert data to quantification

            The data needs to pass throught the preprocessing package
            Parameters
            ------------
            Easypetdata: object PrepareEasyPETdata
            **Kwargs EM
            algorithm: DEFAULT: LM-MRP
                       LM-MLEM: List Mode Maximum Likelihood Expectation Method
                       LM-OSEM: List Mode Ordered Subsets Expectation Method (Future Implementation)
                       LM-MRP: List Mode Median Root à Priori
                       AnalyticalReconstruction: Filter Back Projection (2D)
                       MLEM: List Mode Maximum Likelihood Expectation Method (Histogram mode: Future Implementation
                       OSEM: Ordered Subsets Expectation Method (Future Implementation)

            Algortihm_options: type: list
                               LM-MLEM: None
                               LM-MRP: [beta, kernel_size]  type: [float, int]
                               FPB:[rebining] type: [bool]
            Crystals_geometry: Optional, If the listmode is the second version crystal_geometry is define by that option
            Type_of_reconstruction: [Whole_body, static, dynamic, gated] type: [bool, bool, bool, bool]
                                    -Whole_body: all FOV is reconstructed for the total time of acquisition
                                    -Static: Region of interest reconstruction for the total time of acquisition.
                                            Interface delivers the ROI (Not working)
                                    -Dynamic: Reconstruction the individual frames (1 turn = 1 frame).
                                            Cumulative frames not developed
                                    -Gated: Reconstruction with respiratory_cycle and heart cycle (Not working)
            live_recon: Allows the reconstruction during the scanning
            live_normalization_pre_calculation: reconstrut the normalization Matrix at the scan begining
            number_of_iterations: Iterative algoritms (LM-MLEM, LM-MRP,...)
            number_of_subsets: Iterative algoritms (LM-OSEM, OSEM)
            pixelSizeXY: Defines the voxel dimensions on radial and tangencial direction
            pixelSizeXYZ: Defines the voxel dimension in axial direction. This value changes if the reflector is not taken
                        into consideration (Default)
            type_of_projector: "Box-Counts": Every voxel inside the tube of response has equal probability (default: 1)
                               "Orthogonal": the probability of the voxel is quocient between the orthogonal distance
                               to the center of the FOV and the distance between the center and the corner of the active
                               crystal
                               "Solid Angle": (Slow)
                               "Solid Angle Small Angle approximation":
            recon2D=False,
            number_of_neighbours="Auto"
            study_path: Folder of the data,
            multiple_kernel: Optimizer arch deployment: Type bool
                                    -Single Kernel: Transfer Arrays between CPU and Optimizer as a whole.
                                    -Multiple Kernel: Divide the data in chuncks, which can be better allocated to use the
                                    memory more efficiently. Depending on the Data can reduce the code divergency

            map_precedent: "LM-MLEM" (Not working)
            override_geometry_file: Overrides the geometry file see function _override
            detector_normalization_correction: Use detector Normalization type: bool,
            detector_normalization_data: "On-fly": Calculates for every reconstruction. (Flexible for all the data) (SLOW)
                                         "Stored": Use stored images of a uniforme phantom. The reconstruction of scannings
                                         needs to have the same conditions of the uniform phantom.

            attenuation_correction=False,
            attenuation_data="Calculated Uniform",
            decay_correction=False,
            positron_range=False,
            scatter_angle_correction=False,
            dead_time_correction=False,
            random_correction=False,
            doi_correction=False,
            respiratory_movement_correction=False,
            heart_movement_correction=False,
            save_numpy=True,
            save_raw_data=True,
            signals_interface: Object from easypet UI containing the signals
                                -trigger_projection_2d_offline
            simulation_file=False,
            multiple_save_conditions=None,
            override_geometric_values=None, feet_first=False
            """

    def __init__(self, Easypetdata=None, algorithm='LM-MLEM',
                 live_recon=False, live_normalization_pre_calculation=False, type_of_reconstruction=None,
                 number_of_iterations=20, number_of_subsets=1,
                 study_path=None, override_geometry_file=True,
                 detector_normalization_correction=True, detector_normalization_data="Calculated",
                 attenuation_correction=False, attenuation_data="Calculated Uniform", decay_correction=True,
                 save_numpy=False,
                 save_raw_data=True,
                 signals_interface=None, multiple_save_conditions=None,
                 override_geometric_values=None, feet_first=False, remove_turns=None, **kwargs):

        print(feet_first)
        if study_path is None:
            return

        if type_of_reconstruction is None:
            type_of_reconstruction = [True, False, False, False]

        if multiple_save_conditions is None:
            multiple_save_conditions = [False, None]
        self.current_type_of_reconstruction = None  # WHOLE BODY, STATIC, GATED , DYNAMIC
        # self.Easypetdata = Easypetdata
        self.live_recon = live_recon
        self.live_normalization_pre_calculation = live_normalization_pre_calculation
        self.type_of_reconstruction = type_of_reconstruction
        self.study_path = study_path
        self.override_geometry_file = override_geometry_file
        self.save_numpy = save_numpy
        self.save_raw_data = save_raw_data
        self.save_dicom = True
        self.save_by_iteration = True
        self.signals_interface = signals_interface
        self.detector_normalization_correction = detector_normalization_correction
        self.detector_normalization_data = detector_normalization_data
        self.attenuation_correction = attenuation_correction
        self.attenuation_data = attenuation_data
        self.decay_correction = decay_correction
        self.multiple_save_conditions = multiple_save_conditions

        # Parametric coordinates default options
        self.recon2D = False
        self.number_of_neighbours = "Auto"
        self.shuffle_data = False
        self.simulation_file = False
        # Planes Equations
        self.bool_consider_reflector_in_z_projection = False
        self.bool_consider_reflector_in_xy_projection = False
        # EM class default options
        self.algorithm = "LM-MLEM"
        self.algorithm_options = [0.15, 3]
        self.crystals_geometry = None
        self.number_of_iterations = 25
        self.number_of_subsets = 1
        self.pixelSizeXY = 0.5
        self.pixelSizeXYZ = 0.5
        self.type_of_projector = "Solid-Angle small approximation"
        self.multiple_kernel = True
        self.map_precedent = "LM-MRP"
        self.positron_range = False
        self.scatter_angle_correction = False
        self.dead_time_correction = False
        self.random_correction = False
        self.doi_correction = None
        self.respiratory_movement_correction = False
        self.heart_movement_correction = False
        self.feet_first = feet_first
        self.normalization_matrix = None
        self._geometry_file = None
        self.volume_voxel = None
        self.scan_time = None
        self.number_cumulative_turns = 1
        self.correctDecayBool = False
        self.transform_into_positive = True

        if live_normalization_pre_calculation is False:
            self.entry_im = None
            self.pixeltoangle = False
            self.reading_data = Easypetdata.listMode
            self.header = Easypetdata.header
            self.Version_binary = Easypetdata.Version_binary
            self.dates = Easypetdata.dates
            self.otherinfo = Easypetdata.otherinfo
            self.stringdata = Easypetdata.stringdata
            self.systemConfigurations_info = Easypetdata.systemConfigurations_info
            self.acquisitionInfo = Easypetdata.acquisitionInfo
            self.energyfactor_info = Easypetdata.energyfactor_info
            self.peakMatrix_info = Easypetdata.peakMatrix_info

        if self.reading_data is None:
            return

        self.dynamicFrames = self.acquisitionInfo["Number of turns"]
        self.basename_name_save = "ID_{}_{}".format(self.acquisitionInfo["Acquisition start time"],
                                                    self.acquisitionInfo["ID"])
        if remove_turns is None:
            self.remove_turns = {
                "Cut_per_time": True,
                "Init time": 0,
                "End time": self.reading_data[-1, 6],
                "Whole body": False,
                "Dynamic": False,
                "Static": False,
                "Gated": False
            }

        else:
            self.remove_turns = remove_turns

        # if self.Version_binary == "Version 2" or self.Version_binary == "Version 3":
        self.crystals_geometry = [self.systemConfigurations_info["array_crystal_x"],
                                  self.systemConfigurations_info["array_crystal_y"]]

        # Rewrite entries
        # for key, value in kwargs.items():
        #     exec("self.%s = %s" % (key, value))
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.pixeltoangle = False

        if override_geometry_file:
            self._override_geometry_file_func(override_geometric_values=override_geometric_values)
        else:
            self.geometry_file()
        angle_fov = (np.max(self.reading_data[:, 5]) + np.abs(np.min(self.reading_data[:, 5]))) / 2
        self.fov = np.sin(np.radians(angle_fov + 4 * np.degrees(np.arctan(np.max(self._geometry_file[:, 1]) /
                                                              self.systemConfigurations_info[
                                                                  "distance_between_crystals"])))) * \
                   self.systemConfigurations_info["distance_between_crystals"]
        self.fov = 35  # for testing
        print(self.fov)
        self.max_possible_position = 2 * 75 * np.sqrt(2) / 2
        # np.cos(np.pi/4)**2* 2*np.sin(np.radians(angle_fov +
        #                          np.degrees(np.arctan(np.max(self._geometry_file[:, 1]) /
        #                                                   self.systemConfigurations_info[
        #                                                       "distance_between_crystals"])))) * \
        #                          (self.systemConfigurations_info["distance_between_crystals"]+
        #                           self.systemConfigurations_info['crystal_length']/2)
        self._generate_folders()
        # Algorithm Parameters

        self.algorithm = self._algorithm(algorithm, number_of_iterations, number_of_subsets)
        # Decay Correction
        self.decay_correction_class = DecayCorrection(listMode=self.reading_data, acquisition_info=self.acquisitionInfo,
                                                      correct_decay=self.correctDecayBool)
        self.decay_correction_class.list_mode_decay_correction()
        self.decay_correction_class.activity_on_subject_at_scanning_time()
        self.decay_factor = self.decay_correction_class.decay_factor

        self.parametric_coordinates = SetParametricCoordinates(listMode=self.reading_data,
                                                               geometry_file=self._geometry_file,
                                                               simulation_files=self.simulation_file,
                                                               crystal_width=self.systemConfigurations_info[
                                                                   "crystal_pitch_x"],
                                                               crystal_height=self.systemConfigurations_info[
                                                                   "crystal_pitch_y"],
                                                               shuffle=self.shuffle_data, FoV=self.fov,
                                                               distance_between_motors=self.systemConfigurations_info[
                                                                   "distance_between_motors"],
                                                               distance_crystals=self.systemConfigurations_info[
                                                                   "distance_between_crystals"],
                                                               crystal_depth=self.systemConfigurations_info[
                                                                   'crystal_length'],
                                                               recon2D=self.recon2D,
                                                               number_of_neighbours=self.number_of_neighbours,
                                                               generated_files=False,
                                                               transform_into_positive=self.transform_into_positive)

        self.planes_equation = ParallelepipedProjector(self.parametric_coordinates, pixelSizeXY=self.pixelSizeXY,
                                                       pixelSizeXYZ=self.pixelSizeXYZ,
                                                       crystal_width=self.systemConfigurations_info["crystal_pitch_x"],
                                                       crystal_height=self.systemConfigurations_info["crystal_pitch_y"],
                                                       reflector_xy=self.systemConfigurations_info[
                                                        'reflector_interior_A_y'],
                                                       reflector_z=self.systemConfigurations_info[
                                                        'reflector_interior_A_x'],
                                                       FoV=self.fov,
                                                       bool_consider_reflector_in_z_projection=
                                                    self.bool_consider_reflector_in_z_projection,
                                                       bool_consider_reflector_in_xy_projection=
                                                    self.bool_consider_reflector_in_xy_projection,
                                                       distance_crystals=self.systemConfigurations_info[
                                                        "distance_between_crystals"])
        self.ctx = None
        self.device = None
        self.cuda = cuda
        self.cuda.init()

    def geometry_file(self):
        MatrixCorrection = MatrixGeometryCorrection(operation='r',
                                                    file_path=os.path.join(os.path.dirname(
                                                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                        'system_configurations', 'x_{}__y_{}'.format(
                                                            self.crystals_geometry[0], self.crystals_geometry[1])))
        self._geometry_file = MatrixCorrection.coordinates
        return self._geometry_file

    def start(self):
        # Graphic Hardware initialization

        # CORRECTIONS
        # Normalization Matrix
        self.ctx = cuda.Device(0).make_context()
        self.device = self.ctx.get_device()
        [reading_data_cutted, decay_factor_cutted,
         planes_equations_cutted] = self._pre_processing_list_mode_whole_body()
        self.detector_normalization()
        """ Reconstruction :types"""
        self._whole_body_reconstruction(reading_data_cutted=reading_data_cutted,
                                        decay_factor_cutted=decay_factor_cutted,
                                        planes_equations_cutted=planes_equations_cutted,
                                        activation_bool=self.type_of_reconstruction[0])
        self._static_reconstruction(self.type_of_reconstruction[1])
        self._dynamic_reconstruction(self.type_of_reconstruction[2])
        self._gated_reconstruction(self.type_of_reconstruction[3])

        self.ctx.detach()
        if self.signals_interface is not None:
            self.signals_interface.trigger_update_label_reconstruction_status.emit(
                "Done")

    def _generate_folders(self):
        path_whole_body = os.path.join(self.study_path, "whole_body")
        path_static = os.path.join(self.study_path, "static_image")
        path_dynamic = os.path.join(self.study_path, "dynamic_image")
        path_gated = os.path.join(self.study_path, "gated_image")
        paths = [path_whole_body, path_static, path_dynamic, path_gated]
        for path in paths:
            if not os.path.isdir(path):
                os.makedirs(path)

    def _algorithm(self, algorithm, number_of_iterations, number_of_subsets):
        if algorithm == "FBP":
            self.transform_into_positive = False
            self.detector_normalization_correction = False

        elif algorithm == "SART":
            self.transform_into_positive = False
            self.detector_normalization_correction = False

        elif algorithm == 'LM-MLEM':
            self.number_of_iterations = number_of_iterations
            self.number_of_subsets = 1

        elif algorithm == 'LM-OSEM':
            self.number_of_iterations = number_of_iterations
            self.number_of_subsets = number_of_subsets
            self.shuffle_data = True

        elif algorithm == 'MLEM':
            self.calculate_histogram = True
            self.number_of_iterations = number_of_iterations
            self.number_of_subsets = 1

        elif algorithm == 'OSEM':
            self.calculate_histogram = True
            self.number_of_iterations = number_of_iterations
            self.number_of_subsets = number_of_subsets
            self.shuffle_data = True

        elif algorithm == 'LM-MRP':
            self.number_of_iterations = number_of_iterations
            self.number_of_subsets = number_of_subsets
            self.shuffle_data = False
            # algorithm_options = [beta, local_median_v_number]

        elif algorithm == 'ipet':
            ### passar para frente da normalização
            pixeltoangle = True
            self.number_of_iterations = number_of_iterations
            self.number_of_subsets = 1
            sizefile_m = [80, 80, 64]

        elif algorithm == 'MAP':
            self.save_map_from_pet = True

        return algorithm

    def remove_turns_for_reconstruction(self, ):
        """
        Generalizar para timestamps !!
        """
        time_indexes = self.acquisitionInfo["Turn end index"]
        if self.remove_turns["Cut_per_time"] is True:
            time_init_cut = self.remove_turns["Init time"]
            time_end_cut = self.remove_turns["End time"]

            time_per_turns = self.reading_data[time_indexes, 6]
            diff_init_time = np.abs(time_init_cut - time_per_turns)
            diff_init_index = time_indexes[np.where(diff_init_time == np.min(diff_init_time))[0][0]]
            diff_end_time = np.abs(time_end_cut - time_per_turns)
            diff_end_index = time_indexes[np.where(diff_end_time == np.min(diff_end_time))[0][0]]
        else:
            diff_init_index = time_indexes[int(self.remove_turns["Init time"])]
            diff_end_index = time_indexes[int(self.remove_turns["End time"])]
        reading_data_cutted = self.reading_data[diff_init_index:diff_end_index, :]
        planes_equations_cutted = copy.copy(self.planes_equation)
        planes_equations_cutted.cut_current_frame(diff_init_index, diff_end_index)
        decay_factor_cutted = self.decay_factor[diff_init_index:diff_end_index]
        print("Image cutted between: {} and {} s".format(reading_data_cutted[0, 6], reading_data_cutted[-1, 6]))
        return reading_data_cutted, decay_factor_cutted, planes_equations_cutted

    def _pre_processing_list_mode_whole_body(self):
        if self.remove_turns["Whole body"]:
            reading_data_cutted, decay_factor_cutted, planes_equations_cutted = self.remove_turns_for_reconstruction()
        else:
            reading_data_cutted = self.reading_data
            # decay_factor_cutted = self.decay_factor
            planes_equations_cutted = copy.copy(self.planes_equation)

        self.decay_correction_class = DecayCorrection(listMode=reading_data_cutted,
                                                      acquisition_info=self.acquisitionInfo,
                                                      correct_decay=self.correctDecayBool)
        self.decay_correction_class.list_mode_decay_correction()
        self.decay_correction_class.activity_on_subject_at_scanning_time()
        decay_factor_cutted = self.decay_correction_class.decay_factor
        self.scan_time = reading_data_cutted[-1, 6] - reading_data_cutted[0, 6]
        print("Scan time: {} s".format(self.scan_time))
        print("Initial time: {} s".format(reading_data_cutted[0, 6]))
        print("Final time: {} s".format(reading_data_cutted[-1, 6]))
        # self.scan_time = reading_data_cutted[-1, 6]
        repetir_distribuition = False
        if repetir_distribuition:
            listMode_part = np.copy(reading_data_cutted)
            n = 4
            listMode_part[:, 5] = listMode_part[:, 5] - 1.8/n
            # probability_side_A = np.histogram((reading_data[:, 2]+reading_data[:, 3])/2,64)[0]/len(reading_data)
            # probability_side_B = np.histogram((reading_data[:, 3]),64)[0]/len(reading_data)
            # self.reading_data[:,2] = np.random.choice(number_of_crystals[0]*number_of_crystals[1], len(self.reading_data), p=probability)+1
            # self.reading_data[:,3] = np.random.choice(number_of_crystals[0]*number_of_crystals[1], len(self.reading_data), p=probability)+1
            copy_ = np.copy(decay_factor_cutted)

            for i in range(n):
                listMode_part[:, 5] = listMode_part[:, 5] + 0.9/n
                # listMode_part[:, 4] = listMode_part[:, 4] + 0.9/n
                # listMode_part[:, 5] = listMode_part[:, 5] + 0.1125
                # listMode_part[:, 2] = np.random.choice(64, len(listMode_part), p=probability_side_A)+1
                # listMode_part[:, 3] = np.random.choice(64, len(listMode_part), p=probability_side_A)+1
                reading_data_cutted = np.append(reading_data_cutted, listMode_part, axis=0)
                decay_factor_cutted = np.append(decay_factor_cutted, copy_ , axis=0)

            # listMode_part = np.load("list_mode_recreate.npy")
            # listMode_part = np.copy(reading_data_cutted)
            # listMode_part[:, 5] = listMode_part[:, 5] - 0.45/16
            # # probability_side_A = np.histogram((reading_data[:, 2]+reading_data[:, 3])/2,64)[0]/len(reading_data)
            # # probability_side_B = np.histogram((reading_data[:, 3]),64)[0]/len(reading_data)
            # # self.reading_data[:,2] = np.random.choice(number_of_crystals[0]*number_of_crystals[1], len(self.reading_data), p=probability)+1
            # # self.reading_data[:,3] = np.random.choice(number_of_crystals[0]*number_of_crystals[1], len(self.reading_data), p=probability)+1
            # copy_ = np.copy(decay_factor_cutted)
            # n=16
            for i in range(n):
                listMode_part[:, 5] = listMode_part[:, 5] + 0.9/n
                # listMode_part[:, 4] = listMode_part[:, 4] + 0.9/n
                # listMode_part[:, 5] = listMode_part[:, 5] + 0.1125
                # listMode_part[:, 2] = np.random.choice(64, len(listMode_part), p=probability_side_A)+1
                # listMode_part[:, 3] = np.random.choice(64, len(listMode_part), p=probability_side_A)+1
                reading_data_cutted = np.append(reading_data_cutted, listMode_part, axis=0)
                decay_factor_cutted = np.append(decay_factor_cutted, copy_ , axis=0)
        # # np.save("C:\\Users\\pedro.encarnacao\\Desktop\\reading_data", reading_data)
        #
        regularizHist = False
        if regularizHist:
            n_z = 1
            n_t = 4
            binx = int(self.crystals_geometry[0] * self.crystals_geometry[1])
            binz = int(360 / (self.header[1] / self.header[2]) / n_z)
            binz = len(np.unique(reading_data_cutted[:, 4]))
            bint = int(self.header[5] / (self.header[3] / self.header[4]) / n_t)
            print("Smallest TOP Angle: {}".format(self.header[3] / self.header[4] * n_t))
            bins = [binx, binx, binz, bint]
            print(bins)
            output = np.histogramdd(reading_data_cutted[:, 2:6], [binx, binx, binz, bint])
            from scipy import ndimage
            out = ndimage.median_filter(np.sum(np.sum(output[0], axis=1), axis=0), size=3)
            output[0][:, :, out < 0.01 * np.max(out)] = 0
            # output = np.histogramdd(reading_data_cutted[:, 2:6], bins=[64, 64, 200, 200], range=None, normed=None, weights=None,
            #                         density=None)
            # # output[0][np.sum(np.sum(output[0],axis=3), axis=2)<20] = 0
            # from scipy import ndimage, misc
            # output[0][:, :, ndimage.median_filter(np.sum(np.sum(output[0], axis=1), axis=0), size=3) < 40] = 0
            # hist_shap = output[0].flatten().astype(int)
            list_mode_recreate = np.ones((int(np.sum(output[0])), 7))
            decay_factor_cutted = np.ones(int(np.sum(output[0])))
            el = 0
            for i in range(len(output[1][0]) - 1):
                print(i)
                i_element = np.ceil(output[1][0][i])
                for j in range(len(output[1][1]) - 1):
                    j_element = np.ceil(output[1][1][j])
                    for k in range(len(output[1][2]) - 1):
                        k_element = output[1][2][k]
                        for t in range(len(output[1][3]) - 1):
                            t_element = output[1][3][t]

                            if output[0][i, j, k, t] != 0:

                                for v in range(int(output[0][i, j, k, t])):
                                    list_mode_recreate[el + v, 2] = i_element
                                    list_mode_recreate[el + v, 3] = j_element
                                    list_mode_recreate[el + v, 4] = k_element
                                    list_mode_recreate[el + v, 5] = t_element
                                el += int(output[0][i, j, k, t])
            reading_data_cutted = list_mode_recreate
        self.fov = np.sin(
            np.radians((np.max(reading_data_cutted[:, 5]) + np.abs(np.min(reading_data_cutted[:, 5]))) / 2 +
                       4 * np.degrees(np.arctan(np.max(self._geometry_file[1]) /
                                                self.systemConfigurations_info[
                                                    "distance_between_crystals"])))) * \
                   self.systemConfigurations_info["distance_between_crystals"]
        # np.save("list_mode_recreate.npy", list_mode_recreate)
        return reading_data_cutted, decay_factor_cutted, planes_equations_cutted

    def _whole_body_reconstruction(self, reading_data_cutted=None, decay_factor_cutted=None,
                                   planes_equations_cutted=None,
                                   activation_bool=False, apply_quantification_factor=True, extension=".T"):
        if not activation_bool:
            return
        print("Reconstructing Whole body")
        folder_name = "whole_body"
        current_info_step = "Whole body"
        self.current_type_of_reconstruction = "WHOLE BODY"
        if self.save_dicom:
            path_dcm = os.path.join(self.study_path, folder_name)
            d_f = WriteDicom(self, path_dcm=path_dcm)
            if self.save_by_iteration:
                d_f_by_iteration = [WriteDicom(self, path_dcm=path_dcm)
                                    for _ in range(int(self.number_of_iterations // 5))]

        if self.signals_interface is not None:
            self.signals_interface.trigger_update_label_reconstruction_status.emit(
                "Whole body: Reconstruction started")

        if self.algorithm == "FBP":
            fbp_class = AnalyticalReconstruction(reading_data=self.reading_data,
                                                 parametric_coordinates=self.parametric_coordinates)
            fbp_class.FBP2D()

            self.im = fbp_class.im + np.abs(fbp_class.im.min())  # DICom só aceita positios

        elif self.algorithm == "SART":
            sart_class = AnalyticalReconstruction(reading_data=self.reading_data,
                                                  parametric_coordinates=self.parametric_coordinates)
            sart_class.SART()
            self.im = sart_class.im

        else:
            self.adaptativedoimap = AdaptativeDOIMapping(listMode=reading_data_cutted)
            self.adaptativedoimap.load_doi_files()
            # self.adaptativedoimap.generate_listmode_doi_values()
            self.im = EM(algorithm=self.algorithm, algorithm_options=self.algorithm_options,
                         normalization_matrix=self.normalization_matrix,
                         attenuation_correction=self.attenuation_correction,
                         attenuation_map=self.attenuation_data,
                         time_correction=decay_factor_cutted, doi_correction=self.doi_correction,
                         planes_equation=planes_equations_cutted, number_of_iterations=self.number_of_iterations,
                         number_of_subsets=self.number_of_subsets,
                         directory=self.study_path, cuda_drv=cuda, pixeltoangle=self.pixeltoangle,
                         easypetdata=self.reading_data, saved_image_by_iteration=True,
                         multiple_kernel=self.multiple_kernel,
                         entry_im=self.entry_im, signals_interface=self.signals_interface,
                         current_info_step=current_info_step, doi_mapping=self.adaptativedoimap).im
            # self.im = self.normalization_matrix

        if self.signals_interface is not None:

            self.signals_interface.trigger_projection_2d_offline.emit(False, self.live_recon, 0,
                                                                      [self.im], 0)
            if self.live_recon:
                self.signals_interface.trigger_live_images.emit()
        if self.multiple_save_conditions[0]:
            name_file_raw = '{}_ IMAGE {}Identifier{}'.format(self.basename_name_save,
                                                              self.im.shape, self.multiple_save_conditions[1],
                                                              extension)

        else:
            name_file_raw = '{}_ IMAGE {}{}'.format(self.basename_name_save,
                                                    self.im.shape, extension)
        name_file_numpy = "im.npy"
        self._save_files(file_folder=folder_name, file_name_raw=name_file_raw,
                         file_name_numpy=name_file_numpy, bq_ml=True,
                         apply_quantification_factor=apply_quantification_factor, time_series_id=0, dicom_obj=d_f)

        if self.save_by_iteration:
            if apply_quantification_factor:
                self._recreateImagesFromIterationsFolder(folder_name, d_f_by_iteration, time_series_id=0, name_file_raw_f=name_file_raw)

    def _dynamic_reconstruction(self, activation_bool=False):
        if not activation_bool:
            return

        self.current_type_of_reconstruction = "DYNAMIC"
        time_indexes = self.acquisitionInfo["Turn end index"]
        folder_name = "dynamic_image"
        if self.live_recon:
            time_frame_number = int(len(time_indexes) - 1)
            time_indexes = time_indexes[-2:]  # get the last segment
            print("Time_indexes live {}".format(time_indexes))
        number_cumulative_turns = int(self.number_cumulative_turns)
        self.dynamicFrames = (len(time_indexes) - number_cumulative_turns) // number_cumulative_turns
        print("DynamicFRAMES: " + str(self.dynamicFrames))
        if self.save_dicom:
            path_dcm = os.path.join(self.study_path, folder_name)
            d_f = WriteDicom(self, path_dcm=path_dcm)
            if self.save_by_iteration:
                d_f_by_iteration = [WriteDicom(self, path_dcm=path_dcm)
                                    for _ in range(int(self.number_of_iterations // 5))]

        if self.algorithm == "FBP":
            # parametric_coordinates_temp = copy.copy(self.parametric_coordinates)
            # parametric_coordinates_temp.cutCurrentFrame(time_indexes[i],
            #                                             time_indexes[i + number_cumulative_turns])
            fbp_class = AnalyticalReconstruction(reading_data=self.reading_data,
                                                 parametric_coordinates=self.parametric_coordinates,
                                                 pixel_size=self.pixelSizeXY)

        for i in range(0, len(time_indexes) - number_cumulative_turns, number_cumulative_turns):
            reading_data_partial = self.reading_data[time_indexes[i]:time_indexes[i + number_cumulative_turns], :]
            if len(reading_data_partial) > 2:
                current_info_step = "Dynamic: Frame {}s to {}s".format(int(np.round(reading_data_partial[0, 6], 0)),
                                                                       int(np.round(reading_data_partial[-1, 6], 0)))

                self.scan_time = reading_data_partial[-1, 6] - reading_data_partial[0, 6]
                print(current_info_step)
                if self.signals_interface is not None:
                    self.signals_interface.trigger_update_label_reconstruction_status.emit(
                        current_info_step)

                # self.adaptativedoimap.generate_listmode_doi_values()
                if self.algorithm == "FBP":
                    fbp_class.FBP2D(timecut=[time_indexes[i], time_indexes[i + number_cumulative_turns]])
                    self.im = fbp_class.im + np.abs(fbp_class.im.min())  # DICom só aceita positios
                    # self.im *= 100000

                elif self.algorithm == "SART":
                    sart_class = AnalyticalReconstruction(parent=self,
                                                          parametric_coordinates=self.parametric_coordinates)
                    sart_class.SART()
                    self.im = sart_class.im
                else:
                    planes_equation_temp = copy.copy(self.planes_equation)
                    planes_equation_temp.cut_current_frame(time_indexes[i], time_indexes[i + number_cumulative_turns])

                    self.adaptativedoimap = AdaptativeDOIMapping(listMode=reading_data_partial)
                    self.adaptativedoimap.load_doi_files()
                    self.im = EM(algorithm=self.algorithm, algorithm_options=self.algorithm_options,
                                 normalization_matrix=self.normalization_matrix,
                                 time_correction=self.decay_factor[
                                                 time_indexes[i]:time_indexes[i + number_cumulative_turns]],
                                 planes_equation=planes_equation_temp, number_of_iterations=self.number_of_iterations,
                                 number_of_subsets=self.number_of_subsets,
                                 directory=self.study_path, cuda_drv=cuda, pixeltoangle=self.pixeltoangle,
                                 easypetdata=self.reading_data, saved_image_by_iteration=self.save_by_iteration,
                                 multiple_kernel=self.multiple_kernel,
                                 entry_im=self.entry_im, signals_interface=self.signals_interface,
                                 current_info_step=current_info_step, doi_mapping=self.adaptativedoimap).im

                if not self.live_recon:
                    time_frame_number = int(i + 1)
                if self.signals_interface is not None:

                    self.signals_interface.trigger_projection_2d_offline.emit(True, self.live_recon,
                                                                              self.reading_data[time_indexes[i + 1], 6],
                                                                              [self.im], time_frame_number)
                    if self.live_recon:
                        self.signals_interface.trigger_live_images.emit()

                    # signals_interface.trigger_projection_2d_offline.emit(5, False)
                if self.multiple_save_conditions[0]:
                    name_file_raw = '{}_ IMAGE{} FRAME{}Identifier{}.T'.format(self.basename_name_save,
                                                                               self.im.shape, i,
                                                                               self.multiple_save_conditions[1])
                else:
                    name_file_raw = '{}_ IMAGE {} FRAME {}.T'.format(self.basename_name_save,
                                                                     self.im.shape, i)

                name_file_numpy = "im_{}.npy".format(time_frame_number)

                self._save_files(file_folder=folder_name, file_name_raw=name_file_raw,
                                 file_name_numpy=name_file_numpy, bq_ml=True, time_series_id=i, dicom_obj=d_f)
                if self.save_by_iteration:
                    self._recreateImagesFromIterationsFolder(folder_name, d_f_by_iteration, i, name_file_raw)

    def _static_reconstruction(self, activation_bool=False):
        if not activation_bool:
            return

    def _gated_reconstruction(self, activation_bool=False):
        """ Future implementation """

    def _volume_voxel(self, im_en=None):
        if im_en is None:
            im_en = self.im
        real_pixelSizeXYZ = (self.systemConfigurations_info["array_crystal_x"] * self.systemConfigurations_info[
            "crystal_pitch_x"] +
                             (self.systemConfigurations_info["array_crystal_x"] - 1) *
                             2 * self.systemConfigurations_info["reflector_interior_A_x"]) / im_en.shape[2]

        self.volume_voxel = self.pixelSizeXY * self.pixelSizeXY * real_pixelSizeXYZ * 0.001
        return self.volume_voxel

    def detector_normalization(self, generate_probability=False):
        if not self.detector_normalization_correction:
            return

        if self.detector_normalization_data == "Pre-calculated":
            try:
                self.normalization_matrix = self._load_normalization_matrix(normalized=False)
                # self.normalization_matrix = self._load_sens_matrix(normalized=False, simulation=True)
                # self.sensitivity_matrix = None
            except FileNotFoundError:
                print('File not found')

                # self.sensitivity_matrix = None
                self.normalization_matrix = None
            # self.normalization_matrix = None

        elif self.detector_normalization_data == "On demand parametric calculation":
            print("Generating Normalization map")
            self.normalization_matrix = None
            # self.sensitivity_matrix = None

            if self.signals_interface is not None:
                self.signals_interface.trigger_update_label_reconstruction_status.emit(
                    "Normalization Map: Generating positions")
                self.signals_interface.trigger_progress_reconstruction_partial.emit(
                    int(20))
            anMatrix = AdaptiveNormalizationMatrix(self.reading_data, number_of_crystals=self.crystals_geometry,
                                                   number_of_reps=int(6*6),
                                                   recon_2D=self.recon2D, rangeTopMotor=self.header[5],
                                                   stepTopmotor=self.header[3] / self.header[4])

            #24,24
            if generate_probability:
                print("generate_probability")
                anMatrix.write_probability_phantom()
            anMatrix.normalization_LM()

            reading_data_adaptative_matrix = anMatrix.reading_data
            if self.signals_interface is not None:
                self.signals_interface.trigger_update_label_reconstruction_status.emit(
                    "Normalization Map: Setting Parametric Positions")
                self.signals_interface.trigger_update_label_reconstruction_status.emit(
                    "Normalization Map: Generating positions")
                self.signals_interface.trigger_progress_reconstruction_partial.emit(
                    int(40))

            parametric_coordinates = SetParametricCoordinates(listMode=reading_data_adaptative_matrix,
                                                              geometry_file=self._geometry_file,
                                                              simulation_files=self.simulation_file,
                                                              crystal_width=self.systemConfigurations_info[
                                                                  "crystal_pitch_x"],
                                                              crystal_height=self.systemConfigurations_info[
                                                                  "crystal_pitch_y"],
                                                              shuffle=self.shuffle_data, FoV=self.fov,
                                                              distance_between_motors=self.systemConfigurations_info[
                                                                  "distance_between_motors"],
                                                              distance_crystals=self.systemConfigurations_info[
                                                                  "distance_between_crystals"],
                                                              crystal_depth=self.systemConfigurations_info[
                                                                  'crystal_length'],
                                                              recon2D=self.recon2D,
                                                              number_of_neighbours=self.number_of_neighbours,
                                                              generated_files=False,
                                                              normalization=True)

            if self.signals_interface is not None:
                self.signals_interface.trigger_update_label_reconstruction_status.emit(
                    "Normalization Map: Calculating plane equations")
                self.signals_interface.trigger_progress_reconstruction_partial.emit(
                    int(60))
            planes_equation = ParallelepipedProjector(parametric_coordinates, pixelSizeXY=self.pixelSizeXY,
                                                      pixelSizeXYZ=self.pixelSizeXYZ,
                                                      crystal_width=self.systemConfigurations_info[
                                                       "crystal_pitch_x"],
                                                      crystal_height=self.systemConfigurations_info[
                                                       "crystal_pitch_y"],
                                                      reflector_xy=self.systemConfigurations_info[
                                                       'reflector_interior_A_y'],
                                                      reflector_z=self.systemConfigurations_info[
                                                       'reflector_interior_A_x'], FoV=self.fov,
                                                      bool_consider_reflector_in_z_projection=False,
                                                      bool_consider_reflector_in_xy_projection=False,
                                                      distance_crystals=self.systemConfigurations_info[
                                                       "distance_between_crystals"],
                                                      max_center_position=self.max_possible_position)
            planes_equation.x_range_lim = self.planes_equation.x_range_lim
            planes_equation.y_range_lim = self.planes_equation.y_range_lim
            planes_equation.z_range_lim = self.planes_equation.z_range_lim
            planes_equation.number_of_pixels_x = self.planes_equation.number_of_pixels_x
            planes_equation.number_of_pixels_y = self.planes_equation.number_of_pixels_y
            planes_equation.number_of_pixels_z = self.planes_equation.number_of_pixels_z
            planes_equation.im_index_x = self.planes_equation.im_index_x
            planes_equation.im_index_y = self.planes_equation.im_index_y
            planes_equation.im_index_z = self.planes_equation.im_index_z
            self.adaptativedoimap = AdaptativeDOIMapping(listMode=reading_data_adaptative_matrix)
            self.adaptativedoimap.load_doi_files()
            # self.adaptativedoimap.generate_listmode_doi_values()

            if self.signals_interface is not None:
                self.signals_interface.trigger_update_label_reconstruction_status.emit(
                    "Normalization Map: Generating map")
                self.signals_interface.trigger_progress_reconstruction_partial.emit(
                    int(80))
            self.normalization_matrix = EM(algorithm=self.algorithm, algorithm_options=self.algorithm_options,
                                           normalization_matrix=self.normalization_matrix,
                                           time_correction=np.ones((len(reading_data_adaptative_matrix[:, 1]))),
                                           planes_equation=planes_equation, number_of_iterations=1,
                                           number_of_subsets=1, directory=self.study_path,
                                           cuda_drv=cuda, pixeltoangle=False,
                                           easypetdata=reading_data_adaptative_matrix, saved_image_by_iteration=True,
                                           multiple_kernel=self.multiple_kernel,
                                           signals_interface=self.signals_interface,
                                           entry_im=None,
                                           normalization_calculation_flag=True,
                                           doi_mapping=self.adaptativedoimap).im
            # self.normalization_matrix = None
            # self.normalization_matrix = self.normalization_matrix /4096
            # self.normalization_matrix_simu = self._load_normalization_matrix(normalized=False)
            # self.normalization_matrix_simu = self.normalization_matrix_simu / np.max(self.normalization_matrix_simu)

            self.normalization_matrix[self.normalization_matrix < 1*10**(-5)] = 0
            # self.normalization_matrix[self.normalization_matrix < 1*10**-1] = 0
            self.normalization_matrix = self.normalization_matrix / (np.sum(self.normalization_matrix))

            if self.signals_interface is not None:
                self.signals_interface.trigger_progress_reconstruction_partial.emit(
                    int(100))

        elif self.detector_normalization_data == "Stored parametric calculation":
            self.normalization_matrix = self._load_sens_matrix(normalized=False, simulation=True)

    def _recreateImagesFromIterationsFolder(self, folder_name, dicom_objs_list, time_series_id=0, name_file_raw_f=None):
        print("Recreate Dicom")
        folder = os.path.join(self.study_path, "iterations")
        # find files in folder

        files = glob.glob(os.path.join(folder, "*"))
        size_file_m = self.im.shape
        i = 0
        for file in files[:len(dicom_objs_list)]:
            name_file_raw = f"{os.path.basename(file)}_{name_file_raw_f}"
            r_it = RawDataSetter(file, size_file_m=size_file_m)
            r_it.read_files()
            self.im = r_it.volume
            self._save_files(file_folder=folder_name, file_name_raw=name_file_raw,
                             file_name_numpy="None", bq_ml=True, time_series_id=time_series_id,
                             dicom_obj=dicom_objs_list[i])
            i += 1

    def _save_files(self, file_folder=None, file_name_raw=None, file_name_numpy=None, bq=False, bq_ml=False,
                    apply_quantification_factor=True, time_series_id=0, dicom_obj=None):

        if apply_quantification_factor:
            self.apply_system_quantification_factor()
        if bq:
            self.convert_to_bq()
        elif bq_ml:
            self.convert_to_bq_ml()
        #
        if self.feet_first:
            self.im = np.rot90(self.im, 2, (2, 0))

        if self.save_numpy:
            # np.save(os.path.join(study_path, 'static_image', os.path.basename(study_path)), self.im)
            np.save(os.path.join(self.study_path, file_folder, file_name_numpy), self.im)

        if self.save_raw_data:
            volume = self.im.astype(np.float32)
            length = 1
            for i in volume.shape:
                length *= i
            # length = volume.shape[0] * volume.shape[2] * volume.shape[1]
            if len(volume.shape) > 1:
                data = np.reshape(volume, [1, length], order='F')
            else:
                data = volume
            output_file = open(os.path.join(self.study_path, file_folder, file_name_raw), 'wb')
            arr = array('f', data[0])

            arr.tofile(output_file)
            output_file.close()
        if self.save_dicom:
            dicom_obj.updateVolume(volume)
            dicom_obj.updatetimeIdSeries(time_series_id)
            dicom_obj.write_dicom_file()

    def _override_geometry_file_func(self, override_geometric_values=None):
        geometry_file = self.geometry_file()
        crystals_geometry = self.crystals_geometry
        height = self.systemConfigurations_info["crystal_pitch_x"]
        crystal_width = self.systemConfigurations_info["crystal_pitch_y"]
        reflector_y = 2 * self.systemConfigurations_info["reflector_interior_A_y"]
        # reflector_y = self.systemConfigurations_info["reflector_interior_A_y"]#simula
        # geometry_file[:crystals_geometry[0] * crystals_geometry[1], 1] += 1
        # geometry_file[:, 1] = np.tile(np.round(np.arange(0,crystals_geometry[1]-1,0.8)-2.4,3),crystals_geometry[0])
        geometry_file[:, 1] = np.tile(np.round(np.arange(0, crystals_geometry[1] * crystal_width + 2 * reflector_y,
                                                         crystal_width + reflector_y) - (crystal_width + reflector_y) *
                                               (crystals_geometry[1] - 1) / 2, 3), crystals_geometry[0] * 2)
        if not self.simulation_file:
            geometry_file[crystals_geometry[0] * crystals_geometry[1]:, 1] *= -1
        # geometry_file[crystals_geometry[0] * crystals_geometry[1]:,1] += 0.2
        # # geometry_file[:crystals_geometry[0] * crystals_geometry[1],1] *= -1
        # geometry_file[:,1] += 0.2
        # geometry_file[:crystals_geometry[0] * crystals_geometry[1],1] *= -1
        # geometry_file[:,1] *= -1
        # if self.simulation_file:
        #     geometry_file[0::crystals_geometry[1], 1] = override_geometric_values[0][0]  # simulation
        #     geometry_file[1::crystals_geometry[1], 1] = override_geometric_values[0][1]
        #     geometry_file[crystals_geometry[0] * 2::crystals_geometry[1], 1] = override_geometric_values[0][3]
        #     geometry_file[crystals_geometry[0] * 2 + 1::crystals_geometry[1], 1] = override_geometric_values[0][2]
        # else:
        #     geometry_file[0::crystals_geometry[1], 1] = override_geometric_values[0][0]  # real
        #     geometry_file[1::crystals_geometry[1], 1] = override_geometric_values[0][1]
        #     geometry_file[crystals_geometry[0] * 2::crystals_geometry[1], 1] = override_geometric_values[0][2]
        #     geometry_file[crystals_geometry[0] * 2 + 1::crystals_geometry[1], 1] = override_geometric_values[0][3]

        # geometry_file[0::crystals_geometry[1], 1] = 1.175  # simulation
        # geometry_file[1::crystals_geometry[1], 1] = -1.175
        # geometry_file[crystals_geometry[0]*2::crystals_geometry[1], 1] = -1.175  # simulation
        # geometry_file[crystals_geometry[0]*2+1::crystals_geometry[1], 1] = 1.175

        # ttt_mm = 0
        # geometry_file[0:crystals_geometry[0] * crystals_geometry[1], 0] = -np.arange(0,
        #                                                                              crystals_geometry[0] *
        #                                                                              crystals_geometry[
        #                                                                                  1]) * (
        #                                                                           ttt_mm / crystals_geometry[
        #                                                                       0])  # simulation
        #
        # geometry_file[
        # crystals_geometry[0] * crystals_geometry[1]:crystals_geometry[0] * crystals_geometry[1] * 2,
        # 0] = np.arange(0, crystals_geometry[0] * crystals_geometry[1]) * (ttt_mm / crystals_geometry[0])

        z = np.repeat(np.arange(0, crystals_geometry[0] * height, height), crystals_geometry[1])

        geometry_file[0:crystals_geometry[0] * crystals_geometry[1], 2] = (z + 1)
        ## add 1.5 for 2019 aqusitions
        # geometry_file[32:64, 2] = z + 2.5
        geometry_file[
        crystals_geometry[0] * crystals_geometry[1]:crystals_geometry[0] * crystals_geometry[1] * 2,
        2] = (z + 1)

        self._geometry_file = geometry_file

    def convert_to_bq(self):
        self.im /= self.scan_time

    def convert_to_bq_ml(self):
        if self.scan_time == 0:
            self.scan_time = 1
        self.im /= self.scan_time
        self._volume_voxel(self.im)
        self.im /= self.volume_voxel

    def apply_system_quantification_factor(self):
        self._volume_voxel()
        quant = FactorQuantificationFromUniformPhantom(crystals_geometry=self.crystals_geometry,
                                                       voxel_volume=self.volume_voxel)
        quant.load_info()

        self.im *= quant.quantification_factor
        print("Quantification Factor: {}".format(quant.quantification_factor))

    def generate_probability_and_quantification_factors(self):
        """ """
        self.ctx = cuda.Device(0).make_context()
        self.device = self.ctx.get_device()

        [reading_data_cutted, decay_factor_cutted,
         planes_equations_cutted] = self._pre_processing_list_mode_whole_body()
        self.decay_correction_class = DecayCorrection(listMode=reading_data_cutted,
                                                      acquisition_info=self.acquisitionInfo,
                                                      correct_decay=self.correctDecayBool)
        self.decay_correction_class.list_mode_decay_correction()
        decay_factor_cutted = self.decay_correction_class.decay_factor
        self.detector_normalization(generate_probability=True)
        """ Reconstruction :types"""
        self._whole_body_reconstruction(reading_data_cutted=reading_data_cutted,
                                        decay_factor_cutted=decay_factor_cutted,
                                        planes_equations_cutted=planes_equations_cutted,
                                        activation_bool=True,
                                        apply_quantification_factor=False, extension=".quant")
        # self._whole_body_reconstruction(self.type_of_reconstruction[0], apply_quantification_factor=False)
        image_file = max(glob.iglob(
            os.path.join(self.study_path, "whole_body",
                         '*.quant')), key=os.path.getctime)

        data = RawDataSetter(file_name=image_file)
        data.read_files()
        volume = data.volume
        volume= self.im
        # self._volume_voxel(im_en=volume)

        self.acquisitionInfo["Volume tracer"] = 73
        # activity_at_measure_time = float(self.acquisitionInfo["Total Dose"])
        corrected_activity = self.decay_correction_class.activity_on_subject_at_scanning_time()
        f = FactorQuantificationFromUniformPhantom(activity_phantom=corrected_activity,
                                                   radiotracer_phantom=self.acquisitionInfo['Tracer'],
                                                   positron_fraction_phantom=self.acquisitionInfo['Positron Fraction'],
                                                   phantom_volume=float(self.acquisitionInfo["Volume tracer"]),
                                                   crystals_geometry=self.crystals_geometry,
                                                   image_phantom=volume,
                                                   acquisition_phantom_duration=self.scan_time,
                                                   voxel_volume=self.volume_voxel, voxel_volume_unit="ml")
        f.segment_region_phantom(voi=8, dx=10, dy=10, dz=10)
        f.quantification_factor_calculation(bq_ml=True)
        f.save_info()
        self.ctx.detach()


if __name__ == "__main__":
    # a = ReconstructionInitializer.__init__
    r = ReconstructionInitializer()
    # print(r.__file__)
    # print(r.__name__)

    print(r.__module__)
    # print(a.__code__.co_varnames)
    # print(r.__func__)
    # print(a.__defaults__)
    # print(a.__code__)

    # b = inspect.signature(a)
    # print(inspect.signature(a))
