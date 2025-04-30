import numpy as np
import os
from array import array
import time
import math
import pycuda.driver as cuda
# import pycuda.autoinit
from pycuda.compiler import SourceModule

# from ..Segmentation.findisosurface import FindISOSurface
from .selectable_events import ROIEvents
from .CPU import IterativeAlgorithmCPU
from .gpu_shared_memory import GPUSharedMemorySingleKernel
from .gpu_multiple_kernel import GPUSharedMemoryMultipleKernel


class EM:
    def __init__(self, easypetdata=None, planes_equation=None, algorithm="LM-MLEM", algorithm_options=None,
                 number_of_iterations=10, number_of_subsets=1, projector_type="Box Counts",
                 normalization_matrix=None, attenuation_correction=False, attenuation_map=None,
                 decay_correction=False, time_correction=None, doi_correction=None, doi_mapping=None,
                 normalization_calculation_flag=False, random_correction=False, scatter_correction=False,
                 scatter_angle_correction=False, cut_fov=True,
                 cuda_drv=None, GPU=True, directory=None,
                 shared_memory=True, saved_image_by_iteration=True, pixeltoangle=False,
                 entry_im=None, multiple_kernel=False, signals_interface=None,
                 current_info_step=""):

        if planes_equation is None:
            return

        if directory is None:
            return

        self.easypetdata = easypetdata

        self.algorithm = algorithm
        self.algorithm_options = algorithm_options
        self.cuda_drv = cuda_drv
        self.normalization_calculation_flag = normalization_calculation_flag
        self.signals_interface = signals_interface
        self.current_info_step = current_info_step
        self.directory = directory  # study directory

        self.number_of_events = planes_equation.number_of_events
        # Algorithm
        self.number_of_iterations = number_of_iterations
        self.number_of_subsets = number_of_subsets
        self.saved_image_by_iteration = saved_image_by_iteration

        # Optimizer
        self.shared_memory = shared_memory

        # Geometry

        # self.crystal_central_planes = planes_equation.crystal_planes
        self.number_of_pixels_x = planes_equation.number_of_pixels_x
        self.number_of_pixels_y = planes_equation.number_of_pixels_y
        self.number_of_pixels_z = planes_equation.number_of_pixels_z
        self.distance_to_center_plane = planes_equation.distance_to_central_plane
        self.distance_to_center_plane_normal = planes_equation.distance_to_central_plane_normal
        self.im_index_x = planes_equation.im_index_x
        self.im_index_y = planes_equation.im_index_y
        self.im_index_z = planes_equation.im_index_z
        #
        # self.p1_list = planes_equation.p1_list
        # self.p2_list = planes_equation.p2_list
        # self.p3_list = planes_equation.p3_list
        # self.p4_list = planes_equation.p4_list

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

        self.half_crystal_pitch_z = planes_equation.half_crystal_pitch_z
        self.distance_between_array_pixel = planes_equation.distance_between_array_pixel
        self.half_crystal_pitch_xy = planes_equation.half_crystal_pitch_xy

        self.x_min_f = planes_equation.x_min_f
        self.x_max_f = planes_equation.x_max_f
        self.y_min_f = planes_equation.y_min_f
        self.y_max_f = planes_equation.y_max_f
        self.z_min_f = planes_equation.z_min_f
        self.z_max_f = planes_equation.z_max_f

        self.voxelSize =[planes_equation.pixelSizeXY, planes_equation.pixelSizeXY, planes_equation.pixelSizeXYZ]


        # Corrections
        self.isosurface = None
        self.surface = None
        self.active_pixel_x = None
        self.active_pixel_y = None
        self.active_pixel_z = None

        self.attenuation_matrix = None
        self.adjust_coef = None
        self.sum_pixel = None

        self.doi_mapping = doi_mapping
        self.time_correction = time_correction
        self.time_correction = np.ascontiguousarray(time_correction, dtype=np.float32)

        self.projector_type = projector_type
        self.doi_correction = doi_correction
        self.decay_correction = decay_correction
        self.random_correction = random_correction
        self.scatter_correction = scatter_correction
        self.scatter_angle_correction = scatter_angle_correction

        self.roi_map = None

        if normalization_matrix is None:

            self.normalization_matrix = np.ascontiguousarray(
                np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
                        dtype=np.float32))

        else:
            normalize_sens_size = ((np.array(normalization_matrix.shape) - np.array(self.im_index_x.shape)) / 2).astype(
                np.int16)
            if normalize_sens_size[0] >= 0:
                self.normalization_matrix = normalization_matrix[
                                            normalize_sens_size[0]:(
                                                        normalization_matrix.shape[0] - normalize_sens_size[0]),
                                            normalize_sens_size[1]:(
                                                        normalization_matrix.shape[1] - normalize_sens_size[1]),
                                            normalize_sens_size[2]:(
                                                        normalization_matrix.shape[2] - normalize_sens_size[2])]
            else:
                mask = np.zeros((self.im_index_x.shape))

                normalize_sens_size_abs = np.abs(normalize_sens_size)
                rest = normalize_sens_size_abs % 2
                mask[normalize_sens_size_abs[0]:mask.shape[0] - normalize_sens_size_abs[0]- rest[0],
                normalize_sens_size_abs[1]:mask.shape[1] - normalize_sens_size_abs[1]-rest[1],
                normalize_sens_size_abs[2]:mask.shape[2] - normalize_sens_size_abs[2]-rest[2]] = normalization_matrix

                self.normalization_matrix = mask

            if self.normalization_matrix.shape != self.im_index_x.shape:
                self.normalization_matrix = self.normalization_matrix[1:self.im_index_x.shape[0] + 1,
                                            1: self.im_index_x.shape[1] + 1, 0:self.im_index_x.shape[2]]
            # self.normalization_matrix = np.ascontiguousarray(self.normalization_matrix, dtype=np.float32)
            self.normalization_matrix = np.ascontiguousarray(normalization_matrix, dtype=np.float32)

        if cut_fov:
            self.apply_circular_mask_fov()

        self.load_initial_arrays()

        if pixeltoangle:
            self.load_roi_map()
            if self.roi_map is not None:
                self.get_active_roi_voxels()
                self.remove_events_outside_ROI()
                self.im = self._fullFoVPreviousProbability()
                self.im = np.ascontiguousarray(self.im, dtype=np.float32)
                # self.normalization_matrix *= im
                # self.normalization_matrix /= np.sum(self.normalization_matrix)
                # self.normalization_matrix[self.normalization_matrix < 0.00001] = 0
                # self.im = np.ascontiguousarray(self.im, dtype=np.float32)
            # self.im = np.ascontiguousarray(
            #     np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
            #             dtype=np.float32))
            # self.im = self.roi_map / np.max(self.roi_map)
            #     self.im = self.surface / np.max(self.surface)
            #     self.im = np.ascontiguousarray(self.im, dtype=np.float32)
            self.fov_matrix_cut = self.surface
            self.fov_matrix_cut = np.ascontiguousarray(self.fov_matrix_cut, dtype=np.byte)

        # if self.normalization_calculation_flag:
        #     self.im = np.ascontiguousarray(
        #         np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
        #                 dtype=np.float32))
        #     # self.im = self.roi_map / np.max(self.roi_map)
        #     self.im = self.surface / np.max(self.surface)

        if GPU:

            tic = time.time()
            if not multiple_kernel:
                self.im = self._vor_design_gpu_shared_memory(self.a, self.a_normal, self.a_cf, self.im_index_x, self.b,
                                                             self.b_normal, self.b_cf, self.im_index_y, self.c,
                                                             self.c_normal, self.c_cf, self.im_index_z,
                                                             self.d, self.d_normal, self.d_cf, self.adjust_coef,
                                                             self.im, self.half_crystal_pitch_xy,
                                                             self.half_crystal_pitch_z, self.sum_pixel,
                                                             self.fov_matrix_cut,
                                                             self.normalization_matrix, self.time_correction)
            #     self.im = self._vor_design_gpu_shared_memory_multiple_reads(self.a, self.a_normal,self.a_cf, self.im_index_x, self.b, self.b_normal, self.b_cf, self.im_index_y, self.c, self.c_normal,self.c_cf,self.im_index_z,
            #                        self.d, self.d_normal, self.d_cf, self.adjust_coef, self.im, self.half_crystal_pitch_xy,
            #                                                  self.half_crystal_pitch_z, self.sum_pixel, self.sensivity_matrix,
            #                                                  self.normalization_matrix, self.time_correction)
            elif multiple_kernel:
                gpu_alg = GPUSharedMemoryMultipleKernel(self, optimize_reads_and_calcs=False)
                # gpu_alg.multikernel_optimized_memory_reads()
                gpu_alg.multiplekernel()
                self.im = gpu_alg.im
                # self.cdrf = True
                # self.im = self._vor_design_gpu_shared_memory_multiple_reads(self.a, self.a_normal, self.a_cf,
                #                                                             self.im_index_x, self.b,
                #                                                             self.b_normal, self.b_cf,
                #                                                             self.im_index_y, self.c,
                #                                                             self.c_normal, self.c_cf,
                #                                                             self.im_index_z,
                #                                                             self.d, self.d_normal, self.d_cf,
                #                                                             self.adjust_coef,
                #                                                             self.im, self.half_crystal_pitch_xy,
                #                                                             self.half_crystal_pitch_z,
                #                                                             self.sum_pixel,
                #                                                             self.fov_matrix_cut,
                #                                                             self.normalization_matrix,
                #                                                             self.time_correction)
                # # if use_half_precision:
                #     self.im = self._vor_design_gpu_shared_memory_multiple_reads_streamed(self.a, self.a_normal,
                #                                                                          self.a_cf, self.im_index_x,
                #                                                                          self.b,
                #                                                                          self.b_normal, self.b_cf,
                #                                                                          self.im_index_y, self.c,
                #                                                                          self.c_normal, self.c_cf,
                #                                                                          self.im_index_z,
                #                                                                          self.d, self.d_normal,
                #                                                                          self.d_cf, self.adjust_coef,
                #                                                                          self.im,
                #                                                                          self.half_crystal_pitch_xy,
                #                                                                          self.half_crystal_pitch_z,
                #                                                                          self.sum_pixel,
                #                                                                          self.fov_matrix_cut,
                #                                                                          self.normalization_matrix,
                #                                                                          self.time_correction)
                # else:

            toc = time.time()
            print('Optimizer CALCULATION OSEM: {}'.format(toc - tic))
            # np.save('im.npy', im)

            # self.vor_design_cpu()
            # self.im =self.im_cpu-self.im
        else:
            cpu_alg = IterativeAlgorithmCPU(self)
            self.im = cpu_alg.im
        if self.normalization_calculation_flag:
            self._save_image_by_it(self.im, normalization=True)

    def load_initial_arrays(self):
        self.attenuation_matrix = np.ascontiguousarray(
            np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
                    dtype=np.float32))

        self.adjust_coef = np.ascontiguousarray(
            np.zeros((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.float32))

        self.sum_pixel = np.ascontiguousarray(
            np.zeros(self.a.shape, dtype=np.float32))

        if self.normalization_calculation_flag:
            self.im = np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z)) * \
                      len(self.a) / (self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z)
            self.im = np.ascontiguousarray(self.im, dtype=np.float32)
            # self.im = np.ascontiguousarray(
            #     np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
            #             dtype=np.float32))
            # self.im = self.roi_map / np.max(self.roi_map)
        else:
            if self.roi_map is not None:
                self.im = np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z)) * \
                          len(self.a) / (self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z)
                self.im = np.ascontiguousarray(self.im, dtype=np.float32)
            else:
                self.im = np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z)) * \
                          len(self.a) / (self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z)
                self.im = np.ascontiguousarray(self.im, dtype=np.float32)

    def get_active_roi_voxels(self):
        self.isosurface = FindISOSurface(volume=self.roi_map, directory=self.directory)
        # self.isosurface.threshold3DISOSurface()
        self.isosurface.get_active_pixels()
        self.isosurface.save_calculated_surface()
        active_pixels = self.isosurface.active_pixels
        self.surface = self.isosurface.surface_volume
        self.active_pixel_x = active_pixels[0]
        self.active_pixel_y = active_pixels[1]
        self.active_pixel_z = active_pixels[2]
        self.active_pixel_x = np.ascontiguousarray(self.active_pixel_x, dtype=np.int32)
        self.active_pixel_y = np.ascontiguousarray(self.active_pixel_y, dtype=np.int32)
        self.active_pixel_z = np.ascontiguousarray(self.active_pixel_z, dtype=np.int32)
        self.sum_pixel = np.ascontiguousarray(
            np.zeros(self.a.shape, dtype=np.float32))

        # file_name = os.path.join(self.directory, "map_total")
        # sizefile = self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z
        # output_file = open(file_name, 'rb')
        # a = array('f')
        # a.fromfile(output_file, sizefile)
        # output_file.close()
        # self.roi_map_total = np.array(a)
        # self.roi_map_total = self.roi_map.reshape(
        #     (self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
        #     order='F')

        # self.im = np.ascontiguousarray(
        #     np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
        #             dtype=np.float32)) *self.surface

    def _fullFoVPreviousProbability(self):
        try:
            file_name = os.path.join(self.directory, "whole_body", "ID_26 Jan 2022 - 14h 57m 00s_None_ IMAGE (154, 154, 373).T")
            # file_name = os.path.join(self.directory, "map_heart")
            sizefile = self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z
            # sizefile = 101 * 101 * 153
            sizeMatrix = [154,154,373]
            sizefile = sizeMatrix[0]*sizeMatrix[1]*sizeMatrix[2]
            output_file = open(file_name, 'rb')
            a = array('f')
            a.fromfile(output_file, sizefile)
            output_file.close()
            map_probability= np.array(a)
            # self.roi_map = self.roi_map.reshape(
            #     (self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
            #     order='F')
            # self.roi_map = self.roi_map.reshape(
            #     (101, 101, 153),
            #     order='F')
            map_probability = map_probability.reshape(
                (sizeMatrix),
                order='F')
            import scipy.ndimage
            map_probability = scipy.ndimage.zoom(map_probability, np.array([self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z])/map_probability.shape, order=0)

            map_probability = map_probability / np.sum(map_probability)
            map_probability = 1 - map_probability

        except FileNotFoundError:
            map_probability = None

        return map_probability

    def load_roi_map(self):
        try:
            file_name = os.path.join(self.directory, "map_2")
            # file_name = os.path.join(self.directory, "map_heart")
            sizefile = self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z
            # sizefile = 101 * 101 * 153
            sizeMatrix = [154, 154, 373]
            sizefile = sizeMatrix[0] * sizeMatrix[1] * sizeMatrix[2]
            output_file = open(file_name, 'rb')
            a = array('f')
            a.fromfile(output_file, sizefile)
            output_file.close()
            self.roi_map = np.array(a)
            # self.roi_map = self.roi_map.reshape(
            #     (self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
            #     order='F')
            # self.roi_map = self.roi_map.reshape(
            #     (101, 101, 153),
            #     order='F')
            self.roi_map = self.roi_map.reshape(
                (sizeMatrix),
                order='F')
            import scipy.ndimage
            pixel_size_roi=np.array([0.2,0.2,0.2])
            pixel_size_image=np.array([0.8,0.8,0.8])
            pixel_relation = 1/((pixel_size_image/pixel_size_roi)*(np.array([self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z])/self.roi_map.shape))
            self.roi_map = scipy.ndimage.zoom(self.roi_map, pixel_relation*np.array([self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z])/self.roi_map.shape, order=0)
            mask = np.zeros(self.im_index_x.shape)
            sens_size = np.abs((np.array(self.roi_map.shape) - np.array(self.im_index_x.shape)) / 2).astype(
                    np.int16)
            mask[sens_size[0]:(mask.shape[0] - sens_size[0]),
            sens_size[1]:(mask.shape[1] - sens_size[1]),
            sens_size[2]:(mask.shape[2] - sens_size[2])]+=self.roi_map

            self.roi_map = mask
            # sens_size = ((np.array(self.roi_map.shape) - np.array(self.im_index_x.shape)) / 2).astype(
            #     np.int16)
            # if sens_size[0] >= 0:
            #     self.roi_map = self.roi_map[
            #                                 sens_size[0]:(
            #                                         self.roi_map.shape[0] - sens_size[0]),
            #                                 sens_size[1]:(
            #                                         self.roi_map.shape[1] - sens_size[1]),
            #                                 sens_size[2]:(
            #                                         self.roi_map.shape[2] - sens_size[2])]

            # self.roi_map = self.roi_map / self.roi_map
        except FileNotFoundError:
            self.roi_map = None

    def remove_events_outside_ROI(self):
        roievents = ROIEvents(self)
        valid_vor = roievents.pixel2Position()
        self.a = self.a[valid_vor == 1]
        self.b = self.b[valid_vor == 1]
        self.c = self.c[valid_vor == 1]
        self.d = self.d[valid_vor == 1]

        self.a_normal = self.a_normal[valid_vor == 1]
        self.b_normal = self.b_normal[valid_vor == 1]
        self.c_normal = self.c_normal[valid_vor == 1]
        self.d_normal = self.d_normal[valid_vor == 1]

        self.a_cf = self.a_cf[valid_vor == 1]
        self.b_cf = self.b_cf[valid_vor == 1]
        self.c_cf = self.c_cf[valid_vor == 1]
        self.d_cf = self.d_cf[valid_vor == 1]
        self.time_correction = self.time_correction[valid_vor == 1]
        self.sum_pixel = self.sum_pixel[valid_vor == 1]
        self.distance_to_center_plane_normal = self.distance_to_center_plane_normal[valid_vor == 1]
        self.distance_to_center_plane = self.distance_to_center_plane[valid_vor == 1]
        # self.distance_between_array_pixel = self.distance_between_array_pixel[valid_vor == 1]
        listmode = self.easypetdata
        np.save("_listmode.npy", listmode)
        listmode = listmode[valid_vor == 1]
        np.save("_listmode_cut.npy", listmode)

        self.a = np.ascontiguousarray(self.a, dtype=np.float32)
        self.b = np.ascontiguousarray(self.b, dtype=np.float32)
        self.c = np.ascontiguousarray(self.c, dtype=np.float32)
        self.d = np.ascontiguousarray(self.d, dtype=np.float32)

        self.a_normal = np.ascontiguousarray(self.a_normal, dtype=np.float32)
        self.b_normal = np.ascontiguousarray(self.b_normal, dtype=np.float32)
        self.c_normal = np.ascontiguousarray(self.c_normal, dtype=np.float32)
        self.d_normal = np.ascontiguousarray(self.d_normal, dtype=np.float32)

        self.a_cf = np.ascontiguousarray(self.a_cf, dtype=np.float32)
        self.b_cf = np.ascontiguousarray(self.b_cf, dtype=np.float32)
        self.c_cf = np.ascontiguousarray(self.c_cf, dtype=np.float32)
        self.d_cf = np.ascontiguousarray(self.d_cf, dtype=np.float32)

        self.time_correction = np.ascontiguousarray(self.time_correction, dtype=np.float32)
        self.sum_pixel = np.ascontiguousarray(self.sum_pixel, dtype=np.float32)
        self.distance_to_center_plane_normal = np.ascontiguousarray(self.distance_to_center_plane_normal,
                                                                    dtype=np.float32)
        self.distance_to_center_plane = np.ascontiguousarray(self.distance_to_center_plane, dtype=np.float32)

    def apply_circular_mask_fov(self):
        self.fov_matrix_cut = np.ascontiguousarray(
            np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
                    dtype=np.float32))
        xx = (np.tile(np.arange(0, self.im_index_x.shape[0]), (self.im_index_x.shape[0], 1)) - (
                self.im_index_x.shape[0] - 1) / 2) ** 2
        yy = (np.tile(np.arange(0, self.im_index_x.shape[1]), (self.im_index_x.shape[1], 1)) - (
                self.im_index_x.shape[1] - 1) / 2) ** 2
        xx = xx.T
        # correct xx and yy shape in case of missmatch
        if xx.shape[0] != yy.shape[0]:
            xx, yy = np.meshgrid(np.arange(0, self.im_index_x.shape[0])-((self.im_index_x.shape[0] - 1) / 2) ** 2,
                                 np.arange(0, self.im_index_x.shape[1])-((self.im_index_x.shape[1] - 1) / 2) ** 2)

            xx = xx.T
            yy=yy.T

        # circle_cut = xx + yy - (self.im_index_x.shape[1] * np.sin(np.radians(easypetdata.half_real_range))*0.5) ** 2
        circle_cut = xx + yy - (self.im_index_x.shape[1] * 0.5) ** 2
        # circle_cut = xx + yy - (self.im_index_x.shape[1] * np.sin(np.radians(120/2))*.50) ** 2

        circle_cut[circle_cut > 0] = 0
        circle_cut[circle_cut < 0] = 1
        circle_cut = np.tile(circle_cut[:, :, None], (1, 1, self.im_index_x.shape[2]))
        self.fov_matrix_cut = self.fov_matrix_cut * circle_cut
        self.fov_matrix_cut = np.ascontiguousarray(self.fov_matrix_cut, dtype=np.byte)
        # calculate non-zero pixels


    # def _vor_design_gpu_shared_memory_multiple_reads(self, a, a_normal, a_cf, A, b, b_normal, b_cf, B, c, c_normal,
    #                                                  c_cf, C,
    #                                                  d, d_normal, d_cf, adjust_coef, im, half_crystal_pitch_xy,
    #                                                  half_crystal_pitch_z,
    #                                                  sum_vor, fov_cut_matrix, normalization_matrix, time_factor):
    #     print('Optimizer STARTED - Multiple reads')
    #     # cuda.init()
    #     cuda = self.cuda_drv
    #     # device = cuda.Device(0)  # enter your gpu id here
    #     # ctx = device.make_context()
    #     number_of_events = np.int32(len(a))
    #     weight = np.int32(A.shape[0])
    #     height = np.int32(A.shape[1])
    #     depth = np.int32(A.shape[2])
    #     # start_x = np.int32(A[0, 0, 0])
    #     start_x = np.int32(A[0, 0, 0])
    #     start_y = np.int32(B[0, 0, 0])
    #     start_z = np.int32(C[0, 0, 0])
    #     print("Start_point: {},{},{}".format(start_x, start_y, start_z))
    #     print('Image size: {},{}, {}'.format(weight, height, depth))
    #
    #     half_distance_between_array_pixel = np.float32(self.distance_between_array_pixel / 2)
    #     normalization_matrix = normalization_matrix.reshape(
    #         normalization_matrix.shape[0] * normalization_matrix.shape[1] * normalization_matrix.shape[2])
    #
    #     # SOURCE MODELS (DEVICE CODE)
    #
    #     mod_forward_projection_shared_mem = SourceModule("""
    #     #include <stdint.h>
    #
    #     texture<char, 1> tex;
    #
    #
    #     __global__ void forward_projection
    #     (const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    #     const float crystal_pitch_XY,const float crystal_pitch_Z,const float distance_between_array_pixel,
    #     int number_of_events, const int begin_event_gpu_limitation,  const int end_event_gpu_limitation,
    #     float *a, float *a_normal, float *a_cf, float *b,float *b_normal, float *b_cf,float *c,float *c_normal,
    #     float *c_cf, float *d, float *d_normal, float *d_cf, float *sum_vor, const char *fov_cut_matrix, const float *im_old)
    #                            {
    #                               const int shared_memory_size = 256;
    #                                __shared__ float a_shared[shared_memory_size];
    #                                __shared__ float b_shared[shared_memory_size];
    #                                __shared__ float c_shared[shared_memory_size];
    #                                __shared__ float d_shared[shared_memory_size];
    #                                __shared__ float a_normal_shared[shared_memory_size];
    #                                __shared__ float b_normal_shared[shared_memory_size];
    #                                __shared__ float c_normal_shared[shared_memory_size];
    #                                __shared__ float d_normal_shared[shared_memory_size];
    #                                __shared__ float a_cf_shared[shared_memory_size];
    #                                __shared__ float b_cf_shared[shared_memory_size];
    #                                __shared__ float c_cf_shared[shared_memory_size];
    #                                __shared__ float d_cf_shared[shared_memory_size];
    #                                __shared__ float sum_vor_shared[shared_memory_size];
    #                                __shared__ char fov_cut_matrix_shared[shared_memory_size];
    #                                /*
    #                               const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
    #                               const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    #                               */
    #                               int threadId=blockIdx.x *blockDim.x + threadIdx.x;
    #                               int e;
    #
    #                               float d2;
    #                               float d2_normal;
    #                               float d2_cf;
    #                               float value;
    #                               float value_normal;
    #                               float value_cf;
    #                               float sum_vor_temp;
    #                               int index;
    #                               int x_t;
    #                               int y_t;
    #                               int z_t;
    #                               float max_distance_projector;
    #                               const int number_events_max = end_event_gpu_limitation-begin_event_gpu_limitation;
    #                               const float error_pixel = 0.0000f;
    #                                if (threadIdx.x>shared_memory_size)
    #                                {
    #                                return;
    #                                }
    #                                if(threadId>number_events_max)
    #                                {
    #                                return;
    #                                }
    #
    #                               __syncthreads();
    #                               e = threadId;
    #                               int e_m = threadIdx.x;
    #                               a_shared[e_m] = a[e];
    #                               b_shared[e_m] = b[e];
    #                               c_shared[e_m] = c[e];
    #                               d_shared[e_m] = d[e];
    #                               a_normal_shared[e_m] = a_normal[e];
    #                               b_normal_shared[e_m] = b_normal[e];
    #                               c_normal_shared[e_m] = c_normal[e];
    #                               d_normal_shared[e_m] = d_normal[e];
    #                               a_cf_shared[e_m] = a_cf[e];
    #                               b_cf_shared[e_m] = b_cf[e];
    #                               c_cf_shared[e_m] = c_cf[e];
    #                               d_cf_shared[e_m] = d_cf[e];
    #                               sum_vor_shared[e_m] = sum_vor[e];
    #
    #                               d2_normal = crystal_pitch_XY * sqrt(a_normal_shared[e_m]*a_normal_shared[e_m]+b_normal_shared[e_m]*b_normal_shared[e_m]+c_normal_shared[e_m]*c_normal_shared[e_m]);
    #                               d2 = crystal_pitch_Z * sqrt(a_shared[e_m]*a_shared[e_m] + b_shared[e_m]*b_shared[e_m] + c_shared[e_m]*c_shared[e_m]);
    #                               d2_cf = distance_between_array_pixel*sqrt(a_cf_shared[e_m]*a_cf_shared[e_m]+b_cf_shared[e_m]*b_cf_shared[e_m]+c_cf_shared[e_m]*c_cf_shared[e_m]);
    #                               max_distance_projector=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf);
    #
    #                               for(int l=0; l<n; l++)
    #                               {
    #                                   x_t = l+start_x;
    #                                   for(int j=0; j<m; j++)
    #                                   {
    #                                      y_t = j+start_y;
    #                                      for(int k=0; k<p; k++)
    #                                           {
    #                                           /*
    #                                           index = l+j*n+k*m*n;
    #                                           fov_cut_matrix_shared[k]=fov_cut_matrix[k+j*p+l*p*m];
    #                                           index = l+j*n+k*m*n;
    #                                           */
    #                                             index = k+j*p+l*p*m;
    #
    #
    #                                           if (fov_cut_matrix[index]!=0)
    #                                           {
    #
    #                                               z_t = k+start_z;
    #                                               value_normal = a_normal_shared[e_m]*x_t+b_normal_shared[e_m]*y_t+c_normal_shared[e_m] * z_t -d_normal_shared[e_m];
    #
    #                                                if (value_normal < d2_normal && value_normal >= -d2_normal)
    #                                                {
    #                                                 value = a_shared[e_m]*x_t+b_shared[e_m]*y_t+c_shared[e_m]*z_t-d_shared[e_m];
    #
    #                                                 if (value < d2 && value >=-d2 )
    #                                                  {
    #                                                       value_cf = a_cf_shared[e_m]*x_t+b_cf_shared[e_m]*y_t+c_cf_shared[e_m]*z_t-d_cf_shared[e_m];
    #
    #
    #                                                   if (value_cf >= -d2_cf && value_cf<d2_cf)
    #                                                       {
    #
    #                                                        sum_vor_temp +=im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector);
    #                                                        /* im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector)
    #
    #                                                         */
    #                                                         }
    #
    #
    #                                               }
    #
    #
    #
    #
    #                                           }
    #                                           }
    #
    #                                        }
    #                                    }
    #                                }
    #
    #                                /*
    #                                 sum_vor[e]= sum_vor_temp;
    #                                sum_vor[e]= sum_vor_shared[e_m];
    #                                __syncthreads();
    #                                sum_vor[e]= sum_vor_shared[e_m];
    #                                */
    #                                sum_vor[e]= sum_vor_temp;
    #
    #
    #
    #            }""")
    #     mod_normalization_shared_mem = SourceModule("""
    #                          #include <stdint.h>
    #                          texture<uint8_t, 1> tex;
    #
    #                   __global__ void normalization
    #                    ( int dataset_number, int n, int m, int p, const float crystal_pitch_XY, const float crystal_pitch_Z,
    #                    const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    #                    const int end_event_gpu_limitation, const float *a, const float *a_normal, const float *a_cf, const float *b,
    #                    const float *b_normal, const float *b_cf, const float *c, const float *c_normal, const float *c_cf, const float *d, const float *d_normal,
    #                    const float *d_cf, const short *A, const short *B, const short *C, float *adjust_coef, float *sum_vor,
    #                    char *fov_cut_matrix, float *time_factor)
    #                    {
    #                              extern __shared__ float adjust_coef_shared[];
    #                              int idt=blockIdx.x *blockDim.x + threadIdx.x;
    #
    #
    #                              float d2;
    #                              float d2_normal;
    #                              float d2_cf;
    #                              float normal_value;
    #                              float value;
    #                              float value_cf;
    #                              short a_temp;
    #                              short b_temp;
    #                              short c_temp;
    #                              char fov_cut_temp;
    #                              int i_s=threadIdx.x;
    #
    #                              if (idt>n*m*p)
    #                              {
    #                                   return;
    #                               }
    #
    #                              __syncthreads();
    #                              adjust_coef_shared[i_s] = adjust_coef[idt];
    #                              a_temp = A[idt];
    #                              b_temp = B[idt];
    #                              c_temp = C[idt];
    #                              fov_cut_temp = fov_cut_matrix[idt];
    #
    #
    #
    #                              for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
    #                              {
    #                                  if (fov_cut_temp!=0)
    #                                  {
    #                                  normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                  d2_normal = crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #
    #                                  if (normal_value< d2_normal && normal_value >= -d2_normal)
    #                                  {
    #                                  value = a[e]*a_temp+b[e]*b_temp +c[e]*c_temp- d[e];
    #                                  d2 = crystal_pitch_Z * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);
    #
    #
    #                                      if (value < d2 && value >=-d2)
    #                                      {
    #                                                value_cf = a_cf[e]*a_temp+b_cf[e]*b_temp +c_cf[e] * c_temp-d_cf[e];
    #                                                d2_cf = distance_between_array_pixel*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);
    #
    #
    #                                                  if (value_cf >= -d2_cf && value_cf<d2_cf)
    #                                                  {
    #
    #
    #                                                         adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf));
    #
    #
    #                                                 }
    #
    #                                                      /*
    #
    #                                                      adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf));
    #                                                      adjust_coef_shared[i_s] += 1/sum_vor[e];
    #                                                      normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                       d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #                                                     adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
    #                                                    adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
    #                                                       */
    #
    #
    #
    #
    #
    #                                           }
    #                                       }
    #
    #                              }
    #
    #                              }
    #
    #                              adjust_coef[idt] = adjust_coef_shared[i_s];
    #                              __syncthreads();
    #
    #
    #
    #                          }
    #                          """)
    #
    #     mod_backward_projection_shared_mem = SourceModule("""
    #                   #include <stdint.h>
    #                   texture<uint8_t, 1> tex;
    #
    #            __global__ void backprojection
    #             ( int dataset_number, int n, int m, int p, const float crystal_pitch_XY, const float crystal_pitch_Z,
    #             const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    #             const int end_event_gpu_limitation, const float *a, const float *a_normal, const float *a_cf, const float *b,
    #             const float *b_normal, const float *b_cf, const float *c, const float *c_normal, const float *c_cf, const float *d, const float *d_normal,
    #             const float *d_cf, const short *A, const short *B, const short *C, float *adjust_coef, float *sum_vor,
    #             char *fov_cut_matrix, float *time_factor)
    #             {
    #                       extern __shared__ float adjust_coef_shared[];
    #                       int idt=blockIdx.x *blockDim.x + threadIdx.x;
    #
    #
    #                       float d2;
    #                       float d2_normal;
    #                       float d2_cf;
    #                       float normal_value;
    #                       float value;
    #                       float value_cf;
    #                       short a_temp;
    #                       short b_temp;
    #                       short c_temp;
    #                       char fov_cut_temp;
    #                       int i_s=threadIdx.x;
    #
    #                       if (idt>n*m*p)
    #                       {
    #                            return;
    #                        }
    #
    #                       __syncthreads();
    #                       adjust_coef_shared[i_s] = adjust_coef[idt];
    #                       a_temp = A[idt];
    #                       b_temp = B[idt];
    #                       c_temp = C[idt];
    #                       fov_cut_temp = fov_cut_matrix[idt];
    #
    #
    #
    #                       for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
    #                       {
    #                           if (fov_cut_temp!=0)
    #                           {
    #                           normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                           d2_normal = crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #
    #                           if (normal_value< d2_normal && normal_value >= -d2_normal)
    #                           {
    #                           value = a[e]*a_temp+b[e]*b_temp +c[e]*c_temp- d[e];
    #                           d2 = crystal_pitch_Z * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);
    #
    #
    #                               if (value < d2 && value >=-d2)
    #                               {
    #                                         value_cf = a_cf[e]*a_temp+b_cf[e]*b_temp +c_cf[e] * c_temp-d_cf[e];
    #                                         d2_cf = distance_between_array_pixel*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);
    #
    #
    #                                           if (value_cf >= -d2_cf && value_cf<d2_cf)
    #                                           {
    #
    #
    #
    #                                                 adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]*time_factor[e]);
    #
    #                                          }
    #
    #                                               /*
    #
    #                                               adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]*time_factor[e]);
    #                                               adjust_coef_shared[i_s] += 1/sum_vor[e];
    #                                               normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #                                              adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
    #                                             adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
    #                                                */
    #
    #
    #
    #
    #
    #                                    }
    #                                }
    #
    #                       }
    #
    #                       }
    #
    #                       adjust_coef[idt] = adjust_coef_shared[i_s];
    #                       __syncthreads();
    #
    #
    #
    #                   }
    #                   """)
    #
    #     mod_forward_projection_shared_mem_cdrf = SourceModule("""
    #            #include <stdint.h>
    #
    #            texture<char, 1> tex;
    #
    #
    #            __global__ void forward_projection_cdrf
    #            (const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    #            float *crystal_pitch_XY, float *crystal_pitch_Z,const float distance_between_array_pixel,
    #            int number_of_events, const int begin_event_gpu_limitation,  const int end_event_gpu_limitation,
    #            float *a, float *a_normal, float *a_cf, float *b,float *b_normal, float *b_cf,float *c,float *c_normal,
    #            float *c_cf, float *d, float *d_normal, float *d_cf, float *sum_vor, const char *fov_cut_matrix, const float *im_old)
    #                                   {
    #                                      const int shared_memory_size = 256;
    #                                       __shared__ float a_shared[shared_memory_size];
    #                                       __shared__ float b_shared[shared_memory_size];
    #                                       __shared__ float c_shared[shared_memory_size];
    #                                       __shared__ float d_shared[shared_memory_size];
    #                                       __shared__ float a_normal_shared[shared_memory_size];
    #                                       __shared__ float b_normal_shared[shared_memory_size];
    #                                       __shared__ float c_normal_shared[shared_memory_size];
    #                                       __shared__ float d_normal_shared[shared_memory_size];
    #                                       __shared__ float a_cf_shared[shared_memory_size];
    #                                       __shared__ float b_cf_shared[shared_memory_size];
    #                                       __shared__ float c_cf_shared[shared_memory_size];
    #                                       __shared__ float d_cf_shared[shared_memory_size];
    #                                       __shared__ float sum_vor_shared[shared_memory_size];
    #                                       __shared__ char fov_cut_matrix_shared[shared_memory_size];
    #                                       __shared__ float crystal_pitch_Z_shared[shared_memory_size];
    #                                       __shared__ float crystal_pitch_XY_shared[shared_memory_size];
    #                                       /*
    #                                      const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
    #                                      const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
    #                                      */
    #                                      int threadId=blockIdx.x *blockDim.x + threadIdx.x;
    #                                      int e;
    #
    #                                      float d2;
    #                                      float d2_normal;
    #                                      float d2_cf;
    #                                      float value;
    #                                      float value_normal;
    #                                      float value_cf;
    #                                      float sum_vor_temp;
    #                                      int index;
    #                                      int x_t;
    #                                      int y_t;
    #                                      int z_t;
    #                                           float width;
    #                                       float height;
    #                                     float distance;
    #                                     float distance_other;
    #                                     float solid_angle;
    #
    #
    #                                      float max_distance_projector;
    #
    #                                      const int number_events_max = end_event_gpu_limitation-begin_event_gpu_limitation;
    #                                      const float error_pixel = 0.0000f;
    #                                       if (threadIdx.x>shared_memory_size)
    #                                       {
    #                                       return;
    #                                       }
    #                                       if(threadId>number_events_max)
    #                                       {
    #                                       return;
    #                                       }
    #
    #                                      __syncthreads();
    #                                      e = threadId;
    #                                      int e_m = threadIdx.x;
    #                                      a_shared[e_m] = a[e];
    #                                      b_shared[e_m] = b[e];
    #                                      c_shared[e_m] = c[e];
    #                                      d_shared[e_m] = d[e];
    #                                      a_normal_shared[e_m] = a_normal[e];
    #                                      b_normal_shared[e_m] = b_normal[e];
    #                                      c_normal_shared[e_m] = c_normal[e];
    #                                      d_normal_shared[e_m] = d_normal[e];
    #                                      a_cf_shared[e_m] = a_cf[e];
    #                                      b_cf_shared[e_m] = b_cf[e];
    #                                      c_cf_shared[e_m] = c_cf[e];
    #                                      d_cf_shared[e_m] = d_cf[e];
    #                                      sum_vor_shared[e_m] = sum_vor[e];
    #                                      crystal_pitch_Z_shared[e_m] =  crystal_pitch_Z[e];
    #                                      crystal_pitch_XY_shared[e_m] =  crystal_pitch_XY[e];
    #
    #                                      d2_normal = crystal_pitch_XY_shared[e_m] * sqrt(a_normal_shared[e_m]*a_normal_shared[e_m]+b_normal_shared[e_m]*b_normal_shared[e_m]+c_normal_shared[e_m]*c_normal_shared[e_m]);
    #                                      d2 = crystal_pitch_Z_shared[e_m]* sqrt(a_shared[e_m]*a_shared[e_m] + b_shared[e_m]*b_shared[e_m] + c_shared[e_m]*c_shared[e_m]);
    #                                      d2_cf = distance_between_array_pixel*sqrt(a_cf_shared[e_m]*a_cf_shared[e_m]+b_cf_shared[e_m]*b_cf_shared[e_m]+c_cf_shared[e_m]*c_cf_shared[e_m]);
    #                                      max_distance_projector=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf);
    #
    #                                      for(int l=0; l<n; l++)
    #                                      {
    #                                          x_t = l+start_x;
    #                                          for(int j=0; j<m; j++)
    #                                          {
    #                                             y_t = j+start_y;
    #                                             for(int k=0; k<p; k++)
    #                                                  {
    #                                                  /*
    #                                                  index = l+j*n+k*m*n;
    #                                                  fov_cut_matrix_shared[k]=fov_cut_matrix[k+j*p+l*p*m];
    #                                                  index = l+j*n+k*m*n;
    #                                                  */
    #                                                    index = k+j*p+l*p*m;
    #
    #
    #                                                  if (fov_cut_matrix[index]!=0)
    #                                                  {
    #
    #                                                      z_t = k+start_z;
    #                                                      value_normal = a_normal_shared[e_m]*x_t+b_normal_shared[e_m]*y_t+c_normal_shared[e_m] * z_t -d_normal_shared[e_m];
    #
    #                                                       if (value_normal < d2_normal && value_normal >= -d2_normal)
    #                                                       {
    #                                                        value = a_shared[e_m]*x_t+b_shared[e_m]*y_t+c_shared[e_m]*z_t-d_shared[e_m];
    #
    #                                                        if (value < d2 && value >=-d2 )
    #                                                         {
    #                                                              value_cf = a_cf_shared[e_m]*x_t+b_cf_shared[e_m]*y_t+c_cf_shared[e_m]*z_t-d_cf_shared[e_m];
    #
    #
    #                                                          if (value_cf >= -d2_cf && value_cf<d2_cf)
    #                                                              {
    #                                                              if (sqrt(value*value+value_normal*value_normal+value_cf*value_cf)<=max_distance_projector)
    #                                                              {
    #                                                                  width = 2*(crystal_pitch_Z_shared[e_m]  - abs(value));
    #                                                                 height = 2*(crystal_pitch_XY_shared[e_m] - abs(value_normal));
    #
    #                                                                  distance =d2_cf+abs(value_cf);
    #                                                                 distance_other = abs(d2_cf+value_cf);
    #                                                                 solid_angle = 4*(width*width*height*height/(distance*distance*(4*distance*distance+width*width+height*height)));
    #                                                               sum_vor_shared[e_m]+= im_old[index]*solid_angle;
    #
    #                                                               }
    #                                                               /*
    #                                                               sum_vor_shared[e_m]+= im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector);
    #                                                               4*arctan(
    #                                                               4*np.arctan(width*height/(2*distance_to_anh*np.sqrt(4*distance_between_array_pixel-value_cf)**2+width**2+height**2)))
    #                                                               *(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector)
    #                                                               */
    #                                                                }
    #
    #
    #                                                      }
    #
    #
    #
    #
    #                                                  }
    #                                                  }
    #
    #                                               }
    #                                           }
    #                                       }
    #
    #                                       /*
    #                                        sum_vor[e]= sum_vor_temp;
    #                                       sum_vor[e]= sum_vor_shared[e_m];
    #
    #
    #                                       sum_vor[e]= sum_vor_temp;
    #                                       */
    #                                       __syncthreads();
    #                                       sum_vor[e]= sum_vor_shared[e_m];
    #
    #
    #
    #                   }""")
    #
    #     mod_normalization_shared_mem_cdrf = SourceModule("""
    #                                  #include <stdint.h>
    #                                  texture<uint8_t, 1> tex;
    #
    #                           __global__ void normalization_cdrf
    #                            ( int dataset_number, int n, int m, int p, float *crystal_pitch_XY,  float *crystal_pitch_Z,
    #                            const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    #                            const int end_event_gpu_limitation, const float *a, const float *a_normal, const float *a_cf, const float *b,
    #                            const float *b_normal, const float *b_cf, const float *c, const float *c_normal, const float *c_cf, const float *d, const float *d_normal,
    #                            const float *d_cf, const short *A, const short *B, const short *C, float *adjust_coef, float *sum_vor,
    #                            char *fov_cut_matrix, float *time_factor)
    #                            {
    #                                      extern __shared__ float adjust_coef_shared[];
    #                                      int idt=blockIdx.x *blockDim.x + threadIdx.x;
    #
    #
    #                                      float d2;
    #                                      float d2_normal;
    #                                      float d2_cf;
    #                                      float normal_value;
    #                                      float value;
    #                                      float value_cf;
    #                                      short a_temp;
    #                                      short b_temp;
    #                                      short c_temp;
    #                                      char fov_cut_temp;
    #                                      int i_s=threadIdx.x;
    #                                      float width;
    #                                     float height;
    #                                     float distance;
    #                                     float distance_other;
    #                                      float solid_angle;
    #
    #                                      if (idt>=n*m*p)
    #                                      {
    #                                           return;
    #                                       }
    #
    #                                      __syncthreads();
    #                                      adjust_coef_shared[i_s] = adjust_coef[idt];
    #                                      a_temp = A[idt];
    #                                      b_temp = B[idt];
    #                                      c_temp = C[idt];
    #                                      fov_cut_temp = fov_cut_matrix[idt];
    #
    #
    #
    #                                      for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
    #                                      {
    #                                          if (fov_cut_temp!=0)
    #                                          {
    #                                          normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                          d2_normal =  crystal_pitch_XY[e]* sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #
    #                                          if (normal_value< d2_normal && normal_value >= -d2_normal)
    #                                          {
    #                                          value = a[e]*a_temp+b[e]*b_temp +c[e]*c_temp- d[e];
    #                                          d2 = crystal_pitch_Z[e] * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);
    #
    #
    #                                              if (value < d2 && value >=-d2)
    #                                              {
    #                                                        value_cf = a_cf[e]*a_temp+b_cf[e]*b_temp +c_cf[e] * c_temp-d_cf[e];
    #                                                        d2_cf = distance_between_array_pixel*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);
    #
    #
    #                                                          if (value_cf >= -d2_cf && value_cf<d2_cf)
    #                                                          {
    #                                                             if (sqrt(value*value+normal_value*normal_value+value_cf*value_cf)<=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))
    #                                                             {
    #                                                              width = 2*(crystal_pitch_Z[e]- abs(value));
    #                                                             height = 2*(crystal_pitch_XY[e] - abs(normal_value));
    #                                                             distance =d2_cf+abs(value_cf);
    #
    #                                                             solid_angle = 4*(width*width*height*height/(distance*distance*(4*distance*distance+width*width+height*height)));
    #                                                              adjust_coef_shared[i_s] += solid_angle/(sum_vor[e]);
    #                                                             /*
    #                                                             distance_other =d2_cf-abs(value_cf);
    #                                                             solid_angle = 4*(width*height/(2*distance*sqrt(4*distance*distance+width*width+height*height)));
    #                                                                adjust_coef_shared[i_s] += time_factor[e]*(1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]);
    #                                                               */
    #                                                                }
    #
    #                                                         }
    #
    #                                                              /*
    #                                                              (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/
    #                                                              normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                               d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #                                                             adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
    #                                                            adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
    #                                                               */
    #
    #
    #
    #
    #
    #                                                   }
    #                                               }
    #
    #                                      }
    #
    #                                      }
    #
    #                                      adjust_coef[idt] = adjust_coef_shared[i_s];
    #                                      __syncthreads();
    #
    #
    #
    #                                  }
    #                                  """)
    #
    #     mod_backward_projection_shared_mem_cdrf = SourceModule("""
    #                                 #include <stdint.h>
    #                                 texture<uint8_t, 1> tex;
    #
    #                          __global__ void backprojection_cdrf
    #                           ( int dataset_number, int n, int m, int p, const float *crystal_pitch_XY,  const float *crystal_pitch_Z,
    #                           const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
    #                           const int end_event_gpu_limitation, const float *a, const float *a_normal, const float *a_cf, const float *b,
    #                           const float *b_normal, const float *b_cf, const float *c, const float *c_normal, const float *c_cf, const float *d, const float *d_normal,
    #                           const float *d_cf, const short *A, const short *B, const short *C, float *adjust_coef, float *sum_vor,
    #                           char *fov_cut_matrix, float *time_factor)
    #                           {
    #                                     extern __shared__ float adjust_coef_shared[];
    #                                     int idt=blockIdx.x *blockDim.x + threadIdx.x;
    #
    #
    #                                     float d2;
    #                                     float d2_normal;
    #                                     float d2_cf;
    #                                     float normal_value;
    #                                     float value;
    #                                     float value_cf;
    #                                     short a_temp;
    #                                     short b_temp;
    #                                     short c_temp;
    #                                     char fov_cut_temp;
    #                                     int i_s=threadIdx.x;
    #                                     float width;
    #                                     float height;
    #                                     float distance;
    #                                     float distance_other;
    #                                     float solid_angle;
    #
    #                                     if (idt>=n*m*p)
    #                                     {
    #                                          return;
    #                                      }
    #
    #                                     __syncthreads();
    #                                     adjust_coef_shared[i_s] = adjust_coef[idt];
    #                                     a_temp = A[idt];
    #                                     b_temp = B[idt];
    #                                     c_temp = C[idt];
    #
    #                                     fov_cut_temp = fov_cut_matrix[idt];
    #
    #
    #
    #                                     for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
    #                                     {
    #                                         if (fov_cut_temp!=0)
    #                                         {
    #                                         normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                         d2_normal =  crystal_pitch_XY[e]* sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #
    #                                         if (normal_value< d2_normal && normal_value >= -d2_normal)
    #                                         {
    #                                         value = a[e]*a_temp+b[e]*b_temp +c[e]*c_temp- d[e];
    #                                         d2 = crystal_pitch_Z[e] * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);
    #
    #
    #                                             if (value < d2 && value >=-d2)
    #                                             {
    #                                                       value_cf = a_cf[e]*a_temp+b_cf[e]*b_temp +c_cf[e] * c_temp-d_cf[e];
    #                                                       d2_cf = distance_between_array_pixel*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);
    #
    #
    #                                                         if (value_cf >= -d2_cf && value_cf<d2_cf)
    #                                                         {
    #                                                            if (sqrt(value*value+normal_value*normal_value+value_cf*value_cf)<=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))
    #                                                            {
    #                                                            width = 2*(crystal_pitch_Z[e]- abs(value));
    #                                                            height = 2*(crystal_pitch_XY[e] - abs(normal_value));
    #                                                            distance =d2_cf+abs(value_cf);
    #                                                            distance_other =d2_cf-abs(value_cf);
    #                                                            solid_angle = 4*(width*width*height*height/(distance*distance*(4*distance*distance+width*width+height*height)));
    #                                                            if (sum_vor[e]!=0)
    #
    #                                                             {
    #                                                             adjust_coef_shared[i_s] += solid_angle/(sum_vor[e]);
    #                                                             }
    #                                                              /*
    #                                                              (4 * asin(sin(tan(width/distance))*sin(tan(height/distance))))*(4 * asin(sin(tan(width/distance))*sin(tan(height/distance))))/sum_vor[e];
    #                                                          adjust_coef_shared[i_s] += time_factor[e]*(1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]);
    #
    #                                                              */
    #                                                               }
    #
    #                                                        }
    #
    #                                                             /*
    #                                                             (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/
    #                                                             normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
    #                                              d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
    #                                                            adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
    #                                                           adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
    #                                                              */
    #
    #
    #
    #
    #
    #                                                  }
    #                                              }
    #
    #                                     }
    #
    #                                     }
    #
    #                                     adjust_coef[idt] = adjust_coef_shared[i_s];
    #                                     __syncthreads();
    #
    #
    #
    #                                 }
    #                                 """)
    #
    #     # float crystal_pitch, int number_of_events, float *a, float *b, float *c, float *d, int *A, int *B, int *C, float *im, float *vector_matrix
    #     # Host Code   B, C, im, vector_matrix,
    #     number_of_datasets = np.int32(1)  # Number of datasets (and concurrent operations) used.
    #     number_of_datasets_back = np.int32(1)  # Number of datasets (and concurrent operations) used.
    #     # Start concurrency Test
    #     # Event as reference point
    #     ref = cuda.Event()
    #     ref.record()
    #
    #     # Create the streams and events needed to calculation
    #     stream, event = [], []
    #     marker_names = ['kernel_begin', 'kernel_end']
    #     # Create List to allocate chunks of data
    #     A_cut_gpu, B_cut_gpu, C_cut_gpu = [None] * number_of_datasets, [None] * number_of_datasets, [
    #         None] * number_of_datasets
    #     A_cut, B_cut, C_cut = [None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets
    #
    #     a_gpu, b_gpu, c_gpu, d_gpu = [None] * number_of_datasets, [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets
    #     a_normal_gpu, b_normal_gpu, c_normal_gpu, d_normal_gpu = [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets
    #
    #     a_cut, b_cut, c_cut, d_cut = [None] * number_of_datasets, [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets
    #     a_normal_cut, b_normal_cut, c_normal_cut, d_normal_cut = [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets
    #
    #     a_cf_cut, b_cf_cut, c_cf_cut, d_cf_cut = [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets
    #
    #     a_cut_gpu, b_cut_gpu, c_cut_gpu, d_cut_gpu = [None] * number_of_datasets, [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets
    #
    #     a_cut_normal_gpu, b_cut_normal_gpu, c_cut_normal_gpu, d_cut_normal_gpu = [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets
    #
    #     a_cf_cut_gpu, b_cf_cut_gpu, c_cf_cut_gpu, d_cf_cut_gpu = [None] * number_of_datasets, [
    #         None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets
    #
    #     time_factor_cut_gpu = [None] * number_of_datasets
    #     sum_vor_gpu = [None] * number_of_datasets
    #     sum_vor_cut = [None] * number_of_datasets
    #     probability_cut = [None] * number_of_datasets
    #     probability_gpu = [None] * number_of_datasets
    #     adjust_coef_cut = [None] * number_of_datasets
    #     adjust_coef_gpu = [None] * number_of_datasets
    #     adjust_coef_pinned = [None] * number_of_datasets_back
    #     fov_cut_matrix_cutted_gpu = [None] * number_of_datasets_back
    #     fov_cut_matrix_cut = [None] * number_of_datasets
    #     # fov_cut_matrix_gpu = [None] * number_of_datasets
    #     sum_vor_pinned = [None] * number_of_datasets
    #
    #     distance_to_center_plane_cut = [None] * number_of_datasets
    #     distance_to_center_plane_gpu_cut = [None] * number_of_datasets
    #     distance_to_center_plane_normal_cut = [None] * number_of_datasets
    #     distance_to_center_plane_normal_gpu_cut = [None] * number_of_datasets
    #
    #     # Streams and Events creation
    #     for dataset in range(number_of_datasets):
    #         stream.append(cuda.Stream())
    #         event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))
    #
    #     # Foward Projection Memory Allocation
    #     # Variables that need an unique alocation
    #     # A_shappened = np.ascontiguousarray(A.reshape(A.shape[0]*A.shape[1]*A.shape[2]), dtype=np.int32)
    #     # B_shappened = np.ascontiguousarray(B.reshape(B.shape[0]*B.shape[1]*B.shape[2]), dtype=np.int32)
    #     # C_shappened = np.ascontiguousarray(C.reshape(C.shape[0]*C.shape[1]*C.shape[2]), dtype=np.int32)
    #     im_shappened = np.ascontiguousarray(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]), dtype=np.float32)
    #     fov_cut_matrix_shappened = np.ascontiguousarray(
    #         fov_cut_matrix.reshape(fov_cut_matrix.shape[0] * fov_cut_matrix.shape[1] * fov_cut_matrix.shape[2]),
    #         dtype=np.byte)
    #
    #     # forward_projection_arrays_page_locked_memory_allocations = [A_gpu, B_gpu, C_gpu, im_gpu]
    #
    #     # A_gpu = cuda.mem_alloc(A_shappened.size * A_shappened.dtype.itemsize)
    #     # B_gpu = cuda.mem_alloc(B_shappened.size * B_shappened.dtype.itemsize)
    #     # C_gpu = cuda.mem_alloc(C_shappened.size * C_shappened.dtype.itemsize)
    #     im_gpu = cuda.mem_alloc(im_shappened.size * im_shappened.dtype.itemsize)
    #     fov_cut_matrix_gpu = cuda.mem_alloc(fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
    #     # texref = mod_forward_projection_shared_mem.get_texref('tex')
    #     # texref.set_address(fov_cut_matrix_gpu, fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
    #     # # texref.set_format(cuda.array_format.UNSIGNED_INT8, 1)
    #
    #     # cuda.memcpy_htod_async(A_gpu, A_shappened)
    #     # cuda.memcpy_htod_async(B_gpu, B_shappened)
    #     # cuda.memcpy_htod_async(C_gpu, C_shappened)
    #     cuda.memcpy_htod_async(im_gpu, im_shappened)
    #     cuda.memcpy_htod_async(fov_cut_matrix_gpu, fov_cut_matrix_shappened)
    #
    #     a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
    #     b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
    #     c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
    #     d_gpu = cuda.mem_alloc(d.size * d.dtype.itemsize)
    #     a_normal_gpu = cuda.mem_alloc(a_normal.size * a_normal.dtype.itemsize)
    #     b_normal_gpu = cuda.mem_alloc(b_normal.size * b_normal.dtype.itemsize)
    #     c_normal_gpu = cuda.mem_alloc(c_normal.size * c_normal.dtype.itemsize)
    #     d_normal_gpu = cuda.mem_alloc(d_normal.size * d_normal.dtype.itemsize)
    #
    #     a_cf_gpu = cuda.mem_alloc(a_cf.size * a_cf.dtype.itemsize)
    #     b_cf_gpu = cuda.mem_alloc(b_cf.size * b_cf.dtype.itemsize)
    #     c_cf_gpu = cuda.mem_alloc(c_cf.size * c_cf.dtype.itemsize)
    #     d_cf_gpu = cuda.mem_alloc(d_cf.size * d_cf.dtype.itemsize)
    #     sum_vor_t_gpu = cuda.mem_alloc(sum_vor.size * sum_vor.dtype.itemsize)
    #     distance_to_center_plane_gpu = cuda.mem_alloc(
    #         self.distance_to_center_plane.size * self.distance_to_center_plane.dtype.itemsize)
    #     distance_to_center_plane_normal_gpu = cuda.mem_alloc(
    #         self.distance_to_center_plane_normal.size * self.distance_to_center_plane_normal.dtype.itemsize)
    #
    #     time_factor_gpu = cuda.mem_alloc(time_factor.size * time_factor.dtype.itemsize)
    #     # Transfer memory to Optimizer
    #     cuda.memcpy_htod_async(a_gpu, a)
    #     cuda.memcpy_htod_async(b_gpu, b)
    #     cuda.memcpy_htod_async(c_gpu, c)
    #     cuda.memcpy_htod_async(d_gpu, d)
    #     cuda.memcpy_htod_async(a_normal_gpu, a_normal)
    #     cuda.memcpy_htod_async(b_normal_gpu, b_normal)
    #     cuda.memcpy_htod_async(c_normal_gpu, c_normal)
    #     cuda.memcpy_htod_async(d_normal_gpu, d_normal)
    #
    #     cuda.memcpy_htod_async(a_cf_gpu, a_cf)
    #     cuda.memcpy_htod_async(b_cf_gpu, b_cf)
    #     cuda.memcpy_htod_async(c_cf_gpu, c_cf)
    #     cuda.memcpy_htod_async(d_cf_gpu, d_cf)
    #     cuda.memcpy_htod_async(time_factor_gpu, time_factor)
    #     cuda.memcpy_htod_async(distance_to_center_plane_gpu, self.distance_to_center_plane)
    #     cuda.memcpy_htod_async(distance_to_center_plane_normal_gpu, self.distance_to_center_plane_normal)
    #
    #     for dataset in range(number_of_datasets):
    #         # if dataset == number_of_datasets:
    #         #     begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
    #         #     end_dataset = number_of_events
    #         #     adjust_coef_cut[dataset] = np.ascontiguousarray(
    #         #         adjust_coef[int(np.floor(im_cut_dim[0] * dataset)):adjust_coef.shape[0],
    #         #         int(np.floor(im_cut_dim[1] * dataset)):adjust_coef.shape[1],
    #         #         int(np.floor(im_cut_dim[2] * dataset)):adjust_coef.shape[2]],
    #         #         dtype=np.float32)
    #         #     fov_cut_matrix_cut[dataset] = np.ascontiguousarray(
    #         #         fov_cut_matrix[int(np.floor(im_cut_dim[0] * dataset)):fov_cut_matrix.shape[0],
    #         #         int(np.floor(im_cut_dim[1] * dataset)):fov_cut_matrix.shape[1],
    #         #         int(np.floor(im_cut_dim[2] * dataset)):fov_cut_matrix.shape[2]],
    #         #         dtype=np.float32)
    #         # else:
    #         begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
    #         end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
    #
    #         # Cutting dataset
    #         # For forward projection the data is cutted by number of events. For backprojection is cutted per image pieces of image
    #         a_cut[dataset] = a[begin_dataset:end_dataset]
    #         b_cut[dataset] = b[begin_dataset:end_dataset]
    #         c_cut[dataset] = c[begin_dataset:end_dataset]
    #         d_cut[dataset] = d[begin_dataset:end_dataset]
    #         a_normal_cut[dataset] = a_normal[begin_dataset:end_dataset]
    #         b_normal_cut[dataset] = b_normal[begin_dataset:end_dataset]
    #         c_normal_cut[dataset] = c_normal[begin_dataset:end_dataset]
    #         d_normal_cut[dataset] = d_normal[begin_dataset:end_dataset]
    #
    #         a_cf_cut[dataset] = a_cf[begin_dataset:end_dataset]
    #         b_cf_cut[dataset] = b_cf[begin_dataset:end_dataset]
    #         c_cf_cut[dataset] = c_cf[begin_dataset:end_dataset]
    #         d_cf_cut[dataset] = d_cf[begin_dataset:end_dataset]
    #         sum_vor_cut[dataset] = sum_vor[begin_dataset:end_dataset]
    #         distance_to_center_plane_cut[dataset] = self.distance_to_center_plane[begin_dataset:end_dataset]
    #         distance_to_center_plane_normal_cut[dataset] = self.distance_to_center_plane_normal[
    #                                                        begin_dataset:end_dataset]
    #
    #         # Forward
    #         a_cut_gpu[dataset] = cuda.mem_alloc(a_cut[dataset].size * a_cut[dataset].dtype.itemsize)
    #         b_cut_gpu[dataset] = cuda.mem_alloc(b_cut[dataset].size * b_cut[dataset].dtype.itemsize)
    #         c_cut_gpu[dataset] = cuda.mem_alloc(c_cut[dataset].size * c_cut[dataset].dtype.itemsize)
    #         d_cut_gpu[dataset] = cuda.mem_alloc(d_cut[dataset].size * d_cut[dataset].dtype.itemsize)
    #         a_cut_normal_gpu[dataset] = cuda.mem_alloc(
    #             a_normal_cut[dataset].size * a_normal_cut[dataset].dtype.itemsize)
    #         b_cut_normal_gpu[dataset] = cuda.mem_alloc(
    #             b_normal_cut[dataset].size * b_normal_cut[dataset].dtype.itemsize)
    #         c_cut_normal_gpu[dataset] = cuda.mem_alloc(
    #             c_normal_cut[dataset].size * c_normal_cut[dataset].dtype.itemsize)
    #         d_cut_normal_gpu[dataset] = cuda.mem_alloc(
    #             d_normal_cut[dataset].size * d_normal_cut[dataset].dtype.itemsize)
    #
    #         a_cf_cut_gpu[dataset] = cuda.mem_alloc(a_cf_cut[dataset].size * a_cf_cut[dataset].dtype.itemsize)
    #         b_cf_cut_gpu[dataset] = cuda.mem_alloc(b_cf_cut[dataset].size * b_cf_cut[dataset].dtype.itemsize)
    #         c_cf_cut_gpu[dataset] = cuda.mem_alloc(c_cf_cut[dataset].size * c_cf_cut[dataset].dtype.itemsize)
    #         d_cf_cut_gpu[dataset] = cuda.mem_alloc(d_cf_cut[dataset].size * d_cf_cut[dataset].dtype.itemsize)
    #
    #         distance_to_center_plane_gpu_cut[dataset] = cuda.mem_alloc(
    #             distance_to_center_plane_cut[dataset].size * distance_to_center_plane_cut[dataset].dtype.itemsize)
    #         distance_to_center_plane_normal_gpu_cut[dataset] = cuda.mem_alloc(
    #             distance_to_center_plane_normal_cut[dataset].size * distance_to_center_plane_normal_cut[
    #                 dataset].dtype.itemsize)
    #
    #         sum_vor_gpu[dataset] = cuda.mem_alloc(sum_vor_cut[dataset].size * sum_vor_cut[dataset].dtype.itemsize)
    #
    #         sum_vor_pinned[dataset] = cuda.register_host_memory(sum_vor_cut[dataset])
    #         assert np.all(sum_vor_pinned[dataset] == sum_vor_cut[dataset])
    #         cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_pinned[dataset], stream[dataset])
    #         # sum_vor_gpu[dataset] = np.intp(x.base.get_device_pointer())
    #         # cuda.memcpy_htod_async(probability_gpu[dataset], probability_cut[dataset])
    #         cuda.memcpy_htod_async(a_cut_gpu[dataset], a_cut[dataset])
    #         cuda.memcpy_htod_async(b_cut_gpu[dataset], b_cut[dataset])
    #         cuda.memcpy_htod_async(c_cut_gpu[dataset], c_cut[dataset])
    #         cuda.memcpy_htod_async(d_cut_gpu[dataset], d_cut[dataset])
    #         cuda.memcpy_htod_async(a_cut_normal_gpu[dataset], a_normal_cut[dataset])
    #         cuda.memcpy_htod_async(b_cut_normal_gpu[dataset], b_normal_cut[dataset])
    #         cuda.memcpy_htod_async(c_cut_normal_gpu[dataset], c_normal_cut[dataset])
    #         cuda.memcpy_htod_async(d_cut_normal_gpu[dataset], d_normal_cut[dataset])
    #
    #         cuda.memcpy_htod_async(a_cf_cut_gpu[dataset], a_cf_cut[dataset])
    #         cuda.memcpy_htod_async(b_cf_cut_gpu[dataset], b_cf_cut[dataset])
    #         cuda.memcpy_htod_async(c_cf_cut_gpu[dataset], c_cf_cut[dataset])
    #         cuda.memcpy_htod_async(d_cf_cut_gpu[dataset], d_cf_cut[dataset])
    #         cuda.memcpy_htod_async(distance_to_center_plane_gpu_cut[dataset], distance_to_center_plane_cut[dataset])
    #         cuda.memcpy_htod_async(distance_to_center_plane_normal_gpu_cut[dataset],
    #                                distance_to_center_plane_normal_cut[dataset])
    #
    #     adjust_coef = np.ascontiguousarray(adjust_coef.reshape(
    #         adjust_coef.shape[0] * adjust_coef.shape[1] * adjust_coef.shape[2]),
    #         dtype=np.float32)
    #     # fov_cut_matrix = np.ascontiguousarray(fov_cut_matrix.reshape(
    #     #     fov_cut_matrix.shape[0] * fov_cut_matrix.shape[1] * fov_cut_matrix.shape[2]),
    #     #     dtype=np.float32)
    #     A = np.ascontiguousarray(A.reshape(
    #         A.shape[0] * A.shape[1] * A.shape[2]),
    #         dtype=np.short)
    #     B = np.ascontiguousarray(B.reshape(
    #         B.shape[0] * B.shape[1] * B.shape[2]),
    #         dtype=np.short)
    #     C = np.ascontiguousarray(C.reshape(
    #         C.shape[0] * C.shape[1] * C.shape[2]),
    #         dtype=np.short)
    #
    #     # ---- Divide into datasets variables backprojection
    #     for dataset in range(number_of_datasets_back):
    #         voxels_division = adjust_coef.shape[0] // number_of_datasets_back
    #         adjust_coef_cut[dataset] = np.ascontiguousarray(
    #             adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division],
    #             dtype=np.float32)
    #
    #         fov_cut_matrix_cut[dataset] = np.ascontiguousarray(
    #             fov_cut_matrix_shappened[dataset * voxels_division:(dataset + 1) * voxels_division],
    #             dtype=np.byte)
    #
    #         A_cut[dataset] = np.ascontiguousarray(
    #             A[dataset * voxels_division:(dataset + 1) * voxels_division],
    #             dtype=np.short)
    #
    #         B_cut[dataset] = np.ascontiguousarray(
    #             B[dataset * voxels_division:(dataset + 1) * voxels_division],
    #             dtype=np.short)
    #
    #         C_cut[dataset] = np.ascontiguousarray(
    #             C[dataset * voxels_division:(dataset + 1) * voxels_division],
    #             dtype=np.short)
    #         # Backprojection
    #         adjust_coef_gpu[dataset] = cuda.mem_alloc(
    #             adjust_coef_cut[dataset].size * adjust_coef_cut[dataset].dtype.itemsize)
    #
    #         adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
    #         assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
    #         cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])
    #
    #         fov_cut_matrix_cutted_gpu[dataset] = cuda.mem_alloc(
    #             fov_cut_matrix_cut[dataset].size * fov_cut_matrix_cut[dataset].dtype.itemsize)
    #
    #         A_cut_gpu[dataset] = cuda.mem_alloc(
    #             A_cut[dataset].size * A_cut[dataset].dtype.itemsize)
    #
    #         B_cut_gpu[dataset] = cuda.mem_alloc(
    #             B_cut[dataset].size * B_cut[dataset].dtype.itemsize)
    #
    #         C_cut_gpu[dataset] = cuda.mem_alloc(
    #             C_cut[dataset].size * C_cut[dataset].dtype.itemsize)
    #
    #         cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset])
    #         cuda.memcpy_htod_async(fov_cut_matrix_cutted_gpu[dataset], fov_cut_matrix_cut[dataset])
    #         cuda.memcpy_htod_async(A_cut_gpu[dataset], A_cut[dataset])
    #         cuda.memcpy_htod_async(B_cut_gpu[dataset], B_cut[dataset])
    #         cuda.memcpy_htod_async(C_cut_gpu[dataset], C_cut[dataset])
    #
    #     free, total = cuda.mem_get_info()
    #
    #     print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))
    #
    #     # -------------OSEM---------
    #     it = self.number_of_iterations
    #     subsets = self.number_of_subsets
    #     print('Number events for reconstruction: {}'.format(number_of_events))
    #
    #     im = np.ascontiguousarray(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]), dtype=np.float32)
    #     for i in range(it):
    #         print('Iteration number: {}\n----------------'.format(i + 1))
    #         begin_event = np.int32(0)
    #         end_event = np.int32(number_of_events / subsets)
    #         for sb in range(subsets):
    #             print('Subset number: {}'.format(sb))
    #             number_of_events_subset = np.int32(end_event - begin_event)
    #             tic = time.time()
    #             # Cycle forward Projection
    #             for dataset in range(number_of_datasets):
    #
    #                 if dataset == number_of_datasets:
    #                     begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
    #                     end_dataset = number_of_events_subset
    #                 else:
    #                     begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
    #                     end_dataset = np.int32((dataset + 1) * number_of_events_subset / number_of_datasets)
    #
    #                 threadsperblock = (256, 1, 1)
    #                 blockspergrid_x = int(math.ceil(((end_dataset - begin_dataset)) / threadsperblock[0]))
    #                 blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
    #                 blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
    #                 blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    #                 event[dataset]['kernel_begin'].record(stream[dataset])
    #                 # depth = np.int32(5)
    #                 # weight = np.int32(A.shape[2]/2)
    #                 # height = np.int32(A.shape[2]/2)
    #                 if self.cdrf:
    #                     func_forward = mod_forward_projection_shared_mem_cdrf.get_function("forward_projection_cdrf")
    #                     func_forward(weight, height, depth, start_x, start_y, start_z,
    #                                  distance_to_center_plane_normal_gpu_cut[dataset],
    #                                  distance_to_center_plane_gpu_cut[dataset],
    #                                  half_distance_between_array_pixel,
    #                                  number_of_events, begin_dataset, end_dataset, a_cut_gpu[dataset],
    #                                  a_cut_normal_gpu[dataset], a_cf_cut_gpu[dataset],
    #                                  b_cut_gpu[dataset], b_cut_normal_gpu[dataset], b_cf_cut_gpu[dataset],
    #                                  c_cut_gpu[dataset], c_cut_normal_gpu[dataset],
    #                                  c_cf_cut_gpu[dataset],
    #                                  d_cut_gpu[dataset],
    #                                  d_cut_normal_gpu[dataset], d_cf_cut_gpu[dataset],
    #                                  sum_vor_gpu[dataset], fov_cut_matrix_gpu, im_gpu,
    #                                  block=threadsperblock,
    #                                  grid=blockspergrid,
    #                                  stream=stream[dataset])
    #                 else:
    #                     func_forward = mod_forward_projection_shared_mem.get_function("forward_projection")
    #                     func_forward(weight, height, depth, start_x, start_y, start_z, half_crystal_pitch_xy,
    #                                  half_crystal_pitch_z,
    #                                  half_distance_between_array_pixel,
    #                                  number_of_events, begin_dataset, end_dataset, a_cut_gpu[dataset],
    #                                  a_cut_normal_gpu[dataset], a_cf_cut_gpu[dataset],
    #                                  b_cut_gpu[dataset], b_cut_normal_gpu[dataset], b_cf_cut_gpu[dataset],
    #                                  c_cut_gpu[dataset], c_cut_normal_gpu[dataset],
    #                                  c_cf_cut_gpu[dataset],
    #                                  d_cut_gpu[dataset],
    #                                  d_cut_normal_gpu[dataset], d_cf_cut_gpu[dataset],
    #                                  sum_vor_gpu[dataset], fov_cut_matrix_gpu, im_gpu,
    #                                  block=threadsperblock,
    #                                  grid=blockspergrid,
    #                                  stream=stream[dataset])
    #
    #             # Sincronization of streams
    #             for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
    #                 event[dataset]['kernel_end'].record(stream[dataset])
    #
    #             # Transfering data from Optimizer
    #             for dataset in range(number_of_datasets):
    #                 begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
    #                 end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
    #                 # cuda.memcpy_dtoh_async(, sum_vor_gpu[dataset])
    #                 cuda.memcpy_dtoh_async(sum_vor_pinned[dataset], sum_vor_gpu[dataset], stream[dataset])
    #                 sum_vor[begin_dataset:end_dataset] = sum_vor_pinned[dataset]
    #                 # cuda.cudaStream.Synchronize(stream[dataset])
    #
    #                 toc = time.time()
    #
    #             cuda.Context.synchronize()
    #
    #             print('Time part Forward Projection {} : {}'.format(1, toc - tic))
    #             # number_of_datasets = np.int32(2)
    #             # teste = np.copy(sum_vor)
    #             # # sum_vor[sum_vor<1]=0
    #             # sum_vor = np.ascontiguousarray(teste, dtype=np.float32)
    #             # sum_vor=np.ascontiguousarray(np.ones((self.a.shape)), dtype=np.float32)
    #             print('SUM VOR: {}'.format(np.sum(sum_vor)))
    #             # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)
    #
    #             cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)
    #
    #             # ------------BACKPROJECTION-----------
    #
    #             for dataset in range(number_of_datasets_back):
    #                 dataset = np.int32(dataset)
    #                 begin_dataset = np.int32(0)
    #                 end_dataset = np.int32(number_of_events_subset)
    #
    #                 # begin_dataset = np.int32(0)
    #                 # end_dataset = np.int32(number_of_events)
    #
    #                 event[dataset]['kernel_begin'].record(stream[dataset])
    #                 # weight_cutted, height_cutted, depth_cutted = np.int32(adjust_coef_cut[dataset].shape[0]), np.int32(
    #                 #     adjust_coef_cut[dataset].shape[1]), np.int32(adjust_coef_cut[dataset].shape[2])
    #                 weight_cutted, height_cutted, depth_cutted = np.int32(adjust_coef_cut[dataset].shape[0]), np.int32(
    #                     1), np.int32(1)
    #
    #                 number_of_voxels_thread = 128
    #                 threadsperblock = (np.int(number_of_voxels_thread), 1, 1)
    #                 blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
    #                 blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
    #                 blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
    #                 # blockspergrid_y = int(math.ceil(adjust_coef_cut[dataset].shape[1] / threadsperblock[1]))
    #                 # blockspergrid_z = int(math.ceil(adjust_coef_cut[dataset].shape[2] / threadsperblock[2]))
    #                 blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    #                 shared_memory = threadsperblock[0] * threadsperblock[1] * threadsperblock[2] * 4
    #                 if self.cdrf:
    #                     if self.normalization_calculation_flag:
    #                         func_backward = mod_normalization_shared_mem_cdrf.get_function("normalization_cdrf")
    #
    #
    #                     else:
    #                         func_backward = mod_backward_projection_shared_mem_cdrf.get_function("backprojection_cdrf")
    #                     func_backward(dataset, weight_cutted, height_cutted, depth_cutted,
    #                                   distance_to_center_plane_normal_gpu,
    #                                   distance_to_center_plane_gpu, half_distance_between_array_pixel,
    #                                   number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
    #                                   b_gpu, b_normal_gpu, b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
    #                                   d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset],
    #                                   C_cut_gpu[dataset],
    #                                   adjust_coef_gpu[dataset],
    #                                   sum_vor_t_gpu, fov_cut_matrix_cutted_gpu[dataset], time_factor_gpu,
    #                                   block=threadsperblock,
    #                                   grid=blockspergrid,
    #                                   shared=np.int(4 * number_of_voxels_thread),
    #                                   stream=stream[dataset],
    #                                   )
    #                 else:
    #                     if self.normalization_calculation_flag:
    #                         func_backward = mod_normalization_shared_mem.get_function("normalization")
    #
    #
    #                     else:
    #                         func_backward = mod_backward_projection_shared_mem.get_function("backprojection")
    #
    #                     func_backward(dataset, weight_cutted, height_cutted, depth_cutted, half_crystal_pitch_xy,
    #                                   half_crystal_pitch_z, half_distance_between_array_pixel,
    #                                   number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
    #                                   b_gpu, b_normal_gpu, b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
    #                                   d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset],
    #                                   C_cut_gpu[dataset],
    #                                   adjust_coef_gpu[dataset],
    #                                   sum_vor_t_gpu, fov_cut_matrix_cutted_gpu[dataset], time_factor_gpu,
    #                                   block=threadsperblock,
    #                                   grid=blockspergrid,
    #                                   shared=np.int(4 * number_of_voxels_thread),
    #                                   stream=stream[dataset],
    #                                   )
    #
    #             for dataset in range(number_of_datasets_back):  # Commenting out this line should break concurrency.
    #                 event[dataset]['kernel_end'].record(stream[dataset])
    #
    #             for dataset in range(number_of_datasets_back):
    #                 cuda.memcpy_dtoh_async(adjust_coef_cut[dataset], adjust_coef_gpu[dataset])
    #                 adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division] = adjust_coef_cut[dataset]
    #
    #             cuda.Context.synchronize()
    #             print('Time part Backward Projection {} : {}'.format(1, time.time() - toc))
    #
    #             # Image Normalization
    #             # if i ==0:
    #             #     norm_im=np.copy(adjust_coef)
    #             #     norm_im=norm_im/np.max(norm_im)
    #             #     norm_im[norm_im == 0] = np.min(norm_im[np.nonzero(norm_im)])
    #             # normalization_matrix = gaussian_filter(normalization_matrix, 0.5)
    #
    #             # im_med = np.load("C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\PhD\\Reconstrução\\NAF+FDG\\Easypet Scan 05 Aug 2019 - 14h 36m 33s\\static_image\\Easypet Scan 05 Aug 2019 - 14h 36m 33s mlem.npy")
    #             # self.algorithm = "LM-MRP"
    #             if self.algorithm == "LM-MRP":
    #                 beta = self.algorithm_options[0]
    #                 kernel_filter_size = self.algorithm_options[1]
    #                 im_to_filter = im.reshape(weight, height, depth)
    #                 im_med = median_filter(im_to_filter, kernel_filter_size)
    #                 penalized_term = np.copy(im_to_filter)
    #                 penalized_term[im_med != 0] = 1 + beta * (im_to_filter[im_med != 0] - im_med[im_med != 0]) / im_med[
    #                     im_med != 0]
    #                 penalized_term = np.ascontiguousarray(penalized_term.reshape(weight * height * depth),
    #                                                       dtype=np.float32)
    #                 # penalized_term = np.ascontiguousarray(penalized_term, dtype=np.float32)
    #
    #             if self.algorithm == "MAP":
    #                 beta = 0.5
    #                 im_map = np.load(
    #                     "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\PhD\\Reconstrução\\NAF+FDG\\Easypet Scan 05 Aug 2019 - 14h 36m 33s\\static_image\\Easypet Scan 05 Aug 2019 - 14h 36m 33s mlem.npy")
    #
    #             im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
    #                 normalization_matrix != 0] / (normalization_matrix[normalization_matrix != 0])
    #             im[normalization_matrix == 0] = 0
    #             if self.algorithm == "LM-MRP":
    #                 im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
    #             # im = fourier_gaussian(im, sigma=0.2)
    #             # im = gaussian_filter(im, 0.4)
    #             print('SUM IMAGE: {}'.format(np.sum(im)))
    #             im = np.ascontiguousarray(im, dtype=np.float32)
    #             # im = im * adjust_coef / sensivity_matrix[np.nonzero(sensivity_matrix)]
    #             cuda.memcpy_htod_async(im_gpu, im)
    #
    #             # Clearing variables
    #             sum_vor = np.ascontiguousarray(
    #                 np.zeros(self.a.shape, dtype=np.float32))
    #
    #             adjust_coef = np.ascontiguousarray(
    #                 np.zeros((self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z),
    #                          dtype=np.float32))
    #
    #             for dataset in range(number_of_datasets):
    #                 # if dataset == number_of_datasets:
    #                 #     begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
    #                 #     end_dataset = number_of_events
    #                 #
    #                 # else:
    #                 begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
    #                 end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
    #                 # adjust_coef_cut[dataset] = np.ascontiguousarray(adjust_coef[:, :,
    #                 #                                                 int(np.floor(im_cut_dim[2] * dataset)):int(
    #                 #                                                     np.floor(im_cut_dim[2] * (dataset + 1)))],
    #                 #                                                 dtype=np.float32)
    #
    #                 sum_vor_cut[dataset] = sum_vor[begin_dataset:end_dataset]
    #                 # cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_cut[dataset])
    #                 sum_vor_pinned[dataset] = cuda.register_host_memory(sum_vor_cut[dataset])
    #                 assert np.all(sum_vor_pinned[dataset] == sum_vor_cut[dataset])
    #                 cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_pinned[dataset], stream[dataset])
    #
    #             for dataset in range(number_of_datasets_back):
    #                 adjust_coef_cut[dataset] = adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division]
    #                 adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
    #                 assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
    #                 cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])
    #
    #             if self.saved_image_by_iteration:
    #                 im = im.reshape(weight, height, depth)
    #                 self._save_image_by_it(im, i, sb)
    #
    #             if self.signals_interface is not None:
    #                 self.signals_interface.trigger_update_label_reconstruction_status.emit(
    #                     "{}: Iteration {}".format(self.current_info_step, i + 1))
    #                 self.signals_interface.trigger_progress_reconstruction_partial.emit(
    #                     int(np.round(100 * (i + 1) * (sb + subsets) / (it * subsets), 0)))
    #
    #     im = im.reshape(weight, height, depth)
    #     return im * subsets

    def _vor_design_gpu_shared_memory_multiple_reads_DOI(self, a, a_normal, a_cf, A, b, b_normal, b_cf, B, c, c_normal,
                                                         c_cf, C,
                                                         d, d_normal, d_cf, adjust_coef, im, half_crystal_pitch_xy,
                                                         half_crystal_pitch_z,
                                                         sum_vor, fov_cut_matrix, normalization_matrix, time_factor):
        print('Optimizer STARTED - Multiple reads')
        # cuda.init()
        cuda = self.cuda_drv
        # device = cuda.Device(0)  # enter your gpu id here
        # ctx = device.make_context()
        number_of_events = np.int32(len(a))
        weight = np.int32(A.shape[0])
        height = np.int32(A.shape[1])
        depth = np.int32(A.shape[2])
        # start_x = np.int32(A[0, 0, 0])
        start_x = np.int32(A[0, 0, 0])
        start_y = np.int32(B[0, 0, 0])
        start_z = np.int32(C[0, 0, 0])
        print("Start_point: {},{},{}".format(start_x, start_y, start_z))
        print('Image size: {},{}, {}'.format(weight, height, depth))

        half_distance_between_array_pixel = np.float32(self.distance_between_array_pixel / 2)
        normalization_matrix = normalization_matrix.reshape(
            normalization_matrix.shape[0] * normalization_matrix.shape[1] * normalization_matrix.shape[2])

        # SOURCE MODELS (DEVICE CODE)

        mod_forward_projection_shared_mem = SourceModule("""
        #include <stdint.h>

        texture<char, 1> tex;


        __global__ void forward_projection
        (const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
        const float crystal_pitch_XY,const float crystal_pitch_Z,const float distance_between_array_pixel,
        int number_of_events, const int begin_event_gpu_limitation,  const int end_event_gpu_limitation, 
        float *a, float *a_normal, float *a_cf, float *b,float *b_normal, float *b_cf,float *c,float *c_normal, 
        float *c_cf, float *d, float *d_normal, float *d_cf, float *sum_vor, const char *fov_cut_matrix, const float *im_old)
                               {
                                  const int shared_memory_size = 256;
                                   __shared__ float a_shared[shared_memory_size];
                                   __shared__ float b_shared[shared_memory_size];
                                   __shared__ float c_shared[shared_memory_size];
                                   __shared__ float d_shared[shared_memory_size];
                                   __shared__ float a_normal_shared[shared_memory_size];
                                   __shared__ float b_normal_shared[shared_memory_size];
                                   __shared__ float c_normal_shared[shared_memory_size];
                                   __shared__ float d_normal_shared[shared_memory_size];
                                   __shared__ float a_cf_shared[shared_memory_size];
                                   __shared__ float b_cf_shared[shared_memory_size];
                                   __shared__ float c_cf_shared[shared_memory_size];
                                   __shared__ float d_cf_shared[shared_memory_size];
                                   __shared__ float sum_vor_shared[shared_memory_size];
                                   __shared__ char fov_cut_matrix_shared[shared_memory_size];
                                   /*
                                  const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
                                  const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
                                  */
                                  int threadId=blockIdx.x *blockDim.x + threadIdx.x;
                                  int e;

                                  float d2;
                                  float d2_normal;
                                  float d2_cf;
                                  float value;
                                  float value_normal;
                                  float value_cf;
                                  float sum_vor_temp;
                                  int index;                                  
                                  int x_t;
                                  int y_t;
                                  int z_t;
                                  float max_distance_projector;     
                                  const int number_events_max = end_event_gpu_limitation-begin_event_gpu_limitation;
                                  const float error_pixel = 0.0000f;
                                   if (threadIdx.x>shared_memory_size)
                                   {
                                   return;
                                   }
                                   if(threadId>number_events_max)
                                   {
                                   return;
                                   }

                                  __syncthreads();
                                  e = threadId;
                                  int e_m = threadIdx.x;
                                  a_shared[e_m] = a[e];
                                  b_shared[e_m] = b[e];
                                  c_shared[e_m] = c[e];
                                  d_shared[e_m] = d[e];
                                  a_normal_shared[e_m] = a_normal[e];
                                  b_normal_shared[e_m] = b_normal[e];
                                  c_normal_shared[e_m] = c_normal[e];
                                  d_normal_shared[e_m] = d_normal[e];
                                  a_cf_shared[e_m] = a_cf[e];
                                  b_cf_shared[e_m] = b_cf[e];
                                  c_cf_shared[e_m] = c_cf[e];
                                  d_cf_shared[e_m] = d_cf[e];
                                  sum_vor_shared[e_m] = sum_vor[e];   

                                  d2_normal = crystal_pitch_XY * sqrt(a_normal_shared[e_m]*a_normal_shared[e_m]+b_normal_shared[e_m]*b_normal_shared[e_m]+c_normal_shared[e_m]*c_normal_shared[e_m]); 
                                  d2 = crystal_pitch_Z * sqrt(a_shared[e_m]*a_shared[e_m] + b_shared[e_m]*b_shared[e_m] + c_shared[e_m]*c_shared[e_m]);
                                  d2_cf = distance_between_array_pixel*sqrt(a_cf_shared[e_m]*a_cf_shared[e_m]+b_cf_shared[e_m]*b_cf_shared[e_m]+c_cf_shared[e_m]*c_cf_shared[e_m]);
                                  max_distance_projector=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf);                             

                                  for(int l=0; l<n; l++)
                                  {  
                                      x_t = l+start_x;
                                      for(int j=0; j<m; j++)
                                      {  
                                         y_t = j+start_y;
                                         for(int k=0; k<p; k++)
                                              {
                                              /*
                                              index = l+j*n+k*m*n;                                              
                                              fov_cut_matrix_shared[k]=fov_cut_matrix[k+j*p+l*p*m];
                                              index = l+j*n+k*m*n; 
                                              */
                                                index = k+j*p+l*p*m;                                                


                                              if (fov_cut_matrix[index]!=0)
                                              {                                                

                                                  z_t = k+start_z;
                                                  value_normal = a_normal_shared[e_m]*x_t+b_normal_shared[e_m]*y_t+c_normal_shared[e_m] * z_t -d_normal_shared[e_m];                                                 

                                                   if (value_normal < d2_normal && value_normal >= -d2_normal)
                                                   {
                                                    value = a_shared[e_m]*x_t+b_shared[e_m]*y_t+c_shared[e_m]*z_t-d_shared[e_m];

                                                    if (value < d2 && value >=-d2 )
                                                     {
                                                          value_cf = a_cf_shared[e_m]*x_t+b_cf_shared[e_m]*y_t+c_cf_shared[e_m]*z_t-d_cf_shared[e_m];


                                                      if (value_cf >= -d2_cf && value_cf<d2_cf)
                                                          {

                                                           sum_vor_temp +=im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector);
                                                           /* im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector)

                                                            */
                                                            }


                                                  }




                                              }
                                              }

                                           }
                                       }
                                   }

                                   /*
                                    sum_vor[e]= sum_vor_temp;
                                   sum_vor[e]= sum_vor_shared[e_m];
                                   __syncthreads();
                                   sum_vor[e]= sum_vor_shared[e_m];
                                   */
                                   sum_vor[e]= sum_vor_temp;



               }""")
        mod_normalization_shared_mem = SourceModule("""
                             #include <stdint.h>
                             texture<uint8_t, 1> tex;

                      __global__ void normalization
                       ( int dataset_number, int n, int m, int p, const float crystal_pitch_XY, const float crystal_pitch_Z, 
                       const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation, 
                       const int end_event_gpu_limitation, const float *a, const float *a_normal, const float *a_cf, const float *b,
                       const float *b_normal, const float *b_cf, const float *c, const float *c_normal, const float *c_cf, const float *d, const float *d_normal,
                       const float *d_cf, const short *A, const short *B, const short *C, float *adjust_coef, float *sum_vor, 
                       char *fov_cut_matrix, float *time_factor)
                       {
                                 extern __shared__ float adjust_coef_shared[];      
                                 int idt=blockIdx.x *blockDim.x + threadIdx.x;                            


                                 float d2;
                                 float d2_normal;
                                 float d2_cf;
                                 float normal_value;
                                 float value;
                                 float value_cf;                                              
                                 short a_temp;
                                 short b_temp;
                                 short c_temp;
                                 char fov_cut_temp;   
                                 int i_s=threadIdx.x;

                                 if (idt>n*m*p)
                                 {
                                      return;
                                  }

                                 __syncthreads();
                                 adjust_coef_shared[i_s] = adjust_coef[idt];
                                 a_temp = A[idt];
                                 b_temp = B[idt];
                                 c_temp = C[idt];
                                 fov_cut_temp = fov_cut_matrix[idt];



                                 for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
                                 {
                                     if (fov_cut_temp!=0)
                                     {   
                                     normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
                                     d2_normal = crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);

                                     if (normal_value< d2_normal && normal_value >= -d2_normal)
                                     {
                                     value = a[e]*a_temp+b[e]*b_temp +c[e]*c_temp- d[e];
                                     d2 = crystal_pitch_Z * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);


                                         if (value < d2 && value >=-d2)
                                         {
                                                   value_cf = a_cf[e]*a_temp+b_cf[e]*b_temp +c_cf[e] * c_temp-d_cf[e];
                                                   d2_cf = distance_between_array_pixel*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);


                                                     if (value_cf >= -d2_cf && value_cf<d2_cf)
                                                     {    


                                                            adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf));


                                                    }

                                                         /*

                                                         adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf));
                                                         adjust_coef_shared[i_s] += 1/sum_vor[e];  
                                                         normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
                                          d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
                                                        adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);  
                                                       adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);  
                                                          */





                                              }
                                          }

                                 }

                                 }

                                 adjust_coef[idt] = adjust_coef_shared[i_s];
                                 __syncthreads();



                             }
                             """)
        mod_backward_projection_shared_mem = SourceModule("""
                      #include <stdint.h>
                      texture<uint8_t, 1> tex;

               __global__ void backprojection
                ( int dataset_number, int n, int m, int p, const float crystal_pitch_XY, const float crystal_pitch_Z, 
                const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation, 
                const int end_event_gpu_limitation, const float *a, const float *a_normal, const float *a_cf, const float *b,
                const float *b_normal, const float *b_cf, const float *c, const float *c_normal, const float *c_cf, const float *d, const float *d_normal,
                const float *d_cf, const short *A, const short *B, const short *C, float *adjust_coef, float *sum_vor, 
                char *fov_cut_matrix, float *time_factor)
                {
                          extern __shared__ float adjust_coef_shared[];      
                          int idt=blockIdx.x *blockDim.x + threadIdx.x;                            


                          float d2;
                          float d2_normal;
                          float d2_cf;
                          float normal_value;
                          float value;
                          float value_cf;                                              
                          short a_temp;
                          short b_temp;
                          short c_temp;
                          char fov_cut_temp;   
                          int i_s=threadIdx.x;

                          if (idt>n*m*p)
                          {
                               return;
                           }

                          __syncthreads();
                          adjust_coef_shared[i_s] = adjust_coef[idt];
                          a_temp = A[idt];
                          b_temp = B[idt];
                          c_temp = C[idt];
                          fov_cut_temp = fov_cut_matrix[idt];



                          for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
                          {
                              if (fov_cut_temp!=0)
                              {   
                              normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
                              d2_normal = crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);

                              if (normal_value< d2_normal && normal_value >= -d2_normal)
                              {
                              value = a[e]*a_temp+b[e]*b_temp +c[e]*c_temp- d[e];
                              d2 = crystal_pitch_Z * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);


                                  if (value < d2 && value >=-d2)
                                  {
                                            value_cf = a_cf[e]*a_temp+b_cf[e]*b_temp +c_cf[e] * c_temp-d_cf[e];
                                            d2_cf = distance_between_array_pixel*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);


                                              if (value_cf >= -d2_cf && value_cf<d2_cf)
                                              {    



                                                    adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]*time_factor[e]);

                                             }

                                                  /*

                                                  adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]*time_factor[e]);
                                                  adjust_coef_shared[i_s] += 1/sum_vor[e];  
                                                  normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
                                   d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
                                                 adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);  
                                                adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);  
                                                   */





                                       }
                                   }

                          }

                          }

                          adjust_coef[idt] = adjust_coef_shared[i_s];
                          __syncthreads();



                      }
                      """)
        mod_forward_projection_shared_mem_cdrf = SourceModule("""
               #include <stdint.h>

               texture<char, 1> tex;


               __global__ void forward_projection_cdrf
               (const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
               float *crystal_pitch_XY, float *crystal_pitch_Z,const float distance_between_array_pixel,
               int number_of_events, const int begin_event_gpu_limitation,  const int end_event_gpu_limitation, 
               float *a, float *a_normal, float *a_cf, float *b,float *b_normal, float *b_cf,float *c,float *c_normal, 
               float *c_cf, float *d, float *d_normal, float *d_cf, float *sum_vor, const char *fov_cut_matrix, 
               const float *im_old, float* m_values, float* b_values, float* m_values_at, float* b_values_at,
                                     float* max_D, float* inflex_points_x, float* linear_attenuation_A,
                                     float* linear_attenuation_B)
                                      {
                                         const int shared_memory_size = 256;
                                          __shared__ float a_shared[shared_memory_size];
                                          __shared__ float b_shared[shared_memory_size];
                                          __shared__ float c_shared[shared_memory_size];
                                          __shared__ float d_shared[shared_memory_size];
                                          __shared__ float a_normal_shared[shared_memory_size];
                                          __shared__ float b_normal_shared[shared_memory_size];
                                          __shared__ float c_normal_shared[shared_memory_size];
                                          __shared__ float d_normal_shared[shared_memory_size];
                                          __shared__ float a_cf_shared[shared_memory_size];
                                          __shared__ float b_cf_shared[shared_memory_size];
                                          __shared__ float c_cf_shared[shared_memory_size];
                                          __shared__ float d_cf_shared[shared_memory_size];
                                          __shared__ float sum_vor_shared[shared_memory_size];
                                          __shared__ char fov_cut_matrix_shared[shared_memory_size];
                                          __shared__ float crystal_pitch_Z_shared[shared_memory_size];
                                          __shared__ float crystal_pitch_XY_shared[shared_memory_size];
                                          __shared__ float m_values_init_shared[shared_memory_size];
                                          __shared__ float m_values_end_shared[shared_memory_size];
                                          __shared__ float b_values_init_shared[shared_memory_size];
                                          __shared__ float b_values_end_shared[shared_memory_size];
                                          __shared__ float m_values_at_shared[shared_memory_size];
                                          __shared__ float b_values_at_shared[shared_memory_size];
                                          __shared__ float max_D_shared[shared_memory_size];
                                          __shared__ float inflex_points_x_init_shared[shared_memory_size];
                                          __shared__ float inflex_points_x_end_shared[shared_memory_size];
                                          __shared__ float linear_attenuation_A_shared[shared_memory_size];
                                          __shared__ float linear_attenuation_B_shared[shared_memory_size];
                                          
                                          /*
                                         const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
                                         const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
                                         */
                                         int threadId=blockIdx.x *blockDim.x + threadIdx.x;
                                         int e;

                                         float d2;
                                         float d2_normal;
                                         float d2_cf;
                                         float value;
                                         float value_normal;
                                         float value_cf;
                                         float sum_vor_temp;
                                         int index;                                  
                                         int x_t;
                                         int y_t;
                                         int z_t;
                                         float width;
                                         float height;
                                         float distance;
                                         float distance_other; 
                                         float distance_crystal;
                                         float distance_at;
                                         float idrf;
                                         


                                         float max_distance_projector;

                                         const int number_events_max = end_event_gpu_limitation-begin_event_gpu_limitation;
                                         const float error_pixel = 0.0000f;
                                          if (threadIdx.x>shared_memory_size)
                                          {
                                          return;
                                          }
                                          if(threadId>number_events_max)
                                          {
                                          return;
                                          }

                                         __syncthreads();
                                         e = threadId;
                                         int e_m = threadIdx.x;
                                         a_shared[e_m] = a[e];
                                         b_shared[e_m] = b[e];
                                         c_shared[e_m] = c[e];
                                         d_shared[e_m] = d[e];
                                         a_normal_shared[e_m] = a_normal[e];
                                         b_normal_shared[e_m] = b_normal[e];
                                         c_normal_shared[e_m] = c_normal[e];
                                         d_normal_shared[e_m] = d_normal[e];
                                         a_cf_shared[e_m] = a_cf[e];
                                         b_cf_shared[e_m] = b_cf[e];
                                         c_cf_shared[e_m] = c_cf[e];
                                         d_cf_shared[e_m] = d_cf[e];
                                         sum_vor_shared[e_m] = sum_vor[e];
                                         crystal_pitch_Z_shared[e_m] =  crystal_pitch_Z[e];  
                                         crystal_pitch_XY_shared[e_m] =  crystal_pitch_XY[e];  
                                         m_values_init_shared[e_m] =  m_values[2*e];  
                                         m_values_end_shared[e_m] =  m_values[2*e+1];  
                                         b_values_init_shared[e_m] =  b_values[2*e];  
                                         b_values_end_shared[e_m] =  b_values[2*e+1];  
                                         m_values_at_shared[e_m] =  m_values_at[e];  
                                         b_values_at_shared[e_m] =  b_values_at[e];  
                                         max_D_shared[e_m] = max_D[e];  
                                         inflex_points_x_init_shared[e_m] =  inflex_points_x[2*e];  
                                         inflex_points_x_end_shared[e_m] =  inflex_points_x[2*e+1];  
                                         linear_attenuation_A_shared[e_m] =  linear_attenuation_A[e];  
                                         linear_attenuation_B_shared[e_m] =  linear_attenuation_B[e];  

                                         d2_normal = crystal_pitch_XY_shared[e_m] * sqrt(a_normal_shared[e_m]*a_normal_shared[e_m]+b_normal_shared[e_m]*b_normal_shared[e_m]+c_normal_shared[e_m]*c_normal_shared[e_m]); 
                                         d2 = crystal_pitch_Z_shared[e_m]* sqrt(a_shared[e_m]*a_shared[e_m] + b_shared[e_m]*b_shared[e_m] + c_shared[e_m]*c_shared[e_m]);
                                         d2_cf = distance_between_array_pixel*sqrt(a_cf_shared[e_m]*a_cf_shared[e_m]+b_cf_shared[e_m]*b_cf_shared[e_m]+c_cf_shared[e_m]*c_cf_shared[e_m]);
                                         max_distance_projector=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf);                             

                                         for(int l=0; l<n; l++)
                                         {  
                                             x_t = l+start_x;
                                             for(int j=0; j<m; j++)
                                             {  
                                                y_t = j+start_y;
                                                for(int k=0; k<p; k++)
                                                     {
                                                     /*
                                                     index = l+j*n+k*m*n;                                              
                                                     fov_cut_matrix_shared[k]=fov_cut_matrix[k+j*p+l*p*m];
                                                     index = l+j*n+k*m*n; 
                                                     */
                                                       index = k+j*p+l*p*m;                                                


                                                     if (fov_cut_matrix[index]!=0)
                                                     {                                                

                                                         z_t = k+start_z;
                                                         value_normal = a_normal_shared[e_m]*x_t+b_normal_shared[e_m]*y_t+c_normal_shared[e_m] * z_t -d_normal_shared[e_m];                                                 

                                                          if (value_normal < d2_normal && value_normal >= -d2_normal)
                                                          {
                                                           value = a_shared[e_m]*x_t+b_shared[e_m]*y_t+c_shared[e_m]*z_t-d_shared[e_m];

                                                           if (value < d2 && value >=-d2 )
                                                            {
                                                                 value_cf = a_cf_shared[e_m]*x_t+b_cf_shared[e_m]*y_t+c_cf_shared[e_m]*z_t-d_cf_shared[e_m];


                                                             if (value_cf >= -d2_cf && value_cf<d2_cf)
                                                                 {
                                                                 if (sqrt(value*value+value_normal*value_normal+value_cf*value_cf)<=max_distance_projector)
                                                                 {      
                                                                     width = 2*(crystal_pitch_Z_shared[e_m]  - abs(value));
                                                                    height = 2*(crystal_pitch_XY_shared[e_m] - abs(value_normal));

                                                                     distance =d2_cf+abs(value_cf);
                                                                    distance_other = abs(d2_cf+value_cf);
                                                                        
                                                                    distance_at = m_values_at_shared[e_m]*value+b_values_at_shared[e_m];
                                                                 
                                                                      
                                                                if(value<=inflex_points_x_init_shared[e_m])
                                                                {
                                                                    distance_crystal = m_values_init_shared[e_m]*value+b_values_init_shared[e_m];
                                                                    distance_at = 0;
                                                                }
                                                                 
                                                             
                                                                else if(value>=inflex_points_x_end_shared[e_m])
                                                                {
                                                                    distance_crystal = m_values_end_shared[e_m]*value+b_values_end_shared[e_m];
                                                                }
                                                                
                                                                else 
                                                                {
                                                                    distance_crystal = max_D_shared[e_m];
                                                                    
                                                                    
                                                                }
                                                                
                                                              
                                                              
                                                               idrf=(1-exp(-linear_attenuation_A_shared[e_m]*distance_crystal))*exp(-linear_attenuation_A_shared[e_m]*distance_at);
                                                               if (idrf<0)
                                                               {
                                                              
                                                               idrf = 0;
                                                               }
                                                                
                                                              
                                                              
                                                                
                                                                  sum_vor_shared[e_m]+= idrf*im_old[index];
                                                                  
                                                                  

                                                                  }
                                                                  /*
                                                                  pow(4*atan(width*height/(2*distance*sqrt(4*distance*distance+width*width+height*height))),2)
                                                                  sum_vor_shared[e_m]+= im_old[index]*(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector);
                                                                  4*arctan(
                                                                  4*np.arctan(width*height/(2*distance_to_anh*np.sqrt(4*distance_between_array_pixel-value_cf)**2+width**2+height**2)))
                                                                  *(1-sqrt(value*value+value_normal*value_normal+value_cf*value_cf)/max_distance_projector)
                                                                  */
                                                                   }


                                                         }




                                                     }
                                                     }

                                                  }
                                              }
                                          }

                                          /*
                                           sum_vor[e]= sum_vor_temp;
                                          sum_vor[e]= sum_vor_shared[e_m];


                                          sum_vor[e]= sum_vor_temp;
                                          */
                                          __syncthreads();
                                          sum_vor[e]= sum_vor_shared[e_m];



                      }""")

        mod_normalization_shared_mem_cdrf = SourceModule("""
        #include <stdint.h>
        
        texture<uint8_t, 1> tex;
        __device__ float* three_plane_intersection(float plane1A, float plane1B,
            float plane1C, float plane1D, float plane2A, float plane2B, float plane2C,
            float plane2D, float plane3A, float plane3B, float plane3C, float plane3D);
        __device__ float intersection_determinant(float matrix[3][3]);
        __device__ float point_distance_to_plane(float *point, float A, float B, float C, float D);
        
        __global__ void normalization_cdrf
        (int dataset_number, int n, int m, int p, float* crystal_pitch_XY, float* crystal_pitch_Z,
            const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
            const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
            const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
            const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
            char* fov_cut_matrix, float* time_factor, float* plane_centerA1_A,
            float* plane_centerA1_B, float* plane_centerA1_C, float* plane_centerA1_D,
            float* plane_centerB1_A, float* plane_centerB1_B, float* plane_centerB1_C,
            float* plane_centerB1_D, float* plane_centerC1_A, float* plane_centerC1_B,
            float* plane_centerC1_C, float* plane_centerC1_D, float* intersection_points, float* m_values, float* m_values_at,
            float* b_values, float* b_values_at, float* max_D, float* inflex_points_x, float* linear_attenuation_A,
            float* linear_attenuation_B)
        {
            extern __shared__ float adjust_coef_shared[];
            int idt = blockIdx.x * blockDim.x + threadIdx.x;
        
        
            float d2;
            float d2_normal;
            float d2_cf;
            float normal_value;
            float value;
            float value_cf;
            short a_temp;
            short b_temp;
            short c_temp;
            char fov_cut_temp;
            int i_s = threadIdx.x;
            float width;
            float height;
            float distance;           
            float solid_angle;
            float face_1_distance_to_center;
            float face_2_distance_to_center;
            float face_3_distance_to_center;
            float* p1;          
            float* p2;          
            float* p3;          
            float* p4;          
            float* p5;          
            float* p6;            
            float dist_p1;
            float distance_crystal;    
            float distance_at;
            float idrf;      
           
            if (idt >= n * m * p)
            {
                return;
            }
        
            __syncthreads();
            adjust_coef_shared[i_s] = adjust_coef[idt];
            a_temp = A[idt];
            b_temp = B[idt];
            c_temp = C[idt];
            fov_cut_temp = fov_cut_matrix[idt];
        
            for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
            {
                if (fov_cut_temp != 0)
                {
                    normal_value = a_normal[e] * a_temp + b_normal[e] * b_temp + c_normal[e] * c_temp - d_normal[e];
                    d2_normal = crystal_pitch_XY[e] * sqrt(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);
        
                    if (normal_value < d2_normal && normal_value >= -d2_normal)
                    {
                        value = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
                        d2 = crystal_pitch_Z[e] * sqrt(a[e] * a[e] + b[e] * b[e] + c[e] * c[e]);
        
        
                        if (value < d2 && value >= -d2)
                        {
                            value_cf = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
                            d2_cf = distance_between_array_pixel * sqrt(a_cf[e] * a_cf[e] + b_cf[e] * b_cf[e] + c_cf[e] * c_cf[e]);
        
        
                            if (value_cf >= -d2_cf && value_cf < d2_cf)
                            {
        
                                width = 2 * (crystal_pitch_Z[e] - abs(value));
                                height = 2 * (crystal_pitch_XY[e] - abs(normal_value));
                                distance = d2_cf + abs(value_cf);
                                                              
                               /*
                                face_1_distance_to_center = 1.0f * sqrt(plane_centerA1_A[e] * plane_centerA1_A[e] + plane_centerA1_B[e] * plane_centerA1_B[e] + plane_centerA1_C[e] * plane_centerA1_C[e]);
                                face_2_distance_to_center = 1.0f * sqrt(plane_centerB1_A[e] * plane_centerB1_A[e] + plane_centerB1_B[e] * plane_centerB1_B[e] + plane_centerB1_C[e] * plane_centerB1_C[e]);
                                face_3_distance_to_center = 15.0f * sqrt(plane_centerC1_A[e] * plane_centerC1_A[e] + plane_centerC1_B[e] * plane_centerC1_B[e] + plane_centerC1_C[e] * plane_centerC1_C[e]);
                                
                                p1 = three_plane_intersection(a[e],
                                    b[e], c[e], d[e],
                                    a_normal[e], b_normal[e], c_normal[e],
                                    d_normal[e], plane_centerA1_A[e], plane_centerA1_B[e],
                                    plane_centerA1_C[e], plane_centerA1_D[e]+face_1_distance_to_center);
                                    
                                p2 = three_plane_intersection(a[e],
                                    b[e], c[e], d[e],
                                    a_normal[e], b_normal[e], c_normal[e],
                                    d_normal[e], plane_centerA1_A[e], plane_centerA1_B[e],
                                    plane_centerA1_C[e], plane_centerA1_D[e]-face_1_distance_to_center);                                    
                                               
                                p3 = three_plane_intersection(a[e],
                                    b[e], c[e], d[e],
                                    a_normal[e], b_normal[e], c_normal[e],
                                    d_normal[e], plane_centerB1_A[e], plane_centerB1_B[e],
                                    plane_centerB1_C[e], plane_centerB1_D[e]+face_2_distance_to_center);
                                    
                                p4 = three_plane_intersection(a[e],
                                    b[e], c[e], d[e],
                                    a_normal[e], b_normal[e], c_normal[e],
                                    d_normal[e], plane_centerB1_A[e], plane_centerB1_B[e],
                                    plane_centerB1_C[e], plane_centerB1_D[e]-face_2_distance_to_center);
                                    
                                p5 = three_plane_intersection(a[e],
                                    b[e], c[e], d[e],
                                    a_normal[e], b_normal[e], c_normal[e],
                                    d_normal[e], plane_centerC1_A[e], plane_centerC1_B[e],
                                    plane_centerC1_C[e], plane_centerC1_D[e]+face_3_distance_to_center);
                                    
                                p6 = three_plane_intersection(a[e],
                                    b[e], c[e], d[e],
                                    a_normal[e], b_normal[e], c_normal[e],
                                    d_normal[e], plane_centerC1_A[e], plane_centerC1_B[e],
                                    plane_centerC1_C[e], plane_centerC1_D[e]-face_3_distance_to_center);            
                                    
                                intersection_points[e*6] = point_distance_to_plane(p1, a_cf[e], b_cf[e],c_cf[e], d_cf[e]); 
                                dist_p2 = point_distance_to_plane(p2, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                                
                              
                                                     
                                 printf(" Distance p1: %f", intersection_points[e*6]);
                               
                                  dist_p2 = point_distance_to_plane(p2, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);                     
                                dist_p3 = point_distance_to_plane(p3, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);                     
                                dist_p4 = point_distance_to_plane(p4, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);                     
                                dist_p5 = point_distance_to_plane(p5, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);                     
                                dist_p6 = point_distance_to_plane(p6, a_cf[e], b_cf[e],c_cf[e], d_cf[e]);
                                printf(" Distance p1: %f", dist_p1);
                                printf(" x: %f, y: %f, z: %f ",p1[0], p1[1], p1[2]);
                                printf("----------------");
                                printf(" x_2: %f, y_2: %f, z_2: %f ",p2[0], p2[1], p2[2]);
                              */
                                distance_at = m_values_at[e]*value+b_values_at[e];
                                                                   
                                if(value<=inflex_points_x[2*e])
                                {
                                    distance_crystal = m_values[2*e]*value+b_values[2*e];
                                    distance_at = 0;
                                }
                                 else if(value>=inflex_points_x[2*e+1])
                                {
                                    distance_crystal = m_values[2*e+1]*value+b_values[2*e+1];
                                }
                               
                                else 
                                {
                                    distance_crystal = max_D[e];
                                    
                                    
                                }
                                distance_crystal = max_D[e];
                                idrf = (1-exp(-linear_attenuation_A[e]*distance_crystal))*exp(-linear_attenuation_A[e]*distance_at);
                             
                                
                                
                                solid_angle = 4 * width * height / (2 * distance * sqrt(4 * distance * distance + width * width + height * height));
        
        
                                adjust_coef_shared[i_s] +=  idrf/(sum_vor[e]);
                                 
                                /*
                             else if(value<=inflex_points_x[2*e+1])
                                {
                                    distance_crystal = m_values[2*e+1]*value+b_values[2*e+1];
                                }
                                
                                   printf(" x_2: %f,  distance: %f, att: %f ",idrf,distance_crystal, linear_attenuation_A[e] );
                                 if (sqrt(value*value+normal_value*normal_value+value_cf*value_cf)<=sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))
                                {
                                   adjust_coef_shared[i_s] += time_factor[e]*(1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]);
                                    }
                                  */
        
        
                            }
        
                            /*
                            (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/
                            normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
             d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
                           adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
                          adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
                             */
        
                        }
                    }
        
                }
        
            }
        
            adjust_coef[idt] = adjust_coef_shared[i_s];
            __syncthreads();
        
        }
        __device__ float point_distance_to_plane(float* point, float A, float B, float C, float D)
        {
             float distance;
             distance = abs(A*point[0]+B*point[1]+C*point[2]-D)/sqrt(A*A+B*B+C*C);
             return distance;
        }
        
        __device__  float* three_plane_intersection(float plane1A, float plane1B,
            float plane1C, float plane1D, float plane2A, float plane2B, float plane2C,
            float plane2D, float plane3A, float plane3B, float plane3C, float plane3D)
        {
            float m_det[3][3];
            float m_x[3][3];
            float m_y[3][3];
            float m_z[3][3];
            float det;
            float det_x;
            float det_y;
            float det_z;
            float result[3];
        
            m_det[0][0] = plane1A;
            m_det[0][1] = plane1B;
            m_det[0][2] = plane1C;
            m_det[1][0] = plane2A;
            m_det[1][1] = plane2B;
            m_det[1][2] = plane2C;
            m_det[2][0] = plane3A;
            m_det[2][1] = plane3B;
            m_det[2][2] = plane3C;
        
            m_x[0][0] = plane1D;
            m_x[0][1] = plane1B;
            m_x[0][2] = plane1C;
            m_x[1][0] = plane2D;
            m_x[1][1] = plane2B;
            m_x[1][2] = plane2C;
            m_x[2][0] = plane3D;
            m_x[2][1] = plane3B;
            m_x[2][2] = plane3C;
        
            m_y[0][0] = plane1A;
            m_y[0][1] = plane1D;
            m_y[0][2] = plane1C;
            m_y[1][0] = plane2A;
            m_y[1][1] = plane2D;
            m_y[1][2] = plane2C;
            m_y[2][0] = plane3A;
            m_y[2][1] = plane3D;
            m_y[2][2] = plane3C;
        
            m_z[0][0] = plane1A;
            m_z[0][1] = plane1B;
            m_z[0][2] = plane1D;
            m_z[1][0] = plane2A;
            m_z[1][1] = plane2B;
            m_z[1][2] = plane2D;
            m_z[2][0] = plane3A;
            m_z[2][1] = plane3B;
            m_z[2][2] = plane3D;
        
            det = intersection_determinant(m_det);            
            det_x = intersection_determinant(m_x);
            det_y = intersection_determinant(m_y);
            det_z = intersection_determinant(m_z);
           
           
            if (det != 0.0f)
            {
                result[0] = det_x / det;
                result[1] = det_y / det;
                result[2] = det_z / det;
            }
           
            return result;
        }
        
        __device__ float intersection_determinant(float matrix[3][3])
        {
            float a;
            float b;
            float c;
            float d;
            float e;
            float f;
            float g;
            float h;
            float i;
            float det;
        
            a = matrix[0][0];
            b = matrix[0][1];
            c = matrix[0][2];
            d = matrix[1][0];
            e = matrix[1][1];
            f = matrix[1][2];
            g = matrix[2][0];
            h = matrix[2][1];
            i = matrix[2][2];
        
            det = (a * e * i + b * f * g + c * d * h) - (a * f * h + b * d * i + c * e * g);
            return det;
        }
                                     """)

        mod_backward_projection_shared_mem_cdrf = SourceModule("""
            #include <stdint.h>
            texture<uint8_t, 1> tex;
            
            __global__ void backprojection_cdrf
            (int dataset_number, int n, int m, int p, const float* crystal_pitch_XY, const float* crystal_pitch_Z,
                const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation,
                const int end_event_gpu_limitation, const float* a, const float* a_normal, const float* a_cf, const float* b,
                const float* b_normal, const float* b_cf, const float* c, const float* c_normal, const float* c_cf, const float* d, const float* d_normal,
                const float* d_cf, const short* A, const short* B, const short* C, float* adjust_coef, float* sum_vor,
                char* fov_cut_matrix, float* time_factor, float* m_values, float* m_values_at,
                float* b_values, float* b_values_at, float* max_D, float* inflex_points_x, float* linear_attenuation_A,
                float* linear_attenuation_B)
                       {
                       extern __shared__ float adjust_coef_shared[];
                       int idt = blockIdx.x * blockDim.x + threadIdx.x;


                       float d2;
                       float d2_normal;
                       float d2_cf;
                       float normal_value;
                       float value;
                       float value_cf;
                       short a_temp;
                       short b_temp;
                       short c_temp;
                       char fov_cut_temp;
                       int i_s = threadIdx.x;
                       float width;
                       float height;
                       float distance;
                       float distance_other;
                       float solid_angle;
                       float distance_crystal;
                       float distance_at;
                       float idrf;

                       if (idt >= n * m * p)
                       {
                           return;
                       }

                       __syncthreads();
                       adjust_coef_shared[i_s] = adjust_coef[idt];
                       a_temp = A[idt];
                       b_temp = B[idt];
                       c_temp = C[idt];
                       fov_cut_temp = fov_cut_matrix[idt];



                       for (int e = begin_event_gpu_limitation; e < end_event_gpu_limitation; e++)
                       {
                           if (fov_cut_temp != 0)
                           {
                               normal_value = a_normal[e] * a_temp + b_normal[e] * b_temp + c_normal[e] * c_temp - d_normal[e];
                               d2_normal = crystal_pitch_XY[e] * sqrt(a_normal[e] * a_normal[e] + b_normal[e] * b_normal[e] + c_normal[e] * c_normal[e]);

                               if (normal_value < d2_normal && normal_value >= -d2_normal)
                               {
                                   value = a[e] * a_temp + b[e] * b_temp + c[e] * c_temp - d[e];
                                   d2 = crystal_pitch_Z[e] * sqrt(a[e] * a[e] + b[e] * b[e] + c[e] * c[e]);


                                   if (value < d2 && value >= -d2)
                                   {
                                       value_cf = a_cf[e] * a_temp + b_cf[e] * b_temp + c_cf[e] * c_temp - d_cf[e];
                                       d2_cf = distance_between_array_pixel * sqrt(a_cf[e] * a_cf[e] + b_cf[e] * b_cf[e] + c_cf[e] * c_cf[e]);


                                       if (value_cf >= -d2_cf && value_cf < d2_cf)
                                       {
                                           if (sqrt(value * value + normal_value * normal_value + value_cf * value_cf) <= sqrt(d2 * d2 + d2_normal * d2_normal + d2_cf * d2_cf))
                                           {
                                               width = 2 * (crystal_pitch_Z[e] - abs(value));
                                               height = 2 * (crystal_pitch_XY[e] - abs(normal_value));
                                               distance = d2_cf + abs(value_cf);
                                               distance_other = d2_cf - abs(value_cf);
                                               distance_at = m_values_at[e] * value + b_values_at[e];

                                               if (value <= inflex_points_x[2 * e])
                                               {
                                                   distance_crystal = m_values[2 * e] * value + b_values[2 * e];
                                                   distance_at = 0;
                                               }
                                               else if (value > inflex_points_x[2 * e + 1])
                                               {
                                                   distance_crystal = m_values[2 * e + 1] * value + b_values[2 * e + 1];
                                               }

                                               else
                                               {
                                                   distance_crystal = max_D[e];


                                               }
                                              
                                               idrf = (1 - exp(-linear_attenuation_A[e] * distance_crystal)) * exp(-linear_attenuation_A[e] * distance_at);

                                               if (idrf < 0)
                                               {
                                                   idrf = 0;
                                               }
                                               solid_angle = 4 * width * height / (2 * distance * sqrt(4 * distance * distance + width * width + height * height));

                                                if (sum_vor[e]!=0)
                                                {
                                               adjust_coef_shared[i_s] += idrf / sum_vor[e];
                                                 }


                                               /*
                                               (4 * asin(sin(tan(width/distance))*sin(tan(height/distance))))*(4 * asin(sin(tan(width/distance))*sin(tan(height/distance))))/sum_vor[e];
                                           adjust_coef_shared[i_s] += time_factor[e]*(1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]);

                                               */
                                           }

                                       }

                                       /*
                                       (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/
                                       normal_value= a_normal[e]*a_temp+b_normal[e]*b_temp+c_normal[e] * c_temp-d_normal[e];
                        d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
                                      adjust_coef_shared[i_s] += /(sum_vor[e]*time_factor[e]);
                                     adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[0]*time_factor[e]);
                                        */





                                   }
                               }

                           }

                       }

                       adjust_coef[idt] = adjust_coef_shared[i_s];
                       __syncthreads();
}
                             """)

        # float crystal_pitch, int number_of_events, float *a, float *b, float *c, float *d, int *A, int *B, int *C, float *im, float *vector_matrix
        # Host Code   B, C, im, vector_matrix,
        number_of_datasets = np.int32(1)  # Number of datasets (and concurrent operations) used.
        number_of_datasets_back = np.int32(1)  # Number of datasets (and concurrent operations) used.
        # Start concurrency Test
        # Event as reference point
        ref = cuda.Event()
        ref.record()

        # Create the streams and events needed to calculation
        stream, event = [], []
        marker_names = ['kernel_begin', 'kernel_end']
        # Create List to allocate chunks of data
        A_cut_gpu, B_cut_gpu, C_cut_gpu = [None] * number_of_datasets, [None] * number_of_datasets, [
            None] * number_of_datasets
        A_cut, B_cut, C_cut = [None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets

        a_gpu, b_gpu, c_gpu, d_gpu = [None] * number_of_datasets, [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets
        a_normal_gpu, b_normal_gpu, c_normal_gpu, d_normal_gpu = [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets

        a_cut, b_cut, c_cut, d_cut = [None] * number_of_datasets, [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets

        a_normal_cut, b_normal_cut, c_normal_cut, d_normal_cut = [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets

        a_cf_cut, b_cf_cut, c_cf_cut, d_cf_cut = [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets

        a_cut_gpu, b_cut_gpu, c_cut_gpu, d_cut_gpu = [None] * number_of_datasets, [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets

        a_cut_normal_gpu, b_cut_normal_gpu, c_cut_normal_gpu, d_cut_normal_gpu = [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets

        a_cf_cut_gpu, b_cf_cut_gpu, c_cf_cut_gpu, d_cf_cut_gpu = [None] * number_of_datasets, [
            None] * number_of_datasets, [None] * number_of_datasets, [None] * number_of_datasets

        time_factor_cut_gpu = [None] * number_of_datasets
        sum_vor_gpu = [None] * number_of_datasets
        sum_vor_cut = [None] * number_of_datasets

        adjust_coef_cut = [None] * number_of_datasets
        adjust_coef_gpu = [None] * number_of_datasets
        adjust_coef_pinned = [None] * number_of_datasets_back
        fov_cut_matrix_cutted_gpu = [None] * number_of_datasets_back
        fov_cut_matrix_cut = [None] * number_of_datasets
        # fov_cut_matrix_gpu = [None] * number_of_datasets
        sum_vor_pinned = [None] * number_of_datasets

        distance_to_center_plane_cut = [None] * number_of_datasets
        distance_to_center_plane_gpu_cut = [None] * number_of_datasets
        distance_to_center_plane_normal_cut = [None] * number_of_datasets
        distance_to_center_plane_normal_gpu_cut = [None] * number_of_datasets
        plane_centerA1_cut = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerA1_gpu_cut = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerB1_cut = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerB1_gpu_cut = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerC1_cut = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerC1_gpu_cut = [None] * len(self.crystal_central_planes.plane_centerA1)

        plane_centerA1_gpu = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerB1_gpu = [None] * len(self.crystal_central_planes.plane_centerA1)
        plane_centerC1_gpu = [None] * len(self.crystal_central_planes.plane_centerA1)

        m_values_listMode_cut = [None] * number_of_datasets
        m_values_at_listMode_cut = [None] * number_of_datasets
        b_values_listMode_cut = [None] * number_of_datasets
        b_values_at_listMode_cut = [None] * number_of_datasets
        max_D_listMode_cut = [None] * number_of_datasets
        inflex_points_x_listMode_cut = [None] * number_of_datasets
        linear_attenuation_crystal_A_listMode_cut = [None] * number_of_datasets
        linear_attenuation_crystal_B_listMode_cut = [None] * number_of_datasets
        m_values_gpu_cut = [None] * number_of_datasets
        b_values_gpu_cut = [None] * number_of_datasets
        m_values_at_gpu_cut = [None] * number_of_datasets
        b_values_at_gpu_cut = [None] * number_of_datasets
        max_D_gpu_cut = [None] * number_of_datasets
        inflex_points_x_gpu_cut = [None] * number_of_datasets
        linear_attenuation_A_gpu_cut = [None] * number_of_datasets
        linear_attenuation_B_gpu_cut = [None] * number_of_datasets

        intersection_points = np.ascontiguousarray(np.zeros((number_of_events, 6)), dtype=np.float32)
        intersection_points_gpu = cuda.mem_alloc(intersection_points.size
                                                 * intersection_points.dtype.itemsize)
        cuda.memcpy_htod_async(intersection_points_gpu, intersection_points)

        for i in range(len(self.crystal_central_planes.plane_centerA1)):
            plane_centerA1_cut[i] = [None] * number_of_datasets
            plane_centerA1_gpu_cut[i] = [None] * number_of_datasets
            plane_centerB1_cut[i] = [None] * number_of_datasets
            plane_centerB1_gpu_cut[i] = [None] * number_of_datasets
            plane_centerC1_cut[i] = [None] * number_of_datasets
            plane_centerC1_gpu_cut[i] = [None] * number_of_datasets

            plane_centerA1_gpu[i] = cuda.mem_alloc(self.crystal_central_planes.plane_centerA1[i].size
                                                   * self.crystal_central_planes.plane_centerA1[i].dtype.itemsize)
            plane_centerB1_gpu[i] = cuda.mem_alloc(self.crystal_central_planes.plane_centerB1[i].size
                                                   * self.crystal_central_planes.plane_centerB1[i].dtype.itemsize)
            plane_centerC1_gpu[i] = cuda.mem_alloc(self.crystal_central_planes.plane_centerC1[i].size
                                                   * self.crystal_central_planes.plane_centerC1[i].dtype.itemsize)
            cuda.memcpy_htod_async(plane_centerA1_gpu[i], self.crystal_central_planes.plane_centerA1[i])
            cuda.memcpy_htod_async(plane_centerB1_gpu[i], self.crystal_central_planes.plane_centerB1[i])
            cuda.memcpy_htod_async(plane_centerC1_gpu[i], self.crystal_central_planes.plane_centerC1[i])

        # Streams and Events creation
        for dataset in range(number_of_datasets):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))

        # Foward Projection Memory Allocation
        # Variables that need an unique alocation
        # A_shappened = np.ascontiguousarray(A.reshape(A.shape[0]*A.shape[1]*A.shape[2]), dtype=np.int32)
        # B_shappened = np.ascontiguousarray(B.reshape(B.shape[0]*B.shape[1]*B.shape[2]), dtype=np.int32)
        # C_shappened = np.ascontiguousarray(C.reshape(C.shape[0]*C.shape[1]*C.shape[2]), dtype=np.int32)
        im_shappened = np.ascontiguousarray(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]), dtype=np.float32)
        fov_cut_matrix_shappened = np.ascontiguousarray(
            fov_cut_matrix.reshape(fov_cut_matrix.shape[0] * fov_cut_matrix.shape[1] * fov_cut_matrix.shape[2]),
            dtype=np.byte)

        # forward_projection_arrays_page_locked_memory_allocations = [A_gpu, B_gpu, C_gpu, im_gpu]

        # A_gpu = cuda.mem_alloc(A_shappened.size * A_shappened.dtype.itemsize)
        # B_gpu = cuda.mem_alloc(B_shappened.size * B_shappened.dtype.itemsize)
        # C_gpu = cuda.mem_alloc(C_shappened.size * C_shappened.dtype.itemsize)
        im_gpu = cuda.mem_alloc(im_shappened.size * im_shappened.dtype.itemsize)
        fov_cut_matrix_gpu = cuda.mem_alloc(fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        # texref = mod_forward_projection_shared_mem.get_texref('tex')
        # texref.set_address(fov_cut_matrix_gpu, fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        # # texref.set_format(cuda.array_format.UNSIGNED_INT8, 1)

        # cuda.memcpy_htod_async(A_gpu, A_shappened)
        # cuda.memcpy_htod_async(B_gpu, B_shappened)
        # cuda.memcpy_htod_async(C_gpu, C_shappened)
        cuda.memcpy_htod_async(im_gpu, im_shappened)
        cuda.memcpy_htod_async(fov_cut_matrix_gpu, fov_cut_matrix_shappened)

        a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
        b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
        c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
        d_gpu = cuda.mem_alloc(d.size * d.dtype.itemsize)
        a_normal_gpu = cuda.mem_alloc(a_normal.size * a_normal.dtype.itemsize)
        b_normal_gpu = cuda.mem_alloc(b_normal.size * b_normal.dtype.itemsize)
        c_normal_gpu = cuda.mem_alloc(c_normal.size * c_normal.dtype.itemsize)
        d_normal_gpu = cuda.mem_alloc(d_normal.size * d_normal.dtype.itemsize)

        a_cf_gpu = cuda.mem_alloc(a_cf.size * a_cf.dtype.itemsize)
        b_cf_gpu = cuda.mem_alloc(b_cf.size * b_cf.dtype.itemsize)
        c_cf_gpu = cuda.mem_alloc(c_cf.size * c_cf.dtype.itemsize)
        d_cf_gpu = cuda.mem_alloc(d_cf.size * d_cf.dtype.itemsize)
        sum_vor_t_gpu = cuda.mem_alloc(sum_vor.size * sum_vor.dtype.itemsize)
        distance_to_center_plane_gpu = cuda.mem_alloc(
            self.distance_to_center_plane.size * self.distance_to_center_plane.dtype.itemsize)
        distance_to_center_plane_normal_gpu = cuda.mem_alloc(
            self.distance_to_center_plane_normal.size * self.distance_to_center_plane_normal.dtype.itemsize)

        time_factor_gpu = cuda.mem_alloc(time_factor.size * time_factor.dtype.itemsize)
        # Transfer memory to Optimizer
        cuda.memcpy_htod_async(a_gpu, a)
        cuda.memcpy_htod_async(b_gpu, b)
        cuda.memcpy_htod_async(c_gpu, c)
        cuda.memcpy_htod_async(d_gpu, d)
        cuda.memcpy_htod_async(a_normal_gpu, a_normal)
        cuda.memcpy_htod_async(b_normal_gpu, b_normal)
        cuda.memcpy_htod_async(c_normal_gpu, c_normal)
        cuda.memcpy_htod_async(d_normal_gpu, d_normal)

        cuda.memcpy_htod_async(a_cf_gpu, a_cf)
        cuda.memcpy_htod_async(b_cf_gpu, b_cf)
        cuda.memcpy_htod_async(c_cf_gpu, c_cf)
        cuda.memcpy_htod_async(d_cf_gpu, d_cf)
        cuda.memcpy_htod_async(time_factor_gpu, time_factor)
        cuda.memcpy_htod_async(distance_to_center_plane_gpu, self.distance_to_center_plane)
        cuda.memcpy_htod_async(distance_to_center_plane_normal_gpu, self.distance_to_center_plane_normal)

        m_values_gpu = cuda.mem_alloc(
            self.doi_mapping.m_values_listMode.size * self.doi_mapping.m_values_listMode.dtype.itemsize)
        b_values_gpu = cuda.mem_alloc(
            self.doi_mapping.b_values_listMode.size * self.doi_mapping.b_values_listMode.dtype.itemsize)

        m_values_at_gpu = cuda.mem_alloc(
            self.doi_mapping.m_values_at_listMode.size * self.doi_mapping.m_values_at_listMode.dtype.itemsize)
        b_values_at_gpu = cuda.mem_alloc(
            self.doi_mapping.b_values_at_listMode.size * self.doi_mapping.b_values_at_listMode.dtype.itemsize)

        max_D_gpu = cuda.mem_alloc(
            self.doi_mapping.max_D_listMode.size * self.doi_mapping.max_D_listMode.dtype.itemsize)

        inflex_points_x_gpu = cuda.mem_alloc(
            self.doi_mapping.inflex_points_x_listMode.size * self.doi_mapping.inflex_points_x_listMode.dtype.itemsize)
        linear_attenuation_A_gpu = cuda.mem_alloc(
            self.doi_mapping.linear_attenuation_crystal_A_listMode.size * self.doi_mapping.linear_attenuation_crystal_A_listMode.dtype.itemsize)
        linear_attenuation_B_gpu = cuda.mem_alloc(
            self.doi_mapping.linear_attenuation_crystal_B_listMode.size * self.doi_mapping.linear_attenuation_crystal_B_listMode.dtype.itemsize)

        cuda.memcpy_htod_async(m_values_gpu, self.doi_mapping.m_values_listMode)
        cuda.memcpy_htod_async(b_values_gpu, self.doi_mapping.b_values_listMode)
        cuda.memcpy_htod_async(m_values_at_gpu, self.doi_mapping.m_values_at_listMode)
        cuda.memcpy_htod_async(b_values_at_gpu, self.doi_mapping.b_values_at_listMode)
        cuda.memcpy_htod_async(max_D_gpu, self.doi_mapping.max_D_listMode)
        cuda.memcpy_htod_async(inflex_points_x_gpu, self.doi_mapping.inflex_points_x_listMode)
        cuda.memcpy_htod_async(linear_attenuation_A_gpu, self.doi_mapping.linear_attenuation_crystal_A_listMode)
        cuda.memcpy_htod_async(linear_attenuation_B_gpu, self.doi_mapping.linear_attenuation_crystal_B_listMode)

        for dataset in range(number_of_datasets):
            # if dataset == number_of_datasets:
            #     begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
            #     end_dataset = number_of_events
            #     adjust_coef_cut[dataset] = np.ascontiguousarray(
            #         adjust_coef[int(np.floor(im_cut_dim[0] * dataset)):adjust_coef.shape[0],
            #         int(np.floor(im_cut_dim[1] * dataset)):adjust_coef.shape[1],
            #         int(np.floor(im_cut_dim[2] * dataset)):adjust_coef.shape[2]],
            #         dtype=np.float32)
            #     fov_cut_matrix_cut[dataset] = np.ascontiguousarray(
            #         fov_cut_matrix[int(np.floor(im_cut_dim[0] * dataset)):fov_cut_matrix.shape[0],
            #         int(np.floor(im_cut_dim[1] * dataset)):fov_cut_matrix.shape[1],
            #         int(np.floor(im_cut_dim[2] * dataset)):fov_cut_matrix.shape[2]],
            #         dtype=np.float32)
            # else:
            begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
            end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)

            # Cutting dataset
            # For forward projection the data is cutted by number of events. For backprojection is cutted per image pieces of image
            a_cut[dataset] = a[begin_dataset:end_dataset]
            b_cut[dataset] = b[begin_dataset:end_dataset]
            c_cut[dataset] = c[begin_dataset:end_dataset]
            d_cut[dataset] = d[begin_dataset:end_dataset]
            a_normal_cut[dataset] = a_normal[begin_dataset:end_dataset]
            b_normal_cut[dataset] = b_normal[begin_dataset:end_dataset]
            c_normal_cut[dataset] = c_normal[begin_dataset:end_dataset]
            d_normal_cut[dataset] = d_normal[begin_dataset:end_dataset]

            a_cf_cut[dataset] = a_cf[begin_dataset:end_dataset]
            b_cf_cut[dataset] = b_cf[begin_dataset:end_dataset]
            c_cf_cut[dataset] = c_cf[begin_dataset:end_dataset]
            d_cf_cut[dataset] = d_cf[begin_dataset:end_dataset]
            sum_vor_cut[dataset] = sum_vor[begin_dataset:end_dataset]
            distance_to_center_plane_cut[dataset] = self.distance_to_center_plane[begin_dataset:end_dataset]
            distance_to_center_plane_normal_cut[dataset] = self.distance_to_center_plane_normal[
                                                           begin_dataset:end_dataset]

            # Forward
            a_cut_gpu[dataset] = cuda.mem_alloc(a_cut[dataset].size * a_cut[dataset].dtype.itemsize)
            b_cut_gpu[dataset] = cuda.mem_alloc(b_cut[dataset].size * b_cut[dataset].dtype.itemsize)
            c_cut_gpu[dataset] = cuda.mem_alloc(c_cut[dataset].size * c_cut[dataset].dtype.itemsize)
            d_cut_gpu[dataset] = cuda.mem_alloc(d_cut[dataset].size * d_cut[dataset].dtype.itemsize)
            a_cut_normal_gpu[dataset] = cuda.mem_alloc(
                a_normal_cut[dataset].size * a_normal_cut[dataset].dtype.itemsize)
            b_cut_normal_gpu[dataset] = cuda.mem_alloc(
                b_normal_cut[dataset].size * b_normal_cut[dataset].dtype.itemsize)
            c_cut_normal_gpu[dataset] = cuda.mem_alloc(
                c_normal_cut[dataset].size * c_normal_cut[dataset].dtype.itemsize)
            d_cut_normal_gpu[dataset] = cuda.mem_alloc(
                d_normal_cut[dataset].size * d_normal_cut[dataset].dtype.itemsize)

            a_cf_cut_gpu[dataset] = cuda.mem_alloc(a_cf_cut[dataset].size * a_cf_cut[dataset].dtype.itemsize)
            b_cf_cut_gpu[dataset] = cuda.mem_alloc(b_cf_cut[dataset].size * b_cf_cut[dataset].dtype.itemsize)
            c_cf_cut_gpu[dataset] = cuda.mem_alloc(c_cf_cut[dataset].size * c_cf_cut[dataset].dtype.itemsize)
            d_cf_cut_gpu[dataset] = cuda.mem_alloc(d_cf_cut[dataset].size * d_cf_cut[dataset].dtype.itemsize)

            distance_to_center_plane_gpu_cut[dataset] = cuda.mem_alloc(
                distance_to_center_plane_cut[dataset].size * distance_to_center_plane_cut[dataset].dtype.itemsize)
            distance_to_center_plane_normal_gpu_cut[dataset] = cuda.mem_alloc(
                distance_to_center_plane_normal_cut[dataset].size * distance_to_center_plane_normal_cut[
                    dataset].dtype.itemsize)

            sum_vor_gpu[dataset] = cuda.mem_alloc(sum_vor_cut[dataset].size * sum_vor_cut[dataset].dtype.itemsize)

            sum_vor_pinned[dataset] = cuda.register_host_memory(sum_vor_cut[dataset])
            assert np.all(sum_vor_pinned[dataset] == sum_vor_cut[dataset])

            cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_pinned[dataset], stream[dataset])
            # sum_vor_gpu[dataset] = np.intp(x.base.get_device_pointer())
            # cuda.memcpy_htod_async(probability_gpu[dataset], probability_cut[dataset])
            cuda.memcpy_htod_async(a_cut_gpu[dataset], a_cut[dataset])
            cuda.memcpy_htod_async(b_cut_gpu[dataset], b_cut[dataset])
            cuda.memcpy_htod_async(c_cut_gpu[dataset], c_cut[dataset])
            cuda.memcpy_htod_async(d_cut_gpu[dataset], d_cut[dataset])
            cuda.memcpy_htod_async(a_cut_normal_gpu[dataset], a_normal_cut[dataset])
            cuda.memcpy_htod_async(b_cut_normal_gpu[dataset], b_normal_cut[dataset])
            cuda.memcpy_htod_async(c_cut_normal_gpu[dataset], c_normal_cut[dataset])
            cuda.memcpy_htod_async(d_cut_normal_gpu[dataset], d_normal_cut[dataset])

            cuda.memcpy_htod_async(a_cf_cut_gpu[dataset], a_cf_cut[dataset])
            cuda.memcpy_htod_async(b_cf_cut_gpu[dataset], b_cf_cut[dataset])
            cuda.memcpy_htod_async(c_cf_cut_gpu[dataset], c_cf_cut[dataset])
            cuda.memcpy_htod_async(d_cf_cut_gpu[dataset], d_cf_cut[dataset])
            cuda.memcpy_htod_async(distance_to_center_plane_gpu_cut[dataset], distance_to_center_plane_cut[dataset])
            cuda.memcpy_htod_async(distance_to_center_plane_normal_gpu_cut[dataset],
                                   distance_to_center_plane_normal_cut[dataset])

            m_values_listMode_cut[dataset] = np.copy(self.doi_mapping.m_values_listMode[begin_dataset:end_dataset])
            m_values_at_listMode_cut[dataset] = np.copy(
                self.doi_mapping.m_values_at_listMode[begin_dataset:end_dataset])
            b_values_listMode_cut[dataset] = np.copy(self.doi_mapping.b_values_listMode[begin_dataset:end_dataset])
            b_values_at_listMode_cut[dataset] = np.copy(
                self.doi_mapping.b_values_at_listMode[begin_dataset:end_dataset])
            max_D_listMode_cut[dataset] = np.copy(self.doi_mapping.max_D_listMode[begin_dataset:end_dataset])
            inflex_points_x_listMode_cut[dataset] = np.copy(
                self.doi_mapping.inflex_points_x_listMode[begin_dataset:end_dataset])
            linear_attenuation_crystal_A_listMode_cut[dataset] = np.copy(
                self.doi_mapping.linear_attenuation_crystal_A_listMode[begin_dataset:end_dataset])
            linear_attenuation_crystal_B_listMode_cut[dataset] = np.copy(
                self.doi_mapping.linear_attenuation_crystal_B_listMode[begin_dataset:end_dataset])

            m_values_gpu_cut[dataset] = cuda.mem_alloc(
                m_values_listMode_cut[dataset].size * m_values_listMode_cut[dataset].dtype.itemsize)

            b_values_gpu_cut[dataset] = cuda.mem_alloc(
                b_values_listMode_cut[dataset].size * b_values_listMode_cut[dataset].dtype.itemsize)

            m_values_at_gpu_cut[dataset] = cuda.mem_alloc(
                m_values_at_listMode_cut[dataset].size * m_values_at_listMode_cut[dataset].dtype.itemsize)

            b_values_at_gpu_cut[dataset] = cuda.mem_alloc(
                b_values_at_listMode_cut[dataset].size * b_values_at_listMode_cut[dataset].dtype.itemsize)

            max_D_gpu_cut[dataset] = cuda.mem_alloc(
                max_D_listMode_cut[dataset].size * max_D_listMode_cut[dataset].dtype.itemsize)

            inflex_points_x_gpu_cut[dataset] = cuda.mem_alloc(
                inflex_points_x_listMode_cut[dataset].size * inflex_points_x_listMode_cut[dataset].dtype.itemsize)

            linear_attenuation_A_gpu_cut[dataset] = cuda.mem_alloc(
                linear_attenuation_crystal_A_listMode_cut[dataset].size * linear_attenuation_crystal_A_listMode_cut[
                    dataset].dtype.itemsize)
            linear_attenuation_B_gpu_cut[dataset] = cuda.mem_alloc(
                linear_attenuation_crystal_B_listMode_cut[dataset].size * linear_attenuation_crystal_B_listMode_cut[
                    dataset].dtype.itemsize)

            cuda.memcpy_htod_async(m_values_gpu_cut[dataset], m_values_listMode_cut[dataset])
            cuda.memcpy_htod_async(b_values_gpu_cut[dataset], b_values_listMode_cut[dataset])
            cuda.memcpy_htod_async(m_values_at_gpu_cut[dataset], m_values_at_listMode_cut[dataset])
            cuda.memcpy_htod_async(b_values_at_gpu_cut[dataset], b_values_at_listMode_cut[dataset])
            cuda.memcpy_htod_async(max_D_gpu_cut[dataset], max_D_listMode_cut[dataset])
            cuda.memcpy_htod_async(inflex_points_x_gpu_cut[dataset], inflex_points_x_listMode_cut[dataset])
            cuda.memcpy_htod_async(linear_attenuation_A_gpu_cut[dataset],
                                   linear_attenuation_crystal_A_listMode_cut[dataset])
            cuda.memcpy_htod_async(linear_attenuation_B_gpu_cut[dataset],
                                   linear_attenuation_crystal_B_listMode_cut[dataset])

        adjust_coef = np.ascontiguousarray(adjust_coef.reshape(
            adjust_coef.shape[0] * adjust_coef.shape[1] * adjust_coef.shape[2]),
            dtype=np.float32)
        # fov_cut_matrix = np.ascontiguousarray(fov_cut_matrix.reshape(
        #     fov_cut_matrix.shape[0] * fov_cut_matrix.shape[1] * fov_cut_matrix.shape[2]),
        #     dtype=np.float32)
        A = np.ascontiguousarray(A.reshape(
            A.shape[0] * A.shape[1] * A.shape[2]),
            dtype=np.short)
        B = np.ascontiguousarray(B.reshape(
            B.shape[0] * B.shape[1] * B.shape[2]),
            dtype=np.short)
        C = np.ascontiguousarray(C.reshape(
            C.shape[0] * C.shape[1] * C.shape[2]),
            dtype=np.short)

        # ---- Divide into datasets variables backprojection
        for dataset in range(number_of_datasets_back):
            voxels_division = adjust_coef.shape[0] // number_of_datasets_back
            adjust_coef_cut[dataset] = np.ascontiguousarray(
                adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division],
                dtype=np.float32)

            fov_cut_matrix_cut[dataset] = np.ascontiguousarray(
                fov_cut_matrix_shappened[dataset * voxels_division:(dataset + 1) * voxels_division],
                dtype=np.byte)

            A_cut[dataset] = np.ascontiguousarray(
                A[dataset * voxels_division:(dataset + 1) * voxels_division],
                dtype=np.short)

            B_cut[dataset] = np.ascontiguousarray(
                B[dataset * voxels_division:(dataset + 1) * voxels_division],
                dtype=np.short)

            C_cut[dataset] = np.ascontiguousarray(
                C[dataset * voxels_division:(dataset + 1) * voxels_division],
                dtype=np.short)
            # Backprojection
            adjust_coef_gpu[dataset] = cuda.mem_alloc(
                adjust_coef_cut[dataset].size * adjust_coef_cut[dataset].dtype.itemsize)

            adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
            assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
            cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])

            fov_cut_matrix_cutted_gpu[dataset] = cuda.mem_alloc(
                fov_cut_matrix_cut[dataset].size * fov_cut_matrix_cut[dataset].dtype.itemsize)

            A_cut_gpu[dataset] = cuda.mem_alloc(
                A_cut[dataset].size * A_cut[dataset].dtype.itemsize)

            B_cut_gpu[dataset] = cuda.mem_alloc(
                B_cut[dataset].size * B_cut[dataset].dtype.itemsize)

            C_cut_gpu[dataset] = cuda.mem_alloc(
                C_cut[dataset].size * C_cut[dataset].dtype.itemsize)

            cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset])
            cuda.memcpy_htod_async(fov_cut_matrix_cutted_gpu[dataset], fov_cut_matrix_cut[dataset])
            cuda.memcpy_htod_async(A_cut_gpu[dataset], A_cut[dataset])
            cuda.memcpy_htod_async(B_cut_gpu[dataset], B_cut[dataset])
            cuda.memcpy_htod_async(C_cut_gpu[dataset], C_cut[dataset])

        free, total = cuda.mem_get_info()

        print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))

        # -------------OSEM---------
        it = self.number_of_iterations
        subsets = self.number_of_subsets
        print('Number events for reconstruction: {}'.format(number_of_events))

        im = np.ascontiguousarray(im.reshape(im.shape[0] * im.shape[1] * im.shape[2]), dtype=np.float32)
        for i in range(it):
            print('Iteration number: {}\n----------------'.format(i + 1))
            begin_event = np.int32(0)
            end_event = np.int32(number_of_events / subsets)
            for sb in range(subsets):
                print('Subset number: {}'.format(sb))
                number_of_events_subset = np.int32(end_event - begin_event)
                tic = time.time()
                # Cycle forward Projection
                for dataset in range(number_of_datasets):

                    if dataset == number_of_datasets:
                        begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
                        end_dataset = number_of_events_subset
                    else:
                        begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
                        end_dataset = np.int32((dataset + 1) * number_of_events_subset / number_of_datasets)

                    threadsperblock = (256, 1, 1)
                    blockspergrid_x = int(math.ceil(((end_dataset - begin_dataset)) / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    event[dataset]['kernel_begin'].record(stream[dataset])
                    # depth = np.int32(5)
                    # weight = np.int32(A.shape[2]/2)
                    # height = np.int32(A.shape[2]/2)
                    if self.cdrf:
                        func_forward = mod_forward_projection_shared_mem_cdrf.get_function("forward_projection_cdrf")
                        func_forward(weight, height, depth, start_x, start_y, start_z,
                                     distance_to_center_plane_normal_gpu_cut[dataset],
                                     distance_to_center_plane_gpu_cut[dataset],
                                     half_distance_between_array_pixel,
                                     number_of_events, begin_dataset, end_dataset, a_cut_gpu[dataset],
                                     a_cut_normal_gpu[dataset], a_cf_cut_gpu[dataset],
                                     b_cut_gpu[dataset], b_cut_normal_gpu[dataset], b_cf_cut_gpu[dataset],
                                     c_cut_gpu[dataset], c_cut_normal_gpu[dataset],
                                     c_cf_cut_gpu[dataset],
                                     d_cut_gpu[dataset],
                                     d_cut_normal_gpu[dataset], d_cf_cut_gpu[dataset],
                                     sum_vor_gpu[dataset], fov_cut_matrix_gpu, im_gpu, m_values_gpu_cut[dataset],
                                     b_values_gpu_cut[dataset], m_values_at_gpu_cut[dataset],
                                     b_values_at_gpu_cut[dataset],
                                     max_D_gpu_cut[dataset], inflex_points_x_gpu_cut[dataset],
                                     linear_attenuation_A_gpu_cut[dataset],
                                     linear_attenuation_B_gpu_cut[dataset],
                                     block=threadsperblock,
                                     grid=blockspergrid,
                                     stream=stream[dataset])
                    else:
                        func_forward = mod_forward_projection_shared_mem.get_function("forward_projection")
                        func_forward(weight, height, depth, start_x, start_y, start_z, half_crystal_pitch_xy,
                                     half_crystal_pitch_z,
                                     half_distance_between_array_pixel,
                                     number_of_events, begin_dataset, end_dataset, a_cut_gpu[dataset],
                                     a_cut_normal_gpu[dataset], a_cf_cut_gpu[dataset],
                                     b_cut_gpu[dataset], b_cut_normal_gpu[dataset], b_cf_cut_gpu[dataset],
                                     c_cut_gpu[dataset], c_cut_normal_gpu[dataset],
                                     c_cf_cut_gpu[dataset],
                                     d_cut_gpu[dataset],
                                     d_cut_normal_gpu[dataset], d_cf_cut_gpu[dataset],
                                     sum_vor_gpu[dataset], fov_cut_matrix_gpu, im_gpu,
                                     block=threadsperblock,
                                     grid=blockspergrid,
                                     stream=stream[dataset])

                # Sincronization of streams
                for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                # Transfering data from Optimizer
                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
                    # cuda.memcpy_dtoh_async(sum_vor[begin_dataset:end_dataset], sum_vor_gpu[dataset])
                    cuda.memcpy_dtoh_async(sum_vor_pinned[dataset], sum_vor_gpu[dataset], stream[dataset])
                    # cuda.cudaStream.Synchronize(stream[dataset])

                    toc = time.time()

                cuda.Context.synchronize()

                print('Time part Forward Projection {} : {}'.format(1, toc - tic))
                # number_of_datasets = np.int32(2)
                teste = np.copy(sum_vor)
                # sum_vor[sum_vor<1]=0
                sum_vor = np.ascontiguousarray(teste, dtype=np.float32)
                # sum_vor=np.ascontiguousarray(np.ones((self.a.shape)), dtype=np.float32)
                print('SUM VOR: {}'.format(np.sum(teste)))
                print('SUM VOR: {}'.format(teste))
                # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)

                cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)

                # ------------BACKPROJECTION-----------

                for dataset in range(number_of_datasets_back):
                    dataset = np.int32(dataset)
                    begin_dataset = np.int32(0)
                    end_dataset = np.int32(number_of_events_subset)

                    # begin_dataset = np.int32(0)
                    # end_dataset = np.int32(number_of_events)

                    event[dataset]['kernel_begin'].record(stream[dataset])
                    # weight_cutted, height_cutted, depth_cutted = np.int32(adjust_coef_cut[dataset].shape[0]), np.int32(
                    #     adjust_coef_cut[dataset].shape[1]), np.int32(adjust_coef_cut[dataset].shape[2])
                    weight_cutted, height_cutted, depth_cutted = np.int32(adjust_coef_cut[dataset].shape[0]), np.int32(
                        1), np.int32(1)

                    number_of_voxels_thread = 64
                    threadsperblock = (np.int(number_of_voxels_thread), 1, 1)
                    blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    # blockspergrid_y = int(math.ceil(adjust_coef_cut[dataset].shape[1] / threadsperblock[1]))
                    # blockspergrid_z = int(math.ceil(adjust_coef_cut[dataset].shape[2] / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    shared_memory = threadsperblock[0] * threadsperblock[1] * threadsperblock[2] * 4
                    if self.cdrf:
                        if self.normalization_calculation_flag:
                            func_backward = mod_normalization_shared_mem_cdrf.get_function("normalization_cdrf")
                            func_backward(dataset, weight_cutted, height_cutted, depth_cutted,
                                          distance_to_center_plane_normal_gpu,
                                          distance_to_center_plane_gpu, half_distance_between_array_pixel,
                                          number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
                                          b_gpu, b_normal_gpu, b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
                                          d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset],
                                          C_cut_gpu[dataset],
                                          adjust_coef_gpu[dataset],
                                          sum_vor_t_gpu, fov_cut_matrix_cutted_gpu[dataset], time_factor_gpu,
                                          plane_centerA1_gpu[0], plane_centerA1_gpu[1], plane_centerA1_gpu[2],
                                          plane_centerA1_gpu[3],
                                          plane_centerB1_gpu[0], plane_centerB1_gpu[1], plane_centerB1_gpu[2],
                                          plane_centerB1_gpu[3],
                                          plane_centerC1_gpu[0], plane_centerC1_gpu[1], plane_centerC1_gpu[2],
                                          plane_centerC1_gpu[3], intersection_points_gpu, m_values_gpu, m_values_at_gpu,
                                          b_values_gpu, b_values_at_gpu,
                                          max_D_gpu, inflex_points_x_gpu, linear_attenuation_A_gpu,
                                          linear_attenuation_B_gpu,
                                          block=threadsperblock,
                                          grid=blockspergrid,
                                          shared=np.int(4 * number_of_voxels_thread),
                                          stream=stream[dataset],
                                          )


                        else:
                            func_backward = mod_backward_projection_shared_mem_cdrf.get_function("backprojection_cdrf")
                            func_backward(dataset, weight_cutted, height_cutted, depth_cutted,
                                          distance_to_center_plane_normal_gpu,
                                          distance_to_center_plane_gpu, half_distance_between_array_pixel,
                                          number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
                                          b_gpu, b_normal_gpu, b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
                                          d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset],
                                          C_cut_gpu[dataset],
                                          adjust_coef_gpu[dataset],
                                          sum_vor_t_gpu, fov_cut_matrix_cutted_gpu[dataset], time_factor_gpu,
                                          m_values_gpu, m_values_at_gpu,
                                          b_values_gpu, b_values_at_gpu,
                                          max_D_gpu, inflex_points_x_gpu, linear_attenuation_A_gpu,
                                          linear_attenuation_B_gpu,
                                          block=threadsperblock,
                                          grid=blockspergrid,
                                          shared=np.int(4 * number_of_voxels_thread),
                                          stream=stream[dataset],
                                          )

                    else:
                        if self.normalization_calculation_flag:
                            func_backward = mod_normalization_shared_mem.get_function("normalization")


                        else:
                            func_backward = mod_backward_projection_shared_mem.get_function("backprojection")

                        func_backward(dataset, weight_cutted, height_cutted, depth_cutted, half_crystal_pitch_xy,
                                      half_crystal_pitch_z, half_distance_between_array_pixel,
                                      number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
                                      b_gpu, b_normal_gpu, b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
                                      d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset],
                                      C_cut_gpu[dataset],
                                      adjust_coef_gpu[dataset],
                                      sum_vor_t_gpu, fov_cut_matrix_cutted_gpu[dataset], time_factor_gpu,
                                      block=threadsperblock,
                                      grid=blockspergrid,
                                      shared=np.int(4 * number_of_voxels_thread),
                                      stream=stream[dataset],
                                      )

                for dataset in range(number_of_datasets_back):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                for dataset in range(number_of_datasets_back):
                    cuda.memcpy_dtoh_async(adjust_coef_cut[dataset], adjust_coef_gpu[dataset])
                    adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division] = adjust_coef_cut[dataset]

                cuda.Context.synchronize()
                print('Time part Backward Projection {} : {}'.format(1, time.time() - toc))

                # Image Normalization
                # if i ==0:
                #     norm_im=np.copy(adjust_coef)
                #     norm_im=norm_im/np.max(norm_im)
                #     norm_im[norm_im == 0] = np.min(norm_im[np.nonzero(norm_im)])
                # normalization_matrix = gaussian_filter(normalization_matrix, 0.5)

                # im_med = np.load("C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\PhD\\Reconstrução\\NAF+FDG\\Easypet Scan 05 Aug 2019 - 14h 36m 33s\\static_image\\Easypet Scan 05 Aug 2019 - 14h 36m 33s mlem.npy")
                # self.algorithm = "LM-MRP"
                if self.algorithm == "LM-MRP":
                    beta = self.algorithm_options[0]
                    kernel_filter_size = self.algorithm_options[1]
                    im_to_filter = im.reshape(weight, height, depth)
                    im_med = median_filter(im_to_filter, kernel_filter_size)
                    penalized_term = np.copy(im_to_filter)
                    penalized_term[im_med != 0] = 1 + beta * (im_to_filter[im_med != 0] - im_med[im_med != 0]) / im_med[
                        im_med != 0]
                    penalized_term = np.ascontiguousarray(penalized_term.reshape(weight * height * depth),
                                                          dtype=np.float32)
                    # penalized_term = np.ascontiguousarray(penalized_term, dtype=np.float32)

                if self.algorithm == "MAP":
                    beta = 0.5
                    im_map = np.load(
                        "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\PhD\\Reconstrução\\NAF+FDG\\Easypet Scan 05 Aug 2019 - 14h 36m 33s\\static_image\\Easypet Scan 05 Aug 2019 - 14h 36m 33s mlem.npy")

                im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
                    normalization_matrix != 0] / (normalization_matrix[normalization_matrix != 0])
                im[normalization_matrix == 0] = 0
                if self.algorithm == "LM-MRP":
                    im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
                # im = fourier_gaussian(im, sigma=0.2)
                # im = gaussian_filter(im, 0.4)
                print('SUM IMAGE: {}'.format(np.sum(im)))
                im = np.ascontiguousarray(im, dtype=np.float32)
                # im = im * adjust_coef / sensivity_matrix[np.nonzero(sensivity_matrix)]
                cuda.memcpy_htod_async(im_gpu, im)

                # Clearing variables
                sum_vor = np.ascontiguousarray(
                    np.zeros(self.a.shape, dtype=np.float32))

                adjust_coef = np.ascontiguousarray(
                    np.zeros((self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z),
                             dtype=np.float32))

                for dataset in range(number_of_datasets):
                    # if dataset == number_of_datasets:
                    #     begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                    #     end_dataset = number_of_events
                    #
                    # else:
                    begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
                    # adjust_coef_cut[dataset] = np.ascontiguousarray(adjust_coef[:, :,
                    #                                                 int(np.floor(im_cut_dim[2] * dataset)):int(
                    #                                                     np.floor(im_cut_dim[2] * (dataset + 1)))],
                    #                                                 dtype=np.float32)

                    sum_vor_cut[dataset] = sum_vor[begin_dataset:end_dataset]
                    # cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_cut[dataset])
                    sum_vor_pinned[dataset] = cuda.register_host_memory(sum_vor_cut[dataset])
                    assert np.all(sum_vor_pinned[dataset] == sum_vor_cut[dataset])
                    cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_pinned[dataset], stream[dataset])

                for dataset in range(number_of_datasets_back):
                    adjust_coef_cut[dataset] = adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division]
                    adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
                    assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
                    cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])

                if self.saved_image_by_iteration:
                    im = im.reshape(weight, height, depth)
                    self._save_image_by_it(im, i, sb)

                if self.signals_interface is not None:
                    self.signals_interface.trigger_update_label_reconstruction_status.emit(
                        "{}: Iteration {}".format(self.current_info_step, i + 1))
                    self.signals_interface.trigger_progress_reconstruction_partial.emit(
                        int(np.round(100 * (i + 1) * (sb + subsets) / (it * subsets), 0)))

        im = im.reshape(weight, height, depth)
        return im * subsets

    def _save_image_by_it(self, im, normalization=False, it=None, sb=None):
        directory = os.path.dirname(os.path.abspath(__file__))
        if normalization:
            file_name = os.path.join(self.directory, "Normalization".format(it, sb))
        else:
            file_name = os.path.join(self.directory, "EasyPETScan_it{}_sb{}".format(it, sb))
        print(file_name)
        volume = im.astype(np.float32)
        length = 1
        for i in volume.shape:
            length *= i
        # length = volume.shape[0] * volume.shape[2] * volume.shape[1]
        if len(volume.shape) > 1:
            data = np.reshape(volume, [1, length], order='F')
        else:
            data = volume
        shapeIm = volume.shape

        output_file = open(file_name, 'wb')
        arr = array('f', data[0])

        arr.tofile(output_file)
        output_file.close()
        # file_name = directory+"/Acquisitions/SENS_0_5"
        # file_name = directory+"/Acquisitions/sens_0_25"
