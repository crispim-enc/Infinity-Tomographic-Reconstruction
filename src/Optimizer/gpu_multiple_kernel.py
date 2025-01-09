import numpy as np
from scipy.ndimage import median_filter
from pycuda.compiler import SourceModule
import math
import time
import os
from array import array
import gc
from .GaussianFileGenarator import GaussianParameters


class GPUSharedMemoryMultipleKernel:
    def __init__(self, EM_obj=None, optimize_reads_and_calcs=False):
        self.optimize_reads_and_calcs = optimize_reads_and_calcs
        self.cuda_drv = EM_obj.cuda_drv
        self.number_of_iterations = EM_obj.number_of_iterations
        self.number_of_subsets = EM_obj.number_of_subsets
        self.mod_forward_projection_shared_mem = None
        self.mod_backward_projection_shared_mem = None
        self.directory = EM_obj.directory
        self.saved_image_by_iteration = EM_obj.saved_image_by_iteration
        if self.saved_image_by_iteration:
            self.iterations_path = os.path.join(self.directory,"iterations")
            if not os.path.isdir(self.iterations_path):
                os.makedirs(self.iterations_path)
        self.a = EM_obj.a
        self.a_normal = EM_obj.a_normal
        self.a_cf = EM_obj.a_cf

        self.b = EM_obj.b
        self.b_normal = EM_obj.b_normal
        self.b_cf = EM_obj.b_cf

        self.c = EM_obj.c
        self.c_normal = EM_obj.c_normal
        self.c_cf = EM_obj.c_cf

        self.d = EM_obj.d
        self.d_normal = EM_obj.d_normal
        self.d_cf = EM_obj.d_cf

        self.A = EM_obj.im_index_x
        self.B = EM_obj.im_index_y
        self.C = EM_obj.im_index_z
        self.sum_vor = EM_obj.sum_pixel

        self.number_of_pixels_x = EM_obj.number_of_pixels_x
        self.number_of_pixels_y = EM_obj.number_of_pixels_y
        self.number_of_pixels_z = EM_obj.number_of_pixels_z

        self.adjust_coef = EM_obj.adjust_coef
        self.im = EM_obj.im
        self.fov_matrix_cut = EM_obj.fov_matrix_cut
        self.half_crystal_pitch_xy = EM_obj.half_crystal_pitch_xy
        self.half_crystal_pitch_z = EM_obj.half_crystal_pitch_z
        self.distance_between_array_pixel = EM_obj.distance_between_array_pixel
        self.distance_to_center_plane = EM_obj.distance_to_center_plane
        self.distance_to_center_plane_normal = EM_obj.distance_to_center_plane_normal
        self.time_factor = EM_obj.time_correction
        self.normalization_matrix = EM_obj.normalization_matrix
        self.algorithm = EM_obj.algorithm
        self.algorithm_options = EM_obj.algorithm_options
        self.normalizationFlag = EM_obj.normalization_calculation_flag
        self.number_of_events = np.int32(len(self.a))
        self.weight = np.int32(self.A.shape[0])
        self.height = np.int32(self.A.shape[1])
        self.depth = np.int32(self.A.shape[2])

        # Optimizer corrections
        self.projector_type = EM_obj.projector_type
        self.doi_correction = EM_obj.doi_correction
        self.decay_correction = EM_obj.decay_correction
        self.random_correction = EM_obj.random_correction
        self.scatter_correction = EM_obj.scatter_correction
        self.scatter_angle_correction = EM_obj.scatter_angle_correction

        self.x_min_f = EM_obj.x_min_f
        self.x_max_f = EM_obj.x_max_f
        self.y_min_f = EM_obj.y_min_f
        self.y_max_f = EM_obj.y_max_f
        self.z_min_f = EM_obj.z_min_f
        self.z_max_f = EM_obj.z_max_f

        self.voxelSize = EM_obj.voxelSize


        # self.projector_type_args = [True]
        # if self.projector_type == "Solid-Angle":
        #     self.small_angle_approximation = self.projector_type_args[0]

        self.file_to_open = {
                             1: {"Memory Optimized": False,
                                 "Projector_type": "Solid-Angle small approximation",
                                 "DOI": None,
                                 "Decay": False,
                                 "Random": False,
                                 "Scatter": False,
                                 "Scatter Angle": False,
                                 "File Folder": "Solid Angle",
                                 "Filename Forward": "fw_mk_sasaa.c",
                                 "Filename Back": "bk_mk_sasaa.c"
                                 },


                             2: {"Memory Optimized": False,
                                 "Projector_type": "Box Counts",
                                 "DOI": None,
                                 "Decay": False,
                                 "Random": False,
                                 "Scatter": False,
                                 "Scatter Angle": False,
                                 "File Folder": "Box Counts",
                                 "Filename Forward": "fw_mk_cb.c",
                                 "Filename Back": "bk_mk_cb.c"
                                 },

                             3: {"Memory Optimized": False,
                                 "Projector_type": "Orthogonal Projector",
                                 "DOI": None,
                                 "Decay": False,
                                 "Random": False,
                                 "Scatter": False,
                                 "Scatter Angle": False,
                                 "File Folder": "Orthogonal Projector",
                                 "Filename Forward": "fw_mk_gp.c",
                                 "Filename Back": "bk_mk_gp.c"
                                 },


                             4: {"Memory Optimized": False,
                                  "Projector_type": "Constant gaussian",
                                  "DOI": None,
                                  "Decay": False,
                                  "Random": False,
                                  "Scatter": False,
                                  "Scatter Angle": False,
                                  "File Folder": "Gaussian",
                                  "Filename Forward": "fw_mk_gaussian_fast.c",
                                  "Filename Back": "bk_mk_gaussian_fast.c"
                                  },

                            5:  {"Memory Optimized": False,
                                  "Projector_type": "Variable gaussian",
                                  "DOI": None,
                                  "Decay": False,
                                  "Random": False,
                                  "Scatter": False,
                                  "Scatter Angle": False,
                                  "File Folder": "Gaussian",
                                  "Filename Forward": "fw_mk_variable_gaussian_fast.c",
                                  "Filename Back": "bk_mk_variable_gaussian_fast.c"
                                  },

                             }


    def _load_machine_C_code(self):
        for i in range(1,len(self.file_to_open)+1):
            if self.file_to_open[i]["Memory Optimized"] == self.optimize_reads_and_calcs:
                if self.file_to_open[i]["Projector_type"] == self.projector_type:
                    if self.file_to_open[i]["DOI"] == self.doi_correction:
                        if self.file_to_open[i]["Decay"] == self.decay_correction:
                            if self.file_to_open[i]["Random"] == self.random_correction:
                                self.fw_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                         "machine_code",
                                                                         self.file_to_open[i]["File Folder"],
                                                                         self.file_to_open[i]["Filename Forward"])
                                self.bw_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                         "machine_code",
                                                                         self.file_to_open[i]["File Folder"],
                                                                         self.file_to_open[i]["Filename Back"])
        print("Forward: {}".format(self.fw_source_model_file))
        print("Backward: {}".format(self.bw_source_model_file))
        self.fw_source_model = open(self.fw_source_model_file)

        self.bw_source_model = open(self.bw_source_model_file)

        self.mod_forward_projection_shared_mem = SourceModule("""{}""".format((self.fw_source_model.read())))
        self.mod_backward_projection_shared_mem = SourceModule("""{}""".format((self.bw_source_model.read())))

    def multiplekernel(self):
        """ """
        print('Optimizer STARTED - Multiple reads')
        # cuda.init()
        cuda = self.cuda_drv
        # device = cuda.Device(0)  # enter your gpu id here
        # ctx = device.make_context()
        # start_x = np.int32(A[0, 0, 0])
        start_x = np.int32(self.A[0, 0, 0])
        start_y = np.int32(self.B[0, 0, 0])
        start_z = np.int32(self.C[0, 0, 0])
        print("Start_point: {},{},{}".format(start_x, start_y, start_z))
        print('Image size: {},{}, {}'.format(self.weight, self.height, self.depth))

        half_distance_between_array_pixel = np.float32(self.distance_between_array_pixel / 2)
        normalization_matrix = self.normalization_matrix.reshape(
            self.normalization_matrix.shape[0] * self.normalization_matrix.shape[1] * self.normalization_matrix.shape[
                2])

        # SOURCE MODELS (DEVICE CODE)
        self._load_machine_C_code()

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
        adjust_coef_cut = [None for _ in range(number_of_datasets_back)]
        adjust_coef_gpu = [None] * number_of_datasets
        adjust_coef_pinned = [None] * number_of_datasets_back
        fov_cut_matrix_cutted_gpu = [None] * number_of_datasets_back
        fov_cut_matrix_cut = [None] * number_of_datasets

        # Streams and Events creation
        for dataset in range(number_of_datasets):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[t], cuda.Event()) for t in range(len(marker_names))]))

        # Forward Projection Memory Allocation
        # Variables that need an unique alocation
        im_shappened = np.ascontiguousarray(self.im.reshape(self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
                                            dtype=np.float32)
        fov_cut_matrix_shappened = np.ascontiguousarray(
            self.fov_matrix_cut.reshape(
                self.fov_matrix_cut.shape[0] * self.fov_matrix_cut.shape[1] * self.fov_matrix_cut.shape[2]),
            dtype=np.byte)

        im_gpu = cuda.mem_alloc(im_shappened.size * im_shappened.dtype.itemsize)
        fov_cut_matrix_gpu = cuda.mem_alloc(fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        # texref = mod_forward_projection_shared_mem.get_texref('tex')
        # texref.set_address(fov_cut_matrix_gpu, fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        # # texref.set_format(cuda.array_format.UNSIGNED_INT8, 1)
        cuda.memcpy_htod_async(im_gpu, im_shappened)
        cuda.memcpy_htod_async(fov_cut_matrix_gpu, fov_cut_matrix_shappened)

        # Forward Memory allocation
        if self.projector_type == "Variable gaussian":
            gaussianFeatures = GaussianParameters(voxelSize=self.voxelSize)
            gaussianFeatures.setShiftVariantParameters(self.distance_to_center_plane_normal, self.distance_to_center_plane)

            forward_projection_arrays_all_data = [self.a, self.b, self.c, self.d,
                                                    self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                                    self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                                    self.sum_vor, self.distance_to_center_plane,
                                                    self.distance_to_center_plane_normal, self.time_factor,
                            gaussianFeatures.sigma_y_square, gaussianFeatures.sigma_z_square,
                            gaussianFeatures.gaussian_y_fix_term, gaussianFeatures.gaussian_z_fix_term,
                            gaussianFeatures.acceptableZDistance, gaussianFeatures.acceptableYDistance,
                            gaussianFeatures.invert2timesigma_y_square, gaussianFeatures.invert2timesigma_z_square]



        else:
            forward_projection_arrays_all_data = [self.a, self.b, self.c, self.d,
                                                  self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                                  self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                                  self.sum_vor, self.distance_to_center_plane,
                                                  self.distance_to_center_plane_normal, self.time_factor]

        forward_projection_arrays = [[None] * number_of_datasets for _ in
                                     range(len(forward_projection_arrays_all_data))]
        forward_projection_gpu_arrays = [[None] * number_of_datasets for _ in
                                         range(len(forward_projection_arrays_all_data))]
        forward_projection_pinned_arrays = [[None] * number_of_datasets for _ in
                                            range(len(forward_projection_arrays_all_data))]

        for ar in range(len(forward_projection_arrays)):

            array_original = forward_projection_arrays_all_data[ar]
            array = forward_projection_arrays[ar]
            array_gpu = forward_projection_gpu_arrays[ar]
            array_pinned = forward_projection_pinned_arrays[ar]
            for dataset in range(number_of_datasets):
                begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                array[dataset] = array_original[begin_dataset:end_dataset]
                array_gpu[dataset] = cuda.mem_alloc(array[dataset].size * array[dataset].dtype.itemsize)
                array_pinned[dataset] = cuda.register_host_memory(array[dataset])
                assert np.all(array_pinned[dataset] == array[dataset])
                cuda.memcpy_htod_async(array_gpu[dataset], array_pinned[dataset], stream[dataset])

            forward_projection_arrays_all_data[ar] = array_original
            forward_projection_arrays[ar] = array
            forward_projection_gpu_arrays[ar] = array_gpu
            forward_projection_pinned_arrays[ar] = array_pinned

        free, total = cuda.mem_get_info()
        print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))

        # Back projection Memory allocation
        if self.projector_type == "Variable gaussian":
            gaussianFeatures = GaussianParameters(voxelSize=self.voxelSize)
            gaussianFeatures.setShiftVariantParameters(self.distance_to_center_plane_normal,self.distance_to_center_plane)

            backward_projection_arrays_full_arrays = [self.a, self.b, self.c, self.d,
                                                    self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                                    self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                                    self.sum_vor, self.distance_to_center_plane,
                                                    self.distance_to_center_plane_normal, self.time_factor,
                            gaussianFeatures.sigma_y_square, gaussianFeatures.sigma_z_square,
                            gaussianFeatures.gaussian_y_fix_term, gaussianFeatures.gaussian_z_fix_term,
                            gaussianFeatures.acceptableZDistance, gaussianFeatures.acceptableYDistance,
                            gaussianFeatures.invert2timesigma_y_square, gaussianFeatures.invert2timesigma_z_square]
        else:
            backward_projection_arrays_full_arrays = [self.a, self.b, self.c, self.d,
                                                      self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                                      self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                                      self.sum_vor, self.distance_to_center_plane,
                                                      self.distance_to_center_plane_normal, self.time_factor]
        backward_projection_array_gpu_arrays = [[None] * number_of_datasets for _ in
                                                range(len(backward_projection_arrays_full_arrays))]
        backward_projection_pinned_arrays = [None] * len(backward_projection_arrays_full_arrays)

        for st in range(len(backward_projection_arrays_full_arrays)):
            backward_projection_array_gpu_arrays[st] = \
                cuda.mem_alloc(backward_projection_arrays_full_arrays[st].size
                               * backward_projection_arrays_full_arrays[st].dtype.itemsize)
            cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[st], backward_projection_arrays_full_arrays[st])
            # backward_projection_pinned_arrays[st] = cuda.register_host_memory(backward_projection_pinned_arrays[st])
            # assert np.all(array_pinned[dataset] == array[dataset])
            # cuda.memcpy_htod_async(array_gpu[dataset], array_pinned[dataset], stream[dataset])

        adjust_coef = np.ascontiguousarray(self.adjust_coef.reshape(
            self.adjust_coef.shape[0] * self.adjust_coef.shape[1] * self.adjust_coef.shape[2]),
            dtype=np.float32)

        A = np.ascontiguousarray(self.A.reshape(
            self.A.shape[0] * self.A.shape[1] * self.A.shape[2]),
            dtype=np.short)
        B = np.ascontiguousarray(self.B.reshape(
            self.B.shape[0] * self.B.shape[1] * self.B.shape[2]),
            dtype=np.short)
        C = np.ascontiguousarray(self.C.reshape(
            self.C.shape[0] * self.C.shape[1] * self.C.shape[2]),
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
        print('Number events for reconstruction: {}'.format(self.number_of_events))

        # -------------OSEM---------
        it = self.number_of_iterations
        subsets = self.number_of_subsets

        im = np.ascontiguousarray(self.im.reshape(self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
                                  dtype=np.float32)
        for i in range(it):
            print('Iteration number: {}\n----------------'.format(i + 1))
            print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))
            begin_event = np.int32(0)
            end_event = np.int32(self.number_of_events / subsets)
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
                    blockspergrid_x = int(math.ceil((end_dataset - begin_dataset) / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    event[dataset]['kernel_begin'].record(stream[dataset])

                    func_forward = self.mod_forward_projection_shared_mem.get_function(
                        "forward_projection_cdrf")
                    # if gaussian:
                    if self.projector_type == "Constant gaussian":
                        gaussianFeatures = GaussianParameters(voxelSize=self.voxelSize)
                        gaussianFeatures.setShiftInvariantParameters()
                        func_forward(gaussianFeatures.sigma_y_square, gaussianFeatures.sigma_z_square,
                                      gaussianFeatures.gaussian_y_fix_term, gaussianFeatures.gaussian_z_fix_term,
                                     gaussianFeatures.acceptableZDistance, gaussianFeatures.acceptableYDistance,
                                        gaussianFeatures.invert2timesigma_y_square, gaussianFeatures.invert2timesigma_z_square,
                            self.weight, self.height, self.depth, start_x, start_y, start_z,
                                     forward_projection_gpu_arrays[13][dataset],
                                     forward_projection_gpu_arrays[14][dataset],
                                     half_distance_between_array_pixel,
                                     self.number_of_events, begin_dataset, end_dataset,
                                     forward_projection_gpu_arrays[0][dataset],
                                     forward_projection_gpu_arrays[4][dataset],
                                     forward_projection_gpu_arrays[8][dataset],
                                     forward_projection_gpu_arrays[1][dataset],
                                     forward_projection_gpu_arrays[5][dataset],
                                     forward_projection_gpu_arrays[9][dataset],
                                     forward_projection_gpu_arrays[2][dataset],
                                     forward_projection_gpu_arrays[6][dataset],
                                     forward_projection_gpu_arrays[10][dataset],
                                     forward_projection_gpu_arrays[3][dataset],
                                     forward_projection_gpu_arrays[7][dataset],
                                     forward_projection_gpu_arrays[11][dataset],
                                     forward_projection_gpu_arrays[12][dataset], fov_cut_matrix_gpu, im_gpu,
                                     block=threadsperblock,
                                     grid=blockspergrid,
                                     stream=stream[dataset])
                    elif self.projector_type == "Variable gaussian":
                        func_forward(forward_projection_gpu_arrays[16][dataset], forward_projection_gpu_arrays[17][dataset],
                                        forward_projection_gpu_arrays[18][dataset], forward_projection_gpu_arrays[19][dataset],
                                        forward_projection_gpu_arrays[20][dataset], forward_projection_gpu_arrays[21][dataset],
                                        forward_projection_gpu_arrays[22][dataset], forward_projection_gpu_arrays[23][dataset],
                                    self.weight, self.height, self.depth, start_x, start_y, start_z,
                                     forward_projection_gpu_arrays[13][dataset],
                                     forward_projection_gpu_arrays[14][dataset],
                                     half_distance_between_array_pixel,
                                     self.number_of_events, begin_dataset, end_dataset,
                                     forward_projection_gpu_arrays[0][dataset],
                                     forward_projection_gpu_arrays[4][dataset],
                                     forward_projection_gpu_arrays[8][dataset],
                                     forward_projection_gpu_arrays[1][dataset],
                                     forward_projection_gpu_arrays[5][dataset],
                                     forward_projection_gpu_arrays[9][dataset],
                                     forward_projection_gpu_arrays[2][dataset],
                                     forward_projection_gpu_arrays[6][dataset],
                                     forward_projection_gpu_arrays[10][dataset],
                                     forward_projection_gpu_arrays[3][dataset],
                                     forward_projection_gpu_arrays[7][dataset],
                                     forward_projection_gpu_arrays[11][dataset],
                                     forward_projection_gpu_arrays[12][dataset], fov_cut_matrix_gpu, im_gpu,
                                     block=threadsperblock,
                                     grid=blockspergrid,
                                     stream=stream[dataset])
                    else:
                        func_forward(self.weight, self.height, self.depth, start_x, start_y, start_z,
                                     forward_projection_gpu_arrays[13][dataset],
                                     forward_projection_gpu_arrays[14][dataset],
                                     half_distance_between_array_pixel,
                                     self.number_of_events, begin_dataset, end_dataset,
                                     forward_projection_gpu_arrays[0][dataset], forward_projection_gpu_arrays[4][dataset],
                                     forward_projection_gpu_arrays[8][dataset],
                                     forward_projection_gpu_arrays[1][dataset], forward_projection_gpu_arrays[5][dataset],
                                     forward_projection_gpu_arrays[9][dataset],
                                     forward_projection_gpu_arrays[2][dataset], forward_projection_gpu_arrays[6][dataset],
                                     forward_projection_gpu_arrays[10][dataset],
                                     forward_projection_gpu_arrays[3][dataset], forward_projection_gpu_arrays[7][dataset],
                                     forward_projection_gpu_arrays[11][dataset],
                                     forward_projection_gpu_arrays[12][dataset], fov_cut_matrix_gpu, im_gpu,
                                     block=threadsperblock,
                                     grid=blockspergrid,
                                     stream=stream[dataset])

                # Sincronization of streams
                for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                # Transfering data from Optimizer
                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)
                    # cuda.memcpy_dtoh_async(sum_vor[begin_dataset:end_dataset], sum_vor_gpu[dataset])
                    # cuda.memcpy_dtoh_async(forward_projection_pinned_arrays[12][dataset],
                    #                        forward_projection_gpu_arrays[12][dataset], stream[dataset])
                    # forward_projection_arrays_all_data[12][begin_dataset:end_dataset] = \
                    #     forward_projection_pinned_arrays[12][dataset]

                    cuda.memcpy_dtoh_async(forward_projection_arrays[12][dataset],
                                           forward_projection_gpu_arrays[12][dataset], stream[dataset])
                    forward_projection_arrays_all_data[12][begin_dataset:end_dataset] = \
                        forward_projection_arrays[12][dataset]
                    # cuda.cudaStream.Synchronize(stream[dataset])

                    toc = time.time()

                cuda.Context.synchronize()
                # if self.normalizationFlag:
                #     forward_projection_arrays_all_data[12] = np.ones_like(forward_projection_arrays_all_data[12])
                print('Time part Forward Projection {} : {}'.format(1, toc - tic))
                # number_of_datasets = np.int32(2)
                #
                # teste = np.copy(forward_projection_arrays_all_data[12])
                # # sum_vor[sum_vor<1]=0
                # sum_vor = np.ascontiguousarray(teste, dtype=np.float32)
                # sum_vor=np.ascontiguousarray(np.ones((self.a.shape)), dtype=np.float32)
                print('SUM VOR: {}'.format(np.sum(forward_projection_arrays_all_data[12])))
                print('LEN VOR: {}'.format(len(forward_projection_arrays_all_data[12][forward_projection_arrays_all_data[12]==0])))
                # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)

                cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[12], forward_projection_arrays_all_data[12])

                # ------------BACKPROJECTION-----------

                for dataset in range(number_of_datasets_back):
                    dataset = np.int32(dataset)
                    begin_dataset = np.int32(0)
                    end_dataset = np.int32(number_of_events_subset)
                    event[dataset]['kernel_begin'].record(stream[dataset])
                    weight_cutted, height_cutted, depth_cutted = np.int32(
                        adjust_coef_cut[dataset].shape[0]), np.int32(
                        1), np.int32(1)

                    number_of_voxels_thread = 128
                    threadsperblock = (np.int(number_of_voxels_thread), 1, 1)
                    blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

                    func_backward = self.mod_backward_projection_shared_mem.get_function(
                        "backprojection_cdrf")
                    if self.projector_type == "Constant gaussian":
                        func_backward(gaussianFeatures.sigma_y_square, gaussianFeatures.sigma_z_square,
                                      gaussianFeatures.gaussian_y_fix_term, gaussianFeatures.gaussian_z_fix_term,
                                     gaussianFeatures.acceptableZDistance, gaussianFeatures.acceptableYDistance,
                                        gaussianFeatures.invert2timesigma_y_square, gaussianFeatures.invert2timesigma_z_square,
                                         dataset, weight_cutted, height_cutted, depth_cutted,
                                         backward_projection_array_gpu_arrays[13],
                                         backward_projection_array_gpu_arrays[14], half_distance_between_array_pixel,
                                         self.number_of_events, begin_dataset, end_dataset,
                                         backward_projection_array_gpu_arrays[0],
                                         backward_projection_array_gpu_arrays[4],
                                         backward_projection_array_gpu_arrays[8],
                                         backward_projection_array_gpu_arrays[1],
                                         backward_projection_array_gpu_arrays[5],
                                         backward_projection_array_gpu_arrays[9],
                                         backward_projection_array_gpu_arrays[2],
                                         backward_projection_array_gpu_arrays[6],
                                         backward_projection_array_gpu_arrays[10],
                                         backward_projection_array_gpu_arrays[3],
                                         backward_projection_array_gpu_arrays[7],
                                         backward_projection_array_gpu_arrays[11],
                                         A_cut_gpu[dataset], B_cut_gpu[dataset],
                                         C_cut_gpu[dataset],
                                         adjust_coef_gpu[dataset],
                                         backward_projection_array_gpu_arrays[12], fov_cut_matrix_cutted_gpu[dataset],
                                         backward_projection_array_gpu_arrays[15],
                                         block=threadsperblock,
                                         grid=blockspergrid,
                                         shared=np.int(4 * number_of_voxels_thread),
                                      stream=stream[dataset],
                                      )

                    elif self.projector_type == "Variable gaussian":
                        func_backward(backward_projection_array_gpu_arrays[16], backward_projection_array_gpu_arrays[17],
                                        backward_projection_array_gpu_arrays[18], backward_projection_array_gpu_arrays[19],
                                        backward_projection_array_gpu_arrays[20], backward_projection_array_gpu_arrays[21],
                                        backward_projection_array_gpu_arrays[22], backward_projection_array_gpu_arrays[23],
                                            dataset, weight_cutted, height_cutted, depth_cutted,
                                      backward_projection_array_gpu_arrays[13],
                                      backward_projection_array_gpu_arrays[14], half_distance_between_array_pixel,
                                      self.number_of_events, begin_dataset, end_dataset,
                                      backward_projection_array_gpu_arrays[0],
                                      backward_projection_array_gpu_arrays[4],
                                      backward_projection_array_gpu_arrays[8],
                                      backward_projection_array_gpu_arrays[1],
                                      backward_projection_array_gpu_arrays[5],
                                      backward_projection_array_gpu_arrays[9],
                                      backward_projection_array_gpu_arrays[2],
                                      backward_projection_array_gpu_arrays[6],
                                      backward_projection_array_gpu_arrays[10],
                                      backward_projection_array_gpu_arrays[3],
                                      backward_projection_array_gpu_arrays[7],
                                      backward_projection_array_gpu_arrays[11],
                                      A_cut_gpu[dataset], B_cut_gpu[dataset],
                                      C_cut_gpu[dataset],
                                      adjust_coef_gpu[dataset],
                                      backward_projection_array_gpu_arrays[12], fov_cut_matrix_cutted_gpu[dataset],
                                      backward_projection_array_gpu_arrays[15],
                                      block=threadsperblock,
                                      grid=blockspergrid,
                                      shared=np.int(4 * number_of_voxels_thread),
                                      stream=stream[dataset],
                                      )
                    else:
                        func_backward(dataset, weight_cutted, height_cutted, depth_cutted,
                                      backward_projection_array_gpu_arrays[13],
                                      backward_projection_array_gpu_arrays[14], half_distance_between_array_pixel,
                                      self.number_of_events, begin_dataset, end_dataset,
                                      backward_projection_array_gpu_arrays[0],
                                      backward_projection_array_gpu_arrays[4],
                                      backward_projection_array_gpu_arrays[8],
                                      backward_projection_array_gpu_arrays[1],
                                      backward_projection_array_gpu_arrays[5],
                                      backward_projection_array_gpu_arrays[9],
                                      backward_projection_array_gpu_arrays[2],
                                      backward_projection_array_gpu_arrays[6],
                                      backward_projection_array_gpu_arrays[10],
                                      backward_projection_array_gpu_arrays[3],
                                      backward_projection_array_gpu_arrays[7],
                                      backward_projection_array_gpu_arrays[11],
                                      A_cut_gpu[dataset], B_cut_gpu[dataset],
                                      C_cut_gpu[dataset],
                                      adjust_coef_gpu[dataset],
                                      backward_projection_array_gpu_arrays[12], fov_cut_matrix_cutted_gpu[dataset],
                                      backward_projection_array_gpu_arrays[15],
                                      block=threadsperblock,
                                      grid=blockspergrid,
                                      shared=np.int(4 * number_of_voxels_thread),
                                      stream=stream[dataset],
                                      )
                #
                for dataset in range(number_of_datasets_back):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                for dataset in range(number_of_datasets_back):
                    cuda.memcpy_dtoh_async(adjust_coef_cut[dataset], adjust_coef_gpu[dataset])
                    adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division] = adjust_coef_cut[
                        dataset]

                cuda.Context.synchronize()
                print('Time part Backward Projection {} : {}'.format(1, time.time() - toc))
                print('SUM IMAGE: {}'.format(np.sum(adjust_coef)))
                penalized_term = self._load_penalized_term(im)
                # normalization
                im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
                    normalization_matrix != 0] / (normalization_matrix[normalization_matrix != 0])
                # im[normalization_matrix == 0] = 0
                if self.algorithm == "LM-MRP":
                    im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
                # im = self._apply_penalized_term(im, penalized_term)
                print('SUM IMAGE: {}'.format(np.sum(im)))
                im = np.ascontiguousarray(im, dtype=np.float32)

                cuda.memcpy_htod_async(im_gpu, im)

                # # Clearing variables
                forward_projection_arrays_all_data[12] = np.ascontiguousarray(
                    np.zeros(self.a.shape, dtype=np.float32))

                adjust_coef = np.ascontiguousarray(
                    np.zeros((self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z),
                             dtype=np.float32))

                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                    forward_projection_arrays[12][dataset] = forward_projection_arrays_all_data[12][
                                                             begin_dataset:end_dataset]
                    forward_projection_gpu_arrays[12][dataset] = cuda.mem_alloc(
                        forward_projection_arrays[12][dataset].size * forward_projection_arrays[12][
                            dataset].dtype.itemsize)
                    # forward_projection_pinned_arrays[12][dataset] = cuda.register_host_memory(
                    #     forward_projection_arrays[12][dataset])
                    # assert np.all(
                    #     forward_projection_pinned_arrays[12][dataset] == forward_projection_arrays[12][dataset])
                    # cuda.memcpy_htod_async(forward_projection_gpu_arrays[12][dataset],
                    #                        forward_projection_pinned_arrays[12][dataset], stream[dataset])

                    cuda.memcpy_htod_async(forward_projection_gpu_arrays[12][dataset],
                                           forward_projection_arrays[12][dataset], stream[dataset])

                for dataset in range(number_of_datasets_back):
                    adjust_coef_cut[dataset] = adjust_coef[
                                               dataset * voxels_division:(dataset + 1) * voxels_division]
                    # adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
                    # assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
                    # cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])
                    cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset], stream[dataset])

                if self.saved_image_by_iteration:
                    if i % 2==0:
                        im_to_save = im.reshape(self.weight, self.height, self.depth)
                        self._save_image_by_it(im_to_save, i, sb)
                # del adjust_coef_pinned
                gc.collect()
                # adjust_coef_pinned = [None] * number_of_datasets_back
                # if self.signals_interface is not None:
                #     self.signals_interface.trigger_update_label_reconstruction_status.emit(
                #         "{}: Iteration {}".format(self.current_info_step, i + 1))
                #     self.signals_interface.trigger_progress_reconstruction_partial.emit(
                #         int(np.round(100 * (i + 1) * (sb + subsets) / (it * subsets), 0)))

        im = im.reshape(self.weight, self.height, self.depth)
        self.im = im

    def multikernel_optimized_memory_reads(self):
        print('Optimizer STARTED - Multiple reads')
        # cuda.init()
        cuda = self.cuda_drv
        # device = cuda.Device(0)  # enter your gpu id here
        # ctx = device.make_context()
        # start_x = np.int32(A[0, 0, 0])
        start_x = np.int32(self.A[0, 0, 0])
        start_y = np.int32(self.B[0, 0, 0])
        start_z = np.int32(self.C[0, 0, 0])
        print("Start_point: {},{},{}".format(start_x, start_y, start_z))
        print('Image size: {},{}, {}'.format(self.weight, self.height, self.depth))

        half_distance_between_array_pixel = np.float32(self.distance_between_array_pixel / 2)
        normalization_matrix = self.normalization_matrix.reshape(
            self.normalization_matrix.shape[0] * self.normalization_matrix.shape[1] * self.normalization_matrix.shape[
                2])

        # SOURCE MODELS (DEVICE CODE)
        self._load_machine_C_code()

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
        adjust_coef_cut = [None for _ in range(number_of_datasets_back)]
        adjust_coef_gpu = [None] * number_of_datasets
        adjust_coef_pinned = [None] * number_of_datasets_back
        fov_cut_matrix_cutted_gpu = [None] * number_of_datasets_back
        fov_cut_matrix_cut = [None] * number_of_datasets

        # Streams and Events creation
        for dataset in range(number_of_datasets):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[t], cuda.Event()) for t in range(len(marker_names))]))

        # Forward Projection Memory Allocation
        # Variables that need an unique alocation
        im_shappened = np.ascontiguousarray(self.im.reshape(self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
                                            dtype=np.float32)
        fov_cut_matrix_shappened = np.ascontiguousarray(
            self.fov_matrix_cut.reshape(
                self.fov_matrix_cut.shape[0] * self.fov_matrix_cut.shape[1] * self.fov_matrix_cut.shape[2]),
            dtype=np.byte)

        im_gpu = cuda.mem_alloc(im_shappened.size * im_shappened.dtype.itemsize)
        fov_cut_matrix_gpu = cuda.mem_alloc(fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        # texref = mod_forward_projection_shared_mem.get_texref('tex')
        # texref.set_address(fov_cut_matrix_gpu, fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        # # texref.set_format(cuda.array_format.UNSIGNED_INT8, 1)
        cuda.memcpy_htod_async(im_gpu, im_shappened)
        cuda.memcpy_htod_async(fov_cut_matrix_gpu, fov_cut_matrix_shappened)

        # Forward Memory allocation
        forward_projection_arrays_all_data = [self.a, self.b, self.c, self.d,
                                              self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                              self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                              self.sum_vor, self.distance_to_center_plane,
                                              self.distance_to_center_plane_normal, self.time_factor,
                                              self.x_min_f, self.x_max_f,  self.y_min_f, self.y_max_f,
                                              self.z_min_f, self.z_max_f]

        forward_projection_arrays = [[None] * number_of_datasets for _ in
                                     range(len(forward_projection_arrays_all_data))]
        forward_projection_gpu_arrays = [[None] * number_of_datasets for _ in
                                         range(len(forward_projection_arrays_all_data))]
        forward_projection_pinned_arrays = [[None] * number_of_datasets for _ in
                                            range(len(forward_projection_arrays_all_data))]

        for ar in range(len(forward_projection_arrays)):

            array_original = forward_projection_arrays_all_data[ar]
            array = forward_projection_arrays[ar]
            array_gpu = forward_projection_gpu_arrays[ar]
            # array_pinned = forward_projection_pinned_arrays[ar]
            for dataset in range(number_of_datasets):
                begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                array[dataset] = array_original[begin_dataset:end_dataset]
                array_gpu[dataset] = cuda.mem_alloc(array[dataset].size * array[dataset].dtype.itemsize)
                # array_pinned[dataset] = cuda.register_host_memory(array[dataset])
                # assert np.all(array_pinned[dataset] == array[dataset])
                cuda.memcpy_htod_async(array_gpu[dataset], array[dataset], stream[dataset])

            forward_projection_arrays_all_data[ar] = array_original
            forward_projection_arrays[ar] = array
            forward_projection_gpu_arrays[ar] = array_gpu
            # forward_projection_pinned_arrays[ar] = array_pinned

        free, total = cuda.mem_get_info()
        print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))

        # Back projection Memory allocation
        backward_projection_arrays_full_arrays = [self.a, self.b, self.c, self.d,
                                                  self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                                  self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                                  self.sum_vor, self.distance_to_center_plane,
                                                  self.distance_to_center_plane_normal, self.time_factor,  self.x_min_f, self.x_max_f,  self.y_min_f, self.y_max_f,
                                              self.z_min_f, self.z_max_f]
        backward_projection_array_gpu_arrays = [[None] * number_of_datasets for _ in
                                                range(len(backward_projection_arrays_full_arrays))]
        backward_projection_pinned_arrays = [None] * len(backward_projection_arrays_full_arrays)

        for st in range(len(backward_projection_arrays_full_arrays)):
            backward_projection_array_gpu_arrays[st] = \
                cuda.mem_alloc(backward_projection_arrays_full_arrays[st].size
                               * backward_projection_arrays_full_arrays[st].dtype.itemsize)
            cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[st], backward_projection_arrays_full_arrays[st])
            # backward_projection_pinned_arrays[st] = cuda.register_host_memory(backward_projection_pinned_arrays[st])
            # assert np.all(array_pinned[dataset] == array[dataset])
            # cuda.memcpy_htod_async(array_gpu[dataset], array_pinned[dataset], stream[dataset])

        adjust_coef = np.ascontiguousarray(self.adjust_coef.reshape(
            self.adjust_coef.shape[0] * self.adjust_coef.shape[1] * self.adjust_coef.shape[2]),
            dtype=np.float32)

        A = np.ascontiguousarray(self.A.reshape(
            self.A.shape[0] * self.A.shape[1] * self.A.shape[2]),
            dtype=np.short)
        B = np.ascontiguousarray(self.B.reshape(
            self.B.shape[0] * self.B.shape[1] * self.B.shape[2]),
            dtype=np.short)
        C = np.ascontiguousarray(self.C.reshape(
            self.C.shape[0] * self.C.shape[1] * self.C.shape[2]),
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
        print('Number events for reconstruction: {}'.format(self.number_of_events))

        # -------------OSEM---------
        it = self.number_of_iterations
        subsets = self.number_of_subsets

        im = np.ascontiguousarray(self.im.reshape(self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
                                  dtype=np.float32)
        for i in range(it):
            print('Iteration number: {}\n----------------'.format(i + 1))
            print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))
            begin_event = np.int32(0)
            end_event = np.int32(self.number_of_events / subsets)
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
                    blockspergrid_x = int(math.ceil((end_dataset - begin_dataset) / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    event[dataset]['kernel_begin'].record(stream[dataset])

                    func_forward = self.mod_forward_projection_shared_mem.get_function(
                        "forward_projection_cdrf")

                    func_forward(self.weight, self.height, self.depth, start_x, start_y, start_z,
                                 forward_projection_gpu_arrays[13][dataset],
                                 forward_projection_gpu_arrays[14][dataset],
                                 half_distance_between_array_pixel,
                                 self.number_of_events, begin_dataset, end_dataset,
                                 forward_projection_gpu_arrays[0][dataset], forward_projection_gpu_arrays[4][dataset],
                                 forward_projection_gpu_arrays[8][dataset],
                                 forward_projection_gpu_arrays[1][dataset], forward_projection_gpu_arrays[5][dataset],
                                 forward_projection_gpu_arrays[9][dataset],
                                 forward_projection_gpu_arrays[2][dataset], forward_projection_gpu_arrays[6][dataset],
                                 forward_projection_gpu_arrays[10][dataset],
                                 forward_projection_gpu_arrays[3][dataset], forward_projection_gpu_arrays[7][dataset],
                                 forward_projection_gpu_arrays[11][dataset],
                                 forward_projection_gpu_arrays[12][dataset], fov_cut_matrix_gpu, im_gpu, forward_projection_gpu_arrays[16][dataset],
                                 forward_projection_gpu_arrays[17][dataset], forward_projection_gpu_arrays[18][dataset],
                                    forward_projection_gpu_arrays[19][dataset], forward_projection_gpu_arrays[20][dataset],
                                    forward_projection_gpu_arrays[21][dataset],
                                 block=threadsperblock,
                                 grid=blockspergrid,
                                 stream=stream[dataset])

                # Sincronization of streams
                for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                # Transfering data from Optimizer
                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)
                    # cuda.memcpy_dtoh_async(sum_vor[begin_dataset:end_dataset], sum_vor_gpu[dataset])
                    # cuda.memcpy_dtoh_async(forward_projection_pinned_arrays[12][dataset],
                    #                        forward_projection_gpu_arrays[12][dataset], stream[dataset])
                    # forward_projection_arrays_all_data[12][begin_dataset:end_dataset] = \
                    #     forward_projection_pinned_arrays[12][dataset]

                    cuda.memcpy_dtoh_async(forward_projection_arrays[12][dataset],
                                           forward_projection_gpu_arrays[12][dataset], stream[dataset])
                    forward_projection_arrays_all_data[12][begin_dataset:end_dataset] = \
                        forward_projection_arrays[12][dataset]
                    # cuda.cudaStream.Synchronize(stream[dataset])

                    toc = time.time()

                cuda.Context.synchronize()
                # if self.normalizationFlag:
                #     forward_projection_arrays_all_data[12] = np.ones_like(forward_projection_arrays_all_data[12])
                print('Time part Forward Projection {} : {}'.format(1, toc - tic))
                # number_of_datasets = np.int32(2)
                #
                # teste = np.copy(forward_projection_arrays_all_data[12])
                # # sum_vor[sum_vor<1]=0
                # sum_vor = np.ascontiguousarray(teste, dtype=np.float32)
                # sum_vor=np.ascontiguousarray(np.ones((self.a.shape)), dtype=np.float32)
                print('SUM VOR: {}'.format(np.sum(forward_projection_arrays_all_data[12])))
                print('LEN VOR: {}'.format(
                    len(forward_projection_arrays_all_data[12][forward_projection_arrays_all_data[12] == 0])))
                # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)

                cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[12], forward_projection_arrays_all_data[12])

                # ------------BACKPROJECTION-----------

                for dataset in range(number_of_datasets_back):
                    dataset = np.int32(dataset)
                    begin_dataset = np.int32(0)
                    end_dataset = np.int32(number_of_events_subset)
                    event[dataset]['kernel_begin'].record(stream[dataset])
                    weight_cutted, height_cutted, depth_cutted = np.int32(
                        adjust_coef_cut[dataset].shape[0]), np.int32(
                        1), np.int32(1)

                    number_of_voxels_thread = 128
                    threadsperblock = (np.int(number_of_voxels_thread), 1, 1)
                    blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

                    func_backward = self.mod_backward_projection_shared_mem.get_function(
                        "backprojection_cdrf")
                    func_backward(dataset, self.weight, self.height, self.depth, start_x, start_y, start_z,
                                  backward_projection_array_gpu_arrays[13],
                                  backward_projection_array_gpu_arrays[14], half_distance_between_array_pixel,
                                  self.number_of_events, begin_dataset, end_dataset,
                                  backward_projection_array_gpu_arrays[0],
                                  backward_projection_array_gpu_arrays[4],
                                  backward_projection_array_gpu_arrays[8],
                                  backward_projection_array_gpu_arrays[1],
                                  backward_projection_array_gpu_arrays[5],
                                  backward_projection_array_gpu_arrays[9],
                                  backward_projection_array_gpu_arrays[2],
                                  backward_projection_array_gpu_arrays[6],
                                  backward_projection_array_gpu_arrays[10],
                                  backward_projection_array_gpu_arrays[3],
                                  backward_projection_array_gpu_arrays[7],
                                  backward_projection_array_gpu_arrays[11],
                                  A_cut_gpu[dataset], B_cut_gpu[dataset],
                                  C_cut_gpu[dataset],
                                  adjust_coef_gpu[dataset],
                                  backward_projection_array_gpu_arrays[12], fov_cut_matrix_cutted_gpu[dataset],
                                  backward_projection_array_gpu_arrays[16],
                                  backward_projection_array_gpu_arrays[17], backward_projection_array_gpu_arrays[18],
                                  backward_projection_array_gpu_arrays[19], backward_projection_array_gpu_arrays[20],
                                    backward_projection_array_gpu_arrays[21],
                                  block=threadsperblock,
                                  grid=blockspergrid,
                                  shared=np.int(4 * number_of_voxels_thread),
                                  stream=stream[dataset],
                                  )
                #
                for dataset in range(number_of_datasets_back):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                for dataset in range(number_of_datasets_back):
                    cuda.memcpy_dtoh_async(adjust_coef_cut[dataset], adjust_coef_gpu[dataset])
                    adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division] = adjust_coef_cut[
                        dataset]

                cuda.Context.synchronize()
                print('Time part Backward Projection {} : {}'.format(1, time.time() - toc))
                print('SUM IMAGE: {}'.format(np.sum(adjust_coef)))
                penalized_term = self._load_penalized_term(im)
                # normalization
                im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
                    normalization_matrix != 0] / (normalization_matrix[normalization_matrix != 0])
                # im[normalization_matrix == 0] = 0
                if self.algorithm == "LM-MRP":
                    im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
                # im = self._apply_penalized_term(im, penalized_term)
                print('SUM IMAGE: {}'.format(np.sum(im)))
                im = np.ascontiguousarray(im, dtype=np.float32)

                cuda.memcpy_htod_async(im_gpu, im)

                # # Clearing variables
                forward_projection_arrays_all_data[12] = np.ascontiguousarray(
                    np.zeros(self.a.shape, dtype=np.float32))

                adjust_coef = np.ascontiguousarray(
                    np.zeros((self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z),
                             dtype=np.float32))

                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                    forward_projection_arrays[12][dataset] = forward_projection_arrays_all_data[12][
                                                             begin_dataset:end_dataset]
                    forward_projection_gpu_arrays[12][dataset] = cuda.mem_alloc(
                        forward_projection_arrays[12][dataset].size * forward_projection_arrays[12][
                            dataset].dtype.itemsize)
                    # forward_projection_pinned_arrays[12][dataset] = cuda.register_host_memory(
                    #     forward_projection_arrays[12][dataset])
                    # assert np.all(
                    #     forward_projection_pinned_arrays[12][dataset] == forward_projection_arrays[12][dataset])
                    # cuda.memcpy_htod_async(forward_projection_gpu_arrays[12][dataset],
                    #                        forward_projection_pinned_arrays[12][dataset], stream[dataset])

                    cuda.memcpy_htod_async(forward_projection_gpu_arrays[12][dataset],
                                           forward_projection_arrays[12][dataset], stream[dataset])

                for dataset in range(number_of_datasets_back):
                    adjust_coef_cut[dataset] = adjust_coef[
                                               dataset * voxels_division:(dataset + 1) * voxels_division]
                    # adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
                    # assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
                    # cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])
                    cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset], stream[dataset])

                if self.saved_image_by_iteration:
                    if i % 2 == 0:
                        im_to_save = im.reshape(self.weight, self.height, self.depth)
                        self._save_image_by_it(im_to_save, i, sb)
                # del adjust_coef_pinned
                gc.collect()
                # adjust_coef_pinned = [None] * number_of_datasets_back
                # if self.signals_interface is not None:
                #     self.signals_interface.trigger_update_label_reconstruction_status.emit(
                #         "{}: Iteration {}".format(self.current_info_step, i + 1))
                #     self.signals_interface.trigger_progress_reconstruction_partial.emit(
                #         int(np.round(100 * (i + 1) * (sb + subsets) / (it * subsets), 0)))

        im = im.reshape(self.weight, self.height, self.depth)
        self.im = im

    def multikernel_optimized_memory_reads_review(self):
        print('Optimizer STARTED')
        # cuda.init()
        cuda = self.cuda_drv
        start_x = np.int32(self.A[0, 0, 0])
        start_y = np.int32(self.B[0, 0, 0])
        start_z = np.int32(self.C[0, 0, 0])
        print("Start_point: {},{},{}".format(start_x, start_y, start_z))
        print('Image size: {},{}, {}'.format(self.weight, self.height, self.depth))

        half_distance_between_array_pixel = np.float32(self.distance_between_array_pixel / 2)
        normalization_matrix = self.normalization_matrix.reshape(
            self.normalization_matrix.shape[0] * self.normalization_matrix.shape[1] * self.normalization_matrix.shape[
                2])

        # SOURCE MODELS (DEVICE CODE)
        self._load_machine_C_code()
        # texref = self.mod_backward_projection_shared_mem.get_texref("tex")
        # texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)
        # texref.set_filter_mode(cuda.filter_mode.LINEAR)
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

        adjust_coef_cut = [None] * number_of_datasets
        adjust_coef_gpu = [None] * number_of_datasets
        adjust_coef_pinned = [None] * number_of_datasets_back


        A_cut_gpu, B_cut_gpu, C_cut_gpu = [None for _ in range(number_of_datasets_back)], \
                                          [None for _ in range(number_of_datasets_back)],\
                                          [None for _ in range(number_of_datasets_back)]
        A_cut, B_cut, C_cut = [None for _ in range(number_of_datasets_back)],\
                              [None for _ in range(number_of_datasets_back)], \
                              [None for _ in range(number_of_datasets_back)]
        adjust_coef_cut = [None for _ in range(number_of_datasets_back)]
        adjust_coef_gpu = [None for _ in range(number_of_datasets_back)]
        adjust_coef_pinned = [None for _ in range(number_of_datasets_back)]
        fov_cut_matrix_cutted_gpu = [None for _ in range(number_of_datasets_back)]
        fov_cut_matrix_cut = [None for _ in range(number_of_datasets_back)]

        # Streams and Events creation
        for dataset in range(number_of_datasets_back):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))

        im_shappened = np.ascontiguousarray(self.im.reshape(self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
                                            dtype=np.float32)
        fov_cut_matrix_shappened = np.ascontiguousarray(
            self.fov_matrix_cut.reshape(
                self.fov_matrix_cut.shape[0] * self.fov_matrix_cut.shape[1] * self.fov_matrix_cut.shape[2]),
            dtype=np.byte)

        # Forward Memory allocation
        forward_projection_arrays_all_data = [self.a, self.b, self.c, self.d,
                                              self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                              self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                              self.sum_vor, self.distance_to_center_plane,
                                              self.distance_to_center_plane_normal, self.time_factor,
                                              self.x_min_f, self.x_max_f,  self.y_min_f, self.y_max_f,
                                              self.z_min_f, self.z_max_f]

        forward_projection_arrays = [[None] * number_of_datasets for _ in
                                     range(len(forward_projection_arrays_all_data))]
        forward_projection_gpu_arrays = [[None] * number_of_datasets for _ in
                                         range(len(forward_projection_arrays_all_data))]
        forward_projection_pinned_arrays = [[None] * number_of_datasets for _ in
                                            range(len(forward_projection_arrays_all_data))]

        for ar in range(len(forward_projection_arrays)):

            array_original = forward_projection_arrays_all_data[ar]
            array = forward_projection_arrays[ar]
            array_gpu = forward_projection_gpu_arrays[ar]
            array_pinned = forward_projection_pinned_arrays[ar]
            for dataset in range(number_of_datasets):
                begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                array[dataset] = array_original[begin_dataset:end_dataset]
                array_gpu[dataset] = cuda.mem_alloc(array[dataset].size * array[dataset].dtype.itemsize)
                array_pinned[dataset] = cuda.register_host_memory(array[dataset])
                assert np.all(array_pinned[dataset] == array[dataset])
                cuda.memcpy_htod_async(array_gpu[dataset], array_pinned[dataset], stream[dataset])

            forward_projection_arrays_all_data[ar] = array_original
            forward_projection_arrays[ar] = array
            forward_projection_gpu_arrays[ar] = array_gpu
            forward_projection_pinned_arrays[ar] = array_pinned

        im_gpu = cuda.mem_alloc(im_shappened.size * im_shappened.dtype.itemsize)
        fov_cut_matrix_gpu = cuda.mem_alloc(fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)
        cuda.memcpy_htod_async(im_gpu, im_shappened)
        cuda.memcpy_htod_async(fov_cut_matrix_gpu, fov_cut_matrix_shappened)

        adjust_coef = np.ascontiguousarray(self.adjust_coef.reshape(
            self.adjust_coef.shape[0] * self.adjust_coef.shape[1] * self.adjust_coef.shape[2]),
            dtype=np.float32)

        d2_normal = np.ascontiguousarray(self.distance_to_center_plane * np.sqrt(
            self.a_normal**2 + self.b_normal**2 + self.c_normal**2), dtype=np.float32)

        d2= np.ascontiguousarray(self.distance_to_center_plane_normal * np.sqrt(
            self.a ** 2 + self.b ** 2 + self.c ** 2), dtype=np.float32)

        d2_cf = np.ascontiguousarray(half_distance_between_array_pixel * np.sqrt(
            self.a_cf ** 2 + self.b_cf ** 2 + self.c_cf ** 2), dtype=np.float32)

        voxels_division = adjust_coef.shape[0] // number_of_datasets_back
        backward_projection_arrays_full_arrays = [self.a, self.b, self.c, self.d,
                                                  self.a_normal, self.b_normal, self.c_normal, self.d_normal,
                                                  self.a_cf, self.b_cf, self.c_cf, self.d_cf,
                                                  self.sum_vor, self.distance_to_center_plane,
                                                  self.distance_to_center_plane_normal, self.time_factor,
                                                  self.x_min_f, self.x_max_f, self.y_min_f, self.y_max_f,
                                                  self.z_min_f, self.z_max_f, d2_normal, d2, d2_cf]
        backward_projection_array_gpu_arrays = [[None] * number_of_datasets for _ in
                                                range(len(backward_projection_arrays_full_arrays))]
        backward_projection_pinned_arrays = [None] * len(backward_projection_arrays_full_arrays)

        for st in range(len(backward_projection_arrays_full_arrays)):
            backward_projection_array_gpu_arrays[st] = \
                cuda.mem_alloc(backward_projection_arrays_full_arrays[st].size
                               * backward_projection_arrays_full_arrays[st].dtype.itemsize)
            cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[st], backward_projection_arrays_full_arrays[st])
            # backward_projection_pinned_arrays[st] = cuda.register_host_memory(backward_projection_pinned_arrays[st])
            # assert np.all(array_pinned[dataset] == array[dataset])
            # cuda.memcpy_htod_async(array_gpu[dataset], array_pinned[dataset], stream[dataset])
        # texref.set_address(backward_projection_array_gpu_arrays[0], backward_projection_arrays_full_arrays[0].nbytes)
        # texref.set_format(cuda.array_format.FLOAT, 1)
        # texref.set_flags(cuda.TRSF_NORMALIZED_COORDINATES)

        adjust_coef = np.ascontiguousarray(self.adjust_coef.reshape(
            self.adjust_coef.shape[0] * self.adjust_coef.shape[1] * self.adjust_coef.shape[2]),
            dtype=np.float32)

        A = np.ascontiguousarray(self.A.reshape(
            self.A.shape[0] * self.A.shape[1] * self.A.shape[2]),
            dtype=np.short)
        B = np.ascontiguousarray(self.B.reshape(
            self.B.shape[0] * self.B.shape[1] * self.B.shape[2]),
            dtype=np.short)
        C = np.ascontiguousarray(self.C.reshape(
            self.C.shape[0] * self.C.shape[1] * self.C.shape[2]),
            dtype=np.short)
        adjust_coef_gpu= cuda.mem_alloc(
                    adjust_coef.size * adjust_coef.dtype.itemsize)
        # ---- Divide into datasets variables backprojection
        events_mapping_per_dataset = [None for _ in range(number_of_datasets_back)]
        events_mapping_per_dataset_gpu = [None for _ in range(number_of_datasets_back)]
        events_mapping_per_dataset_pinned = [None for _ in range(number_of_datasets_back)]
        for dataset in range(number_of_datasets_back):
            index_events = self.map_events_gpu(min=dataset * 8, max=(dataset + 1) * 8)


            events_mapping_per_dataset[dataset] = np.ascontiguousarray(index_events, dtype=np.int32)
            events_mapping_per_dataset_gpu[dataset] = cuda.mem_alloc(
                        events_mapping_per_dataset[dataset].size * events_mapping_per_dataset[dataset].dtype.itemsize)
            events_mapping_per_dataset_pinned[dataset] = cuda.register_host_memory(events_mapping_per_dataset[dataset])
            assert np.all(events_mapping_per_dataset_pinned[dataset] == events_mapping_per_dataset[dataset])
            cuda.memcpy_htod_async(events_mapping_per_dataset_gpu[dataset], events_mapping_per_dataset_pinned[dataset], stream[dataset])

        #     voxels_division = adjust_coef.shape[0] // number_of_datasets_back
        #     adjust_coef_cut[dataset] = np.ascontiguousarray(
        #         adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division],
        #         dtype=np.float32)
        #
        #     fov_cut_matrix_cut[dataset] = np.ascontiguousarray(
        #         fov_cut_matrix_shappened[dataset * voxels_division:(dataset + 1) * voxels_division],
        #         dtype=np.byte)
        #
        #     A_cut[dataset] = np.ascontiguousarray(
        #         A[dataset * voxels_division:(dataset + 1) * voxels_division],
        #         dtype=np.short)
        #
        #     B_cut[dataset] = np.ascontiguousarray(
        #         B[dataset * voxels_division:(dataset + 1) * voxels_division],
        #         dtype=np.short)
        #
        #     C_cut[dataset] = np.ascontiguousarray(
        #         C[dataset * voxels_division:(dataset + 1) * voxels_division],
        #         dtype=np.short)
        #     # Backprojection
        #     adjust_coef_gpu[dataset] = cuda.mem_alloc(
        #         adjust_coef_cut[dataset].size * adjust_coef_cut[dataset].dtype.itemsize)
        #
        #     adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
        #     assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
        #     cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])
        #
        #     fov_cut_matrix_cutted_gpu[dataset] = cuda.mem_alloc(
        #         fov_cut_matrix_cut[dataset].size * fov_cut_matrix_cut[dataset].dtype.itemsize)
        #
        #     A_cut_gpu[dataset] = cuda.mem_alloc(
        #         A_cut[dataset].size * A_cut[dataset].dtype.itemsize)
        #
        #     B_cut_gpu[dataset] = cuda.mem_alloc(
        #         B_cut[dataset].size * B_cut[dataset].dtype.itemsize)
        #
        #     C_cut_gpu[dataset] = cuda.mem_alloc(
        #         C_cut[dataset].size * C_cut[dataset].dtype.itemsize)
        #
        #     cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset])
        #     cuda.memcpy_htod_async(fov_cut_matrix_cutted_gpu[dataset], fov_cut_matrix_cut[dataset])
        #     cuda.memcpy_htod_async(A_cut_gpu[dataset], A_cut[dataset])
        #     cuda.memcpy_htod_async(B_cut_gpu[dataset], B_cut[dataset])
        #     cuda.memcpy_htod_async(C_cut_gpu[dataset], C_cut[dataset])


        free, total = cuda.mem_get_info()
        # cuda.memcpy_htod_async(im_gpu, realrow)


        # cuda.matrix_to_texref(realrow, texref, order="C")

        # texref.set_array(realrow)


        print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))

        # -------------OSEM---------
        it = self.number_of_iterations
        subsets = self.number_of_subsets
        print('Number events for reconstruction: {}'.format(self.number_of_events))

        im = np.ascontiguousarray(self.im.reshape(self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
                                  dtype=np.float32)
        for i in range(it):
            print('Iteration number: {}\n----------------'.format(i + 1))
            begin_event = np.int32(0)
            end_event = np.int32(self.number_of_events / subsets)
            for sb in range(subsets):
                print('Subset number: {}'.format(sb))
                number_of_events_subset = np.int32(end_event - begin_event)
                tic = time.time()
                # Cycle forward Projection
                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * number_of_events_subset / number_of_datasets)

                    threadsperblock = (256, 1, 1)
                    blockspergrid_x = int(math.ceil(((end_dataset - begin_dataset)) / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    event[dataset]['kernel_begin'].record(stream[dataset])

                    func_forward = self.mod_forward_projection_shared_mem.get_function("forward_projection_cdrf")


                    func_forward(self.weight, self.height, self.depth, start_x, start_y, start_z,
                                 forward_projection_gpu_arrays[13][dataset],
                                 forward_projection_gpu_arrays[14][dataset],
                                 half_distance_between_array_pixel,
                                 self.number_of_events, begin_dataset, end_dataset,
                                 forward_projection_gpu_arrays[0][dataset], forward_projection_gpu_arrays[4][dataset],
                                 forward_projection_gpu_arrays[8][dataset],
                                 forward_projection_gpu_arrays[1][dataset], forward_projection_gpu_arrays[5][dataset],
                                 forward_projection_gpu_arrays[9][dataset],
                                 forward_projection_gpu_arrays[2][dataset], forward_projection_gpu_arrays[6][dataset],
                                 forward_projection_gpu_arrays[10][dataset],
                                 forward_projection_gpu_arrays[3][dataset], forward_projection_gpu_arrays[7][dataset],
                                 forward_projection_gpu_arrays[11][dataset],
                                 forward_projection_gpu_arrays[12][dataset], fov_cut_matrix_gpu, im_gpu,
                                 forward_projection_gpu_arrays[16][dataset], forward_projection_gpu_arrays[17][dataset],
                                 forward_projection_gpu_arrays[18][dataset], forward_projection_gpu_arrays[19][dataset],
                                 forward_projection_gpu_arrays[20][dataset], forward_projection_gpu_arrays[21][dataset],
                                 block=threadsperblock,
                                 grid=blockspergrid,
                                 stream=stream[dataset]
                                 )

                # Sincronization of streams
                for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                # Transfering data from Optimizer
                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                    cuda.memcpy_dtoh_async(forward_projection_pinned_arrays[12][dataset],
                                           forward_projection_gpu_arrays[12][dataset], stream[dataset])
                    forward_projection_arrays_all_data[12][begin_dataset:end_dataset] = \
                        forward_projection_pinned_arrays[12][dataset]
                #

                #
                cuda.Context.synchronize()
                toc = time.time()
                print('Time part Forward Projection {} : {}'.format(1, toc - tic))
                # # number_of_datasets = np.int32(2)
                teste = np.copy(forward_projection_arrays[12][dataset])
                # # # sum_vor[sum_vor<1]=0
                # # sum_vor = np.ascontiguousarray(teste, dtype=np.float32)
                # # # sum_vor=np.ascontiguousarray(np.ones((self.a.shape)), dtype=np.float32)
                print('SUM VOR: {}'.format(np.sum(teste)))
                # # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)
                #
                # # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)
                t_back_init = time.time()
                # ------------BACKPROJECTION-----------
                cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[12], forward_projection_arrays_all_data[12])

                # ------------BACKPROJECTION-----------

                for dataset in range(number_of_datasets_back):
                    dataset = np.int32(dataset)
                    begin_dataset = np.int32(number_of_events_subset*(dataset)/number_of_datasets_back)
                    end_dataset = np.int32(number_of_events_subset*(dataset+1)/number_of_datasets_back)
                    event[dataset]['kernel_begin'].record(stream[dataset])
                    weight_cutted, height_cutted, depth_cutted = np.int32(
                        adjust_coef.shape[0]), np.int32(
                        1), np.int32(1)

                    number_of_voxels_thread = 128
                    threadsperblock = (np.int(number_of_voxels_thread), 1, 1)
                    # blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
                    blockspergrid_x = int(math.ceil(adjust_coef.shape[0] / threadsperblock[0]))
                    z_min = np.short(np.min(self.z_min_f[begin_dataset:end_dataset]))
                    z_max = np.short(np.max(self.z_max_f[begin_dataset:end_dataset]))

                    blockspergrid_x = int(math.ceil(8*88*88 / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

                    # print(z_min)
                    # print(z_max)
                    # print("---------------")
                    # print(np.int32(len(events_mapping_per_dataset[dataset])))
                    func_backward = self.mod_backward_projection_shared_mem.get_function(
                        "backprojection_cdrf")
                    func_backward(dataset, self.weight, self.height, self.depth,
                                  backward_projection_array_gpu_arrays[13],
                                  backward_projection_array_gpu_arrays[14], half_distance_between_array_pixel,
                                  self.number_of_events, begin_dataset,
                                  np.int32(len(events_mapping_per_dataset[dataset])),
                                  backward_projection_array_gpu_arrays[0],
                                  backward_projection_array_gpu_arrays[4],
                                  backward_projection_array_gpu_arrays[8],
                                  backward_projection_array_gpu_arrays[1],
                                  backward_projection_array_gpu_arrays[5],
                                  backward_projection_array_gpu_arrays[9],
                                  backward_projection_array_gpu_arrays[2],
                                  backward_projection_array_gpu_arrays[6],
                                  backward_projection_array_gpu_arrays[10],
                                  backward_projection_array_gpu_arrays[3],
                                  backward_projection_array_gpu_arrays[7],
                                  backward_projection_array_gpu_arrays[11],
                                  adjust_coef_gpu,
                                  backward_projection_array_gpu_arrays[12],
                                  backward_projection_array_gpu_arrays[15],
                                  z_min, z_max,
                                  backward_projection_array_gpu_arrays[22], backward_projection_array_gpu_arrays[23],
                                  backward_projection_array_gpu_arrays[24], events_mapping_per_dataset_gpu[dataset],
                                  block=threadsperblock,
                                  grid=blockspergrid,
                                  shared=np.int(4 * number_of_voxels_thread*1),
                                  stream=stream[dataset]
                                  )

                for dataset in range(number_of_datasets_back):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                # for dataset in range(number_of_datasets_back):
                #     cuda.memcpy_dtoh_async(adjust_coef_cut[dataset], adjust_coef_gpu[dataset])
                #     adjust_coef[dataset * voxels_division:(dataset + 1) * voxels_division] = adjust_coef_cut[
                #         dataset]
                cuda.memcpy_dtoh_async(adjust_coef, adjust_coef_gpu)
                cuda.Context.synchronize()
                print('Time part Backward Projection {} : {}'.format(1, time.time() - t_back_init))
                print('SUM IMAGE: {}'.format(np.sum(adjust_coef)))
                penalized_term = self._load_penalized_term(im)
                # normalization
                im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
                    normalization_matrix != 0] / (normalization_matrix[normalization_matrix != 0])
                # im[normalization_matrix == 0] = 0

                im = self._apply_penalized_term(im, penalized_term)
                print('SUM IMAGE: {}'.format(np.sum(im)))
                im = np.ascontiguousarray(im, dtype=np.float32)

                cuda.memcpy_htod_async(im_gpu, im)

                # # Clearing variables
                forward_projection_arrays_all_data[12] = np.ascontiguousarray(
                    np.zeros(self.a.shape, dtype=np.float32))

                adjust_coef = np.ascontiguousarray(
                    np.zeros((self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z),
                             dtype=np.float32))

                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                    forward_projection_arrays[12][dataset] = forward_projection_arrays_all_data[12][
                                                             begin_dataset:end_dataset]
                    forward_projection_gpu_arrays[12][dataset] = cuda.mem_alloc(
                        forward_projection_arrays[12][dataset].size * forward_projection_arrays[12][
                            dataset].dtype.itemsize)
                    forward_projection_pinned_arrays[12][dataset] = cuda.register_host_memory(
                        forward_projection_arrays[12][dataset])
                    assert np.all(
                        forward_projection_pinned_arrays[12][dataset] == forward_projection_arrays[12][dataset])
                    cuda.memcpy_htod_async(forward_projection_gpu_arrays[12][dataset],
                                           forward_projection_pinned_arrays[12][dataset], stream[dataset])
                # del my_array
                # del my_object
                # gc.collect()
                # for dataset in range(number_of_datasets_back):
                #     adjust_coef_cut[dataset] = adjust_coef[
                #                                dataset * voxels_division:(dataset + 1) * voxels_division]
                #     adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
                #     assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
                #     cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_pinned[dataset], stream[dataset])

                # if self.saved_image_by_iteration:
                #     im_to_save = im.reshape(weight, height, depth)
                #     self._save_image_by_it(im_to_save, i, sb)
                #
                # if self.signals_interface is not None:
                #     self.signals_interface.trigger_update_label_reconstruction_status.emit(
                #         "{}: Iteration {}".format(self.current_info_step, i + 1))
                #     self.signals_interface.trigger_progress_reconstruction_partial.emit(
                #         int(np.round(100 * (i + 1) * (sb + subsets) / (it * subsets), 0)))

        im = im.reshape(self.weight, self.height, self.depth)
        self.im = im

    def _load_penalized_term(self, im):
        penalized_term = None
        im_c = np.copy(im)
        if self.algorithm == "LM-MRP":
            beta = self.algorithm_options[0]
            kernel_filter_size = self.algorithm_options[1]
            im_to_filter = im_c.reshape(self.weight, self.height, self.depth)
            im_med = median_filter(im_to_filter, kernel_filter_size)
            penalized_term = np.copy(im_to_filter)
            penalized_term[im_med != 0] = 1 + beta * (im_to_filter[im_med != 0] - im_med[im_med != 0]) / \
                                          im_med[
                                              im_med != 0]
            penalized_term = np.ascontiguousarray(penalized_term.reshape(self.weight * self.height * self.depth),
                                                  dtype=np.float32)

        if self.algorithm == "MAP":
            beta = 0.5
            im_map = np.load(
                "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\PhD\\Reconstrução\\NAF+FDG\\Easypet Scan 05 Aug 2019 - 14h 36m 33s\\static_image\\Easypet Scan 05 Aug 2019 - 14h 36m 33s mlem.npy")

        return penalized_term

    def _apply_penalized_term(self, im, penalized_term):
        if self.algorithm == "LM-MRP":
            im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
            return im

    def map_events_gpu(self, min=None, max=None):
        image_cut_limits = np.array([min,max])
        cond = (self.z_min_f > image_cut_limits[1]) & (self.z_max_f > image_cut_limits[1])
        cond_2 = (self.z_min_f < image_cut_limits[0]) & (self.z_max_f < image_cut_limits[0])

        return np.where(~(cond | cond_2))[0]

    def _save_image_by_it(self, im, it=None, sb=None):


        file_name = os.path.join(self.iterations_path, "EasyPETScan_it{}_sb{}".format(it, sb))
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

