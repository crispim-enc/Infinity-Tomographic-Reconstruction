"""
Title: MLEM Host Side
Author: P.M.M.C. Encarnação
Date: 01/14/2023
Description:
"""

import math
import time
import os
from array import array
import numpy as np
from scipy.ndimage import median_filter
from pycuda.compiler import SourceModule


class GPUSharedMemoryMultipleKernel:
    def __init__(self, parent=None, normalizationFlag=False):
        """
        Constructor
        :param parent:
        """
        self.cuda_drv = parent.cuda
        self.number_of_iterations = parent.iterations
        self.number_of_subsets = parent.subsets
        self.mod_forward_projection_shared_mem = None
        self.mod_backward_projection_shared_mem = None
        self.normalizationFlag = normalizationFlag
        self.directory = parent.file_path_output
        self.saved_image_by_iteration = parent.saved_image_by_iteration
        if self.saved_image_by_iteration:

            self.iterations_path = os.path.join(os.path.dirname(self.directory), "iterations")
            if not os.path.isdir(self.iterations_path):
                os.makedirs(self.iterations_path)

        self.planes = parent.projector.planes
        self.countsPerID = parent.projector.countsPerPosition

        self.A = parent.projector.im_index_x
        self.B = parent.projector.im_index_y
        self.C = parent.projector.im_index_z
        self.sum_vor = np.ascontiguousarray(
            np.zeros(self.planes[0][0].shape, dtype=np.float32))

        self.number_of_pixels_x = parent.projector.number_of_pixels_x
        self.number_of_pixels_y = parent.projector.number_of_pixels_y
        self.number_of_pixels_z = parent.projector.number_of_pixels_z

        self.number_of_events = np.int32(len(self.planes[0][0]))
        self.weight = np.int32(self.A.shape[0])
        self.height = np.int32(self.A.shape[1])
        self.depth = np.int32(self.A.shape[2])

        self.algorithm = parent.algorithm
        self.adjust_coef = np.ascontiguousarray(
            np.zeros((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.float32))

        self.sum_pixel = np.ascontiguousarray(
            np.zeros(self.planes[0][0].shape, dtype=np.float32))

        self.im = np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z)) * \
                  len(self.planes[0][0]) / (self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z)

        # self.im = np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z))
        self.im = np.ascontiguousarray(self.im, dtype=np.float32)

        self.normalization_matrix = np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z))
        self.normalization_matrix = np.ascontiguousarray(self.normalization_matrix, dtype=np.float32)

        self.fw_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CT",
                                            "pyramidalProjectorForward.c")

        self.bw_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CT",
                                            "pyramidalProjectorBack.c")

    def _loadMachineCCode(self):
        """
        Load the machine code for the forward and backward projection
        :return:
        """

        self.fw_source_model = open(self.fw_source_model_file)

        self.bw_source_model = open(self.bw_source_model_file)

        self.mod_forward_projection_shared_mem = SourceModule("""{}""".format((self.fw_source_model.read())))
        self.mod_backward_projection_shared_mem = SourceModule("""{}""".format((self.bw_source_model.read())))

    def multipleKernel(self):
        """
        Multiple Kernel MLEM algorithm Host Side

        """
        print('GPU STARTED - Multiple reads')
        # cuda.init()
        cuda = self.cuda_drv

        start_x = np.int32(self.A[0, 0, 0])
        start_y = np.int32(self.B[0, 0, 0])
        start_z = np.int32(self.C[0, 0, 0])
        print("Start_point: {},{},{}".format(start_x, start_y, start_z))
        print('Image size: {},{}, {}'.format(self.weight, self.height, self.depth))

        normalization_matrix = self.normalization_matrix.reshape(
            self.normalization_matrix.shape[0] * self.normalization_matrix.shape[1] * self.normalization_matrix.shape[
                2])

        # SOURCE MODELS (DEVICE CODE)
        self._loadMachineCCode()

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
        adjust_coef_gpu = [None for _ in range(number_of_datasets_back)]
        adjust_coef_pinned = [None for _ in range(number_of_datasets_back)]
        system_matrix_back_cut = [None for _ in range(number_of_datasets_back)]
        system_matrix_back_cut_gpu = [None for _ in range(number_of_datasets_back)]
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
        # fov_cut_matrix_shappened = np.ascontiguousarray(
        #     self.fov_matrix_cut.reshape(
        #         self.fov_matrix_cut.shape[0] * self.fov_matrix_cut.shape[1] * self.fov_matrix_cut.shape[2]),
        #     dtype=np.byte)

        im_gpu = cuda.mem_alloc(im_shappened.size * im_shappened.dtype.itemsize)
        # fov_cut_matrix_gpu = cuda.mem_alloc(fov_cut_matrix_shappened.size * fov_cut_matrix_shappened.dtype.itemsize)

        cuda.memcpy_htod_async(im_gpu, im_shappened)

        system_matrix = np.ascontiguousarray(normalization_matrix, dtype=np.float32)
        system_matrix_gpu = cuda.mem_alloc(system_matrix.size * system_matrix.dtype.itemsize)
        cuda.memcpy_htod_async(system_matrix_gpu, system_matrix)
        # cuda.memcpy_htod_async(fov_cut_matrix_gpu, fov_cut_matrix_shappened)

        # Forward Memory allocation
        # unroll planes

        unroll_planes = [np.ascontiguousarray(self.planes[i,j], dtype=np.float32) for i in range(self.planes.shape[0]) for j in range(self.planes.shape[1])]
        forward_projection_arrays_all_data = unroll_planes + [self.sum_vor]

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
                # cuda.memcpy_htod_async(array_gpu[dataset], array_pinned[dataset], stream[dataset])
                cuda.memcpy_htod_async(array_gpu[dataset], array[dataset], stream[dataset])

            forward_projection_arrays_all_data[ar] = array_original
            forward_projection_arrays[ar] = array
            forward_projection_gpu_arrays[ar] = array_gpu
            # forward_projection_pinned_arrays[ar] = array_pinned

        # Back projection Memory allocation

        backward_projection_arrays_full_arrays = unroll_planes + [self.sum_vor, self.countsPerID]
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
        system_matrix_back = np.ascontiguousarray(normalization_matrix, dtype=np.float32)

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
            system_matrix_back_cut[dataset] = np.ascontiguousarray(system_matrix_back[dataset * voxels_division:(dataset + 1) * voxels_division],
                dtype=np.float32)
            # fov_cut_matrix_cut[dataset] = np.ascontiguousarray(
            #     fov_cut_matrix_shappened[dataset * voxels_division:(dataset + 1) * voxels_division],
            #     dtype=np.byte)

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

            # adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
            # assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
            cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset], stream[dataset])

            # fov_cut_matrix_cutted_gpu[dataset] = cuda.mem_alloc(
            #     fov_cut_matrix_cut[dataset].size * fov_cut_matrix_cut[dataset].dtype.itemsize)
            system_matrix_back_cut_gpu[dataset] = cuda.mem_alloc(
                system_matrix_back_cut[dataset].size * system_matrix_back_cut[dataset].dtype.itemsize)
            A_cut_gpu[dataset] = cuda.mem_alloc(
                A_cut[dataset].size * A_cut[dataset].dtype.itemsize)

            B_cut_gpu[dataset] = cuda.mem_alloc(
                B_cut[dataset].size * B_cut[dataset].dtype.itemsize)

            C_cut_gpu[dataset] = cuda.mem_alloc(
                C_cut[dataset].size * C_cut[dataset].dtype.itemsize)

            cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset])
            cuda.memcpy_htod_async(system_matrix_back_cut_gpu[dataset], system_matrix_back_cut[dataset])
            # cuda.memcpy_htod_async(fov_cut_matrix_cutted_gpu[dataset], fov_cut_matrix_cut[dataset])
            cuda.memcpy_htod_async(A_cut_gpu[dataset], A_cut[dataset])
            cuda.memcpy_htod_async(B_cut_gpu[dataset], B_cut[dataset])
            cuda.memcpy_htod_async(C_cut_gpu[dataset], C_cut[dataset])



        free, total = cuda.mem_get_info()

        print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))
        print('Number events for reconstruction: {}'.format(self.number_of_events))

        # -------------LM_MLEM/OSEM---------
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

                    threadsperblock = (32, 1, 1)
                    blockspergrid_x = int(math.ceil((end_dataset - begin_dataset) / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    event[dataset]['kernel_begin'].record(stream[dataset])

                    func_forward = self.mod_forward_projection_shared_mem.get_function(
                        "forward_projection_cdrf")

                    func_forward(self.weight, self.height, self.depth, start_x, start_y, start_z,
                                 self.number_of_events, begin_dataset, end_dataset,
                                 forward_projection_gpu_arrays[0][dataset], forward_projection_gpu_arrays[1][dataset],
                                 forward_projection_gpu_arrays[2][dataset], forward_projection_gpu_arrays[3][dataset],
                                 forward_projection_gpu_arrays[4][dataset], forward_projection_gpu_arrays[5][dataset],
                                 forward_projection_gpu_arrays[6][dataset], forward_projection_gpu_arrays[7][dataset],
                                 forward_projection_gpu_arrays[8][dataset], forward_projection_gpu_arrays[9][dataset],
                                 forward_projection_gpu_arrays[10][dataset], forward_projection_gpu_arrays[11][dataset],
                                 forward_projection_gpu_arrays[12][dataset], forward_projection_gpu_arrays[13][dataset],
                                 forward_projection_gpu_arrays[14][dataset], forward_projection_gpu_arrays[15][dataset],
                                 forward_projection_gpu_arrays[16][dataset], im_gpu, system_matrix_gpu,
                                 block=threadsperblock,
                                 grid=blockspergrid,
                                 stream=stream[dataset])

                # Sincronization of streams
                for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                # Transfering data from GPU
                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)
                    # cuda.memcpy_dtoh_async(sum_vor[begin_dataset:end_dataset], sum_vor_gpu[dataset])
                    # cuda.memcpy_dtoh_async(forward_projection_pinned_arrays[16][dataset],
                    #                        forward_projection_gpu_arrays[16][dataset], stream[dataset])
                    #
                    # forward_projection_arrays_all_data[16][begin_dataset:end_dataset] = \
                    #     forward_projection_pinned_arrays[16][dataset]

                    cuda.memcpy_dtoh_async(forward_projection_arrays[16][dataset],
                                           forward_projection_gpu_arrays[16][dataset], stream[dataset])
                    forward_projection_arrays_all_data[16][begin_dataset:end_dataset] = \
                        forward_projection_arrays[16][dataset]
                    # cuda.cudaStream.Synchronize(stream[dataset])

                    toc = time.time()

                cuda.Context.synchronize()
                if self.normalizationFlag:
                    forward_projection_arrays_all_data[16] = np.ones_like(forward_projection_arrays_all_data[16])

                # cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)


                print('Time part Forward Projection {} : {}'.format(1, toc - tic))

                print('SUM VOR: {}'.format(np.sum(forward_projection_arrays_all_data[16])))
                print('LEN VOR: {}'.format(
                    len(forward_projection_arrays_all_data[16][forward_projection_arrays_all_data[16] == 0])))
                cuda.memcpy_htod_async(backward_projection_array_gpu_arrays[16], forward_projection_arrays_all_data[16])
                # ------------BACKPROJECTION-----------

                for dataset in range(number_of_datasets_back):
                    dataset = np.int32(dataset)
                    begin_dataset = np.int32(0)
                    end_dataset = np.int32(number_of_events_subset)
                    event[dataset]['kernel_begin'].record(stream[dataset])
                    weight_cutted, height_cutted, depth_cutted = np.int32(
                        adjust_coef_cut[dataset].shape[0]), np.int32(
                        1), np.int32(1)

                    number_of_voxels_thread = 32
                    threadsperblock = (int(number_of_voxels_thread), 1, 1)
                    blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    print("begin_dataset {}".format(begin_dataset))
                    print("end_dataset {}".format(end_dataset))
                    func_backward = self.mod_backward_projection_shared_mem.get_function(
                        "backprojection_cdrf")
                    func_backward(dataset, weight_cutted, height_cutted, depth_cutted,
                                  self.number_of_events, begin_dataset, end_dataset,
                                  backward_projection_array_gpu_arrays[0],
                                  backward_projection_array_gpu_arrays[1],
                                  backward_projection_array_gpu_arrays[2],
                                  backward_projection_array_gpu_arrays[3],
                                  backward_projection_array_gpu_arrays[4],
                                  backward_projection_array_gpu_arrays[5],
                                  backward_projection_array_gpu_arrays[6],
                                  backward_projection_array_gpu_arrays[7],
                                  backward_projection_array_gpu_arrays[8],
                                  backward_projection_array_gpu_arrays[9],
                                  backward_projection_array_gpu_arrays[10],
                                  backward_projection_array_gpu_arrays[11],
                                  backward_projection_array_gpu_arrays[12],
                                  backward_projection_array_gpu_arrays[13],
                                  backward_projection_array_gpu_arrays[14],
                                  backward_projection_array_gpu_arrays[15],
                                  A_cut_gpu[dataset], B_cut_gpu[dataset],
                                  C_cut_gpu[dataset],
                                  adjust_coef_gpu[dataset],
                                  backward_projection_array_gpu_arrays[16], system_matrix_back_cut_gpu[dataset], im_gpu,
                                  backward_projection_array_gpu_arrays[17],
                                  block=threadsperblock,
                                  grid=blockspergrid,
                                  shared=int(4 * number_of_voxels_thread),
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
                print('adjust_coef: {}'.format(np.sum(adjust_coef)))
                penalized_term = self._load_penalized_term(im)
                # normalization

                im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
                    normalization_matrix != 0] / (normalization_matrix[normalization_matrix != 0])
                im[normalization_matrix == 0] = 0
                if self.algorithm == "LM-MRP":
                    im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
                # im = self._apply_penalized_term(im, penalized_term)
                print('SUM IMAGE: {}'.format(np.sum(im)))
                im = np.ascontiguousarray(im, dtype=np.float32)

                cuda.memcpy_htod_async(im_gpu, im)

                # # Clearing variables
                forward_projection_arrays_all_data[16] = np.ascontiguousarray(
                    np.zeros(self.planes[0][0].shape, dtype=np.float32))

                adjust_coef = np.ascontiguousarray(
                    np.zeros((self.number_of_pixels_x * self.number_of_pixels_y * self.number_of_pixels_z),
                             dtype=np.float32))

                for dataset in range(number_of_datasets):
                    begin_dataset = np.int32(dataset * self.number_of_events / number_of_datasets)
                    end_dataset = np.int32((dataset + 1) * self.number_of_events / number_of_datasets)

                    forward_projection_arrays[16][dataset] = forward_projection_arrays_all_data[16][
                                                             begin_dataset:end_dataset]
                    forward_projection_gpu_arrays[16][dataset] = cuda.mem_alloc(
                        forward_projection_arrays[16][dataset].size * forward_projection_arrays[16][
                            dataset].dtype.itemsize)
                    # forward_projection_pinned_arrays[16][dataset] = cuda.register_host_memory(
                    #     forward_projection_arrays[16][dataset])
                    # assert np.all(
                    #     forward_projection_pinned_arrays[16][dataset] == forward_projection_arrays[16][dataset])
                    # cuda.memcpy_htod_async(forward_projection_gpu_arrays[16][dataset],
                    #                        forward_projection_pinned_arrays[16][dataset], stream[dataset])
                    cuda.memcpy_htod_async(forward_projection_gpu_arrays[16][dataset],
                                           forward_projection_arrays[16][dataset], stream[dataset])

                for dataset in range(number_of_datasets_back):
                    adjust_coef_cut[dataset] = adjust_coef[
                                               dataset * voxels_division:(dataset + 1) * voxels_division]
                    # adjust_coef_pinned[dataset] = cuda.register_host_memory(adjust_coef_cut[dataset])
                    # assert np.all(adjust_coef_pinned[dataset] == adjust_coef_cut[dataset])
                    cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset], stream[dataset])

                if self.saved_image_by_iteration:
                    if i % 2 == 0:
                        im_to_save = im.reshape(self.weight, self.height, self.depth)
                        self._save_image_by_it(im_to_save, i, sb)

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

        return penalized_term

    def _apply_penalized_term(self, im, penalized_term):
        if self.algorithm == "LM-MRP":
            im[penalized_term != 0] = im[penalized_term != 0] / penalized_term[penalized_term != 0]
            return im

    def map_events_gpu(self, min=None, max=None):
        image_cut_limits = np.array([min, max])
        cond = (self.z_min_f > image_cut_limits[1]) & (self.z_max_f > image_cut_limits[1])
        cond_2 = (self.z_min_f < image_cut_limits[0]) & (self.z_max_f < image_cut_limits[0])

        return np.where(~(cond | cond_2))[0]

    def _save_image_by_it(self, im, it=None, sb=None):

        file_name = os.path.join(self.iterations_path, "_it{}_sb{}".format(it, sb))
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
