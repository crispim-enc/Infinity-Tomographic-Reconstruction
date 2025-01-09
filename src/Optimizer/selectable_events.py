import numpy as np
import os
from array import array
import time
import math
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt


class ROIEvents:
    def __init__(self, EM_obj):
        self.cuda_drv = EM_obj.cuda_drv
        self.EM_obj = EM_obj
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
        self.im = EM_obj.im
        self.half_crystal_pitch_xy = EM_obj.half_crystal_pitch_xy
        # self.half_crystal_pitch_xy = np.float32(2.0)
        self.half_crystal_pitch_z = EM_obj.half_crystal_pitch_z
        # self.half_crystal_pitch_z = np.float32(2.0)
        self.sum_vor = EM_obj.sum_pixel
        self.active_x = EM_obj.active_pixel_x
        self.active_y = EM_obj.active_pixel_y
        self.active_z = EM_obj.active_pixel_z
        self.distance_between_array_pixel = EM_obj.distance_between_array_pixel
        self.distance_to_center_plane = EM_obj.distance_to_center_plane
        self.distance_to_center_plane_normal = EM_obj.distance_to_center_plane_normal

        self.A_gpu = None
        self.B_gpu = None
        self.C_gpu = None
        self.im_gpu = None
        self.active_x_gpu = None
        self.active_y_gpu = None
        self.active_z_gpu = None
        self.mod_pixel2pos = None
        self.valid_vor = None

        self.weight = np.int32(self.A.shape[0])
        self.height = np.int32(self.A.shape[1])
        self.depth = np.int32(self.A.shape[2])

        self.pixel2pos_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machine_code", "mod_pixel2pos.c")
        self.pixel2pos_source_model = open(self.pixel2pos_source_model_file)
        self._load_machine_C_code()

    def _load_machine_C_code(self):
        self.mod_pixel2pos = SourceModule("""{}""".format((self.pixel2pos_source_model.read())))

    def pixel2Position(self ):
        print('Optimizer-Pixel2Position')
        cuda = self.cuda_drv
        # device = cuda.Device(0)  # enter your gpu id here
        # ctx = device.make_context()
        self.A = np.ascontiguousarray(self.A.reshape(
            self.A.shape[0] * self.A.shape[1] * self.A.shape[2]),
            dtype=np.int32)
        self.B = np.ascontiguousarray(self.B.reshape(
            self.B.shape[0] * self.B.shape[1] * self.B.shape[2]),
            dtype=np.int32)
        self.C = np.ascontiguousarray(self.C.reshape(
            self.C.shape[0] * self.C.shape[1] * self.C.shape[2]),
            dtype=np.int32)

        self.im = np.ascontiguousarray(self.im.reshape(
            self.im.shape[0] * self.im.shape[1] * self.im.shape[2]),
            dtype=np.float32)

        number_of_events = np.int32(len(self.a))

        number_of_datasets = np.int32(1)  # Number of datasets (and concurrent operations) used.
        # Event as reference point
        ref = cuda.Event()
        ref.record()

        # Create the streams and events needed to calculation
        stream, event = [], []
        marker_names = ['kernel_begin', 'kernel_end']
        # Create List to allocate chunks of data

        for dataset in range(number_of_datasets):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))

        one_alloc_original_array = [self.A, self.B, self.C, self.im, self.active_x, self.active_y, self.active_z]
        one_alloc_gpu_array = [None]*len(one_alloc_original_array)
        for st in range(len(one_alloc_original_array)):
            one_alloc_gpu_array[st] = cuda.mem_alloc(one_alloc_original_array[st].size * one_alloc_original_array[st].dtype.itemsize)
            cuda.memcpy_htod_async(one_alloc_gpu_array[st], one_alloc_original_array[st])

        # self.A_gpu = cuda.mem_alloc(self.A.size * self.A.dtype.itemsize)
        # self.B_gpu = cuda.mem_alloc(self.B.size * self.B.dtype.itemsize)
        # self.C_gpu = cuda.mem_alloc(self.C.size * self.C.dtype.itemsize)
        # self.im_gpu = cuda.mem_alloc(self.im.size * self.im.dtype.itemsize)
        # self.active_x_gpu = cuda.mem_alloc(self.active_x.size * self.active_x.dtype.itemsize)
        # self.active_y_gpu = cuda.mem_alloc(self.active_y.size * self.active_y.dtype.itemsize)
        # self.active_z_gpu = cuda.mem_alloc(self.active_z.size * self.active_z.dtype.itemsize)
        #
        # cuda.memcpy_htod_async(self.A_gpu, self.A)
        # cuda.memcpy_htod_async(self.B_gpu, self.B)
        # cuda.memcpy_htod_async(self.C_gpu, self.C)
        # cuda.memcpy_htod_async(self.im_gpu, self.im)
        # cuda.memcpy_htod_async(self.active_x_gpu, self.active_x)
        # cuda.memcpy_htod_async(self.active_y_gpu, self.active_y)
        # cuda.memcpy_htod_async(self.active_z_gpu, self.active_z)

        partial_allocations_original_array = [self.a, self.b, self.c, self.d,
                                              self.a_normal, self.b_normal, self.c_normal,self.d_normal,
                                              self.a_cf, self.b_cf, self.c_cf, self.d_cf, self.sum_vor,
                                              self.distance_to_center_plane, self.distance_to_center_plane_normal]
        # partial_allocations_cutted_array = [a_cut, b_cut,c_cut, d_cut,
        #                        a_normal_cut, b_normal_cut, c_normal_cut, d_normal_cut,
        #                        a_cf_cut, b_cf_cut, c_cf_cut, d_cf_cut, sum_vor_cut]
        #
        partial_allocations_cutted_array = [[None] * number_of_datasets for _ in range(len(partial_allocations_original_array))]

        # partial_allocations_cutted_array_gpu = [a_cut_gpu, b_cut_gpu, c_cut_gpu, d_cut_gpu,
        #                                     a_cut_normal_gpu, b_cut_normal_gpu, c_cut_normal_gpu, d_cut_normal_gpu,
        #                                     a_cf_cut_gpu, b_cf_cut_gpu, c_cf_cut_gpu, d_cf_cut_gpu, sum_vor_gpu]

        partial_allocations_cutted_array_gpu = [[None] * number_of_datasets for _ in
                                            range(len(partial_allocations_original_array))]
        for dataset in range(number_of_datasets):
            begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
            end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
            for st in range(len(partial_allocations_cutted_array)):
                partial_allocations_cutted_array[st][dataset] = partial_allocations_original_array[st][begin_dataset:end_dataset]
                partial_allocations_cutted_array_gpu[st][dataset] = cuda.mem_alloc(partial_allocations_cutted_array[st][dataset].size * partial_allocations_cutted_array[st][dataset].dtype.itemsize)
                cuda.memcpy_htod_async(partial_allocations_cutted_array_gpu[st][dataset], partial_allocations_cutted_array[st][dataset])
            # Cutting dataset
            # For forward projection the data is cutted by number of events. For backprojection is cutted per image pieces of image
            # a_cut[dataset] = self.a[begin_dataset:end_dataset]
            # b_cut[dataset] = self.b[begin_dataset:end_dataset]
            # c_cut[dataset] = self.c[begin_dataset:end_dataset]
            # d_cut[dataset] = d[begin_dataset:end_dataset]
            # a_normal_cut[dataset] = a_normal[begin_dataset:end_dataset]
            # b_normal_cut[dataset] = b_normal[begin_dataset:end_dataset]
            # c_normal_cut[dataset] = c_normal[begin_dataset:end_dataset]
            # d_normal_cut[dataset] = d_normal[begin_dataset:end_dataset]
            #
            # a_cf_cut[dataset] = a_cf[begin_dataset:end_dataset]
            # b_cf_cut[dataset] = b_cf[begin_dataset:end_dataset]
            # c_cf_cut[dataset] = c_cf[begin_dataset:end_dataset]
            # d_cf_cut[dataset] = d_cf[begin_dataset:end_dataset]
            # sum_vor_cut[dataset] = sum_vor[begin_dataset:end_dataset]

            # Forward
            # a_cut_gpu[dataset] = cuda.mem_alloc(a_cut[dataset].size * a_cut[dataset].dtype.itemsize)
            # b_cut_gpu[dataset] = cuda.mem_alloc(b_cut[dataset].size * b_cut[dataset].dtype.itemsize)
            # c_cut_gpu[dataset] = cuda.mem_alloc(c_cut[dataset].size * c_cut[dataset].dtype.itemsize)
            # d_cut_gpu[dataset] = cuda.mem_alloc(d_cut[dataset].size * d_cut[dataset].dtype.itemsize)
            # a_cut_normal_gpu[dataset] = cuda.mem_alloc(
            #     a_normal_cut[dataset].size * a_normal_cut[dataset].dtype.itemsize)
            # b_cut_normal_gpu[dataset] = cuda.mem_alloc(
            #     b_normal_cut[dataset].size * b_normal_cut[dataset].dtype.itemsize)
            # c_cut_normal_gpu[dataset] = cuda.mem_alloc(
            #     c_normal_cut[dataset].size * c_normal_cut[dataset].dtype.itemsize)
            # d_cut_normal_gpu[dataset] = cuda.mem_alloc(
            #     d_normal_cut[dataset].size * d_normal_cut[dataset].dtype.itemsize)
            #
            # a_cf_cut_gpu[dataset] = cuda.mem_alloc(a_cf_cut[dataset].size * a_cf_cut[dataset].dtype.itemsize)
            # b_cf_cut_gpu[dataset] = cuda.mem_alloc(b_cf_cut[dataset].size * b_cf_cut[dataset].dtype.itemsize)
            # c_cf_cut_gpu[dataset] = cuda.mem_alloc(c_cf_cut[dataset].size * c_cf_cut[dataset].dtype.itemsize)
            # d_cf_cut_gpu[dataset] = cuda.mem_alloc(d_cf_cut[dataset].size * d_cf_cut[dataset].dtype.itemsize)
            #
            # sum_vor_gpu[dataset] = cuda.mem_alloc(sum_vor_cut[dataset].size * sum_vor_cut[dataset].dtype.itemsize)
            #
            # cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_cut[dataset])
            #
            # cuda.memcpy_htod_async(a_cut_gpu[dataset], a_cut[dataset])
            # cuda.memcpy_htod_async(b_cut_gpu[dataset], b_cut[dataset])
            # cuda.memcpy_htod_async(c_cut_gpu[dataset], c_cut[dataset])
            # cuda.memcpy_htod_async(d_cut_gpu[dataset], d_cut[dataset])
            # cuda.memcpy_htod_async(a_cut_normal_gpu[dataset], a_normal_cut[dataset])
            # cuda.memcpy_htod_async(b_cut_normal_gpu[dataset], b_normal_cut[dataset])
            # cuda.memcpy_htod_async(c_cut_normal_gpu[dataset], c_normal_cut[dataset])
            # cuda.memcpy_htod_async(d_cut_normal_gpu[dataset], d_normal_cut[dataset])
            #
            # cuda.memcpy_htod_async(a_cf_cut_gpu[dataset], a_cf_cut[dataset])
            # cuda.memcpy_htod_async(b_cf_cut_gpu[dataset], b_cf_cut[dataset])
            # cuda.memcpy_htod_async(c_cf_cut_gpu[dataset], c_cf_cut[dataset])
            # cuda.memcpy_htod_async(d_cf_cut_gpu[dataset], d_cf_cut[dataset])

            # Backprojection
        tic = time.time()
        number_of_active_pixels = np.int32(len(self.active_x))
        for dataset in range(number_of_datasets):
            begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
            end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)

            threadsperblock = (256, 1, 1)
            blockspergrid_x = int(math.ceil(((end_dataset - begin_dataset)) / threadsperblock[0]))
            blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
            blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
            blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
            event[dataset]['kernel_begin'].record(stream[dataset])

            func_forward = self.mod_pixel2pos.get_function("pixeltoangle")
            func_forward(self.weight, self.height, self.depth,  partial_allocations_cutted_array_gpu[13][dataset],
                         partial_allocations_cutted_array_gpu[14][dataset],
                         self.distance_between_array_pixel,
                         number_of_events, begin_dataset, end_dataset,
                         partial_allocations_cutted_array_gpu[0][dataset], partial_allocations_cutted_array_gpu[4][dataset],
                         partial_allocations_cutted_array_gpu[8][dataset],
                         partial_allocations_cutted_array_gpu[1][dataset], partial_allocations_cutted_array_gpu[5][dataset],
                         partial_allocations_cutted_array_gpu[9][dataset],
                         partial_allocations_cutted_array_gpu[2][dataset], partial_allocations_cutted_array_gpu[6][dataset],
                         partial_allocations_cutted_array_gpu[10][dataset],
                         partial_allocations_cutted_array_gpu[3][dataset], partial_allocations_cutted_array_gpu[7][dataset],
                         partial_allocations_cutted_array_gpu[11][dataset],
                         one_alloc_gpu_array[0], one_alloc_gpu_array[1], one_alloc_gpu_array[2],
                         partial_allocations_cutted_array_gpu[12][dataset],
                         one_alloc_gpu_array[3], one_alloc_gpu_array[4], one_alloc_gpu_array[5], one_alloc_gpu_array[6],
                         number_of_active_pixels,
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
            cuda.memcpy_dtoh_async(self.sum_vor, partial_allocations_cutted_array_gpu[12][dataset])
            toc = time.time()
        print('Time part Forward Projection {} : {}'.format(1, toc - tic))
        cuda.memcpy_dtoh_async(self.im, one_alloc_gpu_array[3])
        print('IM: {}'.format(np.sum(self.im)))
        # free, total = cuda.mem_get_info()
        print('Sum_active_pixels: {}'.format(np.sum(self.sum_vor)))
        valid_vor = np.copy(self.sum_vor)
        valid_vor[valid_vor > 0] = 1
        self.save_gpu_results()
        self.valid_vor = valid_vor
        # self.generate_roi_listmode_data()
        return valid_vor

    def generate_roi_listmode_data(self):
        roi_listmode = self.EM_obj.easypetdata.listMode[self.valid_vor == 1]
        # crystal_geometry = self.crystal_geometry
        # self.path_data_validation = os.path.join(os.path.dirname(self.study_file), "Data_Validation")
        # if not os.path.isdir(self.path_data_validation):
        #     os.makedirs(self.path_data_validation)

        f_energy_corrected, ((ax_EA, ax_EB)) = plt.subplots(1, 2)
        f_energy_corrected.suptitle('Energy Cut (keV)', fontsize=16)
        # f_individuals
        f_ids, ((ax_idA, ax_idB)) = plt.subplots(1, 2)
        f_ids.suptitle('Crystal ID', fontsize=16)
        f_motor_c, ((ax_top_c, ax_bot_c)) = plt.subplots(1, 2)
        f_motor_c.suptitle('Motors detection angles (.easypet)', fontsize=16)
        f_time, ax_time = plt.subplots(1, 1)


        u_EA, indices_EA = np.unique(roi_listmode[:, 0], return_index=True)
        u_EB, indices_EB = np.unique(roi_listmode[:, 1], return_index=True)
        u_idA, indices_idA = np.unique(roi_listmode[:, 2], return_index=True)
        u_idB, indices_idB = np.unique(roi_listmode[:, 3], return_index=True)
        u_bot_c, indices_bot = np.unique(roi_listmode[:, 4], return_index=True)
        u_c, indices = np.unique(roi_listmode[:, 5], return_index=True)
        u_time, indices_time = np.unique(roi_listmode[:, 6], return_index=True)

        ax_EA.hist(roi_listmode[:, 0], u_EA, [0, 1200])
        ax_EA.set_xlabel("KeV")
        ax_EA.set_ylabel("Counts")
        ax_EB.hist(roi_listmode[:, 1], u_EB, [0, 1200])
        ax_idA.hist(roi_listmode[:, 2], len(u_idA) + 1, [np.min(roi_listmode[:, 2]), np.max(roi_listmode[:, 2]) + 1])
        ax_idB.hist(roi_listmode[:, 3], len(u_idB) + 1, [np.min(roi_listmode[:, 3]), np.max(roi_listmode[:, 3]) + 1])
        ax_bot_c.hist(roi_listmode[:, 4], u_bot_c)
        ax_top_c.hist(roi_listmode[:, 5], u_c)
        ax_time.hist(roi_listmode[:, 6], u_time)
        plt.show()

    def save_gpu_results(self):
        im = self.im.reshape(self.weight, self.height, self.depth)
        volume = im.astype(np.float32)
        length = volume.shape[0] * volume.shape[2] * volume.shape[1]

        data = np.reshape(volume, [1, length], order='F')
        shapeIm = volume.shape
        file_name_im = os.path.join(self.EM_obj.directory, "gpu_surface")
        file_name_selected_events = os.path.join(self.EM_obj.directory, "sum_vor")
        np.save(file_name_selected_events, self.sum_vor)
        output_file = open(file_name_im, 'wb')
        arr = array('f', data[0])
        # arr = array('d', data[0])
        arr.tofile(output_file)
        output_file.close()