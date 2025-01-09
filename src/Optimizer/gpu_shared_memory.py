import numpy as np
from pycuda.compiler import SourceModule
import math
import time
import os


class GPUSharedMemorySingleKernel:
    def __init__(self, EM_obj=None):
        self.mod_forward_projection_shared_mem = None
        self.mod_backward_projection_shared_mem = None
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

        self.adjust_coef = EM_obj.adjust_coef
        self.im = EM_obj.im
        self.half_crystal_pitch_xy = EM_obj.half_crystal_pitch_xy

        self.fw_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machine_code", "fw_single_kernel.c")
        self.fw_source_model = open(self.fw_source_model_file)
        self.bw_source_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "machine_code", "bw_single_kernel.c")
        self.bw_source_model = open(self.fw_source_model_file)

    def _load_machine_C_code(self):
        self.mod_forward_projection_shared_mem = SourceModule(self.fw_source_model)
        self.mod_backward_projection_shared_mem = SourceModule(self.bw_source_model)
        # self.mod_forward_projection_shared_mem = SourceModule("""__global__ void forward_projection
        #                                                                      (const int n, const int m, const int p, const float crystal_pitch_XY,const float crystal_pitch_Z,const float distance_between_array_pixel,int number_of_events, const int begin_event_gpu_limitation,  const int end_event_gpu_limitation, float *a, float *a_normal,
        #                                                                       float *a_cf, float *b,float *b_normal, float *b_cf,float *c,float *c_normal, float *c_cf, float *d, float *d_normal, float *d_cf, const int *A, const int *B,
        #                                                                       const int *C, float *sum_vor, const char *sensitivity_matrix, const float *im_old, const float *probability)
        #                                                 {
        #                                                    const int shared_memory_size = 256;
        #                                                     __shared__ float a_shared[shared_memory_size];
        #                                                     __shared__ float b_shared[shared_memory_size];
        #                                                     __shared__ float c_shared[shared_memory_size];
        #                                                     __shared__ float d_shared[shared_memory_size];
        #                                                     __shared__ float a_normal_shared[shared_memory_size];
        #                                                     __shared__ float b_normal_shared[shared_memory_size];
        #                                                     __shared__ float c_normal_shared[shared_memory_size];
        #                                                     __shared__ float d_normal_shared[shared_memory_size];
        #                                                     __shared__ float a_cf_shared[shared_memory_size];
        #                                                     __shared__ float b_cf_shared[shared_memory_size];
        #                                                     __shared__ float c_cf_shared[shared_memory_size];
        #
        #                                                     __shared__ float d_cf_shared[shared_memory_size];
        #                                                     __shared__ float sum_vor_shared[shared_memory_size];
        #                                                     /*
        #                                                    const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
        #                                                    const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
        #                                                    */
        #                                                    int threadId=blockIdx.x *blockDim.x + threadIdx.x;
        #                                                    int e;
        #
        #                                                    float d2;
        #                                                    float d2_normal;
        #                                                    float d2_cf;
        #                                                    float value;
        #                                                    float value_normal;
        #                                                    float value_cf;
        #                                                    float sum_vor_temp;
        #                                                    int index;
        #                                                    const float total_length =sqrt(4*crystal_pitch_Z*crystal_pitch_Z+4*crystal_pitch_XY*crystal_pitch_XY+distance_between_array_pixel*distance_between_array_pixel/4);
        #
        #
        #
        #                                                    const int number_events_max = end_event_gpu_limitation-begin_event_gpu_limitation;
        #                                                    const float error_pixel = 0.0000f;
        #                                                     if (threadIdx.x>shared_memory_size)
        #                                                     {
        #                                                     return;
        #                                                     }
        #                                                     if(threadId>=number_events_max)
        #                                                     {
        #                                                     return;
        #                                                     }
        #
        #                                                    __syncthreads();
        #                                                    e = threadId;
        #                                                    int e_m = threadIdx.x;
        #                                                    a_shared[e_m] = a[e];
        #                                                    b_shared[e_m] = b[e];
        #                                                    c_shared[e_m] = c[e];
        #                                                    d_shared[e_m] = d[e];
        #                                                    a_normal_shared[e_m] = a_normal[e];
        #                                                    b_normal_shared[e_m] = b_normal[e];
        #                                                    c_normal_shared[e_m] = c_normal[e];
        #                                                    d_normal_shared[e_m] = d_normal[e];
        #                                                    a_cf_shared[e_m] = a_cf[e];
        #                                                    b_cf_shared[e_m] = b_cf[e];
        #                                                    c_cf_shared[e_m] = c_cf[e];
        #                                                    d_cf_shared[e_m] = d_cf[e];
        #
        #                                                    sum_vor_shared[e_m] = sum_vor[e];
        #
        #
        #                                                    for(int k=0; k<p; k++)
        #                                                    {
        #                                                        for(int j=0; j<m; j++)
        #                                                        {
        #                                                             for(int l=0; l<n; l++)
        #                                                                {
        #                                                                /*index = k+j*p+l*p*m;*/
        #                                                                index = l+j*n+k*m*n;
        #
        #
        #                                                                if (sensitivity_matrix[index]!=0)
        #                                                                {
        #                                                                value = a_shared[e_m]*A[index]+b_shared[e_m]*B[index]+c_shared[e_m]*C[index]- d_shared[e_m];
        #                                                                d2 = error_pixel+crystal_pitch_Z * sqrt(a_shared[e_m]*a_shared[e_m] + b_shared[e_m]*b_shared[e_m] + c_shared[e_m]*c_shared[e_m]);
        #                                                                if (value < d2 && value >=-d2 )
        #                                                                {
        #                                                                    value_normal = a_normal_shared[e_m]*A[index]+b_normal_shared[e_m]*B[index]+c_normal_shared[e_m] * C[index]-d_normal_shared[e_m];
        #                                                                    d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal_shared[e_m]*a_normal_shared[e_m]+b_normal_shared[e_m]*b_normal_shared[e_m]+c_normal_shared[e_m]*c_normal_shared[e_m]);
        #
        #
        #                                                                     if (value_normal < d2_normal && value_normal >= -d2_normal)
        #
        #                                                                     {
        #                                                                            value_cf = a_cf_shared[e_m]*A[index]+b_cf_shared[e_m]*B[index]+c_cf_shared[e_m] * C[index]-d_cf_shared[e_m];
        #                                                                             d2_cf = (distance_between_array_pixel/2)*sqrt(a_cf_shared[e_m]*a_cf_shared[e_m]+b_cf_shared[e_m]*b_cf_shared[e_m]+c_cf_shared[e_m]*c_cf_shared[e_m]);
        #
        #                                                                        if ((value_cf > -d2_cf) && value_cf<=d2_cf)
        #                                                                            {
        #                                                                             sum_vor_shared[e_m] += im_old[index]*(1-(sqrt(value*value+value_normal*value_normal+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf));
        #                                                                              }
        #                                                                             /*
        #
        #                                                                                sum_vor_temp += im_old[index]*(1-(sqrt(value*value+value_normal*value_normal+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf));
        #                                                                             sum_vor_temp += im_old[index]*(1-(sqrt(value*value+value_normal*value_normal+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+distance_between_array_pixel*d2_cf*distance_between_array_pixel*d2_cf/4));
        #                                                                                if ( (value_cf>=0 && value_cf < distance_between_array_pixel*d2_cf && value_cf>=d2_cf) || (value_cf<0 && value_cf > -distance_between_array_pixel*d2_cf && value_cf<=d2_cf))
        #                                                                        {
        #                                                                        }
        #                                                                        *(1-abs(distance_between_array_pixel/2-value_cf)/distance_between_array_pixel/2)
        #
        #                                                                             *(1-((sqrt(value*value+value_normal*value_normal+4*crystal_pitch_XY*crystal_pitch_XY))/sqrt(4*crystal_pitch_Z*crystal_pitch_Z+4*crystal_pitch_XY*crystal_pitch_XY+distance_between_array_pixel*distance_between_array_pixel)));
        #                                                                               sum_vor_shared[e_m] += im_old[index]*(1-((sqrt(value*value+value_normal*value_normal+4*crystal_pitch_XY*crystal_pitch_XY))/sqrt(4*crystal_pitch_Z*crystal_pitch_Z+4*crystal_pitch_XY*crystal_pitch_XY+distance_between_array_pixel*distance_between_array_pixel)));;
        #                                                                             sum_vor_shared[e_m] += im_old[index]*sensitivity_matrix[index]*(1-((sqrt(value*value+value_normal*value_normal))/sqrt(crystal_pitch_Z*crystal_pitch_Z+crystal_pitch_XY*crystal_pitch_XY+)));;
        #                                                                             sum_vor_shared[e_m] += im_old[index];
        #                                                                             sum_vor_temp += sensitivity_matrix[index]*im_old[index]*(1-(sqrt(value*value+value_normal*value_normal+4*crystal_pitch_XY*crystal_pitch_XY))/total_length);
        #                                                                             *(1-(sqrt(value*value+value_normal*value_normal+value_cf*value_cf))/total_length)
        #                                                                             (1-abs(value_cf)/(distance_between_array_pixel/2*d2_cf));
        #
        #                                                                             ;
        #
        #
        #
        #                                                                         }
        #                                                                          */
        #
        #                                                                    }
        #
        #
        #
        #
        #                                                                }
        #                                                                }
        #
        #                                                             }
        #                                                         }
        #                                                     }
        #
        #                                                     /*
        #                                                     sum_vor[e]= sum_vor_shared[e_m];
        #                                                     sum_vor[e]= sum_vor_temp;
        #                                                     */
        #
        #                                                     __syncthreads();
        #                                                     sum_vor[e]= sum_vor_shared[e_m];
        #
        #
        #
        #                                 }""")
        # self.mod_backward_projection_shared_mem = SourceModule("""
        #
        #                             __global__ void backprojection
        #                              ( int dataset_number, int n, int m, int p, const float crystal_pitch_XY, const float crystal_pitch_Z, const float distance_between_array_pixel, int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation, float *a, float *a_normal,
        #                               float *a_cf, float *b,float *b_normal, float *b_cf,float *c,float *c_normal,  float *c_cf, float *d, float *d_normal,  float *d_cf,const int *A, const int *B,
        #                               const int *C, float *adjust_coef, float *sum_vor, char *sensitivity_matrix, const float *probability, float *time_factor)
        #                            {
        #
        #                                /*
        #
        #                                int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
        #                                int idt= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
        #
        #
        #
        #
        #                                const int shared_memory_size = 143;
        #                                __shared__ float adjust_coef_shared[shared_memory_size];
        #
        #                                */
        #                                extern __shared__ float adjust_coef_shared[];
        #                                int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
        #                                int idt = blockId * blockDim.x + threadIdx.x;
        #
        #
        #
        #
        #                                float d2;
        #                                float d2_normal;
        #                                float d2_cf;
        #                                float normal_value;
        #                                float value;
        #                                float value_cf;
        #                                float error_pixel = 0.000f;
        #                                float declive;
        #                                const float total_length = sqrt(4*crystal_pitch_Z*crystal_pitch_Z+4*crystal_pitch_XY*crystal_pitch_XY+distance_between_array_pixel*distance_between_array_pixel/4);
        #
        #
        #                                int i_s=threadIdx.x;
        #
        #                                if (idt>=n*m*p)
        #                                {
        #                                     return;
        #                                 }
        #                                 if (i_s>n)
        #                                {
        #                                     return;
        #                                 }
        #
        #
        #                                 __syncthreads();
        #                                 adjust_coef_shared[i_s] = adjust_coef[idt];
        #
        #                                for(int e=begin_event_gpu_limitation; e<end_event_gpu_limitation; e++)
        #                                {
        #                                        if (sensitivity_matrix[idt]!=0)
        #                                                                {
        #
        #                                        value = a[e]*A[idt]+b[e]*B[idt]+c[e]*C[idt]- d[e];
        #                                        d2 = error_pixel+ crystal_pitch_Z * sqrt(a[e]*a[e] + b[e]*b[e] + c[e]*c[e]);
        #
        #
        #                                        if (value < d2 && value >=-d2)
        #                                        {
        #                                            normal_value= a_normal[e]*A[idt]+b_normal[e]*B[idt]+c_normal[e] * C[idt]-d_normal[e];
        #
        #                                            d2_normal = error_pixel+crystal_pitch_XY * sqrt(a_normal[e]*a_normal[e]+b_normal[e]*b_normal[e]+c_normal[e]*c_normal[e]);
        #
        #
        #                                             if (normal_value< d2_normal && normal_value >= -d2_normal)
        #                                             {
        #
        #                                                  value_cf = a_cf[e]*A[idt]+b_cf[e]*B[idt]+c_cf[e] * C[idt]-d_cf[e];
        #                                                  d2_cf = distance_between_array_pixel/2*sqrt(a_cf[e]*a_cf[e]+b_cf[e]*b_cf[e]+c_cf[e]*c_cf[e]);
        #
        #
        #                                                    if ((value_cf > -d2_cf) && value_cf<=d2_cf)
        #                                                    {
        #                                                         adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]*time_factor[e]);
        #
        #
        #
        #                                                   }
        #
        #                                                        /*
        #                                                      adjust_coef_shared[i_s] += (1-(sqrt(value*value+normal_value*normal_value+value_cf*value_cf))/sqrt(d2*d2+d2_normal*d2_normal+d2_cf*d2_cf))/(sum_vor[e]*time_factor[e]);
        #                                                         */
        #
        #
        #
        #
        #
        #                                             }
        #                                         }
        #
        #                                }
        #
        #                                }
        #
        #                                adjust_coef[idt] = adjust_coef_shared[i_s];
        #                                __syncthreads();
        #
        #
        #
        #                            }
        #                            """)

    def _vor_design_gpu_shared_memory(self):
        print('Optimizer STARTED')
        # cuda.init()
        cuda = self.EM_obj.cuda_drv
        # device = cuda.Device(0)  # enter your gpu id here
        # ctx = device.make_context()
        number_of_events = np.int32(len(a))
        weight = np.int32(A.shape[0])
        height = np.int32(A.shape[1])
        depth = np.int32(A.shape[2])
        print('Image size: {},{}, {}'.format(weight, height, depth))
        print('BACKPROJECTION FUNCTION')
        # probability= self.probability
        probability = np.ascontiguousarray(
            np.ones((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z), dtype=np.float32))
        distance_between_array_pixel = self.distance_between_array_pixel

        # SOURCE MODELS (DEVICE CODE)
        self._load_machine_C_code()
        # float crystal_pitch, int number_of_events, float *a, float *b, float *c, float *d, int *A, int *B, int *C, float *im, float *vector_matrix
        # Host Code   B, C, im, vector_matrix,
        number_of_datasets = np.int32(1)  # Number of datasets (and concurrent operations) used.
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
        probability_cut = [None] * number_of_datasets
        probability_gpu = [None] * number_of_datasets
        adjust_coef_cut = [None] * number_of_datasets
        adjust_coef_gpu = [None] * number_of_datasets
        sensivity_matrix_cut = [None] * number_of_datasets
        sensivity_cut = [None] * number_of_datasets
        sensivity_gpu = [None] * number_of_datasets

        # Streams and Events creation
        for dataset in range(number_of_datasets):
            stream.append(cuda.Stream())
            event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))

        # Memory Allocation
        # Variables that need an unique alocation
        A_gpu = cuda.mem_alloc(A.size * A.dtype.itemsize)
        B_gpu = cuda.mem_alloc(B.size * B.dtype.itemsize)
        C_gpu = cuda.mem_alloc(C.size * C.dtype.itemsize)
        im_gpu = cuda.mem_alloc(im.size * im.dtype.itemsize)
        sensivity_matrix_gpu = cuda.mem_alloc(sensivity_matrix.size * sensivity_matrix.dtype.itemsize)

        cuda.memcpy_htod_async(A_gpu, A)
        cuda.memcpy_htod_async(B_gpu, B)
        cuda.memcpy_htod_async(C_gpu, C)
        cuda.memcpy_htod_async(im_gpu, im)
        cuda.memcpy_htod_async(sensivity_matrix_gpu, sensivity_matrix)

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
        probability_t_gpu = cuda.mem_alloc(probability.size * probability.dtype.itemsize)
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

        im_cut_dim = [im.shape[0], im.shape[1], int(im.shape[2] / number_of_datasets)]  # Dividing image in small chunks
        for dataset in range(number_of_datasets):
            if dataset == number_of_datasets:
                begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                end_dataset = number_of_events
                adjust_coef_cut[dataset] = np.ascontiguousarray(
                    adjust_coef[int(np.floor(im_cut_dim[0] * dataset)):adjust_coef.shape[0],
                    int(np.floor(im_cut_dim[1] * dataset)):adjust_coef.shape[1],
                    int(np.floor(im_cut_dim[2] * dataset)):adjust_coef.shape[2]],
                    dtype=np.float32)
                sensivity_cut[dataset] = np.ascontiguousarray(
                    sensivity_matrix[int(np.floor(im_cut_dim[0] * dataset)):sensivity_matrix.shape[0],
                    int(np.floor(im_cut_dim[1] * dataset)):sensivity_matrix.shape[1],
                    int(np.floor(im_cut_dim[2] * dataset)):sensivity_matrix.shape[2]],
                    dtype=np.float32)
            else:
                begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)

                adjust_coef_cut[dataset] = np.ascontiguousarray(adjust_coef[:, :,
                                                                int(np.floor(im_cut_dim[2] * dataset)):int(
                                                                    np.floor(im_cut_dim[2] * (dataset + 1)))],
                                                                dtype=np.float32)
                sensivity_cut[dataset] = np.ascontiguousarray(sensivity_matrix[:, :,
                                                              int(np.floor(im_cut_dim[2] * dataset)):int(
                                                                  np.floor(im_cut_dim[2] * (dataset + 1)))],
                                                              dtype=np.byte)

                A_cut[dataset] = np.ascontiguousarray(A[:, :,
                                                      int(np.floor(im_cut_dim[2] * dataset)):int(
                                                          np.floor(im_cut_dim[2] * (dataset + 1)))], dtype=np.int32)
                B_cut[dataset] = np.ascontiguousarray(B[:, :,
                                                      int(np.floor(im_cut_dim[2] * dataset)):int(
                                                          np.floor(im_cut_dim[2] * (dataset + 1)))], dtype=np.int32)
                C_cut[dataset] = np.ascontiguousarray(C[:, :,
                                                      int(np.floor(im_cut_dim[2] * dataset)):int(
                                                          np.floor(im_cut_dim[2] * (dataset + 1)))], dtype=np.int32)

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
            probability_cut[dataset] = probability[begin_dataset:end_dataset]

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

            sum_vor_gpu[dataset] = cuda.mem_alloc(sum_vor_cut[dataset].size * sum_vor_cut[dataset].dtype.itemsize)
            probability_gpu[dataset] = cuda.mem_alloc(
                probability_cut[dataset].size * probability_cut[dataset].dtype.itemsize)

            cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_cut[dataset])
            cuda.memcpy_htod_async(probability_gpu[dataset], probability_cut[dataset])
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

            # Backprojection
            adjust_coef_gpu[dataset] = cuda.mem_alloc(
                adjust_coef_cut[dataset].size * adjust_coef_cut[dataset].dtype.itemsize)

            sensivity_gpu[dataset] = cuda.mem_alloc(
                sensivity_cut[dataset].size * sensivity_cut[dataset].dtype.itemsize)

            A_cut_gpu[dataset] = cuda.mem_alloc(
                A_cut[dataset].size * A_cut[dataset].dtype.itemsize)

            B_cut_gpu[dataset] = cuda.mem_alloc(
                B_cut[dataset].size * B_cut[dataset].dtype.itemsize)

            C_cut_gpu[dataset] = cuda.mem_alloc(
                C_cut[dataset].size * C_cut[dataset].dtype.itemsize)

            cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset])
            cuda.memcpy_htod_async(sensivity_gpu[dataset], sensivity_cut[dataset])
            cuda.memcpy_htod_async(A_cut_gpu[dataset], A_cut[dataset])
            cuda.memcpy_htod_async(B_cut_gpu[dataset], B_cut[dataset])
            cuda.memcpy_htod_async(C_cut_gpu[dataset], C_cut[dataset])

        free, total = cuda.mem_get_info()

        print('%.1f %% of device memory is free.' % ((free / float(total)) * 100))

        # -------------OSEM---------
        it = self.number_of_iterations
        subsets = self.number_of_subsets
        print('Number events for reconstruction: {}'.format(number_of_events))
        # total_pixels_per_event = weight * height * depth
        # number_of_pixel_calculated_per_second = 3E9
        # time_to_prevent_watchdog = 1.8  # seconds
        # number_events_cutted_by_watchdog = np.int32(
        #     (number_of_pixel_calculated_per_second * time_to_prevent_watchdog) / total_pixels_per_event)
        # propability calculation
        # begin_event = np.int32(0)
        # end_event = np.int32(number_of_events / subsets)
        # number_of_events_subset = np.int32(end_event - begin_event)
        # for dataset in range(number_of_datasets):
        #
        #     if dataset == number_of_datasets:
        #         begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
        #         end_dataset = number_of_events_subset
        #     else:
        #         begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
        #         end_dataset = np.int32((dataset + 1) * number_of_events_subset / number_of_datasets)
        #
        #     threadsperblock = (512, 1, 1)
        #     blockspergrid_x = int(math.ceil(((end_dataset - begin_dataset)) / threadsperblock[0]))
        #     blockspergrid_y = int(math.ceil(1 / threadsperblock[1]))
        #     blockspergrid_z = int(math.ceil(1 / threadsperblock[2]))
        #     blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        #     event[dataset]['kernel_begin'].record(stream[dataset])
        #
        #     func_prob = mod_probability_shared_mem.get_function("probability")
        #     func_prob(weight, height, depth, half_crystal_pitch_xy, half_crystal_pitch_z,
        #                  number_of_events, begin_dataset, end_dataset, a_cut_gpu[dataset], a_cut_normal_gpu[dataset],
        #                  b_cut_gpu[dataset], b_cut_normal_gpu[dataset], c_cut_gpu[dataset], c_cut_normal_gpu[dataset],
        #                  d_cut_gpu[dataset],
        #                  d_cut_normal_gpu[dataset], A_gpu, B_gpu, C_gpu, probability_gpu[dataset],
        #                  im_gpu,
        #                  block=threadsperblock,
        #                  grid=blockspergrid,
        #                  stream=stream[dataset])
        #
        # # Sincronization of streams
        # for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
        #     event[dataset]['kernel_end'].record(stream[dataset])
        #
        # # Transfering data from Optimizer
        # for dataset in range(number_of_datasets):
        #     begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
        #     end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
        #     cuda.memcpy_dtoh_async(probability[begin_dataset:end_dataset], probability_gpu[dataset])

        # cuda.memcpy_htod_async(probability_t_gpu, probability)

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

                    func_forward = mod_forward_projection_shared_mem.get_function("forward_projection")
                    func_forward(weight, height, depth, half_crystal_pitch_xy, half_crystal_pitch_z,
                                 distance_between_array_pixel,
                                 number_of_events, begin_dataset, end_dataset, a_cut_gpu[dataset],
                                 a_cut_normal_gpu[dataset], a_cf_cut_gpu[dataset],
                                 b_cut_gpu[dataset], b_cut_normal_gpu[dataset], b_cf_cut_gpu[dataset],
                                 c_cut_gpu[dataset], c_cut_normal_gpu[dataset],
                                 c_cf_cut_gpu[dataset],
                                 d_cut_gpu[dataset],
                                 d_cut_normal_gpu[dataset], d_cf_cut_gpu[dataset], A_gpu, B_gpu, C_gpu,
                                 sum_vor_gpu[dataset], sensivity_matrix_gpu, im_gpu, probability_gpu[dataset],
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
                    cuda.memcpy_dtoh_async(sum_vor[begin_dataset:end_dataset], sum_vor_gpu[dataset])
                    toc = time.time()
                    print('Time part Forward Projection {} : {}'.format(1, toc - tic))

                # number_of_datasets = np.int32(2)
                teste = np.copy(sum_vor)
                # sum_vor=np.ascontiguousarray(np.ones((self.a.shape)), dtype=np.float32)
                print('SUM VOR: {}'.format(np.sum(teste)))
                cuda.memcpy_htod_async(sum_vor_t_gpu, sum_vor)

                # ------------BACKPROJECTION-----------

                for dataset in range(number_of_datasets):
                    dataset = np.int32(dataset)
                    if dataset == number_of_datasets:
                        begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
                        end_dataset = number_of_events_subset
                    else:
                        begin_dataset = np.int32(dataset * number_of_events_subset / number_of_datasets)
                        end_dataset = np.int32((dataset + 1) * number_of_events_subset / number_of_datasets)

                    # begin_dataset = np.int32(0)
                    # end_dataset = np.int32(number_of_events)

                    event[dataset]['kernel_begin'].record(stream[dataset])
                    weight_cutted, height_cutted, depth_cutted = np.int32(adjust_coef_cut[dataset].shape[0]), np.int32(
                        adjust_coef_cut[dataset].shape[1]), np.int32(adjust_coef_cut[dataset].shape[2])

                    threadsperblock = (np.int(A.shape[0]), 1, 1)
                    blockspergrid_x = int(math.ceil(adjust_coef_cut[dataset].shape[0] / threadsperblock[0]))
                    blockspergrid_y = int(math.ceil(adjust_coef_cut[dataset].shape[1] / threadsperblock[1]))
                    blockspergrid_z = int(math.ceil(adjust_coef_cut[dataset].shape[2] / threadsperblock[2]))
                    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
                    shared_memory = threadsperblock[0] * threadsperblock[1] * threadsperblock[2] * 4
                    func_backward = mod_backward_projection_shared_mem.get_function("backprojection")

                    # func_backward.prepare(dataset, weight_cutted, height_cutted, depth_cutted, half_crystal_pitch_xy, half_crystal_pitch_z, distance_between_array_pixel,
                    #      number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
                    #      b_gpu, b_normal_gpu,b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
                    #      d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset], C_cut_gpu[dataset], adjust_coef_gpu[dataset],
                    #      sum_vor_t_gpu, sensivity_gpu[dataset], probability_t_gpu, time_factor_gpu,shared=512)
                    #
                    # func.prepared_call(blockspergrid, threadsperblock, dataset, weight_cutted, height_cutted, depth_cutted, half_crystal_pitch_xy, half_crystal_pitch_z, distance_between_array_pixel,
                    #      number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
                    #      b_gpu, b_normal_gpu,b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
                    #      d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset], C_cut_gpu[dataset], adjust_coef_gpu[dataset],
                    #      sum_vor_t_gpu, sensivity_gpu[dataset], probability_t_gpu, time_factor_gpu)
                    func_backward(dataset, weight_cutted, height_cutted, depth_cutted, half_crystal_pitch_xy,
                                  half_crystal_pitch_z, distance_between_array_pixel,
                                  number_of_events, begin_dataset, end_dataset, a_gpu, a_normal_gpu, a_cf_gpu,
                                  b_gpu, b_normal_gpu, b_cf_gpu, c_gpu, c_normal_gpu, c_cf_gpu, d_gpu,
                                  d_normal_gpu, d_cf_gpu, A_cut_gpu[dataset], B_cut_gpu[dataset], C_cut_gpu[dataset],
                                  adjust_coef_gpu[dataset],
                                  sum_vor_t_gpu, sensivity_gpu[dataset], probability_t_gpu, time_factor_gpu,
                                  block=threadsperblock,
                                  grid=blockspergrid,
                                  shared=np.int(4 * A.shape[0]),
                                  stream=stream[dataset],
                                  )

                # shared=np.int(shared_memory),

                for dataset in range(number_of_datasets):  # Commenting out this line should break concurrency.
                    event[dataset]['kernel_end'].record(stream[dataset])

                for dataset in range(number_of_datasets):
                    cuda.memcpy_dtoh_async(adjust_coef_cut[dataset], adjust_coef_gpu[dataset])

                    adjust_coef[:, :,
                    int(np.floor(im_cut_dim[2] * dataset)):int(np.floor(im_cut_dim[2] * (dataset + 1)))] = \
                        adjust_coef_cut[dataset]

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
                    im_med = median_filter(im, kernel_filter_size)
                    penalized_term = np.copy(im)
                    penalized_term[im_med != 0] = 1 + beta * (im[im_med != 0] - im_med[im_med != 0]) / im_med[
                        im_med != 0]
                    penalized_term = np.ascontiguousarray(penalized_term, dtype=np.float32)

                if self.algorithm == "MAP":
                    beta = 0.5
                    im_map = np.load(
                        "C:\\Users\\pedro.encarnacao\\OneDrive - Universidade de Aveiro\\PhD\\Reconstrução\\NAF+FDG\\Easypet Scan 05 Aug 2019 - 14h 36m 33s\\static_image\\Easypet Scan 05 Aug 2019 - 14h 36m 33s mlem.npy")

                im[normalization_matrix != 0] = im[normalization_matrix != 0] * adjust_coef[
                    normalization_matrix != 0] / ((normalization_matrix[normalization_matrix != 0]) /
                                                  self.attenuation_matrix[normalization_matrix != 0])
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
                    np.zeros((self.number_of_pixels_x, self.number_of_pixels_y, self.number_of_pixels_z),
                             dtype=np.float32))

                for dataset in range(number_of_datasets):
                    if dataset == number_of_datasets:
                        begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                        end_dataset = number_of_events

                    else:
                        begin_dataset = np.int32(dataset * number_of_events / number_of_datasets)
                        end_dataset = np.int32((dataset + 1) * number_of_events / number_of_datasets)
                        # adjust_coef_cut[dataset] = np.ascontiguousarray(adjust_coef[:, :,
                        #                                                 int(np.floor(im_cut_dim[2] * dataset)):int(
                        #                                                     np.floor(im_cut_dim[2] * (dataset + 1)))],
                        #                                                 dtype=np.float32)
                        adjust_coef_cut[dataset] = adjust_coef

                    sum_vor_cut[dataset] = sum_vor[begin_dataset:end_dataset]
                    cuda.memcpy_htod_async(sum_vor_gpu[dataset], sum_vor_cut[dataset])
                    cuda.memcpy_htod_async(adjust_coef_gpu[dataset], adjust_coef_cut[dataset])

                if self.saved_image_by_iteration:
                    self._save_image_by_it(im, i, sb)

                if self.signals_interface is not None:
                    self.signals_interface.trigger_update_label_reconstruction_status.emit(
                        "{}: Iteration {}".format(self.current_info_step, i + 1))
                    self.signals_interface.trigger_progress_reconstruction_partial.emit(
                        int(np.round(100 * (i + 1) * (sb + subsets) / (it * subsets), 0)))
        return im * subsets