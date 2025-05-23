#include <stdint.h>

//texture<char, 1> tex;

/*
Machine code for
STEP: FORWARD PROJECTION
CONFIGURATION: Multiple Kernel
PROJECTOR: Solid Angle Small approximation
DOI CORRECTION: FALSE
DECAY CORRECTION: FALSE
*/

__global__ void forward_projection_cdrf
(const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    float* crystal_pitch_XY, float* crystal_pitch_Z, const float distance_between_array_pixel,
    int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation,
    float* a, float* a_normal, float* a_cf, float* b, float* b_normal, float* b_cf, float* c, float* c_normal,
    float* c_cf, float* d, float* d_normal, float* d_cf, float* sum_vor, const char* fov_cut_matrix, const float* im_old,
    short* x_min, short* x_max, short* y_min, short* y_max, short* z_min, short* z_max)
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
    __shared__ short x_min_shared[shared_memory_size];
    __shared__ short y_min_shared[shared_memory_size];
    __shared__ short z_min_shared[shared_memory_size];
    __shared__ short x_max_shared[shared_memory_size];
    __shared__ short y_max_shared[shared_memory_size];
    __shared__ short z_max_shared[shared_memory_size];

    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
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
    float solid_angle;

    const int number_events_max = end_event_gpu_limitation - begin_event_gpu_limitation;
    const float error_pixel = 0.0000f;
    if (threadIdx.x > shared_memory_size)
    {
        return;
    }
    if (threadId > number_events_max)
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
    crystal_pitch_Z_shared[e_m] = crystal_pitch_Z[e];
    crystal_pitch_XY_shared[e_m] = crystal_pitch_XY[e];
    x_min_shared[e_m] = x_min[e];
    x_max_shared[e_m] = x_max[e];
    y_min_shared[e_m] = y_min[e];
    y_max_shared[e_m] = y_max[e];
    z_min_shared[e_m] = z_min[e];
    z_max_shared[e_m] = z_max[e];

    d2_normal = crystal_pitch_Z_shared[e_m] * sqrt(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);
    d2 = crystal_pitch_XY_shared[e_m] * sqrt(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
    d2_cf = distance_between_array_pixel * sqrt(a_cf_shared[e_m] * a_cf_shared[e_m] + b_cf_shared[e_m] * b_cf_shared[e_m] + c_cf_shared[e_m] * c_cf_shared[e_m]);

//
    for (int l = x_min_shared[e_m];  l < x_max_shared[e_m]; l++)
    {
        x_t = l;

        for (int j = y_min_shared[e_m]; j < y_max_shared[e_m]; j++)
        {
            y_t = j;
            for (int k = z_min_shared[e_m]; k < z_max_shared[e_m]; k++)
            {
                z_t = k;
                index = k-start_z + (j-start_y) * p + (l-start_x) * p * m;
                // cilindric cut
                value_normal = a_normal_shared[e_m] * x_t + b_normal_shared[e_m] * y_t + c_normal_shared[e_m] * z_t - d_normal_shared[e_m];
                value = a_shared[e_m] * x_t + b_shared[e_m] * y_t + c_shared[e_m] * z_t - d_shared[e_m];
                if (value*value+value_normal*value_normal<2)
                {
                   sum_vor_shared[e_m] += im_old[index];
                   }
//                    value_normal = fabsf(a_normal_shared[e_m] * x_t + b_normal_shared[e_m] * y_t + c_normal_shared[e_m] * z_t - d_normal_shared[e_m]);
//
//
//                    if (value_normal <= d2_normal)
//                    {
//                        value = fabsf(a_shared[e_m] * x_t + b_shared[e_m] * y_t + c_shared[e_m] * z_t - d_shared[e_m]);
//
//
//                        if (value <= d2 )
//                        {
//                            value_cf = a_cf_shared[e_m] * x_t + b_cf_shared[e_m] * y_t + c_cf_shared[e_m] * z_t - d_cf_shared[e_m];
//
//                            if (value_cf >= -d2_cf && value_cf < d2_cf)
////
//                            {

//                                    width = 2 * (crystal_pitch_XY_shared[e_m] - abs(value));
//                                    height = 2 * (crystal_pitch_Z_shared[e_m] - abs(value_normal));
//
//                                    distance = d2_cf + abs(value_cf);
////                                    distance_other = abs(d2_cf + value_cf);
//                                    distance_other = d2_cf - abs(value_cf);
//                                    solid_angle = 4 * (width * width * height * height / (distance * distance * (4 * distance * distance + width * width + height * height)));

//                                    sum_vor_shared[e_m] += 1;
//                                    sum_vor_shared[e_m] += im_old[index] * solid_angle;


//                            }
//                        }
//                    }
//                }
            }
        }

    }
    __syncthreads();
    sum_vor[e] = sum_vor_shared[e_m];
}
//__global__ void forward_projection_cdrf
//(const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
//    float* crystal_pitch_XY, float* crystal_pitch_Z, const float distance_between_array_pixel,
//    int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation,
//    float* a, float* a_normal, float* a_cf, float* b, float* b_normal, float* b_cf, float* c, float* c_normal,
//    float* c_cf, float* d, float* d_normal, float* d_cf, float* sum_vor, const char* fov_cut_matrix, const float* im_old,
//    short* x_min,  short* x_max, short* y_min, short* y_max,  short* z_min,
//    short* z_max)
//{
//    const int shared_memory_size = 256;
//    __shared__ float a_shared[shared_memory_size];
//    __shared__ float b_shared[shared_memory_size];
//    __shared__ float c_shared[shared_memory_size];
//    __shared__ float d_shared[shared_memory_size];
//    __shared__ float a_normal_shared[shared_memory_size];
//    __shared__ float b_normal_shared[shared_memory_size];
//    __shared__ float c_normal_shared[shared_memory_size];
//    __shared__ float d_normal_shared[shared_memory_size];
//    __shared__ float a_cf_shared[shared_memory_size];
//    __shared__ float b_cf_shared[shared_memory_size];
//    __shared__ float c_cf_shared[shared_memory_size];
//    __shared__ float d_cf_shared[shared_memory_size];
//    __shared__ float sum_vor_shared[shared_memory_size];
//    __shared__ char fov_cut_matrix_shared[shared_memory_size];
//    __shared__ float crystal_pitch_Z_shared[shared_memory_size];
//    __shared__ float crystal_pitch_XY_shared[shared_memory_size];
//    __shared__ short x_min_shared[shared_memory_size];
//    __shared__ short y_min_shared[shared_memory_size];
//    __shared__ short z_min_shared[shared_memory_size];
//    __shared__ short x_max_shared[shared_memory_size];
//    __shared__ short y_max_shared[shared_memory_size];
//    __shared__ short z_max_shared[shared_memory_size];
//    /*
//   const int blockId = blockIdx.x+ blockIdx.y * gridDim.x+ gridDim.x * gridDim.y * blockIdx.z;
//   const int threadId= blockId * (blockDim.x * blockDim.y * blockDim.z)+ (threadIdx.z * (blockDim.x * blockDim.y))+ (threadIdx.y * blockDim.x)+ threadIdx.x;
//   */
//    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
//    int e;
//    float d2;
//    float d2_normal;
//    float d2_cf;
//    float value;
//    float value_normal;
//    float value_cf;
//    float sum_vor_temp;
//    float width;
//    float height;
//    float distance;
//    int index;
//    short x_t;
//    short y_t;
//    short z_t;
//
//    const int number_events_max = end_event_gpu_limitation - begin_event_gpu_limitation;
//    if (threadIdx.x > shared_memory_size)
//    {
//        return;
//    }
//    if (threadId > number_events_max)
//    {
//        return;
//    }
//
//    __syncthreads();
//    e = threadId;
//    int e_m = threadIdx.x;
//    a_shared[e_m] = a[e];
//    b_shared[e_m] = b[e];
//    c_shared[e_m] = c[e];
//    d_shared[e_m] = d[e];
//    a_normal_shared[e_m] = a_normal[e];
//    b_normal_shared[e_m] = b_normal[e];
//    c_normal_shared[e_m] = c_normal[e];
//    d_normal_shared[e_m] = d_normal[e];
//    a_cf_shared[e_m] = a_cf[e];
//    b_cf_shared[e_m] = b_cf[e];
//    c_cf_shared[e_m] = c_cf[e];
//    d_cf_shared[e_m] = d_cf[e];
//    sum_vor_shared[e_m] = sum_vor[e];
//    x_min_shared[e_m] = x_min[e];
//    x_max_shared[e_m] = x_max[e];
//    y_min_shared[e_m] = y_min[e];
//    y_max_shared[e_m] = y_max[e];
//    z_min_shared[e_m] = z_min[e];
//    z_max_shared[e_m] = z_max[e];
//    crystal_pitch_Z_shared[e_m] = crystal_pitch_Z[e];
//    crystal_pitch_XY_shared[e_m] = crystal_pitch_XY[e];
//
//
//
//    d2_normal = crystal_pitch_Z_shared[e_m] * sqrt(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);
//    d2 = crystal_pitch_XY_shared[e_m] * sqrt(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
//    d2_cf = distance_between_array_pixel * sqrt(a_cf_shared[e_m] * a_cf_shared[e_m] + b_cf_shared[e_m] * b_cf_shared[e_m] + c_cf_shared[e_m] * c_cf_shared[e_m]);
////
//    for (short l = x_min_shared[e_m];  l < x_max_shared[e_m]; l++)
//    {
//        x_t = l;
//
//        for (short j = y_min_shared[e_m]; j < y_max_shared[e_m]; j++)
//        {
//            y_t = j;
//            for (short k = z_min_shared[e_m]; k < z_max_shared[e_m]; k++)
//            {
//                z_t = k;
////    for (int l = 0; l < n; l++)
////    {
////
////        x_t = l + start_x;
////        for (int j = 0; j < m; j++)
////        {
////            y_t = j + start_y;
////
////            for (int k = 0; k < p; k++)
////
////            {
////                z_t = k + start_z;
//                /*
//                index = l+j*n+k*m*n;
//                fov_cut_matrix_shared[k]=fov_cut_matrix[k+j*p+l*p*m];
//                index = l+j*n+k*m*n;
//                if (fov_cut_matrix[index]!=0)
//                {
//                */
//
//                value_normal = fabsf(a_normal_shared[e_m] * x_t + b_normal_shared[e_m] * y_t + c_normal_shared[e_m] * z_t - d_normal_shared[e_m]);
//
//                if (value_normal < d2_normal)
//                {
//                    value = fabsf(a_shared[e_m] * x_t + b_shared[e_m] * y_t + c_shared[e_m] * z_t - d_shared[e_m]);
//
//                    if (value < d2)
//                    {
//                        value_cf = fabsf(a_cf_shared[e_m] * x_t + b_cf_shared[e_m] * y_t + c_cf_shared[e_m] * z_t - d_cf_shared[e_m]);
//
//
//                        if (value_cf < d2_cf)
//                        {
//
//                            index = k + j * p + l * p * m;
//                            /*
//                            if (fov_cut_matrix[index] != 0)
//                            {
//                                 width = 2*(2.0f  - value);
//                                 height = 2*(3.0f - value_normal);
//                                 distance =d2_cf+value_cf;
//                                 sum_vor_temp += im_old[index] * 4*(width*width*height*height/(distance*distance*(4*distance*distance+width*width+height*height)));
//                                } */
//                            sum_vor_temp += im_old[index];
//
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    sum_vor[e] = sum_vor_temp;
//}