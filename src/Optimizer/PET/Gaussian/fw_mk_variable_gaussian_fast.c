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
(const float* sigma_y_square, const float* sigma_z_square, const float* gaussian_y_fix_term, const float* gaussian_z_fix_term,
const float* acceptable_z_distance, const float* acceptable_y_distance, const float* invert2timesigma_y_square,
const float* invert2timesigma_z_square,
 const int n, const int m, const int p, const int start_x, const int start_y, const int start_z,
    float* crystal_pitch_XY, float* crystal_pitch_Z, const float distance_between_array_pixel,
    int number_of_events, const int begin_event_gpu_limitation, const int end_event_gpu_limitation,
    float* a, float* a_normal, float* a_cf, float* b, float* b_normal, float* b_cf, float* c, float* c_normal,
    float* c_cf, float* d, float* d_normal, float* d_cf, float* sum_vor, const char* fov_cut_matrix, const float* im_old)
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

    __shared__ float gaussian_y_fix_term_shared[shared_memory_size];
    __shared__ float gaussian_z_fix_term_shared[shared_memory_size];
    __shared__ float acceptable_z_distance_shared[shared_memory_size];
    __shared__ float acceptable_y_distance_shared[shared_memory_size];
    __shared__ float invert2timesigma_y_square_shared[shared_memory_size];
    __shared__ float invert2timesigma_z_square_shared[shared_memory_size];



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
    float z_distance;
    float y_distance;
    float x_distance;
    float gaussian_x;
    float gaussian_y;
    float gaussian_z;
    bool shift_variant = false;

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

    gaussian_y_fix_term_shared[e_m] = gaussian_y_fix_term[e];
    gaussian_z_fix_term_shared[e_m] = gaussian_z_fix_term[e];
    acceptable_z_distance_shared[e_m] = acceptable_z_distance[e];
    acceptable_y_distance_shared[e_m] = acceptable_y_distance[e];
    invert2timesigma_y_square_shared[e_m] = invert2timesigma_y_square[e];
    invert2timesigma_z_square_shared[e_m] = invert2timesigma_z_square[e];
    __syncthreads();
//    crystal_pitch_Z_shared[e_m] = crystal_pitch_Z[e];
//    crystal_pitch_XY_shared[e_m] = crystal_pitch_XY[e];
//    d2_normal = crystal_pitch_Z_shared[e_m] * sqrt(a_normal_shared[e_m] * a_normal_shared[e_m] + b_normal_shared[e_m] * b_normal_shared[e_m] + c_normal_shared[e_m] * c_normal_shared[e_m]);
//    d2 = crystal_pitch_XY_shared[e_m] * sqrt(a_shared[e_m] * a_shared[e_m] + b_shared[e_m] * b_shared[e_m] + c_shared[e_m] * c_shared[e_m]);
//    d2_cf = distance_between_array_pixel * sqrt(a_cf_shared[e_m] * a_cf_shared[e_m] + b_cf_shared[e_m] * b_cf_shared[e_m] + c_cf_shared[e_m] * c_cf_shared[e_m]);

//    const float voxel_size = 0.5;
////    const float sigma_x_square = 40/(voxel_size*voxel_size);
//    const float sigma_y_square = 0.15/(voxel_size*voxel_size);
//    const float sigma_z_square = 0.16/(voxel_size*voxel_size);
//
//
////    const float gaussian_x_fix_term = 1 / rsqrtf(6.28 * sigma_x_square);
//    const float gaussian_y_fix_term = 1 / rsqrtf(6.28 * sigma_y_square);
//    const float gaussian_z_fix_term = 1 / rsqrtf(6.28 * sigma_z_square);
//    const float acceptable_z_distance = 11.09205*sigma_z_square;
//    const float acceptable_y_distance = 11.09205*sigma_y_square;
////    const float invert2timesigma_x_square = 1 / (2 * sigma_x_square);
//    const float invert2timesigma_y_square = 1 / (2 * sigma_y_square);
//    const float invert2timesigma_z_square = 1 / (2 * sigma_z_square);

     for (int l = 0; l < n; l++)

    {
        x_t = l + start_x;

        for (int j = 0; j < m; j++)
        {
            y_t = j + start_y;


            for (int k = 0; k < p; k++)
            {
                index = k + j * p + l * p * m;
//                index = k + l * p + j * p * m;
//                index = l+j*n+k*m*n;
                z_t = k + start_z;
                if (fov_cut_matrix[index] != 0)
                {//
                    z_distance = a_shared[e_m] * x_t + b_shared[e_m] * y_t + c_shared[e_m] * z_t - d_shared[e_m];
//                    z_distance = fabsf(z_distance);
                    if (fabsf(z_distance) <= acceptable_z_distance[e_m])
                    {
                        y_distance = a_normal_shared[e_m] * x_t + b_normal_shared[e_m] * y_t + c_normal_shared[e_m] * z_t - d_normal_shared[e_m];
//                        y_distance = fabsf(y_distance);
//
                        if (fabsf(y_distance) <= acceptable_y_distance[e_m])
                         {
//
//                            x_distance = a_cf_shared[e_m] * x_t + b_cf_shared[e_m] * y_t + c_cf_shared[e_m] * z_t - d_cf_shared[e_m];

//                            gaussian_x = gaussian_x_fix_term * __expf(-x_distance * x_distance * invert2timesigma_x_square);
//                            gaussian_y = gaussian_y_fix_term * __expf(-y_distance * y_distance * invert2timesigma_y_square);
//                            gaussian_z = gaussian_z_fix_term * __expf(-z_distance * z_distance * invert2timesigma_z_square);
                            gaussian_y = gaussian_y_fix_term_shared[e_m] * __expf(-y_distance * y_distance * invert2timesigma_y_square_shared[e_m]);
                            gaussian_z = gaussian_z_fix_term_shared[e_m] * __expf(-z_distance * z_distance * invert2timesigma_z_square_shared[e_m]);

                            sum_vor_shared[e_m] += im_old[index] * gaussian_y*gaussian_z;
//


                        }
                    }
                }
            }
        }
    }
    __syncthreads();
    sum_vor[e] = sum_vor_shared[e_m];
}